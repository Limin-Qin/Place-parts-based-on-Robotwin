"""Fine-tune YOLO-World for part_A, part_B and box without touching source weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = EXAMPLE_DIR / "vision_dataset" / "data.yaml"
DEFAULT_WEIGHTS = EXAMPLE_DIR / "yolov8s-worldv2.pt"
DEFAULT_PROJECT = EXAMPLE_DIR / "yolo_training_runs"
DEFAULT_ARCHIVE = EXAMPLE_DIR / "trained_weights"
EXPECTED_NAMES = {
    0: "part_A",
    1: "part_B",
    2: "box",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("data.yaml 的 names 必须是列表或字典")


def _resolve_split_images(
    dataset_root: Path,
    split_value: str,
) -> list[Path]:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    split_path = split_path.resolve()

    if split_path.is_file():
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            image_path = Path(value)
            if not image_path.is_absolute():
                image_path = split_path.parent / image_path
            images.append(image_path.resolve())
        return images
    if split_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(
            path.resolve()
            for path in split_path.rglob("*")
            if path.suffix.lower() in extensions
        )
    raise FileNotFoundError(f"数据划分路径不存在：{split_path}")


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(
            f"图片路径中缺少 images 目录，无法定位标签：{image_path}"
        ) from exc
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_dataset(data_path: Path) -> dict[str, Any]:
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据配置：{data_path}")

    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = _normalize_names(config.get("names"))
    if names != EXPECTED_NAMES:
        raise ValueError(
            f"类别必须为 {EXPECTED_NAMES}，当前为 {names}"
        )

    dataset_root = Path(config.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root
    dataset_root = dataset_root.resolve()

    split_counts: dict[str, int] = {}
    class_counts = {class_id: 0 for class_id in EXPECTED_NAMES}
    for split in ("train", "val"):
        split_value = config.get(split)
        if not isinstance(split_value, str) or not split_value:
            raise ValueError(f"data.yaml 缺少有效的 {split} 路径")
        images = _resolve_split_images(dataset_root, split_value)
        if not images:
            raise ValueError(f"{split} 中没有图片")

        for image_path in images:
            if not image_path.is_file():
                raise FileNotFoundError(f"训练图片不存在：{image_path}")
            label_path = _label_path_for_image(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"图片缺少同名YOLO标签：{image_path} -> {label_path}"
                )
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number} 应包含5列，实际为{len(fields)}列"
                    )
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if class_id not in EXPECTED_NAMES:
                    raise ValueError(
                        f"{label_path}:{line_number} 类别ID无效：{class_id}"
                    )
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError(
                        f"{label_path}:{line_number} 坐标不在[0,1]内"
                    )
                class_counts[class_id] += 1
        split_counts[split] = len(images)

    return {
        "data": str(data_path),
        "dataset_root": str(dataset_root),
        "split_counts": split_counts,
        "class_instances": {
            EXPECTED_NAMES[class_id]: count
            for class_id, count in class_counts.items()
        },
    }


def _build_training_settings(
    options: argparse.Namespace,
    dataset_summary: dict[str, Any],
    source_weights: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose conservative defaults when adapting an existing custom model."""
    train_image_count = int(dataset_summary["split_counts"]["train"])
    custom_checkpoint = source_weights != DEFAULT_WEIGHTS.resolve()
    safe_finetune = (
        bool(getattr(options, "safe_finetune", True))
        and custom_checkpoint
        and train_image_count < 1000
    )

    effective_batch = int(options.batch)
    if safe_finetune and effective_batch == -1:
        # Ultralytics AutoBatch optimizes memory utilization. On a large A800
        # it selected 86 for 240 images, leaving only three updates per epoch.
        # A fixed smaller batch gives stable, frequent gradient updates.
        effective_batch = min(16, train_image_count)

    settings: dict[str, Any] = {
        "data": str(options.data.resolve()),
        "epochs": options.epochs,
        "imgsz": options.imgsz,
        "batch": effective_batch,
        "patience": options.patience,
        "workers": options.workers,
        "device": options.device,
        "project": str(options.project.resolve()),
        "name": options.run_name,
        "exist_ok": False,
        "seed": options.seed,
        "deterministic": True,
        "plots": True,
    }

    if safe_finetune:
        settings.update(
            {
                "optimizer": (
                    options.optimizer
                    if options.optimizer is not None
                    else "AdamW"
                ),
                "lr0": options.lr0 if options.lr0 is not None else 0.0002,
                "lrf": options.lrf if options.lrf is not None else 0.1,
                "warmup_epochs": (
                    options.warmup_epochs
                    if options.warmup_epochs is not None
                    else 1.0
                ),
                "mosaic": (
                    options.mosaic if options.mosaic is not None else 0.0
                ),
                "close_mosaic": 0,
                "cos_lr": True,
            }
        )
    else:
        optional_settings = {
            "optimizer": options.optimizer,
            "lr0": options.lr0,
            "lrf": options.lrf,
            "warmup_epochs": options.warmup_epochs,
            "mosaic": options.mosaic,
        }
        settings.update(
            {
                key: value
                for key, value in optional_settings.items()
                if value is not None
            }
        )

    profile = {
        "safe_finetune": safe_finetune,
        "custom_checkpoint": custom_checkpoint,
        "train_image_count": train_image_count,
        "effective_batch": effective_batch,
        "optimizer": settings.get("optimizer", "auto"),
        "lr0": settings.get("lr0", "ultralytics_default"),
        "lrf": settings.get("lrf", "ultralytics_default"),
        "warmup_epochs": settings.get(
            "warmup_epochs", "ultralytics_default"
        ),
        "mosaic": settings.get("mosaic", "ultralytics_default"),
    }
    return settings, profile


def train(options: argparse.Namespace) -> None:
    data_path = options.data.resolve()
    source_weights = options.weights.resolve()
    project_dir = options.project.resolve()
    archive_dir = options.archive_dir.resolve()

    dataset_summary = validate_dataset(data_path)
    if not source_weights.is_file():
        raise FileNotFoundError(f"找不到预训练权重：{source_weights}")

    source_hash_before = _sha256(source_weights)
    print("训练前检查通过：")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print(f"只读加载原始权重：{source_weights}")
    print(f"原始权重 SHA256：{source_hash_before}")

    training_settings, training_profile = _build_training_settings(
        options,
        dataset_summary,
        source_weights,
    )
    print("实际训练配置：")
    print(json.dumps(training_profile, ensure_ascii=False, indent=2))

    if options.check_only:
        print("训练配置检查通过；--check-only 未启动训练。")
        return

    # Keep caches inside this example instead of writing elsewhere on server.
    cache_dir = EXAMPLE_DIR / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise RuntimeError(
            "当前RoboTwin环境没有可用的ultralytics，请先安装YOLO-World。"
        ) from exc

    project_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_name = options.run_name or datetime.now().strftime(
        "parts_ab_box_%Y%m%d_%H%M%S"
    )
    run_dir = project_dir / run_name
    archived_best = archive_dir / f"{run_name}_best.pt"
    if run_dir.exists():
        raise FileExistsError(
            f"训练目录已存在，为防止覆盖请更换 --run-name：{run_dir}"
        )
    if archived_best.exists():
        raise FileExistsError(
            f"归档权重已存在，为防止覆盖请更换 --run-name：{archived_best}"
        )

    print(f"训练输出目录：{run_dir}")
    print(f"训练完成后最佳权重将归档为：{archived_best}")
    model = YOLOWorld(str(source_weights))
    training_settings["name"] = run_name
    model.train(**training_settings)

    trainer = getattr(model, "trainer", None)
    best_path = Path(getattr(trainer, "best", run_dir / "weights" / "best.pt"))
    best_path = best_path.resolve()
    if not best_path.is_file():
        raise RuntimeError(f"训练结束但没有找到最佳权重：{best_path}")

    source_hash_after = _sha256(source_weights)
    if source_hash_after != source_hash_before:
        raise RuntimeError(
            "原始预训练权重的SHA256发生变化，请停止使用并检查文件。"
        )

    shutil.copy2(best_path, archived_best)
    summary = {
        "source_weights": str(source_weights),
        "source_weights_sha256": source_hash_after,
        "source_weights_unchanged": True,
        "data": str(data_path),
        "run_dir": str(run_dir.resolve()),
        "best_in_run": str(best_path),
        "archived_best": str(archived_best),
        "epochs": options.epochs,
        "imgsz": options.imgsz,
        "batch": training_settings["batch"],
        "device": options.device,
        "seed": options.seed,
        "training_profile": training_profile,
    }
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("YOLO-World训练完成。")
    print(f"最佳权重：{archived_best}")
    print(f"训练指标与图表：{run_dir}")
    print(f"训练摘要：{summary_path}")
    print(f"原始权重保持不变：{source_weights}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help=(
            "-1通常表示自动选择；小数据安全微调时会自动限制为16，"
            "避免每轮更新次数过少"
        ),
    )
    parser.add_argument(
        "--safe-finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "已有自定义checkpoint且训练图少于1000张时，自动使用"
            "小batch、低学习率和无mosaic的稳定微调配置"
        ),
    )
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="0",
        help="run.sh已选好物理GPU，因此默认0表示当前可见GPU",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查数据、标签和权重，不启动训练",
    )
    options = parser.parse_args()
    if options.epochs <= 0:
        parser.error("--epochs 必须大于0")
    if options.imgsz <= 0:
        parser.error("--imgsz 必须大于0")
    if options.batch == 0 or options.batch < -1:
        parser.error("--batch 必须为-1或正整数")
    if options.workers < 0:
        parser.error("--workers 不能小于0")
    if options.lr0 is not None and options.lr0 <= 0:
        parser.error("--lr0 必须大于0")
    if options.lrf is not None and options.lrf <= 0:
        parser.error("--lrf 必须大于0")
    if options.warmup_epochs is not None and options.warmup_epochs < 0:
        parser.error("--warmup-epochs 不能小于0")
    if options.mosaic is not None and not 0.0 <= options.mosaic <= 1.0:
        parser.error("--mosaic 必须在0到1之间")
    return options


if __name__ == "__main__":
    train(parse_args())
