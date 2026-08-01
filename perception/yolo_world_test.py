"""Single-frame YOLO-World inference for RoboTwin's head camera."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VISUAL_CLASSES = (
    {
        "prompt": "gray hollow block",
        "semantic_label": "part_A",
    },
    {
        "prompt": "small speaker",
        "semantic_label": "part_B",
    },
    {
        "prompt": "blue plastic storage bin",
        "semantic_label": "box",
    },
)

EXPECTED_COUNTS = {
    "part_A": 3,
    "part_B": 2,
    "box": 1,
}

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_path: str):
    """Load each detector once so closed-loop observations stay lightweight."""
    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise RuntimeError(
            "缺少 YOLO-World 运行依赖。请在 RoboTwin 环境中安装 "
            "ultralytics 和 Pillow。"
        ) from exc

    path = Path(model_path).expanduser()
    cache_key = str(path.resolve()) if path.exists() else model_path
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = YOLOWorld(model_path)
    return _MODEL_CACHE[cache_key]


def _result_class_name(result: Any, class_id: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _semantic_label(class_id: int, prompt: str) -> str:
    if prompt in EXPECTED_COUNTS:
        return prompt
    if 0 <= class_id < len(VISUAL_CLASSES):
        return VISUAL_CLASSES[class_id]["semantic_label"]
    for visual_class in VISUAL_CLASSES:
        if visual_class["prompt"] == prompt:
            return visual_class["semantic_label"]
    return "unknown"


def _draw_detections(rgb: np.ndarray, detections: list[dict[str, Any]]):
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {
        "part_A": (255, 80, 80),
        "part_B": (255, 190, 40),
        "box": (40, 180, 255),
        "unknown": (180, 180, 180),
    }

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        color = colors.get(detection["semantic_label"], colors["unknown"])
        label = (
            f'{detection["semantic_label"]} '
            f'{detection["confidence"]:.2f}'
        )
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        try:
            text_box = draw.textbbox((x1, y1), label)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label)
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            (x1, text_y, x1 + text_width + 6, text_y + text_height + 6),
            fill=color,
        )
        draw.text((x1 + 3, text_y + 3), label, fill=(0, 0, 0))

    return image


def run_yolo_world_single_frame(
    rgb: np.ndarray,
    output_dir: Path,
    *,
    model_path: str = "yolov8s-worldv2.pt",
    confidence: float = 0.60,
    image_size: int = 640,
    device: str = "0",
    camera_name: str = "head_camera",
) -> dict[str, Any]:
    """Run open-vocabulary detection and save raw, annotated and JSON outputs."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Pillow，无法保存 YOLO-World 检测图像。"
        ) from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"{camera_name} RGB 应为 HxWx3 数组，实际形状为 {rgb.shape}"
        )
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{camera_name}_rgb.png"
    annotated_path = output_dir / f"{camera_name}_yolo_world.png"
    json_path = output_dir / f"{camera_name}_yolo_world.json"
    Image.fromarray(rgb).save(raw_path)

    model = _load_model(model_path)
    loaded_names = {
        str(name)
        for name in (
            model.names.values()
            if isinstance(model.names, dict)
            else model.names
        )
    }
    trained_classes = set(EXPECTED_COUNTS)
    if loaded_names == trained_classes:
        inference_mode = "trained_fixed_classes"
        visual_classes = [
            {
                "prompt": label,
                "semantic_label": label,
            }
            for label in EXPECTED_COUNTS
        ]
    else:
        inference_mode = "zero_shot_text_prompts"
        visual_classes = list(VISUAL_CLASSES)
        prompts = [item["prompt"] for item in VISUAL_CLASSES]
        model.set_classes(prompts)

    # Ultralytics treats a NumPy source as BGR and converts it internally.
    # RoboTwin returns RGB, so adapt channel order at this API boundary.
    bgr_for_ultralytics = np.ascontiguousarray(rgb[:, :, ::-1])
    results = model.predict(
        source=bgr_for_ultralytics,
        conf=confidence,
        imgsz=image_size,
        device=device,
        agnostic_nms=True,
        verbose=False,
    )
    result = results[0]

    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None:
        xyxy_values = boxes.xyxy.detach().cpu().numpy()
        confidence_values = boxes.conf.detach().cpu().numpy()
        class_values = boxes.cls.detach().cpu().numpy().astype(int)
        for xyxy, score, class_id in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            prompt = _result_class_name(result, int(class_id))
            detections.append(
                {
                    "class_id": int(class_id),
                    "prompt": prompt,
                    "semantic_label": _semantic_label(int(class_id), prompt),
                    "confidence": float(score),
                    "bbox_xyxy": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    "bbox_xywh": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                }
            )

    found_counts = Counter(
        detection["semantic_label"]
        for detection in detections
        if detection["semantic_label"] != "unknown"
    )
    count_check = {
        label: {
            "expected": expected,
            "detected": found_counts.get(label, 0),
            "matches_expected": found_counts.get(label, 0) == expected,
        }
        for label, expected in EXPECTED_COUNTS.items()
    }
    payload = {
        "camera": camera_name,
        "image": {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "color_space": "RGB",
            "raw_path": str(raw_path.resolve()),
            "annotated_path": str(annotated_path.resolve()),
        },
        "model": {
            "weights": model_path,
            "confidence_threshold": confidence,
            "image_size": image_size,
            "device": device,
            "inference_mode": inference_mode,
            "visual_classes": visual_classes,
        },
        "detections": detections,
        "count_check": count_check,
        "all_expected_counts_matched": all(
            item["matches_expected"] for item in count_check.values()
        ),
        "note": (
            "数量检查仅用于零样本识别测试，不代表机器人任务成功，"
            "也未使用深度或仿真器物体真实位姿。"
        ),
    }

    _draw_detections(rgb, detections).save(annotated_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["json_path"] = str(json_path.resolve())
    return payload
