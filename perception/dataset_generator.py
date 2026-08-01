"""Generate an automatically labeled YOLO dataset from RoboTwin rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import trimesh
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]

CLASS_NAMES = {
    0: "part_A",
    1: "part_B",
    2: "box",
}

# Five collision-free tabletop slots. Actors are shuffled between slots for
# every image, with additional translation and yaw jitter.
OBJECT_SLOTS_XY = np.array(
    [
        [-0.28, 0.075],
        [-0.14, 0.155],
        [0.00, 0.075],
        [0.14, 0.155],
        [0.28, 0.075],
    ],
    dtype=float,
)

BOX_INTERIOR_OFFSETS_XY = np.array(
    [
        [-0.075, 0.035],
        [0.075, 0.035],
        [0.000, -0.055],
    ],
    dtype=float,
)


def _load_scaled_visual_vertices(
    model_name: str,
    model_id: int,
    scale: np.ndarray,
) -> np.ndarray:
    mesh_path = (
        REPO_ROOT
        / "assets"
        / "objects"
        / model_name
        / "visual"
        / f"base{model_id}.glb"
    )
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.dump(concatenate=True)
    return np.asarray(mesh.vertices, dtype=float) * np.asarray(scale, dtype=float)


def _sample_orientation_mode(rng: np.random.Generator) -> str:
    # Keep upright views dominant while covering stable side and flipped views.
    return str(
        rng.choice(
            ["upright", "side", "upside_down"],
            p=[0.60, 0.35, 0.05],
        )
    )


def _head_camera(scene: Any):
    cameras = dict(
        zip(
            scene.cameras.static_camera_name,
            scene.cameras.static_camera_list,
        )
    )
    if "head_camera" not in cameras:
        raise RuntimeError("当前机器人配置没有启用 head_camera")
    return cameras["head_camera"]


def _set_actor_pose(
    actor: Any,
    xy: np.ndarray,
    yaw: float,
    orientation_mode: str,
    *,
    drop_clearance: float,
) -> None:
    base_pose = actor._dataset_base_pose
    yaw_quaternion = t3d.quaternions.axangle2quat([0, 0, 1], yaw)
    if orientation_mode == "upright":
        tilt_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    elif orientation_mode == "side":
        axis = [1, 0, 0] if actor._dataset_rng.random() < 0.5 else [0, 1, 0]
        angle = np.pi / 2
        if actor._dataset_rng.random() < 0.5:
            angle *= -1
        tilt_quaternion = t3d.quaternions.axangle2quat(axis, angle)
    elif orientation_mode == "upside_down":
        tilt_quaternion = t3d.quaternions.axangle2quat([1, 0, 0], np.pi)
    else:
        raise ValueError(f"未知零件姿态模式：{orientation_mode}")

    quaternion = t3d.quaternions.qmult(
        yaw_quaternion,
        t3d.quaternions.qmult(tilt_quaternion, base_pose.q),
    )
    rotation = t3d.quaternions.quat2mat(quaternion)
    rotated_vertices = actor._dataset_local_vertices @ rotation.T
    translation_z = (
        float(actor._dataset_support_z)
        - float(rotated_vertices[:, 2].min())
        + float(drop_clearance)
    )
    actor.actor.set_pose(
        sapien.Pose(
            [float(xy[0]), float(xy[1]), translation_z],
            quaternion,
        )
    )
    actor._dataset_requested_orientation_mode = orientation_mode
    actor._dataset_orientation_mode = orientation_mode
    for component in actor.actor.get_components():
        if hasattr(component, "set_linear_velocity"):
            component.set_linear_velocity([0.0, 0.0, 0.0])
        if hasattr(component, "set_angular_velocity"):
            component.set_angular_velocity([0.0, 0.0, 0.0])


def _dynamic_components(actor: Any) -> list[Any]:
    return [
        component
        for component in actor.actor.get_components()
        if (
            hasattr(component, "get_linear_velocity")
            and hasattr(component, "get_angular_velocity")
        )
    ]


def _classify_settled_orientation(actor: Any) -> str:
    rotation = t3d.quaternions.quat2mat(actor.get_pose().q)
    upright_axis = rotation @ actor._dataset_local_upright_axis
    vertical_alignment = float(upright_axis[2])
    if vertical_alignment >= 0.72:
        return "upright"
    if vertical_alignment <= -0.72:
        return "upside_down"
    return "side"


def _has_deep_part_penetration(
    scene: Any,
    movable_parts: list[Any],
    *,
    tolerance: float = 0.008,
) -> bool:
    """Reject visible geometry states backed by a deep physics penetration."""
    part_names = {actor.get_name() for actor in movable_parts}
    for contact in scene.scene.get_contacts():
        body_names = {
            contact.bodies[0].entity.name,
            contact.bodies[1].entity.name,
        }
        if not (body_names & part_names):
            continue
        if any(
            float(point.separation) < -tolerance
            for point in contact.points
        ):
            return True
    return False


def _settle_parts(
    scene: Any,
    movable_parts: list[Any],
    expected_inside_names: set[str],
) -> bool:
    """Let PhysX resolve gravity and collisions before accepting a frame."""
    dynamic_components = [
        component
        for actor in movable_parts
        for component in _dynamic_components(actor)
    ]
    if len(dynamic_components) != len(movable_parts):
        raise RuntimeError("数据集中的零件必须都是可受重力影响的刚体")

    scene._dataset_last_rejection_reason = None
    # At 1/250 s this provides at least one second of physical settling.
    consecutive_stable_checks = 0
    for step_index in range(500):
        scene.scene.step()
        if step_index < 249 or (step_index + 1) % 25:
            continue

        stable = all(
            np.linalg.norm(component.get_linear_velocity()) < 0.025
            and np.linalg.norm(component.get_angular_velocity()) < 0.40
            for component in dynamic_components
        )
        consecutive_stable_checks = (
            consecutive_stable_checks + 1 if stable else 0
        )
        if consecutive_stable_checks >= 3:
            break
    else:
        scene._dataset_last_rejection_reason = "physics_not_stable"
        return False

    actual_inside_names = {
        actor.get_name()
        for actor in movable_parts
        if scene._is_part_placed(actor)
    }
    if actual_inside_names != expected_inside_names:
        scene._dataset_last_rejection_reason = "wrong_box_membership"
        return False

    # Reject parts that fell off the table or remain deeply interpenetrating.
    if any(
        float(actor.get_pose().p[2]) < float(actor._dataset_support_z) - 0.02
        for actor in movable_parts
    ):
        scene._dataset_last_rejection_reason = "part_left_table"
        return False
    if _has_deep_part_penetration(scene, movable_parts):
        scene._dataset_last_rejection_reason = "deep_penetration"
        return False

    for actor in movable_parts:
        actor._dataset_inside_box = actor.get_name() in actual_inside_names
        actor._dataset_orientation_mode = _classify_settled_orientation(actor)
    return True


def _randomize_scene(
    scene: Any,
    rng: np.random.Generator,
    base_camera_matrix: np.ndarray,
    *,
    inside_count: int,
) -> set[str] | None:
    movable_parts = [*scene.parts_a, *scene.parts_b]
    if not 0 <= inside_count <= len(BOX_INTERIOR_OFFSETS_XY):
        raise ValueError(
            f"盒内零件数量必须在0到{len(BOX_INTERIOR_OFFSETS_XY)}之间"
        )

    box_pose = scene.empty_box._dataset_base_pose
    box_xy = np.array(
        [
            rng.uniform(-0.025, 0.025),
            rng.uniform(-0.14, -0.105),
        ]
    )
    box_yaw = rng.uniform(np.deg2rad(-6), np.deg2rad(6))
    yaw_quaternion = t3d.quaternions.axangle2quat([0, 0, 1], box_yaw)
    box_quaternion = t3d.quaternions.qmult(yaw_quaternion, box_pose.q)
    scene.empty_box.actor.set_pose(
        sapien.Pose(
            [box_xy[0], box_xy[1], float(box_pose.p[2])],
            box_quaternion,
        )
    )

    inside_indices = set(
        int(index)
        for index in rng.choice(
            len(movable_parts),
            size=inside_count,
            replace=False,
        )
    )
    inside_offsets = BOX_INTERIOR_OFFSETS_XY[
        rng.permutation(len(BOX_INTERIOR_OFFSETS_XY))
    ][:inside_count]
    box_yaw_rotation = np.array(
        [
            [np.cos(box_yaw), -np.sin(box_yaw)],
            [np.sin(box_yaw), np.cos(box_yaw)],
        ]
    )

    inside_actors = [
        actor
        for index, actor in enumerate(movable_parts)
        if index in inside_indices
    ]
    outside_actors = [
        actor
        for index, actor in enumerate(movable_parts)
        if index not in inside_indices
    ]
    for actor, offset in zip(inside_actors, inside_offsets):
        world_xy = box_xy + box_yaw_rotation @ offset
        _set_actor_pose(
            actor,
            world_xy,
            rng.uniform(-np.pi, np.pi),
            _sample_orientation_mode(rng),
            # Start above the rim. Gravity and the real collision geometry
            # decide the final pose instead of placing through the box.
            drop_clearance=0.13,
        )

    slots = OBJECT_SLOTS_XY[rng.permutation(len(OBJECT_SLOTS_XY))][
        :len(outside_actors)
    ].copy()
    slots[:, 0] += rng.uniform(-0.022, 0.022, size=len(slots))
    slots[:, 1] += rng.uniform(-0.016, 0.016, size=len(slots))
    for actor, xy in zip(outside_actors, slots):
        _set_actor_pose(
            actor,
            xy,
            rng.uniform(-np.pi, np.pi),
            _sample_orientation_mode(rng),
            # A small air gap lets PhysX establish a valid table contact.
            drop_clearance=0.018,
        )

    expected_inside_names = {
        actor.get_name()
        for actor in inside_actors
    }
    if not _settle_parts(scene, movable_parts, expected_inside_names):
        return None

    camera_matrix = base_camera_matrix.copy()
    camera_matrix[:3, 3] += rng.uniform(
        [-0.012, -0.012, -0.008],
        [0.012, 0.012, 0.008],
    )
    angle_jitter = np.deg2rad(rng.uniform(-1.5, 1.5, size=3))
    camera_matrix[:3, :3] = (
        base_camera_matrix[:3, :3]
        @ t3d.euler.euler2mat(*angle_jitter)
    )
    _head_camera(scene).entity.set_pose(sapien.Pose(camera_matrix))

    ambient = rng.uniform(0.38, 0.62, size=3)
    scene.scene.set_ambient_light(ambient)
    for light in scene.direction_light_lst:
        light.set_color(rng.uniform(0.35, 0.70, size=3))
    for light in scene.point_light_lst:
        light.set_color(rng.uniform(0.75, 1.10, size=3))
    return expected_inside_names


def _capture_rgb_and_entity_ids(scene: Any) -> tuple[np.ndarray, np.ndarray]:
    scene._update_render()
    camera = _head_camera(scene)
    camera.take_picture()

    rgba = camera.get_picture("Color")
    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    segmentation = camera.get_picture("Segmentation")
    entity_ids = np.rint(segmentation[..., 1]).astype(np.int64)
    return rgb, entity_ids


def _extract_labels(
    entity_ids: np.ndarray,
    targets: list[tuple[Any, int, str]],
    *,
    min_visible_pixels: int = 100,
) -> list[dict[str, Any]] | None:
    height, width = entity_ids.shape
    labels: list[dict[str, Any]] = []

    for actor, class_id, actor_name in targets:
        mask = entity_ids == int(actor.actor.per_scene_id)
        ys, xs = np.nonzero(mask)
        if len(xs) < min_visible_pixels:
            return None

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        # Parts should be completely inside the image. The large container may
        # approach the lower edge in the real task and is intentionally kept.
        if class_id != 2 and (
            x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
        ):
            return None

        labels.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "actor_name": actor_name,
                "inside_box": bool(
                    getattr(actor, "_dataset_inside_box", False)
                ),
                "orientation_mode": str(
                    getattr(actor, "_dataset_orientation_mode", "upright")
                ),
                "visible_pixels": int(len(xs)),
                "bbox_xyxy": [x1, y1, x2, y2],
                "yolo_xywh": [
                    ((x1 + x2) / 2) / width,
                    ((y1 + y2) / 2) / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
            }
        )

    labels.sort(key=lambda item: (item["class_id"], item["actor_name"]))
    return labels


def _save_preview(
    rgb: np.ndarray,
    labels: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {
        0: (255, 80, 80),
        1: (255, 190, 40),
        2: (40, 180, 255),
    }
    for label in labels:
        color = colors[label["class_id"]]
        x1, y1, x2, y2 = label["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text(
            (x1 + 3, y1 + 3),
            label["class_name"],
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill=color,
        )
    image.save(output_path)


def _save_sample(
    output_dir: Path,
    split: str,
    index: int,
    rgb: np.ndarray,
    labels: list[dict[str, Any]],
    *,
    preview_count: int,
) -> Path:
    stem = f"{split}_{index:06d}"
    image_path = output_dir / "images" / split / f"{stem}.png"
    label_path = output_dir / "labels" / split / f"{stem}.txt"
    Image.fromarray(rgb).save(image_path)

    label_lines = []
    for label in labels:
        x_center, y_center, width, height = label["yolo_xywh"]
        label_lines.append(
            f'{label["class_id"]} '
            f"{x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    if index < preview_count:
        preview_path = output_dir / "previews" / f"{stem}.png"
        _save_preview(rgb, labels, preview_path)
    return image_path.resolve()


def generate_yolo_dataset(
    scene: Any,
    output_dir: Path,
    *,
    train_count: int = 500,
    val_count: int = 100,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate RGB images and exact YOLO boxes from actor segmentation."""
    if train_count <= 0 or val_count <= 0:
        raise ValueError("训练集和验证集数量都必须大于0")

    output_dir = output_dir.resolve()
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    targets: list[tuple[Any, int, str]] = []
    for actor in scene.parts_a:
        targets.append((actor, 0, actor.get_name()))
    for actor in scene.parts_b:
        targets.append((actor, 1, actor.get_name()))
    targets.append((scene.empty_box, 2, "box"))

    for actor, _, _ in targets:
        actor._dataset_base_pose = actor.get_pose()
        actor._dataset_support_z = float(actor.get_pose().p[2])

    camera = _head_camera(scene)
    base_camera_matrix = camera.entity.get_pose().to_transformation_matrix()
    rng = np.random.default_rng(seed)
    for actor, _, _ in targets:
        actor._dataset_rng = rng
    part_a_vertices = _load_scaled_visual_vertices(
        "004_fluted-block",
        0,
        np.asarray(scene.parts_a[0].config["scale"], dtype=float),
    )
    part_b_vertices = _load_scaled_visual_vertices(
        "055_small-speaker",
        1,
        np.asarray(scene.parts_b[0].config["scale"], dtype=float),
    )
    for actor in scene.parts_a:
        actor._dataset_local_vertices = part_a_vertices
    for actor in scene.parts_b:
        actor._dataset_local_vertices = part_b_vertices
    for actor in (*scene.parts_a, *scene.parts_b):
        base_rotation = t3d.quaternions.quat2mat(actor._dataset_base_pose.q)
        actor._dataset_local_upright_axis = (
            base_rotation.T @ np.array([0.0, 0.0, 1.0])
        )

    split_counts = {"train": train_count, "val": val_count}
    manifests: dict[str, list[str]] = {"train": [], "val": []}
    sample_annotations: list[dict[str, Any]] = []
    rejected_samples = 0
    rejection_reasons: dict[str, int] = {}
    accepted_inside_histogram = {str(count): 0 for count in range(4)}
    accepted_orientation_histogram = {
        "upright": 0,
        "side": 0,
        "upside_down": 0,
    }

    for split, requested_count in split_counts.items():
        # Cycling through 0,1,2,3 guarantees that the final dataset contains
        # both empty-box scenes and scenes with up to three parts in the box.
        inside_count_schedule = np.tile(
            np.arange(4),
            int(np.ceil(requested_count / 4)),
        )[:requested_count]
        rng.shuffle(inside_count_schedule)
        accepted = 0
        attempts = 0
        max_attempts = max(requested_count * 20, 100)
        while accepted < requested_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"{split} 生成失败：尝试 {max_attempts} 次后只接受了 "
                    f"{accepted}/{requested_count} 张；请检查相机视野或分割标签。"
                )

            desired_inside_count = int(inside_count_schedule[accepted])
            inside_objects = _randomize_scene(
                scene,
                rng,
                base_camera_matrix,
                inside_count=desired_inside_count,
            )
            if inside_objects is None:
                rejected_samples += 1
                reason = str(
                    getattr(
                        scene,
                        "_dataset_last_rejection_reason",
                        "physics_rejected",
                    )
                )
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            rgb, entity_ids = _capture_rgb_and_entity_ids(scene)
            labels = _extract_labels(entity_ids, targets)
            if labels is None:
                rejected_samples += 1
                rejection_reasons["visibility_or_truncation"] = (
                    rejection_reasons.get("visibility_or_truncation", 0) + 1
                )
                continue

            image_path = _save_sample(
                output_dir,
                split,
                accepted,
                rgb,
                labels,
                preview_count=10 if split == "train" else 5,
            )
            manifests[split].append(str(image_path))
            sample_annotations.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "inside_box_objects": sorted(inside_objects),
                    "labels": labels,
                }
            )
            accepted_inside_histogram[str(desired_inside_count)] += 1
            for label in labels:
                if label["class_id"] in (0, 1):
                    accepted_orientation_histogram[
                        label["orientation_mode"]
                    ] += 1
            accepted += 1
            if accepted % 25 == 0 or accepted == requested_count:
                print(
                    f"[{split}] 已生成 {accepted}/{requested_count} 张，"
                    f"累计丢弃不可见/截断样本 {rejected_samples} 张"
                )

    for split, image_paths in manifests.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(image_paths) + "\n",
            encoding="utf-8",
        )

    annotations_path = output_dir / "sample_annotations.jsonl"
    annotations_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in sample_annotations
        ),
        encoding="utf-8",
    )

    dataset_yaml = {
        "path": str(output_dir),
        "train": "train.txt",
        "val": "val.txt",
        "names": CLASS_NAMES,
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(
            dataset_yaml,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    metadata = {
        "camera": {
            "name": "head_camera",
            "resolution": [640, 480],
            "rgb_used_for_training": True,
        },
        "offline_label_source": (
            "SAPIEN actor/entity segmentation channel; not included in "
            "runtime model input"
        ),
        "classes": CLASS_NAMES,
        "assets": {
            "part_A": {
                "asset": "004_fluted-block/base0",
                "count_per_image": 3,
                "runtime_scale_factor": 0.80,
            },
            "part_B": {
                "asset": "055_small-speaker/base1",
                "count_per_image": 2,
            },
            "box": {
                "asset": "062_plasticbox/base3",
                "count_per_image": 1,
                "runtime_horizontal_scale_factor": 1.55,
            },
        },
        "train_count": train_count,
        "val_count": val_count,
        "seed": seed,
        "rejected_samples": rejected_samples,
        "rejection_reasons": rejection_reasons,
        "accepted_frames_by_inside_part_count": accepted_inside_histogram,
        "accepted_part_instances_by_orientation": (
            accepted_orientation_histogram
        ),
        "randomization": {
            "part_position": True,
            "part_horizontal_yaw_uniform_degrees": [-180, 180],
            "part_orientation_probabilities": {
                "upright": 0.60,
                "side_at_plus_or_minus_90_degrees": 0.35,
                "upside_down_at_180_degrees": 0.05,
            },
            "geometry_based_spawn_height": True,
            "physx_gravity_and_collision_settling": True,
            "reject_deep_penetration_over_meters": 0.008,
            "reject_unstable_or_wrong_box_membership": True,
            "inside_box_part_count": [0, 1, 2, 3],
            "box_position_and_small_yaw": True,
            "head_camera_small_pose_jitter": True,
            "lighting": True,
        },
    }
    metadata_path = output_dir / "generation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "data_yaml": str((output_dir / "data.yaml").resolve()),
        "metadata": str(metadata_path.resolve()),
        "annotations": str(annotations_path.resolve()),
        "train_count": train_count,
        "val_count": val_count,
        "rejected_samples": rejected_samples,
        "rejection_reasons": rejection_reasons,
        "accepted_frames_by_inside_part_count": accepted_inside_histogram,
        "accepted_part_instances_by_orientation": (
            accepted_orientation_histogram
        ),
    }
