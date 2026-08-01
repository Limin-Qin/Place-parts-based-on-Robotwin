"""Generate a small multi-camera, occlusion-focused YOLO supplement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml
from PIL import Image, ImageDraw

from envs.utils import ArmTag
from examples.my_parts_box_scene.agent.robot_skills import RobotSkills

from .dataset_generator import (
    BOX_INTERIOR_OFFSETS_XY,
    CLASS_NAMES,
    _load_scaled_visual_vertices,
    _sample_orientation_mode,
    _set_actor_pose,
    _settle_parts,
)


def _prepare_scene(
    scene: Any,
    rng: np.random.Generator,
) -> tuple[list[Any], list[tuple[Any, int, str]]]:
    parts = [*scene.parts_a, *scene.parts_b]
    targets: list[tuple[Any, int, str]] = [
        *((actor, 0, actor.get_name()) for actor in scene.parts_a),
        *((actor, 1, actor.get_name()) for actor in scene.parts_b),
        (scene.empty_box, 2, "box"),
    ]
    for actor, _, _ in targets:
        actor._dataset_base_pose = actor.get_pose()
        actor._dataset_support_z = float(actor.get_pose().p[2])
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
    for actor in parts:
        base_rotation = t3d.quaternions.quat2mat(
            actor._dataset_base_pose.q
        )
        actor._dataset_local_upright_axis = (
            base_rotation.T @ np.array([0.0, 0.0, 1.0])
        )
    return parts, targets


def _outside_box(candidate: np.ndarray, box_xy: np.ndarray) -> bool:
    relative = candidate - box_xy
    return not (
        abs(float(relative[0])) < 0.21
        and -0.18 < float(relative[1]) < 0.20
    )


def _sample_outside_positions(
    rng: np.random.Generator,
    count: int,
    box_xy: np.ndarray,
    scenario_index: int,
) -> list[np.ndarray] | None:
    """Sample supported, non-overlapping positions with edge/occlusion bias."""
    positions: list[np.ndarray] = []
    edge_sign = -1.0 if scenario_index % 2 == 0 else 1.0
    presets_by_scenario = (
        [
            [edge_sign * 0.52, 0.18],
            [-edge_sign * 0.45, -0.285],
        ],
        [
            [-0.31, -0.12],
            [-0.30, 0.04],
            [0.50, 0.22],
        ],
        [
            [0.31, -0.12],
            [0.30, 0.045],
            [-0.51, 0.24],
        ],
        [
            [-0.50, 0.27],
            [0.50, 0.27],
            [0.36, -0.275],
        ],
        [
            [-0.34, -0.15],
            [-0.33, 0.01],
            [0.52, 0.10],
        ],
    )
    presets = [
        np.asarray(position, dtype=float)
        for position in presets_by_scenario[
            scenario_index % len(presets_by_scenario)
        ]
    ]
    rng.shuffle(presets)

    def acceptable(candidate: np.ndarray) -> bool:
        return bool(
            -0.53 <= candidate[0] <= 0.53
            and -0.29 <= candidate[1] <= 0.30
            and _outside_box(candidate, box_xy)
            and all(
                np.linalg.norm(candidate - existing) >= 0.115
                for existing in positions
            )
        )

    for preset in presets:
        jittered = preset + rng.uniform(
            [-0.012, -0.010],
            [0.012, 0.010],
        )
        if acceptable(jittered):
            positions.append(jittered)
        if len(positions) == count:
            return positions

    for _ in range(4000):
        candidate = np.asarray(
            [
                rng.uniform(-0.53, 0.53),
                rng.uniform(-0.29, 0.30),
            ],
            dtype=float,
        )
        if not acceptable(candidate):
            continue
        positions.append(candidate)
        if len(positions) == count:
            return positions
    return None


def _randomize_physical_scene(
    scene: Any,
    parts: list[Any],
    rng: np.random.Generator,
    scenario_index: int,
) -> set[str] | None:
    box_base = scene.empty_box._dataset_base_pose
    box_xy = np.asarray(
        [
            rng.uniform(-0.018, 0.018),
            rng.uniform(-0.15, -0.12),
        ],
        dtype=float,
    )
    box_yaw = float(rng.uniform(-0.10, 0.10))
    box_quaternion = t3d.quaternions.qmult(
        t3d.quaternions.axangle2quat([0, 0, 1], box_yaw),
        box_base.q,
    )
    scene.empty_box.actor.set_pose(
        sapien.Pose(
            [box_xy[0], box_xy[1], float(box_base.p[2])],
            box_quaternion,
        )
    )

    inside_schedule = (0, 1, 2, 1, 0)
    inside_count = inside_schedule[scenario_index % len(inside_schedule)]
    shuffled_indices = rng.permutation(len(parts))
    inside_indices = set(
        int(index) for index in shuffled_indices[:inside_count]
    )
    inside_actors = [
        actor
        for index, actor in enumerate(parts)
        if index in inside_indices
    ]
    outside_actors = [
        actor
        for index, actor in enumerate(parts)
        if index not in inside_indices
    ]
    rng.shuffle(inside_actors)
    rng.shuffle(outside_actors)

    box_rotation = np.asarray(
        [
            [np.cos(box_yaw), -np.sin(box_yaw)],
            [np.sin(box_yaw), np.cos(box_yaw)],
        ],
        dtype=float,
    )
    offsets = BOX_INTERIOR_OFFSETS_XY[
        rng.permutation(len(BOX_INTERIOR_OFFSETS_XY))
    ][:inside_count]
    for actor, offset in zip(inside_actors, offsets):
        _set_actor_pose(
            actor,
            box_xy + box_rotation @ offset,
            float(rng.uniform(-np.pi, np.pi)),
            _sample_orientation_mode(rng),
            drop_clearance=0.13,
        )

    positions = _sample_outside_positions(
        rng,
        len(outside_actors),
        box_xy,
        scenario_index,
    )
    if positions is None:
        scene._dataset_last_rejection_reason = "position_sampling_failed"
        return None
    rng.shuffle(positions)
    for actor, position in zip(outside_actors, positions):
        _set_actor_pose(
            actor,
            position,
            float(rng.uniform(-np.pi, np.pi)),
            _sample_orientation_mode(rng),
            drop_clearance=0.018,
        )

    expected_inside = {actor.get_name() for actor in inside_actors}
    if not _settle_parts(scene, parts, expected_inside):
        return None

    scene.scene.set_ambient_light(rng.uniform(0.40, 0.62, size=3))
    for light in scene.direction_light_lst:
        light.set_color(rng.uniform(0.38, 0.72, size=3))
    for light in scene.point_light_lst:
        light.set_color(rng.uniform(0.75, 1.08, size=3))
    return expected_inside


def _capture_camera(
    scene: Any,
    camera: Any,
) -> tuple[np.ndarray, np.ndarray]:
    scene.cameras.update_wrist_camera(
        scene.robot.left_camera.get_entity_pose(),
        scene.robot.right_camera.get_entity_pose(),
    )
    scene._update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    segmentation = camera.get_picture("Segmentation")
    entity_ids = np.rint(segmentation[..., 1]).astype(np.int64)
    return rgb, entity_ids


def _extract_partial_labels(
    entity_ids: np.ndarray,
    targets: list[tuple[Any, int, str]],
    *,
    min_visible_pixels: int = 24,
) -> list[dict[str, Any]]:
    height, width = entity_ids.shape
    labels: list[dict[str, Any]] = []
    for actor, class_id, actor_name in targets:
        mask = entity_ids == int(actor.actor.per_scene_id)
        ys, xs = np.nonzero(mask)
        if len(xs) < min_visible_pixels:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        box_area = max(1, (x2 - x1) * (y2 - y1))
        touches_edge = bool(
            x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
        )
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
                "touches_image_edge": touches_edge,
                "visible_fill_ratio": round(float(len(xs) / box_area), 4),
                "partially_occluded_or_truncated": bool(
                    touches_edge or len(xs) / box_area < 0.62
                ),
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


def _preview_image(
    rgb: np.ndarray,
    labels: list[dict[str, Any]],
) -> Image.Image:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {
        0: (255, 80, 80),
        1: (255, 190, 40),
        2: (40, 180, 255),
    }
    for label in labels:
        x1, y1, x2, y2 = label["bbox_xyxy"]
        color = colors[label["class_id"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        suffix = " edge" if label["touches_image_edge"] else ""
        draw.text(
            (x1 + 2, max(0, y1 - 13)),
            f'{label["class_name"]}{suffix}',
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill=color,
        )
    return image


def _save_sample(
    output_dir: Path,
    split: str,
    stem: str,
    rgb: np.ndarray,
    labels: list[dict[str, Any]],
) -> dict[str, str]:
    image_path = output_dir / "images" / split / f"{stem}.png"
    label_path = output_dir / "labels" / split / f"{stem}.txt"
    preview_path = output_dir / "previews" / f"{stem}.png"
    Image.fromarray(rgb).save(image_path)
    label_path.write_text(
        "".join(
            (
                f'{label["class_id"]} '
                f'{label["yolo_xywh"][0]:.8f} '
                f'{label["yolo_xywh"][1]:.8f} '
                f'{label["yolo_xywh"][2]:.8f} '
                f'{label["yolo_xywh"][3]:.8f}\n'
            )
            for label in labels
        ),
        encoding="utf-8",
    )
    _preview_image(rgb, labels).save(preview_path)
    return {
        "image": str(image_path.resolve()),
        "label": str(label_path.resolve()),
        "preview": str(preview_path.resolve()),
    }


def _move_home(scene: Any, arm: ArmTag) -> bool:
    scene.plan_success = True
    return bool(scene.move(scene.back_to_origin(arm_tag=arm)))


def _capture_wrist_target(
    scene: Any,
    skills: RobotSkills,
    targets: list[tuple[Any, int, str]],
    candidate_actors: list[Any],
    arm: ArmTag,
) -> tuple[np.ndarray, list[dict[str, Any]], str] | None:
    camera = (
        scene.cameras.left_camera
        if str(arm) == "left"
        else scene.cameras.right_camera
    )
    ordered = sorted(
        candidate_actors,
        key=lambda actor: float(actor.get_pose().p[0]),
        reverse=str(arm) == "right",
    )
    for actor in ordered:
        scene.plan_success = True
        action = skills._pregrasp_action(
            actor,
            arm,
            pre_grasp_distance=0.15,
        )
        if action is None or not scene.move(action) or not scene.plan_success:
            _move_home(scene, arm)
            continue
        rgb, entity_ids = _capture_camera(scene, camera)
        labels = _extract_partial_labels(entity_ids, targets)
        if any(
            label["actor_name"] == actor.get_name()
            for label in labels
        ):
            return rgb, labels, actor.get_name()
        _move_home(scene, arm)
    return None


def generate_multicamera_supplement(
    scene: Any,
    output_dir: Path,
    *,
    scene_count: int = 5,
    seed: int = 2030,
) -> dict[str, Any]:
    """Generate three views per physical scene: head, left wrist, right wrist."""
    if scene_count <= 0:
        raise ValueError("补充数据场景数量必须大于0")
    output_dir = output_dir.resolve()
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    parts, targets = _prepare_scene(scene, rng)
    skills = RobotSkills(scene)
    camera_by_name = dict(
        zip(
            scene.cameras.static_camera_name,
            scene.cameras.static_camera_list,
        )
    )
    head_camera = camera_by_name["head_camera"]
    records: list[dict[str, Any]] = []
    rejected_scenes = 0
    rejection_reasons: dict[str, int] = {}
    accepted = 0
    attempts = 0
    max_attempts = max(60, scene_count * 15)

    while accepted < scene_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"补充集尝试{max_attempts}次后只生成"
                f"{accepted}/{scene_count}个完整三相机场景"
            )
        _move_home(scene, ArmTag("left"))
        _move_home(scene, ArmTag("right"))
        inside_names = _randomize_physical_scene(
            scene,
            parts,
            rng,
            accepted,
        )
        if inside_names is None:
            rejected_scenes += 1
            reason = str(
                getattr(
                    scene,
                    "_dataset_last_rejection_reason",
                    "physics_rejected",
                )
            )
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        outside_parts = [
            actor
            for actor in parts
            if actor.get_name() not in inside_names
        ]
        if len(outside_parts) < 2:
            rejected_scenes += 1
            rejection_reasons["not_enough_outside_targets"] = (
                rejection_reasons.get("not_enough_outside_targets", 0) + 1
            )
            continue

        head_at_home = accepted % 2 == 1
        head_capture = None
        if head_at_home:
            head_rgb, head_ids = _capture_camera(scene, head_camera)
            head_capture = (
                head_rgb,
                _extract_partial_labels(head_ids, targets),
                "home_arms",
            )

        left_capture = _capture_wrist_target(
            scene,
            skills,
            targets,
            [actor for actor in outside_parts if actor.get_pose().p[0] <= 0.12]
            or outside_parts,
            ArmTag("left"),
        )
        if left_capture is None:
            rejected_scenes += 1
            rejection_reasons["left_wrist_target_not_visible"] = (
                rejection_reasons.get("left_wrist_target_not_visible", 0) + 1
            )
            continue
        if not head_at_home:
            head_rgb, head_ids = _capture_camera(scene, head_camera)
            head_capture = (
                head_rgb,
                _extract_partial_labels(head_ids, targets),
                "left_arm_pregrasp_occlusion",
            )
        _move_home(scene, ArmTag("left"))

        left_focused_actor = left_capture[2]
        right_target_pool = [
            actor
            for actor in outside_parts
            if actor.get_name() != left_focused_actor
        ]
        right_capture = _capture_wrist_target(
            scene,
            skills,
            targets,
            [
                actor
                for actor in right_target_pool
                if actor.get_pose().p[0] >= -0.12
            ]
            or right_target_pool,
            ArmTag("right"),
        )
        if right_capture is None:
            rejected_scenes += 1
            rejection_reasons["right_wrist_target_not_visible"] = (
                rejection_reasons.get("right_wrist_target_not_visible", 0) + 1
            )
            _move_home(scene, ArmTag("right"))
            continue
        _move_home(scene, ArmTag("right"))

        assert head_capture is not None
        if not head_capture[1]:
            rejected_scenes += 1
            rejection_reasons["head_no_visible_labels"] = (
                rejection_reasons.get("head_no_visible_labels", 0) + 1
            )
            continue

        validation_scene_count = max(1, int(round(scene_count * 0.2)))
        split = (
            "val"
            if accepted >= scene_count - validation_scene_count
            else "train"
        )
        scene_stem = f"scene_{accepted:03d}"
        captures = (
            (
                "head_camera",
                head_capture[0],
                head_capture[1],
                head_capture[2],
                None,
            ),
            (
                "left_camera",
                left_capture[0],
                left_capture[1],
                "left_pregrasp",
                left_capture[2],
            ),
            (
                "right_camera",
                right_capture[0],
                right_capture[1],
                "right_pregrasp",
                right_capture[2],
            ),
        )
        for camera_name, rgb, labels, robot_state, focused_actor in captures:
            stem = f"{scene_stem}_{camera_name}"
            paths = _save_sample(
                output_dir,
                split,
                stem,
                rgb,
                labels,
            )
            records.append(
                {
                    "split": split,
                    "scene_index": accepted,
                    "camera": camera_name,
                    "robot_state": robot_state,
                    "focused_actor": focused_actor,
                    "inside_box_objects": sorted(inside_names),
                    "paths": paths,
                    "labels": labels,
                }
            )
        accepted += 1
        if accepted == 1 or accepted % 10 == 0 or accepted == scene_count:
            print(
                f"[补充集] 已生成 {accepted}/{scene_count} 个物理场景，"
                f"共 {accepted * 3}/{scene_count * 3} 张图片",
                flush=True,
            )

    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir),
                "train": "images/train",
                "val": "images/val",
                "names": CLASS_NAMES,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    annotations_path = output_dir / "sample_annotations.jsonl"
    annotations_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    camera_counts = {
        camera_name: sum(
            record["camera"] == camera_name for record in records
        )
        for camera_name in ("head_camera", "left_camera", "right_camera")
    }
    edge_label_count = sum(
        label["touches_image_edge"]
        for record in records
        for label in record["labels"]
    )
    partial_label_count = sum(
        label["partially_occluded_or_truncated"]
        for record in records
        for label in record["labels"]
    )
    metadata = {
        "purpose": (
            "inspection-sized supplement for the 62-degree head camera, "
            "wrist cameras, edge truncation and robot/object occlusion"
        ),
        "scene_count": scene_count,
        "image_count": len(records),
        "camera_image_counts": camera_counts,
        "head_camera_vertical_fov_degrees": 62.0,
        "seed": seed,
        "physics": {
            "gravity_and_collision_settling": True,
            "deep_penetration_rejected": True,
            "wrong_box_membership_rejected": True,
            "minimum_outside_center_clearance_m": 0.115,
        },
        "contains_box_interior_parts": any(
            record["inside_box_objects"] for record in records
        ),
        "edge_touching_label_count": int(edge_label_count),
        "partial_or_occluded_label_count": int(partial_label_count),
        "robot_occlusion_head_scenes": sum(
            record["camera"] == "head_camera"
            and record["robot_state"] != "home_arms"
            for record in records
        ),
        "rejected_scenes": rejected_scenes,
        "rejection_reasons": rejection_reasons,
        "offline_label_source": (
            "SAPIEN entity segmentation; segmentation is not a runtime input"
        ),
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
        **metadata,
    }
