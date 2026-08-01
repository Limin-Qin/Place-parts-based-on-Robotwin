"""Generate RGB-D stress scenes and evaluate position extraction per condition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .dataset_generator import (
    _load_scaled_visual_vertices,
    _set_actor_pose,
    _settle_parts,
)
from .rgbd_position_test import run_rgbd_position_evaluation


def _head_camera(scene: Any):
    cameras = dict(
        zip(
            scene.cameras.static_camera_name,
            scene.cameras.static_camera_list,
        )
    )
    if "head_camera" not in cameras:
        raise RuntimeError("当前场景没有启用head_camera")
    return cameras["head_camera"]


def _prepare_randomization(
    scene: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    parts = [*scene.parts_a, *scene.parts_b]
    for actor in [*parts, scene.empty_box]:
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
        base_rotation = t3d.quaternions.quat2mat(actor._dataset_base_pose.q)
        actor._dataset_local_upright_axis = (
            base_rotation.T @ np.array([0.0, 0.0, 1.0])
        )

    camera = _head_camera(scene)
    return {
        "parts": parts,
        "camera_pose": camera.entity.get_pose(),
        "camera_matrix": camera.entity.get_pose().to_transformation_matrix(),
        "box_pose": scene.empty_box.get_pose(),
    }


def _restore_static_state(scene: Any, state: dict[str, Any]) -> None:
    scene.empty_box.actor.set_pose(state["box_pose"])
    _head_camera(scene).entity.set_pose(state["camera_pose"])


def _place_parts(
    scene: Any,
    state: dict[str, Any],
    specs: list[tuple[np.ndarray, str, bool]],
    rng: np.random.Generator,
) -> set[str]:
    parts = state["parts"]
    if len(specs) != len(parts):
        raise ValueError("每个零件必须有一个鲁棒性场景位姿")
    expected_inside: set[str] = set()
    for actor, (xy, orientation, inside) in zip(parts, specs):
        actor._dataset_rng = rng
        _set_actor_pose(
            actor,
            np.asarray(xy, dtype=float),
            float(rng.uniform(-np.pi, np.pi)),
            orientation,
            drop_clearance=0.13 if inside else 0.018,
        )
        if inside:
            expected_inside.add(actor.get_name())
    if not _settle_parts(scene, parts, expected_inside):
        reason = getattr(scene, "_dataset_last_rejection_reason", "unknown")
        raise RuntimeError(f"鲁棒性场景物理落稳失败：{reason}")
    return expected_inside


def _standard_specs(
    modes: list[str] | None = None,
) -> list[tuple[np.ndarray, str, bool]]:
    positions = [
        [-0.22, 0.08],
        [0.00, 0.11],
        [0.22, 0.08],
        [-0.32, 0.21],
        [0.32, 0.21],
    ]
    modes = modes or ["upright"] * 5
    return [
        (np.asarray(xy, dtype=float), mode, False)
        for xy, mode in zip(positions, modes)
    ]


def _inside_specs() -> list[tuple[np.ndarray, str, bool]]:
    # The first two parts fall into separate regions of the real box collision
    # mesh; the remaining parts stay outside and fully visible.
    return [
        (np.asarray([-0.070, -0.105]), "side", True),
        (np.asarray([0.070, -0.105]), "upright", True),
        (np.asarray([0.000, 0.105]), "side", False),
        (np.asarray([-0.29, 0.19]), "upright", False),
        (np.asarray([0.29, 0.19]), "side", False),
    ]


def _arm_occlusion_specs() -> list[tuple[np.ndarray, str, bool]]:
    # Put the front left/right parts behind the real home-state grippers in
    # image space.  Their world-space gap from the TCPs remains large enough
    # to avoid manufacturing a physical interpenetration just for the test.
    return [
        (np.asarray([-0.285, -0.180]), "upright", False),
        (np.asarray([0.000, 0.105]), "side", False),
        (np.asarray([0.285, -0.180]), "upright", False),
        (np.asarray([-0.20, 0.19]), "side", False),
        (np.asarray([0.20, 0.19]), "upright", False),
    ]


def _crowded_specs() -> list[tuple[np.ndarray, str, bool]]:
    return [
        (np.asarray([-0.145, 0.055]), "upright", False),
        (np.asarray([-0.135, 0.145]), "side", False),
        (np.asarray([0.000, 0.055]), "upright", False),
        (np.asarray([0.010, 0.145]), "side", False),
        (np.asarray([0.145, 0.095]), "upright", False),
    ]


def _perturb_camera(
    scene: Any,
    base_pose: sapien.Pose,
    *,
    translation: np.ndarray,
    rotation_degrees: np.ndarray,
) -> None:
    base_rotation = t3d.quaternions.quat2mat(base_pose.q)
    rotation = t3d.euler.euler2mat(
        *np.deg2rad(np.asarray(rotation_degrees, dtype=float))
    )
    perturbed_rotation = base_rotation @ rotation
    perturbed_position = (
        np.asarray(base_pose.p, dtype=float)
        + np.asarray(translation, dtype=float)
    )
    # Construct the pose explicitly. Passing an already transformed matrix to
    # ``sapien.Pose`` is ambiguous across SAPIEN versions and previously moved
    # this camera close to the robot body instead of applying a small offset.
    _head_camera(scene).entity.set_pose(
        sapien.Pose(
            p=perturbed_position,
            q=t3d.quaternions.mat2quat(perturbed_rotation),
        )
    )


def _degrade_position_image(
    position_image: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_sigma_m: float,
    random_dropout_rate: float,
    edge_dropout_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    degraded = np.asarray(position_image).copy()
    camera_xyz = degraded[..., :3]
    valid = (degraded[..., 3] < 1.0) & np.isfinite(camera_xyz).all(axis=2)
    depth = -camera_xyz[..., 2]

    noise = rng.normal(0.0, noise_sigma_m, size=depth.shape)
    noisy_depth = np.maximum(depth + noise, 0.05)
    scale = np.ones_like(depth)
    scale[valid] = noisy_depth[valid] / np.maximum(depth[valid], 1e-6)
    camera_xyz[valid] *= scale[valid, None]

    gradient_y, gradient_x = np.gradient(depth)
    gradient = np.hypot(gradient_x, gradient_y)
    valid_gradients = gradient[valid & np.isfinite(gradient)]
    edge_threshold = (
        float(np.percentile(valid_gradients, 88))
        if len(valid_gradients)
        else np.inf
    )
    edge_pixels = valid & (gradient >= edge_threshold)
    random_dropout = valid & (
        rng.random(depth.shape) < random_dropout_rate
    )
    edge_dropout = edge_pixels & (
        rng.random(depth.shape) < edge_dropout_rate
    )
    dropout = random_dropout | edge_dropout
    degraded[dropout, :3] = 0.0
    degraded[dropout, 3] = 1.0
    return degraded, {
        "gaussian_depth_noise_sigma_m": noise_sigma_m,
        "random_dropout_rate": random_dropout_rate,
        "edge_dropout_rate": edge_dropout_rate,
        "invalidated_pixels": int(np.sum(dropout)),
    }


def _capture_rgbd(scene: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene._update_render()
    camera = _head_camera(scene)
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    position = camera.get_picture("Position")
    camera_to_world = camera.get_model_matrix()
    return rgb, position, camera_to_world


def run_rgbd_robustness_suite(
    scene: Any,
    output_dir: Path,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    state = _prepare_randomization(scene, rng)
    privileged_actors = {
        "part_A": list(scene.parts_a),
        "part_B": list(scene.parts_b),
    }

    scenarios = [
        {
            "name": "01_random_3d_orientation",
            "description": "随机水平角，3个侧躺、1个翻面、1个直立",
            "specs": _standard_specs(
                ["side", "upside_down", "side", "upright", "side"]
            ),
        },
        {
            "name": "02_crowded_occlusion",
            "description": "零件密集成两列，产生物体间部分遮挡",
            "specs": _crowded_specs(),
        },
        {
            "name": "03_camera_perturbation",
            "description": "相机平移约1厘米并旋转约1度",
            "specs": _standard_specs(),
            "camera_translation": [0.008, -0.006, 0.005],
            "camera_rotation_degrees": [0.8, -0.7, 0.8],
        },
        {
            "name": "04_depth_noise_dropout",
            "description": "6毫米深度噪声、随机缺失和边缘缺失",
            "specs": _standard_specs(),
            "depth_degradation": {
                "noise_sigma_m": 0.006,
                "random_dropout_rate": 0.10,
                "edge_dropout_rate": 0.65,
            },
        },
        {
            "name": "05_parts_inside_box",
            "description": "2个零件落入盒内，其余零件含侧躺姿态",
            "specs": _inside_specs(),
        },
        {
            "name": "06_robot_arm_occlusion",
            "description": "真实双臂夹爪位于相机和前排零件之间",
            "specs": _arm_occlusion_specs(),
        },
        {
            "name": "07_combined_stress",
            "description": "盒内、侧躺、相机扰动和深度退化同时存在",
            "specs": _inside_specs(),
            "camera_translation": [-0.008, 0.006, -0.005],
            "camera_rotation_degrees": [-0.8, 0.7, -0.8],
            "depth_degradation": {
                "noise_sigma_m": 0.008,
                "random_dropout_rate": 0.14,
                "edge_dropout_rate": 0.75,
            },
        },
    ]

    scenario_summaries = []
    for scenario in scenarios:
        _restore_static_state(scene, state)
        inside_names = _place_parts(
            scene,
            state,
            scenario["specs"],
            rng,
        )
        if "camera_translation" in scenario:
            _perturb_camera(
                scene,
                state["camera_pose"],
                translation=np.asarray(scenario["camera_translation"]),
                rotation_degrees=np.asarray(
                    scenario["camera_rotation_degrees"]
                ),
            )

        rgb, position_image, camera_to_world = _capture_rgbd(scene)
        degradation_metadata = None
        if "depth_degradation" in scenario:
            position_image, degradation_metadata = _degrade_position_image(
                position_image,
                rng,
                **scenario["depth_degradation"],
            )

        scenario_dir = output_dir / scenario["name"]
        result = run_rgbd_position_evaluation(
            rgb,
            position_image,
            camera_to_world,
            privileged_actors,
            scenario_dir,
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        summary = {
            "name": scenario["name"],
            "description": scenario["description"],
            "inside_box_objects": sorted(inside_names),
            "camera_translation_m": scenario.get(
                "camera_translation",
                [0.0, 0.0, 0.0],
            ),
            "camera_rotation_degrees": scenario.get(
                "camera_rotation_degrees",
                [0.0, 0.0, 0.0],
            ),
            "depth_degradation": degradation_metadata,
            "detected_counts": {
                label: check["detected"]
                for label, check in result["model"].get(
                    "count_check",
                    {},
                ).items()
            },
            "matched_count": result["metrics"]["matched_count"],
            "missing_actor_names": result["missing_privileged_actors"],
            "failed_depth_detection_count": (
                result["failed_depth_detection_count"]
            ),
            "metrics": result["metrics"],
            "position_overlay": result["outputs"]["position_overlay"],
            "comparison_json": result["outputs"]["comparison_json"],
        }
        # Detection counts live in the detector JSON, not the model metadata.
        detector_json = json.loads(
            Path(result["outputs"]["detector_overlay"])
            .with_name("head_camera_yolo_world.json")
            .read_text(encoding="utf-8")
        )
        summary["detected_counts"] = {
            label: values["detected"]
            for label, values in detector_json["count_check"].items()
        }
        scenario_summaries.append(summary)
        mean_error = result["metrics"]["mean_horizontal_error_m"]
        mean_text = (
            f"{mean_error * 1000:.1f} mm"
            if mean_error is not None
            else "无法计算"
        )
        print(
            f"[{scenario['name']}] "
            f"匹配={result['metrics']['matched_count']}/5，"
            f"平均水平误差={mean_text}"
        )

    valid_mean_errors = [
        item["metrics"]["mean_horizontal_error_m"]
        for item in scenario_summaries
        if item["metrics"]["mean_horizontal_error_m"] is not None
    ]
    payload = {
        "seed": seed,
        "model_path": model_path,
        "scenario_count": len(scenario_summaries),
        "scenarios": scenario_summaries,
        "aggregate": {
            "mean_of_scenario_horizontal_errors_m": (
                round(float(np.mean(valid_mean_errors)), 6)
                if valid_mean_errors
                else None
            ),
            "total_matched_parts": int(
                sum(item["matched_count"] for item in scenario_summaries)
            ),
            "total_expected_parts": 5 * len(scenario_summaries),
        },
        "note": (
            "All position estimates use RGB-D only. Simulator actor poses are "
            "read afterward solely for offline error evaluation."
        ),
    }
    summary_path = output_dir / "robustness_summary.json"
    payload["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
