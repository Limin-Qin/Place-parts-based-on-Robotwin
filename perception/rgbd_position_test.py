"""Estimate tabletop part positions from head RGB-D and evaluate against simulator truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .yolo_world_test import run_yolo_world_single_frame


PART_CLASSES = ("part_A", "part_B")
MIN_RELIABLE_OBJECT_DEPTH_PIXELS = 200
MAX_RELIABLE_XY_CENTER_DISAGREEMENT_M = 0.03


def _world_position_map(
    position_image: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if position_image.ndim != 3 or position_image.shape[2] < 4:
        raise ValueError(
            "SAPIEN Position图应为HxWx4，"
            f"实际形状为{position_image.shape}"
        )
    camera_xyz = np.asarray(position_image[..., :3], dtype=np.float64)
    valid = (
        np.asarray(position_image[..., 3]) < 1.0
    ) & np.isfinite(camera_xyz).all(axis=2)
    rotation = np.asarray(camera_to_world[:3, :3], dtype=np.float64)
    translation = np.asarray(camera_to_world[:3, 3], dtype=np.float64)
    world_xyz = camera_xyz @ rotation.T + translation
    world_xyz[~valid] = np.nan
    return world_xyz, valid


def _estimate_table_height(
    world_xyz: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, int]:
    # The table is the dominant horizontal surface in the calibrated robot
    # workspace. Its height is inferred from RGB-D rather than task state.
    z_values = world_xyz[..., 2][
        valid
        & (world_xyz[..., 2] > 0.55)
        & (world_xyz[..., 2] < 0.90)
    ]
    if len(z_values) < 1000:
        raise RuntimeError("有效桌面深度点不足，无法估计桌面高度")

    bin_width = 0.001
    lower = float(np.floor(z_values.min() / bin_width) * bin_width)
    upper = float(np.ceil(z_values.max() / bin_width) * bin_width + bin_width)
    counts, edges = np.histogram(
        z_values,
        bins=np.arange(lower, upper + bin_width, bin_width),
    )
    peak_index = int(np.argmax(counts))
    peak_center = float((edges[peak_index] + edges[peak_index + 1]) / 2)
    plane_values = z_values[np.abs(z_values - peak_center) <= 0.0025]
    if len(plane_values) < 500:
        raise RuntimeError("没有找到足够稳定的桌面平面深度点")
    return float(np.median(plane_values)), int(len(plane_values))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _estimate_one_detection(
    detection: dict[str, Any],
    world_xyz: np.ndarray,
    valid: np.ndarray,
    table_height: float,
) -> dict[str, Any] | None:
    image_height, image_width = valid.shape
    x1_float, y1_float, x2_float, y2_float = detection["bbox_xyxy"]
    x1 = max(0, int(np.floor(x1_float)))
    y1 = max(0, int(np.floor(y1_float)))
    x2 = min(image_width, int(np.ceil(x2_float)))
    y2 = min(image_height, int(np.ceil(y2_float)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop_xyz = world_xyz[y1:y2, x1:x2]
    crop_valid = valid[y1:y2, x1:x2]
    # Remove the visually inferred table plane. This also prevents the square
    # hole in part_A from returning the table position as the object depth.
    elevated = (
        crop_valid
        & (crop_xyz[..., 2] >= table_height + 0.006)
        & (crop_xyz[..., 2] <= table_height + 0.25)
    )
    object_mask = _largest_component(elevated)
    object_points = crop_xyz[object_mask]
    if len(object_points) < 80:
        return None

    surface_center = np.median(object_points, axis=0)
    surface_low = np.percentile(object_points[:, 2], 5)
    surface_high = np.percentile(object_points[:, 2], 95)
    # A centroid over every visible pixel is biased toward the camera-facing
    # vertical face. Use the upper surface outline instead: its robust XY
    # bounds are symmetric around the tabletop support position.
    top_threshold = float(surface_high - 0.006)
    top_points = object_points[object_points[:, 2] >= top_threshold]
    if len(top_points) < 30:
        top_points = object_points[
            object_points[:, 2] >= np.percentile(object_points[:, 2], 75)
        ]
    xy_low = np.percentile(top_points[:, :2], 2, axis=0)
    xy_high = np.percentile(top_points[:, :2], 98, axis=0)
    horizontal_center = (xy_low + xy_high) / 2.0
    top_surface_center = np.median(top_points, axis=0)
    support_position = np.array(
        [horizontal_center[0], horizontal_center[1], table_height],
        dtype=np.float64,
    )
    xy_center_distances = {
        "support_to_top_surface_median_m": float(
            np.linalg.norm(
                support_position[:2] - top_surface_center[:2]
            )
        ),
        "support_to_point_cloud_median_m": float(
            np.linalg.norm(support_position[:2] - surface_center[:2])
        ),
        "top_surface_to_point_cloud_median_m": float(
            np.linalg.norm(top_surface_center[:2] - surface_center[:2])
        ),
    }
    maximum_xy_center_disagreement = max(xy_center_distances.values())
    quality_failures = []
    if len(object_points) < MIN_RELIABLE_OBJECT_DEPTH_PIXELS:
        quality_failures.append("insufficient_valid_object_depth_pixels")
    if (
        maximum_xy_center_disagreement
        > MAX_RELIABLE_XY_CENTER_DISAGREEMENT_M
    ):
        quality_failures.append("inconsistent_xy_centers")
    ys, xs = np.nonzero(object_mask)
    selected_bbox = [
        int(xs.min() + x1),
        int(ys.min() + y1),
        int(xs.max() + x1 + 1),
        int(ys.max() + y1 + 1),
    ]
    return {
        "semantic_label": detection["semantic_label"],
        "confidence": float(detection["confidence"]),
        "detector_bbox_xyxy": detection["bbox_xyxy"],
        "depth_component_bbox_xyxy": selected_bbox,
        "valid_object_depth_pixels": int(len(object_points)),
        "top_surface_depth_pixels": int(len(top_points)),
        "estimated_support_position_world_xyz_m": (
            np.round(support_position, 6).tolist()
        ),
        "visible_surface_median_world_xyz_m": (
            np.round(surface_center, 6).tolist()
        ),
        "top_surface_median_world_xyz_m": (
            np.round(top_surface_center, 6).tolist()
        ),
        "point_cloud_median_world_xyz_m": (
            np.round(surface_center, 6).tolist()
        ),
        "position_quality": {
            "reliable": not quality_failures,
            "failure_reasons": quality_failures,
            "valid_object_depth_pixels": int(len(object_points)),
            "minimum_valid_object_depth_pixels": (
                MIN_RELIABLE_OBJECT_DEPTH_PIXELS
            ),
            "xy_center_distances_m": {
                key: round(value, 6)
                for key, value in xy_center_distances.items()
            },
            "maximum_xy_center_disagreement_m": round(
                maximum_xy_center_disagreement,
                6,
            ),
            "maximum_allowed_xy_center_disagreement_m": (
                MAX_RELIABLE_XY_CENTER_DISAGREEMENT_M
            ),
        },
        "visible_height_range_world_z_m": [
            round(float(surface_low), 6),
            round(float(surface_high), 6),
        ],
        "estimated_footprint_world_xy_m": {
            "min": np.round(xy_low, 6).tolist(),
            "max": np.round(xy_high, 6).tolist(),
        },
        "_support_position": support_position,
    }


def _estimate_centered_wrist_target(
    world_xyz: np.ndarray,
    valid: np.ndarray,
    table_height: float,
    anchor_position: np.ndarray,
    semantic_label: str,
    maximum_anchor_distance: float,
) -> dict[str, Any] | None:
    """Locate an actively centred wrist target directly from aligned depth.

    The detector is trained from the head-camera domain. At wrist range the
    target can fill most of the image and therefore receive no detector box.
    Before the wrist image is captured the robot has already moved the
    head-RGB-D anchor onto the wrist optical axis, so depth components near
    both that axis and the anchor are valid target candidates. This remains a
    visual/proprioceptive estimate and never reads the simulator actor pose.
    """
    image_height, image_width = valid.shape
    elevated = (
        valid
        & (world_xyz[..., 2] >= table_height + 0.006)
        & (world_xyz[..., 2] <= table_height + 0.18)
    )
    labels, component_count = ndimage.label(
        elevated,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return None

    anchor = np.asarray(anchor_position, dtype=np.float64)
    image_center = np.array(
        [image_width / 2.0, image_height / 2.0],
        dtype=np.float64,
    )
    image_scale = np.array(
        [max(image_width, 1), max(image_height, 1)],
        dtype=np.float64,
    )
    candidates: list[tuple[float, dict[str, Any]]] = []

    for component_id in range(1, component_count + 1):
        component_mask = labels == component_id
        ys, xs = np.nonzero(component_mask)
        if len(xs) < 80:
            continue

        pixel_center = np.array(
            [float(np.median(xs)), float(np.median(ys))],
            dtype=np.float64,
        )
        normalized_center_distance = float(
            np.linalg.norm((pixel_center - image_center) / image_scale)
        )
        # The active-view move puts the target near the optical axis. Reject
        # border components such as links, fingers and distant clutter.
        if normalized_center_distance > 0.42:
            continue

        object_points = world_xyz[component_mask]
        surface_low = np.percentile(object_points[:, 2], 5)
        surface_high = np.percentile(object_points[:, 2], 95)
        top_threshold = float(surface_high - 0.006)
        top_points = object_points[object_points[:, 2] >= top_threshold]
        if len(top_points) < 30:
            top_points = object_points[
                object_points[:, 2]
                >= np.percentile(object_points[:, 2], 75)
            ]
        if len(top_points) < 20:
            continue

        xy_low = np.percentile(top_points[:, :2], 2, axis=0)
        xy_high = np.percentile(top_points[:, :2], 98, axis=0)
        horizontal_center = (xy_low + xy_high) / 2.0
        support_position = np.array(
            [horizontal_center[0], horizontal_center[1], table_height],
            dtype=np.float64,
        )
        anchor_distance = float(
            np.linalg.norm(support_position[:2] - anchor[:2])
        )
        if anchor_distance > maximum_anchor_distance:
            continue

        selected_bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ]
        estimate = {
            "semantic_label": semantic_label,
            "confidence": 0.0,
            "detector_bbox_xyxy": selected_bbox,
            "depth_component_bbox_xyxy": selected_bbox,
            "valid_object_depth_pixels": int(len(object_points)),
            "top_surface_depth_pixels": int(len(top_points)),
            "estimated_support_position_world_xyz_m": (
                np.round(support_position, 6).tolist()
            ),
            "visible_surface_median_world_xyz_m": (
                np.round(np.median(object_points, axis=0), 6).tolist()
            ),
            "visible_height_range_world_z_m": [
                round(float(surface_low), 6),
                round(float(surface_high), 6),
            ],
            "_support_position": support_position,
        }
        # Geometry is the identity gate: prefer the component closest to the
        # previous head-camera anchor, with optical-axis proximity as a
        # secondary cue.
        score = anchor_distance + 0.04 * normalized_center_distance
        candidates.append((score, estimate))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _estimate_container(
    detection: dict[str, Any],
    world_xyz: np.ndarray,
    valid: np.ndarray,
    table_height: float,
) -> dict[str, Any] | None:
    """Estimate the box footprint from elevated RGB-D points in its YOLO box."""
    image_height, image_width = valid.shape
    x1_float, y1_float, x2_float, y2_float = detection["bbox_xyxy"]
    x1 = max(0, int(np.floor(x1_float)))
    y1 = max(0, int(np.floor(y1_float)))
    x2 = min(image_width, int(np.ceil(x2_float)))
    y2 = min(image_height, int(np.ceil(y2_float)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop_xyz = world_xyz[y1:y2, x1:x2]
    crop_valid = valid[y1:y2, x1:x2]
    elevated = (
        crop_valid
        & (crop_xyz[..., 2] >= table_height + 0.004)
        & (crop_xyz[..., 2] <= table_height + 0.30)
    )
    points = crop_xyz[elevated]
    if len(points) < 200:
        return None

    xy_low = np.percentile(points[:, :2], 1, axis=0)
    xy_high = np.percentile(points[:, :2], 99, axis=0)
    center = (xy_low + xy_high) / 2.0
    rim_height = float(
        np.clip(
            np.percentile(points[:, 2], 97),
            table_height + 0.025,
            table_height + 0.16,
        )
    )
    return {
        "semantic_label": "box",
        "confidence": float(detection["confidence"]),
        "detector_bbox_xyxy": detection["bbox_xyxy"],
        "estimated_position_world_xyz_m": np.round(
            [center[0], center[1], table_height],
            6,
        ).tolist(),
        "estimated_footprint_world_xy_m": {
            "min": np.round(xy_low, 6).tolist(),
            "max": np.round(xy_high, 6).tolist(),
        },
        "estimated_rim_height_world_z_m": round(rim_height, 6),
        "valid_container_depth_pixels": int(len(points)),
    }


def _inside_visual_container(
    position_xyz: np.ndarray,
    container: dict[str, Any] | None,
    margin: float = 0.015,
) -> bool:
    if container is None:
        return False
    bounds = container["estimated_footprint_world_xy_m"]
    lower = np.asarray(bounds["min"], dtype=float) + margin
    upper = np.asarray(bounds["max"], dtype=float) - margin
    if np.any(lower >= upper):
        return False
    xy = np.asarray(position_xyz, dtype=float)[:2]
    return bool(np.all(xy >= lower) and np.all(xy <= upper))


def _save_runtime_overlay(
    rgb: np.ndarray,
    estimates: list[dict[str, Any]],
    container: dict[str, Any] | None,
    output_path: Path,
) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {
        "part_A": (255, 80, 80),
        "part_B": (255, 190, 40),
        "box": (40, 180, 255),
    }
    rows = [*estimates]
    if container is not None:
        rows.append(container)
    for row in rows:
        x1, y1, x2, y2 = row["detector_bbox_xyxy"]
        label = row["semantic_label"]
        position = row.get(
            "estimated_support_position_world_xyz_m",
            row.get("estimated_position_world_xyz_m"),
        )
        text = (
            f"{label} "
            f"({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})"
        )
        if row.get("inside_container"):
            text += " in_box"
        color = colors[label]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_y = max(0, int(y1) - 16)
        draw.rectangle(
            (int(x1), text_y, min(image.width, int(x1) + 250), text_y + 16),
            fill=color,
        )
        draw.text((int(x1) + 2, text_y + 2), text, fill=(0, 0, 0))
    image.save(output_path)


def run_rgbd_position_inference(
    rgb: np.ndarray,
    position_image: np.ndarray,
    camera_to_world: np.ndarray,
    output_dir: Path,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    """Infer object and box positions without reading simulator actor poses."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_result = run_yolo_world_single_frame(
        rgb,
        output_dir,
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )

    world_xyz, valid = _world_position_map(position_image, camera_to_world)
    table_height, table_pixel_count = _estimate_table_height(world_xyz, valid)
    part_estimates = []
    failed_part_detections = []
    for detection in detector_result["detections"]:
        if detection["semantic_label"] not in PART_CLASSES:
            continue
        estimate = _estimate_one_detection(
            detection,
            world_xyz,
            valid,
            table_height,
        )
        if estimate is None:
            failed_part_detections.append(detection)
        else:
            part_estimates.append(estimate)

    box_detections = [
        detection
        for detection in detector_result["detections"]
        if detection["semantic_label"] == "box"
    ]
    # Prefer the most confident box; break a near tie by its visible area.
    box_detections.sort(
        key=lambda detection: (
            detection["confidence"],
            detection["bbox_xywh"][2] * detection["bbox_xywh"][3],
        ),
        reverse=True,
    )
    container = (
        _estimate_container(
            box_detections[0],
            world_xyz,
            valid,
            table_height,
        )
        if box_detections
        else None
    )

    public_estimates = []
    for estimate in part_estimates:
        public = {
            key: value
            for key, value in estimate.items()
            if not key.startswith("_")
        }
        public["inside_container"] = _inside_visual_container(
            estimate["_support_position"],
            container,
        )
        public_estimates.append(public)

    overlay_path = output_dir / "head_camera_rgbd_runtime_positions.png"
    _save_runtime_overlay(rgb, public_estimates, container, overlay_path)
    payload = {
        "observation_source": (
            "head_camera RGB + YOLO detection + aligned Position/depth"
        ),
        "coordinate_frame": "SAPIEN world frame",
        "privileged_actor_pose_used": False,
        "model": detector_result["model"],
        "table": {
            "estimated_height_world_z_m": round(table_height, 6),
            "supporting_depth_pixels": table_pixel_count,
        },
        "estimated_parts": public_estimates,
        "estimated_container": container,
        "failed_part_detection_count": len(failed_part_detections),
        "image": {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "outputs": {
            "rgb": detector_result["image"]["raw_path"],
            "detector_overlay": detector_result["image"]["annotated_path"],
            "position_overlay": str(overlay_path),
        },
    }
    json_path = output_dir / "runtime_positions.json"
    payload["outputs"]["positions_json"] = str(json_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def run_wrist_target_inference(
    rgb: np.ndarray,
    position_image: np.ndarray,
    camera_to_world: np.ndarray,
    output_dir: Path,
    *,
    camera_name: str,
    semantic_label: str,
    anchor_position_world_xyz: np.ndarray,
    model_path: str,
    confidence: float,
    image_size: int,
    device: str,
    maximum_anchor_distance: float = 0.10,
) -> dict[str, Any]:
    """Refine one target position from wrist RGB-D near a head-camera anchor."""
    if semantic_label not in PART_CLASSES:
        raise ValueError(f"腕部相机不支持目标类别：{semantic_label}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = np.asarray(anchor_position_world_xyz, dtype=np.float64)
    # Keep head-camera detection strict (normally 0.90). Wrist crops have a
    # different scale/domain, so allow lower-confidence proposals and then
    # reject them using calibrated 3-D distance to the head-camera anchor.
    wrist_detector_confidence = min(float(confidence), 0.25)
    detector_result = run_yolo_world_single_frame(
        rgb,
        output_dir,
        model_path=model_path,
        confidence=wrist_detector_confidence,
        image_size=image_size,
        device=device,
        camera_name=camera_name,
    )
    world_xyz, valid = _world_position_map(position_image, camera_to_world)
    candidates = []
    for detection in detector_result["detections"]:
        # The head camera has already established the target identity. At the
        # close wrist viewpoint the same object can be cropped and classified
        # as another known class, so associate detections geometrically with
        # the 3-D anchor instead of requiring the wrist class name to match.
        # The anchor Z came from head-camera RGB-D and represents the table
        # support plane. It lets the wrist view remove the table without any
        # simulator object pose.
        estimate = _estimate_one_detection(
            detection,
            world_xyz,
            valid,
            float(anchor[2]),
        )
        if estimate is None:
            continue
        position = np.asarray(
            estimate["estimated_support_position_world_xyz_m"],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(position[:2] - anchor[:2]))
        candidates.append((distance, estimate, detection["semantic_label"]))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1] if candidates else None
    selected_distance = candidates[0][0] if candidates else None
    selected_detector_label = candidates[0][2] if candidates else None
    selection_source = "yolo_bbox_plus_rgbd" if selected is not None else None
    if (
        selected is not None
        and selected_distance is not None
        and selected_distance > maximum_anchor_distance
    ):
        selected = None

    if selected is None:
        selected = _estimate_centered_wrist_target(
            world_xyz,
            valid,
            float(anchor[2]),
            anchor,
            semantic_label,
            maximum_anchor_distance,
        )
        if selected is not None:
            selected_distance = float(
                np.linalg.norm(
                    np.asarray(
                        selected[
                            "estimated_support_position_world_xyz_m"
                        ],
                        dtype=np.float64,
                    )[:2]
                    - anchor[:2]
                )
            )
            selected_detector_label = None
            selection_source = "active_center_rgbd_component"

    target_position = (
        selected["estimated_support_position_world_xyz_m"]
        if selected is not None
        else None
    )
    payload = {
        "observation_source": (
            f"{camera_name} RGB + YOLO + aligned Position/depth"
        ),
        "camera": camera_name,
        "semantic_label": semantic_label,
        "selected_detector_label": (
            selected_detector_label if selected is not None else None
        ),
        "selection_source": selection_source,
        "head_detector_confidence_threshold": float(confidence),
        "wrist_detector_confidence_threshold": (
            wrist_detector_confidence
        ),
        "anchor_position_world_xyz_m": np.round(anchor, 6).tolist(),
        "target_position_world_xyz_m": target_position,
        "target_surface_position_world_xyz_m": (
            selected["visible_surface_median_world_xyz_m"]
            if selected is not None
            else None
        ),
        "anchor_to_target_xy_distance_m": (
            round(float(selected_distance), 6)
            if selected is not None
            else None
        ),
        "candidate_count": len(candidates),
        "maximum_anchor_distance_m": maximum_anchor_distance,
        "privileged_actor_pose_used": False,
        "detector_output": detector_result["image"],
    }
    json_path = output_dir / f"{camera_name}_target_position.json"
    payload["output_json"] = str(json_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _match_with_privileged_truth(
    estimates: list[dict[str, Any]],
    privileged_actors: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    comparisons: list[dict[str, Any]] = []
    missing_actor_names: list[str] = []
    matched_estimate_indices: set[int] = set()

    for semantic_label in PART_CLASSES:
        estimate_indices = [
            index
            for index, estimate in enumerate(estimates)
            if estimate["semantic_label"] == semantic_label
        ]
        actors = privileged_actors.get(semantic_label, [])
        if not estimate_indices:
            missing_actor_names.extend(actor.get_name() for actor in actors)
            continue

        estimated_xy = np.asarray(
            [
                estimates[index]["_support_position"][:2]
                for index in estimate_indices
            ]
        )
        truth_xyz = np.asarray(
            [np.asarray(actor.get_pose().p, dtype=float) for actor in actors]
        )
        costs = np.linalg.norm(
            estimated_xy[:, None, :] - truth_xyz[None, :, :2],
            axis=2,
        )
        estimate_rows, actor_columns = linear_sum_assignment(costs)
        assigned_actor_columns = set()
        for estimate_row, actor_column in zip(
            estimate_rows,
            actor_columns,
        ):
            estimate_index = estimate_indices[int(estimate_row)]
            actor = actors[int(actor_column)]
            estimate_position = estimates[estimate_index]["_support_position"]
            truth_position = np.asarray(actor.get_pose().p, dtype=float)
            error = estimate_position - truth_position
            comparisons.append(
                {
                    "semantic_label": semantic_label,
                    "matched_actor_name": actor.get_name(),
                    "confidence": estimates[estimate_index]["confidence"],
                    "estimated_position_world_xyz_m": (
                        np.round(estimate_position, 6).tolist()
                    ),
                    "privileged_position_world_xyz_m": (
                        np.round(truth_position, 6).tolist()
                    ),
                    "error_xyz_m": np.round(error, 6).tolist(),
                    "horizontal_error_m": round(
                        float(np.linalg.norm(error[:2])),
                        6,
                    ),
                    "position_error_3d_m": round(
                        float(np.linalg.norm(error)),
                        6,
                    ),
                    "valid_object_depth_pixels": estimates[estimate_index][
                        "valid_object_depth_pixels"
                    ],
                    "detector_bbox_xyxy": estimates[estimate_index][
                        "detector_bbox_xyxy"
                    ],
                }
            )
            matched_estimate_indices.add(estimate_index)
            assigned_actor_columns.add(int(actor_column))

        missing_actor_names.extend(
            actor.get_name()
            for index, actor in enumerate(actors)
            if index not in assigned_actor_columns
        )

    unmatched_estimate_indices = [
        index
        for index in range(len(estimates))
        if index not in matched_estimate_indices
    ]
    comparisons.sort(
        key=lambda item: (
            item["semantic_label"],
            item["privileged_position_world_xyz_m"][0],
        )
    )
    return comparisons, missing_actor_names, unmatched_estimate_indices


def _metrics(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return {
            "matched_count": 0,
            "mean_horizontal_error_m": None,
            "median_horizontal_error_m": None,
            "max_horizontal_error_m": None,
            "mean_position_error_3d_m": None,
        }
    horizontal = np.asarray(
        [item["horizontal_error_m"] for item in comparisons],
        dtype=float,
    )
    error_3d = np.asarray(
        [item["position_error_3d_m"] for item in comparisons],
        dtype=float,
    )
    return {
        "matched_count": len(comparisons),
        "mean_horizontal_error_m": round(float(horizontal.mean()), 6),
        "median_horizontal_error_m": round(float(np.median(horizontal)), 6),
        "max_horizontal_error_m": round(float(horizontal.max()), 6),
        "mean_position_error_3d_m": round(float(error_3d.mean()), 6),
        "within_10mm_horizontal": int(np.sum(horizontal <= 0.010)),
        "within_20mm_horizontal": int(np.sum(horizontal <= 0.020)),
    }


def _save_depth_image(
    depth_m: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
) -> None:
    valid_depth = depth_m[valid & np.isfinite(depth_m) & (depth_m > 0)]
    if len(valid_depth) == 0:
        raise RuntimeError("head_camera没有有效深度")
    near, far = np.percentile(valid_depth, [2, 98])
    normalized = np.clip((depth_m - near) / max(far - near, 1e-6), 0, 1)
    visualization = ((1.0 - normalized) * 255).astype(np.uint8)
    visualization[~valid] = 0
    Image.fromarray(visualization, mode="L").save(output_path)


def _save_position_overlay(
    rgb: np.ndarray,
    comparisons: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {"part_A": (255, 80, 80), "part_B": (255, 190, 40)}
    for comparison in comparisons:
        x1, y1, x2, y2 = comparison["detector_bbox_xyxy"]
        color = colors[comparison["semantic_label"]]
        estimated = comparison["estimated_position_world_xyz_m"]
        error_mm = comparison["horizontal_error_m"] * 1000
        text = (
            f'{comparison["matched_actor_name"]} '
            f'({estimated[0]:.3f},{estimated[1]:.3f},{estimated[2]:.3f}) '
            f"errXY={error_mm:.1f}mm"
        )
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_y = max(0, int(y1) - 16)
        draw.rectangle(
            (int(x1), text_y, min(image.width, int(x1) + 300), text_y + 16),
            fill=color,
        )
        draw.text((int(x1) + 2, text_y + 2), text, fill=(0, 0, 0))
    image.save(output_path)


def run_rgbd_position_evaluation(
    rgb: np.ndarray,
    position_image: np.ndarray,
    camera_to_world: np.ndarray,
    privileged_actors: dict[str, list[Any]],
    output_dir: Path,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    """Estimate first, then use privileged poses only for evaluation."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_result = run_yolo_world_single_frame(
        rgb,
        output_dir,
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )

    world_xyz, valid = _world_position_map(position_image, camera_to_world)
    table_height, table_pixel_count = _estimate_table_height(world_xyz, valid)
    depth_m = -np.asarray(position_image[..., 2], dtype=np.float64)
    depth_npy_path = output_dir / "head_camera_depth_m.npy"
    depth_png_path = output_dir / "head_camera_depth.png"
    np.save(depth_npy_path, depth_m)
    _save_depth_image(depth_m, valid, depth_png_path)

    estimates = []
    failed_depth_detections = []
    for detection in detector_result["detections"]:
        if detection["semantic_label"] not in PART_CLASSES:
            continue
        estimate = _estimate_one_detection(
            detection,
            world_xyz,
            valid,
            table_height,
        )
        if estimate is None:
            failed_depth_detections.append(detection)
        else:
            estimates.append(estimate)

    comparisons, missing_actors, unmatched_indices = (
        _match_with_privileged_truth(estimates, privileged_actors)
    )
    metrics = _metrics(comparisons)
    overlay_path = output_dir / "head_camera_rgbd_positions.png"
    _save_position_overlay(rgb, comparisons, overlay_path)

    public_estimates = [
        {
            key: value
            for key, value in estimate.items()
            if not key.startswith("_")
        }
        for estimate in estimates
    ]
    payload = {
        "method": {
            "runtime_inputs": [
                "head_camera RGB",
                "head_camera Position/depth",
                "head_camera camera-to-world matrix",
            ],
            "position_estimation": (
                "YOLO box -> remove visually estimated table plane -> "
                "largest elevated depth component -> robust upper-surface "
                "XY bounds center; "
                "table support Z"
            ),
            "coordinate_frame": "SAPIEN world frame",
            "privileged_state_policy": (
                "Actor poses are read only after RGB-D estimation and are "
                "used solely for offline accuracy evaluation."
            ),
        },
        "model": detector_result["model"],
        "table": {
            "estimated_height_world_z_m": round(table_height, 6),
            "supporting_depth_pixels": table_pixel_count,
        },
        "estimated_parts": public_estimates,
        "comparisons": comparisons,
        "metrics": metrics,
        "missing_privileged_actors": missing_actors,
        "failed_depth_detection_count": len(failed_depth_detections),
        "unmatched_estimate_indices": unmatched_indices,
        "outputs": {
            "rgb": detector_result["image"]["raw_path"],
            "detector_overlay": detector_result["image"]["annotated_path"],
            "position_overlay": str(overlay_path),
            "depth_m_npy": str(depth_npy_path),
            "depth_visualization": str(depth_png_path),
        },
    }
    json_path = output_dir / "position_comparison.json"
    payload["outputs"]["comparison_json"] = str(json_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
