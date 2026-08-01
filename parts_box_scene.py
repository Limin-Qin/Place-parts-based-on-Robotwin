"""Standalone RoboTwin scene: a dual-arm mobile robot, three parts A, and a box.

This file deliberately lives outside envs/ and does not register or modify an
official RoboTwin benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sapien.core as sapien
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs._base_task import Base_Task
from envs.utils import ArmTag, create_actor
from envs.utils.actor_utils import Actor
from examples.my_parts_box_scene.agent import (
    AgentGoal,
    AgentPlan,
    AgentPlanner,
    PlanExecutionError,
    PlanExecutor,
    RobotSkills,
    SkillResult,
    load_and_validate_plan,
)
from examples.my_parts_box_scene.perception import (
    generate_multicamera_supplement,
    generate_yolo_dataset,
    run_rgbd_position_evaluation,
    run_rgbd_position_inference,
    run_rgbd_robustness_suite,
    run_wrist_target_inference,
    run_yolo_world_single_frame,
)


LOCAL_CONFIG = Path(__file__).with_name("scene_config.yml")
GENERATED_PLAN_PATH = LOCAL_CONFIG.with_name("generated_plan.json")
EXECUTION_TRACE_PATH = LOCAL_CONFIG.with_name("agent_execution_trace.json")
SESSION_VIDEO_PATH = LOCAL_CONFIG.with_name("agent_execution.mp4")
SESSION_VIDEO_SEGMENT_DIR = LOCAL_CONFIG.with_name(
    ".agent_execution_segments"
)
HEAD_CAMERA_FOVY_DEGREES = 62.0
HEAD_CAMERA_CENTER_X_METERS = 0.0
WEB_WORKER_EVENT_PREFIX = "__ROBOTWIN_WORKER_EVENT__"


class AgentTaskCancelled(Exception):
    """Raised inside dense actions when the Web user requests a stop."""


class NoTrustedVisualTargets(Exception):
    """End a task safely when head-camera detections are not graspable."""


class PartsBoxScene(Base_Task):
    """Scene-only task assembled from existing RoboTwin assets."""

    def setup_demo(self, **kwargs):
        self.scene_only = kwargs.pop("scene_only", False)
        self.video_process = None
        self.head_camera = None
        self.agent_vision_enabled = False
        self.agent_vision_config: dict = {}
        self.agent_visual_object_positions: dict[str, np.ndarray] = {}
        self.agent_visual_object_surface_positions: dict[str, np.ndarray] = {}
        self.agent_visual_object_geometry: dict[str, dict] = {}
        self._agent_visual_track_positions: dict[str, np.ndarray] = {}
        self._agent_visual_next_instance_id = {
            "part_A": 1,
            "part_B": 1,
        }
        self._agent_goal_targets_locked = False
        self._agent_commanded_placed_names: list[str] = []
        self._agent_pending_placed_names: list[str] = []
        self._agent_vision_observation_index = 0
        self._agent_wrist_observation_index = 0
        self._agent_initial_head_quality_checked = False
        self._agent_wide_observation_arms: set[str] = set()
        self._latest_agent_visual_state: dict | None = None
        self._agent_preplanned_drop_xy: dict[str, np.ndarray] = {}
        self._agent_recorded_occupied_drop_regions: set[int] = set()
        self._agent_drop_region_centers_xy: np.ndarray | None = None
        self._agent_latest_drop_region_state: dict = {}
        self.agent_skill_parameter_registry: dict[str, dict] = {}
        # Closed-loop rounds share one executor. A normal round now completes
        # pick-through-retreat; shared state is retained only for safe recovery
        # if a physical transaction stops after the gripper has closed.
        self._agent_plan_executor: PlanExecutor | None = None
        web_stream_text = os.getenv("ROBOTWIN_WEB_STREAM_DIR", "").strip()
        self.web_stream_dir = (
            Path(web_stream_text).expanduser().resolve()
            if web_stream_text
            else None
        )
        web_stop_text = os.getenv("ROBOTWIN_WEB_STOP_FILE", "").strip()
        self.web_stop_request_path = (
            Path(web_stop_text).expanduser().resolve()
            if web_stop_text
            else None
        )
        self._web_stream_sequence = 0
        self._web_stream_last_publish = 0.0
        self._publish_unrecorded_motion = False
        super()._init_task_env_(**kwargs)
        # The base uses 32 ray-tracing samples. Four are sufficient for an
        # operation video and make offscreen recording much faster.
        sapien.render.set_ray_tracing_samples_per_pixel(4)
        camera_by_name = dict(
            zip(self.cameras.static_camera_name, self.cameras.static_camera_list)
        )
        self.head_camera = camera_by_name["head_camera"]
        # Widen the official embodiment head camera itself. RGB, Position/depth
        # and calibration matrices are consequently produced by this same
        # camera projection; no auxiliary observation camera is created.
        self.head_camera.set_fovy(
            np.deg2rad(HEAD_CAMERA_FOVY_DEGREES),
            True,
        )
        # The stock aloha-agilex camera is mounted 3.2 cm left of the robot
        # centre line. This standalone scene uses a centred operator view so
        # the two arms have symmetric framing in the web console.
        head_camera_pose = (
            self.head_camera.entity.get_pose().to_transformation_matrix()
        )
        head_camera_pose[0, 3] = HEAD_CAMERA_CENTER_X_METERS
        self.head_camera.entity.set_pose(sapien.Pose(head_camera_pose))
        head_camera_index = self.cameras.static_camera_name.index(
            "head_camera"
        )
        self.cameras.static_camera_config[head_camera_index] = dict(
            self.cameras.static_camera_config[head_camera_index]
        )
        self.cameras.static_camera_config[head_camera_index][
            "fovy"
        ] = HEAD_CAMERA_FOVY_DEGREES

    def load_robot(self, **kwargs):
        """Skip motion planners only for preview/check mode."""
        if not self.scene_only:
            return super().load_robot(**kwargs)

        from envs.robot import Robot

        self.robot = Robot(self.scene, need_topp=False, **kwargs)
        self.robot.communication_flag = False
        self.robot.init_joints()
        for link in self.robot.left_entity.get_links():
            link.set_mass(1)
        for link in self.robot.right_entity.get_links():
            link.set_mass(1)

    def together_open_gripper(self, save_freq=None, **kwargs):
        if self.scene_only:
            return None
        return super().together_open_gripper(save_freq=save_freq, **kwargs)

    def load_actors(self):
        # Keep the requested light-blue official model, but enlarge only its
        # two horizontal mesh axes at runtime. Official assets stay untouched.
        self.empty_box = self._create_roomier_plastic_box()

        seed_text = os.getenv("MY_PARTS_BOX_SEED")
        random_generator = np.random.default_rng(
            int(seed_text) if seed_text is not None else None
        )
        random_a_xy = self._sample_random_tabletop_positions(
            random_generator,
            count=3,
        )
        random_a_quaternions = [
            self._sample_uniform_quaternion(random_generator)
            for _ in range(3)
        ]
        random_b_xy = self._sample_random_tabletop_positions(
            random_generator,
            count=2,
            existing_positions=random_a_xy,
        )
        random_b_quaternions = [
            self._sample_uniform_quaternion(random_generator)
            for _ in range(2)
        ]
        spawn_height = 0.86

        # Three official fluted-block assets are used as mechanical parts A.
        self.parts_a = []
        for index in range(3):
            pose = sapien.Pose(
                [*random_a_xy[index], spawn_height],
                random_a_quaternions[index],
            )
            part = self._create_scaled_mechanical_part(
                model_id=0,
                pose=pose,
            )
            # This legacy official asset uses older config key names. Normalize
            # them only on this actor instance so grasp_actor can read them.
            if "contact_points_pose" not in part.config:
                part.config["contact_points_pose"] = part.config["contact_pose"]
            if "transform_matrix" not in part.config:
                part.config["transform_matrix"] = part.config["trans_matrix"]
            part.config.setdefault("functional_matrix", [])
            part.config.setdefault("orientation_point", [])
            part.set_name(f"part_A_{index + 1}")
            part.set_mass(0.05)
            self.parts_a.append(part)

        # Two visually distinct official square modules are semantic parts B.
        self.parts_b = []
        for index in range(2):
            part = create_actor(
                scene=self,
                pose=sapien.Pose(
                    [
                        *random_b_xy[index],
                        spawn_height + self.table_z_bias,
                    ],
                    random_b_quaternions[index],
                ),
                modelname="055_small-speaker",
                model_id=1,
                convex=True,
            )
            part.set_name(f"part_B_{index + 1}")
            part.set_mass(0.05)
            self.parts_b.append(part)

    @staticmethod
    def _upright_part_quaternion(yaw_degrees: float) -> np.ndarray:
        """Rotate the asset's stable upright pose around world Z."""
        half_yaw = np.deg2rad(float(yaw_degrees)) / 2.0
        yaw = np.asarray(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=float,
        )
        base = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=float)
        w1, x1, y1, z1 = yaw
        w2, x2, y2, z2 = base
        result = np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
        return result / np.linalg.norm(result)

    @staticmethod
    def _sample_uniform_quaternion(
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample a uniformly distributed 3-D rotation in wxyz order."""
        u1, u2, u3 = random_generator.random(3)
        x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        return np.asarray([w, x, y, z], dtype=float)

    @staticmethod
    def _sample_random_tabletop_positions(
        random_generator: np.random.Generator,
        *,
        count: int,
        existing_positions=(),
    ) -> list[tuple[float, float]]:
        """Sample the whole usable table without semantic placement bands."""
        reserved = [
            np.asarray(position, dtype=float)
            for position in existing_positions
        ]
        positions: list[np.ndarray] = []
        # Keep one shared, continuous sampling region for both categories.
        # Its only task constraint is physical manipulability: points beyond
        # this envelope can be seen and reached at a 10 cm pregrasp pose, but
        # the AgileX arm cannot complete the final contact approach. There is
        # still no semantic left/right band, fixed row, or preferred arm.
        x_limits = (-0.47, 0.47)
        y_limits = (-0.25, 0.18)
        minimum_part_clearance = 0.125

        for _ in range(10_000):
            candidate = np.asarray(
                [
                    random_generator.uniform(*x_limits),
                    random_generator.uniform(*y_limits),
                ],
                dtype=float,
            )

            # The static box already occupies this physical footprint. This
            # exclusion prevents actors spawning through its collision mesh;
            # it is not a semantic position or reachability constraint.
            inside_box_footprint = (
                abs(float(candidate[0])) < 0.19
                and -0.31 < float(candidate[1]) < 0.055
            )
            if inside_box_footprint:
                continue
            if any(
                np.linalg.norm(candidate - existing)
                < minimum_part_clearance
                for existing in (*reserved, *positions)
            ):
                continue
            positions.append(candidate)
            if len(positions) == count:
                return [
                    (float(position[0]), float(position[1]))
                    for position in positions
                ]

        raise RuntimeError(
            f"无法在桌面上生成{count}个互不穿透的随机零件位置"
        )

    def _create_scaled_mechanical_part(
        self,
        model_id: int,
        pose: sapien.Pose,
    ) -> Actor:
        """Build an 80%-scale part with matching visual, collision and labels."""
        model_dir = REPO_ROOT / "assets" / "objects" / "004_fluted-block"
        with (model_dir / f"model_data{model_id}.json").open(
            "r", encoding="utf-8"
        ) as file:
            model_data = json.load(file)

        mesh_scale = (
            np.asarray(model_data["scale"], dtype=float) * 0.80
        )
        # Actor.get_contact_point multiplies annotations by this scale, so the
        # grasp poses stay aligned with the resized collision and visual mesh.
        model_data["scale"] = mesh_scale.tolist()

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_multiple_convex_collisions_from_file(
            filename=str(model_dir / "collision" / f"base{model_id}.glb"),
            scale=mesh_scale,
        )
        builder.add_visual_from_file(
            filename=str(model_dir / "visual" / f"base{model_id}.glb"),
            scale=mesh_scale,
        )
        entity = builder.build(name="004_fluted-block")
        entity.set_name("004_fluted-block")
        entity.set_pose(
            sapien.Pose(
                [pose.p[0], pose.p[1], pose.p[2] + self.table_z_bias],
                pose.q,
            )
        )
        return Actor(entity, model_data)

    def _create_roomier_plastic_box(self) -> Actor:
        """Build a horizontally enlarged copy of official plasticbox/base3."""
        model_dir = REPO_ROOT / "assets" / "objects" / "062_plasticbox"
        with (model_dir / "model_data3.json").open(
            "r", encoding="utf-8"
        ) as file:
            model_data = json.load(file)

        original_scale = np.asarray(model_data["scale"], dtype=float)
        mesh_scale = original_scale * np.array([1.55, 1.0, 1.55])
        model_data["scale"] = mesh_scale.tolist()

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_multiple_convex_collisions_from_file(
            filename=str(model_dir / "collision" / "base3.glb"),
            scale=mesh_scale,
        )
        builder.add_visual_from_file(
            filename=str(model_dir / "visual" / "base3.glb"),
            scale=mesh_scale,
        )
        entity = builder.build(name="062_plasticbox")
        entity.set_name("062_plasticbox")
        entity.set_pose(
            sapien.Pose(
                [0.0, -0.14, 0.74 + self.table_z_bias],
                [0.5, 0.5, 0.5, 0.5],
            )
        )
        return Actor(entity, model_data)

    def play_once(self):
        """Sequentially compose basic skills for each part."""
        skills = RobotSkills(self)

        for index, part in enumerate(self.parts_a, start=1):
            print(f"\n零件 {index}/{len(self.parts_a)}：开始顺序执行基础技能")

            pick_result = self._require_skill(skills.pick(part))
            arm = pick_result.data["arm"]
            self._require_skill(skills.lift(arm, distance=0.10))
            self._require_skill(skills.place_in(part, self.empty_box, arm))
            self._require_skill(skills.retreat(arm, distance=0.08))

        self.info["info"] = {
            "{A}": "004_fluted-block/base0",
            "{B}": "004_fluted-block/base0",
            "{C}": "004_fluted-block/base0",
            "{D}": "062_plasticbox/base3",
        }
        return self.info

    def execute_agent_plan(self, plan):
        """Execute a validated JSON plan through the basic skill library."""
        if self._agent_plan_executor is None:
            self._agent_plan_executor = PlanExecutor(self)
        report = self._agent_plan_executor.execute(plan)
        if report.success and self.agent_vision_enabled:
            self._record_commanded_placements(plan)
        self.info["info"] = {
            "{A}": "004_fluted-block/base0",
            "{B}": "004_fluted-block/base0",
            "{C}": "004_fluted-block/base0",
            "{D}": "062_plasticbox/base3",
            "{E}": "055_small-speaker/base1",
            "{F}": "055_small-speaker/base1",
        }
        return report

    def _agent_held_objects(self) -> dict[str, str]:
        """Return arm-to-visual-object identity from gripper skill state."""
        if self._agent_plan_executor is None:
            return {}
        held = self._agent_plan_executor.skills.held_objects
        return {
            arm: actor.get_name()
            for arm, actor in held.items()
        }

    def _agent_actor_map(self) -> dict:
        """Build a legacy physical map without assuming instance counts."""
        return {
            actor.get_name(): actor
            for actor in (*self.parts_a, *self.parts_b)
        }

    def configure_agent_vision(
        self,
        *,
        model_path: Path,
        confidence: float = 0.60,
        image_size: int = 640,
        device: str = "0",
    ) -> None:
        """Enable RGB-D observations as the Agent's object-position source."""
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Agent视觉权重不存在：{model_path}")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("Agent视觉置信度必须在 (0, 1] 范围内")
        self.agent_vision_config = {
            "model_path": str(model_path),
            "confidence": float(confidence),
            "image_size": int(image_size),
            "device": str(device),
            "output_dir": LOCAL_CONFIG.parent / "agent_vision_observations",
        }
        self._agent_initial_head_quality_checked = False
        self.agent_vision_enabled = True

    def get_agent_visual_position(self, object_name: str) -> np.ndarray:
        """Return the latest RGB-D position, never the actor's simulator pose."""
        try:
            return self.agent_visual_object_positions[object_name].copy()
        except KeyError as exc:
            raise RuntimeError(
                f"最新head_camera观测中没有 {object_name} 的可靠RGB-D位置，"
                "为避免使用特权坐标，已拒绝执行抓取"
            ) from exc

    def get_agent_visual_top_down_quaternion(
        self,
        object_name: str,
    ) -> np.ndarray:
        """Return a category-independent robot top-down convention."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC

        if not AgentPlanner.is_visual_instance_name(object_name):
            raise ValueError(f"未知视觉零件实例：{object_name}")
        return np.asarray(
            GRASP_DIRECTION_DIC["top_down"],
            dtype=float,
        ).copy()

    def get_agent_visual_object_template(self, object_name: str):
        """Return category geometry without assuming a fixed instance pool."""
        if object_name.startswith("part_A_"):
            return self.parts_a[0]
        if object_name.startswith("part_B_"):
            return self.parts_b[0]
        raise ValueError(f"未知视觉零件实例：{object_name}")

    def get_agent_visual_surface_position(
        self,
        object_name: str,
    ) -> np.ndarray:
        """Return a visible target point for active wrist-camera centering."""
        return self.agent_visual_object_surface_positions.get(
            object_name,
            self.get_agent_visual_position(object_name),
        ).copy()

    def _capture_agent_rgbd(self) -> dict:
        self._agent_vision_observation_index += 1
        self._update_render()
        cameras = dict(
            zip(
                self.cameras.static_camera_name,
                self.cameras.static_camera_list,
            )
        )
        head_camera = cameras["head_camera"]
        head_camera.take_picture()
        rgba = head_camera.get_picture("Color")
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        position_image = head_camera.get_picture("Position")
        camera_to_world = head_camera.get_model_matrix()
        output_dir = (
            self.agent_vision_config["output_dir"]
            / f"observation_{self._agent_vision_observation_index:03d}"
        )
        return run_rgbd_position_inference(
            rgb,
            position_image,
            camera_to_world,
            output_dir,
            model_path=self.agent_vision_config["model_path"],
            confidence=self.agent_vision_config["confidence"],
            image_size=self.agent_vision_config["image_size"],
            device=self.agent_vision_config["device"],
        )

    def observe_agent_target_from_wrist(
        self,
        object_name: str,
        arm: Any,
        anchor_position: np.ndarray,
        *,
        center_before_capture: bool = True,
    ) -> dict:
        """Refine a head-camera target using the selected wrist RGB-D camera."""
        arm_name = str(arm)
        if arm_name not in {"left", "right"}:
            raise ValueError(f"未知腕部相机机械臂：{arm_name}")
        if not self.cameras.collect_wrist_camera:
            raise RuntimeError("当前机器人配置没有启用腕部相机")

        if center_before_capture:
            self._center_wrist_camera_on_target(object_name, arm_name)
        self._agent_wrist_observation_index += 1
        self._update_render()
        camera = (
            self.cameras.left_camera
            if arm_name == "left"
            else self.cameras.right_camera
        )
        camera_name = f"{arm_name}_camera"
        camera.take_picture()
        rgba = camera.get_picture("Color")
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        position_image = camera.get_picture("Position")
        camera_to_world = camera.get_model_matrix()
        semantic_label = (
            "part_A" if object_name.startswith("part_A_") else "part_B"
        )
        output_dir = (
            LOCAL_CONFIG.parent
            / "agent_wrist_observations"
            / f"observation_{self._agent_wrist_observation_index:03d}"
            / object_name
        )
        return run_wrist_target_inference(
            rgb,
            position_image,
            camera_to_world,
            output_dir,
            camera_name=camera_name,
            semantic_label=semantic_label,
            anchor_position_world_xyz=np.asarray(
                anchor_position,
                dtype=float,
            ),
            model_path=self.agent_vision_config["model_path"],
            confidence=self.agent_vision_config["confidence"],
            image_size=self.agent_vision_config["image_size"],
            device=self.agent_vision_config["device"],
        )

    def _center_wrist_camera_on_target(
        self,
        object_name: str,
        arm_name: str,
        tolerance: float = 0.012,
        maximum_step_translation: float = 0.012,
    ) -> None:
        """Apply at most one gentle wrist-camera centring translation."""
        self._update_render()
        camera = (
            self.cameras.left_camera
            if arm_name == "left"
            else self.cameras.right_camera
        )
        camera_to_world = np.asarray(
            camera.get_model_matrix(),
            dtype=float,
        )
        target_world = self.get_agent_visual_surface_position(object_name)
        target_camera = (
            np.linalg.inv(camera_to_world)
            @ np.concatenate((target_world, [1.0]))
        )
        # Translating the camera by its local X/Y target coordinates moves the
        # target onto the optical axis. The magnitude comes directly from the
        # calibrated camera geometry rather than a fixed correction step.
        camera_translation = np.array(
            [target_camera[0], target_camera[1], 0.0],
            dtype=float,
        )
        world_translation = (
            camera_to_world[:3, :3] @ camera_translation
        )
        translation_norm = float(np.linalg.norm(world_translation))
        if translation_norm <= tolerance:
            return
        # Exact optical-axis centring is unnecessary for the gripper and made
        # the wrist visibly oscillate. Limit this observation move to one
        # small step; the subsequent RGB-D measurement supplies the remaining
        # bounded correction directly to the grasp descent.
        if translation_norm > maximum_step_translation:
            world_translation *= (
                maximum_step_translation / translation_norm
            )

        current_pose = np.asarray(
            (
                self.robot.get_left_ee_pose()
                if arm_name == "left"
                else self.robot.get_right_ee_pose()
            ),
            dtype=float,
        )
        target_pose = current_pose.copy()
        target_pose[:3] += world_translation
        success = self.move(
            self.move_to_pose(
                arm_tag=ArmTag(arm_name),
                target_pose=target_pose,
            )
        )
        if not success:
            raise RuntimeError(f"{arm_name}腕部相机未能移动到目标观察姿态")

    def _new_visual_instance_name(self, category: str) -> str:
        """Allocate an unbounded runtime ID from actual visual detections."""
        try:
            instance_id = self._agent_visual_next_instance_id[category]
        except KeyError as exc:
            raise ValueError(f"未知视觉类别：{category}") from exc
        self._agent_visual_next_instance_id[category] = instance_id + 1
        return f"{category}_{instance_id}"

    def _reset_agent_task_visual_bindings(self) -> None:
        """Start a Web task with fresh visual IDs but unchanged physics."""
        self._agent_visual_track_positions.clear()
        self._agent_visual_next_instance_id = {
            "part_A": 1,
            "part_B": 1,
        }
        self.agent_visual_object_positions.clear()
        self.agent_visual_object_surface_positions.clear()
        self.agent_visual_object_geometry.clear()
        self._agent_commanded_placed_names.clear()
        self._agent_pending_placed_names.clear()
        self._agent_preplanned_drop_xy.clear()
        self.agent_skill_parameter_registry.clear()
        self._agent_goal_targets_locked = False

    def _rebind_locked_goal_detections(
        self,
        goal: AgentGoal,
        assigned: list[dict],
    ) -> None:
        """Reuse missing task IDs instead of creating post-wide targets."""
        if not self._agent_goal_targets_locked or not goal.target_objects:
            return

        protected_names = set(self._agent_commanded_placed_names)
        present_names = {str(item.get("name")) for item in assigned}
        missing_names = [
            name
            for name in goal.target_objects
            if name not in present_names and name not in protected_names
        ]
        candidates = [
            item
            for item in assigned
            if (
                item.get("semantic_label") == goal.target_category
                and item.get("name") not in goal.target_objects
                and not item.get("inside_container", False)
            )
        ]
        if not missing_names or not candidates:
            return

        from scipy.optimize import linear_sum_assignment

        cost_matrix = np.zeros(
            (len(missing_names), len(candidates)),
            dtype=float,
        )
        for row, name in enumerate(missing_names):
            previous = self._agent_visual_track_positions.get(name)
            for column, detection in enumerate(candidates):
                current = np.asarray(
                    detection["estimated_support_position_world_xyz_m"],
                    dtype=float,
                )
                cost_matrix[row, column] = (
                    float(np.linalg.norm(current[:2] - previous[:2]))
                    if previous is not None
                    else 0.0
                )

        rows, columns = linear_sum_assignment(cost_matrix)
        for row, column in zip(rows.tolist(), columns.tolist()):
            new_name = missing_names[row]
            detection = candidates[column]
            temporary_name = str(detection["name"])
            detection["name"] = new_name

            for cache in (
                self.agent_visual_object_positions,
                self.agent_visual_object_surface_positions,
                self.agent_visual_object_geometry,
            ):
                cache.pop(new_name, None)
                if temporary_name in cache:
                    cache[new_name] = cache.pop(temporary_name)

            current = np.asarray(
                detection["estimated_support_position_world_xyz_m"],
                dtype=float,
            )
            self._agent_visual_track_positions[new_name] = current.copy()
            self._agent_visual_track_positions.pop(temporary_name, None)

    def _assign_visual_instance_names(
        self,
        inference: dict,
    ) -> list[dict]:
        """Track anonymous detections using visual positions and dynamic IDs."""
        assigned_states: list[dict] = []
        previous_positions = {
            name: np.asarray(position, dtype=float).copy()
            for name, position in self._agent_visual_track_positions.items()
        }
        self.agent_visual_object_positions = {}
        self.agent_visual_object_surface_positions = {}
        held_by_arm = self._agent_held_objects()

        for category in ("part_A", "part_B"):
            detections = [
                dict(item)
                for item in inference["estimated_parts"]
                if item["semantic_label"] == category
            ]
            used_names: set[str] = set()

            # A grasped object can move far beyond the normal frame-to-frame
            # tracking radius. Associate a same-category detection near the
            # corresponding end effector before allocating any new ID. This
            # uses only RGB-D and robot proprioception.
            for arm_name, held_name in held_by_arm.items():
                if not held_name.startswith(f"{category}_") or not detections:
                    continue
                ee_position = np.asarray(
                    (
                        self.robot.get_left_ee_pose()
                        if arm_name == "left"
                        else self.robot.get_right_ee_pose()
                    ),
                    dtype=float,
                )[:2]
                distances = [
                    float(
                        np.linalg.norm(
                            np.asarray(
                                detection[
                                    "estimated_support_position_world_xyz_m"
                                ],
                                dtype=float,
                            )[:2]
                            - ee_position
                        )
                    )
                    for detection in detections
                ]
                nearest_index = int(np.argmin(distances))
                if distances[nearest_index] > 0.16:
                    continue
                detection = detections.pop(nearest_index)
                detection["name"] = held_name
                used_names.add(held_name)
                assigned_states.append(detection)

            inside = sorted(
                (
                    item
                    for item in detections
                    if item.get("inside_container", False)
                ),
                key=lambda item: item[
                    "estimated_support_position_world_xyz_m"
                ][0],
            )
            outside = sorted(
                (
                    item
                    for item in detections
                    if not item.get("inside_container", False)
                ),
                key=lambda item: item[
                    "estimated_support_position_world_xyz_m"
                ][0],
            )
            commanded_history = [
                name
                for name in self._agent_commanded_placed_names
                if name.startswith(f"{category}_")
            ]
            pending = [
                name
                for name in self._agent_pending_placed_names
                if name.startswith(f"{category}_")
            ]
            # The detection visible after a successful place belongs to the
            # current transaction before it is offered to older placements.
            # Otherwise one crowded-box detection is repeatedly consumed by
            # the oldest historical object ID.
            commanded = [
                *pending,
                *(
                    name
                    for name in commanded_history
                    if name not in pending
                ),
            ]

            # A successful skill call tells the tracker which identity moved;
            # the following visual observation still has to confirm that an
            # object is actually visible inside the box.
            unassigned_commanded = [
                name for name in commanded
                if name not in used_names
            ]
            for name, detection in zip(unassigned_commanded, inside):
                detection["name"] = name
                used_names.add(name)
                assigned_states.append(detection)

            unassigned_detections = [
                *inside[len(unassigned_commanded):],
                *outside,
            ]
            remaining_previous_names = [
                name
                for name in previous_positions
                if (
                    name.startswith(f"{category}_")
                    and name not in used_names
                )
            ]

            # Match all remaining identities jointly. This preserves the
            # stable objects while allowing a commanded object to keep its ID
            # even when a failed grasp pushes it much farther than the normal
            # frame-to-frame tracking radius.
            matched_detection_indices: set[int] = set()
            if remaining_previous_names and unassigned_detections:
                from scipy.optimize import linear_sum_assignment

                cost_matrix = np.zeros(
                    (
                        len(remaining_previous_names),
                        len(unassigned_detections),
                    ),
                    dtype=float,
                )
                for row, name in enumerate(remaining_previous_names):
                    previous = previous_positions[name]
                    for column, detection in enumerate(
                        unassigned_detections
                    ):
                        current = np.asarray(
                            detection[
                                "estimated_support_position_world_xyz_m"
                            ],
                            dtype=float,
                        )
                        cost_matrix[row, column] = np.linalg.norm(
                            current[:2] - previous[:2]
                        )

                rows, columns = linear_sum_assignment(cost_matrix)
                commanded_set = set(commanded)
                for row, column in zip(rows.tolist(), columns.tolist()):
                    name = remaining_previous_names[row]
                    distance = float(cost_matrix[row, column])
                    if distance > 0.12 and name not in commanded_set:
                        continue
                    detection = unassigned_detections[column]
                    detection["name"] = name
                    used_names.add(name)
                    assigned_states.append(detection)
                    matched_detection_indices.add(column)

            # Any unmatched detection is a newly discovered runtime instance.
            # IDs are allocated on demand and therefore have no 3-A/2-B cap.
            for detection_index, detection in enumerate(
                unassigned_detections
            ):
                if detection_index in matched_detection_indices:
                    continue
                name = self._new_visual_instance_name(category)
                detection["name"] = name
                used_names.add(name)
                assigned_states.append(detection)

        assigned_states.sort(key=lambda item: item["name"])
        for state in assigned_states:
            position = np.asarray(
                state["estimated_support_position_world_xyz_m"],
                dtype=float,
            )
            self._agent_visual_track_positions[state["name"]] = (
                position.copy()
            )
            position_reliable = bool(
                state.get("position_quality", {}).get("reliable", False)
            )
            if position_reliable:
                self.agent_visual_object_positions[state["name"]] = (
                    position.copy()
                )
                self.agent_visual_object_surface_positions[state["name"]] = (
                    np.asarray(
                        state["visible_surface_median_world_xyz_m"],
                        dtype=float,
                    )
                )
            self.agent_visual_object_geometry[state["name"]] = {
                "position_xyz": position.copy().tolist(),
                "footprint_world_xy": state.get(
                    "estimated_footprint_world_xy_m"
                ),
                "visible_height_range_world_z": state.get(
                    "visible_height_range_world_z_m"
                ),
                "source": "head_camera_rgbd",
            }
        return assigned_states

    def _record_commanded_placements(self, plan) -> None:
        current_placements: list[str] = []
        for step in plan.steps:
            arguments = step.get("arguments", {})
            names: tuple[str, ...] = ()
            if (
                step["skill"] == "place_in"
                and arguments.get("container") == "box"
            ):
                names = (arguments["object"],)
            elif (
                step["skill"] == "place_in_dual"
                and arguments.get("container") == "box"
            ):
                names = (
                    arguments["left_object"],
                    arguments["right_object"],
                )
            for name in names:
                if name not in current_placements:
                    current_placements.append(name)
                if name not in self._agent_commanded_placed_names:
                    self._agent_commanded_placed_names.append(name)
        self._agent_pending_placed_names = current_placements

    @staticmethod
    def _require_skill(result: SkillResult) -> SkillResult:
        status = "成功" if result.success else "失败"
        print(f"[{status}] {result.skill}: {result.message}")
        if not result.success:
            raise RuntimeError(f"基础技能执行失败：{result.skill}；{result.message}")
        return result

    def check_success(self):
        part_positions = [part.get_pose().p.copy() for part in self.parts_a]
        print(
            "最终特权坐标：",
            f"box={self.empty_box.get_pose().p.tolist()}, ",
            f"parts={[position.tolist() for position in part_positions]}",
        )
        return all(
            self._is_part_placed(part)
            for part in self.parts_a
        ) and self.is_left_gripper_open() and self.is_right_gripper_open()

    def check_plan_success(self, plan):
        """Check only the objects that this Agent plan asked to place."""
        placed_names = set()
        for step in plan.steps:
            arguments = step["arguments"]
            if (
                step["skill"] == "place_in"
                and arguments.get("container") == "box"
            ):
                placed_names.add(arguments["object"])
            elif (
                step["skill"] == "place_in_dual"
                and arguments.get("container") == "box"
            ):
                placed_names.update(
                    (arguments["left_object"], arguments["right_object"])
                )
        actors = self._agent_actor_map()
        if not placed_names:
            return bool(self.plan_success)

        return bool(
            self.plan_success
            and all(self._is_part_placed(actors[name]) for name in placed_names)
        )

    def _is_part_placed(self, part) -> bool:
        """Test box membership without requiring a predetermined final pose."""
        box_matrix = self.empty_box.get_pose().to_transformation_matrix()
        world_position = np.asarray(part.get_pose().p, dtype=float)
        local_position = box_matrix[:3, :3].T @ (
            world_position - box_matrix[:3, 3]
        )

        scale = np.asarray(self.empty_box.config["scale"], dtype=float)
        center = np.asarray(self.empty_box.config["center"], dtype=float) * scale
        half_extents = (
            np.asarray(self.empty_box.config["extents"], dtype=float)
            * scale
            / 2.0
        )
        inside_horizontal = bool(
            center[0] - half_extents[0] + 0.005
            <= local_position[0]
            <= center[0] + half_extents[0] - 0.005
            and center[2] - half_extents[2] + 0.005
            <= local_position[2]
            <= center[2] + half_extents[2] - 0.005
        )
        # Local Y is vertical for this box. A part may protrude above the rim,
        # but its center must remain above the floor and over the box interior.
        floor = center[1] - half_extents[1]
        rim = center[1] + half_extents[1]
        inside_vertical = bool(
            floor - 0.01 <= local_position[1] <= rim + 0.12
        )
        return inside_horizontal and inside_vertical

    def container_drop_pose_candidates(
        self,
        actor,
        container,
        arm: str,
        extra_occupied_positions=None,
        maximum_candidates: int = 8,
        preserve_quaternion=None,
        visual_object_name: str | None = None,
    ):
        """Rank six fixed box regions for visual placement trials."""
        target_name = visual_object_name or actor.get_name()
        if self.agent_vision_enabled and self._latest_agent_visual_state:
            container_state = self._latest_agent_visual_state["container"]
            occupied_positions = [
                np.asarray(position, dtype=float)
                for position in container_state.get(
                    "visual_occupied_positions_xyz", []
                )
            ]
        else:
            occupied_positions = [
                np.asarray(part.get_pose().p, dtype=float)
                for part in (*self.parts_a, *self.parts_b)
                if part is not actor and self._is_part_placed(part)
            ]
        occupied_positions.extend(
            np.asarray(position, dtype=float)
            for position in (extra_occupied_positions or [])
        )
        if self.agent_vision_enabled and self._latest_agent_visual_state:
            footprint = container_state.get("footprint_world_xy")
            if not isinstance(footprint, dict):
                raise RuntimeError("最新RGB-D观察缺少盒子可见轮廓")
            lower = np.asarray(footprint.get("min"), dtype=float)
            upper = np.asarray(footprint.get("max"), dtype=float)
            if (
                lower.shape != (2,)
                or upper.shape != (2,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(upper <= lower)
            ):
                raise RuntimeError("RGB-D估计的盒子可见轮廓无效")

            # Rebuild the same six relative regions from the currently
            # visible box footprint. Region IDs stay stable across Web tasks
            # even when the measured footprint moves by a few pixels.
            span = upper - lower
            target_state = next(
                (
                    item
                    for item in self._latest_agent_visual_state["objects"]
                    if item["name"] == target_name
                ),
                None,
            )
            if target_state is None:
                raise RuntimeError(
                    f"最新RGB-D观察缺少 {target_name} 的视觉尺寸"
                )
            cached_geometry = self.agent_visual_object_geometry.get(
                target_name,
                {},
            )
            object_footprint = target_state.get(
                "footprint_world_xy"
            ) or cached_geometry.get("footprint_world_xy")
            if not isinstance(object_footprint, dict):
                raise RuntimeError(
                    f"最新RGB-D观察缺少 {target_name} 的视觉轮廓"
                )
            object_lower = np.asarray(
                object_footprint.get("min"),
                dtype=float,
            )
            object_upper = np.asarray(
                object_footprint.get("max"),
                dtype=float,
            )
            if (
                object_lower.shape != (2,)
                or object_upper.shape != (2,)
                or not np.all(np.isfinite(object_lower))
                or not np.all(np.isfinite(object_upper))
                or np.any(object_upper <= object_lower)
            ):
                raise RuntimeError(
                    f"RGB-D估计的 {target_name} 视觉轮廓无效"
                )
            placing_half_extents_xy = 0.5 * (
                object_upper - object_lower
            )

            height_range = target_state.get(
                "visible_height_range_world_z"
            ) or cached_geometry.get("visible_height_range_world_z")
            object_position = np.asarray(
                (
                    target_state.get("position_xyz")
                    or cached_geometry.get("position_xyz")
                ),
                dtype=float,
            )
            if (
                not isinstance(height_range, (list, tuple))
                or len(height_range) != 2
                or object_position.shape != (3,)
                or not np.all(np.isfinite(height_range))
                or not np.all(np.isfinite(object_position))
            ):
                raise RuntimeError(
                    f"最新RGB-D观察缺少 {target_name} 的视觉高度"
                )
            measured_height = float(height_range[1]) - float(
                object_position[2]
            )
            if not 0.005 <= measured_height <= 0.25:
                raise RuntimeError(
                    f"RGB-D估计的 {target_name} 视觉高度无效："
                    f"{measured_height:.3f}米"
                )

            visual_wall_inset = placing_half_extents_xy + 0.005
            region_lower = lower + np.maximum(
                0.10 * span,
                visual_wall_inset,
            )
            region_upper = upper - np.maximum(
                0.10 * span,
                visual_wall_inset,
            )
            if np.any(region_upper <= region_lower):
                raise RuntimeError(
                    "根据盒子RGB-D视觉轮廓，盒内没有可生成六区域的空间"
                )
            table_z = float(container_state["position_xyz"][2])
            # Before grasping, orientation is deliberately unknown. The
            # identity value only keeps the candidate's transport format at
            # seven numbers and is replaced by the proprioceptively estimated
            # held orientation in RobotSkills.place_in(). After grasping,
            # preserve_quaternion already is that held orientation.
            quaternion = np.asarray(
                (
                    preserve_quaternion
                    if preserve_quaternion is not None
                    else [1.0, 0.0, 0.0, 0.0]
                ),
                dtype=float,
            )
            # This is a supported placement pose, not a free-fall release.
            # Lower the object's bottom to roughly 2 mm above the RGB-D
            # support plane before opening; this tiny clearance avoids
            # commanding interpenetration while producing only negligible
            # physical settling.
            placement_z = table_z + 0.5 * measured_height + 0.002
            x_values = region_lower[0] + (
                region_upper[0] - region_lower[0]
            ) * np.asarray([1.0 / 6.0, 0.5, 5.0 / 6.0])
            y_values = region_lower[1] + (
                region_upper[1] - region_lower[1]
            ) * np.asarray([0.25, 0.75])
            region_centers = np.asarray(
                [
                    [x_value, y_value]
                    for y_value in y_values
                    for x_value in x_values
                ],
                dtype=float,
            )
            self._agent_drop_region_centers_xy = region_centers.copy()

            visual_occupied_regions = {
                int(
                    np.argmin(
                        np.linalg.norm(
                            region_centers - occupied[:2], axis=1
                        )
                    )
                )
                + 1
                for occupied in occupied_positions
                if occupied.shape == (3,) and np.all(np.isfinite(occupied))
            }
            recorded_occupied_regions = {
                region_id
                for region_id in self._agent_recorded_occupied_drop_regions
                if 1 <= region_id <= len(region_centers)
            }
            if len(visual_occupied_regions) > len(
                recorded_occupied_regions
            ):
                effective_occupied_regions = visual_occupied_regions
                occupied_source = "vision"
            else:
                effective_occupied_regions = recorded_occupied_regions
                occupied_source = "agent_record"

            region_candidates = {
                region_id: np.concatenate(
                    (
                        np.asarray(
                            [center[0], center[1], placement_z]
                        ),
                        quaternion,
                    )
                )
                for region_id, center in enumerate(
                    region_centers, start=1
                )
                if region_id not in effective_occupied_regions
            }

            def region_score(region_id: int) -> tuple[float, float]:
                position = region_centers[region_id - 1]
                if effective_occupied_regions:
                    minimum_occupied_distance = min(
                        float(
                            np.linalg.norm(
                                position
                                - region_centers[occupied_id - 1]
                            )
                        )
                        for occupied_id in effective_occupied_regions
                    )
                else:
                    minimum_occupied_distance = 0.0
                arm_tiebreak = (
                    -float(position[0])
                    if str(arm) == "left"
                    else float(position[0])
                )
                return minimum_occupied_distance, 0.001 * arm_tiebreak

            ranked_region_ids = sorted(
                region_candidates,
                key=region_score,
                reverse=True,
            )
            preplanned_xy = self._agent_preplanned_drop_xy.get(target_name)
            if preplanned_xy is not None:
                ranked_region_ids.sort(
                    key=lambda region_id: float(
                        np.linalg.norm(
                            region_centers[region_id - 1] - preplanned_xy
                        )
                    )
                )

            self._agent_latest_drop_region_state = {
                "regions_xy": np.round(region_centers, 6).tolist(),
                "visual_occupied_regions": sorted(
                    visual_occupied_regions
                ),
                "recorded_occupied_regions": sorted(
                    recorded_occupied_regions
                ),
                "effective_occupied_regions": sorted(
                    effective_occupied_regions
                ),
                "effective_occupied_source": occupied_source,
                "ranked_available_regions": ranked_region_ids.copy(),
                "selected_region": (
                    ranked_region_ids[0] if ranked_region_ids else None
                ),
            }
            return [
                region_candidates[region_id].tolist()
                for region_id in ranked_region_ids[:maximum_candidates]
            ]
        else:
            # Legacy non-visual execution keeps the official asset interface.
            candidates = [
                np.asarray(
                    container.get_functional_point(index),
                    dtype=float,
                )
                for index in range(
                    len(container.config["functional_matrix"])
                )
            ]

        def candidate_score(candidate):
            position = candidate[:3]
            if occupied_positions:
                clearance = min(
                    np.linalg.norm(position[:2] - occupied[:2])
                    for occupied in occupied_positions
                )
            else:
                clearance = 1.0
            arm_bias = -position[0] if str(arm) == "left" else position[0]
            return clearance + 0.01 * arm_bias

        ranked = sorted(candidates, key=candidate_score, reverse=True)
        return [candidate.tolist() for candidate in ranked]

    def record_agent_drop_region(
        self,
        object_name: str,
        placed_position_xyz,
    ) -> int | None:
        """Record the nearest one of six regions after a successful place."""
        del object_name
        if self._agent_drop_region_centers_xy is None:
            return None
        position = np.asarray(placed_position_xyz, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None
        region_id = int(
            np.argmin(
                np.linalg.norm(
                    self._agent_drop_region_centers_xy - position[:2],
                    axis=1,
                )
            )
        ) + 1
        self._agent_recorded_occupied_drop_regions.add(region_id)
        return region_id

    def choose_container_drop_pose(
        self,
        actor,
        container,
        arm: str,
        extra_occupied_positions=None,
    ):
        """Compatibility wrapper returning the best current visual pose."""
        candidates = self.container_drop_pose_candidates(
            actor,
            container,
            arm,
            extra_occupied_positions=extra_occupied_positions,
            maximum_candidates=1,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _initial_unreliable_head_positions(
        goal: AgentGoal,
        inference: dict,
    ) -> list[dict]:
        """Describe unreliable requested-part positions in the initial frame."""
        image_width = float(inference.get("image", {}).get("width", 640))
        unreliable = []
        for item in inference.get("estimated_parts", []):
            quality = item.get("position_quality", {})
            if (
                item.get("semantic_label") != goal.target_category
                or item.get("inside_container", False)
                or quality.get("reliable", False)
            ):
                continue
            bbox = item.get("detector_bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            bbox_center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            unreliable.append(
                {
                    "suspected_occluding_arm": (
                        "left"
                        if bbox_center_x < image_width / 2.0
                        else "right"
                    ),
                    "detector_bbox_xyxy": bbox,
                    "failure_reasons": quality.get(
                        "failure_reasons",
                        [],
                    ),
                    "valid_object_depth_pixels": quality.get(
                        "valid_object_depth_pixels",
                    ),
                    "maximum_xy_center_disagreement_m": quality.get(
                        "maximum_xy_center_disagreement_m",
                    ),
                }
            )
        return unreliable

    @staticmethod
    def _requested_detection_quality_groups(
        goal: AgentGoal,
        inference: dict,
    ) -> tuple[list[dict], list[dict]]:
        """Split requested detections into safe and unsafe grasp inputs."""
        trusted: list[dict] = []
        untrusted: list[dict] = []
        for index, item in enumerate(inference.get("estimated_parts", []), 1):
            if (
                item.get("semantic_label") != goal.target_category
                or item.get("inside_container", False)
            ):
                continue
            quality = item.get("position_quality", {})
            record = {
                "candidate": f"{goal.target_category}_candidate_{index}",
                "confidence": round(float(item.get("confidence", 0.0)), 4),
                "position_xyz": item.get(
                    "estimated_support_position_world_xyz_m"
                ),
                "bbox_xyxy": item.get("detector_bbox_xyxy"),
            }
            if quality.get("reliable", False):
                trusted.append(record)
            else:
                record["failure_reasons"] = quality.get(
                    "failure_reasons", []
                )
                untrusted.append(record)
        return trusted, untrusted

    def _move_arm_to_wide_observation_pose(self, arm_name: str) -> None:
        """Move one free arm outwards to clear its half of the head view."""
        if arm_name not in {"left", "right"}:
            raise ValueError(f"未知机械臂：{arm_name}")
        if arm_name in self._agent_held_objects():
            raise RuntimeError(
                f"{arm_name}机械臂正在持物，不能移动到初始宽展观察姿态"
            )

        home_pose = np.asarray(
            (
                self.robot.left_original_pose
                if arm_name == "left"
                else self.robot.right_original_pose
            ),
            dtype=float,
        )
        observation_pose = home_pose.copy()
        observation_pose[0] += -0.10 if arm_name == "left" else 0.10
        observation_pose[1] -= 0.06
        observation_pose[2] -= 0.03

        self.plan_success = True
        moved = self.move(
            self.move_to_pose(
                arm_tag=ArmTag(arm_name),
                target_pose=observation_pose,
            )
        )
        if not moved or not self.plan_success:
            raise RuntimeError(
                f"{arm_name}机械臂未能到达宽展观察姿态"
            )
        self._agent_wide_observation_arms.add(arm_name)

    def _clear_head_camera_view(self) -> list[str]:
        """Move only non-home arms away when they occlude visual verification."""
        moved_arms: list[str] = []
        held_arms = set(self._agent_held_objects())
        for arm_name in ("left", "right"):
            # Returning a loaded arm home can collide with the other arm or
            # lose the grasp. A held object is handled by the next place phase.
            if arm_name in held_arms:
                continue
            current = np.asarray(
                (
                    self.robot.get_left_ee_pose()
                    if arm_name == "left"
                    else self.robot.get_right_ee_pose()
                ),
                dtype=float,
            )
            home = np.asarray(
                (
                    self.robot.left_original_pose
                    if arm_name == "left"
                    else self.robot.right_original_pose
                ),
                dtype=float,
            )
            if np.linalg.norm(current[:3] - home[:3]) < 0.04:
                continue
            success = self.move(
                self.back_to_origin(arm_tag=ArmTag(arm_name))
            )
            success = bool(success and self.plan_success)
            if not success:
                raise RuntimeError(
                    f"head_camera视野受遮挡，且{arm_name}机械臂未能移出观察区域"
                )
            self._agent_wide_observation_arms.discard(arm_name)
            moved_arms.append(arm_name)
        return moved_arms

    def recover_after_failed_agent_grasp(self) -> None:
        """Recover free arms while preserving any cross-phase held objects."""
        left = ArmTag("left")
        right = ArmTag("right")
        held_arms = set(self._agent_held_objects())
        # A CuRobo planning failure sets this flag even when no failed target
        # motion was executed. Clear it before issuing explicit safe recovery.
        self.plan_success = True
        free_arms = [
            arm for arm in (left, right)
            if str(arm) not in held_arms
        ]
        if free_arms:
            opened = self.move(
                *[
                    self.open_gripper(arm_tag=arm)
                    for arm in free_arms
                ]
            )
            if not opened:
                raise RuntimeError("失败恢复时未能安全张开空闲夹爪")

        for arm in (left, right):
            if str(arm) in held_arms:
                continue
            current = np.asarray(
                (
                    self.robot.get_left_ee_pose()
                    if str(arm) == "left"
                    else self.robot.get_right_ee_pose()
                ),
                dtype=float,
            )
            home = np.asarray(
                (
                    self.robot.left_original_pose
                    if str(arm) == "left"
                    else self.robot.right_original_pose
                ),
                dtype=float,
            )
            if np.linalg.norm(current[:3] - home[:3]) < 0.04:
                continue
            self.plan_success = True
            returned = self.move(self.back_to_origin(arm_tag=arm))
            if not returned:
                raise RuntimeError(
                    f"抓取失败后{arm}机械臂未能返回安全观察姿态"
                )
        self.plan_success = True

    def raise_if_web_stop_requested(self) -> None:
        """Interrupt the current Agent action after consuming one stop request."""
        request_path = self.web_stop_request_path
        if request_path is None or not request_path.is_file():
            return
        try:
            request_path.unlink()
        except FileNotFoundError:
            return
        raise AgentTaskCancelled("用户已停止当前任务")

    def return_arms_home_after_web_stop(self) -> bool:
        """Release any tracked grasp and return both arms to their home poses."""
        left = ArmTag("left")
        right = ArmTag("right")
        self.plan_success = True

        opened = self.move(
            self.open_gripper(arm_tag=left),
            self.open_gripper(arm_tag=right),
        )
        if self._agent_plan_executor is not None:
            skills = self._agent_plan_executor.skills
            skills.held_objects.clear()
            skills.held_object_in_ee.clear()

        success = bool(opened and self.plan_success)
        for arm in (left, right):
            current = np.asarray(
                (
                    self.robot.get_left_ee_pose()
                    if str(arm) == "left"
                    else self.robot.get_right_ee_pose()
                ),
                dtype=float,
            )
            home = np.asarray(
                (
                    self.robot.left_original_pose
                    if str(arm) == "left"
                    else self.robot.right_original_pose
                ),
                dtype=float,
            )
            if np.linalg.norm(current[:3] - home[:3]) < 0.04:
                continue
            self.plan_success = True
            returned = self.move(self.back_to_origin(arm_tag=arm))
            success = bool(success and returned and self.plan_success)

        self.plan_success = True
        self._latest_agent_visual_state = None
        self._agent_wide_observation_arms.clear()
        self._agent_initial_head_quality_checked = False
        self.publish_web_camera_frames(force=True)
        return success

    def return_arms_home_before_new_task(self) -> bool:
        """Return only displaced, empty arms home before a fresh observation."""
        if self._agent_held_objects():
            return False

        success = True
        self._publish_unrecorded_motion = True
        try:
            for arm in (ArmTag("left"), ArmTag("right")):
                current = np.asarray(
                    (
                        self.robot.get_left_ee_pose()
                        if str(arm) == "left"
                        else self.robot.get_right_ee_pose()
                    ),
                    dtype=float,
                )
                home = np.asarray(
                    (
                        self.robot.left_original_pose
                        if str(arm) == "left"
                        else self.robot.right_original_pose
                    ),
                    dtype=float,
                )
                # An arm already at home needs no motion command. Treating an
                # empty motion as failure made the first task fail immediately.
                if np.linalg.norm(current[:3] - home[:3]) < 0.04:
                    continue
                self.plan_success = True
                # back_to_origin produces a CuRobo trajectory. move executes
                # every trajectory point through physics; it never resets the
                # scene or teleports the articulation to its home pose.
                returned = self.move(self.back_to_origin(arm_tag=arm))
                success = bool(success and returned and self.plan_success)
        finally:
            self._publish_unrecorded_motion = False

        self.plan_success = True
        self._latest_agent_visual_state = None
        self._reset_agent_task_visual_bindings()
        self._agent_wide_observation_arms.clear()
        # Each user command needs an independent initial RGB-D quality check.
        # Otherwise a completed command leaves this flag set and the next
        # command skips its required wide-observation retry.
        self._agent_initial_head_quality_checked = False
        self.publish_web_camera_frames(force=True)
        return success

    def _decode_visual_agent_observation(
        self,
        goal: AgentGoal,
        inference: dict,
    ) -> tuple[dict | None, list[dict], dict[str, dict], list[str], list[str], list[str]]:
        """Interpret one RGB-D frame without falling back to actor poses."""
        container = inference["estimated_container"]
        if container is None:
            return None, [], {}, [], list(goal.target_objects), list(
                goal.target_objects
            )
        assigned = self._assign_visual_instance_names(inference)
        self._rebind_locked_goal_detections(goal, assigned)
        assigned_by_name = {
            item["name"]: item
            for item in assigned
        }
        self._bind_goal_targets_from_visual_detections(goal, assigned)
        if not goal.target_objects:
            missing_category = (
                f"{goal.target_category}"
                "（当前head_camera尚未检测到实例）"
            )
            return (
                container,
                assigned,
                assigned_by_name,
                [],
                [],
                [missing_category],
            )
        # Whether the user's goal is already satisfied is a visual/geometry
        # fact, not an action-history fact.  Requiring a preceding
        # ``place_in`` command caused a newly revealed (or duplicate) YOLO
        # detection inside the box to receive a new runtime ID and then be
        # treated as a fresh tabletop target.  That made the robot reach back
        # into the box for an object it had just placed.  Any requested
        # instance visibly inside the container is therefore complete,
        # including objects that started there or were assigned a new dynamic
        # ID after an occlusion.
        completed = [
            name
            for name in goal.target_objects
            if (
                name in assigned_by_name
                and assigned_by_name[name].get("inside_container", False)
            )
        ]
        # Grasp inputs remain restricted to reliable RGB-D estimates, but a
        # post-place completion check needs only visual evidence that the
        # just-placed category is inside the box. Crowded objects often retain
        # a valid box-membership estimate while failing grasp-grade depth
        # consistency. Consume each such detection at most once and associate
        # it with the current placement before any historical object ID.
        untrusted_inside = [
            item
            for item in inference.get("estimated_parts", [])
            if (
                item.get("inside_container", False)
                and not item.get("position_quality", {}).get(
                    "reliable",
                    False,
                )
            )
        ]
        for name in self._agent_pending_placed_names:
            if name not in goal.target_objects or name in completed:
                continue
            matching_index = next(
                (
                    index
                    for index, item in enumerate(untrusted_inside)
                    if name.startswith(f"{item.get('semantic_label')}_")
                ),
                None,
            )
            if matching_index is None:
                continue
            untrusted_inside.pop(matching_index)
            completed.append(name)
        remaining = [
            name
            for name in goal.target_objects
            if name not in completed
        ]
        missing_targets = [
            name
            for name in remaining
            if name not in assigned_by_name
        ]
        return (
            container,
            assigned,
            assigned_by_name,
            completed,
            remaining,
            missing_targets,
        )

    def _bind_goal_targets_from_visual_detections(
        self,
        goal: AgentGoal,
        assigned: list[dict],
    ) -> None:
        """Bind category-only language goals to YOLO-discovered instances."""
        if self._agent_goal_targets_locked:
            return
        # A new detection that is already inside the destination does not
        # require manipulation and must not enlarge the pending task. Existing
        # target IDs remain eligible here so their inside/outside transition
        # can still be tracked and confirmed.
        category_detections = [
            item
            for item in assigned
            if (
                item.get("semantic_label") == goal.target_category
                and (
                    item["name"] in goal.target_objects
                    or not item.get("inside_container", False)
                )
            )
        ]
        if not category_detections:
            return

        selector = goal.target_selector
        if selector == "all":
            discovered_names = [
                item["name"]
                for item in sorted(
                    category_detections,
                    key=lambda item: item[
                        "estimated_support_position_world_xyz_m"
                    ][0],
                )
            ]
        else:
            if selector == "left":
                selected = min(
                    category_detections,
                    key=lambda item: item[
                        "estimated_support_position_world_xyz_m"
                    ][0],
                )
            elif selector == "right":
                selected = max(
                    category_detections,
                    key=lambda item: item[
                        "estimated_support_position_world_xyz_m"
                    ][0],
                )
            else:
                selected = min(
                    category_detections,
                    key=lambda item: abs(
                        item[
                            "estimated_support_position_world_xyz_m"
                        ][0]
                    ),
                )
            discovered_names = [selected["name"]]

        goal.target_objects = discovered_names
        if goal.target_objects:
            self._agent_goal_targets_locked = True

    def _observe_visual_agent_state(self, goal: AgentGoal) -> dict:
        """Observe with RGB-D, clearing arm occlusion once when necessary."""
        held_by_arm = self._agent_held_objects()
        held_by_name = {
            object_name: arm
            for arm, object_name in held_by_arm.items()
        }
        inference = self._capture_agent_rgbd()
        if not self._agent_initial_head_quality_checked:
            self._agent_initial_head_quality_checked = True
        (
            container,
            assigned,
            assigned_by_name,
            completed,
            remaining,
            missing_targets,
        ) = self._decode_visual_agent_observation(goal, inference)
        # While either arm is loaded, the next legal phase is placement of one
        # held object. Other tabletop targets may be occluded by those arms and
        # are intentionally rediscovered only after both hands become free.
        missing_targets = (
            []
            if held_by_arm
            else [
                name for name in missing_targets
                if name not in held_by_name
            ]
        )

        if container is None or missing_targets:
            moved_arms = self._clear_head_camera_view()
            if moved_arms:
                print(
                    "视觉确认受到机械臂遮挡，已将"
                    f"{'、'.join(moved_arms)}臂移出视野并重新观察。",
                    flush=True,
                )
                inference = self._capture_agent_rgbd()
                (
                    container,
                    assigned,
                    assigned_by_name,
                    completed,
                    remaining,
                    missing_targets,
                ) = self._decode_visual_agent_observation(goal, inference)
                missing_targets = (
                    []
                    if held_by_arm
                    else [
                        name for name in missing_targets
                        if name not in held_by_name
                    ]
                )

        if container is None:
            raise RuntimeError(
                "机械臂移出视野后，head_camera仍未能从RGB和深度中定位box；"
                "为避免回退到特权坐标，Agent已停止"
            )
        if missing_targets:
            raise RuntimeError(
                "机械臂移出视野后，head_camera仍未可靠定位以下待确认目标："
                f"{missing_targets}。为避免错误抓取，未使用仿真器特权坐标补全"
            )

        left_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        right_pose = np.asarray(self.robot.get_right_ee_pose(), dtype=float)
        left_home = np.asarray(self.robot.left_original_pose, dtype=float)
        right_home = np.asarray(self.robot.right_original_pose, dtype=float)
        left_at_home = bool(
            np.linalg.norm(left_pose[:3] - left_home[:3]) < 0.04
        )
        right_at_home = bool(
            np.linalg.norm(right_pose[:3] - right_home[:3]) < 0.04
        )
        arms_distance = float(
            np.linalg.norm(left_pose[:3] - right_pose[:3])
        )

        object_states = []
        for detected in assigned:
            name = detected["name"]
            rough_position = detected[
                "estimated_support_position_world_xyz_m"
            ]
            quality = detected.get("position_quality", {})
            position_reliable = bool(quality.get("reliable", False))
            position = rough_position if position_reliable else None
            object_states.append(
                {
                    "name": name,
                    "position_xyz": position,
                    "rough_position_xyz": rough_position,
                    "position_reliable": position_reliable,
                    "position_failure_reasons": quality.get(
                        "failure_reasons", []
                    ),
                    "surface_position_xyz": detected[
                        "visible_surface_median_world_xyz_m"
                    ],
                    "footprint_world_xy": detected.get(
                        "estimated_footprint_world_xy_m"
                    ),
                    "visible_height_range_world_z": detected.get(
                        "visible_height_range_world_z_m"
                    ),
                    "position_source": "head_camera_rgbd",
                    "confidence": round(
                        float(detected["confidence"]),
                        4,
                    ),
                    "detected": True,
                    "placed_in_box": name in completed,
                    "inside_container": bool(
                        detected.get("inside_container", False)
                    ),
                    "recommended_arm": (
                        (
                            goal.requested_arm
                            if goal.requested_arm in {"left", "right"}
                            else self._recommended_arm_from_position(
                                rough_position
                            )
                        )
                        if rough_position is not None
                        else None
                    ),
                    "arm_workspace": (
                        self._arm_workspace_from_position(rough_position)
                        if rough_position is not None
                        else None
                    ),
                    "held_by": held_by_name.get(name),
                }
            )

        # A gripper can occlude the object it is holding. Its identity and
        # ownership come from gripper proprioception and the successful pick
        # skill, so it remains available to the Agent without reading an actor
        # pose from the simulator.
        visible_names = {item["name"] for item in object_states}
        ee_positions = {
            "left": left_pose[:3].tolist(),
            "right": right_pose[:3].tolist(),
        }
        for object_name, arm_name in held_by_name.items():
            if object_name in visible_names:
                continue
            object_states.append(
                {
                    "name": object_name,
                    "position_xyz": ee_positions[arm_name],
                    "surface_position_xyz": None,
                    "footprint_world_xy": None,
                    "position_source": "gripper_proprioception",
                    "confidence": None,
                    "detected": False,
                    "placed_in_box": False,
                    "recommended_arm": arm_name,
                    "arm_workspace": arm_name,
                    "held_by": arm_name,
                }
            )

        state = {
            "observation_source": (
                "head_camera RGB + YOLO + aligned depth; "
                "robot proprioception"
            ),
            "privileged_object_pose_used": False,
            "objects": object_states,
            "container": {
                "name": "box",
                "position_xyz": container[
                    "estimated_position_world_xyz_m"
                ],
                "position_source": "head_camera_rgbd",
                "confidence": round(
                    float(container["confidence"]),
                    4,
                ),
                "footprint_world_xy": container[
                    "estimated_footprint_world_xy_m"
                ],
                "rim_height_world_z": container.get(
                    "estimated_rim_height_world_z_m"
                ),
                "placement_policy": "runtime_free_space",
                "visual_occupied_positions_xyz": [
                    item["estimated_support_position_world_xyz_m"]
                    for item in inference.get("estimated_parts", [])
                    if (
                        item.get("inside_container", False)
                        and item.get(
                            "estimated_support_position_world_xyz_m"
                        )
                        is not None
                    )
                ],
            },
            "robot": {
                "left_end_effector_xyz": np.round(
                    left_pose[:3],
                    4,
                ).tolist(),
                "right_end_effector_xyz": np.round(
                    right_pose[:3],
                    4,
                ).tolist(),
                "left_arm_at_home": left_at_home,
                "right_arm_at_home": right_at_home,
                "left_gripper_open": bool(self.is_left_gripper_open()),
                "right_gripper_open": bool(self.is_right_gripper_open()),
                "wide_observation_arms": sorted(
                    self._agent_wide_observation_arms
                ),
                "end_effector_distance": round(arms_distance, 4),
                "state_source": "robot_proprioception",
                "held_objects": held_by_arm,
            },
            "safety": {
                "arms_too_close": arms_distance < 0.18,
                "workspace_clearance_recommended": bool(
                    remaining
                    and not held_by_arm
                    and (not left_at_home or not right_at_home)
                ),
                "instruction": (
                    "workspace_clearance_recommended 为 true 时，"
                    "下一阶段必须先将未归位且未持物的机械臂移出工作区；"
                    "持物机械臂应先执行单独放置，不能直接归位。"
                ),
            },
            "completed_objects": completed,
            "remaining_objects": remaining,
            "perception": {
                "failed_part_detection_count": inference[
                    "failed_part_detection_count"
                ],
                "table_height_world_z": inference["table"][
                    "estimated_height_world_z_m"
                ],
                "position_overlay": inference["outputs"][
                    "position_overlay"
                ],
                "positions_json": inference["outputs"][
                    "positions_json"
                ],
            },
        }
        self._latest_agent_visual_state = state

        # Select the free box region before asking the Agent which object to
        # manipulate. Every remaining visible target is therefore presented
        # together with opaque references to perception/planner-derived
        # parameters. The Agent may arrange skills and reference these values,
        # but cannot invent coordinates or motion distances.
        self._agent_preplanned_drop_xy.clear()
        self.agent_skill_parameter_registry.clear()
        object_state_by_name = {
            item["name"]: item for item in object_states
        }
        for object_name in remaining:
            item = object_state_by_name.get(object_name)
            if (
                item is None
                or item.get("position_xyz") is None
            ):
                continue
            arm_name = item.get("recommended_arm")
            if arm_name not in {"left", "right"}:
                continue
            template = self.get_agent_visual_object_template(object_name)
            candidates = self.container_drop_pose_candidates(
                template,
                self.empty_box,
                arm_name,
                maximum_candidates=1,
                visual_object_name=object_name,
            )
            if not candidates:
                continue
            preview = np.asarray(candidates[0][:3], dtype=float)
            self._agent_preplanned_drop_xy[object_name] = (
                preview[:2].copy()
            )
            item["planned_drop_xyz"] = np.round(
                preview,
                6,
            ).tolist()
            item["drop_region_selection"] = dict(
                self._agent_latest_drop_region_state
            )
            observation_id = self._agent_vision_observation_index
            reference_prefix = f"obs{observation_id}:{object_name}"
            grasp_ref = f"{reference_prefix}:grasp"
            lift_ref = f"{reference_prefix}:lift"
            drop_ref = f"{reference_prefix}:drop"
            retreat_ref = f"{reference_prefix}:retreat"
            self.agent_skill_parameter_registry.update(
                {
                    grasp_ref: {
                        "kind": "grasp",
                        "object": object_name,
                        "arm": arm_name,
                        "pre_grasp_distance": 0.12,
                    },
                    lift_ref: {
                        "kind": "lift",
                        "object": object_name,
                        "arm": arm_name,
                        "distance": 0.10,
                    },
                    drop_ref: {
                        "kind": "drop",
                        "object": object_name,
                        "arm": arm_name,
                        "container": "box",
                        "target_pose": np.asarray(
                            candidates[0],
                            dtype=float,
                        ).tolist(),
                    },
                    retreat_ref: {
                        "kind": "retreat",
                        "object": object_name,
                        "arm": arm_name,
                        "distance": 0.08,
                    },
                }
            )
            item["skill_parameters"] = {
                "pick": {
                    "arm": arm_name,
                    "grasp_ref": grasp_ref,
                    "pre_grasp_distance_m": 0.12,
                },
                "lift": {
                    "distance_ref": lift_ref,
                    "distance_m": 0.10,
                },
                "place_in": {
                    "container": "box",
                    "drop_ref": drop_ref,
                    "preview_xyz": np.round(preview, 6).tolist(),
                },
                "retreat": {
                    "distance_ref": retreat_ref,
                    "distance_m": 0.08,
                },
            }
        state["container"]["placement_planned_before_pick"] = True
        state["container"]["recorded_occupied_drop_regions"] = sorted(
            self._agent_recorded_occupied_drop_regions
        )
        state["skill_parameter_policy"] = (
            "Agent只能引用objects.skill_parameters中的ref；"
            "ref由本轮RGB-D和运动约束产生，执行器解析实际数值"
        )
        return state

    @staticmethod
    def _arm_workspace_from_position(position: Any) -> str:
        """Classify a visual target without hard-coding an object identity."""
        x = float(position[0])
        if x <= -0.08:
            return "left"
        if x >= 0.08:
            return "right"
        return "shared"

    def _recommended_arm_from_position(self, position: Any) -> str:
        """Choose the safe side arm; use current reach distance in shared space."""
        workspace = self._arm_workspace_from_position(position)
        if workspace != "shared":
            return workspace
        target = np.asarray(position, dtype=float)[:3]
        left = np.asarray(self.robot.get_left_ee_pose(), dtype=float)[:3]
        right = np.asarray(self.robot.get_right_ee_pose(), dtype=float)[:3]
        return (
            "left"
            if np.linalg.norm(target - left) < np.linalg.norm(target - right)
            else "right"
        )

    @staticmethod
    def _is_dual_grasp_pair_safe(
        left_state: dict,
        right_state: dict,
    ) -> bool:
        """Require visibly separated targets before allowing a dual-arm pick."""
        left_position = left_state.get("position_xyz")
        right_position = right_state.get("position_xyz")
        if (
            left_position is None
            or right_position is None
            or left_state.get("placed_in_box", False)
            or right_state.get("placed_in_box", False)
            or left_state["name"] == right_state["name"]
        ):
            return False
        left_x = float(left_position[0])
        right_x = float(right_position[0])
        return bool(
            left_x <= -0.08
            and right_x >= 0.08
            and right_x - left_x >= 0.24
        )

    def observe_agent_state(self, goal: AgentGoal) -> dict:
        """Return vision/proprioception state, or legacy state when disabled."""
        if self.agent_vision_enabled:
            return self._observe_visual_agent_state(goal)

        # Kept for explicit legacy JSON execution modes. Closed-loop
        # ``--agent-run`` enables the RGB-D branch above.
        actors = self._agent_actor_map()
        completed = [
            name
            for name in goal.target_objects
            if self._is_part_placed(actors[name])
        ]
        remaining = [
            name for name in goal.target_objects if name not in completed
        ]

        left_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        right_pose = np.asarray(self.robot.get_right_ee_pose(), dtype=float)
        left_home = np.asarray(self.robot.left_original_pose, dtype=float)
        right_home = np.asarray(self.robot.right_original_pose, dtype=float)
        left_at_home = bool(np.linalg.norm(left_pose[:3] - left_home[:3]) < 0.04)
        right_at_home = bool(np.linalg.norm(right_pose[:3] - right_home[:3]) < 0.04)
        arms_distance = float(np.linalg.norm(left_pose[:3] - right_pose[:3]))

        object_states = []
        for name, actor in actors.items():
            position = np.asarray(actor.get_pose().p, dtype=float)
            placed = self._is_part_placed(actor)
            object_states.append(
                {
                    "name": name,
                    "position_xyz": np.round(position, 4).tolist(),
                    "placed_in_box": placed,
                }
            )

        return {
            "observation_source": (
                "simulator_privileged_state; this can later be replaced by "
                "camera perception and proprioception"
            ),
            "objects": object_states,
            "container": {
                "name": "box",
                "asset": "062_plasticbox/base3",
                "runtime_horizontal_scale": 1.55,
                "position_xyz": np.round(
                    np.asarray(self.empty_box.get_pose().p, dtype=float), 4
                ).tolist(),
                "placement_policy": "runtime_free_space",
            },
            "robot": {
                "left_end_effector_xyz": np.round(left_pose[:3], 4).tolist(),
                "right_end_effector_xyz": np.round(right_pose[:3], 4).tolist(),
                "left_arm_at_home": left_at_home,
                "right_arm_at_home": right_at_home,
                "left_gripper_open": bool(self.is_left_gripper_open()),
                "right_gripper_open": bool(self.is_right_gripper_open()),
                "end_effector_distance": round(arms_distance, 4),
            },
            "safety": {
                "arms_too_close": arms_distance < 0.18,
                "workspace_clearance_recommended": bool(
                    remaining and (not left_at_home or not right_at_home)
                ),
                "instruction": (
                    "workspace_clearance_recommended 为 true 时，"
                    "下一阶段必须先将所有未归位机械臂移出工作区再抓取。"
                ),
            },
            "completed_objects": completed,
            "remaining_objects": remaining,
        }

    def check_goal_success(self, goal: AgentGoal) -> bool:
        if self.agent_vision_enabled:
            if self._latest_agent_visual_state is None:
                return False
            completed = set(
                self._latest_agent_visual_state["completed_objects"]
            )
            return bool(
                goal.target_objects
                and set(goal.target_objects).issubset(completed)
            )

        actors = self._agent_actor_map()
        return bool(
            self.plan_success
            and all(
                self._is_part_placed(actors[name])
                for name in goal.target_objects
            )
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def start_video(self, output_path: Path, fps: int = 10):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_output_path = output_path
        self.video_temp_path = output_path.with_name(f"{output_path.stem}.recording.mp4")
        video_size = f"{self.head_camera.width}x{self.head_camera.height}"
        self.video_process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", video_size, "-framerate", str(fps),
                "-i", "-", "-pix_fmt", "yuv420p",
                "-vcodec", "libx264", "-crf", "23", str(self.video_temp_path),
            ],
            stdin=subprocess.PIPE,
        )
        self._record_video_frame()

    def _record_video_frame(self):
        if self.video_process is None:
            return
        rgb = self.publish_web_camera_frames()
        self.video_process.stdin.write(rgb.tobytes())

    def publish_web_camera_frames(
        self,
        *,
        force: bool = False,
    ) -> np.ndarray:
        """Capture the three operator cameras and publish one dashboard frame."""
        self.cameras.update_wrist_camera(
            self.robot.left_camera.get_entity_pose(),
            self.robot.right_camera.get_entity_pose(),
        )
        self.scene.update_render()
        self.head_camera.take_picture()
        rgba = self.head_camera.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")
        if force:
            self._web_stream_last_publish = 0.0
        self._publish_web_camera_frames(head_rgb=rgb)
        return rgb

    @staticmethod
    def _camera_rgb(camera) -> np.ndarray:
        camera.take_picture()
        rgba = camera.get_picture("Color")
        return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")

    @staticmethod
    def _atomic_save_jpeg(rgb: np.ndarray, output_path: Path) -> None:
        temporary_path = output_path.with_suffix(".jpg.tmp")
        Image.fromarray(rgb).save(
            temporary_path,
            format="JPEG",
            quality=86,
            optimize=False,
        )
        os.replace(temporary_path, output_path)

    def _publish_web_camera_frames(
        self,
        *,
        head_rgb: np.ndarray,
    ) -> None:
        """Publish the official head view and two wrist views."""
        if self.web_stream_dir is None:
            return

        # JPEG encoding can otherwise dominate dense-action playback. Eight
        # dashboard updates per wall-clock second are enough for monitoring.
        now = time.monotonic()
        if (
            self._web_stream_sequence > 0
            and now - self._web_stream_last_publish < 0.12
        ):
            return

        self.web_stream_dir.mkdir(parents=True, exist_ok=True)
        frames = {
            "head_camera": head_rgb,
        }
        if self.cameras.collect_wrist_camera:
            frames["left_camera"] = self._camera_rgb(
                self.cameras.left_camera
            )
            frames["right_camera"] = self._camera_rgb(
                self.cameras.right_camera
            )

        for camera_name, camera_rgb in frames.items():
            self._atomic_save_jpeg(
                camera_rgb,
                self.web_stream_dir / f"{camera_name}.jpg",
            )

        self._web_stream_sequence += 1
        self._web_stream_last_publish = now
        metadata_path = self.web_stream_dir / "camera_state.json"
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(
                {
                    "sequence": self._web_stream_sequence,
                    "updated_unix_seconds": time.time(),
                    "cameras": sorted(frames),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)

    def _take_picture(self):
        """Hook used by RoboTwin's dense-action executor."""
        self.raise_if_web_stop_requested()
        if self.video_process is None:
            if self._publish_unrecorded_motion:
                self.publish_web_camera_frames(force=True)
            return
        self._record_video_frame()

    def finish_video(self):
        if self.video_process is not None:
            # Always expose the settled final pose to the dashboard even when
            # the previous dense-action frame was published very recently.
            self._web_stream_last_publish = 0.0
            self._record_video_frame()
            self.video_process.stdin.close()
            return_code = self.video_process.wait()
            self.video_process = None
            if return_code != 0:
                raise RuntimeError(f"ffmpeg 编码失败，退出码 {return_code}")
            os.replace(self.video_temp_path, self.video_output_path)


def load_args(render_freq: int, *, scene_only: bool = True) -> dict:
    with LOCAL_CONFIG.open("r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    embodiment_dir = REPO_ROOT / "assets" / "embodiments" / "aloha-agilex"
    with (embodiment_dir / "config.yml").open("r", encoding="utf-8") as file:
        embodiment_config = yaml.safe_load(file)

    args.update(
        {
            "task_name": "my_parts_box_scene",
            "now_ep_num": 0,
            "seed": 0,
            "render_freq": render_freq,
            "save_data": False,
            "need_plan": not scene_only,
            "scene_only": scene_only,
            "left_robot_file": str(embodiment_dir),
            "right_robot_file": str(embodiment_dir),
            "left_embodiment_config": embodiment_config,
            "right_embodiment_config": embodiment_config,
            "dual_arm_embodied": True,
            "embodiment_name": "aloha-agilex",
        }
    )
    if not scene_only:
        args["save_freq"] = 30
    return args


def run_viewer() -> None:
    scene = PartsBoxScene()
    try:
        scene.setup_demo(**load_args(render_freq=1))
        print("场景已创建。关闭 Viewer 窗口即可退出。")
        while not scene.viewer.closed:
            scene.scene.step()
            scene._update_render()
            scene.viewer.render()
    finally:
        if hasattr(scene, "viewer") and not scene.viewer.closed:
            scene.viewer.close()
        scene.close_env()


def save_snapshot(output_path: Path) -> None:
    scene = PartsBoxScene()
    try:
        scene.setup_demo(**load_args(render_freq=0))
        scene.scene.step()
        scene._update_render()
        scene.head_camera.take_picture()
        rgba = scene.head_camera.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(rgb).save(output_path)
        print(f"场景截图已保存：{output_path}")
    finally:
        scene.close_env()


def run_vision_test(
    output_dir: Path,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    device: str,
) -> None:
    """Capture the physical head camera once and run YOLO-World only."""
    scene = PartsBoxScene()
    try:
        print("正在创建场景并采集机器人 head_camera 的单帧 RGB 图像……")
        scene.setup_demo(**load_args(render_freq=0))

        # Let newly created dynamic objects settle before the single capture.
        for _ in range(120):
            scene.scene.step()
        scene._update_render()
        scene.cameras.update_picture()
        camera_rgb = scene.cameras.get_rgb()
        if "head_camera" not in camera_rgb:
            raise RuntimeError(
                "当前机器人配置没有启用 head_camera，请检查本示例 scene_config.yml"
            )
        rgb = camera_rgb["head_camera"]["rgb"]

        result = run_yolo_world_single_frame(
            rgb,
            output_dir,
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        print(f"head_camera 原始图像：{result['image']['raw_path']}")
        print(f"YOLO-World 标注图像：{result['image']['annotated_path']}")
        print(f"检测结果 JSON：{result['json_path']}")
        print(
            "检测数量："
            + "，".join(
                f"{label}={check['detected']}/{check['expected']}"
                for label, check in result["count_check"].items()
            )
        )
        print(
            "数量初检："
            + ("通过" if result["all_expected_counts_matched"] else "未通过")
            + "（请结合标注图检查误检和漏检）"
        )
    finally:
        scene.close_env()


def _latest_trained_weights() -> Path:
    trained_dir = LOCAL_CONFIG.parent / "trained_weights"
    candidates = sorted(
        trained_dir.glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "没有找到训练后的权重。请先训练YOLO，或通过"
            "--position-model指定best.pt。"
        )
    return candidates[0].resolve()


def run_vision_position_test(
    output_dir: Path,
    *,
    model_path: str | None,
    confidence: float,
    image_size: int,
    device: str,
) -> None:
    """Estimate tabletop part positions from head RGB-D without moving arms."""
    scene = PartsBoxScene()
    try:
        selected_model = (
            Path(model_path).expanduser().resolve()
            if model_path
            else _latest_trained_weights()
        )
        print("正在创建场景并采集同一时刻的head_camera RGB-D……")
        print(f"使用训练权重：{selected_model}")
        scene.setup_demo(**load_args(render_freq=0))
        for _ in range(120):
            scene.scene.step()

        scene._update_render()
        cameras = dict(
            zip(
                scene.cameras.static_camera_name,
                scene.cameras.static_camera_list,
            )
        )
        if "head_camera" not in cameras:
            raise RuntimeError("当前场景没有启用head_camera")
        head_camera = cameras["head_camera"]
        head_camera.take_picture()
        rgba = head_camera.get_picture("Color")
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        position_image = head_camera.get_picture("Position")
        camera_to_world = head_camera.get_model_matrix()

        result = run_rgbd_position_evaluation(
            rgb,
            position_image,
            camera_to_world,
            {
                "part_A": list(scene.parts_a),
                "part_B": list(scene.parts_b),
            },
            output_dir,
            model_path=str(selected_model),
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        print(
            "RGB-D位置比较完成："
            f"匹配 {result['metrics']['matched_count']}/5 个桌面零件"
        )
        for comparison in result["comparisons"]:
            estimate = comparison["estimated_position_world_xyz_m"]
            truth = comparison["privileged_position_world_xyz_m"]
            print(
                f"  {comparison['matched_actor_name']}: "
                f"估计={estimate}，特权={truth}，"
                f"水平误差={comparison['horizontal_error_m'] * 1000:.1f} mm，"
                f"三维误差={comparison['position_error_3d_m'] * 1000:.1f} mm"
            )
        metrics = result["metrics"]
        if metrics["matched_count"]:
            print(
                "总体误差："
                f"平均水平={metrics['mean_horizontal_error_m'] * 1000:.1f} mm，"
                f"最大水平={metrics['max_horizontal_error_m'] * 1000:.1f} mm，"
                f"平均三维={metrics['mean_position_error_3d_m'] * 1000:.1f} mm"
            )
        print(f"位置标注图：{result['outputs']['position_overlay']}")
        print(f"深度可视化：{result['outputs']['depth_visualization']}")
        print(f"完整比较JSON：{result['outputs']['comparison_json']}")
        print("本测试未调用运动规划器或机械臂技能。")
    finally:
        scene.close_env()


def run_vision_robustness_test(
    output_dir: Path,
    *,
    model_path: str | None,
    confidence: float,
    image_size: int,
    device: str,
    seed: int,
) -> None:
    """Run RGB-D position extraction under seven stress conditions."""
    scene = PartsBoxScene()
    try:
        selected_model = (
            Path(model_path).expanduser().resolve()
            if model_path
            else _latest_trained_weights()
        )
        print("正在创建一次场景并运行7组RGB-D鲁棒性测试……")
        print(f"使用训练权重：{selected_model}")
        scene.setup_demo(**load_args(render_freq=0))
        result = run_rgbd_robustness_suite(
            scene,
            output_dir,
            model_path=str(selected_model),
            confidence=confidence,
            image_size=image_size,
            device=device,
            seed=seed,
        )
        aggregate = result["aggregate"]
        print(
            "鲁棒性测试完成："
            f"累计匹配{aggregate['total_matched_parts']}/"
            f"{aggregate['total_expected_parts']}个零件"
        )
        if aggregate["mean_of_scenario_horizontal_errors_m"] is not None:
            print(
                "各场景平均水平误差的均值："
                f"{aggregate['mean_of_scenario_horizontal_errors_m'] * 1000:.1f} mm"
            )
        print(f"完整汇总：{result['summary_path']}")
        print("测试全过程未调用运动规划器或机械臂技能。")
    finally:
        scene.close_env()


def run_dataset_generation(
    output_dir: Path,
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> None:
    """Create one scene and repeatedly render exact offline training labels."""
    scene = PartsBoxScene()
    try:
        print(
            "正在创建一次 RoboTwin 场景并自动生成 "
            f"{train_count} 张训练图、{val_count} 张验证图……"
        )
        scene.setup_demo(**load_args(render_freq=0))
        summary = generate_yolo_dataset(
            scene,
            output_dir,
            train_count=train_count,
            val_count=val_count,
            seed=seed,
        )
        print("视觉训练数据生成完成。")
        print(f"数据集目录：{summary['output_dir']}")
        print(f"YOLO配置：{summary['data_yaml']}")
        print(f"生成记录：{summary['metadata']}")
        print(f"逐图标注审计：{summary['annotations']}")
        print(
            f"有效图片：train={summary['train_count']}，"
            f"val={summary['val_count']}；"
            f"丢弃不可见/截断样本={summary['rejected_samples']}"
        )
        print(
            "盒内零件数量分布："
            + "，".join(
                f"{count}个={frames}张"
                for count, frames in
                summary["accepted_frames_by_inside_part_count"].items()
            )
        )
        print(
            "零件姿态实例分布："
            + "，".join(
                f"{mode}={instances}个"
                for mode, instances in
                summary["accepted_part_instances_by_orientation"].items()
            )
        )
    finally:
        scene.close_env()


def run_supplemental_dataset_generation(
    output_dir: Path,
    *,
    scene_count: int,
    seed: int,
) -> None:
    """Generate a small physical multi-camera dataset for visual review."""
    scene = PartsBoxScene()
    try:
        print(
            "正在创建一次带运动规划器的 RoboTwin 场景并生成"
            f"{scene_count}个物理布局、{scene_count * 3}张多相机补充图片……"
        )
        scene.setup_demo(**load_args(render_freq=0, scene_only=False))
        summary = generate_multicamera_supplement(
            scene,
            output_dir,
            scene_count=scene_count,
            seed=seed,
        )
        print("多相机补充训练数据生成完成。")
        print(f"数据集目录：{summary['output_dir']}")
        print(f"标注预览目录：{Path(summary['output_dir']) / 'previews'}")
        print(
            "图片分布："
            + "，".join(
                f"{camera}={count}张"
                for camera, count in summary[
                    "camera_image_counts"
                ].items()
            )
        )
        print(
            f"边缘标签={summary['edge_touching_label_count']}个，"
            "部分遮挡/截断标签="
            f"{summary['partial_or_occluded_label_count']}个，"
            "机械臂遮挡头部场景="
            f"{summary['robot_occlusion_head_scenes']}个"
        )
        print(f"生成记录：{summary['metadata']}")
        print(f"逐图标注审计：{summary['annotations']}")
    finally:
        scene.close_env()


def run_task(output_path: Path) -> None:
    scene = PartsBoxScene()
    try:
        print("正在加载 RoboTwin 运动规划器并创建任务，请稍候……")
        scene.setup_demo(**load_args(render_freq=0, scene_only=False))
        scene.start_video(output_path)
        print("开始执行：抓取 3 个零件 A 并放入盒子。")
        scene.play_once()
        scene.finish_video()
        if not scene.plan_success:
            raise RuntimeError("运动规划失败；视频保留了失败前的过程，可调整位姿后重试。")
        success = scene.check_success()
        print(f"任务执行完成，成功判定：{success}")
        print(f"操作视频已保存：{output_path}")
        if not success:
            raise RuntimeError("动作轨迹执行完成，但物体最终位置未通过成功判定。")
    finally:
        if scene.video_process is not None:
            scene.finish_video()
        scene.close_env()


def run_agent_plan(plan_path: Path, output_path: Path) -> None:
    plan = load_and_validate_plan(plan_path)
    scene = PartsBoxScene()
    try:
        print(f"JSON计划校验通过，共 {len(plan.steps)} 个技能步骤。")
        print("正在加载 RoboTwin 运动规划器并创建任务，请稍候……")
        scene.setup_demo(**load_args(render_freq=0, scene_only=False))
        scene.start_video(output_path)
        report = scene.execute_agent_plan(plan)
        scene.finish_video()

        success = report.success and scene.check_plan_success(plan)
        print(f"Agent计划执行完成，成功判定：{success}")
        print(f"操作视频已保存：{output_path}")
        if not success:
            raise RuntimeError("技能已执行，但目标物体最终位置未通过成功判定。")
        print(f"Agent：{report.final_response}")
    finally:
        if scene.video_process is not None:
            scene.finish_video()
        scene.close_env()


def _print_plan_summary(phase_index: int, plan) -> None:
    print(
        f"\n[闭环规划 {phase_index}] 下一阶段："
        f"{plan.understood_goal}"
    )


def _planned_place_objects(plan) -> set[str]:
    """Return objects that this phase claims it will put into the box."""
    planned: set[str] = set()
    for step in plan.steps:
        arguments = step.get("arguments", {})
        if (
            step.get("skill") == "place_in"
            and arguments.get("container") == "box"
        ):
            planned.add(arguments["object"])
        elif (
            step.get("skill") == "place_in_dual"
            and arguments.get("container") == "box"
        ):
            planned.update(
                (
                    arguments["left_object"],
                    arguments["right_object"],
                )
            )
    return planned


def prepare_closed_loop_agent_scene(scene: PartsBoxScene) -> dict:
    """Load one reusable simulator scene and configure Agent perception."""
    print("正在加载 RoboTwin 运动规划器并创建闭环 Agent 场景，请稍候……")
    scene.setup_demo(**load_args(render_freq=0, scene_only=False))
    vision_model_text = os.getenv("AGENT_VISION_MODEL")
    vision_model = (
        Path(vision_model_text)
        if vision_model_text
        else _latest_trained_weights()
    )
    try:
        vision_confidence = float(
            os.getenv("AGENT_VISION_CONF", "0.60")
        )
        vision_image_size = int(
            os.getenv("AGENT_VISION_IMGSZ", "640")
        )
    except ValueError as exc:
        raise RuntimeError(
            "AGENT_VISION_CONF必须是小数，"
            "AGENT_VISION_IMGSZ必须是整数"
        ) from exc
    scene.configure_agent_vision(
        model_path=vision_model,
        confidence=vision_confidence,
        image_size=vision_image_size,
        device=os.getenv("AGENT_VISION_DEVICE", "0"),
    )
    perception = {
        "source": (
            "head_camera RGB-D position + fixed object quaternion + "
            "category asset contact matrix"
        ),
        "model": str(Path(vision_model).expanduser().resolve()),
        "confidence": vision_confidence,
        "image_size": vision_image_size,
        "privileged_object_pose_used": False,
        "asset_contact_point_used": True,
        "asset_grasp_direction_used": True,
    }
    print(
        "Agent抓取来源：head_camera RGB-D位置 + 固定物体姿态 "
        "+ 类别资产接触矩阵；不读取仿真器物体真实位姿。"
    )
    return perception


def run_visual_placement_test(output_path: Path) -> None:
    """Exercise the real visual pick/place stack without calling an LLM."""
    scene = PartsBoxScene()
    try:
        prepare_closed_loop_agent_scene(scene)
        goal = AgentGoal.from_dict(
            {
                "understood_goal": "放置视觉检测到的一个零件A",
                "target_category": "part_A",
                "target_selector": "all",
                "target_objects": [],
                "container": "box",
                "needs_clarification": False,
                "clarification_question": None,
            }
        )
        state = scene.observe_agent_state(goal)
        candidates = [
            item
            for item in state["objects"]
            if (
                item["name"] in goal.target_objects
                and not item["placed_in_box"]
                and item["position_xyz"] is not None
            )
        ]
        if not candidates:
            raise RuntimeError("放置测试没有检测到可抓取的零件A")

        def reach_distance(item):
            arm = item["recommended_arm"]
            ee = np.asarray(
                state["robot"][f"{arm}_end_effector_xyz"],
                dtype=float,
            )
            return float(
                np.linalg.norm(
                    np.asarray(item["position_xyz"], dtype=float) - ee
                )
            )

        selected = min(candidates, key=reach_distance)
        object_name = selected["name"]
        arm = selected["recommended_arm"]
        print(f"放置测试目标：{object_name}，机械臂：{arm}")
        scene.start_video(output_path)

        pick_phase = AgentPlan.from_dict(
            {
                "understood_goal": f"抓取并抬升{object_name}",
                "needs_clarification": False,
                "clarification_question": None,
                "steps": [
                    {
                        "skill": "pick",
                        "arguments": {
                            "object": object_name,
                            "arm": arm,
                        },
                    },
                    {
                        "skill": "lift",
                        "arguments": {
                            "arm": arm,
                            "distance": 0.10,
                        },
                    },
                ],
                "final_response": "继续测试，还需要什么？",
            }
        )
        scene.execute_agent_plan(pick_phase)
        scene.observe_agent_state(goal)

        place_phase = AgentPlan.from_dict(
            {
                "understood_goal": f"将{object_name}放入盒内最大空闲区",
                "needs_clarification": False,
                "clarification_question": None,
                "steps": [
                    {
                        "skill": "place_in",
                        "arguments": {
                            "object": object_name,
                            "container": "box",
                            "arm": arm,
                        },
                    },
                    {
                        "skill": "retreat",
                        "arguments": {
                            "arm": arm,
                            "distance": 0.08,
                        },
                    },
                ],
                "final_response": "放置测试完成，还需要什么？",
            }
        )
        report = scene.execute_agent_plan(place_phase)
        state_after = scene.observe_agent_state(goal)
        success = bool(
            report.success
            and object_name in state_after["completed_objects"]
        )
        print(f"视觉放置测试成功判定：{success}")
        if not success:
            raise RuntimeError("动作执行后视觉未确认零件进入盒子")
    finally:
        if scene.video_process is not None:
            scene.finish_video()
        scene.close_env()


def run_closed_loop_agent(
    user_text: str,
    output_path: Path,
    *,
    scene: PartsBoxScene | None = None,
    planner: AgentPlanner | None = None,
    prepared_perception: dict | None = None,
) -> None:
    """Re-observe and re-plan after every safe manipulation phase."""
    owns_scene = scene is None
    if scene is None:
        scene = PartsBoxScene()
    if planner is None:
        planner = AgentPlanner()
    history: list[dict] = []
    trace: dict = {
        "user_text": user_text,
        "goal": None,
        "phases": [],
    }
    try:
        perception = (
            prepared_perception
            if prepared_perception is not None
            else prepare_closed_loop_agent_scene(scene)
        )
        trace["perception"] = dict(perception)

        scene.raise_if_web_stop_requested()
        print("新任务开始前，正在让双臂返回home状态……", flush=True)
        home_ok = scene.return_arms_home_before_new_task()
        trace["pre_task_home"] = {"success": bool(home_ok)}
        if not home_ok:
            raise RuntimeError("新任务开始前双臂未能安全返回home状态")
        print("双臂已处于home状态，准备理解并观察新任务。", flush=True)
        scene.raise_if_web_stop_requested()
        goal = planner.understand_goal(user_text)
        scene.raise_if_web_stop_requested()
        trace["goal"] = goal.to_dict()
        print(f"Agent理解到的任务：{goal.understood_goal}")
        if goal.needs_clarification:
            print(f"Agent：{goal.clarification_question}")
            return

        scene.start_video(output_path)
        final_response = "任务完成，还需要什么？"
        max_phases = 10
        consecutive_execution_failures = 0
        maximum_execution_retries = 1
        consecutive_no_progress_phases = 0
        no_progress_attempts_by_object: dict[str, int] = {}
        maximum_no_progress_attempts = 1
        announced_visual_targets: set[str] = set()

        for phase_index in range(1, max_phases + 1):
            scene.raise_if_web_stop_requested()
            try:
                state_before = scene.observe_agent_state(goal)
            except NoTrustedVisualTargets as exc:
                message = str(exc)
                print(message, flush=True)
                trace["result"] = {
                    "status": "no_trusted_visual_targets",
                    "message": message,
                }
                return
            scene.raise_if_web_stop_requested()
            current_visual_targets = set(goal.target_objects)
            newly_detected_targets = (
                current_visual_targets - announced_visual_targets
            )
            if newly_detected_targets:
                print(
                    "head_camera当前检测并绑定了"
                    f"{len(current_visual_targets)}个"
                    f"{goal.target_category}目标："
                    f"{sorted(current_visual_targets)}",
                    flush=True,
                )
                announced_visual_targets = current_visual_targets
            trace["goal"] = goal.to_dict()

            if scene.check_goal_success(goal):
                break

            plan = planner.plan_next(goal, state_before, history)
            scene.raise_if_web_stop_requested()
            _print_plan_summary(phase_index, plan)
            if plan.needs_clarification:
                raise RuntimeError(
                    f"执行中需要用户澄清：{plan.clarification_question}"
                )

            GENERATED_PLAN_PATH.write_text(
                json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                report = scene.execute_agent_plan(plan)
            except PlanExecutionError as exc:
                print(f"[闭环执行 {phase_index}] 失败：{exc}")
                consecutive_execution_failures += 1
                trace["phases"].append(
                    {
                        "phase": phase_index,
                        "state_before": state_before,
                        "plan": plan.to_dict(),
                        "state_after": None,
                        "execution_success": False,
                        "failure": str(exc),
                    }
                )
                history.append(
                    {
                        "phase": phase_index,
                        "skills": [
                            step["skill"] for step in plan.steps
                        ],
                        "execution_success": False,
                        "failure": str(exc),
                    }
                )
                EXECUTION_TRACE_PATH.write_text(
                    json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if (
                    consecutive_execution_failures
                    >= maximum_execution_retries
                ):
                    raise RuntimeError(
                        "抓取执行和重新规划连续失败"
                        f"{maximum_execution_retries}次，已安全停止"
                    ) from exc
                scene.recover_after_failed_agent_grasp()
                print(
                    "已恢复到安全姿态，将重新进行视觉观察和Agent规划。",
                    flush=True,
                )
                continue
            consecutive_execution_failures = 0
            final_response = report.final_response
            state_after = scene.observe_agent_state(goal)
            completed_before = set(state_before["completed_objects"])
            completed_after = set(state_after["completed_objects"])
            newly_completed = completed_after - completed_before
            planned_place_objects = _planned_place_objects(plan)
            visually_completed_planned = (
                newly_completed & planned_place_objects
            )
            visual_progress = bool(
                not planned_place_objects
                or visually_completed_planned
            )

            if visual_progress:
                consecutive_no_progress_phases = 0
                for name in visually_completed_planned:
                    no_progress_attempts_by_object.pop(name, None)
                if visually_completed_planned:
                    print(
                        f"[闭环执行 {phase_index}] 成功：视觉确认新增完成 "
                        f"{sorted(visually_completed_planned)}。",
                        flush=True,
                    )
                else:
                    print(
                        f"[闭环执行 {phase_index}] 成功。",
                        flush=True,
                    )
            else:
                consecutive_no_progress_phases += 1
                for name in planned_place_objects:
                    if name in completed_after:
                        continue
                    no_progress_attempts_by_object[name] = (
                        no_progress_attempts_by_object.get(name, 0) + 1
                    )
                failed_attempts = {
                    name: no_progress_attempts_by_object[name]
                    for name in sorted(planned_place_objects)
                    if name not in completed_after
                }
                print(
                    f"[闭环执行 {phase_index}] 未确认成功：机器人动作已执行，"
                    "但head_camera未确认目标进入盒子；"
                    f"无进展次数={failed_attempts}。",
                    flush=True,
                )

            phase_record = {
                "phase": phase_index,
                "state_before": state_before,
                "plan": plan.to_dict(),
                "state_after": state_after,
                "motion_execution_success": report.success,
                "execution_success": bool(
                    report.success and visual_progress
                ),
                "visual_progress": visual_progress,
                "newly_completed_objects": sorted(newly_completed),
            }
            trace["phases"].append(phase_record)
            history_record = {
                "phase": phase_index,
                "skills": [step["skill"] for step in plan.steps],
                "planned_place_objects": sorted(planned_place_objects),
                "completed_objects_after": state_after["completed_objects"],
                "robot_after": state_after["robot"],
                "execution_success": bool(
                    report.success and visual_progress
                ),
                "visual_progress": visual_progress,
            }
            if not visual_progress:
                history_record["failure"] = (
                    "运动技能执行完毕，但视觉复核没有发现任何目标新增进入盒子；"
                    "上一轮不能视为成功。不要原样重复完全相同的方案。"
                )
                history_record["no_progress_attempts"] = failed_attempts
            history.append(history_record)
            EXECUTION_TRACE_PATH.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            if not report.success:
                raise RuntimeError(f"闭环阶段 {phase_index} 执行失败")
            if not visual_progress:
                exhausted_objects = [
                    name
                    for name, attempts in no_progress_attempts_by_object.items()
                    if attempts >= maximum_no_progress_attempts
                ]
                if (
                    exhausted_objects
                    or consecutive_no_progress_phases
                    >= maximum_no_progress_attempts
                ):
                    scene.recover_after_failed_agent_grasp()
                    raise RuntimeError(
                        "机器人动作连续执行但视觉状态没有任务进展，"
                        f"已在{maximum_no_progress_attempts}次后安全停止；"
                        f"未成功目标={sorted(exhausted_objects or planned_place_objects)}"
                    )
                print(
                    "本轮未取得视觉进展，将把失败结果交给Agent重新规划。",
                    flush=True,
                )
                continue
        else:
            raise RuntimeError(
                f"闭环 Agent 已达到最多 {max_phases} 个阶段，任务仍未完成"
            )

        success = scene.check_goal_success(goal)
        trace["success_check"] = {
            "source": "latest_head_camera_observation",
            "all_targets_observed_inside_box": bool(success),
        }
        EXECUTION_TRACE_PATH.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        scene.finish_video()
        print("目标已由head_camera视觉确认进入盒子。")
        print(f"闭环 Agent 执行完成，成功判定：{success}")
        print(f"操作视频已保存：{output_path}")
        print(f"完整观察与规划轨迹已保存：{EXECUTION_TRACE_PATH}")
        if not success:
            raise RuntimeError("head_camera未确认全部目标位于盒子中。")
        print(f"Agent：{final_response}")
    finally:
        if scene.video_process is not None:
            scene.finish_video()
        if trace["goal"] is not None:
            EXECUTION_TRACE_PATH.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if owns_scene:
            scene.close_env()


def _emit_web_worker_event(event: str, **data) -> None:
    print(
        WEB_WORKER_EVENT_PREFIX
        + json.dumps(
            {"event": event, **data},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _ffconcat_file_line(path: Path) -> str:
    """Return one safely quoted absolute path for an ffconcat manifest."""
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def append_video_segment(
    segment_path: Path,
    session_video_path: Path,
) -> None:
    """Atomically append one task clip to the current Web scene video."""
    if not segment_path.is_file() or segment_path.stat().st_size <= 0:
        return

    session_video_path.parent.mkdir(parents=True, exist_ok=True)
    if not session_video_path.is_file():
        os.replace(segment_path, session_video_path)
        return

    manifest_path = session_video_path.with_name(
        ".agent_execution_concat.txt"
    )
    merged_path = session_video_path.with_name(
        ".agent_execution_merged.mp4"
    )
    manifest_path.write_text(
        "\n".join(
            (
                "ffconcat version 1.0",
                _ffconcat_file_line(session_video_path),
                _ffconcat_file_line(segment_path),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(merged_path),
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "合并本次任务录像失败，"
                f"ffmpeg退出码={completed.returncode}"
            )
        os.replace(merged_path, session_video_path)
        segment_path.unlink()
    finally:
        manifest_path.unlink(missing_ok=True)
        merged_path.unlink(missing_ok=True)


def run_persistent_agent_worker() -> None:
    """Keep one simulator scene alive and execute JSON-line commands."""
    scene = PartsBoxScene()
    try:
        SESSION_VIDEO_PATH.unlink(missing_ok=True)
        SESSION_VIDEO_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        for stale_segment in SESSION_VIDEO_SEGMENT_DIR.glob("*.mp4"):
            stale_segment.unlink()
        _emit_web_worker_event(
            "initializing",
            message="正在预加载机器人场景与运动规划器",
        )
        perception = prepare_closed_loop_agent_scene(scene)
        planner = AgentPlanner()
        if (
            scene.web_stop_request_path is not None
            and scene.web_stop_request_path.is_file()
        ):
            scene.web_stop_request_path.unlink()
        scene.publish_web_camera_frames(force=True)
        _emit_web_worker_event(
            "ready",
            message="机器人场景已就绪，可以发送任务",
        )

        for raw_line in sys.stdin:
            try:
                request = json.loads(raw_line)
                task_id = str(request["task_id"])
                command = str(request["command"]).strip()
                if not command:
                    raise ValueError("机器人任务不能为空")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                _emit_web_worker_event(
                    "request_error",
                    message=f"工作进程收到非法任务：{exc}",
                )
                continue

            _emit_web_worker_event("task_started", task_id=task_id)
            task_video_path = (
                SESSION_VIDEO_SEGMENT_DIR / f"{task_id}.mp4"
            )
            task_video_path.unlink(missing_ok=True)
            result_event: tuple[str, dict] | None = None
            try:
                run_closed_loop_agent(
                    command,
                    task_video_path,
                    scene=scene,
                    planner=planner,
                    prepared_perception=perception,
                )
                scene.publish_web_camera_frames(force=True)
            except AgentTaskCancelled:
                print("收到停止请求，正在释放夹爪并让双臂返回home状态……")
                try:
                    home_ok = scene.return_arms_home_after_web_stop()
                except Exception:
                    traceback.print_exc()
                    home_ok = False
                result_event = (
                    "task_cancelled",
                    {
                        "task_id": task_id,
                        "home_ok": home_ok,
                        "message": (
                            "任务已停止，双臂已返回home状态"
                            if home_ok
                            else "任务已停止，但双臂未能完全返回home状态"
                        ),
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                try:
                    scene.publish_web_camera_frames(force=True)
                except Exception:
                    traceback.print_exc()
                result_event = (
                    "task_result",
                    {
                        "task_id": task_id,
                        "ok": False,
                        "message": str(exc),
                    },
                )
            else:
                result_event = (
                    "task_result",
                    {
                        "task_id": task_id,
                        "ok": True,
                        "message": "任务执行成功",
                    },
                )
            finally:
                if scene.video_process is not None:
                    scene.finish_video()
                if task_video_path.is_file():
                    try:
                        append_video_segment(
                            task_video_path,
                            SESSION_VIDEO_PATH,
                        )
                        print(
                            "当前场景的累计操作视频已更新："
                            f"{SESSION_VIDEO_PATH}",
                            flush=True,
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        if result_event is not None:
                            event_name, event_payload = result_event
                            if (
                                event_name == "task_result"
                                and event_payload.get("ok")
                            ):
                                event_payload["message"] = (
                                    "机器人任务已完成，但累计录像合并失败："
                                    f"{exc}"
                                )
                        print(f"累计录像合并失败：{exc}", flush=True)
            assert result_event is not None
            _emit_web_worker_event(result_event[0], **result_event[1])
    finally:
        if scene.video_process is not None:
            scene.finish_video()
        scene.close_env()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="无窗口创建场景并验证物体是否存在，然后退出",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="无窗口渲染机器人头部相机并保存 scene_preview.png",
    )
    parser.add_argument(
        "--vision-test",
        action="store_true",
        help="采集原生 head_camera RGB 并执行一次 YOLO-World 检测",
    )
    parser.add_argument(
        "--vision-position-test",
        action="store_true",
        help="用head_camera RGB-D估计桌面零件位置并和特权位姿比较",
    )
    parser.add_argument(
        "--position-model",
        default=None,
        metavar="WEIGHTS",
        help="RGB-D位置测试权重；默认自动使用最新训练best.pt",
    )
    parser.add_argument(
        "--vision-robustness-test",
        action="store_true",
        help="运行旋转、遮挡、相机扰动、深度退化和盒内零件测试",
    )
    parser.add_argument(
        "--robustness-seed",
        type=int,
        default=2027,
        metavar="N",
        help="RGB-D鲁棒性测试随机种子，默认2027",
    )
    parser.add_argument(
        "--vision-model",
        default=os.getenv("YOLO_WORLD_MODEL", "yolov8s-worldv2.pt"),
        metavar="WEIGHTS",
        help="YOLO-World 权重路径或名称",
    )
    parser.add_argument(
        "--vision-conf",
        type=float,
        default=0.60,
        metavar="FLOAT",
        help="YOLO-World 置信度阈值，默认 0.60",
    )
    parser.add_argument(
        "--vision-imgsz",
        type=int,
        default=640,
        metavar="PIXELS",
        help="YOLO-World 推理尺寸，默认 640",
    )
    parser.add_argument(
        "--vision-device",
        default=os.getenv("YOLO_WORLD_DEVICE", "0"),
        metavar="DEVICE",
        help="YOLO-World 推理设备，默认使用当前可见 GPU 0",
    )
    parser.add_argument(
        "--generate-vision-dataset",
        action="store_true",
        help="用head_camera RGB和离线actor分割自动生成YOLO训练数据",
    )
    parser.add_argument(
        "--dataset-train-count",
        type=int,
        default=500,
        metavar="N",
        help="自动生成的训练图片数量，默认500",
    )
    parser.add_argument(
        "--dataset-val-count",
        type=int,
        default=100,
        metavar="N",
        help="自动生成的验证图片数量，默认100",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=2026,
        metavar="N",
        help="数据随机化种子，默认2026",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=LOCAL_CONFIG.parent / "vision_dataset",
        metavar="DIR",
        help="数据集输出目录，默认保存在本示例vision_dataset",
    )
    parser.add_argument(
        "--generate-supplemental-vision-dataset",
        action="store_true",
        help=(
            "生成62°头部、左右腕部、边缘和机械臂遮挡的"
            "小规模自动标注补充集"
        ),
    )
    parser.add_argument(
        "--supplemental-scene-count",
        type=int,
        default=5,
        help="补充集物理场景数；每个场景生成头部和左右腕部共3张图",
    )
    parser.add_argument(
        "--supplemental-seed",
        type=int,
        default=2030,
        help="补充集随机种子，默认2030",
    )
    parser.add_argument(
        "--supplemental-output",
        type=Path,
        default=LOCAL_CONFIG.parent / "vision_dataset_multicamera_smoke",
        help="小规模多相机补充集输出目录",
    )
    parser.add_argument(
        "--run-task",
        action="store_true",
        help="使用特权位姿和 RoboTwin 规划器执行抓取任务并保存 MP4",
    )
    parser.add_argument(
        "--test-skills",
        action="store_true",
        help="顺序测试 pick/lift/place_in/retreat 并保存独立视频",
    )
    parser.add_argument(
        "--placement-test",
        action="store_true",
        help="不调用大模型，单独测试一次视觉抓取和盒内空闲区放置",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="列出基础技能，不启动仿真器",
    )
    parser.add_argument(
        "--execute-plan",
        type=Path,
        metavar="JSON_FILE",
        help="校验并执行指定 JSON 技能计划，保存 agent_execution.mp4",
    )
    parser.add_argument(
        "--agent-loop",
        metavar="TEXT",
        help="根据最新仿真状态分阶段规划、执行并保存 agent_execution.mp4",
    )
    parser.add_argument(
        "--agent-worker",
        action="store_true",
        help="为Web服务常驻一个场景并从标准输入接收JSON任务",
    )
    options = parser.parse_args()

    # RoboTwin uses repository-relative paths for assets and task configs.
    os.chdir(REPO_ROOT)

    if options.list_skills:
        print("当前可用基础技能：")
        for signature, description in RobotSkills.describe().items():
            print(f"  - {signature}: {description}")
        return

    if options.execute_plan:
        run_agent_plan(
            options.execute_plan.resolve(),
            LOCAL_CONFIG.parent / "agent_execution.mp4",
        )
        return

    if options.agent_worker:
        run_persistent_agent_worker()
        return

    if options.agent_loop:
        run_closed_loop_agent(
            options.agent_loop,
            LOCAL_CONFIG.parent / "agent_execution.mp4",
        )
        return

    if options.test_skills:
        run_task(LOCAL_CONFIG.parent / "skills_test.mp4")
        return

    if options.placement_test:
        run_visual_placement_test(
            LOCAL_CONFIG.parent / "placement_test.mp4"
        )
        return

    if options.run_task:
        run_task(LOCAL_CONFIG.parent / "parts_into_box.mp4")
        return

    if options.snapshot:
        save_snapshot(LOCAL_CONFIG.parent / "scene_preview.png")
        return

    if options.vision_test:
        run_vision_test(
            LOCAL_CONFIG.parent / "vision_results",
            model_path=options.vision_model,
            confidence=options.vision_conf,
            image_size=options.vision_imgsz,
            device=options.vision_device,
        )
        return

    if options.vision_position_test:
        run_vision_position_test(
            LOCAL_CONFIG.parent / "rgbd_position_results",
            model_path=options.position_model,
            confidence=options.vision_conf,
            image_size=options.vision_imgsz,
            device=options.vision_device,
        )
        return

    if options.vision_robustness_test:
        run_vision_robustness_test(
            LOCAL_CONFIG.parent / "rgbd_robustness_results",
            model_path=options.position_model,
            confidence=options.vision_conf,
            image_size=options.vision_imgsz,
            device=options.vision_device,
            seed=options.robustness_seed,
        )
        return

    if options.generate_vision_dataset:
        run_dataset_generation(
            options.dataset_output.resolve(),
            train_count=options.dataset_train_count,
            val_count=options.dataset_val_count,
            seed=options.dataset_seed,
        )
        return

    if options.generate_supplemental_vision_dataset:
        run_supplemental_dataset_generation(
            options.supplemental_output.resolve(),
            scene_count=options.supplemental_scene_count,
            seed=options.supplemental_seed,
        )
        return

    if not options.check:
        run_viewer()
        return

    scene = PartsBoxScene()
    try:
        scene.setup_demo(**load_args(render_freq=0))
        actor_names = {actor.get_name() for actor in scene.scene.get_all_actors()}
        expected = {
            "part_A_1",
            "part_A_2",
            "part_A_3",
            "part_B_1",
            "part_B_2",
            "062_plasticbox",
        }
        missing = expected - actor_names
        if missing:
            raise RuntimeError(f"场景缺少物体: {sorted(missing)}")
        print(
            "场景验证成功：轮式双臂机器人、桌子、3 个零件 A、"
            "2 个零件 B 和空盒子均已加载。"
        )
    finally:
        scene.close_env()


if __name__ == "__main__":
    main()
