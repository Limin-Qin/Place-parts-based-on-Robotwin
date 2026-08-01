"""Safely execute a validated AgentPlan through the registered RobotSkills."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .planner import AgentPlan, AgentPlanner
    from .robot_skills import RobotSkills, SkillResult
except ImportError:
    from planner import AgentPlan, AgentPlanner
    from robot_skills import RobotSkills, SkillResult


class PlanExecutionError(RuntimeError):
    """Raised when a plan cannot be translated into robot skill calls."""


@dataclass
class ExecutionReport:
    """Results produced while executing one complete plan."""

    success: bool
    results: list[SkillResult] = field(default_factory=list)
    final_response: str = ""


class _VisualObjectReference:
    """Dynamic visual identity backed only by category-level grasp geometry."""

    def __init__(self, name: str, template: Any):
        self._name = name
        self.actor = template.actor
        self.config = template.config

    def get_name(self) -> str:
        return self._name


def load_and_validate_plan(path: str | Path) -> AgentPlan:
    """Load a JSON plan from disk and validate it before simulation starts."""
    plan_path = Path(path)
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanExecutionError(f"计划文件不存在：{plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanExecutionError(f"计划文件不是有效 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise PlanExecutionError("计划 JSON 顶层必须是对象")

    plan = AgentPlan.from_dict(data)
    AgentPlanner().validate(plan)
    return plan


class PlanExecutor:
    """Translate JSON plan steps into an allow-listed set of skill calls."""

    def __init__(
        self,
        scene: Any,
        *,
        skills: RobotSkills | None = None,
        objects: dict[str, Any] | None = None,
    ):
        self.scene = scene
        self.skills = skills or RobotSkills(scene)
        self._explicit_objects = objects is not None
        if objects is not None:
            self.objects = objects
        elif getattr(scene, "agent_vision_enabled", False):
            # Visual object identities are created lazily from detections.
            self.objects = {"box": scene.empty_box}
        else:
            # Retain physical actor names only for explicit legacy/non-visual
            # execution. Closed-loop Agent execution does not use this map.
            self.objects = {
                **scene._agent_actor_map(),
                "box": scene.empty_box,
            }
        self._visual_objects: dict[str, _VisualObjectReference] = {}
        self.last_outputs: dict[str, Any] = {}

    def execute(self, plan: AgentPlan) -> ExecutionReport:
        if plan.needs_clarification:
            raise PlanExecutionError(
                f"计划仍需用户澄清：{plan.clarification_question or '未提供问题'}"
            )

        self.last_outputs.clear()
        results: list[SkillResult] = []
        for index, step in enumerate(plan.steps, start=1):
            skill_name = step["skill"]
            arguments = self._resolve_arguments(
                skill_name,
                step.get("arguments", {}),
            )

            result = self._call_skill(skill_name, arguments)
            results.append(result)

            if not result.success:
                raise PlanExecutionError(
                    f"步骤 {index} 执行失败：{result.skill}；{result.message}"
                )
            self.last_outputs.update(result.data)

        return ExecutionReport(
            success=True,
            results=results,
            final_response=plan.final_response,
        )

    def _resolve_arguments(
        self,
        skill_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        parameter_refs: dict[str, str] = {}
        for name, value in arguments.items():
            if value == "$last.arm":
                if "arm" not in self.last_outputs:
                    raise PlanExecutionError("计划引用了 $last.arm，但此前没有机械臂输出")
                value = self.last_outputs["arm"]
            elif (
                name == "arm"
                and value is None
                and "arm" in self.last_outputs
            ):
                # Tolerate JSON null from a model as an implicit reference to
                # the arm selected by the preceding pick.
                value = self.last_outputs["arm"]

            if name in {"grasp_ref", "distance_ref", "drop_ref"}:
                if not isinstance(value, str) or not value:
                    raise PlanExecutionError(
                        f"{skill_name}.{name} 必须引用本轮视觉参数"
                    )
                parameter_refs[name] = value
            elif name == "object":
                resolved["actor"] = self._object(
                    value,
                    expected="object",
                    allow_unreliable=(skill_name == "observe_wide"),
                )
            elif name == "left_object":
                resolved["left_actor"] = self._object(value, expected="object")
            elif name == "right_object":
                resolved["right_actor"] = self._object(value, expected="object")
            elif name == "container":
                resolved["container"] = self._object(value, expected="container")
            else:
                resolved[name] = value
        self._resolve_parameter_refs(
            skill_name,
            resolved,
            parameter_refs,
        )
        return resolved

    def _resolve_parameter_refs(
        self,
        skill_name: str,
        resolved: dict[str, Any],
        parameter_refs: dict[str, str],
    ) -> None:
        if not parameter_refs:
            return
        registry = getattr(
            self.scene,
            "agent_skill_parameter_registry",
            {},
        )
        expected_ref = {
            "pick": ("grasp_ref", "grasp"),
            "lift": ("distance_ref", "lift"),
            "place_in": ("drop_ref", "drop"),
            "retreat": ("distance_ref", "retreat"),
        }.get(skill_name)
        if expected_ref is None:
            raise PlanExecutionError(
                f"{skill_name} 不接受视觉规划参数引用"
            )
        ref_name, expected_kind = expected_ref
        if set(parameter_refs) != {ref_name}:
            raise PlanExecutionError(
                f"{skill_name} 必须且只能提供 {ref_name}"
            )
        reference = parameter_refs[ref_name]
        entry = registry.get(reference)
        if not isinstance(entry, dict) or entry.get("kind") != expected_kind:
            raise PlanExecutionError(
                f"{reference} 不是本轮有效的 {expected_kind} 参数"
            )

        actor = resolved.get("actor")
        if actor is not None and entry.get("object") != actor.get_name():
            raise PlanExecutionError(
                f"{reference} 不属于目标 {actor.get_name()}"
            )
        requested_arm = str(resolved.get("arm"))
        if requested_arm not in {"None", "$last.arm"} and (
            entry.get("arm") != requested_arm
        ):
            raise PlanExecutionError(
                f"{reference} 只允许 {entry.get('arm')} 机械臂"
            )

        if expected_kind == "grasp":
            resolved["pre_grasp_distance"] = float(
                entry["pre_grasp_distance"]
            )
        elif expected_kind in {"lift", "retreat"}:
            if "distance" in resolved:
                raise PlanExecutionError(
                    f"{skill_name}不能同时提供distance和distance_ref"
                )
            resolved["distance"] = float(entry["distance"])
        else:
            container = resolved.get("container")
            if (
                container is None
                or entry.get("container") != "box"
            ):
                raise PlanExecutionError(
                    f"{reference} 不属于当前box"
                )
            resolved["target_pose"] = entry["target_pose"]

    def _object(
        self,
        name: str,
        *,
        expected: str,
        allow_unreliable: bool = False,
    ) -> Any:
        if expected == "container":
            if name != "box" or name not in self.objects:
                raise PlanExecutionError(f"{name} 不是可放置容器")
            return self.objects[name]

        if not AgentPlanner.is_visual_instance_name(name):
            raise PlanExecutionError(f"{name} 不是有效的视觉零件实例")

        if not getattr(self.scene, "agent_vision_enabled", False):
            if name not in self.objects:
                raise PlanExecutionError(f"场景中找不到对象：{name}")
            return self.objects[name]

        current_positions = getattr(
            self.scene,
            "agent_visual_object_positions",
            {},
        )
        held_names = {
            actor.get_name()
            for actor in self.skills.held_objects.values()
        }
        observed_names = {
            str(item.get("name"))
            for item in (
                getattr(self.scene, "_latest_agent_visual_state", None)
                or {}
            ).get("objects", [])
        }
        if (
            name not in current_positions
            and name not in held_names
            and not (allow_unreliable and name in observed_names)
        ):
            raise PlanExecutionError(f"最新视觉观察中找不到对象：{name}")
        if name not in self._visual_objects:
            template = self.scene.get_agent_visual_object_template(name)
            self._visual_objects[name] = _VisualObjectReference(name, template)
        return self._visual_objects[name]

    def _call_skill(self, skill_name: str, arguments: dict[str, Any]) -> SkillResult:
        # Explicit dispatch is intentional: JSON can only invoke these methods.
        dispatch = {
            "pick": self.skills.pick,
            "pick_head_camera": getattr(
                self.skills,
                "pick_head_camera",
                None,
            ),
            "pick_visual_asset": getattr(
                self.skills,
                "pick_visual_asset",
                None,
            ),
            "observe_wide": getattr(
                self.skills,
                "observe_wide",
                None,
            ),
            "lift": self.skills.lift,
            "place_in": self.skills.place_in,
            "retreat": self.skills.retreat,
            "move_home": self.skills.move_home,
            "pick_dual": self.skills.pick_dual,
            "lift_dual": self.skills.lift_dual,
            "place_in_dual": self.skills.place_in_dual,
            "retreat_dual": self.skills.retreat_dual,
            "move_home_dual": self.skills.move_home_dual,
        }
        try:
            skill = dispatch[skill_name]
        except KeyError as exc:
            raise PlanExecutionError(f"不允许执行未知技能：{skill_name}") from exc
        if skill is None:
            raise PlanExecutionError(f"当前技能实现缺少：{skill_name}")
        try:
            return skill(**arguments)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise PlanExecutionError(
                f"技能 {skill_name} 的参数无法执行：{exc}"
            ) from exc
