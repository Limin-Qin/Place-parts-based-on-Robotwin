"""LLM-backed planner that composes registered robot skills."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import ssl # new set by users
from dataclasses import dataclass
from typing import Any

try:
    from .robot_skills import RobotSkills
    from .scene_catalog import SCENE_CONVENTIONS, SCENE_OBJECTS
except ImportError:
    # Support running plan_command.py directly from this directory.
    from robot_skills import RobotSkills
    from scene_catalog import SCENE_CONVENTIONS, SCENE_OBJECTS


class PlanValidationError(ValueError):
    pass


@dataclass
class AgentGoal:
    """Structured task goal produced before closed-loop execution starts."""

    understood_goal: str
    target_category: str
    target_selector: str
    requested_arm: str
    target_objects: list[str]
    container: str
    needs_clarification: bool
    clarification_question: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentGoal":
        return cls(
            understood_goal=str(data.get("understood_goal", "")),
            target_category=str(data.get("target_category", "")),
            target_selector=str(data.get("target_selector", "all")),
            requested_arm=str(data.get("requested_arm", "auto")),
            target_objects=data.get("target_objects", []),
            container=str(data.get("container", "")),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "understood_goal": self.understood_goal,
            "target_category": self.target_category,
            "target_selector": self.target_selector,
            "requested_arm": self.requested_arm,
            "target_objects": self.target_objects,
            "container": self.container,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }


@dataclass
class AgentPlan:
    understood_goal: str
    needs_clarification: bool
    clarification_question: str | None
    steps: list[dict[str, Any]]
    final_response: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentPlan":
        final_response = str(
            data.get("final_response", "任务完成，还需要什么？")
        ).strip()
        if not final_response.endswith("还需要什么？"):
            final_response = (
                final_response.rstrip("。！？?!，, ")
                + "，还需要什么？"
            )
        return cls(
            understood_goal=str(data.get("understood_goal", "")),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question"),
            steps=data.get("steps", []),
            final_response=final_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "understood_goal": self.understood_goal,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "steps": self.steps,
            "final_response": self.final_response,
        }


class AgentPlanner:
    """Ask an LLM to compose skills, then validate the returned plan."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("AGENT_API_BASE", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("AGENT_API_KEY", "")
        self.model = model or os.getenv("AGENT_MODEL", "")
        self.skills = RobotSkills.schemas()
        self.objects = SCENE_OBJECTS
        self._wide_observation_attempted_objects: set[str] = set()

    def plan(self, user_text: str) -> AgentPlan:
        if not user_text.strip():
            raise ValueError("用户指令不能为空")
        data = self._request_json(
            self._system_prompt(),
            user_text,
            request_label="完整技能计划",
        )
        plan = AgentPlan.from_dict(data)
        self.validate(plan)
        return plan

    def understand_goal(self, user_text: str) -> AgentGoal:
        """Resolve language into target objects without planning robot motions."""
        if not user_text.strip():
            raise ValueError("用户指令不能为空")
        self._wide_observation_attempted_objects.clear()
        data = self._request_json(
            self._goal_prompt(),
            user_text,
            request_label="结构化目标",
        )
        goal = AgentGoal.from_dict(data)
        # Language understanding determines only category/scope. Instance
        # names and counts are bound later from head-camera detections.
        goal.target_objects = []
        self.validate_goal(goal)
        if not goal.needs_clarification:
            category_text = (
                "零件A"
                if goal.target_category == "part_A"
                else "零件B"
            )
            selector_text = {
                "all": "检测到的全部",
                "left": "检测到的左侧",
                "center": "检测到的中间",
                "right": "检测到的右侧",
            }[goal.target_selector]
            arm_text = {
                "auto": "",
                "left": "并使用左侧机械臂",
                "right": "并使用右侧机械臂",
            }[goal.requested_arm]
            # Do not echo a model-hallucinated numeric count to the runtime
            # trace before visual detection has actually happened.
            goal.understood_goal = (
                f"将head_camera{selector_text}{category_text}放入盒子"
                f"{arm_text}"
            )
        return goal

    def plan_next(
        self,
        goal: AgentGoal,
        scene_state: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> AgentPlan:
        """Build the next fixed single-arm transaction without an LLM call.

        The LLM remains responsible for understanding the user's language.
        Once vision has bound concrete instances, the runtime state already
        contains the only legal arm and parameter references, so asking the
        model to reproduce the same four-step transaction adds latency but no
        additional control information.
        """
        del history
        robot = scene_state.get("robot", {})
        held_by_arm = {
            str(arm): str(name)
            for arm, name in robot.get("held_objects", {}).items()
        }
        object_states = {
            str(item.get("name")): item
            for item in scene_state.get("objects", [])
        }

        if held_by_arm:
            arm, object_name = next(iter(held_by_arm.items()))
            parameters = object_states.get(object_name, {}).get(
                "skill_parameters", {}
            )
            drop_ref = parameters.get("place_in", {}).get("drop_ref")
            retreat_ref = parameters.get("retreat", {}).get("distance_ref")
            plan = AgentPlan.from_dict(
                {
                    "understood_goal": (
                        f"继续完成{arm}臂所持{object_name}的放置"
                    ),
                    "needs_clarification": False,
                    "clarification_question": None,
                    "steps": [
                        {
                            "skill": "place_in",
                            "arguments": {
                                "object": object_name,
                                "container": "box",
                                "arm": arm,
                                "drop_ref": drop_ref,
                            },
                            "reason": "恢复上轮中断的持物放置",
                        },
                        {
                            "skill": "retreat",
                            "arguments": {
                                "arm": arm,
                                "distance_ref": retreat_ref,
                            },
                            "reason": "放置后撤离容器",
                        },
                    ],
                    "final_response": "本轮恢复完成，还需要什么？",
                }
            )
        else:
            completed = set(scene_state.get("completed_objects", []))
            remaining = set(goal.target_objects) - completed
            selected = next(
                (
                    item
                    for item in scene_state.get("objects", [])
                    if item.get("name") in remaining
                    and not item.get("placed_in_box", False)
                    and isinstance(item.get("skill_parameters"), dict)
                ),
                None,
            )
            if selected is None:
                unreliable_targets = [
                    item
                    for item in scene_state.get("objects", [])
                    if item.get("name") in remaining
                    and not item.get("placed_in_box", False)
                    and item.get("position_reliable") is False
                ]
                if not unreliable_targets:
                    raise PlanValidationError(
                        "当前视觉状态中没有可执行或可宽展复核的未完成目标"
                    )
                selected_unreliable = next(
                    (
                        item
                        for item in unreliable_targets
                        if str(item.get("name"))
                        not in self._wide_observation_attempted_objects
                    ),
                    None,
                )
                if selected_unreliable is None:
                    attempted_names = sorted(
                        str(item.get("name"))
                        for item in unreliable_targets
                    )
                    raise PlanValidationError(
                        "以下目标分别经过宽展观察后位置仍不可靠："
                        f"{attempted_names}"
                    )
                object_name = str(selected_unreliable["name"])
                arm = selected_unreliable.get("recommended_arm")
                if arm not in {"left", "right"}:
                    raise PlanValidationError(
                        f"{object_name}缺少可用于宽展观察的目标侧机械臂"
                    )
                prefix: list[dict[str, Any]] = []
                if scene_state.get("safety", {}).get(
                    "workspace_clearance_recommended", False
                ):
                    for reset_arm in ("left", "right"):
                        if (
                            reset_arm != arm
                            and not robot.get(
                                f"{reset_arm}_arm_at_home", False
                            )
                        ):
                            prefix.append(
                                {
                                    "skill": "move_home",
                                    "arguments": {"arm": reset_arm},
                                    "reason": "宽展观察前清理另一机械臂的遮挡",
                                }
                            )
                plan = AgentPlan.from_dict(
                    {
                        "understood_goal": (
                            f"使用{arm}臂重新观察位置不可靠的"
                            f"{object_name}"
                        ),
                        "needs_clarification": False,
                        "clarification_question": None,
                        "steps": prefix
                        + [
                            {
                                "skill": "observe_wide",
                                "arguments": {
                                    "object": object_name,
                                    "arm": arm,
                                },
                                "reason": (
                                    "Agent已选择该目标，但其RGB-D位置不可靠，"
                                    "先移动目标侧机械臂扩大观察空间"
                                ),
                            }
                        ],
                        "final_response": "目标重新观察完成，还需要什么？",
                    }
                )
                self.validate(plan)
                self._validate_closed_loop_phase(
                    plan, goal, scene_state
                )
                self._wide_observation_attempted_objects.add(object_name)
                return plan

            object_name = str(selected["name"])
            parameters = selected["skill_parameters"]
            arm = parameters.get("pick", {}).get("arm")
            prefix: list[dict[str, Any]] = []
            if scene_state.get("safety", {}).get(
                "workspace_clearance_recommended", False
            ):
                for reset_arm in ("left", "right"):
                    if not robot.get(f"{reset_arm}_arm_at_home", False):
                        prefix.append(
                            {
                                "skill": "move_home",
                                "arguments": {"arm": reset_arm},
                                "reason": "抓取前清理机械臂工作空间",
                            }
                        )

            plan = AgentPlan.from_dict(
                {
                    "understood_goal": (
                        f"使用{arm}臂将{object_name}放入盒子"
                    ),
                    "needs_clarification": False,
                    "clarification_question": None,
                    "steps": prefix
                    + [
                        {
                            "skill": "pick_visual_asset",
                            "arguments": {
                                "object": object_name,
                                "arm": arm,
                            },
                            "reason": (
                                "使用head_camera RGB-D坐标、固定物体姿态和"
                                "资产接触矩阵执行抓取"
                            ),
                        },
                        {
                            "skill": "lift",
                            "arguments": {
                                "arm": arm,
                                "distance_ref": parameters.get(
                                    "lift", {}
                                ).get("distance_ref"),
                            },
                            "reason": "抓取后抬离桌面",
                        },
                        {
                            "skill": "place_in",
                            "arguments": {
                                "object": object_name,
                                "container": "box",
                                "arm": arm,
                                "drop_ref": parameters.get(
                                    "place_in", {}
                                ).get("drop_ref"),
                            },
                            "reason": "放入视觉预选的盒内空闲区域",
                        },
                        {
                            "skill": "retreat",
                            "arguments": {
                                "arm": arm,
                                "distance_ref": parameters.get(
                                    "retreat", {}
                                ).get("distance_ref"),
                            },
                            "reason": "放置后撤离容器",
                        },
                    ],
                    "final_response": "本轮搬运完成，还需要什么？",
                }
            )

        self.validate(plan)
        self._validate_closed_loop_phase(plan, goal, scene_state)
        return plan

    def validate_goal(self, goal: AgentGoal) -> None:
        if goal.needs_clarification:
            if (
                not goal.clarification_question
                or goal.target_objects
                or goal.target_category
            ):
                raise PlanValidationError(
                    "请求澄清时必须提供问题，且视觉目标必须为空"
                )
            return
        if not goal.understood_goal:
            raise PlanValidationError("结构化目标缺少 understood_goal")
        if goal.target_category not in {"part_A", "part_B"}:
            raise PlanValidationError(
                f"结构化目标包含未知零件类别：{goal.target_category}"
            )
        if goal.target_selector not in {"all", "left", "center", "right"}:
            raise PlanValidationError(
                f"结构化目标包含未知位置范围：{goal.target_selector}"
            )
        if goal.requested_arm not in {"auto", "left", "right"}:
            raise PlanValidationError(
                f"结构化目标包含未知机械臂要求：{goal.requested_arm}"
            )
        if (
            not isinstance(goal.target_objects, list)
            or len(goal.target_objects) != len(set(goal.target_objects))
            or any(
                not self.is_visual_instance_name(name)
                for name in goal.target_objects
            )
        ):
            raise PlanValidationError(
                f"结构化目标包含无效或重复物体：{goal.target_objects}"
            )
        if goal.container != "box":
            raise PlanValidationError(f"结构化目标使用未知容器：{goal.container}")

    def _request_json(
        self,
        system_prompt: str,
        user_content: str,
        *,
        request_label: str,
    ) -> dict[str, Any]:
        self._check_configuration()
        timeout = self._positive_env_number("AGENT_API_TIMEOUT", 60.0)
        max_attempts = self._positive_env_integer("AGENT_API_RETRIES", 3)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response_data: dict[str, Any] | None = None
        last_error: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            print(
                f"正在请求大模型生成{request_label}"
                f"（第 {attempt}/{max_attempts} 次，超时 {timeout:g} 秒）……",
                flush=True,
            )
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=request_data,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout,context=ssl._create_unverified_context()
                ) as response:
                    response_data = json.loads(
                        response.read().decode("utf-8")
                    )
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"模型服务返回 HTTP {exc.code}: {detail}"
                    ) from exc
                last_error = RuntimeError(
                    f"模型服务暂时不可用，HTTP {exc.code}: {detail}"
                )
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc

            if attempt < max_attempts:
                wait_seconds = min(2 ** attempt, 8)
                print(
                    f"模型请求失败：{self._connection_error_text(last_error)}；"
                    f"{wait_seconds} 秒后自动重试。",
                    flush=True,
                )
                time.sleep(wait_seconds)

        if response_data is None:
            raise RuntimeError(
                f"生成{request_label}失败：模型服务连续 {max_attempts} 次"
                f"连接失败或超时（单次超时 {timeout:g} 秒）。"
                f"最后错误：{self._connection_error_text(last_error)}。"
                "这是模型API或网络连接问题，机器人尚未执行本轮动作。"
            ) from last_error

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型响应格式不兼容：{response_data}") from exc

        return self._extract_json(content)

    @staticmethod
    def _positive_env_number(name: str, default: float) -> float:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{name} 必须是正数，当前值：{raw_value!r}") from exc
        if value <= 0:
            raise RuntimeError(f"{name} 必须大于 0，当前值：{raw_value!r}")
        return value

    @staticmethod
    def _positive_env_integer(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} 必须是正整数，当前值：{raw_value!r}"
            ) from exc
        if value <= 0:
            raise RuntimeError(
                f"{name} 必须大于 0，当前值：{raw_value!r}"
            )
        return value

    @staticmethod
    def _connection_error_text(error: BaseException | None) -> str:
        if error is None:
            return "未知连接错误"
        if isinstance(error, urllib.error.URLError):
            return str(error.reason)
        return str(error)

    def _check_configuration(self) -> None:
        if not self.base_url or not self.model:
            raise RuntimeError(
                "缺少模型配置。请设置 AGENT_API_BASE 和 AGENT_MODEL；"
                "需要鉴权时再设置 AGENT_API_KEY。"
            )
        placeholder_values = ("你的模型", "your-model", "模型名称")
        if any(value in self.model.lower() for value in placeholder_values):
            raise RuntimeError(
                f"AGENT_MODEL 当前仍是示例占位值：{self.model!r}。"
                "请替换为模型服务实际提供的模型 ID。"
            )
        if self.api_key and any(
            value in self.api_key.lower()
            for value in ("你的密钥", "your-key", "api密钥")
        ):
            raise RuntimeError(
                "AGENT_API_KEY 当前仍是示例占位值，请替换为真实密钥；"
                "本地服务不需要鉴权时执行：unset AGENT_API_KEY"
            )
        if self.api_key:
            try:
                self.api_key.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise RuntimeError(
                    "AGENT_API_KEY 不能包含中文等非 HTTP Header 字符。"
                    "请设置真实 API 密钥，或对无需鉴权的本地服务执行："
                    "unset AGENT_API_KEY"
                ) from exc

    def validate(self, plan: AgentPlan) -> None:
        if plan.needs_clarification:
            if not plan.clarification_question or plan.steps:
                raise PlanValidationError("请求澄清时必须提供问题，且 steps 必须为空")
            return

        if not plan.understood_goal:
            raise PlanValidationError("计划缺少 understood_goal")
        if not plan.final_response.endswith("还需要什么？"):
            raise PlanValidationError("final_response 必须以“还需要什么？”结尾")
        if not isinstance(plan.steps, list) or not plan.steps:
            raise PlanValidationError("可执行计划必须包含至少一个步骤")
        if len(plan.steps) > 30:
            raise PlanValidationError("计划步骤过多")

        schemas = {skill["name"]: skill for skill in self.skills}
        last_arm_available = False

        for index, step in enumerate(plan.steps, start=1):
            if not isinstance(step, dict):
                raise PlanValidationError(f"步骤 {index} 不是对象")
            skill_name = step.get("skill")
            arguments = step.get("arguments", {})
            if skill_name not in schemas:
                raise PlanValidationError(f"步骤 {index} 使用未知技能：{skill_name}")
            if not isinstance(arguments, dict):
                raise PlanValidationError(f"步骤 {index} 的 arguments 必须是对象")

            schema = schemas[skill_name]
            allowed = set(schema["required"]) | set(schema["optional"])
            missing = [name for name in schema["required"] if name not in arguments]
            unknown = set(arguments) - allowed
            if missing:
                raise PlanValidationError(f"步骤 {index} 缺少参数：{missing}")
            if unknown:
                raise PlanValidationError(f"步骤 {index} 包含未知参数：{sorted(unknown)}")

            for object_argument in ("object", "left_object", "right_object"):
                if object_argument not in arguments:
                    continue
                name = arguments[object_argument]
                if (
                    not isinstance(name, str)
                    or not self.is_visual_instance_name(name)
                ):
                    raise PlanValidationError(f"步骤 {index} 使用未知可抓取物体：{name}")
            if "container" in arguments:
                name = arguments["container"]
                if name != "box":
                    raise PlanValidationError(f"步骤 {index} 使用未知容器：{name}")
            if "arm" in arguments:
                arm = arguments["arm"]
                if arm == "$last.arm" and not last_arm_available:
                    raise PlanValidationError(f"步骤 {index} 在 pick 之前引用 $last.arm")
                # Some OpenAI-compatible models serialize an omitted optional
                # arm as JSON null. For pick this still means runtime visual
                # arm selection; after pick it means the arm returned by pick.
                null_arm_allowed = (
                    arm is None
                    and (
                        skill_name
                        in {"pick", "pick_head_camera", "pick_visual_asset"}
                        or last_arm_available
                    )
                )
                if not null_arm_allowed and (
                    not isinstance(arm, str)
                    or arm not in {"left", "right", "$last.arm"}
                ):
                    raise PlanValidationError(f"步骤 {index} 的 arm 非法：{arm}")
            if "distance" in arguments:
                distance = arguments["distance"]
                max_distance = (
                    0.15
                    if skill_name in {"lift", "lift_dual"}
                    else 0.12
                    if skill_name in {"retreat", "retreat_dual"}
                    else 0.5
                )
                if (
                    isinstance(distance, bool)
                    or not isinstance(distance, (int, float))
                    or not 0 < distance <= max_distance
                ):
                    raise PlanValidationError(
                        f"步骤 {index} 的 {skill_name}.distance 必须在 "
                        f"(0, {max_distance}] 米范围内"
                    )
            if skill_name in {"pick_dual", "place_in_dual"}:
                left_object = arguments["left_object"]
                right_object = arguments["right_object"]
                if left_object == right_object:
                    raise PlanValidationError(
                        f"步骤 {index} 的双臂不能操作同一个物体："
                        f"{left_object}"
                    )

            if skill_name in {
                "pick",
                "pick_head_camera",
                "pick_visual_asset",
            }:
                last_arm_available = True

    @staticmethod
    def is_visual_instance_name(name: Any) -> bool:
        """Accept any positive runtime instance ID discovered by vision."""
        if not isinstance(name, str):
            return False
        for category in ("part_A", "part_B"):
            prefix = f"{category}_"
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                return suffix.isdigit() and int(suffix) > 0
        return False

    def _validate_closed_loop_phase(
        self,
        plan: AgentPlan,
        goal: AgentGoal,
        scene_state: dict[str, Any],
    ) -> None:
        completed = set(scene_state.get("completed_objects", []))
        remaining = set(goal.target_objects) - completed
        robot = scene_state.get("robot", {})
        held_by_arm = {
            str(arm): str(name)
            for arm, name in robot.get("held_objects", {}).items()
        }
        object_states = {
            str(item.get("name")): item
            for item in scene_state.get("objects", [])
        }

        # A held object only exists after an interrupted/failed transaction.
        # Recover with one place+retreat transaction before selecting another.
        if held_by_arm:
            action_steps = [
                step for step in plan.steps
                if step["skill"] not in {"move_home", "move_home_dual"}
            ]
            if [step["skill"] for step in action_steps] != [
                "place_in",
                "retreat",
            ]:
                raise PlanValidationError(
                    "当前机械臂持物，只允许place_in→retreat完成恢复"
                )
            place_args = action_steps[0]["arguments"]
            object_name = place_args["object"]
            expected_arm = next(
                (
                    arm for arm, held_name in held_by_arm.items()
                    if held_name == object_name
                ),
                None,
            )
            if expected_arm is None:
                raise PlanValidationError(
                    f"{object_name}当前没有被机械臂持有"
                )
            if (
                place_args.get("arm") != expected_arm
                or action_steps[1]["arguments"].get("arm")
                != expected_arm
            ):
                raise PlanValidationError(
                    f"持有{object_name}的是{expected_arm}臂，"
                    "放置和撤离必须使用同一机械臂"
                )
            refs = object_states.get(object_name, {}).get(
                "skill_parameters", {}
            )
            if (
                place_args.get("drop_ref")
                != refs.get("place_in", {}).get("drop_ref")
                or action_steps[1]["arguments"].get("distance_ref")
                != refs.get("retreat", {}).get("distance_ref")
            ):
                raise PlanValidationError(
                    "持物恢复必须引用本轮视觉提供的drop_ref和retreat distance_ref"
                )
            return

        non_home_steps = [
            step
            for step in plan.steps
            if step["skill"] not in {"move_home", "move_home_dual"}
        ]
        if [step["skill"] for step in non_home_steps] == [
            "observe_wide"
        ]:
            observe_args = non_home_steps[0]["arguments"]
            object_name = observe_args.get("object")
            arm = observe_args.get("arm")
            object_state = object_states.get(str(object_name), {})
            if object_name not in remaining:
                raise PlanValidationError(
                    "宽展观察目标不是尚未完成的目标"
                )
            if object_state.get("position_reliable") is not False:
                raise PlanValidationError(
                    "只有位置不可靠的目标允许执行宽展观察"
                )
            if (
                arm not in {"left", "right"}
                or arm != object_state.get("recommended_arm")
            ):
                raise PlanValidationError(
                    "宽展观察必须使用所选目标对应的机械臂"
                )
            if any(
                step["skill"] not in {"move_home", "move_home_dual"}
                for step in plan.steps[:-1]
            ):
                raise PlanValidationError(
                    "宽展观察前只允许必要的机械臂归位动作"
                )
            return

        first_pick_index = next(
            (
                index for index, step in enumerate(plan.steps)
                if step["skill"]
                in {"pick", "pick_head_camera", "pick_visual_asset"}
            ),
            None,
        )
        if first_pick_index is None:
            raise PlanValidationError(
                "当前无持物，本轮必须选择一个零件执行完整单臂搬运"
            )
        prefix = plan.steps[:first_pick_index]
        if any(
            step["skill"] not in {"move_home", "move_home_dual"}
            for step in prefix
        ):
            raise PlanValidationError("抓取前只允许必要的机械臂归位动作")
        transaction = plan.steps[first_pick_index:]
        pick_skill = transaction[0]["skill"]
        expected_skills = [pick_skill, "lift", "place_in", "retreat"]
        if pick_skill not in {
            "pick",
            "pick_head_camera",
            "pick_visual_asset",
        }:
            raise PlanValidationError("本轮首个抓取步骤不是已注册的单臂pick技能")
        if [step["skill"] for step in transaction] != expected_skills:
            raise PlanValidationError(
                "每轮必须完整执行单个零件的"
                "pick→lift→place_in→retreat，不能拆分阶段或使用双臂"
            )

        pick_args = transaction[0]["arguments"]
        place_args = transaction[2]["arguments"]
        object_name = pick_args["object"]
        arm = pick_args.get("arm")
        if object_name not in remaining:
            raise PlanValidationError(
                f"本轮选择的{object_name}不是尚未完成的目标"
            )
        if arm not in {"left", "right"}:
            raise PlanValidationError(
                "Agent必须根据当前视觉位置明确选择left或right机械臂"
            )
        if place_args.get("object") != object_name:
            raise PlanValidationError("抓取和放置必须操作同一个零件")
        if place_args.get("container") != "box":
            raise PlanValidationError("目标容器必须是box")
        for step in transaction[1:]:
            step_arm = step["arguments"].get("arm")
            if step_arm not in {arm, "$last.arm"}:
                raise PlanValidationError(
                    "抓取、抬升、放置和撤离必须使用同一机械臂"
                )

        parameter_contract = object_states.get(object_name, {}).get(
            "skill_parameters"
        )
        if not isinstance(parameter_contract, dict):
            raise PlanValidationError(
                f"{object_name}缺少本轮视觉和规划器生成的合法技能参数"
            )
        required_refs = [
            (
                transaction[1]["arguments"].get("distance_ref"),
                parameter_contract.get("lift", {}).get("distance_ref"),
                "lift distance_ref",
            ),
            (
                transaction[2]["arguments"].get("drop_ref"),
                parameter_contract.get("place_in", {}).get("drop_ref"),
                "drop_ref",
            ),
            (
                transaction[3]["arguments"].get("distance_ref"),
                parameter_contract.get("retreat", {}).get("distance_ref"),
                "retreat distance_ref",
            ),
        ]
        if pick_skill == "pick":
            required_refs.insert(
                0,
                (
                    transaction[0]["arguments"].get("grasp_ref"),
                    parameter_contract.get("pick", {}).get("grasp_ref"),
                    "grasp_ref",
                ),
            )
        for supplied, expected, label in required_refs:
            if not isinstance(expected, str) or supplied != expected:
                raise PlanValidationError(
                    f"{object_name}的{label}必须原样引用本轮观测提供的值"
                )
        if (
            pick_args.get("arm")
            != parameter_contract.get("pick", {}).get("arm")
        ):
            raise PlanValidationError(
                f"{object_name}必须使用视觉工作区推荐的机械臂"
            )
        if (
            "distance" in transaction[1]["arguments"]
            or "distance" in transaction[3]["arguments"]
        ):
            raise PlanValidationError(
                "闭环Agent不能自行填写运动距离，只能使用distance_ref"
            )

        safety = scene_state.get("safety", {})
        if safety.get(
            "workspace_clearance_recommended", False
        ):
            reset_steps = prefix
            reset_both = any(
                step["skill"] == "move_home_dual" for step in reset_steps
            )
            reset_arms = {
                step.get("arguments", {}).get("arm")
                for step in reset_steps
                if step["skill"] == "move_home"
            }
            required_arms = {
                arm
                for arm in ("left", "right")
                if not robot.get(f"{arm}_arm_at_home", False)
            }
            if not reset_both and not required_arms.issubset(reset_arms):
                raise PlanValidationError(
                    "当前环境要求先清理工作空间；抓取前必须将未归位机械臂"
                    f"归位：{sorted(required_arms)}"
                )

    def _system_prompt(self) -> str:
        output_contract = {
            "understood_goal": "用中文简述理解到的目标",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "只能使用技能清单中的 name",
                    "arguments": {"参数名": "参数值"},
                    "reason": "选择本步骤的简短原因",
                }
            ],
            "final_response": "任务成功后对用户说的话，必须以“还需要什么？”结尾",
        }
        return (
            "你是机器人任务规划 Agent。你必须根据用户指令、场景物体和技能清单，"
            "自行选择并编排基础技能。没有固定任务模板。你只生成高层技能计划，"
            "不能生成 Python、关节角或未注册技能。\n\n"
            f"场景物体：\n{json.dumps(self.objects, ensure_ascii=False, indent=2)}\n\n"
            f"可用基础技能：\n{json.dumps(self.skills, ensure_ascii=False, indent=2)}\n\n"
            f"场景约定：\n{json.dumps(SCENE_CONVENTIONS, ensure_ascii=False, indent=2)}\n\n"
            "pick 未指定 arm 时由运行时根据物体位置自动选臂。pick 的输出 arm 可在后续"
            "步骤中写成 $last.arm。单臂搬运的必要步骤按 pick、lift、place_in、retreat "
            "编排；双臂搬运的必要步骤按 pick_dual、lift_dual、place_in_dual、"
            "retreat_dual 编排。放置并撤离后机械臂已经可以继续下一个任务，"
            "move_home 和 move_home_dual 是可选技能，只在任务结束或后续动作确有需要时"
            "使用，不要默认在每个物体放置后插入。lift 和 lift_dual 建议使用默认的 "
            "0.10 米安全距离。处理多个物体时，由你根据物体位置决定"
            "使用多个单臂流程，还是对左右两件使用一次双臂流程；不能创造未注册技能。"
            "每个目标物体只能处理一次。place_in 不接受目标位置参数；"
            "机器人技能会在执行时根据容器最新占用状态自动选择空位。"
            "若信息不足，设置 needs_clarification=true、steps=[] 并提出一个问题。\n\n"
            f"严格只输出一个 JSON 对象，不要 Markdown：\n"
            f"{json.dumps(output_contract, ensure_ascii=False, indent=2)}"
        )

    def _goal_prompt(self) -> str:
        contract = {
            "understood_goal": "用中文简述用户真正要求完成的目标",
            "target_category": "part_A或part_B；未知时为空字符串",
            "target_selector": "all、left、center或right",
            "requested_arm": "auto、left或right",
            "container": "box",
            "needs_clarification": False,
            "clarification_question": None,
        }
        category_catalog = [
            {
                "category": "part_A",
                "aliases": ["零件A", "A类零件"],
                "instance_count": "由head_camera视觉检测决定",
            },
            {
                "category": "part_B",
                "aliases": ["零件B", "B类零件"],
                "instance_count": "由head_camera视觉检测决定",
            },
            {
                "category": "container",
                "name": "box",
                "aliases": ["盒子", "料盒", "容器", "塑料框"],
            },
        ]
        return (
            "你是具身 Agent 的语言理解模块。这里只理解目标，不规划机器人动作。"
            "根据用户文字只确定零件类别和位置范围，绝不能猜测画面中的实例数量，"
            "也不能生成part_A_1之类的实例名称。实例名称和数量稍后完全由"
            "head_camera的YOLO检测结果产生。用户未指定位置或数量时，"
            "target_selector=all；指定左、中、右时分别输出left、center、right。"
            "用户明确要求使用左侧机械臂或左臂时requested_arm=left；明确要求"
            "使用右侧机械臂或右臂时requested_arm=right；未指定机械臂时"
            "requested_arm=auto。requested_arm描述机械臂，target_selector描述"
            "零件在画面中的位置，两者不能混淆。"
            "用户没有说明A或B时必须请求澄清；此时target_category为空字符串。"
            f"\n\n类别清单：\n{json.dumps(category_catalog, ensure_ascii=False, indent=2)}"
            f"\n\n场景约定：\n{json.dumps(SCENE_CONVENTIONS, ensure_ascii=False, indent=2)}"
            "\n\n严格只输出一个 JSON 对象，不要 Markdown：\n"
            f"{json.dumps(contract, ensure_ascii=False, indent=2)}"
        )

    def _closed_loop_prompt(self) -> str:
        output_contract = {
            "understood_goal": "说明本轮根据最新环境状态要完成什么",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick",
                    "arguments": {
                        "object": "选择的视觉实例名",
                        "arm": "该实例skill_parameters.pick.arm",
                        "grasp_ref": "该实例skill_parameters.pick.grasp_ref",
                    },
                    "reason": "说明选择目标和机械臂的原因",
                },
                {
                    "skill": "lift",
                    "arguments": {
                        "arm": "与pick相同",
                        "distance_ref": "该实例skill_parameters.lift.distance_ref",
                    },
                    "reason": "说明为何抬离桌面",
                },
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "与pick相同",
                        "container": "box",
                        "arm": "与pick相同",
                        "drop_ref": "该实例skill_parameters.place_in.drop_ref",
                    },
                    "reason": "说明使用本轮预选空闲区域",
                },
                {
                    "skill": "retreat",
                    "arguments": {
                        "arm": "与pick相同",
                        "distance_ref": "该实例skill_parameters.retreat.distance_ref",
                    },
                    "reason": "说明放置后撤离",
                },
            ],
            "final_response": "整个目标完成后对用户说的话，必须以“还需要什么？”结尾",
        }
        closed_loop_skills = [
            skill
            for skill in self.skills
            if skill["name"] not in {
                "pick_dual",
                "lift_dual",
                "place_in_dual",
                "retreat_dual",
                "move_home_dual",
            }
        ]
        return (
            "你是闭环机器人任务规划 Agent。每轮只选择一个尚未完成的零件和一只"
            "机械臂，并由你在同一轮完整编排pick→lift→place_in→retreat四步。"
            "不得拆成抓取轮和放置轮，不得使用任何双臂技能。四步必须操作同一个零件、"
            "同一只明确的left或right机械臂；place_in.container必须为box。"
            "你必须从所选物体的objects[].skill_parameters中原样复制合法参数引用："
            "pick使用grasp_ref，lift使用distance_ref，place_in使用drop_ref，"
            "retreat使用distance_ref。禁止自己生成坐标或距离，也禁止引用其他物体或"
            "旧观察中的参数。执行器会把这些ref解析成RGB-D和运动约束产生的真实参数。"
            "planned_drop_xyz只用于解释当前预选空闲区，不得直接写进技能参数。"
            "完整放置并撤离后，控制器才重新观察并规划下一个零件。"
            "只有上轮异常中断且robot.held_objects非空时，才只生成"
            "place_in→retreat完成恢复。"
            "latest_visual_robot_state 是本轮唯一可信的最新状态：物体位置来自head_camera "
            "RGB-D，机器人状态来自本体传感器；已经完成的物体不得再次操作。"
            "当 safety.workspace_clearance_recommended=true 时，抓取前必须先用 "
            "move_home 将所有未归位机械臂移出工作区；否则根据当前状态"
            "自行判断是否需要归位。这个安全动作由状态触发，不要机械地插入每一轮。"
            "pick.arm必须原样使用skill_parameters.pick.arm；后续三步使用相同机械臂"
            "名称。place_in不接受Agent生成的目标坐标；执行层通过drop_ref使用抓取前"
            "预先计算的空闲点。lift和retreat不得直接填写distance。"
            "执行层会阻止机械臂跨越中线，共享中央工作区可结合recommended_arm选择。"
            "不能生成关节角、Python 或未注册技能。"
            "不要假设零件类别的固定实例数量；可操作实例只能来自"
            "latest_visual_robot_state.objects。"
            "previous_phases中execution_success=false或visual_progress=false表示"
            "上一轮虽然可能完成了机械臂运动，但视觉没有确认零件进入盒子；"
            "不得把它当成成功，也不要原样重复完全相同的物体、机械臂和技能方案。"
            "应结合最新位置、安全可达性和失败原因选择其他安全方案；若没有其他"
            "可行方案，控制器会在有限次尝试后安全停止。"
            f"\n\n可用技能：\n{json.dumps(closed_loop_skills, ensure_ascii=False, indent=2)}"
            f"\n\n场景约定：\n{json.dumps(SCENE_CONVENTIONS, ensure_ascii=False, indent=2)}"
            "\n\n严格只输出一个 JSON 对象，不要 Markdown：\n"
            f"{json.dumps(output_contract, ensure_ascii=False, indent=2)}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise PlanValidationError("模型没有返回 JSON 对象")
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise PlanValidationError("模型输出的顶层必须是 JSON 对象")
        return data
