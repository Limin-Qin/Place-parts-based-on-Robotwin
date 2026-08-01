"""CLI for testing text-to-skill planning without RoboTwin or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner import AgentGoal, AgentPlan, AgentPlanner, PlanValidationError
from plan_executor import PlanExecutor, load_and_validate_plan
from robot_skills import SkillResult


AGENT_DIR = Path(__file__).resolve().parent


def run_self_test() -> None:
    planner = AgentPlanner()
    valid = AgentPlan.from_dict(
        {
            "understood_goal": "把左侧零件A放进盒子",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {"skill": "pick", "arguments": {"object": "part_A_1"}},
                {"skill": "lift", "arguments": {"arm": "$last.arm", "distance": 0.1}},
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "part_A_1",
                        "container": "box",
                        "arm": "$last.arm",
                    },
                },
                {"skill": "retreat", "arguments": {"arm": "$last.arm"}},
            ],
            "final_response": "任务完成，还需要什么？",
        }
    )
    planner.validate(valid)
    null_arm_data = json.loads(json.dumps(valid.to_dict()))
    for step in null_arm_data["steps"]:
        step["arguments"]["arm"] = None
    null_arm_plan = AgentPlan.from_dict(null_arm_data)
    planner.validate(null_arm_plan)

    goal = AgentGoal.from_dict(
        {
            "understood_goal": "把视觉检测到的全部零件A放进盒子",
            "target_category": "part_A",
            "target_selector": "all",
            "target_objects": [],
            "container": "box",
            "needs_clarification": False,
            "clarification_question": None,
        }
    )
    planner.validate_goal(goal)
    goal.target_objects = ["part_A_1", "part_A_2", "part_A_3"]
    b_goal = AgentGoal.from_dict(
        {
            "understood_goal": "把视觉检测到的全部零件B放进盒子",
            "target_category": "part_B",
            "target_selector": "all",
            "target_objects": [],
            "container": "box",
            "needs_clarification": False,
            "clarification_question": None,
        }
    )
    planner.validate_goal(b_goal)
    b_goal.target_objects = ["part_B_1", "part_B_2"]
    closed_loop_state = {
        "completed_objects": [],
        "remaining_objects": ["part_A_1", "part_A_2", "part_A_3"],
        "objects": [
            {
                "name": "part_A_1",
                "arm_workspace": "left",
                "recommended_arm": "left",
                "planned_drop_xyz": [0.0, -0.14, 0.78],
                "skill_parameters": {
                    "pick": {
                        "arm": "left",
                        "grasp_ref": "obs1:part_A_1:grasp",
                    },
                    "lift": {
                        "distance_ref": "obs1:part_A_1:lift",
                    },
                    "place_in": {
                        "drop_ref": "obs1:part_A_1:drop",
                    },
                    "retreat": {
                        "distance_ref": "obs1:part_A_1:retreat",
                    },
                },
            },
            {
                "name": "part_A_2",
                "arm_workspace": "shared",
                "recommended_arm": "right",
                "planned_drop_xyz": [0.04, -0.14, 0.78],
            },
            {
                "name": "part_A_3",
                "arm_workspace": "right",
                "recommended_arm": "right",
                "planned_drop_xyz": [-0.04, -0.14, 0.78],
            },
        ],
        "robot": {
            "held_objects": {},
            "left_arm_at_home": True,
            "right_arm_at_home": True,
        },
        "safety": {},
    }

    full_transaction = AgentPlan.from_dict(
        {
            "understood_goal": "完整搬运一个零件后重新观察",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick",
                    "arguments": {
                        "object": "part_A_1",
                        "arm": "left",
                        "grasp_ref": "obs1:part_A_1:grasp",
                    },
                },
                {
                    "skill": "lift",
                    "arguments": {
                        "arm": "left",
                        "distance_ref": "obs1:part_A_1:lift",
                    },
                },
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "part_A_1",
                        "container": "box",
                        "arm": "left",
                        "drop_ref": "obs1:part_A_1:drop",
                    },
                },
                {
                    "skill": "retreat",
                    "arguments": {
                        "arm": "left",
                        "distance_ref": "obs1:part_A_1:retreat",
                    },
                },
            ],
            # Also check that a harmless model formatting omission is
            # normalized instead of terminating the robot task.
            "final_response": "继续执行",
        }
    )
    planner.validate(full_transaction)
    planner._validate_closed_loop_phase(
        full_transaction,
        goal,
        closed_loop_state,
    )
    if not full_transaction.final_response.endswith("还需要什么？"):
        raise AssertionError("final_response 没有被自动规范化")

    split_pick_phase = AgentPlan.from_dict(
        {
            "understood_goal": "错误地把抓取和放置拆成两轮",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick",
                    "arguments": {"object": "part_A_1", "arm": "left"},
                },
                {
                    "skill": "lift",
                    "arguments": {"arm": "left", "distance": 0.1},
                },
            ],
            "final_response": "继续执行，还需要什么？",
        }
    )
    planner.validate(split_pick_phase)
    try:
        planner._validate_closed_loop_phase(
            split_pick_phase,
            goal,
            closed_loop_state,
        )
    except PlanValidationError:
        pass
    else:
        raise AssertionError("闭环校验器没有拒绝拆分的抓取阶段")

    held_state = {
        **closed_loop_state,
        "robot": {
            "held_objects": {
                "left": "part_A_1",
            },
            "left_arm_at_home": False,
            "right_arm_at_home": True,
        },
    }
    single_place_phase = AgentPlan.from_dict(
        {
            "understood_goal": "先单独放置左臂零件并重新观察",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "part_A_1",
                        "container": "box",
                        "arm": "left",
                        "drop_ref": "obs1:part_A_1:drop",
                    },
                },
                {
                    "skill": "retreat",
                    "arguments": {
                        "arm": "left",
                        "distance_ref": "obs1:part_A_1:retreat",
                    },
                },
            ],
            "final_response": "继续执行，还需要什么？",
        }
    )
    planner.validate(single_place_phase)
    planner._validate_closed_loop_phase(
        single_place_phase,
        goal,
        held_state,
    )

    dual_phase = AgentPlan.from_dict(
        {
            "understood_goal": "错误地使用双臂处理两个零件",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick_dual",
                    "arguments": {
                        "left_object": "part_A_1",
                        "right_object": "part_A_3",
                    },
                },
                {"skill": "lift_dual", "arguments": {"distance": 0.1}},
                {
                    "skill": "place_in_dual",
                    "arguments": {
                        "left_object": "part_A_1",
                        "right_object": "part_A_3",
                        "container": "box",
                    },
                },
                {"skill": "retreat_dual", "arguments": {"distance": 0.08}},
            ],
            "final_response": "任务完成，还需要什么？",
        }
    )
    planner.validate(dual_phase)
    try:
        planner._validate_closed_loop_phase(
            dual_phase,
            goal,
            closed_loop_state,
        )
    except PlanValidationError:
        pass
    else:
        raise AssertionError("闭环校验器没有拒绝一次放置两个零件")

    wrong_reference_data = json.loads(
        json.dumps(full_transaction.to_dict())
    )
    wrong_reference_data["steps"][2]["arguments"]["drop_ref"] = (
        "obs0:part_A_1:invented"
    )
    wrong_reference = AgentPlan.from_dict(wrong_reference_data)
    planner.validate(wrong_reference)
    try:
        planner._validate_closed_loop_phase(
            wrong_reference,
            goal,
            closed_loop_state,
        )
    except PlanValidationError:
        pass
    else:
        raise AssertionError("闭环校验器没有拒绝Agent编造的参数引用")

    invalid = AgentPlan.from_dict(
        {
            "understood_goal": "使用不存在的物体",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [{"skill": "pick", "arguments": {"object": "apple"}}],
            "final_response": "完成",
        }
    )
    try:
        planner.validate(invalid)
    except PlanValidationError:
        pass
    else:
        raise AssertionError("校验器没有拒绝未知物体")

    print(
        "Agent规划器离线自检通过：目标理解、闭环阶段、技能契约、"
        "对象校验和变量引用均正常。"
    )


class _FakeSkills:
    """No-simulator skills used only to check JSON-to-call translation."""

    def __init__(self):
        self.calls: list[str] = []

    def _ok(self, skill: str, arm: str, **data) -> SkillResult:
        self.calls.append(skill)
        return SkillResult(True, skill, "离线模拟成功", {"arm": arm, **data})

    def pick(self, actor, arm=None, pre_grasp_distance=0.1):
        selected = arm or ("left" if actor == "fake_part_1" else "right")
        return self._ok(
            "pick",
            selected,
            object=actor,
            pre_grasp_distance=pre_grasp_distance,
        )

    def lift(self, arm, distance=0.1):
        return self._ok("lift", arm, distance=distance)

    def place_in(self, actor, container, arm, target_pose=None):
        return self._ok(
            "place_in",
            arm,
            object=actor,
            container=container,
            target_pose=target_pose,
        )

    def retreat(self, arm, distance=0.08):
        return self._ok("retreat", arm, distance=distance)

    def move_home(self, arm):
        return self._ok("move_home", arm)

    def pick_dual(self, left_actor, right_actor):
        return self._ok(
            "pick_dual",
            "dual",
            left_object=left_actor,
            right_object=right_actor,
        )

    def lift_dual(self, distance=0.1):
        return self._ok("lift_dual", "dual", distance=distance)

    def place_in_dual(
        self,
        left_actor,
        right_actor,
        container,
    ):
        return self._ok(
            "place_in_dual",
            "dual",
            left_object=left_actor,
            right_object=right_actor,
            container=container,
        )

    def retreat_dual(self, distance=0.08):
        return self._ok("retreat_dual", "dual", distance=distance)

    def move_home_dual(self):
        return self._ok("move_home_dual", "dual")


def run_executor_self_test() -> None:
    plan = load_and_validate_plan(AGENT_DIR / "sample_plan.json")
    fake_skills = _FakeSkills()
    executor = PlanExecutor(
        scene=None,
        skills=fake_skills,
        objects={
            "part_A_1": "fake_part_1",
            "part_A_2": "fake_part_2",
            "part_A_3": "fake_part_3",
            "part_B_1": "fake_part_b1",
            "part_B_2": "fake_part_b2",
            "box": "fake_box",
        },
    )
    report = executor.execute(plan)
    expected = [
        "pick_dual", "lift_dual", "place_in_dual", "retreat_dual",
        "pick", "lift", "place_in", "retreat",
    ]
    if not report.success or fake_skills.calls != expected:
        raise AssertionError(f"执行顺序不正确：{fake_skills.calls}")

    null_arm_plan = AgentPlan.from_dict(
        {
            "understood_goal": "验证模型输出的null机械臂参数",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick",
                    "arguments": {
                        "object": "part_A_1",
                        "arm": None,
                    },
                },
                {
                    "skill": "lift",
                    "arguments": {"arm": None, "distance": 0.1},
                },
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "part_A_1",
                        "container": "box",
                        "arm": None,
                    },
                },
                {
                    "skill": "retreat",
                    "arguments": {"arm": None, "distance": 0.08},
                },
            ],
            "final_response": "任务完成，还需要什么？",
        }
    )
    AgentPlanner().validate(null_arm_plan)
    null_arm_skills = _FakeSkills()
    null_arm_executor = PlanExecutor(
        scene=None,
        skills=null_arm_skills,
        objects=executor.objects,
    )
    null_arm_report = null_arm_executor.execute(null_arm_plan)
    if (
        not null_arm_report.success
        or null_arm_skills.calls
        != ["pick", "lift", "place_in", "retreat"]
    ):
        raise AssertionError("arm:null没有正确继承pick选择的机械臂")

    class _NamedObject:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    class _ReferenceScene:
        pass

    reference_scene = _ReferenceScene()
    reference_scene.agent_skill_parameter_registry = {
        "obs1:part_A_1:grasp": {
            "kind": "grasp",
            "object": "part_A_1",
            "arm": "left",
            "pre_grasp_distance": 0.09,
        },
        "obs1:part_A_1:lift": {
            "kind": "lift",
            "object": "part_A_1",
            "arm": "left",
            "distance": 0.10,
        },
        "obs1:part_A_1:drop": {
            "kind": "drop",
            "object": "part_A_1",
            "arm": "left",
            "container": "box",
            "target_pose": [0.0, -0.14, 0.78, 1.0, 0.0, 0.0, 0.0],
        },
        "obs1:part_A_1:retreat": {
            "kind": "retreat",
            "object": "part_A_1",
            "arm": "left",
            "distance": 0.08,
        },
    }
    reference_plan = AgentPlan.from_dict(
        {
            "understood_goal": "使用本轮视觉参数搬运一个零件",
            "needs_clarification": False,
            "clarification_question": None,
            "steps": [
                {
                    "skill": "pick",
                    "arguments": {
                        "object": "part_A_1",
                        "arm": "left",
                        "grasp_ref": "obs1:part_A_1:grasp",
                    },
                },
                {
                    "skill": "lift",
                    "arguments": {
                        "arm": "left",
                        "distance_ref": "obs1:part_A_1:lift",
                    },
                },
                {
                    "skill": "place_in",
                    "arguments": {
                        "object": "part_A_1",
                        "container": "box",
                        "arm": "left",
                        "drop_ref": "obs1:part_A_1:drop",
                    },
                },
                {
                    "skill": "retreat",
                    "arguments": {
                        "arm": "left",
                        "distance_ref": "obs1:part_A_1:retreat",
                    },
                },
            ],
            "final_response": "任务完成，还需要什么？",
        }
    )
    AgentPlanner().validate(reference_plan)
    reference_skills = _FakeSkills()
    reference_executor = PlanExecutor(
        scene=reference_scene,
        skills=reference_skills,
        objects={
            "part_A_1": _NamedObject("part_A_1"),
            "box": _NamedObject("box"),
        },
    )
    reference_report = reference_executor.execute(reference_plan)
    if (
        not reference_report.success
        or reference_skills.calls
        != ["pick", "lift", "place_in", "retreat"]
    ):
        raise AssertionError("合法视觉参数引用没有被正确解析和执行")
    print("JSON执行器离线自检通过：对象映射、变量解析和技能调用顺序均正常。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", help="让模型根据这条文字指令生成技能计划")
    parser.add_argument("--planner-self-test", action="store_true")
    parser.add_argument("--executor-self-test", action="store_true")
    parser.add_argument("--validate-plan", metavar="JSON_FILE")
    options = parser.parse_args()

    if options.planner_self_test:
        run_self_test()
        return
    if options.executor_self_test:
        run_executor_self_test()
        return
    if options.validate_plan:
        try:
            plan = load_and_validate_plan(options.validate_plan)
        except (RuntimeError, ValueError) as exc:
            parser.exit(2, f"JSON计划校验失败：{exc}\n")
        print(
            f"JSON计划校验通过：共 {len(plan.steps)} 个技能步骤，"
            "可以交给仿真执行器。"
        )
        return
    if not options.plan:
        parser.error(
            "请提供 --plan、--planner-self-test、--executor-self-test "
            "或 --validate-plan"
        )

    try:
        plan = AgentPlanner().plan(options.plan)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"Agent规划失败：{exc}\n")
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
