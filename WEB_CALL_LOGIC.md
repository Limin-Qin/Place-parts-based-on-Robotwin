# `run.sh --web` 当前调用逻辑

本文记录以下命令在当前项目中的真实调用链：

```bash
bash examples/my_parts_box_scene/run.sh --web
```

## 1. 启动 Web 服务

入口是 `examples/my_parts_box_scene/run.sh`。检测到第一个参数为 `--web` 后，
脚本移除该参数并启动：

```bash
python -u examples/my_parts_box_scene/web/server.py
```

`web/server.py` 创建本地 Web 服务和一个持久化的 RoboTwin 工作进程。浏览器
刷新不会重新创建仿真场景；只有重新加载场景或重启服务时才会创建新场景。

## 2. 创建持久化场景工作进程

Web 服务的 `AgentProcessManager._begin_worker_initialization()` 启动后台线程，
随后 `_run_persistent_worker()` 执行：

```bash
bash examples/my_parts_box_scene/run.sh --agent-worker
```

`run.sh` 的普通参数分支最终运行：

```bash
python -u examples/my_parts_box_scene/parts_box_scene.py --agent-worker
```

`parts_box_scene.py:main()` 将 `--agent-worker` 分派给：

```python
run_persistent_agent_worker()
```

该函数只创建一次 `PartsBoxScene`，然后调用
`prepare_closed_loop_agent_scene(scene)`：

1. 加载机器人、零件 A、零件 B、盒子和相机；
2. 加载最新的训练视觉权重，或读取 `AGENT_VISION_MODEL`；
3. 启用 head camera RGB-D 视觉定位；
4. 创建 `AgentPlanner`；
5. 通过标准输入持续等待 Web 任务。

Web 服务和工作进程之间使用逐行 JSON 通信：

```json
{"task_id": "任务ID", "command": "请帮我拿零件A"}
```

## 3. 接收并执行用户任务

浏览器提交任务后，`web/server.py:AgentProcessManager.start()` 将 JSON 写入工作
进程的标准输入。`run_persistent_agent_worker()` 读取任务后调用：

```python
run_closed_loop_agent(
    command,
    task_video_path,
    scene=scene,
    planner=planner,
    prepared_perception=perception,
)
```

闭环 Agent 的主要过程为：

```text
用户文本
  -> AgentPlanner 理解目标
  -> head_camera 采集 RGB-D
  -> 视觉检测并生成 part_A_N / part_B_N 实例
  -> AgentPlanner 生成本轮技能计划
  -> PlanExecutor 校验并分派技能
  -> 执行抓取、抬升、放入盒子和撤离
  -> 再次观察并判断是否还有目标
```

对于当前 Web 闭环，规划器生成的抓取技能是：

```json
{
  "skill": "pick_visual_asset",
  "arguments": {
    "object": "part_A_1",
    "arm": "left"
  }
}
```

`PlanExecutor._call_skill()` 将它分派给：

```python
RobotSkills.pick_visual_asset(...)
```

## 4. Web 抓取接触位姿

`pick_visual_asset()` 使用视觉检测得到的物体位置，不读取仿真器 Actor 的真实
物体位姿。它使用固定物体四元数：

```python
[0.5, 0.5, 0.5, 0.5]
```

零件 A 和零件 B 的 Web 专用接触矩阵来自：

```text
examples/my_parts_box_scene/agent/web_pick_contact_poses.json
```

两类零件当前都只有一个接触位姿：

```json
[
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
]
```

新配置不包含 `scale`，因此 `_VisionGraspActor.get_contact_point()` 会原样使用
矩阵，不对局部平移进行缩放。单位矩阵表示：

```text
接触点局部位置 = [0, 0, 0]
接触坐标系旋转 = 单位旋转
```

随后 `pick_visual_asset()` 调用 RoboTwin 示例已有的抓取逻辑：

```python
self.scene.grasp_actor(
    visual_actor,
    arm_tag=arm,
    pre_grasp_dis=pre_grasp_distance,
)
```

`Base_Task.grasp_actor()` 遍历接触位姿，通过 `get_grasp_pose()` 计算预抓取姿态
和接触姿态，再调用运动规划器执行：

```text
打开夹爪 -> 移动到预抓取位姿 -> 移动到接触位姿 -> 闭合夹爪
```

因此，新文件只替换 Web `pick_visual_asset` 使用的接触位姿；零件资产自身的
视觉网格缩放、碰撞网格缩放、质量和原始资产 JSON 均保持不变。

## 5. 抓取后的调用

抓取成功后，闭环计划继续调用示例中的基础技能：

```text
lift       -> 抬升零件
place_in   -> 将零件放入盒子中的动态空闲位置
retreat    -> 机械臂撤离
move_home  -> 必要时回到初始位置
```

每轮结束后重新采集视觉状态。如果目标类别仍有未处理实例，Agent 会规划下一
轮；全部完成后，工作进程向 Web 服务返回 `task_result` 事件。

## 6. 关键文件

| 作用 | 文件 |
|---|---|
| Shell 入口 | `examples/my_parts_box_scene/run.sh` |
| Web 服务及工作进程管理 | `examples/my_parts_box_scene/web/server.py` |
| 场景创建与持久化 Agent 循环 | `examples/my_parts_box_scene/parts_box_scene.py` |
| 闭环规划 | `examples/my_parts_box_scene/agent/planner.py` |
| 技能安全分派 | `examples/my_parts_box_scene/agent/plan_executor.py` |
| `pick_visual_asset` 实现 | `examples/my_parts_box_scene/agent/robot_skills.py` |
| Web 专用 A/B 接触矩阵 | `examples/my_parts_box_scene/agent/web_pick_contact_poses.json` |
| RoboTwin 通用抓取实现 | `envs/_base_task.py` |

