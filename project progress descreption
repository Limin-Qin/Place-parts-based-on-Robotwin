# 零件与空盒子场景

这是一个独立的 RoboTwin 2.0 场景示例，不修改或注册任何官方 benchmark。

场景包含：

- `aloha-agilex` 轮式底座外观的双臂机器人；
- RoboTwin 默认桌子与背景墙；
- 3 个完全相同的官方 `004_fluted-block/base0` 方孔带槽机械块，作为左、中、
  右三个零件 A；
- 2 个官方 `055_small-speaker/base1` 方形模块，作为随机摆放在左右安全区域的
  两个零件 B；
- 1 个截图中的浅蓝色官方 `062_plasticbox/base3`，作为可容纳三个
  零件的容器。

三个机械零件在本示例运行时统一缩放为官方模型的 0.80 倍，视觉网格、
碰撞网格和抓取标注使用相同尺度；官方资产文件没有被修改。

放置动作接口参考官方容器类 benchmark。为避免三个机械块在盒内叠放，该
模型仅在本示例运行时将两个水平方向放大为 1.55 倍，高度不变；官方资产
文件本身没有被修改。技能会根据盒内最新占用状态动态选择空闲放置区域，
不为某个零件预先绑定最终位置。
机械块使用官方资产的抓取标注。

## 查看场景

在 RoboTwin 仓库根目录执行：

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh
```

等待 SAPIEN Viewer 窗口出现。可以用鼠标调整观察视角，关闭窗口即可退出。

如果当前机器没有桌面显示环境，可先进行无窗口验证：

```bash
bash examples/my_parts_box_scene/run.sh --check
```

无窗口验证只确认场景和物体成功加载，不会显示画面。

服务器没有图形桌面时，可用机器人头部相机离屏渲染一张桌面布局图片：

```bash
bash examples/my_parts_box_scene/run.sh --snapshot
```

图片会保存为：

```text
examples/my_parts_box_scene/scene_preview.png
```

用 VS Code Remote 打开该文件，或用 `scp` 下载到本地即可查看。

## YOLO-World 单帧视觉测试

该测试只创建当前场景，从机器人配置自带的原生 `head_camera` 采集一张 RGB
图像并交给 YOLO-World。本示例将该相机切换为 RoboTwin 已有的
`Large_D435` 配置（640×480、视场角和安装位姿不变）。它不会读取物体真实
位姿、深度图或仿真器分割标签，也不会调用机械臂技能。

在已安装 `ultralytics` 的 RoboTwin 环境中运行：

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh --vision-test
```

首次使用默认权重 `yolov8s-worldv2.pt` 时，Ultralytics 可能需要联网下载权重。
检测结果保存在：

```text
examples/my_parts_box_scene/vision_results/head_camera_rgb.png
examples/my_parts_box_scene/vision_results/head_camera_yolo_world.png
examples/my_parts_box_scene/vision_results/head_camera_yolo_world.json
```

默认视觉提示词及其场景语义映射为：

```text
gray hollow block         -> part_A
small speaker             -> part_B
blue plastic storage bin  -> box
```

JSON 中包含每个检测结果的提示词、场景类别、置信度、`bbox_xyxy` 和
`bbox_xywh` 像素坐标，并对场景预期的 3 个 A、2 个 B 和 1 个盒子进行数量
初检。数量初检只用于帮助查看零样本检测效果，不是机器人任务成功判定。

可以在不修改代码的情况下调整权重、阈值和推理尺寸：

```bash
bash examples/my_parts_box_scene/run.sh --vision-test \
  --vision-model yolov8s-worldv2.pt \
  --vision-conf 0.10 \
  --vision-imgsz 640
```

`run.sh` 会照常自动选择 GPU。手动指定时，`--vision-device 0` 表示
`CUDA_VISIBLE_DEVICES` 中的第一张可见 GPU。

## 自动生成YOLO训练数据

当前三个零件A统一使用相同的 `004_fluted-block/base0` 方孔模型。数据生成器
直接复用当前场景创建逻辑，所以训练图片中的A同样保持0.80倍运行时缩放，
盒子也保持1.55倍水平缩放，不会使用官方资产的原始尺寸。

生成器只创建一次RoboTwin场景，然后重复随机化A、B的位置和完整
`-180°～180°` 水平朝向，并为每个零件独立采样约60%直立、35%侧躺
（绕水平轴±90°）、5%翻面（180°）的稳定姿态。程序读取缩放后资产的实际
网格边界，根据旋转后最低点自动计算桌面或盒底支撑高度，避免悬空和穿透。
此外还随机化盒子的小范围位置和朝向、head_camera的小幅位姿及场景光照。
每张图片按均衡计划包含0、1、2或3个盒内零件；其余零件随机摆在盒子外的
安全桌面区域。每帧同时获取：

```text
head_camera RGB             -> 保存为训练图片
SAPIEN actor segmentation   -> 自动计算A、B、box的像素检测框
```

actor segmentation属于离线制作标签使用的特权信息，不会保存为模型输入，也
不会用于训练后运行时推理。默认生成500张训练图片和100张验证图片：

零件不会直接固定在随机位姿。生成器先把它们放在桌面或盒子上方，再由
SAPIEN/PhysX的重力、碰撞和摩擦自然落稳；未落入预期区域、仍在运动或存在
明显物理穿透的帧会被丢弃并重新采样。

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh --generate-vision-dataset
```

为了先快速检查标注框是否正确，可以生成一个小数据集：

```bash
bash examples/my_parts_box_scene/run.sh --generate-vision-dataset \
  --dataset-train-count 10 \
  --dataset-val-count 5 \
  --dataset-seed 2026
```

默认输出结构：

```text
examples/my_parts_box_scene/vision_dataset/
├── images/train/             # 640x480 head_camera RGB
├── images/val/
├── labels/train/             # YOLO归一化xywh标签
├── labels/val/
├── previews/                 # 人工检查用画框图，不作为训练输入
├── train.txt
├── val.txt
├── data.yaml
├── sample_annotations.jsonl # 逐图记录哪些零件在盒内及像素框
└── generation_metadata.json
```

类别固定为：

```text
0: part_A
1: part_B
2: box
```

如果某个零件可见像素过少或接触图像边缘，该随机样本会被丢弃并重新生成。
完整生成前应先查看 `previews/` 中的小规模结果，确认分割ID与检测框正确。
训练图片本身不画框；YOLO从 `images/` 读取原图，并从 `labels/` 中读取同名
`.txt` 文件里的检测框。把框直接画进训练图片反而会污染训练数据。

## 训练YOLO-World

训练前只检查500张训练图、100张验证图、全部同名标签和原始权重：

```bash
bash examples/my_parts_box_scene/run.sh --train-yolo --check-only
```

开始训练时，`run.sh` 会自动选择GPU。默认使用本示例已有的
`yolov8s-worldv2.pt`，训练100个epoch、输入尺寸640，并自动选择batch size：

```bash
bash examples/my_parts_box_scene/run.sh --train-yolo
```

也可以给本轮训练命名或调整参数：

```bash
bash examples/my_parts_box_scene/run.sh --train-yolo \
  --run-name parts_ab_box_v1 \
  --epochs 100 \
  --imgsz 640 \
  --batch -1
```

每次训练都创建独立目录，不允许覆盖同名结果：

```text
examples/my_parts_box_scene/yolo_training_runs/<本轮名称>/
examples/my_parts_box_scene/trained_weights/<本轮名称>_best.pt
```

原始的 `examples/my_parts_box_scene/yolov8s-worldv2.pt` 只会被加载，不会被
保存或覆盖；程序在训练前后还会比较它的SHA256。训练后的最佳权重会保留在
本轮 `weights/best.pt`，并额外复制到 `trained_weights/` 归档。

用训练后的权重测试当前场景：

```bash
bash examples/my_parts_box_scene/run.sh --vision-test \
  --vision-model examples/my_parts_box_scene/trained_weights/<本轮名称>_best.pt \
  --vision-conf 0.25
```

视觉测试会自动识别该权重已经包含 `part_A`、`part_B`、`box` 三个固定类别，
不会再覆盖成零样本文本提示词；继续使用原始权重时仍保持原来的零样本模式。

## RGB-D位置估计与特权信息比较

只创建场景、采集同一帧head_camera RGB与深度、估计桌面零件位置并与仿真器
Actor真实位姿比较，不加载运动规划器，也不驱动机械臂：

```bash
bash examples/my_parts_box_scene/run.sh --vision-position-test \
  --vision-conf 0.25
```

默认自动使用 `trained_weights/` 中最新的 `*_best.pt`，也可以显式指定：

```bash
bash examples/my_parts_box_scene/run.sh --vision-position-test \
  --position-model examples/my_parts_box_scene/trained_weights/<名称>_best.pt \
  --vision-conf 0.25
```

估计过程仅使用RGB、YOLO检测框、对齐的Position/深度图以及相机外参。程序先
从深度中估计桌面平面，再在每个检测框内去掉桌面，选择最大的抬高深度连通
区域，再用最高表面轮廓的稳健边界中心和桌面支撑高度构造零件位置，避免
朝向相机的竖直表面造成中心偏移。仿真器Actor位姿只在估计完成后用于离线
误差评估，不会进入位置估计算法。

结果保存在：

```text
examples/my_parts_box_scene/rgbd_position_results/
├── head_camera_rgb.png
├── head_camera_yolo_world.png
├── head_camera_depth.png
├── head_camera_depth_m.npy
├── head_camera_rgbd_positions.png
└── position_comparison.json
```

## RGB-D鲁棒性测试

在同一场景进程中依次测试随机三维旋转、物体间遮挡、相机扰动、深度噪声与
缺失、盒内零件、真实机械臂前景遮挡，以及全部条件组合。该测试仍然不会加载
运动规划器或执行机械臂动作：

```bash
bash examples/my_parts_box_scene/run.sh --vision-robustness-test \
  --vision-conf 0.25
```

每种条件的RGB、深度图、检测图、位置比较图和JSON分别保存在：

```text
examples/my_parts_box_scene/rgbd_robustness_results/<场景名称>/
```

总结果保存在：

```text
examples/my_parts_box_scene/rgbd_robustness_results/robustness_summary.json
```

## 基础技能

当前提供以下可复用、带参数的技能：

```text
pick(object, arm=None)
lift(arm, distance=0.1)
place_in(object, container, arm)
retreat(arm, distance=0.08)
move_home(arm)
pick_dual(left_object, right_object)
lift_dual(distance=0.1)
place_in_dual(left_object, right_object, container)
retreat_dual(distance=0.08)
move_home_dual()
```

快速查看技能列表（不会启动仿真器）：

```bash
bash examples/my_parts_box_scene/run.sh --list-skills
```

顺序验证基础技能：

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh --test-skills
```

该命令使用单臂技能依次处理左、中、右三个零件。测试视频保存为：

```text
examples/my_parts_box_scene/skills_test.mp4
```

终端中每个技能会分别显示 `[成功]` 或 `[失败]`，最后还会输出任务成功判定。

## Agent 文字规划器

规划器会读取：

- 用户文字；
- `agent/scene_catalog.py` 中的场景物体；
- `RobotSkills.schemas()` 暴露的技能契约。

语言模型负责自行选择技能、参数和顺序，程序不会为 `put_in` 使用固定任务模板。
模型只能输出注册过的技能 JSON；未知技能、未知物体或非法参数会被本地校验器拒绝。

先运行不需要模型、仿真器或 GPU 的离线自检：

```bash
bash examples/my_parts_box_scene/run.sh --planner-self-test
```

连接任意 OpenAI-compatible 模型服务：

```bash
export AGENT_API_BASE="http://127.0.0.1:8000/v1"
export AGENT_MODEL="你的模型名称"
# 服务需要鉴权时再设置：
export AGENT_API_KEY="你的API密钥"
# 可选：模型请求单次超时秒数和最大尝试次数（以下为默认值）
export AGENT_API_TIMEOUT="60"
export AGENT_API_RETRIES="3"
```

让 Agent 根据文字生成计划：

```bash
bash examples/my_parts_box_scene/run.sh \
  --plan "请帮我拿零件A"
```

这个命令只生成并校验 JSON 计划，不启动 RoboTwin，也不执行机械臂。检查输出：

1. 只能出现技能列表中注册的单臂或双臂技能；
2. 物体只能是 `part_A_1`、`part_A_2`、`part_A_3`、`part_B_1`、
   `part_B_2`，容器只能是 `box`；
3. “拿零件A”应覆盖三个A，“拿零件B”应覆盖两个B；指定类别和位置时只
   操作对应零件；
4. 单臂 `pick` 后的机械臂参数应引用 `$last.arm`；
5. `final_response` 应以“还需要什么？”结尾。

Agent可以自行选择单臂或双臂技能以及执行顺序。一次搬运在放置并撤离后，
机械臂即可直接继续下一个任务；`move_home`/`move_home_dual` 是可选动作，
不再要求每次放置后都回初始姿态。例如可以：

- 三个零件分别执行单臂流程；
- 对左右零件执行双臂流程，对中间零件执行单臂流程，并自主决定两段流程的先后。

这两种计划不是程序按用户指令套用固定模板，而是模型根据场景物体属性、
技能契约和用户语言选择。`sample_plan.json` 只是一份用于执行器自检的示例。

用户可以分别指定零件类别：

```bash
bash examples/my_parts_box_scene/run.sh --agent-run "请帮我拿零件A"
bash examples/my_parts_box_scene/run.sh --agent-run "请帮我拿零件B"
bash examples/my_parts_box_scene/run.sh --agent-run "请把左侧零件B放进盒子"
```

零件B默认在每次创建场景时重新随机采样位置。如需复现实验，可固定本示例
专用随机种子：

```bash
MY_PARTS_BOX_SEED=7 bash examples/my_parts_box_scene/run.sh \
  --agent-run "请帮我拿零件B"
```

## JSON 计划执行器

Agent 与机器人之间使用 JSON 计划连接：

```text
用户文字 → 大模型 Agent 生成 JSON → 本地校验 → PlanExecutor
        → RobotSkills → RoboTwin/CuRobo → 机器人动作
```

先做不需要模型、仿真器或 GPU 的执行器自检：

```bash
bash examples/my_parts_box_scene/run.sh --executor-self-test
```

校验示例计划：

```bash
bash examples/my_parts_box_scene/run.sh \
  --validate-plan examples/my_parts_box_scene/agent/sample_plan.json
```

使用示例 JSON 实际驱动机器人并保存视频。启动脚本会自动选择空闲显存最多
且达到安全显存阈值的 GPU：

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh \
  --execute-plan examples/my_parts_box_scene/agent/sample_plan.json
```

视频保存为：

```text
examples/my_parts_box_scene/agent_execution.mp4
```

`sample_plan.json` 是用于隔离验证“JSON → 机器人”的手写测试计划，不具备
文字理解能力。配置好 OpenAI-compatible 大模型后，可用一个命令启动闭环
Agent：

```bash
bash examples/my_parts_box_scene/run.sh \
  --agent-run "请帮我拿零件A"
```

`--agent-run` 不再在仿真启动前一次性生成全部动作。它先理解用户目标，然后
在同一个仿真器进程中循环执行：

```text
head_camera采集RGB与对齐深度
→ YOLO检测A、B和box并计算世界坐标XYZ
→ 读取机器人本体状态
→ Agent规划下一安全阶段
→ 执行单臂/双臂技能
→ 再次进行视觉观测并重新规划
```

每一阶段可以整理机械臂、单臂搬运一个零件，或双臂同时搬运两个零件；
具体对象、机械臂和先后顺序由 Agent 根据最新状态决定。`--agent-run`
提供给Agent的物体位置、容器位置和盒内判定来自head_camera RGB-D；自动选臂
和抓取目标坐标也使用同一视觉XYZ。抓取后放置所需的物体运动状态由视觉抓取
时建立的“物体—末端”相对变换和机器人本体状态传播，不会回退到Actor位姿。

默认自动使用 `trained_weights` 中最新的 `*_best.pt`。需要覆盖时可设置：

```bash
export AGENT_VISION_MODEL="examples/my_parts_box_scene/trained_weights/xxx_best.pt"
export AGENT_VISION_CONF="0.60"
export AGENT_VISION_IMGSZ="640"
export AGENT_VISION_DEVICE="0"
```

每次闭环观测的原图、检测图、RGB-D位置图和JSON保存在
`agent_vision_observations/observation_XXX/`。若目标或box没有可靠视觉位置，
程序会停止本轮抓取并报错，不会使用仿真器特权坐标补全。

动作后的第一次视觉确认如果因为机械臂停在盒子上方而看不到box或待确认目标，
程序会根据机器人本体状态，仅将尚未归位的机械臂依次移出头部相机视野，然后
重新采集一次RGB-D。该动作只在实际发生遮挡时触发，不会强制每个阶段都归位；
重试后仍无法检测时才安全停止，并且仍不会使用仿真器物体位姿。

Agent执行默认检测阈值为0.90。双臂抓取前还会用本轮视觉XYZ确认两个目标分别
位于左右安全工作区且间距足够；不满足时会让大模型重新规划单臂方案。双臂技能
在安全检查通过后同步规划和执行左右臂抓取。中央工作区仍按末端到目标的实时
距离选择机械臂，因此没有固定禁用某一只手臂。

抓取定位采用由粗到细的闭环方式：head_camera先给出桌面目标的世界坐标，机械臂
同步或单独移动到预抓取区域；随后使用对应的left_camera/right_camera RGB-D再次
检测目标。每轮都用腕部测量坐标与上一目标坐标的实际XYZ误差，重新计算绝对
预抓取姿态，最多进行三轮视觉细化；这里的机械臂移动量来自测量误差，不是固定
的1厘米或2厘米。误差收敛或完成最后一次修正后，才执行夹爪接近和闭合。
腕部检测图片和坐标记录保存在`agent_wrist_observations/`。

进入腕部观察阶段时，控制器会用腕部相机外参和头部视觉给出的目标表面坐标，
计算目标相对光轴的横向误差并主动调整观察姿态，使目标尽量位于腕部画面中央。
由于近距离目标可能被裁切并被分类器错标，腕部阶段的目标关联以head_camera已经
确定的身份和三维空间锚点为准：在腕部检测框中选择世界坐标最接近锚点的实例，
而不是再次完全依赖腕部类别名称。语义类别仍由head_camera负责确定。

如果腕部检测、运动规划或最终抓取仍失败，闭环执行器会张开夹爪、依次将双臂
恢复到安全姿态，再使用head_camera重新观察并请求Agent重新规划。连续三次失败
才会安全停止，避免无限循环；恢复与重试同样不读取仿真器物体位姿。

所有目标第一次被视觉确认在盒内后，程序还会推进约1秒仿真时间，再次进行
RGB-D观测。只有目标仍在盒内、夹爪已经松开，并且这一秒内各目标的视觉表面
位置变化不超过15 mm，最终任务才判定成功。结果写入
`agent_execution_trace.json` 的 `stability_check`。

当前视觉模块输出的是XYZ位置，抓取姿态仍使用本场景中直立零件的已知标准
抓取方向。因此当前直立摆放场景可以使用；如果后续允许零件任意侧躺，还需要
增加姿态/6D Pose估计，再替换标准抓取方向。

终端不再打印每轮完整环境状态、计划 JSON 或逐条技能，只用一行概括下一
阶段任务，并在该阶段结束后明确显示闭环执行成功或失败。
执行器也不再逐条打印技能参数和成功日志；如果某一步失败，异常信息仍会指出
失败步骤和原因。
每轮最新完整计划仍保存为 `generated_plan.json`，完整观察、计划和执行历史
仍保存为 `agent_execution_trace.json`，动作视频保存为
`agent_execution.mp4`。
如果 Agent 判断信息不足，它会提出澄清问题，不执行机器人动作。

只检查当前会自动选择哪个 GPU，不启动仿真：

```bash
bash examples/my_parts_box_scene/run.sh --select-gpu-only
```

自动选择默认要求至少有 `8000 MiB` 空闲显存。如果已经手动设置
`CUDA_VISIBLE_DEVICES`，启动脚本会尊重手动设置，不再自动选择：

```bash
CUDA_VISIBLE_DEVICES=6 bash examples/my_parts_box_scene/run.sh --agent-run \
  "请把左侧零件A放进盒子"
```

## 执行完整抓取任务并保存视频

任务直接使用仿真中的物体真实位姿（特权信息），并复用 RoboTwin 的
`grasp_actor`、`place_actor`、`move_by_displacement` 和 CuRobo 运动规划器。
当前完整任务同样通过上述基础技能顺序编排：

```bash
conda activate RoboTwin
bash examples/my_parts_box_scene/run.sh --run-task
```

完成后视频保存在：

```text
examples/my_parts_box_scene/parts_into_box.mp4
```

该模式无需服务器桌面。正式 `head_camera` 使用机器人原有安装位姿和
62° 垂直视场角，并由同一相机同步产生 RGB、Position/depth 和标定矩阵；
没有额外的全景或辅助 wide 相机。首次初始化 CuRobo 可能需要一两分钟，
并且必须有可用的 NVIDIA GPU。

## 修改摆放

只需编辑 `parts_box_scene.py` 中 `load_actors()` 里的 `sapien.Pose`：

```python
sapien.Pose([x, y, z], [qw, qx, qy, qz])
```

前三个数是位置（米），后四个数是四元数姿态。修改后重新运行 `run.sh`。

## 生成多相机补充训练检查集

下面的命令生成 5 个经过物理落稳检查的随机场景。每个场景分别保存
62° 正式头部相机、左腕相机和右腕相机，共 15 张图片；内容覆盖桌面边缘、
物体部分遮挡、机械臂遮挡和盒内零件：

```bash
bash examples/my_parts_box_scene/run.sh \
  --generate-supplemental-vision-dataset \
  --supplemental-scene-count 5
```

原始训练图、YOLO 标签和带框检查图分别保存在
`vision_dataset_multicamera_smoke/images`、`labels` 和 `previews`。该命令
不会覆盖现有 `vision_dataset`。

正式补充集使用同一个 RoboTwin 场景实例，通过重排零件、盒子和机械臂
生成 100 个物理布局；每个布局保存 1 张头部图和左右腕部各 1 张，共
300 张。左右腕部在同一布局中观察不同零件：

```bash
bash examples/my_parts_box_scene/run.sh \
  --generate-supplemental-vision-dataset \
  --supplemental-scene-count 100 \
  --supplemental-seed 2040 \
  --supplemental-output \
  examples/my_parts_box_scene/vision_dataset_multicamera_supplement
```

当前正式补充集已经与原始数据通过图片清单组成
`vision_dataset_combined_multicamera`，原图、标签和旧权重均未复制或
覆盖。继续微调已有最佳权重：

```bash
bash examples/my_parts_box_scene/run.sh --train-yolo \
  --data examples/my_parts_box_scene/vision_dataset_combined_multicamera/data.yaml \
  --weights examples/my_parts_box_scene/trained_weights/parts_ab_box_20260728_023941_best.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch -1 \
  --patience 15 \
  --run-name parts_ab_box_multicamera_finetune
```
