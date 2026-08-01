#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${example_dir}/../.." && pwd)"

if [[ "${1:-}" == "--list-skills" ]]; then
  python "${example_dir}/agent/list_skills.py"
  exit 0
fi

if [[ "${1:-}" == "--plan" ||
      "${1:-}" == "--planner-self-test" ||
      "${1:-}" == "--executor-self-test" ||
      "${1:-}" == "--validate-plan" ]]; then
  python "${example_dir}/agent/plan_command.py" "$@"
  exit 0
fi

if [[ "${1:-}" == "--select-gpu-only" ]]; then
  python "${example_dir}/agent/select_gpu.py"
  exit 0
fi

if [[ "${1:-}" == "--web" ]]; then
  shift
  python -u "${example_dir}/web/server.py" "$@"
  exit 0
fi

cd "${repo_root}"
if ! python -c "import sapien" >/dev/null 2>&1; then
  echo "当前 Python 环境缺少 RoboTwin 依赖。请先执行: conda activate RoboTwin" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_record="$(python "${example_dir}/agent/select_gpu.py" --machine)"
  IFS='|' read -r selected_gpu free_memory_mb gpu_utilization gpu_name <<<"${gpu_record}"
  export CUDA_VISIBLE_DEVICES="${selected_gpu}"
  echo "自动选择物理 GPU ${selected_gpu}：${gpu_name}，空闲显存 ${free_memory_mb} MiB，利用率 ${gpu_utilization}%"
else
  echo "使用手动指定的 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

if [[ "${1:-}" == "--agent-run" ]]; then
  python -u "${example_dir}/agent/run_agent.py" "$@"
  exit 0
fi

if [[ "${1:-}" == "--train-yolo" ]]; then
  shift
  python -u "${example_dir}/perception/train_yolo_world.py" "$@"
  exit 0
fi

python -u "${example_dir}/parts_box_scene.py" "$@"
