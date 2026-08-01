"""Select the visible NVIDIA GPU with the most currently free memory."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    free_memory_mb: int
    utilization_percent: int
    name: str


def query_gpus() -> list[GpuInfo]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("找不到 nvidia-smi，无法自动选择 NVIDIA GPU") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"nvidia-smi 查询失败：{detail}") from exc

    gpus: list[GpuInfo] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) < 4:
            continue
        try:
            index = int(row[0].strip())
            free_memory_mb = int(row[1].strip())
        except ValueError:
            continue
        try:
            utilization_percent = int(row[2].strip())
        except ValueError:
            utilization_percent = 100
        gpus.append(
            GpuInfo(
                index=index,
                free_memory_mb=free_memory_mb,
                utilization_percent=utilization_percent,
                name=",".join(row[3:]).strip(),
            )
        )

    if not gpus:
        raise RuntimeError("nvidia-smi 没有返回可用 GPU")
    return gpus


def select_gpu() -> GpuInfo:
    # Free memory is the main constraint for CuRobo. Utilization breaks ties.
    selected = max(
        query_gpus(),
        key=lambda gpu: (gpu.free_memory_mb, -gpu.utilization_percent),
    )
    minimum_mb = int(os.getenv("ROBOTWIN_MIN_FREE_GPU_MB", "8000"))
    if selected.free_memory_mb < minimum_mb:
        raise RuntimeError(
            f"空闲显存最多的 GPU {selected.index} 也只有 "
            f"{selected.free_memory_mb} MiB，低于安全阈值 {minimum_mb} MiB。"
            "请等待 GPU 释放，或确认风险后设置 "
            "ROBOTWIN_MIN_FREE_GPU_MB=0。"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine",
        action="store_true",
        help="输出供 run.sh 读取的管道分隔字段",
    )
    options = parser.parse_args()

    try:
        gpu = select_gpu()
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"GPU自动选择失败：{exc}\n")

    if options.machine:
        print(
            f"{gpu.index}|{gpu.free_memory_mb}|"
            f"{gpu.utilization_percent}|{gpu.name}"
        )
    else:
        print(
            f"将自动选择物理 GPU {gpu.index}：{gpu.name}，"
            f"空闲显存 {gpu.free_memory_mb} MiB，"
            f"利用率 {gpu.utilization_percent}%"
        )


if __name__ == "__main__":
    main()
