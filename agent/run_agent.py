"""Start RoboTwin's closed-loop text Agent in the same simulator process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-run", required=True, metavar="TEXT")
    options = parser.parse_args()

    command = [
        sys.executable,
        "-u",
        str(EXAMPLE_DIR / "parts_box_scene.py"),
        "--agent-loop",
        options.agent_run,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
