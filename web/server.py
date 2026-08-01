"""Local web dashboard for the RoboTwin parts-box embodied Agent.

Real mode owns one persistent simulator worker. The scene is initialized when
the web service starts, camera frames remain available across browser refreshes,
and later commands are sent to the same worker over a JSON-lines pipe.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


WEB_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = WEB_DIR.parent
RUN_SCRIPT = EXAMPLE_DIR / "run.sh"
RUNTIME_DIR = WEB_DIR / "runtime"
CAMERA_DIR = RUNTIME_DIR / "cameras"
STOP_REQUEST_PATH = RUNTIME_DIR / "stop.request"
SESSION_VIDEO_PATH = EXAMPLE_DIR / "agent_execution.mp4"
WORKER_EVENT_PREFIX = "__ROBOTWIN_WORKER_EVENT__"

CAMERA_NAMES = (
    "head_camera",
    "left_camera",
    "right_camera",
)

INITIALIZATION_MESSAGE = (
    "正在加载 RoboTwin 运动规划器并创建闭环 Agent 场景，请稍候……"
)

COMPLETION_LOG_PREFIXES = (
    "操作视频已保存：",
    "完整观察与规划轨迹已保存：",
    "当前场景的累计操作视频已更新：",
)


@dataclass
class TaskState:
    task_id: str | None = None
    command: str = ""
    status: str = "idle"
    stage: str = "等待指令"
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    final_message: str = ""
    logs: deque[str] = field(
        default_factory=lambda: deque(maxlen=240)
    )

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["logs"] = list(self.logs)
        return data


class AgentProcessManager:
    def __init__(self, *, demo: bool = False):
        self.demo = demo
        self.state = TaskState()
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.shutting_down = False
        self.reload_in_progress = False
        self.suppressing_traceback = False
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        CAMERA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.demo:
            self._begin_worker_initialization()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = self.state.public_dict()
            result["demo_mode"] = self.demo
            result["camera_state"] = self._camera_state()
            result["artifacts"] = {
                "video_ready": (
                    SESSION_VIDEO_PATH
                ).is_file(),
                "trace_ready": (
                    EXAMPLE_DIR / "agent_execution_trace.json"
                ).is_file(),
            }
            return result

    def _camera_state(self) -> dict[str, Any]:
        state_path = CAMERA_DIR / "camera_state.json"
        if not state_path.is_file():
            return {
                "sequence": 0,
                "updated_unix_seconds": None,
                "cameras": [],
            }
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "sequence": 0,
                "updated_unix_seconds": None,
                "cameras": [],
            }

    def start(self, command: str) -> tuple[bool, str]:
        command = command.strip()
        if not command:
            return False, "请输入机器人任务。"
        if len(command) > 500:
            return False, "指令过长，请控制在500个字符以内。"

        with self.lock:
            if self.state.status in {
                "initializing",
                "starting",
                "understanding",
                "planning",
                "executing",
                "verifying",
                "stopping",
            }:
                return False, "当前任务仍在运行，请等待完成或先停止。"

            retained_logs = deque(self.state.logs, maxlen=240)
            self.state = TaskState(
                task_id=uuid.uuid4().hex,
                command=command,
                status="starting",
                stage="正在向已就绪的机器人场景提交任务",
                started_at=time.time(),
                logs=retained_logs,
            )
            self.state.logs.append(f"用户指令：{command}")
            if self.demo:
                worker = threading.Thread(
                    target=self._run_demo_task,
                    args=(command,),
                    daemon=True,
                )
                worker.start()
                return True, "任务已提交。"

            process = self.process
            if (
                process is None
                or process.poll() is not None
                or process.stdin is None
            ):
                self.state.status = "failed"
                self.state.stage = "机器人场景工作进程未运行"
                return False, "机器人场景尚未就绪，请查看服务端日志。"
            request = json.dumps(
                {
                    "task_id": self.state.task_id,
                    "command": command,
                },
                ensure_ascii=False,
            )
            try:
                if STOP_REQUEST_PATH.is_file():
                    STOP_REQUEST_PATH.unlink()
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.state.logs.append(f"发送任务失败：{exc}")
                self.state.status = "failed"
                self.state.stage = "无法向机器人场景发送任务"
                return False, "机器人场景连接已断开。"
            return True, "任务已提交到当前场景。"

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            process = self.process
            if (
                self.demo
                or process is None
                or process.poll() is not None
                or self.state.status not in {
                    "starting",
                    "understanding",
                    "planning",
                    "executing",
                    "verifying",
                }
            ):
                return False, "当前没有正在运行的任务。"
            self.state.status = "stopping"
            self.state.stage = "正在停止任务并让双臂返回home状态"

        try:
            STOP_REQUEST_PATH.write_text(
                f"{time.time()}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            with self.lock:
                self.state.status = "failed"
                self.state.stage = "无法发送停止请求"
                self.state.logs.append(f"写入停止请求失败：{exc}")
            return False, "无法发送停止请求。"
        return True, "正在停止任务，随后双臂将返回home状态。"

    def reload(self) -> tuple[bool, str]:
        with self.lock:
            if self.reload_in_progress or self.state.status in {
                "initializing",
                "starting",
                "understanding",
                "planning",
                "executing",
                "verifying",
                "stopping",
                "reloading",
            }:
                return False, "当前任务或场景操作仍在进行，请稍候。"
            if self.demo:
                self.state = TaskState(
                    status="ready",
                    stage="演示场景已重新加载",
                    finished_at=time.time(),
                )
                return True, "演示场景已重新加载。"
            self.reload_in_progress = True
            self.state = TaskState(
                status="reloading",
                stage="正在关闭当前场景并准备重新加载",
                started_at=time.time(),
            )
        threading.Thread(
            target=self._reload_worker,
            daemon=True,
            name="robotwin-agent-reloader",
        ).start()
        return True, "正在重新加载机器人场景。"

    def _reload_worker(self) -> None:
        with self.lock:
            process = self.process
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with self.lock:
                    if self.process is None:
                        break
                time.sleep(0.05)
            else:
                raise RuntimeError("旧场景工作进程未能完全退出")

            self._begin_worker_initialization()
        except Exception as exc:
            with self.lock:
                self.state.status = "failed"
                self.state.stage = f"重新加载场景失败：{exc}"
                self.state.finished_at = time.time()
        finally:
            with self.lock:
                self.reload_in_progress = False

    def _clear_live_camera_files(self) -> None:
        for camera_name in CAMERA_NAMES:
            path = CAMERA_DIR / f"{camera_name}.jpg"
            if path.is_file():
                path.unlink()
        state_path = CAMERA_DIR / "camera_state.json"
        if state_path.is_file():
            state_path.unlink()

    def _append_log(self, line: str) -> None:
        clean = line.rstrip()
        if not clean:
            return
        with self.lock:
            if clean == "Traceback (most recent call last):":
                self.suppressing_traceback = True
                return
            if self.suppressing_traceback:
                return
        if clean.startswith(COMPLETION_LOG_PREFIXES):
            return
        if clean.startswith("Agent：") and "完成" in clean:
            return
        if (
            "新任务开始前，正在让双臂返回home状态" in clean
            or "双臂已处于home状态，准备理解并观察新任务" in clean
        ):
            with self.lock:
                self._update_stage_from_line(clean)
            return
        with self.lock:
            self.state.logs.append(clean)
            self._update_stage_from_line(clean)

    def _update_stage_from_line(self, line: str) -> None:
        if "正在让双臂返回home状态" in line:
            self.state.status = "starting"
            self.state.stage = "新任务开始前，双臂正在返回home状态"
        elif "正在请求大模型生成结构化目标" in line:
            self.state.status = "understanding"
            self.state.stage = "Agent 正在理解用户指令"
        elif "Agent理解到的任务" in line:
            self.state.status = "planning"
            self.state.stage = line
        elif "正在请求大模型生成闭环下一阶段计划" in line:
            self.state.status = "planning"
            self.state.stage = "Agent 正在规划下一阶段"
        elif "[闭环规划" in line:
            self.state.status = "executing"
            self.state.stage = line
        elif "[闭环执行" in line:
            self.state.stage = line
        elif line.startswith("Agent："):
            self.state.final_message = line.removeprefix("Agent：").strip()

    def _begin_worker_initialization(self) -> None:
        with self.lock:
            if self.shutting_down:
                return
            self.suppressing_traceback = False
            self._clear_live_camera_files()
            # A Web service restart creates a new simulator scene and therefore
            # starts a new cumulative recording session.
            SESSION_VIDEO_PATH.unlink(missing_ok=True)
            if STOP_REQUEST_PATH.is_file():
                STOP_REQUEST_PATH.unlink()
            self.state = TaskState(
                status="initializing",
                stage=INITIALIZATION_MESSAGE,
                started_at=time.time(),
            )
        threading.Thread(
            target=self._run_persistent_worker,
            daemon=True,
            name="robotwin-agent-worker",
        ).start()

    def _handle_worker_event(self, payload_text: str) -> None:
        try:
            event_data = json.loads(payload_text)
        except json.JSONDecodeError:
            self._append_log(f"无法解析工作进程事件：{payload_text}")
            return

        event = event_data.get("event")
        with self.lock:
            if event == "initializing":
                self.state.status = "initializing"
                self.state.stage = str(
                    event_data.get(
                        "message",
                        "正在预加载机器人场景",
                    )
                )
            elif event == "ready":
                if self.state.status == "initializing":
                    self.state.status = "ready"
                    self.state.stage = str(
                        event_data.get(
                            "message",
                            "机器人场景已就绪，可以发送任务",
                        )
                    )
                    self.state.finished_at = time.time()
            elif event == "task_started":
                if event_data.get("task_id") == self.state.task_id:
                    self.state.status = "understanding"
                    self.state.stage = "Agent 正在理解用户指令"
            elif event == "task_result":
                if event_data.get("task_id") != self.state.task_id:
                    return
                self.suppressing_traceback = False
                ok = bool(event_data.get("ok"))
                self.state.status = "success" if ok else "failed"
                self.state.stage = (
                    "任务完成，还需要什么？"
                    if ok
                    else "任务失败"
                )
                self.state.return_code = 0 if ok else 1
                self.state.finished_at = time.time()
                if ok:
                    self.state.final_message = "任务完成，还需要什么？"
                if not ok:
                    message = str(event_data.get("message", "")).strip()
                    self.state.logs.append(
                        f"任务失败：{message}" if message else "任务失败"
                    )
            elif event == "task_cancelled":
                if event_data.get("task_id") != self.state.task_id:
                    return
                self.suppressing_traceback = False
                home_ok = bool(event_data.get("home_ok"))
                message = str(
                    event_data.get(
                        "message",
                        "任务已停止",
                    )
                )
                self.state.status = "ready" if home_ok else "failed"
                self.state.stage = message
                self.state.return_code = 0 if home_ok else 1
                self.state.finished_at = time.time()
                self.state.final_message = message
                self.state.logs.append(message)
            elif event == "request_error":
                self.state.logs.append(
                    str(event_data.get("message", "工作进程请求错误"))
                )

    def _run_persistent_worker(self) -> None:
        environment = os.environ.copy()
        environment["ROBOTWIN_WEB_STREAM_DIR"] = str(CAMERA_DIR)
        environment["ROBOTWIN_WEB_STOP_FILE"] = str(STOP_REQUEST_PATH)
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [
                    "bash",
                    str(RUN_SCRIPT),
                    "--agent-worker",
                ],
                cwd=str(EXAMPLE_DIR.parents[1]),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self.lock:
                self.process = process
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                if clean.startswith(WORKER_EVENT_PREFIX):
                    self._handle_worker_event(
                        clean.removeprefix(WORKER_EVENT_PREFIX)
                    )
                else:
                    with self.lock:
                        initializing = self.state.status == "initializing"
                    if not initializing:
                        self._append_log(clean)
            return_code = process.wait()
            with self.lock:
                if self.process is process:
                    self.process = None
                if self.shutting_down:
                    return
                if self.state.status == "initializing":
                    self.state.status = "failed"
                    self.state.stage = "机器人场景初始化失败，请查看运行日志"
                    self.state.return_code = return_code
                    self.state.finished_at = time.time()
                elif self.state.status in {
                    "starting",
                    "understanding",
                    "planning",
                    "executing",
                    "verifying",
                    "stopping",
                }:
                    self.state.status = "failed"
                    self.state.stage = "机器人场景工作进程意外退出"
                    self.state.return_code = return_code
                    self.state.finished_at = time.time()
                else:
                    self.state.status = "failed"
                    self.state.stage = "机器人场景工作进程已退出"
                    self.state.return_code = return_code
                    self.state.finished_at = time.time()
        except Exception as exc:
            with self.lock:
                self.state.logs.append(f"Web启动常驻场景失败：{exc}")
                self.state.status = "failed"
                self.state.stage = "无法启动机器人场景"
                self.state.finished_at = time.time()
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None
            if process is not None and process.stdout is not None:
                process.stdout.close()
            if process is not None and process.stdin is not None:
                process.stdin.close()

    def close(self) -> None:
        with self.lock:
            self.shutting_down = True
            process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def _run_demo_task(self, command: str) -> None:
        steps = (
            ("understanding", "Agent 正在理解用户指令", 0.35),
            ("planning", "Agent 正在结合视觉状态规划技能", 0.45),
            ("executing", "机器人正在执行抓取与放置", 0.8),
        )
        for status, stage, delay in steps:
            with self.lock:
                self.state.status = status
                self.state.stage = stage
                self.state.logs.append(stage)
            time.sleep(delay)
        with self.lock:
            self.state.status = "success"
            self.state.stage = "任务完成，还需要什么？"
            self.state.final_message = "任务完成，还需要什么？"
            self.state.return_code = 0
            self.state.finished_at = time.time()


def _latest_path(pattern: str) -> Path | None:
    matches = list(EXAMPLE_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def camera_path(
    camera_name: str,
    *,
    allow_fallback: bool,
) -> Path | None:
    live_path = CAMERA_DIR / f"{camera_name}.jpg"
    if live_path.is_file():
        return live_path
    if not allow_fallback:
        return None
    fallbacks = {
        "head_camera": (
            _latest_path(
                "agent_vision_observations/observation_*/head_camera_rgb.png"
            ),
            EXAMPLE_DIR / "vision_results/head_camera_rgb.png",
            EXAMPLE_DIR / "scene_preview.png",
        ),
        "left_camera": (
            _latest_path(
                "agent_wrist_observations/observation_*/"
                "part_*/left_camera_rgb.png"
            ),
        ),
        "right_camera": (
            _latest_path(
                "agent_wrist_observations/observation_*/"
                "part_*/right_camera_rgb.png"
            ),
        ),
    }
    for candidate in fallbacks.get(camera_name, ()):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RoboTwinDashboard/1.0"

    @property
    def manager(self) -> AgentProcessManager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, format_text: str, *args: Any) -> None:
        # Keep the terminal focused on Agent logs.
        return

    def _send_json(
        self,
        data: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, no_cache: bool = False) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime_type = mimetypes.guess_type(path.name)[0]
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            mime_type or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "no-store" if no_cache else "public, max-age=300",
        )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length非法") from exc
        if length <= 0 or length > 16_384:
            raise ValueError("请求内容为空或过大")
        try:
            return json.loads(
                self.rfile.read(length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求不是有效JSON") from exc

    def do_GET(self) -> None:
        route = unquote(urlparse(self.path).path)
        if route == "/api/status":
            self._send_json(self.manager.snapshot())
            return
        if route.startswith("/api/camera/"):
            camera_name = route.removeprefix("/api/camera/")
            if camera_name not in CAMERA_NAMES:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = camera_path(
                camera_name,
                allow_fallback=self.manager.demo,
            )
            if path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self._send_file(path, no_cache=True)
            return
        if route == "/api/artifacts/video":
            self._send_file(SESSION_VIDEO_PATH, no_cache=True)
            return
        if route == "/api/artifacts/trace":
            self._send_file(
                EXAMPLE_DIR / "agent_execution_trace.json",
                no_cache=True,
            )
            return

        static_name = "index.html" if route == "/" else route.lstrip("/")
        static_path = (WEB_DIR / static_name).resolve()
        if WEB_DIR not in static_path.parents and static_path != WEB_DIR:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(
            static_path,
            no_cache=static_path.name == "index.html",
        )

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(
                {"ok": False, "message": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if route == "/api/command":
            ok, message = self.manager.start(
                str(payload.get("command", ""))
            )
            self._send_json(
                {"ok": ok, "message": message},
                HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT,
            )
            return
        if route == "/api/stop":
            ok, message = self.manager.stop()
            self._send_json(
                {"ok": ok, "message": message},
                HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT,
            )
            return
        if route == "/api/reload":
            ok, message = self.manager.reload()
            self._send_json(
                {"ok": ok, "message": message},
                HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT,
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="只测试网页交互，不启动RoboTwin和大模型",
    )
    options = parser.parse_args()

    server = ThreadingHTTPServer(
        (options.host, options.port),
        DashboardHandler,
    )
    manager = AgentProcessManager(demo=options.demo)
    server.manager = manager  # type: ignore[attr-defined]
    print(
        f"RoboTwin Agent 网页已启动：http://{options.host}:{options.port}"
    )
    if options.demo:
        print("当前为网页布局演示模式，不会驱动机器人。")
    elif options.host == "0.0.0.0":
        print(
            "服务已监听所有网卡；请只在可信网络使用，"
            "或通过SSH端口转发访问。"
        )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在关闭RoboTwin Agent网页……")
    finally:
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
