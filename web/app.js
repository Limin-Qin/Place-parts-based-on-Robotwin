const POLL_INTERVAL_MS = 700;
const CAMERA_INTERVAL_MS = 350;
const RUNNING_STATES = new Set([
  "initializing",
  "starting",
  "understanding",
  "planning",
  "executing",
  "verifying",
  "stopping",
  "reloading",
]);
const STOPPABLE_STATES = new Set([
  "starting",
  "understanding",
  "planning",
  "executing",
  "verifying",
]);

const ui = {
  form: document.querySelector("#commandForm"),
  input: document.querySelector("#commandInput"),
  send: document.querySelector("#sendButton"),
  stop: document.querySelector("#stopButton"),
  reload: document.querySelector("#reloadButton"),
  terminal: document.querySelector("#terminal"),
  stage: document.querySelector("#currentStage"),
  status: document.querySelector("#systemStatus"),
  pulse: document.querySelector("#statusPulse"),
  freshness: document.querySelector("#cameraFreshness"),
  toast: document.querySelector("#toast"),
  mode: document.querySelector("#modeBadge"),
  video: document.querySelector("#videoLink"),
  trace: document.querySelector("#traceLink"),
};

let lastLogsSignature = null;
let cameraSequence = -1;
let toastTimer = null;

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 2600);
}

function statusLabel(status) {
  const labels = {
    idle: "待机",
    initializing: "场景加载中",
    ready: "场景就绪",
    starting: "场景加载中",
    understanding: "理解指令",
    planning: "规划中",
    executing: "执行中",
    verifying: "验证中",
    stopping: "停止中",
    reloading: "重新加载中",
    stopped: "已停止",
    success: "任务成功",
    failed: "任务失败",
  };
  return labels[status] || status;
}

function renderLogs(logs) {
  const signature = JSON.stringify(logs);
  if (signature === lastLogsSignature) return;
  lastLogsSignature = signature;
  ui.terminal.innerHTML = "";
  if (!logs.length) {
    ui.terminal.innerHTML =
      '<p class="terminal-placeholder">任务日志将在这里显示。Agent会理解文字、观察场景、编排技能并控制机器人执行。</p>';
    return;
  }
  logs.forEach((line) => {
    const row = document.createElement("p");
    row.textContent = `› ${line}`;
    ui.terminal.appendChild(row);
  });
  ui.terminal.scrollTop = ui.terminal.scrollHeight;
}

function renderState(state) {
  const running = RUNNING_STATES.has(state.status);
  ui.status.textContent = statusLabel(state.status);
  ui.stage.textContent = state.stage || "等待用户输入任务";
  ui.send.disabled = running;
  ui.stop.disabled = !STOPPABLE_STATES.has(state.status);
  ui.reload.disabled = running;
  ui.input.disabled = running;
  ui.mode.textContent = state.demo_mode ? "DEMO" : "REAL";

  ui.pulse.className = "pulse";
  if (running) ui.pulse.classList.add("running");
  if (state.status === "ready" || state.status === "success") {
    ui.pulse.classList.add("success");
  }
  if (state.status === "failed") ui.pulse.classList.add("failed");

  renderLogs(state.logs || []);
  ui.video.classList.toggle(
    "available",
    Boolean(state.artifacts?.video_ready),
  );
  ui.trace.classList.toggle(
    "available",
    Boolean(state.artifacts?.trace_ready),
  );

  const updated = state.camera_state?.updated_unix_seconds;
  if (updated) {
    const age = Math.max(0, Date.now() / 1000 - updated);
    if (RUNNING_STATES.has(state.status) && state.status !== "initializing") {
      ui.freshness.textContent =
        age < 4 ? "实时画面已连接" : `最近更新 ${Math.round(age)} 秒前`;
    } else {
      ui.freshness.textContent = "场景画面已就绪";
    }
  } else {
    ui.freshness.textContent = state.demo_mode
      ? "显示最近一次采集画面"
      : "等待场景相机";
    if (!state.demo_mode) {
      document.querySelectorAll("img[data-camera]").forEach((image) => {
        image.classList.remove("ready");
        image.removeAttribute("src");
      });
    }
  }
  cameraSequence = state.camera_state?.sequence ?? cameraSequence;
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderState(await response.json());
  } catch (error) {
    ui.status.textContent = "服务断开";
    ui.stage.textContent = "无法连接Web后端";
    ui.pulse.className = "pulse failed";
  }
}

function refreshCameras() {
  const stamp = `${Date.now()}-${cameraSequence}`;
  document.querySelectorAll("img[data-camera]").forEach((image) => {
    const camera = image.dataset.camera;
    const probe = new Image();
    probe.onload = () => {
      image.src = probe.src;
      image.classList.add("ready");
    };
    probe.src = `/api/camera/${camera}?t=${stamp}`;
  });
}

async function submitCommand(command) {
  const value = command.trim();
  if (!value) {
    showToast("请先输入机器人任务");
    ui.input.focus();
    return;
  }
  try {
    const response = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: value }),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.message);
    ui.input.value = "";
    showToast(result.message);
    await pollStatus();
  } catch (error) {
    showToast(error.message || "任务提交失败");
  }
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitCommand(ui.input.value);
});

ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    ui.form.requestSubmit();
  }
});

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.input.value = button.dataset.command;
    ui.input.focus();
  });
});

ui.stop.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.message);
    showToast(result.message);
    await pollStatus();
  } catch (error) {
    showToast(error.message || "停止任务失败");
  }
});

ui.reload.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/reload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.message);
    showToast(result.message);
    await pollStatus();
  } catch (error) {
    showToast(error.message || "重新加载场景失败");
  }
});

pollStatus();
refreshCameras();
setInterval(pollStatus, POLL_INTERVAL_MS);
setInterval(refreshCameras, CAMERA_INTERVAL_MS);
