const state = {
  file: null,
  image: null,
  points: [],
  polygons: [],
  hover: null,
  jobId: null,
  pollTimer: null,
};

const els = {
  imageInput: document.getElementById("imageInput"),
  canvas: document.getElementById("maskCanvas"),
  emptyState: document.getElementById("emptyState"),
  undoPoint: document.getElementById("undoPoint"),
  finishPolygon: document.getElementById("finishPolygon"),
  clearMask: document.getElementById("clearMask"),
  pointCounter: document.getElementById("pointCounter"),
  startRepair: document.getElementById("startRepair"),
  runState: document.getElementById("runState"),
  experimentName: document.getElementById("experimentName"),
  prompt: document.getElementById("prompt"),
  steps: document.getElementById("steps"),
  guidance: document.getElementById("guidance"),
  size: document.getElementById("size"),
  seed: document.getElementById("seed"),
  usePartialUnet: document.getElementById("usePartialUnet"),
  trainBeforeRepair: document.getElementById("trainBeforeRepair"),
  logBox: document.getElementById("logBox"),
  resultFrame: document.getElementById("resultFrame"),
  resultImage: document.getElementById("resultImage"),
  openRepairArea: document.getElementById("openRepairArea"),
  openResult: document.getElementById("openResult"),
  progressLabel: document.getElementById("progressLabel"),
  progressValue: document.getElementById("progressValue"),
  progressFill: document.getElementById("progressFill"),
};

const ctx = els.canvas.getContext("2d");

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function setRunState(text, mode = "") {
  els.runState.textContent = text;
  els.runState.className = `run-state ${mode}`.trim();
}

function setProgress(value, label = "等待任务") {
  const progress = clampNumber(value, 0, 100, 0);
  els.progressFill.style.width = `${progress}%`;
  els.progressValue.textContent = `${Math.round(progress)}%`;
  els.progressLabel.textContent = label;
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function updateButtons() {
  const hasImage = Boolean(state.image);
  const hasCurrent = state.points.length > 0;
  const hasClosed = state.polygons.length > 0;
  const hasExperimentName = Boolean(els.experimentName.value.trim());
  els.undoPoint.disabled = !hasCurrent;
  els.finishPolygon.disabled = state.points.length < 3;
  els.clearMask.disabled = !hasCurrent && !hasClosed;
  els.startRepair.disabled = !hasImage || !hasClosed || !hasExperimentName || els.startRepair.dataset.busy === "true";
  const pointTotal = state.polygons.reduce((sum, poly) => sum + poly.length, 0) + state.points.length;
  const regionText = state.polygons.length ? `，${state.polygons.length} 个区域` : "";
  els.pointCounter.textContent = `${pointTotal} 个点${regionText}`;
}

function canvasPoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (els.canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (els.canvas.height / rect.height);
  return {
    x: Math.max(0, Math.min(els.canvas.width - 1, x)),
    y: Math.max(0, Math.min(els.canvas.height - 1, y)),
  };
}

function drawPolygon(points, fillStyle, strokeStyle, closePath = true) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) {
    ctx.lineTo(point.x, point.y);
  }
  if (closePath) ctx.closePath();
  if (fillStyle) {
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = Math.max(2, Math.round(Math.max(els.canvas.width, els.canvas.height) / 380));
  ctx.stroke();
}

function drawPoint(point, index) {
  const radius = Math.max(4, Math.round(Math.max(els.canvas.width, els.canvas.height) / 180));
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#0f766e";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#fffdf8";
  ctx.stroke();
  ctx.fillStyle = "#fffdf8";
  ctx.font = `${Math.max(10, radius + 4)}px Segoe UI, Arial`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(index + 1), point.x, point.y);
}

function redraw() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!state.image) return;
  ctx.drawImage(state.image, 0, 0, els.canvas.width, els.canvas.height);

  for (const polygon of state.polygons) {
    drawPolygon(polygon, "rgba(255, 255, 255, 0.48)", "#b45309", true);
    polygon.forEach(drawPoint);
  }

  if (state.points.length) {
    const preview = state.hover ? [...state.points, state.hover] : state.points;
    drawPolygon(preview, "rgba(15, 118, 110, 0.16)", "#0f766e", false);
    state.points.forEach(drawPoint);
  }
}

function resetResult() {
  els.resultImage.hidden = true;
  els.resultImage.removeAttribute("src");
  els.openRepairArea.hidden = true;
  els.openRepairArea.href = "#";
  els.openResult.hidden = true;
  els.openResult.href = "#";
  els.resultFrame.querySelector("span").hidden = false;
}

function loadImage(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(url);
    state.file = file;
    state.image = image;
    state.points = [];
    state.polygons = [];
    state.hover = null;
    els.canvas.width = image.naturalWidth;
    els.canvas.height = image.naturalHeight;
    els.canvas.style.display = "block";
    els.emptyState.style.display = "none";
    resetResult();
    els.logBox.textContent = "等待任务";
    setProgress(0, "等待任务");
    setRunState("就绪");
    redraw();
    updateButtons();
  };
  image.src = url;
}

function finishPolygon() {
  if (state.points.length < 3) return;
  state.polygons.push([...state.points]);
  state.points = [];
  state.hover = null;
  redraw();
  updateButtons();
}

function clearMask() {
  state.points = [];
  state.polygons = [];
  state.hover = null;
  redraw();
  updateButtons();
}

async function createJob() {
  const experimentName = els.experimentName.value.trim();
  if (!state.file || !state.polygons.length || !experimentName) return;
  els.startRepair.dataset.busy = "true";
  updateButtons();
  resetResult();
  setRunState("运行中", "busy");
  els.logBox.textContent = "提交任务...";
  setProgress(2, "提交任务");

  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("polygons", JSON.stringify(state.polygons));
  formData.append("experiment_name", experimentName);
  formData.append("prompt", els.prompt.value);
  formData.append("steps", String(clampNumber(els.steps.value, 1, 100, 30)));
  formData.append("guidance", String(clampNumber(els.guidance.value, 1, 20, 5)));
  formData.append("size", String(clampNumber(els.size.value, 128, 2048, 512)));
  formData.append("seed", String(clampNumber(els.seed.value, 0, 2147483647, 1234)));
  formData.append("use_partial_unet", els.usePartialUnet.checked ? "true" : "false");
  formData.append("train_before_repair", els.trainBeforeRepair.checked ? "true" : "false");

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "任务提交失败");
    }
    state.jobId = data.id;
    renderJob(data);
    pollJob();
  } catch (error) {
    els.logBox.textContent = error.message;
    setRunState("失败");
    els.startRepair.dataset.busy = "false";
    updateButtons();
  }
}

function renderJob(job) {
  els.logBox.textContent = job.logs && job.logs.length ? job.logs.join("\n") : job.phase;
  setProgress(job.progress || 0, job.progress_label || job.phase || "运行中");
  if (job.status === "succeeded") {
    setRunState("完成", "done");
    els.startRepair.dataset.busy = "false";
    const imageUrl = `${job.collage_url || job.result_url}?t=${Date.now()}`;
    els.resultImage.src = imageUrl;
    els.resultImage.hidden = false;
    els.resultFrame.querySelector("span").hidden = true;
    els.openRepairArea.href = job.repair_area_url;
    els.openRepairArea.hidden = !job.repair_area_url;
    els.openResult.href = job.collage_url || job.result_url;
    els.openResult.hidden = false;
    stopPolling();
  } else if (job.status === "failed") {
    setRunState("失败");
    setProgress(job.progress || 0, job.progress_label || "修复失败");
    els.startRepair.dataset.busy = "false";
    if (job.error) {
      els.logBox.textContent = `${els.logBox.textContent}\n${job.error}`.trim();
    }
    stopPolling();
  } else {
    setRunState(job.status === "queued" ? "排队中" : "运行中", "busy");
  }
  updateButtons();
}

async function pollJob() {
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    if (!state.jobId) return;
    try {
      const response = await fetch(`/api/jobs/${state.jobId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "读取任务失败");
      renderJob(data);
    } catch (error) {
      els.logBox.textContent = `${els.logBox.textContent}\n${error.message}`.trim();
      setRunState("连接中断");
      stopPolling();
      els.startRepair.dataset.busy = "false";
      updateButtons();
    }
  }, 1200);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

els.imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) loadImage(file);
});

els.canvas.addEventListener("click", (event) => {
  if (!state.image || els.startRepair.dataset.busy === "true") return;
  state.points.push(canvasPoint(event));
  redraw();
  updateButtons();
});

els.canvas.addEventListener("mousemove", (event) => {
  if (!state.image || !state.points.length) return;
  state.hover = canvasPoint(event);
  redraw();
});

els.canvas.addEventListener("mouseleave", () => {
  state.hover = null;
  redraw();
});

els.undoPoint.addEventListener("click", () => {
  state.points.pop();
  state.hover = null;
  redraw();
  updateButtons();
});

els.finishPolygon.addEventListener("click", finishPolygon);
els.clearMask.addEventListener("click", clearMask);
els.startRepair.addEventListener("click", createJob);
els.experimentName.addEventListener("input", updateButtons);

window.addEventListener("load", refreshIcons);
refreshIcons();
updateButtons();
