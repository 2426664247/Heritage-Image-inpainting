import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from inpaint_core import DEFAULT_MODEL, DEFAULT_UNET_WEIGHTS, PROJECT_DIR, UI_OUTPUT_ROOT, InpaintRunner, mask_from_polygons
except ModuleNotFoundError:
    from scripts.inpaint_core import DEFAULT_MODEL, DEFAULT_UNET_WEIGHTS, PROJECT_DIR, UI_OUTPUT_ROOT, InpaintRunner, mask_from_polygons


UI_DIR = PROJECT_DIR / "ui"
TRAIN_SCRIPT = PROJECT_DIR / "scripts" / "train_lora_inpaint_official.py"
TRAIN_DATA_IMAGE_DIRS = "train_data/train,train_data/raw/image,train_data/test"
TRAIN_DATA_MASK_DIR = "train_data/mask"

app = FastAPI(title="Heritage Image Inpainting UI")
executor = ThreadPoolExecutor(max_workers=1)
runner = InpaintRunner()
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.RLock()
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
INVALID_EXPERIMENT_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def now_ts() -> float:
    return time.time()


def new_job_state(job_id: str, experiment_name: str) -> dict[str, Any]:
    job_url_id = quote(job_id)
    return {
        "id": job_id,
        "experiment_name": experiment_name,
        "output_dir": str(UI_OUTPUT_ROOT / job_id),
        "status": "queued",
        "phase": "Queued",
        "logs": [],
        "error": None,
        "input_url": f"/jobs/{job_url_id}/input.png",
        "mask_url": f"/jobs/{job_url_id}/mask.png",
        "result_url": None,
        "collage_url": None,
        "repair_area_url": None,
        "progress": 0,
        "progress_label": "等待任务",
        "created_at": now_ts(),
        "updated_at": now_ts(),
    }


def update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.update(updates)
        job["updated_at"] = now_ts()


def log_job(job_id: str, message: str) -> None:
    message = message.strip()
    if not message:
        return
    timestamp = time.strftime("%H:%M:%S")
    with jobs_lock:
        job = jobs[job_id]
        job["logs"].append(f"[{timestamp}] {message}")
        job["logs"] = job["logs"][-120:]
        job["updated_at"] = now_ts()


def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(jobs[job_id])


def parse_polygons(raw: str) -> list[list[dict[str, float]]]:
    try:
        polygons = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid polygons JSON") from exc

    if not isinstance(polygons, list):
        raise HTTPException(status_code=400, detail="Polygons must be a list")

    parsed = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            continue
        points = []
        for point in polygon:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                continue
            points.append({"x": float(point["x"]), "y": float(point["y"])})
        if len(points) >= 3:
            parsed.append(points)

    if not parsed:
        raise HTTPException(status_code=400, detail="Draw at least one closed polygon")
    return parsed


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def experiment_id_from_name(raw_name: str) -> tuple[str, str]:
    experiment_name = raw_name.strip()
    if not experiment_name:
        raise HTTPException(status_code=400, detail="请输入实验名称")
    if len(experiment_name) > 80:
        raise HTTPException(status_code=400, detail="实验名称不能超过 80 个字符")

    experiment_id = INVALID_EXPERIMENT_CHARS_RE.sub("_", experiment_name).strip(" .")
    if not experiment_id:
        raise HTTPException(status_code=400, detail="实验名称不能只包含路径特殊字符")
    if experiment_id.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        experiment_id = f"{experiment_id}_experiment"
    return experiment_name, experiment_id


def summarize_training_output(raw: str) -> list[str]:
    messages = []
    text = ANSI_RE.sub("", raw).replace("\r", "\n")
    for part in text.splitlines():
        part = part.strip()
        if not part:
            continue
        if "lora-training" in part and "loss=" in part:
            percent = re.search(r"(\d{1,3})%", part)
            loss = re.search(r"loss=([0-9.eE+-]+)", part)
            if percent and loss:
                messages.append(f"Training progress {percent.group(1)}%, loss {loss.group(1)}")
            elif loss:
                messages.append(f"Training loss {loss.group(1)}")
        elif "Loading pipeline components" in part and "100%" in part:
            messages.append("Training pipeline loaded.")
        elif any(token in part for token in ("Traceback", "RuntimeError", "ValueError", "Error:", "Exception")):
            messages.append(part[-500:])
    return messages


def parse_training_progress(message: str) -> Optional[int]:
    match = re.search(r"Training progress (\d{1,3})%", message)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def run_training(job_id: str, job_dir: Path) -> Path:
    train_out = job_dir / "training"
    train_out.mkdir(parents=True, exist_ok=True)
    runner.unload()
    log_job(job_id, "Quick LoRA training started.")
    update_job(job_id, phase="Training LoRA", progress=5, progress_label="正在快速训练 LoRA")

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--image_dirs",
        TRAIN_DATA_IMAGE_DIRS,
        "--mask_dir",
        TRAIN_DATA_MASK_DIR,
        "--size",
        "384",
        "--batch",
        "1",
        "--accum",
        "1",
        "--lr",
        "1e-5",
        "--steps",
        "20",
        "--rank",
        "8",
        "--out",
        str(train_out),
        "--model",
        str(DEFAULT_MODEL),
        "--log_interval",
        "1",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    seen_training_logs = set()
    for line in process.stdout:
        for message in summarize_training_output(line):
            if message in seen_training_logs:
                continue
            seen_training_logs.add(message)
            log_job(job_id, message)
            training_progress = parse_training_progress(message)
            if training_progress is not None:
                update_job(
                    job_id,
                    progress=5 + int(training_progress * 0.45),
                    progress_label=f"正在快速训练 LoRA {training_progress}%",
                )
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Quick LoRA training failed with exit code {return_code}")

    lora_path = train_out / "lora_unet.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"Training finished but LoRA was not created: {lora_path}")
    log_job(job_id, "Quick LoRA training finished.")
    update_job(job_id, progress=55, progress_label="LoRA 快速训练完成")
    return lora_path


def run_job(job_id: str, params: dict[str, Any]) -> None:
    job_dir = UI_OUTPUT_ROOT / job_id
    job_url_id = quote(job_id)
    lora_path = None
    try:
        update_job(job_id, status="running", phase="Preparing", progress=3, progress_label="正在准备输入")
        log_job(job_id, "Received image and mask.")

        runner.unet_weights = DEFAULT_UNET_WEIGHTS if params["use_partial_unet"] else None
        if runner.unet_weights:
            log_job(job_id, "Using partial-tuned UNet.")
        else:
            log_job(job_id, "Using base model UNet.")

        if params["train_before_repair"]:
            lora_path = run_training(job_id, job_dir)

        update_job(job_id, phase="Inpainting", progress=60, progress_label="正在加载修复模型")
        log_job(job_id, "Loading model and starting repair.")
        def update_inpaint_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            percent = max(0, min(100, int(done * 100 / total)))
            update_job(
                job_id,
                progress=60 + int(percent * 0.35),
                progress_label=f"扩散修复中 {done}/{total}",
            )

        result = runner.run_single(
            image_path=job_dir / "input.png",
            mask_path=job_dir / "mask.png",
            output_dir=job_dir,
            prompt=params["prompt"],
            steps=params["steps"],
            guidance=params["guidance"],
            size=params["size"],
            seed=params["seed"],
            rows=1,
            mask_mode="white",
            lora_path=lora_path,
            logger=lambda message: log_job(job_id, message),
            progress_callback=update_inpaint_progress,
        )

        update_job(
            job_id,
            status="succeeded",
            phase="Done",
            progress=100,
            progress_label="修复完成",
            result_url=f"/jobs/{job_url_id}/{result.result_path.name}",
            collage_url=f"/jobs/{job_url_id}/{result.collage_path.name}",
            repair_area_url=f"/jobs/{job_url_id}/{result.repair_area_path.name}",
        )
        log_job(job_id, "Job completed.")
    except Exception as exc:
        update_job(job_id, status="failed", phase="Failed", error=str(exc), progress_label="修复失败")
        log_job(job_id, f"Failed: {exc}")
        runner.unload()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    polygons: str = Form(...),
    experiment_name: str = Form(...),
    prompt: str = Form(""),
    steps: int = Form(30),
    guidance: float = Form(5.0),
    size: int = Form(512),
    seed: int = Form(1234),
    use_partial_unet: bool = Form(True),
    train_before_repair: bool = Form(False),
) -> dict[str, Any]:
    display_name, job_id = experiment_id_from_name(experiment_name)
    parsed_polygons = parse_polygons(polygons)
    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    job_dir = UI_OUTPUT_ROOT / job_id
    mask = mask_from_polygons(pil_image.size, parsed_polygons)
    if mask.getbbox() is None:
        raise HTTPException(status_code=400, detail="Mask is empty")

    params = {
        "prompt": prompt.strip(),
        "steps": clamp_int(steps, 1, 100),
        "guidance": clamp_float(guidance, 1.0, 20.0),
        "size": clamp_int(size, 128, 2048),
        "seed": clamp_int(seed, 0, 2_147_483_647),
        "use_partial_unet": bool(use_partial_unet),
        "train_before_repair": bool(train_before_repair),
    }

    with jobs_lock:
        matching_job_id = next((known_id for known_id in jobs if known_id.casefold() == job_id.casefold()), None)
        existing_job = jobs.get(matching_job_id) if matching_job_id else None
        if existing_job and existing_job["status"] in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="同名实验正在运行，请等待完成后再覆盖")
        if matching_job_id and matching_job_id != job_id:
            jobs.pop(matching_job_id)

        resolved_root = UI_OUTPUT_ROOT.resolve()
        resolved_job_dir = job_dir.resolve()
        if resolved_job_dir.parent != resolved_root:
            raise HTTPException(status_code=400, detail="实验名称生成的目录无效")
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        pil_image.save(job_dir / "input.png")
        mask.save(job_dir / "mask.png")
        (job_dir / "request.json").write_text(
            json.dumps(
                {"experiment_name": display_name, "params": params, "polygons": parsed_polygons},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        jobs[job_id] = new_job_state(job_id, display_name)
        log_job(job_id, f"产物目录：{job_dir}")
    executor.submit(run_job, job_id, params)
    return get_job(job_id)


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict[str, Any]:
    return get_job(job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the heritage inpainting web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import uvicorn

    UI_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port)


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
UI_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/jobs", StaticFiles(directory=str(UI_OUTPUT_ROOT)), name="jobs")


if __name__ == "__main__":
    main()
