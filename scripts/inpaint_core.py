import gc
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from diffusers import StableDiffusionInpaintPipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_DIR
    / "model"
    / "models--stabilityai--stable-diffusion-2-inpainting"
    / "snapshots"
    / "81a84f49b15956b60b4272a405ad3daef3da4590"
)
DEFAULT_UNET_WEIGHTS = PROJECT_DIR / "weights" / "unet_partial_tuned.safetensors"
UI_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "ui" / "jobs"


LogFn = Optional[Callable[[str], None]]
ProgressFn = Optional[Callable[[int, int], None]]


@dataclass
class InpaintResult:
    result_path: Path
    collage_path: Path
    repair_area_path: Path
    seeds: list[int]


def _log(logger: LogFn, message: str) -> None:
    if logger:
        logger(message)


def binarize_mask(mask: Image.Image) -> Image.Image:
    mask = mask.convert("L")
    return mask.point(lambda p: 255 if p >= 128 else 0)


def round_to_multiple(x: float, m: int = 8) -> int:
    return max(m, int(round(x / m)) * m)


def fit_to_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    if not max_size or max_size <= 0:
        return round_to_multiple(width, 8), round_to_multiple(height, 8)
    scale = min(1.0, max_size / max(width, height))
    return round_to_multiple(width * scale, 8), round_to_multiple(height * scale, 8)


def make_repair_mask(mask_bw: Image.Image, mask_mode: str) -> Image.Image:
    if mask_mode.lower() == "black":
        return ImageOps.invert(mask_bw)
    return mask_bw


def overlay_mask_on_image(img: Image.Image, mask_bw: Image.Image, mask_mode: str) -> Image.Image:
    repair_mask = make_repair_mask(mask_bw, mask_mode)
    white = Image.new("RGB", img.size, (255, 255, 255))
    return Image.composite(white, img.convert("RGB"), repair_mask)


def caption_font(caption_height: int):
    size = max(16, int(caption_height * 0.48))
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def make_row(
    left: Image.Image,
    middle: Image.Image,
    right: Image.Image,
    spacing_h: int,
    captions: Optional[list[str]] = None,
) -> Image.Image:
    width, height = left.size
    caption_height = max(34, int(height * 0.08)) if captions else 0
    row = Image.new("RGB", (width * 3 + spacing_h * 2, height + caption_height), color=(255, 255, 255))
    row.paste(left, (0, 0))
    row.paste(middle, (width + spacing_h, 0))
    row.paste(right, (width * 2 + spacing_h * 2, 0))
    if captions:
        draw = ImageDraw.Draw(row)
        font = caption_font(caption_height)
        centers = [
            width // 2,
            width + spacing_h + width // 2,
            width * 2 + spacing_h * 2 + width // 2,
        ]
        y = height + caption_height // 2
        for caption, x in zip(captions, centers):
            draw.text((x, y), caption, fill=(31, 37, 41), anchor="mm", font=font)
    return row


def stack_rows(rows: list[Image.Image], spacing_v: int) -> Optional[Image.Image]:
    if not rows:
        return None
    width, row_height = rows[0].size
    height = row_height * len(rows) + spacing_v * (len(rows) - 1)
    grid = Image.new("RGB", (width, height), color=(255, 255, 255))
    y = 0
    for index, row in enumerate(rows):
        grid.paste(row, (0, y))
        y += row_height + (spacing_v if index < len(rows) - 1 else 0)
    return grid


def mask_from_polygons(
    image_size: tuple[int, int],
    polygons: Iterable[Iterable[dict]],
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image_size
    for polygon in polygons:
        points = []
        for point in polygon:
            x = max(0, min(width - 1, float(point["x"])))
            y = max(0, min(height - 1, float(point["y"])))
            points.append((x, y))
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return mask


class InpaintRunner:
    def __init__(
        self,
        model_path: os.PathLike = DEFAULT_MODEL,
        unet_weights: Optional[os.PathLike] = DEFAULT_UNET_WEIGHTS,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.unet_weights = Path(unet_weights) if unet_weights else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._cache_key = None

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._cache_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_pipeline(self, lora_path: Optional[os.PathLike] = None, logger: LogFn = None):
        lora = Path(lora_path) if lora_path else None
        cache_key = (str(self.model_path), str(self.unet_weights), str(lora) if lora else "")
        if self._pipe is not None and self._cache_key == cache_key:
            return self._pipe

        self.unload()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        _log(logger, "Loading inpainting model...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
        )
        pipe = pipe.to(self.device)

        if self.unet_weights and self.unet_weights.exists():
            _log(logger, f"Loading UNet weights: {self.unet_weights.name}")
            import safetensors.torch as st

            state_dict = st.load_file(str(self.unet_weights))
            pipe.unet.load_state_dict(state_dict, strict=False)
        elif self.unet_weights:
            _log(logger, f"UNet weights not found, using base model: {self.unet_weights}")

        if lora:
            if not lora.exists():
                raise FileNotFoundError(f"LoRA weights not found: {lora}")
            _log(logger, f"Loading LoRA weights: {lora.name}")
            if hasattr(pipe.unet, "load_lora_adapter"):
                pipe.unet.load_lora_adapter(str(lora), prefix=None)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(str(lora))
                pipe.fuse_lora()

        self._pipe = pipe
        self._cache_key = cache_key
        return pipe

    def run_single(
        self,
        image_path: os.PathLike,
        mask_path: os.PathLike,
        output_dir: os.PathLike,
        prompt: str = "",
        steps: int = 30,
        guidance: float = 5.0,
        size: int = 512,
        seed: int = 1234,
        rows: int = 1,
        mask_mode: str = "white",
        lora_path: Optional[os.PathLike] = None,
        collage_spacing_h: int = 20,
        collage_spacing_v: int = 20,
        logger: LogFn = None,
        progress_callback: ProgressFn = None,
    ) -> InpaintResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipe = self.load_pipeline(lora_path=lora_path, logger=logger)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        mask_bw = binarize_mask(mask)
        overlay = overlay_mask_on_image(image, mask_bw, mask_mode)
        mask_for_pipe = make_repair_mask(mask_bw, mask_mode)

        width, height = image.size
        target_width, target_height = fit_to_size(width, height, size)
        image_pipe = image.resize((target_width, target_height), Image.LANCZOS)
        mask_pipe = mask_for_pipe.resize((target_width, target_height), Image.NEAREST)

        result_path = output_dir / "result.png"
        collage_path = output_dir / "collage.png"
        repair_area_path = output_dir / "repair_area.png"
        overlay.save(repair_area_path)
        row_images = []
        seeds = []
        base_seed = seed or 0

        _log(logger, "Starting inpainting...")
        for row_index in range(max(1, rows)):
            generator = torch.Generator(device=self.device)
            row_seed = base_seed + row_index if base_seed > 0 else torch.seed()
            generator.manual_seed(row_seed)
            seeds.append(int(row_seed))

            context = torch.autocast(device_type="cuda") if self.device == "cuda" else nullcontext()
            def on_step_end(pipe, step_index, timestep, callback_kwargs):
                if progress_callback:
                    progress_callback(int(step_index) + 1, int(steps))
                return callback_kwargs

            with torch.inference_mode(), context:
                result = pipe(
                    prompt=prompt,
                    image=image_pipe,
                    mask_image=mask_pipe,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=on_step_end if progress_callback else None,
                ).images[0]

            result = result.resize((width, height), Image.LANCZOS)
            if row_index == 0:
                result.save(result_path)
            row_images.append(
                make_row(
                    image,
                    overlay,
                    result,
                    collage_spacing_h,
                    captions=["原图", "拟修复区域", "修复结果"],
                )
            )

        collage = stack_rows(row_images, collage_spacing_v)
        if collage is not None:
            collage.save(collage_path)
        _log(logger, "Saved result and comparison image.")
        return InpaintResult(
            result_path=result_path,
            collage_path=collage_path,
            repair_area_path=repair_area_path,
            seeds=seeds,
        )
