import argparse
import os
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInpaintPipeline

def center_crop_resize(img, size):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    return img

def binarize_mask(mask):
    mask = mask.convert("L")
    mask = mask.point(lambda p: 255 if p >= 128 else 0)
    return mask

def overlay_mask_on_image(img, mask_bw, white_on_black=True):
    img = img.copy()
    mask_arr = mask_bw.convert("L")
    w, h = img.size
    img_arr = img.convert("RGB")
    img_pix = img_arr.load()
    m_pix = mask_arr.load()
    for y in range(h):
        for x in range(w):
            m = m_pix[x, y]
            if white_on_black:
                if m == 0:
                    img_pix[x, y] = (255, 255, 255)
            else:
                if m == 255:
                    img_pix[x, y] = (255, 255, 255)
    return img_arr

def make_row(img_left, img_mid, img_right, spacing_h):
    w, h = img_left.size
    W = w * 3 + spacing_h * 2
    row = Image.new("RGB", (W, h), color=(255, 255, 255))
    row.paste(img_left, (0, 0))
    row.paste(img_mid, (w + spacing_h, 0))
    row.paste(img_right, (2 * w + 2 * spacing_h, 0))
    return row

def stack_rows(rows, spacing_v):
    if not rows:
        return None
    W, h = rows[0].size
    H = h * len(rows) + spacing_v * (len(rows) - 1)
    grid = Image.new("RGB", (W, H), color=(255, 255, 255))
    y = 0
    for i, r in enumerate(rows):
        grid.paste(r, (0, y))
        y += h + (spacing_v if i < len(rows) - 1 else 0)
    return grid

def round_to_multiple(x, m=8):
    return max(m, int(round(x / m)) * m)

def letterbox(img, size, fill):
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), fill), (0, 0, size, size), (w, h)
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_r = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new(img_r.mode, (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(img_r, (left, top))
    box = (left, top, left + nw, top + nh)
    return canvas, box, (w, h)

def unletterbox(img_sq, box, orig_size):
    crop = img_sq.crop(box)
    return crop.resize(orig_size, Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=False)
    parser.add_argument("--mask", type=str, required=False)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="FirstSoloProject/outputs/result.png")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--unet_weights", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--mask_mode", type=str, default="black")
    parser.add_argument("--batch_imgs_dir", type=str, default=None)
    parser.add_argument("--batch_masks_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--collage_spacing_h", type=int, default=20)
    parser.add_argument("--collage_spacing_v", type=int, default=20)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    if args.lora:
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora()

    if args.unet_weights:
        import safetensors.torch as st
        sd = st.load_file(args.unet_weights)
        pipe.unet.load_state_dict(sd, strict=False)

    def run_single(image_path, mask_path, out_path):
        img_orig = Image.open(image_path).convert("RGB")
        mask_orig = Image.open(mask_path).convert("L")

        if mask_orig.size != img_orig.size:
            mask_orig = mask_orig.resize(img_orig.size, Image.NEAREST)

        mask_bw = binarize_mask(mask_orig)
        overlay = overlay_mask_on_image(img_orig, mask_bw, white_on_black=(args.mask_mode.lower() == "black"))

        mask_for_pipe = ImageOps.invert(mask_bw) if args.mask_mode.lower() == "black" else mask_bw

        w, h = img_orig.size
        tw, th = round_to_multiple(w, 8), round_to_multiple(h, 8)
        img_pipe = img_orig.resize((tw, th), Image.LANCZOS)
        mask_pipe = mask_for_pipe.resize((tw, th), Image.NEAREST)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows_imgs = []
        base_seed = args.seed if args.seed is not None else 0
        for i in range(max(1, args.rows)):
            g = torch.Generator(device=device)
            s = base_seed + i if base_seed > 0 else torch.seed()
            g.manual_seed(s)
            with torch.autocast(device_type=device if device == "cuda" else "cpu") if device == "cuda" else torch.enable_grad():
                result_i = pipe(
                    prompt=args.prompt,
                    image=img_pipe,
                    mask_image=mask_pipe,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    inpaint_full_res=True,
                    inpaint_full_res_padding=0,
                    generator=g,
                ).images[0]
            result_i = result_i.resize((w, h), Image.LANCZOS)
            row = make_row(img_orig, overlay, result_i, args.collage_spacing_h)
            rows_imgs.append(row)
        grid = stack_rows(rows_imgs, args.collage_spacing_v)
        grid.save(out_path)

    def build_file_map(dir_path):
        m = {}
        if dir_path and os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".jpg", ".jpeg")):
                        stem = os.path.splitext(f)[0]
                        m[stem] = os.path.join(root, f)
        return m

    if args.batch_imgs_dir and args.batch_masks_dir:
        os.makedirs(args.output_dir or "FirstSoloProject/outputs/batch", exist_ok=True)
        imgs = build_file_map(args.batch_imgs_dir)
        masks = build_file_map(args.batch_masks_dir)
        matched = sorted(set(imgs.keys()) & set(masks.keys()))
        for stem in matched:
            out_path = os.path.join(args.output_dir or "FirstSoloProject/outputs/batch", f"{stem}_collage.png")
            run_single(imgs[stem], masks[stem], out_path)
    else:
        run_single(args.image, args.mask, args.output)

if __name__ == "__main__":
    main()
