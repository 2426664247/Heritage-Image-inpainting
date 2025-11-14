import os
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0

def list_images(dir_path):
    p = Path(dir_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    if p.exists():
        for fp in p.rglob("*"):
            if fp.suffix.lower() in exts:
                files.append(str(fp))
    return files

def center_crop_resize(img, size):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    return img

def load_mask(path, size):
    m = Image.open(path).convert("L")
    m = center_crop_resize(m, size)
    m = m.point(lambda p: 255 if p >= 128 else 0)
    arr = np.array(m).astype(np.float32) / 255.0
    # pipeline expects white=to inpaint; here mask is white=repair directly for training
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)

class InpaintDataset(Dataset):
    def __init__(self, image_dirs, mask_dir, size):
        self.images = []
        for d in image_dirs:
            self.images += list_images(d)
        self.masks = list_images(mask_dir)
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = random.choice(self.masks)
        img = Image.open(img_path).convert("RGB")
        img = center_crop_resize(img, self.size)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = load_mask(mask_path, self.size)
        if random.random() < 0.5:
            mask = torch.flip(mask, dims=[2])
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
        # masked image is original with masked area zeroed
        masked_img_t = img_t * (1.0 - mask)
        return img_t, masked_img_t, mask

def prepare_unet_lora(unet, rank=8):
    attn_procs = {}
    for name, base_proc in unet.attn_processors.items():
        attn_procs[name] = LoRAAttnProcessor2_0()
    unet.set_attn_processor(attn_procs)
    # collect lora params
    train_params = []
    for n, p in unet.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True
            train_params.append(p)
    return train_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", type=str, default="imgs")
    parser.add_argument("--mask_dir", type=str, default="masks")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=str, default="weights")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--cache_dir", type=str, default="models_cache")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # freeze all but LoRA
    for p in unet.parameters():
        p.requires_grad = False
    train_params = prepare_unet_lora(unet, rank=args.rank)
    optimizer = torch.optim.AdamW(train_params, lr=args.lr)

    image_dirs = [d.strip() for d in args.image_dirs.split(",") if d.strip()]
    ds = InpaintDataset(image_dirs, args.mask_dir, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    vae.eval()
    text_encoder.eval()
    unet.train()

    global_step = 0
    pbar = tqdm(total=args.steps, desc="lora-training")
    while global_step < args.steps:
        for img_t, masked_img_t, mask in dl:
            if global_step >= args.steps:
                break
            img_t = img_t.to(device)
            masked_img_t = masked_img_t.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                img_t = 2.0 * img_t - 1.0
                masked_img_t = 2.0 * masked_img_t - 1.0
                latents = vae.encode(img_t).latent_dist.sample() * 0.18215
                masked_latents = vae.encode(masked_img_t).latent_dist.sample() * 0.18215
                mask_latents = F.interpolate(mask, size=(latents.shape[-2], latents.shape[-1]))
                # text conditioning minimal
                tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                encoder_hidden_states = text_encoder(tokens.input_ids.to(device))[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            added = {"mask": mask_latents, "masked_image": masked_latents}
            model_pred = unet(noisy_latents.to(unet.dtype), timesteps, encoder_hidden_states.to(unet.dtype), added_cond_kwargs=added).sample
            loss = F.mse_loss(model_pred, noise)
            loss.backward()

            if (global_step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)
            if global_step % args.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # save LoRA
    os.makedirs(args.out, exist_ok=True)
    unet.save_attn_procs(os.path.join(args.out, "lora_unet.safetensors"))
    pbar.close()

if __name__ == "__main__":
    main()
