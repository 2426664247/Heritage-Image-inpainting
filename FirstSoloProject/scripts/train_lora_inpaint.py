import os
import math
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from transformers import CLIPTokenizer

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
    arr = 1.0 - arr
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
        mask = load_mask(mask_path, self.size)
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
        masked_img_t = img_t.clone()
        masked_img_t = masked_img_t * (1.0 - mask)
        return img_t, masked_img_t, mask

def add_lora_to_unet(unet, rank):
    for p in unet.parameters():
        p.requires_grad = False
    unet.set_default_attn_processor()
    return []

def select_partial_unet_params(unet):
    for p in unet.parameters():
        p.requires_grad = False
    selected = []
    names = []
    for n, p in unet.named_parameters():
        if any(x in n for x in ["to_q", "to_k", "to_v", "to_out"]):
            p.requires_grad = True
            selected.append(p)
            names.append(n)
    return selected, names

def save_lora(unet, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    unet.save_attn_procs(os.path.join(out_dir, "lora_unet.safetensors"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="FirstSoloProject/train_data")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--out", type=str, default="FirstSoloProject/weights")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-inpainting")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer if hasattr(pipe, "tokenizer") else CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = pipe.text_encoder

    train_params = add_lora_to_unet(unet, args.rank)
    if len(train_params) == 0:
        train_params, tuned_names = select_partial_unet_params(unet)
    optimizer = torch.optim.AdamW(train_params, lr=args.lr)

    image_dirs = [
        os.path.join(args.data_root, "train"),
        os.path.join(args.data_root, "raw", "image"),
        os.path.join(args.data_root, "test"),
    ]
    mask_dir = os.path.join(args.data_root, "mask")
    ds = InpaintDataset(image_dirs, mask_dir, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    global_step = 0
    vae.eval()
    text_encoder.eval()
    unet.train()

    for epoch in range(1000000):
        for img_t, masked_img_t, mask in dl:
            img_t = img_t.to(device, dtype=vae.dtype)
            masked_img_t = masked_img_t.to(device, dtype=vae.dtype)
            mask = mask.to(device, dtype=torch.float32)

            with torch.no_grad():
                img_t = 2.0 * img_t - 1.0
                masked_img_t = 2.0 * masked_img_t - 1.0
                latents = vae.encode(img_t).latent_dist.sample().to(device) * 0.18215
                masked_latents = vae.encode(masked_img_t).latent_dist.sample().to(device) * 0.18215
                mask_latents = F.interpolate(mask, size=(latents.shape[-2], latents.shape[-1]))
                mask_latents = mask_latents.repeat(1, latents.shape[1], 1, 1)

                tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                encoder_hidden_states = text_encoder(tokens.input_ids.to(device))[0]
                encoder_hidden_states = encoder_hidden_states.to(unet.dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            input_latents = torch.cat([noisy_latents, mask_latents[:, :1], masked_latents], dim=1)
            model_pred = unet(input_latents.to(unet.dtype), timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)
            loss.backward()

            if (global_step + 1) % args.accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 100 == 0:
                print(f"step {global_step} loss {loss.item():.4f}")
            if global_step >= args.steps:
                if len(train_params) > 0 and any("lora" in n for n, _ in unet.named_parameters()):
                    save_lora(unet, args.out)
                else:
                    os.makedirs(args.out, exist_ok=True)
                    sd = {n: p.detach().cpu() for n, p in unet.named_parameters() if p.requires_grad}
                    import safetensors.torch as st
                    st.save_file(sd, os.path.join(args.out, "unet_partial_tuned.safetensors"))
                return

if __name__ == "__main__":
    main()
