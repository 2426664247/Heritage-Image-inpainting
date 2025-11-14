# 导入所需的库
import argparse  # 用于解析命令行参数
import subprocess
import sys
import os  # 用于与操作系统交互，如文件路径操作
from PIL import Image, ImageOps  # Pillow库，用于图像处理
import torch  # PyTorch库，用于深度学习张量操作
from diffusers import StableDiffusionInpaintPipeline  # Hugging Face的diffusers库，用于稳定扩散修复模型

# 函数：中心裁剪并缩放图像
def center_crop_resize(img, size):
    """
    将图像从中心裁剪为最大的正方形，然后缩放到指定尺寸。
    这是一种常见的预处理方式，但可能会丢失图像边缘信息。
    """
    w, h = img.size  # 获取原始宽度和高度
    s = min(w, h)  # 取短边作为正方形的边长
    left = (w - s) // 2  # 计算左边裁剪位置
    top = (h - s) // 2  # 计算顶部裁剪位置
    img = img.crop((left, top, left + s, top + s))  # 执行裁剪
    img = img.resize((size, size), Image.LANCZOS)  # 使用高质量LANCZOS算法缩放
    return img

# 函数：将掩码二值化
def binarize_mask(mask):
    """
    将输入的掩码图像转换为纯黑白的二值图像。
    """
    mask = mask.convert("L")  # 转换为灰度图
    # 将灰度值大于等于128的像素设为255（白），否则为0（黑）
    mask = mask.point(lambda p: 255 if p >= 128 else 0)
    return mask

# 函数：在图像上叠加掩码以供可视化
def overlay_mask_on_image(img, mask_bw, white_on_black=True):
    """
    创建一个可视化副本，在原图上用白色覆盖掩码区域。
    用于生成对比图，展示修复的位置。
    """
    img = img.copy()  # 创建原图副本
    mask_arr = mask_bw.convert("L")  # 掩码转为灰度
    w, h = img.size
    img_arr = img.convert("RGB")
    img_pix = img_arr.load()  # 加载图像像素
    m_pix = mask_arr.load()  # 加载掩码像素
    for y in range(h):
        for x in range(w):
            m = m_pix[x, y]
            if white_on_black:  # 如果是黑底掩码（修复区域为黑）
                if m == 0:  # 在黑色区域上覆盖白色
                    img_pix[x, y] = (255, 255, 255)
            else:  # 如果是白底掩码（修复区域为白）
                if m == 255:  # 在白色区域上覆盖白色
                    img_pix[x, y] = (255, 255, 255)
    return img_arr

# 函数：创建包含三张图像的横向对比图
def make_row(img_left, img_mid, img_right, spacing_h):
    """
    将左、中、右三张图水平拼接成一行，用于对比。
    通常是：原图、带掩码的图、修复结果图。
    """
    w, h = img_left.size
    W = w * 3 + spacing_h * 2  # 计算总宽度
    row = Image.new("RGB", (W, h), color=(255, 255, 255))  # 创建白色底图
    row.paste(img_left, (0, 0))  # 粘贴左图
    row.paste(img_mid, (w + spacing_h, 0))  # 粘贴中图
    row.paste(img_right, (2 * w + 2 * spacing_h, 0))  # 粘贴右图
    return row

# 函数：将多个横向对比图垂直堆叠
def stack_rows(rows, spacing_v):
    """
    将多个由make_row生成的图像行垂直堆叠成一个网格图。
    """
    if not rows:
        return None
    W, h = rows[0].size
    H = h * len(rows) + spacing_v * (len(rows) - 1)  # 计算总高度
    grid = Image.new("RGB", (W, H), color=(255, 255, 255))  # 创建白色底图
    y = 0
    for i, r in enumerate(rows):
        grid.paste(r, (0, y))  # 逐行粘贴
        y += h + (spacing_v if i < len(rows) - 1 else 0)  # 更新下一行的Y坐标
    return grid

# 函数：将数值向上取整到最接近的m的倍数
def round_to_multiple(x, m=8):
    """
    确保图像尺寸是8的倍数，这是许多模型架构的要求。
    """
    return max(m, int(round(x / m)) * m)

# 函数：信箱模式缩放（保持宽高比）
def letterbox(img, size, fill):
    """
    将图像等比缩放，并填充黑边，使其适应一个正方形画布。
    此函数在此脚本中未被使用，但作为一种备选的预处理方式保留。
    """
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

# 函数：从信箱模式图像中恢复原图
def unletterbox(img_sq, box, orig_size):
    """
    从letterbox处理后的图像中裁剪出有效区域并恢复到原始尺寸。
    此函数在此脚本中未被使用。
    """
    crop = img_sq.crop(box)
    return crop.resize(orig_size, Image.LANCZOS)

# 主函数
def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="使用Stable Diffusion进行图像修复")
    # 输入/输出参数
    parser.add_argument("--image", type=str, required=False, help="单张输入图像的路径")
    parser.add_argument("--mask", type=str, required=False, help="单张输入掩码的路径")
    parser.add_argument("--output", type=str, default="outputs/result.png", help="单张图像修复结果的保存路径")
    parser.add_argument("--batch_imgs_dir", type=str, default="imgs", help="批量处理的图像文件夹路径")
    parser.add_argument("--batch_masks_dir", type=str, default="masks", help="批量处理的掩码文件夹路径")
    parser.add_argument("--output_dir", type=str, default="outputs/batch", help="批量处理结果的输出文件夹")
    
    # 模型与权重参数
    parser.add_argument("--model", type=str, default="F:\\TraeSoloProject\\.hfhub\\models--stabilityai--stable-diffusion-2-inpainting\\snapshots\\81a84f49b15956b60b4272a405ad3daef3da4590", help="基础修复模型的Hugging Face路径或本地路径")
    parser.add_argument("--lora", type=str, default=None, help="可选的LoRA权重路径")
    parser.add_argument("--unet_weights", type=str, default="weights/unet_partial_tuned.safetensors", help="可选的UNet权重路径 (safetensors格式)")

    # 修复过程参数
    parser.add_argument("--prompt", type=str, default="", help="指导修复的文本提示")
    parser.add_argument("--steps", type=int, default=40, help="扩散过程的步数")
    parser.add_argument("--guidance", type=float, default=5.0, help="引导系数，控制提示与图像的符合程度")
    parser.add_argument("--size", type=int, default=512, help="模型处理时内部使用的尺寸（此脚本中已弃用，改为动态计算）")
    parser.add_argument("--mask_mode", type=str, default="white", choices=['white', 'black'], help="掩码模式：'white'表示白色区域是修复区，'black'表示黑色区域是修复区")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，用于复现结果。设为0则使用随机种子")

    # 结果拼贴图参数
    parser.add_argument("--rows", type=int, default=4, help="为每张输入图生成多少行不同的修复结果")
    parser.add_argument("--collage_spacing_h", type=int, default=20, help="拼贴图中图像的水平间距")
    parser.add_argument("--collage_spacing_v", type=int, default=20, help="拼贴图中图像的垂直间距")
    
    args = parser.parse_args()

    masks_dir = args.batch_masks_dir or "masks"
    masks_inv_dir = "masks_inverted"
    sync_script = os.path.join(os.path.dirname(__file__), "sync_masks.py")
    if os.path.isfile(sync_script):
        subprocess.run([sys.executable, sync_script, "--masks", masks_dir, "--masks_inverted", masks_inv_dir], check=False)

    # --- 2. 初始化模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有可用的GPU
    # 从预训练模型加载修复管线
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # 在GPU上使用半精度以节省显存
    )
    pipe = pipe.to(device)  # 将模型移动到指定设备

    # --- 3. 加载自定义权重 (如果提供) ---
    if args.lora:
        print(f"正在加载 LoRA 权重: {args.lora}")
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora()  # 融合LoRA权重以加速推理

    if args.unet_weights:
        print(f"正在加载 UNet 权重: {args.unet_weights}")
        import safetensors.torch as st
        sd = st.load_file(args.unet_weights)  # 从safetensors文件加载状态字典
        pipe.unet.load_state_dict(sd, strict=False)  # 加载权重到UNet，strict=False允许部分加载

    # --- 4. 定义单张图像处理函数 ---
    def run_single(image_path, mask_path, out_path):
        """处理单张图像和掩码，并保存结果。"""
        print(f"正在处理: {os.path.basename(image_path)}")
        img_orig = Image.open(image_path).convert("RGB")  # 打开并转换为RGB
        mask_orig = Image.open(mask_path).convert("L")  # 打开并转换为灰度

        # 如果掩码和图像尺寸不匹配，将掩码缩放到图像尺寸
        if mask_orig.size != img_orig.size:
            mask_orig = mask_orig.resize(img_orig.size, Image.NEAREST)

        # 预处理：二值化掩码并创建可视化覆盖图
        mask_bw = binarize_mask(mask_orig)
        overlay = overlay_mask_on_image(img_orig, mask_bw, white_on_black=(args.mask_mode.lower() == "black"))

        # 根据掩码模式准备输入给模型的掩码（模型需要白色区域为修复区）
        if args.mask_mode.lower() == "black":
            mask_for_pipe = ImageOps.invert(mask_bw)  # 如果是黑码模式，反转颜色
        else:
            mask_for_pipe = mask_bw

        # 关键步骤：将图像和掩码尺寸调整为8的倍数，避免拉伸失真
        w, h = img_orig.size
        tw, th = round_to_multiple(w, 8), round_to_multiple(h, 8)
        img_pipe = img_orig.resize((tw, th), Image.LANCZOS)
        mask_pipe = mask_for_pipe.resize((tw, th), Image.NEAREST)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 确保输出目录存在
        rows_imgs = []
        base_seed = args.seed if args.seed is not None else 0
        
        # 循环生成多行结果
        for i in range(max(1, args.rows)):
            g = torch.Generator(device=device)
            # 设置种子：如果提供了固定种子，则递增；否则使用随机种子
            s = base_seed + i if base_seed > 0 else torch.seed()
            g.manual_seed(s)
            
            # 使用自动混合精度进行推理（仅在CUDA上）
            with torch.autocast(device_type=device if device == "cuda" else "cpu") if device == "cuda" else torch.enable_grad():
                # 调用修复管线
                result_i = pipe(
                    prompt=args.prompt,
                    image=img_pipe,
                    mask_image=mask_pipe,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    inpaint_full_res=True,  # 使用全分辨率修复
                    inpaint_full_res_padding=0, # 全分辨率修复的填充
                    generator=g,
                ).images[0]
            
            # 后处理：将结果缩放回原始尺寸并创建对比行
            result_i = result_i.resize((w, h), Image.LANCZOS)
            row = make_row(img_orig, overlay, result_i, args.collage_spacing_h)
            rows_imgs.append(row)
            
        # 将所有行堆叠成最终的网格图并保存
        grid = stack_rows(rows_imgs, args.collage_spacing_v)
        if grid:
            grid.save(out_path)
            print(f"结果已保存到: {out_path}")

    # --- 5. 定义批量处理辅助函数 ---
    def build_file_map(dir_path):
        """扫描目录，构建从文件名（不含扩展名）到完整路径的映射。"""
        m = {}
        if dir_path and os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".jpg", ".jpeg")):
                        stem = os.path.splitext(f)[0]  # 获取文件名主干
                        m[stem] = os.path.join(root, f)
        return m

    # --- 6. 执行主逻辑：单张或批量 ---
    # 如果明确提供了单张图像和掩码路径，则执行单张处理
    if args.image and args.mask:
        run_single(args.image, args.mask, args.output)
    # 否则，执行批量处理
    else:
        print("未提供单张图片，进入批量处理模式...")
        os.makedirs(args.output_dir or "outputs/batch", exist_ok=True)
        # 构建图像和掩码的文件映射
        imgs = build_file_map(args.batch_imgs_dir or "imgs")
        masks = build_file_map(args.batch_masks_dir or "masks")
        # 找出文件名主干相同的图像和掩码对
        matched = sorted(set(imgs.keys()) & set(masks.keys()))
        print(f"找到 {len(matched)} 个匹配的图像/掩码对。")
        
        # 遍历所有匹配的文件对并进行处理
        for stem in matched:
            out_path = os.path.join(args.output_dir or "outputs/batch", f"{stem}_collage.png")
            run_single(imgs[stem], masks[stem], out_path)

# 当脚本作为主程序运行时，调用main函数
if __name__ == "__main__":
    main()