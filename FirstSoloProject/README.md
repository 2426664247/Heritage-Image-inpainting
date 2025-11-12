# 文物图像修复项目（扩散模型 Inpainting + 可视化对比网格）
版本：v0.1.0

## 环境准备

- 创建环境
  - `conda create -n heritage-inpaint python=3.10 -y`
  - `conda activate heritage-inpaint`
- 安装 PyTorch（CUDA 12.6 对应 cu121）
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- 安装依赖
  - `pip install -U diffusers transformers accelerate safetensors peft opencv-python pillow numpy`

## 数据目录约定

- 单张推理：提供 `--image` 与 `--mask` 两个文件路径
- 批量推理：按文件名匹配，目录为：
  - `FirstSoloProject/imgs/` 放原图（支持递归子目录）
  - `FirstSoloProject/masks/` 或 `FirstSoloProject/masks_inverted/` 放掩码（支持递归子目录）
- 掩码为黑白图：黑色=修补，白色=保留（见参数 `--mask_mode`）
- 若掩码与原图尺寸不同，脚本会用最近邻将掩码对齐到原图尺寸；原图不会被裁剪或填充。

## 掩码语义与尺寸适配

- 掩码语义通过 `--mask_mode` 指定：
  - `--mask_mode black` 表示你的掩码“黑=修补、白=保留”（推荐）
  - `--mask_mode white` 表示你的掩码“白=修补、黑=保留”（例如使用 `masks_inverted` 目录时）
- 尺寸适配：为保证管线稳定，内部会将图与掩码缩放到“宽高为 8 的倍数”的尺寸送入模型，输出再缩回到原图尺寸；最终可视化与保存全部使用原图尺寸，无白边。

## 可视化输出（横排三图 × 竖向多行）

- 每行三图横排：左=原图，中=原图+掩码涂白，右=修复结果，三者严格对齐
- 行间与列间可自定义间距：
  - `--collage_spacing_h` 横向间距像素（默认 20）
  - `--collage_spacing_v` 纵向间距像素（默认 20）
- 生成多行对比：
  - `--rows 4` 生成四行结果（可根据需要调整）
  - `--seed 1234` 设置基础随机种子，逐行递增产生不同结果；为 0 或不设则每行使用随机种子

## 使用示例

- 单张推理（掩码黑=修补）
  - `python FirstSoloProject/scripts/infer_inpaint.py --image F:\TraeSoloProject\FirstSoloProject\imgs\pic.png --mask F:\TraeSoloProject\FirstSoloProject\masks\pic.png --output FirstSoloProject/outputs/pic_grid.png --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

- 单张推理（掩码白=修补）
  - `python FirstSoloProject/scripts/infer_inpaint.py --image F:\TraeSoloProject\FirstSoloProject\imgs\pic.png --mask F:\TraeSoloProject\FirstSoloProject\masks_inverted\pic.png --output FirstSoloProject/outputs/pic_grid.png --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode white --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

- 批量推理（掩码黑=修补，按文件名匹配）
  - `python FirstSoloProject/scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\FirstSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\FirstSoloProject\masks --output_dir FirstSoloProject/outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

## 参数建议

- `--size 768` 效果更细致；显存不足可用 512
- `--steps 30~50`，`--guidance 4~7`
- `--prompt ""` 或简短描述（如“文物壁画纹理修复”）

## 训练（可选）

- 轻量 LoRA 微调（快速贴合文物纹理域）：
  - `python FirstSoloProject/scripts/train_lora_inpaint.py --data_root FirstSoloProject/train_data --size 512 --batch 1 --accum 4 --lr 1e-4 --steps 1000 --rank 16 --out FirstSoloProject/weights --model stabilityai/stable-diffusion-2-inpainting`
- 使用 LoRA 权重推理：
  - 在命令中加入 `--lora FirstSoloProject/weights/lora_unet.safetensors`

## 常见问题

- 首次下载模型较慢，建议保持网络通畅
- 掩码务必是黑白图，值域整洁（脚本会二值化）；最好与原图尺寸一致（不一致时脚本用最近邻对齐）
- 若显存不足，降低 `--size` 或 `--steps`

## 参数详解（完整）

- 核心：
  - `--model`：inpainting 模型名，推荐 `stabilityai/stable-diffusion-2-inpainting`
  - `--size`：目标分辨率（内部做 8 的倍数适配），推荐 768；显存少用 512
  - `--steps`：采样步数，建议 30–50
  - `--guidance`：文本引导强度，建议 4–7
  - `--prompt`：文本提示，默认空
  - `--mask_mode`：掩码语义，`black`（黑=修补）或 `white`（白=修补）
  - `--rows`：同一输入生成的行数，便于对比不同随机种子结果
  - `--seed`：基础随机种子；>0 时每行递增，0/未设为随机
  - `--collage_spacing_h` / `--collage_spacing_v`：横/纵向间距像素

- 单张：
  - `--image` 原图路径，`--mask` 掩码路径，`--output` 输出路径

- 批量：
  - `--batch_imgs_dir` 原图目录，`--batch_masks_dir` 掩码目录，`--output_dir` 输出目录
  - 递归扫描子目录，支持 `.png/.jpg/.jpeg`
  - 以无扩展名的文件名（stem）匹配原图与掩码
  - 输出文件名为 `<stem>_collage.png`

## 目录结构

- `FirstSoloProject/scripts/infer_inpaint.py` 推理与可视化脚本
- `FirstSoloProject/scripts/train_lora_inpaint.py` 轻量微调脚本
- `FirstSoloProject/weights/` 微调权重（如 `lora_unet.safetensors`）
- `FirstSoloProject/outputs/` 推理与可视化输出（网格拼接 `*_collage.png`）
- `FirstSoloProject/imgs/`、`FirstSoloProject/masks/` 原图与掩码目录

## 推荐使用流程

1. 将原图放入 `imgs/`，掩码放入 `masks/`（黑=修补）或 `masks_inverted/`（白=修补），文件名保持一致
2. 先批量生成网格对比：
   - 黑=修补：
     - `python FirstSoloProject/scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\FirstSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\FirstSoloProject\masks --output_dir FirstSoloProject/outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`
   - 白=修补：
     - `python FirstSoloProject/scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\FirstSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\FirstSoloProject\masks_inverted --output_dir FirstSoloProject/outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode white --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`
3. 若需要进一步贴合文物风格，执行轻量 LoRA 微调，再在推理命令中加入 `--lora` 参数对比前后

## 额外示例与注意事项

- 单张推理加入 LoRA：
  - `python FirstSoloProject/scripts/infer_inpaint.py --image ... --mask ... --output ... --steps 40 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --lora FirstSoloProject/weights/lora_unet.safetensors`
- Windows 路径：建议使用绝对路径或在命令中加引号，如 `"F:\\TraeSoloProject\\..."`
- 掩码制作：保持黑白清晰，避免灰阶；覆盖率建议 10%–60%，过大区域可能导致整体风格漂移
- 输出结构：每个输出为横排三图的行，竖向堆叠 `rows` 行，便于一次性对比
