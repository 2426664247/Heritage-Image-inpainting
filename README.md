# 文物图像修复项目（扩散模型 Inpainting + 可视化对比网格）
版本：v0.1.1

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
  - `imgs/` 放原图（支持递归子目录）
  - `masks/` 或 `masks_inverted/` 放掩码（支持递归子目录）
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
  - `python scripts/infer_inpaint.py --image F:\TraeSoloProject\imgs\pic.png --mask F:\TraeSoloProject\masks\pic.png --output outputs/pic_grid.png --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

- 单张推理（掩码白=修补）
  - `python scripts/infer_inpaint.py --image F:\TraeSoloProject\imgs\pic.png --mask F:\TraeSoloProject\masks_inverted\pic.png --output outputs/pic_grid.png --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode white --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

- 批量推理（掩码黑=修补，按文件名匹配）
  - `python scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\masks --output_dir outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`

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

## 训练（可选）

- 轻量 LoRA 微调（快速贴合文物纹理域）：
  - `python scripts/train_lora_inpaint.py --data_root train_data --size 512 --batch 1 --accum 4 --lr 1e-4 --steps 1000 --rank 16 --out weights --model stabilityai/stable-diffusion-2-inpainting`
- 使用 LoRA 权重推理：
  - 在命令中加入 `--lora weights/lora_unet.safetensors`

## 常见问题与排错

- 输出大小不一致/出现白边：
  - 代码已在送入管线前将尺寸调整到 8 的倍数，并在输出时回缩到原图尺寸；若仍异常，检查掩码是否为极端小图或非二值。
- 未遵循掩码修补：
  - 确保 `--mask_mode` 与你的掩码语义一致；黑=修补用 `black`，白=修补用 `white`
  - 掩码务必为黑白图，灰度值会被二值化（≥128→白，<128→黑）
- 批量只处理部分文件：
  - 只处理“文件名一致”的成对数据；检查 `imgs/` 与 `masks/` 的文件名 stem 是否一致
  - 目录会递归扫描子目录，但仅支持 `.png/.jpg/.jpeg`
- 显存不足/运行缓慢：
  - 将 `--size` 降低为 512 或降低 `--steps`；减少 `--rows`

## 目录结构

- `scripts/infer_inpaint.py` 推理与可视化脚本
- `scripts/train_lora_inpaint.py` 轻量微调脚本
- `weights/` 微调权重（如 `lora_unet.safetensors`）
- `outputs/` 推理与可视化输出（网格拼接 `*_collage.png`）
- `imgs/`、`masks/`（或 `masks_inverted/`）原图与掩码目录
- `train_data.zip` 训练数据压缩包（如需分享数据）

## 推荐使用流程

1. 将原图放入 `imgs/`，掩码放入 `masks/`（黑=修补）或 `masks_inverted/`（白=修补），文件名保持一致
2. 先批量生成网格对比：
   - 黑=修补：
     - `python scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\masks --output_dir outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode black --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`
   - 白=修补：
     - `python scripts/infer_inpaint.py --batch_imgs_dir F:\TraeSoloProject\imgs --batch_masks_dir F:\TraeSoloProject\masks_inverted --output_dir outputs/batch_grid --steps 30 --guidance 5.0 --model stabilityai/stable-diffusion-2-inpainting --size 768 --mask_mode white --rows 4 --collage_spacing_h 20 --collage_spacing_v 20 --seed 1234`
3. 若需要进一步贴合文物风格，执行轻量 LoRA 微调，再在推理命令中加入 `--lora` 参数对比前后

