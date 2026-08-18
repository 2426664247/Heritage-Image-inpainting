# 文物图像修复项目（Stable Diffusion Inpainting + 实验管理 UI）

版本：v0.3.0

本项目使用 Stable Diffusion 2 Inpainting 对文物、壁画等图像的缺损区域进行生成式修复，同时提供终端批处理脚本和本地网页 UI。网页端支持手绘或导入掩码、切换基础/部分微调 UNet、按需快速训练实验专属 LoRA、按实验名称保存产物，以及从历史实验目录一键恢复原图、最终掩码和参数。

## 功能概览

| 功能 | 网页 UI | 终端脚本 |
| --- | --- | --- |
| 单张图像修复 | 支持 | 支持 |
| 批量图像修复 | 不支持 | 支持 |
| 手绘多边形掩码 | 支持 | 不支持 |
| 导入已有掩码 | 支持 | 支持 |
| 导入历史实验 | 支持 | 不支持 |
| 基础 UNet / 部分微调 UNet 对比 | 页面开关 | `--unet_weights` |
| 本次实验快速训练 LoRA | 页面开关 | 单独运行训练脚本 |
| 加载已有 LoRA | 页面不提供 | `--lora` |
| 原图、拟修复区域、结果三栏对比 | 支持 | 支持 |

## 运行前准备

### Python 环境

推荐使用 Python 3.10 和 NVIDIA GPU。CPU 也能加载代码路径，但扩散推理和 LoRA 训练会非常慢。

```powershell
conda create -n heritage-inpaint python=3.10 -y
conda activate heritage-inpaint
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U diffusers transformers accelerate safetensors peft opencv-python pillow numpy
pip install -r requirements-ui.txt
```

`requirements-ui.txt` 只补充 FastAPI、Uvicorn 和文件上传依赖；PyTorch、Diffusers 等模型依赖仍需按前面的命令安装。PyTorch 的 CUDA wheel 应按本机驱动和官方兼容要求选择，示例使用 cu121。

### 本地模型、权重与训练数据

| 类型 | 默认位置 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| SD2 Inpainting 基础模型 | `model/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590/` | 是 | 应包含 `model_index.json`、`unet/`、`vae/`、`text_encoder/` 等 diffusers 文件 |
| 部分微调 UNet | `weights/unet_partial_tuned.safetensors` | 可选 | 网页默认尝试加载；文件不存在时网页会记录日志并回退到基础 UNet |
| 已有 LoRA | `weights/lora_unet.safetensors` | 可选 | 仅终端通过 `--lora` 加载；网页没有“加载已有 LoRA”开关 |
| 快速训练图像 | `train_data/train/`、`train_data/raw/image/`、`train_data/test/` | 可选 | 仅开启网页“使用当前数据集快速训练”时需要 |
| 快速训练掩码 | `train_data/mask/` | 可选 | 白色为训练修复区，黑色为保留区 |

`unet_partial_tuned.safetensors` 是部分 UNet 参数权重，不是 LoRA。仓库代码本身不记录该权重具体由哪一批数据训练得到；如需证明其数据来源，应同时保存训练命令、数据清单和权重校验值。

模型、权重、数据集和输出结果体积较大，默认不应提交到 Git：`model/`、`weights/`、`train_data/`、`imgs/`、`masks/`、`masks_inverted/`、`outputs/` 均应保持为本地目录。

## 快速开始

### 启动网页 UI

在项目根目录运行：

```powershell
conda activate heritage-inpaint
python scripts\ui_app.py --host 127.0.0.1 --port 7860
```

然后使用 Chrome 或 Edge 打开 `http://127.0.0.1:7860/`。历史实验文件夹导入依赖浏览器的文件夹选择能力，推荐使用 Chromium 内核浏览器。

### 运行终端批处理

准备好 `imgs/` 和 `masks/` 后，在项目根目录运行：

```powershell
conda activate heritage-inpaint
python scripts\infer_inpaint.py
```

终端默认读取 `imgs/`，从 `masks/` 生成反码到 `masks_inverted/`，再将结果写入 `outputs/batch/`。终端默认直接读取 `weights/unet_partial_tuned.safetensors`；如果不希望加载该权重，需要显式调整 `--unet_weights` 参数或脚本配置。

## 网页 UI 详细说明

### 三种输入入口

| 页面入口 | 用途 | 选择内容 |
| --- | --- | --- |
| 上传图片 | 开始一个新实验 | 单张原图 |
| 导入掩码 | 为当前原图导入已有修复范围 | 单张黑白掩码图 |
| 导入历史实验 | 恢复以前的实验配置 | `outputs/ui/jobs/<实验名称>/` 文件夹 |

“导入掩码”与“导入历史实验”是两个不同入口。前者只读取一张掩码；后者一次读取实验目录中的原图、最终掩码和参数。浏览器原生的文件输入控件已隐藏，页面只显示上述三个明确按钮。

### 新建实验操作流程

1. 填写实验名称。建议包含数据、模型模式和编号，例如 `敦煌壁画-基础UNet-01`。
2. 点击“上传图片”选择一张待修复原图。
3. 选择一种或两种掩码输入方式：
   - 在图像上依次点击病害边界，至少 3 个点后点击“完成区域”；可重复绘制多个区域。
   - 点击“导入掩码”选择已有黑白掩码。
4. 检查画布上的半透明白色区域，确认拟修复范围是否正确。
5. 填写提示词，并设置采样步数、模型输入尺寸、提示词引导强度和随机种子。
6. 决定是否使用部分微调 UNet，以及是否先快速训练本次实验专属 LoRA。
7. 点击“开始修复”。右侧会显示排队、训练、模型加载、扩散修复和完成状态。
8. 完成后查看三栏对比图，并可单独打开 `repair_area.png` 或 `collage.png`。

网页任务按顺序执行，同一时间只运行一个模型任务。刷新页面不会删除已经写入实验目录的产物，但页面内存中的任务状态不会长期持久化。

### 网页掩码语义与合并规则

- 网页统一使用“白色=修复区、黑色=保留区”。
- 导入掩码会先转为灰度图，再以 128 为阈值二值化：灰度值 `>=128` 变白，`<128` 变黑。
- 掩码尺寸与原图不一致时，使用最近邻插值对齐到原图尺寸，避免产生新的灰度边缘。
- 同时导入掩码并绘制多边形时，最终区域取二者并集；任意一边为白色，该像素就会进入修复区。
- “清空掩码”会同时清除手绘区域和已导入掩码。
- 页面预览以半透明白色覆盖修复区；实际模型输入为实验目录中的二值 `mask.png`。

注意：终端默认 `--mask_mode black`，与网页的白色修复语义不同。跨两种运行方式复用掩码时必须检查 `--mask_mode`，避免修复区域反转。

### 网页参数

| 参数 | 页面默认值 | 后端允许范围 | 作用 |
| --- | ---: | ---: | --- |
| 实验名称 | 必填 | 最长 80 字符 | 决定输出目录名称和同名覆盖行为 |
| 提示词 | 空 | 文本 | 引导修复内容与风格 |
| 扩散采样步数 | 30 | 1–100 | 越高通常越慢，不保证单调提升质量 |
| 模型输入尺寸 | 512 | 128–2048 | 控制送入模型的最长边，内部调整为 8 的倍数 |
| 提示词引导强度 | 5.0 | 1.0–20.0 | 越高越强调文本条件，过高可能造成风格漂移 |
| 随机种子 | 1234 | 0–2147483647 | 大于 0 时可复现；0 表示使用随机种子 |
| 使用部分微调 UNet | 开启 | 开/关 | 开启时尝试加载 `weights/unet_partial_tuned.safetensors` |
| 使用当前数据集快速训练 | 关闭 | 开/关 | 推理前训练并加载本次实验专属 LoRA |

### UNet 与快速 LoRA 的四种组合

| 部分微调 UNet | 快速训练 | 实际模型组合 |
| --- | --- | --- |
| 关闭 | 关闭 | 基础 SD2 Inpainting 自带 UNet |
| 开启 | 关闭 | 基础模型 + `unet_partial_tuned.safetensors` |
| 关闭 | 开启 | 基础模型 UNet + 本次快速训练 LoRA |
| 开启 | 开启 | 部分微调 UNet + 本次快速训练 LoRA |

网页快速训练固定使用：训练尺寸 384、batch 1、梯度累积 1、学习率 `1e-5`、20 steps、rank 8。训练图来自 `train_data/train/`、`train_data/raw/image/`、`train_data/test/`，训练时从 `train_data/mask/` 随机抽取掩码，并不会按同名文件严格配对。生成的 LoRA 只保存在本次实验目录，不会覆盖 `weights/lora_unet.safetensors`。

部分 UNet 开关开启但权重文件不存在时，网页会回退到基础模型 UNet，并在日志中提示；基础模型或快速训练数据缺失时，任务会失败。网页不提供已有预训练 LoRA 的选择开关。

### 导入历史实验

点击“导入历史实验”，选择具体实验文件夹：

```text
outputs/ui/jobs/<实验名称>/
```

所选目录至少需要包含：

- `input.png`：恢复原图。
- `mask.png`：恢复上次实际送入模型的最终掩码。
- `request.json`：恢复实验名称、提示词、采样步数、尺寸、引导强度、随机种子和两个模型开关。

历史导入不会恢复旧的 `result.png`、`collage.png`、运行日志或手绘多边形编辑点；它恢复的是已经合并后的最终 `mask.png`。这样复跑时模型收到的修复范围与上次一致。

如果只是重跑并替换旧结果，保留原实验名称即可。如果要做对比实验，导入后应先修改实验名称，例如从 `壁画01-部分UNet` 改为 `壁画01-基础UNet`，否则同名覆盖会删除上一组结果。

### 实验命名、覆盖与并发规则

每次网页运行必须填写实验名称，产物固定保存到：

```text
outputs/ui/jobs/<实验名称>/
```

- 再次提交同名实验时，会先删除该目录中的旧产物，再写入新结果。
- 同名实验仍处于排队或运行状态时，服务会拒绝重复提交。
- Windows 路径不允许的字符 `< > : " / \ | ? *` 会替换为 `_`，首尾空格和句点会移除。
- 实验名称最长 80 个字符；不同名称清理后若得到同一目录名，也按同名处理。
- 对比实验建议固定原图、最终掩码、提示词、步数、尺寸、引导强度和种子，只改变一个模型开关，并使用不同实验名称。

### 网页产物目录

成功实验的目录结构如下：

```text
outputs/ui/jobs/<实验名称>/
├── input.png             # 本次实验原图
├── mask.png              # 手绘区域与导入掩码合并后的最终二值掩码
├── repair_area.png       # 修复区以纯白色覆盖在原图上的范围示意图
├── result.png            # 第一张模型修复结果
├── collage.png           # 原图、拟修复区域、修复结果三栏对比图
├── request.json          # 实验名称、参数、掩码来源、导入文件名和多边形坐标
└── training/             # 仅开启快速训练时生成
    └── lora_unet.safetensors
```

`repair_area.png` 只用于核对范围，不作为模型输入；模型实际读取 `mask.png`。任务若在训练或推理阶段失败，目录中可能只保留输入、掩码、请求记录或未完成的训练文件，应以页面状态和日志判断是否成功，不能只看目录是否存在。

## 终端默认行为与模型路径

- 读取 `imgs/` 中的所有原图。
- 读取 `masks/` 中的掩码，并在启动时自动生成反码到 `masks_inverted/`。
- 实际推理默认读取 `masks_inverted/`，以 `--mask_mode black` 处理，即黑色区域为修复区、白色区域为保留区。
- 批量结果保存到 `outputs/batch/`，文件名为 `<原文件名>_collage.png`。
- 默认参数为 `--steps 30 --guidance 5.0 --size 512 --rows 1 --seed 1234`。
- 默认基础模型位于 `model/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590/`。
- 默认部分 UNet 权重为 `weights/unet_partial_tuned.safetensors`；终端脚本会直接读取该文件，文件缺失会报错。
- 可选已有 LoRA 为 `weights/lora_unet.safetensors`，运行时使用 `--lora weights/lora_unet.safetensors`。

## 数据目录约定

- 单张推理：提供 `--image` 与 `--mask` 两个文件路径
- 批量推理：按文件名匹配，目录为：
  - `imgs/` 放原图（支持递归子目录）
  - `masks/` 或 `masks_inverted/` 放掩码（支持递归子目录）
- 掩码为黑白图：黑色=修补，白色=保留（见参数 `--mask_mode`）。本地默认流程会先从 `masks/` 生成反码到 `masks_inverted/`，再用 `masks_inverted/` 的黑色区域作为修补区。
- 若掩码与原图尺寸不同，脚本会用最近邻将掩码对齐到原图尺寸；原图不会被裁剪或填充。

## 掩码语义与尺寸适配

- 掩码语义通过 `--mask_mode` 指定：
  - `--mask_mode black` 表示你的掩码“黑=修补、白=保留”（推荐）
  - `--mask_mode white` 表示你的掩码“白=修补、黑=保留”
- 本地默认值是 `--mask_mode black`，并默认使用自动生成的 `masks_inverted/`。如果结果看起来修补区域反了，优先检查 `masks/` 原始掩码和 `masks_inverted/` 反码是否符合预期。
- 尺寸适配：为保证管线稳定，内部会将图与掩码按 `--size` 控制的最长边缩放到“宽高为 8 的倍数”的尺寸送入模型，输出再缩回到原图尺寸；最终可视化与保存全部使用原图尺寸，无白边。若 `--size 0`，则按原图尺寸送入模型。

## 终端脚本可视化输出（横排三图 × 竖向多行）

- 每行三图横排：左=原图，中=拟修复区域示意图，右=修复结果，三者严格对齐
- 行间与列间可自定义间距：
  - `--collage_spacing_h` 横向间距像素（默认 20）
  - `--collage_spacing_v` 纵向间距像素（默认 20）
- 生成多行对比：
  - `--rows 4` 生成四行结果（可根据需要调整）
  - `--seed 1234` 设置基础随机种子，逐行递增产生不同结果；为 0 或不设则每行使用随机种子

## 使用示例

- 默认批量（推荐）
  - `python scripts/infer_inpaint.py`
  - 原图放入 `imgs/`，原始掩码放入 `masks/`，文件名保持一致；脚本会自动生成 `masks_inverted/` 并输出到 `outputs/batch/`。

- 单张
  - `python scripts/infer_inpaint.py --image imgs\pic.png --mask masks_inverted\pic.png --output outputs\result.png`

- 加载训练后的 LoRA 权重推理
  - `python scripts/infer_inpaint.py --lora weights\lora_unet.safetensors --output outputs\result_lora.png`

可选参数（按需添加）：
- 文本提示：`--prompt "修补佛像面部，保持原壁画风格与色彩"`（默认空）
- 采样步数：`--steps 30~50`（默认 30）
- 文本引导：`--guidance 4.5~6.5`（默认 5.0）
- 分辨率：`--size 512/768`（默认 512）
- 掩码语义：`--mask_mode white/black`（默认 black，黑=修补）
- 行数与间距：`--rows 1 --collage_spacing_h 20 --collage_spacing_v 20`
- 输出路径（单张）：`--output outputs\result.png`
- 批量输出目录：`--output_dir outputs\batch`
- 加载 UNet 权重：`--unet_weights weights\unet_partial_tuned.safetensors`
- 加载 LoRA 权重：`--lora weights\lora_unet.safetensors`

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

- 推荐使用 `scripts/train_lora_inpaint_official.py` 做轻量 LoRA 微调。
- 建议直接使用本地基础模型目录，不要依赖在线下载；否则可能遇到 Hugging Face 权限、缓存不完整或 401 问题。
- 训练数据目录建议如下：
  - `train_data/train/`、`train_data/raw/image/`、`train_data/test/` 中任意一个或多个目录放训练原图
  - `train_data/mask/` 放训练掩码
  - 支持图片格式：`.png`、`.jpg`、`.jpeg`、`.bmp`
  - 当前训练脚本会为每张训练图随机抽取一张掩码，不要求图像与掩码文件同名；它学习的是当前图像域在多种缺损形状下的修复方式
- 掩码语义：
  - 训练脚本按“白色=修补区域，黑色=保留区域”读取掩码
  - 若你的原始掩码是黑色表示修补区，请先转换成白色表示修补区后再训练
- 推荐先用小参数做冒烟测试：
  - `python scripts/train_lora_inpaint_official.py --image_dirs train_data\train,train_data\raw\image,train_data\test --mask_dir train_data\mask --size 384 --batch 1 --accum 1 --lr 1e-5 --steps 20 --rank 8 --out weights --model "model\models--stabilityai--stable-diffusion-2-inpainting\snapshots\81a84f49b15956b60b4272a405ad3daef3da4590" --log_interval 1`
- 稳定训练推荐命令：
  - `python scripts/train_lora_inpaint_official.py --image_dirs train_data\train,train_data\raw\image,train_data\test --mask_dir train_data\mask --size 384 --batch 1 --accum 4 --lr 1e-5 --steps 500 --rank 8 --out weights --model "model\models--stabilityai--stable-diffusion-2-inpainting\snapshots\81a84f49b15956b60b4272a405ad3daef3da4590"`
- 输出结果：
  - LoRA 权重默认保存在 `weights/lora_unet.safetensors`
- 训练完成后推理：
  - 在推理命令中加入 `--lora weights/lora_unet.safetensors`
- 调参建议：
  - 显存不足时，优先把 `--size` 降到 `256`
  - 若 loss 出现 `nan`，优先把 `--lr` 降到 `5e-6`，并保持 `--size 256/384`
  - 数据量较少时，先用 `--steps 100~300` 观察效果，再逐步增加

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
- 训练时报 `optimizer got an empty parameter list`：
  - 请使用当前 README 中的 `scripts/train_lora_inpaint_official.py` 命令；旧版 LoRA 训练脚本与部分 `diffusers` 版本不兼容
- 训练时报 `num_samples=0`：
  - 检查 `train_data/train/`、`train_data/raw/image/`、`train_data/test/` 中是否至少有一个目录包含训练图像
  - 检查 `train_data/mask/` 中是否有掩码图像
- 训练时报模型下载或 401 错误：
  - 优先将 `--model` 指向本地模型目录，不要直接使用 Hugging Face 仓库名
- “导入历史实验”后没有恢复：
  - 应选择具体的 `outputs/ui/jobs/<实验名称>/` 文件夹，而不是只选择 `outputs/` 或 `outputs/ui/jobs/`
  - 检查目录是否至少包含 `input.png`、`mask.png`、`request.json`
  - 推荐使用 Chrome 或 Edge；其他浏览器可能不支持文件夹选择
- 历史实验结果被覆盖：
  - 同名提交本来就会删除旧目录后重写；做对比实验时，导入历史实验后先修改实验名称再运行
- 页面仍显示旧按钮或旧逻辑：
  - 先强制刷新页面；当前 HTML 已为 CSS 和 JavaScript 添加版本标识以避免继续使用旧缓存

## 目录结构

- `scripts/infer_inpaint.py` 推理与可视化脚本
- `scripts/inpaint_core.py` 网页 UI 使用的模型加载、推理与结果图生成核心
- `scripts/ui_app.py` FastAPI 网页服务与实验任务管理
- `ui/` 网页界面、交互脚本和样式
- `requirements-ui.txt` 网页服务额外依赖
- `README.pdf` 适合离线查看的项目说明文档（内容可能晚于或早于当前 Markdown，以 `README.md` 和代码为准）
- `scripts/train_lora_inpaint_official.py` 推荐使用的 LoRA 训练脚本
- `scripts/train_lora_inpaint.py` 旧版训练脚本（不再推荐）
- `model/` 本地基础模型目录（不提交）
- `weights/` 微调权重目录（不提交）；终端推理脚本和网页 UI 都默认读取 `weights/unet_partial_tuned.safetensors`
- `train_data/` 网页快速训练和独立 LoRA 训练使用的数据目录（不提交）
- `outputs/` 推理与可视化输出；网页实验位于 `outputs/ui/jobs/<实验名称>/`，终端批量结果位于 `outputs/batch/`
- `imgs/`、`masks/`、`masks_inverted/` 原图、原始掩码与自动生成的反码目录
- `train_data.zip` 训练数据压缩包（如需分享数据）

## 推荐使用流程

### 网页单图与对比实验

1. 放置基础模型；如需默认部分微调模式，再放置 `weights/unet_partial_tuned.safetensors`。
2. 启动 `python scripts/ui_app.py --host 127.0.0.1 --port 7860`。
3. 上传原图并手绘或导入掩码，固定提示词、步数、尺寸、引导强度和随机种子。
4. 先运行一个模型组合并保存为明确实验名，例如 `壁画01-基础UNet-seed1234`。
5. 点击“导入历史实验”恢复同一原图、最终掩码和参数，只改变一个模型开关，并修改实验名称后再运行。
6. 对比两个实验目录中的 `repair_area.png`、`result.png`、`collage.png` 和 `request.json`，确认除目标变量外其余条件一致。

### 终端批处理

1. 将基础模型放入默认 `model/.../snapshot/` 目录。
2. 将部分微调权重放入 `weights/unet_partial_tuned.safetensors`，或通过参数选择其他权重。
3. 将原图放入 `imgs/`，原始掩码放入 `masks/`，文件名保持一致。
4. 运行 `python scripts/infer_inpaint.py`，生成反码并批量输出到 `outputs/batch/`。
5. 若需要独立训练 LoRA，准备 `train_data/` 并运行 `scripts/train_lora_inpaint_official.py`。
6. 训练完成后，在终端推理命令中加入 `--lora weights/lora_unet.safetensors` 对比前后结果。

## 尺寸适配与重采样（与代码一致）

- 当前脚本会将输入图与掩码按 `--size` 控制的最长边缩放到“宽高为 8 的倍数”的尺寸送入 inpainting 管线，然后将输出回缩到原图尺寸（高质量插值）。
- 这能保证生成过程稳定，同时最终保存的三图拼接以原图尺寸为基准；但在高频纹理上可能产生轻微插值失真。若要减少失真，建议适度降低 `--size` 或 `--steps` 并观察对比。

## 当前脚本默认参数（与代码一致）

- 模型：`--model model/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590`
- 分辨率：`--size 512`
- 采样步数：`--steps 30`
- 文本引导：`--guidance 5.0`
- 输出路径（单张）：`--output outputs/result.png`
- 文本提示：`--prompt ""`
- LoRA 权重：`--lora None`
- UNet 权重：`--unet_weights weights/unet_partial_tuned.safetensors`
- 掩码语义：`--mask_mode black`（黑=修补）
- 单张目录：`--image`、`--mask` 默认未设置；未设置时进入批量模式
- 批量目录：`--batch_imgs_dir imgs`、`--batch_masks_dir masks_inverted`
- 批量输出目录：`--output_dir outputs/batch`
- 行数与间距：`--rows 1`，`--collage_spacing_h 20`，`--collage_spacing_v 20`
- 随机种子：`--seed 1234`（为 0 时每行随机）

## 模型原理与实现细节

- 核心思想：扩散模型的“条件去噪”
  - 使用 `StableDiffusionInpaintPipeline`（SD2-inpainting）。输入包含：原图 `image` 与掩码 `mask_image`。
  - 掩码白=修补区域（pipeline 内部约定）。未被掩盖的区域保持不变；被掩盖的区域通过条件扩散生成匹配上下文的内容。

- 条件构成：文本提示 + 图像/掩码条件
  - 文本提示：通过 CLIP 文本编码器提供风格与内容引导，强度由 `guidance_scale` 控制（典型 4–7）。
  - 图像与掩码：掩码白色像素处会被修补，模型在扩散过程中仅对该区域进行生成，确保背景一致。

- 解析与尺寸处理：
  - 为匹配 UNet 的卷积栅格，脚本将原图与掩码缩放到“宽、高均为 8 的倍数”输送给模型，生成后再缩回原图尺寸，避免白边与形变。
  - 可视化拼接：横排三图（原图、修复区以纯白色覆盖的拟修复区域、修复结果），竖向堆叠多行（不同随机种子），便于快速对比。网页 UI 还会单独保存 `repair_area.png`。

- 训练逻辑（轻量域适配）：
  - 目标：让 SD2-inpainting 更贴近文物壁画的色彩与纹理分布，提升修补区域的材质一致性与细节。
  - 数据：训练图目录由 `--image_dirs` 指定，掩码目录由 `--mask_dir` 指定；当前实现为每张图随机抽取一张掩码，并分别进行随机水平翻转。
  - 过程：
    1. 原图与掩码缩放到统一训练尺寸（推荐 512/384/256 视显存而定）
    2. 通过 VAE 将图像与“掩盖后的图像”编码为潜空间 `latents` 与 `masked_latents`
    3. 采样时间步 `t` 并加噪得到 `noisy_latents`
    4. UNet 条件去噪：输入为 `[noisy_latents, mask, masked_latents]` 的拼接，输出预测噪声；用 MSE(loss) 与真实噪声对齐
    5. `train_lora_inpaint_official.py` 冻结基础 UNet，只训练注意力模块上的 LoRA 低秩适配参数
  - 保存：独立训练默认保存为 `weights\lora_unet.safetensors`；网页快速训练保存到对应实验目录的 `training\lora_unet.safetensors`。
  - 限制：当前推荐训练脚本不生成 `unet_partial_tuned.safetensors`；该部分权重需要由其他部分微调流程产生，并单独记录训练来源。

- 为何不建议在 8GB 显存下进行 UNet 全量训练：
  - SD2-inpainting 的 UNet 权重体积很大；全量训练时，优化器状态（动量与二阶矩）会显著增加显存，易 OOM。
  - 解决路径：使用“部分层训练”或 LoRA；降低训练分辨率、增大梯度累积（`accum`）、降低学习率，稳定训练。

- 提示词的作用与调参：
  - 提示词可在修补区域引导生成方向；`guidance_scale` 越高，越“听提示词”。过高可能带来过修或风格漂移。
  - 经验：`guidance=4.5–6.5`；先无提示词对比，再加提示词（例如“修补佛像面部，保持原壁画风格与色彩，纹理自然，五官清晰，边缘平滑过渡”）。

- 缓存与磁盘策略：
  - 使用 `--cache_dir` 指向 F 盘目录，或设置 `HF_HOME/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE` 到 F 盘，避免占用 C 盘空间。

## 版本说明

- v0.3.0
  - 新增黑白掩码直接导入，白色为修复区，并支持与手绘多边形区域取并集。
  - 新增历史实验文件夹导入，一次恢复原图、最终掩码、实验名称和已保存参数。
  - 明确隐藏浏览器原生文件输入，只保留“上传图片”“导入掩码”“导入历史实验”三个入口。
  - 新增部分微调 UNet 开关，默认开启；网页不再提供已有预训练 LoRA 的加载选项。
  - 新增实验名称、同名覆盖、独立 `repair_area.png`、进度显示和结果快捷入口。
  - 补充模型组合、掩码语义、历史恢复、对比实验、训练数据来源、输出目录和排错说明。
- v0.2.0
  - 新增本地网页 UI，可上传图片并通过多边形交互绘制修复区域。
  - 支持页面参数设置、运行状态、日志和结果预览。
- v0.1.2
  - 新增“模型原理与实现细节”章节，说明 inpainting 条件扩散、尺寸与掩码处理、训练目标与流程、调参策略等。
  - 保持脚本与用法不变；推理支持提示词与批量生成网格对比。
