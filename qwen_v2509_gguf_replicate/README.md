# Qwen-Image-Edit-2509 GGUF - Replicate 部署

基于 [QuantStack/Qwen-Image-Edit-2509-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF) 量化模型，使用 ComfyUI + ComfyUI-GGUF 在 Replicate 上部署。

## 特点

- **低显存需求**: Q4_K_M 量化版本仅需 ~13GB 显存
- **成本优化**: 可在 A10G (24GB) 上运行，相比原版 A100 80GB 成本降低约 6 倍
- **基于 ComfyUI**: 使用 city96/ComfyUI-GGUF 自定义节点加载量化模型
- **多图像编辑**: 支持最多3张图像同时输入进行编辑（参考 [Qwen Image Edit Multi Editing Workflow](https://huggingface.co/datasets/stablediffusiontutorials/Qwen_Image_Workflows/blob/main/Qwen_Image_Edit_2509_Multi_Editing.json)）

## 模型文件

| 组件 | 文件名 | 大小 | 来源 |
|------|--------|------|------|
| UNet (Main) | Qwen-Image-Edit-2509-Q4_K_M.gguf | ~13.1 GB | QuantStack/Qwen-Image-Edit-2509-GGUF |
| Text Encoder | Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf | ~4.7 GB | unsloth/Qwen2.5-VL-7B-Instruct-GGUF |
| mmproj | Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf | ~1.5 GB | QuantStack/Qwen-Image-Edit-2509-GGUF |
| VAE | qwen-image-vae.safetensors | ~335 MB | QuantStack/Qwen-Image-Edit-2509-GGUF |

## 快速开始

### 1. 安装 Cog

```bash
# macOS
brew install cog

# 或使用 pip
pip install cog
```

### 2. 本地测试

```bash
# 构建镜像（首次运行会下载模型，需要一些时间）
cog build

# 运行预测（单图像）
cog predict -i image=@input.png -i prompt="将背景改为蓝色"

# 运行预测（多图像编辑）
cog predict -i image=@input1.png -i image2=@input2.png -i image3=@input3.png -i prompt="Replace the cat with a dalmatian"
```

### 3. 推送到 Replicate

```bash
# 登录 Replicate
cog login

# 推送模型（替换为你的用户名和模型名）
cog push r8.im/<your-username>/qwen-image-edit-gguf
```

## API 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | File | 必填 | 输入图像（主图像） |
| `image2` | File | 可选 | 第二张图像（用于多图像编辑） |
| `image3` | File | 可选 | 第三张图像（用于多图像编辑） |
| `prompt` | string | "" | 编辑提示词 |
| `negative_prompt` | string | "" | 负面提示词 |
| `steps` | int | 4 | 推理步数 (1-100)，默认4步（使用 Lightning LoRA） |
| `cfg_scale` | float | 1.0 | CFG 引导强度 (1.0-20.0)，默认1.0（配合 Lightning LoRA） |
| `seed` | int | -1 | 随机种子 (-1 为随机) |

### 多图像编辑说明

- 支持最多3张图像同时输入（image, image2, image3）
- 使用 `TextEncodeQwenImageEditPlus` 节点进行多图像条件编码
- 如果只提供单张图像，会自动回退到单图像模式
- 所有图像会自动调整到相同尺寸（基于第一张图像的尺寸）

## 硬件要求

| 配置 | 显存 | 推荐 |
|------|------|------|
| 最低 | 16 GB | RTX 4090 / A10G |
| 推荐 | 24 GB | A10G / L4 |

## 成本估算 (Replicate)

| GPU | 价格/秒 | 单次推理成本 (28步, ~30秒) |
|-----|---------|---------------------------|
| A10G 24GB | ~$0.000225 | ~$0.007 |
| A40 48GB | ~$0.000725 | ~$0.022 |
| A100 80GB | ~$0.0014 | ~$0.042 |

## 项目结构

```
qwen_v2509_gguf_replicate/
├── cog.yaml                    # Cog 配置文件
├── predict.py                  # 推理逻辑
├── download_models.py          # 模型下载脚本
├── workflow_qwen_image_edit.json  # ComfyUI 工作流（参考）
└── README.md                   # 本文档
```

## 本地预下载模型

如果网络不稳定，可以预先下载模型：

```bash
python download_models.py -o ./models
```

## 注意事项

1. **首次构建**: Docker 构建会下载模型 (~20GB)，请确保网络稳定
2. **量化损失**: Q4 量化会有轻微质量损失，如需最佳效果请使用原版模型
3. **ComfyUI 版本**: 基于 ComfyUI 最新版本，如遇兼容问题请检查更新

## 相关链接

- [QuantStack/Qwen-Image-Edit-2509-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (原版)
- [Qwen Image Edit Multi Editing Workflow](https://huggingface.co/datasets/stablediffusiontutorials/Qwen_Image_Workflows/blob/main/Qwen_Image_Edit_2509_Multi_Editing.json)
- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [Comfy-Org/Qwen-Image_ComfyUI](https://github.com/Comfy-Org/Qwen-Image_ComfyUI)
- [Replicate Cog](https://github.com/replicate/cog)

