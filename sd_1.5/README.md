# Stable Diffusion 1.5 LoRA 微调与室内设计重绘 (RunPod)

本项目用于在 RunPod 上对 `stable-diffusion-v1-5` 进行 LoRA 微调，并提供基于 ControlNet 的室内设计重绘功能。

## 功能特性

- **LoRA 微调**: 支持对 SD 1.5 进行 LoRA 微调
- **基础推理**: 支持文本生成图片 (txt2img) 和图片到图片 (img2img)
- **高级重绘**: 使用 ControlNet + 语义分割进行智能室内设计重绘
  - 自动语义分割，识别室内元素（墙、窗、门、家具等）
  - 智能保留关键结构（窗户、门、柱子等）
  - MLSD 线条检测，保持空间结构
  - LCM 加速，4-8 步即可生成高质量图片

## 快速开始

### 0. RunPod Pod 环境初始化（推荐）

增加环境变量:
    HF_HOME: /workspace/huggingface

```bash
# Ubuntu/Debian
apt-get update
apt-get install -y libcairo2-dev libgirepository1.0-dev pkg-config python3-dev

# 然后重新安装
pip uninstall torchaudio
pip install -r requirements.txt
```

该脚本会自动：
1. 安装系统依赖（包括 cairo、GL 库等）
2. 安装 PyTorch 和 CUDA 支持
3. 安装核心 Python 依赖（精简版，避免不必要的包）
4. 配置 HuggingFace 缓存目录（优先使用网络卷）
5. 设置环境变量
6. 验证安装

**注意**: 脚本会自动检测网络卷（`/runpod-volume`），如果存在会使用它作为模型缓存目录，避免重复下载。

### 1. LoRA 微调（可选）

如果你想进行 LoRA 微调：

```bash
chmod +x train.sh
./train.sh
```

该脚本会自动：
1. 安装 Python 依赖。
2. 下载 Hugging Face 官方的 `train_text_to_image_lora.py` 脚本。
3. 下载示例数据集（Pokemon）并开始训练。

### 2. 使用自己的数据集

1. 准备一个文件夹，包含图片和对应的文本描述。
2. 推荐使用 Hugging Face 的 `datasets` 库格式，或者简单的 `metadata.jsonl` 格式：
   ```json
   {"file_name": "image_01.jpg", "text": "a photo of a cat wearing a hat"}
   {"file_name": "image_02.jpg", "text": "a drawing of a robot"}
   ```
3. 修改 `train.sh` 中的 `DATASET_NAME` 指向你的本地文件夹路径：
   ```bash
   export DATASET_NAME="./my_dataset_folder"
   ```

### 3. 推理测试

#### 基础推理（txt2img）
```bash
python inference.py \
  --mode txt2img \
  --prompt "A red dragon pokemon" \
  --lora_path "lora_output" \
  --output "dragon.png"
```

#### 图片到图片（img2img）
```bash
python inference.py \
  --mode img2img \
  --prompt "modern interior design" \
  --image_path "/workspace/init_image.png" \
  --output "/workspace/image_gen_debug/result.png"
```

#### 高级重绘（ControlNet + 语义分割）
```bash
cd /workspace/designsense-runpod/sd_1.5;python inference.py \
  --mode controlnet \
  --prompt "cyberpunk (kitchen:1.2), (night darkness, neon lights, hot pink electric blue purple green neon, bladerunner dystopian, futuristic, LED strips, glowing screens, holographic displays, metallic surfaces, steel glass furniture, angular shapes, smoke fog, matrix code, graffiti walls, polished concrete floor, cyberpunk aesthetic:1.2), (interior design, indoor space, furniture layout, ambient lighting, cozy atmosphere:0.8), best quality, detailed, realistic" \
  --negative_prompt "daylight, sunlight, bright, natural lighting, cozy, warm, soft, modern minimalist, cartoon, disfigured, deformed, ugly, blurry, people, human, text, watermark" \
  --image_path "/workspace/image_gen_debug/input02.jpg" \
  --output "/workspace/image_gen_debug/result.png" \
  --strength 0.78 \
  --guidance_scale 1.5 \
  --num_inference_steps 6
```

```bash
cd /workspace/image_gen_debug;git add result.png;git commit -m"update output";git push;

```

### 4. RunPod Serverless 部署

#### 启动 RunPod Handler
```bash
python rp_handler.py
```

#### API 调用示例
```python
import requests
import base64

# 准备输入图片（base64 编码）
with open("input.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# 调用 API
response = requests.post(
    "http://localhost:8000/runsync",
    json={
        "input": {
            "image_base64": image_base64,
            "prompt": "modern interior design, minimalist, 8k, photorealistic",
            "negative_prompt": "worst quality, low quality",
            "strength": 0.5,
            "guidance_scale": 1.5,
            "num_inference_steps": 6,
            "seed": 42
        }
    }
)

result = response.json()
output_image_base64 = result["output_image_base64"]

# 保存输出图片
with open("output.png", "wb") as f:
    f.write(base64.b64decode(output_image_base64))
```

#### API 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_url` 或 `image_base64` | string | 必需 | 输入图片 URL 或 base64 编码 |
| `prompt` | string | "modern interior design..." | 正面提示词 |
| `negative_prompt` | string | "worst quality..." | 负面提示词 |
| `strength` | float | 0.5 | 重绘强度 (0.0-1.0) |
| `guidance_scale` | float | 1.5 | CFG Scale (LCM 建议 1.0-2.0) |
| `num_inference_steps` | int | 6 | 推理步数 (LCM 建议 4-8) |
| `seed` | int | None | 随机种子（留空则随机） |

## 文件结构

- `setup_pod.sh`: **Pod 环境初始化脚本**（推荐首次使用）
- `train.sh`: LoRA 微调训练脚本
- `inference.py`: 推理脚本，支持 txt2img、img2img 和 controlnet 模式
- `rp_handler.py`: RunPod serverless handler，用于部署到 RunPod
- `requirements.txt`: 完整依赖列表（包含 Jupyter 等，可能包含不必要的依赖）

## 注意事项

### LCM 参数调优
- **CFG Scale**: LCM 对 guidance_scale 非常敏感，建议保持在 1.0-2.0 之间。设置过高会导致图片"烧毁"
- **推理步数**: LCM 通常只需要 4-8 步即可生成高质量图片，设置更多步数不会显著提升质量
- **Prompt**: 如果 Prompt 非常复杂且必须严格遵循，LCM 可能会产生幻觉，建议简化 Prompt

### 模型说明
- **基础模型**: `SG161222/Realistic_Vision_V6.0_B1_noVAE` (高质量写实风格)
- **LCM LoRA**: `latent-consistency/lcm-lora-sdv1-5` (加速推理)
- **ControlNet**: 
  - `BertChristiaens/controlnet-seg-room` (语义分割)
  - `lllyasviel/sd-controlnet-mlsd` (线条检测)
- **语义分割**: `nvidia/segformer-b5-finetuned-ade-640-640`
