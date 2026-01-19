# SD 1.5 Interior Design RunPod Serverless 部署

在 RunPod Serverless 上部署 Stable Diffusion 1.5 + ControlNet 室内设计重绘模型。

## 硬件要求

- **GPU**: 推荐 RTX 4090 / A100 40GB
- **显存**: 至少 16GB（FP16）
- **网络存储**: 建议 50GB（用于缓存模型）

## 构建 Docker 镜像

cd sd_1.5
docker build --platform linux/amd64 -t your-registry/sd15-interior-design:v1 .
docker push your-registry/sd15-interior-design:v1## RunPod 配置

1. 创建新的 Serverless Endpoint
2. 使用你的 Docker 镜像
3. 配置 GPU（推荐 RTX 4090 或 A100 40GB）
4. 附加网络卷挂载到 `/runpod-volume`
5. 设置环境变量 `HF_HOME=/runpod-volume/huggingface`
6. 设置超时时间为 300 秒

## API 使用示例

### 请求格式

{
  "input": {
    "image_base64": "<base64_encoded_image>",
    "prompt": "modern interior design, minimalist, 8k, photorealistic",
    "negative_prompt": "worst quality, low quality",
    "strength": 0.5,
    "guidance_scale": 1.5,
    "num_inference_steps": 6,
    "seed": 42
  }
}### 响应格式

{
  "output_image_base64": "...",
  "prompt": "modern interior design...",
  "seed": 42,
  "size": "768x512"
}## 支持的风格 LoRA

模型会根据 prompt 自动加载对应的风格 LoRA：

- `cyberpunk interior` - 赛博朋克室内设计
- `floor plan interior` - 平面图室内设计
- `clothing store` - 服装店室内设计
- `tropical exterior` - 热带外观设计
- `tropical interior` - 热带室内设计

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HF_HOME | /runpod-volume/huggingface | HuggingFace 缓存目录 |

## 注意事项

1. 首次启动会下载模型（约 15-20GB），建议使用网络卷持久化缓存
2. 使用 LCM LoRA 加速，推理步数建议 4-8 步
3. guidance_scale 建议保持在 1.0-2.0 之间（LCM 特性）
4. 输入图片会自动调整为最大 768px，并保持宽高比