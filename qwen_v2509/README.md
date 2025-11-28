# Qwen-Image-Edit-2509 RunPod Serverless 部署

在 RunPod Serverless 上部署 Qwen/Qwen-Image-Edit-2509 图像编辑模型。

## 硬件要求

- **GPU**: 推荐 A100 80GB / H100
- **显存**: 至少 40GB（FP16/BF16）
- **网络存储**: 建议 100GB（用于缓存模型）

## 构建 Docker 镜像

```bash
docker build --platform linux/amd64 -t your-registry/qwen-image-edit-2509:v1 .
docker push your-registry/qwen-image-edit-2509:v1
```

## RunPod 配置

1. 创建新的 Serverless Endpoint
2. 使用你的 Docker 镜像
3. 配置 GPU（推荐 A100 80GB）
4. 附加网络卷挂载到 `/workspace/huggingface`
5. 设置超时时间为 600 秒

## API 使用示例

### 请求格式

```json
{
  "input": {
    "source_image": "<base64_encoded_image>",
    "prompt": "将图片中的猫换成狗",
    "max_new_tokens": 4096
  }
}
```

### 响应格式

```json
{
  "output_text": "...",
  "status": "success"
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | Qwen/Qwen-Image-Edit-2509 | 模型路径或 HuggingFace 模型 ID |
| PRELOAD_MODEL | true | 是否在启动时预加载模型 |
| HF_HOME | /runpod-volume/huggingface| HuggingFace 缓存目录(runpod-volume为runpod网络卷的默认位置) |

## 注意事项

1. 首次启动会下载模型（约 30-50GB），建议使用网络卷持久化缓存
2. 模型加载需要较大显存，确保 GPU 配置充足
3. 如使用私有模型，需设置 `HF_TOKEN` 环境变量

