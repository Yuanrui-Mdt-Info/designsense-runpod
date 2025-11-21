# Stable Diffusion WebUI RunPod Serverless Worker

这是一个用于在 [RunPod](https://www.runpod.io/) Serverless 平台上部署 [Stable Diffusion WebUI (AUTOMATIC1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 的 Docker 镜像项目。

它将 WebUI 封装为一个 Serverless Worker，通过 RunPod API 处理图像生成请求，非常适合构建按需付费的 AI 绘图应用。

## 功能特性

- 🚀 基于 NVIDIA CUDA 12.1 & PyTorch 2.3.1 构建
- 🎨 集成 AUTOMATIC1111 Stable Diffusion WebUI
- 🔌 支持多种 API 操作：
  - `txt2img` (文生图)
  - `img2img` (图生图)
  - `ControlNet` (姿态/边缘控制等)
  - `LoRA`
  - 模型管理与选项配置

## 🛠️ 构建 Docker 镜像

### 1. 构建命令

在项目根目录下运行以下命令构建镜像：

```bash
# 请将 your-username 替换为你的 Docker Hub 用户名
docker build -t your-username/sd-runpod-serverless:v1 .
```

### 2. 推送镜像

将镜像推送到 Docker Hub（或其他容器镜像仓库），以便 RunPod 拉取：

```bash
docker push your-username/sd-runpod-serverless:v1
```

## 🚀 部署到 RunPod

### 1. 创建 Template (模板)

1. 登录 [RunPod Console](https://www.runpod.io/console/serverless)。
2. 导航到 **Templates** -> **New Template**。
3. 填写配置：
   - **Template Name**: 例如 `SD WebUI Serverless`
   - **Container Image**: `your-username/sd-runpod-serverless:v1` (你推送的镜像地址)
   - **Container Disk**: 建议至少 `20 GB` (取决于你需要下载多少模型)
   - **Docker Command**: 留空 (使用 Dockerfile 默认 CMD)
   - **Environment Variables** (可选):
     - `MODEL_PATH`: 指定启动时的 Checkpoint 路径 (默认: `models/Stable-diffusion/model.ckpt`)
4. 点击 **Save Template**。

### 2. 创建 Serverless Endpoint

1. 导航到 **Serverless** -> **New Endpoint**。
2. 选择刚才创建的 Template。
3. 配置 GPU：
   - 选择适合的 GPU 类型 (如 RTX 3090, A4000 等)。
   - 设置 Min/Max Workers。
4. 点击 **Create** 部署。

## 📡 API 调用说明

部署完成后，你会获得一个 Endpoint ID。使用 RunPod SDK 或 HTTP 请求调用。

### 请求参数结构

Worker 接收的 `input` 对象包含 `api_name` 和对应的参数。

#### 示例 1: 文生图 (txt2img)

```json
{
  "input": {
    "api_name": "txt2img",
    "prompt": "masterpiece, best quality, 1girl, looking at viewer, solo, upper body, highres, 8k",
    "negative_prompt": "easynegative, low quality, bad anatomy",
    "steps": 25,
    "width": 512,
    "height": 768,
    "sampler_name": "Euler a",
    "cfg_scale": 7
  }
}
```

#### 示例 2: 获取模型列表 (getModels)

```json
{
  "input": {
    "api_name": "getModels"
  }
}
```

### 支持的 API 列表

在 `rp_handler.py` 中定义了所有支持的接口：

- `txt2img`: POST `/sdapi/v1/txt2img`
- `img2img`: POST `/sdapi/v1/img2img`
- `png-info`: POST `/sdapi/v1/png-info`
- `getModels`: GET `/sdapi/v1/sd-models`
- `getOptions`: GET `/sdapi/v1/options`
- `setOptions`: POST `/sdapi/v1/options`
- `getControlNetModels`: GET `/controlnet/model_list`
- `getControlNetModules`: GET `/controlnet/module_list`
- `getControlNetDetect`: POST `/controlnet/detect`
- `getLora`: GET `/sdapi/v1/loras`

## 💻 本地开发/调试

如果你有 NVIDIA GPU，可以在本地运行测试：

```bash
docker run --gpus all -p 3000:3000 your-username/sd-runpod-serverless:v1
```

容器启动后，它会尝试连接 RunPod 服务器。由于没有真实的 RunPod 环境，你可以手动调用 `rp_handler.py` 中的逻辑或进入容器调试。

