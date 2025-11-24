# Stable Diffusion WebUI RunPod Serverless Worker

这是一个用于在 [RunPod](https://www.runpod.io/) Serverless 平台上部署 [Stable Diffusion WebUI (AUTOMATIC1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 的 Docker 镜像项目。

它将 WebUI 封装为一个 Serverless Worker，通过 RunPod API 处理图像生成请求，非常适合构建按需付费的 AI 绘图应用。

## 功能特性

- 🚀 基于 NVIDIA CUDA 12.1 & PyTorch 2.1.0 构建
- 🎨 集成 AUTOMATIC1111 Stable Diffusion WebUI (v1.10.1)
- 🐍 推荐使用 Python 3.10.x 以获得最佳兼容性
- 🔌 支持多种 API 操作：
  - `txt2img` (文生图)
  - `img2img` (图生图)
  - `ControlNet` (姿态/边缘控制等)
  - `LoRA`
  - 模型管理与选项配置

## 🧪 在 RunPod Pod 上快速调试（推荐）

如果你想快速调试而不构建 Docker 镜像，可以直接在 RunPod Pod 上运行：

### 1. 启动 Pod
1. 在 RunPod Console 创建一个 GPU Pod
2. 选择镜像：`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` (Python 3.10)
3. 选择 GPU：RTX 3090 或 RTX 4090（便宜且够用）
4. 启动 Pod，通过 SSH 或 Jupyter Lab 连接

### 2. 运行环境配置脚本
将项目代码上传到 Pod，然后运行：

```bash
cd /workspace/你的项目目录
chmod +x setup_dev.sh
./setup_dev.sh
```

`setup_dev.sh` 会自动完成：
- 安装系统依赖和 Python 包
- 克隆 Stable Diffusion WebUI v1.10.1
- 下载 SD v1.5 模型（约 4GB）
- 配置运行环境

### 3. 启动服务
```bash
cd /workspace/webui
./start.sh
```

看到 `Model loaded in ...s` 和 `WebUI API Service is ready` 即表示成功！

### 4. 测试 API
在 Pod 终端创建测试脚本：

```bash
cat > test_txt2img.py << 'EOF'
import requests

url = "http://127.0.0.1:3000/sdapi/v1/txt2img"
payload = {
    "prompt": "a cute cat, high quality, 8k",
    "steps": 20,
    "width": 512,
    "height": 512
}

response = requests.post(url, json=payload, timeout=120)
if response.status_code == 200:
    print("Success! Image generated.")
else:
    print(f"Error: {response.status_code}")
EOF

python test_txt2img.py
```

## 🛠️ 构建 Docker 镜像（生产部署）

当你在 Pod 上调试完成后，可以构建镜像用于 Serverless 部署。

### 1. 构建命令

在项目根目录下运行以下命令构建镜像：

```bash
# 请将 your-username 替换为你的 Docker Hub 用户名
docker build -t your-username/sd-runpod-serverless:v1 .
```

**注意**：镜像中**不包含模型文件**，模型需要通过 Network Volume 挂载。

### 2. 推送镜像

将镜像推送到 Docker Hub（或其他容器镜像仓库），以便 RunPod 拉取：

```bash
docker push your-username/sd-runpod-serverless:v1
```

## 🚀 部署到 RunPod Serverless

### 0. 准备模型文件（Network Volume）

1. 在 RunPod Console 导航到 **Storage** -> **Network Volumes**
2. 创建一个新的 Network Volume（建议 20GB+）
3. 通过 Pod 挂载这个 Volume，上传模型文件到 `models/Stable-diffusion/` 目录

**推荐模型**：
- **SD v1.5**（快速、兼容性好）：
  ```bash
  wget -O model.safetensors https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
  ```
- **SDXL Turbo**（高质量、1024×1024）：
  ```bash
  wget -O sd_xl_turbo.safetensors https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors
  ```

### 1. 创建 Template (模板)

1. 登录 [RunPod Console](https://www.runpod.io/console/serverless)
2. 导航到 **Templates** -> **New Template**
3. 填写配置：
   - **Template Name**: 例如 `SD WebUI Serverless v1.10.1`
   - **Container Image**: `your-username/sd-runpod-serverless:v1` (你推送的镜像地址)
   - **Container Disk**: 建议 `10 GB` (镜像本身不大)
   - **Docker Command**: 留空 (使用 Dockerfile 默认 CMD)
   - **Volume Mount Path**: `/workspace/webui/models/Stable-diffusion`
   - **Volume Path**: 选择你上传了模型的 Network Volume
4. 点击 **Save Template**

### 2. 创建 Serverless Endpoint

1. 导航到 **Serverless** -> **New Endpoint**
2. 选择刚才创建的 Template
3. 配置 GPU：
   - 选择适合的 GPU 类型 (如 RTX 3090, A4000 等)
   - 设置 Min/Max Workers（建议 Min: 0, Max: 3）
4. 点击 **Create** 部署

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

## 🐛 常见问题

### Q: 为什么推荐 Python 3.10 而不是 3.11/3.12？
A: Stable Diffusion 生态（PyTorch, xformers 等）对 Python 3.10 的支持最好，预编译包最全，可以避免编译失败的问题。

### Q: 为什么会提示 `no module 'xformers'`？
A: xformers 是可选的加速库，没有它也能运行，只是速度会慢 20-30%。如果需要安装：`pip install xformers`。

### Q: SD v1.5 和 SDXL Turbo 该选哪个？
A: 
- **SD v1.5**: 速度快（512×512），显存占用小（4-6GB），插件生态丰富，**推荐调试和快速出图**
- **SDXL Turbo**: 质量高（1024×1024），显存占用大（8-12GB），**推荐生产环境追求质量**

### Q: 如何在 Pod 上持久化数据？
A: RunPod Pod 的 `/workspace` 目录默认是持久化的，即使停止 Pod 再启动，数据依然保留。

## 💻 本地开发/调试

如果你有 NVIDIA GPU，可以在本地运行测试：

```bash
docker run --gpus all -p 3000:3000 \
  -v /path/to/your/models:/workspace/webui/models/Stable-diffusion \
  your-username/sd-runpod-serverless:v1
```

容器启动后，访问 `http://localhost:3000/docs` 查看 API 文档。

## 📝 技术栈

- **Base Image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10.x (推荐)
- **PyTorch**: 2.1.0 (CUDA 11.8)
- **WebUI Version**: AUTOMATIC1111 v1.10.1
- **RunPod SDK**: 1.7.13

## 📄 License

MIT License
