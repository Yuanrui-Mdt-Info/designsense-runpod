#!/bin/bash

# 0. 检查 Python 版本 (提醒用)
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Current Python Version: $PY_VERSION"
if [[ "$PY_VERSION" != "3.10" ]]; then
    echo "WARNING: A1111 推荐使用 Python 3.10。你当前是 $PY_VERSION，可能会遇到依赖包编译失败的问题。"
    echo "建议：重开一个 Pod，使用 'pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime' 镜像，或者手动安装 conda/pyenv。"
    sleep 3
fi

# 获取当前脚本所在的目录
PROJECT_ROOT=$(pwd)

# 1. 安装系统依赖
apt-get update && apt-get install -y libglib2.0-0 libgl1 libgoogle-perftools-dev git wget

# 2. 准备工作目录
mkdir -p /workspace
cd /workspace

# 3. 克隆 Stable Diffusion WebUI 并锁定版本
if [ ! -d "webui" ]; then
    echo "Cloning WebUI..."
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui webui
    
    cd webui
    # === 新增：切换到指定版本 v1.10.1 ===
    echo "Checking out v1.10.1..."
    git checkout v1.10.1
    cd ..
fi

# 4. 准备子仓库
cd webui
mkdir -p repositories

# 注意：不同版本的 WebUI 可能依赖不同版本的子仓库
# v1.10.1 对应的依赖通常在 launch.py 里会自动处理，但为了加速预先 clone 也没问题
if [ ! -d "repositories/stable-diffusion-stability-ai" ]; then
    git clone https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai
    git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models
    git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion
    git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    git clone https://github.com/salesforce/BLIP.git repositories/BLIP
fi

# 5. 安装 Python 依赖
# 升级 pip 以避免 wheel 识别问题
python3 -m pip install --upgrade pip

# 安装 WebUI 核心依赖 (使用 requirements_versions.txt 更稳)
# 强制安装 dctorch 解决 k-diffusion 的依赖问题
# 强制安装 jsonmerge 解决其他潜在依赖问题
pip install dctorch jsonmerge clean-fid resize-right torchdiffeq kornia
pip install -r requirements_versions.txt
pip install runpod requests

# 6. 下载模型文件 (v1.5)
echo "Downloading Stable Diffusion v1.5 Model..."
mkdir -p models/Stable-diffusion
MODEL_FILE="models/Stable-diffusion/model.safetensors"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from HuggingFace..."
    wget -O "$MODEL_FILE" "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
    echo "Model downloaded."
else
    echo "Model already exists, skipping download."
fi

# 7. 复制业务代码
echo "Copying handler and start script..."
cp "$PROJECT_ROOT/rp_handler.py" /workspace/webui/
cp "$PROJECT_ROOT/start.sh" /workspace/webui/
chmod +x /workspace/webui/start.sh

echo "环境配置完成！"
echo "请运行: cd /workspace/webui && ./start.sh"
