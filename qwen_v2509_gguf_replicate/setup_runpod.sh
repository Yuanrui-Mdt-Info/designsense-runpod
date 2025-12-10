#!/bin/bash
# RunPod 调试环境设置脚本
# 用于在 RunPod 上快速设置 Qwen-Image-Edit-2509 GGUF 开发环境

# mkdir -p ~/.ssh
# chmod 700 ~/.ssh
# echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDJOPHN1WpWKnMAC447G5wjvD+54HHIYlJ/HbahYr0sv james@sictecinfotech.com" >> ~/.ssh/authorized_keys
# chmod 600 ~/.ssh/authorized_keys


set -e  # 遇到错误立即退出

# 设置工作路径
WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "=========================================="
echo "设置 RunPod 调试环境"
echo "工作路径: $WORKSPACE"
echo "=========================================="

# 1. 安装系统依赖（如果需要）
echo ""
echo "[1/5] 检查系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    || echo "系统依赖安装完成（部分可能已存在）"

# 2. 安装 Python 依赖
echo ""
echo "[2/5] 安装 Python 依赖包..."
pip install --upgrade pip
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    "numpy<2" \
    "pillow>=10.0.0" \
    aiohttp \
    einops \
    "transformers>=4.44.0" \
    "tokenizers>=0.19.0" \
    sentencepiece \
    safetensors \
    scipy \
    tqdm \
    psutil \
    "kornia>=0.7.0" \
    spandrel \
    "huggingface-hub>=0.25.0" \
    gguf

# 3. 克隆 ComfyUI
echo ""
echo "[3/5] 克隆 ComfyUI..."
if [ ! -d "$WORKSPACE/ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$WORKSPACE/ComfyUI"
else
    echo "ComfyUI 已存在，跳过克隆"
fi

# 安装 ComfyUI 依赖
echo "安装 ComfyUI requirements.txt..."
pip install -r "$WORKSPACE/ComfyUI/requirements.txt"

# 4. 安装 ComfyUI-GGUF 自定义节点
echo ""
echo "[4/5] 安装 ComfyUI-GGUF 自定义节点..."
GGUF_NODE_DIR="$WORKSPACE/ComfyUI/custom_nodes/ComfyUI-GGUF"
GGUF_SYMLINK="$WORKSPACE/ComfyUI/custom_nodes/ComfyUI_GGUF"

if [ ! -d "$GGUF_NODE_DIR" ]; then
    git clone https://github.com/city96/ComfyUI-GGUF.git "$GGUF_NODE_DIR"
else
    echo "ComfyUI-GGUF 已存在，跳过克隆"
fi

# 创建软连接（ComfyUI-GGUF -> ComfyUI_GGUF）
if [ ! -L "$GGUF_SYMLINK" ] && [ ! -d "$GGUF_SYMLINK" ]; then
    ln -s "$GGUF_NODE_DIR" "$GGUF_SYMLINK"
    echo "已创建软连接: ComfyUI-GGUF -> ComfyUI_GGUF"
else
    echo "软连接已存在或目标已存在"
fi

# 5. 创建必要的目录结构
echo ""
echo "[5/5] 创建模型和输出目录..."
mkdir -p "$WORKSPACE/ComfyUI/models/unet"
mkdir -p "$WORKSPACE/ComfyUI/models/clip"
mkdir -p "$WORKSPACE/ComfyUI/models/text_encoders"
mkdir -p "$WORKSPACE/ComfyUI/models/vae"
mkdir -p "$WORKSPACE/ComfyUI/input"
mkdir -p "$WORKSPACE/ComfyUI/output"

echo ""
echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "ComfyUI 路径: $WORKSPACE/ComfyUI"
echo "ComfyUI-GGUF 节点: $GGUF_NODE_DIR"
echo "软连接: $GGUF_SYMLINK"
echo ""
echo "使用前请设置环境变量:"
echo "  export COMFYUI_PATH=$WORKSPACE/ComfyUI"
echo ""
