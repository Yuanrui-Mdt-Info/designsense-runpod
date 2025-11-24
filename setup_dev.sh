#!/bin/bash

# 1. 安装系统依赖 (RunPod 官方镜像通常已有 Python/Git，这里补充 A1111 需要的库)
apt-get update && apt-get install -y libglib2.0-0 libgl1 libgoogle-perftools-dev

# 2. 准备工作目录
mkdir -p /workspace
cd /workspace

# 3. 克隆 Stable Diffusion WebUI
if [ ! -d "webui" ]; then
    echo "Cloning WebUI..."
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui webui
fi

# 4. 准备子仓库 (模拟 Dockerfile 里的手动 clone，加速启动)
cd webui
mkdir -p repositories
# 这里为了节省时间，可以先尝试用 A1111 自带的启动脚本自动安装，
# 或者手动 clone 你 Dockerfile 里指定的这些（更稳）：
git clone https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai
git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models
git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# 5. 安装 Python 依赖
# 注意：RunPod 官方镜像自带 Torch，如果版本不冲突，可以跳过重装 Torch，直接装 requirements
pip install -r requirements.txt
pip install runpod requests

# 6. 回到项目根目录
cd /workspace/designsense-runpod  # 假设你把项目代码放这了
echo "环境配置完成！现在可以直接运行 ./start.sh 了"
