#!/bin/bash
# SDXL ComfyUI 调试环境设置脚本（基于 RunPod ComfyUI 官方模板）

set -e
COMFYUI_PATH="/workspace/runpod-slim/ComfyUI"

echo "=========================================="
echo "设置 SDXL 图生图调试环境"
echo "ComfyUI 路径: $COMFYUI_PATH"
echo "=========================================="

# 1. 确保模型目录存在
mkdir -p "$COMFYUI_PATH/models/checkpoints"
mkdir -p "$COMFYUI_PATH/models/vae"

# 2. 下载 SDXL 模型
echo "下载 SDXL Base 模型（约 6.5GB）..."
pip install -q huggingface_hub

huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    sd_xl_base_1.0.safetensors \
    --local-dir "$COMFYUI_PATH/models/checkpoints" \
    --local-dir-use-symlinks False

echo "=========================================="
echo "完成！SDXL 模型已下载到:"
echo "  $COMFYUI_PATH/models/checkpoints/sd_xl_base_1.0.safetensors"
echo ""
echo "ComfyUI 应该已经在运行，刷新浏览器即可使用"
echo "=========================================="
