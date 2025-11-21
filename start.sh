#!/bin/bash

echo "Worker Initiated"

echo "Starting WebUI API"

# old command
# python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt stable-diffusion-webui/models/Stable-diffusion/model.ckpt --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check  --no-hashing --no-download-sd-model &

cd /workspace/webui

# 使用环境变量设置模型路径，如果未设置则使用默认值
MODEL_PATH="${MODEL_PATH:-models/Stable-diffusion/model.ckpt}"
python webui.py \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-install \
    --ckpt "$MODEL_PATH" \
    --opt-sdp-attention \
    --disable-safe-unpickle \
    --port 3000 \
    --api \
    --nowebui \
    --skip-version-check \
    --no-hashing \
    --no-download-sd-model &


echo "Starting RunPod Handler"
python -u rp_handler.py