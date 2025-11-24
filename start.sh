#!/bin/bash

echo "Worker Initiated"

echo "Starting WebUI API"

# old command
# python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt stable-diffusion-webui/models/Stable-diffusion/model.ckpt --lowram --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check  --no-hashing --no-download-sd-model &

cd /workspace/webui

# 使用环境变量设置模型路径，如果未设置则使用默认值
# MODEL_PATH="${MODEL_PATH:-models/Stable-diffusion/model.ckpt}"

# 启动 WebUI
# 1. 去掉 --ckpt 参数，让 WebUI 自动在 models/Stable-diffusion 下寻找模型
# 2. 将输出重定向到 webui.log 以便调试
# 3. 后台运行 (&)
python -u webui.py \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-install \
    --opt-sdp-attention \
    --disable-safe-unpickle \
    --port 3000 \
    --api \
    --nowebui \
    --skip-version-check \
    --no-hashing \
    --no-download-sd-model > /workspace/webui.log 2>&1 &

# 获取 WebUI 进程 ID
WEBUI_PID=$!
echo "WebUI PID: $WEBUI_PID"

# 启动 tail -f 在后台实时输出日志，这样我们既能让 WebUI 后台跑，又能看到它发生了什么
# 当 start.sh 被 kill 时，这个 tail 也会停止（通常）
tail -f /workspace/webui.log &

echo "Starting RunPod Handler"
# 启动 Handler (这个必须在前台运行，因为它要监听任务)
python -u rp_handler.py
