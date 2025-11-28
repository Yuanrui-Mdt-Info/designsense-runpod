FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y python3 python3-pip git libglib2.0-0 libgl1 wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 安装 PyTorch (CUDA 12.1)
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建 HuggingFace 缓存目录（用于挂载网络卷）
RUN mkdir -p /workspace/huggingface

# 复制 handler 代码
COPY rp_handler.py .

# 设置环境变量
ENV MODEL_PATH=Qwen/Qwen-Image-Edit-2509
ENV PRELOAD_MODEL=true

CMD ["python", "rp_handler.py"]

