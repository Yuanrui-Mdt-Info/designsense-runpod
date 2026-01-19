FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y python3 python3-pip git libglib2.0-0 libgl1 wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 安装 PyTorch (CUDA 12.1)
RUN pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 handler 代码
COPY rp_handler.py .

CMD ["python", "rp_handler.py"]