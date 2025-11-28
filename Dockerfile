FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3 python3-pip git libglib2.0-0 libgl1 libgoogle-perftools-dev wget && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# 安装最稳 torch
RUN pip install torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 拉最新 A1111 稳定版 (指定 v1.10.1)
WORKDIR /workspace
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui webui && \
    cd webui && \
    git checkout v1.10.1

WORKDIR /workspace/webui
RUN mkdir -p repositories && \
    git clone https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai && \
    git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models && \
    git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion && \
    git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer && \
    git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# 安装依赖
# 增加 dctorch 等必要库
RUN pip install dctorch jsonmerge clean-fid resize-right torchdiffeq kornia
RUN pip install -r requirements_versions.txt
RUN pip install runpod==1.7.13 requests==2.32.5

# 创建模型目录（挂载点，实际模型会从 Network Volume 挂载进来）
RUN mkdir -p models/Stable-diffusion

# 复制业务代码
COPY rp_handler.py .
COPY start.sh .
RUN chmod +x start.sh

# 注意：这里改成相对路径，因为 WORKDIR 已经是 /workspace/webui
CMD ["./start.sh"]
