FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip /tmp/*

# 复制 handler 代码
COPY rp_handler.py .

CMD ["python", "rp_handler.py"]
