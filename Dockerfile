FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt install git python3 python3-venv ...

# 克隆最新版 webui（你决定 tag）
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
RUN cd stable-diffusion-webui && git checkout v1.10.1

# 复制你的 rp_handler.py 与 start.sh
COPY rp_handler.py /
COPY start.sh /

# 安装 WebUI 必要的依赖
RUN pip install -r requirements_versions.txt
RUN pip install runpod==1.7.13 requests==2.32.5

CMD ["/start.sh"]
