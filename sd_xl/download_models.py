#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SDXL 模型下载脚本（基于 RunPod ComfyUI 官方模板）"""

from huggingface_hub import hf_hub_download
import os

MODELS = {
    "sdxl_base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "filename": "sd_xl_base_1.0.safetensors",
        "dest_dir": "checkpoints",
        "size": "~6.5 GB",
    },
}


def download_all(comfyui_path: str = "/workspace/runpod-slim/ComfyUI"):
    models_dir = os.path.join(comfyui_path, "models")
    
    for name, info in MODELS.items():
        dest_dir = os.path.join(models_dir, info["dest_dir"])
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, info["filename"])
        
        if os.path.exists(dest_path):
            print(f"[{name}] ✓ 已存在: {info['filename']}")
            continue
            
        print(f"[{name}] 下载中: {info['filename']} ({info['size']})")
        hf_hub_download(
            repo_id=info["repo"],
            filename=info["filename"],
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[{name}] ✓ 完成")


if __name__ == "__main__":
    download_all()
