#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地模型下载脚本 - 用于预先下载所有模型文件
可在本地测试或缓存模型时使用
"""

import os
from huggingface_hub import hf_hub_download

# 模型文件配置
MODELS = {
    "unet": {
        "repo": "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "filename": "Qwen-Image-Edit-2509-Q4_K_M.gguf",
        "size": "~13.1 GB",
    },
    "text_encoder": {
        "repo": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "size": "~4.7 GB",
    },
    "mmproj": {
        "repo": "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
        "size": "~1.5 GB",
    },
    "vae": {
        "repo": "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "filename": "qwen-image-vae.safetensors",
        "size": "~335 MB",
    },
}


def download_all(output_dir: str = "./models"):
    """下载所有模型文件到本地目录"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Qwen-Image-Edit-2509 GGUF 模型下载")
    print("=" * 60)

    for name, info in MODELS.items():
        dest_path = os.path.join(output_dir, info["filename"])

        if os.path.exists(dest_path):
            print(f"\n[{name}] ✓ 已存在: {info['filename']}")
            continue

        print(f"\n[{name}] 下载中: {info['filename']} ({info['size']})")
        print(f"         来源: {info['repo']}")

        try:
            local_path = hf_hub_download(
                repo_id=info["repo"],
                filename=info["filename"],
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"         ✓ 完成: {local_path}")
        except Exception as e:
            print(f"         ✗ 失败: {e}")

    print("\n" + "=" * 60)
    print("下载完成！")
    print(f"模型目录: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 Qwen-Image-Edit GGUF 模型")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./models",
        help="模型保存目录 (默认: ./models)",
    )
    args = parser.parse_args()

    download_all(args.output_dir)

