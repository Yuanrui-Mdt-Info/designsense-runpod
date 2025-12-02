#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import runpod
import torch
from huggingface_hub import snapshot_download
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
from io import BytesIO
import base64
from PIL import Image

# -------------------------------
# 1. 配置模型缓存位置（卷）
# -------------------------------

HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
MODEL_DIR = os.path.join(HF_HOME, "Qwen-Image-Edit-2509")   # 本地解压目录
REPO_ID = "Qwen/Qwen-Image-Edit-2509"

os.makedirs(HF_HOME, exist_ok=True)

print(f"[Init] HF_HOME = {HF_HOME}")
print(f"[Init] MODEL_DIR = {MODEL_DIR}")


# -------------------------------
# 2. 预下载模型到卷（一次性）
# -------------------------------

def preload_model():
    if os.path.exists(MODEL_DIR):
        print("[Init] Model already exists in volume, skip downloading.")
        return MODEL_DIR

    print("[Init] Downloading model (first-time only)...")

    # 下载到卷目录
    snapshot_download(
        repo_id=REPO_ID,
        cache_dir=HF_HOME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print("[Init] Model downloaded to volume.")
    return MODEL_DIR


LOCAL_MODEL_PATH = preload_model()


# -------------------------------
# 3. 加载模型（不会触发下载）
# -------------------------------

print("[Init] Loading model from local volume...")
pipe = QwenImageEditPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
).to("cuda")

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[Init] xformers enabled")
except Exception:
    pipe.enable_attention_slicing()
    print("[Init] attention slicing enabled")


# -------------------------------
# 4. Debug: list /runpod-volume
# -------------------------------

def list_volume():
    root = "/runpod-volume"
    result = {}
    for base, dirs, files in os.walk(root):
        rel = base[len(root):] or "/"
        result[rel] = {"dirs": dirs, "files": files}
    return result


# -------------------------------
# 5. RunPod handler
# -------------------------------

def handler(event):
    # Debug
    if event.get("cmd") == "list-volume":
        return list_volume()

    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        image_base64 = input_data.get("image_base64")

        true_cfg_scale = input_data.get("true_cfg_scale", 6.0)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        negative_prompt = input_data.get("negative_prompt", "blurry, low quality")
        seed = input_data.get("seed", None)

        # Load image
        if image_url:
            input_image = load_image(image_url)
        elif image_base64:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            return {"error": "Missing 'image_url' or 'image_base64'."}

        generator = torch.manual_seed(seed) if seed is not None else None

        output_image = pipe(
            image=input_image,
            prompt=prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=1.0,
            num_images_per_prompt=1,
        ).images[0]

        buf = BytesIO()
        output_image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# 6. Start serverless
# -------------------------------

runpod.serverless.start({"handler": handler})
