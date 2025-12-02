#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import runpod
import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
from io import BytesIO
import base64
from PIL import Image

# HF_HOME 环境变量控制缓存位置
print(f"[Init] HF_HOME = {os.environ.get('HF_HOME', '~/.cache/huggingface')}")


print("[Init] Loading model...")
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16,
).to("cuda")

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[Init] xformers enabled")
except Exception:
    pipe.enable_attention_slicing()
    print("[Init] attention slicing enabled")


def list_volume():
    root = "/runpod-volume"
    result = {}
    for base, dirs, files in os.walk(root):
        rel = base[len(root):] or "/"
        result[rel] = {"dirs": dirs, "files": files}
    return result


# RunPod handler

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
