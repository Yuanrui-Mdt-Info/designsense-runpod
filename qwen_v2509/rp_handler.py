#! /usr/bin/env python
# -*- coding: utf-8 -*-

import runpod
# from diffusers import DiffusionPipeline
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image


# Load model on startup - 使用 bfloat16 更稳定
# pipe = DiffusionPipeline.from_pretrained(
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16,  # bfloat16 比 float16 更稳定
    trust_remote_code=True,
).to("cuda")


def handler(event):
    """
    Runpod handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        image_base64 = input_data.get("image_base64")
        
        true_cfg_scale = input_data.get("true_cfg_scale", 6.0)
        num_inference_steps = input_data.get("num_inference_steps", 50)
        negative_prompt = input_data.get("negative_prompt", "blurry, low quality, distorted")
        seed = input_data.get("seed", None)

        # 支持 URL 或 Base64 输入
        if image_url:
            input_image = load_image(image_url)
        elif image_base64:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            return {"error": "Missing 'image_url' or 'image_base64' parameter."}
        
        # 设置随机种子
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

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}


# Required by Runpod
runpod.serverless.start({"handler": handler})
