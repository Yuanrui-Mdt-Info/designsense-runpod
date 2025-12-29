#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
from io import BytesIO

import runpod
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from diffusers.utils import load_image
from PIL import Image

print("[Init] Loading models...")

# 加载 ControlNet
controlnet_depth = ControlNetModel.from_pretrained(
    "xinsir/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)

controlnet_tile = ControlNetModel.from_pretrained(
    "xinsir/controlnet-tile-sdxl-1.0",
    torch_dtype=torch.float16,
)

# 加载 SDXL VAE
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# 加载管线
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=[controlnet_depth, controlnet_tile],
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# 设置调度器 (DPM++ 2M Karras)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[Init] xformers enabled")
except Exception:
    pipe.enable_attention_slicing()
    print("[Init] attention slicing enabled")

print("[Init] Model loaded successfully!")


def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event["input"]
        
        # 获取输入参数
        prompt = input_data.get("prompt", "Professional photography of a mid-century modern living room, architectural digest style. Cinematic lighting, photorealistic, 8k, highly detailed.")
        negative_prompt = input_data.get("negative_prompt", "cartoon, illustration, 3d render, painting, drawing, anime, low quality, blurry, watermark, text, signature, people, humans, distorted perspective")
        num_inference_steps = input_data.get("num_inference_steps", 20)
        guidance_scale = input_data.get("guidance_scale", 6.0)
        strength = input_data.get("strength", 0.75)
        seed = input_data.get("seed")
        
        # 加载图片 (支持 URL 或 base64)
        image_url = input_data.get("image_url")
        image_base64 = input_data.get("image_base64")
        
        if image_url:
            init_image = load_image(image_url).convert("RGB")
        elif image_base64:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            return {"error": "Missing 'image_url' or 'image_base64'."}
        
        # 缩放到 1024 (保持宽高比，8 的倍数)
        max_size = 1024
        w, h = init_image.size
        scale = max_size / max(w, h)
        new_w = (int(w * scale) // 8) * 8
        new_h = (int(h * scale) // 8) * 8
        init_image = init_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 设置随机种子
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # 生成
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=[init_image, init_image],
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[0.35, 0.15],
            control_guidance_start=[0.25, 0.6],
            control_guidance_end=[0.8, 1.0],
            generator=generator,
        ).images[0]
        
        # 转为 base64 输出
        buf = BytesIO()
        result.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "output_image_base64": img_str,
            "seed": seed,
            "size": f"{new_w}x{new_h}"
        }
        
    except Exception as e:
        return {"error": str(e)}


# 启动 serverless
runpod.serverless.start({"handler": handler})
