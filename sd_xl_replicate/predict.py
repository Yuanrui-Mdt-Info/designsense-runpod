#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from PIL import Image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """加载 SDXL + 双 ControlNet (Depth + Tile)"""
        
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
        
        # 使用 Img2Img 管线 + MultiControlNet
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[controlnet_depth, controlnet_tile],  # 双 ControlNet
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        
        # 设置调度器 (DPM++ 2M Karras)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        image: Path = Input(description="输入图片"),
        prompt: str = Input(
            description="正向提示词",
            default="Professional photography of a mid-century modern living room, architectural digest style. Featuring wood, symmetry, Scandinavian design, mixed textures, vintage, reclaimed furniture, ergonomic layout. Cinematic lighting, atmospheric, photorealistic, sharp focus, 8k, highly detailed.",
        ),
        negative_prompt: str = Input(
            description="负向提示词",
            default="cartoon, illustration, 3d render, painting, drawing, anime, low quality, blurry, watermark, text, signature, people, humans, distorted perspective",
        ),
        num_inference_steps: int = Input(
            description="推理步数", default=20, ge=10, le=50
        ),
        guidance_scale: float = Input(
            description="CFG Scale", default=6.0, ge=1.0, le=15.0
        ),
        strength: float = Input(
            description="Denoise 强度 (img2img)", default=0.75, ge=0.0, le=1.0
        ),
        seed: int = Input(description="随机种子 (留空则随机)", default=None),
    ) -> Path:
        """运行 SDXL ControlNet 室内设计转换"""
        
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # 加载并缩放图片
        init_image = Image.open(image).convert("RGB")
        
        # 缩放到 1024 (保持宽高比，8 的倍数)
        max_size = 1024
        w, h = init_image.size
        scale = max_size / max(w, h)
        new_w = (int(w * scale) // 8) * 8
        new_h = (int(h * scale) // 8) * 8
        init_image = init_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 生成 - 双 ControlNet 配置
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=[init_image, init_image],  # 两个 ControlNet 都用同一张图
            strength=strength,  # denoise = 0.75
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            # 双 ControlNet 强度: [depth, tile]
            controlnet_conditioning_scale=[0.35, 0.15],
            # 双 ControlNet 时间范围
            control_guidance_start=[0.25, 0.6],
            control_guidance_end=[0.8, 1.0],
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)
