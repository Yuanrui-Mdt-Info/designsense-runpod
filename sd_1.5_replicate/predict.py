#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler
from PIL import Image
import shutil

# Model IDs
BASE_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Loading pipeline from build cache...")
        
        # 直接加载构建阶段下载好的模型
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "model_cache",  # 对应 cog.yaml 里创建的目录
            torch_dtype=torch.float16,
            local_files_only=True # 确保不联网
        ).to("cuda")

        # Set up LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load LCM LoRA
        print("Loading LCM LoRA...")
        self.pipe.load_lora_weights(
            "model_cache/lcm_lora", 
            adapter_name="lcm",
            local_files_only=True
        )
        self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        
        # Warmup (optional but good practice)
        # self.pipe(prompt="warmup", image=Image.new('RGB', (512, 512)), num_inference_steps=1)

    def predict(
        self,
        image: Path = Input(description="Input image for img2img"),
        prompt: str = Input(
            description="Input prompt",
            default="modern interior design, minimalist, 8k, photorealistic"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="worst quality, low quality, normal quality, lowres, monochrome, grayscale, watermark, text, signature, bad anatomy, bad perspective"
        ),
        strength: float = Input(
            description="Strength of the img2img transformation (0.0 to 1.0)",
            default=0.75,
            ge=0.0,
            le=1.0
        ),
        guidance_scale: float = Input(
            description="Guidance scale (CFG). For LCM, keep this low (1.0 - 2.0)",
            default=1.5,
            ge=1.0,
            le=3.0
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps. For LCM, 4-8 is usually sufficient.",
            default=6,
            ge=2,
            le=20
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="png"
        ),
        output_quality: int = Input(
            description="Quality of the output image (0-100)",
            default=90,
            ge=0,
            le=100
        )
    ) -> Path:
        """Run a single prediction on the model"""
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # 1. Load Input Image
        init_image = Image.open(image).convert("RGB")
        
        # 2. Resize Logic (Keep aspect ratio, max 768, multiple of 8)
        max_size = 768
        original_width, original_height = init_image.size
        
        if max(original_width, original_height) > max_size:
            scale = max_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            new_width = original_width
            new_height = original_height

        # Ensure multiple of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Input image resized to: {new_width}x{new_height}")

        # 3. Ensure LCM adapter is active (no extra LoRA)
        self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])

        # 4. Run Inference
        result_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        # 5. Save Output
        output_path = f"/tmp/out.{output_format}"
        result_image.save(output_path, quality=output_quality)
        
        return Path(output_path)
