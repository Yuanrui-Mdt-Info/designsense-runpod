#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetInpaintPipeline, LCMScheduler, ControlNetModel, AutoencoderKL
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from controlnet_aux import MLSDdetector
from PIL import Image
import shutil
from typing import Tuple, Union, List, Optional
import numpy as np

# Model IDs
BASE_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"


STYLIZATION_LORA_CONFIG = {
    "cyberpunk_interior": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_cyberpunk_interior_design.safetensors",
        "adapter_name": "cyberpunk_interior",
        "lora_weight": 0.8,
    },
    "floor_plan_interior": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_floor_plan_interior_design.safetensors",
        "adapter_name": "floor_plan_interior",
        "lora_weight": 0.8,
    },
    "clothing_store_interior": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_clothing_store_interior_design.safetensors",
        "adapter_name": "clothing_store_interior",
        "lora_weight": 0.8,
    },
    "tropical_exterior": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_tropical_exterior.safetensors",
        "adapter_name": "tropical_exterior",
        "lora_weight": 0.7,
    },
    "tropical_interior": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_tropical_interior.safetensors",
        "adapter_name": "tropical_interior",
        "lora_weight": 0.7,
    },
}


def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_remove: Union[List, np.ndarray],
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """过滤掉指定项目，用于生成 mask"""
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items


def ade_palette() -> List[List[int]]:
    """ADE20K 调色板，用于语义分割颜色映射"""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


# 颜色到物体的映射（简化版，只包含常用室内设计相关）
COLOR_MAPPING_RGB = {
    (120, 120, 120): "wall",
    (230, 230, 230): "windowpane;window",
    (8, 255, 51): "door;double;door",
    (255, 8, 41): "column;pillar",
    (204, 5, 255): "bed",
    (11, 102, 255): "sofa;couch;lounge",
    (204, 70, 3): "chair",
    (255, 6, 82): "table",
    (255, 7, 71): "shelf",
    (80, 50, 50): "floor;flooring",
    (255, 9, 92): "rug;carpet;carpeting",
    (224, 255, 8): "lamp",
}


def map_colors_rgb(color: tuple) -> str:
    """将 RGB 颜色映射到物体名称"""
    return COLOR_MAPPING_RGB.get(color, "unknown")


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        )
        
        controlnet = [
            ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
            ),
        ]
        
        print("Loading pipeline ...")
        
        # 直接加载构建阶段下载好的模型
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            BASE_MODEL_ID,
            vae=vae,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to("cuda")

        # Set up LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load LCM LoRA
        print("Loading LCM LoRA...")
        self.pipe.load_lora_weights(
            LCM_LORA_ID, 
            adapter_name="lcm",
            # cache_dir="model_cache" # 使用默认缓存目录
        )
        self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        
        self.mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")

        # 初始化语义分割模型
        print("Loading segmentation models...")
        self.seg_image_processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        self.image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to("cuda")
        
        # 控制项：这些区域不会被重绘（保持原样）
        self.control_items = [
            "windowpane;window",
            "column;pillar",
            "door;double;door",
        ]
        
        # Warmup (optional but good practice)
        # self.pipe(prompt="warmup", image=Image.new('RGB', (512, 512)), num_inference_steps=1)
    
    def _select_lora_by_prompt(self, prompt: str):
        """根据 prompt 选择对应的 LoRA 配置"""
        prompt_lower = prompt.lower()
        
        # 匹配逻辑与 inference.py 保持一致
        if "floor plan" in prompt_lower and "interior" in prompt_lower:
            return "floor_plan_interior"
        elif "clothing store" in prompt_lower:
            return "clothing_store_interior"
        elif "cyberpunk" in prompt_lower and "interior" in prompt_lower:
            return "cyberpunk_interior"
        elif "tropical" in prompt_lower and "exterior" in prompt_lower:
            return "tropical_exterior"
        elif "tropical" in prompt_lower and "interior" in prompt_lower:
            return "tropical_interior"
        
        return None
    
    @torch.inference_mode()
    @torch.autocast("cuda")
    def segment_image(self, image):
        """对图像进行语义分割"""
        pixel_values = self.seg_image_processor(image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        # 将 CUDA tensor 转换为 CPU numpy 数组
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
            
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
            
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert("RGB")
        
        return seg_image

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
            default=0.5,
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
        num_images: int = Input(
            description="Number of images to generate (1-4 recommended)",
            default=1,
            ge=1,
            le=4
        ),
        seed: Optional[int] = Input(
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
    ) -> List[Path]:
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
        
        adapter_list = ["lcm", ]
        adapter_weights = [1.0, ]
        
        lora_key = self._select_lora_by_prompt(prompt)
        if lora_key and lora_key in STYLIZATION_LORA_CONFIG:
            lora_config = STYLIZATION_LORA_CONFIG[lora_key]
            # print(f"Loading style LoRA: {lora_key}...")
            
            try:
                # 从 Hugging Face 加载 LoRA（会自动下载和缓存）
                self.pipe.load_lora_weights(
                    lora_config["lora_id"],
                    weight_name=lora_config["filename"],
                    adapter_name=lora_config["adapter_name"],
                )
                adapter_list.append(lora_config["adapter_name"])
                adapter_weights.append(lora_config["lora_weight"])
                # print(f"Successfully loaded LoRA: {lora_key}")
            except Exception as e:
                error_msg = str(e)
                if "already in use" in error_msg:
                    print(f"LoRA {lora_key} already loaded, reusing...")
                    adapter_list.append(lora_config["adapter_name"])
                    adapter_weights.append(lora_config["lora_weight"])
                else:
                    print(f"Failed to load LoRA {lora_key}: {error_msg}")
        
        self.pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
        print(f"Active adapters: {self.pipe.get_active_adapters()}")

        # 3. 语义分割预处理
        print("Processing semantic segmentation...")
        real_seg = np.array(self.segment_image(init_image))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        
        # 过滤掉 control_items，生成 mask
        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=self.control_items,
        )
        
        # 反转逻辑：标记整个房间，然后移除保护区域
        mask = np.ones_like(real_seg)  # 先全部标记为 1（要重绘）
        # 移除保护区域（control_items）
        for color, item in zip(unique_colors, segment_items):
            if item in self.control_items:  # 如果是保护项
                color_matches = (real_seg == color).all(axis=2)
                mask[color_matches] = 0  # 标记为 0（不重绘）

        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

        # MLSD 预处理
        print("Processing MLSD...")
        mlsd_img = self.mlsd_processor(init_image)
        mlsd_img = mlsd_img.resize(init_image.size)

        # Run Inference
        result_images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            mask_image=mask_image,
            control_image=[segmentation_cond_image, mlsd_img],
            controlnet_conditioning_scale=[0.4, 0.2],
            control_guidance_start=[0, 0.1],
            control_guidance_end=[0.5, 0.25],
            num_images_per_prompt=num_images,
        ).images

        # Save Output
        output_paths = []
        for i, result_image in enumerate(result_images):
            if num_images > 1:
                output_path = f"/tmp/out_{i+1}.{output_format}"
            else:
                output_path = f"/tmp/out.{output_format}"
            result_image.save(output_path, quality=output_quality)
            output_paths.append(Path(output_path))
            print(f"Saved image {i+1}/{num_images}: {output_path}")
        
        return output_paths
    
        # output_path = f"/tmp/out.{output_format}"
        # result_image.save(output_path, quality=output_quality)
        
        # return Path(output_path)
