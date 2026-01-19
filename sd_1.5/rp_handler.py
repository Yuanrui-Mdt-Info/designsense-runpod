#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import runpod
import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    LCMScheduler,
    ControlNetModel,
    AutoencoderKL,
)
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from controlnet_aux import MLSDdetector
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from typing import Union, List, Tuple

# HF_HOME 环境变量控制缓存位置
print(f"[Init] HF_HOME = {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

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


def _select_lora_by_prompt(prompt: str):
    """根据 prompt 选择对应的 LoRA 配置"""
    prompt_lower = prompt.lower()
    
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


# ============= 全局初始化模型 =============
print("[Init] Loading models...")

print("[Init] Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
)

print("[Init] Loading ControlNet...")
controlnet = [
    ControlNetModel.from_pretrained(
        "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
    ),
]

print("[Init] Loading pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL_ID,
    vae=vae,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

print("[Init] Loading LCM LoRA...")
pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
pipe.set_adapters(["lcm"], adapter_weights=[1.0])

print("[Init] Loading MLSD processor...")
mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")

print("[Init] Loading segmentation models...")
seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
).to("cuda")

control_items = [
    "windowpane;window",
    "column;pillar",
    "door;double;door",
]

print("[Init] Model loaded successfully!")


@torch.inference_mode()
@torch.autocast("cuda")
def segment_image(image):
    """对图像进行语义分割"""
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
        
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
        
    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert("RGB")
    
    return seg_image


def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event["input"]
        
        # 解析输入参数
        prompt = input_data.get("prompt", "modern interior design, minimalist, 8k, photorealistic")
        negative_prompt = input_data.get("negative_prompt", "worst quality, low quality, normal quality, lowres, monochrome, grayscale, watermark, text, signature, bad anatomy, bad perspective")
        strength = input_data.get("strength", 0.5)
        guidance_scale = input_data.get("guidance_scale", 1.5)
        num_inference_steps = input_data.get("num_inference_steps", 6)
        seed = input_data.get("seed", None)
        
        # 解析图片输入
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            return {"error": "Missing 'image_base64' parameter"}
        
        # 解码图片
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # 图片缩放逻辑
        max_size = 768
        original_width, original_height = init_image.size
        
        if max(original_width, original_height) > max_size:
            scale = max_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
        else:
            new_width = original_width
            new_height = original_height

        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"[Handler] Image resized to: {new_width}x{new_height}")
        
        # 加载风格 LoRA
        adapter_list = ["lcm"]
        adapter_weights = [1.0]
        
        lora_key = _select_lora_by_prompt(prompt)
        if lora_key and lora_key in STYLIZATION_LORA_CONFIG:
            lora_config = STYLIZATION_LORA_CONFIG[lora_key]
            print(f"[Handler] Loading style LoRA: {lora_key}...")
            
            try:
                pipe.load_lora_weights(
                    lora_config["lora_id"],
                    weight_name=lora_config["filename"],
                    adapter_name=lora_config["adapter_name"],
                )
                adapter_list.append(lora_config["adapter_name"])
                adapter_weights.append(lora_config["lora_weight"])
            except Exception as e:
                error_msg = str(e)
                if "already in use" in error_msg:
                    print(f"[Handler] LoRA {lora_key} already loaded, reusing...")
                    adapter_list.append(lora_config["adapter_name"])
                    adapter_weights.append(lora_config["lora_weight"])
                else:
                    print(f"[Handler] Failed to load LoRA {lora_key}: {error_msg}")
        
        pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
        print(f"[Handler] Active adapters: {pipe.get_active_adapters()}")
        
        # 语义分割预处理
        print("[Handler] Processing semantic segmentation...")
        real_seg = np.array(segment_image(init_image))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        
        # 生成 mask
        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=control_items,
        )
        
        mask = np.ones_like(real_seg)
        for color, item in zip(unique_colors, segment_items):
            if item in control_items:
                color_matches = (real_seg == color).all(axis=2)
                mask[color_matches] = 0

        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

        # MLSD 预处理
        print("[Handler] Processing MLSD...")
        mlsd_img = mlsd_processor(init_image)
        mlsd_img = mlsd_img.resize(init_image.size)

        # 设置随机种子
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"[Handler] Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # 运行推理
        print(f"[Handler] Generating: '{prompt}'")
        result_image = pipe(
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
        ).images[0]

        # 编码输出图片
        buf = BytesIO()
        result_image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "output_image_base64": img_str,
            "prompt": prompt,
            "seed": seed,
            "size": f"{new_width}x{new_height}"
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# 启动 serverless
runpod.serverless.start({"handler": handler})
