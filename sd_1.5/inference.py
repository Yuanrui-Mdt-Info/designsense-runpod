#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    EulerDiscreteScheduler,
    LCMScheduler,
    ControlNetModel,
)
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
try:
    from controlnet_aux import MLSDdetector
except ImportError:
    MLSDdetector = None
    print("Warning: controlnet_aux not installed. ControlNet features will be unavailable.")
from PIL import Image
import argparse
import numpy as np
from typing import Tuple, Union, List


def generate_image(prompt, lora_path, output_path="output.png"):
    model_id = "sd-legacy/stable-diffusion-v1-5"
    
    print(f"正在加载基础模型: {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to("cuda")

    # 替换调度器为 Euler a (通常效果更好)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"正在加载 LoRA: {lora_path}...")
    # 加载微调后的 LoRA 权重
    pipe.load_lora_weights(lora_path)

    print(f"正在生成: '{prompt}'")
    
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    image = pipe(
        prompt, 
        negative_prompt="low quality, bad anatomy, blurry", # 对应 WebUI: Negative Prompt
        height=512,                # 对应 WebUI: Height
        width=512,                 # 对应 WebUI: Width
        num_inference_steps=30,    # 对应 WebUI: Steps
        guidance_scale=7.5,        # 对应 WebUI: CFG Scale
        generator=generator        # 对应 WebUI: Seed
    ).images[0]

    image.save(output_path)
    print(f"图片已保存至: {output_path}")


def generate_image_img2img(prompt, image_path, lora_path, output_path="output.png"):
    # model_id = "sd-legacy/stable-diffusion-v1-5"
    model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    
    init_image = Image.open(image_path).convert("RGB")
    max_size = 768
    original_width, original_height = init_image.size
    # 计算缩放比例
    if max(original_width, original_height) > max_size:
        scale = max_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        new_width = original_width
        new_height = original_height
    # 确保尺寸是 8 的倍数 (VAE 要求)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    init_image = init_image.resize((new_width, new_height))
    
    print(f"正在加载基础模型: {model_id}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to("cuda")

    # 使用 LCM Scheduler 和 LoRA 进行加速
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
    
    # 注意：LCM 对 guidance_scale 非常敏感，通常锁死在 1.0 到 2.0 之间。
    # 这意味着你不能像以前那样通过调高 CFG (如 7-15) 来强行让 AI "听从" 复杂的 Prompt。
    # 如果你的 Prompt 非常复杂且必须严格遵循，LCM 可能幻觉严重。
    
    print(f"正在生成: '{prompt}'")
    
    generator = torch.Generator(device="cuda").manual_seed(1337)
    
    image = pipe(
        prompt=prompt,
        image=init_image,           # 必须：传入初始图片
        strength=0.75,              # 关键参数：0.0完全不像prompt(原图)，1.0完全像prompt(不看原图)。0.75是常用值
        guidance_scale=1.5,
        num_inference_steps=5,     # 图生图通常可以少一点步数，或者保持一致
        generator=generator        # 可选：固定种子
    ).images[0]
    
    image.save(output_path)
    print(f"图片已保存至: {output_path}")


# Model IDs for advanced features
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
    "japanese_bedroom": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_japan_bedroom.safetensors",
        "adapter_name": "japanese_bedroom",
        "lora_weight": 0.8,
    },
    "modern_living_room": {
        "lora_id": "Jkshdiaod/interior-design-lora",
        "filename": "sd1.5_modern_livingroom.safetensors",
        "adapter_name": "modern_living_room",
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


@torch.inference_mode()
@torch.autocast("cuda")
def segment_image(image, seg_image_processor, image_segmentor):
    """对图像进行语义分割"""
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
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


def generate_image_controlnet(
    prompt, 
    image_path, 
    output_path="output.png",
    negative_prompt="worst quality, low quality, normal quality, lowres, monochrome, grayscale, watermark, text, signature, bad anatomy, bad perspective",
    strength=0.5,
    guidance_scale=1.5,
    num_inference_steps=6,
    seed=None
):
    """使用 ControlNet 和语义分割进行室内设计重绘"""
    
    if MLSDdetector is None:
        raise ImportError("controlnet_aux is required for ControlNet features. Please install it: pip install controlnet-aux")
    
    # 加载输入图片
    init_image = Image.open(image_path).convert("RGB")
    
    # 图片缩放逻辑（保持宽高比，最大 768px，8 的倍数）
    max_size = 768
    original_width, original_height = init_image.size
    
    if max(original_width, original_height) > max_size:
        scale = max_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        new_width = original_width
        new_height = original_height

    # 确保是 8 的倍数
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"输入图片已调整至: {new_width}x{new_height}")
    
    # 加载 ControlNet 模型
    print("正在加载 ControlNet 模型...")
    controlnet = [
        ControlNetModel.from_pretrained(
            "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        ),
    ]
    
    # 加载 Pipeline
    print(f"正在加载基础模型: {BASE_MODEL_ID}...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    # 设置 LCM Scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # 加载 LCM LoRA
    print("正在加载 LCM LoRA...")
    pipe.load_lora_weights(
        LCM_LORA_ID, 
        adapter_name="lcm",
        # cache_dir="model_cache" # 不指定 cache_dir，使用默认的 HF 缓存（会自动持久化）
    )
    
    adapter_list = ["lcm", ]
    adapter_weights = [1.0, ]
    
    if "cyberpunk" in prompt and "interior" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["cyberpunk_interior"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if "floor plan" in prompt and "interior" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["floor_plan_interior"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if "clothing store" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["clothing_store_interior"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if "japanese" in prompt and "bedroom" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["japanese_bedroom"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if prompt.startswith("modern") and "living room" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["modern_living_room"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if "tropical" in prompt and "exterior" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["tropical_exterior"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    if "tropical" in prompt and "interior" in prompt:
        lora_config = STYLIZATION_LORA_CONFIG["tropical_interior"]
        pipe.load_lora_weights(
            lora_config["lora_id"],
            filename=lora_config["filename"],
            adapter_name=lora_config["adapter_name"],
        )
        adapter_list.append(lora_config["adapter_name"])
        adapter_weights.append(lora_config["lora_weight"])
    
    pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
    
    # 加载 MLSD 处理器
    print("正在加载 MLSD 处理器...")
    mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")

    # 初始化语义分割模型
    print("正在加载语义分割模型...")
    seg_image_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    ).to("cuda")
    
    # 控制项：这些区域不会被重绘（保持原样）
    control_items = [
        "windowpane;window",
        "column;pillar",
        "door;double;door",
    ]
    
    # 语义分割预处理
    print("正在处理语义分割...")
    real_seg = np.array(segment_image(init_image, seg_image_processor, image_segmentor))
    unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(i) for i in unique_colors]
    
    # 过滤掉 control_items，生成 mask
    chosen_colors, segment_items = filter_items(
        colors_list=unique_colors,
        items_list=segment_items,
        items_to_remove=control_items,
    )
    # 反转逻辑 - 标记整个房间，然后移除保护区域
    mask = np.ones_like(real_seg)  # 先全部标记为 1（要重绘）
    # 移除保护区域（control_items）
    for color, item in zip(unique_colors, segment_items):
        if item in control_items:  # 如果是保护项
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 0  # 标记为 0（不重绘）

    segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

    # MLSD 预处理
    print("正在处理 MLSD...")
    mlsd_img = mlsd_processor(init_image)
    mlsd_img = mlsd_img.resize(init_image.size)

    # 设置随机种子
    if seed is None:
        import os
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"使用随机种子: {seed}")
    generator = torch.Generator("cuda").manual_seed(seed)

    # 确保 LCM adapter 激活
    # pipe.set_adapters(["lcm"], adapter_weights=[1.0])
    pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
    print("已加载的适配器:", pipe.get_active_adapters())

    # 运行推理
    print(f"正在生成: '{prompt}'")
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

    result_image.save(output_path)
    print(f"图片已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion 1.5 推理脚本")
    parser.add_argument("--mode", type=str, choices=["txt2img", "img2img", "controlnet"], 
                       default="controlnet", help="生成模式: txt2img, img2img, controlnet")
    parser.add_argument("--prompt", type=str, default="modern interior design, minimalist, 8k, photorealistic", help="提示词")
    parser.add_argument("--image_path", type=str, default="/workspace/init_image.png", help="初始图片路径")
    parser.add_argument("--lora_path", type=str, default="/workspace/lora_output", help="LoRA 模型路径文件夹或 safetensors 文件")
    parser.add_argument("--output", type=str, default="/workspace/test_result.png", help="输出文件名")
    parser.add_argument("--negative_prompt", type=str, 
                       default="worst quality, low quality, normal quality, lowres, monochrome, grayscale, watermark, text, signature, bad anatomy, bad perspective",
                       help="负面提示词")
    parser.add_argument("--strength", type=float, default=0.5, help="img2img 强度 (0.0-1.0)")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="CFG Scale (LCM 建议 1.0-2.0)")
    parser.add_argument("--num_inference_steps", type=int, default=6, help="推理步数 (LCM 建议 4-8)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()
    
    if args.mode == "txt2img":
        generate_image(args.prompt, args.lora_path, args.output)
    elif args.mode == "img2img":
        generate_image_img2img(args.prompt, args.image_path, args.lora_path, args.output)
    elif args.mode == "controlnet":
        generate_image_controlnet(
            args.prompt, 
            args.image_path, 
            args.output,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed
        )
