#! /usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Qwen-Image-Edit-2509 GGUF Predictor for Replicate
基于 ComfyUI + ComfyUI-GGUF 运行量化模型
"""

import os
import sys
import json
import subprocess
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Union
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import numpy as np


# 自动检测 ComfyUI 路径
# 如果环境变量中有 COMFYUI_PATH 则使用，否则默认为 /ComfyUI (Replicate 环境)
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/ComfyUI")

# 添加 ComfyUI 路径
if COMFYUI_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_PATH)

from cog import BasePredictor, Input, Path as CogPath
from PIL import Image


# 模型文件信息 - 使用 COMFYUI_PATH 动态构建路径
MODELS = {
    "unet": {
        "repo": "QuantStack/Qwen-Image-Edit-2509-GGUF",
        "filename": "Qwen-Image-Edit-2509-Q4_K_M.gguf",
        "dest": f"{COMFYUI_PATH}/models/unet/Qwen-Image-Edit-2509-Q4_K_M.gguf",
        # "filename": "Qwen-Image-Edit-2509-Q2_K.gguf",
        # "dest": f"{COMFYUI_PATH}/models/unet/Qwen-Image-Edit-2509-Q2_K.gguf",
    },
    "text_encoder": {
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        # "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        # "dest": f"{COMFYUI_PATH}/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
        "dest": f"{COMFYUI_PATH}/models/text_encoders/qwen_2.5_vl_7b.safetensors",
    },
    # diffusion model not used
    # "mmproj": {
    #     "repo": "QuantStack/Qwen-Image-Edit-GGUF",
    #     "filename": "mmproj/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
    #     "dest": f"{COMFYUI_PATH}/models/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
    # },
    "vae": { 
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": f"{COMFYUI_PATH}/models/vae/qwen_image_vae.safetensors",
        
        # "repo": "QuantStack/Qwen-Image-Edit-GGUF",
        # "filename": "VAE/Qwen_Image-VAE.safetensors",
        # "dest": f"{COMFYUI_PATH}/models/vae/Qwen_Image-VAE.safetensors",
    },
    # 功能性 LoRA
    "lora_white": {
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/loras/Qwen-Image-Edit-2509-White_to_Scene.safetensors",
        "dest": f"{COMFYUI_PATH}/models/loras/Qwen-Image-Edit-2509-White_to_Scene.safetensors",
    },
    "lora_relight": {
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/loras/Qwen-Image-Edit-2509-Relight.safetensors",
        "dest": f"{COMFYUI_PATH}/models/loras/Qwen-Image-Edit-2509-Relight.safetensors",
    },
}


def download_model(repo: str, filename: str, dest: str):
    """使用 huggingface-hub 下载模型文件"""
    if os.path.exists(dest):
        print(f"[Download] {filename} already exists, skipping...")
        return

    print(f"[Download] Downloading {filename} from {repo}...")
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=os.path.dirname(dest),
        local_dir_use_symlinks=False,
    )
    # 如果下载路径和目标路径不同，移动文件
    if local_path != dest:
        os.rename(local_path, dest)
    print(f"[Download] {filename} downloaded to {dest}")


def ade_palette():
    """ADE20K 调色板"""
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
}


def map_colors_rgb(color: tuple) -> str:
    return COLOR_MAPPING_RGB.get(color, "unknown")


class Predictor(BasePredictor):
    def setup(self):
        """加载模型到 GPU - 在容器启动时执行一次"""
        print("[Setup] Downloading models...")

        # 下载所有模型文件
        for name, info in MODELS.items():
            download_model(info["repo"], info["filename"], info["dest"])

        print("[Setup] Initializing ComfyUI...")

        # 导入 ComfyUI 核心模块
        import folder_paths
        import nodes
        import comfy.model_management as model_management

        self.folder_paths = folder_paths
        self.nodes = nodes
        self.model_management = model_management

        # 预加载模型
        print("[Setup] Loading GGUF models...")
        self._load_models()
        
        print("[Setup] Loading segmentation models...")
        self.seg_image_processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        self.image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to("cuda")
        
        self.control_items = [
            "windowpane;window",
            "column;pillar",
            "door;double;door",
            "wall", # 不改变墙面颜色
        ]
        
        print("[Setup] Complete!")

    def _load_models(self):
        """预加载模型"""
        # UNet 使用 GGUF 格式（需要 ComfyUI-GGUF 插件）
        from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF

        # 加载 UNet (GGUF)
        unet_loader = UnetLoaderGGUF()
        self.unet = unet_loader.load_unet(
            unet_name="Qwen-Image-Edit-2509-Q4_K_M.gguf"
        )[0]
        
        # 加载功能性 LoRA
        from nodes import LoraLoaderModelOnly
        lora_loader = LoraLoaderModelOnly()
        self.unet, _ = lora_loader.load_lora_model_only(
            model=self.unet,
            lora_name="Qwen-Image-Edit-2509-White_to_Scene.safetensors",
            strength=0.7
        )
        self.unet, _ = lora_loader.load_lora_model_only(
            model=self.unet,
            lora_name="Qwen-Image-Edit-2509-Relight.safetensors",
            strength=0.5
        )
        
        # 加载 CLIP (Text Encoder) - 使用官方 safetensors 格式
        from custom_nodes.ComfyUI_GGUF.nodes import CLIPLoaderGGUF
        clip_loader = CLIPLoaderGGUF()
        self.clip = clip_loader.load_clip(
            clip_name="qwen_2.5_vl_7b.safetensors",
            type="qwen_image",
        )[0]

        # 加载 VAE
        from nodes import VAELoader
        vae_loader = VAELoader()
        self.vae = vae_loader.load_vae(vae_name="qwen_image_vae.safetensors")[0]
    
    @torch.inference_mode()
    def segment_image(self, image):
        """对图像进行语义分割"""
        # 确保输入是 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = self.seg_image_processor(image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        # 转换为彩色分割图以便匹配
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg.cpu() == label, :] = color
            
        return color_seg

    def predict(
        self,
        image: CogPath = Input(description="输入图像（主图像）"),
        prompt: str = Input(
            description="编辑提示词",
            default="",
        ),
        negative_prompt: str = Input(
            description="负面提示词",
            default="blurry, low quality, distorted",
        ),
        steps: int = Input(
            description="推理步数 (建议: Lightning模式4步 / 普通模式25-30步)",
            default=25,
            ge=1,
            le=100,
        ),
        cfg_scale: float = Input(
            description="CFG Scale (引导强度)",
            default=1.0,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="随机种子 (-1 为随机)",
            default=-1,
        ),
        denoise: float = Input(
            description="去噪强度",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
    ) -> CogPath:
        """执行图像编辑推理"""
        from nodes import (
            KSampler,
            VAEEncode,
            VAEDecode,
        )
        
        # 加载图像列表
        image_str = str(image)
        if not os.path.exists(image_str):
            raise ValueError(f"Image file not found: {image_str}")
            
        img = Image.open(image_str).convert("RGB")
        images = [img]
        
        # print(f"[Predict] Total images loaded: {len(images)}")
        
        # 使用第一张图像的尺寸
        first_image = images[0]
        width, height = first_image.size
        # print(f"[Predict] Base image dimensions: {width}x{height}")

        # 确保尺寸是 8 的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        # print(f"[Predict] Adjusted dimensions (multiple of 8): {width}x{height}")
        
        # 调整所有图像尺寸
        processed_images = []
        # 兼容新旧版本的 Pillow
        try:
            resize_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resize_filter = Image.LANCZOS
            
        for idx, img in enumerate(images, 1):
            if img.size != (width, height):
                img = img.resize((width, height), resize_filter)
                # print(f"[Predict] Image{idx} resized to {width}x{height}")
            processed_images.append(img)

        # 转换为 tensor
        image_tensors = []
        for idx, img in enumerate(processed_images, 1):
            image_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            image_tensors.append(image_tensor)
            # print(f"[Predict] Image{idx} tensor shape: {image_tensor.shape}")

        # 使用 TextEncodeQwenImageEditPlus 节点（如果可用）
        # 否则回退到单图像模式
        use_multi_image = False
        TextEncodeQwenImageEditPlus = None
        
        # 尝试多种可能的导入路径
        import_paths = [
            "custom_nodes.ComfyUI-Qwen-Image.nodes",
            "custom_nodes.comfyui_qwen_image_edit.nodes",
            "custom_nodes.Qwen_Image_ComfyUI.nodes",
        ]
        
        for import_path in import_paths:
            try:
                module = __import__(import_path, fromlist=["TextEncodeQwenImageEditPlus"])
                if hasattr(module, "TextEncodeQwenImageEditPlus"):
                    TextEncodeQwenImageEditPlus = module.TextEncodeQwenImageEditPlus
                    use_multi_image = True
                    # print(f"[Predict] Found TextEncodeQwenImageEditPlus at {import_path}")
                    break
            except (ImportError, AttributeError):
                continue
        
        if not use_multi_image:
            # print("[Predict] TextEncodeQwenImageEditPlus not found, using single image mode")
            pass

        if use_multi_image and TextEncodeQwenImageEditPlus:
            # 单图像编辑模式
            text_encode = TextEncodeQwenImageEditPlus()
            
            # 准备图像输入
            image1_tensor = image_tensors[0]
            
            # 编码 positive conditioning
            try:
                positive_cond = text_encode.encode(
                    clip=self.clip,
                    vae=self.vae,
                    text=prompt if prompt else "",
                    image1=image1_tensor,
                    image2=None,
                    image3=None,
                )[0]
            except TypeError:
                # 如果参数不匹配，尝试其他调用方式
                positive_cond = text_encode.encode(
                    self.clip,
                    prompt if prompt else "",
                    self.vae,
                    image1_tensor,
                    None,
                    None,
                )[0]
            
            # 编码 negative conditioning
            try:
                negative_cond = text_encode.encode(
                    clip=self.clip,
                    vae=self.vae,
                    text=negative_prompt if negative_prompt else "",
                    image1=None,
                    image2=None,
                    image3=None,
                )[0]
            except TypeError:
                negative_cond = text_encode.encode(
                    self.clip,
                    negative_prompt if negative_prompt else "",
                    self.vae,
                    None,
                    None,
                    None,
                )[0]
        else:
            # 单图像模式 - 使用第一张图像
            # print("[Predict] Using single image mode...")
            from nodes import CLIPTextEncode
            
            clip_encode = CLIPTextEncode()
            positive_cond = clip_encode.encode(self.clip, prompt if prompt else "")[0]
            negative_cond = clip_encode.encode(self.clip, negative_prompt if negative_prompt else "")[0]

        # 图像编码到 latent（使用第一张图像）
        vae_encode = VAEEncode()
        latent = vae_encode.encode(self.vae, image_tensors[0])[0]
        
        # 诊断信息 - latent 是字典，包含 'samples' 键
        # if isinstance(latent, dict) and 'samples' in latent:
        #     latent_samples = latent['samples']
        #     print(f"[Predict] Latent type: dict with 'samples' key")
        #     print(f"[Predict] Latent samples shape: {latent_samples.shape}")
        #     print(f"[Predict] Latent samples dtype: {latent_samples.dtype}")
        #     print(f"[Predict] Latent samples min: {latent_samples.min().item():.3f}")
        #     print(f"[Predict] Latent samples max: {latent_samples.max().item():.3f}")
        #     print(f"[Predict] Latent samples mean: {latent_samples.mean().item():.3f}")
        #     print(f"[Predict] Latent samples std: {latent_samples.std().item():.3f}")
        # else:
        #     print(f"[Predict] Latent type: {type(latent)}")
        #     if isinstance(latent, dict):
        #         print(f"[Predict] Latent keys: {list(latent.keys())}")

        # 设置随机种子
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # KSampler 采样
        sampler = KSampler()
        samples = sampler.sample(
            model=self.unet,
            seed=seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name="euler",
            scheduler="simple",
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            denoise=denoise,
        )[0]

        # 解码 latent 到图像
        vae_decode = VAEDecode()
        decoded = vae_decode.decode(self.vae, samples)[0]

        # 转换为 PIL Image
        output_np = (decoded.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)

        # --- 新增：后处理合成逻辑 ---
        # 使用第一张输入图作为结构参考
        ref_image = processed_images[0] # 这是已经 resize 过的输入图
        
        # 如果 output_image 尺寸和 ref_image 不一致，调整 output
        if output_image.size != ref_image.size:
            output_image = output_image.resize(ref_image.size, resize_filter)

        print("[Predict] Applying structural preservation mask...")
        
        # 1. 对原图进行分割
        real_seg = self.segment_image(ref_image)
        
        # 2. 提取需要保留的物体 Mask
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        
        chosen_colors = []
        for color in unique_colors:
            item_name = map_colors_rgb(color)
            if item_name in self.control_items:
                chosen_colors.append(color)
        
        # 生成二值 Mask (1 = 保留原图, 0 = 使用生成图)
        mask = np.zeros((real_seg.shape[0], real_seg.shape[1]), dtype=np.float32)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1.0
            
        # 3. 图像合成： Final = Original * Mask + Generated * (1 - Mask)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        
        # 边缘羽化，使接缝更自然
        from PIL import ImageFilter
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
        
        final_image = Image.composite(ref_image, output_image, mask_pil)
        
        # 保存输出
        output_path = "/tmp/output.png"
        final_image.save(output_path)
        print(f"[Predict] Done! Seed: {seed}")
        return CogPath(output_path)
