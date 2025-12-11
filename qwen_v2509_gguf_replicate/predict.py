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
from pathlib import Path
from typing import Optional, List

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

    def predict(
        self,
        image: CogPath = Input(description="输入图像（主图像）"),
        image2: Optional[CogPath] = Input(
            description="第二张图像（可选，用于多图像编辑）",
            default=None,
        ),
        image3: Optional[CogPath] = Input(
            description="第三张图像（可选，用于多图像编辑）",
            default=None,
        ),
        prompt: str = Input(
            description="编辑提示词",
            default="",
        ),
        negative_prompt: str = Input(
            description="负面提示词",
            default="blurry, low quality, distorted",
        ),
        steps: int = Input(
            description="推理步数",
            default=4,
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
        import torch
        import numpy as np
        from nodes import (
            KSampler,
            VAEEncode,
            VAEDecode,
        )

        print(f"[Predict] Processing images with prompt: {prompt}")
        print(f"[Predict] Input image path: {image}")
        print(f"[Predict] Image2: {image2}")
        print(f"[Predict] Image3: {image3}")
        
        # 加载图像列表
        images = []
        image_paths = [image, image2, image3]
        
        for idx, img_path in enumerate(image_paths, 1):
            if img_path is None:
                continue
                
            img_str = str(img_path)
            if not os.path.exists(img_str):
                print(f"[Predict] Warning: Image{idx} file not found: {img_str}, skipping...")
                continue
                
            try:
                img = Image.open(img_str).convert("RGB")
                print(f"[Predict] Image{idx} loaded: {img.size}")
                images.append(img)
            except Exception as e:
                print(f"[Predict] Error loading image{idx}: {e}")
                raise
        
        if not images:
            raise ValueError("至少需要提供一张输入图像")
        
        print(f"[Predict] Total images loaded: {len(images)}")
        
        # 使用第一张图像的尺寸
        first_image = images[0]
        width, height = first_image.size
        print(f"[Predict] Base image dimensions: {width}x{height}")

        # 确保尺寸是 8 的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        print(f"[Predict] Adjusted dimensions (multiple of 8): {width}x{height}")
        
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
                print(f"[Predict] Image{idx} resized to {width}x{height}")
            processed_images.append(img)

        # 转换为 tensor
        image_tensors = []
        for idx, img in enumerate(processed_images, 1):
            image_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            image_tensors.append(image_tensor)
            print(f"[Predict] Image{idx} tensor shape: {image_tensor.shape}")

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
                    print(f"[Predict] Found TextEncodeQwenImageEditPlus at {import_path}")
                    break
            except (ImportError, AttributeError):
                continue
        
        if not use_multi_image:
            print("[Predict] TextEncodeQwenImageEditPlus not found, using single image mode")
            if len(image_tensors) > 1:
                print("[Predict] Warning: Multiple images provided but multi-image node not available, using first image only")
                image_tensors = [image_tensors[0]]

        if use_multi_image and TextEncodeQwenImageEditPlus and len(image_tensors) > 1:
            # 多图像编辑模式
            print(f"[Predict] Encoding with {len(image_tensors)} images...")
            text_encode = TextEncodeQwenImageEditPlus()
            
            # 准备图像输入（最多3张）
            image1_tensor = image_tensors[0] if len(image_tensors) > 0 else None
            image2_tensor = image_tensors[1] if len(image_tensors) > 1 else None
            image3_tensor = image_tensors[2] if len(image_tensors) > 2 else None
            
            # 编码 positive conditioning
            # TextEncodeQwenImageEditPlus 的 encode 方法签名可能不同，需要根据实际实现调整
            try:
                positive_cond = text_encode.encode(
                    clip=self.clip,
                    vae=self.vae,
                    text=prompt if prompt else "",
                    image1=image1_tensor,
                    image2=image2_tensor,
                    image3=image3_tensor,
                )[0]
            except TypeError:
                # 如果参数不匹配，尝试其他调用方式
                positive_cond = text_encode.encode(
                    self.clip,
                    prompt if prompt else "",
                    self.vae,
                    image1_tensor,
                    image2_tensor,
                    image3_tensor,
                )[0]
            
            # 编码 negative conditioning（如果没有提供负面提示词，使用空字符串）
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
            print("[Predict] Using single image mode...")
            from nodes import CLIPTextEncode
            
            clip_encode = CLIPTextEncode()
            positive_cond = clip_encode.encode(self.clip, prompt if prompt else "")[0]
            negative_cond = clip_encode.encode(self.clip, negative_prompt if negative_prompt else "")[0]

        # 图像编码到 latent（使用第一张图像）
        vae_encode = VAEEncode()
        latent = vae_encode.encode(self.vae, image_tensors[0])[0]
        
        # 诊断信息 - latent 是字典，包含 'samples' 键
        if isinstance(latent, dict) and 'samples' in latent:
            latent_samples = latent['samples']
            print(f"[Predict] Latent type: dict with 'samples' key")
            print(f"[Predict] Latent samples shape: {latent_samples.shape}")
            print(f"[Predict] Latent samples dtype: {latent_samples.dtype}")
            print(f"[Predict] Latent samples min: {latent_samples.min().item():.3f}")
            print(f"[Predict] Latent samples max: {latent_samples.max().item():.3f}")
            print(f"[Predict] Latent samples mean: {latent_samples.mean().item():.3f}")
            print(f"[Predict] Latent samples std: {latent_samples.std().item():.3f}")
        else:
            print(f"[Predict] Latent type: {type(latent)}")
            if isinstance(latent, dict):
                print(f"[Predict] Latent keys: {list(latent.keys())}")

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
        output_np = (decoded.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)

        # 保存输出
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        print(f"[Predict] Done! Seed: {seed}")
        return CogPath(output_path)
