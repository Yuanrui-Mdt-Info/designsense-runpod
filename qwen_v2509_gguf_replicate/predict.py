"""
Qwen-Image-Edit-2509 GGUF Predictor for Replicate
基于 ComfyUI + ComfyUI-GGUF 运行量化模型
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional

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
        # "filename": "Qwen-Image-Edit-2509-Q4_K_M.gguf",
        # "dest": f"{COMFYUI_PATH}/models/unet/Qwen-Image-Edit-2509-Q4_K_M.gguf",
        "filename": "Qwen-Image-Edit-2509-Q2_K.gguf",
        "dest": f"{COMFYUI_PATH}/models/unet/Qwen-Image-Edit-2509-Q2_K.gguf",
    },
    "text_encoder": {
        # "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        # # # "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        # # # "dest": f"{COMFYUI_PATH}/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        # "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
        # "dest": f"{COMFYUI_PATH}/models/text_encoders/qwen_2.5_vl_7b.safetensors",
        
        "repo": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
        "dest": f"{COMFYUI_PATH}/models/text_encoders/Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
    },
    "mmproj": {
        "repo": "QuantStack/Qwen-Image-Edit-GGUF",
        "filename": "mmproj/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
        "dest": f"{COMFYUI_PATH}/models/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
    },
    "vae": {
        # "repo": "Qwen/Qwen-Image-Edit-2509",
        # "filename": "vae/diffusion_pytorch_model.safetensors",
        # "dest": f"{COMFYUI_PATH}/models/vae/diffusion_pytorch_model.safetensors",
        
        # "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        # "filename": "split_files/vae/qwen_image_vae.safetensors",
        # "dest": f"{COMFYUI_PATH}/models/vae/qwen_image_vae.safetensors",
        
        "repo": "QuantStack/Qwen-Image-Edit-GGUF",
        "filename": "VAE/Qwen_Image-VAE.safetensors",
        "dest": f"{COMFYUI_PATH}/models/vae/Qwen_Image-VAE.safetensors",
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
            # unet_name="Qwen-Image-Edit-2509-Q4_K_M.gguf"
            unet_name="Qwen-Image-Edit-2509-Q2_K.gguf"
        )[0]

        # 加载 CLIP (Text Encoder) - 使用官方 safetensors 格式
        
        # from custom_nodes.ComfyUI_GGUF.nodes import CLIPLoaderGGUF
        # clip_loader = CLIPLoaderGGUF()
        # self.clip = clip_loader.load_clip(
        #     clip_name="Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
        #     type="qwen_image",
        # )[0]
        
        from custom_nodes.ComfyUI_GGUF.nodes import DualCLIPLoaderGGUF
        clip_loader = DualCLIPLoaderGGUF()
        self.clip = clip_loader.load_clip(
            clip_name1="Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
            clip_name2="Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
            type="qwen_image",
        )[0]

        # 加载 VAE
        from nodes import VAELoader

        vae_loader = VAELoader()
        # self.vae = vae_loader.load_vae(vae_name="diffusion_pytorch_model.safetensors")[0]
        # self.vae = vae_loader.load_vae(vae_name="qwen_image_vae.safetensors")[0]
        self.vae = vae_loader.load_vae(vae_name="Qwen_Image-VAE.safetensors")[0]

    def predict(
        self,
        image: CogPath = Input(description="输入图像"),
        prompt: str = Input(
            description="编辑提示词",
            default="Enhance the image quality",
        ),
        negative_prompt: str = Input(
            description="负面提示词",
            default="blurry, low quality, distorted",
        ),
        steps: int = Input(
            description="推理步数",
            default=28,
            ge=1,
            le=100,
        ),
        cfg_scale: float = Input(
            description="CFG Scale (引导强度)",
            default=6.0,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="随机种子 (-1 为随机)",
            default=-1,
        ),
    ) -> CogPath:
        """执行图像编辑推理"""
        import torch
        import numpy as np
        from nodes import (
            CLIPTextEncode,
            KSampler,
            VAEEncode,
            VAEDecode,
            EmptyLatentImage,
        )

        print(f"[Predict] Processing image with prompt: {prompt}")

        # 加载输入图像
        input_image = Image.open(str(image)).convert("RGB")
        width, height = input_image.size

        # 确保尺寸是 8 的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        input_image = input_image.resize((width, height), Image.LANCZOS)

        # 转换为 tensor
        image_np = np.array(input_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 编码提示词
        clip_encode = CLIPTextEncode()
        positive_cond = clip_encode.encode(self.clip, prompt)[0]
        negative_cond = clip_encode.encode(self.clip, negative_prompt)[0]

        # 图像编码到 latent
        vae_encode = VAEEncode()
        latent = vae_encode.encode(self.vae, image_tensor)[0]

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
            scheduler="normal",
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            denoise=1.0,
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
