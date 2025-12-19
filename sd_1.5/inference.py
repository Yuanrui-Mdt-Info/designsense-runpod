#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler, LCMScheduler
from PIL import Image
import argparse


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

    """
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if lora_path:
        print(f"正在加载 LoRA: {lora_path}...")
        pipe.load_lora_weights(lora_path)
    """
    
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
    
    # if lora_path:
    #     print(f"正在加载 LoRA: {lora_path}...")
    #     pipe.load_lora_weights(lora_path, adapter_name="pokemon")
    #     pipe.set_adapters(["lcm", "pokemon"], adapter_weights=[1.0, 0.8])
    
    print(f"正在生成: '{prompt}'")
    
    generator = torch.Generator(device="cuda").manual_seed(1337)
    
    """
    LCM 对 guidance_scale 非常敏感，通常锁死在 1.0 到 2.0 之间。
    这意味着你不能像以前那样通过调高 CFG (如 7-15) 来强行让 AI "听从" 复杂的 Prompt。
    如果你的 Prompt 非常复杂且必须严格遵循，LCM 可能幻觉严重。    
    
    """
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cute yoda pokemon", help="提示词")
    parser.add_argument("--image_path", type=str, default="/workspace/init_image.png", help="初始图片路径")
    parser.add_argument("--lora_path", type=str, default="/workspace/lora_output", help="LoRA 模型路径文件夹或 safetensors 文件")
    parser.add_argument("--output", type=str, default="/workspace/test_result.png", help="输出文件名")
    
    args = parser.parse_args()
    # generate_image(args.prompt, args.lora_path, args.output)
    generate_image_img2img(args.prompt, args.image_path, args.lora_path, args.output)
