import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
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
    image = pipe(
        prompt, 
        num_inference_steps=30, 
        guidance_scale=7.5
    ).images[0]

    image.save(output_path)
    print(f"图片已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cute yoda pokemon", help="提示词")
    parser.add_argument("--lora_path", type=str, default="/workspace/lora_output", help="LoRA 模型路径文件夹或 safetensors 文件")
    parser.add_argument("--output", type=str, default="/workspace/test_result.png", help="输出文件名")
    
    args = parser.parse_args()
    generate_image(args.prompt, args.lora_path, args.output)
