# SD 1.5 Replicate Deployment (LCM + Img2Img)

This directory contains the Cog configuration and inference code to deploy a highly optimized Stable Diffusion 1.5 Image-to-Image pipeline on [Replicate](https://replicate.com/).

It features **Real-time Latent Consistency Model (LCM)** acceleration, allowing for high-quality image generation in just **4-8 steps**.

## Features

*   **Base Model**: [SG161222/Realistic_Vision_V6.0_B1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE) (Best-in-class photorealism for SD 1.5).
*   **Acceleration**: [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) (Inference in <1s on A40).
*   **Pipeline**: Image-to-Image (Img2Img).
*   **Optimization**: Automatic smart resizing (Max 768px, multiple of 8) to prevent OOM and composition issues.

## Prerequisites

*   [Docker](https://www.docker.com/)
*   [Cog](https://github.com/replicate/cog) (`brew install cog`)

## Usage

### 1. Build and Test Locally

First, download the model weights (this happens automatically on first run) and run a prediction:

```bash
# You need an input image named 'input.jpg' in the current directory
cog predict -i image=@input.jpg \
    -i prompt="modern interior design, minimalist, 8k, photorealistic" \
    -i strength=0.75
```

### 2. Push to Replicate

Create a model on Replicate (e.g., `your-username/sd-1.5-lcm-img2img`), then push:

```bash
cog login
cog push r8.im/your-username/sd-1.5-lcm-img2img
```

## API Inputs

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `file` | Required | Input image for img2img. |
| `prompt` | `string` | "modern..." | Positive prompt. |
| `strength` | `float` | `0.75` | How much to transform the image (0.0=original, 1.0=full change). |
| `guidance_scale` | `float` | `1.5` | CFG Scale. **Keep low (1.0-2.0) for LCM**. |
| `num_inference_steps` | `integer` | `6` | Number of steps. **LCM needs only 4-8**. |
| `seed` | `integer` | Random | Set for reproducible results. |

## Notes for LCM

*   **CFG Scale**: Unlike standard SD (which uses CFG 7.0+), LCM requires a very low CFG scale (typically 1.0 to 2.0). Setting it higher will "burn" the image.
*   **Steps**: Do not set steps > 10. It won't improve quality significantly and wastes compute.
