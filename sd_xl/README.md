
```bash
# 标准版（约 5GB）
huggingface-cli download xinsir/controlnet-union-sdxl-1.0 \
    diffusion_pytorch_model.safetensors \
    --local-dir /workspace/runpod-slim/ComfyUI/models/controlnet \
    --local-dir-use-symlinks False

# ProMax 版（支持 Tile/Inpaint/Outpaint）
huggingface-cli download xinsir/controlnet-union-sdxl-1.0 \
    diffusion_pytorch_model_promax.safetensors \
    --local-dir /workspace/runpod-slim/ComfyUI/models/controlnet \
    --local-dir-use-symlinks False
```
