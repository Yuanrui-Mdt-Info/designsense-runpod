# test_local.py
import os
os.environ["COMFYUI_PATH"] = "/workspace/ComfyUI"

from predict import Predictor

pred = Predictor()
pred.setup()

# test_local.py
out = pred.predict(
    image="input.jpg",
    prompt="add a cute cat to the bed",
    negative_prompt="blurry, low quality, distorted",
    steps=4,  # Lightning LoRA 快速模式
    cfg_scale=1.0,
    seed=-1
)

print(out)
