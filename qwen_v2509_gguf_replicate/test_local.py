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
    negative_prompt="blurry, low quality, distorted", # 必须传！
    steps=28,
    cfg_scale=6.0,
    seed=-1
)

print(out)
