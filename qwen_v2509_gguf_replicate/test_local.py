# test_local.py
import os
os.environ["COMFYUI_PATH"] = "/workspace/ComfyUI"

from predict import Predictor

pred = Predictor()
pred.setup()

out = pred.predict(
    prompt="add a cute cat to the image",
    image="input.jpg",   # 或者你的接口要求的路径
)

print(out)
