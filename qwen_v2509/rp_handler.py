import runpod
# from diffusers import DiffusionPipeline
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image


# Load model on startup - 使用 bfloat16 更稳定
# pipe = DiffusionPipeline.from_pretrained(
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16,  # bfloat16 比 float16 更稳定
    trust_remote_code=True,
).to("cuda")


def handler(event):
    """
    Runpod handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        image_base64 = input_data.get("image_base64")

        # 支持 URL 或 Base64 输入
        if image_url:
            input_image = load_image(image_url)
        elif image_base64:
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            return {"error": "Missing 'image_url' or 'image_base64' parameter."}

        output_image = pipe(image=input_image, prompt=prompt).images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
