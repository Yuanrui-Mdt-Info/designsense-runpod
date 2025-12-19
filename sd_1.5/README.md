# Stable Diffusion 1.5 LoRA 微调 (RunPod)

本项目用于在 RunPod 上对 `stable-diffusion-v1-5` 进行 LoRA 微调。

## 快速开始

### 1. 安装与准备
```bash
chmod +x train.sh
./train.sh
```
该脚本会自动：
1. 安装 Python 依赖。
2. 下载 Hugging Face 官方的 `train_text_to_image_lora.py` 脚本。
3. 下载示例数据集（Pokemon）并开始训练。

### 2. 使用自己的数据集

1. 准备一个文件夹，包含图片和对应的文本描述。
2. 推荐使用 Hugging Face 的 `datasets` 库格式，或者简单的 `metadata.jsonl` 格式：
   ```json
   {"file_name": "image_01.jpg", "text": "a photo of a cat wearing a hat"}
   {"file_name": "image_02.jpg", "text": "a drawing of a robot"}
   ```
3. 修改 `train.sh` 中的 `DATASET_NAME` 指向你的本地文件夹路径：
   ```bash
   export DATASET_NAME="./my_dataset_folder"
   ```

### 3. 推理测试

训练完成后，使用 `inference.py` 生成图片：

```bash
python inference.py \
  --prompt "A red dragon pokemon" \
  --lora_path "lora_output" \
  --output "dragon.png"
```

## 文件结构

- `train.sh`: 一键启动训练的脚本。
- `inference.py`: 加载 LoRA 并生成图片的 Python 脚本。
- `requirements.txt`: 依赖列表。
