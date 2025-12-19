#!/bin/bash

# 1. 确保已安装依赖
pip install -r requirements.txt

# === 修改点 1: 设置 Hugging Face 缓存路径到 /workspace，避免下载到临时目录 ===
export HF_HOME="/workspace/huggingface"

# 2. 自动下载 diffusers 官方训练脚本 (如果不存在)
if [ ! -f "train_text_to_image_lora.py" ]; then
    echo "正在下载官方训练脚本..."
    wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py
fi

# 3. 设置环境变量
export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export OUTPUT_DIR="/workspace/lora_output"
export DATASET_NAME="lambdalabs/pokemon-blip-captions" # 示例数据集，请替换为您自己的本地路径或 HF 数据集

# 4. 启动训练
# --mixed_precision="fp16": 使用混合精度加速
# --use_8bit_adam: 使用 8bit 优化器节省显存
# --resolution=512: SD 1.5 标准分辨率
# --train_batch_size=1: 显存不足时设为 1
echo "开始训练..."
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --validation_prompt="A cute yoda pokemon" \
  --seed=1337 \
  --use_8bit_adam

echo "训练完成！结果保存在 $OUTPUT_DIR"
