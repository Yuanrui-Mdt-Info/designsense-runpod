# Llama 3.1 8B Instruct Fine-tuning on RunPod

本项目用于在 RunPod 上对 `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` 进行微调。使用 Unsloth 框架可以显著降低显存需求并提高训练速度。

## 环境要求

建议使用 RunPod 的 PyTorch 模板（推荐 CUDA 12.1+）。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行微调

```bash
python finetune.py
```

## 文件说明

- `finetune.py`: 主训练脚本。默认配置为使用 ShareGPT 格式的数据集（如 `mlabonne/FineTome-100k`），并应用 Llama 3.1 的 Chat Template。
- `requirements.txt`: Python 依赖列表。

## 配置说明

在 `finetune.py` 中，你可以修改以下参数：
- `max_seq_length`: 上下文长度，默认 2048。
- `load_in_4bit`: 是否使用 4bit 量化（建议为 True 以节省显存）。
- **数据集格式**：脚本默认处理 ShareGPT 格式（`conversations` 列表包含 `from` 和 `value`）。如果你的数据是 Alpaca 格式，请相应调整 `formatting_prompts_func`。

## 模型保存

训练完成后，LoRA 适配器将保存在 `lora_model` 目录下。
