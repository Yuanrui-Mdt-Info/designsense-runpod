#! /usr/bin/env python
# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
import torch

# 1. 加载微调后的模型
# 注意：这里直接指向保存的 'lora_model' 目录，Unsloth 会自动加载底座模型并挂载 LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 开启推理模式（这一步非常重要，比 model.eval() 更快）
FastLanguageModel.for_inference(model)

# 2. 准备测试 Prompt
# 必须使用与训练时完全一致的 Chat Template！
messages = [
    {"role": "user", "content": "你好，请帮我把这句话翻译成英文：'今天天气真好，我想去公园散步。'"},
]

# 应用 Chat Template
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # 必须为 True，这会添加 assistant 的起始符
    return_tensors = "pt",
).to("cuda")

# 3. 生成回复
outputs = model.generate(
    input_ids = input_ids,
    max_new_tokens = 128,      # 最大生成长度
    use_cache = True,          # 推理时开启 KV Cache 加速
    temperature = 0.7,         # 创造性 (0.1~1.0)
    top_p = 0.9,
)

# 4. 解码并打印
# 只打印生成的部分（去掉输入的 prompt）
response = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
print("================================")
print(f"用户输入: {messages[0]['content']}")
print(f"模型回复: {response}")
print("================================")