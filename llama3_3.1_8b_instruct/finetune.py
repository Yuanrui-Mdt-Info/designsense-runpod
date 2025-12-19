from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 配置参数
max_seq_length = 2048 # 根据显存调整
dtype = None # 自动检测
load_in_4bit = True # 4bit 量化

# 2. 加载模型
print("正在加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. 添加 LoRA 适配器
print("正在配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. 准备数据 (使用 Llama 3.1 Chat Template)
print("正在处理 Chat Template...")

# 使用 Unsloth 提供的 Chat Template 工具，自动适配 Llama-3.1
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # 根据数据集格式调整映射
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# 加载示例数据集 (ShareGPT 格式)
# 注意：这里改用了 mlabonne/FineTome-100k 的一部分作为示例，因为它已经是 Chat 格式
# 如果你使用 Alpaca 格式 (Instruction/Input/Output)，你需要将其转换为 Chat 格式
print("正在加载数据...")
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:1%]") # 仅取 1% 用于演示
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. 训练参数配置
print("开始训练...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. 开始训练
trainer_stats = trainer.train()

# 7. 保存模型
print("保存模型...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("完成！")
