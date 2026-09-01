# -*- coding: utf-8 -*-
"""
最小化测试：Alpaca SFT 脚本能否跑通
功能：用5条数据训练2步，验证 模型加载→LoRA→数据→训练→推理→保存 全流程
说明：此脚本由 Qwen2_5_(7B)_Alpaca.py 改造，仅用于验证流程，不追求效果
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 的 n_gpu>1 时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

from unsloth import FastLanguageModel
import torch

# ========================================
# 路径配置
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct")
ALPACA_MINI = os.path.join(BASE_DIR, "alpaca_mini.json")

print("=" * 60)
print("最小化测试：Alpaca SFT 流程验证")
print("=" * 60)
print(f"模型路径: {MODEL_DIR}")
print(f"模型存在: {os.path.exists(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    files = os.listdir(MODEL_DIR)
    print(f"模型文件: {files}")
print()

# ========================================
# Step 1: 模型加载
# ========================================
print("[1/6] 加载模型...")
max_seq_length = 512  # 缩短以加速
dtype = None
load_in_4bit = True

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("  ✅ 模型加载成功")
except Exception as e:
    print(f"  ❌ 模型加载失败: {e}")
    raise

# ========================================
# Step 2: LoRA配置
# ========================================
print("\n[2/6] 配置LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # 缩小rank加速
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("  ✅ LoRA配置成功")

# ========================================
# Step 3: 数据准备（最小集）
# ========================================
print("\n[3/6] 加载最小数据集...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("json", data_files=ALPACA_MINI, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"  ✅ 数据加载成功: {len(dataset)} 条样本")

# ========================================
# Step 4: 训练（仅2步）
# ========================================
print("\n[4/6] 开始训练（2步）...")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 缩小batch
    gradient_accumulation_steps=1,
    warmup_steps=0,
    max_steps=2,  # 仅2步
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs_test",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    args=training_args,
)

try:
    trainer_stats = trainer.train()
    print(f"  ✅ 训练完成: {trainer_stats.metrics}")
except Exception as e:
    print(f"  ❌ 训练失败: {e}")
    raise

# ========================================
# Step 5: 推理验证
# ========================================
print("\n[5/6] 推理验证...")
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [alpaca_prompt.format("Continue the fibonnaci sequence.", "1, 1, 2, 3, 5, 8", "")],
    return_tensors="pt"
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"  ✅ 推理输出: {response[:100]}...")

# ========================================
# Step 6: 保存模型
# ========================================
print("\n[6/6] 保存模型...")
save_dir = "lora_model_test"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"  ✅ 模型已保存到: {save_dir}")

print("\n" + "=" * 60)
print("✅ Alpaca SFT 全流程验证通过!")
print("=" * 60)
