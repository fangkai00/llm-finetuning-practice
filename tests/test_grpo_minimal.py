# -*- coding: utf-8 -*-
"""
最小化测试：GRPO 强化学习脚本能否跑通
功能：用5条GSM8K数据训练2步，验证 模型加载→LoRA→数据→GRPO训练→保存→推理 全流程
说明：不依赖vllm，使用transformers原生generate
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 的 n_gpu>1 时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import re
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset

# ========================================
# 路径配置
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct")
GSM8K_MINI = os.path.join(BASE_DIR, "gsm8k_mini.parquet")

print("=" * 60)
print("最小化测试：GRPO 强化学习流程验证")
print("=" * 60)
print(f"模型路径: {MODEL_DIR}")
print(f"模型存在: {os.path.exists(MODEL_DIR)}")
print()

# ========================================
# Step 1: 模型加载（不启用vllm）
# ========================================
print("[1/5] 加载模型（fast_inference=False，不用vllm）...")
max_seq_length = 512  # 缩短
lora_rank = 8  # 缩小

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # 不用vllm
    )
    print("  ✅ 模型加载成功")
except Exception as e:
    print(f"  ❌ 模型加载失败: {e}")
    raise

# ========================================
# Step 2: LoRA配置
# ========================================
print("\n[2/5] 配置LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("  ✅ LoRA配置成功")

# ========================================
# Step 3: 数据准备（5条）
# ========================================
print("\n[3/5] 加载最小GSM8K数据集...")
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

data = load_dataset("parquet", data_files=GSM8K_MINI, split="train")
data = data.map(lambda x: {
    'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x['question']}
    ],
    'answer': extract_hash_answer(x['answer'])
})
print(f"  ✅ 数据加载成功: {len(data)} 条样本")

# ========================================
# 奖励函数（与原脚本一致，精简版）
# ========================================
def extract_xml_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

# ========================================
# Step 4: GRPO训练（仅2步）
# ========================================
print("\n[4/5] 开始GRPO训练（2步）...")
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,  # 缩小：2个候选答案（原6）
    max_prompt_length=128,  # 缩短
    max_completion_length=max_seq_length - 128,
    max_steps=2,  # 仅2步
    save_steps=2,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs_grpo_test",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=data,
)

try:
    trainer_stats = trainer.train()
    print(f"  ✅ GRPO训练完成: {trainer_stats.metrics}")
except Exception as e:
    print(f"  ❌ GRPO训练失败: {e}")
    raise

# ========================================
# Step 5: 保存 + 推理验证（用transformers generate，不用vllm）
# ========================================
print("\n[5/5] 保存模型 + 推理验证...")
model.save_lora("grpo_saved_lora_test")
print("  ✅ LoRA已保存到 grpo_saved_lora_test")

text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate 1+1."},
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,  # 缩短
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
    )
output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"  ✅ 推理输出: {output[:100]}...")

print("\n" + "=" * 60)
print("✅ GRPO 强化学习全流程验证通过!")
print("=" * 60)
