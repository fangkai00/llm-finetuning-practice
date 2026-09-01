# -*- coding: utf-8 -*-
"""
SFT预热脚本：为GRPO训练准备基础模型
课程：LLM模型蒸馏与微调实操

功能：用GSM8K数学题做SFT，让模型学会 <reasoning>...</reasoning><answer>...</answer> 格式
这是GRPO成功的关键——先让模型有基础能力（能生成正确格式），再用强化学习优化

原理：
  - GRPO需要模型偶尔生成正确格式才能获得正奖励→有梯度信号→学习
  - 如果模型初始输出全是乱码（奖励=0）→没有梯度→不学习（冷启动失败）
  - SFT预热让模型先学会格式，GRPO从预热后的起点开始优化推理能力

训练流程：
  Step 1: 加载Qwen2.5-7B-Instruct（和GRPO用同一个基础模型）
  Step 2: LoRA配置（和GRPO一致，r=32）
  Step 3: 构造SFT数据（GSM8K → <reasoning>/<answer>格式）
  Step 4: SFT训练（50步，约5分钟）
  Step 5: 保存LoRA到 sft_warmup_lora/（GRPO会加载这个权重）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 在多GPU时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 禁用 torch.compile(inductor)，避免 torch 2.12.1 的 aliasing constraint bug
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Triton缓存目录指向无中文的临时目录
import tempfile
_TRITON_CACHE = os.path.join(tempfile.gettempdir(), "triton_cache")
os.makedirs(_TRITON_CACHE, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE
# 彻底禁用 dynamo（torch.compile 前端）
import torch._dynamo
torch._dynamo.config.disable = True
# 修改 huggingface_hub 端点
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# ========================================
# Step 1: 加载模型（和GRPO用同一个基础模型）
# ========================================
max_seq_length = 1024
lora_rank = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
)


# ========================================
# Step 2: LoRA配置（和GRPO一致）
# ========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ========================================
# Step 3: 构造SFT数据（GSM8K → <reasoning>/<answer>格式）
# ========================================

# 系统提示词（必须和GRPO脚本完全一致）
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML格式模板（必须严格匹配GRPO的strict_format_reward_func正则）
# 正则: ^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def format_sft_example(example):
    """
    将GSM8K数据转换为SFT训练样本

    GSM8K原始格式:
      question: "A concert ticket costs $40..."
      answer: "Mr. Benson bought 12 tickets...discount...$476.\n#### 476"

    转换后:
      prompt: <system>Respond in the following format...</system>
              <user>A concert ticket costs $40...</user>
              <assistant>
      target: <reasoning>\nMr. Benson bought 12 tickets...\n</reasoning>\n<answer>\n476\n</answer>\n
    """
    full_answer = example['answer']

    # 从GSM8K答案中分离推理过程和最终答案
    # 格式: "推理过程\n#### 答案"
    if "####" not in full_answer:
        # 跳过没有####标记的样本
        return {"text": ""}

    # 用最后一个####分割（推理过程中可能有####，虽然罕见）
    parts = full_answer.rsplit("####", 1)
    reasoning = parts[0].strip()
    answer = parts[1].strip()

    # 构造target（严格匹配GRPO的奖励函数期望格式）
    target = XML_COT_FORMAT.format(reasoning=reasoning, answer=answer)

    # 构造prompt（system + user + assistant开头）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['question']},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # SFT训练数据 = prompt + target
    return {"text": prompt + target}


# 加载GSM8K数据集（本地Parquet格式）
GSM8K_DATA_FILE = os.path.join(BASE_DIR, "datasets", "gsm8k", "main", "train-00000-of-00001.parquet")
print(f"加载GSM8K数据: {GSM8K_DATA_FILE}")
dataset = load_dataset("parquet", data_files=GSM8K_DATA_FILE, split="train")

# 格式化为SFT数据
dataset = dataset.map(format_sft_example, remove_columns=dataset.column_names)
# 过滤掉空文本（没有####标记的样本）
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

print(f"SFT训练样本数: {len(dataset)}")
print(f"\n=== 示例样本 ===")
print(dataset[0]["text"][:500])
print("...")
print(dataset[0]["text"][-200:])

# 只取前1000条（SFT预热不需要全部数据，1000条足够学会格式）
dataset = dataset.select(range(min(1000, len(dataset))))
print(f"\n使用前 {len(dataset)} 条做SFT预热")


# ========================================
# Step 4: SFT训练（50步，约5分钟）
# ========================================
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="sft_warmup_outputs",
        max_steps=50,  # 预热50步（足够学会格式）
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # SFT用较高学习率（比GRPO的5e-6高很多）
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_steps=50,
        warmup_steps=5,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
    ),
)

print("\n" + "=" * 50)
print("开始SFT预热训练（50步）")
print("=" * 50)
trainer.train()


# ========================================
# Step 5: 保存LoRA到 sft_warmup_lora/（GRPO会加载这个权重）
# ========================================
print("\n" + "=" * 50)
print("SFT预热完成！保存LoRA权重...")
print("=" * 50)

model.save_lora("sft_warmup_lora")
print("✅ LoRA已保存到 sft_warmup_lora/")
print("\n接下来运行 GRPO 脚本（Qwen2_5_(7B)_R1_GRPO.py），")
print("它会自动加载这个SFT预热的LoRA权重作为GRPO的起点")


# ========================================
# Step 6: 快速验证SFT效果
# ========================================
print("\n" + "=" * 50)
print("验证SFT预热效果（测试1道数学题）")
print("=" * 50)

FastLanguageModel.for_inference(model)

test_question = "Janet's duck lays 16 eggs per month. She eats 3 and sells the rest for $2 each. How much does she earn per month?"
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": test_question},
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
output_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)
output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(f"问题: {test_question}")
print(f"\nSFT预热后输出:")
print(output)
print("\n✅ 如果输出包含 <reasoning> 和 <answer> 标签，说明SFT预热成功！")
print("   接下来可以运行 GRPO 脚本做强化学习优化")
