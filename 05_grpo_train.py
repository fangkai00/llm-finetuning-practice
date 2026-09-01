# -*- coding: utf-8 -*-
"""
Qwen2.5-7B GRPO强化学习训练R1模型
课程：LLM模型蒸馏与微调实操
功能：使用GRPO（Group Relative Policy Optimization）训练Qwen2.5-7B的推理能力
环境：AutoDL GPU实例，建议 A100/A800 40GB+
依赖：pip install unsloth
说明：原版使用vLLM加速推理(fast_inference=True)；Windows环境无vllm，
      已改为fast_inference=False用transformers原生generate，功能等价但稍慢。
"""

# ========================================
# Step 1: 模型加载（transformers原生推理，无需vLLM）
# ========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 在多GPU时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 禁用 torch.compile(inductor)，避免 torch 2.12.1 的 aliasing constraint bug
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Triton缓存目录指向无中文的临时目录，避免权限问题和中文路径Unicode解码bug
import tempfile
_TRITON_CACHE = os.path.join(tempfile.gettempdir(), "triton_cache")
os.makedirs(_TRITON_CACHE, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE
# 彻底禁用 dynamo（torch.compile 前端），双重保险
import torch._dynamo
torch._dynamo.config.disable = True
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import unsloth
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # 可以增加以获得更长的推理轨迹
lora_rank = 32  # 更大的rank让模型更智能，但训练更慢

# 本地路径配置（Windows环境，模型通过 download_model.py 下载到 models/ 目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,  # 不启用vLLM（Windows无vllm），用transformers原生generate
)


# ========================================
# Step 2: LoRA配置
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

# 加载SFT预热的LoRA权重（GRPO的起点，避免冷启动失败）
# 原理：GRPO需要模型能生成正确格式才能获得正奖励→有梯度信号→学习
#       SFT预热让模型先学会<reasoning>/<answer>格式，GRPO从这个起点优化推理能力
sft_warmup_path = os.path.join(BASE_DIR, "sft_warmup_lora")
if os.path.exists(os.path.join(sft_warmup_path, "adapter_model.safetensors")):
    # 用PEFT标准方式加载LoRA权重到已有的LoRA层
    import safetensors.torch
    from peft import set_peft_model_state_dict
    lora_weights = safetensors.torch.load_file(
        os.path.join(sft_warmup_path, "adapter_model.safetensors")
    )
    set_peft_model_state_dict(model, lora_weights)
    print(f"✅ 已加载SFT预热LoRA权重: {sft_warmup_path}")
    print("   GRPO将从SFT预热的起点开始优化推理能力")
else:
    print(f"⚠️ 未找到SFT预热LoRA({sft_warmup_path})")
    print("   建议先运行 sft_warmup_grpo.py 做SFT预热，否则可能冷启动失败")


# ========================================
# Step 3: GSM8K数据准备
# ========================================

import re
from datasets import load_dataset, Dataset

# 系统提示词：定义推理输出格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """从XML格式文本中提取答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """从####标记文本中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    """加载GSM8K数据集"""
    # 本地数据集（Parquet格式，直接指向 main/train-00000-of-00001.parquet）
    GSM8K_DATA_FILE = os.path.join(BASE_DIR, "datasets", "gsm8k", "main", "train-00000-of-00001.parquet")
    data = load_dataset("parquet", data_files=GSM8K_DATA_FILE, split=split)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


dataset = get_gsm8k_questions()


# ========================================
# Step 4: 奖励函数设计
# ========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励：检查答案是否正确（权重最高）"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励：检查答案是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励：完全符合XML格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励：基本符合XML格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """计算XML标签完整性得分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML标签计数奖励"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ========================================
# Step 5: GRPOTrainer训练
# ========================================

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,  # 每个问题生成2个候选答案（原6，本地学习用减为2）
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=20,  # 训练20步（原250，本地学习用减为20，约15-20分钟）
    save_steps=20,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()


# ========================================
# Step 6: 模型测试与保存
# ========================================

# 保存LoRA参数
model.save_lora("grpo_saved_lora")

# 测试模型推理
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

# Windows环境无vllm，改用transformers原生generate推理
# （训练后的model已包含LoRA权重，无需再load_lora）
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output_ids = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
)
output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(output)


# ========================================
# 模型导出选项（按需取消注释）
# ========================================

# 保存为16bit浮点
# model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# 保存为4bit整数
# model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# 仅保存LoRA适配器
# model.save_pretrained_merged("model", tokenizer, save_method="lora")

# 保存为GGUF q4_k_m格式
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
