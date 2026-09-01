# -*- coding: utf-8 -*-
"""
最小化测试：Qwen2.5-VL 视觉微调脚本能否跑通
功能：用2条数据训练2步，验证 视觉模型加载→数据→推理→训练→保存 全流程
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 的 n_gpu>1 时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import json
from PIL import Image
from unsloth import FastVisionModel
import torch
import pandas as pd

# ========================================
# 路径配置
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-VL-3B-Instruct")

print("=" * 60)
print("最小化测试：Qwen2.5-VL 视觉微调流程验证")
print("=" * 60)
print(f"模型路径: {MODEL_DIR}")
print(f"模型存在: {os.path.exists(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    files = os.listdir(MODEL_DIR)
    print(f"模型文件数: {len(files)}")
print()

# ========================================
# Step 1: 模型加载
# ========================================
print("[1/6] 加载Qwen2.5-VL-3B模型...")
try:
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_DIR,
        use_gradient_checkpointing="unsloth",
    )
    print("  ✅ 视觉模型加载成功")
except Exception as e:
    print(f"  ❌ 视觉模型加载失败: {e}")
    raise

# ========================================
# Step 2: LoRA配置
# ========================================
print("\n[2/6] 配置LoRA...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,  # 缩小rank
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("  ✅ LoRA配置成功")

# ========================================
# Step 3: 数据准备
# ========================================
print("\n[3/6] 加载训练数据...")
df = pd.read_excel(os.path.join(BASE_DIR, "data", "qwen-vl-train.xlsx"))
print(f"  数据条数: {len(df)}")

converted_data = []
for idx, row in df.iterrows():
    image_path = row["image"]
    # 兼容相对路径
    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, "data", image_path)
    prompt = row["prompt"]
    response = row["response"]

    if pd.notna(image_path) and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        conversation = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": response}
                ]}
            ]
        }
        converted_data.append(conversation)
        print(f"  ✅ 样本{idx+1}: {image_path}")
    else:
        print(f"  ⚠️ 样本{idx+1} 图片不存在: {image_path}")

print(f"  有效样本: {len(converted_data)}")

# ========================================
# Step 4: 训练前推理
# ========================================
print("\n[4/6] 训练前推理测试...")
FastVisionModel.for_inference(model)
test_image_path = os.path.join(BASE_DIR, "data", "images", "1-vehicle-odometer-reading.jpg")
test_image = Image.open(test_image_path).convert('RGB')
test_instruction = "你是一名汽车保险承保专家。这里有一张车辆里程表的图片。请从中提取关键信息。"
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": test_instruction}
]}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(test_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

try:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, use_cache=True, temperature=0.1)
    pre_response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  ✅ 训练前输出(截断): {pre_response[:80]}...")
except Exception as e:
    print(f"  ⚠️ 训练前推理异常(非致命): {e}")

# ========================================
# Step 5: 训练（仅2步）
# ========================================
print("\n[5/6] 开始训练（2步）...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_data,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=2,  # 仅2步
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_vl_test",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
    max_seq_length=512,  # 缩短
)

try:
    trainer_stats = trainer.train()
    print(f"  ✅ 训练完成: {trainer_stats.metrics}")
except Exception as e:
    print(f"  ❌ 训练失败: {e}")
    raise

# ========================================
# Step 6: 保存
# ========================================
print("\n[6/6] 保存模型...")
save_dir = "vl_model_test"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"  ✅ 模型已保存到: {save_dir}")

print("\n" + "=" * 60)
print("✅ Qwen2.5-VL 视觉微调全流程验证通过!")
print("=" * 60)
