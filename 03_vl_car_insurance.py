# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B 视觉模型微调 - 汽车保险承保专家
课程：LLM模型蒸馏与微调实操
功能：使用Unsloth框架对Qwen2.5-VL-3B进行车辆里程表识别任务的微调
环境：AutoDL GPU实例
"""

# ========================================
# Step 1: 模型加载
# ========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只用1个GPU，避免 transformers 4.57.6 在多GPU时 loss.mean() 报错
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 禁用 torch.compile(inductor)，避免 torch 2.12.1 的 aliasing constraint bug
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# 彻底禁用 dynamo（torch.compile 前端），双重保险
import torch._dynamo
torch._dynamo.config.disable = True
# 在 huggingface_hub 被加载前，直接修改它的常量
import huggingface_hub.constants as _c
_c.DEFAULT_ENDPOINT = "https://hf-mirror.com"
_c.HF_ENDPOINT = "https://hf-mirror.com"
_c.ENDPOINT = "https://hf-mirror.com"

import json
import os
from PIL import Image
from unsloth import FastVisionModel
import torch

# 本地路径配置（Windows环境，模型通过 download_model.py 下载到 models/ 目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-VL-3B-Instruct")

print("正在加载Qwen2.5-VL-3B模型...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_DIR,
    use_gradient_checkpointing="unsloth",
)

print("配置LoRA参数...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # 微调视觉层
    finetune_language_layers=True,    # 微调语言层
    finetune_attention_modules=True,  # 微调注意力模块
    finetune_mlp_modules=True,        # 微调MLP模块
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ========================================
# Step 2: 数据准备（Excel格式）
# ========================================

import pandas as pd

print("加载训练数据...")

def load_excel_dataset(file_path):
    """加载Excel格式的数据集"""
    try:
        df = pd.read_excel(file_path)
        print(f"Excel文件列名: {list(df.columns)}")
        print(f"数据集形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None


def convert_excel_to_training_format(df):
    """将Excel格式转换为训练格式"""
    converted_data = []
    
    for idx, row in df.iterrows():
        image_path = row["image"]
        prompt = row["prompt"]
        response = row["response"]
        
        image_path = os.path.join(BASE_DIR, "data", str(image_path))
        if pd.notna(row["image"]) and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": image}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response}
                            ]
                        }
                    ]
                }
                converted_data.append(conversation)
                print(f"成功处理样本 {idx + 1}: {image_path}")
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
        else:
            print(f"警告：图片文件不存在或路径为空 {image_path}")
    
    return converted_data


train_df = load_excel_dataset(os.path.join(BASE_DIR, "data", "qwen-vl-train.xlsx"))
if train_df is not None:
    converted_dataset = convert_excel_to_training_format(train_df)
else:
    print("无法加载数据集，程序退出")
    exit()

print(f"成功加载 {len(converted_dataset)} 个训练样本")


# ========================================
# Step 3: 训练前推理测试
# ========================================

print("\n训练前模型推理测试...")
FastVisionModel.for_inference(model)

test_image = Image.open(os.path.join(BASE_DIR, "data", "images", "1-vehicle-odometer-reading.jpg")).convert('RGB')
test_instruction = "你是一名汽车保险承保专家。这里有一张车辆里程表的图片。请从中提取关键信息。"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": test_instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("训练前模型输出:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)


# ========================================
# Step 4: 模型训练
# ========================================

print("\n开始训练模型...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
    max_seq_length=2048,
)

# 显存信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. 最大显存 = {max_memory} GB.")
print(f"{start_gpu_memory} GB 显存已使用.")

# 执行训练
trainer_stats = trainer.train()

# 训练统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)

print(f"训练用时: {trainer_stats.metrics['train_runtime']} 秒")
print(f"训练用时: {round(trainer_stats.metrics['train_runtime']/60, 2)} 分钟")
print(f"峰值显存使用: {used_memory} GB")
print(f"LoRA训练显存使用: {used_memory_for_lora} GB")
print(f"显存使用率: {used_percentage}%")


# ========================================
# Step 5: 训练后推理测试
# ========================================

print("\n训练后模型推理测试...")
FastVisionModel.for_inference(model)

inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("训练后模型输出:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)


# ========================================
# Step 6: 保存模型
# ========================================

print("\n保存LoRA适配器...")
model.save_pretrained("car_insurance_lora_model")
tokenizer.save_pretrained("car_insurance_lora_model")
print("训练完成! 模型已保存到 car_insurance_lora_model 目录")
