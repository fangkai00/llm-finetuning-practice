# -*- coding: utf-8 -*-
"""
最小化测试数据生成脚本
功能：从原始数据集中抽取少量样本，生成最小测试数据集
用途：验证微调脚本能否跑通（不追求训练效果）
"""

import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录

# ========================================
# 1. Alpaca 最小数据集（5条）
# ========================================
def make_alpaca_mini():
    src = os.path.join(BASE_DIR, "datasets", "alpaca-cleaned", "alpaca_data_cleaned.json")
    dst = os.path.join(BASE_DIR, "alpaca_mini.json")

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取前5条（保持字段一致）
    mini = data[:5]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(mini, f, ensure_ascii=False, indent=2)

    print(f"[Alpaca] 源数据 {len(data)} 条 → 最小集 {len(mini)} 条")
    print(f"[Alpaca] 保存到: {dst}")
    return dst


# ========================================
# 2. GSM8K 最小数据集（5条，parquet格式）
# ========================================
def make_gsm8k_mini():
    src = os.path.join(BASE_DIR, "datasets", "gsm8k", "main", "train-00000-of-00001.parquet")
    dst = os.path.join(BASE_DIR, "gsm8k_mini.parquet")

    df = pd.read_parquet(src)
    # 取前5条
    mini_df = df.head(5)
    mini_df.to_parquet(dst)

    print(f"[GSM8K] 源数据 {len(df)} 条 → 最小集 {len(mini_df)} 条")
    print(f"[GSM8K] 保存到: {dst}")
    return dst


# ========================================
# 3. VL 训练数据（检查现有xlsx，必要时裁剪）
# ========================================
def check_vl_data():
    xlsx = os.path.join(BASE_DIR, "qwen-vl-train.xlsx")
    if not os.path.exists(xlsx):
        print(f"[VL] 警告: {xlsx} 不存在")
        return None

    df = pd.read_excel(xlsx)
    print(f"[VL] 现有数据 {len(df)} 条，字段: {list(df.columns)}")

    # 检查图片是否存在
    img_dir = os.path.join(BASE_DIR, "data", "images")
    if os.path.exists(img_dir):
        imgs = os.listdir(img_dir)
        print(f"[VL] 图片目录 {img_dir} 下有 {len(imgs)} 个文件: {imgs[:5]}")

    # 若数据超过3条，裁剪到前3条（最小集）
    if len(df) > 3:
        mini_xlsx = os.path.join(BASE_DIR, "qwen-vl-train-mini.xlsx")
        df.head(3).to_excel(mini_xlsx, index=False)
        print(f"[VL] 裁剪到前3条，保存到: {mini_xlsx}")
        return mini_xlsx
    else:
        print(f"[VL] 数据量已很小（{len(df)}条），直接使用原文件")
        return xlsx


if __name__ == "__main__":
    print("=" * 60)
    print("生成最小化测试数据")
    print("=" * 60)
    make_alpaca_mini()
    print()
    make_gsm8k_mini()
    print()
    check_vl_data()
    print()
    print("=" * 60)
    print("最小测试数据生成完成!")
    print("接下来用这些数据验证微调脚本能否跑通")
    print("=" * 60)
