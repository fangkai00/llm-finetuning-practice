# -*- coding: utf-8 -*-
"""
模型下载脚本
课程：LLM模型蒸馏与微调实操
功能：下载训练所需的 Qwen 模型到本地目录
说明：
    - Qwen2.5-7B-Instruct      → 用于 Alpaca SFT 微调、GRPO 强化学习
    - Qwen2.5-VL-3B-Instruct   → 用于视觉模型（汽车保险）微调
    - 使用 HuggingFace + hf-mirror 镜像（国内稳定）
    - 启用 hf_transfer 加速下载（需先 pip install hf_transfer）
"""

import os
import sys

# ========================================
# 配置区域（根据环境修改）
# ========================================

# 使用国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 启用 hf_transfer 多连接加速（若已安装）
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 训练所需的模型列表（HuggingFace 仓库 ID）
MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen2.5-7B-Instruct",      # 7B 语言模型，用于 Alpaca SFT 与 GRPO
    "Qwen/Qwen2.5-VL-3B-Instruct",   # 3B 视觉语言模型，用于 VL 微调
]

# AutoDL环境
AUTODL_CACHE_DIR = "/root/autodl-tmp/models"

# 本地Windows环境（下载到当前脚本所在目录下的 models 文件夹）
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def detect_environment():
    """检测当前运行环境"""
    if sys.platform == "win32":
        return "windows", LOCAL_CACHE_DIR
    elif os.path.exists("/root/autodl-tmp"):
        return "autodl", AUTODL_CACHE_DIR
    else:
        return "linux", LOCAL_CACHE_DIR


def download_with_huggingface(model_id, cache_dir):
    """使用HuggingFace（走hf-mirror镜像）下载模型"""
    from huggingface_hub import snapshot_download

    print(f"使用HuggingFace下载模型: {model_id}")
    print(f"下载目录: {cache_dir}")

    os.makedirs(cache_dir, exist_ok=True)
    # local_dir 指定后，模型直接落到 cache_dir/<组织>/<仓库名>/ 下
    local_dir = os.path.join(cache_dir, *model_id.split("/"))
    # max_workers 并行下载多个文件
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_workers=8,
    )

    print(f"模型下载完成: {model_dir}")
    return model_dir


def download_one(model_id, cache_dir):
    """下载单个模型：HuggingFace（走hf-mirror镜像）"""
    print("\n" + "=" * 60)
    print(f"开始下载: {model_id}")
    print(f"镜像: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"hf_transfer: {'启用' if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1' else '未启用'}")
    print("=" * 60)
    try:
        return download_with_huggingface(model_id, cache_dir)
    except ImportError:
        print("huggingface_hub 未安装")
        print("请先安装: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def main():
    env_name, cache_dir = detect_environment()
    print(f"检测到环境: {env_name}")
    print(f"下载根目录: {cache_dir}")
    print(f"待下载模型: {MODELS_TO_DOWNLOAD}")
    print()

    downloaded = {}
    for model_id in MODELS_TO_DOWNLOAD:
        model_dir = download_one(model_id, cache_dir)
        if model_dir:
            downloaded[model_id] = model_dir

    print()
    print("=" * 60)
    print("下载完成!")
    print("=" * 60)
    for model_id, model_dir in downloaded.items():
        print(f"\n模型: {model_id}")
        print(f"路径: {model_dir}")
        print(f"在微调脚本中使用此路径:")
        print(f'  model_name = r"{model_dir}"')

    if len(downloaded) != len(MODELS_TO_DOWNLOAD):
        missing = set(MODELS_TO_DOWNLOAD) - set(downloaded.keys())
        print(f"\n警告: 以下模型未下载成功: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
