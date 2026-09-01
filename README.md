# LLM 模型微调（SFT · VLM · GRPO）

![GitHub stars](https://img.shields.io/github/stars/fangkai00/llm-finetuning-practice?style=social)
![GitHub forks](https://img.shields.io/github/forks/fangkai00/llm-finetuning-practice?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/fangkai00/llm-finetuning-practice?style=social)
![GitHub issues](https://img.shields.io/github/issues/fangkai00/llm-finetuning-practice)
![GitHub License](https://img.shields.io/github/license/fangkai00/llm-finetuning-practice)
![Python](https://img.shields.io/badge/Python-3.12-blue)

基于 Qwen2.5 系列大模型，完整实践三种主流微调范式：**SFT 监督微调**、**VLM 视觉多模态微调**、**GRPO 强化学习**（含 DeepSeek-R1 式 SFT 预热两阶段训练）。

## 项目亮点

- **三种微调范式完整落地**：SFT（Alpaca 指令微调）→ VLM（里程表图片信息提取）→ GRPO（GSM8K 数学推理强化）
- **复现 DeepSeek-R1 两阶段训练**：SFT 预热 + GRPO，实测解决 GRPO 冷启动失败问题（无预热 loss=0 零学习，有预热 reward 0→3.0）
- **视觉微调效果显著**：30 步训练（37 秒 / 0.5GB 显存）让模型从读数错误修正到精准识别里程表全部信息
- **工程问题全记录**：解决 torch 2.12 编译 bug、Triton 中文路径、多 GPU 兼容、vLLM 依赖移除等 Windows 环境问题

## 训练结果

| 范式 | 脚本 | 核心指标 |
|------|------|---------|
| SFT 监督微调 | `02_sft_alpaca.py` | loss 2.0→0.18（60 步） |
| VLM 视觉微调 | `03_vl_car_insurance.py` | 里程数 529891（错）→ 528,915（精准），loss 2.38→0.018 |
| GRPO 强化学习 | `04_sft_warmup.py` + `05_grpo_train.py` | reward 0→3.0，correctness 0→2.0 |

## 目录结构

```
├── 01_download_models.py      # 模型下载（Qwen2.5-7B + Qwen2.5-VL-3B，走 hf-mirror）
├── 02_sft_alpaca.py           # SFT 监督微调（Alpaca 指令数据）
├── 03_vl_car_insurance.py     # VLM 视觉微调（汽车保险里程表识别）
├── 04_sft_warmup.py           # GRPO 前置：SFT 预热（学会 reasoning/answer 格式）
├── 05_grpo_train.py           # GRPO 强化学习（加载 SFT 预热权重）
├── 06_model_eval.py           # 模型对比评估工具
├── tests/                     # 最小化验证脚本（快速跑通全流程，max_steps=2）
├── data/                      # VLM 训练数据（Excel + 里程表图片）
├── datasets/                  # Alpaca / GSM8K 数据集
└── docs/                      # 项目技术文档（含原理讲解 + 面试问答）
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.12 + CUDA 环境
pip install -r requirements.txt

# 模型下载（约 22GB，国内走 hf-mirror 镜像）
python 01_download_models.py
```

### 2. 按顺序训练（三种范式，由浅入深）

```bash
# ① SFT 监督微调（~10 分钟）
python 02_sft_alpaca.py

# ② VLM 视觉微调（~1 分钟）
python 03_vl_car_insurance.py

# ③ GRPO：先预热再强化（关键两步，缺一不可）
python 04_sft_warmup.py    # SFT 预热（~2 分钟）→ 产出 sft_warmup_lora/
python 05_grpo_train.py    # GRPO 训练（~15 分钟，自动加载预热权重）
```

### 3. 快速验证（不想等完整训练时）

```bash
python tests/prepare_minimal_testdata.py   # 生成 5 条样本的最小数据集
python tests/test_alpaca_minimal.py        # 2 步训练验证 SFT 全流程
python tests/test_vl_minimal.py            # 验证 VLM 全流程
python tests/test_grpo_minimal.py          # 验证 GRPO 全流程
```

## 三种微调范式说明

### ① SFT 监督微调（`02_sft_alpaca.py`）
给标准答案直接模仿（监督学习）。用 Alpaca 5 万条指令数据，LoRA r=16，只训练 0.26% 参数。

### ② VLM 视觉微调（`03_vl_car_insurance.py`）
多模态微调，数据为 messages 格式（text + image 混合 content）。LoRA 需分别配置视觉层/语言层/注意力/MLP。脚本内含**训练前 vs 训练后推理对比**，直观展示微调效果。

### ③ GRPO 强化学习（`04` + `05`，核心亮点）
**为什么需要两步？** 直接跑 GRPO 会遇到**冷启动失败**：
```
模型输出乱码 → 奖励全 0 → 无梯度信号 → 训练零学习
```
**SFT 预热**（`04_sft_warmup.py`）先用 GSM8K 数据教会模型 `<reasoning>/<answer>` 输出格式，`05_grpo_train.py` 通过 `set_peft_model_state_dict`（PEFT 原生 API）把预热权重注入 LoRA 层，GRPO 从有效起点开始优化推理正确性。这是 DeepSeek-R1 论文的标准训练流程。

## 技术栈

| 层 | 工具 | 说明 |
|----|------|------|
| 加速层 | Unsloth | HF 生态之上的加速外壳（非替代品） |
| 训练层 | TRL 0.24 | SFTTrainer / GRPOTrainer |
| LoRA 层 | PEFT | LoRA 矩阵创建/保存/加载的真正实现 |
| 模型层 | Transformers 4.57 | Qwen2.5 模型结构 |
| 底层 | PyTorch 2.12 (CUDA) | - |

> 关于 Unsloth 与 PEFT 的关系详见 `docs/项目技术文档_LLM微调实操.md`

## 硬件要求

- GPU 显存 ≥ 16GB（本项目在 RTX A6000 48GB 上训练，7B 4bit 量化后峰值 < 5GB）
- 磁盘 ≥ 50GB（两个基础模型约 22GB + LoRA 产物）

## 产物说明（不随仓库分发）

训练产物 LoRA 权重（每个 150-170MB，超出 GitHub 单文件 100MB 限制）不包含在仓库中，训练后自动生成：

| 产物 | 来源脚本 |
|------|---------|
| `lora_model/` | 02_sft_alpaca.py |
| `car_insurance_lora_model/` | 03_vl_car_insurance.py |
| `sft_warmup_lora/` | 04_sft_warmup.py（05 的前置依赖） |
| `grpo_saved_lora/` | 05_grpo_train.py |

基础模型（`models/Qwen/`）由 `01_download_models.py` 下载，不入库。

## License

本项目采用 [MIT License](LICENSE) 开源协议。
