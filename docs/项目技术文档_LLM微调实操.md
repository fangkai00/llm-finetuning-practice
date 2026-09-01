# LLM 模型蒸馏与微调实操 — 项目技术文档

> 本文档记录了在 Qwen2.5 系列大语言模型上完成三种微调范式的完整实践，涵盖 SFT 监督微调、VLM 视觉多模态微调、GRPO 强化学习，可作为项目简历的技术参考。

---

## 一、项目概述

### 1.1 项目目标
基于 Qwen2.5-7B-Instruct 和 Qwen2.5-VL-3B-Instruct 模型，实践大语言模型微调的三种主流范式：
1. **SFT（Supervised Fine-Tuning）监督微调** — 用 Alpaca 指令数据微调
2. **VLM（Vision-Language Model）视觉多模态微调** — 用汽车保险里程表图片微调
3. **GRPO（Group Relative Policy Optimization）强化学习** — 用 GSM8K 数学题训练推理能力

### 1.2 技术栈

| 类别 | 技术/工具 | 版本 |
|------|----------|------|
| 基础框架 | PyTorch (CUDA) | 2.12.1+cu130 |
| 微调框架 | Unsloth | 2026.8.19 |
| 模型库 | Transformers | 4.57.6 |
| 训练工具 | TRL (SFTTrainer/GRPOTrainer) | 0.24.0 |
| 数据处理 | Datasets / Pandas | 4.3.0 / 3.0.5 |
| LoRA 工具 | PEFT | (via Unsloth) |
| 模型下载 | HuggingFace Hub + hf-mirror | 0.36.2 |

#### 技术栈分层：Unsloth 与 HuggingFace 生态的关系

很多初学者会困惑：为什么代码里既用 Unsloth 又用 PEFT？它们**不是竞争关系，而是上下层关系**——Unsloth 建立在 HuggingFace 生态之上：

```
┌─────────────────────────────────────────┐
│  Unsloth（最上层：加速器）                 │
│  · FastLanguageModel / FastVisionModel  │
│  · 作用：优化 LoRA 训练速度、省显存        │
│  · 本质：给下面的库"打补丁"加速            │
├─────────────────────────────────────────┤
│  TRL（训练层）                            │
│  · SFTTrainer / GRPOTrainer             │
├─────────────────────────────────────────┤
│  PEFT（LoRA 实现层）                      │
│  · LoRA 矩阵的创建、保存、加载都在这层      │
├─────────────────────────────────────────┤
│  Transformers（模型层）                   │
│  · Qwen2ForCausalLM 等模型结构           │
│  · model.generate / tokenizer           │
├─────────────────────────────────────────┤
│  PyTorch（最底层）                        │
└─────────────────────────────────────────┘
```

**对应到本项目代码：**

| 代码调用 | 实际由谁执行 | 说明 |
|---------|------------|------|
| `FastLanguageModel.from_pretrained()` | Unsloth 入口 | 内部调 transformers 加载，加 4bit/加速补丁 |
| `FastLanguageModel.get_peft_model()` | Unsloth → **内部调 PEFT** | Unsloth 只是加优化，LoRA 层创建是 PEFT 干的 |
| `SFTTrainer` / `GRPOTrainer` | TRL | 训练循环 |
| `model.save_lora()` | Unsloth 便捷方法 | 保存为 PEFT 标准格式 adapter_model.safetensors |
| `set_peft_model_state_dict()` | **纯 PEFT** | Unsloth 没封装"注入权重"功能，回落到 PEFT 原生 API |

> **一句话理解**：Unsloth ≠ PEFT 的替代品，而是 PEFT/Transformers 的"加速外壳"。模型和 LoRA 的本体都在 HuggingFace 生态里，Unsloth 负责让它们跑得更快、更省显存。凡是 Unsloth 没封装的功能（如往已有 LoRA 层注入权重），就回落到 PEFT 原生 API。

### 1.3 硬件环境

| 资源 | 规格 |
|------|------|
| GPU | NVIDIA RTX A6000 (48GB VRAM) |
| CUDA | 8.6 / Toolkit 13.0 |
| 系统 | Windows 10 Pro |
| Python | 3.12.13 (Conda: sft 环境) |

### 1.4 基础模型

| 模型 | 用途 | 大小 |
|------|------|------|
| Qwen2.5-7B-Instruct | SFT + GRPO | ~15GB (4bit) |
| Qwen2.5-VL-3B-Instruct | VLM 视觉微调 | ~7GB |

---

## 二、三种微调范式详解

### 2.1 SFT 监督微调（Alpaca 指令数据）

#### 原理
SFT 是最基础的微调方式：给定「输入-输出」对，让模型通过监督学习直接模仿标准答案。模型学习的是「如何遵循指令回答问题」。

#### 核心代码结构（6步）

```
Step 1: 模型加载（4bit量化，节省显存）
Step 2: LoRA 适配器配置（r=16，7个target_modules）
Step 3: Alpaca 数据格式化（prompt模板 + EOS）
Step 4: SFTTrainer 训练（max_steps=60, lr=2e-4）
Step 5: 推理验证（流式输出测试）
Step 6: 保存 + 重新加载验证
```

#### LoRA 配置

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩，越大表达能力越强
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",       # MLP层
    ],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",  # 节省显存
)
```

#### 训练效果

| 指标 | 值 |
|------|------|
| 训练步数 | 60 步 |
| 可训练参数 | 20,185,088 (0.26%) |
| Loss 变化 | 2.0 → 0.18 |
| 训练时间 | ~10 分钟 |
| 模型保存 | `lora_model/` |

---

### 2.2 VLM 视觉多模态微调（汽车保险里程表识别）

#### 原理
VLM 微调让模型学会「看图说话」——输入图片+文字提示，输出结构化信息。与纯文本 SFT 的区别是数据格式包含 `image` 字段。

#### 与文本SFT的关键区别

| 对比项 | 文本SFT | 视觉VLM |
|--------|---------|---------|
| 模型类 | FastLanguageModel | FastVisionModel |
| LoRA配置 | 7个target_modules | 分别配视觉层/语言层/注意力/MLP |
| 数据格式 | 纯文本prompt | 多模态messages（text+image）|
| 验证方式 | 训练后推理 | 训练前+训练后各推理一次对比 |

#### 多模态数据格式

```python
messages = [
    {"role": "system", "content": "你是一名汽车保险承保专家..."},
    {"role": "user", "content": [
        {"type": "text", "text": "请提取关键信息"},
        {"type": "image", "image": "images/1-vehicle-odometer-reading.jpg"}
    ]}
]
```

#### 训练效果（惊艳！）

| 信息项 | 训练前 | 训练后 | 正确答案 |
|--------|--------|--------|---------|
| 总里程 | 529891.5公里 ❌ | **528,915公里** ✅ | 528,915公里 |
| 当前速度 | 30公里/小时 ❌ | **0公里/小时** ✅ | 0公里/小时 |
| 当前时间 | 未识别 | **19:18** ✅ | 19:18 |
| 当前温度 | 未识别 | **+4.6°C** ✅ | - |
| 当前挡位 | 未识别 | **停车挡** ✅ | - |

| 指标 | 值 |
|------|------|
| 训练步数 | 30 步 |
| 可训练参数 | 41,084,928 (1.08%) |
| Loss 变化 | 2.38 → 0.018 |
| 训练时间 | 37 秒 |
| LoRA 显存 | 0.51 GB |
| 模型保存 | `car_insurance_lora_model/` |

---

### 2.3 GRPO 强化学习（SFT预热 + GRPO）

#### 原理
GRPO 是 DeepSeek-R1 使用的强化学习算法：让模型对同一问题生成多个候选答案，用奖励函数评分，好的答案加强、差的削弱。模型学习的是「如何推理」而非「模仿答案」。

#### 关键挑战：冷启动问题

```
无SFT预热:  模型输出乱码 → 奖励=0 → 无梯度 → 不学习 ❌
有SFT预热:  模型学会格式 → 奖励>0 → 有梯度 → 持续优化 ✅
```

#### 解决方案：SFT 预热（DeepSeek-R1 标准做法）

**两步走策略：**
1. **SFT 预热**：用 GSM8K 数据做 SFT，让模型学会 `<reasoning>...</reasoning><answer>...</answer>` 格式
2. **GRPO 训练**：从 SFT 预热的起点开始强化学习，优化推理正确性

#### SFT 预热数据构造

```python
def format_sft_example(example):
    full_answer = example['answer']  # GSM8K格式: "推理过程\n#### 476"
    parts = full_answer.rsplit("####", 1)  # 拆分推理和答案
    reasoning = parts[0].strip()
    answer = parts[1].strip()
    # 构造target（必须严格匹配GRPO奖励函数正则）
    target = XML_COT_FORMAT.format(reasoning=reasoning, answer=answer)
    # SFT数据 = prompt + 标准答案
    return {"text": prompt + target}
```

#### GRPO 加载 SFT 预热权重

```python
# 用PEFT标准方式加载LoRA权重到已有的LoRA层
import safetensors.torch
from peft import set_peft_model_state_dict
lora_weights = safetensors.torch.load_file(
    os.path.join(sft_warmup_path, "adapter_model.safetensors")
)
set_peft_model_state_dict(model, lora_weights)
```

#### 5个奖励函数设计

| 奖励函数 | 分值 | 作用 |
|---------|------|------|
| correctness_reward_func | +2.0 | 答案是否正确（权重最高）|
| int_reward_func | +0.5 | 答案是否为整数 |
| strict_format_reward_func | +0.5 | 严格匹配XML格式 |
| soft_format_reward_func | +0.5 | 宽松匹配XML格式 |
| xmlcount_reward_func | ±0.125~0.5 | XML标签完整性 |

#### 训练效果对比

| 指标 | 无SFT预热 ❌ | 有SFT预热 ✅ |
|------|-------------|-------------|
| train_loss | 0.0 | **0.04** |
| 模型输出 | 乱码 | 正确格式 |
| 奖励范围 | 0.0 | 1.0-3.0 |
| correctness | 0.0 | **0-2.0** |
| 训练时间 | 41分钟 | 13.7分钟 |
| 最终推理 | 乱码 | π=3.14159... |

#### GRPO 学习曲线

```
步骤1:  reward=1.0  correctness=0.0  ← SFT预热让格式对了，但答错
步骤4:  reward=1.25 correctness=0.0  ← strict_format开始得分
步骤16: reward=2.0  correctness=1.0  ← 🎯 开始答对！
步骤19: reward=3.0  correctness=2.0  ← 🎯 答对更多！
步骤20: reward=1.0  correctness=0.0  ← 格式完美，持续优化中
```

---

## 三、关键技术问题与解决方案

### 3.1 torch 2.12.1 编译器兼容性

**问题**：torch.compile (inductor) 的 aliasing constraint bug 导致训练崩溃
```
RuntimeError: inductor::_alloc_from_pool: output must not also be an input
```

**解决**：禁用 torch.compile 前端和后端
```python
import torch._dynamo
torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
```

### 3.2 Triton 缓存路径问题

**问题**：Triton GPU 内核编译器无法写入默认缓存目录，且中文路径导致 Unicode 解码错误
```
PermissionError: [WinError 5] 拒绝访问
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0
```

**解决**：使用无中文的临时目录
```python
import tempfile
_TRITON_CACHE = os.path.join(tempfile.gettempdir(), "triton_cache")
os.makedirs(_TRITON_CACHE, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE
```

### 3.3 多 GPU 训练错误

**问题**：transformers 4.57.6 在多 GPU 时 `loss.mean()` 报错
```
AttributeError: 'int' object has no attribute 'mean'
```

**解决**：限制只用 1 个 GPU
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 3.4 GRPO 冷启动失败

**问题**：模型初始输出全是乱码 → 所有奖励=0 → advantage=0 → loss=0 → 不学习

**解决**：SFT 预热让模型先学会输出格式，GRPO 从有效起点开始优化

### 3.5 vLLM 依赖移除

**问题**：vLLM 在 Windows 上编译困难，但 GRPO 脚本依赖它

**解决**：改用 transformers 原生 generate，功能等价但稍慢
```python
# 原: model.fast_generate(sampling_params=..., lora_request=...)
# 改: model.generate(max_new_tokens=..., temperature=..., do_sample=True)
```

---

## 四、项目文件结构

```
39-LLM模型蒸馏与微调实操/
├── models/Qwen/                          # 基础模型
│   ├── Qwen2.5-7B-Instruct/
│   └── Qwen2.5-VL-3B-Instruct/
├── 【数据集】/                            # 训练数据
│   ├── alpaca-cleaned/
│   ├── gsm8k/
│   └── qwen-vl-train.xlsx + images/
├── Qwen2_5_(7B)_Alpaca.py               # ① SFT 监督微调
├── qwen_vl_car_insurance_train.py         # ② VLM 视觉微调
├── Qwen2_5_(7B)_R1_GRPO.py               # ③ GRPO 强化学习
├── sft_warmup_grpo.py                    # SFT 预热脚本
├── download_model.py                     # 模型下载脚本
├── lora_model/                           # Alpaca 训练产物
├── car_insurance_lora_model/             # VL 训练产物
├── sft_warmup_lora/                      # SFT 预热产物
└── grpo_saved_lora/                      # GRPO 训练产物
```

---

## 五、项目成果总结

### 5.1 三种微调范式完成情况

| 范式 | 脚本 | 状态 | 核心指标 |
|------|------|------|---------|
| SFT 监督微调 | Qwen2_5_(7B)_Alpaca.py | ✅ 成功 | loss 2.0→0.18 |
| VLM 视觉微调 | qwen_vl_car_insurance_train.py | ✅ 成功 | 里程数精准修正 |
| GRPO 强化学习 | Qwen2_5_(7B)_R1_GRPO.py + sft_warmup_grpo.py | ✅ 成功 | reward 0→3.0, correctness 0→2.0 |

### 5.2 关键技术突破

1. **SFT 预热解决 GRPO 冷启动** — 复现 DeepSeek-R1 的标准训练流程
2. **Windows 环境适配** — 解决 torch.compile、Triton 缓存、多GPU 等兼容性问题
3. **vLLM 依赖移除** — 用 transformers 原生 generate 替代，保持功能等价
4. **高效微调** — LoRA 只训练 0.26%-1.08% 参数，显存占用 0.5-4.3GB

---

## 六、简历项目描述（参考）

### 简版（1-2句）
> **LLM 模型微调实践项目**：基于 Qwen2.5-7B 模型，使用 Unsloth+LoRA 完成三种微调范式（SFT 监督微调、VLM 视觉多模态微调、GRPO 强化学习），在 RTX A6000 上实现高效训练，并复现 DeepSeek-R1 的 SFT 预热 + GRPO 两阶段训练流程。

### 详版（简历项目栏）

**LLM 模型蒸馏与微调实操** | PyTorch, Unsloth, LoRA, GRPO

- 基于 Qwen2.5-7B-Instruct / Qwen2.5-VL-3B-Instruct，完成 SFT、VLM、GRPO 三种微调范式实践
- **SFT 监督微调**：用 Alpaca 5万条指令数据做 LoRA 微调（r=16），训练 60 步 loss 从 2.0 降至 0.18
- **VLM 视觉微调**：用汽车里程表图片做多模态微调，30 步训练后模型从乱码识别（529891）修正为精准读数（528,915），训练仅 37 秒、占用 0.5GB 显存
- **GRPO 强化学习**：实现 SFT 预热 + GRPO 两阶段训练（DeepSeek-R1 方法），解决 GRPO 冷启动失败问题，训练后模型奖励从 0 提升至 3.0，答案正确率 correctness 从 0 提升至 2.0
- **工程能力**：解决 torch 2.12 编译器 bug、Triton 中文路径问题、多 GPU 兼容性等 Windows 环境适配难题；移除 vLLM 依赖改用 transformers 原生推理

### 技术关键词
`LLM微调` `LoRA` `SFT` `VLM多模态` `GRPO强化学习` `DeepSeek-R1` `Unsloth` `PEFT` `4bit量化` `PyTorch` `Qwen2.5`

---

## 七、核心知识点速查

| 概念 | 一句话理解 |
|------|----------|
| Unsloth | HuggingFace 生态（Transformers/PEFT/TRL）之上的加速外壳，不是替代品 |
| PEFT | LoRA 的真正实现层，LoRA 矩阵创建/保存/加载都在这层 |
| LoRA | 冻结原始权重，只训练低秩矩阵（0.26%参数），高效微调 |
| 4bit 量化 | 把 FP16 权重压成 4bit，显存占用降到 1/4 |
| SFT | 给标准答案，模型直接模仿（监督学习）|
| GRPO | 不给答案，给奖励信号，模型自己探索最优策略（强化学习）|
| 冷启动问题 | 模型初始太差→奖励为0→无梯度→不学习（GRPO的致命问题）|
| SFT 预热 | 先用 SFT 让模型有基础能力，再用 GRPO 优化（DeepSeek-R1 方法）|
| Target Modules | LoRA 作用的层：注意力层(q/k/v/o) + MLP层(gate/up/down)|
| Reward Function | 评分模型输出的函数，GRPO 的核心驱动力 |
