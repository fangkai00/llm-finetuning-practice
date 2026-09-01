# -*- coding: utf-8 -*-
"""
微调/蒸馏效果对比评估工具
课程：LLM模型蒸馏与微调实操
功能：对比基座模型 vs SFT微调模型 vs GRPO模型的输出质量
环境：需要GPU，建议在AutoDL上运行
"""

import json
import re
import time
import torch


# ========================================
# 评估维度定义
# ========================================

class EvalMetrics:
    """评估指标计算"""
    
    @staticmethod
    def format_compliance(response, expected_format="xml"):
        """
        格式遵循能力评估
        检查模型输出是否遵循指定格式
        """
        if expected_format == "xml":
            has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', response, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))
            
            score = 0.0
            if has_reasoning:
                score += 0.5
            if has_answer:
                score += 0.5
            return score
        
        elif expected_format == "medical":
            has_structure = "###" in response or len(response) > 20
            return 1.0 if has_structure else 0.0
        
        return 0.0
    
    @staticmethod
    def answer_accuracy(response, reference_answer, threshold=0.5):
        """
        答案准确率评估（简化版）
        基于关键词匹配的准确性判断
        """
        if not reference_answer:
            return 0.0
        
        ref_chars = set(reference_answer)
        resp_chars = set(response)
        overlap = len(ref_chars & resp_chars)
        precision = overlap / len(resp_chars) if resp_chars else 0
        recall = overlap / len(ref_chars) if ref_chars else 0
        
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)
    
    @staticmethod
    def reasoning_quality(response):
        """
        推理链质量评估
        检查推理过程的结构性和逻辑性
        """
        score = 0.0
        
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if not reasoning_match:
            return 0.0
        
        reasoning = reasoning_match.group(1).strip()
        
        # 长度评分（推理过程不应太短）
        if len(reasoning) > 50:
            score += 0.3
        elif len(reasoning) > 20:
            score += 0.15
        
        # 步骤性评分（是否包含分步推理）
        step_indicators = ['first', 'then', 'next', 'finally', 'therefore',
                          '首先', '然后', '接下来', '最后', '因此', '所以',
                          '1.', '2.', '3.', 'step']
        step_count = sum(1 for indicator in step_indicators 
                        if indicator.lower() in reasoning.lower())
        if step_count >= 2:
            score += 0.4
        elif step_count >= 1:
            score += 0.2
        
        # 结论性评分
        conclusion_indicators = ['therefore', 'so', 'thus', '因此', '所以', '综上']
        has_conclusion = any(ind in reasoning.lower() for ind in conclusion_indicators)
        if has_conclusion:
            score += 0.3
        
        return round(min(score, 1.0), 4)
    
    @staticmethod
    def response_length_score(response, min_len=20, max_len=2000):
        """
        回复长度合理性评分
        太短或太长都扣分
        """
        length = len(response)
        if length < min_len:
            return round(length / min_len, 4)
        elif length > max_len:
            return round(max(0, 1.0 - (length - max_len) / max_len), 4)
        return 1.0
    
    @staticmethod
    def language_match(response, expected_language="chinese"):
        """
        语言匹配评估
        检查回复语言是否与期望一致
        """
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', response))
        en_chars = len(re.findall(r'[a-zA-Z]', response))
        total = cn_chars + en_chars
        
        if total == 0:
            return 0.0
        
        if expected_language == "chinese":
            return round(cn_chars / total, 4)
        else:
            return round(en_chars / total, 4)


# ========================================
# 模型推理封装
# ========================================

def inference_base_model(model, tokenizer, prompt_template, question, max_tokens=256):
    """使用基座模型进行推理"""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    
    inputs = tokenizer(
        [prompt_template.format(question, "")],
        return_tensors="pt"
    ).to("cuda")
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    inference_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### 回答：" in response:
        response = response.split("### 回答：")[-1].strip()
    elif "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response, inference_time


# ========================================
# 对比评估
# ========================================

def evaluate_single(response, reference=None, eval_type="medical"):
    """对单条回复进行多维度评估"""
    metrics = EvalMetrics()
    
    result = {
        "response_length": len(response),
        "length_score": metrics.response_length_score(response),
    }
    
    if eval_type == "medical":
        result["format_score"] = metrics.format_compliance(response, "medical")
        result["language_match"] = metrics.language_match(response, "chinese")
    elif eval_type == "reasoning":
        result["format_score"] = metrics.format_compliance(response, "xml")
        result["reasoning_quality"] = metrics.reasoning_quality(response)
    
    if reference:
        result["accuracy"] = metrics.answer_accuracy(response, reference)
    
    # 综合评分
    scores = [v for k, v in result.items() if k.endswith('_score') or k in ['accuracy', 'language_match', 'reasoning_quality']]
    result["overall_score"] = round(sum(scores) / len(scores), 4) if scores else 0
    
    return result


def compare_models(test_cases, model_results, eval_type="medical"):
    """
    对比多个模型的评估结果
    
    参数:
        test_cases: list[dict], 测试用例 [{question, reference}, ...]
        model_results: dict[str, list[str]], 各模型的回复 {model_name: [responses]}
        eval_type: str, 评估类型 medical/reasoning
    """
    print("=" * 70)
    print("模型对比评估报告")
    print("=" * 70)
    
    all_scores = {name: [] for name in model_results}
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        reference = test_case.get("reference", None)
        
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"问题: {question}")
        if reference:
            print(f"参考答案: {reference[:100]}...")
        
        for model_name, responses in model_results.items():
            if i < len(responses):
                response = responses[i]
                result = evaluate_single(response, reference, eval_type)
                all_scores[model_name].append(result)
                
                print(f"\n  [{model_name}]")
                print(f"    回复: {response[:150]}...")
                print(f"    综合评分: {result['overall_score']}")
                for k, v in result.items():
                    if k not in ['overall_score', 'response_length']:
                        print(f"    {k}: {v}")
    
    # 总结
    print(f"\n{'='*70}")
    print("总体评估汇总")
    print(f"{'='*70}")
    
    header = f"{'模型':<20}"
    for model_name in model_results:
        header += f"{model_name:<20}"
    print(header)
    print("-" * 70)
    
    for model_name, scores_list in all_scores.items():
        if scores_list:
            avg_overall = sum(s['overall_score'] for s in scores_list) / len(scores_list)
            print(f"  {model_name}: 平均综合评分 = {avg_overall:.4f}")
    
    return all_scores


# ========================================
# 演示：对比评估示例
# ========================================

def demo_comparison():
    """演示对比评估流程（使用模拟数据）"""
    
    # 模拟测试用例
    test_cases = [
        {
            "question": "我最近总是感觉头晕，应该怎么办？",
            "reference": "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议注意休息，保证充足睡眠，适当补充营养。如持续不好转，建议就医检查。"
        },
        {
            "question": "感冒发烧应该吃什么药？",
            "reference": "感冒发烧可以服用对乙酰氨基酚或布洛芬退烧，同时多喝水多休息。如体温持续超过38.5度超过3天，建议就医。"
        },
        {
            "question": "高血压患者需要注意什么？",
            "reference": "高血压患者需要低盐饮食，规律服药，适当运动，定期监测血压，保持心情舒畅。"
        },
    ]
    
    # 模拟不同模型的输出
    model_results = {
        "基座模型(微调前)": [
            "I can provide general health advice. Dizziness can be caused by many factors.",
            "For cold and fever, please consult your doctor for proper medication.",
            "Hypertension patients should follow medical advice.",
        ],
        "SFT微调模型": [
            "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议您首先注意休息，保证充足的睡眠，适当补充营养。如果症状持续不缓解，建议到医院做详细检查。",
            "感冒发烧建议多喝水、多休息。如果体温超过38.5度，可以服用退烧药物如布洛芬或对乙酰氨基酚。如果持续高烧不退，建议就医检查。",
            "高血压患者要注意：1.低盐饮食，每天盐摄入不超过6克；2.按时服药，不要随意停药；3.适量运动；4.定期测量血压；5.保持心态平和。",
        ],
        "GRPO强化学习模型": [
            "<reasoning>\n患者描述头晕症状，需要分析可能原因：低血糖、贫血、颈椎病、血压异常等。首先建议基础调理，如果不改善则需要就医检查。\n</reasoning>\n<answer>\n头晕可能是低血糖、贫血或颈椎问题导致。建议多休息、保证睡眠和营养，持续不缓解请就医。\n</answer>",
            "<reasoning>\n感冒发烧是常见病症，需要根据体温高低选择处理方式。低于38.5度可以物理降温，超过38.5度需要用退烧药。\n</reasoning>\n<answer>\n发烧超38.5度可服用布洛芬或对乙酰氨基酚，多喝水休息。持续高烧3天以上建议就医。\n</answer>",
            "<reasoning>\n高血压是慢性疾病，需要从饮食、运动、用药、监测多方面综合管理。\n</reasoning>\n<answer>\n高血压管理要点：低盐饮食、规律用药、适度运动、定期测血压、情绪管理。\n</answer>",
        ],
    }
    
    # 执行对比评估
    print("\n===== 医疗场景评估 =====")
    compare_models(test_cases, model_results, eval_type="medical")
    
    # GRPO模型单独做推理质量评估
    print("\n\n===== GRPO模型推理质量评估 =====")
    grpo_results = model_results["GRPO强化学习模型"]
    for i, (case, response) in enumerate(zip(test_cases, grpo_results)):
        result = evaluate_single(response, case.get("reference"), eval_type="reasoning")
        print(f"\n用例{i+1}: {case['question']}")
        print(f"  推理质量: {result.get('reasoning_quality', 'N/A')}")
        print(f"  格式遵循: {result.get('format_score', 'N/A')}")
        print(f"  综合评分: {result['overall_score']}")


def main():
    demo_comparison()
    print("\n\n评估演示完成!")
    print("实际使用时，请替换模拟数据为真实模型推理结果。")


if __name__ == "__main__":
    main()
