#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_task1_to_cot_sc(input_file, output_file):
    """
    将task1的二分类JSON文件转换为CoT-SC (Chain-of-Thought + Self-Consistency)格式
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 标准化的CoT-SC prompt模板
    cot_sc_instruction = """Analyze the given news article to determine if it is legitimate or fake news. Use Chain-of-Thought reasoning with Self-Consistency approach:

1. Generate 5 different reasoning chains to analyze the article
2. For each chain, consider different aspects: content credibility, source reliability, language patterns, logical consistency, factual accuracy, etc.
3. Reach a conclusion for each reasoning chain
4. Vote among the 5 conclusions to determine the final answer

Format your response as follows:

**Reasoning Chain 1:**
[Your first line of reasoning and conclusion]

**Reasoning Chain 2:**
[Your second line of reasoning and conclusion]

**Reasoning Chain 3:**
[Your third line of reasoning and conclusion]

**Reasoning Chain 4:**
[Your fourth line of reasoning and conclusion]

**Reasoning Chain 5:**
[Your fifth line of reasoning and conclusion]

**Self-Consistency Vote:**
[Compare the five conclusions and determine the majority vote]

**Final Answer:** [legitimate/fake]"""

    alpaca_data = []
    for item in data:
        text = item['text']
        label = item['label']
        
        # 根据标签创建期望输出（包含5条推理链和投票过程）
        if label == 1:  # fake news
            expected_output = """**Reasoning Chain 1:**
Examining the content structure and claims: This article contains unverified statements, lacks proper source attribution, and presents information that cannot be independently confirmed. The claims appear sensationalized and designed to provoke emotional responses rather than inform. The absence of credible sources and verifiable facts suggests this is fabricated content.
Conclusion: fake

**Reasoning Chain 2:**
Analyzing language patterns and presentation: The writing style exhibits characteristics commonly found in misinformation - sensational headlines, emotional manipulation, vague sourcing (anonymous sources, "reports suggest"), and lack of balanced perspective. Professional journalism typically includes multiple viewpoints and clear attribution.
Conclusion: fake

**Reasoning Chain 3:**
Evaluating logical consistency and factual accuracy: Cross-referencing the claims with established facts reveals inconsistencies and implausible details. The timeline, locations, or people mentioned may not align with verified information. The article fails basic fact-checking standards expected of legitimate news.
Conclusion: fake

**Reasoning Chain 4:**
Assessing source credibility and verification: The article lacks citations to reputable news organizations, official statements, or verifiable documents. Claims cannot be traced back to primary sources or corroborated through independent fact-checking. This absence of reliable sourcing is a hallmark of fabricated news content.
Conclusion: fake

**Reasoning Chain 5:**
Examining publication context and intent: The article appears designed to spread misinformation rather than inform the public. It may contain clickbait elements, inflammatory language, or content that appeals to biases rather than presenting balanced, factual reporting. The overall presentation suggests malicious intent to deceive.
Conclusion: fake

**Self-Consistency Vote:**
All five reasoning chains (Chain 1: fake, Chain 2: fake, Chain 3: fake, Chain 4: fake, Chain 5: fake) consistently identify this as fake news. The unanimous conclusion across multiple analytical approaches provides very high confidence in the assessment.

**Final Answer:** fake"""
        else:  # legitimate news
            expected_output = """**Reasoning Chain 1:**
Examining the content structure and claims: This article presents information in a structured, factual manner with proper context. The claims made are specific, verifiable, and presented with appropriate sourcing. The content follows journalistic standards with clear attribution and balanced reporting.
Conclusion: legitimate

**Reasoning Chain 2:**
Analyzing language patterns and presentation: The writing demonstrates professional journalistic style with objective tone, proper grammar, and balanced presentation. Sources are clearly identified and credible. The article avoids sensationalism and presents information in a measured, factual way consistent with legitimate news reporting.
Conclusion: legitimate

**Reasoning Chain 3:**
Evaluating logical consistency and factual accuracy: The information presented is internally consistent and aligns with verifiable facts. Timeline, locations, and people mentioned can be cross-referenced with reliable sources. The article meets standard fact-checking criteria and journalistic integrity expectations.
Conclusion: legitimate

**Reasoning Chain 4:**
Assessing source credibility and verification: The article cites reputable news organizations, official statements, or verifiable documents. Claims can be traced back to primary sources and corroborated through independent verification. The sourcing meets professional journalism standards for reliability and transparency.
Conclusion: legitimate

**Reasoning Chain 5:**
Examining publication context and intent: The article appears designed to inform the public with factual, unbiased reporting. It follows ethical journalism principles, presents information objectively without inflammatory language, and serves the public interest. The overall presentation demonstrates professional journalistic integrity.
Conclusion: legitimate

**Self-Consistency Vote:**
All five reasoning chains (Chain 1: legitimate, Chain 2: legitimate, Chain 3: legitimate, Chain 4: legitimate, Chain 5: legitimate) consistently identify this as legitimate news. The unanimous conclusion across multiple analytical approaches provides very high confidence in the assessment.

**Final Answer:** legitimate"""
        
        alpaca_item = {
            "instruction": cot_sc_instruction,
            "input": f"News Article: {text}",
            "output": expected_output
        }
        alpaca_data.append(alpaca_item)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共转换 {len(alpaca_data)} 条数据")
    return len(alpaca_data)

def main():
    base_dir = Path("/root/autodl-tmp/LLaMA-Factory-Megafake2/data")
    
    # Task 1 CoT-SC 转换
    print("=== 开始转换 Task 1 到 CoT-SC 格式 ===")
    task1_dir = base_dir / "data_table" / "task1" / "full"
    task1_output_dir = base_dir / "data_table" / "task1" / "alpaca_full_CoT_SC"
    
    total_converted = 0
    
    # 转换task1的文件
    for json_file in task1_dir.glob("*.json"):
        if json_file.name != "README.md":  # 跳过README文件
            output_file = task1_output_dir / f"cot_sc_{json_file.name}"
            count = convert_task1_to_cot_sc(str(json_file), str(output_file))
            total_converted += count
    
    print(f"\n=== Task 1 CoT-SC 转换完成！总共转换了 {total_converted} 条数据 ===")
    print(f"输出目录: {task1_output_dir}")

if __name__ == "__main__":
    main() 