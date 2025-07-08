#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_task1_to_zs_df(input_file, output_file):
    """
    将task1的二分类JSON文件转换为ZS-DF (Zero-shot + Decomposition)格式
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 标准化的ZS-DF prompt模板
    zs_df_instruction = """Please analyze the given news article step by step to determine if it is legitimate or fake news. Follow this decomposition approach:

1. **Content Analysis**: Examine the factual claims, sources mentioned, and verifiable information
2. **Language Pattern**: Analyze the writing style, emotional tone, and linguistic characteristics  
3. **Source Credibility**: Evaluate the reliability and authenticity of cited sources
4. **Logical Consistency**: Check for internal contradictions or implausible claims
5. **Final Classification**: Based on your analysis, classify as either "legitimate" or "fake"

Provide your reasoning for each step, then conclude with your final answer in one word: legitimate or fake."""

    alpaca_data = []
    for item in data:
        text = item['text']
        label = item['label']
        
        # 根据标签创建期望输出（包含推理过程）
        if label == 1:  # fake news
            expected_output = """**Content Analysis**: The article contains unverified claims and lacks credible source attribution.

**Language Pattern**: The writing exhibits sensationalized language and emotional manipulation typical of misinformation.

**Source Credibility**: Sources are either anonymous, unreliable, or completely absent.

**Logical Consistency**: Contains implausible details or contradictory information that doesn't align with verified facts.

**Final Classification**: Based on the analysis above, this appears to be fabricated content designed to mislead readers.

fake"""
        else:  # legitimate news
            expected_output = """**Content Analysis**: The article presents factual information with proper context and verifiable details.

**Language Pattern**: The writing follows journalistic standards with objective tone and professional presentation.

**Source Credibility**: Information is attributed to reliable, identifiable sources that can be verified.

**Logical Consistency**: All claims are internally consistent and align with established facts and timelines.

**Final Classification**: Based on the analysis above, this appears to be credible journalism from legitimate sources.

legitimate"""
        
        alpaca_item = {
            "instruction": zs_df_instruction,
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
    # 获取脚本所在目录，然后找到data目录
    script_dir = Path(__file__).parent
    base_dir = script_dir / "data"
    
    # Task 1 ZS-DF 转换
    print("=== 开始转换 Task 1 到 ZS-DF 格式 ===")
    task1_dir = base_dir / "data_table" / "task1" / "full"
    task1_output_dir = base_dir / "data_table" / "task1" / "alpaca_full_ZS_DF"
    
    total_converted = 0
    
    # 转换task1的文件
    for json_file in task1_dir.glob("*.json"):
        if json_file.name != "README.md":  # 跳过README文件
            output_file = task1_output_dir / f"zs_df_{json_file.name}"
            count = convert_task1_to_zs_df(str(json_file), str(output_file))
            total_converted += count
    
    print(f"\n=== Task 1 ZS-DF 转换完成！总共转换了 {total_converted} 条数据 ===")
    print(f"输出目录: {task1_output_dir}")

if __name__ == "__main__":
    main() 