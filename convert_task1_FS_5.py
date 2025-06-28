#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path

def create_few_shot_examples(data, num_examples=5):
    """
    从数据中选择平衡的短文本示例作为Few-shot examples
    """
    # 先筛选出长度适中的文章 (50-150字符)
    short_texts = [item for item in data if 50 <= len(item['text']) <= 150]
    
    # 分离真实和虚假新闻
    legitimate_examples = [item for item in short_texts if item['label'] == 0]
    fake_examples = [item for item in short_texts if item['label'] == 1]
    
    print(f"可用的短文本示例: legitimate={len(legitimate_examples)}, fake={len(fake_examples)}")
    
    # 随机选择示例，保持平衡
    random.seed(42)  # 固定随机种子以确保可重复性
    selected_legitimate = random.sample(legitimate_examples, min(3, len(legitimate_examples)))
    selected_fake = random.sample(fake_examples, min(2, len(fake_examples)))
    
    # 组合并打乱顺序
    few_shot_examples = selected_legitimate + selected_fake
    random.shuffle(few_shot_examples)
    
    return few_shot_examples[:num_examples]

def format_few_shot_examples(examples):
    """
    格式化Few-shot示例为prompt格式
    """
    formatted_examples = []
    
    for i, example in enumerate(examples, 1):
        text = example['text']  # 不需要截断，因为已经选择了短文本
        label = "legitimate" if example['label'] == 0 else "fake"
        
        formatted_example = f"""Example {i}:
News Article: {text}
Classification: {label}"""
        formatted_examples.append(formatted_example)
    
    return "\n\n".join(formatted_examples)

def convert_task1_to_fs_5(input_file, output_file):
    """
    将task1的二分类JSON文件转换为FS-5 (Few-shot with 5 examples)格式
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建Few-shot示例
    few_shot_examples = create_few_shot_examples(data, 5)
    formatted_examples = format_few_shot_examples(few_shot_examples)
    
    # 标准化的FS-5 prompt模板
    fs_5_instruction = f"""You are an expert at identifying fake news. Based on the following examples, classify the given news article as either "legitimate" or "fake".

{formatted_examples}

Now, analyze the following news article and classify it as either "legitimate" or "fake". Provide your reasoning briefly, then conclude with your final answer in one word."""

    alpaca_data = []
    
    # 创建用于Few-shot的示例ID集合，避免重复
    example_texts = {example['text'] for example in few_shot_examples}
    
    for item in data:
        text = item['text']
        label = item['label']
        
        # 跳过已经用作Few-shot示例的文章
        if text in example_texts:
            continue
        
        # 根据标签创建期望输出
        if label == 1:  # fake news
            expected_output = """Based on the examples provided, this article exhibits characteristics typical of fake news: it contains unverified claims, lacks credible sources, uses sensationalized language, or presents information that cannot be independently verified. The content appears to be fabricated or misleading.

fake"""
        else:  # legitimate news
            expected_output = """Based on the examples provided, this article exhibits characteristics of legitimate news: it presents factual information, cites credible sources, uses professional journalistic language, and the content can be verified through reliable channels.

legitimate"""
        
        alpaca_item = {
            "instruction": fs_5_instruction,
            "input": f"News Article: {text}",
            "output": expected_output
        }
        alpaca_data.append(alpaca_item)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共转换 {len(alpaca_data)} 条数据")
    print(f"使用的Few-shot示例数量: {len(few_shot_examples)}")
    return len(alpaca_data)

def main():
    base_dir = Path("/root/autodl-tmp/LLaMA-Factory-Megafake2/data")
    
    # Task 1 FS-5 转换
    print("=== 开始转换 Task 1 到 FS-5 格式 ===")
    task1_dir = base_dir / "data_table" / "task1" / "full"
    task1_output_dir = base_dir / "data_table" / "task1" / "alpaca_full_FS_5"
    
    total_converted = 0
    
    # 转换task1的文件
    for json_file in task1_dir.glob("*.json"):
        if json_file.name != "README.md":  # 跳过README文件
            output_file = task1_output_dir / f"fs_5_{json_file.name}"
            count = convert_task1_to_fs_5(str(json_file), str(output_file))
            total_converted += count
    
    print(f"\n=== Task 1 FS-5 转换完成！总共转换了 {total_converted} 条数据 ===")
    print(f"输出目录: {task1_output_dir}")

if __name__ == "__main__":
    main() 