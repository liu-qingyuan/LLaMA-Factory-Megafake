#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_task2_to_alpaca(input_file, output_file, subclass_type, is_fake_news):
    """
    将task2的二分类JSON文件转换为alpaca格式
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    alpaca_data = []
    
    # 根据子类类型创建不同的指令
    if is_fake_news:
        if subclass_type == "style_based":
            instruction_template = "Given that this news is fake, classify whether it is style-based fake news (rephrased legitimate news in tabloid style or fake news in mainstream style) or other types of fake news. Answer with 'style-based' or 'other': {}"
            positive_output = "style-based"
            negative_output = "other"
        elif subclass_type == "content_based":
            instruction_template = "Given that this news is fake, classify whether it is content-based fake news (manipulated legitimate news with modified attributes) or other types of fake news. Answer with 'content-based' or 'other': {}"
            positive_output = "content-based"
            negative_output = "other"
        elif subclass_type == "integration_based":
            instruction_template = "Given that this news is fake, classify whether it is integration-based fake news (integrated fake and legitimate news) or other types of fake news. Answer with 'integration-based' or 'other': {}"
            positive_output = "integration-based"
            negative_output = "other"
        elif subclass_type == "story_based":
            instruction_template = "Given that this news is fake, classify whether it is story-based fake news (generated from a certain message) or other types of fake news. Answer with 'story-based' or 'other': {}"
            positive_output = "story-based"
            negative_output = "other"
    else:  # legitimate news
        if subclass_type == "style_based":
            instruction_template = "Given that this news is legitimate, classify whether it is style-based legitimate news (polished while preserving original information) or integration-based legitimate news. Answer with 'style-based' or 'integration-based': {}"
            positive_output = "style-based"
            negative_output = "integration-based"
        elif subclass_type == "integration_based":
            instruction_template = "Given that this news is legitimate, classify whether it is integration-based legitimate news (condensed various legitimate news into synthesized summary) or style-based legitimate news. Answer with 'integration-based' or 'style-based': {}"
            positive_output = "integration-based"
            negative_output = "style-based"
    
    for item in data:
        text = item['text']
        label = item['label']
        
        # 根据标签创建指令输出
        if label == 1:
            output = positive_output
        else:
            output = negative_output
        
        alpaca_item = {
            "instruction": instruction_template.format(text),
            "input": "",
            "output": output
        }
        alpaca_data.append(alpaca_item)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共转换 {len(alpaca_data)} 条数据")
    return len(alpaca_data)

def main():
    base_dir = Path("/root/autodl-tmp/LLaMA-Factory-Megafake/data")
    
    # Task 2 转换
    print("=== 开始转换 Task 2 ===")
    task2_input_dir = base_dir / "data_table" / "task2" / "full"
    task2_output_dir = base_dir / "data_table" / "task2" / "alpaca_full"
    
    total_converted = 0
    
    # 处理 GLM 数据
    print("\n--- 处理 GLM 数据 ---")
    glm_input_dir = task2_input_dir / "glm"
    glm_output_dir = task2_output_dir / "glm"
    
    # GLM 假新闻子类
    fake_subclasses = ["style_based", "content_based", "integration_based", "story_based"]
    for subclass in fake_subclasses:
        input_file = glm_input_dir / f"glm_{subclass}_fake.json"
        output_file = glm_output_dir / f"alpaca_glm_{subclass}_fake.json"
        if input_file.exists():
            count = convert_task2_to_alpaca(str(input_file), str(output_file), subclass, is_fake_news=True)
            total_converted += count
    
    # GLM 真新闻子类
    legitimate_subclasses = ["style_based", "integration_based"]
    for subclass in legitimate_subclasses:
        input_file = glm_input_dir / f"glm_{subclass}_legitimate.json"
        output_file = glm_output_dir / f"alpaca_glm_{subclass}_legitimate.json"
        if input_file.exists():
            count = convert_task2_to_alpaca(str(input_file), str(output_file), subclass, is_fake_news=False)
            total_converted += count
    
    # 处理 LLaMA 数据
    print("\n--- 处理 LLaMA 数据 ---")
    llama_input_dir = task2_input_dir / "llama"
    llama_output_dir = task2_output_dir / "llama"
    
    # LLaMA 假新闻子类
    for subclass in fake_subclasses:
        input_file = llama_input_dir / f"llama3_{subclass}_fake.json"
        output_file = llama_output_dir / f"alpaca_llama3_{subclass}_fake.json"
        if input_file.exists():
            count = convert_task2_to_alpaca(str(input_file), str(output_file), subclass, is_fake_news=True)
            total_converted += count
    
    # LLaMA 真新闻子类
    for subclass in legitimate_subclasses:
        input_file = llama_input_dir / f"llama3_{subclass}_legitimate.json"
        output_file = llama_output_dir / f"alpaca_llama3_{subclass}_legitimate.json"
        if input_file.exists():
            count = convert_task2_to_alpaca(str(input_file), str(output_file), subclass, is_fake_news=False)
            total_converted += count
    
    print(f"\n=== Task 2 转换完成！总共转换了 {total_converted} 条数据 ===")

if __name__ == "__main__":
    main() 