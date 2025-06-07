#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_data_from_files():
    """从 task2 的多个子类 JSON 文件中分别随机选择数据并保存到新目录"""
    
    # 设置路径
    source_dir = Path("data/data_table/task2/alpaca_full")
    target_dir = Path("data/data_table/task2/small_8k")
    
    # 创建目标目录
    glm_target_dir = target_dir / "glm"
    llama_target_dir = target_dir / "llama"
    glm_target_dir.mkdir(parents=True, exist_ok=True)
    llama_target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在处理 Task2 数据采样...")
    
    # 定义子类文件
    fake_subclasses = ["style_based", "content_based", "integration_based", "story_based"]
    legitimate_subclasses = ["style_based", "integration_based"]
    
    # 每个子类文件采样 8000 条数据
    samples_per_file = 8000
    
    total_glm_sampled = 0
    total_llama_sampled = 0
    
    # 处理 GLM 数据
    print(f"\n=== 处理 GLM 数据 (每个文件目标: {samples_per_file} 条) ===")
    glm_source_dir = source_dir / "glm"
    
    # GLM 假新闻子类
    for subclass in fake_subclasses:
        input_file = glm_source_dir / f"alpaca_glm_{subclass}_fake.json"
        output_file = glm_target_dir / f"alpaca_glm_{subclass}_fake_8k.json"
        
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"GLM {subclass}_fake 文件包含 {len(data)} 条数据")
            
            if len(data) < samples_per_file:
                print(f"警告：GLM {subclass}_fake 数据量 ({len(data)}) 少于目标数量 ({samples_per_file})")
                selected_data = data
            else:
                selected_data = random.sample(data, samples_per_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(selected_data)} 条 GLM {subclass}_fake 数据")
            total_glm_sampled += len(selected_data)
            
            # 统计标签分布
            subclass_with_hyphen = subclass.replace("_", "-")  # 转换为连字符格式
            positive_count = sum(1 for item in selected_data if item['output'] == subclass_with_hyphen)
            negative_count = len(selected_data) - positive_count
            print(f"  - {subclass_with_hyphen}: {positive_count} 条 ({positive_count/len(selected_data)*100:.1f}%)")
            print(f"  - other: {negative_count} 条 ({negative_count/len(selected_data)*100:.1f}%)")
        else:
            print(f"警告：文件 {input_file} 不存在")
    
    # GLM 真新闻子类
    for subclass in legitimate_subclasses:
        input_file = glm_source_dir / f"alpaca_glm_{subclass}_legitimate.json"
        output_file = glm_target_dir / f"alpaca_glm_{subclass}_legitimate_8k.json"
        
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"GLM {subclass}_legitimate 文件包含 {len(data)} 条数据")
            
            if len(data) < samples_per_file:
                print(f"警告：GLM {subclass}_legitimate 数据量 ({len(data)}) 少于目标数量 ({samples_per_file})")
                selected_data = data
            else:
                selected_data = random.sample(data, samples_per_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(selected_data)} 条 GLM {subclass}_legitimate 数据")
            total_glm_sampled += len(selected_data)
            
            # 统计标签分布
            subclass_with_hyphen = subclass.replace("_", "-")  # 转换为连字符格式
            positive_count = sum(1 for item in selected_data if item['output'] == subclass_with_hyphen)
            other_subclass = "integration-based" if subclass == "style_based" else "style-based"
            negative_count = sum(1 for item in selected_data if item['output'] == other_subclass)
            print(f"  - {subclass_with_hyphen}: {positive_count} 条 ({positive_count/len(selected_data)*100:.1f}%)")
            print(f"  - {other_subclass}: {negative_count} 条 ({negative_count/len(selected_data)*100:.1f}%)")
        else:
            print(f"警告：文件 {input_file} 不存在")
    
    # 处理 LLaMA 数据
    print(f"\n=== 处理 LLaMA 数据 (每个文件目标: {samples_per_file} 条) ===")
    llama_source_dir = source_dir / "llama"
    
    # LLaMA 假新闻子类
    for subclass in fake_subclasses:
        input_file = llama_source_dir / f"alpaca_llama3_{subclass}_fake.json"
        output_file = llama_target_dir / f"alpaca_llama3_{subclass}_fake_8k.json"
        
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"LLaMA {subclass}_fake 文件包含 {len(data)} 条数据")
            
            if len(data) < samples_per_file:
                print(f"警告：LLaMA {subclass}_fake 数据量 ({len(data)}) 少于目标数量 ({samples_per_file})")
                selected_data = data
            else:
                selected_data = random.sample(data, samples_per_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(selected_data)} 条 LLaMA {subclass}_fake 数据")
            total_llama_sampled += len(selected_data)
            
            # 统计标签分布
            subclass_with_hyphen = subclass.replace("_", "-")  # 转换为连字符格式
            positive_count = sum(1 for item in selected_data if item['output'] == subclass_with_hyphen)
            negative_count = len(selected_data) - positive_count
            print(f"  - {subclass_with_hyphen}: {positive_count} 条 ({positive_count/len(selected_data)*100:.1f}%)")
            print(f"  - other: {negative_count} 条 ({negative_count/len(selected_data)*100:.1f}%)")
        else:
            print(f"警告：文件 {input_file} 不存在")
    
    # LLaMA 真新闻子类
    for subclass in legitimate_subclasses:
        input_file = llama_source_dir / f"alpaca_llama3_{subclass}_legitimate.json"
        output_file = llama_target_dir / f"alpaca_llama3_{subclass}_legitimate_8k.json"
        
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"LLaMA {subclass}_legitimate 文件包含 {len(data)} 条数据")
            
            if len(data) < samples_per_file:
                print(f"警告：LLaMA {subclass}_legitimate 数据量 ({len(data)}) 少于目标数量 ({samples_per_file})")
                selected_data = data
            else:
                selected_data = random.sample(data, samples_per_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(selected_data)} 条 LLaMA {subclass}_legitimate 数据")
            total_llama_sampled += len(selected_data)
            
            # 统计标签分布
            subclass_with_hyphen = subclass.replace("_", "-")  # 转换为连字符格式
            positive_count = sum(1 for item in selected_data if item['output'] == subclass_with_hyphen)
            other_subclass = "integration-based" if subclass == "style_based" else "style-based"
            negative_count = sum(1 for item in selected_data if item['output'] == other_subclass)
            print(f"  - {subclass_with_hyphen}: {positive_count} 条 ({positive_count/len(selected_data)*100:.1f}%)")
            print(f"  - {other_subclass}: {negative_count} 条 ({negative_count/len(selected_data)*100:.1f}%)")
        else:
            print(f"警告：文件 {input_file} 不存在")
    
    print(f"\n=== 采样完成 ===")
    print(f"GLM 总共采样: {total_glm_sampled} 条数据")
    print(f"LLaMA 总共采样: {total_llama_sampled} 条数据")
    print(f"总计: {total_glm_sampled + total_llama_sampled} 条数据")
    
    return glm_target_dir, llama_target_dir

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    glm_dir, llama_dir = sample_data_from_files()
    print(f"\n完成！输出目录:")
    print(f"- GLM: {glm_dir}")
    print(f"- LLaMA: {llama_dir}") 