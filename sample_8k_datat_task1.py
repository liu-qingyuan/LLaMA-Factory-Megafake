#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_data_from_files():
    """从两个 JSON 文件中分别随机选择 8000 条数据并保存到新目录"""
    
    # 设置路径
    source_dir = Path("data/data_table/task1/alpaca_full")
    target_dir = Path("data/data_table/task1/small_8k")
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件路径
    glm_file = source_dir / "alpaca_megafake_glm_binary.json"
    llama_file = source_dir / "alpaca_megafake_llama_binary.json"
    
    print(f"正在读取数据文件...")
    
    # 读取GLM文件的数据并随机选择8k条
    with open(glm_file, 'r', encoding='utf-8') as f:
        glm_data = json.load(f)
    print(f"GLM 文件包含 {len(glm_data)} 条数据")
    
    target_count = 8000
    if len(glm_data) < target_count:
        print(f"警告：GLM 数据量 ({len(glm_data)}) 少于目标数量 ({target_count})")
        glm_selected = glm_data
    else:
        print(f"正在从 GLM 文件随机选择 {target_count} 条数据...")
        glm_selected = random.sample(glm_data, target_count)
    
    # 保存GLM数据
    glm_output_file = target_dir / "alpaca_megafake_glm_8k.json"
    with open(glm_output_file, 'w', encoding='utf-8') as f:
        json.dump(glm_selected, f, ensure_ascii=False, indent=2)
    
    print(f"已成功保存 {len(glm_selected)} 条 GLM 数据到 {glm_output_file}")
    
    # GLM数据统计
    glm_legitimate = sum(1 for item in glm_selected if item['output'] == 'legitimate')
    glm_fake = sum(1 for item in glm_selected if item['output'] == 'fake')
    
    print(f"GLM 数据统计:")
    print(f"- legitimate: {glm_legitimate} 条 ({glm_legitimate/len(glm_selected)*100:.1f}%)")
    print(f"- fake: {glm_fake} 条 ({glm_fake/len(glm_selected)*100:.1f}%)")
    
    # 读取LLaMA文件的数据并随机选择8k条
    with open(llama_file, 'r', encoding='utf-8') as f:
        llama_data = json.load(f)
    print(f"\nLLaMA 文件包含 {len(llama_data)} 条数据")
    
    if len(llama_data) < target_count:
        print(f"警告：LLaMA 数据量 ({len(llama_data)}) 少于目标数量 ({target_count})")
        llama_selected = llama_data
    else:
        print(f"正在从 LLaMA 文件随机选择 {target_count} 条数据...")
        llama_selected = random.sample(llama_data, target_count)
    
    # 保存LLaMA数据
    llama_output_file = target_dir / "alpaca_megafake_llama_8k.json"
    with open(llama_output_file, 'w', encoding='utf-8') as f:
        json.dump(llama_selected, f, ensure_ascii=False, indent=2)
    
    print(f"已成功保存 {len(llama_selected)} 条 LLaMA 数据到 {llama_output_file}")
    
    # LLaMA数据统计
    llama_legitimate = sum(1 for item in llama_selected if item['output'] == 'legitimate')
    llama_fake = sum(1 for item in llama_selected if item['output'] == 'fake')
    
    print(f"LLaMA 数据统计:")
    print(f"- legitimate: {llama_legitimate} 条 ({llama_legitimate/len(llama_selected)*100:.1f}%)")
    print(f"- fake: {llama_fake} 条 ({llama_fake/len(llama_selected)*100:.1f}%)")
    
    return glm_output_file, llama_output_file

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    glm_file, llama_file = sample_data_from_files()
    print(f"\n完成！输出文件:")
    print(f"- GLM: {glm_file}")
    print(f"- LLaMA: {llama_file}") 