#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_data_from_files():
    """从两个 JSON 文件中分别随机选择 8000 条数据并保存到新目录"""
    
    # 设置路径
    source_dir = Path("data/data_table/task3/alpaca_full")
    target_dir = Path("data/data_table/task3/small_8k")
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件路径
    gossip_file = source_dir / "alpaca_chatglm_gossip_binary.json"
    polifact_file = source_dir / "alpaca_chatglm_polifact_binary.json"
    
    print(f"正在读取数据文件...")
    
    # 读取Gossip文件的数据并随机选择8k条
    with open(gossip_file, 'r', encoding='utf-8') as f:
        gossip_data = json.load(f)
    print(f"Gossip 文件包含 {len(gossip_data)} 条数据")
    
    target_count = 8000
    if len(gossip_data) < target_count:
        print(f"警告：Gossip 数据量 ({len(gossip_data)}) 少于目标数量 ({target_count})")
        gossip_selected = gossip_data
    else:
        print(f"正在从 Gossip 文件随机选择 {target_count} 条数据...")
        gossip_selected = random.sample(gossip_data, target_count)
    
    # 保存Gossip数据
    gossip_output_file = target_dir / "alpaca_chatglm_gossip_8k.json"
    with open(gossip_output_file, 'w', encoding='utf-8') as f:
        json.dump(gossip_selected, f, ensure_ascii=False, indent=2)
    
    print(f"已成功保存 {len(gossip_selected)} 条 Gossip 数据到 {gossip_output_file}")
    
    # Gossip数据统计
    gossip_legitimate = sum(1 for item in gossip_selected if item['output'] == 'legitimate')
    gossip_fake = sum(1 for item in gossip_selected if item['output'] == 'fake')
    
    print(f"Gossip 数据统计:")
    print(f"- legitimate: {gossip_legitimate} 条 ({gossip_legitimate/len(gossip_selected)*100:.1f}%)")
    print(f"- fake: {gossip_fake} 条 ({gossip_fake/len(gossip_selected)*100:.1f}%)")
    
    # 读取Polifact文件的数据并随机选择8k条
    with open(polifact_file, 'r', encoding='utf-8') as f:
        polifact_data = json.load(f)
    print(f"\nPolifact 文件包含 {len(polifact_data)} 条数据")
    
    if len(polifact_data) < target_count:
        print(f"警告：Polifact 数据量 ({len(polifact_data)}) 少于目标数量 ({target_count})")
        polifact_selected = polifact_data
    else:
        print(f"正在从 Polifact 文件随机选择 {target_count} 条数据...")
        polifact_selected = random.sample(polifact_data, target_count)
    
    # 保存Polifact数据
    polifact_output_file = target_dir / "alpaca_chatglm_polifact_8k.json"
    with open(polifact_output_file, 'w', encoding='utf-8') as f:
        json.dump(polifact_selected, f, ensure_ascii=False, indent=2)
    
    print(f"已成功保存 {len(polifact_selected)} 条 Polifact 数据到 {polifact_output_file}")
    
    # Polifact数据统计
    polifact_legitimate = sum(1 for item in polifact_selected if item['output'] == 'legitimate')
    polifact_fake = sum(1 for item in polifact_selected if item['output'] == 'fake')
    
    print(f"Polifact 数据统计:")
    print(f"- legitimate: {polifact_legitimate} 条 ({polifact_legitimate/len(polifact_selected)*100:.1f}%)")
    print(f"- fake: {polifact_fake} 条 ({polifact_fake/len(polifact_selected)*100:.1f}%)")
    
    return gossip_output_file, polifact_output_file

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    gossip_file, polifact_file = sample_data_from_files()
    print(f"\n完成！输出文件:")
    print(f"- Gossip: {gossip_file}")
    print(f"- Polifact: {polifact_file}") 