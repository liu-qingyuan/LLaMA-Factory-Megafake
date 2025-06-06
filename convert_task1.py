#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_task1_to_alpaca(input_file, output_file):
    """
    将task1的二分类JSON文件转换为alpaca格式
    """
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    alpaca_data = []
    for item in data:
        text = item['text']
        label = item['label']
        
        # 根据标签创建指令输出
        if label == 1:
            output = "fake"
        else:
            output = "legitimate"
        
        alpaca_item = {
            "instruction": f"Identify whether the news is legitimate or fake in one word: {text}",
            "input": "",
            "output": "legitimate" if output == "legitimate" else "fake"
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
    
    # Task 1 转换
    print("=== 开始转换 Task 1 ===")
    task1_dir = base_dir / "data_table" / "task1" / "full"
    task1_output_dir = base_dir / "data_table" / "task1" / "alpaca_full"
    
    total_converted = 0
    
    # 转换task1的文件
    for json_file in task1_dir.glob("*.json"):
        if json_file.name != "README.md":  # 跳过README文件
            output_file = task1_output_dir / f"alpaca_{json_file.name}"
            count = convert_task1_to_alpaca(str(json_file), str(output_file))
            total_converted += count
    
    print(f"\n=== Task 1 转换完成！总共转换了 {total_converted} 条数据 ===")

if __name__ == "__main__":
    main() 