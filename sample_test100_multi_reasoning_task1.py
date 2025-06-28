#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_balanced_data(data, num_positive=50, num_negative=50):
    """
    从数据中抽取平衡的正负样本
    """
    # 分离正负样本
    positive_samples = []
    negative_samples = []
    
    for item in data:
        output = item['output'].strip().lower()
        
        # 更精确的判断逻辑，优先检查Final Answer
        if '**final answer:** legitimate' in output or output.endswith('legitimate'):
            positive_samples.append(item)
        elif '**final answer:** fake' in output or output.endswith('fake'):
            negative_samples.append(item)
        elif 'legitimate' in output and 'fake' not in output:
            positive_samples.append(item)
        elif 'fake' in output and 'legitimate' not in output:
            negative_samples.append(item)
    
    print(f"  可用正例: {len(positive_samples)}, 可用负例: {len(negative_samples)}")
    
    # 随机抽样
    selected_positive = random.sample(positive_samples, min(num_positive, len(positive_samples)))
    selected_negative = random.sample(negative_samples, min(num_negative, len(negative_samples)))
    
    # 合并并打乱
    selected_data = selected_positive + selected_negative
    random.shuffle(selected_data)
    
    return selected_data

def sample_test100_from_reasoning_configs():
    """
    从三种推理配置的数据集中分别抽取100条测试数据
    """
    base_dir = Path("data/data_table/task1")
    
    # 定义三种推理配置
    reasoning_configs = [
        {
            "name": "CoT_SC",
            "source_dir": base_dir / "alpaca_full_CoT_SC",
            "target_dir": base_dir / "alpaca_test100_CoT_SC",
            "description": "Chain-of-Thought + Self-Consistency"
        },
        {
            "name": "FS_5", 
            "source_dir": base_dir / "alpaca_full_FS_5",
            "target_dir": base_dir / "alpaca_test100_FS_5",
            "description": "Few-shot with 5 examples"
        },
        {
            "name": "ZS_DF",
            "source_dir": base_dir / "alpaca_full_ZS_DF", 
            "target_dir": base_dir / "alpaca_test100_ZS_DF",
            "description": "Zero-shot + Decomposition"
        }
    ]
    
    # 设置随机种子确保可重现
    random.seed(42)
    
    print("=== 开始从多种推理配置中抽取测试数据 ===\n")
    
    for config in reasoning_configs:
        print(f"处理 {config['name']} ({config['description']}):")
        
        # 创建目标目录
        config["target_dir"].mkdir(parents=True, exist_ok=True)
        
        # 处理该配置下的所有JSON文件
        source_files = list(config["source_dir"].glob("*.json"))
        
        if not source_files:
            print(f"  警告: 在 {config['source_dir']} 中未找到JSON文件")
            continue
            
        for source_file in source_files:
            print(f"  处理文件: {source_file.name}")
            
            # 读取源数据
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  原始数据量: {len(data)}")
            
            # 抽取平衡样本
            selected_data = sample_balanced_data(data, num_positive=50, num_negative=50)
            
            # 生成输出文件名
            output_filename = f"test100_{source_file.name}"
            output_file = config["target_dir"] / output_filename
            
            # 保存数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
            print(f"  已保存 {len(selected_data)} 条测试数据到: {output_file}")
            
            # 统计信息
            positive_count = 0
            negative_count = 0
            for item in selected_data:
                output = item['output'].lower()
                if '**final answer:** legitimate' in output or output.endswith('legitimate'):
                    positive_count += 1
                elif '**final answer:** fake' in output or output.endswith('fake'):
                    negative_count += 1
            
            print(f"  数据分布: legitimate={positive_count}, fake={negative_count}")
            print()
    
    print("=== 所有推理配置的测试数据抽取完成 ===")
    
    # 打印总结
    print("\n=== 生成的测试数据集总结 ===")
    for config in reasoning_configs:
        print(f"{config['name']}: {config['target_dir']}")
        test_files = list(config["target_dir"].glob("*.json"))
        for test_file in test_files:
            file_size = test_file.stat().st_size / 1024  # KB
            print(f"  - {test_file.name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    sample_test100_from_reasoning_configs() 