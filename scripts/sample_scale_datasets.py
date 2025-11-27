#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_and_register_datasets():
    """
    从全量数据中采样不同规模的数据集 (1k, 2k, 5k, 10k, 20k)
    并自动注册到 dataset_info.json
    """
    
    # 配置
    SOURCE_KEY = "task1_full_glm"
    TARGET_SCALES = [1000, 2000, 5000, 10000, 20000]
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_INFO_PATH = REPO_ROOT / "data" / "dataset_info.json"
    
    # 1. 读取 dataset_info.json 获取源文件路径
    with open(DATA_INFO_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
        
    if SOURCE_KEY not in dataset_info:
        print(f"❌ 源数据集 {SOURCE_KEY} 未在 dataset_info.json 中找到")
        return

    source_rel_path = dataset_info[SOURCE_KEY]["file_name"]
    source_full_path = REPO_ROOT / "data" / source_rel_path
    
    if not source_full_path.exists():
        print(f"❌ 源文件不存在: {source_full_path}")
        return

    print(f"📖 正在读取源文件: {source_full_path}")
    with open(source_full_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    total_count = len(full_data)
    print(f"📊 全量数据总数: {total_count}")
    
    # 2. 分离正负样本以确保平衡
    legitimate_samples = [x for x in full_data if x.get('output') == 'legitimate']
    fake_samples = [x for x in full_data if x.get('output') == 'fake']
    
    print(f"   - Legitimate: {len(legitimate_samples)}")
    print(f"   - Fake: {len(fake_samples)}")
    
    # 3. 循环生成各规模数据集
    output_base_dir = REPO_ROOT / "data" / "data_table" / "task1" / "scale_experiment"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(42) # 固定种子
    
    new_registrations = {}
    
    for scale in TARGET_SCALES:
        if scale > total_count:
            print(f"⚠️  跳过规模 {scale}: 超过全量数据 ({total_count})")
            continue
            
        # 尝试平衡采样
        half_scale = scale // 2
        if len(legitimate_samples) >= half_scale and len(fake_samples) >= half_scale:
            # 可以完全平衡
            selected_leg = random.sample(legitimate_samples, half_scale)
            selected_fake = random.sample(fake_samples, half_scale)
            selected_data = selected_leg + selected_fake
        else:
            # 无法完全平衡，退化为随机采样
            print(f"⚠️  规模 {scale} 无法完全平衡 (正/负样本不足 {half_scale})，采用全局随机采样")
            selected_data = random.sample(full_data, scale)
            
        random.shuffle(selected_data) # 打乱顺序
        
        # 生成文件名和 Key
        filename = f"alpaca_megafake_glm_{scale}.json"
        dataset_key = f"task1_scale_{scale}_glm"
        output_path = output_base_dir / filename
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=2)
            
        # 记录相对路径用于注册
        rel_path = str(output_path.relative_to(REPO_ROOT / "data"))
        dataset_info[dataset_key] = {"file_name": rel_path}
        new_registrations[dataset_key] = rel_path
        
        # 统计当前数据集
        curr_leg = sum(1 for x in selected_data if x.get('output') == 'legitimate')
        curr_fake = sum(1 for x in selected_data if x.get('output') == 'fake')
        print(f"✅ 生成 {dataset_key}: {len(selected_data)} 条 (Leg: {curr_leg}, Fake: {curr_fake}) -> {rel_path}")

    # 4. 更新 dataset_info.json
    print(f"💾 更新 dataset_info.json ...")
    with open(DATA_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
    print("🎉 所有大规模数据集已生成并注册。")
    print("👉 现在可以使用以下 key 进行实验:")
    for key in new_registrations:
        print(f"   - {key}")

if __name__ == "__main__":
    sample_and_register_datasets()
