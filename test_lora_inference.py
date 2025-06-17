#!/usr/bin/env python3
"""
测试LoRA推理配置
"""

import sys
import os
sys.path.insert(0, 'scripts')

from multi_model_lora_inference import get_lora_adapter_path, get_save_path, MODEL_CONFIGS, DATASET_CONFIGS, TASK3_CROSS_DOMAIN_MAPPING

def test_lora_inference_config():
    """测试LoRA推理配置"""
    print("🧪 测试LoRA推理配置生成")
    
    # 测试几个示例组合，包括跨域实验
    test_cases = [
        # Task1 测试
        ("/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct", "task1_full_glm"),
        ("/root/autodl-tmp/models/Qwen1.5-7B", "task1_full_llama"),
        
        # Task3 跨域实验测试
        ("/root/autodl-tmp/models/chatglm3-6b", "task3_full_gossip"),  # 用polifact训练在gossip测试
        ("/root/autodl-tmp/models/chatglm3-6b", "task3_full_polifact"),  # 用gossip训练在polifact测试
        ("/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct", "task3_small_gossip"),
        ("/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct", "task3_small_polifact")
    ]
    
    for model_path, dataset_name in test_cases:
        print(f"\n📝 测试: {os.path.basename(model_path)} + {dataset_name}")
        
        # 获取LoRA适配器路径
        adapter_path = get_lora_adapter_path(model_path, dataset_name)
        print(f"   LoRA适配器: {adapter_path}")
        
        # 获取保存路径
        save_path = get_save_path(model_path, dataset_name)
        print(f"   保存路径: {save_path}")
        
        # 对于task3，解释跨域实验的逻辑
        if "task3" in dataset_name and dataset_name in TASK3_CROSS_DOMAIN_MAPPING:
            train_dataset = TASK3_CROSS_DOMAIN_MAPPING[dataset_name]
            train_type = "polifact" if "polifact" in train_dataset else "gossip"
            test_type = "gossip" if "gossip" in dataset_name else "polifact"
            print(f"   🔄 跨域实验: 使用{train_type}训练的模型在{test_type}数据上测试")
        
        # 检查适配器是否存在
        if os.path.exists(adapter_path):
            print(f"   ✅ LoRA适配器存在")
        else:
            print(f"   ❌ LoRA适配器不存在")

if __name__ == "__main__":
    test_lora_inference_config() 