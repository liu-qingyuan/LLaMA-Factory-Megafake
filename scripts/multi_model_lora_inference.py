#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path
import datetime

# 模型配置：模型路径 -> 模板名称
MODEL_CONFIGS = {
    # "/root/autodl-tmp/models/Baichuan2-7B-Chat": "baichuan2",
    "/root/autodl-tmp/models/chatglm3-6b": "chatglm3", 
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": "llama3",
    # "/root/autodl-tmp/models/Mistral-7B-v0.1": "mistral",
    "/root/autodl-tmp/models/Qwen1.5-7B": "qwen"
}

# 数据集配置
DATASET_CONFIGS = {
    "task1_full_glm": "data_table/task1/alpaca_full/alpaca_megafake_glm_binary.json",
    "task1_full_llama": "data_table/task1/alpaca_full/alpaca_megafake_llama_binary.json", 
    # "task1_small_glm": "data_table/task1/small_8k/alpaca_megafake_glm_8k.json",
    # "task1_small_llama": "data_table/task1/small_8k/alpaca_megafake_llama_8k.json",
    "task3_full_gossip": "data_table/task3/alpaca_full/alpaca_chatglm_gossip_binary.json",
    "task3_full_polifact": "data_table/task3/alpaca_full/alpaca_chatglm_polifact_binary.json",
    # "task3_small_gossip": "data_table/task3/small_8k/alpaca_chatglm_gossip_8k.json",
    # "task3_small_polifact": "data_table/task3/small_8k/alpaca_chatglm_polifact_8k.json"
}

# Task3跨域实验映射：推理数据集 -> 训练数据集
TASK3_CROSS_DOMAIN_MAPPING = {
    "task3_full_gossip": "task3_small_polifact",    # 在gossip上推理，使用polifact训练的模型
    "task3_full_polifact": "task3_small_gossip",    # 在polifact上推理，使用gossip训练的模型
    "task3_small_gossip": "task3_small_polifact",   # 在small gossip上推理，使用small polifact训练的模型
    "task3_small_polifact": "task3_small_gossip"    # 在small polifact上推理，使用small gossip训练的模型
}

def get_model_name(model_path):
    """从模型路径提取模型名称"""
    return Path(model_path).name

def get_lora_adapter_path(model_path, dataset_name):
    """根据模型和数据集生成LoRA适配器路径"""
    model_name = get_model_name(model_path)
    
    # 根据数据集名称确定任务类型
    if "task1" in dataset_name:
        task = "task1"
        # Task1: 所有LoRA模型都是用small数据集训练的
        if "full" in dataset_name:
            train_dataset = dataset_name.replace("full", "small")
        else:
            train_dataset = dataset_name
    elif "task2" in dataset_name:
        task = "task2"
        # Task2: 所有LoRA模型都是用small数据集训练的
        if "full" in dataset_name:
            train_dataset = dataset_name.replace("full", "small")
        else:
            train_dataset = dataset_name
    elif "task3" in dataset_name:
        task = "task3"
        # Task3: 跨域实验，使用映射表
        train_dataset = TASK3_CROSS_DOMAIN_MAPPING.get(dataset_name, dataset_name)
    
    adapter_path = f"/root/autodl-tmp/LLaMA-Factory-Megafake2/megafakeTasks/{task}/{train_dataset}/{model_name}/lora/sft"
    return adapter_path

def get_save_path(model_path, dataset_name):
    """根据模型和数据集生成保存路径"""
    model_name = get_model_name(model_path)
    
    # 根据数据集名称确定任务和类型
    if "task1" in dataset_name:
        task = "task1"
        if "full" in dataset_name:
            size = "full"
        else:
            size = "small"
        
        if "glm" in dataset_name:
            data_type = "megafake_glm_binary"
        else:
            data_type = "megafake_llama_binary"
            
    elif "task3" in dataset_name:
        task = "task3"
        if "full" in dataset_name:
            size = "full"
        else:
            size = "small"
            
        if "gossip" in dataset_name:
            data_type = "chatglm_gossip_binary"
        else:
            data_type = "chatglm_polifact_binary"
    
    # 修改保存路径，包含LoRA标识
    # 对于task3跨域实验，需要在文件名中体现训练和测试数据集
    if task == "task3" and dataset_name in TASK3_CROSS_DOMAIN_MAPPING:
        train_dataset = TASK3_CROSS_DOMAIN_MAPPING[dataset_name]
        train_type = "polifact" if "polifact" in train_dataset else "gossip"
        save_path = f"megafakeTasks/{task}/{size}/result_{dataset_name}_{model_name}_LoRA_trained_on_{train_type}.jsonl"
    else:
        save_path = f"megafakeTasks/{task}/{size}/result_{dataset_name}_{model_name}_LoRA.jsonl"
    return save_path

def get_log_path(model_path, dataset_name):
    """生成日志文件路径"""
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inference_LoRA_{model_name}_{dataset_name}_{timestamp}.log"
    return f"logs/{log_filename}"

def run_inference(model_path, template, dataset_name, save_path, max_new_tokens=10):
    """运行单个推理任务"""
    # 获取LoRA适配器路径
    adapter_path = get_lora_adapter_path(model_path, dataset_name)
    
    cmd = [
        "python", "scripts/vllm_infer.py",
        "--model_name_or_path", model_path,
        "--adapter_name_or_path", adapter_path,
        "--template", template,
        "--dataset", dataset_name,
        "--save_name", save_path,
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", "0.1",  # 降低温度以获得更稳定的输出
        "--top_p", "0.9",
        "--batch_size", "1024"  
    ]
    
    # 为某些模型添加 trust_remote_code 参数
    model_name = get_model_name(model_path)
    if "Baichuan" in model_name or "chatglm" in model_name.lower():
        cmd.append("--trust_remote_code")
    
    # 检查LoRA适配器是否存在
    if not os.path.exists(adapter_path):
        print(f"❌ LoRA适配器不存在: {adapter_path}")
        return False
    
    # 创建日志目录
    log_path = get_log_path(model_path, dataset_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"LoRA适配器: {adapter_path}")
    print(f"日志文件: {log_path}")
    
    try:
        # 构建包含环境配置的命令
        env_cmd = [
            "bash", "-c", 
            "export HF_ENDPOINT=https://hf-mirror.com && "
            "source /etc/network_turbo 2>/dev/null || true && "
            f"{' '.join(cmd)}"
        ]
        
        # 打开日志文件，同时输出到控制台和文件
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 记录命令和时间戳
            log_file.write(f"开始时间: {datetime.datetime.now()}\n")
            log_file.write(f"执行命令: {' '.join(cmd)}\n")
            log_file.write(f"LoRA适配器: {adapter_path}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()
            
            # 使用 Popen 来实时输出日志
            process = subprocess.Popen(
                env_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出并同时写入文件和控制台
            for line in process.stdout:
                print(line, end='')  # 输出到控制台
                log_file.write(line)  # 写入文件
                log_file.flush()
            
            # 等待进程结束
            return_code = process.wait()
            
            # 记录结束时间
            log_file.write("=" * 80 + "\n")
            log_file.write(f"结束时间: {datetime.datetime.now()}\n")
            log_file.write(f"返回码: {return_code}\n")
            
            if return_code == 0:
                print(f"✅ 成功完成: {get_model_name(model_path)} + {dataset_name}")
                print(f"   保存至: {save_path}")
                print(f"   日志至: {log_path}")
                return True
            else:
                print(f"❌ 失败: {get_model_name(model_path)} + {dataset_name}")
                print(f"   返回码: {return_code}")
                print(f"   日志文件: {log_path}")
                return False
                
    except Exception as e:
        print(f"❌ 执行异常: {get_model_name(model_path)} + {dataset_name}")
        print(f"   异常信息: {e}")
        return False

def check_model_exists(model_path):
    """检查模型是否存在"""
    if not os.path.exists(model_path):
        return False
    
    # 检查是否有必要的配置文件
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        print(f"⚠️  模型配置文件不存在: {config_file}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 开始多模型LoRA推理任务")
    print(f"📊 总计 {len(MODEL_CONFIGS)} 个模型，{len(DATASET_CONFIGS)} 个数据集")
    print(f"🎯 总任务数: {len(MODEL_CONFIGS) * len(DATASET_CONFIGS)}")

    # 统计信息
    total_tasks = len(MODEL_CONFIGS) * len(DATASET_CONFIGS)
    completed_tasks = 0
    failed_tasks = 0
    
    # 遍历所有模型和数据集组合
    for model_path, template in MODEL_CONFIGS.items():
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        
        # 检查模型路径是否存在
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(DATASET_CONFIGS)
            continue
        
        for dataset_name in DATASET_CONFIGS.keys():
            save_path = get_save_path(model_path, dataset_name)
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查是否已存在结果文件
            if os.path.exists(save_path):
                print(f"⏭️  结果文件已存在，跳过: {save_path}")
                completed_tasks += 1
                continue
            
            # 运行LoRA推理
            print(f"🎯 开始LoRA推理: {model_name} + {dataset_name}")
            success = run_inference(model_path, template, dataset_name, save_path)
            
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1
            
            print(f"📈 进度: {completed_tasks + failed_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 失败: {failed_tasks})")
    
    print(f"\n🎉 LoRA推理任务完成!")
    print(f"📊 总结: {total_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")

if __name__ == "__main__":
    main() 