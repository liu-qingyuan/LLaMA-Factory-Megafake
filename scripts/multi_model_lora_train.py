#!/usr/bin/env python3

import os
import subprocess
import yaml
import json
from pathlib import Path
import datetime
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))


# 模型配置：模型路径 -> (模板名称, trust_remote_code)
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": ("llama3", True),
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True),
    "/root/autodl-tmp/models/Qwen1.5-72B": ("qwen", True),
    "/root/autodl-tmp/models/chatglm3-6b": ("chatglm3", True),
    "/root/autodl-tmp/models/Mistral-7B-v0.1": ("mistral", False),
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": ("baichuan2", True),
}

# 数据集配置
DATASET_CONFIGS = {
    # 旧版大规模实验仍可用：task1_small_*, task3_small_*, ...
    # "task1_full_glm": "task1_full_glm",
    # "task1_full_llama": "task1_full_llama",
    # "task1_small_glm": "task1_small_glm",
    # "task1_small_llama": "task1_small_llama",
    # "task3_full_gossip": "task3_full_gossip",
    # "task3_full_polifact": "task3_full_polifact",
    # "task3_small_gossip": "task3_small_gossip",
    # "task3_small_polifact": "task3_small_polifact",
    # Mini Test100 数据集（100正/100负）用于快速验证整条流水线
    "task1_test200_balanced_glm": "task1_test200_balanced_glm"
}

def get_model_name(model_path):
    """从模型路径提取模型名称"""
    return Path(model_path).name

def get_output_path(model_path, dataset_name):
    """根据模型和数据集生成输出路径"""
    model_name = get_model_name(model_path)
    
    # 根据数据集名称确定任务和类型
    if "task1" in dataset_name:
        task = "task1"
    elif "task2" in dataset_name:
        task = "task2"  
    elif "task3" in dataset_name:
        task = "task3"
    
    # 构建输出路径
    output_path = f"megafakeTasks/{task}/{dataset_name}/{model_name}/lora/sft"
    return output_path

def get_log_path(model_path, dataset_name):
    """生成日志文件路径"""
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{model_name}_LoRA_{dataset_name}_{timestamp}.log"
    return f"logs/{log_filename}"

def create_config_file(model_path, template, trust_remote_code, dataset_name, output_path):
    """创建临时的训练配置文件"""
    model_name = get_model_name(model_path)
    
    # 基础配置模板
    config = {
        # model
        "model_name_or_path": model_path,
        "trust_remote_code": trust_remote_code,
        
        # method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 16,
        "lora_target": "all",
        
        # dataset
        "dataset": dataset_name,
        "template": template,
        "cutoff_len": 2048,
        "max_samples": 8000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        
        # output
        "output_dir": str(REPO_ROOT / output_path),
        "logging_steps": 10,
        "save_steps": 1000,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        
        # train
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": False,
        "fp16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None
    }
    
    # 根据模型类型决定是否启用Flash Attention 2.0
    # 支持FA2的模型：LLaMA系列、Qwen系列
    if any(name in model_name for name in ["Llama", "llama", "Qwen", "qwen"]):
        config["flash_attn"] = "fa2"
        print(f"✅ 为模型 {model_name} 启用 Flash Attention 2.0")
    else:
        # Baichuan、ChatGLM等可能不支持FA2
        print(f"⚠️  模型 {model_name} 不启用 Flash Attention 2.0")
    
    # 为某些模型调整特殊配置
    if "Baichuan" in model_name:
        # Baichuan模型的特殊配置
        pass
    elif "chatglm" in model_name.lower():
        # ChatGLM模型的特殊配置
        pass
    elif "Mistral" in model_name:
        # Mistral模型的特殊配置
        pass
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return f.name

def run_training(model_path, template, trust_remote_code, dataset_name, output_path):
    """运行单个训练任务"""
    
    # 创建临时配置文件
    config_file = create_config_file(model_path, template, trust_remote_code, dataset_name, output_path)
    
    try:
        # 构建包含环境配置的命令
        cmd = [
            "bash", "-c", 
            "export HF_ENDPOINT=https://hf-mirror.com && "
            "source /etc/network_turbo 2>/dev/null || true && "
            f"llamafactory-cli train {config_file}"
        ]
        
        # 创建日志目录和输出目录
        log_path = get_log_path(model_path, dataset_name)
        full_output_path = str(REPO_ROOT / output_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(full_output_path, exist_ok=True)
        
        print(f"运行命令: {' '.join(cmd)}")
        print(f"配置文件: {config_file}")
        print(f"日志文件: {log_path}")
        
        # 打开日志文件，同时输出到控制台和文件
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 记录命令和时间戳
            log_file.write(f"开始时间: {datetime.datetime.now()}\n")
            log_file.write(f"执行命令: {' '.join(cmd)}\n")
            log_file.write(f"配置文件: {config_file}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()
            
            # 使用 Popen 来实时输出日志
            process = subprocess.Popen(
                cmd, 
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
                print(f"✅ 训练成功: {get_model_name(model_path)} + {dataset_name}")
                print(f"   保存至: {output_path}")
                print(f"   日志至: {log_path}")
                return True
            else:
                print(f"❌ 训练失败: {get_model_name(model_path)} + {dataset_name}")
                print(f"   返回码: {return_code}")
                print(f"   日志文件: {log_path}")
                return False
                
    except Exception as e:
        print(f"❌ 执行异常: {get_model_name(model_path)} + {dataset_name}")
        print(f"   异常信息: {e}")
        return False
    finally:
        # 清理临时配置文件
        try:
            os.unlink(config_file)
        except:
            pass

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

def check_training_completed(output_path):
    """检查训练是否已经完成"""
    # 检查是否存在训练完成的标志文件
    adapter_config = os.path.join(output_path, "adapter_config.json")
    adapter_model = os.path.join(output_path, "adapter_model.safetensors")
    
    return os.path.exists(adapter_config) and os.path.exists(adapter_model)

def main():
    """主函数"""
    print("🚀 开始批量Lora模型训练任务")
    print(f"📊 总计 {len(MODEL_CONFIGS)} 个模型，{len(DATASET_CONFIGS)} 个数据集")
    print(f"🎯 总任务数: {len(MODEL_CONFIGS) * len(DATASET_CONFIGS)}")

    # 统计信息
    total_tasks = len(MODEL_CONFIGS) * len(DATASET_CONFIGS)
    completed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    
    # 遍历所有模型和数据集组合
    for model_path, (template, trust_remote_code) in MODEL_CONFIGS.items():
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        
        # 检查模型路径是否存在
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(DATASET_CONFIGS)
            continue
        
        for dataset_key, dataset_name in DATASET_CONFIGS.items():
            output_path = get_output_path(model_path, dataset_key)
            full_output_path = f"/root/autodl-tmp/LLaMA-Factory-Megafake2/{output_path}"
            
            # 检查是否已完成训练
            if check_training_completed(full_output_path):
                print(f"⏭️  训练已完成，跳过: {model_name} + {dataset_key}")
                skipped_tasks += 1
                continue
            
            # 运行训练
            print(f"🎯 开始训练: {model_name} + {dataset_key}")
            success = run_training(model_path, template, trust_remote_code, dataset_name, output_path)
            
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1
            
            print(f"📈 进度: {completed_tasks + failed_tasks + skipped_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 跳过: {skipped_tasks}, 失败: {failed_tasks})")
    
    print(f"\n🎉 批量训练任务完成!")
    print(f"📊 总结: {total_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"⏭️  跳过: {skipped_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")

if __name__ == "__main__":
    main() 
