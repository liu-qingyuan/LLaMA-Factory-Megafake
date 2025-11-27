#!/usr/bin/env python3

import os
import subprocess
import yaml
import json
from pathlib import Path
import datetime
import tempfile
import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

SA_ROOT = REPO_ROOT / "sensitivity_analysis"
SA_LOG_ROOT = SA_ROOT / "logs" / "train"
SA_OUTPUT_ROOT = SA_ROOT / "outputs"
LEGACY_OUTPUT_ROOT = REPO_ROOT / "megafakeTasks"
SA_LOG_ROOT.mkdir(parents=True, exist_ok=True)
SA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_INFO_PATH = REPO_ROOT / "data" / "dataset_info.json"
if DATASET_INFO_PATH.exists():
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        DATASET_INFO = json.load(f)
else:
    DATASET_INFO = {}


# 模型配置：模型路径 -> (模板名称, trust_remote_code)
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": ("llama3", True),
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True),
    "/root/autodl-tmp/models/chatglm3-6b": ("chatglm3", True),
    "/root/autodl-tmp/models/Mistral-7B-v0.1": ("mistral", False),
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": ("baichuan2", True),
}
MODEL_NAME_MAP = {Path(path).name: path for path in MODEL_CONFIGS.keys()}

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
    "task1_test200_balanced_glm": "task1_test200_balanced_glm",
    # 大规模实验数据集 (1k - 20k)
    "task1_scale_1000_glm": "task1_scale_1000_glm",
    "task1_scale_2000_glm": "task1_scale_2000_glm",
    "task1_scale_5000_glm": "task1_scale_5000_glm",
    "task1_scale_10000_glm": "task1_scale_10000_glm",
    "task1_scale_20000_glm": "task1_scale_20000_glm"
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
    # output_path = f"megafakeTasks/{task}/{dataset_name}/{model_name}/lora/sft"
    output_path = SA_OUTPUT_ROOT / task / dataset_name / model_name / "lora" / "sft"
    return output_path

def get_log_path(model_path, dataset_name):
    """生成日志文件路径"""
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{model_name}_LoRA_{dataset_name}_{timestamp}.log"
    # return f"logs/{log_filename}"
    return SA_LOG_ROOT / log_filename

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
        "output_dir": str(output_path),
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
    model_name = get_model_name(model_path)
    
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
        full_output_path = output_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        full_output_path.mkdir(parents=True, exist_ok=True)
        
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
                print(f"✅ 训练成功: {model_name} + {dataset_name}")
                print(f"   保存至: {full_output_path.resolve()}")
                print(f"   日志至: {log_path.resolve()}")
                next_cmd = (
                    "python scripts/multi_model_lora_inference.py "
                    f"--models {model_name} --datasets {dataset_name}"
                )
                print(f"👉 下一步: {next_cmd}")
                return True, str(log_path.resolve())
            else:
                print(f"❌ 训练失败: {model_name} + {dataset_name}")
                print(f"   返回码: {return_code}")
                print(f"   日志文件: {log_path.resolve()}")
                return False, str(log_path.resolve())
                
    except Exception as e:
        print(f"❌ 执行异常: {model_name} + {dataset_name}")
        print(f"   异常信息: {e}")
        return False, None
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
    adapter_config = Path(output_path) / "adapter_config.json"
    adapter_model = Path(output_path) / "adapter_model.safetensors"
    if adapter_config.exists() and adapter_model.exists():
        return True
    try:
        relative = Path(output_path).relative_to(SA_OUTPUT_ROOT)
    except ValueError:
        return False
    legacy_path = LEGACY_OUTPUT_ROOT / relative
    legacy_config = legacy_path / "adapter_config.json"
    legacy_model = legacy_path / "adapter_model.safetensors"
    return legacy_config.exists() and legacy_model.exists()

def resolve_dataset_file(dataset_key):
    """解析数据集文件路径"""
    info = DATASET_INFO.get(dataset_key)
    if not info:
        return None
    file_name = info.get("file_name")
    if not file_name:
        return None
    return REPO_ROOT / "data" / file_name

def dry_run_check(model_path, dataset_key, dataset_name):
    """Dry-run 检查"""
    status_ok = True
    model_exists = check_model_exists(model_path)
    dataset_file = resolve_dataset_file(dataset_key)
    dataset_exists = dataset_file.exists() if dataset_file else False
    output_path = get_output_path(model_path, dataset_key)
    output_parent = output_path.parent
    print(f"\n[Dry-Run] 模型: {model_path}")
    print(f"[Dry-Run] 数据集ID: {dataset_key} ({dataset_name})")
    if dataset_file:
        print(f"[Dry-Run] 数据文件: {dataset_file} {'✅' if dataset_exists else '❌'}")
    else:
        print(f"[Dry-Run] 数据文件: 未在 dataset_info.json 中找到条目 ❌")
    print(f"[Dry-Run] 输出目录: {output_path.resolve()}")
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[Dry-Run] 无法创建输出目录: {exc}")
        status_ok = False
    print(f"[Dry-Run] 模型检查: {'✅' if model_exists else '❌'}")
    if not dataset_exists:
        status_ok = False
    if not model_exists:
        status_ok = False
    return status_ok

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量运行 LoRA 训练（支持 Dry-Run）")
    parser.add_argument("--models", nargs="+", help="指定模型名称（如 Qwen1.5-7B）或绝对路径")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                        help="指定数据集键（默认全量）")
    parser.add_argument("--limit", type=int, help="限制任务数量")
    parser.add_argument("--dry-run", action="store_true", help="仅检查路径与依赖，不实际训练")
    parser.add_argument("--include-large-models", action="store_true", help="允许运行 70B+ 大模型")
    return parser.parse_args()

def select_models(model_filters, include_large=False):
    """根据参数筛选模型"""
    if not model_filters:
        items = list(MODEL_CONFIGS.items())
    else:
        items = []
        for item in model_filters:
            if item in MODEL_CONFIGS:
                items.append((item, MODEL_CONFIGS[item]))
                continue
            if item in MODEL_NAME_MAP:
                path = MODEL_NAME_MAP[item]
                items.append((path, MODEL_CONFIGS[path]))
                continue
            resolved = Path(item).expanduser()
            if str(resolved) in MODEL_CONFIGS:
                items.append((str(resolved), MODEL_CONFIGS[str(resolved)]))
            else:
                print(f"⚠️  未识别的模型: {item}")
    
    # 过滤大模型
    final_selection = []
    for path, config in items:
        if "72B" in str(path) or "70B" in str(path):
            if not include_large:
                print(f"⚠️  跳过大模型 (需 --include-large-models): {path}")
                continue
            else:
                print(f"⚠️  包含大模型 (OOM风险): {path}")
        final_selection.append((path, config))
        
    return final_selection

def select_datasets(dataset_filters):
    """根据参数筛选数据集"""
    if not dataset_filters:
        return list(DATASET_CONFIGS.items())
    selected = []
    for key in dataset_filters:
        name = DATASET_CONFIGS.get(key)
        if name:
            selected.append((key, name))
        else:
            print(f"⚠️  未识别的数据集: {key}")
    return selected

def main():
    """主函数"""
    args = parse_args()
    selected_models = select_models(args.models, args.include_large_models)
    selected_datasets = select_datasets(args.datasets)
    total_tasks = len(selected_models) * len(selected_datasets)
    if args.limit:
        total_tasks = min(total_tasks, args.limit)
    print("🚀 开始批量Lora模型训练任务")
    print(f"📊 选中 {len(selected_models)} 个模型，{len(selected_datasets)} 个数据集")
    print(f"🎯 计划任务数: {total_tasks}")

    if args.dry_run:
        issues = False
        processed = 0
        for model_path, (template, trust_remote_code) in selected_models:
            for dataset_key, dataset_name in selected_datasets:
                if args.limit and processed >= args.limit:
                    break
                processed += 1
                ok = dry_run_check(model_path, dataset_key, dataset_name)
                if not ok:
                    issues = True
            if args.limit and processed >= args.limit:
                break
        if issues:
            print("\n⚠️  Dry-Run 检测到问题，请先修复再运行正式训练")
            return
        print("\n✅ Dry-Run 检查通过，可安全启动训练")
        return

    completed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    processed_tasks = 0
    artifact_records = []
    for model_path, (template, trust_remote_code) in selected_models:
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(selected_datasets)
            continue
        for dataset_key, dataset_name in selected_datasets:
            if args.limit and processed_tasks >= args.limit:
                break
            processed_tasks += 1
            output_path = get_output_path(model_path, dataset_key)
            full_output_path = str(output_path)
            if check_training_completed(full_output_path):
                print(f"⏭️  训练已完成，跳过: {model_name} + {dataset_key}")
                skipped_tasks += 1
                continue
            print(f"🎯 开始训练: {model_name} + {dataset_key}")
            success, log_path = run_training(
                model_path,
                template,
                trust_remote_code,
                dataset_name,
                output_path
            )
            if success:
                completed_tasks += 1
                artifact_records.append({
                    "model": model_name,
                    "dataset": dataset_key,
                    "output": str(output_path),
                    "log": log_path
                })
            else:
                failed_tasks += 1
            print(f"📈 进度: {completed_tasks + failed_tasks + skipped_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 跳过: {skipped_tasks}, 失败: {failed_tasks})")
        if args.limit and processed_tasks >= args.limit:
            break

    print(f"\n🎉 批量训练任务完成!")
    print(f"📊 总结: {processed_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"⏭️  跳过: {skipped_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")
    if artifact_records:
        print("\n📦 本次成功产物:")
        for record in artifact_records:
            print(f"  - {record['model']} @ {record['dataset']}")
            print(f"    LoRA目录: {record['output']}")
            if record["log"]:
                print(f"    日志: {record['log']}")
        model_args = " ".join(sorted({item["model"] for item in artifact_records}))
        dataset_args = " ".join(sorted({item["dataset"] for item in artifact_records}))
        next_cmd = (
            "python scripts/multi_model_lora_inference.py "
            f"--models {model_args} --datasets {dataset_args}"
        )
        print(f"\n🔜 推荐下一步: {next_cmd}")
        print(f"📁 LoRA 输出根目录: {SA_OUTPUT_ROOT}")

if __name__ == "__main__":
    main() 
