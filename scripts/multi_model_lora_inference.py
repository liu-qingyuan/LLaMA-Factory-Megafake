#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path
import datetime
import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent
SA_ROOT = REPO_ROOT / "sensitivity_analysis"
SA_LOG_ROOT = SA_ROOT / "logs" / "infer"
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

# 模型配置：模型路径 -> 模板名称
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": "llama3",
    "/root/autodl-tmp/models/Qwen1.5-7B": "qwen",
    "/root/autodl-tmp/models/chatglm3-6b": "chatglm3",
    "/root/autodl-tmp/models/Mistral-7B-v0.1": "mistral",
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": "baichuan2",
}
MODEL_NAME_MAP = {Path(path).name: path for path in MODEL_CONFIGS.keys()}

# 数据集配置
DATASET_CONFIGS = {
    # "task1_full_glm": "data_table/task1/alpaca_full/alpaca_megafake_glm_binary.json",
    # "task1_full_llama": "data_table/task1/alpaca_full/alpaca_megafake_llama_binary.json",
    # "task1_small_glm": "data_table/task1/small_8k/alpaca_megafake_glm_8k.json",
    # "task1_small_llama": "data_table/task1/small_8k/alpaca_megafake_llama_8k.json",
    # "task3_full_gossip": "data_table/task3/alpaca_full/alpaca_chatglm_gossip_binary.json",
    # "task3_full_polifact": "data_table/task3/alpaca_full/alpaca_chatglm_polifact_binary.json",
    # "task3_small_gossip": "data_table/task3/small_8k/alpaca_chatglm_gossip_8k.json",
    # "task3_small_polifact": "data_table/task3/small_8k/alpaca_chatglm_polifact_8k.json",
    # Mini Test100 基准（100正/100负）
    "task1_test200_balanced_glm": "data_table/task1/alpaca_test100_balanced/alpaca_megafake_glm_test200_balanced.json",
    # 大规模实验数据集 (1k - 20k)
    "task1_scale_1000_glm": "data_table/task1/scale_experiment/alpaca_megafake_glm_1000.json",
    "task1_scale_2000_glm": "data_table/task1/scale_experiment/alpaca_megafake_glm_2000.json",
    "task1_scale_5000_glm": "data_table/task1/scale_experiment/alpaca_megafake_glm_5000.json",
    "task1_scale_10000_glm": "data_table/task1/scale_experiment/alpaca_megafake_glm_10000.json",
    "task1_scale_20000_glm": "data_table/task1/scale_experiment/alpaca_megafake_glm_20000.json"
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
    
    # adapter_path = REPO_ROOT / f"megafakeTasks/{task}/{train_dataset}/{model_name}/lora/sft"
    adapter_path = SA_OUTPUT_ROOT / task / train_dataset / model_name / "lora" / "sft"
    legacy_path = LEGACY_OUTPUT_ROOT / task / train_dataset / model_name / "lora" / "sft"
    if not adapter_path.exists() and legacy_path.exists():
        return str(legacy_path)
    return str(adapter_path)

def get_save_path(model_path, dataset_name):
    """根据模型和数据集生成保存路径"""
    model_name = get_model_name(model_path)
    
    # 根据数据集名称确定任务和类型
    if "task1" in dataset_name:
        task = "task1"
        if "full" in dataset_name:
            size = "full"
        elif "test100" in dataset_name:
            size = "test100"
        elif "test200" in dataset_name:
            size = "test200_balanced"
        elif "scale" in dataset_name:
            # task1_scale_1000_glm -> scale_1000
            parts = dataset_name.split('_')
            try:
                idx = parts.index("scale")
                scale_val = parts[idx+1]
                size = f"scale_{scale_val}"
            except (ValueError, IndexError):
                size = "scale_unknown"
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
        save_path = SA_OUTPUT_ROOT / task / size / f"result_{dataset_name}_{model_name}_LoRA_trained_on_{train_type}.jsonl"
    else:
        # save_path = REPO_ROOT / f"megafakeTasks/{task}/{size}/result_{dataset_name}_{model_name}_LoRA.jsonl"
        save_path = SA_OUTPUT_ROOT / task / size / f"result_{dataset_name}_{model_name}_LoRA.jsonl"
    return str(save_path)

def get_log_path(model_path, dataset_name):
    """生成日志文件路径"""
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inference_LoRA_{model_name}_{dataset_name}_{timestamp}.log"
    # return f"logs/{log_filename}"
    return SA_LOG_ROOT / log_filename

def run_inference(model_path, template, dataset_name, save_path, max_new_tokens=10):
    """运行单个推理任务"""
    # 获取LoRA适配器路径
    adapter_path = get_lora_adapter_path(model_path, dataset_name)
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        print(f"❌ LoRA适配器未找到，跳过: {adapter_path}")
        return False, None
    
    model_name = get_model_name(model_path)
    
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
    
    # Mistral specific config
    if "Mistral" in model_name:
        # 针对某些 VLLM 版本可能需要显式设置 rotary parameters，但如果是 v0.1 且 vllm 较新，通常不需要。
        # 如果必须补齐 partial_rotary_factor，可以尝试如下：
        # cmd.extend(["--vllm_config", '{"partial_rotary_factor": 1.0}'])
        pass
    
    # 检查LoRA适配器是否存在
    if not os.path.exists(adapter_path):
        print(f"❌ LoRA适配器不存在: {adapter_path}")
        return False
    
    # 创建日志目录
    log_path = get_log_path(model_path, dataset_name)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
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
                print(f"✅ 成功完成: {model_name} + {dataset_name}")
                print(f"   保存至: {Path(save_path).resolve()}")
                print(f"   日志至: {log_path.resolve()}")
                next_cmd = (
                    "python scripts/analyze_predictions.py "
                    f"--file {Path(save_path).resolve()} --output sensitivity_analysis/results/{Path(save_path).stem}_metrics.csv"
                )
                print(f"👉 下一步: {next_cmd}")
                return True, str(log_path.resolve())
            else:
                print(f"❌ 失败: {model_name} + {dataset_name}")
                print(f"   返回码: {return_code}")
                print(f"   日志文件: {log_path.resolve()}")
                return False, str(log_path.resolve())
                
    except Exception as e:
        print(f"❌ 执行异常: {model_name} + {dataset_name}")
        print(f"   异常信息: {e}")
        return False, None

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

def resolve_dataset_file(dataset_key):
    """解析数据集文件"""
    info = DATASET_INFO.get(dataset_key)
    if not info:
        return None
    file_name = info.get("file_name")
    if not file_name:
        return None
    return REPO_ROOT / "data" / file_name

def dry_run_check(model_path, dataset_key):
    """LoRA 推理 dry-run 检查"""
    dataset_file = resolve_dataset_file(dataset_key)
    dataset_ready = dataset_file.exists() if dataset_file else False
    adapter_path = get_lora_adapter_path(model_path, dataset_key)
    adapter_ready = Path(adapter_path).exists()
    save_path = get_save_path(model_path, dataset_key)
    save_dir = Path(save_path).parent
    print(f"\n[Dry-Run] 模型: {model_path}")
    print(f"[Dry-Run] 数据集: {dataset_key}")
    if dataset_file:
        print(f"[Dry-Run] 数据文件: {dataset_file} {'✅' if dataset_ready else '❌'}")
    else:
        print(f"[Dry-Run] 数据文件: 未在 dataset_info.json 中登记 ❌")
    print(f"[Dry-Run] LoRA 适配器: {adapter_path} {'✅' if adapter_ready else '❌'}")
    print(f"[Dry-Run] 结果将写入: {Path(save_path).resolve()}")
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[Dry-Run] 创建输出目录失败: {exc}")
        return False
    return dataset_ready and adapter_ready and check_model_exists(model_path)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量运行 LoRA 推理（支持 Dry-Run）")
    parser.add_argument("--models", nargs="+", help="指定模型名称（如 Qwen1.5-7B）或绝对路径")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                        help="指定数据集键（默认全量）")
    parser.add_argument("--limit", type=int, help="限制任务数量")
    parser.add_argument("--dry-run", action="store_true", help="仅检查依赖，不实际推理")
    parser.add_argument("--skip-existing", action="store_true", help="结果存在时跳过")
    parser.add_argument("--max-new-tokens", type=int, help="覆盖默认 max_new_tokens")
    parser.add_argument("--include-large-models", action="store_true", help="允许运行 70B+ 大模型")
    return parser.parse_args()

def select_models(model_filters, include_large=False):
    """筛选模型"""
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
    for path, template in items:
        if "72B" in str(path) or "70B" in str(path):
            if not include_large:
                print(f"⚠️  跳过大模型 (需 --include-large-models): {path}")
                continue
            else:
                print(f"⚠️  包含大模型 (OOM风险): {path}")
        final_selection.append((path, template))
        
    return final_selection

def select_datasets(dataset_filters):
    """筛选数据集"""
    if not dataset_filters:
        return list(DATASET_CONFIGS.keys())
    selected = []
    for key in dataset_filters:
        if key in DATASET_CONFIGS:
            selected.append(key)
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
    print("🚀 开始多模型LoRA推理任务")
    print(f"📊 选中 {len(selected_models)} 个模型，{len(selected_datasets)} 个数据集")
    print(f"🎯 计划任务数: {total_tasks}")

    if args.dry_run:
        issues = False
        processed = 0
        for model_path, template in selected_models:
            for dataset_name in selected_datasets:
                if args.limit and processed >= args.limit:
                    break
                processed += 1
                ok = dry_run_check(model_path, dataset_name)
                if not ok:
                    issues = True
            if args.limit and processed >= args.limit:
                break
        if issues:
            print("\n⚠️  Dry-Run 检测到问题，请修复后再运行")
            return
        print("\n✅ Dry-Run 检查通过，可安全启动推理")
        return

    completed_tasks = 0
    failed_tasks = 0
    processed_tasks = 0
    artifact_records = []
    for model_path, template in selected_models:
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(selected_datasets)
            continue
        for dataset_name in selected_datasets:
            if args.limit and processed_tasks >= args.limit:
                break
            processed_tasks += 1
            save_path = get_save_path(model_path, dataset_name)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and Path(save_path).exists():
                print(f"⏭️  结果文件已存在，跳过: {save_path}")
                completed_tasks += 1
                artifact_records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "result": save_path,
                    "log": None
                })
                continue
            print(f"🎯 开始LoRA推理: {model_name} + {dataset_name}")
            success, log_path = run_inference(
                model_path,
                template,
                dataset_name,
                save_path,
                max_new_tokens=args.max_new_tokens if args.max_new_tokens else 10
            )
            if success:
                completed_tasks += 1
                artifact_records.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "result": save_path,
                    "log": log_path
                })
            else:
                failed_tasks += 1
            print(f"📈 进度: {completed_tasks + failed_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 失败: {failed_tasks})")
        if args.limit and processed_tasks >= args.limit:
            break

    print(f"\n🎉 LoRA推理任务完成!")
    print(f"📊 总结: {processed_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")
    if artifact_records:
        print("\n📦 推理产物:")
        for record in artifact_records:
            print(f"  - {record['model']} @ {record['dataset']}")
            print(f"    结果: {record['result']}")
            if record["log"]:
                print(f"    日志: {record['log']}")
        recommend_cmd = (
            "python scripts/analyze_predictions.py "
            "--dir sensitivity_analysis/outputs "
            "--output sensitivity_analysis/results/mini_test100_metrics.csv"
        )
        print(f"\n🔜 推荐下一步: {recommend_cmd}")
        print(f"📁 推理结果根目录: {SA_OUTPUT_ROOT}")

if __name__ == "__main__":
    main() 
