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
    "task1_test200_balanced_glm": "data_table/task1/alpaca_test100_balanced/alpaca_megafake_glm_test200_balanced.json",
}


def get_model_name(model_path: str) -> str:
    """从模型路径提取模型名称"""
    return Path(model_path).name


def build_relative_output(dataset_name: str, model_name: str) -> Path:
    """构建输出文件的相对路径（不含根目录）"""
    
    if "task1" in dataset_name:
        task = "task1"
        if "cot_sc" in dataset_name:
            reasoning_type = "CoT_SC"
        elif "fs_5" in dataset_name:
            reasoning_type = "FS_5"
        elif "zs_df" in dataset_name:
            reasoning_type = "ZS_DF"
        else:
            reasoning_type = None

        if "full" in dataset_name:
            size = "full"
        elif "test100" in dataset_name:
            size = "test100"
        elif "test200" in dataset_name:
            size = "test200_balanced"
        else:
            size = "small"

        if "glm" in dataset_name:
            if reasoning_type:
                if size in ("test100", "test200_balanced"):
                    data_type = f"test100_{reasoning_type.lower()}_megafake_glm_binary"
                else:
                    data_type = f"{reasoning_type.lower()}_megafake_glm_binary"
            else:
                data_type = "megafake_glm_binary" if size != "test200_balanced" else "test200_balanced_megafake_glm_binary"
        else:
            if reasoning_type:
                if size == "test100":
                    data_type = f"test100_{reasoning_type.lower()}_megafake_llama_binary"
                else:
                    data_type = f"{reasoning_type.lower()}_megafake_llama_binary"
            else:
                data_type = "megafake_llama_binary"

    elif "task2" in dataset_name:
        task = "task2"
        size = "full" if "full" in dataset_name else "small"
        parts = dataset_name.split("_")
        model_source = parts[2]
        news_type = parts[-1]
        subclass_parts = parts[3:-1]
        if "based" in subclass_parts:
            subclass_parts.remove("based")
        subclass = "_".join(subclass_parts)
        if model_source == "glm":
            data_type = f"glm_{subclass}_based_{news_type}"
        else:
            data_type = f"llama3_{subclass}_based_{news_type}"

    else:
        task = "task3"
        size = "full" if "full" in dataset_name else "small"
        data_type = "chatglm_gossip_binary" if "gossip" in dataset_name else "chatglm_polifact_binary"

    relative = Path(task) / size / f"result_{data_type}_{model_name}.jsonl"
    return relative


def get_save_path(model_path: str, dataset_name: str) -> Path:
    """根据模型和数据集生成保存路径"""
    model_name = get_model_name(model_path)
    relative = build_relative_output(dataset_name, model_name)
    return SA_OUTPUT_ROOT / relative


def get_legacy_save_path(model_path: str, dataset_name: str) -> Path:
    model_name = get_model_name(model_path)
    relative = build_relative_output(dataset_name, model_name)
    return LEGACY_OUTPUT_ROOT / relative


def get_log_path(model_path: str, dataset_name: str) -> Path:
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inference_{model_name}_{dataset_name}_{timestamp}.log"
    return SA_LOG_ROOT / log_filename


def run_inference(model_path: str,
                  template: str,
                  dataset_name: str,
                  save_path: Path,
                  max_new_tokens: int | None = None) -> bool:
    if max_new_tokens is None:
        if "cot_sc" in dataset_name:
            max_new_tokens = 512
        elif "zs_df" in dataset_name:
            max_new_tokens = 256
        elif "fs_5" in dataset_name:
            max_new_tokens = 128
        elif "test200" in dataset_name:
            max_new_tokens = 64
        else:
            max_new_tokens = 30

    cmd = [
        "python", "scripts/vllm_infer.py",
        "--model_name_or_path", model_path,
        "--template", template,
        "--dataset", dataset_name,
        "--save_name", str(save_path),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", "0.1",
        "--top_p", "0.9",
        "--batch_size", "1024"
    ]

    model_name = get_model_name(model_path)
    if "Baichuan" in model_name or "chatglm" in model_name.lower():
        cmd.append("--trust_remote_code")

    log_path = get_log_path(model_path, dataset_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"运行命令: {' '.join(cmd)}")
    print(f"日志文件: {log_path}")

    try:
        env_cmd = [
            "bash", "-c",
            "export HF_ENDPOINT=https://hf-mirror.com && "
            "source /etc/network_turbo 2>/dev/null || true && "
            f"{' '.join(cmd)}"
        ]

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"开始时间: {datetime.datetime.now()}\n")
            log_file.write(f"执行命令: {' '.join(cmd)}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()

            process = subprocess.Popen(
                env_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
            log_file.write("=" * 80 + "\n")
            log_file.write(f"结束时间: {datetime.datetime.now()}\n")
            log_file.write(f"返回码: {return_code}\n")

        if return_code == 0:
            print(f"✅ 成功完成: {model_name} + {dataset_name}")
            print(f"   保存至: {save_path.resolve()}")
            print(f"   日志至: {log_path.resolve()}")
            next_cmd = (
                f"python scripts/analyze_predictions.py --file {save_path.resolve()} "
                f"--output sensitivity_analysis/results/{save_path.stem}_metrics.csv"
            )
            print(f"👉 下一步: {next_cmd}")
            return True
        else:
            print(f"❌ 失败: {model_name} + {dataset_name}")
            print(f"   返回码: {return_code}")
            print(f"   日志文件: {log_path}")
            return False
    except Exception as exc:
        print(f"❌ 执行异常: {model_name} + {dataset_name}")
        print(f"   异常信息: {exc}")
        return False


def check_model_exists(model_path: str) -> bool:
    if not os.path.exists(model_path):
        return False
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        print(f"⚠️  模型配置文件不存在: {config_file}")
        return False
    return True


def resolve_dataset_file(dataset_key: str) -> Path | None:
    info = DATASET_INFO.get(dataset_key)
    if not info:
        return None
    file_name = info.get("file_name")
    if not file_name:
        return None
    return REPO_ROOT / "data" / file_name


def dry_run_check(model_path: str, dataset_key: str, dataset_rel: str) -> bool:
    dataset_file = resolve_dataset_file(dataset_key)
    dataset_ready = dataset_file.exists() if dataset_file else False
    save_path = get_save_path(model_path, dataset_key)
    legacy_path = get_legacy_save_path(model_path, dataset_key)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Dry-Run] 模型: {model_path}")
    if dataset_file:
        print(f"[Dry-Run] 数据文件: {dataset_file} {'✅' if dataset_ready else '❌'}")
    else:
        print(f"[Dry-Run] 数据文件: 未在 dataset_info.json 中登记 ❌")
    print(f"[Dry-Run] 输出文件: {save_path.resolve()}")
    if legacy_path.exists():
        print(f"[Dry-Run] 兼容：检测到历史结果 {legacy_path.resolve()}")
    if not dataset_ready:
        return False
    return check_model_exists(model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行基础模型推理（无 LoRA）")
    parser.add_argument("--models", nargs="+", help="指定模型名称或绝对路径")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                        help="指定数据集键")
    parser.add_argument("--limit", type=int, help="限制执行任务数量")
    parser.add_argument("--dry-run", action="store_true", help="仅检查依赖，不实际推理")
    parser.add_argument("--skip-existing", action="store_true", help="若存在结果则跳过")
    parser.add_argument("--max-new-tokens", type=int, help="覆盖默认 max_new_tokens")
    return parser.parse_args()


def select_models(model_filters):
    if not model_filters:
        return list(MODEL_CONFIGS.items())
    selected = []
    for item in model_filters:
        if item in MODEL_CONFIGS:
            selected.append((item, MODEL_CONFIGS[item]))
            continue
        if item in MODEL_NAME_MAP:
            path = MODEL_NAME_MAP[item]
            selected.append((path, MODEL_CONFIGS[path]))
            continue
        resolved = Path(item).expanduser()
        if str(resolved) in MODEL_CONFIGS:
            selected.append((str(resolved), MODEL_CONFIGS[str(resolved)]))
        else:
            print(f"⚠️  未识别的模型: {item}")
    return selected


def select_datasets(dataset_filters):
    if not dataset_filters:
        return list(DATASET_CONFIGS.items())
    selected = []
    for key in dataset_filters:
        rel = DATASET_CONFIGS.get(key)
        if rel:
            selected.append((key, rel))
        else:
            print(f"⚠️  未识别的数据集: {key}")
    return selected


def main():
    args = parse_args()
    selected_models = select_models(args.models)
    selected_datasets = select_datasets(args.datasets)
    total_tasks = len(selected_models) * len(selected_datasets)
    if args.limit:
        total_tasks = min(total_tasks, args.limit)

    print("🚀 开始多模型推理任务")
    print(f"📊 选中 {len(selected_models)} 个模型，{len(selected_datasets)} 个数据集")
    print(f"🎯 计划任务数: {total_tasks}")

    if args.dry_run:
        issues = False
        processed = 0
        for model_path, template in selected_models:
            for dataset_key, dataset_rel in selected_datasets:
                if args.limit and processed >= args.limit:
                    break
                processed += 1
                ok = dry_run_check(model_path, dataset_key, dataset_rel)
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

    for model_path, template in selected_models:
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(selected_datasets)
            continue

        for dataset_key, dataset_rel in selected_datasets:
            if args.limit and processed_tasks >= args.limit:
                break
            processed_tasks += 1
            save_path = get_save_path(model_path, dataset_key)
            legacy_path = get_legacy_save_path(model_path, dataset_key)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and (save_path.exists() or legacy_path.exists()):
                existing_path = save_path if save_path.exists() else legacy_path
                print(f"⏭️  结果文件已存在，跳过: {existing_path}")
                completed_tasks += 1
                continue

            print(f"🎯 开始推理: {model_name} + {dataset_key}")
            success = run_inference(
                model_path,
                template,
                dataset_key,
                save_path,
                max_new_tokens=args.max_new_tokens
            )
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1

            print(f"📈 进度: {completed_tasks + failed_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 失败: {failed_tasks})")

        if args.limit and processed_tasks >= args.limit:
            break

    print("\n🎉 多模型推理任务完成!")
    print(f"📊 总结: {processed_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")


if __name__ == "__main__":
    main()
