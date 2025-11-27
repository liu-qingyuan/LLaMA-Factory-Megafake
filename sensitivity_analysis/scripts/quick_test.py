#!/usr/bin/env python3
"""
快速测试脚本 - 最小化验证配置是否正确
"""

import os
import subprocess
import sys
from pathlib import Path

def test_vllm_only():
    """只测试VLLM推理，最快速的验证"""
    print("🧪 快速测试VLLM推理...")

    cmd = [
        "python", "scripts/vllm_infer.py",
        "--model_name_or_path", "/root/autodl-tmp/models/Qwen1.5-7B",
        "--template", "qwen",
        "--dataset", "task1_small_glm",
        "--save_name", "output/quick_test_output.jsonl",
        "--max_new_tokens", "10",
        "--batch_size", "1024",
        "--max_samples", "5"  # 只处理5个样本
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ VLLM推理测试成功！")

            if Path("output/quick_test_output.jsonl").exists():
                with open("output/quick_test_output.jsonl", 'r') as f:
                    lines = f.readlines()
                print(f"✅ 生成了 {len(lines)} 条结果")

                # 显示第一个结果
                if lines:
                    import json
                    first_result = json.loads(lines[0])
                    print(f"✅ 示例预测: {first_result.get('predict', 'N/A')}")

                # 清理测试文件
                os.remove("output/quick_test_output.jsonl")
                os.rmdir("output")
                return True
            else:
                print("❌ 没有生成输出文件")
                return False
        else:
            print(f"❌ VLLM推理失败:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ VLLM推理超时")
        return False
    except Exception as e:
        print(f"❌ VLLM推理异常: {e}")
        return False

def test_train_only():
    """只测试训练，最小化配置"""
    print("🧪 快速测试LoRA训练...")

    # 创建最小配置
    config = {
        "model_name_or_path": "/root/autodl-tmp/models/Qwen1.5-7B",
        "template": "qwen",
        "dataset": "task1_small_glm",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 32,
        "output_dir": "quick_test_train",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "max_samples": 5,  # 只训练5个样本
        "cutoff_len": 512,
        "bf16": True,
        "overwrite_output_dir": True,
        "logging_steps": 1,
        "save_steps": 10,
        "report_to": "none"
    }

    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name

    try:
        cmd = ["llamafactory-cli", "train", config_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ LoRA训练测试成功！")

            # 清理测试输出
            import shutil
            test_dir = Path("quick_test_train")
            if test_dir.exists():
                shutil.rmtree(test_dir)

            return True
        else:
            print("❌ LoRA训练失败:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ LoRA训练异常: {e}")
        return False
    finally:
        os.unlink(config_file)

def main():
    print("🚀 敏感性分析配置快速验证")
    print("=" * 50)

    # 测试VLLM
    vllm_ok = test_vllm_only()
    print()

    # 测试训练
    train_ok = test_train_only()
    print()

    # 总结
    if vllm_ok and train_ok:
        print("🎉 所有测试通过！")
        print("✅ 配置修正成功，可以运行完整实验:")
        print("   python scripts/sensitivity_analysis.py --quick-test")
    elif vllm_ok:
        print("⚠️ 只有VLLM测试通过")
        print("✅ 可以运行推理相关的实验")
    elif train_ok:
        print("⚠️ 只有训练测试通过")
        print("✅ 可以运行训练相关的实验")
    else:
        print("❌ 所有测试失败")
        print("🔧 需要进一步调试配置")
        return False

    return True

if __name__ == "__main__":
    main()