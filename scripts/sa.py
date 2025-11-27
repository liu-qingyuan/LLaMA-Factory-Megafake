#!/usr/bin/env python3
"""
敏感性分析快速入口脚本
简化命令行调用
"""

import os
import sys
import subprocess
from pathlib import Path

import shutil

def smoke_test():
    """冒烟测试：检查环境、模型、权限和资源"""
    print("🧪 正在执行冒烟测试 (Smoke Test)...")
    
    # 1. 检查关键目录
    repo_root = Path(__file__).resolve().parent.parent
    required_dirs = [
        repo_root / "sensitivity_analysis" / "outputs",
        repo_root / "sensitivity_analysis" / "logs",
        repo_root / "sensitivity_analysis" / "results",
        repo_root / "data"
    ]
    
    print("\n[1/4] 检查目录结构与权限...")
    for d in required_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".write_test"
            test_file.touch()
            test_file.unlink()
            print(f"  ✅ {d} (可读写)")
        except Exception as e:
            print(f"  ❌ {d} (权限异常: {e})")
            return False

    # 2. 检查 Mini Test100 数据集
    print("\n[2/4] 检查 Mini Test100 数据集...")
    test100_path = repo_root / "data" / "data_table" / "task1" / "alpaca_test100_balanced" / "alpaca_megafake_glm_test200_balanced.json"
    if test100_path.exists():
        print(f"  ✅ Test100 数据集就绪: {test100_path}")
    else:
        print(f"  ❌ Test100 数据集缺失: {test100_path}")
        print("     请先运行: python sample_test100_multi_reasoning_task1.py")
        return False

    # 3. 检查模型目录
    print("\n[3/4] 检查模型目录...")
    model_root = Path("/root/autodl-tmp/models")
    models_to_check = ["Qwen1.5-7B", "Meta-Llama-3.1-8B-Instruct"]
    for m in models_to_check:
        p = model_root / m
        if p.exists() and p.is_dir():
            print(f"  ✅ 模型存在: {m}")
        else:
            print(f"  ⚠️  模型缺失: {m} (影响该模型的实验)")

    # 4. 检查系统资源
    print("\n[4/4] 检查系统资源...")
    try:
        total, used, free = shutil.disk_usage(repo_root)
        free_gb = free // (2**30)
        print(f"  💾 磁盘剩余空间: {free_gb} GB")
        if free_gb < 10:
            print("  ⚠️  磁盘空间不足 10GB，建议清理")
        else:
            print("  ✅ 磁盘空间充足")
    except:
        print("  ⚠️  无法获取磁盘空间信息")

    print("\n✅ 冒烟测试通过！具备运行 Mini Test100 全链路的条件。")
    return True

def main():
    if len(sys.argv) == 1:
        print("🔍 敏感性分析工具")
        print("=" * 40)
        print("可用命令:")
        print("  python scripts/sa.py quick --dry-run # 冒烟测试 (Smoke Test)")
        print("  python scripts/sa.py quick        # 快速验证模式")
        print("  python scripts/sa.py full         # 完整数据集分析")
        print("  python scripts/sa.py monitor      # 启动资源监控")
        print("  python scripts/sa.py verify       # 验证模型")
        return

    # 处理 quick --dry-run
    if len(sys.argv) > 2 and sys.argv[1] == "quick" and sys.argv[2] == "--dry-run":
        smoke_test()
        return

    cmd = ["python", "sensitivity_analysis/scripts/run_analysis.py"]

    # 简化的命令映射
    if sys.argv[1] == "test":
        cmd.append("--test-only")
    elif sys.argv[1] == "quick":
        cmd.extend(["--mode", "quick"])
    elif sys.argv[1] == "full":
        cmd.extend(["--mode", "full"])
    elif sys.argv[1] == "monitor":
        subprocess.run(["python", "sensitivity_analysis/scripts/monitor.py"])
        return
    elif sys.argv[1] == "verify":
        subprocess.run(["python", "sensitivity_analysis/model_utils/verify_models.py"])
        return
    else:
        # 传递其他参数
        cmd.extend(sys.argv[1:])

    print(f"🚀 运行: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()