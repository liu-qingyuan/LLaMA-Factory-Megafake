#!/usr/bin/env python3
"""
敏感性分析快速入口脚本
简化命令行调用
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) == 1:
        print("🔍 敏感性分析工具")
        print("=" * 40)
        print("可用命令:")
        print("  python scripts/sa.py test         # 快速测试配置")
        print("  python scripts/sa.py quick        # 快速验证模式")
        print("  python scripts/sa.py full         # 完整数据集分析")
        print("  python scripts/sa.py monitor      # 启动资源监控")
        print("  python scripts/sa.py verify       # 验证模型")
        print()
        print("高级用法:")
        print("  python scripts/sa.py --models Qwen1.5-7B --mode full")
        print("  python scripts/sa.py --memory-optimized --type data")
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