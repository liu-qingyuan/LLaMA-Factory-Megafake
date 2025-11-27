#!/usr/bin/env python3
"""
敏感性分析主脚本
整合所有敏感性分析功能的统一入口
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from sensitivity_analysis.scripts.core import SensitivityAnalysis

def main():
    parser = argparse.ArgumentParser(description="LLM Sensitivity Analysis")

    # 实验类型
    parser.add_argument("--type", choices=["data", "lora", "training", "all"],
                       default="all", help="分析类型")

    # 运行模式
    parser.add_argument("--mode", choices=["quick", "full"],
                       default="quick", help="运行模式")

    # 模型选择
    parser.add_argument("--models", nargs="+",
                       help="指定要分析的模型")

    # 数据集选择
    parser.add_argument("--datasets", nargs="+",
                       help="指定要分析的数据集")

    # 输出配置
    parser.add_argument("--output", "-o",
                       default="sensitivity_analysis/results",
                       help="输出目录")

    # 内存优化
    parser.add_argument("--memory-optimized", action="store_true",
                       help="启用内存优化（完整数据集模式）")

    # 测试模式
    parser.add_argument("--test-only", action="store_true",
                       help="只运行测试，不执行完整分析")

    args = parser.parse_args()

    # 创建分析器
    analyzer = SensitivityAnalysis(
        mode=args.mode,
        analysis_type=args.type,
        models=args.models,
        datasets=args.datasets,
        output_dir=args.output,
        memory_optimized=args.memory_optimized
    )

    # 运行分析
    try:
        if args.test_only:
            success = analyzer.run_test()
        else:
            success = analyzer.run_analysis()

        if success:
            print("✅ 敏感性分析完成！")
            print(f"📁 结果保存在: {args.output}")
        else:
            print("❌ 敏感性分析失败")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断分析")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 分析异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()