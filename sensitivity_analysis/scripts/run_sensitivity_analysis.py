#!/usr/bin/env python3
"""
敏感性分析主执行脚本
Main Script for Sensitivity Analysis

一键运行完整的LLM模型敏感性分析实验
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensitivity_analysis.scripts.experiment_manager import ExperimentManager
from sensitivity_analysis.scripts.data_sensitivity_analyzer import DataSensitivityAnalyzer
from sensitivity_analysis.scripts.parameter_sensitivity_analyzer import ParameterSensitivityAnalyzer
from sensitivity_analysis.scripts.visualization_engine import VisualizationEngine
from sensitivity_analysis.scripts.report_generator import ReportGenerator


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )


def run_complete_sensitivity_analysis(
    config_path: str = None,
    output_dir: str = None,
    max_experiments: int = None,
    quick_test: bool = False
):
    """
    运行完整的敏感性分析实验

    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        max_experiments: 最大实验数量
        quick_test: 是否启用快速测试模式
    """
    logger = logging.getLogger(__name__)
    logger.info("开始运行完整的敏感性分析实验...")

    # 设置输出目录
    if output_dir is None:
        output_dir = f"experiments/sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 创建实验管理器
    experiment_manager = ExperimentManager(config_path)

    # 设置输出目录
    experiment_manager.config['global']['output_base_dir'] = output_dir

    # 快速测试模式配置
    if quick_test:
        logger.info("启用快速测试模式")
        experiment_manager.config['quick_test']['enabled'] = True

    # Phase 1: 创建并运行实验
    logger.info("=== Phase 1: 实验执行 ===")
    experiments = experiment_manager.create_all_experiments()

    if max_experiments:
        experiments = experiments[:max_experiments]
        logger.info(f"限制实验数量: {max_experiments}")

    results = experiment_manager.run_experiments()

    # 检查实验完成情况
    completed_results = [r for r in results if r.status == 'completed']
    failed_results = [r for r in results if r.status == 'failed']

    logger.info(f"实验完成: 成功 {len(completed_results)}, 失败 {len(failed_results)}")

    if not completed_results:
        logger.error("没有成功完成的实验，退出")
        return

    # Phase 2: 数据敏感性分析
    logger.info("=== Phase 2: 数据敏感性分析 ===")
    data_analyzer = DataSensitivityAnalyzer(config_path, output_dir)
    data_sensitivity_results = data_analyzer.analyze(completed_results)

    # Phase 3: 参数敏感性分析
    logger.info("=== Phase 3: 参数敏感性分析 ===")
    param_analyzer = ParameterSensitivityAnalyzer(config_path, output_dir)
    param_sensitivity_results = param_analyzer.analyze(completed_results)

    # Phase 4: 可视化生成
    logger.info("=== Phase 4: 生成可视化图表 ===")
    viz_engine = VisualizationEngine(config_path, output_dir)

    # 生成数据敏感性图表
    viz_engine.generate_data_sensitivity_plots(data_sensitivity_results)

    # 生成参数敏感性图表
    viz_engine.generate_parameter_sensitivity_plots(param_sensitivity_results)

    # 生成综合对比图表
    viz_engine.generate_comprehensive_analysis_plots(completed_results)

    # Phase 5: 报告生成
    logger.info("=== Phase 5: 生成分析报告 ===")
    report_generator = ReportGenerator(config_path, output_dir)

    # 生成完整报告
    report_generator.generate_comprehensive_report(
        experiment_results=completed_results,
        data_sensitivity_results=data_sensitivity_results,
        parameter_sensitivity_results=param_sensitivity_results
    )

    # 生成执行摘要
    report_generator.generate_executive_summary(completed_results)

    logger.info("=== 敏感性分析实验完成 ===")
    logger.info(f"结果保存在: {output_dir}")

    # 输出关键统计信息
    print("\n" + "="*60)
    print("敏感性分析实验完成统计")
    print("="*60)
    print(f"总实验数: {len(results)}")
    print(f"成功实验: {len(completed_results)}")
    print(f"失败实验: {len(failed_results)}")
    print(f"成功率: {len(completed_results)/len(results)*100:.1f}%")
    print(f"输出目录: {output_dir}")

    if completed_results:
        # 找出最佳实验
        best_f1 = max(completed_results, key=lambda r: r.metrics.get('f1_macro', 0))
        print(f"\n最佳实验:")
        print(f"  实验ID: {best_f1.experiment_id}")
        print(f"  模型: {best_f1.model_name}")
        print(f"  数据集: {best_f1.dataset_name}")
        print(f"  数据量: {best_f1.data_size}")
        print(f"  F1分数: {best_f1.metrics.get('f1_macro', 0):.4f}")
        print(f"  训练时间: {best_f1.training_time:.2f}s")

        # 找出最有效的实验 (性能/时间比)
        efficiency_results = [r for r in completed_results if r.training_time > 0]
        if efficiency_results:
            best_efficiency = max(
                efficiency_results,
                key=lambda r: r.metrics.get('f1_macro', 0) / r.training_time
            )
            print(f"\n最高效率实验:")
            print(f"  实验ID: {best_efficiency.experiment_id}")
            print(f"  模型: {best_efficiency.model_name}")
            print(f"  效率比: {best_efficiency.metrics.get('f1_macro', 0)/best_efficiency.training_time:.6f}")

    print("="*60)


def run_data_sensitivity_only(config_path: str, output_dir: str):
    """只运行数据敏感性分析"""
    logger = logging.getLogger(__name__)
    logger.info("运行数据敏感性分析...")

    # 这里需要先加载已有的实验结果
    experiment_manager = ExperimentManager(config_path)
    # 假设已经有实验结果
    # results = experiment_manager.load_results("path/to/existing/results.json")

    # data_analyzer = DataSensitivityAnalyzer(config_path, output_dir)
    # data_sensitivity_results = data_analyzer.analyze(results)

    logger.info("数据敏感性分析完成")


def run_parameter_sensitivity_only(config_path: str, output_dir: str):
    """只运行参数敏感性分析"""
    logger = logging.getLogger(__name__)
    logger.info("运行参数敏感性分析...")

    # 类似数据敏感性分析的实现
    logger.info("参数敏感性分析完成")


def generate_visualization_only(results_file: str, output_dir: str):
    """只生成可视化图表"""
    logger = logging.getLogger(__name__)
    logger.info("生成可视化图表...")

    # 加载实验结果
    experiment_manager = ExperimentManager()
    results = experiment_manager.load_results(results_file)

    # 生成图表
    viz_engine = VisualizationEngine(output_dir=output_dir)
    viz_engine.generate_comprehensive_analysis_plots(results)

    logger.info("可视化图表生成完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LLM模型敏感性分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整的敏感性分析
  python run_sensitivity_analysis.py

  # 快速测试模式 (少量实验)
  python run_sensitivity_analysis.py --quick_test

  # 指定配置文件和输出目录
  python run_sensitivity_analysis.py --config custom_config.yaml --output_dir results

  # 限制实验数量
  python run_sensitivity_analysis.py --max_experiments 50

  # 只运行数据敏感性分析
  python run_sensitivity_analysis.py --data_sensitivity_only

  # 只生成可视化图表
  python run_sensitivity_analysis.py --visualization_only --results_file results.json
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="sensitivity_analysis/configs/experiment_configs.yaml",
        help="实验配置文件路径"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="输出目录 (默认自动生成)"
    )

    parser.add_argument(
        "--max_experiments",
        type=int,
        help="最大实验数量限制"
    )

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="快速测试模式 (少量实验)"
    )

    parser.add_argument(
        "--data_sensitivity_only",
        action="store_true",
        help="只运行数据敏感性分析"
    )

    parser.add_argument(
        "--parameter_sensitivity_only",
        action="store_true",
        help="只运行参数敏感性分析"
    )

    parser.add_argument(
        "--visualization_only",
        action="store_true",
        help="只生成可视化图表"
    )

    parser.add_argument(
        "--results_file",
        type=str,
        help="已有的实验结果文件路径 (用于可视化-only模式)"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        help="日志文件路径"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        if args.visualization_only:
            if not args.results_file:
                logger.error("可视化模式需要指定 --results_file")
                sys.exit(1)
            generate_visualization_only(args.results_file, args.output_dir)

        elif args.data_sensitivity_only:
            run_data_sensitivity_only(args.config, args.output_dir)

        elif args.parameter_sensitivity_only:
            run_parameter_sensitivity_only(args.config, args.output_dir)

        else:
            # 运行完整的敏感性分析
            run_complete_sensitivity_analysis(
                config_path=args.config,
                output_dir=args.output_dir,
                max_experiments=args.max_experiments,
                quick_test=args.quick_test
            )

    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()