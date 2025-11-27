#!/usr/bin/env python3
"""
可视化工具函数
Plot Utilities for Sensitivity Analysis

提供敏感性分析图表生成功能
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SensitivityPlotter:
    """敏感性分析图表生成器"""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("viridis")

    def generate_sensitivity_curves(self, results: List[Dict[str, Any]]) -> str:
        """
        生成主要敏感性分析图（性能曲线）

        Args:
            results: 实验结果列表

        Returns:
            生成的图表文件路径
        """
        try:
            # 按模型分组数据
            model_data = {}
            for result in results:
                model_name = result.get('model_name', 'Unknown')
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append(result)

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('LLM模型敏感性分析 - 性能曲线图', fontsize=16, fontweight='bold')

            # 1. F1 Score vs 数据量
            ax1 = axes[0, 0]
            for model_name, data in model_data.items():
                # 筛选数据敏感性实验
                data_sensitivity = [r for r in data if 'data_size' in r and 'lora_config' not in r]
                if data_sensitivity:
                    data_sensitivity.sort(key=lambda x: x['data_size'])
                    sizes = [r['data_size'] for r in data_sensitivity]
                    f1_scores = [r.get('f1_macro', 0) for r in data_sensitivity]
                    ax1.plot(sizes, f1_scores, marker='o', linewidth=2, label=model_name)

            ax1.set_xlabel('数据量 (样本数)')
            ax1.set_ylabel('F1 Score (Macro)')
            ax1.set_title('F1 Score vs 数据量')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 准确率 vs 数据量
            ax2 = axes[0, 1]
            for model_name, data in model_data.items():
                data_sensitivity = [r for r in data if 'data_size' in r and 'lora_config' not in r]
                if data_sensitivity:
                    data_sensitivity.sort(key=lambda x: x['data_size'])
                    sizes = [r['data_size'] for r in data_sensitivity]
                    accuracies = [r.get('accuracy', 0) for r in data_sensitivity]
                    ax2.plot(sizes, accuracies, marker='s', linewidth=2, label=model_name)

            ax2.set_xlabel('数据量 (样本数)')
            ax2.set_ylabel('准确率')
            ax2.set_title('准确率 vs 数据量')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 训练时间 vs 数据量
            ax3 = axes[1, 0]
            for model_name, data in model_data.items():
                data_sensitivity = [r for r in data if 'data_size' in r and 'lora_config' not in r]
                if data_sensitivity:
                    data_sensitivity.sort(key=lambda x: x['data_size'])
                    sizes = [r['data_size'] for r in data_sensitivity]
                    training_times = [r.get('training_time', 0) for r in data_sensitivity]
                    ax3.plot(sizes, training_times, marker='^', linewidth=2, label=model_name)

            ax3.set_xlabel('数据量 (样本数)')
            ax3.set_ylabel('训练时间 (秒)')
            ax3.set_title('训练时间 vs 数据量')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 内存使用 vs 数据量
            ax4 = axes[1, 1]
            for model_name, data in model_data.items():
                data_sensitivity = [r for r in data if 'data_size' in r and 'lora_config' not in r]
                if data_sensitivity:
                    data_sensitivity.sort(key=lambda x: x['data_size'])
                    sizes = [r['data_size'] for r in data_sensitivity]
                    memory_usage = [r.get('memory_usage', 0) for r in data_sensitivity]
                    ax4.plot(sizes, memory_usage, marker='d', linewidth=2, label=model_name)

            ax4.set_xlabel('数据量 (样本数)')
            ax4.set_ylabel('内存使用 (GB)')
            ax4.set_title('内存使用 vs 数据量')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            output_file = self.output_dir / "main_sensitivity_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"主要敏感性分析图已保存到: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成敏感性曲线图失败: {str(e)}")
            return ""

    def generate_performance_heatmap(self, results: List[Dict[str, Any]]) -> str:
        """
        生成性能热力图

        Args:
            results: 实验结果列表

        Returns:
            生成的图表文件路径
        """
        try:
            # 准备热力图数据
            heatmap_data = []
            model_names = set()
            data_sizes = set()

            for result in results:
                if 'data_size' in result and 'lora_config' not in r:
                    model_name = result.get('model_name', 'Unknown')
                    data_size = result['data_size']
                    f1_score = result.get('f1_macro', 0)

                    heatmap_data.append({
                        'model': model_name,
                        'data_size': data_size,
                        'f1_score': f1_score
                    })
                    model_names.add(model_name)
                    data_sizes.add(data_size)

            if not heatmap_data:
                logger.warning("没有数据敏感性实验结果，无法生成热力图")
                return ""

            # 转换为DataFrame
            df = pd.DataFrame(heatmap_data)
            pivot_df = df.pivot(index='model', columns='data_size', values='f1_score')

            # 创建热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'F1 Score (Macro)'},
                square=True
            )

            plt.title('模型性能热力图 (F1 Score)', fontsize=14, fontweight='bold')
            plt.xlabel('数据量 (样本数)')
            plt.ylabel('模型')
            plt.tight_layout()

            # 保存图表
            output_file = self.output_dir / "performance_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"性能热力图已保存到: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成性能热力图失败: {str(e)}")
            return ""

    def generate_parameter_sensitivity_plot(self, results: List[Dict[str, Any]]) -> str:
        """
        生成参数敏感性分析图

        Args:
            results: 实验结果列表

        Returns:
            生成的图表文件路径
        """
        try:
            # 筛选LoRA参数敏感性实验
            lora_results = [r for r in results if 'lora_config' in r]

            if not lora_results:
                logger.warning("没有LoRA参数敏感性实验结果，无法生成参数敏感性图")
                return ""

            # 准备数据
            param_data = []
            for result in lora_results:
                config = result['lora_config']
                param_data.append({
                    'lora_r': config.get('r', 0),
                    'lora_alpha': config.get('alpha', 0),
                    'lora_dropout': config.get('dropout', 0),
                    'f1_score': result.get('f1_macro', 0),
                    'accuracy': result.get('accuracy', 0)
                })

            df = pd.DataFrame(param_data)

            # 创建图表
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('LoRA参数敏感性分析', fontsize=16, fontweight='bold')

            # 1. LoRA r vs F1 Score
            ax1 = axes[0]
            for r_val in df['lora_r'].unique():
                subset = df[df['lora_r'] == r_val]
                ax1.scatter(subset['lora_alpha'], subset['f1_score'],
                           s=100, alpha=0.7, label=f'r={r_val}')
            ax1.set_xlabel('LoRA Alpha')
            ax1.set_ylabel('F1 Score (Macro)')
            ax1.set_title('LoRA Rank vs F1 Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. LoRA alpha vs F1 Score
            ax2 = axes[1]
            for alpha_val in df['lora_alpha'].unique():
                subset = df[df['lora_alpha'] == alpha_val]
                ax2.scatter(subset['lora_r'], subset['f1_score'],
                           s=100, alpha=0.7, label=f'alpha={alpha_val}')
            ax2.set_xlabel('LoRA Rank')
            ax2.set_ylabel('F1 Score (Macro)')
            ax2.set_title('LoRA Alpha vs F1 Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. LoRA dropout vs F1 Score
            ax3 = axes[2]
            for dropout_val in df['lora_dropout'].unique():
                subset = df[df['lora_dropout'] == dropout_val]
                ax3.scatter(subset['lora_r'], subset['f1_score'],
                           s=100, alpha=0.7, label=f'dropout={dropout_val}')
            ax3.set_xlabel('LoRA Rank')
            ax3.set_ylabel('F1 Score (Macro)')
            ax3.set_title('LoRA Dropout vs F1 Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            output_file = self.output_dir / "parameter_sensitivity_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"参数敏感性分析图已保存到: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成参数敏感性图失败: {str(e)}")
            return ""

    def generate_efficiency_analysis_plot(self, results: List[Dict[str, Any]]) -> str:
        """
        生成效率分析图

        Args:
            results: 实验结果列表

        Returns:
            生成的图表文件路径
        """
        try:
            # 筛选数据敏感性实验
            data_results = [r for r in results if 'data_size' in r and 'lora_config' not in r]

            if not data_results:
                logger.warning("没有数据敏感性实验结果，无法生成效率分析图")
                return ""

            # 计算效率指标
            for result in data_results:
                # 效率 = F1 Score / (训练时间/1000)  # F1 per second
                training_time = result.get('training_time', 1)
                f1_score = result.get('f1_macro', 0)
                result['efficiency'] = f1_score / (training_time / 1000) if training_time > 0 else 0

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('模型效率分析', fontsize=16, fontweight='bold')

            # 按模型分组
            model_data = {}
            for result in data_results:
                model_name = result.get('model_name', 'Unknown')
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append(result)

            # 1. 效率 vs 数据量
            ax1 = axes[0, 0]
            for model_name, data in model_data.items():
                data.sort(key=lambda x: x['data_size'])
                sizes = [r['data_size'] for r in data]
                efficiencies = [r.get('efficiency', 0) for r in data]
                ax1.plot(sizes, efficiencies, marker='o', linewidth=2, label=model_name)

            ax1.set_xlabel('数据量 (样本数)')
            ax1.set_ylabel('效率 (F1/秒)')
            ax1.set_title('效率 vs 数据量')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 内存效率 vs 数据量
            ax2 = axes[0, 1]
            for model_name, data in model_data.items():
                data.sort(key=lambda x: x['data_size'])
                sizes = [r['data_size'] for r in data]
                memory_efficiencies = [r.get('memory_efficiency', 0) for r in data]
                ax2.plot(sizes, memory_efficiencies, marker='s', linewidth=2, label=model_name)

            ax2.set_xlabel('数据量 (样本数)')
            ax2.set_ylabel('内存效率 (样本/GB)')
            ax2.set_title('内存效率 vs 数据量')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 训练效率 vs 数据量
            ax3 = axes[1, 0]
            for model_name, data in model_data.items():
                data.sort(key=lambda x: x['data_size'])
                sizes = [r['data_size'] for r in data]
                training_efficiencies = [r.get('training_efficiency', 0) for r in data]
                ax3.plot(sizes, training_efficiencies, marker='^', linewidth=2, label=model_name)

            ax3.set_xlabel('数据量 (样本数)')
            ax3.set_ylabel('训练效率 (样本/秒)')
            ax3.set_title('训练效率 vs 数据量')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 性能-成本散点图
            ax4 = axes[1, 1]
            for model_name, data in model_data.items():
                f1_scores = [r.get('f1_macro', 0) for r in data]
                training_times = [r.get('training_time', 0) for r in data]
                ax4.scatter(training_times, f1_scores, s=100, alpha=0.7, label=model_name)

            ax4.set_xlabel('训练时间 (秒)')
            ax4.set_ylabel('F1 Score (Macro)')
            ax4.set_title('性能 vs 训练成本')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            output_file = self.output_dir / "efficiency_analysis.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"效率分析图已保存到: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成效率分析图失败: {str(e)}")
            return ""

    def generate_comprehensive_report(self, results: List[Dict[str, Any]], report_file: str = None):
        """
        生成综合性分析报告

        Args:
            results: 实验结果列表
            report_file: 报告输出文件路径
        """
        try:
            if report_file is None:
                report_file = self.output_dir / "comprehensive_analysis_report.html"

            # 计算统计信息
            total_experiments = len(results)
            completed_experiments = sum(1 for r in results if r.get('status') == 'completed')

            # 按模型统计
            model_stats = {}
            for result in results:
                model_name = result.get('model_name', 'Unknown')
                if model_name not in model_stats:
                    model_stats[model_name] = {'total': 0, 'completed': 0}
                model_stats[model_name]['total'] += 1
                if result.get('status') == 'completed':
                    model_stats[model_name]['completed'] += 1

            # 生成HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LLM模型敏感性分析报告</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; color: #333; }}
                    .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>LLM模型敏感性分析报告</h1>
                    <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="summary">
                    <h2>实验概览</h2>
                    <p><strong>总实验数:</strong> {total_experiments}</p>
                    <p><strong>完成实验数:</strong> {completed_experiments}</p>
                    <p><strong>成功率:</strong> {completed_experiments/total_experiments*100:.1f}%</p>
                </div>

                <div class="section">
                    <h2>模型统计</h2>
                    <table>
                        <tr><th>模型</th><th>总实验</th><th>完成实验</th><th>成功率</th></tr>
            """

            for model_name, stats in model_stats.items():
                success_rate = stats['completed'] / stats['total'] * 100 if stats['total'] > 0 else 0
                html_content += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{stats['total']}</td>
                            <td>{stats['completed']}</td>
                            <td>{success_rate:.1f}%</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>

                <div class="section">
                    <h2>主要图表</h2>
                    <ul>
                        <li><a href="main_sensitivity_analysis.png">主要敏感性分析图</a></li>
                        <li><a href="performance_heatmap.png">性能热力图</a></li>
                        <li><a href="parameter_sensitivity_analysis.png">参数敏感性分析图</a></li>
                        <li><a href="efficiency_analysis.png">效率分析图</a></li>
                    </ul>
                </div>
            </body>
            </html>
            """

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"综合分析报告已保存到: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"生成综合报告失败: {str(e)}")
            return ""

    def generate_all_plots(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        生成所有图表

        Args:
            results: 实验结果列表

        Returns:
            生成的图表文件路径字典
        """
        plot_files = {}

        try:
            # 生成各种图表
            sensitivity_curves = self.generate_sensitivity_curves(results)
            if sensitivity_curves:
                plot_files['sensitivity_curves'] = sensitivity_curves

            heatmap = self.generate_performance_heatmap(results)
            if heatmap:
                plot_files['heatmap'] = heatmap

            param_plot = self.generate_parameter_sensitivity_plot(results)
            if param_plot:
                plot_files['parameter_plot'] = param_plot

            efficiency_plot = self.generate_efficiency_analysis_plot(results)
            if efficiency_plot:
                plot_files['efficiency_plot'] = efficiency_plot

            # 生成综合报告
            report = self.generate_comprehensive_report(results)
            if report:
                plot_files['report'] = report

            logger.info(f"所有图表生成完成，共生成 {len(plot_files)} 个文件")
            return plot_files

        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            return {}