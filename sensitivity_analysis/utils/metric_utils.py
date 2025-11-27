#!/usr/bin/env python3
"""
评估工具函数
Metric Utilities for Sensitivity Analysis

提供各种评估指标的计算函数
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        self.supported_metrics = [
            'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro',
            'f1_weighted', 'auc', 'confusion_matrix', 'classification_report'
        ]

    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        计算分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率 (用于AUC计算)

        Returns:
            指标字典
        """
        if len(y_true) != len(y_pred):
            raise ValueError("真实标签和预测标签长度不一致")

        if not y_true:
            return {metric: 0.0 for metric in self.supported_metrics}

        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        f1_micro = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )[2]

        f1_weighted = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted
        }

        # AUC (如果提供了概率)
        if y_prob is not None and len(set(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
                metrics['auc'] = auc
            except Exception as e:
                logger.warning(f"AUC计算失败: {str(e)}")
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0

        return metrics

    def calculate_efficiency_metrics(
        self,
        training_time: float,
        inference_time: float,
        memory_usage: float,
        data_size: int
    ) -> Dict[str, float]:
        """
        计算效率指标

        Args:
            training_time: 训练时间 (秒)
            inference_time: 推理时间 (秒)
            memory_usage: 内存使用 (GB)
            data_size: 数据大小

        Returns:
            效率指标字典
        """
        return {
            'training_time': training_time,
            'inference_time': inference_time,
            'inference_speed': data_size / inference_time if inference_time > 0 else 0,  # 样本/秒
            'memory_usage': memory_usage,
            'memory_efficiency': data_size / memory_usage if memory_usage > 0 else 0,  # 样本/GB
            'training_efficiency': data_size / training_time if training_time > 0 else 0  # 样本/秒
        }

    def calculate_robustness_metrics(
        self,
        base_metrics: Dict[str, float],
        noisy_metrics: Dict[str, float],
        data_drift_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算鲁棒性指标

        Args:
            base_metrics: 基础指标
            noisy_metrics: 噪声数据指标
            data_drift_metrics: 数据漂移指标

        Returns:
            鲁棒性指标字典
        """
        robustness_scores = {}

        for metric in ['accuracy', 'f1_macro', 'precision', 'recall']:
            if metric in base_metrics and metric in noisy_metrics:
                base_score = base_metrics[metric]
                noisy_score = noisy_metrics[metric]

                # 噪声鲁棒性 = 噪声环境下的性能保持率
                noise_robustness = noisy_score / base_score if base_score > 0 else 0
                robustness_scores[f'noise_robustness_{metric}'] = noise_robustness

            if metric in base_metrics and metric in data_drift_metrics:
                base_score = base_metrics[metric]
                drift_score = data_drift_metrics[metric]

                # 数据漂移鲁棒性 = 漂移环境下的性能保持率
                drift_robustness = drift_score / base_score if base_score > 0 else 0
                robustness_scores[f'drift_robustness_{metric}'] = drift_robustness

        # 综合鲁棒性评分
        robustness_scores['overall_robustness'] = np.mean([
            score for key, score in robustness_scores.items()
            if key.endswith('robustness_f1_macro')
        ])

        return robustness_scores

    def calculate_sensitivity_metrics(
        self,
        metrics_by_data_size: Dict[int, Dict[str, float]],
        metrics_by_parameter: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        计算敏感性指标

        Args:
            metrics_by_data_size: 按数据量分组的指标
            metrics_by_parameter: 按参数分组的指标

        Returns:
            敏感性指标字典
        """
        sensitivity_scores = {}

        # 数据量敏感性
        if metrics_by_data_size:
            data_sizes = sorted(metrics_by_data_size.keys())
            if len(data_sizes) >= 2:
                min_size = data_sizes[0]
                max_size = data_sizes[-1]

                for metric in ['accuracy', 'f1_macro']:
                    if all(metric in m for m in metrics_by_data_size.values()):
                        min_score = metrics_by_data_size[min_size][metric]
                        max_score = metrics_by_data_size[max_size][metric]

                        # 敏感性 = 性能提升幅度
                        sensitivity = (max_score - min_score) / min_score if min_score > 0 else 0
                        sensitivity_scores[f'data_sensitivity_{metric}'] = sensitivity

        # 参数敏感性
        for param_name, param_metrics in metrics_by_parameter.items():
            if len(param_metrics) >= 2:
                metric_values = [m.get('f1_macro', 0) for m in param_metrics.values()]
                metric_std = np.std(metric_values)
                metric_mean = np.mean(metric_values)

                # 参数敏感性 = 性能变异系数
                sensitivity = metric_std / metric_mean if metric_mean > 0 else 0
                sensitivity_scores[f'param_sensitivity_{param_name}'] = sensitivity

        return sensitivity_scores

    def generate_performance_report(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        生成性能报告

        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            metrics: 指标字典

        Returns:
            性能报告文本
        """
        report = f"""
# 模型性能报告

## 基本信息
- **模型**: {model_name}
- **数据集**: {dataset_name}

## 分类性能指标
"""

        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro', 'auc']
        for metric in classification_metrics:
            if metric in metrics:
                report += f"- **{metric}**: {metrics[metric]:.4f}\n"

        if any(key in metrics for key in ['training_time', 'inference_time', 'memory_usage']):
            report += "\n## 效率指标\n"
            if 'training_time' in metrics:
                report += f"- **训练时间**: {metrics['training_time']:.2f}秒\n"
            if 'inference_time' in metrics:
                report += f"- **推理时间**: {metrics['inference_time']:.2f}秒\n"
            if 'memory_usage' in metrics:
                report += f"- **内存使用**: {metrics['memory_usage']:.2f}GB\n"

        return report

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str = 'f1_macro'
    ) -> Dict[str, Any]:
        """
        比较多个模型的性能

        Args:
            model_results: 模型结果字典 {model_name: metrics}
            primary_metric: 主要比较指标

        Returns:
            比较结果
        """
        if not model_results:
            return {'ranking': [], 'best_model': None, 'comparison': {}}

        # 排序模型
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get(primary_metric, 0),
            reverse=True
        )

        ranking = [model_name for model_name, _ in sorted_models]
        best_model = ranking[0] if ranking else None

        # 生成对比表
        comparison = {}
        all_metrics = set()
        for metrics in model_results.values():
            all_metrics.update(metrics.keys())

        for metric in sorted(all_metrics):
            comparison[metric] = {
                model_name: metrics.get(metric, 0)
                for model_name, metrics in model_results.items()
            }

        return {
            'ranking': ranking,
            'best_model': best_model,
            'comparison': comparison,
            'primary_metric': primary_metric
        }