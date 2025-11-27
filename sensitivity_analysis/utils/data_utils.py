#!/usr/bin/env python3
"""
数据处理工具函数
Data Utilities for Sensitivity Analysis

提供数据加载、采样、验证等功能
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """数据统计信息"""
    total_samples: int
    label_distribution: Dict[int, int]
    positive_ratio: float
    negative_ratio: float
    avg_text_length: float
    max_text_length: int
    min_text_length: int


class DataLoader:
    """数据加载器"""

    def __init__(self):
        self.data_cache = {}

    def load_data(self, data_path: str, cache: bool = True) -> List[Dict[str, Any]]:
        """
        加载数据文件

        Args:
            data_path: 数据文件路径
            cache: 是否缓存数据

        Returns:
            数据列表
        """
        if cache and data_path in self.data_cache:
            return self.data_cache[data_path]

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)

            if cache:
                self.data_cache[data_path] = data

            logger.info(f"成功加载数据: {data_path}, 样本数: {len(data)}")
            return data

        except Exception as e:
            logger.error(f"加载数据失败: {data_path}, 错误: {str(e)}")
            raise

    def get_data_stats(self, data: List[Dict[str, Any]]) -> DataStats:
        """
        获取数据统计信息

        Args:
            data: 数据列表

        Returns:
            数据统计信息
        """
        if not data:
            return DataStats(0, {}, 0.0, 0.0, 0.0, 0, 0)

        # 统计标签分布
        labels = [item.get('label', 0) for item in data]
        label_counts = Counter(labels)
        total_samples = len(data)

        # 计算标签比例
        positive_count = label_counts.get(1, 0)
        negative_count = label_counts.get(0, 0)
        positive_ratio = positive_count / total_samples if total_samples > 0 else 0.0
        negative_ratio = negative_count / total_samples if total_samples > 0 else 0.0

        # 计算文本长度统计
        text_lengths = [len(str(item.get('text', ''))) for item in data]
        avg_text_length = np.mean(text_lengths) if text_lengths else 0.0
        max_text_length = max(text_lengths) if text_lengths else 0
        min_text_length = min(text_lengths) if text_lengths else 0

        return DataStats(
            total_samples=total_samples,
            label_distribution=dict(label_counts),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            avg_text_length=avg_text_length,
            max_text_length=max_text_length,
            min_text_length=min_text_length
        )

    def sample_data(
        self,
        data_path: str,
        sample_size: int,
        random_seed: int = 42,
        method: str = "stratified",
        preserve_distribution: bool = True
    ) -> List[Dict[str, Any]]:
        """
        采样数据

        Args:
            data_path: 数据文件路径
            sample_size: 采样大小
            random_seed: 随机种子
            method: 采样方法 ("random", "stratified")
            preserve_distribution: 是否保持原始分布

        Returns:
            采样后的数据
        """
        # 加载数据
        data = self.load_data(data_path)
        original_stats = self.get_data_stats(data)

        logger.info(f"原始数据统计: {original_stats.total_samples} 样本, "
                   f"正样本: {original_stats.positive_ratio:.2%}, "
                   f"负样本: {original_stats.negative_ratio:.2%}")

        if sample_size >= original_stats.total_samples:
            logger.info(f"采样大小 ({sample_size}) >= 原始数据大小, 返回全部数据")
            return data

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        if method == "random":
            sampled_data = self._random_sampling(data, sample_size)
        elif method == "stratified":
            sampled_data = self._stratified_sampling(data, sample_size, preserve_distribution)
        else:
            raise ValueError(f"不支持的采样方法: {method}")

        # 验证采样结果
        sampled_stats = self.get_data_stats(sampled_data)

        logger.info(f"采样后统计: {sampled_stats.total_samples} 样本, "
                   f"正样本: {sampled_stats.positive_ratio:.2%}, "
                   f"负样本: {sampled_stats.negative_ratio:.2%}")

        # 检查分布保持情况
        if preserve_distribution:
            pos_diff = abs(original_stats.positive_ratio - sampled_stats.positive_ratio)
            if pos_diff > 0.1:  # 如果差异超过10%
                logger.warning(f"采样后分布变化较大: {pos_diff:.2%}")

        return sampled_data

    def _random_sampling(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """随机采样"""
        return random.sample(data, sample_size)

    def _stratified_sampling(
        self,
        data: List[Dict[str, Any]],
        sample_size: int,
        preserve_distribution: bool = True
    ) -> List[Dict[str, Any]]:
        """分层采样"""
        # 按标签分组
        label_groups = {}
        for item in data:
            label = item.get('label', 0)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        if preserve_distribution:
            # 保持原始分布
            total_samples = len(data)
            for label, group in label_groups.items():
                original_ratio = len(group) / total_samples
                target_size = int(sample_size * original_ratio)

                # 确保每组至少有一个样本
                target_size = max(1, target_size)

                if len(group) < target_size:
                    logger.warning(f"标签 {label} 的样本数 ({len(group)}) 少于目标采样数 ({target_size})")
                    target_size = len(group)

                label_groups[label] = random.sample(group, target_size)
        else:
            # 平衡采样
            min_group_size = min(len(group) for group in label_groups.values())
            samples_per_group = min(sample_size // len(label_groups), min_group_size)

            for label in label_groups:
                label_groups[label] = random.sample(label_groups[label], samples_per_group)

        # 合并所有组
        sampled_data = []
        for group in label_groups.values():
            sampled_data.extend(group)

        # 如果还需要更多样本，随机补充
        if len(sampled_data) < sample_size:
            remaining_size = sample_size - len(sampled_data)
            remaining_candidates = [item for item in data if item not in sampled_data]
            if remaining_candidates:
                additional_samples = random.sample(
                    remaining_candidates,
                    min(remaining_size, len(remaining_candidates))
                )
                sampled_data.extend(additional_samples)

        return sampled_data[:sample_size]

    def create_noisy_data(
        self,
        data: List[Dict[str, Any]],
        noise_level: float = 0.1,
        noise_type: str = "random"
    ) -> List[Dict[str, Any]]:
        """
        创建带噪声的数据

        Args:
            data: 原始数据
            noise_level: 噪声水平 (0.0-1.0)
            noise_type: 噪声类型 ("random", "systematic")

        Returns:
            带噪声的数据
        """
        noisy_data = []

        for item in data:
            noisy_item = item.copy()

            if noise_type == "random":
                # 随机噪声：随机翻转标签
                if random.random() < noise_level:
                    noisy_item['label'] = 1 - noisy_item.get('label', 0)

            elif noise_type == "systematic":
                # 系统噪声：基于某些特征的模式性噪声
                text = str(noisy_item.get('text', ''))
                # 如果文本长度超过平均值，更容易被翻转
                if len(text) > 100 and random.random() < noise_level * 1.5:
                    noisy_item['label'] = 1 - noisy_item.get('label', 0)

            noisy_data.append(noisy_item)

        logger.info(f"创建噪声数据: 噪声水平={noise_level}, 噪声类型={noise_type}")
        return noisy_data

    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        验证数据质量

        Args:
            data: 数据列表

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        if not data:
            errors.append("数据为空")
            return False, errors

        # 检查必需字段
        required_fields = ['text', 'label']
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    errors.append(f"第 {i} 条数据缺少字段: {field}")

        # 检查标签有效性
        valid_labels = {0, 1}
        for i, item in enumerate(data):
            label = item.get('label')
            if label not in valid_labels:
                errors.append(f"第 {i} 条数据标签无效: {label}")

        # 检查文本内容
        for i, item in enumerate(data):
            text = str(item.get('text', ''))
            if len(text.strip()) == 0:
                errors.append(f"第 {i} 条数据文本为空")

        # 检查数据平衡性
        labels = [item.get('label', 0) for item in data]
        label_counts = Counter(labels)
        min_ratio = min(label_counts.values()) / max(label_counts.values()) if max(label_counts.values()) > 0 else 0

        if min_ratio < 0.1:  # 如果比例低于10:1
            errors.append(f"数据严重不平衡: 最少类与最多类的比例 = {min_ratio:.3f}")

        return len(errors) == 0, errors

    def split_data(
        self,
        data: List[Dict[str, Any]],
        test_size: float = 0.2,
        random_seed: int = 42,
        stratify: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        分割数据集

        Args:
            data: 数据列表
            test_size: 测试集比例
            random_seed: 随机种子
            stratify: 是否分层采样

        Returns:
            (训练集, 测试集)
        """
        if stratify:
            # 分层分割
            labels = [item.get('label', 0) for item in data]
            train_data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=random_seed,
                stratify=labels
            )
        else:
            # 随机分割
            train_data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=random_seed
            )

        logger.info(f"数据分割完成: 训练集 {len(train_data)}, 测试集 {len(test_data)}")
        return train_data, test_data

    def save_data(self, data: List[Dict[str, Any]], output_path: str):
        """
        保存数据

        Args:
            data: 数据列表
            output_path: 输出路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.jsonl'):
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"数据已保存到: {output_path}")

        except Exception as e:
            logger.error(f"保存数据失败: {output_path}, 错误: {str(e)}")
            raise

    def create_data_report(self, data: List[Dict[str, Any]], output_path: str):
        """
        创建数据质量报告

        Args:
            data: 数据列表
            output_path: 输出路径
        """
        stats = self.get_data_stats(data)
        is_valid, errors = self.validate_data(data)

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_statistics': {
                'total_samples': stats.total_samples,
                'label_distribution': stats.label_distribution,
                'positive_ratio': stats.positive_ratio,
                'negative_ratio': stats.negative_ratio,
                'avg_text_length': stats.avg_text_length,
                'max_text_length': stats.max_text_length,
                'min_text_length': stats.min_text_length
            },
            'quality_validation': {
                'is_valid': is_valid,
                'errors': errors,
                'error_count': len(errors)
            }
        }

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"数据报告已保存到: {output_path}")


# 示例使用代码
if __name__ == "__main__":
    # 示例：如何使用DataLoader
    loader = DataLoader()

    # 假设有一个示例数据文件
    sample_data = [
        {"text": "这是一条真实的新闻", "label": 0},
        {"text": "这是一条假新闻", "label": 1},
        {"text": "另一条真实新闻", "label": 0},
        {"text": "另一条假新闻", "label": 1}
    ]

    # 保存示例数据
    os.makedirs("temp_data", exist_ok=True)
    loader.save_data(sample_data, "temp_data/sample.json")

    # 加载数据
    data = loader.load_data("temp_data/sample.json")
    print(f"加载数据: {len(data)} 条")

    # 获取统计信息
    stats = loader.get_data_stats(data)
    print(f"统计信息: {stats}")

    # 采样数据
    sampled_data = loader.sample_data("temp_data/sample.json", sample_size=2, random_seed=42)
    print(f"采样数据: {len(sampled_data)} 条")

    # 验证数据
    is_valid, errors = loader.validate_data(data)
    print(f"数据有效: {is_valid}, 错误: {errors}")

    # 创建报告
    loader.create_data_report(data, "temp_data/data_report.json")
    print("数据报告已创建")