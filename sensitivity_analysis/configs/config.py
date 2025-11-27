#!/usr/bin/env python3
"""
Configuration utilities for sensitivity analysis
遵循项目配置约定
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 模型配置 - 遵循项目文档中的模型列表
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Qwen1.5-7B-new": ("qwen", True),
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True),
    "/root/autodl-tmp/models/Mistral-7B-v0.1": ("mistral", False),
    "/root/autodl-tmp/models/chatglm3-6b": ("chatglm3", True),
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": ("llama3", True),
    "/root/autodl-tmp/models/Baichuan2-7B-Chat-new": ("baichuan2", True),  # 重新下载的
}

# 数据集配置 - 遵循项目文档中的数据集定义
DATASET_CONFIGS = {
    # Task 1 数据集
    "task1_full_glm": "data/data_table/task1/alpaca_full/alpaca_megafake_glm_binary.json",
    "task1_full_llama": "data/data_table/task1/alpaca_full/alpaca_megafake_llama_binary.json",
    "task1_small_glm": "data/data_table/task1/small_8k/alpaca_megafake_glm_8k.json",
    "task1_small_llama": "data/data_table/task1/small_8k/alpaca_megafake_llama_8k.json",

    # Task 1 推理格式数据集
    "task1_full_cot_sc_glm": "data/data_table/task1/alpaca_full_CoT_SC/cot_sc_megafake_glm_binary.json",
    "task1_full_cot_sc_llama": "data/data_table/task1/alpaca_full_CoT_SC/cot_sc_megafake_llama_binary.json",
    "task1_full_fs_5_glm": "data/data_table/task1/alpaca_full_FS_5/fs_5_megafake_glm_binary.json",
    "task1_full_fs_5_llama": "data/data_table/task1/alpaca_full_FS_5/fs_5_megafake_llama_binary.json",
    "task1_full_zs_df_glm": "data/data_table/task1/alpaca_full_ZS_DF/zs_df_megafake_glm_binary.json",
    "task1_full_zs_df_llama": "data/data_table/task1/alpaca_full_ZS_DF/zs_df_megafake_llama_binary.json",

    # Task 1 测试数据集
    "task1_test100_cot_sc_glm": "data/data_table/task1/alpaca_test100_CoT_SC/test100_cot_sc_megafake_glm_binary.json",
    "task1_test100_cot_sc_llama": "data/data_table/task1/alpaca_test100_CoT_SC/test100_cot_sc_megafake_llama_binary.json",
    "task1_test100_fs_5_glm": "data/data_table/task1/alpaca_test100_FS_5/test100_fs_5_megafake_glm_binary.json",
    "task1_test100_fs_5_llama": "data/data_table/task1/alpaca_test100_FS_5/test100_fs_5_megafake_llama_binary.json",
    "task1_test100_zs_df_glm": "data/data_table/task1/alpaca_test100_ZS_DF/test100_zs_df_megafake_glm_binary.json",
    "task1_test100_zs_df_llama": "data/data_table/task1/alpaca_test100_ZS_DF/test100_zs_df_megafake_llama_binary.json",

    # Task 2 数据集 (细粒度分类)
    "task2_style_based_fake": "data/data_table/task2/alpaca_full/alpaca_style_based_fake.json",
    "task2_style_based_legitimate": "data/data_table/task2/alpaca_full/alpaca_style_based_legitimate.json",
    "task2_content_based_fake": "data/data_table/task2/alpaca_full/alpaca_content_based_fake.json",
    "task2_content_based_legitimate": "data/data_table/task2/alpaca_full/alpaca_content_based_legitimate.json",
    "task2_integration_based_fake": "data/data_table/task2/alpaca_full/alpaca_integration_based_fake.json",
    "task2_integration_based_legitimate": "data/data_table/task2/alpaca_full/alpaca_integration_based_legitimate.json",

    # Task 3 数据集 (多源)
    "task3_full_gossip": "data/data_table/task3/alpaca_full/alpaca_chatglm_gossip_binary.json",
    "task3_full_polifact": "data/data_table/task3/alpaca_full/alpaca_chatglm_polifact_binary.json"
}

def get_available_models() -> List[str]:
    """获取当前可用的模型列表"""
    import os
    available_models = []

    for model_path in MODEL_CONFIGS.keys():
        if os.path.exists(model_path):
            # 检查模型文件是否完整
            if os.path.isdir(model_path):
                model_name = os.path.basename(model_path)
                available_models.append(model_name)

    return available_models

def get_available_datasets() -> List[str]:
    """获取当前可用的数据集列表"""
    import os
    available_datasets = []

    for dataset_name, file_path in DATASET_CONFIGS.items():
        full_path = Path(project_root) / file_path
        if full_path.exists():
            available_datasets.append(dataset_name)

    return available_datasets

def get_model_template(model_path: str) -> Tuple[str, bool]:
    """获取模型的模板配置"""
    return MODEL_CONFIGS.get(model_path, ("qwen", True))

def create_llama_factory_config(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    training_params: Dict[str, Any],
    lora_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """创建LLaMA-Factory配置文件"""

    template, trust_remote_code = get_model_template(model_path)

    base_config = {
        # 模型配置
        "model_name_or_path": model_path,
        "template": template,
        "trust_remote_code": trust_remote_code,

        # 数据配置
        "dataset": dataset_name,
        "cutoff_len": training_params.get("max_length", 2048),
        "max_samples": training_params.get("data_size", 8000),
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,

        # 训练配置
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_params.get("r", training_params.get("lora_r", 16)) if lora_params else training_params.get("lora_r", 16),
        "lora_alpha": lora_params.get("alpha", training_params.get("lora_alpha", 32)) if lora_params else training_params.get("lora_alpha", 32),
        "lora_dropout": lora_params.get("dropout", training_params.get("lora_dropout", 0.05)) if lora_params else training_params.get("lora_dropout", 0.05),
        "lora_target": "all",

        # 输出配置
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",

        # 训练超参数
        "per_device_train_batch_size": min(training_params.get("batch_size", 16), 8),
        "gradient_accumulation_steps": max(1, 16 // training_params.get("batch_size", 16)),
        "learning_rate": training_params.get("learning_rate", 2e-5),
        "num_train_epochs": training_params.get("epochs", 1),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": training_params.get("use_bf16", True),
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
        "flash_attn": "fa2",

        # 其他配置
        "seed": 42
    }

    return base_config

def get_experiment_config(experiment_type: str = "default") -> Dict[str, Any]:
    """获取实验配置 - 参考multi_model脚本的标准配置"""
    if experiment_type == "quick_test":
        return {
            "data_sizes": [10, 50],
            "lora_params": {
                "r": [16],
                "alpha": [32],
                "dropout": [0.05]
            },
            "training_params": {
                "learning_rate": [1e-4],
                "batch_size": [1],
                "epochs": [1]
            }
        }
    elif experiment_type == "full":
        return {
            "data_sizes": [1000, 2000, 5000, 10000],
            "lora_params": {
                "r": [16],
                "alpha": [32],
                "dropout": [0.05]
            },
            "training_params": {
                "learning_rate": [1e-4],
                "batch_size": [1],
                "epochs": [1]
            }
        }
    else:
        return {
            "data_sizes": [1000, 2000, 5000, 10000],
            "lora_params": {
                "r": [16],
                "alpha": [32],
                "dropout": [0.05]
            },
            "training_params": {
                "learning_rate": [1e-4],
                "batch_size": [1],
                "epochs": [1]
            }
        }

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """验证配置的有效性"""
    errors = []

    # 检查必需的配置项
    required_keys = ["model_name_or_path", "dataset", "output_dir"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config: {key}")

    # 检查模型是否存在
    model_path = config.get("model_name_or_path")
    if model_path and not Path(model_path).exists():
        errors.append(f"Model not found: {model_path}")

    # 检查数据集是否存在
    dataset_name = config.get("dataset")
    if dataset_name and dataset_name not in DATASET_CONFIGS:
        errors.append(f"Unknown dataset: {dataset_name}")

    # 检查参数范围
    learning_rate = config.get("learning_rate")
    if learning_rate and (learning_rate <= 0 or learning_rate > 1):
        errors.append(f"Invalid learning rate: {learning_rate}")

    batch_size = config.get("per_device_train_batch_size")
    if batch_size and batch_size <= 0:
        errors.append(f"Invalid batch size: {batch_size}")

    epochs = config.get("num_train_epochs")
    if epochs and epochs <= 0:
        errors.append(f"Invalid epochs: {epochs}")

    return len(errors) == 0, errors

def create_output_directory_structure(base_dir: str, experiment_id: str) -> str:
    """创建输出目录结构，遵循项目约定"""
    base_path = Path(base_dir)

    # 创建主要输出目录
    output_dir = base_path / "sensitivity_analysis" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (output_dir / "adapter").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    return str(output_dir)

def format_experiment_name(model_name: str, dataset_name: str, params: Dict[str, Any]) -> str:
    """格式化实验名称"""
    base_name = f"{model_name}_{dataset_name}"

    # 添加参数信息
    param_parts = []
    for key, value in params.items():
        if key in ["learning_rate", "batch_size", "epochs", "data_size", "lora_r", "lora_alpha", "lora_dropout"]:
            if isinstance(value, float):
                formatted_value = f"{value:.0e}" if value < 1e-2 else f"{value:.2f}"
            else:
                formatted_value = str(value)
            param_parts.append(f"{key}_{formatted_value}")

    if param_parts:
        base_name += "_" + "_".join(param_parts)

    # 清理特殊字符
    base_name = base_name.replace("/", "_").replace("\\", "_")

    return base_name