# Megafake项目文档

## 项目概述

Megafake是一个基于LLaMA-Factory框架的假新闻检测研究项目，专门针对多模型、多推理方法的假新闻检测任务进行优化。该项目提供了完整的数据预处理、模型训练、推理和评估流水线。

## 数据集介绍

### 任务划分

#### Task 1: 基础假新闻检测
- **任务类型**: 二分类任务（fake vs legitimate）
- **数据来源**: 综合新闻数据集
- **目标**: 判断新闻文章的真实性

#### Task 2: 细粒度假新闻分类
- **任务类型**: 多类别分类任务
- **子分类**:
  - `style_based`: 风格类假新闻（改写合法新闻为小报风格，或将假新闻包装为主流风格）
  - `content_based`: 内容类假新闻（修改合法新闻的属性）
  - `integration_based`: 集成类假新闻（混合假新闻和合法新闻）
- **前提**: 已知新闻为假的前提下进行细粒度分类

#### Task 3: 多源假新闻检测
- **任务类型**: 二分类任务
- **数据来源**:
  - `gossip`: Gossip来源数据
  - `polifact`: Politifact来源数据
- **目标**: 在不同数据源上验证模型的泛化能力

### 原始数据格式

#### 基础数据结构
Task 1 和 Task 3 的原始数据采用简单的二分类JSON格式：

```json
[
  {
    "text": "新闻文章内容...",
    "label": 1  // 1 = fake, 0 = legitimate
  }
]
```

#### Task 2 细粒度格式
Task 2 可能包含更细粒度的标签：

```json
[
  {
    "text": "新闻文章内容...",
    "label": 1,
    "subtype": "style_based"  // 或 "content_based", "integration_based"
  },
  {
    "text": "另一篇新闻文章内容...",
    "label": 0,
    "subtype": "legitimate"
  }
]
```

### 原始数据集目录结构

**位置**: `/root/autodl-tmp/LLaMA-Factory-Megafake/data/data_table/`

```
data/data_table/
├── task1/
│   └── full/                           # 原始完整数据
│       ├── megafake_glm_binary.json    # GLM格式原始数据
│       └── megafake_llama_binary.json  # Llama格式原始数据
├── task2/
│   └── full/                           # Task2原始数据
│       ├── style_based_fake.json       # 风格类假新闻
│       ├── style_based_legitimate.json # 风格类合法新闻
│       ├── content_based_fake.json     # 内容类假新闻
│       ├── integration_based_fake.json # 集成类假新闻
│       └── integration_based_legitimate.json # 集成类合法新闻
└── task3/
    └── full/                           # Task3原始数据
        ├── gossip_binary.json          # Gossip来源数据
        └── polifact_binary.json        # Politifact来源数据
```

### 数据获取和放置说明

#### 1. 数据获取
- 原始数据需要从官方渠道获取
- 确保遵循数据使用许可协议
- 数据应为UTF-8编码的JSON格式

#### 2. 数据放置步骤
1. **创建目录结构**:
   ```bash
   mkdir -p data/data_table/task{1,2,3}/full
   ```

2. **放置原始文件**:
   - 将Task 1数据放置到: `data/data_table/task1/full/`
   - 将Task 2数据放置到: `data/data_table/task2/full/`
   - 将Task 3数据放置到: `data/data_table/task3/full/`

3. **文件命名规范**:
   - Task 1: `megafake_{glm|llama}_binary.json`
   - Task 2: `{subtype}_{fake|legitimate}.json`
   - Task 3: `{source}_binary.json`

#### 3. 数据验证
运行转换脚本前，确保原始数据满足以下要求：
- JSON格式正确
- 包含必需的`text`和`label`字段
- Task 2数据包含`subtype`字段（可选）
- 文件编码为UTF-8
- 数据量充足（建议每个子类别至少1000条）

### 转换格式

项目支持多种推理格式的数据转换：

#### 1. 标准Alpaca格式
```json
{
  "instruction": "Identify whether the news is legitimate or fake in one word: {text}",
  "input": "",
  "output": "legitimate/fake"
}
```

#### 2. CoT-SC格式 (Chain-of-Thought + Self-Consistency)
包含5个推理链的复杂格式：
- 5条不同的推理分析链
- 自洽性投票过程
- 最终答案

#### 3. FS-5格式 (Few-shot with 5 examples)
包含5个示例的少样本学习格式：
- 3个legitimate示例 + 2个fake示例
- 示例长度50-150字符
- 平衡的示例分布

#### 4. ZS-DF格式 (Zero-shot + Decomposition)
分解推理格式：
- Content Analysis（内容分析）
- Language Pattern（语言模式分析）
- Source Credibility（来源可信度评估）
- Logical Consistency（逻辑一致性检查）
- Final Classification（最终分类）

## 支持的模型

### 基础模型
- **Baichuan2-7B-Chat** (模板: baichuan2)
- **ChatGLM3-6B** (模板: chatglm3)
- **Llama-3.1-8B-Instruct** (模板: llama3)
- **Mistral-7B-v0.1** (模板: mistral)
- **Qwen1.5-7B** (模板: qwen)

### 训练方法
- **Full Fine-tuning**: 全参数微调
- **LoRA**: 低秩适应微调
- **QLoRA**: 量化LoRA微调
- **Freeze-tuning**: 冻结部分层微调

## 脚本系统

### 数据转换脚本

#### 基础转换
- `convert_task1.py`: Task1标准Alpaca格式转换
- `convert_task2.py`: Task2细粒度分类转换
- `convert_task3.py`: Task3多源数据转换

#### 高级推理格式转换
- `convert_task1_CoT_SC.py`: CoT-SC格式转换
- `convert_task1_FS_5.py`: FS-5格式转换
- `convert_task1_ZS_DF.py`: ZS-DF格式转换

### 数据采样脚本
- `sample_8k_datat_task1.py`: Task1数据采样（8k条）
- `sample_8k_datat_task2.py`: Task2数据采样（8k条）
- `sample_8k_datat_task3.py`: Task3数据采样（8k条）
- `sample_test100_multi_reasoning_task1.py`: 多推理方法测试集采样

### 训练脚本
- `multi_model_lora_train.py`: 批量LoRA训练
  - 自动遍历所有模型-数据集组合
  - 智能跳过已完成任务
  - 详细的进度追踪和日志记录

### 推理脚本
- `multi_model_inference.py`: 多模型批量推理
- `multi_model_lora_inference.py`: LoRA模型批量推理
- `vllm_infer.py`: VLLM加速推理

### 分析脚本
- `analyze_predictions.py`: 基础预测结果分析
- `analyze_predictions_CoT_SC.py`: CoT-SC格式结果分析
- `analyze_predictions_FS_5.py`: FS-5格式结果分析
- `analyze_predictions_ZS_DF.py`: ZS-DF格式结果分析
- `analyze_predictions_lora.py`: LoRA模型结果分析

## 目录结构

```
megafakeTasks/
├── task1/
│   ├── task1_full_glm/
│   │   ├── Meta-Llama-3.1-8B-Instruct/
│   │   │   ├── lora/sft/                    # LoRA适配器
│   │   │   └── result_*.jsonl               # 推理结果
│   │   ├── Baichuan2-7B-Chat/
│   │   └── ...其他模型
│   ├── task1_small_glm/
│   └── task1_full_llama/
├── task2/
│   ├── style_based/
│   ├── content_based/
│   └── integration_based/
├── task3/
│   ├── gossip/
│   └── polifact/
└── logs/                                    # 训练和推理日志
```

## 使用流程

### 1. 环境准备
```bash
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 2. 数据准备
```bash
# 转换数据格式
python convert_task1.py
python convert_task1_CoT_SC.py
python convert_task1_FS_5.py
python convert_task1_ZS_DF.py

# 采样数据（可选）
python sample_8k_datat_task1.py
```

### 3. 模型训练
```bash
# 单个模型训练
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# 批量LoRA训练
python scripts/multi_model_lora_train.py
```

### 4. 模型推理
```bash
# 单个模型推理
python scripts/vllm_infer.py --model_name_or_path <path> --dataset <dataset>

# 批量推理
python scripts/multi_model_inference.py

# LoRA批量推理
python scripts/multi_model_lora_inference.py
```

### 5. 结果分析
```bash
# 基础分析
python scripts/analyze_predictions.py --input result_file.jsonl

# 特定格式分析
python scripts/analyze_predictions_CoT_SC.py --input cot_sc_result.jsonl
python scripts/analyze_predictions_FS_5.py --input fs_5_result.jsonl
python scripts/analyze_predictions_ZS_DF.py --input zs_df_result.jsonl
```

## 配置说明

### 模型配置
```python
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": ("baichuan2", True),
    "/root/autodl-tmp/models/chatglm3-6b": ("chatglm3", True),
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": ("llama3", True),
    "/root/autodl-tmp/models/Mistral-7B-v0.1": ("mistral", False),
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True)
}
```

### 数据集配置
```python
DATASET_CONFIGS = {
    "task1_full_glm": "data_table/task1/alpaca_full/alpaca_megafake_glm_binary.json",
    "task1_full_llama": "data_table/task1/alpaca_full/alpaca_megafake_llama_binary.json",
    "task3_full_gossip": "data_table/task3/alpaca_full/alpaca_chatglm_gossip_binary.json",
    "task3_full_polifact": "data_table/task3/alpaca_full/alpaca_chatglm_polifact_binary.json"
}
```

## 训练参数

### LoRA训练默认参数
- **LoRA Rank**: 16
- **学习率**: 1.0e-4
- **训练轮数**: 1.0
- **批次大小**: 1 (per_device)
- **梯度累积步数**: 8
- **最大样本数**: 8000
- **序列长度**: 2048

### 推理参数
- **最大新token数**: 10
- **温度**: 0.1
- **Top-p**: 0.9
- **批次大小**: 1024

## 日志和监控

### 日志文件
- 训练日志: `logs/train_{model}_{dataset}_{timestamp}.log`
- 推理日志: `logs/inference_{model}_{dataset}_{timestamp}.log`
- LoRA推理日志: `logs/inference_LoRA_{model}_{dataset}_{timestamp}.log`

### 结果文件
- 标准结果: `result_{dataset}_{model}.jsonl`
- LoRA结果: `result_{dataset}_{model}_LoRA.jsonl`

## 性能优化

### 硬件要求
- **GPU**: 推荐RTX 3090/4090或A100
- **内存**: 至少16GB GPU内存
- **存储**: 每个LoRA适配器约几十MB

### 加速配置
```bash
# HuggingFace镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 网络加速
source /etc/network_turbo

# GPU选择
export CUDA_VISIBLE_DEVICES=0,1
```

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少`batch_size`或`max_samples`
2. **模型加载失败**: 检查模型路径和`trust_remote_code`设置
3. **适配器不存在**: 确保已完成对应的LoRA训练
4. **权限问题**: 确保对输出目录有写权限

### 调试建议
- 使用`test_lora_inference.py`验证LoRA配置
- 检查日志文件了解详细错误信息
- 使用小数据集进行快速测试

## 扩展开发

### 添加新模型
1. 更新`MODEL_CONFIGS`字典
2. 添加对应的模板配置
3. 创建示例配置文件

### 添加新数据集
1. 准备JSON格式数据
2. 添加到`dataset_info.json`
3. 更新`DATASET_CONFIGS`字典

### 自定义推理方法
1. 创建对应的转换脚本
2. 实现分析脚本
3. 更新批量推理配置

## 贡献指南

1. 遵循现有的代码风格和命名规范
2. 添加适当的注释和文档
3. 测试所有修改的功能
4. 更新相关文档

## 许可证

本项目基于Apache-2.0许可证开源。