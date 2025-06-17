# 批量Lora模型训练脚本使用说明

## 概述

`multi_model_lora_train.py` 是一个用于批量训练多个模型和数据集组合的Lora微调脚本。该脚本参考了 `multi_model_inference.py` 的设计，可以自动化地对多个模型进行Lora微调训练。

## 功能特性

- 🚀 **批量训练**: 自动遍历所有模型和数据集的组合进行训练
- 📊 **进度追踪**: 实时显示训练进度和统计信息
- 📝 **日志记录**: 每个训练任务都有独立的日志文件
- ⏭️ **智能跳过**: 自动检测已完成的训练任务并跳过
- 🔧 **配置灵活**: 易于修改模型和数据集配置

## 配置说明

### 模型配置

在 `MODEL_CONFIGS` 中配置模型：

```python
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": ("baichuan2", True),
    "/root/autodl-tmp/models/chatglm3-6b": ("chatglm3", True), 
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": ("llama3", True),
    "/root/autodl-tmp/models/Mistral-7B-v0.1": ("mistral", False),
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True)
}
```

格式：`模型路径: (模板名称, trust_remote_code)`

### 数据集配置

在 `DATASET_CONFIGS` 中配置数据集：

```python
DATASET_CONFIGS = {
    "task1_full_glm": "task1_full_glm",
    "task1_small_glm": "task1_small_glm",
    "task3_full_gossip": "task3_full_gossip",
    # 更多数据集...
}
```

格式：`数据集键名: 数据集名称`

## 训练参数

脚本使用以下默认训练参数（与原始配置保持一致）：

- **Lora Rank**: 16
- **学习率**: 1.0e-4
- **训练轮数**: 1.0
- **批次大小**: 1 (per_device)
- **梯度累积步数**: 8
- **最大样本数**: 8000
- **序列长度**: 2048

## 输出目录结构

训练结果将保存在以下目录结构中：

```
megafakeTasks/
├── task1/
│   ├── task1_small_glm/
│   │   ├── Meta-Llama-3.1-8B-Instruct/
│   │   │   └── lora/
│   │   │       └── sft/
│   │   ├── Baichuan2-7B-Chat/
│   │   └── ...
│   └── task1_full_glm/
└── task3/
    ├── task3_full_gossip/
    └── task3_full_polifact/
```

## 使用方法

### 方法1：直接运行脚本

```bash
cd /root/autodl-tmp/LLaMA-Factory-Megafake2
python scripts/multi_model_lora_train.py
```

### 方法2：使用示例脚本

```bash
cd /root/autodl-tmp/LLaMA-Factory-Megafake2
python examples/multi_train_example.py
```

### 方法3：作为模块导入

```python
from scripts.multi_model_lora_train import main
main()
```

## 日志文件

每个训练任务的日志文件保存在 `logs/` 目录下，命名格式：
```
train_{模型名称}_{数据集名称}_{时间戳}.log
```

例如：
```
logs/train_Meta-Llama-3.1-8B-Instruct_task1_small_glm_20241201_143022.log
```

## 状态检查

脚本会自动检查以下状态：

1. **模型存在性检查**: 验证模型路径和配置文件
2. **训练完成检查**: 通过检查 `adapter_config.json` 和 `adapter_model.safetensors` 判断训练是否完成
3. **目录创建**: 自动创建必要的输出目录和日志目录

## 错误处理

- ❌ **模型不存在**: 跳过该模型的所有训练任务
- ⏭️ **训练已完成**: 跳过已完成的训练任务
- 📝 **异常记录**: 所有异常都会记录在日志文件中

## 自定义配置

如需自定义训练参数，可以修改 `create_config_file` 函数中的配置字典：

```python
config = {
    # 修改这里的参数
    "lora_rank": 32,  # 增加Lora rank
    "learning_rate": 5.0e-5,  # 调整学习率
    "num_train_epochs": 2.0,  # 增加训练轮数
    # ...
}
```

## 注意事项

1. **GPU资源**: 确保有足够的GPU内存进行训练
2. **存储空间**: 每个模型的Lora适配器大约需要几十MB的存储空间
3. **训练时间**: 根据数据集大小和模型复杂度，每个训练任务可能需要几小时
4. **依赖项**: 确保已安装 `llamafactory-cli` 和相关依赖

## 监控建议

- 使用 `htop` 或 `nvidia-smi` 监控系统资源
- 定期检查日志文件了解训练进度
- 可以设置定时任务来运行脚本

## 故障排除

1. **CUDA内存不足**: 减少 `per_device_train_batch_size` 或 `max_samples`
2. **权限问题**: 确保对输出目录有写权限
3. **模型加载失败**: 检查模型路径和 `trust_remote_code` 设置 