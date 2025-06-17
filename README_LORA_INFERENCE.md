# LoRA模型批量推理脚本使用说明

## 概述

`multi_model_lora_inference.py` 是一个用于批量推理训练好的LoRA模型的脚本。该脚本基于原始的多模型推理脚本修改，专门用于使用LoRA适配器进行推理。

## 主要修改

相比原始的推理脚本，主要修改包括：

1. **添加LoRA适配器路径**: 自动根据模型和数据集生成对应的LoRA适配器路径
2. **修改推理命令**: 添加 `--adapter_name_or_path` 参数
3. **适配器存在性检查**: 在推理前检查LoRA适配器是否存在
4. **输出路径优化**: 在结果文件名中包含LoRA标识
5. **环境配置**: 添加HuggingFace镜像和学术加速配置

## 功能特性

- 🎯 **LoRA推理**: 使用训练好的LoRA适配器进行推理
- 🔍 **自动路径匹配**: 自动匹配模型和数据集对应的LoRA适配器
- ✅ **存在性验证**: 推理前验证LoRA适配器是否存在
- 📝 **详细日志**: 记录LoRA适配器路径和推理过程
- 🚀 **加速配置**: 自动配置HuggingFace镜像和网络加速

## LoRA适配器路径规则

脚本会根据以下规则生成LoRA适配器路径：

```
/root/autodl-tmp/LLaMA-Factory-Megafake2/megafakeTasks/{task}/{dataset}/{model_name}/lora/sft
```

例如：
- 模型: `Meta-Llama-3.1-8B-Instruct`
- 数据集: `task1_full_glm`
- 适配器路径: `/root/autodl-tmp/LLaMA-Factory-Megafake2/megafakeTasks/task1/task1_full_glm/Meta-Llama-3.1-8B-Instruct/lora/sft`

## 推理命令格式

生成的推理命令格式：

```bash
export HF_ENDPOINT=https://hf-mirror.com && \
source /etc/network_turbo && \
python scripts/vllm_infer.py \
    --model_name_or_path /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct \
    --adapter_name_or_path /root/autodl-tmp/LLaMA-Factory-Megafake2/megafakeTasks/task1/task1_full_glm/Meta-Llama-3.1-8B-Instruct/lora/sft \
    --template llama3 \
    --dataset task1_full_glm \
    --save_name megafakeTasks/task1/full/result_task1_full_glm_Meta-Llama-3.1-8B-Instruct_LoRA.jsonl \
    --max_new_tokens 10 \
    --temperature 0.1 \
    --top_p 0.9 \
    --batch_size 1024
```

## 输出文件命名

结果文件命名格式：
```
result_{dataset_name}_{model_name}_LoRA.jsonl
```

例如：
- `result_task1_full_glm_Meta-Llama-3.1-8B-Instruct_LoRA.jsonl`
- `result_task3_full_gossip_chatglm3-6b_LoRA.jsonl`

## 使用方法

### 1. 确保LoRA模型已训练

在运行推理前，确保已经使用 `multi_model_lora_train.py` 训练了对应的LoRA模型。

### 2. 运行推理脚本

```bash
cd /root/autodl-tmp/LLaMA-Factory-Megafake2
python scripts/multi_model_lora_inference.py
```

### 3. 检查结果

推理结果保存在 `megafakeTasks/` 目录下对应的任务文件夹中。

## 状态检查

脚本会自动进行以下检查：

1. **基础模型存在性**: 检查原始模型路径是否存在
2. **LoRA适配器存在性**: 检查对应的LoRA适配器是否存在
3. **结果文件存在性**: 如果结果文件已存在，跳过该任务

## 日志文件

日志文件命名格式：
```
logs/inference_LoRA_{model_name}_{dataset_name}_{timestamp}.log
```

日志内容包括：
- 开始时间
- 执行命令
- LoRA适配器路径
- 推理过程输出
- 结束时间和返回码

## 配置说明

### 模型配置

```python
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": "baichuan2",
    "/root/autodl-tmp/models/chatglm3-6b": "chatglm3", 
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": "llama3",
    "/root/autodl-tmp/models/Mistral-7B-v0.1": "mistral",
    "/root/autodl-tmp/models/Qwen1.5-7B": "qwen"
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

## 错误处理

- ❌ **模型不存在**: 跳过该模型的所有推理任务
- ❌ **LoRA适配器不存在**: 跳过该特定组合的推理任务
- ⏭️ **结果已存在**: 跳过已完成的推理任务
- 📝 **异常记录**: 所有错误都会记录在日志文件中

## 注意事项

1. **依赖关系**: 确保已安装 `vllm` 和相关依赖
2. **GPU内存**: LoRA推理相比全参数推理节省内存，但仍需足够的GPU内存
3. **适配器兼容性**: 确保LoRA适配器与基础模型版本兼容
4. **网络环境**: 脚本会自动配置HuggingFace镜像，但首次运行可能需要下载模型

## 测试验证

使用测试脚本验证配置：

```bash
python test_lora_inference.py
```

这会显示各个模型和数据集组合的LoRA适配器路径和存在状态。 