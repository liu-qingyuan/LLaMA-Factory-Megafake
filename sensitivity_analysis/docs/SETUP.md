# 环境设置指南

## 🚀 快速部署

### 1. 基础环境要求

**硬件要求**:
- **GPU**: A100 40GB (推荐), RTX 4090 (最低)
- **内存**: 32GB+ (推荐), 16GB (最低)
- **存储**: 200GB+ SSD
- **网络**: 稳定的互联网连接

**软件要求**:
- Python 3.8+
- CUDA 11.8+
- Git

### 2. 安装步骤

#### 步骤1: 克隆项目
```bash
git clone <repository-url>
cd LLaMA-Factory-Megafake
```

#### 步骤2: 安装依赖
```bash
# 基础安装
pip install -e ".[torch,metrics]" --no-build-isolation

# 完整安装（推荐）
pip install -e ".[torch,metrics,deepspeed,vllm,quantization]" --no-build-isolation
```

#### 步骤3: 配置环境变量
```bash
# HuggingFace镜像加速（国内用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 网络加速（如果可用）
source /etc/network_turbo

# GPU选择
export CUDA_VISIBLE_DEVICES=0,1
```

### 3. 模型设置

#### 支持的模型
```bash
# 模型存储目录
mkdir -p /root/autodl-tmp/models

# 推荐模型列表
MODELS=(
    "Qwen1.5-7B"
    "Meta-Llama-3.1-8B-Instruct"
    "Baichuan2-7B-Chat"
    "Mistral-7B-v0.1"
    "chatglm3-6b"
)
```

#### 模型下载脚本示例
```bash
#!/bin/bash
# download_models.sh

HF_MIRRORS=(
    "https://huggingface.co"
    "https://hf-mirror.com"
)

for model in "${MODELS[@]}"; do
    echo "下载模型: $model"
    # 使用git lfs或其他下载方法
done
```

### 4. 数据集设置

#### 数据集目录结构
```
data/
├── data_table/
│   ├── task1/                          # 假新闻检测
│   │   ├── alpaca_full/
│   │   ├── small_8k/
│   │   └── alpaca_test100_*/
│   ├── task2/                          # 细粒度分类
│   └── task3/                          # 多源验证
└── dataset_info.json                   # 数据集注册表
```

#### 数据集配置
数据集配置在 `data/dataset_info.json` 中定义，确保所有数据集路径正确。

### 5. 验证安装

#### 快速测试
```bash
# 验证基础功能
python scripts/sa.py test

# 验证模型加载
python sensitivity_analysis/model_utils/verify_models.py

# 验证VLLM修复
python sensitivity_analysis/scripts/test_vllm_fix.py
```

#### 功能测试
```bash
# 运行快速敏感性分析
python scripts/sa.py quick

# 检查系统状态
python scripts/sa.py monitor
```

## 🔧 高级配置

### 1. VLLM加速配置

```bash
# 启用VLLM
VLLM_ENABLED=true python scripts/multi_model_inference.py

# VLLM配置示例
python scripts/vllm_infer.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen1.5-7B \
  --template qwen \
  --dataset task1_small_glm \
  --vllm_config '{"tensor_parallel_size": 1}'
```

### 2. 内存优化

```python
# 内存优化模式
python sensitivity_analysis/scripts/run_analysis.py \
  --mode quick \
  --memory-optimized
```

### 3. 多GPU配置

```bash
# 分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config.yaml

# 数据并行
export NCCL_DEBUG=INFO
```

### 4. 性能调优

#### 推荐参数
```yaml
# LoRA参数
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# 训练参数
learning_rate: 2e-5
batch_size: 8
gradient_accumulation_steps: 8

# 优化参数
flash_attn: "auto"
fp16: true
ddp_timeout: 180000000
```

## 🐛 常见问题解决

### 1. 导入错误

**问题**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
# 确保在项目根目录
cd /root/autodl-tmp/LLaMA-Factory-Megafake

# 重新安装
pip install -e ".[torch,metrics,deepspeed,vllm,quantization]" --no-build-isolation
```

### 2. CUDA错误

**问题**: `CUDA error: device-side assert triggered`

**解决方案**:
```bash
# 减少批次大小
export BATCH_SIZE=4

# 使用CPU作为备选
export CUDA_VISIBLE_DEVICES=""

# 检查GPU内存
nvidia-smi
```

### 3. 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 启用梯度检查点
export GRADIENT_CHECKPOINTING=true

# 减少模型并行
export TENSOR_PARALLEL_SIZE=1

# 使用混合精度
export FP16=true
```

### 4. 网络问题

**问题**: 下载模型/数据集失败

**解决方案**:
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 设置代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 增加重试次数
export HF_HUB_OFFLINE=false
```

## 📊 性能基准

### 硬件性能参考

| GPU型号 | 推理速度 | 训练速度 | 内存使用 | 推荐用途 |
|---------|----------|----------|----------|----------|
| A100 40GB | 3x | 1x | 100% | 生产环境 |
| RTX 4090 | 2x | 0.8x | 80% | 开发测试 |
| RTX 3090 | 1.5x | 0.6x | 60% | 小规模实验 |

### 推荐配置

#### 快速开发
- GPU: RTX 3090
- 模型: Qwen1.5-7B
- 数据: 1K-5K样本
- LoRA: r=16, alpha=32

#### 生产环境
- GPU: A100 40GB
- 模型: LLaMA-3.1-8B
- 数据: 10K-50K+样本
- LoRA: r=32, alpha=64

---

**文档版本**: v1.0
**最后更新**: 2025-11-17
**维护状态**: ✅ 活跃维护中