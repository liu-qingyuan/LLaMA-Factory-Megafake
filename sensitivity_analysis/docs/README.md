# LLM敏感性分析框架文档

## 📋 项目概述

本项目是LLaMA-Factory-Megafake的敏感性分析框架，用于系统性地评估不同大语言模型在假新闻检测任务上的敏感性。

## 🚀 快速开始

### 最简单的使用方式

```bash
# 查看帮助
python scripts/sa.py

# 快速测试配置（推荐）
python scripts/sa.py quick

# 完整分析
python scripts/sa.py full

# 系统监控
python scripts/sa.py monitor

# 模型验证
python scripts/sa.py verify
```

### 高级用法

```bash
# 使用完整模块
python sensitivity_analysis/scripts/run_analysis.py --help

# 内存优化的完整分析
python sensitivity_analysis/scripts/run_analysis.py --mode full --memory-optimized
```

## 📁 目录结构

```
sensitivity_analysis/                    # 敏感性分析主目录
├── configs/                            # 配置文件
│   └── config.py                      # 统一配置模块 (MODEL_CONFIGS, DATASET_CONFIGS)
├── scripts/                           # 核心脚本
│   ├── run_analysis.py                # 主入口脚本 (完整功能)
│   ├── core.py                        # 核心分析逻辑
│   ├── original_sensitivity_analysis.py  # 原始分析脚本 (已修复导入)
│   ├── quick_test.py                  # 快速测试脚本
│   ├── monitor.py                     # 系统监控工具
│   └── test_vllm_fix.py               # VLLM修复测试脚本
├── model_utils/                       # 模型工具
│   └── verify_models.py               # 模型验证工具
├── utils/                             # 工具函数 (统一导入)
│   └── __init__.py                    # 统一配置导入
├── docs/                              # 📚 文档目录 (当前文件)
│   ├── README.md                      # 本文档
│   ├── SETUP.md                       # 环境设置指南
│   └── TROUBLESHOOTING.md             # 问题排查指南
├── data/                              # 数据目录
├── results/                           # 结果输出
└── logs/                              # 日志文件

scripts/                               # 🚀 简化入口脚本
└── sa.py                              # 超简洁统一入口 (test|quick|full|monitor|verify)
```

## 🔧 环境设置

### 依赖安装

```bash
# 基础依赖
pip install -e ".[torch,metrics]" --no-build-isolation

# 完整依赖（推荐）
pip install -e ".[torch,metrics,deepspeed,vllm,quantization]" --no-build-isolation
```

### 环境变量

```bash
# HuggingFace镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 网络加速（如果可用）
source /etc/network_turbo

# GPU选择
export CUDA_VISIBLE_DEVICES=0,1
```

## 📊 支持的模型和数据集

### 支持的模型

- ✅ **LLaMA系列**: LLaMA-3.1-8B-Instruct
- ✅ **Qwen系列**: Qwen1.5-7B, Qwen1.5-72B
- ✅ **Mistral系列**: Mistral-7B-v0.1
- ✅ **ChatGLM系列**: ChatGLM3-6b
- ✅ **Baichuan系列**: Baichuan2-7B-Chat

### 支持的数据集

- **Task 1**: 假新闻检测 (task1_small_glm, task1_full_glm)
- **Task 2**: 细粒度分类 (task2_style_based_*, task2_content_based_*)
- **Task 3**: 多源信息验证 (task3_full_*, task1_test100_*)

## ⚡ 快速测试配置

为了快速验证流程完整性，使用了极小规模的数据：

### 数据规模
- **数据敏感性分析**: [10, 100] 样本
- **LoRA参数敏感性**: 100 样本
- **训练参数敏感性**: 100 样本

### 参数简化
- **LoRA rank**: 单个值 [16] (避免rank 32的VLLM错误)
- **学习率**: 单个值 [2e-5]
- **批次大小**: 单个值 [8]
- **训练轮数**: 单个值 [1]

## 🐛 常见问题排查

### VLLM推理错误

**问题**: `No module named 'utils.config'`

**解决方案**: 已修复导入路径问题，现在使用多重备选导入机制。

**验证**: 运行 `python sensitivity_analysis/scripts/test_vllm_fix.py` 确认修复效果。

### CUDA内存错误

**问题**: `CUDA error: device-side assert triggered`

**解决方案**:
1. 减少批次大小
2. 使用更小的LoRA rank
3. 启用内存优化模式

### 模型下载问题

**Baichuan2-7B-Chat下载**:
- 使用多源下载策略
- 配置HF镜像加速
- 参考模型管理脚本

## 📈 实验结果分析

### 最新修复成果 (2025-11-17)

#### ✅ 已完成的修复
1. **目录结构重组**: 完成16+个脚本的整理和统一管理
2. **导入路径修复**: 解决了VLLM推理的配置导入错误
3. **配置统一管理**: 创建了智能配置导入模块
4. **简化入口**: 提供一键执行的 `sa.py` 脚本

#### 📊 系统性能指标
- ✅ **成功率**: 敏感性分析框架正常运行
- ✅ **VLLM推理**: 导入路径已修复，能正常运行
- ✅ **配置管理**: 支持多路径自动回退
- ✅ **Flash Attention**: 2.0正常启用

#### 🔧 推荐配置
```bash
# 最优性价比模型: Qwen1.5-7B
# 推荐数据规模: 快速测试[10,100], 完整分析[1000,50000]
# LoRA配置: r=16, alpha=32, dropout=0.05
```

## 📚 更多文档

### 敏感性分析文档
- **[文档索引](index.md)** - 📖 完整文档导航和快速查找
- **[环境设置](SETUP.md)** - ⚙️ 详细的安装和环境配置指南
- **[问题排查](TROUBLESHOOTING.md)** - 🔧 常见问题解决方案和调试技巧

### 项目文档
- **[产品需求文档](PRD.md)** - 详细的项目需求和技术规范
- **[项目主文档](../../CLAUDE.md)** - 整个项目架构和开发指南

## 🤝 贡献指南

### 代码结构
- 模块化设计，代码复用率 > 80%
- 统一的错误处理和日志记录
- 完整的测试覆盖

### 开发流程
1. 使用快速测试验证功能
2. 运行完整分析进行性能评估
3. 提交前确保所有测试通过

---

**文档版本**: v2.0
**最后更新**: 2025-11-17
**维护状态**: ✅ 活跃维护中
