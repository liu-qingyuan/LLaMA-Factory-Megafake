# LLM敏感性分析

这个目录包含了用于进行大语言模型敏感性分析的所有工具和脚本。

## 📁 目录结构

```
sensitivity_analysis/
├── scripts/              # 核心脚本
│   ├── run_analysis.py   # 主入口脚本（推荐使用）
│   ├── core.py          # 核心分析逻辑
│   └── monitor.py       # 系统监控工具
├── configs/             # 配置文件
├── data/               # 数据文件
├── results/            # 结果输出
└── logs/               # 日志文件
```

## 🚀 快速开始

### 1. 快速测试
```bash
# 测试配置是否正确
python sensitivity_analysis/scripts/run_analysis.py --test-only

# 快速验证模式（小数据集，2个模型）
python sensitivity_analysis/scripts/run_analysis.py --mode quick --type all
```

### 2. 完整分析
```bash
# 完整数据集分析
python sensitivity_analysis/scripts/run_analysis.py --mode full --type all

# 内存优化模式（推荐用于完整数据集）
python sensitivity_analysis/scripts/run_analysis.py --mode full --memory-optimized
```

### 3. 系统监控
```bash
# 在另一个终端监控资源使用
python sensitivity_analysis/scripts/monitor.py
```

## 📋 参数说明

### 运行模式 (--mode)
- `quick`: 快速验证模式，使用小数据集，适合测试
- `full`: 完整分析模式，使用完整数据集

### 分析类型 (--type)
- `data`: 数据敏感性分析
- `lora`: LoRA参数敏感性分析
- `training`: 训练参数敏感性分析
- `all`: 所有分析类型（默认）

### 内存优化
- `--memory-optimized`: 启用内存优化（减小批次大小，分块处理）

## 🔧 故障排除

### 内存不足
如果遇到GPU内存不足，使用内存优化模式：
```bash
python sensitivity_analysis/scripts/run_analysis.py --memory-optimized
```

### 监控资源使用
运行系统监控脚本实时查看资源使用情况：
```bash
python sensitivity_analysis/scripts/monitor.py
```

### 检查配置
运行测试脚本验证配置：
```bash
python sensitivity_analysis/scripts/run_analysis.py --test-only
```
