# LLM敏感性分析

这个目录包含了用于进行大语言模型敏感性分析的所有工具和脚本。

## 📁 目录结构

```
sensitivity_analysis/
├── scripts/              # 核心脚本
│   ├── run_analysis.py   # 主入口脚本（推荐使用）
│   ├── core.py           # 核心分析逻辑
│   ├── monitor.py        # 系统监控工具
│   └── archive/          # 历史脚本（run_sensitivity_analysis 等，仅供参考）
├── configs/              # 配置文件
├── data/                 # 数据文件
├── outputs/              # LoRA/推理产物（对外统一 root，megafakeTasks 通过符号链接指向这里）
├── results/              # 结构化结果与可视化（非 CSV 资产才纳入版本控制）
├── logs/                 # 训练/推理/分析日志
└── experiments/archive/  # 旧 ExperimentManager 产物（只读，旧入口仅留在 archive）

> ⚠️ 旧的 `run_sensitivity_analysis.py` 及其依赖模块已迁移到 `sensitivity_analysis/scripts/archive/`，缺失的 `ExperimentManager` 组件暂不维护，仅作历史参考；请使用 `scripts/multi_model_*` + `analyze_predictions.py` 或 `scripts/sa.py` 运行新的流程。
```

## ⚗️ 实验标准流程

1.  **Mini Test100 链路 (冒烟测试)**
    *   使用 200 条平衡样本 (100正/100负) 进行快速验证。
    *   验证训练、推理、分析全链路是否畅通。
    *   **命令**: `python scripts/sa.py quick --dry-run` 然后执行 `scripts/multi_model_*` 流程。

2.  **正式大规模链路**
    *   在 Mini 验证通过后，扩展至 1k, 2k, 5k, 10k, 20k 数据规模。
    *   生成全量 LoRA 权重和推理结果。

3.  **分析与绘图 (Analyze & Plot)**
    *   解析 JSONL 结果，生成 CSV 指标。
    *   绘制折线图/柱状图/散点图。
    *   **命令**: `python scripts/analyze_predictions.py --plot ...`

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
