# Model Management Tools

这个模块包含了用于模型验证、测试和配置的统一工具集，为敏感性分析和模型实验提供支持。

## 📁 目录结构

```
scripts/model_utils/
├── README.md                       # 本文档
├── verification/                   # 模型验证工具
│   ├── final_model_verification.py # 模型验证脚本
│   ├── final_models_report.py      # 模型状态报告
│   ├── verify_all_models.py        # 批量模型验证
│   └── test_integrity.sh           # 文件完整性检查
├── testing/                        # 模型测试工具
│   ├── test_model_loading.py       # 模型加载测试
│   ├── test_model_with_llamafactory.py # LLaMA-Factory集成测试
│   ├── test_baichuan_llamafactory.py # 百川模型测试
│   ├── quick_model_check.sh        # 快速模型检查
│   └── test_baichuan_download.sh   # 百川下载测试
└── configs/                        # 配置文件
    ├── test_baichuan_config.yaml   # 百川基础配置
    └── test_baichuan_full.yaml     # 百川完整配置
```

## 🔧 模型验证工具

### 验证所有模型
```bash
# 批量验证所有可用模型
python scripts/model_utils/verification/verify_all_models.py

# 生成详细的模型状态报告
python scripts/model_utils/verification/final_models_report.py

# 检查模型文件完整性
bash scripts/model_utils/verification/test_integrity.sh
```

### 单个模型验证
```bash
# 验证特定模型
python scripts/model_utils/verification/final_model_verification.py --model_path /path/to/model
```

## 🧪 模型测试工具

### 基础测试
```bash
# 测试模型加载
python scripts/model_utils/testing/test_model_loading.py --model_path /path/to/model

# 快速检查所有模型
bash scripts/model_utils/testing/quick_model_check.sh

# 测试与LLaMA-Factory集成
python scripts/model_utils/testing/test_model_with_llamafactory.py
```

### 特定模型测试
```bash
# 百川模型测试
python scripts/model_utils/testing/test_baichuan_llamafactory.py
bash scripts/model_utils/testing/test_baichuan_download.sh
```

## ⚙️ 配置文件

### 百川模型配置
- `configs/test_baichuan_config.yaml` - 基础LLaMA-Factory配置
- `configs/test_baichuan_full.yaml` - 完整训练配置

### 使用配置示例
```bash
llamafactory-cli train scripts/model_utils/configs/test_baichuan_config.yaml
```

## 🔗 与敏感性分析的集成

### 使用验证结果
```bash
# 在运行敏感性分析前验证模型
python scripts/model_utils/verification/verify_all_models.py

# 运行敏感性分析（推荐先跑 dry-run，再遵循 multi_model_* → analyze 流程）
python scripts/multi_model_lora_train.py --dry-run --datasets task1_test200_balanced_glm
python scripts/multi_model_lora_inference.py --dry-run --datasets task1_test200_balanced_glm
python scripts/analyze_predictions.py --input sensitivity_analysis/outputs/task1/task1_test200_balanced_glm

> 历史脚本 `scripts/run_sensitivity_analysis.py` 已移至 `sensitivity_analysis/scripts/archive/` 仅供参考，缺失的 `ExperimentManager` 模块尚未恢复。
```

### 自定义模型配置
模型配置可以在 `scripts/utils/config.py` 中的 `MODEL_CONFIGS` 字典进行管理：

```python
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/your-model": ("template_name", True),
}
```

## 📊 常见用法

### 1. 新模型验证流程
```bash
# 1. 添加模型配置到 scripts/utils/config.py
# 2. 验证模型完整性
python scripts/model_utils/verification/final_model_verification.py --model_path /path/to/new/model

# 3. 测试LLaMA-Factory集成
python scripts/model_utils/testing/test_model_with_llamafactory.py --model_path /path/to/new/model

# 4. 运行敏感性分析
python scripts/run_analysis.py --mode quick --type all
# 或遵循 multi_model_* → analyze 流程
```

### 2. 模型问题诊断
```bash
# 快速检查所有模型状态
bash scripts/model_utils/testing/quick_model_check.sh

# 生成详细状态报告
python scripts/model_utils/verification/final_models_report.py

# 检查文件完整性
bash scripts/model_utils/verification/test_integrity.sh
```

### 3. 百川模型特殊处理
```bash
# 百川模型专用测试
python scripts/model_utils/testing/test_baichuan_llamafactory.py

# 使用百川配置
llamafactory-cli train scripts/model_utils/configs/test_baichuan_full.yaml
```

## 🛠️ 故障排除

### 常见问题
1. **模型加载失败**: 检查模型路径和权限
2. **内存不足**: 调整batch size或使用更小的模型
3. **配置错误**: 参考配置文件模板进行调整
4. **权限问题**: 确保模型文件有正确的读写权限

### 调试命令
```bash
# 详细日志模式
python scripts/model_utils/testing/test_model_loading.py --verbose

# 跳过错误的验证
python scripts/model_utils/verification/verify_all_models.py --skip-errors
```

## 📝 更新日志

- **2025-11-15**: 创建统一的模型管理工具结构
- **2025-11-15**: 整合所有零散的模型脚本到统一目录
- **2025-11-15**: 与敏感性分析工具完全集成

## 🤝 贡献指南

1. 新增模型验证工具时，添加到对应的子目录
2. 保持文档更新，说明新工具的用法
3. 遵循现有的代码结构和命名约定
4. 测试新工具与现有敏感性分析的兼容性

---

这些工具为整个项目提供了统一的模型管理能力，确保模型验证、测试和配置的一致性和可维护性。
