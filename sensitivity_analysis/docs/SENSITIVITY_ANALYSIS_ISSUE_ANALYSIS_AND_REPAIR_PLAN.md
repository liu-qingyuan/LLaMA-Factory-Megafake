# LLaMA-Factory 敏感性分析系统问题诊断与修复PRD

## 1. 问题概述

### 1.1 实验基本信息
- **实验日期**: 2025-11-18 13:05:12 - 17:10:25
- **运行时长**: 约4小时5分钟
- **实验脚本**: `sensitivity_analysis/scripts/original_sensitivity_analysis.py`
- **命令参数**: `--all --all`
- **涉及模型**: 5个模型（Qwen1.5-7B, Meta-Llama-3.1-8B-Instruct, Baichuan2-7B-Chat, Mistral-7B-v0.1, chatglm3-6b）
- **数据集**: task1_small_glm

### 1.2 核心问题识别
实验运行完成，但所有实验结果完全相同，存在严重的数据异常问题：

**异常指标**:
- Accuracy: 0.736375 (所有实验完全相同)
- F1-macro: 0.42408753869411847 (所有实验完全相同)
- Precision: 0.3681875 (所有实验完全相同)
- Recall: 0.5 (所有实验完全相同)
- Memory: 12.0GB (所有实验完全相同)

## 2. 问题根本原因分析

### 2.1 关键问题1: 评估逻辑错误

**问题位置**: `original_sensitivity_analysis.py:159-161`
```python
# 简化的标签处理
pred_label = 1 if "fake" in pred.lower() else 0
true_label = 1 if "fake" in true.lower() else 0
```

**问题描述**:
- 标签处理逻辑过于简化，无法处理复杂的预测结果
- 所有非"fake"的预测都被归类为0（legitimate）
- 没有考虑模型输出的多样性（如 "real", "true", "not fake" 等）

### 2.2 关键问题2: 数据集大小与评估不匹配

**发现**: 所有实验的评估文件都包含8000行数据
- 实验 `data_50` → 8000行评估结果
- 实验 `data_1000` → 8000行评估结果
- 实验 `data_10000` → 8000行评估结果

**问题原因**: VLLM评估使用了完整的数据集进行推理，而不是根据训练数据大小调整评估数据大小

### 2.3 关键问题3: 实验范围不完整

**预期**: 5个模型的完整敏感性分析
**实际**: 只完成了2个模型（Qwen1.5-7B和Meta-Llama-3.1-8B-Instruct）
- 缺失模型: Baichuan2-7B-Chat, Mistral-7B-v0.1, chatglm3-6b

### 2.4 关键问题4: 训练效果验证缺失

**问题**: 没有验证LoRA训练是否真正成功
- 可能使用了相同的base model进行推理
- 没有检查adapter是否正确加载
- 训练过程可能失败但脚本继续执行评估

### 2.5 关键问题5: 评估指标计算错误

**问题位置**: `original_sensitivity_analysis.py:171-173`
```python
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
```

**问题描述**: 对于二分类问题使用macro平均可能导致不准确的指标

## 3. 修复任务清单 (TODO List)

### 3.1 高优先级修复任务

#### 任务1: 修复评估逻辑
- [ ] **预估工作量**: 2小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 改进标签处理逻辑，支持更多样的预测格式
  2. 添加标签映射字典，支持同义词识别
  3. 增加预测结果的清洗和标准化
- [ ] **验收标准**:
  - 支持至少10种不同的预测格式（"fake", "false", "not real", "legitimate", "true", "real", "not fake", "authentic", "credible", "reliable"）
  - 单元测试覆盖率达到90%

#### 任务2: 修复数据集大小匹配问题
- [ ] **预估工作量**: 3小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 修改VLLM评估脚本，根据训练数据大小调整评估数据
  2. 实现数据集分层采样，确保评估数据与训练数据的一致性
  3. 添加数据大小验证机制
- [ ] **验收标准**:
  - 评估数据大小与训练数据大小成比例
  - 保持类别分布平衡
  - 采样结果可重现

#### 任务3: 添加训练验证机制
- [ ] **预估工作量**: 4小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 检查LoRA训练是否成功完成
  2. 验证adapter文件是否正确生成
  3. 添加模型参数变化验证
  4. 实现训练前后模型输出对比
- [ ] **验收标准**:
  - 训练失败时自动跳过评估
  - adapter文件完整性检查
  - 模型参数差异验证

### 3.2 中优先级修复任务

#### 任务4: 完善实验覆盖度
- [ ] **预估工作量**: 6小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 调查缺失模型的原因
  2. 修复模型加载问题
  3. 添加模型可用性预检查
  4. 实现实验进度跟踪和断点续传
- [ ] **验收标准**:
  - 5个模型全部参与实验
  - 实验中断后可从断点继续
  - 详细的实验进度报告

#### 任务5: 优化指标计算
- [ ] **预估工作量**: 1小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 改进二分类指标计算方式
  2. 添加更多评估指标（AUC-ROC, confusion_matrix等）
  3. 实现详细的评估报告生成
- [ ] **验收标准**:
  - 准确的二分类指标计算
  - 完整的评估报告
  - 可视化混淆矩阵

### 3.3 低优先级修复任务

#### 任务6: 系统优化和重构
- [ ] **预估工作量**: 8小时
- [ ] **负责人**: 待分配
- [ ] **具体内容**:
  1. 重构脚本结构，提高代码可维护性
  2. 添加详细的日志记录
  3. 实现配置文件验证
  4. 优化内存使用和性能
- [ ] **验收标准**:
  - 代码模块化程度提升
  - 详细的操作日志
  - 内存使用优化20%以上

## 4. 技术实现细节

### 4.1 标签处理改进方案

```python
def improved_label_processing(prediction: str, ground_truth: str) -> tuple:
    """改进的标签处理函数"""

    # 定义标签映射
    FAKE_LABELS = ['fake', 'false', 'not real', 'untrue', 'misleading', 'deceptive']
    LEGITIMATE_LABELS = ['legitimate', 'true', 'real', 'not fake', 'authentic', 'credible', 'reliable', 'accurate']

    pred = prediction.lower().strip()
    true = ground_truth.lower().strip()

    # 检查fake标签
    if any(fake_word in pred for fake_word in FAKE_LABELS):
        pred_label = 1
    elif any(legit_word in pred for legit_word in LEGITIMATE_LABELS):
        pred_label = 0
    else:
        # 对于无法确定的情况，使用简单的关键词匹配
        pred_label = 1 if "fake" in pred else 0

    # 真实标签处理
    true_label = 1 if "fake" in true else 0

    return pred_label, true_label
```

### 4.2 数据集大小匹配方案

```python
def get_evaluation_dataset_size(train_data_size: int, total_eval_size: int = 8000) -> int:
    """根据训练数据大小计算评估数据大小"""

    # 定义评估数据大小与训练数据的比例
    EVAL_RATIO = 0.2  # 评估数据为训练数据的20%
    MIN_EVAL_SIZE = 100  # 最小评估数据大小
    MAX_EVAL_SIZE = 2000  # 最大评估数据大小

    eval_size = max(MIN_EVAL_SIZE, min(MAX_EVAL_SIZE, int(train_data_size * EVAL_RATIO)))

    return eval_size
```

### 4.3 训练验证方案

```python
def validate_training_completion(adapter_path: str, model_name: str) -> bool:
    """验证LoRA训练是否成功完成"""

    # 检查adapter文件是否存在
    adapter_files = [
        "adapter_model.bin",  # 或 adapter_model.safetensors
        "adapter_config.json",
        "training_args.json"
    ]

    for file_name in adapter_files:
        file_path = Path(adapter_path) / file_name
        if not file_path.exists():
            logging.error(f"Missing adapter file: {file_path}")
            return False

    # 检查文件大小，确保不是空文件
    for file_name in adapter_files:
        file_path = Path(adapter_path) / file_name
        if file_path.stat().st_size == 0:
            logging.error(f"Empty adapter file: {file_path}")
            return False

    return True
```

## 5. 测试计划

### 5.1 单元测试
- [ ] 标签处理函数测试
- [ ] 数据集大小计算测试
- [ ] 训练验证函数测试
- [ ] 指标计算函数测试

### 5.2 集成测试
- [ ] 完整的敏感性分析流程测试
- [ ] 不同数据大小的实验测试
- [ ] 不同模型的实验测试

### 5.3 回归测试
- [ ] 修复后的结果与预期结果对比
- [ ] 性能基准测试
- [ ] 内存使用测试

## 6. 风险评估

### 6.1 高风险
- **训练数据丢失**: 修复过程中可能影响现有的训练结果
- **模型兼容性**: 修复可能影响模型加载

### 6.2 中风险
- **性能下降**: 改进的验证逻辑可能增加运行时间
- **资源消耗**: 更详细的验证可能增加内存使用

### 6.3 低风险
- **配置兼容性**: 新的配置可能与现有脚本不兼容

## 7. 成功标准

### 7.1 功能标准
- [ ] 所有实验结果不再完全相同
- [ ] 数据大小与评估结果匹配
- [ ] 所有5个模型都能参与实验
- [ ] 训练失败时能够正确识别和处理

### 7.2 质量标准
- [ ] 单元测试覆盖率 > 90%
- [ ] 代码审查通过
- [ ] 文档完整性检查通过

### 7.3 性能标准
- [ ] 单个实验运行时间 < 30分钟
- [ ] 内存使用 < 16GB
- [ ] 实验成功率 > 95%

## 8. 时间计划

| 阶段 | 任务 | 预计时间 | 开始时间 | 结束时间 |
|------|------|----------|----------|----------|
| 1 | 高优先级修复 | 9小时 | 待定 | 待定 |
| 2 | 中优先级修复 | 7小时 | 待定 | 待定 |
| 3 | 测试和验证 | 4小时 | 待定 | 待定 |
| 4 | 文档更新 | 2小时 | 待定 | 待定 |
| **总计** | | **22小时** | | |

## 9. 交付物

### 9.1 代码交付物
- [ ] 修复后的 `original_sensitivity_analysis.py`
- [ ] 新增的验证和工具函数
- [ ] 单元测试代码
- [ ] 配置文件更新

### 9.2 文档交付物
- [ ] 问题修复报告
- [ ] 测试报告
- [ ] 更新的用户使用指南
- [ ] 技术文档

### 9.3 结果交付物
- [ ] 重新运行后的敏感性分析结果
- [ ] 正确的可视化图表
- [ ] 实验数据和分析报告

## 10. 后续改进建议

1. **实现真正的增量式实验**: 支持实验的暂停、恢复和断点续传
2. **添加分布式训练支持**: 支持多GPU并行训练
3. **实现更智能的超参数搜索**: 使用贝叶斯优化等方法
4. **添加更详细的可视化**: 包括训练曲线、混淆矩阵热图等
5. **实现自动化测试**: CI/CD集成的自动化测试流程

---

**文档版本**: 1.0
**创建日期**: 2025-11-21
**最后更新**: 2025-11-21
**文档状态**: 待审核
**下一步**: 分配负责人并开始实施修复任务