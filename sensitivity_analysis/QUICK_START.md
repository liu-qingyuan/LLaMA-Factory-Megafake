# 敏感性分析快速开始

## 🚀 快速运行（5分钟上手）

### 1. 验证环境
```bash
# 运行模拟测试（无需GPU）
python scripts/mock_sensitivity_test.py
```

### 2. 快速真实测试
```bash
# 超小规模验证测试（约15分钟，10-100样本）
python scripts/sensitivity_analysis.py --quick-test
```

### 3. 完整实验
```bash
# 分阶段运行完整分析（可中断恢复）
python scripts/run_staged_sensitivity.py

# 或一次性运行
python scripts/sensitivity_analysis.py --all
```

## 📋 运行前检查清单

- [ ] GPU显存 > 16GB
- [ ] 磁盘空间 > 100GB
- [ ] 模型已下载到 `/root/autodl-tmp/models/`
- [ ] 数据集准备完成
- [ ] LLaMA-Factory环境已配置

## ⚡ 常见问题快速修复

| 问题 | 解决方案 |
|------|----------|
| VLLM LoRA rank错误 | 已修复，使用rank [8, 16] |
| CUDA内存不足 | 减小batch_size到8 |
| 训练超时 | 已移除超时限制 |
| 想重新开始 | `python scripts/run_staged_sensitivity.py --reset` |

## 📊 预期结果

- **快速验证**: 10-30分钟，6-8个实验（10-100样本）
- **完整分析**: 24-72小时，50-80个实验
- **输出位置**: `megafakeTasks/sensitivity_analysis/`

## 🔗 详细文档

完整说明请查看: [README.md](README.md)