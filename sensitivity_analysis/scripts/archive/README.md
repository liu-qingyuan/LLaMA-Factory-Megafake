# 历史流水线（Archive）

该目录存放早期 AI 生成的敏感性分析流水线脚本，例如 `run_sensitivity_analysis.py` 及其依赖的 `ExperimentManager/DataSensitivityAnalyzer/*`。这些模块在当前仓库中缺失，脚本无法直接运行，仅作为结构参考或回溯使用。

## 现状
- ✅ 主要入口脚本已移入本目录，并在 PRD 中标注为“历史样例，仅供参考”。
- ⚠️ 依赖的 `ExperimentManager/`、`DataSensitivityAnalyzer/`、`ParameterSensitivityAnalyzer/` 等模块未随仓库提供，运行会触发 `ModuleNotFoundError`。
- 📦 历史产物已搬迁到 `sensitivity_analysis/experiments/archive/{sensitivity_analysis,real_sensitivity_analysis}`，后续会逐步迁移/软链到 `sensitivity_analysis/outputs`。

## 推荐做法
1. **新实验**：请使用 `scripts/multi_model_lora_train.py`、`scripts/multi_model_lora_inference.py`、`scripts/analyze_predictions*.py` 及即将推出的 `scripts/sa.py` 冒烟工具。
2. **学习旧结构**：如果需要参考旧的配置示例或结果格式，可阅读本目录脚本或 `sensitivity_analysis/experiments/archive/real_sensitivity_analysis` 中的日志，但不要直接复用。
3. **文档提示**：在 PRD/README/CLI 中明确告知新人“Archive 脚本不可运行”，避免重复报错。

若未来重新实现 ExperimentManager，请在本目录添加 `RESTORE.md` 描述修复计划，同时更新 PRD 状态。
