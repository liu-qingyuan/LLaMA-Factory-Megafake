# LLM敏感性分析 PRD · TODO版

> 所有资产已下沉至 `sensitivity_analysis/`，该清单是团队在 2025-11 的最新交付视图。

## 0. 项目北极星
- [ ] 提供一套可复现、可扩展的 LLM 敏感性分析流水线，覆盖数据量/参数/模型三大维度
- [ ] 输出结构化的训练产物（LoRA、评估 JSONL、可视化）并沉淀到 `megafakeTasks/…` 与 `sensitivity_analysis/results`
- [ ] 配套文档、监控、冒烟测试，确保长时实验的可见性与稳定性
- [ ] **当前优先级**：先完成一个 “100 正 / 100 负” 的迷你实验链路，复用 `sample_test100_multi_reasoning_task1.py` 的采样策略，验证端到端流程（采样→训练→推理→分析→绘图）无误，再扩展至 1k/2k/5k/10k/20k

## 1. 目录与执行链路
- [x] 迁移 PRD 与所有文档到 `sensitivity_analysis/docs/`
- [ ] **迷你通路验证**：以 `sample_test100_multi_reasoning_task1.py` 为蓝本，在 Task1 数据上生成 100 正 / 100 负的平衡数据集（可位于 `data/data_table/task1/alpaca_test100_balanced/`），并贯穿 `multi_model_lora_train.py` → `multi_model_lora_inference.py` → `analyze_predictions*.py`
- [ ] 成功跑通迷你链路后，再恢复原始多模型流水线（1k~20k），保持同样的日志/产物结构；`scripts/sa.py` 与 `sensitivity_analysis/scripts/*` 标记为实验性封装
- [ ] 清理/归档 `run_sensitivity_analysis.py`、`experiments/*`、`sensitivity_analysis/scripts/original_sensitivity_analysis.py` 等 AI 生成但缺失依赖的代码，避免误导
- [ ] 所有训练、推理、评估产物统一落盘到 `megafakeTasks/{taskX}/...` 与 `logs/sensitivity_analysis`，并在 CLI 输出中打印绝对路径
- [ ] 在 `scripts/multi_model_lora_train.py` / `multi_model_inference.py` 新增 `--dry-run` 或 `--check-config`，启动前验证模型/数据集/输出目录；迷你通路也需提供该能力

## 2. 模型与数据覆盖
- [x] 识别可用模型目录 `/root/autodl-tmp/models/{Meta-Llama-3.1-8B-Instruct, Qwen1.5-7B, Qwen1.5-72B, chatglm3-6b, Mistral-7B-v0.1, Baichuan2-7B-Chat}`
- [ ] 校准 Hugging Face 名称与本地路径映射（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct` ↔ `/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct`），以便复现实验描述
- [x] 使用 `convert_task{1,2,3}.py` 产出的 Alpaca 数据 (`data/data_table/task*/alpaca_full/…`) 作为统一数据源
- [x] **迷你数据集**：复用 `sample_test100_multi_reasoning_task1.py` 的逻辑，在 Task1 Alpaca 数据上生成“100 正 / 100 负”*1份*基准集（示例：`data/data_table/task1/alpaca_test100_balanced/alpaca_megafake_glm_test200_balanced.json`），支持 CoT/FS/ZS 等推理格式，并记录路径、样本分布、随机种子
- [ ] **大规模采样管线**：在迷你链路验证通过后，实现/修复 `multi_model_lora_train.py` 内的采样逻辑，生成 1k/2k/5k/10k/20k 子集（含日志与落盘位置）
- [ ] 为 Task2/Task3 建立至少一个基准规模（例如 5k、10k）以便横向对照；ChatGLM、Baichuan、Llama-3.1 需在这些任务上完成样本验证
- [ ] 模型模板/`trust_remote_code` 配置在 `scripts/multi_model_*` 与 `sensitivity_analysis/configs/config.py` 保持一致，避免训练/推理参数漂移

## 3. 实验与评估能力
- [ ] **迷你链路（Test100）**：在 200 条样本上跑通 `multi_model_lora_train.py` → `multi_model_lora_inference.py` → `analyze_predictions*.py` → 绘图，验证 CLI、日志、产物路径、评估、绘图全部正常
- [ ] 为迷你数据集定制训练/推理参数（如 batch size、max_samples），并在输出中标明“Mini Test100 流程”
- [ ] `analyze_predictions_lora.py` 升级：支持 Mini Test100 模式下的标签映射、AUC、混淆矩阵、结果导出；后续可复用相同逻辑处理 1k~20k
- [ ] 在迷你链路确认无误后，再切换到 1k/2k/5k/10k/20k 全量实验；日志结构与结果导出保持一致，便于比较
- [ ] 训练结束即验证 `adapter_model.safetensors`、`adapter_config.json` 是否存在；若缺失则记录失败并停止后续推理
- [ ] 在 `megafakeTasks/task*/<model>/<size>/` 或集中目录生成 `experiment_summary.json`，内容含：模型、数据规模、训练耗时、推理耗时、F1/Accuracy、日志/产物路径

## 4. 可视化与报告
- [ ] 更新 `SENSITIVITY_ANALYSIS_README.md` 与 `sensitivity_analysis/docs/index.md`，明确正式流程：**Mini Test100 链路 → 正式 1k~20k 链路 → analyze → 绘图 → 论文图表**
- [ ] `analyze_predictions*.py` 增加绘图开关，优先实现 Mini Test100 的折线/柱状/散点图输出到 `sensitivity_analysis/results/plots`；确认正确后再扩展至 1k~20k
- [ ] 自动生成论文草图所需的 Markdown/LaTeX 片段：包括图表引用、主要发现（拐点、最优模型）、以及可直接用于“实验设置/结果分析”章节的文字摘要

## 5. 工程与质量保障
- [x] 统一使用 `Makefile` 的 `quality`/`style`/`test` 目标及 `ruff`、`pytest` 规范
- [ ] 为 `scripts/multi_model_*`、`analyze_predictions*.py` 添加强校验（参数解析、目录存在性、配置一致性）的单元测试
- [ ] 在所有 CLI（尤其 multi_model_* 与 vllm_infer）中打印日志/产物路径和下一步命令提示
- [ ] 建立 `logs/{train,infer,analyze}` 轮转策略，避免长时间运行导致磁盘爆满；必要时上传关键信息到 `experiment_summary.json`

## 6. 风险与监控
- [ ] GPU/磁盘资源不足：在冒烟阶段检测剩余显存、磁盘配额
- [ ] 依赖失效：锁定 `requirements.txt` 与 `pyproject.toml` 的关键包版本，配置 HF 镜像
- [ ] 长时间实验可观测性：补全 `sensitivity_analysis/scripts/monitor.py` 使用文档，并提供 Prometheus 友好输出

## 7. 里程碑（参考）
- [x] **Phase 1** 基础结构搭建（入口脚本、配置模块、数据采样骨架）
- [ ] **Phase 2** 数据敏感性与采样报告完成
- [ ] **Phase 3** 参数敏感性、LoRA/训练/推理模块打通
- [ ] **Phase 4** 模型对比、鲁棒性/稳定性指标齐备
- [ ] **Phase 5** 可视化 & 报告自动化
- [ ] **Phase 6** 全链路测试、性能优化、文档交付

> 更新人：团队共识；本 TODO 文档需随每次结构调整同步修订，确保审核时可直接对照状态打钩。
