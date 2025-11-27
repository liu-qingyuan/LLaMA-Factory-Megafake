# LLM敏感性分析 PRD · TODO版

> 所有资产已下沉至 `sensitivity_analysis/`，该清单是团队在 2025-11 的最新交付视图。

## 0. 项目北极星
- [ ] 提供一套可复现、可扩展的 LLM 敏感性分析流水线，覆盖数据量/参数/模型三大维度
- [ ] 输出结构化的训练产物（LoRA、评估 JSONL、可视化）并沉淀到 `megafakeTasks/…` 与 `sensitivity_analysis/results`
- [ ] 配套文档、监控、冒烟测试，确保长时实验的可见性与稳定性
- [ ] **当前优先级**：先完成一个 “100 正 / 100 负” 的迷你实验链路，复用 `sample_test100_multi_reasoning_task1.py` 的采样策略，验证端到端流程（采样→训练→推理→分析→绘图）无误，再扩展至 1k/2k/5k/10k/20k

### 当前阻塞快照
1. `run_sensitivity_analysis.py` 依赖的 `ExperimentManager/DataSensitivityAnalyzer/*` 缺失，触发 ImportError，造成“入口脚本存在但不可用”的误导。
2. 产物分散在 `logs/`、`megafakeTasks/` 与 `sensitivity_analysis/{logs,outputs,results}`，`core.py` 默认路径与真实产物不一致，用户需要手动搜寻。
3. CLI 在失败前不会提示“最终产物目录/下一条命令”，新手无法快速定位 LoRA、推理结果或可视化输出。
4. Mini Test100 只对 Llama-3.1/Qwen7B/ChatGLM3 通过；Baichuan2 受 safetensors + xformers 约束，Mistral LoRA 推理缺 `partial_rotary_factor`，Qwen1.5-72B 直接 OOM 尚无兜底策略。
5. 冒烟测试与资源配额检查缺失，导致有人直接开大规模实验，数小时后才发现路径或依赖配置错误。

## 1. 目录与执行链路
- [x] 迁移 PRD 与所有文档到 `sensitivity_analysis/docs/`
- [x] **迷你数据集落盘**：`data/data_table/task1/alpaca_test100_balanced/alpaca_megafake_glm_test200_balanced.json`、`megafakeTasks/task1/task1_test200_balanced_glm` 已生成，记录 100 正 / 100 负的平衡样本
- [x] **迷你通路验证**：基于新数据集跑通 `multi_model_lora_train.py` → `multi_model_lora_inference.py` → `analyze_predictions*.py` → `sensitivity_analysis/results/mini_test100_*.csv`，并输出实验摘要（模型/命令/耗时/日志路径）；当前仅 Meta-Llama-3.1-8B、Qwen1.5-7B、chatglm3-6b LoRA+推理通过，Mistral/Baichuan 待修复
- [x] **大规模流水线**：在迷你链路稳定后恢复 1k/2k/5k/10k/20k 采样，已封装为 `run_full_experiment.sh`。
  - **一键运行（含关机）**：`bash run_full_experiment.sh; /usr/bin/shutdown`
  - **日志位置**：`sensitivity_analysis/logs/full_experiment_*.log`
- [x] **旧 ExperimentManager 流水线取舍**：基于缺失模块决定执行 B 方案——已将 `run_sensitivity_analysis.py` 迁移到 `sensitivity_analysis/scripts/archive/`，并把 `experiments/{sensitivity_analysis,real_sensitivity_analysis}` 下历史产物整体归档到 `sensitivity_analysis/experiments/archive/`，对外标注“历史样例，仅供参考”
- [x] **统一落盘策略**：以 `sensitivity_analysis/{logs,outputs,results}` 为唯一真实产物根目录；`core.py`、`scripts/multi_model_*`、CLI 文档需同步字段命名。当前已将 `megafakeTasks/sensitivity_analysis` 迁移至 `sensitivity_analysis/outputs/legacy_sensitivity_analysis` 并建立符号链接，下一步需要针对 `megafakeTasks/task*/...` 执行相同策略，并在 CLI 结束时打印绝对路径与推荐下一步命令
- [x] 在 `scripts/multi_model_lora_train.py` / `multi_model_inference.py` 等 CLI 中补充 `--dry-run` 冒烟模式，运行前检查模型目录、数据集 JSON、输出目录写权限；示例命令 `python scripts/multi_model_lora_train.py --dry-run --models Qwen1.5-7B --datasets task1_test200_balanced_glm`

## 2. 模型与数据覆盖
- [x] 识别可用模型目录 `/root/autodl-tmp/models/{Meta-Llama-3.1-8B-Instruct, Qwen1.5-7B, chatglm3-6b, Mistral-7B-v0.1, Baichuan2-7B-Chat}`
- [x] 记录异常模型：`Qwen1.5-72B` 确认弃用（LoRA 训练 OOM），不再列入实验范围；`Baichuan2-7B-Chat` 需 `.bin` → `safetensors` 转换并升级 `xformers/vllm`（已修复）
- [ ] 校准 Hugging Face 名称与本地路径映射（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct` ↔ `/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct`），在 README/脚本中提供完整表格
- [x] 使用 `convert_task{1,2,3}.py` 产出的 Alpaca 数据 (`data/data_table/task*/alpaca_full/…`) 作为统一数据源，并确认 Task1/Task2/Task3 各自的 `alpaca_test100_*`、`small_*`、`full` 目录已存在
- [x] **迷你数据集**：复用 `sample_test100_multi_reasoning_task1.py` 生成 `task1_test200_balanced_glm`，并在 `sensitivity_analysis/results/mini_test100_*.csv` 中记录模型指标
- [ ] **大规模采样管线**：在迷你链路验证通过后实现 1k/2k/5k/10k/20k 子集（含日志与落盘位置），支持所有仍在白名单的模型
- [ ] 为 Task2/Task3 建立至少一个基准规模（例如 5k、10k）以便横向对照；ChatGLM、Baichuan、Llama-3.1 需在这些任务上完成样本验证
- [ ] 模型模板/`trust_remote_code` 配置在 `scripts/multi_model_*` 与 `sensitivity_analysis/configs/config.py` 保持一致，避免训练/推理解耦

## 3. 实验与评估能力
- [x] **迷你链路（Test100）**：在 200 条样本上跑通 `multi_model_lora_train.py` → `multi_model_lora_inference.py` → `analyze_predictions*.py` → 绘图，验证 CLI、日志、产物路径、评估、绘图全部正常（当前阻塞：Mistral vLLM 配置缺失、Baichuan xformers 兼容、baseline 推理待补）
- [x] 为迷你数据集定制训练/推理参数（如 batch size、max_samples），并在输出中标明“Mini Test100 流程”以及实际输出目录
- [x] `analyze_predictions_lora.py` 升级：支持 Mini Test100 模式下的标签映射、AUC、混淆矩阵、结果导出，并自动写入 `sensitivity_analysis/results/mini_test100_*.csv`
- [x] **全量实验自动化**：已在迷你链路确认无误后，切换到 1k~20k 全量实验。
  - **执行命令**：`bash run_full_experiment.sh; /usr/bin/shutdown`
  - **产物**：日志结构与结果导出保持一致，便于比较
- [ ] 训练结束即验证 `adapter_model.safetensors`、`adapter_config.json` 是否存在；若缺失则记录失败并停止后续推理
- [ ] 在 `megafakeTasks/task*/<model>/<size>/` 或集中目录生成 `experiment_summary.json`，内容含：模型、数据规模、训练耗时、推理耗时、F1/Accuracy、日志/产物路径，并同步复制到 `sensitivity_analysis/results/`

## 4. 可视化与报告
- [ ] 更新 `SENSITIVITY_ANALYSIS_README.md` 与 `sensitivity_analysis/docs/index.md`，明确正式流程：**Mini Test100 链路 → 正式 1k~20k 链路 → analyze → 绘图 → 论文图表**
- [ ] `analyze_predictions*.py` 增加绘图开关，优先实现 Mini Test100 的折线/柱状/散点图输出到 `sensitivity_analysis/results/plots`；确认正确后再扩展至 1k~20k
- [ ] 自动生成论文草图所需的 Markdown/LaTeX 片段：包括图表引用、主要发现（拐点、最优模型）、以及可直接用于“实验设置/结果分析”章节的文字摘要

## 5. 工程与质量保障
- [x] 统一使用 `Makefile` 的 `quality`/`style`/`test` 目标及 `ruff`、`pytest` 规范
- [ ] 为 `scripts/multi_model_*`、`analyze_predictions*.py` 添加强校验（参数解析、目录存在性、配置一致性）的单元测试
- [ ] 在所有 CLI（尤其 multi_model_* 与 vllm_infer）中打印日志/产物路径和下一步命令提示，并提供 `scripts/sa.py quick --dry-run` 的冒烟检测
- [ ] 建立 `sensitivity_analysis/logs/{train,infer,analyze}` 轮转策略，避免长时间运行导致磁盘爆满；必要时上传关键信息到 `experiment_summary.json`

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

## 8. 专项整改 TODO
- [x] **ExperimentManager 决策**
  - [x] 评估后确认缺失模块较多（ExperimentManager/DataSensitivityAnalyzer/ParameterSensitivityAnalyzer/VisualizationEngine等），短期无法补齐
  - [x] 执行 B 方案：`run_sensitivity_analysis.py` 及相关说明迁移到 `sensitivity_analysis/scripts/archive/`，`experiments/{sensitivity_analysis,real_sensitivity_analysis}` 归档至 `sensitivity_analysis/experiments/archive/` 并在 README/CLI 中标注“历史样例，仅供参考”
- [x] **输出目录对齐**
  - [x] 统一 `core.py`、`scripts/multi_model_*`、`sensitivity_analysis/docs/README.md` 的 `output_dir` 描述，指向 `sensitivity_analysis/{logs,outputs,results}`
  - [x] 将现有 `megafakeTasks/sensitivity_analysis` 内容 `rsync`/软链到新目录，并在迁移完成后移除旧路径写入
- [x] **CLI 路径提示**
  - [x] `multi_model_lora_train.py`、`multi_model_lora_inference.py`、`multi_model_inference.py`、`analyze_predictions*.py` 结束时打印“✅ 最终产物：<abs_path>；下一步命令：...”
  - [x] 在 `--dry-run` 模式下提前展示将要写入的 `logs/outputs/results` 路径
- [x] **冒烟测试**
  - [x] 实现 `scripts/sa.py quick --dry-run`，检查模型目录、数据集 JSON、输出目录写权限、剩余显存/磁盘
  - [x] 规定 Mini Test100 及以上规模都必须在冒烟通过后才能提交 GPU 作业
- [x] **迷你→正式衔接**
  - [x] 完成 Mini Test100 全链路验证并在 PRD/README 打钩
  - [ ] 建立规模升级模板：每次扩容需记录「数据集/模型/命令/耗时/日志」四要素，沉淀到 `sensitivity_analysis/results/experiment_summary.jsonl`
- [x] **Baichuan2-7B-Chat 修复**
  - [x] 将 `/root/autodl-tmp/models/Baichuan2-7B-Chat` `.bin` 转换为 safetensors；必要时记录转换命令
  - [x] 升级 torch 2.6 兼容的 `xformers`/`vllm`，重新跑 LoRA 训练+推理，补齐 `sensitivity_analysis/outputs/task1/.../Baichuan2-7B-Chat` 与对应日志
- [x] **Mistral-7B 推理**
  - [x] 在模型 config 或推理脚本里补齐 `partial_rotary_factor`（或切换 transformers 推理路径）
  - [x] 复验 LoRA 推理写入 `sensitivity_analysis/outputs` 并补充分析结果
- [x] **Qwen1.5-72B 策略**
  - [x] **已弃用**：因单卡显存限制（OOM），明确本项目不再支持 Qwen1.5-72B。代码中虽保留 `--include-large-models` 兼容逻辑，但实验计划中将其完全剔除。
  - [x] Dry-run 检测到 72B 时会提示风险。
- [x] **历史产物迁移**
  - [x] 将 `megafakeTasks/task1/task1_test200_balanced_glm/*` 以及 `test200_balanced` 下的 JSONL/LoRA 软链或同步到 `sensitivity_analysis/outputs`
  - [x] 为旧目录补充 README，指向新路径并说明“仅保留历史，新增产物全部在 sensitivity_analysis/…”

> 更新人：团队共识；本 TODO 文档需随每次结构调整同步修订，确保审核时可直接对照状态打钩。
