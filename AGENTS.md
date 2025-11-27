# Repository Guidelines

## Project Structure & Module Organization
Core training and inference live in `src/` with reusable CLIs under `scripts/`. All sensitivity-analysis assets (docs, configs, smoke tests, plots) are now anchored in `sensitivity_analysis/`, which also hosts canonical `logs/`, `outputs/`, and `results/`. Old entry points are only in the archive. Legacy long-form adapters remain under `megafakeTasks/<task>/<dataset>/<model>/lora/sft`; keep them readable but route new jobs to `sensitivity_analysis/outputs` and symlink as needed. Dataset conversion helpers (`convert_task1.py`, `convert_task2.py`, `convert_task3.py`) and sampling scripts (e.g., `sample_test100_multi_reasoning_task1.py`) sit at the repo root for easy reuse. Tests mirror the source hierarchy inside `tests/`.

## Build, Test, and Development Commands
Install deps via `pip install -e ".[torch,metrics]" --no-build-isolation`; add `,deepspeed,vllm,quantization` when GPU kernels are required. Run `make quality` for `ruff format --check` + `ruff check`, and `make test` (PyTest) before submitting patches. Sensitivity flows typically call `python scripts/multi_model_lora_train.py --models Meta-Llama-3.1-8B-Instruct --datasets task1_test200_balanced_glm` followed by `python scripts/multi_model_lora_inference.py ...` and `python scripts/analyze_predictions.py --input sensitivity_analysis/outputs/...`. Always append `--dry-run` during wiring changes to confirm directories and configs.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and snake_case for functions/variables (classes stay PascalCase). Dataset keys, result folders, and log prefixes follow `<task>_<size>_<variant>` (e.g., `task1_test200_balanced_glm`). Keep logging explicit: emit absolute `logs/` + `outputs/` paths and the next recommended command. Format code with `ruff format` and lint with `ruff check --fix` prior to committing to avoid CI churn.

## Testing Guidelines
Favor “mini” datasets for validation: regenerate with `python sample_test100_multi_reasoning_task1.py` and run the upcoming `python scripts/sa.py quick --dry-run` to verify GPU/dataset/output readiness before long experiments. Unit and regression tests belong in `tests/` near the code they cover (e.g., `tests/scripts/test_multi_model_lora_train.py`) and should be named `test_<intent>`. When adding analyzers, include fixtures under `sensitivity_analysis/data/` so metrics pipelines can run deterministically.

## Commit & Pull Request Guidelines
Commits use concise Chinese imperatives (`统一 output_dir 提示`, `修复 Baichuan2 推理配置`). PR descriptions must outline motivation, affected modules/configs, validation commands (train/infer/analyze), and where artifacts landed (`sensitivity_analysis/logs/train/...`). Attach representative logs or CSVs whenever metrics change, and cross-reference relevant PRD TODO items so reviewers can trace progress quickly.

## Security & Configuration Tips
Honor the shared HF cache by leaving `HF_HOME=.cache/huggingface` in place and prefer `HF_ENDPOINT=https://hf-mirror.com`. Keep `trust_remote_code` explicit per model, especially when adding new entries to `MODEL_CONFIGS`. Before running Baichuan/Mistral/Qwen variants, confirm weights exist under `/root/autodl-tmp/models/` and that adapters are written to `sensitivity_analysis/outputs` rather than ad-hoc folders.
