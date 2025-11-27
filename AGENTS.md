# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the core LLaMA-Factory codebase (training, inference, evaluation utilities).
- `sensitivity_analysis/` contains configs, scripts, docs, and results specific to the MegaFake sensitivity workflows; logs land in `logs/sensitivity_analysis/`, while adapters and eval JSONL files are under `megafakeTasks/sensitivity_analysis/`.
- `scripts/` exposes entrypoints such as `scripts/sa.py` (recommended CLI wrapper) alongside helper converters.
- `tests/` provides unit and integration coverage; docs and product specs live in `docs/`.
- Experiments archived in `experiments/` are legacy; reference them only for historical metrics.

## Build, Test, and Development Commands
- `pip install -e ".[torch,metrics]" --no-build-isolation` installs editable deps; add `,deepspeed,vllm,quantization` for full GPU feature parity.
- `python scripts/sa.py quick` runs a lightweight end-to-end sensitivity sweep; replace `quick` with `full` or `test` for other modes.
- `make build` publishes wheel/sdist via `python -m build`.
- `make quality` performs `ruff check` and `ruff format --check` across `scripts`, `src`, and `tests`.
- `make test` executes `pytest -vv tests/` with `CUDA_VISIBLE_DEVICES` unset (defaults to CPU) and `WANDB_DISABLED=true`.

## Coding Style & Naming Conventions
- Follow standard PEP8 with 4-space indents; type hints are expected for new Python modules.
- Use descriptive module and directory names (e.g., `data_sensitivity_analyzer.py`), and prefer snake_case for files/functions, PascalCase for classes.
- Before committing, run `make style` (auto-fix via `ruff check --fix` plus `ruff format`); no trailing whitespace or mixed encodings.

## Testing Guidelines
- Tests live in `tests/` and should mirror source structure (`tests/sensitivity_analysis/test_core.py`, etc.).
- Name test functions with `test_...` and keep fixtures reusable.
- Target meaningful coverage for new logic; for GPU-heavy flows, provide minimal-sample smoke tests callable via `scripts/sa.py test`.

## Commit & Pull Request Guidelines
- Recent history favors concise, descriptive Chinese commit titles (e.g., `更新 multi_model_inference.py 文件...`); keep tense imperative and explain the primary change.
- Include context about impacted modules/paths; multi-file work should mention both code and data updates.
- PRs should describe motivation, summarize testing (`make test`, `python scripts/sa.py quick`), and link issues or experiment artefacts; attach logs or screenshots when modifying training flows.

## Security & Configuration Tips
- Store large checkpoints under `/root/autodl-tmp/models` and avoid committing binaries; reference them via config rather than copying into the repo.
- Set `HF_ENDPOINT` before running scripts in restricted environments, and keep API keys in `.env` entries ignored by Git.
