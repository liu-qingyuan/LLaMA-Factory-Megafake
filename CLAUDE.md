# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is LLaMA-Factory-Megafake, a specialized version of LLaMA-Factory customized for multi-model fake news detection tasks. The repository builds upon the LLaMA-Factory framework to provide fine-tuning capabilities for various language models on fake news detection datasets.

## Core Architecture

### Main Components

- **src/llamafactory/**: Core LLaMA-Factory package containing training, inference, and evaluation modules
  - **train/**: Training logic for different approaches (full-tuning, LoRA, QLoRA)
  - **model/**: Model loading and configuration management
  - **data/**: Data processing and template handling
  - **api/**: OpenAI-style API server implementation
  - **webui/**: Gradio-based web interface for model training and chat

- **scripts/**: Custom automation scripts for multi-model workflows
  - `multi_model_inference.py`: Batch inference across multiple models and datasets
  - `multi_model_lora_train.py`: Batch LoRA training automation
  - `multi_model_lora_inference.py`: Batch LoRA inference automation
  - `vllm_infer.py`: VLLM-based inference engine
  - `analyze_predictions*.py`: Various analysis scripts for model predictions

- **examples/**: Configuration templates for different training scenarios
  - `train_lora/`: LoRA fine-tuning configurations
  - `train_qlora/`: QLoRA (quantized LoRA) configurations
  - `inference/`: Model inference configurations

- **data/**: Dataset definitions and processing logic
  - `dataset_info.json`: Dataset registry and metadata
  - Various task-specific data processing modules

### Task Structure

The repository organizes work around specific fake news detection tasks:
- **Task 1**: General fake news detection
- **Task 2**: Content analysis
- **Task 3**: Multi-source information verification

Each task has corresponding datasets, evaluation scripts, and result storage directories.

## Common Development Commands

### Installation and Setup

```bash
# Install from source
pip install -e ".[torch,metrics]" --no-build-isolation

# Install with specific extras for different hardware/backends
pip install -e ".[torch,metrics,deepspeed,vllm,quantization]" --no-build-isolation
```

### Training Commands

```bash
# Standard LoRA fine-tuning
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# QLoRA (quantized) training
llamafactory-cli train examples/train_qlora/llama3_lora_sft.yaml

# Multi-model batch training
python scripts/multi_model_lora_train.py
```

### Inference Commands

```bash
# Single model inference
llamafactory-cli chat examples/inference/llama3.yaml

# VLLM-accelerated batch inference
python scripts/vllm_infer.py --model_name_or_path <path> --dataset <dataset>

# Multi-model batch inference
python scripts/multi_model_inference.py

# LoRA-specific batch inference
python scripts/multi_model_lora_inference.py
```

### Model Management

```bash
# Merge LoRA adapters
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

# Launch API server
llamafactory-cli api examples/inference/llama3.yaml

# Launch Web UI
llamafactory-cli webui
```

### Evaluation and Analysis

```bash
# Model evaluation
llamafactory-cli eval examples/eval_config.yaml

# Analyze predictions
python scripts/analyze_predictions.py

# Task-specific analysis scripts
python scripts/analyze_predictions_CoT_SC.py
python scripts/analyze_predictions_FS_5.py
python scripts/analyze_predictions_ZS_DF.py
```

## Development Workflow

### Adding New Models

1. Update `src/llamafactory/extras/constants.py` with model configuration
2. Add template in `src/llamafactory/data/template.py` if needed
3. Create example configuration in `examples/` directory
4. Update model configs in scripts' `MODEL_CONFIGS` dictionary

### Adding New Datasets

1. Prepare data in supported format (JSON, JSONL)
2. Add dataset entry to `data/dataset_info.json`
3. Create data processing script if needed
4. Update dataset configs in scripts' `DATASET_CONFIGS` dictionary

### Batch Processing Workflow

1. **Training**: Use `multi_model_lora_train.py` to train multiple model-dataset combinations
2. **Inference**: Use `multi_model_lora_inference.py` for batch LoRA inference
3. **Analysis**: Use appropriate `analyze_predictions_*.py` script for result analysis

### Configuration Management

- YAML configurations in `examples/` serve as templates
- Scripts generate dynamic configs based on templates
- Key parameters: model path, template, dataset, LoRA rank, learning rate
- Output follows standardized directory structure under `megafakeTasks/`

## File Organization Patterns

### Result Storage Structure

```
megafakeTasks/
├── task1/
│   ├── {dataset_name}/
│   │   ├── {model_name}/
│   │   │   ├── lora/sft/           # LoRA adapters
│   │   │   └── result_*.jsonl      # Inference results
└── logs/                           # Training/inference logs
```

### Script Configuration

Scripts use configuration dictionaries for:
- `MODEL_CONFIGS`: Maps model paths to (template, trust_remote_code) tuples
- `DATASET_CONFIGS`: Maps dataset keys to dataset names/paths
- Training parameters are centralized in `create_config_file()` functions

## Environment Variables and Acceleration

```bash
# HuggingFace mirror for faster downloads
export HF_ENDPOINT=https://hf-mirror.com

# Network acceleration (if available)
source /etc/network_turbo

# GPU selection
export CUDA_VISIBLE_DEVICES=0,1

# For Ascend NPU
export ASCEND_RT_VISIBLE_DEVICES=0
```

## Testing and Validation

```bash
# Test LoRA inference setup
python test_lora_inference.py

# Run quality checks
make quality

# Run tests
make test
```

## Key Points for Development

1. **Path Management**: Scripts use relative paths and generate absolute paths dynamically
2. **Error Handling**: Comprehensive logging and graceful handling of missing models/adapters
3. **Idempotency**: Scripts check for existing results and skip completed tasks
4. **Modularity**: Each script is self-contained but follows consistent patterns
5. **Resource Awareness**: Scripts include memory and GPU requirement considerations

## Custom Extensions

This repository includes several custom extensions beyond standard LLaMA-Factory:
- Multi-model batch automation scripts
- Task-specific evaluation pipelines
- Custom analysis tools for different reasoning approaches
- Integration with academic network acceleration
- Specialized dataset handling for fake news detection tasks