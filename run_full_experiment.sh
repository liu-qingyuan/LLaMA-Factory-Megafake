#!/bin/bash

# -----------------------------------------------------------------------------
# 全量规模实验自动运行脚本
# 功能: 依次执行 1k -> 2k -> 5k -> 10k -> 20k 的 训练 -> 推理 -> 分析 全流程
# -----------------------------------------------------------------------------

# 定义环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=.cache/huggingface

# 日志文件
LOG_FILE="sensitivity_analysis/logs/full_experiment_$(date +%Y%m%d_%H%M%S).log"
mkdir -p sensitivity_analysis/logs

echo "========================================================" | tee -a "$LOG_FILE"
echo "🚀 开始大规模全量实验 (Start Time: $(date))" | tee -a "$LOG_FILE"
echo "📜 详细日志将写入: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# 定义模型列表
MODELS="Meta-Llama-3.1-8B-Instruct Qwen1.5-7B chatglm3-6b Mistral-7B-v0.1 Baichuan2-7B-Chat"

# 定义数据集规模列表
SCALES=("task1_scale_1000_glm" "task1_scale_2000_glm" "task1_scale_5000_glm" "task1_scale_10000_glm" "task1_scale_20000_glm")

# 循环处理每个规模
for dataset in "${SCALES[@]}"; do
    start_time=$(date +%s)
    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo "📦 正在处理数据集规模: $dataset" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"

    # 1. 训练 (Training)
    echo "👉 [1/3] 开始训练 (Training)..." | tee -a "$LOG_FILE"
    python scripts/multi_model_lora_train.py --models $MODELS --datasets $dataset >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ 训练命令执行完毕" | tee -a "$LOG_FILE"
    else
        echo "❌ 训练命令执行出错 (但将尝试继续推理)" | tee -a "$LOG_FILE"
    fi

    # 2. 推理 (Inference)
    echo "👉 [2/3] 开始推理 (Inference)..." | tee -a "$LOG_FILE"
    python scripts/multi_model_lora_inference.py --models $MODELS --datasets $dataset >> "$LOG_FILE" 2>&1
    
    # 3. 分析 (Analysis)
    echo "👉 [3/3] 开始分析与绘图 (Analysis)..." | tee -a "$LOG_FILE"
    
    # 提取规模数值 (例如 1000)
    scale_val=$(echo $dataset | sed -n 's/.*scale_\([0-9]*\)_glm/\1/p')
    
    result_dir="sensitivity_analysis/outputs/task1/scale_${scale_val}"
    plot_dir="sensitivity_analysis/results/plots/scale_${scale_val}"
    csv_file="sensitivity_analysis/results/scale_${scale_val}_metrics.csv"
    
    if [ -d "$result_dir" ]; then
        python scripts/analyze_predictions.py \
            --dir "$result_dir" \
            --output "$csv_file" \
            --plot --plot-dir "$plot_dir" >> "$LOG_FILE" 2>&1
        
        if [ -f "$csv_file" ]; then
            echo "📊 分析报告已生成: $csv_file" | tee -a "$LOG_FILE"
        else
            echo "⚠️  分析脚本运行完成但未生成 CSV" | tee -a "$LOG_FILE"
        fi
    else
        echo "⚠️  未找到结果目录，跳过分析: $result_dir" | tee -a "$LOG_FILE"
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "⏱️  规模 $dataset 处理耗时: ${duration} 秒" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "🎉 所有规模实验已完成! (End Time: $(date))" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
