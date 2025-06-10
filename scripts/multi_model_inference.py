#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path
import datetime

# 模型配置：模型路径 -> 模板名称
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Baichuan2-7B-Chat": "baichuan2",
    "/root/autodl-tmp/models/chatglm3-6b": "chatglm3", 
    "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct": "llama3",
    "/root/autodl-tmp/models/Mistral-7B-v0.1": "mistral",
    "/root/autodl-tmp/models/Qwen1.5-7B": "qwen"
}

# 数据集配置
DATASET_CONFIGS = {
    # "task1_full_glm": "data_table/task1/alpaca_full/alpaca_megafake_glm_binary.json",
    # "task1_full_llama": "data_table/task1/alpaca_full/alpaca_megafake_llama_binary.json", 
    # "task1_small_glm": "data_table/task1/small_8k/alpaca_megafake_glm_8k.json",
    # "task1_small_llama": "data_table/task1/small_8k/alpaca_megafake_llama_8k.json",
    
    # Task2 - GLM 假新闻子类
    "task2_full_glm_style_based_fake": "data_table/task2/alpaca_full/glm/alpaca_glm_style_based_fake.json",
    "task2_full_glm_content_based_fake": "data_table/task2/alpaca_full/glm/alpaca_glm_content_based_fake.json",
    "task2_full_glm_integration_based_fake": "data_table/task2/alpaca_full/glm/alpaca_glm_integration_based_fake.json",
    "task2_full_glm_story_based_fake": "data_table/task2/alpaca_full/glm/alpaca_glm_story_based_fake.json",
    # Task2 - GLM 真新闻子类
    "task2_full_glm_style_based_legitimate": "data_table/task2/alpaca_full/glm/alpaca_glm_style_based_legitimate.json",
    "task2_full_glm_integration_based_legitimate": "data_table/task2/alpaca_full/glm/alpaca_glm_integration_based_legitimate.json",
    
    # Task2 - LLaMA 假新闻子类
    "task2_full_llama_style_based_fake": "data_table/task2/alpaca_full/llama/alpaca_llama3_style_based_fake.json",
    "task2_full_llama_content_based_fake": "data_table/task2/alpaca_full/llama/alpaca_llama3_content_based_fake.json",
    "task2_full_llama_integration_based_fake": "data_table/task2/alpaca_full/llama/alpaca_llama3_integration_based_fake.json",
    "task2_full_llama_story_based_fake": "data_table/task2/alpaca_full/llama/alpaca_llama3_story_based_fake.json",
    # Task2 - LLaMA 真新闻子类
    "task2_full_llama_style_based_legitimate": "data_table/task2/alpaca_full/llama/alpaca_llama3_style_based_legitimate.json",
    "task2_full_llama_integration_based_legitimate": "data_table/task2/alpaca_full/llama/alpaca_llama3_integration_based_legitimate.json",
    
    # Task2 - 8K 采样数据集（注释掉以减少任务量，需要时可取消注释）
    # "task2_small_glm_style_based_fake": "data_table/task2/small_8k/glm/alpaca_glm_style_based_fake_8k.json",
    # "task2_small_glm_content_based_fake": "data_table/task2/small_8k/glm/alpaca_glm_content_based_fake_8k.json",
    # "task2_small_glm_integration_based_fake": "data_table/task2/small_8k/glm/alpaca_glm_integration_based_fake_8k.json",
    # "task2_small_glm_story_based_fake": "data_table/task2/small_8k/glm/alpaca_glm_story_based_fake_8k.json",
    # "task2_small_glm_style_based_legitimate": "data_table/task2/small_8k/glm/alpaca_glm_style_based_legitimate_8k.json",
    # "task2_small_glm_integration_based_legitimate": "data_table/task2/small_8k/glm/alpaca_glm_integration_based_legitimate_8k.json",
    # "task2_small_llama_style_based_fake": "data_table/task2/small_8k/llama/alpaca_llama3_style_based_fake_8k.json",
    # "task2_small_llama_content_based_fake": "data_table/task2/small_8k/llama/alpaca_llama3_content_based_fake_8k.json",
    # "task2_small_llama_integration_based_fake": "data_table/task2/small_8k/llama/alpaca_llama3_integration_based_fake_8k.json",
    # "task2_small_llama_story_based_fake": "data_table/task2/small_8k/llama/alpaca_llama3_story_based_fake_8k.json",
    # "task2_small_llama_style_based_legitimate": "data_table/task2/small_8k/llama/alpaca_llama3_style_based_legitimate_8k.json",
    # "task2_small_llama_integration_based_legitimate": "data_table/task2/small_8k/llama/alpaca_llama3_integration_based_legitimate_8k.json",
    
    # "task3_full_gossip": "data_table/task3/alpaca_full/alpaca_chatglm_gossip_binary.json",
    # "task3_full_polifact": "data_table/task3/alpaca_full/alpaca_chatglm_polifact_binary.json",
    # "task3_small_gossip": "data_table/task3/small_8k/alpaca_chatglm_gossip_8k.json",
    # "task3_small_polifact": "data_table/task3/small_8k/alpaca_chatglm_polifact_8k.json"
}

def get_model_name(model_path):
    """从模型路径提取模型名称"""
    return Path(model_path).name

def get_save_path(model_path, dataset_name):
    """根据模型和数据集生成保存路径"""
    model_name = get_model_name(model_path)
    
    # 根据数据集名称确定任务和类型
    if "task1" in dataset_name:
        task = "task1"
        if "full" in dataset_name:
            size = "full"
        else:
            size = "small"
        
        if "glm" in dataset_name:
            data_type = "megafake_glm_binary"
        else:
            data_type = "megafake_llama_binary"
            
    elif "task2" in dataset_name:
        task = "task2"
        if "full" in dataset_name:
            size = "full"
        else:
            size = "small"
        
        # 解析 Task2 的子类信息
        parts = dataset_name.split("_")
        # 格式: task2_full_glm_style_based_fake 或 task2_full_llama_content_based_legitimate
        model_source = parts[2]  # glm 或 llama
        
        # 找到最后一个部分作为 news_type (fake 或 legitimate)
        news_type = parts[-1]
        
        # 提取子类名称 (style, content, integration, story)
        # 从 parts[3] 开始到倒数第二个部分，去掉 "based"
        subclass_parts = parts[3:-1]
        if "based" in subclass_parts:
            subclass_parts.remove("based")
        subclass = "_".join(subclass_parts)
        
        if model_source == "glm":
            if news_type == "fake":
                data_type = f"glm_{subclass}_based_fake"
            else:
                data_type = f"glm_{subclass}_based_legitimate"
        else:  # llama
            if news_type == "fake":
                data_type = f"llama3_{subclass}_based_fake"
            else:
                data_type = f"llama3_{subclass}_based_legitimate"
            
    elif "task3" in dataset_name:
        task = "task3"
        if "full" in dataset_name:
            size = "full"
        else:
            size = "small"
            
        if "gossip" in dataset_name:
            data_type = "chatglm_gossip_binary"
        else:
            data_type = "chatglm_polifact_binary"
    
    # 构建保存路径
    save_path = f"megafakeTasks/{task}/{size}/result_{data_type}_{model_name}.jsonl"
    return save_path

def get_log_path(model_path, dataset_name):
    """生成日志文件路径"""
    model_name = get_model_name(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inference_{model_name}_{dataset_name}_{timestamp}.log"
    return f"logs/{log_filename}"

def run_inference(model_path, template, dataset_name, save_path, max_new_tokens=10):
    """运行单个推理任务"""
    cmd = [
        "python", "scripts/vllm_infer.py",
        "--model_name_or_path", model_path,
        "--template", template,
        "--dataset", dataset_name,
        "--save_name", save_path,
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", "0.1",  # 降低温度以获得更稳定的输出
        "--top_p", "0.9",
        "--batch_size", "1024"  
    ]
    
    # 为某些模型添加 trust_remote_code 参数
    model_name = get_model_name(model_path)
    if "Baichuan" in model_name or "chatglm" in model_name.lower():
        cmd.append("--trust_remote_code")
    
    # 创建日志目录
    log_path = get_log_path(model_path, dataset_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"日志文件: {log_path}")
    
    try:
        # 打开日志文件，同时输出到控制台和文件
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 记录命令和时间戳
            log_file.write(f"开始时间: {datetime.datetime.now()}\n")
            log_file.write(f"执行命令: {' '.join(cmd)}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()
            
            # 使用 Popen 来实时输出日志
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时读取输出并同时写入文件和控制台
            for line in process.stdout:
                print(line, end='')  # 输出到控制台
                log_file.write(line)  # 写入文件
                log_file.flush()
            
            # 等待进程结束
            return_code = process.wait()
            
            # 记录结束时间
            log_file.write("=" * 80 + "\n")
            log_file.write(f"结束时间: {datetime.datetime.now()}\n")
            log_file.write(f"返回码: {return_code}\n")
            
            if return_code == 0:
                print(f"✅ 成功完成: {get_model_name(model_path)} + {dataset_name}")
                print(f"   保存至: {save_path}")
                print(f"   日志至: {log_path}")
                return True
            else:
                print(f"❌ 失败: {get_model_name(model_path)} + {dataset_name}")
                print(f"   返回码: {return_code}")
                print(f"   日志文件: {log_path}")
                return False
                
    except Exception as e:
        print(f"❌ 执行异常: {get_model_name(model_path)} + {dataset_name}")
        print(f"   异常信息: {e}")
        return False

def check_model_exists(model_path):
    """检查模型是否存在"""
    if not os.path.exists(model_path):
        return False
    
    # 检查是否有必要的配置文件
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        print(f"⚠️  模型配置文件不存在: {config_file}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 开始多模型推理任务")
    print(f"📊 总计 {len(MODEL_CONFIGS)} 个模型，{len(DATASET_CONFIGS)} 个数据集")
    print(f"🎯 总任务数: {len(MODEL_CONFIGS) * len(DATASET_CONFIGS)}")

    # 统计信息
    total_tasks = len(MODEL_CONFIGS) * len(DATASET_CONFIGS)
    completed_tasks = 0
    failed_tasks = 0
    
    # 遍历所有模型和数据集组合
    for model_path, template in MODEL_CONFIGS.items():
        model_name = get_model_name(model_path)
        print(f"\n🔄 处理模型: {model_name} (模板: {template})")
        
        # 检查模型路径是否存在
        if not check_model_exists(model_path):
            print(f"⚠️  模型路径不存在或配置不完整，跳过: {model_path}")
            failed_tasks += len(DATASET_CONFIGS)
            continue
        
        for dataset_name in DATASET_CONFIGS.keys():
            save_path = get_save_path(model_path, dataset_name)
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 检查是否已存在结果文件
            if os.path.exists(save_path):
                print(f"⏭️  结果文件已存在，跳过: {save_path}")
                completed_tasks += 1
                continue
            
            # 运行推理
            print(f"🎯 开始推理: {model_name} + {dataset_name}")
            success = run_inference(model_path, template, dataset_name, save_path)
            
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1
            
            print(f"📈 进度: {completed_tasks + failed_tasks}/{total_tasks} "
                  f"(成功: {completed_tasks}, 失败: {failed_tasks})")
    
    print(f"\n🎉 推理任务完成!")
    print(f"📊 总结: {total_tasks} 个任务")
    print(f"✅ 成功: {completed_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志文件")

if __name__ == "__main__":
    main() 