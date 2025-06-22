#!/usr/bin/env python3

import json
import os
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
import argparse

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 加载文件失败: {file_path}")
        print(f"   错误信息: {e}")
        return None
    return data

def get_task_type(file_path):
    """根据文件路径判断任务类型"""
    if "task1" in file_path or "task3" in file_path:
        return "binary_classification"  # task1和task3都是二分类
    elif "task2" in file_path:
        return "binary_classification"  # task2实际上也是二分类
    else:
        return "unknown"

def get_task2_subclass(file_path):
    """获取task2的子类信息"""
    # 只从文件名中提取子类信息，避免目录名干扰
    import os
    filename = os.path.basename(file_path)
    
    if "style_based" in filename:
        if "legitimate" in filename:
            return "style_based_legitimate"
        else:
            return "style_based_fake"
    elif "content_based" in filename:
        return "content_based_fake"
    elif "integration_based" in filename:
        if "legitimate" in filename:
            return "integration_based_legitimate"
        else:
            return "integration_based_fake"
    elif "story_based" in filename:
        return "story_based_fake"
    return "unknown"

def normalize_prediction(pred_text, task_type, task2_subclass=None):
    """
    标准化预测文本为统一格式
    """
    if not pred_text:
        return "unknown"
    
    # 更彻底地清理预测文本
    pred_clean = str(pred_text).strip()
    # 移除所有换行符和多余空格
    pred_clean = re.sub(r'[\n\r\t]+', ' ', pred_clean)
    pred_clean = re.sub(r'\s+', ' ', pred_clean)
    pred_clean = pred_clean.strip().lower()
    
    if task_type == "binary_classification":
        # 对于Task2，需要特殊处理
        if task2_subclass:
            # 优先检查不确定匹配
            uncertain_phrases = [
                'cannot determine', 'unable to determine', 'not sure', 'uncertain', 'unclear',
                'hard to tell', 'difficult to determine', 'cannot tell', 'ambiguous',
                '无法确定', '不能确定', '无法判断', '不能判断', '不确定', '不清楚', '难以判断'
            ]
            if any(phrase in pred_clean for phrase in uncertain_phrases):
                return "uncertain"
            
            if "fake" in task2_subclass:
                # 对于fake子类：检查目标类别 vs other
                if task2_subclass == "style_based_fake":
                    style_keywords = ['style-based', 'style based', 'stylistic', 'style', '风格']
                    if any(keyword in pred_clean for keyword in style_keywords):
                        return "legitimate"  # 正类
                elif task2_subclass == "content_based_fake":
                    content_keywords = ['content-based', 'content based', 'content', '内容']
                    if any(keyword in pred_clean for keyword in content_keywords):
                        return "legitimate"  # 正类
                elif task2_subclass == "integration_based_fake":
                    integration_keywords = ['integration-based', 'integration based', 'integration', '整合']
                    if any(keyword in pred_clean for keyword in integration_keywords):
                        return "legitimate"  # 正类
                elif task2_subclass == "story_based_fake":
                    story_keywords = ['story-based', 'story based', 'story', '故事']
                    if any(keyword in pred_clean for keyword in story_keywords):
                        return "legitimate"  # 正类
                
                # 检查other关键词
                other_keywords = ['other', 'others', 'different', '其他']
                if any(keyword in pred_clean for keyword in other_keywords):
                    return "fake"  # 负类
                    
            elif "legitimate" in task2_subclass:
                # 对于legitimate子类：style-based vs integration-based
                style_keywords = ['style-based', 'style based', 'stylistic', 'style', '风格']
                integration_keywords = ['integration-based', 'integration based', 'integration', '整合']
                
                if any(keyword in pred_clean for keyword in style_keywords):
                    return "legitimate"  # style-based
                elif any(keyword in pred_clean for keyword in integration_keywords):
                    return "fake"  # integration-based
        
        # Task1和Task3的传统二分类逻辑
        else:
            # 优先检查不确定匹配（英文+中文）
            uncertain_phrases = [
                # 英文短语
                'cannot determine', 'unable to determine', 'not sure', 'uncertain', 'unclear',
                'hard to tell', 'difficult to determine', 'cannot tell', 'ambiguous',
                # 中文短语
                '无法确定', '不能确定', '无法判断', '不能判断', '不确定', '不清楚', '难以判断'
            ]
            if any(phrase in pred_clean for phrase in uncertain_phrases):
                return "uncertain"
            
            # 然后检查正向匹配 - 表示legitimate/true的词汇（英文+中文）
            legitimate_keywords = [
                # 英文关键词
                'legitimate', 'true', 'real', 'valid', 'genuine', 'authentic', 'correct', 'accurate', 'factual',
                # 中文关键词
                '真实', '真', '合法', '正确', '准确', '可信', '可靠', '真的', '是真的', '不是假的', '确实'
            ]
            if any(word in pred_clean for word in legitimate_keywords):
                return "legitimate"
            
            # 最后检查负向匹配 - 表示fake/false的词汇（英文+中文）
            fake_keywords = [
                # 英文关键词
                'fake', 'false', 'unreal', 'invalid', 'misleading', 'deceptive', 'incorrect', 'inaccurate',
                # 中文关键词
                '假', '虚假', '错误', '不真实', '不准确', '伪造', '编造', '误导', '假的', '是假的', '不真'
            ]
            if any(word in pred_clean for word in fake_keywords):
                return "fake"
                
    elif task_type == "multiclass_classification":
        # Task2的多分类逻辑 - 优先级：不确定 > Other > 具体类型
        
        # 首先检查不确定的表述
        uncertain_phrases = [
            'cannot determine', 'unable to determine', 'not sure', 'uncertain', 'unclear',
            'hard to tell', 'difficult to determine', 'cannot tell', 'ambiguous',
            '无法确定', '不能确定', '无法判断', '不能判断', '不确定', '不清楚', '难以判断'
        ]
        if any(phrase in pred_clean for phrase in uncertain_phrases):
            return "unknown"
        
        # 然后检查Other关键词（英文+中文）
        other_keywords = [
            'other', 'others', 'different', 'else', 'another', 'other type', 'other types',
            '其他', '别的', '其它', '另一种', '不同'
        ]
        if any(keyword in pred_clean for keyword in other_keywords):
            return "other"
        
        # 接着检查具体的类型关键词
        
        # Style-Based 关键词（英文+中文）
        style_keywords = [
            'style-based', 'style based', 'stylistic', 'style',
            '风格', '样式', '文体', '风格化', '样式化', '文风'
        ]
        if any(keyword in pred_clean for keyword in style_keywords):
            return "style-based"
        
        # Content-Based 关键词（英文+中文）
        content_keywords = [
            'content-based', 'content based', 'content', 'manipulated', 'modified',
            '内容', '内容化', '操控', '修改', '篡改', '更改'
        ]
        if any(keyword in pred_clean for keyword in content_keywords):
            return "content-based"
        
        # Integration-Based 关键词（英文+中文）
        integration_keywords = [
            'integration-based', 'integration based', 'integration', 'integrated', 'combined', 'merged',
            '整合', '集成', '综合', '合并', '融合', '结合'
        ]
        if any(keyword in pred_clean for keyword in integration_keywords):
            return "integration-based"
        
        # Story-Based 关键词（英文+中文）
        story_keywords = [
            'story-based', 'story based', 'story', 'narrative', 'generated', 'created',
            '故事', '叙述', '生成', '创造', '编写', '虚构'
        ]
        if any(keyword in pred_clean for keyword in story_keywords):
            return "story-based"
        
        # 最后，对于fake子类，如果没有明确的类型，但有fake相关词汇，归为other
        if task2_subclass and 'fake' in task2_subclass:
            general_fake_keywords = [
                'fake', 'false', 'misleading',
                '假', '虚假', '误导'
            ]
            if any(word in pred_clean for word in general_fake_keywords):
                return "other"
    
    return "unknown"

def calculate_metrics(tp, fp, fn):
    """计算Precision、Recall和F1-Score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def calculate_binary_classification_metrics(data, task_type, task2_subclass=None):
    """计算二分类任务的详细指标"""
    if not data:
        return {}
    
    # 统计真实标签和预测标签
    true_labels = []
    pred_labels = []
    
    for item in data:
        if 'predict' not in item or 'label' not in item:
            continue
            
        # 标准化预测结果
        pred_normalized = normalize_prediction(item['predict'], task_type, task2_subclass)
        
        # 获取真实标签
        true_label = str(item['label']).strip()
        
        if task_type == "binary_classification":
            # 对于Task1和Task3的传统二分类
            if task2_subclass is None:
                if true_label.lower() in ['fake', '0']:
                    true_labels.append("fake")
                elif true_label.lower() in ['legitimate', '1']:
                    true_labels.append("legitimate")
                else:
                    continue  # 跳过未知标签
            # 对于Task2的二分类
            else:
                # 直接使用标签，并映射到legitimate/fake
                true_label_clean = true_label.lower()
                if "fake" in task2_subclass:
                    # 对于fake子类：目标类别 vs other
                    if task2_subclass == "style_based_fake" and true_label_clean == "style-based":
                        true_labels.append("legitimate")  # 这里legitimate表示正类
                    elif task2_subclass == "content_based_fake" and true_label_clean == "content-based":
                        true_labels.append("legitimate")
                    elif task2_subclass == "integration_based_fake" and true_label_clean == "integration-based":
                        true_labels.append("legitimate")
                    elif task2_subclass == "story_based_fake" and true_label_clean == "story-based":
                        true_labels.append("legitimate")
                    elif true_label_clean == "other":
                        true_labels.append("fake")  # 这里fake表示负类
                    else:
                        continue  # 跳过未知标签
                elif "legitimate" in task2_subclass:
                    # 对于legitimate子类：style-based vs integration-based
                    if true_label_clean == "style-based":
                        true_labels.append("legitimate")
                    elif true_label_clean == "integration-based":
                        true_labels.append("fake")
                    else:
                        continue  # 跳过未知标签
                        
            pred_labels.append(pred_normalized)
    
    # 计算混淆矩阵
    tp_legitimate = sum(1 for t, p in zip(true_labels, pred_labels) if t == "legitimate" and p == "legitimate")
    fp_legitimate = sum(1 for t, p in zip(true_labels, pred_labels) if t == "fake" and p == "legitimate")
    fn_legitimate = sum(1 for t, p in zip(true_labels, pred_labels) if t == "legitimate" and p != "legitimate")
    
    tp_fake = sum(1 for t, p in zip(true_labels, pred_labels) if t == "fake" and p == "fake")
    fp_fake = sum(1 for t, p in zip(true_labels, pred_labels) if t == "legitimate" and p == "fake")
    fn_fake = sum(1 for t, p in zip(true_labels, pred_labels) if t == "fake" and p != "fake")
    
    # 计算指标
    precision_legitimate, recall_legitimate, f1_legitimate = calculate_metrics(tp_legitimate, fp_legitimate, fn_legitimate)
    precision_fake, recall_fake, f1_fake = calculate_metrics(tp_fake, fp_fake, fn_fake)
    
    # 计算总体准确率
    correct = tp_legitimate + tp_fake
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'legitimate_metrics': {
            'precision': precision_legitimate,
            'recall': recall_legitimate,
            'f1_score': f1_legitimate,
            'tp': tp_legitimate,
            'fp': fp_legitimate,
            'fn': fn_legitimate
        },
        'fake_metrics': {
            'precision': precision_fake,
            'recall': recall_fake,
            'f1_score': f1_fake,
            'tp': tp_fake,
            'fp': fp_fake,
            'fn': fn_fake
        }
    }

def calculate_multiclass_metrics(data, task2_subclass):
    """计算多分类任务的详细指标"""
    if not data:
        return {}
    
    # 统计真实标签和预测标签
    true_labels = []
    pred_labels = []
    
    for item in data:
        if 'predict' not in item or 'label' not in item:
            continue
            
        # 标准化预测结果
        pred_normalized = normalize_prediction(item['predict'], "multiclass_classification", task2_subclass)
        
        # 获取真实标签并清理格式
        true_label = str(item['label']).strip().lower()
        
        # 直接使用标签，这是已经标准化后的数据
        true_labels.append(true_label)
                
        pred_labels.append(pred_normalized)
    
    # 获取所有可能的类别
    all_classes = list(set(true_labels + pred_labels))
    
    # 计算每个类别的指标
    class_metrics = {}
    total_correct = 0
    
    for class_name in all_classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == class_name and p == class_name)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != class_name and p == class_name)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == class_name and p != class_name)
        
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        total_correct += tp
    
    # 计算总体准确率
    total = len(true_labels)
    accuracy = total_correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': total_correct,
        'class_metrics': class_metrics
    }

def get_model_name_from_path(file_path):
    """从文件路径中提取模型名称"""
    # 从文件名中提取模型名称
    filename = os.path.basename(file_path)
    
    # 注意：顺序很重要！先检查更具体的模型名称
    if "Meta-Llama" in filename:
        return "LLaMA3-8B"
    elif "Qwen" in filename:
        return "Qwen1.5-7B"
    elif "Baichuan" in filename:
        return "Baichuan2-7B"
    elif "Mistral" in filename:
        return "Mistral-7B"
    elif "chatglm3-6b" in filename.lower():
        return "ChatGLM3-6B"
    elif "Llama" in filename or "LLaMA" in filename:
        return "LLaMA3-8B"
    else:
        return "Unknown"

def get_dataset_from_path(file_path):
    """从文件路径中提取数据集信息（训练和推理使用同一个数据集）"""
    filename = os.path.basename(file_path)
    
    # Task1和Task2的数据集识别
    if "_glm_" in filename:
        return "ChatGLM Dataset"
    elif "_llama3_" in filename or "_llama_" in filename:
        return "LLaMA Dataset"
    # Task3使用的是特定数据集
    elif "_gossip_" in filename:
        return "GossipCop Dataset"
    elif "_polifact_" in filename:
        return "PolitiFact Dataset"
    else:
        return "Unknown"

def analyze_predictions(file_path):
    """分析单个文件的预测结果"""
    print(f"\n📊 分析文件: {file_path}")
    
    data = load_jsonl(file_path)
    if data is None:
        return None
    
    # 判断任务类型
    task_type = get_task_type(file_path)
    task2_subclass = get_task2_subclass(file_path) if "task2" in file_path else None
    
    print(f"   任务类型: {task_type}")
    if task2_subclass:
        print(f"   Task2子类: {task2_subclass}")
    print(f"   总样本数: {len(data)}")
    
    # 统计原始预测值
    predictions = []
    labels = []
    
    for item in data:
        if 'predict' in item:
            predictions.append(item['predict'])
        if 'label' in item:
            labels.append(item['label'])
    
    predict_counter = Counter(predictions)
    label_counter = Counter(labels)
    
    # 计算详细指标
    if task_type == "binary_classification":
        metrics = calculate_binary_classification_metrics(data, task_type, task2_subclass)
    else:
        metrics = {}
    
    result = {
        'file_path': file_path,
        'model_name': get_model_name_from_path(file_path),
        'dataset': get_dataset_from_path(file_path),
        'task_type': task_type,
        'task2_subclass': task2_subclass,
        'total_samples': len(data),
        'predict_values': dict(predict_counter),
        'label_values': dict(label_counter),
        'unique_predictions': list(predict_counter.keys()),
        'unique_labels': list(label_counter.keys()),
        'metrics': metrics
    }
    
    return result

def export_to_csv(analysis_results, output_file="prediction_analysis_results.csv"):
    """导出结果到CSV文件"""
    # 按任务类型分组
    task1_results = []
    task2_results = []
    task3_results = []
    
    for result in analysis_results:
        if result and 'metrics' in result:
            file_path = result['file_path']
            if 'task1' in file_path:
                task1_results.append(result)
            elif 'task2' in file_path:
                task2_results.append(result)
            elif 'task3' in file_path:
                task3_results.append(result)
    
    # 创建三个CSV文件
    base_name = output_file.replace('.csv', '')
    
    # Task1 CSV
    if task1_results:
        task1_csv = f"{base_name}_Task1.csv"
        export_binary_task_csv(task1_results, task1_csv, "Task1")
        print(f"📄 Task1 结果已导出到: {task1_csv}")
    
    # Task2 CSV - 按子类分别导出
    if task2_results:
        # 按子类分组
        fake_results = []
        legitimate_results = []
        
        for result in task2_results:
            if "fake" in result['task2_subclass']:
                fake_results.append(result)
            else:
                legitimate_results.append(result)
        
        # 导出fake子类
        if fake_results:
            task2_fake_csv = f"{base_name}_Task2_Fake.csv"
            export_task2_fake_csv(fake_results, task2_fake_csv)
            print(f"📄 Task2 Fake 结果已导出到: {task2_fake_csv}")
        
        # 导出legitimate子类
        if legitimate_results:
            task2_legitimate_csv = f"{base_name}_Task2_Legitimate.csv"
            export_task2_legitimate_csv(legitimate_results, task2_legitimate_csv)
            print(f"📄 Task2 Legitimate 结果已导出到: {task2_legitimate_csv}")
    
    # Task3 CSV
    if task3_results:
        task3_csv = f"{base_name}_Task3.csv"
        export_binary_task_csv(task3_results, task3_csv, "Task3")
        print(f"📄 Task3 结果已导出到: {task3_csv}")

def export_binary_task_csv(results, output_file, task_name):
    """导出二分类任务的CSV"""
    csv_data = []
    
    for result in results:
        metrics = result['metrics']
        model_name = result['model_name']
        dataset = result['dataset']
        
        csv_row = {
            'Model': model_name,
            'Dataset': dataset,
            'Task': task_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Legitimate_Precision': f"{metrics['legitimate_metrics']['precision']:.4f}",
            'Legitimate_Recall': f"{metrics['legitimate_metrics']['recall']:.4f}",
            'Legitimate_F1_Score': f"{metrics['legitimate_metrics']['f1_score']:.4f}",
            'Fake_Precision': f"{metrics['fake_metrics']['precision']:.4f}",
            'Fake_Recall': f"{metrics['fake_metrics']['recall']:.4f}",
            'Fake_F1_Score': f"{metrics['fake_metrics']['f1_score']:.4f}",
        }
        csv_data.append(csv_row)
    
    if csv_data:
        fieldnames = csv_data[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def export_task2_fake_csv(results, output_file):
    """导出Task2 Fake子类的CSV"""
    csv_data = []
    
    for result in results:
        metrics = result['metrics']
        model_name = result['model_name']
        dataset = result['dataset']
        subclass = result['task2_subclass']
        
        legitimate_metrics = metrics['legitimate_metrics']
        fake_metrics = metrics['fake_metrics']
        
        # 对于fake子类：目标类别 vs Other
        target_class = subclass.replace('_fake', '').replace('_', '-').title()
        csv_row = {
            'Model': model_name,
            'Dataset': dataset,
            'Subclass': target_class,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Target_Class_Precision': f"{legitimate_metrics['precision']:.4f}",
            'Target_Class_Recall': f"{legitimate_metrics['recall']:.4f}",
            'Target_Class_F1_Score': f"{legitimate_metrics['f1_score']:.4f}",
            'Other_Precision': f"{fake_metrics['precision']:.4f}",
            'Other_Recall': f"{fake_metrics['recall']:.4f}",
            'Other_F1_Score': f"{fake_metrics['f1_score']:.4f}",
        }
        csv_data.append(csv_row)
    
    if csv_data:
        fieldnames = csv_data[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def export_task2_legitimate_csv(results, output_file):
    """导出Task2 Legitimate子类的CSV"""
    csv_data = []
    
    for result in results:
        metrics = result['metrics']
        model_name = result['model_name']
        dataset = result['dataset']
        subclass = result['task2_subclass']
        
        legitimate_metrics = metrics['legitimate_metrics']
        fake_metrics = metrics['fake_metrics']
        
        # 对于legitimate子类：Style-based vs Integration-based
        csv_row = {
            'Model': model_name,
            'Dataset': dataset,
            'Subclass': subclass,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Style_Based_Precision': f"{legitimate_metrics['precision']:.4f}",
            'Style_Based_Recall': f"{legitimate_metrics['recall']:.4f}",
            'Style_Based_F1_Score': f"{legitimate_metrics['f1_score']:.4f}",
            'Integration_Based_Precision': f"{fake_metrics['precision']:.4f}",
            'Integration_Based_Recall': f"{fake_metrics['recall']:.4f}",
            'Integration_Based_F1_Score': f"{fake_metrics['f1_score']:.4f}",
        }
        csv_data.append(csv_row)
    
    if csv_data:
        fieldnames = csv_data[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def find_result_files(base_dir="megafakeTasks"):
    """查找所有结果文件"""
    result_files = []
    
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return result_files
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jsonl'):
                result_files.append(os.path.join(root, file))
    
    return sorted(result_files)

def print_prediction_analysis(analysis_results):
    """打印预测分析结果"""
    print("\n" + "="*80)
    print("🎯 模型预测结果统计分析")
    print("="*80)
    
    # 按任务类型分组
    task1_results = []
    task2_results = []
    task3_results = []
    
    for result in analysis_results:
        if result and 'metrics' in result:
            file_path = result['file_path']
            if 'task1' in file_path:
                task1_results.append(result)
            elif 'task2' in file_path:
                task2_results.append(result)
            elif 'task3' in file_path:
                task3_results.append(result)
    
    # 打印Task1结果
    if task1_results:
        print(f"\n📈 Task1 (二分类任务) - {len(task1_results)} 个文件:")
        total_accuracy = 0
        
        for result in task1_results:
            metrics = result['metrics']
            print(f"\n📁 {result['file_path']}")
            print(f"   模型: {result['model_name']}")
            print(f"   数据集: {result['dataset']}")
            print(f"   样本数: {metrics['total_samples']}")
            print(f"   总体准确率: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
            
            # Legitimate类别指标
            leg_metrics = metrics['legitimate_metrics']
            print(f"\n   📊 Legitimate类别指标:")
            print(f"      Precision: {leg_metrics['precision']:.4f}")
            print(f"      Recall:    {leg_metrics['recall']:.4f}")
            print(f"      F1-Score:  {leg_metrics['f1_score']:.4f}")
            print(f"      TP: {leg_metrics['tp']}, FP: {leg_metrics['fp']}, FN: {leg_metrics['fn']}")
            
            # Fake类别指标
            fake_metrics = metrics['fake_metrics']
            print(f"\n   📊 Fake类别指标:")
            print(f"      Precision: {fake_metrics['precision']:.4f}")
            print(f"      Recall:    {fake_metrics['recall']:.4f}")
            print(f"      F1-Score:  {fake_metrics['f1_score']:.4f}")
            print(f"      TP: {fake_metrics['tp']}, FP: {fake_metrics['fp']}, FN: {fake_metrics['fn']}")
            
            total_accuracy += metrics['accuracy']
        
        if len(task1_results) > 1:
            avg_accuracy = total_accuracy / len(task1_results)
            print(f"\n🎯 Task1 平均准确率: {avg_accuracy:.4f}")
    
    # 打印Task3结果
    if task3_results:
        print(f"\n📈 Task3 (二分类任务) - {len(task3_results)} 个文件:")
        total_accuracy = 0
        
        for result in task3_results:
            metrics = result['metrics']
            print(f"\n📁 {result['file_path']}")
            print(f"   模型: {result['model_name']}")
            print(f"   数据集: {result['dataset']}")
            print(f"   样本数: {metrics['total_samples']}")
            print(f"   总体准确率: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
            
            # Legitimate类别指标
            leg_metrics = metrics['legitimate_metrics']
            print(f"\n   📊 Legitimate类别指标:")
            print(f"      Precision: {leg_metrics['precision']:.4f}")
            print(f"      Recall:    {leg_metrics['recall']:.4f}")
            print(f"      F1-Score:  {leg_metrics['f1_score']:.4f}")
            print(f"      TP: {leg_metrics['tp']}, FP: {leg_metrics['fp']}, FN: {leg_metrics['fn']}")
            
            # Fake类别指标
            fake_metrics = metrics['fake_metrics']
            print(f"\n   📊 Fake类别指标:")
            print(f"      Precision: {fake_metrics['precision']:.4f}")
            print(f"      Recall:    {fake_metrics['recall']:.4f}")
            print(f"      F1-Score:  {fake_metrics['f1_score']:.4f}")
            print(f"      TP: {fake_metrics['tp']}, FP: {fake_metrics['fp']}, FN: {fake_metrics['fn']}")
            
            total_accuracy += metrics['accuracy']
        
        if len(task3_results) > 1:
            avg_accuracy = total_accuracy / len(task3_results)
            print(f"\n🎯 Task3 平均准确率: {avg_accuracy:.4f}")
    
    # 打印Task2结果
    if task2_results:
        print(f"\n📈 Task2 (二分类任务) - {len(task2_results)} 个文件:")
        total_accuracy = 0
        
        for result in task2_results:
            metrics = result['metrics']
            print(f"\n📁 {result['file_path']}")
            print(f"   模型: {result['model_name']}")
            print(f"   数据集: {result['dataset']}")
            print(f"   子类: {result['task2_subclass']}")
            print(f"   样本数: {metrics['total_samples']}")
            print(f"   总体准确率: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
            
            # 根据子类类型调整标签显示
            if "fake" in result['task2_subclass']:
                # 对于fake子类：正类 vs Other
                target_class = result['task2_subclass'].replace('_fake', '').replace('_', '-').title()
                print(f"\n   📊 {target_class}类别指标 (正类):")
            else:
                # 对于legitimate子类：Style-based vs Integration-based
                print(f"\n   📊 Style-Based类别指标:")
            
            leg_metrics = metrics['legitimate_metrics']
            print(f"      Precision: {leg_metrics['precision']:.4f}")
            print(f"      Recall:    {leg_metrics['recall']:.4f}")
            print(f"      F1-Score:  {leg_metrics['f1_score']:.4f}")
            print(f"      TP: {leg_metrics['tp']}, FP: {leg_metrics['fp']}, FN: {leg_metrics['fn']}")
            
            # 负类指标
            if "fake" in result['task2_subclass']:
                print(f"\n   📊 Other类别指标 (负类):")
            else:
                print(f"\n   📊 Integration-Based类别指标:")
            
            fake_metrics = metrics['fake_metrics']
            print(f"      Precision: {fake_metrics['precision']:.4f}")
            print(f"      Recall:    {fake_metrics['recall']:.4f}")
            print(f"      F1-Score:  {fake_metrics['f1_score']:.4f}")
            print(f"      TP: {fake_metrics['tp']}, FP: {fake_metrics['fp']}, FN: {fake_metrics['fn']}")
            
            total_accuracy += metrics['accuracy']
        
        if len(task2_results) > 1:
            avg_accuracy = total_accuracy / len(task2_results)
            print(f"\n🎯 Task2 平均准确率: {avg_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='分析模型预测结果')
    parser.add_argument('--dir', default='megafakeTasks', help='结果文件目录')
    parser.add_argument('--file', help='指定单个文件进行分析')
    parser.add_argument('--output', default='megafakeTasks/results/prediction_analysis_results_base_models.csv', help='CSV输出文件名')
    parser.add_argument('--no-csv', action='store_true', help='不导出CSV文件')
    
    args = parser.parse_args()
    
    print("🚀 开始分析模型预测结果")
    
    analysis_results = []
    
    if args.file:
        # 分析单个文件
        if os.path.exists(args.file):
            result = analyze_predictions(args.file)
            if result:
                analysis_results.append(result)
        else:
            print(f"❌ 文件不存在: {args.file}")
            return
    else:
        # 查找并分析所有文件
        result_files = find_result_files(args.dir)
        
        if not result_files:
            print(f"❌ 在目录 {args.dir} 中没有找到任何 .jsonl 文件")
            return
        
        print(f"📁 找到 {len(result_files)} 个结果文件")
        
        for file_path in result_files:
            result = analyze_predictions(file_path)
            if result:
                analysis_results.append(result)
    
    # 打印分析结果
    print_prediction_analysis(analysis_results)
    
    # 导出CSV文件
    if not args.no_csv:
        export_to_csv(analysis_results, args.output)
    
    print(f"\n✅ 分析完成！")

if __name__ == "__main__":
    main() 