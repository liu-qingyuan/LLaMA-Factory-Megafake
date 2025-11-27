#!/usr/bin/env python3
"""
内存和进程监控脚本
用于监控敏感性分析实验的内存使用情况
"""

import psutil
import time
import subprocess
import json
from datetime import datetime

def get_gpu_memory():
    """获取GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                used, total = line.split(', ')
                gpu_info.append({
                    'used_mb': int(used),
                    'total_mb': int(total),
                    'usage_percent': (int(used) / int(total)) * 100
                })
            return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    return []

def get_process_info():
    """获取相关进程信息"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in ['python', 'llamafactory', 'vllm', 'sensitivity']):
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent'],
                    'memory_mb': proc.info['memory_percent'] * psutil.virtual_memory().total / 100 / 1024 / 1024,
                    'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def main():
    print("🔍 开始监控系统资源...")
    print("=" * 80)

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 系统内存
            memory = psutil.virtual_memory()
            print(f"⏰ {timestamp}")
            print(f"💾 系统内存: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")

            # GPU内存
            gpu_info = get_gpu_memory()
            for i, gpu in enumerate(gpu_info):
                print(f"🎮 GPU {i}: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['usage_percent']:.1f}%)")

            # 相关进程
            processes = get_process_info()
            if processes:
                print("🔧 相关进程:")
                for proc in processes:
                    print(f"  PID {proc['pid']}: {proc['name']} (CPU: {proc['cpu_percent']:.1f}%, 内存: {proc['memory_mb']:.1f}MB)")
                    print(f"    命令: {proc['cmdline']}")

            print("-" * 80)
            time.sleep(30)  # 每30秒监控一次

    except KeyboardInterrupt:
        print("\n⏹️  监控已停止")

if __name__ == "__main__":
    main()