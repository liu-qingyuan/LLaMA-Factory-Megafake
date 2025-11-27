#!/usr/bin/env python3
"""
系统资源监控脚本
监控GPU内存和进程状态
"""

import psutil
import subprocess
import time
from datetime import datetime

def get_gpu_memory():
    """获取GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        used, total, util = map(int, parts[:3])
                        gpu_info.append({
                            'used_mb': used,
                            'total_mb': total,
                            'usage_percent': (used / total) * 100,
                            'utilization_percent': util
                        })
            return gpu_info
    except Exception:
        pass
    return []

def get_sensitivity_processes():
    """获取敏感性分析相关进程"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in ['sensitivity', 'llamafactory', 'vllm']):
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent'],
                    'memory_mb': proc.info['memory_percent'] * psutil.virtual_memory().total / 100 / 1024 / 1024,
                    'cmdline': cmdline[:80] + '...' if len(cmdline) > 80 else cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def main():
    print("🔍 敏感性分析系统监控")
    print("=" * 60)
    print("按 Ctrl+C 停止监控")
    print()

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 系统内存
            memory = psutil.virtual_memory()
            print(f"⏰ {timestamp}")
            print(f"💾 系统内存: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")

            # GPU内存
            gpu_info = get_gpu_memory()
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    status = "🚨" if gpu['usage_percent'] > 85 else "✅" if gpu['usage_percent'] < 60 else "⚠️"
                    print(f"🎮 GPU {i}: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['usage_percent']:.1f}%) {status}")
            else:
                print("🎮 GPU: 无法获取信息")

            # 相关进程
            processes = get_sensitivity_processes()
            if processes:
                print("🔧 相关进程:")
                for proc in processes:
                    print(f"  PID {proc['pid']:>5}: {proc['name']} (CPU: {proc['cpu_percent']:>5.1f}%, 内存: {proc['memory_mb']:>6.1f}MB)")
            else:
                print("🔧 相关进程: 无")

            print("-" * 60)
            time.sleep(30)  # 30秒更新一次

    except KeyboardInterrupt:
        print("\n⏹️ 监控已停止")

if __name__ == "__main__":
    main()