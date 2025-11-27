# 问题排查指南

## 🔧 常见问题解决方案

### 1. 导入路径错误

#### 问题描述
```
ModuleNotFoundError: No module named 'utils.config'
ModuleNotFoundError: No module named 'sensitivity_analysis.scripts.core'
```

#### 原因分析
- 目录重组后导入路径失效
- Python路径配置不正确
- 模块初始化文件缺失

#### 解决方案

**方案1: 使用统一入口脚本**
```bash
# 推荐使用方式
python scripts/sa.py quick
```

**方案2: 手动修复路径**
```python
# 在脚本开头添加
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
```

**方案3: 验证导入修复**
```bash
python sensitivity_analysis/scripts/test_vllm_fix.py
```

### 2. VLLM推理错误

#### 问题描述
```
VLLM evaluation failed: No module named 'utils.config'
VLLM inference failed: CUDA error
```

#### 解决方案

**步骤1: 验证配置导入**
```bash
python -c "
from sensitivity_analysis.configs.config import MODEL_CONFIGS
print('配置导入成功:', len(MODEL_CONFIGS), '个模型')
"
```

**步骤2: 检查VLLM安装**
```bash
python -c "
from llamafactory.extras.packages import is_vllm_available
print('VLLM可用:', is_vllm_available())
"
```

**步骤3: 运行测试脚本**
```bash
python sensitivity_analysis/scripts/test_vllm_fix.py
```

### 3. CUDA内存错误

#### 问题描述
```
CUDA error: device-side assert triggered
RuntimeError: CUDA out of memory
```

#### 解决方案

**立即措施**
```bash
# 检查GPU状态
nvidia-smi

# 清理GPU缓存
nvidia-smi --gpu-reset
```

**代码调整**
```python
# 减少批次大小
batch_size = 4  # 从8减少到4

# 启用梯度检查点
gradient_checkpointing = True

# 使用混合精度
fp16 = True
```

**配置优化**
```yaml
# 在配置文件中
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
fp16: true
flash_attn: "auto"
```

### 4. 模型加载错误

#### 问题描述
```
OSError: Can't load tokenizer for 'xxx'
FileNotFoundError: [Errno 2] No such file or directory: 'model_name'
```

#### 解决方案

**检查模型路径**
```bash
# 验证模型存在
ls -la /root/autodl-tmp/models/

# 检查模型完整性
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen1.5-7B')
print('模型加载成功')
"
```

**修复模型配置**
```python
# 更新MODEL_CONFIGS
MODEL_CONFIGS = {
    "/root/autodl-tmp/models/Qwen1.5-7B": ("qwen", True),
    # 确保路径正确
}
```

### 5. 数据集错误

#### 问题描述
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset_file'
DatasetDict错误: 数据集格式不正确
```

#### 解决方案

**验证数据集路径**
```bash
# 检查数据集文件
find data/ -name "*.json" -o -name "*.jsonl"

# 验证数据集注册
python -c "
import json
with open('data/dataset_info.json', 'r') as f:
    datasets = json.load(f)
print('数据集数量:', len(datasets))
"
```

**修复数据集路径**
```python
# 使用相对路径
DATASET_CONFIGS = {
    "task1_small_glm": "data/data_table/task1/small_8k/alpaca_megafake_glm_8k.json",
    # 确保路径从项目根目录开始
}
```

### 6. 训练中断错误

#### 问题描述
```
训练过程中突然停止
进程被杀死
实验结果不完整
```

#### 解决方案

**检查系统资源**
```bash
# 查看内存使用
free -h

# 查看磁盘空间
df -h

# 查看GPU使用
nvidia-smi -l 1
```

**启用实验恢复**
```python
# 在脚本中添加断点续训
resume_from_checkpoint: "latest"
```

**使用后台运行**
```bash
# 使用tmux运行
tmux new -s sensitivity
python scripts/sa.py quick

# 使用nohup运行
nohup python scripts/sa.py quick > experiment.log 2>&1 &
```

### 7. 性能问题

#### 问题描述
```
推理速度慢
训练时间长
内存使用过高
```

#### 解决方案

**启用加速**
```python
# Flash Attention
flash_attn: "auto"

# VLLM加速
VLLM_ENABLED=true python scripts/sa.py quick

# 混合精度
fp16: true
bf16: true
```

**优化配置**
```yaml
# 推理优化
per_device_eval_batch_size: 32
max_new_tokens: 10  # 减少生成长度

# 训练优化
dataloader_num_workers: 4
preprocessing_num_workers: 16
```

### 8. 依赖问题

#### 问题描述
```
ImportError: No module named 'transformers'
版本冲突: incompatible versions
```

#### 解决方案

**重新安装依赖**
```bash
# 卸载旧版本
pip uninstall transformers peft accelerate

# 重新安装
pip install transformers>=4.30.0 peft>=0.4.0 accelerate>=0.20.0
```

**检查版本兼容性**
```bash
python -c "
import transformers
import peft
import accelerate
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('accelerate:', accelerate.__version__)
"
```

## 🔍 调试技巧

### 1. 启用详细日志
```bash
export PYTHONPATH=/root/autodl-tmp/LLaMA-Factory-Megafake:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
```

### 2. 使用调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在配置中
debug: true
```

### 3. 分步验证
```bash
# 第1步: 测试配置
python scripts/sa.py test

# 第2步: 快速分析
python scripts/sa.py quick

# 第3步: 监控状态
python scripts/sa.py monitor
```

### 4. 查看日志
```bash
# 查看最新日志
tail -f sensitivity_analysis/results/logs/*.log

# 搜索错误
grep -i error sensitivity_analysis/results/logs/*.log
```

## 📞 获取帮助

### 1. 运行诊断脚本
```bash
python sensitivity_analysis/scripts/test_vllm_fix.py
python sensitivity_analysis/model_utils/verify_models.py
```

### 2. 收集错误信息
```bash
# 收集系统信息
nvidia-smi
free -h
df -h
python --version

# 收集错误日志
ls -la sensitivity_analysis/results/logs/
```

### 3. 参考文档
- [用户手册](README.md)
- [设置指南](SETUP.md)
- [项目主文档](../../CLAUDE.md)

---

**文档版本**: v1.0
**最后更新**: 2025-11-17
**维护状态**: ✅ 活跃维护中