# GPU使用指南

本指南介绍如何在T-HGNN系统中使用GPU加速训练。

## 1. GPU检测

### 检查GPU可用性
```bash
python check_gpu.py
```

### 手动检查
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
```

## 2. GPU优化运行

### 自动GPU优化
```bash
# 根据GPU自动设置最优参数
python run_gpu_optimized.py --auto_settings --mode train

# 内存优化模式（适用于低端GPU）
python run_gpu_optimized.py --auto_settings --memory_optimized --mode train
```

### 手动GPU优化
```bash
# 指定GPU和参数
python run_gpu_optimized.py --gpu 0 --batch_size 16 --hidden_dim 128 --epochs 100
```

### 标准模式（带GPU检测）
```bash
# 使用标准配置但启用GPU
python main.py --mode train --gpu 0 --batch_size 16
```

## 3. GPU配置说明

### 设备设置
- `--gpu 0`: 使用GPU 0
- `--gpu 1`: 使用GPU 1（如果有多GPU）
- 自动检测: 系统会自动选择可用的GPU

### 内存优化
- `--memory_optimized`: 启用内存优化模式
- 自动调整批次大小和模型参数
- 定期清理GPU缓存

## 4. 不同GPU的推荐设置

### 低端GPU (< 4GB)
```bash
python run_gpu_optimized.py --auto_settings --memory_optimized
```
- 批次大小: 4-8
- 隐藏层维度: 32-64
- 注意力头数: 2-4
- 网络层数: 1-2

### 中端GPU (4-8GB)
```bash
python run_gpu_optimized.py --auto_settings
```
- 批次大小: 8-16
- 隐藏层维度: 64-128
- 注意力头数: 4-8
- 网络层数: 2-3

### 高端GPU (> 8GB)
```bash
python run_gpu_optimized.py --auto_settings
```
- 批次大小: 16-32
- 隐藏层维度: 128-256
- 注意力头数: 8-16
- 网络层数: 3-4

## 5. GPU监控

### 实时监控
```python
from src.utils.gpu_utils import GPUUtils

gpu_utils = GPUUtils()
gpu_utils.print_gpu_status()
```

### 内存使用情况
```python
import torch

# 获取GPU内存使用情况
allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
reserved = torch.cuda.memory_reserved() / (1024**3)     # GB
total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

print(f"已分配: {allocated:.2f}GB")
print(f"已缓存: {reserved:.2f}GB")
print(f"总内存: {total:.1f}GB")
```

## 6. 常见问题

### CUDA不可用
1. 检查NVIDIA驱动是否正确安装
2. 安装CUDA版本的PyTorch
3. 检查CUDA版本兼容性

### 内存不足
1. 使用`--memory_optimized`模式
2. 减少批次大小
3. 减少模型参数
4. 使用数据采样

### 性能优化
1. 启用`torch.backends.cudnn.benchmark = True`
2. 使用混合精度训练
3. 优化数据加载
4. 定期清理GPU缓存

## 7. 性能基准

### 训练速度对比
- CPU: 基准速度
- GPU (低端): 2-4x加速
- GPU (中端): 4-8x加速
- GPU (高端): 8-16x加速

### 内存使用
- CPU: 主要使用RAM
- GPU: 主要使用VRAM
- 混合: 同时使用RAM和VRAM

## 8. 最佳实践

1. **选择合适的GPU**: 根据数据量和模型复杂度选择GPU
2. **监控内存使用**: 避免GPU内存溢出
3. **优化批次大小**: 在速度和内存之间找到平衡
4. **使用混合精度**: 在支持的GPU上使用FP16训练
5. **定期清理缓存**: 避免内存泄漏

## 9. 故障排除

### 常见错误
1. `CUDA out of memory`: 减少批次大小或使用内存优化模式
2. `CUDA device not found`: 检查GPU ID是否正确
3. `CUDA version mismatch`: 检查PyTorch和CUDA版本兼容性

### 调试技巧
1. 使用`torch.cuda.is_available()`检查CUDA可用性
2. 使用`torch.cuda.device_count()`检查GPU数量
3. 使用`torch.cuda.current_device()`检查当前设备
4. 使用`torch.cuda.memory_summary()`查看内存使用情况

## 10. 示例命令

```bash
# 完整GPU优化训练
python run_gpu_optimized.py --auto_settings --mode full --memory_optimized

# 指定GPU和参数
python run_gpu_optimized.py --gpu 0 --batch_size 16 --hidden_dim 128 --epochs 100

# 检测GPU状态
python check_gpu.py

# 标准模式（带GPU）
python main.py --mode train --gpu 0 --batch_size 16 --hidden_dim 128
```
