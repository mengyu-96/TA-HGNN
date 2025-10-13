#!/usr/bin/env python3
"""
GPU检测脚本

检测和显示GPU信息，提供优化建议
"""

import sys
import os
import torch
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.gpu_utils import GPUUtils
except ImportError as e:
    print(f"警告: 无法导入GPU工具: {e}")
    GPUUtils = None

def main():
    """主函数"""
    print("=" * 60)
    print("T-HGNN GPU检测工具")
    print("=" * 60)
    
    # 基本CUDA信息
    print("基本CUDA信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print()
        
        # 详细GPU信息
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总内存: {props.total_memory / (1024**3):.1f}GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数量: {props.multi_processor_count}")
            # 移除不存在的属性，使用安全的属性访问
            try:
                if hasattr(props, 'max_threads_per_block'):
                    print(f"  最大线程数: {props.max_threads_per_block}")
                if hasattr(props, 'max_grid_size'):
                    print(f"  最大网格大小: {props.max_grid_size}")
            except AttributeError:
                pass  # 忽略不存在的属性
            print()
        
        # 当前GPU状态
        print("当前GPU状态:")
        current_device = torch.cuda.current_device()
        print(f"  当前设备: {current_device}")
        
        # 内存使用情况
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        
        print(f"  已分配内存: {allocated:.2f}GB")
        print(f"  已缓存内存: {reserved:.2f}GB")
        print(f"  总内存: {total:.1f}GB")
        print(f"  可用内存: {total - reserved:.2f}GB")
        print(f"  内存使用率: {(reserved/total)*100:.1f}%")
        print()
        
        # 使用GPU工具获取更详细信息
        if GPUUtils:
            print("使用GPU工具获取详细信息:")
            gpu_utils = GPUUtils()
            gpu_utils.print_gpu_info()
            gpu_utils.print_gpu_status()
            
            # 健康检查
            health = gpu_utils.check_gpu_health()
            if health['warnings']:
                print("⚠️  警告:")
                for warning in health['warnings']:
                    print(f"  - {warning}")
            
            if health['recommendations']:
                print("💡 建议:")
                for rec in health['recommendations']:
                    print(f"  - {rec}")
            
            # 根据GPU内存推荐设置
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            recommendations = gpu_utils.recommend_optimal_settings(gpu_memory)
            
            print(f"\n针对 {gpu_memory:.1f}GB GPU的推荐设置:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")
            
            print(f"\n推荐运行命令:")
            print(f"python run_gpu_optimized.py --auto_settings --mode train")
            if gpu_memory < 8:
                print(f"python run_gpu_optimized.py --auto_settings --memory_optimized --mode train")
        
    else:
        print("X CUDA不可用，将使用CPU训练")
        print("\n建议:")
        print("  - 安装CUDA版本的PyTorch")
        print("  - 检查NVIDIA驱动是否正确安装")
        print("  - 使用CPU优化模式: python run_memory_optimized.py")
    
    print("\n" + "=" * 60)
    print("检测完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
