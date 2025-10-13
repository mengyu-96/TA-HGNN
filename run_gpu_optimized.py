#!/usr/bin/env python3
"""
GPU优化运行脚本

使用GPU优化配置运行T-HGNN系统
"""

import sys
import os
import subprocess
import argparse
import logging
import torch

def check_gpu_availability():
    """检查GPU可用性"""
    if not torch.cuda.is_available():
        print("X CUDA不可用，将使用CPU训练")
        return False
    
    print(f"V CUDA可用，检测到 {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return True

def get_optimal_settings():
    """根据GPU内存获取最优设置"""
    if not torch.cuda.is_available():
        return {
            'batch_size': 8,
            'hidden_dim': 32,
            'num_heads': 2,
            'num_layers': 1,
            'sample_size': 10000,
            'epochs': 20
        }
    
    # 获取GPU内存信息
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    if gpu_memory < 4:
        # 低端GPU
        return {
            'batch_size': 4,
            'hidden_dim': 32,
            'num_heads': 2,
            'num_layers': 1,
            'sample_size': 10000,
            'epochs': 20
        }
    elif gpu_memory < 8:
        # 中端GPU
        return {
            'batch_size': 8,
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'sample_size': 20000,
            'epochs': 50
        }
    else:
        # 高端GPU
        return {
            'batch_size': 16,
            'hidden_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
            'sample_size': 50000,
            'epochs': 100
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPU优化模式运行T-HGNN系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'detect', 'trace', 'cluster', 'full'],
                       help='运行模式')
    parser.add_argument('--data_path', type=str, 
                       default='./Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    
    # GPU参数
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--auto_settings', action='store_true',
                       help='根据GPU自动设置参数')
    
    # 手动参数（当不使用auto_settings时）
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='网络层数')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='数据采样大小')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--memory_optimized', action='store_true',
                       help='启用内存优化模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("T-HGNN GPU优化模式")
    logger.info("=" * 60)
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    # 获取最优设置
    if args.auto_settings:
        optimal_settings = get_optimal_settings()
        logger.info("根据GPU自动设置参数:")
        for key, value in optimal_settings.items():
            logger.info(f"  {key}: {value}")
    else:
        optimal_settings = {}
    
    # 构建命令
    cmd = [
        sys.executable, 'main.py',
        '--mode', args.mode,
        '--data_path', args.data_path,
        '--output_dir', args.output_dir,
        '--gpu', str(args.gpu),
        '--seed', str(args.seed)
    ]
    
    # 添加参数
    if args.memory_optimized:
        cmd.append('--memory_optimized')
    
    # 使用自动设置或手动设置
    if args.auto_settings:
        cmd.extend([
            '--epochs', str(optimal_settings.get('epochs', 50)),
            '--batch_size', str(optimal_settings.get('batch_size', 8)),
            '--hidden_dim', str(optimal_settings.get('hidden_dim', 64)),
            '--num_heads', str(optimal_settings.get('num_heads', 4)),
            '--num_layers', str(optimal_settings.get('num_layers', 2))
        ])
    else:
        # 使用手动参数或默认值
        if args.epochs is not None:
            cmd.extend(['--epochs', str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(['--batch_size', str(args.batch_size)])
        if args.hidden_dim is not None:
            cmd.extend(['--hidden_dim', str(args.hidden_dim)])
        if args.num_heads is not None:
            cmd.extend(['--num_heads', str(args.num_heads)])
        if args.num_layers is not None:
            cmd.extend(['--num_layers', str(args.num_layers)])
        if args.sample_size is not None:
            cmd.extend(['--sample_size', str(args.sample_size)])
    
    # 添加可视化参数（仅在非训练模式下）
    if args.mode != 'train':
        cmd.append('--visualize')
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("程序执行完成")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"程序执行失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
