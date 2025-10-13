#!/usr/bin/env python3
"""
内存优化运行脚本

使用内存优化配置运行T-HGNN系统
"""

import sys
import os
import subprocess
import argparse
import logging

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='内存优化模式运行T-HGNN系统')
    
    # 基本参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数（默认10）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小（默认8）')
    parser.add_argument('--data_path', type=str, 
                       default='./Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    
    # 内存优化参数
    parser.add_argument('--max_memory_percent', type=float, default=60.0,
                       help='最大内存使用百分比（默认60%）')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='数据采样大小（默认10000）')
    
    # 其他参数
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'detect', 'trace', 'cluster', 'full'],
                       help='运行模式')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("T-HGNN 内存优化模式")
    logger.info("=" * 60)
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"数据采样: {args.sample_size}")
    logger.info(f"最大内存使用: {args.max_memory_percent}%")
    logger.info("=" * 60)
    
    # 构建命令
    cmd = [
        sys.executable, 'main.py',
        '--memory_optimized',  # 启用内存优化模式
        '--mode', args.mode,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--data_path', args.data_path,
        '--output_dir', args.output_dir,
        '--gpu', str(args.gpu),
        '--seed', str(args.seed)
    ]
    
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
