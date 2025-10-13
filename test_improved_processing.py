#!/usr/bin/env python3
"""
测试改进的数据处理功能
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import DataConfig
from src.data.apt_data_processor import APTDataProcessor

def test_improved_data_processing():
    """测试改进的数据处理功能"""
    print("=== 测试改进的数据处理功能 ===")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = DataConfig()
    config.data_path = './Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv'
    
    # 创建数据处理器
    processor = APTDataProcessor(config)
    
    # 加载数据
    print("1. 加载原始数据...")
    try:
        df = pd.read_csv(config.data_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        print(f"   原始数据形状: {df.shape}")
    except Exception as e:
        print(f"   加载失败: {e}")
        return
    
    # 测试改进的数据质量修复
    print("2. 测试数据质量修复...")
    try:
        improved_df = processor.fix_data_quality_issues(df.copy())
        print(f"   修复后数据形状: {improved_df.shape}")
        
        # 检查时间戳
        if 'processed_timestamp' in improved_df.columns:
            timestamp_count = improved_df['processed_timestamp'].notna().sum()
            print(f"   有效时间戳数量: {timestamp_count}/{len(improved_df)} ({timestamp_count/len(improved_df)*100:.1f}%)")
        
        # 检查标签分布
        if 'security_label' in improved_df.columns:
            label_counts = improved_df['security_label'].value_counts()
            print("   标签分布:")
            for label, count in label_counts.items():
                ratio = count / len(improved_df) * 100
                print(f"     {label}: {count:,} ({ratio:.1f}%)")
        
        # 检查缺失值
        missing_stats = improved_df.isnull().sum()
        high_missing = missing_stats[missing_stats > len(improved_df) * 0.5]
        print(f"   高缺失率字段数: {len(high_missing)}")
        
        # 检查特征维度
        feature_cols = [col for col in improved_df.columns if col.endswith('_features')]
        print(f"   特征列数: {len(feature_cols)}")
        for col in feature_cols:
            if col in improved_df.columns:
                sample_features = improved_df[col].iloc[0]
                if isinstance(sample_features, list):
                    print(f"     {col}: 维度 {len(sample_features)}")
        
    except Exception as e:
        print(f"   数据质量修复失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试改进的处理方法
    print("3. 测试改进的处理方法...")
    try:
        improved_df2 = processor.process_data_improved(df.copy())
        print(f"   改进处理后数据形状: {improved_df2.shape}")
        
        # 检查标签分布
        if 'security_label' in improved_df2.columns:
            label_counts = improved_df2['security_label'].value_counts()
            print("   改进后标签分布:")
            for label, count in label_counts.items():
                ratio = count / len(improved_df2) * 100
                print(f"     {label}: {count:,} ({ratio:.1f}%)")
        
    except Exception as e:
        print(f"   改进处理方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_improved_data_processing()

