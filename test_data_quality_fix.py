#!/usr/bin/env python3
"""
测试改进的数据处理方法
"""

import sys
sys.path.append('.')
from src.data.apt_data_processor import APTDataProcessor
from src.config.config import DataConfig
import pandas as pd

def test_improved_processing():
    """测试改进的数据处理方法"""
    print("=== 测试改进的数据处理方法 ===")
    
    # 初始化处理器
    config = DataConfig()
    processor = APTDataProcessor(config)
    
    # 检查方法是否存在
    print('检查改进方法:')
    print('fix_data_quality_issues:', hasattr(processor, 'fix_data_quality_issues'))
    print('process_data_improved:', hasattr(processor, 'process_data_improved'))
    
    # 读取小样本数据进行测试
    try:
        df = pd.read_csv('./Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv', 
                        nrows=1000, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        print(f'测试数据形状: {df.shape}')
        
        # 测试数据质量修复
        if hasattr(processor, 'fix_data_quality_issues'):
            print("开始测试数据质量修复...")
            improved_df = processor.fix_data_quality_issues(df.copy())
            print(f'改进后数据形状: {improved_df.shape}')
            
            # 检查新增字段
            new_fields = set(improved_df.columns) - set(df.columns)
            print(f'新增字段数量: {len(new_fields)}')
            print(f'新增字段示例: {list(new_fields)[:5]}')
            
            # 检查时间戳字段
            if 'parsed_timestamp' in improved_df.columns:
                valid_timestamps = improved_df['parsed_timestamp'].notna().sum()
                print(f'有效时间戳数量: {valid_timestamps}')
            
            # 检查恶意标签
            if 'is_malicious' in improved_df.columns:
                positive_ratio = improved_df['is_malicious'].mean()
                print(f'正样本比例: {positive_ratio:.3f}')
            
            # 检查时间特征
            time_features = ['hour', 'day_of_week', 'is_weekend', 'is_night']
            for feature in time_features:
                if feature in improved_df.columns:
                    print(f'{feature}: {improved_df[feature].nunique()} 个唯一值')
        
        print("测试完成!")
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_processing()

