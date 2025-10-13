#!/usr/bin/env python3
"""
数据分析脚本 - 分析APT数据集的质量问题
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_data_quality():
    """分析数据质量"""
    print("=== APT数据集质量分析 ===")
    
    # 尝试读取数据
    data_path = './Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv'
    
    try:
        # 尝试不同的读取方式
        df = pd.read_csv(data_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        print(f"[OK] 成功读取数据，形状: {df.shape}")
    except Exception as e:
        print(f"[ERROR] UTF-8编码读取失败: {e}")
        try:
            df = pd.read_csv(data_path, on_bad_lines='skip', encoding='latin-1', low_memory=False)
            print(f"[OK] 使用latin-1编码成功读取，形状: {df.shape}")
        except Exception as e2:
            print(f"[ERROR] 所有编码都失败: {e2}")
            return None
    
    print(f"\n=== 基本信息 ===")
    print(f"总记录数: {len(df):,}")
    print(f"总字段数: {len(df.columns)}")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\n=== 缺失值分析 ===")
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100
    
    # 创建缺失值统计表
    missing_df = pd.DataFrame({
        '缺失数量': missing_stats,
        '缺失比例(%)': missing_percent
    }).sort_values('缺失比例(%)', ascending=False)
    
    # 显示缺失值最多的字段
    high_missing = missing_df[missing_df['缺失数量'] > 0]
    print(f"有缺失值的字段数: {len(high_missing)}")
    print("\n缺失值最多的前20个字段:")
    print(high_missing.head(20))
    
    # 分析缺失值严重程度
    critical_missing = high_missing[high_missing['缺失比例(%)'] > 50]
    print(f"\n缺失率超过50%的字段数: {len(critical_missing)}")
    if len(critical_missing) > 0:
        print("严重缺失字段:")
        print(critical_missing)
    
    print(f"\n=== 时间戳字段分析 ===")
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
    print(f"时间戳相关字段数: {len(timestamp_cols)}")
    
    if timestamp_cols:
        print("时间戳字段详情:")
        for col in timestamp_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            print(f"  {col}: 缺失 {missing_count:,}/{len(df):,} ({missing_pct:.1f}%)")
            
            # 显示一些样本值
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                print(f"    样本值: {non_null_values.iloc[0]}")
    else:
        print("[WARNING] 未找到时间戳相关字段")
    
    print(f"\n=== 数据类型分析 ===")
    dtype_counts = df.dtypes.value_counts()
    print("数据类型分布:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 个字段")
    
    print(f"\n=== 字段内容分析 ===")
    # 分析一些关键字段
    key_fields = ['_source.rule.description', '_source.data.command', '_source.data.file', 
                  '_source.agent.name', '_source.predecoder.hostname']
    
    for field in key_fields:
        if field in df.columns:
            non_null_count = df[field].notna().sum()
            unique_count = df[field].nunique()
            print(f"{field}:")
            print(f"  非空值: {non_null_count:,}/{len(df):,} ({non_null_count/len(df)*100:.1f}%)")
            print(f"  唯一值: {unique_count:,}")
            if non_null_count > 0:
                sample_value = df[field].dropna().iloc[0]
                print(f"  样本值: {str(sample_value)[:100]}...")
    
    return df

if __name__ == "__main__":
    df = analyze_data_quality()
    if df is not None:
        print(f"\n=== 分析完成 ===")
        print("数据质量分析完成，请查看上述报告")
    else:
        print("数据读取失败，无法进行分析")
        sys.exit(1)
