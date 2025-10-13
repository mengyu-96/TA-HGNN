#!/usr/bin/env python3
"""
调试数据加载过程
"""

import sys
import os
sys.path.append('src')

import logging
from src.config.improved_config import ImprovedConfig
from src.data.pyg_loader import PyG_LinuxAPTDataLoader
from src.data.improved_apt_data_processor import ImprovedAPTDataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_loading():
    """调试数据加载过程"""
    try:
        # 创建配置
        config = ImprovedConfig()
        logger.info("配置创建成功")
        
        # 验证配置
        if not config.validate():
            logger.error("配置验证失败")
            return False
        logger.info("配置验证通过")
        
        # 创建数据加载器
        data_loader = PyG_LinuxAPTDataLoader(config)
        logger.info("数据加载器创建成功")
        
        # 加载原始数据
        logger.info("开始加载原始数据...")
        df = data_loader.load_data()
        logger.info(f"原始数据加载成功，形状: {df.shape}")
        
        if df.empty:
            logger.error("原始数据为空")
            return False
        
        # 创建APT处理器
        apt_processor = ImprovedAPTDataProcessor(config)
        logger.info("APT处理器创建成功")
        
        # 处理数据
        logger.info("开始处理数据...")
        processed_df = apt_processor.process_raw_data(df)
        logger.info(f"数据处理完成，形状: {processed_df.shape}")
        
        if processed_df.empty:
            logger.error("处理后的数据为空")
            return False
        
        # 获取统计信息
        stats = apt_processor.get_data_statistics(processed_df)
        logger.info(f"数据统计: {stats}")
        
        logger.info("数据加载调试完成")
        return True
        
    except Exception as e:
        logger.error(f"数据加载调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_data_loading()
    if success:
        print("数据加载调试成功")
    else:
        print("数据加载调试失败")

