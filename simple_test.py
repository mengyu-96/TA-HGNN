#!/usr/bin/env python3
"""
简单测试脚本
"""

import sys
import os
sys.path.append('.')

# 强制重新导入
if 'src.data.apt_data_processor' in sys.modules:
    del sys.modules['src.data.apt_data_processor']

from src.data.apt_data_processor import APTDataProcessor
from src.config.config import DataConfig

config = DataConfig()
processor = APTDataProcessor(config)

# 检查方法是否存在
print('检查方法:')
print('fix_data_quality_issues:', hasattr(processor, 'fix_data_quality_issues'))
print('process_data_improved:', hasattr(processor, 'process_data_improved'))

if hasattr(processor, 'fix_data_quality_issues'):
    print('方法存在，可以调用')
else:
    print('方法不存在')

