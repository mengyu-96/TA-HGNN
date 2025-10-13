"""
数据处理模块

包含基于PyTorch Geometric的数据加载器
"""

# 延迟导入，避免在没有安装依赖时出错
try:
    from .pyg_loader import PyG_LinuxAPTDataLoader
    from .improved_apt_data_processor import ImprovedAPTDataProcessor
    from .entity_resolver import EntityResolver
    from .data_quality_evaluator import DataQualityEvaluator
    __all__ = ['PyG_LinuxAPTDataLoader', 'ImprovedAPTDataProcessor', 'EntityResolver', 'DataQualityEvaluator']
except ImportError as e:
    print(f"警告: 无法导入数据处理模块，缺少依赖: {e}")
    PyG_LinuxAPTDataLoader = None
    ImprovedAPTDataProcessor = None
    EntityResolver = None
    DataQualityEvaluator = None
    __all__ = []