"""
核心算法层 - 训练模块

实现模型训练和评估功能：
1. 模型训练器
2. 模型评估器
"""

from .improved_trainer import ImprovedModelTrainer as ModelTrainer
from .evaluator import ModelEvaluator

__all__ = [
    'ModelTrainer',
    'ModelEvaluator'
]
