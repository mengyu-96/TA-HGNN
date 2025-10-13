"""
简单配置类

作为备用配置，当改进的配置不可用时使用
"""

import torch
from typing import Dict, Any


class SimpleDataConfig:
    """简单数据配置"""
    
    def __init__(self):
        self.data_path = './Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv'
        self.output_dir = './output'
        self.chunk_size = 10000
        self.max_memory_usage = 0.8
        self.use_sparse_features = True
        self.feature_seed = 42
        self.num_snapshots = 15
        self.temporal_window_hours = 24


class SimpleModelConfig:
    """简单模型配置"""
    
    def __init__(self):
        self.hidden_dim = 128
        self.temporal_dim = 64
        self.num_layers = 3
        self.num_heads = 8
        self.dropout = 0.3
        self.activation = 'relu'
        self.normalization = 'batch'


class SimpleTrainingConfig:
    """简单训练配置"""
    
    def __init__(self):
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.epochs = 100
        self.batch_size = 64
        self.early_stopping_patience = 20
        self.min_delta = 1e-4
        self.max_grad_norm = 1.0
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True


class SimpleSystemConfig:
    """简单系统配置"""
    
    def __init__(self):
        self.log_level = 'INFO'
        self.log_file = None
        self.seed = 42
        self.visualize = True


class SimpleConfig:
    """简单配置类"""
    
    def __init__(self):
        self.data = SimpleDataConfig()
        self.model = SimpleModelConfig()
        self.training = SimpleTrainingConfig()
        self.system = SimpleSystemConfig()
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def validate(self):
        """验证配置"""
        pass
    
    def print_config(self):
        """打印配置信息"""
        print("使用简单配置")

