"""
改进的配置类

解决配置问题，包括：
1. 更好的默认值
2. 参数验证
3. 动态调整
4. 环境适配
"""

import os
import torch
from typing import Dict, List, Optional, Any
import logging


class ImprovedConfig:
    """改进的配置类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 系统配置
        self.system = SystemConfig()
        
        # 数据配置
        self.data = DataConfig()
        
        # 模型配置
        self.model = ModelConfig()
        
        # 训练配置
        self.training = TrainingConfig()
        
        # 设备配置
        self.device = self._setup_device()
        
        # 验证配置
        self.validate()
    
    def _setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("使用CPU")
        
        return device
    
    def validate(self):
        """验证配置"""
        try:
            # 验证数据路径
            if not os.path.exists(self.data.data_path):
                self.logger.warning(f"数据文件不存在: {self.data.data_path}")
                return False
            
            # 验证输出目录
            os.makedirs(self.data.output_dir, exist_ok=True)
            
            # 验证模型参数
            if self.model.hidden_dim <= 0:
                raise ValueError("hidden_dim必须大于0")
            
            if self.model.num_layers <= 0:
                raise ValueError("num_layers必须大于0")
            
            if self.training.learning_rate <= 0:
                raise ValueError("learning_rate必须大于0")
            
            if self.training.epochs <= 0:
                raise ValueError("epochs必须大于0")
            
            self.logger.info("配置验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
    
    def print_config(self):
        """打印配置信息"""
        self.logger.info("=" * 50)
        self.logger.info("配置信息")
        self.logger.info("=" * 50)
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"数据路径: {self.data.data_path}")
        self.logger.info(f"输出目录: {self.data.output_dir}")
        self.logger.info(f"隐藏维度: {self.model.hidden_dim}")
        self.logger.info(f"时序维度: {self.model.temporal_dim}")
        self.logger.info(f"GNN层数: {self.model.num_layers}")
        self.logger.info(f"注意力头数: {self.model.num_heads}")
        self.logger.info(f"学习率: {self.training.learning_rate}")
        self.logger.info(f"训练轮数: {self.training.epochs}")
        self.logger.info(f"批次大小: {self.training.batch_size}")
        self.logger.info(f"早停耐心值: {self.training.early_stopping_patience}")
        self.logger.info("=" * 50)


class SystemConfig:
    """系统配置"""
    
    def __init__(self):
        self.log_level = 'INFO'
        self.log_file = None
        self.seed = 42
        self.visualize = True
        self.num_workers = 4
        self.pin_memory = True


class DataConfig:
    """数据配置"""
    
    def __init__(self):
        self.data_path = './Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv'
        self.output_dir = './output'
        self.chunk_size = 10000
        self.max_memory_usage = 0.8
        self.use_sparse_features = True
        self.feature_seed = 42
        self.num_snapshots = 15
        self.temporal_window_hours = 24
        
        # 数据增强配置
        self.augmentation = {
            'enabled': True,
            'target_positive_ratio': 0.2,
            'use_smote': True,
            'use_adasyn': False,
            'use_undersampling': True
        }
        
        # 特征工程配置
        self.feature_engineering = {
            'use_mitre_features': True,
            'use_temporal_features': True,
            'use_network_features': True,
            'use_file_features': True,
            'use_command_features': True,
            'use_user_features': True
        }


class ModelConfig:
    """模型配置"""
    
    def __init__(self):
        self.hidden_dim = 128
        self.temporal_dim = 64
        self.num_layers = 3
        self.num_heads = 8
        self.dropout = 0.3
        self.activation = 'relu'
        self.normalization = 'batch'
        
        # 异质图编码器配置
        self.hgnn_encoder = {
            'type': 'gat',  # 'gat', 'gcn', 'transformer'
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.3,
            'use_residual': True,
            'use_layer_norm': True
        }
        
        # 时序编码器配置
        self.temporal_encoder = {
            'type': 'transformer',  # 'transformer', 'lstm', 'gru'
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.3,
            'max_length': 5000
        }
        
        # 节点分类器配置
        self.node_classifier = {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batch_norm': True
        }


class TrainingConfig:
    """训练配置"""
    
    def __init__(self):
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.epochs = 100
        self.batch_size = 64
        self.early_stopping_patience = 20
        self.min_delta = 1e-4
        self.max_grad_norm = 1.0
        
        # 优化器配置
        self.optimizer = {
            'type': 'adamw',  # 'adam', 'adamw', 'sgd'
            'lr': 0.001,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
        
        # 学习率调度器配置
        self.scheduler = {
            'type': 'cosine',  # 'cosine', 'plateau', 'step', 'none'
            'T_max': 100,
            'eta_min': 1e-6,
            'factor': 0.5,
            'patience': 5,
            'step_size': 20,
            'gamma': 0.5
        }
        
        # 损失函数配置
        self.loss = {
            'type': 'focal',  # 'focal', 'cross_entropy', 'weighted_ce'
            'alpha': 0.25,
            'gamma': 2.0,
            'adaptive_alpha': True,
            'adaptive_gamma': True,
            'class_weights': None
        }
        
        # 数据加载器配置
        self.dataloader = {
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': False
        }
        
        # 验证配置
        self.validation = {
            'val_split': 0.2,
            'test_split': 0.1,
            'stratify': True,
            'random_state': 42
        }
        
        # 检查点配置
        self.checkpoint = {
            'save_best': True,
            'save_last': True,
            'save_frequency': 10,
            'monitor': 'val_loss',
            'mode': 'min'
        }
        
        # GPU配置
        self.gpu = {
            'gpu_id': 0,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1
        }


class MemoryOptimizedConfig(ImprovedConfig):
    """内存优化配置"""
    
    def __init__(self):
        super().__init__()
        
        # 调整内存相关参数
        self.data.chunk_size = 5000
        self.data.max_memory_usage = 0.6
        self.data.num_snapshots = 10
        
        self.model.hidden_dim = 64
        self.model.temporal_dim = 32
        
        self.training.batch_size = 32
        self.training.gradient_accumulation_steps = 2
        
        self.logger.info("使用内存优化配置")


class GPUMemoryOptimizedConfig(ImprovedConfig):
    """GPU内存优化配置"""
    
    def __init__(self):
        super().__init__()
        
        # GPU内存优化参数
        self.model.hidden_dim = 96
        self.model.temporal_dim = 48
        
        self.training.batch_size = 48
        self.training.mixed_precision = True
        
        # 减少模型复杂度
        self.model.hgnn_encoder['num_layers'] = 2
        self.model.temporal_encoder['num_layers'] = 1
        
        self.logger.info("使用GPU内存优化配置")


def get_config(config_type: str = 'default') -> ImprovedConfig:
    """获取配置对象"""
    if config_type == 'memory_optimized':
        return MemoryOptimizedConfig()
    elif config_type == 'gpu_optimized':
        return GPUMemoryOptimizedConfig()
    else:
        return ImprovedConfig()
