"""
性能优化配置

专门用于解决模型性能问题的配置
"""

import torch
from .improved_config import ImprovedConfig


class PerformanceOptimizedConfig(ImprovedConfig):
    """性能优化配置类"""
    
    def __init__(self):
        super().__init__()
        
        # 模型配置优化
        self.model.hidden_dim = 256  # 增加隐藏维度
        self.model.temporal_dim = 128  # 增加时间维度
        self.model.num_layers = 4  # 增加层数
        self.model.num_heads = 16  # 增加注意力头数
        self.model.dropout = 0.1  # 减少dropout
        self.model.activation = 'gelu'  # 使用GELU激活函数
        self.model.normalization = 'layer'  # 使用LayerNorm
        
        # 训练配置优化
        self.training.learning_rate = 0.0005  # 降低学习率
        self.training.weight_decay = 1e-5  # 减少权重衰减
        self.training.epochs = 200  # 增加训练轮数
        self.training.batch_size = 32  # 减小批次大小
        self.training.early_stopping_patience = 50  # 增加早停耐心
        self.training.min_delta = 1e-5  # 减小最小变化
        self.training.max_grad_norm = 0.5  # 减小梯度裁剪
        
        # 学习率调度优化
        self.training.lr_scheduler = 'cosine_with_warmup'
        self.training.warmup_epochs = 10
        self.training.lr_decay_factor = 0.1
        self.training.lr_patience = 15
        
        # 数据配置优化
        self.data.chunk_size = 5000  # 减小数据块大小
        self.data.max_memory_usage = 0.7  # 减少内存使用
        self.data.use_sparse_features = True
        self.data.feature_seed = 42
        self.data.num_snapshots = 20  # 增加快照数量
        self.data.temporal_window_hours = 12  # 减小时间窗口
        
        # 损失函数配置
        self.training.loss_type = 'focal'  # 使用Focal Loss
        self.training.focal_alpha = 0.25
        self.training.focal_gamma = 2.0
        self.training.label_smoothing = 0.1  # 标签平滑
        
        # 正则化配置
        self.training.l2_reg = 1e-4
        self.training.dropout_schedule = True
        self.training.dropout_start = 0.3
        self.training.dropout_end = 0.1
        
        # 数据增强配置
        self.training.use_data_augmentation = True
        self.training.augmentation_ratio = 0.3
        self.training.mixup_alpha = 0.2
        self.training.cutmix_alpha = 1.0
        
        # 模型集成配置
        self.training.use_ensemble = True
        self.training.ensemble_size = 3
        self.training.ensemble_method = 'voting'
        
        # 验证配置
        self.training.validation_frequency = 5  # 每5个epoch验证一次
        self.training.save_frequency = 10  # 每10个epoch保存一次
        
        # 优化器配置
        self.training.optimizer = 'adamw'  # 使用AdamW
        self.training.beta1 = 0.9
        self.training.beta2 = 0.999
        self.training.eps = 1e-8
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training.mixed_precision = True  # 使用混合精度
        self.training.gradient_accumulation_steps = 2  # 梯度累积
        
        # 日志配置
        self.system.log_level = 'INFO'
        self.system.visualize = True
        self.system.save_plots = True
        
        # 验证配置
        self.validate()
    
    def validate(self):
        """验证配置参数"""
        super().validate()
        
        # 验证性能优化参数
        assert 0 < self.model.hidden_dim <= 512, "隐藏维度应在1-512之间"
        assert 0 < self.model.temporal_dim <= 256, "时间维度应在1-256之间"
        assert 1 <= self.model.num_layers <= 8, "层数应在1-8之间"
        assert 1 <= self.model.num_heads <= 32, "注意力头数应在1-32之间"
        assert 0 <= self.model.dropout < 1, "Dropout应在0-1之间"
        assert 0 < self.training.learning_rate < 1, "学习率应在0-1之间"
        assert 0 <= self.training.weight_decay < 1, "权重衰减应在0-1之间"
        assert 1 <= self.training.epochs <= 1000, "训练轮数应在1-1000之间"
        assert 1 <= self.training.batch_size <= 256, "批次大小应在1-256之间"
        
        self.logger.info("性能优化配置验证通过")
    
    def print_config(self):
        """打印配置信息"""
        super().print_config()
        self.logger.info("使用性能优化配置")
        self.logger.info(f"隐藏维度: {self.model.hidden_dim}")
        self.logger.info(f"时间维度: {self.model.temporal_dim}")
        self.logger.info(f"层数: {self.model.num_layers}")
        self.logger.info(f"注意力头数: {self.model.num_heads}")
        self.logger.info(f"学习率: {self.training.learning_rate}")
        self.logger.info(f"批次大小: {self.training.batch_size}")
        self.logger.info(f"训练轮数: {self.training.epochs}")
        self.logger.info(f"混合精度: {self.training.mixed_precision}")
        self.logger.info(f"模型集成: {self.training.use_ensemble}")

