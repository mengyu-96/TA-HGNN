"""
改进的Focal Loss实现

解决类别不平衡问题，包括：
1. 动态alpha调整
2. 类别权重自适应
3. 难样本挖掘
4. 多任务学习支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union
import logging


class ImprovedFocalLoss(nn.Module):
    """
    改进的Focal Loss
    
    结合了Focal Loss、类别权重和难样本挖掘的优势
    """
    
    def __init__(self, 
                 alpha: Union[float, List[float], torch.Tensor] = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 class_weights: Optional[torch.Tensor] = None,
                 adaptive_alpha: bool = True,
                 adaptive_gamma: bool = True,
                 min_alpha: float = 0.01,
                 max_alpha: float = 0.99,
                 min_gamma: float = 0.5,
                 max_gamma: float = 5.0):
        """
        初始化改进的Focal Loss
        
        Args:
            alpha: 类别权重，可以是标量、列表或张量
            gamma: 聚焦参数
            reduction: 损失归约方式
            class_weights: 额外的类别权重
            adaptive_alpha: 是否使用自适应alpha
            adaptive_gamma: 是否使用自适应gamma
            min_alpha: alpha的最小值
            max_alpha: alpha的最大值
            min_gamma: gamma的最小值
            max_gamma: gamma的最大值
        """
        super(ImprovedFocalLoss, self).__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_gamma = adaptive_gamma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        # 初始化alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        
        # 自适应参数
        if self.adaptive_alpha:
            self.alpha = nn.Parameter(self.alpha)
        if self.adaptive_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.loss_history = []
        self.class_distribution = {}
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测logits (N, C) 或 (N,)
            targets: 真实标签 (N,) 或 (N, C)
            
        Returns:
            损失值
        """
        # 确保alpha在正确的设备上
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # 处理二分类和多分类
        if inputs.dim() == 1 or inputs.size(1) == 1:
            # 二分类情况
            return self._binary_focal_loss(inputs, targets)
        else:
            # 多分类情况
            return self._multi_class_focal_loss(inputs, targets)
    
    def _binary_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """二分类Focal Loss"""
        # 确保targets是长整型
        targets = targets.long()
        
        # 计算概率
        if inputs.dim() == 1:
            # 单输出
            probs = torch.sigmoid(inputs)
            log_probs = F.logsigmoid(inputs)
        else:
            # 双输出
            probs = F.softmax(inputs, dim=1)
            log_probs = F.log_softmax(inputs, dim=1)
            probs = probs[:, 1]  # 取正类概率
            log_probs = log_probs[:, 1]  # 取正类log概率
        
        # 计算alpha
        alpha = self._get_adaptive_alpha(targets, probs)
        
        # 计算gamma
        gamma = self._get_adaptive_gamma(targets, probs)
        
        # 计算focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 添加数值稳定性检查
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # 限制gamma的范围，避免数值溢出
        gamma = torch.clamp(gamma, min=0.1, max=10.0)
        
        # 计算focal weight，添加数值稳定性
        one_minus_p_t = 1 - p_t
        one_minus_p_t = torch.clamp(one_minus_p_t, min=1e-8, max=1.0 - 1e-8)
        
        # 使用log空间计算避免数值溢出
        log_one_minus_p_t = torch.log(one_minus_p_t)
        log_focal_weight = torch.log(alpha) + gamma * log_one_minus_p_t
        
        # 限制log_focal_weight的范围
        log_focal_weight = torch.clamp(log_focal_weight, min=-10.0, max=10.0)
        focal_weight = torch.exp(log_focal_weight)
        
        # 检查focal_weight是否包含nan或inf
        if torch.isnan(focal_weight).any() or torch.isinf(focal_weight).any():
            self.logger.warning("检测到focal_weight包含nan或inf，使用标准交叉熵损失")
            focal_weight = torch.ones_like(ce_loss)
        
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.class_weights is not None:
            class_weight = self.class_weights[targets]
            focal_loss = focal_loss * class_weight
        
        # 归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _multi_class_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """多分类Focal Loss"""
        targets = targets.long()
        
        # 计算概率和log概率
        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 获取目标类别的概率
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # 计算alpha
        alpha = self._get_adaptive_alpha(targets, p_t)
        
        # 计算gamma
        gamma = self._get_adaptive_gamma(targets, p_t)
        
        # 计算focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 添加数值稳定性检查
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # 限制gamma的范围，避免数值溢出
        gamma = torch.clamp(gamma, min=0.1, max=10.0)
        
        # 计算focal weight，添加数值稳定性
        one_minus_p_t = 1 - p_t
        one_minus_p_t = torch.clamp(one_minus_p_t, min=1e-8, max=1.0 - 1e-8)
        
        # 使用log空间计算避免数值溢出
        log_one_minus_p_t = torch.log(one_minus_p_t)
        log_focal_weight = torch.log(alpha) + gamma * log_one_minus_p_t
        
        # 限制log_focal_weight的范围
        log_focal_weight = torch.clamp(log_focal_weight, min=-10.0, max=10.0)
        focal_weight = torch.exp(log_focal_weight)
        
        # 检查focal_weight是否包含nan或inf
        if torch.isnan(focal_weight).any() or torch.isinf(focal_weight).any():
            self.logger.warning("检测到focal_weight包含nan或inf，使用标准交叉熵损失")
            focal_weight = torch.ones_like(ce_loss)
        
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.class_weights is not None:
            # 确保class_weights和targets在同一设备上
            if self.class_weights.device != targets.device:
                self.class_weights = self.class_weights.to(targets.device)
            class_weight = self.class_weights[targets]
            focal_loss = focal_loss * class_weight
        
        # 归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _get_adaptive_alpha(self, targets: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """获取自适应alpha"""
        if not self.adaptive_alpha:
            return self.alpha
        
        # 基于类别分布调整alpha
        unique_classes, counts = torch.unique(targets, return_counts=True)
        total_samples = len(targets)
        
        # 计算类别频率
        class_freqs = counts.float() / total_samples
        
        # 计算自适应alpha（稀有类别获得更高的alpha）
        adaptive_alpha = 1.0 - class_freqs
        adaptive_alpha = torch.clamp(adaptive_alpha, self.min_alpha, self.max_alpha)
        
        # 映射到目标类别
        alpha_map = torch.zeros_like(targets, dtype=torch.float32)
        for i, cls in enumerate(unique_classes):
            mask = targets == cls
            alpha_map[mask] = adaptive_alpha[i]
        
        return alpha_map
    
    def _get_adaptive_gamma(self, targets: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """获取自适应gamma"""
        if not self.adaptive_gamma:
            return self.gamma
        
        # 基于预测难度调整gamma
        # 预测越困难（概率接近0.5），gamma越大
        difficulty = 1.0 - torch.abs(probs - 0.5) * 2  # 0到1之间
        adaptive_gamma = self.min_gamma + difficulty * (self.max_gamma - self.min_gamma)
        
        return adaptive_gamma
    
    def update_class_weights(self, class_weights: torch.Tensor):
        """更新类别权重"""
        self.class_weights = class_weights.to(self.alpha.device)
        self.logger.info(f"更新类别权重: {class_weights}")
    
    def get_class_weights_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """从标签计算类别权重"""
        unique_classes, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        # 计算类别权重（逆频率）
        class_weights = total_samples / (len(unique_classes) * counts.float())
        
        # 归一化
        class_weights = class_weights / class_weights.sum() * len(unique_classes)
        
        return class_weights
    
    def compute_difficulty_aware_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算难度感知损失"""
        # 计算基础focal loss
        focal_loss = self.forward(inputs, targets)
        
        # 计算预测难度
        if inputs.dim() == 1 or inputs.size(1) == 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)
            probs = probs.max(dim=1)[0]
        
        # 难度权重（预测越不确定，权重越高）
        uncertainty = 1.0 - torch.abs(probs - 0.5) * 2
        difficulty_weight = uncertainty.mean()
        
        # 调整损失
        adjusted_loss = focal_loss * (1.0 + difficulty_weight)
        
        return adjusted_loss
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """获取损失统计信息"""
        if not self.loss_history:
            return {}
        
        losses = torch.tensor(self.loss_history)
        return {
            'mean_loss': losses.mean().item(),
            'std_loss': losses.std().item(),
            'min_loss': losses.min().item(),
            'max_loss': losses.max().item(),
            'recent_trend': losses[-10:].mean().item() if len(losses) >= 10 else losses.mean().item()
        }


class MultiTaskFocalLoss(nn.Module):
    """
    多任务Focal Loss
    
    支持多个任务的联合训练
    """
    
    def __init__(self, 
                 task_configs: List[Dict],
                 task_weights: Optional[List[float]] = None):
        """
        初始化多任务Focal Loss
        
        Args:
            task_configs: 每个任务的配置
            task_weights: 任务权重
        """
        super(MultiTaskFocalLoss, self).__init__()
        
        self.task_configs = task_configs
        self.task_weights = task_weights or [1.0] * len(task_configs)
        
        # 为每个任务创建Focal Loss
        self.focal_losses = nn.ModuleList([
            ImprovedFocalLoss(**config) for config in task_configs
        ])
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 每个任务的预测logits
            targets: 每个任务的真实标签
            
        Returns:
            总损失
        """
        total_loss = 0.0
        
        for i, (focal_loss, input_tensor, target_tensor) in enumerate(zip(self.focal_losses, inputs, targets)):
            task_loss = focal_loss(input_tensor, target_tensor)
            weighted_loss = task_loss * self.task_weights[i]
            total_loss += weighted_loss
            
            self.logger.debug(f"任务 {i} 损失: {task_loss.item():.4f}, 加权损失: {weighted_loss.item():.4f}")
        
        return total_loss
    
    def update_task_weights(self, task_weights: List[float]):
        """更新任务权重"""
        self.task_weights = task_weights
        self.logger.info(f"更新任务权重: {task_weights}")


class AdaptiveFocalLoss(ImprovedFocalLoss):
    """
    自适应Focal Loss
    
    根据训练过程动态调整参数
    """
    
    def __init__(self, *args, **kwargs):
        super(AdaptiveFocalLoss, self).__init__(*args, **kwargs)
        
        # 训练统计
        self.epoch = 0
        self.batch_count = 0
        self.class_accuracy_history = {}
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播，包含自适应调整"""
        # 计算损失
        loss = super().forward(inputs, targets)
        
        # 更新统计信息
        self.batch_count += 1
        self._update_class_statistics(inputs, targets)
        
        # 每100个batch调整一次参数
        if self.batch_count % 100 == 0:
            self._adaptive_adjustment()
        
        return loss
    
    def _update_class_statistics(self, inputs: torch.Tensor, targets: torch.Tensor):
        """更新类别统计信息"""
        with torch.no_grad():
            if inputs.dim() == 1 or inputs.size(1) == 1:
                preds = (torch.sigmoid(inputs) > 0.5).long()
            else:
                preds = torch.argmax(inputs, dim=1)
            
            # 计算每个类别的准确率
            unique_classes = torch.unique(targets)
            for cls in unique_classes:
                mask = targets == cls
                if mask.sum() > 0:
                    accuracy = (preds[mask] == targets[mask]).float().mean().item()
                    
                    if cls.item() not in self.class_accuracy_history:
                        self.class_accuracy_history[cls.item()] = []
                    
                    self.class_accuracy_history[cls.item()].append(accuracy)
    
    def _adaptive_adjustment(self):
        """自适应调整参数"""
        if not self.class_accuracy_history:
            return
        
        # 计算每个类别的平均准确率
        class_accuracies = {}
        for cls, acc_history in self.class_accuracy_history.items():
            if len(acc_history) >= 10:  # 至少10个样本
                class_accuracies[cls] = np.mean(acc_history[-10:])  # 最近10个的平均
        
        if not class_accuracies:
            return
        
        # 调整alpha：准确率低的类别增加alpha
        if self.adaptive_alpha:
            with torch.no_grad():
                for cls, accuracy in class_accuracies.items():
                    if accuracy < 0.5:  # 准确率低于50%
                        # 增加该类别的alpha
                        current_alpha = self.alpha[cls] if self.alpha.dim() > 0 else self.alpha
                        new_alpha = min(current_alpha * 1.1, self.max_alpha)
                        
                        if self.alpha.dim() > 0:
                            self.alpha[cls] = new_alpha
                        else:
                            self.alpha = new_alpha
        
        # 调整gamma：整体准确率低时增加gamma
        if self.adaptive_gamma:
            overall_accuracy = np.mean(list(class_accuracies.values()))
            if overall_accuracy < 0.6:  # 整体准确率低于60%
                with torch.no_grad():
                    new_gamma = min(self.gamma * 1.05, self.max_gamma)
                    self.gamma = new_gamma
        
        self.logger.info(f"自适应调整完成 - 类别准确率: {class_accuracies}, 当前gamma: {self.gamma.item():.3f}")
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.epoch = epoch
        self.batch_count = 0
        self.class_accuracy_history.clear()
