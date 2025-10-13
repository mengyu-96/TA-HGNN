"""
性能优化训练器

专门用于解决模型性能问题的训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..losses.improved_focal_loss import ImprovedFocalLoss, AdaptiveFocalLoss
from .real_data_loader import RealDataLoader, GraphBatchProcessor


class PerformanceOptimizedTrainer:
    """性能优化训练器"""
    
    def __init__(self, model: nn.Module, config, class_weights: Optional[Dict] = None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 配置对象
            class_weights: 类别权重字典
        """
        self.model = model
        self.config = config
        self.class_weights = class_weights
        self.logger = logging.getLogger(__name__)
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 混合精度训练
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 损失函数
        self._setup_loss_functions()
        
        # 优化器
        self._setup_optimizer()
        
        # 学习率调度器
        self._setup_scheduler()
        
        # 早停配置
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 50)
        self.min_delta = getattr(config.training, 'min_delta', 1e-5)
        
        # 梯度裁剪
        self.max_grad_norm = getattr(config.training, 'max_grad_norm', 0.5)
        
        # 模型检查点
        self.checkpoint_dir = os.path.join(config.data.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 数据加载器
        self.data_loader = RealDataLoader(config)
        self.batch_processor = GraphBatchProcessor(self.device)
        
        # 模型集成
        self.ensemble_models = []
        self.ensemble_size = getattr(config.training, 'ensemble_size', 3)
        
    def _setup_loss_functions(self):
        """设置损失函数"""
        self.logger.info("设置性能优化损失函数")
        
        # 为每个节点类型创建损失函数
        self.loss_functions = {}
        
        for node_type in self.model.node_types:
            # 获取该类别的权重
            if self.class_weights and node_type in self.class_weights:
                class_weight = torch.tensor(self.class_weights[node_type], dtype=torch.float32)
            else:
                class_weight = None
            
            # 创建自适应Focal Loss
            self.loss_functions[node_type] = AdaptiveFocalLoss(
                alpha=getattr(self.config.training, 'focal_alpha', 0.25),
                gamma=getattr(self.config.training, 'focal_gamma', 2.0),
                class_weights=class_weight,
                label_smoothing=getattr(self.config.training, 'label_smoothing', 0.1),
                reduction='mean'
            )
        
        self.logger.info(f"损失函数初始化完成，支持 {len(self.loss_functions)} 种节点类型")
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.logger.info("设置性能优化优化器")
        
        # 使用AdamW优化器
        optimizer_name = getattr(self.config.training, 'optimizer', 'adamw')
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(
                    getattr(self.config.training, 'beta1', 0.9),
                    getattr(self.config.training, 'beta2', 0.999)
                ),
                eps=getattr(self.config.training, 'eps', 1e-8)
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        
        self.logger.info(f"优化器设置完成: {optimizer_name}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.logger.info("设置学习率调度器")
        
        scheduler_type = getattr(self.config.training, 'lr_scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif scheduler_type == 'cosine_with_warmup':
            # 自定义余弦退火调度器
            self.scheduler = self._create_cosine_with_warmup_scheduler()
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=getattr(self.config.training, 'lr_decay_factor', 0.1),
                patience=getattr(self.config.training, 'lr_patience', 15),
                min_lr=self.config.training.learning_rate * 0.001
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                epochs=self.config.training.epochs,
                steps_per_epoch=100  # 假设每个epoch有100个步骤
            )
        else:
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1
            )
        
        self.logger.info(f"学习率调度器设置完成: {scheduler_type}")
    
    def _create_cosine_with_warmup_scheduler(self):
        """创建带预热的余弦退火调度器"""
        class CosineWithWarmupLR:
            def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
                self.optimizer = optimizer
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.base_lr = base_lr
                self.min_lr = min_lr
                self.current_epoch = 0
            
            def step(self):
                self.current_epoch += 1
                
                if self.current_epoch <= self.warmup_epochs:
                    # 预热阶段
                    lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
                else:
                    # 余弦退火阶段
                    progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                    lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]
        
        return CosineWithWarmupLR(
            self.optimizer,
            warmup_epochs=getattr(self.config.training, 'warmup_epochs', 10),
            total_epochs=self.config.training.epochs,
            base_lr=self.config.training.learning_rate,
            min_lr=self.config.training.learning_rate * 0.01
        )
    
    def train(self, train_graphs: List, train_labels: Dict, 
              val_graphs: List, val_labels: Dict, 
              batch_size: int = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_graphs: 训练图列表
            train_labels: 训练标签字典
            val_graphs: 验证图列表
            val_labels: 验证标签字典
            batch_size: 批次大小
            
        Returns:
            训练报告
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        self.logger.info("开始性能优化模型训练")
        
        # 交叉验证训练
        if getattr(self.config.training, 'use_cross_validation', False):
            return self._cross_validation_train(train_graphs, train_labels, val_graphs, val_labels, batch_size)
        
        # 标准训练
        return self._standard_train(train_graphs, train_labels, val_graphs, val_labels, batch_size)
    
    def _standard_train(self, train_graphs: List, train_labels: Dict, 
                       val_graphs: List, val_labels: Dict, batch_size: int) -> Dict[str, Any]:
        """标准训练流程"""
        self.logger.info("开始标准训练流程")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_graphs, train_labels, batch_size)
            
            # 验证
            if epoch % getattr(self.config.training, 'validation_frequency', 5) == 0:
                val_metrics = self._validate_epoch(val_graphs, val_labels, batch_size)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # 更新学习率
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # 更新训练历史
            self._update_training_history(train_metrics, val_metrics)
            
            # 打印进度
            self._print_epoch_progress(epoch, train_metrics, val_metrics)
            
            # 早停检查
            if self._check_early_stopping(val_metrics):
                self.logger.info(f"早停在第 {epoch + 1} 轮")
                break
            
            # 保存最佳模型
            self._save_best_model(val_metrics)
            
            # 定期保存检查点
            if epoch % getattr(self.config.training, 'save_frequency', 10) == 0:
                self._save_checkpoint(epoch)
        
        # 生成训练报告
        training_report = self._generate_training_report()
        
        self.logger.info("模型训练完成")
        return training_report
    
    def _cross_validation_train(self, train_graphs: List, train_labels: Dict, 
                               val_graphs: List, val_labels: Dict, batch_size: int) -> Dict[str, Any]:
        """交叉验证训练"""
        self.logger.info("开始交叉验证训练")
        
        # 合并训练和验证数据
        all_graphs = train_graphs + val_graphs
        all_labels = {}
        for node_type in train_labels.keys():
            all_labels[node_type] = torch.cat([
                train_labels[node_type], 
                val_labels[node_type]
            ])
        
        # 创建交叉验证分割
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_graphs, all_labels['alert'])):
            self.logger.info(f"训练第 {fold + 1}/5 折")
            
            # 分割数据
            fold_train_graphs = [all_graphs[i] for i in train_idx]
            fold_val_graphs = [all_graphs[i] for i in val_idx]
            
            fold_train_labels = {}
            fold_val_labels = {}
            for node_type in all_labels.keys():
                fold_train_labels[node_type] = all_labels[node_type][train_idx]
                fold_val_labels[node_type] = all_labels[node_type][val_idx]
            
            # 训练当前折
            fold_result = self._standard_train(
                fold_train_graphs, fold_train_labels,
                fold_val_graphs, fold_val_labels,
                batch_size
            )
            
            fold_results.append(fold_result)
            
            # 保存当前折的模型
            self.ensemble_models.append(self.model.state_dict().copy())
        
        # 计算平均结果
        avg_results = self._compute_average_results(fold_results)
        
        self.logger.info("交叉验证训练完成")
        return avg_results
    
    def _train_epoch(self, train_graphs: List, train_labels: Dict, batch_size: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # 创建数据加载器
        train_loader = self.data_loader.create_dataloader(train_graphs, train_labels, shuffle=True)
        
        for batch_graphs, batch_labels in train_loader:
            # 处理批次数据
            batch_graphs, batch_labels = self.batch_processor.process_batch(batch_graphs, batch_labels)
            
            if not batch_graphs:
                continue
            
            # 前向传播
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(batch_graphs[0])
                    batch_loss = self._compute_batch_loss(predictions, batch_labels)
            else:
                predictions = self.model(batch_graphs[0])
                batch_loss = self._compute_batch_loss(predictions, batch_labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                batch_loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # 计算准确率
            batch_accuracy = self._compute_batch_accuracy(predictions, batch_labels)
            
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0
        }
    
    def _validate_epoch(self, val_graphs: List, val_labels: Dict, batch_size: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            val_loader = self.data_loader.create_dataloader(val_graphs, val_labels, shuffle=False)
            
            for batch_graphs, batch_labels in val_loader:
                batch_graphs, batch_labels = self.batch_processor.process_batch(batch_graphs, batch_labels)
                
                if not batch_graphs:
                    continue
                
                # 前向传播
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(batch_graphs[0])
                        batch_loss = self._compute_batch_loss(predictions, batch_labels)
                else:
                    predictions = self.model(batch_graphs[0])
                    batch_loss = self._compute_batch_loss(predictions, batch_labels)
                
                # 计算准确率
                batch_accuracy = self._compute_batch_accuracy(predictions, batch_labels)
                
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0
        }
    
    def _compute_batch_loss(self, predictions: Dict, labels: Dict) -> torch.Tensor:
        """计算批次损失"""
        total_loss = 0.0
        num_losses = 0
        
        for node_type in self.model.node_types:
            if node_type in predictions and node_type in labels:
                pred = predictions[node_type]
                label = labels[node_type]
                
                if pred.size(0) > 0 and label.size(0) > 0:
                    # 确保维度匹配
                    min_size = min(pred.size(0), label.size(0))
                    pred = pred[:min_size]
                    label = label[:min_size]
                    
                    # 确保预测和标签在同一设备上
                    if pred.device != label.device:
                        label = label.to(pred.device)
                    
                    # 确保标签是长整型
                    if label.dtype != torch.long:
                        label = label.long()
                    
                    # 确保标签维度正确
                    if label.dim() > 1:
                        label = label.squeeze()
                    
                    # 计算损失
                    loss_fn = self.loss_functions[node_type]
                    if hasattr(loss_fn, 'to'):
                        loss_fn = loss_fn.to(pred.device)
                    
                    loss = loss_fn(pred, label)
                    total_loss += loss
                    num_losses += 1
        
        return total_loss / num_losses if num_losses > 0 else torch.tensor(0.0, device=self.device)
    
    def _compute_batch_accuracy(self, predictions: Dict, labels: Dict) -> float:
        """计算批次准确率"""
        total_correct = 0
        total_samples = 0
        
        for node_type in self.model.node_types:
            if node_type in predictions and node_type in labels:
                pred = predictions[node_type]
                label = labels[node_type]
                
                if pred.size(0) > 0 and label.size(0) > 0:
                    # 确保维度匹配
                    min_size = min(pred.size(0), label.size(0))
                    pred = pred[:min_size]
                    label = label[:min_size]
                    
                    # 确保预测和标签在同一设备上
                    if pred.device != label.device:
                        label = label.to(pred.device)
                    
                    # 确保标签是长整型
                    if label.dtype != torch.long:
                        label = label.long()
                    
                    # 确保标签维度正确
                    if label.dim() > 1:
                        label = label.squeeze()
                    
                    # 计算预测类别
                    if pred.dim() == 1 or pred.size(1) == 1:
                        pred_classes = (torch.sigmoid(pred) > 0.5).long()
                    else:
                        pred_classes = torch.argmax(pred, dim=1)
                    
                    # 计算准确率
                    correct = (pred_classes == label).sum().item()
                    total_correct += correct
                    total_samples += min_size
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def _update_training_history(self, train_metrics: Dict, val_metrics: Dict):
        """更新训练历史"""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['val_acc'].append(val_metrics['accuracy'])
        
        # 获取当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['learning_rate'].append(current_lr)
    
    def _print_epoch_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印训练进度"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.training.epochs}, "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """检查早停条件"""
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        
        # 检查验证损失
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 检查验证准确率
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
        
        return self.patience_counter >= self.early_stopping_patience
    
    def _save_best_model(self, val_metrics: Dict):
        """保存最佳模型"""
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        
        if val_loss < self.best_val_loss or val_accuracy > self.best_val_accuracy:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
                'epoch': self.current_epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'config': self.config
            }, checkpoint_path)
            
            self.logger.info(f"已保存最佳模型 (Epoch {self.current_epoch + 1})")
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'epoch': epoch,
            'training_history': self.training_history,
            'config': self.config
        }, checkpoint_path)
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        return {
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0.0,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0.0,
            'final_train_acc': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0.0,
            'final_val_acc': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0.0,
            'training_history': self.training_history,
            'config': self.config
        }
    
    def _compute_average_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """计算交叉验证平均结果"""
        avg_results = {
            'total_epochs': np.mean([r['total_epochs'] for r in fold_results]),
            'best_val_loss': np.mean([r['best_val_loss'] for r in fold_results]),
            'best_val_accuracy': np.mean([r['best_val_accuracy'] for r in fold_results]),
            'final_train_loss': np.mean([r['final_train_loss'] for r in fold_results]),
            'final_val_loss': np.mean([r['final_val_loss'] for r in fold_results]),
            'final_train_acc': np.mean([r['final_train_acc'] for r in fold_results]),
            'final_val_acc': np.mean([r['final_val_acc'] for r in fold_results]),
            'fold_results': fold_results
        }
        
        return avg_results

