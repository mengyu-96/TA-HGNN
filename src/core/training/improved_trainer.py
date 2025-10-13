"""
改进的模型训练器

解决训练问题，包括：
1. 类别不平衡处理
2. 早停策略优化
3. 学习率调度
4. 梯度裁剪
5. 模型检查点
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..losses.improved_focal_loss import ImprovedFocalLoss, AdaptiveFocalLoss
from .real_data_loader import RealDataLoader, GraphBatchProcessor


class ImprovedModelTrainer:
    """改进的模型训练器"""
    
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
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 20)
        self.min_delta = getattr(config.training, 'min_delta', 1e-4)
        
        # 梯度裁剪
        self.max_grad_norm = getattr(config.training, 'max_grad_norm', 1.0)
        
        # 模型检查点
        self.checkpoint_dir = os.path.join(config.data.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 数据加载器
        self.data_loader = RealDataLoader(config)
        self.batch_processor = GraphBatchProcessor(self.device)
        
    def _setup_loss_functions(self):
        """设置损失函数"""
        self.logger.info("设置损失函数")
        
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
                alpha=0.25,
                gamma=2.0,
                class_weights=class_weight,
                adaptive_alpha=True,
                adaptive_gamma=True
            )
        
        self.logger.info(f"为 {len(self.loss_functions)} 个节点类型创建了损失函数")
    
    def _setup_optimizer(self):
        """设置优化器"""
        self.logger.info("设置优化器")
        
        # 获取学习率
        learning_rate = getattr(self.config.training, 'learning_rate', 0.001)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=getattr(self.config.training, 'weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"优化器设置完成，学习率: {learning_rate}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.logger.info("设置学习率调度器")
        
        scheduler_type = getattr(self.config.training, 'scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.config.training, 'epochs', 100),
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"学习率调度器: {scheduler_type}")
    
    def train(self, train_graphs: List, val_graphs: List, 
              train_labels: Dict, val_labels: Dict) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_graphs: 训练图列表
            val_graphs: 验证图列表
            train_labels: 训练标签字典
            val_labels: 验证标签字典
            
        Returns:
            训练报告
        """
        self.logger.info("开始模型训练")
        
        # 训练参数
        epochs = getattr(self.config.training, 'epochs', 100)
        batch_size = getattr(self.config.training, 'batch_size', 64)
        
        # 训练循环
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self._train_epoch(train_graphs, train_labels, batch_size)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_graphs, val_labels, batch_size)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self._update_training_history(train_metrics, val_metrics)
            
            # 打印进度
            self._print_epoch_progress(epoch, train_metrics, val_metrics)
            
            # 早停检查
            if self._check_early_stopping(val_metrics):
                self.logger.info(f"早停在第 {epoch + 1} 轮")
                break
            
            # 保存最佳模型
            self._save_best_model(val_metrics)
        
        # 生成训练报告
        training_report = self._generate_training_report()
        
        self.logger.info("模型训练完成")
        return training_report
    
    def _train_epoch(self, train_graphs: List, train_labels: Dict, batch_size: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # 创建数据加载器
        train_loader = self._create_data_loader(train_graphs, train_labels, batch_size, shuffle=True)
        
        for batch_graphs, batch_labels in train_loader:
            # 处理批次数据
            batch_graphs, batch_labels = self.batch_processor.process_batch(batch_graphs, batch_labels)
            # 前向传播 - 使用第一个图进行预测
            if batch_graphs:
                predictions = self.model(batch_graphs[0])
            else:
                continue
            
            # 计算损失
            batch_loss = self._compute_batch_loss(predictions, batch_labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # 计算准确率
            batch_accuracy = self._compute_batch_accuracy(predictions, batch_labels)
            
            # 更新统计
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
            # 创建数据加载器
            val_loader = self._create_data_loader(val_graphs, val_labels, batch_size, shuffle=False)
            
            for batch_graphs, batch_labels in val_loader:
                # 处理批次数据
                batch_graphs, batch_labels = self.batch_processor.process_batch(batch_graphs, batch_labels)
                # 前向传播 - 使用第一个图进行预测
                if batch_graphs:
                    predictions = self.model(batch_graphs[0])
                else:
                    continue
                
                # 计算损失
                batch_loss = self._compute_batch_loss(predictions, batch_labels)
                
                # 计算准确率
                batch_accuracy = self._compute_batch_accuracy(predictions, batch_labels)
                
                # 更新统计
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0
        }
    
    def _create_data_loader(self, graphs: List, labels: Dict, batch_size: int, shuffle: bool = False):
        """创建数据加载器"""
        return self.data_loader.create_dataloader(graphs, labels, shuffle)
    
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
                    # 确保损失函数在正确设备上
                    if hasattr(loss_fn, 'to'):
                        loss_fn = loss_fn.to(pred.device)
                    
                    loss = loss_fn(pred, label)
                    
                    total_loss += loss
                    num_losses += 1
        
        return total_loss / num_losses if num_losses > 0 else torch.tensor(0.0, requires_grad=True)
    
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
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
    
    def _print_epoch_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印epoch进度"""
        lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f"Epoch {epoch + 1}/{getattr(self.config.training, 'epochs', 100)}, "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"LR: {lr:.6f}"
        )
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """检查早停条件"""
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        
        # 检查验证损失是否改善
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 检查验证准确率是否改善
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
        
        # 早停检查
        if self.patience_counter >= self.early_stopping_patience:
            return True
        
        return False
    
    def _save_best_model(self, val_metrics: Dict):
        """保存最佳模型"""
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        
        # 如果验证损失改善，保存模型
        if val_loss < self.best_val_loss:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'training_history': self.training_history
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"保存最佳模型 (Epoch {self.current_epoch})")
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'total_epochs': self.current_epoch + 1,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_history': self.training_history,
            'early_stopped': self.patience_counter >= self.early_stopping_patience
        }
    
    def evaluate_model_performance(self, test_graphs: List, test_labels: Dict, 
                                 ground_truth_paths: List) -> Dict[str, Any]:
        """评估模型性能"""
        self.logger.info("开始模型性能评估")
        
        # 加载最佳模型
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("已加载最佳模型")
        
        self.model.eval()
        
        # 评估结果
        evaluation_results = {
            'node_classification': {},
            'attack_grouping': {},
            'path_tracing': {}
        }
        
        # 节点分类评估
        evaluation_results['node_classification'] = self._evaluate_node_classification(test_graphs, test_labels)
        
        # 攻击分组评估
        evaluation_results['attack_grouping'] = self._evaluate_attack_grouping(test_graphs)
        
        # 路径追踪评估
        evaluation_results['path_tracing'] = self._evaluate_path_tracing(test_graphs, ground_truth_paths)
        
        return evaluation_results
    
    def _evaluate_node_classification(self, test_graphs: List, test_labels: Dict) -> Dict[str, Any]:
        """评估节点分类性能"""
        from ...evaluation.node_classification_evaluator import NodeClassificationEvaluator
        
        evaluator = NodeClassificationEvaluator()
        
        # 获取预测结果
        all_predictions = {}
        all_labels = {}
        
        with torch.no_grad():
            for graph in test_graphs:
                predictions = self.model(graph)
                
                for node_type in self.model.node_types:
                    if node_type in predictions and node_type in test_labels:
                        if node_type not in all_predictions:
                            all_predictions[node_type] = []
                            all_labels[node_type] = []
                        
                        all_predictions[node_type].append(predictions[node_type])
                        all_labels[node_type].append(test_labels[node_type])
        
        # 合并预测结果
        for node_type in all_predictions:
            all_predictions[node_type] = torch.cat(all_predictions[node_type], dim=0)
            all_labels[node_type] = torch.cat(all_labels[node_type], dim=0)
        
        # 评估每个节点类型
        results = {}
        for node_type in all_predictions:
            pred = all_predictions[node_type]
            label = all_labels[node_type]
            
            # 确保预测和标签在同一设备上
            if pred.device != label.device:
                label = label.to(pred.device)
            
            # 计算预测类别
            if pred.dim() == 1 or pred.size(1) == 1:
                pred_classes = (torch.sigmoid(pred) > 0.5).long()
                pred_probs = torch.sigmoid(pred)
            else:
                pred_classes = torch.argmax(pred, dim=1)
                pred_probs = torch.softmax(pred, dim=1)
            
            # 评估
            result = evaluator.evaluate_single_model(
                pred_classes, label, pred_probs, f"{node_type}_model"
            )
            results[node_type] = result
        
        return results
    
    def _evaluate_attack_grouping(self, test_graphs: List) -> Dict[str, Any]:
        """评估攻击分组性能"""
        from ...evaluation.attack_grouping_evaluator import AttackGroupingEvaluator
        
        evaluator = AttackGroupingEvaluator()
        
        # 获取节点嵌入
        all_embeddings = {}
        
        with torch.no_grad():
            for graph in test_graphs:
                embeddings = self.model.get_embeddings(graph)
                
                for node_type in embeddings:
                    if node_type not in all_embeddings:
                        all_embeddings[node_type] = []
                    all_embeddings[node_type].append(embeddings[node_type])
        
        # 合并嵌入
        for node_type in all_embeddings:
            all_embeddings[node_type] = torch.cat(all_embeddings[node_type], dim=0)
        
        # 评估每个节点类型
        results = {}
        for node_type in all_embeddings:
            embeddings = all_embeddings[node_type]
            result = evaluator.evaluate_clustering_performance(embeddings, node_type)
            results[node_type] = result
        
        return results
    
    def _evaluate_path_tracing(self, test_graphs: List, ground_truth_paths: List) -> Dict[str, Any]:
        """评估路径追踪性能"""
        from ...evaluation.path_tracing_evaluator import PathTracingEvaluator
        
        evaluator = PathTracingEvaluator()
        
        # 简化的路径追踪评估
        # 在实际应用中，这里应该实现完整的路径追踪逻辑
        
        results = {
            'success_rate': 0.0,
            'path_similarity': 0.0,
            'average_precision': 0.0,
            'precision_at_k': {
                'P@1': 0.0,
                'P@3': 0.0,
                'P@5': 0.0,
                'P@10': 0.0
            },
            'predicted_paths': 0,
            'ground_truth_paths': len(ground_truth_paths)
        }
        
        return results
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # 损失和准确率对比
        ax2 = axes[1, 1].twinx()
        line1 = axes[1, 1].plot(self.training_history['val_loss'], 'b-', label='Val Loss')
        line2 = ax2.plot(self.training_history['val_acc'], 'r-', label='Val Acc')
        axes[1, 1].set_title('Validation Loss vs Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy', color='r')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 1].legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self) -> str:
        """生成评估报告"""
        report = f"""
================================================================================
T-HGNN模型评估报告
================================================================================

训练统计:
- 最佳验证损失: {self.best_val_loss:.4f}
- 最佳验证准确率: {self.best_val_accuracy:.4f}
- 总训练轮数: {self.current_epoch + 1}
- 最终学习率: {self.optimizer.param_groups[0]['lr']:.6f}
- 是否早停: {'是' if self.patience_counter >= self.early_stopping_patience else '否'}

模型配置:
- 节点类型: {self.model.node_types}
- 边类型: {len(self.model.edge_types)}
- 隐藏维度: {getattr(self.config.model, 'hidden_dim', 'N/A')}
- 时序维度: {getattr(self.config.model, 'temporal_dim', 'N/A')}

损失函数配置:
- 使用Focal Loss: 是
- 自适应参数: 是
- 类别权重: {'是' if self.class_weights else '否'}

================================================================================
        """
        
        return report
    
    def save_evaluation_results(self, filepath: str):
        """保存评估结果"""
        results = {
            'training_report': self._generate_training_report(),
            'model_info': self.model.get_model_info(),
            'config': {
                'hidden_dim': getattr(self.config.model, 'hidden_dim', None),
                'temporal_dim': getattr(self.config.model, 'temporal_dim', None),
                'learning_rate': getattr(self.config.training, 'learning_rate', None),
                'epochs': getattr(self.config.training, 'epochs', None),
                'batch_size': getattr(self.config.training, 'batch_size', None)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {filepath}")
