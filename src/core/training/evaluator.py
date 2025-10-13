"""
模型评估器

实现T-HGNN模型的评估功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
import json

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, config):
        """
        初始化模型评估器
        
        Args:
            model: T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 评估指标 - 重构版本，重点关注类别不平衡
        self.metrics = {
            # 核心指标（优先级最高）
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'average_precision': 0.0,
            'balanced_accuracy': 0.0,
            'mcc': 0.0,
            
            # 基础指标
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            
            # 高级指标
            'g_means': 0.0,
            'f0.5_score': 0.0,
            'f2_score': 0.0,
            
            # 类别分布
            'positive_ratio': 0.0,
            'positive_samples': 0,
            'negative_samples': 0,
            'total_samples': 0,
        }
        
        # 评估结果
        self.evaluation_results = {}
        
    def evaluate_imbalanced_model(self, test_data, test_labels=None, 
                                 model_name: str = "Model") -> Dict[str, float]:
        """
        评估模型 - 专门针对类别不平衡场景优化
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签
            model_name: 模型名称
            
        Returns:
            评估指标字典
        """
        self.logger.info(f"开始评估模型: {model_name}")
        
        # 设置模型为评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 获取预测结果
            if isinstance(test_data, list):
                # 时序快照数据
                predictions = self.model.forward_snapshots(test_data)
            else:
                # 单图数据
                predictions = self.model(test_data)
            
            # 处理预测结果
            all_predictions = []
            all_probabilities = []
            all_labels = []
            
            for node_type, pred_tensor in predictions.items():
                if node_type in test_labels:
                    labels = test_labels[node_type]
                    
                    # 获取预测概率
                    if pred_tensor.dim() > 1 and pred_tensor.size(1) > 1:
                        # 多类预测，取正类概率
                        probabilities = torch.softmax(pred_tensor, dim=1)[:, 1]
                        pred_labels = torch.argmax(pred_tensor, dim=1)
                    else:
                        # 二分类预测
                        probabilities = torch.sigmoid(pred_tensor).flatten()
                        pred_labels = (probabilities > 0.5).long()
                    
                    all_predictions.extend(pred_labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            if not all_predictions:
                self.logger.warning("没有找到有效的预测结果")
                return self.metrics.copy()
            
            # 转换为numpy数组
            y_true = np.array(all_labels)
            y_pred = np.array(all_predictions)
            y_prob = np.array(all_probabilities)
            
            # 使用重构的指标计算器
            from src.evaluation.metrics_calculator import MetricsCalculator
            calculator = MetricsCalculator()
            metrics = calculator.calculate_imbalanced_metrics(y_true, y_pred, y_prob)
            
            # 更新内部指标
            self.metrics.update(metrics)
            
            # 记录评估结果
            self.evaluation_results[model_name] = metrics.copy()
            
            # 输出关键指标
            self.logger.info(f"{model_name} 评估结果 (类别不平衡优化):")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            self.logger.info(f"  AP: {metrics['average_precision']:.4f}")
            self.logger.info(f"  平衡准确率: {metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"  MCC: {metrics['mcc']:.4f}")
            self.logger.info(f"  正样本比例: {metrics['positive_ratio']:.4f}")
            
            return metrics.copy()
    
    def evaluate_model(self, test_data, test_labels=None):
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签（可选）
            
        Returns:
            Dict: 评估结果
        """
        self.logger.info("开始模型评估...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 获取模型预测
            if hasattr(test_data, 'node_types'):
                # 异构图数据
                predictions = self.model.predict(test_data)
                embeddings = self.model.get_embeddings(test_data)
            else:
                # 普通图数据
                predictions = self.model(test_data)
                embeddings = self.model.get_embeddings(test_data)
            
            # 计算评估指标
            if test_labels is not None:
                metrics = self._calculate_metrics(predictions, test_labels)
                self.metrics.update(metrics)
            
            # 生成评估报告
            evaluation_report = self._generate_evaluation_report(
                predictions, embeddings, test_labels
            )
            
            self.evaluation_results = evaluation_report
            
            self.logger.info(f"模型评估完成，准确率: {self.metrics['accuracy']:.4f}")
            
            return evaluation_report
    
    def _calculate_metrics(self, predictions, labels):
        """
        计算评估指标
        
        Args:
            predictions: 模型预测
            labels: 真实标签
            
        Returns:
            Dict: 评估指标
        """
        metrics = {}
        
        try:
            # 转换为numpy数组
            if isinstance(predictions, dict):
                # 异构图预测
                pred_values = []
                label_values = []
                
                for node_type in predictions:
                    if node_type in labels:
                        pred_values.extend(predictions[node_type].cpu().numpy())
                        label_values.extend(labels[node_type].cpu().numpy())
                
                pred_array = np.array(pred_values)
                label_array = np.array(label_values)
            else:
                # 普通图预测
                pred_array = predictions.cpu().numpy()
                label_array = labels.cpu().numpy()
            
            # 计算准确率
            if pred_array.ndim > 1:
                pred_labels = np.argmax(pred_array, axis=1)
            else:
                pred_labels = (pred_array > 0.5).astype(int)
            
            accuracy = np.mean(pred_labels == label_array)
            metrics['accuracy'] = float(accuracy)
            
            # 计算精确率、召回率、F1分数
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(label_array, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(label_array, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(label_array, pred_labels, average='weighted', zero_division=0)
            
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1_score'] = float(f1)
            
            # 计算AUC
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            try:
                if pred_array.ndim > 1:
                    auc_roc = roc_auc_score(label_array, pred_array[:, 1])
                    auc_pr = average_precision_score(label_array, pred_array[:, 1])
                else:
                    auc_roc = roc_auc_score(label_array, pred_array)
                    auc_pr = average_precision_score(label_array, pred_array)
                
                metrics['auc_roc'] = float(auc_roc)
                metrics['auc_pr'] = float(auc_pr)
            except ValueError:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"计算评估指标时出错: {e}")
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'auc_pr': 0.0
            }
        
        return metrics
    
    def _generate_evaluation_report(self, predictions, embeddings, labels=None):
        """
        生成评估报告
        
        Args:
            predictions: 模型预测
            embeddings: 节点嵌入
            labels: 真实标签
            
        Returns:
            Dict: 评估报告
        """
        report = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'model_type': 'T-HGNN',
                'evaluation_mode': 'comprehensive'
            },
            'performance_metrics': self.metrics.copy(),
            'prediction_statistics': {},
            'embedding_statistics': {},
            'data_quality': {}
        }
        
        # 预测统计
        if isinstance(predictions, dict):
            for node_type, pred in predictions.items():
                if hasattr(pred, 'cpu'):
                    pred_np = pred.cpu().numpy()
                else:
                    pred_np = pred
                
                report['prediction_statistics'][node_type] = {
                    'mean': float(np.mean(pred_np)),
                    'std': float(np.std(pred_np)),
                    'min': float(np.min(pred_np)),
                    'max': float(np.max(pred_np)),
                    'shape': list(pred_np.shape)
                }
        else:
            if hasattr(predictions, 'cpu'):
                pred_np = predictions.cpu().numpy()
            else:
                pred_np = predictions
            
            report['prediction_statistics']['overall'] = {
                'mean': float(np.mean(pred_np)),
                'std': float(np.std(pred_np)),
                'min': float(np.min(pred_np)),
                'max': float(np.max(pred_np)),
                'shape': list(pred_np.shape)
            }
        
        # 嵌入统计
        if isinstance(embeddings, dict):
            for node_type, emb in embeddings.items():
                if hasattr(emb, 'cpu'):
                    emb_np = emb.cpu().numpy()
                else:
                    emb_np = emb
                
                report['embedding_statistics'][node_type] = {
                    'mean': float(np.mean(emb_np)),
                    'std': float(np.std(emb_np)),
                    'min': float(np.min(emb_np)),
                    'max': float(np.max(emb_np)),
                    'shape': list(emb_np.shape)
                }
        else:
            if hasattr(embeddings, 'cpu'):
                emb_np = embeddings.cpu().numpy()
            else:
                emb_np = embeddings
            
            report['embedding_statistics']['overall'] = {
                'mean': float(np.mean(emb_np)),
                'std': float(np.std(emb_np)),
                'min': float(np.min(emb_np)),
                'max': float(np.max(emb_np)),
                'shape': list(emb_np.shape)
            }
        
        # 数据质量评估
        if labels is not None:
            if isinstance(labels, dict):
                for node_type, label in labels.items():
                    if hasattr(label, 'cpu'):
                        label_np = label.cpu().numpy()
                    else:
                        label_np = label
                    
                    report['data_quality'][node_type] = {
                        'label_distribution': {
                            'unique_values': int(len(np.unique(label_np))),
                            'class_balance': float(np.mean(label_np))
                        }
                    }
            else:
                if hasattr(labels, 'cpu'):
                    label_np = labels.cpu().numpy()
                else:
                    label_np = labels
                
                report['data_quality']['overall'] = {
                    'label_distribution': {
                        'unique_values': int(len(np.unique(label_np))),
                        'class_balance': float(np.mean(label_np))
                    }
                }
        
        return report
    
    def save_evaluation_results(self, output_path):
        """
        保存评估结果
        
        Args:
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估结果已保存到: {output_path}")
    
    def get_model_performance(self):
        """
        获取模型性能指标
        
        Returns:
            Dict: 性能指标
        """
        return self.metrics.copy()
    
    def get_evaluation_summary(self):
        """
        获取评估摘要
        
        Returns:
            Dict: 评估摘要
        """
        return {
            'performance_metrics': self.metrics.copy(),
            'evaluation_status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
