"""
节点分类器

实现大纲中提到的节点分类器
通过多层感知机（MLP）将节点的表征向量映射为恶意/正常的概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging


def _get_config_value(config, key, default_value):
    """安全地获取配置值"""
    value = getattr(config, key, default_value)
    if hasattr(config, 'model'):
        value = getattr(config.model, key, value)
    return value


class NodeClassifier(nn.Module):
    """
    节点分类器
    
    实现大纲中提到的节点分类器
    通过多层感知机（MLP）将节点的表征向量映射为恶意/正常的概率
    """
    
    def __init__(self, config, node_types: List[str], 
                 in_dims: Dict[str, int], num_classes: int = 2):
        """
        初始化节点分类器
        
        Args:
            config: 模型配置
            node_types: 节点类型列表
            in_dims: 输入维度字典
            num_classes: 分类数量（默认2：正常/恶意）
        """
        super(NodeClassifier, self).__init__()
        self.config = config
        self.node_types = node_types
        self.in_dims = in_dims
        self.num_classes = num_classes
        
        self.logger = logging.getLogger(__name__)
        
        # 为每种节点类型创建分类器
        self.classifiers = nn.ModuleDict()
        
        for ntype in node_types:
            if ntype in in_dims:
                in_dim = in_dims[ntype]
            else:
                in_dim = _get_config_value(config, 'hidden_dim', 64)  # 从128减少到64
                self.logger.warning(f"节点类型 {ntype} 的输入维度未指定，使用默认值 {in_dim}")
            
            # 构建MLP分类器
            self.classifiers[ntype] = self._build_classifier(in_dim, num_classes)
        
        # 共享分类器（用于未知节点类型）
        hidden_dim = _get_config_value(config, 'hidden_dim', 64)  # 从128减少到64
        self.shared_classifier = self._build_classifier(hidden_dim, num_classes)
        
        # 分类器融合层（可选）
        if len(node_types) > 1:
            self.fusion_layer = nn.Linear(
                hidden_dim * len(node_types),
                hidden_dim
            )
        else:
            self.fusion_layer = None
        
        self.logger.info(f"节点分类器初始化完成，节点类型: {node_types}")
        self.logger.info(f"分类数量: {num_classes}")
        self.logger.info(f"输入维度: {in_dims}")
    
    def _build_classifier(self, in_dim: int, num_classes: int) -> nn.Module:
        """
        构建MLP分类器
        
        Args:
            in_dim: 输入维度
            num_classes: 分类数量
            
        Returns:
            MLP分类器
        """
        # 根据配置确定隐藏层大小 - 优化：减少隐藏层大小
        hidden_dims = getattr(self.config, 'classifier_hidden_dims', [128, 64])  # 从[512, 256, 128]减少到[128, 64]
        
        # 构建MLP层
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(_get_config_value(self.config, 'dropout', 0.5))  # 从0.3增加到0.5
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_embeddings: 节点嵌入字典
            
        Returns:
            分类结果字典
        """
        predictions = {}
        
        for ntype, embeddings in node_embeddings.items():
            if ntype in self.classifiers:
                # 使用对应的分类器
                classifier = self.classifiers[ntype]
            else:
                # 使用共享分类器
                classifier = self.shared_classifier
                self.logger.warning(f"节点类型 {ntype} 使用共享分类器")
            
            # 分类
            pred = classifier(embeddings)
            predictions[ntype] = pred
        
        return predictions
    
    def predict_proba(self, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测概率
        
        Args:
            node_embeddings: 节点嵌入字典
            
        Returns:
            概率预测字典
        """
        # 获取logits
        logits = self.forward(node_embeddings)
        
        # 转换为概率
        probabilities = {}
        for ntype, logit in logits.items():
            if self.num_classes == 2:
                # 二分类：使用sigmoid
                prob = torch.sigmoid(logit)
            else:
                # 多分类：使用softmax
                prob = F.softmax(logit, dim=-1)
            
            probabilities[ntype] = prob
        
        return probabilities
    
    def predict_classes(self, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测类别
        
        Args:
            node_embeddings: 节点嵌入字典
            
        Returns:
            类别预测字典
        """
        # 获取logits
        logits = self.forward(node_embeddings)
        
        # 预测类别
        predictions = {}
        for ntype, logit in logits.items():
            if self.num_classes == 2:
                # 二分类：使用阈值0.5
                pred = (logit > 0).long()
            else:
                # 多分类：使用argmax
                pred = torch.argmax(logit, dim=-1)
            
            predictions[ntype] = pred
        
        return predictions
    
    def get_classification_confidence(self, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取分类置信度
        
        Args:
            node_embeddings: 节点嵌入字典
            
        Returns:
            置信度字典
        """
        # 获取概率
        probabilities = self.predict_proba(node_embeddings)
        
        # 计算置信度
        confidences = {}
        for ntype, prob in probabilities.items():
            if self.num_classes == 2:
                # 二分类：使用最大概率
                confidence = torch.max(prob, dim=-1)[0]
            else:
                # 多分类：使用最大概率
                confidence = torch.max(prob, dim=-1)[0]
            
            confidences[ntype] = confidence
        
        return confidences
    
    def get_feature_importance(self, node_embeddings: Dict[str, torch.Tensor], 
                              node_type: str) -> Dict[str, torch.Tensor]:
        """
        获取特征重要性
        
        Args:
            node_embeddings: 节点嵌入字典
            node_type: 节点类型
            
        Returns:
            特征重要性字典
        """
        if node_type not in node_embeddings:
            return {}
        
        embeddings = node_embeddings[node_type]
        
        # 计算梯度
        embeddings.requires_grad_(True)
        
        # 前向传播
        if node_type in self.classifiers:
            classifier = self.classifiers[node_type]
        else:
            classifier = self.shared_classifier
        
        pred = classifier(embeddings)
        
        # 计算梯度
        if pred.dim() > 1:
            # 多分类：使用最大logit的梯度
            max_logit = torch.max(pred, dim=-1)[0]
            grad = torch.autograd.grad(max_logit.sum(), embeddings, retain_graph=True)[0]
        else:
            # 二分类：使用logit的梯度
            grad = torch.autograd.grad(pred.sum(), embeddings, retain_graph=True)[0]
        
        # 计算特征重要性（梯度的绝对值）
        feature_importance = torch.abs(grad)
        
        return {
            'feature_importance': feature_importance,
            'gradient_norm': torch.norm(grad, dim=-1),
            'max_importance': torch.max(feature_importance, dim=-1)[0]
        }
    
    def explain_prediction(self, node_embeddings: Dict[str, torch.Tensor], 
                          node_id: str, node_type: str) -> Dict[str, Any]:
        """
        解释预测结果
        
        Args:
            node_embeddings: 节点嵌入字典
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            解释结果
        """
        if node_type not in node_embeddings:
            return {
                'error': f'节点类型 {node_type} 不存在',
                'node_id': node_id,
                'node_type': node_type
            }
        
        embeddings = node_embeddings[node_type]
        
        # 获取预测结果
        predictions = self.forward(node_embeddings)
        probabilities = self.predict_proba(node_embeddings)
        confidences = self.get_classification_confidence(node_embeddings)
        
        # 获取特征重要性
        feature_importance = self.get_feature_importance(node_embeddings, node_type)
        
        # 构建解释
        explanation = {
            'node_id': node_id,
            'node_type': node_type,
            'prediction': predictions[node_type],
            'probability': probabilities[node_type],
            'confidence': confidences[node_type],
            'feature_importance': feature_importance,
            'explanation_text': self._generate_explanation_text(
                node_id, node_type, predictions[node_type], 
                probabilities[node_type], confidences[node_type]
            )
        }
        
        return explanation
    
    def _generate_explanation_text(self, node_id: str, node_type: str, 
                                  prediction: torch.Tensor, 
                                  probability: torch.Tensor, 
                                  confidence: torch.Tensor) -> str:
        """
        生成解释文本
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            prediction: 预测结果
            probability: 概率
            confidence: 置信度
            
        Returns:
            解释文本
        """
        if self.num_classes == 2:
            # 二分类
            pred_class = "恶意" if prediction.float().mean().item() > 0 else "正常"
            prob_value = probability.mean().item()
            conf_value = confidence.mean().item()
            
            return f"节点 {node_id} ({node_type}) 被分类为 {pred_class}，概率为 {prob_value:.3f}，置信度为 {conf_value:.3f}"
        else:
            # 多分类
            pred_class = prediction.float().mean().item()
            prob_value = probability.mean().item()
            conf_value = confidence.mean().item()
            
            return f"节点 {node_id} ({node_type}) 被分类为类别 {pred_class}，概率为 {prob_value:.3f}，置信度为 {conf_value:.3f}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        classifier_info = {}
        for ntype, classifier in self.classifiers.items():
            classifier_info[ntype] = {
                'parameters': sum(p.numel() for p in classifier.parameters()),
                'layers': len(list(classifier.children()))
            }
        
        return {
            'model_type': 'NodeClassifier',
            'node_types': self.node_types,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'classifier_info': classifier_info,
            'has_fusion_layer': self.fusion_layer is not None
        }
    
    def save_classifier(self, filepath: str):
        """
        保存分类器
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'node_types': self.node_types,
            'in_dims': self.in_dims,
            'num_classes': self.num_classes
        }, filepath)
        
        self.logger.info(f"节点分类器已保存到: {filepath}")
    
    def load_classifier(self, filepath: str):
        """
        加载分类器
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"节点分类器已从 {filepath} 加载")
