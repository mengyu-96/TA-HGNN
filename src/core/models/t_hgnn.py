"""
T-HGNN主模型

实现大纲中提到的核心算法：时序异质图神经网络
整合异质图编码器、时序编码模块和节点分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GCNConv, GATConv
    from torch_geometric.nn import TransformerConv
except ImportError:
    HeteroData = None
    HeteroConv = None
    GCNConv = None
    GATConv = None
    TransformerConv = None

from src.core.models.hgnn_encoder import HGNNEncoder
from src.core.models.temporal_encoder import TemporalEncoder
from src.core.models.node_classifier import NodeClassifier


def _get_config_value(config, key, default_value):
    """安全地获取配置值"""
    value = getattr(config, key, default_value)
    if hasattr(config, 'model'):
        value = getattr(config.model, key, value)
    return value


class T_HGNN(nn.Module):
    """
    时序异质图神经网络主模型
    
    实现大纲中的核心算法：
    1. 异质图编码器：理解语义，采用HGT或HGAT等模型
    2. 时序编码模块：理解因果，通过RPE或时序注意力
    3. 节点分类器：输出概率，通过MLP映射为恶意/正常概率
    """
    
    def __init__(self, config, node_types: List[str], 
                 edge_types: List[Tuple[str, str, str]], 
                 in_dims: Dict[str, int]):
        """
        初始化T-HGNN模型
        
        Args:
            config: 模型配置
            node_types: 节点类型列表
            edge_types: 边类型列表
            in_dims: 输入维度字典 {node_type: dimension}
        """
        super(T_HGNN, self).__init__()
        self.config = config
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dims = in_dims
        
        self.logger = logging.getLogger(__name__)
        
        # 输入特征投影层 - 优化：减少隐藏维度
        hidden_dim = _get_config_value(config, 'hidden_dim', 64)  # 从128减少到64
        self.hidden_dim = hidden_dim  # 保存隐藏维度供其他模块使用
        self.input_projections = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in in_dims.items()
        })
        
        # 异质图编码器
        self.hgnn_encoder = HGNNEncoder(
            config=config,
            node_types=node_types,
            edge_types=edge_types,
            in_dims={ntype: hidden_dim for ntype in node_types}
        )
        
        # 时序编码模块 - 优化：减少时序维度
        temporal_dim = _get_config_value(config, 'temporal_dim', 32)  # 从64减少到32
        if hasattr(config, 'model'):
            temporal_dim = _get_config_value(config, 'temporal_embedding_dim', temporal_dim)
        self.temporal_encoder = TemporalEncoder(config)
        
        # 融合层 - 优化：添加正则化
        # 注意：实际融合时，异质图输出维度可能不是hidden_dim，需要动态计算
        self.fusion_layers = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim + temporal_dim, hidden_dim),  # 异质图输出 + 时序输出
                nn.BatchNorm1d(hidden_dim),  # 添加批归一化
                nn.ReLU(),
                nn.Dropout(0.3)  # 添加dropout
            )
            for ntype in node_types
        })
        
        # 动态融合层，用于处理不同维度的输入
        self.dynamic_fusion = nn.Sequential(
            nn.Linear(hidden_dim + temporal_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 节点分类器
        self.node_classifier = NodeClassifier(
            config=config,
            node_types=node_types,
            in_dims={ntype: hidden_dim for ntype in node_types}
        )
        
        # Dropout层
        dropout = _get_config_value(config, 'dropout', 0.3)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
        self.logger.info(f"T-HGNN模型初始化完成")
        self.logger.info(f"节点类型: {node_types}")
        self.logger.info(f"边类型: {edge_types}")
        self.logger.info(f"隐藏维度: {hidden_dim}")
        self.logger.info(f"时序维度: {temporal_dim}")
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward_snapshots(self, snapshots: List[HeteroData]) -> Dict[str, torch.Tensor]:
        """
        处理时序快照序列
        
        Args:
            snapshots: 时序快照列表
            
        Returns:
            每个节点类型的预测结果
        """
        if not snapshots:
            raise ValueError("快照列表不能为空")
        
        self.logger.info(f"处理 {len(snapshots)} 个时序快照")
        
        # 处理每个快照：先进行异质图编码，保留结构特征
        hgnn_snapshot_embeddings = []
        for i, snapshot in enumerate(snapshots):
            self.logger.debug(f"处理快照 {i+1}/{len(snapshots)}")

            # 为当前快照构造输入特征并投影到隐藏维度
            projected_features = {}
            for ntype in self.node_types:
                if ntype in snapshot.node_types and getattr(snapshot[ntype], 'x', None) is not None:
                    feat = snapshot[ntype].x
                    projected = self.input_projections[ntype](feat)
                    projected = F.relu(projected)
                    projected = self.dropout(projected)
                    projected_features[ntype] = projected
                else:
                    # 为没有特征的节点创建零特征
                    num_nodes = snapshot[ntype].num_nodes if ntype in snapshot.node_types else 0
                    if num_nodes > 0:
                        device = next(self.parameters()).device
                        zero_feat = torch.zeros(num_nodes, self.hidden_dim, device=device)
                        projected_features[ntype] = zero_feat

            # 异质图编码，得到结构嵌入
            hgnn_embeddings = self.hgnn_encoder(projected_features, snapshot)
            hgnn_snapshot_embeddings.append(hgnn_embeddings)
        
        # 使用时序编码器处理快照序列
        final_embeddings = {}
        for ntype in snapshots[0].node_types:
            # 提取该节点类型在所有快照中的结构特征
            ntype_features = []
            ntype_timestamps = []

            for i in range(len(snapshots)):
                hgnn_emb = hgnn_snapshot_embeddings[i]
                if ntype in hgnn_emb:
                    ntype_features.append(hgnn_emb[ntype])
                    # 提取时间戳信息（按快照）
                    if hasattr(snapshots[i], 'timestamps') and snapshots[i].timestamps is not None:
                        ntype_timestamps.append(snapshots[i].timestamps)
                    else:
                        ntype_timestamps.append(None)

            if ntype_features:
                # 使用时序编码器融合跨快照的特征
                final_embeddings[ntype] = self.temporal_encoder.forward_snapshots(
                    ntype_features, ntype_timestamps
                )

        # 结构特征与时序特征融合后进行节点分类
        predictions = {}
        fused_embeddings = {}
        for ntype in final_embeddings:
            # 使用最后一个快照的结构嵌入与时序嵌入融合
            if ntype in hgnn_snapshot_embeddings[-1]:
                combined = torch.cat([
                    hgnn_snapshot_embeddings[-1][ntype],
                    final_embeddings[ntype]
                ], dim=-1)
                actual_dim = combined.size(-1)
                expected_dim = self.hidden_dim + self.temporal_encoder.temporal_dim
                if actual_dim != expected_dim:
                    temp_fusion = nn.Sequential(
                        nn.Linear(actual_dim, self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ).to(combined.device)
                    fused_embeddings[ntype] = temp_fusion(combined)
                else:
                    fused_embeddings[ntype] = self.fusion_layers[ntype](combined)

        for ntype in fused_embeddings:
            predictions[ntype] = self.node_classifier(fused_embeddings[ntype])
        
        return predictions
    
    def forward(self, data: HeteroData, 
                timestamps: Optional[Dict[str, torch.Tensor]] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: 异构图数据
            timestamps: 时间戳字典 {node_type: timestamps}
            return_embeddings: 是否返回节点嵌入
            
        Returns:
            如果return_embeddings=True，返回节点嵌入字典
            否则返回分类logits字典
        """
        # 1. 输入特征投影
        projected_features = {}
        for ntype in self.node_types:
            if ntype in data.node_types and data[ntype].x is not None:
                feat = data[ntype].x
                projected_features[ntype] = self.input_projections[ntype](feat)
                projected_features[ntype] = F.relu(projected_features[ntype])
                projected_features[ntype] = self.dropout(projected_features[ntype])
            else:
                # 为没有特征的节点创建零特征
                num_nodes = data[ntype].num_nodes if ntype in data.node_types else 0
                if num_nodes > 0:
                    device = next(self.parameters()).device
                    zero_feat = torch.zeros(num_nodes, self.hidden_dim, device=device)
                    projected_features[ntype] = zero_feat
        
        # 2. 异质图编码
        hgnn_embeddings = self.hgnn_encoder(projected_features, data)
        
        # 3. 时序编码
        temporal_embeddings = {}
        temporal_dim = self.temporal_encoder.temporal_dim
        
        if timestamps is not None:
            for ntype in self.node_types:
                if ntype in hgnn_embeddings and ntype in timestamps:
                    temporal_emb = self.temporal_encoder(
                        hgnn_embeddings[ntype], 
                        timestamps[ntype]
                    )
                    temporal_embeddings[ntype] = temporal_emb
                else:
                    # 如果没有时序信息，使用零填充
                    if ntype in hgnn_embeddings:
                        device = hgnn_embeddings[ntype].device
                        temporal_embeddings[ntype] = torch.zeros(
                            hgnn_embeddings[ntype].size(0), 
                            temporal_dim, 
                            device=device
                        )
        else:
            # 如果没有时序信息，使用零填充
            for ntype in self.node_types:
                if ntype in hgnn_embeddings:
                    device = hgnn_embeddings[ntype].device
                    temporal_embeddings[ntype] = torch.zeros(
                        hgnn_embeddings[ntype].size(0), 
                        temporal_dim, 
                        device=device
                    )
        
        # 4. 特征融合
        fused_embeddings = {}
        for ntype in self.node_types:
            if ntype in hgnn_embeddings and ntype in temporal_embeddings:
                # 拼接异质图嵌入和时序嵌入
                combined = torch.cat([
                    hgnn_embeddings[ntype], 
                    temporal_embeddings[ntype]
                ], dim=-1)
                
                # 通过融合层（现在包含正则化）
                # 动态调整融合层以适应实际维度
                actual_dim = combined.size(-1)
                if actual_dim != (self.hidden_dim + self.temporal_encoder.temporal_dim):  # 如果维度不匹配
                    # 创建临时融合层
                    temp_fusion = nn.Sequential(
                        nn.Linear(actual_dim, self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ).to(combined.device)
                    fused_embeddings[ntype] = temp_fusion(combined)
                else:
                    fused_embeddings[ntype] = self.fusion_layers[ntype](combined)
        
        # 5. 节点分类
        if return_embeddings:
            return fused_embeddings
        else:
            predictions = self.node_classifier(fused_embeddings)
            return predictions
    
    def get_node_embeddings(self, data: HeteroData, 
                           timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        获取节点嵌入
        
        Args:
            data: 异构图数据
            timestamps: 时间戳字典
            
        Returns:
            节点嵌入字典
        """
        return self.forward(data, timestamps, return_embeddings=True)
    
    def predict(self, data: HeteroData, 
                timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        预测节点类别
        
        Args:
            data: 异构图数据
            timestamps: 时间戳字典
            
        Returns:
            预测结果字典
        """
        return self.forward(data, timestamps, return_embeddings=False)
    
    def compute_loss(self, data: HeteroData, 
                     labels: Dict[str, torch.Tensor], 
                     masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算损失
        
        Args:
            data: 异构图数据
            labels: 标签字典
            masks: 掩码字典
            
        Returns:
            损失值
        """
        # 获取预测结果
        predictions = self.forward(data, return_embeddings=False)
        
        # 计算损失
        total_loss = 0.0
        num_losses = 0
        
        for ntype in self.node_types:
            if ntype in predictions and ntype in labels and ntype in masks:
                pred = predictions[ntype]
                label = labels[ntype]
                mask = masks[ntype]
                
                if pred.size(0) > 0 and label.size(0) > 0 and mask.size(0) > 0:
                    # 确保维度匹配
                    min_size = min(pred.size(0), label.size(0), mask.size(0))
                    pred = pred[:min_size]
                    label = label[:min_size]
                    mask = mask[:min_size]
                    
                    # 计算交叉熵损失
                    loss = F.cross_entropy(pred, label, reduction='none')
                    
                    # 应用掩码
                    masked_loss = loss * mask.float()
                    
                    # 计算平均损失
                    if mask.sum() > 0:
                        avg_loss = masked_loss.sum() / mask.sum()
                        total_loss += avg_loss
                        num_losses += 1
        
        # 返回平均损失
        if num_losses > 0:
            return total_loss / num_losses
        else:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'T-HGNN',
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'hidden_dim': getattr(self.config, 'hidden_dim', 128),
            'temporal_dim': getattr(self.config, 'temporal_dim', 64),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hgnn_encoder_params': sum(p.numel() for p in self.hgnn_encoder.parameters()),
            'temporal_encoder_params': sum(p.numel() for p in self.temporal_encoder.parameters()),
            'node_classifier_params': sum(p.numel() for p in self.node_classifier.parameters())
        }
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        # 只保存模型状态字典，不保存配置对象
        torch.save({
            'model_state_dict': self.state_dict(),
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            'in_dims': self.in_dims
        }, filepath)
        
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def explain_prediction(self, data: HeteroData, 
                          node_id: str, 
                          node_type: str,
                          timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        解释预测结果
        
        Args:
            data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            timestamps: 时间戳字典
            
        Returns:
            解释结果
        """
        # 获取节点嵌入
        embeddings = self.get_node_embeddings(data, timestamps)
        
        # 获取预测结果
        predictions = self.predict(data, timestamps)
        
        # 获取注意力权重（如果可用）
        attention_weights = {}
        if hasattr(self.hgnn_encoder, 'get_attention_weights'):
            attention_weights = self.hgnn_encoder.get_attention_weights()
        
        # 构建解释
        explanation = {
            'node_id': node_id,
            'node_type': node_type,
            'prediction': predictions.get(node_type, None),
            'embedding': embeddings.get(node_type, None),
            'attention_weights': attention_weights.get(node_type, None),
            'model_confidence': self._calculate_model_confidence(predictions, node_type),
            'explanation_text': self._generate_explanation_text(node_id, node_type, predictions)
        }
        
        return explanation
    
    def _calculate_model_confidence(self, predictions: Dict[str, torch.Tensor], 
                                   node_type: str) -> float:
        """
        计算模型置信度
        
        Args:
            predictions: 预测结果
            node_type: 节点类型
            
        Returns:
            置信度分数
        """
        if node_type not in predictions:
            return 0.0
        
        pred = predictions[node_type]
        if pred.dim() > 1:
            # 多分类情况，使用最大概率
            confidence = torch.max(F.softmax(pred, dim=-1), dim=-1)[0]
            return confidence.mean().item()
        else:
            # 二分类情况，使用sigmoid概率
            confidence = torch.sigmoid(pred)
            return confidence.mean().item()
    
    def _generate_explanation_text(self, node_id: str, node_type: str, 
                                  predictions: Dict[str, torch.Tensor]) -> str:
        """
        生成解释文本
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            predictions: 预测结果
            
        Returns:
            解释文本
        """
        if node_type not in predictions:
            return f"节点 {node_id} ({node_type}) 没有预测结果"
        
        pred = predictions[node_type]
        if pred.dim() > 1:
            # 多分类情况
            class_probs = F.softmax(pred, dim=-1)
            predicted_class = torch.argmax(class_probs, dim=-1)
            confidence = torch.max(class_probs, dim=-1)[0]
            
            return f"节点 {node_id} ({node_type}) 被预测为类别 {predicted_class.float().mean().item():.0f}，置信度为 {confidence.mean().item():.3f}"
        else:
            # 二分类情况
            prob = torch.sigmoid(pred)
            predicted_class = (prob > 0.5).int()
            
            return f"节点 {node_id} ({node_type}) 被预测为 {'恶意' if predicted_class.float().mean().item() > 0.5 else '正常'}，概率为 {prob.mean().item():.3f}"
    
    def get_embeddings(self, data, timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        获取节点嵌入
        
        Args:
            data: 输入图数据
            timestamps: 时间戳字典
            
        Returns:
            节点嵌入字典
        """
        return self.get_node_embeddings(data, timestamps)
