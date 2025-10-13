"""
异质图编码器

实现大纲中提到的异质图编码器（HGNN）
采用HGT或HGAT等模型，尊重不同类型节点和关系的差异，进行信息传播与聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import math

try:
    from torch_geometric.nn import HeteroConv, GCNConv, GATConv, TransformerConv
    from torch_geometric.nn import HeteroLinear, HeteroDictLinear
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
except ImportError:
    HeteroConv = None
    GCNConv = None
    GATConv = None
    TransformerConv = None
    HeteroLinear = None
    HeteroDictLinear = None
    MessagePassing = None
    softmax = None


def _get_config_value(config, key, default_value):
    """安全地获取配置值"""
    value = getattr(config, key, default_value)
    if hasattr(config, 'model'):
        value = getattr(config.model, key, value)
    return value


class HeteroGATLayer(nn.Module):
    """
    异质图注意力层
    
    实现异质图上的注意力机制，考虑不同节点类型和边类型的差异
    """
    
    def __init__(self, in_channels, out_channels: int, 
                 num_heads: int = 8, dropout: float = 0.0):
        """
        初始化异质图注意力层
        
        Args:
            in_channels: 输入通道数（可以是int或Dict[str, int]）
            out_channels: 输出通道数
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(HeteroGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 线性变换层 - 支持不同节点类型的不同输入维度
        if isinstance(in_channels, dict):
            self.W = nn.ModuleDict({
                ntype: nn.Linear(in_dim, out_channels * num_heads, bias=False)
                for ntype, in_dim in in_channels.items()
            })
        else:
            self.W = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        self.a = nn.Parameter(torch.empty(1, num_heads, 2 * out_channels))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        if isinstance(self.W, nn.ModuleDict):
            for layer in self.W.values():
                nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典
            
        Returns:
            输出特征字典
        """
        # 线性变换
        h_dict = {}
        for node_type, x in x_dict.items():
            if isinstance(self.W, nn.ModuleDict):
                # 使用对应节点类型的线性层
                h_dict[node_type] = self.W[node_type](x).view(-1, self.num_heads, self.out_channels)
            else:
                # 使用统一的线性层
                h_dict[node_type] = self.W(x).view(-1, self.num_heads, self.out_channels)
        
        # 计算注意力
        out_dict = {}
        for node_type in x_dict.keys():
            out_dict[node_type] = []
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            if src_type in h_dict and dst_type in h_dict:
                src_h = h_dict[src_type]
                dst_h = h_dict[dst_type]
                
                # 计算注意力分数
                src_indices = edge_index[0]
                dst_indices = edge_index[1]
                
                src_features = src_h[src_indices]  # [num_edges, num_heads, out_channels]
                dst_features = dst_h[dst_indices]  # [num_edges, num_heads, out_channels]
                
                # 拼接特征
                concat_features = torch.cat([src_features, dst_features], dim=-1)  # [num_edges, num_heads, 2*out_channels]
                
                # 计算注意力分数
                attention_scores = (concat_features * self.a).sum(dim=-1)  # [num_edges, num_heads]
                attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
                
                # 应用softmax
                attention_weights = softmax(attention_scores, dst_indices, num_nodes=dst_h.size(0))
                
                # 应用dropout
                if self.training and self.dropout > 0:
                    attention_weights = F.dropout(attention_weights, p=self.dropout, training=True)
                
                # 聚合邻居信息
                weighted_features = src_features * attention_weights.unsqueeze(-1)
                
                # 按目标节点聚合
                out_features = torch.zeros(dst_h.size(0), self.num_heads, self.out_channels, 
                                         device=dst_h.device, dtype=dst_h.dtype)
                out_features.scatter_add_(0, dst_indices.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_features), 
                                        weighted_features)
                
                # 存储结果
                if dst_type not in out_dict:
                    out_dict[dst_type] = []
                out_dict[dst_type].append(out_features)
        
        # 合并同一节点类型的多个边类型的结果
        final_out = {}
        for node_type in out_dict.keys():
            if out_dict[node_type]:
                # 平均聚合
                out_features = torch.stack(out_dict[node_type], dim=0).mean(dim=0)
                # 拼接多头输出
                final_out[node_type] = out_features.view(-1, self.num_heads * self.out_channels)
            else:
                # 如果没有边连接到该节点类型，使用零填充
                num_nodes = x_dict[node_type].size(0)
                final_out[node_type] = torch.zeros(num_nodes, self.num_heads * self.out_channels, 
                                                 device=x_dict[node_type].device, dtype=x_dict[node_type].dtype)
        
        return final_out


class HeteroGCNLayer(nn.Module):
    """
    异质图卷积层
    
    实现异质图上的图卷积操作
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0.0):
        """
        初始化异质图卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            dropout: Dropout率
        """
        super(HeteroGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # 线性变换层
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典
            
        Returns:
            输出特征字典
        """
        # 线性变换
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.W(x)
        
        # 图卷积
        out_dict = {}
        for node_type in x_dict.keys():
            out_dict[node_type] = []
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            if src_type in h_dict and dst_type in h_dict:
                src_h = h_dict[src_type]
                dst_h = h_dict[dst_type]
                
                # 获取边索引
                src_indices = edge_index[0]
                dst_indices = edge_index[1]
                
                # 聚合邻居信息
                src_features = src_h[src_indices]  # [num_edges, out_channels]
                
                # 按目标节点聚合
                out_features = torch.zeros(dst_h.size(0), self.out_channels, 
                                         device=dst_h.device, dtype=dst_h.dtype)
                out_features.scatter_add_(0, dst_indices.unsqueeze(-1).expand_as(src_features), 
                                        src_features)
                
                # 存储结果
                if dst_type not in out_dict:
                    out_dict[dst_type] = []
                out_dict[dst_type].append(out_features)
        
        # 合并同一节点类型的多个边类型的结果
        final_out = {}
        for node_type in out_dict.keys():
            if out_dict[node_type]:
                # 平均聚合
                out_features = torch.stack(out_dict[node_type], dim=0).mean(dim=0)
                # 应用激活函数
                final_out[node_type] = F.relu(out_features)
                # 应用dropout
                if self.training and self.dropout > 0:
                    final_out[node_type] = F.dropout(final_out[node_type], p=self.dropout, training=True)
            else:
                # 如果没有边连接到该节点类型，使用零填充
                num_nodes = x_dict[node_type].size(0)
                final_out[node_type] = torch.zeros(num_nodes, self.out_channels, 
                                                 device=x_dict[node_type].device, dtype=x_dict[node_type].dtype)
        
        return final_out


class HGNNEncoder(nn.Module):
    """
    异质图编码器
    
    实现大纲中提到的异质图编码器（HGNN）
    采用HGT或HGAT等模型，尊重不同类型节点和关系的差异，进行信息传播与聚合
    """
    
    def __init__(self, config, node_types: List[str], 
                 edge_types: List[Tuple[str, str, str]], 
                 in_dims: Dict[str, int]):
        """
        初始化异质图编码器
        
        Args:
            config: 模型配置
            node_types: 节点类型列表
            edge_types: 边类型列表
            in_dims: 输入维度字典
        """
        super(HGNNEncoder, self).__init__()
        self.config = config
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dims = in_dims
        
        self.logger = logging.getLogger(__name__)
        
        # 选择编码器类型
        self.encoder_type = getattr(config, 'hgnn_encoder_type', 'gat')  # 'gat', 'gcn', 'transformer'
        
        # 构建编码器层
        self.encoder_layers = nn.ModuleList()
        
        num_layers = _get_config_value(config, 'num_layers', 2)  # 从3层减少到2层
        hidden_dim = _get_config_value(config, 'hidden_dim', 64)  # 从128减少到64
        
        for i in range(num_layers):
            if i == 0:
                in_channels = {ntype: in_dims[ntype] for ntype in node_types}
            else:
                # 对于GAT，第二层及以后的输入维度是第一层的输出维度
                if self.encoder_type == 'gat':
                    gat_output_dim = _get_config_value(config, 'num_heads', 8) * hidden_dim
                    in_channels = {ntype: gat_output_dim for ntype in node_types}
                else:
                    in_channels = {ntype: hidden_dim for ntype in node_types}
            
            if self.encoder_type == 'gat':
                layer = HeteroGATLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    num_heads=_get_config_value(config, 'num_heads', 8),
                    dropout=_get_config_value(config, 'dropout', 0.3)
                )
            elif self.encoder_type == 'gcn':
                layer = HeteroGCNLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    dropout=_get_config_value(config, 'dropout', 0.3)
                )
            else:
                # 默认使用GAT
                layer = HeteroGATLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_heads=_get_config_value(config, 'num_heads', 8),
                    dropout=_get_config_value(config, 'dropout', 0.3)
                )
            
            self.encoder_layers.append(layer)
        
        # 输出投影层 - 根据编码器类型调整维度
        if self.encoder_type == 'gat':
            # GAT输出维度是 num_heads * hidden_dim，需要投影到 hidden_dim
            output_dim = _get_config_value(config, 'num_heads', 8) * hidden_dim
            self.output_projections = nn.ModuleDict({
                ntype: nn.Linear(output_dim, hidden_dim)
                for ntype in node_types
            })
        else:
            self.output_projections = nn.ModuleDict({
                ntype: nn.Linear(hidden_dim, hidden_dim)
                for ntype in node_types
            })
        
        # 层归一化 - 根据编码器类型调整维度
        if self.encoder_type == 'gat':
            # GAT输出维度是 num_heads * hidden_dim
            norm_dim = _get_config_value(config, 'num_heads', 8) * hidden_dim
        else:
            norm_dim = hidden_dim
            
        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(norm_dim)
            for ntype in node_types
        })
        
        # Dropout
        self.dropout = nn.Dropout(_get_config_value(config, 'dropout', 0.3))
        
        # 注意力权重存储
        self.attention_weights = {}
        
        self.logger.info(f"异质图编码器初始化完成，类型: {self.encoder_type}")
        self.logger.info(f"编码器层数: {num_layers}")
        self.logger.info(f"隐藏维度: {hidden_dim}")
        self.logger.info(f"注意力头数: {_get_config_value(config, 'num_heads', 8)}")
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                data) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
            data: 异构图数据
            
        Returns:
            编码后的节点特征字典
        """
        # 构建边索引字典
        edge_index_dict = {}
        for edge_type in data.edge_types:
            if edge_type in data.edge_index_dict:
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # 通过编码器层
        current_features = x_dict
        for i, layer in enumerate(self.encoder_layers):
            # 编码
            encoded_features = layer(current_features, edge_index_dict)
            
            # 残差连接
            if i > 0:
                for ntype in self.node_types:
                    if ntype in current_features and ntype in encoded_features:
                        encoded_features[ntype] = encoded_features[ntype] + current_features[ntype]
            
            # 层归一化
            for ntype in self.node_types:
                if ntype in encoded_features:
                    encoded_features[ntype] = self.layer_norms[ntype](encoded_features[ntype])
            
            # Dropout
            for ntype in self.node_types:
                if ntype in encoded_features:
                    encoded_features[ntype] = self.dropout(encoded_features[ntype])
            
            current_features = encoded_features
        
        # 输出投影
        final_features = {}
        for ntype in self.node_types:
            if ntype in current_features:
                final_features[ntype] = self.output_projections[ntype](current_features[ntype])
                final_features[ntype] = F.relu(final_features[ntype])
                final_features[ntype] = self.dropout(final_features[ntype])
            else:
                # 如果没有特征，创建零特征
                num_nodes = x_dict[ntype].size(0) if ntype in x_dict else 0
                if num_nodes > 0:
                    device = next(self.parameters()).device
                    hidden_dim = _get_config_value(self.config, 'hidden_dim', 64)
                    final_features[ntype] = torch.zeros(num_nodes, hidden_dim, device=device)
        
        return final_features
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取注意力权重
        
        Returns:
            注意力权重字典
        """
        return self.attention_weights
    
    def get_node_embeddings(self, x_dict: Dict[str, torch.Tensor], 
                           data) -> Dict[str, torch.Tensor]:
        """
        获取节点嵌入
        
        Args:
            x_dict: 节点特征字典
            data: 异构图数据
            
        Returns:
            节点嵌入字典
        """
        return self.forward(x_dict, data)
    
    def explain_attention(self, x_dict: Dict[str, torch.Tensor], 
                         data, 
                         node_id: str, 
                         node_type: str) -> Dict[str, Any]:
        """
        解释注意力机制
        
        Args:
            x_dict: 节点特征字典
            data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            注意力解释
        """
        # 获取注意力权重
        attention_weights = self.get_attention_weights()
        
        # 构建解释
        explanation = {
            'node_id': node_id,
            'node_type': node_type,
            'attention_weights': attention_weights.get(node_type, None),
            'explanation_text': f"节点 {node_id} ({node_type}) 的注意力权重分析"
        }
        
        return explanation
