"""
异常检测器

实现基于T-HGNN的高级异常检测功能，包括深度学习异常检测、图结构异常检测和多尺度异常检测
支持多种异常检测算法，能够处理时序异质图数据，并提供可解释的异常检测结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA, TruncatedSVD
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
import os
import time

try:
    from torch_geometric.data import HeteroData, Batch
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
    from torch_geometric.nn import GINConv, TransformerConv, HeteroConv
    from torch_geometric.utils import to_networkx, subgraph, k_hop_subgraph
    from torch_geometric.utils import add_self_loops, remove_self_loops
    from torch_geometric.transforms import ToUndirected
    from torch_geometric.loader import NeighborLoader
except ImportError:
    HeteroData = None
    Batch = None
    GCNConv = None
    GATConv = None
    SAGEConv = None
    GraphConv = None
    GINConv = None
    TransformerConv = None
    HeteroConv = None
    to_networkx = None
    subgraph = None
    k_hop_subgraph = None
    add_self_loops = None
    remove_self_loops = None
    ToUndirected = None
    NeighborLoader = None


class VariationalAutoEncoder(nn.Module):
    """
    变分自编码器
    
    使用VAE架构进行异常检测，通过学习数据的潜在分布来识别异常
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], latent_dim=16, dropout=0.2):
        """
        初始化变分自编码器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            dropout: Dropout比率
        """
        super(VariationalAutoEncoder, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和方差预测
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_layers = []
        decoder_dims = [latent_dim] + list(reversed(hidden_dims))
        prev_dim = decoder_dims[0]
        for i, dim in enumerate(decoder_dims[1:]):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            if i < len(decoder_dims) - 2:  # 不在最后一层添加dropout
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # 输出层
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        
    def encode(self, x):
        """编码过程，返回均值和对数方差"""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧，使得反向传播可行"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """解码过程，从潜在空间重建输入"""
        return self.decoder(z)
    
    def get_anomaly_score(self, x):
        """计算异常分数"""
        self.eval()
        with torch.no_grad():
            x_reconstructed, mu, log_var = self.forward(x)
            recon_error = F.mse_loss(x_reconstructed, x, reduction='none').sum(dim=1)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            anomaly_score = recon_error + kl_divergence
        return anomaly_score
    
    def forward(self, x):
        """前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
    
    def loss_function(self, x_reconstructed, x, mu, log_var):
        """VAE损失函数：重建损失 + KL散度"""
        # 重建损失
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # 总损失
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_anomaly_score(self, x):
        """计算异常分数"""
        self.eval()
        with torch.no_grad():
            x_reconstructed, mu, log_var = self.forward(x)
            recon_error = F.mse_loss(x_reconstructed, x, reduction='none').sum(dim=1)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            anomaly_score = recon_error + kl_divergence
        return anomaly_score
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
    
    def encode(self, x):
        """
        编码过程
        
        Args:
            x: 输入特征
            
        Returns:
            均值和对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        重参数化技巧
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            采样的潜在变量
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        解码过程
        
        Args:
            z: 潜在变量
            
        Returns:
            重构的输入
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            重构特征, 均值, 对数方差
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
    
    def get_anomaly_score(self, x):
        """
        计算异常分数
        
        Args:
            x: 输入特征
            
        Returns:
            异常分数
        """
        x_hat, mu, log_var = self.forward(x)
        
        # 重构误差
        recon_error = F.mse_loss(x_hat, x, reduction='none').sum(dim=1)
        
        # KL散度
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # 总异常分数 = 重构误差 + KL散度
        anomaly_score = recon_error + kl_div
        
        return anomaly_score


class DeepAnomalyDetector(nn.Module):
    """
    深度异常检测模型
    
    使用高级自编码器架构进行异常检测，支持多种损失函数和正则化技术
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2, 
                 activation='leaky_relu', use_attention=True, use_residual=True):
        """
        初始化深度异常检测模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比率
            activation: 激活函数类型
            use_attention: 是否使用自注意力机制
            use_residual: 是否使用残差连接
        """
        super(DeepAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.LeakyReLU(0.2)
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 自注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=4,
                dropout=dropout
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # 解码器
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))
        prev_dim = hidden_dims[-1]
        for i, dim in enumerate(hidden_dims_reversed[1:]):
            if use_residual and i > 0 and prev_dim == dim:
                # 残差块
                decoder_layers.append(ResidualBlock(prev_dim, dim, dropout, self.activation))
            else:
                # 标准层
                decoder_layers.append(nn.Linear(prev_dim, dim))
                decoder_layers.append(nn.BatchNorm1d(dim))
                decoder_layers.append(self.activation)
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # 输出层
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 异常分数预测器
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 原始输入和重构输入的拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            重构特征和异常分数
        """
        # 编码
        encoded = self.encoder(x)
        
        # 应用自注意力（如果启用）
        if self.use_attention:
            # 调整维度以适应注意力层 [batch_size, seq_len=1, hidden_dim]
            encoded_reshaped = encoded.unsqueeze(1)
            attended, _ = self.attention(
                encoded_reshaped, encoded_reshaped, encoded_reshaped
            )
            attended = attended.squeeze(1)
            encoded = encoded + self.attention_norm(attended)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 预测异常分数
        combined = torch.cat([x, decoded], dim=1)
        anomaly_score = self.anomaly_predictor(combined)
        
        return decoded, anomaly_score
    
    def get_reconstruction_error(self, x):
        """
        计算重构误差
        
        Args:
            x: 输入特征
            
        Returns:
            重构误差
        """
        # 前向传播
        x_hat, _ = self.forward(x)
        
        # 计算MSE
        mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        
        return mse
    
    def get_anomaly_score(self, x):
        """
        获取异常分数
        
        Args:
            x: 输入特征
            
        Returns:
            异常分数
        """
        _, anomaly_score = self.forward(x)
        return anomaly_score.squeeze(-1)


class ResidualBlock(nn.Module):
    """
    残差块
    
    实现残差连接，帮助训练更深的网络
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=nn.LeakyReLU(0.2)):
        """
        初始化残差块
        
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            dropout: Dropout比率
            activation: 激活函数
        """
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        # 如果输入输出维度不同，添加投影层
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            输出特征
        """
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class GraphAnomalyDetector(nn.Module):
    """
    图结构异常检测模型
    
    使用高级图神经网络进行异常检测，支持多种图卷积操作和注意力机制
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2,
                 gnn_type='gat', num_heads=4, use_edge_features=False,
                 edge_dim=None, pooling='mean'):
        """
        初始化图异常检测模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout比率
            gnn_type: GNN类型 ('gcn', 'gat', 'sage', 'gin', 'transformer')
            num_heads: 注意力头数 (用于GAT和Transformer)
            use_edge_features: 是否使用边特征
            edge_dim: 边特征维度
            pooling: 池化方法 ('mean', 'max', 'sum', 'attention')
        """
        super(GraphAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.use_edge_features = use_edge_features
        self.pooling = pooling
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN层
        self.convs = nn.ModuleList()
        
        # 第一层
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim if use_edge_features else None))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gin':
            nn_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_model))
        elif gnn_type == 'transformer':
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim if use_edge_features else None))
        else:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 后续层
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim if use_edge_features else None))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gin':
                nn_model = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(nn_model))
            elif gnn_type == 'transformer':
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=edge_dim if use_edge_features else None))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 注意力池化
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # 异常分数预测
        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            batch: 批处理索引 [num_nodes]
            
        Returns:
            节点异常分数和图异常分数
        """
        # 输入投影
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # GNN层
        for i, conv in enumerate(self.convs):
            if self.gnn_type in ['gat', 'transformer'] and self.use_edge_features:
                h_new = conv(h, edge_index, edge_attr)
            else:
                h_new = conv(h, edge_index)
            
            # 残差连接
            if i > 0:
                h = h + h_new
            else:
                h = h_new
            
            # 层归一化
            h = self.layer_norms[i](h)
            
            # 激活和dropout
            h = F.relu(h)
            h = self.dropout(h)
        
        # 节点级异常分数
        node_scores = self.score_predictor(h).squeeze(-1)
        
        # 如果提供了批处理索引，计算图级异常分数
        if batch is not None:
            if self.pooling == 'mean':
                # 平均池化
                graph_h = scatter_mean(h, batch, dim=0)
            elif self.pooling == 'max':
                # 最大池化
                graph_h = scatter_max(h, batch, dim=0)[0]
            elif self.pooling == 'sum':
                # 求和池化
                graph_h = scatter_sum(h, batch, dim=0)
            elif self.pooling == 'attention':
                # 注意力池化
                scores = self.attention_pool(h).squeeze(-1)
                scores = torch.softmax(scores, dim=0)
                graph_h = scatter_sum(h * scores.unsqueeze(-1), batch, dim=0)
            else:
                # 默认使用平均池化
                graph_h = scatter_mean(h, batch, dim=0)
            
            # 图级异常分数
            graph_scores = self.score_predictor(graph_h).squeeze(-1)
            return node_scores, graph_scores
        
        return node_scores
    
    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """
        获取注意力权重
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征
            
        Returns:
            注意力权重列表
        """
        if self.gnn_type not in ['gat', 'transformer']:
            return None
        
        attention_weights = []
        
        # 输入投影
        h = self.input_proj(x)
        h = F.relu(h)
        
        # 获取每层的注意力权重
        for i, conv in enumerate(self.convs):
            if hasattr(conv, '_alpha'):
                if self.use_edge_features:
                    _ = conv(h, edge_index, edge_attr)
                else:
                    _ = conv(h, edge_index)
                attention_weights.append(conv._alpha)
        
        return attention_weights


class MultiScaleAnomalyDetector(nn.Module):
    """
    多尺度异常检测模型
    
    结合不同尺度的特征进行异常检测，包括节点级、子图级和图级特征
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_scales=3, dropout=0.2,
                 gnn_type='gat', num_heads=4, use_edge_features=False, edge_dim=None):
        """
        初始化多尺度异常检测模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_scales: 尺度数量
            dropout: Dropout比率
            gnn_type: GNN类型
            num_heads: 注意力头数
            use_edge_features: 是否使用边特征
            edge_dim: 边特征维度
        """
        super(MultiScaleAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_scales = num_scales
        
        # 节点级特征提取器
        self.node_feature_extractor = DeepAnomalyDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_attention=True
        )
        
        # 图级特征提取器
        self.graph_detectors = nn.ModuleList([
            GraphAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=hidden_dims[0],
                num_layers=i+1,  # 不同层数代表不同感受野
                dropout=dropout,
                gnn_type=gnn_type,
                num_heads=num_heads,
                use_edge_features=use_edge_features,
                edge_dim=edge_dim,
                pooling='attention'
            )
            for i in range(num_scales)
        ])
        
        # 多尺度融合
        fusion_input_dim = hidden_dims[0] * num_scales + hidden_dims[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征
            batch: 批处理索引
            
        Returns:
            多尺度异常分数
        """
        # 节点级特征
        node_recon, node_scores = self.node_feature_extractor(x)
        
        # 不同尺度的图级特征
        scale_features = []
        for detector in self.graph_detectors:
            if batch is not None:
                _, graph_feat = detector(x, edge_index, edge_attr, batch)
                scale_features.append(graph_feat)
            else:
                # 如果没有批处理索引，创建基于节点索引的批处理信息
                dummy_batch = self._create_default_batch_index(x.size(0), x.device)
                _, graph_feat = detector(x, edge_index, edge_attr, dummy_batch)
                scale_features.append(graph_feat)
        
        # 节点级特征聚合
        if batch is not None:
            node_feat = scatter_mean(self.node_feature_extractor.encoder(x), batch, dim=0)
        else:
            dummy_batch = self._create_default_batch_index(x.size(0), x.device)
            node_feat = scatter_mean(self.node_feature_extractor.encoder(x), dummy_batch, dim=0)
        
        # 多尺度融合
        multi_scale_feat = torch.cat([node_feat] + scale_features, dim=1)
        anomaly_score = self.fusion(multi_scale_feat)
        
        return anomaly_score.squeeze(-1)
    
    def _create_default_batch_index(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        创建默认的批处理索引
        
        Args:
            num_nodes: 节点数量
            device: 设备
            
        Returns:
            批处理索引张量
        """
        # 基于节点数量创建合理的批处理索引
        # 按一定数量分组，便于处理
        batch_size = min(32, num_nodes)  # 最大批次大小32
        batch_index = torch.arange(0, (num_nodes // batch_size + 1) * batch_size, 
                                 device=device) % ((num_nodes + batch_size - 1) // batch_size)
        return batch_index[:num_nodes]

class AnomalyDetector:
    """
    异常检测器
    
    实现基于T-HGNN的高级异常检测功能，包括:
    1. 深度学习异常检测: 使用自编码器、变分自编码器等深度学习模型
    2. 图结构异常检测: 利用图结构信息进行异常检测
    3. 多尺度异常检测: 结合不同尺度的特征进行异常检测
    4. 时序异常检测: 考虑时间信息进行异常检测
    5. 集成异常检测: 结合多种异常检测算法
    """
    
    def __init__(self, model, config):
        """
        初始化异常检测器
        
        Args:
            model: 训练好的T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 异常检测参数
        self.anomaly_threshold = getattr(config, 'anomaly_threshold', 0.7)
        self.suspicious_threshold = getattr(config, 'suspicious_threshold', 0.5)
        self.use_deep_detector = getattr(config, 'use_deep_detector', True)
        self.use_graph_detector = getattr(config, 'use_graph_detector', True)
        self.use_multi_scale = getattr(config, 'use_multi_scale', True)
        
        # 获取设备
        self.device = next(model.parameters()).device
        
        # 深度学习异常检测模型
        self.deep_detectors = {}
        
        # 图结构异常检测模型
        self.graph_detectors = {}
        
        # 传统异常检测模型
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(min_samples=5, eps=0.5)  # 修复缺失的HDBSCAN依赖
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
        self.scaler = StandardScaler()
        
        # 多尺度参数
        self.scales = [0.1, 0.5, 1.0]  # 不同时间尺度的权重
        
        # 正常行为基线
        self.normal_baseline = {}
        
        # 历史异常记录
        self.historical_anomalies = {}
        
        # 动态阈值字典
        self.thresholds = {}
        
        # 初始化深度学习检测器
        self._init_deep_detectors()
        
        self.logger.info("异常检测器初始化完成")
    
    def _init_deep_detectors(self):
        """初始化深度学习检测器"""
        if not self.use_deep_detector:
            return
            
        # 为每种节点类型创建深度检测器
        for node_type in self.model.node_types:
            # 获取嵌入维度
            hidden_dim = self.model.hidden_dim
            
            # 创建深度异常检测器
            self.deep_detectors[node_type] = DeepAnomalyDetector(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim//2, hidden_dim//4, hidden_dim//8],
                dropout=0.2
            ).to(self.device)
            
            # 如果启用图结构检测
            if self.use_graph_detector and GCNConv is not None:
                self.graph_detectors[node_type] = GraphAnomalyDetector(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim//2,
                    num_layers=2,
                    dropout=0.2
                ).to(self.device)
        
    def detect_anomalies(self, hetero_data: HeteroData, 
                        embeddings: Dict[str, torch.Tensor],
                        time_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        检测异常节点
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            time_data: 时间相关数据，包含时间戳和时间窗口信息
            
        Returns:
            异常检测结果
        """
        self.logger.info("开始多尺度异常检测")
        
        anomalies = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anomalous_nodes': {},
            'suspicious_nodes': {},
            'anomaly_scores': {},
            'clusters': {},
            'temporal_patterns': {},
            'graph_anomalies': {},
            'multi_scale_scores': {},
            'summary': {}
        }
        
        # 获取当前时间戳
        current_time = datetime.now()
        
        # 对每种节点类型进行异常检测
        for ntype in hetero_data.node_types:
            if ntype in embeddings and hetero_data[ntype].x is not None:
                node_embeddings = embeddings[ntype]
                self.logger.info(f"检测 {ntype} 类型节点的异常...")
                
                # 1. 基于深度学习的异常检测
                deep_scores = self._deep_anomaly_detection(ntype, node_embeddings)
                
                # 2. 基于图结构的异常检测
                graph_scores = self._graph_anomaly_detection(ntype, node_embeddings, hetero_data)
                
                # 3. 基于统计的异常检测
                stat_scores = self._calculate_anomaly_scores(node_embeddings)
                
                # 4. 多尺度异常检测
                if time_data is not None and self.use_multi_scale:
                    multi_scale_scores = self._multi_scale_detection(ntype, node_embeddings, time_data)
                    anomalies['multi_scale_scores'][ntype] = multi_scale_scores.tolist()
                else:
                    multi_scale_scores = torch.zeros_like(stat_scores)
                
                # 融合不同方法的异常分数
                anomaly_scores = self._fuse_anomaly_scores([
                    deep_scores, graph_scores, stat_scores, multi_scale_scores
                ])
                
                # 识别异常节点
                threshold = self.thresholds.get(ntype, self.anomaly_threshold)
                anomalous_indices = torch.where(anomaly_scores > threshold)[0]
                suspicious_indices = torch.where(
                    (anomaly_scores > self.suspicious_threshold) & 
                    (anomaly_scores <= threshold)
                )[0]
                
                if len(anomalous_indices) > 0:
                    # 创建详细的异常节点信息
                    anomalous_nodes = []
                    for i, idx in enumerate(anomalous_indices.tolist()):
                        score = anomaly_scores[anomalous_indices][i].item()
                        
                        # 获取节点属性（如果可用）
                        node_attrs = {}
                        if hasattr(hetero_data[ntype], 'attrs'):
                            for attr_name, attr_values in hetero_data[ntype].attrs.items():
                                if idx < len(attr_values):
                                    node_attrs[attr_name] = attr_values[idx]
                        
                        # 确定攻击阶段（基于异常分数和图结构）
                        attack_stage = self._determine_attack_stage(
                            ntype, idx, score, graph_scores[idx].item(), hetero_data
                        )
                        
                        # 生成描述
                        description = self._generate_anomaly_description(
                            ntype, idx, score, attack_stage, node_attrs
                        )
                        
                        # 获取时间戳
                        timestamp = anomalies['timestamp']
                        if time_data is not None and 'timestamps' in time_data and ntype in time_data['timestamps']:
                            if idx < len(time_data['timestamps'][ntype]):
                                timestamp = time_data['timestamps'][ntype][idx]
                        
                        anomalous_nodes.append({
                            'node_id': f'{ntype}_{idx}',
                            'node_type': ntype,
                            'anomaly_score': float(score),
                            'attack_stage': attack_stage,
                            'description': description,
                            'timestamp': timestamp,
                            'deep_score': float(deep_scores[idx].item()),
                            'graph_score': float(graph_scores[idx].item()),
                            'stat_score': float(stat_scores[idx].item()),
                            'multi_scale_score': float(multi_scale_scores[idx].item()),
                            'attributes': node_attrs
                        })
                    
                    anomalies['anomalous_nodes'][ntype] = anomalous_nodes
                
                if len(suspicious_indices) > 0:
                    anomalies['suspicious_nodes'][ntype] = {
                        'indices': suspicious_indices.tolist(),
                        'scores': anomaly_scores[suspicious_indices].tolist()
                    }
                
                anomalies['anomaly_scores'][ntype] = anomaly_scores.tolist()
                
                # 记录图结构异常
                if graph_scores is not None:
                    graph_anomalous_indices = torch.where(graph_scores > threshold)[0]
                    if len(graph_anomalous_indices) > 0:
                        anomalies['graph_anomalies'][ntype] = {
                            'indices': graph_anomalous_indices.tolist(),
                            'scores': graph_scores[graph_anomalous_indices].tolist()
                        }
        
        # 聚类分析（提供必需的embeddings参数）
        anomalies['clusters'] = self._cluster_anomalies(anomalies, embeddings)
        
        # 时序模式分析
        if time_data is not None:
            anomalies['temporal_patterns'] = self._analyze_temporal_patterns(anomalies, time_data)
        
        # 生成摘要
        anomalies['summary'] = self._generate_anomaly_summary(anomalies)
        
        # 更新历史异常记录
        self._update_historical_anomalies(anomalies)
        
        self.logger.info(f"异常检测完成，发现 {sum(len(nodes) for nodes in anomalies['anomalous_nodes'].values())} 个异常节点")
        
        return anomalies
    
    def _deep_anomaly_detection(self, node_type: str, embeddings: torch.Tensor) -> torch.Tensor:
        """
        使用深度学习模型进行异常检测
        
        Args:
            node_type: 节点类型
            embeddings: 节点嵌入
            
        Returns:
            异常分数
        """
        # 如果未启用深度检测或没有对应的检测器，返回零分数
        if not self.use_deep_detector or node_type not in self.deep_detectors:
            return torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 获取深度检测器
        detector = self.deep_detectors[node_type]
        
        # 计算重构误差
        with torch.no_grad():
            reconstruction_errors = detector.get_reconstruction_error(embeddings)
        
        # 归一化到[0,1]
        if torch.max(reconstruction_errors) > 0:
            anomaly_scores = reconstruction_errors / torch.max(reconstruction_errors)
        else:
            anomaly_scores = reconstruction_errors
        
        return anomaly_scores
    
    def _graph_anomaly_detection(self, node_type: str, embeddings: torch.Tensor, 
                                hetero_data: HeteroData) -> torch.Tensor:
        """
        基于图结构的异常检测
        
        Args:
            node_type: 节点类型
            embeddings: 节点嵌入
            hetero_data: 异构图数据
            
        Returns:
            异常分数
        """
        # 如果未启用图结构检测或没有对应的检测器，返回零分数
        if not self.use_graph_detector or node_type not in self.graph_detectors or GCNConv is None:
            return torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 获取图结构检测器
        detector = self.graph_detectors[node_type]
        
        # 获取节点的边索引
        # 注意：这里需要根据异构图的结构获取正确的边索引
        edge_indices = []
        for edge_type in hetero_data.edge_types:
            if edge_type[0] == node_type and edge_type[2] == node_type:
                # 同类型节点之间的边
                edge_indices.append(hetero_data[edge_type].edge_index)
        
        if not edge_indices:
            # 如果没有同类型节点之间的边，返回零分数
            return torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 合并边索引
        edge_index = torch.cat(edge_indices, dim=1)
        
        # 使用图检测器计算异常分数
        with torch.no_grad():
            anomaly_scores = detector(embeddings, edge_index)
        
        return anomaly_scores
    
    def _multi_scale_detection(self, node_type: str, embeddings: torch.Tensor, 
                              time_data: Dict[str, Any]) -> torch.Tensor:
        """
        多尺度异常检测
        
        Args:
            node_type: 节点类型
            embeddings: 节点嵌入
            time_data: 时间相关数据
            
        Returns:
            异常分数
        """
        # 如果未启用多尺度检测或没有时间数据，返回零分数
        if not self.use_multi_scale or 'time_windows' not in time_data:
            return torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 获取时间窗口数据
        time_windows = time_data.get('time_windows', {}).get(node_type, [])
        if not time_windows:
            return torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 多尺度异常分数
        multi_scale_scores = torch.zeros(embeddings.size(0), device=embeddings.device)
        
        # 对每个时间尺度进行检测
        for i, window in enumerate(time_windows):
            # 获取当前窗口的节点索引
            indices = window.get('indices', [])
            if not indices:
                continue
            
            # 获取当前窗口的嵌入
            window_embeddings = embeddings[indices]
            
            # 计算当前窗口的异常分数
            window_scores = self._calculate_anomaly_scores(window_embeddings)
            
            # 将分数分配给对应的节点
            for j, idx in enumerate(indices):
                if idx < len(multi_scale_scores):
                    # 根据时间尺度加权
                    scale_weight = self.scales[min(i, len(self.scales)-1)]
                    multi_scale_scores[idx] += scale_weight * window_scores[j]
        
        # 归一化
        if torch.max(multi_scale_scores) > 0:
            multi_scale_scores = multi_scale_scores / torch.max(multi_scale_scores)
        
        return multi_scale_scores
    
    def _fuse_anomaly_scores(self, score_list: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多种异常检测方法的分数
        
        Args:
            score_list: 异常分数列表
            
        Returns:
            融合后的异常分数
        """
        # 过滤掉None值
        valid_scores = [s for s in score_list if s is not None]
        
        if not valid_scores:
            # 如果没有有效分数，返回零分数
            return torch.zeros(0, device=self.device)
        
        # 确保所有分数的形状一致
        shapes = [s.shape[0] for s in valid_scores]
        if len(set(shapes)) > 1:
            # 如果形状不一致，使用最小的形状
            min_shape = min(shapes)
            valid_scores = [s[:min_shape] for s in valid_scores]
        
        # 融合策略：加权平均
        weights = [0.3, 0.3, 0.2, 0.2]  # 深度学习、图结构、统计、多尺度的权重
        
        # 调整权重数量以匹配有效分数数量
        weights = weights[:len(valid_scores)]
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
        else:
            weights = [1.0/len(valid_scores)] * len(valid_scores)
        
        # 加权平均
        fused_scores = torch.zeros_like(valid_scores[0])
        for i, score in enumerate(valid_scores):
            fused_scores += weights[i] * score
        
        return fused_scores
    
    def _determine_attack_stage(self, node_type: str, node_idx: int, 
                               anomaly_score: float, graph_score: float,
                               hetero_data: HeteroData) -> str:
        """
        确定攻击阶段
        
        Args:
            node_type: 节点类型
            node_idx: 节点索引
            anomaly_score: 异常分数
            graph_score: 图结构异常分数
            hetero_data: 异构图数据
            
        Returns:
            攻击阶段
        """
        # 基于MITRE ATT&CK框架的攻击阶段
        stages = [
            "initial_access", "execution", "persistence", "privilege_escalation",
            "defense_evasion", "credential_access", "discovery", "lateral_movement",
            "collection", "exfiltration", "command_and_control", "impact"
        ]
        
        # 简单实现：基于异常分数和图分数确定攻击阶段
        if anomaly_score > 0.9 and graph_score > 0.8:
            # 高异常分数和高图分数，可能是后期攻击阶段
            return stages[9]  # exfiltration
        elif anomaly_score > 0.8 and graph_score > 0.7:
            # 较高异常分数和较高图分数，可能是中期攻击阶段
            return stages[7]  # lateral_movement
        elif anomaly_score > 0.7:
            # 中等异常分数，可能是早期攻击阶段
            return stages[2]  # persistence
        else:
            # 低异常分数，可能是初始攻击阶段
            return stages[0]  # initial_access
    
    def _generate_anomaly_description(self, node_type: str, node_idx: int,
                                     anomaly_score: float, attack_stage: str,
                                     node_attrs: Dict) -> str:
        """
        生成异常描述
        
        Args:
            node_type: 节点类型
            node_idx: 节点索引
            anomaly_score: 异常分数
            attack_stage: 攻击阶段
            node_attrs: 节点属性
            
        Returns:
            异常描述
        """
        # 基于节点类型和攻击阶段生成描述
        descriptions = {
            "process": {
                "initial_access": "可疑进程可能是初始访问载体",
                "execution": "可疑进程执行了异常操作",
                "persistence": "可疑进程建立了持久化机制",
                "privilege_escalation": "可疑进程尝试提升权限",
                "defense_evasion": "可疑进程尝试逃避防御",
                "credential_access": "可疑进程尝试访问凭证",
                "discovery": "可疑进程执行了系统发现操作",
                "lateral_movement": "可疑进程尝试横向移动",
                "collection": "可疑进程收集敏感数据",
                "exfiltration": "可疑进程尝试数据泄露",
                "command_and_control": "可疑进程建立了命令控制通道",
                "impact": "可疑进程造成系统影响"
            },
            "file": {
                "initial_access": "可疑文件可能是初始访问载体",
                "execution": "可疑文件被异常执行",
                "persistence": "可疑文件用于建立持久化",
                "privilege_escalation": "可疑文件用于提升权限",
                "defense_evasion": "可疑文件用于逃避防御",
                "credential_access": "可疑文件包含凭证信息",
                "discovery": "可疑文件用于系统发现",
                "lateral_movement": "可疑文件用于横向移动",
                "collection": "可疑文件包含收集的数据",
                "exfiltration": "可疑文件被泄露",
                "command_and_control": "可疑文件用于命令控制",
                "impact": "可疑文件造成系统影响"
            },
            "connection": {
                "initial_access": "可疑连接可能是初始访问载体",
                "execution": "可疑连接用于执行命令",
                "persistence": "可疑连接用于建立持久化",
                "privilege_escalation": "可疑连接用于提升权限",
                "defense_evasion": "可疑连接用于逃避防御",
                "credential_access": "可疑连接用于获取凭证",
                "discovery": "可疑连接用于系统发现",
                "lateral_movement": "可疑连接用于横向移动",
                "collection": "可疑连接用于数据收集",
                "exfiltration": "可疑连接用于数据泄露",
                "command_and_control": "可疑连接是命令控制通道",
                "impact": "可疑连接造成系统影响"
            }
        }
        
        # 获取默认描述
        default_desc = f"{node_type}异常节点，异常分数: {anomaly_score:.3f}，攻击阶段: {attack_stage}"
        
        # 获取特定描述
        specific_desc = descriptions.get(node_type, {}).get(attack_stage, "")
        
        # 添加节点属性信息
        attrs_desc = ""
        if node_attrs:
            attrs = []
            for k, v in node_attrs.items():
                if k in ["name", "path", "cmd", "ip", "port", "user"]:
                    attrs.append(f"{k}={v}")
            if attrs:
                attrs_desc = "，".join(attrs)
        
        # 组合描述
        if specific_desc and attrs_desc:
            return f"{specific_desc}，{attrs_desc}，异常分数: {anomaly_score:.3f}"
        elif specific_desc:
            return f"{specific_desc}，异常分数: {anomaly_score:.3f}"
        elif attrs_desc:
            return f"{default_desc}，{attrs_desc}"
        else:
            return default_desc
    
    def _analyze_temporal_patterns(self, anomalies: Dict[str, Any], 
                                  time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析时序模式
        
        Args:
            anomalies: 异常检测结果
            time_data: 时间相关数据
            
        Returns:
            时序模式分析结果
        """
        temporal_patterns = {}
        
        # 获取时间窗口
        time_windows = time_data.get('time_windows', {})
        
        # 对每种节点类型进行分析
        for ntype in anomalies['anomalous_nodes']:
            if ntype not in time_windows:
                continue
                
            # 获取异常节点
            anomalous_nodes = anomalies['anomalous_nodes'][ntype]
            
            # 按时间窗口统计异常
            window_stats = []
            for i, window in enumerate(time_windows[ntype]):
                window_indices = set(window.get('indices', []))
                
                # 统计当前窗口的异常节点
                anomalous_in_window = []
                for node in anomalous_nodes:
                    node_idx = int(node['node_id'].split('_')[1])
                    if node_idx in window_indices:
                        anomalous_in_window.append(node)
                
                if anomalous_in_window:
                    window_stats.append({
                        'window_id': i,
                        'start_time': window.get('start_time', f"window_{i}_start"),
                        'end_time': window.get('end_time', f"window_{i}_end"),
                        'anomalous_count': len(anomalous_in_window),
                        'anomalous_nodes': [n['node_id'] for n in anomalous_in_window]
                    })
            
            # 检测时序模式
            if len(window_stats) > 1:
                # 计算异常增长率
                growth_rates = []
                for i in range(1, len(window_stats)):
                    prev_count = max(1, window_stats[i-1]['anomalous_count'])
                    curr_count = window_stats[i]['anomalous_count']
                    growth_rate = (curr_count - prev_count) / prev_count
                    growth_rates.append(growth_rate)
                
                # 检测突增
                spikes = [i+1 for i, rate in enumerate(growth_rates) if rate > 0.5]
                
                # 检测持续增长
                sustained_growth = all(rate >= 0 for rate in growth_rates) and len(growth_rates) >= 2
                
                temporal_patterns[ntype] = {
                    'window_stats': window_stats,
                    'growth_rates': growth_rates,
                    'spikes': spikes,
                    'sustained_growth': sustained_growth,
                    'pattern_type': 'spike' if spikes else ('growth' if sustained_growth else 'stable')
                }
        
        return temporal_patterns
    
    def _update_historical_anomalies(self, anomalies: Dict[str, Any]):
        """
        更新历史异常记录
        
        Args:
            anomalies: 异常检测结果
        """
        # 对每种节点类型进行更新
        for ntype, nodes in anomalies['anomalous_nodes'].items():
            if ntype not in self.historical_anomalies:
                self.historical_anomalies[ntype] = []
            
            # 添加新的异常记录
            for node in nodes:
                # 检查是否已存在
                node_id = node['node_id']
                existing = [n for n in self.historical_anomalies[ntype] if n['node_id'] == node_id]
                
                if existing:
                    # 更新现有记录
                    existing[0].update(node)
                else:
                    # 添加新记录
                    self.historical_anomalies[ntype].append(node.copy())
            
            # 限制历史记录数量
            max_history = 1000
            if len(self.historical_anomalies[ntype]) > max_history:
                self.historical_anomalies[ntype] = self.historical_anomalies[ntype][-max_history:]
    
    def _calculate_anomaly_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算基于统计的异常分数
        
        Args:
            embeddings: 节点嵌入
            
        Returns:
            异常分数
        """
        # 使用多种方法计算异常分数
        scores = []
        
        # 1. 基于距离的异常分数
        if len(embeddings) > 1:
            # 计算到中心的距离
            center = torch.mean(embeddings, dim=0)
            distances = torch.norm(embeddings - center, dim=1)
            distance_scores = distances / torch.max(distances)
            scores.append(distance_scores)
        
        # 2. 基于Isolation Forest的异常分数
        if len(embeddings) > 10:  # 需要足够的样本
            try:
                # 训练Isolation Forest
                self.isolation_forest.fit(embeddings.detach().cpu().numpy())
                if_scores = self.isolation_forest.decision_function(embeddings.detach().cpu().numpy())
                if_scores = torch.tensor(if_scores, device=embeddings.device)
                # 归一化到0-1
                if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
                scores.append(if_scores)
            except Exception as e:
                self.logger.warning(f"Isolation Forest异常检测失败: {e}")
        
        # 3. 基于LOF的异常分数
        if len(embeddings) > 10:  # 需要足够的样本
            try:
                # 训练LOF
                self.lof.fit(embeddings.detach().cpu().numpy())
                lof_scores = -self.lof.decision_function(embeddings.detach().cpu().numpy())
                lof_scores = torch.tensor(lof_scores, device=embeddings.device)
                # 归一化到0-1
                lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
                scores.append(lof_scores)
            except Exception as e:
                self.logger.warning(f"LOF异常检测失败: {e}")
        
        # 4. 基于统计的异常分数
        if len(embeddings) > 1:
            # 计算Z-score
            mean_emb = torch.mean(embeddings, dim=0)
            std_emb = torch.std(embeddings, dim=0) + 1e-8
            z_scores = torch.norm((embeddings - mean_emb) / std_emb, dim=1)
            z_scores = torch.sigmoid(z_scores)  # 归一化到0-1
            scores.append(z_scores)
        
        # 综合异常分数
        if scores:
            anomaly_scores = torch.stack(scores, dim=0).mean(dim=0)
        else:
            anomaly_scores = torch.zeros(embeddings.size(0), device=embeddings.device)
        
        return anomaly_scores
    
    def _cluster_anomalies(self, anomalies: Dict[str, Any], embeddings: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """
        对异常节点进行聚类
        
        Args:
            anomalies: 异常检测结果
            
        Returns:
            聚类结果
        """
        clusters = {}
        
        for ntype, node_info in anomalies['anomalous_nodes'].items():
            if isinstance(node_info, list) and len(node_info) > 1:
                try:
                    # 获取异常节点的嵌入
                    embeddings_list = []
                    for node_idx in node_info:
                        # 从实际节点嵌入中提取特征
                        node_embedding = embeddings.get(ntype, torch.zeros(10))
                        embeddings_list.append(node_embedding[node_idx].cpu().numpy())
                    
                    if embeddings_list:
                        embeddings_array = np.array(embeddings_list)
                        cluster_labels = self.dbscan.fit_predict(embeddings_array)
                    else:
                        # 如果无法获取嵌入，基于节点属性进行聚类
                        cluster_labels = np.zeros(len(node_info), dtype=int)
                    
                    clusters[ntype] = {
                        'labels': cluster_labels.tolist(),
                        'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                        'n_noise': list(cluster_labels).count(-1)
                    }
                except Exception as e:
                    self.logger.warning(f"聚类分析失败 {ntype}: {e}")
                    clusters[ntype] = {'labels': [], 'n_clusters': 0, 'n_noise': 0}
        
        return clusters
    
    def _generate_anomaly_summary(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成异常检测摘要
        
        Args:
            anomalies: 异常检测结果
            
        Returns:
            摘要信息
        """
        total_anomalous = sum(len(nodes) if isinstance(nodes, list) else len(nodes.get('indices', [])) for nodes in anomalies['anomalous_nodes'].values())
        total_suspicious = sum(len(nodes) if isinstance(nodes, list) else len(nodes.get('indices', [])) for nodes in anomalies['suspicious_nodes'].values())
        
        # 计算平均异常分数
        avg_scores = {}
        for ntype, scores in anomalies['anomaly_scores'].items():
            avg_scores[ntype] = np.mean(scores)
        
        # 风险等级评估
        if total_anomalous > 10:
            risk_level = 'high'
        elif total_anomalous > 5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'total_anomalous_nodes': total_anomalous,
            'total_suspicious_nodes': total_suspicious,
            'affected_node_types': list(anomalies['anomalous_nodes'].keys()),
            'average_anomaly_scores': avg_scores,
            'risk_level': risk_level,
            'clusters_found': sum(cluster['n_clusters'] for cluster in anomalies['clusters'].values())
        }
    
    def update_baseline(self, normal_embeddings: Dict[str, torch.Tensor]):
        """
        更新正常行为基线
        
        Args:
            normal_embeddings: 正常节点的嵌入
        """
        self.logger.info("更新正常行为基线")
        
        # 合并所有正常节点嵌入
        all_embeddings = []
        for ntype, embeddings in normal_embeddings.items():
            all_embeddings.append(embeddings.detach().cpu().numpy())
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            self.normal_baseline = {
                'mean': np.mean(combined_embeddings, axis=0),
                'std': np.std(combined_embeddings, axis=0),
                'cov': np.cov(combined_embeddings.T) if combined_embeddings.shape[1] > 1 else np.array([[1.0]])
            }
            
            # 重新训练异常检测模型
            self.isolation_forest.fit(combined_embeddings)
            self.scaler.fit(combined_embeddings)
            
            self.logger.info("正常行为基线更新完成")
    
    def get_detection_report(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成检测报告
        
        Args:
            anomalies: 异常检测结果
            
        Returns:
            检测报告
        """
        summary = anomalies['summary']
        
        report = {
            'detection_time': np.datetime64('now').astype(str),
            'total_anomalies': summary['total_anomalous_nodes'],
            'total_suspicious': summary['total_suspicious_nodes'],
            'risk_level': summary['risk_level'],
            'affected_types': summary['affected_node_types'],
            'recommendations': self._generate_recommendations(summary),
            'detailed_findings': anomalies['anomalous_nodes']
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """
        生成建议
        
        Args:
            summary: 异常检测摘要
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if summary['risk_level'] == 'high':
            recommendations.extend([
                "立即隔离异常节点",
                "检查系统完整性",
                "启动应急响应程序"
            ])
        elif summary['risk_level'] == 'medium':
            recommendations.extend([
                "密切监控异常节点",
                "加强安全防护措施",
                "分析异常模式"
            ])
        else:
            recommendations.extend([
                "继续监控系统状态",
                "定期检查安全日志"
            ])
        
        if summary['clusters_found'] > 0:
            recommendations.append("分析异常节点聚类，可能存在协同攻击")
        
        return recommendations
