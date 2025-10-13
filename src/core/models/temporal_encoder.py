"""
时序编码模块

实现大纲中提到的时序编码模块
通过随机位置编码（RPE）或时序注意力，将时间戳信息注入消息传递过程，使模型感知事件发生的先后顺序
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import math
from datetime import datetime, timedelta

try:
    from torch_geometric.nn import TransformerConv
    from torch_geometric.utils import softmax
except ImportError:
    TransformerConv = None
    softmax = None


def _get_config_value(config, key, default_value):
    """安全地获取配置值"""
    value = getattr(config, key, default_value)
    if hasattr(config, 'model'):
        value = getattr(config.model, key, value)
    return value


class RandomPositionalEncoding(nn.Module):
    """
    随机位置编码（RPE）
    
    实现大纲中提到的随机位置编码，用于时序建模
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化随机位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super(RandomPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # 基于正弦函数的位置编码矩阵（非随机）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 学习的位置编码参数
        self.learned_pe = nn.Parameter(torch.zeros(max_len, d_model))


        
        # 位置编码融合层
        self.pe_fusion = nn.Linear(d_model * 2, d_model)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"随机位置编码初始化完成，维度: {d_model}, 最大长度: {max_len}")
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            positions: 位置索引 [batch_size, seq_len]
            
        Returns:
            位置编码后的特征
        """
        seq_len = x.size(1)
        
        if positions is not None:
            # 使用指定的位置索引
            pos_enc = self.pe[positions]  # [batch_size, seq_len, d_model]
            learned_pos_enc = self.learned_pe[positions]  # [batch_size, seq_len, d_model]
        else:
            # 使用默认位置索引
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            pos_enc = self.pe[positions]  # [batch_size, seq_len, d_model]
            learned_pos_enc = self.learned_pe[positions]  # [batch_size, seq_len, d_model]
        
        # 融合随机位置编码和学习的位置编码
        combined_pe = torch.cat([pos_enc, learned_pos_enc], dim=-1)  # [batch_size, seq_len, 2*d_model]
        fused_pe = self.pe_fusion(combined_pe)  # [batch_size, seq_len, d_model]
        
        # 添加到输入特征
        x = x + fused_pe
        
        # 应用dropout
        x = self.dropout(x)
        
        return x


class TemporalAttention(nn.Module):
    """
    时序注意力机制
    
    实现时序感知的注意力机制，考虑时间间隔对注意力的影响
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化时序注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 查询、键、值投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 时序权重层
        self.temporal_weight = nn.Linear(1, num_heads, bias=False)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"时序注意力初始化完成，维度: {d_model}, 头数: {num_heads}")
    
    def forward(self, x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            timestamps: 时间戳 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            注意力输出
        """
        batch_size, seq_len, d_model = x.size()
        
        # 计算查询、键、值
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用时序权重
        if timestamps is not None:
            # 计算时间间隔
            time_diff = timestamps.unsqueeze(1) - timestamps.unsqueeze(2)  # [batch_size, seq_len, seq_len]
            time_diff = time_diff.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
            
            # 计算时序权重
            temporal_weights = self.temporal_weight(time_diff.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_heads, seq_len, seq_len]
            
            # 应用时序权重到注意力分数
            scores = scores + temporal_weights
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用dropout
        attention_weights = self.dropout_layer(attention_weights)
        
        # 计算加权和
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 重塑为原始形状
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        output = self.W_o(context)
        
        return output


class TemporalConvolution(nn.Module):
    """
    时序卷积层
    
    使用卷积操作捕获时序模式
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dropout: float = 0.1):
        """
        初始化时序卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dropout: Dropout率
        """
        super(TemporalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # 时序卷积层
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"时序卷积层初始化完成，输入通道: {in_channels}, 输出通道: {out_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, in_channels]
            
        Returns:
            卷积输出
        """
        # 转换为卷积格式 [batch_size, in_channels, seq_len]
        x = x.transpose(1, 2)
        
        # 卷积
        x = self.conv1d(x)
        
        # 批归一化
        x = self.batch_norm(x)
        
        # 激活函数
        x = F.relu(x)
        
        # Dropout
        x = self.dropout_layer(x)
        
        # 转换回原始格式 [batch_size, seq_len, out_channels]
        x = x.transpose(1, 2)
        
        return x


class TemporalEncoder(nn.Module):
    """
    时序编码模块
    
    实现大纲中提到的时序编码模块
    通过随机位置编码（RPE）或时序注意力，将时间戳信息注入消息传递过程
    """
    
    def __init__(self, config):
        """
        初始化时序编码模块
        
        Args:
            config: 模型配置
        """
        super(TemporalEncoder, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 时序维度 - 优化：减少维度以降低参数数量
        hidden_dim = _get_config_value(config, 'hidden_dim', 64)  # 从128减少到64
        self.temporal_dim = _get_config_value(config, 'temporal_dim', 32)  # 从hidden_dim减少到32
        if hasattr(config, 'model'):
            self.temporal_dim = _get_config_value(config, 'temporal_embedding_dim', self.temporal_dim)
        
        # 随机位置编码
        self.positional_encoding = RandomPositionalEncoding(
            d_model=self.temporal_dim,
            max_len=5000,
            dropout=_get_config_value(config, 'dropout', 0.3)
        )
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(
            d_model=self.temporal_dim,
            num_heads=_get_config_value(config, 'num_heads', 8),
            dropout=_get_config_value(config, 'dropout', 0.3)
        )
        
        # 时序卷积层 - 优化：减少卷积层数量
        self.temporal_conv_layers = nn.ModuleList([
            TemporalConvolution(
                in_channels=self.temporal_dim,
                out_channels=self.temporal_dim,
                kernel_size=3,
                dropout=_get_config_value(config, 'dropout', 0.4)  # 增加dropout
            ),
            TemporalConvolution(
                in_channels=self.temporal_dim,
                out_channels=self.temporal_dim,
                kernel_size=5,
                dropout=_get_config_value(config, 'dropout', 0.4)  # 增加dropout
            )
        ])
        
        # 多尺度融合
        self.multi_scale_fusion = nn.Linear(
            self.temporal_dim * len(self.temporal_conv_layers),
            self.temporal_dim
        )
        
        # 输出投影 - 优化：简化结构并增加正则化
        self.output_projection = nn.Sequential(
            nn.Linear(self.temporal_dim, self.temporal_dim),
            nn.BatchNorm1d(self.temporal_dim),  # 添加批归一化
            nn.ReLU(),
            nn.Dropout(_get_config_value(config, 'dropout', 0.5)),  # 增加dropout
            nn.Linear(self.temporal_dim, self.temporal_dim)
        )
        
        # 时序掩码生成器
        self.mask_generator = TemporalMaskGenerator()
        
        # 快照级别的注意力机制
        self.snapshot_attention = nn.MultiheadAttention(
            embed_dim=self.temporal_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.logger.info(f"时序编码模块初始化完成，时序维度: {self.temporal_dim}")
    
    def forward_snapshots(self, snapshots: List[torch.Tensor], 
                         timestamps: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        处理多个时序快照
        
        Args:
            snapshots: 快照列表，每个快照是 [num_nodes, feature_dim]
            timestamps: 时间戳列表，每个快照对应的时间戳 [num_nodes]
            
        Returns:
            融合后的时序特征
        """
        if not snapshots:
            raise ValueError("快照列表不能为空")
        
        # 处理每个快照
        encoded_snapshots = []
        for i, snapshot in enumerate(snapshots):
            snapshot_timestamps = timestamps[i] if timestamps and i < len(timestamps) else None
            encoded_snapshot = self.forward(snapshot, snapshot_timestamps)
            encoded_snapshots.append(encoded_snapshot)
        
        # 将快照堆叠成序列
        # encoded_snapshots: List[Tensor] -> Tensor [num_snapshots, num_nodes, feature_dim]
        sequence_features = torch.stack(encoded_snapshots, dim=0)

        # MultiheadAttention期望输入形状为 [batch, seq, embed] 且batch_first=True
        # 这里将num_nodes视为batch，num_snapshots视为seq
        sequence_features = sequence_features.transpose(0, 1)  # [num_nodes, num_snapshots, feature_dim]

        # 应用快照级注意力融合不同时间步的信息
        if hasattr(self, 'snapshot_attention') and self.snapshot_attention is not None:
            attn_out, _ = self.snapshot_attention(sequence_features, sequence_features, sequence_features)
            sequence_features = attn_out

        # 返回最后一个时间步的特征（或可改为平均/加权融合）
        return sequence_features[:, -1, :]  # [num_nodes, feature_dim]
    
    def forward(self, node_features: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, feature_dim]
            timestamps: 时间戳 [num_nodes]
            
        Returns:
            时序编码后的特征
        """
        # 如果输入是2D，添加序列维度
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(1)  # [num_nodes, 1, feature_dim]
            if timestamps is not None:
                timestamps = timestamps.unsqueeze(1)  # [num_nodes, 1]
        
        # 投影到时序维度
        if node_features.size(-1) != self.temporal_dim:
            projection = nn.Linear(node_features.size(-1), self.temporal_dim).to(node_features.device)
            node_features = projection(node_features)
        
        # 随机位置编码
        if timestamps is not None:
            # 将时间戳转换为位置索引
            positions = self._timestamp_to_positions(timestamps)
            node_features = self.positional_encoding(node_features, positions)
        else:
            node_features = self.positional_encoding(node_features)
        
        # 时序注意力
        node_features = self.temporal_attention(node_features, timestamps)
        
        # 多尺度时序卷积
        conv_outputs = []
        for conv_layer in self.temporal_conv_layers:
            conv_output = conv_layer(node_features)
            conv_outputs.append(conv_output)
        
        # 融合多尺度特征
        if len(conv_outputs) > 1:
            multi_scale_features = torch.cat(conv_outputs, dim=-1)
            node_features = self.multi_scale_fusion(multi_scale_features)
        else:
            node_features = conv_outputs[0]
        
        # 输出投影
        node_features = self.output_projection(node_features)
        
        # 如果输入是2D，移除序列维度
        if node_features.dim() == 3 and node_features.size(1) == 1:
            node_features = node_features.squeeze(1)
        
        return node_features
    
    def _timestamp_to_positions(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        将时间戳转换为位置索引
        
        Args:
            timestamps: 时间戳张量
            
        Returns:
            位置索引张量
        """
        if timestamps.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=timestamps.device)
        
        # 获取时间戳的统计信息
        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()
        time_range = max_timestamp - min_timestamp
        
        # 动态确定时间粒度
        if time_range <= 3600:  # 1小时内，使用秒级粒度
            time_unit = 1.0
        elif time_range <= 86400:  # 1天内，使用分钟级粒度
            time_unit = 60.0
        elif time_range <= 604800:  # 1周内，使用小时级粒度
            time_unit = 3600.0
        else:  # 更长时间，使用天级粒度
            time_unit = 86400.0
        
        # 将时间戳转换为相对位置
        relative_timestamps = timestamps - min_timestamp
        
        # 将相对时间戳转换为位置索引
        positions = (relative_timestamps / time_unit).long()
        
        # 确保位置索引从0开始
        positions = positions - positions.min()
        
        return positions
    
    def get_temporal_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取时序注意力权重
        
        Returns:
            注意力权重字典
        """
        return {
            'temporal_attention': self.temporal_attention.attention_weights if hasattr(self.temporal_attention, 'attention_weights') else None
        }


class TemporalMaskGenerator:
    """
    时序掩码生成器
    
    生成时序掩码，确保信息只能从过去流向未来
    """
    
    def __init__(self):
        """初始化时序掩码生成器"""
        self.logger = logging.getLogger(__name__)
    
    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            因果掩码 [seq_len, seq_len]
        """
        # 创建下三角掩码，确保信息只能从过去流向未来
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def generate_temporal_mask(self, timestamps: torch.Tensor, 
                              max_time_diff: float = 3600) -> torch.Tensor:
        """
        生成时序掩码
        
        Args:
            timestamps: 时间戳 [seq_len]
            max_time_diff: 最大时间间隔（秒）
            
        Returns:
            时序掩码 [seq_len, seq_len]
        """
        seq_len = timestamps.size(0)
        device = timestamps.device
        
        # 计算时间差矩阵
        time_diff = timestamps.unsqueeze(0) - timestamps.unsqueeze(1)  # [seq_len, seq_len]
        
        # 创建时序掩码：只允许时间差在合理范围内的连接
        temporal_mask = (time_diff >= 0) & (time_diff <= max_time_diff)
        
        return temporal_mask.float()
    
    def generate_combined_mask(self, seq_len: int, timestamps: torch.Tensor, 
                              max_time_diff: float = 3600) -> torch.Tensor:
        """
        生成组合掩码（因果掩码 + 时序掩码）
        
        Args:
            seq_len: 序列长度
            timestamps: 时间戳
            max_time_diff: 最大时间间隔
            
        Returns:
            组合掩码
        """
        # 因果掩码
        causal_mask = self.generate_causal_mask(seq_len, timestamps.device)
        
        # 时序掩码
        temporal_mask = self.generate_temporal_mask(timestamps, max_time_diff)
        
        # 组合掩码
        combined_mask = causal_mask * temporal_mask
        
        return combined_mask
