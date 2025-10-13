"""
真实的数据加载器实现

用于T-HGNN模型的训练和评估
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from typing import List, Dict, Any, Tuple
import logging


class HeteroGraphDataset(Dataset):
    """异构图数据集"""
    
    def __init__(self, graphs: List[HeteroData], labels: Dict[str, torch.Tensor]):
        """
        初始化数据集
        
        Args:
            graphs: 异构图列表
            labels: 标签字典 {node_type: labels}
        """
        self.graphs = graphs
        self.labels = labels
        self.logger = logging.getLogger(__name__)
        
        # 验证数据
        self._validate_data()
    
    def _validate_data(self):
        """验证数据完整性"""
        if not self.graphs:
            raise ValueError("图列表不能为空")
        
        if not self.labels:
            raise ValueError("标签字典不能为空")
        
        # 检查标签长度是否与图数量匹配
        for node_type, label_tensor in self.labels.items():
            if len(label_tensor) != len(self.graphs):
                # 如果标签数量大于图数量，这是正常的（每个节点类型有多个标签）
                if len(label_tensor) > len(self.graphs):
                    self.logger.debug(f"节点类型 {node_type} 的标签长度({len(label_tensor)})大于图数量({len(self.graphs)})，这是正常的")
                else:
                    self.logger.warning(f"节点类型 {node_type} 的标签长度({len(label_tensor)})小于图数量({len(self.graphs)})，可能存在问题")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if idx >= len(self.graphs):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.graphs)})")
        
        graph = self.graphs[idx]
        
        # 提取标签
        sample_labels = {}
        for node_type, label_tensor in self.labels.items():
            if idx < len(label_tensor):
                # 如果标签是标量，直接使用
                if label_tensor.dim() == 0:
                    sample_labels[node_type] = label_tensor.unsqueeze(0)
                else:
                    sample_labels[node_type] = label_tensor[idx]
            else:
                # 如果标签长度不足，使用第一个标签或创建零标签
                if len(label_tensor) > 0:
                    if label_tensor.dim() == 0:
                        sample_labels[node_type] = label_tensor.unsqueeze(0)
                    else:
                        sample_labels[node_type] = label_tensor[0]
                else:
                    sample_labels[node_type] = torch.zeros(1, dtype=label_tensor.dtype)
        
        return graph, sample_labels


class RealDataLoader:
    """真实的数据加载器"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据加载器配置
        self.batch_size = getattr(config.training, 'batch_size', 64)
        self.num_workers = getattr(config.training, 'num_workers', 4)
        self.pin_memory = getattr(config.training, 'pin_memory', True)
        self.shuffle = getattr(config.training, 'shuffle', True)
    
    def create_dataloader(self, graphs: List[HeteroData], labels: Dict[str, torch.Tensor], 
                         shuffle: bool = None) -> DataLoader:
        """
        创建PyTorch DataLoader
        
        Args:
            graphs: 异构图列表
            labels: 标签字典
            shuffle: 是否打乱数据
            
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = self.shuffle
        
        # 创建数据集
        dataset = HeteroGraphDataset(graphs, labels)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self._collate_fn
        )
        
        self.logger.info(f"创建数据加载器: 批次大小={self.batch_size}, 样本数={len(dataset)}")
        
        return dataloader
    
    def _collate_fn(self, batch):
        """
        自定义批处理函数
        
        Args:
            batch: 批次数据列表 [(graph, labels), ...]
            
        Returns:
            批处理后的数据 (batch_graphs, batch_labels)
        """
        graphs, labels_list = zip(*batch)
        
        # 处理图数据
        batch_graphs = list(graphs)
        
        # 处理标签数据
        batch_labels = {}
        for node_type in labels_list[0].keys():
            # 收集该节点类型的所有标签
            node_labels = []
            for labels in labels_list:
                if node_type in labels:
                    node_labels.append(labels[node_type])
            
            if node_labels:
                # 堆叠标签
                try:
                    batch_labels[node_type] = torch.stack(node_labels)
                except RuntimeError as e:
                    # 如果标签形状不一致，使用pad_sequence
                    self.logger.warning(f"节点类型 {node_type} 标签形状不一致，使用填充: {e}")
                    batch_labels[node_type] = torch.nn.utils.rnn.pad_sequence(
                        node_labels, batch_first=True, padding_value=0
                    )
        
        return batch_graphs, batch_labels
    
    def create_train_val_test_loaders(self, graphs: List[HeteroData], labels: Dict[str, torch.Tensor],
                                    train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建训练、验证和测试数据加载器
        
        Args:
            graphs: 异构图列表
            labels: 标签字典
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        from sklearn.model_selection import train_test_split
        
        # 计算分割点
        total_size = len(graphs)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 分割数据
        train_graphs = graphs[:train_size]
        val_graphs = graphs[train_size:train_size + val_size]
        test_graphs = graphs[train_size + val_size:]
        
        # 分割标签
        train_labels = {}
        val_labels = {}
        test_labels = {}
        
        for node_type, label_tensor in labels.items():
            train_labels[node_type] = label_tensor[:train_size]
            val_labels[node_type] = label_tensor[train_size:train_size + val_size]
            test_labels[node_type] = label_tensor[train_size + val_size:]
        
        # 创建数据加载器
        train_loader = self.create_dataloader(train_graphs, train_labels, shuffle=True)
        val_loader = self.create_dataloader(val_graphs, val_labels, shuffle=False)
        test_loader = self.create_dataloader(test_graphs, test_labels, shuffle=False)
        
        self.logger.info(f"数据分割完成: 训练集={len(train_graphs)}, 验证集={len(val_graphs)}, 测试集={len(test_graphs)}")
        
        return train_loader, val_loader, test_loader


class GraphBatchProcessor:
    """图批处理器"""
    
    def __init__(self, device):
        """
        初始化批处理器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, batch_graphs: List[HeteroData], batch_labels: Dict[str, torch.Tensor]) -> Tuple[List[HeteroData], Dict[str, torch.Tensor]]:
        """
        处理批次数据
        
        Args:
            batch_graphs: 批次图列表
            batch_labels: 批次标签字典
            
        Returns:
            处理后的批次数据
        """
        # 将图移动到设备
        processed_graphs = []
        for graph in batch_graphs:
            graph = graph.to(self.device)
            processed_graphs.append(graph)
        
        # 将标签移动到设备
        processed_labels = {}
        for node_type, labels in batch_labels.items():
            processed_labels[node_type] = labels.to(self.device)
        
        return processed_graphs, processed_labels
    
    def extract_features(self, batch_graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
        """
        从批次图中提取特征
        
        Args:
            batch_graphs: 批次图列表
            
        Returns:
            特征字典 {node_type: features}
        """
        features = {}
        
        for graph in batch_graphs:
            for node_type in graph.node_types:
                if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                    if node_type not in features:
                        features[node_type] = []
                    features[node_type].append(graph[node_type].x)
        
        # 合并特征
        for node_type in features:
            try:
                features[node_type] = torch.cat(features[node_type], dim=0)
            except RuntimeError as e:
                self.logger.warning(f"节点类型 {node_type} 特征合并失败: {e}")
                # 使用填充
                features[node_type] = torch.nn.utils.rnn.pad_sequence(
                    features[node_type], batch_first=True, padding_value=0
                )
        
        return features

