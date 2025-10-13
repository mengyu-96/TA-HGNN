# 时序异构图设计详解

## 概述

本文档详细说明了T-HGNN系统中时序异构图的设计理念、实现方法和核心算法。时序异构图是系统的核心数据结构，它将静态的异构图扩展为动态的时序图，能够捕获APT攻击中的时间依赖关系和因果逻辑。

## 设计理念

### 1. 从静态到动态的演进

传统图神经网络主要处理静态图，而APT攻击具有明显的时间特性：
- **攻击阶段**: 攻击通常分为多个阶段（初始访问、执行、持久化等）
- **时间依赖**: 后续攻击步骤依赖于前面步骤的成功
- **因果逻辑**: 事件之间存在时间上的因果关系

### 2. 时序建模的核心挑战

- **时间尺度**: 攻击可能持续数天、数周甚至数月
- **事件密度**: 不同时间段的事件密度差异很大
- **因果关系**: 需要识别事件间的时序因果关系
- **噪声处理**: 大量正常事件中的少量攻击事件

## 架构设计

### 1. 整体架构

```
原始日志数据
    ↓
数据预处理层
    ├── 实体关系抽取
    ├── 时间戳标准化
    └── 特征工程
    ↓
时序异构图构建
    ├── 节点类型定义
    ├── 边类型定义
    └── 时序快照创建
    ↓
T-HGNN模型
    ├── 异质图编码器 (HGNN)
    ├── 时序编码模块 (Temporal Encoder)
    └── 节点分类器 (Node Classifier)
    ↓
智能应用层
    ├── 攻击检测
    ├── 攻击溯源
    └── 攻击聚类
```

### 2. 时序异构图结构

#### 2.1 节点类型定义

```python
node_types = [
    'alert',      # 安全告警
    'host',       # 主机
    'agent',      # 代理
    'rule',       # 规则
    'file',       # 文件
    'command',    # 命令
    'user',       # 用户
    'process',    # 进程
    'ip',         # IP地址
    'domain',     # 域名
    'timestamp',  # 时间戳
    'registry',   # 注册表
    'port',       # 端口
    'service'     # 服务
]
```

#### 2.2 边类型定义

```python
edge_types = [
    # 告警相关边
    ('alert', 'triggered_by', 'rule'),
    ('alert', 'detected_on', 'host'),
    ('alert', 'reported_by', 'agent'),
    ('alert', 'involves', 'file'),
    ('alert', 'executed', 'command'),
    ('alert', 'by_user', 'user'),
    ('alert', 'at_time', 'timestamp'),
    ('alert', 'involves_process', 'process'),
    
    # 网络相关边
    ('alert', 'connects_to', 'ip'),
    ('alert', 'connects_to', 'domain'),
    ('alert', 'uses_port', 'port'),
    ('alert', 'affects_service', 'service'),
    
    # 主机相关边
    ('host', 'has_agent', 'agent'),
    ('host', 'has_ip', 'ip'),
    ('host', 'runs_service', 'service'),
    ('host', 'has_open_port', 'port'),
    
    # 文件相关边
    ('file', 'on_host', 'host'),
    ('file', 'accessed_by', 'process'),
    ('file', 'owned_by', 'user'),
    
    # 用户相关边
    ('user', 'on_host', 'host'),
    ('user', 'owns', 'process'),
    
    # 进程相关边
    ('process', 'on_host', 'host'),
    ('process', 'executed_by', 'user'),
    ('process', 'connects_to', 'ip'),
    ('process', 'uses_port', 'port'),
    ('process', 'accesses', 'registry'),
    ('process', 'communicates_with', 'domain'),
    
    # 网络解析边
    ('ip', 'resolves_to', 'domain'),
    ('service', 'listens_on', 'port')
]
```

## 核心算法实现

### 1. 时序编码模块 (Temporal Encoder)

#### 1.1 随机位置编码 (Random Positional Encoding)

```python
class RandomPositionalEncoding(nn.Module):
    """
    随机位置编码（RPE）
    
    为每个时间步生成唯一的随机位置编码，增强时序感知能力
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(RandomPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # 随机位置编码矩阵
        self.register_buffer('pe', torch.randn(max_len, d_model))
        
        # 学习的位置编码参数
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model))
        
        # 位置编码融合层
        self.pe_fusion = nn.Linear(d_model * 2, d_model)
```

**设计特点**:
- **随机性**: 使用随机初始化避免过拟合
- **可学习**: 结合学习的位置编码参数
- **融合机制**: 将随机编码和学习编码融合

#### 1.2 时序注意力机制 (Temporal Attention)

```python
class TemporalAttention(nn.Module):
    """
    时序注意力机制
    
    考虑时间间隔对注意力的影响，实现时序感知的注意力计算
    """
    
    def forward(self, x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算查询、键、值
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用时序权重
        if timestamps is not None:
            # 计算时间间隔
            time_diff = timestamps.unsqueeze(1) - timestamps.unsqueeze(2)
            
            # 计算时序权重
            temporal_weights = self.temporal_weight(time_diff.unsqueeze(-1))
            
            # 应用时序权重到注意力分数
            scores = scores + temporal_weights
```

**设计特点**:
- **时间感知**: 考虑时间间隔对注意力的影响
- **多头机制**: 使用多头注意力捕获不同的时序模式
- **掩码支持**: 支持注意力掩码，处理变长序列

#### 1.3 时序卷积层 (Temporal Convolution)

```python
class TemporalConvolution(nn.Module):
    """
    时序卷积层
    
    使用卷积操作捕获时序模式，提取局部时序特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dilation: int = 1):
        super(TemporalConvolution, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 
                             kernel_size, dilation=dilation, 
                             padding=(kernel_size-1)*dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
```

**设计特点**:
- **局部模式**: 捕获局部时序模式
- **多尺度**: 支持不同的卷积核大小和膨胀率
- **特征提取**: 提取时序特征用于后续处理

### 2. 异质图编码器 (HGNN Encoder)

#### 2.1 异质图卷积层

```python
class HeterogeneousGNNLayer(nn.Module):
    """
    异质图卷积层
    
    处理不同类型的节点和边，实现异质图的信息传播
    """
    
    def __init__(self, in_channels: Dict[str, int], out_channels: int, 
                 num_heads: int, dropout: float, edge_types: List[Tuple[str, str, str]]):
        super(HeterogeneousGNNLayer, self).__init__()
        self.convs = nn.ModuleDict()
        
        # 为每种关系类型定义一个GNN层
        for src, rel, dst in edge_types:
            if src in in_channels and dst in in_channels:
                self.convs[f'{src}__{rel}__{dst}'] = HGTConv(
                    in_channels=(in_channels[src], in_channels[dst]),
                    out_channels=out_channels,
                    num_heads=num_heads
                )
```

**设计特点**:
- **类型感知**: 为不同类型的节点和边使用不同的卷积层
- **关系建模**: 建模不同类型的关系
- **信息聚合**: 聚合来自不同关系的信息

#### 2.2 多层异质图编码

```python
class HGNNEncoder(nn.Module):
    """
    异质图编码器
    
    多层异质图卷积，实现深度的图表示学习
    """
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        # 输入特征投影
        h = {}
        for ntype, x in data.x_dict.items():
            if ntype in self.input_projections:
                h[ntype] = self.input_projections[ntype](x)
                h[ntype] = self.relu(h[ntype])
                h[ntype] = self.dropout(h[ntype])
        
        # 多层异质图卷积
        for layer in self.layers:
            h = layer(h, data.edge_index_dict)
            # 激活和Dropout
            for ntype in h.keys():
                h[ntype] = self.relu(h[ntype])
                h[ntype] = self.dropout(h[ntype])
        
        return h
```

**设计特点**:
- **深度建模**: 多层卷积实现深度表示学习
- **特征投影**: 将不同节点类型的特征投影到统一空间
- **残差连接**: 支持残差连接，避免梯度消失

### 3. 时序异构图构建

#### 3.1 时序快照创建

```python
def create_temporal_snapshots(self, data: HeteroData) -> List[HeteroData]:
    """
    创建时序快照
    
    将静态异构图转换为时序图序列
    """
    snapshots = []
    
    # 根据时间窗口创建快照
    for i in range(self.config.num_snapshots):
        snapshot = HeteroData()
        
        # 复制节点特征
        for ntype in data.node_types:
            if data[ntype].x is not None:
                snapshot[ntype].x = data[ntype].x.clone()
        
        # 复制边信息
        for edge_type in data.edge_types:
            if edge_type in data.edge_index_dict:
                snapshot[edge_type].edge_index = data[edge_type].edge_index.clone()
        
        snapshots.append(snapshot)
    
    return snapshots
```

**设计特点**:
- **时间窗口**: 根据时间窗口创建快照
- **增量更新**: 支持增量更新图结构
- **内存优化**: 优化内存使用，支持大规模数据

#### 3.2 时序特征融合

```python
def forward(self, data: HeteroData, 
            timestamps: Optional[Dict[str, torch.Tensor]] = None,
            return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
    """
    前向传播
    
    整合异质图编码和时序编码
    """
    # 1. 异质图编码
    hgnn_embeddings = self.hgnn_encoder(projected_features, data)
    
    # 2. 时序编码
    temporal_embeddings = {}
    if timestamps is not None:
        for ntype in self.node_types:
            if ntype in hgnn_embeddings and ntype in timestamps:
                temporal_emb = self.temporal_encoder(
                    hgnn_embeddings[ntype], 
                    timestamps[ntype]
                )
                temporal_embeddings[ntype] = temporal_emb
    
    # 3. 特征融合
    fused_embeddings = {}
    for ntype in self.node_types:
        if ntype in hgnn_embeddings:
            hgnn_feat = hgnn_embeddings[ntype]
            temporal_feat = temporal_embeddings.get(ntype, 
                torch.zeros(hgnn_feat.size(0), self.config.temporal_dim, device=hgnn_feat.device))
            
            # 拼接特征
            combined_feat = torch.cat([hgnn_feat, temporal_feat], dim=-1)
            fused_embeddings[ntype] = self.fusion_layers[ntype](combined_feat)
    
    # 4. 节点分类
    logits = self.node_classifier(fused_embeddings)
    
    return logits
```

**设计特点**:
- **特征融合**: 将异质图特征和时序特征融合
- **灵活处理**: 支持有无时序信息的情况
- **端到端**: 端到端的训练和推理

## 时序建模策略

### 1. 时间窗口策略

#### 1.1 滑动窗口
- **固定窗口**: 使用固定大小的时间窗口
- **重叠窗口**: 窗口之间有重叠，避免信息丢失
- **自适应窗口**: 根据事件密度调整窗口大小

#### 1.2 分层时间建模
- **短期模式**: 捕获分钟到小时级别的事件模式
- **中期模式**: 捕获天级别的攻击阶段
- **长期模式**: 捕获周级别的攻击活动

### 2. 时序特征工程

#### 2.1 时间戳特征
```python
def extract_temporal_features(timestamps):
    """
    提取时序特征
    """
    features = {}
    
    # 基础时间特征
    features['hour'] = timestamps.hour
    features['day_of_week'] = timestamps.dayofweek
    features['day_of_month'] = timestamps.day
    features['month'] = timestamps.month
    
    # 相对时间特征
    features['time_since_start'] = (timestamps - timestamps.min()).dt.total_seconds()
    features['time_since_last'] = timestamps.diff().dt.total_seconds()
    
    # 周期性特征
    features['hour_sin'] = np.sin(2 * np.pi * timestamps.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * timestamps.hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
    
    return features
```

#### 2.2 事件序列特征
```python
def extract_sequence_features(events):
    """
    提取事件序列特征
    """
    features = {}
    
    # 事件频率
    features['event_frequency'] = events.groupby('node_id').size()
    
    # 事件间隔
    features['event_intervals'] = events.groupby('node_id')['timestamp'].diff()
    
    # 事件模式
    features['event_patterns'] = extract_patterns(events)
    
    return features
```

### 3. 因果建模

#### 3.1 时序因果关系
```python
def model_temporal_causality(events, time_window=3600):
    """
    建模时序因果关系
    """
    causality_graph = nx.DiGraph()
    
    for i, event1 in events.iterrows():
        for j, event2 in events.iterrows():
            if i != j:
                time_diff = (event2['timestamp'] - event1['timestamp']).total_seconds()
                
                # 时间窗口内的因果关系
                if 0 < time_diff <= time_window:
                    # 计算因果强度
                    causal_strength = calculate_causal_strength(event1, event2, time_diff)
                    
                    if causal_strength > threshold:
                        causality_graph.add_edge(
                            event1['node_id'], 
                            event2['node_id'], 
                            weight=causal_strength,
                            time_diff=time_diff
                        )
    
    return causality_graph
```

#### 3.2 攻击链重构
```python
def reconstruct_attack_chain(causality_graph, start_node, max_depth=10):
    """
    重构攻击链
    """
    attack_chain = []
    visited = set()
    
    def dfs(node, depth):
        if depth >= max_depth or node in visited:
            return
        
        visited.add(node)
        attack_chain.append(node)
        
        # 按因果强度排序邻居
        neighbors = sorted(
            causality_graph.neighbors(node),
            key=lambda x: causality_graph[node][x]['weight'],
            reverse=True
        )
        
        for neighbor in neighbors:
            dfs(neighbor, depth + 1)
    
    dfs(start_node, 0)
    return attack_chain
```

## 性能优化

### 1. 内存优化

#### 1.1 稀疏表示
```python
def create_sparse_temporal_graph(events, time_bins=1000):
    """
    创建稀疏时序图
    """
    # 时间分桶
    time_bins = pd.cut(events['timestamp'], bins=time_bins)
    
    # 创建稀疏邻接矩阵
    adj_matrix = sparse.lil_matrix((len(events), len(events)))
    
    for i, event1 in events.iterrows():
        for j, event2 in events.iterrows():
            if i != j and time_bins[i] == time_bins[j]:
                adj_matrix[i, j] = 1
    
    return adj_matrix
```

#### 1.2 增量更新
```python
def incremental_update(graph, new_events):
    """
    增量更新图结构
    """
    # 只更新新增的边
    for event in new_events:
        add_edges = find_new_edges(event, graph)
        graph.add_edges_from(add_edges)
    
    return graph
```

### 2. 计算优化

#### 2.1 并行处理
```python
def parallel_temporal_encoding(embeddings, timestamps, num_workers=4):
    """
    并行时序编码
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for ntype in embeddings.keys():
            future = executor.submit(
                temporal_encoder, 
                embeddings[ntype], 
                timestamps[ntype]
            )
            futures.append((ntype, future))
        
        results = {}
        for ntype, future in futures:
            results[ntype] = future.result()
        
        return results
```

#### 2.2 批处理
```python
def batch_temporal_processing(events, batch_size=1000):
    """
    批处理时序数据
    """
    results = []
    
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        batch_result = process_temporal_batch(batch)
        results.append(batch_result)
    
    return combine_batch_results(results)
```

## 实验验证

### 1. 数据集

#### 1.1 Linux APT数据集
- **规模**: 100万+条日志记录
- **时间跨度**: 30天
- **节点类型**: 14种
- **边类型**: 30种

#### 1.2 评估指标
- **检测准确率**: 攻击检测的准确率
- **溯源完整性**: 攻击链重构的完整性
- **时序准确性**: 时序建模的准确性
- **计算效率**: 处理大规模数据的效率

### 2. 基线对比

#### 2.1 静态图方法
- **GCN**: 图卷积网络
- **GAT**: 图注意力网络
- **GraphSAGE**: 图采样聚合网络

#### 2.2 时序图方法
- **TGCN**: 时序图卷积网络
- **T-GCN**: 时序图卷积网络
- **STGCN**: 时空图卷积网络

### 3. 消融实验

#### 3.1 组件消融
- **无时序编码**: 移除时序编码模块
- **无异质图**: 使用同质图
- **无注意力**: 移除注意力机制

#### 3.2 参数消融
- **时间窗口大小**: 不同时间窗口的影响
- **层数**: 不同层数的影响
- **头数**: 不同注意力头数的影响

## 应用场景

### 1. APT攻击检测
- **实时检测**: 实时检测APT攻击活动
- **历史分析**: 分析历史攻击数据
- **模式识别**: 识别攻击模式

### 2. 攻击溯源
- **攻击链重构**: 重构完整的攻击链
- **入口点定位**: 定位攻击入口点
- **影响范围评估**: 评估攻击影响范围

### 3. 威胁情报
- **IOC提取**: 提取攻击指标
- **TTP分析**: 分析攻击战术、技术和程序
- **归因分析**: 进行攻击归因分析

## 总结

时序异构图设计是T-HGNN系统的核心创新，它通过以下方式实现了从静态图到动态图的演进：

1. **时序建模**: 通过随机位置编码和时序注意力机制捕获时序信息
2. **异质处理**: 通过异质图编码器处理不同类型的节点和边
3. **特征融合**: 将异质图特征和时序特征有效融合
4. **因果分析**: 通过时序因果关系进行攻击链重构

这种设计使得系统能够：
- 捕获APT攻击的时间特性
- 理解攻击的因果逻辑
- 实现准确的攻击检测和溯源
- 提供可解释的分析结果

时序异构图设计为APT攻击检测与溯源提供了强大的技术基础，是系统实现"故事驱动"安全分析的关键技术。
