"""
攻击检测器

实现基于T-HGNN的高级攻击检测功能，包括基于ATT&CK框架的攻击模式识别、攻击链重构和攻击行为分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from datetime import datetime, timedelta
import networkx as nx
import json
import os
import re
import pandas as pd
from collections import defaultdict, Counter, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
import math
import time
import random
from tqdm import tqdm

try:
    from torch_geometric.data import HeteroData, Batch
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.utils import to_networkx, k_hop_subgraph
    from torch_geometric.utils import add_self_loops, remove_self_loops
except ImportError:
    HeteroData = None
    Batch = None
    GCNConv = None
    GATConv = None
    SAGEConv = None
    to_networkx = None
    k_hop_subgraph = None
    add_self_loops = None
    remove_self_loops = None


class AttackPatternMatcher(nn.Module):
    """
    攻击模式匹配器
    
    基于图神经网络实现攻击模式的匹配和识别
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_patterns=10, dropout=0.2):
        """
        初始化攻击模式匹配器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_patterns: 攻击模式数量
            dropout: Dropout比率
        """
        super(AttackPatternMatcher, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 攻击模式分类器
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_patterns),
            nn.Sigmoid()
        )
        
        # 攻击阶段分类器
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 12),  # 12个MITRE ATT&CK战术阶段
            nn.Sigmoid()
        )
        
        # 攻击技术分类器
        self.technique_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 50),  # 常见的ATT&CK技术
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            攻击模式概率, 攻击阶段概率, 攻击技术概率
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 攻击模式分类
        pattern_probs = self.pattern_classifier(features)
        
        # 攻击阶段分类
        stage_probs = self.stage_classifier(features)
        
        # 攻击技术分类
        technique_probs = self.technique_classifier(features)
        
        return pattern_probs, stage_probs, technique_probs


class AttackChainReconstructor:
    """
    攻击链重构器
    
    基于时序图和攻击模式重构完整的攻击链
    """
    
    def __init__(self, config):
        """
        初始化攻击链重构器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 攻击链重构参数
        self.min_chain_length = getattr(config, 'min_chain_length', 3)
        self.max_time_gap = getattr(config, 'max_time_gap', 3600)  # 默认1小时
        self.min_confidence = getattr(config, 'min_confidence', 0.6)
        self.max_paths = getattr(config, 'max_paths', 10)
        
        # 攻击阶段转换概率矩阵
        self.stage_transition = self._init_stage_transition()
        
        # 缓存已识别的攻击链
        self.attack_chains = []
    
    def _init_stage_transition(self):
        """
        初始化攻击阶段转换概率矩阵
        
        Returns:
            攻击阶段转换概率矩阵
        """
        # 基于MITRE ATT&CK战术的转换概率
        # 行: 当前阶段, 列: 下一阶段
        stages = [
            'initial_access', 'execution', 'persistence', 'privilege_escalation', 
            'defense_evasion', 'credential_access', 'discovery', 'lateral_movement',
            'collection', 'command_and_control', 'exfiltration', 'impact'
        ]
        
        # 初始化转换矩阵
        n_stages = len(stages)
        transition = np.zeros((n_stages, n_stages))
        
        # 设置转换概率 (基于攻击链常见路径)
        # 初始访问 -> 执行
        transition[0, 1] = 0.8
        # 执行 -> 持久化, 防御规避, 发现
        transition[1, 2] = 0.4
        transition[1, 4] = 0.3
        transition[1, 6] = 0.3
        # 持久化 -> 权限提升, 凭证访问
        transition[2, 3] = 0.6
        transition[2, 5] = 0.4
        # 权限提升 -> 防御规避, 凭证访问, 发现
        transition[3, 4] = 0.3
        transition[3, 5] = 0.3
        transition[3, 6] = 0.4
        # 防御规避 -> 凭证访问, 发现, 横向移动
        transition[4, 5] = 0.3
        transition[4, 6] = 0.4
        transition[4, 7] = 0.3
        # 凭证访问 -> 发现, 横向移动
        transition[5, 6] = 0.6
        transition[5, 7] = 0.4
        # 发现 -> 横向移动, 收集
        transition[6, 7] = 0.5
        transition[6, 8] = 0.5
        # 横向移动 -> 执行, 发现, 收集
        transition[7, 1] = 0.3
        transition[7, 6] = 0.3
        transition[7, 8] = 0.4
        # 收集 -> 命令与控制, 数据窃取
        transition[8, 9] = 0.6
        transition[8, 10] = 0.4
        # 命令与控制 -> 数据窃取, 影响
        transition[9, 10] = 0.7
        transition[9, 11] = 0.3
        # 数据窃取 -> 影响
        transition[10, 11] = 1.0
        
        # 归一化
        row_sums = transition.sum(axis=1)
        transition = np.divide(transition, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis]!=0)
        
        return {stages[i]: {stages[j]: transition[i, j] for j in range(n_stages) if transition[i, j] > 0} 
                for i in range(n_stages)}
    
    def reconstruct_attack_chain(self, suspicious_events, timestamps, node_types, edge_index, edge_types=None):
        """
        重构攻击链
        
        Args:
            suspicious_events: 可疑事件列表
            timestamps: 时间戳
            node_types: 节点类型
            edge_index: 边索引
            edge_types: 边类型
            
        Returns:
            重构的攻击链
        """
        self.logger.info(f"开始重构攻击链，可疑事件数量: {len(suspicious_events)}")
        
        # 按时间排序事件
        sorted_events = sorted(suspicious_events, key=lambda x: timestamps[x['node_id']])
        
        # 构建事件图
        event_graph = self._build_event_graph(sorted_events, timestamps, node_types, edge_index, edge_types)
        
        # 识别攻击阶段
        events_with_stages = self._identify_attack_stages(sorted_events, event_graph)
        
        # 重构攻击链
        attack_chains = self._find_attack_chains(events_with_stages, event_graph)
        
        # 评分和过滤攻击链
        scored_chains = self._score_attack_chains(attack_chains)
        
        # 更新缓存
        self.attack_chains = scored_chains
        
        return scored_chains
    
    def _build_event_graph(self, events, timestamps, node_types, edge_index, edge_types=None):
        """
        构建事件图
        
        Args:
            events: 事件列表
            timestamps: 时间戳
            node_types: 节点类型
            edge_index: 边索引
            edge_types: 边类型
            
        Returns:
            事件图
        """
        G = nx.DiGraph()
        
        # 添加节点
        for event in events:
            node_id = event['node_id']
            G.add_node(
                node_id,
                timestamp=timestamps[node_id],
                node_type=node_types[node_id] if isinstance(node_types, dict) else node_types[int(node_id)],
                confidence=event.get('confidence', 0.5),
                **event
            )
        
        # 添加边
        if edge_index is not None:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if str(src) in G.nodes and str(dst) in G.nodes:
                    # 计算时间差
                    time_diff = abs(timestamps[str(dst)] - timestamps[str(src)])
                    edge_type = edge_types[i] if edge_types is not None else 'unknown'
                    
                    # 只添加符合时间顺序的边
                    if timestamps[str(dst)] >= timestamps[str(src)] and time_diff <= self.max_time_gap:
                        G.add_edge(
                            str(src), str(dst),
                            time_diff=time_diff,
                            edge_type=edge_type,
                            weight=1.0 / (1.0 + time_diff / 3600.0)  # 时间差越小权重越大
                        )
        
        # 添加时序边（按时间顺序连接事件）
        events_by_time = sorted(events, key=lambda x: timestamps[x['node_id']])
        for i in range(len(events_by_time) - 1):
            curr_event = events_by_time[i]
            next_event = events_by_time[i + 1]
            curr_id = curr_event['node_id']
            next_id = next_event['node_id']
            
            # 计算时间差
            time_diff = timestamps[next_id] - timestamps[curr_id]
            
            # 只添加时间差在阈值内的边
            if 0 <= time_diff <= self.max_time_gap:
                G.add_edge(
                    curr_id, next_id,
                    time_diff=time_diff,
                    edge_type='temporal',
                    weight=1.0 / (1.0 + time_diff / 3600.0)  # 时间差越小权重越大
                )
        
        return G
    
    def _identify_attack_stages(self, events, event_graph):
        """
        识别攻击阶段
        
        Args:
            events: 事件列表
            event_graph: 事件图
            
        Returns:
            带有攻击阶段的事件列表
        """
        events_with_stages = []
        
        # 攻击阶段映射
        stage_mapping = {
            'process': {
                'creation': 'execution',
                'termination': 'impact',
                'injection': 'defense_evasion',
                'scanning': 'discovery'
            },
            'file': {
                'creation': 'persistence',
                'modification': 'defense_evasion',
                'deletion': 'defense_evasion',
                'read': 'collection',
                'write': 'persistence'
            },
            'network': {
                'connection': 'command_and_control',
                'dns_query': 'command_and_control',
                'http_request': 'command_and_control',
                'download': 'initial_access',
                'upload': 'exfiltration'
            },
            'registry': {
                'creation': 'persistence',
                'modification': 'persistence',
                'deletion': 'defense_evasion',
                'read': 'discovery'
            },
            'authentication': {
                'success': 'credential_access',
                'failure': 'credential_access',
                'escalation': 'privilege_escalation'
            },
            'user': {
                'creation': 'persistence',
                'modification': 'privilege_escalation',
                'deletion': 'impact'
            }
        }
        
        # 为每个事件分配攻击阶段
        for event in events:
            node_id = event['node_id']
            node_data = event_graph.nodes[node_id]
            
            # 获取节点类型和事件类型
            node_type = node_data.get('node_type', '').lower()
            event_type = node_data.get('event_type', '').lower()
            
            # 根据节点类型和事件类型映射攻击阶段
            stage = None
            if node_type in stage_mapping and event_type in stage_mapping[node_type]:
                stage = stage_mapping[node_type][event_type]
            else:
                # 默认映射
                if 'scan' in event_type or 'enum' in event_type:
                    stage = 'discovery'
                elif 'exec' in event_type or 'run' in event_type:
                    stage = 'execution'
                elif 'connect' in event_type or 'comm' in event_type:
                    stage = 'command_and_control'
                elif 'access' in event_type or 'auth' in event_type:
                    stage = 'credential_access'
                elif 'persist' in event_type or 'startup' in event_type:
                    stage = 'persistence'
                elif 'escalate' in event_type or 'admin' in event_type:
                    stage = 'privilege_escalation'
                elif 'collect' in event_type or 'gather' in event_type:
                    stage = 'collection'
                elif 'exfil' in event_type or 'upload' in event_type:
                    stage = 'exfiltration'
                elif 'impact' in event_type or 'damage' in event_type:
                    stage = 'impact'
                elif 'lateral' in event_type or 'move' in event_type:
                    stage = 'lateral_movement'
                elif 'evade' in event_type or 'bypass' in event_type:
                    stage = 'defense_evasion'
                elif 'initial' in event_type or 'entry' in event_type:
                    stage = 'initial_access'
                else:
                    # 默认为发现阶段
                    stage = 'discovery'
            
            # 添加攻击阶段信息
            event_with_stage = event.copy()
            event_with_stage['attack_stage'] = stage
            events_with_stages.append(event_with_stage)
        
        return events_with_stages
    
    def _find_attack_chains(self, events_with_stages, event_graph):
        """
        查找攻击链
        
        Args:
            events_with_stages: 带有攻击阶段的事件列表
            event_graph: 事件图
            
        Returns:
            攻击链列表
        """
        attack_chains = []
        
        # 按时间排序事件
        sorted_events = sorted(events_with_stages, key=lambda x: event_graph.nodes[x['node_id']]['timestamp'])
        
        # 查找所有可能的起始点（初始访问或执行阶段）
        start_events = [
            event for event in sorted_events 
            if event['attack_stage'] in ['initial_access', 'execution']
        ]
        
        # 如果没有明确的起始点，使用时间最早的事件
        if not start_events and sorted_events:
            start_events = [sorted_events[0]]
        
        # 从每个起始点开始查找攻击链
        for start_event in start_events:
            start_id = start_event['node_id']
            
            # 使用BFS查找所有可能的攻击路径
            paths = self._find_attack_paths(start_id, event_graph, events_with_stages)
            
            # 过滤有效的攻击链
            valid_chains = []
            for path in paths:
                if len(path) >= self.min_chain_length:
                    # 提取路径上的事件和阶段
                    chain_events = []
                    chain_stages = []
                    for node_id in path:
                        event = next((e for e in events_with_stages if e['node_id'] == node_id), None)
                        if event:
                            chain_events.append(event)
                            chain_stages.append(event['attack_stage'])
                    
                    # 检查攻击链是否包含多个不同的攻击阶段
                    if len(set(chain_stages)) >= 2:
                        valid_chains.append({
                            'path': path,
                            'events': chain_events,
                            'stages': chain_stages,
                            'start_time': event_graph.nodes[path[0]]['timestamp'],
                            'end_time': event_graph.nodes[path[-1]]['timestamp'],
                            'duration': event_graph.nodes[path[-1]]['timestamp'] - event_graph.nodes[path[0]]['timestamp']
                        })
            
            # 添加有效的攻击链
            attack_chains.extend(valid_chains)
        
        # 按持续时间排序
        attack_chains.sort(key=lambda x: x['duration'], reverse=True)
        
        # 限制返回的攻击链数量
        return attack_chains[:self.max_paths]
    
    def _find_attack_paths(self, start_id, event_graph, events_with_stages):
        """
        查找攻击路径
        
        Args:
            start_id: 起始节点ID
            event_graph: 事件图
            events_with_stages: 带有攻击阶段的事件列表
            
        Returns:
            攻击路径列表
        """
        paths = []
        
        # 获取事件阶段映射
        event_stages = {event['node_id']: event['attack_stage'] for event in events_with_stages}
        
        # 使用DFS查找所有可能的路径
        def dfs(current_id, current_path, visited_stages):
            # 如果路径已经足够长，添加到结果中
            if len(current_path) >= self.min_chain_length and len(visited_stages) >= 2:
                paths.append(current_path.copy())
                
                # 如果路径太长，停止继续搜索
                if len(current_path) >= 10:
                    return
            
            # 获取所有可能的下一个节点
            neighbors = list(event_graph.successors(current_id))
            
            # 按照攻击阶段转换概率排序邻居节点
            current_stage = event_stages.get(current_id)
            if current_stage and current_stage in self.stage_transition:
                # 计算每个邻居的转换概率
                neighbor_probs = []
                for neighbor in neighbors:
                    neighbor_stage = event_stages.get(neighbor)
                    if neighbor_stage:
                        # 获取转换概率
                        prob = self.stage_transition[current_stage].get(neighbor_stage, 0.0)
                        neighbor_probs.append((neighbor, prob))
                    else:
                        neighbor_probs.append((neighbor, 0.0))
                
                # 按概率排序
                neighbors = [n for n, _ in sorted(neighbor_probs, key=lambda x: x[1], reverse=True)]
            
            # 遍历所有邻居
            for neighbor in neighbors:
                # 避免环路
                if neighbor not in current_path:
                    # 获取邻居的攻击阶段
                    neighbor_stage = event_stages.get(neighbor)
                    if neighbor_stage:
                        # 更新访问过的阶段
                        new_visited_stages = visited_stages.copy()
                        new_visited_stages.add(neighbor_stage)
                        
                        # 继续DFS
                        dfs(neighbor, current_path + [neighbor], new_visited_stages)
        
        # 从起始节点开始DFS
        start_stage = event_stages.get(start_id)
        if start_stage:
            dfs(start_id, [start_id], {start_stage})
        
        return paths
    
    def _score_attack_chains(self, attack_chains):
        """
        评分攻击链
        
        Args:
            attack_chains: 攻击链列表
            
        Returns:
            评分后的攻击链列表
        """
        scored_chains = []
        
        for chain in attack_chains:
            # 计算攻击链得分
            # 1. 阶段完整性得分：包含的不同攻击阶段数量
            stage_diversity = len(set(chain['stages'])) / 12.0  # 归一化
            
            # 2. 阶段连贯性得分：相邻阶段之间的转换概率
            stage_coherence = 0.0
            for i in range(len(chain['stages']) - 1):
                current_stage = chain['stages'][i]
                next_stage = chain['stages'][i + 1]
                if current_stage in self.stage_transition and next_stage in self.stage_transition[current_stage]:
                    stage_coherence += self.stage_transition[current_stage][next_stage]
            
            if len(chain['stages']) > 1:
                stage_coherence /= (len(chain['stages']) - 1)  # 归一化
            
            # 3. 时间合理性得分：事件时间间隔的合理性
            time_reasonability = 1.0
            for i in range(len(chain['path']) - 1):
                current_id = chain['path'][i]
                next_id = chain['path'][i + 1]
                time_diff = chain['events'][i + 1]['timestamp'] - chain['events'][i]['timestamp']
                
                # 时间间隔太长会降低得分
                if time_diff > self.max_time_gap:
                    time_reasonability *= 0.5
            
            # 4. 事件置信度得分：路径上事件的平均置信度
            confidence = sum(event.get('confidence', 0.5) for event in chain['events']) / len(chain['events'])
            
            # 5. 路径长度得分：路径长度的归一化得分
            path_length = min(1.0, len(chain['path']) / 10.0)  # 最多考虑10个节点
            
            # 综合得分
            score = (
                0.25 * stage_diversity +
                0.25 * stage_coherence +
                0.2 * time_reasonability +
                0.2 * confidence +
                0.1 * path_length
            )
            
            # 添加得分信息
            scored_chain = chain.copy()
            scored_chain['score'] = score
            scored_chain['stage_diversity'] = stage_diversity
            scored_chain['stage_coherence'] = stage_coherence
            scored_chain['time_reasonability'] = time_reasonability
            scored_chain['confidence'] = confidence
            scored_chain['path_length'] = path_length
            
            scored_chains.append(scored_chain)
        
        # 按得分排序
        scored_chains.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_chains


class AttackBehaviorAnalyzer:
    """
    攻击行为分析器
    
    分析攻击行为模式和特征
    """
    
    def __init__(self, config):
        """
        初始化攻击行为分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 攻击行为分析参数
        self.min_cluster_size = getattr(config, 'min_cluster_size', 3)
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        
        # 攻击行为特征
        self.behavior_features = self._init_behavior_features()
        
        # 攻击行为模式库
        self.behavior_patterns = {}
    
    def _init_behavior_features(self):
        """
        初始化攻击行为特征
        
        Returns:
            攻击行为特征字典
        """
        return {
            # 命令执行特征
            'command_execution': {
                'keywords': ['cmd.exe', 'powershell.exe', 'bash', 'sh', 'python', 'perl', 'exec', 'eval'],
                'patterns': [
                    r'cmd\.exe\s+/c\s+', 
                    r'powershell\.exe\s+-[eE][nN][cC][oO][dD][eE][dD][cC][oO][mM][mM][aA][nN][dD]',
                    r'bash\s+-c\s+',
                    r'eval\s*\(',
                    r'exec\s*\('
                ]
            },
            
            # 权限提升特征
            'privilege_escalation': {
                'keywords': ['sudo', 'su', 'runas', 'administrator', 'root', 'admin', 'uac'],
                'patterns': [
                    r'sudo\s+',
                    r'su\s+-\s*',
                    r'runas\s+/user:',
                    r'net\s+user\s+\S+\s+/add',
                    r'net\s+localgroup\s+administrators\s+\S+\s+/add'
                ]
            },
            
            # 数据窃取特征
            'data_exfiltration': {
                'keywords': ['upload', 'exfil', 'transfer', 'copy', 'zip', 'rar', 'tar', 'compress'],
                'patterns': [
                    r'(ftp|sftp|scp)\s+',
                    r'curl\s+--upload',
                    r'wget\s+--post-file',
                    r'(zip|rar|tar)\s+',
                    r'(cp|copy|xcopy)\s+'
                ]
            },
            
            # 持久化特征
            'persistence': {
                'keywords': ['startup', 'registry', 'cron', 'scheduled', 'service', 'daemon'],
                'patterns': [
                    r'reg\s+add\s+HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                    r'reg\s+add\s+HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                    r'schtasks\s+/create',
                    r'crontab\s+-e',
                    r'systemctl\s+enable'
                ]
            },
            
            # 防御规避特征
            'defense_evasion': {
                'keywords': ['disable', 'delete', 'clear', 'logs', 'history', 'firewall', 'antivirus'],
                'patterns': [
                    r'wevtutil\s+cl',
                    r'clear\s+-history',
                    r'rm\s+.*\.log',
                    r'del\s+.*\.log',
                    r'netsh\s+firewall\s+set\s+opmode\s+disable',
                    r'netsh\s+advfirewall\s+set\s+.*\s+state\s+off'
                ]
            },
            
            # 横向移动特征
            'lateral_movement': {
                'keywords': ['psexec', 'wmic', 'winrm', 'ssh', 'rdp', 'vnc'],
                'patterns': [
                    r'psexec\s+\\\\',
                    r'wmic\s+/node:',
                    r'winrm\s+quickconfig',
                    r'ssh\s+\S+@\S+',
                    r'mstsc\s+/v:'
                ]
            },
            
            # 命令与控制特征
            'command_and_control': {
                'keywords': ['connect', 'beacon', 'callback', 'http', 'https', 'dns', 'tcp', 'udp'],
                'patterns': [
                    r'(http|https)://\S+',
                    r'nc\s+-[el]',
                    r'ncat\s+',
                    r'certutil\s+-urlcache\s+-f',
                    r'bitsadmin\s+/transfer'
                ]
            }
        }
    
    def analyze_behavior(self, events, attack_chains=None):
        """
        分析攻击行为
        
        Args:
            events: 事件列表
            attack_chains: 攻击链列表
            
        Returns:
            攻击行为分析结果
        """
        self.logger.info(f"开始分析攻击行为，事件数量: {len(events)}")
        
        # 提取行为特征
        behavior_features = self._extract_behavior_features(events)
        
        # 聚类相似行为
        behavior_clusters = self._cluster_behaviors(behavior_features)
        
        # 识别攻击模式
        attack_patterns = self._identify_attack_patterns(behavior_clusters, attack_chains)
        
        # 生成行为分析报告
        behavior_report = self._generate_behavior_report(behavior_clusters, attack_patterns)
        
        return behavior_report
    
    def _extract_behavior_features(self, events):
        """
        提取行为特征
        
        Args:
            events: 事件列表
            
        Returns:
            行为特征列表
        """
        behavior_features = []
        
        for event in events:
            # 提取事件内容
            event_content = event.get('content', '')
            if not event_content and 'description' in event:
                event_content = event['description']
            
            if not event_content:
                continue
            
            # 检测行为类型
            behavior_types = []
            behavior_matches = {}
            
            for behavior_type, features in self.behavior_features.items():
                # 关键词匹配
                keyword_matches = [kw for kw in features['keywords'] if kw.lower() in event_content.lower()]
                
                # 正则表达式匹配
                pattern_matches = []
                for pattern in features['patterns']:
                    if re.search(pattern, event_content, re.IGNORECASE):
                        pattern_matches.append(pattern)
                
                # 如果有匹配，添加行为类型
                if keyword_matches or pattern_matches:
                    behavior_types.append(behavior_type)
                    behavior_matches[behavior_type] = {
                        'keywords': keyword_matches,
                        'patterns': pattern_matches
                    }
            
            # 如果没有匹配任何行为类型，使用默认类型
            if not behavior_types:
                behavior_types = ['unknown']
            
            # 创建行为特征
            behavior_feature = {
                'event_id': event.get('node_id', ''),
                'timestamp': event.get('timestamp', 0),
                'content': event_content,
                'behavior_types': behavior_types,
                'behavior_matches': behavior_matches,
                'confidence': event.get('confidence', 0.5)
            }
            
            behavior_features.append(behavior_feature)
        
        return behavior_features
    
    def _cluster_behaviors(self, behavior_features):
        """
        聚类相似行为
        
        Args:
            behavior_features: 行为特征列表
            
        Returns:
            行为聚类结果
        """
        if not behavior_features:
            return []
        
        # 计算行为之间的相似度矩阵
        n = len(behavior_features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 计算两个行为之间的相似度
                    similarity = self._calculate_behavior_similarity(
                        behavior_features[i], behavior_features[j]
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=1.0 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='precomputed'
        ).fit(1.0 - similarity_matrix)  # 将相似度转换为距离
        
        # 获取聚类标签
        labels = clustering.labels_
        
        # 整理聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # 忽略噪声点
                clusters[label].append(behavior_features[i])
        
        # 将聚类结果转换为列表
        cluster_list = []
        for label, behaviors in clusters.items():
            # 按时间排序
            behaviors.sort(key=lambda x: x['timestamp'])
            
            # 计算聚类的主要行为类型
            behavior_type_counts = Counter()
            for behavior in behaviors:
                for btype in behavior['behavior_types']:
                    behavior_type_counts[btype] += 1
            
            main_behavior_types = [btype for btype, count in behavior_type_counts.most_common(3)]
            
            cluster_list.append({
                'cluster_id': label,
                'behaviors': behaviors,
                'main_behavior_types': main_behavior_types,
                'size': len(behaviors),
                'start_time': behaviors[0]['timestamp'],
                'end_time': behaviors[-1]['timestamp']
            })
        
        # 按大小排序
        cluster_list.sort(key=lambda x: x['size'], reverse=True)
        
        return cluster_list
    
    def _calculate_behavior_similarity(self, behavior1, behavior2):
        """
        计算两个行为之间的相似度
        
        Args:
            behavior1: 第一个行为
            behavior2: 第二个行为
            
        Returns:
            相似度得分 (0-1)
        """
        # 1. 行为类型相似度
        type_set1 = set(behavior1['behavior_types'])
        type_set2 = set(behavior2['behavior_types'])
        
        if not type_set1 or not type_set2:
            type_similarity = 0.0
        else:
            type_similarity = len(type_set1.intersection(type_set2)) / max(len(type_set1), len(type_set2))
        
        # 2. 内容相似度 (简单实现，可以使用更复杂的文本相似度算法)
        content1 = behavior1['content'].lower()
        content2 = behavior2['content'].lower()
        
        # 分词
        words1 = set(re.findall(r'\w+', content1))
        words2 = set(re.findall(r'\w+', content2))
        
        if not words1 or not words2:
            content_similarity = 0.0
        else:
            content_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        # 3. 时间相似度
        time_diff = abs(behavior1['timestamp'] - behavior2['timestamp'])
        time_similarity = max(0.0, 1.0 - time_diff / (24 * 3600))  # 一天内的事件有相似性
        
        # 综合相似度
        similarity = 0.5 * type_similarity + 0.3 * content_similarity + 0.2 * time_similarity
        
        return similarity
    
    def _identify_attack_patterns(self, behavior_clusters, attack_chains=None):
        """
        识别攻击模式
        
        Args:
            behavior_clusters: 行为聚类结果
            attack_chains: 攻击链列表
            
        Returns:
            攻击模式列表
        """
        attack_patterns = []
        
        # 如果有攻击链信息，使用攻击链来识别模式
        if attack_chains:
            for chain in attack_chains:
                # 提取攻击链中的行为
                chain_behaviors = []
                for event in chain['events']:
                    event_id = event['node_id']
                    
                    # 查找对应的行为
                    for cluster in behavior_clusters:
                        for behavior in cluster['behaviors']:
                            if behavior['event_id'] == event_id:
                                chain_behaviors.append(behavior)
                                break
                
                if chain_behaviors:
                    # 识别攻击模式
                    pattern = {
                        'pattern_id': f"pattern_{len(attack_patterns)}",
                        'name': self._generate_pattern_name(chain),
                        'behaviors': chain_behaviors,
                        'attack_stages': chain['stages'],
                        'score': chain.get('score', 0.5),
                        'confidence': chain.get('confidence', 0.5),
                        'start_time': chain['start_time'],
                        'end_time': chain['end_time']
                    }
                    
                    attack_patterns.append(pattern)
        else:
            # 如果没有攻击链信息，直接使用行为聚类
            for i, cluster in enumerate(behavior_clusters):
                # 只考虑足够大的聚类
                if cluster['size'] >= self.min_cluster_size:
                    # 识别攻击模式
                    pattern = {
                        'pattern_id': f"pattern_{i}",
                        'name': self._generate_pattern_name_from_cluster(cluster),
                        'behaviors': cluster['behaviors'],
                        'behavior_types': cluster['main_behavior_types'],
                        'size': cluster['size'],
                        'confidence': sum(b.get('confidence', 0.5) for b in cluster['behaviors']) / cluster['size'],
                        'start_time': cluster['start_time'],
                        'end_time': cluster['end_time']
                    }
                    
                    attack_patterns.append(pattern)
        
        # 按置信度排序
        attack_patterns.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        return attack_patterns
    
    def _generate_pattern_name(self, chain):
        """
        生成攻击模式名称
        
        Args:
            chain: 攻击链
            
        Returns:
            攻击模式名称
        """
        # 获取攻击阶段
        stages = chain['stages']
        
        # 根据攻击阶段生成名称
        if 'initial_access' in stages and 'impact' in stages:
            return "完整攻击链"
        elif 'lateral_movement' in stages:
            return "横向移动攻击"
        elif 'exfiltration' in stages:
            return "数据窃取攻击"
        elif 'privilege_escalation' in stages:
            return "权限提升攻击"
        elif 'persistence' in stages:
            return "持久化攻击"
        elif 'defense_evasion' in stages:
            return "防御规避攻击"
        elif 'execution' in stages:
            return "命令执行攻击"
        else:
            return "未知攻击模式"
    
    def _generate_pattern_name_from_cluster(self, cluster):
        """
        从聚类生成攻击模式名称
        
        Args:
            cluster: 行为聚类
            
        Returns:
            攻击模式名称
        """
        # 获取主要行为类型
        behavior_types = cluster['main_behavior_types']
        
        # 根据行为类型生成名称
        if 'command_and_control' in behavior_types:
            return "命令与控制活动"
        elif 'lateral_movement' in behavior_types:
            return "横向移动活动"
        elif 'data_exfiltration' in behavior_types:
            return "数据窃取活动"
        elif 'privilege_escalation' in behavior_types:
            return "权限提升活动"
        elif 'persistence' in behavior_types:
            return "持久化活动"
        elif 'defense_evasion' in behavior_types:
            return "防御规避活动"
        elif 'command_execution' in behavior_types:
            return "命令执行活动"
        else:
            return "未知活动模式"
    
    def _generate_behavior_report(self, behavior_clusters, attack_patterns):
        """
        生成行为分析报告
        
        Args:
            behavior_clusters: 行为聚类结果
            attack_patterns: 攻击模式列表
            
        Returns:
            行为分析报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_behaviors': sum(cluster['size'] for cluster in behavior_clusters),
                'behavior_clusters': len(behavior_clusters),
                'attack_patterns': len(attack_patterns)
            },
            'behavior_clusters': behavior_clusters,
            'attack_patterns': attack_patterns,
            'recommendations': self._generate_recommendations(attack_patterns)
        }
        
        return report
    
    def _generate_recommendations(self, attack_patterns):
        """
        生成安全建议
        
        Args:
            attack_patterns: 攻击模式列表
            
        Returns:
            安全建议列表
        """
        recommendations = []
        
        # 根据攻击模式生成建议
        for pattern in attack_patterns:
            pattern_name = pattern['name']
            
            if "命令与控制" in pattern_name:
                recommendations.append({
                    'title': "阻断命令与控制通信",
                    'description': "检测并阻断与已知恶意域名和IP的通信，实施网络分段和出站流量过滤。",
                    'severity': "高",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "横向移动" in pattern_name:
                recommendations.append({
                    'title': "限制横向移动",
                    'description': "实施最小权限原则，网络分段，禁用不必要的远程服务，加强身份验证。",
                    'severity': "高",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "数据窃取" in pattern_name:
                recommendations.append({
                    'title': "防止数据泄露",
                    'description': "实施数据加密，数据泄露防护，监控异常数据传输，限制敏感数据访问。",
                    'severity': "高",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "权限提升" in pattern_name:
                recommendations.append({
                    'title': "防止权限提升",
                    'description': "保持系统和应用程序更新，实施应用程序白名单，限制管理员权限。",
                    'severity': "高",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "持久化" in pattern_name:
                recommendations.append({
                    'title': "防止持久化",
                    'description': "监控启动项和计划任务，实施应用程序白名单，定期扫描系统异常。",
                    'severity': "中",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "防御规避" in pattern_name:
                recommendations.append({
                    'title': "加强防御机制",
                    'description': "实施多层防御，保护安全工具和日志，使用行为分析检测规避技术。",
                    'severity': "中",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "命令执行" in pattern_name:
                recommendations.append({
                    'title': "防止恶意命令执行",
                    'description': "实施应用程序白名单，输入验证，限制脚本执行，监控命令行活动。",
                    'severity': "中",
                    'related_pattern': pattern['pattern_id']
                })
            
            elif "完整攻击链" in pattern_name:
                recommendations.append({
                    'title': "全面安全加固",
                    'description': "实施深度防御策略，加强边界防护，内部网络分段，提高检测和响应能力。",
                    'severity': "高",
                    'related_pattern': pattern['pattern_id']
                })
        
        # 去重
        unique_recommendations = []
        titles = set()
        for rec in recommendations:
            if rec['title'] not in titles:
                titles.add(rec['title'])
                unique_recommendations.append(rec)
        
        return unique_recommendations


class AttackDetector:
    """
    攻击检测器
    
    实现基于T-HGNN的高级攻击检测功能，包括:
    1. 基于ATT&CK框架的攻击模式识别
    2. 攻击链重构
    3. 攻击行为分析
    """
    
    def __init__(self, model, config):
        """
        初始化攻击检测器
        
        Args:
            model: 训练好的T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 攻击检测参数
        self.attack_threshold = getattr(config, 'attack_threshold', 0.7)
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.6)
        self.min_chain_length = getattr(config, 'min_chain_length', 3)
        
        # 攻击模式定义
        self.attack_patterns = self._define_attack_patterns()
        
        # 攻击检测历史
        self.detection_history = []
        
        # 攻击模式匹配器
        input_dim = getattr(config, 'node_embedding_dim', 128)
        self.pattern_matcher = AttackPatternMatcher(
            input_dim=input_dim,
            hidden_dim=input_dim,
            num_patterns=len(self.attack_patterns),
            dropout=0.2
        )
        
        # 攻击链重构器
        self.chain_reconstructor = AttackChainReconstructor(config)
        
        # 攻击行为分析器
        self.behavior_analyzer = AttackBehaviorAnalyzer(config)
        
    def _define_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        定义基于MITRE ATT&CK框架的攻击模式
        
        Returns:
            攻击模式字典
        """
        return {
            'apt_attack': {
                'name': 'APT攻击',
                'description': '高级持续性威胁攻击，通常包含多个攻击阶段',
                'stages': ['initial_access', 'execution', 'persistence', 'privilege_escalation', 
                          'defense_evasion', 'credential_access', 'discovery', 'lateral_movement',
                          'collection', 'command_and_control', 'exfiltration', 'impact'],
                'min_stages': 3,
                'confidence_threshold': 0.8,
                'severity': 'critical',
                'tactics': ['TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 
                           'TA0007', 'TA0008', 'TA0009', 'TA0010', 'TA0011', 'TA0040'],
                'techniques': {
                    'initial_access': ['T1566.001', 'T1566.002', 'T1078.003', 'T1078.004', 'T1189', 'T1190', 'T1133'],
                    'execution': ['T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1203', 'T1053.005', 'T1106', 'T1204'],
                    'persistence': ['T1098', 'T1136', 'T1078.003', 'T1547.001', 'T1546.001', 'T1543.003', 'T1053.005'],
                    'privilege_escalation': ['T1548.002', 'T1134', 'T1068', 'T1078.002', 'T1484.002', 'T1055'],
                    'defense_evasion': ['T1027', 'T1070.001', 'T1070.004', 'T1036.005', 'T1112', 'T1497', 'T1562.001'],
                    'credential_access': ['T1110.001', 'T1003.001', 'T1003.002', 'T1003.003', 'T1552.001', 'T1555.003', 'T1212'],
                    'discovery': ['T1087', 'T1082', 'T1083', 'T1018', 'T1046', 'T1033', 'T1049'],
                    'lateral_movement': ['T1021.001', 'T1021.002', 'T1021.003', 'T1091', 'T1072', 'T1210', 'T1534'],
                    'collection': ['T1560.001', 'T1560.002', 'T1213', 'T1005', 'T1039', 'T1025', 'T1114.001', 'T1114.002'],
                    'command_and_control': ['T1071.001', 'T1071.002', 'T1071.004', 'T1105', 'T1104', 'T1095', 'T1132'],
                    'exfiltration': ['T1048.001', 'T1048.002', 'T1048.003', 'T1041', 'T1011.001', 'T1052', 'T1567'],
                    'impact': ['T1485', 'T1486', 'T1489', 'T1529', 'T1565', 'T1490', 'T1491.001', 'T1491.002']
                },
                'indicators': [
                    'long_duration_attack', 'multiple_stages', 'sophisticated_tactics',
                    'persistent_access', 'data_exfiltration', 'command_control_communication'
                ]
            },
            'ransomware_attack': {
                'name': '勒索软件攻击',
                'description': '加密用户数据并要求支付赎金的攻击',
                'stages': ['initial_access', 'execution', 'persistence', 
                          'discovery', 'collection', 'impact'],
                'min_stages': 2,
                'confidence_threshold': 0.75,
                'severity': 'critical',
                'tactics': ['TA0001', 'TA0002', 'TA0003', 'TA0007', 'TA0009', 'TA0040'],
                'techniques': {
                    'initial_access': ['T1566.001', 'T1566.002', 'T1190', 'T1133'],
                    'execution': ['T1059.001', 'T1059.003', 'T1204'],
                    'persistence': ['T1547.001', 'T1546.001'],
                    'discovery': ['T1083', 'T1018', 'T1082'],
                    'collection': ['T1560.001', 'T1005', 'T1039'],
                    'impact': ['T1486', 'T1489', 'T1490', 'T1491.001']
                },
                'indicators': [
                    'file_encryption', 'ransom_note', 'data_destruction',
                    'rapid_file_modification', 'suspicious_process_creation'
                ]
            },
            'data_exfiltration': {
                'name': '数据窃取',
                'description': '窃取敏感数据并传输到外部服务器',
                'stages': ['discovery', 'collection', 'command_and_control', 'exfiltration'],
                'min_stages': 2,
                'confidence_threshold': 0.85,
                'severity': 'high',
                'tactics': ['TA0007', 'TA0009', 'TA0011', 'TA0010'],
                'techniques': {
                    'discovery': ['T1087', 'T1083', 'T1046', 'T1082'],
                    'collection': ['T1560.001', 'T1560.002', 'T1213', 'T1005', 'T1039'],
                    'command_and_control': ['T1071.001', 'T1071.002', 'T1105', 'T1132'],
                    'exfiltration': ['T1048.001', 'T1048.002', 'T1048.003', 'T1041', 'T1567']
                },
                'indicators': [
                    'large_data_transfer', 'encrypted_communication', 'unusual_network_traffic',
                    'data_compression', 'suspicious_file_upload', 'unusual_access_patterns'
                ]
            },
            'lateral_movement': {
                'name': '横向移动',
                'description': '攻击者在网络内部横向移动以获取更多访问权限',
                'stages': ['discovery', 'credential_access', 'lateral_movement', 'execution'],
                'min_stages': 2,
                'confidence_threshold': 0.8,
                'severity': 'high',
                'tactics': ['TA0007', 'TA0006', 'TA0008', 'TA0002'],
                'techniques': {
                    'discovery': ['T1087', 'T1018', 'T1046', 'T1082'],
                    'credential_access': ['T1110.001', 'T1003.001', 'T1003.002', 'T1552.001'],
                    'lateral_movement': ['T1021.001', 'T1021.002', 'T1021.003', 'T1091', 'T1210'],
                    'execution': ['T1059.001', 'T1059.003', 'T1053.005']
                },
                'indicators': [
                    'multiple_host_access', 'credential_abuse', 'network_scanning',
                    'remote_service_creation', 'unusual_network_connections'
                ]
            },
            'insider_threat': {
                'name': '内部威胁',
                'description': '内部人员滥用权限进行恶意活动',
                'stages': ['privilege_escalation', 'discovery', 'collection', 'exfiltration'],
                'min_stages': 2,
                'confidence_threshold': 0.85,
                'severity': 'high',
                'tactics': ['TA0004', 'TA0007', 'TA0009', 'TA0010'],
                'techniques': {
                    'privilege_escalation': ['T1078.002', 'T1078.003', 'T1484.002'],
                    'discovery': ['T1087', 'T1083', 'T1082', 'T1046'],
                    'collection': ['T1213', 'T1005', 'T1039', 'T1025'],
                    'exfiltration': ['T1048.001', 'T1048.002', 'T1052', 'T1567']
                },
                'indicators': [
                    'privilege_abuse', 'unusual_access_pattern', 'data_access_outside_hours',
                    'large_data_download', 'unauthorized_resource_access'
                ]
            },
            'supply_chain_attack': {
                'name': '供应链攻击',
                'description': '通过第三方供应商或软件更新渠道进行攻击',
                'stages': ['initial_access', 'execution', 'persistence', 'defense_evasion'],
                'min_stages': 2,
                'confidence_threshold': 0.9,
                'severity': 'critical',
                'tactics': ['TA0001', 'TA0002', 'TA0003', 'TA0005'],
                'techniques': {
                    'initial_access': ['T1195', 'T1195.001', 'T1195.002', 'T1195.003'],
                    'execution': ['T1059.001', 'T1059.003', 'T1204'],
                    'persistence': ['T1547.001', 'T1546.001', 'T1543.003'],
                    'discovery': ['T1087', 'T1083', 'T1082'],
                    'collection': ['T1560.001', 'T1005', 'T1039']
                },
                'indicators': [
                    'software_update_anomaly', 'third_party_compromise', 'unusual_software_behavior',
                    'suspicious_update_source', 'unexpected_network_connections'
                ]
            },
            'ddos_attack': {
                'name': 'DDoS攻击',
                'description': '分布式拒绝服务攻击',
                'stages': ['initial_access', 'execution', 'impact'],
                'min_stages': 1,
                'confidence_threshold': 0.6,
                'severity': 'medium',
                'tactics': ['TA0001', 'TA0002', 'TA0040'],
                'techniques': {
                    'initial_access': ['T1078.003', 'T1078.004'],
                    'execution': ['T1059.001', 'T1059.003'],
                    'impact': ['T1499.001', 'T1499.002', 'T1499.003']
                },
                'indicators': [
                    'high_traffic_volume', 'multiple_sources', 'service_unavailability',
                    'unusual_network_patterns', 'resource_exhaustion'
                ]
            },
            'phishing_attack': {
                'name': '钓鱼攻击',
                'description': '通过虚假信息获取凭据或执行恶意代码',
                'stages': ['initial_access', 'execution', 'credential_access'],
                'min_stages': 1,
                'confidence_threshold': 0.7,
                'severity': 'medium',
                'tactics': ['TA0001', 'TA0002', 'TA0006'],
                'techniques': {
                    'initial_access': ['T1566.001', 'T1566.002', 'T1566.003'],
                    'execution': ['T1059.001', 'T1059.003', 'T1204'],
                    'credential_access': ['T1110.001', 'T1552.001', 'T1555.003']
                },
                'indicators': [
                    'suspicious_email', 'fake_website', 'credential_harvesting',
                    'malicious_attachment', 'suspicious_url', 'unusual_login_attempts'
                ]
            }
        }
    
    def detect_attacks(self, hetero_data: HeteroData, 
                      embeddings: Dict[str, torch.Tensor],
                      anomalies: Optional[Dict[str, Any]] = None,
                      timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        检测攻击活动
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            timestamps: 时间戳字典
            
        Returns:
            攻击检测结果
        """
        self.logger.info("开始攻击检测")
        
        # 1. 检测可疑节点
        if anomalies and 'anomalous_nodes' in anomalies:
            # 使用异常检测结果
            anomalous_nodes = anomalies['anomalous_nodes']
            self.logger.info(f"使用异常检测结果，异常节点数: {len(anomalous_nodes) if isinstance(anomalous_nodes, list) else 'N/A'}")
            
            # 处理异常节点数据格式
            if isinstance(anomalous_nodes, dict):
                # 将字典格式转换为列表格式
                node_list = []
                for node_type, nodes in anomalous_nodes.items():
                    if isinstance(nodes, list):
                        for node in nodes:
                            if isinstance(node, dict):
                                node['node_type'] = node_type
                                node_list.append(node)
                            else:
                                node_list.append({
                                    'node_id': str(node),
                                    'node_type': node_type,
                                    'confidence': 0.5,
                                    'attack_stage': 'unknown',
                                    'description': f'异常节点: {node}'
                                })
                    elif isinstance(nodes, dict):
                        nodes['node_type'] = node_type
                        node_list.append(nodes)
                anomalous_nodes = node_list
            
            # 按节点类型分组
            node_groups = {}
            for node in anomalous_nodes:
                if isinstance(node, str):
                    node_info = {
                        'node_id': node,
                        'node_type': 'unknown',
                        'confidence': 0.5,
                        'attack_stage': 'unknown',
                        'description': f'异常节点: {node}'
                    }
                else:
                    node_info = node
                
                node_type = node_info.get('node_type', 'unknown')
                if node_type not in node_groups:
                    node_groups[node_type] = []
                node_groups[node_type].append(node_info)
            
            # 为每种节点类型创建攻击链
            attack_chains = []
            for node_type, nodes in node_groups.items():
                if len(nodes) >= self.min_chain_length:
                    # 按置信度排序
                    nodes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                    
                    # 创建攻击链
                    chain = {
                        'pattern_name': f'{node_type}_attack_chain',
                        'confidence': np.mean([n.get('confidence', 0) for n in nodes]),
                        'risk_score': np.mean([n.get('confidence', 0) for n in nodes]),
                        'attack_stages': [n.get('attack_stage', 'unknown') for n in nodes],
                        'path': [n.get('node_id', f'node_{i}') for i, n in enumerate(nodes)],
                        'timeline': [
                            {
                                'time': n.get('timestamp', f'09:{i:02d}:00'),
                                'event': n.get('description', f'{node_type}事件'),
                                'node': n.get('node_id', f'node_{i}'),
                                'timestamp': n.get('timestamp', f'2024-01-01T09:{i:02d}:00'),
                                'attack_stage': n.get('attack_stage', 'unknown'),
                                'confidence': n.get('confidence', 0),
                                'node_type': n.get('node_type', node_type)
                            }
                            for i, n in enumerate(nodes)
                        ],
                        'key_nodes': [
                            {
                                'node_id': n.get('node_id', f'node_{i}'),
                                'node_type': n.get('node_type', node_type),
                                'confidence': n.get('confidence', 0),
                                'attack_stage': n.get('attack_stage', 'unknown'),
                                'description': n.get('description', f'{node_type}关键节点')
                            }
                            for i, n in enumerate(nodes[:3])
                        ]
                    }
                    attack_chains.append(chain)
            
            suspicious_nodes = anomalous_nodes
            attack_patterns = []
        else:
            # 使用原有的检测逻辑
            suspicious_nodes = self._detect_suspicious_nodes(hetero_data, embeddings)
            attack_patterns = self._identify_attack_patterns(suspicious_nodes, timestamps)
            attack_chains = self._build_attack_chains(hetero_data, suspicious_nodes, attack_patterns)
        
        # 4. 评估攻击严重性
        attack_severity = self._assess_attack_severity(attack_chains)
        
        # 5. 生成检测报告
        detection_report = self._generate_detection_report(
            suspicious_nodes, attack_patterns, attack_chains, attack_severity
        )
        
        # 6. 更新检测历史
        self._update_detection_history(detection_report)
        
        self.logger.info(f"攻击检测完成，发现 {len(attack_chains)} 个攻击链")
        
        return {
            'suspicious_nodes': suspicious_nodes,
            'attack_patterns': attack_patterns,
            'attack_chains': attack_chains,
            'attack_severity': attack_severity,
            'detection_report': detection_report
        }
    
    def _detect_suspicious_nodes(self, hetero_data: HeteroData, 
                                embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        检测可疑节点
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            
        Returns:
            可疑节点信息
        """
        suspicious_nodes = {}
        
        for ntype in hetero_data.node_types:
            if ntype in embeddings and hetero_data[ntype].x is not None:
                node_embeddings = embeddings[ntype]
                
                # 使用模型预测节点类别
                with torch.no_grad():
                    predictions = self.model.predict(hetero_data)
                    if ntype in predictions:
                        node_predictions = predictions[ntype]
                        
                        # 计算可疑分数
                        if node_predictions.dim() > 1:
                            # 多分类情况
                            suspicious_scores = F.softmax(node_predictions, dim=-1)[:, 1]  # 假设1是恶意类别
                        else:
                            # 二分类情况
                            suspicious_scores = torch.sigmoid(node_predictions)
                        
                        # 识别可疑节点
                        suspicious_indices = torch.where(suspicious_scores > self.confidence_threshold)[0]
                        
                        if len(suspicious_indices) > 0:
                            suspicious_nodes[ntype] = {
                                'indices': suspicious_indices.tolist(),
                                'scores': suspicious_scores[suspicious_indices].tolist(),
                                'predictions': node_predictions[suspicious_indices].tolist()
                            }
        
        return suspicious_nodes
    
    def _identify_attack_patterns(self, suspicious_nodes: Dict[str, Any], 
                                 timestamps: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """
        识别攻击模式
        
        Args:
            suspicious_nodes: 可疑节点信息
            timestamps: 时间戳字典
            
        Returns:
            攻击模式列表
        """
        attack_patterns = []
        
        # 获取所有可疑节点的攻击阶段
        all_stages = set()
        node_stages = {}
        
        # 处理不同格式的可疑节点数据
        if isinstance(suspicious_nodes, dict):
            # 如果是字典格式，可能有两种情况
            if 'anomalous_nodes' in suspicious_nodes:
                # 来自异常检测器的结果
                for node_type, nodes in suspicious_nodes['anomalous_nodes'].items():
                    for node in nodes:
                        if 'attack_stage' in node:
                            stage = node['attack_stage']
                            all_stages.add(stage)
                            node_id = node.get('node_id', '')
                            node_stages[node_id] = {
                                'stage': stage,
                                'confidence': node.get('anomaly_score', 0.5),
                                'timestamp': node.get('timestamp', datetime.now().isoformat()),
                                'node_type': node_type
                            }
            else:
                # 直接的节点类型到节点列表的映射
                for node_type, nodes in suspicious_nodes.items():
                    if isinstance(nodes, list):
                        for node in nodes:
                            if isinstance(node, dict) and 'attack_stage' in node:
                                stage = node['attack_stage']
                                all_stages.add(stage)
                                node_id = node.get('node_id', '')
                                node_stages[node_id] = {
                                    'stage': stage,
                                    'confidence': node.get('anomaly_score', 0.5),
                                    'timestamp': node.get('timestamp', datetime.now().isoformat()),
                                    'node_type': node_type
                                }
        
        self.logger.info(f"识别到的攻击阶段: {all_stages}")
        
        # 分析每种攻击模式
        for pattern_name, pattern_config in self.attack_patterns.items():
            # 获取模式所需的阶段
            required_stages = set(pattern_config['stages'])
            min_stages = pattern_config['min_stages']
            
            # 检查检测到的阶段与模式所需阶段的交集
            detected_stages = all_stages.intersection(required_stages)
            
            # 如果检测到足够多的阶段，认为匹配该攻击模式
            if len(detected_stages) >= min_stages:
                # 计算模式匹配的置信度
                confidence = self._calculate_pattern_confidence(detected_stages, required_stages, node_stages)
                
                # 如果置信度超过阈值，添加到检测结果
                if confidence >= pattern_config.get('confidence_threshold', 0.5):
                    # 获取相关的战术和技术
                    tactics = pattern_config.get('tactics', [])
                    techniques = {}
                    for stage in detected_stages:
                        stage_techniques = pattern_config.get('techniques', {}).get(stage, [])
                        if stage_techniques:
                            techniques[stage] = stage_techniques
                    
                    # 创建攻击模式对象
                    attack_pattern = {
                        'pattern_name': pattern_name,
                        'pattern_display_name': pattern_config.get('name', pattern_name),
                        'description': pattern_config.get('description', ''),
                        'confidence': confidence,
                        'severity': pattern_config.get('severity', 'medium'),
                        'stages_detected': list(detected_stages),
                        'stages_total': list(required_stages),
                        'completeness': len(detected_stages) / len(required_stages),
                        'tactics': tactics,
                        'techniques': techniques,
                        'timestamp': datetime.now().isoformat()
                    }
                    attack_patterns.append(attack_pattern)
        
        # 按置信度排序
        attack_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.logger.info(f"识别到 {len(attack_patterns)} 种攻击模式")
        return attack_patterns
    
    def _calculate_pattern_confidence(self, detected_stages: set, required_stages: set, 
                                    node_stages: Dict[str, Dict[str, Any]]) -> float:
        """
        计算攻击模式的置信度
        
        Args:
            detected_stages: 检测到的攻击阶段
            required_stages: 模式所需的攻击阶段
            node_stages: 节点的攻击阶段信息
            
        Returns:
            置信度分数
        """
        # 计算阶段匹配率
        stage_match_ratio = len(detected_stages) / len(required_stages)
        
        # 计算节点置信度的平均值
        node_confidences = []
        for node_id, info in node_stages.items():
            if info['stage'] in detected_stages:
                node_confidences.append(info['confidence'])
        
        avg_node_confidence = np.mean(node_confidences) if node_confidences else 0.5
        
        # 综合计算置信度
        confidence = 0.6 * stage_match_ratio + 0.4 * avg_node_confidence
        
        return confidence
    
    def _build_attack_chain(self, hetero_data: HeteroData, suspicious_nodes: Dict[str, Any], 
                           attack_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建攻击链
        
        Args:
            hetero_data: 异构图数据
            suspicious_nodes: 可疑节点信息
            attack_patterns: 攻击模式列表
            
        Returns:
            攻击链列表
        """
        attack_chains = []
        
        # 为每个攻击模式构建攻击链
        for pattern in attack_patterns:
            # 初始化攻击链
            chain = {
                'pattern_name': pattern['pattern_name'],
                'confidence': pattern['confidence'],
                'risk_score': 0.0,  # 初始化风险分数
                'attack_stages': pattern['stages_detected'],
                'path': [],  # 攻击路径
                'timeline': [],  # 攻击时间线
                'key_nodes': []  # 关键节点
            }
            
            # 处理不同格式的可疑节点数据
            if isinstance(suspicious_nodes, list):
                # 如果是节点列表，直接使用
                nodes_list = suspicious_nodes
                # 按时间排序
                nodes_list.sort(key=lambda x: x.get('timestamp', ''), reverse=False)
                
                # 构建路径和时间线
                for node in nodes_list:
                    node_id = node.get('node_id', '')
                    if node_id:
                        chain['path'].append(node_id)
                        
                        # 添加到时间线
                        chain['timeline'].append({
                            'time': node.get('timestamp', ''),
                            'event': node.get('description', '未知事件'),
                            'node': node_id,
                            'timestamp': node.get('timestamp', ''),
                            'attack_stage': node.get('attack_stage', 'unknown'),
                            'confidence': node.get('confidence', 0.0),
                            'node_type': node.get('node_type', 'unknown')
                        })
                        
                        # 添加关键节点（选择置信度最高的前3个）
                        if len(chain['key_nodes']) < 3 or node.get('confidence', 0.0) > min([k.get('confidence', 0.0) for k in chain['key_nodes']]):
                            key_node = {
                                'node_id': node_id,
                                'node_type': node.get('node_type', 'unknown'),
                                'confidence': node.get('confidence', 0.0),
                                'attack_stage': node.get('attack_stage', 'unknown'),
                                'description': node.get('description', '未知节点')
                            }
                            chain['key_nodes'].append(key_node)
                            # 保持最多3个关键节点，移除置信度最低的
                            if len(chain['key_nodes']) > 3:
                                chain['key_nodes'].sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
                                chain['key_nodes'] = chain['key_nodes'][:3]
            
            # 计算攻击链风险分数
            chain['risk_score'] = self._calculate_chain_risk_score(chain)
            
            attack_chains.append(chain)
        
        return attack_chains
    
    def _check_attack_pattern(self, suspicious_nodes: Dict[str, Any], 
                             pattern_config: Dict[str, Any], 
                             timestamps: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """
        检查是否满足攻击模式
        
        Args:
            suspicious_nodes: 可疑节点信息
            pattern_config: 攻击模式配置
            timestamps: 时间戳字典
            
        Returns:
            是否满足攻击模式
        """
        # 完整的攻击模式匹配实现
        min_stages = pattern_config.get('min_stages', 2)
        min_nodes_per_stage = pattern_config.get('min_nodes_per_stage', 1)
        time_window = pattern_config.get('time_window', 3600)  # 1小时
        
        # 检查每个攻击阶段是否有足够的可疑节点
        stage_counts = {}
        total_suspicious = 0
        
        for stage, nodes_info in suspicious_nodes.items():
            if isinstance(nodes_info, dict) and 'indices' in nodes_info:
                node_count = len(nodes_info['indices'])
                stage_counts[stage] = node_count
                total_suspicious += node_count
            else:
                stage_counts[stage] = 0
        
        # 检查是否满足最小阶段数要求
        active_stages = sum(1 for count in stage_counts.values() if count >= min_nodes_per_stage)
        if active_stages < min_stages:
            return False
        
        # 检查时间窗口内的活动
        if timestamps:
            current_time = max(timestamps.values()) if timestamps.values() else 0
            recent_activity = sum(1 for ts in timestamps.values() 
                                if current_time - ts <= time_window)
            if recent_activity < min_stages:
                return False
        
        # 检查攻击链的连续性
        if len(stage_counts) > 1:
            # 检查是否有足够的阶段转换
            stage_transitions = 0
            stage_list = list(stage_counts.keys())
            for i in range(len(stage_list) - 1):
                if stage_counts[stage_list[i]] > 0 and stage_counts[stage_list[i+1]] > 0:
                    stage_transitions += 1
            
            # 至少需要一定的阶段转换来构成攻击链
            min_transitions = min_stages - 1
            if stage_transitions < min_transitions:
                return False
        
        return True
    
    def _calculate_pattern_confidence(self, suspicious_nodes: Dict[str, Any], 
                                    pattern_config: Dict[str, Any]) -> float:
        """
        计算攻击模式置信度
        
        Args:
            suspicious_nodes: 可疑节点信息
            pattern_config: 攻击模式配置
            
        Returns:
            置信度分数
        """
        if not suspicious_nodes:
            return 0.0
        
        # 计算平均可疑分数
        all_scores = []
        for nodes in suspicious_nodes.values():
            all_scores.extend(nodes['scores'])
        
        if all_scores:
            avg_confidence = np.mean(all_scores)
            return min(avg_confidence, 1.0)
        
        return 0.0
    
    def _get_detected_stages(self, suspicious_nodes: Dict[str, Any], 
                            pattern_config: Dict[str, Any]) -> List[str]:
        """
        获取检测到的攻击阶段
        
        Args:
            suspicious_nodes: 可疑节点信息
            pattern_config: 攻击模式配置
            
        Returns:
            检测到的攻击阶段列表
        """
        # 完整的攻击阶段推断实现
        detected_stages = []
        stage_confidence = {}
        
        # 定义攻击阶段的关键词和权重
        stage_keywords = {
            'initial_access': {
                'keywords': ['email', 'phishing', 'malware', 'download', 'attachment', 'url'],
                'weight': 1.0
            },
            'execution': {
                'keywords': ['command', 'execution', 'process', 'script', 'powershell', 'cmd'],
                'weight': 1.0
            },
            'persistence': {
                'keywords': ['registry', 'persistence', 'service', 'scheduled', 'startup'],
                'weight': 1.0
            },
            'privilege_escalation': {
                'keywords': ['privilege', 'escalation', 'admin', 'root', 'sudo'],
                'weight': 1.0
            },
            'defense_evasion': {
                'keywords': ['evasion', 'bypass', 'disable', 'firewall', 'antivirus'],
                'weight': 1.0
            },
            'credential_access': {
                'keywords': ['credential', 'password', 'hash', 'token', 'key'],
                'weight': 1.0
            },
            'discovery': {
                'keywords': ['discovery', 'scan', 'enumeration', 'reconnaissance'],
                'weight': 1.0
            },
            'lateral_movement': {
                'keywords': ['network', 'lateral', 'remote', 'ssh', 'rdp', 'smb'],
                'weight': 1.0
            },
            'collection': {
                'keywords': ['collection', 'gather', 'collect', 'harvest'],
                'weight': 1.0
            },
            'exfiltration': {
                'keywords': ['data', 'exfiltration', 'upload', 'transfer', 'exfil'],
                'weight': 1.0
            },
            'command_and_control': {
                'keywords': ['c2', 'command', 'control', 'beacon', 'callback'],
                'weight': 1.0
            },
            'impact': {
                'keywords': ['impact', 'destruction', 'encryption', 'ransomware'],
                'weight': 1.0
            }
        }
        
        # 分析每个可疑节点类型
        for ntype, nodes_info in suspicious_nodes.items():
            if isinstance(nodes_info, dict) and 'indices' in nodes_info:
                node_count = len(nodes_info['indices'])
                if node_count == 0:
                    continue
                
                # 计算每个攻击阶段的匹配分数
                for stage, config in stage_keywords.items():
                    score = 0
                    keywords = config['keywords']
                    weight = config['weight']
                    
                    # 检查关键词匹配
                    for keyword in keywords:
                        if keyword in ntype.lower():
                            score += weight
                    
                    # 考虑节点数量
                    if score > 0:
                        score *= min(node_count / 10.0, 1.0)  # 标准化节点数量
                        
                        if stage not in stage_confidence:
                            stage_confidence[stage] = 0
                        stage_confidence[stage] += score
        
        # 选择置信度最高的攻击阶段
        threshold = 0.3  # 置信度阈值
        for stage, confidence in stage_confidence.items():
            if confidence >= threshold:
                detected_stages.append(stage)
        
        # 如果没有检测到任何阶段，使用简化的关键词匹配作为后备
        if not detected_stages:
            for ntype in suspicious_nodes.keys():
                if 'email' in ntype or 'phishing' in ntype:
                    detected_stages.append('initial_access')
                elif 'command' in ntype or 'execution' in ntype:
                    detected_stages.append('execution')
                elif 'registry' in ntype or 'persistence' in ntype:
                    detected_stages.append('persistence')
                elif 'network' in ntype or 'lateral' in ntype:
                    detected_stages.append('lateral_movement')
                elif 'data' in ntype or 'exfiltration' in ntype:
                    detected_stages.append('exfiltration')
        
        return list(set(detected_stages))
    
    def _build_attack_chains(self, hetero_data: HeteroData, 
                           suspicious_nodes: Dict[str, Any], 
                           attack_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建攻击链
        
        Args:
            hetero_data: 异构图数据
            suspicious_nodes: 可疑节点信息
            attack_patterns: 攻击模式列表
            
        Returns:
            攻击链列表
        """
        attack_chains = []
        
        for pattern in attack_patterns:
            # 为每个攻击模式构建攻击链
            chain = {
                'pattern_name': pattern['pattern_name'],
                'confidence': pattern['confidence'],
                'risk_score': 0.0,  # 初始化风险分数
                'attack_stages': pattern['stages_detected'],
                'path': [],  # 攻击路径
                'timeline': [],  # 攻击时间线
                'key_nodes': []  # 关键节点
            }
            
            # 构建攻击路径和时间线
            if isinstance(suspicious_nodes, list):
                # 如果是节点列表，直接使用
                nodes_list = suspicious_nodes
                # 按时间排序
                nodes_list.sort(key=lambda x: x.get('timestamp', ''), reverse=False)
                
                # 构建路径和时间线
                for node in nodes_list:
                    node_id = node.get('node_id', '')
                    if node_id:
                        chain['path'].append(node_id)
                        
                        # 添加到时间线
                        chain['timeline'].append({
                            'time': node.get('timestamp', ''),
                            'event': node.get('description', '未知事件'),
                            'node': node_id,
                            'timestamp': node.get('timestamp', ''),
                            'attack_stage': node.get('attack_stage', 'unknown'),
                            'confidence': node.get('confidence', 0.0),
                            'node_type': node.get('node_type', 'unknown')
                        })
                        
                        # 添加关键节点（选择置信度最高的前3个）
                        if len(chain['key_nodes']) < 3 or node.get('confidence', 0.0) > min([k.get('confidence', 0.0) for k in chain['key_nodes']]):
                            key_node = {
                                'node_id': node_id,
                                'node_type': node.get('node_type', 'unknown'),
                                'confidence': node.get('confidence', 0.0),
                                'attack_stage': node.get('attack_stage', 'unknown'),
                                'description': node.get('description', '未知节点')
                            }
                            chain['key_nodes'].append(key_node)
                            # 保持最多3个关键节点，移除置信度最低的
                            if len(chain['key_nodes']) > 3:
                                chain['key_nodes'].sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
                                chain['key_nodes'] = chain['key_nodes'][:3]
            
            # 计算攻击链风险分数
            chain['risk_score'] = self._calculate_chain_risk_score(chain)
            
            attack_chains.append(chain)
        
        return attack_chains
    
    def _calculate_chain_completeness(self, chain: Dict[str, Any]) -> float:
        """
        计算攻击链完整性
        
        Args:
            chain: 攻击链信息
            
        Returns:
            完整性分数
        """
        pattern_name = chain['pattern_name']
        if pattern_name in self.attack_patterns:
            expected_stages = self.attack_patterns[pattern_name]['stages']
            detected_stages = chain['stages']
            
            completeness = len(detected_stages) / len(expected_stages)
            return min(completeness, 1.0)
        
        return 0.0
    
    def _calculate_chain_risk_score(self, chain: Dict[str, Any]) -> float:
        """
        计算攻击链风险分数
        
        Args:
            chain: 攻击链信息
            
        Returns:
            风险分数
        """
        # 基于置信度、完整性和节点数量计算风险分数
        confidence = chain.get('confidence', 0.5)
        
        # 计算完整性分数
        attack_stages = chain.get('attack_stages', [])
        pattern_name = chain.get('pattern_name', '')
        
        if pattern_name in self.attack_patterns:
            expected_stages = self.attack_patterns[pattern_name].get('stages', [])
            completeness = len(attack_stages) / max(len(expected_stages), 1) if expected_stages else 0.5
        else:
            completeness = 0.5
        
        # 计算节点数量分数
        path_length = len(chain.get('path', []))
        node_score = min(path_length / 10.0, 1.0)  # 归一化到0-1
        
        # 综合风险分数
        risk_score = (confidence * 0.4 + completeness * 0.4 + node_score * 0.2)
        
        return min(risk_score, 1.0)
    
    def _assess_attack_severity(self, attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估攻击严重性
        
        Args:
            attack_chains: 攻击链列表
            
        Returns:
            严重性评估结果
        """
        if not attack_chains:
            return {'level': 'none', 'score': 0.0, 'description': '未检测到攻击活动'}
        
        # 计算最高风险分数
        max_risk_score = max(chain['risk_score'] for chain in attack_chains)
        
        # 计算平均风险分数
        avg_risk_score = np.mean([chain['risk_score'] for chain in attack_chains])
        
        # 计算攻击链数量分数
        chain_count_score = min(len(attack_chains) / 5.0, 1.0)
        
        # 综合严重性分数
        severity_score = (max_risk_score * 0.5 + avg_risk_score * 0.3 + chain_count_score * 0.2)
        
        # 确定严重性等级
        if severity_score > 0.8:
            level = 'critical'
            description = '检测到严重攻击活动，需要立即响应'
        elif severity_score > 0.6:
            level = 'high'
            description = '检测到高风险攻击活动，需要密切监控'
        elif severity_score > 0.4:
            level = 'medium'
            description = '检测到中等风险攻击活动，需要关注'
        elif severity_score > 0.2:
            level = 'low'
            description = '检测到低风险攻击活动，需要监控'
        else:
            level = 'minimal'
            description = '检测到轻微攻击活动'
        
        return {
            'level': level,
            'score': severity_score,
            'description': description,
            'max_risk_score': max_risk_score,
            'avg_risk_score': avg_risk_score,
            'chain_count': len(attack_chains)
        }
    
    def _generate_detection_report(self, suspicious_nodes: Dict[str, Any], 
                                 attack_patterns: List[Dict[str, Any]], 
                                 attack_chains: List[Dict[str, Any]], 
                                 attack_severity: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成检测报告
        
        Args:
            suspicious_nodes: 可疑节点信息
            attack_patterns: 攻击模式列表
            attack_chains: 攻击链列表
            attack_severity: 攻击严重性
            
        Returns:
            检测报告
        """
        # 处理suspicious_nodes格式
        if isinstance(suspicious_nodes, list):
            total_suspicious_nodes = len(suspicious_nodes)
        elif isinstance(suspicious_nodes, dict):
            total_suspicious_nodes = sum(len(nodes['indices']) for nodes in suspicious_nodes.values())
        else:
            total_suspicious_nodes = 0
        
        report = {
            'detection_time': datetime.now().isoformat(),
            'summary': {
                'total_suspicious_nodes': total_suspicious_nodes,
                'attack_patterns_detected': len(attack_patterns),
                'attack_chains_detected': len(attack_chains),
                'severity_level': attack_severity.get('level', 'unknown'),
                'severity_score': attack_severity.get('score', 0.0)
            },
            'suspicious_nodes': suspicious_nodes,
            'attack_patterns': attack_patterns,
            'attack_chains': attack_chains,
            'severity_assessment': attack_severity,
            'recommendations': self._generate_attack_recommendations(attack_severity, attack_chains)
        }
        
        return report
    
    def _generate_attack_recommendations(self, attack_severity: Dict[str, Any], 
                                       attack_chains: List[Dict[str, Any]]) -> List[str]:
        """
        生成攻击响应建议
        
        Args:
            attack_severity: 攻击严重性
            attack_chains: 攻击链列表
            
        Returns:
            建议列表
        """
        recommendations = []
        
        severity_level = attack_severity['level']
        
        if severity_level == 'critical':
            recommendations.extend([
                "立即启动应急响应程序",
                "隔离所有受影响系统",
                "通知安全团队和高级管理层",
                "收集和保存证据",
                "联系执法部门（如需要）"
            ])
        elif severity_level == 'high':
            recommendations.extend([
                "立即隔离可疑节点",
                "加强监控和日志记录",
                "通知安全团队",
                "分析攻击向量和影响范围"
            ])
        elif severity_level == 'medium':
            recommendations.extend([
                "密切监控可疑活动",
                "加强安全防护措施",
                "分析攻击模式",
                "更新安全策略"
            ])
        else:
            recommendations.extend([
                "继续监控系统状态",
                "定期检查安全日志",
                "保持安全更新"
            ])
        
        # 基于攻击链类型添加特定建议
        for chain in attack_chains:
            pattern_name = chain['pattern_name']
            if pattern_name == 'apt_attack':
                recommendations.append("检测到APT攻击，需要长期监控和深度分析")
            elif pattern_name == 'data_exfiltration':
                recommendations.append("检测到数据外泄，立即检查数据完整性")
            elif pattern_name == 'lateral_movement':
                recommendations.append("检测到横向移动，检查网络分段和访问控制")
        
        return list(set(recommendations))  # 去重
    
    def _update_detection_history(self, detection_report: Dict[str, Any]):
        """
        更新检测历史
        
        Args:
            detection_report: 检测报告
        """
        self.detection_history.append(detection_report)
        
        # 保持历史记录在合理范围内
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            统计信息
        """
        if not self.detection_history:
            return {'total_detections': 0, 'average_severity': 0.0}
        
        total_detections = len(self.detection_history)
        severity_scores = [report['severity_assessment']['score'] for report in self.detection_history]
        average_severity = np.mean(severity_scores)
        
        # 统计严重性等级分布
        severity_distribution = {}
        for report in self.detection_history:
            level = report['severity_assessment']['level']
            severity_distribution[level] = severity_distribution.get(level, 0) + 1
        
        return {
            'total_detections': total_detections,
            'average_severity': average_severity,
            'severity_distribution': severity_distribution,
            'recent_detections': self.detection_history[-5:]  # 最近5次检测
        }
