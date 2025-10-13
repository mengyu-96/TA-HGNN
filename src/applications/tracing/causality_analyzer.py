"""
因果分析器

实现攻击行为之间的因果关系分析，确保信息只能从过去流向未来
这是大纲中提到的"理解因果"的核心实现
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import heapq
try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    from sklearn.metrics import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class CausalityAnalyzer:
    """
    因果分析器
    
    负责：
    1. 分析攻击行为之间的因果关系
    2. 确保时序约束（信息只能从过去流向未来）
    3. 识别攻击链中的关键因果节点
    4. 量化因果强度
    """
    
    def __init__(self, config):
        """
        初始化因果分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 因果分析参数
        self.causality_threshold = getattr(config, 'causality_threshold', 0.3)
        self.temporal_window = getattr(config, 'temporal_window', 3600)  # 1小时
        self.max_lag = getattr(config, 'max_lag', 300)  # 5分钟
        
        # 因果模式定义
        self.causal_patterns = self._define_causal_patterns()
        
    def _define_causal_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """定义因果模式"""
        return {
            'temporal_causality': [
                {'pattern': 'A -> B', 'description': 'A在时间上先于B发生'},
                {'pattern': 'A -> B -> C', 'description': 'A导致B，B导致C'},
                {'pattern': 'A -> (B, C)', 'description': 'A同时导致B和C'}
            ],
            'logical_causality': [
                {'pattern': 'execution -> persistence', 'description': '执行导致持久化'},
                {'pattern': 'persistence -> privilege_escalation', 'description': '持久化导致权限提升'},
                {'pattern': 'privilege_escalation -> lateral_movement', 'description': '权限提升导致横向移动'},
                {'pattern': 'lateral_movement -> data_collection', 'description': '横向移动导致数据收集'},
                {'pattern': 'data_collection -> exfiltration', 'description': '数据收集导致数据外泄'}
            ],
            'resource_causality': [
                {'pattern': 'file_access -> process_creation', 'description': '文件访问导致进程创建'},
                {'pattern': 'network_connection -> data_transfer', 'description': '网络连接导致数据传输'},
                {'pattern': 'user_login -> resource_access', 'description': '用户登录导致资源访问'}
            ]
        }
    
    def analyze_causality(self, hetero_data: HeteroData, 
                         attack_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析攻击路径中的因果关系
        
        Args:
            hetero_data: 异构图数据
            attack_path: 攻击路径
            
        Returns:
            因果分析结果
        """
        self.logger.info("开始因果分析")
        
        # 1. 提取时序信息
        temporal_info = self._extract_temporal_info(attack_path)
        
        # 2. 分析时序因果关系
        temporal_causality = self._analyze_temporal_causality(temporal_info)
        
        # 3. 分析逻辑因果关系
        logical_causality = self._analyze_logical_causality(attack_path)
        
        # 4. 分析资源因果关系
        resource_causality = self._analyze_resource_causality(hetero_data, attack_path)
        
        # 5. 识别关键因果节点
        key_causal_nodes = self._identify_key_causal_nodes(
            temporal_causality, logical_causality, resource_causality
        )
        
        # 6. 生成因果图
        causal_graph = self._build_causal_graph(
            temporal_causality, logical_causality, resource_causality
        )
        
        # 7. 计算因果强度
        causality_strength = self._calculate_causality_strength(causal_graph)
        
        self.logger.info(f"因果分析完成，识别出 {len(key_causal_nodes)} 个关键因果节点")
        
        return {
            'temporal_causality': temporal_causality,
            'logical_causality': logical_causality,
            'resource_causality': resource_causality,
            'key_causal_nodes': key_causal_nodes,
            'causal_graph': causal_graph,
            'causality_strength': causality_strength,
            'summary': self._generate_causality_summary(
                temporal_causality, logical_causality, resource_causality
            )
        }
    
    def _extract_temporal_info(self, attack_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取时序信息
        
        Args:
            attack_path: 攻击路径
            
        Returns:
            时序信息
        """
        temporal_info = {
            'nodes': [],
            'timestamps': [],
            'durations': [],
            'intervals': []
        }
        
        if 'path' not in attack_path or 'timestamps' not in attack_path:
            return temporal_info
        
        path = attack_path['path']
        timestamps = attack_path['timestamps']
        
        for i, (node, timestamp) in enumerate(zip(path, timestamps)):
            temporal_info['nodes'].append(node)
            temporal_info['timestamps'].append(timestamp)
            
            # 计算持续时间
            if i > 0:
                duration = (timestamp - timestamps[i-1]).total_seconds()
                temporal_info['durations'].append(duration)
                
                # 计算时间间隔
                if i < len(timestamps) - 1:
                    interval = (timestamps[i+1] - timestamp).total_seconds()
                    temporal_info['intervals'].append(interval)
        
        return temporal_info
    
    def _analyze_temporal_causality(self, temporal_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析时序因果关系
        
        Args:
            temporal_info: 时序信息
            
        Returns:
            时序因果关系
        """
        temporal_causality = {
            'causal_pairs': [],
            'temporal_ordering': [],
            'causality_scores': []
        }
        
        if len(temporal_info['nodes']) < 2:
            return temporal_causality
        
        # 分析相邻节点对的因果关系
        for i in range(len(temporal_info['nodes']) - 1):
            current_node = temporal_info['nodes'][i]
            next_node = temporal_info['nodes'][i + 1]
            current_time = temporal_info['timestamps'][i]
            next_time = temporal_info['timestamps'][i + 1]
            
            # 检查时序约束
            time_diff = (next_time - current_time).total_seconds()
            if 0 <= time_diff <= self.temporal_window:
                # 计算因果强度
                causality_score = self._calculate_temporal_causality_score(
                    current_node, next_node, time_diff
                )
                
                if causality_score > self.causality_threshold:
                    temporal_causality['causal_pairs'].append({
                        'cause': current_node,
                        'effect': next_node,
                        'time_diff': time_diff,
                        'causality_score': causality_score
                    })
                    
                    temporal_causality['temporal_ordering'].append({
                        'node': current_node,
                        'position': i,
                        'timestamp': current_time
                    })
                    
                    temporal_causality['causality_scores'].append(causality_score)
        
        return temporal_causality
    
    def _calculate_temporal_causality_score(self, cause_node: str, 
                                          effect_node: str, 
                                          time_diff: float) -> float:
        """
        计算时序因果强度
        
        Args:
            cause_node: 原因节点
            effect_node: 结果节点
            time_diff: 时间差
            
        Returns:
            因果强度分数
        """
        # 基于时间差的因果强度计算
        # 时间差越小，因果强度越高
        if time_diff <= 0:
            return 0.0
        
        # 使用指数衰减函数
        decay_factor = np.exp(-time_diff / (self.temporal_window / 4))
        
        # 基于节点类型的因果强度
        node_type_score = self._get_node_type_causality_score(cause_node, effect_node)
        
        # 综合因果强度
        causality_score = decay_factor * node_type_score
        
        return min(causality_score, 1.0)
    
    def _get_node_type_causality_score(self, cause_node: str, effect_node: str) -> float:
        """
        获取节点类型因果强度
        
        Args:
            cause_node: 原因节点
            effect_node: 结果节点
            
        Returns:
            节点类型因果强度
        """
        # 简化的节点类型因果强度计算
        # 实际实现中需要更复杂的逻辑
        
        # 基于节点名称的简单启发式规则
        cause_lower = cause_node.lower()
        effect_lower = effect_node.lower()
        
        # 定义高因果强度的模式
        high_causality_patterns = [
            ('email', 'attachment'),
            ('attachment', 'execution'),
            ('execution', 'persistence'),
            ('persistence', 'privilege'),
            ('privilege', 'lateral'),
            ('lateral', 'collection'),
            ('collection', 'exfiltration')
        ]
        
        for pattern in high_causality_patterns:
            if pattern[0] in cause_lower and pattern[1] in effect_lower:
                return 0.9
        
        # 默认因果强度
        return 0.5
    
    def _analyze_logical_causality(self, attack_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析逻辑因果关系
        
        Args:
            attack_path: 攻击路径
            
        Returns:
            逻辑因果关系
        """
        logical_causality = {
            'logical_chains': [],
            'attack_stage_transitions': [],
            'logical_consistency': 0.0
        }
        
        if 'attack_stages' not in attack_path:
            return logical_causality
        
        attack_stages = attack_path['attack_stages']
        
        # 分析攻击阶段转换
        for i in range(len(attack_stages) - 1):
            current_stage = attack_stages[i]
            next_stage = attack_stages[i + 1]
            
            # 检查逻辑转换
            logical_score = self._calculate_logical_transition_score(
                current_stage, next_stage
            )
            
            logical_causality['attack_stage_transitions'].append({
                'from_stage': current_stage,
                'to_stage': next_stage,
                'logical_score': logical_score
            })
            
            if logical_score > self.causality_threshold:
                logical_causality['logical_chains'].append({
                    'from_stage': current_stage,
                    'to_stage': next_stage,
                    'logical_score': logical_score
                })
        
        # 计算逻辑一致性
        logical_causality['logical_consistency'] = self._calculate_logical_consistency(
            logical_causality['attack_stage_transitions']
        )
        
        return logical_causality
    
    def _calculate_logical_transition_score(self, from_stage: str, to_stage: str) -> float:
        """
        计算逻辑转换分数
        
        Args:
            from_stage: 起始阶段
            to_stage: 目标阶段
            
        Returns:
            逻辑转换分数
        """
        # 定义合理的阶段转换及其分数
        valid_transitions = {
            ('initial_access', 'execution'): 0.9,
            ('execution', 'persistence'): 0.8,
            ('persistence', 'privilege_escalation'): 0.7,
            ('privilege_escalation', 'defense_evasion'): 0.6,
            ('defense_evasion', 'credential_access'): 0.7,
            ('credential_access', 'discovery'): 0.8,
            ('discovery', 'lateral_movement'): 0.9,
            ('lateral_movement', 'collection'): 0.8,
            ('collection', 'command_and_control'): 0.7,
            ('command_and_control', 'exfiltration'): 0.9,
            ('exfiltration', 'impact'): 0.8
        }
        
        return valid_transitions.get((from_stage, to_stage), 0.3)
    
    def _calculate_logical_consistency(self, transitions: List[Dict[str, Any]]) -> float:
        """
        计算逻辑一致性
        
        Args:
            transitions: 阶段转换列表
            
        Returns:
            逻辑一致性分数
        """
        if not transitions:
            return 0.0
        
        # 计算平均逻辑分数
        logical_scores = [t['logical_score'] for t in transitions]
        avg_logical_score = np.mean(logical_scores)
        
        # 计算转换的连续性
        continuity_score = self._calculate_continuity_score(transitions)
        
        # 综合一致性分数
        consistency_score = (avg_logical_score * 0.7 + continuity_score * 0.3)
        
        return consistency_score
    
    def _calculate_continuity_score(self, transitions: List[Dict[str, Any]]) -> float:
        """
        计算连续性分数
        
        Args:
            transitions: 阶段转换列表
            
        Returns:
            连续性分数
        """
        if len(transitions) < 2:
            return 1.0
        
        # 检查是否存在跳跃
        jumps = 0
        for i in range(len(transitions) - 1):
            current_to = transitions[i]['to_stage']
            next_from = transitions[i + 1]['from_stage']
            
            if current_to != next_from:
                jumps += 1
        
        # 连续性分数 = 1 - 跳跃比例
        continuity_score = 1.0 - (jumps / (len(transitions) - 1))
        
        return continuity_score
    
    def _analyze_resource_causality(self, hetero_data: HeteroData, 
                                   attack_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析资源因果关系
        
        Args:
            hetero_data: 异构图数据
            attack_path: 攻击路径
            
        Returns:
            资源因果关系
        """
        resource_causality = {
            'resource_dependencies': [],
            'resource_flows': [],
            'resource_conflicts': []
        }
        
        if 'path' not in attack_path or 'path_types' not in attack_path:
            return resource_causality
        
        path = attack_path['path']
        path_types = attack_path['path_types']
        
        # 分析资源依赖关系
        for i in range(len(path) - 1):
            current_node = path[i]
            current_type = path_types[i]
            next_node = path[i + 1]
            next_type = path_types[i + 1]
            
            # 检查资源依赖
            dependency_score = self._calculate_resource_dependency_score(
                current_type, next_type
            )
            
            if dependency_score > self.causality_threshold:
                resource_causality['resource_dependencies'].append({
                    'resource': current_node,
                    'resource_type': current_type,
                    'dependent': next_node,
                    'dependent_type': next_type,
                    'dependency_score': dependency_score
                })
        
        # 分析资源流
        resource_causality['resource_flows'] = self._analyze_resource_flows(
            hetero_data, attack_path
        )
        
        # 分析资源冲突
        resource_causality['resource_conflicts'] = self._analyze_resource_conflicts(
            hetero_data, attack_path
        )
        
        return resource_causality
    
    def _calculate_resource_dependency_score(self, resource_type: str, 
                                           dependent_type: str) -> float:
        """
        计算资源依赖分数
        
        Args:
            resource_type: 资源类型
            dependent_type: 依赖类型
            
        Returns:
            资源依赖分数
        """
        # 定义资源依赖关系
        resource_dependencies = {
            ('file', 'process'): 0.9,  # 文件被进程访问
            ('process', 'network'): 0.8,  # 进程创建网络连接
            ('user', 'process'): 0.9,  # 用户启动进程
            ('network', 'data'): 0.8,  # 网络传输数据
            ('registry', 'process'): 0.7,  # 注册表影响进程
            ('service', 'process'): 0.8,  # 服务启动进程
        }
        
        return resource_dependencies.get((resource_type, dependent_type), 0.3)
    
    def _analyze_resource_flows(self, hetero_data: HeteroData, 
                               attack_path: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析资源流
        
        Args:
            hetero_data: 异构图数据
            attack_path: 攻击路径
            
        Returns:
            资源流列表
        """
        resource_flows = []
        
        # 简化的资源流分析
        # 实际实现中需要更复杂的图分析
        
        for i in range(len(attack_path['path']) - 1):
            current_node = attack_path['path'][i]
            next_node = attack_path['path'][i + 1]
            
            # 检查是否存在资源流
            flow_score = self._calculate_resource_flow_score(
                current_node, next_node, hetero_data
            )
            
            if flow_score > self.causality_threshold:
                resource_flows.append({
                    'from_node': current_node,
                    'to_node': next_node,
                    'flow_score': flow_score,
                    'flow_type': 'data_flow'  # 简化实现
                })
        
        return resource_flows
    
    def _calculate_resource_flow_score(self, from_node: str, to_node: str, 
                                     hetero_data: HeteroData) -> float:
        """
        计算资源流分数
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            hetero_data: 异构图数据
            
        Returns:
            资源流分数
        """
        # 简化的资源流分数计算
        # 实际实现中需要分析图中的边和边属性
        
        # 基于节点名称的简单启发式
        if 'data' in from_node.lower() and 'network' in to_node.lower():
            return 0.8
        elif 'file' in from_node.lower() and 'process' in to_node.lower():
            return 0.7
        elif 'user' in from_node.lower() and 'process' in to_node.lower():
            return 0.9
        
        return 0.3
    
    def _analyze_resource_conflicts(self, hetero_data: HeteroData, 
                                   attack_path: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析资源冲突
        
        Args:
            hetero_data: 异构图数据
            attack_path: 攻击路径
            
        Returns:
            资源冲突列表
        """
        resource_conflicts = []
        
        # 简化的资源冲突分析
        # 实际实现中需要更复杂的冲突检测逻辑
        
        return resource_conflicts
    
    def _identify_key_causal_nodes(self, temporal_causality: Dict[str, Any], 
                                  logical_causality: Dict[str, Any], 
                                  resource_causality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        识别关键因果节点
        
        Args:
            temporal_causality: 时序因果关系
            logical_causality: 逻辑因果关系
            resource_causality: 资源因果关系
            
        Returns:
            关键因果节点列表
        """
        key_causal_nodes = []
        
        # 从时序因果关系中识别关键节点
        for pair in temporal_causality.get('causal_pairs', []):
            if pair['causality_score'] > 0.8:
                key_causal_nodes.append({
                    'node': pair['cause'],
                    'type': 'temporal_cause',
                    'score': pair['causality_score'],
                    'description': '时序因果关系中的关键原因节点'
                })
        
        # 从逻辑因果关系中识别关键节点
        for chain in logical_causality.get('logical_chains', []):
            if chain['logical_score'] > 0.8:
                key_causal_nodes.append({
                    'node': chain['from_stage'],
                    'type': 'logical_cause',
                    'score': chain['logical_score'],
                    'description': '逻辑因果关系中的关键原因节点'
                })
        
        # 从资源因果关系中识别关键节点
        for dep in resource_causality.get('resource_dependencies', []):
            if dep['dependency_score'] > 0.8:
                key_causal_nodes.append({
                    'node': dep['resource'],
                    'type': 'resource_cause',
                    'score': dep['dependency_score'],
                    'description': '资源因果关系中的关键资源节点'
                })
        
        # 去重并排序
        unique_nodes = {}
        for node in key_causal_nodes:
            node_id = node['node']
            if node_id not in unique_nodes or node['score'] > unique_nodes[node_id]['score']:
                unique_nodes[node_id] = node
        
        key_causal_nodes = list(unique_nodes.values())
        key_causal_nodes.sort(key=lambda x: x['score'], reverse=True)
        
        return key_causal_nodes
    
    def _build_causal_graph(self, temporal_causality: Dict[str, Any], 
                           logical_causality: Dict[str, Any], 
                           resource_causality: Dict[str, Any]) -> nx.DiGraph:
        """
        构建因果图
        
        Args:
            temporal_causality: 时序因果关系
            logical_causality: 逻辑因果关系
            resource_causality: 资源因果关系
            
        Returns:
            因果图
        """
        causal_graph = nx.DiGraph()
        
        # 添加时序因果关系边
        for pair in temporal_causality.get('causal_pairs', []):
            causal_graph.add_edge(
                pair['cause'], 
                pair['effect'],
                weight=pair['causality_score'],
                type='temporal'
            )
        
        # 添加逻辑因果关系边
        for chain in logical_causality.get('logical_chains', []):
            causal_graph.add_edge(
                chain['from_stage'],
                chain['to_stage'],
                weight=chain['logical_score'],
                type='logical'
            )
        
        # 添加资源因果关系边
        for dep in resource_causality.get('resource_dependencies', []):
            causal_graph.add_edge(
                dep['resource'],
                dep['dependent'],
                weight=dep['dependency_score'],
                type='resource'
            )
        
        return causal_graph
    
    def _calculate_causality_strength(self, causal_graph: nx.DiGraph) -> Dict[str, float]:
        """
        计算因果强度
        
        Args:
            causal_graph: 因果图
            
        Returns:
            因果强度统计
        """
        if not causal_graph.nodes():
            return {'overall_strength': 0.0, 'max_strength': 0.0, 'avg_strength': 0.0}
        
        # 计算边的权重
        edge_weights = [data['weight'] for u, v, data in causal_graph.edges(data=True)]
        
        if not edge_weights:
            return {'overall_strength': 0.0, 'max_strength': 0.0, 'avg_strength': 0.0}
        
        # 计算各种强度指标
        overall_strength = np.mean(edge_weights)
        max_strength = np.max(edge_weights)
        avg_strength = np.mean(edge_weights)
        
        # 计算图的连通性
        connectivity = nx.algorithms.connectivity.edge_connectivity(causal_graph)
        
        return {
            'overall_strength': overall_strength,
            'max_strength': max_strength,
            'avg_strength': avg_strength,
            'connectivity': connectivity,
            'num_edges': len(edge_weights),
            'num_nodes': len(causal_graph.nodes())
        }
    
    def _generate_causality_summary(self, temporal_causality: Dict[str, Any], 
                                   logical_causality: Dict[str, Any], 
                                   resource_causality: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成因果分析摘要
        
        Args:
            temporal_causality: 时序因果关系
            logical_causality: 逻辑因果关系
            resource_causality: 资源因果关系
            
        Returns:
            因果分析摘要
        """
        return {
            'temporal_causal_pairs': len(temporal_causality.get('causal_pairs', [])),
            'logical_chains': len(logical_causality.get('logical_chains', [])),
            'resource_dependencies': len(resource_causality.get('resource_dependencies', [])),
            'logical_consistency': logical_causality.get('logical_consistency', 0.0),
            'avg_temporal_score': np.mean([p['causality_score'] for p in temporal_causality.get('causal_pairs', [])]) if temporal_causality.get('causal_pairs') else 0.0,
            'avg_logical_score': np.mean([c['logical_score'] for c in logical_causality.get('logical_chains', [])]) if logical_causality.get('logical_chains') else 0.0,
            'avg_resource_score': np.mean([d['dependency_score'] for d in resource_causality.get('resource_dependencies', [])]) if resource_causality.get('resource_dependencies') else 0.0
        }
