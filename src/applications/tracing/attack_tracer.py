"""
高级攻击溯源器

实现基于图遍历的溯源算法、时序约束的路径搜索和攻击路径评分机制
这是从"防御者思维"转向"攻击者思维"的关键实现，能够有效还原攻击链
"""

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
import logging
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import heapq
import math
from scipy.stats import entropy

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_networkx, k_hop_subgraph
except ImportError:
    HeteroData = None
    to_networkx = None
    k_hop_subgraph = None


class TemporalConstraint:
    """
    时序约束类，用于定义和检查时间约束条件
    """
    
    def __init__(self, max_time_diff: int = 3600, min_time_diff: int = 0, 
                 time_decay_factor: float = 0.5, time_unit: str = 'seconds'):
        """
        初始化时序约束
        
        Args:
            max_time_diff: 最大时间差（秒）
            min_time_diff: 最小时间差（秒）
            time_decay_factor: 时间衰减因子
            time_unit: 时间单位
        """
        self.max_time_diff = max_time_diff
        self.min_time_diff = min_time_diff
        self.time_decay_factor = time_decay_factor
        self.time_unit = time_unit
        
    def check_constraint(self, time1: int, time2: int) -> bool:
        """
        检查两个时间点是否满足时序约束
        
        Args:
            time1: 时间点1
            time2: 时间点2
            
        Returns:
            是否满足约束
        """
        time_diff = abs(time1 - time2)
        return self.min_time_diff <= time_diff <= self.max_time_diff
    
    def calculate_temporal_weight(self, time1: int, time2: int) -> float:
        """
        计算基于时间差的权重
        
        Args:
            time1: 时间点1
            time2: 时间点2
            
        Returns:
            时间权重
        """
        time_diff = abs(time1 - time2)
        if time_diff > self.max_time_diff:
            return 0.0
        
        # 使用指数衰减函数计算权重
        return math.exp(-self.time_decay_factor * time_diff / self.max_time_diff)


class AttackPathScorer:
    """
    攻击路径评分器，用于评估攻击路径的可能性和严重性
    """
    
    def __init__(self, attack_patterns: Dict[str, List[str]], config: Any = None):
        """
        初始化攻击路径评分器
        
        Args:
            attack_patterns: 攻击模式字典
            config: 配置对象
        """
        self.attack_patterns = attack_patterns
        self.config = config
        
        # 攻击阶段转换概率矩阵（基于MITRE ATT&CK战术顺序）
        self.stage_transition_probs = self._initialize_stage_transition_probs()
        
        # 评分权重
        self.weights = {
            'path_length': 0.15,           # 路径长度权重
            'temporal_coherence': 0.20,    # 时间一致性权重
            'attack_pattern_match': 0.25,  # 攻击模式匹配权重
            'node_importance': 0.15,       # 节点重要性权重
            'edge_confidence': 0.25,       # 边置信度权重
        }
        
    def _initialize_stage_transition_probs(self) -> Dict[str, Dict[str, float]]:
        """
        初始化攻击阶段转换概率矩阵
        
        Returns:
            攻击阶段转换概率字典
        """
        # 攻击阶段（按照MITRE ATT&CK战术顺序）
        stages = [
            'initial_access', 'execution', 'persistence', 
            'privilege_escalation', 'defense_evasion', 'credential_access',
            'discovery', 'lateral_movement', 'collection', 
            'command_and_control', 'exfiltration', 'impact'
        ]
        
        # 初始化转换概率矩阵
        transition_probs = defaultdict(dict)
        
        # 设置默认转换概率
        for i, stage1 in enumerate(stages):
            for j, stage2 in enumerate(stages):
                # 默认概率
                prob = 0.01
                
                # 相同阶段
                if i == j:
                    prob = 0.1
                
                # 相邻阶段（按照攻击链顺序）
                if j == i + 1:
                    prob = 0.5
                
                # 跳过一个阶段
                if j == i + 2:
                    prob = 0.2
                
                # 特殊规则：某些阶段可以直接跳到特定阶段
                if stage1 == 'initial_access' and stage2 == 'execution':
                    prob = 0.8
                if stage1 == 'privilege_escalation' and stage2 == 'lateral_movement':
                    prob = 0.6
                if stage1 == 'lateral_movement' and stage2 == 'collection':
                    prob = 0.7
                if stage1 == 'collection' and stage2 == 'exfiltration':
                    prob = 0.7
                
                transition_probs[stage1][stage2] = prob
        
        return transition_probs
    
    def score_path(self, path: Dict[str, Any], temporal_data: Dict[str, Any] = None) -> float:
        """
        评分攻击路径
        
        Args:
            path: 攻击路径
            temporal_data: 时间数据
            
        Returns:
            路径评分（0-1之间）
        """
        scores = {}
        
        # 1. 路径长度评分
        path_length = len(path['path'])
        if path_length <= 1:
            scores['path_length'] = 0.0
        else:
            # 较长的路径得分较高，但有上限
            optimal_length = 7  # 假设最佳路径长度为7
            scores['path_length'] = min(1.0, path_length / optimal_length)
        
        # 2. 时间一致性评分
        if temporal_data and 'timestamps' in path and len(path['timestamps']) > 1:
            time_diffs = []
            for i in range(len(path['timestamps']) - 1):
                time_diff = abs(path['timestamps'][i] - path['timestamps'][i+1])
                time_diffs.append(time_diff)
            
            # 计算时间差的一致性
            if time_diffs:
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                variance = sum((diff - avg_time_diff) ** 2 for diff in time_diffs) / len(time_diffs)
                # 方差越小，一致性越高
                scores['temporal_coherence'] = 1.0 / (1.0 + math.sqrt(variance) / avg_time_diff)
            else:
                scores['temporal_coherence'] = 0.5
        else:
            scores['temporal_coherence'] = 0.5
        
        # 3. 攻击模式匹配评分
        if 'attack_stages' in path and len(path['attack_stages']) > 1:
            stage_transition_score = 0.0
            for i in range(len(path['attack_stages']) - 1):
                stage1 = path['attack_stages'][i]
                stage2 = path['attack_stages'][i+1]
                if stage1 in self.stage_transition_probs and stage2 in self.stage_transition_probs[stage1]:
                    stage_transition_score += self.stage_transition_probs[stage1][stage2]
            
            if len(path['attack_stages']) > 1:
                stage_transition_score /= (len(path['attack_stages']) - 1)
            
            scores['attack_pattern_match'] = stage_transition_score
        else:
            scores['attack_pattern_match'] = 0.3
        
        # 4. 节点重要性评分
        if 'node_importance' in path:
            scores['node_importance'] = path['node_importance']
        else:
            scores['node_importance'] = 0.5
        
        # 5. 边置信度评分
        if 'confidence_scores' in path and path['confidence_scores']:
            scores['edge_confidence'] = sum(path['confidence_scores']) / len(path['confidence_scores'])
        else:
            scores['edge_confidence'] = 0.5
        
        # 计算加权总分
        total_score = sum(scores[key] * self.weights[key] for key in self.weights)
        
        return total_score
    
    def rank_paths(self, paths: List[Dict[str, Any]], temporal_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        对攻击路径进行排序
        
        Args:
            paths: 攻击路径列表
            temporal_data: 时间数据
            
        Returns:
            排序后的攻击路径列表
        """
        # 计算每条路径的评分
        for path in paths:
            path['score'] = self.score_path(path, temporal_data)
        
        # 按评分降序排序
        sorted_paths = sorted(paths, key=lambda x: x['score'], reverse=True)
        
        return sorted_paths


class GraphTraversalEngine:
    """
    图遍历引擎，实现高效的图遍历算法
    """
    
    def __init__(self, model, config):
        """
        初始化图遍历引擎
        
        Args:
            model: 训练好的模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 遍历参数
        self.max_depth = getattr(config, 'max_trace_depth', 10)
        self.beam_width = getattr(config, 'beam_width', 5)
        self.min_confidence = getattr(config, 'min_confidence', 0.3)
        
        # 时序约束
        self.temporal_constraint = TemporalConstraint(
            max_time_diff=getattr(config, 'max_time_diff', 3600),
            min_time_diff=getattr(config, 'min_time_diff', 0),
            time_decay_factor=getattr(config, 'time_decay_factor', 0.5)
        )
    
    def bidirectional_search(self, hetero_data: HeteroData, 
                            source_node: str, source_type: str,
                            target_node: Optional[str] = None, target_type: Optional[str] = None,
                            timestamps: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """
        双向搜索算法
        
        Args:
            hetero_data: 异构图数据
            source_node: 源节点
            source_type: 源节点类型
            target_node: 目标节点（可选）
            target_type: 目标节点类型（可选）
            timestamps: 时间戳字典
            
        Returns:
            路径列表
        """
        # 如果没有指定目标节点，则使用单向搜索
        if target_node is None or target_type is None:
            return self.beam_search(hetero_data, source_node, source_type, timestamps)
        
        # 前向搜索（从源节点到中间节点）
        forward_paths = self.beam_search(
            hetero_data, source_node, source_type, timestamps, 
            max_depth=self.max_depth // 2, direction='forward'
        )
        
        # 后向搜索（从目标节点到中间节点）
        backward_paths = self.beam_search(
            hetero_data, target_node, target_type, timestamps,
            max_depth=self.max_depth // 2, direction='backward'
        )
        
        # 合并路径
        merged_paths = self._merge_paths(forward_paths, backward_paths)
        
        return merged_paths
    
    def beam_search(self, hetero_data: HeteroData,
                   start_node: str, node_type: str,
                   timestamps: Optional[Dict[str, torch.Tensor]] = None,
                   max_depth: Optional[int] = None,
                   direction: str = 'backward') -> List[Dict[str, Any]]:
        """
        束搜索算法
        
        Args:
            hetero_data: 异构图数据
            start_node: 起始节点
            node_type: 节点类型
            timestamps: 时间戳字典
            max_depth: 最大深度
            direction: 搜索方向 ('forward' 或 'backward')
            
        Returns:
            路径列表
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        # 初始化束
        beam = [{
            'current_node': start_node,
            'current_type': node_type,
            'path': [start_node],
            'path_types': [node_type],
            'confidence_scores': [],
            'attack_stages': [],
            'timestamps': [],
            'evidence': [],
            'depth': 0,
            'score': 1.0  # 初始分数
        }]
        
        # 所有完成的路径
        completed_paths = []
        
        # 已访问节点集合
        visited = set([(start_node, node_type)])
        
        # 束搜索
        for depth in range(max_depth):
            # 候选路径
            candidates = []
            
            # 扩展当前束中的每条路径
            for path_state in beam:
                # 获取邻居节点
                neighbors = self._get_neighbors(
                    hetero_data,
                    path_state['current_node'],
                    path_state['current_type'],
                    timestamps,
                    direction
                )
                
                # 扩展路径
                for neighbor in neighbors:
                    neighbor_node, neighbor_type, edge_type, confidence, timestamp = neighbor
                    
                    # 检查是否已访问
                    if (neighbor_node, neighbor_type) in visited:
                        continue
                    
                    # 检查置信度
                    if confidence < self.min_confidence:
                        continue
                    
                    # 检查时序约束
                    if path_state['timestamps'] and timestamp is not None:
                        if not self.temporal_constraint.check_constraint(path_state['timestamps'][-1], timestamp):
                            continue
                    
                    # 创建新路径状态
                    new_path_state = {
                        'current_node': neighbor_node,
                        'current_type': neighbor_type,
                        'path': path_state['path'] + [neighbor_node],
                        'path_types': path_state['path_types'] + [neighbor_type],
                        'confidence_scores': path_state['confidence_scores'] + [confidence],
                        'attack_stages': path_state['attack_stages'] + [],  # 待填充
                        'timestamps': path_state['timestamps'] + [timestamp] if timestamp is not None else path_state['timestamps'],
                        'evidence': path_state['evidence'] + [(edge_type, confidence)],
                        'depth': path_state['depth'] + 1,
                        'score': path_state['score'] * confidence  # 更新分数
                    }
                    
                    candidates.append(new_path_state)
                    visited.add((neighbor_node, neighbor_type))
            
            # 如果没有候选路径，则结束搜索
            if not candidates:
                break
            
            # 按分数排序并保留前beam_width个
            beam = sorted(candidates, key=lambda x: x['score'], reverse=True)[:self.beam_width]
            
            # 将完成的路径添加到结果中
            completed_paths.extend(beam)
        
        return completed_paths
    
    def _get_neighbors(self, hetero_data: HeteroData,
                      node: str, node_type: str,
                      timestamps: Optional[Dict[str, torch.Tensor]] = None,
                      direction: str = 'backward') -> List[Tuple[str, str, str, float, Optional[int]]]:
        """
        获取节点的邻居
        
        Args:
            hetero_data: 异构图数据
            node: 节点ID
            node_type: 节点类型
            timestamps: 时间戳字典
            direction: 搜索方向
            
        Returns:
            邻居列表，每个元素为 (节点ID, 节点类型, 边类型, 置信度, 时间戳)
        """
        neighbors = []
        
        # 获取节点索引
        try:
            node_idx = int(node)
        except ValueError:
            # 如果节点ID不是整数，则需要查找对应的索引
            node_mapping = getattr(hetero_data, f'{node_type}_mapping', None)
            if node_mapping is not None and node in node_mapping:
                node_idx = node_mapping[node]
            else:
                return []
        
        # 遍历所有边类型
        for edge_key in hetero_data.edge_types:
            src_type, edge_type, dst_type = edge_key
            
            # 根据搜索方向确定要查找的边
            if direction == 'backward':
                if dst_type == node_type:
                    # 查找指向当前节点的边
                    edge_index = hetero_data[edge_key].edge_index
                    edge_attr = getattr(hetero_data[edge_key], 'edge_attr', None)
                    
                    # 找到所有指向当前节点的边
                    dst_indices = edge_index[1]
                    matches = (dst_indices == node_idx).nonzero().view(-1)
                    
                    for match_idx in matches:
                        src_idx = edge_index[0][match_idx].item()
                        
                        # 获取置信度
                        confidence = 0.8  # 默认置信度
                        if edge_attr is not None and match_idx < edge_attr.size(0):
                            # 假设最后一个特征是置信度
                            confidence = edge_attr[match_idx][-1].item()
                        
                        # 获取时间戳
                        timestamp = None
                        if timestamps is not None and edge_key in timestamps:
                            if match_idx < timestamps[edge_key].size(0):
                                timestamp = timestamps[edge_key][match_idx].item()
                        
                        neighbors.append((str(src_idx), src_type, edge_type, confidence, timestamp))
            else:  # forward
                if src_type == node_type:
                    # 查找从当前节点出发的边
                    edge_index = hetero_data[edge_key].edge_index
                    edge_attr = getattr(hetero_data[edge_key], 'edge_attr', None)
                    
                    # 找到所有从当前节点出发的边
                    src_indices = edge_index[0]
                    matches = (src_indices == node_idx).nonzero().view(-1)
                    
                    for match_idx in matches:
                        dst_idx = edge_index[1][match_idx].item()
                        
                        # 获取置信度
                        confidence = 0.8  # 默认置信度
                        if edge_attr is not None and match_idx < edge_attr.size(0):
                            # 假设最后一个特征是置信度
                            confidence = edge_attr[match_idx][-1].item()
                        
                        # 获取时间戳
                        timestamp = None
                        if timestamps is not None and edge_key in timestamps:
                            if match_idx < timestamps[edge_key].size(0):
                                timestamp = timestamps[edge_key][match_idx].item()
                        
                        neighbors.append((str(dst_idx), dst_type, edge_type, confidence, timestamp))
        
        return neighbors
    
    def _merge_paths(self, forward_paths: List[Dict[str, Any]], 
                    backward_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并前向和后向路径
        
        Args:
            forward_paths: 前向路径列表
            backward_paths: 后向路径列表
            
        Returns:
            合并后的路径列表
        """
        merged_paths = []
        
        for forward_path in forward_paths:
            for backward_path in backward_paths:
                # 检查是否有共同节点
                forward_nodes = set(zip(forward_path['path'], forward_path['path_types']))
                backward_nodes = set(zip(backward_path['path'], backward_path['path_types']))
                common_nodes = forward_nodes.intersection(backward_nodes)
                
                if common_nodes:
                    # 选择第一个共同节点作为连接点
                    common_node, common_type = next(iter(common_nodes))
                    
                    # 找到共同节点在两条路径中的位置
                    forward_idx = forward_path['path'].index(common_node)
                    backward_idx = backward_path['path'].index(common_node)
                    
                    # 合并路径
                    merged_path = {
                        'path': forward_path['path'][:forward_idx+1] + backward_path['path'][backward_idx+1:],
                        'path_types': forward_path['path_types'][:forward_idx+1] + backward_path['path_types'][backward_idx+1:],
                        'confidence_scores': forward_path['confidence_scores'][:forward_idx] + backward_path['confidence_scores'][backward_idx:],
                        'attack_stages': forward_path['attack_stages'][:forward_idx+1] + backward_path['attack_stages'][backward_idx+1:],
                        'timestamps': forward_path['timestamps'][:forward_idx+1] + backward_path['timestamps'][backward_idx+1:],
                        'evidence': forward_path['evidence'][:forward_idx] + backward_path['evidence'][backward_idx:],
                        'depth': forward_path['depth'] + backward_path['depth'] - 1,
                        'score': (forward_path['score'] + backward_path['score']) / 2  # 平均分数
                    }
                    
                    merged_paths.append(merged_path)
        
        return merged_paths


class AttackTracer:
    """
    高级攻击溯源器
    
    实现基于图遍历的溯源算法、时序约束的路径搜索和攻击路径评分机制
    1. 从高置信度的恶意节点（如数据外泄点）出发
    2. 沿时间逆序，在图上游走
    3. 基于模型学习到的关联强度，选择最可能的路径进行回溯
    4. 直至定位攻击入口点，还原攻击链
    """
    
    def __init__(self, model, config):
        """
        初始化攻击溯源器
        
        Args:
            model: 训练好的T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 溯源参数
        self.max_depth = getattr(config, 'max_trace_depth', 10)
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        self.temporal_window = getattr(config, 'temporal_window', 3600)  # 1小时
        
        # 攻击模式定义
        self.attack_patterns = self._define_attack_patterns()
        
        # 初始化图遍历引擎
        self.traversal_engine = GraphTraversalEngine(model, config)
        
        # 初始化攻击路径评分器
        self.path_scorer = AttackPathScorer(self.attack_patterns, config)
        
    def _define_attack_patterns(self) -> Dict[str, List[str]]:
        """定义攻击模式，用于指导溯源方向"""
        return {
            'initial_access': [
                'phishing', 'spear_phishing', 'malicious_email',
                'drive_by_download', 'exploit_kit', 'watering_hole',
                'supply_chain_compromise', 'social_engineering'
            ],
            'execution': [
                'command_execution', 'script_execution', 'binary_execution',
                'powershell', 'wmi', 'scheduled_task', 'service_creation'
            ],
            'persistence': [
                'registry_modification', 'startup_folder', 'scheduled_task',
                'service_installation', 'dll_hijacking', 'bootkit'
            ],
            'privilege_escalation': [
                'local_exploit', 'token_manipulation', 'bypass_uac',
                'dll_hijacking', 'service_abuse', 'kernel_exploit'
            ],
            'defense_evasion': [
                'process_hollowing', 'dll_injection', 'code_injection',
                'rootkit', 'anti_vm', 'packing', 'obfuscation'
            ],
            'credential_access': [
                'credential_dumping', 'keylogger', 'credential_harvesting',
                'password_spraying', 'brute_force', 'hash_cracking'
            ],
            'discovery': [
                'network_scanning', 'port_scanning', 'service_enumeration',
                'system_information_gathering', 'network_mapping'
            ],
            'lateral_movement': [
                'remote_service_creation', 'wmi', 'psexec', 'rdp',
                'ssh', 'smb', 'winrm', 'lateral_tool_transfer'
            ],
            'collection': [
                'data_from_local_system', 'data_from_network_shared_drive',
                'data_from_removable_media', 'data_from_cloud_storage'
            ],
            'command_and_control': [
                'web_service', 'dns', 'custom_protocol', 'data_encoding',
                'encrypted_channel', 'proxy', 'remote_access_tool'
            ],
            'exfiltration': [
                'data_compression', 'data_encryption', 'exfiltration_over_c2',
                'exfiltration_over_other_network_medium', 'scheduled_transfer'
            ],
            'impact': [
                'data_encrypted_for_impact', 'data_destruction',
                'service_stop', 'system_shutdown', 'inhibit_system_recovery'
            ]
        }
    
    def trace_attack_path(self, hetero_data: HeteroData, 
                         malicious_node: str, 
                         node_type: str,
                         timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        从恶意节点开始，沿时间逆序回溯攻击路径
        
        这是大纲中提到的核心算法实现
        
        Args:
            hetero_data: 异构图数据
            malicious_node: 恶意节点ID
            node_type: 节点类型
            timestamps: 时间戳字典
            
        Returns:
            攻击路径信息
        """
        self.logger.info(f"开始从节点 {malicious_node} ({node_type}) 溯源攻击路径")
        
        try:
            # 使用束搜索算法进行溯源
            attack_paths = self.traversal_engine.beam_search(
                hetero_data, malicious_node, node_type, timestamps
            )
            
            # 为每条路径分配攻击阶段
            attack_paths = self._assign_attack_stages(attack_paths, hetero_data)
            
            # 使用攻击路径评分器对路径进行评分和排序
            ranked_paths = self.path_scorer.rank_paths(attack_paths, {'timestamps': timestamps})
            
            # 选择最佳路径
            best_path = ranked_paths[0] if ranked_paths else {
                'path': [], 'path_types': [], 'confidence_scores': [],
                'attack_stages': [], 'timestamps': [], 'evidence': [],
                'score': 0.0
            }
            
            # 分析攻击链
            attack_chain = self._analyze_attack_chain(best_path)
            
            self.logger.info(f"溯源完成，发现 {len(ranked_paths)} 条可能路径")
            self.logger.info(f"最佳路径长度: {len(best_path['path'])}")
            
            return {
                'attack_paths': ranked_paths,
                'best_path': best_path,
                'attack_chain': attack_chain,
                'summary': self._generate_trace_summary(best_path, attack_chain)
            }
            
        except Exception as e:
            self.logger.error(f"攻击路径溯源过程中发生错误: {e}")
            return {
                'attack_paths': [],
                'best_path': {'path': [], 'confidence': 0.0, 'error': str(e)},
                'attack_chain': {'stages': [], 'risk_level': 'unknown', 'error': str(e)},
                'summary': {'error': str(e), 'status': 'failed'}
            }
        
        if not neighbors:
            return [trace_state]
        
        # 计算邻居的恶意概率
        neighbor_probs = self._calculate_neighbor_probabilities(
            hetero_data, neighbors, trace_state
        )
        
        # 选择最可疑的邻居
        suspicious_neighbors = self._select_suspicious_neighbors(
            neighbors, neighbor_probs
        )
        
        # 递归溯源
        all_paths = []
        for neighbor_info in suspicious_neighbors:
            # 检查是否已经在路径中，避免环路
            if neighbor_info['node_id'] in trace_state['path']:
                continue
                
            new_state = trace_state.copy()
            new_state['current_node'] = neighbor_info['node_id']
            new_state['current_type'] = neighbor_info['node_type']
            new_state['path'].append(neighbor_info['node_id'])
            new_state['path_types'].append(neighbor_info['node_type'])
            new_state['confidence_scores'].append(neighbor_info['confidence'])
            new_state['attack_stages'].append(neighbor_info['attack_stage'])
            new_state['timestamps'].append(neighbor_info['timestamp'])
            new_state['evidence'].append(neighbor_info['evidence'])
            new_state['depth'] += 1
            
            # 递归调用
            sub_paths = self._trace_recursive(hetero_data, new_state, timestamps)
            all_paths.extend(sub_paths)
        
        return all_paths
        
    def trace_attack_path_bidirectional(self, hetero_data: HeteroData, 
                                      malicious_node: str, 
                                      node_type: str,
                                      entry_points: List[Dict[str, Any]] = None,
                                      timestamps: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        使用双向搜索算法溯源攻击路径
        
        从恶意节点和可能的入口点同时开始搜索，寻找连接点
        
        Args:
            hetero_data: 异构图数据
            malicious_node: 恶意节点ID
            node_type: 节点类型
            entry_points: 可能的入口点列表
            timestamps: 时间戳字典
            
        Returns:
            攻击路径信息
        """
        self.logger.info(f"开始从节点 {malicious_node} ({node_type}) 进行双向溯源")
        
        if not entry_points:
            # 如果没有提供入口点，使用启发式方法找出可能的入口点
            entry_points = self._identify_potential_entry_points(hetero_data, timestamps)
            
        if not entry_points:
            self.logger.warning("未找到可能的入口点，回退到单向溯源")
            return self.trace_attack_path(hetero_data, malicious_node, node_type, timestamps)
        
        try:
            # 从恶意节点开始的正向搜索
            forward_paths = self._bidirectional_search(
                hetero_data,
                malicious_node,
                node_type,
                entry_points,
                timestamps,
                direction="forward"
            )
            
            # 从入口点开始的反向搜索
            backward_paths = []
            for entry in entry_points:
                entry_paths = self._bidirectional_search(
                    hetero_data,
                    entry['node_id'],
                    entry['node_type'],
                    [{'node_id': malicious_node, 'node_type': node_type}],
                    timestamps,
                    direction="backward"
                )
                backward_paths.extend(entry_paths)
            
            # 合并路径
            merged_paths = self._merge_bidirectional_paths(forward_paths, backward_paths)
            
            # 选择最佳路径
            best_path = self._select_best_path(merged_paths)
            
            # 分析攻击链
            attack_chain = self._analyze_attack_chain(best_path)
            
            self.logger.info(f"双向溯源完成，发现 {len(merged_paths)} 条可能路径")
            
            return {
                'attack_paths': merged_paths,
                'best_path': best_path,
                'attack_chain': attack_chain,
                'summary': self._generate_trace_summary(best_path, attack_chain)
            }
            
        except Exception as e:
            self.logger.error(f"双向攻击路径溯源过程中发生错误: {e}")
            return {
                'attack_paths': [],
                'best_path': {'path': [], 'confidence': 0.0, 'error': str(e)},
                'attack_chain': {'stages': [], 'risk_level': 'unknown', 'error': str(e)},
                'summary': {'error': str(e), 'status': 'failed'}
            }
    
    def _bidirectional_search(self, hetero_data: HeteroData,
                            start_node: str,
                            start_type: str,
                            target_nodes: List[Dict[str, Any]],
                            timestamps: Optional[Dict[str, torch.Tensor]] = None,
                            direction: str = "forward",
                            max_depth: int = None) -> List[Dict[str, Any]]:
        """
        执行双向搜索的一半（正向或反向）
        
        Args:
            hetero_data: 异构图数据
            start_node: 起始节点ID
            start_type: 起始节点类型
            target_nodes: 目标节点列表
            timestamps: 时间戳字典
            direction: 搜索方向 ("forward" 或 "backward")
            max_depth: 最大搜索深度
            
        Returns:
            可能的路径列表
        """
        if max_depth is None:
            max_depth = self.max_depth // 2  # 双向搜索，每个方向深度减半
        
        # 初始化队列和访问集合
        queue = deque()
        visited = set()
        
        # 初始化起始状态
        initial_state = {
            'current_node': start_node,
            'current_type': start_type,
            'path': [start_node],
            'path_types': [start_type],
            'confidence_scores': [],
            'attack_stages': [],
            'timestamps': [],
            'evidence': [],
            'depth': 0
        }
        
        queue.append(initial_state)
        visited.add((start_node, start_type))
        
        # 目标节点集合
        target_set = {(node['node_id'], node['node_type']) for node in target_nodes}
        
        # 存储找到的路径
        found_paths = []
        
        # BFS搜索
        while queue and len(found_paths) < 10:  # 限制找到的路径数量
            current_state = queue.popleft()
            
            # 检查是否到达目标
            current_key = (current_state['current_node'], current_state['current_type'])
            if current_key in target_set:
                found_paths.append(current_state)
                continue
            
            # 检查深度限制
            if current_state['depth'] >= max_depth:
                continue
            
            # 获取邻居节点
            neighbors = self._get_temporal_neighbors(
                hetero_data,
                current_state['current_node'],
                current_state['current_type'],
                timestamps,
                direction=direction
            )
            
            if not neighbors:
                continue
            
            # 计算邻居的恶意概率
            neighbor_probs = self._calculate_neighbor_probabilities(
                hetero_data, neighbors, current_state
            )
            
            # 选择最可疑的邻居
            suspicious_neighbors = self._select_suspicious_neighbors(
                neighbors, neighbor_probs
            )
            
            # 扩展搜索
            for neighbor_info in suspicious_neighbors:
                neighbor_key = (neighbor_info['node_id'], neighbor_info['node_type'])
                
                # 检查是否已访问
                if neighbor_key in visited:
                    continue
                
                # 创建新状态
                new_state = current_state.copy()
                new_state['current_node'] = neighbor_info['node_id']
                new_state['current_type'] = neighbor_info['node_type']
                new_state['path'] = current_state['path'] + [neighbor_info['node_id']]
                new_state['path_types'] = current_state['path_types'] + [neighbor_info['node_type']]
                new_state['confidence_scores'] = current_state['confidence_scores'] + [neighbor_info['confidence']]
                new_state['attack_stages'] = current_state['attack_stages'] + [neighbor_info['attack_stage']]
                new_state['timestamps'] = current_state['timestamps'] + [neighbor_info['timestamp']]
                new_state['evidence'] = current_state['evidence'] + [neighbor_info['evidence']]
                new_state['depth'] = current_state['depth'] + 1
                
                # 添加到队列
                queue.append(new_state)
                visited.add(neighbor_key)
        
        return found_paths
    
    def _merge_bidirectional_paths(self, forward_paths: List[Dict[str, Any]], 
                                 backward_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并双向搜索的路径
        
        Args:
            forward_paths: 正向搜索路径
            backward_paths: 反向搜索路径
            
        Returns:
            合并后的路径列表
        """
        merged_paths = []
        
        # 如果任一方向没有找到路径，返回另一方向的路径
        if not forward_paths:
            return backward_paths
        if not backward_paths:
            return forward_paths
        
        # 尝试合并路径
        for forward_path in forward_paths:
            for backward_path in backward_paths:
                # 检查路径是否可以连接（末尾节点和起始节点匹配）
                if (forward_path['path'][-1] == backward_path['path'][0] and 
                    forward_path['path_types'][-1] == backward_path['path_types'][0]):
                    
                    # 合并路径
                    merged_path = {
                        'path': forward_path['path'] + backward_path['path'][1:],
                        'path_types': forward_path['path_types'] + backward_path['path_types'][1:],
                        'confidence_scores': forward_path['confidence_scores'] + backward_path['confidence_scores'],
                        'attack_stages': forward_path['attack_stages'] + backward_path['attack_stages'],
                        'timestamps': forward_path['timestamps'] + backward_path['timestamps'],
                        'evidence': forward_path['evidence'] + backward_path['evidence'],
                        'depth': forward_path['depth'] + backward_path['depth']
                    }
                    
                    # 计算合并路径的置信度
                    avg_confidence = sum(merged_path['confidence_scores']) / len(merged_path['confidence_scores']) if merged_path['confidence_scores'] else 0
                    merged_path['confidence'] = avg_confidence
                    
                    merged_paths.append(merged_path)
        
        # 如果没有成功合并的路径，返回所有路径的组合
        if not merged_paths:
            merged_paths = forward_paths + backward_paths
        
        return merged_paths
    
    def _identify_potential_entry_points(self, hetero_data: HeteroData, 
                                       timestamps: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """
        识别可能的攻击入口点
        
        Args:
            hetero_data: 异构图数据
            timestamps: 时间戳字典
            
        Returns:
            可能的入口点列表
        """
        potential_entries = []
        
        # 入口点类型和关键词
        entry_node_types = ['user', 'email', 'process', 'file', 'connection']
        entry_keywords = ['external', 'internet', 'email', 'download', 'attachment', 'browser', 'web']
        
        # 遍历所有节点类型
        for node_type in hetero_data.node_types:
            if node_type not in entry_node_types:
                continue
                
            # 获取节点特征
            x = hetero_data[node_type].x
            
            # 获取节点数量
            num_nodes = hetero_data[node_type].num_nodes
            
            # 检查每个节点
            for i in range(num_nodes):
                node_id = self._get_node_id(i, node_type)
                
                # 检查节点是否可能是入口点
                if self._is_potential_entry_point(hetero_data, node_id, node_type, entry_keywords):
                    # 获取时间戳
                    timestamp = self._get_node_timestamp(node_id, node_type, timestamps)
                    
                    potential_entries.append({
                        'node_id': node_id,
                        'node_type': node_type,
                        'timestamp': timestamp,
                        'confidence': 0.7  # 默认置信度
                    })
        
        # 按时间戳排序（最早的可能入口点优先）
        potential_entries.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '')
        
        # 限制返回的入口点数量
        return potential_entries[:5]
    
    def _is_potential_entry_point(self, hetero_data: HeteroData, 
                                node_id: str, 
                                node_type: str, 
                                keywords: List[str]) -> bool:
        """
        检查节点是否可能是入口点
        
        Args:
            hetero_data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            keywords: 关键词列表
            
        Returns:
            是否可能是入口点
        """
        # 检查节点属性
        node_attrs = self._get_node_attributes(hetero_data, node_id, node_type)
        
        # 检查节点名称或描述是否包含关键词
        for key, value in node_attrs.items():
            if isinstance(value, str):
                for keyword in keywords:
                    if keyword.lower() in value.lower():
                        return True
        
        # 检查节点的入度
        in_degree = self._get_node_in_degree(hetero_data, node_id, node_type)
        
        # 入度为0或很小的节点可能是入口点
        if in_degree == 0 or (in_degree < 3 and self._get_node_out_degree(hetero_data, node_id, node_type) > 3):
            return True
            
        return False
    
    def _get_node_attributes(self, hetero_data: HeteroData, 
                           node_id: str, 
                           node_type: str) -> Dict[str, Any]:
        """
        获取节点属性
        
        Args:
            hetero_data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            节点属性字典
        """
        # 完整的节点属性获取实现
        node_attributes = {}
        
        try:
            # 获取节点特征
            if hasattr(hetero_data, 'x_dict') and node_type in hetero_data.x_dict:
                node_features = hetero_data.x_dict[node_type]
                if node_id in hetero_data[node_type]:
                    node_idx = hetero_data[node_type].index(node_id)
                    if node_idx < len(node_features):
                        node_attributes['features'] = node_features[node_idx].tolist()
            
            # 获取节点时间戳
            if hasattr(hetero_data, 'timestamp_dict') and node_type in hetero_data.timestamp_dict:
                timestamps = hetero_data.timestamp_dict[node_type]
                if node_id in hetero_data[node_type]:
                    node_idx = hetero_data[node_type].index(node_id)
                    if node_idx < len(timestamps):
                        node_attributes['timestamp'] = timestamps[node_idx].item()
            
            # 获取节点标签
            if hasattr(hetero_data, 'y_dict') and node_type in hetero_data.y_dict:
                labels = hetero_data.y_dict[node_type]
                if node_id in hetero_data[node_type]:
                    node_idx = hetero_data[node_type].index(node_id)
                    if node_idx < len(labels):
                        node_attributes['label'] = labels[node_idx].item()
            
            # 获取节点度信息
            node_attributes['in_degree'] = self._get_node_in_degree(hetero_data, node_id, node_type)
            node_attributes['out_degree'] = self._get_node_out_degree(hetero_data, node_id, node_type)
            
            # 获取邻居信息
            neighbors = self._get_node_neighbors(hetero_data, node_id, node_type)
            node_attributes['neighbors'] = neighbors
            
            # 获取节点类型特定的属性
            if node_type == 'process':
                node_attributes['process_type'] = self._get_process_type(node_id)
                node_attributes['parent_process'] = self._get_parent_process(node_id)
            elif node_type == 'file':
                node_attributes['file_type'] = self._get_file_type(node_id)
                node_attributes['file_size'] = self._get_file_size(node_id)
            elif node_type == 'network':
                node_attributes['protocol'] = self._get_network_protocol(node_id)
                node_attributes['port'] = self._get_network_port(node_id)
            
        except Exception as e:
            self.logger.warning(f"获取节点 {node_id} 属性失败: {e}")
            node_attributes['error'] = str(e)
        
        return node_attributes
    
    def _get_process_type(self, node_id: str) -> str:
        """获取进程类型"""
        # 根据进程ID或名称推断进程类型
        if 'cmd' in node_id.lower() or 'powershell' in node_id.lower():
            return 'shell'
        elif 'explorer' in node_id.lower():
            return 'gui'
        elif 'system' in node_id.lower():
            return 'system'
        else:
            return 'unknown'
    
    def _get_parent_process(self, node_id: str) -> str:
        """获取父进程"""
        # 这里需要根据实际数据获取父进程信息
        return 'unknown'
    
    def _get_file_type(self, node_id: str) -> str:
        """获取文件类型"""
        if node_id.endswith('.exe'):
            return 'executable'
        elif node_id.endswith('.dll'):
            return 'library'
        elif node_id.endswith('.txt') or node_id.endswith('.log'):
            return 'text'
        else:
            return 'unknown'
    
    def _get_file_size(self, node_id: str) -> int:
        """获取文件大小"""
        # 这里需要根据实际数据获取文件大小
        return 0
    
    def _get_network_protocol(self, node_id: str) -> str:
        """获取网络协议"""
        if 'tcp' in node_id.lower():
            return 'tcp'
        elif 'udp' in node_id.lower():
            return 'udp'
        else:
            return 'unknown'
    
    def _get_network_port(self, node_id: str) -> int:
        """获取网络端口"""
        # 从节点ID中提取端口号
        import re
        port_match = re.search(r':(\d+)', node_id)
        if port_match:
            return int(port_match.group(1))
        return 0
    
    def _get_node_neighbors(self, hetero_data: HeteroData, node_id: str, node_type: str) -> List[Dict[str, Any]]:
        """获取节点邻居"""
        neighbors = []
        try:
            # 获取所有边类型
            for edge_type in hetero_data.edge_types:
                if edge_type[0] == node_type:  # 出边
                    edge_index = hetero_data[edge_type].edge_index
                    if node_id in hetero_data[node_type]:
                        node_idx = hetero_data[node_type].index(node_id)
                        neighbor_indices = edge_index[1][edge_index[0] == node_idx]
                        for neighbor_idx in neighbor_indices:
                            neighbor_id = hetero_data[edge_type[2]][neighbor_idx]
                            neighbors.append({
                                'node_id': neighbor_id,
                                'node_type': edge_type[2],
                                'edge_type': edge_type[1]
                            })
                elif edge_type[2] == node_type:  # 入边
                    edge_index = hetero_data[edge_type].edge_index
                    if node_id in hetero_data[node_type]:
                        node_idx = hetero_data[node_type].index(node_id)
                        neighbor_indices = edge_index[0][edge_index[1] == node_idx]
                        for neighbor_idx in neighbor_indices:
                            neighbor_id = hetero_data[edge_type[0]][neighbor_idx]
                            neighbors.append({
                                'node_id': neighbor_id,
                                'node_type': edge_type[0],
                                'edge_type': edge_type[1]
                            })
        except Exception as e:
            self.logger.warning(f"获取节点 {node_id} 邻居失败: {e}")
        
        return neighbors
    
    def _get_node_type_features(self, node_type: str) -> List[float]:
        """获取节点类型特征"""
        # 为不同节点类型分配特征向量
        type_features = {
            'process': [1.0, 0.0, 0.0, 0.0],  # 进程
            'file': [0.0, 1.0, 0.0, 0.0],      # 文件
            'network': [0.0, 0.0, 1.0, 0.0],   # 网络
            'registry': [0.0, 0.0, 0.0, 1.0],   # 注册表
            'user': [0.5, 0.5, 0.0, 0.0],      # 用户
            'service': [0.5, 0.0, 0.5, 0.0],    # 服务
        }
        return type_features.get(node_type, [0.0, 0.0, 0.0, 0.0])
    
    def _calculate_malicious_probability_heuristic(self, features: List[float], neighbor: Dict[str, Any]) -> float:
        """使用启发式规则计算恶意概率"""
        prob = 0.0
        
        # 基于节点类型的恶意概率
        node_type = neighbor['node_type']
        if node_type == 'process':
            # 检查进程名称
            node_id = neighbor['node_id'].lower()
            if any(suspicious in node_id for suspicious in ['cmd', 'powershell', 'wscript', 'cscript']):
                prob += 0.3
            if any(malicious in node_id for malicious in ['malware', 'virus', 'trojan', 'backdoor']):
                prob += 0.5
        elif node_type == 'file':
            # 检查文件扩展名
            node_id = neighbor['node_id'].lower()
            if node_id.endswith(('.exe', '.bat', '.cmd', '.ps1', '.vbs')):
                prob += 0.2
            if any(suspicious in node_id for suspicious in ['temp', 'tmp', 'download']):
                prob += 0.1
        elif node_type == 'network':
            # 检查网络连接
            node_id = neighbor['node_id'].lower()
            if any(suspicious in node_id for suspicious in ['tor', 'proxy', 'vpn']):
                prob += 0.3
            if ':80' in node_id or ':443' in node_id:
                prob += 0.1  # HTTP/HTTPS连接
        
        # 基于特征的恶意概率
        if len(features) > 0:
            # 时间异常性
            if len(features) > 4:  # 确保有时间特征
                time_anomaly = features[-1] if features[-1] > 0 else 0
                prob += min(time_anomaly * 0.2, 0.3)
            
            # 度异常性
            if len(features) >= 2:
                in_degree, out_degree = features[-2], features[-1]
                if in_degree > 10 or out_degree > 10:  # 高度数节点
                    prob += 0.1
                if in_degree == 0 and out_degree > 5:  # 只有出边
                    prob += 0.2
        
        # 限制概率范围
        return min(max(prob, 0.0), 1.0)
    
    def _get_node_in_degree(self, hetero_data: HeteroData, 
                          node_id: str, 
                          node_type: str) -> int:
        """
        获取节点的入度
        
        Args:
            hetero_data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            节点入度
        """
        in_degree = 0
        
        # 获取节点索引
        node_idx = self._get_node_index(node_id, node_type)
        
        # 遍历所有边类型
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            if dst_type == node_type:
                # 当前节点是目标节点
                edge_index = hetero_data[edge_type].edge_index
                in_degree += torch.sum(edge_index[1] == node_idx).item()
        
        return in_degree
    
    def _get_node_out_degree(self, hetero_data: HeteroData, 
                           node_id: str, 
                           node_type: str) -> int:
        """
        获取节点的出度
        
        Args:
            hetero_data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            节点出度
        """
        out_degree = 0
        
        # 获取节点索引
        node_idx = self._get_node_index(node_id, node_type)
        
        # 遍历所有边类型
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            if src_type == node_type:
                # 当前节点是源节点
                edge_index = hetero_data[edge_type].edge_index
                out_degree += torch.sum(edge_index[0] == node_idx).item()
        
        return out_degree
        
    def _get_temporal_neighbors(self, hetero_data: HeteroData, 
                              node_id: str, 
                              node_type: str,
                              timestamps: Optional[Dict[str, torch.Tensor]] = None,
                              direction: str = "forward") -> List[Dict[str, Any]]:
        """
        获取节点的时序邻居
        
        Args:
            hetero_data: 异构图数据
            node_id: 节点ID
            node_type: 节点类型
            timestamps: 时间戳字典
            direction: 搜索方向 ("forward" 或 "backward")
            
        Returns:
            邻居节点列表
        """
        neighbors = []
        current_timestamp = self._get_node_timestamp(node_id, node_type, timestamps)
        
        # 遍历所有边类型
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            # 根据搜索方向确定边的方向
            if direction == "forward":
                # 正向搜索：当前节点是源节点，寻找目标节点
                if src_type == node_type:
                    edge_index = hetero_data[edge_type].edge_index
                    connected_indices = torch.where(edge_index[0] == self._get_node_index(node_id, node_type))[0]
                    neighbor_indices = edge_index[1][connected_indices]
                    neighbor_type = dst_type
                else:
                    continue  # 跳过不符合方向的边
            else:
                # 反向搜索：当前节点是目标节点，寻找源节点
                if dst_type == node_type:
                    edge_index = hetero_data[edge_type].edge_index
                    connected_indices = torch.where(edge_index[1] == self._get_node_index(node_id, node_type))[0]
                    neighbor_indices = edge_index[0][connected_indices]
                    neighbor_type = src_type
                else:
                    continue  # 跳过不符合方向的边
            
            # 处理邻居节点
            for neighbor_idx in neighbor_indices:
                neighbor_id = self._get_node_id(neighbor_idx, neighbor_type)
                
                # 检查时序约束
                if self._check_temporal_constraint(current_timestamp, neighbor_id, neighbor_type, timestamps, direction):
                    neighbors.append({
                        'node_id': neighbor_id,
                        'node_type': neighbor_type,
                        'edge_type': edge_type,
                        'relationship': rel_type,
                        'timestamp': self._get_node_timestamp(neighbor_id, neighbor_type, timestamps)
                    })
        
        return neighbors
        
    def _check_temporal_constraint(self, current_timestamp: str, 
                                 neighbor_id: str, 
                                 neighbor_type: str, 
                                 timestamps: Optional[Dict[str, torch.Tensor]] = None,
                                 direction: str = "forward") -> bool:
        """
        检查时序约束
        
        Args:
            current_timestamp: 当前时间戳
            neighbor_id: 邻居节点ID
            neighbor_type: 邻居节点类型
            timestamps: 时间戳字典
            direction: 搜索方向 ("forward" 或 "backward")
            
        Returns:
            是否满足时序约束
        """
        if not timestamps or not current_timestamp:
            return True
            
        neighbor_timestamp = self._get_node_timestamp(neighbor_id, neighbor_type, timestamps)
        
        if not neighbor_timestamp:
            return True
            
        try:
            current_time = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
            neighbor_time = datetime.fromisoformat(neighbor_timestamp.replace('Z', '+00:00'))
            
            if direction == "forward":
                # 正向搜索：邻居时间应该早于当前时间
                return neighbor_time <= current_time
            else:
                # 反向搜索：邻居时间应该晚于当前时间
                return neighbor_time >= current_time
                
        except (ValueError, TypeError):
            # 时间格式错误，默认允许
            return True
    
    def _get_temporal_neighbors(self, hetero_data: HeteroData, 
                               node_id: str, node_type: str,
                               timestamps: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """
        获取时序邻居节点
        
        Args:
            hetero_data: 异构图数据
            node_id: 当前节点ID
            node_type: 当前节点类型
            timestamps: 时间戳字典
            
        Returns:
            邻居节点信息列表
        """
        neighbors = []
        
        # 获取当前节点的时间戳
        current_timestamp = None
        if timestamps and node_type in timestamps:
            # 尝试从时间戳字典中获取当前节点的时间戳
            try:
                node_ids = getattr(hetero_data[node_type], 'node_ids', None)
                if node_ids is not None and node_id < len(node_ids):
                    node_key = node_ids[node_id]
                    current_timestamp = timestamps[node_type].get(node_key)
                else:
                    # 如果无法获取具体节点时间戳，尝试获取该节点类型的通用时间戳
                    current_timestamp = timestamps[node_type].get('default')
            except (KeyError, IndexError, AttributeError) as e:
                # 记录错误但不中断执行
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"无法获取节点时间戳: {e}")
                current_timestamp = None
        
        # 遍历所有边类型，寻找连接关系
        for edge_type in hetero_data.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            # 检查是否与当前节点相关
            if src_type == node_type or dst_type == node_type:
                edge_index = hetero_data[edge_type].edge_index
                
                # 找到与当前节点相连的边
                if src_type == node_type:
                    # 当前节点是源节点
                    connected_indices = torch.where(edge_index[0] == self._get_node_index(node_id, node_type))[0]
                    neighbor_indices = edge_index[1][connected_indices]
                    neighbor_type = dst_type
                else:
                    # 当前节点是目标节点
                    connected_indices = torch.where(edge_index[1] == self._get_node_index(node_id, node_type))[0]
                    neighbor_indices = edge_index[0][connected_indices]
                    neighbor_type = src_type
                
                # 处理邻居节点
                for neighbor_idx in neighbor_indices:
                    neighbor_id = self._get_node_id(neighbor_idx, neighbor_type)
                    
                    # 检查时序约束
                    if self._check_temporal_constraint(current_timestamp, neighbor_id, neighbor_type, timestamps):
                        neighbors.append({
                            'node_id': neighbor_id,
                            'node_type': neighbor_type,
                            'edge_type': edge_type,
                            'relationship': rel_type,
                            'timestamp': self._get_node_timestamp(neighbor_id, neighbor_type, timestamps)
                        })
        
        return neighbors
    
    def _calculate_neighbor_probabilities(self, hetero_data: HeteroData, 
                                        neighbors: List[Dict[str, Any]],
                                        trace_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算邻居节点的恶意概率
        
        Args:
            hetero_data: 异构图数据
            neighbors: 邻居节点列表
            trace_state: 当前溯源状态
            
        Returns:
            邻居节点概率字典
        """
        probabilities = {}
        
        for neighbor in neighbors:
            # 使用模型预测恶意概率
            try:
                # 完整的恶意概率预测实现
                # 使用多种特征进行预测
                features = []
                
                # 1. 节点特征
                if hasattr(hetero_data, 'x_dict') and neighbor['node_type'] in hetero_data.x_dict:
                    node_features = hetero_data.x_dict[neighbor['node_type']]
                    if neighbor['node_id'] in hetero_data[neighbor['node_type']]:
                        node_idx = hetero_data[neighbor['node_type']].index(neighbor['node_id'])
                        if node_idx < len(node_features):
                            features.extend(node_features[node_idx].tolist())
                
                # 2. 时间特征
                if hasattr(hetero_data, 'timestamp_dict') and neighbor['node_type'] in hetero_data.timestamp_dict:
                    timestamps = hetero_data.timestamp_dict[neighbor['node_type']]
                    if neighbor['node_id'] in hetero_data[neighbor['node_type']]:
                        node_idx = hetero_data[neighbor['node_type']].index(neighbor['node_id'])
                        if node_idx < len(timestamps):
                            timestamp = timestamps[node_idx].item()
                            # 时间异常性特征
                            current_time = max(timestamps.tolist()) if len(timestamps) > 0 else timestamp
                            time_anomaly = abs(timestamp - current_time) / (current_time + 1e-6)
                            features.append(time_anomaly)
                
                # 3. 度特征
                in_degree = self._get_node_in_degree(hetero_data, neighbor['node_id'], neighbor['node_type'])
                out_degree = self._get_node_out_degree(hetero_data, neighbor['node_id'], neighbor['node_type'])
                features.extend([in_degree, out_degree])
                
                # 4. 节点类型特征
                node_type_features = self._get_node_type_features(neighbor['node_type'])
                features.extend(node_type_features)
                
                # 5. 邻居特征
                neighbor_count = len(self._get_node_neighbors(hetero_data, neighbor['node_id'], neighbor['node_type']))
                features.append(neighbor_count)
                
                # 使用简单的启发式规则计算恶意概率
                prob = self._calculate_malicious_probability_heuristic(features, neighbor)
                probabilities[neighbor['node_id']] = prob
            except Exception as e:
                self.logger.warning(f"计算节点 {neighbor['node_id']} 概率失败: {e}")
                probabilities[neighbor['node_id']] = 0.0
        
        return probabilities
    
    def _select_suspicious_neighbors(self, neighbors: List[Dict[str, Any]], 
                                   probabilities: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        选择最可疑的邻居节点
        
        Args:
            neighbors: 邻居节点列表
            probabilities: 概率字典
            
        Returns:
            可疑邻居节点列表
        """
        # 按概率排序
        suspicious_neighbors = []
        for neighbor in neighbors:
            prob = probabilities.get(neighbor['node_id'], 0.0)
            if prob > self.confidence_threshold:
                neighbor['confidence'] = prob
                neighbor['attack_stage'] = self._classify_attack_stage(neighbor)
                neighbor['evidence'] = self._extract_evidence(neighbor)
                suspicious_neighbors.append(neighbor)
        
        # 按置信度排序，选择前几个
        suspicious_neighbors.sort(key=lambda x: x['confidence'], reverse=True)
        return suspicious_neighbors[:3]  # 最多选择3个最可疑的邻居
    
    def _predict_malicious_probability(self, hetero_data: HeteroData, 
                                     neighbor: Dict[str, Any]) -> float:
        """
        预测节点的恶意概率
        
        Args:
            hetero_data: 异构图数据
            neighbor: 邻居节点信息
            
        Returns:
            恶意概率
        """
        try:
            # 使用模型进行预测
            if hasattr(self, 'model') and self.model is not None:
                # 将节点特征转换为模型输入格式
                node_feature = torch.tensor(neighbor['features'], dtype=torch.float32)
                if len(node_feature.shape) == 1:
                    node_feature = node_feature.unsqueeze(0)
                
                with torch.no_grad():
                    predictions = self.model(node_feature)
                    # 提取恶意概率
                    if isinstance(predictions, dict):
                        node_type = neighbor.get('node_type', 'unknown')
                        if node_type in predictions:
                            prob = torch.softmax(predictions[node_type], dim=-1)
                            return float(prob[:, 1].item())  # 恶意类别的概率
                    
                    elif predictions.dim() >= 2:
                        prob = torch.softmax(predictions, dim=-1)
                        return float(prob[:, 1].item())
                    
                    else:
                        prob = torch.sigmoid(predictions)
                        return float(prob.mean().item())
            
            # 如果没有模型，基于特征启发式计算
            return self._compute_heuristic_malicious_probability(neighbor)
            
        except Exception as e:
            self.logger.warning(f"预测节点 {neighbor['node_id']} 恶意概率失败: {e}")
            return 0.0
    
    def _compute_heuristic_malicious_probability(self, neighbor: Dict[str, Any]) -> float:
        """
        基于特征启发式计算恶意概率
        
        Args:
            neighbor: 邻居节点信息
            
        Returns:
            恶意概率 (0-1)
        """
        score = 0.0
        
        # 基于节点类型的评分
        node_type = neighbor.get('node_type', '').lower()
        if node_type in ['alert', 'process', 'file']:
            # 如果是关键安全实体类型，起始分数较高
            base_score = 0.1
        elif node_type in ['ip', 'domain']:
            base_score = 0.05
        else:
            base_score = 0.01
            
        score += base_score
        
        # 基于特征的评分
        features = neighbor.get('features', [])
        if features:
            # 如果有异常特征值，增加恶意概率
            feature_sum = abs(sum(features))
            if feature_sum > 0:
                score += min(feature_sum * 0.1, 0.8)  # 限制最大贡献
        
        # 基于拓扑特征的评分
        degree = neighbor.get('degree', 0)
        if degree > 10:  # 高连接度节点
            score += 0.2
        elif degree > 5:
            score += 0.1
        
        # 基于时间特征的评分
        timestamp = neighbor.get('timestamp')
        if timestamp:
            # 简化时间分析：如果时间戳过新或过旧，可能异常
            import time
            current_time = time.time()
            time_diff = abs(current_time - timestamp)
            if time_diff > 86400:  # 超过1天
                score += 0.15
        
        # 确保概率在合理范围内
        return max(0.0, min(score, 0.95))
    
    def _classify_attack_stage(self, neighbor: Dict[str, Any]) -> str:
        """
        分类攻击阶段
        
        Args:
            neighbor: 邻居节点信息
            
        Returns:
            攻击阶段
        """
        # 根据节点类型和关系类型判断攻击阶段
        node_type = neighbor['node_type']
        relationship = neighbor['relationship']
        
        # 简化的攻击阶段分类逻辑
        if 'email' in node_type or 'phishing' in relationship:
            return 'initial_access'
        elif 'command' in node_type or 'execution' in relationship:
            return 'execution'
        elif 'registry' in node_type or 'persistence' in relationship:
            return 'persistence'
        elif 'privilege' in relationship or 'escalation' in relationship:
            return 'privilege_escalation'
        elif 'network' in node_type or 'scanning' in relationship:
            return 'discovery'
        elif 'lateral' in relationship or 'movement' in relationship:
            return 'lateral_movement'
        elif 'data' in node_type or 'exfiltration' in relationship:
            return 'exfiltration'
        else:
            return 'unknown'
    
    def _extract_evidence(self, neighbor: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取证据信息
        
        Args:
            neighbor: 邻居节点信息
            
        Returns:
            证据信息
        """
        return {
            'node_type': neighbor['node_type'],
            'relationship': neighbor['relationship'],
            'timestamp': neighbor['timestamp'],
            'confidence': neighbor.get('confidence', 0.0),
            'attack_stage': neighbor.get('attack_stage', 'unknown')
        }
    
    def _check_temporal_constraint(self, current_timestamp: Optional[datetime],
                                 neighbor_id: str, neighbor_type: str,
                                 timestamps: Optional[Dict[str, torch.Tensor]]) -> bool:
        """
        检查时序约束
        
        Args:
            current_timestamp: 当前节点时间戳
            neighbor_id: 邻居节点ID
            neighbor_type: 邻居节点类型
            timestamps: 时间戳字典
            
        Returns:
            是否满足时序约束
        """
        if current_timestamp is None:
            return True
        
        # 获取邻居节点时间戳
        neighbor_timestamp = self._get_node_timestamp(neighbor_id, neighbor_type, timestamps)
        if neighbor_timestamp is None:
            return True
        
        # 检查时序约束：邻居节点应该在当前节点之前
        time_diff = (current_timestamp - neighbor_timestamp).total_seconds()
        return 0 <= time_diff <= self.temporal_window
    
    def _get_node_index(self, node_id: str, node_type: str) -> int:
        """获取节点索引（简化实现）"""
        # 这里需要根据实际的节点ID映射来实现
        return hash(node_id) % 1000
    
    def _get_node_id(self, node_index: int, node_type: str) -> str:
        """获取节点ID（简化实现）"""
        # 这里需要根据实际的节点索引映射来实现
        return f"{node_type}_{node_index}"
    
    def _get_node_timestamp(self, node_id: str, node_type: str, 
                           timestamps: Optional[Dict[str, torch.Tensor]]) -> Optional[datetime]:
        """获取节点时间戳（简化实现）"""
        # 这里需要根据实际的时间戳映射来实现
        return datetime.now()
    
    def _select_best_path(self, attack_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从多条可能的攻击路径中选择最佳路径
        
        使用多维度评分机制，考虑路径置信度、完整性、时序一致性和关键节点覆盖率
        
        Args:
            attack_paths: 攻击路径列表
            
        Returns:
            最佳攻击路径
        """
        if not attack_paths:
            return {'path': [], 'confidence': 0.0, 'path_score': 0.0}
            
        # 计算每条路径的综合评分
        scored_paths = []
        for path in attack_paths:
            path_score = self._calculate_path_score(path)
            path['path_score'] = path_score
            scored_paths.append(path)
            
        # 选择评分最高的路径
        best_path = max(scored_paths, key=lambda x: x.get('path_score', 0))
        
        return best_path
        
    def _calculate_path_score(self, path: Dict[str, Any]) -> float:
        """
        计算攻击路径的综合评分
        
        评分维度包括：
        1. 置信度 - 路径中节点的平均置信度
        2. 完整性 - 路径是否覆盖了完整的攻击阶段
        3. 时序一致性 - 路径中的时间戳是否符合逻辑顺序
        4. 关键节点覆盖率 - 路径是否包含关键节点类型
        5. 路径长度 - 适当长度的路径（不过长也不过短）
        
        Args:
            path: 攻击路径
            
        Returns:
            综合评分 (0-1)
        """
        # 1. 置信度评分 (0-1)
        confidence_scores = path.get('confidence_scores', [])
        confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # 2. 完整性评分 (0-1)
        completeness_score = self._calculate_completeness_score(path)
        
        # 3. 时序一致性评分 (0-1)
        temporal_score = self._calculate_temporal_consistency(path)
        
        # 4. 关键节点覆盖率评分 (0-1)
        coverage_score = self._calculate_critical_node_coverage(path)
        
        # 5. 路径长度评分 (0-1)
        length_score = self._calculate_path_length_score(path)
        
        # 综合评分 - 加权平均
        weights = {
            'confidence': 0.3,
            'completeness': 0.25,
            'temporal': 0.2,
            'coverage': 0.15,
            'length': 0.1
        }
        
        final_score = (
            weights['confidence'] * confidence_score +
            weights['completeness'] * completeness_score +
            weights['temporal'] * temporal_score +
            weights['coverage'] * coverage_score +
            weights['length'] * length_score
        )
        
        return final_score
        
    def _calculate_completeness_score(self, path: Dict[str, Any]) -> float:
        """
        计算路径的完整性评分
        
        评估路径是否覆盖了完整的攻击阶段序列
        
        Args:
            path: 攻击路径
            
        Returns:
            完整性评分 (0-1)
        """
        # 定义理想的攻击阶段序列
        ideal_stages = ['initial_access', 'execution', 'persistence', 'privilege_escalation', 
                       'defense_evasion', 'credential_access', 'discovery', 'lateral_movement', 
                       'collection', 'exfiltration', 'command_and_control', 'impact']
        
        # 获取路径中的攻击阶段
        path_stages = path.get('attack_stages', [])
        
        # 如果路径中没有攻击阶段信息，返回低分
        if not path_stages:
            return 0.1
            
        # 计算路径中出现的不同攻击阶段数量
        unique_stages = set(path_stages)
        
        # 计算阶段覆盖率
        coverage_ratio = len(unique_stages) / len(ideal_stages)
        
        # 检查阶段顺序是否合理
        order_score = self._check_stage_order(path_stages, ideal_stages)
        
        # 综合评分 (覆盖率 * 0.7 + 顺序合理性 * 0.3)
        return coverage_ratio * 0.7 + order_score * 0.3
        
    def _check_stage_order(self, path_stages: List[str], ideal_stages: List[str]) -> float:
        """
        检查攻击阶段的顺序是否合理
        
        Args:
            path_stages: 路径中的攻击阶段
            ideal_stages: 理想的攻击阶段序列
            
        Returns:
            顺序合理性评分 (0-1)
        """
        # 如果路径中的阶段少于2个，无法评估顺序
        if len(path_stages) < 2:
            return 0.5
            
        # 计算路径中每个阶段在理想序列中的位置
        stage_positions = []
        for stage in path_stages:
            if stage in ideal_stages:
                stage_positions.append(ideal_stages.index(stage))
            else:
                # 未知阶段，给一个中间位置
                stage_positions.append(len(ideal_stages) // 2)
                
        # 检查位置是否大致递增
        increasing_count = 0
        for i in range(1, len(stage_positions)):
            if stage_positions[i] >= stage_positions[i-1]:
                increasing_count += 1
                
        # 计算顺序合理性比例
        if len(stage_positions) <= 1:
            return 0.5
        else:
            return increasing_count / (len(stage_positions) - 1)
            
    def _calculate_temporal_consistency(self, path: Dict[str, Any]) -> float:
        """
        计算路径的时序一致性评分
        
        评估路径中的时间戳是否符合逻辑顺序
        
        Args:
            path: 攻击路径
            
        Returns:
            时序一致性评分 (0-1)
        """
        timestamps = path.get('timestamps', [])
        
        # 如果路径中没有时间戳信息，返回中等分数
        if not timestamps or len(timestamps) < 2:
            return 0.5
            
        # 尝试解析时间戳
        parsed_times = []
        for ts in timestamps:
            try:
                if ts:
                    parsed_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    parsed_times.append(parsed_time)
            except (ValueError, TypeError):
                # 时间戳格式错误，跳过
                continue
                
        # 如果解析后的时间戳少于2个，无法评估时序一致性
        if len(parsed_times) < 2:
            return 0.5
            
        # 检查时间戳是否按时间顺序排列
        consistent_count = 0
        for i in range(1, len(parsed_times)):
            # 在攻击溯源中，时间应该是逆序的（从最新到最早）
            if parsed_times[i] <= parsed_times[i-1]:
                consistent_count += 1
                
        # 计算时序一致性比例
        return consistent_count / (len(parsed_times) - 1)
        
    def _calculate_critical_node_coverage(self, path: Dict[str, Any]) -> float:
        """
        计算路径的关键节点覆盖率评分
        
        评估路径是否包含关键节点类型
        
        Args:
            path: 攻击路径
            
        Returns:
            关键节点覆盖率评分 (0-1)
        """
        # 定义关键节点类型
        critical_node_types = ['user', 'process', 'file', 'connection', 'registry']
        
        # 获取路径中的节点类型
        path_types = path.get('path_types', [])
        
        # 如果路径中没有节点类型信息，返回低分
        if not path_types:
            return 0.1
            
        # 计算路径中出现的关键节点类型数量
        covered_types = set(path_types).intersection(set(critical_node_types))
        
        # 计算覆盖率
        coverage_ratio = len(covered_types) / len(critical_node_types)
        
        return coverage_ratio
        
    def _calculate_path_length_score(self, path: Dict[str, Any]) -> float:
        """
        计算路径长度评分
        
        评估路径长度是否合适（不过长也不过短）
        
        Args:
            path: 攻击路径
            
        Returns:
            路径长度评分 (0-1)
        """
        path_length = len(path.get('path', []))
        
        # 如果路径为空，返回0分
        if path_length == 0:
            return 0.0
            
        # 定义理想的路径长度范围
        ideal_min_length = 3
        ideal_max_length = 15
        
        # 如果路径长度在理想范围内，给满分
        if ideal_min_length <= path_length <= ideal_max_length:
            return 1.0
            
        # 如果路径过短
        if path_length < ideal_min_length:
            return path_length / ideal_min_length
            
        # 如果路径过长，分数随长度增加而降低
        return max(0.0, 1.0 - (path_length - ideal_max_length) / ideal_max_length)
    
    def _analyze_attack_chain(self, attack_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析攻击链
        
        Args:
            attack_path: 攻击路径
            
        Returns:
            攻击链分析结果
        """
        if not attack_path:
            return {}
        
        # 统计攻击阶段
        stage_counts = defaultdict(int)
        for stage in attack_path['attack_stages']:
            stage_counts[stage] += 1
        
        # 识别关键节点
        key_nodes = []
        for i, (node_id, node_type) in enumerate(zip(attack_path['path'], attack_path['path_types'])):
            if attack_path['confidence_scores'][i] > 0.8:
                key_nodes.append({
                    'node_id': node_id,
                    'node_type': node_type,
                    'confidence': attack_path['confidence_scores'][i],
                    'attack_stage': attack_path['attack_stages'][i]
                })
        
        # 计算攻击复杂度
        complexity = self._calculate_attack_complexity(attack_path)
        
        return {
            'total_stages': len(set(attack_path['attack_stages'])),
            'stage_distribution': dict(stage_counts),
            'key_nodes': key_nodes,
            'complexity_score': complexity,
            'timeline': self._create_attack_timeline(attack_path)
        }
    
    def _calculate_attack_complexity(self, attack_path: Dict[str, Any]) -> float:
        """
        计算攻击复杂度
        
        Args:
            attack_path: 攻击路径
            
        Returns:
            复杂度分数
        """
        # 基于路径长度、阶段数量和置信度计算复杂度
        path_length = len(attack_path['path'])
        stage_diversity = len(set(attack_path['attack_stages']))
        avg_confidence = np.mean(attack_path['confidence_scores']) if attack_path['confidence_scores'] else 0.0
        
        complexity = (path_length * 0.3 + stage_diversity * 0.4 + avg_confidence * 0.3) / 10.0
        return min(complexity, 1.0)
    
    def _create_attack_timeline(self, attack_path: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        创建攻击时间线
        
        Args:
            attack_path: 攻击路径
            
        Returns:
            攻击时间线
        """
        timeline = []
        for i, (node_id, node_type, timestamp, stage) in enumerate(zip(
            attack_path['path'],
            attack_path['path_types'],
            attack_path['timestamps'],
            attack_path['attack_stages']
        )):
            timeline.append({
                'step': i + 1,
                'node_id': node_id,
                'node_type': node_type,
                'timestamp': timestamp,
                'attack_stage': stage,
                'confidence': attack_path['confidence_scores'][i]
            })
        
        return timeline
    
    def _generate_trace_summary(self, best_path: Dict[str, Any], 
                               attack_chain: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成溯源摘要
        
        Args:
            best_path: 最佳攻击路径
            attack_chain: 攻击链分析结果
            
        Returns:
            溯源摘要
        """
        return {
            'path_length': len(best_path['path']),
            'total_stages': attack_chain.get('total_stages', 0),
            'key_nodes_count': len(attack_chain.get('key_nodes', [])),
            'complexity_score': attack_chain.get('complexity_score', 0.0),
            'confidence_avg': np.mean(best_path['confidence_scores']) if best_path['confidence_scores'] else 0.0,
            'attack_stages': list(set(best_path['attack_stages'])),
            'timeline_length': len(attack_chain.get('timeline', [])),
            'risk_level': self._assess_risk_level(attack_chain)
        }
    
    def _assess_risk_level(self, attack_chain: Dict[str, Any]) -> str:
        """
        评估风险等级
        
        Args:
            attack_chain: 攻击链分析结果
            
        Returns:
            风险等级
        """
        complexity = attack_chain.get('complexity_score', 0.0)
        total_stages = attack_chain.get('total_stages', 0)
        
        if complexity > 0.8 and total_stages > 5:
            return 'high'
        elif complexity > 0.5 and total_stages > 3:
            return 'medium'
        else:
            return 'low'
