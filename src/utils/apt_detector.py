"""
APT攻击检测与溯源模块

实现基于T-HGNN的APT攻击检测和攻击链溯源功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime, timedelta

try:
    from ..models.pyg_t_hgnn import PyG_T_HGNN
    from ..data.pyg_loader import PyG_LinuxAPTDataLoader
except ImportError:
    PyG_T_HGNN = None
    PyG_LinuxAPTDataLoader = None


class APTDetector:
    """APT攻击检测器"""
    
    def __init__(self, model: PyG_T_HGNN, config):
        """
        初始化APT检测器
        
        Args:
            model: 训练好的T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 攻击模式定义
        self.attack_patterns = self._define_attack_patterns()
        
        # 检测阈值
        self.anomaly_threshold = 0.7
        self.suspicious_threshold = 0.5
        
    def _define_attack_patterns(self) -> Dict[str, List[str]]:
        """定义攻击模式"""
        return {
            'lateral_movement': [
                'alert_detected_on_host',
                'alert_by_user',
                'alert_involves_process',
                'alert_connects_to_ip'
            ],
            'data_exfiltration': [
                'alert_involves_file',
                'alert_connects_to_ip',
                'alert_connects_to_domain'
            ],
            'privilege_escalation': [
                'alert_by_user',
                'alert_involves_process',
                'alert_involves_file'
            ],
            'persistence': [
                'alert_involves_file',
                'alert_involves_process',
                'alert_detected_on_host'
            ]
        }
    
    def detect_anomalies(self, data, embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        检测异常节点
        
        Args:
            data: 异构图数据
            embeddings: 节点嵌入
            
        Returns:
            异常检测结果
        """
        self.logger.info("开始异常检测")
        
        anomalies = {
            'suspicious_nodes': {},
            'attack_patterns': {},
            'risk_scores': {},
            'recommendations': []
        }
        
        # 对每种节点类型进行异常检测
        for ntype in data.node_types:
            if ntype in embeddings and data[ntype].x is not None:
                node_embeddings = embeddings[ntype]
                
                # 计算异常分数（使用简单的统计方法）
                anomaly_scores = self._calculate_anomaly_scores(node_embeddings)
                
                # 识别异常节点
                suspicious_indices = torch.where(anomaly_scores > self.suspicious_threshold)[0]
                anomaly_indices = torch.where(anomaly_scores > self.anomaly_threshold)[0]
                
                if len(suspicious_indices) > 0:
                    anomalies['suspicious_nodes'][ntype] = {
                        'indices': suspicious_indices.tolist(),
                        'scores': anomaly_scores[suspicious_indices].tolist()
                    }
                
                if len(anomaly_indices) > 0:
                    anomalies['risk_scores'][ntype] = {
                        'indices': anomaly_indices.tolist(),
                        'scores': anomaly_scores[anomaly_indices].tolist()
                    }
        
        # 检测攻击模式
        anomalies['attack_patterns'] = self._detect_attack_patterns(data, anomalies)
        
        # 生成建议
        anomalies['recommendations'] = self._generate_recommendations(anomalies)
        
        self.logger.info(f"检测到 {len(anomalies['suspicious_nodes'])} 种类型的可疑节点")
        return anomalies
    
    def _calculate_anomaly_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算异常分数"""
        # 使用Z-score方法计算异常分数
        mean_emb = torch.mean(embeddings, dim=0)
        std_emb = torch.std(embeddings, dim=0)
        
        # 避免除零
        std_emb = torch.where(std_emb == 0, torch.ones_like(std_emb), std_emb)
        
        # 计算每个节点到中心的距离
        distances = torch.norm(embeddings - mean_emb, dim=1)
        
        # 归一化到[0,1]范围
        max_dist = torch.max(distances)
        if max_dist > 0:
            anomaly_scores = distances / max_dist
        else:
            anomaly_scores = torch.zeros_like(distances)
        
        return anomaly_scores
    
    def _detect_attack_patterns(self, data, anomalies: Dict) -> Dict[str, Any]:
        """检测攻击模式"""
        patterns = {}
        
        for pattern_name, pattern_edges in self.attack_patterns.items():
            pattern_score = 0.0
            pattern_nodes = set()
            
            # 检查是否存在相关的边类型
            for edge_type in data.edge_types:
                if any(edge in str(edge_type) for edge in pattern_edges):
                    if edge_type in data.edge_index_dict:
                        edge_count = data[edge_type].edge_index.size(1)
                        pattern_score += edge_count * 0.1  # 简单的评分机制
            
            if pattern_score > 0:
                patterns[pattern_name] = {
                    'score': pattern_score,
                    'confidence': min(pattern_score / 100.0, 1.0),
                    'affected_nodes': list(pattern_nodes)
                }
        
        return patterns
    
    def _generate_recommendations(self, anomalies: Dict) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 基于检测到的异常类型生成建议
        if anomalies['suspicious_nodes']:
            recommendations.append("发现可疑节点，建议进行深入调查")
        
        if anomalies['attack_patterns']:
            for pattern_name, pattern_info in anomalies['attack_patterns'].items():
                if pattern_info['confidence'] > 0.5:
                    recommendations.append(f"检测到{pattern_name}攻击模式，置信度: {pattern_info['confidence']:.2f}")
        
        if not recommendations:
            recommendations.append("未发现明显的攻击活动")
        
        return recommendations
    
    def trace_attack_chain(self, data, anomalies: Dict, max_depth: int = 5) -> Dict[str, Any]:
        """
        溯源攻击链
        
        Args:
            data: 异构图数据
            anomalies: 异常检测结果
            max_depth: 最大溯源深度
            
        Returns:
            攻击链溯源结果
        """
        self.logger.info("开始攻击链溯源")
        
        attack_chains = {
            'chains': [],
            'timeline': [],
            'affected_assets': set(),
            'attack_vectors': []
        }
        
        # 限制处理的异常节点数量，避免性能问题
        max_nodes_per_type = 20  # 进一步减少到20个节点
        processed_count = 0
        total_limit = 50  # 总共最多处理50个异常节点
        max_processing_time = 30  # 最大处理时间30秒
        
        import time
        start_time = time.time()
        
        # 从异常节点开始溯源
        for ntype, node_info in anomalies['suspicious_nodes'].items():
            if processed_count >= total_limit:
                self.logger.warning(f"达到处理限制({total_limit})，停止溯源")
                break
                
            if time.time() - start_time > max_processing_time:
                self.logger.warning(f"达到时间限制({max_processing_time}秒)，停止溯源")
                break
                
            # 按异常分数排序，优先处理高分数的节点
            sorted_indices = sorted(zip(node_info['indices'], node_info['scores']), 
                                  key=lambda x: x[1], reverse=True)
            
            for idx, score in sorted_indices[:max_nodes_per_type]:
                if processed_count >= total_limit:
                    break
                    
                if time.time() - start_time > max_processing_time:
                    break
                    
                if score > self.anomaly_threshold:
                    try:
                        # 显示进度
                        if processed_count % 10 == 0:
                            elapsed = time.time() - start_time
                            self.logger.info(f"溯源进度: {processed_count}/{total_limit}, 耗时: {elapsed:.1f}秒")
                        
                        chain = self._trace_single_chain(data, ntype, idx, max_depth)
                        if chain:
                            attack_chains['chains'].append(chain)
                            attack_chains['affected_assets'].update(chain['nodes'])
                        processed_count += 1
                    except Exception as e:
                        self.logger.warning(f"溯源节点 {ntype}:{idx} 时出错: {e}")
                        continue
        
        # 构建时间线
        attack_chains['timeline'] = self._build_attack_timeline(attack_chains['chains'])
        
        # 识别攻击向量
        attack_chains['attack_vectors'] = self._identify_attack_vectors(attack_chains['chains'])
        
        self.logger.info(f"溯源到 {len(attack_chains['chains'])} 条攻击链")
        return attack_chains
    
    def _trace_single_chain(self, data, start_ntype: str, start_idx: int, max_depth: int) -> Optional[Dict]:
        """溯源单条攻击链 - 优化版本"""
        chain = {
            'start_node': (start_ntype, start_idx),
            'nodes': [(start_ntype, start_idx)],
            'edges': [],
            'depth': 0
        }
        
        # 使用BFS进行溯源，但限制节点数量和时间
        queue = deque([(start_ntype, start_idx, 0)])
        visited = {(start_ntype, start_idx)}
        max_nodes_per_chain = 50  # 减少到50个节点
        max_iterations = 1000  # 最大迭代次数
        iteration_count = 0
        
        # 预构建边索引映射以提高查找效率
        edge_maps = self._build_edge_maps(data)
        
        while (queue and chain['depth'] < max_depth and 
               len(chain['nodes']) < max_nodes_per_chain and 
               iteration_count < max_iterations):
            
            current_ntype, current_idx, depth = queue.popleft()
            iteration_count += 1
            
            # 查找相邻节点 - 使用预构建的映射
            neighbors = self._find_neighbors_fast(edge_maps, current_ntype, current_idx)
            
            for neighbor_ntype, neighbor_idx in neighbors:
                neighbor_node = (neighbor_ntype, neighbor_idx)
                if (neighbor_node not in visited and 
                    len(chain['nodes']) < max_nodes_per_chain):
                    
                    chain['nodes'].append(neighbor_node)
                    chain['edges'].append((current_ntype, current_idx, neighbor_ntype, neighbor_idx))
                    visited.add(neighbor_node)
                    queue.append((neighbor_ntype, neighbor_idx, depth + 1))
            
            chain['depth'] = max(chain['depth'], depth)
        
        return chain if len(chain['nodes']) > 1 else None
    
    def _build_edge_maps(self, data) -> Dict:
        """预构建边索引映射以提高查找效率"""
        edge_maps = {}
        
        for edge_type in data.edge_types:
            if edge_type in data.edge_index_dict:
                edge_index = data[edge_type].edge_index
                src_type, _, dst_type = edge_type
                
                # 构建出边映射
                if src_type not in edge_maps:
                    edge_maps[src_type] = {}
                if 'out_edges' not in edge_maps[src_type]:
                    edge_maps[src_type]['out_edges'] = {}
                
                # 构建入边映射
                if dst_type not in edge_maps:
                    edge_maps[dst_type] = {}
                if 'in_edges' not in edge_maps[dst_type]:
                    edge_maps[dst_type]['in_edges'] = {}
                
                # 填充映射
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    dst_idx = edge_index[1, i].item()
                    
                    # 出边映射
                    if src_idx not in edge_maps[src_type]['out_edges']:
                        edge_maps[src_type]['out_edges'][src_idx] = []
                    edge_maps[src_type]['out_edges'][src_idx].append((dst_type, dst_idx))
                    
                    # 入边映射
                    if dst_idx not in edge_maps[dst_type]['in_edges']:
                        edge_maps[dst_type]['in_edges'][dst_idx] = []
                    edge_maps[dst_type]['in_edges'][dst_idx].append((src_type, src_idx))
        
        return edge_maps
    
    def _find_neighbors_fast(self, edge_maps: Dict, ntype: str, idx: int) -> List[Tuple[str, int]]:
        """快速查找邻居节点"""
        neighbors = []
        
        if ntype in edge_maps:
            # 查找出边
            if 'out_edges' in edge_maps[ntype] and idx in edge_maps[ntype]['out_edges']:
                neighbors.extend(edge_maps[ntype]['out_edges'][idx])
            
            # 查找入边
            if 'in_edges' in edge_maps[ntype] and idx in edge_maps[ntype]['in_edges']:
                neighbors.extend(edge_maps[ntype]['in_edges'][idx])
        
        return neighbors
    
    def _build_attack_timeline(self, chains: List[Dict]) -> List[Dict]:
        """构建攻击时间线"""
        timeline = []
        
        for i, chain in enumerate(chains):
            timeline.append({
                'chain_id': i,
                'start_time': datetime.now() - timedelta(hours=len(chain['nodes'])),
                'end_time': datetime.now(),
                'duration': len(chain['nodes']),
                'nodes_count': len(chain['nodes']),
                'severity': 'high' if len(chain['nodes']) > 5 else 'medium'
            })
        
        # 按时间排序
        timeline.sort(key=lambda x: x['start_time'])
        return timeline
    
    def _identify_attack_vectors(self, chains: List[Dict]) -> List[str]:
        """识别攻击向量"""
        vectors = []
        
        for chain in chains:
            # 分析攻击链中的节点类型
            node_types = [node[0] for node in chain['nodes']]
            
            if 'user' in node_types and 'process' in node_types:
                vectors.append("用户权限滥用")
            if 'file' in node_types and 'process' in node_types:
                vectors.append("恶意文件执行")
            if 'ip' in node_types and 'domain' in node_types:
                vectors.append("网络通信异常")
            if 'host' in node_types and 'agent' in node_types:
                vectors.append("主机入侵")
        
        return list(set(vectors))  # 去重
    
    def generate_report(self, anomalies: Dict, attack_chains: Dict) -> Dict[str, Any]:
        """生成检测报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_suspicious_nodes': sum(len(nodes['indices']) for nodes in anomalies['suspicious_nodes'].values()),
                'total_attack_chains': len(attack_chains['chains']),
                'affected_assets': len(attack_chains['affected_assets']),
                'attack_vectors': len(attack_chains['attack_vectors'])
            },
            'anomalies': anomalies,
            'attack_chains': attack_chains,
            'risk_assessment': self._assess_risk(anomalies, attack_chains),
            'recommendations': anomalies['recommendations']
        }
        
        return report
    
    def _assess_risk(self, anomalies: Dict, attack_chains: Dict) -> Dict[str, Any]:
        """评估风险等级"""
        risk_score = 0.0
        
        # 基于异常节点数量
        total_suspicious = sum(len(nodes['indices']) for nodes in anomalies['suspicious_nodes'].values())
        risk_score += min(total_suspicious * 0.1, 0.4)
        
        # 基于攻击链数量
        risk_score += min(len(attack_chains['chains']) * 0.2, 0.4)
        
        # 基于攻击模式
        for pattern_info in anomalies['attack_patterns'].values():
            risk_score += pattern_info['confidence'] * 0.1
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score > 0.7:
            risk_level = "高"
        elif risk_score > 0.4:
            risk_level = "中"
        else:
            risk_level = "低"
        
        return {
            'score': risk_score,
            'level': risk_level,
            'description': f"当前系统风险等级为{risk_level}，风险分数: {risk_score:.2f}"
        }

