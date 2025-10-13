"""
攻击路径溯源评估器

实现攻击路径溯源的评估指标，包括回溯成功率、路径相似度等核心评估指标
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


@dataclass
class AttackPath:
    """
    攻击路径数据类
    
    表示一条真实的攻击路径或模型预测的路径
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]
    attack_type: str
    confidence: float
    length: int
    
    def __post_init__(self):
        """后初始化处理"""
        self.length = len(self.nodes)


class PathTracingEvaluator:
    """
    攻击路径溯源评估器
    
    评估模型攻击路径溯源的能力
    """
    
    def __init__(self):
        """初始化路径溯源评估器"""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_path_tracing(self, test_data: List, 
                             ground_truth_paths: List[AttackPath],
                             max_paths_per_graph: int = 5) -> Dict[str, Any]:
        """
        评估攻击路径溯源能力
        
        Args:
            test_data: 测试图数据
            ground_truth_paths: 真实攻击路径列表
            max_paths_per_graph: 每个图最多评估的路径数
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始攻击路径溯源评估")
        
        # 从模型提取真实的预测路径
        predicted_paths = self._extract_real_paths_from_model(test_data, model=getattr(self, 'model', None))
        
        # 计算回溯成功率
        backtrack_success_rate = self._calculate_backtrack_success_rate(
            predicted_paths, ground_truth_paths
        )
        
        # 计算路径相似度
        path_similarities = self._calculate_path_similarities(
            predicted_paths, ground_truth_paths
        )
        
        # 计算精确率@K
        precision_at_k = self._calculate_precision_at_k(
            predicted_paths, ground_truth_paths, k_values=[1, 3, 5, 10]
        )
        
        # 计算平均溯源长度
        avg_trace_length = self._calculate_average_trace_length(predicted_paths)
        
        # 计算路径覆盖率
        path_coverage = self._calculate_path_coverage(predicted_paths, ground_truth_paths)
        
        results = {
            'backtrack_success_rate': backtrack_success_rate,
            'path_similarity': np.mean(path_similarities) if path_similarities else 0.0,
            'precision_at_k': precision_at_k,
            'average_trace_length': avg_trace_length,
            'path_coverage': path_coverage,
            'predicted_path_count': len(predicted_paths),
            'ground_truth_path_count': len(ground_truth_paths),
            'detailed_similarities': path_similarities
        }
        
        self.logger.info("攻击路径溯源评估完成")
        return results
    
    def _extract_real_paths_from_model(self, test_data: List, model) -> List[AttackPath]:
        """
        从模型提取真实的预测路径
        
        Args:
            test_data: 测试数据
            model: 训练好的模型
            
        Returns:
            预测路径列表
        """
        predicted_paths = []
        
        for data in test_data:
            # 使用模型进行推理获取节点嵌入
            with torch.no_grad():
                embeddings = model(data, return_embeddings=True)
            
            # 基于节点嵌入分析攻击路径
            # 使用图遍历算法寻找潜在的攻击路径
            attack_paths = self._analyze_attack_paths_from_embeddings(data, embeddings)
            predicted_paths.extend(attack_paths)
        
        return predicted_paths
    
    def _analyze_attack_paths_from_embeddings(self, data, embeddings) -> List[AttackPath]:
        """
        基于节点嵌入分析攻击路径
        
        Args:
            data: 图数据
            embeddings: 节点嵌入
            
        Returns:
            攻击路径列表
        """
        attack_paths = []
        
        if len(embeddings) == 0:
            return attack_paths
        
        # 寻找高威胁度的节点作为路径起点
        high_threat_nodes = self._identify_threat_nodes(data, embeddings)
        
        # 从高威胁节点开始进行路径分析
        for threat_node in high_threat_nodes[:5]:  # 最多分析5个高威胁节点
            path = self._trace_attack_path(data, embeddings, threat_node)
            if path and len(path.nodes) >= 2:
                attack_paths.append(path)
        
        return attack_paths
    
    def _identify_threat_nodes(self, data, embeddings) -> List[str]:
        """识别高威胁度的节点"""
        threat_nodes = []
        
        # 分析alert类型节点的威胁度
        if 'alert' in embeddings:
            alert_embeddings = embeddings['alert']
            # 使用嵌入向量的模值作为威胁度指标
            threat_scores = torch.norm(alert_embeddings, p=2, dim=1)
            threshold = torch.quantile(threat_scores, 0.8)  # 80%分位数
            
            high_threat_indices = torch.where(threat_scores > threshold)[0]
            for idx in high_threat_indices:
                threat_nodes.append(f"alert_{idx.item()}")
        
        return threat_nodes
    
    def _trace_attack_path(self, data, embeddings, start_node) -> Optional[AttackPath]:
        """追踪攻击路径"""
        try:
            path_nodes = [start_node]
            path_edges = []
            
            # 完整的图遍历算法实现
            node_type = start_node.split('_')[0]
            
            if node_type in embeddings:
                start_idx = int(start_node.split('_')[1])
                if start_idx < embeddings[node_type].size(0):
                    # 使用多种图遍历策略
                    path_nodes, path_edges = self._advanced_path_tracing(
                        start_node, start_idx, embeddings, data
                    )
                    
                    # 计算节点相似度
                    start_embedding = embeddings[node_type][start_idx:start_idx+1]
                    similarities = self._compute_node_similarities(start_embedding, embeddings)
                    
                    for other_type, sim_scores in similarities.items():
                        if other_type != node_type and len(sim_scores) > 0:
                            best_nbr_idx = torch.argmax(sim_scores).item()
                            if best_nbr_idx < len(sim_scores):
                                neighbor_node = f"{other_type}_{best_nbr_idx}"
                                path_nodes.append(neighbor_node)
                                path_edges.append((start_node, neighbor_node))
                                break
            
            if len(path_nodes) >= 2:
                return AttackPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    attack_type="traced",
                    confidence=self._calculate_path_confidence(path_nodes, embeddings),
                    length=len(path_nodes)
                )
            
        except Exception as e:
            self.logger.warning(f"路径追踪失败: {e}")
        
        return None
    
    def _compute_node_similarities(self, target_embedding, embeddings):
        """计算节点相似度"""
        similarities = {}
        
        for ntype, node_embeddings in embeddings.items():
            if target_embedding.size(1) == node_embeddings.size(1):
                # 计算余弦相似度
                sim_scores = F.cosine_similarity(
                    target_embedding, node_embeddings, dim=1
                )
                similarities[ntype] = sim_scores
        
        return similarities
    
    def _advanced_path_tracing(self, start_node: str, start_idx: int, 
                              embeddings: Dict[str, torch.Tensor], 
                              data: HeteroData) -> Tuple[List[str], List[Dict[str, Any]]]:
        """高级路径追踪算法"""
        path_nodes = [start_node]
        path_edges = []
        visited_nodes = {start_node}
        max_path_length = 10  # 最大路径长度
        
        current_node = start_node
        current_idx = start_idx
        current_type = start_node.split('_')[0]
        
        for step in range(max_path_length):
            # 1. 基于嵌入相似性的节点选择
            similar_nodes = self._find_similar_nodes_by_embedding(
                current_node, current_idx, embeddings, visited_nodes
            )
            
            # 2. 基于图结构的邻居节点选择
            neighbor_nodes = self._find_neighbor_nodes_by_graph(
                current_node, current_type, data, visited_nodes
            )
            
            # 3. 综合评分选择下一个节点
            next_node = self._select_next_node(
                similar_nodes, neighbor_nodes, [], 
                current_node, embeddings
            )
            
            if next_node is None:
                break
            
            # 添加到路径
            path_nodes.append(next_node)
            visited_nodes.add(next_node)
            
            # 创建边
            edge = {
                'source': current_node,
                'target': next_node,
                'edge_type': 'traced',
                'confidence': self._calculate_edge_confidence(current_node, next_node, embeddings)
            }
            path_edges.append(edge)
            
            # 更新当前节点
            current_node = next_node
            current_type = next_node.split('_')[0]
            current_idx = int(next_node.split('_')[1])
        
        return path_nodes, path_edges
    
    def _find_similar_nodes_by_embedding(self, current_node: str, current_idx: int,
                                       embeddings: Dict[str, torch.Tensor],
                                       visited_nodes: set) -> List[Dict[str, Any]]:
        """基于嵌入相似性查找节点"""
        similar_nodes = []
        current_type = current_node.split('_')[0]
        
        if current_type in embeddings:
            current_embedding = embeddings[current_type][current_idx:current_idx+1]
            
            for node_type, embedding_matrix in embeddings.items():
                if node_type == current_type:
                    continue
                
                similarities = self._compute_node_similarities(current_embedding, {node_type: embedding_matrix})
                if node_type in similarities:
                    sim_scores = similarities[node_type]
                    for idx, score in enumerate(sim_scores):
                        candidate_node = f"{node_type}_{idx}"
                        if candidate_node not in visited_nodes and score > 0.5:
                            similar_nodes.append({
                                'node': candidate_node,
                                'similarity': score.item(),
                                'type': 'embedding_similarity'
                            })
        
        return sorted(similar_nodes, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def _find_neighbor_nodes_by_graph(self, current_node: str, current_type: str,
                                    data: HeteroData, visited_nodes: set) -> List[Dict[str, Any]]:
        """基于图结构查找邻居节点"""
        neighbor_nodes = []
        
        try:
            # 查找所有与当前节点相连的边
            for edge_type in data.edge_types:
                if edge_type[0] == current_type:  # 出边
                    edge_index = data[edge_type].edge_index
                    current_idx = int(current_node.split('_')[1])
                    
                    # 找到当前节点的出边
                    outgoing_edges = edge_index[1][edge_index[0] == current_idx]
                    for target_idx in outgoing_edges:
                        target_node = f"{edge_type[2]}_{target_idx.item()}"
                        if target_node not in visited_nodes:
                            neighbor_nodes.append({
                                'node': target_node,
                                'edge_type': edge_type[1],
                                'type': 'graph_neighbor'
                            })
                
                elif edge_type[2] == current_type:  # 入边
                    edge_index = data[edge_type].edge_index
                    current_idx = int(current_node.split('_')[1])
                    
                    # 找到当前节点的入边
                    incoming_edges = edge_index[0][edge_index[1] == current_idx]
                    for source_idx in incoming_edges:
                        source_node = f"{edge_type[0]}_{source_idx.item()}"
                        if source_node not in visited_nodes:
                            neighbor_nodes.append({
                                'node': source_node,
                                'edge_type': edge_type[1],
                                'type': 'graph_neighbor'
                            })
        except Exception as e:
            self.logger.warning(f"图结构邻居查找失败: {e}")
        
        return neighbor_nodes[:5]
    
    def _select_next_node(self, similar_nodes: List[Dict[str, Any]], 
                         neighbor_nodes: List[Dict[str, Any]],
                         temporal_nodes: List[Dict[str, Any]],
                         current_node: str, embeddings: Dict[str, torch.Tensor]) -> Optional[str]:
        """综合评分选择下一个节点"""
        candidates = []
        
        # 添加相似性节点
        for node_info in similar_nodes:
            candidates.append({
                'node': node_info['node'],
                'score': node_info['similarity'] * 0.4,  # 相似性权重
                'type': 'similarity'
            })
        
        # 添加邻居节点
        for node_info in neighbor_nodes:
            candidates.append({
                'node': node_info['node'],
                'score': 0.6,  # 图结构权重
                'type': 'neighbor'
            })
        
        if not candidates:
            return None
        
        # 选择得分最高的节点
        best_candidate = max(candidates, key=lambda x: x['score'])
        return best_candidate['node']
    
    def _calculate_edge_confidence(self, source_node: str, target_node: str,
                                 embeddings: Dict[str, torch.Tensor]) -> float:
        """计算边的置信度"""
        try:
            source_type = source_node.split('_')[0]
            target_type = target_node.split('_')[0]
            source_idx = int(source_node.split('_')[1])
            target_idx = int(target_node.split('_')[1])
            
            if source_type in embeddings and target_type in embeddings:
                source_embedding = embeddings[source_type][source_idx:source_idx+1]
                target_embedding = embeddings[target_type][target_idx:target_idx+1]
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(source_embedding, target_embedding, dim=1)
                return similarity.item()
        except Exception as e:
            self.logger.warning(f"计算边置信度失败: {e}")
        
        return 0.5  # 默认置信度
    
    def _calculate_path_confidence(self, path_nodes, embeddings) -> float:
        """计算路径置信度"""
        if not path_nodes:
            return 0.0
        
        confidence_scores = []
        for node in path_nodes:
            node_type, node_idx = node.split('_', 1)
            try:
                idx = int(node_idx)
                if node_type in embeddings and idx < embeddings[node_type].size(0):
                    node_embedding = embeddings[node_type][idx:idx+1]
                    # 使用嵌入向量的模值作为置信度
                    confidence = float(torch.norm(node_embedding).item())
                    confidence_scores.append(min(confidence / 10.0, 1.0))  # 归一化到[0,1]
            except (ValueError, IndexError):
                confidence_scores.append(0.5)
        
        return float(np.mean(confidence_scores)) if confidence_scores else 0.0
    
    def _calculate_backtrack_success_rate(self, predicted_paths: List[AttackPath], 
                                         ground_truth_paths: List[AttackPath]) -> float:
        """
        计算回溯成功率
        
        Args:
            predicted_paths: 预测路径列表
            ground_truth_paths: 真实路径列表
            
        Returns:
            回溯成功率
        """
        if not ground_truth_paths:
            return 0.0
        
        successful_backtracks = 0
        correct_endpoints = sum(len(path.nodes) for path in ground_truth_paths)
        
        for pred_path in predicted_paths:
            for true_path in ground_truth_paths:
                if self._paths_share_endpoint(pred_path, true_path):
                    if self._is_valid_backtrack(pred_path, true_path):
                        successful_backtracks += 1
                        break
        
        return successful_backtracks / len(ground_truth_paths) if ground_truth_paths else 0.0
    
    def _calculate_path_similarities(self, predicted_paths: List[AttackPath], 
                                   ground_truth_paths: List[AttackPath]) -> List[float]:
        """
        计算路径相似度
        
        Args:
            predicted_paths: 预测路径列表
            ground_truth_paths: 真实路径列表
            
        Returns:
            相似度列表
        """
        similarities = []
        
        for pred_path in predicted_paths:
            max_similarity = 0.0
            for true_path in ground_truth_paths:
                # 计算多种相似度指标
                edit_sim = self._calculate_edit_distance_similarity(pred_path, true_path)
                overlap_sim = self._calculate_overlap_similarity(pred_path, true_path)
                sequence_sim = self._calculate_sequence_similarity(pred_path, true_path)
                temporal_sim = self._calculate_temporal_similarity(pred_path, true_path)
                
                # 加权综合相似度
                combined_sim = (0.3 * edit_sim + 0.3 * overlap_sim + 
                               0.2 * sequence_sim + 0.2 * temporal_sim)
                max_similarity = max(max_similarity, combined_sim)
            
            similarities.append(max_similarity)
        
        return similarities
    
    def _calculate_sequence_similarity(self, pred_path: AttackPath, true_path: AttackPath) -> float:
        """计算序列相似度（考虑顺序）"""
        if not pred_path.path or not true_path.path:
            return 0.0
        
        # 使用最长公共子序列算法
        m, n = len(pred_path.path), len(true_path.path)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_path.path[i-1] == true_path.path[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_length = max(m, n)
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _calculate_temporal_similarity(self, pred_path: AttackPath, true_path: AttackPath) -> float:
        """计算时间相似度"""
        if not pred_path.timeline or not true_path.timeline:
            return 0.0
        
        # 提取时间戳
        pred_times = [event.get('timestamp', 0) for event in pred_path.timeline]
        true_times = [event.get('timestamp', 0) for event in true_path.timeline]
        
        if not pred_times or not true_times:
            return 0.0
        
        # 计算时间序列的相似度
        min_len = min(len(pred_times), len(true_times))
        if min_len == 0:
            return 0.0
        
        # 计算时间差的相似度
        pred_diffs = [pred_times[i+1] - pred_times[i] for i in range(len(pred_times)-1)]
        true_diffs = [true_times[i+1] - true_times[i] for i in range(len(true_times)-1)]
        
        if not pred_diffs or not true_diffs:
            return 0.0
        
        # 计算时间差序列的相似度
        min_diff_len = min(len(pred_diffs), len(true_diffs))
        if min_diff_len == 0:
            return 0.0
        
        # 计算相对时间差的相似度
        total_diff = 0.0
        for i in range(min_diff_len):
            diff = abs(pred_diffs[i] - true_diffs[i])
            max_diff = max(pred_diffs[i], true_diffs[i])
            if max_diff > 0:
                total_diff += diff / max_diff
        
        avg_diff = total_diff / min_diff_len
        return max(0.0, 1.0 - avg_diff)
    
    def _calculate_precision_at_k(self, predicted_paths: List[AttackPath], 
                                 ground_truth_paths: List[AttackPath], 
                                 k_values: List[int]) -> Dict[str, float]:
        """
        计算精确率@K
        
        Args:
            predicted_paths: 预测路径列表
            ground_truth_paths: 真实路径列表
            k_values: K值列表
            
        Returns:
            精确率@K字典
        """
        precision_at_k = {}
        
        for k in k_values:
            if k > len(predicted_paths):
                precision_at_k[f'P@{k}'] = 0.0
                continue
            
            # 取前K条路径进行评估
            top_k_paths = predicted_paths[:k]
            
            # 计算有多少条路径是正确的
            correct_paths = 0
            for pred_path in top_k_paths:
                for true_path in ground_truth_paths:
                    # 计算多种相似度指标
                    edit_sim = self._calculate_edit_distance_similarity(pred_path, true_path)
                    overlap_sim = self._calculate_overlap_similarity(pred_path, true_path)
                    sequence_sim = self._calculate_sequence_similarity(pred_path, true_path)
                    temporal_sim = self._calculate_temporal_similarity(pred_path, true_path)
                    
                    # 加权综合相似度
                    similarity = (0.3 * edit_sim + 0.3 * overlap_sim + 
                                0.2 * sequence_sim + 0.2 * temporal_sim)
                    
                    if similarity >= 0.8:  # 相似度阈值
                        correct_paths += 1
                        break
            
            precision_at_k[f'P@{k}'] = correct_paths / k if k > 0 else 0.0
        
        return precision_at_k
    
    def _calculate_average_trace_length(self, predicted_paths: List[AttackPath]) -> float:
        """
        计算平均溯源长度
        
        Args:
            predicted_paths: 预测路径列表
            
        Returns:
            平均溯源长度
        """
        if not predicted_paths:
            return 0.0
        
        total_length = sum(path.length for path in predicted_paths)
        return total_length / len(predicted_paths)
    
    def _calculate_path_coverage(self, predicted_paths: List[AttackPath], 
                               ground_truth_paths: List[AttackPath]) -> float:
        """
        计算路径覆盖率
        
        Args:
            predicted_paths: 预测路径列表
            ground_truth_paths: 真实路径列表
            
        Returns:
            路径覆盖率
        """
        if not ground_truth_paths:
            return 0.0
        
        # 收集所有真实路径中的节点
        all_true_nodes = set()
        for true_path in ground_truth_paths:
            all_true_nodes.update(true_path.path)
        
        if not all_true_nodes:
            return 0.0
        
        # 收集所有预测路径中的节点
        all_pred_nodes = set()
        for pred_path in predicted_paths:
            all_pred_nodes.update(pred_path.path)
        
        # 计算覆盖率
        covered_nodes = all_true_nodes.intersection(all_pred_nodes)
        coverage = len(covered_nodes) / len(all_true_nodes) if all_true_nodes else 0.0
        
        return coverage
    
    def _paths_share_endpoint(self, path1: AttackPath, path2: AttackPath) -> bool:
        """检查两条路径是否共享终点"""
        if not path1.nodes or not path2.nodes:
            return False
        return path1.nodes[-1] == path2.nodes[-1]
    
    def _is_valid_backtrack(self, pred_path: AttackPath, true_path: AttackPath) -> bool:
        """检查是否为有效的回溯"""
        # 完整的回溯有效性检查
        # 1. 检查路径长度相似性
        length_diff = abs(pred_path.length - true_path.length)
        if length_diff > 3:  # 路径长度差异过大
            return False
        
        # 2. 检查节点重叠度
        common_nodes = set(pred_path.nodes) & set(true_path.nodes)
        overlap_ratio = len(common_nodes) / max(len(pred_path.nodes), len(true_path.nodes))
        if overlap_ratio < 0.3:  # 重叠度太低
            return False
        
        # 3. 检查路径方向性
        if pred_path.nodes and true_path.nodes:
            pred_direction = self._analyze_path_direction(pred_path)
            true_direction = self._analyze_path_direction(true_path)
            if pred_direction != true_direction:
                return False
        
        # 4. 检查时间序列合理性
        if hasattr(pred_path, 'timestamps') and hasattr(true_path, 'timestamps'):
            time_consistency = self._check_time_consistency(pred_path, true_path)
            if not time_consistency:
                return False
        
        # 5. 检查攻击阶段连续性
        if hasattr(pred_path, 'stages') and hasattr(true_path, 'stages'):
            stage_consistency = self._check_stage_consistency(pred_path, true_path)
            if not stage_consistency:
                return False
        
        return True
    
    def _analyze_path_direction(self, path: AttackPath) -> str:
        """分析路径方向"""
        if not path.nodes or len(path.nodes) < 2:
            return 'unknown'
        
        # 分析节点类型的变化趋势
        node_types = [node.split('_')[0] for node in path.nodes]
        
        # 检查是否有明显的方向性
        if len(set(node_types)) == 1:
            return 'same_type'
        
        # 检查是否有层级变化
        type_hierarchy = ['user', 'process', 'file', 'network', 'registry']
        type_indices = [type_hierarchy.index(t) if t in type_hierarchy else -1 for t in node_types]
        
        if all(idx >= 0 for idx in type_indices):
            if type_indices[-1] > type_indices[0]:
                return 'escalating'
            elif type_indices[-1] < type_indices[0]:
                return 'de-escalating'
        
        return 'mixed'
    
    def _check_time_consistency(self, pred_path: AttackPath, true_path: AttackPath) -> bool:
        """检查时间序列一致性"""
        try:
            pred_times = getattr(pred_path, 'timestamps', [])
            true_times = getattr(true_path, 'timestamps', [])
            
            if not pred_times or not true_times:
                return True  # 没有时间信息，认为一致
            
            # 检查时间序列是否单调递增
            pred_monotonic = all(pred_times[i] <= pred_times[i+1] for i in range(len(pred_times)-1))
            true_monotonic = all(true_times[i] <= true_times[i+1] for i in range(len(true_times)-1))
            
            return pred_monotonic and true_monotonic
        except Exception:
            return True
    
    def _check_stage_consistency(self, pred_path: AttackPath, true_path: AttackPath) -> bool:
        """检查攻击阶段一致性"""
        try:
            pred_stages = getattr(pred_path, 'stages', [])
            true_stages = getattr(true_path, 'stages', [])
            
            if not pred_stages or not true_stages:
                return True  # 没有阶段信息，认为一致
            
            # 检查阶段序列是否合理
            valid_stages = ['initial_access', 'execution', 'persistence', 'privilege_escalation',
                           'defense_evasion', 'credential_access', 'discovery', 'lateral_movement',
                           'collection', 'exfiltration', 'command_and_control', 'impact']
            
            pred_valid = all(stage in valid_stages for stage in pred_stages)
            true_valid = all(stage in valid_stages for stage in true_stages)
            
            return pred_valid and true_valid
        except Exception:
            return True
    
    def _calculate_edit_distance_similarity(self, path1: AttackPath, path2: AttackPath) -> float:
        """基于编辑距离计算路径相似度"""
        # 完整的编辑距离计算实现
        nodes1, nodes2 = path1.nodes, path2.nodes
        
        if not nodes1 or not nodes2:
            return 1.0 if not nodes1 and not nodes2 else 0.0
        
        # 使用动态规划计算编辑距离
        m, n = len(nodes1), len(nodes2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nodes1[i-1] == nodes2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # 删除
                        dp[i][j-1],      # 插入
                        dp[i-1][j-1]     # 替换
                    )
        
        # 计算相似度
        max_len = max(m, n)
        edit_distance = dp[m][n]
        similarity = 1.0 - (edit_distance / max_len)
        
        return max(0.0, similarity)
    
    def _calculate_overlap_similarity(self, path1: AttackPath, path2: AttackPath) -> float:
        """计算路径重叠度相似度"""
        set1, set2 = set(path1.nodes), set(path2.nodes)
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_path_match(self, pred_path: AttackPath, true_path: AttackPath, threshold: float = 0.8) -> bool:
        """检查预测路径是否与真实路径匹配"""
        similarity = self._calculate_overlap_similarity(pred_path, true_path)
        return similarity >= threshold
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            results: 评估结果
            
        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("攻击路径溯源评估报告")
        report_lines.append("=" * 50)
        
        report_lines.append(f"回溯成功率: {results.get('backtrack_success_rate', 0.0):.4f}")
        report_lines.append(f"路径相似度: {results.get('path_similarity', 0.0):.4f}")
        report_lines.append(f"平均溯源长度: {results.get('average_trace_length', 0.0):.4f}")
        report_lines.append(f"路径覆盖率: {results.get('path_coverage', 0.0):.4f}")
        
        # 精确率@K
        precision_k = results.get('precision_at_k', {})
        report_lines.append("\n精确율@K:")
        for k, precision in precision_k.items():
            report_lines.append(f"  {k}: {precision:.4f}")
        
        report_lines.append(f"\n预测路径数: {results.get('predicted_path_count', 0)}")
        report_lines.append(f"真实路径数: {results.get('ground_truth_path_count', 0)}")
        
        return "\n".join(report_lines)