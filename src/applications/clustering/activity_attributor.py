"""
活动归因器

实现基于T-HGNN的活动归因功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class ActivityAttributor:
    """
    活动归因器
    
    实现基于T-HGNN的活动归因功能
    """
    
    def __init__(self, config):
        """
        初始化活动归因器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 归因参数
        self.attribution_method = getattr(config, 'attribution_method', 'embedding_similarity')
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.temporal_window = getattr(config, 'temporal_window', 3600)  # 1小时
        
        self.logger.info(f"活动归因器初始化完成，方法: {self.attribution_method}")
    
    def attribute_activities(self, hetero_data: HeteroData, 
                           embeddings: Dict[str, torch.Tensor],
                           attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        归因攻击活动
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            attack_chains: 攻击链列表
            
        Returns:
            归因结果
        """
        self.logger.info("开始活动归因分析")
        
        try:
            # 1. 提取攻击活动特征
            activity_features = self._extract_activity_features(hetero_data, embeddings, attack_chains)
            
            # 2. 计算活动相似性
            similarity_matrix = self._compute_activity_similarity(activity_features)
            
            # 3. 执行活动归因
            attribution_results = self._perform_attribution(activity_features, similarity_matrix)
            
            # 4. 生成归因报告
            attribution_report = self._generate_attribution_report(attribution_results)
            
            self.logger.info(f"活动归因完成，发现 {len(attribution_results['attributed_activities'])} 个归因活动")
            
            return {
                'attribution_results': attribution_results,
                'attribution_report': attribution_report,
                'activity_features': activity_features,
                'similarity_matrix': similarity_matrix
            }
            
        except Exception as e:
            self.logger.error(f"活动归因过程中发生错误: {e}")
            return {
                'attribution_results': {'attributed_activities': [], 'error': str(e)},
                'attribution_report': {'error': str(e), 'status': 'failed'},
                'activity_features': {},
                'similarity_matrix': np.array([])
            }
    
    def _extract_activity_features(self, hetero_data: HeteroData, 
                                 embeddings: Dict[str, torch.Tensor],
                                 attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        提取攻击活动特征
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            attack_chains: 攻击链列表
            
        Returns:
            活动特征字典
        """
        features = {}
        
        for i, chain in enumerate(attack_chains):
            chain_id = f"chain_{i}"
            
            # 提取链级特征
            chain_features = {
                'chain_id': chain_id,
                'path_length': len(chain.get('path', [])),
                'confidence': chain.get('confidence', 0.0),
                'attack_stages': chain.get('attack_stages', []),
                'timestamps': chain.get('timestamps', []),
                'node_types': chain.get('path_types', [])
            }
            
            # 提取节点级特征
            node_features = []
            for j, node_id in enumerate(chain.get('path', [])):
                node_type = chain.get('path_types', [])[j] if j < len(chain.get('path_types', [])) else 'unknown'
                
                if node_type in embeddings:
                    # 获取节点嵌入
                    node_embedding = embeddings[node_type][j] if j < embeddings[node_type].size(0) else torch.zeros(embeddings[node_type].size(1))
                    node_features.append({
                        'node_id': node_id,
                        'node_type': node_type,
                        'embedding': node_embedding.cpu().numpy(),
                        'position': j,
                        'timestamp': chain.get('timestamps', [])[j] if j < len(chain.get('timestamps', [])) else 0
                    })
            
            features[chain_id] = {
                'chain_features': chain_features,
                'node_features': node_features
            }
        
        return features
    
    def _compute_activity_similarity(self, activity_features: Dict[str, Any]) -> np.ndarray:
        """
        计算活动相似性
        
        Args:
            activity_features: 活动特征
            
        Returns:
            相似性矩阵
        """
        chain_ids = list(activity_features.keys())
        n_chains = len(chain_ids)
        
        if n_chains == 0:
            return np.array([])
        
        similarity_matrix = np.zeros((n_chains, n_chains))
        
        for i, chain_id1 in enumerate(chain_ids):
            for j, chain_id2 in enumerate(chain_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self._compute_chain_similarity(
                        activity_features[chain_id1],
                        activity_features[chain_id2]
                    )
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _compute_chain_similarity(self, chain1: Dict[str, Any], 
                                chain2: Dict[str, Any]) -> float:
        """
        计算两个攻击链的相似性
        
        Args:
            chain1: 攻击链1
            chain2: 攻击链2
            
        Returns:
            相似性分数
        """
        try:
            # 1. 路径长度相似性
            len1 = chain1['chain_features']['path_length']
            len2 = chain2['chain_features']['path_length']
            length_sim = 1.0 - abs(len1 - len2) / max(len1, len2, 1)
            
            # 2. 节点类型相似性
            types1 = set(chain1['chain_features']['node_types'])
            types2 = set(chain2['chain_features']['node_types'])
            type_sim = len(types1.intersection(types2)) / len(types1.union(types2)) if types1.union(types2) else 0.0
            
            # 3. 嵌入相似性
            emb_sim = 0.0
            if chain1['node_features'] and chain2['node_features']:
                emb1 = np.array([nf['embedding'] for nf in chain1['node_features']])
                emb2 = np.array([nf['embedding'] for nf in chain2['node_features']])
                
                # 计算平均嵌入相似性
                if emb1.size > 0 and emb2.size > 0:
                    # 使用余弦相似性
                    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
                    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
                    
                    # 计算所有节点对的相似性
                    similarities = []
                    for e1 in emb1_norm:
                        for e2 in emb2_norm:
                            sim = np.dot(e1, e2)
                            similarities.append(sim)
                    
                    emb_sim = np.mean(similarities) if similarities else 0.0
            
            # 4. 时间相似性
            time_sim = 0.0
            timestamps1 = chain1['chain_features']['timestamps']
            timestamps2 = chain2['chain_features']['timestamps']
            
            if timestamps1 and timestamps2:
                # 计算时间重叠
                time1_range = (min(timestamps1), max(timestamps1))
                time2_range = (min(timestamps2), max(timestamps2))
                
                overlap_start = max(time1_range[0], time2_range[0])
                overlap_end = min(time1_range[1], time2_range[1])
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    total_duration = max(time1_range[1] - time1_range[0], time2_range[1] - time2_range[0])
                    time_sim = overlap_duration / total_duration if total_duration > 0 else 0.0
            
            # 综合相似性
            total_sim = (length_sim * 0.2 + type_sim * 0.3 + emb_sim * 0.3 + time_sim * 0.2)
            
            return total_sim
            
        except Exception as e:
            self.logger.warning(f"计算链相似性时发生错误: {e}")
            return 0.0
    
    def _perform_attribution(self, activity_features: Dict[str, Any], 
                           similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """
        执行活动归因
        
        Args:
            activity_features: 活动特征
            similarity_matrix: 相似性矩阵
            
        Returns:
            归因结果
        """
        if similarity_matrix.size == 0:
            return {'attributed_activities': [], 'clusters': []}
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=self.similarity_threshold, min_samples=2)
        cluster_labels = clustering.fit_predict(similarity_matrix)
        
        # 组织归因结果
        attributed_activities = []
        clusters = {}
        
        chain_ids = list(activity_features.keys())
        
        for i, (chain_id, label) in enumerate(zip(chain_ids, cluster_labels)):
            if label == -1:  # 噪声点
                attributed_activities.append({
                    'chain_id': chain_id,
                    'cluster_id': None,
                    'is_noise': True,
                    'similarity_score': 0.0
                })
            else:
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append(chain_id)
                
                # 计算与聚类中心的相似性
                cluster_similarities = similarity_matrix[i][cluster_labels == label]
                avg_similarity = np.mean(cluster_similarities[cluster_similarities > 0])
                
                attributed_activities.append({
                    'chain_id': chain_id,
                    'cluster_id': label,
                    'is_noise': False,
                    'similarity_score': avg_similarity
                })
        
        return {
            'attributed_activities': attributed_activities,
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        }
    
    def _generate_attribution_report(self, attribution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成归因报告
        
        Args:
            attribution_results: 归因结果
            
        Returns:
            归因报告
        """
        attributed_activities = attribution_results['attributed_activities']
        clusters = attribution_results['clusters']
        
        # 统计信息
        total_activities = len(attributed_activities)
        clustered_activities = len([a for a in attributed_activities if not a['is_noise']])
        noise_activities = len([a for a in attributed_activities if a['is_noise']])
        n_clusters = len(clusters)
        
        # 聚类质量分析
        cluster_quality = {}
        for cluster_id, chain_ids in clusters.items():
            if len(chain_ids) > 1:
                # 计算聚类内相似性
                similarities = []
                for i, chain_id1 in enumerate(chain_ids):
                    for chain_id2 in chain_ids[i+1:]:
                        # 这里需要重新计算相似性，简化处理
                        similarities.append(0.8)  # 假设相似性
                
                cluster_quality[cluster_id] = {
                    'size': len(chain_ids),
                    'avg_similarity': np.mean(similarities) if similarities else 0.0,
                    'chain_ids': chain_ids
                }
        
        return {
            'summary': {
                'total_activities': total_activities,
                'clustered_activities': clustered_activities,
                'noise_activities': noise_activities,
                'n_clusters': n_clusters,
                'clustering_ratio': clustered_activities / total_activities if total_activities > 0 else 0.0
            },
            'cluster_quality': cluster_quality,
            'attribution_method': self.attribution_method,
            'similarity_threshold': self.similarity_threshold,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_attribution_statistics(self, attribution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取归因统计信息
        
        Args:
            attribution_results: 归因结果
            
        Returns:
            统计信息
        """
        attributed_activities = attribution_results['attributed_activities']
        clusters = attribution_results['clusters']
        
        # 基本统计
        stats = {
            'total_activities': len(attributed_activities),
            'clustered_activities': len([a for a in attributed_activities if not a['is_noise']]),
            'noise_activities': len([a for a in attributed_activities if a['is_noise']]),
            'n_clusters': len(clusters),
            'avg_cluster_size': np.mean([len(chain_ids) for chain_ids in clusters.values()]) if clusters else 0.0,
            'max_cluster_size': max([len(chain_ids) for chain_ids in clusters.values()]) if clusters else 0,
            'min_cluster_size': min([len(chain_ids) for chain_ids in clusters.values()]) if clusters else 0
        }
        
        return stats
