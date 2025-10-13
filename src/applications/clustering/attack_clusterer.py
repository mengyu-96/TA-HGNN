"""
攻击聚类器

实现基于T-HGNN的攻击聚类功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class AttackClusterer:
    """
    攻击聚类器
    
    实现基于T-HGNN的攻击聚类功能
    """
    
    def __init__(self, config):
        """
        初始化攻击聚类器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 聚类参数
        self.clustering_method = getattr(config, 'clustering_method', 'dbscan')
        self.min_cluster_size = getattr(config, 'min_cluster_size', 3)
        self.eps = getattr(config, 'eps', 0.5)
        self.n_clusters = getattr(config, 'n_clusters', 5)
        
        # 聚类模型
        self.clustering_models = self._initialize_clustering_models()
        self.scaler = StandardScaler()
        
        # 聚类历史
        self.clustering_history = []
        
    def _initialize_clustering_models(self) -> Dict[str, Any]:
        """初始化聚类模型"""
        models = {
            'dbscan': DBSCAN(eps=self.eps, min_samples=self.min_cluster_size),
            'kmeans': KMeans(n_clusters=self.n_clusters, random_state=42),
            'agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters)
        }
        return models
    
    def cluster_attacks(self, hetero_data: HeteroData, 
                       embeddings: Dict[str, torch.Tensor],
                       attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚类攻击活动
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            attack_chains: 攻击链列表
            
        Returns:
            聚类结果
        """
        self.logger.info("开始攻击聚类")
        
        # 1. 提取特征
        features = self._extract_clustering_features(hetero_data, embeddings, attack_chains)
        
        # 2. 执行聚类
        cluster_labels = self._perform_clustering(features)
        
        # 3. 分析聚类结果
        cluster_analysis = self._analyze_clusters(features, cluster_labels, attack_chains)
        
        # 4. 生成聚类报告
        clustering_report = self._generate_clustering_report(cluster_analysis)
        
        # 5. 更新聚类历史
        self._update_clustering_history(clustering_report)
        
        self.logger.info(f"攻击聚类完成，发现 {len(set(cluster_labels))} 个聚类")
        
        return {
            'features': features,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'clustering_report': clustering_report
        }
    
    def _extract_clustering_features(self, hetero_data: HeteroData, 
                                   embeddings: Dict[str, torch.Tensor], 
                                   attack_chains: List[Dict[str, Any]]) -> np.ndarray:
        """
        提取聚类特征
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            attack_chains: 攻击链列表
            
        Returns:
            特征矩阵
        """
        features = []
        
        for chain in attack_chains:
            # 提取攻击链特征
            chain_features = self._extract_chain_features(chain, embeddings)
            features.append(chain_features)
        
        if features:
            features = np.vstack(features)
            # 标准化特征
            features = self.scaler.fit_transform(features)
        else:
            features = np.array([]).reshape(0, 10)  # 默认特征维度
        
        return features
    
    def _extract_chain_features(self, chain: Dict[str, Any], 
                               embeddings: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        提取单个攻击链的特征
        
        Args:
            chain: 攻击链信息
            embeddings: 节点嵌入
            
        Returns:
            特征向量
        """
        features = []
        
        # 1. 置信度特征
        confidence = chain.get('confidence', 0.0)
        features.append(confidence)
        
        # 2. 完整性特征
        completeness = chain.get('completeness', 0.0)
        features.append(completeness)
        
        # 3. 风险分数特征
        risk_score = chain.get('risk_score', 0.0)
        features.append(risk_score)
        
        # 4. 攻击阶段数量
        stages = chain.get('stages', [])
        stage_count = len(stages)
        features.append(stage_count)
        
        # 5. 节点数量特征
        nodes = chain.get('nodes', {})
        total_nodes = sum(len(node_info['indices']) for node_info in nodes.values())
        features.append(total_nodes)
        
        # 6. 时间跨度特征
        timestamp = chain.get('timestamp', datetime.now().isoformat())
        # 完整的时间跨度特征实现
        timestamp = chain.get('timestamp', datetime.now().isoformat())
        
        # 解析时间戳
        try:
            if isinstance(timestamp, str):
                # 尝试解析ISO格式时间戳
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
            
            # 计算时间特征
            current_time = datetime.now()
            time_diff = (current_time - dt).total_seconds()
            
            # 时间特征：小时、星期几、是否工作日
            hour_feature = dt.hour / 24.0
            weekday_feature = dt.weekday() / 7.0
            is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
            
            # 时间异常性：深夜或凌晨活动
            night_activity = 1.0 if dt.hour < 6 or dt.hour > 22 else 0.0
            
            # 时间跨度特征
            time_span_feature = min(time_diff / (24 * 3600), 30) / 30.0  # 标准化到30天
            
            features.extend([hour_feature, weekday_feature, is_weekend, night_activity, time_span_feature])
            
        except Exception as e:
            self.logger.warning(f"解析时间戳失败: {e}")
            # 使用默认值
            features.extend([0.5, 0.5, 0.0, 0.0, 0.5])
        
        # 7. 攻击模式特征
        pattern_name = chain.get('pattern_name', 'unknown')
        pattern_feature = hash(pattern_name) % 100 / 100.0
        features.append(pattern_feature)
        
        # 8. 节点类型多样性
        node_types = len(nodes)
        features.append(node_types)
        
        # 9. 平均可疑分数
        if nodes:
            all_scores = []
            for node_info in nodes.values():
                all_scores.extend(node_info.get('scores', [0.0]))
            avg_score = np.mean(all_scores) if all_scores else 0.0
        else:
            avg_score = 0.0
        features.append(avg_score)
        
        # 10. 最大可疑分数
        if nodes:
            all_scores = []
            for node_info in nodes.values():
                all_scores.extend(node_info.get('scores', [0.0]))
            max_score = np.max(all_scores) if all_scores else 0.0
        else:
            max_score = 0.0
        features.append(max_score)
        
        return np.array(features)
    
    def _perform_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        执行聚类
        
        Args:
            features: 特征矩阵
            
        Returns:
            聚类标签
        """
        if len(features) == 0:
            return np.array([])
        
        # 选择聚类方法
        if self.clustering_method == 'dbscan':
            model = self.clustering_models['dbscan']
        elif self.clustering_method == 'kmeans':
            model = self.clustering_models['kmeans']
        elif self.clustering_method == 'agglomerative':
            model = self.clustering_models['agglomerative']
        else:
            model = self.clustering_models['dbscan']
        
        # 执行聚类
        try:
            cluster_labels = model.fit_predict(features)
        except Exception as e:
            self.logger.warning(f"聚类失败: {e}")
            cluster_labels = np.zeros(len(features))
        
        return cluster_labels
    
    def _analyze_clusters(self, features: np.ndarray, 
                         cluster_labels: np.ndarray, 
                         attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析聚类结果
        
        Args:
            features: 特征矩阵
            cluster_labels: 聚类标签
            attack_chains: 攻击链列表
            
        Returns:
            聚类分析结果
        """
        analysis = {
            'clusters': {},
            'cluster_statistics': {},
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0
        }
        
        if len(features) == 0:
            return analysis
        
        # 计算聚类质量指标
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 1:
            try:
                analysis['silhouette_score'] = silhouette_score(features, cluster_labels)
                analysis['calinski_harabasz_score'] = calinski_harabasz_score(features, cluster_labels)
            except Exception as e:
                self.logger.warning(f"计算聚类质量指标失败: {e}")
        
        # 分析每个聚类
        for cluster_id in unique_labels:
            if cluster_id == -1:  # 噪声点
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_chains = [attack_chains[i] for i in range(len(attack_chains)) if cluster_mask[i]]
            
            # 计算聚类统计信息
            cluster_stats = {
                'size': len(cluster_chains),
                'features_mean': np.mean(cluster_features, axis=0).tolist(),
                'features_std': np.std(cluster_features, axis=0).tolist(),
                'confidence_mean': np.mean([chain.get('confidence', 0.0) for chain in cluster_chains]),
                'risk_score_mean': np.mean([chain.get('risk_score', 0.0) for chain in cluster_chains]),
                'pattern_distribution': self._analyze_pattern_distribution(cluster_chains),
                'stage_distribution': self._analyze_stage_distribution(cluster_chains),
                'time_distribution': self._analyze_time_distribution(cluster_chains)
            }
            
            analysis['clusters'][f'cluster_{cluster_id}'] = {
                'cluster_id': cluster_id,
                'chains': cluster_chains,
                'statistics': cluster_stats
            }
        
        # 计算总体统计信息
        analysis['cluster_statistics'] = {
            'total_clusters': n_clusters,
            'total_chains': len(attack_chains),
            'noise_points': np.sum(cluster_labels == -1),
            'avg_cluster_size': np.mean([len(analysis['clusters'][f'cluster_{cid}']['chains']) 
                                       for cid in unique_labels if cid != -1]) if n_clusters > 0 else 0
        }
        
        return analysis
    
    def _analyze_pattern_distribution(self, cluster_chains: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析攻击模式分布"""
        pattern_dist = {}
        for chain in cluster_chains:
            pattern = chain.get('pattern_name', 'unknown')
            pattern_dist[pattern] = pattern_dist.get(pattern, 0) + 1
        return pattern_dist
    
    def _analyze_stage_distribution(self, cluster_chains: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析攻击阶段分布"""
        stage_dist = {}
        for chain in cluster_chains:
            stages = chain.get('stages', [])
            for stage in stages:
                stage_dist[stage] = stage_dist.get(stage, 0) + 1
        return stage_dist
    
    def _analyze_time_distribution(self, cluster_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间分布"""
        timestamps = [chain.get('timestamp', datetime.now().isoformat()) for chain in cluster_chains]
        
        # 完整的时间分布分析实现
        timestamps = [chain.get('timestamp', datetime.now().isoformat()) for chain in cluster_chains]
        
        if not timestamps:
            return {
                'count': 0,
                'earliest': None,
                'latest': None,
                'time_span': 0,
                'hourly_distribution': {},
                'daily_distribution': {},
                'time_patterns': []
            }
        
        # 解析时间戳
        parsed_timestamps = []
        for ts in timestamps:
            try:
                if isinstance(ts, str):
                    if 'T' in ts:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromisoformat(ts)
                else:
                    dt = ts
                parsed_timestamps.append(dt)
            except Exception as e:
                self.logger.warning(f"解析时间戳失败: {e}")
                continue
        
        if not parsed_timestamps:
            return {
                'count': len(timestamps),
                'earliest': min(timestamps) if timestamps else None,
                'latest': max(timestamps) if timestamps else None,
                'time_span': 0,
                'hourly_distribution': {},
                'daily_distribution': {},
                'time_patterns': []
            }
        
        # 计算时间统计
        earliest = min(parsed_timestamps)
        latest = max(parsed_timestamps)
        time_span = (latest - earliest).total_seconds() / 3600  # 小时
        
        # 小时分布
        hourly_dist = {}
        for dt in parsed_timestamps:
            hour = dt.hour
            hourly_dist[hour] = hourly_dist.get(hour, 0) + 1
        
        # 星期分布
        daily_dist = {}
        for dt in parsed_timestamps:
            weekday = dt.weekday()
            daily_dist[weekday] = daily_dist.get(weekday, 0) + 1
        
        # 时间模式分析
        time_patterns = []
        
        # 检查是否有深夜活动模式
        night_activities = sum(1 for dt in parsed_timestamps if dt.hour < 6 or dt.hour > 22)
        if night_activities > len(parsed_timestamps) * 0.3:
            time_patterns.append('night_activity')
        
        # 检查是否有工作日模式
        weekday_activities = sum(1 for dt in parsed_timestamps if dt.weekday() < 5)
        if weekday_activities > len(parsed_timestamps) * 0.7:
            time_patterns.append('weekday_activity')
        
        # 检查是否有集中时间模式
        if time_span < 24:  # 24小时内
            time_patterns.append('concentrated_time')
        
        return {
            'count': len(timestamps),
            'earliest': earliest.isoformat(),
            'latest': latest.isoformat(),
            'time_span': time_span,
            'hourly_distribution': hourly_dist,
            'daily_distribution': daily_dist,
            'time_patterns': time_patterns
        }
    
    def _generate_clustering_report(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成聚类报告
        
        Args:
            cluster_analysis: 聚类分析结果
            
        Returns:
            聚类报告
        """
        report = {
            'clustering_time': datetime.now().isoformat(),
            'summary': {
                'total_clusters': cluster_analysis['cluster_statistics'].get('total_clusters', 0),
                'total_chains': cluster_analysis['cluster_statistics'].get('total_chains', 0),
                'noise_points': cluster_analysis['cluster_statistics'].get('noise_points', 0),
                'avg_cluster_size': cluster_analysis['cluster_statistics'].get('avg_cluster_size', 0.0),
                'silhouette_score': cluster_analysis.get('silhouette_score', 0.0),
                'calinski_harabasz_score': cluster_analysis.get('calinski_harabasz_score', 0.0)
            },
            'clusters': cluster_analysis['clusters'],
            'cluster_statistics': cluster_analysis['cluster_statistics'],
            'quality_assessment': self._assess_clustering_quality(cluster_analysis)
        }
        
        return report
    
    def _assess_clustering_quality(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估聚类质量
        
        Args:
            cluster_analysis: 聚类分析结果
            
        Returns:
            质量评估结果
        """
        silhouette_score = cluster_analysis['silhouette_score']
        calinski_harabasz_score = cluster_analysis['calinski_harabasz_score']
        n_clusters = cluster_analysis['cluster_statistics'].get('total_clusters', 0)
        
        # 评估聚类质量
        if silhouette_score > 0.7:
            quality_level = 'excellent'
        elif silhouette_score > 0.5:
            quality_level = 'good'
        elif silhouette_score > 0.3:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        # 评估聚类数量
        if n_clusters == 0:
            cluster_count_assessment = 'no_clusters'
        elif n_clusters < 3:
            cluster_count_assessment = 'too_few'
        elif n_clusters > 10:
            cluster_count_assessment = 'too_many'
        else:
            cluster_count_assessment = 'appropriate'
        
        return {
            'quality_level': quality_level,
            'silhouette_score': silhouette_score,
            'calinski_harabasz_score': calinski_harabasz_score,
            'cluster_count_assessment': cluster_count_assessment,
            'n_clusters': n_clusters
        }
    
    def _generate_clustering_recommendations(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """
        生成聚类建议
        
        Args:
            cluster_analysis: 聚类分析结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        quality_assessment = cluster_analysis['quality_assessment']
        quality_level = quality_assessment['quality_level']
        cluster_count_assessment = quality_assessment['cluster_count_assessment']
        
        # 基于聚类质量的建议
        if quality_level == 'poor':
            recommendations.append("聚类质量较差，建议调整聚类参数或特征提取方法")
        elif quality_level == 'fair':
            recommendations.append("聚类质量一般，建议优化特征工程或尝试不同的聚类算法")
        elif quality_level == 'good':
            recommendations.append("聚类质量良好，可以用于进一步分析")
        elif quality_level == 'excellent':
            recommendations.append("聚类质量优秀，结果可信度高")
        
        # 基于聚类数量的建议
        if cluster_count_assessment == 'no_clusters':
            recommendations.append("未发现明显的聚类，可能需要降低聚类阈值")
        elif cluster_count_assessment == 'too_few':
            recommendations.append("聚类数量过少，建议降低聚类阈值或增加数据")
        elif cluster_count_assessment == 'too_many':
            recommendations.append("聚类数量过多，建议提高聚类阈值或合并相似聚类")
        
        # 基于噪声点的建议
        noise_points = cluster_analysis['cluster_statistics']['noise_points']
        total_chains = cluster_analysis['cluster_statistics']['total_chains']
        noise_ratio = noise_points / total_chains if total_chains > 0 else 0
        
        if noise_ratio > 0.3:
            recommendations.append("噪声点比例较高，建议检查数据质量或调整聚类参数")
        
        return recommendations
    
    def _update_clustering_history(self, clustering_report: Dict[str, Any]):
        """
        更新聚类历史
        
        Args:
            clustering_report: 聚类报告
        """
        self.clustering_history.append(clustering_report)
        
        # 保持历史记录在合理范围内
        if len(self.clustering_history) > 50:
            self.clustering_history = self.clustering_history[-50:]
    
    def get_clustering_statistics(self) -> Dict[str, Any]:
        """
        获取聚类统计信息
        
        Returns:
            统计信息
        """
        if not self.clustering_history:
            return {'total_clusterings': 0, 'avg_cluster_count': 0.0}
        
        total_clusterings = len(self.clustering_history)
        cluster_counts = [report['summary']['total_clusters'] for report in self.clustering_history]
        avg_cluster_count = np.mean(cluster_counts)
        
        # 统计聚类质量分布
        quality_distribution = {}
        for report in self.clustering_history:
            quality = report['quality_assessment']['quality_level']
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        return {
            'total_clusterings': total_clusterings,
            'avg_cluster_count': avg_cluster_count,
            'quality_distribution': quality_distribution,
            'recent_clusterings': self.clustering_history[-5:]  # 最近5次聚类
        }
    
    def optimize_clustering_parameters(self, features: np.ndarray) -> Dict[str, Any]:
        """
        优化聚类参数
        
        Args:
            features: 特征矩阵
            
        Returns:
            优化结果
        """
        if len(features) == 0:
            return {'best_method': 'dbscan', 'best_params': {}}
        
        best_score = -1
        best_method = 'dbscan'
        best_params = {}
        
        # 测试不同的聚类方法
        methods = ['dbscan', 'kmeans', 'agglomerative']
        
        for method in methods:
            try:
                if method == 'dbscan':
                    # 测试不同的eps值
                    eps_values = [0.3, 0.5, 0.7, 1.0]
                    for eps in eps_values:
                        model = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
                        labels = model.fit_predict(features)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        if n_clusters > 1:
                            score = silhouette_score(features, labels)
                            if score > best_score:
                                best_score = score
                                best_method = method
                                best_params = {'eps': eps, 'min_samples': self.min_cluster_size}
                
                elif method == 'kmeans':
                    # 测试不同的聚类数量
                    n_clusters_values = range(2, min(10, len(features)))
                    for n_clusters in n_clusters_values:
                        model = KMeans(n_clusters=n_clusters, random_state=42)
                        labels = model.fit_predict(features)
                        score = silhouette_score(features, labels)
                        
                        if score > best_score:
                            best_score = score
                            best_method = method
                            best_params = {'n_clusters': n_clusters}
                
                elif method == 'agglomerative':
                    # 测试不同的聚类数量
                    n_clusters_values = range(2, min(10, len(features)))
                    for n_clusters in n_clusters_values:
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                        labels = model.fit_predict(features)
                        score = silhouette_score(features, labels)
                        
                        if score > best_score:
                            best_score = score
                            best_method = method
                            best_params = {'n_clusters': n_clusters}
            
            except Exception as e:
                self.logger.warning(f"优化聚类参数失败 {method}: {e}")
                continue
        
        return {
            'best_method': best_method,
            'best_params': best_params,
            'best_score': best_score
        }
