"""
模式分析器

实现基于T-HGNN的攻击模式分析功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import networkx as nx

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class PatternAnalyzer:
    """
    模式分析器
    
    实现基于T-HGNN的攻击模式分析功能
    """
    
    def __init__(self, config):
        """
        初始化模式分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分析参数
        self.min_pattern_length = getattr(config, 'min_pattern_length', 3)
        self.max_pattern_length = getattr(config, 'max_pattern_length', 10)
        self.pattern_support_threshold = getattr(config, 'pattern_support_threshold', 0.1)
        
        self.logger.info(f"模式分析器初始化完成，最小模式长度: {self.min_pattern_length}")
    
    def analyze_patterns(self, hetero_data: HeteroData, 
                        attack_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析攻击模式
        
        Args:
            hetero_data: 异构图数据
            attack_chains: 攻击链列表
            
        Returns:
            模式分析结果
        """
        self.logger.info("开始攻击模式分析")
        
        try:
            # 1. 提取攻击序列
            attack_sequences = self._extract_attack_sequences(attack_chains)
            
            # 2. 发现频繁模式
            frequent_patterns = self._discover_frequent_patterns(attack_sequences)
            
            # 3. 分析模式特征
            pattern_features = self._analyze_pattern_features(frequent_patterns, attack_sequences)
            
            # 4. 生成模式报告
            pattern_report = self._generate_pattern_report(frequent_patterns, pattern_features)
            
            self.logger.info(f"模式分析完成，发现 {len(frequent_patterns)} 个频繁模式")
            
            return {
                'frequent_patterns': frequent_patterns,
                'pattern_features': pattern_features,
                'pattern_report': pattern_report,
                'attack_sequences': attack_sequences
            }
            
        except Exception as e:
            self.logger.error(f"模式分析过程中发生错误: {e}")
            return {
                'frequent_patterns': [],
                'pattern_features': {},
                'pattern_report': {'error': str(e), 'status': 'failed'},
                'attack_sequences': []
            }
    
    def _extract_attack_sequences(self, attack_chains: List[Dict[str, Any]]) -> List[List[str]]:
        """
        提取攻击序列
        
        Args:
            attack_chains: 攻击链列表
            
        Returns:
            攻击序列列表
        """
        sequences = []
        
        for chain in attack_chains:
            path = chain.get('path', [])
            path_types = chain.get('path_types', [])
            
            # 创建序列：节点类型 -> 节点ID
            sequence = []
            for i, node_id in enumerate(path):
                node_type = path_types[i] if i < len(path_types) else 'unknown'
                sequence.append(f"{node_type}:{node_id}")
            
            if len(sequence) >= self.min_pattern_length:
                sequences.append(sequence)
        
        return sequences
    
    def _discover_frequent_patterns(self, sequences: List[List[str]]) -> List[Dict[str, Any]]:
        """
        发现频繁模式
        
        Args:
            sequences: 攻击序列列表
            
        Returns:
            频繁模式列表
        """
        if not sequences:
            return []
        
        # 使用简化的频繁模式挖掘算法
        pattern_counts = defaultdict(int)
        total_sequences = len(sequences)
        
        # 统计所有可能的子序列
        for sequence in sequences:
            for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    pattern = tuple(sequence[i:i + length])
                    pattern_counts[pattern] += 1
        
        # 过滤频繁模式
        frequent_patterns = []
        for pattern, count in pattern_counts.items():
            support = count / total_sequences
            if support >= self.pattern_support_threshold:
                frequent_patterns.append({
                    'pattern': list(pattern),
                    'support': support,
                    'count': count,
                    'length': len(pattern)
                })
        
        # 按支持度排序
        frequent_patterns.sort(key=lambda x: x['support'], reverse=True)
        
        return frequent_patterns
    
    def _analyze_pattern_features(self, patterns: List[Dict[str, Any]], 
                                sequences: List[List[str]]) -> Dict[str, Any]:
        """
        分析模式特征
        
        Args:
            patterns: 频繁模式列表
            sequences: 攻击序列列表
            
        Returns:
            模式特征字典
        """
        features = {}
        
        for i, pattern in enumerate(patterns):
            pattern_id = f"pattern_{i}"
            
            # 基本特征
            pattern_sequence = pattern['pattern']
            features[pattern_id] = {
                'pattern_id': pattern_id,
                'pattern_sequence': pattern_sequence,
                'support': pattern['support'],
                'count': pattern['count'],
                'length': pattern['length'],
                
                # 节点类型分布
                'node_type_distribution': self._analyze_node_type_distribution(pattern_sequence),
                
                # 序列特征
                'sequence_features': self._analyze_sequence_features(pattern_sequence),
                
                # 时间特征（如果有时间信息）
                'temporal_features': self._analyze_temporal_features(pattern_sequence, sequences),
                
                # 复杂度特征
                'complexity_features': self._analyze_complexity_features(pattern_sequence)
            }
        
        return features
    
    def _analyze_node_type_distribution(self, pattern_sequence: List[str]) -> Dict[str, Any]:
        """
        分析节点类型分布
        
        Args:
            pattern_sequence: 模式序列
            
        Returns:
            节点类型分布
        """
        # 提取节点类型
        node_types = []
        for item in pattern_sequence:
            if ':' in item:
                node_type = item.split(':')[0]
                node_types.append(node_type)
        
        # 统计分布
        type_counts = Counter(node_types)
        total_types = len(node_types)
        
        distribution = {}
        for node_type, count in type_counts.items():
            distribution[node_type] = {
                'count': count,
                'frequency': count / total_types if total_types > 0 else 0.0
            }
        
        return {
            'distribution': distribution,
            'unique_types': len(type_counts),
            'total_nodes': total_types,
            'most_common_type': type_counts.most_common(1)[0][0] if type_counts else None
        }
    
    def _analyze_sequence_features(self, pattern_sequence: List[str]) -> Dict[str, Any]:
        """
        分析序列特征
        
        Args:
            pattern_sequence: 模式序列
            
        Returns:
            序列特征
        """
        # 序列长度
        length = len(pattern_sequence)
        
        # 唯一节点数
        unique_nodes = len(set(pattern_sequence))
        
        # 重复节点数
        repeated_nodes = length - unique_nodes
        
        # 序列复杂度（基于节点类型变化）
        node_types = [item.split(':')[0] if ':' in item else item for item in pattern_sequence]
        type_changes = sum(1 for i in range(1, len(node_types)) if node_types[i] != node_types[i-1])
        
        # 序列多样性
        diversity = unique_nodes / length if length > 0 else 0.0
        
        return {
            'length': length,
            'unique_nodes': unique_nodes,
            'repeated_nodes': repeated_nodes,
            'type_changes': type_changes,
            'diversity': diversity,
            'repetition_ratio': repeated_nodes / length if length > 0 else 0.0
        }
    
    def _analyze_temporal_features(self, pattern_sequence: List[str], 
                                 sequences: List[List[str]]) -> Dict[str, Any]:
        """
        分析时间特征
        
        Args:
            pattern_sequence: 模式序列
            sequences: 所有攻击序列
            
        Returns:
            时间特征
        """
        # 简化实现：基于序列位置分析时间特征
        pattern_length = len(pattern_sequence)
        
        # 分析模式在序列中的位置分布
        positions = []
        for sequence in sequences:
            for i in range(len(sequence) - pattern_length + 1):
                if sequence[i:i + pattern_length] == pattern_sequence:
                    positions.append(i)
        
        if positions:
            return {
                'avg_position': np.mean(positions),
                'position_std': np.std(positions),
                'min_position': min(positions),
                'max_position': max(positions),
                'occurrence_count': len(positions)
            }
        else:
            return {
                'avg_position': 0.0,
                'position_std': 0.0,
                'min_position': 0,
                'max_position': 0,
                'occurrence_count': 0
            }
    
    def _analyze_complexity_features(self, pattern_sequence: List[str]) -> Dict[str, Any]:
        """
        分析复杂度特征
        
        Args:
            pattern_sequence: 模式序列
            
        Returns:
            复杂度特征
        """
        # 节点类型熵
        node_types = [item.split(':')[0] if ':' in item else item for item in pattern_sequence]
        type_counts = Counter(node_types)
        total_types = len(node_types)
        
        entropy = 0.0
        for count in type_counts.values():
            p = count / total_types
            if p > 0:
                entropy -= p * np.log2(p)
        
        # 序列规律性
        regularity = 1.0 - entropy / np.log2(len(type_counts)) if len(type_counts) > 1 else 1.0
        
        # 模式复杂度
        complexity = len(pattern_sequence) * entropy
        
        return {
            'entropy': entropy,
            'regularity': regularity,
            'complexity': complexity,
            'type_richness': len(type_counts),
            'type_evenness': entropy / np.log2(len(type_counts)) if len(type_counts) > 1 else 0.0
        }
    
    def _generate_pattern_report(self, patterns: List[Dict[str, Any]], 
                               features: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成模式报告
        
        Args:
            patterns: 频繁模式列表
            features: 模式特征
            
        Returns:
            模式报告
        """
        if not patterns:
            return {
                'summary': {'total_patterns': 0, 'status': 'no_patterns_found'},
                'generated_at': datetime.now().isoformat()
            }
        
        # 基本统计
        total_patterns = len(patterns)
        avg_support = np.mean([p['support'] for p in patterns])
        max_support = max([p['support'] for p in patterns])
        min_support = min([p['support'] for p in patterns])
        
        # 长度分布
        length_distribution = Counter([p['length'] for p in patterns])
        
        # 支持度分布
        support_ranges = {
            'high': len([p for p in patterns if p['support'] >= 0.5]),
            'medium': len([p for p in patterns if 0.2 <= p['support'] < 0.5]),
            'low': len([p for p in patterns if p['support'] < 0.2])
        }
        
        # 模式质量分析
        quality_analysis = self._analyze_pattern_quality(patterns, features)
        
        return {
            'summary': {
                'total_patterns': total_patterns,
                'avg_support': avg_support,
                'max_support': max_support,
                'min_support': min_support,
                'length_distribution': dict(length_distribution),
                'support_distribution': support_ranges
            },
            'quality_analysis': quality_analysis,
            'top_patterns': patterns[:10],  # 前10个模式
            'generated_at': datetime.now().isoformat()
        }
    
    def _analyze_pattern_quality(self, patterns: List[Dict[str, Any]], 
                               features: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析模式质量
        
        Args:
            patterns: 频繁模式列表
            features: 模式特征
            
        Returns:
            质量分析结果
        """
        if not patterns:
            return {}
        
        # 计算质量指标
        quality_metrics = []
        
        for pattern in patterns:
            pattern_id = f"pattern_{patterns.index(pattern)}"
            if pattern_id in features:
                feature = features[pattern_id]
                
                # 综合质量分数
                quality_score = (
                    pattern['support'] * 0.4 +  # 支持度权重
                    feature['sequence_features']['diversity'] * 0.3 +  # 多样性权重
                    feature['complexity_features']['regularity'] * 0.3  # 规律性权重
                )
                
                quality_metrics.append({
                    'pattern_id': pattern_id,
                    'quality_score': quality_score,
                    'support': pattern['support'],
                    'diversity': feature['sequence_features']['diversity'],
                    'regularity': feature['complexity_features']['regularity']
                })
        
        # 排序
        quality_metrics.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return {
            'quality_metrics': quality_metrics,
            'avg_quality_score': np.mean([m['quality_score'] for m in quality_metrics]),
            'high_quality_patterns': len([m for m in quality_metrics if m['quality_score'] >= 0.7]),
            'medium_quality_patterns': len([m for m in quality_metrics if 0.4 <= m['quality_score'] < 0.7]),
            'low_quality_patterns': len([m for m in quality_metrics if m['quality_score'] < 0.4])
        }
    
    def get_pattern_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取模式统计信息
        
        Args:
            patterns: 频繁模式列表
            
        Returns:
            统计信息
        """
        if not patterns:
            return {'total_patterns': 0}
        
        return {
            'total_patterns': len(patterns),
            'avg_support': np.mean([p['support'] for p in patterns]),
            'max_support': max([p['support'] for p in patterns]),
            'min_support': min([p['support'] for p in patterns]),
            'avg_length': np.mean([p['length'] for p in patterns]),
            'max_length': max([p['length'] for p in patterns]),
            'min_length': min([p['length'] for p in patterns])
        }
