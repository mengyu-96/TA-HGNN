"""
路径重构器

实现攻击路径的重构和优化，确保路径的完整性和逻辑性
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
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class PathReconstructor:
    """
    路径重构器
    
    负责：
    1. 重构不完整的攻击路径
    2. 优化路径的时序逻辑
    3. 填补路径中的缺失环节
    4. 验证路径的合理性
    """
    
    def __init__(self, config):
        """
        初始化路径重构器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 重构参数
        self.max_gap_fill = getattr(config, 'max_gap_fill', 3)
        self.temporal_tolerance = getattr(config, 'temporal_tolerance', 300)  # 5分钟
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.6)
        
        # 攻击模式知识库
        self.attack_patterns = self._load_attack_patterns()
        
    def _load_attack_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载攻击模式知识库"""
        return {
            'initial_access': [
                {'pattern': 'phishing -> email_attachment', 'confidence': 0.9},
                {'pattern': 'spear_phishing -> malicious_link', 'confidence': 0.8},
                {'pattern': 'drive_by_download -> exploit_kit', 'confidence': 0.7},
                {'pattern': 'watering_hole -> malicious_website', 'confidence': 0.8}
            ],
            'execution': [
                {'pattern': 'command_execution -> powershell', 'confidence': 0.9},
                {'pattern': 'script_execution -> javascript', 'confidence': 0.8},
                {'pattern': 'binary_execution -> executable', 'confidence': 0.9},
                {'pattern': 'wmi_execution -> wmi_command', 'confidence': 0.7}
            ],
            'persistence': [
                {'pattern': 'registry_modification -> startup_key', 'confidence': 0.9},
                {'pattern': 'scheduled_task -> task_creation', 'confidence': 0.8},
                {'pattern': 'service_installation -> service_creation', 'confidence': 0.9},
                {'pattern': 'dll_hijacking -> dll_replacement', 'confidence': 0.7}
            ],
            'privilege_escalation': [
                {'pattern': 'local_exploit -> privilege_escalation', 'confidence': 0.8},
                {'pattern': 'token_manipulation -> token_stealing', 'confidence': 0.7},
                {'pattern': 'bypass_uac -> uac_bypass', 'confidence': 0.8}
            ],
            'lateral_movement': [
                {'pattern': 'remote_service_creation -> psexec', 'confidence': 0.9},
                {'pattern': 'wmi_execution -> lateral_wmi', 'confidence': 0.8},
                {'pattern': 'rdp_connection -> lateral_rdp', 'confidence': 0.9},
                {'pattern': 'smb_execution -> lateral_smb', 'confidence': 0.8}
            ],
            'exfiltration': [
                {'pattern': 'data_compression -> zip_creation', 'confidence': 0.8},
                {'pattern': 'data_encryption -> encryption', 'confidence': 0.7},
                {'pattern': 'network_transfer -> data_exfiltration', 'confidence': 0.9}
            ]
        }
    
    def reconstruct_path(self, partial_path: Dict[str, Any], 
                        hetero_data: HeteroData) -> Dict[str, Any]:
        """
        重构攻击路径
        
        Args:
            partial_path: 部分攻击路径
            hetero_data: 异构图数据
            
        Returns:
            重构后的完整攻击路径
        """
        self.logger.info("开始重构攻击路径")
        
        try:
            # 1. 分析路径中的缺失环节
            gaps = self._identify_gaps(partial_path)
            
            # 2. 填补缺失环节
            filled_path = self._fill_gaps(partial_path, gaps, hetero_data)
            
            # 3. 优化时序逻辑
            optimized_path = self._optimize_temporal_logic(filled_path)
            
            # 4. 验证路径合理性
            validated_path = self._validate_path(optimized_path, hetero_data)
            
            # 5. 生成重构报告
            reconstruction_report = self._generate_reconstruction_report(
                partial_path, validated_path
            )
            
            self.logger.info(f"路径重构完成，填补了 {len(gaps)} 个缺失环节")
            
            return {
                'original_path': partial_path,
                'reconstructed_path': validated_path,
                'gaps_filled': gaps,
                'reconstruction_report': reconstruction_report,
                'confidence': reconstruction_report.get('confidence', 0.0),
                'reconstruction_method': 'pattern_based'
            }
            
        except Exception as e:
            self.logger.error(f"路径重构失败: {e}")
            return {
                'original_path': partial_path,
                'reconstructed_path': partial_path,
                'gaps_filled': [],
                'reconstruction_report': {'confidence': 0.0, 'errors': [str(e)]},
                'confidence': 0.0,
                'reconstruction_method': 'failed'
            }
    
    def _identify_gaps(self, path: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        识别路径中的缺失环节
        
        Args:
            path: 攻击路径
            
        Returns:
            缺失环节列表
        """
        gaps = []
        
        if not path or 'path' not in path:
            return gaps
        
        path_nodes = path['path']
        path_types = path.get('path_types', [])
        attack_stages = path.get('attack_stages', [])
        
        # 检查相邻节点之间的逻辑跳跃
        for i in range(len(path_nodes) - 1):
            current_stage = attack_stages[i] if i < len(attack_stages) else 'unknown'
            next_stage = attack_stages[i + 1] if i + 1 < len(attack_stages) else 'unknown'
            
            # 检查是否存在逻辑跳跃
            if self._is_logical_jump(current_stage, next_stage):
                gap = {
                    'position': i + 1,
                    'from_stage': current_stage,
                    'to_stage': next_stage,
                    'gap_type': 'logical_jump',
                    'confidence': 0.8
                }
                gaps.append(gap)
        
        # 检查时序跳跃
        timestamps = path.get('timestamps', [])
        for i in range(len(timestamps) - 1):
            if i + 1 < len(timestamps):
                time_diff = abs((timestamps[i + 1] - timestamps[i]).total_seconds())
                if time_diff > self.temporal_tolerance:
                    gap = {
                        'position': i + 1,
                        'time_gap': time_diff,
                        'gap_type': 'temporal_jump',
                        'confidence': 0.7
                    }
                    gaps.append(gap)
        
        return gaps
    
    def _is_logical_jump(self, from_stage: str, to_stage: str) -> bool:
        """
        检查是否存在逻辑跳跃
        
        Args:
            from_stage: 起始阶段
            to_stage: 目标阶段
            
        Returns:
            是否存在逻辑跳跃
        """
        # 定义合理的阶段转换
        valid_transitions = {
            'initial_access': ['execution', 'persistence'],
            'execution': ['persistence', 'privilege_escalation', 'defense_evasion'],
            'persistence': ['execution', 'privilege_escalation'],
            'privilege_escalation': ['defense_evasion', 'credential_access', 'discovery'],
            'defense_evasion': ['credential_access', 'discovery', 'lateral_movement'],
            'credential_access': ['discovery', 'lateral_movement'],
            'discovery': ['lateral_movement', 'collection'],
            'lateral_movement': ['collection', 'command_and_control'],
            'collection': ['command_and_control', 'exfiltration'],
            'command_and_control': ['exfiltration', 'impact'],
            'exfiltration': ['impact'],
            'impact': []
        }
        
        if from_stage in valid_transitions:
            return to_stage not in valid_transitions[from_stage]
        
        return True  # 未知阶段，认为存在跳跃
    
    def _fill_gaps(self, path: Dict[str, Any], gaps: List[Dict[str, Any]], 
                   hetero_data: HeteroData) -> Dict[str, Any]:
        """
        填补路径中的缺失环节
        
        Args:
            path: 原始路径
            gaps: 缺失环节列表
            hetero_data: 异构图数据
            
        Returns:
            填补后的路径
        """
        filled_path = path.copy()
        
        for gap in gaps:
            if gap['gap_type'] == 'logical_jump':
                # 填补逻辑跳跃
                intermediate_stages = self._find_intermediate_stages(
                    gap['from_stage'], gap['to_stage']
                )
                
                # 在路径中插入中间阶段
                filled_path = self._insert_intermediate_stages(
                    filled_path, gap['position'], intermediate_stages
                )
            
            elif gap['gap_type'] == 'temporal_jump':
                # 填补时序跳跃
                filled_path = self._fill_temporal_gap(
                    filled_path, gap['position'], gap['time_gap']
                )
        
        return filled_path
    
    def _find_intermediate_stages(self, from_stage: str, to_stage: str) -> List[str]:
        """
        查找中间阶段
        
        Args:
            from_stage: 起始阶段
            to_stage: 目标阶段
            
        Returns:
            中间阶段列表
        """
        # 使用攻击模式知识库查找中间阶段
        intermediate_stages = []
        
        for pattern_info in self.attack_patterns.get(from_stage, []):
            pattern = pattern_info['pattern']
            if to_stage in pattern:
                # 提取中间阶段
                parts = pattern.split(' -> ')
                if len(parts) > 1:
                    intermediate_stages.extend(parts[1:-1])
        
        # 去重并返回
        return list(set(intermediate_stages))
    
    def _insert_intermediate_stages(self, path: Dict[str, Any], 
                                   position: int, 
                                   intermediate_stages: List[str]) -> Dict[str, Any]:
        """
        在路径中插入中间阶段
        
        Args:
            path: 原始路径
            position: 插入位置
            intermediate_stages: 中间阶段列表
            
        Returns:
            插入后的路径
        """
        filled_path = path.copy()
        
        # 为每个中间阶段创建虚拟节点
        for i, stage in enumerate(intermediate_stages):
            virtual_node_id = f"virtual_{stage}_{position}_{i}"
            virtual_node_type = self._get_node_type_for_stage(stage)
            
            # 插入到路径中
            filled_path['path'].insert(position + i, virtual_node_id)
            filled_path['path_types'].insert(position + i, virtual_node_type)
            filled_path['attack_stages'].insert(position + i, stage)
            filled_path['confidence_scores'].insert(position + i, 0.5)  # 虚拟节点置信度
            
            # 插入时间戳（插值）
            if 'timestamps' in filled_path and len(filled_path['timestamps']) > position:
                prev_time = filled_path['timestamps'][position - 1] if position > 0 else datetime.now()
                next_time = filled_path['timestamps'][position] if position < len(filled_path['timestamps']) else datetime.now()
                
                # 线性插值
                time_diff = (next_time - prev_time).total_seconds()
                step_time = time_diff / (len(intermediate_stages) + 1)
                new_time = prev_time + timedelta(seconds=step_time * (i + 1))
                
                filled_path['timestamps'].insert(position + i, new_time)
        
        return filled_path
    
    def _get_node_type_for_stage(self, stage: str) -> str:
        """
        根据攻击阶段获取节点类型
        
        Args:
            stage: 攻击阶段
            
        Returns:
            节点类型
        """
        stage_to_type = {
            'initial_access': 'email',
            'execution': 'command',
            'persistence': 'registry',
            'privilege_escalation': 'process',
            'defense_evasion': 'file',
            'credential_access': 'user',
            'discovery': 'network',
            'lateral_movement': 'connection',
            'collection': 'file',
            'command_and_control': 'network',
            'exfiltration': 'network',
            'impact': 'system'
        }
        
        return stage_to_type.get(stage, 'unknown')
    
    def _fill_temporal_gap(self, path: Dict[str, Any], position: int, 
                          time_gap: float) -> Dict[str, Any]:
        """
        填补时序跳跃
        
        Args:
            path: 原始路径
            position: 位置
            time_gap: 时间间隔
            
        Returns:
            填补后的路径
        """
        filled_path = path.copy()
        
        if 'timestamps' not in filled_path or len(filled_path['timestamps']) <= position:
            return filled_path
        
        # 在时间间隔中插入中间时间点
        prev_time = filled_path['timestamps'][position - 1] if position > 0 else datetime.now()
        next_time = filled_path['timestamps'][position] if position < len(filled_path['timestamps']) else datetime.now()
        
        # 计算需要插入的时间点数量
        num_interpolations = min(int(time_gap / 60), self.max_gap_fill)  # 最多插入max_gap_fill个点
        
        for i in range(num_interpolations):
            # 线性插值
            ratio = (i + 1) / (num_interpolations + 1)
            new_time = prev_time + timedelta(seconds=time_gap * ratio)
            
            # 插入虚拟节点
            virtual_node_id = f"temporal_fill_{position}_{i}"
            virtual_node_type = 'temporal_fill'
            
            filled_path['path'].insert(position + i, virtual_node_id)
            filled_path['path_types'].insert(position + i, virtual_node_type)
            filled_path['attack_stages'].insert(position + i, 'temporal_fill')
            filled_path['confidence_scores'].insert(position + i, 0.3)
            filled_path['timestamps'].insert(position + i, new_time)
        
        return filled_path
    
    def _optimize_temporal_logic(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化时序逻辑
        
        Args:
            path: 原始路径
            
        Returns:
            优化后的路径
        """
        optimized_path = path.copy()
        
        # 确保时间戳按顺序排列
        if 'timestamps' in optimized_path and len(optimized_path['timestamps']) > 1:
            # 按时间戳排序所有相关列表
            sorted_indices = sorted(range(len(optimized_path['timestamps'])), 
                                  key=lambda i: optimized_path['timestamps'][i])
            
            # 重新排列所有列表
            optimized_path['path'] = [optimized_path['path'][i] for i in sorted_indices]
            optimized_path['path_types'] = [optimized_path['path_types'][i] for i in sorted_indices]
            optimized_path['attack_stages'] = [optimized_path['attack_stages'][i] for i in sorted_indices]
            optimized_path['confidence_scores'] = [optimized_path['confidence_scores'][i] for i in sorted_indices]
            optimized_path['timestamps'] = [optimized_path['timestamps'][i] for i in sorted_indices]
        
        return optimized_path
    
    def _validate_path(self, path: Dict[str, Any], 
                      hetero_data: HeteroData) -> Dict[str, Any]:
        """
        验证路径的合理性
        
        Args:
            path: 攻击路径
            hetero_data: 异构图数据
            
        Returns:
            验证后的路径
        """
        validated_path = path.copy()
        
        # 添加验证信息
        validation_info = {
            'is_valid': True,
            'validation_errors': [],
            'confidence_score': 0.0,
            'completeness_score': 0.0
        }
        
        # 检查路径完整性
        if not path.get('path') or len(path['path']) < 2:
            validation_info['is_valid'] = False
            validation_info['validation_errors'].append('路径长度不足')
        
        # 检查时序逻辑
        if 'timestamps' in path and len(path['timestamps']) > 1:
            for i in range(len(path['timestamps']) - 1):
                if path['timestamps'][i] > path['timestamps'][i + 1]:
                    validation_info['is_valid'] = False
                    validation_info['validation_errors'].append(f'时序逻辑错误：位置 {i}')
        
        # 检查攻击阶段逻辑
        if 'attack_stages' in path:
            for i in range(len(path['attack_stages']) - 1):
                if self._is_logical_jump(path['attack_stages'][i], path['attack_stages'][i + 1]):
                    validation_info['validation_errors'].append(f'攻击阶段逻辑跳跃：{path["attack_stages"][i]} -> {path["attack_stages"][i + 1]}')
        
        # 计算置信度分数
        if 'confidence_scores' in path and path['confidence_scores']:
            validation_info['confidence_score'] = np.mean(path['confidence_scores'])
        
        # 计算完整性分数
        validation_info['completeness_score'] = self._calculate_completeness_score(path)
        
        validated_path['validation_info'] = validation_info
        
        return validated_path
    
    def _calculate_completeness_score(self, path: Dict[str, Any]) -> float:
        """
        计算路径完整性分数
        
        Args:
            path: 攻击路径
            
        Returns:
            完整性分数
        """
        if not path.get('path'):
            return 0.0
        
        # 检查关键攻击阶段是否存在
        required_stages = ['initial_access', 'execution', 'persistence']
        present_stages = set(path.get('attack_stages', []))
        
        stage_score = len(present_stages.intersection(set(required_stages))) / len(required_stages)
        
        # 检查路径长度
        path_length = len(path['path'])
        length_score = min(path_length / 10.0, 1.0)  # 路径越长分数越高，最高1.0
        
        # 检查置信度
        confidence_scores = path.get('confidence_scores', [])
        confidence_score = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # 综合分数
        completeness_score = (stage_score * 0.4 + length_score * 0.3 + confidence_score * 0.3)
        
        return completeness_score
    
    def _generate_reconstruction_report(self, original_path: Dict[str, Any], 
                                      reconstructed_path: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成重构报告
        
        Args:
            original_path: 原始路径
            reconstructed_path: 重构后路径
            
        Returns:
            重构报告
        """
        original_length = len(original_path.get('path', []))
        reconstructed_length = len(reconstructed_path.get('path', []))
        
        gaps_filled = reconstructed_length - original_length
        
        validation_info = reconstructed_path.get('validation_info', {})
        
        return {
            'original_length': original_length,
            'reconstructed_length': reconstructed_length,
            'gaps_filled': gaps_filled,
            'is_valid': validation_info.get('is_valid', False),
            'confidence_score': validation_info.get('confidence_score', 0.0),
            'completeness_score': validation_info.get('completeness_score', 0.0),
            'validation_errors': validation_info.get('validation_errors', []),
            'improvement_ratio': reconstructed_length / original_length if original_length > 0 else 1.0
        }
