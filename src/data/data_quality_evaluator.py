import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict

class DataQualityEvaluator:
    """
    数据质量评估器，用于评估数据质量并提供改进建议
    
    功能：
    1. 完整性评估：检查数据缺失情况
    2. 准确性评估：检查数据值是否符合预期
    3. 一致性评估：检查数据是否存在矛盾
    4. 时效性评估：检查数据是否及时更新
    5. 唯一性评估：检查数据是否存在重复
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化数据质量评估器
        
        Args:
            config: 配置字典，包含评估规则和阈值
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'completeness': {
                'threshold': 0.8,  # 完整性阈值，低于此值视为不合格
                'required_fields': []  # 必填字段列表
            },
            'accuracy': {
                'threshold': 0.9,  # 准确性阈值，低于此值视为不合格
                'field_types': {},  # 字段类型映射
                'value_ranges': {}  # 字段值范围
            },
            'consistency': {
                'threshold': 0.95,  # 一致性阈值，低于此值视为不合格
                'field_dependencies': []  # 字段依赖关系
            },
            'timeliness': {
                'threshold': 0.9,  # 时效性阈值，低于此值视为不合格
                'max_delay': 86400  # 最大延迟时间（秒）
            },
            'uniqueness': {
                'threshold': 0.99,  # 唯一性阈值，低于此值视为不合格
                'unique_fields': []  # 唯一字段列表
            }
        }
    
    def evaluate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据质量
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            质量评估结果字典，包含各维度的评分和建议
        """
        self.logger.info(f"开始评估数据质量，数据集大小: {len(df)}")
        
        # 初始化评估结果
        result = {
            'overall_score': 0.0,
            'dimensions': {},
            'issues': [],
            'suggestions': []
        }
        
        # 评估完整性
        completeness_result = self._evaluate_completeness(df)
        result['dimensions']['completeness'] = completeness_result
        
        # 评估准确性
        accuracy_result = self._evaluate_accuracy(df)
        result['dimensions']['accuracy'] = accuracy_result
        
        # 评估一致性
        consistency_result = self._evaluate_consistency(df)
        result['dimensions']['consistency'] = consistency_result
        
        # 评估时效性
        timeliness_result = self._evaluate_timeliness(df)
        result['dimensions']['timeliness'] = timeliness_result
        
        # 评估唯一性
        uniqueness_result = self._evaluate_uniqueness(df)
        result['dimensions']['uniqueness'] = uniqueness_result
        
        # 计算总体评分
        dimension_scores = [
            completeness_result['score'],
            accuracy_result['score'],
            consistency_result['score'],
            timeliness_result['score'],
            uniqueness_result['score']
        ]
        result['overall_score'] = sum(dimension_scores) / len(dimension_scores)
        
        # 汇总问题和建议
        for dimension, dimension_result in result['dimensions'].items():
            result['issues'].extend(dimension_result.get('issues', []))
            result['suggestions'].extend(dimension_result.get('suggestions', []))
        
        self.logger.info(f"数据质量评估完成，总体评分: {result['overall_score']:.2f}")
        
        return result
    
    def _evaluate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据完整性
        
        检查数据缺失情况，特别是必填字段
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            完整性评估结果
        """
        self.logger.debug("评估数据完整性")
        
        # 初始化评估结果
        result = {
            'score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # 计算每个字段的缺失率
        missing_rates = df.isnull().mean().to_dict()
        
        # 计算总体缺失率
        overall_missing_rate = df.isnull().mean().mean()
        overall_completeness = 1 - overall_missing_rate
        
        # 检查必填字段
        required_fields = self.config['completeness']['required_fields']
        missing_required_fields = []
        
        for field in required_fields:
            if field in df.columns:
                missing_rate = missing_rates[field]
                if missing_rate > 0:
                    missing_required_fields.append((field, missing_rate))
            else:
                missing_required_fields.append((field, 1.0))
        
        # 计算完整性评分
        if required_fields:
            required_completeness = 1 - sum(rate for _, rate in missing_required_fields) / len(required_fields)
            completeness_score = (overall_completeness + required_completeness) / 2
        else:
            completeness_score = overall_completeness
        
        result['score'] = completeness_score
        
        # 添加问题和建议
        if completeness_score < self.config['completeness']['threshold']:
            result['issues'].append(f"数据完整性评分较低: {completeness_score:.2f}")
            result['suggestions'].append("提高数据采集质量，确保必填字段不缺失")
        
        for field, rate in missing_required_fields:
            result['issues'].append(f"必填字段 '{field}' 缺失率: {rate:.2%}")
            result['suggestions'].append(f"补充字段 '{field}' 的缺失值")
        
        # 添加高缺失率的非必填字段
        for field, rate in missing_rates.items():
            if rate > 0.5 and field not in [f for f, _ in missing_required_fields]:
                result['issues'].append(f"字段 '{field}' 缺失率高: {rate:.2%}")
                result['suggestions'].append(f"考虑补充字段 '{field}' 的缺失值或移除该字段")
        
        return result
    
    def _evaluate_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据准确性
        
        检查数据值是否符合预期类型和范围
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            准确性评估结果
        """
        self.logger.debug("评估数据准确性")
        
        # 初始化评估结果
        result = {
            'score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # 检查字段类型
        field_types = self.config['accuracy']['field_types']
        type_errors = []
        
        for field, expected_type in field_types.items():
            if field in df.columns:
                if expected_type == 'int':
                    invalid_count = (~df[field].apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()))).sum()
                elif expected_type == 'float':
                    invalid_count = (~df[field].apply(lambda x: isinstance(x, (int, float, np.integer, np.float)))).sum()
                elif expected_type == 'str':
                    invalid_count = (~df[field].apply(lambda x: isinstance(x, str))).sum()
                elif expected_type == 'bool':
                    invalid_count = (~df[field].apply(lambda x: isinstance(x, bool))).sum()
                elif expected_type == 'datetime':
                    invalid_count = (~pd.to_datetime(df[field], errors='coerce').notna()).sum()
                else:
                    invalid_count = 0
                
                if invalid_count > 0:
                    error_rate = invalid_count / len(df)
                    type_errors.append((field, expected_type, error_rate))
        
        # 检查字段值范围
        value_ranges = self.config['accuracy']['value_ranges']
        range_errors = []
        
        for field, value_range in value_ranges.items():
            if field in df.columns:
                min_val, max_val = value_range
                if pd.api.types.is_numeric_dtype(df[field]):
                    out_of_range = ((df[field] < min_val) | (df[field] > max_val)).sum()
                    if out_of_range > 0:
                        error_rate = out_of_range / len(df)
                        range_errors.append((field, min_val, max_val, error_rate))
        
        # 计算准确性评分
        if field_types or value_ranges:
            type_error_rate = sum(rate for _, _, rate in type_errors) / len(field_types) if field_types else 0
            range_error_rate = sum(rate for _, _, _, rate in range_errors) / len(value_ranges) if value_ranges else 0
            
            if field_types and value_ranges:
                accuracy_score = 1 - (type_error_rate + range_error_rate) / 2
            elif field_types:
                accuracy_score = 1 - type_error_rate
            else:
                accuracy_score = 1 - range_error_rate
        else:
            # 如果没有配置类型和范围检查，默认为满分
            accuracy_score = 1.0
        
        result['score'] = accuracy_score
        
        # 添加问题和建议
        if accuracy_score < self.config['accuracy']['threshold']:
            result['issues'].append(f"数据准确性评分较低: {accuracy_score:.2f}")
            result['suggestions'].append("改进数据验证流程，确保数据类型和值范围正确")
        
        for field, expected_type, rate in type_errors:
            result['issues'].append(f"字段 '{field}' 类型错误率: {rate:.2%}，期望类型: {expected_type}")
            result['suggestions'].append(f"修正字段 '{field}' 的数据类型")
        
        for field, min_val, max_val, rate in range_errors:
            result['issues'].append(f"字段 '{field}' 值超出范围的比例: {rate:.2%}，期望范围: [{min_val}, {max_val}]")
            result['suggestions'].append(f"检查并修正字段 '{field}' 的异常值")
        
        return result
    
    def _evaluate_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据一致性
        
        检查数据是否存在矛盾，如字段间的依赖关系
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            一致性评估结果
        """
        self.logger.debug("评估数据一致性")
        
        # 初始化评估结果
        result = {
            'score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # 检查字段依赖关系
        field_dependencies = self.config['consistency']['field_dependencies']
        dependency_errors = []
        
        for dependency in field_dependencies:
            if len(dependency) == 3:
                field1, field2, condition = dependency
                if field1 in df.columns and field2 in df.columns:
                    # 解析条件
                    if condition == 'equal':
                        inconsistent = (df[field1] != df[field2]).sum()
                    elif condition == 'less_than':
                        inconsistent = (df[field1] >= df[field2]).sum()
                    elif condition == 'greater_than':
                        inconsistent = (df[field1] <= df[field2]).sum()
                    elif condition == 'implies':
                        # field1为True时，field2也应为True
                        inconsistent = ((df[field1] == True) & (df[field2] != True)).sum()
                    else:
                        inconsistent = 0
                    
                    if inconsistent > 0:
                        error_rate = inconsistent / len(df)
                        dependency_errors.append((field1, field2, condition, error_rate))
        
        # 计算一致性评分
        if field_dependencies:
            consistency_score = 1 - sum(rate for _, _, _, rate in dependency_errors) / len(field_dependencies)
        else:
            # 如果没有配置依赖关系检查，默认为满分
            consistency_score = 1.0
        
        result['score'] = consistency_score
        
        # 添加问题和建议
        if consistency_score < self.config['consistency']['threshold']:
            result['issues'].append(f"数据一致性评分较低: {consistency_score:.2f}")
            result['suggestions'].append("改进数据验证流程，确保字段间的依赖关系正确")
        
        for field1, field2, condition, rate in dependency_errors:
            result['issues'].append(f"字段 '{field1}' 和 '{field2}' 的关系 '{condition}' 不一致，错误率: {rate:.2%}")
            result['suggestions'].append(f"检查并修正字段 '{field1}' 和 '{field2}' 之间的关系")
        
        return result
    
    def _evaluate_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据时效性
        
        检查数据是否及时更新
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            时效性评估结果
        """
        self.logger.debug("评估数据时效性")
        
        # 初始化评估结果
        result = {
            'score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # 检查时间戳字段
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'event_time']
        timestamp_field = None
        
        for field in timestamp_fields:
            if field in df.columns:
                timestamp_field = field
                break
        
        if timestamp_field:
            # 转换为datetime类型
            df[timestamp_field] = pd.to_datetime(df[timestamp_field], errors='coerce')
            
            # 计算当前时间
            current_time = pd.Timestamp.now()
            
            # 计算数据延迟
            delays = (current_time - df[timestamp_field]).dt.total_seconds()
            
            # 计算超过最大延迟的比例
            max_delay = self.config['timeliness']['max_delay']
            delayed_count = (delays > max_delay).sum()
            delayed_rate = delayed_count / len(df)
            
            # 计算时效性评分
            timeliness_score = 1 - delayed_rate
            
            # 添加问题和建议
            if timeliness_score < self.config['timeliness']['threshold']:
                result['issues'].append(f"数据时效性评分较低: {timeliness_score:.2f}")
                result['suggestions'].append("提高数据更新频率，减少数据延迟")
            
            if delayed_count > 0:
                result['issues'].append(f"有 {delayed_count} 条数据延迟超过 {max_delay} 秒，比例: {delayed_rate:.2%}")
                result['suggestions'].append("检查数据采集流程，确保数据及时更新")
        else:
            # 如果没有时间戳字段，默认为满分
            timeliness_score = 1.0
            result['issues'].append("未找到时间戳字段，无法评估数据时效性")
            result['suggestions'].append("添加时间戳字段，如 'timestamp'、'created_at' 等")
        
        result['score'] = timeliness_score
        
        return result
    
    def _evaluate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据唯一性
        
        检查数据是否存在重复
        
        Args:
            df: 待评估的DataFrame
            
        Returns:
            唯一性评估结果
        """
        self.logger.debug("评估数据唯一性")
        
        # 初始化评估结果
        result = {
            'score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # 检查唯一字段
        unique_fields = self.config['uniqueness']['unique_fields']
        duplicate_errors = []
        
        # 检查整体重复
        duplicate_count = len(df) - len(df.drop_duplicates())
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0
        
        if duplicate_count > 0:
            duplicate_errors.append(('all_fields', duplicate_rate))
        
        # 检查单个唯一字段的重复
        for field in unique_fields:
            if field in df.columns:
                field_duplicate_count = len(df) - len(df.drop_duplicates(subset=[field]))
                field_duplicate_rate = field_duplicate_count / len(df) if len(df) > 0 else 0
                
                if field_duplicate_count > 0:
                    duplicate_errors.append((field, field_duplicate_rate))
        
        # 计算唯一性评分
        if unique_fields:
            uniqueness_score = 1 - sum(rate for _, rate in duplicate_errors) / (len(unique_fields) + 1)
        else:
            uniqueness_score = 1 - duplicate_rate
        
        result['score'] = uniqueness_score
        
        # 添加问题和建议
        if uniqueness_score < self.config['uniqueness']['threshold']:
            result['issues'].append(f"数据唯一性评分较低: {uniqueness_score:.2f}")
            result['suggestions'].append("改进数据去重流程，确保数据唯一性")
        
        if duplicate_count > 0:
            result['issues'].append(f"存在 {duplicate_count} 条完全重复的数据，比例: {duplicate_rate:.2%}")
            result['suggestions'].append("移除重复数据")
        
        for field, rate in duplicate_errors:
            if field != 'all_fields':
                result['issues'].append(f"字段 '{field}' 存在重复值，重复率: {rate:.2%}")
                result['suggestions'].append(f"确保字段 '{field}' 的值唯一")
        
        return result
    
    def get_quality_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Args:
            df: 待评估的DataFrame
            output_path: 报告输出路径，如果为None则不输出到文件
            
        Returns:
            数据质量报告
        """
        # 评估数据质量
        quality_result = self.evaluate_quality(df)
        
        # 生成报告
        report = {
            'summary': {
                'total_records': len(df),
                'total_fields': len(df.columns),
                'overall_score': quality_result['overall_score'],
                'quality_level': self._get_quality_level(quality_result['overall_score']),
                'pass_threshold': quality_result['overall_score'] >= 0.8
            },
            'dimensions': quality_result['dimensions'],
            'field_statistics': self._get_field_statistics(df),
            'issues': quality_result['issues'],
            'suggestions': quality_result['suggestions']
        }
        
        # 输出到文件
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"数据质量报告已输出到 {output_path}")
        
        return report
    
    def _get_quality_level(self, score: float) -> str:
        """
        根据评分获取质量等级
        
        Args:
            score: 质量评分
            
        Returns:
            质量等级
        """
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "一般"
        elif score >= 0.6:
            return "较差"
        else:
            return "差"
    
    def _get_field_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        获取字段统计信息
        
        Args:
            df: DataFrame
            
        Returns:
            字段统计信息
        """
        field_stats = {}
        
        for column in df.columns:
            stats = {
                'missing_rate': df[column].isnull().mean(),
                'unique_count': df[column].nunique(),
                'unique_rate': df[column].nunique() / len(df) if len(df) > 0 else 0
            }
            
            # 数值型字段的统计
            if pd.api.types.is_numeric_dtype(df[column]):
                stats.update({
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'mean': df[column].mean(),
                    'median': df[column].median(),
                    'std': df[column].std()
                })
            
            # 字符串字段的统计
            elif pd.api.types.is_string_dtype(df[column]):
                stats.update({
                    'min_length': df[column].str.len().min(),
                    'max_length': df[column].str.len().max(),
                    'mean_length': df[column].str.len().mean()
                })
            
            field_stats[column] = stats
        
        return field_stats
    
    def set_config(self, config: Dict) -> None:
        """
        设置配置
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
    
    def get_config(self) -> Dict:
        """
        获取配置
        
        Returns:
            配置字典
        """
        return self.config