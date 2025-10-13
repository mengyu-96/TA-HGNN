"""
改进的APT数据预处理器

解决数据质量问题，包括：
1. 标签分布严重不均衡
2. 时间序列处理失败
3. 数据增强效果有限
4. 特征工程不足
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import re
import json


class ImprovedAPTDataProcessor:
    """改进的APT数据预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据质量阈值
        self.min_samples_per_class = 10
        self.max_class_imbalance_ratio = 0.95
        self.min_temporal_span_hours = 1
        
        # 特征工程配置
        self.feature_config = {
            'use_mitre_features': True,
            'use_temporal_features': True,
            'use_network_features': True,
            'use_file_features': True,
            'use_command_features': True,
            'use_user_features': True
        }
        
        # 标签生成策略
        self.label_strategies = {
            'rule_level': {'threshold': 7, 'weight': 0.3},
            'mitre_tactics': {'weight': 0.4},
            'file_suspicious': {'weight': 0.2},
            'command_suspicious': {'weight': 0.1}
        }
    
    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理原始数据"""
        self.logger.info("开始改进的数据预处理")
        
        # 1. 数据质量修复
        df = self._fix_data_quality_issues(df)
        
        # 2. 时间序列处理
        df = self._process_temporal_features(df)
        
        # 3. 特征工程
        df = self._engineer_comprehensive_features(df)
        
        # 4. 标签生成
        df = self._generate_improved_labels(df)
        
        # 5. 数据平衡
        df = self._balance_dataset(df)
        
        self.logger.info(f"数据预处理完成，最终形状: {df.shape}")
        return df
    
    def _fix_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复数据质量问题"""
        self.logger.info("修复数据质量问题")
        
        initial_shape = df.shape
        
        # 1. 处理缺失值
        df = self._handle_missing_values_advanced(df)
        
        # 2. 数据清洗
        df = self._clean_data_advanced(df)
        
        # 3. 数据类型转换
        df = self._convert_data_types(df)
        
        # 4. 异常值处理
        df = self._handle_outliers(df)
        
        self.logger.info(f"数据质量修复完成: {initial_shape} -> {df.shape}")
        return df
    
    def _handle_missing_values_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级缺失值处理"""
        self.logger.info("处理缺失值")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # 字符串列：用空字符串填充
                df[col] = df[col].fillna('')
            elif df[col].dtype in ['int64', 'float64']:
                # 数值列：用中位数填充
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # 其他类型：用前向填充
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _clean_data_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级数据清洗"""
        self.logger.info("执行高级数据清洗")
        
        initial_rows = len(df)
        
        # 1. 移除完全重复的行
        df = df.drop_duplicates()
        
        # 2. 移除空行
        df = df.dropna(how='all')
        
        # 3. 移除异常行（所有列都相同）
        df = df[df.nunique(axis=1) > 1]
        
        # 4. 移除明显无效的行
        if '_source.rule.level' in df.columns:
            # 移除规则级别异常的行
            df = df[df['_source.rule.level'].astype(str).str.isdigit()]
            # 安全地转换类型，处理大数值
            try:
                df = df[pd.to_numeric(df['_source.rule.level'], errors='coerce').between(0, 15)]
            except (OverflowError, ValueError):
                # 如果转换失败，移除包含非数字值的行
                df = df[df['_source.rule.level'].astype(str).str.match(r'^\d+$')]
                df = df[pd.to_numeric(df['_source.rule.level'], errors='coerce').between(0, 15)]
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"移除了 {removed_rows} 个无效行")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        self.logger.info("转换数据类型")
        
        # 转换规则级别为数值
        if '_source.rule.level' in df.columns:
            df['_source.rule.level'] = pd.to_numeric(df['_source.rule.level'], errors='coerce').fillna(0)
        
        # 转换时间戳列
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        self.logger.info("处理异常值")
        
        # 对数值列进行异常值处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and df[col].nunique() > 10:  # 只处理有足够多样性的列
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值限制在边界内
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理时间序列特征"""
        self.logger.info("处理时间序列特征")
        
        # 1. 统一时间戳处理
        df = self._unify_timestamps(df)
        
        # 2. 提取时间特征
        df = self._extract_temporal_features(df)
        
        # 3. 创建时间序列索引
        df = self._create_temporal_index(df)
        
        return df
    
    def _unify_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一时间戳格式"""
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        
        # 创建统一的时间戳列
        df['unified_timestamp'] = None
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    # 尝试解析时间戳
                    parsed_timestamps = pd.to_datetime(df[col], errors='coerce')
                    valid_count = parsed_timestamps.notna().sum()
                    
                    if valid_count > 0:
                        # 如果当前列的有效时间戳更多，则使用它
                        if df['unified_timestamp'].notna().sum() < valid_count:
                            df['unified_timestamp'] = parsed_timestamps
                            self.logger.info(f"使用列 {col} 作为主时间戳，有效数量: {valid_count}")
                
                except Exception as e:
                    self.logger.warning(f"处理时间戳列 {col} 时出错: {e}")
        
        # 如果没有有效时间戳，创建基于索引的时间戳
        if df['unified_timestamp'].notna().sum() == 0:
            self.logger.warning("没有找到有效时间戳，创建基于索引的时间戳")
            base_time = pd.Timestamp('2020-01-01')
            df['unified_timestamp'] = base_time + pd.to_timedelta(df.index, unit='minutes')
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        if 'unified_timestamp' not in df.columns:
            return df
        
        timestamp_col = df['unified_timestamp']
        
        # 基础时间特征
        df['timestamp_hour'] = timestamp_col.dt.hour
        df['timestamp_day'] = timestamp_col.dt.day
        df['timestamp_weekday'] = timestamp_col.dt.weekday
        df['timestamp_month'] = timestamp_col.dt.month
        df['timestamp_year'] = timestamp_col.dt.year
        
        # 周期性特征
        df['is_weekend'] = (timestamp_col.dt.weekday >= 5).astype(int)
        df['is_night'] = ((timestamp_col.dt.hour >= 22) | (timestamp_col.dt.hour <= 6)).astype(int)
        df['is_business_hours'] = ((timestamp_col.dt.hour >= 9) & (timestamp_col.dt.hour <= 17)).astype(int)
        
        # 时间间隔特征
        if len(df) > 1:
            time_diffs = timestamp_col.diff().dt.total_seconds().fillna(0)
            df['time_since_last'] = time_diffs
            df['time_since_last_log'] = np.log1p(time_diffs)
        
        return df
    
    def _create_temporal_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间序列索引"""
        if 'unified_timestamp' not in df.columns:
            return df
        
        # 按时间排序
        df = df.sort_values('unified_timestamp').reset_index(drop=True)
        
        # 创建时间序列索引
        df['temporal_index'] = range(len(df))
        
        # 计算时间跨度
        if len(df) > 1:
            time_span = (df['unified_timestamp'].max() - df['unified_timestamp'].min()).total_seconds() / 3600
            df['temporal_span_hours'] = time_span
        else:
            df['temporal_span_hours'] = 0
        
        return df
    
    def _engineer_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """综合特征工程"""
        self.logger.info("执行综合特征工程")
        
        # 1. MITRE ATT&CK特征
        if self.feature_config['use_mitre_features']:
            df = self._engineer_mitre_features(df)
        
        # 2. 网络特征
        if self.feature_config['use_network_features']:
            df = self._engineer_network_features(df)
        
        # 3. 文件特征
        if self.feature_config['use_file_features']:
            df = self._engineer_file_features(df)
        
        # 4. 命令特征
        if self.feature_config['use_command_features']:
            df = self._engineer_command_features(df)
        
        # 5. 用户特征
        if self.feature_config['use_user_features']:
            df = self._engineer_user_features(df)
        
        # 6. 规则特征
        df = self._engineer_rule_features(df)
        
        return df
    
    def _engineer_mitre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """MITRE ATT&CK特征工程"""
        # 战术特征
        if '_source.rule.mitre_tactics' in df.columns:
            df['has_mitre_tactics'] = df['_source.rule.mitre_tactics'].notna().astype(int)
            df['mitre_tactics_count'] = df['_source.rule.mitre_tactics'].astype(str).str.count(r'TA\d{4}')
            max_count = df['mitre_tactics_count'].max()
            if max_count > 0:
                df['mitre_tactics_density'] = df['mitre_tactics_count'] / max_count
            else:
                df['mitre_tactics_density'] = 0
        
        # 技术特征
        if '_source.rule.mitre_techniques' in df.columns:
            df['has_mitre_techniques'] = df['_source.rule.mitre_techniques'].notna().astype(int)
            df['mitre_techniques_count'] = df['_source.rule.mitre_techniques'].astype(str).str.count(r'T\d{4}')
            max_count = df['mitre_techniques_count'].max()
            if max_count > 0:
                df['mitre_techniques_density'] = df['mitre_techniques_count'] / max_count
            else:
                df['mitre_techniques_density'] = 0
        
        # 组合特征
        df['mitre_complexity'] = (df.get('mitre_tactics_count', 0) + df.get('mitre_techniques_count', 0)) / 2
        df['is_mitre_related'] = (df.get('has_mitre_tactics', 0) | df.get('has_mitre_techniques', 0)).astype(int)
        
        return df
    
    def _engineer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """网络特征工程"""
        # IP特征
        if '_source.data.srcip' in df.columns:
            df['has_src_ip'] = df['_source.data.srcip'].notna().astype(int)
            df['src_ip_is_private'] = df['_source.data.srcip'].astype(str).str.match(r'^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)').astype(int)
        
        if '_source.data.dstip' in df.columns:
            df['has_dst_ip'] = df['_source.data.dstip'].notna().astype(int)
            df['dst_ip_is_private'] = df['_source.data.dstip'].astype(str).str.match(r'^(10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.)').astype(int)
        
        # 端口特征
        if '_source.data.dstport' in df.columns:
            df['has_dst_port'] = df['_source.data.dstport'].notna().astype(int)
            df['dst_port_numeric'] = pd.to_numeric(df['_source.data.dstport'], errors='coerce').fillna(0)
            df['is_common_port'] = df['dst_port_numeric'].isin([80, 443, 22, 21, 25, 53, 110, 143, 993, 995]).astype(int)
            df['is_high_port'] = (df['dst_port_numeric'] > 1024).astype(int)
        
        # 域名特征
        if '_source.data.url' in df.columns:
            df['has_url'] = df['_source.data.url'].notna().astype(int)
            df['url_length'] = df['_source.data.url'].astype(str).str.len()
            df['is_https'] = df['_source.data.url'].astype(str).str.startswith('https://').astype(int)
        
        return df
    
    def _engineer_file_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """文件特征工程"""
        if '_source.data.file' in df.columns:
            file_col = df['_source.data.file'].astype(str)
            
            # 路径特征
            df['file_path_length'] = file_col.str.len()
            df['file_path_depth'] = file_col.str.count('/')
            df['file_extension'] = file_col.str.extract(r'\.([^.]+)$')[0].fillna('')
            
            # 系统文件特征
            df['is_system_file'] = file_col.str.contains(r'(?i)(/bin/|/sbin/|/usr/bin/|/usr/sbin/)', regex=True).astype(int)
            df['is_temp_file'] = file_col.str.contains(r'(?i)(/tmp/|/var/tmp/|/temp/)', regex=True).astype(int)
            df['is_home_file'] = file_col.str.contains(r'(?i)(/home/|/Users/)', regex=True).astype(int)
            
            # 可执行文件特征
            df['is_executable'] = file_col.str.contains(r'(?i)(\.exe|\.bat|\.cmd|\.sh|\.py|\.pl)$', regex=True).astype(int)
            df['is_script'] = file_col.str.contains(r'(?i)(\.sh|\.py|\.pl|\.rb|\.js)$', regex=True).astype(int)
            
            # 敏感文件特征
            df['is_config_file'] = file_col.str.contains(r'(?i)(\.conf|\.config|\.ini|\.cfg)$', regex=True).astype(int)
            df['is_log_file'] = file_col.str.contains(r'(?i)(\.log|\.out|\.err)$', regex=True).astype(int)
        
        return df
    
    def _engineer_command_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """命令特征工程"""
        if '_source.data.command' in df.columns:
            cmd_col = df['_source.data.command'].astype(str)
            
            # 基础特征
            df['command_length'] = cmd_col.str.len()
            df['command_word_count'] = cmd_col.str.split().str.len()
            
            # 命令类型特征
            df['is_network_command'] = cmd_col.str.contains(r'(?i)(wget|curl|nc|netcat|ssh|telnet)', regex=True).astype(int)
            df['is_privilege_command'] = cmd_col.str.contains(r'(?i)(sudo|su|chmod|chown|chgrp)', regex=True).astype(int)
            df['is_encoding_command'] = cmd_col.str.contains(r'(?i)(base64|encode|decode|xxd)', regex=True).astype(int)
            df['is_compression_command'] = cmd_col.str.contains(r'(?i)(tar|zip|gzip|bzip2)', regex=True).astype(int)
            df['is_system_command'] = cmd_col.str.contains(r'(?i)(ps|top|kill|killall|pkill)', regex=True).astype(int)
            
            # 可疑命令特征
            df['is_suspicious_command'] = cmd_col.str.contains(r'(?i)(rm\s+-rf|dd\s+if=|mkfs|fdisk)', regex=True).astype(int)
            df['has_pipes'] = cmd_col.str.contains(r'\|').astype(int)
            df['has_redirection'] = cmd_col.str.contains(r'[><]').astype(int)
            df['has_background'] = cmd_col.str.contains(r'&').astype(int)
            
            # 命令复杂度
            df['command_complexity'] = (
                df['command_word_count'] * 0.3 +
                df['has_pipes'] * 0.2 +
                df['has_redirection'] * 0.2 +
                df['has_background'] * 0.1 +
                (df['command_length'] > 50).astype(int) * 0.2
            )
        
        return df
    
    def _engineer_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """用户特征工程"""
        # 源用户特征
        if '_source.data.srcuser' in df.columns:
            src_user = df['_source.data.srcuser'].astype(str)
            df['is_root_user'] = src_user.str.contains(r'(?i)(root|admin|administrator)', regex=True).astype(int)
            df['is_system_user'] = src_user.str.contains(r'(?i)(system|daemon|nobody|www-data)', regex=True).astype(int)
            df['user_name_length'] = src_user.str.len()
        
        # 目标用户特征
        if '_source.data.dstuser' in df.columns:
            dst_user = df['_source.data.dstuser'].astype(str)
            df['has_dst_user'] = dst_user.notna().astype(int)
            df['is_dst_root_user'] = dst_user.str.contains(r'(?i)(root|admin|administrator)', regex=True).astype(int)
        
        # 用户ID特征
        if '_source.data.uid' in df.columns:
            df['uid_numeric'] = pd.to_numeric(df['_source.data.uid'], errors='coerce').fillna(0)
            df['is_privileged_uid'] = (df['uid_numeric'] == 0).astype(int)  # root用户
            df['is_system_uid'] = ((df['uid_numeric'] > 0) & (df['uid_numeric'] < 1000)).astype(int)
        
        return df
    
    def _engineer_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """规则特征工程"""
        # 规则级别特征
        if '_source.rule.level' in df.columns:
            level = pd.to_numeric(df['_source.rule.level'], errors='coerce').fillna(0)
            df['rule_level_normalized'] = level / 15.0
            df['is_high_severity'] = (level >= 7).astype(int)
            df['is_critical_severity'] = (level >= 10).astype(int)
            df['is_low_severity'] = (level <= 3).astype(int)
        
        # 规则组特征
        if '_source.rule.groups' in df.columns:
            df['has_rule_groups'] = df['_source.rule.groups'].notna().astype(int)
            df['rule_groups_count'] = df['_source.rule.groups'].astype(str).str.count(',') + 1
        
        # 规则描述特征
        if '_source.rule.description' in df.columns:
            desc = df['_source.rule.description'].astype(str)
            df['rule_description_length'] = desc.str.len()
            df['rule_has_keywords'] = desc.str.contains(r'(?i)(attack|malware|virus|trojan|backdoor)', regex=True).astype(int)
        
        return df
    
    def _generate_improved_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成改进的标签"""
        self.logger.info("生成改进的标签")
        
        # 初始化恶意分数
        df['malicious_score'] = 0.0
        df['attack_classification'] = 'Normal_Activity'
        
        # 1. 基于规则级别的标签
        if '_source.rule.level' in df.columns:
            level = pd.to_numeric(df['_source.rule.level'], errors='coerce').fillna(0)
            rule_score = (level / 15.0) * self.label_strategies['rule_level']['weight']
            df['malicious_score'] += rule_score
            
            # 高严重性规则标记为可疑
            high_severity_mask = level >= self.label_strategies['rule_level']['threshold']
            df.loc[high_severity_mask, 'attack_classification'] = 'Suspicious_Activity'
        
        # 2. 基于MITRE ATT&CK的标签
        if '_source.rule.mitre_tactics' in df.columns:
            mitre_mask = df['_source.rule.mitre_tactics'].notna()
            mitre_score = self.label_strategies['mitre_tactics']['weight']
            df.loc[mitre_mask, 'malicious_score'] += mitre_score
            df.loc[mitre_mask, 'attack_classification'] = 'Low_Risk_Attack'
        
        # 3. 基于文件特征的标签
        if 'is_executable' in df.columns and 'is_temp_file' in df.columns:
            suspicious_file_mask = (df['is_executable'] == 1) & (df['is_temp_file'] == 1)
            file_score = self.label_strategies['file_suspicious']['weight']
            df.loc[suspicious_file_mask, 'malicious_score'] += file_score
            df.loc[suspicious_file_mask, 'attack_classification'] = 'Medium_Risk_Attack'
        
        # 4. 基于命令特征的标签
        if 'is_suspicious_command' in df.columns:
            suspicious_cmd_mask = df['is_suspicious_command'] == 1
            cmd_score = self.label_strategies['command_suspicious']['weight']
            df.loc[suspicious_cmd_mask, 'malicious_score'] += cmd_score
            df.loc[suspicious_cmd_mask, 'attack_classification'] = 'High_Risk_Attack'
        
        # 5. 基于用户权限的标签
        if 'is_root_user' in df.columns and 'is_privilege_command' in df.columns:
            privilege_mask = (df['is_root_user'] == 1) & (df['is_privilege_command'] == 1)
            df.loc[privilege_mask, 'malicious_score'] += 0.1
            df.loc[privilege_mask, 'attack_classification'] = 'High_Risk_Attack'
        
        # 6. 基于时间特征的标签
        if 'is_night' in df.columns and 'is_weekend' in df.columns:
            unusual_time_mask = (df['is_night'] == 1) | (df['is_weekend'] == 1)
            df.loc[unusual_time_mask, 'malicious_score'] += 0.05
        
        # 确保恶意分数在0-1范围内
        df['malicious_score'] = df['malicious_score'].clip(0, 1)
        
        # 根据恶意分数调整分类
        df.loc[df['malicious_score'] >= 0.8, 'attack_classification'] = 'High_Risk_Attack'
        df.loc[(df['malicious_score'] >= 0.6) & (df['malicious_score'] < 0.8), 'attack_classification'] = 'Medium_Risk_Attack'
        df.loc[(df['malicious_score'] >= 0.3) & (df['malicious_score'] < 0.6), 'attack_classification'] = 'Low_Risk_Attack'
        df.loc[(df['malicious_score'] >= 0.1) & (df['malicious_score'] < 0.3), 'attack_classification'] = 'Suspicious_Activity'
        
        # 统计标签分布
        label_counts = df['attack_classification'].value_counts()
        self.logger.info(f"标签分布: {label_counts.to_dict()}")
        
        return df
    
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """平衡数据集"""
        self.logger.info("平衡数据集")
        
        # 检查类别分布
        label_counts = df['attack_classification'].value_counts()
        total_samples = len(df)
        
        # 计算正样本比例
        positive_classes = ['Suspicious_Activity', 'Low_Risk_Attack', 'Medium_Risk_Attack', 'High_Risk_Attack']
        positive_count = sum(label_counts.get(cls, 0) for cls in positive_classes)
        positive_ratio = positive_count / total_samples
        
        self.logger.info(f"当前正样本比例: {positive_ratio:.3f}")
        
        # 如果正样本比例过低，进行过采样
        if positive_ratio < 0.1:  # 如果正样本比例低于10%
            self.logger.info("正样本比例过低，执行过采样")
            
            # 准备特征和标签
            feature_cols = [col for col in df.columns if col not in ['malicious_score', 'attack_classification', 'unified_timestamp']]
            X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            y = df['attack_classification']
            
            if len(X.columns) > 0 and len(X) > 0:
                try:
                    # 使用SMOTE进行过采样
                    smote = SMOTE(random_state=42, k_neighbors=min(3, positive_count-1))
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # 创建新的DataFrame
                    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                    df_resampled['attack_classification'] = y_resampled
                    df_resampled['malicious_score'] = df_resampled['attack_classification'].map({
                        'Normal_Activity': 0.0,
                        'Suspicious_Activity': 0.2,
                        'Low_Risk_Attack': 0.4,
                        'Medium_Risk_Attack': 0.6,
                        'High_Risk_Attack': 0.8
                    })
                    
                    # 合并原始数据和重采样数据
                    df = pd.concat([df, df_resampled], ignore_index=True)
                    
                    # 去重
                    df = df.drop_duplicates()
                    
                    self.logger.info(f"过采样完成: {len(df)} 个样本")
                    
                except Exception as e:
                    self.logger.warning(f"过采样失败: {e}，使用原始数据")
        
        # 如果正样本比例过高，进行欠采样
        elif positive_ratio > 0.8:  # 如果正样本比例高于80%
            self.logger.info("正样本比例过高，执行欠采样")
            
            # 对正常活动进行欠采样
            normal_mask = df['attack_classification'] == 'Normal_Activity'
            normal_samples = df[normal_mask]
            attack_samples = df[~normal_mask]
            
            if len(normal_samples) > len(attack_samples) * 2:
                # 随机欠采样正常样本
                normal_sampled = normal_samples.sample(n=len(attack_samples) * 2, random_state=42)
                df = pd.concat([normal_sampled, attack_samples], ignore_index=True)
                
                self.logger.info(f"欠采样完成: {len(df)} 个样本")
        
        # 最终统计
        final_label_counts = df['attack_classification'].value_counts()
        final_positive_count = sum(final_label_counts.get(cls, 0) for cls in positive_classes)
        final_positive_ratio = final_positive_count / len(df)
        
        self.logger.info(f"最终正样本比例: {final_positive_ratio:.3f}")
        self.logger.info(f"最终标签分布: {final_label_counts.to_dict()}")
        
        return df
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'total_records': len(df),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict()
        }
        
        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # 标签分布
        if 'attack_classification' in df.columns:
            stats['label_distribution'] = df['attack_classification'].value_counts().to_dict()
            stats['positive_ratio'] = (df['attack_classification'] != 'Normal_Activity').mean()
        
        return stats