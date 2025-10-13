"""
数据增强模块

针对Wazuh安全告警数据的数据增强策略
"""

import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import re

class WazuhDataAugmentation:
    """Wazuh数据增强器"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 定义高风险关键词模式
        self.high_risk_patterns = [
            r'(?i)trojaned',
            r'(?i)malicious',
            r'(?i)attack',
            r'(?i)intrusion',
            r'(?i)exploit',
            r'(?i)backdoor',
            r'(?i)rootkit',
            r'(?i)ransomware',
            r'(?i)phishing',
            r'(?i)suspicious',
            r'(?i)anomaly',
            r'(?i)unauthorized',
            r'(?i)failed login',
            r'(?i)brute force',
            r'(?i)privilege escalation'
        ]
        
        # 定义MITRE ATT&CK技术模式
        self.mitre_patterns = [
            r'T\d{4}',  # MITRE技术ID
            r'TA\d{4}',  # MITRE战术ID
        ]
        
        # 定义系统文件路径模式
        self.system_paths = [
            r'/bin/',
            r'/sbin/',
            r'/usr/bin/',
            r'/usr/sbin/',
            r'/etc/',
            r'/var/log/',
            r'/root/',
            r'/home/'
        ]
    
    def augment_positive_samples(self, df: pd.DataFrame, target_ratio: float = 0.2) -> pd.DataFrame:
        """
        增强正样本数据
        
        Args:
            df: 原始数据框
            target_ratio: 目标正样本比例
            
        Returns:
            增强后的数据框
        """
        self.logger.info(f"开始数据增强，目标正样本比例: {target_ratio}")
        
        # 计算当前正样本比例
        if 'malicious_score' in df.columns:
            current_positive_ratio = (df['malicious_score'] >= 0.4).mean()
        else:
            current_positive_ratio = 0.0
        
        self.logger.info(f"当前正样本比例: {current_positive_ratio:.3f}")
        
        if current_positive_ratio >= target_ratio:
            self.logger.info("正样本比例已达到目标，无需增强")
            return df
        
        # 策略1：基于规则等级的数据增强
        df_enhanced = self._enhance_by_rule_level(df, target_ratio)
        
        # 策略2：基于MITRE ATT&CK的数据增强
        df_enhanced = self._enhance_by_mitre_patterns(df_enhanced, target_ratio)
        
        # 策略3：基于关键词的数据增强
        df_enhanced = self._enhance_by_keywords(df_enhanced, target_ratio)
        
        # 策略4：基于文件路径的数据增强
        df_enhanced = self._enhance_by_file_paths(df_enhanced, target_ratio)
        
        # 策略5：基于时间模式的数据增强
        df_enhanced = self._enhance_by_temporal_patterns(df_enhanced, target_ratio)
        
        # 验证增强效果
        if 'malicious_score' in df_enhanced.columns:
            new_positive_ratio = (df_enhanced['malicious_score'] >= 0.4).mean()
            self.logger.info(f"增强后正样本比例: {new_positive_ratio:.3f}")
        
        return df_enhanced
    
    def _enhance_by_rule_level(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """基于规则等级的数据增强"""
        if '_source.rule.level' not in df.columns:
            return df
        
        # 获取高等级规则
        rule_levels = pd.to_numeric(df['_source.rule.level'], errors='coerce').fillna(0)
        high_level_mask = rule_levels >= 7
        
        if high_level_mask.sum() == 0:
            return df
        
        # 复制高等级规则数据
        high_level_data = df[high_level_mask].copy()
        
        # 添加变体
        variants = []
        for i in range(3):  # 生成3个变体
            variant = high_level_data.copy()
            
            # 变体1：修改时间戳
            if '_source.@timestamp' in variant.columns:
                variant['_source.@timestamp'] = self._modify_timestamp(variant['_source.@timestamp'])
            
            # 变体2：修改主机名
            if '_source.agent.name' in variant.columns:
                variant['_source.agent.name'] = self._modify_hostname(variant['_source.agent.name'])
            
            # 变体3：修改文件路径
            if '_source.data.file' in variant.columns:
                variant['_source.data.file'] = self._modify_file_path(variant['_source.data.file'])
            
            variants.append(variant)
        
        # 合并变体
        enhanced_df = pd.concat([df] + variants, ignore_index=True)
        
        self.logger.info(f"基于规则等级增强: 原始{len(df)}条 -> 增强后{len(enhanced_df)}条")
        return enhanced_df
    
    def _enhance_by_mitre_patterns(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """基于MITRE ATT&CK模式的数据增强"""
        if '_source.rule.mitre_techniques' not in df.columns:
            return df
        
        # 找到包含MITRE技术的记录
        mitre_mask = df['_source.rule.mitre_techniques'].notna()
        mitre_data = df[mitre_mask].copy()
        
        if len(mitre_data) == 0:
            return df
        
        # 生成MITRE技术变体
        variants = []
        for i in range(2):  # 生成2个变体
            variant = mitre_data.copy()
            
            # 添加额外的MITRE技术
            variant['_source.rule.mitre_techniques'] = self._add_mitre_techniques(
                variant['_source.rule.mitre_techniques']
            )
            
            # 修改描述以包含更多攻击信息
            if '_source.rule.description' in variant.columns:
                variant['_source.rule.description'] = self._enhance_attack_description(
                    variant['_source.rule.description']
                )
            
            variants.append(variant)
        
        # 合并变体
        enhanced_df = pd.concat([df] + variants, ignore_index=True)
        
        self.logger.info(f"基于MITRE模式增强: 原始{len(df)}条 -> 增强后{len(enhanced_df)}条")
        return enhanced_df
    
    def _enhance_by_keywords(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """基于关键词的数据增强"""
        # 找到包含高风险关键词的记录
        text_columns = ['_source.full_log', '_source.rule.description', '_source.data.command']
        keyword_mask = pd.Series([False] * len(df))
        
        for col in text_columns:
            if col in df.columns:
                for pattern in self.high_risk_patterns:
                    mask = df[col].astype(str).str.contains(pattern, na=False)
                    keyword_mask |= mask
        
        keyword_data = df[keyword_mask].copy()
        
        if len(keyword_data) == 0:
            return df
        
        # 生成关键词变体
        variants = []
        for i in range(2):  # 生成2个变体
            variant = keyword_data.copy()
            
            # 增强描述
            if '_source.rule.description' in variant.columns:
                variant['_source.rule.description'] = self._enhance_attack_description(
                    variant['_source.rule.description']
                )
            
            # 增强命令
            if '_source.data.command' in variant.columns:
                variant['_source.data.command'] = self._enhance_command(
                    variant['_source.data.command']
                )
            
            variants.append(variant)
        
        # 合并变体
        enhanced_df = pd.concat([df] + variants, ignore_index=True)
        
        self.logger.info(f"基于关键词增强: 原始{len(df)}条 -> 增强后{len(enhanced_df)}条")
        return enhanced_df
    
    def _enhance_by_file_paths(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """基于文件路径的数据增强"""
        if '_source.data.file' not in df.columns:
            return df
        
        # 找到系统文件路径的记录
        system_file_mask = df['_source.data.file'].astype(str).str.contains(
            '|'.join(self.system_paths), na=False
        )
        
        system_file_data = df[system_file_mask].copy()
        
        if len(system_file_data) == 0:
            return df
        
        # 生成文件路径变体
        variants = []
        for i in range(2):  # 生成2个变体
            variant = system_file_data.copy()
            
            # 修改文件路径
            variant['_source.data.file'] = self._modify_file_path(variant['_source.data.file'])
            
            # 增加文件操作相关描述
            if '_source.rule.description' in variant.columns:
                variant['_source.rule.description'] = self._add_file_operation_description(
                    variant['_source.rule.description']
                )
            
            variants.append(variant)
        
        # 合并变体
        enhanced_df = pd.concat([df] + variants, ignore_index=True)
        
        self.logger.info(f"基于文件路径增强: 原始{len(df)}条 -> 增强后{len(enhanced_df)}条")
        return enhanced_df
    
    def _enhance_by_temporal_patterns(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """基于时间模式的数据增强"""
        if '_source.@timestamp' not in df.columns:
            return df
        
        # 找到夜间或异常时间的记录
        timestamps = pd.to_datetime(df['_source.@timestamp'], errors='coerce')
        night_mask = (timestamps.dt.hour >= 22) | (timestamps.dt.hour <= 6)
        weekend_mask = timestamps.dt.weekday >= 5
        
        temporal_mask = night_mask | weekend_mask
        temporal_data = df[temporal_mask].copy()
        
        if len(temporal_data) == 0:
            return df
        
        # 生成时间变体
        variants = []
        for i in range(2):  # 生成2个变体
            variant = temporal_data.copy()
            
            # 修改时间戳到不同的异常时间
            variant['_source.@timestamp'] = self._modify_timestamp(variant['_source.@timestamp'])
            
            # 增加时间相关的攻击描述
            if '_source.rule.description' in variant.columns:
                variant['_source.rule.description'] = self._add_temporal_attack_description(
                    variant['_source.rule.description']
                )
            
            variants.append(variant)
        
        # 合并变体
        enhanced_df = pd.concat([df] + variants, ignore_index=True)
        
        self.logger.info(f"基于时间模式增强: 原始{len(df)}条 -> 增强后{len(enhanced_df)}条")
        return enhanced_df
    
    def _modify_timestamp(self, timestamps: pd.Series) -> pd.Series:
        """修改时间戳"""
        modified = timestamps.copy()
        
        for i, ts in enumerate(timestamps):
            try:
                if pd.notna(ts):
                    dt = pd.to_datetime(ts)
                    # 添加随机时间偏移（1-24小时）
                    offset_hours = np.random.randint(1, 25)
                    modified.iloc[i] = dt + pd.Timedelta(hours=offset_hours)
            except:
                continue
        
        return modified
    
    def _modify_hostname(self, hostnames: pd.Series) -> pd.Series:
        """修改主机名"""
        modified = hostnames.copy()
        
        for i, hostname in enumerate(hostnames):
            if pd.notna(hostname):
                # 添加后缀或前缀
                suffix = f"-{np.random.randint(1, 100)}"
                modified.iloc[i] = str(hostname) + suffix
        
        return modified
    
    def _modify_file_path(self, file_paths: pd.Series) -> pd.Series:
        """修改文件路径"""
        modified = file_paths.copy()
        
        for i, path in enumerate(file_paths):
            if pd.notna(path):
                path_str = str(path)
                # 添加随机目录或修改文件名
                if '/' in path_str:
                    parts = path_str.split('/')
                    if len(parts) > 1:
                        # 修改文件名
                        filename = parts[-1]
                        name, ext = os.path.splitext(filename) if '.' in filename else (filename, '')
                        new_filename = f"{name}_variant_{np.random.randint(1, 100)}{ext}"
                        parts[-1] = new_filename
                        modified.iloc[i] = '/'.join(parts)
        
        return modified
    
    def _add_mitre_techniques(self, mitre_techniques: pd.Series) -> pd.Series:
        """添加MITRE技术"""
        modified = mitre_techniques.copy()
        
        additional_techniques = ['T1055', 'T1083', 'T1105', 'T1112', 'T1140']
        
        for i, techniques in enumerate(mitre_techniques):
            if pd.notna(techniques):
                tech_str = str(techniques)
                # 添加随机技术
                additional = np.random.choice(additional_techniques, size=1)[0]
                if additional not in tech_str:
                    modified.iloc[i] = f"{tech_str}, {additional}"
        
        return modified
    
    def _enhance_attack_description(self, descriptions: pd.Series) -> pd.Series:
        """增强攻击描述"""
        modified = descriptions.copy()
        
        attack_phrases = [
            "suspicious activity detected",
            "potential security threat",
            "anomalous behavior observed",
            "unauthorized access attempt",
            "malicious payload identified"
        ]
        
        for i, desc in enumerate(descriptions):
            if pd.notna(desc):
                desc_str = str(desc)
                # 添加攻击短语
                phrase = np.random.choice(attack_phrases)
                modified.iloc[i] = f"{desc_str} - {phrase}"
        
        return modified
    
    def _enhance_command(self, commands: pd.Series) -> pd.Series:
        """增强命令"""
        modified = commands.copy()
        
        suspicious_commands = [
            "wget", "curl", "nc", "netcat", "base64", "chmod 777"
        ]
        
        for i, cmd in enumerate(commands):
            if pd.notna(cmd):
                cmd_str = str(cmd)
                # 添加可疑命令
                if not any(susp in cmd_str.lower() for susp in suspicious_commands):
                    additional = np.random.choice(suspicious_commands)
                    modified.iloc[i] = f"{cmd_str} && {additional}"
        
        return modified
    
    def _add_file_operation_description(self, descriptions: pd.Series) -> pd.Series:
        """添加文件操作描述"""
        modified = descriptions.copy()
        
        file_operations = [
            "file modification detected",
            "sensitive file accessed",
            "system file altered",
            "configuration file changed"
        ]
        
        for i, desc in enumerate(descriptions):
            if pd.notna(desc):
                desc_str = str(desc)
                operation = np.random.choice(file_operations)
                modified.iloc[i] = f"{desc_str} - {operation}"
        
        return modified
    
    def _add_temporal_attack_description(self, descriptions: pd.Series) -> pd.Series:
        """添加时间相关攻击描述"""
        modified = descriptions.copy()
        
        temporal_phrases = [
            "unusual time activity",
            "off-hours access attempt",
            "weekend security event",
            "nighttime anomaly detected"
        ]
        
        for i, desc in enumerate(descriptions):
            if pd.notna(desc):
                desc_str = str(desc)
                phrase = np.random.choice(temporal_phrases)
                modified.iloc[i] = f"{desc_str} - {phrase}"
        
        return modified
