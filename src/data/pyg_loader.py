"""
基于PyTorch Geometric的数据加载器

负责从Linux-APT-Dataset加载数据并构建PyG异构图
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import psutil
import gc
import networkx as nx

try:
    from ..config.improved_config import DataConfig
except ImportError:
    from ..config.simple_config import SimpleDataConfig as DataConfig


class PyG_LinuxAPTDataLoader:
    """基于PyG的Linux APT数据加载器"""
    
    def __init__(self, config: DataConfig):
        """
        初始化数据加载器
        
        Args:
            config: 数据配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self._data_cache = None
        self._processed_data = None
        
        # 节点和边类型定义 - 针对Wazuh安全告警优化
        self.node_types = [
            'alert', 'rule', 'file', 'command', 'user', 'process', 
            'host', 'agent', 'mitre_tactic', 'mitre_technique',
            'timestamp', 'ip', 'domain', 'port', 'service'
        ]
        
        self.edge_types = [
            # 核心安全关系
            ('alert', 'triggered_by', 'rule'),
            ('alert', 'involves', 'file'),
            ('alert', 'executed', 'command'),
            ('alert', 'by_user', 'user'),
            ('alert', 'involves_process', 'process'),
            ('alert', 'at_time', 'timestamp'),
            ('alert', 'detected_on', 'host'),
            ('alert', 'reported_by', 'agent'),
            
            # MITRE ATT&CK关系
            ('rule', 'maps_to', 'mitre_tactic'),
            ('rule', 'uses', 'mitre_technique'),
            ('alert', 'exhibits', 'mitre_tactic'),
            ('alert', 'demonstrates', 'mitre_technique'),
            
            # 文件和命令关系
            ('file', 'executed_by', 'command'),
            ('file', 'accessed_by', 'process'),
            ('file', 'owned_by', 'user'),
            ('file', 'located_on', 'host'),
            
            # 进程和用户关系
            ('process', 'executed_by', 'user'),
            ('process', 'runs_on', 'host'),
            ('process', 'connects_to', 'ip'),
            ('process', 'uses_port', 'port'),
            
            # 网络关系
            ('alert', 'connects_to', 'ip'),
            ('alert', 'connects_to', 'domain'),
            ('alert', 'uses_port', 'port'),
            ('alert', 'affects_service', 'service'),
            ('host', 'has_ip', 'ip'),
            ('host', 'runs_service', 'service'),
            ('host', 'has_open_port', 'port'),
            ('ip', 'resolves_to', 'domain'),
            ('service', 'listens_on', 'port'),
            
            # 时间关系
            ('alert', 'follows', 'alert'),  # 时序关系
            ('process', 'spawns', 'process'),  # 进程关系
        ]
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """加载Linux APT数据集"""
        if self._data_cache is not None and not force_reload:
            self.logger.info("使用缓存的数据")
            return self._data_cache
        
        self.logger.info(f"加载数据: {self.config.data.data_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.config.data.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data.data_path}")
        
        try:
            # 根据文件大小选择加载策略
            file_size = os.path.getsize(self.config.data.data_path) / (1024 * 1024)  # MB
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            
            if file_size > available_memory * 0.5:
                self.logger.info("使用分块加载策略")
                df = self._load_in_chunks()
            else:
                self.logger.info("使用直接加载策略")
                df = self._load_directly()
            
            # 缓存数据
            self._data_cache = df
            return df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def _load_directly(self) -> pd.DataFrame:
        """直接加载数据"""
        try:
            df = pd.read_csv(
                self.config.data.data_path,
                low_memory=False,
                on_bad_lines='skip'
            )
            return df
        except Exception as e:
            self.logger.warning(f"直接加载失败: {e}")
            try:
                df = pd.read_csv(
                    self.config.data.data_path,
                    low_memory=False,
                    on_bad_lines='skip',
                    encoding='utf-8',
                    sep=','
                )
                return df
            except Exception as e2:
                self.logger.error(f"所有直接加载方法都失败: {e2}")
                raise
    
    def _load_in_chunks(self) -> pd.DataFrame:
        """分块加载数据"""
        self.logger.info(f"使用分块加载，块大小: {self.config.chunk_size}")
        
        chunks = []
        chunk_count = 0
        
        try:
            for chunk in pd.read_csv(
                self.config.data.data_path,
                chunksize=self.config.chunk_size,
                low_memory=False,
                on_bad_lines='skip'
            ):
                chunks.append(chunk)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    memory_usage = self._get_memory_usage()
                    self.logger.info(f"已加载 {chunk_count} 个块，内存使用: {memory_usage:.2f} MB")
                    
                    if memory_usage > psutil.virtual_memory().total * self.config.max_memory_usage:
                        gc.collect()
                        self.logger.warning("内存使用过高，执行垃圾回收")
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            return df
            
        except Exception as e:
            self.logger.error(f"分块加载失败: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据 - 使用改进的方法"""
        self.logger.info("开始改进的数据预处理")
        
        # 创建数据副本
        processed_df = df.copy()
        
        # 使用改进的APT数据处理器
        try:
            from .improved_apt_data_processor import ImprovedAPTDataProcessor
            processor = ImprovedAPTDataProcessor(self.config)
            
            # 使用改进的数据质量修复方法
            processed_df = processor.process_raw_data(processed_df)
            
            self.logger.info("使用改进的APT数据处理器完成预处理")
            
        except Exception as e:
            self.logger.warning(f"无法使用改进的处理器: {e}，使用原始方法")
            # 回退到原始方法
            processed_df = self._preprocess_data_original(processed_df)
        
        # 缓存处理后的数据
        self._processed_data = processed_df
        
        return processed_df
    
    def _preprocess_data_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """原始预处理方法（备用）"""
        self.logger.info("使用原始预处理方法")
        
        # 1. 处理缺失值
        df = self._handle_missing_values(df)
        
        # 2. 处理时间戳
        df = self._process_timestamps(df)
        
        # 3. 数据清洗
        df = self._clean_data(df)
        
        # 4. 特征工程
        df = self._engineer_features(df)
        
        self.logger.info(f"原始预处理完成，最终数据形状: {df.shape}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        self.logger.info("处理缺失值")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理时间戳"""
        self.logger.info("处理时间戳")
        
        # 查找时间戳列
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        self.logger.info(f"找到时间戳列: {timestamp_cols}")
        
        # 创建一个统一的时间戳列
        df['processed_timestamp'] = None
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    parsed_timestamps = self._parse_timestamps_flexible(df[col])
                    valid_count = parsed_timestamps.notna().sum()
                    self.logger.info(f"列 {col} 有效时间戳数量: {valid_count}/{len(df)}")
                    
                    if valid_count > 0:
                        if df['processed_timestamp'].notna().sum() < valid_count:
                            df['processed_timestamp'] = parsed_timestamps
                            self.logger.info(f"使用列 {col} 作为主时间戳")
                    
                except Exception as e:
                    self.logger.warning(f"处理时间戳列 {col} 时出错: {e}")
        
        # 如果仍然没有有效时间戳，创建基于索引的伪时间戳
        if df['processed_timestamp'].notna().sum() == 0:
            self.logger.warning("没有找到有效时间戳，创建基于索引的伪时间戳")
            base_time = pd.Timestamp('2020-01-01')
            df['processed_timestamp'] = base_time + pd.to_timedelta(df.index, unit='minutes')
        
        return df
    
    def _parse_timestamps_flexible(self, series: pd.Series) -> pd.Series:
        """
        灵活解析时间戳，支持多种格式
        
        Args:
            series: 包含时间戳的pandas Series
            
        Returns:
            解析后的时间戳Series
        """
        result = pd.Series(index=series.index, dtype='datetime64[ns]')
        
        # 定义多种时间戳格式，优先Wazuh格式
        timestamp_formats = [
            # Wazuh特定格式（优先级最高）
            '%b %d, %Y @ %H:%M:%S.%f',     # Oct 1, 2023 @ 00:49:18.889
            '%b %d, %Y @ %H:%M:%S',        # Oct 1, 2023 @ 00:49:18
            '%B %d, %Y @ %H:%M:%S.%f',     # October 1, 2023 @ 00:49:18.889
            '%B %d, %Y @ %H:%M:%S',        # October 1, 2023 @ 00:49:18
            '%b %d %Y @ %H:%M:%S.%f',      # Oct 1 2023 @ 00:49:18.889
            '%b %d %Y @ %H:%M:%S',         # Oct 1 2023 @ 00:49:18
            '%d %b %Y @ %H:%M:%S.%f',      # 1 Oct 2023 @ 00:49:18.889
            '%d %b %Y @ %H:%M:%S',         # 1 Oct 2023 @ 00:49:18
            
            # 标准格式
            '%Y-%m-%d %H:%M:%S',           # 2023-01-01 12:00:00
            '%Y-%m-%d %H:%M:%S.%f',        # 2023-01-01 12:00:00.123456
            '%Y-%m-%dT%H:%M:%S',           # 2023-01-01T12:00:00
            '%Y-%m-%dT%H:%M:%S.%f',        # 2023-01-01T12:00:00.123456
            '%Y-%m-%dT%H:%M:%SZ',          # 2023-01-01T12:00:00Z
            '%Y-%m-%dT%H:%M:%S.%fZ',       # 2023-01-01T12:00:00.123456Z
            '%Y-%m-%dT%H:%M:%S%z',         # 2023-01-01T12:00:00+00:00
            '%Y-%m-%dT%H:%M:%S.%f%z',      # 2023-01-01T12:00:00.123456+00:00
            '%Y-%m-%d',                    # 2023-01-01
            '%Y/%m/%d %H:%M:%S',           # 2023/01/01 12:00:00
            '%Y/%m/%d %H:%M:%S.%f',        # 2023/01/01 12:00:00.123456
            '%Y/%m/%d',                    # 2023/01/01
            '%m/%d/%Y %H:%M:%S',           # 01/01/2023 12:00:00
            '%m/%d/%Y %H:%M:%S.%f',        # 01/01/2023 12:00:00.123456
            '%m/%d/%Y',                    # 01/01/2023
            '%d/%m/%Y %H:%M:%S',           # 01/01/2023 12:00:00
            '%d/%m/%Y %H:%M:%S.%f',        # 01/01/2023 12:00:00.123456
            '%d/%m/%Y',                    # 01/01/2023
        ]
        
        for idx, value in series.items():
            if pd.isna(value) or value == '':
                continue
                
            try:
                parsed = None
                
                # 处理Wazuh格式的时间戳（包含@符号）
                if parsed is None and isinstance(value, str) and '@' in value:
                    try:
                        # 直接尝试Wazuh特定格式（不清理@符号）
                        wazuh_formats = [
                            '%b %d, %Y @ %H:%M:%S.%f',  # Oct 1, 2023 @ 00:49:18.889
                            '%b %d, %Y @ %H:%M:%S',     # Oct 1, 2023 @ 00:49:18
                            '%B %d, %Y @ %H:%M:%S.%f',  # October 1, 2023 @ 00:49:18.889
                            '%B %d, %Y @ %H:%M:%S',     # October 1, 2023 @ 00:49:18
                            '%b %d %Y @ %H:%M:%S.%f',   # Oct 1 2023 @ 00:49:18.889
                            '%b %d %Y @ %H:%M:%S',      # Oct 1 2023 @ 00:49:18
                            '%d %b %Y @ %H:%M:%S.%f',   # 1 Oct 2023 @ 00:49:18.889
                            '%d %b %Y @ %H:%M:%S'       # 1 Oct 2023 @ 00:49:18
                        ]
                        
                        for fmt in wazuh_formats:
                            try:
                                parsed = pd.to_datetime(value, format=fmt)
                                break
                            except:
                                continue
                        
                        # 如果直接格式失败，尝试清理@符号后解析
                        if parsed is None:
                            clean_value = value.replace('@', '').strip()
                            clean_formats = [
                                '%b %d, %Y %H:%M:%S.%f',  # Oct 1, 2023 00:49:18.889
                                '%b %d, %Y %H:%M:%S',     # Oct 1, 2023 00:49:18
                                '%B %d, %Y %H:%M:%S.%f',  # October 1, 2023 00:49:18.889
                                '%B %d, %Y %H:%M:%S',     # October 1, 2023 00:49:18
                                '%b %d %Y %H:%M:%S.%f',   # Oct 1 2023 00:49:18.889
                                '%b %d %Y %H:%M:%S',      # Oct 1 2023 00:49:18
                                '%d %b %Y %H:%M:%S.%f',   # 1 Oct 2023 00:49:18.889
                                '%d %b %Y %H:%M:%S'       # 1 Oct 2023 00:49:18
                            ]
                            
                            for fmt in clean_formats:
                                try:
                                    parsed = pd.to_datetime(clean_value, format=fmt)
                                    break
                                except:
                                    continue
                        
                        # 如果Wazuh格式都失败，尝试pandas自动解析
                        if parsed is None:
                            parsed = pd.to_datetime(value, errors='coerce')
                    except:
                        pass
                
                # 处理Unix时间戳
                if parsed is None:
                    try:
                        if isinstance(value, (int, float)) and value > 1000000000:
                            parsed = pd.to_datetime(value, unit='s')
                    except:
                        pass
                
                # 尝试使用pandas的自动解析
                if parsed is None:
                    try:
                        parsed = pd.to_datetime(value, infer_datetime_format=True)
                    except:
                        pass
                
                # 尝试手动格式匹配
                if parsed is None and isinstance(value, str):
                    for fmt in timestamp_formats:
                        try:
                            parsed = pd.to_datetime(value, format=fmt)
                            break
                        except:
                            continue
                
                # 如果解析成功，存储结果
                if parsed is not None:
                    result[idx] = parsed
                    
            except Exception as e:
                self.logger.debug(f"解析时间戳失败 {value}: {e}")
                continue
        
        return result
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        self.logger.info("数据清洗")
        
        # 移除完全重复的行
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"移除了 {removed_rows} 个重复行")
        
        # 移除空行
        initial_rows = len(df)
        df = df.dropna(how='all')
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.info(f"移除了 {removed_rows} 个空行")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程 - 针对Wazuh安全告警优化"""
        self.logger.info("开始Wazuh安全特征工程")
        
        # 1. 安全等级特征
        if '_source.rule.level' in df.columns:
            df['rule_level_numeric'] = pd.to_numeric(df['_source.rule.level'], errors='coerce').fillna(0)
            df['rule_level_normalized'] = df['rule_level_numeric'] / 15.0  # 归一化到0-1
            df['is_high_severity'] = (df['rule_level_numeric'] >= 7).astype(int)
            df['is_critical_severity'] = (df['rule_level_numeric'] >= 10).astype(int)
        
        # 2. 文件路径特征
        if '_source.data.file' in df.columns:
            df['file_path_length'] = df['_source.data.file'].astype(str).str.len()
            df['is_system_file'] = df['_source.data.file'].astype(str).str.contains(r'(?i)(/bin/|/sbin/|/usr/bin/)', regex=True).astype(int)
            df['is_temp_file'] = df['_source.data.file'].astype(str).str.contains(r'(?i)(/tmp/|/var/tmp/)', regex=True).astype(int)
            df['is_executable'] = df['_source.data.file'].astype(str).str.contains(r'(?i)(\.exe|\.bat|\.cmd|\.sh)$', regex=True).astype(int)
        
        # 3. 命令特征
        if '_source.data.command' in df.columns:
            df['command_length'] = df['_source.data.command'].astype(str).str.len()
            df['is_network_command'] = df['_source.data.command'].astype(str).str.contains(r'(?i)(wget|curl|nc|netcat)', regex=True).astype(int)
            df['is_privilege_command'] = df['_source.data.command'].astype(str).str.contains(r'(?i)(sudo|su|chmod|chown)', regex=True).astype(int)
            df['is_encoding_command'] = df['_source.data.command'].astype(str).str.contains(r'(?i)(base64|encode|decode)', regex=True).astype(int)
        
        # 4. MITRE ATT&CK特征
        if '_source.rule.mitre_tactics' in df.columns:
            df['has_mitre_tactics'] = df['_source.rule.mitre_tactics'].notna().astype(int)
            df['mitre_tactics_count'] = df['_source.rule.mitre_tactics'].astype(str).str.count('TA\\d{4}')
        
        if '_source.rule.mitre_techniques' in df.columns:
            df['has_mitre_techniques'] = df['_source.rule.mitre_techniques'].notna().astype(int)
            df['mitre_techniques_count'] = df['_source.rule.mitre_techniques'].astype(str).str.count('T\\d{4}')
        
        # 5. 时间特征
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if col in df.columns and df[col].dtype == 'datetime64[ns]':
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
                df[f'{col}_is_weekend'] = (df[col].dt.weekday >= 5).astype(int)
                df[f'{col}_is_night'] = ((df[col].dt.hour >= 22) | (df[col].dt.hour <= 6)).astype(int)
        
        # 6. 用户和进程特征
        if '_source.data.user' in df.columns:
            df['is_root_user'] = df['_source.data.user'].astype(str).str.contains(r'(?i)(root|admin)', regex=True).astype(int)
        
        if '_source.data.process' in df.columns:
            df['process_name_length'] = df['_source.data.process'].astype(str).str.len()
            df['is_system_process'] = df['_source.data.process'].astype(str).str.contains(r'(?i)(systemd|init|kernel)', regex=True).astype(int)
        
        # 7. 网络特征
        if '_source.data.srcip' in df.columns:
            df['has_src_ip'] = df['_source.data.srcip'].notna().astype(int)
        
        if '_source.data.dstport' in df.columns:
            df['has_dst_port'] = df['_source.data.dstport'].notna().astype(int)
            df['is_common_port'] = df['_source.data.dstport'].astype(str).str.contains(r'(?i)(80|443|22|21|25)', regex=True).astype(int)
        
        # 8. 恶意分数特征（如果存在）
        if 'malicious_score' in df.columns:
            df['is_malicious'] = (df['malicious_score'] >= 0.4).astype(int)
            df['malicious_level'] = pd.cut(df['malicious_score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 1, 2, 3, 4])
        
        # 9. 添加文本长度特征（保持原有功能）
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        for col in text_cols:
            if col in df.columns and f'{col}_length' not in df.columns:
                df[f'{col}_length'] = df[col].astype(str).str.len()
        
        self.logger.info(f"Wazuh安全特征工程完成")
        return df
    
    def build_hetero_graph(self, df: pd.DataFrame) -> HeteroData:
        """构建PyG异构图"""
        self.logger.info("构建PyG异构图")
        
        # 数据采样（如果数据量过大）- 进一步减少采样量
        if len(df) > 20000:  # 从50000减少到20000
            sample_size = min(20000, len(df))
            self.logger.info(f"数据量过大({len(df)}行)，采样{sample_size}行进行处理")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # 创建异构图数据
        data = HeteroData()
        
        # 节点ID映射
        node_id_maps = {ntype: {} for ntype in self.node_types}
        
        # 统计信息
        stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'alerts_processed': 0
        }
        
        # 分批处理数据 - 减少批次大小以节省内存
        batch_size = 500  # 从1000减少到500
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            self.logger.info(f"处理批次 {batch_idx + 1}/{total_batches} "
                           f"(行 {start_idx}-{end_idx-1})")
            
            # 处理当前批次
            for global_idx, row in batch_df.iterrows():
                try:
                    # 创建警报节点
                    alert_id = f"alert_{global_idx}"
                    self._add_node_if_not_exists(alert_id, 'alert', node_id_maps, stats)
                    
                    # 提取并添加各种实体节点
                    self._extract_and_add_entities(row, alert_id, node_id_maps, stats)
                    
                except Exception as e:
                    self.logger.warning(f"处理第 {global_idx} 行时出错: {e}")
                    continue
            
            # 批次完成后清理内存
            gc.collect()
        
        # 构建边索引
        edge_index_dict = self._build_edge_indices(node_id_maps, df)
        
        # 添加边到异构图
        for edge_type, edge_index in edge_index_dict.items():
            if len(edge_index[0]) > 0:
                data[edge_type].edge_index = edge_index
        
        # 添加节点特征
        self._add_node_features(data, node_id_maps)
        
        # 添加时间戳信息到异构图
        if 'processed_timestamp' in df.columns:
            valid_timestamps = df['processed_timestamp'].dropna()
            if len(valid_timestamps) > 0:
                # 将时间戳转换为数值（Unix时间戳）
                numeric_timestamps = torch.tensor([
                    ts.timestamp() if hasattr(ts, 'timestamp') else float(ts)
                    for ts in valid_timestamps
                ], dtype=torch.float32)
                
                data.timestamps = numeric_timestamps  # 使用复数形式避免与节点类型冲突
                self.logger.info(f"添加时间戳信息: {len(numeric_timestamps)} 个时间戳")
            else:
                self.logger.warning("没有有效时间戳数据")
        else:
            self.logger.warning("数据中没有processed_timestamp列")
        
        self.logger.info(f"PyG图构建完成:")
        self.logger.info(f"  节点类型: {data.node_types}")
        self.logger.info(f"  边类型: {data.edge_types}")
        
        return data
    
    def _extract_and_add_entities(self, row: pd.Series, alert_id: str, 
                                 node_id_maps: Dict, stats: Dict):
        """提取并添加实体节点"""
        
        # 提取主机信息
        host_name = self._extract_value(row, ['_source.agent.name', '_source.manager.name'])
        if host_name:
            self._add_node_if_not_exists(host_name, 'host', node_id_maps, stats)
        
        # 提取代理信息
        agent_id = self._extract_value(row, ['_source.agent.id'])
        if agent_id:
            self._add_node_if_not_exists(agent_id, 'agent', node_id_maps, stats)
        
        # 提取规则信息
        rule_id = self._extract_value(row, ['_source.rule.id'])
        if rule_id:
            self._add_node_if_not_exists(rule_id, 'rule', node_id_maps, stats)
        
        # 提取文件信息
        file_path = self._extract_value(row, ['_source.syscheck.path', '_source.data.file', '_source.data.sca.check.file'])
        if file_path:
            self._add_node_if_not_exists(file_path, 'file', node_id_maps, stats)
        
        # 提取命令信息
        command = self._extract_value(row, ['_source.data.command', '_source.data.sca.check.command'])
        if command:
            self._add_node_if_not_exists(command, 'command', node_id_maps, stats)
        
        # 提取用户信息
        user = self._extract_value(row, ['_source.data.srcuser', '_source.data.dstuser', '_source.data.uid'])
        if user:
            self._add_node_if_not_exists(user, 'user', node_id_maps, stats)
        
        # 提取进程信息
        process = self._extract_value(row, ['_source.data.process', '_source.predecoder.program_name'])
        if process:
            self._add_node_if_not_exists(process, 'process', node_id_maps, stats)
        
        # 提取IP信息
        src_ip = self._extract_value(row, ['_source.data.srcip', '_source.agent.ip'])
        if src_ip:
            self._add_node_if_not_exists(src_ip, 'ip', node_id_maps, stats)
        
        dst_ip = self._extract_value(row, ['_source.data.dstip'])
        if dst_ip and dst_ip != src_ip:
            self._add_node_if_not_exists(dst_ip, 'ip', node_id_maps, stats)
        
        # 提取域名信息
        domain = self._extract_value(row, ['_source.data.url', '_source.data.hostname', '_source.predecoder.hostname'])
        if domain:
            self._add_node_if_not_exists(domain, 'domain', node_id_maps, stats)
        
        # 提取时间戳信息
        timestamp = self._extract_value(row, ['processed_timestamp', '_source.@timestamp', '_source.timestamp', '_source.predecoder.timestamp'])
        if timestamp:
            timestamp_str = self._normalize_timestamp(timestamp)
            if timestamp_str:
                self._add_node_if_not_exists(timestamp_str, 'timestamp', node_id_maps, stats)
        
        # 提取MITRE ATT&CK战术
        mitre_tactics = self._extract_value(row, ['_source.rule.mitre_tactics', '_source.rule.mitre.tactic'])
        if mitre_tactics:
            tactics_list = self._parse_mitre_tactics(mitre_tactics)
            for tactic in tactics_list:
                self._add_node_if_not_exists(tactic, 'mitre_tactic', node_id_maps, stats)
        
        # 提取MITRE ATT&CK技术
        mitre_techniques = self._extract_value(row, ['_source.rule.mitre_techniques', '_source.rule.mitre.technique'])
        if mitre_techniques:
            techniques_list = self._parse_mitre_techniques(mitre_techniques)
            for technique in techniques_list:
                self._add_node_if_not_exists(technique, 'mitre_technique', node_id_maps, stats)
        
        # 提取端口信息
        port = self._extract_value(row, ['_source.data.dstport', '_source.data.srcport'])
        if port:
            port_str = f"port_{port}"
            self._add_node_if_not_exists(port_str, 'port', node_id_maps, stats)
    
    def _extract_value(self, row: pd.Series, columns: List[str]) -> Optional[str]:
        """从行中提取值"""
        for col in columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()
        return None
    
    def _add_node_if_not_exists(self, node_id: str, node_type: str, 
                               node_id_maps: Dict, stats: Dict):
        """如果节点不存在则添加"""
        if node_id not in node_id_maps[node_type]:
            node_id_maps[node_type][node_id] = len(node_id_maps[node_type])
            stats['nodes_added'] += 1
    
    def _parse_mitre_tactics(self, tactics_str: str) -> List[str]:
        """解析MITRE ATT&CK战术"""
        tactics = []
        if pd.isna(tactics_str) or tactics_str == '':
            return tactics
        
        # 解析JSON格式的战术
        import re
        tactics_matches = re.findall(r'TA\d{4}', str(tactics_str))
        tactics.extend(tactics_matches)
        
        return list(set(tactics))  # 去重
    
    def _parse_mitre_techniques(self, techniques_str: str) -> List[str]:
        """解析MITRE ATT&CK技术"""
        techniques = []
        if pd.isna(techniques_str) or techniques_str == '':
            return techniques
        
        # 解析JSON格式的技术
        import re
        techniques_matches = re.findall(r'T\d{4}(?:\.\d{3})?', str(techniques_str))
        techniques.extend(techniques_matches)
        
        return list(set(techniques))  # 去重
    
    def _normalize_timestamp(self, timestamp: Any) -> Optional[str]:
        """标准化时间戳"""
        try:
            if isinstance(timestamp, str):
                if '@' in timestamp:
                    clean_timestamp = timestamp.replace('@', '').strip()
                    dt = pd.to_datetime(clean_timestamp, errors='coerce')
                    if pd.notna(dt):
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                
                dt = pd.to_datetime(timestamp, errors='coerce')
                if pd.notna(dt):
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                    
            elif isinstance(timestamp, (pd.Timestamp, datetime)):
                return timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
            elif isinstance(timestamp, (int, float)):
                if timestamp > 1000000000:
                    dt = pd.to_datetime(timestamp, unit='s')
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                elif timestamp > 1000000000000:
                    dt = pd.to_datetime(timestamp, unit='ms')
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                    
        except Exception as e:
            self.logger.warning(f"时间戳标准化失败: {e}")
        return None
    
    def _build_edge_indices(self, node_id_maps: Dict, df: pd.DataFrame) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """构建边索引"""
        edge_index_dict = {}
        
        # 初始化边列表
        for edge_type in self.edge_types:
            edge_index_dict[edge_type] = ([], [])
        
        # 添加额外的边类型（在构建过程中可能出现的）
        additional_edge_types = [
            ('process', 'accesses', 'file'),  # 移除重复的边类型
            ('process', 'communicates_with', 'domain'),
            ('ip', 'resolves_to', 'domain')
        ]
        
        for edge_type in additional_edge_types:
            if edge_type not in edge_index_dict:
                edge_index_dict[edge_type] = ([], [])
        
        # 处理数据构建边
        for global_idx, row in df.iterrows():
            alert_id = f"alert_{global_idx}"
            
            # 构建警报相关的边
            self._build_alert_edges(row, alert_id, node_id_maps, edge_index_dict)
        
        # 转换为张量
        for edge_type in edge_index_dict:
            if len(edge_index_dict[edge_type][0]) > 0:
                edge_index_dict[edge_type] = torch.tensor([
                    edge_index_dict[edge_type][0],
                    edge_index_dict[edge_type][1]
                ], dtype=torch.long)
            else:
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index_dict
    
    def _build_alert_edges(self, row: pd.Series, alert_id: str, 
                          node_id_maps: Dict, edge_index_dict: Dict):
        """构建警报相关的边"""
        
        # 警报-规则边
        rule_id = self._extract_value(row, ['_source.rule.id'])
        if rule_id and rule_id in node_id_maps['rule']:
            edge_type = ('alert', 'triggered_by', 'rule')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['rule'][rule_id]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-主机边
        host_name = self._extract_value(row, ['_source.agent.name', '_source.manager.name'])
        if host_name and host_name in node_id_maps['host']:
            edge_type = ('alert', 'detected_on', 'host')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['host'][host_name]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-代理边
        agent_id = self._extract_value(row, ['_source.agent.id'])
        if agent_id and agent_id in node_id_maps['agent']:
            edge_type = ('alert', 'reported_by', 'agent')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['agent'][agent_id]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-文件边
        file_path = self._extract_value(row, ['_source.syscheck.path', '_source.data.file', '_source.data.sca.check.file'])
        if file_path and file_path in node_id_maps['file']:
            edge_type = ('alert', 'involves', 'file')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['file'][file_path]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-命令边
        command = self._extract_value(row, ['_source.data.command', '_source.data.sca.check.command'])
        if command and command in node_id_maps['command']:
            edge_type = ('alert', 'executed', 'command')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['command'][command]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-用户边
        user = self._extract_value(row, ['_source.data.srcuser', '_source.data.dstuser', '_source.data.uid'])
        if user and user in node_id_maps['user']:
            edge_type = ('alert', 'by_user', 'user')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['user'][user]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-进程边
        process = self._extract_value(row, ['_source.data.process', '_source.predecoder.program_name'])
        if process and process in node_id_maps['process']:
            edge_type = ('alert', 'involves_process', 'process')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['process'][process]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-IP边
        src_ip = self._extract_value(row, ['_source.data.srcip', '_source.agent.ip'])
        if src_ip and src_ip in node_id_maps['ip']:
            edge_type = ('alert', 'connects_to', 'ip')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['ip'][src_ip]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-域名边
        domain = self._extract_value(row, ['_source.data.url', '_source.data.hostname', '_source.predecoder.hostname'])
        if domain and domain in node_id_maps['domain']:
            edge_type = ('alert', 'connects_to', 'domain')
            src_idx = node_id_maps['alert'][alert_id]
            dst_idx = node_id_maps['domain'][domain]
            edge_index_dict[edge_type][0].append(src_idx)
            edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-端口边
        port = self._extract_value(row, ['_source.data.dstport', '_source.data.srcport'])
        if port:
            port_str = f"port_{port}"
            if port_str in node_id_maps['port']:
                edge_type = ('alert', 'uses_port', 'port')
                src_idx = node_id_maps['alert'][alert_id]
                dst_idx = node_id_maps['port'][port_str]
                edge_index_dict[edge_type][0].append(src_idx)
                edge_index_dict[edge_type][1].append(dst_idx)
        
        # 警报-时间戳边
        timestamp = self._extract_value(row, ['processed_timestamp', '_source.@timestamp', '_source.timestamp', '_source.predecoder.timestamp'])
        if timestamp:
            timestamp_str = self._normalize_timestamp(timestamp)
            if timestamp_str and timestamp_str in node_id_maps['timestamp']:
                edge_type = ('alert', 'at_time', 'timestamp')
                src_idx = node_id_maps['alert'][alert_id]
                dst_idx = node_id_maps['timestamp'][timestamp_str]
                edge_index_dict[edge_type][0].append(src_idx)
                edge_index_dict[edge_type][1].append(dst_idx)
    
    def _add_node_features(self, data: HeteroData, node_id_maps: Dict):
        """为异构图添加节点特征"""
        for node_type, id_map in node_id_maps.items():
            num_nodes = len(id_map)
            
            if num_nodes == 0:
                continue
            
            # 根据节点类型确定特征维度
            if node_type == 'alert':
                feature_dim = 16  # 警报特征：规则ID、严重性、时间等
            elif node_type == 'command':
                feature_dim = 16  # 命令特征：命令类型、参数、执行时间等
            elif node_type == 'file':
                feature_dim = 24  # 文件特征：路径、大小、权限、修改时间等
            elif node_type == 'process':
                feature_dim = 20  # 进程特征：PID、PPID、命令行、创建时间等
            elif node_type == 'user':
                feature_dim = 12  # 用户特征：用户名、权限、登录时间等
            elif node_type == 'host':
                feature_dim = 16  # 主机特征：IP、操作系统、主机名等
            elif node_type == 'ip':
                feature_dim = 8   # IP特征：地址、地理位置、威胁情报等
            elif node_type == 'domain':
                feature_dim = 12  # 域名特征：域名长度、子域名数量、威胁情报等
            elif node_type == 'port':
                feature_dim = 8   # 端口特征：端口号、协议、服务类型等
            elif node_type == 'rule':
                feature_dim = 16  # 规则特征：规则ID、类型、严重性等
            elif node_type == 'service':
                feature_dim = 10  # 服务特征：服务名、状态、启动类型等
            else:
                feature_dim = 16  # 默认特征维度
            
            # 生成基于节点类型和ID的特征
            if hasattr(self.config, 'use_sparse_features') and self.config.use_sparse_features:
                if hasattr(self.config, 'feature_seed'):
                    torch.manual_seed(self.config.feature_seed)
                
                # 使用更复杂的特征生成方法
                features = torch.zeros(num_nodes, feature_dim)
                for i in range(num_nodes):
                    # 使用节点类型和索引生成特征
                    node_id = f"{node_type}_{i}"
                    hash_val = hash(node_id) % (2**32)
                    
                    # 生成不同类型的特征
                    if node_type == 'alert':
                        # 警报特征：规则ID、严重性、时间等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 规则ID
                        features[i, 1] = (hash_val >> 8) & 0x0F  # 严重性
                        features[i, 2] = (hash_val >> 12) & 0xFF  # 时间特征
                        features[i, 3] = (hash_val >> 20) & 0xFF  # 其他特征
                    elif node_type == 'command':
                        # 命令特征：命令类型、参数等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 命令类型
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 参数特征
                        features[i, 2] = (hash_val >> 16) & 0xFF  # 执行特征
                    elif node_type == 'file':
                        # 文件特征：路径、大小、权限等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 路径特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 大小特征
                        features[i, 2] = (hash_val >> 16) & 0xFF  # 权限特征
                    elif node_type == 'process':
                        # 进程特征：PID、PPID、命令行等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # PID特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # PPID特征
                        features[i, 2] = (hash_val >> 16) & 0xFF  # 命令行特征
                    elif node_type == 'user':
                        # 用户特征：用户名、权限等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 用户名特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 权限特征
                    elif node_type == 'host':
                        # 主机特征：IP、操作系统等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # IP特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 操作系统特征
                    elif node_type == 'ip':
                        # IP特征：地址、地理位置等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 地址特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 地理位置特征
                    elif node_type == 'domain':
                        # 域名特征：域名长度、子域名数量等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 域名长度特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 子域名特征
                    elif node_type == 'port':
                        # 端口特征：端口号、协议等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 端口号特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 协议特征
                    elif node_type == 'rule':
                        # 规则特征：规则ID、类型等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 规则ID特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 规则类型特征
                    elif node_type == 'service':
                        # 服务特征：服务名、状态等
                        features[i, 0] = (hash_val >> 0) & 0xFF  # 服务名特征
                        features[i, 1] = (hash_val >> 8) & 0xFF  # 状态特征
                    else:
                        # 默认特征生成
                        for j in range(min(feature_dim, 32)):
                            features[i, j] = (hash_val >> j) & 1
                    
                    # 填充剩余特征
                    for j in range(feature_dim):
                        if features[i, j] == 0:
                            features[i, j] = (hash_val >> (j % 32)) & 1
            else:
                # 使用单位矩阵作为特征
                features = torch.eye(num_nodes)
                if features.size(1) > feature_dim:
                    features = features[:, :feature_dim]
                elif features.size(1) < feature_dim:
                    # 扩展特征维度
                    extra_features = torch.zeros(num_nodes, feature_dim - features.size(1))
                    features = torch.cat([features, extra_features], dim=1)
            
            data[node_type].x = features
            self.logger.info(f"为节点类型 {node_type} 添加了 {num_nodes} 个节点的特征，"
                           f"特征维度: {features.shape[1]}")
    
    def _extract_timestamps_from_data(self, data: HeteroData) -> Optional[torch.Tensor]:
        """
        从原始数据中提取时间戳信息
        
        Args:
            data: 异构图数据
            
        Returns:
            时间戳张量或None
        """
        try:
            # 尝试从alert节点的时间戳特征中提取
            if 'alert' in data.node_types and hasattr(data['alert'], 'x'):
                alert_features = data['alert'].x
                # 假设时间戳特征在特定位置（需要根据实际特征排列调整）
                if alert_features.size(1) > 0:
                    # 使用第一个特征作为时间戳（这里需要根据实际特征排列调整）
                    timestamps = alert_features[:, 0]  # 假设第一个特征是时间戳
                    if timestamps.numel() > 0:
                        return timestamps
            
            # 如果没有找到时间戳，返回None
            return None
        except Exception as e:
            self.logger.warning(f"提取时间戳失败: {e}")
            return None
    
    def create_temporal_snapshots(self, data: HeteroData) -> List[HeteroData]:
        """
        创建时序快照
        
        Args:
            data: 异构图数据
            
        Returns:
            时序快照列表
        """
        self.logger.info("创建时序快照")
        
        # 检查是否有时间戳信息
        if not hasattr(data, 'timestamps') or data.timestamps is None:
            # 尝试从原始数据中提取时间戳信息
            timestamps = self._extract_timestamps_from_data(data)
            if timestamps is not None:
                data.timestamps = timestamps
                self.logger.info(f"从原始数据中提取了 {len(timestamps)} 个时间戳")
            else:
                self.logger.warning("没有时间戳信息，返回原始数据作为单个快照")
                return [data]
        
        timestamps = data.timestamps
        if timestamps.numel() == 0:
            self.logger.warning("时间戳为空，返回原始数据作为单个快照")
            return [data]
        
        # 将时间戳转换为数值（Unix时间戳）
        if isinstance(timestamps, torch.Tensor):
            # 如果已经是数值，直接使用
            if timestamps.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                numeric_timestamps = timestamps.float()
            else:
                # 如果是字符串或其他类型，尝试转换
                numeric_timestamps = torch.zeros_like(timestamps, dtype=torch.float32)
                for i, ts in enumerate(timestamps):
                    try:
                        if isinstance(ts, str):
                            dt = pd.to_datetime(ts)
                            numeric_timestamps[i] = dt.timestamp()
                        elif hasattr(ts, 'timestamp'):
                            numeric_timestamps[i] = ts.timestamp()
                        else:
                            numeric_timestamps[i] = float(ts)
                    except:
                        numeric_timestamps[i] = 0.0
        else:
            # 处理pandas Series或其他类型
            numeric_timestamps = torch.tensor([
                ts.timestamp() if hasattr(ts, 'timestamp') else float(ts) 
                for ts in timestamps
            ], dtype=torch.float32)
        
        # 过滤无效时间戳
        valid_mask = numeric_timestamps > 0
        if valid_mask.sum() == 0:
            self.logger.warning("没有有效时间戳，返回原始数据作为单个快照")
            return [data]
        
        numeric_timestamps = numeric_timestamps[valid_mask]
        
        # 计算时间范围
        min_time = numeric_timestamps.min().item()
        max_time = numeric_timestamps.max().item()
        time_range = max_time - min_time
        
        if time_range <= 0:
            self.logger.warning("时间范围为0，返回原始数据作为单个快照")
            return [data]
        
        # 计算快照数量（基于配置和实际时间范围）
        max_snapshots = min(30, getattr(self.config, 'num_snapshots', 15))  # 最多30个快照
        time_window = max(900, time_range / max_snapshots)  # 至少15分钟的时间窗口
        num_snapshots = max(1, min(max_snapshots, int(time_range / time_window)))
        
        # 确保至少生成多个快照
        if num_snapshots < 10 and time_range > 3600:  # 如果时间范围超过1小时但快照少于10个
            # 按更小的时间窗口分割
            if time_range > 86400:  # 超过1天
                num_snapshots = min(20, max(10, int(time_range / 3600)))  # 按小时分割
            else:  # 1小时到1天之间
                num_snapshots = min(15, max(10, int(time_range / 1800)))  # 按30分钟分割
        
        self.logger.info(f"将创建 {num_snapshots} 个时序快照，时间窗口: {time_window:.0f}秒")
        
        # 创建时间窗口边界
        time_boundaries = torch.linspace(min_time, max_time, num_snapshots + 1)
        snapshots = []
        
        for i in range(num_snapshots):
            start_time = time_boundaries[i]
            end_time = time_boundaries[i + 1]
            
            # 找到在当前时间窗口内的节点
            mask = (numeric_timestamps >= start_time) & (numeric_timestamps < end_time)
            
            if mask.sum() == 0:
                continue
            
            # 创建快照
            snapshot = HeteroData()
            
            # 复制节点特征（不进行时间窗口过滤，因为节点特征与时间戳不是一一对应的）
            for ntype in data.node_types:
                if hasattr(data[ntype], 'x') and data[ntype].x is not None:
                    snapshot[ntype].x = data[ntype].x.clone()
            
            # 复制边信息（简化处理，不进行时间窗口过滤）
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_index'):
                    edge_index = data[edge_type].edge_index
                    if edge_index.numel() > 0:
                        snapshot[edge_type].edge_index = edge_index.clone()
                        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                            snapshot[edge_type].edge_attr = data[edge_type].edge_attr.clone()
            
            # 添加时间戳信息
            snapshot.timestamps = numeric_timestamps[mask]
            
            # 添加时间窗口信息
            snapshot.time_window = (start_time, end_time)
            snapshot.snapshot_id = i
            
            snapshots.append(snapshot)
        
        self.logger.info(f"成功创建 {len(snapshots)} 个时序快照")
        return snapshots
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据信息"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        return info
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def clear_cache(self):
        """清理缓存"""
        self._data_cache = None
        self._processed_data = None
        gc.collect()
        self.logger.info("数据缓存已清理")
