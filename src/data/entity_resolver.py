import logging
import pandas as pd
import numpy as np
import re
import ipaddress
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict
from fuzzywuzzy import fuzz, process

class EntityResolver:
    """
    实体解析器，用于解析和关联不同数据源中的实体
    
    功能：
    1. 实体标准化：将不同格式的实体标准化为统一格式
    2. 实体去重：识别并合并重复的实体
    3. 实体关联：关联不同数据源中的相同实体
    4. 实体丰富：使用外部数据源丰富实体信息
    """
    
    def __init__(self, entity_db: Dict[str, Dict] = None, similarity_threshold: float = 85.0):
        """
        初始化实体解析器
        
        Args:
            entity_db: 实体数据库，用于存储已知实体
            similarity_threshold: 实体匹配的相似度阈值
        """
        self.logger = logging.getLogger(__name__)
        self.entity_db = entity_db or {
            'host': {},      # 主机实体
            'user': {},      # 用户实体
            'process': {},   # 进程实体
            'file': {},      # 文件实体
            'ip': {},        # IP地址实体
            'domain': {},    # 域名实体
            'hash': {},      # 哈希值实体
            'url': {},       # URL实体
            'email': {}      # 邮箱实体
        }
        self.similarity_threshold = similarity_threshold
        
        # 实体ID计数器
        self.entity_counters = {entity_type: 0 for entity_type in self.entity_db.keys()}
        
        # 实体别名映射
        self.entity_aliases = {entity_type: {} for entity_type in self.entity_db.keys()}
        
        # 正则表达式模式
        self.patterns = {
            'ip': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'domain': re.compile(r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'md5': re.compile(r'^[a-fA-F0-9]{32}$'),
            'sha1': re.compile(r'^[a-fA-F0-9]{40}$'),
            'sha256': re.compile(r'^[a-fA-F0-9]{64}$'),
            'url': re.compile(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$')
        }
    
    def resolve_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析实体
        
        从数据中提取实体，标准化，去重，关联，并丰富实体信息
        
        Args:
            df: 原始DataFrame
            
        Returns:
            处理后的DataFrame，添加了实体ID和关联信息
        """
        self.logger.info(f"开始解析实体，数据集大小: {len(df)}")
        
        # 创建新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 初始化实体ID列
        for entity_type in self.entity_db.keys():
            result_df[f'{entity_type}_id'] = None
        
        # 解析主机实体
        if 'host' in result_df.columns:
            self._resolve_host_entities(result_df)
        
        # 解析用户实体
        if 'user' in result_df.columns:
            self._resolve_user_entities(result_df)
        
        # 解析进程实体
        if 'process' in result_df.columns:
            self._resolve_process_entities(result_df)
        
        # 解析文件实体
        if 'file' in result_df.columns:
            self._resolve_file_entities(result_df)
        
        # 解析IP地址实体
        for ip_col in ['ip', 'src_ip', 'dst_ip']:
            if ip_col in result_df.columns:
                self._resolve_ip_entities(result_df, ip_col)
        
        # 解析域名实体
        if 'domain' in result_df.columns:
            self._resolve_domain_entities(result_df)
        
        # 解析哈希值实体
        for hash_col in ['hash', 'md5', 'sha1', 'sha256']:
            if hash_col in result_df.columns:
                self._resolve_hash_entities(result_df, hash_col)
        
        # 解析URL实体
        if 'url' in result_df.columns:
            self._resolve_url_entities(result_df)
        
        # 解析邮箱实体
        if 'email' in result_df.columns:
            self._resolve_email_entities(result_df)
        
        # 添加实体关系
        result_df = self._add_entity_relationships(result_df)
        
        self.logger.info(f"实体解析完成，共解析{sum(self.entity_counters.values())}个实体")
        
        return result_df
    
    def _resolve_host_entities(self, df: pd.DataFrame) -> None:
        """解析主机实体"""
        for idx, host in enumerate(df['host']):
            if not isinstance(host, str) or not host:
                continue
            
            # 标准化主机名
            normalized_host = self._normalize_host(host)
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('host', normalized_host)
            
            # 设置实体ID
            df.at[idx, 'host_id'] = entity_id
    
    def _resolve_user_entities(self, df: pd.DataFrame) -> None:
        """解析用户实体"""
        for idx, user in enumerate(df['user']):
            if not isinstance(user, str) or not user:
                continue
            
            # 标准化用户名
            normalized_user = self._normalize_user(user)
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('user', normalized_user)
            
            # 设置实体ID
            df.at[idx, 'user_id'] = entity_id
    
    def _resolve_process_entities(self, df: pd.DataFrame) -> None:
        """解析进程实体"""
        for idx, process in enumerate(df['process']):
            if not isinstance(process, str) or not process:
                continue
            
            # 标准化进程名
            normalized_process = self._normalize_process(process)
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('process', normalized_process)
            
            # 设置实体ID
            df.at[idx, 'process_id'] = entity_id
    
    def _resolve_file_entities(self, df: pd.DataFrame) -> None:
        """解析文件实体"""
        for idx, file_path in enumerate(df['file']):
            if not isinstance(file_path, str) or not file_path:
                continue
            
            # 标准化文件路径
            normalized_file = self._normalize_file(file_path)
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('file', normalized_file)
            
            # 设置实体ID
            df.at[idx, 'file_id'] = entity_id
    
    def _resolve_ip_entities(self, df: pd.DataFrame, ip_col: str) -> None:
        """解析IP地址实体"""
        for idx, ip in enumerate(df[ip_col]):
            if not isinstance(ip, str) or not ip:
                continue
            
            # 标准化IP地址
            normalized_ip = self._normalize_ip(ip)
            if not normalized_ip:
                continue
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('ip', normalized_ip)
            
            # 设置实体ID
            df.at[idx, 'ip_id'] = entity_id
    
    def _resolve_domain_entities(self, df: pd.DataFrame) -> None:
        """解析域名实体"""
        for idx, domain in enumerate(df['domain']):
            if not isinstance(domain, str) or not domain:
                continue
            
            # 标准化域名
            normalized_domain = self._normalize_domain(domain)
            if not normalized_domain:
                continue
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('domain', normalized_domain)
            
            # 设置实体ID
            df.at[idx, 'domain_id'] = entity_id
    
    def _resolve_hash_entities(self, df: pd.DataFrame, hash_col: str) -> None:
        """解析哈希值实体"""
        for idx, hash_value in enumerate(df[hash_col]):
            if not isinstance(hash_value, str) or not hash_value:
                continue
            
            # 标准化哈希值
            normalized_hash = self._normalize_hash(hash_value)
            if not normalized_hash:
                continue
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('hash', normalized_hash)
            
            # 设置实体ID
            df.at[idx, 'hash_id'] = entity_id
    
    def _resolve_url_entities(self, df: pd.DataFrame) -> None:
        """解析URL实体"""
        for idx, url in enumerate(df['url']):
            if not isinstance(url, str) or not url:
                continue
            
            # 标准化URL
            normalized_url = self._normalize_url(url)
            if not normalized_url:
                continue
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('url', normalized_url)
            
            # 设置实体ID
            df.at[idx, 'url_id'] = entity_id
    
    def _resolve_email_entities(self, df: pd.DataFrame) -> None:
        """解析邮箱实体"""
        for idx, email in enumerate(df['email']):
            if not isinstance(email, str) or not email:
                continue
            
            # 标准化邮箱
            normalized_email = self._normalize_email(email)
            if not normalized_email:
                continue
            
            # 查找或创建实体
            entity_id = self._get_or_create_entity('email', normalized_email)
            
            # 设置实体ID
            df.at[idx, 'email_id'] = entity_id
    
    def _normalize_host(self, host: str) -> str:
        """标准化主机名"""
        # 转换为小写
        host = host.lower()
        
        # 移除域名后缀
        if '.' in host:
            parts = host.split('.')
            if len(parts) > 1 and all(len(part) > 0 for part in parts):
                # 检查是否是FQDN
                if len(parts) > 2 and parts[-1].isalpha() and len(parts[-1]) >= 2:
                    # 保留主机名部分
                    host = parts[0]
        
        # 移除特殊字符
        host = re.sub(r'[^a-z0-9\-_]', '', host)
        
        return host
    
    def _normalize_user(self, user: str) -> str:
        """标准化用户名"""
        # 转换为小写
        user = user.lower()
        
        # 移除域名前缀
        if '\\' in user:
            user = user.split('\\')[-1]
        
        # 移除特殊字符
        user = re.sub(r'[^a-z0-9\-_]', '', user)
        
        return user
    
    def _normalize_process(self, process: str) -> str:
        """标准化进程名"""
        # 提取进程名
        if '/' in process:
            process = process.split('/')[-1]
        elif '\\' in process:
            process = process.split('\\')[-1]
        
        # 移除参数
        if ' ' in process:
            process = process.split(' ')[0]
        
        # 转换为小写
        process = process.lower()
        
        return process
    
    def _normalize_file(self, file_path: str) -> str:
        """标准化文件路径"""
        # 转换为小写
        file_path = file_path.lower()
        
        # 标准化路径分隔符
        file_path = file_path.replace('\\', '/')
        
        # 移除多余的路径分隔符
        file_path = re.sub(r'/+', '/', file_path)
        
        # 移除结尾的路径分隔符
        file_path = file_path.rstrip('/')
        
        return file_path
    
    def _normalize_ip(self, ip: str) -> Optional[str]:
        """标准化IP地址"""
        # 检查是否是有效的IP地址
        if not self.patterns['ip'].match(ip):
            return None
        
        try:
            # 使用ipaddress模块验证IP地址
            ip_obj = ipaddress.ip_address(ip)
            return str(ip_obj)
        except:
            return None
    
    def _normalize_domain(self, domain: str) -> Optional[str]:
        """标准化域名"""
        # 转换为小写
        domain = domain.lower()
        
        # 移除协议前缀
        if '://' in domain:
            domain = domain.split('://', 1)[1]
        
        # 移除路径和查询参数
        if '/' in domain:
            domain = domain.split('/', 1)[0]
        
        # 移除端口号
        if ':' in domain:
            domain = domain.split(':', 1)[0]
        
        # 检查是否是有效的域名
        if not self.patterns['domain'].match(domain):
            return None
        
        return domain
    
    def _normalize_hash(self, hash_value: str) -> Optional[str]:
        """标准化哈希值"""
        # 转换为小写
        hash_value = hash_value.lower()
        
        # 移除空白字符
        hash_value = re.sub(r'\s', '', hash_value)
        
        # 检查哈希值类型
        if self.patterns['md5'].match(hash_value) or self.patterns['sha1'].match(hash_value) or self.patterns['sha256'].match(hash_value):
            return hash_value
        
        return None
    
    def _normalize_url(self, url: str) -> Optional[str]:
        """标准化URL"""
        # 转换为小写
        url = url.lower()
        
        # 移除结尾的斜杠
        url = url.rstrip('/')
        
        # 检查是否是有效的URL
        if not self.patterns['url'].match(url):
            return None
        
        return url
    
    def _normalize_email(self, email: str) -> Optional[str]:
        """标准化邮箱"""
        # 转换为小写
        email = email.lower()
        
        # 移除空白字符
        email = re.sub(r'\s', '', email)
        
        # 检查是否是有效的邮箱
        if not self.patterns['email'].match(email):
            return None
        
        return email
    
    def _get_or_create_entity(self, entity_type: str, entity_value: str) -> str:
        """
        获取或创建实体
        
        如果实体已存在，返回实体ID；否则创建新实体并返回ID
        """
        # 检查实体是否已存在
        if entity_value in self.entity_db[entity_type]:
            return self.entity_db[entity_type][entity_value]['id']
        
        # 检查是否有别名匹配
        if entity_value in self.entity_aliases[entity_type]:
            canonical_value = self.entity_aliases[entity_type][entity_value]
            return self.entity_db[entity_type][canonical_value]['id']
        
        # 检查是否有相似实体
        similar_entity = self._find_similar_entity(entity_type, entity_value)
        if similar_entity:
            # 添加别名
            self.entity_aliases[entity_type][entity_value] = similar_entity
            return self.entity_db[entity_type][similar_entity]['id']
        
        # 创建新实体
        entity_id = f"{entity_type}_{self.entity_counters[entity_type]}"
        self.entity_counters[entity_type] += 1
        
        # 添加到实体数据库
        self.entity_db[entity_type][entity_value] = {
            'id': entity_id,
            'value': entity_value,
            'type': entity_type,
            'aliases': [entity_value],
            'first_seen': pd.Timestamp.now(),
            'last_seen': pd.Timestamp.now(),
            'count': 1,
            'attributes': {}
        }
        
        return entity_id
    
    def _find_similar_entity(self, entity_type: str, entity_value: str) -> Optional[str]:
        """
        查找相似实体
        
        使用模糊匹配查找相似实体
        """
        # 如果实体数据库为空，直接返回None
        if not self.entity_db[entity_type]:
            return None
        
        # 使用模糊匹配查找相似实体
        choices = list(self.entity_db[entity_type].keys())
        best_match, score = process.extractOne(entity_value, choices)
        
        # 如果相似度超过阈值，返回最佳匹配
        if score >= self.similarity_threshold:
            return best_match
        
        return None
    
    def _add_entity_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加实体关系
        
        根据实体ID添加实体之间的关系
        """
        # 创建新的DataFrame，避免修改原始数据
        result_df = df.copy()
        
        # 初始化实体关系列
        result_df['entity_relationships'] = [[] for _ in range(len(result_df))]
        
        # 遍历每一行
        for idx in range(len(result_df)):
            relationships = []
            
            # 获取所有实体ID
            entity_ids = {}
            for entity_type in self.entity_db.keys():
                entity_id_col = f'{entity_type}_id'
                if entity_id_col in result_df.columns and pd.notna(result_df.at[idx, entity_id_col]):
                    entity_ids[entity_type] = result_df.at[idx, entity_id_col]
            
            # 添加实体关系
            for entity_type1, entity_id1 in entity_ids.items():
                for entity_type2, entity_id2 in entity_ids.items():
                    if entity_type1 != entity_type2:
                        relationships.append({
                            'source_type': entity_type1,
                            'source_id': entity_id1,
                            'target_type': entity_type2,
                            'target_id': entity_id2,
                            'relationship_type': 'associated_with',
                            'timestamp': result_df.at[idx, 'timestamp'] if 'timestamp' in result_df.columns else pd.Timestamp.now()
                        })
            
            # 设置实体关系
            result_df.at[idx, 'entity_relationships'] = relationships
        
        return result_df
    
    def export_entity_db(self, output_path: str) -> None:
        """
        导出实体数据库
        
        将实体数据库导出到文件
        """
        # 将实体数据库转换为DataFrame
        entities = []
        for entity_type, entities_dict in self.entity_db.items():
            for entity_value, entity_data in entities_dict.items():
                entities.append({
                    'id': entity_data['id'],
                    'type': entity_type,
                    'value': entity_value,
                    'aliases': entity_data['aliases'],
                    'first_seen': entity_data['first_seen'],
                    'last_seen': entity_data['last_seen'],
                    'count': entity_data['count'],
                    'attributes': entity_data['attributes']
                })
        
        # 创建DataFrame
        entity_df = pd.DataFrame(entities)
        
        # 导出到文件
        entity_df.to_csv(output_path, index=False)
        
        self.logger.info(f"实体数据库已导出到{output_path}，共{len(entity_df)}个实体")
    
    def import_entity_db(self, input_path: str) -> None:
        """
        导入实体数据库
        
        从文件导入实体数据库
        """
        # 从文件读取实体数据
        entity_df = pd.read_csv(input_path)
        
        # 重置实体数据库
        self.entity_db = {entity_type: {} for entity_type in self.entity_db.keys()}
        self.entity_aliases = {entity_type: {} for entity_type in self.entity_db.keys()}
        self.entity_counters = {entity_type: 0 for entity_type in self.entity_db.keys()}
        
        # 导入实体数据
        for _, row in entity_df.iterrows():
            entity_type = row['type']
            entity_value = row['value']
            entity_id = row['id']
            
            # 更新实体计数器
            entity_counter = int(entity_id.split('_')[-1]) + 1
            self.entity_counters[entity_type] = max(self.entity_counters[entity_type], entity_counter)
            
            # 添加到实体数据库
            self.entity_db[entity_type][entity_value] = {
                'id': entity_id,
                'value': entity_value,
                'type': entity_type,
                'aliases': eval(row['aliases']) if isinstance(row['aliases'], str) else [entity_value],
                'first_seen': pd.to_datetime(row['first_seen']),
                'last_seen': pd.to_datetime(row['last_seen']),
                'count': row['count'],
                'attributes': eval(row['attributes']) if isinstance(row['attributes'], str) else {}
            }
            
            # 添加别名
            for alias in self.entity_db[entity_type][entity_value]['aliases']:
                if alias != entity_value:
                    self.entity_aliases[entity_type][alias] = entity_value
        
        self.logger.info(f"实体数据库已从{input_path}导入，共{len(entity_df)}个实体")