"""
Wazuh安全告警数据标注策略

基于rule.level + MITRE技术 + 关键词的智能标注
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

class WazuhLabelingStrategy:
    """Wazuh安全告警智能标注策略"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Wazuh规则等级映射（0-15，越高越危险）
        self.rule_level_mapping = {
            0: 0.0,   # 信息
            1: 0.1,   # 低风险
            2: 0.2,   # 低风险
            3: 0.3,   # 中低风险
            4: 0.4,   # 中风险
            5: 0.5,   # 中高风险
            6: 0.6,   # 高风险
            7: 0.7,   # 高风险
            8: 0.8,   # 很高风险
            9: 0.9,   # 很高风险
            10: 1.0,  # 极高风险
            11: 1.0,  # 极高风险
            12: 1.0,  # 极高风险
            13: 1.0,  # 极高风险
            14: 1.0,  # 极高风险
            15: 1.0   # 极高风险
        }
        
        # MITRE ATT&CK战术风险权重
        self.mitre_tactics_weights = {
            'TA0001': 0.9,  # Initial Access
            'TA0002': 0.8,  # Execution
            'TA0003': 0.9,  # Persistence
            'TA0004': 0.8,  # Privilege Escalation
            'TA0005': 0.7,  # Defense Evasion
            'TA0006': 0.8,  # Credential Access
            'TA0007': 0.7,  # Discovery
            'TA0008': 0.8,  # Lateral Movement
            'TA0009': 0.9,  # Collection
            'TA0010': 0.9,  # Exfiltration
            'TA0011': 0.9,  # Command and Control
            'TA0040': 0.8,  # Impact
        }
        
        # MITRE ATT&CK技术风险权重（高风险技术）
        self.high_risk_techniques = {
            'T1055': 1.0,  # Process Injection
            'T1059': 0.9,  # Command and Scripting Interpreter
            'T1071': 0.9,  # Application Layer Protocol
            'T1078': 0.8,  # Valid Accounts
            'T1083': 0.7,  # File and Directory Discovery
            'T1105': 0.9,  # Ingress Tool Transfer
            'T1200': 0.8,  # Hardware Additions
            'T1548': 0.9,  # Abuse Elevation Control Mechanism
            'T1566': 0.8,  # Phishing
            'T1573': 0.9,  # Encrypted Channel
        }
        
        # 高风险关键词模式
        self.high_risk_keywords = [
            # 恶意软件相关
            r'(?i)(malware|virus|trojan|backdoor|rootkit|keylogger)',
            r'(?i)(exploit|payload|shellcode|injection)',
            r'(?i)(botnet|command.*control|c2)',
            
            # 权限提升
            r'(?i)(privilege.*escalation|sudo|su|admin|root)',
            r'(?i)(elevation|escalation|privilege)',
            
            # 网络攻击
            r'(?i)(brute.*force|dictionary.*attack|password.*crack)',
            r'(?i)(port.*scan|network.*scan|reconnaissance)',
            r'(?i)(ddos|dos|flood|attack)',
            
            # 数据泄露
            r'(?i)(exfiltration|data.*theft|leak|breach)',
            r'(?i)(unauthorized.*access|illegal.*access)',
            
            # 系统破坏
            r'(?i)(system.*damage|destruction|wipe|format)',
            r'(?i)(registry.*modification|system.*modification)',
            
            # 可疑行为
            r'(?i)(suspicious|anomalous|unusual|abnormal)',
            r'(?i)(covert|stealth|hidden|concealed)',
        ]
        
        # 文件路径风险模式
        self.risky_file_paths = [
            r'(?i)(/tmp/|/var/tmp/|/dev/shm/)',  # 临时目录
            r'(?i)(/bin/|/sbin/|/usr/bin/)',     # 系统目录
            r'(?i)(\.exe|\.bat|\.cmd|\.scr)',    # 可执行文件
            r'(?i)(\.dll|\.so|\.dylib)',         # 动态库
        ]
        
        # 命令风险模式
        self.risky_commands = [
            r'(?i)(wget|curl|nc|netcat)',        # 网络工具
            r'(?i)(base64|encode|decode)',       # 编码工具
            r'(?i)(chmod|chown|chattr)',         # 权限修改
            r'(?i)(iptables|firewall)',          # 防火墙
            r'(?i)(crontab|cron)',              # 定时任务
            r'(?i)(ssh|scp|rsync)',             # 远程访问
        ]
    
    def calculate_rule_level_score(self, rule_level: Any) -> float:
        """计算规则等级分数"""
        try:
            if pd.isna(rule_level) or rule_level == '':
                return 0.0
            
            level = int(float(str(rule_level)))
            return self.rule_level_mapping.get(level, 0.0)
        except:
            return 0.0
    
    def calculate_mitre_score(self, mitre_tactics: Any, mitre_techniques: Any) -> float:
        """计算MITRE ATT&CK分数"""
        tactics_score = 0.0
        techniques_score = 0.0
        
        # 处理MITRE战术
        if not pd.isna(mitre_tactics) and mitre_tactics != '':
            tactics_str = str(mitre_tactics)
            # 解析JSON格式的战术
            tactics_matches = re.findall(r'TA\d{4}', tactics_str)
            if tactics_matches:
                tactics_score = max([self.mitre_tactics_weights.get(t, 0.5) for t in tactics_matches])
        
        # 处理MITRE技术
        if not pd.isna(mitre_techniques) and mitre_techniques != '':
            techniques_str = str(mitre_techniques)
            # 解析JSON格式的技术
            techniques_matches = re.findall(r'T\d{4}(?:\.\d{3})?', techniques_str)
            if techniques_matches:
                techniques_score = max([self.high_risk_techniques.get(t, 0.5) for t in techniques_matches])
        
        return max(tactics_score, techniques_score)
    
    def calculate_keyword_score(self, text_fields: List[str]) -> float:
        """计算关键词风险分数"""
        max_score = 0.0
        
        for text in text_fields:
            if pd.isna(text) or text == '':
                continue
                
            text_str = str(text).lower()
            
            # 检查高风险关键词
            for pattern in self.high_risk_keywords:
                if re.search(pattern, text_str):
                    max_score = max(max_score, 0.8)
                    break
            
            # 检查文件路径风险
            for pattern in self.risky_file_paths:
                if re.search(pattern, text_str):
                    max_score = max(max_score, 0.6)
                    break
            
            # 检查命令风险
            for pattern in self.risky_commands:
                if re.search(pattern, text_str):
                    max_score = max(max_score, 0.5)
                    break
        
        return max_score
    
    def calculate_contextual_score(self, row: pd.Series) -> float:
        """计算上下文风险分数"""
        contextual_score = 0.0
        
        # 检查规则描述
        if not pd.isna(row.get('_source.rule.description', '')):
            desc_score = self.calculate_keyword_score([row['_source.rule.description']])
            contextual_score = max(contextual_score, desc_score)
        
        # 检查文件路径
        if not pd.isna(row.get('_source.data.file', '')):
            file_score = self.calculate_keyword_score([row['_source.data.file']])
            contextual_score = max(contextual_score, file_score)
        
        # 检查命令
        if not pd.isna(row.get('_source.data.command', '')):
            cmd_score = self.calculate_keyword_score([row['_source.data.command']])
            contextual_score = max(contextual_score, cmd_score)
        
        # 检查完整日志
        if not pd.isna(row.get('_source.full_log', '')):
            log_score = self.calculate_keyword_score([row['_source.full_log']])
            contextual_score = max(contextual_score, log_score)
        
        return contextual_score
    
    def generate_malicious_score(self, row: pd.Series) -> float:
        """生成恶意分数 - 优化版本，提高正样本比例"""
        # 规则等级分数 (权重: 0.3)
        rule_score = self.calculate_rule_level_score(row.get('_source.rule.level', 0))
        
        # MITRE分数 (权重: 0.3)
        mitre_score = self.calculate_mitre_score(
            row.get('_source.rule.mitre_tactics', ''),
            row.get('_source.rule.mitre_techniques', '')
        )
        
        # 上下文分数 (权重: 0.4)
        contextual_score = self.calculate_contextual_score(row)
        
        # 加权计算最终分数
        final_score = (
            rule_score * 0.3 +
            mitre_score * 0.3 +
            contextual_score * 0.4
        )
        
        # 降低阈值，提高正样本比例
        # 原始阈值调整：0.4 -> 0.2
        if final_score >= 0.2:
            final_score = min(final_score * 1.2, 1.0)  # 放大分数
        
        return min(final_score, 1.0)  # 确保不超过1.0
    
    def generate_attack_classification(self, malicious_score: float) -> str:
        """生成攻击分类 - 优化版本，降低阈值"""
        if malicious_score >= 0.7:
            return "High_Risk_Attack"
        elif malicious_score >= 0.5:
            return "Medium_Risk_Attack"
        elif malicious_score >= 0.3:
            return "Low_Risk_Attack"
        elif malicious_score >= 0.15:  # 降低阈值从0.2到0.15
            return "Suspicious_Activity"
        else:
            return "Normal_Activity"
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理整个数据框"""
        self.logger.info(f"开始Wazuh智能标注，处理 {len(df)} 条记录")
        
        # 生成恶意分数
        malicious_scores = []
        attack_classifications = []
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                self.logger.info(f"处理进度: {idx}/{len(df)}")
            
            malicious_score = self.generate_malicious_score(row)
            attack_classification = self.generate_attack_classification(malicious_score)
            
            malicious_scores.append(malicious_score)
            attack_classifications.append(attack_classification)
        
        # 添加新列
        df['malicious_score'] = malicious_scores
        df['attack_classification'] = attack_classifications
        
        # 统计结果
        self.logger.info("标注完成，统计结果:")
        self.logger.info(f"恶意分数分布:")
        self.logger.info(f"  0.0-0.2: {sum(1 for s in malicious_scores if 0.0 <= s < 0.2)}")
        self.logger.info(f"  0.2-0.4: {sum(1 for s in malicious_scores if 0.2 <= s < 0.4)}")
        self.logger.info(f"  0.4-0.6: {sum(1 for s in malicious_scores if 0.4 <= s < 0.6)}")
        self.logger.info(f"  0.6-0.8: {sum(1 for s in malicious_scores if 0.6 <= s < 0.8)}")
        self.logger.info(f"  0.8-1.0: {sum(1 for s in malicious_scores if 0.8 <= s <= 1.0)}")
        
        attack_counts = pd.Series(attack_classifications).value_counts()
        self.logger.info(f"攻击分类分布:")
        for classification, count in attack_counts.items():
            self.logger.info(f"  {classification}: {count}")
        
        # 计算正样本比例
        positive_samples = sum(1 for s in malicious_scores if s >= 0.4)
        positive_ratio = positive_samples / len(malicious_scores)
        self.logger.info(f"正样本比例: {positive_ratio:.3f} ({positive_samples}/{len(malicious_scores)})")
        
        return df
