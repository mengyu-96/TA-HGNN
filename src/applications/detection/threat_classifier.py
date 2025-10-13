"""
威胁分类器

实现基于T-HGNN的威胁分类功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import json

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class ThreatClassifier:
    """
    威胁分类器
    
    实现基于T-HGNN的威胁分类功能
    """
    
    def __init__(self, model, config):
        """
        初始化威胁分类器
        
        Args:
            model: 训练好的T-HGNN模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 威胁分类参数
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        self.top_k = getattr(config, 'top_k', 3)
        
        # 威胁类别定义
        self.threat_categories = self._define_threat_categories()
        
        # 分类历史
        self.classification_history = []
        
    def _define_threat_categories(self) -> Dict[str, Dict[str, Any]]:
        """定义威胁类别"""
        return {
            'apt_attack': {
                'name': 'APT攻击',
                'description': '高级持续威胁攻击',
                'severity': 'high',
                'indicators': ['long_duration', 'multiple_stages', 'sophisticated_tactics'],
                'confidence_threshold': 0.8
            },
            'malware_infection': {
                'name': '恶意软件感染',
                'description': '系统被恶意软件感染',
                'severity': 'medium',
                'indicators': ['suspicious_execution', 'file_modification', 'network_communication'],
                'confidence_threshold': 0.7
            },
            'data_exfiltration': {
                'name': '数据外泄',
                'description': '敏感数据被非法获取',
                'severity': 'critical',
                'indicators': ['large_data_transfer', 'encrypted_communication', 'unusual_access'],
                'confidence_threshold': 0.9
            },
            'insider_threat': {
                'name': '内部威胁',
                'description': '内部人员恶意行为',
                'severity': 'high',
                'indicators': ['privilege_abuse', 'unusual_access_pattern', 'data_access'],
                'confidence_threshold': 0.8
            },
            'lateral_movement': {
                'name': '横向移动',
                'description': '攻击者在网络内横向移动',
                'severity': 'medium',
                'indicators': ['multiple_host_access', 'credential_abuse', 'network_scanning'],
                'confidence_threshold': 0.7
            },
            'ddos_attack': {
                'name': 'DDoS攻击',
                'description': '分布式拒绝服务攻击',
                'severity': 'medium',
                'indicators': ['high_traffic_volume', 'multiple_sources', 'service_unavailability'],
                'confidence_threshold': 0.6
            },
            'phishing_attack': {
                'name': '钓鱼攻击',
                'description': '通过虚假信息获取凭据',
                'severity': 'low',
                'indicators': ['suspicious_email', 'fake_website', 'credential_harvesting'],
                'confidence_threshold': 0.6
            },
            'normal_activity': {
                'name': '正常活动',
                'description': '正常的系统活动',
                'severity': 'none',
                'indicators': ['expected_behavior', 'authorized_access', 'normal_patterns'],
                'confidence_threshold': 0.5
            }
        }
    
    def classify_threats(self, hetero_data: HeteroData, 
                        embeddings: Dict[str, torch.Tensor],
                        suspicious_nodes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分类威胁
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            suspicious_nodes: 可疑节点信息
            
        Returns:
            威胁分类结果
        """
        self.logger.info("开始威胁分类")
        
        # 1. 获取节点预测
        predictions = self._get_node_predictions(hetero_data, embeddings)
        
        # 2. 分类威胁类型
        threat_classifications = self._classify_threat_types(predictions, suspicious_nodes)
        
        # 3. 计算威胁分数
        threat_scores = self._calculate_threat_scores(threat_classifications)
        
        # 4. 生成分类报告
        classification_report = self._generate_classification_report(
            threat_classifications, threat_scores
        )
        
        # 5. 更新分类历史
        self._update_classification_history(classification_report)
        
        self.logger.info(f"威胁分类完成，识别出 {len(threat_classifications)} 种威胁类型")
        
        return {
            'predictions': predictions,
            'threat_classifications': threat_classifications,
            'threat_scores': threat_scores,
            'classification_report': classification_report
        }
    
    def _get_node_predictions(self, hetero_data: HeteroData, 
                             embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取节点预测
        
        Args:
            hetero_data: 异构图数据
            embeddings: 节点嵌入
            
        Returns:
            节点预测结果
        """
        predictions = {}
        
        for ntype in hetero_data.node_types:
            if ntype in embeddings and hetero_data[ntype].x is not None:
                # 使用模型进行预测
                with torch.no_grad():
                    node_predictions = self.model.predict(hetero_data)
                    if ntype in node_predictions:
                        predictions[ntype] = node_predictions[ntype]
        
        return predictions
    
    def _classify_threat_types(self, predictions: Dict[str, torch.Tensor], 
                              suspicious_nodes: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        分类威胁类型
        
        Args:
            predictions: 节点预测结果
            suspicious_nodes: 可疑节点信息
            
        Returns:
            威胁分类结果列表
        """
        threat_classifications = []
        
        # 分析每种威胁类别
        for category_id, category_info in self.threat_categories.items():
            # 计算该威胁类别的置信度
            confidence = self._calculate_category_confidence(
                category_id, predictions, suspicious_nodes
            )
            
            if confidence > category_info['confidence_threshold']:
                classification = {
                    'category_id': category_id,
                    'category_name': category_info['name'],
                    'description': category_info['description'],
                    'severity': category_info['severity'],
                    'confidence': confidence,
                    'indicators': self._identify_indicators(category_id, predictions, suspicious_nodes),
                    'timestamp': datetime.now().isoformat()
                }
                threat_classifications.append(classification)
        
        # 按置信度排序
        threat_classifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        return threat_classifications
    
    def _calculate_category_confidence(self, category_id: str, 
                                     predictions: Dict[str, torch.Tensor], 
                                     suspicious_nodes: Optional[Dict[str, Any]] = None) -> float:
        """
        计算威胁类别置信度
        
        Args:
            category_id: 威胁类别ID
            predictions: 节点预测结果
            suspicious_nodes: 可疑节点信息
            
        Returns:
            置信度分数
        """
        # 基于预测结果计算置信度
        all_confidences = []
        
        for ntype, pred in predictions.items():
            if pred.dim() > 1:
                # 多分类情况
                confidences = F.softmax(pred, dim=-1)
                max_confidence = torch.max(confidences, dim=-1)[0]
            else:
                # 二分类情况
                confidences = torch.sigmoid(pred)
                max_confidence = confidences
            
            all_confidences.extend(max_confidence.tolist())
        
        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            
            # 基于威胁类别调整置信度
            category_info = self.threat_categories[category_id]
            adjusted_confidence = avg_confidence * self._get_category_weight(category_id)
            
            return min(adjusted_confidence, 1.0)
        
        return 0.0
    
    def _get_category_weight(self, category_id: str) -> float:
        """
        获取威胁类别权重
        
        Args:
            category_id: 威胁类别ID
            
        Returns:
            权重值
        """
        # 基于威胁严重性设置权重
        severity_weights = {
            'critical': 1.2,
            'high': 1.1,
            'medium': 1.0,
            'low': 0.9,
            'none': 0.5
        }
        
        category_info = self.threat_categories[category_id]
        severity = category_info['severity']
        
        return severity_weights.get(severity, 1.0)
    
    def _identify_indicators(self, category_id: str, 
                           predictions: Dict[str, torch.Tensor], 
                           suspicious_nodes: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        识别威胁指标
        
        Args:
            category_id: 威胁类别ID
            predictions: 节点预测结果
            suspicious_nodes: 可疑节点信息
            
        Returns:
            威胁指标列表
        """
        category_info = self.threat_categories[category_id]
        expected_indicators = category_info['indicators']
        
        # 基于预测结果和可疑节点识别指标
        detected_indicators = []
        
        # 简化实现：基于节点类型和预测分数识别指标
        for ntype, pred in predictions.items():
            if pred.dim() > 1:
                max_scores = torch.max(pred, dim=-1)[0]
            else:
                max_scores = torch.sigmoid(pred)
            
            high_confidence_indices = torch.where(max_scores > self.confidence_threshold)[0]
            
            if len(high_confidence_indices) > 0:
                # 基于节点类型推断指标
                if 'email' in ntype or 'phishing' in ntype:
                    if 'suspicious_email' in expected_indicators:
                        detected_indicators.append('suspicious_email')
                elif 'command' in ntype or 'execution' in ntype:
                    if 'suspicious_execution' in expected_indicators:
                        detected_indicators.append('suspicious_execution')
                elif 'network' in ntype or 'traffic' in ntype:
                    if 'high_traffic_volume' in expected_indicators:
                        detected_indicators.append('high_traffic_volume')
                elif 'data' in ntype or 'file' in ntype:
                    if 'large_data_transfer' in expected_indicators:
                        detected_indicators.append('large_data_transfer')
        
        return list(set(detected_indicators))
    
    def _calculate_threat_scores(self, threat_classifications: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算威胁分数
        
        Args:
            threat_classifications: 威胁分类结果
            
        Returns:
            威胁分数字典
        """
        threat_scores = {}
        
        for classification in threat_classifications:
            category_id = classification['category_id']
            confidence = classification['confidence']
            severity = classification['severity']
            
            # 基于严重性和置信度计算威胁分数
            severity_weights = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4,
                'none': 0.0
            }
            
            severity_weight = severity_weights.get(severity, 0.5)
            threat_score = confidence * severity_weight
            
            threat_scores[category_id] = threat_score
        
        return threat_scores
    
    def _generate_classification_report(self, threat_classifications: List[Dict[str, Any]], 
                                      threat_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        生成分类报告
        
        Args:
            threat_classifications: 威胁分类结果
            threat_scores: 威胁分数
            
        Returns:
            分类报告
        """
        # 计算总体威胁等级
        if threat_scores:
            max_threat_score = max(threat_scores.values())
            avg_threat_score = np.mean(list(threat_scores.values()))
        else:
            max_threat_score = 0.0
            avg_threat_score = 0.0
        
        # 确定威胁等级
        if max_threat_score > 0.8:
            threat_level = 'critical'
        elif max_threat_score > 0.6:
            threat_level = 'high'
        elif max_threat_score > 0.4:
            threat_level = 'medium'
        elif max_threat_score > 0.2:
            threat_level = 'low'
        else:
            threat_level = 'minimal'
        
        # 统计威胁类型分布
        threat_type_distribution = {}
        for classification in threat_classifications:
            threat_type = classification['category_name']
            threat_type_distribution[threat_type] = threat_type_distribution.get(threat_type, 0) + 1
        
        # 统计严重性分布
        severity_distribution = {}
        for classification in threat_classifications:
            severity = classification['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        report = {
            'classification_time': datetime.now().isoformat(),
            'summary': {
                'total_threat_types': len(threat_classifications),
                'threat_level': threat_level,
                'max_threat_score': max_threat_score,
                'avg_threat_score': avg_threat_score
            },
            'threat_classifications': threat_classifications,
            'threat_scores': threat_scores,
            'threat_type_distribution': threat_type_distribution,
            'severity_distribution': severity_distribution,
            'recommendations': self._generate_threat_recommendations(threat_classifications, threat_level)
        }
        
        return report
    
    def _generate_threat_recommendations(self, threat_classifications: List[Dict[str, Any]], 
                                       threat_level: str) -> List[str]:
        """
        生成威胁响应建议
        
        Args:
            threat_classifications: 威胁分类结果
            threat_level: 威胁等级
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于威胁等级生成通用建议
        if threat_level == 'critical':
            recommendations.extend([
                "立即启动应急响应程序",
                "隔离所有受影响系统",
                "通知高级管理层和安全团队",
                "收集和保存所有相关证据"
            ])
        elif threat_level == 'high':
            recommendations.extend([
                "立即采取防护措施",
                "加强监控和日志记录",
                "通知安全团队",
                "分析威胁影响范围"
            ])
        elif threat_level == 'medium':
            recommendations.extend([
                "密切监控威胁活动",
                "加强安全防护措施",
                "分析威胁模式",
                "更新安全策略"
            ])
        else:
            recommendations.extend([
                "继续监控系统状态",
                "保持安全更新",
                "定期检查安全日志"
            ])
        
        # 基于具体威胁类型生成特定建议
        for classification in threat_classifications:
            category_id = classification['category_id']
            
            if category_id == 'apt_attack':
                recommendations.append("检测到APT攻击，需要长期监控和深度分析")
            elif category_id == 'data_exfiltration':
                recommendations.append("检测到数据外泄，立即检查数据完整性")
            elif category_id == 'insider_threat':
                recommendations.append("检测到内部威胁，需要调查内部人员活动")
            elif category_id == 'lateral_movement':
                recommendations.append("检测到横向移动，检查网络分段和访问控制")
            elif category_id == 'ddos_attack':
                recommendations.append("检测到DDoS攻击，启动流量清洗和防护措施")
            elif category_id == 'phishing_attack':
                recommendations.append("检测到钓鱼攻击，加强用户教育和邮件过滤")
        
        return list(set(recommendations))  # 去重
    
    def _update_classification_history(self, classification_report: Dict[str, Any]):
        """
        更新分类历史
        
        Args:
            classification_report: 分类报告
        """
        self.classification_history.append(classification_report)
        
        # 保持历史记录在合理范围内
        if len(self.classification_history) > 100:
            self.classification_history = self.classification_history[-100:]
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """
        获取分类统计信息
        
        Returns:
            统计信息
        """
        if not self.classification_history:
            return {'total_classifications': 0, 'average_threat_score': 0.0}
        
        total_classifications = len(self.classification_history)
        threat_scores = [report['summary']['avg_threat_score'] for report in self.classification_history]
        average_threat_score = np.mean(threat_scores)
        
        # 统计威胁类型分布
        threat_type_distribution = {}
        for report in self.classification_history:
            for threat_type, count in report['threat_type_distribution'].items():
                threat_type_distribution[threat_type] = threat_type_distribution.get(threat_type, 0) + count
        
        # 统计威胁等级分布
        threat_level_distribution = {}
        for report in self.classification_history:
            threat_level = report['summary']['threat_level']
            threat_level_distribution[threat_level] = threat_level_distribution.get(threat_level, 0) + 1
        
        return {
            'total_classifications': total_classifications,
            'average_threat_score': average_threat_score,
            'threat_type_distribution': threat_type_distribution,
            'threat_level_distribution': threat_level_distribution,
            'recent_classifications': self.classification_history[-5:]  # 最近5次分类
        }
    
    def evaluate_classification_performance(self, true_labels: Dict[str, torch.Tensor], 
                                          predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        评估分类性能
        
        Args:
            true_labels: 真实标签
            predictions: 预测结果
            
        Returns:
            性能评估结果
        """
        performance_metrics = {}
        
        for ntype in true_labels.keys():
            if ntype in predictions:
                true = true_labels[ntype].cpu().numpy()
                pred = predictions[ntype].cpu().numpy()
                
                # 计算准确率
                if pred.ndim > 1:
                    pred_classes = np.argmax(pred, axis=1)
                else:
                    pred_classes = (pred > 0.5).astype(int)
                
                accuracy = np.mean(true == pred_classes)
                
                # 计算混淆矩阵
                cm = confusion_matrix(true, pred_classes)
                
                # 计算分类报告
                report = classification_report(true, pred_classes, output_dict=True)
                
                performance_metrics[ntype] = {
                    'accuracy': accuracy,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report
                }
        
        return performance_metrics
