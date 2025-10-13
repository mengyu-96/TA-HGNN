"""
实时监控仪表板

实现基于T-HGNN的实时威胁监控仪表板
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import threading
import time

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class RealTimeMonitor:
    """
    实时监控仪表板
    
    实现基于T-HGNN的实时威胁监控仪表板
    """
    
    def __init__(self, config):
        """
        初始化实时监控仪表板
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 监控配置
        self.update_interval = getattr(config, 'update_interval', 5)  # 秒
        self.max_data_points = getattr(config, 'max_data_points', 1000)
        self.alert_threshold = getattr(config, 'alert_threshold', 0.8)
        
        # 数据存储
        self.monitoring_data = {
            'timestamps': [],
            'threat_counts': [],
            'anomaly_scores': [],
            'attack_chains': [],
            'alerts': []
        }
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        self.logger.info(f"实时监控仪表板初始化完成，更新间隔: {self.update_interval}秒")
    
    def start_monitoring(self, model, data_loader):
        """
        开始实时监控
        
        Args:
            model: T-HGNN模型
            data_loader: 数据加载器
        """
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(model, data_loader)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("实时监控已启动")
    
    def stop_monitoring(self):
        """
        停止实时监控
        """
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("实时监控已停止")
    
    def _monitoring_loop(self, model, data_loader):
        """
        监控循环
        
        Args:
            model: T-HGNN模型
            data_loader: 数据加载器
        """
        while self.is_monitoring:
            try:
                # 获取最新数据
                latest_data = self._get_latest_data(data_loader)
                
                if latest_data is not None:
                    # 运行模型推理
                    results = self._run_model_inference(model, latest_data)
                    
                    # 更新监控数据
                    self._update_monitoring_data(results)
                    
                    # 检查告警
                    self._check_alerts(results)
                
                # 等待下次更新
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环中发生错误: {e}")
                time.sleep(self.update_interval)
    
    def _get_latest_data(self, data_loader):
        """
        获取最新数据
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            最新数据
        """
        try:
            # 从数据加载器获取最新的异构图数据
            if hasattr(data_loader, 'get_latest_snapshot'):
                latest_data = data_loader.get_latest_snapshot()
                return {
                    'timestamp': datetime.now(),
                    'hetero_data': latest_data,  # 返回最新的异构图数据
                    'node_count': sum(data.num_nodes for data in latest_data if hasattr(data, 'num_nodes')) if isinstance(latest_data, list) else latest_data.num_nodes,
                    'edge_count': sum(data.num_edges for data in latest_data if hasattr(data, 'num_edges')) if isinstance(latest_data, list) else latest_data.num_edges
                }
            else:
                # 如果数据加载器没有提供最新数据方法，尝试从当前批处理中获取
                if hasattr(data_loader, 'dataset') and len(data_loader.dataset) > 0:
                    latest_data = data_loader.dataset[-1]
                    return {
                        'timestamp': datetime.now(),
                        'data': latest_data,  # 返回最新条目的数据
                        'node_count': latest_data.num_nodes if hasattr(latest_data, 'num_nodes') else 0,
                        'edge_count': latest_data.num_edges if hasattr(latest_data, 'num_edges') else 0
                    }
                else:
                    self.logger.warning("无法获取最新数据")
                    return None
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {e}")
            return None
    
    def _run_model_inference(self, model, data):
        """
        运行模型推理
        
        Args:
            model: T-HGNN模型
            data: 输入数据
            
        Returns:
            推理结果
        """
        try:
            # 确保模型处于评估模式
            model.eval()
            
            with torch.no_grad():
                # 获取模型预测和嵌入
                predictions = model(data, return_embeddings=False)
                embeddings = model(data, return_embeddings=True)
                
                # 分析威胁情况
                threat_count = 0
                anomaly_scores = []
                attack_indicators = []
                
                # 从预测结果中提取威胁信息
                for ntype, pred in predictions.items():
                    if ntype in ['alert', 'process', 'file']:  # 主要威胁节点类型
                        # 计算恶意概率（假设最后一行为恶意概率）
                        if len(pred.shape) == 2 and pred.shape[1] > 1:
                            malicious_probs = torch.softmax(pred, dim=1)[:, 1]  # 假设第1列是恶意概率
                            threat_count += (malicious_probs > 0.5).sum().item()
                            anomaly_scores.extend(malicious_probs.cpu().numpy())
                            
                            # 收集高置信度的威胁指标
                            high_conf_indices = torch.where(malicious_probs > 0.7)[0]
                            for idx in high_conf_indices:
                                attack_indicators.append({
                                    'node_type': ntype,
                                    'node_id': idx.item(),
                                    'confidence': malicious_probs[idx].item()
                                })
                
                avg_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
                
                # 构建推理结果
                inference_result = {
                    'threat_count': threat_count,
                    'anomaly_score': float(avg_anomaly_score),
                    'attack_chains': attack_indicators[:5],  # 最多返回5个威胁指示器
                    'confidence': float(np.mean([indicator['confidence'] for indicator in attack_indicators])) if attack_indicators else 0.0,
                    'model_output_size': sum(pred.numel() for pred in predictions.values()),
                    'embedding_dim': sum(emb.shape[1] for emb in embeddings.values()) if embeddings else 0
                }
                
                return inference_result
                
        except Exception as e:
            self.logger.error(f"模型推理失败: {e}")
            return {
                'threat_count': 0,
                'anomaly_score': 0.0,
                'attack_chains': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _update_monitoring_data(self, results):
        """
        更新监控数据
        
        Args:
            results: 推理结果
        """
        if results is None:
            return
        
        current_time = datetime.now()
        
        # 添加新数据点
        self.monitoring_data['timestamps'].append(current_time)
        self.monitoring_data['threat_counts'].append(results.get('threat_count', 0))
        self.monitoring_data['anomaly_scores'].append(results.get('anomaly_score', 0.0))
        self.monitoring_data['attack_chains'].append(results.get('attack_chains', []))
        
        # 保持数据点数量在限制内
        if len(self.monitoring_data['timestamps']) > self.max_data_points:
            for key in self.monitoring_data:
                if key != 'alerts':
                    self.monitoring_data[key] = self.monitoring_data[key][-self.max_data_points:]
    
    def _check_alerts(self, results):
        """
        检查告警
        
        Args:
            results: 推理结果
        """
        if results is None:
            return
        
        # 检查异常分数告警
        anomaly_score = results.get('anomaly_score', 0.0)
        if anomaly_score > self.alert_threshold:
            alert = {
                'timestamp': datetime.now(),
                'type': 'anomaly_alert',
                'severity': 'high' if anomaly_score > 0.9 else 'medium',
                'message': f'检测到异常活动，分数: {anomaly_score:.3f}',
                'anomaly_score': anomaly_score
            }
            self.monitoring_data['alerts'].append(alert)
            self.logger.warning(f"异常告警: {alert['message']}")
        
        # 检查威胁数量告警
        threat_count = results.get('threat_count', 0)
        if threat_count > 5:
            alert = {
                'timestamp': datetime.now(),
                'type': 'threat_count_alert',
                'severity': 'high' if threat_count > 10 else 'medium',
                'message': f'检测到大量威胁活动，数量: {threat_count}',
                'threat_count': threat_count
            }
            self.monitoring_data['alerts'].append(alert)
            self.logger.warning(f"威胁数量告警: {alert['message']}")
    
    def create_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        创建监控仪表板
        
        Returns:
            监控仪表板数据
        """
        self.logger.info("创建实时监控仪表板")
        
        try:
            # 1. 实时威胁趋势图
            threat_trend = self._create_threat_trend_chart()
            
            # 2. 异常分数图
            anomaly_chart = self._create_anomaly_chart()
            
            # 3. 告警列表
            alerts_table = self._create_alerts_table()
            
            # 4. 系统状态
            system_status = self._create_system_status()
            
            # 5. 综合仪表板
            comprehensive_dashboard = self._create_comprehensive_monitoring_dashboard()
            
            return {
                'threat_trend': threat_trend,
                'anomaly_chart': anomaly_chart,
                'alerts_table': alerts_table,
                'system_status': system_status,
                'comprehensive_dashboard': comprehensive_dashboard,
                'monitoring_data': self.monitoring_data,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'is_monitoring': self.is_monitoring,
                    'update_interval': self.update_interval,
                    'total_data_points': len(self.monitoring_data['timestamps'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"创建监控仪表板失败: {e}")
            return {
                'error': str(e),
                'threat_trend': None,
                'anomaly_chart': None,
                'alerts_table': None,
                'system_status': None,
                'comprehensive_dashboard': None,
                'monitoring_data': self.monitoring_data,
                'metadata': {'error': str(e)}
            }
    
    def _create_threat_trend_chart(self) -> go.Figure:
        """
        创建威胁趋势图
        
        Returns:
            威胁趋势图
        """
        try:
            timestamps = self.monitoring_data['timestamps']
            threat_counts = self.monitoring_data['threat_counts']
            
            if not timestamps:
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=threat_counts,
                mode='lines+markers',
                name='威胁数量',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            # 添加告警阈值线
            fig.add_hline(
                y=5, 
                line_dash="dash", 
                line_color="orange",
                annotation_text="告警阈值"
            )
            
            fig.update_layout(
                title="实时威胁趋势",
                xaxis_title="时间",
                yaxis_title="威胁数量",
                height=300,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁趋势图失败: {e}")
            return go.Figure()
    
    def _create_anomaly_chart(self) -> go.Figure:
        """
        创建异常分数图
        
        Returns:
            异常分数图
        """
        try:
            timestamps = self.monitoring_data['timestamps']
            anomaly_scores = self.monitoring_data['anomaly_scores']
            
            if not timestamps:
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=anomaly_scores,
                mode='lines+markers',
                name='异常分数',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # 添加告警阈值线
            fig.add_hline(
                y=self.alert_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text="告警阈值"
            )
            
            fig.update_layout(
                title="实时异常分数",
                xaxis_title="时间",
                yaxis_title="异常分数",
                height=300,
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建异常分数图失败: {e}")
            return go.Figure()
    
    def _create_alerts_table(self) -> go.Figure:
        """
        创建告警表格
        
        Returns:
            告警表格
        """
        try:
            alerts = self.monitoring_data['alerts']
            
            if not alerts:
                return go.Figure()
            
            # 准备表格数据
            timestamps = [alert['timestamp'].strftime('%H:%M:%S') for alert in alerts[-10:]]  # 最近10条
            types = [alert['type'] for alert in alerts[-10:]]
            severities = [alert['severity'] for alert in alerts[-10:]]
            messages = [alert['message'] for alert in alerts[-10:]]
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['时间', '类型', '严重程度', '消息'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=[timestamps, types, severities, messages],
                    fill_color='white',
                    align='left'
                )
            )])
            
            fig.update_layout(
                title="最近告警",
                height=300
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建告警表格失败: {e}")
            return go.Figure()
    
    def _create_system_status(self) -> go.Figure:
        """
        创建系统状态图
        
        Returns:
            系统状态图
        """
        try:
            # 系统状态指标
            status_metrics = {
                '监控状态': '运行中' if self.is_monitoring else '已停止',
                '数据点数': len(self.monitoring_data['timestamps']),
                '告警数量': len(self.monitoring_data['alerts']),
                '更新间隔': f"{self.update_interval}秒",
                '最后更新': self.monitoring_data['timestamps'][-1].strftime('%H:%M:%S') if self.monitoring_data['timestamps'] else '无'
            }
            
            fig = go.Figure()
            
            # 创建状态指示器
            for i, (metric, value) in enumerate(status_metrics.items()):
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=value,
                    title={"text": metric},
                    number={'font': {'size': 20}}
                ))
            
            fig.update_layout(
                title="系统状态",
                height=200,
                grid=dict(rows=2, columns=3)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建系统状态图失败: {e}")
            return go.Figure()
    
    def _create_comprehensive_monitoring_dashboard(self) -> go.Figure:
        """
        创建综合监控仪表板
        
        Returns:
            综合监控仪表板
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('威胁趋势', '异常分数', '系统状态', '告警列表'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "indicator"}, {"type": "table"}]]
            )
            
            # 威胁趋势
            timestamps = self.monitoring_data['timestamps']
            threat_counts = self.monitoring_data['threat_counts']
            
            if timestamps:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=threat_counts,
                    mode='lines+markers',
                    name='威胁数量'
                ), row=1, col=1)
            
            # 异常分数
            anomaly_scores = self.monitoring_data['anomaly_scores']
            
            if timestamps:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=anomaly_scores,
                    mode='lines+markers',
                    name='异常分数'
                ), row=1, col=2)
            
            # 系统状态
            fig.add_trace(go.Indicator(
                mode="number",
                value=len(self.monitoring_data['timestamps']),
                title={"text": "数据点数"}
            ), row=2, col=1)
            
            fig.update_layout(
                title="综合实时监控仪表板",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建综合监控仪表板失败: {e}")
            return go.Figure()
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """
        获取监控统计信息
        
        Returns:
            统计信息
        """
        return {
            'is_monitoring': self.is_monitoring,
            'total_data_points': len(self.monitoring_data['timestamps']),
            'total_alerts': len(self.monitoring_data['alerts']),
            'update_interval': self.update_interval,
            'alert_threshold': self.alert_threshold,
            'last_update': self.monitoring_data['timestamps'][-1] if self.monitoring_data['timestamps'] else None
        }
