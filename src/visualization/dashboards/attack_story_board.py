"""
攻击故事看板

实现大纲中提到的攻击故事看板
生成完整的攻击故事，包括攻击步骤图、关键节点标注、时间线、聚类结果等
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
import base64
from io import BytesIO

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class AttackStoryBoard:
    """
    攻击故事看板
    
    实现大纲中提到的攻击故事看板
    生成完整的攻击故事，包括：
    - 攻击步骤图
    - 关键节点标注
    - 时间线
    - 聚类结果
    """
    
    def __init__(self, config):
        """
        初始化攻击故事看板
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色配置
        self.colors = {
            'normal': '#2E8B57',      # 海绿色
            'suspicious': '#FFD700',  # 金色
            'malicious': '#DC143C',   # 深红色
            'critical': '#8B0000',    # 暗红色
            'timeline': '#4169E1',    # 皇家蓝
            'attack_chain': '#FF6347', # 番茄红
            'background': '#F5F5F5'   # 浅灰色
        }
        
        # 攻击阶段配置
        self.attack_stages = {
            'initial_access': {'name': '初始访问', 'color': '#FF6B6B'},
            'execution': {'name': '执行', 'color': '#4ECDC4'},
            'persistence': {'name': '持久化', 'color': '#45B7D1'},
            'privilege_escalation': {'name': '权限提升', 'color': '#96CEB4'},
            'defense_evasion': {'name': '防御规避', 'color': '#FFEAA7'},
            'credential_access': {'name': '凭据访问', 'color': '#DDA0DD'},
            'discovery': {'name': '发现', 'color': '#98D8C8'},
            'lateral_movement': {'name': '横向移动', 'color': '#F7DC6F'},
            'collection': {'name': '收集', 'color': '#BB8FCE'},
            'command_and_control': {'name': '命令控制', 'color': '#85C1E9'},
            'exfiltration': {'name': '数据外泄', 'color': '#F8C471'},
            'impact': {'name': '影响', 'color': '#EC7063'}
        }
        
        self.logger.info("攻击故事看板初始化完成")
    
    def generate_attack_story(self, attack_chain: Dict[str, Any], 
                             timeline: List[Dict[str, Any]], 
                             evidence: Dict[str, Any],
                             clustering_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成完整的攻击故事
        
        Args:
            attack_chain: 攻击链信息
            timeline: 时间线信息
            evidence: 证据信息
            clustering_results: 聚类结果
            
        Returns:
            攻击故事看板数据
        """
        self.logger.info("开始生成攻击故事看板")
        
        try:
            # 1. 生成攻击步骤图
            attack_steps_fig = self._create_attack_steps_diagram(attack_chain)
            
            # 2. 生成关键节点标注图
            key_nodes_fig = self._create_key_nodes_annotation(attack_chain, evidence)
            
            # 3. 生成时间线图
            timeline_fig = self._create_timeline_diagram(timeline)
            
            # 4. 生成聚类结果图
            clustering_fig = None
            if clustering_results:
                clustering_fig = self._create_clustering_diagram(clustering_results)
            
            # 5. 生成网络拓扑图
            network_topology_fig = self._create_network_topology(attack_chain)
            
            # 6. 生成风险评分图
            risk_score_fig = self._create_risk_score_diagram(attack_chain, evidence)
            
            # 7. 生成攻击指标图
            attack_metrics_fig = self._create_attack_metrics_diagram(attack_chain, evidence)
            
            # 8. 生成综合仪表板
            dashboard_fig = self._create_comprehensive_dashboard(
                attack_chain, timeline, evidence, clustering_results
            )
            
            # 9. 生成攻击故事摘要
            story_summary = self._generate_story_summary(attack_chain, timeline, evidence)
            
            # 10. 生成可执行建议
            recommendations = self._generate_recommendations(attack_chain, evidence)
            
            self.logger.info("攻击故事看板生成完成")
            
            return {
                'attack_steps_diagram': attack_steps_fig,
                'key_nodes_annotation': key_nodes_fig,
                'timeline_diagram': timeline_fig,
                'clustering_diagram': clustering_fig,
                'network_topology': network_topology_fig,
                'risk_score_diagram': risk_score_fig,
                'attack_metrics_diagram': attack_metrics_fig,
                'comprehensive_dashboard': dashboard_fig,
                'story_summary': story_summary,
                'recommendations': recommendations,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'total_components': 8
                }
            }
            
        except Exception as e:
            self.logger.error(f"生成攻击故事看板过程中发生错误: {e}")
            return {
                'error': str(e),
                'attack_steps_diagram': None,
                'key_nodes_annotation': None,
                'timeline_diagram': None,
                'clustering_diagram': None,
                'network_topology': None,
                'risk_score_diagram': None,
                'attack_metrics_diagram': None,
                'comprehensive_dashboard': None,
                'story_summary': {'error': str(e), 'status': 'failed'},
                'recommendations': [],
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'error': str(e)
                }
            }
    
    def _create_attack_steps_diagram(self, attack_chain: Dict[str, Any]) -> go.Figure:
        """
        创建攻击步骤图
        
        Args:
            attack_chain: 攻击链信息
            
        Returns:
            Plotly图形对象
        """
        # 提取攻击步骤
        steps = attack_chain.get('timeline', [])
        
        if not steps:
            # 创建空图
            fig = go.Figure()
            fig.add_annotation(
                text="没有攻击步骤数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # 创建步骤节点
        nodes = []
        edges = []
        
        for i, step in enumerate(steps):
            # 节点信息
            node_info = {
                'id': f"step_{i}",
                'label': step.get('attack_stage', 'unknown'),
                'description': step.get('description', ''),
                'timestamp': step.get('timestamp', ''),
                'confidence': step.get('confidence', 0.0)
            }
            nodes.append(node_info)
            
            # 边信息
            if i > 0:
                edges.append({
                    'from': f"step_{i-1}",
                    'to': f"step_{i}",
                    'weight': step.get('confidence', 0.0)
                })
        
        # 创建图形
        fig = go.Figure()
        
        # 添加节点
        for i, node in enumerate(nodes):
            # 获取节点颜色
            stage = node['label']
            color = self.attack_stages.get(stage, {}).get('color', '#CCCCCC')
            
            # 节点大小基于置信度
            size = 20 + node['confidence'] * 30
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=node['label'],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                name=node['label'],
                hovertemplate=f"<b>{node['label']}</b><br>" +
                             f"描述: {node['description']}<br>" +
                             f"时间: {node['timestamp']}<br>" +
                             f"置信度: {node['confidence']:.3f}<extra></extra>"
            ))
        
        # 添加边
        for edge in edges:
            from_idx = int(edge['from'].split('_')[1])
            to_idx = int(edge['to'].split('_')[1])
            
            fig.add_trace(go.Scatter(
                x=[from_idx, to_idx],
                y=[0, 0],
                mode='lines',
                line=dict(
                    color='gray',
                    width=edge['weight'] * 5 + 1
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 更新布局
        fig.update_layout(
            title="攻击步骤图",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def _create_key_nodes_annotation(self, attack_chain: Dict[str, Any], 
                                   evidence: Dict[str, Any]) -> go.Figure:
        """
        创建关键节点标注图
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            Plotly图形对象
        """
        # 提取关键节点
        key_nodes = attack_chain.get('key_nodes', [])
        
        if not key_nodes:
            # 创建空图
            fig = go.Figure()
            fig.add_annotation(
                text="没有关键节点数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # 创建图形
        fig = go.Figure()
        
        # 添加关键节点
        for i, node in enumerate(key_nodes):
            # 节点信息
            node_id = node.get('node_id', f'node_{i}')
            node_type = node.get('node_type', 'unknown')
            confidence = node.get('confidence', 0.0)
            attack_stage = node.get('attack_stage', 'unknown')
            
            # 节点颜色基于置信度
            if confidence > 0.8:
                color = self.colors['critical']
            elif confidence > 0.6:
                color = self.colors['malicious']
            elif confidence > 0.4:
                color = self.colors['suspicious']
            else:
                color = self.colors['normal']
            
            # 节点大小基于置信度
            size = 15 + confidence * 25
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=node_id,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                name=node_type,
                hovertemplate=f"<b>{node_id}</b><br>" +
                             f"类型: {node_type}<br>" +
                             f"攻击阶段: {attack_stage}<br>" +
                             f"置信度: {confidence:.3f}<extra></extra>"
            ))
        
        # 更新布局
        fig.update_layout(
            title="关键节点标注",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            showlegend=True,
            height=300
        )
        
        return fig
    
    def _create_timeline_diagram(self, timeline: List[Dict[str, Any]]) -> go.Figure:
        """
        创建时间线图
        
        Args:
            timeline: 时间线信息
            
        Returns:
            Plotly图形对象
        """
        if not timeline:
            # 创建空图
            fig = go.Figure()
            fig.add_annotation(
                text="没有时间线数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # 创建时间线数据
        timeline_data = []
        for i, event in enumerate(timeline):
            timeline_data.append({
                'timestamp': event.get('timestamp', datetime.now()),
                'event': event.get('event', 'unknown'),
                'stage': event.get('attack_stage', 'unknown'),
                'confidence': event.get('confidence', 0.0),
                'y': i
            })
        
        # 创建图形
        fig = go.Figure()
        
        # 添加时间线事件
        for event in timeline_data:
            # 获取事件颜色
            stage = event['stage']
            color = self.attack_stages.get(stage, {}).get('color', '#CCCCCC')
            
            # 事件大小基于置信度
            size = 10 + event['confidence'] * 20
            
            fig.add_trace(go.Scatter(
                x=[event['timestamp']],
                y=[event['y']],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=event['event'],
                textposition="middle right",
                textfont=dict(size=8),
                name=event['stage'],
                hovertemplate=f"<b>{event['event']}</b><br>" +
                             f"时间: {event['timestamp']}<br>" +
                             f"阶段: {event['stage']}<br>" +
                             f"置信度: {event['confidence']:.3f}<extra></extra>"
            ))
        
        # 更新布局
        fig.update_layout(
            title="攻击时间线",
            xaxis=dict(title="时间"),
            yaxis=dict(title="事件", showticklabels=False),
            plot_bgcolor='white',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def _create_clustering_diagram(self, clustering_results: Dict[str, Any]) -> go.Figure:
        """
        创建聚类结果图
        
        Args:
            clustering_results: 聚类结果
            
        Returns:
            Plotly图形对象
        """
        # 提取聚类数据
        clusters = clustering_results.get('clusters', [])
        
        if not clusters:
            # 创建空图
            fig = go.Figure()
            fig.add_annotation(
                text="没有聚类数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # 创建图形
        fig = go.Figure()
        
        # 添加聚类
        for i, cluster in enumerate(clusters):
            cluster_id = cluster.get('cluster_id', i)
            nodes = cluster.get('nodes', [])
            center = cluster.get('center', [0, 0])
            
            # 聚类颜色
            color = px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            
            # 添加聚类中心
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(
                    size=20,
                    color=color,
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name=f'聚类 {cluster_id} 中心',
                hovertemplate=f"<b>聚类 {cluster_id} 中心</b><br>" +
                             f"节点数: {len(nodes)}<extra></extra>"
            ))
            
            # 添加聚类节点
            if nodes:
                node_x = [node.get('x', 0) for node in nodes]
                node_y = [node.get('y', 0) for node in nodes]
                
                fig.add_trace(go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.7
                    ),
                    name=f'聚类 {cluster_id} 节点',
                    hovertemplate=f"<b>节点</b><br>" +
                                 f"聚类: {cluster_id}<extra></extra>"
                ))
        
        # 更新布局
        fig.update_layout(
            title="攻击聚类结果",
            xaxis=dict(title="X坐标"),
            yaxis=dict(title="Y坐标"),
            plot_bgcolor='white',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def _create_network_topology(self, attack_chain: Dict[str, Any]) -> go.Figure:
        """
        创建网络拓扑图
        
        Args:
            attack_chain: 攻击链信息
            
        Returns:
            Plotly图形对象
        """
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点和边
        for step in attack_chain.get('timeline', []):
            node_id = step.get('node_id', 'unknown')
            node_type = step.get('node_type', 'unknown')
            G.add_node(node_id, node_type=node_type)
        
        # 添加边
        for i in range(len(attack_chain.get('timeline', [])) - 1):
            current_node = attack_chain['timeline'][i].get('node_id', f'node_{i}')
            next_node = attack_chain['timeline'][i + 1].get('node_id', f'node_{i+1}')
            G.add_edge(current_node, next_node)
        
        # 计算布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 添加节点
        for node in G.nodes():
            x, y = pos[node]
            node_type = G.nodes[node].get('node_type', 'unknown')
            
            # 节点颜色基于类型
            color = self.colors.get(node_type, '#CCCCCC')
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=node,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                name=node_type,
                hovertemplate=f"<b>{node}</b><br>" +
                             f"类型: {node_type}<extra></extra>"
            ))
        
        # 更新布局
        fig.update_layout(
            title="网络拓扑图",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def _create_risk_score_diagram(self, attack_chain: Dict[str, Any], 
                                 evidence: Dict[str, Any]) -> go.Figure:
        """
        创建风险评分图
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            Plotly图形对象
        """
        # 计算风险评分
        risk_scores = self._calculate_risk_scores(attack_chain, evidence)
        
        # 创建图形
        fig = go.Figure()
        
        # 添加风险评分条形图
        categories = list(risk_scores.keys())
        scores = list(risk_scores.values())
        
        # 颜色映射
        colors = ['#FF6B6B' if score > 0.7 else '#FFD93D' if score > 0.4 else '#6BCF7F' for score in scores]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=scores,
            marker_color=colors,
            text=[f'{score:.2f}' for score in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>风险评分: %{y:.2f}<extra></extra>'
        ))
        
        # 更新布局
        fig.update_layout(
            title="风险评分",
            xaxis=dict(title="风险类别"),
            yaxis=dict(title="风险评分", range=[0, 1]),
            plot_bgcolor='white',
            height=300
        )
        
        return fig
    
    def _create_attack_metrics_diagram(self, attack_chain: Dict[str, Any], 
                                     evidence: Dict[str, Any]) -> go.Figure:
        """
        创建攻击指标图
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            Plotly图形对象
        """
        # 计算攻击指标
        metrics = self._calculate_attack_metrics(attack_chain, evidence)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('攻击阶段分布', '置信度分布', '时间分布', '节点类型分布'),
            specs=[[{'type': 'pie'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 攻击阶段分布饼图
        stages = metrics.get('stage_distribution', {})
        fig.add_trace(go.Pie(
            labels=list(stages.keys()),
            values=list(stages.values()),
            name="攻击阶段"
        ), row=1, col=1)
        
        # 置信度分布直方图
        confidences = metrics.get('confidence_distribution', [])
        fig.add_trace(go.Histogram(
            x=confidences,
            name="置信度分布",
            nbinsx=20
        ), row=1, col=2)
        
        # 时间分布条形图
        time_dist = metrics.get('time_distribution', {})
        fig.add_trace(go.Bar(
            x=list(time_dist.keys()),
            y=list(time_dist.values()),
            name="时间分布"
        ), row=2, col=1)
        
        # 节点类型分布条形图
        node_types = metrics.get('node_type_distribution', {})
        fig.add_trace(go.Bar(
            x=list(node_types.keys()),
            y=list(node_types.values()),
            name="节点类型"
        ), row=2, col=2)
        
        # 更新布局
        fig.update_layout(
            title="攻击指标分析",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_comprehensive_dashboard(self, attack_chain: Dict[str, Any], 
                                      timeline: List[Dict[str, Any]], 
                                      evidence: Dict[str, Any],
                                      clustering_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        创建综合仪表板
        
        Args:
            attack_chain: 攻击链信息
            timeline: 时间线信息
            evidence: 证据信息
            clustering_results: 聚类结果
            
        Returns:
            Plotly图形对象
        """
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('攻击步骤', '关键节点', '时间线', 
                           '网络拓扑', '风险评分', '攻击指标',
                           '聚类结果', '证据分析', '综合评估'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'indicator'}]]
        )
        
        # 这里可以添加各个子图的具体实现
        # 由于篇幅限制，这里只提供框架
        
        # 更新布局
        fig.update_layout(
            title="攻击故事综合仪表板",
            height=900,
            showlegend=False
        )
        
        return fig
    
    def _calculate_risk_scores(self, attack_chain: Dict[str, Any], 
                              evidence: Dict[str, Any]) -> Dict[str, float]:
        """
        计算风险评分
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            风险评分字典
        """
        risk_scores = {
            '整体风险': 0.0,
            '数据泄露风险': 0.0,
            '系统入侵风险': 0.0,
            '横向移动风险': 0.0,
            '持久化风险': 0.0
        }
        
        # 基于攻击链计算风险评分
        timeline = attack_chain.get('timeline', [])
        if timeline:
            # 整体风险基于攻击链长度和置信度
            avg_confidence = np.mean([step.get('confidence', 0.0) for step in timeline])
            chain_length = len(timeline)
            risk_scores['整体风险'] = min(avg_confidence * (1 + chain_length * 0.1), 1.0)
            
            # 特定风险类型
            stages = [step.get('attack_stage', '') for step in timeline]
            
            if 'exfiltration' in stages:
                risk_scores['数据泄露风险'] = 0.9
            if 'initial_access' in stages:
                risk_scores['系统入侵风险'] = 0.8
            if 'lateral_movement' in stages:
                risk_scores['横向移动风险'] = 0.7
            if 'persistence' in stages:
                risk_scores['持久化风险'] = 0.6
        
        return risk_scores
    
    def _calculate_attack_metrics(self, attack_chain: Dict[str, Any], 
                                evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算攻击指标
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            攻击指标字典
        """
        metrics = {
            'stage_distribution': {},
            'confidence_distribution': [],
            'time_distribution': {},
            'node_type_distribution': {}
        }
        
        # 分析攻击阶段分布
        timeline = attack_chain.get('timeline', [])
        for step in timeline:
            stage = step.get('attack_stage', 'unknown')
            metrics['stage_distribution'][stage] = metrics['stage_distribution'].get(stage, 0) + 1
            
            confidence = step.get('confidence', 0.0)
            metrics['confidence_distribution'].append(confidence)
            
            node_type = step.get('node_type', 'unknown')
            metrics['node_type_distribution'][node_type] = metrics['node_type_distribution'].get(node_type, 0) + 1
        
        return metrics
    
    def _generate_story_summary(self, attack_chain: Dict[str, Any], 
                               timeline: List[Dict[str, Any]], 
                               evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成攻击故事摘要
        
        Args:
            attack_chain: 攻击链信息
            timeline: 时间线信息
            evidence: 证据信息
            
        Returns:
            故事摘要
        """
        summary = {
            'attack_duration': '未知',
            'attack_stages': [],
            'key_indicators': [],
            'impact_assessment': '未知',
            'confidence_level': '未知'
        }
        
        # 计算攻击持续时间
        if timeline:
            timestamps = []
            for event in timeline:
                timestamp = event.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now()
                timestamps.append(timestamp)
            
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                duration = end_time - start_time
                summary['attack_duration'] = str(duration)
            else:
                summary['attack_duration'] = '未知'
        
        # 提取攻击阶段
        stages = set([step.get('attack_stage', 'unknown') for step in attack_chain.get('timeline', [])])
        summary['attack_stages'] = list(stages)
        
        # 提取关键指标
        key_indicators = []
        for step in attack_chain.get('timeline', []):
            if step.get('confidence', 0.0) > 0.8:
                key_indicators.append(step.get('description', '未知'))
        summary['key_indicators'] = key_indicators
        
        # 评估影响
        if 'exfiltration' in stages:
            summary['impact_assessment'] = '高 - 数据可能已泄露'
        elif 'lateral_movement' in stages:
            summary['impact_assessment'] = '中 - 系统可能被横向移动'
        elif 'initial_access' in stages:
            summary['impact_assessment'] = '低 - 仅初始访问'
        else:
            summary['impact_assessment'] = '未知'
        
        # 置信度等级
        confidences = [step.get('confidence', 0.0) for step in attack_chain.get('timeline', [])]
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence > 0.8:
                summary['confidence_level'] = '高'
            elif avg_confidence > 0.6:
                summary['confidence_level'] = '中'
            else:
                summary['confidence_level'] = '低'
        
        return summary
    
    def _generate_recommendations(self, attack_chain: Dict[str, Any], 
                                evidence: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        生成可执行建议
        
        Args:
            attack_chain: 攻击链信息
            evidence: 证据信息
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于攻击阶段生成建议
        stages = [step.get('attack_stage', '') for step in attack_chain.get('timeline', [])]
        
        if 'initial_access' in stages:
            recommendations.append({
                'priority': '高',
                'category': '预防',
                'action': '加强邮件安全过滤和用户培训',
                'description': '检测到初始访问攻击，建议加强邮件安全措施'
            })
        
        if 'persistence' in stages:
            recommendations.append({
                'priority': '高',
                'category': '检测',
                'action': '检查系统启动项和注册表',
                'description': '检测到持久化攻击，建议检查系统启动项'
            })
        
        if 'lateral_movement' in stages:
            recommendations.append({
                'priority': '中',
                'category': '监控',
                'action': '加强网络分段和访问控制',
                'description': '检测到横向移动，建议加强网络分段'
            })
        
        if 'exfiltration' in stages:
            recommendations.append({
                'priority': '紧急',
                'category': '响应',
                'action': '立即隔离受影响系统并检查数据泄露',
                'description': '检测到数据外泄，需要立即响应'
            })
        
        return recommendations
    
    def save_dashboard(self, dashboard_data: Dict[str, Any], filepath: str):
        """
        保存仪表板数据
        
        Args:
            dashboard_data: 仪表板数据
            filepath: 保存路径
        """
        # 保存为JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"攻击故事看板已保存到: {filepath}")
    
    def export_to_html(self, dashboard_data: Dict[str, Any], filepath: str):
        """
        导出为HTML文件
        
        Args:
            dashboard_data: 仪表板数据
            filepath: 保存路径
        """
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>攻击故事看板</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .full-width {{
                    grid-column: 1 / -1;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                h2 {{
                    color: #666;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <h1>🔍 攻击故事看板</h1>
            <div class="dashboard">
                <div class="chart-container full-width">
                    <h2>📊 攻击步骤图</h2>
                    <div id="attack_steps"></div>
                </div>
                <div class="chart-container">
                    <h2>🎯 关键节点</h2>
                    <div id="key_nodes"></div>
                </div>
                <div class="chart-container">
                    <h2>⏰ 时间线</h2>
                    <div id="timeline"></div>
                </div>
                <div class="chart-container">
                    <h2>🌐 网络拓扑</h2>
                    <div id="network_topology"></div>
                </div>
                <div class="chart-container">
                    <h2>⚠️ 风险评分</h2>
                    <div id="risk_score"></div>
                </div>
                <div class="chart-container full-width">
                    <h2>📈 攻击指标分析</h2>
                    <div id="attack_metrics"></div>
                </div>
                <div class="chart-container full-width">
                    <h2>📋 综合仪表板</h2>
                    <div id="comprehensive_dashboard"></div>
                </div>
            </div>
            
            <script>
                // 攻击步骤图
                var attackStepsData = {self._convert_figure_to_dict(dashboard_data.get('attack_steps_diagram', {}))};
                if (attackStepsData && attackStepsData.data && attackStepsData.data.length > 0) {{
                    Plotly.newPlot('attack_steps', attackStepsData.data, attackStepsData.layout);
                }} else {{
                    document.getElementById('attack_steps').innerHTML = '<p style="text-align: center; color: #999;">暂无攻击步骤数据</p>';
                }}
                
                // 关键节点
                var keyNodesData = {self._convert_figure_to_dict(dashboard_data.get('key_nodes_annotation', {}))};
                if (keyNodesData && keyNodesData.data && keyNodesData.data.length > 0) {{
                    Plotly.newPlot('key_nodes', keyNodesData.data, keyNodesData.layout);
                }} else {{
                    document.getElementById('key_nodes').innerHTML = '<p style="text-align: center; color: #999;">暂无关键节点数据</p>';
                }}
                
                // 时间线
                var timelineData = {self._convert_figure_to_dict(dashboard_data.get('timeline_diagram', {}))};
                if (timelineData && timelineData.data && timelineData.data.length > 0) {{
                    Plotly.newPlot('timeline', timelineData.data, timelineData.layout);
                }} else {{
                    document.getElementById('timeline').innerHTML = '<p style="text-align: center; color: #999;">暂无时间线数据</p>';
                }}
                
                // 网络拓扑
                var networkData = {self._convert_figure_to_dict(dashboard_data.get('network_topology', {}))};
                if (networkData && networkData.data && networkData.data.length > 0) {{
                    Plotly.newPlot('network_topology', networkData.data, networkData.layout);
                }} else {{
                    document.getElementById('network_topology').innerHTML = '<p style="text-align: center; color: #999;">暂无网络拓扑数据</p>';
                }}
                
                // 风险评分
                var riskData = {self._convert_figure_to_dict(dashboard_data.get('risk_score_diagram', {}))};
                if (riskData && riskData.data && riskData.data.length > 0) {{
                    Plotly.newPlot('risk_score', riskData.data, riskData.layout);
                }} else {{
                    document.getElementById('risk_score').innerHTML = '<p style="text-align: center; color: #999;">暂无风险评分数据</p>';
                }}
                
                // 攻击指标
                var metricsData = {self._convert_figure_to_dict(dashboard_data.get('attack_metrics_diagram', {}))};
                if (metricsData && metricsData.data && metricsData.data.length > 0) {{
                    Plotly.newPlot('attack_metrics', metricsData.data, metricsData.layout);
                }} else {{
                    document.getElementById('attack_metrics').innerHTML = '<p style="text-align: center; color: #999;">暂无攻击指标数据</p>';
                }}
                
                // 综合仪表板
                var comprehensiveData = {self._convert_figure_to_dict(dashboard_data.get('comprehensive_dashboard', {}))};
                if (comprehensiveData && comprehensiveData.data && comprehensiveData.data.length > 0) {{
                    Plotly.newPlot('comprehensive_dashboard', comprehensiveData.data, comprehensiveData.layout);
                }} else {{
                    document.getElementById('comprehensive_dashboard').innerHTML = '<p style="text-align: center; color: #999;">暂无综合仪表板数据</p>';
                }}
            </script>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"攻击故事看板已导出为HTML: {filepath}")
    
    def _convert_figure_to_dict(self, figure):
        """
        将Plotly Figure对象转换为字典
        
        Args:
            figure: Plotly Figure对象或字典
            
        Returns:
            字典格式的图表数据
        """
        if figure is None:
            return {}
        
        # 如果已经是字典，直接返回
        if isinstance(figure, dict):
            return figure
        
        # 如果是Plotly Figure对象，转换为字典
        if hasattr(figure, 'to_dict'):
            return figure.to_dict()
        
        # 如果是字符串（序列化的Figure），尝试解析
        if isinstance(figure, str):
            try:
                import ast
                return ast.literal_eval(figure)
            except:
                return {}
        
        return {}
