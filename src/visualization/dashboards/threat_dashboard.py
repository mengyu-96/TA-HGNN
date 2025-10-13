"""
威胁仪表板

实现基于T-HGNN的威胁可视化仪表板
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

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


class ThreatDashboard:
    """
    威胁仪表板
    
    实现基于T-HGNN的威胁可视化仪表板
    """
    
    def __init__(self, config):
        """
        初始化威胁仪表板
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 仪表板配置
        self.theme = getattr(config, 'dashboard_theme', 'plotly_white')
        self.color_scheme = getattr(config, 'color_scheme', 'viridis')
        self.update_interval = getattr(config, 'update_interval', 30)  # 秒
        
        self.logger.info(f"威胁仪表板初始化完成，主题: {self.theme}")
    
    def create_threat_dashboard(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建威胁仪表板
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            仪表板数据
        """
        self.logger.info("开始创建威胁仪表板")
        
        try:
            # 1. 威胁概览图
            threat_overview = self._create_threat_overview(threat_data)
            
            # 2. 威胁趋势图
            threat_trends = self._create_threat_trends(threat_data)
            
            # 3. 威胁分布图
            threat_distribution = self._create_threat_distribution(threat_data)
            
            # 4. 威胁网络图
            threat_network = self._create_threat_network(threat_data)
            
            # 5. 威胁指标图
            threat_metrics = self._create_threat_metrics(threat_data)
            
            # 6. 综合仪表板
            comprehensive_dashboard = self._create_comprehensive_dashboard(threat_data)
            
            self.logger.info("威胁仪表板创建完成")
            
            return {
                'threat_overview': threat_overview,
                'threat_trends': threat_trends,
                'threat_distribution': threat_distribution,
                'threat_network': threat_network,
                'threat_metrics': threat_metrics,
                'comprehensive_dashboard': comprehensive_dashboard,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'total_components': 6
                }
            }
            
        except Exception as e:
            self.logger.error(f"创建威胁仪表板过程中发生错误: {e}")
            return {
                'error': str(e),
                'threat_overview': None,
                'threat_trends': None,
                'threat_distribution': None,
                'threat_network': None,
                'threat_metrics': None,
                'comprehensive_dashboard': None,
                'metadata': {'error': str(e)}
            }
    
    def _create_threat_overview(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建威胁概览图
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            威胁概览图
        """
        try:
            # 提取威胁统计信息
            stats = threat_data.get('threat_statistics', {})
            
            # 创建指标卡片
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('总威胁数', '高风险威胁', '威胁类型', '威胁状态'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # 总威胁数
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=stats.get('total_threats', 0),
                    title={"text": "总威胁数"},
                    number={'font': {'size': 40}}
                ),
                row=1, col=1
            )
            
            # 高风险威胁
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=stats.get('high_risk_threats', 0),
                    title={"text": "高风险威胁"},
                    number={'font': {'size': 40}}
                ),
                row=1, col=2
            )
            
            # 威胁类型分布
            threat_types = stats.get('threat_types', {})
            if threat_types:
                fig.add_trace(
                    go.Bar(
                        x=list(threat_types.keys()),
                        y=list(threat_types.values()),
                        name="威胁类型"
                    ),
                    row=2, col=1
                )
            
            # 威胁状态分布
            threat_status = stats.get('threat_status', {})
            if threat_status:
                fig.add_trace(
                    go.Pie(
                        labels=list(threat_status.keys()),
                        values=list(threat_status.values()),
                        name="威胁状态"
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="威胁概览仪表板",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁概览图失败: {e}")
            return go.Figure()
    
    def _create_threat_trends(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建威胁趋势图
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            威胁趋势图
        """
        try:
            # 提取时间序列数据
            timeline = threat_data.get('timeline', [])
            
            if not timeline:
                return go.Figure()
            
            # 创建时间序列图
            fig = go.Figure()
            
            # 威胁数量趋势
            timestamps = [item.get('timestamp', '') for item in timeline]
            threat_counts = [item.get('threat_count', 0) for item in timeline]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=threat_counts,
                mode='lines+markers',
                name='威胁数量',
                line=dict(color='red', width=2)
            ))
            
            # 高风险威胁趋势
            high_risk_counts = [item.get('high_risk_count', 0) for item in timeline]
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=high_risk_counts,
                mode='lines+markers',
                name='高风险威胁',
                line=dict(color='darkred', width=2)
            ))
            
            fig.update_layout(
                title="威胁趋势分析",
                xaxis_title="时间",
                yaxis_title="威胁数量",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁趋势图失败: {e}")
            return go.Figure()
    
    def _create_threat_distribution(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建威胁分布图
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            威胁分布图
        """
        try:
            # 提取威胁分布数据
            distribution = threat_data.get('threat_distribution', {})
            
            if not distribution:
                return go.Figure()
            
            # 创建热力图
            fig = go.Figure(data=go.Heatmap(
                z=distribution.get('heatmap_data', []),
                x=distribution.get('x_labels', []),
                y=distribution.get('y_labels', []),
                colorscale='Reds'
            ))
            
            fig.update_layout(
                title="威胁分布热力图",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁分布图失败: {e}")
            return go.Figure()
    
    def _create_threat_network(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建威胁网络图
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            威胁网络图
        """
        try:
            # 提取网络数据
            network_data = threat_data.get('threat_network', {})
            
            if not network_data:
                return go.Figure()
            
            # 创建网络图
            G = nx.Graph()
            
            # 添加节点和边
            nodes = network_data.get('nodes', [])
            edges = network_data.get('edges', [])
            
            for node in nodes:
                G.add_node(node['id'], **node.get('attributes', {}))
            
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], **edge.get('attributes', {}))
            
            # 计算布局
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 创建边轨迹
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 创建节点轨迹
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # 根据节点属性设置颜色
                node_attrs = G.nodes[node]
                if node_attrs.get('threat_level') == 'high':
                    node_colors.append('red')
                elif node_attrs.get('threat_level') == 'medium':
                    node_colors.append('orange')
                else:
                    node_colors.append('blue')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='black')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='威胁网络图',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="威胁节点网络",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color='black', size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=500
                          ))
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁网络图失败: {e}")
            return go.Figure()
    
    def _create_threat_metrics(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建威胁指标图
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            威胁指标图
        """
        try:
            # 提取指标数据
            metrics = threat_data.get('threat_metrics', {})
            
            if not metrics:
                return go.Figure()
            
            # 创建雷达图
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='威胁指标'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="威胁指标雷达图",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁指标图失败: {e}")
            return go.Figure()
    
    def _create_comprehensive_dashboard(self, threat_data: Dict[str, Any]) -> go.Figure:
        """
        创建综合仪表板
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            综合仪表板
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('威胁概览', '威胁趋势', '威胁分布', '威胁网络', '威胁指标', '威胁统计'),
                specs=[[{"type": "indicator"}, {"type": "scatter"}],
                       [{"type": "heatmap"}, {"type": "scatter"}],
                       [{"type": "scatterpolar"}, {"type": "bar"}]]
            )
            
            # 创建完整的综合威胁仪表板
            # 添加具体的图表内容
            
            # 1. 威胁概览热图
            threat_overview = self._create_threat_overview(threat_data)
            fig.add_trace(threat_overview.data[0], row=1, col=1)
            
            # 2. 威胁趋势图
            threat_trends = self._create_threat_trends(threat_data)
            fig.add_trace(threat_trends.data[0], row=1, col=2)
            
            # 3. 威胁分布图
            threat_distribution = self._create_threat_distribution(threat_data)
            fig.add_trace(threat_distribution.data[0], row=2, col=1)
            
            # 4. 威胁网络图
            threat_network = self._create_threat_network(threat_data)
            fig.add_trace(threat_network.data[0], row=2, col=2)
            
            # 5. 威胁指标图
            threat_metrics = self._create_threat_metrics(threat_data)
            fig.add_trace(threat_metrics.data[0], row=3, col=1)
            
            # 6. 威胁时间线
            threat_timeline = self._create_threat_timeline(threat_data)
            fig.add_trace(threat_timeline.data[0], row=3, col=2)
            
            fig.update_layout(
                title="综合威胁仪表板",
                height=1200,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建综合仪表板失败: {e}")
            return go.Figure()
    
    def _create_threat_timeline(self, threat_data: Dict[str, Any]) -> go.Figure:
        """创建威胁时间线图"""
        try:
            # 提取时间数据
            timeline_data = threat_data.get('timeline', [])
            
            if not timeline_data:
                # 创建示例时间线数据
                timeline_data = [
                    {'time': '2024-01-01', 'threat_level': 'Low', 'count': 5},
                    {'time': '2024-01-02', 'threat_level': 'Medium', 'count': 12},
                    {'time': '2024-01-03', 'threat_level': 'High', 'count': 8},
                    {'time': '2024-01-04', 'threat_level': 'Critical', 'count': 3},
                    {'time': '2024-01-05', 'threat_level': 'Medium', 'count': 15}
                ]
            
            # 按威胁级别分组
            threat_levels = ['Low', 'Medium', 'High', 'Critical']
            colors = ['green', 'yellow', 'orange', 'red']
            
            fig = go.Figure()
            
            for i, level in enumerate(threat_levels):
                level_data = [item for item in timeline_data if item.get('threat_level') == level]
                if level_data:
                    times = [item['time'] for item in level_data]
                    counts = [item['count'] for item in level_data]
                    
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=counts,
                        mode='lines+markers',
                        name=level,
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=8)
                    ))
            
            fig.update_layout(
                title="威胁时间线",
                xaxis_title="时间",
                yaxis_title="威胁数量",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建威胁时间线失败: {e}")
            return go.Figure()
    
    def export_dashboard(self, dashboard_data: Dict[str, Any], 
                        output_path: str, format: str = 'html') -> bool:
        """
        导出仪表板
        
        Args:
            dashboard_data: 仪表板数据
            output_path: 输出路径
            format: 导出格式
            
        Returns:
            是否成功
        """
        try:
            if format == 'html':
                # 导出为HTML
                comprehensive_dashboard = dashboard_data.get('comprehensive_dashboard')
                if comprehensive_dashboard:
                    comprehensive_dashboard.write_html(output_path)
                    self.logger.info(f"仪表板已导出到: {output_path}")
                    return True
            elif format == 'json':
                # 导出为JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)
                self.logger.info(f"仪表板数据已导出到: {output_path}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"导出仪表板失败: {e}")
            return False
