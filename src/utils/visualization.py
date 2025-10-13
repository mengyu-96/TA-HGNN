"""
可视化模块

提供攻击链可视化、图结构可视化、模型性能可视化等功能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from datetime import datetime
import torch
import dgl

try:
    from ..config.config import SystemConfig
except ImportError:
    from ..config.simple_config import SimpleSystemConfig as SystemConfig


class AttackChainVisualizer:
    """攻击链可视化器"""
    
    def __init__(self, config: SystemConfig):
        """
        初始化攻击链可视化器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 节点类型颜色映射
        self.node_colors = {
            'alert': '#FF6B6B',      # 红色 - 警报
            'host': '#4ECDC4',       # 青色 - 主机
            'agent': '#45B7D1',      # 蓝色 - 代理
            'rule': '#96CEB4',       # 绿色 - 规则
            'file': '#FFEAA7',       # 黄色 - 文件
            'command': '#DDA0DD',    # 紫色 - 命令
            'user': '#98D8C8',       # 薄荷绿 - 用户
            'process': '#F7DC6F',    # 金色 - 进程
            'ip': '#BB8FCE',         # 淡紫色 - IP
            'domain': '#85C1E9',     # 天蓝色 - 域名
            'timestamp': '#F8C471',  # 橙色 - 时间戳
            'registry': '#82E0AA',   # 浅绿色 - 注册表
            'port': '#F1948A',       # 粉红色 - 端口
            'service': '#D7BDE2'     # 淡紫色 - 服务
        }
        
        # 边类型样式映射
        self.edge_styles = {
            'alert_detected_on_host': {'color': '#FF6B6B', 'width': 2, 'style': 'solid'},
            'alert_triggered_by_rule': {'color': '#96CEB4', 'width': 1.5, 'style': 'dashed'},
            'alert_involves_file': {'color': '#FFEAA7', 'width': 1.5, 'style': 'solid'},
            'alert_executed_command': {'color': '#DDA0DD', 'width': 1.5, 'style': 'dotted'},
            'alert_by_user': {'color': '#98D8C8', 'width': 1.5, 'style': 'solid'},
            'alert_involves_process': {'color': '#F7DC6F', 'width': 1.5, 'style': 'solid'},
            'alert_connects_to_ip': {'color': '#BB8FCE', 'width': 1.5, 'style': 'solid'},
            'alert_connects_to_domain': {'color': '#85C1E9', 'width': 1.5, 'style': 'solid'},
            'alert_uses_port': {'color': '#F1948A', 'width': 1, 'style': 'dotted'},
            'host_has_ip': {'color': '#4ECDC4', 'width': 2, 'style': 'solid'},
            'process_accesses_file': {'color': '#F7DC6F', 'width': 1, 'style': 'solid'},
            'user_owns_process': {'color': '#98D8C8', 'width': 1.5, 'style': 'solid'},
            'ip_resolves_to_domain': {'color': '#BB8FCE', 'width': 1, 'style': 'dashed'}
        }
    
    def visualize_attack_chain(self, g: nx.DiGraph, attack_chain: List[str], 
                              output_file: str = "attack_chain.png") -> None:
        """
        可视化攻击链
        
        Args:
            g: NetworkX图
            attack_chain: 攻击链节点列表
            output_file: 输出文件路径
        """
        self.logger.info(f"可视化攻击链，包含 {len(attack_chain)} 个节点")
        
        # 创建子图
        subgraph = g.subgraph(attack_chain)
        
        if subgraph.number_of_nodes() == 0:
            self.logger.warning("攻击链子图为空")
            return
        
        # 设置图形大小
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # 计算布局
        pos = self._compute_attack_chain_layout(subgraph, attack_chain)
        
        # 绘制节点
        self._draw_nodes(subgraph, pos, ax)
        
        # 绘制边
        self._draw_edges(subgraph, pos, ax)
        
        # 添加标签
        self._add_labels(subgraph, pos, ax)
        
        # 添加图例
        self._add_legend(ax)
        
        # 设置标题
        ax.set_title(f"APT攻击链可视化\n包含 {subgraph.number_of_nodes()} 个节点，{subgraph.number_of_edges()} 条边", 
                    fontsize=16, fontweight='bold')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"攻击链可视化已保存到: {output_file}")
    
    def visualize_graph_structure(self, g: nx.DiGraph, 
                                 output_file: str = "graph_structure.png",
                                 max_nodes: int = 500) -> None:
        """
        可视化图结构
        
        Args:
            g: NetworkX图
            output_file: 输出文件路径
            max_nodes: 最大节点数
        """
        self.logger.info(f"可视化图结构，原始图包含 {g.number_of_nodes()} 个节点")
        
        # 如果图太大，创建子图
        if g.number_of_nodes() > max_nodes:
            subgraph = self._create_representative_subgraph(g, max_nodes)
            self.logger.info(f"创建了包含 {subgraph.number_of_nodes()} 个节点的子图")
        else:
            subgraph = g
        
        # 设置图形大小
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左图：整体图结构
        self._draw_graph_overview(subgraph, ax1)
        
        # 右图：节点类型分布
        self._draw_node_type_distribution(subgraph, ax2)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"图结构可视化已保存到: {output_file}")
    
    def visualize_temporal_evolution(self, snapshots: List[dgl.DGLHeteroGraph],
                                   output_file: str = "temporal_evolution.png") -> None:
        """
        可视化时序演化
        
        Args:
            snapshots: DGL图快照列表
            output_file: 输出文件路径
        """
        self.logger.info(f"可视化时序演化，包含 {len(snapshots)} 个快照")
        
        # 计算每个快照的统计信息
        stats = []
        for i, snapshot in enumerate(snapshots):
            total_nodes = sum([snapshot.num_nodes(ntype) for ntype in snapshot.ntypes])
            # 使用canonical_etypes以避免当同一etype字符串在不同(src,etype,dst)中重复时产生歧义
            try:
                total_edges = sum([snapshot.num_edges(c_etype) for c_etype in snapshot.canonical_etypes])
            except Exception:
                # 回退到按etype字符串统计
                total_edges = sum([snapshot.num_edges(etype) for etype in snapshot.etypes])
            stats.append({
                'snapshot': i + 1,
                'nodes': total_nodes,
                'edges': total_edges,
                'alerts': snapshot.num_nodes('alert') if 'alert' in snapshot.ntypes else 0
            })
        
        # 创建DataFrame
        df = pd.DataFrame(stats)
        
        # 设置图形大小
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # 节点数量演化
        ax1.plot(df['snapshot'], df['nodes'], marker='o', linewidth=2, markersize=8)
        ax1.set_title('节点数量时序演化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间快照')
        ax1.set_ylabel('节点数量')
        ax1.grid(True, alpha=0.3)
        
        # 边数量演化
        ax2.plot(df['snapshot'], df['edges'], marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_title('边数量时序演化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间快照')
        ax2.set_ylabel('边数量')
        ax2.grid(True, alpha=0.3)
        
        # 警报数量演化
        ax3.plot(df['snapshot'], df['alerts'], marker='^', linewidth=2, markersize=8, color='red')
        ax3.set_title('警报数量时序演化', fontsize=14, fontweight='bold')
        ax3.set_xlabel('时间快照')
        ax3.set_ylabel('警报数量')
        ax3.grid(True, alpha=0.3)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"时序演化可视化已保存到: {output_file}")
    
    def visualize_model_performance(self, metrics: Dict[str, List[float]],
                                  output_file: str = "model_performance.png") -> None:
        """
        可视化模型性能
        
        Args:
            metrics: 性能指标字典 {metric_name: values}
            output_file: 输出文件路径
        """
        self.logger.info("可视化模型性能")
        
        # 设置图形大小
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘制各种指标
        for i, (metric_name, values) in enumerate(metrics.items()):
            if i >= 4:
                break
            
            ax = axes[i]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # 添加最佳值标注
            if values:
                best_idx = np.argmax(values) if 'accuracy' in metric_name.lower() or 'f1' in metric_name.lower() else np.argmin(values)
                best_value = values[best_idx]
                ax.annotate(f'Best: {best_value:.4f}', 
                           xy=(best_idx + 1, best_value),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 隐藏多余的子图
        for i in range(len(metrics), 4):
            axes[i].set_visible(False)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"模型性能可视化已保存到: {output_file}")
    
    def visualize_attention_weights(self, attention_weights: Dict[str, torch.Tensor],
                                  output_file: str = "attention_weights.png") -> None:
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重字典
            output_file: 输出文件路径
        """
        self.logger.info("可视化注意力权重")
        
        # 设置图形大小
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 绘制注意力权重热力图
        for i, (layer_name, weights) in enumerate(attention_weights.items()):
            if i >= 4:
                break
            
            ax = axes[i]
            
            # 转换为numpy数组
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            
            # 如果是3D张量，取平均
            if weights.ndim == 3:
                weights = weights.mean(axis=0)
            
            # 绘制热力图
            im = ax.imshow(weights, cmap='Blues', aspect='auto')
            ax.set_title(f'{layer_name} 注意力权重', fontsize=12, fontweight='bold')
            ax.set_xlabel('目标节点')
            ax.set_ylabel('源节点')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
        
        # 隐藏多余的子图
        for i in range(len(attention_weights), 4):
            axes[i].set_visible(False)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"注意力权重可视化已保存到: {output_file}")
    
    def _compute_attack_chain_layout(self, g: nx.DiGraph, attack_chain: List[str]) -> Dict[str, Tuple[float, float]]:
        """计算攻击链布局"""
        # 使用层次布局，按时间顺序排列
        pos = {}
        
        # 按节点类型分组
        node_groups = {}
        for node in attack_chain:
            if node in g.nodes:
                node_type = g.nodes[node].get('type', 'unknown')
                if node_type not in node_groups:
                    node_groups[node_type] = []
                node_groups[node_type].append(node)
        
        # 按层次排列
        y_positions = {
            'alert': 0,
            'host': 1,
            'user': 2,
            'process': 3,
            'file': 4,
            'ip': 5,
            'domain': 6,
            'timestamp': 7
        }
        
        for node_type, nodes in node_groups.items():
            y = y_positions.get(node_type, 8)
            for i, node in enumerate(nodes):
                x = i * 2
                pos[node] = (x, y)
        
        return pos
    
    def _draw_nodes(self, g: nx.DiGraph, pos: Dict[str, Tuple[float, float]], ax) -> None:
        """绘制节点"""
        for node, (x, y) in pos.items():
            if node in g.nodes:
                node_type = g.nodes[node].get('type', 'unknown')
                color = self.node_colors.get(node_type, '#CCCCCC')
                
                # 根据节点类型设置大小
                size = 300 if node_type == 'alert' else 200
                
                ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='black', linewidth=1)
    
    def _draw_edges(self, g: nx.DiGraph, pos: Dict[str, Tuple[float, float]], ax) -> None:
        """绘制边"""
        for edge in g.edges(data=True):
            src, dst, data = edge
            if src in pos and dst in pos:
                x1, y1 = pos[src]
                x2, y2 = pos[dst]
                
                edge_type = data.get('type', 'unknown')
                style = self.edge_styles.get(edge_type, {'color': '#CCCCCC', 'width': 1, 'style': 'solid'})
                
                ax.plot([x1, x2], [y1, y2], 
                       color=style['color'], 
                       linewidth=style['width'],
                       linestyle=style['style'],
                       alpha=0.7)
                
                # 添加箭头
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', 
                                         color=style['color'],
                                         lw=style['width']))
    
    def _add_labels(self, g: nx.DiGraph, pos: Dict[str, Tuple[float, float]], ax) -> None:
        """添加标签"""
        for node, (x, y) in pos.items():
            if node in g.nodes:
                # 只标注重要节点
                node_type = g.nodes[node].get('type', 'unknown')
                if node_type in ['alert', 'host', 'user']:
                    label = f"{node_type}:{node[:8]}"
                    ax.annotate(label, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
    
    def _add_legend(self, ax) -> None:
        """添加图例"""
        legend_elements = []
        for node_type, color in self.node_colors.items():
            legend_elements.append(
                mpatches.Patch(color=color, label=node_type)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _create_representative_subgraph(self, g: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
        """创建代表性子图"""
        # 选择警报节点作为起点
        alert_nodes = [node for node, data in g.nodes(data=True) if data.get('type') == 'alert']
        
        if not alert_nodes:
            # 如果没有警报节点，选择任意节点
            all_nodes = list(g.nodes())
            start_nodes = all_nodes[:5] if len(all_nodes) >= 5 else all_nodes
        else:
            # 随机选择几个警报节点
            start_nodes = np.random.choice(alert_nodes, min(5, len(alert_nodes)), replace=False).tolist()
        
        # 使用BFS扩展子图
        subgraph_nodes = set(start_nodes)
        queue = list(start_nodes)
        
        while queue and len(subgraph_nodes) < max_nodes:
            current_node = queue.pop(0)
            
            # 添加邻居节点
            for neighbor in g.neighbors(current_node):
                if len(subgraph_nodes) < max_nodes and neighbor not in subgraph_nodes:
                    subgraph_nodes.add(neighbor)
                    queue.append(neighbor)
        
        return g.subgraph(subgraph_nodes)
    
    def _draw_graph_overview(self, g: nx.DiGraph, ax) -> None:
        """绘制图概览"""
        # 计算布局
        pos = nx.spring_layout(g, k=1, iterations=50)
        
        # 绘制节点
        for node, data in g.nodes(data=True):
            node_type = data.get('type', 'unknown')
            color = self.node_colors.get(node_type, '#CCCCCC')
            size = 50 if node_type == 'alert' else 30
            
            ax.scatter(pos[node][0], pos[node][1], c=color, s=size, alpha=0.7)
        
        # 绘制边
        for edge in g.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'图结构概览\n{g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边')
        ax.axis('off')
    
    def _draw_node_type_distribution(self, g: nx.DiGraph, ax) -> None:
        """绘制节点类型分布"""
        # 统计节点类型
        node_types = {}
        for node, data in g.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # 绘制饼图
        labels = list(node_types.keys())
        sizes = list(node_types.values())
        colors = [self.node_colors.get(label, '#CCCCCC') for label in labels]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('节点类型分布')
    
    def create_interactive_visualization(self, g: nx.DiGraph, 
                                       output_file: str = "interactive_graph.html") -> None:
        """
        创建交互式可视化（使用plotly）
        
        Args:
            g: NetworkX图
            output_file: 输出HTML文件路径
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot
            
            self.logger.info("创建交互式可视化")
            
            # 计算布局
            pos = nx.spring_layout(g, k=1, iterations=50)
            
            # 准备节点数据
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node, (x, y) in pos.items():
                node_x.append(x)
                node_y.append(y)
                
                node_type = g.nodes[node].get('type', 'unknown')
                node_text.append(f"{node_type}: {node}")
                node_colors.append(self.node_colors.get(node_type, '#CCCCCC'))
            
            # 准备边数据
            edge_x = []
            edge_y = []
            
            for edge in g.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # 创建图形
            fig = go.Figure()
            
            # 添加边
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=0.5, color='#888'),
                                   hoverinfo='none',
                                   mode='lines'))
            
            # 添加节点
            fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                   mode='markers',
                                   hoverinfo='text',
                                   text=node_text,
                                   marker=dict(size=10,
                                             color=node_colors,
                                             line=dict(width=2, color='black'))))
            
            # 设置布局
            fig.update_layout(title='交互式APT攻击图',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="拖拽节点进行交互",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color='#888', size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            
            # 保存HTML文件
            plot(fig, filename=output_file, auto_open=False)
            
            self.logger.info(f"交互式可视化已保存到: {output_file}")
            
        except ImportError:
            self.logger.warning("plotly未安装，无法创建交互式可视化")
        except Exception as e:
            self.logger.error(f"创建交互式可视化失败: {e}")


class ModelPerformanceVisualizer:
    """模型性能可视化器"""
    
    def __init__(self, config: SystemConfig):
        """
        初始化模型性能可视化器
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                           train_accs: List[float], val_accs: List[float],
                           output_file: str = "training_curves.png") -> None:
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_accs: 训练准确率列表
            val_accs: 验证准确率列表
            output_file: 输出文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已保存到: {output_file}")
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                            class_names: List[str] = None,
                            output_file: str = "confusion_matrix.png") -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            output_file: 输出文件路径
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"混淆矩阵已保存到: {output_file}")
    
    def plot_roc_curve(self, y_true: List[int], y_scores: List[float],
                      output_file: str = "roc_curve.png") -> None:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            output_file: 输出文件路径
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC曲线已保存到: {output_file}")
