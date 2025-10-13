#!/usr/bin/env python3
"""
时序异构图演示

演示时序异构图的构建、可视化和分析过程
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import json

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.config.simple_config import SimpleConfig
    from src.data.apt_data_processor import APTDataProcessor
    from src.data.pyg_loader import PyG_LinuxAPTDataLoader
    from src.core.models.t_hgnn import T_HGNN
    from src.core.models.temporal_encoder import RandomPositionalEncoding, TemporalAttention
    from src.visualization.dashboards.attack_story_board import AttackStoryBoard
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def create_temporal_sample_data():
    """创建时序示例数据"""
    logger = logging.getLogger(__name__)
    logger.info("创建时序示例数据")
    
    # 创建时间序列数据
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(100)]
    
    # 模拟APT攻击序列
    attack_sequence = [
        # 阶段1: 初始访问 (0-20分钟)
        {'event': 'phishing_email', 'node_type': 'alert', 'time_offset': 0},
        {'event': 'malicious_attachment', 'node_type': 'file', 'time_offset': 5},
        {'event': 'macro_execution', 'node_type': 'process', 'time_offset': 10},
        {'event': 'powershell_execution', 'node_type': 'command', 'time_offset': 15},
        
        # 阶段2: 执行 (20-40分钟)
        {'event': 'payload_download', 'node_type': 'process', 'time_offset': 25},
        {'event': 'privilege_escalation', 'node_type': 'process', 'time_offset': 30},
        {'event': 'persistence_mechanism', 'node_type': 'registry', 'time_offset': 35},
        
        # 阶段3: 持久化 (40-60分钟)
        {'event': 'backdoor_installation', 'node_type': 'file', 'time_offset': 45},
        {'event': 'scheduled_task', 'node_type': 'process', 'time_offset': 50},
        {'event': 'service_creation', 'node_type': 'service', 'time_offset': 55},
        
        # 阶段4: 横向移动 (60-80分钟)
        {'event': 'network_scan', 'node_type': 'process', 'time_offset': 65},
        {'event': 'credential_dump', 'node_type': 'process', 'time_offset': 70},
        {'event': 'lateral_movement', 'node_type': 'process', 'time_offset': 75},
        
        # 阶段5: 数据收集 (80-100分钟)
        {'event': 'data_collection', 'node_type': 'process', 'time_offset': 85},
        {'event': 'data_exfiltration', 'node_type': 'process', 'time_offset': 90},
        {'event': 'cleanup', 'node_type': 'process', 'time_offset': 95}
    ]
    
    # 创建DataFrame
    data = []
    for i, attack in enumerate(attack_sequence):
        event_time = base_time + timedelta(minutes=attack['time_offset'])
        
        # 创建相关节点
        alert_id = f"alert_{i:03d}"
        process_id = f"process_{i:03d}"
        file_id = f"file_{i:03d}"
        command_id = f"command_{i:03d}"
        user_id = f"user_{i:03d}"
        host_id = f"host_{i:03d}"
        
        # 添加告警事件
        data.append({
            'timestamp': event_time.strftime('%Y-%m-%d %H:%M:%S'),
            'alert_id': alert_id,
            'process_name': attack['event'],
            'pid': 1000 + i,
            'ppid': 999 + i,
            'command_line': f"{attack['event']} --args",
            'file_path': f"/tmp/{attack['event']}.exe",
            'network_connection': f"192.168.1.{i%254+1}:{8080+i}",
            'event_type': 'process_start',
            'node_type': attack['node_type'],
            'attack_stage': self._get_attack_stage(attack['time_offset']),
            'malicious_score': min(0.9, 0.1 + i * 0.05)
        })
    
    return pd.DataFrame(data)


def _get_attack_stage(time_offset):
    """获取攻击阶段"""
    if time_offset < 20:
        return 'initial_access'
    elif time_offset < 40:
        return 'execution'
    elif time_offset < 60:
        return 'persistence'
    elif time_offset < 80:
        return 'lateral_movement'
    else:
        return 'data_collection'


def visualize_temporal_graph(hetero_data, timestamps, output_dir):
    """可视化时序异构图"""
    logger = logging.getLogger(__name__)
    logger.info("可视化时序异构图")
    
    try:
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('时序异构图可视化', fontsize=16, fontweight='bold')
        
        # 1. 节点类型分布
        ax1 = axes[0, 0]
        node_counts = {ntype: hetero_data[ntype].num_nodes for ntype in hetero_data.node_types}
        ax1.bar(node_counts.keys(), node_counts.values())
        ax1.set_title('节点类型分布')
        ax1.set_xlabel('节点类型')
        ax1.set_ylabel('节点数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 边类型分布
        ax2 = axes[0, 1]
        edge_counts = {edge_type: hetero_data[edge_type].edge_index.size(1) for edge_type in hetero_data.edge_types}
        ax2.bar(range(len(edge_counts)), list(edge_counts.values()))
        ax2.set_title('边类型分布')
        ax2.set_xlabel('边类型索引')
        ax2.set_ylabel('边数量')
        ax2.set_xticks(range(len(edge_counts)))
        ax2.set_xticklabels([f"{src}->{dst}" for src, _, dst in edge_counts.keys()], rotation=45)
        
        # 3. 时序演化
        ax3 = axes[1, 0]
        if timestamps:
            # 创建时序图
            G = nx.DiGraph()
            
            # 添加节点
            for ntype in hetero_data.node_types:
                if ntype in timestamps:
                    for i, timestamp in enumerate(timestamps[ntype]):
                        node_id = f"{ntype}_{i}"
                        G.add_node(node_id, node_type=ntype, timestamp=timestamp.item())
            
            # 添加边（基于时间顺序）
            for edge_type in hetero_data.edge_types:
                if edge_type in hetero_data.edge_index_dict:
                    edge_index = hetero_data[edge_type].edge_index
                    for i in range(edge_index.size(1)):
                        src_idx, dst_idx = edge_index[:, i]
                        src_node = f"{edge_type[0]}_{src_idx.item()}"
                        dst_node = f"{edge_type[2]}_{dst_idx.item()}"
                        if src_node in G and dst_node in G:
                            G.add_edge(src_node, dst_node, edge_type=edge_type[1])
            
            # 绘制时序图
            pos = nx.spring_layout(G, k=1, iterations=50)
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node]['node_type']
                if node_type == 'alert':
                    node_colors.append('red')
                elif node_type == 'process':
                    node_colors.append('blue')
                elif node_type == 'file':
                    node_colors.append('green')
                else:
                    node_colors.append('gray')
            
            nx.draw(G, pos, ax=ax3, node_color=node_colors, 
                   node_size=100, with_labels=False, arrows=True, 
                   edge_color='gray', alpha=0.7)
            ax3.set_title('时序异构图结构')
        
        # 4. 时序特征分析
        ax4 = axes[1, 1]
        if timestamps:
            # 计算时序统计
            all_timestamps = []
            for ntype in timestamps.keys():
                all_timestamps.extend(timestamps[ntype].tolist())
            
            if all_timestamps:
                ax4.hist(all_timestamps, bins=20, alpha=0.7, edgecolor='black')
                ax4.set_title('时间戳分布')
                ax4.set_xlabel('时间戳')
                ax4.set_ylabel('频次')
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(output_dir, "temporal_graph_visualization.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时序异构图可视化已保存到: {plot_path}")
        
    except Exception as e:
        logger.error(f"可视化时序异构图失败: {e}")


def demonstrate_temporal_encoding():
    """演示时序编码"""
    logger = logging.getLogger(__name__)
    logger.info("演示时序编码")
    
    try:
        # 创建示例数据
        batch_size = 32
        seq_len = 100
        d_model = 128
        
        # 创建输入特征（演示用随机数据）
        torch.manual_seed(42)  # 确保演示结果一致
        x = torch.randn(batch_size, seq_len, d_model)
        timestamps = torch.linspace(0, 100, seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # 1. 随机位置编码
        logger.info("测试随机位置编码")
        rpe = RandomPositionalEncoding(d_model=d_model, max_len=5000)
        rpe_output = rpe(x, timestamps)
        logger.info(f"RPE输入形状: {x.shape}")
        logger.info(f"RPE输出形状: {rpe_output.shape}")
        
        # 2. 时序注意力
        logger.info("测试时序注意力")
        temporal_attn = TemporalAttention(d_model=d_model, num_heads=8)
        attn_output = temporal_attn(x, timestamps)
        logger.info(f"时序注意力输入形状: {x.shape}")
        logger.info(f"时序注意力输出形状: {attn_output.shape}")
        
        # 3. 时序特征分析
        logger.info("分析时序特征")
        
        # 计算时序统计
        time_diff = timestamps[:, 1:] - timestamps[:, :-1]
        logger.info(f"时间间隔统计: 均值={time_diff.mean():.4f}, 标准差={time_diff.std():.4f}")
        
        # 计算注意力权重
        with torch.no_grad():
            Q = temporal_attn.W_q(x)
            K = temporal_attn.W_k(x)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
            attention_weights = torch.softmax(scores, dim=-1)
            
            # 计算平均注意力权重
            avg_attention = attention_weights.mean(dim=1).mean(dim=0)
            logger.info(f"平均注意力权重形状: {avg_attention.shape}")
            logger.info(f"注意力权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
        
        return {
            'rpe_output': rpe_output,
            'attn_output': attn_output,
            'attention_weights': attention_weights,
            'time_diff_stats': {
                'mean': time_diff.mean().item(),
                'std': time_diff.std().item()
            }
        }
        
    except Exception as e:
        logger.error(f"演示时序编码失败: {e}")
        return None


def analyze_temporal_patterns(hetero_data, timestamps):
    """分析时序模式"""
    logger = logging.getLogger(__name__)
    logger.info("分析时序模式")
    
    try:
        patterns = {}
        
        # 1. 时序统计
        all_timestamps = []
        for ntype in timestamps.keys():
            all_timestamps.extend(timestamps[ntype].tolist())
        
        if all_timestamps:
            patterns['temporal_stats'] = {
                'min_timestamp': min(all_timestamps),
                'max_timestamp': max(all_timestamps),
                'time_span': max(all_timestamps) - min(all_timestamps),
                'total_events': len(all_timestamps)
            }
        
        # 2. 节点类型时序分析
        patterns['node_temporal'] = {}
        for ntype in timestamps.keys():
            node_timestamps = timestamps[ntype].tolist()
            if node_timestamps:
                patterns['node_temporal'][ntype] = {
                    'count': len(node_timestamps),
                    'first_event': min(node_timestamps),
                    'last_event': max(node_timestamps),
                    'duration': max(node_timestamps) - min(node_timestamps)
                }
        
        # 3. 时序间隔分析
        patterns['intervals'] = {}
        for ntype in timestamps.keys():
            node_timestamps = timestamps[ntype].tolist()
            if len(node_timestamps) > 1:
                intervals = [node_timestamps[i+1] - node_timestamps[i] for i in range(len(node_timestamps)-1)]
                patterns['intervals'][ntype] = {
                    'mean_interval': np.mean(intervals),
                    'std_interval': np.std(intervals),
                    'min_interval': min(intervals),
                    'max_interval': max(intervals)
                }
        
        # 4. 时序相关性分析
        patterns['correlations'] = {}
        node_types = list(timestamps.keys())
        for i, ntype1 in enumerate(node_types):
            for ntype2 in node_types[i+1:]:
                if ntype1 in timestamps and ntype2 in timestamps:
                    ts1 = timestamps[ntype1].numpy()
                    ts2 = timestamps[ntype2].numpy()
                    
                    # 计算时序相关性
                    if len(ts1) > 1 and len(ts2) > 1:
                        # 对齐时间序列
                        min_len = min(len(ts1), len(ts2))
                        ts1_aligned = ts1[:min_len]
                        ts2_aligned = ts2[:min_len]
                        
                        correlation = np.corrcoef(ts1_aligned, ts2_aligned)[0, 1]
                        patterns['correlations'][f"{ntype1}_{ntype2}"] = correlation
        
        logger.info(f"时序模式分析完成，发现 {len(patterns)} 种模式")
        return patterns
        
    except Exception as e:
        logger.error(f"分析时序模式失败: {e}")
        return {}


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始时序异构图演示")
    
    try:
        # 1. 创建配置
        logger.info("步骤1: 创建配置")
        config = SimpleConfig()
        config.data.output_dir = "./output_temporal_demo"
        config.model.hidden_dim = 128
        config.model.temporal_dim = 64
        config.model.num_heads = 8
        config.model.num_layers = 3
        
        # 创建输出目录
        os.makedirs(config.data.output_dir, exist_ok=True)
        
        # 2. 创建时序示例数据
        logger.info("步骤2: 创建时序示例数据")
        df = create_temporal_sample_data()
        logger.info(f"创建了 {len(df)} 条时序示例数据")
        
        # 3. 数据处理
        logger.info("步骤3: 数据处理")
        processor = APTDataProcessor(config.data)
        processed_df = processor.process_raw_data(df)
        logger.info(f"数据处理完成，最终数据形状: {processed_df.shape}")
        
        # 4. 构建时序异构图
        logger.info("步骤4: 构建时序异构图")
        loader = PyG_LinuxAPTDataLoader(config.data)
        hetero_data = loader.build_hetero_graph(processed_df)
        logger.info(f"图构建完成，节点类型: {hetero_data.node_types}")
        logger.info(f"边类型: {hetero_data.edge_types}")
        
        # 5. 创建时序快照
        logger.info("步骤5: 创建时序快照")
        snapshots = loader.create_temporal_snapshots(hetero_data)
        logger.info(f"创建了 {len(snapshots)} 个时序快照")
        
        # 6. 生成时间戳
        logger.info("步骤6: 生成时间戳")
        timestamps = {}
        for ntype in hetero_data.node_types:
            if ntype in hetero_data.node_types and hetero_data[ntype].num_nodes > 0:
                # 生成随机时间戳
                num_nodes = hetero_data[ntype].num_nodes
                timestamps[ntype] = torch.linspace(0, 100, num_nodes)
        
        logger.info(f"生成了 {len(timestamps)} 种节点类型的时间戳")
        
        # 7. 演示时序编码
        logger.info("步骤7: 演示时序编码")
        encoding_results = demonstrate_temporal_encoding()
        if encoding_results:
            logger.info("时序编码演示完成")
        
        # 8. 分析时序模式
        logger.info("步骤8: 分析时序模式")
        temporal_patterns = analyze_temporal_patterns(hetero_data, timestamps)
        logger.info(f"时序模式分析完成，发现 {len(temporal_patterns)} 种模式")
        
        # 9. 可视化时序异构图
        logger.info("步骤9: 可视化时序异构图")
        visualize_temporal_graph(hetero_data, timestamps, config.data.output_dir)
        
        # 10. 创建T-HGNN模型
        logger.info("步骤10: 创建T-HGNN模型")
        node_types = list(hetero_data.node_types)
        edge_types = loader.edge_types
        in_dims = {ntype: 128 for ntype in node_types}
        
        model = T_HGNN(config, node_types, edge_types, in_dims)
        logger.info("T-HGNN模型创建完成")
        
        # 11. 测试模型前向传播
        logger.info("步骤11: 测试模型前向传播")
        model.eval()
        with torch.no_grad():
            # 测试无时序信息
            predictions_no_time = model.predict(hetero_data)
            logger.info(f"无时序信息预测完成，输出类型: {list(predictions_no_time.keys())}")
            
            # 测试有时序信息
            predictions_with_time = model.predict(hetero_data, timestamps)
            logger.info(f"有时序信息预测完成，输出类型: {list(predictions_with_time.keys())}")
            
            # 获取节点嵌入
            embeddings = model.get_embeddings(hetero_data, timestamps)
            logger.info(f"节点嵌入获取完成，嵌入类型: {list(embeddings.keys())}")
        
        # 12. 保存结果
        logger.info("步骤12: 保存结果")
        
        # 保存时序模式分析结果
        patterns_path = os.path.join(config.data.output_dir, "temporal_patterns.json")
        with open(patterns_path, 'w', encoding='utf-8') as f:
            json.dump(temporal_patterns, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存模型信息
        model_info = model.get_model_info()
        model_info_path = os.path.join(config.data.output_dir, "model_info.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存编码结果
        if encoding_results:
            encoding_path = os.path.join(config.data.output_dir, "encoding_results.json")
            encoding_data = {
                'rpe_output_shape': list(encoding_results['rpe_output'].shape),
                'attn_output_shape': list(encoding_results['attn_output'].shape),
                'attention_weights_shape': list(encoding_results['attention_weights'].shape),
                'time_diff_stats': encoding_results['time_diff_stats']
            }
            with open(encoding_path, 'w', encoding='utf-8') as f:
                json.dump(encoding_data, f, ensure_ascii=False, indent=2)
        
        # 13. 生成演示报告
        logger.info("步骤13: 生成演示报告")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_processed': len(processed_df),
            'node_types': list(hetero_data.node_types),
            'edge_types': [f"{src}->{dst}" for src, _, dst in hetero_data.edge_types],
            'snapshots_created': len(snapshots),
            'temporal_patterns': len(temporal_patterns),
            'model_parameters': model_info['total_parameters'],
            'encoding_successful': encoding_results is not None,
            'output_directory': config.data.output_dir
        }
        
        report_path = os.path.join(config.data.output_dir, "temporal_demo_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("时序异构图演示完成")
        logger.info("=" * 60)
        logger.info(f"处理数据: {len(processed_df)} 条")
        logger.info(f"节点类型: {len(hetero_data.node_types)} 种")
        logger.info(f"边类型: {len(hetero_data.edge_types)} 种")
        logger.info(f"时序快照: {len(snapshots)} 个")
        logger.info(f"时序模式: {len(temporal_patterns)} 种")
        logger.info(f"模型参数: {model_info['total_parameters']:,} 个")
        logger.info(f"输出目录: {config.data.output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"时序异构图演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
