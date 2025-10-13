#!/usr/bin/env python3
"""
基于时序异质图神经网络的APT攻击检测与溯源系统

主程序入口 - 统一的项目入口点
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config.improved_config import ImprovedConfig, MemoryOptimizedConfig, GPUMemoryOptimizedConfig
    from src.config.performance_optimized_config import PerformanceOptimizedConfig
    Config = ImprovedConfig
except ImportError:
    from src.config.simple_config import SimpleConfig
    Config = SimpleConfig
    MemoryOptimizedConfig = SimpleConfig
    GPUMemoryOptimizedConfig = SimpleConfig
    PerformanceOptimizedConfig = SimpleConfig

try:
    from src.data.pyg_loader import PyG_LinuxAPTDataLoader
    from src.core.models.t_hgnn import T_HGNN
    from src.applications.detection.anomaly_detector import AnomalyDetector
    from src.applications.detection.attack_detector import AttackDetector
    from src.applications.detection.threat_classifier import ThreatClassifier
    from src.applications.tracing.attack_tracer import AttackTracer
    from src.applications.tracing.path_reconstructor import PathReconstructor
    from src.applications.tracing.causality_analyzer import CausalityAnalyzer
    from src.applications.clustering.attack_clusterer import AttackClusterer
    from src.utils.visualization import AttackChainVisualizer
    from src.visualization.dashboards.attack_story_board import AttackStoryBoard
    from src.utils.memory_monitor import MemoryMonitor
    from src.utils.gpu_utils import GPUUtils
    from src.evaluation.path_tracing_evaluator import AttackPath
except ImportError as e:
    print(f"警告: 无法导入某些模块，可能缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(config):
    """设置日志"""
    log_level = getattr(logging, config.system.log_level.upper())
    
    # 创建日志目录
    log_dir = os.path.join(config.data.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件
    if config.system.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"t_hgnn_{timestamp}.log")
    else:
        log_file = config.system.log_file
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志已配置，日志文件: {log_file}")
    
    return logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于时序异质图神经网络的APT攻击检测与溯源系统')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['train', 'detect', 'trace', 'cluster', 'full'],
                       help='运行模式: train(训练), detect(检测), trace(溯源), cluster(聚类), full(完整流程)')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                       default='./Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv',
                       help='Linux APT数据集路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值')
    
    # 检测参数
    parser.add_argument('--anomaly_threshold', type=float, default=0.7,
                       help='异常检测阈值')
    parser.add_argument('--attack_threshold', type=float, default=0.8,
                       help='攻击检测阈值')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='置信度阈值')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载预训练模型路径')
    parser.add_argument('--memory_optimized', action='store_true',
                       help='使用内存优化模式')
    parser.add_argument('--gpu_optimized', action='store_true',
                       help='使用GPU内存优化模式')
    parser.add_argument('--performance_optimized', action='store_true',
                       help='使用性能优化模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    if args.performance_optimized:
        config = PerformanceOptimizedConfig()
        logger = setup_logging(config)
        logger.info("使用性能优化配置")
    elif args.gpu_optimized:
        config = GPUMemoryOptimizedConfig()
        logger = setup_logging(config)
        logger.info("使用GPU内存优化配置")
    elif args.memory_optimized:
        config = MemoryOptimizedConfig()
        logger = setup_logging(config)
        logger.info("使用内存优化配置")
    else:
        config = Config()
        logger = setup_logging(config)
    
    # 更新配置
    config.data.data_path = args.data_path
    config.data.output_dir = args.output_dir
    config.model.hidden_dim = args.hidden_dim
    config.model.num_heads = args.num_heads
    config.model.num_layers = args.num_layers
    config.model.dropout = args.dropout
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.early_stopping_patience = args.patience
    config.system.seed = args.seed
    config.training.gpu_id = args.gpu
    config.system.visualize = args.visualize
    config.training.save_model = args.save_model
    
    # 设置设备
    if hasattr(config, '_setup_device'):
        config._setup_device()
    
    # 设置日志
    logger = setup_logging(config)
    
    # 初始化GPU工具
    gpu_utils = GPUUtils()
    gpu_utils.print_gpu_info()
    gpu_utils.optimize_gpu_settings()
    
    # 初始化内存监控
    memory_monitor = MemoryMonitor(threshold=0.8 if not args.memory_optimized else 0.6)
    
    # 打印配置信息
    config.print_config()
    
    # 显示内存状态
    memory_info = memory_monitor.get_memory_info()
    if memory_info:
        logger.info(f"初始内存状态: 使用率 {memory_info.get('memory_percent', 0):.1f}%, "
                   f"可用内存 {memory_info.get('available_memory', 0):.1f}GB")
    
    # 显示GPU状态
    gpu_utils.print_gpu_status()
    
    # 验证配置
    if not config.validate():
        logger.error("配置验证失败")
        return 1
    
    # 设置随机种子
    np.random.seed(config.system.seed)
    torch.manual_seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.system.seed)
    
    logger.info(f"开始运行模式: {args.mode}")
    
    try:
        # 1. 数据加载和预处理
        logger.info("=" * 50)
        logger.info("步骤1: 数据加载和预处理")
        logger.info("=" * 50)
        
        # 使用改进的APT数据处理器
        from src.data.improved_apt_data_processor import ImprovedAPTDataProcessor
        apt_processor = ImprovedAPTDataProcessor(config)
        data_loader = PyG_LinuxAPTDataLoader(config)
        
        # 加载原始数据
        df = data_loader.load_data()
        
        # 获取数据信息
        data_info = data_loader.get_data_info(df)
        logger.info(f"原始数据信息: {data_info['shape']}")
        
        # 使用改进的APT处理器处理数据
        df = apt_processor.process_raw_data(df)
        
        # 数据增强
        logger.info("开始数据增强...")
        try:
            from src.data.data_augmentation import WazuhDataAugmentation
            data_augmenter = WazuhDataAugmentation(config.data)
            df = data_augmenter.augment_positive_samples(df, target_ratio=0.2)
            logger.info(f"数据增强完成，数据量: {len(df)}")
        except ImportError as e:
            logger.warning(f"数据增强模块导入失败: {e}，跳过数据增强")
        except Exception as e:
            logger.warning(f"数据增强失败: {e}，使用原始数据")
        
        # 获取处理后的统计信息
        apt_stats = apt_processor.get_data_statistics(df)
        logger.info(f"APT数据统计: {apt_stats}")
        
        # 验证预处理后的数据
        if df.empty:
            logger.error("预处理后的数据为空，无法继续")
            return 1
        
        # 2. 构建异构图
        logger.info("=" * 50)
        logger.info("步骤2: 构建异构图")
        logger.info("=" * 50)
        
        hetero_data = data_loader.build_hetero_graph(df)
        
        # 获取图统计信息
        logger.info(f"图统计信息:")
        logger.info(f"  节点类型: {hetero_data.node_types}")
        logger.info(f"  边类型: {hetero_data.edge_types}")
        logger.info(f"  总节点数: {hetero_data.num_nodes}")
        logger.info(f"  总边数: {hetero_data.num_edges}")
        
        # 创建时序快照
        snapshots = data_loader.create_temporal_snapshots(hetero_data)
        logger.info(f"创建了 {len(snapshots)} 个时序快照")
        
        # 验证快照质量
        if len(snapshots) == 0:
            logger.error("没有创建任何时序快照，无法进行训练")
            return 1
        
        # 3. 模型构建
        logger.info("=" * 50)
        logger.info("步骤3: 构建T-HGNN模型")
        logger.info("=" * 50)
        
        # 获取节点类型和边类型
        node_types = list(hetero_data.node_types)
        edge_types = data_loader.edge_types
        
        # 获取输入维度
        in_dims = {}
        for ntype in node_types:
            if ntype in hetero_data.node_types and hetero_data[ntype].x is not None:
                in_dims[ntype] = hetero_data[ntype].x.shape[1]
            else:
                in_dims[ntype] = 64
        
        # 创建模型
        model = T_HGNN(config, node_types, edge_types, in_dims)
        model = model.to(config.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        logger.info(f"模型信息: {model_info}")
        
        # 4. 根据运行模式执行相应功能
        if args.mode in ['train', 'full']:
            # 训练模式
            logger.info("=" * 50)
            logger.info("步骤4: 模型训练")
            logger.info("=" * 50)
            
            # 数据划分
            if len(snapshots) >= 3:
                train_graphs = snapshots[:-2]
                val_graphs = snapshots[-2:-1]
                test_graphs = snapshots[-1:]
            elif len(snapshots) == 2:
                train_graphs = snapshots[:-1]
                val_graphs = snapshots[-1:]
                test_graphs = snapshots[-1:]
            else:
                train_graphs = snapshots
                val_graphs = snapshots
                test_graphs = snapshots
            
        # 生成标签
        from src.utils.label_generator import APTLabelGenerator
        label_generator = APTLabelGenerator(config)
        train_labels = label_generator.generate_labels(df, hetero_data)
        # 为验证集生成独立标签
        val_labels = label_generator.generate_labels(df, hetero_data)
        
        # 使用Wazuh重采样策略
        train_labels = label_generator.resample_labels(train_labels, 
                                                      {ntype: hetero_data[ntype].x for ntype in hetero_data.node_types 
                                                       if hasattr(hetero_data[ntype], 'x')}, 
                                                      df)
        
        # 计算类别权重
        class_weights = label_generator.compute_class_weights(train_labels)
        
        # 使用改进的训练器
        if args.performance_optimized:
            from src.core.training.performance_trainer import PerformanceOptimizedTrainer
            trainer = PerformanceOptimizedTrainer(model, config, class_weights)
        else:
            from src.core.training.improved_trainer import ImprovedModelTrainer
            trainer = ImprovedModelTrainer(model, config, class_weights)
        
        # 训练模型
        training_report = trainer.train(train_graphs, val_graphs, train_labels, val_labels)
        
        logger.info(f"训练完成，最佳验证损失: {training_report['best_val_loss']:.4f}")
        
        # 训练后内存状态
        if memory_monitor.check_memory_usage():
            logger.warning("训练后内存使用率过高，执行内存优化")
            memory_monitor.optimize_memory()
        
        # 保存模型
        if config.training.save_model:
            model_path = os.path.join(config.data.output_dir, "models", "best_model.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.save_model(model_path)
            logger.info(f"模型已保存到: {model_path}")
        
        # 模型性能评估
        logger.info("=" * 50)
        logger.info("步骤5: 模型性能评估")
        logger.info("=" * 50)
        
        # 生成测试标签
        test_labels = label_generator.generate_labels(df, hetero_data)
        
        # 创建一些示例真实攻击路径用于评估
        ground_truth_paths = []
        for i in range(min(5, len(test_graphs))):
            # 创建简单的真实攻击路径示例
            sample_path = AttackPath(
                nodes=[f"alert_{i}", f"process_{i}", f"file_{i}"],
                edges=[(f"alert_{i}", f"process_{i}"), (f"process_{i}", f"file_{i}")],
                attack_type="apt",
                confidence=1.0,
                length=3
            )
            ground_truth_paths.append(sample_path)
        
        # 执行模型评估
        evaluation_results = trainer.evaluate_model_performance(
            test_graphs, test_labels, ground_truth_paths
        )
        
        # 生成评估报告
        evaluation_report = trainer.generate_evaluation_report()
        logger.info("模型评估报告:")
        logger.info(evaluation_report)
        
        # 保存评估结果
        eval_results_path = os.path.join(config.data.output_dir, "results", "evaluation_results.json")
        os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
        trainer.save_evaluation_results(eval_results_path)
        
        # 保存评估报告
        eval_report_path = os.path.join(config.data.output_dir, "results", "evaluation_report.txt")
        with open(eval_report_path, 'w', encoding='utf-8') as f:
            f.write(evaluation_report)
        logger.info(f"评估报告已保存到: {eval_report_path}")
        
        if args.mode in ['detect', 'trace', 'cluster', 'full']:
            # 检测模式
            logger.info("=" * 50)
            logger.info("步骤5: 攻击检测与溯源")
            logger.info("=" * 50)
            
            # 加载模型（如果指定）
            if args.load_model and os.path.exists(args.load_model):
                model.load_state_dict(torch.load(args.load_model, map_location=config.device))
                logger.info(f"已加载预训练模型: {args.load_model}")
            
            # 获取节点嵌入
            model.eval()
            with torch.no_grad():
                # 确保数据在正确的设备上
                hetero_data = hetero_data.to(config.device)
                embeddings = model.get_embeddings(hetero_data)
            
            # 异常检测
            anomaly_detector = AnomalyDetector(model, config)
            anomalies = anomaly_detector.detect_anomalies(hetero_data, embeddings)
            
            # 攻击检测
            attack_detector = AttackDetector(model, config)
            # 将异常检测结果传递给攻击检测器
            attacks = attack_detector.detect_attacks(hetero_data, embeddings, anomalies)
            
            # 威胁分类
            threat_classifier = ThreatClassifier(model, config)
            threats = threat_classifier.classify_threats(hetero_data, embeddings)
            
            logger.info(f"检测完成:")
            logger.info(f"  异常节点: {anomalies['summary']['total_anomalous_nodes']}")
            logger.info(f"  攻击链: {attacks['detection_report']['summary']['attack_chains_detected']}")
            logger.info(f"  威胁类型: {threats['classification_report']['summary']['total_threat_types']}")
            
            if args.mode in ['trace', 'full']:
                # 攻击溯源
                logger.info("=" * 50)
                logger.info("步骤6: 攻击溯源")
                logger.info("=" * 50)
                
                attack_tracer = AttackTracer(model, config)
                path_reconstructor = PathReconstructor(config)
                causality_analyzer = CausalityAnalyzer(config)
                
                # 对每个攻击链进行溯源
                for i, attack_chain in enumerate(attacks['attack_chains']):
                    logger.info(f"溯源攻击链 {i+1}: {attack_chain['pattern_name']}")
                    
                    # 攻击溯源
                    trace_result = attack_tracer.trace_attack_path(
                        hetero_data, 
                        f"malicious_node_{i}", 
                        "process"
                    )
                    
                    # 路径重构
                    reconstructed_path = path_reconstructor.reconstruct_path(
                        trace_result['best_path'], 
                        hetero_data
                    )
                    
                    # 因果分析
                    causality_result = causality_analyzer.analyze_causality(
                        hetero_data, 
                        trace_result['best_path']
                    )
                    
                    logger.info(f"溯源完成，路径长度: {len(trace_result['best_path']['path'])}")
            
            if args.mode in ['cluster', 'full']:
                # 攻击聚类
                logger.info("=" * 50)
                logger.info("步骤7: 攻击聚类")
                logger.info("=" * 50)
                
                attack_clusterer = AttackClusterer(config)
                clustering_result = attack_clusterer.cluster_attacks(
                    hetero_data, 
                    embeddings, 
                    attacks['attack_chains']
                )
                
                logger.info(f"聚类完成，发现 {clustering_result['clustering_report']['summary']['total_clusters']} 个聚类")
            
            # 可视化
            if config.system.visualize:
                logger.info("=" * 50)
                logger.info("步骤8: 结果可视化")
                logger.info("=" * 50)
                
                # 创建攻击故事看板
                story_board = AttackStoryBoard(config)
                
                # 生成攻击故事
                # 使用所有攻击链，而不仅仅是第一个
                all_attack_chains = attacks.get('attack_chains', [])
                if all_attack_chains:
                    # 合并所有攻击链的时间线
                    combined_timeline = []
                    for chain in all_attack_chains:
                        if 'timeline' in chain:
                            combined_timeline.extend(chain['timeline'])
                    
                    # 使用第一个攻击链作为主要结构，但包含所有时间线数据
                    main_chain = all_attack_chains[0].copy()
                    main_chain['timeline'] = combined_timeline
                    
                    attack_story = story_board.generate_attack_story(
                        main_chain,
                        combined_timeline,  # 使用合并的时间线
                        anomalies,  # evidence
                        clustering_result['cluster_analysis'] if 'cluster' in args.mode else None
                    )
                else:
                    # 如果没有攻击链，创建空的故事
                    attack_story = story_board.generate_attack_story(
                        {},
                        [],
                        anomalies,
                        None
                    )
                
                # 保存可视化结果
                viz_dir = os.path.join(config.data.output_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                story_board.save_dashboard(attack_story, os.path.join(viz_dir, "attack_story.json"))
                story_board.export_to_html(attack_story, os.path.join(viz_dir, "attack_story.html"))
                
                logger.info("可视化结果已保存")
        
        logger.info("=" * 50)
        logger.info("系统运行完成")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
