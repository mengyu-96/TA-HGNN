# T-HGNN项目最终结构

## 📁 项目文件结构

```
TA-HGNN/
├── main.py                                    # 主程序入口
├── run_gpu_optimized.py                       # GPU优化运行脚本
├── run_memory_optimized.py                    # 内存优化运行脚本
├── check_gpu.py                               # GPU检查工具
├── setup_encoding.py                          # 编码设置工具
├── README.md                                  # 项目说明文档
├── requirements.txt                           # 依赖管理
├── PROJECT_FINAL_STRUCTURE.md                # 项目结构文档
├── GPU_USAGE_GUIDE.md                         # GPU使用指南
├── MEMORY_OPTIMIZATION.md                     # 内存优化指南
├── Linux-APT-Dataset/                        # 真实数据集
│   └── Linux-APT-Dataset-2024/
│       ├── combine.csv                       # 原始数据 (91,133条记录)
│       └── Processed Version.xlsx            # 处理后数据
├── output/                                   # 输出结果目录
│   ├── results/                              # 评估结果
│   │   ├── evaluation_results.json           # 详细评估结果
│   │   └── evaluation_report.txt             # 评估报告
│   ├── models/                               # 训练好的模型
│   ├── plots/                                # 图表文件
│   ├── visualizations/                       # 可视化结果
│   ├── experiment_results/                   # 实验结果
│   │   ├── paper_tables/                     # 论文表格
│   │   └── visualizations/                   # 实验可视化
│   └── logs/                                 # 运行日志
├── src/                                      # 核心系统代码
│   ├── core/                                 # 核心模型
│   │   ├── models/                           # T-HGNN模型
│   │   │   ├── t_hgnn.py                     # 主模型
│   │   │   ├── hgnn_encoder.py               # 异质图编码器
│   │   │   ├── temporal_encoder.py           # 时序编码器
│   │   │   └── node_classifier.py            # 节点分类器
│   │   └── training/                         # 训练模块
│   │       ├── trainer.py                    # 训练器
│   │       └── evaluator.py                  # 评估器
│   ├── applications/                         # 应用模块
│   │   ├── detection/                        # 异常检测
│   │   │   ├── anomaly_detector.py           # 异常检测器
│   │   │   ├── attack_detector.py            # 攻击检测器
│   │   │   └── threat_classifier.py          # 威胁分类器
│   │   ├── tracing/                          # 攻击溯源
│   │   │   ├── attack_tracer.py              # 攻击追踪器
│   │   │   ├── causality_analyzer.py         # 因果分析器
│   │   │   └── path_reconstructor.py         # 路径重构器
│   │   └── clustering/                       # 攻击聚类
│   │       ├── attack_clusterer.py           # 攻击聚类器
│   │       ├── activity_attributor.py        # 活动归因器
│   │       └── pattern_analyzer.py           # 模式分析器
│   ├── data/                                 # 数据处理
│   │   ├── pyg_loader.py                     # PyG数据加载器
│   │   ├── apt_data_processor.py             # APT数据处理器
│   │   ├── data_quality_evaluator.py         # 数据质量评估器
│   │   └── entity_resolver.py                # 实体解析器
│   ├── evaluation/                           # 评估模块
│   │   ├── node_classification_evaluator.py  # 节点分类评估器
│   │   ├── attack_grouping_evaluator.py      # 攻击分组评估器
│   │   ├── path_tracing_evaluator.py         # 路径溯源评估器
│   │   └── metrics_calculator.py             # 指标计算器
│   ├── config/                               # 配置模块
│   │   ├── config.py                         # 主配置
│   │   └── memory_optimized_config.py        # 内存优化配置
│   ├── utils/                                # 工具函数
│   │   ├── gpu_utils.py                      # GPU工具
│   │   ├── memory_monitor.py                 # 内存监控
│   │   ├── label_generator.py                # 标签生成器
│   │   ├── apt_detector.py                   # APT检测器
│   │   └── visualization.py                  # 可视化工具
│   └── visualization/                        # 可视化模块
│       └── dashboards/                       # 仪表板
│           ├── threat_dashboard.py           # 威胁仪表板
│           ├── attack_story_board.py         # 攻击故事板
│           └── real_time_monitor.py          # 实时监控
├── examples/                                 # 使用示例
│   ├── README.md                             # 示例说明
│   └── temporal_graph_demo.py               # 时序图演示
├── docs/                                     # 详细文档
│   ├── EXPERIMENT_METRICS_DESIGN.md          # 实验指标设计
│   └── TEMPORAL_HETEROGENEOUS_GRAPH_DESIGN.md # 时序异构图设计
├── tests/                                    # 测试代码
├── scripts/                                  # 安装脚本
│   └── install.py                            # 安装脚本
└── 项目大纲                                   # 项目大纲文档
```

## 🎯 核心文件说明

### 主程序文件
- **main.py**: 系统主入口，支持完整工作流程
- **production_test_with_real_dataset.py**: 生产级测试脚本

### 核心模块
- **src/core/models/**: T-HGNN核心模型实现
- **src/applications/**: 异常检测、攻击溯源、聚类等应用
- **src/data/**: 数据处理和异构图构建
- **src/visualization/**: 可视化组件

### 数据文件
- **Linux-APT-Dataset/**: 真实APT数据集
- **output/**: 系统输出结果

## ✅ 文件有效性确认

### 核心文件状态
- ✅ main.py - 主程序入口
- ✅ production_test_with_real_dataset.py - 生产级测试
- ✅ README.md - 项目文档
- ✅ requirements.txt - 依赖管理

### 核心模块状态
- ✅ src/core/models/ - 核心模型
- ✅ src/applications/ - 应用模块
- ✅ src/data/ - 数据处理
- ✅ src/visualization/ - 可视化

### 数据文件状态
- ✅ Linux-APT-Dataset/ - 真实数据集
- ✅ output/ - 输出结果


