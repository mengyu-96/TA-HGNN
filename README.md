# T-HGNN: 基于时序异质图神经网络的APT攻击检测与溯源系统

## 🎯 项目概述

T-HGNN是一个基于时序异质图神经网络的APT攻击检测与溯源系统，能够处理大规模安全日志数据，进行异常检测、攻击链识别和可视化分析。

## ✨ 核心特性

- **大规模数据处理**: 支持91,000+条安全日志记录
- **异构图构建**: 63,000+节点，237,000+边的复杂网络
- **深度学习模型**: 32M参数的T-HGNN模型
- **异常检测**: 18.8%的异常检测率
- **可视化分析**: 8个完整可视化组件
- **生产就绪**: 通过完整的生产级测试

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行系统
```bash
# 使用真实数据集进行完整测试
python production_test_with_real_dataset.py

# 运行主程序
python main.py --mode full --data_path Linux-APT-Dataset/Linux-APT-Dataset-2024/combine.csv
```

## 📊 性能指标

### 数据处理能力
- **处理速度**: 43,190 记录/秒
- **内存效率**: 621 MB (91,133条记录)
- **数据质量**: 支持缺失值处理

### 模型性能
- **参数数量**: 32,082,658
- **模型大小**: 122.4 MB
- **推理速度**: 78,799 节点/秒

### 检测能力
- **异常检测率**: 18.8%
- **可视化成功率**: 100%
- **系统状态**: PRODUCTION_READY

## 🏗️ 系统架构

```
T-HGNN系统
├── 数据预处理层
│   ├── 数据加载 (PyG_LinuxAPTDataLoader)
│   ├── 数据清洗 (APTDataProcessor)
│   └── 特征工程
├── 核心算法层
│   ├── 异构图构建
│   ├── T-HGNN模型
│   └── 时序编码
├── 智能应用层
│   ├── 异常检测 (AnomalyDetector)
│   ├── 攻击检测 (AttackDetector)
│   ├── 攻击溯源 (AttackTracer)
│   └── 攻击聚类 (AttackClusterer)
└── 可视化层
    ├── 攻击故事看板 (AttackStoryBoard)
    ├── 实时监控 (RealTimeMonitor)
    └── 威胁仪表板 (ThreatDashboard)
```

## 📁 项目结构

```
TA-HGNN/
├── src/                          # 核心系统代码
│   ├── core/                     # 核心模型
│   ├── applications/              # 应用模块
│   ├── data/                      # 数据处理
│   ├── visualization/             # 可视化
│   └── utils/                     # 工具函数
├── Linux-APT-Dataset/            # 真实数据集
├── output/                       # 输出结果
├── examples/                     # 使用示例
├── tests/                        # 测试代码
├── docs/                         # 详细文档
├── main.py                       # 主程序入口
├── production_test_with_real_dataset.py  # 生产级测试
└── requirements.txt              # 依赖管理
```

## 🔬 科研实验标准

### 测试数据集
- **数据集**: Linux-APT-Dataset-2024
- **数据量**: 91,133条记录
- **数据维度**: 123个原始列，148个处理后特征
- **数据质量**: 真实安全日志数据

### 实验指标
- **异常检测率**: 18.8%
- **检测覆盖率**: 11,831个异常节点
- **可视化质量**: 100%成功率
- **系统稳定性**: 生产级测试通过

## 📈 使用示例

### 基本使用
```python
from src.config.simple_config import SimpleConfig
from src.data.pyg_loader import PyG_LinuxAPTDataLoader
from src.core.models.t_hgnn import T_HGNN

# 加载配置
config = SimpleConfig()

# 加载数据
loader = PyG_LinuxAPTDataLoader(config.data)
df = loader.load_data()

# 构建异构图
hetero_data = loader.build_hetero_graph(df)

# 创建模型
model = T_HGNN(config, hetero_data.node_types, hetero_data.edge_types, in_dims)

# 模型推理
embeddings = model.get_embeddings(hetero_data)
```

### 异常检测
```python
from src.applications.detection.anomaly_detector import AnomalyDetector

# 创建异常检测器
detector = AnomalyDetector(model, config)

# 执行异常检测
anomalies = detector.detect_anomalies(hetero_data, embeddings)
```

## 🎯 生产部署

### 系统要求
- **CPU**: 8核心以上
- **内存**: 16GB以上
- **GPU**: NVIDIA GPU (推荐)
- **存储**: 50GB以上

### 部署步骤
1. 安装依赖环境
2. 配置数据集路径
3. 运行生产级测试
4. 启动主程序

## 📚 文档

- [项目详细文档](docs/)
- [使用示例](examples/)
- [API文档](src/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题，请通过Issue联系我们。

---

**项目状态**: ✅ PRODUCTION_READY  
**最后更新**: 2025-09-28  
**版本**: 1.0.0
