# T-HGNN系统使用示例

本目录包含了T-HGNN系统的使用示例，帮助用户快速上手和深入了解系统功能。

## 示例文件

### 1. 基本使用示例 (`basic_usage.py`)

**用途**: 演示T-HGNN系统的基本功能和使用流程

**功能包括**:
- 数据加载和预处理
- 异构图构建
- 模型创建和推理
- 异常检测
- 攻击检测
- 攻击溯源
- 攻击聚类
- 攻击故事看板生成

**运行方法**:
```bash
cd examples
python basic_usage.py
```

**预期输出**:
- 处理后的数据统计
- 检测结果摘要
- 攻击故事看板
- 执行摘要报告

### 2. 高级使用示例 (`advanced_usage.py`)

**用途**: 演示T-HGNN系统的高级功能，包括模型训练、参数调优、性能评估等

**功能包括**:
- 高级数据生成
- 模型训练和验证
- 训练曲线可视化
- 性能指标评估
- 模型保存和加载
- 完整的检测和溯源流程
- 综合结果报告

**运行方法**:
```bash
cd examples
python advanced_usage.py
```

**预期输出**:
- 训练曲线图
- 性能评估报告
- 完整的检测结果
- 综合摘要报告

## 使用前准备

### 1. 安装依赖

确保已安装所有必要的依赖包：

```bash
pip install -r requirements.txt
```

### 2. 检查Python版本

确保使用Python 3.8或更高版本：

```bash
python --version
```

### 3. 准备数据（可选）

如果需要使用真实数据，请将数据文件放在项目根目录的`Linux-APT-Dataset/Linux-APT-Dataset-2024/`目录下。

## 示例说明

### 基本使用示例

基本使用示例展示了T-HGNN系统的核心功能，适合初学者快速了解系统能力：

1. **数据准备**: 创建模拟的Linux APT数据
2. **数据处理**: 使用APT数据处理器处理原始数据
3. **图构建**: 构建时序异构图
4. **模型推理**: 使用预训练模型进行推理
5. **检测分析**: 执行异常检测、攻击检测等
6. **结果可视化**: 生成攻击故事看板

### 高级使用示例

高级使用示例展示了系统的完整工作流程，包括模型训练和性能评估：

1. **数据生成**: 创建更复杂的模拟数据
2. **模型训练**: 训练T-HGNN模型
3. **性能评估**: 评估模型性能并生成报告
4. **完整流程**: 执行端到端的检测和溯源
5. **结果分析**: 生成综合的分析报告

## 输出文件

### 基本使用示例输出

```
output/
├── attack_story.json          # 攻击故事看板
└── execution_summary.json     # 执行摘要
```

### 高级使用示例输出

```
output_advanced/
├── trained_model.pth          # 训练好的模型
├── training_curves.png        # 训练曲线图
├── training_report.json       # 训练报告
├── performance_metrics.json   # 性能指标
├── detection_results.json     # 检测结果
├── attack_story.json          # 攻击故事看板
└── final_summary.json         # 最终摘要
```

## 自定义配置

可以通过修改示例中的配置参数来适应不同的需求：

```python
# 模型参数
config.model.hidden_dim = 128      # 隐藏层维度
config.model.num_heads = 8         # 注意力头数
config.model.num_layers = 3        # GNN层数

# 训练参数
config.training.epochs = 50        # 训练轮数
config.training.learning_rate = 0.001  # 学习率
config.training.batch_size = 32    # 批次大小

# 检测参数
config.anomaly_threshold = 0.7     # 异常检测阈值
config.attack_threshold = 0.8      # 攻击检测阈值
config.confidence_threshold = 0.7  # 置信度阈值
```

## 故障排除

### 常见问题

1. **导入错误**: 确保已安装所有依赖包
2. **内存不足**: 减少数据量或使用更小的模型
3. **训练不收敛**: 调整学习率或增加训练轮数
4. **可视化失败**: 检查matplotlib和plotly安装

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展示例

### 添加新的检测算法

```python
# 在示例中添加自定义检测算法
class CustomDetector:
    def __init__(self, config):
        self.config = config
    
    def detect(self, data):
        # 实现自定义检测逻辑
        pass
```

### 自定义可视化

```python
# 创建自定义可视化组件
def create_custom_visualization(data):
    # 实现自定义可视化逻辑
    pass
```

## 性能优化

### 1. 使用GPU加速

```python
# 在配置中启用GPU
config.training.gpu_id = 0
```

### 2. 调整批次大小

```python
# 根据内存情况调整批次大小
config.training.batch_size = 64  # 增加批次大小
```

### 3. 使用多进程

```python
# 在数据处理中使用多进程
config.data.num_workers = 4
```

## 联系支持

如果在使用示例过程中遇到问题，请：

1. 检查日志输出中的错误信息
2. 确认所有依赖包已正确安装
3. 查看项目文档和FAQ
4. 在项目Issues中报告问题

---

**注意**: 这些示例仅用于演示目的，实际使用时请根据具体需求进行调整和优化。
