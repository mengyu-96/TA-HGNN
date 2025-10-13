"""
恶意节点分类指标评估器

实现任务一：恶意节点分类的完整指标评估
包括准确率、精确率、召回率、F1-Score、AUC-ROC等指标
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict


class NodeClassificationEvaluator:
    """恶意节点分类评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def evaluate_single_model(self, 
                            y_true: torch.Tensor, 
                            y_pred: torch.Tensor,
                            y_prob: torch.Tensor = None,
                            model_name: str = "Model") -> Dict[str, float]:
        """
        评估单个模型的分类性能
        
        Args:
            y_true: 真实标签 (N,)
            y_pred: 预测标签 (N,)
            y_prob: 预测概率 (N, 2) 或 (N,)
            model_name: 模型名称
            
        Returns:
            评估指标字典
        """
        # 转换为numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        # 确保标签是二分类
        y_true = (y_true > 0).astype(int)
        y_pred = (y_pred > 0).astype(int)
        
        # 计算基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算AUC-ROC
        auc_roc = 0.0
        if y_prob is not None:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # 多类概率，取正类概率
                y_prob_positive = y_prob[:, 1]
            else:
                # 单类概率
                y_prob_positive = y_prob
            try:
                auc_roc = roc_auc_score(y_true, y_prob_positive)
            except ValueError:
                self.logger.warning(f"无法计算AUC-ROC，可能所有标签都是同一类")
                auc_roc = 0.5
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 计算额外指标
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算类别分布
        total_samples = len(y_true)
        positive_samples = np.sum(y_true)
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0.0
        
        # 计算额外的高级指标
        try:
            if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                avg_prec_score = average_precision_score(y_true, y_prob[:, 1])
            else:
                avg_prec_score = precision
        except Exception as e:
            self.logger.warning(f"计算平均精确率时出错: {e}")
            avg_prec_score = precision
        
        # 计算F-beta分数 (F0.5和F2)
        f05_score = self._compute_fbeta_score(y_true, y_pred, beta=0.5)
        f2_score = self._compute_fbeta_score(y_true, y_pred, beta=2.0)
        
        # 计算MCC (Matthews Correlation Coefficient)
        mcc = self._compute_mcc(y_true, y_pred)
        
        # 计算平衡准确率
        balanced_accuracy = (recall + specificity) / 2
        
        # 计算G-means
        g_means = np.sqrt(recall * specificity)
        
        # 计算额外的临床诊断指标
        negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # 负预测值 (NPV)
        positive_likelihood_ratio = recall / (1 - specificity) if (1 - specificity) > 0 else 0.0  # 阳性似然比 (PLR)
        negative_likelihood_ratio = (1 - recall) / specificity if specificity > 0 else 0.0  # 阴性似然比 (NLR)
        diagnostic_odds_ratio = positive_likelihood_ratio / negative_likelihood_ratio if negative_likelihood_ratio > 0 else 0.0  # 诊断优势比 (DOR)
        youden_index = recall + specificity - 1  # 约登指数 (Youden's Index)
        
        # 计算Cohen's Kappa（用于一致性评估）
        observed_agreement = accuracy
        expected_agreement = (tp + fn) / (tp + fn + fp + tn) * (tp + fp) / (tp + fn + fp + tn) + (tn + fp) / (tp + fn + fp + tn) * (tn + fn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0
        cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 0
        
        # 计算马修斯相关系数平方 (MCC^2)
        mcc_squared = mcc ** 2
        
        # 计算错分类成本敏感指标
        false_positive_cost = fp * 0.1  # 误报成本
        false_negative_cost = fn * 1.0  # 漏报成本（更高成本）
        total_cost = false_positive_cost + false_negative_cost
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'average_precision': avg_prec_score,
            'f0.5_score': f05_score,
            'f2_score': f2_score,
            'mcc': mcc,
            'mcc_squared': mcc_squared,
            'cohen_kappa': cohen_kappa,
            'g_means': g_means,
            'negative_predictive_value': negative_predictive_value,
            'positive_likelihood_ratio': positive_likelihood_ratio,
            'negative_likelihood_ratio': negative_likelihood_ratio,
            'diagnostic_odds_ratio': diagnostic_odds_ratio,
            'youden_index': youden_index,
            'false_positive_cost': false_positive_cost,
            'false_negative_cost': false_negative_cost,
            'total_cost': total_cost,
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': positive_ratio,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            }
        }
        
        self.logger.info(f"{model_name} 分类性能:")
        self.logger.info(f"  准确率: {accuracy:.4f}")
        self.logger.info(f"  精确率: {precision:.4f}")
        self.logger.info(f"  召回率: {recall:.4f}")
        self.logger.info(f"  F1-Score: {f1:.4f}")
        self.logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"  正样本比例: {positive_ratio:.4f}")
        
        return results
    
    def evaluate_by_node_type(self, 
                            node_predictions: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        """
        按节点类型分别评估
        
        Args:
            node_predictions: {
                'node_type': {
                    'y_true': tensor,
                    'y_pred': tensor,
                    'y_prob': tensor (optional)
                }
            }
            
        Returns:
            按节点类型的评估结果
        """
        results_by_type = {}
        
        for node_type, predictions in node_predictions.items():
            self.logger.info(f"评估节点类型: {node_type}")
            
            y_true = predictions.get('y_true')
            y_pred = predictions.get('y_pred')
            y_prob = predictions.get('y_prob')
            
            if y_true is None or y_pred is None:
                self.logger.warning(f"节点类型 {node_type} 缺少必要数据")
                continue
            
            # 评估该节点类型
            type_results = self.evaluate_single_model(
                y_true, y_pred, y_prob, f"{node_type}_classification"
            )
            results_by_type[node_type] = type_results
        
        return results_by_type
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        对比多个模型的性能
        
        Args:
            model_results: {
                'model_name': {
                    'y_true': tensor,
                    'y_pred': tensor,
                    'y_prob': tensor (optional)
                }
            }
            
        Returns:
            模型对比结果
        """
        comparison_results = {}
        
        for model_name, predictions in model_results.items():
            self.logger.info(f"评估模型: {model_name}")
            
            y_true = predictions.get('y_true')
            y_pred = predictions.get('y_pred')
            y_prob = predictions.get('y_prob')
            
            if y_true is None or y_pred is None:
                self.logger.warning(f"模型 {model_name} 缺少必要数据")
                continue
            
            # 评估该模型
            model_results_dict = self.evaluate_single_model(
                y_true, y_pred, y_prob, model_name
            )
            comparison_results[model_name] = model_results_dict
        
        # 生成对比表格
        comparison_table = self._generate_comparison_table(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'comparison_table': comparison_table,
            'best_model': self._find_best_model(comparison_results)
        }
    
    def _generate_comparison_table(self, results: Dict[str, Dict[str, float]]) -> Dict[str, List]:
        """生成对比表格"""
        table = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': [],
            'Positive_Ratio': []
        }
        
        for model_name, metrics in results.items():
            table['Model'].append(model_name)
            table['Accuracy'].append(f"{metrics['accuracy']:.4f}")
            table['Precision'].append(f"{metrics['precision']:.4f}")
            table['Recall'].append(f"{metrics['recall']:.4f}")
            table['F1-Score'].append(f"{metrics['f1_score']:.4f}")
            table['AUC-ROC'].append(f"{metrics['auc_roc']:.4f}")
            table['Positive_Ratio'].append(f"{metrics['positive_ratio']:.4f}")
        
        return table
    
    def _find_best_model(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """找到最佳模型"""
        best_model = None
        best_f1 = -1
        
        for model_name, metrics in results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model = model_name
        
        return {
            'model_name': best_model,
            'f1_score': best_f1,
            'metrics': results.get(best_model, {})
        }
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """生成详细的评估报告"""
        report = []
        report.append("=" * 60)
        report.append("恶意节点分类评估报告")
        report.append("=" * 60)
        
        if 'individual_results' in results:
            # 对比多个模型
            for model_name, metrics in results['individual_results'].items():
                report.append(f"\n{model_name} 详细指标:")
                report.append(f"  准确率: {metrics['accuracy']:.4f}")
                report.append(f"  精确率: {metrics['precision']:.4f}")
                report.append(f"  召回率: {metrics['recall']:.4f}")
                report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
                report.append(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                report.append(f"  特异性: {metrics['specificity']:.4f}")
                report.append(f"  敏感性: {metrics['sensitivity']:.4f}")
                report.append(f"  样本分布: {metrics['positive_samples']}/{metrics['total_samples']} ({metrics['positive_ratio']:.4f})")
                
                cm = metrics['confusion_matrix']
                report.append(f"  混淆矩阵: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
            
            # 最佳模型
            if 'best_model' in results:
                best = results['best_model']
                report.append(f"\n最佳模型: {best['model_name']} (F1-Score: {best['f1_score']:.4f})")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """保存评估结果"""
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {filepath}")
    
    def _compute_fbeta_score(self, y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
        """计算F_beta分数"""
        # 确保输入是numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.cpu().numpy()
        else:
            y_pred_np = y_pred
            
        tn, fp, fn, tp = self._compute_confusion_matrix_from_np(y_true_np, y_pred_np)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision == 0 and recall == 0:
            return 0.0
        
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        return f_beta
    
    def _compute_mcc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """计算Matthews Correlation Coefficient"""
        # 确保输入是numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.cpu().numpy()
        else:
            y_pred_np = y_pred
            
        tn, fp, fn, tp = self._compute_confusion_matrix_from_np(y_true_np, y_pred_np)
        
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def average_precision_score(self, y_true, y_score):
        """计算平均精确率"""
        try:
            from sklearn.metrics import average_precision_score as sk_avg_precision
            return sk_avg_precision(y_true, y_score)
        except:
            # 简化计算：如果sklearn不可用，使用精确率
            tn, fp, fn, tp = self._compute_confusion_matrix_from_np(y_true, y_score > 0.5)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _compute_confusion_matrix_from_np(self, y_true, y_pred):
        """从numpy数组计算混淆矩阵"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return tn, fp, fn, tp


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建评估器
    evaluator = NodeClassificationEvaluator()
    
    # 模拟数据
    y_true = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1])
    y_prob = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6],
                          [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])
    
    # 评估单个模型
    results = evaluator.evaluate_single_model(y_true, y_pred, y_prob, "Test_Model")
    print(evaluator.generate_detailed_report({'individual_results': {'Test_Model': results}}))


