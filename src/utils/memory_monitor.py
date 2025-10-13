"""
内存监控工具

提供内存使用监控和优化建议
"""

import psutil
import gc
import torch
import logging
from typing import Dict, Any, Optional
import time
from datetime import datetime


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, threshold: float = 0.8, log_interval: int = 10):
        """
        初始化内存监控器
        
        Args:
            threshold: 内存使用阈值（0-1）
            log_interval: 日志记录间隔（秒）
        """
        self.threshold = threshold
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.last_log_time = 0
        self.memory_history = []
        
    def get_memory_info(self) -> Dict[str, Any]:
        """获取详细的内存信息"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # GPU内存信息
            gpu_memory = {}
            if torch.cuda.is_available():
                gpu_memory = {
                    'gpu_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                    'gpu_cached': torch.cuda.memory_reserved() / (1024**3),     # GB
                    'gpu_max_allocated': torch.cuda.max_memory_allocated() / (1024**3),  # GB
                }
            
            return {
                'total_memory': memory.total / (1024**3),  # GB
                'available_memory': memory.available / (1024**3),  # GB
                'used_memory': memory.used / (1024**3),  # GB
                'memory_percent': memory.percent,
                'swap_total': swap.total / (1024**3),  # GB
                'swap_used': swap.used / (1024**3),  # GB
                'swap_percent': swap.percent,
                'gpu_memory': gpu_memory,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
            return {}
    
    def check_memory_usage(self) -> bool:
        """检查内存使用是否超过阈值"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return False
        
        memory_percent = memory_info.get('memory_percent', 0)
        is_over_threshold = memory_percent > (self.threshold * 100)
        
        # 记录内存历史
        self.memory_history.append(memory_info)
        if len(self.memory_history) > 100:  # 保留最近100条记录
            self.memory_history = self.memory_history[-100:]
        
        # 定期记录内存使用情况
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self._log_memory_status(memory_info, is_over_threshold)
            self.last_log_time = current_time
        
        return is_over_threshold
    
    def _log_memory_status(self, memory_info: Dict[str, Any], is_over_threshold: bool):
        """记录内存状态"""
        status = "警告" if is_over_threshold else "正常"
        
        self.logger.info(
            f"内存状态: {status} | "
            f"使用率: {memory_info.get('memory_percent', 0):.1f}% | "
            f"已用: {memory_info.get('used_memory', 0):.1f}GB | "
            f"可用: {memory_info.get('available_memory', 0):.1f}GB"
        )
        
        # GPU内存信息
        gpu_memory = memory_info.get('gpu_memory', {})
        if gpu_memory:
            self.logger.info(
                f"GPU内存: 已分配 {gpu_memory.get('gpu_allocated', 0):.1f}GB | "
                f"已缓存 {gpu_memory.get('gpu_cached', 0):.1f}GB | "
                f"最大分配 {gpu_memory.get('gpu_max_allocated', 0):.1f}GB"
            )
    
    def optimize_memory(self) -> Dict[str, Any]:
        """执行内存优化"""
        self.logger.info("开始内存优化...")
        
        optimization_results = {
            'before': self.get_memory_info(),
            'actions_taken': [],
            'after': {}
        }
        
        # 1. Python垃圾回收
        before_gc = self.get_memory_info()
        gc.collect()
        after_gc = self.get_memory_info()
        
        if after_gc and before_gc:
            memory_freed = before_gc.get('used_memory', 0) - after_gc.get('used_memory', 0)
            if memory_freed > 0:
                optimization_results['actions_taken'].append(f"Python垃圾回收释放了 {memory_freed:.2f}GB")
        
        # 2. PyTorch缓存清理
        if torch.cuda.is_available():
            before_torch = self.get_memory_info()
            torch.cuda.empty_cache()
            after_torch = self.get_memory_info()
            
            if after_torch and before_torch:
                gpu_freed = (before_torch.get('gpu_memory', {}).get('gpu_cached', 0) - 
                            after_torch.get('gpu_memory', {}).get('gpu_cached', 0))
                if gpu_freed > 0:
                    optimization_results['actions_taken'].append(f"PyTorch缓存清理释放了 {gpu_freed:.2f}GB GPU内存")
        
        # 3. 清理内存历史
        if len(self.memory_history) > 50:
            self.memory_history = self.memory_history[-25:]
            optimization_results['actions_taken'].append("清理了内存历史记录")
        
        optimization_results['after'] = self.get_memory_info()
        
        self.logger.info(f"内存优化完成，执行了 {len(optimization_results['actions_taken'])} 项优化")
        for action in optimization_results['actions_taken']:
            self.logger.info(f"  - {action}")
        
        return optimization_results
    
    def get_memory_recommendations(self) -> list:
        """获取内存优化建议"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return ["无法获取内存信息"]
        
        recommendations = []
        memory_percent = memory_info.get('memory_percent', 0)
        
        if memory_percent > 90:
            recommendations.append("内存使用率过高(>90%)，建议立即减少批次大小或数据量")
        elif memory_percent > 80:
            recommendations.append("内存使用率较高(>80%)，建议减少模型参数或数据采样")
        elif memory_percent > 70:
            recommendations.append("内存使用率中等(>70%)，建议监控内存使用情况")
        
        # GPU内存建议
        gpu_memory = memory_info.get('gpu_memory', {})
        if gpu_memory:
            gpu_allocated = gpu_memory.get('gpu_allocated', 0)
            if gpu_allocated > 8:  # 8GB
                recommendations.append("GPU内存使用较高，建议减少模型大小或使用CPU训练")
        
        # 交换空间建议
        swap_percent = memory_info.get('swap_percent', 0)
        if swap_percent > 50:
            recommendations.append("交换空间使用率较高，系统可能内存不足")
        
        return recommendations
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return {}
        
        # 计算内存使用趋势
        if len(self.memory_history) >= 2:
            recent_memory = [h.get('memory_percent', 0) for h in self.memory_history[-5:]]
            trend = "上升" if recent_memory[-1] > recent_memory[0] else "下降"
        else:
            trend = "未知"
        
        return {
            'current_usage': memory_info.get('memory_percent', 0),
            'total_memory': memory_info.get('total_memory', 0),
            'available_memory': memory_info.get('available_memory', 0),
            'trend': trend,
            'recommendations': self.get_memory_recommendations(),
            'is_critical': memory_info.get('memory_percent', 0) > (self.threshold * 100)
        }


# 全局内存监控器实例
memory_monitor = MemoryMonitor()
