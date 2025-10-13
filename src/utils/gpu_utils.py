"""
GPU工具

提供GPU检测、优化和监控功能
"""

import torch
import logging
from typing import Dict, Any, Optional, List
import psutil
import subprocess
import os


class GPUUtils:
    """GPU工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_info = self._get_gpu_info()
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'current_device': None,
            'devices': []
        }
        
        if gpu_info['cuda_available']:
            gpu_info['current_device'] = torch.cuda.current_device()
            
            for i in range(gpu_info['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': props.total_memory,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count
                }
                
                # 安全地添加可选属性
                if hasattr(props, 'max_threads_per_block'):
                    device_info['max_threads_per_block'] = props.max_threads_per_block
                if hasattr(props, 'max_grid_size'):
                    device_info['max_grid_size'] = props.max_grid_size
                gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def print_gpu_info(self):
        """打印GPU信息"""
        print("=" * 50)
        print("GPU信息")
        print("=" * 50)
        
        if not self.gpu_info['cuda_available']:
            print("X CUDA不可用，将使用CPU训练")
            return
        
        print(f"V CUDA可用")
        print(f"CUDA版本: {self.gpu_info['cuda_version']}")
        print(f"cuDNN版本: {self.gpu_info['cudnn_version']}")
        print(f"GPU数量: {self.gpu_info['gpu_count']}")
        print(f"当前设备: {self.gpu_info['current_device']}")
        print()
        
        for device in self.gpu_info['devices']:
            print(f"GPU {device['device_id']}: {device['name']}")
            print(f"  总内存: {device['total_memory_gb']:.1f}GB")
            print(f"  计算能力: {device['major']}.{device['minor']}")
            print(f"  多处理器数量: {device['multi_processor_count']}")
            print()
    
    def get_gpu_memory_usage(self, device_id: Optional[int] = None) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return {}
        
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        # 获取内存使用情况
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)     # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'usage_percent': (reserved / total) * 100
        }
    
    def optimize_gpu_settings(self):
        """优化GPU设置"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，跳过GPU优化")
            return
        
        # 启用cuDNN基准测试
        torch.backends.cudnn.benchmark = True
        self.logger.info("已启用cuDNN基准测试")
        
        # 设置内存分配策略
        torch.cuda.empty_cache()
        self.logger.info("已清理GPU缓存")
        
        # 设置内存分配策略为增长
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        self.logger.info("已设置CUDA内存分配策略")
    
    def check_gpu_health(self) -> Dict[str, Any]:
        """检查GPU健康状态"""
        health_status = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_usage': {},
            'temperature': {},
            'warnings': [],
            'recommendations': []
        }
        
        if not health_status['cuda_available']:
            health_status['warnings'].append("CUDA不可用")
            return health_status
        
        # 检查每个GPU
        for i in range(health_status['gpu_count']):
            try:
                # 内存使用情况
                memory_usage = self.get_gpu_memory_usage(i)
                health_status['memory_usage'][f'gpu_{i}'] = memory_usage
                
                # 检查内存使用率
                if memory_usage['usage_percent'] > 90:
                    health_status['warnings'].append(f"GPU {i} 内存使用率过高: {memory_usage['usage_percent']:.1f}%")
                
                # 检查可用内存
                if memory_usage['free_gb'] < 1.0:
                    health_status['warnings'].append(f"GPU {i} 可用内存不足: {memory_usage['free_gb']:.1f}GB")
                
            except Exception as e:
                health_status['warnings'].append(f"检查GPU {i} 时出错: {e}")
        
        # 生成建议
        if health_status['memory_usage']:
            total_allocated = sum(usage['allocated_gb'] for usage in health_status['memory_usage'].values())
            if total_allocated > 8:  # 8GB
                health_status['recommendations'].append("GPU内存使用较高，建议减少批次大小")
            
            if any(usage['usage_percent'] > 80 for usage in health_status['memory_usage'].values()):
                health_status['recommendations'].append("GPU内存使用率较高，建议启用内存优化模式")
        
        return health_status
    
    def get_nvidia_smi_info(self) -> Dict[str, Any]:
        """获取nvidia-smi信息"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                
                for i, line in enumerate(lines):
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        gpu_data.append({
                            'gpu_id': i,
                            'name': parts[0],
                            'memory_total': int(parts[1]),
                            'memory_used': int(parts[2]),
                            'memory_free': int(parts[3]),
                            'temperature': int(parts[4]) if parts[4] != 'N/A' else None,
                            'utilization': int(parts[5]) if parts[5] != 'N/A' else None
                        })
                
                return {'success': True, 'data': gpu_data}
            else:
                return {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'nvidia-smi超时'}
        except FileNotFoundError:
            return {'success': False, 'error': 'nvidia-smi未找到'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def print_gpu_status(self):
        """打印GPU状态"""
        print("=" * 50)
        print("GPU状态")
        print("=" * 50)
        
        # PyTorch GPU信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_usage = self.get_gpu_memory_usage(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  内存: {memory_usage['allocated_gb']:.1f}GB / {memory_usage['total_gb']:.1f}GB "
                      f"({memory_usage['usage_percent']:.1f}%)")
                print(f"  缓存: {memory_usage['reserved_gb']:.1f}GB")
                print(f"  可用: {memory_usage['free_gb']:.1f}GB")
                print()
        else:
            print("X CUDA不可用")
        
        # nvidia-smi信息
        nvidia_info = self.get_nvidia_smi_info()
        if nvidia_info['success']:
            print("nvidia-smi信息:")
            for gpu in nvidia_info['data']:
                print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
                print(f"  内存: {gpu['memory_used']}MB / {gpu['memory_total']}MB "
                      f"({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                if gpu['temperature'] is not None:
                    print(f"  温度: {gpu['temperature']}°C")
                if gpu['utilization'] is not None:
                    print(f"  利用率: {gpu['utilization']}%")
                print()
        else:
            print(f"nvidia-smi信息获取失败: {nvidia_info['error']}")
    
    def recommend_optimal_settings(self, available_memory_gb: float) -> Dict[str, Any]:
        """根据可用内存推荐最优设置"""
        recommendations = {
            'batch_size': 16,
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'sample_size': 20000,
            'epochs': 50
        }
        
        if available_memory_gb < 4:
            # 低内存配置
            recommendations.update({
                'batch_size': 4,
                'hidden_dim': 32,
                'num_heads': 2,
                'num_layers': 1,
                'sample_size': 10000,
                'epochs': 20
            })
        elif available_memory_gb < 8:
            # 中等内存配置
            recommendations.update({
                'batch_size': 8,
                'hidden_dim': 48,
                'num_heads': 3,
                'num_layers': 1,
                'sample_size': 15000,
                'epochs': 30
            })
        elif available_memory_gb >= 16:
            # 高内存配置
            recommendations.update({
                'batch_size': 32,
                'hidden_dim': 128,
                'num_heads': 8,
                'num_layers': 3,
                'sample_size': 50000,
                'epochs': 100
            })
        
        return recommendations


# 全局GPU工具实例
gpu_utils = GPUUtils()
