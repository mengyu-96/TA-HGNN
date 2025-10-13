#!/usr/bin/env python3
"""
编码设置脚本

解决Windows终端中文显示乱码问题
"""

import os
import sys
import subprocess
import platform

def setup_encoding():
    """设置正确的编码"""
    print("=" * 50)
    print("设置终端编码")
    print("=" * 50)
    
    # 设置Python编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # 设置系统编码
    if platform.system() == 'Windows':
        try:
            # 设置代码页为UTF-8
            subprocess.run(['chcp', '65001'], shell=True, check=True)
            print("+ 已设置代码页为UTF-8 (65001)")
        except subprocess.CalledProcessError:
            print("[WARNING] 无法设置代码页，请手动运行: chcp 65001")
    
    # 设置环境变量
    env_vars = {
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUTF8': '1',
        'LANG': 'zh_CN.UTF-8',
        'LC_ALL': 'zh_CN.UTF-8'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"+ 已设置环境变量: {key}={value}")
    
    print("\n编码设置完成！")
    print("如果仍有乱码，请尝试以下方法：")
    print("1. 重启终端")
    print("2. 在PowerShell中运行: chcp 65001")
    print("3. 设置终端字体为支持中文的字体（如Consolas、Microsoft YaHei Mono）")

def test_encoding():
    """测试编码设置"""
    print("\n" + "=" * 50)
    print("测试中文显示")
    print("=" * 50)
    
    test_strings = [
        "+ 中文显示测试",
        "[GPU] GPU加速训练",
        "[CHART] 性能监控",
        "[MEMORY] 内存优化",
        "[CONFIG] 设备配置"
    ]
    
    for s in test_strings:
        print(s)
    
    print("\n如果上述字符显示正常，说明编码设置成功！")

if __name__ == "__main__":
    setup_encoding()
    test_encoding()
