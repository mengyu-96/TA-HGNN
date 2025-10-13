#!/usr/bin/env python3
"""
项目依赖安装脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装requirements.txt中的依赖"""
    try:
        print("正在安装项目依赖...")
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        requirements_file = os.path.join(project_root, "requirements.txt")
        
        # 检查requirements.txt是否存在
        if not os.path.exists(requirements_file):
            print("❌ requirements.txt文件不存在")
            return False
        
        # 安装依赖
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖安装成功！")
            print("安装输出:")
            print(result.stdout)
            return True
        else:
            print("❌ 依赖安装失败！")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 安装过程中出现错误: {e}")
        return False

def check_installation():
    """检查关键依赖是否安装成功"""
    try:
        print("\n正在检查关键依赖...")
        
        # 检查torch
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        # 检查torch-geometric
        import torch_geometric
        print(f"✓ PyTorch Geometric版本: {torch_geometric.__version__}")
        
        # 检查其他依赖
        import pandas
        print(f"✓ Pandas版本: {pandas.__version__}")
        
        import numpy
        print(f"✓ NumPy版本: {numpy.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib版本: {matplotlib.__version__}")
        
        print("✅ 所有关键依赖检查通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        return False

if __name__ == "__main__":
    print("=== T-HGNN项目依赖安装脚本 ===")
    
    # 安装依赖
    if install_requirements():
        # 检查安装
        if check_installation():
            print("\n🎉 所有依赖安装完成！现在可以运行项目了：")
            print("  python main_pyg.py")
        else:
            print("\n⚠️ 依赖安装完成，但检查时发现问题。")
    else:
        print("\n❌ 依赖安装失败，请手动安装。")

