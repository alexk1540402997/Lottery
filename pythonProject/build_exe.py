"""
PyInstaller 打包脚本
使用方法: python build_exe.py

生成单个EXE文件：彩票预测系统3.0.exe
"""

import os
import sys
import subprocess
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    missing = []
    for module in ['pandas', 'numpy', 'openpyxl', 'sklearn', 'tkinter']:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - 未安装")
            if module == 'sklearn':
                missing.append('scikit-learn')
            elif module == 'tkinter':
                print("    tkinter是Python标准库，请确保Python安装完整")
            else:
                missing.append(module)

    if missing:
        print(f"\n安装缺失依赖: pip install {' '.join(missing)}")
        return False
    return True


def clean_build():
    """清理旧的构建文件"""
    for d in ['build', 'dist', '__pycache__']:
        path = os.path.join(PROJECT_DIR, d)
        if os.path.exists(path):
            print(f"清理: {d}")
            shutil.rmtree(path)


def build_exe():
    """构建EXE"""
    print("\n" + "=" * 60)
    print("  彩票预测系统 3.0 - PyInstaller 打包")
    print("=" * 60)

    if not check_dependencies():
        print("\n请先安装依赖后再运行本脚本")
        return 1

    clean_build()

    # 主入口文件
    main_script = os.path.join(PROJECT_DIR, "MainSystem.py")

    if not os.path.exists(main_script):
        print(f"错误: 找不到主入口文件 {main_script}")
        return 1

    # 图标文件（如果有的话）
    icon_file = os.path.join(PROJECT_DIR, "icon.ico")
    icon_args = []
    if os.path.exists(icon_file):
        icon_args = ['--icon', icon_file]

    # PyInstaller 命令
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',           # 单个EXE文件
        '--console',           # 控制台窗口（可看到日志输出）
        '--name', '彩票预测系统3.0',
        '--add-data', f'{PROJECT_DIR}/*.py{os.pathsep}.',
        '--hidden-import', 'sklearn.ensemble',
        '--hidden-import', 'sklearn.cluster',
        '--hidden-import', 'sklearn.preprocessing',
        '--hidden-import', 'sklearn.model_selection',
        '--hidden-import', 'openpyxl',
        '--hidden-import', 'pandas',
        '--hidden-import', 'numpy',
        '--hidden-import', 'tkinter',
        '--hidden-import', 'multiprocessing',
        '--hidden-import', 'concurrent.futures',
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'scipy',  # 可选依赖
        '--clean',
        '--noconfirm',
    ] + icon_args + [main_script]

    print("\n构建命令:")
    print(' '.join(cmd))
    print()

    # 运行构建
    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode == 0:
        exe_path = os.path.join(PROJECT_DIR, 'dist', '彩票预测系统3.0.exe')
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"  ✅ 打包成功!")
            print(f"  📦 {exe_path}")
            print(f"  📏 文件大小: {size_mb:.1f} MB")
            print(f"{'='*60}")
            return 0

    print("\n打包失败，请检查错误信息")
    return 1


if __name__ == "__main__":
    sys.exit(build_exe())
