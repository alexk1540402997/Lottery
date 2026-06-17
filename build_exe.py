"""
PyInstaller 打包脚本 4.0
========================
将彩票预测系统打包为单个EXE文件。

用法:
    pip install pyinstaller
    python build_exe.py
"""

import os
import sys
import subprocess
import shutil


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    """检查依赖"""
    print("=" * 60)
    print("  检查依赖...")
    print("=" * 60)
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
    # 清理spec文件
    for f in os.listdir(PROJECT_DIR):
        if f.endswith('.spec'):
            os.remove(os.path.join(PROJECT_DIR, f))
            print(f"清理: {f}")


def build_exe():
    """构建EXE"""
    print("\n" + "=" * 60)
    print("  彩票预测系统 4.0 - PyInstaller 打包")
    print("=" * 60)

    if not check_dependencies():
        print("\n请先安装依赖后再运行本脚本")
        print("  pip install pandas numpy openpyxl scikit-learn pyinstaller")
        return 1

    clean_build()

    # 主入口文件
    main_script = os.path.join(PROJECT_DIR, "main.py")

    if not os.path.exists(main_script):
        print(f"错误: 找不到主入口文件 {main_script}")
        return 1

    # PyInstaller 命令
    # 使用 --add-data 将所有.py模块文件包含进去
    data_args = []
    for f in os.listdir(PROJECT_DIR):
        if f.endswith('.py') and f != 'build_exe.py':
            data_args.extend(['--add-data',
                            f'{os.path.join(PROJECT_DIR, f)}{os.pathsep}.'])

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',              # 单个EXE文件
        '--console',              # 控制台窗口
        '--name', '彩票预测系统4.0',
        '--hidden-import', 'sklearn.ensemble._forest',
        '--hidden-import', 'sklearn.ensemble',
        '--hidden-import', 'sklearn.cluster',
        '--hidden-import', 'sklearn.preprocessing',
        '--hidden-import', 'sklearn.model_selection',
        '--hidden-import', 'sklearn.utils._typedefs',
        '--hidden-import', 'sklearn.neighbors._typedefs',
        '--hidden-import', 'openpyxl',
        '--hidden-import', 'pandas',
        '--hidden-import', 'numpy',
        '--hidden-import', 'tkinter',
        '--hidden-import', 'concurrent.futures',
        '--hidden-import', 'queue',
        '--hidden-import', 'json',
        '--hidden-import', 'hashlib',
        '--hidden-import', 'threading',
        '--hidden-import', 'importlib',
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'scipy',
        '--clean',
        '--noconfirm',
        '--collect-all', 'sklearn',
    ] + data_args + [main_script]

    print("\n构建命令:")
    print(' '.join(cmd[:20]) + ' ...')
    print()

    # 运行构建
    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode == 0:
        exe_path = os.path.join(PROJECT_DIR, 'dist', '彩票预测系统4.0.exe')
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"  ✅ 打包成功!")
            print(f"  📦 {exe_path}")
            print(f"  📏 文件大小: {size_mb:.1f} MB")
            print(f"{'='*60}")
            print(f"\n使用方法: 双击 彩票预测系统4.0.exe 启动GUI")
            print(f"或命令行: 彩票预测系统4.0.exe --predict 双色球.xlsx")
            return 0

    print("\n打包失败，请检查错误信息")
    print("常见问题:")
    print("  1. 确保已安装 pyinstaller: pip install pyinstaller")
    print("  2. 确保所有依赖已安装: pip install pandas numpy openpyxl scikit-learn")
    print("  3. 尝试以管理员权限运行")
    return 1


if __name__ == "__main__":
    sys.exit(build_exe())
