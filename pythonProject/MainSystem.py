"""
================================================================================
  彩票号码预测系统 - 统一主系统 3.0
================================================================================

  整合四大子系统：
  ① 预测引擎 (彩票号码预测系统3.0.py) - 8种方法+5种颗粒度
  ② 结果打包 (分析结果打包2.0.py)     - 多颗粒度合并+最终推荐
  ③ 回测系统 (unified_backtester_3.0.py) - 历史回测+性能评估
  ④ 优化系统 (内置)                    - 权重优化+闭环迭代

  支持模式：
  - GUI模式: python MainSystem.py
  - CLI模式: python MainSystem.py --predict data.xlsx --granularity 100
            python MainSystem.py --backtest data.xlsx
            python MainSystem.py --pipeline data.xlsx
            python MainSystem.py --merge
================================================================================
"""

import os
import sys
import argparse
import importlib.util
from datetime import datetime


# ================================================================================
#  CLI 模式
# ================================================================================

def find_module_file(pattern: str) -> str:
    """在项目目录中查找模块文件"""
    base = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(base):
        if f.endswith('.py') and pattern in f and '3.0' in f:
            return os.path.join(base, f)
    # fallback
    for f in os.listdir(base):
        if f.endswith('.py') and pattern in f:
            return os.path.join(base, f)
    return ""


def load_class_from_file(filepath: str, class_pattern: str = 'Analyzer'):
    """从文件动态加载类"""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    # 清理模块名中的特殊字符
    module_name = module_name.replace('.', '_').replace('-', '_')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and class_pattern.lower() in attr.__name__.lower():
            return attr
    return None


def cmd_predict(args):
    """CLI: 执行预测分析"""
    print("=" * 60)
    print("  彩票号码预测 - CLI模式")
    print("=" * 60)

    analyzer_file = find_module_file('彩票号码预测系统')
    if not analyzer_file:
        print("错误: 找不到预测引擎文件 (彩票号码预测系统3.0.py)")
        return 1

    print(f"分析器: {os.path.basename(analyzer_file)}")
    print(f"数据文件: {args.data}")

    AnalyzerClass = load_class_from_file(analyzer_file, 'LotteryAnalyzerComplete')
    if AnalyzerClass is None:
        print("错误: 无法加载分析器类")
        return 1

    analyzer = AnalyzerClass()
    success, msg = analyzer.load_excel_file(args.data)
    if not success:
        print(f"错误: {msg}")
        return 1

    print(f"彩票类型: {analyzer.lottery_type}")
    print(f"数据量: {len(analyzer.data_reverse)}期")

    # 设置颗粒度
    granularity_map = {'50': 50, '100': 100, '500': 500, '1000': 1000, 'all': 0}
    gran = granularity_map.get(args.granularity, 100)
    analyzer.set_analysis_granularity(gran)

    print(f"分析颗粒度: {args.granularity}期")
    print("正在运行8种分析方法...")

    import time
    start = time.time()
    results = analyzer.analyze_all_methods()

    if 'error' in results:
        print(f"分析失败: {results['error']}")
        return 1

    elapsed = time.time() - start
    print(f"分析完成! 耗时: {elapsed:.1f}秒")
    print()

    # 输出结果
    method_names = {
        'method_1': '统计概率分析', 'method_2': '时间序列分析',
        'method_3': '模式识别分析', 'method_4': '机器学习分析',
        'method_5': '马尔可夫分析', 'method_6': '蒙特卡罗模拟',
        'method_7': '聚类分析',     'method_8': 'N-gram分析',
        'comprehensive': '★★★ 综合推荐 ★★★'
    }

    for key in ['method_1', 'method_2', 'method_3', 'method_4',
                'method_5', 'method_6', 'method_7', 'method_8', 'comprehensive']:
        if key in results:
            r = results[key]
            if 'error' in r:
                print(f"  {method_names.get(key, key)}: 错误 - {r['error']}")
                continue
            pred = r.get('predictions', {})
            if analyzer.lottery_type == 'ssq':
                reds = pred.get('red', [])[:6]
                blues = pred.get('blue', [])[:1]
                print(f"  {method_names.get(key, key)}:")
                print(f"    红球: {' '.join(f'{n:02d}' for n in reds)}")
                print(f"    蓝球: {' '.join(f'{n:02d}' for n in blues)}")
            else:
                fronts = pred.get('front', [])[:5]
                backs = pred.get('back', [])[:2]
                print(f"  {method_names.get(key, key)}:")
                print(f"    前区: {' '.join(f'{n:02d}' for n in fronts)}")
                print(f"    后区: {' '.join(f'{n:02d}' for n in backs)}")

    # 保存结果
    if args.save:
        analyzer.save_analysis_results("analysis_results")
        print(f"\n分析结果已保存到 analysis_results/ 目录")

    return 0


def cmd_merge(args):
    """CLI: 合并分析结果"""
    print("=" * 60)
    print("  分析结果合并 - CLI模式")
    print("=" * 60)

    merger_file = find_module_file('分析结果打包')
    if not merger_file:
        print("错误: 找不到合并工具文件")
        return 1

    MergerClass = load_class_from_file(merger_file, 'Merger')
    if MergerClass is None:
        print("错误: 无法加载合并器类")
        return 1

    # 查找analysis_results目录下的所有xlsx文件
    results_dir = "analysis_results"
    if not os.path.exists(results_dir):
        print(f"错误: {results_dir} 目录不存在，请先运行预测分析")
        return 1

    xlsx_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir)
                  if f.endswith('.xlsx')]
    if not xlsx_files:
        print(f"错误: {results_dir} 目录中没有Excel文件")
        return 1

    print(f"找到 {len(xlsx_files)} 个分析结果文件")
    for f in xlsx_files:
        print(f"  - {os.path.basename(f)}")

    merger = MergerClass()
    success, msg, output_path = merger.merge_excel_files(xlsx_files, "merged_results")

    if success:
        print(f"\n合并完成! 结果: {output_path}")
        return 0
    else:
        print(f"\n合并失败: {msg}")
        return 1


def cmd_backtest(args):
    """CLI: 运行回测"""
    print("=" * 60)
    print("  历史回测 - CLI模式")
    print("=" * 60)

    backtester_file = find_module_file('unified_backtester')
    if not backtester_file:
        print("错误: 找不到回测引擎文件")
        return 1

    print(f"回测引擎: {os.path.basename(backtester_file)}")
    print(f"数据文件: {args.data}")

    BacktesterClass = load_class_from_file(backtester_file, 'Backtester')
    if BacktesterClass is None:
        print("错误: 无法加载回测器类")
        return 1

    # 设置参数（不启动GUI）
    analyzer_file = find_module_file('彩票号码预测系统')
    if not analyzer_file:
        print("错误: 找不到预测引擎文件")
        return 1

    # 加载分析器
    AnalyzerClass = load_class_from_file(analyzer_file, 'LotteryAnalyzerComplete')

    # 加载数据
    analyzer = AnalyzerClass()
    success, msg = analyzer.load_excel_file(args.data)
    if not success:
        print(f"错误: {msg}")
        return 1

    data = analyzer.data_reverse
    lottery_type = analyzer.lottery_type
    print(f"彩票类型: {lottery_type}, 数据量: {len(data)}期")

    # 回测配置
    if args.fast:
        granularities = [50, 100, 0]
        methods = ['method_1', 'method_2', 'method_3', 'method_4', 'comprehensive']
        start_idx = 50
    else:
        granularities = [50, 100, 500, 1000, 0]
        methods = [f'method_{i}' for i in range(1, 9)] + ['comprehensive']
        start_idx = 100

    end_idx = max(3, len(data) // 20)
    end_test = len(data) - end_idx - 1
    num_periods = end_test - start_idx + 1

    print(f"回测范围: {start_idx}~{end_test}期 ({num_periods}个测试点)")
    print(f"颗粒度: {granularities}")
    print(f"方法数: {len(methods)}")
    print(f"总分析次数: {num_periods * len(granularities)}")
    print()

    # 运行回测
    import time
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import defaultdict

    def analyze_period_gran(period_idx, gran):
        """单个(期数, 颗粒度)的分析"""
        train = data.iloc[:period_idx].copy()
        actual_row = data.iloc[period_idx]

        if lottery_type == 'ssq':
            actual = {
                'red': [int(actual_row[f'red_{i}']) for i in range(1, 7)],
                'blue': [int(actual_row['blue'])]
            }
        else:
            actual = {
                'front': [int(actual_row[f'front_{i}']) for i in range(1, 6)],
                'back': [int(actual_row[f'back_{i}']) for i in range(1, 3)]
            }

        ta = AnalyzerClass()
        ta.data_reverse = train
        ta.lottery_type = lottery_type
        ta.set_analysis_granularity(gran)
        all_r = ta.analyze_all_methods()

        results = []
        for mk in methods:
            if mk not in all_r:
                continue
            mr = all_r[mk]
            if 'error' in mr or 'predictions' not in mr:
                continue
            pred = mr['predictions']

            if lottery_type == 'ssq':
                pred_set = set(pred.get('red', [])[:6])
                actual_set = set(actual['red'])
                blue_set = set(pred.get('blue', [])[:1])
                actual_blue_set = set(actual['blue'])
                main_hits = len(pred_set & actual_set)
                aux_hit = 1 if (blue_set & actual_blue_set) else 0
            else:
                pred_set = set(pred.get('front', [])[:5])
                actual_set = set(actual['front'])
                back_set = set(pred.get('back', [])[:2])
                actual_back_set = set(actual['back'])
                main_hits = len(pred_set & actual_set)
                aux_hit = len(back_set & actual_back_set)

            results.append({
                'period': len(data) - period_idx,
                'granularity': f'最近{gran}期' if gran > 0 else '全部期',
                'method': mk,
                'main_hits': main_hits,
                'aux_hits': aux_hit,
                'total': main_hits + aux_hit,
            })
        return results

    # 生成任务
    tasks = []
    for p in range(start_idx, end_test + 1):
        for g in granularities:
            if g > 0 and g > p:
                continue
            tasks.append((p, g))

    print(f"开始回测 ({len(tasks)}个分析任务)...")
    start_time = time.time()
    all_results = []

    workers = min(args.workers, len(tasks))
    if workers > 1:
        print(f"使用 {workers} 个线程并行处理")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(analyze_period_gran, p, g): (p, g)
                      for p, g in tasks}
            done = 0
            for future in as_completed(futures):
                try:
                    chunk = future.result()
                    all_results.extend(chunk)
                except Exception as e:
                    pass
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / done) * (len(tasks) - done)
                    print(f"  进度: {done}/{len(tasks)} ({done/len(tasks)*100:.0f}%) "
                          f"耗时{elapsed:.0f}s 预计剩余{eta:.0f}s")
    else:
        for i, (p, g) in enumerate(tasks):
            all_results.extend(analyze_period_gran(p, g))
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i+1)) * (len(tasks) - i - 1)
                print(f"  进度: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.0f}%) "
                      f"耗时{elapsed:.0f}s 预计剩余{eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n回测完成! 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"有效结果: {len(all_results)}")

    if not all_results:
        print("错误: 没有获得任何回测结果")
        return 1

    # 统计分析
    df = pd.DataFrame(all_results)
    method_names_map = {
        'method_1': '统计概率', 'method_2': '时间序列', 'method_3': '模式识别',
        'method_4': '机器学习', 'method_5': '马尔可夫', 'method_6': '蒙特卡罗',
        'method_7': '聚类分析', 'method_8': 'N-gram', 'comprehensive': '综合推荐'
    }
    df['method_name'] = df['method'].map(method_names_map)

    print("\n" + "=" * 60)
    print("  回测结果统计")
    print("=" * 60)

    print("\n按方法:")
    for mk in df['method'].unique():
        sub = df[df['method'] == mk]
        name = method_names_map.get(mk, mk)
        print(f"  {name}: 平均{sub['total'].mean():.3f}命中 "
              f"(主球{sub['main_hits'].mean():.3f}+辅助{sub['aux_hits'].mean():.3f}) "
              f"最高{int(sub['total'].max())}")

    print("\n按颗粒度:")
    for g in df['granularity'].unique():
        sub = df[df['granularity'] == g]
        print(f"  {g}: 平均{sub['total'].mean():.3f}命中 最高{int(sub['total'].max())}")

    print(f"\n整体: 平均总命中 {df['total'].mean():.3f}, "
          f"中位数 {df['total'].median():.3f}, 最高 {int(df['total'].max())}")

    # 保存报告
    os.makedirs("backtest_reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("backtest_reports", f"回测报告_{lottery_type}_{ts}.xlsx")

    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        summary = [
            ["回测报告", ""],
            ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["引擎版本", "3.0 CLI"],
            ["彩票类型", lottery_type],
            ["总耗时", f"{elapsed:.1f}秒 ({elapsed/60:.1f}分钟)"],
            ["总评估数", len(df)],
            ["平均总命中", f"{df['total'].mean():.3f}"],
            ["最高总命中", f"{int(df['total'].max())}"],
        ]
        pd.DataFrame(summary, columns=["项目", "值"]).to_excel(
            writer, sheet_name="摘要", index=False)

        # 按方法
        method_stats = df.groupby('method_name').agg(
            平均命中=('total', 'mean'),
            最高命中=('total', 'max'),
            标准差=('total', 'std'),
            评估次数=('total', 'count')
        ).round(3).sort_values('平均命中', ascending=False)
        method_stats.to_excel(writer, sheet_name="方法表现")

        # 按颗粒度
        gran_stats = df.groupby('granularity').agg(
            平均命中=('total', 'mean'),
            最高命中=('total', 'max'),
            评估次数=('total', 'count')
        ).round(3).sort_values('平均命中', ascending=False)
        gran_stats.to_excel(writer, sheet_name="颗粒度表现")

        df.to_excel(writer, sheet_name="详细结果", index=False)

    print(f"\n报告已保存: {report_path}")
    return 0


def cmd_pipeline(args):
    """CLI: 一键全流程"""
    print("=" * 60)
    print("  一键全流程 Pipeline")
    print("=" * 60)
    print(f"数据: {args.data}")
    print()

    # Step 1: 预测
    print("[1/3] 运行预测分析...")
    ret = cmd_predict(args)
    if ret != 0:
        print("预测失败，终止流程")
        return ret

    # Step 2: 合并
    print("\n[2/3] 合并分析结果...")
    ret = cmd_merge(args)
    if ret != 0:
        print("合并失败，但仍继续...")

    # Step 3: 回测
    print("\n[3/3] 运行回测评估...")
    ret = cmd_backtest(args)

    print("\n" + "=" * 60)
    print("  全流程完成!")
    print("=" * 60)
    print("输出目录:")
    print("  analysis_results/   - 分析结果")
    print("  merged_results/     - 合并结果")
    print("  backtest_reports/   - 回测报告")
    return 0


def cmd_gui(args):
    """启动GUI模式"""
    print("启动GUI模式...")

    # 尝试启动3.0版预测引擎的GUI
    analyzer_file = find_module_file('彩票号码预测系统')
    if analyzer_file:
        AnalyzerClass = load_class_from_file(analyzer_file, 'LotteryAnalyzerComplete')
        if AnalyzerClass:
            # 启动分析系统GUI
            from 彩票号码预测系统3_0 import LotteryAnalysisGUI
            # Actually we can't do this because the module name has dots. Let's use the GUI class from the loaded module.
            pass

    # 简化方案：启动 tkinter 选择界面
    _launch_selector_gui(args)


def _launch_selector_gui(args):
    """启动简洁的系统选择GUI"""
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("彩票号码预测系统 3.0 - 主控制台")
    root.geometry("700x550")
    root.configure(bg="#f5f5f5")

    # 标题
    title = tk.Label(root, text="彩票号码预测系统 3.0",
                   font=("Arial", 22, "bold"), fg="#1565C0", bg="#f5f5f5")
    title.pack(pady=25)

    subtitle = tk.Label(root, text="集成预测 · 合并 · 回测 · 优化",
                      font=("Arial", 12), fg="#757575", bg="#f5f5f5")
    subtitle.pack(pady=5)

    # 按钮区域
    btn_frame = tk.Frame(root, bg="#f5f5f5")
    btn_frame.pack(pady=30)

    def launch_module(module_name, file_pattern):
        """启动子模块"""
        filepath = find_module_file(file_pattern)
        if not filepath:
            messagebox.showerror("错误", f"找不到 {module_name} 模块文件")
            return
        # 在新进程中启动（保持主控制台运行）
        python_exe = sys.executable
        os.system(f'start "{module_name}" "{python_exe}" "{filepath}"')

    buttons = [
        ("① 预测分析\n8种方法·5种颗粒度\n生成预测号码", "#2196F3", "彩票号码预测系统"),
        ("② 结果打包\n多颗粒度合并\n最终推荐号码", "#4CAF50", "分析结果打包"),
        ("③ 历史回测\n性能评估·权重优化\n闭环迭代", "#FF9800", "unified_backtester"),
        ("④ 一键全流程\n预测→合并→回测\n自动化流水线", "#9C27B0", "pipeline"),
    ]

    for i, (text, color, pattern) in enumerate(buttons):
        row, col = i // 2, i % 2
        btn = tk.Button(btn_frame, text=text, font=("Arial", 11),
                       bg=color, fg="white", width=28, height=4,
                       relief=tk.RAISED, bd=2,
                       command=lambda p=pattern: launch_module(
                           text.split('\n')[0], p))
        btn.grid(row=row, column=col, padx=10, pady=10)

    # 信息栏
    info_frame = tk.Frame(root, bg="#f5f5f5")
    info_frame.pack(pady=20)

    tk.Label(info_frame, text=f"项目路径: {os.path.dirname(os.path.abspath(__file__))}",
            font=("Arial", 8), fg="#9E9E9E", bg="#f5f5f5").pack()

    tk.Label(info_frame, text="版本 3.0 | 2026年6月",
            font=("Arial", 9), fg="#757575", bg="#f5f5f5").pack(pady=5)

    root.mainloop()


# ================================================================================
#  主入口
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="彩票号码预测系统 3.0 - 统一主系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python MainSystem.py                                    # 启动GUI主控制台
  python MainSystem.py --predict 双色球.xlsx              # CLI预测分析
  python MainSystem.py --predict 双色球.xlsx -g 500 -s   # 预测并保存
  python MainSystem.py --backtest 双色球.xlsx --fast      # 快速回测
  python MainSystem.py --backtest 双色球.xlsx -w 8        # 完整回测8线程
  python MainSystem.py --merge                            # 合并分析结果
  python MainSystem.py --pipeline 双色球.xlsx             # 一键全流程
        """
    )

    parser.add_argument('--predict', '-p', nargs='?', const='__gui__', metavar='DATA',
                       help='运行预测分析 (可选: Excel文件路径)')
    parser.add_argument('--backtest', '-b', nargs='?', const='__gui__', metavar='DATA',
                       help='运行历史回测 (可选: Excel文件路径)')
    parser.add_argument('--merge', '-m', action='store_true',
                       help='合并分析结果')
    parser.add_argument('--pipeline', nargs='?', const='__gui__', metavar='DATA',
                       help='一键全流程 (可选: Excel文件路径)')
    parser.add_argument('--gui', '-g', action='store_true',
                       help='强制启动GUI模式')
    parser.add_argument('--granularity', choices=['50', '100', '500', '1000', 'all'],
                       default='100', help='分析颗粒度 (默认: 100)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='保存分析结果到Excel')
    parser.add_argument('--fast', '-f', action='store_true',
                       help='快速回测模式')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='并行线程数 (默认: 4)')

    args = parser.parse_args()

    # 确定 mode 并设置 data 属性（兼容 cmd_predict 等函数）
    if args.predict is not None:
        args.mode = 'predict'
        args.data = None if args.predict == '__gui__' else args.predict
    elif args.backtest is not None:
        args.mode = 'backtest'
        args.data = None if args.backtest == '__gui__' else args.backtest
    elif args.pipeline is not None:
        args.mode = 'pipeline'
        args.data = None if args.pipeline == '__gui__' else args.pipeline
    elif args.merge:
        args.mode = 'merge'
    elif args.gui:
        args.mode = 'gui'
    else:
        args.mode = 'gui'

    if args.mode == 'predict':
        return cmd_predict(args)
    elif args.mode == 'backtest':
        return cmd_backtest(args)
    elif args.mode == 'merge':
        return cmd_merge(args)
    elif args.mode == 'pipeline':
        return cmd_pipeline(args)
    elif args.mode == 'gui':
        return cmd_gui(args)
    else:
        return cmd_gui(args)


if __name__ == "__main__":
    sys.exit(main())
