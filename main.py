"""
彩票号码预测系统 4.0 - 主入口
=============================
集成预测、回测、优化、日志四大功能。
支持GUI模式和CLI命令行模式。

用法:
  python main.py                          # 启动GUI
  python main.py --predict 双色球.xlsx    # CLI预测
  python main.py --backtest 双色球.xlsx   # CLI回测
  python main.py --cli                    # CLI交互模式
"""

import os
import sys
import argparse
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import LotteryPredictor, DEFAULT_PARAMS, METHOD_NAMES_NEW
from merger import ResultMerger, METHOD_NAMES, GRANULARITY_NAMES, GRANULARITY_VALUES
from backtester import BacktestEngine
from config_manager import ConfigManager


# ============================================================================
#  CLI预测
# ============================================================================

def cmd_predict(args):
    """CLI模式：运行预测"""
    print("=" * 60)
    print("  彩票号码预测系统 4.0 - 预测模式")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"错误: 文件不存在 - {args.data}")
        return 1

    # 加载数据
    print(f"数据文件: {args.data}")
    data_rev, lt = LotteryPredictor.load_data(args.data)
    print(f"彩票类型: {lt}, 总期数: {len(data_rev)}")

    # 创建预测器
    predictor = LotteryPredictor(lt)

    # 解析颗粒度
    gran_map = {'50': 50, '100': 100, '500': 500, '1000': 1000, 'all': 0}
    gran = gran_map.get(args.granularity, 100)

    gran_text = f"最近{gran}期" if gran > 0 else "全部期"
    train_data = data_rev.head(gran) if gran > 0 else data_rev

    print(f"分析颗粒度: {gran_text} ({len(train_data)}期)")
    print()

    # 运行预测
    print("正在运行8种分析方法...")
    t0 = time.time()
    results = predictor.predict_all(train_data, seed=42)
    elapsed = time.time() - t0
    print(f"分析完成! 耗时: {elapsed:.1f}秒\n")

    # 输出结果
    for key in ['method_1', 'method_2', 'method_3', 'method_4',
                'method_5', 'method_6', 'method_7', 'method_8', 'comprehensive']:
        if key in results:
            r = results[key]
            if 'error' in r:
                print(f"  {METHOD_NAMES_NEW.get(key, key)}: 错误 - {r['error']}")
                continue
            pred = r.get('predictions', {})
            if lt == 'ssq':
                reds = pred.get('red', [])[:6]
                blues = pred.get('blue', [])[:1]
                print(f"  {r['method']}:")
                print(f"    红球: {'  '.join(f'{n:02d}' for n in reds)}")
                print(f"    蓝球: {'  '.join(f'{n:02d}' for n in blues)}")
            else:
                fronts = pred.get('front', [])[:5]
                backs = pred.get('back', [])[:2]
                print(f"  {r['method']}:")
                print(f"    前区: {'  '.join(f'{n:02d}' for n in fronts)}")
                print(f"    后区: {'  '.join(f'{n:02d}' for n in backs)}")

    # 保存
    if args.save:
        from merger import batch_merge_to_excel
        all_preds = {gran_text: results}
        merger = ResultMerger(lt)
        path = batch_merge_to_excel(all_preds, merger)
        print(f"\n结果已保存: {path}")

    return 0


# ============================================================================
#  CLI回测
# ============================================================================

def cmd_backtest(args):
    """CLI模式：运行回测"""
    print("=" * 60)
    print("  彩票号码预测系统 4.0 - 回测模式")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"错误: 文件不存在 - {args.data}")
        return 1

    # 创建引擎
    engine = BacktestEngine()
    ok, msg = engine.load_data(args.data)
    if not ok:
        print(f"错误: {msg}")
        return 1

    print(f"数据: {msg}")

    # 配置
    test_periods = args.periods if hasattr(args, 'periods') and args.periods else 50
    search_mode = 'baseline' if args.fast else 'grid'
    max_combos = 50 if args.fast else 100
    use_fast = args.fast if hasattr(args, 'fast') else False
    max_time = args.time_limit * 60 if hasattr(args, 'time_limit') and args.time_limit else 0

    print(f"测试期数: {test_periods}")
    print(f"搜索模式: {search_mode}")
    print(f"最大参数组: {max_combos}")
    print()

    # 运行
    result = engine.run_backtest(
        search_mode=search_mode,
        max_combinations=max_combos,
        test_periods=test_periods,
        use_fast_space=use_fast,
    )

    if not result['success']:
        print(f"回测失败: {result.get('error', '未知错误')}")
        return 1

    print(f"\n{'='*60}")
    print(f"  回测完成! 耗时: {result['total_time']:.1f}秒")
    print(f"{'='*60}")
    print(f"  最佳合并平均命中: {result['best_merged_avg_hits']:.3f}")
    print(f"  最佳合并最高命中: {result['best_merged_max_hits']}")
    print(f"  尝试参数组合: {result['total_combinations_tried']}")

    # 保存报告
    report_path = engine.generate_report()
    if report_path:
        print(f"\n报告已保存: {report_path}")

    # 保存版本
    config_mgr = ConfigManager()
    config_mgr.save_params_version(
        result.get('best_params', {}),
        description=f"CLI回测 (合并平均命中{result['best_merged_avg_hits']:.3f})",
        lottery_type=engine.lottery_type,
        backtest_score=result['best_merged_avg_hits'],
    )
    print("参数版本已保存")

    return 0


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="彩票号码预测系统 4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                   # 启动GUI
  python main.py --predict 双色球.xlsx -g 100     # CLI预测
  python main.py --predict 双色球.xlsx --save      # 预测并保存
  python main.py --backtest 双色球.xlsx            # CLI回测
  python main.py --backtest 双色球.xlsx --fast     # CLI快速回测
        """
    )

    parser.add_argument('--predict', '-p', nargs='?', const='__gui__',
                       metavar='FILE', help='运行预测分析')
    parser.add_argument('--backtest', '-b', nargs='?', const='__gui__',
                       metavar='FILE', help='运行历史回测')
    parser.add_argument('--granularity', '-g', choices=['50', '100', '500', '1000', 'all'],
                       default='100', help='分析颗粒度 (默认100)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='保存分析结果')
    parser.add_argument('--fast', '-f', action='store_true',
                       help='快速模式（回测用基准参数）')
    parser.add_argument('--periods', type=int, default=50,
                       help='回测测试期数 (默认50)')
    parser.add_argument('--time-limit', type=int, default=0,
                       help='最大搜索时间(分钟, 0=不限制)')

    args = parser.parse_args()

    # 判断模式
    if args.predict is not None:
        if args.predict == '__gui__':
            args.data = None
            print("请使用 --predict FILE 指定数据文件")
            return 1
        args.data = args.predict
        return cmd_predict(args)
    elif args.backtest is not None:
        if args.backtest == '__gui__':
            args.data = None
            print("请使用 --backtest FILE 指定数据文件")
            return 1
        args.data = args.backtest
        return cmd_backtest(args)
    else:
        # 默认启动GUI
        from gui import main as gui_main
        gui_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
