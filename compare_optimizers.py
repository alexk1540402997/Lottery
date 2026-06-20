"""
优化器三方对比测试脚本 4.3
===========================
依次用 BO / CMA-ES / SA 各跑一轮回测（相同条件），对比关键指标。

用法:
  python compare_optimizers.py                           # 默认: 双色球, 10期, 120秒/优化器
  python compare_optimizers.py --lottery dlt              # 大乐透
  python compare_optimizers.py --periods 5 --time 60      # 5期, 60秒/优化器
  python compare_optimizers.py --skip-solve               # 只测回测，不测求解模式
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from backtester import BacktestEngine, SolveEngine, OPTIMIZERS_AVAILABLE


# ============================================================================
#  辅助函数
# ============================================================================

def find_data_file() -> str:
    """自动查找数据文件"""
    base = os.path.dirname(os.path.abspath(__file__))
    for name in ['双色球.xlsx', '大乐透.xlsx']:
        for subdir in ['', 'pythonProject']:
            path = os.path.join(base, subdir, name) if subdir else os.path.join(base, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError("未找到数据文件（双色球.xlsx / 大乐透.xlsx）")


def resolve_lottery_type(filepath: str) -> str:
    """从文件名推断彩票类型"""
    basename = os.path.basename(filepath)
    if '大乐透' in basename or 'dlt' in basename.lower():
        return 'dlt'
    return 'ssq'


# ============================================================================
#  单个优化器测试
# ============================================================================

def run_single_optimizer(
    data_file: str,
    optimizer: str,        # 'bo' | 'cmaes' | 'sa'
    test_periods: int = 10,
    max_time: int = 120,
    num_workers: int = 4,
    verbose: bool = True,
) -> Dict:
    """
    用指定优化器运行回测，返回结果摘要。
    """
    label_map = {'bo': 'BO 贝叶斯优化', 'cmaes': 'CMA-ES', 'sa': 'SA 模拟退火'}
    label = label_map.get(optimizer, optimizer)

    engine = BacktestEngine()
    engine.load_data(data_file)
    engine.set_config(
        test_periods=test_periods,
        max_search_time=max_time,
        num_workers=num_workers,
    )

    # 初始化优化器
    if optimizer == 'bo':
        engine.init_bo('backtest')
    elif optimizer == 'cmaes':
        engine.init_cmaes()
    elif optimizer == 'sa':
        engine.init_sa(n_chains=4)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  测试: {label}")
        print(f"{'='*60}")

    t0 = time.time()
    result = engine.run()
    elapsed = time.time() - t0

    # 提取摘要
    scores_over_time = []
    if result.get('all_results'):
        for r in result['all_results']:
            if 'avg_total_hits' in r:
                scores_over_time.append(r['avg_total_hits'])

    # 收敛速度：达到最佳得分的 90% 需要多少组合
    best = result.get('best_avg_hits', 0)
    convergence_90 = 0
    if scores_over_time and best > 0:
        target = best * 0.90
        for i, s in enumerate(scores_over_time):
            if s >= target:
                convergence_90 = i + 1
                break

    # 前 N 个组合的平均得分（衡量初始质量）
    early_avg = np.mean(scores_over_time[:5]) if len(scores_over_time) >= 5 else (
        np.mean(scores_over_time) if scores_over_time else 0)

    phase_map = {'bo_warmup': 'exploration', 'bo_active': 'exploration', 'bo_fallback': 'exploration'}
    # 统计各优化器的阶段分布
    phase_stats = result.get('phase_stats', {})

    summary = {
        'optimizer': optimizer,
        'label': label,
        'success': result.get('success', False),
        'best_avg_hits': round(float(best), 4),
        'best_max_hits': result.get('best_max_hits', 0),
        'best_hit_rate_5plus': result.get('best_hit_rate_5plus', 0),
        'total_combos': result.get('total_combos_tried', 0),
        'total_skipped': result.get('total_combos_skipped', 0),
        'total_time': round(elapsed, 1),
        'combos_per_min': round(result.get('total_combos_tried', 0) / max(1, elapsed / 60), 1),
        'convergence_90_at': convergence_90,
        'early_avg_5': round(float(early_avg), 4),
        'phase_distribution': phase_stats,
    }

    # 输出单结果摘要
    if verbose:
        print(f"  成功: {summary['success']}")
        print(f"  最佳平均命中: {summary['best_avg_hits']:.4f}")
        print(f"  最佳最高命中: {summary['best_max_hits']}")
        print(f"  5+命中率: {summary['best_hit_rate_5plus']:.1%}")
        print(f"  评估组合数: {summary['total_combos']}")
        print(f"  总耗时: {summary['total_time']:.0f}s")
        print(f"  收敛到90%: 第{summary['convergence_90_at']}个组合")

    return summary


# ============================================================================
#  求解模式测试 (BO+线性)
# ============================================================================

def run_solve_mode(
    data_file: str,
    solve_periods: int = 1,
    tolerance_main: int = 5,
    tolerance_aux: int = 1,
    max_time: int = 120,
    num_workers: int = 4,
    verbose: bool = True,
) -> Dict:
    """BO+线性求解模式测试"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  测试: 求解模式 (BO+线性权重)")
        print(f"{'='*60}")

    engine = SolveEngine()
    engine.load_data(data_file)
    engine.set_solve_config(
        solve_periods=solve_periods,
        tolerance_main=tolerance_main,
        tolerance_aux=tolerance_aux,
        max_search_time=max_time,
        num_workers=num_workers,
    )

    if OPTIMIZERS_AVAILABLE:
        engine.init_bo('solve')
        engine.use_linear_weights = True

    t0 = time.time()
    result = engine.run()
    elapsed = time.time() - t0

    summary = {
        'optimizer': 'bo_linear_solve',
        'label': '求解(BO+线性)',
        'success': result.get('success', False),
        'solutions_found': len(result.get('solutions', [])),
        'total_evaluated': result.get('total_evaluated', 0),
        'total_time': round(elapsed, 1),
        'solve_method': result.get('solve_method', 'random'),
        'solve_config': result.get('solve_config', {}),
        'solutions': result.get('solutions', []),
    }

    if verbose:
        print(f"  求解方法: {summary['solve_method']}")
        print(f"  找到解: {summary['solutions_found']}个")
        print(f"  评估组合数: {summary['total_evaluated']}")
        print(f"  总耗时: {summary['total_time']:.0f}s")

    return summary


# ============================================================================
#  对比报告
# ============================================================================

def print_comparison(results: List[Dict], solve_result: Dict = None):
    """打印对比报告"""
    print(f"\n{'='*70}")
    print(f"  优 化 器 三 方 对 比 报 告")
    print(f"{'='*70}")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 表头
    print(f"\n{'─'*70}")
    print(f"{'指标':<24} {'BO':>10} {'CMA-ES':>12} {'SA':>12}")
    print(f"{'─'*70}")

    bo = next((r for r in results if r['optimizer'] == 'bo'), None)
    cma = next((r for r in results if r['optimizer'] == 'cmaes'), None)
    sa = next((r for r in results if r['optimizer'] == 'sa'), None)

    def row(label: str, key: str, fmt: str = '.4f', bo_val=None, cma_val=None, sa_val=None):
        bv = bo_val if bo_val is not None else (bo.get(key, 0) if bo else '—')
        cv = cma_val if cma_val is not None else (cma.get(key, 0) if cma else '—')
        sv = sa_val if sa_val is not None else (sa.get(key, 0) if sa else '—')
        b_str = f"{bv:{fmt}}" if isinstance(bv, (int, float)) else str(bv)
        c_str = f"{cv:{fmt}}" if isinstance(cv, (int, float)) else str(cv)
        s_str = f"{sv:{fmt}}" if isinstance(sv, (int, float)) else str(sv)
        print(f"  {label:<22} {b_str:>10} {c_str:>12} {s_str:>12}")

    row('最佳平均命中', 'best_avg_hits')
    row('最佳最高命中', 'best_max_hits', 'd')
    row('5+命中率', 'best_hit_rate_5plus', '.1%')
    row('评估组合数', 'total_combos', 'd')
    row('总耗时(s)', 'total_time', '.0f')
    row('速度(组合/分钟)', 'combos_per_min', '.1f')
    row('90%收敛位置', 'convergence_90_at', 'd')
    row('前5组合平均得分', 'early_avg_5')
    row('跳过重复', 'total_skipped', 'd')

    print(f"{'─'*70}")

    # 排名
    print(f"\n  ── 排名 ──")
    metrics = ['best_avg_hits', 'best_max_hits', 'combos_per_min', 'early_avg_5']
    for metric in metrics:
        vals = {}
        if bo and bo.get(metric) is not None:
            vals['BO'] = bo[metric]
        if cma and cma.get(metric) is not None:
            vals['CMA-ES'] = cma[metric]
        if sa and sa.get(metric) is not None:
            vals['SA'] = sa[metric]
        if not vals:
            continue
        # 高分排前
        ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        names = ' > '.join(f"{n}({v})" for n, v in ranked)
        label_map = {
            'best_avg_hits': '平均命中',
            'best_max_hits': '最高命中',
            'combos_per_min': '评估速度',
            'early_avg_5': '初始质量',
        }
        print(f"  {label_map.get(metric, metric)}: {names}")

    # 求解模式结果
    if solve_result:
        print(f"\n{'─'*70}")
        print(f"  求解模式结果")
        print(f"{'─'*70}")
        print(f"  求解方法: {solve_result.get('solve_method', '?')}")
        print(f"  找到解: {solve_result.get('solutions_found', 0)}个")
        print(f"  评估组合数: {solve_result.get('total_evaluated', 0)}")
        print(f"  总耗时: {solve_result.get('total_time', 0):.0f}s")
        cfg = solve_result.get('solve_config', {})
        print(f"  配置: {cfg.get('periods', '?')}期, "
              f"主球≥{cfg.get('tolerance_main', '?')}, "
              f"辅助球≥{cfg.get('tolerance_aux', '?')}")

        solutions = solve_result.get('solutions', [])
        if solutions:
            print(f"\n  解的详情:")
            for i, sol in enumerate(solutions[:5]):
                print(f"    解#{i+1}: 平均命中={sol.get('avg_total_hits', '?'):.3f}, "
                      f"最高={sol.get('max_total_hits', '?')}")

    print(f"\n{'='*70}")


def save_report(results: List[Dict], solve_result: Dict = None,
                output_dir: str = "backtest_reports"):
    """保存对比报告"""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(output_dir, f"optimizer_comparison_{ts}.json")

    report = {
        'timestamp': ts,
        'backtest_results': results,
        'solve_result': solve_result,
    }
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n对比报告已保存: {fpath}")
    return fpath


# ============================================================================
#  主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='优化器三方对比测试')
    parser.add_argument('--lottery', type=str, default='ssq',
                        choices=['ssq', 'dlt'], help='彩票类型')
    parser.add_argument('--periods', type=int, default=10,
                        help='回测测试期数 (默认10)')
    parser.add_argument('--time', type=int, default=120,
                        help='每个优化器的搜索时间上限/秒 (默认120)')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行线程数 (默认4)')
    parser.add_argument('--optimizers', type=str, default='bo,cmaes,sa',
                        help='要测试的优化器，逗号分隔 (默认: bo,cmaes,sa)')
    parser.add_argument('--skip-solve', action='store_true',
                        help='跳过求解模式对比')
    parser.add_argument('--solve-periods', type=int, default=1,
                        help='求解模式期数 (默认1)')
    parser.add_argument('--solve-tol-main', type=int, default=5,
                        help='求解模式主球容差 (默认5)')
    parser.add_argument('--solve-tol-aux', type=int, default=1,
                        help='求解模式辅助球容差 (默认1)')
    args = parser.parse_args()

    # 查找数据文件
    data_file = find_data_file()
    lt = resolve_lottery_type(data_file)
    if args.lottery != lt:
        # 用户指定了不同彩票类型，找对应文件
        base = os.path.dirname(os.path.abspath(__file__))
        fname = '双色球.xlsx' if args.lottery == 'ssq' else '大乐透.xlsx'
        for subdir in ['', 'pythonProject']:
            path = os.path.join(base, subdir, fname) if subdir else os.path.join(base, fname)
            if os.path.exists(path):
                data_file = path
                break

    print(f"{'='*70}")
    print(f"  优化器三方对比测试")
    print(f"  数据: {os.path.basename(data_file)}")
    print(f"  测试期数: {args.periods}期")
    print(f"  每优化器时间上限: {args.time}秒")
    print(f"  并行线程: {args.workers}")
    print(f"{'='*70}")

    if not OPTIMIZERS_AVAILABLE:
        print("[错误] 优化器模块不可用，请检查 optimizers/ 目录")
        return

    # 逐项测试
    backtest_results = []
    optimizers_to_test = [o.strip() for o in args.optimizers.split(',')]

    for opt in optimizers_to_test:
        summary = run_single_optimizer(
            data_file=data_file,
            optimizer=opt,
            test_periods=args.periods,
            max_time=args.time,
            num_workers=args.workers,
        )
        backtest_results.append(summary)

    # 求解模式对比
    solve_result = None
    if not args.skip_solve:
        solve_result = run_solve_mode(
            data_file=data_file,
            solve_periods=args.solve_periods,
            tolerance_main=args.solve_tol_main,
            tolerance_aux=args.solve_tol_aux,
            max_time=args.time,
            num_workers=args.workers,
        )

    # 打印报告
    print_comparison(backtest_results, solve_result)

    # 保存
    save_report(backtest_results, solve_result)


if __name__ == "__main__":
    main()
