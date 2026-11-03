#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
500 期截断对照实验（Phase 0 扩展）

同一配置、同一批测试期、同一 5 颗粒度，对比：
  A. 真实窗口：1000 期=最近 1000 期，全部期=该期之前全部历史
  B. 500 上限：训练集先截到 500 期，1000/全部期实际都变成 500 期

为节省时间，50/100/500 三种颗粒度的预测只算一次；
500 上限模式下的 1000/全部期直接复用 500 期的预测结果。

用法：
  python tools/truncation_experiment.py --lottery ssq --periods 100 --workers 12
  python tools/truncation_experiment.py --lottery dlt --periods 100 --workers 12
"""
import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime

import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRANS = [50, 100, 500, 1000, 0]
GRAN_NAME = {50: '50期', 100: '100期', 500: '500期', 1000: '1000期', 0: '全部期'}

_DATA = None
_LOTTERY = None
_PARAMS = None
_PREDICTOR = None
_MERGER = None


def _init_worker(data_path, lottery, params, weights):
    global _DATA, _LOTTERY, _PARAMS, _PREDICTOR, _MERGER
    from predictor import LotteryPredictor
    from merger import ResultMerger

    with redirect_stdout(io.StringIO()):
        _DATA, lt = LotteryPredictor.load_data(data_path)
    _LOTTERY = lt
    _PARAMS = params
    _PREDICTOR = LotteryPredictor(lt)
    _MERGER = ResultMerger(lt)
    _MERGER.import_weights({'composite_weights': weights})


def _actual_numbers(row):
    if _LOTTERY == 'ssq':
        main = sorted(int(row[f'red_{i}']) for i in range(1, 7))
        aux = sorted([int(row['blue'])])
    else:
        main = sorted(int(row[f'front_{i}']) for i in range(1, 6))
        aux = sorted(int(row[f'back_{i}']) for i in range(1, 3))
    return main, aux


def _compute_votes(gran_predictions):
    main_votes = defaultdict(float)
    aux_votes = defaultdict(float)
    for gran_name, methods in gran_predictions.items():
        for method_key, result in methods.items():
            if method_key == 'comprehensive':
                continue
            if not isinstance(result, dict) or 'error' in result:
                continue
            weight = _MERGER.compute_weight(gran_name, method_key)
            pred = result.get('predictions', {})
            for num in pred.get(_MERGER.main_name, []) or []:
                if _MERGER.main_range[0] <= num <= _MERGER.main_range[1]:
                    main_votes[num] += weight
            for num in pred.get(_MERGER.aux_name, []) or []:
                if _MERGER.aux_range[0] <= num <= _MERGER.aux_range[1]:
                    aux_votes[num] += weight
    return main_votes, aux_votes


def _ranks(actual, votes, number_range):
    numbers = list(range(number_range[0], number_range[1] + 1))
    ordered = sorted(numbers, key=lambda n: (-votes.get(n, 0.0), n))
    rank = {n: i + 1 for i, n in enumerate(ordered)}
    return [rank[n] for n in actual]


def _near(actual, predicted, threshold):
    return sum(1 for a in actual if any(abs(a - p) <= threshold for p in predicted))


def _metrics(gran_predictions, actual_main, actual_aux):
    merged = _MERGER.merge_results(gran_predictions)
    pred_main = sorted(merged['predictions'][_MERGER.main_name])
    pred_aux = sorted(merged['predictions'][_MERGER.aux_name])

    main_hits = len(set(pred_main) & set(actual_main))
    aux_hits = len(set(pred_aux) & set(actual_aux))
    main_votes, aux_votes = _compute_votes(gran_predictions)
    main_ranks = _ranks(actual_main, main_votes, _MERGER.main_range)
    aux_ranks = _ranks(actual_aux, aux_votes, _MERGER.aux_range)

    return {
        'predicted_main': pred_main,
        'predicted_aux': pred_aux,
        'main_hits': main_hits,
        'aux_hits': aux_hits,
        'total_hits': main_hits + aux_hits,
        'near_main_1': _near(actual_main, pred_main, 1),
        'near_main_2': _near(actual_main, pred_main, 2),
        'near_aux_1': _near(actual_aux, pred_aux, 1),
        'near_aux_2': _near(actual_aux, pred_aux, 2),
        'actual_main_avg_rank': round(sum(main_ranks) / len(main_ranks), 2),
        'actual_aux_avg_rank': round(sum(aux_ranks) / len(aux_ranks), 2),
    }


def _eval_period(period_idx):
    train_full = _DATA.iloc[period_idx + 1:]
    actual_main, actual_aux = _actual_numbers(_DATA.iloc[period_idx])

    # 真实窗口下的 5 种颗粒度预测
    true_preds = {}
    for g_idx, gran in enumerate(GRANS):
        if gran == 0:
            gran_data = train_full
        else:
            gran_data = train_full.head(min(len(train_full), gran))
        if len(gran_data) < 10:
            continue
        true_preds[GRAN_NAME[gran]] = _PREDICTOR.predict_all(
            gran_data, params=_PARAMS, seed=42 + period_idx * 100 + g_idx)

    # 500 上限：1000/全部期复用 500 期的预测
    cap_preds = {}
    for name in ('50期', '100期', '500期'):
        if name in true_preds:
            cap_preds[name] = true_preds[name]
    if '500期' in true_preds:
        cap_preds['1000期'] = true_preds['500期']
        cap_preds['全部期'] = true_preds['500期']

    return {
        'period_idx': period_idx,
        'period': int(_DATA.iloc[period_idx]['period']),
        'actual_main': actual_main,
        'actual_aux': actual_aux,
        'true': _metrics(true_preds, actual_main, actual_aux),
        'cap': _metrics(cap_preds, actual_main, actual_aux),
    }


def _summarize(results, key):
    rows = [r[key] for r in results]
    n = len(rows)
    totals = [r['total_hits'] for r in rows]
    return {
        'periods': n,
        'avg_total_hits': round(sum(totals) / n, 4),
        'avg_main_hits': round(sum(r['main_hits'] for r in rows) / n, 4),
        'avg_aux_hits': round(sum(r['aux_hits'] for r in rows) / n, 4),
        'max_total_hits': max(totals),
        'hit_distribution': {str(h): totals.count(h) for h in range(0, 8)},
        'rate_ge_3': round(sum(1 for h in totals if h >= 3) / n, 4),
        'rate_ge_4': round(sum(1 for h in totals if h >= 4) / n, 4),
        'rate_ge_5': round(sum(1 for h in totals if h >= 5) / n, 4),
        'avg_near_main_1': round(sum(r['near_main_1'] for r in rows) / n, 2),
        'avg_near_main_2': round(sum(r['near_main_2'] for r in rows) / n, 2),
        'avg_actual_main_rank': round(
            sum(r['actual_main_avg_rank'] for r in rows) / n, 2),
        'avg_actual_aux_rank': round(
            sum(r['actual_aux_avg_rank'] for r in rows) / n, 2),
    }


def run(args):
    from config_manager import ConfigManager

    data_file = args.data or ('双色球.xlsx' if args.lottery == 'ssq' else '大乐透.xlsx')
    cm = ConfigManager('logs', args.lottery)
    params = {k: v for k, v in cm.current_params.items() if k != '_meta'}
    weights = cm.current_weights.get('composite_weights', {})
    param_meta = cm.current_params.get('_meta', {})
    weight_meta = cm.current_weights.get('_meta', {})

    with redirect_stdout(io.StringIO()):
        data, _ = __import__('predictor').LotteryPredictor.load_data(data_file)
    total = len(data)
    n_periods = min(args.periods, total - 10)

    print('=' * 68)
    print(f'500 期截断对照 | 彩种={args.lottery} | 数据={data_file}')
    print(f'参数 v{param_meta.get("version")} | 权重 v{weight_meta.get("version")} | '
          f'测试 {n_periods} 期 | 进程 {args.workers}')
    print('=' * 68)

    t0 = time.perf_counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(data_file, args.lottery, params, weights),
    ) as pool:
        results = pool.map(_eval_period, range(n_periods), chunksize=1)

    results.sort(key=lambda r: r['period_idx'])
    elapsed = time.perf_counter() - t0

    true_summary = _summarize(results, 'true')
    cap_summary = _summarize(results, 'cap')
    deltas = [r['true']['total_hits'] - r['cap']['total_hits'] for r in results]
    improved = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    worse = sum(1 for d in deltas if d < 0)

    comparison = {
        'avg_total_delta_true_minus_cap': round(sum(deltas) / len(deltas), 4),
        'improved': improved,
        'same': same,
        'worse': worse,
        'near_main_1_delta': round(
            true_summary['avg_near_main_1'] - cap_summary['avg_near_main_1'], 4),
        'actual_main_rank_delta': round(
            true_summary['avg_actual_main_rank'] - cap_summary['avg_actual_main_rank'], 4),
    }

    report = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lottery_type': args.lottery,
        'data_file': data_file,
        'periods': n_periods,
        'workers': args.workers,
        'param_version': param_meta.get('version'),
        'weight_version': weight_meta.get('version'),
        'true_windows': true_summary,
        'cap_500': cap_summary,
        'comparison': comparison,
        'periods_detail': results,
        'elapsed_seconds': round(elapsed, 1),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f'truncation_{args.lottery}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('-' * 68)
    print('真实窗口 vs 500 上限')
    print(f'  平均总命中: {true_summary["avg_total_hits"]} vs {cap_summary["avg_total_hits"]} '
          f'| 差值 {comparison["avg_total_delta_true_minus_cap"]:+}')
    print(f'  ≥3 比例: {true_summary["rate_ge_3"]:.1%} vs {cap_summary["rate_ge_3"]:.1%}')
    print(f'  邻近主1: {true_summary["avg_near_main_1"]} vs {cap_summary["avg_near_main_1"]} '
          f'| 差值 {comparison["near_main_1_delta"]:+}')
    print(f'  实际主球平均排名: {true_summary["avg_actual_main_rank"]} vs '
          f'{cap_summary["avg_actual_main_rank"]} '
          f'| 差值 {comparison["actual_main_rank_delta"]:+}')
    print(f'  逐期改善/持平/变差: {improved} / {same} / {worse}')
    print(f'  耗时: {elapsed:.1f} 秒')
    print(f'报告: {out_path}')
    return out_path


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description='500 期截断对照实验')
    parser.add_argument('--lottery', choices=['ssq', 'dlt'], default='ssq')
    parser.add_argument('--data', default=None)
    parser.add_argument('--periods', type=int, default=100)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output-dir', default='baseline_reports')
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
