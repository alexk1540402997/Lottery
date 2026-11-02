#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线评估脚本（Phase 0）

用途：
  用当前配置（或指定的 champion 版本）在最近 N 期上做滚动评估，
  输出平均命中、命中分布、邻近命中、实际号码平均排名等指标。
  后续所有优化都用同一套指标做 A/B 对比。

用法示例：
  python tools/baseline_eval.py --lottery ssq --periods 5 --granularities 100,500
  python tools/baseline_eval.py --lottery ssq --periods 10 --granularities 50,100,500
  python tools/baseline_eval.py --lottery ssq --true-windows --periods 3 --granularities 1000,all

说明：
  --max-train 500 表示每个测试期的训练集最多 500 期（当前回测的默认行为）。
  --true-windows 等价于 --max-train 0，即 1000 期/全部期按真实窗口计算。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# 允许从项目根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import LotteryPredictor
from merger import ResultMerger
from config_manager import ConfigManager


GRAN_MAP = {'50': 50, '100': 100, '500': 500, '1000': 1000, 'all': 0}
GRAN_NAME = {50: '50期', 100: '100期', 500: '500期', 1000: '1000期', 0: '全部期'}


def parse_granularities(text):
    result = []
    for part in text.split(','):
        part = part.strip().lower()
        if not part:
            continue
        if part not in GRAN_MAP:
            raise argparse.ArgumentTypeError(f'未知颗粒度: {part}')
        result.append(GRAN_MAP[part])
    return result


def actual_numbers(row, lottery_type):
    if lottery_type == 'ssq':
        main = sorted(int(row[f'red_{i}']) for i in range(1, 7))
        aux = sorted([int(row['blue'])])
    else:
        main = sorted(int(row[f'front_{i}']) for i in range(1, 6))
        aux = sorted(int(row[f'back_{i}']) for i in range(1, 3))
    return main, aux


def compute_votes(gran_predictions, merger):
    """按合并器同样的规则，计算每个号码的原始票数。"""
    main_votes = defaultdict(float)
    aux_votes = defaultdict(float)

    for gran_name, methods in gran_predictions.items():
        for method_key, result in methods.items():
            if method_key == 'comprehensive':
                continue
            if not isinstance(result, dict) or 'error' in result:
                continue
            weight = merger.compute_weight(gran_name, method_key)
            pred = result.get('predictions', {})
            for num in pred.get(merger.main_name, []) or []:
                if merger.main_range[0] <= num <= merger.main_range[1]:
                    main_votes[num] += weight
            for num in pred.get(merger.aux_name, []) or []:
                if merger.aux_range[0] <= num <= merger.aux_range[1]:
                    aux_votes[num] += weight
    return main_votes, aux_votes


def ranks_for(actual, votes, number_range):
    """返回实际号码在原始票数排序中的名次（1 为最高）。"""
    numbers = list(range(number_range[0], number_range[1] + 1))
    ordered = sorted(numbers, key=lambda n: (-votes.get(n, 0.0), n))
    rank = {n: i + 1 for i, n in enumerate(ordered)}
    return [rank[n] for n in actual]


def near_hits(actual, predicted, threshold):
    """实际号码中，与任一预测号码距离 <= threshold 的个数。"""
    count = 0
    for a in actual:
        if any(abs(a - p) <= threshold for p in predicted):
            count += 1
    return count


def load_config_source(args):
    """按优先级加载配置：combo-id > 指定文件 > 当前配置。"""
    if args.combo_id is not None:
        history_path = os.path.join('logs', 'backtest_history.json')
        with open(history_path, 'r', encoding='utf-8') as f:
            combos = json.load(f).get('combos', [])
        matches = [e for e in combos if e.get('combo_id') == args.combo_id]
        if not matches:
            raise SystemExit(f'历史记录中未找到 combo_id={args.combo_id}')
        entry = matches[-1]
        params = entry.get('params_snapshot', {})
        weights = entry.get('weights_snapshot', {}).get('composite_weights', {})
        meta = {
            'source': 'history',
            'combo_id': args.combo_id,
            'avg_hits': entry.get('avg_hits'),
            'max_hits': entry.get('max_hits'),
            'timestamp': entry.get('timestamp'),
        }
        return params, weights, meta

    if args.params_file or args.weights_file:
        params, weights = {}, {}
        if args.params_file:
            with open(args.params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
        if args.weights_file:
            with open(args.weights_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            weights = loaded.get('composite_weights', loaded)
        meta = {
            'source': 'files',
            'params_file': args.params_file,
            'weights_file': args.weights_file,
        }
        return params, weights, meta

    cm = ConfigManager('logs', args.lottery)
    params = cm.current_params
    weights = cm.current_weights.get('composite_weights', {})
    meta = {
        'source': 'current',
        'version': cm.current_params.get('_meta', {}).get('version'),
        'weight_version': cm.current_weights.get('_meta', {}).get('version'),
        'combo_id': cm.current_params.get('_meta', {}).get('combo_id'),
    }
    return params, weights, meta


def evaluate(args):
    data_file = args.data
    if not data_file:
        data_file = '双色球.xlsx' if args.lottery == 'ssq' else '大乐透.xlsx'

    print('=' * 64)
    print(f'基线评估 | 彩种={args.lottery} | 数据={data_file}')
    print('=' * 64)

    data_rev, lottery_type = LotteryPredictor.load_data(data_file)
    if lottery_type != args.lottery:
        print(f'警告: 数据识别为 {lottery_type}，与参数 {args.lottery} 不一致')
        args.lottery = lottery_type

    raw_params, weights, source_meta = load_config_source(args)
    params = {k: v for k, v in raw_params.items() if k != '_meta'}
    param_meta = raw_params.get('_meta', {})
    weight_meta = {'version': source_meta.get('weight_version')}

    print(f'配置来源: {source_meta}')
    print(f'参数版本: v{param_meta.get("version", "?")} | '
          f'权重版本: v{weight_meta.get("version", "?")} | '
          f'权重数量: {len(weights)}')

    predictor = LotteryPredictor(args.lottery)
    merger = ResultMerger(args.lottery)
    merger.import_weights({'composite_weights': weights})

    max_train = 0 if args.true_windows else args.max_train
    actual_test_count = min(args.periods, len(data_rev) - 10)
    granularities = args.granularities

    print(f'测试期数: {actual_test_count} | 颗粒度: '
          f'{", ".join(GRAN_NAME[g] for g in granularities)} | '
          f'训练上限: {max_train if max_train > 0 else "不限(真实窗口)"}')
    print('-' * 64)

    period_results = []
    t_all = time.perf_counter()

    for period_idx in range(actual_test_count):
        t0 = time.perf_counter()
        train_full = data_rev.iloc[period_idx + 1:]
        if len(train_full) < 20:
            continue
        if max_train > 0:
            train_full = train_full.head(min(len(train_full), max_train))

        actual_main, actual_aux = actual_numbers(data_rev.iloc[period_idx], args.lottery)

        gran_predictions = {}
        for g_idx, gran in enumerate(granularities):
            if gran == 0:
                gran_data = train_full
            else:
                gran_data = train_full.head(min(len(train_full), gran))
            if len(gran_data) < 10:
                continue
            gran_name = GRAN_NAME[gran]
            try:
                gran_predictions[gran_name] = predictor.predict_all(
                    gran_data, params=params,
                    seed=42 + period_idx * 100 + g_idx)
            except Exception as exc:
                print(f'  第{period_idx + 1}期 {gran_name} 预测异常: {exc}')

        if not gran_predictions:
            continue

        merged = merger.merge_results(gran_predictions)
        merged_main = sorted(merged['predictions'][merger.main_name])
        merged_aux = sorted(merged['predictions'][merger.aux_name])

        main_hits = len(set(merged_main) & set(actual_main))
        aux_hits = len(set(merged_aux) & set(actual_aux))
        total_hits = main_hits + aux_hits

        main_votes, aux_votes = compute_votes(gran_predictions, merger)
        main_ranks = ranks_for(actual_main, main_votes, merger.main_range)
        aux_ranks = ranks_for(actual_aux, aux_votes, merger.aux_range)

        result = {
            'period_idx': period_idx,
            'period': int(data_rev.iloc[period_idx]['period']),
            'predicted_main': merged_main,
            'predicted_aux': merged_aux,
            'actual_main': actual_main,
            'actual_aux': actual_aux,
            'main_hits': main_hits,
            'aux_hits': aux_hits,
            'total_hits': total_hits,
            'near_main_1': near_hits(actual_main, merged_main, 1),
            'near_main_2': near_hits(actual_main, merged_main, 2),
            'near_aux_1': near_hits(actual_aux, merged_aux, 1),
            'near_aux_2': near_hits(actual_aux, merged_aux, 2),
            'actual_main_avg_rank': round(sum(main_ranks) / len(main_ranks), 2),
            'actual_aux_avg_rank': round(sum(aux_ranks) / len(aux_ranks), 2),
            'elapsed': round(time.perf_counter() - t0, 1),
        }
        period_results.append(result)
        print(f'第{result["period"]}期: 主{main_hits}+辅{aux_hits}={total_hits} | '
              f'预测主 {merged_main} 辅 {merged_aux} | '
              f'实际主 {actual_main} 辅 {actual_aux} | '
              f'邻近(1/2) {result["near_main_1"]}/{result["near_main_2"]} | '
              f'实际主平均排名 {result["actual_main_avg_rank"]} | '
              f'{result["elapsed"]}s')

    if not period_results:
        print('没有可评估的期数')
        return None

    totals = [r['total_hits'] for r in period_results]
    main_hits = [r['main_hits'] for r in period_results]
    aux_hits = [r['aux_hits'] for r in period_results]
    n = len(period_results)

    summary = {
        'periods_evaluated': n,
        'avg_total_hits': round(sum(totals) / n, 4),
        'avg_main_hits': round(sum(main_hits) / n, 4),
        'avg_aux_hits': round(sum(aux_hits) / n, 4),
        'max_total_hits': max(totals),
        'min_total_hits': min(totals),
        'hit_distribution': {str(h): totals.count(h) for h in range(0, 8)},
        'rate_ge_3': round(sum(1 for h in totals if h >= 3) / n, 4),
        'rate_ge_4': round(sum(1 for h in totals if h >= 4) / n, 4),
        'rate_ge_5': round(sum(1 for h in totals if h >= 5) / n, 4),
        'avg_near_main_1': round(sum(r['near_main_1'] for r in period_results) / n, 2),
        'avg_near_main_2': round(sum(r['near_main_2'] for r in period_results) / n, 2),
        'avg_near_aux_1': round(sum(r['near_aux_1'] for r in period_results) / n, 2),
        'avg_near_aux_2': round(sum(r['near_aux_2'] for r in period_results) / n, 2),
        'avg_actual_main_rank': round(
            sum(r['actual_main_avg_rank'] for r in period_results) / n, 2),
        'avg_actual_aux_rank': round(
            sum(r['actual_aux_avg_rank'] for r in period_results) / n, 2),
        'total_elapsed': round(time.perf_counter() - t_all, 1),
    }

    report = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lottery_type': args.lottery,
        'data_file': data_file,
        'periods_requested': args.periods,
        'granularities': [GRAN_NAME[g] for g in granularities],
        'max_train': max_train if max_train > 0 else None,
        'true_windows': bool(args.true_windows),
        'param_version': param_meta.get('version'),
        'weight_version': weight_meta.get('version'),
        'combo_id': param_meta.get('combo_id'),
        'config_source': source_meta,
        'summary': summary,
        'periods': period_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = (f'baseline_{args.lottery}_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('-' * 64)
    print('基线汇总:')
    print(f'  评估期数: {n}')
    print(f'  平均总命中: {summary["avg_total_hits"]}')
    print(f'  平均主球命中: {summary["avg_main_hits"]} | 平均辅助命中: {summary["avg_aux_hits"]}')
    print(f'  最高/最低总命中: {summary["max_total_hits"]} / {summary["min_total_hits"]}')
    print(f'  命中分布: {summary["hit_distribution"]}')
    print(f'  ≥3 / ≥4 / ≥5 比例: {summary["rate_ge_3"]:.1%} / '
          f'{summary["rate_ge_4"]:.1%} / {summary["rate_ge_5"]:.1%}')
    print(f'  邻近命中均值(主1/主2/辅1/辅2): '
          f'{summary["avg_near_main_1"]} / {summary["avg_near_main_2"]} / '
          f'{summary["avg_near_aux_1"]} / {summary["avg_near_aux_2"]}')
    print(f'  实际号码平均排名(主/辅): '
          f'{summary["avg_actual_main_rank"]} / {summary["avg_actual_aux_rank"]}')
    print(f'  总耗时: {summary["total_elapsed"]} 秒')
    print(f'报告已保存: {out_path}')
    return out_path


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description='基线评估（Phase 0）')
    parser.add_argument('--lottery', choices=['ssq', 'dlt'], default='ssq',
                        help='彩票类型')
    parser.add_argument('--data', default=None, help='数据文件路径')
    parser.add_argument('--periods', type=int, default=5,
                        help='评估最近多少期（默认5）')
    parser.add_argument('--granularities', type=parse_granularities,
                        default=parse_granularities('100,500'),
                        help='逗号分隔的颗粒度：50,100,500,1000,all')
    parser.add_argument('--max-train', type=int, default=500,
                        help='每个测试期训练集上限（0=不限，默认500）')
    parser.add_argument('--true-windows', action='store_true',
                        help='按真实窗口计算（等价于 --max-train 0）')
    parser.add_argument('--combo-id', type=int, default=None,
                        help='从 logs/backtest_history.json 读取指定 combo 的参数和权重')
    parser.add_argument('--params-file', default=None,
                        help='指定参数 JSON 文件（覆盖当前配置）')
    parser.add_argument('--weights-file', default=None,
                        help='指定权重 JSON 文件（覆盖当前配置）')
    parser.add_argument('--output-dir', default='baseline_reports',
                        help='报告输出目录')
    args = parser.parse_args()

    evaluate(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
