"""
回测与优化引擎 4.0
==================
全新回测系统，核心特性：
- 可配置搜索时间上限（用户设定最大搜索分钟/小时数）
- 参数网格搜索 + 随机搜索混合策略
- 确定性评估（相同参数+相同种子→相同结果）
- 自动学习合并权重
- 版本日志记录（可回退）
- 支持用户指定测试期数（1期/5期/10期/50期/自定义N期）
"""

import os
import sys
import time
import json
import hashlib
import traceback
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue as queue_module

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 导入预测器和合并器
from predictor import (
    LotteryPredictor, DEFAULT_PARAMS,
    _deterministic_top_k, _compute_frequencies,
)
from merger import ResultMerger, METHOD_NAMES, GRANULARITY_NAMES

# ============================================================================
#  参数搜索空间定义
# ============================================================================

# 各方法的可调参数及其范围（用于网格/随机搜索）
PARAM_SEARCH_SPACE = {
    'statistical': {
        'freq_weight': [0.4, 0.5, 0.6, 0.7, 0.8],
        'missing_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
        'hot_boost': [0.05, 0.10, 0.15, 0.20],
        'cold_penalty': [0.05, 0.10, 0.15, 0.20],
    },
    'timeseries': {
        'window_weights_short': [0.4, 0.5, 0.6],    # 短期窗口权重
        'window_weights_mid': [0.2, 0.3, 0.4],       # 中期
        'trend_bonus': [0.1, 0.2, 0.3, 0.4],
        'sum_tolerance': [5, 8, 12, 15],
    },
    'pattern': {
        'zone_weight': [0.10, 0.15, 0.20, 0.25],
        'prime_weight': [0.10, 0.15, 0.20],
        'digit_weight': [0.10, 0.15, 0.20, 0.25],
        'freq_weight': [0.35, 0.40, 0.45, 0.50],
        'consecutive_threshold': [0.25, 0.30, 0.35, 0.40],
    },
    'ml': {
        'n_estimators': [50, 100, 150],
        'max_depth': [6, 8, 10, 12],
        'min_samples_split': [3, 5, 8],
    },
    'markov': {
        'transition_weight': [0.5, 0.6, 0.7, 0.8],
        'base_freq_weight': [0.2, 0.3, 0.4, 0.5],
    },
    'montecarlo': {
        'num_simulations': [1000, 2000, 3000, 5000],
        'sum_sigma_range': [1.5, 2.0, 2.5],
        'in_range_bonus': [1.2, 1.5, 2.0],
    },
    'clustering': {
        'n_clusters_min': [2, 3],
        'n_clusters_max': [4, 5, 6],
    },
    'ngram': {
        'ngram_size': [2, 3],
        'similarity_threshold': [0.2, 0.25, 0.30, 0.35],
        'adjacent_weight': [0.5, 0.6, 0.75, 0.85],
        'top_k_similar': [5, 10, 15, 20],
    },
}

# 快速搜索空间（时间有限时使用，更少的组合）
FAST_PARAM_SPACE = {
    'statistical': {
        'freq_weight': [0.5, 0.6, 0.7],
        'missing_weight': [0.3, 0.4, 0.5],
    },
    'timeseries': {
        'window_weights_short': [0.4, 0.5, 0.6],
        'trend_bonus': [0.2, 0.3],
        'sum_tolerance': [5, 10],
    },
    'pattern': {
        'zone_weight': [0.15, 0.20, 0.25],
        'prime_weight': [0.10, 0.15, 0.20],
        'freq_weight': [0.35, 0.45],
    },
    'ml': {
        'n_estimators': [50, 100],
        'max_depth': [6, 10],
    },
    'markov': {
        'transition_weight': [0.6, 0.7],
    },
    'montecarlo': {
        'num_simulations': [1000, 3000],
        'sum_sigma_range': [2.0, 2.5],
    },
    'clustering': {
        'n_clusters_max': [4, 5],
    },
    'ngram': {
        'similarity_threshold': [0.25, 0.30],
        'adjacent_weight': [0.6, 0.75],
        'top_k_similar': [10, 15],
    },
}


# ============================================================================
#  回测引擎
# ============================================================================

class BacktestEngine:
    """回测与优化引擎 4.0"""

    def __init__(self):
        # 核心数据
        self.data_reverse = None       # 完整数据（倒序）
        self.lottery_type = ''
        self.predictor = None

        # 回测配置
        self.test_periods = 50          # 测试期数（用户可设定）
        self.granularities = [50, 100, 500, 1000, 0]
        self.methods = [f'method_{i}' for i in range(1, 9)]

        # 结果
        self.results: List[Dict] = []
        self.best_params: Dict = {}
        self.best_weights: Dict = {}
        self.best_score = 0.0

        # 进度
        self.running = False
        self.progress_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None
        self.start_time = 0.0
        self.max_search_time = 0  # 0 = 不限制

    def load_data(self, filepath: str) -> Tuple[bool, str]:
        """加载数据"""
        try:
            data_rev, lt = LotteryPredictor.load_data(filepath)
            self.data_reverse = data_rev
            self.lottery_type = lt
            self.predictor = LotteryPredictor(lt)
            return True, f"加载成功: {len(data_rev)}条{lt}记录"
        except Exception as e:
            return False, f"加载失败: {e}"

    def set_test_config(self, test_periods: int = 50,
                        granularities: List[int] = None,
                        methods: List[str] = None,
                        max_search_time: int = 0):
        """
        设置回测配置。

        参数:
            test_periods: 测试期数（1/5/10/50/自定义）
            granularities: 颗粒度列表
            methods: 方法列表
            max_search_time: 最大搜索时间（秒），0=不限制
        """
        self.test_periods = test_periods
        if granularities is not None:
            self.granularities = granularities
        if methods is not None:
            self.methods = methods
        self.max_search_time = max_search_time

    def set_callbacks(self, progress: Optional[Callable] = None,
                      log: Optional[Callable] = None):
        """设置进度和日志回调函数"""
        self.progress_callback = progress
        self.log_callback = log

    def _log(self, msg: str):
        """输出日志"""
        ts = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        print(full_msg)
        if self.log_callback:
            self.log_callback(full_msg)

    def _progress(self, pct: float, msg: str = ""):
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(pct, msg)

    # ========================================================================
    #  单期评估
    # ========================================================================

    def evaluate_single_period(self, period_idx: int, granularity: int,
                                params: Dict, seed: int = 0
                                ) -> Optional[Dict]:
        """
        对单独一期评估给定参数。

        参数:
            period_idx: 评估的期数索引（在data_reverse中的位置）
            granularity: 颗粒度
            params: 方法参数字典
            seed: 确定性种子

        返回:
            {period_idx, granularity, method_key, predictions, actual, hits, ...}
        """
        if self.data_reverse is None or self.predictor is None:
            return None

        total = len(self.data_reverse)

        # 训练数据：period_idx之后的所有数据
        if period_idx >= total - 1:
            return None

        train_data = self.data_reverse.iloc[period_idx + 1:].copy()
        if granularity > 0 and len(train_data) > granularity:
            train_data = train_data.head(granularity)

        if len(train_data) < 10:
            return None

        # 实际开奖号码
        actual_row = self.data_reverse.iloc[period_idx]
        if self.lottery_type == 'ssq':
            actual_main = sorted([int(actual_row[f'red_{i}']) for i in range(1, 7)])
            actual_aux = sorted([int(actual_row['blue'])])
        else:
            actual_main = sorted([int(actual_row[f'front_{i}']) for i in range(1, 6)])
            actual_aux = sorted([int(actual_row[f'back_{i}']) for i in range(1, 3)])

        # 运行预测
        try:
            all_results = self.predictor.predict_all(
                train_data, params=params, seed=seed,
                methods=[k.replace('method_', '').replace('1', 'statistical')
                        .replace('2', 'timeseries').replace('3', 'pattern')
                        .replace('4', 'ml').replace('5', 'markov')
                        .replace('6', 'montecarlo').replace('7', 'clustering')
                        .replace('8', 'ngram')
                        for k in self.methods]
            )
        except Exception as e:
            return None

        # 评估每个方法
        records = []
        actual_main_set = set(actual_main)
        actual_aux_set = set(actual_aux)

        for mk in self.methods:
            if mk not in all_results or 'error' in all_results[mk]:
                continue
            pred = all_results[mk].get('predictions', {})
            pred_main = pred.get(self.predictor.main_name, [])
            pred_aux = pred.get(self.predictor.aux_name, [])

            main_hits = len(set(pred_main) & actual_main_set)
            aux_hits = len(set(pred_aux) & actual_aux_set)

            # 评分：主球和辅助球分别计分
            if self.lottery_type == 'ssq':
                score = main_hits * 1.0 + aux_hits * 2.0  # 蓝球加权
            else:
                score = main_hits * 1.0 + aux_hits * 1.5  # 后区加权

            records.append({
                'period_idx': period_idx,
                'period_num': total - period_idx,
                'granularity': granularity,
                'granularity_text': '全部期' if granularity == 0 else f'{granularity}期',
                'method_key': mk,
                'method_name': METHOD_NAMES.get(mk, mk),
                'pred_main': pred_main,
                'pred_aux': pred_aux,
                'actual_main': actual_main,
                'actual_aux': actual_aux,
                'main_hits': main_hits,
                'aux_hits': aux_hits,
                'total_hits': main_hits + aux_hits,
                'score': score,
            })

        return records

    # ========================================================================
    #  参数组合评估（多期平均）
    # ========================================================================

    def evaluate_params(self, params: Dict,
                         test_start_idx: int,
                         test_count: int,
                         seed: int = 0
                         ) -> Dict[str, Any]:
        """
        对给定参数组合在连续N期上评估。

        参数:
            params: 方法参数字典
            test_start_idx: 测试起始位置（在data_reverse中的索引）
            test_count: 测试期数
            seed: 基础种子

        返回:
            {
                'params': params,
                'results': [...],  # 各期各方法的评估记录
                'summary': {...},  # 汇总统计
                'avg_total_hits': float,
                'max_total_hits': int,
                'best_period': int,
                'evaluation_time': float,
            }
        """
        t0 = time.time()
        all_records = []

        for i in range(test_count):
            period_idx = test_start_idx + i
            if period_idx >= len(self.data_reverse) - 1:
                break

            for gran in self.granularities:
                if gran > 0 and (period_idx + 1) < gran:
                    continue  # 数据不够
                records = self.evaluate_single_period(
                    period_idx, gran, params, seed=seed + i)
                if records:
                    all_records.extend(records)

        elapsed = time.time() - t0

        if not all_records:
            return {
                'params': params,
                'results': [],
                'avg_total_hits': 0,
                'max_total_hits': 0,
                'evaluation_time': elapsed,
            }

        # 汇总统计
        df = pd.DataFrame(all_records)

        # 计算综合评分：取各方法预测 → 加权合并 → 与实际的命中
        # 这里用各方法独立命中数汇总
        summary = {
            'total_evaluations': len(all_records),
            'avg_main_hits': round(df['main_hits'].mean(), 3),
            'avg_aux_hits': round(df['aux_hits'].mean(), 3),
            'avg_total_hits': round(df['total_hits'].mean(), 3),
            'max_total_hits': int(df['total_hits'].max()),
            'median_total_hits': round(df['total_hits'].median(), 3),
        }

        # 计算合并后的命中率（用简单等权合并模拟实际使用场景）
        # 对每期，将所有方法的预测合并（加权投票）
        period_merged_scores = []
        for period_idx in df['period_idx'].unique():
            period_records = df[df['period_idx'] == period_idx]
            if len(period_records) == 0:
                continue

            # 投票合并该期所有方法的预测
            main_votes = defaultdict(float)
            aux_votes = defaultdict(float)
            actual_main = None
            actual_aux = None

            for _, row in period_records.iterrows():
                if actual_main is None:
                    actual_main = set(row['actual_main'])
                    actual_aux = set(row['actual_aux'])
                for num in row['pred_main']:
                    main_votes[num] += 1
                for num in row['pred_aux']:
                    aux_votes[num] += 1

            # Top-K选号
            main_sorted = sorted(main_votes.items(), key=lambda x: x[1], reverse=True)
            aux_sorted = sorted(aux_votes.items(), key=lambda x: x[1], reverse=True)

            merged_main = [n for n, _ in main_sorted[:self.predictor.main_count]]
            merged_aux = [n for n, _ in aux_sorted[:self.predictor.aux_count]]

            merged_main_hits = len(set(merged_main) & actual_main)
            merged_aux_hits = len(set(merged_aux) & actual_aux)

            if self.lottery_type == 'ssq':
                merged_score = merged_main_hits * 1.0 + merged_aux_hits * 2.0
            else:
                merged_score = merged_main_hits * 1.0 + merged_aux_hits * 1.5

            period_merged_scores.append({
                'period_idx': period_idx,
                'merged_total_hits': merged_main_hits + merged_aux_hits,
                'merged_score': merged_score,
            })

        if period_merged_scores:
            ms_df = pd.DataFrame(period_merged_scores)
            summary['merged_avg_hits'] = round(ms_df['merged_total_hits'].mean(), 3)
            summary['merged_max_hits'] = int(ms_df['merged_total_hits'].max())
            summary['merged_avg_score'] = round(ms_df['merged_score'].mean(), 3)

        # 寻找最佳单期
        best_row = df.loc[df['total_hits'].idxmax()]
        best_period = int(best_row['period_num'])

        return {
            'params': params,
            'results': all_records,
            'summary': summary,
            'avg_total_hits': summary['avg_total_hits'],
            'max_total_hits': summary['max_total_hits'],
            'merged_avg_hits': summary.get('merged_avg_hits', summary['avg_total_hits']),
            'merged_max_hits': summary.get('merged_max_hits', summary['max_total_hits']),
            'best_period': best_period,
            'evaluation_time': elapsed,
        }

    # ========================================================================
    #  参数搜索
    # ========================================================================

    def _flatten_params(self, param_dict: Dict) -> Dict:
        """将嵌套的搜索空间展开为平铺的参数组合"""
        # 处理 window_weights 特殊字段
        flat = dict(param_dict)

        # 重建 window_weights 列表
        if 'window_weights_short' in flat:
            short = flat.pop('window_weights_short')
            mid = flat.pop('window_weights_mid', 0.3)
            long_w = round(1.0 - short - mid, 2)
            if long_w > 0:
                flat['window_weights'] = [short, mid, long_w]
                flat['window_ratios'] = [0.20, 0.50, 1.00]

        return flat

    def _generate_param_combinations(self, search_space: Dict,
                                      max_combinations: int = 500
                                      ) -> List[Dict]:
        """
        生成参数组合列表。
        随机采样以避免组合爆炸。

        参数:
            search_space: 各方法的参数搜索范围
            max_combinations: 最大组合数

        返回:
            [{method_name: {param: value}, ...}, ...]
        """
        # 计算总组合数
        total_combos = 1
        method_params_list = []
        for method_name, space in search_space.items():
            method_combos = []
            keys = list(space.keys())
            if not keys:
                continue

            # 对该方法生成参数组合（随机采样）
            n_combos = 1
            for k in keys:
                n_combos *= len(space[k])

            # 如果组合太多，随机采样
            rng = np.random.RandomState(42)
            max_per_method = min(n_combos, max(50, max_combinations // len(search_space)))

            sampled_configs = []
            for _ in range(max_per_method):
                config = {}
                for k in keys:
                    config[k] = space[k][rng.randint(0, len(space[k]))]
                if config not in sampled_configs:
                    sampled_configs.append(config)
                if len(sampled_configs) >= max_per_method:
                    break

            method_params_list.append((method_name, sampled_configs))
            total_combos *= len(sampled_configs)

        # 组合各方法的参数
        if total_combos > max_combinations:
            # 随机配对
            rng = np.random.RandomState(42)
            param_combinations = []
            for _ in range(max_combinations):
                combo = {}
                for method_name, configs in method_params_list:
                    combo[method_name] = configs[rng.randint(0, len(configs))]
                param_combinations.append(combo)
        else:
            # 笛卡尔积（组合数合理时）
            param_combinations = [{}]
            for method_name, configs in method_params_list:
                new_combos = []
                for combo in param_combinations:
                    for config in configs:
                        new_combo = dict(combo)
                        new_combo[method_name] = config
                        new_combos.append(new_combo)
                param_combinations = new_combos

        return param_combinations

    # ========================================================================
    #  主回测流程
    # ========================================================================

    def run_backtest(self, search_mode: str = 'grid',
                      max_combinations: int = 100,
                      test_periods: int = 50,
                      use_fast_space: bool = False
                      ) -> Dict[str, Any]:
        """
        运行回测搜索。

        参数:
            search_mode: 'grid'(网格采样) / 'baseline'(仅默认参数)
            max_combinations: 最大尝试的参数组合数
            test_periods: 测试期数（即N期命中率中的N）
            use_fast_space: 是否使用快速搜索空间（更少组合）

        返回:
            {
                'best_params': {...},
                'best_score': float,
                'all_results': [...],
                'evaluation_time': float,
                'total_combinations_tried': int,
                ...
            }
        """
        if self.data_reverse is None:
            raise ValueError("请先加载数据")

        self.running = True
        self.start_time = time.time()

        total_data = len(self.data_reverse)
        # 测试区：从第 test_periods 期开始，往前 test_periods 期
        test_start_idx = max(test_periods // 2, 20)  # 保留一些前期数据
        actual_test_count = min(test_periods,
                                total_data - test_start_idx - 10)

        self._log(f"回测配置: {actual_test_count}个测试期, "
                  f"颗粒度{self.granularities}, "
                  f"最多{max_combinations}组参数")

        # 选择搜索空间
        if search_mode == 'baseline':
            param_combinations = [{}]  # 使用全部默认参数
        elif use_fast_space:
            param_combinations = self._generate_param_combinations(
                FAST_PARAM_SPACE, max_combinations)
        else:
            param_combinations = self._generate_param_combinations(
                PARAM_SEARCH_SPACE, max_combinations)

        self._log(f"生成{len(param_combinations)}组参数组合")

        # 逐个评估（可在后续改为并行）
        all_evaluations = []
        best_eval = None
        best_score = -1

        # 估算单次评估时间
        estimated_time_per_eval = None

        for i, params_combo in enumerate(param_combinations):
            if not self.running:
                break

            # 时间检查
            elapsed = time.time() - self.start_time
            if self.max_search_time > 0 and elapsed > self.max_search_time:
                self._log(f"达到时间上限({self.max_search_time}秒)，停止搜索")
                break

            # 进度报告
            pct = (i + 1) / len(param_combinations) * 100
            if estimated_time_per_eval is None and i > 0:
                estimated_time_per_eval = elapsed / (i + 1)
                eta = estimated_time_per_eval * (len(param_combinations) - i - 1)
                self._progress(pct, f"第{i+1}/{len(param_combinations)}组, "
                              f"预计剩余{eta:.0f}秒")
            else:
                self._progress(pct, f"第{i+1}/{len(param_combinations)}组")

            # 评估
            eval_result = self.evaluate_params(
                params_combo, test_start_idx, actual_test_count, seed=i)

            eval_result['combo_index'] = i
            all_evaluations.append(eval_result)

            # 检查是否更好
            current_score = eval_result.get('merged_avg_hits',
                                            eval_result.get('avg_total_hits', 0))
            if current_score > best_score:
                best_score = current_score
                best_eval = eval_result
                self._log(f"  ★ 新最佳: 参数组#{i}, 合并平均命中={current_score:.3f}, "
                          f"合并最高命中={eval_result.get('merged_max_hits', 'N/A')}")

            # 每10组输出一次
            if (i + 1) % 10 == 0:
                self._log(f"  进度: {i+1}/{len(param_combinations)}, "
                          f"当前最佳={best_score:.3f}")

        total_time = time.time() - self.start_time

        # 构建结果
        if best_eval is None:
            self._log("错误: 所有参数组合评估失败")
            return {'success': False, 'error': '所有参数组合评估失败'}

        self.best_params = best_eval['params']
        self.best_score = best_score
        self.results = all_evaluations

        # 学习最优权重
        self._learn_optimal_weights(best_eval, test_start_idx, actual_test_count)

        self.running = False

        return {
            'success': True,
            'best_params': best_eval['params'],
            'best_score': best_score,
            'best_summary': best_eval['summary'],
            'best_merged_avg_hits': best_eval.get('merged_avg_hits', 0),
            'best_merged_max_hits': best_eval.get('merged_max_hits', 0),
            'best_period': best_eval.get('best_period', 0),
            'total_combinations': len(param_combinations),
            'total_combinations_tried': len(all_evaluations),
            'total_time': total_time,
            'test_periods': actual_test_count,
            'all_evaluations': all_evaluations,
            'optimal_weights': {
                'method_weights': self.best_weights.get('method_weights', {}),
                'granularity_weights': self.best_weights.get('granularity_weights', {}),
            },
        }

    def _learn_optimal_weights(self, best_eval: Dict,
                                test_start: int, test_count: int):
        """从最佳参数的结果中学习最优合并权重"""
        all_records = best_eval.get('results', [])
        if not all_records:
            return

        df = pd.DataFrame(all_records)

        # 方法权重
        method_perf = df.groupby('method_key')['total_hits'].agg(['mean', 'max'])
        method_perf = method_perf[method_perf['mean'] > 0]
        if not method_perf.empty:
            total = method_perf['mean'].sum()
            mw = {}
            for m in method_perf.index:
                mw[m] = round(float(method_perf.loc[m, 'mean'] / total * len(method_perf)), 4)
            self.best_weights['method_weights'] = mw

        # 颗粒度权重
        gran_perf = df.groupby('granularity_text')['total_hits'].agg(['mean', 'max'])
        if not gran_perf.empty:
            total = gran_perf['mean'].sum()
            gw = {}
            for g in gran_perf.index:
                gw[g] = round(float(gran_perf.loc[g, 'mean'] / total * len(gran_perf)), 4)
            self.best_weights['granularity_weights'] = gw

    def stop(self):
        """停止回测"""
        self.running = False

    # ========================================================================
    #  报告生成
    # ========================================================================

    def generate_report(self, output_dir: str = "backtest_reports") -> str:
        """生成回测报告Excel"""
        if not self.results:
            return ""

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lottery_name = "双色球" if self.lottery_type == 'ssq' else "大乐透"
        fname = f"回测报告_{lottery_name}_{ts}.xlsx"
        fpath = os.path.join(output_dir, fname)

        # 收集所有详细记录
        all_rows = []
        for eval_result in self.results:
            for rec in eval_result.get('results', []):
                all_rows.append({
                    '参数组': eval_result.get('combo_index', 0),
                    '期号': rec.get('period_num', 0),
                    '颗粒度': rec.get('granularity_text', ''),
                    '方法': rec.get('method_name', ''),
                    '主球命中': rec.get('main_hits', 0),
                    '辅助球命中': rec.get('aux_hits', 0),
                    '总命中': rec.get('total_hits', 0),
                    '评分': rec.get('score', 0),
                })

        if not all_rows:
            return ""

        df = pd.DataFrame(all_rows)

        with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
            # 摘要
            summary_rows = [
                ["回测报告", ""],
                ["彩票类型", lottery_name],
                ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["测试期数", self.test_periods],
                ["总评估数", len(df)],
                ["平均总命中", f"{df['总命中'].mean():.3f}"],
                ["最高总命中", f"{int(df['总命中'].max())}"],
                ["总耗时", f"{self.start_time:.1f}秒"],
                ["", ""],
                ["最优合并平均命中", f"{self.best_score:.3f}"],
                ["最优参数", json.dumps(self.best_params, ensure_ascii=False,
                                      indent=2) if self.best_params else "默认参数"],
                ["", ""],
                ["最优方法权重", json.dumps(
                    self.best_weights.get('method_weights', {}),
                    ensure_ascii=False)],
                ["最优颗粒度权重", json.dumps(
                    self.best_weights.get('granularity_weights', {}),
                    ensure_ascii=False)],
            ]
            pd.DataFrame(summary_rows, columns=["项目", "值"]).to_excel(
                writer, sheet_name="报告摘要", index=False)

            # 方法表现
            method_stats = df.groupby('方法').agg(
                平均命中=('总命中', 'mean'),
                最高命中=('总命中', 'max'),
                平均主球=('主球命中', 'mean'),
                平均辅助=('辅助球命中', 'mean'),
                评估次数=('总命中', 'count'),
            ).round(3).sort_values('平均命中', ascending=False)
            method_stats.to_excel(writer, sheet_name="方法表现")

            # 颗粒度表现
            gran_stats = df.groupby('颗粒度').agg(
                平均命中=('总命中', 'mean'),
                最高命中=('总命中', 'max'),
                评估次数=('总命中', 'count'),
            ).round(3).sort_values('平均命中', ascending=False)
            gran_stats.to_excel(writer, sheet_name="颗粒度表现")

            # 方法×颗粒度组合
            combo_stats = df.groupby(['方法', '颗粒度']).agg(
                平均命中=('总命中', 'mean'),
                最高命中=('总命中', 'max'),
            ).round(3).sort_values('平均命中', ascending=False)
            combo_stats.to_excel(writer, sheet_name="最佳组合")

            # 参数组对比
            param_comparison = []
            for eval_result in self.results:
                s = eval_result.get('summary', {})
                param_comparison.append({
                    '参数组': eval_result.get('combo_index', 0),
                    '平均总命中': s.get('avg_total_hits', 0),
                    '最高总命中': s.get('max_total_hits', 0),
                    '合并平均命中': eval_result.get('merged_avg_hits', 0),
                    '合并最高命中': eval_result.get('merged_max_hits', 0),
                    '评估耗时(秒)': round(eval_result.get('evaluation_time', 0), 1),
                })
            if param_comparison:
                pd.DataFrame(param_comparison).sort_values(
                    '合并平均命中', ascending=False).to_excel(
                    writer, sheet_name="参数组对比", index=False)

        print(f"回测报告已保存: {fpath}")
        return fpath


# ============================================================================
#  线程安全的回测运行器（供GUI调用）
# ============================================================================

class BacktestRunner:
    """在后台线程中运行回测，通过回调通知GUI"""

    def __init__(self, engine: BacktestEngine):
        self.engine = engine
        self.thread = None
        self.result = None

    def run_async(self, search_mode: str = 'grid',
                  max_combinations: int = 100,
                  test_periods: int = 50,
                  use_fast_space: bool = False,
                  max_search_time: int = 0,
                  on_progress: Callable = None,
                  on_log: Callable = None,
                  on_done: Callable = None):
        """异步运行回测"""
        self.engine.set_callbacks(on_progress, on_log)
        self.engine.max_search_time = max_search_time
        self.result = None

        def _run():
            try:
                self.result = self.engine.run_backtest(
                    search_mode=search_mode,
                    max_combinations=max_combinations,
                    test_periods=test_periods,
                    use_fast_space=use_fast_space,
                )
                if on_done:
                    on_done(self.result)
            except Exception as e:
                if on_log:
                    on_log(f"回测异常: {e}\n{traceback.format_exc()}")
                if on_done:
                    on_done({'success': False, 'error': str(e)})

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        """停止回测"""
        self.engine.stop()


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    for f in ['双色球.xlsx', '大乐透.xlsx']:
        path = os.path.join(base, f)
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"回测测试: {f}")
            print(f"{'='*60}")

            engine = BacktestEngine()
            engine.load_data(path)

            # 快速基线测试
            result = engine.run_backtest(
                search_mode='baseline',
                test_periods=5,
                use_fast_space=True,
            )

            if result['success']:
                print(f"\n测试完成!")
                print(f"  最佳合并平均命中: {result['best_merged_avg_hits']:.3f}")
                print(f"  最佳合并最高命中: {result['best_merged_max_hits']}")
                print(f"  耗时: {result['total_time']:.1f}秒")
            else:
                print(f"测试失败: {result.get('error', '未知错误')}")

            engine.generate_report()
            break
