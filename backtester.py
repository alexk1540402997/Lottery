"""
回测与优化引擎 4.1
==================
核心设计：
- 测试最新N期（index=0到N-1），历史数据做训练
- 同时搜索模型参数 + 合并权重（参数和权重都需要优化）
- 多线程并行评估，时间优先（到点即停，无参数组合上限）
- 所有尝试过的组合自动记录，后续回测自动跳过
- 支持从上次回测断点继续（去重日志持久化）

回测目标：
  找到一组(模型参数, 合并权重)，使得对最新N期的每一期：
    历史数据 → 5颗粒度×8方法=40组预测 → 加权合并 → 最终号码
    最终号码 与 真实开奖号码 命中数尽可能高
"""

import os
import sys
import time
import json
import hashlib
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from itertools import product

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from predictor import (
    LotteryPredictor, DEFAULT_PARAMS, METHOD_NAMES_NEW,
    _deterministic_top_k,
)
from merger import ResultMerger, GRANULARITY_NAMES, GRANULARITY_VALUES

# ============================================================================
#  参数 + 权重 联合搜索空间
# ============================================================================

# 各方法可调参数范围
PARAM_SEARCH_SPACE = {
    'statistical': {
        'freq_weight':       [0.4, 0.5, 0.6, 0.7, 0.8],
        'missing_weight':    [0.2, 0.3, 0.4, 0.5, 0.6],
        'hot_boost':         [0.05, 0.10, 0.15, 0.20, 0.25],
        'cold_penalty':      [0.05, 0.10, 0.15, 0.20],
    },
    'timeseries': {
        'window_weights_short': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        'window_weights_mid':   [0.20, 0.25, 0.30, 0.35, 0.40],
        'trend_bonus':       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'sum_tolerance':     [4, 6, 8, 10, 12, 15],
    },
    'pattern': {
        'zone_weight':       [0.10, 0.15, 0.20, 0.25, 0.30],
        'prime_weight':      [0.08, 0.12, 0.15, 0.20, 0.25],
        'digit_weight':      [0.10, 0.15, 0.20, 0.25],
        'freq_weight':       [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        'consecutive_threshold': [0.25, 0.30, 0.35, 0.40, 0.45],
    },
    'ml': {
        'n_estimators':      [30, 50, 80, 100],
        'max_depth':         [4, 6, 8, 10],
        'num_leaves':        [10, 15, 20, 31],
        'learning_rate':     [0.05, 0.10, 0.15],
    },
    'markov': {
        'transition_weight': [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'base_freq_weight':  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    },
    'montecarlo': {
        'num_simulations':   [1000, 2000, 3000, 5000],
        'sum_sigma_range':   [1.5, 2.0, 2.5, 3.0],
        'in_range_bonus':    [1.2, 1.5, 1.8, 2.0],
    },
    'clustering': {
        'n_clusters_min':    [2, 3],
        'n_clusters_max':    [4, 5, 6],
    },
    'ngram': {
        'similarity_threshold': [0.15, 0.20, 0.25, 0.30, 0.35],
        'adjacent_weight':      [0.50, 0.60, 0.70, 0.75, 0.85],
        'top_k_similar':        [5, 10, 15, 20, 25],
    },
    'xgboost': {
        'n_estimators':      [20, 40, 60, 80],
        'max_depth':         [3, 5, 7, 9],
        'learning_rate':     [0.05, 0.10, 0.15, 0.20],
        'subsample':         [0.7, 0.8, 0.9],
        'reg_alpha':         [0.5, 1.0, 2.0],
    },
    'bayesian': {
        'prior_strength':    [1.0, 2.0, 3.0, 5.0],
        'freq_weight':       [0.40, 0.50, 0.55, 0.60, 0.65],
        'recent_weight':     [0.15, 0.20, 0.25, 0.30, 0.35],
        'missing_weight':    [0.10, 0.15, 0.20, 0.25],
    },
    'kalman': {
        'process_noise':     [0.005, 0.01, 0.02, 0.05],
        'measurement_noise': [0.05, 0.10, 0.20, 0.30],
        'trend_weight':      [0.30, 0.40, 0.50, 0.60],
    },
    'poisson': {
        'alpha':             [0.1, 0.5, 1.0, 2.0],
        'freq_weight':       [0.40, 0.50, 0.60, 0.70],
    },
    'cooccurrence': {
        'cooccur_threshold': [0.10, 0.15, 0.20, 0.25, 0.30],
        'mutual_weight':     [0.40, 0.50, 0.55, 0.60, 0.65],
        'window_size':       [50, 100, 200, 500],
    },
}

# 合并权重搜索空间（连续值，随机采样）
WEIGHT_SEARCH_SPACE = {
    'method_weight_range': (0.3, 3.0),       # 方法权重范围
    'granularity_weight_range': (0.3, 3.0),  # 颗粒度权重范围
}


# ============================================================================
#  回测引擎
# ============================================================================

class BacktestEngine:
    """回测与优化引擎 4.1"""

    def __init__(self):
        self.data_reverse = None       # 完整数据（倒序，index=0=最新）
        self.lottery_type = ''
        self.predictor = None
        self.total_periods = 0

        # 回测配置
        self.test_periods = 50         # 测试最新N期
        self.granularities = [50, 100, 500]  # 回测用3种颗粒度加速
        self.gran_names = ['50期', '100期', '500期']
        self.max_search_time = 0       # 最大搜索时间（秒），0=不限制
        self.num_workers = 4           # 并行线程数
        self.max_train_periods = 500   # 限制训练数据最多500期（加速ML）

        # 已尝试组合去重
        self.tried_combos: Dict[str, float] = {}  # hash → best_score
        self.tried_log_file = "logs/backtest_tried_combos.json"

        # 结果
        self.best_combo = None         # 最佳(参数, 权重)
        self.best_score = 0.0
        self.best_period_results = []  # 每期详细命中
        self.all_results = []          # 所有组合的评估结果

        # 运行时状态
        self.running = False
        self.start_time = 0.0
        self.progress_callback = None
        self.log_callback = None

        # 加载历史去重记录
        self._load_tried_combos()

    # ========================================================================
    #  数据加载
    # ========================================================================

    def load_data(self, filepath: str) -> Tuple[bool, str]:
        """加载数据"""
        try:
            data_rev, lt = LotteryPredictor.load_data(filepath)
            self.data_reverse = data_rev
            self.lottery_type = lt
            self.predictor = LotteryPredictor(lt)
            self.total_periods = len(data_rev)
            return True, f"加载成功: {self.total_periods}条{lt}记录"
        except Exception as e:
            return False, f"加载失败: {e}"

    def set_config(self, test_periods: int = 50,
                   granularities: List[int] = None,
                   max_search_time: int = 0,
                   num_workers: int = 4,
                   max_train_periods: int = 500):
        """设置回测配置"""
        self.test_periods = test_periods
        if granularities is not None:
            self.granularities = granularities
            self.gran_names = [
                f'{g}期' if g > 0 else '全部期' for g in granularities]
        self.max_search_time = max_search_time
        self.num_workers = max(1, num_workers)
        self.max_train_periods = max_train_periods

    def set_callbacks(self, progress=None, log=None):
        """设置进度和日志回调"""
        self.progress_callback = progress
        self.log_callback = log

    # ========================================================================
    #  去重日志
    # ========================================================================

    def _combo_hash(self, params: Dict, weights: Dict) -> str:
        """生成参数+权重的唯一哈希"""
        raw = json.dumps({'params': params, 'weights': weights},
                        sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _load_tried_combos(self):
        """加载历史已尝试的组合记录"""
        if os.path.exists(self.tried_log_file):
            try:
                with open(self.tried_log_file, 'r', encoding='utf-8') as f:
                    self.tried_combos = json.load(f)
            except Exception:
                self.tried_combos = {}

    def _save_tried_combos(self):
        """保存已尝试组合记录"""
        os.makedirs(os.path.dirname(self.tried_log_file), exist_ok=True)
        try:
            with open(self.tried_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.tried_combos, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========================================================================
    #  参数/权重组合生成器（无限流，由时间控制停止）
    # ========================================================================

    def _sample_params(self, rng: np.random.RandomState) -> Dict[str, Dict]:
        """随机采样一组模型参数"""
        sampled = {}
        for method_name, space in PARAM_SEARCH_SPACE.items():
            config = {}
            for pname, pvalues in space.items():
                config[pname] = pvalues[rng.randint(0, len(pvalues))]
            sampled[method_name] = config
        return sampled

    def _sample_weights(self, rng: np.random.RandomState) -> Dict[str, Dict]:
        """随机采样一组合并权重"""
        m_min, m_max = WEIGHT_SEARCH_SPACE['method_weight_range']
        g_min, g_max = WEIGHT_SEARCH_SPACE['granularity_weight_range']

        mw = {}
        for mk in METHOD_NAMES_NEW:
            mw[mk] = round(float(rng.uniform(m_min, m_max)), 4)

        gw = {}
        for gn in GRANULARITY_NAMES:
            gw[gn] = round(float(rng.uniform(g_min, g_max)), 4)

        return {'method_weights': mw, 'granularity_weights': gw}

    def _generate_combo(self, rng: np.random.RandomState,
                        prefer_new: bool = True) -> Tuple[Dict, Dict, str]:
        """
        生成一组(参数, 权重)组合，跳过已尝试过的。

        参数:
            rng: 随机数生成器
            prefer_new: True则尽量生成新的组合，False则可能重复

        返回:
            (params, weights, combo_hash)
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            params = self._sample_params(rng)
            weights = self._sample_weights(rng)
            h = self._combo_hash(params, weights)
            if prefer_new and h not in self.tried_combos:
                return params, weights, h
            elif not prefer_new:
                return params, weights, h

        # 实在生成不了新的，就用最后一次的
        params = self._sample_params(rng)
        weights = self._sample_weights(rng)
        h = self._combo_hash(params, weights)
        return params, weights, h

    # ========================================================================
    #  核心评估：一组(参数+权重)在最新N期上的命中表现
    # ========================================================================

    def evaluate_combo(self, params: Dict, weights: Dict, seed: int = 0,
                       combo_id: int = 0) -> Dict[str, Any]:
        """
        评估一组参数+权重在最新N期上的表现。

        test_periods = N → 测试 data_reverse[0] 到 data_reverse[N-1]（最新N期）
        对每期，用该期之前的历史数据训练和预测。

        返回:
            {
                'combo_id': int,
                'params': {...},
                'weights': {...},
                'combo_hash': str,
                'period_results': [{period_idx, merged_main, merged_aux, actual_main, actual_aux, total_hits, ...}],
                'avg_total_hits': float,
                'max_total_hits': int,
                'hit_rate_5plus': float,  # 命中5+的期数占比
                'evaluation_time': float,
            }
        """
        if not self.running:
            return {'error': 'stopped'}

        t0 = time.time()
        merger = ResultMerger(self.lottery_type)
        merger.import_weights(weights)

        period_results = []
        actual_test_count = min(self.test_periods, self.total_periods - 10)

        # 早停参数
        early_check_periods = min(3, actual_test_count)  # 前N期检查
        early_stop_min_hits = 1.5  # 前N期平均低于此值则跳过

        for period_idx in range(actual_test_count):
            # 检查时间（细粒度检查）
            if self.max_search_time > 0:
                elapsed = time.time() - self.start_time
                if elapsed > self.max_search_time:
                    break

            if not self.running:
                break

            # 早停检查：前N期表现太差则跳过（仅在测试期数较多时启用）
            if (period_idx >= early_check_periods and
                    actual_test_count >= 10 and period_results):
                recent_hits = [r['total_hits'] for r in period_results[:early_check_periods]]
                recent_avg = sum(recent_hits) / len(recent_hits)
                if recent_avg < early_stop_min_hits:
                    break

            # 训练数据：period_idx之后的历史（该期之前的数据）
            # 限制最大训练期数加速（500期足够捕捉趋势）
            full_train = self.data_reverse.iloc[period_idx + 1:]
            if len(full_train) < 20:
                continue
            train_data = full_train.head(
                min(len(full_train), self.max_train_periods)).copy()

            # 实际开奖号码
            actual_row = self.data_reverse.iloc[period_idx]
            if self.lottery_type == 'ssq':
                actual_main = sorted([int(actual_row[f'red_{i}']) for i in range(1, 7)])
                actual_aux = sorted([int(actual_row['blue'])])
            else:
                actual_main = sorted([int(actual_row[f'front_{i}']) for i in range(1, 6)])
                actual_aux = sorted([int(actual_row[f'back_{i}']) for i in range(1, 3)])

            # 对5种颗粒度分别预测
            gran_predictions = {}
            for g_idx, gran in enumerate(self.granularities):
                if gran > 0 and len(train_data) < gran:
                    gran_data = train_data
                elif gran > 0:
                    gran_data = train_data.head(gran)
                else:
                    gran_data = train_data

                if len(gran_data) < 10:
                    continue

                try:
                    gran_results = self.predictor.predict_all(
                        gran_data, params=params, seed=seed + period_idx * 100 + g_idx)
                    gran_name = self.gran_names[g_idx] if g_idx < len(self.gran_names) else f'{gran}期'
                    gran_predictions[gran_name] = gran_results
                except Exception:
                    continue

            if not gran_predictions:
                continue

            # 用当前权重合并40组结果
            merged = merger.merge_results(gran_predictions)
            merged_main = merged['predictions'][self.predictor.main_name]
            merged_aux = merged['predictions'][self.predictor.aux_name]

            main_hits = len(set(merged_main) & set(actual_main))
            aux_hits = len(set(merged_aux) & set(actual_aux))

            period_results.append({
                'period_idx': period_idx,
                'period_num': self.total_periods - period_idx,
                'merged_main': sorted(merged_main),
                'merged_aux': sorted(merged_aux),
                'actual_main': actual_main,
                'actual_aux': actual_aux,
                'main_hits': main_hits,
                'aux_hits': aux_hits,
                'total_hits': main_hits + aux_hits,
            })

        elapsed = time.time() - t0

        if not period_results:
            return {
                'combo_id': combo_id,
                'params': params,
                'weights': weights,
                'period_results': [],
                'avg_total_hits': 0,
                'max_total_hits': 0,
                'hit_rate_5plus': 0,
                'evaluation_time': elapsed,
            }

        total_hits_list = [r['total_hits'] for r in period_results]
        avg_hits = np.mean(total_hits_list)
        max_hits = max(total_hits_list)
        hit_5plus = sum(1 for h in total_hits_list if h >= 5) / len(total_hits_list)

        return {
            'combo_id': combo_id,
            'params': params,
            'weights': weights,
            'period_results': period_results,
            'avg_total_hits': round(float(avg_hits), 4),
            'max_total_hits': int(max_hits),
            'hit_rate_5plus': round(float(hit_5plus), 4),
            'evaluation_time': round(elapsed, 1),
            'num_periods_evaluated': len(period_results),
        }

    # ========================================================================
    #  主回测循环
    # ========================================================================

    def run(self, num_combos_to_try: int = None) -> Dict[str, Any]:
        """
        运行回测搜索。

        参数:
            num_combos_to_try: 尝试的组合数上限（None=无上限，仅由时间控制）

        返回:
            {
                'success': bool,
                'best_combo': {...},         # 最佳组合详情
                'best_score': float,         # 最佳平均命中
                'total_combos_tried': int,   # 已尝试组合数
                'total_combos_skipped': int, # 跳过（已试过）的组合数
                'total_time': float,         # 总耗时
                'all_results': [...],        # 所有结果
            }
        """
        if self.data_reverse is None:
            return {'success': False, 'error': '请先加载数据'}

        self.running = True
        self.start_time = time.time()
        self.best_combo = None
        self.best_score = -1.0
        self.all_results = []

        actual_test_count = min(self.test_periods, self.total_periods - 10)
        self._log(f"回测启动: 测试最新{actual_test_count}期, "
                  f"颗粒度{len(self.granularities)}种, "
                  f"时间上限{'不限' if self.max_search_time == 0 else f'{self.max_search_time}秒'}, "
                  f"{self.num_workers}线程并行")
        self._log(f"已记录{len(self.tried_combos)}组已尝试组合，将自动跳过")

        # 生成初始组合池
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        combo_pool = []
        skipped = 0

        # 预先生成一批组合
        batch_size = 200
        while len(combo_pool) < batch_size:
            params, weights, h = self._generate_combo(rng, prefer_new=True)
            if h in self.tried_combos:
                skipped += 1
                # 如果太多被跳过，说明大部分组合都试过了，降低去重标准
                if skipped > batch_size * 2:
                    params, weights, h = self._generate_combo(rng, prefer_new=False)
            combo_pool.append((params, weights, h, len(combo_pool)))

        if skipped > 0:
            self._log(f"生成组合时跳过{skipped}组已尝试过的组合")

        # 多线程并行评估（一次只提交少量，避免CPU竞争）
        combo_idx = 0
        batch_submitted = 0
        active_futures = {}

        # 限制同时活跃的任务数
        max_concurrent = max(1, min(self.num_workers, 8))

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:

            # 持续提交+收集结果
            while self.running:
                # 时间检查
                if self.max_search_time > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.max_search_time:
                        self._log(f"达到时间上限({self.max_search_time}秒)，停止搜索")
                        for f in list(active_futures.keys()):
                            f.cancel()
                        break

                # 如果组合池不够，补充更多
                if batch_submitted >= len(combo_pool):
                    for _ in range(20):
                        params, weights, h = self._generate_combo(rng, prefer_new=True)
                        if h in self.tried_combos:
                            skipped += 1
                        combo_pool.append((params, weights, h, len(combo_pool)))

                # 提交新任务（使用实际的max_concurrent）
                while (len(active_futures) < max_concurrent and
                       batch_submitted < len(combo_pool)):
                    params, weights, h, cid = combo_pool[batch_submitted]
                    future = executor.submit(self.evaluate_combo, params, weights,
                                            seed=cid, combo_id=cid)
                    active_futures[future] = (cid, h, params, weights)
                    batch_submitted += 1

                if not active_futures:
                    if self.max_search_time == 0:
                        break  # 不做定时限制且没有活跃任务
                    time.sleep(0.5)
                    continue

                # 等待任意一个完成（有超时）
                try:
                    done = list(as_completed(active_futures, timeout=5.0))
                except TimeoutError:
                    continue

                for future in done:
                    cid, h, params, weights = active_futures.pop(future)
                    try:
                        result = future.result(timeout=5)
                    except Exception as e:
                        self._log(f"组合#{cid}评估异常: {e}")
                        continue

                    if 'error' in result:
                        continue

                    # 记录结果
                    self.all_results.append(result)
                    self.tried_combos[h] = result['avg_total_hits']

                    # 更新最佳
                    if result['avg_total_hits'] > self.best_score:
                        self.best_score = result['avg_total_hits']
                        self.best_combo = result
                        self._log(
                            f"★ 新最佳 #{cid}: 平均命中={result['avg_total_hits']:.3f}, "
                            f"最高={result['max_total_hits']}, "
                            f"5+率={result['hit_rate_5plus']:.1%}, "
                            f"耗时={result['evaluation_time']:.0f}s")

                    combo_idx += 1

                    # 进度报告
                    elapsed = time.time() - self.start_time
                    pct = min(95, (elapsed / self.max_search_time * 100)
                             if self.max_search_time > 0 else (combo_idx * 10))
                    status = (f"已试{combo_idx}组, "
                             f"最佳={self.best_score:.3f}, "
                             f"耗时{elapsed:.0f}s")
                    self._progress(pct, status)

                    # 组合数上限
                    if num_combos_to_try and combo_idx >= num_combos_to_try:
                        self._log(f"达到组合数上限({num_combos_to_try})")
                        for f in list(active_futures.keys()):
                            f.cancel()
                        active_futures.clear()
                        break

                if num_combos_to_try and combo_idx >= num_combos_to_try:
                    break

        total_time = time.time() - self.start_time
        self.running = False

        # 保存去重日志
        self._save_tried_combos()

        if not self.best_combo:
            return {
                'success': False,
                'error': f'在{total_time:.0f}秒内没有完成任何有效评估',
                'total_time': total_time,
            }

        self._log(f"\n回测完成! 总耗时{total_time:.0f}秒, "
                  f"尝试{len(self.all_results)}组, 跳过{skipped}组已试")
        self._log(f"最佳结果: 平均命中={self.best_score:.3f}, "
                  f"最高命中={self.best_combo['max_total_hits']}, "
                  f"5+命中率={self.best_combo['hit_rate_5plus']:.1%}")

        return {
            'success': True,
            'best_combo': self.best_combo,
            'best_score': self.best_score,
            'best_avg_hits': self.best_combo['avg_total_hits'],
            'best_max_hits': self.best_combo['max_total_hits'],
            'best_hit_rate_5plus': self.best_combo['hit_rate_5plus'],
            'total_combos_tried': len(self.all_results),
            'total_combos_skipped': skipped,
            'total_time': total_time,
            'all_results': self.all_results,
        }

    def stop(self):
        """停止回测"""
        self.running = False

    # ========================================================================
    #  日志/进度
    # ========================================================================

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        print(full_msg)
        if self.log_callback:
            self.log_callback(full_msg)

    def _progress(self, pct: float, msg: str = ""):
        if self.progress_callback:
            self.progress_callback(pct, msg)

    # ========================================================================
    #  报告生成
    # ========================================================================

    def generate_report(self, output_dir: str = "backtest_reports") -> str:
        """生成回测报告Excel"""
        if not self.best_combo:
            return ""

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lottery_name = "双色球" if self.lottery_type == 'ssq' else "大乐透"
        fname = f"回测报告_{lottery_name}_{ts}.xlsx"
        fpath = os.path.join(output_dir, fname)

        with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
            # Sheet 1: 摘要
            bc = self.best_combo
            summary = [
                ["回测报告", ""],
                ["彩票类型", lottery_name],
                ["数据总期数", self.total_periods],
                ["测试最新N期", self.test_periods],
                ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["", ""],
                ["--- 最佳结果 ---", ""],
                ["平均总命中", f"{bc['avg_total_hits']:.4f}"],
                ["最高总命中", bc['max_total_hits']],
                ["5+命中率", f"{bc['hit_rate_5plus']:.1%}"],
                ["评估期数", bc.get('num_periods_evaluated', 'N/A')],
                ["", ""],
                ["最优方法权重", json.dumps(
                    bc['weights'].get('method_weights', {}),
                    ensure_ascii=False, indent=2)],
                ["最优颗粒度权重", json.dumps(
                    bc['weights'].get('granularity_weights', {}),
                    ensure_ascii=False, indent=2)],
                ["", ""],
                ["--- 最优模型参数 ---", ""],
            ]
            # 展开模型参数（按方法+参数名显示）
            params = bc.get('params', {})
            param_method_names = {
                'statistical': '方法1: 统计概率分析',
                'timeseries': '方法2: 时间序列分析',
                'pattern': '方法3: 模式识别分析',
                'ml': '方法4: 机器学习分析',
                'markov': '方法5: 马尔可夫分析',
                'montecarlo': '方法6: 蒙特卡罗模拟',
                'clustering': '方法7: 聚类分析',
                'ngram': '方法8: N-gram分析',
            }
            for method_key, method_label in param_method_names.items():
                method_params = params.get(method_key, {})
                if method_params:
                    summary.append([method_label, json.dumps(
                        method_params, ensure_ascii=False, indent=2)])
            pd.DataFrame(summary, columns=["项目", "值"]).to_excel(
                writer, sheet_name="报告摘要", index=False)

            # Sheet 2: 每期详情
            period_rows = []
            for pr in bc.get('period_results', []):
                period_rows.append({
                    '期号': pr['period_num'],
                    '预测主球': ' '.join(f'{n:02d}' for n in pr['merged_main']),
                    '实际主球': ' '.join(f'{n:02d}' for n in pr['actual_main']),
                    '预测辅助球': ' '.join(f'{n:02d}' for n in pr['merged_aux']),
                    '实际辅助球': ' '.join(f'{n:02d}' for n in pr['actual_aux']),
                    '主球命中': pr['main_hits'],
                    '辅助球命中': pr['aux_hits'],
                    '总命中': pr['total_hits'],
                })
            if period_rows:
                pd.DataFrame(period_rows).to_excel(
                    writer, sheet_name="每期详情", index=False)

            # Sheet 3: 所有组合对比
            if self.all_results:
                combo_rows = []
                for r in self.all_results:
                    if 'error' in r:
                        continue
                    combo_rows.append({
                        '组合ID': r.get('combo_id', 0),
                        '平均总命中': r['avg_total_hits'],
                        '最高总命中': r['max_total_hits'],
                        '5+命中率': r['hit_rate_5plus'],
                        '耗时(秒)': r.get('evaluation_time', 0),
                        '组合哈希': self._combo_hash(
                            r.get('params', {}), r.get('weights', {})),
                    })
                if combo_rows:
                    pd.DataFrame(combo_rows).sort_values(
                        '平均总命中', ascending=False).to_excel(
                        writer, sheet_name="所有组合", index=False)

        print(f"回测报告已保存: {fpath}")
        return fpath


# ============================================================================
#  GUI辅助：后台运行器
# ============================================================================

import threading

class BacktestRunner:
    """在后台线程中运行回测"""

    def __init__(self, engine: BacktestEngine):
        self.engine = engine
        self.thread = None
        self.result = None

    def run_async(self,
                  on_progress=None,
                  on_log=None,
                  on_done=None):
        """异步运行回测"""
        self.engine.set_callbacks(on_progress, on_log)
        self.result = None

        def _run():
            try:
                self.result = self.engine.run()
                if on_done:
                    on_done(self.result)
            except Exception as e:
                err = {'success': False, 'error': str(e)}
                if on_log:
                    on_log(f"回测异常: {e}\n{traceback.format_exc()}")
                if on_done:
                    on_done(err)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.engine.stop()


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    for f in ['双色球.xlsx', '大乐透.xlsx']:
        path = os.path.join(base, f)
        alt = os.path.join(base, 'pythonProject', f)
        if os.path.exists(path):
            data_file = path
            break
        if os.path.exists(alt):
            data_file = alt
            break
    else:
        print("未找到数据文件")
        sys.exit(1)

    print(f"数据文件: {data_file}")
    engine = BacktestEngine()
    engine.load_data(data_file)
    engine.set_config(test_periods=3, max_search_time=120, num_workers=2)

    result = engine.run()

    if result['success']:
        print(f"\n最佳平均命中: {result['best_avg_hits']:.3f}")
        print(f"最佳最高命中: {result['best_max_hits']}")
        print(f"尝试组合数: {result['total_combos_tried']}")
        print(f"总耗时: {result['total_time']:.0f}秒")

        # 打印每期详情
        for pr in result['best_combo']['period_results']:
            print(f"  第{pr['period_num']}期: 命中{pr['total_hits']} "
                  f"(主球{pr['main_hits']}+辅助{pr['aux_hits']})")

        engine.generate_report()
    else:
        print(f"失败: {result['error']}")
