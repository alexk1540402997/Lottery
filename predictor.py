"""
彩票预测引擎 4.0
================
8种分析方法，全部参数化，确定性预测。
支持双色球(SSQ)和大乐透(DLT)。

核心改进：
- 所有可调参数外部传入，不再硬编码
- 确定性Top-K选择替代随机抽样（seed控制可复现）
- N-gram改用相似度匹配+相邻号码等价
- 同时兼顾红球和蓝球的预测
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# 泊松回归 (statsmodels更准确但重，sklearn的PoissonRegressor也可用)
try:
    from sklearn.linear_model import PoissonRegressor
    HAS_POISSON = True
except ImportError:
    HAS_POISSON = False


# ============================================================================
#  默认参数配置
# ============================================================================

DEFAULT_PARAMS = {
    # 方法1: 统计概率分析
    'statistical': {
        'freq_weight': 0.60,        # 频率权重
        'missing_weight': 0.40,     # 遗漏值权重
        'hot_boost': 0.15,          # 热号额外加成
        'cold_penalty': 0.10,       # 冷号额外惩罚
    },
    # 方法2: 时间序列分析
    'timeseries': {
        'window_ratios': [0.20, 0.50, 1.00],  # 短/中/长期窗口比例
        'window_weights': [0.50, 0.30, 0.20],  # 对应窗口权重
        'trend_bonus': 0.30,        # 升温趋势加成
        'sum_tolerance': 8,         # 和值容忍范围
        'sum_iterations': 300,      # 和值约束搜索次数
    },
    # 方法3: 模式识别分析
    'pattern': {
        'zone_weight': 0.20,        # 区间分布权重
        'prime_weight': 0.15,       # 质数偏好权重
        'digit_weight': 0.20,       # 尾数频率权重
        'freq_weight': 0.45,        # 基础频率权重(归一化)
        'consecutive_threshold': 0.35,  # 触发连号概率
        'zone_boundaries_ssq': [11, 22],     # SSQ三区分界
        'zone_boundaries_dlt': [12, 24],     # DLT三区分界
    },
    # 方法4: LightGBM (替代RF)
    'ml': {
        'n_estimators': 50,
        'max_depth': 6,
        'num_leaves': 15,
        'min_child_samples': 20,
        'learning_rate': 0.1,
        'lookback_windows': [5, 10, 20],
        'train_ratio': 0.80,
        'random_state': 42,
    },
    # 方法5: 马尔可夫分析
    'markov': {
        'state_percentiles': [0.33, 0.67],  # 热/温/冷分界
        'transition_weight': 0.70,          # 转移概率权重
        'base_freq_weight': 0.30,           # 基础频率权重
    },
    # 方法6: 蒙特卡罗模拟
    'montecarlo': {
        'num_simulations': 3000,
        'sum_sigma_range': 2.0,     # 和值容差(σ倍数)
        'in_range_bonus': 1.5,      # 正常和值计数加成
        'out_range_penalty': 0.5,   # 异常和值计数折扣
    },
    # 方法7: 聚类分析
    'clustering': {
        'n_clusters_min': 2,
        'n_clusters_max': 5,
        'n_init': 10,
        'features_ssq': ['sum_val', 'span', 'odd_count', 'small_count'],
        'features_dlt': ['sum_val', 'span'],
    },
    # 方法8: N-gram序列匹配
    'ngram': {
        'ngram_size': 2,
        'similarity_threshold': 0.30,   # 相似度阈值
        'adjacent_weight': 0.75,        # 相邻号码权重
        'top_k_similar': 10,            # 取最相似的K期
    },
    # 方法9: XGBoost
    'xgboost': {
        'n_estimators': 40,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'lookback_windows': [5, 10, 20],
        'train_ratio': 0.80,
        'random_state': 42,
    },
    # 方法10: 贝叶斯推断
    'bayesian': {
        'prior_strength': 2.0,      # 先验强度 (Beta先验的伪计数)
        'freq_weight': 0.55,        # 后验频率权重
        'recent_weight': 0.25,      # 近期表现权重
        'missing_weight': 0.20,     # 遗漏值权重
        'recent_window': 30,        # 近期窗口大小
    },
    # 方法11: 卡尔曼滤波
    'kalman': {
        'process_noise': 0.01,      # 过程噪声（状态变化速度）
        'measurement_noise': 0.1,   # 测量噪声
        'initial_uncertainty': 1.0, # 初始不确定性
        'trend_weight': 0.40,       # 趋势得分权重
        'freq_weight': 0.60,        # 稳态频率权重
    },
    # 方法12: 泊松回归
    'poisson': {
        'lookback_windows': [5, 10, 30],
        'alpha': 0.5,               # L2正则化强度
        'train_ratio': 0.80,
        'freq_weight': 0.50,
        'poisson_weight': 0.50,
    },
    # 方法13: 共生矩阵分析
    'cooccurrence': {
        'cooccur_threshold': 0.20,   # 共生频率阈值
        'mutual_weight': 0.55,       # 互信息权重
        'freq_weight': 0.45,         # 基础频率权重
        'window_size': 100,          # 分析窗口
    },
    # 综合推荐
    'comprehensive': {
        'method_weights': {
            'method_1': 1.0, 'method_2': 1.0,
            'method_3': 1.0, 'method_4': 1.0,
            'method_5': 1.0, 'method_6': 1.0,
            'method_7': 1.0, 'method_8': 1.0,
            'method_9': 1.0, 'method_10': 1.0,
            'method_11': 1.0, 'method_12': 1.0,
            'method_13': 1.0,
        },
    },
}


# ============================================================================
#  工具函数
# ============================================================================

def _deterministic_top_k(candidates: List[int], weights: List[float], k: int,
                          diversity_gap: int = 1, seed: int = 0) -> List[int]:
    """
    确定性的Top-K选择算法。
    按权重排序，依次选取，维护号码间最小间距（diversity_gap）。
    如果权重相同，按号码本身排序保证确定性。

    参数:
        candidates: 候选号码列表
        weights: 对应权重
        k: 选择数量
        diversity_gap: 选中号码间的最小间隔（1=允许相邻，2=至少间隔1个号码）
        seed: 随机种子（用于权重微调打破平局，保证同输入同输出）
    """
    if len(candidates) <= k:
        return sorted(candidates)

    # 创建(号码, 权重)对，按权重降序、号码升序排序（确定性）
    pairs = list(zip(candidates, weights))
    # 先用权重降序，再用号码升序打破平局
    pairs.sort(key=lambda x: (-x[1], x[0]))

    # 用seed对权重做微小扰动（1e-6级别），确保打破所有平局
    rng = np.random.RandomState(seed)
    tiny_noise = rng.rand(len(pairs)) * 1e-6
    pairs = [(num, w + tiny_noise[i]) for i, (num, w) in enumerate(pairs)]
    pairs.sort(key=lambda x: -x[1])

    selected = []
    selected_set = set()

    for num, _ in pairs:
        if num in selected_set:
            continue
        # 检查间距约束
        too_close = False
        for s in selected:
            if abs(num - s) < diversity_gap:
                too_close = True
                break
        if too_close:
            continue
        selected.append(num)
        selected_set.add(num)
        if len(selected) >= k:
            break

    # 如果不够，放宽间距约束再选
    if len(selected) < k:
        for num, _ in pairs:
            if num not in selected_set:
                selected.append(num)
                selected_set.add(num)
                if len(selected) >= k:
                    break

    return sorted(selected[:k])


# 频率计算缓存（避免同一数据重复计算）
_freq_cache = {}
_freq_cache_max_size = 8  # 最多缓存8个不同数据集的结果


def _compute_frequencies(data: pd.DataFrame, main_cols: List[str],
                          aux_cols: List[str],
                          main_range: Tuple[int, int],
                          aux_range: Tuple[int, int]
                          ) -> Tuple[Dict, Dict, Dict, Dict]:
    """计算频率和遗漏值（带缓存，避免重复计算）"""
    # 用数据的内存地址做缓存键
    cache_key = id(data)
    if cache_key in _freq_cache:
        return _freq_cache[cache_key]

    main_min, main_max = main_range
    aux_min, aux_max = aux_range
    n = len(data)

    # 转换为numpy数组加速（向量化）
    main_array = data[list(main_cols)].values.astype(int)
    aux_array = data[list(aux_cols)].values.astype(int)

    main_freq = defaultdict(int)
    aux_freq = defaultdict(int)

    for row in main_array:
        for num in row:
            if main_min <= num <= main_max:
                main_freq[int(num)] += 1
    for row in aux_array:
        for num in row:
            if aux_min <= num <= aux_max:
                aux_freq[int(num)] += 1

    # 遗漏值计算（向量化：为每个号码创建出现矩阵）
    main_missing = {}
    for num in range(main_min, main_max + 1):
        appeared = np.any(main_array == num, axis=1)
        if np.any(appeared):
            main_missing[num] = int(np.argmax(appeared))
        else:
            main_missing[num] = n

    aux_missing = {}
    for num in range(aux_min, aux_max + 1):
        appeared = np.any(aux_array == num, axis=1)
        if np.any(appeared):
            aux_missing[num] = int(np.argmax(appeared))
        else:
            aux_missing[num] = n

    result = (dict(main_freq), dict(aux_freq), main_missing, aux_missing)

    # 缓存管理（LRU简单实现：超限时清空）
    if len(_freq_cache) >= _freq_cache_max_size:
        _freq_cache.clear()
    _freq_cache[cache_key] = result

    return result


# ============================================================================
#  主预测器类
# ============================================================================

class LotteryPredictor:
    """彩票预测器 4.0 - 参数化的8种分析方法"""

    def __init__(self, lottery_type: str = 'ssq'):
        """
        参数:
            lottery_type: 'ssq' (双色球) 或 'dlt' (大乐透)
        """
        self.lottery_type = lottery_type.lower()
        if self.lottery_type not in ('ssq', 'dlt'):
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

        # 号码范围
        if self.lottery_type == 'ssq':
            self.main_name = 'red'
            self.aux_name = 'blue'
            self.main_count = 6
            self.aux_count = 1
            self.main_range = (1, 33)
            self.aux_range = (1, 16)
            self.main_cols = [f'red_{i}' for i in range(1, 7)]
            self.aux_cols = ['blue']
        else:
            self.main_name = 'front'
            self.aux_name = 'back'
            self.main_count = 5
            self.aux_count = 2
            self.main_range = (1, 35)
            self.aux_range = (1, 12)
            self.main_cols = [f'front_{i}' for i in range(1, 6)]
            self.aux_cols = [f'back_{i}' for i in range(1, 3)]

        # 质数集合
        if self.lottery_type == 'ssq':
            self.primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
        else:
            self.primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    # ========================================================================
    #  方法1: 统计概率分析
    # ========================================================================

    def predict_statistical(self, data: pd.DataFrame,
                            params: Optional[Dict] = None,
                            seed: int = 0) -> Dict[str, Any]:
        """
        统计概率分析。

        参数:
            data: 历史数据DataFrame（倒序，最新在index=0）
            params: 参数字典，见DEFAULT_PARAMS['statistical']
            seed: 确定性种子

        返回:
            {method, description, predictions, statistics}
        """
        p = {**DEFAULT_PARAMS['statistical'], **(params or {})}
        main_freq, aux_freq, main_missing, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range
        )

        total = len(data)
        main_min, main_max = self.main_range
        aux_min, aux_max = self.aux_range

        # 计算主球综合评分
        main_scores = {}
        for num in range(main_min, main_max + 1):
            freq_score = main_freq.get(num, 0) / max(total, 1)
            missing_penalty = main_missing.get(num, total) / max(total, 1)
            score = freq_score * p['freq_weight'] + (1 - missing_penalty) * p['missing_weight']
            # 热号加成
            if main_freq.get(num, 0) >= np.median(list(main_freq.values())) if main_freq else 0:
                score += p['hot_boost']
            # 冷号惩罚
            if main_missing.get(num, 0) >= np.median(list(main_missing.values())) if main_missing else 0:
                score -= p['cold_penalty']
            main_scores[num] = max(0.0001, score)

        # 计算辅助球评分
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            freq_score = aux_freq.get(num, 0) / max(total, 1)
            missing_penalty = aux_missing.get(num, total) / max(total, 1)
            score = freq_score * p['freq_weight'] + (1 - missing_penalty) * p['missing_weight']
            aux_scores[num] = max(0.0001, score)

        # 和值/跨度统计
        sums = data[self.main_cols].sum(axis=1)
        spans = data[self.main_cols].max(axis=1) - data[self.main_cols].min(axis=1)

        # 确定性选号
        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(candidates, weights, self.main_count,
                                               diversity_gap=1, seed=seed)

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(aux_candidates, aux_weights, self.aux_count,
                                              diversity_gap=1, seed=seed + 1000)

        return {
            'method': '统计概率分析',
            'description': '基于频率、遗漏值、和值、跨度的综合统计分析',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'avg_sum': round(sums.mean(), 2),
                'avg_span': round(spans.mean(), 2),
                'total_records': total,
                'hot_numbers': ', '.join(f'{n:02d}' for n in sorted(main_freq,
                                         key=main_freq.get, reverse=True)[:self.main_count]),
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法2: 时间序列分析
    # ========================================================================

    def predict_timeseries(self, data: pd.DataFrame,
                           params: Optional[Dict] = None,
                           seed: int = 0) -> Dict[str, Any]:
        """
        时间序列分析：多窗口趋势 + 和值约束。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['timeseries']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['timeseries'], **(params or {})}
        total = len(data)
        n = total

        if n < 20:
            return {'method': '时间序列分析', 'description': '数据不足(需≥20期)',
                    'error': '数据不足'}

        # 计算主球和值移动平均
        sums = data[self.main_cols].sum(axis=1)
        window_size = min(10, max(3, n // 8))
        ma_sums = sums.rolling(window=window_size, min_periods=1).mean()
        predicted_sum = int(ma_sums.iloc[0])  # 最新值
        predicted_sum = max(self.main_range[0] * self.main_count,
                            min(self.main_range[1] * self.main_count, predicted_sum))

        # 多窗口分析
        windows = []
        for ratio in p['window_ratios']:
            win_size = max(1, int(n * ratio))
            windows.append(data.iloc[:win_size])

        # 对每个主球号码计算趋势得分
        main_min, main_max = self.main_range
        main_scores = {}
        for num in range(main_min, main_max + 1):
            freq_list = []
            for win in windows:
                count = sum(1 for _, row in win.iterrows()
                           for col in self.main_cols if int(row[col]) == num)
                freq_list.append(count / max(len(win), 1))

            # 加权趋势得分
            score = sum(f * w for f, w in zip(freq_list, p['window_weights']))

            # 趋势方向加成
            if len(freq_list) >= 2:
                trend = freq_list[0] - freq_list[-1]  # 正=升温
                if trend > 0:
                    score += trend * p['trend_bonus']

            main_scores[num] = max(0.0001, score)

        # 辅助球趋势
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            freq_list = []
            for win in windows:
                count = sum(1 for _, row in win.iterrows()
                           for col in self.aux_cols if int(row[col]) == num)
                freq_list.append(count / max(len(win), 1))

            score = sum(f * w for f, w in zip(freq_list, p['window_weights']))
            if len(freq_list) >= 2:
                trend = freq_list[0] - freq_list[-1]
                if trend > 0:
                    score += trend * p['trend_bonus']
            aux_scores[num] = max(0.0001, score)

        # 和值约束搜索（确定性）
        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]

        # 生成多组候选，选和值最接近预测值的
        rng = np.random.RandomState(seed)
        best_combo = None
        best_diff = float('inf')

        for i in range(p['sum_iterations']):
            # 用seed+i确保确定性
            local_seed = seed + i
            combo = _deterministic_top_k(
                candidates,
                [w + rng.random() * 1e-6 for w in weights],  # 微小扰动
                self.main_count, diversity_gap=1, seed=local_seed
            )
            diff = abs(sum(combo) - predicted_sum)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo
            if diff <= p['sum_tolerance']:
                break

        predicted_main = best_combo if best_combo else sorted(
            candidates[:self.main_count])

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 2000)

        return {
            'method': '时间序列分析',
            'description': '多窗口趋势分析+和值约束的确定性预测',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'predicted_sum': predicted_sum,
                'avg_sum': round(sums.mean(), 2),
                'window_size': window_size,
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法3: 模式识别分析
    # ========================================================================

    def predict_pattern(self, data: pd.DataFrame,
                        params: Optional[Dict] = None,
                        seed: int = 0) -> Dict[str, Any]:
        """
        模式识别分析：连号、区间分布、质合比、AC值、尾数分布。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['pattern']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['pattern'], **(params or {})}
        total = len(data)

        if total < 30:
            return {'method': '模式识别分析', 'description': '数据不足(需≥30期)',
                    'error': '数据不足'}

        # 获取区间边界
        if self.lottery_type == 'ssq':
            z1_end, z2_end = p['zone_boundaries_ssq']
        else:
            z1_end, z2_end = p['zone_boundaries_dlt']

        main_min, main_max = self.main_range

        # 1. 连号统计
        consecutive_2 = 0
        consecutive_3 = 0
        for _, row in data.iterrows():
            nums = sorted([int(row[col]) for col in self.main_cols])
            has_2 = False
            for i in range(len(nums) - 1):
                if nums[i+1] - nums[i] == 1:
                    has_2 = True
                    if i < len(nums) - 2 and nums[i+2] - nums[i+1] == 1:
                        consecutive_3 += 1
                        break
            if has_2:
                consecutive_2 += 1

        prob_2 = consecutive_2 / total
        prob_3 = consecutive_3 / total

        # 2. 区间分布
        zone_counts = {1: [], 2: [], 3: []}
        for _, row in data.iterrows():
            nums = [int(row[col]) for col in self.main_cols]
            zone_counts[1].append(sum(1 for n in nums if n <= z1_end))
            zone_counts[2].append(sum(1 for n in nums if z1_end < n <= z2_end))
            zone_counts[3].append(sum(1 for n in nums if n > z2_end))

        avg_zone = {z: np.mean(zone_counts[z]) for z in range(1, 4)}

        # 3. 质合比
        prime_counts = []
        for _, row in data.iterrows():
            nums = [int(row[col]) for col in self.main_cols]
            prime_counts.append(sum(1 for n in nums if n in self.primes))
        avg_primes = np.mean(prime_counts)

        # 4. AC值
        def calc_ac(nums):
            diffs = set()
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    diffs.add(abs(nums[j] - nums[i]))
            return len(diffs) - (len(nums) - 1)

        ac_values = [calc_ac(sorted([int(row[c]) for c in self.main_cols]))
                     for _, row in data.iterrows()]
        avg_ac = np.mean(ac_values)

        # 5. 尾数分布
        last_digit_counts = defaultdict(int)
        for _, row in data.iterrows():
            for col in self.main_cols:
                last_digit = int(row[col]) % 10
                last_digit_counts[last_digit] += 1

        # 6. 基础频率
        main_freq, aux_freq, main_missing, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        # 7. 综合评分
        # 归一化各维度权重
        total_w = p['zone_weight'] + p['prime_weight'] + p['digit_weight'] + p['freq_weight']
        zw = p['zone_weight'] / total_w
        pw = p['prime_weight'] / total_w
        dw = p['digit_weight'] / total_w
        fw = p['freq_weight'] / total_w

        main_scores = {}
        for num in range(main_min, main_max + 1):
            # 区间得分
            if num <= z1_end:
                zone_sc = avg_zone[1] / self.main_count
            elif num <= z2_end:
                zone_sc = avg_zone[2] / self.main_count
            else:
                zone_sc = avg_zone[3] / self.main_count

            # 质数得分
            prime_sc = avg_primes / self.main_count if num in self.primes else 0

            # 尾数得分
            digit_sc = last_digit_counts.get(num % 10, 0) / (total * self.main_count)

            # 频率得分
            freq_sc = main_freq.get(num, 0) / max(total, 1)

            main_scores[num] = (zone_sc * zw + prime_sc * pw +
                                digit_sc * dw + freq_sc * fw)
            main_scores[num] = max(0.0001, main_scores[num])

        # 辅助球评分
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4

        # 确定性选号
        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        # 连号调整（确定性）
        if prob_2 > p['consecutive_threshold'] and len(predicted_main) >= 2:
            rng = np.random.RandomState(seed + 5000)
            if rng.random() < prob_2:
                for i in range(len(predicted_main) - 1):
                    if predicted_main[i+1] - predicted_main[i] == 2:
                        mid = (predicted_main[i] + predicted_main[i+1]) // 2
                        if mid not in predicted_main:
                            predicted_main[i+1] = mid
                            predicted_main.sort()
                            break

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 6000)

        return {
            'method': '模式识别分析',
            'description': '连号/区间/质合比/AC值/尾数分布的模式匹配分析',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'patterns': {
                '连号概率(≥2组)': f'{prob_2:.1%}',
                '连号概率(≥3组)': f'{prob_3:.1%}',
                '一区均值': f'{avg_zone[1]:.1f}个',
                '二区均值': f'{avg_zone[2]:.1f}个',
                '三区均值': f'{avg_zone[3]:.1f}个',
                '平均质数个数': f'{avg_primes:.1f}',
                '平均AC值': f'{avg_ac:.1f}',
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法4: LightGBM 梯度提升 (替代RF)
    # ========================================================================

    def predict_ml(self, data: pd.DataFrame,
                   params: Optional[Dict] = None,
                   seed: int = 0) -> Dict[str, Any]:
        """
        LightGBM梯度提升预测（RF回退）。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['ml']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['ml'], **(params or {})}
        total = len(data)

        if not HAS_SKLEARN and not HAS_LIGHTGBM:
            return {'method': '机器学习分析',
                    'description': 'scikit-learn和LightGBM均未安装',
                    'error': 'no_ml_library'}
        if total < 50:
            return {'method': '机器学习分析',
                    'description': '数据不足(需≥50期)',
                    'error': '数据不足'}

        main_min, main_max = self.main_range

        # 构建特征矩阵（共享代码）
        X, _ = self._build_ml_features(data, p, total, main_min, main_max)

        # 训练/预测划分
        split_idx = max(30, int(total * (1 - p['train_ratio'])))
        X_train = X[split_idx:]
        X_pred = X[:1]

        # 算法选择：LightGBM优先
        use_lgb = HAS_LIGHTGBM
        model_label = 'LightGBM' if use_lgb else 'RandomForest'

        predicted_main = []
        for pos in range(len(self.main_cols)):
            col = self.main_cols[pos]
            y = np.array([int(data.iloc[i][col]) for i in range(total)], dtype=int)
            valid_mask = (y >= main_min) & (y <= main_max)

            if valid_mask.sum() < 20:
                main_freq, _, _, _ = _compute_frequencies(
                    data, self.main_cols, self.aux_cols,
                    self.main_range, self.aux_range)
                hot = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)
                for num, _ in hot:
                    if num not in predicted_main:
                        predicted_main.append(num)
                        break
                continue

            y_train = y[split_idx:][valid_mask[split_idx:]]
            X_train_f = X_train[valid_mask[split_idx:]]

            try:
                if use_lgb:
                    model = LGBMClassifier(
                        n_estimators=min(p.get('n_estimators', 50), 100),
                        max_depth=p.get('max_depth', 6),
                        num_leaves=p.get('num_leaves', 15),
                        min_child_samples=p.get('min_child_samples', 20),
                        learning_rate=p.get('learning_rate', 0.1),
                        random_state=p['random_state'] + pos,
                        n_jobs=1, verbose=-1,
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=min(p.get('n_estimators', 50), 100),
                        max_depth=p.get('max_depth', 6),
                        min_samples_split=p.get('min_child_samples', 10),
                        random_state=p['random_state'] + pos,
                        n_jobs=1,
                    )
                model.fit(X_train_f, y_train)

                if X_pred.shape[0] > 0:
                    proba = model.predict_proba(X_pred)[0]
                    sorted_idx = np.argsort(proba)[::-1]
                    for idx in sorted_idx:
                        pred_num = model.classes_[idx]
                        if (main_min <= pred_num <= main_max and
                                pred_num not in predicted_main):
                            predicted_main.append(int(pred_num))
                            break
                    else:
                        for num in range(main_min, main_max + 1):
                            if num not in predicted_main:
                                predicted_main.append(num)
                                break
            except Exception:
                main_freq, _, _, _ = _compute_frequencies(
                    data, self.main_cols, self.aux_cols,
                    self.main_range, self.aux_range)
                hot = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)
                for num, _ in hot:
                    if num not in predicted_main:
                        predicted_main.append(num)
                        break

        # 确保完整
        predicted_main = list(dict.fromkeys(predicted_main))
        for num in range(main_min, main_max + 1):
            if len(predicted_main) >= self.main_count:
                break
            if num not in predicted_main:
                predicted_main.append(num)
        predicted_main = sorted(predicted_main[:self.main_count])

        # 辅助球：频率+遗漏法
        _, aux_freq, _, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 7000)

        return {
            'method': '机器学习分析',
            'description': f'{model_label}({p.get("n_estimators",50)}树,深度{p.get("max_depth",6)})预测',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'algorithm': model_label,
                'n_estimators': p.get('n_estimators', 50),
                'max_depth': p.get('max_depth', 6),
                'training_samples': len(X_train),
            },
            'params': p,
            'seed': seed,
        }

    def _build_ml_features(self, data: pd.DataFrame, p: Dict,
                           total: int, main_min: int, main_max: int
                           ) -> Tuple[np.ndarray, int]:
        """构建ML特征矩阵（向量化，LightGBM/XGBoost/RF共享）"""
        # 预转换为numpy数组（关键优化：避免iterrows()）
        main_array = data[list(self.main_cols)].values.astype(int)
        n_cols = len(self.main_cols)
        n_nums = main_max - main_min + 1
        n_windows = len(p['lookback_windows'])

        # 为每个时期预计算窗口起始索引
        lookbacks = p['lookback_windows']
        max_lookback = max(lookbacks)
        max_feat_len = n_nums * n_windows + n_nums + 4

        # 预分配特征矩阵
        X = np.zeros((total, max_feat_len), dtype=np.float64)

        # 逐行构建（仍需循环，但用numpy操作代替iterrows）
        for i in range(total):
            past_start = i + 1
            if past_start >= total:
                past_start = 1
            past_len = total - past_start
            if past_len < 1:
                continue
            past_array = main_array[past_start:total]  # 该期之前的历史

            feat_idx = 0

            # 窗口频率特征（向量化：用np.bincount累加各窗口的号码出现次数）
            for lookback in lookbacks:
                win_size = min(lookback, past_len)
                if win_size > 0:
                    win_data = past_array[:win_size]
                    # 展平窗口数据并用bincount统计频率
                    counts = np.bincount(win_data.ravel(), minlength=main_max + 1)
                    for num in range(main_min, main_max + 1):
                        X[i, feat_idx] = counts[num] / win_size
                        feat_idx += 1
                else:
                    feat_idx += n_nums

            # 遗漏值特征（向量化：用argmax找每个号码首次出现的位置）
            for num in range(main_min, main_max + 1):
                appeared = np.any(past_array == num, axis=1)
                if np.any(appeared):
                    X[i, feat_idx] = np.argmax(appeared) / total
                else:
                    X[i, feat_idx] = past_len / total
                feat_idx += 1

            # 上期全局特征
            if past_len > 0:
                last_nums = past_array[0]
                X[i, feat_idx] = np.sum(last_nums) / (main_max * n_cols); feat_idx += 1
                X[i, feat_idx] = (np.max(last_nums) - np.min(last_nums)) / main_max; feat_idx += 1
                X[i, feat_idx] = np.sum(last_nums % 2) / n_cols; feat_idx += 1
                X[i, feat_idx] = np.sum(last_nums <= (main_min + main_max)//2) / n_cols; feat_idx += 1
            else:
                feat_idx += 4

        # 修剪到实际使用长度
        X = X[:, :feat_idx]
        n_features = feat_idx
        return X, n_features

    # ========================================================================
    #  方法5: 马尔可夫分析
    # ========================================================================

    def predict_markov(self, data: pd.DataFrame,
                       params: Optional[Dict] = None,
                       seed: int = 0) -> Dict[str, Any]:
        """
        马尔可夫状态转移分析。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['markov']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['markov'], **(params or {})}
        total = len(data)

        if total < 30:
            return {'method': '马尔可夫分析', 'description': '数据不足(需≥30期)',
                    'error': '数据不足'}

        main_min, main_max = self.main_range

        # 1. 频率统计
        main_freq = defaultdict(int)
        for _, row in data.iterrows():
            for col in self.main_cols:
                main_freq[int(row[col])] += 1

        # 2. 热/温/冷分类
        sorted_nums = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_nums)
        hot_threshold = sorted_nums[int(n * p['state_percentiles'][0])][1] if n >= 3 else 0
        cold_threshold = sorted_nums[int(n * p['state_percentiles'][1])][1] if n >= 3 else 0

        def get_state(num):
            f = main_freq.get(num, 0)
            if f >= hot_threshold:
                return 'hot'
            elif f >= cold_threshold:
                return 'warm'
            return 'cold'

        # 3. 状态转移矩阵
        transitions = defaultdict(lambda: {
            'appear_to_appear': 0, 'appear_to_disappear': 0,
            'disappear_to_appear': 0, 'disappear_to_disappear': 0
        })

        # 构建每期的号码集合
        period_nums = []
        for _, row in data.iterrows():
            nums = {int(row[col]) for col in self.main_cols}
            period_nums.append(nums)

        # 统计转移
        for t in range(len(period_nums) - 1):
            curr = period_nums[t]
            nxt = period_nums[t + 1]
            for num in range(main_min, main_max + 1):
                was = num in curr
                is_now = num in nxt
                if was and is_now:
                    transitions[num]['appear_to_appear'] += 1
                elif was and not is_now:
                    transitions[num]['appear_to_disappear'] += 1
                elif not was and is_now:
                    transitions[num]['disappear_to_appear'] += 1
                else:
                    transitions[num]['disappear_to_disappear'] += 1

        # 4. 计算出现概率
        latest_nums = period_nums[0] if period_nums else set()
        appear_probs = {}
        state_counts = defaultdict(int)

        for num in range(main_min, main_max + 1):
            counts = transitions[num]
            total_appear = counts['appear_to_appear'] + counts['appear_to_disappear']
            total_disappear = counts['disappear_to_appear'] + counts['disappear_to_disappear']

            if total_appear > 0:
                p_a_given_a = counts['appear_to_appear'] / total_appear
            else:
                p_a_given_a = main_freq.get(num, 0) / max(total, 1)

            if total_disappear > 0:
                p_a_given_d = counts['disappear_to_appear'] / total_disappear
            else:
                p_a_given_d = main_freq.get(num, 0) / max(total, 1)

            was_in_latest = num in latest_nums
            if was_in_latest:
                prob = (p_a_given_a * p['transition_weight'] +
                        main_freq.get(num, 0) / max(total, 1) * p['base_freq_weight'])
            else:
                prob = (p_a_given_d * p['transition_weight'] +
                        main_freq.get(num, 0) / max(total, 1) * p['base_freq_weight'])

            appear_probs[num] = max(0.0001, prob)
            state_counts[get_state(num)] += 1

        # 5. 确定性选号
        candidates = list(range(main_min, main_max + 1))
        weights = [appear_probs[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        # 辅助球
        _, aux_freq, _, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4
        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 8000)

        hot_nums = [n for n in range(main_min, main_max + 1)
                    if get_state(n) == 'hot']

        return {
            'method': '马尔可夫分析',
            'description': '基于出现/不出现状态转移概率的马尔可夫预测',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'hot_count': state_counts['hot'],
                'warm_count': state_counts['warm'],
                'cold_count': state_counts['cold'],
                'hot_numbers': ', '.join(f'{n:02d}' for n in sorted(hot_nums)[:self.main_count]),
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法6: 蒙特卡罗模拟
    # ========================================================================

    def predict_montecarlo(self, data: pd.DataFrame,
                           params: Optional[Dict] = None,
                           seed: int = 0) -> Dict[str, Any]:
        """
        蒙特卡罗模拟分析。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['montecarlo']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['montecarlo'], **(params or {})}
        total = len(data)

        if total < 20:
            return {'method': '蒙特卡罗模拟', 'description': '数据不足(需≥20期)',
                    'error': '数据不足'}

        main_freq, aux_freq, _, _ = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        # 和值统计
        sums = data[self.main_cols].sum(axis=1)
        avg_sum = sums.mean()
        std_sum = sums.std()

        main_min, main_max = self.main_range
        aux_min, aux_max = self.aux_range

        # 确定性模拟
        rng = np.random.RandomState(seed)
        main_counts = defaultdict(float)
        aux_counts = defaultdict(float)

        sum_min = avg_sum - p['sum_sigma_range'] * std_sum
        sum_max = avg_sum + p['sum_sigma_range'] * std_sum

        for i in range(p['num_simulations']):
            local_seed = seed + i * 100

            # 主球抽样
            freq_items = list(main_freq.items())
            nums_list = [x[0] for x in freq_items]
            w_list = [x[1] for x in freq_items]

            if len(nums_list) < self.main_count:
                nums_list = list(range(main_min, main_max + 1))
                w_list = [1] * len(nums_list)

            sampled = _deterministic_top_k(
                nums_list, w_list, self.main_count,
                diversity_gap=1, seed=local_seed)

            s = sum(sampled)
            if sum_min <= s <= sum_max:
                bonus = p['in_range_bonus']
            else:
                bonus = p['out_range_penalty']

            for num in sampled:
                main_counts[num] += bonus

            # 辅助球
            if self.lottery_type == 'ssq':
                aux_items = list(aux_freq.items())
                aux_nums = [x[0] for x in aux_items]
                aux_ws = [x[1] for x in aux_items]
                aux_sampled = _deterministic_top_k(
                    aux_nums, aux_ws, 1, diversity_gap=1,
                    seed=local_seed + 10000)
            else:
                aux_items = list(aux_freq.items())
                aux_nums = [x[0] for x in aux_items]
                aux_ws = [x[1] for x in aux_items]
                aux_sampled = _deterministic_top_k(
                    aux_nums, aux_ws, 2, diversity_gap=1,
                    seed=local_seed + 10000)

            for num in aux_sampled:
                aux_counts[num] += 1.0

        # 按累计计数排序取Top-K
        sorted_main = sorted(main_counts.items(), key=lambda x: x[1], reverse=True)
        predicted_main = sorted([n for n, _ in sorted_main[:self.main_count]])

        sorted_aux = sorted(aux_counts.items(), key=lambda x: x[1], reverse=True)
        predicted_aux = sorted([n for n, _ in sorted_aux[:self.aux_count]])

        return {
            'method': '蒙特卡罗模拟',
            'description': f'确定性蒙特卡罗模拟({p["num_simulations"]}次)，和值约束范围{sum_min:.0f}-{sum_max:.0f}',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'simulations': p['num_simulations'],
                'sum_range': f'{sum_min:.0f}-{sum_max:.0f}',
                'avg_sum': round(avg_sum, 2),
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法7: 聚类分析 (KMeans)
    # ========================================================================

    def predict_clustering(self, data: pd.DataFrame,
                           params: Optional[Dict] = None,
                           seed: int = 0) -> Dict[str, Any]:
        """
        KMeans聚类分析。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['clustering']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['clustering'], **(params or {})}
        total = len(data)

        if not HAS_SKLEARN:
            return {'method': '聚类分析', 'description': 'scikit-learn未安装',
                    'error': 'sklearn_not_available'}
        if total < 30:
            return {'method': '聚类分析', 'description': '数据不足(需≥30期)',
                    'error': '数据不足'}

        # 构建特征DataFrame
        df = data.copy()
        df['sum_val'] = df[self.main_cols].sum(axis=1)
        df['span'] = df[self.main_cols].max(axis=1) - df[self.main_cols].min(axis=1)
        df['odd_count'] = df[self.main_cols].map(
            lambda x: x % 2 if pd.notna(x) else 0).sum(axis=1)
        df['small_count'] = df[self.main_cols].map(
            lambda x: 1 if pd.notna(x) and x <= (
                self.main_range[0] + self.main_range[1]) // 2 else 0).sum(axis=1)

        # 选择特征
        if self.lottery_type == 'ssq':
            feature_names = p['features_ssq']
        else:
            feature_names = p['features_dlt']

        available_features = [f for f in feature_names if f in df.columns]
        if not available_features:
            available_features = ['sum_val']

        X = df[available_features].values
        X = StandardScaler().fit_transform(X)

        # 确定聚类数
        n_clusters = min(p['n_clusters_max'],
                         max(p['n_clusters_min'], total // 20))

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=p['n_init'])
        clusters = kmeans.fit_predict(X)

        # 最新一期所属聚类
        latest_cluster = clusters[0]
        cluster_data = df[clusters == latest_cluster]

        # 聚类内频率统计
        cluster_main_freq = defaultdict(int)
        cluster_aux_freq = defaultdict(int)
        for _, row in cluster_data.iterrows():
            for col in self.main_cols:
                cluster_main_freq[int(row[col])] += 1
            for col in self.aux_cols:
                cluster_aux_freq[int(row[col])] += 1

        cluster_size = len(cluster_data)
        main_min, main_max = self.main_range
        aux_min, aux_max = self.aux_range

        main_scores = {}
        for num in range(main_min, main_max + 1):
            main_scores[num] = cluster_main_freq.get(num, 0) / max(cluster_size, 1)

        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = cluster_aux_freq.get(num, 0) / max(cluster_size, 1)

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 9000)

        return {
            'method': '聚类分析',
            'description': f'KMeans({n_clusters}类)聚类，匹配最新期聚类特征',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'clusters': n_clusters,
                'matched_cluster': int(latest_cluster),
                'cluster_size': cluster_size,
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法8: N-gram序列相似度匹配
    # ========================================================================

    def predict_ngram(self, data: pd.DataFrame,
                      params: Optional[Dict] = None,
                      seed: int = 0) -> Dict[str, Any]:
        """
        N-gram相似度匹配分析（4.0改进版）。

        改进：
        1. 不再要求完全匹配，使用Jaccard相似度
        2. 相邻号码等价：如果历史有2，3和4也视为匹配（相似度加成）
        3. 同时考虑红球和蓝球的匹配

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['ngram']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['ngram'], **(params or {})}
        total = len(data)

        if total < 20:
            return {'method': 'N-gram分析', 'description': '数据不足(需≥20期)',
                    'error': '数据不足'}

        # 构建每期的（主球集合, 辅助球集合）
        period_main = []
        period_aux = []
        for _, row in data.iterrows():
            main_set = {int(row[col]) for col in self.main_cols}
            aux_set = {int(row[col]) for col in self.aux_cols}
            period_main.append(main_set)
            period_aux.append(aux_set)

        # 最新一期
        latest_main = period_main[0]
        latest_aux = period_aux[0]

        # 计算每期与最新期的相似度
        def calc_similarity(set_a, set_b, adjacent_weight):
            """计算两个号码集合的相似度（含相邻号码加成）"""
            if not set_a or not set_b:
                return 0.0

            # 扩展集合：把相邻号码也加入
            expanded_a = set(set_a)
            expanded_b = set(set_b)
            for n in set_a:
                expanded_a.add(n - 1)
                expanded_a.add(n + 1)
            for n in set_b:
                expanded_b.add(n - 1)
                expanded_b.add(n + 1)

            # Jaccard相似度（精确匹配）
            intersection_exact = len(set_a & set_b)
            union_exact = len(set_a | set_b)
            exact_sim = intersection_exact / max(union_exact, 1)

            # 扩展相似度（含相邻）
            intersection_exp = len(expanded_a & expanded_b)
            union_exp = len(expanded_a | expanded_b)
            expanded_sim = intersection_exp / max(union_exp, 1)

            # 综合相似度
            return exact_sim * (1 - adjacent_weight) + expanded_sim * adjacent_weight

        # 计算各期相似度
        similarities = []
        for i in range(1, total):  # 从第1期开始（跳过最新期自身）
            main_sim = calc_similarity(latest_main, period_main[i],
                                       p['adjacent_weight'])
            aux_sim = calc_similarity(latest_aux, period_aux[i],
                                      p['adjacent_weight'])
            # 综合相似度：主球权重0.75，辅助球0.25
            total_sim = main_sim * 0.75 + aux_sim * 0.25
            similarities.append((i, total_sim, main_sim, aux_sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 取Top-K相似期
        top_matches = similarities[:p['top_k_similar']]

        # 如果没有任何匹配超过阈值，使用全部数据统计
        valid_matches = [(i, s) for i, s, _, _ in top_matches
                        if s >= p['similarity_threshold']]

        if not valid_matches:
            # 降级到全部数据的频率法
            main_freq, aux_freq, _, _ = _compute_frequencies(
                data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
            main_min, main_max = self.main_range
            aux_min, aux_max = self.aux_range

            candidates = list(range(main_min, main_max + 1))
            weights = [main_freq.get(n, 0) for n in candidates]
            predicted_main = _deterministic_top_k(
                candidates, weights, self.main_count, diversity_gap=1, seed=seed)

            aux_candidates = list(range(aux_min, aux_max + 1))
            aux_weights = [aux_freq.get(n, 0) for n in aux_candidates]
            predicted_aux = _deterministic_top_k(
                aux_candidates, aux_weights, self.aux_count,
                diversity_gap=1, seed=seed)

            return {
                'method': 'N-gram分析',
                'description': f'相似度匹配（{len(valid_matches)}个匹配>阈值{p["similarity_threshold"]}），降级为频率法',
                'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
                'statistics': {'matched_periods': 0, 'total_records': total},
                'params': p, 'seed': seed,
            }

        # 从匹配期的下一期统计号码频率（加权：相似度越高权重越大）
        main_min, main_max = self.main_range
        aux_min, aux_max = self.aux_range

        weighted_main_freq = defaultdict(float)
        weighted_aux_freq = defaultdict(float)

        for match_idx, sim_score in valid_matches:
            # 匹配期的下一期
            next_idx = match_idx - 1
            if next_idx < 0:
                continue

            weight = sim_score  # 相似度即权重
            next_main = period_main[next_idx]
            next_aux = period_aux[next_idx]

            for num in next_main:
                weighted_main_freq[num] += weight
            for num in next_aux:
                weighted_aux_freq[num] += weight

        # 如果没有有效数据，降级
        if not weighted_main_freq:
            main_freq, aux_freq, _, _ = _compute_frequencies(
                data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
            for num, cnt in main_freq.items():
                weighted_main_freq[num] = cnt
            for num, cnt in aux_freq.items():
                weighted_aux_freq[num] = cnt

        # 确定性选号
        candidates = list(range(main_min, main_max + 1))
        weights = [weighted_main_freq.get(n, 0) for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [weighted_aux_freq.get(n, 0) for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 10000)

        return {
            'method': 'N-gram分析',
            'description': f'相似度匹配({len(valid_matches)}个匹配期)，含相邻号码等价',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'matched_periods': len(valid_matches),
                'avg_similarity': f'{np.mean([s for _, s in valid_matches]):.3f}',
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法9: XGBoost 集成学习
    # ========================================================================

    def predict_xgboost(self, data: pd.DataFrame,
                        params: Optional[Dict] = None,
                        seed: int = 0) -> Dict[str, Any]:
        """
        XGBoost集成学习预测（与LightGBM互补）。

        参数:
            data: 历史数据（倒序）
            params: 见DEFAULT_PARAMS['xgboost']
            seed: 确定性种子
        """
        p = {**DEFAULT_PARAMS['xgboost'], **(params or {})}
        total = len(data)

        if not HAS_XGBOOST:
            return {'method': 'XGBoost分析',
                    'description': 'XGBoost未安装(pip install xgboost)',
                    'error': 'xgboost_not_available'}
        if total < 50:
            return {'method': 'XGBoost分析',
                    'description': '数据不足(需≥50期)',
                    'error': '数据不足'}

        main_min, main_max = self.main_range

        # 复用ML特征构建
        X, _ = self._build_ml_features(data, p, total, main_min, main_max)

        split_idx = max(30, int(total * (1 - p['train_ratio'])))
        X_train = X[split_idx:]
        X_pred = X[:1]

        predicted_main = []
        for pos in range(len(self.main_cols)):
            col = self.main_cols[pos]
            y = np.array([int(data.iloc[i][col]) for i in range(total)], dtype=int)
            valid_mask = (y >= main_min) & (y <= main_max)

            if valid_mask.sum() < 20:
                main_freq, _, _, _ = _compute_frequencies(
                    data, self.main_cols, self.aux_cols,
                    self.main_range, self.aux_range)
                hot = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)
                for num, _ in hot:
                    if num not in predicted_main:
                        predicted_main.append(num)
                        break
                continue

            y_train = y[split_idx:][valid_mask[split_idx:]]
            X_train_f = X_train[valid_mask[split_idx:]]
            try:
                model = XGBClassifier(
                    n_estimators=p.get('n_estimators', 40),
                    max_depth=p.get('max_depth', 5),
                    learning_rate=p.get('learning_rate', 0.1),
                    subsample=p.get('subsample', 0.8),
                    colsample_bytree=p.get('colsample_bytree', 0.8),
                    reg_alpha=p.get('reg_alpha', 1.0),
                    reg_lambda=p.get('reg_lambda', 1.0),
                    random_state=p['random_state'] + pos,
                    n_jobs=1, verbosity=0,
                )
                model.fit(X_train_f, y_train)

                if X_pred.shape[0] > 0:
                    proba = model.predict_proba(X_pred)[0]
                    sorted_idx = np.argsort(proba)[::-1]
                    for idx in sorted_idx:
                        pred_num = model.classes_[idx]
                        if (main_min <= pred_num <= main_max and
                                pred_num not in predicted_main):
                            predicted_main.append(int(pred_num))
                            break
            except Exception:
                main_freq, _, _, _ = _compute_frequencies(
                    data, self.main_cols, self.aux_cols,
                    self.main_range, self.aux_range)
                hot = sorted(main_freq.items(), key=lambda x: x[1], reverse=True)
                for num, _ in hot:
                    if num not in predicted_main:
                        predicted_main.append(num)
                        break

        predicted_main = list(dict.fromkeys(predicted_main))
        for num in range(main_min, main_max + 1):
            if len(predicted_main) >= self.main_count:
                break
            if num not in predicted_main:
                predicted_main.append(num)
        predicted_main = sorted(predicted_main[:self.main_count])

        _, aux_freq, _, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 8000)

        return {
            'method': 'XGBoost分析',
            'description': f'XGBoost({p.get("n_estimators",40)}树,深度{p.get("max_depth",5)})预测',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'algorithm': 'XGBoost',
                'n_estimators': p.get('n_estimators', 40),
                'max_depth': p.get('max_depth', 5),
                'training_samples': len(X_train),
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法10: 贝叶斯推断
    # ========================================================================

    def predict_bayesian(self, data: pd.DataFrame,
                         params: Optional[Dict] = None,
                         seed: int = 0) -> Dict[str, Any]:
        """
        贝叶斯推断：基于Beta-Binomial共轭分布的号码概率估计。

        核心思想：每个号码的出现频率服从 Beta-Binomial 共轭分布。
        先验 Beta(α, β) + 观测(出现k次,缺失n-k次) → 后验 Beta(α+k, β+n-k)
        后验均值作为该号码的合理出现概率。

        速度极快（~0.01s），无需训练，解析解。
        """
        p = {**DEFAULT_PARAMS['bayesian'], **(params or {})}
        total = len(data)

        main_min, main_max = self.main_range

        # 全量频率
        main_freq, aux_freq, main_missing, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        # Beta-Binomial 先验
        alpha_prior = p['prior_strength']
        beta_prior = p['prior_strength']

        # 后验估计主球每个号码的后验概率
        main_scores = {}
        for num in range(main_min, main_max + 1):
            k = main_freq.get(num, 0)
            m = main_missing.get(num, total)
            n_th = total  # 名义观测次数

            # 后验 Beta(α+k, β+n-k)
            alpha_post = alpha_prior + k
            beta_post = beta_prior + (n_th - k)

            posterior_mean = alpha_post / (alpha_post + beta_post)

            # 近期表现（最近N期的频率）
            recent_n = min(p['recent_window'], total)
            recent_data = data.head(recent_n)
            recent_count = sum(1 for _, row in recent_data.iterrows()
                             for col in self.main_cols if int(row[col]) == num)
            recent_rate = recent_count / max(recent_n * len(self.main_cols), 1)

            score = (posterior_mean * p['freq_weight'] +
                    recent_rate * p['recent_weight'] +
                    (1 - m / max(total, 1)) * p['missing_weight'])
            main_scores[num] = max(0.0001, score)

        # 辅助球
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            k = aux_freq.get(num, 0)
            m = aux_missing.get(num, total)
            alpha_post = alpha_prior + k
            beta_post = beta_prior + (total - k)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            aux_scores[num] = max(0.0001, posterior_mean * 0.7 +
                                  (1 - m / max(total, 1)) * 0.3)

        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 9000)

        return {
            'method': '贝叶斯推断',
            'description': f'Beta-Binomial共轭先验(α={alpha_prior})贝叶斯概率估计',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'prior_alpha': alpha_prior,
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法11: 卡尔曼滤波
    # ========================================================================

    def predict_kalman(self, data: pd.DataFrame,
                       params: Optional[Dict] = None,
                       seed: int = 0) -> Dict[str, Any]:
        """
        卡尔曼滤波趋势追踪。

        核心思想：每个号码的"真实频率强度"是一个随时间缓慢变化的隐状态。
        每期观测到该号码出现(1)或不出现(0)，用卡尔曼滤波递推估计最新状态。

        速度极快（~0.01s），递推公式，天然适合在线更新。
        """
        p = {**DEFAULT_PARAMS['kalman'], **(params or {})}
        total = len(data)

        main_min, main_max = self.main_range
        process_noise = p['process_noise']
        measurement_noise = p['measurement_noise']
        initial_uncertainty = p['initial_uncertainty']

        # 为每个号码构建出现序列
        main_array = data[list(self.main_cols)].values.astype(int)
        aux_array = data[list(self.aux_cols)].values.astype(int)
        # 倒序 → 顺序排列（早→晚）
        main_seq = main_array[::-1]
        aux_seq = aux_array[::-1]

        # 卡尔曼滤波器（标量状态，递推）
        def kalman_filter(observations):
            """递推估计隐状态（真实出现概率）"""
            x = np.mean(observations) if len(observations) > 0 else 0.1  # 初始状态
            P = initial_uncertainty  # 初始误差协方差

            for z in observations:
                # 预测
                x_pred = x  # 状态不变（慢变假设）
                P_pred = P + process_noise

                # 更新
                K = P_pred / (P_pred + measurement_noise)  # 卡尔曼增益
                x = x_pred + K * (z - x_pred)
                P = (1 - K) * P_pred

            return x  # 最终状态估计

        # 主球卡尔曼滤波
        main_scores = {}
        freq_main, _, _, _ = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        for num in range(main_min, main_max + 1):
            # 构建出现序列（0/1）
            obs = np.any(main_seq == num, axis=1).astype(float)
            if len(obs) == 0:
                main_scores[num] = 0.01
                continue

            # 卡尔曼滤波估计当前状态
            kalman_state = kalman_filter(obs)

            # 稳态频率
            steady_freq = freq_main.get(num, 0) / max(total, 1)

            # 综合得分
            score = (kalman_state * p['trend_weight'] +
                    steady_freq * p['freq_weight'])
            main_scores[num] = max(0.0001, score)

        # 辅助球卡尔曼滤波
        aux_min, aux_max = self.aux_range
        _, freq_aux, _, _ = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            obs = np.any(aux_seq == num, axis=1).astype(float)
            if len(obs) == 0:
                aux_scores[num] = 0.01
                continue
            kalman_state = kalman_filter(obs)
            steady_freq = freq_aux.get(num, 0) / max(total, 1)
            aux_scores[num] = max(0.0001,
                kalman_state * 0.5 + steady_freq * 0.5)

        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 10000)

        return {
            'method': '卡尔曼滤波',
            'description': '递推卡尔曼滤波追踪每个号码的频率趋势',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'process_noise': process_noise,
                'total_records': total,
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  方法12: 泊松回归
    # ========================================================================

    def predict_poisson(self, data: pd.DataFrame,
                        params: Optional[Dict] = None,
                        seed: int = 0) -> Dict[str, Any]:
        """
        泊松回归：将每个号码的出现建模为计数过程。

        号码出现次数是计数数据，适合泊松分布。Poisson GLM用协变量
        （近期频率、遗漏值、和值等）预测每个号码的期望出现率。

        速度：秒级（sklearn PoissonRegressor）。
        """
        p = {**DEFAULT_PARAMS['poisson'], **(params or {})}
        total = len(data)

        if not HAS_POISSON:
            return {'method': '泊松回归',
                    'description': '泊松回归不可用(需sklearn>=1.0)',
                    'error': 'poisson_not_available'}
        if total < 30:
            return {'method': '泊松回归',
                    'description': '数据不足(需≥30期)',
                    'error': '数据不足'}

        main_min, main_max = self.main_range

        # 构建特征：每期每个号码的(近期频率, 遗漏值, 和值偏差)
        features_list = []
        targets = []
        for i in range(1, total):  # 从第1期开始（需要历史）
            past = data.iloc[i:]
            if len(past) < 5:
                continue
            curr_row = data.iloc[i - 1]
            curr_nums = [int(curr_row[c]) for c in self.main_cols]
            curr_sum = sum(curr_nums)

            past_freq, _, past_missing, _ = _compute_frequencies(
                past, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

            for num in range(main_min, main_max + 1):
                feat = [
                    past_freq.get(num, 0) / max(len(past), 1),
                    past_missing.get(num, len(past)) / max(len(past), 1),
                    abs(curr_sum - num * len(self.main_cols)) / (main_max * self.main_count),
                    num / main_max,  # 号码位置归一化
                ]
                features_list.append(feat)
                targets.append(1 if num in curr_nums else 0)

        if len(features_list) < 50:
            # 降级到频率法
            return self._fallback_frequency(data, seed, '泊松回归(降级频率法)')

        X = np.array(features_list, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)

        # 取最新一期的特征
        latest_data = data.iloc[1:] if len(data) > 1 else data
        latest_freq, _, latest_missing, _ = _compute_frequencies(
            latest_data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        latest_row = data.iloc[0]
        latest_nums = [int(latest_row[c]) for c in self.main_cols]
        latest_sum = sum(latest_nums)

        X_pred = []
        for num in range(main_min, main_max + 1):
            feat = [
                latest_freq.get(num, 0) / max(len(latest_data), 1),
                latest_missing.get(num, len(latest_data)) / max(len(latest_data), 1),
                abs(latest_sum - num * len(self.main_cols)) / (main_max * self.main_count),
                num / main_max,
            ]
            X_pred.append(feat)
        X_pred = np.array(X_pred, dtype=np.float64)

        try:
            model = PoissonRegressor(
                alpha=p['alpha'],
                max_iter=200,
            )
            model.fit(X, y)
            predicted_rates = model.predict(X_pred)
            predicted_rates = np.clip(predicted_rates, 0.001, None)
        except Exception:
            return self._fallback_frequency(data, seed, '泊松回归(拟合失败)')

        candidates = list(range(main_min, main_max + 1))
        weights = list(predicted_rates)
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        # 辅助球：频率法
        _, aux_freq, _, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4
        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 11000)

        return {
            'method': '泊松回归',
            'description': f'Poisson GLM预测号码出现期望率',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'training_samples': len(features_list),
                'alpha': p['alpha'],
            },
            'params': p,
            'seed': seed,
        }

    def _fallback_frequency(self, data: pd.DataFrame, seed: int,
                            method_name: str) -> Dict[str, Any]:
        """降级频率法（作为后备）"""
        total = len(data)
        main_freq, aux_freq, main_missing, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        main_min, main_max = self.main_range
        main_scores = {}
        for num in range(main_min, main_max + 1):
            main_scores[num] = (main_freq.get(num, 0) / max(total, 1) * 0.6 +
                               (1 - main_missing.get(num, total) / max(total, 1)) * 0.4)

        candidates = list(range(main_min, main_max + 1))
        weights = [main_scores[n] for n in candidates]
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4
        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 12000)

        return {
            'method': method_name,
            'description': '数据不足或模型失败，降级使用频率法',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {'total_records': total},
            'params': {}, 'seed': seed,
        }

    # ========================================================================
    #  方法13: 共生矩阵分析
    # ========================================================================

    def predict_cooccurrence(self, data: pd.DataFrame,
                             params: Optional[Dict] = None,
                             seed: int = 0) -> Dict[str, Any]:
        """
        共生矩阵分析：捕捉号码间的共现关系。

        统计号码对的共现频率，构建共生矩阵。如果号码A与最近热号
        高度共现，则A的得分被提升。

        速度极快（~0.02s），纯矩阵运算。
        """
        p = {**DEFAULT_PARAMS['cooccurrence'], **(params or {})}
        total = len(data)

        main_min, main_max = self.main_range
        n_nums = main_max - main_min + 1

        # 统计基础频率
        main_freq, aux_freq, main_missing, aux_missing = _compute_frequencies(
            data, self.main_cols, self.aux_cols, self.main_range, self.aux_range)

        # 构建共生矩阵（只取窗口内的数据）
        window_size = min(p['window_size'], total)
        window_data = data.head(window_size)

        # 共生矩阵：cooc[i][j] = 号码i和号码j同时出现的次数
        cooc = np.zeros((n_nums, n_nums), dtype=np.float64)
        for _, row in window_data.iterrows():
            nums = [int(row[col]) for col in self.main_cols]
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    a = nums[i] - main_min
                    b = nums[j] - main_min
                    if 0 <= a < n_nums and 0 <= b < n_nums:
                        cooc[a][b] += 1
                        cooc[b][a] += 1

        # 归一化 → 共生频率
        for i in range(n_nums):
            row_sum = cooc[i].sum()
            if row_sum > 0:
                cooc[i] /= row_sum

        # 基础频率得分
        freq_scores = np.zeros(n_nums)
        for num in range(main_min, main_max + 1):
            idx = num - main_min
            freq_scores[idx] = (main_freq.get(num, 0) / max(total, 1) * 0.7 +
                               (1 - main_missing.get(num, total) / max(total, 1)) * 0.3)

        # 用共生矩阵增强：如果与高频号码共现，得分提升
        enhanced_scores = freq_scores.copy()
        freq_threshold = np.percentile(freq_scores, 70)
        top_indices = np.where(freq_scores >= freq_threshold)[0]

        for idx in range(n_nums):
            # 与高频号码的共现加成
            if len(top_indices) > 0:
                cooc_bonus = cooc[idx][top_indices].mean()
                enhanced_scores[idx] = (freq_scores[idx] * p['freq_weight'] +
                                       cooc_bonus * p['mutual_weight'])

        candidates = list(range(main_min, main_max + 1))
        weights = list(enhanced_scores)
        predicted_main = _deterministic_top_k(
            candidates, weights, self.main_count, diversity_gap=1, seed=seed)

        # 辅助球
        aux_min, aux_max = self.aux_range
        aux_scores = {}
        for num in range(aux_min, aux_max + 1):
            aux_scores[num] = aux_freq.get(num, 0) / max(total, 1) * 0.6
            aux_scores[num] += (1 - aux_missing.get(num, total) / max(total, 1)) * 0.4
        aux_candidates = list(range(aux_min, aux_max + 1))
        aux_weights = [aux_scores[n] for n in aux_candidates]
        predicted_aux = _deterministic_top_k(
            aux_candidates, aux_weights, self.aux_count,
            diversity_gap=1, seed=seed + 13000)

        # 计算最显著的共现对
        significant_pairs = []
        for i in range(n_nums):
            for j in range(i + 1, n_nums):
                if cooc[i][j] > p['cooccur_threshold']:
                    significant_pairs.append(
                        (main_min + i, main_min + j, round(float(cooc[i][j]), 3)))

        return {
            'method': '共生矩阵分析',
            'description': f'号码共生矩阵增强(窗口{window_size}期, {len(significant_pairs)}个显著共现对)',
            'predictions': {self.main_name: predicted_main, self.aux_name: predicted_aux},
            'statistics': {
                'significant_pairs': len(significant_pairs),
                'window_size': window_size,
                'top_pair': f'{significant_pairs[0][0]}-{significant_pairs[0][1]}'
                           if significant_pairs else 'N/A',
            },
            'params': p,
            'seed': seed,
        }

    # ========================================================================
    #  综合推荐：跨方法加权投票
    # ========================================================================

    def predict_comprehensive(self, method_results: Dict[str, Dict],
                               weights: Optional[Dict[str, float]] = None
                               ) -> Dict[str, Any]:
        """
        综合8种方法的预测结果，加权投票产生最终推荐。

        参数:
            method_results: {method_key: result_dict} 各方法的预测结果
            weights: 各方法权重，默认使用DEFAULT_PARAMS中的预设值
        """
        if weights is None:
            weights = DEFAULT_PARAMS['comprehensive']['method_weights']

        all_main_votes = []
        all_aux_votes = []

        for method_key, result in method_results.items():
            if 'error' in result or 'predictions' not in result:
                continue

            w = weights.get(method_key, 1.0)
            predictions = result.get('predictions', {})

            if isinstance(predictions, dict):
                main_balls = predictions.get(self.main_name, [])
                aux_balls = predictions.get(self.aux_name, [])

                if isinstance(main_balls, list) and main_balls:
                    for num in main_balls:
                        all_main_votes.extend([num] * int(w * 10))
                if isinstance(aux_balls, list) and aux_balls:
                    for num in aux_balls:
                        all_aux_votes.extend([num] * int(w * 10))

        # 投票统计
        if all_main_votes:
            main_counter = Counter(all_main_votes)
            predicted_main = [num for num, _ in
                             main_counter.most_common(self.main_count)]
        else:
            predicted_main = sorted(
                list(range(self.main_range[0],
                          self.main_range[0] + self.main_count)))

        if all_aux_votes:
            aux_counter = Counter(all_aux_votes)
            predicted_aux = [num for num, _ in
                            aux_counter.most_common(self.aux_count)]
        else:
            predicted_aux = sorted(
                list(range(self.aux_range[0],
                          self.aux_range[0] + self.aux_count)))

        return {
            'method': '综合推荐',
            'description': '8种方法加权投票综合推荐',
            'predictions': {self.main_name: sorted(predicted_main),
                           self.aux_name: sorted(predicted_aux)},
            'weights_used': weights,
        }

    # ========================================================================
    #  统一预测入口
    # ========================================================================

    def predict_all(self, data: pd.DataFrame,
                    params: Optional[Dict[str, Dict]] = None,
                    seed: int = 0,
                    methods: Optional[List[str]] = None
                    ) -> Dict[str, Any]:
        """
        运行所有（或指定）分析方法。

        参数:
            data: 历史数据（倒序，最新在index=0）
            params: 各方法参数字典，格式：
                    {'statistical': {...}, 'timeseries': {...}, ...}
                    未指定的方法使用默认参数。
            seed: 全局确定性种子
            methods: 要运行的方法列表，默认全部13个
                    可选值: ['statistical', 'timeseries', 'pattern', 'ml',
                            'markov', 'montecarlo', 'clustering', 'ngram',
                            'xgboost', 'bayesian', 'kalman', 'poisson', 'cooccurrence']

        返回:
            {method_key: result_dict, ..., 'comprehensive': result_dict}
        """
        if methods is None:
            methods = ['statistical', 'timeseries', 'pattern', 'ml',
                      'markov', 'montecarlo', 'clustering', 'ngram',
                      'xgboost', 'bayesian', 'kalman', 'poisson', 'cooccurrence']

        all_params = {}
        for key in DEFAULT_PARAMS:
            all_params[key] = {**DEFAULT_PARAMS[key], **(params or {}).get(key, {})}

        results = {}
        method_map = {
            'statistical': (self.predict_statistical, 'method_1'),
            'timeseries': (self.predict_timeseries, 'method_2'),
            'pattern': (self.predict_pattern, 'method_3'),
            'ml': (self.predict_ml, 'method_4'),
            'markov': (self.predict_markov, 'method_5'),
            'montecarlo': (self.predict_montecarlo, 'method_6'),
            'clustering': (self.predict_clustering, 'method_7'),
            'ngram': (self.predict_ngram, 'method_8'),
            'xgboost': (self.predict_xgboost, 'method_9'),
            'bayesian': (self.predict_bayesian, 'method_10'),
            'kalman': (self.predict_kalman, 'method_11'),
            'poisson': (self.predict_poisson, 'method_12'),
            'cooccurrence': (self.predict_cooccurrence, 'method_13'),
        }

        for method_name in methods:
            if method_name in method_map:
                func, key = method_map[method_name]
                param_key = method_name
                try:
                    results[key] = func(data, all_params.get(param_key), seed)
                except Exception as e:
                    results[key] = {
                        'method': method_name,
                        'description': f'执行异常: {str(e)}',
                        'error': str(e),
                    }

        # 综合推荐
        results['comprehensive'] = self.predict_comprehensive(
            results, all_params['comprehensive']['method_weights'])

        return results

    # ========================================================================
    #  数据加载（兼容旧版Excel格式）
    # ========================================================================

    @staticmethod
    def load_data(filepath: str) -> Tuple[pd.DataFrame, str]:
        """
        加载Excel数据文件，自动识别彩票类型。

        返回:
            (data_reverse, lottery_type)
            data_reverse: 倒序DataFrame（最新在index=0）
            lottery_type: 'ssq' 或 'dlt'
        """
        df = pd.read_excel(filepath)
        columns = [str(c).strip() for c in df.columns]

        # 识别类型
        ssq_indicators = ['红球号码1', '红球1', 'red_1', '蓝球', 'blue']
        dlt_indicators = ['前区号码1', '前区1', 'front_1', '后区1', 'back_1']

        if any(ind in str(cols) for ind in ssq_indicators for cols in columns):
            lottery_type = 'ssq'
        elif any(ind in str(cols) for ind in dlt_indicators for cols in columns):
            lottery_type = 'dlt'
        else:
            raise ValueError(f"无法识别彩票类型，列名: {columns}")

        # 处理数据
        result = pd.DataFrame()

        # 期号
        for col in ['期号', '开奖期号', '期数']:
            if col in df.columns:
                result['period'] = df[col].astype(str).str.strip()
                break
        if 'period' not in result.columns:
            result['period'] = [f"{i:05d}" for i in range(1, len(df)+1)]

        # 日期
        for col in ['开奖日期', '日期']:
            if col in df.columns:
                result['draw_date'] = df[col].astype(str)
                break
        if 'draw_date' not in result.columns:
            result['draw_date'] = ""

        # 号码列
        if lottery_type == 'ssq':
            for i in range(1, 7):
                for cn in [f'红球号码{i}', f'红球{i}', f'red_{i}']:
                    if cn in df.columns:
                        result[f'red_{i}'] = pd.to_numeric(
                            df[cn], errors='coerce').fillna(0).astype(int)
                        break
                if f'red_{i}' not in result.columns:
                    result[f'red_{i}'] = 0
            for cn in ['蓝球', 'blue']:
                if cn in df.columns:
                    result['blue'] = pd.to_numeric(
                        df[cn], errors='coerce').fillna(0).astype(int)
                    break
            if 'blue' not in result.columns:
                result['blue'] = 0
        else:
            for i in range(1, 6):
                for cn in [f'前区号码{i}', f'前区{i}', f'front_{i}']:
                    if cn in df.columns:
                        result[f'front_{i}'] = pd.to_numeric(
                            df[cn], errors='coerce').fillna(0).astype(int)
                        break
                if f'front_{i}' not in result.columns:
                    result[f'front_{i}'] = 0
            for i in range(1, 3):
                for cn in [f'后区号码{i}', f'后区{i}', f'back_{i}']:
                    if cn in df.columns:
                        result[f'back_{i}'] = pd.to_numeric(
                            df[cn], errors='coerce').fillna(0).astype(int)
                        break
                if f'back_{i}' not in result.columns:
                    result[f'back_{i}'] = 0

        # 按日期升序排列
        try:
            result['_period_int'] = result['period'].astype(str).str.strip().astype(int)
            result = result.sort_values('_period_int', ascending=True)
            result = result.drop('_period_int', axis=1)
        except Exception:
            pass

        result = result.reset_index(drop=True)
        result = result.dropna()

        # 倒序（最新在前）
        data_reverse = result.iloc[::-1].reset_index(drop=True)

        print(f"数据加载成功: {len(result)}条{lottery_type}记录")
        return data_reverse, lottery_type


# ============================================================================
#  快速测试
# ============================================================================

if __name__ == "__main__":
    import os, sys
    # 查找数据文件
    base = os.path.dirname(os.path.abspath(__file__))
    for f in ['双色球.xlsx', '大乐透.xlsx']:
        path = os.path.join(base, f)
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"测试: {f}")
            print(f"{'='*60}")
            data_rev, lt = LotteryPredictor.load_data(path)
            predictor = LotteryPredictor(lt)

            # 取最近100期
            test_data = data_rev.head(100)
            print(f"分析数据: 最近{len(test_data)}期")

            results = predictor.predict_all(test_data, seed=42)
            for key, r in results.items():
                if 'error' in r:
                    print(f"  {key}: 错误 - {r['error']}")
                else:
                    pred = r['predictions']
                    print(f"  {r['method']}: "
                          f"{predictor.main_name}={pred[predictor.main_name]} "
                          f"{predictor.aux_name}={pred[predictor.aux_name]}")
            break

# 导出方法名称映射（供其他模块使用）
METHOD_NAMES_NEW = {
    'method_1': '统计概率分析',
    'method_2': '时间序列分析',
    'method_3': '模式识别分析',
    'method_4': 'LightGBM分析',
    'method_5': '马尔可夫分析',
    'method_6': '蒙特卡罗模拟',
    'method_7': '聚类分析',
    'method_8': 'N-gram分析',
    'method_9': 'XGBoost分析',
    'method_10': '贝叶斯推断',
    'method_11': '卡尔曼滤波',
    'method_12': '泊松回归',
    'method_13': '共生矩阵分析',
}
