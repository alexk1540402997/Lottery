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

# 优化器模块（4.3+）
try:
    from optimizers.bo_surrogate import BOBridge
    from optimizers.linear_weight_solver import LinearWeightSolver
    from optimizers.cmaes_sampler import CMAESBridge
    from optimizers.sa_sampler import SABridge
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False
    BOBridge = None
    LinearWeightSolver = None
    CMAESBridge = None
    SABridge = None

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
        'sum_iterations':    [20],  # 固定20，已验证300→20预测不变
    },
    'pattern': {
        'zone_weight':       [0.10, 0.15, 0.20, 0.25, 0.30],
        'prime_weight':      [0.08, 0.12, 0.15, 0.20, 0.25],
        'digit_weight':      [0.10, 0.15, 0.20, 0.25],
        'freq_weight':       [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        'consecutive_threshold': [0.25, 0.30, 0.35, 0.40, 0.45],
    },
    'ml': {
        'n_estimators':      [15, 20, 30, 50],
        'max_depth':         [4, 6, 8, 10],
        'num_leaves':        [10, 15, 20, 31],
        'learning_rate':     [0.05, 0.10, 0.15],
    },
    'markov': {
        'transition_weight': [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        'base_freq_weight':  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    },
    'montecarlo': {
        'num_simulations':   [300, 500, 800, 1200],
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

# 合并权重搜索空间（65个独立 composite_weights，范围 [-500.0, 500.0]）
WEIGHT_SEARCH_SPACE = {
    'composite_weight_range': (-500.0, 500.0),  # 每个(方法×颗粒度)组合的独立权重范围
}


# ============================================================================
#  回测引擎
# ============================================================================

class BacktestEngine:
    """回测与优化引擎 4.3"""

    def __init__(self):
        self.data_reverse = None       # 完整数据（倒序，index=0=最新）
        self.lottery_type = ''
        self.predictor = None
        self.total_periods = 0

        # 回测配置
        self.test_periods = 50         # 测试最新N期
        self.ALL_GRANULARITIES = [50, 100, 500, 1000, 0]  # 所有可用颗粒度
        self.granularities = [100, 500]  # 回测颗粒度（智能调整）
        self.gran_names = ['100期', '500期']
        self.max_search_time = 0       # 最大搜索时间（秒），0=不限制
        self.num_workers = 4           # 并行线程数
        self.max_train_periods = 500   # 限制训练数据最多500期（加速ML）

        # 优化器模式（4.3+）
        self.use_bo = False            # 是否启用贝叶斯优化
        self.bo_bridge = None          # BOBridge 实例
        self.use_cmaes = False         # 是否启用 CMA-ES
        self.cmaes_bridge = None       # CMAESBridge 实例
        self.use_sa = False            # 是否启用模拟退火
        self.sa_bridge = None          # SABridge 实例

        # 已尝试组合去重
        self.tried_combos: Dict[str, float] = {}  # hash → best_score
        self.tried_log_file = "logs/backtest_tried_combos.json"

        # 结果
        self.best_combo = None         # 最佳(参数, 权重)
        self.best_score = 0.0
        self.best_period_results = []  # 每期详细命中
        self.all_results = []          # 所有组合的评估结果

        # 最优参数持久化（供求解模式读取）
        self.best_overall = None       # 全局最优 (最高 avg_hits)
        self.best_recent = None        # 最新期最优 (period_idx=0 最高命中)

        # 智能搜索：参数表现追踪
        self.param_performance: Dict = {}  # {method: {param: {value: {count, total_score}}}}
        self.history_detail: List[Dict] = []  # 全部组合详情
        self.top_combos: List[Dict] = []      # Top-K最佳（最多保存10个）
        self.combo_counter = 0                # 累计组合序号

        # 搜索阶段控制
        self.phase = 'exploration'      # exploration → convergence
        self.phase_switch_ratio = 0.30  # 前30%时间探索
        self.pulse_interval = 20        # 每N个组合一次随机脉冲
        self.perturb_ratio = 0.20       # 收敛期扰动参数比例（默认20%）
        self._phase_switched_at = 0     # 阶段切换时的combo_counter

        # 运行时状态
        self.running = False
        self.start_time = 0.0
        self.progress_callback = None
        self.log_callback = None

        # 加载历史记录
        self._load_tried_combos()
        self._load_history_detail()

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
        else:
            # 智能颗粒度：测试期少→少颗粒度（加速），测试期多→多颗粒度
            if test_periods <= 2:
                self.granularities = [500]  # 1-2期测试只需1种颗粒度
                self.gran_names = ['500期']
            elif test_periods <= 10:
                self.granularities = [100, 500]  # 2种颗粒度
                self.gran_names = ['100期', '500期']
            else:
                self.granularities = [50, 100, 500]  # 3种颗粒度
                self.gran_names = ['50期', '100期', '500期']
        self.max_search_time = max_search_time
        self.num_workers = max(1, num_workers)
        self.max_train_periods = max_train_periods

    def set_callbacks(self, progress=None, log=None):
        """设置进度和日志回调"""
        self.progress_callback = progress
        self.log_callback = log

    def init_bo(self, mode: str = 'backtest'):
        """
        初始化贝叶斯优化器。

        参数:
          mode: 'backtest'（参数+权重联合优化）
                或 'solve'（只优化参数，权重线性求解）

        注意: 需要 OPTIMIZERS_AVAILABLE = True
        """
        if not OPTIMIZERS_AVAILABLE:
            self._log("[BO] 优化器模块不可用，回退到随机搜索")
            self.use_bo = False
            return False

        self.bo_bridge = BOBridge(
            param_search_space=PARAM_SEARCH_SPACE,
            weight_search_space=WEIGHT_SEARCH_SPACE,
            mode=mode,
        )
        self.use_bo = True
        self._log(f"[BO] 贝叶斯优化已初始化 (模式={mode}, "
                  f"维度={self.bo_bridge.bo.encoder.n_free})")
        return True

    def init_cmaes(self, population_size: int = None, initial_sigma: float = 0.3):
        """
        初始化 CMA-ES 采样器。

        参数:
          population_size: 种群大小（默认自动: 4+3*ln(dim)）
          initial_sigma: 初始步长
        """
        if not OPTIMIZERS_AVAILABLE:
            self._log("[CMA-ES] 优化器模块不可用")
            self.use_cmaes = False
            return False

        self.cmaes_bridge = CMAESBridge(
            param_search_space=PARAM_SEARCH_SPACE,
            population_size=population_size,
            initial_sigma=initial_sigma,
        )
        self.use_cmaes = True
        stats = self.cmaes_bridge.get_stats()
        self._log(f"[CMA-ES] 已初始化 (维度={stats['dim']}, "
                  f"种群={stats['population_size']})")
        return True

    def init_sa(self, n_chains: int = 4, T_max: float = 1.0,
                cooling_rate: float = 0.95):
        """
        初始化模拟退火采样器。

        参数:
          n_chains: 并行链数
          T_max: 初始温度
          cooling_rate: 冷却率（0.90~0.99）
        """
        if not OPTIMIZERS_AVAILABLE:
            self._log("[SA] 优化器模块不可用")
            self.use_sa = False
            return False

        self.sa_bridge = SABridge(
            param_search_space=PARAM_SEARCH_SPACE,
            n_chains=n_chains,
            T_max=T_max,
            cooling_rate=cooling_rate,
        )
        self.use_sa = True
        stats = self.sa_bridge.get_stats()
        self._log(f"[SA] 已初始化 (维度={stats.get('n_active_chains', '?')}, "
                  f"链数={n_chains})")
        return True

    def init_optimizer(self) -> str:
        """
        自动选择并初始化最优优化器。

        策略: BO(≤30维) → CMA-ES → SA → 随机搜索

        返回:
            'BO' | 'CMA-ES' | 'SA' | 'random'
        """
        if not OPTIMIZERS_AVAILABLE:
            self._log("[优化器] scipy不可用，使用随机搜索")
            return 'random'

        # 获取参数维度
        from optimizers.bo_surrogate import ParameterEncoder
        encoder = ParameterEncoder(PARAM_SEARCH_SPACE)
        n_dims = encoder.n_free
        self._log(f"[优化器] 参数空间: 总{encoder.n_total}个, 自由{n_dims}个")

        # 始终优先尝试 BO
        if n_dims > 30:
            self._log(f"[优化器] ⚠ 维度{n_dims}>30, BO可能较慢，但仍将尝试")
        self._log(f"[优化器] 尝试贝叶斯优化(BO)...")
        try:
            self.init_bo('backtest')
            if self.use_bo:
                self._log(f"[优化器] ✅ 已启用: 贝叶斯优化 (BO), 维度={n_dims}")
                return 'BO'
        except Exception as e:
            self._log(f"[优化器] BO 初始化失败: {e}")

        # BO失败 → 尝试 CMA-ES
        self._log(f"[优化器] BO不可用, 尝试CMA-ES...")
        try:
            self.init_cmaes()
            if self.use_cmaes:
                self._log("[优化器] ✅ 已启用: CMA-ES")
                return 'CMA-ES'
        except Exception as e:
            self._log(f"[优化器] CMA-ES 初始化失败: {e}")

        # CMA-ES失败 → 尝试 SA（模拟退火）
        self._log(f"[优化器] 尝试模拟退火(SA)...")
        try:
            self.init_sa()
            if self.use_sa:
                self._log("[优化器] ✅ 已启用: 模拟退火 (SA)")
                return 'SA'
        except Exception as e:
            self._log(f"[优化器] SA 初始化失败: {e}")

        # 全部失败 → 随机搜索
        self._log("[优化器] 所有优化器不可用，使用随机搜索")
        return 'random'

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

    def _load_history_detail(self):
        """加载历史详情记录"""
        history_file = os.path.join(os.path.dirname(self.tried_log_file),
                                    "backtest_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.history_detail = data.get('combos', [])
                self.param_performance = data.get('param_perf', {})
                self.combo_counter = data.get('counter', 0)
                # 恢复top_combos（从history_detail中取top-10，键名需转换）
                sorted_combos = sorted(
                    self.history_detail,
                    key=lambda x: x.get('avg_hits', 0), reverse=True)
                self.top_combos = []
                for entry in sorted_combos[:10]:
                    self.top_combos.append({
                        'params': entry.get('params_snapshot', {}),
                        'weights': entry.get('weights_snapshot', {}),
                        'avg_hits': entry.get('avg_hits', 0),
                        'combo_id': entry.get('combo_id', 0),
                    })
            except Exception:
                self.history_detail = []
                self.param_performance = {}
                self.combo_counter = 0
                self.top_combos = []

    def _save_history_detail(self):
        """保存历史详情记录"""
        history_file = os.path.join(os.path.dirname(self.tried_log_file),
                                    "backtest_history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'counter': self.combo_counter,
                    'combos': self.history_detail,
                    'param_perf': self.param_performance,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _save_best_params(self):
        """保存回测最优参数（供求解模式读取）"""
        best_file = os.path.join(os.path.dirname(self.tried_log_file),
                                 "best_backtest_params.json")
        os.makedirs(os.path.dirname(best_file), exist_ok=True)

        # 加载已有记录（保留历史最优）
        existing = {}
        if os.path.exists(best_file):
            try:
                with open(best_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # 更新全局最优
        if self.best_overall:
            prev_best = existing.get('best_overall', {})
            if self.best_overall['avg_hits'] >= prev_best.get('avg_hits', -999):
                existing['best_overall'] = self.best_overall

        # 更新最新期最优
        if self.best_recent:
            prev_recent = existing.get('best_recent', {})
            if self.best_recent['total_hits'] >= prev_recent.get('total_hits', -999):
                existing['best_recent'] = self.best_recent

        existing['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        existing['lottery_type'] = self.lottery_type

        try:
            with open(best_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def load_best_params(lottery_type: str = None) -> Dict:
        """静态方法：加载回测最优参数"""
        best_file = os.path.join("logs", "best_backtest_params.json")
        if not os.path.exists(best_file):
            return {}
        try:
            with open(best_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if lottery_type and data.get('lottery_type') != lottery_type:
                return {}
            return data
        except Exception:
            return {}

    def _update_param_performance(self, params: Dict, score: float):
        """更新参数表现追踪（加权采样依据）"""
        for method_name, method_params in params.items():
            if method_name not in self.param_performance:
                self.param_performance[method_name] = {}
            for pname, pval in method_params.items():
                if pname not in self.param_performance[method_name]:
                    self.param_performance[method_name][pname] = {}
                pval_str = str(pval)
                if pval_str not in self.param_performance[method_name][pname]:
                    self.param_performance[method_name][pname][pval_str] = {
                        'count': 0, 'total_score': 0.0}
                perf = self.param_performance[method_name][pname][pval_str]
                perf['count'] += 1
                perf['total_score'] += score

    def _get_param_weight(self, method_name: str, pname: str, pval) -> float:
        """获取某个参数值的加权权重（用于加权采样）"""
        perf = self.param_performance.get(method_name, {}).get(pname, {}).get(str(pval))
        if perf and perf['count'] > 0:
            avg = perf['total_score'] / perf['count']
            return max(0.01, avg + 0.1)  # 保底权重0.01，避免完全排除
        return 1.0  # 新值默认权重

    def _get_perturb_ratio(self) -> float:
        """根据当前最佳命中率动态调整扰动比例（命中越高扰动越小）"""
        if self.best_score >= 5.0:
            return 0.02   # 7中5: 扰动~1个参数，高度收敛
        elif self.best_score >= 4.0:
            return 0.05   # 7中4: 扰动~3个参数
        elif self.best_score >= 3.0:
            return 0.08   # 7中3: 扰动~5个参数
        return self.perturb_ratio  # 默认20%

    # ========================================================================
    #  智能组合生成器（3阶段：探索→收敛→随机脉冲）
    # ========================================================================

    def _sample_params_weighted(self, rng: np.random.RandomState
                                ) -> Dict[str, Dict]:
        """加权采样：历史表现好的参数值有更高概率被选中"""
        sampled = {}
        for method_name, space in PARAM_SEARCH_SPACE.items():
            config = {}
            for pname, pvalues in space.items():
                # 计算每个候选值的权重
                w_list = []
                for pv in pvalues:
                    w = self._get_param_weight(method_name, pname, pv)
                    w_list.append(w)
                total_w = sum(w_list)
                if total_w > 0:
                    probs = [w / total_w for w in w_list]
                    idx = rng.choice(len(pvalues), p=probs)
                else:
                    idx = rng.randint(0, len(pvalues))
                config[pname] = pvalues[idx]
            sampled[method_name] = config
        return sampled

    def _sample_params_random(self, rng: np.random.RandomState
                              ) -> Dict[str, Dict]:
        """纯随机采样（脉冲用）"""
        sampled = {}
        for method_name, space in PARAM_SEARCH_SPACE.items():
            config = {}
            for pname, pvalues in space.items():
                config[pname] = pvalues[rng.randint(0, len(pvalues))]
            sampled[method_name] = config
        return sampled

    def _sample_params_perturbed(self, rng: np.random.RandomState,
                                  base_params: Dict, perturb_ratio: float
                                  ) -> Dict[str, Dict]:
        """在基础参数上扰动：随机替换一定比例的参数值"""
        import copy
        new_params = copy.deepcopy(base_params)

        # 收集所有可扰动的参数位置
        all_params_flat = []
        for method_name, space in PARAM_SEARCH_SPACE.items():
            method_params = new_params.get(method_name, {})
            for pname, pvalues in space.items():
                all_params_flat.append((method_name, pname, pvalues))

        # 随机选择要扰动的参数
        n_perturb = max(1, int(len(all_params_flat) * perturb_ratio))
        perturb_indices = rng.choice(
            len(all_params_flat), n_perturb, replace=False)

        for idx in perturb_indices:
            method_name, pname, pvalues = all_params_flat[idx]
            if method_name not in new_params:
                new_params[method_name] = {}
            # 随机选一个不同于当前值的新值
            current = new_params[method_name].get(pname)
            candidates = [v for v in pvalues if v != current]
            if candidates:
                new_params[method_name][pname] = candidates[
                    rng.randint(0, len(candidates))]
            else:
                new_params[method_name][pname] = pvalues[
                    rng.randint(0, len(pvalues))]

        return new_params

    def _sample_weights(self, rng: np.random.RandomState) -> Dict[str, Dict]:
        """随机采样 65 个独立 composite_weights（范围 [-500.0, 500.0]）"""
        w_min, w_max = WEIGHT_SEARCH_SPACE['composite_weight_range']

        cw = {}
        for mk in METHOD_NAMES_NEW:
            for gn in GRANULARITY_NAMES:
                key = f'{mk}@{gn}'
                cw[key] = round(float(rng.uniform(w_min, w_max)), 4)

        return {'composite_weights': cw}

    def _perturb_weights(self, rng: np.random.RandomState,
                         base_weights: Dict, sigma: float = 0.15
                         ) -> Dict[str, Dict]:
        """在最优 composite_weights 基础上加高斯噪声"""
        cw = {}
        for key, w in base_weights.get('composite_weights', {}).items():
            noise = rng.normal(0, sigma)
            cw[key] = round(max(-500.0, min(500.0, w + noise)), 4)

        return {'composite_weights': cw}

    def _determine_phase(self) -> str:
        """确定当前搜索阶段"""
        if self.max_search_time > 0:
            elapsed = time.time() - self.start_time
            if elapsed < self.max_search_time * self.phase_switch_ratio:
                return 'exploration'
        elif self.combo_counter < 50:
            return 'exploration'
        return 'convergence'

    def _generate_combo(self, rng: np.random.RandomState,
                        prefer_new: bool = True) -> Tuple[Dict, Dict, str]:
        """
        智能生成一组(参数, 权重)组合。

        BO模式: 用贝叶斯优化建议替代随机采样
        随机模式（原3阶段策略）: exploration/convergence/pulse

        返回:
            (params, weights, combo_hash, phase_label)
        """
        # BO 模式：用贝叶斯优化建议
        if self.use_bo and self.bo_bridge is not None:
            params, weights, phase_label = self.bo_bridge.suggest_combo()
            h = self._combo_hash(params, weights)
            if prefer_new and h in self.tried_combos:
                params = self._sample_params_random(rng)
                weights = self._sample_weights(rng)
                h = self._combo_hash(params, weights)
                phase_label = 'bo_fallback'
            return params, weights, h, phase_label

        # CMA-ES 模式：种群进化建议
        if self.use_cmaes and self.cmaes_bridge is not None:
            params, weights, phase_label = self.cmaes_bridge.suggest_combo()
            h = self._combo_hash(params, weights)
            return params, weights, h, phase_label

        # SA 模式：多链退火建议
        if self.use_sa and self.sa_bridge is not None:
            params, weights, phase_label = self.sa_bridge.suggest_combo()
            h = self._combo_hash(params, weights)
            return params, weights, h, phase_label

        # 原3阶段随机搜索（保留作为备用）
        max_attempts = 500
        self.phase = self._determine_phase()
        phase_label = self.phase

        for _ in range(max_attempts):
            # 随机脉冲：每N个组合做一次纯随机
            if (self.combo_counter > 0 and
                    self.combo_counter % self.pulse_interval == 0):
                params = self._sample_params_random(rng)
                weights = self._sample_weights(rng)
                phase_label = 'pulse'
            elif self.phase == 'exploration':
                # 探索期：加权随机采样
                params = self._sample_params_weighted(rng)
                weights = self._sample_weights(rng)
            else:
                # 收敛期：基于Top-5微调
                if self.top_combos and rng.random() < 0.85:
                    # 85%概率：从最优组合开始微调
                    base = self.top_combos[rng.randint(
                        0, min(len(self.top_combos), 5))]
                    perturb_ratio = self._get_perturb_ratio()
                    params = self._sample_params_perturbed(
                        rng, base['params'], perturb_ratio)
                    weights = self._perturb_weights(
                        rng, base['weights'], sigma=0.1)
                else:
                    # 15%概率：加权随机（保持探索）
                    params = self._sample_params_weighted(rng)
                    weights = self._sample_weights(rng)

            h = self._combo_hash(params, weights)
            if prefer_new and h not in self.tried_combos:
                return params, weights, h, phase_label
            elif not prefer_new:
                return params, weights, h, phase_label

        # fallback
        params = self._sample_params_random(rng)
        weights = self._sample_weights(rng)
        h = self._combo_hash(params, weights)
        return params, weights, h, 'fallback'

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

        # 生成初始组合池（预生成少量，后续按需生成）
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        combo_pool = []  # [(params, weights, h, cid, phase_label)]
        skipped = 0

        batch_size = 10  # 小批量让阶段切换更及时
        while len(combo_pool) < batch_size:
            params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=True)
            if h in self.tried_combos:
                skipped += 1
                if skipped > batch_size * 3:
                    params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=False)
            combo_pool.append((params, weights, h, len(combo_pool), phase_lbl))
            self.combo_counter += 1

        if skipped > 0:
            self._log(f"生成组合时跳过{skipped}组已尝试过的组合")

        # 多线程并行评估
        combo_idx = 0
        batch_submitted = 0
        active_futures = {}

        max_concurrent = max(1, min(self.num_workers, 8))
        self._phase_switched_at = self.combo_counter

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:

            while self.running:
                if self.max_search_time > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.max_search_time:
                        self._log(f"达到时间上限({self.max_search_time}秒)，停止搜索")
                        for f in list(active_futures.keys()):
                            f.cancel()
                        break

                # 按需补充组合池
                if batch_submitted >= len(combo_pool):
                    for _ in range(10):
                        params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=True)
                        if h in self.tried_combos:
                            skipped += 1
                        self.combo_counter += 1
                        combo_pool.append((params, weights, h, len(combo_pool), phase_lbl))

                # 提交新任务
                while (len(active_futures) < max_concurrent and
                       batch_submitted < len(combo_pool)):
                    params, weights, h, cid, phase_lbl = combo_pool[batch_submitted]
                    future = executor.submit(self.evaluate_combo, params, weights,
                                            seed=cid, combo_id=cid)
                    active_futures[future] = (cid, h, params, weights, phase_lbl)
                    batch_submitted += 1

                if not active_futures:
                    if self.max_search_time == 0:
                        break
                    time.sleep(0.5)
                    continue

                try:
                    done = list(as_completed(active_futures, timeout=5.0))
                except TimeoutError:
                    continue

                for future in done:
                    cid, h, params, weights, phase_lbl = active_futures.pop(future)
                    try:
                        result = future.result(timeout=5)
                    except Exception as e:
                        self._log(f"组合#{cid}评估异常: {e}")
                        continue

                    if 'error' in result:
                        continue

                    # 记录结果
                    score = result['avg_total_hits']
                    self.all_results.append(result)
                    self.tried_combos[h] = score

                    # 优化器更新（4.3+）
                    if self.use_bo and self.bo_bridge is not None:
                        self.bo_bridge.update(params, score)
                    if self.use_cmaes and self.cmaes_bridge is not None:
                        self.cmaes_bridge.update(params, score)
                    if self.use_sa and self.sa_bridge is not None:
                        self.sa_bridge.update(params, score)

                    # 更新智能搜索追踪（BO模式下仍保留，用于日志/降级）
                    self._update_param_performance(params, score)
                    history_entry = {
                        'combo_id': cid,
                        'phase': phase_lbl,
                        'avg_hits': round(score, 4),
                        'max_hits': result['max_total_hits'],
                        'hit_rate_5plus': round(result['hit_rate_5plus'], 4),
                        'eval_time': result.get('evaluation_time', 0),
                        'params_snapshot': {
                            mk: {pk: pv for pk, pv in mp.items()}
                            for mk, mp in params.items()
                        },
                        'weights_snapshot': {
                            'composite_weights': dict(weights.get('composite_weights', {})),
                        },
                    }
                    self.history_detail.append(history_entry)

                    # 更新Top-10
                    self.top_combos.append({
                        'params': params, 'weights': weights,
                        'avg_hits': score, 'combo_id': cid,
                    })
                    self.top_combos.sort(key=lambda x: x['avg_hits'], reverse=True)
                    self.top_combos = self.top_combos[:10]

                    # 更新最佳（全局）
                    if score > self.best_score:
                        self.best_score = score
                        self.best_combo = result
                        self.best_overall = {
                            'params': {mk: dict(mp) for mk, mp in params.items()},
                            'weights': {
                                'composite_weights': dict(weights.get('composite_weights', {})),
                            },
                            'avg_hits': round(score, 4),
                            'max_hits': result['max_total_hits'],
                            'test_periods': len(result.get('period_results', [])),
                            'combo_id': cid,
                            'lottery_type': self.lottery_type,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        phase_str = phase_lbl
                        perturb = self._get_perturb_ratio()
                        self._log(
                            f"★ 新最佳 #{cid}[{phase_str}]: 平均命中={score:.3f}, "
                            f"最高={result['max_total_hits']}, "
                            f"5+率={result['hit_rate_5plus']:.1%}, "
                            f"扰动率={perturb:.0%}, "
                            f"耗时={result['evaluation_time']:.0f}s")

                    # 追踪最新期（period_idx=0）的最高命中
                    for pr in result.get('period_results', []):
                        if pr['period_idx'] == 0:
                            if (self.best_recent is None or
                                    pr['total_hits'] > self.best_recent['total_hits']):
                                self.best_recent = {
                                    'params': {mk: dict(mp) for mk, mp in params.items()},
                                    'weights': {
                                        'composite_weights': dict(weights.get('composite_weights', {})),
                                    },
                                    'total_hits': pr['total_hits'],
                                    'main_hits': pr['main_hits'],
                                    'aux_hits': pr['aux_hits'],
                                    'combo_id': cid,
                                    'lottery_type': self.lottery_type,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                }
                            break

                    combo_idx += 1

                    # 进度报告
                    elapsed = time.time() - self.start_time
                    pct = min(95, (elapsed / self.max_search_time * 100)
                             if self.max_search_time > 0 else (combo_idx * 10))
                    status = (f"已试{combo_idx}组[{self.phase}], "
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

        # 保存去重日志 + 历史详情 + 最优参数
        self._save_tried_combos()
        self._save_history_detail()
        self._save_best_params()

        if not self.best_combo:
            return {
                'success': False,
                'error': f'在{total_time:.0f}秒内没有完成任何有效评估',
                'total_time': total_time,
            }

        n_explore = sum(1 for h in self.history_detail if h['phase'] == 'exploration')
        n_converge = sum(1 for h in self.history_detail if h['phase'] == 'convergence')
        n_pulse = sum(1 for h in self.history_detail if h['phase'] == 'pulse')

        self._log(f"\n回测完成! 总耗时{total_time:.0f}秒, "
                  f"尝试{len(self.all_results)}组, 跳过{skipped}组已试")
        self._log(f"阶段分布: 探索{n_explore} | 收敛{n_converge} | 脉冲{n_pulse}")
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
            'phase_stats': {'exploration': n_explore, 'convergence': n_converge, 'pulse': n_pulse},
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
                ['复合权重 (方法@颗粒度)', json.dumps(
                    bc['weights'].get('composite_weights', {}),
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
                'ml': '方法4: LightGBM',
                'markov': '方法5: 马尔可夫分析',
                'montecarlo': '方法6: 蒙特卡罗模拟',
                'clustering': '方法7: 聚类分析',
                'ngram': '方法8: N-gram分析',
                'xgboost': '方法9: XGBoost',
                'bayesian': '方法10: 贝叶斯推断',
                'kalman': '方法11: 卡尔曼滤波',
                'poisson': '方法12: 泊松回归',
                'cooccurrence': '方法13: 共生矩阵分析',
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

            # Sheet 4: 搜索历史详情（含阶段、参数、权重）
            if self.history_detail:
                history_rows = []
                for h in self.history_detail:
                    # 展平关键参数
                    flat_params = {}
                    for mk, mp in h.get('params_snapshot', {}).items():
                        for pk, pv in mp.items():
                            flat_params[f'{mk}.{pk}'] = pv
                    row = {
                        '组合ID': h.get('combo_id', 0),
                        '阶段': h.get('phase', '?'),
                        '平均命中': h.get('avg_hits', 0),
                        '最高命中': h.get('max_hits', 0),
                        '5+命中率': h.get('hit_rate_5plus', 0),
                        '评估耗时(s)': h.get('eval_time', 0),
                    }
                    # 合并展平的参数
                    row.update(flat_params)
                    history_rows.append(row)
                if history_rows:
                    hist_df = pd.DataFrame(history_rows)
                    hist_df = hist_df.sort_values('平均命中', ascending=False)
                    hist_df.to_excel(
                        writer, sheet_name="搜索历史", index=False)

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
#  求解模式引擎：反向搜索满足容差条件的参数组合
# ============================================================================

class SolveEngine(BacktestEngine):
    """求解模式引擎 4.3 — 找到所有满足容差条件的(参数,权重)组合"""

    def __init__(self):
        super().__init__()
        self.solutions = []          # 所有满足条件的解
        self.tolerance_main = 5      # 主球最低命中
        self.tolerance_aux = 1       # 辅助球最低命中
        self.solve_periods = 1       # 求解期数
        self._solution_hashes = set()  # 去重（避免重复收集同一组合）
        self._best_solve_score = 0.0  # 用于智能搜索引导

        # 线性权重求解（4.3+）
        self.weight_solver = None     # LinearWeightSolver 实例
        self.use_linear_weights = False  # 是否用线性求解替代权重采样

        # 求解模式选择（4.3+）
        self.solve_mode = 'bo_linear'  # 'bo_linear' | 'best_params' | 'random'

    def set_solve_config(self, solve_periods: int = 1,
                         tolerance_main: int = 5,
                         tolerance_aux: int = 1,
                         max_search_time: int = 0,
                         num_workers: int = 4):
        """设置求解参数（一次性设置所有配置，避免被覆盖）"""
        self.solve_periods = solve_periods
        self.tolerance_main = tolerance_main
        self.tolerance_aux = tolerance_aux
        self.test_periods = solve_periods
        self.max_search_time = max_search_time
        self.num_workers = max(1, num_workers)
        # 求解模式始终使用全部5种颗粒度（只需跑一次，无需智能缩减）
        self.granularities = [50, 100, 500, 1000, 0]
        self.gran_names = ['50期', '100期', '500期', '1000期', '全部期']

    def _check_solution(self, result: Dict) -> Tuple[bool, float]:
        """
        检查组合是否为有效解。

        返回: (is_solution, closeness_score)
        closeness_score 用于智能搜索引导：越接近容差越高
        """
        if 'error' in result or not result.get('period_results'):
            return False, 0.0

        all_pass = True
        total_closeness = 0.0
        n_periods = len(result['period_results'])

        if n_periods == 0:
            return False, 0.0

        for pr in result['period_results']:
            main_ok = pr['main_hits'] >= self.tolerance_main
            aux_ok = pr['aux_hits'] >= self.tolerance_aux
            if not main_ok or not aux_ok:
                all_pass = False
            # 每期的贴近度（0~1，越接近容差越高）
            main_closeness = min(1.0, pr['main_hits'] / max(1, self.tolerance_main))
            aux_closeness = min(1.0, pr['aux_hits'] / max(1, self.tolerance_aux))
            total_closeness += (main_closeness * 0.7 + aux_closeness * 0.3)

        avg_closeness = total_closeness / n_periods
        return all_pass, avg_closeness

    # ------------------------------------------------------------------
    #  线性权重求解集成（4.3+）
    # ------------------------------------------------------------------

    def _evaluate_params_only(self, params: Dict, seed: int = 0
                               ) -> Tuple[List[List[Dict]], List[Tuple[List[int], List[int]]]]:
        """
        仅运行预测（不合并），返回每期各组合的原始预测。

        返回:
          all_period_predictions: [
            [{method_key, granularity, predicted_main, predicted_aux}, ...],
            ...每期一组...
          ]
          all_period_actuals: [(main_list, aux_list), ...]
        """
        import time as _time
        actual_test_count = min(self.test_periods, self.total_periods - 10)
        all_period_preds = []
        all_period_actuals = []

        for period_idx in range(actual_test_count):
            # 注意: 不检查 self.running（此方法可能在 run() 外部调用）

            # 训练数据
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

            all_period_actuals.append((actual_main, actual_aux))

            # 各颗粒度预测
            period_preds = []
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
                    gran_name = self.gran_names[g_idx] if g_idx < len(self.gran_names) else f'{gran}期'
                    gran_results = self.predictor.predict_all(
                        gran_data, params=params,
                        seed=seed + period_idx * 100 + g_idx)

                    for mk, result in gran_results.items():
                        if mk == 'comprehensive' or 'error' in result:
                            continue
                        pred = result.get('predictions', {})
                        period_preds.append({
                            'method_key': mk,
                            'granularity': gran_name,
                            'predicted_main': pred.get(self.predictor.main_name, []),
                            'predicted_aux': pred.get(self.predictor.aux_name, []),
                        })
                except Exception:
                    continue

            all_period_preds.append(period_preds)

        return all_period_preds, all_period_actuals

    def _merge_with_weights(self, period_predictions: List[Dict],
                             weights: Dict) -> Dict[str, List[int]]:
        """用指定权重复制合并（composite_weights 直接查表）"""
        from collections import defaultdict
        composite_w = weights.get('composite_weights', {})

        main_votes = defaultdict(float)
        aux_votes = defaultdict(float)

        for entry in period_predictions:
            mk = entry['method_key']
            gk = entry['granularity']
            key = f'{mk}@{gk}'
            weight = composite_w.get(key, 1.0)

            for n in entry.get('predicted_main', []):
                main_votes[n] += weight
            for n in entry.get('predicted_aux', []):
                aux_votes[n] += weight

        sorted_main = sorted(main_votes.items(), key=lambda x: x[1], reverse=True)
        sorted_aux = sorted(aux_votes.items(), key=lambda x: x[1], reverse=True)

        main_count = 6 if self.lottery_type == 'ssq' else 5
        aux_count = 1 if self.lottery_type == 'ssq' else 2

        return {
            'main': sorted([n for n, _ in sorted_main[:main_count]]),
            'aux': sorted([n for n, _ in sorted_aux[:aux_count]]),
        }

    def run(self, num_combos_to_try: int = None) -> Dict[str, Any]:
        """
        运行求解搜索。

        支持三种模式:
          - best_params: 读取回测最优参数 → 固定参数 → 线性求解权重 → 直接输出
          - bo_linear: BO搜索参数 + 线性求解权重（需 init_bo('solve') + use_linear_weights=True）
          - random: 纯随机搜索（兼容旧版）
        """
        if self.data_reverse is None:
            return {'success': False, 'error': '请先加载数据'}

        # 模式1: 回测最优 + 线性求解（不需搜索，直接出结果）
        if self.solve_mode == 'best_params':
            return self._run_best_params_solve()

        # 模式2: BO + 线性求解
        if self.solve_mode == 'bo_linear':
            if self.use_linear_weights and self.use_bo and self.bo_bridge is not None:
                return self._run_bo_linear_solve(num_combos_to_try)
            # 降级：如果BO未初始化，自动初始化
            if OPTIMIZERS_AVAILABLE and not self.use_bo:
                self.init_bo('solve')
                self.use_linear_weights = True
                if self.bo_bridge is not None:
                    return self._run_bo_linear_solve(num_combos_to_try)

        # 模式3: 原随机模式（兼容/降级）
        return self._run_random_solve(num_combos_to_try)

    def _run_bo_linear_solve(self, num_combos_to_try: int = None) -> Dict[str, Any]:
        """
        BO + 线性权重求解模式:
          1. BO 建议模型参数（不含权重）
          2. evaluate_params_only → 各期各组合的原始预测
          3. LinearWeightSolver → 最优合并权重（NNLS）
          4. 合并验证 → 检查容差 → 收集解
        """
        self.running = True
        self.start_time = time.time()
        self.solutions = []
        self._solution_hashes = set()
        self.best_score = -1.0
        self.all_results = []
        self._best_solve_score = 0.0

        # 初始化线性权重求解器
        if self.weight_solver is None:
            self.weight_solver = LinearWeightSolver(self.lottery_type)

        actual_test_count = min(self.solve_periods, self.total_periods - 10)
        self._log(f"[BO+线性] 求解启动: 最新{actual_test_count}期, "
                  f"容差=主{self.tolerance_main}/辅{self.tolerance_aux}, "
                  f"上限={'不限' if self.max_search_time == 0 else f'{self.max_search_time}秒'}")

        total_evaluated = 0
        bo = self.bo_bridge.bo  # 底层 BOSuggestor

        while self.running:
            # 时间检查
            if self.max_search_time > 0:
                elapsed = time.time() - self.start_time
                if elapsed > self.max_search_time:
                    self._log(f"达到时间上限({self.max_search_time}秒)，停止搜索")
                    break

            # BO 建议模型参数（不含权重）
            x = bo.suggest()
            params = bo.encoder.decode(x)

            # 评估：只做预测，不合并
            all_period_preds, all_period_actuals = self._evaluate_params_only(
                params, seed=total_evaluated)

            if not all_period_preds or not all_period_actuals:
                total_evaluated += 1
                continue

            # 线性求解最优权重
            solve_result = self.weight_solver.solve_multi_period(
                all_period_preds, all_period_actuals)

            if 'error' in solve_result:
                total_evaluated += 1
                continue

            optimal_weights = {
                'composite_weights': solve_result['composite_weights'],
            }

            # 验证：用最优权重复制合并并计算命中
            period_results = []
            all_main_hits = []
            all_aux_hits = []
            all_total_hits = []

            for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                    zip(all_period_preds, all_period_actuals)):
                merged = self._merge_with_weights(period_preds, optimal_weights)
                main_hits = len(set(merged['main']) & set(actual_main))
                aux_hits = len(set(merged['aux']) & set(actual_aux))

                period_results.append({
                    'period_idx': p_idx,
                    'merged_main': merged['main'],
                    'merged_aux': merged['aux'],
                    'actual_main': actual_main,
                    'actual_aux': actual_aux,
                    'main_hits': main_hits,
                    'aux_hits': aux_hits,
                    'total_hits': main_hits + aux_hits,
                })
                all_main_hits.append(main_hits)
                all_aux_hits.append(aux_hits)
                all_total_hits.append(main_hits + aux_hits)

            if not period_results:
                continue

            avg_hits = np.mean(all_total_hits)
            max_hits = max(all_total_hits)

            result = {
                'params': params,
                'weights': optimal_weights,
                'period_results': period_results,
                'avg_total_hits': round(float(avg_hits), 4),
                'max_total_hits': int(max_hits),
            }

            total_evaluated += 1
            self.all_results.append(result)

            # 检查是否为解
            is_solution, closeness = self._check_solution(result)
            combo_h = self._combo_hash(params, optimal_weights)

            if is_solution and combo_h not in self._solution_hashes:
                self._solution_hashes.add(combo_h)
                self.solutions.append({
                    'combo_id': total_evaluated,
                    'params': params,
                    'weights': optimal_weights,
                    'combo_hash': combo_h,
                    'period_results': period_results,
                    'avg_total_hits': round(float(avg_hits), 4),
                    'max_total_hits': int(max_hits),
                    'solve_method': 'bo_linear',
                })
                self._log(
                    f"★ 找到解 #{len(self.solutions)}! "
                    f"#{total_evaluated} 平均={avg_hits:.3f}, "
                    f"线性权重求解")

            # BO 更新（用 closeness 作为目标）
            solve_score = closeness if closeness > 0 else avg_hits / 7.0
            bo.update(x, solve_score)

            # 进度
            elapsed = time.time() - self.start_time
            pct = min(95, (elapsed / self.max_search_time * 100)
                     if self.max_search_time > 0 else (total_evaluated * 10))
            status = (f"[BO+线性] 已试{total_evaluated}组 | "
                     f"找到{len(self.solutions)}个解 | "
                     f"耗时{elapsed:.0f}s")
            self._progress(pct, status)

            if num_combos_to_try and total_evaluated >= num_combos_to_try:
                break

        total_time = time.time() - self.start_time
        self.running = False

        self._log(f"\n[BO+线性] 求解完成! 总耗时{total_time:.0f}秒, "
                  f"评估{total_evaluated}组, 找到{len(self.solutions)}个解")

        return {
            'success': True,
            'solutions': self.solutions,
            'total_evaluated': total_evaluated,
            'total_time': total_time,
            'solve_method': 'bo_linear',
            'solve_config': {
                'periods': self.solve_periods,
                'tolerance_main': self.tolerance_main,
                'tolerance_aux': self.tolerance_aux,
            },
        }

    def _run_best_params_solve(self) -> Dict[str, Any]:
        """
        回测最优 + 线性求解模式（反解权重配方）:

        ┌─────────────────────────────────────────────────────┐
        │ 1. 固定参数 = 回测历史最优                            │
        │ 2. 给定答案 = 最新一期实际开奖号码（作为约束条件）        │
        │ 3. NNLS 反解 → 方法权重 + 颗粒度权重                   │
        │ 4. 验算：用反解的权重合并 → 是否完全重合实际号码？       │
        │ 5. 预测：用同样的(参数, 权重)预测未开奖的最新一期        │
        │                                                      │
        │ 输出核心: 模型参数 + 方法权重 + 颗粒度权重               │
        │           + 验算结果(是否完全重合) + 下一期预测          │
        └─────────────────────────────────────────────────────┘
        """
        self.running = True
        self.start_time = time.time()

        # ════════════════════════════════════════════════════════
        # 阶段 1: 加载回测最优参数（固定不动）
        # ════════════════════════════════════════════════════════
        best_data = BacktestEngine.load_best_params(self.lottery_type)

        if not best_data:
            self.running = False
            return {
                'success': False, 'solve_mode': 'best_params',
                'error': '没有找到回测最优参数。请先运行至少一次回测模式。',
                'total_time': time.time() - self.start_time,
            }

        # 优先用最新期最优 → 全局最优
        best_recent = best_data.get('best_recent')
        best_overall = best_data.get('best_overall')

        if best_recent:
            best_params = best_recent['params']
            param_source = 'best_recent'
            param_score = (f"最新期命中={best_recent['total_hits']}"
                          f"({best_recent['main_hits']}主+{best_recent['aux_hits']}辅)"
                          f" @ combo#{best_recent['combo_id']}")
        elif best_overall:
            best_params = best_overall['params']
            param_source = 'best_overall'
            param_score = (f"平均命中={best_overall['avg_hits']}"
                          f" @ combo#{best_overall['combo_id']}")
        else:
            self.running = False
            return {
                'success': False, 'solve_mode': 'best_params',
                'error': '回测最优参数数据不完整。请重新运行回测。',
                'total_time': time.time() - self.start_time,
            }

        self._log(f"[回测最优+求解] 参数来源: {param_source} | {param_score}")

        # ════════════════════════════════════════════════════════
        # 阶段 2: 固定参数 → 对最新一期跑预测（获取投票矩阵 M）
        # ════════════════════════════════════════════════════════
        if self.weight_solver is None:
            self.weight_solver = LinearWeightSolver(self.lottery_type)

        # 只求解最新已开奖的1期（period_idx=0）
        solve_periods = min(self.solve_periods, self.total_periods - 10)
        all_period_preds, all_period_actuals = self._evaluate_params_only(
            best_params, seed=0)

        if not all_period_preds or not all_period_actuals:
            self.running = False
            return {
                'success': False, 'solve_mode': 'best_params',
                'error': '预测阶段失败，无法生成投票矩阵。',
                'total_time': time.time() - self.start_time,
            }

        self._log(f"  已获取 {len(all_period_preds)} 期投票矩阵, "
                 f"每期 {len(all_period_preds[0]) if all_period_preds else 0} 组预测")

        # ════════════════════════════════════════════════════════
        # 阶段 3: 线性规划反解权重
        #   约束: 每个实际号码得票 > 每个非实际号码得票
        #   求解: scipy.optimize.linprog (HiGHS)
        #   若不可行 → LSTSQ 降级
        # ════════════════════════════════════════════════════════
        solve_result = self.weight_solver.solve_lp_multi_period(
            all_period_preds, all_period_actuals, epsilon=0.01)

        if 'error' in solve_result:
            self.running = False
            return {
                'success': False, 'solve_mode': 'best_params',
                'error': f"线性求解失败: {solve_result['error']}",
                'total_time': time.time() - self.start_time,
            }

        solved_composite_weights = solve_result['composite_weights']
        lp_success = solve_result.get('lp_success', False)
        lp_status = solve_result.get('lp_status', '?')

        optimal_weights = {
            'composite_weights': solved_composite_weights,
        }

        if lp_success:
            self._log(f"  LP 精确求解完成, 约束数={solve_result.get('n_constraints', '?')}")
        else:
            self._log(f"  LP 不可行 → LSTSQ 降级 ({lp_status})")

        # ════════════════════════════════════════════════════════
        # 阶段 4: 验算 — 用反解的权重合并，必须与实际号码完全重合
        # ════════════════════════════════════════════════════════
        verification_results = []
        all_verified = True

        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_preds, all_period_actuals)):
            merged = self._merge_with_weights(period_preds, optimal_weights)

            main_match = set(merged['main']) == set(actual_main)
            aux_match = set(merged['aux']) == set(actual_aux)
            all_match = main_match and aux_match

            verification_results.append({
                'period_idx': p_idx,
                'period_num': self.total_periods - p_idx,
                'merged_main': merged['main'],
                'actual_main': actual_main,
                'merged_aux': merged['aux'],
                'actual_aux': actual_aux,
                'main_match': main_match,
                'aux_match': aux_match,
                'all_match': all_match,
                'main_hits': len(set(merged['main']) & set(actual_main)),
                'aux_hits': len(set(merged['aux']) & set(actual_aux)),
            })

            if not all_match:
                all_verified = False

            status = '[OK] 完全重合' if all_match else '[FAIL] 未完全重合'
            self._log(f"  验算第{self.total_periods - p_idx}期: {status}")
            if not all_match:
                self._log(f"    合并主球={merged['main']}")
                self._log(f"    实际主球={actual_main}")
                self._log(f"    合并辅助={merged['aux']}")
                self._log(f"    实际辅助={actual_aux}")

        # ════════════════════════════════════════════════════════
        # 阶段 5: 用同样的(参数, 权重)预测未开奖的最新一期
        # ════════════════════════════════════════════════════════
        prediction = None
        try:
            # 用全部历史数据训练 → 预测下一期
            all_train_data = self.data_reverse.head(
                min(len(self.data_reverse), self.max_train_periods)).copy()

            future_preds = []
            for g_idx, gran in enumerate(self.granularities):
                if gran > 0 and len(all_train_data) < gran:
                    gran_data = all_train_data
                elif gran > 0:
                    gran_data = all_train_data.head(gran)
                else:
                    gran_data = all_train_data

                if len(gran_data) < 10:
                    continue

                try:
                    gran_name = self.gran_names[g_idx] if g_idx < len(self.gran_names) else f'{gran}期'
                    gran_results = self.predictor.predict_all(
                        gran_data, params=best_params, seed=9999)
                    for mk, result in gran_results.items():
                        if mk == 'comprehensive' or 'error' in result:
                            continue
                        pred = result.get('predictions', {})
                        future_preds.append({
                            'method_key': mk,
                            'granularity': gran_name,
                            'predicted_main': pred.get(self.predictor.main_name, []),
                            'predicted_aux': pred.get(self.predictor.aux_name, []),
                        })
                except Exception:
                    continue

            if future_preds:
                pred_merged = self._merge_with_weights(future_preds, optimal_weights)
                prediction = {
                    'main': pred_merged['main'],
                    'aux': pred_merged['aux'],
                    'num_method_predictions': len(future_preds),
                }
                self._log(f"  预测未开奖期: 主球={prediction['main']}, "
                         f"辅助球={prediction['aux']}")
        except Exception as e:
            self._log(f"  [警告] 预测阶段异常: {e}")

        # ════════════════════════════════════════════════════════
        # 组装最终输出
        # ════════════════════════════════════════════════════════
        total_time = time.time() - self.start_time
        self.running = False

        result = {
            'success': True,
            'solve_mode': 'best_params',

            # ★ 核心产出: 参数 + 权重 配方
            'params': {
                mk: dict(mp) for mk, mp in best_params.items()
            },
            'composite_weights': dict(solved_composite_weights),

            # ★ 验算: 配方能否还原实际号码
            'verification': {
                'all_verified': all_verified,
                'num_periods': len(verification_results),
                'details': verification_results,
            },

            # ★ 预测: 用配方预测未开奖期
            'prediction': prediction,

            # 元信息
            'param_source': param_source,
            'param_score': param_score,
            'lp_success': lp_success,
            'lp_status': str(lp_status),
            'total_time': round(total_time, 1),
            'solve_config': {
                'periods': solve_periods,
                'lottery_type': self.lottery_type,
            },
        }

        self._log(f"\n[回测最优+求解] 完成! 耗时{total_time:.1f}秒")
        self._log(f"  参数来源: {param_source} | {param_score}")
        self._log(f"  求解方法: {'LP精确' if lp_success else 'LSTSQ降级'} ({lp_status})")
        self._log(f"  验算: {'[OK] 全部通过' if all_verified else '[FAIL] 存在不重合'}")

        return result

    def _run_random_solve(self, num_combos_to_try: int = None) -> Dict[str, Any]:
        """
        原随机搜索求解模式（保留作为备用）。
        """
        self.running = True
        self.start_time = time.time()
        self.solutions = []
        self._solution_hashes = set()
        self.best_score = -1.0
        self.all_results = []
        self._best_solve_score = 0.0

        actual_test_count = min(self.solve_periods, self.total_periods - 10)
        self._log(f"求解启动: 求解最新{actual_test_count}期, "
                  f"容差=主球≥{self.tolerance_main},辅助球≥{self.tolerance_aux}, "
                  f"时间上限={'不限' if self.max_search_time == 0 else f'{self.max_search_time}秒'}, "
                  f"{self.num_workers}线程并行")
        self._log(f"已记录{len(self.tried_combos)}组已尝试组合")
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        combo_pool = []
        skipped = 0

        batch_size = 10
        while len(combo_pool) < batch_size:
            params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=True)
            if h in self.tried_combos:
                skipped += 1
                if skipped > batch_size * 3:
                    params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=False)
            combo_pool.append((params, weights, h, len(combo_pool), phase_lbl))
            self.combo_counter += 1

        combo_idx = 0
        batch_submitted = 0
        active_futures = {}
        total_evaluated = 0

        max_concurrent = max(1, min(self.num_workers, 8))

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            while self.running:
                if self.max_search_time > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.max_search_time:
                        self._log(f"达到时间上限({self.max_search_time}秒)，停止搜索")
                        for f in list(active_futures.keys()):
                            f.cancel()
                        break

                if batch_submitted >= len(combo_pool):
                    for _ in range(10):
                        params, weights, h, phase_lbl = self._generate_combo(rng, prefer_new=True)
                        if h in self.tried_combos:
                            skipped += 1
                        self.combo_counter += 1
                        combo_pool.append((params, weights, h, len(combo_pool), phase_lbl))

                while (len(active_futures) < max_concurrent and
                       batch_submitted < len(combo_pool)):
                    params, weights, h, cid, phase_lbl = combo_pool[batch_submitted]
                    future = executor.submit(self.evaluate_combo, params, weights,
                                            seed=cid, combo_id=cid)
                    active_futures[future] = (cid, h, params, weights, phase_lbl)
                    batch_submitted += 1

                if not active_futures:
                    if self.max_search_time == 0:
                        break
                    time.sleep(0.5)
                    continue

                try:
                    done = list(as_completed(active_futures, timeout=5.0))
                except TimeoutError:
                    continue

                for future in done:
                    cid, h, params, weights, phase_lbl = active_futures.pop(future)
                    try:
                        result = future.result(timeout=5)
                    except Exception as e:
                        self._log(f"求解#{cid}评估异常: {e}")
                        continue

                    if 'error' in result:
                        continue

                    total_evaluated += 1
                    score = result['avg_total_hits']
                    self.all_results.append(result)
                    self.tried_combos[h] = score

                    # 检查是否为解
                    is_solution, closeness = self._check_solution(result)
                    if is_solution and h not in self._solution_hashes:
                        self._solution_hashes.add(h)
                        self.solutions.append({
                            'combo_id': cid,
                            'params': params,
                            'weights': weights,
                            'combo_hash': h,
                            'period_results': result.get('period_results', []),
                            'avg_total_hits': score,
                            'max_total_hits': result['max_total_hits'],
                            'phase': phase_lbl,
                        })
                        self._log(
                            f"★ 找到解 #{len(self.solutions)}! "
                            f"#{cid} 平均命中={score:.3f}, "
                            f"主球容差={self.tolerance_main}, "
                            f"辅助球容差={self.tolerance_aux}")

                    # 用 closeness 引导智能搜索（替代 avg_hits）
                    solve_score = closeness if closeness > 0 else score / 7.0
                    self._update_param_performance(params, solve_score)
                    self.history_detail.append({
                        'combo_id': cid,
                        'phase': phase_lbl,
                        'avg_hits': round(score, 4),
                        'max_hits': result['max_total_hits'],
                        'closeness': round(closeness, 4),
                        'is_solution': is_solution,
                        'eval_time': result.get('evaluation_time', 0),
                        'params_snapshot': {mk: dict(mp) for mk, mp in params.items()},
                        'weights_snapshot': {
                            'composite_weights': dict(weights.get('composite_weights', {})),
                        },
                    })

                    # 更新Top-10
                    if solve_score >= self._best_solve_score:
                        self._best_solve_score = solve_score
                    self.top_combos.append({
                        'params': params, 'weights': weights,
                        'avg_hits': solve_score, 'combo_id': cid,
                    })
                    self.top_combos.sort(key=lambda x: x['avg_hits'], reverse=True)
                    self.top_combos = self.top_combos[:10]

                    combo_idx += 1
                    elapsed = time.time() - self.start_time
                    pct = min(95, (elapsed / self.max_search_time * 100)
                             if self.max_search_time > 0 else (combo_idx * 10))
                    status = (f"已试{total_evaluated}组 | "
                             f"找到{len(self.solutions)}个解 | "
                             f"耗时{elapsed:.0f}s")
                    self._progress(pct, status)

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
        self._save_tried_combos()
        self._save_history_detail()

        self._log(f"\n求解完成! 总耗时{total_time:.0f}秒, "
                  f"评估{total_evaluated}组, 找到{len(self.solutions)}个解")

        return {
            'success': True,
            'solutions': self.solutions,
            'total_evaluated': total_evaluated,
            'total_combos_skipped': skipped,
            'total_time': total_time,
            'solve_config': {
                'periods': self.solve_periods,
                'tolerance_main': self.tolerance_main,
                'tolerance_aux': self.tolerance_aux,
            },
        }


class SolveRunner:
    """在后台线程中运行求解模式"""

    def __init__(self, engine: SolveEngine):
        self.engine = engine
        self.thread = None
        self.result = None

    def run_async(self, on_progress=None, on_log=None, on_done=None,
                  on_solution=None):
        """异步运行求解"""
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
                    on_log(f"求解异常: {e}\n{traceback.format_exc()}")
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
