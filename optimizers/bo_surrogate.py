"""
贝叶斯优化代理模型（Bayesian Optimization Surrogate）
=====================================================

用高斯过程 (GP) 代理模型 + 期望改进 (EI) 采集函数，
替代纯随机/扰动采样，大幅提升搜索效率。

核心组件:
  ParameterEncoder  — 离散参数 ↔ 连续向量 编解码
  GPSurrogate       — 高斯过程代理模型
  EIAcquisition     — 期望改进采集函数
  BOSuggestor       — 顶层接口，组合以上组件

用法:
  bo = BOSuggestor(param_space, objective_type='avg_hits')
  x = bo.suggest()           # 建议下一个评估点
  bo.update(x, y)            # 用评估结果更新模型
  params = bo.decode(x)      # 将连续向量转回参数字典

依赖:
  sklearn.gaussian_process (已有)
"""

import copy
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler


# ============================================================================
#  参数编解码器
# ============================================================================

class ParameterEncoder:
    """
    离散参数 ↔ 连续向量 双向编解码。

    离散处理策略:
      每个参数有一组候选值 [v₁, v₂, ..., vₖ]
      连续表示 = 归一化索引 ∈ [0, 1]
      解码: 连续值 × (k-1) → 四舍五入 → 取候选值
    """

    def __init__(self, param_search_space: Dict[str, Dict]):
        """
        参数:
          param_search_space: PARAM_SEARCH_SPACE 格式
            {method_name: {param_name: [candidate_values]}}
        """
        self.space = param_search_space
        self._build_mapping()

    def _build_mapping(self):
        """构建参数映射"""
        self.param_list = []          # [(method_name, param_name, candidate_values)]
        self.free_params = []         # 排除固定参数（只有1个候选值）

        for method_name in sorted(self.space.keys()):
            space = self.space[method_name]
            for pname in sorted(space.keys()):
                pvalues = space[pname]
                self.param_list.append((method_name, pname, pvalues))
                if len(pvalues) > 1:
                    self.free_params.append((method_name, pname, pvalues))

        self.n_total = len(self.param_list)
        self.n_free = len(self.free_params)

    def encode(self, params: Dict[str, Dict]) -> np.ndarray:
        """
        参数字典 → 连续向量 [0,1]ⁿ

        参数:
          params: {method_name: {param_name: value}}

        返回:
          x ∈ [0, 1]^n_free（只编码自由参数）
        """
        x = np.zeros(self.n_free)
        for i, (method_name, pname, pvalues) in enumerate(self.free_params):
            current = params.get(method_name, {}).get(pname, pvalues[0])
            try:
                idx = pvalues.index(current)
            except ValueError:
                idx = 0
            x[i] = idx / max(1, len(pvalues) - 1)
        return x

    def decode(self, x: np.ndarray) -> Dict[str, Dict]:
        """
        连续向量 [0,1]ⁿ → 参数字典

        参数:
          x: 连续向量（长度必须等于 n_free）

        返回:
          {method_name: {param_name: value}}
        """
        params = {}
        for i, (method_name, pname, pvalues) in enumerate(self.free_params):
            if method_name not in params:
                params[method_name] = {}
            # [0,1] → 离散索引
            idx = int(round(np.clip(x[i], 0.0, 1.0) * (len(pvalues) - 1)))
            params[method_name][pname] = pvalues[idx]

        # 补全固定参数
        for method_name, pname, pvalues in self.param_list:
            if len(pvalues) == 1:
                if method_name not in params:
                    params[method_name] = {}
                params[method_name][pname] = pvalues[0]

        return params

    def random_sample(self, rng: np.random.RandomState = None) -> np.ndarray:
        """在连续空间中随机采样"""
        if rng is None:
            rng = np.random
        return rng.uniform(0.0, 1.0, self.n_free)

    def get_bounds(self) -> np.ndarray:
        """返回参数边界 [(0,1), ...]"""
        return np.array([(0.0, 1.0)] * self.n_free)


# ============================================================================
#  高斯过程代理模型
# ============================================================================

class GPSurrogate:
    """
    GP 代理模型，近似 f(x) → y。

    使用 sklearn GaussianProcessRegressor:
      - RBF 核（平滑假设）
      - WhiteKernel（观测噪声）
      - 自动超参数优化
    """

    def __init__(self, n_dimensions: int,
                 kernel_type: str = 'rbf',
                 alpha: float = 1e-6,
                 normalize_y: bool = True):
        """
        参数:
          n_dimensions: 输入维度
          kernel_type: 'rbf' 或 'matern'
          alpha: 噪声水平
          normalize_y: 是否标准化 y
        """
        self.n_dim = n_dimensions

        # 构建核函数
        if kernel_type == 'matern':
            base_kernel = Matern(
                length_scale=[1.0] * n_dimensions,
                length_scale_bounds=(1e-3, 10.0),
                nu=2.5
            )
        else:
            base_kernel = RBF(
                length_scale=[1.0] * n_dimensions,
                length_scale_bounds=(1e-3, 10.0)
            )

        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
            base_kernel + WhiteKernel(noise_level=alpha,
                                       noise_level_bounds=(1e-6, 1e-1))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=normalize_y,
            random_state=42,
        )

        self.X_train = []
        self.y_train = []
        self._fitted = False
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        用观测数据拟合 GP。

        参数:
          X: shape (n_samples, n_dims)
          y: shape (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if len(X) < 2:
            # 样本不足，不能拟合
            self._fitted = False
            return

        self.X_train = X.copy()
        self.y_train = y.copy()

        # 标准化 y（如果启用 normalize_y，GP 内部也会做）
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std < 1e-10:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std

        try:
            self.gp.fit(X, y_norm)
            self._fitted = True
            self._y_mean = y_mean
            self._y_std = y_std
        except Exception:
            self._fitted = False

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GP 预测。

        返回:
          (mean, std) — 均值和标准差（原始尺度）
        """
        X = np.atleast_2d(X)

        if not self._fitted or self.X_train.shape[0] < 2:
            # 未拟合时返回先验
            return np.zeros(X.shape[0]), np.ones(X.shape[0])

        try:
            mu_norm, std_norm = self.gp.predict(X, return_std=True)
            mu = mu_norm * self._y_std + self._y_mean
            std = std_norm * self._y_std
            return mu, std
        except Exception:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])

    def update(self, x: np.ndarray, y: float):
        """增量更新（添加一个样本并重新拟合）"""
        x = np.atleast_1d(x)
        new_X = np.vstack([self.X_train, x.reshape(1, -1)]) \
            if len(self.X_train) > 0 else x.reshape(1, -1)
        new_y = np.append(self.y_train, y) \
            if len(self.y_train) > 0 else np.array([y])
        self.fit(new_X, new_y)


# ============================================================================
#  期望改进 (EI) 采集函数
# ============================================================================

class EIAcquisition:
    """
    期望改进 (Expected Improvement)。

    EI(x) = E[max(0, f(x) - y_best)]
          = σ(x) × [z × Φ(z) + φ(z)]

    其中:
      z = (μ(x) - y_best) / σ(x)
      Φ = 标准正态 CDF
      φ = 标准正态 PDF

    当 σ → 0 时，EI → 0（已探索区域）
    当 μ ≫ y_best 且 σ 大时，EI 大（有潜力的未探索区域）
    """

    def __init__(self, xi: float = 0.01):
        """
        参数:
          xi: 探索系数（越大越倾向探索，0=纯利用）
        """
        self.xi = xi
        self.y_best = -np.inf

    def set_best(self, y_best: float):
        self.y_best = y_best

    def compute(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        计算 EI。

        参数:
          mu: 预测均值
          sigma: 预测标准差

        返回:
          EI 值数组
        """
        sigma = np.maximum(sigma, 1e-10)  # 避免除零
        improvement = mu - self.y_best - self.xi
        z = improvement / sigma

        # z × Φ(z) + φ(z)
        from scipy.stats import norm
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        # 当 sigma 很小时 EI 应该接近 0
        ei[sigma < 1e-10] = 0.0
        # 负的 EI 归零
        ei = np.maximum(ei, 0.0)

        return ei

    def suggest(self, gp: GPSurrogate, bounds: np.ndarray,
                n_candidates: int = 2000,
                rng: np.random.RandomState = None
                ) -> np.ndarray:
        """
        用 EI 选择下一个最值得评估的点。

        算法:
          1. 在参数空间中随机采样 n_candidates 个候选点
          2. 用 GP 预测每个候选点的 (mu, sigma)
          3. 计算每个候选点的 EI
          4. 返回 EI 最高的点

        参数:
          gp: 已拟合的 GP 代理模型
          bounds: 参数边界 [(low, high), ...]
          n_candidates: 随机候选点数（越大越精确但越慢）
          rng: 随机数生成器

        返回:
          最优建议点 x*
        """
        if rng is None:
            rng = np.random

        n_dim = bounds.shape[0]

        # 在边界内随机采样候选点
        candidates = rng.uniform(0, 1, (n_candidates, n_dim))
        candidates = np.clip(candidates, bounds[:, 0], bounds[:, 1])

        # GP 预测
        mu, sigma = gp.predict(candidates)

        # 计算 EI
        ei_values = self.compute(mu, sigma)

        # 选 EI 最高的点
        best_idx = np.argmax(ei_values)

        # 如果 EI 全为 0（所有点都不比当前最优更好），
        # 选不确定度最高的点（纯探索）
        if ei_values[best_idx] < 1e-10:
            best_idx = np.argmax(sigma)

        return candidates[best_idx]

    def suggest_batch(self, gp: GPSurrogate, bounds: np.ndarray,
                      batch_size: int = 5,
                      n_candidates: int = 2000,
                      rng: np.random.RandomState = None
                      ) -> List[np.ndarray]:
        """
        批量建议多个点（用于并行评估）。

        使用贪心"幻想"策略:
          1. 选 EI 最高的点 x₁
          2. 假设 x₁ 的得分 = μ(x₁)，将 (x₁, μ(x₁)) 临时加入 GP
          3. 重新选 EI 最高的点 x₂
          4. 重复直到 batch_size 个点

        注意: 幻想点只在建议期间存在，不会真正更新 GP。
        """
        if rng is None:
            rng = np.random

        batch = []
        # 使用 GP 的副本
        gp_copy = copy.deepcopy(gp)

        for _ in range(batch_size):
            x = self.suggest(gp_copy, bounds, n_candidates, rng)
            batch.append(x)

            # 幻想更新：将 x 以 μ(x) 分数临时加入
            mu, _ = gp_copy.predict(x.reshape(1, -1))
            gp_copy.update(x, mu[0])

        return batch


# ============================================================================
#  BO 建议器（顶层接口）
# ============================================================================

class BOSuggestor:
    """
    贝叶斯优化建议器 — 顶层接口。

    组合 ParameterEncoder + GPSurrogate + EIAcquisition，
    提供与 BacktestEngine 兼容的接口。

    用法:
      bo = BOSuggestor(PARAM_SEARCH_SPACE)
      bo.warmup(n_random=15)  # 生成初始随机点

      # 替代 _generate_combo:
      x = bo.suggest()              # 连续向量
      params = bo.encoder.decode(x)  # → 参数字典
      # ... evaluate_combo(params)...  #
      y = result['avg_total_hits']   # 得分
      bo.update(x, y)                # 反馈

      # 或一步到位:
      params = bo.suggest_and_decode()
      # ... evaluate ...
      bo.update(params, y)
    """

    def __init__(self, param_search_space: Dict[str, Dict],
                 objective_type: str = 'avg_hits',
                 kernel_type: str = 'rbf',
                 xi: float = 0.01,
                 batch_size: int = 1):
        """
        参数:
          param_search_space: PARAM_SEARCH_SPACE
          objective_type: 'avg_hits'（最大化）或 'closeness'（最大化）
          kernel_type: 'rbf' 或 'matern'
          xi: EI 探索系数
          batch_size: 批量并行建议数
        """
        self.encoder = ParameterEncoder(param_search_space)
        self.objective_type = objective_type

        self.gp = GPSurrogate(
            n_dimensions=self.encoder.n_free,
            kernel_type=kernel_type
        )
        self.ei = EIAcquisition(xi=xi)
        self.bounds = self.encoder.get_bounds()
        self.batch_size = batch_size

        self.rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        self.n_evaluated = 0
        self.warmup_size = max(10, self.encoder.n_free * 2)  # 至少 2×维度
        self.warmup_done = False

        # 批量管理
        self._pending_batch = []
        self._batch_idx = 0

        # 统计
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'best_score': -np.inf,
            'warmup_phase': True,
        }

    # ------------------------------------------------------------------
    #  热启动
    # ------------------------------------------------------------------

    def warmup(self, n_random: int = None) -> List[np.ndarray]:
        """
        生成初始随机评估点。

        返回:
          随机采样的连续向量列表
        """
        if n_random is None:
            n_random = self.warmup_size

        points = []
        for _ in range(n_random):
            points.append(self.encoder.random_sample(self.rng))

        return points

    # ------------------------------------------------------------------
    #  建议
    # ------------------------------------------------------------------

    def suggest(self) -> np.ndarray:
        """
        建议下一个评估点。

        热启动阶段: 随机采样
        BO 阶段: EI 采集函数选点

        返回:
          连续向量 x
        """
        self.stats['n_suggestions'] += 1

        # 热启动：前 warmup_size 个点随机
        if self.n_evaluated < self.warmup_size:
            x = self.encoder.random_sample(self.rng)
            return x

        self.warmup_done = True
        self.stats['warmup_phase'] = False

        # 批量并行管理
        if self.batch_size > 1:
            if not self._pending_batch or self._batch_idx >= len(self._pending_batch):
                # 生成新批次
                self._pending_batch = self.ei.suggest_batch(
                    self.gp, self.bounds, self.batch_size,
                    rng=self.rng
                )
                self._batch_idx = 0

            x = self._pending_batch[self._batch_idx]
            self._batch_idx += 1
            return x
        else:
            # 单点建议
            return self.ei.suggest(self.gp, self.bounds, rng=self.rng)

    def suggest_and_decode(self) -> Dict[str, Dict]:
        """建议下一个点并解码为参数字典"""
        x = self.suggest()
        return self.encoder.decode(x)

    # ------------------------------------------------------------------
    #  更新
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray, y: float):
        """
        用评估结果更新 GP 模型。

        参数:
          x: 连续向量
          y: 得分（越大越好）
        """
        self.n_evaluated += 1
        self.stats['n_updates'] += 1

        # 更新最优
        if y > self.stats['best_score']:
            self.stats['best_score'] = y

        # 更新 EI 的最优值
        if y > self.ei.y_best:
            self.ei.y_best = y

        # 更新 GP
        self.gp.update(x, y)

    def update_by_params(self, params: Dict[str, Dict], y: float):
        """用参数字典直接更新"""
        x = self.encoder.encode(params)
        self.update(x, y)

    # ------------------------------------------------------------------
    #  状态查询
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """BO 是否已完成热启动，可以产生有意义的建议"""
        return self.n_evaluated >= self.warmup_size and self.gp._fitted

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = dict(self.stats)
        stats['n_evaluated'] = self.n_evaluated
        stats['warmup_remaining'] = max(0, self.warmup_size - self.n_evaluated)
        stats['n_dimensions'] = self.encoder.n_free
        stats['n_total_params'] = self.encoder.n_total
        stats['kernel_type'] = self.gp.gp.kernel.__class__.__name__ \
            if hasattr(self.gp, 'gp') else 'unknown'
        return stats

    def reset(self):
        """重置 BO 状态"""
        self.gp = GPSurrogate(n_dimensions=self.encoder.n_free)
        self.ei = EIAcquisition(xi=self.ei.xi)
        self.n_evaluated = 0
        self.warmup_done = False
        self._pending_batch = []
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'best_score': -np.inf,
            'warmup_phase': True,
        }


# ============================================================================
#  与 BacktestEngine 的集成适配器
# ============================================================================

class BOBridge:
    """
    BO 与 BacktestEngine 之间的适配器。

    维护权重采样器（用于回测模式中权重与参数一起优化），
    以及线性权重求解器接口（用于求解模式）。
    """

    def __init__(self, param_search_space: Dict,
                 weight_search_space: Dict = None,
                 mode: str = 'backtest'):
        """
        参数:
          param_search_space: PARAM_SEARCH_SPACE
          weight_search_space: WEIGHT_SEARCH_SPACE
          mode: 'backtest'（参数+权重一体优化）
                或 'solve'（只优化参数，权重线性求解）
        """
        self.mode = mode
        self.bo = BOSuggestor(param_search_space,
                               objective_type='avg_hits' if mode == 'backtest' else 'closeness')

        self.weight_space = weight_search_space or {}
        self._weight_rng = np.random.RandomState()

    def suggest_combo(self) -> Tuple[Dict, Dict, str]:
        """
        建议一组 (params, weights)。

        替代 BacktestEngine._generate_combo()。

        返回:
          (params_dict, weights_dict, phase_label)
        """
        if self.mode == 'backtest':
            # 参数 + 权重一起优化
            params = self.bo.suggest_and_decode()
            weights = self._sample_weights()
            phase = 'bo_warmup' if not self.bo.is_ready() else 'bo_active'
            return params, weights, phase
        else:
            # 求解模式：只优化参数，权重留待线性求解
            params = self.bo.suggest_and_decode()
            # 返回等权（求解模式中权重会被线性求解器覆盖）
            weights = self._default_weights()
            phase = 'bo_warmup' if not self.bo.is_ready() else 'bo_active'
            return params, weights, phase

    def update(self, params: Dict, score: float):
        """用评估结果更新 BO"""
        self.bo.update_by_params(params, score)

    def _sample_weights(self) -> Dict:
        """采样合并权重"""
        method_range = self.weight_space.get(
            'method_weight_range', (0.3, 3.0))
        gran_range = self.weight_space.get(
            'granularity_weight_range', (0.3, 3.0))

        method_weights = {}
        for i in range(1, 14):
            method_weights[f'method_{i}'] = round(
                self._weight_rng.uniform(*method_range), 4)

        gran_weights = {
            '50期': round(self._weight_rng.uniform(*gran_range), 4),
            '100期': round(self._weight_rng.uniform(*gran_range), 4),
            '500期': round(self._weight_rng.uniform(*gran_range), 4),
            '1000期': round(self._weight_rng.uniform(*gran_range), 4),
            '全部期': round(self._weight_rng.uniform(*gran_range), 4),
        }

        return {
            'method_weights': method_weights,
            'granularity_weights': gran_weights,
        }

    def _default_weights(self) -> Dict:
        """返回等权权重"""
        method_weights = {f'method_{i}': 1.0 for i in range(1, 14)}
        gran_weights = {gk: 1.0 for gk in
                        ['50期', '100期', '500期', '1000期', '全部期']}
        return {
            'method_weights': method_weights,
            'granularity_weights': gran_weights,
        }

    def get_stats(self) -> Dict:
        return self.bo.get_stats()


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  贝叶斯优化代理模型 测试")
    print("=" * 60)

    # 模拟 PARAM_SEARCH_SPACE
    test_space = {
        'model_a': {
            'param_1': [1, 2, 3, 5, 8],
            'param_2': [0.1, 0.5, 1.0],
        },
        'model_b': {
            'param_3': [10, 20, 50],
        },
    }

    # 测试 1: 编解码器
    print("\n[测试1] 参数编解码")
    encoder = ParameterEncoder(test_space)
    print(f"  总参数: {encoder.n_total}, 自由参数: {encoder.n_free}")

    params = {'model_a': {'param_1': 5, 'param_2': 0.5},
              'model_b': {'param_3': 20}}
    x = encoder.encode(params)
    print(f"  编码: {params} → {x}")

    decoded = encoder.decode(x)
    print(f"  解码: {x} → {decoded}")
    for mk in params:
        for pk in params[mk]:
            assert params[mk][pk] == decoded[mk][pk], \
                f"Mismatch: {mk}.{pk}"
    print("  [OK] 编解码一致")

    # 测试 2: GP 代理模型
    print("\n[测试2] GP 代理模型")
    gp = GPSurrogate(n_dimensions=encoder.n_free, kernel_type='rbf')

    # 生成一些模拟数据
    rng = np.random.RandomState(42)
    X_train = rng.uniform(0, 1, (20, encoder.n_free))
    # 模拟目标函数（带噪声的正弦）
    y_train = np.sin(X_train[:, 0] * np.pi * 3) + \
        np.cos(X_train[:, 1] * np.pi * 2) * 0.5 + \
        rng.normal(0, 0.02, 20)
    y_train = np.maximum(0, y_train)  # 命中数≥0

    gp.fit(X_train, y_train)

    X_test = rng.uniform(0, 1, (5, encoder.n_free))
    mu, sigma = gp.predict(X_test)
    print(f"  测试预测: mu={mu[:3]}, sigma={sigma[:3]}")
    print("  [OK] GP 预测正常")

    # 测试 3: EI 采集函数
    print("\n[测试3] EI 采集函数")
    acqui = EIAcquisition(xi=0.01)
    acqui.set_best(np.max(y_train))

    bounds = np.array([(0.0, 1.0)] * encoder.n_free)
    x_best = acqui.suggest(gp, bounds, n_candidates=500)
    print(f"  EI 建议点: {x_best}")

    # 批量建议
    batch = acqui.suggest_batch(gp, bounds, batch_size=3, n_candidates=500)
    print(f"  批量建议: {len(batch)} 个点")
    print("  [OK] EI 采集正常工作")

    # 测试 4: 顶层 BOSuggestor
    print("\n[测试4] BOSuggestor 完整流程")
    bo = BOSuggestor(test_space, objective_type='avg_hits')

    for i in range(30):
        x_suggest = bo.suggest()
        params_suggest = bo.encoder.decode(x_suggest)
        # 模拟评估
        y = np.sin(x_suggest[0] * np.pi * 3) + \
            np.cos(x_suggest[1] * np.pi * 2) * 0.5 + \
            np.random.normal(0, 0.05)
        y = max(0, y)
        bo.update(x_suggest, y)

        if i == 14:
            print(f"  热启动完成 (第{i+1}次), 进入BO阶段")

    print(f"  评估次数: {bo.n_evaluated}")
    print(f"  最优得分: {bo.stats['best_score']:.4f}")
    print("  [OK] BO 完整流程正常")

    print("\n[OK] 贝叶斯优化代理模型测试全部通过")
