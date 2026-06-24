"""
CMA-ES 协方差矩阵自适应进化采样器
==================================
实现 μ/μ_w,λ CMA-ES，用多元正态分布引导参数搜索。

核心机制:
  每代从 N(mean, sigma^2 * C) 采样 λ 个候选点
  评估后选最优 μ 个更新分布:
    - mean → 加权平均（最优μ个点的重心）
    - C → 协方差矩阵自适应（学习搜索方向）
    - sigma → 步长自适应（进化路径控制）

离散参数处理:
  连续松弛（同BO）：离散候选值索引 → [0,1] 连续区间 → 舍入解码

参考: Hansen (2006) "The CMA Evolution Strategy: A Tutorial"

用法:
  cmaes = CMAESSampler(param_search_space, population_size=12)
  x = cmaes.ask()              # 建议下一个评估点
  cmaes.tell(x, score)         # 反馈评估结果
  params = cmaes.decode(x)     # 转回参数字典
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
#  CMA-ES 采样器
# ============================================================================

class CMAESSampler:
    """
    μ/μ_w,λ CMA-ES。

    参数:
      param_search_space: PARAM_SEARCH_SPACE
      population_size: λ（每代采样数），默认 4 + 3*ln(dim)
      initial_sigma: 初始步长
      max_generations: 最大代数（无限制时不停采样）
    """

    def __init__(self, param_search_space: Dict[str, Dict],
                 population_size: int = None,
                 initial_sigma: float = 0.3,
                 max_generations: int = None):
        self.space = param_search_space
        self._build_param_list()

        self.dim = self.n_free
        if self.dim == 0:
            raise ValueError("参数空间无自由参数")

        # 种群大小
        self.lam = population_size or (4 + int(3 * np.log(self.dim)))
        self.mu = self.lam // 2  # 用于更新的父代数量

        # 权重（μ个父代，排序后越靠前权重越大）
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()

        # 有效种群大小
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # 策略参数
        self.sigma = initial_sigma

        # 均值（初始在中心）
        self.mean = np.full(self.dim, 0.5)

        # 协方差矩阵
        self.C = np.eye(self.dim)

        # 进化路径
        self.pc = np.zeros(self.dim)  # C的进化路径
        self.ps = np.zeros(self.dim)  # sigma的进化路径

        # 学习率
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                       ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs

        # 期望值
        self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # 状态
        self.generation = 0
        self.max_generations = max_generations
        self.population = []       # 当前代的候选点
        self.scores = []           # 对应的得分
        self._pop_idx = 0          # 当前代内索引
        self._generated_pop = False

        # 统计
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'n_generations': 0,
            'best_score': -np.inf,
            'sigma_history': [],
        }

        self.rng = np.random.RandomState(int(time.time() * 1000) % 10000)

    def _build_param_list(self):
        """构建参数列表"""
        self.free_params = []
        self.param_list = []
        for method_name in sorted(self.space.keys()):
            space = self.space[method_name]
            for pname in sorted(space.keys()):
                pvalues = space[pname]
                self.param_list.append((method_name, pname, pvalues))
                if len(pvalues) > 1:
                    self.free_params.append((method_name, pname, pvalues))
        self.n_total = len(self.param_list)
        self.n_free = len(self.free_params)

    # ------------------------------------------------------------------
    #  编解码
    # ------------------------------------------------------------------

    def decode(self, x: np.ndarray) -> Dict[str, Dict]:
        """连续向量 → 参数字典"""
        params = {}
        x_clipped = np.clip(x, 0.0, 1.0)
        for i, (method_name, pname, pvalues) in enumerate(self.free_params):
            if method_name not in params:
                params[method_name] = {}
            idx = int(round(x_clipped[i] * (len(pvalues) - 1)))
            idx = max(0, min(idx, len(pvalues) - 1))
            params[method_name][pname] = pvalues[idx]
        # 固定参数
        for method_name, pname, pvalues in self.param_list:
            if len(pvalues) == 1:
                if method_name not in params:
                    params[method_name] = {}
                params[method_name][pname] = pvalues[0]
        return params

    # ------------------------------------------------------------------
    #  采样
    # ------------------------------------------------------------------

    def _sample_population(self):
        """从当前分布采样一代"""
        # Cholesky分解 C = B D (B D)^T
        try:
            B, D_sq = np.linalg.eigh(self.C)
            D_sq = np.maximum(D_sq, 1e-10)
            D = np.sqrt(D_sq)
        except np.linalg.LinAlgError:
            B = np.eye(self.dim)
            D = np.ones(self.dim)

        self.population = []
        for _ in range(self.lam):
            z = self.rng.randn(self.dim)
            # y ~ N(0, C) = B D z
            y = B.dot(D * z)
            x = self.mean + self.sigma * y
            x = np.clip(x, 0.0, 1.0)
            self.population.append(x)

        self.scores = [None] * self.lam
        self._pop_idx = 0
        self._generated_pop = True
        self.generation += 1

    def ask(self) -> np.ndarray:
        """返回下一个评估点"""
        self.stats['n_suggestions'] += 1

        # 检查代数限制
        if self.max_generations and self.generation >= self.max_generations:
            # 超过代数限制，随机采样
            return self.rng.uniform(0, 1, self.dim)

        # 需要新的一代
        if not self._generated_pop or self._pop_idx >= self.lam:
            self._sample_population()

        x = self.population[self._pop_idx]
        return x

    def ask_batch(self) -> List[np.ndarray]:
        """返回整代的所有候选点（用于并行评估）"""
        if not self._generated_pop or self._pop_idx >= self.lam:
            self._sample_population()
        batch = self.population[self._pop_idx:]
        self._pop_idx = self.lam
        return batch

    # ------------------------------------------------------------------
    #  更新
    # ------------------------------------------------------------------

    def tell(self, x: np.ndarray, score: float):
        """
        反馈评估结果。
        当收集完一代所有点时，更新分布。
        """
        self.stats['n_updates'] += 1

        if score > self.stats['best_score']:
            self.stats['best_score'] = score

        if not self._generated_pop:
            return

        # 找到 x 在种群中的索引
        idx = None
        for i in range(self._pop_idx):
            if np.allclose(x, self.population[i], atol=1e-10):
                idx = i
                break
        if idx is None:
            idx = self._pop_idx
            self._pop_idx += 1

        self.scores[idx] = score

        # 一代完成？更新分布
        if all(s is not None for s in self.scores):
            self._update_distribution()

    def tell_batch(self, scores: List[float]):
        """批量反馈一整代的评估结果"""
        for i, s in enumerate(scores):
            self.scores[i] = s
        self._pop_idx = self.lam
        if all(s is not None for s in self.scores):
            self._update_distribution()

    def _update_distribution(self):
        """用当前代的评估结果更新 N(mean, sigma^2*C)"""
        # 按得分降序排列
        order = np.argsort(self.scores)[::-1]
        sorted_pop = [self.population[i] for i in order]

        # 前 μ 个点
        mu_best = sorted_pop[:self.mu]

        # 更新 mean → 加权平均
        old_mean = self.mean.copy()
        self.mean = np.zeros(self.dim)
        for i in range(self.mu):
            self.mean += self.weights[i] * mu_best[i]
        self.mean = np.clip(self.mean, 0.0, 1.0)

        # 更新进化路径和协方差
        y_mean = (self.mean - old_mean) / self.sigma
        inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim)))
        inv_sqrt_C_y = inv_sqrt_C.dot(y_mean)

        # ps (sigma进化路径)
        self.ps = (1 - self.cs) * self.ps + \
            np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * inv_sqrt_C_y

        # sigma 更新
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 0.01, 0.5)

        # pc (C进化路径)
        h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < \
            (1.4 + 2 / (self.dim + 1)) * self.chi_n else 0.0

        self.pc = (1 - self.cc) * self.pc + \
            h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_mean

        # C 更新
        C_update = self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]
        C_rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y_i = (mu_best[i] - old_mean) / self.sigma
            C_rank_mu += self.weights[i] * y_i[:, np.newaxis] @ y_i[np.newaxis, :]

        self.C = (1 - self.c1 - self.cmu) * self.C + \
            self.c1 * C_update + self.cmu * C_rank_mu

        # 强制对称正定
        self.C = (self.C + self.C.T) / 2
        eigvals = np.linalg.eigvalsh(self.C)
        if eigvals.min() < 1e-10:
            self.C += np.eye(self.dim) * 1e-6

        self.stats['n_generations'] += 1
        self.stats['sigma_history'].append(float(self.sigma))

        # 准备下一代
        self._generated_pop = False

    # ------------------------------------------------------------------
    #  状态查询
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """是否已有足够评估来产生有意义的建议"""
        return self.stats['n_generations'] >= 1

    def get_stats(self) -> Dict:
        stats = dict(self.stats)
        stats['dim'] = self.dim
        stats['population_size'] = self.lam
        stats['parent_size'] = self.mu
        stats['current_sigma'] = float(self.sigma)
        stats['current_generation'] = self.generation
        return stats

    def reset(self):
        """重置采样器"""
        self.mean = np.full(self.dim, 0.5)
        self.C = np.eye(self.dim)
        self.sigma = 0.3
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.generation = 0
        self.population = []
        self.scores = []
        self._pop_idx = 0
        self._generated_pop = False
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'n_generations': 0,
            'best_score': -np.inf,
            'sigma_history': [],
        }


# ============================================================================
#  与 BacktestEngine 的适配
# ============================================================================

class CMAESBridge:
    """
    CMA-ES 与 BacktestEngine 的适配器。

    用法:
      bridge = CMAESBridge(PARAM_SEARCH_SPACE)
      params = bridge.suggest_and_decode()
      # ... evaluate ...
      bridge.update(params, score)
    """

    def __init__(self, param_search_space: Dict[str, Dict],
                 population_size: int = None,
                 initial_sigma: float = 0.3):
        self.cmaes = CMAESSampler(param_search_space, population_size, initial_sigma)
        self._weight_rng = np.random.RandomState()
        self._pending_x = None  # 跟踪当前建议点以匹配 tell

    def suggest_combo(self) -> Tuple[Dict, Dict, str]:
        """建议一组 (params, weights)，与 _generate_combo 接口兼容"""
        x = self.cmaes.ask()
        self._pending_x = x
        params = self.cmaes.decode(x)
        weights = self._sample_weights()
        phase = 'cmaes_pop' if self.cmaes._pop_idx < self.cmaes.lam else 'cmaes_update'
        return params, weights, phase

    def update(self, params: Dict, score: float):
        """用评估结果更新 CMA-ES"""
        if self._pending_x is not None:
            self.cmaes.tell(self._pending_x, score)
            self._pending_x = None

    def _sample_weights(self) -> Dict:
        """随机采样 65 个独立 composite_weights"""
        composite_weights = {}
        for i in range(1, 14):
            for gn in ['50期', '100期', '500期', '1000期', '全部期']:
                key = f'method_{i}@{gn}'
                composite_weights[key] = round(
                    self._weight_rng.uniform(-10000.0, 10000.0), 4)
        return {'composite_weights': composite_weights}

    def get_stats(self) -> Dict:
        return self.cmaes.get_stats()


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CMA-ES 采样器 测试")
    print("=" * 60)

    # 模拟参数空间
    test_space = {
        'model_a': {'p1': [1, 2, 3, 5, 8], 'p2': [0.1, 0.5, 1.0]},
        'model_b': {'p3': [10, 20, 50]},
    }

    cmaes = CMAESSampler(test_space, population_size=8)
    print(f"维度: {cmaes.dim}, 种群: {cmaes.lam}, 父代: {cmaes.mu}")

    # 运行 3 代
    for gen in range(3):
        print(f"\n第 {gen+1} 代: mean={cmaes.mean}, sigma={cmaes.sigma:.4f}")

        batch = cmaes.ask_batch()
        print(f"  采样 {len(batch)} 个点")

        # 模拟评估（越接近中心得分越高）
        scores = []
        for x in batch:
            dist = np.linalg.norm(x - np.array([0.5] * cmaes.dim))
            score = max(0, 1.0 - dist) + np.random.normal(0, 0.05)
            scores.append(score)

        cmaes.tell_batch(scores)
        best = max(scores)
        print(f"  最优得分: {best:.4f}")
        print(f"  解码示例: {cmaes.decode(batch[0])}")

    print(f"\n[OK] CMA-ES 测试完成")
    print(f"  统计: {cmaes.get_stats()}")
