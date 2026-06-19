"""
模拟退火多链采样器（Simulated Annealing Multi-Chain Sampler）
============================================================
多链并行模拟退火，每条链在参数空间中独立探索。

核心机制:
  每条链维护当前状态 (current_x, current_score)
  每步: 随机扰动当前状态 → 评估新状态 → 根据 Metropolis 准则接受/拒绝
  接受准则: P(accept) = 1 if better, exp(ΔE/T) if worse
  温度 T 按指数衰减: T_k = T_0 * alpha^k

多链并行:
  N 条独立链同时运行，每步各链交替提议下一个评估点。
  天然适合 ThreadPoolExecutor 并发评估。

离散参数处理:
  扰动 = 随机选几个参数 → 随机改到不同于当前值的候选值

优势:
  - 极简实现，零额外依赖
  - 天然处理离散参数（无需连续松弛）
  - 理论全局收敛（无限时间下）
  - 多链天然支持并行

用法:
  sa = SASampler(param_search_space, n_chains=4)
  x = sa.ask()              # 从某条链建议下一个评估点
  sa.tell(x, score)         # 反馈评估结果
  params = sa.decode(x)     # 转回参数字典
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================================
#  模拟退火采样器
# ============================================================================

class SASampler:
    """
    多链并行模拟退火。

    参数:
      param_search_space: PARAM_SEARCH_SPACE
      n_chains: 并行链数（建议 = 并行线程数）
      T_max: 初始温度
      T_min: 最低温度（低于此值停止降温）
      cooling_rate: 冷却率（每步温度乘以此值，0.90~0.99）
      steps_per_chain: 每条链的最大步数（None=不限）
    """

    def __init__(self, param_search_space: Dict[str, Dict],
                 n_chains: int = 4,
                 T_max: float = 1.0,
                 T_min: float = 0.001,
                 cooling_rate: float = 0.95,
                 steps_per_chain: int = None):
        self.space = param_search_space
        self._build_param_list()

        self.n_chains = n_chains
        self.T_max = T_max
        self.T_min = T_min
        self.cooling_rate = cooling_rate
        self.steps_per_chain = steps_per_chain

        self.rng = np.random.RandomState(int(time.time() * 1000) % 10000)

        # 链状态: [{current_x, current_score, T, step, pending_x}]
        self.chains = []
        self._init_chains()

        # 轮询索引
        self._chain_idx = 0

        # 统计
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'n_accepted': 0,
            'n_rejected': 0,
            'best_score': -np.inf,
            'best_x': None,
            'chain_stats': [],
        }

    def _build_param_list(self):
        """构建参数列表（SA 直接操作离散值，不需要连续编码）"""
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

    def _init_chains(self):
        """初始化所有链"""
        self.chains = []
        for i in range(self.n_chains):
            # 随机初始点
            x = self._random_params()
            chain = {
                'id': i,
                'current_x': x,
                'current_score': -np.inf,
                'T': self.T_max,
                'step': 0,
                'pending_x': None,
                'pending_score': None,
                'n_accepted': 0,
                'n_rejected': 0,
            }
            self.chains.append(chain)

    # ------------------------------------------------------------------
    #  参数操作
    # ------------------------------------------------------------------

    def _random_params(self) -> Dict[str, Dict]:
        """随机生成一组参数"""
        params = {}
        for method_name, pname, pvalues in self.free_params:
            if method_name not in params:
                params[method_name] = {}
            params[method_name][pname] = pvalues[
                self.rng.randint(0, len(pvalues))]
        # 固定参数
        for method_name, pname, pvalues in self.param_list:
            if len(pvalues) == 1:
                if method_name not in params:
                    params[method_name] = {}
                params[method_name][pname] = pvalues[0]
        return params

    def _perturb_params(self, params: Dict) -> Dict:
        """
        随机扰动参数（直接操作离散值）。

        扰动策略:
          随机选 1~3 个自由参数，将它们的值改为不同的候选值。
        """
        import copy
        new_params = copy.deepcopy(params)

        # 随机选 1~3 个参数扰动
        n_perturb = self.rng.randint(1, min(4, self.n_free + 1))
        perturb_indices = self.rng.choice(
            self.n_free, n_perturb, replace=False)

        for idx in perturb_indices:
            method_name, pname, pvalues = self.free_params[idx]
            current = new_params.get(method_name, {}).get(pname)
            candidates = [v for v in pvalues if v != current]
            if candidates:
                if method_name not in new_params:
                    new_params[method_name] = {}
                new_params[method_name][pname] = candidates[
                    self.rng.randint(0, len(candidates))]

        return new_params

    def encode(self, params: Dict) -> np.ndarray:
        """参数字典 → 连续向量 [0,1]ⁿ（用于外部接口兼容）"""
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
        """连续向量 → 参数字典"""
        params = {}
        x_clipped = np.clip(x, 0.0, 1.0)
        for i, (method_name, pname, pvalues) in enumerate(self.free_params):
            if method_name not in params:
                params[method_name] = {}
            idx = int(round(x_clipped[i] * (len(pvalues) - 1)))
            idx = max(0, min(idx, len(pvalues) - 1))
            params[method_name][pname] = pvalues[idx]
        for method_name, pname, pvalues in self.param_list:
            if len(pvalues) == 1:
                if method_name not in params:
                    params[method_name] = {}
                params[method_name][pname] = pvalues[0]
        return params

    # ------------------------------------------------------------------
    #  建议 + 更新
    # ------------------------------------------------------------------

    def ask(self) -> Dict[str, Dict]:
        """
        从下一条活动链建议一个评估点。

        返回:
          参数字典（可直接传给 evaluate_combo）

        策略:
          轮询各链，每链交替提议。
        """
        self.stats['n_suggestions'] += 1

        # 找一条有未完成评估的链
        for _ in range(self.n_chains):
            chain = self.chains[self._chain_idx]
            self._chain_idx = (self._chain_idx + 1) % self.n_chains

            # 检查是否已达到步数上限
            if self.steps_per_chain and chain['step'] >= self.steps_per_chain:
                continue

            # 检查是否已冷却到最低温
            if chain['T'] < self.T_min:
                continue

            # 该链有未完成的评估 → 返回之前提议的点
            if chain['pending_x'] is not None:
                continue

            # 生成新提议
            if chain['step'] == 0:
                # 第一步：使用初始随机点
                new_params = chain['current_x']
            else:
                # 后续步：扰动当前状态
                new_params = self._perturb_params(chain['current_x'])

            chain['pending_x'] = new_params
            return new_params

        # 所有链都在忙碌或已终止 → 随机生成
        return self._random_params()

    def ask_batch(self, n: int = None) -> List[Dict[str, Dict]]:
        """批量建议多个点"""
        if n is None:
            n = self.n_chains
        return [self.ask() for _ in range(n)]

    def tell(self, params: Dict, score: float):
        """
        反馈评估结果给对应的链。

        Metropolis 准则:
          如果 score > current_score → 接受（总是）
          否则 → 以概率 exp((score - current) / T) 接受
        """
        self.stats['n_updates'] += 1

        if score > self.stats['best_score']:
            self.stats['best_score'] = score
            self.stats['best_x'] = {
                mk: dict(mp) for mk, mp in params.items()
            }

        # 找到匹配的链
        for chain in self.chains:
            if chain['pending_x'] is not None and self._params_equal(
                    params, chain['pending_x']):
                chain['pending_score'] = score

                # Metropolis 准则
                delta = score - chain['current_score']
                if delta > 0 or (
                    chain['T'] > 1e-10 and
                    self.rng.random() < np.exp(delta / max(chain['T'], 1e-10))
                ):
                    # 接受
                    chain['current_x'] = chain['pending_x']
                    chain['current_score'] = score
                    chain['n_accepted'] += 1
                    self.stats['n_accepted'] += 1
                else:
                    # 拒绝（保留 current_x）
                    chain['n_rejected'] += 1
                    self.stats['n_rejected'] += 1

                chain['pending_x'] = None
                chain['step'] += 1

                # 降温
                chain['T'] *= self.cooling_rate
                break

    def tell_batch(self, params_list: List[Dict], scores: List[float]):
        """批量反馈"""
        for p, s in zip(params_list, scores):
            self.tell(p, s)

    # ------------------------------------------------------------------
    #  状态查询
    # ------------------------------------------------------------------

    def _params_equal(self, p1: Dict, p2: Dict) -> bool:
        """简单比较两个参数字典"""
        try:
            for mk in set(list(p1.keys()) + list(p2.keys())):
                d1 = p1.get(mk, {})
                d2 = p2.get(mk, {})
                if set(d1.keys()) != set(d2.keys()):
                    return False
                for pk in d1:
                    if d1[pk] != d2[pk]:
                        return False
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        """是否可产生有意义的建议（始终可用）"""
        return self.stats['n_updates'] > 0

    def is_active(self) -> bool:
        """是否还有活动链（未全部冷却）"""
        return any(
            chain['T'] >= self.T_min and
            (self.steps_per_chain is None or chain['step'] < self.steps_per_chain)
            for chain in self.chains
        )

    def get_stats(self) -> Dict:
        stats = dict(self.stats)
        stats['n_active_chains'] = sum(
            1 for c in self.chains
            if c['T'] >= self.T_min and
               (self.steps_per_chain is None or c['step'] < self.steps_per_chain))
        stats['chain_temperatures'] = [round(c['T'], 4) for c in self.chains]
        stats['chain_steps'] = [c['step'] for c in self.chains]
        stats['acceptance_rate'] = (
            stats['n_accepted'] / max(1, stats['n_accepted'] + stats['n_rejected'])
        )
        return stats

    def reset(self):
        """重置所有链"""
        self._init_chains()
        self.stats = {
            'n_suggestions': 0,
            'n_updates': 0,
            'n_accepted': 0,
            'n_rejected': 0,
            'best_score': -np.inf,
            'best_x': None,
            'chain_stats': [],
        }


# ============================================================================
#  与 BacktestEngine 的适配
# ============================================================================

class SABridge:
    """
    模拟退火与 BacktestEngine 的适配器。

    用法:
      bridge = SABridge(PARAM_SEARCH_SPACE, n_chains=4)
      params = bridge.suggest_and_decode()
      # ... evaluate ...
      bridge.update(params, score)
    """

    def __init__(self, param_search_space: Dict[str, Dict],
                 n_chains: int = 4,
                 T_max: float = 1.0,
                 cooling_rate: float = 0.95):
        self.sa = SASampler(param_search_space, n_chains, T_max,
                            cooling_rate=cooling_rate)
        self._weight_rng = np.random.RandomState()

    def suggest_combo(self) -> Tuple[Dict, Dict, str]:
        """建议一组 (params, weights)"""
        params = self.sa.ask()
        weights = self._sample_weights()
        # 确定阶段标签
        phase = 'sa_active' if self.sa.is_active() else 'sa_cooled'
        return params, weights, phase

    def update(self, params: Dict, score: float):
        self.sa.tell(params, score)

    def _sample_weights(self) -> Dict:
        method_weights = {}
        for i in range(1, 14):
            method_weights[f'method_{i}'] = round(
                self._weight_rng.uniform(0.3, 3.0), 4)
        gran_weights = {
            '50期': round(self._weight_rng.uniform(0.3, 3.0), 4),
            '100期': round(self._weight_rng.uniform(0.3, 3.0), 4),
            '500期': round(self._weight_rng.uniform(0.3, 3.0), 4),
            '1000期': round(self._weight_rng.uniform(0.3, 3.0), 4),
            '全部期': round(self._weight_rng.uniform(0.3, 3.0), 4),
        }
        return {'method_weights': method_weights, 'granularity_weights': gran_weights}

    def get_stats(self) -> Dict:
        return self.sa.get_stats()


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  模拟退火多链采样器 测试")
    print("=" * 60)

    test_space = {
        'model_a': {'p1': [1, 2, 3, 5, 8], 'p2': [0.1, 0.5, 1.0]},
        'model_b': {'p3': [10, 20, 50]},
    }

    sa = SASampler(test_space, n_chains=3, T_max=2.0,
                   cooling_rate=0.90, steps_per_chain=30)
    print(f"自由参数: {sa.n_free}, 链数: {sa.n_chains}")

    # 模拟 90 次评估
    scores_record = []
    for i in range(90):
        params = sa.ask()

        # 模拟评估（基于参数值的简单函数）
        p1_val = params.get('model_a', {}).get('p1', 0)
        p2_val = params.get('model_a', {}).get('p2', 0)
        p3_val = params.get('model_b', {}).get('p3', 0)

        # 越接近 (p1=5, p2=0.5, p3=20) 得分越高
        score = -abs(p1_val - 5) / 10 - abs(p2_val - 0.5) * 3 - abs(p3_val - 20) / 100
        score = max(0, 1 + score) + np.random.normal(0, 0.05)
        scores_record.append(score)

        sa.tell(params, score)

        if (i + 1) % 20 == 0:
            stats = sa.get_stats()
            print(f"\n  评估 {i+1}/90:")
            print(f"    最优得分: {stats['best_score']:.4f}")
            print(f"    接受率: {stats['acceptance_rate']:.2%}")
            print(f"    活动链: {stats['n_active_chains']}/3")
            print(f"    链温度: {stats['chain_temperatures']}")
            print(f"    链步数: {stats['chain_steps']}")

    stats = sa.get_stats()
    print(f"\n[OK] 模拟退火测试完成")
    print(f"  统计: {stats}")
    if stats['best_x']:
        print(f"  最优参数: {stats['best_x']}")
