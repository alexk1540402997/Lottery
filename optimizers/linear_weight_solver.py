"""
线性权重求解器 5.0
==================
对于求解模式，将权重优化从离散搜索中分离出来，
用线性规划 (LP) 或最小二乘 (LSTSQ) 精确求解最优合并权重。

核心原理:
  合并层投票运算是线性的：
    v_n = SUM_j w_j × I[号码n被组合j预测]
    其中 v_n 是号码n的得票，w_j 是组合j的权重

  65个(方法 × 颗粒度)组合各有独立权重 w_j，允许正/负/零。

用法:
  solver = LinearWeightSolver('ssq')
  solution = solver.solve_lp_multi_period(predictions_by_period, actuals_by_period)
  # solution = {'composite_weights': {'method_1@500期': 1.5, ...}, ...}

依赖:
  scipy.optimize.linprog, numpy.linalg.lstsq
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ============================================================================
#  常量
# ============================================================================

METHOD_KEYS = [f'method_{i}' for i in range(1, 14)]

GRANULARITY_NAMES = ['50期', '100期', '500期', '1000期', '全部期']
GRANULARITY_VALUES = [50, 100, 500, 1000, 0]


# ============================================================================
#  线性权重求解器
# ============================================================================

class LinearWeightSolver:
    """
    求解 65 个独立 composite_weights。

    问题形式化:
      X: (N个候选号码) × (M个方法×颗粒度组合) 的0/1矩阵
         X[n, j] = 1 表示组合j预测了号码n
      t: 目标向量，t[n] = 1 若n是实际开奖号码，否则 0
      w: 待求解的权重向量 (允许正/负/零，范围 [-500.0, 500.0])

    两种方法:
      LP:  min 0  s.t. (X[b,:]-X[a,:])·w <= -ε  ∀ 实际a, 非实际b
           保证排序正确（若有可行解）
      LSTSQ: min ||X w - t||^2
             LP不可行时的降级方案
    """

    COMPOSITE_WEIGHT_MAX = 10000.0  # 权重绝对值上限（LP用，近乎无界）

    def __init__(self, lottery_type: str = 'ssq'):
        self.lottery_type = lottery_type.lower()
        if self.lottery_type == 'ssq':
            self.main_name = 'red'
            self.aux_name = 'blue'
            self.main_count = 6
            self.aux_count = 1
            self.main_range = (1, 33)
            self.aux_range = (1, 16)
        else:
            self.main_name = 'front'
            self.aux_name = 'back'
            self.main_count = 5
            self.aux_count = 2
            self.main_range = (1, 35)
            self.aux_range = (1, 12)

        self.n_main_candidates = self.main_range[1] - self.main_range[0] + 1
        self.n_aux_candidates = self.aux_range[1] - self.aux_range[0] + 1

    # ------------------------------------------------------------------
    #  构建线性系统
    # ------------------------------------------------------------------

    def _build_design_matrix(self,
                              per_group_predictions: List[Dict]
                              ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        构建设计矩阵 X。

        每行 = 一个候选号码
        每列 = 一个 (方法 × 颗粒度) 组合
        值 = 1 如果该组合预测了该号码，否则 0

        参数:
          per_group_predictions: [
            {
              'method_key': 'method_1',
              'granularity': '500期',
              'predicted_main': [1, 5, 12, ...],
              'predicted_aux': [7],
            },
            ...
          ]

        返回:
          (X_main, X_aux, column_labels, row_labels)
        """
        column_labels = []
        for entry in per_group_predictions:
            label = f"{entry['method_key']}@{entry['granularity']}"
            column_labels.append(label)

        n_columns = len(per_group_predictions)
        n_main_rows = self.n_main_candidates
        n_aux_rows = self.n_aux_candidates

        X_main = np.zeros((n_main_rows, n_columns), dtype=int)
        X_aux = np.zeros((n_aux_rows, n_columns), dtype=int)

        main_row_labels = [
            str(n) for n in range(self.main_range[0], self.main_range[1] + 1)]
        aux_row_labels = [
            str(n) for n in range(self.aux_range[0], self.aux_range[1] + 1)]

        for j, entry in enumerate(per_group_predictions):
            for num in entry.get('predicted_main', []):
                if self.main_range[0] <= num <= self.main_range[1]:
                    row = num - self.main_range[0]
                    X_main[row, j] = 1
            for num in entry.get('predicted_aux', []):
                if self.aux_range[0] <= num <= self.aux_range[1]:
                    row = num - self.aux_range[0]
                    X_aux[row, j] = 1

        return X_main, X_aux, column_labels, main_row_labels, aux_row_labels

    def _build_target_vector(self,
                              actual_main: List[int],
                              actual_aux: List[int]
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建目标向量 t。

        t[n] = 1 若号码n是实际开奖号码，否则 0
        """
        t_main = np.zeros(self.n_main_candidates, dtype=int)
        t_aux = np.zeros(self.n_aux_candidates, dtype=int)

        for num in actual_main:
            if self.main_range[0] <= num <= self.main_range[1]:
                t_main[num - self.main_range[0]] = 1
        for num in actual_aux:
            if self.aux_range[0] <= num <= self.aux_range[1]:
                t_aux[num - self.aux_range[0]] = 1

        return t_main, t_aux

    # ------------------------------------------------------------------
    #  求解
    # ------------------------------------------------------------------

    def solve_single_period(self,
                             per_group_predictions: List[Dict],
                             actual_main: List[int],
                             actual_aux: List[int]
                             ) -> Dict[str, Any]:
        """
        对单期数据求解最优权重（最小二乘，无正负约束）。

        返回:
          {
            'composite_weights': {...},     # {label: weight} 字典
            'residual': float,              # LSTSQ 残差
            'actual_score': float,          # 预测的实际号码平均得票
            'column_labels': [...],
          }
        """
        X_main, X_aux, col_labels, _, _ = self._build_design_matrix(
            per_group_predictions)
        t_main, t_aux = self._build_target_vector(actual_main, actual_aux)

        # 合并主球和辅助球
        X = np.vstack([X_main, X_aux])
        t = np.concatenate([t_main, t_aux])

        if X.shape[0] == 0 or X.shape[1] == 0:
            return {'error': '设计矩阵为空', 'composite_weights': {}}

        # 检查是否有有效列（至少有一个非零元素）
        col_sums = X.sum(axis=0)
        valid_cols = col_sums > 0
        if not valid_cols.any():
            return {'error': '所有列均为零 — 预测结果全空',
                    'composite_weights': {}}

        # 如果只有部分列有效，只对有效列求解
        if not valid_cols.all():
            X_valid = X[:, valid_cols]
            valid_labels = [lbl for i, lbl in enumerate(col_labels) if valid_cols[i]]
        else:
            X_valid = X
            valid_labels = col_labels

        # 最小二乘求解（无正负约束）
        w_valid, residuals, rank, sv = np.linalg.lstsq(X_valid, t, rcond=None)
        residual = float(residuals[0]) if len(residuals) > 0 else 0.0

        # 映射回完整权重向量
        if not valid_cols.all():
            w_full = np.zeros(X.shape[1])
            valid_indices = np.where(valid_cols)[0]
            for i, wi in zip(valid_indices, w_valid):
                w_full[i] = wi
        else:
            w_full = w_valid

        # LSTSQ 无裁剪 — 允许任意大小的权重

        # 计算实际号码的预测得票
        if X.shape[1] > 0:
            predicted_scores = X.dot(w_full)
            actual_score = predicted_scores[t > 0].mean() if t.sum() > 0 else 0.0
        else:
            actual_score = 0.0

        # 构建 composite_weights 字典
        composite_weights = {
            col_labels[i]: round(float(w_full[i]), 6)
            for i in range(len(col_labels))
            if abs(w_full[i]) > 1e-10
        }

        return {
            'composite_weights': composite_weights,
            'residual': residual,
            'actual_score': float(actual_score),
            'column_labels': col_labels,
        }

    def solve_lp_multi_period(self,
    all_period_predictions: List[List[Dict]],
    all_period_actuals: List[Tuple[List[int], List[int]]],
    epsilon: float = 0.01,
    max_weight: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        用线性规划精确求解权重，使得实际号码得票严格高于非实际号码。

        问题形式化:
          对每个实际号码 a 和非实际号码 b:
            vote[a] - vote[b] >= epsilon
            → Σ_j (X[a,j] - X[b,j]) × w_j >= epsilon

          min  Σ_j w_j
          s.t. (X[b,:] - X[a,:]) · w <= -epsilon  (对所有 a∈actual, b∉actual)
               0 <= w_j <= max_weight

        相比 NNLS 的优势:
          NNLS 最小化均方误差，不保证排序正确。
          LP 直接约束"每个实际号码排前k"，保证验算100%通过（如果可行解存在）。

        参数:
          all_period_predictions: 每期的预测列表
          all_period_actuals: 每期的实际开奖号码
          epsilon: 最小胜出票差（防止平票），默认0.01
          max_weight: 单权重上限（防无穷大），默认5.0

        返回:
          同 solve_multi_period，额外包含 lp_status
        """
        from scipy.optimize import linprog

        if not all_period_predictions:
            return {'error': '无预测数据'}

        n_periods = len(all_period_predictions)

        # 收集所有列标签（确保各期对齐）
        all_col_labels = set()
        for period_preds in all_period_predictions:
            for entry in period_preds:
                all_col_labels.add(
                    f"{entry['method_key']}@{entry['granularity']}")
        col_labels = sorted(all_col_labels)
        n_cols = len(col_labels)

        if n_cols == 0:
            return {'error': '无有效预测组'}

        # 构建所有约束
        A_ub_rows = []  # 不等式约束矩阵
        b_ub_vals = []  # 不等式约束右侧

        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_predictions, all_period_actuals)):

            X_main, X_aux, p_col_labels, main_rows, aux_rows = \
                self._build_design_matrix(period_preds)

            # 对齐列
            X_main_aligned = np.zeros((X_main.shape[0], n_cols))
            X_aux_aligned = np.zeros((X_aux.shape[0], n_cols))
            for j_old, label in enumerate(p_col_labels):
                j_new = col_labels.index(label) if label in col_labels else -1
                if j_new >= 0:
                    X_main_aligned[:, j_new] = X_main[:, j_old]
                    X_aux_aligned[:, j_new] = X_aux[:, j_old]

            # ── 主球约束: 每个实际号码 vs 每个非实际号码 ──
            actual_main_indices = []
            for num in actual_main:
                if self.main_range[0] <= num <= self.main_range[1]:
                    actual_main_indices.append(num - self.main_range[0])

            non_actual_main_indices = [
                i for i in range(self.n_main_candidates)
                if i not in actual_main_indices
            ]

            for a_idx in actual_main_indices:
                for b_idx in non_actual_main_indices:
                    # 约束: (X[b,:] - X[a,:]) · w <= -epsilon
                    constraint_row = X_main_aligned[b_idx, :] - X_main_aligned[a_idx, :]
                    A_ub_rows.append(constraint_row)
                    b_ub_vals.append(-epsilon)

            # ── 辅助球约束: 每个实际号码 vs 每个非实际号码 ──
            actual_aux_indices = []
            for num in actual_aux:
                if self.aux_range[0] <= num <= self.aux_range[1]:
                    actual_aux_indices.append(num - self.aux_range[0])

            non_actual_aux_indices = [
                i for i in range(self.n_aux_candidates)
                if i not in actual_aux_indices
            ]

            for a_idx in actual_aux_indices:
                for b_idx in non_actual_aux_indices:
                    constraint_row = X_aux_aligned[b_idx, :] - X_aux_aligned[a_idx, :]
                    A_ub_rows.append(constraint_row)
                    b_ub_vals.append(-epsilon)

        if not A_ub_rows:
            return {'error': '无有效约束（实际号码为空或覆盖所有候选）'}

        A_ub = np.array(A_ub_rows, dtype=float)
        b_ub = np.array(b_ub_vals, dtype=float)

        # 目标函数: 微小正数 (倾向小权重，避免极端值，同时几乎不影响可行性)
        c = np.full(n_cols, 1e-6, dtype=float)

        # 边界: -max_weight <= w <= max_weight (允许负权重)
        bounds = [(-max_weight, max_weight) for _ in range(n_cols)]

        # 求解
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'maxiter': 5000},
        )

        if result.success:
            w_full = result.x
        else:
            # LP 不可行 → 用 LSTSQ 作为降级方案
            return self._lstsq_fallback(
                all_period_predictions, all_period_actuals,
                col_labels, lp_status=result.message)

        # 构建 composite_weights 字典（不再分解）
        composite_weights = {
            col_labels[i]: round(float(w_full[i]), 6)
            for i in range(n_cols)
            if abs(w_full[i]) > 1e-10
        }

        return {
            'composite_weights': composite_weights,
            'residual': 0.0,  # LP 精确求解，无残差
            'lp_success': True,
            'lp_status': result.message,
            'column_labels': col_labels,
            'n_periods': n_periods,
            'n_constraints': len(A_ub_rows),
        }

    def _lstsq_fallback(self,
                        all_period_predictions: List[List[Dict]],
                        all_period_actuals: List[Tuple[List[int], List[int]]],
                        col_labels: List[str],
                        lp_status: str = 'unknown'
                        ) -> Dict[str, Any]:
        """LP 不可行时的最小二乘降级方案，附带诊断信息"""
        n_periods = len(all_period_predictions)

        # 诊断: 找出未被任何方法覆盖的实际号码
        uncovered_main = set()
        uncovered_aux = set()

        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_predictions, all_period_actuals)):

            all_predicted_main = set()
            all_predicted_aux = set()
            for entry in period_preds:
                all_predicted_main.update(entry.get('predicted_main', []))
                all_predicted_aux.update(entry.get('predicted_aux', []))

            for a in actual_main:
                if a not in all_predicted_main:
                    uncovered_main.add(a)
            for a in actual_aux:
                if a not in all_predicted_aux:
                    uncovered_aux.add(a)

        # LSTSQ 求解（无正负约束）
        X_blocks = []
        t_blocks = []

        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_predictions, all_period_actuals)):

            X_main, X_aux, p_col_labels, _, _ = self._build_design_matrix(
                period_preds)
            t_main, t_aux = self._build_target_vector(actual_main, actual_aux)
            X_p = np.vstack([X_main, X_aux])
            t_p = np.concatenate([t_main, t_aux])

            X_p_aligned = np.zeros((X_p.shape[0], len(col_labels)))
            for j_old, label in enumerate(p_col_labels):
                j_new = col_labels.index(label) if label in col_labels else -1
                if j_new >= 0:
                    X_p_aligned[:, j_new] = X_p[:, j_old]

            X_blocks.append(X_p_aligned)
            t_blocks.append(t_p)

        X = np.vstack(X_blocks)
        t = np.concatenate(t_blocks)

        col_sums = X.sum(axis=0)
        valid_cols = col_sums > 0
        if not valid_cols.any():
            return {'error': '所有列均为零'}

        X_valid = X[:, valid_cols] if not valid_cols.all() else X
        w_valid, residuals, rank, sv = np.linalg.lstsq(X_valid, t, rcond=None)
        residual = float(residuals[0]) if len(residuals) > 0 else 0.0

        if not valid_cols.all():
            w_full = np.zeros(len(col_labels))
            valid_indices = np.where(valid_cols)[0]
            for i, wi in zip(valid_indices, w_valid):
                w_full[i] = wi
        else:
            w_full = w_valid

        # LSTSQ 无裁剪 — 允许任意大小的权重

        # 构建 composite_weights 字典
        composite_weights = {
            col_labels[i]: round(float(w_full[i]), 6)
            for i in range(len(col_labels))
            if abs(w_full[i]) > 1e-10
        }

        return {
            'composite_weights': composite_weights,
            'residual': residual,
            'lp_success': False,
            'lp_status': 'LP不可行(参数未覆盖实际号码) → 已降级为LSTSQ近似解',
            'lp_diagnostic': {
                'uncovered_main': sorted(list(uncovered_main)),
                'uncovered_aux': sorted(list(uncovered_aux)),
                'is_infeasible': True,
                'advice': ('以下实际号码未被任何方法预测到，'
                          '请增加回测运行时间以优化模型参数: '
                          + (f'主球={sorted(list(uncovered_main))} ' if uncovered_main else '')
                          + (f'辅助球={sorted(list(uncovered_aux))}' if uncovered_aux else '')) if (uncovered_main or uncovered_aux) else '',
            },
            'column_labels': col_labels,
            'n_periods': n_periods,
        }

    def solve_multi_period(self,
                            all_period_predictions: List[List[Dict]],
                            all_period_actuals: List[Tuple[List[int], List[int]]],
                            period_weights: Optional[List[float]] = None
                            ) -> Dict[str, Any]:
        """
        对多期数据联合求解一组通用权重。

        将所有周期的设计矩阵垂直堆叠，
        求解一组对所有周期都尽可能适用的权重。

        参数:
          all_period_predictions: 每期的预测列表
            [
              period_0: [{method_key, granularity, predicted_main, predicted_aux}, ...],
              period_1: [...],
            ]
          all_period_actuals: 每期的实际开奖号码
            [(main_list, aux_list), ...]
          period_weights: 各期的权重（默认等权），越近的期可以给更高权重

        返回:
          同 solve_single_period
        """
        from scipy.optimize import nnls  # 保留仅用于兼容，实际上下面用lstsq

        if not all_period_predictions:
            return {'error': '无预测数据'}

        n_periods = len(all_period_predictions)
        if period_weights is None:
            # 默认: 越近的期权重越高（指数衰减）
            decay = 0.9
            period_weights = [decay ** (n_periods - 1 - i)
                              for i in range(n_periods)]
            pw_sum = sum(period_weights)
            period_weights = [w / pw_sum * n_periods for w in period_weights]

        # 收集所有组合标签（确保各期对齐）
        all_col_labels = set()
        for period_preds in all_period_predictions:
            for entry in period_preds:
                all_col_labels.add(
                    f"{entry['method_key']}@{entry['granularity']}")
        col_labels = sorted(all_col_labels)

        # 构建聚合设计矩阵
        X_blocks = []
        t_blocks = []

        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_predictions, all_period_actuals)):

            X_main, X_aux, p_col_labels, _, _ = self._build_design_matrix(
                period_preds)
            t_main, t_aux = self._build_target_vector(
                actual_main, actual_aux)

            X_p = np.vstack([X_main, X_aux])
            t_p = np.concatenate([t_main, t_aux])

            # 对齐列
            X_p_aligned = np.zeros((X_p.shape[0], len(col_labels)))
            for j_old, label in enumerate(p_col_labels):
                j_new = col_labels.index(label) if label in col_labels else -1
                if j_new >= 0:
                    X_p_aligned[:, j_new] = X_p[:, j_old]

            # 应用周期权重
            pw = period_weights[p_idx]
            X_blocks.append(X_p_aligned * np.sqrt(pw))
            t_blocks.append(t_p * np.sqrt(pw))

        X = np.vstack(X_blocks)
        t = np.concatenate(t_blocks)

        if X.shape[0] == 0 or X.shape[1] == 0:
            return {'error': '设计矩阵为空'}

        # 移除全零列
        col_sums = X.sum(axis=0)
        valid_cols = col_sums > 0
        if not valid_cols.any():
            return {'error': '所有列均为零'}
        if not valid_cols.all():
            X_valid = X[:, valid_cols]
            valid_labels = [lbl for i, lbl in enumerate(col_labels) if valid_cols[i]]
        else:
            X_valid = X
            valid_labels = col_labels

        # 最小二乘求解（无正负约束）
        w_valid, residuals, rank, sv = np.linalg.lstsq(X_valid, t, rcond=None)
        residual = float(residuals[0]) if len(residuals) > 0 else 0.0

        if not valid_cols.all():
            w_full = np.zeros(len(col_labels))
            valid_indices = np.where(valid_cols)[0]
            for i, wi in zip(valid_indices, w_valid):
                w_full[i] = wi
        else:
            w_full = w_valid

        # LSTSQ 无裁剪 — 允许任意大小的权重

        # 计算每期的得分
        period_scores = []
        for p_idx, (period_preds, (actual_main, actual_aux)) in enumerate(
                zip(all_period_predictions, all_period_actuals)):
            X_main, X_aux, p_col_labels, _, _ = self._build_design_matrix(
                period_preds)
            X_p = np.vstack([X_main, X_aux])
            t_p = np.concatenate([
                self._build_target_vector(actual_main, actual_aux)[0],
                self._build_target_vector(actual_main, actual_aux)[1]
            ])

            # 对齐
            X_p_aligned = np.zeros((X_p.shape[0], len(col_labels)))
            for j_old, label in enumerate(p_col_labels):
                j_new = col_labels.index(label) if label in col_labels else -1
                if j_new >= 0:
                    X_p_aligned[:, j_new] = X_p[:, j_old]

            scores = X_p_aligned.dot(w_full)
            actual_score = scores[t_p > 0].mean() if t_p.sum() > 0 else 0.0
            period_scores.append(float(actual_score))

        # 构建 composite_weights 字典（不再分解）
        composite_weights = {
            col_labels[i]: round(float(w_full[i]), 6)
            for i in range(len(col_labels))
            if abs(w_full[i]) > 1e-10
        }

        return {
            'composite_weights': composite_weights,
            'residual': residual,
            'period_scores': period_scores,
            'avg_score': float(np.mean(period_scores)) if period_scores else 0.0,
            'column_labels': col_labels,
            'n_periods': n_periods,
        }

    # ------------------------------------------------------------------
    #  权重分解 — 已移除。权重直接使用 65 个独立 composite_weights。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  验证
    # ------------------------------------------------------------------

    def validate_solution(self,
                           solution: Dict,
                           per_group_predictions: List[Dict],
                           actual_main: List[int],
                           actual_aux: List[int]
                           ) -> Dict[str, Any]:
        """
        验证求解结果：用解出的 composite_weights 重新计算得票和Top-K命中。

        返回:
          {
            'predicted_main': [...],
            'predicted_aux': [...],
            'main_hits': int, 'aux_hits': int, 'total_hits': int,
          }
        """
        composite_w = solution.get('composite_weights', {})

        # 计算每个号码的加权得票
        main_votes = defaultdict(float)
        aux_votes = defaultdict(float)

        for entry in per_group_predictions:
            mk = entry['method_key']
            gk = entry['granularity']
            key = f'{mk}@{gk}'
            weight = composite_w.get(key, 1.0)

            for num in entry.get('predicted_main', []):
                main_votes[num] += weight
            for num in entry.get('predicted_aux', []):
                aux_votes[num] += weight

        # Top-K
        sorted_main = sorted(main_votes.items(), key=lambda x: x[1], reverse=True)
        predicted_main = [n for n, _ in sorted_main[:self.main_count]]

        sorted_aux = sorted(aux_votes.items(), key=lambda x: x[1], reverse=True)
        predicted_aux = [n for n, _ in sorted_aux[:self.aux_count]]

        # 命中
        main_hits = len(set(predicted_main) & set(actual_main))
        aux_hits = len(set(predicted_aux) & set(actual_aux))

        return {
            'predicted_main': sorted(predicted_main),
            'predicted_aux': sorted(predicted_aux),
            'actual_main': sorted(actual_main),
            'actual_aux': sorted(actual_aux),
            'main_hits': main_hits,
            'aux_hits': aux_hits,
            'total_hits': main_hits + aux_hits,
        }


# ============================================================================
#  便捷函数
# ============================================================================

def extract_predictions_from_result(period_results: List[Dict],
                                     gran_names: List[str],
                                     method_merge: Any = None
                                     ) -> List[List[Dict]]:
    """
    从 evaluate_combo 风格的 period_results 中提取各组合的预测。

    参数:
      period_results: [{period_idx, gran_predictions: {gran_name: {method_key: {predictions: {...}}}}}]
      gran_names: 颗粒度名称列表

    返回:
      [[{method_key, granularity, predicted_main, predicted_aux}, ...], ...]
    """
    all_period_preds = []

    for pr in period_results:
        period_preds = []
        gran_preds = pr.get('gran_predictions', {})
        for gran_name in gran_names:
            methods_dict = gran_preds.get(gran_name, {})
            for mk, result in methods_dict.items():
                if 'error' in result or 'predictions' not in result:
                    continue
                pred = result['predictions']
                period_preds.append({
                    'method_key': mk,
                    'granularity': gran_name,
                    'predicted_main': pred.get('red', pred.get('front', [])),
                    'predicted_aux': pred.get('blue', pred.get('back', [])),
                })
        all_period_preds.append(period_preds)

    return all_period_preds


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  线性权重求解器 5.0 测试")
    print("=" * 60)

    solver = LinearWeightSolver('ssq')

    # 模拟 3 个 (方法 × 颗粒度) 组合的预测
    mock_predictions = [
        {'method_key': 'method_1', 'granularity': '500期',
         'predicted_main': [1, 5, 10, 15, 20, 25],
         'predicted_aux': [3, 7]},
        {'method_key': 'method_2', 'granularity': '500期',
         'predicted_main': [2, 6, 11, 16, 21, 26],
         'predicted_aux': [5, 9]},
        {'method_key': 'method_1', 'granularity': '100期',
         'predicted_main': [1, 3, 10, 18, 22, 28],
         'predicted_aux': [4, 8]},
    ]

    actual_main = [1, 5, 10, 16, 22, 26]
    actual_aux = [7]

    result = solver.solve_single_period(
        mock_predictions, actual_main, actual_aux)

    print(f"\n单期求解结果 (LSTSQ, 无正负约束):")
    print(f"  残差: {result.get('residual', 'N/A'):.6f}")
    print(f"  实际号码平均得票: {result.get('actual_score', 'N/A'):.4f}")
    print(f"  composite_weights ({len(result.get('composite_weights', {}))}个非零权重):")
    for key, w in sorted(result.get('composite_weights', {}).items(),
                         key=lambda x: abs(x[1]), reverse=True):
        print(f"    {key}: {w:+.6f}")

    # 验证
    validation = solver.validate_solution(
        result, mock_predictions, actual_main, actual_aux)
    print(f"\n验证:")
    print(f"  预测号码: {validation['predicted_main']} + {validation['predicted_aux']}")
    print(f"  实际号码: {sorted(actual_main)} + {sorted(actual_aux)}")
    print(f"  命中: 主{validation['main_hits']} 辅{validation['aux_hits']} "
          f"总{validation['total_hits']}")

    # 测试多期求解
    print(f"\n--- 多期求解测试 (LSTSQ) ---")
    multi_result = solver.solve_multi_period(
        [mock_predictions, mock_predictions],
        [(actual_main, actual_aux), ([2, 6, 11, 16, 21, 26], [5])]
    )
    print(f"  平均得分: {multi_result.get('avg_score', 'N/A'):.4f}")
    print(f"  各期得分: {multi_result.get('period_scores', [])}")
    print(f"  composite_weights ({len(multi_result.get('composite_weights', {}))}个):")
    for key, w in sorted(multi_result.get('composite_weights', {}).items(),
                         key=lambda x: abs(x[1]), reverse=True):
        print(f"    {key}: {w:+.6f}")

    # 测试 LP 求解
    print(f"\n--- 多期求解测试 (LP 精确求解) ---")
    lp_result = solver.solve_lp_multi_period(
        [mock_predictions, mock_predictions],
        [(actual_main, actual_aux), ([2, 6, 11, 16, 21, 26], [5])],
        epsilon=0.01, max_weight=500.0,
    )
    print(f"  LP成功: {lp_result.get('lp_success', False)}")
    print(f"  LP状态: {lp_result.get('lp_status', 'N/A')[:80]}")
    if lp_result.get('composite_weights'):
        print(f"  composite_weights ({len(lp_result['composite_weights'])}个):")
        for key, w in sorted(lp_result['composite_weights'].items(),
                             key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"    {key}: {w:+.6f}")

    print("\n[OK] 线性权重求解器 5.0 测试完成")
