"""
线性权重求解器
===============
对于求解模式，将权重优化从离散搜索中分离出来，
用非负最小二乘 (NNLS) 精确求解最优合并权重。

核心原理:
  合并层投票运算是线性的：
    v_n = SUM_j w_j × I[号码n被组合j预测]
    其中 v_n 是号码n的得票，w_j 是组合j的权重

  给定固定模型参数（各组合的预测结果已确定），
  最优权重是使得实际号码得票最大化的 w 向量。

用法:
  solver = LinearWeightSolver('ssq')
  solution = solver.solve(predictions_by_period, actual_numbers_by_period)
  # solution = {'method_weights': {...}, 'granularity_weights': {...}}

依赖:
  scipy.optimize.nnls (标准库)
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
    用非负最小二乘 (NNLS) 求解最优合并权重。

    问题形式化:
      X: (N个候选号码) × (M个方法×颗粒度组合) 的0/1矩阵
         X[n, j] = 1 表示组合j预测了号码n
      t: 目标向量，t[n] = 1 若n是实际开奖号码，否则 0
      w: 待求解的权重向量 (非负)

      min || X w - t ||^2
      s.t. w >= 0
    """

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
        对单期数据求解最优权重。

        返回:
          {
            'composite_weights': [...],  # 每个组合的扁平权重
            'method_weights': {...},
            'granularity_weights': {...},
            'residual': float,             # NNLS 残差
            'predicted_score': float,      # 预测的实际号码平均得票
            'column_labels': [...],
          }
        """
        from scipy.optimize import nnls

        X_main, X_aux, col_labels, _, _ = self._build_design_matrix(
            per_group_predictions)
        t_main, t_aux = self._build_target_vector(actual_main, actual_aux)

        # 合并主球和辅助球
        X = np.vstack([X_main, X_aux])
        t = np.concatenate([t_main, t_aux])

        if X.shape[0] == 0 or X.shape[1] == 0:
            return {'error': '设计矩阵为空', 'composite_weights': []}

        # 检查是否有有效列（至少有一个非零元素）
        col_sums = X.sum(axis=0)
        valid_cols = col_sums > 0
        if not valid_cols.any():
            return {'error': '所有列均为零 — 预测结果全空',
                    'composite_weights': np.zeros(X.shape[1])}

        # 如果只有部分列有效，只对有效列求解
        if not valid_cols.all():
            X_valid = X[:, valid_cols]
            valid_labels = [lbl for i, lbl in enumerate(col_labels) if valid_cols[i]]
        else:
            X_valid = X
            valid_labels = col_labels

        # NNLS 求解
        w_valid, residual = nnls(X_valid, t)

        # 映射回完整权重向量
        if not valid_cols.all():
            w_full = np.zeros(X.shape[1])
            valid_indices = np.where(valid_cols)[0]
            for i, wi in zip(valid_indices, w_valid):
                w_full[i] = wi
        else:
            w_full = w_valid

        # 归一化（避免权重过大或过小）
        w_sum = w_full.sum()
        if w_sum > 1e-10:
            w_full = w_full / w_sum

        # 计算实际号码的预测得票
        if X.shape[1] > 0:
            predicted_scores = X.dot(w_full)
            actual_score = predicted_scores[t > 0].mean() if t.sum() > 0 else 0.0
        else:
            actual_score = 0.0

        # 分解为 method_weights 和 granularity_weights
        method_weights, gran_weights = self._decompose_weights(
            w_full, per_group_predictions, col_labels)

        return {
            'composite_weights': w_full.tolist(),
            'method_weights': method_weights,
            'granularity_weights': gran_weights,
            'residual': float(residual),
            'actual_score': float(actual_score),
            'column_labels': col_labels,
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
        from scipy.optimize import nnls

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
        row_weights_blocks = []

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
            X = X[:, valid_cols]
            valid_labels = [lbl for i, lbl in enumerate(col_labels) if valid_cols[i]]
        else:
            valid_labels = col_labels

        w_valid, residual = nnls(X, t)

        if not valid_cols.all():
            w_full = np.zeros(len(col_labels))
            valid_indices = np.where(valid_cols)[0]
            for i, wi in zip(valid_indices, w_valid):
                w_full[i] = wi
        else:
            w_full = w_valid

        # 归一化
        w_sum = w_full.sum()
        if w_sum > 1e-10:
            w_full = w_full / w_sum

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

        # 分解权重
        # 需要每个组合的信息，从第一个周期获取
        method_weights, gran_weights = self._decompose_weights(
            w_full, all_period_predictions[0], col_labels)

        return {
            'composite_weights': w_full.tolist(),
            'method_weights': method_weights,
            'granularity_weights': gran_weights,
            'residual': float(residual),
            'period_scores': period_scores,
            'avg_score': float(np.mean(period_scores)) if period_scores else 0.0,
            'column_labels': col_labels,
            'n_periods': n_periods,
        }

    # ------------------------------------------------------------------
    #  权重分解
    # ------------------------------------------------------------------

    def _decompose_weights(self,
                            composite_weights: np.ndarray,
                            per_group_entries: List[Dict],
                            col_labels: List[str]
                            ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        将扁平复合权重分解为 method_weights 和 granularity_weights。

        composite_weight[j] ≈ method_weight[method_j] × granularity_weight[gran_j]

        方法：对数线性最小二乘
          log(w_j) ≈ log(m_method_j) + log(g_gran_j)
          用线性最小二乘求解
        """
        # 构建对齐后的映射
        label_to_idx = {lbl: i for i, lbl in enumerate(col_labels)}

        methods_seen = set()
        grans_seen = set()
        rows = []

        for entry in per_group_entries:
            label = f"{entry['method_key']}@{entry['granularity']}"
            idx = label_to_idx.get(label)
            if idx is None:
                continue
            w = composite_weights[idx]
            if w <= 1e-10:
                continue
            methods_seen.add(entry['method_key'])
            grans_seen.add(entry['granularity'])
            rows.append((entry['method_key'], entry['granularity'],
                          np.log(w)))

        if len(rows) < 2:
            # 样本不足，返回均权
            return (
                {mk: 1.0 for mk in methods_seen} or {mk: 1.0 for mk in METHOD_KEYS},
                {gk: 1.0 for gk in grans_seen} or {gk: 1.0 for gk in GRANULARITY_NAMES},
            )

        # 构建设计矩阵（one-hot 编码方法 + 颗粒度）
        methods_list = sorted(methods_seen)
        grans_list = sorted(grans_seen)
        n_methods = len(methods_list)
        n_grans = len(grans_list)

        A = np.zeros((len(rows), n_methods + n_grans))
        b = np.zeros(len(rows))

        for i, (mk, gk, log_w) in enumerate(rows):
            A[i, methods_list.index(mk)] = 1.0
            A[i, n_methods + grans_list.index(gk)] = 1.0
            b[i] = log_w

        # 最小二乘求解
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

        # 转回线性空间
        log_method = x[:n_methods]
        log_gran = x[n_methods:]

        # 归一化
        method_raw = np.exp(log_method - log_method.mean())
        gran_raw = np.exp(log_gran - log_gran.mean())

        method_weights = {mk: round(float(method_raw[i]), 4)
                          for i, mk in enumerate(methods_list)}
        gran_weights = {gk: round(float(gran_raw[i]), 4)
                        for i, gk in enumerate(grans_list)}

        # 补全未出现的方法/颗粒度
        for mk in METHOD_KEYS:
            if mk not in method_weights:
                method_weights[mk] = 1.0
        for gk in GRANULARITY_NAMES:
            if gk not in gran_weights:
                gran_weights[gk] = 1.0

        return method_weights, gran_weights

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
        验证求解结果：用解出的权重重新计算得票和Top-K命中。

        返回:
          {
            'predicted_main': [...],  # 按加权得票排序的Top-K号码
            'predicted_aux': [...],
            'main_hits': int,
            'aux_hits': int,
            'total_hits': int,
            'vote_details': {...},
          }
        """
        method_w = solution.get('method_weights', {})
        gran_w = solution.get('granularity_weights', {})

        # 计算每个号码的加权得票
        main_votes = defaultdict(float)
        aux_votes = defaultdict(float)

        for entry in per_group_predictions:
            mk = entry['method_key']
            gk = entry['granularity']
            weight = method_w.get(mk, 1.0) * gran_w.get(gk, 1.0)

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
    print("  线性权重求解器 测试")
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

    print(f"\n求解结果:")
    print(f"  残差: {result.get('residual', 'N/A'):.6f}")
    print(f"  实际号码平均得票: {result.get('actual_score', 'N/A'):.4f}")
    print(f"  方法权重: {result.get('method_weights', {})}")
    print(f"  颗粒度权重: {result.get('granularity_weights', {})}")

    # 验证
    validation = solver.validate_solution(
        result, mock_predictions, actual_main, actual_aux)
    print(f"\n验证:")
    print(f"  预测号码: {validation['predicted_main']} + {validation['predicted_aux']}")
    print(f"  实际号码: {sorted(actual_main)} + {sorted(actual_aux)}")
    print(f"  命中: 主{validation['main_hits']} 辅{validation['aux_hits']} "
          f"总{validation['total_hits']}")

    # 测试多期求解
    print(f"\n--- 多期求解测试 ---")
    multi_result = solver.solve_multi_period(
        [mock_predictions, mock_predictions],  # 同一组预测重复两次
        [(actual_main, actual_aux), ([2, 6, 11, 16, 21, 26], [5])]
    )
    print(f"  平均得分: {multi_result.get('avg_score', 'N/A'):.4f}")
    print(f"  各期得分: {multi_result.get('period_scores', [])}")

    print("\n[OK] 线性权重求解器测试完成")
