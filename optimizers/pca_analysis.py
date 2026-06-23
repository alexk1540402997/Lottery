"""
PCA 参数自由度分析工具
=======================
分析参数→预测输出的有效自由度，为贝叶斯优化提供维度信息。

分析方法:
  1. 参数敏感性分析 — 单参数扰动，检测"死参数"
  2. 输出空间 PCA — 随机采样，分析预测输出的有效维度
  3. 参数相关性矩阵 — 参数之间的冗余度

用法:
  python optimizers/pca_analysis.py --lottery ssq --samples 200

输出:
  - PCA 报告（控制台 + 文本文件）
  - 有效自由度估算
  - 参数重要性排名
  - 优化器选型建议
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 导入项目模块
from predictor import LotteryPredictor, METHOD_NAMES_NEW
from backtester import PARAM_SEARCH_SPACE, WEIGHT_SEARCH_SPACE


# ============================================================================
#  工具函数
# ============================================================================

def count_free_params(search_space: Dict) -> Tuple[int, List[str]]:
    """统计搜索空间中的自由参数数量"""
    param_names = []
    for method, params in search_space.items():
        for pname, pvalues in params.items():
            if len(pvalues) > 1:  # 多于1个候选值才算自由参数
                param_names.append(f"{method}.{pname}")
    return len(param_names), param_names


def sample_random_params(rng: np.random.RandomState) -> Dict[str, Dict]:
    """从搜索空间中随机采样一组参数"""
    sampled = {}
    for method_name, space in PARAM_SEARCH_SPACE.items():
        config = {}
        for pname, pvalues in space.items():
            config[pname] = pvalues[rng.randint(0, len(pvalues))]
        sampled[method_name] = config
    return sampled


def sample_random_weights(rng: np.random.RandomState) -> Dict:
    """随机采样 65 个独立 composite_weights"""
    w_min, w_max = WEIGHT_SEARCH_SPACE['composite_weight_range']

    composite_weights = {}
    for i in range(1, 14):
        for gn in ['50期', '100期', '500期', '1000期', '全部期']:
            key = f'method_{i}@{gn}'
            composite_weights[key] = round(rng.uniform(w_min, w_max), 4)

    return {'composite_weights': composite_weights}


def params_to_vector(params: Dict) -> np.ndarray:
    """
    将参数字典扁平化为向量。

    使用连续松弛：将离散候选值映射到 [0,1] 区间。
    """
    flat = []
    for method_name in sorted(PARAM_SEARCH_SPACE.keys()):
        space = PARAM_SEARCH_SPACE[method_name]
        for pname in sorted(space.keys()):
            pvalues = space[pname]
            if len(pvalues) == 1:
                continue  # 固定参数跳过
            current = params.get(method_name, {}).get(pname, pvalues[0])
            # 映射到 [0,1]
            try:
                idx = pvalues.index(current)
            except ValueError:
                idx = 0
            normalized = idx / max(1, len(pvalues) - 1)
            flat.append(normalized)
    return np.array(flat)


def params_to_binary_features(all_predictions: Dict[str, List[int]],
                              main_range: Tuple[int, int]
                              ) -> np.ndarray:
    """
    将预测输出转为二值特征向量。

    对于每个方法和每个可能的号码，标记是否被预测（1/0）。
    返回: [method1_num1, method1_num2, ..., method13_numN]
    """
    features = []
    for method_key in sorted(all_predictions.keys()):
        pred = all_predictions.get(method_key, [])
        if isinstance(pred, list):
            for num in range(main_range[0], main_range[1] + 1):
                features.append(1 if num in pred else 0)
    return np.array(features, dtype=int)


# ============================================================================
#  PCA 分析器
# ============================================================================

class PCAAnalyzer:
    """参数自由度 PCA 分析器"""

    def __init__(self, lottery_type: str = 'ssq'):
        self.lottery_type = lottery_type.lower()
        self.predictor = LotteryPredictor(lottery_type)
        self.param_names = []
        self.n_free_params = 0

        # 分析结果
        self.sensitivity_results = {}     # 敏感性分析
        self.pca_model = None
        self.pca_explained_variance = None
        self.effective_dim = 0
        self.param_importance = {}        # 参数重要性排名
        self.dead_params = []             # 死参数列表

    # ------------------------------------------------------------------
    #  加载数据
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """
        加载训练数据。

        使用 LotteryPredictor.load_data() 确保列名标准化
        （原始Excel使用中文列名如'红球号码1'，需映射为'red_1'）

        返回: data_reverse（倒序，最新在 index=0）
        """
        data_dir = PROJECT_ROOT / 'pythonProject'
        if self.lottery_type == 'ssq':
            data_file = data_dir / '双色球.xlsx'
        else:
            data_file = data_dir / '大乐透.xlsx'

        if not data_file.exists():
            alt = PROJECT_ROOT / f'{"双色球" if self.lottery_type == "ssq" else "大乐透"}.xlsx'
            if alt.exists():
                data_file = alt

        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        # 使用 Predictor 的 load_data 确保列名标准化 + 倒序
        data_rev, detected_type = LotteryPredictor.load_data(str(data_file))
        print(f"[数据] 加载 {len(data_rev)} 期数据 (类型={detected_type}): {data_file}")
        print(f"[数据] 列名: {list(data_rev.columns[:10])}")
        return data_rev

    # ------------------------------------------------------------------
    #  分析 1: 参数敏感性（单参数扰动）
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(self, train_data: pd.DataFrame):
        """
        逐个参数单独扰动，检测对预测结果的影响。

        对每个自由参数:
          - 固定其他参数为默认值
          - 遍历该参数的所有候选值
          - 检查预测结果是否发生变化
          - 无变化的参数标记为"惰性参数"
        """
        print("\n" + "=" * 60)
        print("  分析 1: 参数敏感性（单参数扰动）")
        print("=" * 60)

        from predictor import DEFAULT_PARAMS
        base_params = DEFAULT_PARAMS.copy()

        # 获取默认参数（深拷贝）
        base = {}
        for method, config in base_params.items():
            # 只保留 PARAM_SEARCH_SPACE 中存在的参数
            if method in PARAM_SEARCH_SPACE:
                base[method] = {}
                for pname in PARAM_SEARCH_SPACE[method]:
                    if pname in config:
                        base[method][pname] = config[pname]
                    elif len(PARAM_SEARCH_SPACE[method][pname]) > 0:
                        base[method][pname] = PARAM_SEARCH_SPACE[method][pname][0]

        # 基准预测
        print("[敏感性] 生成基准预测...")
        base_predictions = self._predict_with_params(train_data, base)
        base_main = set(base_predictions.get('merged_main', []))
        print(f"  基准预测: {sorted(base_main)}")

        total_params = 0
        sensitive_params = 0
        dead_params = []
        param_effects = []

        for method_name in sorted(PARAM_SEARCH_SPACE.keys()):
            space = PARAM_SEARCH_SPACE[method_name]
            for pname in sorted(space.keys()):
                pvalues = space[pname]
                if len(pvalues) <= 1:
                    continue

                total_params += 1
                method_label = METHOD_NAMES_NEW.get(
                    f'method_{list(PARAM_SEARCH_SPACE.keys()).index(method_name) + 1}',
                    method_name)

                # 测试每个候选值
                changed = False
                unique_outputs = set()

                for pval in pvalues:
                    test_params = _deep_copy_params(base)
                    if method_name not in test_params:
                        test_params[method_name] = {}
                    test_params[method_name][pname] = pval

                    predictions = self._predict_with_params(train_data, test_params)
                    main_set = frozenset(predictions.get('merged_main', []))
                    unique_outputs.add(main_set)

                n_unique = len(unique_outputs)
                has_effect = n_unique > 1
                if has_effect:
                    sensitive_params += 1
                else:
                    dead_params.append(f"{method_name}.{pname}")

                param_effects.append({
                    'parameter': f"{method_name}.{pname}",
                    'method': method_label,
                    'n_values': len(pvalues),
                    'n_unique_outputs': n_unique,
                    'has_effect': has_effect,
                })

                status = f"→ {n_unique}种不同输出" if has_effect else "→ 无影响(死参数)"
                print(f"  [{method_name}.{pname}] 测试{len(pvalues)}个值 {status}")

        self.sensitivity_results = {
            'total_free_params': total_params,
            'sensitive_params': sensitive_params,
            'dead_params': dead_params,
            'param_effects': param_effects,
        }
        self.dead_params = dead_params

        print(f"\n[敏感性结果] {total_params}个自由参数 → "
              f"{sensitive_params}个有影响, {len(dead_params)}个死参数")
        if dead_params:
            print(f"  死参数: {', '.join(dead_params)}")

    # ------------------------------------------------------------------
    #  分析 2: 输出空间 PCA
    # ------------------------------------------------------------------

    def run_output_pca(self, train_data: pd.DataFrame, n_samples: int = 200):
        """
        随机采样参数组合 → 运行预测 → 对预测输出做 PCA。

        使用预测的号码集合作为特征（而非命中数），
        因为号码集合包含了更丰富的信息。
        """
        print("\n" + "=" * 60)
        print(f"  分析 2: 输出空间 PCA（{n_samples} 组随机采样）")
        print("=" * 60)

        n_free, param_names = count_free_params(PARAM_SEARCH_SPACE)
        self.param_names = param_names
        self.n_free_params = n_free
        print(f"[PCA] 自由参数总数: {n_free}")
        print(f"[PCA] 参数列表: {', '.join(param_names[:10])}...")

        rng = np.random.RandomState(42)

        # 收集数据
        X_params = []        # 参数向量
        X_outputs = []       # 预测输出特征（号码二值向量）
        combo_records = []   # 详细记录

        main_range = (1, 33) if self.lottery_type == 'ssq' else (1, 35)

        print(f"\n[PCA] 开始采样 {n_samples} 组...")
        t0 = time.time()

        for i in range(n_samples):
            params = sample_random_params(rng)
            weights = sample_random_weights(rng)

            predictions = self._predict_with_params(
                train_data, params, weights)

            if 'error' in predictions:
                print(f"  [{i+1}/{n_samples}] 评估失败，跳过")
                continue

            # 参数向量（连续松弛）
            param_vec = params_to_vector(params)
            X_params.append(param_vec)

            # 输出特征：每种方法预测的号码集合
            output_features = self._extract_output_features(
                predictions, main_range)
            X_outputs.append(output_features)

            combo_records.append({
                'sample_id': i + 1,
                'params': {mk: dict(mp) for mk, mp in params.items()},
                'weights': weights,
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate
                print(f"  [{i+1}/{n_samples}] 已完成, "
                      f"速率={rate:.1f}组/秒, 预计剩余{eta:.0f}秒")

        elapsed = time.time() - t0
        n_valid = len(X_outputs)
        print(f"\n[PCA] 采样完成: {n_valid}/{n_samples} 组有效, "
              f"耗时 {elapsed:.0f} 秒")

        if n_valid < 10:
            print("[PCA] 有效样本不足，无法进行 PCA 分析")
            return

        # 转换为 numpy
        X_out = np.array(X_outputs)  # shape: (n_valid, n_features)
        n_features = X_out.shape[1]
        print(f"[PCA] 输出特征维度: {n_features} (={n_features}个二值特征)")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_out.astype(float))

        # PCA
        n_components = min(n_valid - 1, n_features, 50)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        self.pca_model = pca
        self.pca_explained_variance = pca.explained_variance_ratio_

        # 计算有效维度（解释 90% 方差所需的主成分数）
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        self.effective_dim_90 = int(np.searchsorted(cumsum, 0.90) + 1)
        self.effective_dim_95 = int(np.searchsorted(cumsum, 0.95) + 1)

        print(f"\n[PCA 结果]")
        print(f"  有效维度 (90%方差): {self.effective_dim_90}")
        print(f"  有效维度 (95%方差): {self.effective_dim_95}")
        print(f"  参数空间维度:      {n_free}")
        print(f"  压缩比:            {n_free / max(1, self.effective_dim_90):.1f}:1")
        print(f"  前10主成分方差占比:")
        for i in range(min(10, n_components)):
            print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
                  f"(累计 {cumsum[i]:.4f})")

        # 输出解释摘要
        print(f"\n[PCA 解读]")
        if self.effective_dim_90 <= 8:
            print(f"  [OK] 有效维度很低({self.effective_dim_90})，"
                  f"参数高度冗余，BO 可以使用低维嵌入")
        elif self.effective_dim_90 <= 15:
            print(f"  [OK] 有效维度适中({self.effective_dim_90})，"
                  f"标准 BO 可行")
        elif self.effective_dim_90 <= 25:
            print(f"  [WARN] 有效维度偏高({self.effective_dim_90})，"
                  f"BO 可能需要稀疏 GP 或随机嵌入")
        else:
            print(f"  [WARN] 有效维度很高({self.effective_dim_90})，"
                  f"建议降维或使用 CMA-ES")

        # X_params PCA（参数空间分析）
        if X_params:
            Xp = np.array(X_params)
            if Xp.shape[0] > 10 and Xp.shape[1] > 1:
                pca_params = PCA(n_components=min(Xp.shape[0]-1, Xp.shape[1]))
                pca_params.fit(Xp)
                params_dim_90 = int(
                    np.searchsorted(np.cumsum(pca_params.explained_variance_ratio_), 0.90) + 1)
                print(f"\n[参数空间 PCA]")
                print(f"  参数有效维度 (90%): {params_dim_90}")
                print(f"  参数空间原始维度: {Xp.shape[1]}")
                self.params_dim_90 = params_dim_90

    # ------------------------------------------------------------------
    #  分析 3: 参数重要性排名
    # ------------------------------------------------------------------

    def run_importance_analysis(self, train_data: pd.DataFrame, n_trials: int = 100):
        """
        通过随机采样，分析每个参数对预测输出的影响程度。

        方法：对每个参数，计算其不同取值下输出变化的方差。
        """
        print("\n" + "=" * 60)
        print(f"  分析 3: 参数重要性排名（{n_trials} 组随机评估）")
        print("=" * 60)

        rng = np.random.RandomState(123)
        main_range = (1, 33) if self.lottery_type == 'ssq' else (1, 35)

        # 对每组参数，记录参数值和对应的输出特征
        param_values = defaultdict(list)  # param_name → [values across trials]
        output_diversity = []             # 每组输出的唯一性度量

        for trial in range(n_trials):
            params = sample_random_params(rng)
            predictions = self._predict_with_params(train_data, params)

            if 'error' in predictions:
                continue

            # 记录参数值
            for method_name, config in params.items():
                for pname, pval in config.items():
                    full_name = f"{method_name}.{pname}"
                    param_values[full_name].append(pval)

            # 记录输出特征
            features = self._extract_output_features(predictions, main_range)
            output_diversity.append(features)

        # 对每个参数，计算其值的变化与输出变化的关联
        importance_scores = {}
        output_array = np.array(output_diversity)

        for param_name, values in param_values.items():
            if len(set(values)) <= 1:
                importance_scores[param_name] = 0.0
                continue

            # 将参数值映射为数值
            if all(isinstance(v, (int, float)) for v in values):
                numeric_vals = np.array(values)
            else:
                # 非数值→hash映射
                unique_vals = sorted(set(values))
                val_map = {v: i for i, v in enumerate(unique_vals)}
                numeric_vals = np.array([val_map[v] for v in values])

            # 按参数值分组，计算组间输出方差 / 总方差
            # 使用简化的 ANOVA 方案
            unique_groups = np.unique(numeric_vals)
            if len(unique_groups) < 2:
                importance_scores[param_name] = 0.0
                continue

            group_means = []
            for g in unique_groups:
                mask = numeric_vals == g
                if mask.sum() > 0:
                    group_means.append(output_array[mask].mean(axis=0))

            if len(group_means) >= 2:
                # 组间方差
                group_means_arr = np.array(group_means)
                between_var = group_means_arr.var(axis=0).mean()
                total_var = output_array.var(axis=0).mean()
                if total_var > 0:
                    importance_scores[param_name] = float(
                        min(1.0, between_var / total_var))
                else:
                    importance_scores[param_name] = 0.0
            else:
                importance_scores[param_name] = 0.0

        # 排序
        ranked = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        self.param_importance = dict(ranked)

        print(f"\n[参数重要性排名] (归一化影响力 0~1)")
        print(f"{'排名':<5} {'参数':<40} {'影响力':>8}")
        print("-" * 55)
        for rank, (pname, score) in enumerate(ranked[:20], 1):
            bar = "#" * int(score * 30)
            print(f"{rank:<5} {pname:<40} {score:>6.4f} {bar}")

        # 标记低影响力参数（排名后25%，影响力<0.1）
        threshold = max(0.05, ranked[len(ranked)//4*3][1])
        low_impact = [p for p, s in ranked if s < max(0.1, threshold)]
        print(f"\n[低影响力参数] (可考虑固定为默认值):")
        if low_impact:
            for p in low_impact[:15]:
                print(f"  - {p}")
        else:
            print("  无显著低影响力参数")

    # ------------------------------------------------------------------
    #  辅助方法
    # ------------------------------------------------------------------

    def _predict_with_params(self, train_data: pd.DataFrame,
                             params: Dict,
                             weights: Dict = None
                             ) -> Dict:
        """用指定参数运行预测（单颗粒度500期加速）"""
        try:
            from merger import ResultMerger

            # 用500期训练
            gran_data = train_data.head(min(len(train_data), 500))

            if len(gran_data) < 20:
                return {'error': '数据不足'}

            # 运行13种方法预测
            all_predictions = self.predictor.predict_all(
                gran_data, params=params, seed=42)

            # 合并
            merger = ResultMerger(self.lottery_type)
            if weights:
                merger.import_weights(weights)

            merged = merger.merge_results({'500期': all_predictions})
            merged_main = merged['predictions'][self.predictor.main_name]
            merged_aux = merged['predictions'][self.predictor.aux_name]

            # 返回各方法预测详情 + 合并结果
            result = {
                'merged_main': sorted(merged_main),
                'merged_aux': sorted(merged_aux),
            }
            for mk, pred in all_predictions.items():
                if 'predictions' in pred:
                    result[mk] = pred['predictions'].get(
                        self.predictor.main_name, [])
                elif 'error' not in pred:
                    result[mk] = pred.get(self.predictor.main_name, [])

            return result
        except Exception as e:
            return {'error': str(e)}

    def _extract_output_features(self, predictions: Dict,
                                 main_range: Tuple[int, int]
                                 ) -> np.ndarray:
        """
        从预测结果中提取二值特征向量。

        对每种方法：标记预测了哪些号码（33维二值向量）
        最终特征 = 所有方法的二值向量拼接
        """
        features = []
        for i in range(1, 14):
            mk = f'method_{i}'
            pred_nums = predictions.get(mk, [])
            if isinstance(pred_nums, dict):
                pred_nums = pred_nums.get(self.predictor.main_name, [])
            if not isinstance(pred_nums, list):
                pred_nums = []
            for num in range(main_range[0], main_range[1] + 1):
                features.append(1 if num in pred_nums else 0)
        return np.array(features, dtype=int)

    # ------------------------------------------------------------------
    #  报告生成
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """生成完整的分析报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("  PCA 参数自由度分析报告")
        lines.append("=" * 60)
        lines.append(f"  彩票类型: {self.lottery_type.upper()}")
        lines.append(f"  自由参数总数: {self.n_free_params}")
        lines.append(f"  分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 敏感性结果
        if self.sensitivity_results:
            sr = self.sensitivity_results
            lines.append("-" * 60)
            lines.append("  1. 参数敏感性分析")
            lines.append("-" * 60)
            lines.append(f"  总自由参数: {sr['total_free_params']}")
            lines.append(f"  有影响的参数: {sr['sensitive_params']}")
            lines.append(f"  死参数: {len(sr['dead_params'])}")
            if sr['dead_params']:
                lines.append(f"  死参数列表: {', '.join(sr['dead_params'])}")
                lines.append("  [TIP] 建议: 死参数可从搜索空间中移除，缩减维度")
            lines.append("")

        # PCA 结果
        if hasattr(self, 'effective_dim_90'):
            lines.append("-" * 60)
            lines.append("  2. 输出空间 PCA 分析")
            lines.append("-" * 60)
            lines.append(f"  有效维度 (90%方差): {self.effective_dim_90}")
            lines.append(f"  有效维度 (95%方差): {self.effective_dim_95}")
            lines.append(f"  参数空间原始维度: {self.n_free_params}")
            ratio = self.n_free_params / max(1, self.effective_dim_90)
            lines.append(f"  压缩比: {ratio:.1f}:1")
            lines.append("")

            # 优化器选型建议
            lines.append("  [TIP] 优化器选型建议:")
            if self.effective_dim_90 <= 8:
                lines.append(f"     有效维度低({self.effective_dim_90})，"
                             f"标准 GP-BO 是最佳选择")
                lines.append(f"     推荐: RBF 核 + EI 采集函数")
                lines.append(f"     CMA-ES 和 SA 作为备选")
            elif self.effective_dim_90 <= 15:
                lines.append(f"     有效维度适中({self.effective_dim_90})，"
                             f"标准 BO 可行")
                lines.append(f"     推荐: BO 为主，CMA-ES 为辅")
                lines.append(f"     可使用 Matern 核提升鲁棒性")
            elif self.effective_dim_90 <= 25:
                lines.append(f"     有效维度偏高({self.effective_dim_90})，"
                             f"BO 需要降维辅助")
                lines.append(f"     推荐: 稀疏GP 或 随机嵌入BO")
                lines.append(f"     CMA-ES 可能表现更好")
            else:
                lines.append(f"     有效维度很高({self.effective_dim_90})，"
                             f"GP-BO 可能退化")
                lines.append(f"     推荐: CMA-ES 为主，BO 做局部精化")
                lines.append(f"     建议用 PCA 降维预处理")
            lines.append("")

        # 参数重要性
        if self.param_importance:
            lines.append("-" * 60)
            lines.append("  3. 参数重要性排名 (Top-15)")
            lines.append("-" * 60)
            lines.append(f"  {'排名':<5} {'参数':<40} {'影响力':>8}")
            lines.append("  " + "-" * 53)
            ranked = sorted(self.param_importance.items(),
                            key=lambda x: x[1], reverse=True)
            for rank, (pname, score) in enumerate(ranked[:15], 1):
                lines.append(f"  {rank:<5} {pname:<40} {score:>8.4f}")
            lines.append("")

            # 可固定参数建议
            low_impact = [(p, s) for p, s in ranked if s < 0.05]
            if low_impact:
                lines.append("  [TIP] 可固定为默认值的参数(影响力<0.05):")
                for p, s in low_impact:
                    lines.append(f"     - {p} (影响力={s:.4f})")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, output_dir: str = None):
        """保存报告到文件"""
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / 'logs')
        os.makedirs(output_dir, exist_ok=True)

        ts = time.strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'pca_analysis_{ts}.txt')

        report = self.generate_report()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 同时保存 JSON 结构化数据
        json_path = os.path.join(output_dir, f'pca_analysis_{ts}.json')
        json_data = {
            'lottery_type': self.lottery_type,
            'n_free_params': self.n_free_params,
            'effective_dim_90': getattr(self, 'effective_dim_90', None),
            'effective_dim_95': getattr(self, 'effective_dim_95', None),
            'sensitivity': self.sensitivity_results,
            'param_importance': self.param_importance,
            'dead_params': self.dead_params,
            'timestamp': ts,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n[报告已保存]")
        print(f"  文本: {report_path}")
        print(f"  JSON: {json_path}")
        return report_path


# ============================================================================
#  辅助
# ============================================================================

def _deep_copy_params(params: Dict) -> Dict:
    """深拷贝参数字典"""
    import copy
    return copy.deepcopy(params)


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PCA 参数自由度分析工具')
    parser.add_argument('--lottery', type=str, default='ssq',
                        choices=['ssq', 'dlt'],
                        help='彩票类型 (默认: ssq)')
    parser.add_argument('--samples', type=int, default=150,
                        help='PCA 随机采样数 (默认: 150)')
    parser.add_argument('--importance-trials', type=int, default=80,
                        help='参数重要性评估次数 (默认: 80)')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='跳过敏感性分析（加速）')
    parser.add_argument('--skip-pca', action='store_true',
                        help='跳过 PCA 分析')
    parser.add_argument('--skip-importance', action='store_true',
                        help='跳过重要性分析')
    args = parser.parse_args()

    print("=" * 60)
    print("  PCA 参数自由度分析工具")
    print("=" * 60)
    print(f"  彩票类型: {args.lottery.upper()}")
    print(f"  PCA 采样数: {args.samples}")
    print(f"  重要性评估: {args.importance_trials}")
    print(f"  预计耗时: ~{(args.samples * 2 + args.importance_trials * 2) / 60:.0f} 分钟")
    print("")

    # 初始化
    analyzer = PCAAnalyzer(args.lottery)

    # 加载数据
    df = analyzer.load_data()

    # 分析 1: 敏感性
    if not args.skip_sensitivity:
        analyzer.run_sensitivity_analysis(df)

    # 分析 2: PCA
    if not args.skip_pca:
        analyzer.run_output_pca(df, n_samples=args.samples)

    # 分析 3: 重要性
    if not args.skip_importance:
        analyzer.run_importance_analysis(
            df, n_trials=args.importance_trials)

    # 生成报告
    print("\n" + analyzer.generate_report())
    analyzer.save_report()

    print("\n[OK] PCA 分析完成！")
    print("请查看报告后决定 BO 的维度策略和核函数选型。")


if __name__ == "__main__":
    main()
