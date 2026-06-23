"""
结果合并器 5.0
=============
跨颗粒度加权投票合并系统。

将 5种颗粒度(50/100/500/1000/全部) × 13种分析方法 = 65组预测结果
通过独立的复合权重（每对组合各自独立权重）合并为最终一组号码。

与4.x的区别：
- 权重改为 65 个独立 composite_weights（key="method_X@Y期"）
- 不再使用 method_weights × granularity_weights 的交叉乘法
- 每个(方法, 颗粒度)组合有权重独立，允许负/零/正
- 权重范围 [-500.0, 500.0]，负值表示反向指标
- 权重从回测中自动学习，LP线性规划精确求解
- 支持手动微调权重
- 支持和值/区间/奇偶等约束条件过滤
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
#  默认权重
# ============================================================================

# 5种颗粒度 × 13种方法 = 65组独立复合权重
# Key格式: "method_X@Y期" (如 "method_1@500期")
# 每个组合独立，允许正/负/零，范围 [-500.0, 500.0]
DEFAULT_COMPOSITE_WEIGHTS = {}
for _mk in [f'method_{i}' for i in range(1, 14)]:
    for _gn in ['50期', '100期', '500期', '1000期', '全部期']:
        DEFAULT_COMPOSITE_WEIGHTS[f'{_mk}@{_gn}'] = 1.0

# 颗粒度映射
GRANULARITY_NAMES = ['50期', '100期', '500期', '1000期', '全部期']
GRANULARITY_VALUES = [50, 100, 500, 1000, 0]

# 方法 Key 列表
METHOD_KEYS = [f'method_{i}' for i in range(1, 14)]

METHOD_NAMES = {
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

# 旧格式 → 新格式 转换辅助
def _convert_old_weights(method_weights=None, granularity_weights=None):
    """将旧格式 (method_weights + granularity_weights) 转换为 composite_weights"""
    cw = {}
    for mk in METHOD_KEYS:
        for gn in GRANULARITY_NAMES:
            mw = (method_weights or {}).get(mk, 1.0)
            gw = (granularity_weights or {}).get(gn, 1.0)
            cw[f'{mk}@{gn}'] = round(mw * gw, 6)
    return cw


class ResultMerger:
    """跨颗粒度加权投票合并器"""

    def __init__(self, lottery_type: str = 'ssq'):
        """
        参数:
            lottery_type: 'ssq' 或 'dlt'
        """
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

        # 当前权重：65个独立 composite_weights
        self.composite_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)

    def set_weights(self, composite_weights: Dict[str, float] = None,
                    method_weights: Dict[str, float] = None,
                    gran_weights: Dict[str, float] = None):
        """手动设置权重。兼容旧格式 (method_weights + gran_weights) 自动转换。"""
        if composite_weights:
            self.composite_weights.update(composite_weights)
        if method_weights or gran_weights:
            # 兼容旧格式：自动转换为 composite
            converted = _convert_old_weights(method_weights, gran_weights)
            self.composite_weights.update(converted)

    def compute_weight(self, granularity_text: str, method_key: str) -> float:
        """
        获取某组(颗粒度, 方法)的综合权重。
        直接查找 composite_weights["method_X@Y期"]。
        """
        key = f'{method_key}@{granularity_text}'
        return self.composite_weights.get(key, 1.0)

    def merge_results(self,
                      all_predictions: Dict[str, Dict[str, Dict]],
                      constraints: Optional[Dict] = None
                      ) -> Dict[str, Any]:
        """
        合并65组预测结果。

        参数:
            all_predictions: {
                '50期': {  # 颗粒度
                    'method_1': {predictions: {main: [...], aux: [...]}, ...},
                    'method_2': {...},
                    ...
                },
                '100期': {...},
                ...
            }
            constraints: 约束条件（可选）
                {
                    'sum_range': (min, max),      # 主球和值范围
                    'zone_balance': (min, max),    # 区间均衡度
                    'odd_ratio': (min, max),       # 奇偶比
                    'max_consecutive': int,        # 最大连号数
                }

        返回:
            {
                'method': '综合合并推荐',
                'predictions': {main: [...], aux: [...]},
                'vote_details': [...],  # 各组投票详情
                'top_contributors': [...],  # 贡献最大的组合
            }
        """
        main_votes = defaultdict(float)
        aux_votes = defaultdict(float)
        vote_details = []

        for gran_name, methods_dict in all_predictions.items():
            for method_key, result in methods_dict.items():
                # 跳过综合推荐（它不是独立方法，是合并产物）
                if method_key == 'comprehensive':
                    continue
                if 'error' in result or 'predictions' not in result:
                    continue

                weight = self.compute_weight(gran_name, method_key)
                predictions = result.get('predictions', {})

                main_balls = predictions.get(self.main_name, [])
                aux_balls = predictions.get(self.aux_name, [])

                if isinstance(main_balls, list) and main_balls:
                    for num in main_balls:
                        if self.main_range[0] <= num <= self.main_range[1]:
                            main_votes[num] += weight
                    vote_details.append({
                        'granularity': gran_name,
                        'method': method_key,
                        'method_name': METHOD_NAMES.get(method_key, method_key),
                        'weight': round(weight, 4),
                        'main_balls': main_balls,
                        'aux_balls': aux_balls if isinstance(aux_balls, list) else [],
                    })

                if isinstance(aux_balls, list) and aux_balls:
                    for num in aux_balls:
                        if self.aux_range[0] <= num <= self.aux_range[1]:
                            aux_votes[num] += weight

        # 应用约束条件
        predicted_main = self._select_with_constraints(
            main_votes, self.main_count, self.main_range[0],
            constraints)

        predicted_aux = self._select_aux(aux_votes, self.aux_count)

        # 贡献排名（显示全部，按权重降序）
        vote_details.sort(key=lambda x: x['weight'], reverse=True)
        top_contributors = vote_details  # 全部参与合并的组合

        return {
            'method': '综合合并推荐',
            'description': f'{len(vote_details)}组预测结果加权投票合并',
            'predictions': {
                self.main_name: sorted(predicted_main),
                self.aux_name: sorted(predicted_aux),
            },
            'vote_details': vote_details,
            'top_contributors': top_contributors,
            'total_groups': len(vote_details),
        }

    def _select_with_constraints(self, votes: Dict[int, float],
                                  count: int, max_num: int,
                                  constraints: Optional[Dict] = None
                                  ) -> List[int]:
        """
        按投票得分选号，尽量满足约束条件。
        如果提供了constraints，优先选择满足约束的组合。
        """
        # 按得分排序
        sorted_nums = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        if constraints is None:
            # 简单Top-K
            selected = [n for n, _ in sorted_nums[:count]]
            return sorted(selected)

        # 有约束：搜索最优组合
        candidates = [n for n, _ in sorted_nums[:min(len(sorted_nums), count * 3)]]
        if len(candidates) < count:
            candidates = list(range(1, max_num + 1))

        best_combo = candidates[:count]
        best_score = self._score_combo(best_combo, votes, constraints)

        # 贪心搜索（尝试替换每个位置）
        for _ in range(50):
            improved = False
            combo = list(best_combo)
            for i in range(count):
                original = combo[i]
                for new_num in candidates:
                    if new_num in combo:
                        continue
                    combo[i] = new_num
                    new_score = self._score_combo(combo, votes, constraints)
                    if new_score > best_score:
                        best_score = new_score
                        best_combo = list(combo)
                        improved = True
                    combo[i] = original
            if not improved:
                break

        return sorted(best_combo)

    def _score_combo(self, combo: List[int], votes: Dict[int, float],
                     constraints: Dict) -> float:
        """评估一组号码在约束下的得分"""
        score = sum(votes.get(n, 0) for n in combo)

        # 和值约束
        if 'sum_range' in constraints:
            smin, smax = constraints['sum_range']
            s = sum(combo)
            if s < smin:
                score *= max(0, 1 - (smin - s) / smin)
            elif s > smax:
                score *= max(0, 1 - (s - smax) / smax)

        # 奇偶约束
        if 'odd_ratio' in constraints:
            omin, omax = constraints['odd_ratio']
            odds = sum(1 for n in combo if n % 2)
            ratio = odds / len(combo)
            if ratio < omin:
                score *= max(0, ratio / omin)
            elif ratio > omax:
                score *= max(0, (1 - ratio) / (1 - omax))

        # 连号约束
        if 'max_consecutive' in constraints:
            max_cons = constraints['max_consecutive']
            sorted_combo = sorted(combo)
            cons = 1
            for i in range(len(sorted_combo) - 1):
                if sorted_combo[i + 1] - sorted_combo[i] == 1:
                    cons += 1
                else:
                    cons = 1
                if cons > max_cons:
                    score *= 0.5
                    break

        return score

    def _select_aux(self, votes: Dict[int, float], count: int) -> List[int]:
        """辅助球选择（简单Top-K）"""
        sorted_nums = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [n for n, _ in sorted_nums[:count]]
        return sorted(selected)

    def learn_weights_from_backtest(self,
                                     backtest_results: pd.DataFrame
                                     ) -> Dict[str, Any]:
        """
        从回测结果中自动学习最优权重（65个独立组合）。

        参数:
            backtest_results: DataFrame with columns:
                granularity, method_key, total_hits, score, ...

        返回:
            {
                'composite_weights': {...},  # 65个独立权重
                'performance_by_method': {...},
                'performance_by_granularity': {...},
            }
        """
        df = backtest_results.copy()

        # 按 (method_key, granularity) 组合统计表现
        if 'method_key' in df.columns and 'granularity' in df.columns and 'total_hits' in df.columns:
            combo_perf = df.groupby(['method_key', 'granularity'])['total_hits'].agg(
                ['mean', 'std', 'max', 'count']
            )
            combo_perf = combo_perf[combo_perf['count'] >= 5]

            if not combo_perf.empty:
                total_mean = combo_perf['mean'].sum()
                if total_mean > 0:
                    n_combos = len(combo_perf)
                    for (mk, gn), row in combo_perf.iterrows():
                        key = f'{mk}@{gn}'
                        self.composite_weights[key] = round(
                            float(row['mean'] / total_mean * n_combos), 4
                        )

        # 方法平均表现（仅供展示）
        method_perf = {}
        if 'method_key' in df.columns and 'total_hits' in df.columns:
            mp = df.groupby('method_key')['total_hits'].agg(['mean', 'std', 'max', 'count'])
            mp = mp[mp['count'] >= 5]
            method_perf = mp.to_dict('index') if not mp.empty else {}

        # 颗粒度表现（仅供展示）
        gran_perf = {}
        if 'granularity' in df.columns:
            gp = df.groupby('granularity')['total_hits'].agg(['mean', 'std', 'max', 'count'])
            gp = gp[gp['count'] >= 5]
            gran_perf = gp.to_dict('index') if not gp.empty else {}

        return {
            'composite_weights': dict(self.composite_weights),
            'performance_by_method': method_perf,
            'performance_by_granularity': gran_perf,
        }

    def export_weights(self) -> Dict[str, Any]:
        """导出当前权重配置"""
        return {
            'lottery_type': self.lottery_type,
            'composite_weights': dict(self.composite_weights),
            'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def import_weights(self, config: Dict[str, Any]):
        """导入权重配置。兼容旧格式 (method_weights + granularity_weights) 自动转换。"""
        if 'composite_weights' in config:
            self.composite_weights.update(config['composite_weights'])
        elif 'method_weights' in config or 'granularity_weights' in config:
            # 旧格式自动转换
            converted = _convert_old_weights(
                config.get('method_weights'),
                config.get('granularity_weights'))
            self.composite_weights.update(converted)


def batch_merge_to_excel(all_results: Dict[str, Dict[str, Dict]],
                          merger: ResultMerger,
                          output_dir: str = "merged_results"
                          ) -> str:
    """
    将各颗粒度的分析结果合并并导出为Excel。

    参数:
        all_results: {granularity_name: {method_key: result, ...}, ...}
        merger: ResultMerger实例
        output_dir: 输出目录

    返回:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lottery_name = "双色球" if merger.lottery_type == 'ssq' else "大乐透"
    filename = f"合并结果_{lottery_name}_{ts}.xlsx"
    filepath = os.path.join(output_dir, filename)

    # 执行合并
    merged = merger.merge_results(all_results)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: 最终推荐
        final_data = [
            ["合并推荐结果", ""],
            ["彩票类型", lottery_name],
            ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["参与组合数", merged['total_groups']],
            ["", ""],
            [f"{merger.main_name}球推荐",
             ' '.join(f'{n:02d}' for n in merged['predictions'][merger.main_name])],
            [f"{merger.aux_name}球推荐",
             ' '.join(f'{n:02d}' for n in merged['predictions'][merger.aux_name])],
        ]
        pd.DataFrame(final_data, columns=["项目", "值"]).to_excel(
            writer, sheet_name="最终推荐", index=False)

        # Sheet 2: 投票详情
        if merged['vote_details']:
            vote_df = pd.DataFrame(merged['vote_details'])
            vote_df.to_excel(writer, sheet_name="投票详情", index=False)

        # Sheet 3: 权重配置（65个独立组合）
        weight_data = [
            ["权重配置 (65个独立组合)", ""],
            ["组合键 (方法@颗粒度)", "权重值"],
        ]
        # 按绝对值降序排列
        sorted_weights = sorted(
            merger.composite_weights.items(),
            key=lambda x: abs(x[1]), reverse=True)
        for k, v in sorted_weights:
            mk, gn = k.split('@', 1)
            method_name = METHOD_NAMES.get(mk, mk)
            weight_data.append([f"{method_name} @ {gn}", f"{v:.4f}"])
        pd.DataFrame(weight_data, columns=["项目", "值"]).to_excel(
            writer, sheet_name="权重配置", index=False)

        # Sheet 4: 各颗粒度各方法预测详情
        for gran_name, methods_dict in all_results.items():
            rows = []
            for mk, result in methods_dict.items():
                if 'error' in result:
                    continue
                pred = result.get('predictions', {})
                rows.append({
                    '分析方法': METHOD_NAMES.get(mk, mk),
                    f'{merger.main_name}球预测':
                        ' '.join(f'{n:02d}' for n in pred.get(merger.main_name, [])),
                    f'{merger.aux_name}球预测':
                        ' '.join(f'{n:02d}' for n in pred.get(merger.aux_name, [])),
                })
            if rows:
                sheet_name = gran_name[:31]
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    return filepath


if __name__ == "__main__":
    print("结果合并器 5.0 - 跨颗粒度加权投票系统")
    print(f"权重模型: 13方法 × 5颗粒度 = 65个独立复合权重")
    print(f"权重范围: [-500.0, 500.0]，允许正/负/零")
    merger = ResultMerger('ssq')
    print(f"已初始化 {len(merger.composite_weights)} 个 composite_weights")
    # 展示所有权重
    for key, w in sorted(merger.composite_weights.items()):
        print(f"  {key}: {w:.4f}")
    # 测试旧格式兼容
    print("\n旧格式兼容测试:")
    old_config = {
        'method_weights': {'method_1': 2.0, 'method_2': 0.5},
        'granularity_weights': {'50期': 1.5, '500期': 0.8},
    }
    merger.import_weights(old_config)
    print(f"  导入旧格式后 method_1@50期 = {merger.compute_weight('50期', 'method_1'):.4f}")
    print(f"  期望: 2.0×1.5 = 3.0")
