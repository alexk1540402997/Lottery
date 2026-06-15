#!/usr/bin/env python3
"""
彩票分析系统优化控制器
功能：自动查找回测结果，生成优化配置、补丁文件和集成指南
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class LotteryOptimizationController:
    """彩票优化控制器 - 完整版本"""

    def __init__(self):
        self.results_df = None
        self.optimized_config = None
        self.project_root = os.getcwd()

    def find_latest_backtest_report(self):
        """查找最新的回测结果文件"""
        reports_dir = "backtest_reports"

        # 检查回测结果目录是否存在
        if not os.path.exists(reports_dir):
            print(f"错误: 回测结果目录 '{reports_dir}' 不存在")
            return None

        # 查找所有回测报告Excel文件
        excel_files = []
        for file in os.listdir(reports_dir):
            if file.endswith('.xlsx') and '回测报告' in file:
                file_path = os.path.join(reports_dir, file)
                excel_files.append((file_path, os.path.getmtime(file_path)))

        if not excel_files:
            print(f"错误: 在 '{reports_dir}' 目录中找不到回测报告文件")
            return None

        # 按修改时间排序，获取最新的文件
        excel_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = excel_files[0][0]

        print(f"找到最新的回测报告: {latest_file}")
        return latest_file

    def load_backtest_results(self, backtest_file):
        """加载回测结果"""
        print(f"加载回测结果: {backtest_file}")

        try:
            with pd.ExcelFile(backtest_file) as xls:
                # 尝试读取不同的工作表名称
                sheet_names = xls.sheet_names

                # 查找包含详细结果的工作表
                detail_sheet = None
                for sheet in sheet_names:
                    if sheet in ['详细结果', '详细数据', '详细结果样本']:
                        detail_sheet = sheet
                        break

                if detail_sheet is None:
                    # 如果没有找到标准名称，尝试第一个工作表
                    detail_sheet = sheet_names[0]

                # 读取数据
                self.results_df = pd.read_excel(xls, sheet_name=detail_sheet)

                # 规范化列名，去除空格
                self.results_df.columns = [str(col).strip() for col in self.results_df.columns]

                print(f"成功加载 {len(self.results_df)} 条回测记录")
                print(f"列名: {list(self.results_df.columns)}")

                return True

        except Exception as e:
            print(f"加载回测结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_method_performance(self):
        """分析方法表现"""
        if self.results_df is None or len(self.results_df) == 0:
            print("错误: 没有可分析的数据")
            return {}

        # 确保必要的列存在
        required_cols = ['方法', '总分', '评估分数']
        for col in required_cols:
            if col not in self.results_df.columns:
                print(f"错误: 数据中缺少列 '{col}'")
                print(f"可用的列: {list(self.results_df.columns)}")
                return {}

        # 按方法分组统计
        method_stats = self.results_df.groupby('方法').agg({
            '总分': ['mean', 'max', 'count', 'std'],
            '评估分数': ['mean', 'max', 'std']
        }).round(3)

        # 按平均评估分排序
        if ('评估分数', 'mean') in method_stats.columns:
            method_stats = method_stats.sort_values(
                ('评估分数', 'mean'), ascending=False
            )

        print("\n" + "=" * 60)
        print("各方法表现统计:")
        print(method_stats.to_string())

        return method_stats

    def analyze_granularity_performance(self):
        """分析颗粒度表现"""
        if self.results_df is None or len(self.results_df) == 0:
            print("错误: 没有可分析的数据")
            return {}

        # 确保必要的列存在
        if '颗粒度' not in self.results_df.columns:
            print("错误: 数据中缺少列 '颗粒度'")
            return {}

        # 按颗粒度分组统计
        granularity_stats = self.results_df.groupby('颗粒度').agg({
            '总分': ['mean', 'max', 'std', 'count'],
            '评估分数': ['mean', 'max', 'std']
        }).round(3)

        # 按平均评估分排序
        if ('评估分数', 'mean') in granularity_stats.columns:
            granularity_stats = granularity_stats.sort_values(
                ('评估分数', 'mean'), ascending=False
            )

        print("\n" + "=" * 60)
        print("各颗粒度表现统计:")
        print(granularity_stats.to_string())

        return granularity_stats

    def find_best_combinations(self, top_n=10):
        """找出最佳的方法-颗粒度组合"""
        if self.results_df is None or len(self.results_df) == 0:
            print("错误: 没有可分析的数据")
            return []

        # 确保必要的列存在
        required_cols = ['方法', '颗粒度', '总分', '评估分数']
        for col in required_cols:
            if col not in self.results_df.columns:
                print(f"错误: 数据中缺少列 '{col}'")
                return []

        # 计算每个组合的平均分
        combinations = []

        for (method, granularity), group in self.results_df.groupby(['方法', '颗粒度']):
            if len(group) < 5:  # 至少有5个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()
            std_score = group['评估分数'].std()
            count = len(group)

            combinations.append({
                'method': str(method),
                'granularity': str(granularity),
                'avg_hit': float(round(avg_hit, 3)),
                'avg_score': float(round(avg_score, 3)),
                'std_score': float(round(std_score, 3) if not pd.isna(std_score) else 0.0),
                'count': int(count)
            })

        # 按平均评估分排序
        combinations.sort(key=lambda x: x['avg_score'], reverse=True)

        # 取前N个
        best_combinations = combinations[:top_n]

        print("\n" + "=" * 60)
        print(f"最佳组合 (前{min(top_n, len(best_combinations))}个):")
        for i, combo in enumerate(best_combinations, 1):
            print(f"{i:2d}. 方法: {combo['method']:<12} "
                  f"颗粒度: {combo['granularity']:<8} "
                  f"平均总分: {combo['avg_hit']:<5.3f} "
                  f"平均评估分: {combo['avg_score']:<6.3f} "
                  f"次数: {combo['count']}")

        return best_combinations

    def calculate_method_weights(self):
        """计算方法权重"""
        if self.results_df is None or len(self.results_df) == 0:
            print("错误: 没有可分析的数据")
            return {}

        # 计算每个方法的平均表现
        method_performance = {}

        for method, group in self.results_df.groupby('方法'):
            if len(group) < 10:  # 至少有10个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()

            # 使用加权综合评分
            # 总分权重0.6，评估分权重0.4
            performance_score = avg_hit * 0.6 + avg_score * 0.4

            method_performance[method] = {
                'avg_hit': float(round(avg_hit, 3)),
                'avg_score': float(round(avg_score, 3)),
                'performance': float(round(performance_score, 3)),
                'count': int(len(group))
            }

        if not method_performance:
            return {}

        # 计算归一化权重
        total_performance = sum(info['performance'] for info in method_performance.values())
        method_weights = {}

        for method, info in method_performance.items():
            weight = info['performance'] / total_performance
            method_weights[method] = {
                'weight': float(round(weight, 4)),
                'avg_hit': info['avg_hit'],
                'avg_score': info['avg_score'],
                'performance': info['performance'],
                'count': info['count']
            }

        # 按权重排序
        sorted_weights = sorted(
            method_weights.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("优化后的方法权重:")
        for method, info in sorted_weights:
            print(f"  {method:<15}: 权重={info['weight']:.4f}, "
                  f"平均分={info['avg_score']:.3f}, "
                  f"平均命中={info['avg_hit']:.3f}")

        return method_weights

    def calculate_granularity_weights(self):
        """计算颗粒度权重"""
        if self.results_df is None or len(self.results_df) == 0:
            print("错误: 没有可分析的数据")
            return {}

        # 计算每个颗粒度的平均表现
        granularity_performance = {}

        for granularity, group in self.results_df.groupby('颗粒度'):
            if len(group) < 10:  # 至少有10个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()

            # 使用加权综合评分
            performance_score = avg_hit * 0.6 + avg_score * 0.4

            granularity_performance[granularity] = {
                'avg_hit': float(round(avg_hit, 3)),
                'avg_score': float(round(avg_score, 3)),
                'performance': float(round(performance_score, 3)),
                'count': int(len(group))
            }

        if not granularity_performance:
            return {}

        # 计算归一化权重
        total_performance = sum(info['performance'] for info in granularity_performance.values())
        granularity_weights = {}

        for granularity, info in granularity_performance.items():
            weight = info['performance'] / total_performance
            granularity_weights[granularity] = {
                'weight': float(round(weight, 4)),
                'avg_hit': info['avg_hit'],
                'avg_score': info['avg_score'],
                'performance': info['performance'],
                'count': info['count']
            }

        # 按权重排序
        sorted_weights = sorted(
            granularity_weights.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("优化后的颗粒度权重:")
        for granularity, info in sorted_weights:
            print(f"  {str(granularity):<10}: 权重={info['weight']:.4f}, "
                  f"平均分={info['avg_score']:.3f}")

        return granularity_weights

    def generate_optimized_config(self):
        """生成优化配置文件"""
        print("\n" + "=" * 60)
        print("生成优化配置文件...")

        # 计算所有统计数据
        method_stats = self.analyze_method_performance()
        granularity_stats = self.analyze_granularity_performance()
        best_combinations = self.find_best_combinations(10)
        method_weights = self.calculate_method_weights()
        granularity_weights = self.calculate_granularity_weights()

        # 汇总统计数据
        summary = {
            'total_records': int(len(self.results_df)),
            'unique_methods': int(self.results_df['方法'].nunique()),
            'unique_granularities': int(self.results_df['颗粒度'].nunique()),
            'period_range': f"{int(self.results_df['期号'].min())}期 到 {int(self.results_df['期号'].max())}期",
            'avg_hit': float(round(self.results_df['总分'].mean(), 3)),
            'avg_score': float(round(self.results_df['评估分数'].mean(), 3)),
            'std_hit': float(round(self.results_df['总分'].std(), 3)),
            'std_score': float(round(self.results_df['评估分数'].std(), 3))
        }

        # 构建优化配置
        self.optimized_config = {
            'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': summary,
            'method_weights': method_weights,
            'granularity_weights': granularity_weights,
            'best_combinations': best_combinations,
            'method_stats': method_stats.to_dict() if not method_stats.empty else {},
            'granularity_stats': granularity_stats.to_dict() if not granularity_stats.empty else {}
        }

        # 保存为JSON文件
        config_file = "optimized_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimized_config, f, ensure_ascii=False, indent=2)

        print(f"优化配置文件已保存: {config_file}")
        return config_file

    def generate_analysis_patch(self):
        """生成分析系统补丁"""
        print("\n" + "=" * 60)
        print("生成分析系统补丁...")

        if not self.optimized_config:
            print("错误: 请先生成优化配置")
            return None

        # 生成补丁代码
        patch_code = '''#!/usr/bin/env python3
"""
彩票分析系统优化补丁
功能：将优化权重集成到分析系统中
使用方法：将此文件中的方法复制到您的分析系统中
"""

import json
import os

def load_optimized_weights():
    """加载优化权重配置"""
    config_file = "optimized_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('method_weights', {})
        except Exception as e:
            print(f"加载优化配置失败: {e}")
            return {}
    return {}

def optimized_comprehensive_recommendation(method_results):
    """
    优化后的综合推荐方法
    使用加权投票，权重来自回测优化结果
    """
    # 加载优化权重
    optimized_weights = load_optimized_weights()

    if not optimized_weights:
        print("警告: 未找到优化权重，使用等权重投票")
        # 使用等权重
        for method in method_results:
            optimized_weights[method] = {'weight': 1.0}

    # 初始化投票箱
    red_votes = {}
    blue_votes = {}

    # 对每个方法的预测结果进行加权投票
    for method_key, result in method_results.items():
        if 'predictions' not in result:
            continue

        predictions = result['predictions']

        # 获取该方法的权重
        weight_info = optimized_weights.get(method_key, {})
        weight = weight_info.get('weight', 1.0)

        # 处理红球预测
        reds = predictions.get('red', [])
        if isinstance(reds, list):
            for num in reds:
                if 1 <= num <= 33:  # 双色球红球范围
                    red_votes[num] = red_votes.get(num, 0) + weight

        # 处理蓝球预测
        blues = predictions.get('blue', [])
        if isinstance(blues, list):
            for num in blues:
                if 1 <= num <= 16:  # 双色球蓝球范围
                    blue_votes[num] = blue_votes.get(num, 0) + weight

    # 选择得票最高的红球（前6个）
    recommended_reds = []
    if red_votes:
        # 按票数排序
        sorted_reds = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)
        # 取前6个
        top_reds = sorted_reds[:6]
        recommended_reds = [num for num, _ in top_reds]
    else:
        recommended_reds = []

    # 选择得票最高的蓝球（前1个）
    recommended_blues = []
    if blue_votes:
        # 按票数排序
        sorted_blues = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)
        # 取前1个
        top_blues = sorted_blues[:1]
        recommended_blues = [num for num, _ in top_blues]
    else:
        recommended_blues = []

    # 返回优化后的推荐结果
    return {
        'red': recommended_reds,
        'blue': recommended_blues,
        'method': 'optimized_weighted_vote',
        'weights_used': {k: v.get('weight', 1.0) for k, v in optimized_weights.items() if k in method_results}
    }

# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_method_results = {
        'method_1': {
            'predictions': {
                'red': [1, 5, 12, 18, 25, 30],
                'blue': [7]
            }
        },
        'method_2': {
            'predictions': {
                'red': [5, 12, 18, 22, 25, 33],
                'blue': [7]
            }
        }
    }

    # 运行优化推荐
    result = optimized_comprehensive_recommendation(test_method_results)
    print("优化推荐结果:")
    print(f"红球: {result['red']}")
    print(f"蓝球: {result['blue']}")
    print(f"使用的权重: {result['weights_used']}")'''

        # 保存补丁文件
        patch_file = "analysis_patch.py"
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(patch_code)

        print(f"分析系统补丁已生成: {patch_file}")
        return patch_file

    def generate_merger_patch(self):
        """生成合并系统补丁"""
        print("\n" + "=" * 60)
        print("生成合并系统补丁...")

        if not self.optimized_config:
            print("错误: 请先生成优化配置")
            return None

        # 生成补丁代码
        patch_code = '''#!/usr/bin/env python3
"""
彩票合并系统优化补丁
功能：将优化权重集成到合并系统中
使用方法：将此文件中的方法复制到您的合并系统中
"""

import json
import os
import pandas as pd
from datetime import datetime

def load_optimized_config():
    """加载优化配置"""
    config_file = "optimized_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载优化配置失败: {e}")
    return {}

def optimized_create_final_recommendation(file_info):
    """
    优化后的最终推荐方法
    使用加权统计，权重来自回测优化结果
    """
    # 加载优化配置
    config = load_optimized_config()
    method_weights = config.get('method_weights', {})

    if not method_weights:
        print("警告: 未找到优化权重，使用等权重统计")

    # 初始化加权统计
    red_weighted_counts = {}
    blue_weighted_counts = {}

    # 处理每个文件
    for info in file_info:
        file_path = info.get('path', '')
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        try:
            # 读取Excel文件
            excel_data = pd.read_excel(file_path, sheet_name=None)

            # 查找预测汇总工作表
            predictions_sheet = None
            for sheet_name in excel_data.keys():
                if "预测汇总" in sheet_name or "predictions" in sheet_name.lower():
                    predictions_sheet = sheet_name
                    break

            if predictions_sheet is None:
                # 如果没有找到预测汇总，尝试第一个工作表
                predictions_sheet = list(excel_data.keys())[0]

            df_predictions = excel_data[predictions_sheet]

            # 规范化列名
            df_predictions.columns = [str(col).strip() for col in df_predictions.columns]

            # 处理每一行预测
            for _, row in df_predictions.iterrows():
                # 获取方法名称
                method_name = None
                for col in df_predictions.columns:
                    if '方法' in col or 'method' in col.lower():
                        method_name = str(row[col]) if pd.notna(row[col]) else '未知'
                        break

                if method_name is None:
                    method_name = '未知'

                # 获取权重
                weight = 1.0
                if method_name in method_weights:
                    weight = method_weights[method_name].get('weight', 1.0)

                # 处理红球预测
                red_col = None
                for col in df_predictions.columns:
                    if '红球' in col or 'red' in col.lower():
                        red_col = col
                        break

                if red_col and red_col in row and pd.notna(row[red_col]):
                    red_balls = str(row[red_col]).strip()
                    if red_balls:
                        for num_str in red_balls.split():
                            try:
                                num = int(float(num_str))  # 处理可能的浮点数
                                if 1 <= num <= 33:
                                    red_weighted_counts[num] = red_weighted_counts.get(num, 0) + weight
                            except (ValueError, TypeError):
                                continue

                # 处理蓝球预测
                blue_col = None
                for col in df_predictions.columns:
                    if '蓝球' in col or 'blue' in col.lower():
                        blue_col = col
                        break

                if blue_col and blue_col in row and pd.notna(row[blue_col]):
                    blue_balls = str(row[blue_col]).strip()
                    if blue_balls:
                        for num_str in blue_balls.split():
                            try:
                                num = int(float(num_str))  # 处理可能的浮点数
                                if 1 <= num <= 16:
                                    blue_weighted_counts[num] = blue_weighted_counts.get(num, 0) + weight
                            except (ValueError, TypeError):
                                continue

        except Exception as e:
            print(f"处理文件 {file_path} 失败: {e}")
            continue

    # 生成最终推荐
    final_data = []
    final_data.append(["优化后的最终推荐", ""])
    final_data.append(["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    final_data.append(["优化策略", "加权统计（基于回测表现）"])

    # 红球推荐
    if red_weighted_counts:
        # 按权重排序
        sorted_reds = sorted(red_weighted_counts.items(), key=lambda x: x[1], reverse=True)

        # 主选6个红球
        main_reds = [num for num, _ in sorted_reds[:6]]
        main_reds_str = ' '.join(f'{num:02d}' for num in sorted(main_reds))
        final_data.append(["红球推荐", main_reds_str])

        # 候补4个红球
        if len(sorted_reds) > 6:
            backup_reds = [num for num, _ in sorted_reds[6:10]]
            backup_reds_str = ' '.join(f'{num:02d}' for num in sorted(backup_reds))
            final_data.append(["红球候补", backup_reds_str])
        else:
            final_data.append(["红球候补", "无"])
    else:
        final_data.append(["红球推荐", "暂无数据"])

    # 蓝球推荐
    if blue_weighted_counts:
        # 按权重排序
        sorted_blues = sorted(blue_weighted_counts.items(), key=lambda x: x[1], reverse=True)

        # 主选1个蓝球
        main_blues = [num for num, _ in sorted_blues[:1]]
        main_blues_str = ' '.join(f'{num:02d}' for num in sorted(main_blues))
        final_data.append(["蓝球推荐", main_blues_str])

        # 候补2个蓝球
        if len(sorted_blues) > 1:
            backup_blues = [num for num, _ in sorted_blues[1:3]]
            backup_blues_str = ' '.join(f'{num:02d}' for num in sorted(backup_blues))
            final_data.append(["蓝球候补", backup_blues_str])
        else:
            final_data.append(["蓝球候补", "无"])
    else:
        final_data.append(["蓝球推荐", "暂无数据"])

    return final_data

# 使用示例
if __name__ == "__main__":
    # 注意：这里需要实际有这些测试文件才能运行
    # 为了演示，我们创建一个虚拟的最终推荐
    print("合并系统优化补丁测试")
    print("=" * 50)

    # 模拟优化配置
    test_config = {
        'method_weights': {
            '统计概率分析': {'weight': 0.3},
            '时间序列分析': {'weight': 0.25},
            '综合推荐': {'weight': 0.45}
        }
    }

    # 保存测试配置
    with open('optimized_config.json', 'w', encoding='utf-8') as f:
        json.dump(test_config, f, ensure_ascii=False, indent=2)

    print("优化配置已创建，可以运行优化后的合并系统")'''

        # 保存补丁文件
        patch_file = "merger_patch.py"
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(patch_code)

        print(f"合并系统补丁已生成: {patch_file}")
        return patch_file

    def generate_integration_guide(self):
        """生成集成指南"""
        print("\n" + "=" * 60)
        print("生成集成指南...")

        # 创建集成指南内容
        guide_content = '''# 彩票分析系统优化集成指南

## 文件结构
