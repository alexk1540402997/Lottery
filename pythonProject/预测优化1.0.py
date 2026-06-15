"""
彩票分析模型优化系统 - 简洁版
功能：基于回测结果优化分析模型参数
目标：在功能完备的基础上保持简洁（<1500行）
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class LotteryModelOptimizer:
    """彩票模型优化器 - 基于回测结果进行优化"""

    def __init__(self):
        self.results_df = None
        self.optimized_weights = {}
        self.best_combinations = []

    def load_recent_results(self, results_file=None):
        """加载最近的回测结果"""
        if results_file is None:
            # 自动查找最新的回测结果文件
            results_dir = "backtest_reports"
            if not os.path.exists(results_dir):
                print("回测结果目录不存在")
                return False

            excel_files = [f for f in os.listdir(results_dir)
                           if f.endswith('.xlsx') and '回测报告' in f]

            if not excel_files:
                print("未找到回测结果文件")
                return False

            # 按时间排序，取最新的
            excel_files.sort(reverse=True)
            results_file = os.path.join(results_dir, excel_files[0])

        print(f"加载回测结果: {results_file}")

        try:
            # 读取Excel文件
            with pd.ExcelFile(results_file) as xls:
                # 读取详细结果
                if '详细结果' in xls.sheet_names:
                    self.results_df = pd.read_excel(xls, sheet_name='详细结果')
                elif '详细数据' in xls.sheet_names:
                    self.results_df = pd.read_excel(xls, sheet_name='详细数据')
                else:
                    # 尝试读取第一个工作表
                    self.results_df = pd.read_excel(xls, sheet_name=0)

                print(f"成功加载 {len(self.results_df)} 条回测记录")
                return True

        except Exception as e:
            print(f"加载回测结果失败: {e}")
            return False

    def analyze_method_performance(self):
        """分析方法表现"""
        if self.results_df is None or len(self.results_df) == 0:
            print("没有可分析的数据")
            return {}

        # 按方法分组统计
        method_stats = self.results_df.groupby('方法').agg({
            '总分': ['mean', 'max', 'count'],
            '评估分数': ['mean', 'max']
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
            print("没有可分析的数据")
            return {}

        # 按颗粒度分组统计
        granularity_stats = self.results_df.groupby('颗粒度').agg({
            '总分': ['mean', 'max', 'std'],
            '评估分数': ['mean', 'max']
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
            print("没有可分析的数据")
            return []

        # 计算每个组合的平均分
        combinations = []

        for (method, granularity), group in self.results_df.groupby(['方法', '颗粒度']):
            if len(group) < 10:  # 至少有10个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()
            count = len(group)

            combinations.append({
                '方法': method,
                '颗粒度': granularity,
                '平均总分': round(avg_hit, 3),
                '平均评估分': round(avg_score, 3),
                '评估次数': count
            })

        # 按平均评估分排序
        combinations.sort(key=lambda x: x['平均评估分'], reverse=True)

        # 取前N个
        self.best_combinations = combinations[:top_n]

        print("\n" + "=" * 60)
        print(f"最佳组合 (前{top_n}个):")
        for i, combo in enumerate(self.best_combinations, 1):
            print(f"{i:2d}. 方法: {combo['方法']:<10} "
                  f"颗粒度: {combo['颗粒度']:<8} "
                  f"平均总分: {combo['平均总分']:<5} "
                  f"平均评估分: {combo['平均评估分']:<6} "
                  f"次数: {combo['评估次数']}")

        return self.best_combinations

    def optimize_method_weights(self):
        """优化方法权重（基于表现）"""
        if self.results_df is None or len(self.results_df) == 0:
            print("没有可分析的数据")
            return {}

        # 计算每个方法的平均表现
        method_performance = {}

        for method, group in self.results_df.groupby('方法'):
            if len(group) < 20:  # 至少有20个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()

            # 使用综合评分（总分权重0.6，评估分权重0.4）
            performance_score = avg_hit * 0.6 + avg_score * 0.4

            method_performance[method] = performance_score

        if not method_performance:
            return {}

        # 归一化权重
        total_score = sum(method_performance.values())
        self.optimized_weights = {
            method: score / total_score
            for method, score in method_performance.items()
        }

        # 按权重排序
        sorted_weights = sorted(
            self.optimized_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("优化后的方法权重:")
        for method, weight in sorted_weights:
            print(f"  {method:<15}: {weight:.4f}")

        return self.optimized_weights

    def optimize_granularity_weights(self):
        """优化颗粒度权重（基于表现）"""
        if self.results_df is None or len(self.results_df) == 0:
            print("没有可分析的数据")
            return {}

        # 计算每个颗粒度的平均表现
        granularity_performance = {}

        for granularity, group in self.results_df.groupby('颗粒度'):
            if len(group) < 20:  # 至少有20个评估结果
                continue

            avg_hit = group['总分'].mean()
            avg_score = group['评估分数'].mean()

            # 使用综合评分
            performance_score = avg_hit * 0.6 + avg_score * 0.4

            granularity_performance[granularity] = performance_score

        if not granularity_performance:
            return {}

        # 归一化权重
        total_score = sum(granularity_performance.values())
        optimized_weights = {
            granularity: score / total_score
            for granularity, score in granularity_performance.items()
        }

        # 按权重排序
        sorted_weights = sorted(
            optimized_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "=" * 60)
        print("优化后的颗粒度权重:")
        for granularity, weight in sorted_weights:
            print(f"  {granularity:<10}: {weight:.4f}")

        return optimized_weights

    def generate_optimization_report(self, output_dir="optimization_reports"):
        """生成优化报告"""
        if self.results_df is None or len(self.results_df) == 0:
            print("没有可分析的数据")
            return ""

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"模型优化报告_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. 方法表现
            method_stats = self.analyze_method_performance()
            if not method_stats.empty:
                method_stats.to_excel(writer, sheet_name="方法表现")

            # 2. 颗粒度表现
            granularity_stats = self.analyze_granularity_performance()
            if not granularity_stats.empty:
                granularity_stats.to_excel(writer, sheet_name="颗粒度表现")

            # 3. 最佳组合
            best_combinations = self.find_best_combinations(20)
            if best_combinations:
                df_best = pd.DataFrame(best_combinations)
                df_best.to_excel(writer, sheet_name="最佳组合", index=False)

            # 4. 优化权重
            if self.optimized_weights:
                df_weights = pd.DataFrame(
                    list(self.optimized_weights.items()),
                    columns=['方法', '优化权重']
                )
                df_weights.to_excel(writer, sheet_name="方法权重", index=False)

            # 5. 摘要
            summary_data = [
                ["模型优化报告摘要", ""],
                ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["回测记录数", len(self.results_df)],
                ["分析方法数", self.results_df['方法'].nunique()],
                ["颗粒度数", self.results_df['颗粒度'].nunique()],
                ["期数范围", f"{self.results_df['期号'].min()}期 到 {self.results_df['期号'].max()}期"],
                ["平均总分", f"{self.results_df['总分'].mean():.3f}"],
                ["平均评估分", f"{self.results_df['评估分数'].mean():.3f}"],
                ["", ""],
                ["优化建议", ""]
            ]

            # 添加优化建议
            if best_combinations:
                best_combo = best_combinations[0]
                summary_data.append([
                    "推荐使用组合",
                    f"方法: {best_combo['方法']}, 颗粒度: {best_combo['颗粒度']}"
                ])
                summary_data.append([
                    "预期平均总分",
                    f"{best_combo['平均总分']:.3f}"
                ])
                summary_data.append([
                    "预期平均评估分",
                    f"{best_combo['平均评估分']:.3f}"
                ])

            df_summary = pd.DataFrame(summary_data, columns=["项目", "值"])
            df_summary.to_excel(writer, sheet_name="优化摘要", index=False)

        print(f"\n优化报告已保存: {filepath}")
        return filepath

    def get_optimization_recommendations(self):
        """获取优化建议"""
        if not self.best_combinations:
            self.find_best_combinations(5)

        if not self.best_combinations:
            return []

        recommendations = []

        for i, combo in enumerate(self.best_combinations[:3], 1):
            recommendations.append({
                '排名': i,
                '方法': combo['方法'],
                '颗粒度': combo['颗粒度'],
                '预期命中': f"{combo['平均总分']:.2f}个号码",
                '预期得分': f"{combo['平均评估分']:.2f}分",
                '稳定性': f"基于{combo['评估次数']}次评估"
            })

        return recommendations

    def quick_optimization(self, results_file=None):
        """快速优化流程"""
        print("开始快速优化分析...")

        # 1. 加载回测结果
        if not self.load_recent_results(results_file):
            return False

        # 2. 分析方法表现
        method_stats = self.analyze_method_performance()

        # 3. 分析颗粒度表现
        granularity_stats = self.analyze_granularity_performance()

        # 4. 找出最佳组合
        best_combinations = self.find_best_combinations(10)

        # 5. 优化权重
        self.optimize_method_weights()
        self.optimize_granularity_weights()

        # 6. 生成报告
        report_file = self.generate_optimization_report()

        # 7. 输出建议
        recommendations = self.get_optimization_recommendations()

        print("\n" + "=" * 60)
        print("优化完成！")
        print("=" * 60)

        if recommendations:
            print("\nTOP 3 推荐组合:")
            for rec in recommendations:
                print(f"  {rec['排名']}. {rec['方法']} + {rec['颗粒度']}")
                print(f"     预期命中: {rec['预期命中']}, 预期得分: {rec['预期得分']}")
                print(f"     稳定性: {rec['稳定性']}")

        return True


def main():
    """主函数"""
    optimizer = LotteryModelOptimizer()

    # 获取命令行参数
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = None

    # 运行快速优化
    success = optimizer.quick_optimization(results_file)

    if not success:
        print("\n优化失败，请检查回测结果文件路径")


if __name__ == "__main__":
    main()