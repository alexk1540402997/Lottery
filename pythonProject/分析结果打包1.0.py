"""
彩票分析结果合并工具
功能：将多个颗粒度分析结果Excel文件合并到一个Excel文件中
每个颗粒度的所有分析结果放在一个工作表中
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import warnings

warnings.filterwarnings('ignore')


class LotteryResultsMerger:
    """彩票分析结果合并器"""

    def __init__(self):
        self.selected_files = []
        self.output_dir = "merged_results"

    def select_files(self) -> list:
        """选择多个Excel文件"""
        file_paths = filedialog.askopenfilenames(
            title="选择要合并的Excel文件（可多选）",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
            initialdir="analysis_results"
        )
        return list(file_paths) if file_paths else []

    def extract_granularity_from_filename(self, filename: str) -> str:
        """从文件名中提取颗粒度信息"""
        # 常见的颗粒度关键词
        granularity_keywords = ["50期", "100期", "500期", "1000期", "全部期"]

        for keyword in granularity_keywords:
            if keyword in filename:
                return keyword

        # 如果没有匹配到关键词，返回文件名的一部分
        base_name = os.path.basename(filename)
        # 去除扩展名
        name_without_ext = os.path.splitext(base_name)[0]
        # 尝试从文件名中提取颗粒度
        if "双色球" in name_without_ext or "大乐透" in name_without_ext:
            # 假设格式为: 彩票类型_分析结果_颗粒度_时间戳
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                for part in parts[2:]:  # 从第三个部分开始检查
                    for keyword in granularity_keywords:
                        if keyword in part:
                            return keyword

        return "未知颗粒度"

    def merge_excel_files(self, file_paths: list, output_dir: str = "merged_results") -> tuple:
        """
        合并多个Excel文件到一个新的Excel文件
        返回: (成功标志, 消息, 输出文件路径)
        """
        if not file_paths:
            return False, "没有选择任何文件", ""

        # 按颗粒度排序
        def get_granularity_order(granularity: str) -> int:
            """获取颗粒度排序顺序"""
            order_map = {
                "50期": 1,
                "最近50期": 1,
                "100期": 2,
                "最近100期": 2,
                "500期": 3,
                "最近500期": 3,
                "1000期": 4,
                "最近1000期": 4,
                "全部期": 5
            }
            return order_map.get(granularity, 99)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"合并分析结果_{timestamp}.xlsx"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 创建Excel写入器
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 处理每个文件
                file_info = []
                for file_path in file_paths:
                    if not os.path.exists(file_path):
                        print(f"文件不存在: {file_path}")
                        continue

                    try:
                        # 提取颗粒度
                        granularity = self.extract_granularity_from_filename(file_path)

                        # 读取Excel文件中的所有工作表
                        excel_file = pd.ExcelFile(file_path)
                        sheet_names = excel_file.sheet_names

                        if not sheet_names:
                            print(f"文件没有工作表: {file_path}")
                            continue

                        # 获取彩票类型
                        lottery_type = "未知"
                        if "双色球" in file_path:
                            lottery_type = "双色球"
                        elif "大乐透" in file_path:
                            lottery_type = "大乐透"

                        file_info.append({
                            'path': file_path,
                            'granularity': granularity,
                            'lottery_type': lottery_type,
                            'sheet_names': sheet_names
                        })

                        print(f"处理文件: {os.path.basename(file_path)}, 颗粒度: {granularity}")

                    except Exception as e:
                        print(f"读取文件失败 {file_path}: {str(e)}")
                        continue

                if not file_info:
                    return False, "没有找到有效的Excel文件", ""

                # 按颗粒度排序
                file_info.sort(key=lambda x: get_granularity_order(x['granularity']))

                # 创建一个摘要工作表
                summary_data = []
                summary_data.append(["合并分析结果摘要", ""])
                summary_data.append(["合并时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                summary_data.append(["合并文件数", len(file_info)])
                summary_data.append(["", ""])
                summary_data.append(["文件列表", ""])

                for i, info in enumerate(file_info, 1):
                    summary_data.append([f"文件{i}", os.path.basename(info['path'])])
                    summary_data.append(["  彩票类型", info['lottery_type']])
                    summary_data.append(["  颗粒度", info['granularity']])
                    summary_data.append(["  包含工作表数", len(info['sheet_names'])])
                    summary_data.append(["", ""])

                df_summary = pd.DataFrame(summary_data, columns=["项目", "值"])
                df_summary.to_excel(writer, sheet_name="合并摘要", index=False)

                # 为每个文件创建一个工作表
                for info in file_info:
                    file_path = info['path']
                    granularity = info['granularity']
                    lottery_type = info['lottery_type']
                    sheet_names = info['sheet_names']

                    # 工作表名称（避免过长和非法字符）
                    sheet_name = f"{granularity}"
                    if len(sheet_name) > 31:  # Excel限制31字符
                        sheet_name = sheet_name[:31]

                    # 读取原始文件的所有工作表
                    excel_data = pd.read_excel(file_path, sheet_name=None)

                    # 创建合并数据
                    merged_data = []

                    # 添加文件信息
                    merged_data.append([f"文件: {os.path.basename(file_path)}", ""])
                    merged_data.append(["彩票类型", lottery_type])
                    merged_data.append(["颗粒度", granularity])
                    merged_data.append(["", ""])

                    # 添加所有工作表的内容
                    for sheet in sheet_names:
                        if sheet in excel_data:
                            df = excel_data[sheet]
                            if not df.empty:
                                # 添加工作表标题
                                merged_data.append([f"【{sheet}】", ""])

                                # 添加列标题
                                columns = df.columns.tolist()
                                merged_data.append(columns)

                                # 添加数据行
                                for _, row in df.iterrows():
                                    merged_data.append(row.tolist())

                                # 添加分隔行
                                merged_data.append(["", ""])
                                merged_data.append(["", ""])

                    # 创建DataFrame
                    df_merged = pd.DataFrame(merged_data)
                    df_merged.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

                # 创建一个预测汇总表
                self._create_prediction_summary(writer, file_info)

                # 创建一个对比表
                self._create_comparison_table(writer, file_info)

            return True, f"合并完成！结果已保存到: {output_path}", output_path

        except Exception as e:
            return False, f"合并失败: {str(e)}", ""

    def _create_prediction_summary(self, writer, file_info):
        """创建预测汇总表"""
        summary_data = []

        # 表头
        summary_data.append(["颗粒度", "分析方法", "预测结果"])

        for info in file_info:
            file_path = info['path']
            granularity = info['granularity']

            try:
                # 读取原始文件
                excel_data = pd.read_excel(file_path, sheet_name=None)

                # 检查是否有"预测汇总"工作表
                if "预测汇总" in excel_data:
                    df_predictions = excel_data["预测汇总"]

                    # 获取分析方法列
                    if "分析方法" in df_predictions.columns:
                        for _, row in df_predictions.iterrows():
                            method = row["分析方法"] if "分析方法" in df_predictions.columns else ""

                            # 根据彩票类型提取预测结果
                            if "双色球" in file_path:
                                red_balls = ""
                                blue_balls = ""

                                if "红球预测号码" in df_predictions.columns:
                                    red_balls = row["红球预测号码"] if pd.notna(row["红球预测号码"]) else ""

                                if "蓝球预测号码" in df_predictions.columns:
                                    blue_balls = row["蓝球预测号码"] if pd.notna(row["蓝球预测号码"]) else ""

                                if red_balls or blue_balls:
                                    prediction_text = f"红球: {red_balls} 蓝球: {blue_balls}"
                                    summary_data.append([granularity, method, prediction_text])

                            elif "大乐透" in file_path:
                                front_balls = ""
                                back_balls = ""

                                if "前区预测号码" in df_predictions.columns:
                                    front_balls = row["前区预测号码"] if pd.notna(row["前区预测号码"]) else ""

                                if "后区预测号码" in df_predictions.columns:
                                    back_balls = row["后区预测号码"] if pd.notna(row["后区预测号码"]) else ""

                                if front_balls or back_balls:
                                    prediction_text = f"前区: {front_balls} 后区: {back_balls}"
                                    summary_data.append([granularity, method, prediction_text])

                # 检查是否有"综合推荐"工作表
                if "综合推荐" in excel_data:
                    df_comprehensive = excel_data["综合推荐"]

                    # 查找预测结果
                    for _, row in df_comprehensive.iterrows():
                        if isinstance(row.iloc[0], str) and "红球预测" in row.iloc[0]:
                            red_balls = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "蓝球预测" in row.iloc[0]:
                            blue_balls = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "前区预测" in row.iloc[0]:
                            front_balls = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "后区预测" in row.iloc[0]:
                            back_balls = row.iloc[1] if len(row) > 1 else ""

                    # 根据彩票类型添加预测结果
                    if "双色球" in file_path:
                        if 'red_balls' in locals() and 'blue_balls' in locals():
                            prediction_text = f"红球: {red_balls} 蓝球: {blue_balls}"
                            summary_data.append([granularity, "综合推荐", prediction_text])
                    elif "大乐透" in file_path:
                        if 'front_balls' in locals() and 'back_balls' in locals():
                            prediction_text = f"前区: {front_balls} 后区: {back_balls}"
                            summary_data.append([granularity, "综合推荐", prediction_text])

            except Exception as e:
                print(f"处理文件 {file_path} 的预测汇总时出错: {str(e)}")
                continue

        if len(summary_data) > 1:  # 如果有数据（除了表头）
            df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])
            df_summary.to_excel(writer, sheet_name="预测汇总对比", index=False)

    def _create_comparison_table(self, writer, file_info):
        """创建对比表"""
        comparison_data = []

        # 表头
        comparison_data.append(["颗粒度", "彩票类型", "分析时间", "数据量", "和值均值", "跨度均值"])

        for info in file_info:
            file_path = info['path']
            granularity = info['granularity']
            lottery_type = info['lottery_type']

            try:
                # 读取原始文件
                excel_data = pd.read_excel(file_path, sheet_name=None)

                # 初始化变量
                analysis_time = ""
                data_count = ""
                avg_sum = ""
                avg_span = ""

                # 从分析摘要获取信息
                if "分析摘要" in excel_data:
                    df_summary = excel_data["分析摘要"]
                    for _, row in df_summary.iterrows():
                        if isinstance(row.iloc[0], str) and "分析时间" in row.iloc[0]:
                            analysis_time = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "实际使用" in row.iloc[0]:
                            data_count = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "总数据量" in row.iloc[0]:
                            data_count = row.iloc[1] if len(row) > 1 else ""

                # 从统计概率分析获取信息
                if "统计概率分析" in excel_data:
                    df_stats = excel_data["统计概率分析"]
                    for _, row in df_stats.iterrows():
                        if isinstance(row.iloc[0], str) and "avg_sum" in row.iloc[0]:
                            avg_sum = row.iloc[1] if len(row) > 1 else ""
                        elif isinstance(row.iloc[0], str) and "avg_span" in row.iloc[0]:
                            avg_span = row.iloc[1] if len(row) > 1 else ""

                comparison_data.append([granularity, lottery_type, analysis_time, data_count, avg_sum, avg_span])

            except Exception as e:
                print(f"处理文件 {file_path} 的对比表时出错: {str(e)}")
                continue

        if len(comparison_data) > 1:  # 如果有数据（除了表头）
            df_comparison = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])
            df_comparison.to_excel(writer, sheet_name="颗粒度对比", index=False)


class LotteryResultsMergerGUI:
    """彩票分析结果合并工具GUI界面"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("彩票分析结果合并工具")
        self.window.geometry("800x600")

        self.merger = LotteryResultsMerger()
        self.selected_files = []

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 标题
        title_label = tk.Label(self.window, text="彩票分析结果合并工具",
                               font=("Arial", 20, "bold"))
        title_label.pack(pady=20)

        # 说明文字
        description = """
        功能说明：
        1. 选择多个颗粒度的分析结果Excel文件
        2. 将这些文件合并到一个新的Excel文件中
        3. 每个颗粒度的所有分析结果放在一个工作表中
        4. 生成预测汇总对比和颗粒度对比表

        使用步骤：
        1. 点击"选择Excel文件"按钮，选择要合并的文件
        2. 在下方列表中查看已选择的文件
        3. 点击"开始合并"按钮进行合并
        4. 合并完成后会自动打开结果文件夹
        """

        description_label = tk.Label(self.window, text=description,
                                     font=("Arial", 10), justify=tk.LEFT, wraplength=700)
        description_label.pack(pady=10)

        # 文件选择按钮
        select_frame = tk.Frame(self.window)
        select_frame.pack(pady=10)

        self.select_button = tk.Button(select_frame, text="选择Excel文件（可多选）",
                                       command=self.select_files, font=("Arial", 12),
                                       bg="#2196F3", fg="white")
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(select_frame, text="清空列表",
                                      command=self.clear_files, font=("Arial", 12))
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 已选择文件列表
        list_frame = tk.LabelFrame(self.window, text="已选择的文件", font=("Arial", 12))
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 创建列表框和滚动条
        self.file_listbox = tk.Listbox(list_frame, height=10, font=("Courier", 10))
        scrollbar = tk.Scrollbar(list_frame, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 文件数量标签
        self.file_count_label = tk.Label(self.window, text="已选择0个文件",
                                         font=("Arial", 10), fg="gray")
        self.file_count_label.pack(pady=5)

        # 合并按钮
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)

        self.merge_button = tk.Button(button_frame, text="开始合并",
                                      command=self.merge_files, font=("Arial", 14),
                                      state=tk.DISABLED, bg="#4CAF50", fg="white")
        self.merge_button.pack(side=tk.LEFT, padx=5)

        self.open_folder_button = tk.Button(button_frame, text="打开合并结果文件夹",
                                            command=self.open_results_folder, font=("Arial", 12))
        self.open_folder_button.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_bar = tk.Label(self.window, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_files(self):
        """选择文件"""
        file_paths = self.merger.select_files()

        if file_paths:
            for file_path in file_paths:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)

                    # 显示文件名和颗粒度
                    granularity = self.merger.extract_granularity_from_filename(file_path)
                    display_text = f"{os.path.basename(file_path)}  [{granularity}]"
                    self.file_listbox.insert(tk.END, display_text)

            self.update_file_count()
            self.merge_button.config(state=tk.NORMAL)
            self.update_status(f"已选择 {len(self.selected_files)} 个文件")

    def clear_files(self):
        """清空文件列表"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.update_file_count()
        self.merge_button.config(state=tk.DISABLED)
        self.update_status("已清空文件列表")

    def update_file_count(self):
        """更新文件数量显示"""
        count = len(self.selected_files)
        self.file_count_label.config(text=f"已选择{count}个文件")

    def merge_files(self):
        """合并文件"""
        if not self.selected_files:
            messagebox.showwarning("警告", "请先选择要合并的文件")
            return

        # 禁用按钮
        self.select_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.merge_button.config(state=tk.DISABLED)

        # 更新状态
        self.update_status("正在合并文件，请稍候...")
        self.window.update()

        # 执行合并
        success, message, output_path = self.merger.merge_excel_files(self.selected_files)

        # 恢复按钮状态
        self.select_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)

        if success:
            self.update_status("合并完成！")
            messagebox.showinfo("成功", message)

            # 清空列表
            self.clear_files()

            # 启用打开文件夹按钮
            self.open_folder_button.config(state=tk.NORMAL)
        else:
            self.update_status("合并失败")
            messagebox.showerror("错误", message)

    def open_results_folder(self):
        """打开结果文件夹"""
        folder_path = "merged_results"
        if os.path.exists(folder_path):
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":  # macOS
                    os.system(f'open "{folder_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{folder_path}"')
                self.update_status(f"已打开文件夹: {folder_path}")
            except Exception as e:
                self.update_status(f"打开文件夹失败: {e}")
                messagebox.showerror("错误", f"打开文件夹失败: {e}")
        else:
            self.update_status("文件夹不存在")
            messagebox.showwarning("警告", f"文件夹不存在: {folder_path}")

    def update_status(self, message: str):
        """更新状态栏"""
        self.status_bar.config(text=f"状态: {message}")
        self.window.update()

    def run(self):
        """运行GUI"""
        self.window.mainloop()


def main():
    """主函数"""
    app = LotteryResultsMergerGUI()
    app.run()


if __name__ == "__main__":
    main()