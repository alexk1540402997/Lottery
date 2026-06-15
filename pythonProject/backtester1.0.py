"""
彩票分析历史回测系统 - 修复版
功能：修复多进程问题和导入错误
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import importlib.util
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Manager, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import traceback
warnings.filterwarnings('ignore')

class LotteryBacktesterFixed:
    """修复版彩票分析历史回测器"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("彩票分析历史回测系统 - 修复版")
        self.window.geometry("900x700")

        # 初始化变量
        self.analyzer_class = None
        self.analyzer = None
        self.results = []
        self.backtest_running = False

        # 文件路径
        self.analyzer_file = ""
        self.data_file = ""

        # 回测参数
        self.start_periods = 100
        self.end_periods = 20
        self.granularities = [50, 100, 500, 0]

        # 进度相关
        self.progress_queue = None
        self.current_progress = 0
        self.total_tasks = 0

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(self.window, text="彩票分析历史回测系统 - 修复版",
                              font=("Arial", 20, "bold"))
        title_label.pack(pady=20)

        # 说明文字
        description = """
        功能说明：
        1. 修复多进程队列共享问题
        2. 修复子进程导入错误
        3. 添加详细错误日志
        4. 优化内存使用
        
        使用步骤：
        1. 选择分析器文件（您的彩票分析系统主代码）
        2. 选择历史数据Excel文件
        3. 配置回测参数
        4. 开始回测
        """

        description_label = tk.Label(self.window, text=description,
                                   font=("Arial", 10), justify=tk.LEFT, wraplength=800)
        description_label.pack(pady=10)

        # 文件选择区域
        file_frame = tk.LabelFrame(self.window, text="文件选择", font=("Arial", 12))
        file_frame.pack(fill=tk.X, padx=20, pady=10)

        # 分析器文件选择
        analyzer_frame = tk.Frame(file_frame)
        analyzer_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(analyzer_frame, text="分析器文件:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        self.analyzer_label = tk.Label(analyzer_frame, text="未选择", font=("Arial", 10),
                                      fg="gray", width=60, anchor="w")
        self.analyzer_label.pack(side=tk.LEFT, padx=5)

        tk.Button(analyzer_frame, text="选择", command=self.select_analyzer_file,
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # 数据文件选择
        data_frame = tk.Frame(file_frame)
        data_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(data_frame, text="历史数据文件:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        self.data_label = tk.Label(data_frame, text="未选择", font=("Arial", 10),
                                  fg="gray", width=60, anchor="w")
        self.data_label.pack(side=tk.LEFT, padx=5)

        tk.Button(data_frame, text="选择", command=self.select_data_file,
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # 参数设置区域
        param_frame = tk.LabelFrame(self.window, text="回测参数设置", font=("Arial", 12))
        param_frame.pack(fill=tk.X, padx=20, pady=10)

        # 回测模式选择
        mode_frame = tk.Frame(param_frame)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(mode_frame, text="回测模式:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        self.mode_var = tk.StringVar(value="fast")

        tk.Radiobutton(mode_frame, text="快速回测", variable=self.mode_var,
                      value="fast", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="标准回测", variable=self.mode_var,
                      value="standard", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="完整回测", variable=self.mode_var,
                      value="full", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        # CPU核心数设置
        cpu_frame = tk.Frame(param_frame)
        cpu_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(cpu_frame, text="CPU核心数:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        # 获取可用CPU核心数
        available_cores = cpu_count()
        # Windows上建议使用较少核心，避免内存问题
        self.cpu_var = tk.IntVar(value=min(4, available_cores // 2))

        core_options = list(range(1, min(8, available_cores) + 1))
        self.cpu_combo = ttk.Combobox(cpu_frame, textvariable=self.cpu_var,
                                      values=core_options, width=10, state="readonly")
        self.cpu_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(cpu_frame, text=f"可用核心: {available_cores}", font=("Arial", 9), fg="blue").pack(side=tk.LEFT, padx=10)

        # 控制按钮区域
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="开始回测",
                                     command=self.start_backtest,
                                     font=("Arial", 12), bg="#4CAF50", fg="white",
                                     width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="停止回测",
                                    command=self.stop_backtest,
                                    font=("Arial", 12), bg="#f44336", fg="white",
                                    width=15, height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.open_folder_button = tk.Button(button_frame, text="打开报告文件夹",
                                           command=self.open_reports_folder,
                                           font=("Arial", 12), width=15, height=2)
        self.open_folder_button.pack(side=tk.LEFT, padx=10)

        # 进度条区域
        progress_frame = tk.LabelFrame(self.window, text="回测进度", font=("Arial", 12))
        progress_frame.pack(fill=tk.X, padx=20, pady=10)

        # 进度条
        self.progress_bar = ttk.Progressbar(progress_frame, length=800, mode='determinate')
        self.progress_bar.pack(padx=10, pady=10)

        # 进度标签
        self.progress_label = tk.Label(progress_frame, text="准备就绪", font=("Arial", 10))
        self.progress_label.pack(pady=5)

        # 日志输出区域
        log_frame = tk.LabelFrame(self.window, text="回测日志", font=("Arial", 12))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 创建文本区域和滚动条
        self.log_text = tk.Text(log_frame, height=12, font=("Courier", 9))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_bar = tk.Label(self.window, text="就绪", bd=1,
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 进度更新定时器
        self.window.after(100, self.update_progress)

    def select_analyzer_file(self):
        """选择分析器文件"""
        file_path = filedialog.askopenfilename(
            title="选择分析器文件",
            filetypes=[("Python文件", "*.py"), ("所有文件", "*.*")],
            initialdir=os.getcwd()
        )

        if file_path:
            self.analyzer_file = file_path
            self.analyzer_label.config(text=os.path.basename(file_path), fg="black")
            self.log_message(f"已选择分析器文件: {file_path}")

    def select_data_file(self):
        """选择数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择历史数据Excel文件",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
            initialdir=os.getcwd()
        )

        if file_path:
            self.data_file = file_path
            self.data_label.config(text=os.path.basename(file_path), fg="black")
            self.log_message(f"已选择数据文件: {file_path}")

    def log_message(self, message: str):
        """在日志区域显示消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_line)
        self.log_text.see(tk.END)
        self.window.update_idletasks()

    def update_status(self, message: str):
        """更新状态栏"""
        self.status_bar.config(text=f"状态: {message}")
        self.window.update_idletasks()

    def update_progress(self):
        """更新进度条"""
        try:
            if self.progress_queue:
                while True:
                    # 尝试从队列获取进度更新
                    try:
                        progress_data = self.progress_queue.get_nowait()
                        if progress_data:
                            progress_type, data = progress_data

                            if progress_type == "start":
                                self.total_tasks = data
                                self.progress_bar['maximum'] = data
                                self.current_progress = 0
                                self.progress_label.config(text=f"0/{data} 任务")

                            elif progress_type == "update":
                                self.current_progress += data
                                percentage = (self.current_progress / self.total_tasks * 100) if self.total_tasks > 0 else 0
                                self.progress_bar['value'] = percentage
                                self.progress_label.config(text=f"{self.current_progress}/{self.total_tasks} 任务 ({percentage:.1f}%)")

                            elif progress_type == "complete":
                                self.progress_bar['value'] = 100
                                self.progress_label.config(text="完成")

                            elif progress_type == "error":
                                self.log_message(f"子进程错误: {data}")

                    except queue.Empty:
                        break
        except Exception as e:
            self.log_message(f"进度更新错误: {e}")

        # 继续定时更新
        self.window.after(100, self.update_progress)

    def import_analyzer_module(self) -> bool:
        """导入分析器模块"""
        if not self.analyzer_file or not os.path.exists(self.analyzer_file):
            self.log_message("错误: 未选择分析器文件或文件不存在")
            return False

        try:
            # 获取模块名
            module_name = os.path.splitext(os.path.basename(self.analyzer_file))[0]

            # 创建模块规范
            spec = importlib.util.spec_from_file_location(module_name, self.analyzer_file)

            if spec is None:
                self.log_message(f"无法从文件创建模块规范: {self.analyzer_file}")
                return False

            # 创建模块
            module = importlib.util.module_from_spec(spec)

            # 将模块添加到sys.modules
            sys.modules[module_name] = module

            # 执行模块代码
            spec.loader.exec_module(module)

            # 查找LotteryAnalyzerComplete类
            if hasattr(module, 'LotteryAnalyzerComplete'):
                self.analyzer_class = module.LotteryAnalyzerComplete
                self.log_message(f"成功导入分析器类: LotteryAnalyzerComplete")
                return True
            else:
                # 尝试查找其他可能的类名
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and 'Analyzer' in attr.__name__:
                        self.analyzer_class = attr
                        self.log_message(f"找到分析器类: {attr.__name__}")
                        return True

                self.log_message("在模块中找不到合适的分析器类")
                return False

        except Exception as e:
            self.log_message(f"导入分析器模块失败: {str(e)}")
            self.log_message(traceback.format_exc())
            return False

    def set_backtest_params(self):
        """根据选择的模式设置回测参数"""
        mode = self.mode_var.get()

        if mode == "fast":
            self.start_periods = 50
            self.end_periods = 10
            self.granularities = [50, 100, 0]
            self.log_message("使用快速回测参数")
        elif mode == "standard":
            self.start_periods = 100
            self.end_periods = 20
            self.granularities = [50, 100, 500, 0]
            self.log_message("使用标准回测参数")
        else:  # full
            self.start_periods = 100
            self.end_periods = 20
            self.granularities = [50, 100, 500, 1000, 0]
            self.log_message("使用完整回测参数")

    def start_backtest(self):
        """开始回测"""
        # 检查文件
        if not self.analyzer_file:
            messagebox.showwarning("警告", "请先选择分析器文件")
            return

        if not self.data_file:
            messagebox.showwarning("警告", "请先选择历史数据文件")
            return

        # 设置参数
        self.set_backtest_params()

        # 导入分析器
        self.log_message("正在导入分析器模块...")
        if not self.import_analyzer_module():
            messagebox.showerror("错误", "导入分析器失败")
            return

        # 禁用开始按钮，启用停止按钮
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.backtest_running = True

        # 清空日志
        self.log_text.delete(1.0, tk.END)

        # 创建Manager和队列
        self.manager = Manager()
        self.progress_queue = self.manager.Queue()

        # 在新线程中运行回测
        self.log_message("开始并行回测，请稍候...")
        self.update_status("回测运行中...")

        thread = threading.Thread(target=self.run_backtest_parallel, daemon=True)
        thread.start()

    def stop_backtest(self):
        """停止回测"""
        self.backtest_running = False
        self.log_message("正在停止回测...")
        self.update_status("正在停止回测...")

    def run_backtest_parallel(self):
        """使用多进程并行回测"""
        try:
            # 加载数据
            self.log_message(f"加载数据: {self.data_file}")
            analyzer_instance = self.analyzer_class()
            success, message = analyzer_instance.load_excel_file(self.data_file)

            if not success:
                self.log_message(f"数据加载失败: {message}")
                self.backtest_complete(False, f"数据加载失败: {message}")
                return

            self.log_message(f"数据加载成功: {message}")
            self.log_message(f"彩票类型: {analyzer_instance.lottery_type}")

            data = analyzer_instance.data_reverse
            total_periods = len(data)

            # 确保有足够的数据
            if total_periods < self.start_periods + 10:
                self.log_message(f"数据不足，至少需要{self.start_periods + 10}期数据")
                self.backtest_complete(False, "数据不足")
                return

            # 回测范围
            start_idx = self.start_periods
            end_idx = total_periods - self.end_periods - 1
            total_tasks = (end_idx - start_idx + 1) * len(self.granularities)

            self.log_message(f"总数据量: {total_periods}期")
            self.log_message(f"回测范围: 第{start_idx}期 到 第{end_idx}期")
            self.log_message(f"测试颗粒度: {self.granularities}")
            self.log_message(f"总任务数: {total_tasks}")
            self.log_message(f"使用CPU核心: {self.cpu_var.get()}")
            self.log_message("=" * 60)

            # 发送进度开始信号
            self.progress_queue.put(("start", total_tasks))

            # 准备并行任务
            period_indices = list(range(start_idx, end_idx + 1))

            # 计算每个进程处理的期数
            num_workers = min(self.cpu_var.get(), len(period_indices))
            chunks = np.array_split(period_indices, num_workers)

            self.log_message(f"将数据分成 {num_workers} 个批次进行并行处理")

            # 准备参数
            params = {
                'analyzer_file': self.analyzer_file,
                'data_file': self.data_file,
                'granularities': self.granularities
            }

            # 使用进程池并行处理
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []

                for i, chunk in enumerate(chunks):
                    if len(chunk) == 0:
                        continue

                    # 提交任务
                    future = executor.submit(
                        process_chunk_fixed,
                        chunk.tolist(),
                        params,
                        i,
                        self.progress_queue
                    )
                    futures.append(future)
                    self.log_message(f"提交批次 {i+1}: 处理 {len(chunk)} 期数据")

                # 收集结果
                all_results = []
                completed = 0

                for future in as_completed(futures):
                    try:
                        chunk_results = future.result(timeout=7200)  # 2小时超时
                        all_results.extend(chunk_results)
                        completed += 1
                        self.log_message(f"批次 {completed}/{len(futures)} 完成，获得 {len(chunk_results)} 条结果")
                    except Exception as e:
                        self.log_message(f"批次处理失败: {str(e)}")
                        self.progress_queue.put(("error", str(e)))

            # 合并结果
            self.results = all_results

            # 分析结果
            self.log_message("\n" + "=" * 60)
            self.log_message(f"回测完成! 共获得 {len(self.results)} 个预测评估")

            if len(self.results) == 0:
                self.log_message("警告: 没有获得任何评估结果，可能所有子进程都失败了")
                self.backtest_complete(False, "回测失败: 没有获得任何结果")
                return

            # 生成报告
            report_file = self.generate_report(analyzer_instance.lottery_type)

            # 发送进度完成信号
            self.progress_queue.put(("complete", None))

            self.backtest_complete(True, f"回测完成，报告已保存: {report_file}")

        except Exception as e:
            self.log_message(f"回测过程中发生错误: {str(e)}")
            self.log_message(traceback.format_exc())
            self.backtest_complete(False, f"回测失败: {str(e)}")

    def generate_report(self, lottery_type: str) -> str:
        """生成回测报告"""
        if not self.results:
            return ""

        # 创建输出目录
        output_dir = "backtest_reports"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"回测报告_{lottery_type}_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        # 转换为DataFrame以便分析
        df_data = []
        for result in self.results:
            eval_data = result.get('evaluation', {})

            if lottery_type == "ssq":
                hit_data = eval_data.get('hit_analysis', {})
                df_data.append({
                    '期号': result.get('period_number', 0),
                    '颗粒度': result.get('granularity_text', ''),
                    '方法': result.get('method_name', ''),
                    '红球命中数': hit_data.get('red_hit_count', 0),
                    '蓝球命中': hit_data.get('blue_hit', 0),
                    '总分': hit_data.get('total_hit_score', 0),
                    '评估分数': eval_data.get('score_summary', {}).get('total_score', 0)
                })
            else:
                hit_data = eval_data.get('hit_analysis', {})
                df_data.append({
                    '期号': result.get('period_number', 0),
                    '颗粒度': result.get('granularity_text', ''),
                    '方法': result.get('method_name', ''),
                    '前区命中数': hit_data.get('front_hit_count', 0),
                    '后区命中数': hit_data.get('back_hit_count', 0),
                    '总分': hit_data.get('total_hit_score', 0),
                    '评估分数': eval_data.get('score_summary', {}).get('total_score', 0)
                })

        df = pd.DataFrame(df_data)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. 摘要工作表
            summary_data = []
            summary_data.append(["回测报告摘要", ""])
            summary_data.append(["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            summary_data.append(["彩票类型", lottery_type])
            summary_data.append(["数据文件", os.path.basename(self.data_file)])
            summary_data.append(["总评估数", len(df)])
            summary_data.append(["分析方法数", df['方法'].nunique()])
            summary_data.append(["颗粒度数", df['颗粒度'].nunique()])
            summary_data.append(["期数范围", f"{df['期号'].min()}期 到 {df['期号'].max()}期"])
            summary_data.append(["回测模式", self.mode_var.get()])
            summary_data.append(["CPU核心数", self.cpu_var.get()])

            df_summary = pd.DataFrame(summary_data, columns=["项目", "值"])
            df_summary.to_excel(writer, sheet_name="报告摘要", index=False)

            # 2. 各方法表现
            if not df.empty:
                method_stats = df.groupby(['方法', '颗粒度']).agg({
                    '总分': ['mean', 'max', 'count'],
                    '评估分数': ['mean', 'max']
                }).round(3)
                method_stats.to_excel(writer, sheet_name="方法表现")

            # 3. 颗粒度表现
            if not df.empty:
                granularity_stats = df.groupby('颗粒度').agg({
                    '总分': ['mean', 'max', 'std'],
                    '评估分数': ['mean', 'max']
                }).round(3)
                granularity_stats.to_excel(writer, sheet_name="颗粒度表现")

            # 4. 最佳组合推荐
            if not df.empty:
                # 计算每个方法-颗粒度组合的平均分
                combinations = []
                for (method, granularity), group in df.groupby(['方法', '颗粒度']):
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
                df_top = pd.DataFrame(combinations[:20])
                df_top.to_excel(writer, sheet_name="最佳组合", index=False)

            # 5. 详细结果
            df.to_excel(writer, sheet_name="详细结果", index=False)

        self.log_message(f"报告已保存到: {filepath}")
        return filepath

    def backtest_complete(self, success: bool, message: str):
        """回测完成后的处理"""
        # 更新按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.backtest_running = False

        if success:
            self.log_message("\n" + "=" * 60)
            self.log_message("回测成功完成!")
            self.log_message(message)
            self.update_status("回测完成")

            # 显示成功消息
            self.window.after(100, lambda: messagebox.showinfo("成功", f"回测完成!\n{message}"))
        else:
            self.log_message("\n" + "=" * 60)
            self.log_message("回测失败!")
            self.log_message(message)
            self.update_status("回测失败")

            # 显示错误消息
            self.window.after(100, lambda: messagebox.showerror("错误", f"回测失败!\n{message}"))

    def open_reports_folder(self):
        """打开报告文件夹"""
        folder_path = "backtest_reports"
        if os.path.exists(folder_path):
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":
                    os.system(f'open "{folder_path}"')
                else:
                    os.system(f'xdg-open "{folder_path}"')
                self.log_message(f"已打开文件夹: {folder_path}")
            except Exception as e:
                self.log_message(f"打开文件夹失败: {e}")
        else:
            self.log_message(f"文件夹不存在: {folder_path}")

    def run(self):
        """运行GUI"""
        self.window.mainloop()


def process_chunk_fixed(chunk_indices, params, chunk_id, progress_queue):
    """处理一个数据块（在子进程中运行）- 修复版"""
    results = []

    try:
        # 在每个子进程中重新导入分析器
        analyzer_file = params['analyzer_file']
        data_file = params['data_file']
        granularities = params['granularities']

        # 动态导入分析器
        module_name = os.path.splitext(os.path.basename(analyzer_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, analyzer_file)
        if spec is None:
            progress_queue.put(("error", f"子进程{chunk_id}: 无法创建模块规范"))
            return results

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 查找分析器类
        analyzer_class = None
        if hasattr(module, 'LotteryAnalyzerComplete'):
            analyzer_class = module.LotteryAnalyzerComplete
        else:
            # 查找其他可能的类名
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and 'Analyzer' in attr.__name__:
                    analyzer_class = attr
                    break

        if analyzer_class is None:
            progress_queue.put(("error", f"子进程{chunk_id}: 找不到分析器类"))
            return results

        # 创建分析器实例
        analyzer = analyzer_class()

        # 加载数据
        success, message = analyzer.load_excel_file(data_file)
        if not success:
            progress_queue.put(("error", f"子进程{chunk_id}: 数据加载失败: {message}"))
            return results

        data = analyzer.data_reverse

        # 处理方法名称映射
        method_names = {
            'method_1': '统计概率分析',
            'method_2': '时间序列分析',
            'method_3': '模式识别分析',
            'method_4': '机器学习分析',
            'method_5': '马尔可夫分析',
            'method_6': '蒙特卡罗模拟',
            'method_7': '聚类分析',
            'method_8': 'N-gram分析',
            'comprehensive': '综合推荐'
        }

        # 处理块中的每个期数
        for period_idx in chunk_indices:
            # 1. 获取训练数据
            train_data = data.iloc[:period_idx]

            # 2. 获取实际开奖号码
            try:
                row = data.iloc[period_idx]

                if analyzer.lottery_type == "ssq":
                    actual_numbers = {
                        "red": [int(row[f'red_{i}']) for i in range(1, 7)],
                        "blue": [int(row['blue'])]
                    }
                else:
                    actual_numbers = {
                        "front": [int(row[f'front_{i}']) for i in range(1, 6)],
                        "back": [int(row[f'back_{i}']) for i in range(1, 3)]
                    }
            except Exception as e:
                # 跳过错误数据
                continue

            # 3. 用每种颗粒度进行分析
            for granularity in granularities:
                # 检查数据是否足够
                if granularity > 0 and len(train_data) < granularity:
                    continue

                # 执行分析
                predictions = analyze_with_data_single_fixed(analyzer_class, train_data, granularity, analyzer.lottery_type)

                if not predictions:
                    continue

                # 4. 评估每种方法的预测结果
                for method_key, method_predictions in predictions.items():
                    if not method_predictions:
                        continue

                    # 调用评估模块
                    try:
                        eval_result = analyzer.evaluate_prediction(method_predictions, actual_numbers)

                        if 'error' in eval_result:
                            continue

                        # 5. 记录结果
                        granularity_text = "全部期" if granularity == 0 else f"最近{granularity}期"

                        result_record = {
                            'period_index': period_idx,
                            'period_number': len(data) - period_idx,
                            'granularity': granularity,
                            'granularity_text': granularity_text,
                            'method_key': method_key,
                            'method_name': method_names.get(method_key, method_key),
                            'actual_numbers': actual_numbers,
                            'predictions': method_predictions,
                            'evaluation': eval_result,
                            'train_data_size': len(train_data)
                        }

                        results.append(result_record)
                    except Exception as e:
                        # 跳过评估错误
                        pass

                # 发送进度更新
                progress_queue.put(("update", 1))

    except Exception as e:
        error_msg = f"子进程{chunk_id}错误: {str(e)}"
        progress_queue.put(("error", error_msg))

    return results


def analyze_with_data_single_fixed(analyzer_class, data_subset, granularity, lottery_type):
    """使用指定数据进行分析（单次）- 修复版"""
    try:
        # 创建临时的分析器实例
        temp_analyzer = analyzer_class()

        # 复制数据
        temp_analyzer.data_reverse = data_subset
        temp_analyzer.lottery_type = lottery_type

        # 设置颗粒度
        temp_analyzer.set_analysis_granularity(granularity)

        # 执行分析
        results = temp_analyzer.analyze_all_methods()

        # 只返回预测部分
        predictions_summary = {}
        for method_key, result in results.items():
            if method_key.startswith('method_') and 'predictions' in result:
                predictions_summary[method_key] = result['predictions']
            elif method_key == 'comprehensive' and 'predictions' in result:
                predictions_summary['comprehensive'] = result['predictions']

        return predictions_summary

    except Exception:
        return {}


def main():
    """主函数"""
    # 设置多进程的启动方法
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)

    app = LotteryBacktesterFixed()
    app.run()


if __name__ == "__main__":
    main()
