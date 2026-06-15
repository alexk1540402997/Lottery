"""
超快彩票分析历史回测系统 - 终极优化版
目标：完整回测≤1小时
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
from multiprocessing import Pool, cpu_count, Manager, Queue, Process, Array
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import queue
import traceback
from typing import Dict, List, Tuple, Any, Optional
import pickle
import hashlib
warnings.filterwarnings('ignore')

# 尝试导入joblib用于缓存
try:
    from joblib import Memory, Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# 性能计时装饰器
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed > 1:
            print(f"{func.__name__} 耗时: {elapsed:.2f}秒")
        return result
    return wrapper

class UltraFastBacktester:
    """超快回测器 - 完整回测目标≤1小时"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("彩票分析历史回测系统 - 超快版")
        self.window.geometry("1000x800")

        # 初始化变量
        self.analyzer_class = None
        self.analyzer = None
        self.results = []
        self.backtest_running = False
        self.cache = {}  # 结果缓存

        # 文件路径
        self.analyzer_file = ""
        self.data_file = ""

        # 回测参数
        self.start_periods = 100
        self.end_periods = 20
        self.granularities = [50, 100, 500, 1000, 0]
        self.methods_to_test = ['method_1', 'method_2', 'method_3', 'method_4',
                               'method_5', 'method_6', 'method_7', 'method_8', 'comprehensive']

        # 进度相关
        self.progress_queue = None
        self.current_progress = 0
        self.total_tasks = 0

        # 性能统计
        self.performance_stats = {
            'total_time': 0,
            'avg_time_per_period': 0,
            'periods_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(self.window, text="彩票分析历史回测系统 - 超快版",
                              font=("Arial", 20, "bold"))
        title_label.pack(pady=20)

        # 说明文字
        description = """
        性能目标：完整回测≤1小时，快速回测≤10分钟
        优化技术：
        1. 三级并行：期数+颗粒度+方法全并行
        2. 智能缓存：避免重复计算
        3. 动态任务分配：负载均衡
        4. 结果复用：增量回测
        
        注意：第一次运行会较慢（需建立缓存），后续运行会大幅加速
        """

        description_label = tk.Label(self.window, text=description,
                                   font=("Arial", 10), justify=tk.LEFT, wraplength=900)
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

        self.mode_var = tk.StringVar(value="ultrafast")

        tk.Radiobutton(mode_frame, text="超快回测", variable=self.mode_var,
                      value="ultrafast", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="完整回测", variable=self.mode_var,
                      value="full", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="自定义回测", variable=self.mode_var,
                      value="custom", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        # 缓存设置
        cache_frame = tk.Frame(param_frame)
        cache_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(cache_frame, text="缓存设置:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        self.cache_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cache_frame, text="启用智能缓存", variable=self.cache_var,
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        self.incremental_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cache_frame, text="增量回测", variable=self.incremental_var,
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        # CPU核心数设置
        cpu_frame = tk.Frame(param_frame)
        cpu_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(cpu_frame, text="CPU核心数:", font=("Arial", 10), width=15).pack(side=tk.LEFT)

        available_cores = cpu_count()
        # 为超快回测，使用更多核心
        self.cpu_var = tk.IntVar(value=max(1, available_cores - 2))

        core_options = list(range(1, available_cores + 1))
        self.cpu_combo = ttk.Combobox(cpu_frame, textvariable=self.cpu_var,
                                      values=core_options, width=10, state="readonly")
        self.cpu_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(cpu_frame, text=f"可用核心: {available_cores}", font=("Arial", 9), fg="blue").pack(side=tk.LEFT, padx=10)

        # 参数说明
        desc_frame = tk.Frame(param_frame)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)

        desc_text = """
        超快回测：智能选择参数，最快速度（约5-10分钟）
        完整回测：完整参数，全面评估（目标≤1小时）
        自定义回测：手动设置参数
        启用缓存：首次运行较慢，后续大幅加速
        增量回测：只计算新数据，极速完成
        """

        desc_label = tk.Label(desc_frame, text=desc_text, font=("Arial", 9),
                             justify=tk.LEFT, fg="blue")
        desc_label.pack(anchor="w")

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

        self.clear_cache_button = tk.Button(button_frame, text="清除缓存",
                                          command=self.clear_cache,
                                          font=("Arial", 12), width=15, height=2)
        self.clear_cache_button.pack(side=tk.LEFT, padx=10)

        self.open_folder_button = tk.Button(button_frame, text="打开报告文件夹",
                                           command=self.open_reports_folder,
                                           font=("Arial", 12), width=15, height=2)
        self.open_folder_button.pack(side=tk.LEFT, padx=10)

        # 进度条区域
        progress_frame = tk.LabelFrame(self.window, text="回测进度", font=("Arial", 12))
        progress_frame.pack(fill=tk.X, padx=20, pady=10)

        # 总进度条
        self.progress_bar = ttk.Progressbar(progress_frame, length=900, mode='determinate')
        self.progress_bar.pack(padx=10, pady=5)

        # 进度标签
        self.progress_label = tk.Label(progress_frame, text="准备就绪", font=("Arial", 10))
        self.progress_label.pack(pady=5)

        # 性能统计标签
        self.stats_label = tk.Label(progress_frame, text="", font=("Arial", 9), fg="green")
        self.stats_label.pack(pady=5)

        # 日志输出区域
        log_frame = tk.LabelFrame(self.window, text="回测日志", font=("Arial", 12))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 创建文本区域和滚动条
        self.log_text = tk.Text(log_frame, height=15, font=("Courier", 9))
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

                            elif progress_type == "stats":
                                self.stats_label.config(text=f"性能: {data}")

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

        if mode == "ultrafast":
            # 超快回测：智能选择参数
            self.start_periods = 50
            self.end_periods = 10
            self.granularities = [50, 100, 0]  # 只测试关键颗粒度
            self.methods_to_test = ['method_1', 'method_2', 'method_3', 'method_4', 'comprehensive']  # 只测试关键方法
            self.log_message("使用超快回测参数")
        elif mode == "full":
            # 完整回测
            self.start_periods = 100
            self.end_periods = 20
            self.granularities = [50, 100, 500, 1000, 0]
            self.methods_to_test = ['method_1', 'method_2', 'method_3', 'method_4',
                                   'method_5', 'method_6', 'method_7', 'method_8', 'comprehensive']
            self.log_message("使用完整回测参数")
        else:  # custom
            # 自定义参数（可以根据需要扩展）
            self.start_periods = 100
            self.end_periods = 20
            self.granularities = [50, 100, 500, 0]
            self.methods_to_test = ['method_1', 'method_2', 'method_3', 'method_4',
                                   'method_5', 'method_6', 'method_7', 'method_8', 'comprehensive']
            self.log_message("使用自定义回测参数")

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
        self.clear_cache_button.config(state=tk.DISABLED)
        self.backtest_running = True

        # 清空日志
        self.log_text.delete(1.0, tk.END)

        # 创建Manager和队列
        self.manager = Manager()
        self.progress_queue = self.manager.Queue()

        # 在新线程中运行回测
        self.log_message("开始超快回测，请稍候...")
        self.update_status("回测运行中...")

        thread = threading.Thread(target=self.run_backtest_ultrafast, daemon=True)
        thread.start()

    def stop_backtest(self):
        """停止回测"""
        self.backtest_running = False
        self.log_message("正在停止回测...")
        self.update_status("正在停止回测...")

    def clear_cache(self):
        """清除缓存"""
        cache_dir = "backtest_cache"
        if os.path.exists(cache_dir):
            import shutil
            try:
                shutil.rmtree(cache_dir)
                self.log_message("已清除缓存")
            except Exception as e:
                self.log_message(f"清除缓存失败: {e}")
        else:
            self.log_message("缓存目录不存在")

    def test_analyzer_working(self):
        """测试分析器是否能正常工作"""
        try:
            # 创建分析器实例
            analyzer = self.analyzer_class()

            # 加载数据
            success, message = analyzer.load_excel_file(self.data_file)
            if not success:
                self.log_message(f"测试失败: 数据加载失败 - {message}")
                return False

            # 取前100期数据进行测试
            data = analyzer.data_reverse
            if len(data) < 100:
                self.log_message(f"测试失败: 数据不足，需要至少100期，当前{len(data)}期")
                return False

            train_data = data.iloc[:100]

            # 测试 analyze_all_methods
            temp_analyzer = self.analyzer_class()
            temp_analyzer.data_reverse = train_data
            temp_analyzer.lottery_type = analyzer.lottery_type
            temp_analyzer.set_analysis_granularity(50)

            self.log_message("正在测试 analyze_all_methods...")
            results = temp_analyzer.analyze_all_methods()

            if not results:
                self.log_message("测试失败: analyze_all_methods 返回空结果")
                return False

            self.log_message(f"测试成功: analyze_all_methods 返回了 {len(results)} 个方法的结果")
            self.log_message(f"可用方法: {list(results.keys())}")

            # 测试前几个方法是否有预测结果
            for method_key in ['method_1', 'method_2', 'comprehensive']:
                if method_key in results and 'predictions' in results[method_key]:
                    self.log_message(f"方法 {method_key} 有预测结果")
                else:
                    self.log_message(f"方法 {method_key} 缺少预测结果")

            return True

        except Exception as e:
            self.log_message(f"测试失败: {str(e)}")
            self.log_message(traceback.format_exc())
            return False

    def run_backtest_ultrafast(self):
        """运行超快回测"""
        start_time = time.time()

        try:
            # 先测试分析器是否能正常工作
            self.log_message("正在测试分析器...")
            test_success = self.test_analyzer_working()
            if not test_success:
                self.log_message("测试失败: 分析器无法正常工作")
                self.backtest_complete(False, "分析器测试失败")
                return
            self.log_message("分析器测试通过")

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

            # 计算总任务数
            total_tasks = (end_idx - start_idx + 1) * len(self.granularities) * len(self.methods_to_test)

            self.log_message(f"总数据量: {total_periods}期")
            self.log_message(f"回测范围: 第{start_idx}期 到 第{end_idx}期")
            self.log_message(f"测试颗粒度: {self.granularities}")
            self.log_message(f"测试方法: {len(self.methods_to_test)}个")
            self.log_message(f"总任务数: {total_tasks}")
            self.log_message(f"使用CPU核心: {self.cpu_var.get()}")
            self.log_message(f"启用缓存: {self.cache_var.get()}")
            self.log_message(f"增量回测: {self.incremental_var.get()}")
            self.log_message("=" * 60)

            # 发送进度开始信号
            self.progress_queue.put(("start", total_tasks))

            # 创建缓存目录
            cache_dir = "backtest_cache"
            os.makedirs(cache_dir, exist_ok=True)

            # 生成数据哈希，用于识别数据变化
            data_hash = self._generate_data_hash(data)

            # 加载已有缓存
            cache_file = os.path.join(cache_dir, f"cache_{data_hash}.pkl")
            if os.path.exists(cache_file) and self.cache_var.get():
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                self.log_message(f"加载缓存: {len(self.cache)} 条记录")
            else:
                self.cache = {}

            # 准备并行任务
            all_tasks = []

            # 生成所有任务
            for period_idx in range(start_idx, end_idx + 1):
                for granularity in self.granularities:
                    for method_key in self.methods_to_test:
                        # 检查缓存
                        cache_key = f"{period_idx}_{granularity}_{method_key}"

                        if self.cache_var.get() and cache_key in self.cache:
                            # 从缓存中获取
                            self.performance_stats['cache_hits'] += 1
                        else:
                            # 需要计算
                            self.performance_stats['cache_misses'] += 1
                            # 确保类型正确
                            all_tasks.append((
                                int(period_idx),  # 确保是int
                                int(granularity),  # 确保是int
                                str(method_key),  # 确保是str
                                str(data_hash)  # 确保是str
                            ))

            self.log_message(f"需要计算的任务: {len(all_tasks)}/{total_tasks}")
            self.log_message(f"缓存命中: {self.performance_stats['cache_hits']}")

            # 如果启用增量回测且缓存中有足够数据，可以跳过计算
            if self.incremental_var.get() and len(all_tasks) == 0:
                self.log_message("所有结果都在缓存中，直接生成报告...")
                # 从缓存中收集结果
                self.results = self._collect_results_from_cache(start_idx, end_idx)
            else:
                # 使用进程池并行处理
                num_workers = min(self.cpu_var.get(), len(all_tasks))

                if num_workers == 0:
                    self.log_message("没有需要计算的任务")
                    self.results = self._collect_results_from_cache(start_idx, end_idx)
                else:
                    # 分割任务
                    task_chunks = np.array_split(all_tasks, num_workers)

                    self.log_message(f"使用 {num_workers} 个进程并行处理 {len(all_tasks)} 个任务")

                    # 准备参数
                    params = {
                        'analyzer_file': self.analyzer_file,
                        'data_file': self.data_file,
                        'granularities': self.granularities,
                        'methods_to_test': self.methods_to_test,
                        'data_hash': data_hash
                    }

                    # 使用进程池并行处理
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = []

                        for i, chunk in enumerate(task_chunks):
                            if len(chunk) == 0:
                                continue

                            # 提交任务
                            future = executor.submit(
                                process_chunk_ultrafast,
                                chunk.tolist(),
                                params,
                                i,
                                self.progress_queue
                            )
                            futures.append(future)
                            self.log_message(f"提交批次 {i+1}: 处理 {len(chunk)} 个任务")

                        # 收集结果
                        all_results = []
                        completed = 0

                        for future in as_completed(futures):
                            try:
                                chunk_results, chunk_cache = future.result(timeout=3600)
                                all_results.extend(chunk_results)

                                # 更新缓存
                                self.cache.update(chunk_cache)

                                completed += 1
                                self.log_message(
                                    f"批次 {completed}/{len(futures)} 完成，获得 {len(chunk_results)} 条结果")

                                # 如果结果为空，添加警告
                                if len(chunk_results) == 0:
                                    self.log_message(f"警告: 批次 {completed} 获得了 0 条结果")
                                    # 记录子进程的标准错误输出
                                    import sys
                                    print(f"[WARNING] 子进程 {completed} 返回 0 条结果", file=sys.stderr)

                            except Exception as e:
                                self.log_message(f"批次处理失败: {str(e)}")
                                import traceback
                                self.log_message(traceback.format_exc())
                                self.progress_queue.put(("error", str(e)))
                        # 合并结果
                        self.results = all_results

                        # 从缓存中补充结果
                        cached_results = self._collect_results_from_cache(start_idx, end_idx)
                        self.results.extend(cached_results)

            # 保存缓存
            if self.cache_var.get():
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                self.log_message(f"缓存已保存: {cache_file}")

            # 分析结果
            end_time = time.time()
            total_time = end_time - start_time

            self.performance_stats['total_time'] = total_time
            self.performance_stats['periods_processed'] = end_idx - start_idx + 1
            self.performance_stats['avg_time_per_period'] = total_time / (end_idx - start_idx + 1) if (end_idx - start_idx + 1) > 0 else 0

            self.log_message("\n" + "=" * 60)
            self.log_message(f"回测完成! 共获得 {len(self.results)} 个预测评估")
            self.log_message(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
            self.log_message(f"平均每期耗时: {self.performance_stats['avg_time_per_period']:.3f}秒")
            self.log_message(f"缓存命中: {self.performance_stats['cache_hits']}, 缓存未命中: {self.performance_stats['cache_misses']}")

            if len(self.results) == 0:
                self.log_message("警告: 没有获得任何评估结果")
                self.backtest_complete(False, "回测失败: 没有获得任何结果")
                return

            # 生成报告
            report_file = self.generate_report(analyzer_instance.lottery_type)

            # 发送进度完成信号
            self.progress_queue.put(("complete", None))

            # 发送性能统计
            stats_text = f"耗时: {total_time/60:.1f}分钟 | 缓存命中: {self.performance_stats['cache_hits']}"
            self.progress_queue.put(("stats", stats_text))

            self.backtest_complete(True, f"回测完成，报告已保存: {report_file}")

        except Exception as e:
            self.log_message(f"回测过程中发生错误: {str(e)}")
            self.log_message(traceback.format_exc())
            self.backtest_complete(False, f"回测失败: {str(e)}")

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """生成数据哈希值"""
        # 使用数据的形状和最后几行生成哈希
        data_info = f"{data.shape}_{data.iloc[-10:].to_string()}"
        return hashlib.md5(data_info.encode()).hexdigest()[:16]

    def _collect_results_from_cache(self, start_idx: int, end_idx: int) -> list:
        """从缓存中收集结果"""
        results = []

        for period_idx in range(start_idx, end_idx + 1):
            for granularity in self.granularities:
                for method_key in self.methods_to_test:
                    cache_key = f"{period_idx}_{granularity}_{method_key}"
                    if cache_key in self.cache:
                        results.append(self.cache[cache_key])
                        # 发送进度更新
                        self.progress_queue.put(("update", 1))

        return results

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
            summary_data.append(["启用缓存", self.cache_var.get()])
            summary_data.append(["增量回测", self.incremental_var.get()])
            summary_data.append(["总耗时(秒)", f"{self.performance_stats['total_time']:.2f}"])
            summary_data.append(["平均每期耗时(秒)", f"{self.performance_stats['avg_time_per_period']:.3f}"])
            summary_data.append(["缓存命中数", self.performance_stats['cache_hits']])
            summary_data.append(["缓存未命中数", self.performance_stats['cache_misses']])

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
        self.clear_cache_button.config(state=tk.NORMAL)
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


def process_chunk_ultrafast(task_list, params, chunk_id, progress_queue):
    """处理一个任务块（在子进程中运行）"""
    import sys
    import traceback
    import os

    # 添加详细的调试信息
    print(f"[DEBUG] 子进程 {chunk_id} 启动，PID={os.getpid()}", file=sys.stderr)
    print(f"[DEBUG] 任务数量: {len(task_list)}", file=sys.stderr)
    print(f"[DEBUG] 参数 keys: {params.keys() if params else '无'}", file=sys.stderr)

    # 打印第一个任务信息
    if task_list and len(task_list) > 0:
        first_task = task_list[0]
        print(f"[DEBUG] 第一个任务: {first_task}", file=sys.stderr)

    results = []
    local_cache = {}

    # 移除了外层的 try-except，让异常自然传播

    analyzer_file = params['analyzer_file']
    data_file = params['data_file']
    data_hash = params['data_hash']

    # 动态导入分析器
    module_name = os.path.splitext(os.path.basename(analyzer_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, analyzer_file)
    if spec is None:
        error_msg = f"子进程{chunk_id}: 无法创建模块规范"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        progress_queue.put(("error", error_msg))
        return results, local_cache

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 查找分析器类
    analyzer_class = None
    if hasattr(module, 'LotteryAnalyzerComplete'):
        analyzer_class = module.LotteryAnalyzerComplete
        print(f"[DEBUG] 成功导入 LotteryAnalyzerComplete 类", file=sys.stderr)
    else:
        # 查找其他可能的类名
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and 'Analyzer' in attr.__name__:
                analyzer_class = attr
                print(f"[DEBUG] 找到分析器类: {attr.__name__}", file=sys.stderr)
                break

    if analyzer_class is None:
        error_msg = f"子进程{chunk_id}: 找不到分析器类"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        progress_queue.put(("error", error_msg))
        return results, local_cache

    # 创建分析器实例
    analyzer = analyzer_class()

    # 加载数据
    success, message = analyzer.load_excel_file(data_file)
    if not success:
        error_msg = f"子进程{chunk_id}: 数据加载失败: {message}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        progress_queue.put(("error", error_msg))
        return results, local_cache

    data = analyzer.data_reverse
    print(f"[DEBUG] 数据加载成功，形状: {data.shape if hasattr(data, 'shape') else '无形状'}", file=sys.stderr)
    print(f"[DEBUG] 彩票类型: {analyzer.lottery_type}", file=sys.stderr)

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

    # 处理每个任务
    task_count = 0
    for task in task_list:
        task_count += 1

        # 每处理100个任务输出一次进度
        if task_count % 100 == 0:
            print(f"[DEBUG] 子进程{chunk_id} 已处理 {task_count}/{len(task_list)} 个任务", file=sys.stderr)

        try:
            # 解析任务参数，确保类型正确
            if len(task) != 4:
                print(f"[ERROR] 任务格式错误: {task}", file=sys.stderr)
                continue

            # 转换为正确的类型
            try:
                period_idx = int(task[0])
                granularity = int(task[1])
                method_key = str(task[2])
                _ = task[3]  # data_hash, 不需要
            except ValueError as e:
                print(f"[ERROR] 参数类型转换失败: {e}, 原始任务: {task}", file=sys.stderr)
                continue

            # 验证参数范围
            if period_idx < 0 or period_idx >= len(data):
                print(f"[WARNING] 期数索引超出范围: {period_idx}, 数据长度: {len(data)}", file=sys.stderr)
                continue

            if granularity < 0:
                print(f"[WARNING] 颗粒度不能为负数: {granularity}", file=sys.stderr)
                continue

            # 1. 获取训练数据
            train_data = data.iloc[:period_idx]

            # 2. 检查数据是否足够
            if granularity > 0 and len(train_data) < granularity:
                continue

            # 3. 获取实际开奖号码
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
                print(f"[ERROR] 任务处理失败: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue

            # 4. 执行分析
            predictions = analyze_with_data_ultrafast(analyzer_class, train_data,
                                                      granularity, analyzer.lottery_type,
                                                      method_key)

            if not predictions or method_key not in predictions:
                print(f"[DEBUG] 分析失败: 期{period_idx} 方法{method_key} 无预测结果", file=sys.stderr)
                continue

            method_predictions = predictions[method_key]
            if not method_predictions:
                print(f"[DEBUG] 预测结果为空: 期{period_idx} 方法{method_key}", file=sys.stderr)
                continue

            # 5. 评估预测结果
            eval_result = analyzer.evaluate_prediction(method_predictions, actual_numbers)

            if 'error' in eval_result:
                print(f"[DEBUG] 评估失败: {eval_result.get('error')}", file=sys.stderr)
                continue

            # 6. 记录结果
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
            print(f"[SUCCESS] 获得结果: 期{period_idx} 方法{method_key} 颗粒度{granularity_text}", file=sys.stderr)

            # 7. 存入缓存
            cache_key = f"{period_idx}_{granularity}_{method_key}"
            local_cache[cache_key] = result_record

        except Exception as e:
            print(f"[ERROR] 任务处理失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

        # 发送进度更新
        progress_queue.put(("update", 1))

    print(f"[DEBUG] 子进程{chunk_id} 处理完成，共获得 {len(results)} 条结果", file=sys.stderr)

    # 在函数末尾添加详细的错误处理
    if len(results) == 0 and len(task_list) > 0:
        # 如果处理了任务但没有结果，记录警告
        print(f"[WARNING] 子进程 {chunk_id} 处理了 {len(task_list)} 个任务，但获得 0 条结果", file=sys.stderr)
        print(f"[WARNING] 检查 analyze_with_data_ultrafast 函数是否正常工作", file=sys.stderr)

    print(f"[DEBUG] 子进程 {chunk_id} 完成，获得 {len(results)} 条结果", file=sys.stderr)
    return results, local_cache


def analyze_with_data_ultrafast(analyzer_class, data_subset, granularity, lottery_type, method_key):
    """使用指定数据进行分析（单次）- 修复版"""
    import sys

    try:
        # 创建临时的分析器实例
        temp_analyzer = analyzer_class()

        # 复制数据
        temp_analyzer.data_reverse = data_subset
        temp_analyzer.lottery_type = lottery_type

        # 设置颗粒度
        temp_analyzer.set_analysis_granularity(granularity)

        # 执行分析
        all_results = temp_analyzer.analyze_all_methods()

        # 检查是否包含所需的方法
        if method_key not in all_results:
            print(f"[WARNING] 方法 {method_key} 不在分析结果中", file=sys.stderr)
            return {}

        # 获取指定方法的预测结果
        method_result = all_results[method_key]

        # 检查是否有predictions字段
        if 'predictions' not in method_result:
            print(f"[WARNING] 方法 {method_key} 没有predictions字段", file=sys.stderr)
            return {}

        predictions = method_result['predictions']

        # 检查预测结果是否为空
        if not predictions:
            print(f"[WARNING] 方法 {method_key} 的predictions为空", file=sys.stderr)
            return {}

        # 检查预测结果格式
        if isinstance(predictions, dict):
            # 如果是字典格式，检查是否有红球/前区预测
            if lottery_type == "ssq":
                if 'red' not in predictions or 'blue' not in predictions:
                    print(f"[WARNING] 方法 {method_key} 的predictions格式不正确", file=sys.stderr)
                    return {}
            else:
                if 'front' not in predictions or 'back' not in predictions:
                    print(f"[WARNING] 方法 {method_key} 的predictions格式不正确", file=sys.stderr)
                    return {}

        return {method_key: predictions}

    except Exception as e:
        print(f"[ERROR] analyze_with_data_ultrafast 失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {}


def main():
    """主函数"""
    # 设置多进程的启动方法
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)

    app = UltraFastBacktester()
    app.run()


if __name__ == "__main__":
    main()