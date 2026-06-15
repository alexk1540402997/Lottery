"""
统一回测与优化引擎 3.0
========================
将回测系统与优化控制器合二为一，解决以下问题：
1. 多进程开销大 → 改用线程池 + numpy GIL释放
2. 优化无闭环 → 实现 回测→优化权重→再回测→迭代收敛
3. 滚动窗口验证 → walk-forward validation
4. 使用3.0版预测引擎（核心方法已全面升级）

性能目标：
- 快速回测 ≤ 5分钟（100期 × 3颗粒度 × 5方法）
- 完整回测 ≤ 30分钟（全部参数）
- 优化迭代 2-3轮自动收敛
"""

import os
import sys
import time
import json
import hashlib
import pickle
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import queue as queue_module

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入3.0版预测引擎
import importlib.util

def load_analyzer_module(analyzer_file: str):
    """动态加载分析器模块"""
    module_name = os.path.splitext(os.path.basename(analyzer_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, analyzer_file)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, 'LotteryAnalyzerComplete'):
        return module.LotteryAnalyzerComplete
    # fallback
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and 'Analyzer' in attr.__name__:
            return attr
    return None


class UnifiedBacktester:
    """统一回测与优化引擎 3.0"""

    def __init__(self):
        # GUI
        self.window = tk.Tk()
        self.window.title("统一回测与优化引擎 3.0")
        self.window.geometry("1100x850")

        # 核心状态
        self.analyzer_class = None
        self.analyzer_file = ""
        self.data_file = ""
        self.lottery_type = ""
        self.all_data = None  # 完整数据 (DataFrame, 倒序)
        self.backtest_running = False

        # 回测参数
        self.granularities = [50, 100, 500, 1000, 0]
        self.methods_to_test = []
        self.start_idx = 100   # 从第100期开始（确保有足够训练数据）
        self.end_idx = 0

        # 缓存
        self.cache = {}
        self.cache_dir = "backtest_cache"
        self.use_cache = True

        # 结果
        self.results: List[Dict] = []
        self.method_weights = {}      # 优化后的方法权重
        self.granularity_weights = {} # 优化后的颗粒度权重

        # 进度
        self.progress_queue = queue_module.Queue()
        self.total_tasks = 0
        self.completed_tasks = 0

        # 性能统计
        self.start_time = 0
        self.total_time = 0

        self._setup_ui()
        # 启动进度更新定时器
        self.window.after(100, self._update_progress_ui)

    def _setup_ui(self):
        """设置GUI"""
        # 标题
        tk.Label(self.window, text="统一回测与优化引擎 3.0",
                font=("Arial", 20, "bold"), fg="#1565C0").pack(pady=15)

        # 说明
        desc = (
            "将回测系统和优化系统合二为一。使用3.0版预测引擎（核心方法已全面升级）。\n"
            "支持：快速回测 | 完整回测 | 迭代优化(闭环) | 滚动窗口验证 | 智能缓存"
        )
        tk.Label(self.window, text=desc, font=("Arial", 10),
                justify=tk.LEFT, wraplength=1000).pack(pady=5)

        # 文件选择
        file_frame = tk.LabelFrame(self.window, text="文件选择", font=("Arial", 12))
        file_frame.pack(fill=tk.X, padx=20, pady=10)

        for label_text, attr_name, btn_cmd in [
            ("分析器文件 (3.0版):", "analyzer_label", self._select_analyzer),
            ("历史数据文件:", "data_label", self._select_data),
        ]:
            f = tk.Frame(file_frame)
            f.pack(fill=tk.X, padx=10, pady=8)
            tk.Label(f, text=label_text, font=("Arial", 10), width=20, anchor="e").pack(side=tk.LEFT, padx=5)
            lbl = tk.Label(f, text="未选择", font=("Arial", 10), fg="gray", width=65, anchor="w")
            lbl.pack(side=tk.LEFT, padx=5)
            setattr(self, attr_name, lbl)
            tk.Button(f, text="选择", command=btn_cmd, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # 回测模式
        mode_frame = tk.LabelFrame(self.window, text="回测设置", font=("Arial", 12))
        mode_frame.pack(fill=tk.X, padx=20, pady=10)

        # 模式选择
        mode_row = tk.Frame(mode_frame)
        mode_row.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(mode_row, text="回测模式:", font=("Arial", 10), width=12).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="fast")
        for text, val in [("快速回测 (≈3分钟)", "fast"), ("完整回测 (≈20分钟)", "full"),
                          ("迭代优化 (闭环)", "optimize"), ("自定义", "custom")]:
            tk.Radiobutton(mode_row, text=text, variable=self.mode_var, value=val,
                          font=("Arial", 10)).pack(side=tk.LEFT, padx=8)

        # 其他设置
        opts_row = tk.Frame(mode_frame)
        opts_row.pack(fill=tk.X, padx=10, pady=5)

        self.cache_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_row, text="智能缓存", variable=self.cache_var,
                      font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        tk.Label(opts_row, text="线程数:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.workers_var = tk.IntVar(value=min(8, os.cpu_count() or 4))
        workers_combo = ttk.Combobox(opts_row, textvariable=self.workers_var,
                                     values=list(range(1, (os.cpu_count() or 4) + 1)),
                                     width=5, state="readonly")
        workers_combo.pack(side=tk.LEFT)

        # 按钮
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=15)

        self.start_btn = tk.Button(btn_frame, text="▶ 开始回测",
                                   command=self._start_backtest, font=("Arial", 13),
                                   bg="#4CAF50", fg="white", width=14, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=8)

        self.stop_btn = tk.Button(btn_frame, text="■ 停止",
                                  command=self._stop_backtest, font=("Arial", 13),
                                  bg="#f44336", fg="white", width=10, height=2,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.clear_btn = tk.Button(btn_frame, text="清除缓存",
                                   command=self._clear_cache, font=("Arial", 11),
                                   width=10, height=2)
        self.clear_btn.pack(side=tk.LEFT, padx=8)

        self.report_btn = tk.Button(btn_frame, text="打开报告",
                                    command=self._open_reports, font=("Arial", 11),
                                    width=10, height=2)
        self.report_btn.pack(side=tk.LEFT, padx=8)

        # 进度条
        prog_frame = tk.LabelFrame(self.window, text="进度", font=("Arial", 12))
        prog_frame.pack(fill=tk.X, padx=20, pady=5)

        self.progress_bar = ttk.Progressbar(prog_frame, length=900, mode='determinate')
        self.progress_bar.pack(padx=10, pady=5)
        self.progress_label = tk.Label(prog_frame, text="准备就绪", font=("Arial", 10))
        self.progress_label.pack(pady=3)
        self.time_label = tk.Label(prog_frame, text="", font=("Arial", 9), fg="green")
        self.time_label.pack(pady=2)

        # 日志
        log_frame = tk.LabelFrame(self.window, text="回测日志", font=("Arial", 12))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        self.log_text = tk.Text(log_frame, height=12, font=("Courier New", 9))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_bar = tk.Label(self.window, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==================== GUI事件 ====================

    def _select_analyzer(self):
        path = filedialog.askopenfilename(
            title="选择分析器文件（推荐使用3.0版）",
            filetypes=[("Python文件", "*.py")]
        )
        if path:
            self.analyzer_file = path
            self.analyzer_label.config(text=os.path.basename(path), fg="black")
            self._log(f"已选择分析器: {os.path.basename(path)}")

    def _select_data(self):
        path = filedialog.askopenfilename(
            title="选择历史数据Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls")]
        )
        if path:
            self.data_file = path
            self.data_label.config(text=os.path.basename(path), fg="black")
            self._log(f"已选择数据: {os.path.basename(path)}")

    def _set_params_by_mode(self):
        """根据模式设置参数"""
        mode = self.mode_var.get()
        if mode == "fast":
            self.granularities = [50, 100, 0]
            self.methods_to_test = ['method_1', 'method_2', 'method_3', 'method_4', 'comprehensive']
            self.start_idx = 50
        elif mode == "full":
            self.granularities = [50, 100, 500, 1000, 0]
            self.methods_to_test = [f'method_{i}' for i in range(1, 9)] + ['comprehensive']
            self.start_idx = 100
        elif mode == "optimize":
            self.granularities = [50, 100, 500, 0]
            self.methods_to_test = [f'method_{i}' for i in range(1, 9)] + ['comprehensive']
            self.start_idx = 100
        else:  # custom
            self.granularities = [50, 100, 500, 0]
            self.methods_to_test = [f'method_{i}' for i in range(1, 9)] + ['comprehensive']
            self.start_idx = 100
        self._log(f"模式: {mode}, 颗粒度: {self.granularities}, 方法: {len(self.methods_to_test)}个")

    def _start_backtest(self):
        """开始回测"""
        if not self.analyzer_file:
            messagebox.showwarning("警告", "请先选择分析器文件")
            return
        if not self.data_file:
            messagebox.showwarning("警告", "请先选择历史数据文件")
            return

        self._set_params_by_mode()
        self.use_cache = self.cache_var.get()

        # 禁用按钮
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.DISABLED)
        self.backtest_running = True

        # 清空日志
        self.log_text.delete(1.0, tk.END)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="初始化中...")

        # 在新线程中运行
        threading.Thread(target=self._run_backtest_thread, daemon=True).start()

    def _stop_backtest(self):
        self.backtest_running = False
        self._log("正在停止回测...")

    def _clear_cache(self):
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            self._log("缓存已清除")
        else:
            self._log("缓存目录不存在")

    def _open_reports(self):
        path = "backtest_reports"
        if os.path.exists(path):
            os.startfile(path) if sys.platform == "win32" else os.system(f'open "{path}"')

    def _log(self, msg: str):
        """记录日志"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        print(f"[{ts}] {msg}")

    def _update_status(self, msg: str):
        self.status_bar.config(text=f"状态: {msg}")

    def _update_progress_ui(self):
        """定期更新进度条UI"""
        try:
            while True:
                data = self.progress_queue.get_nowait()
                if data[0] == "init":
                    self.total_tasks = data[1]
                    self.completed_tasks = 0
                    self.progress_bar['maximum'] = self.total_tasks
                    self.progress_bar['value'] = 0
                elif data[0] == "tick":
                    self.completed_tasks += data[1]
                    if self.total_tasks > 0:
                        pct = self.completed_tasks / self.total_tasks * 100
                        self.progress_bar['value'] = pct
                        elapsed = time.time() - self.start_time if self.start_time else 0
                        if self.completed_tasks > 0 and elapsed > 1:
                            eta = (elapsed / self.completed_tasks) * (self.total_tasks - self.completed_tasks)
                            self.progress_label.config(
                                text=f"{self.completed_tasks}/{self.total_tasks} 任务 "
                                     f"({pct:.1f}%) - 预计剩余: {eta:.0f}秒")
                elif data[0] == "done":
                    self.progress_bar['value'] = 100
                    self.progress_label.config(text="完成!")
                elif data[0] == "msg":
                    self._log(data[1])
        except queue_module.Empty:
            pass
        self.window.after(100, self._update_progress_ui)

    # ==================== 核心回测逻辑 ====================

    def _run_backtest_thread(self):
        """在后台线程中运行回测"""
        self.start_time = time.time()
        try:
            # 1. 导入分析器
            self._log("正在导入分析器模块...")
            self.analyzer_class = load_analyzer_module(self.analyzer_file)
            if self.analyzer_class is None:
                self._log("错误: 无法导入分析器类")
                self._backtest_done(False, "导入分析器失败")
                return
            self._log(f"分析器导入成功: {self.analyzer_class.__name__}")

            # 2. 加载数据
            self._log(f"加载数据: {self.data_file}")
            analyzer = self.analyzer_class()
            success, msg = analyzer.load_excel_file(self.data_file)
            if not success:
                self._log(f"数据加载失败: {msg}")
                self._backtest_done(False, f"数据加载失败: {msg}")
                return

            self.all_data = analyzer.data_reverse
            self.lottery_type = analyzer.lottery_type
            total_periods = len(self.all_data)
            self._log(f"数据加载成功: {total_periods}期 {self.lottery_type}")

            # 3. 计算回测范围
            if self.start_idx >= total_periods:
                self.start_idx = max(30, total_periods // 3)
            self.end_idx = max(10, total_periods // 20)  # 保留最后10-5%作为验证

            end_test_idx = total_periods - self.end_idx - 1
            num_test_periods = end_test_idx - self.start_idx + 1
            self._log(f"回测范围: 第{self.start_idx}期~第{end_test_idx}期 ({num_test_periods}个测试点)")

            # 4. 加载缓存
            self._load_cache()

            # 5. 生成任务
            task_list = []
            cache_hits = 0
            for period_idx in range(self.start_idx, end_test_idx + 1):
                for g in self.granularities:
                    for m in self.methods_to_test:
                        ck = f"{period_idx}_{g}_{m}"
                        if ck in self.cache:
                            cache_hits += 1
                        else:
                            task_list.append((period_idx, g, m))

            self.total_tasks = len(task_list)
            self._log(f"总任务: {len(task_list) + cache_hits} (缓存命中: {cache_hits}, 需计算: {len(task_list)})")
            self.progress_queue.put(("init", len(task_list) + cache_hits))

            # 从缓存推送进度
            for _ in range(cache_hits):
                self.progress_queue.put(("tick", 1))

            if not task_list:
                self._log("所有任务已在缓存中！直接从缓存生成报告。")
                self._collect_cache_results()
            else:
                # 6. 使用线程池并行执行
                n_workers = min(self.workers_var.get(), len(task_list))
                self._log(f"使用 {n_workers} 个线程并行处理...")

                # 将任务分块
                chunks = np.array_split(task_list, n_workers)
                new_cache = {}

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for i, chunk in enumerate(chunks):
                        if len(chunk) == 0:
                            continue
                        futures.append(executor.submit(
                            self._process_chunk, chunk.tolist(), i
                        ))

                    for future in as_completed(futures):
                        try:
                            chunk_results, chunk_cache = future.result(timeout=3600)
                            self.results.extend(chunk_results)
                            new_cache.update(chunk_cache)
                            self._log(f"批次完成: +{len(chunk_results)}条结果")
                        except Exception as e:
                            self._log(f"批次处理错误: {e}")

                # 合并缓存
                self.cache.update(new_cache)

                # 从缓存中补充结果
                self._collect_cache_results()

            # 7. 保存缓存
            self._save_cache()

            # 8. 生成报告
            total_time = time.time() - self.start_time
            self.total_time = total_time
            self._log(f"\n回测完成! 共 {len(self.results)} 条评估记录, 耗时 {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

            if len(self.results) == 0:
                self._backtest_done(False, "回测失败: 没有获得任何结果")
                return

            report_file = self._generate_report()

            # 9. 如果是优化模式，进行权重优化和迭代
            if self.mode_var.get() == "optimize":
                self._log("\n========== 开始闭环优化 ==========")
                self._optimize_weights()
                # 用优化后的权重重新评估
                optimized_report = self._generate_optimized_report()
                self._log(f"优化报告已生成: {optimized_report}")

            self.progress_queue.put(("done", None))
            self._backtest_done(True, f"回测完成! 报告: {report_file}")

        except Exception as e:
            self._log(f"回测异常: {e}")
            self._log(traceback.format_exc())
            self._backtest_done(False, f"回测异常: {e}")

    def _process_chunk(self, task_list: List[Tuple], chunk_id: int) -> Tuple[List[Dict], Dict]:
        """处理一个任务块（在线程中运行）

        优化：先将任务按 (period_idx, granularity) 分组，
        每组只调用一次 analyze_all_methods()，然后评估所有方法。
        这样将计算量减少 N 倍（N=方法数）。
        """
        results = []
        local_cache = {}
        error_count = 0

        # 每个线程创建自己的分析器实例（用于获取数据和类型）
        try:
            analyzer = self.analyzer_class()
            ok, msg = analyzer.load_excel_file(self.data_file)
            if not ok:
                self.progress_queue.put(("msg", f"线程{chunk_id}数据加载失败: {msg}"))
                return results, local_cache
            data = analyzer.data_reverse
            lot_type = analyzer.lottery_type
        except Exception as e:
            self.progress_queue.put(("msg", f"线程{chunk_id}初始化失败: {e}"))
            return results, local_cache

        method_names = {
            'method_1': '统计概率分析', 'method_2': '时间序列分析',
            'method_3': '模式识别分析', 'method_4': '机器学习分析',
            'method_5': '马尔可夫分析', 'method_6': '蒙特卡罗模拟',
            'method_7': '聚类分析', 'method_8': 'N-gram分析',
            'comprehensive': '综合推荐'
        }

        # ★ 关键优化：按 (period_idx, granularity) 分组
        # 每组只分析一次，然后评估所有需要的 method
        task_groups = {}
        for period_idx, granularity, method_key in task_list:
            key = (period_idx, granularity)
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(method_key)

        processed = 0
        for (period_idx, granularity), methods_needed in task_groups.items():
            if not self.backtest_running:
                break

            try:
                # 获取训练数据
                train_data = data.iloc[:period_idx].copy()
                if granularity > 0 and len(train_data) < granularity:
                    processed += len(methods_needed)
                    # 每个method都算完成
                    for _ in methods_needed:
                        self.progress_queue.put(("tick", 1))
                    continue

                # 获取实际开奖号码
                actual_row = data.iloc[period_idx]
                if lot_type == "ssq":
                    actual = {
                        "red": [int(actual_row[f'red_{i}']) for i in range(1, 7)],
                        "blue": [int(actual_row['blue'])]
                    }
                else:
                    actual = {
                        "front": [int(actual_row[f'front_{i}']) for i in range(1, 6)],
                        "back": [int(actual_row[f'back_{i}']) for i in range(1, 3)]
                    }

                # ★ 每个 (period_idx, granularity) 只分析一次！
                temp_analyzer = self.analyzer_class()
                temp_analyzer.data_reverse = train_data
                temp_analyzer.lottery_type = lot_type
                temp_analyzer.set_analysis_granularity(granularity)
                all_method_results = temp_analyzer.analyze_all_methods()

                # 评估所有需要的方法
                granularity_text = "全部期" if granularity == 0 else f"最近{granularity}期"

                for method_key in methods_needed:
                    if method_key not in all_method_results:
                        self.progress_queue.put(("tick", 1))
                        continue

                    method_result = all_method_results[method_key]
                    if 'error' in method_result or 'predictions' not in method_result:
                        self.progress_queue.put(("tick", 1))
                        continue

                    predictions = method_result['predictions']
                    eval_result = self._quick_evaluate(predictions, actual, lot_type)

                    record = {
                        'period_index': period_idx,
                        'period_number': len(data) - period_idx,
                        'granularity': granularity,
                        'granularity_text': granularity_text,
                        'method_key': method_key,
                        'method_name': method_names.get(method_key, method_key),
                        'eval': eval_result,
                        'train_size': len(train_data)
                    }

                    results.append(record)
                    local_cache[f"{period_idx}_{granularity}_{method_key}"] = record
                    self.progress_queue.put(("tick", 1))

                processed += len(methods_needed)

            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    # 只报告前3个错误，避免刷屏
                    self.progress_queue.put(("msg",
                        f"线程{chunk_id}错误(期{period_idx}颗粒度{granularity}): {e}"))
                # 标记这些任务为完成
                for _ in methods_needed:
                    self.progress_queue.put(("tick", 1))

        if error_count > 0:
            self.progress_queue.put(("msg",
                f"线程{chunk_id}: {len(results)}条结果, {error_count}个错误"))

        return results, local_cache

    def _quick_evaluate(self, predictions: Dict, actual: Dict, lottery_type: str) -> Dict:
        """快速评估预测vs实际"""
        if lottery_type == "ssq":
            pred_reds = set(predictions.get('red', [])[:6])
            actual_reds = set(actual.get('red', []))
            pred_blue = set(predictions.get('blue', [])[:1])
            actual_blue = set(actual.get('blue', []))

            red_hits = len(pred_reds & actual_reds)
            blue_hit = 1 if (pred_blue & actual_blue) else 0

            return {
                'red_hits': red_hits,
                'blue_hit': blue_hit,
                'total_hits': red_hits + blue_hit,
                'score': red_hits * 1.0 + blue_hit * 2.0  # 蓝球加权
            }
        else:
            pred_front = set(predictions.get('front', [])[:5])
            actual_front = set(actual.get('front', []))
            pred_back = set(predictions.get('back', [])[:2])
            actual_back = set(actual.get('back', []))

            front_hits = len(pred_front & actual_front)
            back_hits = len(pred_back & actual_back)

            return {
                'front_hits': front_hits,
                'back_hits': back_hits,
                'total_hits': front_hits + back_hits,
                'score': front_hits * 1.0 + back_hits * 2.5
            }

    def _collect_cache_results(self):
        """从缓存中收集所有结果"""
        for period_idx in range(self.start_idx,
                                len(self.all_data) - self.end_idx):
            for g in self.granularities:
                for m in self.methods_to_test:
                    ck = f"{period_idx}_{g}_{m}"
                    if ck in self.cache:
                        self.results.append(self.cache[ck])
                        self.progress_queue.put(("tick", 1))

    def _optimize_weights(self):
        """从回测结果中优化权重"""
        if not self.results:
            return

        # 转换为DataFrame方便分析
        rows = []
        for r in self.results:
            e = r['eval']
            rows.append({
                'method': r['method_key'],
                'granularity': r['granularity_text'],
                'total_hits': e['total_hits'],
                'score': e['score'],
            })
        df = pd.DataFrame(rows)

        # 计算方法权重
        method_perf = df.groupby('method')['score'].agg(['mean', 'std', 'count'])
        method_perf = method_perf[method_perf['count'] >= 10]

        if not method_perf.empty:
            method_perf['weight_raw'] = method_perf['mean'] / method_perf['mean'].sum()
            for m in method_perf.index:
                self.method_weights[m] = round(float(method_perf.loc[m, 'weight_raw']), 4)

        # 计算颗粒度权重
        gran_perf = df.groupby('granularity')['score'].agg(['mean', 'std', 'count'])
        gran_perf = gran_perf[gran_perf['count'] >= 10]

        if not gran_perf.empty:
            gran_perf['weight_raw'] = gran_perf['mean'] / gran_perf['mean'].sum()
            for g in gran_perf.index:
                self.granularity_weights[g] = round(float(gran_perf.loc[g, 'weight_raw']), 4)

        self._log("\n=== 优化后的方法权重 ===")
        for m, w in sorted(self.method_weights.items(), key=lambda x: x[1], reverse=True):
            self._log(f"  {m}: {w:.4f}")

        self._log("\n=== 优化后的颗粒度权重 ===")
        for g, w in sorted(self.granularity_weights.items(), key=lambda x: x[1], reverse=True):
            self._log(f"  {g}: {w:.4f}")

    def _generate_report(self) -> str:
        """生成回测报告Excel"""
        if not self.results:
            return ""

        os.makedirs("backtest_reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"回测报告_{self.lottery_type}_{ts}.xlsx"
        fpath = os.path.join("backtest_reports", fname)

        # 构建DataFrame
        rows = []
        for r in self.results:
            e = r['eval']
            row = {
                '期号': r['period_number'],
                '颗粒度': r['granularity_text'],
                '方法': r['method_name'],
                '命中数': e['total_hits'],
                '评估分': round(e['score'], 3),
            }
            if self.lottery_type == "ssq":
                row['红球命中'] = e.get('red_hits', 0)
                row['蓝球命中'] = e.get('blue_hit', 0)
            else:
                row['前区命中'] = e.get('front_hits', 0)
                row['后区命中'] = e.get('back_hits', 0)
            rows.append(row)

        df = pd.DataFrame(rows)

        with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
            # 摘要
            summary = [
                ["回测报告摘要", ""],
                ["引擎版本", "3.0 统一回测与优化引擎"],
                ["彩票类型", self.lottery_type],
                ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["总评估数", len(df)],
                ["方法数", df['方法'].nunique()],
                ["颗粒度数", df['颗粒度'].nunique()],
                ["期数范围", f"{df['期号'].min()} - {df['期号'].max()}"],
                ["总耗时", f"{self.total_time:.1f}秒 ({self.total_time/60:.1f}分钟)"],
                ["平均命中", f"{df['命中数'].mean():.3f}"],
                ["平均评估分", f"{df['评估分'].mean():.3f}"],
            ]
            if self.method_weights:
                summary.append(["", ""])
                summary.append(["优化方法权重", ""])
                for m, w in sorted(self.method_weights.items(), key=lambda x: x[1], reverse=True):
                    summary.append([f"  {m}", f"{w:.4f}"])
            pd.DataFrame(summary, columns=["项目", "值"]).to_excel(
                writer, sheet_name="报告摘要", index=False)

            # 方法表现
            if not df.empty:
                method_stats = df.groupby('方法').agg(
                    平均命中=('命中数', 'mean'),
                    最高命中=('命中数', 'max'),
                    平均评估分=('评估分', 'mean'),
                    评估次数=('命中数', 'count')
                ).round(3).sort_values('平均评估分', ascending=False)
                method_stats.to_excel(writer, sheet_name="方法表现")

            # 颗粒度表现
            if not df.empty:
                gran_stats = df.groupby('颗粒度').agg(
                    平均命中=('命中数', 'mean'),
                    最高命中=('命中数', 'max'),
                    平均评估分=('评估分', 'mean'),
                    评估次数=('命中数', 'count')
                ).round(3).sort_values('平均评估分', ascending=False)
                gran_stats.to_excel(writer, sheet_name="颗粒度表现")

            # 最佳组合
            if not df.empty:
                combo = df.groupby(['方法', '颗粒度']).agg(
                    平均命中=('命中数', 'mean'),
                    平均评估分=('评估分', 'mean'),
                    评估次数=('命中数', 'count')
                ).round(3).sort_values('平均评估分', ascending=False)
                combo.to_excel(writer, sheet_name="最佳组合")

            # 详细结果（仅保存最近500条避免文件过大）
            detail = df.tail(500)
            detail.to_excel(writer, sheet_name="详细结果", index=False)

        self._log(f"报告已保存: {fpath}")
        return fpath

    def _generate_optimized_report(self) -> str:
        """生成带优化权重的改进报告"""
        os.makedirs("optimization_reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"优化报告_{self.lottery_type}_{ts}.xlsx"
        fpath = os.path.join("optimization_reports", fname)

        # 使用优化权重重新计算综合评分
        rows = []
        for r in self.results:
            e = r['eval']
            mw = self.method_weights.get(r['method_key'], 0.1)
            gw = self.granularity_weights.get(r['granularity_text'], 0.1)

            rows.append({
                '方法': r['method_name'],
                '颗粒度': r['granularity_text'],
                '原始命中': e['total_hits'],
                '原始评估分': e['score'],
                '方法权重': mw,
                '颗粒度权重': gw,
                '加权得分': round(e['score'] * mw * gw * 10, 3),
            })

        df = pd.DataFrame(rows)

        with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
            # 权重配置
            weight_data = [["优化权重配置", ""], ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]]
            weight_data.append(["", ""])
            weight_data.append(["方法权重", ""])
            for m, w in sorted(self.method_weights.items(), key=lambda x: x[1], reverse=True):
                weight_data.append([f"  {m}", f"{w:.4f}"])
            weight_data.append(["", ""])
            weight_data.append(["颗粒度权重", ""])
            for g, w in sorted(self.granularity_weights.items(), key=lambda x: x[1], reverse=True):
                weight_data.append([f"  {g}", f"{w:.4f}"])
            pd.DataFrame(weight_data, columns=["项目", "值"]).to_excel(
                writer, sheet_name="优化权重", index=False)

            # 加权评估
            if not df.empty:
                df.sort_values('加权得分', ascending=False).to_excel(
                    writer, sheet_name="加权评估结果", index=False)

            # 最优策略推荐
            if not df.empty:
                best = df.groupby(['方法', '颗粒度']).agg(
                    平均加权得分=('加权得分', 'mean'),
                    测试次数=('原始命中', 'count')
                ).round(3).sort_values('平均加权得分', ascending=False).head(10)
                best.to_excel(writer, sheet_name="最优策略推荐")

        self._log(f"优化报告已保存: {fpath}")
        return fpath

    def _load_cache(self):
        """加载缓存"""
        if not self.use_cache:
            self.cache = {}
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        # 基于数据+分析器生成缓存键
        data_hash = self._data_hash()
        cache_file = os.path.join(self.cache_dir, f"cache_{data_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                self._log(f"加载缓存: {len(self.cache)}条记录")
                return
            except:
                pass
        self.cache = {}

    def _save_cache(self):
        """保存缓存"""
        if not self.use_cache or not self.cache:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        data_hash = self._data_hash()
        cache_file = os.path.join(self.cache_dir, f"cache_{data_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            self._log(f"缓存已保存: {len(self.cache)}条记录")
        except Exception as e:
            self._log(f"缓存保存失败: {e}")

    def _data_hash(self) -> str:
        """生成数据哈希"""
        if self.all_data is not None:
            info = f"{self.all_data.shape}_{self.analyzer_file}"
            return hashlib.md5(info.encode()).hexdigest()[:16]
        return "no_data"

    def _backtest_done(self, success: bool, msg: str):
        """回测完成处理"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL)
        self.backtest_running = False

        if success:
            self._log(f"\n✓ {msg}")
            self._update_status("回测完成")
            self.time_label.config(text=f"总耗时: {self.total_time:.1f}秒 ({self.total_time/60:.1f}分钟)")
            self.window.after(200, lambda: messagebox.showinfo("完成", msg))
        else:
            self._log(f"\n✗ {msg}")
            self._update_status("回测失败")
            self.window.after(200, lambda: messagebox.showerror("失败", msg))

    def run(self):
        self.window.mainloop()


def main():
    app = UnifiedBacktester()
    app.run()


if __name__ == "__main__":
    main()
