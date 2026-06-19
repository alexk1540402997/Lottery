"""
彩票预测系统 4.0 - GUI界面
==========================
现代扁平化界面，预测/回测/优化/日志四个功能Tab。
文件选择、按钮交互、下拉列表，操作简单直观。
"""

import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入业务模块
from predictor import LotteryPredictor, DEFAULT_PARAMS, METHOD_NAMES_NEW
from merger import ResultMerger, METHOD_NAMES, GRANULARITY_NAMES
from backtester import BacktestEngine, BacktestRunner
from config_manager import ConfigManager

# 颗粒度映射
GRANULARITY_MAP = {
    "50期": 50, "100期": 100, "500期": 500,
    "1000期": 1000, "全部期": 0,
}


class LotterySystemGUI:
    """彩票预测系统 4.0 主界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("彩票号码预测系统 4.0")
        self.root.geometry("1100x780")
        self.root.minsize(900, 600)

        # 配色方案
        self.colors = {
            'bg': '#f8f9fa',
            'sidebar': '#2c3e50',
            'sidebar_active': '#3498db',
            'sidebar_text': '#ecf0f1',
            'primary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'text': '#2c3e50',
            'text_light': '#7f8c8d',
            'border': '#ddd',
            'card': '#ffffff',
        }

        self.root.configure(bg=self.colors['bg'])

        # 状态
        self.data_file = None
        self.data_reverse = None
        self.lottery_type = 'ssq'
        self.predictor = None
        self.all_predictions = {}  # 各颗粒度的预测结果
        self.best_backtest_combo = None  # 最新回测的最优(参数+权重)
        self.backtest_engine = None
        self.backtest_runner = None
        self.config_mgr = ConfigManager()

        # 线程安全
        self.running = False
        self.log_queue = []

        self._setup_ui()
        self._update_log_display()
        self._update_status("就绪")

    # ========================================================================
    #  UI布局
    # ========================================================================

    def _setup_ui(self):
        """构建UI"""
        # === 顶部标题栏 ===
        header = tk.Frame(self.root, bg=self.colors['primary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="彩票号码预测系统 4.0",
                font=("Microsoft YaHei", 18, "bold"),
                fg="white", bg=self.colors['primary']).pack(side=tk.LEFT, padx=25, pady=12)

        tk.Label(header, text="预测 · 回测 · 优化 · 日志",
                font=("Microsoft YaHei", 10),
                fg="#aed6f1", bg=self.colors['primary']).pack(side=tk.RIGHT, padx=25, pady=15)

        # === 主内容区 ===
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # 左侧：文件选择 + 控制面板（可滚动）
        left_outer = tk.Frame(main_frame, bg=self.colors['card'], width=320,
                             relief=tk.FLAT, bd=1)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, bg=self.colors['card'],
                               width=318, highlightthickness=0)
        left_scroll = tk.Scrollbar(left_outer, orient=tk.VERTICAL,
                                  command=left_canvas.yview)
        left_panel = tk.Frame(left_canvas, bg=self.colors['card'])
        left_panel.bind("<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_panel, anchor=tk.NW,
                                 width=318)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 右侧：Notebook (Tab区域)
        right_panel = tk.Frame(main_frame, bg=self.colors['bg'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 创建各Tab
        self.tab_predict = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.tab_backtest = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.tab_optimize = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.tab_log = tk.Frame(self.notebook, bg=self.colors['bg'])

        self.notebook.add(self.tab_predict, text="  预 测  ")
        self.notebook.add(self.tab_backtest, text="  回 测  ")
        self.notebook.add(self.tab_optimize, text="  优 化  ")
        self.notebook.add(self.tab_log, text="  日 志  ")

        # 构建左侧面板
        self._build_left_panel(left_panel)

        # 构建各Tab
        self._build_predict_tab()
        self._build_backtest_tab()
        self._build_optimize_tab()
        self._build_log_tab()

        # === 底部状态栏 ===
        status_frame = tk.Frame(self.root, bg=self.colors['text'], height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="就绪",
                                    font=("Microsoft YaHei", 9),
                                    fg="white", bg=self.colors['text'],
                                    anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        self.progress_bar = ttk.Progressbar(status_frame, length=250, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=15, pady=4)

    def _build_left_panel(self, parent):
        """构建左侧控制面板"""
        # 标题
        tk.Label(parent, text="控制面板", font=("Microsoft YaHei", 13, "bold"),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(15, 5), padx=20,
                                                                      anchor=tk.W)

        # 分隔线
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20)

        # --- 彩票类型 ---
        tk.Label(parent, text="彩票类型", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(
                    pady=(12, 3), padx=20, anchor=tk.W)

        self.lottery_var = tk.StringVar(value="ssq")
        type_frame = tk.Frame(parent, bg=self.colors['card'])
        type_frame.pack(fill=tk.X, padx=20, pady=2)

        tk.Radiobutton(type_frame, text="双色球 (SSQ)", variable=self.lottery_var,
                      value="ssq", font=("Microsoft YaHei", 10),
                      bg=self.colors['card'], activebackground=self.colors['card'],
                      command=self._on_lottery_type_changed).pack(anchor=tk.W, pady=2)
        tk.Radiobutton(type_frame, text="大乐透 (DLT)", variable=self.lottery_var,
                      value="dlt", font=("Microsoft YaHei", 10),
                      bg=self.colors['card'], activebackground=self.colors['card'],
                      command=self._on_lottery_type_changed).pack(anchor=tk.W, pady=2)

        # --- 数据文件 ---
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(parent, text="数据文件", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(
                    pady=(12, 3), padx=20, anchor=tk.W)

        self.file_label = tk.Label(parent, text="未选择文件",
                                  font=("Microsoft YaHei", 9),
                                  bg=self.colors['card'], fg=self.colors['text_light'],
                                  wraplength=280, justify=tk.LEFT)
        self.file_label.pack(padx=20, pady=3, anchor=tk.W)

        btn_frame = tk.Frame(parent, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Button(btn_frame, text="选择文件", command=self._select_file,
                 font=("Microsoft YaHei", 10), bg=self.colors['primary'],
                 fg="white", relief=tk.FLAT, padx=12, pady=4,
                 cursor="hand2").pack(side=tk.LEFT, padx=(0, 8))

        self.load_btn = tk.Button(btn_frame, text="加载", command=self._load_data,
                                 font=("Microsoft YaHei", 10),
                                 bg=self.colors['success'], fg="white",
                                 relief=tk.FLAT, padx=12, pady=4,
                                 state=tk.DISABLED, cursor="hand2")
        self.load_btn.pack(side=tk.LEFT)

        # --- 颗粒度 ---
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(parent, text="分析颗粒度", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(
                    pady=(12, 3), padx=20, anchor=tk.W)

        gran_frame = tk.Frame(parent, bg=self.colors['card'])
        gran_frame.pack(fill=tk.X, padx=20, pady=2)

        # 颗粒度下拉和多选
        self.gran_listbox_vars = {}
        for i, gname in enumerate(GRANULARITY_NAMES):
            var = tk.BooleanVar(value=(gname in ['100期']))
            self.gran_listbox_vars[gname] = var
            tk.Checkbutton(gran_frame, text=gname, variable=var,
                          font=("Microsoft YaHei", 9),
                          bg=self.colors['card'],
                          activebackground=self.colors['card']).pack(anchor=tk.W)

        # --- 方法选择 ---
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(parent, text="分析方法", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(
                    pady=(12, 3), padx=20, anchor=tk.W)

        method_frame = tk.Frame(parent, bg=self.colors['card'])
        method_frame.pack(fill=tk.X, padx=20, pady=2)

        self.method_vars = {}
        # 2列布局，节省垂直空间（13个方法只需7行）
        col_frame = [tk.Frame(method_frame, bg=self.colors['card']),
                     tk.Frame(method_frame, bg=self.colors['card'])]
        col_frame[0].pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        col_frame[1].pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

        methods_list = list(METHOD_NAMES_NEW.items())
        half = (len(methods_list) + 1) // 2
        for i, (mk, mname) in enumerate(methods_list):
            col = 0 if i < half else 1
            var = tk.BooleanVar(value=True)
            self.method_vars[mk] = var
            tk.Checkbutton(col_frame[col], text=mname, variable=var,
                          font=("Microsoft YaHei", 8),
                          bg=self.colors['card'],
                          activebackground=self.colors['card']).pack(anchor=tk.W)

        # --- 快速操作按钮 ---
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(parent, text="快速操作", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['card'], fg=self.colors['text_light']).pack(
                    pady=(12, 3), padx=20, anchor=tk.W)

        quick_frame = tk.Frame(parent, bg=self.colors['card'])
        quick_frame.pack(fill=tk.X, padx=20, pady=5)

        self.btn_predict = tk.Button(quick_frame, text="▶ 开始预测",
                                    command=self._start_prediction,
                                    font=("Microsoft YaHei", 10, "bold"),
                                    bg=self.colors['primary'], fg="white",
                                    relief=tk.FLAT, padx=8, pady=6,
                                    state=tk.DISABLED, cursor="hand2")
        self.btn_predict.pack(fill=tk.X, pady=3)

        self.btn_backtest = tk.Button(quick_frame, text="▶ 开始回测",
                                     command=self._start_backtest,
                                     font=("Microsoft YaHei", 10, "bold"),
                                     bg=self.colors['warning'], fg="white",
                                     relief=tk.FLAT, padx=8, pady=6,
                                     state=tk.DISABLED, cursor="hand2")
        self.btn_backtest.pack(fill=tk.X, pady=3)

        self.btn_stop = tk.Button(quick_frame, text="■ 停止",
                                 command=self._stop_operation,
                                 font=("Microsoft YaHei", 10),
                                 bg=self.colors['danger'], fg="white",
                                 relief=tk.FLAT, padx=8, pady=4,
                                 state=tk.DISABLED, cursor="hand2")
        self.btn_stop.pack(fill=tk.X, pady=3)

    # ========================================================================
    #  预测Tab
    # ========================================================================

    def _build_predict_tab(self):
        """构建预测结果Tab"""
        # 上部分：控制
        ctrl_frame = tk.Frame(self.tab_predict, bg=self.colors['bg'])
        ctrl_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

        tk.Label(ctrl_frame, text="预测结果",
                font=("Microsoft YaHei", 14, "bold"),
                bg=self.colors['bg'], fg=self.colors['text']).pack(side=tk.LEFT)

        tk.Button(ctrl_frame, text="保存结果", command=self._save_predictions,
                 font=("Microsoft YaHei", 10),
                 bg=self.colors['success'], fg="white",
                 relief=tk.FLAT, padx=10, pady=3,
                 state=tk.DISABLED, cursor="hand2").pack(side=tk.RIGHT)

        self.btn_save_pred = ctrl_frame.winfo_children()[-1] if ctrl_frame.winfo_children() else None

        # 结果展示区（Notebook内嵌Tab）
        self.pred_notebook = ttk.Notebook(self.tab_predict)
        self.pred_notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # 创建各方法的结果Tab
        self.pred_tabs = {}
        tab_names = ["综合推荐"] + [f"{METHOD_NAMES_NEW[f'method_{i}']}"
                                 for i in range(1, 14)]

        for name in tab_names:
            frame = tk.Frame(self.pred_notebook, bg=self.colors['bg'])
            text = tk.Text(frame, font=("Consolas", 10), wrap=tk.WORD,
                          bg=self.colors['card'], fg=self.colors['text'],
                          relief=tk.FLAT, bd=5, padx=10, pady=10)
            scroll = tk.Scrollbar(frame, command=text.yview)
            text.config(yscrollcommand=scroll.set)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.pred_notebook.add(frame, text=name[:25])
            self.pred_tabs[name] = text

    def _save_predictions(self):
        """保存预测结果到Excel"""
        if not self.all_predictions:
            messagebox.showwarning("警告", "没有可保存的预测结果")
            return

        from merger import batch_merge_to_excel
        merger = ResultMerger(self.lottery_type)
        try:
            path = batch_merge_to_excel(self.all_predictions, merger)
            messagebox.showinfo("成功", f"结果已保存到:\n{path}")
            self._log(f"预测结果已保存: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    # ========================================================================
    #  回测Tab
    # ========================================================================

    def _build_backtest_tab(self):
        """构建回测Tab"""
        # 设置区
        settings_frame = tk.LabelFrame(self.tab_backtest, text="回测设置",
                                       font=("Microsoft YaHei", 11),
                                       bg=self.colors['bg'],
                                       fg=self.colors['text'])
        settings_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

        # Row 1: 测试期数 + 线程数
        row1 = tk.Frame(settings_frame, bg=self.colors['bg'])
        row1.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(row1, text="测试最新N期:", font=("Microsoft YaHei", 10),
                bg=self.colors['bg']).pack(side=tk.LEFT)
        self.backtest_periods_var = tk.StringVar(value="10")
        periods_combo = ttk.Combobox(row1, textvariable=self.backtest_periods_var,
                                     values=["1", "5", "10", "30", "50", "100", "200"],
                                     width=8, state="readonly", font=("Microsoft YaHei", 10))
        periods_combo.pack(side=tk.LEFT, padx=(5, 30))

        tk.Label(row1, text="并行线程:", font=("Microsoft YaHei", 10),
                bg=self.colors['bg']).pack(side=tk.LEFT)
        self.num_workers_var = tk.StringVar(value="2")
        workers_combo = ttk.Combobox(row1, textvariable=self.num_workers_var,
                                     values=["1", "2", "4", "6", "8"],
                                     width=5, state="readonly", font=("Microsoft YaHei", 10))
        workers_combo.pack(side=tk.LEFT, padx=5)

        # Row 2: 最大搜索时间（唯一的停止条件）
        row2 = tk.Frame(settings_frame, bg=self.colors['bg'])
        row2.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(row2, text="最大搜索时间(分钟):", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['bg'], fg=self.colors['danger']).pack(side=tk.LEFT)
        self.time_limit_var = tk.StringVar(value="0")
        time_combo = ttk.Combobox(row2, textvariable=self.time_limit_var,
                                  values=["0(不限制)", "1", "3", "5", "10", "30",
                                          "60", "120", "240"],
                                  width=12, state="readonly", font=("Microsoft YaHei", 10))
        time_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="← 时间到即停止（无参数组合数量上限）",
                font=("Microsoft YaHei", 8), bg=self.colors['bg'],
                fg=self.colors['text_light']).pack(side=tk.LEFT, padx=10)

        # 说明区
        info_frame = tk.Frame(settings_frame, bg="#e8f5e9", bd=1, relief=tk.SOLID)
        info_frame.pack(fill=tk.X, padx=10, pady=(8, 5))

        info_text = (
            "说明：回测将自动搜索最优的模型参数和合并权重组合。\n"
            "对最新N期的每一期，用历史数据预测 → 加权合并 → 与真实开奖对比命中数。\n"
            "系统会持续尝试不同的参数/权重组合，直到搜索时间耗尽，选出平均命中率最高的组合。\n"
            "所有已尝试的组合会自动记录，下次回测自动跳过，逐步收敛到最优解。"
        )
        tk.Label(info_frame, text=info_text, font=("Microsoft YaHei", 8),
                bg="#e8f5e9", fg="#2e7d32", justify=tk.LEFT,
                wraplength=700).pack(padx=10, pady=8)

        # 进度显示
        self.backtest_elapsed_label = tk.Label(settings_frame, text="",
                                              font=("Microsoft YaHei", 9),
                                              bg=self.colors['bg'],
                                              fg=self.colors['primary'])
        self.backtest_elapsed_label.pack(pady=2)

        # 结果区
        result_frame = tk.LabelFrame(self.tab_backtest, text="回测结果",
                                      font=("Microsoft YaHei", 11),
                                      bg=self.colors['bg'],
                                      fg=self.colors['text'])
        result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.backtest_result_text = tk.Text(result_frame, font=("Consolas", 10),
                                            wrap=tk.WORD, bg=self.colors['card'],
                                            fg=self.colors['text'],
                                            relief=tk.FLAT, bd=5, padx=10, pady=10)
        scroll = tk.Scrollbar(result_frame, command=self.backtest_result_text.yview)
        self.backtest_result_text.config(yscrollcommand=scroll.set)
        self.backtest_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ========================================================================
    #  优化Tab
    # ========================================================================

    def _build_optimize_tab(self):
        """构建优化/日志Tab"""
        # 当前配置显示
        config_frame = tk.LabelFrame(self.tab_optimize, text="当前最优配置",
                                      font=("Microsoft YaHei", 11),
                                      bg=self.colors['bg'],
                                      fg=self.colors['text'])
        config_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 5))

        self.optimize_text = tk.Text(config_frame, font=("Consolas", 10),
                                     wrap=tk.WORD, bg=self.colors['card'],
                                     fg=self.colors['text'],
                                     relief=tk.FLAT, bd=5, padx=10, pady=10,
                                     height=12)
        scroll = tk.Scrollbar(config_frame, command=self.optimize_text.yview)
        self.optimize_text.config(yscrollcommand=scroll.set)
        self.optimize_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 版本管理按钮
        btn_frame = tk.Frame(self.tab_optimize, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Button(btn_frame, text="加载当前配置", command=self._refresh_optimize_display,
                 font=("Microsoft YaHei", 10), relief=tk.FLAT,
                 padx=10, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="回退到上一版本", command=self._rollback_config,
                 font=("Microsoft YaHei", 10), bg=self.colors['warning'],
                 fg="white", relief=tk.FLAT, padx=10, pady=4,
                 cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="重置为默认", command=self._reset_config,
                 font=("Microsoft YaHei", 10), bg=self.colors['danger'],
                 fg="white", relief=tk.FLAT, padx=10, pady=4,
                 cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="导出配置备份", command=self._export_config,
                 font=("Microsoft YaHei", 10), relief=tk.FLAT,
                 padx=10, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="★ 应用最优组合", command=self._apply_best_combo,
                 font=("Microsoft YaHei", 10, "bold"),
                 bg=self.colors['primary'], fg="white",
                 relief=tk.FLAT, padx=12, pady=4,
                 cursor="hand2").pack(side=tk.LEFT, padx=10)

        # 方法权重调整区
        weight_frame = tk.LabelFrame(self.tab_optimize, text="合并权重调整（手动微调）",
                                      font=("Microsoft YaHei", 11),
                                      bg=self.colors['bg'],
                                      fg=self.colors['text'])
        weight_frame.pack(fill=tk.X, padx=15, pady=5)

        self.weight_vars = {}
        wf1 = tk.Frame(weight_frame, bg=self.colors['bg'])
        wf1.pack(fill=tk.X, padx=10, pady=5)
        for i, (mk, mname) in enumerate(METHOD_NAMES_NEW.items()):
            col = i % 4
            row = i // 4
            f = tk.Frame(wf1 if i < 4 else (
                tk.Frame(weight_frame, bg=self.colors['bg'])), bg=self.colors['bg'])
            f.pack(side=tk.LEFT, padx=8, pady=3)

            tk.Label(f, text=f"{mname[:4]}:", font=("Microsoft YaHei", 8),
                    bg=self.colors['bg']).pack(side=tk.LEFT)
            var = tk.StringVar(value="1.0")
            self.weight_vars[mk] = var
            tk.Entry(f, textvariable=var, width=5, font=("Microsoft YaHei", 8)).pack(
                side=tk.LEFT, padx=2)

        if len(METHOD_NAMES_NEW) > 4:
            wf2 = tk.Frame(weight_frame, bg=self.colors['bg'])
            wf2.pack(fill=tk.X, padx=10, pady=5)
            # 重新布局剩余方法
            for i, (mk, mname) in enumerate(METHOD_NAMES_NEW.items()):
                if i < 4:
                    continue
                f = tk.Frame(wf2, bg=self.colors['bg'])
                f.pack(side=tk.LEFT, padx=8, pady=3)
                tk.Label(f, text=f"{mname[:4]}:", font=("Microsoft YaHei", 8),
                        bg=self.colors['bg']).pack(side=tk.LEFT)
                var = tk.StringVar(value="1.0")
                self.weight_vars[mk] = var
                tk.Entry(f, textvariable=var, width=5, font=("Microsoft YaHei", 8)).pack(
                    side=tk.LEFT, padx=2)

        tk.Button(weight_frame, text="应用权重", command=self._apply_weights,
                 font=("Microsoft YaHei", 10), bg=self.colors['success'],
                 fg="white", relief=tk.FLAT, padx=10, pady=3,
                 cursor="hand2").pack(pady=8)

    # ========================================================================
    #  日志Tab
    # ========================================================================

    def _build_log_tab(self):
        """构建日志Tab"""
        self.log_text = tk.Text(self.tab_log, font=("Consolas", 9),
                               wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4",
                               relief=tk.FLAT, bd=5, padx=10, pady=10,
                               insertbackground="white")
        scroll = tk.Scrollbar(self.tab_log, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置日志颜色标签
        self.log_text.tag_config("info", foreground="#4fc3f7")
        self.log_text.tag_config("success", foreground="#66bb6a")
        self.log_text.tag_config("warning", foreground="#ffa726")
        self.log_text.tag_config("error", foreground="#ef5350")
        self.log_text.tag_config("time", foreground="#888")

    # ========================================================================
    #  事件处理
    # ========================================================================

    def _on_lottery_type_changed(self):
        """彩票类型切换"""
        lt = self.lottery_var.get()
        if lt != self.lottery_type:
            self.lottery_type = lt
            self.data_reverse = None
            self.all_predictions = {}
            if self.predictor:
                self.predictor = LotteryPredictor(lt)
            self._log(f"彩票类型已切换为: {lt}")

    def _select_file(self):
        """选择Excel数据文件"""
        path = filedialog.askopenfilename(
            title="选择彩票历史数据Excel文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if path:
            self.data_file = path
            self.file_label.config(text=os.path.basename(path), fg=self.colors['text'])
            self.load_btn.config(state=tk.NORMAL)
            self._update_status(f"已选择: {os.path.basename(path)}")

    def _load_data(self):
        """加载数据"""
        if not self.data_file:
            return

        self._update_status("加载数据中...")
        self.progress_bar.start()

        def _load():
            try:
                data_rev, lt = LotteryPredictor.load_data(self.data_file)
                self.data_reverse = data_rev
                self.lottery_type = lt
                self.predictor = LotteryPredictor(lt)
                self.lottery_var.set(lt)

                self.root.after(0, lambda: self._on_data_loaded(lt, len(data_rev)))
            except Exception as e:
                self.root.after(0, lambda: self._on_load_error(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_data_loaded(self, lt: str, count: int):
        """数据加载完成"""
        self.progress_bar.stop()
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_backtest.config(state=tk.NORMAL)
        self._update_status(f"数据加载成功: {count}条{lt}记录")
        self._log(f"✓ 数据加载成功: {count}条{lt}记录", "success")

        # 显示数据摘要
        latest = self.data_reverse.iloc[0]
        oldest = self.data_reverse.iloc[-1]
        if lt == 'ssq':
            red_str = ' '.join(f'{int(latest[f"red_{i}"]):02d}' for i in range(1,7))
            blue_str = f'{int(latest["blue"]):02d}'
            latest_nums = f'红球:{red_str} 蓝球:{blue_str}'
        else:
            front_str = ' '.join(f'{int(latest[f"front_{i}"]):02d}' for i in range(1,6))
            back_str = ' '.join(f'{int(latest[f"back_{i}"]):02d}' for i in range(1,3))
            latest_nums = f'前区:{front_str} 后区:{back_str}'
        self._log(f"  最新期号: {latest['period']}  {latest_nums}")
        self._log(f"  最早期号: {oldest['period']}")
        self._log(f"  总期数: {count}")

    def _on_load_error(self, error: str):
        """数据加载失败"""
        self.progress_bar.stop()
        self._update_status(f"加载失败: {error}")
        self._log(f"✗ 数据加载失败: {error}", "error")
        messagebox.showerror("加载失败", error)

    def _start_prediction(self):
        """执行预测"""
        if self.data_reverse is None or self.predictor is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        # 收集选中的颗粒度
        selected_grans = [gname for gname, var in self.gran_listbox_vars.items()
                         if var.get()]
        if not selected_grans:
            messagebox.showwarning("警告", "请至少选择一个颗粒度")
            return

        self.running = True
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._update_status("正在预测...")
        self.progress_bar.start()

        # 清空旧结果
        for tab in self.pred_tabs.values():
            tab.delete(1.0, tk.END)
            tab.insert(tk.END, "分析中，请稍候...\n")

        def _run():
            try:
                all_results = {}
                total = len(selected_grans)
                for i, gname in enumerate(selected_grans):
                    if not self.running:
                        break
                    gran = GRANULARITY_MAP[gname]

                    if gran == 0:
                        train_data = self.data_reverse.copy()
                    else:
                        train_data = self.data_reverse.head(gran)

                    if len(train_data) < 10:
                        continue

                    self.root.after(0, lambda g=gname: self._update_status(
                        f"正在分析: {g}..."))
                    self._log(f"分析中: {gname} ({len(train_data)}期)")

                    results = self.predictor.predict_all(train_data, seed=42)
                    all_results[gname] = results

                self.all_predictions = all_results
                self.root.after(0, lambda: self._on_prediction_done(all_results))
            except Exception as e:
                self.root.after(0, lambda: self._on_operation_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_prediction_done(self, all_results: Dict):
        """预测完成"""
        self.progress_bar.stop()
        self.running = False
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        if not all_results:
            self._update_status("预测失败：无结果")
            self._log("✗ 预测失败：没有获得任何结果", "error")
            return

        self._update_status("预测完成!")
        self._log(f"✓ 预测完成! 颗粒度: {list(all_results.keys())}", "success")

        # 显示所有颗粒度的结果（每个方法Tab内按颗粒度分段展示）
        tab_map = {'comprehensive': '综合推荐'}
        for mk, mname in METHOD_NAMES_NEW.items():
            tab_map[mk] = mname

        # 先清空所有Tab
        for text_widget in self.pred_tabs.values():
            text_widget.delete(1.0, tk.END)

        # 按方法聚合：收集每个方法在各颗粒度下的结果
        method_results = {}  # {tab_name: [(gran_name, result), ...]}
        for gran_name, gran_results in all_results.items():
            for key, result in gran_results.items():
                tab_name = tab_map.get(key, key)
                if tab_name not in method_results:
                    method_results[tab_name] = []
                method_results[tab_name].append((gran_name, result))

        # 渲染每个Tab
        for tab_name, gran_list in method_results.items():
            if tab_name not in self.pred_tabs:
                continue
            text_widget = self.pred_tabs[tab_name]

            # 取第一个有效结果显示方法描述
            first_valid = None
            for gn, r in gran_list:
                if 'error' not in r:
                    first_valid = r
                    break
            if first_valid:
                text_widget.insert(tk.END, f"方法: {first_valid.get('method', '')}\n")
                text_widget.insert(tk.END, f"描述: {first_valid.get('description', '')}\n\n")

            # 逐个颗粒度显示
            for gran_name, result in gran_list:
                text_widget.insert(tk.END, f"━━━ [{gran_name}] ━━━\n")

                if 'error' in result:
                    text_widget.insert(tk.END, f"  错误: {result['error']}\n\n")
                    continue

                pred = result.get('predictions', {})
                if self.lottery_type == 'ssq':
                    reds = pred.get('red', [])
                    blues = pred.get('blue', [])
                    text_widget.insert(tk.END,
                        f"  红球: {'  '.join(f'{n:02d}' for n in reds[:6])}\n")
                    text_widget.insert(tk.END,
                        f"  蓝球: {'  '.join(f'{n:02d}' for n in blues[:1])}\n")
                else:
                    fronts = pred.get('front', [])
                    backs = pred.get('back', [])
                    text_widget.insert(tk.END,
                        f"  前区: {'  '.join(f'{n:02d}' for n in fronts[:5])}\n")
                    text_widget.insert(tk.END,
                        f"  后区: {'  '.join(f'{n:02d}' for n in backs[:2])}\n")

                # 统计信息（只显示第一条，避免重复）
                if gran_name == gran_list[0][0]:
                    if 'statistics' in result:
                        text_widget.insert(tk.END, "  [统计]\n")
                        for k, v in result['statistics'].items():
                            text_widget.insert(tk.END, f"    {k}: {v}\n")
                    if 'patterns' in result:
                        text_widget.insert(tk.END, "  [模式]\n")
                        for k, v in result['patterns'].items():
                            text_widget.insert(tk.END, f"    {k}: {v}\n")
                text_widget.insert(tk.END, "\n")

        # 切换到预测Tab
        self.notebook.select(0)

        # 启用保存
        for child in self.tab_predict.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button) and btn['text'] == '保存结果':
                        btn.config(state=tk.NORMAL)

    def _start_backtest(self):
        """启动回测"""
        if self.data_reverse is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        test_periods = int(self.backtest_periods_var.get())
        time_limit_str = self.time_limit_var.get()
        num_workers = int(self.num_workers_var.get())

        # 解析时间限制
        if '不' in time_limit_str or time_limit_str == '0':
            max_search_time = 0
        else:
            max_search_time = int(time_limit_str) * 60

        self.running = True
        self.btn_backtest.config(state=tk.DISABLED)
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._update_status(f"回测中... (测试最新{test_periods}期, "
                           f"{num_workers}线程, "
                           f"{'不限时' if max_search_time==0 else f'{max_search_time//60}分钟'})")

        self.progress_bar.start()  # 不确定模式（无法预知总组合数）

        self.backtest_result_text.delete(1.0, tk.END)
        self.backtest_result_text.insert(tk.END,
            "回测初始化中...\n"
            f"目标: 测试最新{test_periods}期\n"
            f"搜索: 模型参数 + 合并权重\n"
            f"停止: {'不限' if max_search_time==0 else f'{max_search_time//60}分钟后自动停止'}\n"
            f"线程: {num_workers}个并行\n"
            f"已记录{len(self.backtest_engine.tried_combos) if self.backtest_engine else 0}组已尝试组合\n\n"
            "等待首个结果中...\n")
        self.notebook.select(1)

        # 启动计时器更新耗时显示
        self._backtest_start_clock = time.time()
        self._update_backtest_clock()

        # 创建引擎
        self.backtest_engine = BacktestEngine()
        ok, msg = self.backtest_engine.load_data(self.data_file)
        if not ok:
            self._log(f"回测数据加载失败: {msg}", "error")
            self._on_operation_error(msg)
            return

        self.backtest_engine.set_config(
            test_periods=test_periods,
            max_search_time=max_search_time,
            num_workers=num_workers,
        )

        def on_progress(pct, msg):
            self.root.after(0, lambda: self._update_status(
                f"回测中: {msg}"))

        def on_log(msg):
            self._log(msg)

        def on_done(result):
            self.root.after(0, lambda: self._on_backtest_done(result))

        self.backtest_runner = BacktestRunner(self.backtest_engine)
        self.backtest_runner.run_async(
            on_progress=on_progress,
            on_log=on_log,
            on_done=on_done,
        )

    def _update_backtest_clock(self):
        """更新回测耗时显示"""
        if self.running and hasattr(self, '_backtest_start_clock'):
            elapsed = time.time() - self._backtest_start_clock
            self.backtest_elapsed_label.config(
                text=f"已运行: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
            self.root.after(1000, self._update_backtest_clock)

    def _on_backtest_done(self, result: Dict):
        """回测完成"""
        self.progress_bar.stop()
        self.running = False
        self.btn_backtest.config(state=tk.NORMAL)
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        self.backtest_result_text.delete(1.0, tk.END)
        text = self.backtest_result_text

        if not result.get('success'):
            error = result.get('error', '未知错误')
            total_time = result.get('total_time', 0)
            self._update_status(f"回测结束: {error}")
            self._log(f"回测结束: {error}", "warning")
            text.insert(tk.END,
                f"╔══════════════════════════════════╗\n"
                f"║     回测结束（未找到结果）      ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"原因: {error}\n"
                f"耗时: {total_time:.0f}秒\n\n"
                f"建议: 增加搜索时间或在GUI中加载更多期数据\n")
            self.backtest_elapsed_label.config(text="")
            return

        # ============ 显示结果 ============
        best = result['best_combo']
        # 保存最优组合供"应用最优组合"按钮使用
        self.best_backtest_combo = {
            'params': best.get('params', {}),
            'weights': best.get('weights', {}),
            'avg_hits': best.get('avg_total_hits', 0),
        }

        text.insert(tk.END,
            "╔══════════════════════════════════╗\n"
            "║     回 测 结 果 报 告           ║\n"
            "╚══════════════════════════════════╝\n\n")

        text.insert(tk.END, f"彩票类型: {self.lottery_type}\n")
        text.insert(tk.END, f"测试最新N期: {self.backtest_periods_var.get()}\n")
        text.insert(tk.END, f"总尝试组合: {result['total_combos_tried']}组 "
                           f"(跳过{result['total_combos_skipped']}组已试)\n")
        text.insert(tk.END, f"总耗时: {result['total_time']:.0f}秒 "
                           f"({result['total_time']/60:.1f}分钟)\n\n")

        text.insert(tk.END, "━━━ 最佳结果 ━━━\n")
        text.insert(tk.END, f"平均总命中: {best['avg_total_hits']:.4f}\n")
        text.insert(tk.END, f"最高总命中: {best['max_total_hits']}\n")
        text.insert(tk.END, f"命中5+的期占比: {best['hit_rate_5plus']:.1%}\n")
        text.insert(tk.END, f"有效评估期数: {best.get('num_periods_evaluated', 'N/A')}\n\n")

        # 每期详细结果
        text.insert(tk.END, "━━━ 最新各期预测 vs 实际 ━━━\n")
        text.insert(tk.END, f"{'期号':<8} {'预测号码':<38} {'实际号码':<38} {'命中':>4}\n")
        text.insert(tk.END, "-" * 90 + "\n")

        for pr in best.get('period_results', []):
            if self.lottery_type == 'ssq':
                pred_str = (f"红:{' '.join(f'{n:02d}' for n in pr['merged_main'])} "
                           f"蓝:{' '.join(f'{n:02d}' for n in pr['merged_aux'])}")
                actual_str = (f"红:{' '.join(f'{n:02d}' for n in pr['actual_main'])} "
                             f"蓝:{' '.join(f'{n:02d}' for n in pr['actual_aux'])}")
            else:
                pred_str = (f"前:{' '.join(f'{n:02d}' for n in pr['merged_main'])} "
                           f"后:{' '.join(f'{n:02d}' for n in pr['merged_aux'])}")
                actual_str = (f"前:{' '.join(f'{n:02d}' for n in pr['actual_main'])} "
                             f"后:{' '.join(f'{n:02d}' for n in pr['actual_aux'])}")

            text.insert(tk.END,
                f"第{pr['period_num']:<5}期 "
                f"{pred_str:<38} "
                f"{actual_str:<38} "
                f"{pr['total_hits']:>2}中 "
                f"(主{pr['main_hits']}+辅{pr['aux_hits']})\n")

        # 最优权重
        text.insert(tk.END, "\n━━━ 最优方法权重 ━━━\n")
        mw = best['weights'].get('method_weights', {})
        for mk, w in sorted(mw.items(), key=lambda x: x[1], reverse=True):
            mname = METHOD_NAMES_NEW.get(mk, mk)
            text.insert(tk.END, f"  {mname}: {w:.4f}\n")

        text.insert(tk.END, "\n━━━ 最优颗粒度权重 ━━━\n")
        gw = best['weights'].get('granularity_weights', {})
        for gn, w in sorted(gw.items(), key=lambda x: x[1], reverse=True):
            text.insert(tk.END, f"  {gn}: {w:.4f}\n")

        # 最优模型参数
        params = best.get('params', {})
        if params:
            text.insert(tk.END, "\n━━━ 最优模型参数 ━━━\n")
            param_method_names = {
                'statistical': '方法1: 统计概率分析',
                'timeseries': '方法2: 时间序列分析',
                'pattern': '方法3: 模式识别分析',
                'ml': '方法4: LightGBM',
                'markov': '方法5: 马尔可夫分析',
                'montecarlo': '方法6: 蒙特卡罗模拟',
                'clustering': '方法7: 聚类分析',
                'ngram': '方法8: N-gram分析',
                'xgboost': '方法9: XGBoost',
                'bayesian': '方法10: 贝叶斯推断',
                'kalman': '方法11: 卡尔曼滤波',
                'poisson': '方法12: 泊松回归',
                'cooccurrence': '方法13: 共生矩阵分析',
            }
            for method_key, method_label in param_method_names.items():
                method_params = params.get(method_key, {})
                if method_params:
                    text.insert(tk.END, f"  [{method_label}]\n")
                    for pname, pval in sorted(method_params.items()):
                        text.insert(tk.END, f"    {pname}: {pval}\n")

        # 保存版本
        self.config_mgr.save_params_version(
            best.get('params', {}),
            description=f"回测优化 (平均命中{best['avg_total_hits']:.3f}, "
                       f"{best['hit_rate_5plus']:.1%}命中5+)",
            lottery_type=self.lottery_type,
            backtest_score=best['avg_total_hits'],
        )
        self.config_mgr.save_weights_version(
            mw, gw,
            description=f"回测优化权重 (平均命中{best['avg_total_hits']:.3f})",
            backtest_score=best['avg_total_hits'],
        )

        # 生成Excel报告
        if self.backtest_engine:
            report_path = self.backtest_engine.generate_report()
            if report_path:
                text.insert(tk.END, f"\n📊 详细报告: {report_path}\n")

        self._update_status(
            f"回测完成! 最佳平均命中{best['avg_total_hits']:.3f}, "
            f"最高{best['max_total_hits']}中, "
            f"耗时{result['total_time']:.0f}秒")
        self._log(f"回测完成! 最佳={best['avg_total_hits']:.3f}, "
                  f"最高={best['max_total_hits']}, "
                  f"尝试{result['total_combos_tried']}组", "success")

        self.backtest_elapsed_label.config(text="")
        self._refresh_optimize_display()

    def _stop_operation(self):
        """停止当前操作"""
        self.running = False
        if self.backtest_runner:
            self.backtest_runner.stop()
        self.progress_bar.stop()
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_backtest.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._update_status("已停止")
        self._log("操作已停止", "warning")

    def _on_operation_error(self, error: str):
        """操作异常"""
        self.progress_bar.stop()
        self.running = False
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_backtest.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._update_status(f"错误: {error}")
        self._log(f"✗ 操作异常: {error}", "error")
        messagebox.showerror("错误", error)

    # ========================================================================
    #  优化/配置管理
    # ========================================================================

    def _refresh_optimize_display(self):
        """刷新优化Tab的配置显示"""
        self.optimize_text.delete(1.0, tk.END)
        config = self.config_mgr.get_current_config()

        text = self.optimize_text
        text.insert(tk.END, "╔══════════════════════════════════╗\n")
        text.insert(tk.END, "║     当 前 配 置 信 息           ║\n")
        text.insert(tk.END, "╚══════════════════════════════════╝\n\n")

        # 参数版本信息
        meta = config['params'].get('_meta', {})
        text.insert(tk.END, f"模型参数版本: v{meta.get('version', 0)}\n")
        text.insert(tk.END, f"创建时间: {meta.get('created', 'N/A')}\n")
        text.insert(tk.END, f"描述: {meta.get('description', 'N/A')}\n")
        text.insert(tk.END, f"回测得份: {meta.get('backtest_score', 'N/A')}\n\n")

        # 权重版本信息
        wmeta = config['weights'].get('_meta', {})
        text.insert(tk.END, f"权重配置版本: v{wmeta.get('version', 0)}\n")
        text.insert(tk.END, f"创建时间: {wmeta.get('created', 'N/A')}\n")
        text.insert(tk.END, f"描述: {wmeta.get('description', 'N/A')}\n\n")

        # 版本统计
        text.insert(tk.END, f"参数版本总数: {config['param_versions_count']}\n")
        text.insert(tk.END, f"权重版本总数: {config['weight_versions_count']}\n\n")

        # 当前方法权重
        mw = config['weights'].get('method_weights', {})
        if mw:
            text.insert(tk.END, "━━━ 当前方法权重 ━━━\n")
            for mk, w in sorted(mw.items(), key=lambda x: x[1], reverse=True):
                mname = METHOD_NAMES_NEW.get(mk, mk)
                text.insert(tk.END, f"  {mname}: {w:.4f}\n")

        # 当前颗粒度权重
        gw = config['weights'].get('granularity_weights', {})
        if gw:
            text.insert(tk.END, "\n━━━ 当前颗粒度权重 ━━━\n")
            for gn, w in sorted(gw.items(), key=lambda x: x[1], reverse=True):
                text.insert(tk.END, f"  {gn}: {w:.4f}\n")

        # 当前模型参数
        text.insert(tk.END, "\n━━━ 当前模型参数 ━━━\n")
        param_method_names = {
            'statistical': '方法1: 统计概率分析',
            'timeseries': '方法2: 时间序列分析',
            'pattern': '方法3: 模式识别分析',
            'ml': '方法4: LightGBM',
            'markov': '方法5: 马尔可夫分析',
            'montecarlo': '方法6: 蒙特卡罗模拟',
            'clustering': '方法7: 聚类分析',
            'ngram': '方法8: N-gram分析',
            'xgboost': '方法9: XGBoost',
            'bayesian': '方法10: 贝叶斯推断',
            'kalman': '方法11: 卡尔曼滤波',
            'poisson': '方法12: 泊松回归',
            'cooccurrence': '方法13: 共生矩阵分析',
        }
        has_params = False
        for method_key, method_label in param_method_names.items():
            method_params = config['params'].get(method_key, {})
            if method_params:
                has_params = True
                text.insert(tk.END, f"  [{method_label}]\n")
                for pname, pval in sorted(method_params.items()):
                    text.insert(tk.END, f"    {pname}: {pval}\n")
        if not has_params:
            text.insert(tk.END, "  (使用默认参数)\n")

    def _rollback_config(self):
        """回退配置到上一版本"""
        param_versions = self.config_mgr.list_param_versions()
        if len(param_versions) < 2:
            messagebox.showinfo("提示", "只有默认版本，无法回退")
            return

        # 回退到上一版本（索引1，因为0是最新的）
        target = param_versions[1] if len(param_versions) > 1 else param_versions[0]
        ok, msg, _ = self.config_mgr.rollback_params(target['version'])
        if ok:
            self._log(f"✓ {msg}", "success")
            self._refresh_optimize_display()
            messagebox.showinfo("成功", msg)
        else:
            self._log(f"✗ {msg}", "error")
            messagebox.showerror("失败", msg)

    def _reset_config(self):
        """重置为默认配置"""
        if messagebox.askyesno("确认", "确定要重置所有参数和权重为默认值吗？"):
            ok, msg = self.config_mgr.reset_to_defaults()
            if ok:
                self._log(f"✓ {msg}", "success")
                self._refresh_optimize_display()
            else:
                self._log(f"✗ {msg}", "error")

    def _export_config(self):
        """导出配置备份"""
        try:
            path = self.config_mgr.export_all_versions()
            self._log(f"✓ 配置已导出: {path}", "success")
            messagebox.showinfo("成功", f"配置已导出到:\n{path}")
        except Exception as e:
            self._log(f"✗ 导出失败: {e}", "error")

    def _apply_best_combo(self):
        """应用回测最优组合：将最佳(参数+权重)写入当前配置"""
        if not self.best_backtest_combo:
            messagebox.showwarning("警告", "没有可用的回测结果。\n请先运行回测。")
            return

        combo = self.best_backtest_combo
        avg_hits = combo.get('avg_hits', 0)
        params = combo.get('params', {})
        weights = combo.get('weights', {})

        if not params or not weights:
            messagebox.showerror("错误", "最优组合数据不完整")
            return

        # 确认
        confirm = messagebox.askyesno(
            "确认应用",
            f"将回测最优组合(平均命中{avg_hits:.3f})应用到当前配置？\n\n"
            f"这将覆盖:\n"
            f"  - 所有13种方法的模型参数\n"
            f"  - 方法权重和颗粒度权重\n\n"
            f"应用后，预测和后续回测都将使用此组合。"
        )
        if not confirm:
            return

        try:
            # 保存模型参数
            self.config_mgr.save_params_version(
                params,
                description=f"应用回测最优组合 (平均命中{avg_hits:.3f})",
                lottery_type=self.lottery_type,
                backtest_score=avg_hits,
            )
            # 保存权重
            mw = weights.get('method_weights', {})
            gw = weights.get('granularity_weights', {})
            self.config_mgr.save_weights_version(
                mw, gw,
                description=f"应用回测最优权重 (平均命中{avg_hits:.3f})",
                backtest_score=avg_hits,
            )
            self._log(f"✓ 已应用回测最优组合! 平均命中 {avg_hits:.3f}", "success")
            self._refresh_optimize_display()
            messagebox.showinfo("成功",
                f"最优组合已应用!\n\n"
                f"平均命中: {avg_hits:.3f}\n"
                f"参数方法数: {len(params)}\n"
                f"权重方法数: {len(mw)}\n\n"
                f"优化Tab可查看详情。")
        except Exception as e:
            self._log(f"✗ 应用最优组合失败: {e}", "error")
            messagebox.showerror("失败", f"应用失败: {e}")

    def _apply_weights(self):
        """应用手动调整的权重"""
        try:
            mw = {}
            for mk, var in self.weight_vars.items():
                w = float(var.get())
                if w < 0:
                    raise ValueError(f"权重不能为负数: {w}")
                mw[mk] = w

            self.config_mgr.save_weights_version(
                mw,
                self.config_mgr.current_weights.get('granularity_weights', {}),
                description="手动调整",
            )
            self._log(f"✓ 权重已更新", "success")
            self._refresh_optimize_display()
            messagebox.showinfo("成功", "权重已更新并保存")
        except ValueError as e:
            messagebox.showerror("错误", f"权重格式错误: {e}")

    # ========================================================================
    #  日志
    # ========================================================================

    def _log(self, msg: str, level: str = "info"):
        """添加日志"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.append((ts, msg, level))

    def _update_log_display(self):
        """定期更新日志显示"""
        while self.log_queue:
            ts, msg, level = self.log_queue.pop(0)
            self.log_text.insert(tk.END, f"[{ts}] ", "time")
            self.log_text.insert(tk.END, f"{msg}\n", level)
            self.log_text.see(tk.END)
        self.root.after(200, self._update_log_display)

    def _update_status(self, msg: str):
        """更新状态栏"""
        self.status_label.config(text=f"  {msg}")

    def run(self):
        """启动GUI"""
        self._refresh_optimize_display()
        self.root.mainloop()


# ============================================================================
#  入口
# ============================================================================

def main():
    app = LotterySystemGUI()
    app.run()


if __name__ == "__main__":
    main()
