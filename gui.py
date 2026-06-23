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
from backtester import BacktestEngine, BacktestRunner, SolveRunner, OPTIMIZERS_AVAILABLE
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
        self.config_mgr = ConfigManager(lottery_type=self.lottery_type)

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

        # 鼠标滚轮支持（仅绑定左侧面板，不影响右侧内容区）
        def _on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        for w in (left_canvas, left_panel, left_outer):
            w.bind("<MouseWheel>", _on_left_mousewheel)

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
        status_frame = tk.Frame(self.root, bg=self.colors['primary'], height=32)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="  就绪",
                                    font=("Microsoft YaHei", 9, "bold"),
                                    fg="white", bg=self.colors['primary'],
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

        self.btn_solve = tk.Button(quick_frame, text="◆ 求解模式",
                                   command=self._open_solve_window,
                                   font=("Microsoft YaHei", 10, "bold"),
                                   bg="#8e44ad", fg="white",
                                   relief=tk.FLAT, padx=8, pady=6,
                                   state=tk.DISABLED, cursor="hand2")
        self.btn_solve.pack(fill=tk.X, pady=3)

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

        # 搜索模式选择
        mode_frame = tk.Frame(settings_frame, bg=self.colors['bg'])
        mode_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(mode_frame, text="搜索模式:", font=("Microsoft YaHei", 10, "bold"),
                bg=self.colors['bg'], fg=self.colors['primary']).pack(side=tk.LEFT)
        self.search_mode_var = tk.StringVar(value="从零搜索")
        self.search_mode_combo = ttk.Combobox(mode_frame, textvariable=self.search_mode_var,
            values=["从零搜索", "接续优化"],
            width=14, state="readonly", font=("Microsoft YaHei", 10))
        self.search_mode_combo.pack(side=tk.LEFT, padx=5)
        self.search_mode_combo.bind('<<ComboboxSelected>>', self._on_search_mode_changed)

        # 接续优化：历史组合选择列表
        self.continue_frame = tk.Frame(settings_frame, bg=self.colors['bg'])
        # 初始隐藏，选择接续优化时显示
        tk.Label(self.continue_frame, text="接续目标组合（选一个）:",
                font=("Microsoft YaHei", 9), bg=self.colors['bg'],
                fg=self.colors['text_light']).pack(anchor=tk.W, padx=0, pady=(0, 3))

        combo_tree_frame = tk.Frame(self.continue_frame, bg=self.colors['bg'])
        combo_tree_frame.pack(fill=tk.X)

        combo_cols = ("ID", "平均命中", "最高命中", "阶段")
        self.continue_combo_tree = ttk.Treeview(combo_tree_frame, columns=combo_cols,
            show="headings", height=5, selectmode="browse")
        self.continue_combo_tree.heading("ID", text="ID")
        self.continue_combo_tree.heading("平均命中", text="平均命中")
        self.continue_combo_tree.heading("最高命中", text="最高命中")
        self.continue_combo_tree.heading("阶段", text="阶段")
        self.continue_combo_tree.column("ID", width=50, anchor="center")
        self.continue_combo_tree.column("平均命中", width=70, anchor="center")
        self.continue_combo_tree.column("最高命中", width=70, anchor="center")
        self.continue_combo_tree.column("阶段", width=80, anchor="center")
        self.continue_combo_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._continue_combo_data = {}  # {iid: combo_dict}

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
        """构建优化/日志Tab（可滚动）"""
        # 外层Canvas+滚动条（内容可能超出一屏）
        opt_canvas = tk.Canvas(self.tab_optimize, bg=self.colors['bg'],
                               highlightthickness=0)
        opt_scroll = tk.Scrollbar(self.tab_optimize, orient=tk.VERTICAL,
                                  command=opt_canvas.yview)
        opt_content = tk.Frame(opt_canvas, bg=self.colors['bg'])
        opt_content.bind("<Configure>",
            lambda e: opt_canvas.configure(scrollregion=opt_canvas.bbox("all")))
        _opt_win_id = opt_canvas.create_window((0, 0), window=opt_content, anchor=tk.NW)
        opt_canvas.configure(yscrollcommand=opt_scroll.set)
        opt_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        opt_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮支持（仅在优化Tab内生效）
        def _on_opt_mousewheel(event):
            opt_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        opt_canvas.bind("<MouseWheel>", _on_opt_mousewheel)
        opt_content.bind("<MouseWheel>", _on_opt_mousewheel)

        # 让Canvas内frame宽度跟随窗口
        def _on_opt_configure(event):
            opt_canvas.itemconfig(_opt_win_id, width=event.width)
        opt_canvas.bind("<Configure>", _on_opt_configure)

        # 当前配置显示
        config_frame = tk.LabelFrame(opt_content, text="当前最优配置",
                                      font=("Microsoft YaHei", 11),
                                      bg=self.colors['bg'],
                                      fg=self.colors['text'])
        config_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

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
        btn_frame = tk.Frame(opt_content, bg=self.colors['bg'])
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
        weight_frame = tk.LabelFrame(opt_content, text="合并权重调整（手动微调）",
                                      font=("Microsoft YaHei", 11),
                                      bg=self.colors['bg'],
                                      fg=self.colors['text'])
        weight_frame.pack(fill=tk.X, padx=15, pady=5)

        self.weight_vars = {}
        # 标题
        tk.Label(weight_frame, text="手动调整权重 (65组: 方法@颗粒度, 范围[-500.0,500.0])",
                font=("Microsoft YaHei", 9), bg=self.colors['bg'],
                fg=self.colors['text']).pack(anchor=tk.W, padx=10, pady=(5, 0))

        # 使用 Text 组件显示和编辑 65 个 composite_weights
        wt_frame = tk.Frame(weight_frame, bg=self.colors['bg'])
        wt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.weight_text = tk.Text(wt_frame, font=("Consolas", 9),
                                   width=55, height=12,
                                   bg="#1e1e1e", fg="#d4d4d4",
                                   relief=tk.FLAT, bd=5,
                                   insertbackground="white")
        wt_scroll = tk.Scrollbar(wt_frame, command=self.weight_text.yview)
        self.weight_text.config(yscrollcommand=wt_scroll.set)
        self.weight_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wt_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 预填默认权重
        self._reset_weight_text()

        # 按钮行
        btn_row = tk.Frame(weight_frame, bg=self.colors['bg'])
        btn_row.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(btn_row, text="重置为默认", command=self._reset_weight_text,
                 font=("Microsoft YaHei", 9), bg=self.colors['primary'],
                 fg="white", relief=tk.FLAT, padx=8, pady=2,
                 cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="从当前配置加载", command=self._load_weights_to_editor,
                 font=("Microsoft YaHei", 9), bg=self.colors['primary'],
                 fg="white", relief=tk.FLAT, padx=8, pady=2,
                 cursor="hand2").pack(side=tk.LEFT, padx=2)

        tk.Button(weight_frame, text="应用权重", command=self._apply_weights,
                 font=("Microsoft YaHei", 10), bg=self.colors['success'],
                 fg="white", relief=tk.FLAT, padx=10, pady=3,
                 cursor="hand2").pack(pady=8)

        # 版本历史浏览区
        ver_frame = tk.LabelFrame(opt_content, text="版本历史（参数+权重同步快照）",
                                  font=("Microsoft YaHei", 11),
                                  bg=self.colors['bg'],
                                  fg=self.colors['text'])
        ver_frame.pack(fill=tk.X, padx=15, pady=5)

        tree_frame = tk.Frame(ver_frame, bg=self.colors['bg'])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("#", "类型", "时间", "得分", "描述")
        self.version_tree = ttk.Treeview(tree_frame, columns=columns,
                                          show="headings", height=8,
                                          selectmode="browse")
        self.version_tree.heading("#", text="#")
        self.version_tree.heading("类型", text="类型")
        self.version_tree.heading("时间", text="创建时间")
        self.version_tree.heading("得分", text="回测得份")
        self.version_tree.heading("描述", text="描述")
        self.version_tree.column("#", width=40, anchor="center")
        self.version_tree.column("类型", width=60, anchor="center")
        self.version_tree.column("时间", width=140)
        self.version_tree.column("得分", width=70, anchor="center")
        self.version_tree.column("描述", width=280)

        tree_scroll = tk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                   command=self.version_tree.yview)
        self.version_tree.config(yscrollcommand=tree_scroll.set)
        self.version_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 版本操作按钮
        ver_btn_frame = tk.Frame(ver_frame, bg=self.colors['bg'])
        ver_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Button(ver_btn_frame, text="查看版本详情", command=self._view_version_detail,
                 font=("Microsoft YaHei", 9), bg=self.colors['primary'],
                 fg="white", relief=tk.FLAT, padx=10, pady=3,
                 cursor="hand2").pack(side=tk.LEFT, padx=3)

        tk.Button(ver_btn_frame, text="应用此版本", command=self._apply_selected_version,
                 font=("Microsoft YaHei", 9), bg=self.colors['success'],
                 fg="white", relief=tk.FLAT, padx=10, pady=3,
                 cursor="hand2").pack(side=tk.LEFT, padx=3)

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
        """彩票类型切换（同步切换配置存储）"""
        lt = self.lottery_var.get()
        if lt != self.lottery_type:
            self.lottery_type = lt
            self.data_reverse = None
            self.all_predictions = {}
            if self.predictor:
                self.predictor = LotteryPredictor(lt)
            # ★ 切换到对应彩票类型的参数/权重存储
            self.config_mgr.switch_lottery_type(lt)
            self._log(f"彩票类型已切换为: {lt} (配置存储: {lt})")
            self._refresh_optimize_display()

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
        self.btn_solve.config(state=tk.NORMAL)
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

        # 初始化回测引擎（加载历史记录，供接续优化模式使用）
        self.backtest_engine = BacktestEngine()
        self.backtest_engine.load_data(self.data_file)
        self._log(f"  已加载{len(self.backtest_engine.history_detail)}条历史回测记录")

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
                # ★ 读取当前最优参数和权重（打通回测→预测反哺通道）
                config = self.config_mgr.get_current_config()
                current_params = config['params']
                current_weights = config['weights'].get('composite_weights', {})
                # 去除 _meta 避免传入方法
                current_params = {k: v for k, v in current_params.items()
                                  if not k.startswith('_')}

                # 日志确认参数来源
                param_version = config['params'].get('_meta', {}).get('version', 0)
                weight_version = config['weights'].get('_meta', {}).get('version', 0)
                self.root.after(0, lambda: self._log(
                    f'预测使用: 参数v{param_version}, 权重v{weight_version} '
                    f'({len(current_weights)}个composite_weights)'))

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

                    # ★ 传递当前最优参数运行预测
                    results = self.predictor.predict_all(
                        train_data, params=current_params, seed=42)
                    all_results[gname] = results

                self.all_predictions = all_results
                self.all_composite_weights = current_weights
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

        # 先清空所有Tab
        for text_widget in self.pred_tabs.values():
            text_widget.delete(1.0, tk.END)

        # ════════════════════════════════════════════════════════
        # ★ 综合推荐: 用 65 个 composite_weights 合并所有65组预测
        # ════════════════════════════════════════════════════════
        comp_text = self.pred_tabs.get('综合推荐')
        if comp_text:
            cw = getattr(self, 'all_composite_weights', {})
            n_cw = len(cw)

            from merger import ResultMerger
            merger = ResultMerger(self.lottery_type)
            if cw:
                merger.import_weights({'composite_weights': cw})

            merged = merger.merge_results(all_results)

            comp_text.insert(tk.END, "╔══════════════════════════════════╗\n")
            comp_text.insert(tk.END, "║   综 合 合 并 推 荐             ║\n")
            comp_text.insert(tk.END, "╚══════════════════════════════════╝\n\n")
            comp_text.insert(tk.END,
                f"合并方式: {merged['total_groups']}组预测 × 65个独立复合权重\n"
                f"权重来源: {'当前配置(' + str(n_cw) + '个)' if cw else '默认全1.0'}\n\n")

            comp_text.insert(tk.END,
                f"★ 推荐主球: {' '.join(f'{n:02d}' for n in merged['predictions'][merger.main_name])}\n"
                f"★ 推荐辅助球: {' '.join(f'{n:02d}' for n in merged['predictions'][merger.aux_name])}\n\n")

            # 全部贡献组合（按权重降序）
            if merged.get('top_contributors'):
                comp_text.insert(tk.END, f"━━━ 全部贡献组合 (共{len(merged['top_contributors'])}组) ━━━\n")
                for tc in merged['top_contributors']:
                    comp_text.insert(tk.END,
                        f"  {tc['method_name']} @ {tc['granularity']}: "
                        f"权重={tc['weight']:+.4f}\n")
                comp_text.insert(tk.END, "\n")

        # ════════════════════════════════════════════════════════
        # 13个方法Tab: 按颗粒度分段展示各自预测结果
        # ════════════════════════════════════════════════════════
        tab_map = {}
        for mk, mname in METHOD_NAMES_NEW.items():
            tab_map[mk] = mname

        # 按方法聚合
        method_results = {}
        for gran_name, gran_results in all_results.items():
            for key, result in gran_results.items():
                if key == 'comprehensive':
                    continue  # 综合推荐已单独处理
                tab_name = tab_map.get(key, key)
                if tab_name not in method_results:
                    method_results[tab_name] = []
                method_results[tab_name].append((gran_name, result))

        # 渲染每个方法Tab
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
        """启动回测（支持从零搜索 / 接续优化两种模式）"""
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

        # 判断搜索模式
        search_mode = self.search_mode_var.get()
        is_continue = ('接续' in search_mode)
        seed_combo = None
        if is_continue:
            selection = self.continue_combo_tree.selection()
            if not selection:
                messagebox.showwarning("提示", "接续优化模式需要选择一个历史组合")
                return
            seed_combo = self._continue_combo_data.get(selection[0])
            if not seed_combo:
                messagebox.showerror("错误", "选中组合数据无效")
                return

        self.running = True
        self.btn_backtest.config(state=tk.DISABLED)
        self.btn_predict.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        mode_label = f"接续优化(组合#{seed_combo['combo_id']})" if is_continue else "从零搜索"
        self._update_status(f"回测中... [{mode_label}] 测试最新{test_periods}期, "
                           f"{num_workers}线程, "
                           f"{'不限时' if max_search_time==0 else f'{max_search_time//60}分钟'}")

        self.progress_bar.start()

        self.backtest_result_text.delete(1.0, tk.END)
        info_lines = [
            f"搜索模式: {mode_label}",
            f"目标: 测试最新{test_periods}期",
            f"搜索: 模型参数 + 合并权重",
        ]
        if is_continue and seed_combo:
            info_lines.append(
                f"种子组合: #{seed_combo['combo_id']} "
                f"(原评分{seed_combo.get('avg_hits','?'):.3f})")
        info_lines.append(
            f"停止: {'不限' if max_search_time==0 else f'{max_search_time//60}分钟后自动停止'}")
        info_lines.append(f"线程: {num_workers}个并行")

        self.backtest_result_text.insert(tk.END,
            "回测初始化中...\n" + '\n'.join(info_lines) + "\n\n等待首个结果中...\n")
        self.notebook.select(1)

        # 启动计时器
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

        # 接续优化模式不需要 BO/CMAES 冷启动（以种子为中心扰动）
        if is_continue:
            self._log(f"优化器: 接续优化 (种子扰动)")
        else:
            optimizer_name = self.backtest_engine.init_optimizer()
            self._log(f"优化器: {optimizer_name}")

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
            mode='continue' if is_continue else 'fresh',
            seed_combo=seed_combo,
        )

    def _on_search_mode_changed(self, event=None):
        """搜索模式切换：接续优化时显示组合列表"""
        mode = self.search_mode_var.get()
        if '接续' in mode:
            self.continue_frame.pack(fill=tk.X, padx=10, pady=(5, 0),
                                     before=self.backtest_elapsed_label)
            self._refresh_continue_combo_list()
        else:
            self.continue_frame.pack_forget()

    def _refresh_continue_combo_list(self):
        """刷新接续优化的历史组合列表"""
        self.continue_combo_tree.delete(*self.continue_combo_tree.get_children())
        self._continue_combo_data = {}
        if not self.backtest_engine:
            return
        combos = self.backtest_engine.get_historical_combos()
        for c in combos:
            iid = f"c{c['combo_id']}"
            self.continue_combo_tree.insert("", tk.END, iid=iid,
                values=(c['combo_id'], f"{c['avg_hits']:.3f}",
                       c['max_hits'], c.get('phase', '?')))
            self._continue_combo_data[iid] = c

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

        # 最优权重（65个独立 composite_weights，按绝对值降序）
        cw = best['weights'].get('composite_weights', {})
        sorted_cw = sorted(cw.items(), key=lambda x: abs(x[1]), reverse=True)
        text.insert(tk.END, f"\n━━━ 全部复合权重 (共{len(cw)}个) ━━━\n")
        for key, w in sorted_cw:
            mk, gn = key.split('@', 1)
            mname = METHOD_NAMES_NEW.get(mk, mk)
            text.insert(tk.END, f"  {mname} @ {gn}: {w:+.4f}\n")

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
            composite_weights=cw,
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

    def _open_solve_window(self):
        """打开求解模式独立窗口"""
        if self.data_reverse is None:
            messagebox.showwarning("警告", "请先加载数据")
            return
        SolveWindow(self.root, self.data_file, self.colors)

    def _stop_operation(self):
        """停止当前操作"""
        self.running = False
        if self.backtest_runner:
            self.backtest_runner.stop()
        self.progress_bar.stop()
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_backtest.config(state=tk.NORMAL)
        self.btn_solve.config(state=tk.NORMAL)
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

        # 当前 composite_weights（全部，按绝对值降序）
        cw = config['weights'].get('composite_weights', {})
        if cw:
            text.insert(tk.END,
                f"━━━ 当前复合权重 (全部{len(cw)}个) ━━━\n")
            sorted_cw = sorted(cw.items(), key=lambda x: abs(x[1]), reverse=True)
            for key, w in sorted_cw:
                if '@' in key:
                    mk, gn = key.split('@', 1)
                else:
                    mk, gn = key, '?'
                mname = METHOD_NAMES_NEW.get(mk, mk)
                text.insert(tk.END, f"  {mname} @ {gn}: {w:+.4f}\n")

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

        self._refresh_version_list()

    def _rollback_config(self):
        """回退配置到上一版本（参数+权重同步回退）"""
        param_versions = self.config_mgr.list_param_versions()
        weight_versions = self.config_mgr.list_weight_versions()
        if len(param_versions) < 2:
            messagebox.showinfo("提示", "只有默认版本，无法回退")
            return

        # 回退到上一版本（索引1，因为0是最新的）
        target = param_versions[1] if len(param_versions) > 1 else param_versions[0]
        ok, msg, _ = self.config_mgr.rollback_params(target['version'])
        if not ok:
            self._log(f"✗ {msg}", "error")
            messagebox.showerror("失败", msg)
            return

        # 同步回退权重到对应版本
        if len(weight_versions) >= 2:
            wt_target = weight_versions[min(1, len(weight_versions) - 1)]
            wok, wmsg, _ = self.config_mgr.rollback_weights(wt_target['version'])
            if wok:
                msg += f"\n{wmsg}"
                self._log(f"✓ {wmsg}", "success")

        self._log(f"✓ {msg}", "success")
        self._refresh_optimize_display()
        self._refresh_version_list()
        messagebox.showinfo("成功", msg)

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
            cw = weights.get('composite_weights', {})
            self.config_mgr.save_weights_version(
                composite_weights=cw,
                description=f"应用回测最优权重 (平均命中{avg_hits:.3f})",
                backtest_score=avg_hits,
            )
            self._log(f"✓ 已应用回测最优组合! 平均命中 {avg_hits:.3f}", "success")
            self._refresh_optimize_display()
            messagebox.showinfo("成功",
                f"最优组合已应用!\n\n"
                f"平均命中: {avg_hits:.3f}\n"
                f"参数方法数: {len(params)}\n"
                f"权重组合数: {len(cw)}\n\n"
                f"优化Tab可查看详情。")
        except Exception as e:
            self._log(f"✗ 应用最优组合失败: {e}", "error")
            messagebox.showerror("失败", f"应用失败: {e}")

    # ========================================================================
    #  版本历史浏览
    # ========================================================================

    def _refresh_version_list(self):
        """刷新版本历史列表（参数+权重的合并视图）"""
        self.version_tree.delete(*self.version_tree.get_children())

        param_versions = self.config_mgr.list_param_versions()
        weight_versions = self.config_mgr.list_weight_versions()

        # 构建权重版本的版本号→条目映射
        wv_map = {v['version']: v for v in weight_versions}

        # 按时间倒序合并显示（最新的在前）
        combined = []
        for pv in param_versions:
            pv_time = pv.get('created', '')
            wv = wv_map.get(pv['version'], {})
            wv_time = wv.get('created', '') if wv else ''
            combined.append({
                'version': pv['version'],
                'type': '参数+权重' if wv else '参数',
                'time': pv_time,
                'score': pv.get('backtest_score', 0),
                'desc': pv.get('description', ''),
                'param_version': pv['version'],
                'weight_version': wv.get('version') if wv else None,
            })

        # 加上只存在于权重列表但不在参数列表中的版本
        pv_nums = {v['version'] for v in param_versions}
        for wv in weight_versions:
            if wv['version'] not in pv_nums:
                combined.append({
                    'version': wv['version'],
                    'type': '权重',
                    'time': wv.get('created', ''),
                    'score': wv.get('backtest_score', 0),
                    'desc': wv.get('description', ''),
                    'param_version': None,
                    'weight_version': wv['version'],
                })

        combined.sort(key=lambda x: x['version'], reverse=True)

        for item in combined:
            score_str = f"{item['score']:.2f}" if item['score'] else "—"
            self.version_tree.insert("", tk.END,
                iid=f"v{item['version']}",
                values=(item['version'], item['type'], item['time'],
                       score_str, item['desc']))

        # 存储合并数据供查看/应用使用
        self._version_data = {f"v{item['version']}": item for item in combined}

    def _view_version_detail(self):
        """查看选中版本的完整详情"""
        selection = self.version_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先在版本列表中选择一个版本")
            return

        item_id = selection[0]
        item = self._version_data.get(item_id, {})
        if not item:
            return

        # 构建详情文本
        lines = []
        lines.append("╔══════════════════════════════════╗")
        lines.append(f"║  版本 v{item['version']} 详情{' ' * (23 - len(str(item['version'])))}║")
        lines.append("╚══════════════════════════════════╝")
        lines.append("")
        lines.append(f"类型: {item['type']}")
        lines.append(f"创建时间: {item['time']}")
        lines.append(f"回测得份: {item['score']}")
        lines.append(f"描述: {item['desc']}")

        # 加载参数详情
        if item['param_version']:
            params = self.config_mgr._load_params_version(item['param_version'])
            if params:
                lines.append("")
                lines.append("━━━ 模型参数 ━━━")
                param_names = {
                    'statistical': '方法1: 统计概率分析',
                    'timeseries': '方法2: 时间序列分析',
                    'pattern': '方法3: 模式识别分析',
                    'ml': '方法4: LightGBM',
                    'markov': '方法5: 马尔可夫',
                    'montecarlo': '方法6: 蒙特卡罗',
                    'clustering': '方法7: 聚类分析',
                    'ngram': '方法8: N-gram',
                    'xgboost': '方法9: XGBoost',
                    'bayesian': '方法10: 贝叶斯',
                    'kalman': '方法11: 卡尔曼',
                    'poisson': '方法12: 泊松',
                    'cooccurrence': '方法13: 共生矩阵',
                }
                for mk, ml in param_names.items():
                    mp = params.get(mk, {})
                    if mp:
                        lines.append(f"  [{ml}]")
                        for pn, pv in sorted(mp.items()):
                            lines.append(f"    {pn}: {pv}")

        # 加载权重详情
        if item['weight_version']:
            w = self.config_mgr._load_weights_version(item['weight_version'])
            if w:
                cw = w.get('composite_weights', {})
                if cw:
                    lines.append("")
                    lines.append(f"━━━ 复合权重 (共{len(cw)}个) ━━━")
                    sorted_cw = sorted(cw.items(), key=lambda x: abs(x[1]), reverse=True)
                    for key, val in sorted_cw:
                        lines.append(f"  {key}: {val:+.4f}")

        detail = '\n'.join(lines)

        # 弹出详情窗口
        popup = tk.Toplevel(self.root)
        popup.title(f"版本 v{item['version']} 详情")
        popup.geometry("750x600")
        popup.configure(bg=self.colors['card'])

        detail_text = tk.Text(popup, font=("Consolas", 9),
                             wrap=tk.WORD, bg=self.colors['card'],
                             fg=self.colors['text'],
                             relief=tk.FLAT, bd=10, padx=10, pady=10)
        detail_scroll = tk.Scrollbar(popup, command=detail_text.yview)
        detail_text.config(yscrollcommand=detail_scroll.set)
        detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        detail_text.insert(tk.END, detail)
        detail_text.config(state=tk.DISABLED)

        tk.Button(popup, text="关闭", command=popup.destroy,
                 font=("Microsoft YaHei", 10), bg=self.colors['primary'],
                 fg="white", relief=tk.FLAT, padx=20, pady=5,
                 cursor="hand2").pack(pady=10)

    def _apply_selected_version(self):
        """将选中的历史版本应用到当前配置（→自动传递到预测系统）"""
        selection = self.version_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先在版本列表中选择一个版本")
            return

        item_id = selection[0]
        item = self._version_data.get(item_id, {})
        if not item:
            return

        # 确认
        parts = [f"版本 v{item['version']}"]
        if item['param_version']:
            parts.append("模型参数")
        if item['weight_version']:
            parts.append("合并权重")
        confirm = messagebox.askyesno(
            "确认应用历史版本",
            f"确认将历史版本 v{item['version']} 应用到当前配置？\n\n"
            f"类型: {item['type']}\n"
            f"时间: {item['time']}\n"
            f"得分: {item['score']}\n"
            f"描述: {item['desc']}\n\n"
            f"将覆盖: {' + '.join(parts)}\n\n"
            f"应用后，下次'开始预测'将使用此版本的参数和权重。"
        )
        if not confirm:
            return

        # 应用参数
        msgs = []
        if item['param_version']:
            ok, msg, _ = self.config_mgr.rollback_params(item['param_version'])
            if ok:
                msgs.append(msg)
            else:
                self._log(f"✗ 参数回退失败: {msg}", "error")
                messagebox.showerror("失败", msg)
                return

        # 应用权重
        if item['weight_version']:
            ok, msg, _ = self.config_mgr.rollback_weights(item['weight_version'])
            if ok:
                msgs.append(msg)
            else:
                self._log(f"✗ 权重回退失败: {msg}", "error")
                messagebox.showerror("失败", msg)
                return

        self._log(f"✓ 已应用历史版本 v{item['version']}: {'; '.join(msgs)}", "success")
        self._refresh_optimize_display()
        self._refresh_version_list()
        messagebox.showinfo("成功",
            f"历史版本 v{item['version']} 已应用!\n\n"
            f"{chr(10).join(msgs)}\n\n"
            f"下次'开始预测'将使用此版本的参数和权重。\n"
            f"(当前配置已写入 logs/current_model_params.json\n"
            f" 和 logs/current_merge_weights.json)")

    def _apply_weights(self):
        """应用手动调整的权重（从文本编辑器解析）"""
        try:
            raw_text = self.weight_text.get(1.0, tk.END).strip()
            cw = {}
            for line in raw_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, val_str = line.split(':', 1)
                    key = key.strip()
                    val_str = val_str.strip()
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip()
                    else:
                        continue
                try:
                    w = float(val_str)
                except ValueError:
                    raise ValueError(f"无法解析权重值: {line}")
                if w < -500.0 or w > 500.0:
                    raise ValueError(f"权重超出范围 [-500.0, 500.0]: {key}={w}")
                cw[key] = w

            if not cw:
                raise ValueError("未找到有效的权重条目")

            self.config_mgr.save_weights_version(
                composite_weights=cw,
                description="手动调整",
            )
            self._log(f"✓ 权重已更新 ({len(cw)}个)", "success")
            self._refresh_optimize_display()
            messagebox.showinfo("成功", f"权重已更新并保存 ({len(cw)}个组合)")
        except ValueError as e:
            messagebox.showerror("错误", f"权重格式错误: {e}")

    def _reset_weight_text(self):
        """重置权重编辑器为默认值"""
        self.weight_text.delete(1.0, tk.END)
        lines = []
        for mk in METHOD_NAMES_NEW:
            for gn in ['50期', '100期', '500期', '1000期', '全部期']:
                lines.append(f"{mk}@{gn}: 1.0")
        self.weight_text.insert(tk.END, '\n'.join(lines))

    def _load_weights_to_editor(self):
        """从当前配置加载权重到编辑器"""
        config = self.config_mgr.get_current_config()
        cw = config['weights'].get('composite_weights', {})
        self.weight_text.delete(1.0, tk.END)
        lines = []
        for mk in METHOD_NAMES_NEW:
            for gn in ['50期', '100期', '500期', '1000期', '全部期']:
                key = f"{mk}@{gn}"
                val = cw.get(key, 1.0)
                lines.append(f"{key}: {val:.4f}")
        self.weight_text.insert(tk.END, '\n'.join(lines))

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
#  求解模式独立窗口
# ============================================================================

class SolveWindow:
    """求解模式 — 反向搜索满足容差条件的参数组合"""

    def __init__(self, parent, data_file: str, colors: dict):
        self.data_file = data_file
        self.colors = colors
        self.solve_engine = None
        self.solve_runner = None
        self.running = False
        self.solutions_found = 0

        self.win = tk.Toplevel(parent)
        self.win.title("求解模式 — 容差反向搜索")
        self.win.geometry("900x650")
        self.win.minsize(700, 500)
        self.win.configure(bg=colors['bg'])

        self._setup_ui()
        self._load_engine()

    def _setup_ui(self):
        """构建求解窗口UI"""
        c = self.colors

        # 顶栏
        header = tk.Frame(self.win, bg=c['primary'], height=45)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="◆ 求解模式 — 反向搜索满足容差条件的参数组合",
                font=("Microsoft YaHei", 13, "bold"),
                fg="white", bg=c['primary']).pack(side=tk.LEFT, padx=20, pady=10)

        # 设置区
        settings = tk.LabelFrame(self.win, text="求解设置", font=("Microsoft YaHei", 11),
                                bg=c['bg'], fg=c['text'])
        settings.pack(fill=tk.X, padx=15, pady=(10, 5))

        row1 = tk.Frame(settings, bg=c['bg'])
        row1.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(row1, text="求解期数:", font=("Microsoft YaHei", 10),
                bg=c['bg']).pack(side=tk.LEFT)
        self.periods_var = tk.StringVar(value="1")
        ttk.Combobox(row1, textvariable=self.periods_var,
                    values=["1", "5", "10", "20", "50"],
                    width=6, state="readonly", font=("Microsoft YaHei", 10)
                    ).pack(side=tk.LEFT, padx=(5, 20))

        tk.Label(row1, text="主球最低命中:", font=("Microsoft YaHei", 10),
                bg=c['bg']).pack(side=tk.LEFT)
        self.main_tol_var = tk.StringVar(value="5")
        ttk.Combobox(row1, textvariable=self.main_tol_var,
                    values=["3", "4", "5", "6"],
                    width=4, state="readonly", font=("Microsoft YaHei", 10)
                    ).pack(side=tk.LEFT, padx=(5, 20))

        tk.Label(row1, text="辅助球最低命中:", font=("Microsoft YaHei", 10),
                bg=c['bg']).pack(side=tk.LEFT)
        self.aux_tol_var = tk.StringVar(value="1")
        ttk.Combobox(row1, textvariable=self.aux_tol_var,
                    values=["0", "1", "2"],
                    width=4, state="readonly", font=("Microsoft YaHei", 10)
                    ).pack(side=tk.LEFT, padx=5)

        row2 = tk.Frame(settings, bg=c['bg'])
        row2.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(row2, text="最大时间(分钟):", font=("Microsoft YaHei", 10, "bold"),
                bg=c['bg'], fg=c['danger']).pack(side=tk.LEFT)
        self.time_var = tk.StringVar(value="5")
        ttk.Combobox(row2, textvariable=self.time_var,
                    values=["1", "3", "5", "10", "30", "60", "120", "0(不限)"],
                    width=10, state="readonly", font=("Microsoft YaHei", 10)
                    ).pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="并行线程:", font=("Microsoft YaHei", 10),
                bg=c['bg']).pack(side=tk.LEFT, padx=(20, 5))
        self.worker_var = tk.StringVar(value="4")
        ttk.Combobox(row2, textvariable=self.worker_var,
                    values=["1", "2", "4", "6", "8"],
                    width=4, state="readonly", font=("Microsoft YaHei", 10)
                    ).pack(side=tk.LEFT, padx=5)

        # 求解模式选择（4.3+）
        row3 = tk.Frame(settings, bg=c['bg'])
        row3.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(row3, text="求解模式:", font=("Microsoft YaHei", 10, "bold"),
                bg=c['bg'], fg=c['primary']).pack(side=tk.LEFT)
        self.solve_mode_var = tk.StringVar(value="BO+线性求解")
        self.solve_mode_combo = ttk.Combobox(
            row3, textvariable=self.solve_mode_var,
            values=["BO+线性求解", "回测最优+求解"],
            width=18, state="readonly", font=("Microsoft YaHei", 10))
        self.solve_mode_combo.pack(side=tk.LEFT, padx=5)
        self.solve_mode_combo.bind('<<ComboboxSelected>>', self._on_solve_mode_changed)

        tk.Label(row3, text="(回测最优模式无需容差，直接输出结果)",
                font=("Microsoft YaHei", 8), bg=c['bg'],
                fg=c['text_light']).pack(side=tk.LEFT, padx=10)

        # 按钮
        btn_frame = tk.Frame(settings, bg=c['bg'])
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        self.btn_start = tk.Button(btn_frame, text="▶ 开始求解",
                                  command=self._start_solve,
                                  font=("Microsoft YaHei", 11, "bold"),
                                  bg=c['primary'], fg="white",
                                  relief=tk.FLAT, padx=15, pady=5,
                                  cursor="hand2")
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(btn_frame, text="■ 停止",
                                 command=self._stop_solve,
                                 font=("Microsoft YaHei", 11),
                                 bg=c['danger'], fg="white",
                                 relief=tk.FLAT, padx=15, pady=5,
                                 state=tk.DISABLED, cursor="hand2")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(btn_frame, text="就绪",
                                    font=("Microsoft YaHei", 9),
                                    bg=c['bg'], fg=c['text_light'])
        self.status_label.pack(side=tk.LEFT, padx=15)

        # 进度条
        self.progress = ttk.Progressbar(settings, length=600, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 8))

        # 结果区
        result_frame = tk.LabelFrame(self.win, text="求解结果",
                                    font=("Microsoft YaHei", 11),
                                    bg=c['bg'], fg=c['text'])
        result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.result_text = tk.Text(result_frame, font=("Consolas", 9),
                                  wrap=tk.WORD, bg=c['card'], fg=c['text'],
                                  relief=tk.FLAT, bd=5, padx=10, pady=10)
        scroll = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text.insert(tk.END,
            "求解模式说明:\n"
            "  给定最新N期开奖号码，反向搜索所有满足容差条件的(参数,权重)组合。\n"
            "  例如: 求解1期, 主球≥5, 辅助球≥1 → 找到能精确命中5+1的组合。\n"
            "  智能搜索策略：加权采样+收敛微调+随机脉冲，与回测共享历史数据。\n\n"
            "等待开始...\n")

    def _load_engine(self):
        """加载求解引擎"""
        from backtester import SolveEngine
        self.solve_engine = SolveEngine()
        ok, msg = self.solve_engine.load_data(self.data_file)
        if not ok:
            messagebox.showerror("错误", f"数据加载失败: {msg}")
        else:
            self.result_text.insert(tk.END, f"数据加载成功: {msg}\n")

    def _on_solve_mode_changed(self, event=None):
        """求解模式切换时的UI调整"""
        mode = self.solve_mode_var.get()
        if '回测最优' in mode:
            # 回测最优模式不需要容差设置和长时间搜索
            self.main_tol_var.set("—")
            self.aux_tol_var.set("—")
            self.time_var.set("1")  # 只需很短时间（单次预测+线性求解）
        else:
            # BO+求解模式恢复默认设置
            if self.main_tol_var.get() == "—":
                self.main_tol_var.set("5")
            if self.aux_tol_var.get() == "—":
                self.aux_tol_var.set("1")
            self.time_var.set("5")

    def _log(self, msg: str):
        """添加日志"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.result_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.result_text.see(tk.END)

    def _start_solve(self):
        """开始求解"""
        solve_periods = int(self.periods_var.get())

        # 解析容差（回测最优模式不需要容差）
        try:
            tol_main = int(self.main_tol_var.get())
            tol_aux = int(self.aux_tol_var.get())
        except ValueError:
            tol_main = 5
            tol_aux = 1

        time_str = self.time_var.get()
        num_workers = int(self.worker_var.get())

        max_time = 0 if '不限' in time_str else int(time_str) * 60

        # 确定求解模式
        mode_selection = self.solve_mode_var.get()
        if '回测最优' in mode_selection:
            # 模式: 回测最优 + 线性求解
            self.solve_engine.solve_mode = 'best_params'
            # 此模式不需要 BO 初始化和 use_linear_weights
            # 引擎内部会直接读取最优参数 + 线性求解
        else:
            # 模式: BO + 线性求解
            self.solve_engine.solve_mode = 'bo_linear'
            if OPTIMIZERS_AVAILABLE:
                self.solve_engine.init_bo('solve')
                self.solve_engine.use_linear_weights = True
            else:
                self.solve_engine.solve_mode = 'random'

        self.solve_engine.set_solve_config(
            solve_periods=solve_periods,
            tolerance_main=tol_main,
            tolerance_aux=tol_aux,
            max_search_time=max_time,
            num_workers=num_workers,
        )

        self.running = True
        self.solutions_found = 0
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress.start()
        self.status_label.config(text="求解中...")

        # 日志
        mode_labels = {
            'best_params': '回测最优+线性求解',
            'bo_linear': 'BO+线性求解',
            'random': '随机搜索',
        }
        mode_label = mode_labels.get(self.solve_engine.solve_mode, '未知')

        self.result_text.delete(1.0, tk.END)
        if self.solve_engine.solve_mode == 'best_params':
            self._log(f"求解启动 [{mode_label}]: 最新{solve_periods}期, "
                     f"固定回测最优参数 + 线性求解权重")
        else:
            self._log(f"求解启动 [{mode_label}]: 最新{solve_periods}期, "
                     f"主球≥{tol_main}, 辅助球≥{tol_aux}, "
                     f"时间={'不限' if max_time==0 else f'{max_time//60}分钟'}, "
                     f"{num_workers}线程")

        def on_progress(pct, msg):
            self.win.after(0, lambda: self.status_label.config(text=msg))

        def on_log(msg):
            self.win.after(0, lambda: self._log(msg))

        def on_done(result):
            self.win.after(0, lambda: self._on_solve_done(result))

        self.solve_runner = SolveRunner(self.solve_engine)
        self.solve_runner.run_async(
            on_progress=on_progress,
            on_log=on_log,
            on_done=on_done,
        )

    def _stop_solve(self):
        """停止求解"""
        self.running = False
        if self.solve_runner:
            self.solve_runner.stop()
        self.progress.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="已停止")

    def _on_solve_done(self, result: dict):
        """求解完成"""
        self.progress.stop()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        self.result_text.delete(1.0, tk.END)

        if not result.get('success'):
            self.status_label.config(text="求解失败")
            self.result_text.insert(tk.END,
                f"求解失败: {result.get('error', '未知错误')}\n")
            return

        solve_mode = result.get('solve_mode', 'default')
        total_time = result.get('total_time', 0)
        cfg = result.get('solve_config', {})

        self.result_text.insert(tk.END,
            "╔══════════════════════════════════╗\n"
            "║     求 解 结 果 报 告           ║\n"
            "╚══════════════════════════════════╝\n\n")

        # ── 回测最优 + 求解模式 ──
        if solve_mode == 'best_params':
            self._show_best_params_result(result, cfg, total_time)

        # ── BO+求解 / 随机模式 ──
        else:
            self._show_bo_solve_result(result, cfg, total_time)

    def _show_best_params_result(self, result, cfg, total_time):
        """展示回测最优+求解模式的结果（重点：权重配方）"""
        verification = result.get('verification', {})
        all_verified = verification.get('all_verified', False)
        prediction = result.get('prediction')
        lp_success = result.get('lp_success', False)
        lp_status = result.get('lp_status', '?')

        # ── 头信息 ──
        solver_label = 'LP精确求解' if lp_success else 'LSTSQ降级(近似解)'
        self.result_text.insert(tk.END,
            f"求解模式: 回测最优 + 线性权重求解\n"
            f"求解方法: {solver_label}\n"
            f"求解状态: {lp_status}\n"
            f"参数来源: {result.get('param_source', '?')}\n"
            f"参数评分: {result.get('param_score', '?')}\n"
            f"总耗时: {total_time:.1f}秒\n\n")

        # LP 不可行时的诊断
        lp_diag = result.get('lp_diagnostic', {})
        if lp_diag.get('is_infeasible'):
            uncovered_m = lp_diag.get('uncovered_main', [])
            uncovered_a = lp_diag.get('uncovered_aux', [])
            if uncovered_m or uncovered_a:
                self.result_text.insert(tk.END,
                    "━━━ 诊断：LP为何不可行 ━━━\n\n"
                    "以下实际号码未被任何方法预测到（参数质量不足）:\n")
                if uncovered_m:
                    self.result_text.insert(tk.END,
                        f"  主球未覆盖: {' '.join(f'{n:02d}' for n in uncovered_m)}\n")
                if uncovered_a:
                    self.result_text.insert(tk.END,
                        f"  辅助球未覆盖: {' '.join(f'{n:02d}' for n in uncovered_a)}\n")
                self.result_text.insert(tk.END,
                    "\n  → 请增加回测运行时间以优化模型参数。\n"
                    "  → 参数优化后，这些号码被方法覆盖，LP即可精确求解。\n\n")

        # ── ★ 核心产出: 复合权重 (方法@颗粒度) ──
        composite_weights = result.get('composite_weights', {})
        if composite_weights:
            self.result_text.insert(tk.END,
                f"━━━ ★ 复合权重 (全部{len(composite_weights)}个) (方法@颗粒度) ★━━━\n\n")
            method_names = {
                'method_1': '统计概率', 'method_2': '时间序列', 'method_3': '模式识别',
                'method_4': 'LightGBM', 'method_5': '马尔可夫', 'method_6': '蒙特卡罗',
                'method_7': '聚类分析', 'method_8': 'N-gram', 'method_9': 'XGBoost',
                'method_10': '贝叶斯推断', 'method_11': '卡尔曼滤波', 'method_12': '泊松回归',
                'method_13': '共生矩阵',
            }
            sorted_cw = sorted(composite_weights.items(),
                              key=lambda x: abs(x[1]), reverse=True)
            max_abs = max((abs(wv) for _, wv in sorted_cw), default=1)
            for ck, wv in sorted_cw:  # 显示全部65个
                if '@' in ck:
                    mk, gn = ck.split('@', 1)
                else:
                    mk, gn = ck, '?'
                name = method_names.get(mk, mk)
                # 条形图按最大权重等比缩放，最长50字符，不换行
                bar_len = max(1, int(abs(wv) / max_abs * 50))
                bar = '█' * bar_len
                sign = '+' if wv >= 0 else ''
                self.result_text.insert(tk.END,
                    f"  {name:<10} @ {gn:<8} {sign}{wv:>7.4f}  {bar}\n")
            self.result_text.insert(tk.END, "\n")

        # ── ★ 验算: 配方能否还原实际号码 ──
        self.result_text.insert(tk.END,
            "━━━ ★ 验算：配方是否完全还原实际号码 ━━━\n\n")
        verify_status = '[OK] 全部完全重合 — 配方正确！' if all_verified else '[FAIL] 存在不重合 — 参数需继续优化'
        self.result_text.insert(tk.END, f"  验算结果: {verify_status}\n\n")

        for vd in verification.get('details', []):
            match_icon = '[OK]' if vd['all_match'] else '[FAIL]'
            self.result_text.insert(tk.END,
                f"  第{vd['period_num']}期 {match_icon}\n"
                f"    合并主球: {' '.join(f'{n:02d}' for n in vd['merged_main'])}\n"
                f"    实际主球: {' '.join(f'{n:02d}' for n in vd['actual_main'])}\n"
                f"    合并辅助: {' '.join(f'{n:02d}' for n in vd['merged_aux'])}\n"
                f"    实际辅助: {' '.join(f'{n:02d}' for n in vd['actual_aux'])}\n"
                f"    主球{'重合' if vd['main_match'] else str(vd['main_hits']) + '个命中'}, "
                f"辅助{'重合' if vd['aux_match'] else str(vd['aux_hits']) + '个命中'}\n\n")

        # ── ★ 预测: 未开奖最新一期 ──
        if prediction:
            self.result_text.insert(tk.END,
                "━━━ ★ 预测：未开奖最新一期 ━━━\n\n"
                f"  预测主球: {' '.join(f'{n:02d}' for n in prediction['main'])}\n"
                f"  预测辅助: {' '.join(f'{n:02d}' for n in prediction['aux'])}\n"
                f"  (基于 {prediction.get('num_method_predictions', '?')} 组预测投票)\n\n")

        self.status_label.config(
            text=f"完成! 验算={'通过' if all_verified else '未通过'}, 耗时{total_time:.0f}s")

    def _show_bo_solve_result(self, result, cfg, total_time):
        """展示 BO+求解 / 随机模式的结果"""
        solutions = result.get('solutions', [])
        total = result.get('total_evaluated', 0)

        self.result_text.insert(tk.END,
            f"求解期数: {cfg.get('periods', '?')}期\n"
            f"容差条件: 主球≥{cfg.get('tolerance_main', '?')}, "
            f"辅助球≥{cfg.get('tolerance_aux', '?')}\n"
            f"总评估组合: {total}组\n"
            f"找到解: {len(solutions)}个\n"
            f"总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)\n\n")

        if solutions:
            self.result_text.insert(tk.END,
                f"━━━ 找到 {len(solutions)} 个有效解 ━━━\n\n")
            for i, sol in enumerate(solutions):
                self.result_text.insert(tk.END,
                    f"【解 #{i+1}】 组合ID={sol['combo_id']} "
                    f"平均命中={sol['avg_total_hits']:.3f} "
                    f"最高={sol['max_total_hits']}\n")
        else:
            self.result_text.insert(tk.END,
                "未找到满足容差条件的解。\n"
                "建议: 降低容差条件或延长搜索时间。\n")

        self.status_label.config(
            text=f"完成! 找到{len(solutions)}个解, 耗时{total_time:.0f}s")

# ============================================================================
#  入口
# ============================================================================

def main():
    app = LotterySystemGUI()
    app.run()


if __name__ == "__main__":
    main()
