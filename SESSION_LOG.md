# 彩票号码预测系统 — 会话状态日志

> 最后更新：2026-06-19  
> 当前版本：4.2（智能搜索 + 13种方法）

---

## 项目概览

| 项目 | 详情 |
|------|------|
| **仓库** | [https://github.com/alexk1540402997/Lottery](https://github.com/alexk1540402997/Lottery) |
| **远程** | `git@github.com:alexk1540402997/Lottery.git`（SSH） |
| **分支** | `master` |
| **彩票类型** | 双色球(SSQ) + 大乐透(DLT) |
| **数据文件** | `pythonProject/双色球.xlsx`（3464期）、`pythonProject/大乐透.xlsx` |

---

## 核心模块

| 文件 | 功能 | 关键版本 |
|------|------|----------|
| `predictor.py` | 13种分析方法引擎 | 4.2 — LightGBM + 5种新方法 |
| `merger.py` | 跨颗粒度加权投票合并器 | 4.0 |
| `backtester.py` | 回测引擎 + 智能搜索 | 4.2 — 3阶段混合搜索 |
| `config_manager.py` | 参数/权重版本管理 | 4.0 |
| `gui.py` | Tkinter GUI界面 | 4.2 — 13方法 + 可滚动面板 |
| `main.py` | 入口 | — |
| `build_exe.py` | 打包脚本 | — |

---

## 13种分析方法

| Key | 方法 | 类型 | 耗时(500期) | 状态 |
|-----|------|------|-------------|------|
| method_1 | 统计概率分析 | 频率统计 | 0.01s | ✅ 原始方法 |
| method_2 | 时间序列分析 | 趋势+和值 | 1.9s | ✅ iter优化至30 |
| method_3 | 模式识别分析 | 连号/区间/质合 | 0.15s | ✅ 原始方法 |
| method_4 | LightGBM | 梯度提升 | ~1.5s | ✅ trees=20 |
| method_5 | 马尔可夫分析 | 状态转移 | 0.05s | ✅ 原始方法 |
| method_6 | 蒙特卡罗模拟 | 模拟采样 | ~0.2s | ✅ sims=500 |
| method_7 | 聚类分析 | KMeans | 0.09s | ✅ 保留 |
| method_8 | N-gram分析 | 相似度匹配 | 0.04s | ✅ 原始方法 |
| method_9 | XGBoost | 集成学习 | 0.4s | ✅ 新增 |
| method_10 | 贝叶斯推断 | Beta-Binomial | 0.05s | ✅ 新增 |
| method_11 | 卡尔曼滤波 | 递推状态估计 | 0.01s | ✅ 新增 |
| method_12 | 泊松回归 | Poisson GLM | 1.0s | ✅ 新增 |
| method_13 | 共生矩阵分析 | 号码共现 | 0.01s | ✅ 新增 |

---

## 智能搜索策略（3阶段）

```
探索期（前30%时间）→  收敛期（后70%时间）
       ↓                     ↓
  加权随机采样           Top-5微调(85%)
  (历史表现好的            + 加权随机(15%)
   参数值概率更高)

  每20组合 → 随机脉冲（防局部最优）

  扰动率自适应:  默认20% → best≥3→8% → best≥4→5% → best≥5→2%
```

---

## 已完成优化清单

| # | 优化项 | 状态 |
|---|--------|:--:|
| 1 | 修复并发限制（`while < 2` → `while < max_concurrent`） | ✅ |
| 2 | 频率计算缓存 + numpy向量化 | ✅ |
| 3 | 智能颗粒度选择 | ✅ |
| 4 | 早停机制 | ✅ |
| 5 | LightGBM 替代 RF（trees 50→20） | ✅ |
| 6 | 新增 XGBoost / 贝叶斯 / 卡尔曼 / 泊松 / 共生矩阵 | ✅ |
| 7 | ML 特征构建向量化（6.87s→0.26s，26x） | ✅ |
| 8 | 蒙特卡罗 sims 3000→500 | ✅ |
| 9 | 时间序列 iter 300→30 | ✅ |
| 10 | GUI 左侧面板可滚动 + 方法2列布局 | ✅ |
| 11 | 预测结果显示所有颗粒度 | ✅ |
| 12 | 「★ 应用最优组合」按钮 | ✅ |
| 13 | 回测报告显示模型参数（Excel + GUI） | ✅ |
| 14 | 智能搜索：3阶段混合搜索 + 参数表现追踪 | ✅ |
| 15 | 搜索历史详情表（JSON + Excel「搜索历史」Sheet） | ✅ |

---

## 待完成/可选优化

| # | 项目 | 优先级 |
|---|------|:--:|
| 1 | ProcessPoolExecutor 替代 ThreadPoolExecutor（绕过GIL） | 中 |
| 2 | 更多 numpy 向量化（消除剩余 iterrows） | 低 |
| 3 | 打包 exe 测试 | 中 |
| 4 | 蒙特卡罗替换为隐马尔可夫模型？ | 低（待讨论） |
| 5 | 蒙特卡罗 numpy.int64 修复 | 低 |

---

## 运行时配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `test_periods` | 10 | 回测测试最新N期 |
| `max_search_time` | 0（不限） | GUI中按分钟设置 |
| `num_workers` | 4 | 并行线程数 |
| `max_train_periods` | 500 | 训练数据上限 |
| `granularities` | 智能选择 | 1期→[500], 2-10→[100,500], >10→[50,100,500] |
| `pulse_interval` | 20 | 随机脉冲间隔 |
| `phase_switch_ratio` | 0.30 | 探索期占比 |

---

## Git 提交历史（最近）

```
9e50d21 修正扰动率: 命中越高扰动越小
099715a 智能搜索策略: 3阶段混合搜索 + 历史追踪表
92cd597 修复: 预测显示所有颗粒度 + 优化Tab模型参数13方法
19f5115 参数优化: LightGBM/MonteCarlo/Timeseries 零牺牲提速
76e254d 修复: 预测Tab显示13方法 + 参数名同步
2a81e46 GUI: 左侧面板可滚动 + 方法复选框2列布局
860aade ML特征构建向量化 — 26x提速
8022ea4 阶段3: 智能颗粒度 + 应用最优组合按钮 + 早停
a5b18ca 阶段2: 方法升级 — 13种方法
b564cf6 回测报告增加模型参数值显示
```

---

## 快速启动指引

```bash
# 安装依赖
pip install lightgbm xgboost pandas numpy openpyxl scikit-learn

# 启动GUI
cd "C:/Users/AlexK/Desktop/彩票号码预测系统"
python main.py

# 直接回测
python backtester.py

# 打包exe
python build_exe.py
```

---

## 历史记录文件

| 文件 | 说明 |
|------|------|
| `logs/backtest_tried_combos.json` | 已尝试组合去重 |
| `logs/backtest_history.json` | 搜索历史详情（参数表现追踪） |
| `logs/current_model_params.json` | 当前模型参数 |
| `logs/current_merge_weights.json` | 当前合并权重 |
| `logs/version_index.json` | 版本索引 |
| `logs/versions/` | 历史版本快照 |
