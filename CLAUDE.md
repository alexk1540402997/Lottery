# CLAUDE.md — 彩票号码预测系统

## 会话连续性

**每次启动必须先读取** `SESSION_LOG.md`，了解项目最新状态、已完成工作和待办事项。

## 语言

永远使用中文回复用户。所有对话、解释、代码注释、Git提交信息均使用中文。

## 项目架构

```
predictor.py     → 13种分析方法引擎（核心预测）
merger.py        → 跨颗粒度加权投票合并
backtester.py    → 回测引擎 + 智能搜索策略
config_manager.py → 参数/权重版本管理
gui.py           → Tkinter GUI
main.py          → 入口
build_exe.py     → 打包脚本
```

## 关键路径

- 数据文件：`pythonProject/双色球.xlsx`（3464期）、`pythonProject/大乐透.xlsx`
- 日志目录：`logs/`（包含历史记录、当前配置、版本快照）
- 回测报告：`backtest_reports/`
- 合并结果：`merged_results/`

## 依赖

```
pip install lightgbm xgboost pandas numpy openpyxl scikit-learn
```

## Git

- 远程：`git@github.com:alexk1540402997/Lottery.git`（SSH）
- 分支：`master`
