"""
配置管理器 4.0
=============
管理模型参数和合并权重的版本日志，支持保存、回退、对比。

存储结构:
  logs/
    versions/
      model_params_v001_20260617_143052.json
      model_params_v002_20260617_150830.json
      merge_weights_v001_20260617_143052.json
      ...
    current_model_params.json    ← 当前使用的模型参数
    current_merge_weights.json   ← 当前使用的合并权重
    version_index.json           ← 版本索引
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from predictor import DEFAULT_PARAMS
from merger import DEFAULT_COMPOSITE_WEIGHTS, _convert_old_weights


class ConfigManager:
    """参数和权重配置管理器（按彩票类型隔离存储）"""

    def __init__(self, base_dir: str = "logs", lottery_type: str = "ssq"):
        """
        参数:
            base_dir: 日志和配置的基础目录
            lottery_type: 彩票类型 'ssq' 或 'dlt'（决定存储文件后缀）
        """
        self.base_dir = base_dir
        self.lottery_type = lottery_type.lower()

        # 根据彩票类型确定文件后缀（SSQ保持原文件名不变）
        suffix = '' if self.lottery_type == 'ssq' else f'_{self.lottery_type}'
        self.versions_dir = os.path.join(base_dir, f"versions{suffix}")
        self.current_params_file = os.path.join(base_dir, f"current_model_params{suffix}.json")
        self.current_weights_file = os.path.join(base_dir, f"current_merge_weights{suffix}.json")
        self.index_file = os.path.join(base_dir, f"version_index{suffix}.json")

        # 确保目录存在
        os.makedirs(self.versions_dir, exist_ok=True)

        # 当前参数和权重（从文件加载或创建默认）
        self.current_params = self._load_or_create_params()
        self.current_weights = self._load_or_create_weights()

        # 版本索引
        self.version_index = self._load_version_index()

        # ★ 记录本次启动时的初始状态作为"重置为默认"的基准
        self.baseline_params = json.loads(json.dumps(self.current_params))
        self.baseline_weights = json.loads(json.dumps(self.current_weights))

    def switch_lottery_type(self, lottery_type: str):
        """
        切换到另一种彩票类型的配置存储。

        保存当前配置后，加载目标类型的配置。
        SSQ和DLT使用完全独立的文件，互不影响。
        """
        new_type = lottery_type.lower()
        if new_type == self.lottery_type:
            return  # 相同类型，无需切换

        # 保存当前类型配置
        self._save_current_params(self.current_params)
        self._save_current_weights(self.current_weights)

        # 切换文件路径
        self.lottery_type = new_type
        suffix = '' if self.lottery_type == 'ssq' else f'_{self.lottery_type}'
        self.versions_dir = os.path.join(self.base_dir, f"versions{suffix}")
        self.current_params_file = os.path.join(self.base_dir, f"current_model_params{suffix}.json")
        self.current_weights_file = os.path.join(self.base_dir, f"current_merge_weights{suffix}.json")
        self.index_file = os.path.join(self.base_dir, f"version_index{suffix}.json")

        os.makedirs(self.versions_dir, exist_ok=True)

        # 重新加载目标类型配置
        self.current_params = self._load_or_create_params()
        self.current_weights = self._load_or_create_weights()
        self.version_index = self._load_version_index()

        # 更新基线
        self.baseline_params = json.loads(json.dumps(self.current_params))
        self.baseline_weights = json.loads(json.dumps(self.current_weights))

    def _load_or_create_params(self) -> Dict:
        """从文件加载当前参数，如果文件不存在或损坏则创建默认"""
        if os.path.exists(self.current_params_file):
            try:
                with open(self.current_params_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        # 文件不存在或损坏 → 从 DEFAULT_PARAMS 创建
        params = {}
        for key, value in DEFAULT_PARAMS.items():
            params[key] = dict(value)
        params['_meta'] = {
            'version': 0,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '默认参数配置',
            'lottery_type': 'ssq',
        }
        self._save_current_params(params)
        return params

    def _load_or_create_weights(self) -> Dict:
        """从文件加载当前权重，如果文件不存在或损坏则创建默认"""
        if os.path.exists(self.current_weights_file):
            try:
                with open(self.current_weights_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 自动转换旧格式
                if 'composite_weights' not in data:
                    if 'method_weights' in data or 'granularity_weights' in data:
                        data['composite_weights'] = _convert_old_weights(
                            data.get('method_weights'),
                            data.get('granularity_weights'))
                        data.pop('method_weights', None)
                        data.pop('granularity_weights', None)
                        data['_meta']['description'] = (data['_meta'].get('description', '')
                            + ' [自动转换为composite_weights]')
                        self._save_current_weights(data)
                return data
            except Exception:
                pass
        # 文件不存在或损坏 → 创建默认
        weights = {
            'composite_weights': dict(DEFAULT_COMPOSITE_WEIGHTS),
            '_meta': {
                'version': 0,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': '默认权重配置 (65个独立composite_weights)',
            }
        }
        self._save_current_weights(weights)
        return weights

    def _load_version_index(self) -> Dict:
        """加载版本索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'param_versions': [], 'weight_versions': []}

    def _save_version_index(self):
        """保存版本索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.version_index, f, ensure_ascii=False, indent=2)

    def _save_current_params(self, params: Dict):
        """保存当前参数到文件"""
        with open(self.current_params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def _save_current_weights(self, weights: Dict):
        """保存当前权重到文件"""
        with open(self.current_weights_file, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

    # ========================================================================
    #  保存版本
    # ========================================================================

    def save_params_version(self, params: Dict,
                            description: str = "",
                            lottery_type: str = "ssq",
                            backtest_score: float = 0.0
                            ) -> str:
        """
        保存模型参数版本。

        参数:
            params: 参数字典
            description: 版本描述
            lottery_type: 彩票类型
            backtest_score: 回测得分（用于排序）

        返回:
            版本文件名
        """
        version_num = len(self.version_index['param_versions']) + 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_params_v{version_num:03d}_{ts}.json"
        filepath = os.path.join(self.versions_dir, filename)

        # 添加元信息
        params_copy = dict(params)
        params_copy['_meta'] = {
            'version': version_num,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': description,
            'lottery_type': lottery_type,
            'backtest_score': backtest_score,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params_copy, f, ensure_ascii=False, indent=2)

        # 更新索引
        self.version_index['param_versions'].append({
            'version': version_num,
            'filename': filename,
            'created': params_copy['_meta']['created'],
            'description': description,
            'backtest_score': backtest_score,
        })
        self._save_version_index()

        # 更新当前参数
        self.current_params = params_copy
        self._save_current_params(params_copy)

        return filename

    def save_weights_version(self, composite_weights: Dict[str, float] = None,
                              method_weights: Dict[str, float] = None,
                              gran_weights: Dict[str, float] = None,
                              description: str = "",
                              backtest_score: float = 0.0
                              ) -> str:
        """
        保存合并权重版本。兼容旧格式自动转换。

        参数:
            composite_weights: 65个独立 composite_weights (新格式)
            method_weights: 方法权重 (旧格式，自动转换)
            gran_weights: 颗粒度权重 (旧格式，自动转换)
            description: 版本描述
            backtest_score: 回测得分

        返回:
            版本文件名
        """
        # 兼容旧格式自动转换
        if composite_weights is None and (method_weights or gran_weights):
            composite_weights = _convert_old_weights(method_weights, gran_weights)
        if composite_weights is None:
            composite_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)

        version_num = len(self.version_index['weight_versions']) + 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"merge_weights_v{version_num:03d}_{ts}.json"
        filepath = os.path.join(self.versions_dir, filename)

        weights = {
            'composite_weights': composite_weights,
            '_meta': {
                'version': version_num,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': description,
                'backtest_score': backtest_score,
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

        self.version_index['weight_versions'].append({
            'version': version_num,
            'filename': filename,
            'created': weights['_meta']['created'],
            'description': description,
            'backtest_score': backtest_score,
        })
        self._save_version_index()

        self.current_weights = weights
        self._save_current_weights(weights)

        return filename

    # ========================================================================
    #  版本回退
    # ========================================================================

    def list_param_versions(self) -> List[Dict]:
        """列出所有模型参数版本"""
        versions = self.version_index.get('param_versions', [])
        # 补充检查文件系统中是否有额外版本
        if os.path.exists(self.versions_dir):
            existing_files = set(os.listdir(self.versions_dir))
            for v in versions:
                if v['filename'] not in existing_files:
                    v['missing'] = True
        return sorted(versions, key=lambda x: x['version'], reverse=True)

    def list_weight_versions(self) -> List[Dict]:
        """列出所有权重版本"""
        versions = self.version_index.get('weight_versions', [])
        if os.path.exists(self.versions_dir):
            existing_files = set(os.listdir(self.versions_dir))
            for v in versions:
                if v['filename'] not in existing_files:
                    v['missing'] = True
        return sorted(versions, key=lambda x: x['version'], reverse=True)

    def rollback_params(self, version_num: int) -> Tuple[bool, str, Optional[Dict]]:
        """
        回退模型参数到指定版本。

        参数:
            version_num: 版本号

        返回:
            (成功, 消息, 加载后的参数)
        """
        for v in self.version_index['param_versions']:
            if v['version'] == version_num:
                filepath = os.path.join(self.versions_dir, v['filename'])
                if not os.path.exists(filepath):
                    return False, f"版本文件不存在: {v['filename']}", None
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                    self.current_params = params
                    self._save_current_params(params)
                    return True, f"已回退到模型参数版本 v{version_num}", params
                except Exception as e:
                    return False, f"加载失败: {e}", None

        return False, f"未找到版本 v{version_num}", None

    def rollback_weights(self, version_num: int) -> Tuple[bool, str, Optional[Dict]]:
        """
        回退合并权重到指定版本。

        参数:
            version_num: 版本号

        返回:
            (成功, 消息, 加载后的权重)
        """
        for v in self.version_index['weight_versions']:
            if v['version'] == version_num:
                filepath = os.path.join(self.versions_dir, v['filename'])
                if not os.path.exists(filepath):
                    return False, f"版本文件不存在: {v['filename']}", None
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        weights = json.load(f)
                    self.current_weights = weights
                    self._save_current_weights(weights)
                    return True, f"已回退到权重版本 v{version_num}", weights
                except Exception as e:
                    return False, f"加载失败: {e}", None

        return False, f"未找到版本 v{version_num}", None

    # ========================================================================
    #  比较和查询
    # ========================================================================

    def compare_params_versions(self, v1: int, v2: int) -> Dict:
        """比较两个参数版本的差异"""
        p1 = self._load_params_version(v1)
        p2 = self._load_params_version(v2)

        if p1 is None or p2 is None:
            return {'error': '版本加载失败'}

        diff = {}
        for method_key in set(list(p1.keys()) + list(p2.keys())):
            if method_key.startswith('_'):
                continue
            m1 = p1.get(method_key, {})
            m2 = p2.get(method_key, {})
            method_diff = {}
            for param_key in set(list(m1.keys()) + list(m2.keys())):
                if param_key.startswith('_'):
                    continue
                val1 = m1.get(param_key)
                val2 = m2.get(param_key)
                if val1 != val2:
                    method_diff[param_key] = {'v1': val1, 'v2': val2}
            if method_diff:
                diff[method_key] = method_diff

        return diff

    def _load_params_version(self, version_num: int) -> Optional[Dict]:
        """加载指定版本的参数"""
        for v in self.version_index['param_versions']:
            if v['version'] == version_num:
                filepath = os.path.join(self.versions_dir, v['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
        return None

    def _load_weights_version(self, version_num: int) -> Optional[Dict]:
        """加载指定版本的权重"""
        for v in self.version_index['weight_versions']:
            if v['version'] == version_num:
                filepath = os.path.join(self.versions_dir, v['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
        return None

    def get_current_config(self) -> Dict[str, Any]:
        """获取当前完整配置"""
        return {
            'params': self.current_params,
            'weights': self.current_weights,
            'param_versions_count': len(self.version_index['param_versions']),
            'weight_versions_count': len(self.version_index['weight_versions']),
        }

    # ========================================================================
    #  重置
    # ========================================================================

    def reset_to_defaults(self) -> Tuple[bool, str]:
        """
        重置为本次程序启动时的初始配置（baseline）。

        不删除任何历史版本，只是将当前配置回退到启动时的状态。
        后续通过 save_params_version / save_weights_version 创建的新版本
        会按原有顺序号继续往后叠加。
        """
        try:
            baseline_p = self.baseline_params
            baseline_w = self.baseline_weights
            pv = baseline_p.get('_meta', {}).get('version', 0)
            wv = baseline_w.get('_meta', {}).get('version', 0)

            self.current_params = json.loads(json.dumps(baseline_p))
            self.current_weights = json.loads(json.dumps(baseline_w))
            self._save_current_params(self.current_params)
            self._save_current_weights(self.current_weights)
            return True, f"已重置为启动时配置 (参数v{pv}, 权重v{wv})"
        except Exception as e:
            return False, f"重置失败: {e}"

    def export_all_versions(self, output_dir: str = "config_export") -> str:
        """导出所有版本配置到一个目录（用于备份）"""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(output_dir, f"config_backup_{ts}")
        os.makedirs(export_dir, exist_ok=True)

        # 复制当前配置
        shutil.copy2(self.current_params_file,
                     os.path.join(export_dir, "current_model_params.json"))
        shutil.copy2(self.current_weights_file,
                     os.path.join(export_dir, "current_merge_weights.json"))

        # 复制版本目录
        if os.path.exists(self.versions_dir):
            export_versions = os.path.join(export_dir, "versions")
            shutil.copytree(self.versions_dir, export_versions)

        # 复制索引
        shutil.copy2(self.index_file,
                     os.path.join(export_dir, "version_index.json"))

        return export_dir


# ============================================================================
#  测试
# ============================================================================

if __name__ == "__main__":
    print("配置管理器 4.0 测试")

    cm = ConfigManager()

    # 保存测试版本
    test_params = DEFAULT_PARAMS.copy()
    test_params['statistical']['freq_weight'] = 0.75
    fname = cm.save_params_version(
        test_params,
        description="测试：提高统计概率分析的频率权重",
        backtest_score=1.85)
    print(f"已保存参数版本: {fname}")

    # 列出版本
    versions = cm.list_param_versions()
    print(f"参数版本数: {len(versions)}")
    for v in versions:
        print(f"  v{v['version']}: {v['description']} ({v.get('backtest_score', 'N/A')})")

    # 回退到默认
    if len(versions) >= 2:
        ok, msg, _ = cm.rollback_params(versions[-1]['version'])
        print(f"回退结果: {msg}")

    print("测试完成")
