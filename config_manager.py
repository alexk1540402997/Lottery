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
from merger import DEFAULT_WEIGHT_MATRIX


class ConfigManager:
    """参数和权重配置管理器"""

    def __init__(self, base_dir: str = "logs"):
        """
        参数:
            base_dir: 日志和配置的基础目录
        """
        self.base_dir = base_dir
        self.versions_dir = os.path.join(base_dir, "versions")
        self.current_params_file = os.path.join(base_dir, "current_model_params.json")
        self.current_weights_file = os.path.join(base_dir, "current_merge_weights.json")
        self.index_file = os.path.join(base_dir, "version_index.json")

        # 确保目录存在
        os.makedirs(self.versions_dir, exist_ok=True)

        # 当前参数和权重
        self.current_params = self._init_default_params()
        self.current_weights = self._init_default_weights()

        # 版本索引
        self.version_index = self._load_version_index()

    def _init_default_params(self) -> Dict:
        """初始化默认参数"""
        if os.path.exists(self.current_params_file):
            try:
                with open(self.current_params_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        # 复制默认参数
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

    def _init_default_weights(self) -> Dict:
        """初始化默认权重"""
        if os.path.exists(self.current_weights_file):
            try:
                with open(self.current_weights_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        weights = {
            'method_weights': dict(DEFAULT_WEIGHT_MATRIX['method_base_weights']),
            'granularity_weights': dict(DEFAULT_WEIGHT_MATRIX['granularity_base_weights']),
            '_meta': {
                'version': 0,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': '默认权重配置',
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

    def save_weights_version(self, method_weights: Dict[str, float],
                              gran_weights: Dict[str, float],
                              description: str = "",
                              backtest_score: float = 0.0
                              ) -> str:
        """
        保存合并权重版本。

        参数:
            method_weights: 方法权重 {method_key: weight}
            gran_weights: 颗粒度权重 {gran_name: weight}
            description: 版本描述
            backtest_score: 回测得分

        返回:
            版本文件名
        """
        version_num = len(self.version_index['weight_versions']) + 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"merge_weights_v{version_num:03d}_{ts}.json"
        filepath = os.path.join(self.versions_dir, filename)

        weights = {
            'method_weights': method_weights,
            'granularity_weights': gran_weights,
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
        """重置为默认参数和权重"""
        try:
            self.current_params = self._init_default_params()
            self.current_weights = self._init_default_weights()
            self._save_current_params(self.current_params)
            self._save_current_weights(self.current_weights)
            return True, "已重置为默认配置"
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
