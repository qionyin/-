"""
CST 2025 接口工具
提供CST仿真软件的参数修改和结果读取功能
"""
import os
import re
from typing import Dict, Optional
from pathlib import Path


class CSTInterface:
    """CST 2025 接口类"""

    def __init__(self, project_path: str = None, results_path: str = None):
        """初始化CST接口

        Args:
            project_path: CST项目文件路径 (.cst 文件)
            results_path: CST仿真结果文件夹路径
        """
        self.project_path = project_path
        self.results_path = results_path
        self.parameters = {}
        self.results = {}

    def load_project(self, project_path: str = None):
        """加载CST项目文件

        Args:
            project_path: CST项目文件路径，如果为None则使用初始化时设置的路径
        """
        path = project_path or self.project_path

        if not path:
            raise ValueError("未提供CST项目文件路径")

        if not os.path.exists(path):
            raise FileNotFoundError(f"项目文件不存在: {path}")

        self.project_path = path
        print(f"✅ 已加载CST项目: {path}")

        # 自动读取当前参数
        self._read_current_parameters()

# ... existing code ...

    def read_results_and_summarize(self, results_path: str = None) -> str:
        """读取CST仿真结果并总结为一句话

        Args:
            results_path: 结果文件路径（默认为配置的路径或桌面文件1中的result文件夹）

        Returns:
            一句话总结，便于传给后续agent
        """
        # 优先使用传入的路径，其次使用实例变量，最后使用默认路径
        if results_path is None:
            results_path = self.results_path

        if results_path is None:
            desktop_path = Path.home() / "Desktop" / "文件1" / "result"
            results_path = str(desktop_path)

        print("\n" + "=" * 60)
        print(f"读取CST仿真结果")
        print(f"路径: {results_path}")
        print("=" * 60)

        if not os.path.exists(results_path):
            print(f"❌ 结果路径不存在: {results_path}")
            return "错误：结果路径不存在"

        results = {
            's_parameters': {},
            'vswr': {},
            'gain': {},
            'efficiency': {},
            'files_read': []
        }

        for filename in os.listdir(results_path):
            filepath = os.path.join(results_path, filename)

            if not os.path.isfile(filepath):
                continue

            try:
                if filename.endswith(('.txt', '.csv', '.dat')):
                    data = self._parse_result_file(filepath)

                    if 's11' in filename.lower() or 's-parameter' in filename.lower():
                        results['s_parameters'] = data
                        results['files_read'].append(filename)
                        print(f"  📄 读取S参数: {filename}")

                    elif 'vswr' in filename.lower():
                        results['vswr'] = data
                        results['files_read'].append(filename)
                        print(f"  📄 读取VSWR: {filename}")

                    elif 'gain' in filename.lower():
                        results['gain'] = data
                        results['files_read'].append(filename)
                        print(f"  📄 读取增益: {filename}")

                    elif 'efficiency' in filename.lower() or 'eff' in filename.lower():
                        results['efficiency'] = data
                        results['files_read'].append(filename)
                        print(f"  📄 读取效率: {filename}")

            except Exception as e:
                print(f"  ⚠️ 读取文件 {filename} 失败: {e}")

        self.results = results
        print(f"\n✅ 共读取 {len(results['files_read'])} 个结果文件")

        # 生成一句话总结
        summary = self._generate_one_sentence_summary(results)

        print(f"\n📝 一句话总结:")
        print(f"   {summary}")

        return summary

# ... existing code ...
