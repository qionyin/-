

"""
CST仿真优化RAG系统 - 基于长期记忆和LLM的自动优化系统
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from config import RAGConfig
from llm_client import QwenLLMClient
from .memory_manager import MemoryManager
from cst.cst_tools import CSTInterface



class CSTOptimizationRAG:
    """CST仿真优化RAG系统"""

    def __init__(self, config: RAGConfig = None, use_fast_mode: bool = True):
        """
        初始化CST优化RAG系统

        Args:
            config: RAG配置
            use_fast_mode: 是否使用快速模式（不使用向量嵌入）
        """
        self.config = config or RAGConfig()
        self.llm_client = QwenLLMClient(self.config)

        # 使用相同的长期记忆数据库
        self.memory_manager = MemoryManager(
            short_term_max_turns=10,
            long_term_persist_dir=self.config.chroma_persist_dir,
            long_term_collection_name="long_memory",
            use_embedding=not use_fast_mode
        )

        self.is_initialized = False
        self.optimization_history = []

# ... existing code ...


    def initialize(self):
        """初始化系统"""
        print("=" * 60)
        print("CST仿真优化RAG系统初始化中...")
        print("=" * 60)

        # 显示CST配置信息
        if self.config.cst_project_path:
            print(f"CST项目路径: {self.config.cst_project_path}")
        else:
            print("⚠️ 未配置CST项目路径，需要在运行时指定")

        if self.config.cst_results_path:
            print(f"CST结果路径: {self.config.cst_results_path}")
        else:
            print("⚠️ 未配置CST结果路径，将使用默认路径")

        # 初始化LLM
        try:
            self.llm_client._init_llm()
        except Exception as e:
            print(f"警告: LLM初始化失败: {e}")

        self.is_initialized = True
        print("CST仿真优化RAG系统初始化完成！")
        print("=" * 60)

    def create_cst_interface(self) -> CSTInterface:
        """
        创建CST接口实例（使用配置中的路径）

        Returns:
            CSTInterface实例
        """
        cst = CSTInterface(
            project_path=self.config.cst_project_path,
            results_path=self.config.cst_results_path
        )

        # 如果配置了项目路径，自动加载
        if self.config.cst_project_path:
            cst.load_project()

        return cst

    def analyze_current_results(self,
                               current_summary: str,
                               current_params: Dict,
                               optimization_goal: str) -> Dict:
        """
        分析当前仿真结果并结合长期记忆给出优化建议

        Args:
            current_summary: 当前仿真结果总结
            current_params: 当前仿真参数
            optimization_goal: 优化目标描述

        Returns:
            优化建议字典
        """
        if not self.is_initialized:
            self.initialize()

        print("\n" + "=" * 60)
        print("步骤1: 搜索相似的历史仿真记录")
        print("=" * 60)

        # 1. 从长期记忆中搜索相似的仿真
        search_query = f"{optimization_goal} {' '.join([f'{k}={v}' for k, v in current_params.items()])}"
        similar_simulations = self.memory_manager.search_similar_simulations(
            query=search_query,
            top_k=5
        )

        print(f"找到 {len(similar_simulations)} 个相似仿真记录")

        # 2. 构建历史经验上下文
        history_context = self._build_history_context(similar_simulations)

        print("\n" + "=" * 60)
        print("步骤2: 请求LLM生成优化建议")
        print("=" * 60)

        # 3. 调用LLM生成优化建议
        optimization_prompt = self._build_optimization_prompt(
            current_summary=current_summary,
            current_params=current_params,
            optimization_goal=optimization_goal,
            history_context=history_context
        )

        llm_response = self.llm_client.chat(
            user_message=optimization_prompt,
            context=""
        )

        # 4. 解析LLM返回的优化建议
        optimization_suggestion = self._parse_optimization_suggestion(llm_response)

        result = {
            'current_summary': current_summary,
            'current_params': current_params,
            'optimization_goal': optimization_goal,
            'similar_simulations': similar_simulations,
            'llm_suggestion': llm_response,
            'parsed_suggestion': optimization_suggestion
        }

        print("\n" + "=" * 60)
        print("优化建议生成完成")
        print("=" * 60)

        return result

    def _build_history_context(self, similar_simulations: List[Dict]) -> str:
        """构建历史经验上下文"""
        if not similar_simulations:
            return "暂无历史相似仿真记录。"

        context_lines = ["历史相似仿真记录：\n"]

        for i, sim in enumerate(similar_simulations, 1):
            context_lines.append(f"记录{i}:")
            context_lines.append(f"  仿真ID: {sim['simulation_id']}")
            context_lines.append(f"  参数: {json.dumps(sim['parameters'], ensure_ascii=False)}")
            context_lines.append(f"  结果总结: {sim['summary']}")
            context_lines.append(f"  相似度: {sim['relevance_score']:.2%}")
            context_lines.append("")

        return "\n".join(context_lines)

    def _build_optimization_prompt(self,
                                  current_summary: str,
                                  current_params: Dict,
                                  optimization_goal: str,
                                  history_context: str) -> str:
        """构建优化建议提示词"""

        prompt = f"""你是一个CST仿真优化专家。请根据当前仿真结果、优化目标和历史经验，提供具体的参数优化建议。

【当前仿真参数】
{json.dumps(current_params, ensure_ascii=False, indent=2)}

【当前仿真结果】
{current_summary}

【优化目标】
{optimization_goal}

【历史相似仿真记录】
{history_context}

【任务要求】
请分析以上信息，并提供：

1. **问题分析**：当前仿真结果存在什么问题，距离优化目标还有哪些差距

2. **优化策略**：基于历史经验和专业知识，应该调整哪些参数，为什么

3. **具体建议**：给出明确的参数调整建议，格式为JSON：

4. **风险评估**：可能的风险和注意事项

请确保给出的参数值是具体的数值，便于直接应用于CST仿真。"""

        return prompt

    def _parse_optimization_suggestion(self, llm_response: str) -> Dict:
        """解析LLM返回的优化建议"""
        import re

        suggestion = {
            'parameter_changes': {},
            'reasoning': '',
            'expected_improvement': '',
            'raw_response': llm_response
        }

        # 尝试提取JSON格式的
        json_pattern = r'json\s*(.?)\s'


        match = re.search(json_pattern, llm_response, re.DOTALL)

        if match:
            try:
                json_data = json.loads(match.group(1))
                suggestion['parameter_changes'] = json_data.get('parameter_changes', {})
                suggestion['reasoning'] = json_data.get('reasoning', '')
                suggestion['expected_improvement'] = json_data.get('expected_improvement', '')
            except json.JSONDecodeError:
                print("警告: JSON解析失败，使用默认解析方式")
                suggestion['parameter_changes'] = self._fallback_parse(llm_response)
        else:
            # 降级解析
            suggestion['parameter_changes'] = self._fallback_parse(llm_response)

        return suggestion


    def _fallback_parse(self, response: str) -> Dict:
        """降级解析方式"""
        import re

        param_changes = {}

        # 尝试匹配 "参数名: 值" 或 "参数名 = 值" 的模式
        patterns = [
            r'(\w+)\s*[=:]\s*([\d.]+)',
            r'将\s*(\w+)\s*(?:调整为|改为|设置为)\s*([\d.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            for name, value in matches:
                try:
                    param_changes[name] = float(value)
                except ValueError:
                    continue

        return param_changes

    def apply_optimization(self,
                          cst_interface: CSTInterface,
                          parameter_changes: Dict,
                          verify: bool = True) -> bool:
        """
        应用优化建议到CST模型

        Args:
            cst_interface: CST接口实例
            parameter_changes: 参数修改字典
            verify: 是否验证修改

        Returns:
            是否成功应用
        """
        print("\n" + "=" * 60)
        print("应用优化建议")
        print("=" * 60)

        if not parameter_changes:
            print("❌ 没有参数需要修改")
            return False

        print("\n即将修改的参数:")
        for param_name, new_value in parameter_changes.items():
            old_value = cst_interface.parameters.get(param_name, 'N/A')
            print(f"  {param_name}: {old_value} → {new_value}")

        # 调用CST接口修改参数
        success = cst_interface.modify_parameters(parameter_changes)

        if success and verify:
            print("\n✅ 参数修改成功并已验证")
        elif success:
            print("\n✅ 参数修改成功")
        else:
            print("\n❌ 参数修改失败")

        return success

    def run_optimization_cycle(self,
                              cst_interface: CSTInterface = None,
                              results_path: str = None,
                              optimization_goal: str = "",
                              max_iterations: int = 3,
                              tags: Optional[List[str]] = None) -> List[Dict]:
        """
        运行完整的优化循环（读取结果 → 分析 → 优化 → 仿真 → 自动保存）

        Args:
            cst_interface: CST接口实例，如果为None则使用配置创建
            results_path: 结果文件路径，如果为None则使用配置中的路径
            optimization_goal: 优化目标描述
            max_iterations: 最大迭代次数
            tags: 标签列表

        Returns:
            优化历史记录
        """
        if tags is None:
            tags = ["CST优化"]

        # 如果没有提供CST接口，使用配置创建
        if cst_interface is None:
            cst_interface = self.create_cst_interface()

        # 如果没有提供结果路径，使用配置中的路径
        if results_path is None:
            results_path = self.config.cst_results_path

        print("\n" + "=" * 70)
        print("开始CST自动优化循环")
        print(f"优化目标: {optimization_goal}")
        print(f"最大迭代次数: {max_iterations}")
        print("=" * 70)

        optimization_history = []

        for iteration in range(1, max_iterations + 1):
            print("\n" + "=" * 70)
            print(f"第 {iteration}/{max_iterations} 次优化迭代")
            print("=" * 70)

            # 1. 读取当前仿真结果
            print("\n【阶段1】读取当前仿真结果...")
            current_summary = cst_interface.read_results_and_summarize(results_path)
            current_params = cst_interface.parameters.copy()

            # 2. 分析并生成优化建议
            print("\n【阶段2】分析结果并生成优化建议...")
            analysis_result = self.analyze_current_results(
                current_summary=current_summary,
                current_params=current_params,
                optimization_goal=optimization_goal
            )

            parameter_changes = analysis_result['parsed_suggestion']['parameter_changes']

            if not parameter_changes:
                print("\n⚠️ 未获取到有效的参数优化建议，停止优化")
                break

            # 3. 应用优化建议
            print("\n【阶段3】应用优化建议...")
            success = self.apply_optimization(
                cst_interface=cst_interface,
                parameter_changes=parameter_changes
            )

            if not success:
                print("\n❌ 参数应用失败，停止优化")
                break

            # 4. 记录本次优化
            iteration_record = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'before_params': current_params,
                'after_params': cst_interface.parameters.copy(),
                'parameter_changes': parameter_changes,
                'summary_before': current_summary,
                'analysis': analysis_result
            }

            optimization_history.append(iteration_record)

            print(f"\n✅ 第 {iteration} 次优化完成")
            print(f"   修改参数: {list(parameter_changes.keys())}")

            # 询问是否继续下一次迭代
            if iteration < max_iterations:
                user_input = input("\n是否继续下一次优化？(y/n，默认y): ").strip().lower()
                if user_input == 'n':
                    print("用户选择停止优化")
                    break

        # 5. 最终仿真并自动保存到长期记忆
        print("\n" + "=" * 70)
        print("执行最终仿真并保存到长期记忆")
        print("=" * 70)

        final_summary = cst_interface.read_results_and_summarize(results_path)

        # 自动生成仿真ID并保存
        simulation_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.memory_manager.save_simulation_result(
            simulation_id=simulation_id,
            parameters=cst_interface.parameters.copy(),
            results=cst_interface.results,
            summary=f"[优化后] {final_summary}",
            tags=tags + ["优化结果", f"迭代{len(optimization_history)}次"]
        )

        # 保存优化历史记录
        optimization_record = {
            'simulation_id': simulation_id,
            'optimization_goal': optimization_goal,
            'iterations': len(optimization_history),
            'optimization_history': optimization_history,
            'final_params': cst_interface.parameters.copy(),
            'final_summary': final_summary,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 70)
        print("优化流程完成")
        print("=" * 70)
        print(f"最终仿真ID: {simulation_id}")
        print(f"总迭代次数: {len(optimization_history)}")
        print(f"最终结果: {final_summary}")

        return optimization_history

    def get_optimization_report(self, optimization_history: List[Dict]) -> str:
        """生成优化报告"""
        report_lines = [
            "\n" + "=" * 70,
            "CST优化报告",
            "=" * 70,
            f"优化迭代次数: {len(optimization_history)}",
            ""
        ]

        for i, record in enumerate(optimization_history, 1):
            report_lines.append(f"第 {i} 次迭代:")
            report_lines.append(f"  修改参数: {record['parameter_changes']}")
            report_lines.append(f"  优化前结果: {record['summary_before']}")
            report_lines.append("")

        return "\n".join(report_lines)
