"""
CST RAG 功能测试脚本
"""
import sys
import os
from config import RAGConfig
from cst.cst_rag import CSTOptimizationRAG


def test_initialization():
    """测试1: 系统初始化"""
    print("\n" + "=" * 70)
    print("测试1: CST RAG 系统初始化")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)
        optimizer.initialize()

        print("\n✅ 系统初始化成功")
        print(f"   - 快速模式: True")
        print(f"   - 长期记忆集合: long_memory")
        return True
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_operations():
    """测试2: 记忆操作"""
    print("\n" + "=" * 70)
    print("测试2: 记忆操作测试")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        # 保存测试数据
        print("\n保存测试仿真记录...")
        optimizer.memory_manager.save_simulation_result(
            simulation_id="test_cst_001",
            parameters={'W': 15.5, 'H': 8.0, 'L': 20.0},
            results={'s11_min': -12.5, 'vswr_max': 1.8, 'gain': 7.5},
            summary="测试仿真：S11最低-12.5dB未达标，VSWR最大1.8，增益7.5dBi",
            tags=["测试", "天线设计"]
        )

        optimizer.memory_manager.save_simulation_result(
            simulation_id="test_cst_002",
            parameters={'W': 16.0, 'H': 8.5, 'L': 21.0},
            results={'s11_min': -16.2, 'vswr_max': 1.4, 'gain': 8.3},
            summary="优化后仿真：S11最低-16.2dB达标，VSWR最大1.4，增益8.3dBi",
            tags=["测试", "优化", "成功案例"]
        )

        print("✅ 保存成功")

        # 列出记录
        print("\n列出所有仿真记录...")
        all_sims = optimizer.memory_manager.long_term.list_simulations(limit=10)
        print(f"✅ 共有 {len(all_sims)} 条记录")

        # 搜索相似仿真
        print("\n搜索相似仿真（查询：S11优化）...")
        similar = optimizer.memory_manager.search_similar_simulations(
            query="S11优化 天线",
            top_k=3
        )
        print(f"✅ 找到 {len(similar)} 个相似仿真")

        for i, sim in enumerate(similar, 1):
            print(f"\n   {i}. ID: {sim['simulation_id']}")
            print(f"      总结: {sim['summary']}")
            print(f"      相似度: {sim['relevance_score']:.2%}")

        # 清理测试数据
        print("\n清理测试数据...")
        optimizer.memory_manager.long_term.delete_simulation("test_cst_001")
        optimizer.memory_manager.long_term.delete_simulation("test_cst_002")
        print("✅ 清理完成")

        return True
    except Exception as e:
        print(f"\n❌ 记忆操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_history_context_building():
    """测试3: 历史上下文构建"""
    print("\n" + "=" * 70)
    print("测试3: 历史上下文构建")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        # 先保存一些测试数据
        optimizer.memory_manager.save_simulation_result(
            simulation_id="test_ctx_001",
            parameters={'W': 15.0, 'H': 7.5},
            results={'s11_min': -11.0},
            summary="S11最低-11.0dB未达标",
            tags=["测试"]
        )

        # 搜索并构建上下文
        print("\n搜索并构建历史上下文...")
        similar = optimizer.memory_manager.search_similar_simulations(
            query="S11 天线",
            top_k=2
        )

        context = optimizer._build_history_context(similar)
        print(f"\n构建的上下文:\n{context}")

        # 清理
        optimizer.memory_manager.long_term.delete_simulation("test_ctx_001")

        print("\n✅ 历史上下文构建成功")
        return True
    except Exception as e:
        print(f"\n❌ 历史上下文构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_prompt_building():
    """测试4: 优化提示词构建"""
    print("\n" + "=" * 70)
    print("测试4: 优化提示词构建")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        current_summary = "仿真结果：S11最低-12.5dB未达标，VSWR最大1.8需优化"
        current_params = {'W': 15.5, 'H': 8.0}
        optimization_goal = "S11低于-15dB，VSWR低于1.5"
        history_context = "历史相似仿真记录：\n记录1: S11最低-16.2dB达标"

        print("\n构建优化提示词...")
        prompt = optimizer._build_optimization_prompt(
            current_summary=current_summary,
            current_params=current_params,
            optimization_goal=optimization_goal,
            history_context=history_context
        )

        print(f"\n提示词长度: {len(prompt)} 字符")
        print(f"\n提示词预览:\n{prompt[:300]}...")

        print("\n✅ 优化提示词构建成功")
        return True
    except Exception as e:
        print(f"\n❌ 优化提示词构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parse_suggestion():
    """测试5: 解析优化建议"""
    print("\n" + "=" * 70)
    print("测试5: 解析优化建议")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        # 测试JSON格式的响应
        json_response = """
        根据分析，建议调整以下参数：        """

        print("\n解析JSON格式响应...")
        suggestion = optimizer._parse_optimization_suggestion(json_response)

        print(f"\n解析结果:")
        print(f"  参数调整: {suggestion['parameter_changes']}")
        print(f"  推理: {suggestion['reasoning']}")
        print(f"  预期改善: {suggestion['expected_improvement']}")

        # 测试纯文本响应
        text_response = """
        建议将W调整为16.5，H设置为8.5，这样可以改善匹配。
        """

        print("\n解析纯文本响应...")
        suggestion2 = optimizer._parse_optimization_suggestion(text_response)
        print(f"  参数调整: {suggestion2['parameter_changes']}")

        print("\n✅ 解析优化建议成功")
        return True
    except Exception as e:
        print(f"\n❌ 解析优化建议失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_cst_interface():
    """测试6: 创建CST接口"""
    print("\n" + "=" * 70)
    print("测试6: 创建CST接口")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        print("\n创建CST接口实例...")
        cst = optimizer.create_cst_interface()

        print(f"✅ CST接口创建成功")
        print(f"   - 项目路径: {cst.project_path or '未设置'}")
        print(f"   - 结果路径: {cst.results_path or '未设置'}")

        return True
    except Exception as e:
        print(f"\n❌ CST接口创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_analysis_workflow():
    """测试7: 完整分析流程（不含LLM调用）"""
    print("\n" + "=" * 70)
    print("测试7: 完整分析流程测试")
    print("=" * 70)

    try:
        config = RAGConfig()
        optimizer = CSTOptimizationRAG(config, use_fast_mode=True)

        # 准备测试数据
        current_summary = "仿真结果：S11最低-12.5dB未达标，VSWR最大1.8需优化，增益7.5dBi"
        current_params = {'W': 15.5, 'H': 8.0, 'L': 20.0}
        optimization_goal = "S11低于-15dB，VSWR低于1.5，增益高于8dBi"

        print(f"\n当前状态:")
        print(f"  参数: {current_params}")
        print(f"  结果: {current_summary}")
        print(f"  目标: {optimization_goal}")

        # 先保存一些历史数据供搜索
        print("\n保存历史仿真数据...")
        optimizer.memory_manager.save_simulation_result(
            simulation_id="test_hist_001",
            parameters={'W': 16.0, 'H': 8.5, 'L': 21.0},
            results={'s11_min': -16.2, 'vswr_max': 1.4, 'gain': 8.3},
            summary="成功优化案例：S11-16.2dB，VSWR1.4，增益8.3dBi",
            tags=["历史", "成功案例"]
        )

        # 执行分析（跳过LLM调用部分）
        print("\n执行搜索和分析...")
        search_query = f"{optimization_goal} {' '.join([f'{k}={v}' for k, v in current_params.items()])}"
        similar = optimizer.memory_manager.search_similar_simulations(
            query=search_query,
            top_k=3
        )

        print(f"✅ 找到 {len(similar)} 个相似仿真")

        history_context = optimizer._build_history_context(similar)
        print(f"✅ 构建历史上下文成功")

        prompt = optimizer._build_optimization_prompt(
            current_summary=current_summary,
            current_params=current_params,
            optimization_goal=optimization_goal,
            history_context=history_context
        )
        print(f"✅ 构建优化提示词成功 ({len(prompt)} 字符)")

        # 清理
        optimizer.memory_manager.long_term.delete_simulation("test_hist_001")

        print("\n✅ 完整分析流程测试成功")
        return True
    except Exception as e:
        print(f"\n❌ 完整分析流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("CST RAG 功能测试套件")
    print("=" * 70)

    tests = [
        ("系统初始化", test_initialization),
        ("记忆操作", test_memory_operations),
        ("历史上下文构建", test_history_context_building),
        ("优化提示词构建", test_optimization_prompt_building),
        ("解析优化建议", test_parse_suggestion),
        ("创建CST接口", test_create_cst_interface),
        ("完整分析流程", test_full_analysis_workflow),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 打印测试结果汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！CST RAG 系统功能正常")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查错误信息")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


