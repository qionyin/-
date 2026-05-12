"""
RAG系统使用示例
"""
import os
from dotenv import load_dotenv
from config import RAGConfig
from rag_system import RAGSystem
from cst.cst_rag import CSTOptimizationRAG
from cst.cst_tools import CSTInterface


def main():
    """主函数"""

    # 加载环境变量
    load_dotenv()

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请在.env文件中设置或在代码中直接配置")

    # 创建配置
    config = RAGConfig(
        dashscope_api_key=api_key or "",
        qwen_model_name="qwen3-max",
        chroma_persist_dir="./chroma_db",
        collection_name="demo_collection",
        top_k=3,
        chunk_size=500,
        chunk_overlap=50,
        cst_project_path=os.getenv("CST_PROJECT_PATH", ""),
        cst_results_path=os.getenv("CST_RESULTS_PATH", "")
    )

    print("\n请选择要使用的系统:")
    print("1. 通用RAG问答系统")
    print("2. CST仿真自动优化系统")

    choice = input("\n请输入选择 (1/2): ").strip()

    if choice == "2":
        run_cst_optimization(config)
    else:
        run_general_rag(config)


def run_general_rag(config: RAGConfig):
    """运行通用RAG系统"""
    # 创建RAG系统（使用快速模式）
    rag = RAGSystem(config, use_fast_mode=True)

    # 初始化系统
    rag.initialize()

    # 交互式对话模式
    print("\n" + "=" * 60)
    print("进入交互对话模式（输入'quit'退出）")
    print("=" * 60)

    while True:
        user_input = input("\n您的问题: ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("感谢使用，再见！")
            break

        if not user_input:
            continue

        try:
            result = rag.chat(user_input, top_k=3)
        except Exception as e:
            print(f"查询出错: {e}")


def run_cst_optimization(config: RAGConfig):
    """运行CST优化系统"""
    print("\n" + "=" * 70)
    print("CST仿真自动优化系统")
    print("=" * 70)

    # 显示配置信息
    if config.cst_project_path:
        print(f"CST项目路径: {config.cst_project_path}")
    else:
        print("⚠️ 未配置CST项目路径")

    if config.cst_results_path:
        print(f"CST结果路径: {config.cst_results_path}")
    else:
        print("⚠️ 未配置CST结果路径，将使用默认路径")

    # 初始化优化器（使用快速模式）
    optimizer = CSTOptimizationRAG(config, use_fast_mode=True)
    optimizer.initialize()

    # 创建CST接口（使用配置中的路径）
    cst = optimizer.create_cst_interface()

    # 如果配置中没有项目路径，询问用户
    if not config.cst_project_path:
        project_path = input("\n请输入CST项目文件路径 (.cst): ").strip()
        if project_path:
            try:
                cst.load_project(project_path)
            except FileNotFoundError as e:
                print(f"错误: {e}")
                return

    # 如果配置中没有结果路径，询问用户
    results_path = None
    if not config.cst_results_path:
        results_path = input("请输入仿真结果文件夹路径 (直接回车使用默认路径): ").strip()
        if not results_path:
            results_path = None  # 使用默认路径

    # 询问优化目标
    print("\n请输入优化目标（例如：S11低于-15dB，增益高于8dBi）:")
    print("可以输入多行，输入空行结束:")

    optimization_lines = []
    while True:
        line = input()
        if not line:
            break
        optimization_lines.append(line)

    optimization_goal = "\n".join(optimization_lines)

    if not optimization_goal.strip():
        print("错误: 优化目标不能为空")
        return

    # 询问最大迭代次数
    max_iter_input = input("\n最大优化迭代次数 (默认3): ").strip()
    max_iterations = int(max_iter_input) if max_iter_input.isdigit() else 3

    # 运行优化循环
    optimization_history = optimizer.run_optimization_cycle(
        cst_interface=cst,
        results_path=results_path,
        optimization_goal=optimization_goal,
        max_iterations=max_iterations,
        tags=["自动优化"]
    )

    # 生成并显示优化报告
    report = optimizer.get_optimization_report(optimization_history)
    print(report)

    print("\n优化完成！结果已自动保存到长期记忆库。")


if __name__ == "__main__":
    main()
