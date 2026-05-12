
"""
配置管理中心
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 通义千问配置
    dashscope_api_key: str = "sk-0e434c239b454d59bedddf62eaa3f7fe"
    qwen_model_name: str = "qwen3-max"

    # 数据库配置
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "papers"

    # 检索参数
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_context_chars: int = 3000

    # BM25参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # CST配置
    cst_project_path: str = ""
    cst_results_path: str = ""

    def __post_init__(self):
        """初始化后从环境变量加载配置"""
        if not self.dashscope_api_key:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

        # 从环境变量加载CST路径配置
        if not self.cst_project_path:
            self.cst_project_path = os.getenv("CST_PROJECT_PATH", "")

        if not self.cst_results_path:
            self.cst_results_path = os.getenv("CST_RESULTS_PATH", "")

