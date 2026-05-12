"""
RAG主系统（编排层）- 纯对话系统
"""
from typing import List, Dict, Optional
from config import RAGConfig
from retriever import HybridRetriever
from llm_client import QwenLLMClient
from utils import format_context, extract_citations, split_text_into_chunks, normalize_whitespace
from cst.memory_manager import MemoryManager


class RAGSystem:
    """RAG对话系统主类"""

    def __init__(self, config: RAGConfig = None, use_fast_mode: bool = True):
        """初始化RAG系统"""
        self.config = config or RAGConfig()
        self.retriever = HybridRetriever(self.config)
        self.llm_client = QwenLLMClient(self.config)
        self.memory_manager = MemoryManager(
            short_term_max_turns=10,
            long_term_persist_dir=self.config.chroma_persist_dir,
            long_term_collection_name="long_memory",
            use_embedding=not use_fast_mode
        )
        self.is_initialized = False

    def initialize(self):
        """初始化系统"""
        print("=" * 50)
        print("RAG系统初始化中...")
        print("=" * 50)

        # 使用阿里云DashScope嵌入服务
        print("使用阿里云DashScope文本嵌入服务")

        # 初始化LLM
        try:
            self.llm_client._init_llm()
        except Exception as e:
            print(f"警告: LLM初始化失败: {e}")
            print("系统将仅支持检索功能，无法生成回答")

        self.is_initialized = True
        print("RAG系统初始化完成！")
        print("=" * 50)

    def add_documents(self,
                     texts: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     auto_chunk: bool = True,
                     chunk_size: int = None,
                     chunk_overlap: int = None):
        """添加文档到系统"""
        if not self.is_initialized:
            self.initialize()

        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.config.chunk_overlap

        # 如果需要自动分块
        if auto_chunk:
            all_chunks = []
            all_metadatas = []

            for idx, text in enumerate(texts):
                text = normalize_whitespace(text)
                chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

                metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}

                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = chunk_idx
                    chunk_metadata['source'] = metadata.get('source', f'document_{idx}')
                    all_metadatas.append(chunk_metadata)

            print(f"文档分块完成: {len(texts)} 个文档 -> {len(all_chunks)} 个文本块")
            texts = all_chunks
            metadatas = all_metadatas
        else:
            # 标准化文本
            texts = [normalize_whitespace(text) for text in texts]
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]

        # 添加到检索器
        self.retriever.add_documents(texts, metadatas)
        print(f"成功添加 {len(texts)} 个文档到系统")

    def query(self,
             question: str,
             top_k: int = None,
             use_llm: bool = True) -> Dict:
        """查询问题（不保存对话历史）"""
        if not self.is_initialized:
            self.initialize()

        if top_k is None:
            top_k = self.config.top_k

        print(f"\n正在处理问题: {question}")
        print("-" * 50)

        # 1. 检索相关文档
        print("步骤1: 检索相关文档...")
        relevant_docs = self.retriever.retrieve(question, top_k)
        print(f"找到 {len(relevant_docs)} 个相关文档")

        # 2. 格式化上下文
        print("步骤2: 格式化上下文...")
        context = format_context(relevant_docs)

        # 3. 提取引用
        citations = extract_citations(relevant_docs)

        # 4. 生成答案
        answer = ""
        if use_llm:
            print("步骤3: 生成答案...")
            try:
                answer = self.llm_client.chat(
                    user_message=question,
                    context=context
                )
                print("答案生成完成")
            except Exception as e:
                print(f"LLM调用失败: {e}")
                answer = f"抱歉，生成答案时出错: {str(e)}"
        else:
            print("跳过LLM生成（use_llm=False）")
            answer = "检索完成，但未调用LLM生成答案"

        # 5. 构建结果
        result = {
            'question': question,
            'answer': answer,
            'relevant_documents': relevant_docs,
            'citations': citations,
            'context': context
        }

        return result

    def chat(self,
            question: str,
            top_k: int = None,
            use_history: bool = True,
            save_to_memory: bool = True) -> Dict:
        """对话模式（支持多轮对话，自动保存到短期记忆）"""
        if not self.is_initialized:
            self.initialize()

        if top_k is None:
            top_k = self.config.top_k

        print(f"\n用户: {question}")
        print("-" * 50)

        # 保存用户消息到短期记忆
        if save_to_memory:
            self.memory_manager.add_user_message(question)

        # 检索相关文档
        relevant_docs = self.retriever.retrieve(question, top_k)
        context = format_context(relevant_docs)
        citations = extract_citations(relevant_docs)

        # 生成答案（使用历史）
        if use_history:
            answer = self.llm_client.chat_with_history(
                user_message=question,
                context=context
            )
        else:
            answer = self.llm_client.chat(
                user_message=question,
                context=context
            )

        # 保存助手回答到短期记忆
        if save_to_memory:
            self.memory_manager.add_assistant_message(answer)

        print(f"助手: {answer}")

        result = {
            'question': question,
            'answer': answer,
            'relevant_documents': relevant_docs,
            'citations': citations
        }

        return result

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """获取对话历史"""
        return self.memory_manager.short_term.get_history(last_n)

    def clear_chat_history(self):
        """清空聊天历史"""
        self.llm_client.clear_history()
        self.memory_manager.clear_short_term()

    def display_result(self, result: Dict):
        """显示查询结果"""
        print("\n" + "=" * 50)
        print("查询结果")
        print("=" * 50)
        print(f"问题: {result['question']}")
        print("-" * 50)
        print(f"答案:\n{result['answer']}")
        print("-" * 50)
        print(f"相关文档数量: {len(result['relevant_documents'])}")

        if result.get('citations'):
            print("\n引用来源:")
            for idx, citation in enumerate(result['citations'], 1):
                print(f"  [{idx}] {citation['source']}")

        print("=" * 50)
