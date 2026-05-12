"""
记忆管理系统 - 短期记忆和长期记忆
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings


@dataclass
class SimulationResult:
    """仿真结果数据结构"""
    simulation_id: str
    timestamp: str
    parameters: Dict
    results: Dict
    summary: str
    tags: List[str]
    metadata: Optional[Dict] = None


class ShortTermMemory:
    """短期记忆 - 存储当前对话历史"""

    def __init__(self, max_turns: int = 10):
        """
        初始化短期记忆

        Args:
            max_turns: 最大保留的对话轮数
        """
        self.max_turns = max_turns
        self.conversation_history: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        添加消息到短期记忆

        Args:
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            metadata: 附加元数据
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)

        # 限制历史记录长度
        if len(self.conversation_history) > self.max_turns * 2:
            self.conversation_history = self.conversation_history[-(self.max_turns * 2):]

    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        获取对话历史

        Args:
            last_n: 获取最近n条消息，None表示全部

        Returns:
            对话历史列表
        """
        if last_n is None:
            return self.conversation_history.copy()
        return self.conversation_history[-last_n:].copy()

    def clear(self):
        """清空短期记忆"""
        self.conversation_history = []
        print(f"短期记忆已清空 (会话ID: {self.session_id})")

    def get_context_string(self, last_n: Optional[int] = None) -> str:
        """
        获取格式化的上下文字符串

        Args:
            last_n: 获取最近n条消息

        Returns:
            格式化的对话历史字符串
        """
        history = self.get_history(last_n)
        context_lines = []
        for msg in history:
            role_label = "用户" if msg['role'] == 'user' else "助手"
            context_lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(context_lines)


class LongTermMemory:
    """长期记忆 - 存储仿真结果供后续参考"""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "long_memory",
                 use_embedding: bool = True):
        """
        初始化长期记忆

        Args:
            persist_dir: ChromaDB持久化目录
            collection_name: 集合名称
            use_embedding: 是否使用向量嵌入（设为False可大幅提升速度）
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.use_embedding = use_embedding
        self.chroma_client = None
        self.collection = None

        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化向量数据库
        self._init_db()

    def _init_db(self):
        """初始化ChromaDB"""
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
            print(f"长期记忆初始化完成，集合: {self.collection_name}，已有 {count} 条仿真记录")

    def add_simulation_result(self,
                             simulation_id: str,
                             parameters: Dict,
                             results: Dict,
                             summary: str,
                             tags: Optional[List[str]] = None,
                             metadata: Optional[Dict] = None):
        """
        添加仿真结果到长期记忆

        Args:
            simulation_id: 仿真ID
            parameters: 仿真参数字典
            results: 仿真结果字典
            summary: 仿真结果的一句话总结
            tags: 标签列表，用于分类
            metadata: 其他元数据
        """
        self._init_db()

        if tags is None:
            tags = []

        # 构建文档内容（用于向量检索）
        param_text = " ".join([f"{k}: {v}" for k, v in parameters.items()])
        result_text = summary + " " + " ".join([f"{k}: {v}" for k, v in results.items()])
        document_content = f"{param_text} {result_text} {' '.join(tags)}"

        # 构建元数据
        doc_metadata = {
            'simulation_id': simulation_id,
            'timestamp': datetime.now().isoformat(),
            'parameters_json': json.dumps(parameters, ensure_ascii=False),
            'results_json': json.dumps(results, ensure_ascii=False),
            'summary': summary,
            'tags_json': json.dumps(tags, ensure_ascii=False),
        }

        if metadata:
            doc_metadata.update(metadata)

        # 添加到ChromaDB（不使用嵌入向量，直接使用ID索引）
        if self.use_embedding:
            # 使用嵌入向量（慢，但支持语义搜索）
            try:
                import dashscope
                from dashscope import TextEmbedding

                response = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v2,
                    input=[document_content]
                )

                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    self.collection.add(
                        documents=[document_content],
                        embeddings=[embedding],
                        metadatas=[doc_metadata],
                        ids=[simulation_id]
                    )
                else:
                    # 降级：不使用嵌入
                    self.collection.add(
                        documents=[document_content],
                        metadatas=[doc_metadata],
                        ids=[simulation_id]
                    )
            except Exception as e:
                print(f"⚠️ 嵌入生成失败，使用无嵌入模式: {e}")
                self.collection.add(
                    documents=[document_content],
                    metadatas=[doc_metadata],
                    ids=[simulation_id]
                )
        else:
            # 不使用嵌入向量（快，但不支持语义搜索）
            self.collection.add(
                documents=[document_content],
                metadatas=[doc_metadata],
                ids=[simulation_id]
            )

        print(f"✅ 仿真结果已自动保存到长期记忆: {simulation_id}")

    def search_similar_simulations(self,
                                  query: str,
                                  top_k: int = 5,
                                  filter_params: Optional[Dict] = None) -> List[Dict]:
        """
        搜索相似的仿真结果

        Args:
            query: 查询文本（可以是参数描述或结果描述）
            top_k: 返回结果数量
            filter_params: 过滤条件（如特定参数范围）

        Returns:
            相似的仿真结果列表
        """
        self._init_db()

        if self.use_embedding:
            # 使用向量搜索
            try:
                import dashscope
                from dashscope import TextEmbedding

                response = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v2,
                    input=[query]
                )

                if response.status_code == 200:
                    query_embedding = response.output['embeddings'][0]['embedding']
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k
                    )
                else:
                    # 降级：使用关键词匹配
                    results = self._keyword_search(query, top_k)
            except Exception as e:
                print(f"⚠️ 向量搜索失败，使用关键词搜索: {e}")
                results = self._keyword_search(query, top_k)
        else:
            # 使用关键词搜索
            results = self._keyword_search(query, top_k)

        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]

            # 解析JSON字段
            try:
                parameters = json.loads(metadata.get('parameters_json', '{}'))
                results_data = json.loads(metadata.get('results_json', '{}'))
                tags = json.loads(metadata.get('tags_json', '[]'))
            except json.JSONDecodeError:
                parameters = {}
                results_data = {}
                tags = []

            formatted_results.append({
                'simulation_id': metadata.get('simulation_id', ''),
                'timestamp': metadata.get('timestamp', ''),
                'parameters': parameters,
                'results': results_data,
                'summary': metadata.get('summary', ''),
                'tags': tags,
                'distance': results['distances'][0][i] if 'distances' in results else 0,
                'relevance_score': 1.0 / (1.0 + results['distances'][0][i]) if 'distances' in results else 0
            })

        return formatted_results

    def _keyword_search(self, query: str, top_k: int) -> Dict:
        """
        关键词搜索（快速但不精确）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            搜索结果
        """
        # 获取所有记录
        count = self.collection.count()
        if count == 0:
            return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}

        all_results = self.collection.get(
            limit=min(count, 100),
            include=['metadatas', 'documents']
        )

        # 简单关键词匹配
        query_lower = query.lower()
        scored_results = []

        for i, doc in enumerate(all_results['documents']):
            if doc and query_lower in doc.lower():
                scored_results.append((i, 0.1))  # 匹配的距离设为0.1
            else:
                scored_results.append((i, 1.0))  # 不匹配的距离设为1.0

        # 按距离排序
        scored_results.sort(key=lambda x: x[1])
        top_indices = [idx for idx, _ in scored_results[:top_k]]

        # 构建返回格式
        filtered_ids = [all_results['ids'][i] for i in top_indices]
        filtered_metadatas = [all_results['metadatas'][i] for i in top_indices]
        filtered_distances = [scored_results[i][1] for i in range(len(top_indices))]

        return {
            'ids': [filtered_ids],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances]
        }

    def get_simulation_by_id(self, simulation_id: str) -> Optional[Dict]:
        """
        根据ID获取仿真结果

        Args:
            simulation_id: 仿真ID

        Returns:
            仿真结果字典，未找到返回None
        """
        self._init_db()

        try:
            results = self.collection.get(ids=[simulation_id])

            if not results['ids']:
                return None

            metadata = results['metadatas'][0][0]

            # 解析JSON字段
            parameters = json.loads(metadata.get('parameters_json', '{}'))
            results_data = json.loads(metadata.get('results_json', '{}'))
            tags = json.loads(metadata.get('tags_json', '[]'))

            return {
                'simulation_id': metadata.get('simulation_id', ''),
                'timestamp': metadata.get('timestamp', ''),
                'parameters': parameters,
                'results': results_data,
                'summary': metadata.get('summary', ''),
                'tags': tags
            }
        except Exception as e:
            print(f"获取仿真结果失败: {e}")
            return None

    def list_simulations(self,
                        tags: Optional[List[str]] = None,
                        limit: int = 100) -> List[Dict]:
        """
        列出仿真记录

        Args:
            tags: 按标签过滤
            limit: 返回数量限制

        Returns:
            仿真记录列表
        """
        self._init_db()

        # 获取所有记录
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.get(
            limit=min(limit, count),
            include=['metadatas']
        )

        formatted_results = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]

            # 如果指定了标签过滤
            if tags:
                record_tags = json.loads(metadata.get('tags_json', '[]'))
                if not any(tag in record_tags for tag in tags):
                    continue

            try:
                parameters = json.loads(metadata.get('parameters_json', '{}'))
                results_data = json.loads(metadata.get('results_json', '{}'))
                tags_list = json.loads(metadata.get('tags_json', '[]'))
            except json.JSONDecodeError:
                parameters = {}
                results_data = {}
                tags_list = []

            formatted_results.append({
                'simulation_id': metadata.get('simulation_id', ''),
                'timestamp': metadata.get('timestamp', ''),
                'parameters': parameters,
                'results': results_data,
                'summary': metadata.get('summary', ''),
                'tags': tags_list
            })

        return formatted_results

    def delete_simulation(self, simulation_id: str) -> bool:
        """
        删除仿真记录

        Args:
            simulation_id: 仿真ID

        Returns:
            是否删除成功
        """
        self._init_db()

        try:
            self.collection.delete(ids=[simulation_id])
            print(f"仿真记录已删除: {simulation_id}")
            return True
        except Exception as e:
            print(f"删除仿真记录失败: {e}")
            return False

    def clear_all(self):
        """清空所有长期记忆"""
        self._init_db()

        count = self.collection.count()
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = None
        self._init_db()

        print(f"长期记忆已清空，删除了 {count} 条记录")


class MemoryManager:
    """记忆管理器 - 统一管理短期和长期记忆"""

    def __init__(self,
                 short_term_max_turns: int = 10,
                 long_term_persist_dir: str = "./chroma_db",
                 long_term_collection_name: str = "long_memory",
                 use_embedding: bool = False):  # 默认不使用嵌入，提升速度
        """
        初始化记忆管理器

        Args:
            short_term_max_turns: 短期记忆最大对话轮数
            long_term_persist_dir: 长期记忆持久化目录
            long_term_collection_name: 长期记忆集合名称
            use_embedding: 是否使用向量嵌入（False=快速模式）
        """
        self.short_term = ShortTermMemory(max_turns=short_term_max_turns)
        self.long_term = LongTermMemory(
            persist_dir=long_term_persist_dir,
            collection_name=long_term_collection_name,
            use_embedding=use_embedding
        )

        mode = "向量模式（支持语义搜索）" if use_embedding else "快速模式（关键词搜索）"
        print("记忆管理系统初始化完成")
        print(f"  - 短期记忆: 最多保留 {short_term_max_turns} 轮对话")
        print(f"  - 长期记忆: {long_term_persist_dir} (集合: {long_term_collection_name})")
        print(f"  - 搜索模式: {mode}")

    def add_user_message(self, message: str, metadata: Optional[Dict] = None):
        """添加用户消息到短期记忆"""
        self.short_term.add_message('user', message, metadata)

    def add_assistant_message(self, message: str, metadata: Optional[Dict] = None):
        """添加助手消息到短期记忆"""
        self.short_term.add_message('assistant', message, metadata)

    def save_simulation_result(self,
                              simulation_id: str,
                              parameters: Dict,
                              results: Dict,
                              summary: str,
                              tags: Optional[List[str]] = None):
        """保存仿真结果到长期记忆"""
        self.long_term.add_simulation_result(
            simulation_id=simulation_id,
            parameters=parameters,
            results=results,
            summary=summary,
            tags=tags
        )

    def search_similar_simulations(self,
                                  query: str,
                                  top_k: int = 5) -> List[Dict]:
        """搜索相似的仿真结果"""
        return self.long_term.search_similar_simulations(query, top_k)

    def get_conversation_context(self, last_n: Optional[int] = None) -> str:
        """获取对话上下文"""
        return self.short_term.get_context_string(last_n)

    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term.clear()

    def clear_long_term(self):
        """清空长期记忆"""
        self.long_term.clear_all()
