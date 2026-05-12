"""
向量嵌入与检索工具
"""
import numpy as np
from typing import List, Dict, Optional
import dashscope
from dashscope import TextEmbedding
from config import RAGConfig


class EmbeddingUtils:
    """向量嵌入工具类"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chroma_client = None
        self.collection = None

        # 设置DashScope API密钥
        if config.dashscope_api_key:
            dashscope.api_key = config.dashscope_api_key

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """为文档生成嵌入向量"""
        embeddings = []

        # DashScope批量处理，每次最多25个文档
        batch_size = 25
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v2,
                input=batch
            )

            if response.status_code == 200:
                for item in response.output['embeddings']:
                    embeddings.append(item['embedding'])
            else:
                raise Exception(f"嵌入生成失败: {response.code} - {response.message}")

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """为查询生成嵌入向量"""
        response = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v2,
            input=[query]
        )

        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"嵌入生成失败: {response.code} - {response.message}")

    def init_chromadb(self):
        """初始化ChromaDB"""
        if self.chroma_client is None:
            import chromadb
            from chromadb.config import Settings

            self.chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"ChromaDB初始化完成，集合: {self.config.collection_name}")

    def add_documents(self,
                     documents: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None):
        """添加文档到ChromaDB"""
        self.init_chromadb()

        # 生成嵌入向量
        print(f"正在为 {len(documents)} 个文档生成嵌入向量...")
        embeddings = self.embed_documents(documents)

        # 如果没有提供ID，自动生成
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # 如果没有提供元数据，使用空字典
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        # 添加到ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"成功添加 {len(documents)} 个文档到ChromaDB")

    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """相似性搜索"""
        self.init_chromadb()

        if top_k is None:
            top_k = self.config.top_k

        # 生成查询向量
        query_embedding = self.embed_query(query)

        # 搜索相似文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })

        return formatted_results

    def clear_collection(self):
        """清空集合"""
        self.init_chromadb()
        count = self.collection.count()
        self.chroma_client.delete_collection(name=self.config.collection_name)
        self.collection = None
        print(f"已清空集合，删除了 {count} 个文档")
