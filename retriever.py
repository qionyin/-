"""
混合检索器 - 结合向量检索和BM25关键词检索
"""
import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict
from config import RAGConfig
from embedding_utils import EmbeddingUtils


class BM25Indexer:
    """BM25关键词索引"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents = {}
        self.doc_freq = defaultdict(int)
        self.term_freq = defaultdict(lambda: defaultdict(int))
        self.num_docs = 0
        self.avg_doc_len = 0
        self.total_terms = 0
        self.vocabulary = set()

    def tokenize(self, text: str) -> List[str]:
        """文本分词（简单中文分词）"""
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 对于中文，按字符分割（实际应用中可使用jieba等分词工具）
        tokens = list(text)
        # 过滤空白字符
        tokens = [t for t in tokens if t.strip()]
        return tokens

    def build(self, documents: Dict[str, str]):
        """构建BM25索引"""
        self.documents = documents
        self.num_docs = len(documents)

        # 计算每个文档的词频和文档频率
        for doc_id, doc_text in documents.items():
            tokens = self.tokenize(doc_text)
            self.total_terms += len(tokens)

            # 统计词频
            for token in tokens:
                self.term_freq[doc_id][token] += 1
                self.vocabulary.add(token)

            # 统计文档频率（去重）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1

        # 计算平均文档长度
        self.avg_doc_len = self.total_terms / self.num_docs if self.num_docs > 0 else 0

        print(f"BM25索引构建完成: {self.num_docs} 个文档, {len(self.vocabulary)} 个唯一词项")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25搜索"""
        query_tokens = self.tokenize(query)
        scores = defaultdict(float)

        k1 = self.config.bm25_k1
        b = self.config.bm25_b

        for token in query_tokens:
            if token not in self.vocabulary:
                continue

            df = self.doc_freq[token]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf_dict in self.term_freq.items():
                if token in tf_dict:
                    tf = tf_dict[token]
                    doc_len = sum(tf_dict.values())

                    # BM25公式
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    bm25_score = idf * (numerator / denominator)

                    scores[doc_id] += bm25_score

        # 排序并返回top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append({
                'id': doc_id,
                'content': self.documents[doc_id],
                'score': score
            })

        return results


class HybridRetriever:
    """混合检索器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_utils = EmbeddingUtils(config)
        self.bm25_indexer = BM25Indexer(config)
        self.documents_store = {}

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到检索器"""
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        # 存储文档
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_ids.append(doc_id)
            self.documents_store[doc_id] = doc

        # 添加到向量数据库
        self.embedding_utils.add_documents(documents, metadatas, doc_ids)

        # 构建BM25索引
        self.bm25_indexer.build(self.documents_store)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """混合检索"""
        if top_k is None:
            top_k = self.config.top_k

        # 向量检索
        vector_results = self.embedding_utils.similarity_search(query, top_k=top_k * 2)

        # BM25检索
        bm25_results = self.bm25_indexer.search(query, top_k=top_k * 2)

        # 融合结果
        merged_results = self._merge_results(vector_results, bm25_results, top_k)

        return merged_results

    def _merge_results(self,
                      vector_results: List[Dict],
                      bm25_results: List[Dict],
                      top_k: int) -> List[Dict]:
        """融合向量检索和BM25检索结果"""

        # 创建文档评分字典
        doc_scores = {}

        # 处理向量检索结果（距离越小越相似，转换为分数）
        for result in vector_results:
            doc_id = result['id']
            # 将距离转换为相似度分数
            score = 1.0 / (1.0 + result.get('distance', 1.0))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': score,
                    'bm25_score': 0,
                    'combined_score': 0
                }
            else:
                doc_scores[doc_id]['vector_score'] = score

        # 处理BM25检索结果
        for result in bm25_results:
            doc_id = result['id']
            score = result['score']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': {},
                    'vector_score': 0,
                    'bm25_score': score,
                    'combined_score': 0
                }
            else:
                doc_scores[doc_id]['bm25_score'] = score

        # 计算综合分数（加权平均）
        alpha = 0.7  # 向量检索权重
        beta = 0.3   # BM25权重

        for doc_id, doc_info in doc_scores.items():
            doc_info['combined_score'] = alpha * doc_info['vector_score'] + beta * doc_info['bm25_score']

        # 按综合分数排序
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]

        return sorted_results