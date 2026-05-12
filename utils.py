"""
工具函数
"""
import re
from typing import List, Dict


def normalize_whitespace(text: str) -> str:
    """标准化空白字符"""
    # 将多个空格、换行符等替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_context(documents: List[Dict]) -> str:
    """格式化上下文文档"""
    if not documents:
        return ""

    context_parts = []
    for idx, doc in enumerate(documents, 1):
        content = doc.get('content', '')
        source = doc.get('metadata', {}).get('source', f'文档#{idx}')
        chunk_id = doc.get('id', f'chunk_{idx}')

        # 截取内容片段（限制长度）
        if len(content) > 300:
            content = content[:300] + "..."

        context_part = f"[{idx}] 来源: {source}\n内容: {content}\n"
        context_parts.append(context_part)

    return "\n".join(context_parts)


def extract_citations(documents: List[Dict]) -> List[Dict]:
    """提取引用信息"""
    citations = []
    for doc in documents:
        citation = {
            'id': doc.get('id', ''),
            'source': doc.get('metadata', {}).get('source', '未知来源'),
            'chunk_id': doc.get('id', ''),
            'content_preview': doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
        }
        citations.append(citation)
    return citations


def split_text_into_chunks(text: str,
                          chunk_size: int = 500,
                          chunk_overlap: int = 50) -> List[str]:
    """将文本分割成块"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # 如果不在文本末尾，尝试在句子边界处分割
        if end < text_length:
            # 查找最近的句子结束符
            for sep in ['。', '！', '？', '.', '!', '?', '\n']:
                last_sep = text.rfind(sep, start, end)
                if last_sep != -1:
                    end = last_sep + 1
                    break

        # 提取文本块
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 移动起始位置（考虑重叠）
        start = end - chunk_overlap if end < text_length else end

    return chunks


def load_text_file(file_path: str) -> str:
    """加载文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """加载PDF文件（需要安装PyPDF2或pdfplumber）"""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except ImportError:
        print("请安装pdfplumber: pip install pdfplumber")
        return ""
    except Exception as e:
        print(f"读取PDF文件失败: {e}")
        return ""