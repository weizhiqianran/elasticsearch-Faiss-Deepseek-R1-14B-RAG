from main import *
from pydantic import BaseModel, Field, field_validator  # 确保导入 field_validator
from typing import List, Dict, Any
import hashlib
import json
from io import StringIO
import os
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

def extract_text(filepath):
    """改进的PDF文本提取方法"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

def safe_doc_access(docs: List[Document], index: int) -> Document:
    try:
        return docs[index]
    except IndexError:
        return Document(page_content="", metadata={})

def safe_metadata(meta) -> Dict[str, Any]:
    """安全转换元数据为字典"""
    if isinstance(meta, dict):
        return meta
    try:
        return json.loads(meta)
    except:
        return {"source": "unknown"}

class SafeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        raw_docs = self.backend.search(query)
        return [
            Document(
                page_content=str(doc.content),
                metadata=safe_metadata(doc.metadata)
            ) for doc in raw_docs
        ]

def safe_get_metadata(doc, default=None) -> dict:
    meta = getattr(doc, 'metadata', {})
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except:
            return {"source": meta}  # 降级处理
    return meta or default

class MergedRetrievalResult(BaseModel):
    """统一检索结果结构"""
    content: str
    metadata: dict = Field(default_factory=dict)  # 使用 default_factory 指定默认值
    score: float
    source: str

    @field_validator('metadata', mode='before')  # 使用 field_validator，mode='before' 表示在验证前处理
    def validate_metadata(cls, v):
        if isinstance(v, str):
            return {"raw_meta": v}  # 自动转换字符串元数据
        return v or {}

def safe_get_metadata(doc) -> dict:  # 注意：这里有两个同名函数，后定义的会覆盖前一个
    """多层防御的元数据获取"""
    try:
        meta = getattr(doc, 'metadata', {})
        if isinstance(meta, str):
            return json.loads(meta)
        return dict(meta)  # 强制转换为字典
    except Exception:
        return {"source": "unknown", "page": 0}

def get_docx_info(directory, filetype):
    file_info = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                info = {
                    '文件名': file,
                    '文件路径': file_path,
                    '文件类型': filetype,
                    '大小(MB)': file_size
                }
                file_info.append(info)
    return file_info

def V_analyze(strText):
    print('验证分词效果...')
    es = Elasticsearch(hosts=[ES_HOST])
    test_analyze = es.indices.analyze(
        index=ES_INDEX,
        body={
            "analyzer": "my_ik_analyzer",
            "text": strText
        }
    )
    print("分词结果：", [t["token"] for t in test_analyze["tokens"]])

def retrieve_with_scores(retriever: BaseRetriever, question) -> tuple:
    if hasattr(retriever, 'get_relevant_documents_with_scores'):
        return retriever.get_relevant_documents_with_scores(question)
    else:
        docs = retriever._get_relevant_documents(question)
        return docs, [1.0] * len(docs)

def merge_results(es_docs, faiss_docs, es_scores, faiss_scores, es_weight=0.4, faiss_weight=0.6, merge_strategy="weighted"):
    merged = []

    for i in range(len(es_docs)):
        doc = safe_doc_access(es_docs, i)
        meta = safe_get_metadata(doc)
        meta.setdefault("source", f"es_doc_{i}")
        meta.setdefault("page", 0)
        score = es_scores[i] if i < len(es_scores) else 0.0
        merged.append(
            MergedRetrievalResult(
                content=str(doc.page_content),
                metadata=meta,
                score=score * es_weight,
                source="es"
            )
        )

    for i in range(len(faiss_docs)):
        doc = safe_doc_access(faiss_docs, i)
        score = faiss_scores[i] if i < len(faiss_scores) else 0.0
        merged.append(
            MergedRetrievalResult(
                content=str(doc.page_content),
                metadata=safe_get_metadata(doc),
                score=score * faiss_weight,
                source="faiss"
            )
        )

    seen = set()
    unique_results = []
    for res in merged:
        content_hash = hashlib.md5(res.content.encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_results.append(res)

    print("Merged results before sorting:")
    for res in unique_results:
        print(res)

    if merge_strategy == "weighted":
        unique_results.sort(key=lambda x: x.score, reverse=True)
    elif merge_strategy == "reciprocal_rank":
        for i, res in enumerate(unique_results):
            rank = i + 1
            res.score = 1.0 / (60 + rank)
        unique_results.sort(key=lambda x: x.score, reverse=True)

    return unique_results

def format_documents(docs: List[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", 0)
        rgx = doc.metadata.get("retriever_source", "未知")
        content = doc.page_content.replace("\n", " ").strip()[:300]
        formatted.append(
            f"【文档{idx}】\n"
            f"来源: {source}\n"
            f"es/faiss: {rgx}\n"
            f"页码: {page}\n"
            f"内容: {content}..."
        )
    return "\n\n".join(formatted)

def validate_metadata(doc: Document) -> Document:
    return Document(
        page_content=doc.page_content,
        metadata=safe_get_metadata(doc)
    )