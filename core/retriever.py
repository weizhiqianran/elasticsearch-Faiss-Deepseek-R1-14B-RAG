from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from typing import List
from .utils import safe_get_metadata
from .models import MergedRetrievalResult

class ESRetriever(BaseRetriever):
    def __init__(self, es: Elasticsearch, index: str, k: int = 15):
        self.es = es
        self.index = index
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        body = {"query": {"match": {"text": {"query": query}}}, "size": self.k}
        response = self.es.search(index=self.index, body=body)
        return [Document(page_content=hit["_source"]["text"], metadata=hit["_source"].get("metadata", {})) 
                for hit in response['hits']['hits']]

class FAISSRetriever(BaseRetriever):
    def __init__(self, vectorstore: FAISS, score_threshold: float = 0.3):
        self.vectorstore = vectorstore
        self.score_threshold = score_threshold
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=20)
        return [doc for doc, score in docs_with_scores if score >= self.score_threshold] or [docs_with_scores[0][0]]

def merge_results(es_docs: List[Document], faiss_docs: List[Document], es_scores: List[float], faiss_scores: List[float], 
                 es_weight: float = 0.4, faiss_weight: float = 0.6) -> List[MergedRetrievalResult]:
    """合并 ES 和 FAISS 检索结果"""
    merged = []
    for i, doc in enumerate(es_docs):
        score = es_scores[i] if i < len(es_scores) else 0.0
        merged.append(MergedRetrievalResult(content=doc.page_content, metadata=safe_get_metadata(doc), 
                                           score=score * es_weight, source="es"))
    
    for i, doc in enumerate(faiss_docs):
        score = faiss_scores[i] if i < len(faiss_scores) else 0.0
        merged.append(MergedRetrievalResult(content=doc.page_content, metadata=safe_get_metadata(doc), 
                                           score=score * faiss_weight, source="faiss"))
    
    # 去重并排序
    seen = set()
    unique_results = [res for res in merged if not (res.content.encode() in seen or seen.add(res.content.encode()))]
    unique_results.sort(key=lambda x: x.score, reverse=True)
    return unique_results