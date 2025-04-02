from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from elasticsearch import Elasticsearch
from config.settings import *
from core.retriever import ESRetriever, FAISSRetriever, merge_results
from core.utils import safe_get_metadata
from services.formatter import format_documents
import requests

def rag_chain(question: str, max_docs: int = 5):
    """RAG 问答链"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(str(FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)
    es = Elasticsearch(hosts=[ES_HOST])
    
    es_retriever = ESRetriever(es=es, index=ES_INDEX)
    faiss_retriever = FAISSRetriever(vectorstore=vector_db)
    
    # 检索
    es_docs = es_retriever._get_relevant_documents(question)
    faiss_docs = faiss_retriever._get_relevant_documents(question)
    
    # 合并结果
    merged_results = merge_results(es_docs, faiss_docs, [1.0] * len(es_docs), [1.0] * len(faiss_docs))
    top_docs = merged_results[:max_docs]
    
    # 生成回答
    prompt = f"你是一位资深的合同、标书文档专家，请根据以下内容回答：\n{format_documents([doc for doc in top_docs])}\n问题：{question}"
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}, stream=True)
    
    answer = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode()).get("response", "")
            answer += chunk
            yield answer, top_docs