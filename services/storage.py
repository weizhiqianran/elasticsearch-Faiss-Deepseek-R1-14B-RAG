from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from config.settings import *
from core.document_loader import load_documents, split_documents

def store_documents(directory: str):
    """将文档存储到 ES 和 FAISS"""
    es = Elasticsearch(hosts=[ES_HOST])
    
    # 创建索引
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, body={
        "settings": {"analysis": {"analyzer": {"my_ik_analyzer": {"type": "custom", "tokenizer": "ik_max_word"}}}},
        "mappings": {"properties": {"text": {"type": "text", "analyzer": "my_ik_analyzer"}, "embedding": {"type": "dense_vector", "dims": 384}}}
    })
    
    # 加载和分块
    documents = load_documents(directory)
    chunks = split_documents(documents)
    
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})
    
    # 存储到 FAISS
    faiss_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "faiss"}) for chunk in chunks]
    vectorstore = FAISS.from_documents(faiss_chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX))
    
    # 存储到 ES
    es_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "es"}) for chunk in chunks]
    ElasticsearchStore.from_documents(documents=es_chunks, embedding=embeddings, es_connection=es, index_name=ES_INDEX)