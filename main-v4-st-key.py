# -*- coding: utf-8 -*-
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from elasticsearch import Elasticsearch
from config import *  # 导入 OLLAMA_HOST 和其他配置
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from typing import List
import hashlib
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import time  # 新增，用于生成唯一 key
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from assistant_fun import *

# 配置 requests 重试机制
requests.adapters.DEFAULT_RETRIES = 3
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

# 加载 PDF 文件（保持不变）
def load_pdf(pdfpath):
    documents = []
    for filename in os.listdir(pdfpath):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdfpath, filename)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            documents.extend(document)
    return documents

# 上传文件并存储到 ES/FAISS
def upload_files(file_path):
    st.write('1. 开始执行数据预处理与存储步骤...')
    file_info = []

    # 初始化 Elasticsearch 连接
    st.write('2. 初始化 Elasticsearch 并创建索引...')
    es = Elasticsearch(hosts=[ES_HOST])
    index_settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "my_ik_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    }
                }
            },
            # 添加向量检索支持（假设使用 dense_vector 字段）
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "my_ik_analyzer",
                    "search_analyzer": "my_ik_analyzer"
                },
                "embedding": {  # 添加向量字段
                    "type": "dense_vector",
                    "dims": 384  # 根据嵌入模型调整维度，例如 all-MiniLM-L6-v2 是 384
                }
            }
        }
    }
    # 检查并重建索引
    try:
        if es.indices.exists(index=ES_INDEX):
            st.write(f"索引 '{ES_INDEX}' 已存在，正在删除...")
            es.indices.delete(index=ES_INDEX)
        es.indices.create(index=ES_INDEX, body=index_settings)
        st.write(f"成功创建索引 '{ES_INDEX}'")
    except Exception as e:
        st.error(f"创建索引失败: {str(e)}")
        return []

    st.write('3. 处理docx文件...')
    loader = DirectoryLoader(file_path, glob="**/*.docx")
    documents1 = loader.load()
    st.write(f'4. 成功加载 {len(documents1)} 个文档。')
    file_info1 = get_docx_info(file_path, 'docx')
    st.write('处理pdf文件...')
    documents2 = load_pdf(file_path)
    file_info2 = get_docx_info(file_path, 'pdf')

    file_info = file_info1 + file_info2
    documents = documents1 + documents2

    st.write('5. 正在进行智能分块...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n# ", "\n## ", "\n### ", "\n\n"])
    chunks = text_splitter.split_documents(documents)
    st.write(f'6. 分块完成，共 {len(chunks)} 个文本块。')

    st.write('7. 正在初始化向量化模型...')
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True})

    st.write('8. 正在将文本块存储到FAISS...')
    faiss_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "faiss"}) for chunk in chunks]
    vectorstore = FAISS.from_documents(faiss_chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vectorstore.save_local(str(FAISS_INDEX))
    st.write('9. FAISS存储完成。')

    st.write('10. 正在将文本块存入Elasticsearch...')
    es_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "es"}) for chunk in chunks]
    try:
        ElasticsearchStore.from_documents(
            documents=es_chunks, embedding=embeddings, es_connection=es, index_name=ES_INDEX,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True, query_model_id="sentence-transformers/all-MiniLM-L6-v2")
        )
        st.write('11. 成功存入Elasticsearch。')
    except Exception as e:
        st.error(f"存储到Elasticsearch失败: {str(e)}")
        return []

    st.write('12. 验证Elasticsearch存储...')
    try:
        test_query = {"query": {"match_all": {}}, "size": 1}
        test_result = es.search(index=ES_INDEX, body=test_query)
        if test_result['hits']['total']['value'] > 0:
            st.write("ES存储验证成功，首条记录元数据:", test_result['hits']['hits'][0]['_source'].get('metadata'))
        else:
            st.error("ES存储验证失败！")
    except Exception as e:
        st.error(f"验证Elasticsearch存储失败: {str(e)}")

    try:
        count_response = es.count(index=ES_INDEX)
        file_count = count_response['count']
        st.write(f"文件总数: {file_count}")
    except Exception as e:
        st.error(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {"query": {"match_all": {}}, "_source": ["metadata.source"], "size": 10000}
    try:
        search_response = es.search(index=ES_INDEX, body=search_body)
        hits = search_response['hits']['hits']
        seen_filenames = set()
        fileList = []
        for hit in hits:
            source = hit['_source'].get('metadata', {}).get('source', '')
            if source:
                filename = os.path.basename(source)
                if filename in seen_filenames:
                    continue
                seen_filenames.add(filename)
                _, file_extension = os.path.splitext(filename)
                file_type = file_extension[1:] if file_extension else '未知'
                try:
                    file_size = os.path.getsize(source) / (1024 * 1024)
                    file_size = round(file_size, 2)
                except FileNotFoundError:
                    file_size = '未知'
                info = [str(filename), str(file_type), f"{file_size:.2f}"]
                fileList.append(info)
        return fileList
    except Exception as e:
        st.error(f"搜索Elasticsearch失败: {str(e)}")
        return []

# 从 ES 加载文件信息
def load_file_info_from_es():
    es = Elasticsearch(hosts=[ES_HOST])
    try:
        if not es.indices.exists(index=ES_INDEX):
            st.write(f"索引 '{ES_INDEX}' 不存在，未上传任何文件。")
            return []
    except Exception as e:
        st.error(f"检查索引存在性失败: {str(e)}")
        return []

    try:
        count_response = es.count(index=ES_INDEX)
        file_count = count_response['count']
    except Exception as e:
        st.error(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {"query": {"match_all": {}}, "_source": ["metadata.source"], "size": 10000}
    try:
        search_response = es.search(index=ES_INDEX, body=search_body)
        hits = search_response['hits']['hits']
        seen_filenames = set()
        fileList = []
        for hit in hits:
            source = hit['_source'].get('metadata', {}).get('source', '')
            if source:
                filename = os.path.basename(source)
                if filename in seen_filenames:
                    continue
                seen_filenames.add(filename)
                _, file_extension = os.path.splitext(filename)
                file_type = file_extension[1:] if file_extension else '未知'
                try:
                    file_size = os.path.getsize(source) / (1024 * 1024)
                    file_size = round(file_size, 2)
                except FileNotFoundError:
                    file_size = '未知'
                info = [str(filename), str(file_type), f"{file_size:.2f}"]
                fileList.append(info)
        return fileList
    except Exception as e:
        st.error(f"搜索Elasticsearch失败: {str(e)}")
        return []

# RAG 问答链
def rag_chain(question):
    st.write('13. 正在加载检索器...')
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(str(FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)
    es = Elasticsearch(hosts=[ES_HOST])

    class CustomESRetriever(BaseRetriever, BaseModel):
        es: Elasticsearch = Field(...)
        es_index: str = Field(...)
        k: int = Field(default=15)

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> List[Document]:
            try:
                body = {"query": {"match": {"text": {"query": query, "analyzer": "my_ik_analyzer"}}}, "size": self.k}
                response = self.es.search(index=self.es_index, body=body)
                return [Document(page_content=hit["_source"]["text"], metadata=hit["_source"].get("metadata", {})) for hit in response['hits']['hits']]
            except Exception as e:
                st.error(f"Elasticsearch检索失败: {str(e)}")
                return []

    es_retriever = CustomESRetriever(es=es, es_index=ES_INDEX)

    class CustomFAISSRetriever(BaseRetriever, BaseModel):
        vectorstore: FAISS = Field(...)
        score_threshold: float = Field(default=0.5)

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> List[Document]:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=20)
                return [doc for doc, score in docs_with_scores if score >= self.score_threshold]
            except Exception as e:
                st.error(f"FAISS检索失败: {str(e)}")
                return []

    faiss_retriever = CustomFAISSRetriever(vectorstore=vector_db, score_threshold=0.5)

    st.write('并行检索...')
    es_docs, es_scores = retrieve_with_scores(es_retriever, question)
    faiss_docs, faiss_scores = retrieve_with_scores(faiss_retriever, question)

    st.write('合并查询结果...')
    merged_results = merge_results(es_docs=es_docs, faiss_docs=faiss_docs, es_scores=es_scores, faiss_scores=faiss_scores, es_weight=0.4, faiss_weight=0.6, merge_strategy="reciprocal_rank")

    final_docs = [validate_metadata(Document(page_content=res.content, metadata=res.metadata)) for res in merged_results]
    formatted_docs = format_documents(final_docs[:3])
    prompt = f"""你是一位资深的合同、标书文档专家，请根据以下内容回答：
                {formatted_docs}
                问题：{question}"""

    try:
        response = session.post(f"{OLLAMA_HOST}/api/generate", json={"model": "deepseek-r1:14b", "prompt": prompt, "stream": True}, timeout=120, stream=True)
        full_answer = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                full_answer += chunk
                yield full_answer
    except Exception as e:
        st.error(f"模型生成回答失败: {str(e)}")
        yield "生成回答时发生错误，请稍后重试。"

# Streamlit UI
# Streamlit UI
def main():
    st.set_page_config(page_title="RAG (ES+FAISS+Deepseek-R1:14b)", layout="wide")
    st.markdown("""
        <style>
            .stApp {max-width: 2000px;}
            .answer-box {min-height: 500px;}
            .green-button {background-color: green; color: white; padding: 10px; border-radius: 5px;}
            .blue-button {background-color: #ADD8E6; color: white; padding: 10px; border-radius: 5px;}
            .large-font {font-size: 36px;}
            .center-text {text-align: center;}
            .red-text {color: red;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="blue-button large-font center-text">RAG (ES+FAISS+Deepseek-R1:14b)</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("## 📂 DOCX/PDF文件路径")
        # 设置默认路径，用户可修改
        default_path = "/teamspace/studios/this_studio/elasticsearch-Faiss-Deepseek-R1-14B-RAG/AAA-database"
        file_path = st.text_input("输入文件夹路径", value=default_path, key="file_path_input")
        if st.button("文件存储到ES/FAISS库", key="upload_btn"):
            if file_path:
                with st.spinner("处理中..."):
                    file_list = upload_files(file_path)
                    st.session_state['file_list'] = file_list
            else:
                st.error("请提供有效的文件夹路径！")

        st.markdown("## ❓ 提问")
        question = st.text_area("输入问题", placeholder="例如：本文档的主要观点是什么？", height=100, key="question_input")
        if st.button("🔍 开始提问", key="ask_btn"):
            if question and 'file_list' in st.session_state:
                with st.spinner("生成回答中..."):
                    answer_container = st.empty()
                    # 使用计数器生成唯一 key
                    counter = 0
                    for answer in rag_chain(question):
                        counter += 1
                        unique_key = f"answer_output_{int(time.time()*1000)}_{counter}"
                        answer_container.text_area("回答", value=answer, height=400, key=unique_key)
            else:
                st.error("请先上传文件并输入问题！")

    with col2:
        st.markdown("## 已存储文件列表")
        if 'file_list' in st.session_state:
            st.dataframe(st.session_state['file_list'], column_config={"0": "文件名", "1": "文件类型", "2": "大小(MB)"})
        else:
            file_list = load_file_info_from_es()
            st.session_state['file_list'] = file_list
            st.dataframe(file_list, column_config={"0": "文件名", "1": "文件类型", "2": "大小(MB)"})

        st.markdown("## 📝 答案")
        st.markdown("*回答生成可能需要1-2分钟，请耐心等待<br>*支持多轮对话，可基于前文继续提问", unsafe_allow_html=True)

if __name__ == "__main__":
    main()