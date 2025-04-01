# -*- coding: utf-8 -*-
import os
import time
import json
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List
from pydantic import BaseModel, Field

import streamlit as st
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import *  # 导入 OLLAMA_HOST 和其他配置
from assistant_fun import *

# 配置 requests 重试机制
requests.adapters.DEFAULT_RETRIES = 3
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))


# --------------------
# 文件加载与处理函数
# --------------------
def load_pdf(pdfpath: str) -> List[Document]:
    """加载指定路径中的所有 PDF 文件"""
    documents = []
    for filename in os.listdir(pdfpath):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdfpath, filename)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            documents.extend(document)
    return documents


# --------------------
# 数据存储与索引函数
# --------------------
def upload_files(file_path: str) -> List[List[str]]:
    """上传文件并存储到 Elasticsearch 和 FAISS"""
    st.write('1. 开始执行数据预处理与存储步骤...')
    file_info = []

    # 初始化 Elasticsearch 连接和索引
    st.write('2. 初始化 Elasticsearch 并创建索引...')
    es = Elasticsearch(hosts=[ES_HOST])
    index_settings = {
        "settings": {
            "analysis": {"analyzer": {"my_ik_analyzer": {"type": "custom", "tokenizer": "ik_max_word"}}},
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "my_ik_analyzer", "search_analyzer": "my_ik_analyzer"},
                "embedding": {"type": "dense_vector", "dims": 384}  # 适配 all-MiniLM-L6-v2 模型
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

    # 加载并处理文件
    st.write('3. 处理 DOCX 文件...')
    loader = DirectoryLoader(file_path, glob="**/*.docx")
    documents1 = loader.load()
    st.write(f'4. 成功加载 {len(documents1)} 个 DOCX 文档。')
    file_info1 = get_docx_info(file_path, 'docx')

    st.write('5. 处理 PDF 文件...')
    documents2 = load_pdf(file_path)
    file_info2 = get_docx_info(file_path, 'pdf')

    file_info = file_info1 + file_info2
    documents = documents1 + documents2

    # 智能分块
    st.write('6. 正在进行智能分块...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n# ", "\n## ", "\n### ", "\n\n"]
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f'7. 分块完成，共 {len(chunks)} 个文本块。')

    # 初始化向量化模型
    st.write('8. 正在初始化向量化模型...')
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True}
    )

    # 存储到 FAISS
    st.write('9. 正在将文本块存储到 FAISS...')
    faiss_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "faiss"}) for chunk in chunks]
    vectorstore = FAISS.from_documents(faiss_chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vectorstore.save_local(str(FAISS_INDEX))
    st.write('10. FAISS 存储完成。')

    # 存储到 Elasticsearch
    st.write('11. 正在将文本块存入 Elasticsearch...')
    es_chunks = [Document(page_content=chunk.page_content, metadata={**chunk.metadata, "retriever_source": "es"}) for chunk in chunks]
    try:
        ElasticsearchStore.from_documents(
            documents=es_chunks, embedding=embeddings, es_connection=es, index_name=ES_INDEX,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True, query_model_id="sentence-transformers/all-MiniLM-L6-v2")
        )
        st.write('12. 成功存入 Elasticsearch。')
    except Exception as e:
        st.error(f"存储到 Elasticsearch 失败: {str(e)}")
        return []

    # 验证 Elasticsearch 存储
    st.write('13. 验证 Elasticsearch 存储...')
    try:
        test_query = {"query": {"match_all": {}}, "size": 1}
        test_result = es.search(index=ES_INDEX, body=test_query)
        if test_result['hits']['total']['value'] > 0:
            st.write("ES 存储验证成功，首条记录元数据:", test_result['hits']['hits'][0]['_source'].get('metadata'))
        else:
            st.error("ES 存储验证失败！")
    except Exception as e:
        st.error(f"验证 Elasticsearch 存储失败: {str(e)}")

    # 获取文件总数和列表
    try:
        count_response = es.count(index=ES_INDEX)
        st.write(f"文件总数: {count_response['count']}")
    except Exception as e:
        st.error(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {"query": {"match_all": {}}, "_source": ["metadata.source"], "size": 10000}
    try:
        search_response = es.search(index=ES_INDEX, body=search_body)
        hits = search_response['hits']['hits']
        seen_filenames = set()
        file_list = []
        for hit in hits:
            source = hit['_source'].get('metadata', {}).get('source', '')
            if source and (filename := os.path.basename(source)) not in seen_filenames:
                seen_filenames.add(filename)
                file_type = os.path.splitext(filename)[1][1:] or '未知'
                file_size = round(os.path.getsize(source) / (1024 * 1024), 2) if os.path.exists(source) else '未知'
                file_list.append([filename, file_type, f"{file_size:.2f}"])
        return file_list
    except Exception as e:
        st.error(f"搜索 Elasticsearch 失败: {str(e)}")
        return []


def load_file_info_from_es() -> List[List[str]]:
    """从 Elasticsearch 加载已存储的文件信息"""
    es = Elasticsearch(hosts=[ES_HOST])
    if not es.indices.exists(index=ES_INDEX):
        st.write(f"索引 '{ES_INDEX}' 不存在，未上传任何文件。")
        return []

    try:
        count_response = es.count(index=ES_INDEX)
        st.write(f"文件总数: {count_response['count']}")
    except Exception as e:
        st.error(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {"query": {"match_all": {}}, "_source": ["metadata.source"], "size": 10000}
    try:
        search_response = es.search(index=ES_INDEX, body=search_body)
        hits = search_response['hits']['hits']
        seen_filenames = set()
        file_list = []
        for hit in hits:
            source = hit['_source'].get('metadata', {}).get('source', '')
            if source and (filename := os.path.basename(source)) not in seen_filenames:
                seen_filenames.add(filename)
                file_type = os.path.splitext(filename)[1][1:] or '未知'
                file_size = round(os.path.getsize(source) / (1024 * 1024), 2) if os.path.exists(source) else '未知'
                file_list.append([filename, file_type, f"{file_size:.2f}"])
        return file_list
    except Exception as e:
        st.error(f"搜索 Elasticsearch 失败: {str(e)}")
        return []


# --------------------
# RAG 问答链
# --------------------

# --------------------
# RAG 问答链
# --------------------
def rag_chain(question: str):
    """RAG 问答链实现，分别生成 ES 和 FAISS 的回答"""
    st.write('14. 正在加载检索器...')
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(str(FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)
    es = Elasticsearch(hosts=[ES_HOST])

    # 自定义 Elasticsearch 检索器
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
                st.error(f"Elasticsearch 检索失败: {str(e)}")
                return []

    # 自定义 FAISS 检索器
    class CustomFAISSRetriever(BaseRetriever, BaseModel):
        vectorstore: FAISS = Field(...)
        score_threshold: float = Field(default=0.3)  # 降低阈值

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> List[Document]:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=20)
                # 调试：输出所有文档的得分
                st.write("FAISS 检索得分：", [(doc.page_content[:50], score) for doc, score in docs_with_scores])
                filtered_docs = [doc for doc, score in docs_with_scores if score >= self.score_threshold]
                if not filtered_docs and docs_with_scores:
                    # 如果没有文档满足阈值，返回得分最高的文档
                    st.write("没有文档满足阈值，返回得分最高的文档...")
                    filtered_docs = [docs_with_scores[0][0]]  # 返回得分最高的文档
                return filtered_docs
            except Exception as e:
                st.error(f"FAISS 检索失败: {str(e)}")
                return []

    # 初始化检索器
    es_retriever = CustomESRetriever(es=es, es_index=ES_INDEX)
    faiss_retriever = CustomFAISSRetriever(vectorstore=vector_db)

    # 分别检索 ES 和 FAISS
    st.write('15. 正在进行 Elasticsearch 检索...')
    es_docs, es_scores = retrieve_with_scores(es_retriever, question)
    st.write(f"Elasticsearch 检索结果数量: {len(es_docs)}")
    if es_docs:
        st.write(f"第一个 ES 文档: {es_docs[0]}")
    es_final_docs = [validate_metadata(Document(page_content=doc.page_content, metadata=doc.metadata)) for doc in es_docs]
    es_formatted_docs = format_documents(es_final_docs[:3])

    st.write('16. 正在进行 FAISS 检索...')
    faiss_docs, faiss_scores = retrieve_with_scores(faiss_retriever, question)
    st.write(f"FAISS 检索结果数量: {len(faiss_docs)}")
    if faiss_docs:
        st.write(f"第一个 FAISS 文档: {faiss_docs[0]}")
    faiss_final_docs = [validate_metadata(Document(page_content=doc.page_content, metadata=doc.metadata)) for doc in faiss_docs]
    faiss_formatted_docs = format_documents(faiss_final_docs[:3])

    # 生成 ES 的提示词并调用模型
    es_prompt = f"""你是一位资深的合同、标书文档专家，请根据以下内容回答：
                {es_formatted_docs}
                问题：{question}"""
    es_answer = ""
    try:
        es_response = session.post(
            f"{OLLAMA_HOST}/api/generate", json={"model": "deepseek-r1:14b", "prompt": es_prompt, "stream": True},
            timeout=120, stream=True
        )
        for line in es_response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                es_answer += chunk
                yield "es", es_answer
    except Exception as e:
        st.error(f"Elasticsearch 模型生成回答失败: {str(e)}")
        yield "es", "生成回答时发生错误，请稍后重试。"

    # 生成 FAISS 的提示词并调用模型
    if not faiss_formatted_docs:
        # 如果 FAISS 检索结果为空，提供默认回答
        yield "faiss", "FAISS 检索未找到相关文档，可能是查询与文档内容不匹配。请尝试调整问题或上传更多相关文档。"
    else:
        faiss_prompt = f"""你是一位资深的合同、标书文档专家，请根据以下内容回答：
                    {faiss_formatted_docs}
                    问题：{question}"""
        faiss_answer = ""
        try:
            faiss_response = session.post(
                f"{OLLAMA_HOST}/api/generate", json={"model": "deepseek-r1:14b", "prompt": faiss_prompt, "stream": True},
                timeout=120, stream=True
            )
            for line in faiss_response.iter_lines():
                if line:
                    chunk = json.loads(line.decode()).get("response", "")
                    faiss_answer += chunk
                    yield "faiss", faiss_answer
        except Exception as e:
            st.error(f"FAISS 模型生成回答失败: {str(e)}")
            yield "faiss", "生成回答时发生错误，请稍后重试。"



# --------------------
# Streamlit UI
# --------------------
def main():
    """Streamlit 主界面"""
    # 设置页面配置
    st.set_page_config(page_title="RAG (ES+FAISS+Deepseek-R1:14b)", layout="wide")

    # 自定义 CSS 样式
    st.markdown("""
        <style>
            .stApp {
                max-width: 1400px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #1e3a8a;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .section-title {
                font-size: 24px;
                font-weight: bold;
                color: #1e3a8a;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .info-box {
                background-color: #e0f7fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-size: 16px;
                color: #01579b;
            }
            .custom-button {
                background-color: #0288d1;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                text-align: center;
                display: inline-block;
                cursor: pointer;
            }
            .custom-button:hover {
                background-color: #0277bd;
            }
            .file-list {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .answer-box {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                min-height: 200px;
                margin-top: 15px;
            }
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                border: 1px solid #0288d1;
                border-radius: 5px;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # 页面标题
    st.markdown('<div class="header">RAG (ES+FAISS+Deepseek-R1:14b)</div>', unsafe_allow_html=True)

    # 整体布局：两列布局，左侧为操作区，右侧为展示区
    col1, col2 = st.columns([1, 2], gap="medium")

    # 左侧：操作区
    with col1:
        # 文件上传部分
        st.markdown('<div class="section-title">📂 文件上传</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">支持 DOCX 和 PDF 文件，上传后将存储到 ES/FAISS 库</div>', unsafe_allow_html=True)
        default_path = "/teamspace/studios/this_studio/elasticsearch-Faiss-Deepseek-R1-14B-RAG/AAA-database"
        file_path = st.text_input("输入文件夹路径", value=default_path, key="file_path_input")
        if st.button("上传文件", key="upload_btn", help="点击上传文件到 ES/FAISS"):
            if file_path:
                with st.spinner("正在处理文件，请稍候..."):
                    file_list = upload_files(file_path)
                    st.session_state['file_list'] = file_list
                    st.success("文件上传成功！")
            else:
                st.error("请提供有效的文件夹路径！")

        # 提问部分
        st.markdown('<div class="section-title">❓ 提问</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">请输入您的问题，系统将基于上传的文件生成回答</div>', unsafe_allow_html=True)
        question = st.text_area(
            "输入您的问题",
            placeholder="例如：本文档的主要观点是什么？",
            height=150,
            key="question_input"
        )
        if st.button("提交问题", key="ask_btn", help="点击提交问题"):
            if question and 'file_list' in st.session_state:
                with st.spinner("正在生成回答，请稍候..."):
                    # 创建两个独立的容器用于显示 ES 和 FAISS 的回答
                    es_answer_container = st.empty()
                    faiss_answer_container = st.empty()
                    counter = 0
                    for source, answer in rag_chain(question):
                        counter += 1
                        unique_key = f"answer_output_{source}_{int(time.time()*1000)}_{counter}"
                        if source == "es":
                            es_answer_container.text_area(
                                "Elasticsearch 回答",
                                value=answer,
                                height=200,
                                key=unique_key
                            )
                        elif source == "faiss":
                            faiss_answer_container.text_area(
                                "FAISS 回答",
                                value=answer,
                                height=200,
                                key=unique_key
                            )
            else:
                st.error("请先上传文件并输入问题！")

    # 右侧：展示区
    with col2:
        # 已存储文件列表
        st.markdown('<div class="section-title">📄 已存储文件列表</div>', unsafe_allow_html=True)
        if 'file_list' in st.session_state:
            st.dataframe(
                st.session_state['file_list'],
                column_config={"0": "文件名", "1": "文件类型", "2": "大小(MB)"},
                use_container_width=True,
                height=200
            )
        else:
            file_list = load_file_info_from_es()
            st.session_state['file_list'] = file_list
            st.dataframe(
                file_list,
                column_config={"0": "文件名", "1": "文件类型", "2": "大小(MB)"},
                use_container_width=True,
                height=200
            )

        # 答案展示
        st.markdown('<div class="section-title">📝 回答</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">回答生成可能需要1-2分钟，请耐心等待<br>支持多轮对话，可基于前文继续提问</div>', unsafe_allow_html=True)
        # 两个聊天框分别显示 ES 和 FAISS 的回答
        st.markdown('<div class="answer-box" id="es-answer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="answer-box" id="faiss-answer"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()