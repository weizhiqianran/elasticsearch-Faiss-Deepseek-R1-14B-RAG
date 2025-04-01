# -*- coding: utf-8 -*-
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from elasticsearch import Elasticsearch
from config import *  # 导入 OLLAMA_HOST 和其他配置
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import hashlib
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gradio as gr
import os
import socket
import webbrowser
from pdfminer.high_level import extract_text_to_fp
from io import StringIO
import tkinter as tk
from tkinter import filedialog
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from assistant_fun import *

requests.adapters.DEFAULT_RETRIES = 3  # 增加重试次数
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', HTTPAdapter(max_retries=retries))

def load_pdf(pdfpath):
    documents = []
    for filename in os.listdir(pdfpath):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdfpath, filename)
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            documents.extend(document)
    return documents

def upload_files():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askdirectory()
        root.destroy()
    except Exception as e:
        print(f"读取文件错误: {e}")
        return []

    print('1.开始执行数据预处理与存储步骤...')
    file_info = []

    # 初始化 Elasticsearch 连接
    print('2.初始化 Elasticsearch 并创建索引...')
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

    # 检查索引是否存在，若存在则删除并重新创建
    try:
        if es.indices.exists(index=ES_INDEX):
            print(f"索引 '{ES_INDEX}' 已存在，正在删除...")
            es.indices.delete(index=ES_INDEX)
        es.indices.create(index=ES_INDEX, body=index_settings)
        print(f"成功创建索引 '{ES_INDEX}'")
    except Exception as e:
        print(f"创建索引失败: {str(e)}")
        return []

    print('3.处理docx文件...', file_path)
    loader = DirectoryLoader(file_path, glob="**/*.docx")
    documents1 = loader.load()
    print(f'4.成功加载 {len(documents1)} 个文档。')
    file_info1 = get_docx_info(file_path, 'docx')
    print('处理pdf文件')
    documents2 = load_pdf(file_path)
    file_info2 = get_docx_info(file_path, 'pdf')

    file_info = file_info1 + file_info2
    documents = documents1 + documents2

    print(file_info)
    print('5.正在进行智能分块...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n# ", "\n## ", "\n### ", "\n\n"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f'6.分块完成，共 {len(chunks)} 个文本块。')

    print('7.正在初始化向量化模型...')
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print('8.正在将文本块存储到FAISS...')
    faiss_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "retriever_source": "faiss"}
        ) for chunk in chunks
    ]
    vectorstore = FAISS.from_documents(faiss_chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vectorstore.save_local(str(FAISS_INDEX))
    print('9.FAISS存储完成。')

    print("检查分块元数据示例：")
    if len(chunks) > 0:
        sample_chunk = chunks[0]
        print(f"内容长度: {len(sample_chunk.page_content)}")
        print(f"元数据: {sample_chunk.metadata}")

    print('10.正在将文本块存入Elasticsearch...')
    es_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "retriever_source": "es"}
        ) for chunk in chunks
    ]
    try:
        ElasticsearchStore.from_documents(
            documents=es_chunks,
            embedding=embeddings,
            es_connection=es,
            index_name=ES_INDEX,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=True,
                query_model_id="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        print('11.成功存入Elasticsearch。')
    except Exception as e:
        print(f"存储到Elasticsearch失败: {str(e)}")
        return []

    print('12.验证Elasticsearch存储...')
    try:
        test_query = {"query": {"match_all": {}}, "size": 1}
        test_result = es.search(index=ES_INDEX, body=test_query)
        if test_result['hits']['total']['value'] > 0:
            print("ES存储验证成功，首条记录元数据:", test_result['hits']['hits'][0]['_source'].get('metadata'))
        else:
            print("ES存储验证失败！")
    except Exception as e:
        print(f"验证Elasticsearch存储失败: {str(e)}")

    try:
        count_response = es.count(index=ES_INDEX)
        file_count = count_response['count']
        print(f"文件总数: {file_count}")
    except Exception as e:
        print(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {
        "query": {"match_all": {}},
        "_source": ["metadata.source"],
        "size": 10000
    }
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
                print(f"文件名: {filename}, 文件类型: {file_type}, 文件大小: {file_size} MB")
                info = [str(filename), str(file_type), f"{file_size:.2f}"]
                fileList.append(info)
        return fileList
    except Exception as e:
        print(f"搜索Elasticsearch失败: {str(e)}")
        return []

def Load_file_info_FrmES():
    es = Elasticsearch(hosts=[ES_HOST])
    
    # 检查索引是否存在
    try:
        if not es.indices.exists(index=ES_INDEX):
            print(f"索引 '{ES_INDEX}' 不存在，未上传任何文件。")
            return []
    except Exception as e:
        print(f"检查索引存在性失败: {str(e)}")
        return []

    # 获取文件总数
    try:
        count_response = es.count(index=ES_INDEX)
        file_count = count_response['count']
    except Exception as e:
        print(f"获取文件总数失败: {str(e)}")
        return []

    search_body = {
        "query": {"match_all": {}},
        "_source": ["metadata.source"],
        "size": 10000
    }
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
                print(f"文件名: {filename}, 文件类型: {file_type}, 文件大小: {file_size} MB")
                info = [str(filename), str(file_type), f"{file_size:.2f}"]
                fileList.append(info)
        return fileList
    except Exception as e:
        print(f"搜索Elasticsearch失败: {str(e)}")
        return []

def rag_chain(question):
    print('13.正在加载检索器...')
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
                body = {
                    "query": {
                        "match": {
                            "text": {"query": query, "analyzer": "my_ik_analyzer"}
                        }
                    },
                    "size": self.k
                }
                response = self.es.search(index=self.es_index, body=body)
                return [
                    Document(
                        page_content=hit["_source"]["text"],
                        metadata=hit["_source"].get("metadata", {})
                    ) for hit in response['hits']['hits']
                ]
            except Exception as e:
                print(f"Elasticsearch检索失败: {str(e)}")
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
                print(f"FAISS检索失败: {str(e)}")
                return []

    faiss_retriever = CustomFAISSRetriever(vectorstore=vector_db, score_threshold=0.5)

    print('并行检索')
    es_docs, es_scores = retrieve_with_scores(es_retriever, question)
    faiss_docs, faiss_scores = retrieve_with_scores(faiss_retriever, question)

    print('合并查询结果')
    merged_results = merge_results(
        es_docs=es_docs,
        faiss_docs=faiss_docs,
        es_scores=es_scores,
        faiss_scores=faiss_scores,
        es_weight=0.4,
        faiss_weight=0.6,
        merge_strategy="reciprocal_rank"
    )
    print(merged_results)

    final_docs = [
        validate_metadata(Document(page_content=res.content, metadata=res.metadata))
        for res in merged_results
    ]

    print('合并后结果:')
    print(final_docs)

    formatted_docs = format_documents(final_docs[:3])
    prompt = f"""你是一位资深的合同、标书文档专家，请根据以下内容回答：
                {formatted_docs}

                问题：{question}
                """
    print('模型开始问答 prompt:', prompt)
    try:
        response = session.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "deepseek-r1:14b",
                "prompt": prompt,
                "stream": True
            },
            timeout=120,
            stream=True
        )
        full_answer = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                full_answer += chunk
                yield full_answer
    except Exception as e:
        print(f"模型生成回答失败: {str(e)}")
        yield "生成回答时发生错误，请稍后重试。"

# Gradio UI 和其他部分保持不变，仅更新核心函数
with gr.Blocks(
    title="hi,GBE今天怎么样",
    css="""
        .gradio-container {max-width: 2000px !important}
        .answer-box {min-height: 500px !important;}
        .left-panel {padding-right: 20px; border-right: 1px solid #eee;}
        .right-panel {height: 100vh;}
        .wide-row { width: 80%; }
        .green-button {background-color: green; color: white;}
        .blue-button {background-color: #ADD8E6; color: white;}
        .gradio-label {font-size: 8px !important; font-weight: normal !important;}
        .gradio-container input {font-size: 8px !important;}
        .gradio-container textbox {font-size: 8px !important;}
        .gray-background textarea, .gray-background input[type="text"] {background-color: #cccccc !important;}
        .large-font {font-size: 72px !important;}
        .bold-font {font-weight: bold !important;}
        .center-text {text-align: center !important;}
        .red-text {color: red !important;}
    """
) as demo:
    gr.Markdown("## RAG.(ES+FAISS+Deepseek-R1:14b).", elem_classes="blue-button large-font font-weight center-text")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="left-panel"):
            gr.Markdown("## 📂 DOCX/PDF文件路径")
            with gr.Group():
                upload_btn = gr.Button("文件存储到ES/FAISS库", variant="primary")
                upload_status = gr.Textbox(label="处理状态", interactive=False)

            gr.Markdown("## ❓ 提问")
            with gr.Group():
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=4,
                    placeholder="例如：本文档的主要观点是什么？",
                    elem_id="question-input"
                )
                ask_btn = gr.Button("🔍 开始提问", variant="primary", elem_classes="green-button")

        with gr.Column(scale=3, elem_classes="right-panel"):
            gr.Markdown("## 已经存储文件列表 ")
            file_df = gr.Dataframe(
                headers=["文件名", "文件类型", "大小(MB)"],
                datatype=["str", "str", "str"],
                interactive=False,
                elem_classes="File_list-box",
                show_copy_button=True
            )

            gr.Markdown("## 📝 答案", elem_classes="blue-button")
            answer_output = gr.Textbox(
                label="回答",
                interactive=False,
                lines=25,
                elem_classes="answer-box",
                autoscroll=True,
                show_copy_button=True
            )
            gr.Markdown("""
            <div class="footer-note">
                *回答生成可能需要1-2分钟，请耐心等待<br>
                *支持多轮对话，可基于前文继续提问
            </div>
            """)

    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>🔄 系统初始化中，请稍候...</h3>
    </div>
    """)

    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span>当前进度：</span>
            <span id="current-step" style="color: #2b6de3;">初始化...</span>
            <span id="progress-percent" style="margin-left:15px;color: #e32b2b;">0%</span>
        </div>
        """)

    ask_btn.click(fn=rag_chain, inputs=question_input, outputs=[answer_output], show_progress="hidden")
    upload_btn.click(fn=upload_files, inputs=None, outputs=file_df)
    demo.load(fn=lambda: Load_file_info_FrmES(), inputs=None, outputs=file_df)

demo._js = """
function gradioApp() {
    const observer = new MutationObserver((mutations) => {
        document.getElementById("loading").style.display = "none";
        const progress = document.querySelector('.progress-text');
        if (progress) {
            const percent = document.querySelector('.progress > div')?.innerText || '';
            const step = document.querySelector('.progress-description')?.innerText || '';
            document.getElementById('current-step').innerText = step;
            document.getElementById('progress-percent').innerText = percent;
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
}
"""

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0

def check_environment():
    try:
        model_check = session.post(
            f"{OLLAMA_HOST}/api/show",
            json={"name": "deepseek-r1:14b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:7b")
            return False

        response = session.get(
            f"{OLLAMA_HOST}/api/tags",
            proxies={"http": None, "https": None},
            timeout=5
        )
        if response.status_code != 200:
            print("Ollama服务异常，返回状态码:", response.status_code)
            return False
        return True
    except Exception as e:
        print("Ollama连接失败:", str(e))
        return False

if __name__ == "__main__":
    if not check_environment():
        exit(1)
    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)

    try:
        ollama_check = session.get(OLLAMA_HOST, timeout=5)
        if ollama_check.status_code != 200:
            print("Ollama服务未正常启动！")
            print("请先执行：ollama serve 启动服务")
            exit(1)

        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")