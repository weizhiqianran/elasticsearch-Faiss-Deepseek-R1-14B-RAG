# -*- coding: utf-8 -*-
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS, ElasticsearchStore
from elasticsearch import Elasticsearch
from config import *
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field,field_validator
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
    documents=[]
    # 遍历文件夹中的所有 PDF 文件
    for filename in os.listdir(pdfpath):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdfpath, filename)
            #print("pdf_path:",pdf_path)
            # 读取 PDF 文件
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            #print("pdf document:", document)
            #documents.append(document)
            documents.extend(document)
    return documents

def upload_files():
    try:
       root = tk.Tk()
       # 隐藏根窗口
       root.withdraw()
       # 确保窗口置顶
       root.attributes("-topmost", True)
       # 打开文件夹选择对话框
       file_path = filedialog.askdirectory()
       # 销毁 tkinter 根窗口
       root.destroy()
    except Exception as e:
        print(f"读取文件错误: {e}")

    """数据预处理与存储"""
    print('1.开始执行数据预处理与存储步骤...')
    file_info = []

    # 处理docx文件
    print('2.处理docx文件...',file_path)
    loader = DirectoryLoader(file_path, glob="**/*.docx")
    documents1 = loader.load()
    print(f'3.成功加载 {len(documents1)} 个文档。')
    file_info1 = get_docx_info(file_path,'docx')
    #print(documents1)
    print('处理pdf文件')
    #处理pdf文件
    documents2 = load_pdf(file_path)
    file_info2 =  get_docx_info(file_path, 'pdf')
    #print(documents2)

    #pdf，docx合并一起处理
    file_info = file_info1 + file_info2
    documents = documents1+documents2

    print(file_info)
    # 智能分块
    print('4.正在进行智能分块...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n# ", "\n## ", "\n### ", "\n\n"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f'5.分块完成，共 {len(chunks)} 个文本块。')

    # 初始化嵌入模型
    print('6.正在初始化向量化模型...')
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 存储到FAISS
    print('7.正在将文本块存储到FAISS...')
    faiss_chunks = [
            Document(
                page_content=chunk.page_content,
                metadata={
            ** chunk.metadata,
            "retriever_source": "faiss"  # 明确添加来源标记
            }
        ) for chunk in chunks
    ]
    vectorstore = FAISS.from_documents(faiss_chunks, embeddings,distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vectorstore.save_local(str(FAISS_INDEX))
    print('8.FAISS存储完成。')

    # 元数据完整性检查
    print("检查分块元数据示例：")
    if len(chunks) > 0:
        sample_chunk = chunks[0]
        print(f"内容长度: {len(sample_chunk.page_content)}")
        print(f"元数据: {sample_chunk.metadata}")

    # 存储到Elasticsearch
    print('9.正在将文本块存入Elasticsearch...')
    es = Elasticsearch(hosts=[ES_HOST])

    # 自定义索引设置，使用更适合中文的分词器
    index_settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "my_ik_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "my_ik_analyzer",
                    "search_analyzer": "my_ik_analyzer"
                }
            }
        }
    }

    #删除索引
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX, body=index_settings)

    # 使用 Elasticsearch.options() 设置选项
    es_with_options = es.options(request_timeout=10)

    # 调用 API 方法
    es_with_options.indices.create(
        index=ES_INDEX,
        body=index_settings,
        ignore=400
    )

    es_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={
                ** chunk.metadata,
                "retriever_source": "es"  # 明确添加来源标记
                }
            ) for chunk in chunks
    ]
    ElasticsearchStore.from_documents(
        #documents=es_chunks,
        documents=[
            Document(
                        page_content=chunk.page_content,
                        metadata={
             ** dict(chunk.metadata),  # 确保转换为字典
            "retriever_source": "es"
            }
            ) for chunk in es_chunks
            ],
        embedding=embeddings,
        es_connection=es,
        index_name=ES_INDEX,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(
            hybrid=True,
            query_model_id="sentence-transformers/all-MiniLM-L6-v2"  # all-mpnet-base-v2
        )
    )

    print('10.成功存入Elasticsearch。')

    print('11.验证Elasticsearch存储...')
    test_query = {"query": {"match_all": {}}, "size": 1}
    test_result = es.search(index=ES_INDEX, body=test_query)
    if test_result['hits']['total']['value'] > 0:
        print("ES存储验证成功，首条记录元数据:", test_result['hits']['hits'][0]['_source'].get('metadata'))
    else:
        print("ES存储验证失败！")

    # 查询已保存文件的数量
    count_response = es.count(index=ES_INDEX)
    file_count = count_response['count']
    print(file_count)
    # 查询文件名、文件大小和文件类型
    search_body = {
        "query": {
            "match_all": {}
        },
        "_source": ["metadata.source"],
        "size": 10000
    }
    search_response = es.search(index=ES_INDEX, body=search_body)
    hits = search_response['hits']['hits']
    seen_filenames = set()
    fileList=[]
    for hit in hits:
        source =  hit['_source'].get('metadata', {}).get('source', '')
        #print('source',source)
        if source:
            # 获取文件名
            filename = os.path.basename(source)
            #print('filename', filename)
            if filename in seen_filenames:
                # 如果文件名已经出现过，则跳过
                continue
            # 将文件名添加到已见集合中
            seen_filenames.add(filename)
            # 获取文件类型
            _, file_extension = os.path.splitext(filename)
            file_type = file_extension[1:] if file_extension else '未知'
            try:
                # 获取文件大小（假设文件在本地存在）
                file_size = os.path.getsize(source) / (1024 * 1024)
                file_size = round(file_size, 2)
            except FileNotFoundError:
                file_size = '未知'
            print(f"文件名: {filename}, 文件类型: {file_type}, 文件大小: {file_size} MB")
            info = [
                str(filename),
                str(file_type),
                f"{file_size:.2f}"
            ]
            # 将文件信息添加到列表中
            fileList.append(info)
        else:
            print("未找到有效的文件路径信息")
    #print(fileList)
    return fileList

def Load_file_info_FrmES():
    es = Elasticsearch(hosts=[ES_HOST])
    count_response = es.count(index=ES_INDEX)
    file_count = count_response['count']

    # 查询文件名、文件大小和文件类型
    search_body = {
        "query": {
            "match_all": {}
        },
        "_source": ["metadata.source"],
        "size": 10000
    }
    search_response = es.search(index=ES_INDEX, body=search_body)
    hits = search_response['hits']['hits']
    seen_filenames = set()
    fileList = []
    for hit in hits:
        source = hit['_source'].get('metadata', {}).get('source', '')
        if source:
            # 获取文件名
            filename = os.path.basename(source)
            if filename in seen_filenames:
                # 如果文件名已经出现过，则跳过
                continue
            # 将文件名添加到已见集合中
            seen_filenames.add(filename)
            # 获取文件类型
            _, file_extension = os.path.splitext(filename)
            file_type = file_extension[1:] if file_extension else '未知'
            try:
                # 获取文件大小（假设文件在本地存在）
                file_size = os.path.getsize(source) / (1024 * 1024)
                file_size = round(file_size, 2)
            except FileNotFoundError:
                file_size = '未知'
            print(f"文件名: {filename}, 文件类型: {file_type}, 文件大小: {file_size} MB")
            info = [
                str(filename),
                str(file_type),
                f"{file_size:.2f}"
            ]
            # 将文件信息添加到列表中
            fileList.append(info)
        else:
            print("未找到有效的文件路径信息")

    return fileList

def rag_chain(question):
    """构建RAG链"""
    # 加载检索器
    print('13.正在加载检索器...')
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(str(FAISS_INDEX), embeddings, allow_dangerous_deserialization=True)
    es = Elasticsearch(hosts=[ES_HOST])

    # 构建ES检索器（直接使用Elasticsearch查询）
    class CustomESRetriever(BaseRetriever, BaseModel):
        es: Elasticsearch = Field(...)
        es_index: str = Field(...)
        k: int = Field(default=15)

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str) -> List[Document]:
            body = {
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                            "analyzer": "my_ik_analyzer"
                        }
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

    es_retriever = CustomESRetriever(
        es=es,
        es_index=ES_INDEX
    )
    class CustomFAISSRetriever(BaseRetriever, BaseModel):
        vectorstore: FAISS = Field(...)
        score_threshold: float = Field(default=0.5)

        class Config:
            arbitrary_types_allowed = True  # 允许非Pydantic类型

        def _get_relevant_documents(self, query: str) -> List[Document]:
            """带分数过滤的检索方法"""
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=20)
            return [doc for doc, score in docs_with_scores if score >= self.score_threshold]

    faiss_retriever = CustomFAISSRetriever(
        vectorstore=vector_db,  # 使用关键字参数
        score_threshold=0.5
    )


    #question = query["question"]
    print('并行检索')
    # 并行检索
    #es_docs=[]
    #es_scores =[]# retrieve_with_scores(es_retriever, question)
    es_docs, es_scores = retrieve_with_scores(es_retriever, question)
    faiss_docs, faiss_scores = retrieve_with_scores(faiss_retriever, question)

    print('合并查询结果')
    #合并查询结果
    merged_results = merge_results(
        es_docs=es_docs,
        faiss_docs=faiss_docs,
        es_scores=es_scores,
        faiss_scores=faiss_scores,
        es_weight=0.4,  # 可调整权重
        faiss_weight=0.6,  # 可调整权重
        merge_strategy="reciprocal_rank"
    )
    print(merged_results)

    # 格式转换（适配后续处理）
    final_docs = [
        validate_metadata(
            Document(
                page_content=res.content,
                metadata=res.metadata
            )
        ) for res in merged_results
    ]

    print('合并后结果:')
    print(final_docs)

    formatted_docs = format_documents(final_docs[:3])
    prompt = f"""你是一位资深的合同、标书文档专家，请根据以下内容回答：
                {formatted_docs}

                问题：{question}
                """
    # 流式请求
    print('模型开始问答 prompt:', prompt)
    response = session.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:14b",
            "prompt": prompt,
            "stream": True  # 启用流式
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

# 修改界面布局部分
with gr.Blocks(
        title="hi,GBE今天怎么样",
        css="""
                .gradio-container {max-width: 2000px !important}
                .answer-box {min-height: 500px !important;}
                .left-panel {padding-right: 20px; border-right: 1px solid #eee;}
                .right-panel {height: 100vh;}
                .wide-row { width: 80%; }
                .green-button {
                    background-color: green;
                    color: white; 
                }
                .blue-button {
                    background-color: #ADD8E6;
                    color: white; 
                }
               .gradio-label {
                    font-size: 8px !important;
                    font-weight: normal !important;
                }
                .gradio-container input {
                    font-size: 8px !important;
                }
                .gradio-container textbox {
                    font-size: 8px !important;
                }
                .gray-background textarea, .gray-background input[type="text"] {
                background-color: #cccccc !important;
                }      
                .large-font {
                    font-size: 72px !important;
                }     
                .bold-font {
                    font-weight: bold !important;
                }
                .center-text {
                    text-align: center !important;
                }
                .red-text {
                    color: red !important;
                }   
                """
) as demo:
    gr.Markdown("## RAG.(ES+FAISS+Deepseek-R1:14b).",  elem_classes="blue-button large-font font-weight center-text")

    with gr.Row():
        # 左侧操作面板
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
                ask_btn = gr.Button("🔍 开始提问", variant="primary",elem_classes="green-button")
                #ask_btn222 = gr.Button("🔍 开始提问2222222", variant="primary")

        # 右侧答案显示区
        with gr.Column(scale=3, elem_classes="right-panel"):
            gr.Markdown("## 已经存储文件列表 ")
            file_df = gr.Dataframe(
                headers=["文件名" , "文件类型", "大小(MB)"],  # 表格列名
                datatype=["str", "str", "str"],  # 数据类型
                interactive=False,  # 不可交互编辑
                elem_classes="File_list-box",
                show_copy_button=True
            )

            gr.Markdown("## 📝 答案",elem_classes="blue-button")
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

    # 调整后的加载提示
    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>🔄 系统初始化中，请稍候...</h3>
    </div>
    """)

    # 进度显示组件调整到左侧面板下方
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span>当前进度：</span>
            <span id="current-step" style="color: #2b6de3;">初始化...</span>
            <span id="progress-percent" style="margin-left:15px;color: #e32b2b;">0%</span>
        </div>
        """)

    # 在界面组件定义之后添加按钮事件
    ask_btn.click(
        fn=rag_chain,
        inputs=question_input,
        outputs=[answer_output],
        show_progress="hidden"
    )

    upload_btn.click(
        fn=upload_files,
        inputs=None,
        outputs=file_df
    )

    # 在页面加载时显示 ES 文件列表
    demo.load(
        fn=lambda: Load_file_info_FrmES(),
        inputs=None,
        outputs=file_df
    )
# 修改JavaScript注入部分为兼容写法
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

# 修改端口检查函数
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0  # 更可靠的检测方式

def check_environment():
    """环境依赖检查"""
    try:
        # 添加模型存在性检查
        model_check = session.post(
            "http://localhost:11434/api/show",
            json={"name": "deepseek-r1:14b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:7b")
            return False

        # 原有检查保持不变...
        response = session.get(
            "http://localhost:11434/api/tags",
            proxies={"http": None, "https": None},  # 禁用代理
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
        ollama_check = session.get("http://localhost:11434", timeout=5)
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