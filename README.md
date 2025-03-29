使用elasticsearch+Faiss+Deepseek R1 14B构建RAG

一、环境及准备:

1、下载lasticsearch-8.17.3，并安装
https://www.elastic.co/downloads/past-releases/elasticsearch-8-17-3

安装后打开config/jvm.options根据实际需要修改
-Xms24g
-Xmx24g

运行bin/elasticsearch.bat

验证服务启动是否成功
C:\Users\Administrator>curl -X GET "localhost:9200/"
{
  "name" : "DESKTOP-EEGR6L5",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "hU1-3T1OReCF6rbcFGGuDQ",
  "version" : {
    "number" : "8.17.3",
    "build_flavor" : "default",
    "build_type" : "zip",
    "build_hash" : "a091390de485bd4b127884f7e565c0cad59b10d2",
    "build_date" : "2025-02-28T10:07:26.089129809Z",
    "build_snapshot" : false,
    "lucene_version" : "9.12.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}

2、下载ollama安装
安装之后下载Deepseek R1 14B
运行启动服务：ollama serve &
C:\Users\Administrator>ollama serve &

3、环境安装，很重要
pip install -U langchain langchain-community langchain-core langchain-ollama transformers elasticsearch
conda install -c conda-forge faiss-gpu
pip install "sentence-transformers==2.3.0" "torch==2.5.1+cu118" --extra-index-url https://download.pytorch.org/whl/cu118
pip install rank_bm25
pip install numpy==1.26.4
pip install unstructured
pip install "unstructured[docx]"

二、rag系统搭建问题
1、环境依赖比较复杂，各种版本会冲突
2、es和faiss混合查询合并比较麻烦，需要慢慢调试
3、各种组件，服务的特性需要研究
4、大模型R1只是整个链条中的一个环节，不是最重要的，最重要的是前面的内容检索
精准的检索能才能发挥R1的作用
5、最困难点：各种不同的docx和pdf文档结构比较复杂，难以通用化处理
