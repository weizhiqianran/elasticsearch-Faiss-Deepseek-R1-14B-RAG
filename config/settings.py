from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DOC_DIR = BASE_DIR / "bid_docs"
FAISS_INDEX = BASE_DIR / "faiss_index"

# Elasticsearch 配置
ES_HOST = "http://localhost:9200"
ES_INDEX = "bid_docs"

# 模型配置
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
OLLAMA_MODEL = "deepseek-r1:14b"
OLLAMA_HOST = "http://localhost:11434"