from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def load_documents(directory: str) -> list[Document]:
    """加载 DOCX 和 PDF 文件"""
    documents = []
    
    # 加载 DOCX
    loader = DirectoryLoader(directory, glob="**/*.docx")
    documents.extend(loader.load())
    
    # 加载 PDF
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """智能分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n# ", "\n## ", "\n### ", "\n\n"]
    )
    return text_splitter.split_documents(documents)