from langchain_core.documents import Document

def format_documents(docs: list[Document]) -> str:
    """格式化文档输出"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", 0)
        content = doc.page_content.replace("\n", " ").strip()[:300]
        formatted.append(f"【文档{idx}】\n来源: {source}\n页码: {page}\n内容: {content}...")
    return "\n\n".join(formatted)