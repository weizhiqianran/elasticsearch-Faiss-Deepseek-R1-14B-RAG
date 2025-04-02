import json
from typing import Any, Dict
from langchain_core.documents import Document

def safe_get_metadata(doc: Document) -> Dict[str, Any]:
    """安全获取元数据"""
    meta = getattr(doc, 'metadata', {})
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except:
            return {"source": meta}
    return meta or {"source": "unknown"}