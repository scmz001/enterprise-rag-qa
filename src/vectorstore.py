"""向量库封装：把切分好的文本块存入 ChromaDB，并对外提供检索接口。

要点：
    - 使用本地持久化模式（storage/chroma/），不需要单独启动任何服务。
    - 向量化走 LM Studio 的 OpenAI 兼容接口，数据不出本机。
    - check_embedding_ctx_length=False 是关键：LangChain 默认会先用 tiktoken
      把文本切成 token 数组再发送，而 LM Studio 只接受纯字符串，不关掉会报错。
"""

from __future__ import annotations

from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config


def get_embeddings() -> OpenAIEmbeddings:
    """构造指向 LM Studio 的向量化模型。"""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.LM_STUDIO_BASE_URL,
        api_key=config.LM_STUDIO_API_KEY,  # LM Studio 不校验，占位值即可
        check_embedding_ctx_length=False,
    )


def get_vectorstore() -> Chroma:
    """打开（不存在则创建）本地向量库。"""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # 用余弦相似度衡量语义接近程度
    )


def reset_collection() -> None:
    """删除整个集合，用于全量重建，避免重复入库产生冗余数据。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        # 集合本来就不存在，属于正常情况
        pass


def index_documents(docs: list[Document], batch_size: int = 64) -> int:
    """把文本块写入向量库，分批提交并打印进度。返回实际写入数量。"""
    if not docs:
        raise ValueError("没有可入库的文本块，请先检查 data/docs/ 是否有文档")

    store = get_vectorstore()
    total = len(docs)
    for start in range(0, total, batch_size):
        batch = docs[start : start + batch_size]
        store.add_documents(batch)
        done = min(start + batch_size, total)
        print(f"  正在向量化并写入：{done}/{total}")
    return total


def collection_stats() -> dict[str, object]:
    """统计向量库现状，用于确认数据是否真的入库。"""
    import chromadb

    if not config.CHROMA_DIR.exists():
        return {"exists": False, "count": 0, "sources": {}}

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    names = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME not in names:
        return {"exists": True, "collection": False, "count": 0, "sources": {}}

    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["metadatas"])
    sources = Counter((m or {}).get("source", "未知") for m in data["metadatas"])
    return {
        "exists": True,
        "collection": True,
        "count": collection.count(),
        "sources": dict(sorted(sources.items())),
    }


def print_stats(sample_size: int = 2) -> None:
    """终端友好地打印向量库现状，并抽样展示库中真实内容。"""
    stats = collection_stats()
    print("\n【向量库现状】")
    if not stats.get("collection"):
        print("  向量库为空，尚未入库。请运行：python -m src.ingest")
        return
    print(f"  集合名：{config.COLLECTION_NAME}")
    print(f"  文本块总数：{stats['count']}")
    print("  各文档块数：")
    for name, n in stats["sources"].items():
        print(f"    - {name:<32} {n:>3} 块")

    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"], limit=sample_size)
    if data["documents"]:
        print("\n  入库内容抽样（确认数据真实存在）：")
        for text, meta in zip(data["documents"], data["metadatas"]):
            meta = meta or {}
            loc = meta.get("section", "?")
            if meta.get("page"):
                loc += f"（第 {meta['page']} 页）"
            body = " ".join(text.split())
            print(f"    ── {meta.get('source', '?')} ｜ {loc}")
            print(f"       {body[:100]}{'…' if len(body) > 100 else ''}")


def similarity_search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """按语义相似度检索，返回（文本块, 距离）列表，距离越小越相关。"""
    store = get_vectorstore()
    k = k or config.TOP_K
    return store.similarity_search_with_score(query, k=k)
