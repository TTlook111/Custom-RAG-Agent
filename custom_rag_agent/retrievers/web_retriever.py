"""检索器构建模块。"""

import hashlib
import json
import logging
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from custom_rag_agent.config.config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_MODEL_NAME,
    PROJECT_ROOT_DIR,
)

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - 运行时依赖缺失的兜底分支
    FAISS = None

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _build_cache_key(urls: list[str]) -> str:
    """基于关键参数生成向量缓存键。"""
    payload = {
        "urls": sorted(urls),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _load_url_docs(urls: list[str]) -> list:
    """逐个加载 URL，容错处理失败链接。"""
    docs_list = []
    failed_urls: list[str] = []
    for url in urls:
        try:
            docs_list.extend(WebBaseLoader(url).load())
        except Exception as exc:  # noqa: BLE001
            failed_urls.append(url)
            logger.warning("加载 URL 失败: %s, error=%s", url, exc)

    if failed_urls:
        logger.warning("以下 URL 加载失败，已跳过: %s", failed_urls)
    if not docs_list:
        raise ValueError("所有 URL 均加载失败，无法构建检索器。")
    return docs_list


def build_retriever(urls: list[str]) -> BaseRetriever:
    """基于网页地址列表构建检索器。

    Args:
        urls: 需要加载并建立索引的网页地址列表。

    Returns:
        BaseRetriever: 构建完成的检索器对象，可直接用于相似度检索。
    """
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    cache_key = _build_cache_key(urls)
    cache_dir = Path(PROJECT_ROOT_DIR) / ".cache" / "faiss" / cache_key

    if FAISS is not None and (cache_dir / "index.faiss").exists() and (cache_dir / "index.pkl").exists():
        vectorstore = FAISS.load_local(
            folder_path=str(cache_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("命中本地向量缓存: %s", cache_dir)
        return vectorstore.as_retriever()

    docs_list = _load_url_docs(urls)
    # 先对文档进行切分，避免单段文本过长影响向量检索效果。
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 优先使用可持久化向量库；若 FAISS 不可用，则退化为内存向量库。
    if FAISS is not None:
        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embeddings)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        logger.info("已写入本地向量缓存: %s", cache_dir)
    else:
        logger.warning("未安装 FAISS，当前使用内存向量库（重启后需重建索引）。")
        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=embeddings,
        )

    return vectorstore.as_retriever()
