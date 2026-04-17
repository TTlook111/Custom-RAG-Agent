"""检索器构建模块。"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.model_factory import build_embedding_model


def build_retriever(urls: list[str]) -> BaseRetriever:
    """基于网页地址列表构建检索器。

    Args:
        urls: 需要加载并建立索引的网页地址列表。

    Returns:
        BaseRetriever: 构建完成的检索器对象，可直接用于相似度检索。
    """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    # 先对文档进行切分，避免单段文本过长影响向量检索效果。
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=50,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    # 将切分后的文档写入内存向量库，并转换成检索器接口。
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=build_embedding_model(),
    )
    return vectorstore.as_retriever()
