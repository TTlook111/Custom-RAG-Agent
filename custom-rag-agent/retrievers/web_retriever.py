"""检索器构建模块。"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.model_factory import build_embedding_model


def build_retriever(urls: list[str]) -> BaseRetriever:
    """基于网页列表构建检索器。"""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=50,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=build_embedding_model(),
    )
    return vectorstore.as_retriever()
