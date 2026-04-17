"""模型工厂：负责构建聊天模型与向量模型。"""

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

from config.config import (
    CHAT_MODEL_NAME,
    DASHSCOPE_API_KEY,
    EMBEDDING_MODEL_NAME,
)

def build_chat_model() -> ChatTongyi:
    """构建聊天模型实例。"""
    return ChatTongyi(
        model=CHAT_MODEL_NAME,
        temperature=0,
        api_key=DASHSCOPE_API_KEY,
    )


def build_embedding_model() -> DashScopeEmbeddings:
    """构建向量模型实例。"""
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
