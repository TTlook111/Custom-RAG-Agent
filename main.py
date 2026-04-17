"""RAG 项目入口：组装并导出 graph。"""

import sys
from pathlib import Path

PROJECT_SRC_DIR = Path(__file__).resolve().parent / "custom-rag-agent"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from models.model_factory import build_chat_model
from retrievers.web_retriever import build_retriever
from sources.blog_urls import BLOG_URLS
from tools.retrieval_tools import build_retriever_tool
from workflows.rag_graph import build_graph


def build_app_graph():
    """构建应用级 graph，供外部直接导入使用。

    Args:
        无。

    Returns:
        CompiledStateGraph: 构建完成的 RAG 工作流图对象。
    """
    retriever = build_retriever(BLOG_URLS)
    retriever_tool = build_retriever_tool(retriever)
    response_model = build_chat_model()
    grader_model = build_chat_model()
    return build_graph(
        response_model=response_model,
        grader_model=grader_model,
        retriever_tool=retriever_tool,
    )


graph = build_app_graph()
