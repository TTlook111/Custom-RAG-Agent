"""RAG 项目入口：组装并导出 graph。"""

import sys
from pathlib import Path

PROJECT_SRC_DIR = Path(__file__).resolve().parent / "custom-rag-agent"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from retrievers.web_retriever import build_retriever
from workflows.rag_graph import build_graph

BLOG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def build_app_graph():
    """构建应用级 graph，供外部直接导入使用。

    Args:
        无。

    Returns:
        CompiledStateGraph: 构建完成的 RAG 工作流图对象。
    """
    retriever = build_retriever(BLOG_URLS)
    return build_graph(retriever)


graph = build_app_graph()
