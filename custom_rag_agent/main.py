"""RAG 项目入口：组装并导出 graph。"""

from functools import lru_cache
from typing import Any

BLOG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


@lru_cache(maxsize=1)
def build_app_graph():
    """构建应用级 graph，供外部直接导入使用。

    Args:
        无。

    Returns:
        CompiledStateGraph: 构建完成的 RAG 工作流图对象。
    """
    # 延迟导入，避免仅 import 模块时就触发环境变量校验与模型初始化。
    from custom_rag_agent.retrievers.web_retriever import build_retriever
    from custom_rag_agent.workflows.rag_graph import build_graph

    retriever = build_retriever(BLOG_URLS)
    return build_graph(retriever)


def get_graph():
    """懒加载 graph，首次调用时才构建检索器和工作流。"""
    return build_app_graph()


class LazyGraph:
    """延迟代理：兼容 `from ... import graph` 的使用方式。"""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_graph(), name)


graph = LazyGraph()
