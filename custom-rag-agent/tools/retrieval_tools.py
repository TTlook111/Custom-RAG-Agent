"""Tool 封装模块。"""

from langchain.tools import tool
from langchain_core.retrievers import BaseRetriever


def build_retriever_tool(retriever: BaseRetriever):
    """创建检索工具。"""

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve_blog_posts
