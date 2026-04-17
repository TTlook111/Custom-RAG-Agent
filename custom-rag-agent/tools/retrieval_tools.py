"""检索工具封装模块。"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool


class BlogRetrieverToolService:
    """博客检索工具服务。

    当一个对象需要长期持有 `retriever` 这类状态时，用类会比“函数里再套一个函数”更清楚。

    Args:
        retriever: 已构建好的检索器实例。
    """

    def __init__(self, retriever: BaseRetriever) -> None:
        """初始化检索工具服务。

        Args:
            retriever: 已构建好的检索器实例。

        Returns:
            None
        """
        self.retriever = retriever

    def retrieve_blog_posts(self, query: str) -> str:
        """根据用户问题检索博客内容。

        Args:
            query: 用户输入的问题。

        Returns:
            str: 检索得到的相关文本内容，多个片段之间使用空行拼接。
        """
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def build_tool(self) -> BaseTool:
        """构建可供 LangGraph 调用的检索工具。

        Args:
            无。

        Returns:
            BaseTool: 封装完成的检索工具对象。
        """
        return StructuredTool.from_function(
            func=self.retrieve_blog_posts,
            name="retrieve_blog_posts",
            description="根据用户问题检索博客文章中的相关内容。",
        )


def build_retriever_tool(retriever: BaseRetriever) -> BaseTool:
    """创建检索工具。

    Args:
        retriever: 已构建好的检索器实例。

    Returns:
        BaseTool: 封装完成的检索工具对象。
    """
    tool_service = BlogRetrieverToolService(retriever)
    return tool_service.build_tool()
