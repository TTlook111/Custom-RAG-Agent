"""LangGraph 工作流编排。"""

from typing import Literal, TypedDict

from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from custom_rag_agent.config.config import CHAT_MODEL_NAME, DASHSCOPE_API_KEY

MAX_REWRITE_ATTEMPTS = 2

GRADE_PROMPT = (
    "你是一名文档相关性评估助手。\n"
    "下面是检索到的文档内容：\n\n{context}\n\n"
    "下面是用户问题：{question}\n"
    "如果文档中的关键词或语义与问题相关，请判断为相关。\n"
    "请只返回 yes 或 no，用来表示该文档是否与问题相关。"
)

REWRITE_PROMPT = (
    "请分析用户输入，并理解其背后的真实语义意图。\n"
    "下面是原始问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "请输出一个更适合检索的改写问题："
)

GENERATE_PROMPT = (
    "你是一个问答助手。"
    "请根据下面检索到的上下文来回答用户问题。"
    "如果你不知道答案，就直接说明不知道。"
    "最多使用三句话，并保持回答简洁。\n"
    "问题：{question}\n"
    "上下文：{context}"
)


class GradeDocuments(BaseModel):
    """文档相关性评分结果。"""

    binary_score: str = Field(description="相关性评分：相关返回 yes，不相关返回 no。")


class RAGState(MessagesState, TypedDict, total=False):
    """RAG 工作流状态。"""

    rewrite_count: int


def build_chat_model() -> ChatTongyi:
    """构建通义聊天模型。

    Args:
        无。

    Returns:
        ChatTongyi: 可用于对话与评分的模型实例。
    """
    return ChatTongyi(
        model=CHAT_MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
    )


def build_graph(retriever: BaseRetriever):
    """构建 RAG 工作流图。

    Args:
        retriever: 已构建好的检索器对象。

    Returns:
        CompiledStateGraph: 可直接运行的 LangGraph 工作流对象。
    """
    response_model = build_chat_model()
    grader_model = build_chat_model()

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """根据用户问题检索博客文章中的相关内容。"""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_query_or_respond(state: RAGState) -> dict:
        """决定直接回答还是先调用检索工具。"""
        response = response_model.bind_tools([retrieve_blog_posts]).invoke(state["messages"])
        return {"messages": [response], "rewrite_count": state.get("rewrite_count", 0)}

    def grade_documents(state: RAGState) -> Literal["generate_answer", "rewrite_question"]:
        """判断检索结果是否与用户问题相关。"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}],
        )
        normalized_score = response.binary_score.strip().lower()
        if normalized_score == "yes":
            return "generate_answer"
        if state.get("rewrite_count", 0) >= MAX_REWRITE_ATTEMPTS:
            return "generate_answer"
        return "rewrite_question"

    def rewrite_question(state: RAGState) -> dict:
        """改写用户问题，让后续检索更准确。"""
        question = state["messages"][0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {
            "messages": [HumanMessage(content=response.content)],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def generate_answer(state: RAGState) -> dict:
        """基于检索上下文生成最终答案。"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    workflow = StateGraph(RAGState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retrieve_blog_posts]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    return workflow.compile()
