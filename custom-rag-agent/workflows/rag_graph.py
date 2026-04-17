"""LangGraph 工作流编排。"""

from typing import Literal

from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from prompts.rag_prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT


class GradeDocuments(BaseModel):
    """文档相关性评分结果。"""

    binary_score: str = Field(
        description="相关性评分：相关返回 yes，不相关返回 no。",
    )


class RagWorkflowService:
    """RAG 工作流服务。

    该类负责集中管理工作流依赖与节点方法，避免在一个方法里继续套定义多个内部方法。

    Args:
        response_model: 负责普通对话与最终回答生成的聊天模型。
        grader_model: 负责判断检索结果是否相关的聊天模型。
        retriever_tool: 提供给工作流调用的检索工具。
    """

    def __init__(
        self,
        response_model: BaseChatModel,
        grader_model: BaseChatModel,
        retriever_tool: BaseTool,
    ) -> None:
        """初始化工作流服务。

        Args:
            response_model: 负责普通对话与最终回答生成的聊天模型。
            grader_model: 负责判断检索结果是否相关的聊天模型。
            retriever_tool: 提供给工作流调用的检索工具。

        Returns:
            None
        """
        self.response_model = response_model
        self.grader_model = grader_model
        self.retriever_tool = retriever_tool

    def generate_query_or_respond(self, state: MessagesState) -> dict:
        """决定直接回答还是先调用检索工具。

        Args:
            state: 当前工作流状态，内部包含历史消息。

        Returns:
            dict: 新的消息状态，内容为模型本轮输出。
        """
        response = self.response_model.bind_tools([self.retriever_tool]).invoke(
            state["messages"],
        )
        return {"messages": [response]}

    def grade_documents(
        self,
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """判断检索结果是否与用户问题相关。

        Args:
            state: 当前工作流状态，包含问题与检索结果。

        Returns:
            Literal["generate_answer", "rewrite_question"]:
                相关则进入生成答案节点，否则进入改写问题节点。
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = self.grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}],
        )
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(self, state: MessagesState) -> dict:
        """改写用户问题，让后续检索更准确。

        Args:
            state: 当前工作流状态，包含用户原始问题。

        Returns:
            dict: 新的消息状态，内容为改写后的问题。
        """
        question = state["messages"][0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(self, state: MessagesState) -> dict:
        """基于检索上下文生成最终答案。

        Args:
            state: 当前工作流状态，包含问题与检索上下文。

        Returns:
            dict: 新的消息状态，内容为最终回答。
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def build_graph(self):
        """组装并编译 RAG 工作流图。

        Args:
            无。

        Returns:
            CompiledStateGraph: 可直接运行的 LangGraph 工作流对象。
        """
        workflow = StateGraph(MessagesState)
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
        workflow.add_conditional_edges("retrieve", self.grade_documents)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        return workflow.compile()


def build_graph(
    response_model: BaseChatModel,
    grader_model: BaseChatModel,
    retriever_tool: BaseTool,
):
    """对外提供统一的工作流构建入口。

    Args:
        response_model: 负责普通对话与最终回答生成的聊天模型。
        grader_model: 负责判断检索结果是否相关的聊天模型。
        retriever_tool: 提供给工作流调用的检索工具。

    Returns:
        CompiledStateGraph: 可直接运行的 LangGraph 工作流对象。
    """
    workflow_service = RagWorkflowService(
        response_model=response_model,
        grader_model=grader_model,
        retriever_tool=retriever_tool,
    )
    return workflow_service.build_graph()
