"""LangGraph 工作流编排。"""

from typing import Literal

from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from prompts.rag_prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant",
    )


def build_graph(
    response_model: BaseChatModel,
    grader_model: BaseChatModel,
    retriever_tool,
):
    """组装并编译 RAG 工作流。"""

    def generate_query_or_respond(state: MessagesState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}],
        )
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    def rewrite_question(state: MessagesState):
        question = state["messages"][0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(state: MessagesState):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
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
