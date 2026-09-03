"""NL2SQL 自修正工作流（LangGraph StateGraph）：生成 SQL → 执行 → 执行异常（含 MCP Server 拒绝）错误回灌重试。

循环与条件分支正是使用 LangGraph 的场景；固定链路（RAG 问答）则用普通流水线。
SQL 安全校验由 MCP Server 端管控。
注意：max_attempts 为总尝试次数上限（含首次）；与 pipeline 的 max_retries（额外重试次数）语义不同。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finrag.logging import get_logger
from finrag.pipelines.nl2sql import NL2SQLPipeline

logger = get_logger("finrag.workflow.nl2sql")


class NL2SQLState(TypedDict):
    question: str
    sql: str
    error: str
    attempts: int
    max_attempts: int
    result: dict


def _generate(pipeline: NL2SQLPipeline) -> object:
    def node(state: NL2SQLState) -> dict:
        sql = pipeline.generate_sql(state["question"], state.get("error", ""))
        return {"sql": sql, "error": ""}

    return node


def _route_after_execute(state: NL2SQLState) -> str:
    """执行异常且未超过重试上限 → 回到生成节点；否则结束。"""
    if state.get("error") and state.get("attempts", 0) < state.get("max_attempts", 2):
        return "generate"
    return "end"


def _execute(pipeline: NL2SQLPipeline) -> object:
    def node(state: NL2SQLState) -> dict:
        try:
            result = pipeline._executor.execute_sql(state["sql"])
            return {
                "result": {
                    "ok": True,
                    "sql": state["sql"],
                    "columns": result.columns,
                    "rows": result.rows,
                    "affected_rows": result.affected_rows,
                    "attempts": state.get("attempts", 1),
                },
                "error": "",
            }
        except Exception as exc:
            return {
                "result": {"ok": False, "error": str(exc)[:300], "attempts": state.get("attempts", 1)},
                "error": str(exc)[:300],
                "attempts": state.get("attempts", 0) + 1,
            }

    return node


def build_nl2sql_graph(pipeline: NL2SQLPipeline, max_attempts: int = 2) -> "StateGraph":
    """构建 NL2SQL 自修正图（编译后返回）。"""
    graph = StateGraph(NL2SQLState)
    graph.add_node("generate", _generate(pipeline))
    graph.add_node("execute", _execute(pipeline))

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", _route_after_execute, {"generate": "generate", "end": END})
    return graph.compile()


def invoke_nl2sql(pipeline: NL2SQLPipeline, question: str, max_attempts: int = 2) -> dict:
    """以工作流方式执行一次 NL2SQL 查询（执行失败自动回灌重试）。"""
    graph = build_nl2sql_graph(pipeline, max_attempts)
    result = graph.invoke(
        {"question": question, "sql": "", "error": "", "attempts": 0,
         "max_attempts": max_attempts, "result": {}}
    )
    logger.info("nl2sql_workflow_done", attempts=result.get("attempts"), ok=result.get("result", {}).get("ok"))
    return result["result"]
