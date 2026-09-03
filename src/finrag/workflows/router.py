"""三场景路由工作流（LangGraph StateGraph）。

意图识别 → 条件边分流到 数据字典 / 智能问数 / 知识库。
框架阶段意图识别用关键词规则（可替换为 LLM 分类），新增场景只需注册节点。
"""

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finrag.logging import get_logger

logger = get_logger("finrag.workflow.router")

# 问数类关键词（结构化数据查询）
NL2SQL_KEYWORDS = [
    "多少", "统计", "环比", "同比", "增长率", "汇总", "查询", "金额", "数量",
    "排名", "top", "平均", "sum", "count", "max", "min", "趋势",
]
# 数据字典类关键词（字段口径/元数据）
DICTIONARY_KEYWORDS = [
    "口径", "字段", "表结构", "字典", "含义", "定义", "在哪张表", "元数据", "schema",
]


class SceneState(TypedDict):
    question: str
    scene: str
    intent_reason: str
    payload: dict


def classify_scene(state: SceneState) -> dict:
    """关键词规则意图识别（后续可替换为 LLM few-shot 分类）。"""
    q = state["question"].lower()
    if any(k.lower() in q for k in NL2SQL_KEYWORDS):
        return {"scene": "nl2sql", "intent_reason": "keywords:query"}
    if any(k.lower() in q for k in DICTIONARY_KEYWORDS):
        return {"scene": "dictionary", "intent_reason": "keywords:dictionary"}
    return {"scene": "knowledge", "intent_reason": "default"}


def build_scene_router(handlers: dict[str, Callable]) -> "StateGraph":
    """构建场景路由图。

    Args:
        handlers: {"nl2sql": node_fn, "dictionary": node_fn, "knowledge": node_fn}
                  每个 node_fn 接收 state 返回部分 state 更新。
    """
    graph = StateGraph(SceneState)
    graph.add_node("classify", classify_scene)

    for name, fn in handlers.items():
        graph.add_node(f"scene_{name}", fn)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: f"scene_{state['scene']}",
        {f"scene_{n}": f"scene_{n}" for n in handlers},
    )
    for name in handlers:
        graph.add_edge(f"scene_{name}", END)
    return graph.compile()
