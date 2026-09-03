"""对话 API 契约（三大场景统一入口，mode 分流）。"""

from pydantic import BaseModel, Field

from finrag.schemas.common import Citation


class ChatRequest(BaseModel):
    session_id: int | None = None
    question: str = Field(min_length=1, max_length=2000)
    # auto=自动意图路由（两级混合：规则先行 + LLM 兜底，详见 core/intent_router.py）
    mode: str = Field(default="knowledge", pattern=r"^(knowledge|nl2sql|dictionary|auto)$")
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: int
    message_id: int | None = None
    mode: str
    citations: list[Citation] = []
    sql: str | None = None
    affected_rows: int | None = None
    columns: list[str] | None = None
    rows: list[list] | None = None
    latency_ms: float = 0.0


class SessionCreate(BaseModel):
    # auto=自动意图路由（每轮按当前问题重新路由，不锁定场景）
    mode: str = Field(default="knowledge", pattern=r"^(knowledge|nl2sql|dictionary|auto)$")
    title: str = ""


class SessionOut(BaseModel):
    id: int
    mode: str
    title: str

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list | None = None
    sql_text: str | None = None
    latency_ms: float | None = None

    model_config = {"from_attributes": True}
