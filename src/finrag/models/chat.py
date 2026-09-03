"""对话模型：会话 / 消息（含引用溯源、SQL 与评估指标）。"""

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from finrag.models.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    """一次问答会话，mode 区分三大场景。"""

    __tablename__ = "chat_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, comment="dictionary/nl2sql/knowledge")
    title: Mapped[str] = mapped_column(String(128), default="", comment="会话标题")


class ChatMessage(Base, TimestampMixin):
    """单条消息。assistant 消息携带引用、SQL、评估指标，实现全链路可观测。"""

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_session.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="user/assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True, comment="引用溯源：chunk/字段出处列表")
    sql_text: Mapped[str | None] = mapped_column(Text, nullable=True, comment="智能问数生成的 SQL")
    sql_affected_rows: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="SQL 返回行数")
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True, comment="处理耗时")
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="检索/生成指标快照")
