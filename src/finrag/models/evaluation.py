"""离线评估模型：评估用例 / 评估报告。"""

from sqlalchemy import JSON, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from finrag.models.base import Base, TimestampMixin


class EvalCase(Base, TimestampMixin):
    """一条评估用例（含人工标注的金标准）。"""

    __tablename__ = "eval_case"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene: Mapped[str] = mapped_column(String(16), nullable=False, index=True, comment="knowledge/nl2sql/dictionary")
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="问题")
    golden_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="金标准答案")
    golden_sql: Mapped[str | None] = mapped_column(Text, nullable=True, comment="金标准 SQL（nl2sql 场景）")
    expected_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True, comment="预期命中 chunk id 列表")


class EvalReport(Base, TimestampMixin):
    """一次评估运行的指标报告（RAGAS 结果）。"""

    __tablename__ = "eval_report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True, comment="运行批次号")
    scene: Mapped[str] = mapped_column(String(16), nullable=False)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True, comment="RAGAS 忠实度")
    relevancy: Mapped[float | None] = mapped_column(Float, nullable=True, comment="RAGAS 答案相关性")
    sql_success_rate: Mapped[float | None] = mapped_column(Float, nullable=True, comment="NL2SQL 执行成功率")
    case_count: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="逐用例明细")
