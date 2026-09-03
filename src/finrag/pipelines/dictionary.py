"""数据字典流水线：语义检索字段/表 + LLM 口径说明（可选）。"""

import time
from dataclasses import dataclass, field

from finrag.core.llm_gateway import LLMGateway
from finrag.core.schema_linker import SchemaLinker
from finrag.logging import get_logger

logger = get_logger("finrag.dictionary")


@dataclass
class FieldHit:
    table_name: str
    field_name: str
    field_type: str
    comment: str
    calibre: str
    synonyms: list[str]


@dataclass
class DictSearchResult:
    question: str
    hits: list[FieldHit] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class DictionaryAnswer:
    question: str
    hits: list[FieldHit] = field(default_factory=list)
    summary: str = ""
    latency_ms: float = 0.0

    @property
    def answer(self) -> str:
        """统一答案出口：LLM 口径汇总优先，无 LLM/失败时回退为字段清单文本。"""
        if self.summary:
            return self.summary
        if not self.hits:
            return "未在数据字典中检索到相关字段，请换个说法或确认字段名。"
        lines = ["与该问题相关的字段："]
        for h in self.hits[:10]:
            desc = f"- {h.table_name}.{h.field_name}（{h.field_type}）"
            if h.comment:
                desc += f"：{h.comment}"
            if h.calibre:
                desc += f"（口径：{h.calibre}）"
            lines.append(desc)
        return "\n".join(lines)


class DictionaryPipeline:
    """数据字典问答：先语义检索字段，再用 LLM 汇总口径（金融场景对口径准确性要求高）。"""

    def __init__(self, schema_linker: SchemaLinker, llm_gateway: LLMGateway | None = None) -> None:
        self._linker = schema_linker
        self._llm = llm_gateway

    @staticmethod
    def _to_field_hit(f: dict) -> FieldHit:
        return FieldHit(
            table_name=f.get("table_name", ""),
            field_name=f.get("field_name", ""),
            field_type=f.get("field_type", ""),
            comment=f.get("comment", ""),
            calibre=f.get("calibre", ""),
            synonyms=f.get("synonyms", []),
        )

    def search(self, question: str, top_k: int = 10) -> DictSearchResult:
        start = time.perf_counter()
        ctx = self._linker.link(question, top_k=top_k)
        hits = [self._to_field_hit(f) for f in ctx.fields][:top_k]
        latency = round((time.perf_counter() - start) * 1000, 1)
        logger.info("dictionary_search", hits=len(hits), latency_ms=latency)
        return DictSearchResult(question=question, hits=hits, latency_ms=latency)

    def answer(self, question: str) -> DictionaryAnswer:
        start = time.perf_counter()
        ctx = self._linker.link(question)
        hits = [self._to_field_hit(f) for f in ctx.fields]
        summary = ""
        if hits and self._llm is not None:
            prompt = (
                f"问题：{question}\n字段信息：\n{ctx.to_prompt()}\n"
                "请用中文回答：哪些字段与该问题相关、各自的统计口径是什么（≤150 字）。"
            )
            try:
                summary = self._llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
            except Exception as exc:
                logger.warning("dictionary_summary_failed", error=str(exc)[:200])
                summary = ""
        latency = round((time.perf_counter() - start) * 1000, 1)
        logger.info("dictionary_answer", hits=len(hits), latency_ms=latency)
        return DictionaryAnswer(question=question, hits=hits, summary=summary, latency_ms=latency)
