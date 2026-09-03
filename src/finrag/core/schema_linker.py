"""Schema Linking（自研核心组件）。

思路：数据字典表结构本身就是"RAG 索引"——先用问题检索相关表/字段（避免全量 schema 撑爆上下文），
再把选中表的 schema 片段注入 NL2SQL prompt。
混合检索路径：复用 HybridRetriever（向量 + BM25 + RRF），字典专用集合与专用 BM25 实例。
降级路径：向量/BM25 不可用时回退到关键词重叠打分（平滑升级）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finrag.logging import get_logger

logger = get_logger("finrag.schema_linker")


@dataclass
class SchemaContext:
    """注入 prompt 的 schema 上下文。"""

    tables: list[dict] = field(default_factory=list)
    fields: list[dict] = field(default_factory=list)
    matched_tables: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines: list[str] = []
        for table in self.tables:
            lines.append(f"表 {table['table_name']}（{table.get('business_domain', '')}）: {table.get('description', '')}")
            for f in [fld for fld in self.fields if fld["table_name"] == table["table_name"]]:
                lines.append(
                    f"  - {f['field_name']} {f.get('field_type', '')} "
                    f"注释={f.get('comment', '')} 口径={f.get('calibre', '')}"
                )
        return "\n".join(lines)


class SchemaLinker:
    """根据问题从数据字典中检索相关表与字段。

    优先走混合检索（注入 retriever/embedding/indexer 时）；不可用时回退关键词打分。
    link(question, top_k=25)：top_k 控制混合检索返回字段数；默认 25 以保留 NL2SQL 上下文规模。
    """

    def __init__(
        self,
        table_provider=None,
        top_k: int = 5,
        *,
        retriever: HybridRetriever | None = None,
        embedding: EmbeddingProvider | None = None,
        indexer: DictionaryIndexer | None = None,
    ) -> None:
        self._provider = table_provider or (lambda: ([], []))
        self._top_k = top_k
        self._cache: tuple[list, list] | None = None
        self._retriever = retriever
        self._embedding = embedding
        self._indexer = indexer
        self._index_built = False

    def _all_data(self) -> tuple[list, list]:
        if self._cache is None:
            self._cache = self._provider()
        return self._cache

    def link(self, question: str, top_k: int = 25) -> SchemaContext:
        if self._retriever is not None and self._embedding is not None and self._indexer is not None:
            ctx = self._hybrid_link(question, top_k)
            if ctx is not None:
                return ctx
            logger.info("schema_link_fallback", reason="hybrid_unavailable")
        return self._keyword_link(question)

    def _ensure_indexed(self) -> None:
        if not self._index_built:
            try:
                self._indexer.build()
            except Exception as exc:
                logger.warning("schema_link_build_failed", error=str(exc)[:200])
            finally:
                self._index_built = True

    def _hybrid_link(self, question: str, top_k: int) -> SchemaContext | None:
        self._ensure_indexed()
        if self._indexer.size == 0:
            return None
        try:
            query_vector = self._embedding.embed_query(question)
            hits = self._retriever.search(query_vector, question, top_k=top_k)
        except Exception as exc:
            logger.warning("schema_link_hybrid_failed", error=str(exc)[:200])
            return None
        fields = [hit.payload for hit in hits if hit.payload]
        if not fields:
            return None
        tables, _ = self._all_data()
        matched_names = {f.get("table_name", "") for f in fields}
        matched = [t for t in tables if t.get("table_name") in matched_names]
        ctx = SchemaContext(
            tables=matched,
            fields=fields,
            matched_tables=sorted(matched_names),
        )
        logger.info(
            "schema_link",
            mode="hybrid",
            matched_tables=ctx.matched_tables,
            top_fields=[(f.get("table_name", ""), f.get("field_name", "")) for f in fields[:5]],
        )
        return ctx

    def rebuild(self) -> None:
        """字典数据变更后主动重建索引并清空关键词缓存。"""
        self._index_built = False
        self._cache = None
        if self._indexer is not None:
            try:
                self._indexer.build()
                self._index_built = True
            except Exception as exc:
                logger.warning("schema_link_rebuild_failed", error=str(exc)[:200])

    def _keyword_link(self, question: str) -> SchemaContext:
        tables, fields = self._all_data()
        if not tables:
            return SchemaContext()

        question_tokens = self._tokens(question)
        table_scores = {
            t["table_name"]: self._score(question_tokens, self._table_text(t, fields))
            for t in tables
        }
        scored_fields: list[tuple[float, dict]] = []
        for f in fields:
            haystack = " ".join(
                [
                    f.get("field_name", ""),
                    f.get("comment", ""),
                    f.get("calibre", ""),
                    " ".join(f.get("synonyms", [])),
                ]
            )
            fs = self._score(question_tokens, haystack)
            if fs > 0:
                scored_fields.append((fs, f))
        scored_fields.sort(key=lambda item: (-item[0], -table_scores.get(item[1]["table_name"], 0.0)))
        limit = max(self._top_k * 5, 20)
        top_fields = [f for _, f in scored_fields[:limit]]

        if not top_fields:
            hit_tables = {t for t, s in table_scores.items() if s > 0}
            top_fields = [f for f in fields if f.get("table_name") in hit_tables]
            matched_names = set(hit_tables)
        else:
            matched_names = {f["table_name"] for f in top_fields}

        matched = [t for t in tables if t["table_name"] in matched_names]
        ctx = SchemaContext(
            tables=matched,
            fields=top_fields,
            matched_tables=sorted(matched_names),
        )
        logger.info(
            "schema_link",
            mode="keyword",
            matched_tables=ctx.matched_tables,
            top_fields=[(f["table_name"], f["field_name"]) for f in top_fields[:5]],
        )
        return ctx

    @staticmethod
    def _score(question_tokens: set[str], haystack: str) -> float:
        return float(len(question_tokens & SchemaLinker._tokens(haystack)))

    @staticmethod
    def _table_text(table: dict, fields: list[dict]) -> str:
        return " ".join(
            [
                table.get("table_name", ""),
                table.get("description", ""),
                table.get("business_domain", ""),
                " ".join(
                    f.get("comment", "")
                    + " "
                    + f.get("calibre", "")
                    + " "
                    + " ".join(f.get("synonyms", []))
                    for f in fields
                    if f.get("table_name") == table.get("table_name")
                ),
            ]
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        for cjk in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.add(cjk)
            if len(cjk) > 1:
                tokens.update(cjk[i : i + 2] for i in range(len(cjk) - 1))
            if len(cjk) > 2:
                tokens.update(cjk[i : i + 3] for i in range(len(cjk) - 2))
        tokens.update(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))
        return tokens
