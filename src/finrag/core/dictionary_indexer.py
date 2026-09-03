"""数据字典索引构建器：将 dict_field 向量化并写入字典专用向量集合 + BM25Index。

复用 EmbeddingProvider / VectorStore / BM25Index，与文档检索物理隔离（专用集合 + 专用 BM25 实例）。
build() 幂等：重复调用先清旧索引再重建，避免陈旧残留。
"""

import uuid
from collections.abc import Callable

from finrag.core.bm25 import BM25Index
from finrag.core.embedding import EmbeddingProvider
from finrag.core.vectorstore import VectorPoint, VectorStore
from finrag.logging import get_logger


def _doc_id(field_id: int) -> str:
    """确定性的 Qdrant 合法 point id（UUID5）。

    Qdrant 仅接受无符号整数或 UUID 字符串作为 point id；
    形如 "dict_field_1" 的非 UUID 字符串会被 400 拒绝。
    UUID5 保证同一字段重建后 id 稳定 → upsert 覆盖、delete 精准。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"dict_field_{field_id}"))

logger = get_logger("finrag.dictionary_indexer")


def _field_text(f: dict) -> str:
    """字段检索文本：字段名 + 注释 + 口径 + 同义词。"""
    return " ".join(
        [
            f.get("field_name", ""),
            f.get("comment", ""),
            f.get("calibre", ""),
            " ".join(f.get("synonyms") or []),
        ]
    )


class DictionaryIndexer:
    """数据字典索引构建器（向量 + BM25 双写）。"""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        field_provider: Callable[[], list[dict]],
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._field_provider = field_provider
        self._indexed_ids: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._indexed_ids)

    def build(self) -> int:
        """幂等重建字典索引，返回索引字段数。"""
        fields = self._field_provider() or []

        # 先清旧索引，保证幂等
        if self._indexed_ids:
            try:
                self._vector_store.delete_by_ids(sorted(self._indexed_ids))
            except Exception as exc:
                logger.warning("dict_index_delete_failed", error=str(exc)[:200])
        self._bm25.clear()
        self._indexed_ids = set()

        if not fields:
            logger.info("dict_index_empty")
            return 0

        texts = [_field_text(f) for f in fields]
        try:
            vectors = self._embedding.embed(texts)
        except Exception as exc:
            logger.error("dict_index_embed_failed", error=str(exc)[:300])
            return 0

        doc_ids: list[str] = []
        points: list[VectorPoint] = []
        for idx, f in enumerate(fields):
            doc_id = _doc_id(f["id"])
            doc_ids.append(doc_id)
            points.append(VectorPoint(id=doc_id, vector=vectors[idx], payload=f))

        try:
            self._vector_store.upsert(points)
        except Exception as exc:
            logger.error("dict_index_upsert_failed", error=str(exc)[:300])
            return 0

        for doc_id, text in zip(doc_ids, texts):
            self._bm25.add(doc_id, text)
        # 仅在向量库 + BM25 双写成功后才登记，避免半成功导致 size 与实际不符
        self._indexed_ids = set(doc_ids)

        logger.info("dict_index_built", fields=len(fields))
        return len(fields)
