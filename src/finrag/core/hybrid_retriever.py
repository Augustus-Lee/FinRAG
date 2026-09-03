"""混合检索：向量 + BM25，RRF 融合（自研核心组件）。

RRF（Reciprocal Rank Fusion）：
    score(doc) = Σ w_i / (k + rank_i(doc))
不依赖分数尺度，只依赖排序位置，天然适合融合异构检索结果。
w_vector / w_bm25 支持双路加权（默认 1:1，等价经典 RRF）。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from finrag.core.bm25 import BM25Index
from finrag.core.vectorstore import SearchHit, VectorStore
from finrag.logging import get_logger

logger = get_logger("finrag.hybrid_retriever")


@dataclass
class HybridHit:
    doc_id: str
    rrf_score: float
    vector_score: float
    bm25_score: float
    payload: dict[str, Any]


def rrf_fuse(
    vector_hits: list[SearchHit],
    bm25_hits: list[Any],
    k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> dict[str, dict[str, float]]:
    """加权 RRF 融合两个排名列表。

    Args:
        vector_hits: 向量检索结果（含 id/score/payload）
        bm25_hits: BM25 结果（含 doc_id/score/payload）
        k: RRF 常数（经验值 60）
        vector_weight: 向量路权重
        bm25_weight: BM25 路权重

    Returns:
        doc_id -> {"rrf": 分数, "vector": 分数, "bm25": 分数}
    """
    fused: dict[str, dict[str, float]] = {}

    for rank, hit in enumerate(vector_hits):
        fused.setdefault(hit.id, {"rrf": 0.0, "vector": 0.0, "bm25": 0.0})
        fused[hit.id]["rrf"] += vector_weight / (k + rank + 1)
        fused[hit.id]["vector"] = hit.score

    for rank, hit in enumerate(bm25_hits):
        fused.setdefault(hit.doc_id, {"rrf": 0.0, "vector": 0.0, "bm25": 0.0})
        fused[hit.doc_id]["rrf"] += bm25_weight / (k + rank + 1)
        fused[hit.doc_id]["bm25"] = hit.score

    return fused


class HybridRetriever:
    """统一检索入口：向量召回 + BM25 召回 → 加权 RRF 融合 → Top-K。"""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = 60,
        retrieve_top_k: int = 20,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        ensure_indexed: Callable[[], None] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._rrf_k = rrf_k
        self._retrieve_top_k = retrieve_top_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._ensure_indexed = ensure_indexed
        self._ensured = False  # ensure_indexed 仅首次调用执行一次

    def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int | None = None,
        filter_: dict | None = None,
    ) -> list[HybridHit]:
        top_k = top_k or self._retrieve_top_k
        candidate_k = max(top_k * 2, 20)  # 融合前扩大召回，保证并集质量

        # BM25 为内存索引：进程重启后丢失，首次调用前从元数据库懒重建
        if self._ensure_indexed is not None and not self._ensured:
            try:
                self._ensure_indexed()
            except Exception as exc:
                logger.warning("bm25_ensure_indexed_failed", error=str(exc)[:200])
            finally:
                self._ensured = True

        vector_hits = self._vector_store.search(query_vector, top_k=candidate_k, filter_=filter_)
        bm25_hits = self._bm25.search(query_text, top_k=candidate_k)

        fused = rrf_fuse(
            vector_hits,
            bm25_hits,
            k=self._rrf_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )
        ranked = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:top_k]

        # payload 双源合并：向量命中优先，BM25 命中补齐（仅 BM25 命中的文档不再丢内容）
        payload_map: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            payload_map[hit.id] = hit.payload
        for hit in bm25_hits:
            payload_map.setdefault(hit.doc_id, hit.payload)

        results = [
            HybridHit(
                doc_id=doc_id,
                rrf_score=scores["rrf"],
                vector_score=scores["vector"],
                bm25_score=scores["bm25"],
                payload=payload_map.get(doc_id, {}),
            )
            for doc_id, scores in ranked
        ]
        logger.info(
            "hybrid_search",
            vector_hits=len(vector_hits),
            bm25_hits=len(bm25_hits),
            fused=len(fused),
            returned=len(results),
        )
        return results
