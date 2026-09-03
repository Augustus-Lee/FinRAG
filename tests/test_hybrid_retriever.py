"""混合检索测试：RRF 融合排序 + HybridRetriever 端到端。"""

from types import SimpleNamespace

from finrag.core.bm25 import BM25Index
from finrag.core.hybrid_retriever import HybridRetriever, rrf_fuse
from finrag.core.vectorstore import SearchHit, VectorPoint, VectorStore


class FakeVectorStore(VectorStore):
    """测试桩：返回预置命中。"""

    def __init__(self, hits: list[SearchHit]):
        self._hits = hits

    def upsert(self, points: list[VectorPoint]) -> None:
        pass

    def search(self, vector, top_k, filter_=None):
        return self._hits[:top_k]

    def delete_by_ids(self, point_ids: list[str]) -> None:
        pass

    def healthcheck(self) -> bool:
        return True


def test_rrf_fuse_ranks_doc_in_both_lists_first():
    vec = [SearchHit(id="a", score=0.9), SearchHit(id="b", score=0.6)]
    bm = [
        SimpleNamespace(doc_id="b", score=8.0, payload={}),
        SimpleNamespace(doc_id="c", score=4.0, payload={}),
    ]
    fused = rrf_fuse(vec, bm, k=60)

    # b 同时在两个列表 → RRF 累加最高
    assert fused["b"]["rrf"] > fused["a"]["rrf"]
    assert fused["b"]["rrf"] > fused["c"]["rrf"]
    # 单一来源分数：a 在向量结果 rank0 → 1/61；b 在向量 rank1 + BM25 rank0 → 1/62 + 1/61
    assert abs(fused["a"]["rrf"] - 1 / 61) < 1e-9
    assert abs(fused["b"]["rrf"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused["c"]["rrf"] - 1 / 62) < 1e-9
    # 缺失渠道分数为 0
    assert fused["c"]["vector"] == 0.0


def test_hybrid_retriever_fuses_and_carries_payload():
    vec_hits = [
        SearchHit(id="doc1", score=0.8, payload={"title": "存款利率"}),
        SearchHit(id="doc2", score=0.5, payload={"title": "贷款规则"}),
    ]
    store = FakeVectorStore(vec_hits)

    bm25 = BM25Index()
    bm25.add("doc2", "贷款规则 利率 说明")
    bm25.add("doc3", "存款产品 说明")

    retriever = HybridRetriever(store, bm25, rrf_k=60, retrieve_top_k=10)
    results = retriever.search([0.1, 0.2, 0.3], "贷款规则", top_k=2)

    # doc2 同时命中向量与 BM25 → 排第一
    assert [r.doc_id for r in results][0] == "doc2"
    assert len(results) == 2
    # payload 从向量命中透传
    assert results[0].payload.get("title") == "贷款规则"


def test_rrf_weights_change_ranking():
    """调大 bm25_weight 后，BM25-only 命中应反超 vector-only 命中。"""
    vec = [SearchHit(id="vec_only", score=0.9)]
    bm = [SimpleNamespace(doc_id="bm_only", score=8.0, payload={})]

    # 1:1 时两者均单路 rank0 → 分数相同；bm25 加权后 bm_only 反超
    fused_even = rrf_fuse(vec, bm, k=60)
    assert abs(fused_even["vec_only"]["rrf"] - fused_even["bm_only"]["rrf"]) < 1e-9

    fused_weighted = rrf_fuse(vec, bm, k=60, vector_weight=1.0, bm25_weight=2.0)
    assert fused_weighted["bm_only"]["rrf"] > fused_weighted["vec_only"]["rrf"]
    assert abs(fused_weighted["bm_only"]["rrf"] - 2.0 / 61) < 1e-9


def test_bm25_only_hit_payload_backfilled():
    """仅命中 BM25 的文档，payload 应从 BM25 侧补齐（不再是空 dict）。"""
    vec_hits = [SearchHit(id="doc1", score=0.8, payload={"content": "向量命中内容"})]
    store = FakeVectorStore(vec_hits)

    bm25 = BM25Index()
    bm25.add("doc1", "向量命中内容")
    bm25.add("doc2", "独占 BM25 关键词内容", payload={"content": "独占 BM25 关键词内容", "chunk_id": 5})

    retriever = HybridRetriever(store, bm25, rrf_k=60, retrieve_top_k=10)
    results = retriever.search([0.1, 0.2, 0.3], "独占 BM25 关键词", top_k=5)

    by_id = {r.doc_id: r for r in results}
    assert "doc2" in by_id
    assert by_id["doc2"].payload.get("content") == "独占 BM25 关键词内容"
    assert by_id["doc2"].payload.get("chunk_id") == 5


def test_ensure_indexed_runs_once_and_survives_failure():
    """懒重建回调仅执行一次；抛异常也不阻断检索（降级为双路现状）。"""
    calls = []

    def flaky() -> None:
        calls.append(1)
        raise RuntimeError("rebuild down")

    store = FakeVectorStore([SearchHit(id="doc1", score=0.9, payload={"content": "x"})])
    bm25 = BM25Index()
    bm25.add("doc1", "x 内容")
    retriever = HybridRetriever(store, bm25, ensure_indexed=flaky)

    r1 = retriever.search([0.1], "x 内容", top_k=5)
    r2 = retriever.search([0.1], "x 内容", top_k=5)
    assert len(r1) == 1 and len(r2) == 1
    assert len(calls) == 1  # 异常后置标志，不重试不阻断
