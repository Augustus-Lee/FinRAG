"""RAG 流水线测试：rerank 分数回写、retrieved_contexts 一致性。"""

from unittest.mock import MagicMock

from finrag.core.hybrid_retriever import HybridHit
from finrag.pipelines.rag import RAGPipeline


def _hit(doc_id: str, content: str, rrf: float = 0.01) -> HybridHit:
    return HybridHit(doc_id=doc_id, rrf_score=rrf, vector_score=rrf, bm25_score=0.0,
                     payload={"content": content, "section_path": "s"})


def _make(hits, rerank_scores=None):
    retriever = MagicMock()
    retriever.search.return_value = hits
    reranker = MagicMock()
    reranker.rerank.return_value = rerank_scores or []
    llm = MagicMock()
    llm.stream_chat.return_value = "答案权威"
    return RAGPipeline(retriever, reranker, llm, rerank_top_k=2), retriever, reranker, llm


def test_rerank_score_writes_back_to_citations():
    hits = [_hit("a", "内容A", 0.03), _hit("b", "内容B", 0.02), _hit("c", "内容C", 0.01)]
    # rerank 给出与 RRF 相反的排序：c 最高、b 次之
    p, *_ = _make(hits, rerank_scores=[0.1, 0.2, 0.9])

    ans = p.answer("问题", [0.1])

    # 精排后前 2 名是 c、b，citation.score 应等于 rerank 分（非 RRF 分）
    assert [c.chunk_id for c in ans.citations] == ["c", "b"]
    assert ans.citations[0].score == 0.9
    assert ans.citations[1].score == 0.2
    # 进入上下文的也是精排后的内容
    assert ans.retrieved_contexts == ["内容C", "内容B"]


def test_no_rerank_citation_score_stays_rrf():
    # 命中数 ≤ rerank_top_k → 不触发 rerank，score 保持 RRF 分
    hits = [_hit("a", "内容A", 0.03), _hit("b", "内容B", 0.02)]
    p, _, reranker, _ = _make(hits)

    ans = p.answer("问题", [0.1])

    reranker.rerank.assert_not_called()
    assert [c.chunk_id for c in ans.citations] == ["a", "b"]
    assert ans.citations[0].score == 0.03
    assert ans.retrieved_contexts == ["内容A", "内容B"]


def test_retrieved_contexts_matches_prompt_context():
    hits = [_hit("a", "内容A", 0.03), _hit("b", "内容B", 0.02), _hit("c", "内容C", 0.01)]
    p, _, _, llm = _make(hits, rerank_scores=[0.1, 0.9, 0.5])

    ans = p.answer("问题", [0.1])

    sent = llm.stream_chat.call_args.args[0][-1]["content"]  # 最后一条 user 消息
    # prompt 中的参考片段与 retrieved_contexts 一一对应
    assert f"[1] {ans.retrieved_contexts[0]}" in sent
    assert f"[2] {ans.retrieved_contexts[1]}" in sent
    assert "[3]" not in sent
