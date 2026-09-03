"""RAG 问答流水线：混合检索 → Rerank → 生成（带引用溯源）。"""

import time
from dataclasses import dataclass, field

from finrag.core.hybrid_retriever import HybridRetriever
from finrag.core.llm_gateway import LLMGateway
from finrag.core.reranker import Reranker
from finrag.logging import get_logger

logger = get_logger("finrag.rag")

SYSTEM_PROMPT = (
    "你是一名严谨的金融知识助手。回答必须基于提供的【参考片段】，不得编造事实；"
    "涉及数字、口径、金额时必须与参考片段一致；回答末尾标注引用的来源编号，例如[1][2]。"
    "若参考片段不足以回答问题，明确说明'参考材料中未找到相关依据'。"
)


@dataclass
class Citation:
    chunk_id: str
    content: str
    section_path: str
    score: float


@dataclass
class RAGAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    metrics: dict = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)  # 进入生成上下文的完整片段（RAGAS 评估用）


class RAGPipeline:
    """知识库问答主链路。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm_gateway: LLMGateway,
        rerank_top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm_gateway
        self._rerank_top_k = rerank_top_k

    def answer(
        self,
        question: str,
        query_vector: list[float],
        history: list[dict] | None = None,
        kb_filter: dict | None = None,
    ) -> RAGAnswer:
        start = time.perf_counter()

        # 1) 混合检索（向量 + BM25 + RRF）
        hits = self._retriever.search(
            query_vector=query_vector,
            query_text=question,
            filter_=kb_filter,
        )

        # 2) Rerank 精排（分数回写：citation 分数反映真实精排结果）
        docs = [h.payload.get("content", "") for h in hits]
        score_of: dict[str, float] = {h.doc_id: h.rrf_score for h in hits}
        if docs and self._rerank_top_k < len(docs):
            scores = self._reranker.rerank(question, docs)
            ranked = sorted(zip(hits, scores, strict=False), key=lambda item: item[1], reverse=True)
            hits = [h for h, _ in ranked[: self._rerank_top_k]]
            score_of = {h.doc_id: s for h, s in ranked[: self._rerank_top_k]}

        retrieved_contexts = [h.payload.get("content", "") for h in hits]

        # 3) 组装上下文并生成
        context_blocks = [f"[{idx + 1}] {c}" for idx, c in enumerate(retrieved_contexts)]
        context = "\n\n".join(context_blocks)

        messages = []
        if history:
            messages.extend(history[-6:])  # 最近 6 条做多轮上下文
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append(
            {
                "role": "user",
                "content": f"【参考片段】\n{context}\n\n【问题】\n{question}\n\n请基于参考片段回答，并标注引用编号。",
            }
        )

        answer_text = self._llm.stream_chat(messages)

        citations = [
            Citation(
                chunk_id=h.doc_id,
                content=h.payload.get("content", "")[:200],
                section_path=h.payload.get("section_path", ""),
                score=round(score_of.get(h.doc_id, h.rrf_score), 4),
            )
            for h in hits[: self._rerank_top_k]
        ]
        latency = round((time.perf_counter() - start) * 1000, 1)
        logger.info("rag_answer", latency_ms=latency, citations=len(citations))
        return RAGAnswer(
            answer=answer_text,
            citations=citations,
            latency_ms=latency,
            metrics={"hits": len(hits), "rerank_top_k": self._rerank_top_k},
            retrieved_contexts=retrieved_contexts,
        )
