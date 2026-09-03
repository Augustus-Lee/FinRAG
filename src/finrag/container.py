"""依赖容器：组件装配与全局单例（生产可替换为正式 DI 框架）。"""

from functools import lru_cache

from sqlalchemy import create_engine

from finrag.config import get_settings
from finrag.core.bm25 import BM25Index
from finrag.core.chunker import create_chunker
from finrag.core.dictionary_indexer import DictionaryIndexer
from finrag.core.embedding import EmbeddingProvider, create_embedding_provider
from finrag.core.hybrid_retriever import HybridRetriever
from finrag.core.llm_gateway import LLMGateway
from finrag.core.mcp_executor import DbDirectExecutor, McpExecutor
from finrag.core.reranker import Reranker, create_reranker
from finrag.core.ragas_evaluator import RagasEvaluator
from finrag.core.schema_linker import SchemaLinker
from finrag.core.vectorstore import QdrantVectorStore, VectorStore, create_vector_store
from finrag.db.session import SessionLocal
from finrag.logging import get_logger
from finrag.pipelines.dictionary import DictionaryPipeline
from finrag.pipelines.ingest import IngestPipeline
from finrag.pipelines.nl2sql import NL2SQLPipeline
from finrag.pipelines.rag import RAGPipeline

logger = get_logger("finrag.container")


@lru_cache
def get_llm_gateway() -> LLMGateway:
    return LLMGateway(get_settings())


@lru_cache
def get_intent_router():
    """意图路由器（mode=auto）：规则先行 + LLM 兜底，默认 knowledge。"""
    from finrag.core.intent_router import create_intent_router

    return create_intent_router(get_settings(), llm=get_llm_gateway())


@lru_cache
def get_query_rewriter():
    """查询改写器（多轮）：无历史/无指代信号零成本透传，否则 LLM 改写为 self-contained question。"""
    from finrag.core.query_rewriter import create_query_rewriter

    return create_query_rewriter(get_settings(), llm=get_llm_gateway())


@lru_cache
def get_embedding() -> EmbeddingProvider:
    return create_embedding_provider(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return create_vector_store(s)


@lru_cache
def get_bm25() -> BM25Index:
    """全局内存 BM25 索引（单机部署；多实例需换外部索引/共享 Redis）。"""
    return BM25Index()


_ensure_docs_bm25_done = False


def _ensure_docs_bm25() -> None:
    """主链路 BM25 懒重建：进程重启后内存索引丢失，从 KBChunk 元数据重建。

    仅在索引为空时重建一次；doc_id 用 vector_id 与 Qdrant point id 对齐，
    保证 RRF 融合时双路命中能合并到同一文档。
    """
    global _ensure_docs_bm25_done
    if _ensure_docs_bm25_done:
        return
    bm25 = get_bm25()
    if len(bm25) > 0:
        _ensure_docs_bm25_done = True
        return

    from finrag.models import KBChunk

    db = SessionLocal()
    try:
        chunks = db.query(KBChunk).all()
        if not chunks:
            logger.info("bm25_lazy_rebuild_skipped", reason="no_chunks_in_db")
            return
        bm25.add_batch(
            [
                (
                    c.vector_id,
                    c.content,
                    {
                        "doc_id": c.doc_id,
                        "chunk_id": c.seq_no,
                        "content": c.content,
                        "section_path": c.section_path,
                        "table_meta": c.table_meta,
                    },
                )
                for c in chunks
            ]
        )
        logger.info("bm25_lazy_rebuilt", chunks=len(chunks))
    finally:
        db.close()
        _ensure_docs_bm25_done = True


@lru_cache
def get_reranker() -> Reranker:
    """按配置装配 reranker：local(BGE CrossEncoder) / api(Cohere 兼容) / noop(未启用)。"""
    return create_reranker(get_settings())


@lru_cache
def get_ragas_evaluator() -> RagasEvaluator:
    """RAGAS 评估器（M3）：忠实度/答案相关性计算。"""
    return RagasEvaluator(get_settings())


@lru_cache
def get_retriever() -> HybridRetriever:
    s = get_settings()
    return HybridRetriever(
        vector_store=get_vector_store(),
        bm25_index=get_bm25(),
        rrf_k=s.rrf_k,
        retrieve_top_k=s.retrieve_top_k,
        vector_weight=s.rrf_vector_weight,
        bm25_weight=s.rrf_bm25_weight,
        ensure_indexed=_ensure_docs_bm25,
    )


def _dictionary_provider() -> tuple[list, list]:
    """从元数据库读取数据字典（SchemaLinker 的数据源）。"""
    from finrag.models import DictField, DictTable

    db = SessionLocal()
    try:
        tables = [
            {
                "table_name": t.table_name,
                "business_domain": t.business_domain,
                "description": t.description,
            }
            for t in db.query(DictTable).all()
        ]
        fields = [
            {
                "table_name": f.table.table_name if f.table else "",
                "field_name": f.field_name,
                "field_type": f.field_type,
                "comment": f.comment,
                "calibre": f.calibre,
                "synonyms": f.synonyms or [],
            }
            for f in db.query(DictField).all()
        ]
        return tables, fields
    finally:
        db.close()


def _dictionary_field_provider() -> list[dict]:
    """读取 dict_field（含 id），供 DictionaryIndexer 构建索引。"""
    from finrag.models import DictField

    db = SessionLocal()
    try:
        return [
            {
                "id": f.id,
                "table_name": f.table.table_name if f.table else "",
                "field_name": f.field_name,
                "field_type": f.field_type,
                "comment": f.comment,
                "calibre": f.calibre,
                "synonyms": f.synonyms or [],
            }
            for f in db.query(DictField).all()
        ]
    finally:
        db.close()


@lru_cache
def get_dictionary_vector_store() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        host=s.qdrant_host,
        port=s.qdrant_port,
        collection=s.qdrant_dict_collection,
        vector_size=s.vector_size,
    )


@lru_cache
def get_dictionary_bm25() -> BM25Index:
    """字典专用 BM25 索引（与文档检索隔离）。"""
    return BM25Index()


@lru_cache
def get_dictionary_retriever() -> HybridRetriever:
    s = get_settings()
    return HybridRetriever(
        vector_store=get_dictionary_vector_store(),
        bm25_index=get_dictionary_bm25(),
        rrf_k=s.rrf_k,
        retrieve_top_k=s.retrieve_top_k,
        vector_weight=s.rrf_vector_weight,
        bm25_weight=s.rrf_bm25_weight,
    )


@lru_cache
def get_dictionary_indexer() -> DictionaryIndexer:
    return DictionaryIndexer(
        embedding=get_embedding(),
        vector_store=get_dictionary_vector_store(),
        bm25_index=get_dictionary_bm25(),
        field_provider=_dictionary_field_provider,
    )


@lru_cache
def get_schema_linker() -> SchemaLinker:
    return SchemaLinker(
        table_provider=_dictionary_provider,
        retriever=get_dictionary_retriever(),
        embedding=get_embedding(),
        indexer=get_dictionary_indexer(),
    )


@lru_cache
def get_mcp_executor() -> McpExecutor:
    s = get_settings()
    # 直连兜底：MCP 不可用（未启用/未配置/连接失败）时的执行器，也是 HttpMcpExecutor 的运行时降级目标
    direct = DbDirectExecutor(
        create_engine(s.business_db_url or s.db_url, pool_pre_ping=True)
    )
    if s.mcp_enabled:
        if s.mcp_server_url:
            from finrag.core.mcp_executor import HttpMcpExecutor

            # 连接类异常（Server 挂/网络断）在执行时自动降级 direct；SQL 业务错误照常抛给 NL2SQL 自修正
            return HttpMcpExecutor(
                s.mcp_server_url,
                timeout=s.mcp_timeout,
                api_key=s.mcp_api_key,
                fallback=direct,
            )
        if s.mcp_server_command:
            from finrag.core.mcp_executor import StdioMcpExecutor

            return StdioMcpExecutor(s.mcp_server_command, timeout=s.mcp_timeout)
        logger.warning("mcp_enabled_but_no_endpoint_fallback_direct")
    return direct


@lru_cache
def get_ingest_pipeline() -> IngestPipeline:
    # 解析器不在此固定：IngestPipeline 按 doc.file_type 动态选择（同步/异步路径统一）
    s = get_settings()
    return IngestPipeline(
        chunker=create_chunker(s),
        embedding=get_embedding(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25(),
    )


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    s = get_settings()
    return RAGPipeline(
        retriever=get_retriever(),
        reranker=get_reranker(),
        llm_gateway=get_llm_gateway(),
        rerank_top_k=s.rerank_top_k,
    )


@lru_cache
def get_nl2sql_pipeline() -> NL2SQLPipeline:
    return NL2SQLPipeline(
        schema_linker=get_schema_linker(),
        llm_gateway=get_llm_gateway(),
        executor=get_mcp_executor(),
        max_rows=get_settings().sql_max_rows,
    )


@lru_cache
def get_dictionary_pipeline() -> DictionaryPipeline:
    return DictionaryPipeline(schema_linker=get_schema_linker(), llm_gateway=get_llm_gateway())
