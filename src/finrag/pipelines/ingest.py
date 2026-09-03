"""文档摄入流水线：解析 → 切分 → Embedding → 向量库 + BM25 + 元数据入库。"""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from finrag.core.bm25 import BM25Index
from finrag.core.chunker import Chunker, ParsedDocument
from finrag.core.document_parser import DocumentParser, create_parser
from finrag.core.embedding import EmbeddingProvider
from finrag.core.vectorstore import VectorPoint, VectorStore
from finrag.logging import get_logger
from finrag.models import KBDocument

logger = get_logger("finrag.ingest")


def _vector_point_id(doc_id: int, seq: int) -> str:
    """生成确定性的向量 point id。

    Qdrant 的 point id 仅接受无符号整数或 UUID（字符串别名会被 400 拒绝）；
    uuid5 保证 (doc_id, seq) 到 UUID 的稳定映射：重复摄入幂等覆盖、删除可精确命中，
    同时 BM25 doc_id 与 KBChunk.vector_id 复用同一 id，保证 RRF 融合时两路能聚合到同一 chunk。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"finrag:doc{doc_id}_c{seq}"))


@dataclass
class IngestResult:
    doc_id: int
    chunk_count: int
    status: str


class IngestPipeline:
    """文档处理主链路（异步任务中调用）。"""

    def __init__(
        self,
        chunker: Chunker,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        parser: DocumentParser | None = None,
    ) -> None:
        # parser 为 None 时按 doc.file_type 动态选择（文件类型是文档的属性，
        # 同步降级与 Celery 异步两条路径由此统一）；非 None 供测试注入
        self._parser = parser
        self._chunker = chunker
        self._embedding = embedding
        self._vector_store = vector_store
        self._bm25 = bm25_index

    def ingest_document(self, db: Session, doc: KBDocument, file_path: str) -> IngestResult:
        """同步执行摄入（Celery 任务调用）。失败时更新文档状态为 failed。"""
        try:
            doc.status = "parsing"
            db.commit()

            parser = self._parser or create_parser(doc.file_type)
            parsed: ParsedDocument = parser.parse(file_path)
            chunks = self._chunker.split(parsed)
            if not chunks:
                raise ValueError("解析结果为空，无法切分")

            # Embedding（批量）
            texts = [c.content for c in chunks]
            vectors = self._embedding.embed(texts)

            # 向量入库
            points = [
                VectorPoint(
                    id=_vector_point_id(doc.id, idx),
                    vector=vectors[idx],
                    payload={
                        "doc_id": doc.id,
                        "chunk_id": idx,
                        "content": chunks[idx].content,
                        "section_path": chunks[idx].section_path,
                        # Qdrant payload 不接受 None，非表格块统一为空 dict
                        "table_meta": chunks[idx].table_meta or {},
                    },
                )
                for idx in range(len(chunks))
            ]
            self._vector_store.upsert(points)

            # BM25 索引（doc_id 与向量 point id 一致，RRF 融合才能聚合）
            self._bm25.add_batch(
                [(_vector_point_id(doc.id, idx), chunks[idx].content, None) for idx in range(len(chunks))]
            )

            # 元数据入库
            from finrag.models import KBChunk

            for idx, chunk in enumerate(chunks):
                db.add(
                    KBChunk(
                        doc_id=doc.id,
                        seq_no=idx,
                        content=chunk.content,
                        table_meta=chunk.table_meta,
                        section_path=chunk.section_path,
                        token_count=chunk.token_count,
                        vector_id=_vector_point_id(doc.id, idx),
                    )
                )

            doc.status = "ready"
            doc.chunk_count = len(chunks)
            db.commit()
            logger.info("ingest_done", doc_id=doc.id, chunks=len(chunks))
            return IngestResult(doc_id=doc.id, chunk_count=len(chunks), status="ready")
        except Exception as exc:
            db.rollback()
            doc.status = "failed"
            doc.error_msg = str(exc)[:500]
            db.commit()
            logger.error("ingest_failed", doc_id=doc.id, error=str(exc)[:300])
            raise
