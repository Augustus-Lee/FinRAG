"""知识库服务：文档 CRUD 与摄入任务派发。"""

from sqlalchemy.orm import Session

from finrag import container
from finrag.logging import get_logger
from finrag.models import KBCategory, KBDocument
from finrag.utils.errors import NotFoundError

logger = get_logger("finrag.knowledge_service")


class KnowledgeService:
    def create_category(self, db: Session, name: str, description: str, owner_id: int = 1) -> KBCategory:
        cat = KBCategory(name=name, description=description, owner_id=owner_id)
        db.add(cat)
        db.commit()
        db.refresh(cat)
        return cat

    def create_document(
        self, db: Session, kb_id: int, name: str, file_type: str, file_path: str, owner_id: int = 1
    ) -> KBDocument:
        """创建文档记录（pending）并触发异步摄入（Celery，不可用时同步执行）。"""
        if not db.get(KBCategory, kb_id):
            raise NotFoundError(f"知识库不存在: kb_id={kb_id}")

        doc = KBDocument(
            kb_id=kb_id, name=name, file_type=file_type, file_path=file_path, owner_id=owner_id
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            from finrag.tasks.ingest_tasks import ingest_document_task

            ingest_document_task.delay(doc.id, file_path)
            logger.info("ingest_task_dispatched", doc_id=doc.id)
        except Exception:
            # Celery 不可用（如未启动 worker）：同步降级执行
            logger.warning("ingest_task_unavailable_fallback_sync", doc_id=doc.id)
            pipeline = container.get_ingest_pipeline()
            pipeline.ingest_document(db, doc, file_path)
        return doc

    def get_document(self, db: Session, doc_id: int) -> KBDocument:
        doc = db.get(KBDocument, doc_id)
        if not doc:
            raise NotFoundError(f"文档不存在: doc_id={doc_id}")
        return doc

    def delete_document(self, db: Session, doc_id: int) -> None:
        """删除文档：同时清理向量（point 前缀）与 BM25 索引。"""
        doc = self.get_document(db, doc_id)
        from finrag.models import KBChunk

        chunks = db.query(KBChunk).filter(KBChunk.doc_id == doc_id).all()
        point_ids = [c.vector_id for c in chunks if c.vector_id]
        if point_ids:
            container.get_vector_store().delete_by_ids(point_ids)
        for c in chunks:
            db.delete(c)
        db.delete(doc)
        db.commit()
        logger.info("document_deleted", doc_id=doc_id, chunks=len(chunks))
