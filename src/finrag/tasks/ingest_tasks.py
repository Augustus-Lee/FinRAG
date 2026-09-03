"""文档摄入异步任务。"""

from finrag import container
from finrag.db.session import SessionLocal
from finrag.logging import get_logger
from finrag.models import KBDocument
from finrag.tasks.celery_app import celery_app

logger = get_logger("finrag.tasks.ingest")


@celery_app.task(name="finrag.ingest_document")
def ingest_document_task(doc_id: int, file_path: str) -> dict:
    """Celery 任务：同步执行摄入流水线（配合 task_acks_late 保证至少一次）。

    解析器按 doc.file_type 在 IngestPipeline.ingest_document 内部选择，
    这里不再篡改缓存单例的私有属性。
    """
    db = SessionLocal()
    try:
        doc = db.get(KBDocument, doc_id)
        if not doc:
            raise ValueError(f"文档不存在: doc_id={doc_id}")
        pipeline = container.get_ingest_pipeline()
        result = pipeline.ingest_document(db, doc, file_path)
        return {"doc_id": doc_id, "status": result.status, "chunks": result.chunk_count}
    finally:
        db.close()
