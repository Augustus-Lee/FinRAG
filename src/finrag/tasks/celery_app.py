"""Celery 应用（文档处理异步化）。"""

from celery import Celery

from finrag.config import get_settings

settings = get_settings()

celery_app = Celery(
    "finrag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["finrag.tasks.ingest_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
