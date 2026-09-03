"""健康检查：探活与依赖服务状态。"""

from fastapi import APIRouter

from finrag import container

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """服务探活；附带 Qdrant 健康状态。"""
    qdrant_ok = container.get_vector_store().healthcheck()
    return {"status": "ok", "qdrant": "up" if qdrant_ok else "down"}
