"""向量存储抽象层 + Qdrant 实现。

抽象层是面试选型论证的关键：向量库不锁死单一实现，规模增长可切换 Milvus / ES。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.vectorstore")


@dataclass
class VectorPoint:
    """一条待写入向量记录。payload 保存 chunk 元数据。"""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """检索命中。"""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """向量存储统一接口。"""

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int, filter_: dict | None = None) -> list[SearchHit]: ...

    @abstractmethod
    def delete_by_ids(self, point_ids: list[str]) -> None: ...

    @abstractmethod
    def healthcheck(self) -> bool: ...


class QdrantVectorStore(VectorStore):
    """Qdrant 实现（Rust 内核，单机高性能，支持过滤与命名空间）。"""

    def __init__(self, host: str, port: int, collection: str, vector_size: int) -> None:
        self._collection = collection
        self._vector_size = vector_size
        self._client = None
        self._host, self._port = host, port

    def _lazy_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # 延迟导入

            self._client = QdrantClient(host=self._host, port=self._port)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams  # 延迟导入

        client = self._client
        existing = client.collection_exists(self._collection)
        if not existing:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", collection=self._collection, size=self._vector_size)

    def upsert(self, points: list[VectorPoint]) -> None:
        from qdrant_client.models import PointStruct  # 延迟导入

        client = self._lazy_client()
        batch = [
            PointStruct(id=point.id, vector=point.vector, payload=point.payload) for point in points
        ]
        client.upsert(collection_name=self._collection, points=batch)

    def search(self, vector: list[float], top_k: int, filter_: dict | None = None) -> list[SearchHit]:
        client = self._lazy_client()
        hits = client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        ).points
        return [SearchHit(id=hit.id, score=hit.score, payload=hit.payload or {}) for hit in hits]

    def delete_by_ids(self, point_ids: list[str]) -> None:
        client = self._lazy_client()
        if point_ids:
            client.delete(collection_name=self._collection, points_selector=point_ids)

    def healthcheck(self) -> bool:
        try:
            self._lazy_client().get_collection(self._collection)
            return True
        except Exception:
            return False


def create_vector_store(settings: Settings) -> VectorStore:
    return QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
        vector_size=settings.vector_size,
    )
