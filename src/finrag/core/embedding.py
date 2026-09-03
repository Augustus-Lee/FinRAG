"""Embedding 提供者：本地 BGE（延迟导入 sentence-transformers）/ 云端 OpenAI 兼容。

重量级依赖通过延迟导入隔离，框架层冒烟测试不依赖 GPU/torch。
"""

from abc import ABC, abstractmethod

import httpx

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.embedding")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class LocalBgeEmbedding(EmbeddingProvider):
    """本地 BGE 系列 embedding（sentence-transformers，延迟导入）。"""

    def __init__(self, model_name: str, dimension: int = 1024) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # 延迟导入

            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class ApiEmbedding(EmbeddingProvider):
    """云端 OpenAI 兼容 embedding 端点。"""

    def __init__(self, base_url: str, api_key: str, model: str, dimension: int = 1024) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        # 持久化连接 + 直连模式（不走系统代理）
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            trust_env=False,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._http.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts},
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
        )
        resp.raise_for_status()
        data = resp.json()
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """按配置构建 embedding 提供者（云模式模型名独立配置，为空回退 embedding_model）。"""
    if settings.embedding_provider == "api":
        return ApiEmbedding(
            base_url=settings.embedding_api_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_api_model or settings.embedding_model,
            dimension=settings.vector_size,
        )
    return LocalBgeEmbedding(model_name=settings.embedding_model, dimension=settings.vector_size)
