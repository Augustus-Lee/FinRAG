"""重排序器抽象 + 本地 BGE Reranker（延迟导入）+ 云端 API Reranker + Noop 兜底。

双模式设计：
- local: BGE CrossEncoder（sentence-transformers，延迟导入，无 torch 环境不加载）
- api:   云端 rerank 端点，支持两种主流格式：
  - cohere:    扁平 POST /rerank（Jina / 硅基流动 / Cohere / 阿里 qwen3-rerank compatible-api）
  - dashscope: 嵌套 POST /text-rerank/text-rerank（阿里 gte-rerank-v2 / qwen3-vl-rerank 原生接口）
"""

from abc import ABC, abstractmethod

import httpx

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.reranker")


class Reranker(ABC):
    """对检索候选做精排，返回与输入等长的相关性分数（越大越相关）。"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class NoopReranker(Reranker):
    """原序返回（未启用 rerank 时的兜底）。"""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(len(documents) - i) for i in range(len(documents))]


class BGEReranker(Reranker):
    """本地 BGE Reranker（sentence-transformers CrossEncoder，延迟导入）。"""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # 延迟导入

            logger.info("loading_rerank_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        model = self._load()
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        return [float(s) for s in scores]


class ApiReranker(Reranker):
    """云端 rerank 端点，支持 cohere / dashscope 两种格式。

    cohere 格式（Jina / 硅基流动 / Cohere / 阿里 qwen3-rerank compatible-api）:
        请求: POST {base_url}/rerank  {"model", "query", "documents", "top_n"}
        响应: {"results": [{"index": i, "relevance_score": s}, ...]}

    dashscope 格式（阿里 gte-rerank-v2 / qwen3-vl-rerank 原生接口）:
        请求: POST {base_url}/text-rerank/text-rerank
              {"model", "input": {"query", "documents"}, "parameters": {"top_n"}}
        响应: {"output": {"results": [{"index": i, "relevance_score": s}, ...]}}

    契约：返回与输入等长的分数列表；API 未返回（top_n 截断）的文档填 0.0。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        fmt: str = "cohere",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._fmt = fmt
        self._timeout = timeout
        # 持久化连接 + 直连模式（不走系统代理）
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
            trust_env=False,
        )

    def _build_request(self, query: str, documents: list[str]) -> tuple[str, dict]:
        n = len(documents)
        if self._fmt == "dashscope":
            path = "/text-rerank/text-rerank"
            body = {
                "model": self._model,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": n, "return_documents": False},
            }
        else:
            path = "/rerank"
            body = {"model": self._model, "query": query, "documents": documents, "top_n": n}
        return path, body

    def _parse_scores(self, resp_json: dict, n: int) -> list[float]:
        if self._fmt == "dashscope":
            results = resp_json.get("output", {}).get("results", [])
        else:
            results = resp_json.get("results", [])
        scores = [0.0] * n
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < n:
                scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        path, body = self._build_request(query, documents)
        # base_url 已含端点路径时不重复拼接（如 .../v1/rerank → 不再拼 /rerank）
        if self._base_url.endswith(path):
            url = self._base_url
        else:
            url = f"{self._base_url.rstrip('/')}{path}"
        resp = self._http.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
        )
        resp.raise_for_status()
        return self._parse_scores(resp.json(), len(documents))


def create_reranker(settings: Settings) -> Reranker:
    """按配置构建 reranker（与 create_embedding_provider 对称的工厂）。"""
    if not settings.rerank_enabled:
        return NoopReranker()
    if settings.rerank_provider == "api":
        if not settings.rerank_api_base_url:
            logger.warning("rerank_api_base_url_empty_fallback_local", provider=settings.rerank_provider)
            return BGEReranker(settings.rerank_model)
        return ApiReranker(
            base_url=settings.rerank_api_base_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_api_model or settings.rerank_model,
            fmt=settings.rerank_api_format,
        )
    return BGEReranker(settings.rerank_model)
