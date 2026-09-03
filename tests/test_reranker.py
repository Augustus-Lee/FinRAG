"""Reranker 双模式（local / api）单元测试。

不依赖 torch / sentence-transformers / 真实网络：云端调用通过 mock httpx.Client.post 验证契约。
"""

from finrag.config import Settings
from finrag.core.reranker import ApiReranker, BGEReranker, NoopReranker, create_reranker


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _patch_http_post(reranker: ApiReranker, monkeypatch, captured: dict | None = None):
    """把 ApiReranker 内部 httpx.Client.post 替换为 fake，可选收集调用参数。"""

    def fake_post(url, json=None, headers=None, **kwargs):
        if captured is not None:
            captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(
            {"results": [{"index": 1, "relevance_score": 0.98}, {"index": 2, "relevance_score": 0.42}]}
        )

    monkeypatch.setattr(reranker._http, "post", fake_post)


def test_noop_returns_descending_scores():
    scores = NoopReranker().rerank("q", ["a", "b", "c"])
    assert scores == [3.0, 2.0, 1.0]
    assert NoopReranker().rerank("q", []) == []


def test_api_reranker_maps_scores_by_index(monkeypatch):
    captured: dict = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(
            {"results": [{"index": 1, "relevance_score": 0.98}, {"index": 2, "relevance_score": 0.42}]}
        )

    reranker = ApiReranker(base_url="https://api.example.com/v1/", api_key="sk-test", model="bge-reranker-v2-m3")
    monkeypatch.setattr(reranker._http, "post", fake_post)

    docs = ["文档零", "文档一", "文档二"]
    scores = reranker.rerank("查询", docs)

    assert scores == [0.0, 0.98, 0.42]
    assert captured["url"] == "https://api.example.com/v1/rerank"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    body = captured["json"]
    assert body["model"] == "bge-reranker-v2-m3"
    assert body["query"] == "查询"
    assert body["documents"] == docs
    assert body["top_n"] == 3


def test_api_reranker_empty_documents_short_circuits(monkeypatch):
    called = []

    def fake_post(*args, **kwargs):
        called.append(1)
        return _FakeResponse({"results": []})

    reranker = ApiReranker("https://api.example.com", "k", "m")
    monkeypatch.setattr(reranker._http, "post", fake_post)
    assert reranker.rerank("q", []) == []
    assert called == []


def test_api_reranker_base_url_with_full_path_no_duplicate(monkeypatch):
    """base_url 已含 /rerank 时不应拼成 /rerank/rerank（真实坑：硅基流动联调 404）。"""
    captured: dict = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured["url"] = url
        return _FakeResponse({"results": [{"index": 0, "relevance_score": 0.5}]})

    reranker = ApiReranker(base_url="https://api.siliconflow.cn/v1/rerank", api_key="k", model="m")
    monkeypatch.setattr(reranker._http, "post", fake_post)
    reranker.rerank("q", ["d1"])
    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"


def test_create_reranker_local():
    r = create_reranker(Settings(rerank_enabled=True, rerank_provider="local", rerank_model="BAAI/bge-reranker-v2-m3"))
    assert isinstance(r, BGEReranker)


def test_create_reranker_api_with_model_override():
    r = create_reranker(
        Settings(
            rerank_enabled=True,
            rerank_provider="api",
            rerank_api_base_url="https://api.siliconflow.cn/v1",
            rerank_api_key="sk-x",
            rerank_api_model="Qwen/Qwen3-Reranker-8B",
        )
    )
    assert isinstance(r, ApiReranker)
    assert r._model == "Qwen/Qwen3-Reranker-8B"


def test_create_reranker_api_model_falls_back():
    r = create_reranker(
        Settings(
            rerank_enabled=True,
            rerank_provider="api",
            rerank_api_base_url="https://api.jina.ai/v1",
            rerank_model="BAAI/bge-reranker-v2-m3",
        )
    )
    assert isinstance(r, ApiReranker)
    assert r._model == "BAAI/bge-reranker-v2-m3"


def test_create_reranker_api_without_url_falls_back_local():
    r = create_reranker(Settings(rerank_enabled=True, rerank_provider="api", rerank_api_base_url=""))
    assert isinstance(r, BGEReranker)


def test_create_reranker_disabled_returns_noop():
    r = create_reranker(Settings(rerank_enabled=False, rerank_provider="api"))
    assert isinstance(r, NoopReranker)


# ---- dashscope 格式（阿里 gte-rerank-v2 原生接口）----


def test_api_reranker_dashscope_request_body(monkeypatch):
    """gte-rerank-v2 原生接口：请求体嵌套 input/parameters，路径 /text-rerank/text-rerank。"""
    captured: dict = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(
            {"output": {"results": [{"index": 0, "relevance_score": 0.91}, {"index": 1, "relevance_score": 0.23}]}}
        )

    reranker = ApiReranker(
        base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank",
        api_key="sk-dash",
        model="gte-rerank-v2",
        fmt="dashscope",
    )
    monkeypatch.setattr(reranker._http, "post", fake_post)

    docs = ["文档甲", "文档乙"]
    scores = reranker.rerank("查询", docs)

    assert scores == [0.91, 0.23]
    assert captured["url"] == "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    body = captured["json"]
    assert body["model"] == "gte-rerank-v2"
    assert body["input"]["query"] == "查询"
    assert body["input"]["documents"] == docs
    assert body["parameters"]["top_n"] == 2
    assert body["parameters"]["return_documents"] is False


def test_api_reranker_dashscope_missing_scores_fill_zero(monkeypatch):
    """dashscope 响应只返回 top_n 条，未返回的文档应填 0.0。"""

    def fake_post(*args, **kwargs):
        return _FakeResponse({"output": {"results": [{"index": 2, "relevance_score": 0.88}]}})

    reranker = ApiReranker("https://dashscope.aliyuncs.com/api/v1/services/rerank", "k", "gte-rerank-v2", fmt="dashscope")
    monkeypatch.setattr(reranker._http, "post", fake_post)
    scores = reranker.rerank("q", ["a", "b", "c"])
    assert scores == [0.0, 0.0, 0.88]


def test_create_reranker_dashscope_format():
    """工厂正确装配 dashscope 格式。"""
    r = create_reranker(
        Settings(
            rerank_enabled=True,
            rerank_provider="api",
            rerank_api_base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank",
            rerank_api_key="sk-dash",
            rerank_api_model="gte-rerank-v2",
            rerank_api_format="dashscope",
        )
    )
    assert isinstance(r, ApiReranker)
    assert r._fmt == "dashscope"
    assert r._model == "gte-rerank-v2"
