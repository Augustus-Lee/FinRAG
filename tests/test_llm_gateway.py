"""LLMGateway 流式改造单元测试。

不依赖真实网络：通过 fake 流式响应对象（带 __enter__/__exit__/raise_for_status/iter_lines）
替换 httpx.Client.stream，验证 SSE 解析、重试、降级逻辑。
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from finrag.config import Settings
from finrag.core.llm_gateway import LLMGateway, OpenAICompatClient


def _sse_line(content: str | None, **extra) -> str:
    """构造一行 OpenAI 兼容 SSE data 行。content=None 表示无文本（如首帧只带 role）。"""
    payload = {"choices": [{"delta": {"content": content, **extra}}]}
    return "data: " + json.dumps(payload, ensure_ascii=False)


class _FakeStreamResp:
    """模拟 httpx 流式响应：作为 with 语句的上下文管理器 + 提供 iter_lines。"""

    def __init__(self, lines: list[str], *, status_ok: bool = True) -> None:
        self._lines = list(lines)
        self._status_ok = status_ok

    def __enter__(self) -> "_FakeStreamResp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def iter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttp:
    """最小 httpx.Client 替身：仅实现 stream()。stream() 按 side_effect 返回 _FakeStreamResp。"""

    def __init__(self, responses) -> None:
        # responses: 单个 _FakeStreamResp 或列表（按调用顺序消费，用于重试测试）
        self._responses = responses if isinstance(responses, list) else [responses]
        self._idx = 0
        self.calls: list[dict] = []

    def stream(self, method, url, *, json=None, headers=None, **kw) -> _FakeStreamResp:
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        if self._idx >= len(self._responses):
            raise AssertionError("stream() called more times than configured")
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_client(max_retries: int = 1) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        mode="cloud",
        timeout=10.0,
        max_retries=max_retries,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """禁用重试退避的真实 sleep，保证测试零延迟。"""
    monkeypatch.setattr("finrag.core.llm_gateway.time.sleep", lambda s: None)


def test_stream_chat_aggregates_delta_content():
    client = _make_client(max_retries=0)
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),  # 首帧无 content
        _sse_line("你好"),
        _sse_line("，世界"),
        "data: [DONE]",
    ]
    client._http = _FakeHttp(_FakeStreamResp(lines))  # type: ignore[assignment]
    text = client.stream_chat([{"role": "user", "content": "hi"}])
    assert text == "你好，世界"


def test_stream_chat_skips_non_data_lines():
    client = _make_client(max_retries=0)
    lines = [
        "",  # 空行
        ": keep-alive",  # SSE 注释行
        _sse_line("OK"),
        "data: [DONE]",
    ]
    client._http = _FakeHttp(_FakeStreamResp(lines))  # type: ignore[assignment]
    assert client.stream_chat([]) == "OK"


def test_stream_chat_skips_non_json_heartbeat():
    client = _make_client(max_retries=0)
    lines = [
        "data: ping",  # 非 JSON 心跳行
        _sse_line("real"),
        "data: [DONE]",
    ]
    client._http = _FakeHttp(_FakeStreamResp(lines))  # type: ignore[assignment]
    assert client.stream_chat([]) == "real"


def test_stream_chat_sends_stream_true_in_payload():
    client = _make_client(max_retries=0)
    fake = _FakeHttp(_FakeStreamResp(["data: [DONE]"]))
    client._http = fake  # type: ignore[assignment]
    client.stream_chat([{"role": "user", "content": "q"}], temperature=0.2)
    assert fake.calls[0]["json"]["stream"] is True
    assert fake.calls[0]["json"]["temperature"] == 0.2
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer k"


def test_stream_chat_retries_on_connection_error():
    client = _make_client(max_retries=1)
    # 首次抛连接错误，第二次成功
    fake = _FakeHttp(
        [
            httpx.ConnectError("conn refused"),  # type: ignore[arg-type]
            _FakeStreamResp([_sse_line("recovered"), "data: [DONE]"]),
        ]
    )
    client._http = fake  # type: ignore[assignment]
    text = client.stream_chat([])
    assert text == "recovered"
    assert len(fake.calls) == 2  # 重试一次后成功


def test_stream_chat_raises_after_max_retries():
    client = _make_client(max_retries=1)
    fake = _FakeHttp(
        [
            httpx.ReadTimeout("t1"),  # type: ignore[arg-type]
            httpx.ReadTimeout("t2"),  # type: ignore[arg-type]
        ]
    )
    client._http = fake  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="LLM 调用失败"):
        client.stream_chat([])
    assert len(fake.calls) == 2  # 初次 + 1 次重试


def test_stream_chat_retries_on_non_2xx_status():
    client = _make_client(max_retries=1)
    fake = _FakeHttp(
        [
            _FakeStreamResp([], status_ok=False),  # 鉴权失败 → raise_for_status 抛错
            _FakeStreamResp([_sse_line("ok"), "data: [DONE]"]),
        ]
    )
    client._http = fake  # type: ignore[assignment]
    assert client.stream_chat([]) == "ok"


def test_gateway_stream_chat_delegates_to_client():
    settings = Settings()
    settings.llm_mode = "cloud"
    settings.llm_cloud_api_key = "k"
    settings.llm_stream_enabled = True
    gw = LLMGateway(settings)
    cloud = MagicMock()
    cloud.stream_chat.return_value = "streamed"
    cloud.mode = "cloud"
    cloud.model_name = "m"
    gw._cloud = cloud  # type: ignore[assignment]
    text = gw.stream_chat([{"role": "user", "content": "q"}])
    assert text == "streamed"
    cloud.stream_chat.assert_called_once()


def test_stream_disabled_falls_back_to_chat():
    settings = Settings()
    settings.llm_mode = "cloud"
    settings.llm_cloud_api_key = "k"
    settings.llm_stream_enabled = False
    gw = LLMGateway(settings)
    cloud = MagicMock()
    cloud.chat.return_value = "from-chat"
    cloud.mode = "cloud"
    cloud.model_name = "m"
    gw._cloud = cloud  # type: ignore[assignment]
    text = gw.stream_chat([{"role": "user", "content": "q"}])
    assert text == "from-chat"
    cloud.chat.assert_called_once()
    cloud.stream_chat.assert_not_called()
