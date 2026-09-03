"""双模 LLM 网关（自研核心组件）。

- 云端：OpenAI 兼容协议（DeepSeek / 通义 / OpenAI ...）
- 本地：Ollama / vLLM（同样暴露 OpenAI 兼容端点）
统一封装超时、重试、模式选择（cloud / local / auto），对外暴露 chat / stream_chat 接口。
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.llm_gateway")

# SSE 结束标记
_DONE = "[DONE]"
_DATA_PREFIX = "data:"


class LLMClient(ABC):
    """统一的 LLM 客户端协议。"""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        """执行一次对话补全，返回文本（非流式，整段等待）。"""

    @abstractmethod
    def stream_chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        """以流式方式调用 LLM，逐 chunk 读取保持连接活跃、规避 read 超时，
        最终聚合为完整文本返回（对外契约与 chat() 一致）。"""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """cloud | local"""


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容端点客户端（云端与本地 Ollama/vLLM 通用）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        mode: str,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._mode = mode
        self._timeout = timeout
        self._max_retries = max_retries
        # 持久化连接池：避免每次请求新建 TCP 连接（减少握手延迟）
        # 分离超时：connect 10s 快速失败，read 给足时间等 LLM 生成
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
            trust_env=False,  # 云端 API 走直连，不经过系统代理
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def mode(self) -> str:
        return self._mode

    def chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict = {"model": self._model, "messages": messages, "temperature": temperature}
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # 网络/超时/5xx 重试
                last_err = exc
                logger.warning(
                    "llm_chat_retry",
                    attempt=attempt + 1,
                    model=self._model,
                    error=str(exc)[:200],
                )
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败: {last_err}")

    def stream_chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        """流式调用：stream=true + SSE 逐 chunk 读取，规避长生成 read 超时。

        httpx 在流式模式下 read 超时按「相邻 chunk 间隔」计时，只要 token 持续到达
        （间隔远小于 llm_timeout）就不会触发 ReadTimeout；最终聚合 delta.content 返回完整文本。
        """
        url = f"{self._base_url}/chat/completions"
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            buf: list[str] = []
            chunks = 0
            first_token_ms: float | None = None
            try:
                start = time.perf_counter()
                # stream() 上下文管理器：进入时建连，退出时关闭流；iter_lines() 逐行吐 SSE
                with self._http.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()  # 非 2xx（含鉴权失败）在此抛出，进入重试
                    for line in resp.iter_lines():
                        # SSE 行：`data: {...}` / `data: [DONE]` / 空行 / `: keep-alive` 注释
                        if not line or not line.startswith(_DATA_PREFIX):
                            continue
                        data = line[len(_DATA_PREFIX):].strip()
                        if data == _DONE:
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            # 个别 provider 偶发非 JSON 心跳行，跳过保活
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}) or {}
                        text_piece = delta.get("content")
                        if text_piece:
                            if first_token_ms is None:
                                first_token_ms = round((time.perf_counter() - start) * 1000, 1)
                            buf.append(text_piece)
                            chunks += 1
                # TTFT / chunk 数通过结构化日志观测（与 LLMGateway 顶层 latency 互补）
                logger.info(
                    "llm_stream_chunk_ok",
                    model=self._model,
                    chunks=chunks,
                    first_token_ms=first_token_ms,
                )
                return "".join(buf)
            except Exception as exc:  # 连接错误 / 非 2xx / 流中断：整次重试
                last_err = exc
                logger.warning(
                    "llm_stream_retry",
                    attempt=attempt + 1,
                    model=self._model,
                    error=str(exc)[:200],
                    chunks=chunks,
                )
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败: {last_err}")


class LLMGateway:
    """网关入口：按配置模式选择云端或本地客户端。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cloud = OpenAICompatClient(
            base_url=settings.llm_cloud_base_url,
            api_key=settings.llm_cloud_api_key,
            model=settings.llm_cloud_model,
            mode="cloud",
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._local = OpenAICompatClient(
            base_url=settings.llm_local_base_url,
            api_key="",
            model=settings.llm_local_model,
            mode="local",
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def get_client(self) -> LLMClient:
        mode = self._settings.llm_mode
        if mode == "cloud":
            return self._cloud
        if mode == "local":
            return self._local
        # auto：优先云端（有 key），否则本地
        return self._cloud if self._settings.llm_cloud_api_key else self._local

    def chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        client = self.get_client()
        start = time.perf_counter()
        text = client.chat(messages, temperature=temperature, **kwargs)
        logger.info(
            "llm_chat_ok",
            mode=client.mode,
            model=client.model_name,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return text

    def stream_chat(self, messages: list[dict], temperature: float = 0.1, **kwargs: Any) -> str:
        """流式入口（knowledge 场景用）。

        开关收敛在此：llm_stream_enabled=False 或 provider 不支持 SSE 时，
        直接回退到非流式 chat()，调用方无感。
        """
        if not self._settings.llm_stream_enabled:
            return self.chat(messages, temperature=temperature, **kwargs)
        client = self.get_client()
        start = time.perf_counter()
        text = client.stream_chat(messages, temperature=temperature, **kwargs)
        logger.info(
            "llm_stream_ok",
            mode=client.mode,
            model=client.model_name,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return text
