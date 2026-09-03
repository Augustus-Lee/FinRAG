"""MCP 执行器抽象（智能问数执行层，可插拔）。

安全边界设计：SQL 安全校验（SELECT-only/注入防护/条数限制）由 MCP Server 端管控，执行层原样透传 SQL。
- DbDirectExecutor：本地直连数据库执行（开发/受信环境 fallback；无本地校验，SQL 原样执行）
- HttpMcpExecutor：通过 MCP 协议（streamable-http）调用外部 DB MCP Server，生产主路径
- StdioMcpExecutor：stdio 子进程传输（预留，调用时抛 NotImplementedError）

HttpMcpExecutor 通用适配策略：
- 工具发现：连接后 list_tools()，按关键词优先级映射到三个语义槽位（不绑定特定 Server）
- 结果解析：兼容 columns/rows dict、list[dict]、纯文本三种主流返回格式
- 运行时降级：连接类异常自动切 fallback（注入的 DbDirectExecutor）；SQL 业务错误原样抛
  （走 NL2SQL 自修正重试——只有 SQL 错误才值得重试，连接错误重试无意义还会白等 timeout）
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import text

from finrag.logging import get_logger

logger = get_logger("finrag.mcp_executor")


@dataclass
class SqlResult:
    columns: list[str]
    rows: list[list]
    affected_rows: int = 0


class McpExecutor(ABC):
    """执行层统一接口。"""

    @abstractmethod
    def list_tables(self) -> list[str]: ...

    @abstractmethod
    def get_schema(self, table: str) -> dict: ...

    @abstractmethod
    def execute_sql(self, sql: str) -> SqlResult: ...


class DbDirectExecutor(McpExecutor):
    """本地数据库直连执行（开发/受信环境 fallback；无本地校验，SQL 原样执行）。"""

    def __init__(self, engine) -> None:
        self._engine = engine

    def list_tables(self) -> list[str]:
        with self._engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            return [row[0] for row in result]

    def get_schema(self, table: str) -> dict:
        with self._engine.connect() as conn:
            result = conn.execute(text(f"DESCRIBE `{table}`"))
            return {"columns": [{"name": row[0], "type": row[1]} for row in result]}

    def execute_sql(self, sql: str) -> SqlResult:
        with self._engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                columns = list(result.keys())
                rows = [list(row) for row in result.fetchall()]
                return SqlResult(columns=columns, rows=rows, affected_rows=len(rows))
            return SqlResult(columns=[], rows=[], affected_rows=result.rowcount or 0)


# ---------------------------------------------------------------------------
# HttpMcpExecutor：外部 DB MCP Server 接入（streamable-http 传输）
# ---------------------------------------------------------------------------

# 工具槽位 → 候选关键词（按优先级排序：越靠前越精确，避免歧义映射）
_TOOL_SLOTS: dict[str, list[list[str]]] = {
    "execute_sql": [["execute_sql", "execute-sql"], ["read_query", "read-query"], ["query"], ["sql"]],
    "list_tables": [["list_tables", "list-tables"], ["show_tables", "show-tables"], ["tables"]],
    "get_schema": [["get_schema", "get-schema"], ["describe"], ["schema"]],
}

# 连接类异常（触发降级）；其余视为 SQL 业务错误（走 NL2SQL 自修正重试）
_CONNECT_ERRORS: tuple = ()
try:  # httpx 与 anyio 均为 mcp SDK 传递依赖，缺库时降级判断退化为通用 Exception
    import httpx as _httpx

    _CONNECT_ERRORS = _CONNECT_ERRORS + (
        _httpx.ConnectError,
        _httpx.ConnectTimeout,
        _httpx.ReadTimeout,
        _httpx.WriteTimeout,
        _httpx.PoolTimeout,
        _httpx.RemoteProtocolError,
    )
    import anyio as _anyio  # noqa: F401

    _CONNECT_ERRORS = _CONNECT_ERRORS + (TimeoutError,)
except ImportError:  # pragma: no cover
    pass


def _norm(name: str) -> str:
    """工具名归一：大小写与连字符统一，便于关键词匹配。"""
    return name.lower().replace("-", "_")


class HttpMcpExecutor(McpExecutor):
    """通过 MCP 协议（streamable-http）调用外部 DB MCP Server。

    连接生命周期：单次调用独立会话（connect → call → close）。
    智能问数为对话级 QPS，无需连接池；换来无状态、无泄漏、实现简单。
    Pipeline 全同步，异步 SDK 经 asyncio.run 驱动（FastAPI sync 路由跑线程池，线程内安全）。
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        api_key: str = "",
        fallback: McpExecutor | None = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._fallback = fallback
        # 工具槽位映射（list_tools 发现后缓存；发现失败置 None 触发下次重试）
        self._tool_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # McpExecutor 接口
    # ------------------------------------------------------------------
    def list_tables(self) -> list[str]:
        raw = self._call_slot("list_tables")
        return self._parse_table_list(raw)

    def get_schema(self, table: str) -> dict:
        raw = self._call_slot("get_schema", {"table": table, "table_name": table})
        return self._parse_schema(raw)

    def execute_sql(self, sql: str) -> SqlResult:
        try:
            raw = self._call_slot("execute_sql", {"sql": sql, "query": sql})
        except (Exception, asyncio.CancelledError) as exc:
            # CancelledError 继承 BaseException：httpx 超时在 SDK 任务组内表现为取消，
            # 属连接类故障，同样触发降级
            if self._is_connect_error(exc):
                return self._fallback_execute(sql, exc)
            raise
        return self._parse_sql_result(raw)

    # ------------------------------------------------------------------
    # 内部：调用与解析
    # ------------------------------------------------------------------
    def _call_slot(self, slot: str, arguments: dict | None = None) -> object:
        """调用槽位对应的 MCP 工具，返回 tool result 的 text content。"""
        import time

        tool = self._resolve_tool(slot)
        start = time.perf_counter()
        try:
            result = self._invoke(tool, arguments or {})
        finally:
            logger.info(
                "mcp_tool_call",
                tool=tool,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        return result

    def _resolve_tool(self, slot: str) -> str:
        """确保工具映射已发现（惰性 + 失败后可重试）。"""
        if self._tool_map is None:
            self._tool_map = self._discover_tools()
        tool = self._tool_map.get(slot)
        if not tool:
            raise RuntimeError(
                f"MCP Server 缺少 {slot} 工具; 实际提供: {sorted(self._tool_map.values()) or '(无)'}"
            )
        return tool

    def _discover_tools(self) -> dict[str, str]:
        """list_tools 动态发现 + 关键词优先级映射（通用适配核心）。"""

        async def _list() -> list[str]:
            async with self._session() as (_, session):
                result = await session.list_tools()
                return [t.name for t in result.tools]

        names = self._run(_list())
        tool_map: dict[str, str] = {}
        for slot, keyword_groups in _TOOL_SLOTS.items():
            for keywords in keyword_groups:
                # 精确优先：先找全名等于候选的工具，再退化为包含匹配
                exact = [n for n in names if _norm(n) == keywords[0]]
                if exact:
                    tool_map[slot] = exact[0]
                    break
                partial = [n for n in names if all(k in _norm(n) for k in keywords)]
                if partial:
                    tool_map[slot] = sorted(partial)[0]  # 稳定选择，可复现
                    break
        logger.info("mcp_tools_discovered", server=self._server_url, tools=names, mapped=tool_map)
        return tool_map

    def _invoke(self, tool: str, arguments: dict) -> object:
        """调用工具并提取 text content（isError 转为 RuntimeError，保持业务错误语义）。"""

        async def _call():
            async with self._session() as (_, session):
                result = await session.call_tool(tool, arguments=arguments)
                if getattr(result, "isError", False):
                    raise _ToolError(self._extract_text(result))
                return self._extract_text(result)

        return self._run(_call())

    def _session(self):
        """MCP streamable-http 会话上下文（官方嵌套模式：transport → ClientSession）。

        返回 (transport_cm, session) 的 async with 包装；httpx2.AsyncClient 携带
        超时与 Bearer 鉴权（SDK 2.x 经 http_client 参数注入）。
        """

        async def _enter():
            import httpx

            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=min(10.0, self._timeout)),
                headers=headers,
            )
            transport_cm = streamable_http_client(self._server_url, http_client=http_client)
            streams = await transport_cm.__aenter__()
            # SDK 2.x 返回 (read, write) 2 元组；1.x 为 (read, write, get_session_id) 3 元组
            if len(streams) == 3:
                read, write, _ = streams
            else:
                read, write = streams
            session_cm = ClientSession(read, write)
            session = await session_cm.__aenter__()
            await session.initialize()
            return transport_cm, session_cm, session

        async def _exit(stack):
            transport_cm, session_cm, _ = stack
            try:
                await session_cm.__aexit__(None, None, None)
            finally:
                await transport_cm.__aexit__(None, None, None)

        class _SessionCtx:
            def __init__(self, enter, exit_) -> None:
                self._enter, self._exit = enter, exit_
                self._stack = None

            async def __aenter__(self):
                self._stack = await self._enter()
                return self._stack[0], self._stack[2]  # (transport_cm, session)

            async def __aexit__(self, exc_type, exc, tb):
                await self._exit(self._stack)

        return _SessionCtx(_enter, _exit)

    def _run(self, coro):
        """同步驱动异步调用（线程内无 event loop 时安全）。"""
        return asyncio.run(coro)

    @staticmethod
    def _extract_text(result) -> str:
        """取 tool result 首个 text content；无 text 时序列化兜底。"""
        content = getattr(result, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                return text
        return json.dumps({"raw": str(result)}, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # 解析：三种主流返回格式
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sql_result(raw: object) -> SqlResult:
        data = _try_json(raw)
        if isinstance(data, dict) and isinstance(data.get("columns"), list) and isinstance(data.get("rows"), list):
            return SqlResult(
                columns=[str(c) for c in data["columns"]],
                rows=[[v for v in row] for row in data["rows"]],
                affected_rows=len(data["rows"]),
            )
        if isinstance(data, list) and data and isinstance(data[0], dict):
            columns = [str(k) for k in data[0].keys()]
            rows = [[row.get(c) for c in columns] for row in data]
            return SqlResult(columns=columns, rows=rows, affected_rows=len(rows))
        # 纯文本保底：不炸，至少把结果带回去
        text = raw if isinstance(raw, str) else str(raw)
        return SqlResult(columns=["result"], rows=[[text]], affected_rows=1)

    @staticmethod
    def _parse_table_list(raw: object) -> list[str]:
        data = _try_json(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            for key in ("tables", "table_names", "results"):
                if isinstance(data.get(key), list):
                    return [str(x) for x in data[key]]
        return [str(raw)]

    @staticmethod
    def _parse_schema(raw: object) -> dict:
        data = _try_json(raw)
        if isinstance(data, dict):
            cols = data.get("columns")
            if isinstance(cols, list):
                normalized = [
                    {"name": c.get("name", c.get("Field", str(c))), "type": c.get("type", c.get("Type", ""))}
                    if isinstance(c, dict)
                    else {"name": str(c), "type": ""}
                    for c in cols
                ]
                return {"columns": normalized}
        return {"columns": [], "raw": str(raw)[:500]}

    # ------------------------------------------------------------------
    # 降级
    # ------------------------------------------------------------------
    @staticmethod
    def _is_connect_error(exc: BaseException) -> bool:
        if isinstance(exc, _ToolError):
            return False  # 工具返回的业务错误（isError），值得走 SQL 自修正重试
        if isinstance(exc, asyncio.CancelledError):
            return True  # SDK 任务组内超时/连接失败的统一表现
        if _CONNECT_ERRORS and isinstance(exc, _CONNECT_ERRORS):
            return True
        # 其余异常按业务错误处理，保守不降级
        return False

    def _fallback_execute(self, sql: str, cause: Exception) -> SqlResult:
        if self._fallback is None:
            raise cause
        logger.warning("mcp_fallback_to_direct", error=str(cause)[:200], sql=sql[:120])
        return self._fallback.execute_sql(sql)


class _ToolError(RuntimeError):
    """MCP 工具执行返回 isError（SQL 语义错误），非连接问题。"""


def _try_json(raw: object) -> object:
    """text → JSON 尝试解析（失败返回原字符串，由调用方走文本保底）。"""
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


class StdioMcpExecutor(McpExecutor):
    """stdio 子进程传输（预留）。构造轻量；实际调用时抛 NotImplementedError，
    与 HttpMcpExecutor 的「调用时失败才降级」语义统一。"""

    def __init__(self, server_command: str, timeout: float = 30.0) -> None:
        self._server_command = server_command
        self._timeout = timeout

    def _not_implemented(self) -> None:
        raise NotImplementedError(
            f"StdioMcpExecutor 未实现（command={self._server_command}）；"
            "请配置 FINRAG_MCP_SERVER_URL 使用 HttpMcpExecutor，或保持 mcp_enabled=false 走 DbDirectExecutor"
        )

    def list_tables(self) -> list[str]:
        self._not_implemented()

    def get_schema(self, table: str) -> dict:
        self._not_implemented()

    def execute_sql(self, sql: str) -> SqlResult:
        self._not_implemented()
