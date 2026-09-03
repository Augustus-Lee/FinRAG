"""HttpMcpExecutor 单元测试：mock MCP ClientSession（不依赖真实 MCP Server）。

通过 monkeypatch 替换 _session() 注入 FakeSession，验证：
工具发现映射 / 三种返回格式解析 / 缺工具报错 / 连接失败降级 / 业务错误透传 / 工厂装配。
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from finrag.core.mcp_executor import DbDirectExecutor, HttpMcpExecutor, SqlResult


# ---------------------------------------------------------------------------
# Fakes：模拟 mcp SDK 的 list_tools / call_tool 返回结构
# ---------------------------------------------------------------------------


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _ToolListResult:
    def __init__(self, names: list[str]) -> None:
        self.tools = [_Tool(n) for n in names]


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _CallResult:
    def __init__(self, payload: object, is_error: bool = False) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.content = [_TextBlock(text)]
        self.isError = is_error


class FakeSession:
    """替代 HttpMcpExecutor._session() 的伪会话。"""

    def __init__(self, tools: list[str], responses: dict, *, fail_connect: Exception | None = None) -> None:
        self._tools = tools
        self._responses = responses  # tool_name -> _CallResult 或 Exception
        self.calls: list[tuple[str, dict]] = []
        self._fail_connect = fail_connect

    async def initialize(self) -> None:
        if self._fail_connect:
            raise self._fail_connect

    async def list_tools(self) -> _ToolListResult:
        if self._fail_connect:
            raise self._fail_connect
        return _ToolListResult(self._tools)

    async def call_tool(self, name: str, arguments: dict | None = None):
        self.calls.append((name, arguments or {}))
        resp = self._responses[name]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _install(monkeypatch, executor: HttpMcpExecutor, session: FakeSession) -> None:
    """把 executor 的会话构造替换为返回 FakeSession 的上下文。"""

    class _Ctx:
        async def __aenter__(self):
            if session._fail_connect:
                raise session._fail_connect
            return None, session

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(executor, "_session", lambda: _Ctx())


DBHUB_TOOLS = ["list_tables", "describe_table", "read_query"]
FULL_TOOLS = ["execute_sql", "list_tables", "get_schema"]


# ---------------------------------------------------------------------------
# 1. 工具发现与映射
# ---------------------------------------------------------------------------


def test_discovers_and_maps_tools(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    session = FakeSession(DBHUB_TOOLS, {})
    _install(monkeypatch, ex, session)

    tool_map = ex._discover_tools()
    # dbhub 风格：read_query 命中 execute_sql 槽（read_query 优先于泛化 query）
    assert tool_map["execute_sql"] == "read_query"
    assert tool_map["list_tables"] == "list_tables"
    assert tool_map["get_schema"] == "describe_table"


def test_exact_name_preferred_over_partial(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    # 同时存在 execute_sql（精确）与 execute_write_sql（部分匹配）：必须选精确
    session = FakeSession(["execute_sql", "execute_write_sql", "list_tables", "get_schema"], {})
    _install(monkeypatch, ex, session)

    tool_map = ex._discover_tools()
    assert tool_map["execute_sql"] == "execute_sql"


def test_tool_map_cached(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    session = FakeSession(FULL_TOOLS, {"list_tables": _CallResult(["t1", "t2"])})
    _install(monkeypatch, ex, session)

    assert ex.list_tables() == ["t1", "t2"]
    assert ex.list_tables() == ["t1", "t2"]
    # 工具调用两次，但 list_tools（发现）只在首次发生——映射已缓存
    assert len(session.calls) == 2
    assert all(name == "list_tables" for name, _ in session.calls)


# ---------------------------------------------------------------------------
# 2. 结果解析：三种主流格式
# ---------------------------------------------------------------------------


def test_execute_sql_parses_columns_rows_format(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    payload = {"columns": ["product", "amount"], "rows": [["A", 100], ["B", 200]]}
    session = FakeSession(FULL_TOOLS, {"execute_sql": _CallResult(payload)})
    _install(monkeypatch, ex, session)

    result = ex.execute_sql("SELECT ...")
    assert result.columns == ["product", "amount"]
    assert result.rows == [["A", 100], ["B", 200]]
    assert result.affected_rows == 2


def test_execute_sql_parses_list_of_dicts(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    payload = [{"product": "A", "amount": 100}, {"product": "B", "amount": 200}]
    session = FakeSession(FULL_TOOLS, {"execute_sql": _CallResult(payload)})
    _install(monkeypatch, ex, session)

    result = ex.execute_sql("SELECT ...")
    assert result.columns == ["product", "amount"]
    assert result.rows == [["A", 100], ["B", 200]]


def test_execute_sql_parses_plain_text(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    session = FakeSession(FULL_TOOLS, {"execute_sql": _CallResult("3 rows affected")})
    _install(monkeypatch, ex, session)

    result = ex.execute_sql("SELECT ...")
    assert result.columns == ["result"]
    assert result.rows == [["3 rows affected"]]


def test_get_schema_and_list_tables(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    session = FakeSession(
        FULL_TOOLS,
        {
            "list_tables": _CallResult({"tables": ["product_sales", "customer_account"]}),
            "get_schema": _CallResult({"columns": [{"name": "id", "type": "int"}, {"name": "name"}]}),
        },
    )
    _install(monkeypatch, ex, session)

    assert ex.list_tables() == ["product_sales", "customer_account"]
    schema = ex.get_schema("product_sales")
    assert schema["columns"][0] == {"name": "id", "type": "int"}
    assert schema["columns"][1] == {"name": "name", "type": ""}  # list[str] 字段补空 type


# ---------------------------------------------------------------------------
# 3. 缺工具报错（消息含实际工具清单）
# ---------------------------------------------------------------------------


def test_missing_tool_raises_with_tool_list(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    # 只有 list_tables，无 execute_sql 语义工具
    session = FakeSession(["list_tables", "health_check"], {})
    _install(monkeypatch, ex, session)

    with pytest.raises(RuntimeError, match="execute_sql"):
        ex.execute_sql("SELECT ...")


# ---------------------------------------------------------------------------
# 4. 连接失败 → 降级 fallback
# ---------------------------------------------------------------------------


def test_connection_error_falls_back_to_direct(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    fallback = MagicMock()
    fallback.execute_sql.return_value = SqlResult(columns=["ok"], rows=[[1]], affected_rows=1)
    ex._fallback = fallback

    session = FakeSession(FULL_TOOLS, {}, fail_connect=httpx.ConnectError("refused"))
    _install(monkeypatch, ex, session)

    result = ex.execute_sql("SELECT ...")
    assert result.columns == ["ok"]
    fallback.execute_sql.assert_called_once()  # 降级恰好一次


def test_no_fallback_reraises_connect_error(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")  # 未注入 fallback
    session = FakeSession(FULL_TOOLS, {}, fail_connect=httpx.ConnectError("refused"))
    _install(monkeypatch, ex, session)

    with pytest.raises(httpx.ConnectError):
        ex.execute_sql("SELECT ...")


# ---------------------------------------------------------------------------
# 5. SQL 业务错误（isError）→ 原样抛（走 NL2SQL 自修正），不降级
# ---------------------------------------------------------------------------


def test_sql_business_error_propagates(monkeypatch):
    ex = HttpMcpExecutor("http://mcp:8080/mcp")
    fallback = MagicMock()
    ex._fallback = fallback

    session = FakeSession(
        FULL_TOOLS, {"execute_sql": _CallResult("Unknown column 'x'", is_error=True)}
    )
    _install(monkeypatch, ex, session)

    with pytest.raises(RuntimeError, match="Unknown column"):
        ex.execute_sql("SELECT x FROM t")
    fallback.execute_sql.assert_not_called()  # 业务错误不降级


# ---------------------------------------------------------------------------
# 6. 容器工厂装配
# ---------------------------------------------------------------------------


def test_container_factory_prefers_http_when_configured(monkeypatch):
    from finrag import container
    from finrag.config import Settings

    settings = Settings()
    settings.mcp_enabled = True
    settings.mcp_server_url = "http://mcp-db:8080/mcp"
    settings.mcp_api_key = "sk-test"
    monkeypatch.setattr(container, "get_settings", lambda: settings)
    container.get_mcp_executor.cache_clear()

    executor = container.get_mcp_executor()
    try:
        assert isinstance(executor, HttpMcpExecutor)
        assert executor._server_url == "http://mcp-db:8080/mcp"
        assert executor._api_key == "sk-test"
        assert isinstance(executor._fallback, DbDirectExecutor)  # 注入了运行时降级
    finally:
        container.get_mcp_executor.cache_clear()


def test_container_factory_direct_when_disabled(monkeypatch):
    from finrag import container
    from finrag.config import Settings

    settings = Settings()
    settings.mcp_enabled = False
    monkeypatch.setattr(container, "get_settings", lambda: settings)
    container.get_mcp_executor.cache_clear()

    executor = container.get_mcp_executor()
    try:
        assert isinstance(executor, DbDirectExecutor)
    finally:
        container.get_mcp_executor.cache_clear()


def test_container_factory_direct_when_no_url(monkeypatch):
    from finrag import container
    from finrag.config import Settings

    settings = Settings()
    settings.mcp_enabled = True
    settings.mcp_server_url = ""  # 启用但未配 endpoint → 降级 direct
    monkeypatch.setattr(container, "get_settings", lambda: settings)
    container.get_mcp_executor.cache_clear()

    executor = container.get_mcp_executor()
    try:
        assert isinstance(executor, DbDirectExecutor)
    finally:
        container.get_mcp_executor.cache_clear()
