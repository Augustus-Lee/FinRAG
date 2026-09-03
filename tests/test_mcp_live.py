"""真实 MCP Server 集成测试（Windows 宿主机 8008，多库：MySQL / DolphinDB）。

网络拓扑：FinRAG 跑在 WSL2/容器内，MCP Server 在 Windows 宿主机监听 0.0.0.0:8008。
WSL2 访问 Windows 宿主走默认网关 IP（如 172.28.208.1）。

前置条件（Windows 侧，管理员 PowerShell 放行 WSL 子网入站）：
    New-NetFirewallRule -DisplayName "MCP 8008 for WSL" `
        -Direction Inbound -Protocol TCP -LocalPort 8008 -Action Allow

地址解析优先级：
    1. 环境变量 FINRAG_TEST_MCP_URL（如 http://172.28.208.1:8008/mcp）
    2. 自动取 WSL 默认网关，依次尝试 http://<gw>:8008/mcp 与 http://<gw>:8008
Server 不可达时整个模块自动 skip（不影响无 Server 的 CI/本地环境）。
"""

import asyncio
import os
import re
import subprocess

import pytest

from finrag.core.mcp_executor import HttpMcpExecutor, SqlResult


def _default_gateway() -> str | None:
    """取 WSL2 默认网关（即 Windows 宿主在 WSL 子网的地址）。"""
    try:
        out = subprocess.run(
            ["ip", "route"], capture_output=True, text=True, timeout=3
        ).stdout
        for line in out.splitlines():
            if line.startswith("default"):
                return line.split()[2]
    except Exception:
        pass
    return None


def _candidate_urls() -> list[str]:
    urls: list[str] = []
    if env_url := os.environ.get("FINRAG_TEST_MCP_URL"):
        urls.append(env_url.rstrip("/"))
    else:
        if gw := _default_gateway():
            urls += [f"http://{gw}:8008/mcp", f"http://{gw}:8008"]
    return urls


async def _list_tools_raw(url: str) -> list:
    """原生 SDK 列出全部工具（含参数 schema），供快照用例打印。"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


@pytest.fixture(scope="module")
def live():
    """连接真实 MCP Server，返回 (url, executor, tool_map)；不可达则 skip 整个模块。"""
    candidates = _candidate_urls()
    if not candidates:
        pytest.skip("无法确定 MCP Server 地址（非 WSL 网关环境且未设 FINRAG_TEST_MCP_URL）")
    last_err = ""
    for url in candidates:
        executor = HttpMcpExecutor(url, timeout=10.0)
        try:
            tool_map = executor._discover_tools()
            return url, executor, tool_map
        except Exception as exc:
            last_err = f"{url} -> {type(exc).__name__}: {str(exc)[:120]}"
    pytest.skip(
        f"MCP Server 不可达（已尝试: {candidates}；最后错误: {last_err}）。"
        "Windows 侧请放行防火墙（管理员 PowerShell）："
        "New-NetFirewallRule -DisplayName 'MCP 8008 for WSL' -Direction Inbound "
        "-Protocol TCP -LocalPort 8008 -Action Allow"
    )


def test_tools_snapshot(live):
    """工具清单快照：列出全部工具名与参数 schema（多库 Server 的分库参数会在这里显现）。"""
    url, _, _ = live
    tools = asyncio.run(_list_tools_raw(url))
    assert tools, "Server 未提供任何工具"
    print(f"\n=== MCP Server 工具清单 ({url}) ===")
    for t in tools:
        print(f"  {t.name}: {(t.description or '')[:100]}")
        schema = getattr(t, "input_schema", None) or {}
        if schema:
            required = set(schema.get("required", []))
            for k, v in schema.get("properties", {}).items():
                mark = "*" if k in required else " "
                print(f"    {mark} {k} ({v.get('type', '?')}): {str(v.get('description', ''))[:80]}")


def test_slot_mapping(live):
    """HttpMcpExecutor 槽位映射：execute_sql 必须映射成功（通用适配核心）。"""
    _, _, tool_map = live
    assert "execute_sql" in tool_map, (
        f"execute_sql 槽位未映射到任何工具；实际映射: {tool_map}。"
        "若工具名带库前缀（如 mysql_execute_sql），需扩展 _TOOL_SLOTS 关键词"
    )
    # 元信息槽位为软校验：多库 Server 可能未提供独立 list_tables/get_schema 工具
    for slot in ("list_tables", "get_schema"):
        status = "映射" if slot in tool_map else "未提供（跳过）"
        print(f"  slot {slot}: {status} -> {tool_map.get(slot)}")


def test_execute_sql_real_query(live):
    """真实执行一条 SELECT，验证全链路（连接 → 工具调用 → 结果解析为 SqlResult）。"""
    _, executor, _ = live
    result = executor.execute_sql("SELECT 1 AS probe")
    assert isinstance(result, SqlResult)
    assert result.columns, "结果缺少列名"
    assert result.rows, "结果为空"
    print(f"\n=== execute_sql('SELECT 1') -> columns={result.columns} rows={result.rows}")


# ---------------------------------------------------------------------------
# 多库测试（Server 支持 mysql / dolphindb 多数据库连接）
# ---------------------------------------------------------------------------

async def _call_tool(url: str, tool: str, arguments: dict) -> tuple[bool, str]:
    """原生调用 MCP 工具，返回 (is_error, 文本)。"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments=arguments)
            text = "\n".join(b.text for b in result.content if getattr(b, "text", None))
            return result.is_error, text


# list_databases 文本表格行格式："1  chinabond (当前)  mysql  -  127.0.0.1:3306"
_DB_ROW = re.compile(r"^\s*\d+\s+(\S+?)(?:\s*\(当前\))?\s+(mysql|dolphindb)\b", re.MULTILINE)


def _databases(url: str) -> list[tuple[str, str]]:
    _, text = asyncio.run(_call_tool(url, "list_databases", {}))
    return [(m.group(1), m.group(2)) for m in _DB_ROW.finditer(text)]


def test_multi_database_config(live):
    """Server 配置了 mysql 与 dolphindb 两类库（多库能力就绪）。"""
    url, _, _ = live
    dbs = _databases(url)
    print(f"\n=== 已配置数据库: {dbs}")
    types = {t for _, t in dbs}
    assert "mysql" in types, f"未发现 mysql 库: {dbs}"
    assert "dolphindb" in types, f"未发现 dolphindb 库: {dbs}"


def test_query_mysql_explicit_database(live):
    """显式指定 database 参数查询 MySQL 库（多库路由路径）。"""
    url, _, _ = live
    mysql_db = next((n for n, t in _databases(url) if t == "mysql"), None)
    if not mysql_db:
        pytest.skip("Server 未配置 mysql 库")
    is_err, text = asyncio.run(
        _call_tool(url, "query_database", {"sql": "SELECT 1 AS probe", "database": mysql_db})
    )
    assert not is_err, text[:200]
    assert "1" in text
    print(f"\n=== mysql[{mysql_db}] SELECT 1 -> {text.strip()[:60]}")


def test_query_dolphindb_database(live):
    """显式指定 database 参数查询 DolphinDB 库（DolphinDB 语法：select top N）。"""
    url, _, _ = live
    ddb = next((n for n, t in _databases(url) if t == "dolphindb"), None)
    if not ddb:
        pytest.skip("Server 未配置 dolphindb 库")
    is_err, text = asyncio.run(
        _call_tool(url, "query_database", {"sql": "select top 1 1 as probe", "database": ddb})
    )
    if "numpy" in text:
        # Server 侧环境问题：dolphindb 客户端与 numpy 2.x 二进制不兼容（错误文本包装在正常返回里）
        pytest.skip(f"MCP Server 侧 dolphindb 客户端 numpy ABI 不兼容: {text[:120]}")
    assert not is_err, text[:200]
    assert "1" in text
    print(f"\n=== dolphindb[{ddb}] select top 1 -> {text.strip()[:60]}")
