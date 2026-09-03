"""LangGraph NL2SQL 自修正工作流测试：轻量 Fake pipeline（不碰 DB/LLM/网络）。

针对「移除 RAG 层 SQL 校验（安全委托 MCP Server 端）」改造验证：
- 图只有 generate/execute 两个节点（无 validate）
- 执行异常错误回灌重生成，attempts 达到 max_attempts 封顶（不死循环）
- 成功结果携带 sql/columns/rows/affected_rows；失败结果携带 error
"""

from finrag.core.mcp_executor import SqlResult
from finrag.workflows.nl2sql_retry import invoke_nl2sql

EXEC_ERROR = "syntax error near 'FROM'"
BAD_SQL = "SELECT bad"
GOOD_SQL = "SELECT COUNT(*) AS total FROM orders LIMIT 100"


class _RecordingExecutor:
    """sql 命中 bad_sqls 则抛异常，否则返回固定结果；记录每次收到的 sql。"""

    def __init__(self, bad_sqls: set[str]) -> None:
        self._bad_sqls = bad_sqls
        self.sqls: list[str] = []

    def execute_sql(self, sql: str) -> SqlResult:
        self.sqls.append(sql)
        if sql in self._bad_sqls:
            raise RuntimeError(EXEC_ERROR)
        return SqlResult(columns=["total"], rows=[[100]], affected_rows=1)


class FakePipeline:
    """generate_sql 按序返回预设 SQL（耗尽后重复最后一个，模拟永远生成失败 SQL）。"""

    def __init__(self, responses: list[str], bad_sqls: set[str]) -> None:
        self._responses = list(responses)
        self._last = ""
        self.generate_calls: list[tuple[str, str]] = []  # (question, error_feedback)
        self._executor = _RecordingExecutor(bad_sqls)

    def generate_sql(self, question: str, error: str = "") -> str:
        self.generate_calls.append((question, error))
        if self._responses:
            self._last = self._responses.pop(0)
        return self._last


def test_retry_after_execute_error_then_success():
    pipeline = FakePipeline([BAD_SQL, GOOD_SQL], bad_sqls={BAD_SQL})

    result = invoke_nl2sql(pipeline, "总订单量是多少？", max_attempts=2)

    assert result["ok"] is True
    assert result["sql"] == GOOD_SQL
    assert result["columns"] == ["total"]
    assert result["rows"] == [[100]]
    assert result["affected_rows"] == 1
    # 失败回灌后重生成：共 2 次 generate，第二次收到第一次的执行错误
    assert len(pipeline.generate_calls) == 2
    assert pipeline.generate_calls[0] == ("总订单量是多少？", "")
    assert pipeline.generate_calls[1] == ("总订单量是多少？", EXEC_ERROR)
    assert pipeline._executor.sqls == [BAD_SQL, GOOD_SQL]


def test_always_fail_caps_attempts_and_returns_error():
    pipeline = FakePipeline([BAD_SQL], bad_sqls={BAD_SQL})

    result = invoke_nl2sql(pipeline, "总订单量是多少？", max_attempts=2)

    assert result["ok"] is False
    assert result["error"]
    assert EXEC_ERROR in result["error"]
    # 封顶不死循环：attempts 达到 max_attempts 即终止（共 max_attempts 次生成/执行）
    assert len(pipeline.generate_calls) == 2
    assert pipeline._executor.sqls == [BAD_SQL, BAD_SQL]
