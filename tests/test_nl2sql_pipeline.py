"""NL2SQLPipeline 单元测试：全 Fake（不碰 DB/LLM/网络）。

针对「移除 RAG 层 SQL 校验（安全委托 MCP Server 端）」改造验证：
- LLM 生成的 SQL 原样（仅 strip）透传 executor，无本地校验/改写
- 执行异常（含 MCP Server 拒绝）错误回灌自修正重试
- 重试耗尽返回失败应答（不抛异常）
- 一次成功不做多余重试
"""

from finrag.core.mcp_executor import SqlResult
from finrag.pipelines.nl2sql import NL2SQLPipeline

EXEC_ERROR = "syntax error near 'FROM'"
BAD_SQL = "SELECT bad"
GOOD_SQL = "SELECT COUNT(*) AS total FROM orders LIMIT 100"
TRANSLATE_TEXT = "查询到 1 行，合计 100。"


class FakeLLM:
    """按调用序返回预设回复；耗尽后返回 default（供 _translate 转述调用）。"""

    def __init__(self, responses: list[str], default: str = TRANSLATE_TEXT) -> None:
        self._responses = list(responses)
        self._default = default
        self.calls: list[list[dict]] = []  # 每次 chat 收到的 messages

    def chat(self, messages: list[dict], temperature: float = 0, **kwargs) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return self._default


class _SchemaContext:
    def to_prompt(self) -> str:
        return "schema context"


class FakeLinker:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def link(self, question: str) -> _SchemaContext:
        self.questions.append(question)
        return _SchemaContext()


class FakeExecutor:
    """sql 命中 bad_sqls 则抛异常（模拟执行失败/MCP Server 拒绝），否则返回固定结果。"""

    def __init__(self, bad_sqls: set[str] | None = None) -> None:
        self._bad_sqls = set(bad_sqls or [])
        self.sqls: list[str] = []  # 每次收到的 sql（透传断言用）

    def execute_sql(self, sql: str) -> SqlResult:
        self.sqls.append(sql)
        if sql in self._bad_sqls:
            raise RuntimeError(EXEC_ERROR)
        return SqlResult(columns=["total"], rows=[[100]], affected_rows=1)


def _make(llm: FakeLLM, executor: FakeExecutor) -> NL2SQLPipeline:
    return NL2SQLPipeline(FakeLinker(), llm, executor)


def test_execute_error_feeds_back_and_retry_succeeds():
    # 第一次生成的 SQL 执行失败 → 错误回灌 → 第二次重生成后成功
    llm = FakeLLM([BAD_SQL, GOOD_SQL])
    executor = FakeExecutor(bad_sqls={BAD_SQL})
    pipeline = _make(llm, executor)

    ans = pipeline.answer("总订单量是多少？")

    assert ans.attempts == 2
    assert ans.sql == GOOD_SQL
    assert ans.columns == ["total"]
    assert ans.rows == [[100]]
    assert ans.affected_rows == 1
    assert ans.answer == TRANSLATE_TEXT
    # 第二次执行的是重生成后的 SQL
    assert executor.sqls == [BAD_SQL, GOOD_SQL]
    # 回灌生效：第二次生成的 user prompt 含第一次的执行错误，第一次则没有
    assert EXEC_ERROR not in llm.calls[0][-1]["content"]
    assert "上次错误" not in llm.calls[0][-1]["content"]
    assert EXEC_ERROR in llm.calls[1][-1]["content"]
    assert "上次错误" in llm.calls[1][-1]["content"]


def test_retry_exhausted_returns_failure_answer():
    # 预设多于重试上限的失败 SQL：若重试环失控会消耗更多（用例会失败）
    llm = FakeLLM([BAD_SQL] * 5)
    executor = FakeExecutor(bad_sqls={BAD_SQL})
    pipeline = _make(llm, executor)

    ans = pipeline.answer("总订单量是多少？")  # 不抛异常

    assert ans.attempts == 3  # 默认 max_retries=2 → 共 3 次尝试
    assert ans.sql == ""
    assert ans.columns == []
    assert ans.rows == []
    assert ans.affected_rows == 0
    assert "未能生成" in ans.answer
    # 每次都真实下发执行（错误来自 executor，而非本地校验拦截）
    assert executor.sqls == [BAD_SQL] * 3


def test_success_on_first_attempt_no_extra_retry():
    llm = FakeLLM([GOOD_SQL])
    executor = FakeExecutor()
    pipeline = _make(llm, executor)

    ans = pipeline.answer("总订单量是多少？")

    assert ans.attempts == 1
    assert ans.sql == GOOD_SQL
    assert ans.columns == ["total"]
    assert ans.rows == [[100]]
    assert ans.affected_rows == 1
    # 只执行一次，无多余重试
    assert executor.sqls == [GOOD_SQL]


def test_sql_passed_through_verbatim():
    # LLM 输出带首尾空白：pipeline 仅 strip 后透传，无本地校验改写（不加 LIMIT 等）
    raw = f"  {GOOD_SQL}  \n"
    llm = FakeLLM([raw])
    executor = FakeExecutor()
    pipeline = _make(llm, executor)

    ans = pipeline.answer("总订单量是多少？")

    assert executor.sqls == [GOOD_SQL]
    assert ans.sql == GOOD_SQL
