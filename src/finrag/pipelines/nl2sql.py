"""智能问数流水线：Schema Linking → SQL 生成 → 执行（MCP Server 端管控安全：SELECT-only/注入/条数）→ 转述。

执行异常（含 Server 拒绝）错误回灌自修正重试。
执行与自修正也可走 LangGraph 工作流（见 workflows.nl2sql_retry）。
"""

import time
from dataclasses import dataclass, field

from finrag.core.llm_gateway import LLMGateway
from finrag.core.mcp_executor import McpExecutor
from finrag.core.schema_linker import SchemaLinker
from finrag.logging import get_logger

logger = get_logger("finrag.nl2sql")

SYSTEM_PROMPT = (
    "你是金融数据查询 SQL 专家。基于给定的表结构（schema）生成 MySQL SELECT 语句：\n"
    "- 仅生成 SELECT，禁止 DDL/DML/多语句\n"
    "- 必须带 LIMIT（不超过 {max_rows} 行）\n"
    "- 仅使用给定的表与字段，不得臆造列名\n"
    "- 金额比较使用 DECIMAL 精度，注意口径\n"
    "直接输出 SQL 文本，不要解释。"
)


@dataclass
class NL2SQLAnswer:
    question: str
    answer: str = ""
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    affected_rows: int = 0
    attempts: int = 0
    latency_ms: float = 0.0


class NL2SQLPipeline:
    def __init__(
        self,
        schema_linker: SchemaLinker,
        llm_gateway: LLMGateway,
        executor: McpExecutor,
        max_retries: int = 2,
        max_rows: int = 100,
    ) -> None:
        self._linker = schema_linker
        self._llm = llm_gateway
        self._executor = executor
        self._max_retries = max_retries
        self._max_rows = max_rows

    def generate_sql(self, question: str, error_feedback: str = "") -> str:
        """生成 SQL（Schema Linking + 提示词）。error_feedback 用于自修正回灌。"""
        ctx = self._linker.link(question)
        system = SYSTEM_PROMPT.format(max_rows=self._max_rows)
        user = f"【表结构】\n{ctx.to_prompt()}\n\n【问题】\n{question}"
        if error_feedback:
            user += f"\n\n【上次错误】\n{error_feedback}\n请修正后重新生成 SQL。"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._llm.chat(messages, temperature=0).strip()

    def answer(self, question: str) -> NL2SQLAnswer:
        start = time.perf_counter()
        ans = NL2SQLAnswer(question=question)
        error_feedback = ""

        for attempt in range(self._max_retries + 1):
            ans.attempts = attempt + 1
            sql = self.generate_sql(question, error_feedback)
            try:
                exec_result = self._executor.execute_sql(sql)
                ans.sql = sql
                ans.columns = exec_result.columns
                ans.rows = exec_result.rows
                ans.affected_rows = exec_result.affected_rows
                break
            except Exception as exc:
                error_feedback = str(exc)[:300]
                continue

        ans.answer = self._translate(question, ans)
        ans.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info("nl2sql_answer", attempts=ans.attempts, rows=ans.affected_rows, latency_ms=ans.latency_ms)
        return ans

    def _translate(self, question: str, ans: NL2SQLAnswer) -> str:
        """把查询结果转述为自然语言。"""
        if not ans.sql:
            return "未能生成可执行的 SQL 查询，请尝试调整提问方式或检查数据权限。"
        if not ans.rows:
            return "查询执行成功，但未返回任何数据。"
        preview = ans.rows[:5]
        prompt = (
            f"问题：{question}\nSQL：{ans.sql}\n返回 {ans.affected_rows} 行，样例：{preview}\n"
            "请用中文简洁总结查询结果（不超过 100 字）。"
        )
        try:
            return self._llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200
            ).strip()
        except Exception:
            return f"查询成功，返回 {ans.affected_rows} 行数据。"
