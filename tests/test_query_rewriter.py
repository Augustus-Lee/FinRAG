"""查询改写器测试：短路层 / LLM 触发 / 按意图定制 prompt / 防御性解析 / 异常回退 / 开关。"""

from unittest.mock import MagicMock

from finrag.core.query_rewriter import QueryRewriter, create_query_rewriter


def _rewriter(llm=None, **kw) -> QueryRewriter:
    return QueryRewriter(llm=llm, **kw)


_HISTORY = [
    {"role": "user", "content": "2024年7月股票交易的总金额是多少"},
    {"role": "assistant", "content": "7月股票交易总金额为 1,200 万元。"},
]


# ---------------------------------------------------------------------------
# 短路层：无历史 / 无指代信号 → 零成本透传
# ---------------------------------------------------------------------------


def test_short_circuit_no_history():
    llm = MagicMock()
    rewriter = _rewriter(llm=llm)
    # 首轮：即使问题满是指代词，无历史也不改写
    assert rewriter.rewrite("它的总和是多少", None) == "它的总和是多少"
    llm.chat.assert_not_called()


def test_short_circuit_self_contained_question():
    llm = MagicMock()
    rewriter = _rewriter(llm=llm)
    # 有历史但问题完整自包含 → 透传
    q = "2024年8月股票交易的总金额是多少"
    assert rewriter.rewrite(q, _HISTORY, mode="nl2sql") == q
    llm.chat.assert_not_called()


def test_anaphora_signals_detected():
    rewriter = _rewriter(llm=None)
    for q in (
        "那6月呢",
        "它的总和是多少",
        "继续说说",
        "该字段是什么意思",
        "上面说的是什么",
        "换成张三呢",
        "同月的清算金额是多少",
    ):
        assert rewriter._needs_rewrite(q), q


def test_common_self_contained_not_flagged():
    # 高频误报排查：这些完整问句不应触发改写
    rewriter = _rewriter(llm=None)
    for q in (
        "2024年7月股票交易总金额是多少",
        "trade_amount 字段的口径是什么",
        "什么是净赎回",
        "高风险产品的销售金额是多少",
        "product_sales 表里有哪些字段",
        "SQL 安全校验有哪几层",
        "应该怎么处理违约客户",  # 「应该」不含回指「该」
        "2024年之前的交易总额",  # 时间表达式「年之前」
    ):
        assert not rewriter._needs_rewrite(q), q


# ---------------------------------------------------------------------------
# LLM 层：触发改写 + 按意图定制
# ---------------------------------------------------------------------------


def test_anaphora_triggers_llm_rewrite():
    llm = MagicMock()
    llm.chat.return_value = "2024年6月股票交易的总金额是多少"
    rewriter = _rewriter(llm=llm)
    out = rewriter.rewrite("那6月呢", _HISTORY, mode="nl2sql")
    assert out == "2024年6月股票交易的总金额是多少"
    llm.chat.assert_called_once()
    # 确定性参数
    assert llm.chat.call_args.kwargs["temperature"] == 0


def test_prompt_contains_history_and_mode_instruction():
    llm = MagicMock()
    llm.chat.return_value = "2024年6月股票交易的总金额是多少"
    rewriter = _rewriter(llm=llm)
    rewriter.rewrite("那6月呢", _HISTORY, mode="nl2sql")
    prompt = llm.chat.call_args[0][0][0]["content"]
    assert "SQL" in prompt  # nl2sql 指令：可直接翻译成 SQL
    assert "7月股票交易" in prompt  # 历史进入 prompt
    assert "那6月呢" in prompt


def test_mode_specific_instruction_dictionary():
    llm = MagicMock()
    llm.chat.return_value = "settle_amount 字段的口径是什么"
    rewriter = _rewriter(llm=llm)
    out = rewriter.rewrite("它的口径是什么", _HISTORY, mode="dictionary")
    assert out == "settle_amount 字段的口径是什么"
    prompt = llm.chat.call_args[0][0][0]["content"]
    assert "字段名" in prompt  # dictionary 指令：术语归一到标准字段名


def test_mode_specific_instruction_knowledge():
    llm = MagicMock()
    llm.chat.return_value = "净赎回的风险有哪些"
    rewriter = _rewriter(llm=llm)
    rewriter.rewrite("它有什么风险", _HISTORY, mode="knowledge")
    prompt = llm.chat.call_args[0][0][0]["content"]
    assert "检索问题" in prompt  # knowledge 指令：面向召回的改写


def test_history_window_truncated():
    llm = MagicMock()
    llm.chat.return_value = "2024年6月股票交易的总金额是多少"
    rewriter = _rewriter(llm=llm)
    long_history = [{"role": "user", "content": f"第{i}个问题"} for i in range(20)]
    rewriter.rewrite("那6月呢", long_history)
    prompt = llm.chat.call_args[0][0][0]["content"]
    assert "第19个问题" in prompt  # 最近消息保留
    assert "第13个问题" not in prompt  # 早期消息被窗口截断


# ---------------------------------------------------------------------------
# 输出防御：噪音 / 拒答 / 膨胀 / 多行
# ---------------------------------------------------------------------------


def test_empty_output_falls_back():
    llm = MagicMock()
    llm.chat.return_value = ""
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


def test_refusal_output_falls_back():
    llm = MagicMock()
    llm.chat.return_value = "无法改写该问题，因为历史中缺少必要信息"
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


def test_noise_prefix_stripped():
    llm = MagicMock()
    llm.chat.return_value = "改写后的问题：2024年6月股票交易总金额是多少"
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "2024年6月股票交易总金额是多少"


def test_multiline_takes_first_line():
    llm = MagicMock()
    llm.chat.return_value = "2024年6月股票交易总金额是多少\n（说明：时间范围来自上文）"
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "2024年6月股票交易总金额是多少"


def test_bloated_output_falls_back():
    llm = MagicMock()
    llm.chat.return_value = "这个问题需要结合上下文分析：" + "很长的解释内容" * 60
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


def test_identical_output_passthrough():
    llm = MagicMock()
    llm.chat.return_value = "那6月呢"  # LLM 判定无需改写
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


# ---------------------------------------------------------------------------
# 异常与开关
# ---------------------------------------------------------------------------


def test_llm_exception_falls_back():
    llm = MagicMock()
    llm.chat.side_effect = TimeoutError("llm down")
    rewriter = _rewriter(llm=llm)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


def test_no_llm_passthrough():
    rewriter = _rewriter(llm=None)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"


def test_disabled_switch():
    llm = MagicMock()
    rewriter = _rewriter(llm=llm, enabled=False)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def test_create_query_rewriter_from_settings():
    from finrag.config import Settings

    settings = Settings()
    rewriter = create_query_rewriter(settings, llm=MagicMock())
    assert rewriter._enabled is True


def test_create_query_rewriter_disabled():
    from finrag.config import Settings

    settings = Settings()
    settings.query_rewrite_enabled = False
    rewriter = create_query_rewriter(settings)
    assert rewriter.rewrite("那6月呢", _HISTORY) == "那6月呢"
