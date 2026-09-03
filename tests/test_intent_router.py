"""意图路由器测试：规则层命中 / LLM 兜底 / 异常回退 / 开关与默认值。"""

from unittest.mock import MagicMock

from finrag.core.intent_router import IntentRouter, create_intent_router


def _router(llm=None, **kw) -> IntentRouter:
    return IntentRouter(llm=llm, **kw)


# ---------------------------------------------------------------------------
# 规则层：nl2sql（聚合/数值疑问词）
# ---------------------------------------------------------------------------


def test_rule_nl2sql_aggregation_words():
    router = _router()
    assert router.classify("高风险产品的销售金额是多少") == "nl2sql"
    assert router.classify("总资产超过100万的客户有几个") == "nl2sql"
    assert router.classify("交易金额最大的客户是谁") == "nl2sql"
    assert router.classify("按产品统计销售总额") == "nl2sql"
    assert router.classify("销量排名前10的产品") == "nl2sql"


def test_rule_nl2sql_no_llm_call():
    # 规则层命中时不应触发 LLM
    llm = MagicMock()
    router = _router(llm=llm)
    router.classify("卖了多少")
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 规则层：dictionary（元信息词 + 字段实体）
# ---------------------------------------------------------------------------


def test_rule_dictionary_identifier_plus_meta():
    router = _router()
    assert router.classify("trade_amount 字段的口径是什么") == "dictionary"
    assert router.classify("mobile 字段是什么含义、存在哪张表") == "dictionary"
    assert router.classify("product_sales 表有哪些字段") == "dictionary"


def test_rule_dictionary_strong_meta_without_identifier():
    # 明确问字段/表结构，即使无英文标识符也路由 dictionary
    router = _router()
    assert router.classify("客户手机号字段的单位是什么") == "dictionary"


def test_rule_concept_question_not_dictionary():
    # "xx 的含义"但无字段实体 → 不路由 dictionary（交给 LLM 层）
    llm = MagicMock()
    llm.chat.return_value = "knowledge"
    router = _router(llm=llm)
    assert router.classify("风险提示的含义是什么") == "knowledge"
    llm.chat.assert_called_once()


# ---------------------------------------------------------------------------
# LLM 兜底层
# ---------------------------------------------------------------------------


def test_llm_fallback_for_ambiguous():
    llm = MagicMock()
    llm.chat.return_value = "nl2sql"
    router = _router(llm=llm)
    # 无规则信号的问题（省略问句/指代）→ LLM 判定
    assert router.classify("客户张三的手机号是什么") == "nl2sql"
    llm.chat.assert_called_once()


def test_llm_output_noise_tolerated():
    llm = MagicMock()
    llm.chat.return_value = "答案是：dictionary。"
    router = _router(llm=llm)
    assert router.classify("这个字段什么意思") == "dictionary"


def test_llm_unparsed_falls_back_knowledge():
    llm = MagicMock()
    llm.chat.return_value = "无法判断这个问题"  # 不含任何合法 mode 词
    router = _router(llm=llm)
    assert router.classify("随便问问") == "knowledge"


def test_llm_exception_falls_back_knowledge():
    llm = MagicMock()
    llm.chat.side_effect = TimeoutError("llm down")
    router = _router(llm=llm)
    assert router.classify("随便问问") == "knowledge"


def test_no_llm_defaults_knowledge():
    router = _router(llm=None)
    assert router.classify("客户张三的手机号是什么") == "knowledge"


# ---------------------------------------------------------------------------
# 开关
# ---------------------------------------------------------------------------


def test_disabled_router_always_knowledge():
    llm = MagicMock()
    router = _router(llm=llm, enabled=False)
    assert router.classify("销售金额是多少") == "knowledge"  # 规则层也跳过
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def test_create_intent_router_from_settings():
    from finrag.config import Settings

    settings = Settings()
    router = create_intent_router(settings, llm=MagicMock())
    assert router._enabled is True
    assert router._confidence_threshold == 0.6


def test_create_intent_router_disabled():
    from finrag.config import Settings

    settings = Settings()
    settings.intent_router_enabled = False
    router = create_intent_router(settings)
    assert router.classify("多少钱") == "knowledge"
