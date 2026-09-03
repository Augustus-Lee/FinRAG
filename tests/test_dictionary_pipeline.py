"""数据字典流水线测试：search 透传 top_k、耗时统计、answer 单次 link、LLM 复用 ctx。"""

import time
from unittest.mock import MagicMock

from finrag.core.schema_linker import SchemaContext
from finrag.pipelines.dictionary import DictionaryPipeline, FieldHit

_FIELD = {
    "table_name": "customer_account",
    "field_name": "mobile",
    "field_type": "VARCHAR(20)",
    "comment": "手机号",
    "calibre": "",
    "synonyms": ["手机"],
}


def _ctx(fields):
    # 模拟真实 link()：tables 由命中字段的 table_name 推导，使 to_prompt() 能渲染字段
    table_names: list[str] = []
    for f in fields:
        if f["table_name"] not in table_names:
            table_names.append(f["table_name"])
    return SchemaContext(
        fields=fields,
        tables=[{"table_name": tn, "business_domain": "", "description": ""} for tn in table_names],
        matched_tables=table_names,
    )


def test_search_passes_top_k_to_link_and_slices():
    linker = MagicMock()
    linker.link.return_value = _ctx([_FIELD])
    p = DictionaryPipeline(linker)
    result = p.search("客户手机号", top_k=5)
    assert linker.link.call_args.kwargs["top_k"] == 5
    assert len(result.hits) == 1
    assert result.hits[0].field_name == "mobile"


def test_search_reports_real_latency_not_hardcoded():
    # link 耗时 5ms → latency_ms 应反映真实耗时而非硬编码 0
    linker = MagicMock()
    linker.link.side_effect = lambda *a, **kw: (time.sleep(0.005), _ctx([_FIELD]))[1]
    p = DictionaryPipeline(linker)
    result = p.search("客户手机号", top_k=5)
    assert result.latency_ms >= 1.0
    assert isinstance(result.latency_ms, float)


def test_service_search_passes_through_latency(monkeypatch):
    # service 层必须透传 pipeline 统计的 latency_ms（修复硬编码 0.0）
    from finrag import container
    from finrag.pipelines.dictionary import DictSearchResult
    from finrag.services.dictionary_service import DictionaryService

    hit = FieldHit(
        table_name="customer_account",
        field_name="mobile",
        field_type="VARCHAR(20)",
        comment="手机号",
        calibre="",
        synonyms=["手机"],
    )
    pipeline = MagicMock()
    pipeline.search.return_value = DictSearchResult(question="q", hits=[hit], latency_ms=12.3)
    monkeypatch.setattr(container, "get_dictionary_pipeline", lambda: pipeline)

    resp = DictionaryService().search(db=MagicMock(), question="q", top_k=5)
    assert resp.latency_ms == 12.3
    pipeline.search.assert_called_once_with("q", top_k=5)
    assert resp.hits[0].field_name == "mobile"


def test_answer_calls_link_once_and_reuses_ctx():
    linker = MagicMock()
    linker.link.return_value = _ctx([_FIELD])
    p = DictionaryPipeline(linker, llm_gateway=None)
    ans = p.answer("客户手机号是多少")
    assert linker.link.call_count == 1  # 关键：answer 不再重复调用 link
    assert len(ans.hits) == 1
    assert ans.hits[0].field_name == "mobile"
    assert ans.summary == ""  # 无 LLM → 汇总为空


def test_answer_uses_llm_with_ctx_prompt_and_single_link():
    linker = MagicMock()
    linker.link.return_value = _ctx([_FIELD])
    llm = MagicMock()
    llm.chat.return_value = "口径汇总"
    p = DictionaryPipeline(linker, llm_gateway=llm)
    ans = p.answer("客户手机号是多少")
    assert ans.summary == "口径汇总"
    sent = llm.chat.call_args.args[0][0]["content"]
    assert "mobile" in sent  # ctx.to_prompt() 输出被复用进 prompt
    assert linker.link.call_count == 1


def test_answer_llm_failure_does_not_break_and_logs():
    linker = MagicMock()
    linker.link.return_value = _ctx([_FIELD])
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("llm down")
    p = DictionaryPipeline(linker, llm_gateway=llm)
    ans = p.answer("客户手机号是多少")
    assert ans.summary == ""  # 异常被吞，汇总为空
    assert len(ans.hits) == 1  # 命中不受影响
