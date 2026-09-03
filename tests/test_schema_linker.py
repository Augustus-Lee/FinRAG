"""Schema Linking 测试：中文整句 → 命中字段注释/同义词（n-gram 降级分词）。

覆盖：
- 关键词降级路径（无混合检索依赖时的默认行为）
- 混合检索路径（注入 mock retriever/embedding/indexer）：top_k 透传、payload 还原
- 混合检索异常/空索引时回退关键词
- rebuild() 触发索引重建
"""

from unittest.mock import MagicMock

from finrag.core.hybrid_retriever import HybridHit
from finrag.core.schema_linker import SchemaLinker

_TABLES = [
    {"table_name": "product_sales", "business_domain": "销售", "description": "产品销售事实表"},
    {"table_name": "customer_account", "business_domain": "客户", "description": "客户账户主数据"},
]

_FIELDS = [
    {
        "table_name": "product_sales",
        "field_name": "sales_amount",
        "field_type": "DECIMAL(18,2)",
        "comment": "销售金额",
        "calibre": "含增值税",
        "synonyms": ["销售额", "成交金额"],
    },
    {
        "table_name": "customer_account",
        "field_name": "mobile",
        "field_type": "VARCHAR(20)",
        "comment": "手机号",
        "calibre": "",
        "synonyms": ["手机", "手机号码", "联系电话"],
    },
]


def _linker() -> SchemaLinker:
    return SchemaLinker(table_provider=lambda: (_TABLES, _FIELDS), top_k=5)


def test_chinese_sentence_hits_field_comment():
    # 整句中文："客户手机号是多少" 应命中 customer_account.mobile（"手机号"）
    ctx = _linker().link("客户手机号是多少")
    assert ctx.matched_tables == ["customer_account"]
    names = {(f["table_name"], f["field_name"]) for f in ctx.fields}
    assert ("customer_account", "mobile") in names


def test_chinese_sentence_hits_synonym():
    # 通过同义词"销售额"命中 product_sales.sales_amount
    ctx = _linker().link("上个月的销售额是多少")
    assert "product_sales" in ctx.matched_tables
    names = {(f["table_name"], f["field_name"]) for f in ctx.fields}
    assert ("product_sales", "sales_amount") in names


def test_english_table_name_match():
    ctx = _linker().link("product_sales 有哪些字段")
    assert "product_sales" in ctx.matched_tables


def test_no_match_returns_empty():
    ctx = _linker().link("天气怎么样")
    assert ctx.matched_tables == []
    assert ctx.fields == []


def test_fields_sorted_by_relevance():
    # "客户手机号是多少"：mobile（synonyms 含"手机号"）应排在其所在表字段首位
    ctx = _linker().link("客户手机号是多少")
    names = [f["field_name"] for f in ctx.fields if f["table_name"] == "customer_account"]
    assert names == ["mobile"]


# --------------------------------------------------------------------------- #
# 混合检索路径（注入 mock retriever/embedding/indexer）
# --------------------------------------------------------------------------- #


def _hybrid_linker(field_payloads, captured):
    """构造注入混合检索依赖的 SchemaLinker；captured 收集 top_k / query。"""

    def _search(query_vector, query_text, top_k=None, filter_=None):
        captured["top_k"] = top_k
        captured["query"] = query_text
        return [
            HybridHit(doc_id=f"r{i}", rrf_score=1.0, vector_score=1.0, bm25_score=0.0, payload=p)
            for i, p in enumerate(field_payloads)
        ]

    emb = MagicMock()
    emb.embed_query.return_value = [0.1, 0.2]
    retriever = MagicMock()
    retriever.search.side_effect = _search
    indexer = MagicMock()
    indexer.size = len(field_payloads)
    indexer.build.return_value = len(field_payloads)
    linker = SchemaLinker(
        table_provider=lambda: (_TABLES, _FIELDS),
        retriever=retriever,
        embedding=emb,
        indexer=indexer,
    )
    return linker, retriever, emb


def test_hybrid_link_returns_payload_fields_and_passthrough_top_k():
    payload = [
        {
            "table_name": "customer_account",
            "field_name": "mobile",
            "field_type": "VARCHAR(20)",
            "comment": "手机号",
            "calibre": "",
            "synonyms": ["手机"],
        }
    ]
    captured = {}
    linker, retriever, emb = _hybrid_linker(payload, captured)
    ctx = linker.link("客户手机号", top_k=7)
    assert ctx.fields == payload
    assert ctx.matched_tables == ["customer_account"]
    assert captured["top_k"] == 7
    emb.embed_query.assert_called_once_with("客户手机号")


def test_hybrid_link_empty_index_falls_back_to_keyword():
    captured = {}
    linker, retriever, emb = _hybrid_linker([], captured)
    # indexer.size == 0 → hybrid 短路返回 None → 关键词降级
    ctx = linker.link("客户手机号是多少")
    assert ("customer_account", "mobile") in {
        (f["table_name"], f["field_name"]) for f in ctx.fields
    }
    retriever.search.assert_not_called()


def test_hybrid_link_retriever_raises_falls_back_to_keyword():
    payload = [
        {
            "table_name": "customer_account",
            "field_name": "mobile",
            "field_type": "",
            "comment": "手机号",
            "calibre": "",
            "synonyms": [],
        }
    ]
    captured = {}
    linker, retriever, emb = _hybrid_linker(payload, captured)
    retriever.search.side_effect = RuntimeError("boom")
    ctx = linker.link("客户手机号是多少")
    assert ("customer_account", "mobile") in {
        (f["table_name"], f["field_name"]) for f in ctx.fields
    }


def test_rebuild_triggers_indexer_build_and_clears_cache():
    emb = MagicMock()
    retriever = MagicMock()
    indexer = MagicMock()
    indexer.build.return_value = 0
    linker = SchemaLinker(
        table_provider=lambda: (_TABLES, _FIELDS),
        retriever=retriever,
        embedding=emb,
        indexer=indexer,
    )
    linker.link("客户手机号是多少")  # 首次 link 触发懒加载建索引
    assert indexer.build.call_count == 1
    linker.rebuild()
    assert indexer.build.call_count == 2
