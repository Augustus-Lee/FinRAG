"""DictionaryIndexer 测试：build 幂等、空字典、双写、UUID point id、失败不残留。"""

import uuid
from unittest.mock import MagicMock

from finrag.core.bm25 import BM25Index
from finrag.core.dictionary_indexer import DictionaryIndexer, _doc_id


def _fields():
    return [
        {
            "id": 1,
            "table_name": "t",
            "field_name": "mobile",
            "field_type": "VARCHAR",
            "comment": "手机号",
            "calibre": "",
            "synonyms": ["手机"],
        },
        {
            "id": 2,
            "table_name": "t",
            "field_name": "amt",
            "field_type": "DECIMAL",
            "comment": "金额",
            "calibre": "含税",
            "synonyms": [],
        },
    ]


def _make(provider=None):
    emb = MagicMock()
    emb.embed.return_value = [[0.1], [0.2]]
    vs = MagicMock()
    bm = BM25Index()
    idx = DictionaryIndexer(emb, vs, bm, provider or _fields)
    return idx, emb, vs, bm


def test_doc_id_is_valid_deterministic_uuid():
    val = _doc_id(5)
    uuid.UUID(val)  # 合法 UUID
    assert _doc_id(5) == _doc_id(5)  # 确定性
    assert _doc_id(5) != _doc_id(6)  # 不同字段不同 id


def test_build_indexes_all_fields_and_double_writes():
    idx, emb, vs, bm = _make()
    n = idx.build()
    assert n == 2
    assert idx.size == 2
    points = vs.upsert.call_args.args[0]
    assert len(points) == 2
    assert {p.id for p in points} == {_doc_id(1), _doc_id(2)}
    # BM25 可检索
    hits = bm.search("手机", top_k=5)
    assert len(hits) >= 1
    assert hits[0].doc_id in {_doc_id(1), _doc_id(2)}


def test_build_is_idempotent_on_rebuild():
    idx, emb, vs, bm = _make()
    assert idx.build() == 2
    assert idx.build() == 2  # 幂等重建
    # 第二次 build 前会先 delete 旧 id
    assert vs.delete_by_ids.called
    assert vs.upsert.call_count == 2


def test_build_empty_dict_returns_zero():
    idx, emb, vs, bm = _make(provider=lambda: [])
    assert idx.build() == 0
    assert idx.size == 0
    assert not vs.upsert.called


def test_build_upsert_failure_returns_zero_no_half_state():
    idx, emb, vs, bm = _make()
    vs.upsert.side_effect = RuntimeError("qdrant down")
    assert idx.build() == 0
    assert idx.size == 0  # 失败时 _indexed_ids 不应被登记


def test_build_embed_failure_returns_zero():
    idx, emb, vs, bm = _make()
    emb.embed.side_effect = RuntimeError("embed down")
    assert idx.build() == 0
    assert idx.size == 0
    assert not vs.upsert.called
