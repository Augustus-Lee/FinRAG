"""BM25 索引测试：payload 存储、clear 幂等、__len__。"""

from finrag.core.bm25 import BM25Index


def test_payload_roundtrip_via_search():
    idx = BM25Index()
    idx.add("doc1", "存款利率 调整", payload={"section": "利率政策"})
    hits = idx.search("存款利率", top_k=1)
    assert hits[0].doc_id == "doc1"
    assert hits[0].payload == {"section": "利率政策"}


def test_add_without_payload_defaults_to_empty():
    idx = BM25Index()
    idx.add("doc1", "贷款规则")
    hits = idx.search("贷款规则", top_k=1)
    assert hits[0].payload == {}


def test_clear_resets_payload_and_len():
    idx = BM25Index()
    idx.add("doc1", "存款利率", payload={"a": 1})
    assert len(idx) == 1
    idx.clear()
    assert len(idx) == 0
    assert idx.search("存款利率", top_k=5) == []


def test_readd_overwrites_payload():
    idx = BM25Index()
    idx.add("doc1", "存款利率", payload={"v": 1})
    idx.add("doc1", "存款利率", payload={"v": 2})
    hits = idx.search("存款利率", top_k=1)
    assert hits[0].payload == {"v": 2}
    assert len(idx) == 1
