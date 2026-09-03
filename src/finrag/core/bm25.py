"""自研 BM25 检索（核心自研组件）。

- 分词器可注入（默认 jieba，金融场景中文友好）
- 经典 BM25 公式：idf 采用 Robertson 平滑，避免 df > N/2 时 idf 为负
- 内存索引，适合单机百万 chunk 级；可替换为 ES 等外部实现
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

_TOKEN_SPLIT = re.compile(r"[\s\W]+", re.UNICODE)

# 极简中文停用词（可扩展）
_STOPWORDS = {
    "的", "了", "和", "是", "在", "与", "及", "或", "等", "对", "中", "为", "于",
    "the", "a", "an", "of", "to", "in", "and", "or", "for",
}


def _default_tokenizer(text: str) -> list[str]:
    try:
        import jieba

        return [t for t in jieba.lcut(text) if t and t not in _STOPWORDS]
    except ImportError:
        # 无 jieba 时的降级分词：按非字母数字拆分
        return [t for t in _TOKEN_SPLIT.split(text.lower()) if t and t not in _STOPWORDS]


@dataclass
class BM25Hit:
    doc_id: str
    score: float
    payload: dict


class BM25Index:
    """内存 BM25 索引。"""

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._tokenizer = tokenizer or _default_tokenizer
        self._k1 = k1
        self._b = b
        self._docs: dict[str, str] = {}          # doc_id -> text
        self._tokens_cache: dict[str, list[str]] = {}  # doc_id -> 分词缓存（避免查询时重复分词）
        self._doc_len: dict[str, int] = {}       # doc_id -> token 数
        self._df: dict[str, int] = {}            # term -> 出现文档数
        self._payloads: dict[str, dict] = {}     # doc_id -> payload（供检索结果透传）
        self._avgdl: float = 0.0
        self._ready = False

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str, payload: dict | None = None) -> None:
        tokens = self._tokenizer(text)
        self._docs[doc_id] = text
        self._tokens_cache[doc_id] = tokens
        self._doc_len[doc_id] = len(tokens)
        self._payloads[doc_id] = payload or {}
        for term in set(tokens):
            self._df[term] = self._df.get(term, 0) + 1
        self._ready = False

    def add_batch(self, items: list[tuple[str, str, dict | None]]) -> None:
        for doc_id, text, payload in items:
            self.add(doc_id, text, payload)

    def clear(self) -> None:
        """清空索引（用于幂等重建）。"""
        self._docs.clear()
        self._tokens_cache.clear()
        self._doc_len.clear()
        self._df.clear()
        self._payloads.clear()
        self._avgdl = 0.0
        self._ready = False

    def _finalize(self) -> None:
        if self._ready:
            return
        total = sum(self._doc_len.values())
        n = len(self._doc_len)
        self._avgdl = total / n if n else 0.0
        self._ready = True

    def search(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        """对查询做 BM25 打分，返回降序结果。"""
        self._finalize()
        if not self._docs:
            return []

        query_terms = self._tokenizer(query)
        n_docs = len(self._docs)
        scores: dict[str, float] = {doc_id: 0.0 for doc_id in self._docs}

        for term in set(query_terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            # Robertson 平滑 idf，避免负值
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, doc_len in self._doc_len.items():
                # 该 term 在 doc 中的出现次数（复用分词缓存）
                tf = sum(1 for t in self._tokens_cache[doc_id] if t == term)
                if tf == 0:
                    continue
                denom = tf + self._k1 * (1 - self._b + self._b * (doc_len / self._avgdl if self._avgdl else 1.0))
                scores[doc_id] += idf * (tf * (self._k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            BM25Hit(doc_id=doc_id, score=score, payload=self._payloads.get(doc_id, {}))
            for doc_id, score in ranked[:top_k]
        ]
