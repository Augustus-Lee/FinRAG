"""金融规则切分器测试：标题分层 / 表格保护 / 参数校验。"""

import pytest

from finrag.core.chunker import FinancialChunker, ParsedDocument, estimate_tokens


def test_section_path_from_headings():
    doc = ParsedDocument(
        title="产品说明",
        content="# 产品概述\n这是产品概述内容。\n## 收益计算\n按日计息，复利滚存。",
    )
    chunks = FinancialChunker().split(doc)
    paths = {c.section_path for c in chunks}
    assert "产品概述" in paths
    assert "产品概述 / 收益计算" in paths


def test_table_kept_as_single_chunk_with_meta():
    content = (
        "# 费率表\n"
        "| 档位 | 费率 | 门槛 |\n"
        "| --- | --- | --- |\n"
        "| A | 0.1% | 1万 |\n"
        "| B | 0.05% | 100万 |\n"
    )
    doc = ParsedDocument(title="费率", content=content)
    chunks = FinancialChunker().split(doc)

    table_chunks = [c for c in chunks if c.table_meta]
    assert len(table_chunks) == 1
    meta = table_chunks[0].table_meta
    assert meta["headers"] == ["档位", "费率", "门槛"]
    assert meta["rows"] == 2
    # 关键：表格作为整体保留，数字与分隔符不被切断
    assert "0.1%" in table_chunks[0].content
    assert "0.05%" in table_chunks[0].content


def test_empty_content_returns_no_chunks():
    assert FinancialChunker().split(ParsedDocument(title="空文档", content="   ")) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        FinancialChunker(chunk_size=100, overlap=100)


def test_table_section_path_uses_heading_at_table_position():
    # 回归：旧实现表格 chunk 取文档末尾标题状态，导致归属错位
    content = (
        "# 费率表\n"
        "| 档位 | 费率 |\n"
        "| --- | --- |\n"
        "| A | 0.1% |\n"
        "# 下一章\n"
        "后续正文内容。"
    )
    chunks = FinancialChunker().split(ParsedDocument(title="t", content=content))
    table_chunks = [c for c in chunks if c.table_meta]
    assert len(table_chunks) == 1
    assert table_chunks[0].section_path == "费率表"  # 而非"下一章"
    # 文本块归属"下一章"
    text_chunks = [c for c in chunks if not c.table_meta]
    assert all(c.section_path == "下一章" for c in text_chunks)


def test_table_and_text_chunks_in_document_order():
    content = (
        "# 第一章\n"
        "第一段正文。\n"
        "\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "# 第二章\n"
        "第二段正文。"
    )
    chunks = FinancialChunker().split(ParsedDocument(title="t", content=content))
    # 按文档顺序：第一段文本 → 表格 → 第二段文本
    assert [bool(c.table_meta) for c in chunks] == [False, True, False]
    assert chunks[0].content == "第一段正文。"
    assert chunks[1].table_meta["headers"] == ["A", "B"]
    assert chunks[2].content == "第二段正文。"


def test_estimate_tokens_cjk_rule():
    # 中文约 2 字符/token：20 个中文 → 约 10 token
    assert estimate_tokens("中" * 20) == 10
    # 英文约 4 字符/token
    assert estimate_tokens("abcd" * 10) == 10


# ---------------------------------------------------------------------------
# _split_text 句子级切分（回归：旧实现按空白切"词"对中文失效）
# ---------------------------------------------------------------------------


def test_long_cjk_paragraph_gets_split():
    """回归①：无空格中文长段落旧实现永不切分（单 chunk 超大，embedding 截断丢内容）。"""
    # 300 句 × 每句 14 字（7 token）≈ 2100 token > 512
    text = "这是用于测试切分器的中文句子。" * 300
    chunks = FinancialChunker()._split_text(text, "测试")

    assert len(chunks) > 1
    # 每个 chunk 都不超限（块体 ≤512 + overlap 逐句回取最多溢出一句 ~3 token）
    for c in chunks:
        assert c.token_count <= 512 + 64 + 8
    # 内容无丢失（overlap 会导致句子重复出现，故用 >=）
    joined = "".join(c.content for c in chunks)
    assert joined.count("这是用于测试切分器的中文句子。") >= 300


def test_multi_paragraph_no_repeat_bloat():
    """回归②：多段落场景旧实现把前一整块当 overlap 带走，chunk 依次膨胀（500→600→700…）。"""
    # 每句 7 字（3 token）→ 每段 6 token；200 段 = 1200 token > 512
    paras = ["利率按日计息。复利滚存至到期日。" for _ in range(200)]
    text = "\n".join(paras)
    chunks = FinancialChunker(chunk_size=512, overlap=64)._split_text(text, "测试")

    assert len(chunks) >= 2
    # 关键断言：无重复膨胀 —— 每个 chunk 受控（≤ chunk_size + overlap + 一句容差）。
    # 旧实现会膨胀到 600/700/800…递增；新实现所有块都稳定在此上限内
    for c in chunks:
        assert c.token_count <= 512 + 64 + 8, f"chunk 膨胀: {c.token_count}"
    # 内容无丢失：200 段全部保留（overlap 会重复少量句子，只多不少）
    joined = "".join(c.content for c in chunks)
    assert joined.count("利率按日计息。") >= 200
    # 单段不被拆散：同一句完整保留（不被字符级硬切）
    assert "复利滚存至到期日。" in joined


def test_oversized_single_sentence_hard_split():
    """兜底：单句超 chunk_size（无句读可切）→ 字符滑窗硬切，不产生超大 chunk。"""
    # 单句 6000 字（3000 token），远超 512
    text = "无" * 6000
    chunks = FinancialChunker(chunk_size=512, overlap=64)._split_text(text, "测试")

    assert len(chunks) > 1
    for c in chunks:
        # 滑窗窗口 1024 字符 = 512 token，硬切块允许略超（窗口固定）
        assert c.token_count <= 513
    # 无内容丢失
    assert sum(c.content.count("无") for c in chunks) >= 6000


def test_overlap_present_between_chunks():
    """overlap 生效：相邻 chunk 尾部句子在下一 chunk 开头重现。"""
    text = "第一句内容。第二句内容。第三句内容。" * 240  # 720 句 × 3 token = 2160 token
    chunks = FinancialChunker(chunk_size=512, overlap=64)._split_text(text, "测试")

    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1].content.split("\n")[-1]  # 上一块最后一句
        assert prev_tail in chunks[i].content, f"chunk{i} 缺少与上一块的 overlap"


def test_small_text_single_chunk_unchanged():
    """短文本整段保留（既有行为回归）。"""
    text = "短文本不需要切分。"
    chunks = FinancialChunker()._split_text(text, "路径")
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].section_path == "路径"


def test_english_text_still_splits():
    """英文按句读切分（.!?;），行为与旧实现等价。"""
    text = "This is sentence one. Here is another sentence! " * 80  # 160 句 × ~5 token ≈ 800 token
    chunks = FinancialChunker(chunk_size=512, overlap=64)._split_text(text, "en")

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 512 + 64
    # 句子不被拆散
    assert "This is sentence one." in "".join(c.content for c in chunks)
