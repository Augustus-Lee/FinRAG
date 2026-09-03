"""金融规则切分器（自研核心组件）。

目标：保证金融文档中的「表格结构」「数字口径」「标题层级」不被切断——
- Markdown 表格块整体保留为一个 chunk（table_meta 记录表头/行数）
- 按标题层级（# 系列）记录 section_path，支持引用溯源定位
- 文本块按 token 估算切分并保留重叠
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from finrag.config import Settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
# 中文句读（。！？；）与英文句读（.!?;）后切句，保留结尾标点
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class Chunk:
    content: str
    section_path: str = ""
    table_meta: dict | None = None
    token_count: int = 0


@dataclass
class ParsedDocument:
    """解析后的文档中间表示。"""

    title: str = ""
    content: str = ""
    tables: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Chunker(ABC):
    @abstractmethod
    def split(self, document: ParsedDocument) -> list[Chunk]: ...


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 2 字符/token，英文约 4 字符/token）。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk // 2 + other // 4)


class FinancialChunker(Chunker):
    """按标题分层 + 表格整体保留 + 段落重叠切分。"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        lines = document.content.splitlines()

        # 单次遍历：文本按标题分层切分；表格块在原文位置整体成块，
        # section_path 取表格出现时的标题状态（修复旧实现取文档末尾标题导致的归属错位）
        section_stack: list[str] = []
        current_text: list[str] = []

        def flush_text() -> None:
            if current_text:
                path = " / ".join(section_stack)
                chunks.extend(self._split_text("\n".join(current_text).strip(), path))
                current_text.clear()

        def flush_table(block: list[str]) -> None:
            headers = [c.strip() for c in block[0].strip("|").split("|")]
            content = "\n".join(block)
            chunks.append(
                Chunk(
                    content=content,
                    section_path=" / ".join(section_stack),
                    table_meta={"headers": headers, "rows": len(block) - 2},
                    token_count=estimate_tokens(content),
                )
            )

        i = 0
        while i < len(lines):
            line = lines[i]
            if _TABLE_ROW_RE.match(line):
                flush_text()  # 表格前的文本先成块，保证 chunk 按文档顺序产出
                block: list[str] = []
                while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                    block.append(lines[i].strip())
                    i += 1
                if block:
                    flush_table(block)
                continue
            m = _HEADING_RE.match(line)
            if m:
                flush_text()
                level, title = len(m.group(1)), m.group(2).strip()
                section_stack = [t for t in section_stack[: level - 1]] + [title]
                i += 1
                continue
            current_text.append(line)
            i += 1
        flush_text()

        for chunk in chunks:
            chunk.token_count = chunk.token_count or estimate_tokens(chunk.content)

        return [c for c in chunks if c.content.strip()]

    def _split_text(self, text: str, section_path: str) -> list[Chunk]:
        """按句子切分 + token 回退 overlap。

        中文文档没有空白词边界：旧实现按空白切"词"，无空格的中文长段落会
        整段成为一个超大 chunk（embedding 静默截断丢内容），多段落场景则把
        前一整块当作 overlap 带走（chunk 重复膨胀）。本实现以「句」为累积单元：
        - 中英句读（。！？；.!?;）切句，句内不再切分（语义完整）
        - overlap 从上一 chunk 尾部按句回取，凑满 ~overlap token 为止
        - 超大单句（> chunk_size）按字符滑窗硬切兜底
        """
        if estimate_tokens(text) <= self._chunk_size:
            return [Chunk(content=text, section_path=section_path, token_count=estimate_tokens(text))]

        chunks: list[Chunk] = []
        current: list[str] = []

        def emit() -> None:
            nonlocal current
            if current:
                chunks.append(
                    Chunk(content="\n".join(current), section_path=section_path)
                )
                current = []

        for sent in self._iter_sentences(text):
            # 超大单句：按字符滑窗硬切（保留 overlap 字符），独立成块
            sent_tokens = estimate_tokens(sent)
            if sent_tokens > self._chunk_size:
                emit()
                chunks.extend(self._hard_split_sentence(sent, section_path))
                continue
            if current:
                # 累积计数用 join 后整体重算：换行的 token 成本只有聚合后才显现
                # （estimate_tokens 整除规则下单句 +"\n" 计 0，n 个换行聚合 ≈ n/4）
                current.append(sent)
                joined_tokens = estimate_tokens("\n".join(current))
                if joined_tokens > self._chunk_size:
                    current.pop()
                    emit()
                    current.append(sent)
            else:
                current.append(sent)
        emit()

        # overlap：每个 chunk（首块除外）前面拼上一 chunk 尾部句子（凑满 ~overlap token）。
        # 单行超过 overlap 时（如硬切产生的大行）跳过该行，避免把整块当 overlap 造成膨胀
        if self._overlap > 0:
            for idx in range(1, len(chunks)):
                prev_lines = chunks[idx - 1].content.split("\n")
                tail: list[str] = []
                tail_tokens = 0
                for line in reversed(prev_lines):
                    line_tokens = estimate_tokens(line)
                    if line_tokens > self._overlap:
                        if tail:
                            break
                        continue  # 尾行超长（硬切块）→ 跳过它继续往前找可作 overlap 的句子
                    if tail and tail_tokens + line_tokens > self._overlap:
                        break
                    tail.insert(0, line)
                    tail_tokens += line_tokens
                    if tail_tokens >= self._overlap:
                        break
                if tail:
                    chunks[idx].content = "\n".join(tail) + "\n" + chunks[idx].content

        for chunk in chunks:
            chunk.token_count = estimate_tokens(chunk.content)
        return chunks

    def _iter_sentences(self, text: str) -> list[str]:
        """切句：按中英句读分割；无句读的长串（如无标点中文）退化为整段，
        由 _hard_split_sentence 兜底。换行保留在句内，段落结构不破坏。"""
        sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        return sentences or ([text] if text.strip() else [])

    def _hard_split_sentence(self, sent: str, section_path: str) -> list[Chunk]:
        """超大单句兜底：按字符滑窗硬切，窗口间保留 overlap 字符（约 overlap*2 中文字符）。"""
        # token 估算规则：中文 2 字符/token → overlap token ≈ overlap*2 中文字符
        char_step = max(1, self._chunk_size * 2 - self._overlap * 2)
        char_window = self._chunk_size * 2
        pieces: list[str] = []
        start = 0
        while start < len(sent):
            pieces.append(sent[start : start + char_window])
            if start + char_window >= len(sent):
                break
            start += char_step
        return [
            Chunk(content=p.strip(), section_path=section_path) for p in pieces if p.strip()
        ]


def create_chunker(settings: Settings) -> FinancialChunker:
    return FinancialChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
