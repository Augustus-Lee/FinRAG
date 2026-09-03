"""查询改写器（意图识别之后、pipeline 分发之前）。

定位：多轮对话中的指代消解与省略补全，把残缺问题（"那6月呢"）改写成
self-contained question，让三个 pipeline（nl2sql / dictionary / knowledge）
都拿到完整约束，无需各自感知对话历史。

设计（与 intent_router 同构的两级思路）：
1. 短路层（零成本零延迟）：无历史（首轮）或问题不含指代/省略信号 → 原样返回，不碰 LLM。
   检测故意偏召回：漏检 = 多轮链路损坏，误检 = 一次 LLM 调用原样返回，代价不对称。
2. LLM 层（按意图定制改写目标，temperature=0）：
   - nl2sql：补全时间范围/筛选条件/聚合对象（SQL 生成需要完整约束）
   - dictionary：口语术语归一到历史中出现过的标准字段/表名
   - knowledge：指代消解 + 补全主语主题（提升检索召回）
3. 防御：LLM 输出经清洗与校验（去前缀/取首行/拒答词/长度膨胀检查），
   不可用则回退原文——改写失败绝不阻断问答主链路。
"""

import re

from finrag.core.llm_gateway import LLMGateway
from finrag.logging import get_logger

logger = get_logger("finrag.query_rewriter")

# 进入 prompt 的历史条数上限（约 3 轮对话）
_MAX_HISTORY_MESSAGES = 6
# 单条历史消息截断长度（控制 prompt 规模）
_MAX_MESSAGE_CHARS = 300

# ---- 短路层信号：指代 / 省略 / 续问 ----
# 命中任一模式才认为问题可能不自包含，需要 LLM 结合历史改写
_ANAPHORA_PATTERNS: list[re.Pattern] = [
    # 代词：他们/它们/它，独立的他/她（lookbehind 排除「其他」）
    re.compile(r"(他们|她们|它们|它|(?<!其)他(?!们)|(?<!其)她(?!们))"),
    # 指示代词 + 量词：这个/那笔/这些/那种...
    re.compile(r"(这|那)(个|些|笔|支|只|条|家|项|种|次|份|张|款|名|位)"),
    # 句首承接/转换话题：那 6 月呢 / 换成张三
    re.compile(r"^(那|那么|则|换个?成?)[，,？?\s]*"),
    # 文内回指：该（排除「应该」）/ 此（排除「此时/此外」）/ 其（排除「其中/其他」）
    # 之前（排除「2024年之前」类时间表达式）
    re.compile(r"(?<!应)该|此(?![外时])|其(?![中他])|上述|上面|前面|(?<![0-9年月日时分])之前|上一个|上一条"),
    # 续问/追加
    re.compile(r"(继续|接着|再说|详细说|展开说|还有|另外|再加)"),
    # 同类省略
    re.compile(r"(同样|同月|同年|同一|当月|当年)"),
    # 句尾「呢」：省略问句的典型形态（那 6 月呢 / 换成 6 月呢）
    re.compile(r"呢[?？。!！]?\s*$"),
]

# ---- 按意图定制的改写目标 ----
_REWRITE_INSTRUCTIONS = {
    "nl2sql": (
        "你是智能问数系统的查询改写器。把用户最新问题改写为一个独立完整、"
        "可直接翻译成 SQL 的查询问题：结合对话历史补全省略的时间范围、筛选条件、"
        "分组维度与聚合对象；保留最新问题中的全部数值、实体与排序要求；"
        "历史中不存在的信息不要编造。"
    ),
    "dictionary": (
        "你是数据字典问答系统的查询改写器。把用户最新问题改写为一个独立完整的"
        "字段/表元信息查询问题：将口语化业务术语归一为对话历史中出现过的标准字段名"
        "或表名（如「清算金额」→ settle_amount）；补全省略的查询对象；"
        "历史中未出现过的字段不要引入。"
    ),
    "knowledge": (
        "你是知识库问答系统的查询改写器。把用户最新问题改写为一个独立完整的检索问题："
        "消解指代（它/这个/该 → 历史中的具体对象）；补全省略的主语与主题；"
        "保留专业术语原文，不要扩展出历史之外的无关内容。"
    ),
}

_REWRITE_PROMPT = """{instruction}

对话历史：
{history}

用户最新问题：{question}

只输出改写后的一个问题，不要解释、不要加引号；若最新问题本身已经完整，原样输出它。"""

# LLM 输出防御
_NOISE_PREFIX_RE = re.compile(r"^(改写后(的问题)?|重写后|新问题|问题)\s*[:：]\s*")
_REFUSAL_WORDS = ("无法", "不能", "不需要", "已完整", "抱歉")


class QueryRewriter:
    """两级查询改写：短路层 → LLM 改写 → 防御性解析，任何失败回退原文。"""

    def __init__(self, llm: LLMGateway | None = None, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled

    def rewrite(self, question: str, history: list[dict] | None, mode: str = "knowledge") -> str:
        """返回改写后的问题；不可改写/LLM 失败时返回原文（永不抛异常）。"""
        if not self._enabled or not question:
            return question
        # 短路层：无历史（首轮）或问题已自包含 → 零成本透传
        if not history or not self._needs_rewrite(question):
            return question
        if self._llm is None:
            return question

        try:
            rewritten = self._llm_rewrite(question, history, mode)
        except Exception as exc:
            logger.warning("query_rewrite_failed_use_original", error=str(exc)[:200])
            return question
        if not rewritten:
            return question
        logger.info("query_rewritten", mode=mode, original=question[:60], rewritten=rewritten[:80])
        return rewritten

    # ------------------------------------------------------------------
    # 短路层：指代/省略信号检测
    # ------------------------------------------------------------------
    def _needs_rewrite(self, question: str) -> bool:
        return any(p.search(question) for p in _ANAPHORA_PATTERNS)

    # ------------------------------------------------------------------
    # LLM 层
    # ------------------------------------------------------------------
    def _llm_rewrite(self, question: str, history: list[dict], mode: str) -> str | None:
        """单次改写调用。返回清洗后的新问题；不可用时 None。"""
        instruction = _REWRITE_INSTRUCTIONS.get(mode, _REWRITE_INSTRUCTIONS["knowledge"])
        prompt = _REWRITE_PROMPT.format(
            instruction=instruction,
            history=self._format_history(history),
            question=question,
        )
        text = self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        return self._clean_output(text, question)

    def _format_history(self, history: list[dict]) -> str:
        """取最近 N 条消息，逐条截断（控制 prompt 规模）。"""
        lines: list[str] = []
        for h in history[-_MAX_HISTORY_MESSAGES:]:
            if not isinstance(h, dict):
                continue
            role = "用户" if h.get("role") == "user" else "助手"
            content = str(h.get("content") or "")[:_MAX_MESSAGE_CHARS]
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) or "（无）"

    # ------------------------------------------------------------------
    # 输出防御：LLM 可能输出前缀噪音/多行解释/拒答/异常膨胀，均回退原文
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_output(text: str, question: str) -> str | None:
        out = (text or "").strip()
        if not out:
            return None
        out = out.splitlines()[0].strip()  # 偶发多行解释，取首行
        out = _NOISE_PREFIX_RE.sub("", out).strip()
        out = out.strip("\"'“”‘’「」『』")
        if not out or out == question:
            return None
        if any(w in out for w in _REFUSAL_WORDS):
            return None
        if len(out) > max(120, len(question) * 3):  # 异常膨胀 → 疑似输出了解释
            return None
        return out


def create_query_rewriter(settings, llm: LLMGateway | None = None) -> QueryRewriter:
    """工厂：与 container 其他 create_* 对称。"""
    return QueryRewriter(
        llm=llm,
        enabled=getattr(settings, "query_rewrite_enabled", True),
    )
