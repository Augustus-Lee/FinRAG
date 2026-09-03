"""意图路由器（mode=auto）：两级混合路由（规则先行 + LLM 兜底）。

三场景可分性：
- nl2sql：问数据值（多少/几个/排名）——聚合/数值疑问词信号强
- dictionary：问字段元信息（口径/含义/单位）——字段/表实体 + 元信息疑问词
- knowledge：问文档内容（规则/流程/概念）——默认兜底（最通用）

两级设计：
1. 规则层（零成本零延迟）：拦截明显 case，直接路由
2. LLM 层（规则不确定时触发一次小分类调用，temperature=0）：
   few-shot 三分类 + 置信度；低置信或失败 → 默认 knowledge

多轮语境由 ChatService 层处理（有 session 沿用 session.mode，不走本路由）。
"""

import re

from finrag.core.llm_gateway import LLMGateway
from finrag.logging import get_logger

logger = get_logger("finrag.intent_router")

_MODES = ("knowledge", "nl2sql", "dictionary")

# ---- 规则层信号 ----

# nl2sql：聚合/数值疑问模式（问「数据的值」）
_NL2SQL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(多少|几个|几条|多少个|多少条)"),
    re.compile(r"(总和|总量|合计|平均|均值|占比|比例|排名|排行|前\d+|top\s*\d+)", re.IGNORECASE),
    re.compile(r"(最大|最小|最高|最低|最多|最少)"),
    re.compile(r"(超过|大于|小于|高于|低于|以上|以下).{0,12}(的|有多少|的有|的数量)"),
    re.compile(r"(统计|汇总|计数|求和|求平均)"),
]

# dictionary：元信息疑问词（问「字段的定义」）
_DICT_META_WORDS = ("口径", "含义", "意思", "定义", "单位", "什么类型", "哪种类型", "字段类型", "存在哪", "存在哪个库", "属于哪张表", "哪张表", "哪个表", "数据类型", "哪些字段", "什么字段", "都有哪些字段", "表结构")

# 字段实体信号：英文标识符词（snake_case 字段/表名的典型形态）
_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", re.IGNORECASE)

_LLM_PROMPT = """你是金融问答系统的意图分类器。判断用户问题应路由到哪个场景，只输出一个词：

- nl2sql：查询数据库中的数据值（金额、数量、排名、统计、某人的具体信息）
- dictionary：询问数据字段/表的元信息（字段的口径、含义、单位、类型、存在于哪张表）
- knowledge：询问业务规则、流程、产品概念等文档内容

示例：
"高风险产品的销售金额是多少" → nl2sql
"总资产超过100万的客户有几个" → nl2sql
"客户张三的手机号是什么" → nl2sql
"trade_amount 字段的口径是什么" → dictionary
"mobile 字段是什么含义、存在哪张表" → dictionary
"product_sales 表里有哪些字段" → dictionary
"SQL 安全校验有哪几层" → knowledge
"FinRAG 的整体架构是怎样的" → knowledge
"赎回费率的规定是什么" → knowledge

用户问题：{question}

只输出 nl2sql、dictionary、knowledge 中的一个词，不要解释。"""


class IntentRouter:
    """两级混合路由：规则层 → LLM 兜底 → 默认 knowledge。"""

    def __init__(
        self,
        llm: LLMGateway | None = None,
        confidence_threshold: float = 0.6,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._confidence_threshold = confidence_threshold
        self._enabled = enabled

    def classify(self, question: str) -> str:
        """返回 knowledge / nl2sql / dictionary。"""
        if not self._enabled:
            return "knowledge"

        # ---- 第一级：规则层 ----
        rule_mode = self._rule_classify(question)
        if rule_mode:
            logger.info("intent_routed_by_rule", mode=rule_mode, question=question[:60])
            return rule_mode

        # ---- 第二级：LLM 兜底（规则不确定的混淆带）----
        if self._llm is None:
            return "knowledge"
        try:
            mode = self._llm_classify(question)
            if mode:
                logger.info("intent_routed_by_llm", mode=mode, question=question[:60])
                return mode
        except Exception as exc:
            logger.warning("intent_llm_failed_default_knowledge", error=str(exc)[:200])

        # ---- 第三级：默认 knowledge（最通用场景）----
        return "knowledge"

    # ------------------------------------------------------------------
    # 规则层
    # ------------------------------------------------------------------
    def _rule_classify(self, question: str) -> str | None:
        q = question.strip()

        # nl2sql：命中聚合/数值疑问模式即路由（信号强，优先判定）
        if any(p.search(q) for p in _NL2SQL_PATTERNS):
            return "nl2sql"

        # dictionary：元信息疑问词 + 字段/表实体信号（两者都命中才路由，
        # 避免把"xx 的含义是什么"这类纯概念问题误判成字典查询）
        has_meta = any(w in q for w in _DICT_META_WORDS)
        has_identifier = bool(_IDENTIFIER_RE.search(q))
        if has_meta and has_identifier:
            return "dictionary"
        # 强元信息问法：即使没有英文标识符，明确问"字段/表"结构也算字典场景
        if has_meta and re.search(r"(字段|表结构|数据表|库表)", q):
            return "dictionary"

        return None  # 交给 LLM 层

    # ------------------------------------------------------------------
    # LLM 层
    # ------------------------------------------------------------------
    def _llm_classify(self, question: str) -> str | None:
        """单次小分类调用。返回合法 mode；无法解析时 None。"""
        text = self._llm.chat(
            [{"role": "user", "content": _LLM_PROMPT.format(question=question)}],
            temperature=0,
            max_tokens=8,
        ).strip().lower()
        # 防御：LLM 可能输出 "nl2sql." / "答案：knowledge" 等噪音，提取首个合法词
        for mode in _MODES:
            if mode in text:
                return mode
        logger.warning("intent_llm_unparsed", raw=text[:80])
        return None


def create_intent_router(settings, llm: LLMGateway | None = None) -> IntentRouter:
    """工厂：与 container 其他 create_* 对称。"""
    return IntentRouter(
        llm=llm,
        confidence_threshold=getattr(settings, "intent_confidence_threshold", 0.6),
        enabled=getattr(settings, "intent_router_enabled", True),
    )
