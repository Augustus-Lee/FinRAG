"""对话服务：三大场景统一入口。

链路：意图路由（mode=auto）→ 场景权限校验（RBAC）→ 查询改写（多轮指代消解）→ 分发 pipeline。
"""

from sqlalchemy.orm import Session

from finrag import container
from finrag.logging import get_logger
from finrag.models import ChatMessage, ChatSession
from finrag.schemas.chat import ChatResponse
from finrag.schemas.common import Citation
from finrag.services.auth_service import AuthContext
from finrag.utils.errors import ForbiddenError, NotFoundError

logger = get_logger("finrag.chat_service")

# 三大场景码与权限码一一对应（knowledge/nl2sql/dictionary）
_SCENE_PERMS = ("knowledge", "nl2sql", "dictionary")


class ChatService:
    def create_session(
        self,
        db: Session,
        mode: str,
        title: str,
        user_id: int = 1,
        auth: AuthContext | None = None,
    ) -> ChatSession:
        # auto 会话不锁定场景：每轮重新路由，无需校验具体场景权限（逐轮校验）
        if mode in _SCENE_PERMS and auth is not None:
            self._ensure_scene_perm(auth, mode)
        session = ChatSession(user_id=user_id, mode=mode, title=title or f"{mode} 会话")
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def ask(
        self,
        db: Session,
        question: str,
        mode: str,
        session_id: int | None,
        history: list[dict] | None,
        auth: AuthContext | None = None,
    ) -> ChatResponse:
        user_id = auth.user_id if auth else 1

        # 会话创建/校验
        if session_id is None:
            # mode=auto：新会话先经意图路由解析为具体场景，再落会话（后续轮次继承）
            if mode == "auto":
                mode = self._resolve_intent(question)
            # 落库前校验场景权限（auto 已解析为具体场景）
            self._ensure_scene_perm(auth, mode)
            session = self.create_session(db, mode, question[:40], user_id, auth=auth)
        else:
            session = db.get(ChatSession, session_id)
            if not session:
                raise NotFoundError(f"会话不存在: session_id={session_id}")
            # 会话归属：他人会话不可访问
            if auth is not None and session.user_id != auth.user_id:
                raise ForbiddenError("无权访问该会话")
            # 已有会话以 session.mode 为准（忽略请求体 mode，避免不一致）；
            # 例外：会话是 auto（历史版本/手动创建）时按当前问题重新路由
            mode = session.mode if session.mode != "auto" else self._resolve_intent(question)
            # 继承/重路由后的场景同样要过权限
            self._ensure_scene_perm(auth, mode)

        db.add(ChatMessage(session_id=session.id, role="user", content=question))
        db.commit()

        # 查询改写（意图识别之后、pipeline 分发之前）：多轮指代消解/省略补全。
        # 短路层零成本：无历史或无指代信号 → 原样透传；DB 存用户原文，pipeline 用改写后问题。
        effective_question = self._rewrite_query(question, history, mode)

        answer = self._dispatch(effective_question, mode, history)

        # 统一构造 pydantic Citation 列表（answer.citations 是 pipelines.rag 的 dataclass，
        # 无 model_dump()；先转 pydantic，DB JSON 列与响应共用，避免重复转换）
        citations: list[Citation] = []
        if hasattr(answer, "citations"):
            citations = [
                Citation(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    section_path=c.section_path,
                    score=c.score,
                )
                for c in answer.citations
            ]

        msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer.answer,
            citations=[c.model_dump() for c in citations] or None,
            sql_text=getattr(answer, "sql", None) or None,
            sql_affected_rows=getattr(answer, "affected_rows", None),
            latency_ms=answer.latency_ms,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)

        return ChatResponse(
            answer=answer.answer,
            session_id=session.id,
            message_id=msg.id,
            mode=mode,
            citations=citations,
            sql=getattr(answer, "sql", None) or None,
            affected_rows=getattr(answer, "affected_rows", None),
            columns=getattr(answer, "columns", None),
            rows=(getattr(answer, "rows", []) if len(getattr(answer, "rows", [])) <= 10 else None),
            latency_ms=answer.latency_ms,
        )

    def list_messages(
        self, db: Session, session_id: int, auth: AuthContext | None = None
    ) -> list[ChatMessage]:
        session = db.get(ChatSession, session_id)
        if not session:
            raise NotFoundError(f"会话不存在: session_id={session_id}")
        if auth is not None and session.user_id != auth.user_id:
            raise ForbiddenError("无权访问该会话")
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
            .all()
        )

    # ------------------------------------------------------------------
    # 权限 / 路由 / 改写
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_scene_perm(auth: AuthContext | None, mode: str) -> None:
        """场景权限校验：mode 与权限码一一对应（knowledge/nl2sql/dictionary）。

        未认证（auth=None，内部调用/脚本）放行；auto 由调用方在解析后校验。
        """
        if auth is None or mode == "auto":
            return
        if not auth.has_perm(mode):
            raise ForbiddenError(
                f"当前角色无 {mode} 场景权限，请联系管理员开通",
                detail={"required_perm": mode, "roles": auth.roles},
            )

    def _resolve_intent(self, question: str) -> str:
        """mode=auto 时经意图路由器解析（两级混合：规则 + LLM 兜底）。"""
        return container.get_intent_router().classify(question)

    def _rewrite_query(self, question: str, history: list[dict] | None, mode: str) -> str:
        """查询改写：意图识别后、分发前，把残缺问题补全为 self-contained question。"""
        return container.get_query_rewriter().rewrite(question, history, mode)

    def _dispatch(self, question: str, mode: str, history: list[dict] | None):
        """按场景路由。"""
        if mode == "nl2sql":
            return container.get_nl2sql_pipeline().answer(question)
        if mode == "dictionary":
            return container.get_dictionary_pipeline().answer(question)
        # knowledge：需先向量化问题
        query_vector = container.get_embedding().embed_query(question)
        return container.get_rag_pipeline().answer(question, query_vector, history=history)
