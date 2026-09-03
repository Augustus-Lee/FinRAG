"""ChatService 知识问答引用序列化测试。

回归点：RAG pipeline 返回的 citations 是 dataclass（finrag.pipelines.rag.Citation，
无 model_dump()），ChatService 必须能转成 pydantic Citation 写库 + 返回响应。
曾因误用 dataclass.model_dump() 报 AttributeError（流式改造后 LLM 不再超时、路径打通才暴露）。
"""

from finrag.db.session import SessionLocal
from finrag.models import ChatMessage, ChatSession
from finrag.pipelines.rag import Citation as RAGCitation
from finrag.pipelines.rag import RAGAnswer
from finrag.services.chat_service import ChatService


def _rag_answer() -> RAGAnswer:
    """带真实 dataclass Citation 的 RAGAnswer（正是触发回归的对象类型）。"""
    return RAGAnswer(
        answer="答案权威",
        citations=[RAGCitation(chunk_id="c1", content="片段内容", section_path="s", score=0.9)],
        latency_ms=12.3,
        retrieved_contexts=["片段内容"],
    )


def _cleanup(db, session_id: int) -> None:
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.id == session_id).delete(synchronize_session=False)
    db.commit()


def test_knowledge_citations_serialized_without_model_dump(monkeypatch):
    db = SessionLocal()
    try:
        svc = ChatService()
        monkeypatch.setattr(svc, "_dispatch", lambda q, m, h: _rag_answer())

        resp = svc.ask(db, "问题", "knowledge", session_id=None, history=None)

        assert resp.mode == "knowledge"
        assert resp.answer == "答案权威"
        assert len(resp.citations) == 1
        c = resp.citations[0]
        assert c.chunk_id == "c1" and c.score == 0.9 and c.content == "片段内容"
        # 响应 citation 是 pydantic 模型，可 model_dump（dataclass 不可，正是回归点）
        assert c.model_dump()["chunk_id"] == "c1"

        # DB 中 assistant 消息 citations 列写入的是 dict 列表（可 JSON 序列化）
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == resp.session_id, ChatMessage.role == "assistant")
            .all()
        )
        assert msgs and msgs[0].citations is not None
        assert msgs[0].citations[0]["chunk_id"] == "c1"

        _cleanup(db, resp.session_id)
    finally:
        db.close()


def test_ask_rewrites_residual_question_before_dispatch(monkeypatch):
    """接线回归：多轮残缺问题（「那6月呢」）经改写器补全后才进 pipeline，DB 存用户原文。"""
    import finrag.container as container

    db = SessionLocal()
    try:
        svc = ChatService()
        captured: dict = {}

        def fake_dispatch(q, m, h):
            captured["question"] = q
            captured["history"] = h
            return _rag_answer()

        monkeypatch.setattr(svc, "_dispatch", fake_dispatch)

        class StubRewriter:
            def rewrite(self, question, history, mode):
                return "2024年6月股票交易的总金额是多少"

        monkeypatch.setattr(container, "get_query_rewriter", lambda: StubRewriter())

        history = [
            {"role": "user", "content": "2024年7月股票交易的总金额是多少"},
            {"role": "assistant", "content": "7月股票交易总金额为 1,200 万元。"},
        ]
        resp = svc.ask(db, "那6月呢", "nl2sql", session_id=None, history=history)

        # pipeline 拿到改写后的自包含问题，且 history 原样透传
        assert captured["question"] == "2024年6月股票交易的总金额是多少"
        assert captured["history"] == history

        # DB 中 user 消息存的是用户原文（改写只是内部执行细节）
        user_msg = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == resp.session_id, ChatMessage.role == "user")
            .first()
        )
        assert user_msg.content == "那6月呢"

        _cleanup(db, resp.session_id)
    finally:
        db.close()
