"""对话接口：会话管理 + 三大场景问答（登录后可用，场景级权限在 ChatService 校验）。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finrag.api.deps import db_session, get_current_user
from finrag.schemas.chat import ChatRequest, ChatResponse, MessageOut, SessionCreate, SessionOut
from finrag.services.auth_service import AuthContext
from finrag.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

_service = ChatService()


@router.post(
    "/sessions",
    response_model=SessionOut,
    status_code=201,
    dependencies=[Depends(get_current_user)],
)
def create_session(
    req: SessionCreate,
    db: Session = Depends(db_session),
    auth: AuthContext = Depends(get_current_user),
) -> SessionOut:
    return SessionOut.model_validate(
        _service.create_session(db, req.mode, req.title, auth.user_id, auth=auth)
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatResponse,
    dependencies=[Depends(get_current_user)],
)
def ask(
    session_id: int,
    req: ChatRequest,
    db: Session = Depends(db_session),
    auth: AuthContext = Depends(get_current_user),
) -> ChatResponse:
    """发送问题（mode 路由：knowledge / nl2sql / dictionary / auto，场景权限逐轮校验）。"""
    return _service.ask(db, req.question, req.mode, session_id, req.history, auth=auth)


@router.post(
    "/messages",
    response_model=ChatResponse,
    dependencies=[Depends(get_current_user)],
)
def ask_new(
    req: ChatRequest,
    db: Session = Depends(db_session),
    auth: AuthContext = Depends(get_current_user),
) -> ChatResponse:
    """新建会话提问（session_id 为空时自动建会话）。"""
    return _service.ask(db, req.question, req.mode, req.session_id, req.history, auth=auth)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageOut],
    dependencies=[Depends(get_current_user)],
)
def list_messages(
    session_id: int,
    db: Session = Depends(db_session),
    auth: AuthContext = Depends(get_current_user),
) -> list[MessageOut]:
    return [MessageOut.model_validate(m) for m in _service.list_messages(db, session_id, auth=auth)]
