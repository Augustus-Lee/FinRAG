"""认证接口：登录签发 JWT（查库校验，RBAC 实名认证）。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finrag.api.deps import Settings, db_session, get_settings
from finrag.schemas.auth import LoginRequest, TokenResponse
from finrag.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    db: Session = Depends(db_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """登录换取 token：校验 sys_user 凭据（pbkdf2），返回角色与权限码清单。"""
    token, context = AuthService(settings).login(db, req.username, req.password)
    return TokenResponse(
        access_token=token,
        role=context.roles[0] if context.roles else "user",
        roles=context.roles,
        perms=sorted(context.perms),
    )
