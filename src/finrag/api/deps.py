"""API 公共依赖：DB 会话 / 配置 / 认证与权限。"""

from collections.abc import Callable, Generator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from finrag.config import Settings, get_settings
from finrag.db.session import get_db
from finrag.services.auth_service import AuthContext, AuthService, load_auth_context
from finrag.utils.errors import ForbiddenError, UnauthorizedError

__all__ = [
    "Settings",
    "get_settings",
    "get_db",
    "db_session",
    "get_current_user",
    "require_perms",
]

# auto_error=False：缺 Authorization 头时自行抛统一格式的 401（而非 FastAPI 默认 403）
_bearer_scheme = HTTPBearer(auto_error=False)


def db_session() -> Generator[Session, None, None]:
    """别名：路由层统一用 db: Session = Depends(db_session)。"""
    yield from get_db()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(db_session),
    settings: Settings = Depends(get_settings),
) -> AuthContext:
    """解析 Bearer token → 校验签名/有效期 → 从 DB 重载角色权限。

    token 只携带身份，权限每请求从 DB 装配（角色/权限撤销即时生效）。
    """
    if credentials is None:
        raise UnauthorizedError("缺少访问令牌，请先登录")
    user_id = AuthService(settings).parse_token(credentials.credentials)
    return load_auth_context(db, user_id)


def require_perms(*codes: str) -> Callable[[AuthContext], AuthContext]:
    """依赖工厂：要求持有任一给定权限码（any-of），否则 403。"""

    def checker(auth: AuthContext = Depends(get_current_user)) -> AuthContext:
        if not any(auth.has_perm(c) for c in codes):
            raise ForbiddenError(
                "无权访问：缺少权限",
                detail={"required": list(codes), "roles": auth.roles},
            )
        return auth

    return checker
