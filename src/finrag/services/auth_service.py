"""认证服务：密码哈希（pbkdf2）/ JWT 签发校验 / 认证上下文装配。

设计要点：
- 密码：stdlib pbkdf2_hmac（SHA-256，600k 迭代，每用户随机盐）。passlib 已停更，
  pbkdf2 零依赖且 NIST 认可。存储格式 `pbkdf2$<iter>$<salt_hex>$<hash_hex>`（≤128 字符）。
- token：PyJWT HS256，claims 最小化（sub/iat/exp）——权限不塞 token，
  每请求从 DB 重载（角色/权限撤销即时生效，不受 token TTL 拖尾）。
- 登录失败不区分「用户不存在/密码错/已停用」，统一 401 防用户枚举。
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field

import jwt
from sqlalchemy.orm import Session

from finrag.config import Settings
from finrag.logging import get_logger
from finrag.models import SysUser
from finrag.utils.errors import UnauthorizedError

logger = get_logger("finrag.auth_service")

# OWASP 2023 推荐 pbkdf2-sha256 ≥ 600k 迭代
_PBKDF2_ITERATIONS = 600_000
_HASH_ALGO = "pbkdf2_sha256"


# ---------------------------------------------------------------------------
# 密码哈希
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"{_HASH_ALGO}${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """恒定时间比较，防时序侧信道。格式异常按密码错误处理（不抛异常）。"""
    try:
        algo, iterations, salt, hash_hex = stored.split("$")
        if algo != _HASH_ALGO:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# 认证上下文
# ---------------------------------------------------------------------------

@dataclass
class AuthContext:
    """请求级认证上下文：身份 + 角色 + 权限并集。"""

    user_id: int
    username: str
    roles: list[str] = field(default_factory=list)
    perms: set[str] = field(default_factory=set)

    def has_perm(self, code: str) -> bool:
        return code in self.perms


def load_auth_context(db: Session, user_id: int) -> AuthContext:
    """从 DB 装配认证上下文（selectin 级联：用户 → 角色 → 权限各一次查询）。

    用户不存在或已停用 → 401（token 可能指向已被停用的账号）。
    """
    user = db.get(SysUser, user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("账号不存在或已被停用")
    roles = [r.role_code for r in user.roles]
    perms = {p.perm_code for r in user.roles for p in r.permissions}
    return AuthContext(user_id=user.id, username=user.username, roles=roles, perms=perms)


# ---------------------------------------------------------------------------
# 认证服务
# ---------------------------------------------------------------------------

class AuthService:
    """登录与 token 签发。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ---- JWT ----
    def _issue_token(self, user_id: int) -> str:
        now = int(time.time())
        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": now + int(self._settings.auth_token_expire_hours * 3600),
        }
        return jwt.encode(payload, self._settings.secret_key, algorithm="HS256")

    def parse_token(self, token: str) -> int:
        """校验签名与有效期，返回 user_id；任何失败统一 401。"""
        try:
            payload = jwt.decode(token, self._settings.secret_key, algorithms=["HS256"])
            return int(payload["sub"])
        except jwt.ExpiredSignatureError:
            raise UnauthorizedError("登录已过期，请重新登录") from None
        except jwt.InvalidTokenError:
            raise UnauthorizedError("无效的访问令牌") from None

    # ---- 登录 ----
    def login(self, db: Session, username: str, password: str) -> tuple[str, AuthContext]:
        """校验凭据，返回 (token, 认证上下文)。失败统一 401（防用户枚举）。"""
        user = db.query(SysUser).filter(SysUser.username == username).first()
        if user is None or not verify_password(password, user.password_hash):
            raise UnauthorizedError("用户名或密码错误")
        if not user.is_active:
            raise UnauthorizedError("用户名或密码错误")  # 停用与密码错同提示
        context = load_auth_context(db, user.id)
        logger.info("user_login", user_id=user.id, username=username, roles=context.roles)
        return self._issue_token(user.id), context
