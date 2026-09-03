"""认证 API 契约。"""

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str = "user"  # 首个角色（兼容旧前端）
    roles: list[str] = []  # 全部角色码（菜单渲染用）
    perms: list[str] = []  # 权限码并集（按钮/场景级前端显隐用）
