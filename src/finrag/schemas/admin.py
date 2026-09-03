"""管理 API 契约（用户/角色/权限，需 system_manage 权限）。"""

from pydantic import BaseModel, Field


class PermissionOut(BaseModel):
    id: int
    perm_code: str
    perm_name: str
    description: str = ""


# ---- 角色 ----

class RoleCreate(BaseModel):
    role_code: str = Field(min_length=2, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")
    role_name: str = Field(min_length=1, max_length=64)
    description: str = ""
    perm_codes: list[str] = []


class RoleUpdate(BaseModel):
    role_name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = None
    perm_codes: list[str] | None = None  # 全量替换


class RoleOut(BaseModel):
    id: int
    role_code: str
    role_name: str
    description: str = ""
    perm_codes: list[str] = []


# ---- 用户 ----

class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=6, max_length=128)
    role_ids: list[int] = []
    is_active: bool = True


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=128)  # 重置密码
    role_ids: list[int] | None = None  # 全量替换
    is_active: bool | None = None  # 停用/启用（软删）


class UserOut(BaseModel):
    id: int
    username: str
    is_active: bool
    roles: list[str] = []
