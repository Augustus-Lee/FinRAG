"""系统模型：标准五表 RBAC（用户/角色/权限 + 两组关联）/ 模型配置。"""

from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finrag.models.base import Base, TimestampMixin


class SysUser(Base, TimestampMixin):
    """用户。角色经 sys_user_role 多对多关联（权限取角色并集）。

    注：线上旧表的废弃 role 列保留不删（模型不映射即可），避免破坏性 DDL。
    """

    __tablename__ = "sys_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="软删：False 后登录与鉴权均拒绝")

    # 角色多对多（selectin 加载：一次 in 查询取全，避免 joined 笛卡尔积）
    roles: Mapped[list["SysRole"]] = relationship(
        "SysRole", secondary="sys_user_role", lazy="selectin", viewonly=True
    )


class SysRole(Base, TimestampMixin):
    """角色：一组权限的命名集合。"""

    __tablename__ = "sys_role"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    role_name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(255), default="")

    # 权限多对多（viewonly：关联行只经管理 API 维护）
    permissions: Mapped[list["SysPermission"]] = relationship(
        "SysPermission", secondary="sys_role_permission", lazy="selectin", viewonly=True
    )


class SysPermission(Base):
    """权限码。场景码（knowledge/nl2sql/dictionary）与 chat mode 值天然对齐。"""

    __tablename__ = "sys_permission"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    perm_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    perm_name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(255), default="")


class SysUserRole(Base):
    """用户-角色关联（多对多）。"""

    __tablename__ = "sys_user_role"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("sys_user.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("sys_role.id", ondelete="CASCADE"), primary_key=True
    )


class SysRolePermission(Base):
    """角色-权限关联（多对多）。"""

    __tablename__ = "sys_role_permission"

    role_id: Mapped[int] = mapped_column(
        ForeignKey("sys_role.id", ondelete="CASCADE"), primary_key=True
    )
    perm_id: Mapped[int] = mapped_column(
        ForeignKey("sys_permission.id", ondelete="CASCADE"), primary_key=True
    )


class SysModelConf(Base, TimestampMixin):
    """LLM/Embedding 模型接入配置（网关数据源，支持多模型切换）。"""

    __tablename__ = "sys_model_conf"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="配置名")
    provider: Mapped[str] = mapped_column(String(16), nullable=False, comment="cloud/local")
    base_url: Mapped[str] = mapped_column(String(256), default="")
    api_key_enc: Mapped[str] = mapped_column(String(256), default="", comment="加密后的 API Key")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
