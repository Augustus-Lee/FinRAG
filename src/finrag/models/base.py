"""SQLAlchemy 声明式基类与通用 Mixin。"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。"""


class TimestampMixin:
    """自动维护 created_at / updated_at。"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间", nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间", nullable=False
    )
