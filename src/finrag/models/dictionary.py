"""数据字典模型：表元数据 / 字段元数据。"""

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finrag.models.base import Base, TimestampMixin


class DictTable(Base, TimestampMixin):
    """金融数据表元数据。"""

    __tablename__ = "dict_table"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, comment="表名")
    business_domain: Mapped[str] = mapped_column(String(64), default="", comment="业务域，如 交易/客户/风控")
    description: Mapped[str] = mapped_column(Text, default="", comment="表说明")
    owner: Mapped[str] = mapped_column(String(64), default="", comment="负责人")


class DictField(Base, TimestampMixin):
    """字段元数据（检索与 NL2SQL 的核心口径来源）。"""

    __tablename__ = "dict_field"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("dict_table.id"), nullable=False, index=True)
    # 反向关联：joined 加载避免逐字段 N+1 查询（字典规模小，直接 join 更简单）
    table: Mapped["DictTable"] = relationship("DictTable", lazy="joined")
    field_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="字段名")
    field_type: Mapped[str] = mapped_column(String(32), default="", comment="字段类型")
    comment: Mapped[str] = mapped_column(Text, default="", comment="字段含义")
    calibre: Mapped[str] = mapped_column(Text, default="", comment="统计口径说明（金融场景关键）")
    synonyms: Mapped[list] = mapped_column(JSON, default=list, comment="同义词/别名，辅助语义检索")
    is_sensitive: Mapped[bool] = mapped_column(default=False, comment="是否敏感字段（脱敏）")
