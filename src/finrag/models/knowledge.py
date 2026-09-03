"""知识库相关模型：分类 / 文档 / 分块。"""

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from finrag.models.base import Base, TimestampMixin


class KBCategory(Base, TimestampMixin):
    """知识库（分类）。支持 owner + 可见角色（RBAC 基础）。"""

    __tablename__ = "kb_category"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="知识库名称")
    description: Mapped[str] = mapped_column(Text, default="", comment="描述")
    owner_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="创建人")
    visible_roles: Mapped[list] = mapped_column(JSON, default=list, comment="可见角色列表，空=所有人")


class KBDocument(Base, TimestampMixin):
    """知识库文档。状态机：pending -> parsing -> ready | failed。"""

    __tablename__ = "kb_document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("kb_category.id"), nullable=False, comment="所属知识库")
    name: Mapped[str] = mapped_column(String(256), nullable=False, comment="文件名")
    file_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="pdf/word/excel/md")
    status: Mapped[str] = mapped_column(String(16), default="pending", comment="pending/parsing/ready/failed")
    version: Mapped[int] = mapped_column(Integer, default=1, comment="版本号，重复上传自增")
    owner_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="上传人")
    file_path: Mapped[str] = mapped_column(String(512), default="", comment="存储路径")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="切分数")
    error_msg: Mapped[str] = mapped_column(Text, default="", comment="解析失败原因")


class KBChunk(Base, TimestampMixin):
    """文档分块（检索单元）。table_meta 保留表格结构；section_path 记录层级。"""

    __tablename__ = "kb_chunk"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("kb_document.id"), nullable=False, index=True)
    seq_no: Mapped[int] = mapped_column(Integer, default=0, comment="文档内顺序")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="chunk 文本")
    table_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="表格结构（若为表格块）")
    section_path: Mapped[str] = mapped_column(String(512), default="", comment="标题层级路径，如 第三章/3.2/3.2.1")
    token_count: Mapped[int] = mapped_column(Integer, default=0, comment="token 估算数")
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="向量库中的 point id")
