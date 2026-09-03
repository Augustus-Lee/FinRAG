"""知识库 API 契约。"""

from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    kb_id: int
    name: str = Field(min_length=1, max_length=256)
    file_type: str = Field(pattern=r"^(pdf|md|markdown|txt|word|excel|docx|xlsx)$")
    file_path: str


class DocumentOut(BaseModel):
    id: int
    kb_id: int
    name: str
    file_type: str
    status: str
    version: int
    chunk_count: int = 0
    error_msg: str = ""

    model_config = {"from_attributes": True}


class KBCategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""


class KBCategoryOut(BaseModel):
    id: int
    name: str
    description: str
    owner_id: int

    model_config = {"from_attributes": True}
