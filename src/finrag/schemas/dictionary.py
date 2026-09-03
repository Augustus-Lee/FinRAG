"""数据字典 API 契约。"""

from pydantic import BaseModel, Field


class DictSearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class FieldHitOut(BaseModel):
    table_name: str
    field_name: str
    field_type: str = ""
    comment: str = ""
    calibre: str = ""
    synonyms: list[str] = []


class DictSearchResponse(BaseModel):
    question: str
    hits: list[FieldHitOut]
    summary: str = ""
    latency_ms: float = 0.0


class DictTableOut(BaseModel):
    id: int
    table_name: str
    business_domain: str = ""
    description: str = ""

    model_config = {"from_attributes": True}


class DictFieldOut(BaseModel):
    id: int
    table_id: int
    field_name: str
    field_type: str = ""
    comment: str = ""
    calibre: str = ""
    synonyms: list[str] = []
    is_sensitive: bool = False

    model_config = {"from_attributes": True}
