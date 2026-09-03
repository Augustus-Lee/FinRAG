"""通用 schema：引用 / 错误。"""

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    content: str
    section_path: str = ""
    score: float = 0.0


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict = {}
