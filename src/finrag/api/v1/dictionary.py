"""数据字典接口：语义检索 + 表清单（需 dictionary 权限）。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finrag.api.deps import db_session, require_perms
from finrag.schemas.dictionary import (
    DictFieldOut,
    DictSearchRequest,
    DictSearchResponse,
    DictTableOut,
)
from finrag.services.dictionary_service import DictionaryService

# router 级保护：三个端点都是字典场景的只读检索
router = APIRouter(
    prefix="/dictionary", tags=["dictionary"], dependencies=[Depends(require_perms("dictionary"))]
)

_service = DictionaryService()


@router.post("/search", response_model=DictSearchResponse)
def search(req: DictSearchRequest, db: Session = Depends(db_session)) -> DictSearchResponse:
    return _service.search(db, req.question, req.top_k)


@router.get("/tables", response_model=list[DictTableOut])
def list_tables(db: Session = Depends(db_session)) -> list[DictTableOut]:
    return [DictTableOut.model_validate(t) for t in _service.list_tables(db)]


@router.get("/tables/{table_id}/fields", response_model=list[DictFieldOut])
def list_fields(table_id: int, db: Session = Depends(db_session)) -> list[DictFieldOut]:
    from finrag.models import DictField

    fields = db.query(DictField).filter(DictField.table_id == table_id).all()
    return [DictFieldOut.model_validate(f) for f in fields]
