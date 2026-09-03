"""知识库接口：分类 / 文档上传与状态 / 删除。

权限分级：读（查文档）需 knowledge；写（上传/删除/建分类/建文档）需 kb_manage。
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from finrag.api.deps import db_session, require_perms
from finrag.config import get_settings
from finrag.core.document_parser import _SUPPORTED
from finrag.schemas.knowledge import (
    DocumentCreateRequest,
    DocumentOut,
    KBCategoryCreate,
    KBCategoryOut,
)
from finrag.services.knowledge_service import KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_service = KnowledgeService()


@router.post(
    "/categories",
    response_model=KBCategoryOut,
    status_code=201,
    dependencies=[Depends(require_perms("kb_manage"))],
)
def create_category(req: KBCategoryCreate, db: Session = Depends(db_session)) -> KBCategoryOut:
    cat = _service.create_category(db, req.name, req.description)
    return KBCategoryOut.model_validate(cat)


@router.post(
    "/documents",
    response_model=DocumentOut,
    status_code=201,
    dependencies=[Depends(require_perms("kb_manage"))],
)
def create_document(req: DocumentCreateRequest, db: Session = Depends(db_session)) -> DocumentOut:
    doc = _service.create_document(db, req.kb_id, req.name, req.file_type, req.file_path)
    return DocumentOut.model_validate(doc)


@router.post(
    "/documents/upload",
    response_model=DocumentOut,
    status_code=201,
    dependencies=[Depends(require_perms("kb_manage"))],
    summary="multipart 文件上传（落盘后走摄入链路）",
)
async def upload_document(
    kb_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(db_session),
) -> DocumentOut:
    """上传真实文件：校验扩展名 → 落盘 upload_dir → 创建文档并触发摄入。

    file_type 按扩展名规范化（docx/xlsx/pdf/md/txt...），与解析器白名单对齐。
    """
    raw_name = Path(file.filename or "").name  # 防路径穿越，仅取文件名
    ext = Path(raw_name).suffix.lower().lstrip(".")
    if ext not in _SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型: .{ext}（支持 {sorted(_SUPPORTED - {'pdfs', 'word', 'excel'})}）",
        )

    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = upload_dir / f"{uuid.uuid4().hex[:12]}_{raw_name}"
    content = await file.read()
    saved.write_bytes(content)

    doc = _service.create_document(db, kb_id, raw_name, ext, str(saved))
    return DocumentOut.model_validate(doc)


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentOut,
    dependencies=[Depends(require_perms("knowledge", "kb_manage"))],
)
def get_document(doc_id: int, db: Session = Depends(db_session)) -> DocumentOut:
    return DocumentOut.model_validate(_service.get_document(db, doc_id))


@router.delete(
    "/documents/{doc_id}",
    status_code=204,
    dependencies=[Depends(require_perms("kb_manage"))],
)
def delete_document(doc_id: int, db: Session = Depends(db_session)) -> None:
    _service.delete_document(db, doc_id)
