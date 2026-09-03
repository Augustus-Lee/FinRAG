"""multipart 文件上传端点测试：落盘 + 文档创建 + 类型规范化（摄入用 stub 隔离）。"""

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from finrag.main import app
from finrag.models import KBCategory, KBDocument
from finrag.db.session import SessionLocal

client = TestClient(app)


def _admin_headers() -> dict:
    """上传需 kb_manage 权限（admin 有）；RBAC 接入后受保护端点必须带 token。"""
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _make_xlsx(path: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "客户信息"
    ws.append(["姓名", "资产"])
    ws.append(["张三", 1000000])
    wb.save(str(path))
    return path


def _stub_ingest(monkeypatch, tmp_path):
    """屏蔽真实摄入（embedding/向量库依赖）：stub pipeline 直接标记 ready。"""

    class _StubPipeline:
        def ingest_document(self, db, doc, file_path):
            doc.status = "ready"
            doc.chunk_count = 1
            db.commit()
            from finrag.pipelines.ingest import IngestResult

            return IngestResult(doc_id=doc.id, chunk_count=1, status="ready")

    # Celery dispatch 抛错 → 同步降级 → stub pipeline
    from finrag.tasks.ingest_tasks import ingest_document_task

    def _boom(*args, **kwargs):
        raise RuntimeError("test: no celery")

    monkeypatch.setattr(ingest_document_task, "delay", _boom)
    monkeypatch.setattr("finrag.container.get_ingest_pipeline", lambda: _StubPipeline())
    # 上传目录指到 tmp，不污染仓库
    monkeypatch.setattr("finrag.api.v1.knowledge.get_settings", lambda: MagicMock(upload_dir=str(tmp_path)))


def test_upload_xlsx_creates_document_and_saves_file(tmp_path, monkeypatch):
    db = SessionLocal()
    cat_id = None
    doc_id = None
    try:
        cat = KBCategory(name="上传测试库", owner_id=1)
        db.add(cat)
        db.commit()
        db.refresh(cat)
        cat_id = cat.id

        _stub_ingest(monkeypatch, tmp_path / "uploads")
        src = _make_xlsx(tmp_path / "客户.xlsx")

        with src.open("rb") as f:
            resp = client.post(
                "/api/v1/knowledge/documents/upload",
                data={"kb_id": cat_id},
                files={"file": ("客户.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                headers=_admin_headers(),
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["file_type"] == "xlsx"  # 扩展名规范化
        assert body["status"] == "ready"  # stub 摄入完成
        doc_id = body["id"]

        # 文件已落盘（uuid 前缀 + 原文件名）
        saved = list((tmp_path / "uploads").glob("*_客户.xlsx"))
        assert len(saved) == 1

        # 文档记录指向落盘路径
        doc = db.get(KBDocument, doc_id)
        assert doc.file_path == str(saved[0])
    finally:
        if doc_id:
            db.query(KBDocument).filter(KBDocument.id == doc_id).delete(synchronize_session=False)
        if cat_id:
            db.query(KBCategory).filter(KBCategory.id == cat_id).delete(synchronize_session=False)
        db.commit()
        db.close()


def test_upload_rejects_unsupported_extension(tmp_path):
    resp = client.post(
        "/api/v1/knowledge/documents/upload",
        data={"kb_id": 1},
        files={"file": ("evil.exe", b"\x00", "application/octet-stream")},
        headers=_admin_headers(),  # 带权限后才到达扩展名校验（否则 401 掩盖 415）
    )
    assert resp.status_code == 415
    assert "不支持" in resp.json()["detail"]
