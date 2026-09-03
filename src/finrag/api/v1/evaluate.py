"""评估接口：触发评估运行 / 查询报告（需 eval_manage 权限）。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finrag.api.deps import db_session, require_perms
from finrag.schemas.evaluate import EvalReportOut, EvalRunRequest
from finrag.services.eval_service import EvalService

# router 级保护：评估触发与报告查询同属评估管理
router = APIRouter(
    prefix="/evaluate", tags=["evaluate"], dependencies=[Depends(require_perms("eval_manage"))]
)

_service = EvalService()


@router.post("/run", response_model=EvalReportOut)
def run(req: EvalRunRequest, db: Session = Depends(db_session)) -> EvalReportOut:
    report = _service.run(db, req.scene, req.run_id)
    return EvalReportOut.model_validate(report)


@router.get("/reports", response_model=list[EvalReportOut])
def list_reports(scene: str | None = None, db: Session = Depends(db_session)) -> list[EvalReportOut]:
    return [EvalReportOut.model_validate(r) for r in _service.list_reports(db, scene)]
