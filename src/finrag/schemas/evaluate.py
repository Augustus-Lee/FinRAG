"""评估 API 契约。"""

from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    scene: str = Field(pattern=r"^(knowledge|nl2sql|dictionary)$")
    run_id: str | None = None


class EvalReportOut(BaseModel):
    id: int
    run_id: str
    scene: str
    faithfulness: float | None = None
    relevancy: float | None = None
    sql_success_rate: float | None = None
    case_count: int = 0

    model_config = {"from_attributes": True}
