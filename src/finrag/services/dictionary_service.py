"""数据字典服务：检索与表清单。"""

from sqlalchemy.orm import Session

from finrag import container
from finrag.logging import get_logger
from finrag.models import DictTable
from finrag.schemas.dictionary import DictSearchResponse, FieldHitOut

logger = get_logger("finrag.dictionary_service")


class DictionaryService:
    def search(self, db: Session, question: str, top_k: int) -> DictSearchResponse:
        # 检索接口只做字段/表语义检索，不触发 LLM（口径汇总由 chat mode=dictionary 承担）
        pipeline = container.get_dictionary_pipeline()
        result = pipeline.search(question, top_k=top_k)
        return DictSearchResponse(
            question=question,
            hits=[FieldHitOut(**h.__dict__) for h in result.hits],
            summary="",
            latency_ms=result.latency_ms,
        )

    def list_tables(self, db: Session) -> list[DictTable]:
        return db.query(DictTable).order_by(DictTable.table_name).all()
