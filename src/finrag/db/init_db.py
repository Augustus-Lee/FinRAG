"""建表脚本：基于全部 ORM 模型创建表结构。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag.db.session import engine  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.models import Base  # noqa: E402

logger = get_logger("finrag.init_db")


def init_db() -> None:
    """创建所有未存在的表（幂等）。"""
    Base.metadata.create_all(bind=engine)
    tables = sorted(Base.metadata.tables.keys())
    logger.info("database_tables_ready", count=len(tables), tables=tables)


if __name__ == "__main__":
    init_db()
