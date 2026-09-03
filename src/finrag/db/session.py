"""数据库会话管理。"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finrag.config import get_settings


def _create_engine(url: str):
    """按 URL 选择池策略：
    - 内存 SQLite（测试）：StaticPool 共享单连接，保证跨线程可见
    - 文件 SQLite：check_same_thread=False，便于 FastAPI 线程池访问
    - 其他（MySQL/PG）：标准连接池 + pool_pre_ping
    """
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        if url == "sqlite:///:memory:" or "mode=memory" in url:
            return create_engine(
                url, poolclass=StaticPool, connect_args=connect_args, future=True
            )
        return create_engine(url, pool_pre_ping=True, connect_args=connect_args, future=True)
    return create_engine(url, pool_pre_ping=True, future=True)


_settings = get_settings()

engine = _create_engine(_settings.db_url)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：请求级数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
