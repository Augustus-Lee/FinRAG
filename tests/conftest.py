"""测试全局配置：内存 SQLite，避免依赖外部服务（qdrant/redis/mysql）。"""

import os

# 必须在导入 finrag 之前设置（Settings 使用 lru_cache 缓存）
# FINRAG_DB_URL 强制覆盖（不能用 setdefault）：docker compose exec 环境注入的
# MySQL 地址会压过 setdefault，导致测试跑在真库上、被 seed 数据污染
os.environ["FINRAG_DB_URL"] = "sqlite:///:memory:"
os.environ.setdefault("FINRAG_DEBUG", "false")
os.environ.setdefault("FINRAG_RERANK_ENABLED", "false")
os.environ.setdefault("FINRAG_EMBEDDING_PROVIDER", "local")

import pytest  # noqa: E402

from finrag.db.init_db import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _memory_db():
    """会话级：在共享内存 SQLite 中建表 + 灌 RBAC 种子（admin 可登录）。"""
    init_db()
    from finrag.scripts.seed_rbac import seed

    seed()  # 幂等：权限/角色/admin 账号
    yield


@pytest.fixture(autouse=True)
def _isolate_rate_limiter():
    """用例级限流隔离：强制内存降级后端（容器内 Redis 真实可达，避免 ZSET 跨用例累积）
    并清空计数——限流语义测试在 test_rate_limiter.py 内自建实例覆盖双路径。"""
    from finrag import main as main_mod

    main_mod._rate_limiter._circuit_open_until = float("inf")  # 永久熔断 → 走内存
    main_mod._rate_limiter.reset_memory()
    yield
    main_mod._rate_limiter.reset_memory()
