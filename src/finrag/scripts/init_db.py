"""数据库建表 CLI（幂等）。

用法:
    python -m finrag.scripts.init_db

说明:
    - 基于全部 ORM 模型（Base.metadata）创建尚未存在的表；
    - 已存在的表不会修改结构（无迁移能力，正式环境请引入 Alembic）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag.db.init_db import init_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 FinRAG 元数据库表结构（幂等）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将创建的表，不执行")
    args = parser.parse_args()

    if args.dry_run:
        from finrag.models import Base

        print("将创建以下表：")
        for name in sorted(Base.metadata.tables.keys()):
            print(f"  - {name}")
        return

    init_db()
    print("数据库表结构初始化完成。")


if __name__ == "__main__":
    main()
