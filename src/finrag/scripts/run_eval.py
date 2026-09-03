"""评估运行 CLI（M3）：独立进程入口，支持环境变量覆盖做 A/B 调参。

用法（容器内）:
    python -m finrag.scripts.run_eval --scene knowledge --run-id baseline
    python -m finrag.scripts.run_eval --scene nl2sql
    python -m finrag.scripts.run_eval --scene dictionary

A/B（环境变量覆盖检索参数）:
    docker compose exec -e FINRAG_RRF_VECTOR_WEIGHT=1.5 -e FINRAG_RRF_BM25_WEIGHT=0.7 api \
        python -m finrag.scripts.run_eval --scene knowledge --run-id weighted
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag.db.session import SessionLocal  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.services.eval_service import EvalService  # noqa: E402

logger = get_logger("finrag.run_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一次离线评估")
    parser.add_argument("--scene", required=True, choices=["knowledge", "nl2sql", "dictionary", "intent"])
    parser.add_argument("--run-id", default=None, help="运行批次号（默认随机 12 位）")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        report = EvalService().run(db, scene=args.scene, run_id=args.run_id)
    finally:
        db.close()

    detail = report.detail or {}
    aggregate = detail.get("aggregate", {})
    config = detail.get("config", {})

    print(f"\n===== 评估报告 =====")
    print(f"run_id        : {report.run_id}")
    print(f"scene         : {report.scene}")
    print(f"case_count    : {report.case_count}")
    print(f"faithfulness  : {report.faithfulness}")
    print(f"relevancy     : {report.relevancy}")
    print(f"sql_success   : {report.sql_success_rate}")
    if aggregate:
        print(f"aggregate     : {aggregate}")
    if config:
        print(f"config        : {config}")


if __name__ == "__main__":
    main()
