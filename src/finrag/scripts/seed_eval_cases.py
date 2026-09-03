"""评估集种子脚本（M3）：灌入 45 题 EvalCase + 幂等上传知识库语料。

用法（容器内）:
    python -m finrag.scripts.seed_eval_cases               # 仅灌评估集
    python -m finrag.scripts.seed_eval_cases --upload-docs # 同时上传知识库语料（knowledge 评估前置）

数据文件：data/eval/{dictionary,nl2sql,knowledge}_cases.jsonl
幂等：按 (scene, question) 存在则更新；文档按文件名查重跳过。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag import container  # noqa: E402
from finrag.db.init_db import init_db  # noqa: E402
from finrag.db.session import SessionLocal  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.models import EvalCase, KBCategory, KBDocument  # noqa: E402

logger = get_logger("finrag.seed_eval_cases")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"

CASE_FILES = ["dictionary_cases.jsonl", "nl2sql_cases.jsonl", "knowledge_cases.jsonl"]

# knowledge 评估语料（相对仓库根）
CORPUS_FILES = [
    "README.md",
    "docs/FinRAG-需求分析.md",
    "docs/specs/001-project-framework.md",
]


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for name in CASE_FILES:
        path = DATA_DIR / name
        if not path.exists():
            logger.warning("eval_case_file_missing", file=str(path))
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def seed_cases(db) -> dict[str, int]:
    stats = {"inserted": 0, "updated": 0}
    for c in load_cases():
        row = (
            db.query(EvalCase)
            .filter(EvalCase.scene == c["scene"], EvalCase.question == c["question"])
            .first()
        )
        if row is None:
            db.add(
                EvalCase(
                    scene=c["scene"],
                    question=c["question"],
                    golden_answer=c.get("golden_answer"),
                    golden_sql=c.get("golden_sql"),
                    expected_chunks=c.get("expected"),
                )
            )
            stats["inserted"] += 1
        else:
            row.golden_answer = c.get("golden_answer")
            row.golden_sql = c.get("golden_sql")
            row.expected_chunks = c.get("expected")
            stats["updated"] += 1
    db.commit()
    return stats


def upload_docs(db) -> list[str]:
    """幂等上传语料到知识库（不经 HTTP，直接同步摄入）。"""
    uploaded: list[str] = []

    # 确保默认知识库存在
    cat = db.query(KBCategory).filter(KBCategory.name == "评估语料库").first()
    if cat is None:
        cat = KBCategory(name="评估语料库", description="M3 评估用知识库语料", owner_id=1)
        db.add(cat)
        db.commit()
        db.refresh(cat)

    root = Path(__file__).resolve().parents[3]
    pipeline = container.get_ingest_pipeline()
    for rel in CORPUS_FILES:
        path = root / rel
        if not path.exists():
            logger.warning("corpus_file_missing", file=str(path))
            continue
        name = path.name
        existing = db.query(KBDocument).filter(KBDocument.name == name).first()
        if existing is not None:
            logger.info("corpus_doc_skipped", name=name, status=existing.status)
            continue
        doc = KBDocument(
            kb_id=cat.id, name=name, file_type="md", file_path=str(path), owner_id=1
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        result = pipeline.ingest_document(db, doc, str(path))
        uploaded.append(f"{name}({result.chunk_count} chunks)")
        logger.info("corpus_doc_ingested", name=name, chunks=result.chunk_count)
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="灌入 M3 评估集")
    parser.add_argument("--upload-docs", action="store_true", help="同时上传知识库语料")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        stats = seed_cases(db)
        print(f"评估集：新增 {stats['inserted']} 条，更新 {stats['updated']} 条")

        if args.upload_docs:
            uploaded = upload_docs(db)
            if uploaded:
                print(f"知识库语料：{len(uploaded)} 份已入库 → {', '.join(uploaded)}")
            else:
                print("知识库语料：无需上传（已存在或文件缺失）")
    finally:
        db.close()


if __name__ == "__main__":
    main()
