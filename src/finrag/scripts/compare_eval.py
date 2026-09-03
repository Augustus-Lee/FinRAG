"""评估对比 CLI（M3）：两次评估运行的指标对比 + 配置 diff，产出优化前后对比报告。

用法（容器内）:
    python -m finrag.scripts.compare_eval --run-id-a baseline --run-id-b weighted
    python -m finrag.scripts.compare_eval --run-id-a baseline --run-id-b weighted \
        --out docs/eval-reports/m3-ab-report.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finrag.db.session import SessionLocal  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.models import EvalReport  # noqa: E402

logger = get_logger("finrag.compare_eval")

METRICS = [
    ("faithfulness", "忠实度"),
    ("relevancy", "答案相关性"),
    ("sql_success_rate", "SQL 执行成功率"),
]

AGGREGATE_METRICS = [
    "hit_rate_at_5",
    "recall_at_5",
    "mrr",
    "golden_match_rate",
    "ragas_available",
    "accuracy",
]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(a, b) -> str:
    if a is None or b is None or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "-"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _load(db, run_id: str, scene: str | None = None) -> EvalReport | None:
    query = db.query(EvalReport).filter(EvalReport.run_id == run_id)
    if scene:
        query = query.filter(EvalReport.scene == scene)
    return query.order_by(EvalReport.id.desc()).first()


def build_report(ra: EvalReport, rb: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# 评估对比报告（M3）")
    lines.append("")
    lines.append(f"- A（基线）: run_id=`{ra.run_id}` scene={ra.scene} cases={ra.case_count}")
    lines.append(f"- B（对比）: run_id=`{rb.run_id}` scene={rb.scene} cases={rb.case_count}")
    lines.append("")

    lines.append("## 指标对比")
    lines.append("")
    lines.append("| 指标 | A | B | Δ |")
    lines.append("|---|---|---|---|")
    for key, label in METRICS:
        va, vb = getattr(ra, key), getattr(rb, key)
        lines.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {_delta(va, vb)} |")

    agg_a = (ra.detail or {}).get("aggregate", {})
    agg_b = (rb.detail or {}).get("aggregate", {})
    for key in AGGREGATE_METRICS:
        if key in agg_a or key in agg_b:
            va, vb = agg_a.get(key), agg_b.get(key)
            lines.append(f"| {key} | {_fmt(va)} | {_fmt(vb)} | {_delta(va, vb)} |")
    lines.append("")

    lines.append("## 配置快照 diff（仅展示有差异的参数）")
    lines.append("")
    cfg_a = (ra.detail or {}).get("config", {})
    cfg_b = (rb.detail or {}).get("config", {})
    diff = {k for k in set(cfg_a) | set(cfg_b) if cfg_a.get(k) != cfg_b.get(k)}
    if diff:
        lines.append("| 参数 | A | B |")
        lines.append("|---|---|---|")
        for k in sorted(diff):
            lines.append(f"| {k} | {_fmt(cfg_a.get(k))} | {_fmt(cfg_b.get(k))} |")
    else:
        lines.append("两次运行检索参数完全一致（仅系统/数据状态不同）。")
    lines.append("")

    lines.append("## A 全量配置")
    lines.append("```json")
    import json

    lines.append(json.dumps(cfg_a, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两次评估运行")
    parser.add_argument("--run-id-a", required=True, help="基线 run_id")
    parser.add_argument("--run-id-b", required=True, help="对比 run_id")
    parser.add_argument("--scene", default=None, choices=["knowledge", "nl2sql", "dictionary"],
                        help="按场景过滤（同一 run_id 多场景时区分）")
    parser.add_argument("--out", default=None, help="输出 Markdown 路径（默认仅打印）")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        ra = _load(db, args.run_id_a, args.scene)
        rb = _load(db, args.run_id_b, args.scene)
        if ra is None or rb is None:
            missing = [rid for rid, r in ((args.run_id_a, ra), (args.run_id_b, rb)) if r is None]
            print(f"错误：未找到评估运行 {missing}（scene={args.scene}，可用 run_id 见 eval_report 表）")
            sys.exit(1)

        content = build_report(ra, rb)
        print(content)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content + "\n", encoding="utf-8")
            print(f"\n已写入: {out_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
