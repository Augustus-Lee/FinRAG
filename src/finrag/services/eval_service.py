"""评估服务（M3）：三场景离线评估闭环。

- knowledge：RAGPipeline 逐题执行 → RAGAS 忠实度/答案相关性
- nl2sql：NL2SQLPipeline 逐题执行 → 执行成功率 + 金标准行集合对照
- dictionary：DictionaryPipeline 检索 → hit_rate@5 / recall@5 / MRR

报告 detail 结构：
    {"cases": [...逐题明细...], "aggregate": {...聚合指标...},
     "config": {...检索参数快照（A/B 对比可解释性）...}}
"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from finrag import container
from finrag.config import get_settings
from finrag.logging import get_logger
from finrag.models import EvalCase, EvalReport

logger = get_logger("finrag.eval_service")


def _config_snapshot() -> dict:
    """检索参数快照：支撑 A/B 对比时区分'参数变了'还是'系统变了'。"""
    s = get_settings()
    return {
        "rrf_k": s.rrf_k,
        "rrf_vector_weight": s.rrf_vector_weight,
        "rrf_bm25_weight": s.rrf_bm25_weight,
        "retrieve_top_k": s.retrieve_top_k,
        "rerank_top_k": s.rerank_top_k,
        "rerank_enabled": s.rerank_enabled,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "llm_model": s.llm_cloud_model,
        "embedding_model": s.embedding_api_model or s.embedding_model,
    }


def _rows_to_multiset(rows: list[list]) -> set[tuple]:
    """行集合归一（顺序无关），用于金标准结果对照。"""
    return {tuple(str(v) for v in row) for row in rows}


class EvalService:
    def run(self, db: Session, scene: str, run_id: str | None = None) -> EvalReport:
        """运行一次评估：对 EvalCase 逐条执行 pipeline，计算场景指标。"""
        run_id = run_id or uuid.uuid4().hex[:12]
        # intent 场景特殊：跨全部三场景用例（路由器需覆盖所有意图类型）
        if scene == "intent":
            cases = db.query(EvalCase).filter(EvalCase.scene.in_(["knowledge", "nl2sql", "dictionary"])).all()
        else:
            cases = db.query(EvalCase).filter(EvalCase.scene == scene).all()
        logger.info("eval_run_started", run_id=run_id, scene=scene, cases=len(cases))

        detail: dict = {"cases": [], "config": _config_snapshot()}
        faithfulness = relevancy = sql_success_rate = None

        if scene == "knowledge":
            faithfulness, relevancy, detail = self._run_knowledge(cases)
        elif scene == "nl2sql":
            sql_success_rate, detail = self._run_nl2sql(cases)
        elif scene == "dictionary":
            detail = self._run_dictionary(cases)
        elif scene == "intent":
            detail = self._run_intent(cases)
        else:
            detail["error"] = f"unknown scene: {scene}"

        detail.setdefault("config", _config_snapshot())
        detail.setdefault("aggregate", {})
        detail["finished_at"] = datetime.utcnow().isoformat()

        report = EvalReport(
            run_id=run_id,
            scene=scene,
            faithfulness=faithfulness,
            relevancy=relevancy,
            sql_success_rate=sql_success_rate,
            case_count=len(cases),
            detail=detail,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        logger.info(
            "eval_run_finished",
            run_id=run_id,
            scene=scene,
            faithfulness=faithfulness,
            relevancy=relevancy,
            sql_success_rate=sql_success_rate,
        )
        return report

    # ------------------------------------------------------------------
    # knowledge：RAG + RAGAS
    # ------------------------------------------------------------------
    def _run_knowledge(self, cases: list[EvalCase]) -> tuple:
        pipeline = container.get_rag_pipeline()
        embedding = container.get_embedding()

        samples: list[dict] = []
        case_details: list[dict] = []
        for c in cases:
            entry: dict = {"question": c.question}
            try:
                vec = embedding.embed_query(c.question)
                ans = pipeline.answer(c.question, vec)
                entry.update(
                    {
                        "answer": ans.answer,
                        "citations": [ct.chunk_id for ct in ans.citations],
                        "latency_ms": ans.latency_ms,
                        "ok": True,
                    }
                )
                samples.append(
                    {
                        "question": c.question,
                        "answer": ans.answer,
                        "contexts": ans.retrieved_contexts,
                        "reference": c.golden_answer or "",
                    }
                )
            except Exception as exc:
                entry.update({"ok": False, "error": str(exc)[:200]})
            case_details.append(entry)

        ragas_result = container.get_ragas_evaluator().evaluate(samples)
        aggregate = {"ragas_available": ragas_result is not None}
        if ragas_result is None:
            aggregate["ragas_error"] = "ragas evaluate unavailable/failed (see logs)"
            faithfulness = relevancy = None
        else:
            faithfulness = ragas_result["faithfulness"]
            relevancy = ragas_result["answer_relevancy"]
            aggregate.update(ragas_result)

        ok = sum(1 for e in case_details if e.get("ok"))
        aggregate["pipeline_ok"] = ok
        aggregate["pipeline_failed"] = len(case_details) - ok
        detail = {"cases": case_details, "aggregate": aggregate}
        return faithfulness, relevancy, detail

    # ------------------------------------------------------------------
    # nl2sql：执行成功率 + 金标准对照
    # ------------------------------------------------------------------
    def _run_nl2sql(self, cases: list[EvalCase]) -> tuple:
        pipeline = container.get_nl2sql_pipeline()
        executor = container.get_mcp_executor()

        case_details: list[dict] = []
        success = 0
        matched = 0
        for c in cases:
            entry: dict = {"question": c.question}
            try:
                ans = pipeline.answer(c.question)
                entry.update(
                    {
                        "sql": ans.sql,
                        "rows": ans.affected_rows,
                        "attempts": ans.attempts,
                        "latency_ms": ans.latency_ms,
                        "ok": True,
                    }
                )
                if ans.sql and ans.rows:
                    success += 1
                    # 金标准对照：行集合与 golden_sql 完全一致（更严格）
                    if c.golden_sql:
                        try:
                            golden = executor.execute_sql(c.golden_sql)
                            entry["result_match"] = (
                                _rows_to_multiset(ans.rows) == _rows_to_multiset(golden.rows)
                            )
                            if entry["result_match"]:
                                matched += 1
                        except Exception as exc:
                            entry["golden_error"] = str(exc)[:200]
            except Exception as exc:
                entry.update({"ok": False, "error": str(exc)[:200]})
            case_details.append(entry)

        sql_success_rate = round(success / len(cases), 4) if cases else None
        aggregate = {
            "sql_success_rate": sql_success_rate,
            "golden_result_match": matched,
            "golden_match_rate": round(matched / len(cases), 4) if cases else None,
        }
        return sql_success_rate, {"cases": case_details, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # dictionary：检索质量指标（hit_rate@5 / recall@5 / MRR）
    # ------------------------------------------------------------------
    def _run_dictionary(self, cases: list[EvalCase]) -> dict:
        pipeline = container.get_dictionary_pipeline()
        top_k = 5

        case_details: list[dict] = []
        hits_n = 0
        recalled = 0
        expected_total = 0
        rr_sum = 0.0
        for c in cases:
            expected: list[str] = c.expected_chunks or []
            entry: dict = {"question": c.question, "expected": expected}
            try:
                result = pipeline.search(c.question, top_k=top_k)
                got = [f"{h.table_name}.{h.field_name}" for h in result.hits]
                entry.update({"hits": got, "latency_ms": result.latency_ms, "ok": True})

                expected_set = set(expected)
                hit_ranks = [i + 1 for i, g in enumerate(got) if g in expected_set]
                if hit_ranks:
                    hits_n += 1
                    rr_sum += 1.0 / hit_ranks[0]
                recalled += len({g for g in got if g in expected_set})
                expected_total += len(expected_set)
            except Exception as exc:
                entry.update({"ok": False, "error": str(exc)[:200]})
            case_details.append(entry)

        n = len(cases)
        aggregate = {
            "hit_rate_at_5": round(hits_n / n, 4) if n else None,
            "recall_at_5": round(recalled / expected_total, 4) if expected_total else None,
            "mrr": round(rr_sum / n, 4) if n else None,
        }
        return {"cases": case_details, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # intent：意图路由准确率（mode=auto 两级混合路由）
    # ------------------------------------------------------------------
    def _run_intent(self, cases: list[EvalCase]) -> dict:
        router = container.get_intent_router()

        case_details: list[dict] = []
        confusion: dict[str, dict[str, int]] = {}
        correct = 0
        for c in cases:
            expected = c.scene  # EvalCase 的场景标注即意图标签
            entry: dict = {"question": c.question, "expected": expected}
            try:
                got = router.classify(c.question)
                entry["got"] = got
                entry["ok"] = got == expected
                correct += got == expected
                confusion.setdefault(expected, {}).setdefault(got, 0)
                confusion[expected][got] += 1
            except Exception as exc:
                entry.update({"ok": False, "error": str(exc)[:200]})
            case_details.append(entry)

        n = len(cases)
        # 分场景准确率（对角线 / 行合计）
        per_scene = {
            scene: round(row.get(scene, 0) / row_total, 4)
            for scene, row in confusion.items()
            if (row_total := sum(row.values()))
        }
        aggregate = {
            "accuracy": round(correct / n, 4) if n else None,
            "per_scene_accuracy": per_scene,
            "confusion": confusion,  # expected -> {got -> count}
        }
        return {"cases": case_details, "aggregate": aggregate}

    def list_reports(self, db: Session, scene: str | None = None) -> list[EvalReport]:
        query = db.query(EvalReport).order_by(EvalReport.id.desc())
        if scene:
            query = query.filter(EvalReport.scene == scene)
        return query.limit(20).all()
