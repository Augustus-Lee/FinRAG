"""评估服务测试（M3）：三场景指标计算、ragas 降级、单 case 容错。"""

from unittest.mock import MagicMock

from finrag.db.session import SessionLocal
from finrag.models import EvalCase, EvalReport
from finrag.pipelines.dictionary import DictSearchResult, FieldHit
from finrag.pipelines.rag import RAGAnswer, Citation
from finrag.services.eval_service import EvalService
from finrag.services.eval_service import _rows_to_multiset


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _add_case(db, scene, question, **kw):
    c = EvalCase(scene=scene, question=question, **kw)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _cleanup(db, questions: list[str]):
    db.query(EvalCase).filter(EvalCase.question.in_(questions)).delete(synchronize_session=False)
    db.query(EvalReport).filter(EvalReport.scene.in_(["knowledge", "nl2sql", "dictionary"])).delete(
        synchronize_session=False
    )
    db.commit()


def _rag_answer():
    return RAGAnswer(
        answer="三大场景：数据字典、智能问数、知识库。",
        citations=[Citation(chunk_id="doc1_c0", content="x", section_path="s", score=0.9)],
        latency_ms=12.3,
        retrieved_contexts=["FinRAG 面向数据字典、智能问数、知识库三大场景。"],
    )


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------

def test_knowledge_scene_with_ragas(monkeypatch):
    db = SessionLocal()
    qs = [f"知识测试问题{i}？" for i in range(2)]
    try:
        for q in qs:
            _add_case(db, "knowledge", q, golden_answer="标准答案")

        pipeline = MagicMock()
        pipeline.answer.return_value = _rag_answer()
        embedding = MagicMock()
        embedding.embed_query.return_value = [0.1, 0.2]
        ragas = MagicMock()
        ragas.evaluate.return_value = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        monkeypatch.setattr("finrag.container.get_rag_pipeline", lambda: pipeline)
        monkeypatch.setattr("finrag.container.get_embedding", lambda: embedding)
        monkeypatch.setattr("finrag.container.get_ragas_evaluator", lambda: ragas)

        report = EvalService().run(db, "knowledge", run_id="t-know-1")

        assert report.faithfulness == 0.9
        assert report.relevancy == 0.85
        assert report.case_count == 2
        agg = report.detail["aggregate"]
        assert agg["ragas_available"] is True
        assert agg["pipeline_ok"] == 2
        # 样本进入 ragas：question/contexts/reference 传递正确
        samples = ragas.evaluate.call_args.args[0]
        assert samples[0]["reference"] == "标准答案"
        assert samples[0]["contexts"] == _rag_answer().retrieved_contexts
        # 配置快照存在（A/B 对比依据）
        assert "rrf_vector_weight" in report.detail["config"]
    finally:
        _cleanup(db, qs)


def test_knowledge_scene_ragas_unavailable_degrades(monkeypatch):
    db = SessionLocal()
    qs = ["ragas挂了怎么办？"]
    try:
        _add_case(db, "knowledge", qs[0], golden_answer="x")
        pipeline = MagicMock()
        pipeline.answer.return_value = _rag_answer()
        monkeypatch.setattr("finrag.container.get_rag_pipeline", lambda: pipeline)
        monkeypatch.setattr("finrag.container.get_embedding", lambda: MagicMock())
        ragas = MagicMock()
        ragas.evaluate.return_value = None
        monkeypatch.setattr("finrag.container.get_ragas_evaluator", lambda: ragas)

        report = EvalService().run(db, "knowledge", run_id="t-know-2")

        # ragas 失败 → 指标 None + detail 记 ragas_error，报告照常落库
        assert report.faithfulness is None
        assert report.relevancy is None
        assert report.detail["aggregate"]["ragas_available"] is False
        assert "ragas_error" in report.detail["aggregate"]
        assert report.case_count == 1
    finally:
        _cleanup(db, qs)


# ---------------------------------------------------------------------------
# nl2sql
# ---------------------------------------------------------------------------

def _nl2sql_ans(rows):
    ans = MagicMock()
    ans.sql = "SELECT COUNT(*) FROM customer_account LIMIT 1;"
    ans.rows = rows
    ans.affected_rows = len(rows)
    ans.attempts = 1
    ans.latency_ms = 50.0
    return ans


def test_nl2sql_scene_success_and_golden_match(monkeypatch):
    db = SessionLocal()
    qs = ["问数测试1", "问数测试2"]
    try:
        for q in qs:
            _add_case(db, "nl2sql", q, golden_sql="SELECT COUNT(*) FROM customer_account LIMIT 1;")

        golden_exec = MagicMock()
        golden_exec.rows = [[3]]
        executor = MagicMock()
        executor.execute_sql.return_value = golden_exec
        pipeline = MagicMock()
        pipeline.answer.side_effect = lambda q: _nl2sql_ans([[3]])
        monkeypatch.setattr("finrag.container.get_nl2sql_pipeline", lambda: pipeline)
        monkeypatch.setattr("finrag.container.get_mcp_executor", lambda: executor)

        report = EvalService().run(db, "nl2sql", run_id="t-nl2sql-1")

        assert report.sql_success_rate == 1.0
        agg = report.detail["aggregate"]
        assert agg["golden_result_match"] == 2
        assert agg["golden_match_rate"] == 1.0
        assert all(c["result_match"] for c in report.detail["cases"])
    finally:
        _cleanup(db, qs)


def test_nl2sql_golden_mismatch_detected(monkeypatch):
    db = SessionLocal()
    qs = ["结果不一致的题"]
    try:
        _add_case(db, "nl2sql", qs[0], golden_sql="SELECT COUNT(*) FROM t LIMIT 1;")
        golden_exec = MagicMock()
        golden_exec.rows = [[3]]
        executor = MagicMock()
        executor.execute_sql.return_value = golden_exec
        pipeline = MagicMock()
        pipeline.answer.side_effect = lambda q: _nl2sql_ans([[5]])  # 生成结果 5 ≠ 金标准 3
        monkeypatch.setattr("finrag.container.get_nl2sql_pipeline", lambda: pipeline)
        monkeypatch.setattr("finrag.container.get_mcp_executor", lambda: executor)

        report = EvalService().run(db, "nl2sql", run_id="t-nl2sql-2")

        # 执行成功但金标准不匹配：success_rate=1.0，golden_match_rate=0.0
        assert report.sql_success_rate == 1.0
        assert report.detail["aggregate"]["golden_match_rate"] == 0.0
        assert report.detail["cases"][0]["result_match"] is False
    finally:
        _cleanup(db, qs)


# ---------------------------------------------------------------------------
# dictionary
# ---------------------------------------------------------------------------

def _dict_result(hit_specs):
    hits = [
        FieldHit(table_name=t, field_name=f, field_type="", comment="", calibre="", synonyms=[])
        for t, f in hit_specs
    ]
    return DictSearchResult(question="q", hits=hits, latency_ms=5.0)


def test_dictionary_scene_metrics(monkeypatch):
    db = SessionLocal()
    qs = ["字典题1", "字典题2", "字典题3"]
    try:
        _add_case(db, "dictionary", qs[0], expected_chunks=["customer_account.mobile"])
        _add_case(
            db, "dictionary", qs[1], expected_chunks=["transaction_record.trade_amount", "transaction_record.commission"]
        )
        _add_case(db, "dictionary", qs[2], expected_chunks=["product_sales.sales_amount"])

        results = {
            qs[0]: _dict_result([("customer_account", "mobile")]),  # rank1 命中
            qs[1]: _dict_result(
                [("customer_account", "x"), ("transaction_record", "trade_amount")]  # rank2 命中 1/2
            ),
            qs[2]: _dict_result([("customer_account", "x")]),  # 未命中
        }
        pipeline = MagicMock()
        pipeline.search.side_effect = lambda q, top_k: results[q]
        monkeypatch.setattr("finrag.container.get_dictionary_pipeline", lambda: pipeline)

        report = EvalService().run(db, "dictionary", run_id="t-dict-1")

        assert report.faithfulness is None and report.relevancy is None
        agg = report.detail["aggregate"]
        # hit_rate：2/3 题（题1、题2 命中）
        assert agg["hit_rate_at_5"] == round(2 / 3, 4)
        # recall：命中 2 个 expected / 总 expected 4 个（题1 1个+题2 2个+题3 1个）
        assert agg["recall_at_5"] == round(2 / 4, 4)
        # MRR：题1 rank1 → 1.0；题2 rank2 → 0.5；题3 → 0；(1.0+0.5+0)/3
        assert agg["mrr"] == round(1.5 / 3, 4)
    finally:
        _cleanup(db, qs)


# ---------------------------------------------------------------------------
# 容错
# ---------------------------------------------------------------------------

def test_single_case_failure_does_not_break_run(monkeypatch):
    db = SessionLocal()
    qs = ["会炸的题", "正常的题"]
    try:
        for q in qs:
            _add_case(db, "dictionary", q, expected_chunks=["customer_account.mobile"])

        pipeline = MagicMock()

        def flaky(q, top_k):
            if q == "会炸的题":
                raise RuntimeError("boom")
            return _dict_result([("customer_account", "mobile")])

        pipeline.search.side_effect = flaky
        monkeypatch.setattr("finrag.container.get_dictionary_pipeline", lambda: pipeline)

        report = EvalService().run(db, "dictionary", run_id="t-err-1")

        cases = report.detail["cases"]
        by_q = {c["question"]: c for c in cases}
        assert by_q["会炸的题"]["ok"] is False
        assert "boom" in by_q["会炸的题"]["error"]
        assert by_q["正常的题"]["ok"] is True
        assert report.case_count == 2  # 整轮未中断
        assert report.detail["aggregate"]["hit_rate_at_5"] == round(1 / 2, 4)
    finally:
        _cleanup(db, qs)


def test_rows_to_multiset_normalizes_types():
    a = _rows_to_multiset([[3], [1]])
    b = _rows_to_multiset([["1"], [3]])
    assert a == b  # str(3) == "3"，顺序无关
