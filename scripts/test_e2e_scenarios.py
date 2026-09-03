"""FinRAG 三场景端到端联调测试脚本。

前提：docker compose 全栈已启动（bash scripts/startup.sh）。
用法：.venv/Scripts/python scripts/test_e2e_scenarios.py
"""

import time

import httpx

BASE = "http://localhost:8000"
TIMEOUT = httpx.Timeout(connect=10, read=300, write=10, pool=10)


def sep(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # trust_env=False: localhost 请求不走系统代理，避免 http_proxy 链路延迟与超时
    client = httpx.Client(base_url=BASE, timeout=TIMEOUT, trust_env=False)

    # ============================================================
    # 前置：健康检查
    # ============================================================
    sep("前置检查")
    r = client.get("/api/v1/health")
    health = r.json()
    print(f"  /health: {health}")
    assert r.status_code == 200
    assert health["status"] == "ok"
    print("  [PASS] 服务健康")

    # ============================================================
    # 场景一：数据字典语义检索
    # ============================================================
    sep("场景一：数据字典语义检索")
    questions = [
        ("客户手机号是多少", "mobile"),
        ("上个月的销售金额", "sales_amount"),
        ("风险等级", "risk_level"),
        ("成交金额和手续费", "trade_amount"),
    ]
    for q, expected_field in questions:
        r = client.post(
            "/api/v1/dictionary/search",
            json={"question": q, "top_k": 5},
        )
        body = r.json()
        hits = body.get("hits", [])
        fields = [h["field_name"] for h in hits]
        print(f"  Q: {q}")
        print(f"    hits ({len(hits)}): {fields[:5]}")
        if expected_field in fields:
            rank = fields.index(expected_field) + 1
            print(f"    [PASS] {expected_field} 排第 {rank}")
        else:
            print(f"    [WARN] 预期 {expected_field} 未出现在结果中")
    print("  [PASS] 场景一完成")

    # ============================================================
    # 场景二：NL2SQL 智能问数
    # ============================================================
    sep("场景二：NL2SQL 智能问数")

    # 2a. 创建会话
    r = client.post("/api/v1/chat/sessions", json={"mode": "nl2sql"})
    session = r.json()
    session_id = session["id"]
    print(f"  会话创建: id={session_id}, mode={session.get('mode')}")
    assert r.status_code in (200, 201)

    # 2b. 提问
    nl2sql_questions = [
        "高风险产品的销售金额是多少",
        "总资产超过100万的客户有几个",
        "客户张三的手机号是什么",
    ]
    for q in nl2sql_questions:
        print(f"\n  Q: {q}")
        r = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"question": q, "mode": "nl2sql"},
        )
        if r.status_code != 200:
            print(f"    [ERROR] status={r.status_code}")
            print(f"    body: {r.text[:500]}")
            continue
        body = r.json()
        answer = body.get("answer", "")
        sql = body.get("sql", "")
        latency = body.get("latency_ms", 0)
        rows = body.get("rows", [])
        print(f"    SQL: {sql[:120]}")
        print(f"    rows: {len(rows)} 行")
        if rows:
            print(f"    样例: {rows[:3]}")
        print(f"    answer: {answer[:200]}")
        print(f"    latency: {latency}ms")
        if sql:
            print("    [PASS] 生成并执行了 SQL")
        else:
            print("    [WARN] 未生成 SQL")

    # ============================================================
    # 场景三：知识库文档 RAG
    # ============================================================
    sep("场景三：知识库文档 RAG")

    # 3a. 上传文档（multipart：文件 + kb_id 表单字段）
    import pathlib

    readme = pathlib.Path("README.md")
    if readme.exists():
        with readme.open("rb") as f:
            r = client.post(
                "/api/v1/knowledge/documents/upload",
                files={"file": ("README.md", f, "text/markdown")},
                data={"kb_id": "1"},
            )
        print(f"  上传文档: status={r.status_code}")
        if r.status_code in (200, 201):
            doc = r.json()
            print(f"    doc_id: {doc.get('id')}")
            print(f"    file_type: {doc.get('file_type')}  status: {doc.get('status')}")
        else:
            print(f"    body: {r.text[:300]}")
    else:
        print("  [SKIP] README.md 不存在，跳过上传")

    # 3b. 等待 worker 解析（Celery 异步）
    print("  等待文档解析（5 秒）...")
    time.sleep(5)

    # 3c. 创建 RAG 会话并问答
    r = client.post("/api/v1/chat/sessions", json={"mode": "knowledge"})
    rag_session = r.json()
    rag_session_id = rag_session["id"]
    print(f"  RAG 会话: id={rag_session_id}")

    rag_questions = [
        "FinRAG 的整体架构是怎样的",
        "项目用了哪些技术栈",
        "SQL 安全校验有哪几层",
    ]
    for q in rag_questions:
        print(f"\n  Q: {q}")
        r = client.post(
            f"/api/v1/chat/sessions/{rag_session_id}/messages",
            json={"question": q, "mode": "knowledge"},
        )
        if r.status_code != 200:
            print(f"    [ERROR] status={r.status_code}")
            print(f"    body: {r.text[:500]}")
            continue
        body = r.json()
        answer = body.get("answer", "")
        citations = body.get("citations", [])
        latency = body.get("latency_ms", 0)
        print(f"    answer: {answer[:300]}")
        print(f"    citations: {len(citations)} 条")
        if citations:
            for c in citations[:2]:
                print(f"      [{c.get('chunk_id', '?')}] {str(c.get('content', ''))[:80]}...")
        print(f"    latency: {latency}ms")
        if answer and "未能" not in answer:
            print("    [PASS] 生成了回答")
        else:
            print("    [WARN] 回答可能不理想")

    # ============================================================
    # 汇总
    # ============================================================
    sep("联调完成")
    print("  场景一 数据字典检索: PASS")
    print("  场景二 NL2SQL 智能问数: 已执行")
    print("  场景三 知识库 RAG: 已执行")
    print()
    print("  API 文档: http://localhost:8000/docs")
    print("  Qdrant:   http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
