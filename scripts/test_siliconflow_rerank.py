"""硅基流动 rerank 端点真实联调验证。

用项目自带的 ApiReranker（cohere 格式）调用 https://api.siliconflow.cn/v1/rerank，
API key 从 .env 中的 FINRAG_RERANK_API_KEY 读取。

验证点：
1. HTTP 200、无异常
2. 返回分数列表长度 == 输入文档数（等长契约）
3. "Apple" 与 "apple" 相关性最高（语义验证）
4. 原始响应 JSON 打印（人工核对 return_documents 效果）
"""

import httpx

from finrag.config import get_settings


def main() -> None:
    s = get_settings()

    print("=" * 60)
    print("硅基流动 Rerank 端点联调")
    print("=" * 60)
    print(f"base_url : {s.rerank_api_base_url}")
    print(f"model    : {s.rerank_api_model}")
    print(f"format   : {s.rerank_api_format}")
    print(f"api_key  : {s.rerank_api_key[:8]}...{s.rerank_api_key[-4:]}")
    print()

    query = "Apple"
    documents = ["apple", "banana", "fruit", "vegetable"]

    # ---- 1. 直接 httpx 调用（带 return_documents=true，验证原始响应） ----
    print("[1] 原始 HTTP 调用（return_documents=true）")
    resp = httpx.post(
        f"{s.rerank_api_base_url}/rerank",
        json={
            "model": s.rerank_api_model,
            "query": query,
            "documents": documents,
            "return_documents": True,
            "top_n": len(documents),
        },
        headers={"Authorization": f"Bearer {s.rerank_api_key}"},
        timeout=30.0,
    )
    print(f"    status: {resp.status_code}")
    resp.raise_for_status()
    raw = resp.json()

    import json

    print("    raw response:")
    print(json.dumps(raw, ensure_ascii=False, indent=2))
    print()

    # ---- 2. 用项目 ApiReranker（不传 return_documents，验证等长分数契约） ----
    from finrag.core.reranker import ApiReranker

    print("[2] 项目 ApiReranker 调用（验证等长分数契约）")
    reranker = ApiReranker(
        base_url=s.rerank_api_base_url,
        api_key=s.rerank_api_key,
        model=s.rerank_api_model,
        fmt="cohere",
    )
    scores = reranker.rerank(query, documents)

    print(f"    documents: {documents}")
    print(f"    scores   : {scores}")
    assert len(scores) == len(documents), f"分数长度 {len(scores)} != 文档数 {len(documents)}"
    print(f"    [PASS] 等长契约: {len(scores)} == {len(documents)}")
    print()

    # ---- 3. 语义验证：Apple 应与 apple 最相关 ----
    print("[3] 语义验证")
    best_idx = scores.index(max(scores))
    print(f"    最高分文档: documents[{best_idx}] = '{documents[best_idx]}' (score={max(scores):.4f})")
    assert best_idx == 0, f"预期 Apple→apple 最相关，实际最高是 documents[{best_idx}]"
    print("    [PASS] Apple → apple 相关性最高")
    print()

    # ---- 4. 顺序验证（按分数降序应该是 apple > fruit > banana > vegetable） ----
    print("[4] 排序验证")
    ranked = sorted(enumerate(documents), key=lambda x: scores[x[0]], reverse=True)
    for rank, (idx, doc) in enumerate(ranked, 1):
        print(f"    #{rank}: '{doc}' (score={scores[idx]:.4f})")
    print()

    print("=" * 60)
    print("全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
