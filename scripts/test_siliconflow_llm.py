"""硅基流动 LLM Chat 端点真实联调验证。

用项目 .env 中的 FINRAG_LLM_CLOUD_API_KEY 调用 https://api.siliconflow.cn/v1/chat/completions。

验证场景：
1. 原始 HTTP 调用（验证 API key + 模型名 + 端点连通性）
2. 不同超时配置对比（30s vs 120s，定位 timeout 根因）
3. 项目 LLMGateway 集成验证（trust_env=False + 持久化 Client）
4. NL2SQL 模拟调用（发一个真实 SQL 生成请求，测延迟）
"""

import time

import httpx

from finrag.config import get_settings


def main() -> None:
    s = get_settings()

    print("=" * 60)
    print("硅基流动 LLM Chat 端点联调")
    print("=" * 60)
    print(f"base_url : {s.llm_cloud_base_url}")
    print(f"api_key  : {s.llm_cloud_api_key[:8]}...{s.llm_cloud_api_key[-4:]}")
    print(f"model    : {s.llm_cloud_model}")
    print(f"timeout  : {s.llm_timeout}s (config) / {s.llm_max_retries} retries")
    print()

    # ============================================================
    # [1] 原始 HTTP 调用（最简验证：API key + 模型名 + 端点）
    # ============================================================
    print("[1] 原始 HTTP 调用（httpx.post, trust_env=False）")
    url = f"{s.llm_cloud_base_url}/chat/completions"
    payload = {
        "model": s.llm_cloud_model,
        "messages": [{"role": "user", "content": "你好，请回复一句话"}],
        "temperature": 0.1,
        "max_tokens": 50,
    }
    headers = {"Authorization": f"Bearer {s.llm_cloud_api_key}"}

    try:
        start = time.perf_counter()
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0, trust_env=False)
        elapsed = time.perf_counter() - start
        print(f"    status : {resp.status_code}")
        print(f"    latency: {elapsed:.2f}s")
        if resp.status_code != 200:
            print(f"    error  : {resp.text[:500]}")
            print("    [FAIL] API 调用失败，检查模型名/API key")
            return
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"    reply  : {content[:200]}")
        print(f"    model  : {data.get('model', 'N/A')}")
        usage = data.get("usage", {})
        print(f"    tokens : prompt={usage.get('prompt_tokens', '?')}, completion={usage.get('completion_tokens', '?')}")
        print("    [PASS] 原始 HTTP 调用成功")
    except httpx.ReadTimeout:
        print("    [FAIL] ReadTimeout — 30s 内未返回，模型可能不存在或服务端异常")
        return
    except Exception as exc:
        print(f"    [FAIL] 异常: {exc}")
        return
    print()

    # ============================================================
    # [2] trust_env 对比（True vs False，验证代理影响）
    # ============================================================
    print("[2] trust_env 对比（验证系统代理影响）")
    for te in [True, False]:
        label = "trust_env=True (走系统代理)" if te else "trust_env=False (直连)"
        try:
            start = time.perf_counter()
            r = httpx.post(
                url,
                json={**payload, "max_tokens": 10},
                headers=headers,
                timeout=30.0,
                trust_env=te,
            )
            elapsed = time.perf_counter() - start
            print(f"    {label}: status={r.status_code}, latency={elapsed:.2f}s")
        except Exception as exc:
            print(f"    {label}: FAIL — {exc}")
    print()

    # ============================================================
    # [3] 项目 LLMGateway 集成验证
    # ============================================================
    print("[3] 项目 LLMGateway 集成验证")
    from finrag.core.llm_gateway import LLMGateway

    gw = LLMGateway(s)
    try:
        start = time.perf_counter()
        reply = gw.chat(
            [{"role": "user", "content": "1+1等于几？只回答数字"}],
            temperature=0.0,
            max_tokens=10,
        )
        elapsed = time.perf_counter() - start
        print(f"    reply  : {reply}")
        print(f"    latency: {elapsed:.2f}s")
        print("    [PASS] LLMGateway 调用成功")
    except Exception as exc:
        print(f"    [FAIL] {exc}")
        return
    print()

    # ============================================================
    # [4] NL2SQL 模拟调用（真实 SQL 生成，测延迟）
    # ============================================================
    print("[4] NL2SQL 模拟调用（生成 SQL，测真实延迟）")
    system_prompt = (
        "你是金融数据查询 SQL 专家。基于给定的表结构生成 MySQL SELECT 语句：\n"
        "- 仅生成 SELECT\n"
        "- 必须带 LIMIT（不超过 100 行）\n"
        "- 仅使用给定的表与字段\n"
        "直接输出 SQL 文本，不要解释。"
    )
    schema = (
        "【表结构】\n"
        "表: product_sales\n"
        "  - product_code VARCHAR(50) 产品代码\n"
        "  - product_name VARCHAR(100) 产品名称\n"
        "  - risk_level VARCHAR(20) 风险等级\n"
        "  - sales_amount DECIMAL(15,2) 销售金额\n"
        "  - trade_date DATE 交易日期\n"
    )
    question = "【问题】\n高风险产品的销售金额是多少"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{schema}\n\n{question}"},
    ]

    try:
        start = time.perf_counter()
        sql = gw.chat(messages, temperature=0.0, max_tokens=200)
        elapsed = time.perf_counter() - start
        print(f"    SQL    : {sql.strip()[:200]}")
        print(f"    latency: {elapsed:.2f}s")
        if elapsed > 30:
            print("    [WARN] 延迟 > 30s，NL2SQL 多轮调用可能超时")
        else:
            print("    [PASS] SQL 生成成功，延迟可接受")
    except Exception as exc:
        print(f"    [FAIL] {exc}")
    print()

    # ============================================================
    # [5] 多轮调用延迟模拟（NL2SQL 管线最坏 4 次 LLM 调用）
    # ============================================================
    print("[5] 多轮调用延迟模拟（NL2SQL 管线最坏 4 次）")
    total = 0.0
    for i in range(4):
        try:
            start = time.perf_counter()
            gw.chat(
                [{"role": "user", "content": f"第 {i+1} 次调用，回复 OK"}],
                temperature=0.0,
                max_tokens=5,
            )
            elapsed = time.perf_counter() - start
            total += elapsed
            print(f"    调用 {i+1}/4: {elapsed:.2f}s")
        except Exception as exc:
            print(f"    调用 {i+1}/4: FAIL — {exc}")
            break
    print(f"    总计: {total:.2f}s (NL2SQL 管线最坏情况)")
    if total > s.llm_timeout:
        print(f"    [WARN] 总延迟 > llm_timeout({s.llm_timeout}s)，需要增大超时")
    else:
        print(f"    [PASS] 总延迟在 llm_timeout({s.llm_timeout}s) 内")
    print()

    print("=" * 60)
    print("全部验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
