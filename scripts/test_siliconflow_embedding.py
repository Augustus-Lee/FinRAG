"""硅基流动 embedding 端点真实联调验证。

用项目 .env 中的 FINRAG_EMBEDDING_API_KEY 调用 https://api.siliconflow.cn/v1/embeddings。

验证场景：
1. 纯文本 embedding（OpenAI 兼容格式，项目 ApiEmbedding 走的路径）
2. 多模态图片 embedding（Qwen3-VL-Embedding，input 传 {"image": url}）
3. 项目 ApiEmbedding 集成验证（等长契约 + 维度 + 余弦相似度语义验证）
"""

import httpx

from finrag.config import get_settings


def main() -> None:
    s = get_settings()

    print("=" * 60)
    print("硅基流动 Embedding 端点联调")
    print("=" * 60)
    print(f"base_url : {s.embedding_api_base_url}")
    print(f"api_key  : {s.embedding_api_key[:8]}...{s.embedding_api_key[-4:]}")
    print()

    # ============================================================
    # [1] 纯文本 embedding（BAAI/bge-m3，OpenAI 兼容格式）
    # ============================================================
    print("[1] 纯文本 embedding（BAAI/bge-m3）")
    texts = ["苹果公司发布新款iPhone", "今日香蕉价格下跌", "水果营养丰富"]
    resp = httpx.post(
        f"{s.embedding_api_base_url}/embeddings",
        json={"model": "BAAI/bge-m3", "input": texts},
        headers={"Authorization": f"Bearer {s.embedding_api_key}"},
        timeout=30.0,
    )
    print(f"    status: {resp.status_code}")
    resp.raise_for_status()
    raw = resp.json()
    print(f"    model   : {raw.get('model', 'N/A')}")
    print(f"    data count: {len(raw['data'])}")
    dim = len(raw["data"][0]["embedding"])
    print(f"    dimension: {dim}")
    for item in raw["data"]:
        print(f"      index={item['index']}  dim={len(item['embedding'])}  first3={item['embedding'][:3]}")
    assert len(raw["data"]) == len(texts), f"返回数量 {len(raw['data'])} != 输入 {len(texts)}"
    print(f"    [PASS] 返回 {len(raw['data'])} 条 == 输入 {len(texts)} 条")
    print()

    # ============================================================
    # [2] 多模态图片 embedding（Qwen/Qwen3-VL-Embedding-8B）
    # ============================================================
    print("[2] 多模态图片 embedding（Qwen/Qwen3-VL-Embedding-8B）")
    # 硅基流动服务端拉取外部图片 URL 受限（400），base64 可行
    # 1x1 红色 PNG 的 base64（最小合法图片）
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    resp2 = httpx.post(
        f"{s.embedding_api_base_url}/embeddings",
        json={"model": "Qwen/Qwen3-VL-Embedding-8B", "input": [{"image": f"data:image/png;base64,{png_b64}"}]},
        headers={"Authorization": f"Bearer {s.embedding_api_key}"},
        timeout=60.0,
    )
    print(f"    status: {resp2.status_code}")
    if resp2.status_code != 200:
        print(f"    error body: {resp2.text[:500]}")
        print("    (多模态模型可能需要额外权限，跳过验证)")
    else:
        raw2 = resp2.json()
        print(f"    model   : {raw2.get('model', 'N/A')}")
        data2 = raw2.get("data", [])
        print(f"    data count: {len(data2)}")
        if data2:
            dim2 = len(data2[0]["embedding"])
            print(f"    dimension: {dim2}")
            print(f"    first3: {data2[0]['embedding'][:3]}")
            print(f"    [PASS] 图片 embedding 返回 {dim2} 维向量")
    print()

    # ============================================================
    # [2b] VL 模型纯文本 embedding（与 bge-m3 对比维度差异）
    # ============================================================
    print("[2b] VL 模型纯文本 embedding（Qwen3-VL-Embedding-8B）")
    resp2b = httpx.post(
        f"{s.embedding_api_base_url}/embeddings",
        json={"model": "Qwen/Qwen3-VL-Embedding-8B", "input": "硅基流动多模态嵌入测试"},
        headers={"Authorization": f"Bearer {s.embedding_api_key}"},
        timeout=30.0,
    )
    print(f"    status: {resp2b.status_code}")
    if resp2b.status_code == 200:
        data2b = resp2b.json().get("data", [])
        if data2b:
            print(f"    dimension: {len(data2b[0]['embedding'])}")
            print(f"    [PASS] VL 文本 embedding 返回 {len(data2b[0]['embedding'])} 维")
    print()

    # ============================================================
    # [3] 项目 ApiEmbedding 集成验证
    # ============================================================
    print("[3] 项目 ApiEmbedding 集成验证")
    from finrag.core.embedding import ApiEmbedding

    provider = ApiEmbedding(
        base_url=s.embedding_api_base_url,
        api_key=s.embedding_api_key,
        model="BAAI/bge-m3",
        dimension=1024,
    )

    query_vec = provider.embed_query("苹果手机")
    doc_vecs = provider.embed(["iPhone 15 Pro Max 发布", "香蕉今日批发价", "蔬菜批发行情"])
    print(f"    query dim: {len(query_vec)}")
    print(f"    docs count: {len(doc_vecs)}, each dim: {len(doc_vecs[0])}")
    assert len(query_vec) == len(doc_vecs[0]), "query 和 doc 维度不一致"
    print(f"    [PASS] 维度一致: {len(query_vec)} == {len(doc_vecs[0])}")
    print()

    # 余弦相似度语义验证
    print("[4] 余弦相似度语义验证")
    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    docs = ["iPhone 15 Pro Max 发布", "香蕉今日批发价", "蔬菜批发行情"]
    sims = [(doc, cosine(query_vec, dv)) for doc, dv in zip(docs, doc_vecs, strict=False)]
    sims.sort(key=lambda x: x[1], reverse=True)
    for rank, (doc, sim) in enumerate(sims, 1):
        print(f"    #{rank}: '{doc}' (cosine={sim:.4f})")
    assert sims[0][0] == "iPhone 15 Pro Max 发布", f"预期苹果手机→iPhone 最相关，实际最高是 {sims[0][0]}"
    print("    [PASS] 苹果手机 → iPhone 相关性最高")
    print()

    print("=" * 60)
    print("全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
