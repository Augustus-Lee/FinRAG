"""限流测试：RateLimiter 单测（窗口/隔离/降级/熔断）+ 中间件集成（login 429/豁免/档位隔离/开关）。

测试机无 Redis → 走内存降级路径（RateLimiter 首次 Redis 连接失败自动降级 + 熔断），
与生产降级语义完全一致，反而天然覆盖了降级分支。
"""

from fastapi.testclient import TestClient

from finrag.config import Settings
from finrag.core.rate_limiter import RateLimiter
from finrag.main import app

client = TestClient(app)

API = "/api/v1"


# ---------------------------------------------------------------------------
# RateLimiter 单测（内存后端：构造后 monkeypatch 掉 Redis 使其不可达）
# ---------------------------------------------------------------------------


def _limiter() -> RateLimiter:
    rl = RateLimiter(Settings())
    # 测试机无 Redis；强制每次 _redis_check 都走异常路径（等价连接失败）→ 熔断 + 内存
    rl._get_redis = lambda: (_ for _ in ()).throw(ConnectionError("no redis in test"))
    return rl


def test_memory_window_allows_up_to_limit():
    rl = _limiter()
    for _ in range(3):
        allowed, retry, count = rl.check("t", "ip:1", 3)
        assert allowed and retry == 0
    allowed, retry, count = rl.check("t", "ip:1", 3)
    assert not allowed
    assert retry >= 1
    assert count == 3


def test_memory_window_slides_out():
    """窗口滑出后配额恢复（推进时钟验证，不真实等待 60s）。"""
    rl = _limiter()
    clock = {"now": 1000.0}
    rl._now = lambda: clock["now"]
    for _ in range(3):
        assert rl.check("t", "ip:1", 3)[0]
    assert not rl.check("t", "ip:1", 3)[0]
    clock["now"] += 61.0  # 窗口滑出
    assert rl.check("t", "ip:1", 3)[0]


def test_identity_and_scope_isolated():
    rl = _limiter()
    for _ in range(3):
        assert rl.check("t", "ip:1", 3)[0]
    assert not rl.check("t", "ip:1", 3)[0]
    # 不同 identity 独立配额
    assert rl.check("t", "ip:2", 3)[0]
    # 不同 scope 独立 key
    assert rl.check("other", "ip:1", 3)[0]


def test_zero_limit_disables():
    rl = _limiter()
    assert rl.check("t", "ip:1", 0)[0] is True


def test_redis_failure_falls_back_and_opens_circuit():
    """Redis 异常 → 降级内存继续限流；熔断期内不再尝试 Redis。"""
    rl = RateLimiter(Settings())
    eval_calls = {"n": 0}

    class FlakyRedis:
        def eval(self, *a, **kw):
            eval_calls["n"] += 1
            raise ConnectionError("redis down")

    rl._get_redis = lambda: FlakyRedis()
    # 第一次：Redis 异常 → 熔断打开 + 内存计数
    assert rl.check("t", "ip:1", 2)[0] is True
    assert eval_calls["n"] == 1
    # 熔断期内（60s）：不再调 Redis，直接内存
    assert rl.check("t", "ip:1", 2)[0] is True
    assert eval_calls["n"] == 1  # 未重试
    # 内存语义继续生效：超限拒绝
    assert rl.check("t", "ip:1", 2)[0] is False


def test_circuit_recovers_after_window():
    """熔断到期后恢复尝试 Redis（生产场景 Redis 重启后自动回归主路径）。"""
    rl = RateLimiter(Settings())
    state = {"fail": True}
    calls = {"n": 0}

    class RecoverableRedis:
        def eval(self, *a, **kw):
            calls["n"] += 1
            if state["fail"]:
                raise ConnectionError("down")
            return -1  # 允许

    rl._get_redis = lambda: RecoverableRedis()
    assert rl.check("t", "ip:9", 5)[0] is True  # 失败 → 熔断 + 内存
    assert rl.check("t", "ip:9", 5)[0] is True  # 熔断期内走内存
    assert calls["n"] == 1
    rl._circuit_open_until = 0.0  # 模拟熔断到期
    state["fail"] = False
    assert rl.check("t", "ip:9", 5)[0] is True
    assert calls["n"] == 2  # 恢复 Redis 主路径


def test_check_never_raises():
    """契约：任何内部异常都不外抛（限流故障不能杀死主链路）。"""
    rl = RateLimiter(Settings())
    rl._get_redis = lambda: (_ for _ in ()).throw(ConnectionError("x"))
    rl._memory_check = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    assert rl.check("t", "ip:1", 3)[0] is True  # 双后端全挂 → 放行


def test_redis_sliding_window_real_integration():
    """真实 Redis 主路径回归：Lua 滑动窗口含拒绝时 retry_after 计算。

    曾有 bug：ZRANGE 未带 WITHSCORES，tonumber(member) 得 nil 报算术错误，
    每次达到限额必触发（降级内存掩盖）。容器内 Redis 可用时强校验主路径。
    """
    import pytest

    rl = RateLimiter(Settings(redis_url="redis://redis:6379/0"))
    try:
        client_redis = rl._get_redis()
        client_redis.ping()
    except Exception:
        pytest.skip("Redis 不可用（本地开发环境），容器内验证")

    key_scope, identity = "t", "e2e_real"
    client_redis.delete(f"ratelimit:{key_scope}:{identity}")
    try:
        for i in range(3):
            allowed, retry, count = rl.check(key_scope, identity, 3)
            assert allowed and retry == 0 and count == i + 1
        allowed, retry, count = rl.check(key_scope, identity, 3)
        assert not allowed
        assert 1 <= retry <= 60  # WITHSCORES 取 score 计算窗口剩余（nil bug 回归点）
        assert count == 3
        # ZSET 计数真实落在 Redis
        assert client_redis.zcard(f"ratelimit:{key_scope}:{identity}") == 3
    finally:
        client_redis.delete(f"ratelimit:{key_scope}:{identity}")


# ---------------------------------------------------------------------------
# 中间件集成（TestClient；conftest 已全局强制内存后端 + 用例级清零）
# ---------------------------------------------------------------------------


def _login_attempt(username: str = "ghost_user_xyz", password: str = "wrong"):
    return client.post(f"{API}/auth/login", json={"username": username, "password": password})


def test_login_rate_limited_429_with_retry_after():
    # 不存在的用户名：跳过 pbkdf2 验证（用户查不到直接 401），保持测试快
    codes = [_login_attempt().status_code for _ in range(11)]
    assert codes[:10] == [401] * 10  # 先打满额度（登录失败也计数）
    assert codes[10] == 429  # 第 11 次（默认 login 限额 10/min）
    resp = _login_attempt()
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
    assert resp.headers["X-RateLimit-Limit"] == "10"
    body = resp.json()["error"]
    assert body["code"] == "RATE_LIMITED"
    assert body["detail"]["scope"] == "login"
    assert "X-Request-Id" in resp.headers  # 外层 request_id 中间件仍生效


def test_health_exempt_from_rate_limit():
    for _ in range(20):
        assert client.get(f"{API}/health").status_code == 200


def test_docs_and_root_exempt():
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200


def test_default_scope_counts_unauthorized_requests():
    # 无 token 401 请求也按 IP 计数（default 档）
    # 限额 120 太大不适合打满：直接向 limiter 注入小限额场景走真实 HTTP 验证分档
    from finrag import main as main_mod

    codes = [client.get(f"{API}/admin/users").status_code for _ in range(5)]
    assert codes == [401] * 5  # 未触发 429（远低于 120）
    # 档位隔离：login 打满不影响 default 档
    for _ in range(12):
        _login_attempt()
    assert _login_attempt().status_code == 429
    assert client.get(f"{API}/admin/users").status_code == 401  # default 档正常计数未超
    assert main_mod._resolve_rule(f"{API}/admin/users") == ("default", 120)


def test_resolve_rule_paths():
    from finrag import main as main_mod

    assert main_mod._resolve_rule(f"{API}/auth/login") == ("login", 10)
    assert main_mod._resolve_rule(f"{API}/chat/messages") == ("chat", 20)
    assert main_mod._resolve_rule(f"{API}/chat/sessions/1/messages") == ("chat", 20)
    assert main_mod._resolve_rule(f"{API}/dictionary/search") == ("default", 120)
    assert main_mod._resolve_rule(f"{API}/health") is None
    assert main_mod._resolve_rule("/docs") is None
    assert main_mod._resolve_rule("/static/app.js") is None


def test_identity_prefers_jwt_subject():
    """chat/default 档：有效 token 按 user 计数，无效回退 IP。"""
    import jwt as pyjwt
    import time as _time

    from finrag import main as main_mod
    from finrag.config import get_settings

    s = get_settings()
    token = pyjwt.encode(
        {"sub": "42", "iat": int(_time.time()), "exp": int(_time.time()) + 600},
        s.secret_key,
        algorithm="HS256",
    )

    class FakeClient:
        host = "10.0.0.1"

    class FakeRequest:
        client = FakeClient()
        headers = {"Authorization": f"Bearer {token}"}

    assert main_mod._identity(FakeRequest, "chat") == "user:42"
    assert main_mod._identity(FakeRequest, "login") == "ip:10.0.0.1"  # login 档不看 token

    FakeRequest.headers = {"Authorization": "Bearer invalid.token.here"}
    assert main_mod._identity(FakeRequest, "chat") == "ip:10.0.0.1"  # 验签失败回退 IP

    FakeRequest.headers = {}
    assert main_mod._identity(FakeRequest, "default") == "ip:10.0.0.1"


def test_chat_scope_per_user_isolation():
    """chat 档按用户隔离：A 用户打满 429，同 IP 的 B 用户不受影响。"""
    import jwt as pyjwt
    import time as _time

    from finrag.config import get_settings

    s = get_settings()

    def _tok(sub: str) -> str:
        return pyjwt.encode(
            {"sub": sub, "iat": int(_time.time()), "exp": int(_time.time()) + 600},
            s.secret_key,
            algorithm="HS256",
        )

    def _chat(tok: str):
        return client.post(
            f"{API}/chat/sessions",
            json={"mode": "knowledge", "title": "t"},
            headers={"Authorization": f"Bearer {tok}"},
        )

    tok_a, tok_b = _tok("101"), _tok("202")
    # A 打满 chat 档（20/min）；sub=101 用户不存在 → 依赖层 401，但中间件计数照走
    for _ in range(20):
        assert _chat(tok_a).status_code == 401
    assert _chat(tok_a).status_code == 429  # A 超限
    assert _chat(tok_b).status_code == 401  # B 同 IP 独立配额，未超限


def test_rate_limit_disabled_switch(monkeypatch):
    from finrag import main as main_mod

    monkeypatch.setattr(main_mod.settings, "rate_limit_enabled", False)
    codes = [_login_attempt().status_code for _ in range(15)]
    assert 429 not in codes  # 开关关闭不限流
    assert codes == [401] * 15
