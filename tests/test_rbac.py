"""RBAC 测试：密码哈希 / JWT 认证 / 端点保护矩阵 / chat 场景权限 / 会话归属 / 管理 API。

前置：conftest 已在内存库灌入种子（admin/admin123 + analyst + kb_operator 角色）。
"""

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from finrag.config import get_settings
from finrag.db.session import SessionLocal
from finrag.main import app
from finrag.services.auth_service import hash_password, verify_password

client = TestClient(app)

API = "/api/v1"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _login(username: str, password: str):
    resp = client.post(f"{API}/auth/login", json={"username": username, "password": password})
    return resp


def _token(username: str = "admin", password: str = "admin123") -> str:
    resp = _login(username, password)
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def analyst_token():
    """创建 analyst 用户并登录（module 级复用，测后清理）。"""
    admin = _token()
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "analyst01", "password": "analyst123", "role_ids": [], "is_active": True},
        headers=_h(admin),
    )
    assert resp.status_code == 201, resp.text
    # 绑 analyst 角色（先查角色 id）
    roles = client.get(f"{API}/admin/roles", headers=_h(admin)).json()
    analyst_role = next(r for r in roles if r["role_code"] == "analyst")
    uid = resp.json()["id"]
    patch = client.patch(
        f"{API}/admin/users/{uid}", json={"role_ids": [analyst_role["id"]]}, headers=_h(admin)
    )
    assert patch.status_code == 200
    yield _token("analyst01", "analyst123")
    # 清理（软删即可，不破坏其他用例）
    client.patch(f"{API}/admin/users/{uid}", json={"is_active": False}, headers=_h(admin))


# ---------------------------------------------------------------------------
# 密码哈希
# ---------------------------------------------------------------------------

def test_password_hash_roundtrip():
    stored = hash_password("s3cret!")
    assert stored.startswith("pbkdf2_sha256$")
    assert len(stored) <= 128  # 列宽约束
    assert "s3cret!" not in stored
    assert verify_password("s3cret!", stored)
    assert not verify_password("wrong", stored)


def test_password_hash_salt_unique():
    assert hash_password("same") != hash_password("same")  # 随机盐


def test_verify_password_garbage_stored():
    assert not verify_password("x", "not-a-valid-format")


# ---------------------------------------------------------------------------
# 登录
# ---------------------------------------------------------------------------

def test_login_ok_returns_jwt_and_perms():
    resp = _login("admin", "admin123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "admin"
    assert set(body["roles"]) == {"admin"}
    assert "system_manage" in body["perms"] and "nl2sql" in body["perms"]


def test_login_unknown_user_401():
    assert _login("ghost", "x").status_code == 401


def test_login_disabled_user_401():
    admin = _token()
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "disabled01", "password": "pass123456", "role_ids": [], "is_active": True},
        headers=_h(admin),
    )
    uid = resp.json()["id"]
    client.patch(f"{API}/admin/users/{uid}", json={"is_active": False}, headers=_h(admin))
    # 停用后登录被拒（与密码错误同提示，防枚举）
    assert _login("disabled01", "pass123456").status_code == 401


# ---------------------------------------------------------------------------
# token 校验
# ---------------------------------------------------------------------------

def test_protected_endpoint_no_token_401():
    assert client.post(f"{API}/dictionary/search", json={"question": "成交金额"}).status_code == 401


def test_tampered_token_401():
    resp = client.post(
        f"{API}/dictionary/search",
        json={"question": "成交金额"},
        headers=_h("eyJhbGciOiJIUzI1NiJ9.forged.signature"),
    )
    assert resp.status_code == 401


def test_expired_token_401():
    settings = get_settings()
    expired = pyjwt.encode(
        {"sub": "1", "iat": 0, "exp": 1}, settings.secret_key, algorithm="HS256"
    )
    resp = client.post(
        f"{API}/dictionary/search", json={"question": "x"}, headers=_h(expired)
    )
    assert resp.status_code == 401


def test_token_for_missing_user_401():
    settings = get_settings()
    import time

    token = pyjwt.encode(
        {"sub": "99999", "iat": int(time.time()), "exp": int(time.time()) + 600},
        settings.secret_key,
        algorithm="HS256",
    )
    assert client.get(f"{API}/admin/users", headers=_h(token)).status_code == 401


# ---------------------------------------------------------------------------
# 端点保护矩阵
# ---------------------------------------------------------------------------

def test_admin_can_access_dictionary():
    resp = client.post(
        f"{API}/dictionary/search", json={"question": "成交金额"}, headers=_h(_token())
    )
    assert resp.status_code == 200


def test_analyst_can_access_dictionary(analyst_token):
    resp = client.post(
        f"{API}/dictionary/search", json={"question": "成交金额"}, headers=_h(analyst_token)
    )
    assert resp.status_code == 200


def test_analyst_cannot_manage_kb(analyst_token):
    # kb_operator 专属：analyst 只有 nl2sql/dictionary
    resp = client.post(
        f"{API}/knowledge/categories",
        json={"name": "无权分类", "description": ""},
        headers=_h(analyst_token),
    )
    assert resp.status_code == 403


def test_analyst_cannot_run_eval(analyst_token):
    assert client.get(f"{API}/evaluate/reports", headers=_h(analyst_token)).status_code == 403


def test_analyst_cannot_access_admin(analyst_token):
    assert client.get(f"{API}/admin/users", headers=_h(analyst_token)).status_code == 403


# ---------------------------------------------------------------------------
# chat 场景权限（effective mode，覆盖 auto / 显式 / 继承）
# ---------------------------------------------------------------------------

def test_chat_explicit_mode_without_perm_403(analyst_token):
    resp = client.post(
        f"{API}/chat/messages",
        json={"question": "什么是净赎回", "mode": "knowledge"},
        headers=_h(analyst_token),
    )
    assert resp.status_code == 403
    assert "knowledge" in resp.json()["error"]["message"]


def test_chat_create_session_without_perm_403(analyst_token):
    resp = client.post(
        f"{API}/chat/sessions",
        json={"mode": "knowledge", "title": "t"},
        headers=_h(analyst_token),
    )
    assert resp.status_code == 403


def test_chat_auto_routes_to_forbidden_scene_403(analyst_token, monkeypatch):
    # 意图路由判定 knowledge（analyst 无此权限）→ 403 而非降级
    from finrag.services import chat_service as cs

    monkeypatch.setattr(cs.ChatService, "_resolve_intent", lambda self, q: "knowledge")
    resp = client.post(
        f"{API}/chat/messages",
        json={"question": "什么是净赎回", "mode": "auto"},
        headers=_h(analyst_token),
    )
    assert resp.status_code == 403


def test_chat_session_inherits_mode_with_perm_check(analyst_token, monkeypatch):
    # 会话 mode 继承路径：他人 knowledge 会话 → 归属校验先行 403
    admin = _token()
    created = client.post(
        f"{API}/chat/sessions", json={"mode": "knowledge", "title": "admin的会话"}, headers=_h(admin)
    ).json()
    resp = client.post(
        f"{API}/chat/sessions/{created['id']}/messages",
        json={"question": "风险提示是什么", "mode": "knowledge"},
        headers=_h(analyst_token),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 会话归属
# ---------------------------------------------------------------------------

def test_session_owner_isolation(analyst_token):
    admin = _token()
    created = client.post(
        f"{API}/chat/sessions", json={"mode": "knowledge", "title": "归属测试"}, headers=_h(admin)
    ).json()
    # 另一用户（analyst）不可读 admin 的会话消息
    resp = client.get(f"{API}/chat/sessions/{created['id']}/messages", headers=_h(analyst_token))
    assert resp.status_code == 403
    # 本人可读
    assert client.get(f"{API}/chat/sessions/{created['id']}/messages", headers=_h(admin)).status_code == 200


# ---------------------------------------------------------------------------
# 管理 API 全链路
# ---------------------------------------------------------------------------

def test_admin_crud_flow():
    admin = _token()
    headers = _h(admin)

    # 角色创建 + 权限绑定
    resp = client.post(
        f"{API}/admin/roles",
        json={
            "role_code": "readonly_qa",
            "role_name": "只读问答",
            "description": "仅知识库问答",
            "perm_codes": ["knowledge"],
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    role = resp.json()
    assert role["perm_codes"] == ["knowledge"]

    # 权限更新（全量替换）
    patch = client.patch(
        f"{API}/admin/roles/{role['id']}",
        json={"perm_codes": ["knowledge", "dictionary"]},
        headers=headers,
    )
    assert patch.status_code == 200
    assert sorted(patch.json()["perm_codes"]) == ["dictionary", "knowledge"]

    # 用户创建并绑定角色
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "qa01", "password": "qa123456", "role_ids": [role["id"]]},
        headers=headers,
    )
    assert resp.status_code == 201
    user = resp.json()
    assert user["roles"] == ["readonly_qa"]

    # 重置密码 + 停用
    patch = client.patch(
        f"{API}/admin/users/{user['id']}",
        json={"password": "newpass123", "is_active": False},
        headers=headers,
    )
    assert patch.status_code == 200
    assert patch.json()["is_active"] is False
    # 旧密码与停用双重拒绝
    assert _login("qa01", "qa123456").status_code == 401

    # 停用用户的存量 token 立即失效（DB 重载语义）
    qa_token = None
    client.patch(f"{API}/admin/users/{user['id']}", json={"is_active": True}, headers=headers)
    qa_token = _token("qa01", "newpass123")
    client.patch(f"{API}/admin/users/{user['id']}", json={"is_active": False}, headers=headers)
    assert client.get(f"{API}/admin/users", headers=_h(qa_token)).status_code == 401


def test_admin_duplicate_username_422():
    admin = _token()
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "admin", "password": "whatever123"},
        headers=_h(admin),
    )
    assert resp.status_code == 422


def test_admin_unknown_perm_code_422():
    resp = client.post(
        f"{API}/admin/roles",
        json={"role_code": "bad_role", "role_name": "x", "perm_codes": ["nonexistent"]},
        headers=_h(_token()),
    )
    assert resp.status_code == 422


def test_admin_unknown_role_id_422():
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "u_unknown_role", "password": "pass123456", "role_ids": [9999]},
        headers=_h(_token()),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 权限撤销即时生效（DB 重载，非信任 token claims）
# ---------------------------------------------------------------------------

def test_perm_revocation_takes_effect_immediately():
    admin = _token()
    headers = _h(admin)
    # 建一个有 dictionary 权限的用户
    resp = client.post(
        f"{API}/admin/roles",
        json={"role_code": "tmp_dict", "role_name": "临时字典", "perm_codes": ["dictionary"]},
        headers=headers,
    )
    role = resp.json()
    resp = client.post(
        f"{API}/admin/users",
        json={"username": "dict01", "password": "dict123456", "role_ids": [role["id"]]},
        headers=headers,
    )
    uid = resp.json()["id"]
    token = _token("dict01", "dict123456")
    assert client.post(
        f"{API}/dictionary/search", json={"question": "x"}, headers=_h(token)
    ).status_code == 200
    # 撤掉角色权限 → 同一 token 立即 403
    client.patch(f"{API}/admin/roles/{role['id']}", json={"perm_codes": []}, headers=headers)
    assert client.post(
        f"{API}/dictionary/search", json={"question": "x"}, headers=_h(token)
    ).status_code == 403
    client.patch(f"{API}/admin/users/{uid}", json={"is_active": False}, headers=headers)


# ---------------------------------------------------------------------------
# 服务层：未认证直调（内部调用放行）与权限拒绝
# ---------------------------------------------------------------------------

def test_chat_service_internal_call_without_auth_allowed():
    from finrag.services.auth_service import AuthContext
    from finrag.services.chat_service import ChatService

    auth = AuthContext(user_id=2, username="a", roles=["analyst"], perms={"nl2sql"})
    # nl2sql 有权限：校验通过不抛
    ChatService._ensure_scene_perm(auth, "nl2sql")
    # knowledge 无权限：403
    with pytest.raises(Exception):
        ChatService._ensure_scene_perm(auth, "knowledge")
    # 未认证（内部/脚本）：放行
    ChatService._ensure_scene_perm(None, "knowledge")


def test_load_auth_context_assembles_roles_and_perms():
    from finrag.services.auth_service import load_auth_context

    db = SessionLocal()
    try:
        ctx = load_auth_context(db, 1)  # seed 的 admin
        assert ctx.username == "admin"
        assert "admin" in ctx.roles
        assert "system_manage" in ctx.perms
        assert ctx.has_perm("nl2sql")
    finally:
        db.close()
