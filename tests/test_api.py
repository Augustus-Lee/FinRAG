"""API 冒烟测试：root / health / auth / 会话创建（不依赖外部服务）。"""

import jwt
from fastapi.testclient import TestClient

from finrag.config import get_settings
from finrag.main import app

client = TestClient(app)


def _login(username: str = "admin", password: str = "admin123") -> str:
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "FinRAG"
    assert "/docs" in body["docs"]
    assert "health" in body


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # 无 Qdrant 环境下应返回 down 而非报错（延迟导入 + 异常兜底）
    assert body["qdrant"] in {"up", "down"}


def test_login_success():
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    # JWT 可用服务端密钥校验通过（auth.py 重写后不再是 "admin.xxx" 格式）
    payload = jwt.decode(body["access_token"], get_settings().secret_key, algorithms=["HS256"])
    assert payload["sub"] == "1"
    assert body["role"] == "admin"
    assert "system_manage" in body["perms"]


def test_login_wrong_password_returns_401():
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_chat_session_created_with_memory_db():
    token = _login()
    resp = client.post(
        "/api/v1/chat/sessions",
        json={"mode": "knowledge", "title": "冒烟测试会话"},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["mode"] == "knowledge"
    assert body["id"] > 0


def test_unknown_route_returns_404_json():
    resp = client.get("/api/v1/nonexistent")
    assert resp.status_code == 404
