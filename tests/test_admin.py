import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_api as svc  # noqa: E402


@pytest.fixture(autouse=True)
def configure_admin(monkeypatch):
    hashed = "sha256:" + hashlib.sha256(b"secret123").hexdigest()
    monkeypatch.setattr(svc.settings, "admin_username", "admin")
    monkeypatch.setattr(svc.settings, "admin_password_hash", hashed)
    monkeypatch.setattr(svc.settings, "admin_session_secret", "test-secret")
    monkeypatch.setattr(svc.settings, "admin_session_ttl_minutes", 5)
    monkeypatch.setattr(svc.settings, "admin_login_rpm", 10)
    monkeypatch.setattr(svc, "REDIS", None)
    yield


def test_admin_login_flow_sets_cookie_and_allows_reports():
    with TestClient(svc.app) as client:
        res = client.get("/admin/reports/system-health")
        assert res.status_code in {401, 403}

        res = client.post("/admin/login", json={"username": "admin", "password": "secret123"})
        assert res.status_code == 200
        cookie_name = svc.settings.admin_session_cookie
        assert cookie_name in res.cookies

        res = client.get("/admin/reports/system-health")
        assert res.status_code == 200
        assert res.json()["service"] == "simetrix-api"

        res = client.post("/admin/logout")
        assert res.status_code == 200

        res = client.get("/admin/reports/system-health")
        assert res.status_code in {401, 403}


def test_admin_login_rejects_bad_password():
    with TestClient(svc.app) as client:
        res = client.post("/admin/login", json={"username": "admin", "password": "wrong"})
        assert res.status_code == 401
