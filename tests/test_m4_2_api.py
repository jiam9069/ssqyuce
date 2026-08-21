"""M4.2 方法开关 API 冒烟测试（FastAPI TestClient）。"""
import json
import os

import pytest

from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    # 复位方法开关环境变量，避免污染同一会话中其它测试的“默认行为”
    monkeypatch.setenv("LOTT_METHODS", "")
    monkeypatch.setenv("LOTT_METHOD_MODE", "production")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    monkeypatch.setattr(config, "METHODS_CONFIG_FILE", tmp_path / "methods_config.json")
    # 复位 config 全局，避免 set_methods 的改动泄漏到其它测试
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    db.close()
    from lottery import api_app
    return TestClient(api_app.app)


def test_recommendations_endpoint(client):
    r = client.get("/api/eval/recommendations?limit=120&min_sample=60")
    assert r.status_code == 200
    data = r.json()
    assert data["baseline"] == "uniform"
    assert "recommendations" in data


def test_methods_status(client):
    r = client.get("/api/methods/status")
    assert r.status_code == 200
    st = r.json()
    assert st["mode"] in ("production", "research")
    assert "raw" in st
    assert st["spec"]["mode"] in ("all", "allow", "deny")
    assert st["effective"]["mode"] in ("all", "allow", "deny")
    assert set(st["families"]) == {"stat", "blend", "llm", "ml", "uniform"}
    assert {e["method"] for e in st["registry"]} >= {"stat:freq", "llm", "ml", "uniform"}


def test_methods_config_update(client):
    r = client.post("/api/methods/config",
                    json={"methods": "-llm", "mode": "research"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["mode"] == "research"
    st = client.get("/api/methods/status").json()
    assert st["mode"] == "research"
    assert st["raw"] == "-llm"
    assert st["effective"]["mode"] == "all"     # 研究模式忽略开关

    # 切回生产：应严格应用 -llm
    r2 = client.post("/api/methods/config",
                     json={"methods": "-llm", "mode": "production"})
    assert r2.status_code == 200
    st2 = client.get("/api/methods/status").json()
    assert st2["effective"]["mode"] == "deny"
    assert st2["families"]["llm"] is False
    assert st2["families"]["stat"] is True


def test_methods_config_validation(client):
    r = client.post("/api/methods/config", json={"methods": "$bad!tokens"})
    assert r.status_code == 400
    assert r.json()["ok"] is False
    r = client.post("/api/methods/config", json={"mode": "banana"})
    assert r.status_code == 400
    r = client.post("/api/methods/config", json={"methods": 123})
    assert r.status_code == 400
    # 非法请求不污染当前状态
    st = client.get("/api/methods/status").json()
    assert st["mode"] in ("production", "research")