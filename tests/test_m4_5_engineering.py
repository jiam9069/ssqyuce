from fastapi.testclient import TestClient


def test_migrations_are_ordered_and_idempotent(tmp_path, monkeypatch):
    from lottery import db, migrations
    import sqlite3
    conn = sqlite3.connect(tmp_path / "migration.db")
    conn.row_factory = sqlite3.Row
    assert migrations.apply(conn) == 2
    assert migrations.apply(conn) == 2
    assert conn.execute("select count(*) from schema_version").fetchone()[0] == 2
    conn.close()


def test_api_token_optional_and_required(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "api.db"))
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DB_PATH = tmp_path / "api.db"
    config.API_TOKEN = "test-secret"
    db.close()
    from lottery.api_app import app
    client = TestClient(app)
    assert client.get("/api/info").status_code == 401
    assert client.get("/api/info", headers={"Authorization": "Bearer test-secret"}).status_code == 200
    assert client.get("/api/health", headers={"Authorization": "Bearer test-secret"}).json()["api_auth_enabled"] is True
    config.API_TOKEN = None
    db.close()
