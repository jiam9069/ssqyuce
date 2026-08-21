import os


def _setup(tmp_path):
    os.environ["LOTT_HOME"] = str(tmp_path)
    os.environ["LOTT_DB"] = str(tmp_path / "test.db")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    db.close()
    return config, db


def test_notify_no_new_evaluation_is_noop(tmp_path):
    config, db = _setup(tmp_path)
    from lottery import notify
    config.NOTIFY_WEBHOOK = "https://example.invalid"
    result = notify.notify_after_check({"newly_checked": 0, "rows": []})
    assert result["sent"] is False
    assert result["reason"] == "no_new_evaluation"


def test_notify_webhook_failure_is_silent(tmp_path, monkeypatch):
    config, db = _setup(tmp_path)
    from lottery import notify
    config.NOTIFY_WEBHOOK = "https://example.invalid"
    db.save_eval("2026001", 1, 0, 0, 1)
    monkeypatch.setattr(notify, "_post_json", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(db, "load_draws", lambda: [{"issue": "2026001", "reds": [1,2,3,4,5,6], "blue": 7}])
    result = notify.notify_after_check({"newly_checked": 1, "rows": [{"issue": "2026001", "red_hits": 1, "blue_hit": 0, "reward": 0}]})
    assert result["sent"] is False
    assert result["channels"]["webhook"]["sent"] is False
