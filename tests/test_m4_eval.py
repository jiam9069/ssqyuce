import os
import tempfile


def _setup_db(tmp_path):
    os.environ["LOTT_HOME"] = str(tmp_path)
    os.environ["LOTT_DB"] = str(tmp_path / "test.db")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    db.close()
    return db


def test_m4_eval_details_idempotent(tmp_path):
    db = _setup_db(tmp_path)
    draw = {"issue": "2026001", "date": "2026-01-01", "reds": [1, 2, 3, 4, 5, 6], "blue": 7}
    tickets = [
        {"reds": [1, 2, 3, 4, 5, 8], "blue": 7},
        {"reds": [10, 11, 12, 13, 14, 15], "blue": 1},
    ]
    db.save_eval_meta("2026001", tickets, {"llm_used": False})
    first = db.save_eval_details("2026001", "stat:freq", tickets, draw)
    second = db.save_eval_details("2026001", "stat:freq", tickets, draw)
    assert first["tickets"] == second["tickets"] == 2
    report = db.cumulative_eval(limit=120)
    assert len(report["methods"]) == 1
    assert report["methods"][0]["tickets"] == 2
    assert len(report["methods"][0]["rows"]) == 2
    assert report["rolling_windows"] == [10, 30, 60]
    assert report["methods"][0]["rolling"][-1]["w10"]["red_hits"]["n"] == 2


def test_m4_eval_meta_methods(tmp_path):
    db = _setup_db(tmp_path)
    tickets = [
        {"reds": [1, 2, 3, 4, 5, 6], "blue": 1},
        {"reds": [7, 8, 9, 10, 11, 12], "blue": 2},
    ]
    db.save_eval_meta("2026002", tickets, {"llm_used": True, "llm_models": ["test"]})
    rows = db.load_eval_meta("2026002")
    assert {r["method"] for r in rows} == {"mixed"}
    assert rows[0]["llm_enabled"] == 1


def test_m4_method_recommendations_need_paired_uniform(tmp_path):
    db = _setup_db(tmp_path)
    draw = {"issue": "2026004", "date": "2026-01-01", "reds": [1, 2, 3, 4, 5, 6], "blue": 7}
    stat = [{"reds": [1, 2, 3, 4, 5, 8], "blue": 7}]
    uniform = [{"reds": [10, 11, 12, 13, 14, 15], "blue": 1}]
    db.save_eval_details("2026004", "stat:freq", stat, draw)
    db.save_eval_details("2026004", "uniform", uniform, draw)
    report = db.method_recommendations(limit=120, min_sample=2)
    assert report["recommendations"][0]["paired_issues"] == 1
    assert report["recommendations"][0]["status"] == "insufficient_sample"


def test_m4_eval_meta_snapshot_has_methods(tmp_path):
    """M4.2：config_snapshot 应含方法模式/原始串/规格快照。"""
    import json
    db = _setup_db(tmp_path)
    tickets = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 3}]
    db.save_eval_meta("2026003", tickets, {"llm_used": False})
    rows = db.load_eval_meta("2026003")
    snap = json.loads(rows[0]["config_snapshot_json"] or "{}")
    assert snap["method_mode"] in ("production", "research")
    assert isinstance(snap["methods_raw"], str)
    assert snap["methods_spec"]["mode"] in ("all", "allow", "deny")
    assert isinstance(snap["methods_spec"]["tokens"], list)
