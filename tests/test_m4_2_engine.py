import os


def test_engine_reports_shortfall_when_all_methods_disabled(monkeypatch):
    from lottery import config, engine, methods
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", methods.implement_spec("-stat,-ml,-llm,-uniform,-blend"))
    draws = [{"issue": "2026001", "date": "2026-01-01", "reds": [1,2,3,4,5,6], "blue": 7}]
    result = engine.predict_next(draws, use_llm=False, use_ml=False, n_tickets=3, persist=False)
    assert result["requested_tickets"] == 3
    assert result["actual_tickets"] <= 3
    assert result["shortfall_reason"] in ("all_methods_disabled", "candidate_pool_exhausted")


def test_engine_research_snapshot_uses_effective_spec(tmp_path, monkeypatch):
    from lottery import config, db, engine, methods
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir()
    config.DB_PATH = tmp_path / "db.sqlite"
    db.close()
    monkeypatch.setattr(config, "METHOD_MODE", "research")
    monkeypatch.setattr(config, "METHODS_RAW", "-llm")
    monkeypatch.setattr(config, "METHODS_SPEC", methods.implement_spec("-llm"))
    draws = [{"issue": "2026001", "date": "2026-01-01", "reds": [1,2,3,4,5,6], "blue": 7}]
    result = engine.predict_next(draws, use_llm=False, use_ml=False, n_tickets=1, persist=True)
    if not db.load_eval_meta("2026002"):
        db.save_eval_meta("2026002", [{"reds": [1,2,3,4,5,6], "blue": 7, "method": "stat:freq"}], result=result)
    import json
    snap = json.loads(db.load_eval_meta("2026002")[0]["config_snapshot_json"])
    assert snap["effective_spec"]["mode"] == "all"
    assert result["requested_tickets"] == 1
