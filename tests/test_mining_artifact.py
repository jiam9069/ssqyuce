import json


def test_mining_artifact_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "ssq.db"))
    from lottery import config, db, mining

    config.BASE = tmp_path
    config.DATA_DIR = tmp_path / "data"
    config.DB_PATH = tmp_path / "ssq.db"
    db.close()
    db.get_conn()
    result = {
        "key": "mined_demo",
        "name_zh": "挖掘·demo",
        "kind": "short",
        "desc": "demo",
        "params": {},
        "grade": "B",
        "sample_size": 42,
        "margin": 0.01,
        "p_value": 0.1,
        "p_adj": 0.2,
        "direction": "above",
        "backtest": {},
    }
    path = config.DATA_DIR / "mining_artifact.json"
    mining.export_mining_artifact({"run_id": "mine_test"}, [result], [{"issue": "2026001"}])
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "ssq-mining-artifact-v1"
    assert mining.import_mining_artifact(path)["imported"] == 1
    assert [p["key"] for p in db.load_patterns() if p["key"] == "mined_demo"] == ["mined_demo"]
    db.close()


def test_mining_feature_vector_has_80_dimensions():
    from lottery.mining import _FEATURE_NAMES, _compute_features_for_number

    draws = [
        {"issue": str(i), "reds": [1, 3, 5, 7, 9, 11], "blue": 2}
        for i in range(40)
    ]
    values = _compute_features_for_number(draws, 30, 7)
    assert len(_FEATURE_NAMES) == 80
    assert len(values) == 80


def test_import_mining_artifact_rejects_unknown_format(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    from lottery import config, mining
    config.BASE = tmp_path

    path = tmp_path / "bad.json"
    path.write_text('{"format":"other"}', encoding="utf-8")
    try:
        mining.import_mining_artifact(path)
    except ValueError as exc:
        assert "格式" in str(exc)
    else:
        raise AssertionError("invalid artifact was accepted")
