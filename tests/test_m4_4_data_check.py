from lottery.data_check import reconcile
from lottery import db


def draw(issue, reds, blue):
    return {"issue": issue, "reds": reds, "blue": blue}


def test_reconcile_matching_sources():
    a = [draw("1", [1,2,3,4,5,6], 7)]
    b = [draw("1", [1,2,3,4,5,6], 7)]
    result = reconcile(a, b)
    assert result["ok"] is True
    assert result["common"] == 1
    assert result["mismatches"] == []


def test_reconcile_detects_missing_and_number_mismatch():
    a = [draw("1", [1,2,3,4,5,6], 7), draw("2", [1,2,3,4,5,6], 8)]
    b = [draw("1", [1,2,3,4,5,6], 9), draw("3", [1,2,3,4,5,6], 1)]
    result = reconcile(a, b)
    assert result["ok"] is False
    assert result["only_primary"] == ["2"]
    assert result["only_secondary"] == ["3"]
    assert result["mismatches"][0]["issue"] == "1"


def test_reconcile_run_is_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(db.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db.config, "DB_PATH", tmp_path / "audit.db")
    db.close()
    audit_id = db.save_reconcile_run({"ok": False, "mismatches": [{"issue": "1"}]}, "backup")
    rows = db.load_reconcile_runs()
    assert audit_id == rows[0]["id"]
    assert rows[0]["status"] == "mismatch"
    assert rows[0]["summary"]["mismatches"][0]["issue"] == "1"
    db.close()
