import os


def test_draw_source_is_persisted(tmp_path):
    os.environ["LOTT_HOME"] = str(tmp_path)
    os.environ["LOTT_DB"] = str(tmp_path / "test.db")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    db.close()
    db.upsert_draws([{"issue":"2026001", "date":"2026-01-01", "reds":[1,2,3,4,5,6], "blue":7,
                      "order":[1,2,3,4,5,6], "source":"backup"}])
    assert db.load_draws()[0]["source"] == "backup"
