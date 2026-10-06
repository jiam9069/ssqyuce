"""v3.0 U2/U3/U4 后端升级测试：蓝球专项 / 方法自适应降权 / 组合覆盖优化。"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _mk_draws(n=400, seed=7):
    """合成 n 期随机开奖（独立均匀，双色球真实口径）。"""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        reds = sorted(rng.sample(range(1, 34), 6))
        out.append({
            "issue": f"{2026001 + i}",
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "reds": reds,
            "blue": rng.randint(1, 16),
            "order": reds,  # upsert_draws 需要 order 字段
        })
    return out


# ---------------------------------------------------------------------------
# U2 蓝球专项
# ---------------------------------------------------------------------------

def test_blue_specialist_is_valid_16dim():
    from lottery import models as M
    draws = _mk_draws(500)
    p = M.blue_specialist(draws)
    assert p.shape == (16,)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert (p > 0).all()
    # 与均匀的差距不应过大（收缩保证不极端集中）
    maxp = float(p.max())
    assert maxp < 0.25


def test_blue_specialist_warmup_fallback():
    from lottery import models as M
    draws = _mk_draws(20)
    p = M.blue_specialist(draws)
    assert np.allclose(p, 1.0 / 16, atol=1e-6)


def test_blue_coverage_priority_picks_distinct_blues():
    from lottery import engine as E
    rng = random.Random(12)
    scored = [{"reds": sorted(rng.sample(range(1, 34), 6)), "blue": b,
               "confidence": float(90 - i)}
              for i, b in enumerate([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
    picked = E._pick_blue_coverage(scored, 10)
    blues = [t["blue"] for t in picked]
    assert len(picked) == 10
    # 优先铺满不同蓝球（前 10 个不同蓝球）
    assert len(set(blues)) >= 6
    # 旧版兜底：同蓝上限
    legacy = E._pick_legacy_blue(scored, 10)
    assert len(legacy) == 10
    assert max(sum(1 for b in legacy if b == x) for x in set(blues)) <= max(2, 10 // 5)


def test_blue_compound_expands_k_distinct_blues_fixed_reds():
    from lottery import engine as E
    rng = random.Random(3)
    blue_probs = np.full(16, 1.0 / 16)
    reds = [1, 2, 3, 4, 5, 6]
    tickets = E._expand_blue_compound(reds, blue_probs, 5, rng)
    assert len(tickets) == 5
    assert len({t["blue"] for t in tickets}) == 5
    assert all(set(t["reds"]) == set(reds) for t in tickets)


def test_dan_tuo_expands_multiple_tickets():
    from lottery import engine as E
    rng = random.Random(5)
    draws = _mk_draws(500)
    red_blend = np.full(33, 6.0 / 33)
    ctx = E.constraint_ctx(draws)
    dan = [10, 20, 30]
    tickets = E._expand_dan_tuo(dan, 7, 8, rng, red_blend, ctx)
    assert len(tickets) == 8
    # 每注都包含所有胆
    assert all(all(d in t["reds"] for d in dan) for t in tickets)
    assert all(len(t["reds"]) == 6 for t in tickets)
    assert all(t["blue"] == 7 for t in tickets)


def test_blue_mode_uniform_returns_uniform():
    from lottery import engine as E, config
    old = config.BLUE_MODE
    config.BLUE_MODE = "uniform"
    try:
        out = E._apply_blue_mode(np.full(16, 1.0 / 8))
        assert np.allclose(out, 1.0 / 16)
    finally:
        config.BLUE_MODE = old


# ---------------------------------------------------------------------------
# U3 方法自适应降权 + LLM 角色重定位
# ---------------------------------------------------------------------------

def test_refresh_adaptive_weights_downweights_poor_method(tmp_path, monkeypatch):
    from lottery import config, db, methods as METH
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "db.sqlite"
    config.ADAPTIVE_STATE_FILE = tmp_path / "adaptive_state.json"
    config.ADAPTIVE_ENABLED = True
    config.ADAPTIVE_K = 20
    config.ADAPTIVE_DOWNWEIGHT_FACTOR = 0.5
    config.METHOD_MODE = "production"
    db.close()

    # 构造 60 期 paired 数据：bad 方法红球命中恒为 0，uniform 恒为 1
    for i in range(60):
        issue = f"2026{i + 1:04d}"
        for method, rh in (("uniform", 1), ("stat:bad", 0), ("stat:good", 1)):
            for seq in range(1, 4):
                db.get_conn().execute(
                    """INSERT INTO eval_details
                       (issue,method,seq,red_hits,blue_hit,prize_level,reward,ticket_cost,net_return,evaluated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (issue, method, seq, rh, 0, 0, 0.0, 2.0, -2.0, 1.0))
    db.get_conn().commit()

    state = METH.refresh_adaptive_weights(window=60)
    assert state["stat:bad"]["periods"] >= 50
    assert state["stat:bad"]["downweighted"] is True
    assert state["stat:bad"]["weight"] <= config.ADAPTIVE_DOWNWEIGHT_FACTOR
    assert state["stat:good"]["downweighted"] is False
    assert state["stat:good"]["weight"] == 1.0
    # 蓝球命中率统计正确
    assert state["stat:bad"]["blue_hit_rate"] == 0.0


def test_adaptive_weights_research_mode_no_downweight(tmp_path, monkeypatch):
    from lottery import config, db, methods as METH
    config.DATA_DIR = tmp_path / "data2"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "db2.sqlite"
    config.ADAPTIVE_STATE_FILE = tmp_path / "adaptive_state.json"
    config.ADAPTIVE_ENABLED = True
    config.ADAPTIVE_K = 20
    config.METHOD_MODE = "research"
    db.close()
    # 写入一个“差”方法的持久化状态
    METH._save_adaptive({"stat:bad": {"weight": 0.5, "downweighted": True,
                                      "strike": 25, "periods": 30,
                                      "red_hits_mean": 0.0, "blue_hit_rate": 0.0,
                                      "prize_rate_ge5": 0.0}})
    w = METH.adaptive_weights()
    # 研究模式 + 开启：返回空（不施加降权）
    assert w == {}


def test_blue_running_rate(tmp_path, monkeypatch):
    from lottery import config, db, methods as METH
    config.DATA_DIR = tmp_path / "data3"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "db3.sqlite"
    db.close()
    # 6 个蓝球命中里 2 个命中 → 2/6
    for i, hit in enumerate([1, 0, 0, 1, 0, 0]):
        db.get_conn().execute(
            """INSERT INTO eval_details
               (issue,method,seq,red_hits,blue_hit,prize_level,reward,ticket_cost,net_return,evaluated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (f"2026{i:04d}", "stat:freq", 1, 0, hit, 0, 0.0, 2.0, -2.0, 1.0))
    db.get_conn().commit()
    rate = METH._blue_running_rate(30)
    assert abs(rate - (2 / 6)) < 1e-6


def test_llm_blue_overridden_by_blue_probs(monkeypatch):
    """U3：LLM 不再决定蓝球——blue_probs（含 blue_specialist）覆盖 LLM 的 blue。"""
    from lottery import engine as E, llm_client

    def _fake_chat_json(system, user, **kw):
        # 观察轮：输出 schema 为 {"long_term": ...}；选号轮：输出 schema 含 "tickets"
        if '"tickets"' in str(user):
            return {"tickets": [{"reds": [1, 2, 3, 4, 5, 6], "blue": 16, "confidence": 60,
                                 "reasoning": "x", "patterns_used": [],
                                 "evidence": {}, "counter_evidence": [],
                                 "structure_scores": {}}]}
        return {"long_term": [], "mid_term": [], "short_term": [],
                "caveats": ["随机"]}

    monkeypatch.setattr(llm_client, "chat_json", _fake_chat_json)
    from lottery import config as _cfg
    monkeypatch.setattr(_cfg, "LLM_DISABLED", False)
    monkeypatch.setattr(_cfg, "llm_model_list",
                        lambda: [{"name": "fake", "base_url": "http://x/v1",
                                  "api_key": "k", "model": "fake"}])
    draws = _mk_draws(500)
    stats = __import__("lottery.features", fromlist=["features"]).compute_features(draws)
    rng = random.Random(1)
    # 蓝球全压到 1 号，确保覆盖后必为 1
    blue_probs = np.zeros(16)
    blue_probs[0] = 1.0
    red_probs = np.full(33, 6.0 / 33)
    tickets = E.llm_tickets(draws, stats, [], rng, llm_samples=1, llm_verify=False,
                            red_probs=red_probs, blue_probs=blue_probs)
    assert tickets, "LLM 通道应产出候选"
    assert all(t["blue"] == 1 for t in tickets), "LLM 的 blue(16) 应被 blue_specialist(1) 覆盖"


# ---------------------------------------------------------------------------
# U4 组合覆盖优化
# ---------------------------------------------------------------------------

def test_coverage_optimize_returns_budget_and_improves_blue_coverage():
    from lottery import engine as E
    rng = random.Random(11)
    rng_pool = random.Random(22)
    pool = [{"reds": sorted(rng_pool.sample(range(1, 34), 6)), "blue": rng_pool.randint(1, 16),
             "method": "stat:freq"}
            for _ in range(200)]
    picked = E.coverage_optimize(pool, 10, r=3, n_blues=4, trials=2000, rng=rng)
    assert len(picked) == 10
    assert len({tuple(t["reds"]) for t in picked}) == 10  # 去重
    # 蓝球覆盖应显著优于随机取 10 注的期望（>3 个不同蓝球）
    assert len({t["blue"] for t in picked}) >= 4


def test_predict_next_coverage_and_bet_meta():
    from lottery import engine as E
    draws = _mk_draws(500)
    rng = random.Random(99)
    res = E.predict_next(draws, use_llm=False, use_ml=False, n_tickets=8, persist=False,
                         rng=rng, coverage=True)
    assert res["coverage_mode"] is True
    assert res["bet_mode"] == "single"
    assert res["actual_tickets"] == 8
    assert 0 < res["blue_coverage_rate"] <= 1.0
    assert "adaptive" in res


# ---------------------------------------------------------------------------
# API 暴露
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOTT_METHODS", "")
    monkeypatch.setenv("LOTT_METHOD_MODE", "production")
    monkeypatch.setenv("LOTT_ML_ENABLED", "0")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    config.ADAPTIVE_STATE_FILE = tmp_path / "adaptive_state.json"
    monkeypatch.setattr(config, "METHODS_CONFIG_FILE", tmp_path / "methods_config.json")
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    db.close()
    # 写入一批开奖数据，让 predict 可用
    db.upsert_draws(_mk_draws(320))
    from lottery import api_app
    return TestClient(api_app.app)


def test_api_predict_bet_modes(client):
    r = client.get("/api/predict?use_llm=false&n_tickets=6&bet_mode=blue_compound")
    assert r.status_code == 200
    data = r.json()
    assert data["from_cache"] in (True, False)
    assert data["bet_mode"] in ("single", "blue_compound")
    assert 0 < data["blue_covered"] <= 6


def test_api_info_exposes_bet_and_adaptive(client):
    r = client.get("/api/info")
    assert r.status_code == 200
    data = r.json()
    assert data["bet_mode"] in ("single", "blue_compound", "dan_tuo")
    assert data["blue_mode"] in ("model", "uniform")
    assert "adaptive" in data and "enabled" in data["adaptive"]


def test_api_eval_cumulative_has_adaptive(client):
    r = client.get("/api/eval/cumulative")
    assert r.status_code == 200
    data = r.json()
    assert "adaptive" in data
    assert "enabled" in data["adaptive"]


def test_api_eval_coverage_endpoint(client):
    r = client.post("/api/eval/coverage?issues=10&n=6&coverage_r=3")
    assert r.status_code == 200
    data = r.json()
    assert "single" in data and "coverage" in data
    assert data["coverage"]["avg_blue_cover"] >= 1.0