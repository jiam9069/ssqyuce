"""LLM 实际生成注数模式（LOTT_LLM_ONLY_OUTPUT，默认 1）。

背景：引擎原本把 LLM 候选与统计/ML 候选混在一起，再按“蓝球覆盖优先”补齐到
N_TICKETS。当 1 轮 LLM 只产出 4 注时，用户会看到 10 注，但只有 4 注真正来自
LLM 推理——口径不一致。本测试锁定新语义：

- LLM 产出候选 → 最终注数 = LLM 实际产出注数，**不补齐**统计票；
- LLM 未参与（纯统计 / LLM 失败降级）→ 保持原有“补齐到 N_TICKETS”行为；
- coverage / blue_compound / dan_tuo 等显式组合策略模式不受影响。
"""
from __future__ import annotations

import random

import pytest


def _mk_draws(n: int):
    rnd = random.Random(20260101)
    out = []
    for i in range(n):
        reds = sorted(rnd.sample(range(1, 34), 6))
        out.append({
            "issue": f"{2026001 + i}", "date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "reds": reds, "blue": rnd.randint(1, 16),
        })
    return out


def _llm_ticket(i: int, blue: int):
    """构造一个合法的 LLM 候选（method 以 llm: 开头）。"""
    return {
        "reds": sorted([(i + j) % 33 + 1 for j in range(6)]),
        "blue": blue,
        "method": "llm:fake-model",
        "confidence": 60 - i,
        "reasoning": "结构均衡",
    }


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOTT_ML_ENABLED", "0")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    config.ADAPTIVE_STATE_FILE = tmp_path / "adaptive_state.json"
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    db.close()
    return config


def test_llm_only_returns_actual_count_without_padding(env, monkeypatch):
    """1 轮 LLM 产出 4 注、请求 10 注 → 只输出这 4 注，且全部来自 LLM。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    produced = [_llm_ticket(i * 3 + 1, b) for i, b in enumerate([1, 9, 10, 2])]
    assert len(produced) == 4

    monkeypatch.setattr(E, "llm_tickets", lambda *a, **k: [dict(t) for t in produced])
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", True)

    res = E.predict_next(draws, use_llm=True, use_ml=False, n_tickets=10,
                         persist=False, rng=random.Random(1))

    assert res["actual_tickets"] == 4, "应只输出 LLM 实际产出的 4 注，不补齐到 10"
    assert res["requested_tickets"] == 10
    assert len(res["tickets"]) == 4
    assert all(t["method"].startswith("llm:") for t in res["tickets"]), \
        "补齐的统计票不应出现——所有输出注必须来自 LLM"
    assert res["llm_only_output"] is True
    assert res["shortfall_reason"] == "llm_only_actual_count"


def test_llm_only_no_duplicate_tickets(env, monkeypatch):
    """小候选池下不得产生重复注（_pick_blue_coverage 补齐路径会重复，这里必须绕开）。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    produced = [_llm_ticket(i * 3 + 1, b) for i, b in enumerate([3, 5, 8])]
    monkeypatch.setattr(E, "llm_tickets", lambda *a, **k: [dict(t) for t in produced])
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", True)

    res = E.predict_next(draws, use_llm=True, use_ml=False, n_tickets=10,
                         persist=False, rng=random.Random(2))

    keys = [(tuple(t["reds"]), t["blue"]) for t in res["tickets"]]
    assert len(keys) == len(set(keys)), "输出注不得重复"


def test_llm_only_respects_n_tickets_cap(env, monkeypatch):
    """LLM 产出多于 n_tickets 时，按置信度截断到 n_tickets（上限仍是请求注数）。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    produced = [_llm_ticket(i * 2 + 1, (i % 15) + 1) for i in range(6)]
    monkeypatch.setattr(E, "llm_tickets", lambda *a, **k: [dict(t) for t in produced])
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", True)

    res = E.predict_next(draws, use_llm=True, use_ml=False, n_tickets=3,
                         persist=False, rng=random.Random(3))

    assert res["actual_tickets"] == 3
    assert all(t["method"].startswith("llm:") for t in res["tickets"])


def test_pure_stat_mode_still_fills_to_n_tickets(env, monkeypatch):
    """LLM 未参与时，行为不变：统计候选池补齐到 n_tickets。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", True)

    res = E.predict_next(draws, use_llm=False, use_ml=False, n_tickets=10,
                         persist=False, rng=random.Random(4))

    assert res["actual_tickets"] == 10, "纯统计模式仍应补齐到请求注数"
    assert res["llm_only_output"] is False
    assert res["shortfall_reason"] is None


def test_llm_only_disabled_restores_padding(env, monkeypatch):
    """LOTT_LLM_ONLY_OUTPUT=0 时恢复旧行为：混选补齐到 n_tickets。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    produced = [_llm_ticket(i * 3 + 1, b) for i, b in enumerate([1, 9, 10, 2])]
    monkeypatch.setattr(E, "llm_tickets", lambda *a, **k: [dict(t) for t in produced])
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", False)

    res = E.predict_next(draws, use_llm=True, use_ml=False, n_tickets=10,
                         persist=False, rng=random.Random(5))

    assert res["llm_only_output"] is False
    assert res["actual_tickets"] == 10, "关闭开关后应回到“混选补齐”旧行为"
    assert any(not t["method"].startswith("llm:") for t in res["tickets"])


def test_coverage_mode_unaffected_by_llm_only(env, monkeypatch):
    """coverage 是显式组合策略模式，不受 LLM 实际注数模式影响。"""
    from lottery import engine as E
    draws = _mk_draws(400)
    produced = [_llm_ticket(i * 3 + 1, b) for i, b in enumerate([1, 9, 10, 2])]
    monkeypatch.setattr(E, "llm_tickets", lambda *a, **k: [dict(t) for t in produced])
    monkeypatch.setattr(env, "LLM_ONLY_OUTPUT", True)

    res = E.predict_next(draws, use_llm=True, use_ml=False, n_tickets=8,
                         persist=False, rng=random.Random(6), coverage=True)

    assert res["coverage_mode"] is True
    assert res["llm_only_output"] is False, "coverage 模式不启用 llm_only 路径"
    assert res["actual_tickets"] == 8
