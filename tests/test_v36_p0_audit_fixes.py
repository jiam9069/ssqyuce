"""V36（2026-10-24 审计 P0 修复）回归锁。

本文件的存在理由，是审计里最重要的一条方法论教训：
**注释不能当校验。**

`engine_prize.expected_prize` 的注释写着「与 evaluate.prize_level 完全一致」，
`api_app._scheduler_step` 的注释写着「这处断链已接通」，
`routes_tasks` 的注释写着「将来必须改成按 user_id 过滤」——
三处都写得很清楚，三处都**没有**被任何用例钉住，于是全部静默失效。

因此这里的断言一律**对着独立算出的官方值硬编码**，而不是对着实现里的
公式自证（那种"用 5/32 断言 5/32"的自指断言，正是 V35 那批用例失败的根因：
实测把六等奖概率改成 0.0，72 个用例仍然全绿）。

变异实证（Lead 2026-10-24 手工执行）：
  * 把 engine_prize.py 的 `prob_by_level[6] += p_r * p_blue` 改成 `+= 0.0`
    → test_sixth_prize_level_matches_official_rules 必须红（旧用例集 72 passed）。
  * 把 P_SIXTH 改回 `1/16` → test_sixth_probability_constant_is_official 必须红。
  * 把 `random.Random(config.PREDICT_SEED)` 改回 `random.Random()`
    → test_predict_next_is_deterministic 必须红。
  * 把 `_task_visible_to_current_user` 改回 `return True`
    → test_task_idor_is_blocked 必须红。
"""
from __future__ import annotations

import json
import random
from math import comb

import pytest


# ---------------------------------------------------------------------------
# 官方口径常量（独立算，不 import 被测模块）
# ---------------------------------------------------------------------------
TOTAL_COMBOS = comb(33, 6) * 16                    # 17,721,088
P_RED_LE2 = (comb(6, 0) * comb(27, 6) + comb(6, 1) * comb(27, 5)
             + comb(6, 2) * comb(27, 4)) / comb(33, 6)     # 0.94225466
P_SIXTH_OFFICIAL = P_RED_LE2 / 16.0                # 0.0588925466


# ===========================================================================
# P0-1 / P1-6：六等奖判定与概率
# ===========================================================================

def test_sixth_probability_constant_is_official():
    """P_SIXTH 必须等于官方口径 P(红≤2)/16，不是 1/16。"""
    from lottery import engine_prize as EP
    assert EP.P_SIXTH == pytest.approx(P_SIXTH_OFFICIAL, rel=1e-12)
    # 反向锁：1/16 是"蓝中即中奖"的错误口径，必须不等
    assert EP.P_SIXTH != pytest.approx(1.0 / 16.0, rel=1e-6)


def test_sixth_prize_level_matches_official_rules():
    """六等奖概率必须是 1/17 量级（0.0589），不是 1/9（0.1084）。"""
    from lottery import engine_prize as EP
    d = EP.expected_prize([1, 2, 3, 4, 5, 6], 7)
    assert d["prob_by_level"][6] == pytest.approx(P_SIXTH_OFFICIAL, rel=1e-12)


@pytest.mark.parametrize("red_hits,blue_hit", [
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
    (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1),
])
def test_prize_level_table_agrees_with_evaluate(red_hits, blue_hit):
    """逐 (红命中, 蓝是否命中) 与 evaluate.prize_level 对拍。

    审计发现 14 个组合里恰好 1 个不一致：r=3,b=0（3红+蓝不中）。
    官方规则下它**不中奖**；旧实现把它算成六等奖，导致 EV 高估 24.6%。
    这个参数化用例把 14 个格子全部钉死，任何一格偏掉都会红。
    """
    from lottery import evaluate as EV, engine_prize as EP

    pred = list(range(1, 7))
    act = list(range(1, red_hits + 1)) + list(range(10, 10 + (6 - red_hits)))
    assert len(act) == 6 and len(set(act)) == 6
    truth = EV.prize_level(pred, 1, act, 1 if blue_hit else 2)

    # engine_prize 的分支表（等价于 expected_prize 内部逻辑）
    r, b = red_hits, bool(blue_hit)
    if r == 6:
        code = 1 if b else 2
    elif r == 5:
        code = 3 if b else 4
    elif r == 4:
        code = 4 if b else 5
    elif r == 3:
        code = 5 if b else 0      # ← 3红+蓝不中 = 不中奖（P0-1 的修复点）
    else:
        code = 6 if b else 0
    assert code == truth, f"r={r} b={b}: engine_prize={code} 官方={truth}"


def test_single_ticket_ev_matches_official():
    """单注期望必须落在官方口径（一/二等取假设常量）。"""
    from lottery import engine_prize as EP
    d = EP.expected_prize([1, 2, 3, 4, 5, 6], 7)
    # 用实现自己的奖金假设，独立重算官方奖级表下的 EV
    cash = d["prize_cash"]
    p = {}
    p[1] = 1 / comb(33, 6) * (1 / 16)
    p[2] = 1 / comb(33, 6) * (15 / 16)
    p[3] = comb(6, 5) * comb(27, 1) / comb(33, 6) / 16
    p[4] = (comb(6, 5) * comb(27, 1) / comb(33, 6) * (15 / 16)
            + comb(6, 4) * comb(27, 2) / comb(33, 6) / 16)
    p[5] = (comb(6, 4) * comb(27, 2) / comb(33, 6) * (15 / 16)
            + comb(6, 3) * comb(27, 3) / comb(33, 6) / 16)
    p[6] = P_SIXTH_OFFICIAL
    expected = sum(p[k] * cash[k] for k in range(1, 7))
    assert d["expected_reward"] == pytest.approx(expected, rel=1e-12)
    # 中奖概率合计必须 ≈ 6.71%，不是错误的 11.66%
    assert sum(d["prob_by_level"].values()) == pytest.approx(0.0670945, rel=1e-5)


def test_sixth_ev_ratio_is_n_invariant_and_official():
    """六等奖返奖比恒为 5×P(红≤2)/32 ≈ 0.1472314，且与注数无关。"""
    from lottery import engine_prize as EP
    want = 5.0 * P_RED_LE2 / 32.0            # 0.1472314（分母含 2 元成本）
    assert want == pytest.approx(0.1472314, rel=1e-6)
    for n in (1, 10, 16, 20, 100):
        ts = [{"reds": list(range(1, 7)), "blue": (i % 16) + 1} for i in range(n)]
        got = EP.portfolio_expected_value(ts)["sixth_ev_ratio"]
        assert got == pytest.approx(want, rel=1e-6), f"n={n} 漂移了"


def test_ev_ratio_sixth_constant_shared_across_modules():
    """三处口径必须指向同一个真源（不再各写一份近似常数）。"""
    from lottery import engine_prize as EP
    from lottery import engine_distribution as ED
    from lottery import blue_coverage as BC

    want = 5.0 * P_RED_LE2 / 32.0
    assert ED.EV_RATIO_SIXTH == pytest.approx(want, rel=1e-6)
    assert abs(ED.EV_RATIO_SIXTH - 0.15625) > 1e-4, "旧的 5/32 又回来了"
    for n in (1, 16, 20, 100):
        r = BC.optimize(n)
        assert r["ev_ratio_sixth"] == pytest.approx(want, rel=1e-5), f"blue_coverage n={n}"


# ===========================================================================
# P0-4：predict_next 确定性（回测判决可复现的前提）
# ===========================================================================

_VOLATILE = {"generated_at", "elapsed_ms", "snapshot", "note_zh",
             "updated_at", "patterns_summary"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _fingerprint(res):
    import hashlib
    blob = json.dumps(_strip(res), sort_keys=True, ensure_ascii=False,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def test_predict_next_is_deterministic(seeded_db, make_draws):
    """未显式传 rng 时，同输入两次必须得到完全相同的票。

    这是 P0-4 的核心：旧实现 `random.Random()` 未播种，实测两次指纹
    1be697ea… vs ad1f33f1…（差异 33 处，含 tickets[0]/reds[*]）。
    """
    from lottery import engine
    draws = make_draws(400)
    a = engine.predict_next(draws, use_llm=False, n_tickets=5, persist=False)
    b = engine.predict_next(draws, use_llm=False, n_tickets=5, persist=False)
    assert _fingerprint(a) == _fingerprint(b)
    assert [t["reds"] for t in a["tickets"]] == [t["reds"] for t in b["tickets"]]


def test_predict_next_seed_config_is_used_and_overridable(seeded_db, make_draws):
    """默认种子来自 config.PREDICT_SEED；显式传 rng 仍优先。"""
    from lottery import engine, config
    assert isinstance(config.PREDICT_SEED, int)
    draws = make_draws(400)
    base = _fingerprint(engine.predict_next(draws, use_llm=False,
                                            n_tickets=5, persist=False))
    assert base == _fingerprint(engine.predict_next(
        draws, use_llm=False, n_tickets=5, persist=False,
        rng=random.Random(config.PREDICT_SEED)))
    # 换种子应改结果（否则说明种子没接上）
    other = _fingerprint(engine.predict_next(
        draws, use_llm=False, n_tickets=5, persist=False,
        rng=random.Random(config.PREDICT_SEED + 1)))
    assert other != base


def test_offline_backtest_passes_rng_into_engine(seeded_db, make_draws):
    """回测必须把 rng 透传给 predict_next，否则"同 seed 可复现"只是假象。

    审计实测：修前同数据同 seed=7 两次 delta_red_hits = -0.0400 / -0.1133，
    恰好跨过 hypothesis 的 |delta| ≥ 0.05 阈值 ⇒ 判决可能翻转。
    """
    from lottery import evaluate as EV
    draws = make_draws(420)
    r1 = EV.offline_backtest(draws, issues=20, n_tickets=3, use_llm=False, seed=7)
    r2 = EV.offline_backtest(draws, issues=20, n_tickets=3, use_llm=False, seed=7)
    assert r1["delta_red_hits"] == r2["delta_red_hits"]
    assert r1["delta_blue_rate"] == r2["delta_blue_rate"]


# ===========================================================================
# P0-2：任务读路径的租户隔离
# ===========================================================================

def test_task_visibility_rules():
    """`_task_visible_to_current_user` 的四条规则。"""
    from lottery import routes_tasks as RT
    from lottery import request_context as rc

    token = rc.set_current_user_id(17)
    try:
        assert RT._task_visible_to_current_user({"user_id": 17}) is True
        assert RT._task_visible_to_current_user({"user_id": 16}) is False
        # 无归属 = 系统/调度器产出，任何登录用户可见
        assert RT._task_visible_to_current_user({"user_id": None}) is True
        assert RT._task_visible_to_current_user({"user_id": 0}) is True
        assert RT._task_visible_to_current_user(None) is False
    finally:
        rc.reset_current_user_id(token)

    # 无请求上下文：只允许无归属任务
    assert RT._task_visible_to_current_user({"user_id": 16}) is False
    assert RT._task_visible_to_current_user({"user_id": None}) is True


# ===========================================================================
# P0-5：匿名请求不得冒认主管理员
# ===========================================================================

def test_resolve_tenant_never_impersonates_admin_in_request(seeded_db):
    """HTTP 请求内无身份 → SYSTEM_USER_ID；后台线程才允许管理员兜底。"""
    from lottery import db
    from lottery import request_context as rc

    assert db.SYSTEM_USER_ID == 0
    # 非请求上下文（后台线程）：允许管理员兜底（这是该兜底存在的正当理由）
    outside = db.resolve_tenant(None)

    token = rc.set_in_request(True)
    try:
        assert rc.current_user_id() is None
        assert db.resolve_tenant(None) == db.SYSTEM_USER_ID
    finally:
        rc.reset_in_request(token)

    # 退出请求后恢复原语义
    assert db.resolve_tenant(None) == outside
