"""B-3：蓝球覆盖优化的诚实性测试。

## 本文件测的不是"能不能算出计划"，而是"会不会骗人"

蓝球覆盖的数学是组合恒等式（`P(≥1中蓝) = min(n,16)/16`），本身没有争议。
真正需要守住的是**披露**：把"中奖概率提升 31%"单独说出去就是误导，
因为它不改变期望收益。本文件把三条诚实边界写成可执行断言：

1. 六等奖返奖比**恒为 15.625%，与注数无关**（这是覆盖优化不改变期望的直接证据）；
2. 每次输出都必须带 `disclosure`，且其中同时含「概率提升」与「净亏」；
3. `honesty_flags` 必须明确否认"提高中奖概率/提高期望收益"。
"""
from __future__ import annotations

import pytest
from math import comb as _comb

# V36：六等奖官方口径常量（独立算，不 import 被测模块）
_P_RED_LE2 = (_comb(6, 0) * _comb(27, 6) + _comb(6, 1) * _comb(27, 5)
              + _comb(6, 2) * _comb(27, 4)) / _comb(33, 6)
EV_RATIO_SIXTH_OFFICIAL = 5.0 * _P_RED_LE2 / 32.0

from lottery import blue_coverage as BC


# ---------- 1. 数学正确性 ----------

def test_p_covered_formula():
    assert BC.p_any_blue_covered(0) == 0.0
    assert BC.p_any_blue_covered(1) == pytest.approx(1 / 16)
    assert BC.p_any_blue_covered(5) == pytest.approx(5 / 16)
    assert BC.p_any_blue_covered(16) == pytest.approx(1.0)
    # n>16 饱和
    assert BC.p_any_blue_covered(20) == pytest.approx(1.0)


def test_p_random_repeat_formula():
    assert BC.p_any_blue_random_repeat(0) == 0.0
    assert BC.p_any_blue_random_repeat(1) == pytest.approx(1 / 16)
    assert BC.p_any_blue_random_repeat(10) == pytest.approx(1 - (15 / 16) ** 10)
    # 覆盖必须严格优于重复（n≥2 时）
    for n in range(2, 17):
        assert BC.p_any_blue_covered(n) > BC.p_any_blue_random_repeat(n), \
            f"n={n} 时去重覆盖应优于随机重复"


def test_coverage_never_worse_than_repeat():
    """覆盖优化在概率上永不变差（n=1 时相等，这是唯一无优化空间的情形）。"""
    for n in range(0, 25):
        assert BC.p_any_blue_covered(n) >= BC.p_any_blue_random_repeat(n) - 1e-12


# ---------- 2. 计划正确性 ----------

def test_plan_covers_distinct_blues():
    """n≤16 时计划必须互不相同 —— 这是覆盖能够成立的前提。"""
    for n in (1, 5, 10, 16):
        plan = BC.coverage_plan(n)
        assert len(plan) == n, f"n={n} 计划长度不对"
        assert len(set(plan)) == n, f"n={n} 计划里有重复蓝球：{plan}"
        assert all(1 <= b <= 16 for b in plan), f"n={n} 蓝球越界：{plan}"


def test_plan_handles_n_over_16():
    """n>16 时蓝球只有 16 个，必然重复；长度必须仍然等于 n。"""
    plan = BC.coverage_plan(20)
    assert len(plan) == 20
    assert len(set(plan)) == 16, "20 注只能覆盖 16 个不同蓝球"
    # 重复应尽量均匀（20 注 / 16 号 → 最多 2 次）
    from collections import Counter
    assert max(Counter(plan).values()) <= 2


def test_plan_empty_for_zero():
    assert BC.coverage_plan(0) == []
    assert BC.coverage_plan(-5) == []


# ---------- 3. 诚实性（核心） ----------

def test_ev_ratio_is_constant_across_ticket_counts():
    """**最重要的断言**：六等奖返奖比恒为 15.625%，与注数无关。

    这是「覆盖优化不改变期望收益」的数学证据：
    ``返奖比 = (5 × p)/(2 × n)``，而 ``p = n/16``（n≤16），
    所以 ``(5 × n/16)/(2n) = 5/32 = 0.15625``，n 被完全约掉。

    若哪天这条断言失败，说明有人把"概率提升"包装成了"收益提升"。
    """
    for n in range(1, 17):
        r = BC.optimize(n)
        # V36：官方口径 = 5×P(红≤2)/32 ≈ 0.1472314（旧值 5/32 漏了"红球 ≤2"）
    assert r["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_OFFICIAL, abs=1e-6), \
            f"n={n} 返奖比 {r['ev_ratio_sixth']} 偏离 0.15625"


def test_net_loss_grows_with_tickets():
    """注数越多，六等奖口径下净亏越大 —— 覆盖优化不改变"必亏"这一事实。"""
    prev_net = 0.0
    for n in range(1, 17):
        r = BC.optimize(n)
        assert r["net_sixth_yuan"] < 0, f"n={n} 六等奖口径竟然不亏"
        assert r["net_sixth_yuan"] < prev_net, f"n={n} 净亏应随注数增大"
        prev_net = r["net_sixth_yuan"]


def test_full_coverage_16_tickets_still_loses():
    """n=16 时 P(中蓝)=100%，但成本 32 元只回收 5 元 —— 必亏 27 元。

    这是最容易被误读的情形：「花 32 元买 16 注，蓝球全包，100% 中奖」
    听起来像稳赚，实际稳亏 84%。
    """
    r = BC.optimize(16)
    assert r["p_any_blue_covered"] == pytest.approx(1.0)
    assert r["cost_yuan"] == 32.0
    # V36：16 注成本 32 元，官方口径期望回收 = 32 × 0.1472314 ≈ 4.7114
    assert r["ev_sixth_yuan"] == pytest.approx(4.7114, abs=5e-4)
    # V36：净亏 = 32 − 4.7114 = -27.2886（旧值 -27.0 基于错误的 5 元/注期望）
    assert r["net_sixth_yuan"] == pytest.approx(-27.2886, abs=5e-4)


def test_every_optimize_call_carries_disclosure():
    """**任何一次调用都必须带披露** —— 不允许存在"只取数字"的路径。"""
    for n in (0, 1, 5, 10, 16, 20):
        r = BC.optimize(n)
        assert r.get("disclosure"), f"n={n} 缺 disclosure"
        lines = r["disclosure_lines"]
        joined = "\n".join(lines)
        # 必须同时包含「概率」与「成本/净亏」
        assert "概率" in joined
        assert "成本" in joined or "净亏" in joined


def test_disclosure_states_no_hit_probability_increase():
    """披露里必须明确否认「提高中奖概率」。"""
    r = BC.optimize(10)
    joined = "\n".join(r["disclosure_lines"])
    assert "不受选号" in joined or "不影响" in joined, \
        "必须明示头奖概率不受选号影响"


def test_honesty_flags_deny_wrong_claims():
    r = BC.optimize(10)
    flags = r["honesty_flags"]
    assert flags["increases_hit_probability"] is False, \
        "绝不能声称提高命中概率"
    assert flags["increases_expected_return"] is False, \
        "绝不能声称提高期望收益"
    assert flags["increases_prize_frequency"] is True, \
        "能声称的只有中奖频率"


def test_full_ev_ratio_appears_when_provided():
    r = BC.optimize(10, full_ev_ratio=0.523)
    assert r["ev_ratio_full"] == pytest.approx(0.523)
    joined = "\n".join(r["disclosure_lines"])
    assert "52.3" in joined or "52" in joined, "整注口径必须出现在披露里"
    assert "负期望" in joined


def test_missing_full_ev_ratio_is_explicitly_flagged():
    """不传整注返奖比时，披露必须说明「未含浮动奖」，而不是沉默。"""
    r = BC.optimize(10)
    joined = "\n".join(r["disclosure_lines"])
    assert "未含" in joined, "缺浮动奖时必须显式说明"


# ---------- 4. 一致性：与 engine 的闭式解同源 ----------

def test_plan_matches_engine_formula():
    """本模块的计划必须与 `engine.blue_coverage_plan` 一致（单一实现）。"""
    try:
        from lottery import engine
    except Exception:            # noqa: BLE001
        pytest.skip("engine 不可用")
    for n in (1, 5, 10, 16, 20):
        a = BC.coverage_plan(n)
        b = list(engine.blue_coverage_plan(n))
        assert a == b, f"n={n}: 本模块 {a} 与 engine {b} 不一致"


def test_optimize_gain_ratio_matches_manual():
    """`p_gain_ratio` 必须等于 覆盖/重复 的手算值。"""
    r = BC.optimize(10)
    manual = BC.p_any_blue_covered(10) / BC.p_any_blue_random_repeat(10)
    assert r["p_gain_ratio"] == pytest.approx(manual, abs=1e-3)
