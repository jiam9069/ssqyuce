"""V33：策略可区分性判据（`/api/rigor/detectability`）。

## 守什么

用户 2026-10-24：「抛开『能不能预测』，而是『在什么条件下，一个策略的表现
在统计上能与其随机基线区分开』」。

这是一个**设计工具**，不是审计工具。它必须：
1. **公式正确**：所需期数 = ((z_α+z_β)·sd(T)/(δ·n·E[H]))²
2. **vart_ratio 杠杆可见**：0.26 vs 1.0 必须给出 3.85 倍样本等价
3. **反向可用**：给出现有数据的最小可检出效应
"""
from __future__ import annotations

import math

import pytest

from lottery import routes_rigor as RR

E1 = 6 * 6 / 33
VAR1 = 6 * (6 / 33) * (1 - 6 / 33) * (33 - 6) / (33 - 1)


def _expect_periods(effect_pct, n, var_ratio, alpha=0.05, power=0.80):
    """独立复算（不用端点代码），作为交叉核对。"""
    def z(p):
        lo, hi = 0.0, 8.0
        for _ in range(90):
            mid = (lo + hi) / 2
            if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    sd = math.sqrt(n * VAR1 * var_ratio)
    delta = (effect_pct / 100) * n * E1
    return ((z(1 - alpha) + z(power)) * sd / delta) ** 2


def test_periods_formula_matches_independent_calc():
    """端点算出的期数必须与独立复算一致（公式是核心，不能错）。

    变异实证：把分母的 `n_tickets*e1` 改成 `e1`，本测试变红。
    """
    for eff in (20, 10, 5, 2):
        r = RR.rigor_detectability(n_tickets=10, effect_pct=eff, var_ratio=1.0)
        got = r["forward"]["periods_needed"]
        want = _expect_periods(eff, 10, 1.0)
        # 端点四舍五入到整数期（对外不呈现小数期数），故容差取 1 期
        assert got == pytest.approx(want, abs=1), (
            f"效应 {eff}%：端点 {got} vs 复算 {want}")


def test_stricter_alpha_needs_more_periods():
    """α 越小、检验力越高 → 需要更多期。方向必须正确。"""
    a = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0, alpha=0.05)
    b = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0, alpha=0.01)
    assert b["forward"]["periods_needed"] > a["forward"]["periods_needed"]

    c = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0, power=0.9)
    assert c["forward"]["periods_needed"] > a["forward"]["periods_needed"]


def test_smaller_effect_needs_more_periods():
    """效应越小 → 期数越多（与效应平方成反比）。"""
    a = RR.rigor_detectability(n_tickets=10, effect_pct=10, var_ratio=1.0)
    b = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0)
    # 效应减半 → 期数约 ×4
    ratio = b["forward"]["periods_needed"] / a["forward"]["periods_needed"]
    assert ratio == pytest.approx(4.0, rel=0.05), f"应为 4 倍，实际 {ratio:.2f}"


def test_var_ratio_leverage_is_reported():
    """**核心**：var_ratio 从 1.0 降到 0.26 必须给出约 3.85 倍样本等价。

    这是 V27 方差优化的真实理论价值 —— 不改变期望，但改变检验力。
    """
    r = RR.rigor_detectability(n_tickets=10, effect_pct=1.0, var_ratio=0.26)
    lev = r["var_ratio_leverage"]
    # 实际倍数 = n_independent / n_current ≈ 1/var_ratio
    assert lev["equivalent_sample_multiplier"] == pytest.approx(1 / 0.26, rel=0.05), (
        f"样本等价倍数应约 3.85，实际 {lev['equivalent_sample_multiplier']}")
    assert "V27" in lev["note_zh"], "必须点明这是 V27 的价值"


def test_reverse_gives_min_detectable_effect():
    """反向计算：现有 3518 期能检出多弱的效应。"""
    r = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0)
    rev = r["reverse"]
    assert rev["n_draws_available"] == 3518
    # 每期 10 注、var_ratio=1 时约 1.05%
    assert 0.8 < rev["min_detectable_effect_pct"] < 1.3, (
        f"最小可检出效应应约 1.05%，实际 {rev['min_detectable_effect_pct']}")


def test_more_tickets_lowers_detectable_effect():
    """注数越多，最小可检出效应越低（单期噪声相对变小）。"""
    prev = None
    for nt in (1, 5, 10, 20, 50):
        r = RR.rigor_detectability(n_tickets=nt, effect_pct=5, var_ratio=1.0)
        e = r["reverse"]["min_detectable_effect_pct"]
        if prev is not None:
            assert e < prev, f"n={nt} 的可检出效应 {e} 未低于上一档 {prev}"
        prev = e


def test_design_table_covers_range():
    """设计表必须覆盖多档注数，且单调递减。"""
    r = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0)
    rows = r["design_table"]["rows"]
    assert len(rows) >= 5
    vals = [row["min_detectable_effect_pct"] for row in rows]
    assert vals == sorted(vals, reverse=True), f"应随注数递减，实际 {vals}"


def test_verdict_scale_switches():
    """结论句必须随年数**三档**变化（1 年内 / 数年 / 不可验证）。

    变异实证：把中间档（years<=5）文案改成第一档的「1 年内可检出」，
    本测试必须变红。只测头尾两档不够 —— 中间档被写死时头尾仍不同，会假绿。
    """
    fast = RR.rigor_detectability(n_tickets=10, effect_pct=20, var_ratio=1.0)
    mid = RR.rigor_detectability(n_tickets=10, effect_pct=3, var_ratio=1.0)   # ≈2.9 年
    slow = RR.rigor_detectability(n_tickets=10, effect_pct=1, var_ratio=1.0)
    ya = fast["forward"]["years_needed"]
    yb = mid["forward"]["years_needed"]
    yc = slow["forward"]["years_needed"]
    assert ya < 1, f"第一档应 <1 年，实际 {ya}"
    assert 1 < yb < 30, f"中段应 1~30 年，实际 {yb}"
    assert yc > 20, f"末档应 >20 年，实际 {yc}"
    va = fast["forward"]["verdict_zh"]
    vb = mid["forward"]["verdict_zh"]
    vc = slow["forward"]["verdict_zh"]
    assert len({va, vb, vc}) == 3, f"三档文案应互不相同：{va[:20]}/{vb[:20]}/{vc[:20]}"
    assert "1 年内" in va
    # 中段必须**不含**「1 年内」，且报出自己年数 —— 这两条是变异抓手
    assert "1 年内" not in vb, f"中段（{yb} 年）不该说「1 年内」：{vb[:40]}"
    assert f"{yb:.1f}" in vb, f"中段必须报出实际年数 {yb}：{vb[:40]}"
    assert "不可验证" in vc or str(int(yc)) in vc


def test_query_defaults_are_normalized():
    """直接调用时 Query 默认对象必须被归一（否则 TypeError/500）。"""
    r = RR.rigor_detectability()   # 全部用默认值
    assert r["ok"] is True
    assert r["params"]["n_tickets"] == 10
    assert r["params"]["effect_pct"] == 5.0


def test_honesty_clarifies_it_does_not_estimate_effect():
    """诚实声明必须说清：本端点不估效应量，只说『若真有这么大能不能测出』。"""
    r = RR.rigor_detectability(n_tickets=10, effect_pct=5, var_ratio=1.0)
    h = r["honesty_zh"]
    assert "不评价" in h or "不估" in h
    assert "若效应真有这么大" in h or "能不能测出来" in h
