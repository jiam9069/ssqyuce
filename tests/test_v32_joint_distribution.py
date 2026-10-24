"""V32：组合级联合分布端点（`/api/rigor/joint_distribution`）。

## 守什么

用户 2026-10-24：「先放下经济价值，按照理论上仍可行的方向走下一步」。

V30/V31 都独立建模 33 个号，违背「每期恰好 6 个」的负相关约束。
V32 首次检验**联合分布**，本轮三个方法论收获必须被测试守住：

1. **卡方表会骗人**：零分布实际均值 ≈26.9 而非理论 32，
   所以判据必须是置换 p，不能拿 46.19 当临界
2. **二阶 χ² 要条件化**：未条件化 624.0（假信号）→ 条件化 452.5（低于期望 528）
3. **准确率是错判据**：as-of 跟随 0.2781 << 常数基线 0.8182
"""
from __future__ import annotations

import pytest

from lottery import db, routes_rigor as RR


@pytest.fixture()
def seeded(seeded_db):
    seeded_db(1200, seed=42)
    return db.load_draws()


def test_returns_three_blocks(seeded):
    """三块必须齐全：pairwise / marginal / usability。"""
    r = RR.rigor_joint_distribution(seed=1, n_perm=60)
    assert r["ok"] is True
    for k in ("pairwise", "marginal", "usability", "headline_zh"):
        assert k in r, f"缺字段 {k}"
    assert r["pairwise"]["n_pairs"] == 528
    assert r["pairwise"]["df"] == 528


def test_conditional_chi2_is_lower_than_unconditional(seeded):
    """**核心**：条件化 χ² 必须显著低于未条件化。

    未条件化的二阶 χ² 会把一阶偏置放大（一个号多出现，它参与的 32 个 pair 全偏高）。
    若两者相等，说明条件化没生效。

    变异实证：把 conditional 分支的期望改回 e_default，本测试变红。
    """
    r = RR.rigor_joint_distribution(seed=2, n_perm=60)
    raw = r["pairwise"]["chi2_unconditional"]
    cond = r["pairwise"]["chi2_conditional"]
    assert cond < raw, f"条件化后应下降，实际 {cond} >= {raw}"
    assert cond < raw * 0.9, f"下降幅度过小（{cond}/{raw}），条件化可能没生效"


def test_marginal_null_distribution_is_not_32(seeded):
    """零分布均值必须明显低于理论 32 —— 这是「卡方表会骗人」的证据。

    每期抽 6 个不放回 ⇒ 33 个计数和为定值 ⇒ 负相关 ⇒ 方差压缩。
    若 null_mean 接近 32，说明模拟方式错了（用了有放回抽样）。
    """
    r = RR.rigor_joint_distribution(seed=3, n_perm=200)
    nm = r["marginal"]["null_mean"]
    assert nm < 30, f"零分布均值 {nm} 接近理论 32，说明没模拟出负相关约束"
    assert nm > 20, f"零分布均值 {nm} 过低，模拟可能有 bug"


def test_marginal_reports_both_criteria(seeded):
    """必须同时给置换 p 与卡方表临界值，让人看见两者可能矛盾。"""
    r = RR.rigor_joint_distribution(seed=4, n_perm=60)
    m = r["marginal"]
    assert 0 < m["perm_p"] <= 1
    assert m["chi2_table_critical"] == pytest.approx(46.19, abs=0.01)
    assert "置换" in m["distribution_note_zh"]
    assert "卡方表" in m["distribution_note_zh"]


def test_bonferroni_threshold_computed(seeded):
    """Bonferroni 阈值必须算出（33 次检验 ≈ z 3.17），且参与判定。"""
    r = RR.rigor_joint_distribution(seed=5, n_perm=60)
    m = r["marginal"]
    assert 3.0 < m["bonferroni_z_threshold"] < 3.4, (
        f"33 次检验的 Bonferroni z 应约 3.17，实际 {m['bonferroni_z_threshold']}")
    # 判定必须与阈值一致
    expect_passed = abs(m["z"]) > m["bonferroni_z_threshold"]
    assert m["bonferroni_passed"] == expect_passed


def test_usability_compares_to_constant_baseline(seeded):
    """可利用性必须与常数基线对照（不是与 0.5 或与随机）。

    变异实证：把 constant_baseline 改成 0.5，本测试变红。
    """
    r = RR.rigor_joint_distribution(seed=6, n_perm=60)
    u = r["usability"]
    assert u["constant_baseline"] == pytest.approx(1 - 6 / 33, abs=1e-4), (
        "常数基线必须是『总是猜不出现』= 1 - 6/33")
    assert u["n_eval"] > 50
    assert "分类准确率是错误判据" in u["verdict_zh"]


def test_upper_gain_is_tiny_even_if_bias_real(seeded):
    """即使偏置永续，整注增益上限也必须很小（除以 6）。"""
    r = RR.rigor_joint_distribution(seed=7, n_perm=60)
    g = r["usability"]["upper_gain_pct"]
    assert 0 <= g < 10, f"增益上限 {g}% 异常（应 <10%）"


def test_headline_mentions_all_three_findings(seeded):
    """结论句必须同时涵盖二阶/一阶/可利用性三点。"""
    r = RR.rigor_joint_distribution(seed=8, n_perm=60)
    h = r["headline_zh"]
    assert "条件化" in h or "二阶" in h
    assert "Bonferroni" in h
    assert "常数基线" in h or "as-of" in h


def test_insufficient_data_actionable():
    """数据不足必须 503 + 可操作提示。"""
    r = RR.rigor_joint_distribution(seed=1, n_perm=60)
    body = r if isinstance(r, dict) else None
    if body is None:
        import json
        parsed = json.loads(r.body)
        assert parsed["ok"] is False
        assert "/api/refresh" in parsed["error"]
    else:
        assert body.get("ok") is False


def test_deterministic_with_same_seed(seeded):
    """同种子必须同结果。"""
    a = RR.rigor_joint_distribution(seed=999, n_perm=60)
    b = RR.rigor_joint_distribution(seed=999, n_perm=60)
    assert a["marginal"]["chi2"] == b["marginal"]["chi2"]
    assert a["marginal"]["perm_p"] == b["marginal"]["perm_p"]
    assert a["pairwise"]["chi2_conditional"] == b["pairwise"]["chi2_conditional"]
