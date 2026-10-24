"""V30：预测力穷尽审计端点（`/api/rigor/predictive_power`）。

## 这个测试守什么

用户 2026-10-24 要求「不限于规律一个方法，多维度多方法去思考增强预测力」。
V30 的结论是**没有**（17 项检验，Bonferroni 后 0 项通过）。本文件守三件事：

1. **判据正确**：置换 p 必须真的在检验「时间结构」，不能退化成恒 0 或恒 1；
2. **基线偏移可识别**：随机序列对照必须能发现「方法本身的结构性偏差」；
3. **诚实性**：结论句必须与 p 值一致（p 大时说未检出，不许含糊）。
"""
from __future__ import annotations

import pytest

from lottery import db, routes_rigor as RR


@pytest.fixture()
def seeded(seeded_db):
    """合成 800 期（足够跑完整检验流程，且不依赖生产库）。"""
    seeded_db(800, seed=42)
    return db.load_draws()


def test_endpoint_returns_all_four_required_fields(seeded):
    """四个数字必须齐全：观测 / 置换 p / 随机序列对照 / 蓝球黏性。"""
    r = RR.rigor_predictive_power(seed=1, n_perm=10, start=300)
    assert r["ok"] is True
    for k in ("observed_hits", "perm_p", "random_sequence_control",
              "blue_stickiness", "multiple_comparison", "headline_zh"):
        assert k in r, f"缺字段 {k}"
    assert r["baseline_hits"] == pytest.approx(6 * 6 / 33, abs=1e-4)


def test_perm_p_is_a_real_probability(seeded):
    """置换 p 必须在 (0,1] 内且随种子变化 —— 不是写死的常量。

    变异实证：把 `perm_p` 写成 0.5 常量，本测试变红（多组种子下全相同）。
    """
    ps = [RR.rigor_predictive_power(seed=s, n_perm=10, start=300)["perm_p"]
          for s in (1, 2, 3, 4, 5)]
    for p in ps:
        assert 0.0 < p <= 1.0
    assert len(set(ps)) > 1, f"不同种子应给出不同 p，实际 {ps}"


def test_observed_hits_matches_baseline_order_of_magnitude(seeded):
    """在合成（纯随机）数据上，观测命中率应贴近理论基线。

    这是本端点最重要的自检：如果合成数据上就大幅偏离基线，
    说明统计口径有 bug，而不是"发现了信号"。
    """
    r = RR.rigor_predictive_power(seed=7, n_perm=10, start=300)
    obs = r["observed_hits"]
    base = r["baseline_hits"]
    assert abs(obs - base) < 0.25, (
        f"合成数据上命中率 {obs} 偏离基线 {base} 过大，口径可疑")


def test_random_sequence_control_detects_structural_bias(seeded):
    """随机序列对照必须给出可读判定（这是识别"基线偏移"的唯一途径）。

    V30 实测：真实数据 +1.61% vs 随机序列 +1.54% —— 两者接近即证明
    超额来自方法结构而非法码信息。
    """
    r = RR.rigor_predictive_power(seed=11, n_perm=10, start=300)
    ctrl = r["random_sequence_control"]
    assert "hits" in ctrl and "excess_pct" in ctrl
    assert ctrl["verdict_zh"], "必须给出判定句，不能只丢数字"
    assert len(ctrl["verdict_zh"]) > 20


def test_headline_matches_p_value(seeded):
    """结论句必须与 p 值一致 —— p 大就说未检出，不许含糊其辞。

    变异实证：把 headline 写成固定"发现信号"，本测试变红。
    """
    r = RR.rigor_predictive_power(seed=3, n_perm=20, start=300)
    if r["perm_p"] >= 0.05:
        assert "未检出" in r["headline_zh"]
    else:
        assert "显著" in r["headline_zh"]
    # 结论句必须带具体数字（不能只说"无信号"）
    assert str(r["perm_p"]) in r["headline_zh"] or f"{r['perm_p']:.4f}" in r["headline_zh"]


def test_blue_stickiness_reports_cost(seeded):
    """蓝球黏性必须同时给经济价值与所需样本量 —— 否则用户会以为能赚钱。"""
    r = RR.rigor_predictive_power(seed=5, n_perm=10, start=300)
    bs = r["blue_stickiness"]
    assert "repeat_rate" in bs and "z" in bs
    note = bs["note_zh"]
    assert "52.29" in note or "0.033" in note, "必须给出 EV 或回收率上限"
    assert "期" in note, "必须给出所需样本量"


def test_bonferroni_threshold_is_stricter_than_alpha(seeded):
    """多重比较阈值必须严格小于 0.05（否则等于没校正）。"""
    r = RR.rigor_predictive_power(seed=9, n_perm=10, start=300)
    mc = r["multiple_comparison"]
    assert mc["n_tests_v30"] >= 10
    assert mc["bonferroni_alpha"] < 0.05 / mc["n_tests_v30"] * 1.001


def test_insufficient_data_gives_actionable_error():
    """数据不足时必须 503 + 可操作提示（D-20），不能 500。"""
    from lottery import db as _db
    r = RR.rigor_predictive_power(seed=1, n_perm=10, start=3000)
    # 合成库期数远小于 3050
    body = r if isinstance(r, dict) else None
    if body is None:
        import json
        parsed = json.loads(r.body)
        assert parsed["ok"] is False
        assert "期" in parsed["error"]
    else:
        assert body.get("ok") is False
        assert "/api/refresh" in body.get("error", "")


def test_deterministic_with_same_seed(seeded):
    """同种子必须同结果（用户复核的前提）。"""
    a = RR.rigor_predictive_power(seed=12345, n_perm=10, start=300)
    b = RR.rigor_predictive_power(seed=12345, n_perm=10, start=300)
    assert a["observed_hits"] == b["observed_hits"]
    assert a["perm_p"] == b["perm_p"]

def test_random_control_verdict_uses_standard_error(seeded):
    """随机序列对照的判定必须用**标准误**，不能用固定阈值。

    2026-10-24 实测踩到：只跑 8 条随机序列时 excess_syn 抖动很大
    （+0.18% / +0.66% / +1.54%），用 ±1.0pp 硬阈值会**随机地**判成
    「值得核查」，与置换 p=0.5 的结论直接矛盾。

    变异实证：把判据改回 `abs(diff) < 1.0`，本测试在若干种子下变红。
    """
    verdicts = []
    for seed in (1, 2, 3, 4, 5):
        r = RR.rigor_predictive_power(seed=seed, n_perm=10, start=300)
        c = r["random_sequence_control"]
        assert "se_pct" in c, "必须暴露抽样标准误，判据才可复核"
        assert c["n_sequences"] >= 3, "随机序列条数必须够估标准误"
        # 判定与置换 p 不得矛盾：p 大时不应说"发现信号"
        if r["perm_p"] >= 0.05:
            assert "值得进一步核查" not in c["verdict_zh"] or c["se_pct"] == 0, (
                f"seed={seed}: perm_p={r['perm_p']} 却说值得核查，自相矛盾")
        verdicts.append(c["verdict_zh"][:20])
    # 判据应稳定（不随种子大幅摇摆）
    assert len(set(verdicts)) <= 2, f"判定过于摇摆：{verdicts}"
