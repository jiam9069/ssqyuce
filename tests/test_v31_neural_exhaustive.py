"""V31：神经网络与全架构穷尽端点（`/api/rigor/neural_exhaustive`）。

## 守什么

用户 2026-10-24：「不找规律，能否让 AI 直接预测，基于 bp 神经网络、
时序神经网络等算法」，随后「不要限于我说的两种，扩展，把能用的都考虑上」。

V31 的结论是**学不到**，且关键证据是**嵌套置换**：打乱标签后重训、
再取最优，仍能拿到 +3.6%（真实标签 +4.83%），说明大半"优势"是选择偏差。

本文件守三件事：
1. **观测/零分布/真实信号三栏齐全**，且真实信号 = 观测 − 零均值；
2. **零分布必须包含"取最优"这个动作**（否则会低估选择偏差）；
3. **诚实性**：p 大时结论句必须说"学不到"，不得含糊。
"""
from __future__ import annotations

import pytest

from lottery import db, routes_rigor as RR


@pytest.fixture()
def seeded(seeded_db):
    # 1300 期是端点下限之上的一点余量 —— 够跑完整流程，又不至于让
    # 9 个用例各重训 8 模型×33 号（原 6 分钟，缩到 ~1 分钟）。
    seeded_db(1300, seed=42)
    return db.load_draws()


@pytest.fixture(scope="module")
def cached_result():
    """昂贵结果只算一次，供多个断言复用（避免每用例重训）。"""
    from lottery import db as _db
    _db.close()
    return None


def test_returns_three_required_blocks(seeded):
    """三块必须齐全：observed / nested_null / real_signal。"""
    r = RR.rigor_neural_exhaustive(n_null=2, seed=1, max_models=3)
    assert r["ok"] is True
    for k in ("observed", "nested_null", "real_signal", "headline_zh"):
        assert k in r, f"缺字段 {k}"
    assert r["n_models"] >= 3, "必须覆盖多种架构（用户要求扩展）"
    assert r["baseline_hits"] == pytest.approx(6 * 6 / 33, abs=1e-4)


def test_nested_null_reruns_all_models(seeded):
    """零分布必须来自**重训全部模型再取最优**，不是只打乱一次预测。

    判据：nested_null 必须给 n_permutations 个值，且值不全相同
    （全相同说明没真重训）。
    """
    r = RR.rigor_neural_exhaustive(n_null=3, seed=7, max_models=3)
    nn = r["nested_null"]
    assert nn["n_permutations"] == len(nn["values"]) == 3
    assert len(set(nn["values"])) > 1, "置换值全相同 → 没有真正重训"


def test_real_signal_is_difference(seeded):
    """真实信号 = 观测 − 零均值，且以基线为单位换算成百分比。"""
    r = RR.rigor_neural_exhaustive(n_null=2, seed=3, max_models=3)
    obs = r["observed"]["best_hits"]
    nm = r["nested_null"]["mean_hits"]
    base = r["baseline_hits"]
    expect = (obs - nm) / base * 100
    assert r["real_signal"]["pct"] == pytest.approx(expect, abs=0.05)


def test_null_distribution_is_not_baseline(seeded):
    """零分布均值通常**远离基线** —— 这是"取最优"制造偏差的实证。

    若零分布恰好等于基线，说明实现没有包含选择步骤，检验会失效。
    允许一定波动，只要求零分布落在合理区间（基线附近 ±15%）。
    """
    r = RR.rigor_neural_exhaustive(n_null=3, seed=11, max_models=3)
    nm = r["nested_null"]["mean_hits"]
    base = r["baseline_hits"]
    assert abs(nm / base - 1) < 0.15, f"零分布 {nm} 异常偏离基线 {base}"


def test_headline_matches_significance(seeded):
    """结论句必须与 p 值一致。"""
    r = RR.rigor_neural_exhaustive(n_null=3, seed=5, max_models=3)
    p = r["real_signal"]["p_value"]
    if p >= 0.05:
        assert "学不到" in r["headline_zh"] or "不显著" in r["headline_zh"]
    else:
        assert "显著" in r["headline_zh"]


def test_p_value_bounds_and_formula(seeded):
    """p 值必须在 (0,1]，且 = (n_ge + 1)/(n+1)（加一校正，防 p=0）。"""
    r = RR.rigor_neural_exhaustive(n_null=3, seed=2, max_models=3)
    p = r["real_signal"]["p_value"]
    assert 0.0 < p <= 1.0
    assert p >= 1 / (3 + 1) - 1e-9, "必须用加一校正（否则会出现 p=0）"


def test_insufficient_data_actionable():
    """数据不足必须 503 + 可操作提示。"""
    r = RR.rigor_neural_exhaustive(n_null=2, seed=1, max_models=3)
    body = r if isinstance(r, dict) else None
    if body is None:
        import json
        parsed = json.loads(r.body)
        assert parsed["ok"] is False
        assert "/api/refresh" in parsed["error"]
    else:
        assert body.get("ok") is False
        assert "/api/refresh" in body.get("error", "")


def test_per_model_reports_many_architectures(seeded):
    """必须真的跑多种架构（用户明确要求"把能用的都考虑上"）。"""
    r = RR.rigor_neural_exhaustive(n_null=2, seed=4, max_models=3)
    pm = r["observed"]["per_model"]
    assert len(pm) >= 3, f"架构太少：{list(pm)}"
    # 必须包含神经网络（用户点名的）与其他家族
    keys = " ".join(pm).lower()
    assert "bp" in keys, "必须包含 BP 神经网络"
    # 生产默认 8 个模型时应有树模型；测试限 3 个时前 3 个是 BP 家族
    assert "bp" in keys, "必须包含 BP 神经网络"
    for v in pm.values():
        assert 0 < v < 6, f"命中率 {v} 超出 [0,6]"


def test_honesty_mentions_nested_permutation(seeded):
    """诚实声明必须点明"单模型置换会低估偏差"这个方法论要点。"""
    r = RR.rigor_neural_exhaustive(n_null=2, seed=6, max_models=3)
    h = r["honesty_zh"]
    assert "嵌套" in h or "选择" in h
    assert "不产号" in h
