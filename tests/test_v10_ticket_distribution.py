"""V10-03 票单分布（解析）+ 目标函数重构 的验收测试。

对应 docs/V10_IMPROVEMENT_PLAN.md §3.4（V10-03）与
docs/reviews/R1_ARCHITECTURE_REVIEW.md P0-5、S2_PLAN_FEASIBILITY_REVIEW.md P0-2。

测试原则（本项目传统）：
* **用构造性票组**（全同 / 全不相交 / 固定重叠 / 每号恰 k 次），禁止用随机票组
  去比较两个票组的 var_ratio —— 随机 10 注的 Ā 抽样分布是 1.09±0.13，
  差异会被噪声完全淹没（docs §3.4.2 坑 3）。
* **解析值必须自己复算**：这里用 C(33,6)=1,107,568 全枚举交叉验证闭式，
  不是拿公式反过来验证公式。
* **不做"负值用例"**：var_ratio 的真实下界是 0（n=11 且每个红号恰出现 2 次时
  Var[T]=0），"可为负"是 v2.0 的错误结论（S2 P0-2）。这里改为**下界用例**。
"""
from __future__ import annotations

# V36（2026-10-24 审计 P1-6）：六等奖官方口径常量（独立算，不 import 被测模块）。
# 官方六等奖 = 蓝中 **且** 红 ≤ 2；旧基准 0.15625 = 5/32 隐含"蓝中即中奖"，
# 漏了红球条件（偏高 6.25%）。正确值 = 5×P(红≤2)/32 ≈ 0.1472314。
from math import comb as _comb_v36
P_RED_LE2_V36 = (_comb_v36(6, 0) * _comb_v36(27, 6) + _comb_v36(6, 1) * _comb_v36(27, 5)
                 + _comb_v36(6, 2) * _comb_v36(27, 4)) / _comb_v36(33, 6)
EV_RATIO_SIXTH_V36 = 5.0 * P_RED_LE2_V36 / 32.0

import inspect
import json
import random

import numpy as np
import pytest

from lottery import engine as E

# ---------------------------------------------------------------------------
# 构造工具（全部确定性，不用随机）
# ---------------------------------------------------------------------------

R = 33


def _t(reds, blue=1):
    return {"reds": sorted(int(x) for x in reds), "blue": int(blue)}


def _identical(n=5, reds=(1, 2, 3, 4, 5, 6)):
    return [_t(reds, blue=(i % 16) + 1) for i in range(n)]


def _disjoint(n=5):
    """n 注两两不相交（仅 n ≤ 5 可行：6 注需要 36 个红号 > 33）。"""
    assert n * 6 <= R
    return [_t(range(1 + 6 * i, 7 + 6 * i), blue=i + 1) for i in range(n)]


def _each_number_k_times(n, k):
    """构造 n 注、每个红号恰出现 k 次（6n = 33k 时成立；如 n=11,k=2）。

    轮转分配保证同一注内号码不重复（每个号出现次数 k < n 时，k 个副本会落在
    不同的注上）。用于取 var_ratio 的下界（n=11,k=2 → Var[T]=0）。
    """
    nums = sorted(num for num in range(1, R + 1) for _ in range(k))
    assert len(nums) == 6 * n, f"需要 6n 个号位，实际 {len(nums)}"
    return [_t(nums[i::n], blue=(i % 16) + 1) for i in range(n)]


def _paved(n):
    """「铺开」票组：把 6n 个号位尽量平均摊到 33 个红号上，再轮转成 n 注。

    n=5 → 30 个号各 1 次（并集 30）；n=10 → 27 个号各 2 次 + 6 个号各 1 次
    （共 60 号位、并集 33、Ā=0.6）。轮转保证同一注内不重号。
    """
    slots = 6 * n
    base, extra = divmod(slots, R)
    counts = [base + (1 if i < extra else 0) for i in range(R)]
    nums = sorted(num for num, c in zip(range(1, R + 1), counts) for _ in range(c))
    assert len(nums) == slots
    tickets = [_t(nums[i::n], blue=(i % 16) + 1) for i in range(n)]
    assert all(len(t["reds"]) == 6 for t in tickets), "铺开构造必须保证每注 6 个不重复红号"
    return tickets


def _abar_target_sum_a(n, abar):
    """给定 n 与目标 Ā 反推 ΣA（须为整数，否则构造不存在）。"""
    pairs = n * (n - 1) // 2
    val = abar * pairs
    assert abs(val - round(val)) < 1e-12, f"n={n}, Ā={abar} 不是整数 ΣA"
    return int(round(val))


def _sum_a(tickets):
    reds = [set(t["reds"]) for t in tickets]
    return sum(len(reds[i] & reds[j]) for i in range(len(reds)) for j in range(i + 1, len(reds)))


def _pool(n=120, seed=22):
    rng = random.Random(seed)
    return [{"reds": sorted(rng.sample(range(1, R + 1), 6)),
             "blue": rng.randint(1, 16),
             "confidence": round(rng.random() * 100, 3),
             "method": "stat:freq"} for _ in range(n)]


# ===========================================================================
# 1. 三个解析边界值（±1e-9）—— §3.4.2 的表
# ===========================================================================

def test_five_identical_tickets_var_ratio_is_five():
    """5 注完全相同：ΣA=60 → Ā=6 → var_ratio 恰为 5.000000（±1e-9）。"""
    d = E.ticket_distribution(_identical(5))
    assert d["n"] == 5
    assert d["union_coverage"] == 6
    assert d["redundancy"] == pytest.approx(6.0, abs=1e-9)      # Ā = ΣA/C(5,2) = 60/10
    assert d["redundancy_sum"] == pytest.approx(60.0, abs=1e-9)
    assert d["var_T"] == pytest.approx(18.827479338842974, abs=1e-9)
    assert d["var_ratio"] == pytest.approx(5.0, abs=1e-9)
    # nσ₁² = 5 × 729/968
    assert d["n_sigma1_sq"] == pytest.approx(5 * 729 / 968, abs=1e-12)


def test_five_disjoint_tickets_var_ratio_is_one_ninth():
    """5 注两两不相交：ΣA=0 → Ā=0 → var_ratio = 1+(4)(11/54)(−12/11) = 1/9。"""
    d = E.ticket_distribution(_disjoint(5))
    assert d["union_coverage"] == 30
    assert d["redundancy"] == pytest.approx(0.0, abs=1e-12)
    assert d["var_T"] == pytest.approx(0.41838842975206614, abs=1e-9)
    assert d["var_ratio"] == pytest.approx(1.0 / 9.0, abs=1e-9)


def test_redundancy_neutral_point_gives_var_ratio_one():
    """Ā = 12/11 是**中性点**（var_ratio = 1），不是 0。

    构造：n=11 时 C(11,2)=55，ΣA=60 即 Ā=60/55=12/11。用
    「18 个号各出现 3 次 + 6 个号各出现 2 次」的 66 号位轮转分配实现
    （3x+2y=66、9x+4y+z=186 ⇒ 3x+y=60，取 x=18,y=6）。
    """
    n = 11
    assert _abar_target_sum_a(n, 12 / 11) == 60
    nums = ([num for num in range(1, 19) for _ in range(3)]
            + [num for num in range(19, 25) for _ in range(2)])
    tickets = [_t(nums[i::n], blue=(i % 16) + 1) for i in range(n)]
    assert all(len(t["reds"]) == 6 for t in tickets), "构造必须保证每注 6 个不重复红号"
    assert _sum_a(tickets) == 60
    d = E.ticket_distribution(tickets, with_distribution_exact=False)
    assert d["redundancy"] == pytest.approx(12 / 11, abs=1e-12)
    assert d["redundancy_neutral"] == pytest.approx(12 / 11, abs=1e-12)
    assert d["var_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_var_ratio_lower_bound_is_zero_not_negative():
    """**下界用例**（替代 v2.0 要求的"负值用例"，后者不可构造）。

    n=11 且每个红号恰出现 2 次 → 每注命中数之和恒为定值 → Var[T]=0、var_ratio=0。
    """
    tickets = _each_number_k_times(11, 2)
    assert _sum_a(tickets) == 33          # 每个号出现 2 次 → 33 个 C(2,2)
    d = E.ticket_distribution(tickets)
    assert d["var_T"] == 0.0
    assert d["var_ratio"] == 0.0
    assert d["var_ratio"] >= 0.0


def test_var_ratio_never_negative_over_all_feasible_n():
    """穷举 n=2..33 的**最小可行 Ā**（把 6n 个号位尽量平摊），确认 var_ratio ≥ 0。

    并顺带验证 n=11/22/33 三个"整倍覆盖"点取值恰为 0（不是接近 0）。
    """
    zeros = []
    for n in range(2, R + 1):
        slots = 6 * n
        # 平摊：每个号出现 floor 或 ceil(slots/33) 次
        base, extra = divmod(slots, R)
        counts = [base + (1 if i < extra else 0) for i in range(R)]
        sum_a = sum(c * (c - 1) // 2 for c in counts)
        stats = E._variance_stats(n, sum_a)
        assert stats["var_ratio"] >= 0.0, f"n={n} 出现负 var_ratio {stats['var_ratio']}"
        assert stats["var_T"] >= 0.0
        if stats["var_ratio"] == 0.0:
            zeros.append(n)
    # Ā 能达到"每号同次数"的点：n=11(2次)/22(4次)/33(6次)
    assert zeros == [11, 22, 33], f"下界 0 应恰在 n=11/22/33 取到，实际 {zeros}"


# ===========================================================================
# 2. 闭式 vs C(33,6) 全枚举交叉验证（自己复算，不照抄规格）
# ===========================================================================

def test_analytic_var_T_matches_full_enumeration():
    """Var[T] 的闭式必须与全枚举一致（±1e-9），否则协方差公式写错了。"""
    for tickets in (_identical(5), _disjoint(5), _each_number_k_times(11, 2)):
        d = E.ticket_distribution(tickets, with_distribution_exact=True)
        assert d["var_T_exact_enum"] == pytest.approx(d["var_T"], abs=1e-9), (
            f"闭式 var_T={d['var_T']} 与枚举 {d['var_T_exact_enum']} 不一致")


def test_expected_hits_identity_and_mean_matches_enumeration():
    """E[命中总数] = n × 12/11（与选号无关的恒等式），枚举均值也须吻合。"""
    for tickets in (_identical(5), _disjoint(5)):
        d = E.ticket_distribution(tickets)
        assert d["expected_hits"] == pytest.approx(5 * 12 / 11, abs=1e-12)
        assert d["sigma1_sq"] == pytest.approx(729 / 968, abs=1e-15)


def test_single_ticket_tail_probabilities_match_hypergeometric():
    """单注 P(K≥r) 与超几何解析值一致（r=3/4/5）。"""
    d = E.ticket_distribution([_t((1, 2, 3, 4, 5, 6))])
    assert d["p_at_least_r"]["3"] == pytest.approx(0.05771925516085694, abs=1e-12)
    assert d["p_at_least_r"]["4"] == pytest.approx(0.004900827759559684, abs=1e-12)
    assert d["p_at_least_r"]["5"] == pytest.approx(0.00014716929344293082, abs=1e-12)
    assert d["p_at_least_r_method"] == "exact_enum"


def test_identical_tickets_are_perfectly_correlated_in_tail_probability():
    """5 注全同时「至少一注命中≥4红」= 单注概率（完全相关），不是 iid 的 5 倍空间。"""
    one = E.ticket_distribution([_t((1, 2, 3, 4, 5, 6))])
    five = E.ticket_distribution(_identical(5))
    assert five["p_at_least_r"]["4"] == pytest.approx(one["p_at_least_r"]["4"], abs=1e-12)
    # 对照：独立假设（泊松二项）给的值明显偏高 —— 这就是为什么精确值必须枚举
    assert five["p_at_least_r_iid"]["4"] > 4 * one["p_at_least_r"]["4"]
    # 相关性误差 > 100%，远不是"可忽略近似"
    rel = (five["p_at_least_r_iid"]["4"] - five["p_at_least_r"]["4"]) / five["p_at_least_r"]["4"]
    assert rel > 1.0


def test_paving_beats_iid_only_for_small_n_and_collapses_for_large_n():
    """覆盖铺开**不是**"总能提高 P(≥4红)"：n=5 略升，n=10 反而暴跌（票间负相关）。

    这条是 pmax_high **不能**用「最小化冗余度」当代理目标的直接反例（§3.4.4）：
    n=10 的"每号 2 次"票组 Ā=0.6（var_ratio 仅 0.1，方差被压到极小），
    结果「至少一注中 4 红」的概率比独立假设低 ~44% —— 结构与概率方向相反。
    """
    d5 = E.ticket_distribution(_paved(5), r_values=(4,))
    assert d5["redundancy"] == pytest.approx(0.0, abs=1e-12)          # 5 注不相交
    assert d5["p_at_least_r"]["4"] > d5["p_at_least_r_iid"]["4"]      # 铺开略升（+1%）

    d10 = E.ticket_distribution(_paved(10), r_values=(4,))
    assert d10["union_coverage"] == 33
    assert d10["redundancy"] == pytest.approx(0.6, abs=1e-12)
    assert d10["var_ratio"] == pytest.approx(0.1, abs=1e-12)          # 方差被压到 sqrt(0.1)
    rel = (d10["p_at_least_r_iid"]["4"] - d10["p_at_least_r"]["4"]) / d10["p_at_least_r"]["4"]
    assert rel > 0.30, f"10 注铺开应明显低于 iid 近似，实际只差 {rel:.1%}"


# ===========================================================================
# 3. 蒙特卡洛 vs 解析（固定种子，误差 < 0.2%）
# ===========================================================================

@pytest.mark.parametrize("tickets,r", [
    (_disjoint(5), 3),
    (_disjoint(5), 4),
    (_identical(5), 4),
    (_pool(6, seed=5), 4),
])
def test_monte_carlo_matches_exact_enumeration(tickets, r):
    """MC（固定种子 200k 次）与全枚举之差 < 0.2%（绝对，即 0.2 个百分点）。

    说明：p(≥1≥4红) 量级约 2.5%，其**相对**误差要到 0.2% 需要 ~1e7 次采样，
    单次 MC 做不到；规格的 0.2% 按"百分点"口径执行，并在报文里同时给出
    stderr 与相对误差，避免把统计噪声当成偏差。
    """
    exact = E.ticket_distribution(tickets, r_values=(r,))["p_at_least_r"][str(r)]
    mc = E.p_at_least_r_mc(tickets, r=r, n_draws=200_000, seed=20261017)
    assert mc["stderr"] < 0.002
    assert abs(mc["p"] - exact) < 0.002, (
        f"MC {mc['p']:.6f} 与枚举 {exact:.6f} 相差 {abs(mc['p'] - exact):.6f}"
        f"（stderr={mc['stderr']:.6f}，相对 {abs(mc['p'] - exact) / exact:.2%}）")


def test_monte_carlo_is_reproducible_with_fixed_seed():
    """固定种子必须完全可复现（否则"交叉校验"不可复核）。"""
    tickets = _disjoint(5)
    a = E.p_at_least_r_mc(tickets, r=4, n_draws=20_000, seed=7)
    b = E.p_at_least_r_mc(tickets, r=4, n_draws=20_000, seed=7)
    assert a["p"] == b["p"]
    assert a["n_draws"] == 20_000 and a["r"] == 4


# ===========================================================================
# 4. 蓝球覆盖 / 六等奖 EV / 整注 EV 口径（F8）
# ===========================================================================

def test_sixteen_distinct_blues_gives_certain_blue_hit_and_sixth_ev_ratio():
    """16 注覆盖全部蓝球：p_any_blue == 1.0；ev_ratio_sixth 恒为 5×P(红≤2)/32 ≈ 0.1472314。"""
    tickets = [_t((1, 2, 3, 4, 5, 6), blue=b) for b in range(1, 17)]
    d = E.ticket_distribution(tickets, with_distribution_exact=False)
    assert d["p_any_blue"] == 1.0
    assert d["blue_coverage_count"] == 16
    assert d["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)
    # 六等奖 EV 比与注数无关：1 注与 16 注都是 0.1472314
    one = E.ticket_distribution([_t((1, 2, 3, 4, 5, 6))], with_distribution_exact=False)
    assert one["ev_ratio_sixth"] == pytest.approx(d["ev_ratio_sixth"], abs=1e-15)


def test_ev_ratio_full_is_not_the_sixth_prize_number_and_carries_assumption():
    """F8：ev_ratio_sixth(0.1472314) 与 ev_ratio_full(≈0.4567) 是两个口径，必须并排给出。

    禁止裸 ``ev_ratio``：只给 0.1472314 会让用户以为整体返奖率只有 15.6%。
    """
    d = E.ticket_distribution([_t((1, 2, 3, 4, 5, 6))], with_distribution_exact=False)
    assert d["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)
    assert d["ev_ratio_full"] == pytest.approx(0.45673239701760976, abs=1e-9)
    assert d["ev_ratio_full"] > 2.5 * d["ev_ratio_sixth"]
    assert isinstance(d["prize_assumption"], str) and "一等" in d["prize_assumption"]
    # 严格互斥奖级口径（与 evaluate.prize_level 一致）也要给出，供口径对照
    assert 0.40 < d["ev_ratio_full_strict"] < 0.46
    # D-15：原此处写 `assert "ev_ratio" not in d or True` —— 恒真，是废断言。
    # 「裸 ev_ratio 不得作为字段」这条契约已由下一条
    # `test_no_bare_ev_ratio_field_in_ticket_distribution` 真测，此处不重复，只留指针。
    # （该测试名为"静态契约"但实际是运行时检查，命名不准，已在 D-15 记录。）


def test_ev_ratio_full_changes_with_jackpot_assumption():
    """换奖金假设表必须改变 ev_ratio_full（防止被实现成硬编码常数）。"""
    t = [_t((1, 2, 3, 4, 5, 6))]
    a = E.ticket_distribution(t, with_distribution_exact=False)
    b = E.ticket_distribution(t, with_distribution_exact=False,
                              jackpot={"first": 10_000_000.0, "second": 300_000.0,
                                       "label": "测试假设表"})
    assert b["ev_ratio_full"] > a["ev_ratio_full"]
    assert "测试假设表" in b["prize_assumption"]
    # 表 B（engine.JACKPOT_ASSUMPTION）必须与表 A 不同
    assert a["ev_ratio_full_table_b"] != pytest.approx(a["ev_ratio_full"], abs=1e-6)


def test_no_bare_ev_ratio_field_in_ticket_distribution():
    """静态契约：本函数返回体里不得出现裸 ``ev_ratio``（必须带 _sixth/_full 后缀）。"""
    d = E.ticket_distribution(_identical(3), with_distribution_exact=False)
    assert "ev_ratio" not in d
    assert "ev_ratio_sixth" in d and "ev_ratio_full" in d


# ===========================================================================
# 5. 入参校验 / JSON 可序列化 / 原生类型
# ===========================================================================

@pytest.mark.parametrize("bad", [
    [{"reds": [1, 2, 3, 4, 5]}],                       # 只有 5 个红号
    [{"reds": [1, 2, 3, 4, 5, 5]}],                    # 红号重复
    [{"reds": [1, 2, 3, 4, 5, 34]}],                   # 越界
    [{"reds": [0, 1, 2, 3, 4, 5]}],                    # 越界
    [{"reds": None}],                                  # 非列表
])
def test_invalid_tickets_raise_value_error(bad):
    with pytest.raises(ValueError):
        E.ticket_distribution(bad, with_distribution_exact=False)


def test_distribution_is_json_serializable_and_uses_native_types():
    """返回值必须是原生 float/int（不能混入 numpy 标量），前端 JSON 才能直接吃。"""
    d = E.ticket_distribution(_disjoint(5))
    json.dumps(d, ensure_ascii=False)                   # 不抛异常即通过
    for key, val in d.items():
        if isinstance(val, dict):
            for v in val.values():
                assert isinstance(v, (int, float, str, bool, type(None))), f"{key} 含非原生值 {v!r}"
        else:
            assert isinstance(val, (int, float, str, bool, type(None))), f"{key} 是 {type(val)}"
    # numpy 标量不是 float 的实例（np.float64 是，但 np.float32/np.int64 不是）
    assert not isinstance(d["var_ratio"], np.ndarray)
    assert type(d["var_ratio"]) is float
    assert type(d["n"]) is int


def test_empty_tickets_is_safe():
    d = E.ticket_distribution([], with_distribution_exact=False)
    assert d["n"] == 0
    assert d["var_T"] == 0.0 and d["var_ratio"] == 0.0
    assert d["p_at_least_r"] == {"3": 0.0, "4": 0.0, "5": 0.0}
    assert d["p_any_blue"] == 0.0


# ===========================================================================
# 6. 目标函数：4 种模式语义 + 蓝球职责边界（零梯度修复）
# ===========================================================================

def test_objective_names_normalize_via_alias_table():
    """别名表：expected_value/ev → ev_conserving；未知值回退诚实默认（不抛异常）。"""
    assert E._normalize_coverage_objective("expected_value") == "ev_conserving"
    assert E._normalize_coverage_objective("ev") == "ev_conserving"
    assert E._normalize_coverage_objective("EV_CONSERVING") == "ev_conserving"
    assert E._normalize_coverage_objective("coverage") == "coverage"
    assert E._normalize_coverage_objective("pmax_high") == "pmax_high"
    assert E._normalize_coverage_objective("custom_weight") == "custom_weight"
    assert E._normalize_coverage_objective("nonsense") == "ev_conserving"
    assert E._normalize_coverage_objective(None) == "ev_conserving"


def test_no_objective_contains_an_ev_term():
    """任何模式都不许把目标写成「提高期望」：_ticket_objective **代码**里不得有 EV 项。

    ## D-18a：为什么不能再用 `in inspect.getsource(...)`

    `getsource` **包含 docstring**，于是两个方向都会错：

    * **假红**（已实证）：只在 `_ticket_objective` 的 docstring 里补一句
      「本目标函数不使用 portfolio_expected_value」——
      **代码一字未动，断言就红了**，逼人去删注释而不是改代码。
    * **假绿**：若有人把真实引用删掉、只在 docstring 里留名，断言仍通过。

    改用 `code_text()`：先剥 docstring（`ast`），再 `ast.unparse` 成纯代码文本。
    注释与 docstring 都不进结果，标识符文本形态不变（已实测）。
    """
    from _source_assert import code_text
    code = code_text(E._ticket_objective)
    assert "portfolio_expected_value" not in code
    assert "expected_reward" not in code
    meta = E._objective_meta("ev_conserving", 4, blue_owned_by_plan=True)
    assert meta["optimizes_ev"] is False
    meta2 = E._objective_meta("pmax_high", 4, blue_owned_by_plan=True)
    assert meta2["optimizes_ev"] is False


def test_dead_code_removed_from_objective():
    """§3.4.1 死代码：``cover_bonus * n / max(1, n)``（恒等于 cover_bonus）必须消失。

    ## D-18a：改判据 + 反向误报已实证

    原写法 `assert "max(1, n)" not in inspect.getsource(...)` 有一个**真实的误报**：

        只往 `_ticket_objective` 的 docstring 加一行
        「已删除死代码 cover_bonus * n / max(1, n)」
        → 代码逻辑一字未动 → 本测试变红（已实测复现）

    这对断言正是「注释解释历史」最容易触碰的地方，所以必须剥 docstring。

    修后仍保留**文本匹配**（不走 AST 结构）—— 理由：
    `max(1, n)` 这个**字面形态**才是死代码的特征；换成 AST 结构判断会漂移成
    「不存在这样的调用」，且变量改名（`cover_bonus` → `cb`）后文本仍能抓到，
    AST 结构反而抓不到。`ast.unparse` 会保留 `max(1, n)` 的空格形态（已实测）。
    """
    from _source_assert import code_text
    code = code_text(E._ticket_objective)
    assert "max(1, n)" not in code
    assert "cover_bonus" not in code


def test_ev_conserving_objective_is_blind_to_blue_when_plan_owns_blue():
    """**零梯度修复的核心断言**：蓝球被 plan 接管时，目标必须对蓝球完全不敏感。

    旧实现：选完红球后 `out[i]["blue"] = plan[i]` 强制改写蓝球，而目标里仍有
    distinct_blue 项 → 优化器为一个马上被覆盖的属性爬坡。
    """
    same_blue = [_t((1, 2, 3, 4, 5, 6), blue=1), _t((7, 8, 9, 10, 11, 12), blue=1)]
    diff_blue = [_t((1, 2, 3, 4, 5, 6), blue=1), _t((7, 8, 9, 10, 11, 12), blue=2)]
    for obj in ("ev_conserving", "coverage", "pmax_high"):
        owned_a = E._ticket_objective(same_blue, obj, 4, blue_owned_by_plan=True)
        owned_b = E._ticket_objective(diff_blue, obj, 4, blue_owned_by_plan=True)
        assert owned_a == owned_b, f"{obj}: plan 接管蓝球后目标仍随蓝球变化（零梯度未修）"
    # 蓝球未被接管时，覆盖数是真实自由度 → 目标必须能分辨
    free_a = E._ticket_objective(same_blue, "ev_conserving", 4, blue_owned_by_plan=False)
    free_b = E._ticket_objective(diff_blue, "ev_conserving", 4, blue_owned_by_plan=False)
    assert free_b > free_a


def test_ev_conserving_keeps_expected_reward_identical():
    """EV 守恒：无论优化器怎么选，n 注的期望奖金恒为 n × 单注 EV（逐位相等）。"""
    pool = _pool(80, seed=13)
    for obj in ("ev_conserving", "coverage", "pmax_high"):
        picked = E.coverage_optimize(pool, 6, r=4, trials=120, rng=random.Random(4),
                                     objective=obj)
        assert len(picked) == 6
        ev = E.portfolio_expected_value(picked)["expected_reward"]
        one = E.portfolio_expected_value(pool[:1])["expected_reward"]
        assert ev == pytest.approx(one * 6, abs=1e-12), f"{obj}: 期望不再与票单结构无关"


def test_pmax_high_actually_maximizes_the_true_probability():
    """pmax_high 必须真的最大化 P(≥1注≥4红)，而不是拿冗余度/覆盖度当代理。

    构造一个有区分度的池：一半是同一注的副本（完全相关、无增益），
    一半是互不相交的票（n=5 时铺开确实更高）。基线 = 按置信度取前 n 注。
    """
    copies = [dict(_t((1, 2, 3, 4, 5, 6), blue=(i % 16) + 1), confidence=99.0 - i)
              for i in range(20)]
    spread = [dict(t, confidence=10.0 + i) for i, t in enumerate(_disjoint(5))]
    pool = copies + spread
    n = 5
    rng = random.Random(11)
    picked = E.coverage_optimize(pool, n, r=4, trials=800, rng=rng, objective="pmax_high")
    p_opt = E.ticket_distribution(picked, r_values=(4,))["p_at_least_r"]["4"]
    baseline = sorted(pool, key=lambda x: -x["confidence"])[:n]
    p_base = E.ticket_distribution(baseline, r_values=(4,))["p_at_least_r"]["4"]
    assert p_opt > p_base, f"pmax_high 未提高真实概率：{p_opt} vs 基线 {p_base}"
    # 对照：ev_conserving 也偏向铺开（结构目标），但不承诺概率最优
    picked_ev = E.coverage_optimize(pool, n, r=4, trials=800, rng=random.Random(11),
                                    objective="ev_conserving")
    assert E.ticket_distribution(picked_ev, r_values=(4,))["p_at_least_r"]["4"] > p_base


def test_pmax_high_beats_structure_proxies_on_discriminating_pool():
    """区分性反例：**结构目标与真实概率目标给出相反的选择**。

    池子 = A（10 注铺开：并集 33、Ā=0.6、P(≥4红)=0.0333）+ B（10 注随机：并集 27、
    Ā≈1.28、P(≥4红)=0.0472）。铺开"看起来覆盖更全"，但两两正相关更强 → 真概率更低。
      * ``ev_conserving`` / ``coverage``（结构目标）→ 全选 A（并集 33，p4 最低）；
      * ``pmax_high``（真目标）→ 混选，p4 升到 0.048 以上，**高于两者各自的 p4**。
    这条证明 pmax_high 没有偷懒用「最小化冗余度/最大化并集」当代理（§3.4.4）。
    """
    a_set = _paved(10)
    rng = random.Random(2024)
    b_set = [_t(sorted(rng.sample(range(1, R + 1), 6)), blue=(i % 16) + 1) for i in range(10)]
    pool = [dict(t, confidence=50.0) for t in a_set + b_set]

    def _p4(tickets):
        return E.ticket_distribution(tickets, r_values=(4,))["p_at_least_r"]["4"]

    p4_a, p4_b = _p4(a_set), _p4(b_set)
    assert p4_a < p4_b, "构造前提：铺开票组 A 的真实概率必须低于随机票组 B"

    picked_struct = E.coverage_optimize(pool, 10, r=4, trials=400, rng=random.Random(7),
                                        objective="ev_conserving")
    picked_pmax = E.coverage_optimize(pool, 10, r=4, trials=400, rng=random.Random(7),
                                      objective="pmax_high")
    p4_struct, p4_pmax = _p4(picked_struct), _p4(picked_pmax)
    assert p4_struct == pytest.approx(p4_a, abs=1e-9), "结构目标应整组选 A（并集最大）"
    assert p4_pmax > max(p4_a, p4_b), (
        f"pmax_high 必须超过两侧上限：{p4_pmax} vs A={p4_a} / B={p4_b}")
    assert p4_pmax > 1.4 * p4_struct


def test_pmax_high_reports_var_ratio_cost():
    """pmax_high 必须同时给出方差代价（var_ratio），否则用户看不到"铺开"的另一面。"""
    pool = _pool(40, seed=3)
    cmp = E.coverage_optimize_compare(pool, 5, r=4, trials=80, rng=random.Random(2),
                                      objective="pmax_high")
    meta = cmp["objective_meta"]
    assert cmp["objective_effective"] == "pmax_high"
    assert cmp["objective"] == "pmax_high"
    assert isinstance(meta["var_ratio_after"], float)
    assert isinstance(meta["var_ratio_before"], float)
    assert "p_at_least_4_after" in meta and "p_at_least_4_before" in meta
    assert meta["p_at_least_r_method"] == "exact_enum"


def test_custom_weight_equals_pmax_high_when_weights_are_degenerate():
    """custom_weight 的语义：权重全给 pmax_high 时必须与 pmax_high 选择完全一致。"""
    pool = _pool(60, seed=17)
    w = {"pmax_high": 1.0, "union": 0.0, "neg_redundancy": 0.0, "blue_coverage": 0.0}
    a = E.coverage_optimize(pool, 6, r=4, trials=150, rng=random.Random(9),
                            objective="pmax_high")
    b = E.coverage_optimize(pool, 6, r=4, trials=150, rng=random.Random(9),
                            objective="custom_weight", objective_weights=w)
    assert [(tuple(t["reds"]), t["blue"]) for t in a] == [(tuple(t["reds"]), t["blue"]) for t in b]


def test_custom_weight_weights_merge_order_and_validation():
    """权重合并：默认 ← config 覆盖 ← 调用参数；未知键与非法值被安全忽略。"""
    base = E._custom_objective_weights(None)
    assert base == E.DEFAULT_CUSTOM_WEIGHTS or set(base) == set(E.DEFAULT_CUSTOM_WEIGHTS)
    merged = E._custom_objective_weights({"union": 2.0, "nope": 9.0, "pmax_high": "bad"})
    assert merged["union"] == 2.0
    assert "nope" not in merged
    assert merged["pmax_high"] == E.DEFAULT_CUSTOM_WEIGHTS["pmax_high"]


def test_coverage_optimize_signature_and_contract_unchanged():
    """公开签名/契约不变：返回 n 注、去重、给定 blue_probs 时蓝球全覆盖且标记 planned。"""
    pool = _pool(80, seed=31)
    picked = E.coverage_optimize(pool, 10, r=3, n_blues=4, trials=50,
                                 rng=random.Random(1), blue_probs=np.full(16, 1 / 16))
    assert len(picked) == 10
    assert len({(tuple(t["reds"]), t["blue"]) for t in picked}) == 10
    assert len({t["blue"] for t in picked}) == 10
    assert all(t.get("blue_planned") for t in picked)
    # 负注数/空池安全
    assert E.coverage_optimize(pool, 0) == []
    assert E.coverage_optimize([], 3) == []


def test_coverage_optimize_compare_reports_gradient_and_distribution():
    """compare 必须如实回报：蓝球零梯度标志、var_ratio、精确尾部概率。"""
    pool = _pool(60, seed=41)
    cmp = E.coverage_optimize_compare(pool, 6, trials=60, rng=random.Random(3),
                                      objective="expected_value",
                                      blue_probs=np.full(16, 1 / 16))
    assert cmp["objective"] == "expected_value"          # 调用方原值（V4 兼容）
    assert cmp["objective_effective"] == "ev_conserving"  # 规范化值
    meta = cmp["objective_meta"]
    assert meta["blue_owned_by_plan"] is True
    assert meta["gradient_blue_degenerate"] is True
    assert meta["optimizes_ev"] is False
    assert meta["var_ratio_after"] >= 0.0
    # 未传 blue_probs 时蓝球不是 plan 的职责 → 梯度不退化为零
    cmp2 = E.coverage_optimize_compare(pool, 6, trials=30, rng=random.Random(3),
                                       objective="ev_conserving")
    assert cmp2["objective_meta"]["gradient_blue_degenerate"] is False


def test_predict_next_coverage_metadata_includes_distribution():
    """端到端：coverage 模式把票单分布元数据带进 predict_next（热路径用 iid 近似并标注）。"""
    from lottery import engine as E2

    draws = [{"issue": f"20{2013 + i // 150}{i % 150 + 1:03d}",
              "date": f"20{2013 + i // 150}-01-01",
              "reds": sorted(random.Random(i).sample(range(1, 34), 6)),
              "blue": random.Random(i + 999).randint(1, 16)} for i in range(600)]
    res = E2.predict_next(draws, use_llm=False, use_ml=False, n_tickets=6,
                          persist=False, rng=random.Random(5), coverage=True)
    dist = res["coverage_distribution"]
    assert isinstance(dist, dict)
    assert dist["n"] == len(res["tickets"])
    assert dist["p_at_least_r_method"] == "iid_approx"     # 热路径：近似且已标注
    assert dist["var_ratio"] >= 0.0
    assert dist["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)
    assert dist["p_any_blue"] == pytest.approx(res["blue_covered"] / 16.0, abs=1e-9)
    # 非 coverage 模式不产出该元数据（保持 V4 结果契约不变）
    res2 = E2.predict_next(draws, use_llm=False, use_ml=False, n_tickets=4,
                           persist=False, rng=random.Random(6))
    assert res2["coverage_distribution"] is None
