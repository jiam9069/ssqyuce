"""V40 覆盖率欠债 A 类：**有真实计算逻辑**的无直测模块（真值锁）。

## 这个文件为什么存在

V40 覆盖率审计实测发现：`lottery/` 下有 23 个模块**没有任何测试文件直接 import**
（见 `docs/reviews/DEBT_COVERAGE_REPORT.md` 的实测表）。其中一批是**纯数学/决策**
模块 —— 它们算错了不会有任何测试变红。

本文件只覆盖"错了会悄悄给错数"的那部分：

| 模块 | 本文件覆盖的函数 |
|---|---|
| `engine_blue` | `blue_coverage_plan`（降序取号 / 边界形状 / >16 注补号） |
| `engine_solver` | `_balanced_counts` / `_balanced_min_sum_a` / `_balanced_portfolio` / `_portfolio_plan` / `_exact_p_at_least_r` |
| `engine_orchestrate` | `_ticket_objective` / `_normalize_coverage_objective` / `_custom_objective_weights` / `_PmaxEstimator` |
| `engine_rules` | `apply_rules_to_candidates`（off / shadow / active 三模式） |
| `engine_constraints` | `constraint_ctx` / `pass_constraints`（五维 + 向后兼容） |
| `jsonutil` | `jsonable`（numpy 标量 / 键转换 / 深度兜底） |
| `diagnose` | `structure_profile` / `find_similar_draws` / `diagnose` |
| `llm_config_store` | `save` / `read`（部分更新、旧键保留、模型列表替换） |

## 真值从哪来（铁律：外部推导，不抄公式）

每条真值锁的注释里写明**独立推导过程**，不是把被测公式重写一遍：

* 蓝球覆盖 `min(n,16)/16` —— 组合论证：n 注取 n 个不同蓝号，命中概率 = 覆盖数/16。
  随机重复的对照值 `1-(15/16)^n` 是独立的伯努利补事件计算，两者来源不同。
* `ΣA* = Σ_x C(cnt_x,2)` 的均摊值 —— 交换论证得出，再用 `divmod(6n,33)` 手算数列。
* `p_ge4 = 0.004901` —— **超几何分布** `C(6,4)C(27,2)/C(33,6)`，与被测的位运算
  暴力枚举完全无关；两注不相交时 `2×0.004901` 是容斥能算清的等式。
* `AC = 10` —— AC 值定义（不同差值个数 − (k−1)），手数列出 15 个差值。
* 结构画像各字段（和值 84 / 三区 (3,2,1) / 跨度 27 / 质数 2 个）全部手算。

## 变异验证（本文件每条真值锁都验过）

见 `docs/reviews/DEBT_COVERAGE_REPORT.md` 的「变异验证记录」表。
要点：**每条断言都有对应的变异体让它变红**；只在 `git worktree` 里改坏源码，
共享工作区从未被污染。
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ===========================================================================
# ① engine_blue.blue_coverage_plan —— 全项目唯一「真实可得的改进」
# ===========================================================================

def test_blue_coverage_plan_n_at_most_16_yields_distinct_blues():
    """n ≤ 16：计划必须**互不相同**，且「至少中一个蓝球」的概率 = min(n,16)/16。

    真值来源（组合论证，不是抄实现）：
    若 n 注的蓝球两两不同，则「至少中一次蓝球」⇔ 开奖蓝球 ∈ 这 n 个号，
    而开奖蓝球在 16 个号上均匀 ⇒ P = n/16。
    对照组「随机重复」（每注独立抽，可能撞号）：P = 1 − (15/16)^n。
    两者的差正是覆盖优化的全部价值，且它在 n=1 时为 0（单注无优化空间）。
    """
    from lottery.engine_blue import blue_coverage_plan

    expected = {1: 0.0625, 2: 0.125, 3: 0.1875, 5: 0.3125, 10: 0.625, 16: 1.0}
    for n, want in expected.items():
        plan = blue_coverage_plan(n)
        assert len(plan) == n, f"n={n} 应产出 {n} 注"
        assert len(set(plan)) == n, f"n={n} 的蓝球必须两两不同，实际 {plan}"
        assert all(1 <= b <= 16 for b in plan), f"n={n} 出现越界蓝号 {plan}"

        p_covered = len(set(plan)) / 16
        p_repeat = 1 - (15 / 16) ** n
        assert p_covered == pytest.approx(want), f"n={n} 覆盖概率算错"
        assert p_covered >= p_repeat - 1e-12
    # n=1 是唯一「覆盖无收益」的点：两条路径概率相同
    assert 1 / 16 == pytest.approx(1 - (15 / 16) ** 1)


def test_blue_coverage_plan_orders_by_probability_descending():
    """概率入参：**降序取号**，因此既是「最多不同号」也是「概率最高的一组」。

    三元组 (5, 10, 1) 是我手写的：把三个峰放在 5/10/1 号，其余为 0。
    降序前三只能是 5、10、1（概率 0.9 > 0.5 > 0.1）。
    """
    from lottery.engine_blue import blue_coverage_plan

    probs = np.zeros(16)
    probs[4] = 0.9    # 蓝 5
    probs[9] = 0.5    # 蓝 10
    probs[0] = 0.1    # 蓝 1
    assert blue_coverage_plan(3, probs) == [5, 10, 1]
    # 取 5 注：前 3 个仍是峰，后 2 个是 0 概率号里**下标最小**的两个（2、3 号）
    plan5 = blue_coverage_plan(5, probs)
    assert plan5[:3] == [5, 10, 1]
    assert plan5[3:] == [2, 3], f"0 概率区没有按稳定序取号：{plan5}"


@pytest.mark.parametrize("bad_probs,expected_plan", [
    # 含 NaN：形状合法、和为 NaN（比较恒 False）⇒ 只有 isfinite 守卫能拦住它。
    # 不回退时 argsort(-nan) 产出 [1,3,4,5]（实测）⇒ 断言 [1,2,3,4] 即钉住回退。
    (np.array([1.0, float("nan")] + [0.0] * 14), [1, 2, 3, 4]),
])
def test_blue_coverage_plan_falls_back_to_uniform_on_bad_probs(bad_probs, expected_plan):
    """坏入参一律回退均匀权重 → 计划退化为 1,2,3,4（而不是抛异常或给出空计划）。

    这是**边界必测**：调用方（`predict_next`）在 numpy 形状漂移时不该整体 500。

    ⚠️ 变异验证记录（三条教训，全部实测）：
      ① 第一版用 NaN×16 / 全负 / 全零当"坏输入"，变异对照发现「删掉
         isfinite/sum<=0 守卫」是**不可观察变异** —— np.argsort 对这些退化
         输入恰好仍产出 [1,2,3,4]，断言不红 ⇒ 假绿，重写。
      ② 第二版把 [3,0,1,0…] 当坏输入 —— 但它**根本不是坏输入**（和 4>0、
         有限），回退反而错了；变异"删守卫"当然不红。这说明"什么算坏"
         必须与实现契约对齐：形状错 / 类型错 / 含非有限值 / 和 ≤ 0。
      ③ 最终版用**含单个 NaN 的合法形状**：只有 isfinite 守卫拦得住，
         删守卫必红（变异实测 1 failed）。全零/全负分支仍由实现兜底，
         但对排序结果不可观察，故不再在此断言。
    """
    from lottery.engine_blue import blue_coverage_plan

    plan = blue_coverage_plan(4, bad_probs)
    assert plan == expected_plan, (
        f"含 NaN 的 probs 未回退均匀（得到 {plan}；若不回退应为 [1,3,4,5]）")


def test_blue_coverage_plan_zero_and_negative_n_are_empty():
    """n ≤ 0 → 空计划（不是抛错、也不是 16 个号）。"""
    from lottery.engine_blue import blue_coverage_plan

    assert blue_coverage_plan(0) == []
    assert blue_coverage_plan(-5) == []


def test_blue_coverage_plan_over_16_fills_least_used_first():
    """n > 16：蓝球只有 16 个号，超出部分**优先补当前出现次数最少**的号。

    外部推导：先铺 16 个不同号（各 1 次），再补 n−16 个；
    每补一个都选「出现次数最少」的 ⇒ 前 n−16 个号各再得 1 次。
    n=20 ⇒ 号 1..4 各 2 次、号 5..16 各 1 次（总计 4·2 + 12·1 = 20）。
    """
    from collections import Counter

    from lottery.engine_blue import blue_coverage_plan

    plan = blue_coverage_plan(20)
    assert len(plan) == 20
    counts = Counter(plan)
    assert [counts[b] for b in (1, 2, 3, 4)] == [2, 2, 2, 2], f"补号不均：{counts}"
    assert [counts[b] for b in (5, 16)] == [1, 1]
    assert sorted(counts) == list(range(1, 17)), "16 个号必须全部出现"
    # 至少一个蓝球的中奖概率饱和在 100%，且成本翻倍（诚实边界，不是 bug）
    assert len(set(plan)) / 16 == 1.0


# ===========================================================================
# ② engine_solver —— 号位均摊构造（ΣA 全局最优）
# ===========================================================================

def test_balanced_counts_divmod_hand_computation():
    """`6n` 个号位在 33 个红号上均摊：次数只能是 floor/ceil(6n/33)，差 ≤ 1。

    外部真值（`divmod` 手算）：
      n=1  → 6n=6   ⇒ 6 个号各 1 次、27 个号 0 次
      n=6  → 36=33+3  ⇒ 3 个号 2 次、30 个号 1 次
      n=11 → 66=33·2 整除 ⇒ **每个号恰好 2 次**
      n=16 → 96=33·2+30 ⇒ 30 个号 3 次、3 个号 2 次
    总和必须恰好等于 6n（红球总号位守恒）。
    """
    from lottery.engine_solver import _balanced_counts

    for n, expect in [
        (1, {"zero": 27, "one": 6, "two": 0, "three": 0}),
        (6, {"zero": 0, "one": 30, "two": 3, "three": 0}),
        (11, {"zero": 0, "one": 0, "two": 33, "three": 0}),
        (16, {"zero": 0, "one": 0, "two": 3, "three": 30}),
    ]:
        counts = _balanced_counts(n)
        assert len(counts) == 33, f"n={n} 必须给出 33 个红号的次数"
        assert sum(counts) == 6 * n, f"n={n} 号位总数 {sum(counts)} != {6 * n}"
        got = {"zero": counts.count(0), "one": counts.count(1),
               "two": counts.count(2), "three": counts.count(3)}
        assert got == expect, f"n={n} 次数分布 {got} != 手算 {expect}"
        assert max(counts) - min(counts) <= 1, "均摊性质被破坏（差 > 1）"


def test_balanced_min_sum_a_matches_hand_table():
    """ΣA* = Σ_x C(cnt_x, 2) 的**全局最优值**（交换论证，不是实现复述）。

    交换论证：若某号出现 cnt_a ≥ cnt_b + 2 次，把 a 的一次出现换成 b，
    ΣA 的变化量 = −(cnt_a − 1) + cnt_b < 0 ⇒ 最优必为均摊。
    手算表（独立于实现）：n≤5 → 0；n=6 → 3；n=11 → 33；n=16 → 93。
    """
    from lottery.engine_solver import _balanced_min_sum_a

    hand = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 3, 7: 9, 8: 15, 9: 21, 10: 27,
            11: 33, 12: 45, 13: 57, 14: 69, 15: 81, 16: 93}
    for n, expected in hand.items():
        assert _balanced_min_sum_a(n) == expected, f"n={n} 的 ΣA* 手算为 {expected}"


def test_balanced_portfolio_is_disjoint_up_to_5_and_valid_beyond():
    """n ≤ 5：号位均摊 ⇒ **两两互不相交**（30 个号正好铺满 5 注）。

    外部推导：5 注需要 30 个不同号，而 6n=30 ≤ 33 ⇒ 均摊次数为 0/1 ⇒ 不相交。
    n ≥ 6 时不可能不相交（36 > 33），此时要求形态合法且每号次数等于均摊上限。
    """
    from lottery.engine_solver import _balanced_min_sum_a, _balanced_portfolio

    for n in (1, 2, 3, 4, 5):
        tickets = _balanced_portfolio(n, seed=0)
        assert len(tickets) == n
        assert all(len(t) == 6 for t in tickets), f"n={n} 存在非 6 长注：{tickets}"
        flat = [x for t in tickets for x in t]
        assert len(flat) == len(set(flat)) == 6 * n, f"n={n} 应两两不相交"
        assert all(1 <= x <= 33 for x in flat)
        assert all(t == sorted(t) for t in tickets), "每注必须升序"
        assert _balanced_min_sum_a(n) == 0

    for n in (6, 11):
        tickets = _balanced_portfolio(n, seed=3)
        assert all(len(t) == 6 for t in tickets)
        assert all(len(set(t)) == 6 for t in tickets), "注内不得重号"
        flat = [x for t in tickets for x in t]
        assert max(flat.count(x) for x in set(flat)) == math.ceil(6 * n / 33), (
            f"n={n} 的每号出现次数不等于均摊上限")


def test_portfolio_plan_blues_are_distinct_and_presets_differ():
    """`_portfolio_plan` 的两种 preset：蓝球 1..n 互不相同；spread 不相交、overlap 共享 4 号。

    外部推导：函数约定蓝球取 1..n ⇒ n ≤ 16 时「至少一蓝」概率取闭式上限 n/16。
    overlap 的语义是「4 个共有号 + 每注 2 个随机号」⇒ 两注交集恰为 4。
    """
    from lottery.engine_solver import _portfolio_plan

    for n in (1, 3, 16):
        plan = _portfolio_plan(n, "spread", seed=0)
        assert [t["blue"] for t in plan] == list(range(1, n + 1))

    plan5 = _portfolio_plan(5, "spread", seed=0)
    flat = [x for t in plan5 for x in t["reds"]]
    assert len(set(flat)) == 30, "n=5 的 spread 必须互不相交"

    ov = _portfolio_plan(2, "overlap", seed=1)
    assert len(set(ov[0]["reds"]) & set(ov[1]["reds"])) == 4


def test_portfolio_plan_rejects_out_of_range_and_unknown_preset():
    """边界：n 越界、未知 preset 都必须**显式报错**（不是静默返回空）。"""
    from lottery.engine_solver import _portfolio_plan

    with pytest.raises(ValueError):
        _portfolio_plan(0)
    with pytest.raises(ValueError):
        _portfolio_plan(17)
    with pytest.raises(ValueError):
        _portfolio_plan(3, "banana")
    # 大小写/空白容忍
    assert len(_portfolio_plan(2, "  SPREAD  ", seed=0)) == 2


def test_exact_p_at_least_r_matches_hypergeometric_truth():
    """**真值锁**：全枚举结果必须等于超几何分布的解析值。

    外部推导（与被测的位运算暴力枚举完全无关）：
      单注 6 红 vs 开奖 6 红，恰好 k 个重合的概率 = C(6,k)·C(27,6−k) / C(33,6)
        k=4: 15·351/1107568 = 0.0047546…
        k=5: 6·27/1107568  = 0.0001463…
        k=6: 1/1107568     = 0.0000009…
      ⇒ P(≥4) = **0.004901**、P(≥5) = **0.000147**
      两注红球集不相交时，开奖不可能同时让两注各命中 4 个（4+4=8 > 6），
      故 P(≥1 注 ≥4) **恰好** = 2×0.004901 = 0.009802（可算清的等式，非近似）。
    """
    from lottery.engine_solver import _exact_p_at_least_r

    one = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 0}]
    assert _exact_p_at_least_r(one, 4) == pytest.approx(0.004901, abs=1e-6)
    assert _exact_p_at_least_r(one, 5) == pytest.approx(0.000147, abs=1e-6)
    assert _exact_p_at_least_r(one, 6) == pytest.approx(1 / 1107568, abs=1e-9)

    two_disjoint = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 0},
                    {"reds": [7, 8, 9, 10, 11, 12], "blue": 0}]
    assert _exact_p_at_least_r(two_disjoint, 4) == pytest.approx(2 * 0.004901, abs=1e-6)


# ===========================================================================
# ③ engine_orchestrate —— 目标函数（为零梯度缺陷做的守卫）
# ===========================================================================

def test_ticket_objective_ev_conserving_hand_computed():
    """`ev_conserving` = 并集覆盖 − 冗余度 Ā（+ 蓝球项，仅当蓝球未被 plan 接管）。

    外部手算两注不相交 [1..6] + [7..12]：并集 12、两两交集 0 ⇒ Ā=0 ⇒ 得分 12。
    同两注若蓝球也自由（1、2 互异）⇒ +1.5×2 = **15**。
    完全相同的两注：并集 6、交集 6 ⇒ Ā=6 ⇒ 得分 **0**（结构上毫无改善）。
    """
    from lottery.engine_orchestrate import _ticket_objective

    disjoint = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 1},
                {"reds": [7, 8, 9, 10, 11, 12], "blue": 2}]
    assert _ticket_objective(disjoint, "ev_conserving",
                             blue_owned_by_plan=True) == 12.0
    assert _ticket_objective(disjoint, "ev_conserving",
                             blue_owned_by_plan=False) == 15.0

    identical = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 1},
                 {"reds": [1, 2, 3, 4, 5, 6], "blue": 2}]
    assert _ticket_objective(identical, "ev_conserving",
                             blue_owned_by_plan=True) == 0.0
    # 空票组 → 0.0（不是异常）
    assert _ticket_objective([], "ev_conserving") == 0.0


def test_ticket_objective_has_no_blue_gradient_when_owned():
    """**零梯度守卫**：蓝球被 plan 接管时，目标对蓝球分布必须完全无感。

    做法：把两注的蓝球分别换成 `[1,2]`（互异）与 `[1,1]`（相同），红球不变
    —— 若目标里残留蓝球项，两个得分会不同。
    这一条抓住的是 V10-03 修的那个缺陷（蓝球项在 plan 接管后变成常数噪声）。
    """
    from lottery.engine_orchestrate import _ticket_objective

    red_a = [1, 2, 3, 4, 5, 6]
    red_b = [7, 8, 9, 10, 11, 12]
    distinct_blue = [{"reds": red_a, "blue": 1}, {"reds": red_b, "blue": 2}]
    same_blue = [{"reds": red_a, "blue": 1}, {"reds": red_b, "blue": 1}]

    for objective in ("ev_conserving", "coverage"):
        owned = [_ticket_objective(ts, objective, blue_owned_by_plan=True)
                 for ts in (distinct_blue, same_blue)]
        assert owned[0] == owned[1], (
            f"{objective}：蓝球归 plan 时两种蓝球分布得分不同 ⇒ 目标里残留零梯度蓝球项")
        free = [_ticket_objective(ts, objective, blue_owned_by_plan=False)
                for ts in (distinct_blue, same_blue)]
        assert free[0] > free[1], (
            f"{objective}：蓝球自由时互异蓝球必须严格优于相同蓝球")


def test_normalize_coverage_objective_aliases_and_fallback():
    """目标名归一：空白/大小写容忍；未知值回退 `ev_conserving`（不抛错）。

    回退而不是报错是**接口约定**（前端传旧名不该 500）；因此这里钉死回退值本身，
    这样"把回退改成 unknown/抛错"会立刻变红。
    """
    from lottery.engine_orchestrate import _normalize_coverage_objective

    assert _normalize_coverage_objective(" EV_CONSERVING ") == "ev_conserving"
    assert _normalize_coverage_objective("Pmax_High") == "pmax_high"
    assert _normalize_coverage_objective("coverage") == "coverage"
    assert _normalize_coverage_objective("custom_weight") == "custom_weight"
    assert _normalize_coverage_objective("banana") == "ev_conserving"
    assert _normalize_coverage_objective(None) == "ev_conserving"
    assert _normalize_coverage_objective("") == "ev_conserving"


def test_custom_objective_weights_ignores_unknown_keys_and_bad_values():
    """权重合并：未知键丢弃、坏值保留默认（不让一个手误的字符串把权重变成 0/NaN）。"""
    from lottery.engine_orchestrate import DEFAULT_CUSTOM_WEIGHTS, _custom_objective_weights

    merged = _custom_objective_weights({"evil": 5.0, "union": "x", "pmax_high": 2.5})
    assert "evil" not in merged
    assert merged["union"] == DEFAULT_CUSTOM_WEIGHTS["union"], "坏值必须保留默认"
    assert merged["pmax_high"] == 2.5
    assert set(merged) == set(DEFAULT_CUSTOM_WEIGHTS)


def test_pmax_estimator_common_random_numbers_are_deterministic():
    """搜索期估计器必须**可复现**（公共随机数），否则搜索排序变成随机噪声。

    外部推导：同一 estimator 对同一票组两次调用必须逐位相同；
    两个不同 seed 的估计器给出**不同**值（证明它真的在抽样，不是返回常数）。
    """
    from lottery.engine_orchestrate import _PmaxEstimator

    tickets = [{"reds": [1, 2, 3, 4, 5, 6], "blue": 0},
               {"reds": [7, 8, 9, 10, 11, 12], "blue": 0}]
    est = _PmaxEstimator(r=4, n_draws=5000, seed=42)
    first = est.p(tickets)
    second = est.p(tickets)
    assert first == second, "同一 CRN 估计器两次调用必须完全相同"

    other = _PmaxEstimator(r=4, n_draws=5000, seed=7)
    assert other.p(tickets) != first, "不同 seed 给出同一值 ⇒ 估计器没在抽样"

    # 空票组 → 0.0（边界）
    assert est.p([]) == 0.0
    # r=6 的稀有事件（1e-6）在 5000 次抽样里不应命中
    assert _PmaxEstimator(r=6, n_draws=5000, seed=1).p(
        [{"reds": [1, 2, 3, 4, 5, 6], "blue": 0}]) == 0.0


# ===========================================================================
# ④ engine_rules.apply_rules_to_candidates —— off / shadow / active
# ===========================================================================

def _snapshot_with_usable_repeat() -> dict:
    """可用规律快照：维度 `repeat`（上期重号），无过期标记。"""
    return {
        "rules": [{
            "id": "r1", "statement": "上期重号延续", "usable": True,
            "dim": {"key": "repeat", "params": {}},
            "regime": {"status": "usable"},
        }],
        "counts": {"monitor": 0, "rejected": 0},
        "stale": False,
    }


#: 历史只有 1 期，红球 {3,7,8,1,2,9} ⇒ `repeat` 维度的候选命中集合就是这 6 个号。
_HISTORY_ONE = [{"issue": "2025001", "date": "2025-01-01",
                 "reds": [3, 7, 8, 1, 2, 9], "order": [3, 7, 8, 1, 2, 9], "blue": 5}]


def test_apply_rules_off_when_no_snapshot():
    """没有快照 ⇒ `mode="off"`，候选**原样返回**（confidence 一个都不许动）。"""
    from lottery.engine_rules import apply_rules_to_candidates

    cands = [{"reds": [3, 7, 8, 15, 21, 30], "blue": 5, "confidence": 10.0}]
    before = [dict(c) for c in cands]
    out = apply_rules_to_candidates(cands, _HISTORY_ONE, None)

    assert out["mode"] == "off"
    assert out["changed"] is False
    assert out["usable_rules"] == []
    assert cands == before, "off 模式不得修改候选"
    # 诚实文案：必须说明"等同纯模型基线"，而不是假装规律起了作用
    assert "基线" in out["note_zh"]


def test_apply_rules_off_when_snapshot_has_no_usable_rules_and_reports_counts():
    """有快照但 0 条 usable ⇒ `mode="off"`，且文案**如实报出** monitor/rejected 条数。

    这一条钉的是「0 条可用」不能被渲染成「系统没工作」：monitor=3 / rejected=4
    必须出现在文案里（它们是三道门槛如实执行的证据）。
    """
    from lottery.engine_rules import apply_rules_to_candidates

    snap = {
        "rules": [{"id": "r2", "usable": False, "dim": {}, "statement": "仅监控"}],
        "counts": {"monitor": 3, "rejected": 4},
        "stale": False,
    }
    out = apply_rules_to_candidates(
        [{"reds": [1, 2, 3, 4, 5, 6], "blue": 1, "confidence": 5.0}],
        _HISTORY_ONE, snap)
    assert out["mode"] == "off"
    assert out["changed"] is False
    assert "3" in out["note_zh"] and "4" in out["note_zh"], (
        f"文案没有如实报出 monitor/rejected 条数：{out['note_zh']}")
    assert out["snapshot"]["counts"] == {"monitor": 3, "rejected": 4}


def test_apply_rules_shadow_boosts_confidence_but_never_reorders():
    """影子模式：**算出**规律偏置并记录，但**绝不改变候选顺序与号码**。

    构造：候选 B 在前（无命中）、候选 A 在后（命中 3 个重号）。
    影子模式下 A 的 confidence 必须真的被抬高，但**顺序必须不变**。
    """
    from lottery import config, patterns_runtime
    from lottery.engine_rules import apply_rules_to_candidates

    assert patterns_runtime.apply_enabled() is False, (
        "本用例要求默认影子模式（LOTT_RULE_APPLY=0）；若默认值被改成生效，"
        "这条用例的前提就变了 —— 请显式 monkeypatch 而非放宽断言")

    cands = [
        {"reds": [10, 11, 12, 13, 14, 15], "blue": 6, "confidence": 10.0},  # 无命中
        {"reds": [3, 7, 8, 15, 21, 30], "blue": 5, "confidence": 10.0},     # 命中 3 个
    ]
    out = apply_rules_to_candidates(cands, _HISTORY_ONE, _snapshot_with_usable_repeat())

    assert out["mode"] == "shadow"
    assert [t["reds"][0] for t in cands] == [10, 3], "影子模式改变了候选顺序（不允许）"
    assert out["n_boosted"] == 1, "只有命中规律的注才该被加成"
    boosted = next(t for t in cands if t["reds"][0] == 3)
    assert boosted["confidence"] == pytest.approx(10.0 + config.RULE_CONF_BOOST * 3)
    untouched = next(t for t in cands if t["reds"][0] == 10)
    assert untouched["confidence"] == 10.0, "未命中的注一个数字都不许动"


def test_apply_rules_active_does_reorder(monkeypatch):
    """生效模式（`apply_enabled()=True`）**必须真的重排** —— 这是 shadow 的对照组。

    没有这一条，把排序整段删掉也不会有测试变红（影子模式的断言反而会通过）。
    """
    from lottery import patterns_runtime
    from lottery.engine_rules import apply_rules_to_candidates

    monkeypatch.setattr(patterns_runtime, "apply_enabled", lambda: True)

    cands = [
        {"reds": [10, 11, 12, 13, 14, 15], "blue": 6, "confidence": 10.0},  # 无命中
        {"reds": [3, 7, 8, 15, 21, 30], "blue": 5, "confidence": 10.0},     # 命中 3 个
    ]
    out = apply_rules_to_candidates(cands, _HISTORY_ONE, _snapshot_with_usable_repeat())

    assert out["mode"] == "active"
    assert [t["reds"][0] for t in cands] == [3, 10], "生效模式没有把被规律命中的注排到前面"
    assert out["n_resorted"] > 0
    assert out["swapped"], "顺序变化必须给出可审计的 swapped 记录"
    first = out["swapped"][0]
    assert first["conf_after"] > first["conf_before"]


def test_apply_rules_reports_stale_snapshot():
    """过期快照：不作可用规律，且文案必须带过期警示。"""
    from lottery.engine_rules import apply_rules_to_candidates

    snap = _snapshot_with_usable_repeat()
    snap["stale"] = True
    out = apply_rules_to_candidates(
        [{"reds": [3, 7, 8, 15, 21, 30], "blue": 5, "confidence": 10.0}],
        _HISTORY_ONE, snap)
    assert out["mode"] == "off"
    assert "过期" in out["note_zh"], f"过期快照未提示：{out['note_zh']}"


# ===========================================================================
# ⑤ engine_constraints —— 空输入拒绝 + 五维硬约束
# ===========================================================================

def test_constraint_ctx_rejects_empty_history():
    """空历史 → 明确的 `InsufficientDataError`（而不是 numpy 内部抛穿的 IndexError）。

    这是 W02 的边界契约：调用方 `except ENG.InsufficientDataError` 必须接得住，
    所以那个类必须是 `ValueError` 的子类，且与 `engine.InsufficientDataError`
    是**同一个类对象**（两边不是同一个类的话 except 会漏）。
    """
    import lottery.engine as E
    from lottery.engine_constraints import _InsufficientDataError

    assert issubclass(_InsufficientDataError, ValueError)
    assert E.InsufficientDataError is _InsufficientDataError, (
        "engine 反向 import 的必须是同一个类对象，否则调用方 except 接不住")
    with pytest.raises(E.InsufficientDataError):
        E.constraint_ctx([])
    with pytest.raises(ValueError):
        E.constraint_ctx([])


def test_constraint_ctx_produces_five_dimension_band():
    """非空历史：必须给出 5 个维度，且区间**包住**合法形态、**排除**极端形态。

    20 期全部是 [1..6]：和值 21、奇数 3 个、三区 (6,0,0)、连号组数 1。
    跨度不使用历史分位，而是 shape_rules 的**组合恒等带**（17~31）——
    所以这里的断言是「跨度下界 ≥ 1」，而不是「包住 5」：跨度 5 这种极端形态
    本来就该被硬约束拦下（这是设计意图，不是缺陷）。
    """
    import lottery.engine as E

    draws = [{"issue": f"{i:04d}", "date": "2025-01-01",
              "reds": [1, 2, 3, 4, 5, 6], "order": [1, 2, 3, 4, 5, 6],
              "blue": i % 16 + 1} for i in range(1, 21)]
    ctx = E.constraint_ctx(draws)

    for key in ("sum_min", "sum_max", "top_zones", "odd_range",
                "span_min", "span_max", "max_consecutive"):
        assert key in ctx, f"constraint_ctx 缺少维度 {key}"
    assert ctx["sum_min"] <= 21 <= ctx["sum_max"]
    assert 3 in ctx["odd_range"]
    assert (6, 0, 0) in ctx["top_zones"], f"三区比应含 (6,0,0)：{ctx['top_zones'][:5]}"
    assert ctx["max_consecutive"] >= 2
    # 跨度是组合恒等带（不是历史分位）：必须有下界且落在 33 号空间内
    assert ctx["span_min"] >= 1 and ctx["span_max"] <= 32
    assert ctx["span_min"] < ctx["span_max"]


def test_pass_constraints_each_dimension_independently_blocks():
    """五维硬约束：**每一维单独**都能把同一注拦下（隔离到只剩该维起作用）。

    [3,7,8,15,21,30] 手算：和值 84、三区 (3,2,1)、奇数 4 个、跨度 27。
    先放开到"恰好通过"，再把那一维改坏 ⇒ 必须 False。
    这是防止「某一维的检查被删掉也不会变红」的隔离设计（本项目踩过的假绿形态）。
    """
    import lottery.engine as E

    base = {"sum_min": 84, "sum_max": 84, "top_zones": [(3, 2, 1)],
            "odd_range": [4], "span_min": 27, "span_max": 27, "max_consecutive": 6}
    cand = [3, 7, 8, 15, 21, 30]
    assert E.pass_constraints(cand, 5, base) is True, "基准上下文应放行"

    assert E.pass_constraints([1, 2, 3, 4, 5, 6], 5, base) is False, "和值维失效"
    assert E.pass_constraints(cand, 5, dict(base, top_zones=[(6, 0, 0)])) is False, "三区维失效"
    assert E.pass_constraints(cand, 5, dict(base, odd_range=[1])) is False, "奇偶维失效"
    assert E.pass_constraints(cand, 5, dict(base, span_min=30, span_max=30)) is False, "跨度维失效"


def test_pass_constraints_consecutive_group_counting():
    """连号计数语义：**长度 ≥2 的连续段数**（一个 3 连号算 1 组，不是 2 组）。

    用两个样本钉住「段」与「相邻对」的差别（⚠️ 变异验证记录：只用
    [1,2,9,10,20,21] 时，把段头计数简化成相邻对计数是**不可观察变异**
    —— 该样本两者都得 3；加入 [1,2,3,9,10,11]（4 个相邻对 / 2 段）后必红）。
    手算 [1,2,9,10,20,21]：3 段、和值 63、三区 (4,2,0)、奇数 3 个、跨度 20。
    手算 [1,2,3,9,10,11]：2 段（{1,2,3} 与 {9,10,11}）、和值 36、三区 (6,0,0)、
    奇数 4 个（1,3,9,11）、跨度 10。
    """
    import lottery.engine as E

    cand = [1, 2, 9, 10, 20, 21]
    base = {"sum_min": 63, "sum_max": 63, "top_zones": [(4, 2, 0)],
            "odd_range": [3], "span_min": 20, "span_max": 20}
    assert E.pass_constraints(cand, 5, dict(base, max_consecutive=2)) is False
    assert E.pass_constraints(cand, 5, dict(base, max_consecutive=3)) is True

    cand2 = [1, 2, 3, 9, 10, 11]
    base2 = {"sum_min": 36, "sum_max": 36, "top_zones": [(6, 0, 0)],
             "odd_range": [4], "span_min": 10, "span_max": 10}
    # 段数 = 2 ⇒ max=2 放行；相邻对数 = 4 ⇒ 若实现错数成"相邻对"，这里会被拦
    assert E.pass_constraints(cand2, 5, dict(base2, max_consecutive=2)) is True
    assert E.pass_constraints(cand2, 5, dict(base2, max_consecutive=1)) is False


def test_pass_constraints_backward_compatible_with_missing_keys():
    """向后兼容：老 ctx（三区/奇偶口径）不得因缺 span/max_consecutive 键而全部拦下。

    这是"三维 ctx 老调用方还能用"的契约 —— 缺键时该维度**跳过**（取 ±1e9 / 9 兜底），
    而不是取 0 把所有注拦死。用 [1..6]（和值 21、三区 (6,0,0)、奇数 3 个）验证：
    只放开和值 + 三区 + 奇偶即可放行；缺 span 键不该改变结果。
    """
    import lottery.engine as E

    legacy = {"sum_min": 0, "sum_max": 300, "top_zones": [(6, 0, 0)], "odd_range": [3]}
    assert E.pass_constraints([1, 2, 3, 4, 5, 6], 5, legacy) is True
    assert E.pass_constraints([1, 2, 3, 4, 5, 6], 5,
                              dict(legacy, sum_max=10)) is False
    # 缺 span 键时，跨度 31 的极端注也不该被 span 维拦下
    # [2,4,6,8,10,33] 手算：和值 63、奇数 1 个、三区 (5,0,1)、跨度 31、无连号、1 个号在 33 区
    assert E.pass_constraints([2, 4, 6, 8, 10, 33], 5,
                              {"sum_min": 0, "sum_max": 300,
                               "top_zones": [(5, 0, 1)], "odd_range": [1]}) is True


# ===========================================================================
# ⑥ jsonutil.jsonable —— numpy → JSON 原生的唯一枢纽
# ===========================================================================

def test_jsonable_converts_numpy_and_keys():
    """numpy 标量/数组必须变成 Python 原生；dict 键必须全部转 `str`。

    FastAPI 对 numpy 标量直接报错 ⇒ 这里不转换就是「算法算对、接口 500」。
    `json.dumps` 可序列化本身就是断言（numpy 标量会抛 TypeError）。

    注意 `np.float64.item()` 返回的是 **Python float**（`type is float`），
    但 `np.bool_.item()` 返回 Python bool —— 两者都要钉死，因为
    「转成 float64 就交差」的实现会在 JSON 编码处再次炸掉。
    """
    import json

    from lottery.jsonutil import jsonable

    payload = {
        "f": np.float64(1.5), "i": np.int64(3), "arr": np.array([1, 2, 3]),
        "b": np.bool_(True),
        "nested": {1: "x", (2, 3): [np.float64(0.5)]},
    }
    out = jsonable(payload)
    text = json.dumps(out, ensure_ascii=False)
    assert '"f": 1.5' in text and '"i": 3' in text
    assert out["arr"] == [1, 2, 3]
    assert out["b"] is True and type(out["b"]) is bool
    assert set(out["nested"]) == {"1", "(2, 3)"}, "dict 键必须全部 str"
    assert type(out["i"]) is int, f"np.int64 未转成 Python int：{type(out['i'])}"
    # np.float64 本身是 `float` 的子类（`json.dumps` 能吃），所以这里断言
    # **可序列化 + 数值正确**，而不是 `type(...) is float`（后者是 np.float64 做不到的，
    # 会变成一条"永远红"的假红断言）。
    assert isinstance(out["f"], float) and float(out["f"]) == 1.5


def test_jsonable_passes_through_primitives_and_none():
    """原生类型原样返回（含 None / bool，不能被 bool 当 int 处理坏）。"""
    from lottery.jsonutil import jsonable

    for v in (None, True, False, 0, 1, -1, 1.5, "", "x"):
        assert jsonable(v) == v
        assert type(jsonable(v)) is type(v), f"{v!r} 的类型被改变了"


def test_jsonable_stringifies_unserializable_objects():
    """不可序列化对象 → `repr()` 字符串（宁可显示成字符串，也不让端点 500）。"""
    from lottery.jsonutil import jsonable

    class Weird:
        def __repr__(self):
            return "<weird-object>"

    assert jsonable(Weird()) == "<weird-object>"
    # 复数也是常见来源（某些统计量）
    assert isinstance(jsonable(complex(1, 2)), str)


def test_jsonable_depth_guard_stops_runaway_recursion():
    """深度 >12 的嵌套必须被 repr 兜底截断，而不是 RecursionError。

    外部推导：实现上限是 `_depth > 12`，从 0 起算 ⇒ 最多保留 12 层 dict。
    这里构造 14 层，验证保留层数 ≤ 12、兜底处是可序列化的字符串。
    """
    import json

    from lottery.jsonutil import jsonable

    deep: dict = {}
    cur = deep
    for _ in range(14):
        cur["n"] = {}
        cur = cur["n"]
    out = jsonable(deep)

    depth = 0
    node = out
    while isinstance(node, dict) and isinstance(node.get("n"), dict):
        node = node["n"]
        depth += 1
    assert depth <= 12, f"深度保护未生效：保留了 {depth} 层"
    json.dumps(out)      # 兜底处必须可序列化（repr 字符串）


# ===========================================================================
# ⑦ diagnose —— 结构画像 / 相似注 / 完整报告
# ===========================================================================

def test_structure_profile_hand_computed():
    """`structure_profile` 的真值锁：每个字段都由我手算（与实现无关）。

    号码 [3,7,8,15,21,30] + 蓝 9 的手算：
      和值 84；奇数 4 个（3,7,15,21）/ 偶数 2；三区 (3,2,1)（≤11 有 3、7、8）
      跨度 30−3 = 27；AC：15 个两两差值去重后仍 15 个 − (6−1) = **10**
      连号 True（7-8）；同尾 False（尾数 3,7,8,5,1,0 全不同）
      质数 2 个（3、7）；小号(1-16) 4 个（3,7,8,15）；模 3 为 0 的 4 个（3,15,21,30）
    """
    from lottery.diagnose import structure_profile

    p = structure_profile([3, 7, 8, 15, 21, 30], 9, [])

    assert p["reds"] == [3, 7, 8, 15, 21, 30]
    assert p["blue"] == 9
    assert p["sum"] == 84
    assert (p["odd_count"], p["even_count"]) == (4, 2)
    assert p["zone_counts"] == [3, 2, 1]
    assert p["span"] == 27
    assert p["ac"] == 10
    assert p["has_consecutive"] is True
    assert p["has_same_tail"] is False
    assert p["prime_count"] == 2
    assert p["size_count_small"] == 4
    assert p["route_0_count"] == 4


def test_structure_profile_with_empty_history_degrades_honestly():
    """无历史时：热/冷计数为 0，`sum_in_range=False`（**不许**假装在区间内）。"""
    from lottery.diagnose import structure_profile

    p = structure_profile([3, 7, 8, 15, 21, 30], 9, [])
    assert p["hot_count"] == 0
    assert p["cold_count"] == 0
    assert p["sum_in_range"] is False
    assert p["sum_pct_low"] == 0 and p["sum_pct_high"] == 0
    assert isinstance(p["omit_detail"], list)


def test_find_similar_draws_hamming_and_ordering():
    """相似注 = 红球**对称差**（要替换几个号），按 (差值升序, 蓝球命中优先) 排序。

    手算（查询票 {3,7,8,15,21,30}，蓝 9）：
      D   {3,7,8,15,21,30} 蓝 1 → 差 0，红中 6，蓝不中
      A   {3,7,8,15,21,31} 蓝 9 → 差 2，红中 5，蓝中
      E   {3,7,8,15,23,30} 蓝 1 → 差 2，红中 5，**蓝不中**  ← 钉 tiebreak 用
      C   {3,7,8,15,22,30} 蓝 9 → 差 2，红中 5，蓝中
      B   {1..6}           蓝 9 → 差 10，被 max_diff 过滤
    期望顺序 D, A, C, E：diff 升序；diff 相同时**蓝中优先** ⇒ A、C 在 E 前
    （A、C 之间保持输入顺序）。

    ⚠️ 变异验证记录：
      ① 若样本里没有 E（第一版），把排序键 `-x[1]` 反转成 `x[1]` 是**不可观察
         变异**（A、C 蓝都中、D 是唯一 diff=0）—— 加 E 后必红。
      ② 第一版 E 写成 [3,7,8,15,22,31]，与查询票的对称差其实是 **4**（换了 21→22
         和 30→31 两个位置），被 diff 主键排到最后，tiebreak 依旧测不到 ——
         这一点是我实际跑变异时发现的，说明「凭直觉造样本」会造出无效对照。
         正确样本只改一个号：23 替换 21 ⇒ 对称差 2。
    """
    from lottery.diagnose import find_similar_draws

    draws = [
        {"issue": "A", "date": "2025-01-01", "reds": [3, 7, 8, 15, 21, 31], "blue": 9},
        {"issue": "B", "date": "2025-01-02", "reds": [1, 2, 3, 4, 5, 6], "blue": 9},
        {"issue": "C", "date": "2025-01-03", "reds": [3, 7, 8, 15, 22, 30], "blue": 9},
        {"issue": "D", "date": "2025-01-04", "reds": [3, 7, 8, 15, 21, 30], "blue": 1},
        {"issue": "E", "date": "2025-01-05", "reds": [3, 7, 8, 15, 23, 30], "blue": 1},
    ]
    res = find_similar_draws(draws, [3, 7, 8, 15, 21, 30], 9, n_similar=10, max_diff=4)

    assert [r["issue"] for r in res] == ["D", "A", "C", "E"], "排序/过滤与手算不符"
    assert res[0]["similarity_diff"] == 0 and res[0]["red_hit"] == 6
    assert res[0]["blue_hit"] is False
    assert res[1]["similarity_diff"] == 2 and res[1]["red_hit"] == 5
    assert res[1]["blue_hit"] is True
    assert [r["issue"] for r in find_similar_draws(
        draws, [3, 7, 8, 15, 21, 30], 9, max_diff=0)] == ["D"]
    assert [r["issue"] for r in find_similar_draws(
        draws, [3, 7, 8, 15, 21, 30], 9, n_similar=1)] == ["D"]
    assert find_similar_draws([], [3, 7, 8, 15, 21, 30], 9) == []


@pytest.mark.parametrize("reds,blue,expect_error_fragment", [
    ([1, 1, 2, 3, 4, 5], 7, "红球"),      # 有重复
    ([1, 2, 3, 4, 5, 34], 7, "红球"),     # 越上界
    ([0, 2, 3, 4, 5, 6], 7, "红球"),      # 越下界
    ([1, 2, 3, 4, 5], 7, "红球"),         # 只有 5 个
    ([1, 2, 3, 4, 5, 6], 0, "蓝球"),      # 蓝球越界
    ([1, 2, 3, 4, 5, 6], 17, "蓝球"),     # 蓝球越界
])
def test_diagnose_rejects_invalid_tickets(reds, blue, expect_error_fragment):
    """边界：非法号码必须返回**可读的 error 字段**（不是抛异常、不是空报告）。"""
    from lottery.diagnose import diagnose

    out = diagnose(reds, blue, [])
    assert "error" in out
    assert expect_error_fragment in out["error"]
    assert "profile" not in out, "非法输入不应产出画像"


def test_diagnose_full_report_shape_on_empty_history():
    """完整报告结构：空历史下也必须给出全部键（并带诚实声明）。"""
    from lottery.diagnose import diagnose

    out = diagnose([3, 7, 8, 15, 21, 30], 9, [])
    for key in ("profile", "similar_count", "similar_red_hits_dist",
                "similar_blue_hit_rate", "similar_draws", "note"):
        assert key in out, f"诊断报告缺少 {key}"
    assert out["similar_count"] == 0
    assert out["similar_red_hits_dist"] == {}
    assert out["similar_blue_hit_rate"] == 0.0
    assert "不构成中奖预测" in out["note"], "必须带诚实声明"


# ===========================================================================
# ⑧ llm_config_store —— 全局共享通道的读写边界
# ===========================================================================

#: `llm_config_store.save()` 会**直接赋值**的 config 模块全局（llm_config_store.py:66-87）。
#: 它们不在 conftest 的 `_ISOLATED_CONFIG_ATTRS` 名单里 —— V40 审计实测发现：
#: 本文件调 `save()` 后，同 worker 后续测试的 `llm_configured()` 会恒 True。
#: 本 fixture 是**本文件自兜底**（快照/恢复 6 属性 + 对应环境变量）；
#: conftest 扩名单由 test-speed 泳道补，两边互不冲突。
_LLM_SAVE_ATTRS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
                   "LLM_MODEL_LIST", "LLM_SAMPLES", "LLM_DISABLED")
_LLM_SAVE_ENV = {"LLM_BASE_URL": "LOTT_LLM_BASE_URL", "LLM_API_KEY": "LOTT_LLM_API_KEY",
                 "LLM_MODEL": "LOTT_LLM_MODEL", "LLM_SAMPLES": "LOTT_LLM_SAMPLES",
                 "LLM_DISABLED": "LOTT_LLM_DISABLED"}


@pytest.fixture
def llm_config_guard():
    import os

    from lottery import config

    attrs = {name: getattr(config, name, None) for name in _LLM_SAVE_ATTRS}
    envs = {env: os.environ.get(env) for env in _LLM_SAVE_ENV.values()}
    yield
    for name, value in attrs.items():
        if value is not None:
            setattr(config, name, value)
    for env, value in envs.items():
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value


def test_llm_config_save_replaces_model_list_and_clamps_samples(llm_config_guard):
    """保存模型是**替换**（不是前插）；samples 夹到 [1,20]，坏值回 3。

    为什么替换很重要：前插会把 .env 遗留旧模型留在轮转列表里，
    新模型一旦失败，旧模型的 404 会掩盖真实原因（见模块注释）。
    """
    from lottery import config, llm_config_store

    llm_config_store.save({"model": "m1", "samples": "99"})
    assert config.LLM_MODEL_LIST == ["m1"], "模型列表必须被替换而非前插"
    assert config.LLM_SAMPLES == 20, "samples=99 应夹到 20"

    llm_config_store.save({"samples": "abc"})
    assert config.LLM_SAMPLES == 3, "坏 samples 应回 3（不是抛错、也不是 0）"

    llm_config_store.save({"samples": 0})
    assert config.LLM_SAMPLES == 1, "samples 下界应为 1"


def test_llm_config_save_strips_url_and_key_whitespace(llm_config_guard):
    """`base_url` 去空白与尾部 `/`；`api_key` 去空白。空串 ⇒ None。"""
    from lottery import config, llm_config_store

    llm_config_store.save({"base_url": "  http://example.com/api/  ",
                           "api_key": "  k2  "})
    assert config.LLM_BASE_URL == "http://example.com/api"
    assert config.LLM_API_KEY == "k2"

    llm_config_store.save({"base_url": "   ", "api_key": ""})
    assert config.LLM_BASE_URL is None and config.LLM_API_KEY is None


def test_llm_config_save_preserves_handwritten_keys(llm_config_guard):
    """**部分更新**：payload 提到的键才改；文件里的手工键（extra_body/total_timeout）必须保留。"""
    import json

    from lottery import config, llm_config_store

    config.LLM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.LLM_CONFIG_FILE.write_text(json.dumps(
        {"extra_body": {"x": 1}, "total_timeout": 99, "model": "old"}), encoding="utf-8")

    res = llm_config_store.save({"model": "m2"})
    assert res["ok"] is True

    on_disk = json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    assert on_disk["extra_body"] == {"x": 1}, "手工 extra_body 被抹掉了"
    assert on_disk["total_timeout"] == 99, "手工 total_timeout 被抹掉了"
    assert on_disk["model"] == "m2"


def test_llm_config_read_never_exposes_api_key(llm_config_guard):
    """`read()` 是公开视图：**绝不含 api_key 明文**（只给 configured 布尔）。"""
    from lottery import config, llm_config_store

    secret = "sk-super-secret-1234"
    llm_config_store.save({"api_key": secret, "base_url": "http://x", "model": "m"})
    view = llm_config_store.read()

    assert "api_key" not in view, f"公开视图泄露了 api_key：{sorted(view)}"
    assert secret not in repr(view), "api_key 明文出现在 read() 的返回里"
    assert view["configured"] is True
    assert view["model"] == "m" and view["base_url"] == "http://x"
    # 关键字段齐备（前端据此渲染设置页）
    for key in ("disabled", "base_url", "model", "model_list", "samples",
                "max_tokens", "learned_max_tokens", "total_timeout", "verify", "configured"):
        assert key in view, f"read() 缺少字段 {key}"
    # 模型列表必须是**独立副本**（外部改动不能污染 config）
    view["model_list"].append("injected")
    assert "injected" not in config.LLM_MODEL_LIST
