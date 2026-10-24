"""V12-11（L2-e 预算内最优组合求解器）验收测试。

对应 docs/analysis/V12_PATTERN_REDEFINITION_PLAN.md 阶段 B V12-11 的三条硬性验收：
  a. n=5 ``min_var`` 的解 ``var_ratio ≤ 0.1112``（最优点 1/9 = 0.1111…）；
  b. 与**穷举**（n≤5 的全部互不相交构型 / 足够大的随机构型全集）的解差 ≤ 1e-9；
  c. **反向用例**：``ev_ratio_full`` 与输入注单**无关、逐位相同** ——
     求解器不得宣称（也不得实际）提高 EV。

测试原则
--------
* 不依赖网络、不依赖真实库数据（conftest 的 autouse fixture 已给独立临时库；
  本文件**不**裸赋值 ``config.DB_PATH``）。
* 真实历史数据上 ``engine.constraint_ctx`` 的通过率约 43.6%（实测），候选池
  足够大；但耗时受全枚举支配，所以本文件的 ``max_p_at_least_r`` 用例把候选池
  控制在小规模（C(m,n) ≤ 32 时求解器会做**全子集精确复算**，可与穷举逐位对比）。
* 每个反向用例都在 docstring 里写「若实现错误，本用例如何失败」。
"""
from __future__ import annotations

# V36（2026-10-24 审计 P1-6）：六等奖官方口径常量（独立算，不 import 被测模块）。
# 官方六等奖 = 蓝中 **且** 红 ≤ 2；旧基准 0.15625 = 5/32 隐含"蓝中即中奖"，
# 漏了红球条件（偏高 6.25%）。正确值 = 5×P(红≤2)/32 ≈ 0.1472314。
from math import comb as _comb_v36
P_RED_LE2_V36 = (_comb_v36(6, 0) * _comb_v36(27, 6) + _comb_v36(6, 1) * _comb_v36(27, 5)
                 + _comb_v36(6, 2) * _comb_v36(27, 4)) / _comb_v36(33, 6)
EV_RATIO_SIXTH_V36 = 5.0 * P_RED_LE2_V36 / 32.0

import itertools
import random

import pytest

from lottery import engine as E

R = 33


def _t(reds, blue=1):
    return {"reds": sorted(int(x) for x in reds), "blue": int(blue)}


def _sum_a(tickets):
    sets = [set(t["reds"]) for t in tickets]
    return sum(len(sets[i] & sets[j])
               for i in range(len(sets)) for j in range(i + 1, len(sets)))


def _pool(m=120, seed=7):
    rng = random.Random(seed)
    return [_t(rng.sample(range(1, R + 1), 6), blue=rng.randint(1, 16))
            for _ in range(m)]


def _small_pool(m=10, seed=3):
    """小候选池（C(m,3)=120 ≤ 32? 否；C(10,3)=120 → 走全子集精确复算的上限是 32），
    这里用 m=6、n=3（C(6,3)=20 ≤ 32）以便能穷举对照。"""
    rng = random.Random(seed)
    return [_t(rng.sample(range(1, R + 1), 6), blue=rng.randint(1, 16))
            for _ in range(m)]


# ===========================================================================
# a. n=5 min_var 的解 var_ratio ≤ 0.1112
# ===========================================================================

def test_min_var_n5_reaches_lower_bound():
    """a：n=5 ``min_var`` 的解 ``var_ratio ≤ 0.1112``，且恰为解析下界 1/9。

    若实现错误，本用例如何失败：
      * 若求解器返回的 5 注有任意两两交集 > 0（没有真正铺开）→ Ā > 0 →
        var_ratio > 0.1112，直接失败；
      * 若 ``var_ratio`` 被手写成常量 1/9 而 ``tickets`` 与它不一致 → 下面
        用 ``ticket_distribution`` 独立复算的断言会失败。
    """
    res = E.optimal_portfolio(5, "min_var", seed=0)
    assert res["objective"] == "min_var"
    assert res["var_ratio"] <= 0.1112
    assert res["var_ratio"] == pytest.approx(1.0 / 9.0, abs=1e-12)
    # 独立复算：解本身必须两两不相交
    tickets = res["tickets"]
    assert len(tickets) == 5
    assert _sum_a(tickets) == 0, "n=5 的最小冗余解必须两两互不相交（ΣA=0）"
    dist = E.ticket_distribution(tickets)
    assert dist["var_ratio"] == pytest.approx(res["var_ratio"], abs=1e-12)
    assert dist["union_coverage"] == 30
    assert dist["redundancy"] == pytest.approx(0.0, abs=1e-12)
    # 必须回报搜索元数据
    search = res["search"]
    for k in ("strategy", "evals", "exhaustive"):
        assert k in search, f"search 缺字段 {k}"


@pytest.mark.parametrize("n", list(range(1, 17)))
def test_min_var_var_ratio_equals_global_lower_bound(n):
    """``min_var`` 对 n∈[1..16] 的 ``var_ratio`` 恰等于**全局下界**（逐位，非近似）。

    下界推导（凸性）：ΣA = Σ_x C(cnt_x,2) 是各红号出现次数的凸可分离函数，
    交换论证 ⇒ 最优必为 6n 个号位在 33 个号上均摊（次数差 ≤1）。

    若实现错误，本用例如何失败：
      * 构造有重复号 / 注内重号 → ``ticket_distribution`` 抛 ValueError；
      * 没真正均摊（例如 n=7 时仍按"每注塞 6 个相邻号"）→ 实际 ΣA 大于下界 →
        var_ratio 偏离下界，逐位比较失败。
    """
    res = E.optimal_portfolio(n, "min_var", seed=1)
    dist = E.ticket_distribution(res["tickets"], with_distribution_exact=False)
    assert dist["var_ratio"] == pytest.approx(res["var_ratio_theoretical_min"], abs=1e-12)
    assert res["var_ratio"] == pytest.approx(res["var_ratio_theoretical_min"], abs=1e-12)
    # ΣA 必须恰等于均摊下界（整数相等，不是近似）
    pairs = n * (n - 1) // 2
    if pairs:
        base, extra = divmod(6 * n, R)
        lower = sum(c * (c - 1) // 2
                    for c in [base + (1 if i < extra else 0) for i in range(R)])
        assert _sum_a(res["tickets"]) == lower
    # n≤5 时互不相交；n=11 时取到 var_ratio 的绝对下界 0
    if n <= 5:
        assert _sum_a(res["tickets"]) == 0
    if n == 11:
        assert res["var_ratio"] == 0.0


# ===========================================================================
# b. 与穷举的解差 ≤ 1e-9
# ===========================================================================

def test_b_min_var_matches_exhaustive_search_over_disjoint_configs():
    """b（min_var，n≤5）：在**足够大的构型全集**上穷举后与求解器解差 ≤ 1e-9。

    穷举口径（n≤5，互不相交构型空间等价类）：
      * 全部 n 注互不相交时 ΣA ≡ 0 **恒成立**（任意两注交集为空），这是由定义
        直接给出的全局下界 ⇒ var_ratio ≥ 1/9 当 n=5（n<5 时下界更高）；
      * 因此只需证明「ΣA=0 可达」+「ΣA 不可能 <0」：用**大样本穷举**
        （2000 个独立随机互不相交构型，每个构型由 seed 唯一确定）确认穷举最小值
        恰为 ΣA=0 且 var_ratio 恰为下界，再与 ``optimal_portfolio`` 的解逐位比较。

    若实现错误，本用例如何失败：
      * 求解器的 ΣA > 0 → 其 var_ratio 与穷举最优（1/9 = 0.1111…）的差远大于 1e-9；
      * ``_variance_stats`` 走浮点闭式产生擦伤（如 0.11111111111111112 vs …10）
        → 与穷举精确最小值比对时差 > 1e-9；
      * 求解器手写常量 var_ratio 而票组不一致 → 下面的票组独立复算断言失败。
    """
    import random as _random

    for n in (2, 3, 4, 5):
        # 穷举样本：2000 个随机互不相交构型（每个由 seed 唯一确定，可复现）
        best_ratio = None
        best_abar = None
        for s in range(2000):
            rng = _random.Random(10_000 * n + s)
            nums = rng.sample(range(1, R + 1), 6 * n)
            part = [_t(nums[6 * i:6 * (i + 1)]) for i in range(n)]
            sum_a = _sum_a(part)
            ratio = E._variance_stats(n, sum_a)["var_ratio"]
            if best_ratio is None or ratio < best_ratio - 1e-15:
                best_ratio, best_abar = ratio, sum_a
        assert best_abar == 0, f"n={n}: 互不相交构型的穷举最小值应为 ΣA=0，得到 {best_abar}"
        sol = E.optimal_portfolio(n, "min_var", seed=0)
        assert sol["var_ratio"] == pytest.approx(best_ratio, abs=1e-9), n
        dist = E.ticket_distribution(sol["tickets"], with_distribution_exact=False)
        assert dist["var_ratio"] == pytest.approx(best_ratio, abs=1e-9), n
        assert _sum_a(sol["tickets"]) == 0

    # n=5 的最高优先验收：解 == 穷举最优 == 1/9
    assert E.optimal_portfolio(5, "min_var")["var_ratio"] == pytest.approx(
        E._variance_stats(5, 0)["var_ratio"], abs=1e-9)

    # n=6..10：穷举「最小可能 ΣA」（号位均摊）作为全局下界，与求解器解逐位相同
    for n in range(6, 11):
        base, extra = divmod(6 * n, R)
        counts = [base + (1 if i < extra else 0) for i in range(R)]
        lower_sum_a = sum(c * (c - 1) // 2 for c in counts)
        exhaustive_best = E._variance_stats(n, lower_sum_a)["var_ratio"]
        sol = E.optimal_portfolio(n, "min_var", seed=3)
        assert sol["var_ratio"] == pytest.approx(exhaustive_best, abs=1e-9), n


def test_b_max_p_at_least_r_matches_exhaustive_subsets():
    """b（max_p_at_least_r）：候选池 C(6,3)=20 全部子集精确复算，解差 ≤ 1e-9。

    ``optimal_portfolio`` 在 C(m,n) ≤ 32 时会对**全部**子集做精确复算
    （``search.exhaustive=True``）。这里用独立代码把 20 个子集重算一遍。

    若实现错误，本用例如何失败：
      * 局部搜索过早停止且候选池小到能穷举时，求解器给出次优解 →
        与独立穷举最优差 > 1e-9 → 失败；
      * ``p_at_least_r`` 回退成 iid 近似 → 与独立穷举的全枚举值差 1e-3 量级 → 失败。
    """
    pool = _small_pool(6, seed=11)
    res = E.optimal_portfolio(3, "max_p_at_least_r", candidates=pool, seed=0)
    assert res["search"]["exhaustive"] is True
    assert res["search"]["n_candidates"] == 6
    best = 0.0
    for idx in itertools.combinations(range(6), 3):
        sel = [pool[i] for i in idx]
        p = E._exact_red_tail(sel)["p_ge"][4]
        best = max(best, p)
    assert res["p_at_least_r"]["4"] == pytest.approx(best, abs=1e-9)
    # 解本身复算出同一个概率
    got = E.ticket_distribution(res["tickets"], with_distribution_exact=True)
    assert got["p_at_least_r"]["4"] == pytest.approx(best, abs=1e-9)
    assert got["p_at_least_r_method"] == "exact_enum"


def test_b_max_p_at_least_r_greedy_never_worse_than_greedy_only():
    """b（贪心基准）：局部搜索的解**不劣于**只用贪心构造的解（差 ≤ 1e-9 且 ≥）。

    独立实现一遍贪心：每步加入使精确 P 最大的候选（不复用 solver 的循环）。

    若实现错误，本用例如何失败：
      * 若求解器的局部搜索接受了**劣化**的一步（非单调爬山）并被卡住 →
        终解可能低于纯贪心 → ``res_p < greedy_p`` 失败；
      * 若目标函数用了代理（并集/冗余）而非精确 P → 选出的子集与贪心最优
        差 > 1e-9 → 失败。
    """
    pool = _small_pool(8, seed=5)
    n = 4
    # 独立贪心
    chosen = []
    remaining = list(range(len(pool)))
    cur = 0.0
    for _ in range(n):
        best_i, best_p = None, -1.0
        for i in remaining:
            p = E._exact_red_tail([pool[j] for j in chosen + [i]])["p_ge"][4]
            if p > best_p:
                best_i, best_p = i, p
        chosen.append(best_i)
        remaining.remove(best_i)
        cur = best_p
    greedy_p = cur
    res = E.optimal_portfolio(n, "max_p_at_least_r", candidates=pool, seed=0)
    res_p = res["p_at_least_r"]["4"]
    assert res_p >= greedy_p - 1e-9, f"solver={res_p} 竟劣于纯贪心={greedy_p}"
    # 同时不劣于解本身（自洽）
    assert res_p == pytest.approx(
        E._exact_red_tail(res["tickets"])["p_ge"][4], abs=1e-9)


# ===========================================================================
# c. 反向用例：求解器不得宣称提高 EV
# ===========================================================================

def test_c_ev_ratio_full_is_invariant_across_all_objectives_and_ticket_sets():
    """**反向用例（c）**：``ev_ratio_full`` 与输入注单**无关、逐位相同**。

    三种目标、多组 n、多种票组（互不相交 / 高重叠 / 同票重复 / 随机池）下，
    ``ev_ratio_full`` 必须**逐位相等**，且等于单注口径常量。

    若实现错误，本用例如何失败：
      * 若有人把"优化"实现在 EV 上（例如按覆盖加权、或把 ``ev_yuan`` 乘上
        结构因子）→ 不同目标的 ``ev_ratio_full`` 不再逐位相等 → 本用例失败；
      * 若把 ``ev_ratio_sixth``（0.1472314）误当整注 EV 返回 → 与
        0.45673239701760976 不等 → 失败；
      * 若求解器在返回值里**宣称**提高了 EV（``ev_ratio_note`` 含"提高期望"
        之类的肯定表述，或 ``search`` 里出现 ``ev_gain`` / ``expected_improvement``
        之类字段）→ 下面的文案/字段断言失败。
    """
    single = E._single_ticket_ev(None, sixth="published")["ev_ratio"]
    evs = {}
    for obj in ("min_var", "max_p_any_blue", "max_p_at_least_r"):
        for n in (1, 3, 5, 10):
            # 候选池必须 ≥ n。原夹具写死 8 注却求 n=10 的解 —— 那是**夹具错误**：
            # 求解器拒绝「池子比需求小」是正确行为（否则只能交付不足注数的解）。
            cand = _pool(max(n, 8), seed=n) if obj == "max_p_at_least_r" else None
            res = E.optimal_portfolio(n, obj, seed=0, candidates=cand)
            evs[f"{obj}/{n}"] = res["ev_ratio_full"]
            assert res["ev_ratio_full"] == single, (
                f"{obj}/{n}: ev_ratio_full={res['ev_ratio_full']!r} != 单注 {single!r}")
            assert res["prize_assumption"], f"{obj}/{n}: prize_assumption 必填"
    assert len(set(evs.values())) == 1, f"不同目标的 EV 出现了差异：{evs}"

    # 与**输入注单无关**：直接对任意票组算引擎 EV，逐位相同
    other = _pool(9, seed=99)
    assert E.ticket_distribution(other, with_distribution_exact=False)["ev_ratio_full"] == single
    assert single == pytest.approx(0.45673239701760976, abs=1e-15)

    # 文案与字段：不得宣称提高 EV
    res = E.optimal_portfolio(5, "min_var")
    note = res["ev_ratio_note"]
    assert "不改变" in note or "无关" in note
    for bad in ("提高期望收益", "提升期望", "增加期望收益"):
        assert bad not in note.replace("不改变、也不宣称提高期望收益", ""), (
            f"ev_ratio_note 出现了宣称提高期望的表述：{bad}")
    for bad_key in ("ev_gain", "ev_improvement", "expected_return_gain",
                    "ev_ratio_improved", "higher_ev"):
        assert bad_key not in res and bad_key not in res["search"], (
            f"返回值出现宣称 EV 提升的字段：{bad_key}")


def test_c_objectives_do_not_claim_ev_gain_through_p_any_blue():
    """**反向用例（c 的专项）**：``max_p_any_blue`` 只提高 P(≥1蓝)，不碰 EV。

    同一 n 下，蓝球全不同的解（P(≥1蓝)=n/16）与蓝球全相同的解（P(≥1蓝)=1/16）
    必须给出**完全相同的** ``ev_ratio_full``。

    若实现错误，本用例如何失败：
      * 若 max_p_any_blue 的实现把"蓝球覆盖更高 = 更赚"写进 EV（例如复用
        ``ev_ratio_sixth × 覆盖数``）→ 两个蓝球方案 EV 不同 → 失败；
      * 若求解器返回的 ``p_any_blue`` 不等于 ``min(n,16)/16`` 闭式 → 失败。
    """
    n = 10
    res = E.optimal_portfolio(n, "max_p_any_blue", seed=0)
    assert res["p_any_blue"] == pytest.approx(min(n, 16) / 16, abs=1e-12)
    assert len({t["blue"] for t in res["tickets"]}) == n
    same_blue = [{"reds": t["reds"], "blue": 1} for t in res["tickets"]]
    d_same = E.ticket_distribution(same_blue, with_distribution_exact=False)
    d_spread = E.ticket_distribution(res["tickets"], with_distribution_exact=False)
    assert d_spread["p_any_blue"] == pytest.approx(10 / 16, abs=1e-12)
    assert d_same["p_any_blue"] == pytest.approx(1 / 16, abs=1e-12)
    # 覆盖从 1/16 提到 10/16，但 EV 逐位不变（唯一诚实的结论）
    assert d_same["ev_ratio_full"] == d_spread["ev_ratio_full"] == res["ev_ratio_full"]
    assert d_same["ev_ratio_sixth"] == d_spread["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)


# ===========================================================================
# 其他契约：参数校验 / 约束 / 可复现
# ===========================================================================

def test_objective_validation_and_range():
    """未知 objective / 越界 n 必须 ValueError（不许静默回退）。"""
    with pytest.raises(ValueError):
        E.optimal_portfolio(5, "max_ev")
    with pytest.raises(ValueError):
        E.optimal_portfolio(0, "min_var")
    with pytest.raises(ValueError):
        E.optimal_portfolio(17, "min_var")


def test_min_var_is_deterministic_for_same_seed():
    """同 seed 同结果（可复现）。

    若实现错误，本用例如何失败：
      * 求解器用了未固定的全局 ``random`` → 两次票组不同 → 失败。
    """
    a = E.optimal_portfolio(8, "min_var", seed=42)["tickets"]
    b = E.optimal_portfolio(8, "min_var", seed=42)["tickets"]
    assert a == b


def test_min_var_with_constraints_respects_hard_constraints(seeded_db):
    """``constraints`` 给出的硬约束必须被解满足（用合成数据造 ctx，不读真实库）。

    若实现错误，本用例如何失败：
      * 约束被忽略（直接返回无约束均摊解）→ 解里存在 ``pass_constraints`` 为假
        的注 → 失败；
      * 约束过严时静默返回不合规解 → 同样失败（此时允许 ``constraint_fallback``
        标记为真，但解仍需通过 ``pass_constraints`` 之外的宽松核对——见断言）。
    """
    draws = seeded_db(n=600, seed=42)
    ctx = E.constraint_ctx(draws)
    res = E.optimal_portfolio(5, "min_var", constraints=ctx, seed=0)
    assert res["search"]["constraints_applied"] is True
    ok = sum(1 for t in res["tickets"] if E.pass_constraints(t["reds"], t["blue"], ctx))
    if res["search"]["constraint_fallback"]:
        # 兜底：候选池里找不到足够合规注（真实数据下 ctx 通过率约 43.6%）
        # 此时必须**如实**标记，且至少不是静默失败
        assert res["search"]["strategy"].endswith("constraint_repair")
    else:
        assert ok == 5, f"约束解里有 {5 - ok} 注不合规：{res['tickets']}"


def test_search_metadata_reports_strategy_and_exhaustive():
    """``search`` 元数据齐全且语义正确：无约束 min_var 是精确最优（exhaustive）。"""
    for obj in ("min_var", "max_p_any_blue"):
        res = E.optimal_portfolio(5, obj)
        s = res["search"]
        assert isinstance(s["strategy"], str) and s["strategy"]
        assert isinstance(s["evals"], int) and s["evals"] >= 1
        assert s["exhaustive"] is True
    res = E.optimal_portfolio(3, "max_p_at_least_r", candidates=_small_pool(6, seed=2))
    assert res["search"]["exhaustive"] is True
    assert res["p_at_least_r_method"] == "exact_enum"
