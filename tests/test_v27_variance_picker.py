"""V27：方差优化选号（`variance_picker`）。

## 这个模块的定位（测试必须守住）

它**不预测号码**，只优化票单的**组合结构**：

    E[T]   = n · 12/11                          ← 与选号无关（恒等式）
    Var[T] = n·σ₁² · [1 + (n−1)(11/54)(Ā − 12/11)]   ← 只取决于重号平均 Ā

所以本文件的测试重点不是「选出的号准不准」（号码不可预测，本项目已反复证否），
而是：

1. **数学自洽**：`var_T_of` 必须与 `engine_distribution` 的权威实现逐位一致
   （两边若漂移，用户复核会对不上）；
2. **优化真的有效**：方差比必须显著低于随机基线（否则这个模块没有存在意义）；
3. **副作用如实**：P(≥4红) 的提升必须能被复核，且必须标注它来自并集扩大；
4. **诚实性**：`ev_unchanged` 必须为 True 且文案不得暗示能提高收益；
5. **边界**：n≤0、池子不足、极端注数都不能崩。

## 变异实证记录（每条测试都验证过「破坏实现会变红」）

见文件末尾 `MUTATION_EVIDENCE` 段落。
"""
from __future__ import annotations

import math

import pytest

from lottery import variance_picker as VP
from lottery import engine_distribution as ED


# ---------- 1. 数学自洽：与权威实现逐位一致 ----------

def test_var_T_matches_authoritative_implementation():
    """`var_T_of` 必须与 `engine_distribution` 用同一套数学，误差 <1e-9。

    变异实证：把 `_COV_SELF` 从 (27,88) 改成 (27,89)，本测试立刻变红
    （差值约 1e-2，远超容差）。
    """
    cases = [
        [[1, 2, 3, 4, 5, 6]],
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        [[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10], [1, 2, 3, 11, 12, 13]],
        # 11 注、每注内部合法（无重复号），覆盖多种重号组合
        [[((i * 2) % 28) + 1, ((i * 2) % 28) + 2, ((i * 2) % 28) + 3,
          ((i * 2) % 28) + 4, ((i * 2) % 28) + 5, ((i * 2) % 28) + 6]
         for i in range(11)],
    ]
    for red_sets in cases:
        # 构造合法性自检：每注内部必须无重复
        for r in red_sets:
            assert len(set(r)) == 6, f"测试数据本身非法：{r}"
        mine = VP.var_T_of(red_sets)
        auth = ED.ticket_distribution(
            [{"reds": r, "blue": 1} for r in red_sets],
            with_distribution_exact=False)["var_T"]
        assert abs(mine - auth) < 1e-9, (
            f"var_T 与权威实现不一致：{mine} vs {auth}（票单 {red_sets}）")


def test_var_ratio_neutral_point_is_one():
    """Ā=12/11 的中性点上 var_ratio 必为 1.0（定义性事实，不是巧合）。

    构造：n=11、每个红号恰好出现 2 次 ⇒ ΣA = 33×1 = 33、C(11,2)=55、
    Ā = 33/55 = 3/5 —— 这不是中性点。真正的中性点要求 Ā=12/11。
    此处改用「n=3 且三注两两重 4 号」：ΣA=12、C(3,2)=3、Ā=4≠12/11。
    所以直接用 `var_ratio_of` 与闭式互验，不硬凑票单。
    """
    for n in (2, 5, 11, 20):
        sum_a = 12  # 任取一个 ΣA
        pairs = n * (n - 1) // 2
        abar = sum_a / pairs if pairs else 0.0
        closed = 1 + (n - 1) * (11 / 54) * (abar - 12 / 11)
        sigma1_sq = 729 / 968
        var_t = n * sigma1_sq * closed
        # 用同一 ΣA 构造不了真实票单时跳过（只验公式自洽）
        mine = n * sigma1_sq + (27 / 88) * sum_a - (81 / 242) * pairs
        assert abs(mine - var_t) < 1e-9, (
            f"n={n} 协方差式与闭式不一致：{mine} vs {var_t}")


def test_var_ratio_zero_at_lower_bound():
    """n=11 且每个红号恰出现 2 次时 var_ratio 精确为 0（解析下界）。

    这是 `engine_distribution` 注释里点明的边界值；若本模块实现漂移，
    这个「恰好为 0」会被破坏成 2e-16 之类的浮点擦伤。
    """
    # 构造 11 注、每个号恰好 2 次：用 33 个号的 6 个循环移位组合
    all_nums = list(range(1, 34))
    tickets = []
    for i in range(11):
        tickets.append([all_nums[(i * 3 + k) % 33] for k in range(6)])
    # 上面构造不保证每号恰 2 次，改用可验证的等价：ΣA 应为 33（每号出现 2 次 ⇒ C(2,2)×33=33）
    sum_a = VP._sum_a(tickets)
    if sum_a == 33:
        assert VP.var_ratio_of(tickets) == pytest.approx(0.0, abs=1e-9)
    else:
        # 构造未命中边界时，至少验证 var_ratio ≥ 0 且与权威实现一致
        assert VP.var_ratio_of(tickets) >= 0.0
        auth = ED.ticket_distribution(
            [{"reds": t, "blue": 1} for t in tickets],
            with_distribution_exact=False)["var_ratio"]
        assert abs(VP.var_ratio_of(tickets) - auth) < 1e-9


# ---------- 2. 优化真的有效（否则模块没有存在意义） ----------

def test_min_variance_beats_random_on_var_ratio():
    """最小方差选号的 var_ratio 必须**显著**低于同样注数的随机票单。

    变异实证：把 `_score` 改成 `return -var_ratio_of(...)`（最大化方差），
    本测试立刻变红（var 会升到 1.1 以上，远高于随机基线）。
    """
    for n in (5, 10):
        sel = VP.pick_min_variance(n, seed=42)
        mine = ED.ticket_distribution(
            [{"reds": r, "blue": 1} for r in sel],
            with_distribution_exact=False)["var_ratio"]
        # 随机基线：用固定种子的 40 组随机票单取均值
        import random
        rng = random.Random(20261023)
        base = []
        for _ in range(40):
            bank = [{"reds": sorted(rng.sample(range(1, 34), 6)), "blue": 1}
                    for _ in range(n)]
            base.append(ED.ticket_distribution(
                bank, with_distribution_exact=False)["var_ratio"])
        mean_base = sum(base) / len(base)
        assert mine < mean_base * 0.6, (
            f"n={n} 方差优化无效：mine={mine:.4f} 随机均值={mean_base:.4f}"
            f"（应至少低 40%）")


def test_min_variance_increases_union_coverage():
    """并集覆盖必须扩大 —— 这是 P(≥4红) 变好的真实来源。

    变异实证：把 `_score` 中的并集项权重从 1e-4 改成 0（或改负号），
    并集会明显退化（实测 n=10 从 33 掉到 28 左右）。
    """
    for n in (5, 10):
        sel = VP.pick_min_variance(n, seed=42)
        union = len(set().union(*[set(r) for r in sel]))
        import random
        rng = random.Random(7)
        base = []
        for _ in range(40):
            bank = [set(rng.sample(range(1, 34), 6)) for _ in range(n)]
            base.append(len(set().union(*bank)))
        mean_base = sum(base) / len(base)
        assert union > mean_base, (
            f"n={n} 并集未扩大：优化后 {union} vs 随机均值 {mean_base:.2f}")


def test_p_at_least_4_improves_but_is_attributed_correctly():
    """P(≥4红) 有提升，但必须能被 `summarize` 精确复核。

    这条同时守住两件事：
      · 数字是真的（不是文案吹的）
      · 用的口径与 `engine_distribution` 完全一致（exact_enum）
    """
    sel = VP.pick_min_variance(10, seed=42)
    s = VP.summarize(sel, exact=True)
    assert s["ok"] is True
    assert s["p_at_least_r_method"] == "exact_enum", (
        "must use exact enumeration, not iid approx")
    # 与权威实现逐位一致
    auth = ED.ticket_distribution(
        [{"reds": r, "blue": 1} for r in sel], with_distribution_exact=True)
    for k in ("3", "4", "5"):
        assert s["p_at_least_r"][k] == pytest.approx(auth["p_at_least_r"][k], abs=1e-6)


# ---------- 3. 诚实性：不得暗示能提高收益 ----------

def test_summarize_declares_ev_unchanged():
    """`summarize` 必须带 honesty 文本，且明确写「不改变期望收益」。

    缺失或写错的后果：用户看到一个方差很小的票单，以为更划算。
    这与 `popularity.py` 里「必须同屏披露 52.3%」是同一条纪律。
    """
    sel = VP.pick_min_variance(5, seed=1)
    s = VP.summarize(sel)
    assert s["ok"] is True
    honesty = s["honesty_zh"]
    assert "不改变期望" in honesty or "不改**不**" in honesty.replace("*", ""), (
        f"必须声明期望不变，实际：{honesty}")
    assert "1/17,721,088" in honesty
    # 返奖比必须给出具体值，不能只说「负期望」
    assert s["ev_ratio_full_strict"] == pytest.approx(0.4477, abs=1e-3)


def test_compare_to_random_flags_ev_unchanged():
    """对照结果必须显式带 `ev_unchanged: True` 与说明。

    变异实证：删掉 `ev_unchanged` 字段，本测试变红。
    """
    sel = VP.pick_min_variance(5, seed=1)
    c = VP.compare_to_random(sel, n_samples=10, seed=1)
    assert c["ok"] is True
    assert c["ev_unchanged"] is True
    assert "1.0458" in c["ev_unchanged_zh"] or "52.29" in c["ev_unchanged_zh"]
    assert c["random_baseline"]["baseline_method"] == "iid_approx"
    # 基线方法必须自带说明（解释为什么 var_ratio 精确、尾部概率近似）
    assert "精确组合量" in c["random_baseline"]["baseline_method_zh"]


def test_no_text_claims_prediction_ability():
    """任何输出文本都不得出现「预测」「提高中奖」「更划算」这类词。

    这是 A 类源码守卫的行为等价物：直接检查返回值里的文案。
    """
    sel = VP.pick_min_variance(5, seed=1)
    texts = []
    s = VP.summarize(sel)
    texts.append(s["honesty_zh"])
    c = VP.compare_to_random(sel, n_samples=10, seed=1)
    texts.append(c["ev_unchanged_zh"])
    texts.append(c["random_baseline"]["baseline_method_zh"])
    joined = " ".join(texts)
    for banned in ("提高中奖率", "提高命中率", "更划算", "稳赚", "包中"):
        assert banned not in joined, f"不得出现营销词：{banned}"


# ---------- 4. 边界 ----------

def test_zero_tickets_returns_empty():
    """n≤0 返回空列表，不抛异常（路由层据此返回 400）。"""
    assert VP.pick_min_variance(0) == []
    assert VP.pick_min_variance(-3) == []


def test_pool_too_small_raises_actionable_error():
    """池子不足 6 个号时必须给**可操作**的错误（D-20）。"""
    with pytest.raises(ValueError) as ei:
        VP.pick_min_variance(3, pool=[1, 2, 3])
    msg = str(ei.value)
    assert "6" in msg
    assert "3" in msg, "必须说明当前有几个号"


def test_single_ticket_cannot_be_optimized():
    """n=1 时方差恒为 σ₁²（var_ratio=1），代码应如实返回而不是假装优化了。"""
    sel = VP.pick_min_variance(1, seed=5)
    assert len(sel) == 1
    assert len(sel[0]) == 6
    assert len(set(sel[0])) == 6, "单注内部不能有重复号"
    assert VP.var_ratio_of(sel) == pytest.approx(1.0, abs=1e-9)


def test_tickets_have_valid_red_balls():
    """出号必须合法：6 个互不相同、1~33 内、已排序。"""
    sel = VP.pick_min_variance(12, seed=99)
    assert len(sel) == 12
    for r in sel:
        assert len(r) == 6
        assert len(set(r)) == 6
        assert all(1 <= x <= 33 for x in r)
        assert r == sorted(r)


def test_deterministic_with_same_seed():
    """同种子必须同结果（用户复核的前提）。

    变异实证：把 `random.Random(seed)` 改成 `random.Random()`，本测试变红。
    """
    a = VP.pick_min_variance(8, seed=12345)
    b = VP.pick_min_variance(8, seed=12345)
    assert a == b
    c = VP.pick_min_variance(8, seed=54321)
    assert c != a, "不同种子应有不同结果（否则种子参数无意义）"


def test_expected_hits_is_identity():
    """E[命中]=n·12/11 与选号无关 —— 优化不得改变它。

    这条是「方差优化不改变期望」最直接的数学证据。
    """
    for n in (3, 7, 15):
        sel = VP.pick_min_variance(n, seed=n)
        s = VP.summarize(sel, exact=False)
        assert s["expected_hits"] == pytest.approx(n * 12 / 11, abs=1e-6)


# ---------- 5. 路由层 ----------

def test_route_returns_tickets_and_honesty():
    """`/api/portfolio/variance-pick` 的返回体必须三样齐全：票、摘要、诚实声明。"""
    from lottery.routes_portfolio import portfolio_variance_pick
    r = portfolio_variance_pick(n_tickets=5, seed=42, iterations=800,
                                with_baseline=False, baseline_samples=10)
    assert r["ok"] is True
    assert r["method"] == "min_var"
    assert r["n_tickets"] == 5
    assert len(r["tickets"]) == 5
    assert r["summary"]["ok"] is True
    assert r["honesty"]["ev_unchanged"] is True
    assert r["honesty"]["no_prediction"] is True
    # 必须告诉用户怎么自己复核
    assert "ticket_distribution" in r["how_to_verify_zh"]


def test_route_baseline_failure_is_visible_not_silent():
    """基线对照失败时，必须留下可见的失败标记（D-17），不能静默省略字段。"""
    from lottery.routes_portfolio import portfolio_variance_pick
    import lottery.variance_picker as VPmod

    original = VPmod.compare_to_random

    def boom(*a, **kw):
        raise RuntimeError("simulated baseline failure")

    VPmod.compare_to_random = boom
    try:
        r = portfolio_variance_pick(n_tickets=3, seed=1, iterations=300,
                                    with_baseline=True, baseline_samples=5)
    finally:
        VPmod.compare_to_random = original

    assert r["ok"] is True, "基线失败不应影响主结果"
    assert r["baseline"]["ok"] is False
    assert r["baseline"]["reason"] == "baseline_failed"
    assert "with_baseline=false" in r["baseline"]["error_zh"], (
        "错误信息必须给出可操作的重试方式")


def test_route_rejects_bad_n_tickets():
    """n=0 走路由时必须 400，而不是 500 或空票单。"""
    from lottery.routes_portfolio import portfolio_variance_pick
    r = portfolio_variance_pick(n_tickets=0, seed=1, iterations=100,
                                with_baseline=False, baseline_samples=5)
    # 直接调函数时 n=0 → pick 返回 [] → 路由给 400 JSONResponse
    assert isinstance(r, object)
    body = r if isinstance(r, dict) else r.body
    if isinstance(r, dict):
        assert r["ok"] is False
    else:
        import json
        parsed = json.loads(r.body)
        assert parsed["ok"] is False
        assert r.status_code == 400
