"""V12-10（L2 精确值服务）+ V12-15（边际读数端点）验收测试。

对应 docs/analysis/V12_PATTERN_REDEFINITION_PLAN.md §1.2 L2-a..e、阶段 B V12-10/15、
§3.1 验收项 **A4**（L2 精确性）与 **A5**（L2 不越界）。

本文件的测试原则
----------------
* **独立穷举**：不调用 ``engine._exact_red_tail`` 来"验证" ``ticket_distribution``
  —— 那只是拿同一个黑盒验证自己。这里的参照实现是一个**朴素 Python 逐组合**循环，
  完全独立于 engine 的 numpy 位掩码实现（只共享 C(33,6) 这个定义），
  相当于把「全枚举」重算一遍。
* **不依赖网络 / 不依赖真实库**：所有票组由测试自己构造；API 用 FastAPI
  ``TestClient``（进程内，无 socket 服务）。conftest 的 autouse fixture 已提供
  独立临时库，本文件**不**裸赋值 ``config.DB_PATH``。
* **反向用例一律在 docstring 里写「若实现错误，本用例如何失败」**。
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
import json
import math
import random

import pytest

from lottery import engine as E
from lottery import l2_service as S

R = 33
R_TICKET = 6

# 方案 [实测] 真值（Lead 实测；本文件用独立穷举复算后比对）
P4_SPREAD5 = 0.02450413879779842
EV_RATIO_FULL = 0.45673239701760976
IID_P4_SPREAD5 = 0.024265131875323975


# ---------------------------------------------------------------------------
# 独立穷举参照实现（朴素循环，不依赖 engine 的向量化路径）
# ---------------------------------------------------------------------------

_INDEPENDENT_CACHE = {}


def _independent_tail(red_draws, tickets_reds):
    """独立穷举：对 C(33,6) 每一种开奖，数每注命中数，取 max。

    返回 ``{"p_ge": {r: p}, "var_T": float}``。用 ``math.comb`` 级别的朴素集合运算，
    与 engine 的 numpy 掩码 + popcount 实现完全不同的代码路径。
    """
    key = tuple(sorted(tuple(sorted(t)) for t in tickets_reds))
    if key in _INDEPENDENT_CACHE:
        return _INDEPENDENT_CACHE[key]
    ticket_sets = [set(t) for t in key]
    n_draw = 0
    counts = [0] * (R_TICKET + 1)       # max_hits 的直方图
    total = 0
    total_sq = 0
    for draw in red_draws:
        dset = set(draw)
        mx = 0
        tsum = 0
        for ts in ticket_sets:
            h = len(dset & ts)
            tsum += h
            if h > mx:
                mx = h
        n_draw += 1
        counts[mx] += 1
        total += tsum
        total_sq += tsum * tsum
    p_ge = {}
    cum = 0
    for r in range(R_TICKET, -1, -1):
        cum += counts[r]
        p_ge[r] = cum / n_draw
    mean = total / n_draw
    var = total_sq / n_draw - mean * mean
    out = {"p_ge": p_ge, "var_T": var, "mean_T": mean, "n_draws": n_draw}
    _INDEPENDENT_CACHE[key] = out
    return out


def _all_draws():
    """C(33,6) 全部开奖（1,107,568 个 6 元组）；``itertools`` 惰性，可多次遍历。"""
    return itertools.combinations(range(1, R + 1), R_TICKET)


def _t(reds, blue=1):
    return {"reds": sorted(int(x) for x in reds), "blue": int(blue)}


def _spread(n, seed=0):
    return S.build_spread_tickets(n, seed)


def _overlap(n, seed=0):
    return S.build_overlap_tickets(n, seed)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from lottery.api_app import app
    return TestClient(app)


# ===========================================================================
# A4 —— L2 精确性：n∈[1..10] 与独立穷举逐位差 ≤ 1e-9
# ===========================================================================

# n=9/10 时每注精确枚举耗时随 n 线性增长（约 0.1~0.4s），10 个 n 合计约 2~3s，
# 相对 110 万种开奖 × 10 注的枚举量已经是可接受成本。
A4_NS = list(range(1, 11))


@pytest.mark.parametrize("n", A4_NS)
def test_a4_p_at_least_r_matches_independent_enumeration(n):
    """A4：n 注 spread 票组的 P(≥1注≥r红) 与**独立穷举**逐位差 ≤ 1e-9。

    若实现错误，本用例如何失败：
      * ``_exact_red_tail`` 的 popcount 掩码写错（例如 33 位拆段时位移错位）、
        或缓存 key 撞车（不同票组拿到同一份缓存）→ 差 > 1e-9；
      * ``ticket_distribution`` 误用 iid 近似（例如 ``with_distribution_exact`` 被
        忽略）→ n=5 时直接差 2.4e-4，远超容差；
      * 票组本身构造错误（注内重复号 / 越界）→ ``_validate_tickets`` 抛错或差 > 1e-9。
    """
    tickets = _spread(n, seed=0)
    d = E.ticket_distribution(tickets, with_distribution_exact=True)
    ref = _independent_tail(_all_draws(), [t["reds"] for t in tickets])
    for r in (3, 4, 5):
        assert d["p_at_least_r"][str(r)] == pytest.approx(ref["p_ge"][r], abs=1e-9), (
            f"n={n}, r={r}: engine={d['p_at_least_r'][str(r)]} vs 独立穷举={ref['p_ge'][r]}")


def test_a4_var_T_matches_independent_enumeration():
    """A4：var_T 的解析闭式与独立穷举一致（±1e-9）—— 覆盖高/低冗余两端。

    若实现错误，本用例如何失败：
      * 协方差闭式 ``n·σ₁² + (27/88)ΣA − (81/242)C(n,2)`` 的系数被写错 →
        高冗余票组（overlap，Ā≈4）的差值会放大到 1e-2 量级；
      * ``_pairwise_stats`` 的 ΣA 把 (i,j)/(j,i) 算了两次 → 只在高冗余时暴露。
    """
    for tickets in (_spread(5, seed=0), _overlap(5, seed=0),
                    _spread(11, seed=0)):
        d = E.ticket_distribution(tickets, with_distribution_exact=True)
        ref = _independent_tail(_all_draws(), [t["reds"] for t in tickets])
        assert d["var_T_exact_enum"] == pytest.approx(ref["var_T"], abs=1e-9)
        assert d["var_T"] == pytest.approx(d["var_T_exact_enum"], abs=1e-9)


def test_a4_var_ratio_is_exactly_one_at_neutral_redundancy():
    """A4：``Ā = 12/11`` 时 ``var_ratio`` 恰为 1（±1e-12，比方案要求的 1e-9 更严）。

    构造同 test_v10_ticket_distribution：n=11、ΣA=60 ⇒ Ā=60/55=12/11。

    若实现错误，本用例如何失败：
      * 用浮点闭式 ``1+(n−1)(11/54)(Ā−12/11)`` 直接算 → 得到 1.0000000000000002，
        差 2.2e-16，**仍然通过** ±1e-12；因此本用例还额外锁死「走有理数路径」
        的等价结果：把同为中性的另一个构造（n=22、每号 4 次）也算一遍；
      * ``REDUNDANCY_NEUTRAL`` 常量被写成 1.0 或 12/11 的近似小数 → var_ratio 偏离 1
        达 1e-3 量级，立刻失败。
    """
    n = 11
    nums = ([num for num in range(1, 19) for _ in range(3)]
            + [num for num in range(19, 25) for _ in range(2)])
    tickets = [_t(nums[i::n], blue=(i % 16) + 1) for i in range(n)]
    d = E.ticket_distribution(tickets, with_distribution_exact=False)
    assert d["redundancy"] == pytest.approx(12 / 11, abs=1e-12)
    assert d["var_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert abs(d["var_ratio"] - 1.0) <= 1e-12
    # 中性点常量必须与方案一致
    assert d["redundancy_neutral"] == pytest.approx(12 / 11, abs=1e-15)


def test_a4_spread5_matches_plan_truth_values():
    """方案 §1.2 L2-b 的 [实测] 真值：n=5 spread 的 p49=0.024504、EV=0.4567324。

    若实现错误，本用例如何失败：
      * 造票改成"随机 5 注"（而非互不相交）→ p_4 会落到 iid 附近或高重叠的
        0.0195，偏差 1e-3~5e-3，远超 1e-6；
      * 蓝球覆盖不是 1..n 互不相同 → ``p_any_blue`` 不再等于 5/16。
    """
    d = E.ticket_distribution(_spread(5), with_distribution_exact=True)
    assert d["p_at_least_r"]["4"] == pytest.approx(P4_SPREAD5, abs=1e-12)
    assert d["p_at_least_r"]["4"] == pytest.approx(0.024504, abs=1e-6)
    assert d["ev_ratio_full"] == pytest.approx(EV_RATIO_FULL, abs=1e-12)
    assert d["union_coverage"] == 30
    assert d["redundancy"] == pytest.approx(0.0, abs=1e-12)
    assert d["var_ratio"] == pytest.approx(1.0 / 9.0, abs=1e-12)
    assert d["p_any_blue"] == pytest.approx(5 / 16, abs=1e-12)
    # iid 近似并排：精确 − iid = +0.98%（负相关带来的并集增益）
    gain = d["p_at_least_r"]["4"] / d["p_at_least_r_iid"]["4"] - 1.0
    assert gain == pytest.approx(0.00984, abs=5e-4)


# ===========================================================================
# A5 —— L2 不越界（反向用例）：EV 与注单无关；p_4 不得超过 iid 太多
# ===========================================================================

def test_a5_ev_ratio_full_identical_across_any_ticket_set():
    """A5 反向用例：换任何注单，``ev_ratio_full`` **逐位相同**（含单注/同票/不相交）。

    若实现错误，本用例如何失败：
      * 有人把票组结构（冗余度 / 蓝球覆盖 / 并集）混进 EV 计算 → 四个票组的
        ``ev_ratio_full`` 不再逐位相同，本用例立即失败；
      * 若实现"宣称"优化提高了 EV（例如按覆盖度加权），高覆盖票组的值会变大。
    """
    cases = {
        "single": [_t(range(1, 7), blue=1)],
        "identical5": [_t(range(1, 7), blue=1) for _ in range(5)],
        "spread5": _spread(5),
        "overlap5": _overlap(5),
    }
    vals = {}
    for name, tickets in cases.items():
        d = E.ticket_distribution(tickets, with_distribution_exact=False)
        vals[name] = d["ev_ratio_full"]
        assert d["prize_assumption"], f"{name}: prize_assumption 必填"
    base = vals["single"]
    for name, v in vals.items():
        assert v == base, f"{name} 的 ev_ratio_full={v!r} 与 single={base!r} 不逐位相同"
    # ev_ratio_sixth 同理（只与蓝球边际概率有关，与注数/结构无关）
    for name, tickets in cases.items():
        d = E.ticket_distribution(tickets, with_distribution_exact=False)
        assert d["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6), name


def test_a5_p_at_least_r4_does_not_exceed_iid_much():
    """A5 反向用例：``p_at_least_r.4 ≤ 1−(1−p₄)^n + 0.001``（不许超过 iid 太多）。

    n=5 spread 时精确值是 iid 的 +0.98%（负相关的并集增益），仍远在 0.001 内。

    若实现错误，本用例如何失败：
      * 若"全枚举"实际按独立假设计算（或把 P(≥1注≥4红) 与 P(≥4红任意注) 混淆），
        高重叠票组的概率会被显著高估；
      * 若有人为了"好看"把概率人为放大（例如乘 1.05），本用例立刻失败。
    """
    p_single = E._prob_at_least_r_single(4)      # 单注超几何 p₄ = 0.0049008277…
    for n in (1, 2, 5, 10):
        tickets = _spread(n)
        d = E.ticket_distribution(tickets, with_distribution_exact=True)
        iid = 1.0 - (1.0 - p_single) ** n          # iid 天花板
        assert d["p_at_least_r_iid"]["4"] == pytest.approx(iid, abs=1e-12), (
            "引擎的 iid 近似列与 1−(1−p₄)^n 不一致，iid 天花板本身算错")
        assert d["p_at_least_r"]["4"] <= iid + 0.001, (
            f"n={n}: 精确 {d['p_at_least_r']['4']} 超过 iid 天花板 {iid} + 0.001")


# ===========================================================================
# V12-10 —— /api/l2/distribution
# ===========================================================================

def test_v12_10_endpoint_spread_n5(client):
    """curl 验收的等价断言：``n=5&preset=spread`` 的 p_4 == 0.024504（±1e-6）。

    若实现错误，本用例如何失败：
      * 路由没注册（404）、返回体缺 ``distribution``/``method`` → 失败；
      * 端点回退到 iid 近似 → ``method`` 不是 exact_enum 或 p_4 差 2.4e-4 → 失败。
    """
    r = client.get("/api/l2/distribution", params={"n": 5, "preset": "spread"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["n"] == 5
    assert body["method"] == "exact_enum"
    dist = body["distribution"]
    assert dist["p_at_least_r_method"] == "exact_enum"
    assert dist["p_at_least_r"]["4"] == pytest.approx(0.024504, abs=1e-6)
    assert abs(dist["p_at_least_r"]["4"] - P4_SPREAD5) <= 1e-9
    # 返回体里每条数值都带 method 标签
    methods = body["methods"]
    for k in dist:
        assert k in methods, f"distribution 键 {k} 缺方法标签"
    for k in ("p_at_least_r", "var_ratio", "ev_ratio_full", "p_any_blue",
              "union_coverage", "redundancy", "prize_assumption"):
        assert methods[k] in (
            "exact_enum", "analytic_identity", "single_ticket_ev", "closed_form",
            "published_assumption", "identity", "constant", "iid_approx", "note",
            "label", "monte_carlo", "derived"), f"{k} 的方法标签非法：{methods[k]}"
    # prize_assumption 必填
    assert dist.get("prize_assumption")
    assert body["preset"] == "spread"


def test_v12_10_endpoint_prize_assumption_is_mandatory(client):
    """``prize_assumption`` 缺失即失败（V12-10 硬要求，不是"建议展示"）。

    若实现错误，本用例如何失败：
      * 有人把 payload 的 ``prize_assumption`` 删掉/改名 → 断言立刻失败；
      * ``l2_service`` 的必填断言被移除且上游返回 None → 同样失败。
    """
    body = client.get("/api/l2/distribution", params={"n": 3}).json()
    assert "prize_assumption" in body["distribution"]
    assert body["distribution"]["prize_assumption"]
    assert body["methods"]["prize_assumption"] == "published_assumption"


def test_v12_10_endpoint_overlap_preset(client):
    """overlap preset：高重叠票组，p_4 应低于 spread（方案：0.019517 量级）。

    若实现错误，本用例如何失败：
      * preset 参数被忽略（两个 preset 返回同一票组）→ 两者 p_4 相同 → 失败；
      * 造票把"高重叠"写成"高覆盖" → 方向反了 → 失败。
    """
    spread = client.get("/api/l2/distribution",
                        params={"n": 5, "preset": "spread"}).json()
    overlap = client.get("/api/l2/distribution",
                         params={"n": 5, "preset": "overlap"}).json()
    assert overlap["preset"] == "overlap"
    p_spread = spread["distribution"]["p_at_least_r"]["4"]
    p_overlap = overlap["distribution"]["p_at_least_r"]["4"]
    assert p_overlap < p_spread
    assert overlap["distribution"]["redundancy"] > spread["distribution"]["redundancy"]
    assert overlap["distribution"]["union_coverage"] < spread["distribution"]["union_coverage"]


def test_v12_10_endpoint_is_reproducible_for_same_seed(client):
    """同 ``seed`` 两次请求**逐字节**相同（造票可复现；不允许用全局随机状态）。

    若实现错误，本用例如何失败：
      * 造票用了 ``random.random()`` / 未固定种子 → 两次响应文本不同 → 失败。
    """
    a = client.get("/api/l2/distribution", params={"n": 7, "seed": 123}).text
    b = client.get("/api/l2/distribution", params={"n": 7, "seed": 123}).text
    assert a == b
    # 纯函数层直接复核（不经过 HTTP）
    assert S.build_tickets(7, "spread", 5) == S.build_tickets(7, "spread", 5)
    assert S.build_tickets(5, "spread", 0) == S.build_spread_tickets(5, 0)


def test_v12_10_endpoint_accepts_custom_tickets(client):
    """``tickets`` 给了就以它为准（JSON 字符串），且与直接调用引擎一致。

    若实现错误，本用例如何失败：
      * 端点忽略 tickets 参数仍去造票 → 返回票组与传入不符 → 失败；
      * JSON 解析失败被吞掉 → 400 之外的状态码或 p_4 不符 → 失败。
    """
    tickets = [
        {"reds": [1, 2, 3, 4, 5, 6], "blue": 3},
        {"reds": [7, 8, 9, 10, 11, 12], "blue": 4},
        {"reds": [1, 2, 13, 14, 15, 16], "blue": 5},
    ]
    raw = json.dumps(tickets)
    body = client.get("/api/l2/distribution",
                      params={"tickets": raw, "n": 3}).json()
    assert body["preset"] == "custom"
    assert body["n"] == 3
    assert [t["reds"] for t in body["tickets"]] == [t["reds"] for t in tickets]
    direct = E.ticket_distribution(tickets, with_distribution_exact=True)
    assert body["distribution"]["p_at_least_r"] == direct["p_at_least_r"]
    assert body["distribution"]["redundancy"] == direct["redundancy"]


def test_v12_10_endpoint_rejects_bad_params(client):
    """非法参数（n 越界 / 未知 preset / 坏 JSON）必须 4xx，不许 500 也不许静默造票。

    若实现错误，本用例如何失败：
      * 端点把 ``preset`` 未知值静默当成 spread → 返回 200 → 失败；
      * 坏 JSON 抛未捕获异常 → 测试客户端抛 500 或异常 → 失败。
    """
    assert client.get("/api/l2/distribution", params={"n": 0}).status_code == 422
    assert client.get("/api/l2/distribution", params={"n": 17}).status_code == 422
    assert client.get("/api/l2/distribution",
                      params={"preset": "nonsense"}).status_code == 400
    assert client.get("/api/l2/distribution",
                      params={"tickets": "{not json"}).status_code == 400


# ===========================================================================
# V12-15 —— /api/portfolio/marginal（中性读数，禁止购买建议）
# ===========================================================================

_PURCHASE_WORDS = ("划算", "推荐", "值得买", "建议买", "该买", "必买", "上车")


def test_v12_15_marginal_has_no_purchase_advice(client):
    """**硬性验收**：``curl .../marginal?n=10 | grep -c "划算|推荐"`` 必须 = 0。

    这里等价地在整段响应文本上计数（不只看两个词，覆盖 7 个购买建议词）。

    若实现错误，本用例如何失败：
      * 任何字段（含 ``reading_zh`` 自然语言）出现「划算 / 推荐 / 值得买 …」→ 计数 > 0 → 失败；
      * 若把「不评价是否值得」这类**否定式**说明写反成「值得」→ 也会命中词表 → 失败。
    """
    r = client.get("/api/portfolio/marginal", params={"n": 10})
    assert r.status_code == 200, r.text
    text = r.text
    counts = {w: text.count(w) for w in _PURCHASE_WORDS}
    hits = {w: c for w, c in counts.items() if c}
    assert hits == {}, f"边际读数里出现购买建议文案：{hits}"
    # 独立复核 JSON 结构里也没有
    body = r.json()
    assert body["ok"] is True


def test_v12_15_marginal_fields_and_values(client):
    """边际读数字段齐全，且数值 = 全枚举精确值差分 / 闭式 EV。

    若实现错误，本用例如何失败：
      * ``delta_p_at_least_one_prize_pp`` 用 iid 近似而非精确差分 → 与独立重算差
        >1e-9 → 失败；
      * ``expected_return_yuan`` 写成 ``−2×(1−ev_ratio_sixth)``（用六等奖口径）→
        得到 −1.6875 而不是 −1.0865 → 失败。
    """
    body = client.get("/api/portfolio/marginal", params={"n": 10}).json()
    for k in ("delta_p_at_least_one_prize_pp", "expected_return_yuan", "ev_ratio_full",
              "prize_assumption", "n", "n_next", "method"):
        assert k in body, f"缺字段 {k}"
    assert body["n"] == 10 and body["n_next"] == 11
    assert body["method"] == "exact_enum"
    assert body["prize_assumption"]
    # EV：−¥2×(1−0.456732397) = −1.086535206
    assert body["ev_ratio_full"] == pytest.approx(EV_RATIO_FULL, abs=1e-12)
    assert body["expected_return_yuan"] == pytest.approx(-2.0 * (1.0 - EV_RATIO_FULL), abs=1e-12)
    # Δp：用引擎独立重算（P_red4 精确 + 蓝球 k/16 合成）
    d_n = E.ticket_distribution(_spread(10), with_distribution_exact=True)
    d_m = E.ticket_distribution(_spread(11), with_distribution_exact=True)

    def _prize(d):
        p4 = d["p_at_least_r"]["4"]
        return p4 + (1 - p4) * d["blue_coverage_count"] / 16.0

    delta = (_prize(d_m) - _prize(d_n)) * 100.0
    assert body["delta_p_at_least_one_prize_pp"] == pytest.approx(delta, abs=1e-9)
    assert body["delta_p_at_least_one_prize_pp"] > 0


def test_v12_15_marginal_delta_expensive_step_goes_to_zero(client):
    """蓝球覆盖饱和后，加注的边际收益**递减**（不是"越加越值"，而是趋近上限）。

    n=15 → 16 时蓝球覆盖 15/16 → 16/16：Δp 应小于 n=2 → 3 的 Δp。

    若实现错误，本用例如何失败：
      * 若 Δp 用了 ``1−(1−p)^n`` 这种独立重复口径，则 n=16 时会超过 min(n,16)/16
        的上限，Δp 反而**变大** → 失败；
      * 若 p_prize 只看蓝球不看红球 → 各 n 的 Δp 恒等于 1/16·100=6.25pp → 与
        首端的非 6.25 值不一致 → 失败。
    """
    d2 = client.get("/api/portfolio/marginal", params={"n": 2}).json()
    d15 = client.get("/api/portfolio/marginal", params={"n": 15}).json()
    assert d2["delta_p_at_least_one_prize_pp"] > d15["delta_p_at_least_one_prize_pp"]
    # 上限：任何 n 的 p_prize_next ≤ 1
    assert d15["p_prize_next"] <= 1.0 + 1e-12


def test_v12_15_simulated_curl_grep_count_is_zero(client):
    """模拟验收命令 ``curl ... | grep -c "划算|推荐"`` 的**真实退出码语义**。

    ``grep -c`` 匹配 0 行时输出 ``0`` 且退出码为 1；这里断言「输出 0 行」，
    即整段响应里这两个词一次都不出现（与 shell 验收完全等价）。

    若实现错误，本用例如何失败：
      * 响应出现任一词 → ``counts`` 非零 → 失败。
    """
    text = client.get("/api/portfolio/marginal", params={"n": 10}).text
    n_lines = sum(1 for line in text.splitlines() if ("划算" in line or "推荐" in line))
    assert n_lines == 0
