"""V4-C 奖金期望模型（覆盖优化的目标函数）。

## 迁移说明（D-02 拆分 S7）

本文件内容从 `lottery/engine.py` **逐字迁移**（原 1365–1481 行），逻辑未改一行。
拆出的理由：这是「覆盖优化」的目标函数，是被独立引用最多的一块
（`routes_portfolio` / `blue_coverage` / `evaluate` 都直接用）。

## 为什么用 `
`engine` 薄壳用 `__getattr__` 转发，**每次访问都查模块属性**。
若这里写 `from .engine import predict_next`，引用会在 import 时**绑死**，
`monkeypatch.setattr(engine, "predict_next", ...)` 就打不穿 ——
这就是本项目 2026-10-21 记录过的「镜像双源、单跑绿全量红」。
守卫见 `tests/test_v14_engine_split.py::test_no_module_bound_engine_function_references`。

## 诚实边界（迁移时保留原文）

任何「选号算法」都无法改变各奖级概率 —— 它们只由组合数学决定。
本模块的作用是给出**目标函数**，让「覆盖优化」比的是期望值而不是幻觉命中率。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from math import comb        # V36：模块级常量 P_SIXTH 需要在 import 期就可用

from . import config      # noqa: F401  `.` = lottery，与拆前的 engine.py 完全相同
from .engine_base import R_MAX, B_MAX, TICKET_COST  # noqa: F401  纯常量，无循环

# 拆前这些常量是 engine.py 的**模块级全局**，函数里直接裸用（如 `B_MAX`）。
# 迁出后本文件是独立命名空间，裸名会 NameError —— 必须显式从 engine 取。
# 用 `# 后者会在 import 期绑死值，`monkeypatch.setattr(engine, "B_MAX", 8)` 就打不穿。


def _b_max() -> int:
    """取 engine 的 _b_max()。每次现取，保证打桩穿透（见 test_v14_engine_split）。"""
    return B_MAX


def _ticket_cost() -> float:
    """取 engine 的 _ticket_cost()（每注 2 元）。每次现取，理由同上。"""
    return TICKET_COST

PRIZE_CASH_FIXED = {3: 3000.0, 4: 200.0, 5: 10.0, 6: 5.0}   # 固定奖级（元）

# ---- 六等奖的唯一真源（V36，2026-10-24 审计 P1-5/P1-6）----
#
# 官方六等奖条件：**蓝球命中，且红球命中 0/1/2 个**（不是"蓝中即可"）。
# 项目里曾出现两个都错的副本：
#   * `engine_prize` 用"至少中一次蓝的概率"当期望注数（n>16 时低估）；
#   * `engine_distribution.EV_RATIO_SIXTH = (5/16)/2 = 0.15625`（既非官方 0.294463，
#     也非"仅蓝位"的 0.3125）。
# 这里把概率算一次、算对，两处都引用它，避免再次各改一份。
_SIXTH_CASH = 5.0                       # 六等奖固定奖金（元）
_P_RED_LE2 = sum(comb(6, k) * comb(27, 6 - k) for k in (0, 1, 2)) / comb(33, 6)
#: 六等奖中奖概率（≈0.0588925466 = 1/16.98）
P_SIXTH = _P_RED_LE2 * (1.0 / 16.0)

JACKPOT_ASSUMPTION = {
    "first": 6_500_000.0,    # 一等奖浮动奖假设（历史均值近似，非真实值）
    "second": 180_000.0,     # 二等奖浮动奖假设（历史均值近似，非真实值）
    "source": "evaluate.PRIZE_CASH 历史均值近似；一/二等奖为浮动奖，此处仅为假设常量",
}


def expected_prize(reds: List[int], blue: int,
                   jackpot: Optional[Dict[str, float]] = None) -> Dict:
    """单注期望奖金（元）与其奖级概率分解。

    对 6 红 + 1 蓝的单注，枚举「实际开奖的红球命中数 r」的分布并求和：
    实际开奖的 6 个红球是从 33 个中均匀取出的 6 个，预测的 6 个红球中命中 r 个的
    概率服从超几何分布

        P(r) = C(6, r) * C(27, 6-r) / C(33, 6)

    蓝球命中概率恒为 1/16（独立）。于是各奖级概率为 P(r) × p_blue 的组合。

    诚实边界：任何「选号算法」都无法改变 P(r) 与 p_blue——它们只由组合数学决定。
    本函数的作用是给出**目标函数**，让「覆盖优化」比的是期望值而不是幻觉命中率。
    """
    from math import comb
    jp = dict(JACKPOT_ASSUMPTION)
    if jackpot:
        jp.update({k: float(v) for k, v in jackpot.items() if k in ("first", "second")})
    total = comb(33, 6)
    p_blue = 1.0 / _b_max()
    cash = dict(PRIZE_CASH_FIXED)
    cash[1] = float(jp["first"])
    cash[2] = float(jp["second"])
    ev = 0.0
    prob_by_level: Dict[int, float] = {}
    for lvl in range(1, 7):
        prob_by_level[lvl] = 0.0
    prob_by_hit: Dict[int, float] = {}
    for r in range(0, 7):
        p_r = comb(6, r) * comb(27, 6 - r) / total
        prob_by_hit[r] = p_r
        # 与 evaluate.prize_level 完全一致的奖级判定
        if r == 6:
            prob_by_level[1] += p_r * p_blue
            prob_by_level[2] += p_r * (1.0 - p_blue)
        elif r == 5:
            prob_by_level[3] += p_r * p_blue
            prob_by_level[4] += p_r * (1.0 - p_blue)
        elif r == 4:
            prob_by_level[4] += p_r * p_blue
            prob_by_level[5] += p_r * (1.0 - p_blue)
        elif r == 3:
            prob_by_level[5] += p_r * p_blue
            # ⚠ 3 红 + 蓝不中 **不中奖**（V36 修复，2026-10-24 审计 P0-1）。
            #   此处原先写 `prob_by_level[6] += p_r * (1.0 - p_blue)`，把 3+0
            #   当成六等奖，使六等奖概率从 1/17 虚增到 1/9、单注 EV 高估 24.6%，
            #   且与 `evaluate.prize_level`（判 0，正确）自相矛盾——而上一行注释
            #   恰恰写着「与 evaluate.prize_level 完全一致」。注释不能当校验用，
            #   现在由 tests/test_v36_p0_audit_fixes.py 逐 (r, 蓝) 组合对拍钉住。
            pass
        else:
            prob_by_level[6] += p_r * p_blue
    for lvl, p in prob_by_level.items():
        ev += p * cash.get(lvl, 0.0)
    return {
        "expected_reward": ev,
        "cost": _ticket_cost(),
        "ev_ratio": ev / _ticket_cost(),
        "prob_by_level": prob_by_level,
        "prob_by_red_hits": prob_by_hit,
        "blue_hit_prob": p_blue,
        "jackpot_assumption": dict(jp),
        "prize_cash": cash,
    }


def _portfolio_blue_hit_prob(blues: List[int]) -> float:
    """一组蓝球的「至少中 1 个蓝球」概率 = 不同蓝球数 / 16（独立均匀开奖）。"""
    if not blues:
        return 0.0
    return min(len(set(int(b) for b in blues)), _b_max()) / float(_b_max())


def portfolio_expected_value(tickets: List[Dict],
                             jackpot: Optional[Dict[str, float]] = None) -> Dict:
    """整组注单的期望奖金与覆盖指标。

    返回 {expected_reward, cost, ev_ratio, blue_coverage_count,
          blue_hit_prob, prize_prob_by_level, jackpot_assumption}。

    **同注数下期望奖金恒等**：因为每注的红球命中分布都是同一个超几何分布、
    蓝球命中恒为 1/16，n 注的期望严格是单注的 n 倍——覆盖优化**不改变期望**，
    只改变「中奖频率的分布」（例如把「至少中 1 个蓝球」的概率从
    1-(15/16)^n 提到 min(n,16)/16）。这是本项目最重要的诚实边界：
    同样 10 注，中个蓝球的概率从 47.6% 提到 62.5%，但期望返奖比始终是 15.6%。
    """
    tickets = list(tickets or [])
    n = len(tickets)
    one = expected_prize(tickets[0]["reds"] if n else [],
                         int(tickets[0]["blue"]) if n else 1, jackpot=jackpot)
    blues = [int(t["blue"]) for t in tickets]
    distinct = len(set(blues))
    ev = one["expected_reward"] * n
    cost = _ticket_cost() * n
    # 至少中一个蓝球：随机重复 vs 去重覆盖（对照展示）
    p_naive = 1.0 - (1.0 - 1.0 / _b_max()) ** n if n > 0 else 0.0
    p_cover = _portfolio_blue_hit_prob(blues)
    return {
        "tickets": n,
        "expected_reward": ev,
        "cost": cost,
        "net": ev - cost,
        "ev_ratio": (ev / cost) if cost > 0 else 0.0,
        "blue_coverage_count": distinct,
        "blue_hit_prob": p_cover,
        "blue_hit_prob_naive": p_naive,
        # 六等奖返奖比（V36 修复，2026-10-24 审计 P1-5）。
        #
        # 旧写法：`(5.0 * p_cover) / cost`，其中 p_cover = 不同蓝球数/16 是
        # **「至少中一次蓝」的概率**，不是**期望命中注数**。两者在 n ≤ 16 时
        # 数值恰好相同，只在 n > 16 分叉：p_cover 封顶 1.0 而期望注数仍按 n/16
        # 增长。实测 n=1/16/20/100 → 0.15625 / 0.15625 / 0.125 / **0.025**，
        # 即注数越多"返奖比"越假性地趋零。
        # 这与 blue_coverage.py 在 2026-10-21 修掉的是**同一个 bug**（该文件
        # 明确记录了"至少中一次"与"期望命中注数"不可混用的教训），只是当时
        # 没同步到这个副本 —— 本次按 blue_coverage 的口径统一。
        #
        # 口径推导（务必看分母，第一版实现就在这里错过一次）：
        #   期望六等奖注数 = n × P(蓝中) × P(红≤2) = n/16 × P_RED_LE2
        #   期望奖金       = 5 × 上面那一项
        #   返奖比         = 期望奖金 / 成本(2n) = 5 × P_RED_LE2 / 32
        #                    = **0.1472314**（与 n 无关，n 被约掉）
        #   等价的一个注数口径：P_SIXTH × 5 / TICKET_COST —— 同样是 0.1472314。
        #   注意它不等于我在审计报告里写的 0.294463：那个数**漏了除以 2 注成本**
        #   （0.294463 = 5×P_SIXTH，是"每 2 元成本"还是"每 1 元"的差别）。
        #   以本文件的历史口径（返奖比 = 期望奖金 / 成本）为准，正确值是 0.1472314。
        # n 被约掉这一点与 `blue_coverage.py` 的结论一致（"对任意 n 恒成立"），
        # 变的只是常数：0.15625 → 0.1472314。
        "sixth_ev_ratio": (_SIXTH_CASH * _P_RED_LE2 * (n / float(_b_max()))) / cost
                          if cost > 0 else 0.0,        # 全部注单的奖级概率（取并集的上界近似：不同注之间不独立，此处按独立求和）
        "prize_prob_by_level": {k: v * n for k, v in one["prob_by_level"].items()},
        "prize_prob_by_level_note": "按各注独立求和的上界近似，非精确联合分布",
        "jackpot_assumption": one["jackpot_assumption"],
        "prize_cash": one["prize_cash"],
    }
