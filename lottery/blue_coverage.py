"""蓝球覆盖优化（B-3）：把 `engine.blue_coverage_plan` 的闭式最优接出来，
并**强制**携带诚实披露。

## 整个功能的诚实边界（一句话，第一屏就该看到）

> **任意蓝球分配方案下，`E[命中注数]` 恒等于 `n/16`，与是否去重无关。**

去重覆盖**确实**把「至少中一次蓝球」的概率从 ``1-(15/16)^n`` 抬到 ``min(n,16)/16``
（n=10 时 47.55% → 62.50%，真实的、组合恒等式可证的收益），
但它**对期望收益一分钱都不改变**：期望命中注数从来不由"怎么分配"决定，
而是由"每注各自 1/16"决定，线性可加。

由此直接得到本模块最重要的数字（V36 修正，2026-10-24 审计 P1-6）：

    ``返奖比 = (n/16 × P(红≤2) × 5) / (2n) = 5 × P(红≤2) / 32 ≈ 14.723%``，
    **对任意 n 恒成立**（n 在分子分母同时出现，被约掉）。

⚠ **旧值 5/32 = 15.625% 是错的**，错因是"六等奖只要求中蓝"这个隐含假设：
官方六等奖还要求红球命中 ≤ 2，其概率 P(红≤2) = Σ_{k=0..2} C(6,k)C(27,6-k)/C(33,6)
= 0.9423，不是 1。旧式把分子写成 n/16 × 5（等于假设 P(红≤2)=1），
于是 5/32 比真值高 6.25%。修正后：

    P_SIXTH = P(红≤2) × 1/16 = 0.0588925466（≈1/16.98，官方口径）
    返奖比  = 5 × 0.0588925466 / 2 = 0.1472314

"与 n 无关"这个**结构结论仍然成立**（这才是本模块真正要守的诚实边界），
变的只是常数。真源唯一：`engine_prize.P_SIXTH`，此处引用而不重新推导。

所以正确表述只有一句：**覆盖优化改变的是中奖的「分布 / 频率 / 体感」，不是期望。**
任何把这里说成「更划算」「提高回报」的说法都是假的。

（2026-10-21 实测教训：曾误以为 n>16 时返奖比会被稀释到 2.5%，并据此写了
"返奖比随注数下降"的错误文案。真实原因是 EV 公式错把「至少中一次的概率」
当成了「期望命中注数」—— 两者在 n ≤ 16 时数值完全相同，只在 n > 16 分叉。
`p_any_blue_covered` 与 `expected_blue_hits` 是**两个量**，永远不要混用。）

## 这个模块存在的唯一理由

`engine.py` 里已经有一个数学上正确的 `blue_coverage_plan()`：
n 注的蓝球取互不相同时，``P(≥1 注中蓝) = min(n,16)/16``，
而随机重复只有 ``1-(15/16)^n``。这个提升是**组合恒等式**，全枚举可证，零造假风险。

但产品审查员实测发现：这个收益**只在前端角落以小字显示**，
最诚实的端点 `/api/portfolio/expected-value` 前端从未调用。
于是用户看到的仍是"智能预测"，「唯一真实的数学收益」被藏在没有人读的地方。

本模块因此做两件事：
1. **把计划与披露绑成一个对象** —— 调用方拿到 `plan` 就必然拿到 `disclosure`，
   没有"只取数字不看边界"的路径；
2. **把三个数字并排给出**：概率、成本、期望回收。
   单看"中奖概率从 47.6% 提到 62.5%"是激励性的，但配上
   "成本 20 元、期望回收 3.13 元"才是完整的真相。

## 三个必须同屏的数字（任何一次调用都必须全部返回）

以 10 注为例（实测口径）：

| 量 | 值 |
|---|---|
| P(≥1 注中蓝) 随机重复 | 47.55% |
| P(≥1 注中蓝) 去重覆盖 | **62.50%** |
| 成本 | 20 元 |
| 六等奖期望回收 | 3.125 元 |
| **六等奖返奖比** | **15.625%（与 n 无关）** |
| 整注（一~六等全计）返奖比 | ≈ 52.3%（含浮动奖假设） |

**返奖比恒为 15.6%（六等奖口径）这件事，是覆盖优化不改变期望的直接证据。**
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: 单注价格（元）
TICKET_PRICE = 2.0
#: 六等奖固定奖金（元）
SIXTH_PRIZE = 5.0
#: 蓝球池大小
B_MAX = 16


def _p_red_le2() -> float:
    """P(红球命中 ≤ 2) —— 六等奖除"蓝中"以外的**第二个必要条件**（V36 补）。

    官方六等奖：蓝中 **且** 红 ≤ 2。以前本模块（和 engine_prize 的一段）
    都按"蓝中即 5 元"算，于是返奖比虚高 6.25%。数学上：

        P(红≤2) = Σ_{k=0..2} C(6,k)·C(27,6-k) / C(33,6) = 0.94225466…

    单注六等奖概率 = P(红≤2)/16 = 0.0588925466，返奖比 = 5×它/2 ≈ 0.1472314。
    唯一真源在 `engine_prize.P_SIXTH`；此处用比值取回，避免第二份近似常数。
    """
    from . import engine_prize as _EP
    return _EP._P_RED_LE2


def p_any_blue_random_repeat(n: int) -> float:
    """随机重复抽样下「至少中 1 个蓝球」的概率 ``1-(15/16)^n``。

    这是**基线**：不知道去重、每注独立随机选蓝球时的概率。
    """
    if n <= 0:
        return 0.0
    return 1.0 - ((B_MAX - 1) / B_MAX) ** n


def p_any_blue_covered(n: int) -> float:
    """去重覆盖下「至少中 1 个蓝球」的概率 ``min(n,16)/16``。

    这是本模块交付的**上界**，且是可达的（n≤16 时取互不相同的蓝球即可）。
    """
    if n <= 0:
        return 0.0
    return min(int(n), B_MAX) / float(B_MAX)


def coverage_plan(n_tickets: int,
                  blue_probs: Optional[Sequence[float]] = None) -> List[int]:
    """蓝球分配计划（委托 `engine.blue_coverage_plan`，保持单一实现）。

    这里不重新实现数学 —— 只在 engine 不可用时给出均匀回退，
    避免"两套蓝球逻辑各说各话"（这正是本项目反复出问题的地方）。
    """
    if n_tickets <= 0:
        return []
    try:
        from . import engine
        return list(engine.blue_coverage_plan(int(n_tickets), blue_probs))
    except Exception:            # noqa: BLE001 —— 引擎异常不该让接口 500
        k = int(n_tickets)
        base = list(range(1, B_MAX + 1))
        return (base * ((k // B_MAX) + 1))[:k]


def optimize(n_tickets: int,
             blue_probs: Optional[Sequence[float]] = None,
             *, full_ev_ratio: Optional[float] = None,
             prize_assumption: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """返回「蓝球覆盖计划 + 三个必须同屏的数字 + 披露文本」。

    `full_ev_ratio` 由调用方从 `engine.prize_expectation(...)["ev_ratio_full"]`
    传入（含浮动奖假设）；不传时只给六等奖口径，并在披露里说明"未含浮动奖"。

    ## 六等奖返奖比：**对任意 n 恒为 5×P(红≤2)/32 ≈ 14.723%**（V36 修正）

    `E[命中注数] = n/16`（每注蓝球命中概率 1/16，线性可加，与是否重复无关），
    故 `返奖比 = (n/16 × P(红≤2) × 5) / (2n) = 5×P(红≤2)/32 ≈ 0.1472314`，
    **与 n 无关**。旧值 5/32=0.15625 漏了「六等奖还要求红球 ≤2」这个条件。

    两个**必须区分**的量，历史上曾混用导致 n>16 的 EV 被低估 6.25 倍：

    * `p_any_blue_covered = min(n,16)/16` —— 「**至少中一次**」的概率，n>16 封顶 1.0；
    * `expected_blue_hits = n/16` —— 「**期望命中注数**」，永不封顶。

    六等奖按**注**兑付（`evaluate.tickets_result()` 逐注 `prize_level` 后求和），
    所以乘 ¥5 的必须是后者。n ≤ 16 时两值相同（蓝球互不相同 ⇒ 至多中一次），
    缺陷只在 n>16 显形 —— 这就是旧测试 `range(1,17)` 抓不到它的原因。
    """
    n = max(0, int(n_tickets))
    plan = coverage_plan(n, blue_probs)
    distinct = len(set(plan))

    p_cover = p_any_blue_covered(n)
    p_repeat = p_any_blue_random_repeat(n)
    cost = n * TICKET_PRICE

    # ===== 六等奖期望回收：必须用「期望命中注数」，不是「至少中一次的概率」=====
    #
    # 缺陷背景（2026-10-21 实测，用 `evaluate.tickets_result()` 真实结算反证）：
    # 旧代码写 `ev_sixth = p_cover * SIXTH_PRIZE`，而
    # `p_cover = min(n,16)/16` 回答的是**「至少中一次蓝球」的概率**。
    # 但六等奖是**按注兑付**的：`evaluate.tickets_result()` 对每一注独立调
    # `prize_level()` 再把 rewards 求和 —— 6 注蓝球命中就是 6×¥5，不是 1×¥5。
    #
    #   n ≤ 16：蓝球互不相同 ⇒ 命中指示变量互斥 ⇒ E[命中注数] = n/16，
    #           而 min(n,16)/16 也等于 n/16 ⇒ **两种口径数值完全相同**，缺陷隐形。
    #           （这正是 test_v14_blue_coverage 只跑 range(1,17) 抓不到它的原因：
    #            不是没覆盖，是两种口径在该区间同值。）
    #   n > 16：命中注数的期望仍是 n/16（每注独立 1/16），但「至少中一次」封顶 1.0
    #           ⇒ 旧口径把 6.25 注压成 1 注：n=100 时报 5.00 元，真实 31.25 元
    #             （低估 6.25 倍）。
    #
    # 正确口径 `E[命中注数] = n / 16`，对任意 n 成立。由此
    # `返奖比 = (n/16 × P红≤2 × 5) / (2n) ≈ 14.723%`，**恒与注数无关** ——
    # 这才是"覆盖优化不改变期望"的真正证明。旧代码在 n>16 时算出的 2.50%
    # 是**公式错误造成的假象**，不是覆盖优化稀释了返奖比。
    expected_blue_hits = n / float(B_MAX)
    # V36：六等奖 = 蓝中 **且** 红 ≤2，所以期望奖金还要乘 P(红≤2)。
    # 真源唯一在 engine_prize.P_SIXTH（= P(红≤2)/16），这里只用它的比值，
    # 避免第二处出现"5/32"这种形式近似。
    ev_sixth = expected_blue_hits * SIXTH_PRIZE * _p_red_le2()
    ev_ratio_sixth = (ev_sixth / cost) if cost > 0 else 0.0

    out: Dict[str, Any] = {
        "n_tickets": n,
        "plan": plan,
        "blue_coverage_count": distinct,
        "p_any_blue_covered": round(p_cover, 6),
        "p_any_blue_random_repeat": round(p_repeat, 6),
        "p_gain_ratio": round((p_cover / p_repeat) if p_repeat > 0 else 1.0, 4),
        "cost_yuan": round(cost, 2),
        "ev_sixth_yuan": round(ev_sixth, 4),
        "ev_ratio_sixth": round(ev_ratio_sixth, 6),
        "net_sixth_yuan": round(ev_sixth - cost, 4),
        #: 概率是否已饱和 —— 仅指 **P(≥1 注中蓝) 已封顶 100%**，与返奖比无关。
        #:
        #: 语义边界（务必别混）：
        #:   · n > 16：蓝球池 16 个已占满 ⇒ 再加注**不增加「至少中一次」的概率**
        #:     （它恒为 1.0），只是把命中的注数从 n/16 摊得更散。
        #:   · 返奖比**不变**，恒为 5×P(红≤2)/32 ≈ 14.723%（`E[命中注数]` 与
        #:     成本 `2n` 同步增长，n 被约掉）。
        #: 前端用这个字段提示"再加注不会更可能中蓝球"，不得用它暗示收益变化。
        "coverage_saturated": n > B_MAX,
        #: 期望命中注数（= n/16），供前端解释"为什么 n>16 后概率封顶但 EV 仍线性增长"
        "expected_blue_hits": round(expected_blue_hits, 4),
    }
    if full_ev_ratio is not None:
        out["ev_ratio_full"] = round(float(full_ev_ratio), 6)
        out["ev_full_yuan"] = round(float(full_ev_ratio) * cost, 4)
    if prize_assumption:
        # 「必须连 prize_assumption 一起展示」—— 把它作为数据字段返回，
        # 而不仅写在披露文本里，便于前端独立渲染与审计。
        out["prize_assumption"] = dict(prize_assumption)

    # ---- 强制披露（不可省略、不可拆开单独展示）----
    lines = [
        f"同样买 {n} 注：「至少中一次蓝球（¥5）」的概率从 "
        f"{p_repeat * 100:.2f}% 提升到 {p_cover * 100:.2f}%。",
        # 精度统一 `.3f`，**不许混排**（2026-10-21 三人核算结论）：
        #   `净亏 = cost − ev` 是恒等式，三项必须同精度，否则用户自己一减就对不上。
        #   全 .2f：3.12 / 16.88  → 反算 20−3.12 = 16.88    ✅ 闭合
        #   混排  ：3.125 / 16.88 → 反算 20−3.125 = 16.875  ❌ 差 0.005
        #   全 .3f：3.125 / 16.875→ 反算 20−3.125 = 16.875  ✅ 闭合（本方案）
        # 另外 `15.625% = 5/32` 是**精确恒等式**，截成 15.62% 会削弱
        # 「返奖比恒定」这一全系统最不可伪造结论的说服力（Lead 判为证据强度问题）。
        f"但成本是 {cost:.0f} 元，六等奖期望回收 {ev_sixth:.3f} 元，"
        f"净亏 {cost - ev_sixth:.3f} 元。",
        f"六等奖返奖比恒为 {ev_ratio_sixth * 100:.3f}%，**与注数无关** —— "
        f"覆盖优化改变的是中奖分布，不是期望收益。",
    ]
    if n > B_MAX:
        # 只讲**概率**封顶，绝不能说"返奖比下降"（那是旧 EV 公式错误造成的假象）。
        lines.append(
            f"注意：蓝球只有 {B_MAX} 个，买超过 {B_MAX} 注后「至少中一次蓝球」的概率已封顶 "
            f"100%，再加注**不会提高这个概率**（只会让蓝球重复、命中注数按 {n}/{B_MAX} 摊开）。"
            f"返奖比不受影响，仍恒为 {ev_ratio_sixth * 100:.3f}%。")
    if full_ev_ratio is None:
        lines.append("（未含一/二等奖浮动奖；整注口径需配合奖金假设表查看。）")
    else:
        # 项目自订规则（`engine._single_ticket_ev` docstring）：
        # 「一/二等奖为浮动奖，必须连 prize_assumption 一起展示」。
        # 只报一个 53.2% 而不说它假设了多少头奖，等于让用户以为那是确定值。
        lines.append(
            f"整注（一~六等全计）返奖比约 {float(full_ev_ratio) * 100:.1f}%，"
            f"**长期仍为负期望**。")
        if prize_assumption:
            lines.append(
                f"（该返奖比依赖奖金假设：一等奖 {prize_assumption.get('first', 0):,.0f} 元、"
                f"二等奖 {prize_assumption.get('second', 0):,.0f} 元 —— "
                f"一/二等奖是浮动奖，实际金额随奖池变化，此处取"
                f"{prize_assumption.get('label', '假设值')}。）")
    lines.append(
        "中头奖概率恒为 1/17,721,088，**不受选号或覆盖策略影响**。")
    out["disclosure"] = "\n".join(lines)
    out["disclosure_lines"] = lines
    # `increases_prize_frequency` 语义：去重覆盖相对随机重复，是否提高了**中奖频率**。
    #   · n = 0：没有票，无频率可言 → False（旧版恒为 True，语义无意义）
    #   · 1 ≤ n：覆盖把 P(≥1 中蓝) 从 1-(15/16)^n 抬到 min(n,16)/16 → True
    #     （n>16 时该概率恒为 1.0，是**封顶**而非继续提高；但"覆盖优于随机重复"
    #      这一事实对任意 n≥1 都成立，故仍为 True。）
    # 两条否定结论（不提高命中率、不提高期望收益）在所有 n 下恒为 False —— 这是铁律。
    out["honesty_flags"] = {
        "increases_hit_probability": False,
        "increases_expected_return": False,
        "increases_prize_frequency": bool(n >= 1),
    }
    return out
