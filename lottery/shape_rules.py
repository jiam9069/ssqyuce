"""形态规律（组合恒等约束）—— 扩大「规律」定义的落地（V11）。

背景（用户 2026-10-18 指令）
--------------------------
「继续想办法找更多的规律，而不是死守现有规律」+「扩大规律的定义」。
在 3515 期真实数据上，**预测下一次开奖号码**的统计规律已被 5 轮独立检验证伪
（频率 χ²、91~124 档维度扫描、条件交叉、短窗口延续、CUSUM 延迟）——
数据与均匀随机不可区分（理论上无从预测）。

但**组合数学层面存在恒等约束**：任何一期开奖，其和值/跨度/三区/奇偶/连号
形态都几乎必然落在某个区间内。实测（3515 期 vs C(33,6) 全空间理论抽样）：

    =================  ============  ============  ========
    形态                理论(全枚举)  实测(3515期)   偏差
    =================  ============  ============  ========
    和值均值             102.3         101.0        -1.3%
    和值 5~95 分位      [67, 138]     [66, 136]     一致
    跨度均值             24.3          24.3         0.0%
    奇数个数 2~4 占比     82.5%         ~82%         一致
    =================  ============  ============  ========

这不是"预测规律"，而是**挑选注单的规律**：LLM 在这种约束下生成的注单，形态上
必然"像真实开奖"。这是「用规律经 LLM 分析出号」在数学上唯一诚实的实现。

本模块职责
----------
1. ``shape_constraint_ctx(draws)``：生成硬约束（供 engine.pass_constraints 消费，
   扩展为和值/三区/奇偶/**跨度**/连号 5 维）。
2. ``shape_rules_report(draws)``：可复现的形态规律报告（理论分布 vs 实测吻合度，
   每条给命中率）—— 供前端「形态证据卡」展示。

   ⚠️ **口径（V12-04）**：本模块产出的是 **L2 结构型证据项**，不是「可用规律」。
   L2 的计数语义是「约束满足率」（组合恒等式，不需要样本量、p 值、功效），
   与 L1 的「检出/未检出」**不是同一种计数，禁止相加**（裁决 R-2）。
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import features as F

R_MAX = 33


# ---------------------------------------------------------------------------
# 1. 理论分布（组合数学解析；不依赖任何数据，可复现）
# ---------------------------------------------------------------------------

def _theoretical_reds(n: int = 20000, seed: int = 0) -> List[tuple]:
    """从 C(33,6) 全空间随机抽 n 注作理论基线（与实测同口径比较）。"""
    all_c = list(combinations(range(1, R_MAX + 1), 6))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_c), min(n, len(all_c)), replace=False)
    return [all_c[i] for i in idx]


def _stats_of(reds_list) -> Dict[str, Any]:
    arr = np.array([list(r) for r in reds_list])
    sums = arr.sum(axis=1)
    spans = arr.max(axis=1) - arr.min(axis=1)
    n_odd = (arr % 2 == 1).sum(axis=1)
    z1 = (arr <= 11).sum(axis=1)
    z3 = (arr > 22).sum(axis=1)
    return {
        "sum_mean": float(sums.mean()),
        "sum_p5": float(np.percentile(sums, 5)),
        "sum_p95": float(np.percentile(sums, 95)),
        "span_mean": float(spans.mean()),
        "odd_mean": float(n_odd.mean()),
        "odd_2_4_rate": float(((n_odd >= 2) & (n_odd <= 4)).mean()),
        "z1_mean": float(z1.mean()),
        "z3_mean": float(z3.mean()),
    }


_THEO = None


def theoretical_stats() -> Dict[str, Any]:
    """组合数学理论分布（惰性缓存）。"""
    global _THEO
    if _THEO is None:
        _THEO = _stats_of(_theoretical_reds())
    return dict(_THEO)


# ---------------------------------------------------------------------------
# 1b. 精确通过率（V12-12 / 裁决 R-3）
# ---------------------------------------------------------------------------
#
# 背景：`shape_rules_report` 里 `theoretical_rate` 曾是**手写估计值**
# （sum 0.90 / span 0.95 / zones 0.90 / consec 0.99）。这些数没有来源、
# 无法复现，且与全枚举真值偏差很大（span 真实 0.99985，不是 0.95）。
#
# 本轮「形态约束通过率」还发生过**三方各报一个数**的事故：
#   P6 报 34.12%（误取 `len(arr)<100` 的理论兜底分支，生产永不进入）
#   Lead 报 39.69%（手搓判据时把 top_zones 写成前 5，真实是 top-10）
#   真值 68.2992%（`engine.pass_constraints` 全枚举）
# 因此这里提供**全枚举精确率**，并由 `tests/test_v12_shape_lock.py` 锁死，
# 杜绝再出现「同一个数三个值」。

_ENUM_RATES: Optional[Dict[str, float]] = None
R_MAX_FOR_ENUM = 33


def enumerated_rates() -> Dict[str, float]:
    """C(33,6) 全枚举（1107568 注）下的精确通过率。惰性缓存（约 1~2 秒）。

    返回键：``sum`` / ``span`` / ``odd`` / ``consec`` / ``shape5``。
    ``shape5`` 是五维形态约束（和值 5~95 分位 + 跨度 7~32 + 奇偶 2~4 + 连号≤2）
    的全枚举通过率——注意它**不等于** `engine.pass_constraints` 的生产通过率
    68.2992%（后者还含三区 top-10 与放宽的 odd_range=[1..5]），两者口径不同，
    见 `tests/test_v12_shape_lock.py` 的口径表。
    """
    global _ENUM_RATES
    if _ENUM_RATES is not None:
        return dict(_ENUM_RATES)

    from itertools import combinations
    all_c = combinations(range(1, R_MAX_FOR_ENUM + 1), 6)
    n = 0
    n_sum = 0
    n_span = 0
    n_odd = 0
    n_consec = 0
    n_shape5 = 0
    for c in all_c:
        n += 1
        s = sum(c)
        sp = c[-1] - c[0]
        odd = sum(1 for x in c if x % 2 == 1)
        groups = sum(1 for i in range(5)
                     if c[i + 1] == c[i] + 1 and (i == 0 or c[i] != c[i - 1] + 1))
        in_sum = 67 <= s <= 138
        in_span = 7 <= sp <= 32
        in_odd = 2 <= odd <= 4
        in_consec = groups <= 2
        n_sum += in_sum
        n_span += in_span
        n_odd += in_odd
        n_consec += in_consec
        n_shape5 += (in_sum and in_span and in_odd and in_consec)

    _ENUM_RATES = {
        "n_total": float(n),
        "sum": n_sum / n,
        "span": n_span / n,
        "odd": n_odd / n,
        "consec": n_consec / n,
        "shape5": n_shape5 / n,
    }
    return dict(_ENUM_RATES)



# ---------------------------------------------------------------------------
# 2. 实测统计（3515 期真实数据）
# ---------------------------------------------------------------------------

def _actual_reds(draws) -> List[tuple]:
    return [tuple(sorted(d.get("reds") or [])) for d in draws
            if len(d.get("reds") or []) == 6]


def actual_stats(draws) -> Dict[str, Any]:
    reds = _actual_reds(draws)
    if not reds:
        return {}
    th = theoretical_stats()
    st = _stats_of(reds)
    # 命中率：n 期里有多少期落在理论 5~95 分位区间内
    arr = np.array([list(r) for r in reds])
    sums = arr.sum(axis=1)
    n_in = float(((sums >= th["sum_p5"]) & (sums <= th["sum_p95"])).mean())
    st["sum_in_band_rate"] = n_in
    st["n"] = len(reds)
    return st


# ---------------------------------------------------------------------------
# 3. 约束生成（供 pass_constraints 消费；从**最近期**估计，与理论交叉）
# ---------------------------------------------------------------------------

def shape_constraint_ctx(draws: Sequence[Dict]) -> Dict[str, Any]:
    """5 维硬约束：和值 / 三区 / 奇偶 / **跨度** / **连号**。

    前三维沿用既有 constraint_ctx 口径；新增 span 与连号上限。
    全部从最近 500 期估计（与既有做法一致），并用理论 5~95 分位做兜底：
      实际分位若与理论严重偏离（<10% 期落在带内），回退到理论带 ——
      防止"用当前数据"把约束带歪。
    """
    th = theoretical_stats()
    arr = np.array([list(r) for r in _actual_reds(draws[-500:])])
    if len(arr) < 100:
        return {"sum_min": th["sum_p5"], "sum_max": th["sum_p95"],
                "span_min": 7.0, "span_max": 32.0,
                "top_zones": [(1, 3, 2), (2, 2, 2), (2, 3, 1), (1, 2, 3)],
                "odd_range": [2, 3, 4], "max_consecutive": 2}
    sums = arr.sum(axis=1)
    spans = arr.max(axis=1) - arr.min(axis=1)
    # 三区配置（与既有口径一致）
    z1 = (arr <= 11).sum(axis=1)
    z2 = 6 - z1 - (arr > 22).sum(axis=1)
    z3 = (arr > 22).sum(axis=1)
    from collections import Counter
    zc = Counter((int(a), int(b), int(c)) for a, b, c in zip(z1, z2, z3))
    top_zones = [k for k, _ in zc.most_common(10)]
    # 奇偶范围：**组合恒等约束** —— 理论枚举 6 选奇数的分布下，奇数个数 2~4 合计 82.5%，
    # 1 与 5 各约 5.4%，0 与 6 极罕见。实测（3515 期）与理论一致。
    # 约束取理论常见带 [2,3,4]：这是"合理形态"的本质，不随最近 500 期波动。
    # （若未来数据真的让奇数分布长期偏移，理论带模型本身会翻车——但那是结构变化，
    #   不属于"形态恒等"。）
    odd_range = [2, 3, 4]
    # 连号组数上限（99% 分位）
    consec = np.array([_n_consecutive_groups(r) for r in arr])
    max_consec = int(np.percentile(consec, 99)) if len(consec) else 2

    # 和值带：实际 vs 理论交叉
    sum_p5, sum_p95 = np.percentile(sums, 5), np.percentile(sums, 95)
    in_band = ((sums >= th["sum_p5"]) & (sums <= th["sum_p95"])).mean()
    if in_band < 0.90:
        sum_p5, sum_p95 = th["sum_p5"], th["sum_p95"]     # 理论兜底

    span_p5, span_p95 = np.percentile(spans, 5), np.percentile(spans, 95)
    # 跨度：理论 5~95 分位 ≈ [16, 31]；若最近 500 期实测与理论偏离 >10%（带外率过高），
    # 回退到理论带，保证"组合恒等"而非"跟随历史波动"。
    th_span_rate = ((spans >= 7) & (spans <= 32)).mean()
    if th_span_rate < 0.90:
        span_p5, span_p95 = 7.0, 32.0
    # 更严格：理论 5~95 分位（约 [17,31]）作为组合恒等带；实测带仅作参考
    span_p5, span_p95 = 17.0, 31.0
    return {
        "sum_min": float(sum_p5), "sum_max": float(sum_p95),
        "span_min": float(span_p5), "span_max": float(span_p95),
        "top_zones": top_zones, "odd_range": odd_range,
        "max_consecutive": max(1, max_consec),
    }


def _n_consecutive_groups(reds: Sequence[int]) -> int:
    rs = sorted(reds)
    groups = 0
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and rs[j + 1] == rs[j] + 1:
            j += 1
        if j > i:
            groups += 1
        i = j + 1
    return groups


def check_shape(reds: Sequence[int], ctx: Dict) -> bool:
    """5 维形态校验（与 engine.pass_constraints 兼容扩展）。

    满足全部约束才返回 True —— 这是严格模式 LLM 出号的硬门槛。
    """
    rs = list(reds)
    if len(rs) != 6:
        return False
    s = sum(rs)
    if not (ctx.get("sum_min", -1e9) <= s <= ctx.get("sum_max", 1e9)):
        return False
    sp = max(rs) - min(rs)
    if not (ctx.get("span_min", -1e9) <= sp <= ctx.get("span_max", 1e9)):
        return False
    z1 = sum(1 for r in rs if r <= 11)
    z2 = sum(1 for r in rs if 12 <= r <= 22)
    z3 = 6 - z1 - z2
    if (z1, z2, z3) not in ctx.get("top_zones", [(1, 3, 2)]):
        return False
    odd = sum(1 for r in rs if r % 2 == 1)
    if odd not in ctx.get("odd_range", [2, 3, 4]):
        return False
    if _n_consecutive_groups(rs) > int(ctx.get("max_consecutive", 9)):
        return False
    return True


def explain_shape(reds: Sequence[int], ctx: Dict) -> List[str]:
    """逐条说明该注违反了哪些 L2 约束（人读文本，用于前端与报告）。

    `check_shape` 只给 bool —— 但「不合规」若不说清楚是哪一条，用户无法判断
    该不该信、也不知道该改哪。V28 起每条违规都要能翻译成一句话。

    返回值顺序固定（和值→跨度→三区→奇偶→连号），空列表表示全部合规。
    与 `check_shape` **必须同源**：`check_shape(r,ctx) == (explain_shape(r,ctx) == [])`
    （`tests/test_v28_llm_l2_gate.py` 有对应用例，双向核对）。
    """
    rs = list(reds)
    if len(rs) != 6:
        return [f"需要 6 个红球，实际 {len(rs)} 个"]
    out: List[str] = []
    s = sum(rs)
    lo, hi = ctx.get("sum_min", -1e9), ctx.get("sum_max", 1e9)
    if not (lo <= s <= hi):
        out.append(f"和值={s} 不在 [{lo:.0f}, {hi:.0f}]")
    sp = max(rs) - min(rs)
    lo2, hi2 = ctx.get("span_min", -1e9), ctx.get("span_max", 1e9)
    if not (lo2 <= sp <= hi2):
        out.append(f"跨度={sp} 不在 [{lo2:.0f}, {hi2:.0f}]")
    z1 = sum(1 for r in rs if r <= 11)
    z2 = sum(1 for r in rs if 12 <= r <= 22)
    z3 = 6 - z1 - z2
    if (z1, z2, z3) not in ctx.get("top_zones", [(1, 3, 2)]):
        allowed = sorted(ctx.get("top_zones") or [])
        shown = "、".join("".join(map(str, z)) for z in allowed[:5])
        out.append(f"三区比={z1}-{z2}-{z3} 不在常见集合（如 {shown}…）")
    odd = sum(1 for r in rs if r % 2 == 1)
    rng = ctx.get("odd_range", [2, 3, 4])
    if odd not in rng:
        out.append(f"奇数个数={odd} 不在 {list(rng)}")
    mx = int(ctx.get("max_consecutive", 9))
    if _n_consecutive_groups(rs) > mx:
        out.append(f"连号组数={_n_consecutive_groups(rs)} 超过上限 {mx}")
    return out


# ---------------------------------------------------------------------------
# 4. 形态规律报告（供前端卡片；理论 vs 实测吻合度）
# ---------------------------------------------------------------------------

_ZONES_TOP10_CACHE: Optional[float] = None


def _zones_top10_rate() -> float:
    """三区配置的覆盖率（**动态口径**，V12-12 说明）。

    ⚠️ 性能：这个值**不依赖任何数据**（枚举的是全空间 C(33,6)，取前 10 个
    最常见三区形状），因此是**常量**，必须缓存。初版每次调用都全枚举 110 万注
    （实测约 1.5~2 秒），/api/shape/rules 被前端轮询时会把容器 CPU 打满。
    

    为什么这一维不能像其它四维一样用固定全枚举值：
    生产 `engine.constraint_ctx` 取的是「最近 500 期出现最多的 **top-10** 三区配置」，
    该集合随数据滚动，因此没有唯一的理论值。

    这里的做法：用 C(33,6) 全枚举，统计「落在**全部** 6 种三区形状
    (z1,z2,z3)（z1+z2+z3=6, 各分量≥0）中的前 10 个最常见者」的覆盖率。
    由于三区形状总共只有 C(8,2)=28 种，top-10 覆盖的比例很高，
    这比原先手写的 0.90 有据可依。
    """
    global _ZONES_TOP10_CACHE
    if _ZONES_TOP10_CACHE is not None:
        return _ZONES_TOP10_CACHE

    from itertools import combinations
    shapes: Dict[Tuple[int, int, int], int] = {}
    for c in combinations(range(1, R_MAX_FOR_ENUM + 1), 6):
        z1 = sum(1 for x in c if x <= 11)
        z2 = sum(1 for x in c if 12 <= x <= 22)
        z3 = 6 - z1 - z2
        shapes[(z1, z2, z3)] = shapes.get((z1, z2, z3), 0) + 1
    total = sum(shapes.values())
    top10 = sorted(shapes.values(), reverse=True)[:10]
    _ZONES_TOP10_CACHE = sum(top10) / total
    return _ZONES_TOP10_CACHE


def shape_rules_report(draws: Sequence[Dict]) -> Dict[str, Any]:
    """可复现的形态规律清单 + 理论/实测吻合度。

    每条"证据项"都是组合恒等式，命中率 ≈ 理论覆盖率 ——
    这正是它成立的理由：不是猜中下期，而是保证形态合理。
    **它属于 L2 结构层**（约束满足率），不是 L1 的「可用规律」（检出/未检出）。
    """
    th = theoretical_stats()
    ac = actual_stats(draws)
    # V12-12：所有 theoretical_rate 一律取**全枚举精确值**，不再手写估计。
    # 其中 zones 依赖「最近 500 期 top-10 三区配置」这一动态口径，
    # 故单列 _zones_top10_rate() 说明；其余四维取 C(33,6) 全枚举。
    _enum = enumerated_rates()
    _er = {
        "sum": _enum["sum"],
        "span": _enum["span"],
        "odd": _enum["odd"],
        "consec": _enum["consec"],
        "zones": _zones_top10_rate(),
    }
    return {
        "title_zh": "形态规律（组合恒等约束）",
        "note_zh": ("这些不是【预测下一期】的规律，而是组合数学恒等约束："
                    "任何一期开奖的形态（和值/跨度/三区/奇偶/连号）都几乎必然"
                    "落在这些区间内。LLM 在此约束下生成的注单形态必然合理。"
                    "这是严格模式采用的 **L2 结构约束**（约束满足率），"
                    "不是「发现了能预测下期的规律」。"),
        "n_draws": ac.get("n", 0),
        "enum_rates": _er,
        "rules": [
            {"key": "sum", "name_zh": "和值区间",
             "desc_zh": f"6 红和值应落在 [{th['sum_p5']:.0f}, {th['sum_p95']:.0f}]"
                        f"（理论 5~95 分位，均值 {th['sum_mean']:.1f}）",
             "theoretical_rate": _er["sum"],          # V12-12：全枚举精确率（原手写 0.90）
             "actual_mean": ac.get("sum_mean"),
             "actual_rate": ac.get("sum_in_band_rate"),
             "on": True},
            {"key": "span", "name_zh": "跨度区间",
             "desc_zh": f"跨度（最大-最小）均值 {th['span_mean']:.1f}，"
                        f"几乎总在 [7, 32]",
             "theoretical_rate": _er["span"],         # V12-12：全枚举精确率（原手写 0.95，真实 0.99985）
             "actual_mean": ac.get("span_mean"),
             "on": True},
            {"key": "parity", "name_zh": "奇偶个数",
             "desc_zh": f"奇数个数 2~4 个（理论占比 {_er['odd']:.2%}）",
             # V12-12：改用全枚举精确率 0.825159；原先的 th["odd_2_4_rate"]
             # 来自 20000 次随机抽样（0.8257），与其它四维口径不一致。
             "theoretical_rate": _er["odd"],
             "actual_mean": ac.get("odd_mean"),
             "on": True},
            {"key": "zones", "name_zh": "三区配置",
             "desc_zh": "三区（1-11/12-22/23-33）分布几乎总落在若干固定配置内",
             "theoretical_rate": _er["zones"],        # V12-12：见下 zones 口径说明
             "actual_mean": None,
             "on": True},
            {"key": "consec", "name_zh": "连号组数",
             "desc_zh": "连号组数有上限（极少出现 ≥3 组连号）",
             "theoretical_rate": _er["consec"],       # V12-12：全枚举精确率（原手写 0.99）
             "actual_mean": None,
             "on": True},
        ],
        "theoretical": th,
        "actual": ac,
    }