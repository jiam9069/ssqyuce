"""V10-03 票单分布（S8）—— 解析计算，**不是模拟**。

## 迁移说明（D-02 拆分 S8）

内容从 `lottery/engine.py` 原 704–1220 行**逐字迁移**，逻辑未改一行。

## 为什么整块迁移、不拆散（方案 §4.1 的明确判断）

「数学 + 三个边界」共用一组常量，且 `_single_ticket_ev` 直接读
`PRIZE_CASH_FIXED` / `TICKET_COST` / `B_MAX`。**拆散会让同一套数学分散到多处，
比留在一个文件里更难核对。**

## 诚实边界（原文保留，不得在迁移中丢失）

* 全部闭式，**不是蒙特卡洛**；测试用 `C(33,6)=1,107,568` 全枚举交叉验证。
* `E[命中注数]` 恒为 `n/16`，与是否去重**无关**；
  蓝球覆盖只提高「至少中一次」的概率，**不改变期望收益**。
* 一/二等奖是浮动奖，本模块只接受**调用方显式传入的假设常量**，
  并在返回体以 `jackpot_assumption` 原样标注，绝不假装知道真实金额。
"""
from __future__ import annotations

import itertools
import random
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config        # noqa: F401
from .engine_base import R_MAX, B_MAX, TICKET_COST  # noqa: F401  纯常量，无循环

# 这两个常量已随 S7 迁到 `engine_prize.py`（真源）。
# **从真源直接取，不经薄壳 `E.`** —— 它们是常量不是函数，薄壳只做转发，
# 绕一层没有收益，反而让依赖关系更难读。
from .engine_prize import JACKPOT_ASSUMPTION, PRIZE_CASH_FIXED  # noqa: F401
# V36：六等奖概率真源（P_SIXTH）用于 EV_RATIO_SIXTH，避免第二份近似常数。
from .engine_prize import P_SIXTH as _EP_P_SIXTH  # noqa: F401

# `_expand_dan_tuo` 要用硬约束过滤（胆拖展开时必须剔除违反约束的组合）。
# 硬约束已随 S1 迁到 `engine_constraints.py`（真源），从这里直取。
from .engine_constraints import (  # noqa: F401
    pass_constraints,
    _weighted_sample,
    ensemble_mass,
)




# ===========================================================================
#
# 为什么放在 engine.py 而不是 rigor.py：rigor.py 刻意不 import engine（避免
# ``battle → engine → rigor → engine`` 循环），而票单层语义（奖金假设、注数、
# blue_coverage_plan）本来就都住在 engine.py。见 docs/V10 §3.4 落点变更说明。
#
# ---------- 数学（全部闭式；测试用 C(33,6)=1,107,568 全枚举交叉验证） ----------
#
# 单注红球命中数 K ~ Hypergeometric(N=33, K=6, n=6)：
#     E[K]   = 6*6/33 = 12/11                = 1.0909090909…
#     Var(K) = 729/968                       = 0.7530991736…   (sd ≈ 0.867813)
# 第 i、j 两注的红球交集大小为 A_ij，则
#     Cov(K_i, K_j) = (27/176)*A_ij - 81/484
# 因此整组命中总数 T = Σ_i K_i：
#     Var[T]    = n*σ₁² + (27/88)*Σ_{i<j}A_ij - (81/242)*C(n,2)
#     var_ratio = Var[T]/(n*σ₁²)
#               = 1 + (n-1)*(11/54)*(Ā - 12/11),   Ā = Σ_{i<j}A_ij / C(n,2)
#
# ---------- 三个必须记住的边界 ----------
# 1. **var_ratio 恒 ≥ 0，下界恰为 0**：n=11 且每个红号恰出现 2 次时 Var[T]=0。
#    它**不是**普通"方差比"，而是协方差结构的线性泛函；UI 必须同时给出
#    var_T 与 nσ₁² 两个正数，不能只显示 var_ratio。
# 2. Ā 的中性值是 **12/11 ≈ 1.0909**（此时 var_ratio=1），**不是 0**；
#    Ā=0 只在 n ≤ 5（5 注互不相交）时可行，6 注起需要 36 个号 > 33。
# 3. 改变票单分布**不改变期望**（EV 恒为 n × 单注 EV），只改变
#    （a）命中总数方差、（b）「至少一注命中 ≥r 红」的概率。任何把目标写成
#    「提高期望」的实现都是错的。

RED_PER_TICKET = 6
SIGMA1_SQ = 729.0 / 968.0                       # 单注红球命中数方差
E_HITS_PER_TICKET = RED_PER_TICKET * RED_PER_TICKET / float(R_MAX)   # 12/11
VAR_RATIO_SLOPE = 11.0 / 54.0                   # 闭式斜率 (n-1)*(11/54)
REDUNDANCY_NEUTRAL = 12.0 / 11.0                # Ā 中性值：var_ratio == 1
REDUNDANCY_MAX = float(RED_PER_TICKET)          # Ā 的上界（两注完全相同）
# 六等奖返奖比（V36 修正，2026-10-24 审计 P1-6）。
# 旧值 `(5/16)/2 = 0.15625` 隐含"蓝中即得 5 元"，漏了官方六等奖的
# 「红球命中 ≤ 2」这一条件。正确值：5 × P(红≤2)/16 / 2 = **0.1472314**。
# 唯一真源 `engine_prize.P_SIXTH`（= P(红≤2)/16），此处只做换算，不再自推常数。
EV_RATIO_SIXTH = (5.0 * _EP_P_SIXTH) / TICKET_COST

# 奖金假设表（一/二等奖为浮动奖，只能给"假设常量"，必须随结果一起展示）
PRIZE_ASSUMPTION_TABLE_A = {
    "label": "V4 假设表 A",
    "first": 5_000_000.0,          # 一等奖浮动奖假设（V4_ACCURACY_DIAGNOSIS.md:177-187）
    "second": 150_000.0,           # 二等奖浮动奖假设
    "note": "一/二等奖为浮动奖，此处为假设常量（V4 口径），非真实金额",
}
PRIZE_ASSUMPTION_TABLE_B = {
    "label": "V4 假设表 B（= engine.JACKPOT_ASSUMPTION）",
    "first": float(JACKPOT_ASSUMPTION["first"]),
    "second": float(JACKPOT_ASSUMPTION["second"]),
    "note": "一/二等奖为浮动奖，此处为历史均值近似假设常量，非真实金额",
}

_POPCOUNT_LUT: Optional[np.ndarray] = None
_ALL_RED_MASKS: Optional[np.ndarray] = None
_EXACT_TAIL_CACHE: Dict = {}
_EXACT_TAIL_CACHE_MAX = 64


def _popcount_lut() -> np.ndarray:
    global _POPCOUNT_LUT
    if _POPCOUNT_LUT is None:
        _POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(2048)], dtype=np.uint8)
    return _POPCOUNT_LUT


def _popcount33(x: np.ndarray) -> np.ndarray:
    """33 位掩码的 popcount（拆 3 段 11 位查表，避免建 2^33 表）。"""
    lut = _popcount_lut()
    return (lut[x & 2047] + lut[(x >> 11) & 2047] + lut[(x >> 22) & 2047]).astype(np.int8)


def _red_mask(reds) -> int:
    m = 0
    for r in reds:
        m |= 1 << (int(r) - 1)
    return m


def _all_red_masks() -> np.ndarray:
    """C(33,6) 全部可能开奖红球组合的位掩码（1,107,568 项 ≈ 8.9MB，惰性构建一次）。"""
    global _ALL_RED_MASKS
    if _ALL_RED_MASKS is None:
        import itertools
        combos = np.array(list(itertools.combinations(range(R_MAX), RED_PER_TICKET)), dtype=np.int64)
        masks = np.bitwise_or.reduce(np.left_shift(np.int64(1), combos), axis=1)
        _ALL_RED_MASKS = masks.astype(np.uint64)
    return _ALL_RED_MASKS


def _validate_tickets(tickets) -> List[Dict]:
    """规范化为 ``[{"reds": [...6 个升序唯一红号...], "blue": int}]``；非法输入抛 ValueError。"""
    out: List[Dict] = []
    for idx, t in enumerate(tickets or []):
        reds = t.get("reds") if isinstance(t, dict) else None
        if not isinstance(reds, (list, tuple)) or len(reds) != RED_PER_TICKET:
            raise ValueError(f"第 {idx} 注 reds 必须是 {RED_PER_TICKET} 个红号，实际 {reds!r}")
        try:
            rs = sorted(int(x) for x in reds)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {idx} 注 reds 含非整数：{reds!r}") from exc
        if len(set(rs)) != RED_PER_TICKET:
            raise ValueError(f"第 {idx} 注红号有重复：{rs}")
        if rs[0] < 1 or rs[-1] > R_MAX:
            raise ValueError(f"第 {idx} 注红号越界（应在 1..{R_MAX}）：{rs}")
        blue = t.get("blue") if isinstance(t, dict) else None
        try:
            b = int(blue) if blue is not None else 0
        except (TypeError, ValueError):
            b = 0
        out.append({"reds": rs, "blue": b})
    return out


def _pairwise_stats(reds_list: List[List[int]]) -> Dict:
    """两两交集统计：ΣA_ij、Ā、并集覆盖、冗余度。O(n²)，热路径可用。"""
    n = len(reds_list)
    sets = [set(r) for r in reds_list]
    union: set = set()
    for s in sets:
        union |= s
    sum_a = 0
    for i in range(n):
        si = sets[i]
        for j in range(i + 1, n):
            sum_a += len(si & sets[j])
    n_pairs = n * (n - 1) // 2
    abar = (sum_a / n_pairs) if n_pairs else 0.0
    return {
        "n": n,
        "sum_a": int(sum_a),
        "n_pairs": int(n_pairs),
        "abar": float(abar),
        "union_coverage": int(len(union)),
    }


def _variance_from_abar(n: int, abar: float) -> float:
    """Var[T] = n·σ₁² + (27/88)·ΣA − (81/242)·C(n,2)。"""
    sum_a = abar * (n * (n - 1) // 2)
    return (n * SIGMA1_SQ + (27.0 / 88.0) * sum_a
            - (81.0 / 242.0) * (n * (n - 1) // 2))


def _var_ratio(n: int, abar: float) -> float:
    """var_ratio = 1 + (n−1)·(11/54)·(Ā − 12/11)；恒 ≥ 0（下界 0）。"""
    if n <= 0:
        return 0.0
    return 1.0 + (n - 1) * VAR_RATIO_SLOPE * (abar - REDUNDANCY_NEUTRAL)


def _variance_stats(n: int, sum_a: int) -> Dict[str, float]:
    """用**有理数精确**算 ``Var[T]`` 与 ``var_ratio``，再转 float。

    为什么不用浮点闭式：两个边界值会在浮点下"擦伤"——n=11（每号恰 2 次）时
    ``Var[T]`` 恰为 0，但闭式 ``1+(n−1)(11/54)(Ā−12/11)`` 算出来是 2.2e-16；
    Ā=12/11 时 var_ratio 恰为 1，闭式是 1.0000000000000002。测试要求
    ±1e-9 甚至精确 0/1，所以这里走 ``fractions.Fraction``（n≤33，开销可忽略）。
    """
    from fractions import Fraction
    n = int(n)
    if n <= 0:
        return {"var_T": 0.0, "var_ratio": 0.0, "n_sigma1_sq": 0.0}
    s1 = Fraction(729, 968)
    pairs = n * (n - 1) // 2
    var_exact = n * s1 + Fraction(27, 88) * int(sum_a) - Fraction(81, 242) * pairs
    n_sig_exact = n * s1
    ratio_exact = var_exact / n_sig_exact
    # 交叉核对闭式（1+(n−1)(11/54)(Ā−12/11)）与协方差式一致
    abar_exact = Fraction(int(sum_a), pairs) if pairs else Fraction(0)
    closed = 1 + Fraction(n - 1) * Fraction(11, 54) * (abar_exact - Fraction(12, 11))
    if abs(float(closed) - float(ratio_exact)) > 1e-9:      # pragma: no cover - 纯防御
        raise AssertionError(f"var_ratio 闭式与协方差式不一致：{closed} vs {ratio_exact}")
    return {
        "var_T": float(var_exact),
        "var_ratio": float(ratio_exact),
        "n_sigma1_sq": float(n_sig_exact),
    }


def _prob_at_least_r_single(r: int) -> float:
    """单注命中 ≥r 红球的超几何概率。"""
    from math import comb
    total = comb(R_MAX, RED_PER_TICKET)
    return sum(comb(RED_PER_TICKET, k) * comb(R_MAX - RED_PER_TICKET, RED_PER_TICKET - k)
               for k in range(int(r), RED_PER_TICKET + 1)) / total


def _poisson_binomial_at_least_one(ps) -> float:
    """泊松二项：**独立**事件下 P(至少一个发生) = 1 − Π(1−p_i)。

    仅当把 n 注的红球命中数当成相互独立时才成立。真实票单里票间是相关的
    （两注相同时完全相关 → 真值是 p 而不是 1−(1−p)^n；两注不相交时是负相关），
    所以它只能当**热路径近似**用，精确值走 :func:`ticket_distribution` 的全枚举。
    """
    q = 1.0
    for p in ps:
        q *= (1.0 - float(p))
    return 1.0 - q


def _iid_tail_probs(tickets: List[Dict], r_values) -> Dict[int, float]:
    """独立（iid）近似下的 P(≥1 注命中 ≥r 红)，每注边际概率相同。"""
    n = len(tickets)
    out: Dict[int, float] = {}
    for rv in r_values:
        p1 = _prob_at_least_r_single(rv)
        # 每注边际概率相同 → 泊松二项退化为 1−(1−p)^n
        out[int(rv)] = _poisson_binomial_at_least_one([p1] * n)
    return out


def _exact_red_tail(tickets: List[Dict]) -> Dict:
    """全枚举 C(33,6) 求精确 P(≥1 注命中 ≥r 红) 与 Var[T]；按票组哈希缓存。

    复杂度 O(C(33,6)·n)（n=10 约 0.1~0.2s），只在评估/报表/测试用；
    ``predict_next`` 的热路径不要调用它（用 ``with_distribution_exact=False``）。
    """
    key = tuple(sorted(tuple(int(x) for x in t["reds"]) for t in tickets))
    cached = _EXACT_TAIL_CACHE.get(key)
    if cached is not None:
        return cached
    if not key:
        cached = {"p_ge": {r: 0.0 for r in (1, 2, 3, 4, 5, 6)}, "mean_T": 0.0,
                  "var_T": 0.0, "n_draws": 0}
    else:
        masks_all = _all_red_masks()
        max_hits = np.zeros(masks_all.shape[0], dtype=np.int8)
        total_hits = np.zeros(masks_all.shape[0], dtype=np.int16)
        for reds in key:
            h = _popcount33(masks_all & np.uint64(_red_mask(reds)))
            np.maximum(max_hits, h, out=max_hits)
            total_hits += h
        cached = {
            "p_ge": {r: float((max_hits >= r).mean()) for r in (1, 2, 3, 4, 5, 6)},
            "mean_T": float(total_hits.mean()),
            "var_T": float(total_hits.var()),
            "n_draws": int(masks_all.shape[0]),
        }
    if len(_EXACT_TAIL_CACHE) >= _EXACT_TAIL_CACHE_MAX:
        _EXACT_TAIL_CACHE.clear()
    _EXACT_TAIL_CACHE[key] = cached
    return cached


def p_at_least_r_mc(tickets: List[Dict], r: int = 4, n_draws: int = 200_000,
                    seed: int = 0) -> Dict:
    """固定种子蒙特卡洛：估计 P(≥1 注命中 ≥r 红)，用于**交叉校验**解析值。

    返回 ``{"p": float, "r": int, "n_draws": int, "seed": int, "stderr": float}``。
    stderr 为标准误 ``sqrt(p(1−p)/n)``——拿它和解析值比较时才有意义：
    固定种子下结果可复现，但**不是**精确值。
    """
    ts = _validate_tickets(tickets)
    r = int(r)
    n_draws = int(n_draws)
    if n_draws <= 0 or not ts:
        return {"p": 0.0, "r": r, "n_draws": max(0, n_draws), "seed": int(seed), "stderr": 0.0}
    rng = np.random.default_rng(int(seed))
    tmasks = [np.uint64(_red_mask(t["reds"])) for t in ts]
    chunk = 100_000
    hit_any = 0
    done = 0
    while done < n_draws:
        c = min(chunk, n_draws - done)
        # 均匀随机取 6 个不同红号：对 33 个均匀 key 取最小的 6 个 ⇒ 均匀 6-子集
        keys = rng.random((c, R_MAX))
        picks = np.argpartition(keys, RED_PER_TICKET, axis=1)[:, :RED_PER_TICKET]
        draw_masks = np.bitwise_or.reduce(
            np.left_shift(np.int64(1), picks.astype(np.int64)), axis=1).astype(np.uint64)
        max_hits = np.zeros(c, dtype=np.int8)
        for tm in tmasks:
            h = _popcount33(draw_masks & tm)
            np.maximum(max_hits, h, out=max_hits)
        hit_any += int((max_hits >= r).sum())
        done += c
    p = hit_any / float(n_draws)
    return {"p": float(p), "r": r, "n_draws": n_draws, "seed": int(seed),
            "stderr": float((p * (1.0 - p) / n_draws) ** 0.5)}


def _single_ticket_ev(jackpot: Optional[Dict] = None, *, sixth: str = "strict") -> Dict:
    """单注期望奖金（元）与各奖级贡献。

    ``sixth="published"``：V4 文档口径 —— 六等奖按「蓝球命中」的**边际**概率 ×¥5
    计（0.0625×5 = ¥0.3125），与高奖级存在重复计数；这是项目自 V4 起对外发表、
    docs/V10 §3.4.5 冻结的 `ev_ratio_full ≈ 0.45675` 的来源（一等 ¥5,000,000 /
    二等 ¥150,000 时单注 ¥0.9135）。
    ⚠ V39 起**默认已改为 ``strict``**：此前默认 published 会让任何"忘记显式传
    sixth 的调用方"拿到一个数学上不成立的重复计数口径（概率合计 0.0707 > 官方
    0.0671，EV 高估 2.01%）。现所有生产路由都已显式传 strict，默认值同步收紧；
    published 保留仅供 `ev_ratio_full` 与 `ev_ratio_full_strict` 并排展示用。
    ``sixth="strict"``：按 ``evaluate.prize_level`` 的**互斥**奖级严格计算
    （六等奖只含「蓝球命中且红球 ≤2」）；同一假设下为 ¥0.8954（0.4477），
    与上面差 ~2% —— 两者都给出，让口径可核对而不是二选一。
    """
    from math import comb
    jp = dict(PRIZE_ASSUMPTION_TABLE_A)
    if isinstance(jackpot, dict):
        for k in ("first", "second"):
            if k in jackpot:
                jp[k] = float(jackpot[k])
        if jackpot.get("label"):
            jp["label"] = str(jackpot["label"])
    total = comb(R_MAX, RED_PER_TICKET)

    def p_r(r: int) -> float:
        return comb(RED_PER_TICKET, r) * comb(R_MAX - RED_PER_TICKET, RED_PER_TICKET - r) / total

    pb = 1.0 / B_MAX
    cash = dict(PRIZE_CASH_FIXED)
    cash[1] = float(jp["first"])
    cash[2] = float(jp["second"])
    by_level = {
        1: p_r(6) * pb,
        2: p_r(6) * (1 - pb),
        3: p_r(5) * pb,
        4: p_r(5) * (1 - pb) + p_r(4) * pb,
        5: p_r(4) * (1 - pb) + p_r(3) * pb,
    }
    if sixth == "strict":
        by_level[6] = (p_r(0) + p_r(1) + p_r(2)) * pb          # 互斥：蓝球命中且红球 ≤2
    else:
        by_level[6] = pb                                        # V4 发表口径：蓝球边际概率
    contrib = {lvl: by_level[lvl] * cash.get(lvl, 0.0) for lvl in range(1, 7)}
    ev = float(sum(contrib.values()))
    return {
        "ev_yuan": ev,
        "ev_ratio": ev / TICKET_COST,
        "by_level_prob": {k: float(v) for k, v in by_level.items()},
        "by_level_contrib_yuan": {k: float(v) for k, v in contrib.items()},
        "prize_cash": {int(k): float(v) for k, v in cash.items()},
        "prize_assumption": (f"{jp['label']}: 一等 ¥{jp['first']:,.0f} / 二等 ¥{jp['second']:,.0f}"
                             f"（{'V4 六等奖按蓝球边际概率' if sixth != 'strict' else '严格互斥奖级'}）"),
        "prize_assumption_detail": jp,
    }


def ticket_distribution(tickets: List[Dict], *,
                        with_distribution_exact: bool = True,
                        r_values=(3, 4, 5),
                        jackpot: Optional[Dict] = None,
                        mc_draws: int = 0,
                        mc_seed: int = 0) -> Dict:
    """**票单分布**：一组注单的解析统计（期望 / 尾部概率 / 方差结构 / EV 口径）。

    这是「票单分布」卡的唯一数据来源（V10-10 前端消费），也是目标函数的度量工具。

    返回（全部为原生 ``float``/``int``，可直接 ``json.dumps``）：

    ==================  ===========================================================
    ``n``               注数
    ``union_coverage``  并集覆盖的红号个数（≤33）
    ``redundancy``      冗余度 Ā = Σ_{i<j}A_ij / C(n,2)，**中性值 12/11 ≈ 1.0909**
    ``redundancy_neutral``  中性值常量（=12/11，供前端并排展示）
    ``expected_hits``   命中总数期望 = n·12/11（与选号无关，是恒等式）
    ``var_T``           命中总数方差（解析，见模块注释）
    ``n_sigma1_sq``     n·σ₁²（与 var_T 一起看：var_ratio 只是二者的商）
    ``var_ratio``       var_T / (n·σ₁²) = 1+(n−1)(11/54)(Ā−12/11)；**恒 ≥ 0，下界 0**
    ``p_at_least_r``    P(≥1 注命中 ≥r 红)，键 "3"/"4"/"5"
    ``p_at_least_r_method``  ``"exact_enum"``（C(33,6) 全枚举）或 ``"iid_approx"``
    ``p_at_least_r_iid``     泊松二项（独立假设）近似值，与上面并排以便看见相关性误差
    ``p_any_blue``      P(≥1 注命中蓝球) = 不同蓝球数/16（blue_coverage_plan 闭式最优）
                        ⚠ 这是**概率**（n>16 封顶 1.0），不是期望注数。
    ``expected_blue_hits``  **期望命中注数** = n/16（与是否去重无关，是恒等式）
                        ⚠ 与 `p_any_blue` 是两个量，永远不要混用。
                        n=20 时：p_any_blue=1.0 而 expected_blue_hits=1.25。
    ``ev_ratio_sixth``  仅六等奖的返奖比 = (5/16)/2 = **0.15625**（与 n 无关）
    ``ev_ratio_full``   整注（一~六等奖全计）返奖比 ≈ **0.45673**，**必须连
                        ``prize_assumption`` 一起展示**
    ``prize_assumption``  奖金假设表（一/二等奖是浮动奖，只能给假设常量）
    ``ev_ratio_full_strict``  同上但按互斥奖级严格算（≈0.4477），口径对照用
    ==================  ===========================================================

    ``with_distribution_exact=True``（默认）：``p_at_least_r`` 走 C(33,6) 全枚举，
    精确值，按票组哈希缓存（首次 n=10 约 0.1~0.2s）。这是本函数**唯一诚实**的口径
    ——泊松二项忽略票间相关（5 注全同时会把 0.0049 估成 0.0243，高估 5 倍）。
    ``False``：用 iid 近似（``p_at_least_r_method="iid_approx"`），热路径用；
    结果里同时给出精确/近似两列，调用方不得把近似值当结论。

    ``mc_draws>0``：额外用固定种子蒙特卡洛交叉校验，结果放 ``p_at_least_r_mc``。
    """
    ts = _validate_tickets(tickets)
    r_values = tuple(int(x) for x in r_values)
    n = len(ts)
    pair = _pairwise_stats([t["reds"] for t in ts])
    abar = pair["abar"]
    vstats = _variance_stats(n, pair["sum_a"])
    var_t = vstats["var_T"]
    n_sig = vstats["n_sigma1_sq"]
    ratio = vstats["var_ratio"]
    # var_ratio 恒 ≥ 0（n=11 且每个红号恰 2 次时恰为 0）；精确有理数下不应出现负值
    if ratio < 0.0:                                          # pragma: no cover - 纯防御
        raise AssertionError(f"var_ratio 不应为负（解析式下界 0）：{ratio}")

    iid_tail = _iid_tail_probs(ts, r_values) if n else {r: 0.0 for r in r_values}
    if with_distribution_exact and n:
        exact = _exact_red_tail(ts)
        tail = {r: float(exact["p_ge"].get(r, 0.0)) for r in r_values}
        method = "exact_enum"
        enum_var_t = float(exact["var_T"])
    else:
        tail = dict(iid_tail)
        method = "iid_approx"
        enum_var_t = None

    blues = [int(t["blue"]) for t in ts if 1 <= int(t["blue"]) <= B_MAX]
    distinct_blue = len(set(blues))
    # `p_any_blue` 是**概率**（≥1 注命中），n>16 封顶 1.0。
    p_any_blue = min(distinct_blue, B_MAX) / float(B_MAX)
    # `expected_blue_hits` 是**期望注数** = n/16，**与是否去重无关**，不封顶。
    #
    # ## 为什么必须两个都给（D-13）
    #
    # `blue_coverage.optimize()` 早就区分了这两个量（`p_any_blue_covered` /
    # `expected_blue_hits`），但本模块首版只给了前者。
    # 下游若拿 `p_any_blue` 当期望用（如 n=20 时算成 1.0 而非 1.25），
    # 会把六等奖期望回收低估 20% —— 这正是 `blue_coverage` 修复前那个
    # 「概率当注数」缺陷（n=100 低估 6.25 倍）的同族形态。
    #
    # 两个量并列给出，让调用方无法「不小心」混用。
    expected_blue_hits = n / float(B_MAX)

    ev_pub = _single_ticket_ev(jackpot, sixth="published")
    ev_strict = _single_ticket_ev(jackpot, sixth="strict")
    ev_b = _single_ticket_ev(dict(PRIZE_ASSUMPTION_TABLE_B), sixth="published")

    out = {
        "n": int(n),
        "union_coverage": int(pair["union_coverage"]),
        "redundancy": float(abar),
        "redundancy_neutral": float(REDUNDANCY_NEUTRAL),
        "redundancy_sum": float(pair["sum_a"]),
        "redundancy_max": float(REDUNDANCY_MAX),
        "expected_hits": float(n * E_HITS_PER_TICKET),
        "sigma1_sq": float(SIGMA1_SQ),
        "n_sigma1_sq": float(n_sig),
        "var_T": float(var_t),
        "var_ratio": float(ratio),
        "var_ratio_note": ("var_T/(nσ₁²) 恒 ≥ 0；下界 0 在 n=11 且每个红号恰出现 2 次时取到，"
                           "中性点 Ā=12/11 时 var_ratio=1。改变分布不改变期望。"),
        "p_at_least_r": {str(r): float(v) for r, v in tail.items()},
        "p_at_least_r_method": method,
        "p_at_least_r_iid": {str(r): float(v) for r, v in iid_tail.items()},
        "p_any_blue": float(p_any_blue),
        "expected_blue_hits": float(expected_blue_hits),
        "blue_coverage_count": int(distinct_blue),
        "ev_ratio_sixth": float(EV_RATIO_SIXTH),
        "ev_ratio_full": float(ev_pub["ev_ratio"]),
        "ev_reward_yuan": float(ev_pub["ev_yuan"]),
        "prize_assumption": ev_pub["prize_assumption"],
        "prize_cash": ev_pub["prize_cash"],
        "ev_ratio_full_strict": float(ev_strict["ev_ratio"]),
        "ev_ratio_full_table_b": float(ev_b["ev_ratio"]),
        "ev_ratio_full_note": ("ev_ratio_sixth 只是六等奖口径（0.15625），整注口径见 "
                               "ev_ratio_full；一/二等奖为浮动奖，必须连 prize_assumption 一起展示"),
    }
    if mc_draws:
        mc = {str(r): float(p_at_least_r_mc(ts, r=r, n_draws=mc_draws, seed=mc_seed)["p"])
              for r in r_values}
        out["p_at_least_r_mc"] = mc
        out["p_at_least_r_mc_detail"] = {k: v for k, v in
                                         p_at_least_r_mc(ts, r=r_values[0] if r_values else 4,
                                                         n_draws=mc_draws, seed=mc_seed).items()}
    if enum_var_t is not None:
        out["var_T_exact_enum"] = float(enum_var_t)
    return out


def _expand_blue_compound(base_reds: List[int], blue_probs: np.ndarray,
                          k: int, rng: random.Random,
                          red_blend: Optional[np.ndarray] = None,
                          blue_blend: Optional[np.ndarray] = None) -> List[Dict]:
    """U2 蓝球复式：红 6 固定 + 蓝 k 个不同 → k 注。

    蓝球按 blue_probs 加权不放回抽样（蓝球覆盖 k 个不同号）。
    """
    k = max(1, min(k, B_MAX))
    blues = _weighted_sample(blue_probs, k, rng)
    out: List[Dict] = []
    for b in blues:
        t = {"reds": list(base_reds), "blue": b, "method": "blend:blue_compound"}
        if red_blend is not None and blue_blend is not None:
            t["confidence"] = min(100.0, max(1.0, round(10 + 0.25 * ensemble_mass(
                red_blend, blue_blend, t["reds"], t["blue"]), 1)))
        else:
            t["confidence"] = 50.0
        out.append(t)
    return out


def _expand_dan_tuo(dan: List[int], blue: int, n_tickets: int,
                    rng: random.Random, red_blend: np.ndarray,
                    ctx: Dict, blue_blend: Optional[np.ndarray] = None) -> List[Dict]:
    """U2 红胆拖：胆 n 个固定，围绕胆从剩余红池抽“拖”补齐 6 红，产出 n_tickets 注。

    胆来自模型红球概率 Top；拖号从去掉胆后的红池按概率加权抽样。
    """
    dan = sorted(set(dan))
    n_dan = len(dan)
    if n_dan >= 6:
        t = {"reds": dan[:6], "blue": blue, "method": "blend:dan_tuo"}
        t["confidence"] = 50.0
        return [t]
    need = 6 - n_dan
    # 剩余红池：除胆外按 red_blend 加权（仅对 remaining 建立权重，长度一致）
    remaining = [x for x in range(1, R_MAX + 1) if x not in dan]
    base_w = red_blend.astype(float).copy()
    for d in dan:
        base_w[d - 1] = 0.0
    rem_w = np.array([base_w[x - 1] for x in remaining], dtype=float)
    out: List[Dict] = []
    seen = set()
    for _ in range(n_tickets * 8):
        if len(out) >= n_tickets:
            break
        if rem_w.sum() <= 0:
            rem_w = np.ones(len(remaining))
        p = (rem_w / rem_w.sum()).tolist()
        picks = set()
        tries = 0
        while len(picks) < need and tries < 200:
            c = rng.choices(remaining, weights=p, k=1)[0]
            picks.add(c)
            tries += 1
        if len(picks) < need:
            continue
        reds = sorted(list(dan) + list(picks))
        if pass_constraints(reds, blue, ctx):
            key = tuple(reds)
            if key in seen:
                continue
            seen.add(key)
            t = {"reds": reds, "blue": blue, "method": "blend:dan_tuo"}
            if blue_blend is not None:
                t["confidence"] = min(100.0, max(1.0, round(10 + 0.25 * ensemble_mass(
                    red_blend, blue_blend, reds, blue), 1)))
            else:
                t["confidence"] = 50.0
            out.append(t)
    return out

