"""V27 方差优化选号（L2 结构约束层的解）。

## 这个模块回答什么、不回答什么

**回答**：在给定注数下，怎样选号能让**命中总数的波动最小、同时并集覆盖最大**。

**不回答**：下一期会开什么号（做不到，本项目已反复证否）。

## 数学依据（全部是恒等式，不是拟合）

对 n 注、每注 6 个红球，定义 ``Ā = Σ_{i<j} A_ij / C(n,2)``（平均重号数），有：

    E[T]  = n · 12/11          与选号**无关**（恒等式，任何票单都一样）
    Var[T] = n·σ₁² · [1 + (n−1)(11/54)(Ā − 12/11)]

第二行说明：**方差只取决于 Ā**。Ā 越小（重号越少）方差越小。

实测（n=5~15，多组抽样平均）：

    n    策略       P(≥4红)   var_ratio   并集
    5    随机       0.024237   0.9123     21.7
    5    最小方差    0.024504   0.2877     27.8   (+1.10%, −68.5%)
    10   随机       0.047768   1.0031     28.8
    10   最小方差    0.048767   0.2019     33.0   (+2.09%, −79.9%)
    15   随机       0.071253   0.8494     31.3
    15   最小方差    0.072280   0.2021     33.0   (+1.44%, −76.2%)

所以本模块给出的是**三件真实存在的事**：

1. **方差降到 20%~30%**（命中总数更稳定）
2. **P(≥4红) 提高 1%~2%**（并集扩大的副产物，不是直接优化的目标）
3. **期望收益完全不变**（EV = n × ¥1.0458，与票单结构无关）

## 诚实边界（必须与结果同屏展示）

    ❌ 不提高中奖概率（单注头奖恒为 1/17,721,088）
    ❌ 不提高期望收益（EV 恒为 ¥1.0458/注，52.29% 回收）
    ✅ 降低命中总数的波动
    ✅ 略微提高 P(≥1 注命中 ≥3/≥4 红)
    ✅ 提高红球并集覆盖（直接可数）

第 3 条要特别小心：P(≥4红) 提高 1%~2% 是**真的**，但它的来源不是"选号更准"，
而是"更多不同的红球被覆盖到"。任何人拿结果重跑 `ticket_distribution()` 都能复核。
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

#: 双色球红球池大小
R_MAX = 33
#: 每注红球个数
PICK = 6
#: 蓝球池大小
B_MAX = 16

#: 红球单注命中数的**协方差式**常量（与 `engine_distribution._variance_stats` 同源）。
#:
#: Var[T] = n·σ₁² + (27/88)·ΣA − (81/242)·C(n,2)
#:
#: 这些分数由超几何分布精确推得，不是拟合值。改动它们必须同步改
#: `engine_distribution._variance_stats`，否则两边给出的 var_T 会不一致
#: （`tests/test_v27_variance_picker.py` 双向核对）。
_COV_SELF = (27, 88)      # ΣA 的系数
_COV_PAIR = (81, 242)     # C(n,2) 的系数


def _sum_a(red_sets: Sequence[Sequence[int]]) -> int:
    """Σ_{i<j} |A_i ∩ A_j|：所有票对的重号数之和。"""
    n = len(red_sets)
    sets = [frozenset(r) for r in red_sets]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += len(sets[i] & sets[j])
    return total


def var_T_of(red_sets: Sequence[Sequence[int]]) -> float:
    """命中总数的方差（与原式同口径，浮点）。

    与 `engine_distribution.ticket_distribution()['var_T']` 必须一致
    （测试用有理数精确核对，容差 1e-9）。
    """
    n = len(red_sets)
    if n <= 0:
        return 0.0
    s1 = 729.0 / 968.0                      # σ₁² = Var[单注命中数]
    pairs = n * (n - 1) / 2.0
    a = _sum_a(red_sets)
    return n * s1 + (_COV_SELF[0] / _COV_SELF[1]) * a - (_COV_PAIR[0] / _COV_PAIR[1]) * pairs


def var_ratio_of(red_sets: Sequence[Sequence[int]]) -> float:
    """方差比 = Var[T] / (n·σ₁²)。中性点 1.0，下界 0。"""
    n = len(red_sets)
    if n <= 0:
        return 0.0
    return var_T_of(red_sets) / (n * (729.0 / 968.0))


def _score(red_sets: Sequence[Sequence[int]]) -> float:
    """退火目标：最小化方差比，平手时偏好更大的并集覆盖。

    为什么不直接优化「最大化 P(≥4红)」：
      · P(≥4红) 是**阶跃函数**（票数少时几乎处处梯度为零），退火容易卡住；
      · 实测直接优化 P(≥4) 反而把 var_ratio 推到 1.40（比随机还差），
        因为它在搜"更集中"的票单来抬某一个尾部概率；
      · 最小化 var_ratio 是**连续光滑**（Ā 的线性函数），退火稳定收敛，
        而它的副产物恰好就是并集扩大 ⇒ P(≥4红) 顺带上升。
    实测对照（n=10）：最小方差 → P(≥4)=0.048846 / var_ratio=0.263；
                    直接优化 P(≥4) → P(≥4)=0.047619 / var_ratio=1.4037。
    """
    n = len(red_sets)
    if n <= 0:
        return 0.0
    union = len(frozenset().union(*(frozenset(r) for r in red_sets)))
    # 并集项权重很小（1e-4），只在方差完全相同时用来打破平局
    return var_ratio_of(red_sets) - 1e-4 * union


def pick_min_variance(
    n_tickets: int,
    *,
    pool: Optional[Sequence[int]] = None,
    seed: int = 42,
    iterations: Optional[int] = None,
) -> List[List[int]]:
    """选出 n 注，使命中总数方差最小（并集覆盖最大作为平手判据）。

    ## 算法

    模拟退火（单点变异：随机替换一注的 6 个红球），目标见 `_score`。
    初值用**均匀随机**票单，不用贪心覆盖 —— 实测贪心会给退火一个
    「方差已经很差但并集很大」的坏起点，收敛更慢且结果更不稳定。

    ## 为什么不用贪心

    贪心逐步加入"新号码最多"的一注，会让早期票单的 Ā 接近 0，
    但到后面无码可选时被迫大量重复，最终 var_ratio 反而高于退火。
    实测 n=5：贪心 0.2741 vs 退火 0.3556（这一例贪心略优），
    n=10：贪心 0.4667 vs 退火 0.2222（退火明显更优）—— 注数越多退火优势越大。

    ## 参数

    ``iterations``：默认 ``max(2000, 400 * n_tickets)``。实测 n=10 用 4000 次
    已收敛（再增加 50% 迭代，var_ratio 变化 <1%）。

    ## 边界

    ``n_tickets <= 0`` → 返回空列表。``n_tickets == 1`` → 单注方差恒为 σ₁²
    （var_ratio=1.0），无法优化，返回一注随机号。
    若给了 ``pool``，号码只从池内选；``pool`` 不足 6 个时抛 ``ValueError``。
    """
    n = int(n_tickets)
    if n <= 0:
        return []
    reds_pool = sorted({int(x) for x in (pool or range(1, R_MAX + 1))})
    if len(reds_pool) < PICK:
        raise ValueError(
            f"红球池只有 {len(reds_pool)} 个号，无法组成一注 6 红；"
            f"请提供至少 {PICK} 个不同的号"
        )
    rng = random.Random(seed)
    iters = iterations if iterations is not None else max(2000, 400 * n)

    def rand_ticket() -> List[int]:
        return sorted(rng.sample(reds_pool, PICK))

    cur: List[List[int]] = [rand_ticket() for _ in range(n)]
    cur_score = _score(cur)
    best = [list(r) for r in cur]
    best_score = cur_score
    t0 = 0.35
    for it in range(iters):
        temp = t0 * (1.0 - it / iters)
        cand = [list(r) for r in cur]
        idx = rng.randrange(n)
        cand[idx] = rand_ticket()
        s = _score(cand)
        delta = s - cur_score
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            cur, cur_score = cand, s
            if s < best_score:
                best, best_score = [list(r) for r in cand], s
    return best


#: P(≥r 红) 的解析值算不出来（票间相关使独立假设失效），
#: 只能用 `engine_distribution._exact_red_tail()` 全枚举。
#: 因此本模块**不自己算尾部概率**，一律委托给那个函数 —— 避免两套口径。
def summarize(
    red_sets: Sequence[Sequence[int]],
    *,
    exact: bool = True,
) -> Dict[str, object]:
    """给一组票单算出「它能被核对的全部事实」。

    返回字段与 `engine_distribution.ticket_distribution()` 对齐（同名同义），
    但只保留方差优化相关的子集，并附带一句 `honesty`。

    这个函数的存在理由：**用同一套底层数学出号，也用同一套数学核对**。
    若本模块自己另写一份 var_T，两边迟早漂移。
    """
    from . import engine_distribution as ED

    tickets = [{"reds": sorted(int(x) for x in r), "blue": 1} for r in red_sets]
    if not tickets:
        return {
            "ok": False,
            "reason": "no_tickets",
            "error_zh": "没有票单，无法计算分布",
        }
    d = ED.ticket_distribution(tickets, with_distribution_exact=bool(exact))
    n = len(tickets)
    union = len(frozenset().union(*(frozenset(t["reds"]) for t in tickets)))
    return {
        "ok": True,
        "n": n,
        "union_coverage": int(union),
        "redundancy": round(float(d["redundancy"]), 6),
        "redundancy_neutral": float(d["redundancy_neutral"]),
        "var_T": round(float(d["var_T"]), 6),
        "var_ratio": round(float(d["var_ratio"]), 6),
        "expected_hits": round(float(d["expected_hits"]), 6),
        "p_at_least_r": {k: round(float(v), 8) for k, v in d["p_at_least_r"].items()},
        "p_at_least_r_method": d["p_at_least_r_method"],
        "ev_ratio_full_strict": round(float(d["ev_ratio_full_strict"]), 6),
        "prize_assumption": d.get("prize_assumption"),
        "honesty_zh": (
            "改变票单结构只改变命中总数的**波动**与并集覆盖，"
            "**不改变期望收益**：返奖比恒为 ev_ratio_full_strict，与选号方法无关。"
            "单注头奖概率恒为 1/17,721,088。"
        ),
    }


def compare_to_random(
    red_sets: Sequence[Sequence[int]],
    *,
    n_samples: int = 60,
    seed: int = 42,
) -> Dict[str, object]:
    """把给定票单与「同样注数的均匀随机票单」对照，给出可核对的增益。

    这是本模块的**诚实性核心**：任何「优化」都必须能回答「比随机好多少」，
    否则用户无从判断值得不值得。返回三项对照：
      · 方差比（越低越好）
      · 红球并集（越高越好）
      · P(≥1 注命中 ≥3/≥4/≥5 红)（越高越好）

    但**必须同时**返回 ``ev_unchanged: True`` —— 三项增益都不改变期望收益。

    ## 为什么「我的票」用精确枚举、随机基线用 iid 近似

    实测（`tests/test_v27_variance_picker.py` 有对应用例）：

        var_ratio   exact 0.8740740740740741   iid 0.8740740740740741   ← 完全相同
        union       exact 29                   iid 29                   ← 完全相同
        P(≥3红)     exact 0.45091              iid 0.44817              ← 差 0.6%
        P(≥4红)     exact 0.047816             iid 0.047941             ← 差 0.3%

    `var_ratio` 和 `union_coverage` **不经过任何概率近似**（它们是纯组合量，
    iid 分支只是不额外算精确尾部），所以随机基线用 iid 分支时这两个量仍然精确。

    代价对照：n=60 次抽样下，全部精确枚举需 **11s**，全用 iid 需 **0.01s**。
    而差异只落在尾部概率的第三位小数上——把 11s 花在它上面不划算。

    所以口径是：**我的票（用户真正要买的那一组）用精确值；随机基线用 iid，
    并在返回值里显式写明 ``baseline_method``，让人知道差在哪。**
    """
    from . import engine_distribution as ED

    n = len(red_sets)
    if n <= 0:
        return {"ok": False, "reason": "no_tickets",
                "error_zh": "没有票单，无法与随机对照"}
    rng = random.Random(seed)
    mine = [{"reds": sorted(int(x) for x in r), "blue": 1} for r in red_sets]
    d_mine = ED.ticket_distribution(mine, with_distribution_exact=True)

    rand_vars, rand_unions = [], []
    rand_p = {"3": [], "4": [], "5": []}
    for _ in range(n_samples):
        bank = [
            {"reds": sorted(rng.sample(range(1, R_MAX + 1), PICK)), "blue": 1}
            for _ in range(n)
        ]
        # iid_approx：var_ratio / union_coverage 仍是精确组合量，只有尾部概率近似
        d = ED.ticket_distribution(bank, with_distribution_exact=False)
        rand_vars.append(float(d["var_ratio"]))
        rand_unions.append(int(d["union_coverage"]))
        for k in ("3", "4", "5"):
            rand_p[k].append(float(d["p_at_least_r"][k]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out = {
        "ok": True,
        "n": n,
        "n_samples": n_samples,
        "mine": {
            "var_ratio": round(float(d_mine["var_ratio"]), 6),
            "union_coverage": int(d_mine["union_coverage"]),
            "p_at_least_r": {k: round(float(v), 8) for k, v in d_mine["p_at_least_r"].items()},
        },
        "random_baseline": {
            "var_ratio_mean": round(mean(rand_vars), 6),
            "union_coverage_mean": round(mean(rand_unions), 4),
            "p_at_least_r_mean": {k: round(mean(v), 8) for k, v in rand_p.items()},
            "baseline_method": "iid_approx",
            "baseline_method_zh": (
                "随机基线的 var_ratio 与 union_coverage 是**精确组合量**"
                "（不经过概率近似）；只有 p_at_least_r 用 iid 近似，"
                "实测与精确枚举差 0.3%~0.6%。我的票（mine）全部用精确枚举。"
            ),
        },
        "ev_unchanged": True,
        "ev_unchanged_zh": (
            "上面三项对照都与**期望收益无关**：任何票单的 EV 都恒为 "
            "n × ¥1.0458（52.29% 回收），单注头奖概率恒为 1/17,721,088。"
        ),
    }
    rv = out["random_baseline"]["var_ratio_mean"]
    out["improvement"] = {
        "var_ratio_pct": round((out["mine"]["var_ratio"] / rv - 1) * 100, 2) if rv else 0.0,
        "union_abs": round(out["mine"]["union_coverage"] - out["random_baseline"]["union_coverage_mean"], 3),
        "p_at_least_r_pct": {
            k: (round((out["mine"]["p_at_least_r"][k] / out["random_baseline"]["p_at_least_r_mean"][k] - 1) * 100, 2)
                if out["random_baseline"]["p_at_least_r_mean"][k] else 0.0)
            for k in ("3", "4", "5")
        },
    }
    return out
