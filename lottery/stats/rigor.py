"""严格统计检验层（rigor）：只做检验，不产生预测。

本模块的职责边界（务必严格遵守）
--------------------------------
1. **本模块不产生任何预测**。它不挑选号码、不输出概率分布、不参与集成融合。
   它唯一的工作是：把「某个声称的信号是否真实存在」变成一个可被证伪的统计判决。
2. **任何策略在通过本模块的多重检验校正之前，一律不允许被描述为「有效」**。
   未校正的 ``p < 0.05`` 只是候选线索，不是证据；在搜索了 ``n_trials`` 种策略后，
   出现若干 ``p < 0.05`` 属于纯随机的预期结果（family-wise error）。
3. 因此当系统的实际预测准确率「过低」时，正确结论通常不是「模型需要调参」，
   而是「双色球开奖与均匀随机不可区分，任何策略的长期期望命中率都等于理论基线」。
   本模块的价值在于让这个结论无法被自欺地推翻。

数据约定
--------
``draws`` 是**按时间升序**排列的 ``list[dict]``，每项至少含：

- ``reds``: 6 个互不相同的 int，取值 1..33（顺序无关）
- ``blue``: 1 个 int，取值 1..16

与 ``lottery/db.py::_row_to_draw`` / ``lottery/features.py`` 的既有结构一致。

统计口径与常量出处
------------------
- 红球单号出现概率：每期抽 6 个不同号码，故 P = 6/33 = 1/11 ≈ 0.0909。
- 蓝球单号出现概率：P = 1/16 = 0.0625。
- 单注「红球命中数」的期望 = 6 * 6/33 = 36/33 = 1.0909…
  （6 个所选号码 × 每个号码的中签率 6/33；超几何分布均值 k*n/N = 6*6/33）。
- 单注「红球命中数」的方差 = k * (n/N) * (1 - n/N) * (N-k)/(N-1)
  = 6 * (6/33) * (27/33) * (27/32) ≈ 0.7531，sd ≈ 0.8678。
- 「至少中 1 个红球」概率 = 1 - C(27,6)/C(33,6) ≈ 0.6683。
- 数值检验一律用双侧 p 值；判决阈值统一由 ``alpha`` 控制（默认 0.05）。

依赖：仅 numpy / scipy（不写数据库、不读 config、不访问网络）。
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

__all__ = [
    "R_MAX",
    "B_MAX",
    "PICK",
    "BASELINE_MEAN_HITS",
    "BASELINE_SD_HITS",
    "randomness_report",
    "strategy_significance",
    "min_sample_size",
    "evaluate_signal",
    "chi_square_uniformity",
    "sum_distribution",
    "theory_consecutive_rate",
    "bonferroni",
    "benjamini_hochberg",
    "adjust_pvalues",
    "normal_cdf",
    "normal_ppf",
]

R_MAX = 33          # 红球号码上限（1..33）
B_MAX = 16          # 蓝球号码上限（1..16）
PICK = 6            # 每期摇出 6 个红球
Z_95 = 1.959963984540054  # 标准正态 97.5% 分位点，z_{1-alpha/2} (alpha=0.05)

_P_RED = PICK / R_MAX                          # 6/33 = 0.181818…：单个红球每期中签率
_HITS_MEAN = PICK * _P_RED                     # = 36/33 ≈ 1.0909：单注红球命中数期望
_HITS_VAR = (PICK * _P_RED * (1 - _P_RED) * (R_MAX - PICK) / (R_MAX - 1))
_HITS_SD = math.sqrt(_HITS_VAR)                # ≈ 0.8678：单注红球命中数标准差

BASELINE_MEAN_HITS = _HITS_MEAN                # 对外暴露：理论基线 1.0909
BASELINE_SD_HITS = _HITS_SD                    # 对外暴露：理论 sd 0.8678

# 红球和值直方图分组数（用于把 163 个可能和值压缩成 ~82 个自由度可控的箱）
_SUM_N_BINS = 82


# --------------------------------------------------------------------------
# 正态分布辅助（scipy 可用时直接用 scipy，缺省则用自实现近似）
# --------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    """标准正态累积分布函数 Φ(x)（scipy 优先，回退到 erf 实现）。"""
    try:
        return float(stats.norm.cdf(x))
    except Exception:  # pragma: no cover - scipy 存在时不会触发
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(q: float) -> float:
    """标准正态分位函数 Φ⁻¹(q)（scipy 优先，回退到 Acklam 有理逼近）。"""
    if not 0.0 < q < 1.0:
        raise ValueError("q 必须落在开区间 (0, 1) 内")
    try:
        return float(stats.norm.ppf(q))
    except Exception:  # pragma: no cover
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.4e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if q < plow:
            r = math.sqrt(-2 * math.log(q))
            return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                   ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
        if q > phigh:
            r = math.sqrt(-2 * math.log(1 - q))
            return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                    ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
        r = q - 0.5
        s = r * r
        return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
               (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


def _two_sided_p_from_z(z: float) -> float:
    """由 z 值给出双侧 p 值：p = 2 * (1 - Φ(|z|))。"""
    return float(min(1.0, max(0.0, 2.0 * (1.0 - normal_cdf(abs(z))))))


def _verdict(p: float, alpha: float) -> str:
    """统一判决口径：p < alpha 判 deviation，否则判 consistent-with-random。"""
    return "deviation" if p < alpha else "consistent-with-random"


# --------------------------------------------------------------------------
# 多重检验校正
# --------------------------------------------------------------------------

def bonferroni(p_values: Sequence[float]) -> List[float]:
    """Bonferroni 校正：p_adj = min(1, m * p)。控制 family-wise error rate。

    m 为传入的检验个数。当只关心「有没有任何一个显著」时使用，
    代价是极保守；若 m 很大则几乎不可能显著（这正是搜索数十种策略的真实处境）。
    """
    m = len(p_values)
    if m == 0:
        return []
    return [float(min(1.0, p * m)) for p in p_values]


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg (BH) FDR 校正，返回与输入同序的 adjusted p。

    算法：
    1. 升序排序 p_(1) <= ... <= p_(m)；
    2. p_adj_(i) = min_{j >= i} ( p_(j) * m / j )（单调化，从右向左取累计最小）；
    3. 截断到 [0, 1] 并按原顺序返回。

    注意：BH 的 adjusted p 必然 <= Bonferroni 的 adjusted p（j <= m 时 m/j >= 1），
    因此 BH 更容易显著；它控制的是 FDR（假发现比例）而非 FWER。
    """
    m = len(p_values)
    if m == 0:
        return []
    p = np.asarray([float(x) for x in p_values], dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def wilson_lower_bound(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 得分区间的**下界**（小样本比例不做正态近似）。

    ## 为什么这个函数住在 stats 包里（V40 审计收尾）

    2026-10-24 审计 P2-1 的「三/五份重复统计实现」实测为：
    **Wilson 下界/区间 ×4 份 + BH 校正 ×3 份**并行实现，真源与副本
    无一致性守卫 —— 「真源修 bug 副本不跟进」正是 V36 P0-1
    （六等奖）与 P1-5（sixth_ev_ratio）的共同根因。
    本函数从 2026-10-25 起是 Wilson 下界的**唯一公式真源**；
    `backtest._wilson_ci` / `hypothesis._wilson_lower` /
    `decision_skill._wilson_lower` / `rule_lifecycle.wilson_interval`
    全部是薄壳（见各自 docstring），公式只此一份。

    纯标准库实现（math.sqrt），满足 stats 包「不 import 业务模块」纪律。
    n<=0 返回 0.0：与历史各副本的边界语义一致（无样本 → 无证据 → 下界 0），
    tests/test_v10_cost.py::test_wilson_lower_matches_hypothesis_implementation
    以 `(0,0) == 0.0` 钉住。
    """
    n = int(n)
    if n <= 0:
        return 0.0
    p = float(hits) / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson 95%（或按 z 指定置信度）区间，返回 ``(lo, hi)``。

    n<=0 返回 ``(None, None)``：与 rule_lifecycle 历史语义一致
    （「无样本」与「下界为 0」是两回事，调用方据此区分「无法判定」），
    tests/test_v11_rule_lifecycle.py::test_t10_wilson_matches_reference 以
    `wilson_interval(0, 0) == (None, None)` 钉住。
    """
    if n <= 0:
        return None, None
    p = float(k) / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def adjust_pvalues(p_values: Sequence[float]) -> Dict[str, List[float]]:
    """同时给出 Bonferroni 与 BH 两种校正结果（同序）。"""
    return {
        "bonferroni": bonferroni(p_values),
        "benjamini_hochberg": benjamini_hochberg(p_values),
    }


# --------------------------------------------------------------------------
# 理论分布（组合枚举，结果缓存于模块级 lru_cache）
# --------------------------------------------------------------------------

def _sum_distribution() -> Tuple[np.ndarray, np.ndarray, int]:
    """红球和值精确分布：从 C(33,6)=1_107_568 个组合枚举。

    返回 (sums, counts, total)，sums 取值 21..183。
    """
    counts: Dict[int, int] = {}
    for combo in itertools.combinations(range(1, R_MAX + 1), PICK):
        s = sum(combo)
        counts[s] = counts.get(s, 0) + 1
    sums = np.array(sorted(counts), dtype=int)
    cnt = np.array([counts[int(s)] for s in sums], dtype=np.int64)
    return sums, cnt, int(cnt.sum())


_SUM_CACHE: Optional[Tuple[np.ndarray, np.ndarray, int]] = None


def sum_distribution() -> Tuple[np.ndarray, np.ndarray, int]:
    """返回 (sums, counts, total)：红球和值的精确理论分布（精确枚举，非近似）。

    枚举 C(33,6)=1_107_568 个组合约需 1 秒，因此缓存结果。
    """
    global _SUM_CACHE
    if _SUM_CACHE is None:
        _SUM_CACHE = _sum_distribution()
    return _SUM_CACHE


_CONSECUTIVE_CACHE: Optional[float] = None
_CONSECUTIVE_CACHE_ANY: Optional[float] = None
_CONSECUTIVE_CACHE_TAIL: Optional[float] = None


def theory_consecutive_rate() -> float:
    """理论「至少一组连号」概率 = 1 - C(33-6+1, 6)/C(33,6) = 1 - C(28,6)/C(33,6)。

    推导：6 个号码中不含任何连号，等价于从 33-6+1=28 个「间隔单元」中选 6 个
    （stars and bars / 组合插板），故不含连号的组合数为 C(28,6)。
    精确枚举校验值 ≈ 0.65985。
    """
    global _CONSECUTIVE_CACHE
    if _CONSECUTIVE_CACHE is None:
        _CONSECUTIVE_CACHE = 1.0 - math.comb(R_MAX - PICK + 1, PICK) / math.comb(R_MAX, PICK)
    return _CONSECUTIVE_CACHE


def theory_any_same_tail_rate() -> float:
    """理论「6 个红球中至少两个同尾(个位相同)」概率（精确枚举）。"""
    global _CONSECUTIVE_CACHE_ANY
    if _CONSECUTIVE_CACHE_ANY is None:
        hit = 0
        total = 0
        for combo in itertools.combinations(range(1, R_MAX + 1), PICK):
            total += 1
            if len({c % 10 for c in combo}) < PICK:
                hit += 1
        _CONSECUTIVE_CACHE_ANY = hit / total
    return _CONSECUTIVE_CACHE_ANY


def _sum_bin_edges() -> np.ndarray:
    """把和值直方图分成 ``_SUM_N_BINS`` 个箱（等宽），边缘取整数以免样本落空。"""
    sums, _, _ = sum_distribution()
    lo, hi = int(sums[0]), int(sums[-1])
    width = (hi - lo) / _SUM_N_BINS
    edges = [lo + width * i for i in range(_SUM_N_BINS + 1)]
    edges = [round(e, 6) for e in edges]
    return np.array(sorted(set(edges)), dtype=float)


# --------------------------------------------------------------------------
# 单项检验
# --------------------------------------------------------------------------

def chi_square_uniformity(observed: Sequence[float]) -> Dict:
    """卡方均匀性检验：观测频数 vs 「每个类别等概率」的期望频数。

    ``observed`` 为各类别的计数（长度即类别数 m），期望 = 总数/m。
    df = m - 1。当期望频数不足时（总数 < m），返回 unknown 而不是硬算
    （卡方近似在期望频数 < 1 时不可靠）。
    """
    obs = np.asarray([float(x) for x in observed], dtype=float)
    m = obs.size
    total = float(obs.sum())
    if m < 2:
        return {"statistic": 0.0, "df": 0, "p_value": 1.0, "m": int(m),
                "n": int(total), "verdict": "insufficient-data",
                "note": "类别数 < 2，无法做均匀性检验"}
    if total < m:
        return {"statistic": 0.0, "df": m - 1, "p_value": 1.0, "m": int(m),
                "n": int(total), "verdict": "insufficient-data",
                "note": "样本量小于类别数，卡方近似不可靠"}
    exp = np.full(m, total / m)
    stat = float(np.sum((obs - exp) ** 2 / exp))
    df = m - 1
    p = float(stats.chi2.sf(stat, df))
    return {"statistic": stat, "df": df, "p_value": p, "m": int(m),
            "n": int(total), "expected": float(total / m)}


def _pool_sparse_bins(obs: np.ndarray, exp: np.ndarray,
                      min_expected: float = 5.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """把期望频数 < ``min_expected`` 的箱合并成「其他」箱。

    返回 (pooled_obs, pooled_exp)；若合并后有效箱不足 2 个则返回 None。
    观测与期望必须同进同出地累加——只加一侧会让统计量虚高（曾经的实现缺陷）。
    """
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    if obs.size != exp.size or obs.size == 0:
        return None
    keep = exp >= min_expected
    if keep.all():
        return obs, exp
    if keep.sum() >= 2:
        out_obs = np.concatenate([obs[keep], [obs[~keep].sum()]])
        out_exp = np.concatenate([exp[keep], [exp[~keep].sum()]])
    else:
        # 期望过于分散：全部并入单一箱，无法做卡方
        return None
    # 丢弃期望为 0 的箱（理论上不应出现）
    mask = out_exp > 0
    out_obs, out_exp = out_obs[mask], out_exp[mask]
    if out_obs.size < 2:
        return None
    return out_obs, out_exp


def _binned_chi_square(counts: np.ndarray, probs: np.ndarray, n: int,
                       alpha: float) -> Optional[Dict]:
    """把理论分布合并成期望频数 >= 5 的箱后做卡方拟合优度检验。

    合并规则：从左到右累加箱，使每个箱的期望频数 >= 5（经典 Cochran 规则）。
    剩余不足 5 的箱会并入最后一箱。返回 None 表示样本量太小（总期望 < 5*2）。
    """
    idx: List[int] = []
    cur_obs = 0.0
    cur_exp = 0.0
    obs_list: List[float] = []
    exp_list: List[float] = []
    for i in range(len(counts)):
        idx.append(i)
        cur_obs += float(counts[i])
        cur_exp += float(probs[i]) * n
        if cur_exp >= 5.0:
            obs_list.append(cur_obs)
            exp_list.append(cur_exp)
            cur_obs = 0.0
            cur_exp = 0.0
            idx = []
    if idx:  # 尾部残余
        if obs_list:
            obs_list[-1] += cur_obs
            exp_list[-1] += cur_exp
        else:
            obs_list.append(cur_obs)
            exp_list.append(cur_exp)

    k = len(obs_list)
    if k < 2 or n < 2 * 5:
        return None
    obs = np.asarray(obs_list)
    exp = np.asarray(exp_list)
    # 重新归一化期望，消除合并过程的浮点漂移
    exp = exp * (obs.sum() / exp.sum())
    stat = float(np.sum((obs - exp) ** 2 / exp))
    df = k - 1
    p = float(stats.chi2.sf(stat, df))
    return {"statistic": stat, "df": df, "p_value": p,
            "n_bins": k, "verdict": _verdict(p, alpha)}


# --------------------------------------------------------------------------
# 1. randomness_report
# --------------------------------------------------------------------------

def _valid_draws(draws: Optional[List[Dict]]) -> List[Dict]:
    """过滤出结构合法的期次（reds 为 6 个 1..33 的不同整数，blue 在 1..16）。

    保留 ``issue``（若存在），以便 ``evaluate_signal`` 的 ``hit_series``
    能标注每一期预测对应的期号，方便前端定位。
    """
    out: List[Dict] = []
    for d in draws or []:
        try:
            reds = [int(x) for x in d["reds"]]
            blue = int(d["blue"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(reds) != PICK or len(set(reds)) != PICK:
            continue
        if any(not 1 <= r <= R_MAX for r in reds) or not 1 <= blue <= B_MAX:
            continue
        out.append({"reds": sorted(reds), "blue": blue,
                    "issue": d.get("issue")})
    return out


def randomness_report(draws: List[Dict], alpha: float = 0.05) -> Dict:
    """对开奖历史做完整的随机性体检，返回逐项检验结果 + 综合结论。

    检验项（每项含 ``statistic`` / ``df`` / ``p_value`` / ``verdict``）：

    1. ``red_frequency``：红球出现频数均匀性卡方（33 类，df=32）。
    2. ``blue_frequency``：蓝球出现频数均匀性卡方（16 类，df=15）。
    3. ``red_sum``：和值分布拟合优度卡方（理论分布由 C(33,6) 全枚举，df≈81）。
    4. ``odd_count``：奇数个数分布卡方（理论为超几何 Hypergeom(33,6,17)，df=5）。
    5. ``zone_count``：三区（1-11 / 12-22 / 23-33）个数组合分布卡方（df=27）。
    6. ``consecutive``：至少一组连号的观察频率 vs 理论 1-C(28,6)/C(33,6) 的正态近似。
    7. ``same_tail``：至少两个同尾号的观察频率 vs 精确枚举理论值的正态近似。
    8. ``red_repeat``：相邻两期红球重叠个数分布卡方（理论为 Hypergeom(33,6,6)，df=6）。
    9. ``blue_repeat``：蓝球与上期相同的比例 vs 1/16 的正态近似。

    综合 verdict 规则：任一项判 ``deviation`` 则顶层 verdict 为 ``deviation``；
    样本量不足以做任何检验时为 ``insufficient-data``；否则 ``consistent-with-random``。

    **注意顶层 verdict 的假阳性率不是 alpha**：它等价于「m 项检验任一显著」，
    纯随机数据下的假阳性率 = ``1-(1-alpha)^m``。m=9、alpha=0.05 时理论 ≈37%，
    实测 **≈39%**（V10-14 体检，300 组真实随机数据）。因此另有：
      * ``family_wise``: 逐项 Bonferroni/BH 校正结果与理论假阳性率；
      * ``verdict_fwer``: 校正后仍显著才判 deviation 的**家庭化口径**（如需「这套
        数据是否整体偏离随机」的严格结论，请用这个）。
    ``verdict`` 保持原语义不变（向后兼容）。

    参数
    ----
    draws: 时间升序的期次列表（每项含 ``reds`` 与 ``blue``）。
    alpha: 判决阈值，默认 0.05（双侧）。

    返回
    ----
    dict，关键字段：
    - ``verdict``: "consistent-with-random" | "deviation" | "insufficient-data"
    - ``n_draws`` / ``n_valid``: 输入期数与有效期数
    - ``alpha``: 使用的阈值
    - ``tests``: ``{名称: {statistic, df, p_value, verdict, ...}}``
    - ``min_detectable_effect``: dict，见下
    - ``baseline``: 理论基线（单注红球命中数均值/标准差、各单号概率）
    - ``warnings``: 人类可读的诚实提示（中文）
    - ``method``: 方法学备注

    ``min_detectable_effect`` 字段（回答「还要多少期才有结论」）：
    - ``alpha`` / ``power``: 检验的显著性水平与目标功效
    - ``n``: 用于计算的样本量（= 有效期数）
    - ``baseline_mean_hits``: 1.0909（= 6*6/33）
    - ``sd``: 0.8678（超几何标准差）
    - ``delta_hits``: 可在给定功效下检出的最小期望增量（命中数单位）
    - ``relative_lift``: 相对提升，如 0.05 表示 +5%
    - ``hit_rate_pp``: 等价于每期平均多命中多少个百分点（100*delta/6）
    - ``per_draw_probability_gain_pp``: 单注「至少中 1 红」概率的绝对提升（百分点）
    - ``note_zh``: 中文说明

    ``min_detectable_effect`` 公式（双侧检验、目标功效 1-β）：
        δ = (z_{1-α/2} + z_{power}) * sd / sqrt(n)
    其中 ``power`` 是**目标功效**，对应的临界值是 ``z_{power}``（功效 0.8 → 0.8416）。

    边界：空列表、样本量过小、单期数据均不崩溃，返回
    ``verdict="insufficient-data"`` 且相应检验项标注 unknown。
    """
    valid = _valid_draws(draws)
    n_all = len(list(draws)) if draws else 0
    n = len(valid)
    warnings: List[str] = []

    base: Dict = {
        "baseline": {
            "p_single_red": _P_RED,          # 6/33
            "p_single_blue": 1.0 / B_MAX,    # 1/16
            "mean_hits": BASELINE_MEAN_HITS, # 6*6/33
            "sd_hits": BASELINE_SD_HITS,
        },
        "method": ("频率类检验用 Pearson 卡方（等概率期望）；和值用 C(33,6) 精确枚举的"
                   "理论分布；奇偶/三区/重号用超几何或组合精确分布；"
                   "比例类用正态近似两比例检验。判决阈值 alpha，双侧。"),
        "alpha": float(alpha),
        "n_draws": n_all,
        "n_valid": n,
        "tests": {},
        "warnings": warnings,
        "verdict": "insufficient-data",
        "min_detectable_effect": _min_detectable_effect(n, alpha),
    }

    if n < 30:
        warnings.append(
            f"有效期数仅 {n} 期，低于 30 期的检验没有任何功效，"
            "任何「显著」结论都不可信。"
        )
        for name in ("red_frequency", "blue_frequency", "red_sum", "odd_count",
                     "zone_count", "consecutive", "same_tail", "red_repeat",
                     "blue_repeat"):
            base["tests"][name] = {"statistic": None, "df": None, "p_value": None,
                                   "verdict": "insufficient-data", "n": n}
        return base

    red_cnt = np.zeros(R_MAX, dtype=int)
    blue_cnt = np.zeros(B_MAX, dtype=int)
    for d in valid:
        for r in d["reds"]:
            red_cnt[r - 1] += 1
        blue_cnt[d["blue"] - 1] += 1

    # --- 1. 红球频率均匀性 ---
    t = chi_square_uniformity(red_cnt)
    t["verdict"] = _verdict(t["p_value"], alpha)
    base["tests"]["red_frequency"] = t

    # --- 2. 蓝球频率均匀性 ---
    t = chi_square_uniformity(blue_cnt)
    t["verdict"] = _verdict(t["p_value"], alpha)
    base["tests"]["blue_frequency"] = t

    sums_arr, sum_counts, sum_total = sum_distribution()
    sum_probs = sum_counts / sum_total
    edges = _sum_bin_edges()
    observed_sum, _ = np.histogram([sum(d["reds"]) for d in valid], bins=edges)
    theory_sum, _ = np.histogram(sums_arr, bins=edges, weights=sum_probs)
    theory_sum = theory_sum / theory_sum.sum()  # 归一化，消除浮点漂移
    r = _binned_chi_square(observed_sum, theory_sum, n, alpha)
    base["tests"]["red_sum"] = r if r else {
        "statistic": None, "df": None, "p_value": None,
        "verdict": "insufficient-data", "n": n}
    if r:
        base["tests"]["red_sum"]["theory_mean"] = 102.0  # 6 个 1..33 均匀号码之和的均值
        base["tests"]["red_sum"]["theory_sd"] = 21.4243  # 精确枚举值
        base["tests"]["red_sum"]["observed_mean"] = float(np.mean([sum(d["reds"]) for d in valid]))

    # --- 4. 奇数个数分布（理论：Hypergeom(M=33, n=6, N=17)）---
    odd_all = (np.arange(1, R_MAX + 1) % 2 == 1).sum()  # = 17
    odd_theo = np.array([stats.hypergeom.pmf(k, R_MAX, PICK, odd_all) for k in range(PICK + 1)])
    odd_obs = np.zeros(PICK + 1, dtype=float)
    for d in valid:
        odd_obs[sum(1 for x in d["reds"] if x % 2 == 1)] += 1
    odd_p = float(stats.chisquare(odd_obs, odd_theo * n).pvalue)
    odd_stat = float(stats.chisquare(odd_obs, odd_theo * n).statistic)
    base["tests"]["odd_count"] = {
        "statistic": odd_stat, "df": PICK, "p_value": odd_p,
        "verdict": _verdict(odd_p, alpha),
        "observed_mean": float((odd_obs * np.arange(PICK + 1)).sum() / n),
        "theory_mean": float(odd_all * PICK / R_MAX),  # 6*17/33 = 3.0909
    }

    # --- 5. 三区分布（1-11 / 12-22 / 23-33，各 11 个号码）---
    zone_keys = [k for k in itertools.product(range(PICK + 1), repeat=3)
                 if sum(k) == PICK]
    zone_theo_map: Dict[Tuple[int, int, int], float] = {}
    denom = math.comb(R_MAX, PICK)
    for k in zone_keys:
        if max(k) > 11:
            continue
        zone_theo_map[k] = (math.comb(11, k[0]) * math.comb(11, k[1]) *
                            math.comb(11, k[2]) / denom)
    zone_theo = np.array([zone_theo_map[k] for k in zone_keys])
    zone_obs = np.zeros(len(zone_keys), dtype=float)
    zidx = {k: i for i, k in enumerate(zone_keys)}
    for d in valid:
        z1 = sum(1 for x in d["reds"] if x <= 11)
        z2 = sum(1 for x in d["reds"] if 12 <= x <= 22)
        zone_obs[zidx[(z1, z2, PICK - z1 - z2)]] += 1
    # 合并期望频数 < 5 的箱（pooled bin 的观测与期望必须同时累加）
    pooled = _pool_sparse_bins(zone_obs, zone_theo * n, min_expected=5.0)
    if pooled is None:
        base["tests"]["zone_count"] = {
            "statistic": None, "df": None, "p_value": None,
            "verdict": "insufficient-data", "n_bins": 0,
            "theory_zone_mean": float(11 * PICK / R_MAX),
        }
    else:
        obs_k, exp_k = pooled
        exp_k = exp_k * (obs_k.sum() / exp_k.sum())  # 消除浮点漂移
        zone_stat = float(np.sum((obs_k - exp_k) ** 2 / exp_k))
        zone_df = max(1, len(obs_k) - 1)
        zone_p = float(stats.chi2.sf(zone_stat, zone_df))
        base["tests"]["zone_count"] = {
            "statistic": zone_stat, "df": zone_df, "p_value": zone_p,
            "verdict": _verdict(zone_p, alpha),
            "n_bins": int(len(obs_k)),
            "theory_zone_mean": float(11 * PICK / R_MAX),  # 每区期望个数 6*11/33 = 2.0
        }

    # --- 6. 连号 ---
    consec_obs = np.mean([1.0 if any(b - a == 1 for a, b in zip(d["reds"], d["reds"][1:]))
                          else 0.0 for d in valid])
    consec_theo = theory_consecutive_rate()
    se = math.sqrt(consec_theo * (1 - consec_theo) / n)
    z = (consec_obs - consec_theo) / se if se > 0 else 0.0
    base["tests"]["consecutive"] = {
        "statistic": float(z), "df": None, "p_value": _two_sided_p_from_z(z),
        "verdict": _verdict(_two_sided_p_from_z(z), alpha),
        "observed_rate": float(consec_obs), "theory_rate": float(consec_theo),
        "se": float(se), "method": "normal-approx-one-proportion",
    }

    # --- 7. 同尾 ---
    tail_obs = np.mean([1.0 if len({x % 10 for x in d["reds"]}) < PICK else 0.0
                        for d in valid])
    tail_theo = theory_any_same_tail_rate()
    se_t = math.sqrt(tail_theo * (1 - tail_theo) / n)
    z_t = (tail_obs - tail_theo) / se_t if se_t > 0 else 0.0
    base["tests"]["same_tail"] = {
        "statistic": float(z_t), "df": None, "p_value": _two_sided_p_from_z(z_t),
        "verdict": _verdict(_two_sided_p_from_z(z_t), alpha),
        "observed_rate": float(tail_obs), "theory_rate": float(tail_theo),
        "se": float(se_t), "method": "normal-approx-one-proportion",
    }

    # --- 8. 相邻期红球重叠（理论：Hypergeom(33, 6, 6)）---
    if n >= 2:
        rep_theo = np.array([stats.hypergeom.pmf(k, R_MAX, PICK, PICK)
                             for k in range(PICK + 1)])
        rep_obs = np.zeros(PICK + 1, dtype=float)
        for a, b in zip(valid, valid[1:]):
            rep_obs[len(set(a["reds"]) & set(b["reds"]))] += 1
        m_rep = int(rep_obs.sum())
        rep_p = float(stats.chisquare(rep_obs, rep_theo * m_rep).pvalue)
        rep_stat = float(stats.chisquare(rep_obs, rep_theo * m_rep).statistic)
        base["tests"]["red_repeat"] = {
            "statistic": rep_stat, "df": PICK, "p_value": rep_p,
            "verdict": _verdict(rep_p, alpha), "n_pairs": m_rep,
            "observed_mean": float((rep_obs * np.arange(PICK + 1)).sum() / m_rep),
            "theory_mean": float(PICK * PICK / R_MAX),  # 6*6/33 = 1.0909
        }
    else:
        base["tests"]["red_repeat"] = {"statistic": None, "df": None, "p_value": None,
                                       "verdict": "insufficient-data", "n_pairs": 0}

    # --- 9. 蓝球与上期相同 ---
    if n >= 2:
        b_rep_obs = float(np.mean([1.0 if a["blue"] == b["blue"] else 0.0
                                   for a, b in zip(valid, valid[1:])]))
        b_rep_theo = 1.0 / B_MAX
        se_b = math.sqrt(b_rep_theo * (1 - b_rep_theo) / (n - 1))
        z_b = (b_rep_obs - b_rep_theo) / se_b if se_b > 0 else 0.0
        base["tests"]["blue_repeat"] = {
            "statistic": float(z_b), "df": None, "p_value": _two_sided_p_from_z(z_b),
            "verdict": _verdict(_two_sided_p_from_z(z_b), alpha),
            "observed_rate": b_rep_obs, "theory_rate": b_rep_theo,
            "se": float(se_b), "method": "normal-approx-one-proportion",
        }
    else:
        base["tests"]["blue_repeat"] = {"statistic": None, "df": None, "p_value": None,
                                        "verdict": "insufficient-data", "n_pairs": 0}

    # --- 综合结论 ---
    verdicts = [v["verdict"] for v in base["tests"].values()]
    base["verdict"] = "deviation" if "deviation" in verdicts else "consistent-with-random"
    if base["verdict"] == "consistent-with-random":
        warnings.append(
            "全部结构指标与均匀随机不可区分：这是彩票应有的结果，"
            "也意味着任何基于历史开奖的选号策略长期期望命中率都等于理论基线，"
            "不存在可利用的信号。"
        )
    base["n_deviation"] = int(sum(1 for v in verdicts if v == "deviation"))
    base["n_tests"] = len(verdicts)

    # --- V10-14：家庭化（family-wise）口径 ---
    # 顶层 verdict 的语义是「任一项偏离即 deviation」，其假阳性率不是 alpha，
    # 而是 1-(1-alpha)^m。实测（真实随机数据，m=9, alpha=0.05）：
    #   **顶层 verdict=deviation 的频率 ≈ 39%**（期望 5%）。
    # 这里额外给出经多重检验校正后的结论 verdict_fwer 与逐项校正 p，
    # 供报告/前端按需采用；**verdict 保持原语义不变**（向后兼容，不改已有行为）。
    _pw = {k: v for k, v in base["tests"].items() if v.get("p_value") is not None}
    if _pw:
        _names = list(_pw)
        _ps = [float(_pw[k]["p_value"]) for k in _names]
        _bonf = bonferroni(_ps)
        _bh = benjamini_hochberg(_ps)
        for _k, _pb, _ph in zip(_names, _bonf, _bh):
            base["tests"][_k]["p_bonferroni"] = float(_pb)
            base["tests"][_k]["p_bh"] = float(_ph)
            base["tests"][_k]["survives_bonferroni"] = bool(_pb < alpha)
            base["tests"][_k]["survives_bh"] = bool(_ph < alpha)
        base["family_wise"] = {
            "n_tests": len(_names),
            "alpha": float(alpha),
            "uncorrected_false_positive_rate": float(1 - (1 - alpha) ** len(_names)),
            "min_p_bonferroni": float(min(_bonf)),
            "min_p_bh": float(min(_bh)),
            "n_survive_bonferroni": int(sum(1 for x in _bonf if x < alpha)),
            "n_survive_bh": int(sum(1 for x in _bh if x < alpha)),
        }
        # FWER 口径：Bonferroni 校正后仍有一项显著才判 deviation
        base["verdict_fwer"] = ("deviation"
                                if base["family_wise"]["n_survive_bonferroni"] > 0
                                else "consistent-with-random")
    else:
        base["family_wise"] = {"n_tests": 0, "alpha": float(alpha),
                               "uncorrected_false_positive_rate": 0.0,
                               "min_p_bonferroni": None, "min_p_bh": None,
                               "n_survive_bonferroni": 0, "n_survive_bh": 0}
        base["verdict_fwer"] = base["verdict"]
    base["multiple_comparison_note"] = (
        f"共做 {len(verdicts)} 项检验，按 alpha={alpha} 单项判定时，纯随机数据仍有 "
        f"{1 - (1 - alpha) ** len(verdicts):.1%} 的概率至少出现一项假阳性；"
        "把单项 p 值送入 benjamini_hochberg() 做 FDR 校正后再下结论。"
    )
    # V38（2026-10-24 审计 P2）：顶层 `verdict` 是**未校正**的单项口径，
    # 在纯随机数据上也有相当概率报 "deviation"（实测 N=3517 时确实如此）。
    # 它保留原语义（向后兼容），但必须**显式标注**，否则任何读 `verdict`
    # 的下游（前端卡片、脚本、外部调用者）都会得到一个偏激进且未校正的结论。
    base["verdict_is_uncorrected"] = True
    base["verdict_authoritative_field"] = "verdict_fwer"
    base["verdict_note_zh"] = (
        f"顶层 `verdict` 是**单项未校正**判定（alpha={alpha}），"
        f"在纯随机数据上也有 {1 - (1 - alpha) ** max(1, len(verdicts)):.1%} 的概率"
        "至少出现一次假阳性；**要做『这份数据整体是否偏离随机』的结论，"
        "请读 `verdict_fwer`（Bonferroni 家庭化校正后）**。"
        "两者不一致时以 verdict_fwer 为准。"
    )
    return base


def _min_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.8) -> Dict:
    """当前样本量 n 下、给定检验水平与目标功效能检出的最小效应（命中数尺度）。

    δ = (z_{1-α/2} + z_{power}) * sd / sqrt(n)

    注意：``power`` 是**目标功效**（例如 0.8），对应的临界值是 z_{power} 而非
    z_{1-power}；功效 0.8 → z = 0.8416。
    """
    z_a = normal_ppf(1 - alpha / 2)
    z_p = normal_ppf(power)
    if n <= 0:
        return {
            "alpha": float(alpha), "power": float(power), "n": 0,
            "baseline_mean_hits": BASELINE_MEAN_HITS, "sd": BASELINE_SD_HITS,
            "delta_hits": None, "relative_lift": None,
            "hit_rate_pp": None, "per_draw_probability_gain_pp": None,
            "note_zh": "没有可用历史数据，无法评估检验功效。",
        }
    delta = (z_a + z_p) * BASELINE_SD_HITS / math.sqrt(n)
    rel = delta / BASELINE_MEAN_HITS
    # 单注「至少中 1 红」概率的理论值与其对提升的近似导数
    p0 = 1.0 - math.comb(R_MAX - PICK, PICK) / math.comb(R_MAX, PICK)  # ≈0.6683
    # d(p_at_least_1)/d(mean_hits) ≈ (1-p0)/mean_hits，来自 p≈1-exp(-mean) 的一阶近似：
    # 命中数近似泊松(λ=mean)，p(>=1)=1-e^{-λ}，dp/dλ = e^{-λ} = 1-p0。
    gain_pp = 100.0 * (1.0 - p0) * delta
    return {
        "alpha": float(alpha),
        "power": float(power),
        "n": int(n),
        "baseline_mean_hits": BASELINE_MEAN_HITS,
        "sd": BASELINE_SD_HITS,
        "delta_hits": float(delta),
        "relative_lift": float(rel),
        "hit_rate_pp": float(100.0 * delta / PICK),
        "per_draw_probability_gain_pp": float(gain_pp),
        "z_alpha": float(z_a),
        "z_power": float(z_p),
        "note_zh": (
            f"在 {n} 期样本下，只有期望命中数相对基线提升 ≥ {rel:.1%}"
            f"（每期多命中 {delta:.4f} 个红球）才可能在 α={alpha}、功效={power} 下被检出。"
            "低于该幅度的「优势」与随机波动无法区分。"
        ),
    }


# --------------------------------------------------------------------------
# 2. strategy_significance
# --------------------------------------------------------------------------

def strategy_significance(
    strategy_hits: Union[Sequence[float], float, int],
    baseline_hits: Union[Sequence[float], float, int],
    n_trials: int = 1,
    alpha: float = 0.05,
) -> Dict:
    """评估某策略命中数相对基线的显著性，并做多重检验校正。

    参数
    ----
    strategy_hits: 策略在每期的命中数序列；也可传标量（标量时只做总体比较，
                   无法做 t 检验/符号检验，返回相应 unknown 字段）。
    baseline_hits: 理论或对照的每期期望命中数。可传：
                   - 标量（常数基线，默认场景：6*6/33 = 1.0909）；
                   - 与 strategy_hits 等长的序列（逐期基线，用于对照策略配对检验）。
    n_trials: 搜索了多少种策略。**这是本函数最重要的参数**：
              你在 N 种策略里挑出最好的那个，其未校正 p 值必须按 N 校正。
    alpha: 显著性水平，默认 0.05。

    返回（关键字段）
    ----------------
    - ``n``: 有效配对数
    - ``mean`` / ``baseline_mean`` / ``delta`` / ``relative_lift``: 描述统计
    - ``sd``: 逐期差值（或策略命中数）的标准差
    - ``se``: 均值标准误
    - ``t`` / ``p_value``: 单样本 t 检验（若可算；**单侧**，见下方警告）
    - ``sign_test``: ``{n_pos, n_neg, statistic, p_value}`` 符号检验
      （statistic 为正差期数，零差期被剔除；p 为双侧二项检验）
    - ``p_uncorrected``: = ``p_value``（显式命名，提醒这是未校正值）
    - ``n_trials``: 搜索的策略总数
    - ``bonferroni_p``: Bonferroni 校正后的 p（= min(1, p_uncorrected * n_trials)）
    - ``bh_p``: **单策略场景下为 None**（见下），多策略请用
      ``strategy_significance_multi()``
    - ``adjusted_p``: 最终采用的校正后 p。``n_trials > 1`` 时等于 ``bonferroni_p``；
      ``n_trials == 1`` 时等于 ``p_uncorrected``（此时确实无需校正）
    - ``adjust_method``: "bonferroni" | "none"
    - ``survives_correction``: ``bool``，``adjusted_p < alpha`` 才为 True
    - ``verdict``: "significant-after-correction" | "not-significant" |
                   "insufficient-data" 等
    - ``warning_zh`` / ``message_zh``: 人类可读结论（中文）
    - ``direction``: "above" / "below" / "none"

    - ``p_two_sided`` / ``p_uncorrected_two_sided`` / ``bonferroni_p_two_sided`` /
      ``adjusted_p_two_sided`` / ``survives_correction_two_sided`` /
      ``verdict_two_sided``: **显式双侧口径**（V10-14 / D-1 新增）

    单侧 vs 双侧（**必须读这段，历史上因此漏判过劣绩**）
    ---------------------------------------------------
    ``p_value`` / ``p_uncorrected`` / ``adjusted_p`` / ``survives_correction``
    都是**单侧**的：方向由观测差值符号决定（``delta>0`` 取 ``p_two/2``，
    ``delta<0`` 取 ``1-p_two/2``）。后果是——当策略比基线**显著更差**时，
    这些字段全部趋向 **1.0**，``survives_correction`` 恒为 False，
    调用方**永远判不出「显著更差」**（实测：n=120、每期少 0.32 个红球时
    ``adjusted_p=1.0``，而 ``p_two_sided=7.6e-06``）。

    因此：
      * 只关心「是否更优」→ 可以继续用单侧字段（功率略高）；
      * 任何 **A/B 双向判断**（优劣、对照臂、报表"更差"档）→ **必须**用
        ``*_two_sided`` 字段，否则「更差」分支是死代码。

    多重检验语义（重要）
    -------------------
    未校正的 ``p < 0.05`` 在 ``n_trials`` 较大时**不构成证据**。若在 68 种策略中
    取最好者，期望假阳性个数已达 68 * 0.05 = 3.4 个，因此必须看到
    ``survives_correction=True`` 才能说「有效」。

    **关于 BH**：Benjamini-Hochberg 的定义要求输入**整组** m 个 p 值，其 adjusted p
    为 ``min_{j>=i}(p_(j) * m / j)``。本接口一次只评估**一个**策略，拿不到那组 p 值，
    因此单策略时 ``bh_p`` 返回 ``None`` 并附 ``bh_note="not-applicable-single-p"``，
    ``adjusted_p`` 退回 Bonferroni。**绝不可**用 ``benjamini_hochberg([p])[0]``
    充当 BH——那会因 m=1 退化为恒等函数，其值恰等于未校正 p，一旦参与取最小值
    就会静默吞掉 Bonferroni，使整套校正形同虚设。
    需要真正的 BH 时请用 ``strategy_significance_multi()``。

    不变式：``n_trials > 1`` 时 ``adjusted_p >= p_uncorrected``；
    ``n_trials == 1`` 时 ``adjusted_p == p_uncorrected``。

    边界：空序列、单期数据、n_trials<=0 均不崩溃；样本量 < 3 时 t 检验标注
    insufficient-data，但校正与描述统计仍然给出。
    """
    m_trials = max(1, int(n_trials))

    s_arr = _as_float_array(strategy_hits)
    b_arr = _as_float_array(baseline_hits)

    out: Dict = {
        "n": int(s_arr.size),
        "n_trials": m_trials,
        "alpha": float(alpha),
        "mean": None, "baseline_mean": None, "delta": None, "relative_lift": None,
        "sd": None, "se": None, "t": None, "p_value": None,
        "p_uncorrected": None, "bonferroni_p": None, "bh_p": None,
        "adjusted_p": None, "survives_correction": False,
        # ---- V10-14 / D-1：显式双侧口径（**新增字段，旧字段一律保留不改名**）----
        # 为什么必须有：``p_value`` / ``p_uncorrected`` / ``adjusted_p`` 是**单侧**的
        # （方向由观测差值符号决定）。当策略**显著更差**时它们全部趋向 1.0 →
        # ``survives_correction=False``，于是调用方**永远判不出「显著更差」**。
        # 任何 A/B 双向判断（优劣、对照臂）都必须读 ``*_two_sided``。
        "p_two_sided": None,
        "p_uncorrected_two_sided": None,
        "bonferroni_p_two_sided": None,
        "adjusted_p_two_sided": None,
        "survives_correction_two_sided": False,
        "verdict_two_sided": "insufficient-data",
        "sign_test": None, "direction": "none", "verdict": "insufficient-data",
        "warning_zh": ("未校正的 p<0.05 在 n_trials 较大时不构成证据；"
                       "必须通过多重检验校正才可称为「有效」。"
                       "注意 p_value/adjusted_p 是**单侧**口径（判「显著更优」用），"
                       "判「显著更差」或任何 A/B 双向结论必须用 p_two_sided / "
                       "adjusted_p_two_sided。"),
        "message_zh": "",
    }

    if s_arr.size == 0:
        out["message_zh"] = "策略命中序列为空，无法评估。"
        return out

    out["mean"] = float(s_arr.mean())
    # 基线：标量 → 常数；序列 → 逐期配对
    if b_arr.size == 0:
        out["message_zh"] = "基线命中序列为空，无法评估。"
        return out
    if b_arr.size == 1:
        base_vec = np.full(s_arr.size, float(b_arr[0]))
        out["baseline_mean"] = float(b_arr[0])
    else:
        if b_arr.size != s_arr.size:
            raise ValueError(
                f"baseline_hits 序列长度 {b_arr.size} 与 strategy_hits 长度 {s_arr.size} 不一致"
            )
        base_vec = b_arr
        out["baseline_mean"] = float(b_arr.mean())

    diff = s_arr - base_vec
    out["delta"] = float(diff.mean())
    if out["baseline_mean"]:
        out["relative_lift"] = float(out["delta"] / out["baseline_mean"])
    out["direction"] = "above" if out["delta"] > 0 else ("below" if out["delta"] < 0 else "none")
    out["sd"] = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    out["se"] = float(out["sd"] / math.sqrt(diff.size)) if diff.size else None

    # 差值是否「实质无变异」：sd 相对量级可忽略时，直接走解析分支。
    # 这样既避免 scipy 在近常数数据上的 catastrophic-cancellation 警告/精度损失，
    # 也让「恒定差值」这一决定性情形得到精确处理（t=±inf, p=0）。
    _scale = max(abs(out["delta"]), float(np.max(np.abs(diff))) if diff.size else 0.0, 1e-12)
    _degenerate = bool(out["sd"] == 0 or (out["sd"] / _scale) < 1e-9)

    # --- t 检验（逐期差值的单样本 t 检验，H0: 差值均值 = 0）---
    if diff.size >= 3 and not _degenerate:
        res = stats.ttest_1samp(diff, 0.0)
        p_two = float(res.pvalue)
        t_val = float(res.statistic)
        # 单侧 p：方向由观测差值符号决定（与 backtest.py 的历史口径一致）
        p_one = p_two / 2 if out["delta"] > 0 else 1.0 - p_two / 2
        out["t"] = t_val
        out["p_value"] = float(np.clip(p_one, 0.0, 1.0))
        out["p_two_sided"] = float(np.clip(p_two, 0.0, 1.0))
    elif diff.size >= 2 and _degenerate:
        # 退化但**决定性**的情形：每期差值恒等于同一个非零常数。
        # 此时 t = ±∞，p → 0，绝不能报成「不可用/p=1」（那会漏掉真实信号）。
        if out["delta"] != 0:
            out["t"] = float("inf") if out["delta"] > 0 else float("-inf")
            out["p_value"] = 0.0
            out["p_two_sided"] = 0.0
            out["note"] = "每期差值恒为常数且非零，检验统计量退化但结论明确"
        else:
            out["t"] = 0.0
            out["p_value"] = 1.0
            out["p_two_sided"] = 1.0
            out["note"] = "每期差值恒为 0，无任何效应"

    # --- 符号检验 ---
    if diff.size >= 1:
        n_pos = int(np.sum(diff > 0))
        n_neg = int(np.sum(diff < 0))
        n_eff = n_pos + n_neg
        if n_eff > 0:
            sign_p = float(stats.binomtest(n_pos, n_eff, 0.5,
                                           alternative="two-sided").pvalue)
            out["sign_test"] = {"n_pos": n_pos, "n_neg": n_neg, "n_eff": n_eff,
                                "statistic": n_pos, "p_value": sign_p,
                                "method": "two-sided-binomial-p=0.5"}
            if out["p_value"] is None:
                out["p_value"] = sign_p
        else:
            out["sign_test"] = {"n_pos": 0, "n_neg": 0, "n_eff": 0,
                                "statistic": 0, "p_value": 1.0,
                                "method": "two-sided-binomial-p=0.5"}

    # 样本量不足（< 3 期）时不允许下任何显著性结论：宁可返回 insufficient-data，
    # 也不要把退化的 p=1 当成「已检验过」的真实结果传给调用方/前端。
    if out["n"] < 3:
        out["verdict"] = "insufficient-data"
        out["verdict_two_sided"] = "insufficient-data"
        out["p_uncorrected"] = None
        out["p_uncorrected_two_sided"] = None
        out["adjusted_p"] = None
        out["adjusted_p_two_sided"] = None
        out["bonferroni_p"] = None
        out["bonferroni_p_two_sided"] = None
        out["survives_correction"] = False
        out["survives_correction_two_sided"] = False
        out["message_zh"] = (
            f"有效样本仅 {out['n']} 期（< 3），不足以做任何显著性检验；"
            "请勿据此下结论。"
        )
        return out

    if out["p_value"] is None:
        out["verdict"] = "insufficient-data"
        out["verdict_two_sided"] = "insufficient-data"
        out["message_zh"] = f"有效样本 {out['n']} 期，不足以做任何显著性检验。"
        return out

    # p_two_sided 兜底：符号检验的 p 本身就是双侧的（t 检验分支已显式赋过值）
    if out.get("p_two_sided") is None:
        _sign = out.get("sign_test") or {}
        out["p_two_sided"] = float(_sign.get("p_value", out["p_value"]))

    out["p_uncorrected"] = float(out["p_value"])
    # --- 多重检验校正 ---
    # 关键：BH 必须在「整组 m_trials 个 p 值」上计算才成立。本函数只拿到**一个**
    # 策略的 p 值，无法构造那组 p 值，因此单策略场景下 BH 标记为不可计算（None），
    # 唯一严谨可用的是 Bonferroni（p × n_trials，封顶 1.0）。
    # 曾经这里写成 benjamini_hochberg([p])[0]，m=1 使 BH 退化为恒等函数，
    # 其返回值恰等于未校正 p，再被 min() 选中，导致 Bonferroni 被静默丢弃、
    # adjusted_p 恒等于 p_uncorrected —— 校正完全失效。绝不能再用 min() 把
    # 「不可计算」当成「更优」。
    out["bonferroni_p"] = float(min(1.0, out["p_uncorrected"] * m_trials))
    # 双侧口径（D-1 修复）：同一套校正规则作用在 p_two_sided 上
    out["p_uncorrected_two_sided"] = float(out["p_two_sided"])
    out["bonferroni_p_two_sided"] = float(min(1.0, out["p_two_sided"] * m_trials))
    if m_trials > 1:
        out["bh_p"] = None
        out["bh_note"] = (
            "not-applicable-single-p：BH(FDR) 需要整组策略的 p 值分布，"
            "本接口只有单个 p 值，无法计算。请用 strategy_significance_multi() "
            "或在调用方收集所有策略的 p 值后交给 benjamini_hochberg()。"
        )
        out["adjusted_p"] = out["bonferroni_p"]
        out["adjusted_p_two_sided"] = out["bonferroni_p_two_sided"]
        out["adjust_method"] = "bonferroni"
    else:
        # n_trials == 1：没有做多重搜索，无需校正，BH 亦退化为恒等（此处是正确语义）
        out["bh_p"] = out["p_uncorrected"]
        out["adjusted_p"] = out["p_uncorrected"]
        out["adjusted_p_two_sided"] = out["p_uncorrected_two_sided"]
        out["adjust_method"] = "none"
    # 不变式：n_trials > 1 时 adjusted_p >= p_uncorrected
    out["survives_correction"] = bool(out["adjusted_p"] < alpha)
    # 双侧判决（D-1）：这才是「显著更优 / 显著更差」的唯一正确依据
    out["survives_correction_two_sided"] = bool(out["adjusted_p_two_sided"] < alpha)
    out["verdict_two_sided"] = ("significant-after-correction"
                                if out["survives_correction_two_sided"]
                                else "not-significant")

    if out["survives_correction"]:
        out["verdict"] = "significant-after-correction"
        out["message_zh"] = (
            f"该策略在 {m_trials} 次搜索后仍显著（校正后 p={out['adjusted_p']:.3g} < "
            f"{alpha}），这是需要进一步用独立样本复现的候选信号，**不是**已确认的优势。"
        )
    else:
        out["verdict"] = "not-significant"
        if out["p_uncorrected"] < alpha:
            out["message_zh"] = (
                f"未校正 p={out['p_uncorrected']:.3g} 看似显著，但在 {m_trials} 次策略搜索下"
                f"校正后 p={out['adjusted_p']:.3g} ≥ {alpha}：这是预期中的假阳性，"
                "不构成证据。"
            )
        else:
            out["message_zh"] = (
                f"未校正 p={out['p_uncorrected']:.3g} 亦不显著（校正后 "
                f"p={out['adjusted_p']:.3g}）：没有可检出的信号。"
            )
    return out


def strategy_significance_multi(
    per_strategy_hits: Sequence[Union[Sequence[float], float, int]],
    baseline_hits: Union[Sequence[float], float, int] = BASELINE_MEAN_HITS,
    alpha: float = 0.05,
) -> Dict:
    """对**一整组**策略同时做显著性检验，并在该组 p 值上做真正的 BH(FDR) 校正。

    这是 BH 唯一正确的用法：先对 m 个策略各算出一个未校正 p 值，再在**这 m 个
    p 值构成的集合**上计算 ``min_{j>=i}(p_(j) * m / j)``。单个策略单独调用
    ``strategy_significance`` 时无法计算 BH（那里 ``bh_p`` 为 None）。

    参数
    ----
    per_strategy_hits: 每个策略的逐期命中数序列的列表（每个元素长度应一致）。
    baseline_hits: 与 ``strategy_significance`` 同义（标量常数基线或逐期序列）。
    alpha: 显著性水平。

    返回
    ----
    dict：
    - ``m``: 策略个数
    - ``n_trials``: = m（这组策略就是全部被搜索的对象）
    - ``strategies``: 每个策略的检验结果（含 ``p_uncorrected`` / ``bonferroni_p``
      / ``bh_p`` / ``adjusted_p`` / ``survives_correction`` / ``verdict``）
    - ``n_survive_bonferroni`` / ``n_survive_bh``: 通过各自校正的策略数
    - ``p_values``: 未校正 p 值列表（便于外部复核）
    - ``adjusted_bonferroni`` / ``adjusted_bh``: 两种校正后的 p 值列表（同序）
    - ``note_zh``: 中文说明
    """
    strategies = list(per_strategy_hits or [])
    m = len(strategies)
    out: Dict = {
        "m": m, "n_trials": m, "alpha": float(alpha),
        "strategies": [], "p_values": [],
        "adjusted_bonferroni": [], "adjusted_bh": [],
        "n_survive_bonferroni": 0, "n_survive_bh": 0,
        # V10-14 / D-1：整组双侧口径（旧字段全部保留）
        "p_values_two_sided": [],
        "adjusted_bonferroni_two_sided": [], "adjusted_bh_two_sided": [],
        "n_survive_bonferroni_two_sided": 0, "n_survive_bh_two_sided": 0,
        "note_zh": "",
    }
    if m == 0:
        out["note_zh"] = "没有传入任何策略，无法做多重检验校正。"
        return out

    # 第一步：逐策略计算统计量与未校正 p 值
    raw: List[Dict] = []
    p_values: List[float] = []
    p_values_two: List[float] = []
    for hits in strategies:
        s = strategy_significance(hits, baseline_hits, n_trials=1, alpha=alpha)
        raw.append(s)
        if s["p_uncorrected"] is None:
            out["note_zh"] = "存在无法计算 p 值的策略（样本不足），请先补齐数据。"
            out["strategies"] = raw
            return out
        p_values.append(float(s["p_uncorrected"]))
        p_values_two.append(float(s["p_two_sided"]))

    # 第二步：在整组 p 值上做两种校正（这是 BH 的正确用法）
    bonf = bonferroni(p_values)
    bh = benjamini_hochberg(p_values)
    bonf_two = bonferroni(p_values_two)
    bh_two = benjamini_hochberg(p_values_two)

    for s, pb, ph, pb2, ph2 in zip(raw, bonf, bh, bonf_two, bh_two):
        s["p_uncorrected"] = float(s["p_value"])
        # V10-14 / D-1：整组双侧校正（A/B 双向结论的唯一正确依据）
        s["adjusted_p_two_sided"] = float(min(pb2, ph2))
        s["survives_correction_two_sided"] = bool(s["adjusted_p_two_sided"] < alpha)
        s["verdict_two_sided"] = ("significant-after-correction"
                                  if s["survives_correction_two_sided"]
                                  else "not-significant")
        s["bonferroni_p"] = float(pb)
        s["bh_p"] = float(ph)
        # 整组场景下两种校正都有定义，取更小者（BH <= Bonferroni 恒成立）
        s["adjusted_p"] = float(min(pb, ph))
        s["adjust_method"] = "min(bonferroni, benjamini_hochberg)"
        s["survives_correction"] = bool(s["adjusted_p"] < alpha)
        if s["survives_correction"]:
            s["verdict"] = "significant-after-correction"
            s["message_zh"] = (
                f"在 {m} 个策略的多重检验下仍显著（BH 校正后 p={ph:.3g}，"
                f"Bonferroni p={pb:.3g}），值得用独立样本复现。"
            )
        else:
            s["verdict"] = "not-significant"
            s["message_zh"] = (
                f"在 {m} 个策略中未通过校正（BH p={ph:.3g}，Bonferroni p={pb:.3g}"
                f" ≥ {alpha}）：与「搜索多个策略必然出现的假阳性」无法区分。"
            )

    out["strategies"] = raw
    out["p_values"] = p_values
    out["adjusted_bonferroni"] = [float(x) for x in bonf]
    out["adjusted_bh"] = [float(x) for x in bh]
    out["n_survive_bonferroni"] = int(sum(1 for x in bonf if x < alpha))
    out["n_survive_bh"] = int(sum(1 for x in bh if x < alpha))
    # V10-14 / D-1：双侧整组口径
    out["p_values_two_sided"] = [float(x) for x in p_values_two]
    out["adjusted_bonferroni_two_sided"] = [float(x) for x in bonf_two]
    out["adjusted_bh_two_sided"] = [float(x) for x in bh_two]
    out["n_survive_bonferroni_two_sided"] = int(sum(1 for x in bonf_two if x < alpha))
    out["n_survive_bh_two_sided"] = int(sum(1 for x in bh_two if x < alpha))
    out["note_zh"] = (
        f"对 {m} 个策略的 p 值集合做 Bonferroni 与 BH 校正。"
        f"Bonferroni 通过 {out['n_survive_bonferroni']} 个，BH(FDR) 通过 "
        f"{out['n_survive_bh']} 个。BH 控制的是假发现比例，比 Bonferroni 宽松。"
    )
    return out


def _as_float_array(x: Union[Sequence[float], float, int, None]) -> np.ndarray:
    """把标量/序列统一成 float 数组；None → 空数组。"""
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([float(x)], dtype=float)
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------
# 3. min_sample_size
# --------------------------------------------------------------------------

def min_sample_size(effect: float, sd: float, alpha: float = 0.05,
                    power: float = 0.8) -> int:
    """检出给定效应量所需的样本量（每期一注的检验场景）。

    公式（双侧单样本均值检验，正态近似）：
        n = ceil( ( (z_{1-α/2} + z_{power}) * sd / |effect| ) ** 2 )

    ``power`` 是目标功效，临界值取 z_{power}（0.8 → 0.8416），不是 z_{1-power}。

    参数
    ----
    effect: 要检出的效应量。单位必须与 ``sd`` 一致。若按「命中数 (hits)」计，
            则 +1% 相对提升对应 effect = 0.01 * 1.0909 = 0.010909。
    sd: 单期指标的标准差（命中数场景用 0.8678）。
    alpha: 第一类错误率，默认 0.05。
    power: 目标功效 1-β，默认 0.8。

    返回
    ----
    int，至少为 1。``|effect|`` 为 0（或 ``sd`` <= 0）时返回极大哨兵值 ``10**9``，
    表示「需要无限样本」——任何真实有限样本都无法检出零效应。
    同理，对**极小但非零**的 effect，若所需样本量超过 ``10**9`` 也会封顶为
    ``10**9``（避免浮点溢出崩溃；语义是「实际上不可能检出」）。

    单调性：n 随 ``sd`` 增大而增大，随 ``|effect|``、``power`` 增大而减小。
    这一单调性在封顶区间之外严格成立；一旦进入 ``10**9`` 封顶区间，
    返回值恒为哨兵值（仍满足「不小于」的弱单调）。
    """
    if sd <= 0 or effect == 0:
        return 10 ** 9
    a = abs(float(effect))
    # 极小 effect 会让 (sd/effect)**2 溢出为 inf 甚至抛 OverflowError
    # （如 effect=1e-300 时 Python float ** 2 溢出）。提前退化为哨兵值。
    if not math.isfinite(a) or a < 1e-300:
        return 10 ** 9
    z_a = normal_ppf(1 - alpha / 2)
    z_p = normal_ppf(power)
    try:
        n = ((z_a + z_p) * sd / a) ** 2
    except OverflowError:
        return 10 ** 9
    if not math.isfinite(n) or n >= 10 ** 9:
        return 10 ** 9
    return int(max(1, math.ceil(n - 1e-9)))


# --------------------------------------------------------------------------
# 4. evaluate_signal
# --------------------------------------------------------------------------

def evaluate_signal(
    draws: List[Dict],
    strategy_fn: Callable[..., Sequence[int]],
    window: int,
    n_trials: int = 1,
    alpha: float = 0.05,
    **kw,
) -> Dict:
    """用 walk-forward（滚动前推）方式评估一个选号策略，并与理论基线比较。

    ``strategy_fn`` 契约
    --------------------
        strategy_fn(history, **kw) -> 红球号码序列（任意长度 k >= 1）

    - ``history``: 时间升序的历史期次列表，**只包含第 t 期之前**的数据（无未来信息）。
    - ``kw``: 透传的关键字参数（例如 ``top_k=6``）。
    - 返回的号码用于与第 t 期实际红球求交集，命中数 ∈ [0, k]。

    基线（关键推导）
    ----------------
    - 若策略固定选 k 个号码，每期期望命中数 = 6k/33（超几何均值 k*n/N，n=6, N=33）。
    - 本函数的基线取 **k=6 的标准单注**：6*6/33 = 36/33 = 1.0909…，
      方差 6*(6/33)*(27/33)*(27/32) ≈ 0.7531（sd ≈ 0.8678）。
    - 若策略返回的 k 不是 6，基线按逐期 6*k_t/33 计算（配对检验），
      并在返回的 ``avg_k`` 中体现；这是必要的修正，否则「少选号码」会被误判为变差。

    做法
    ----
    对 t = window, window+1, ..., n-1 逐期：
    调用 ``strategy_fn(draws[:t], **kw)`` → 与 ``draws[t]["reds"]`` 求交 → 记录命中数
    与当期理论期望 6*k/33；随后把逐期差值送入 ``strategy_significance`` 做
    t 检验 / 符号检验 + 多重检验校正。

    参数
    ----
    draws: 时间升序期次列表。
    strategy_fn: 选号函数，见上。
    window: 至少需要多少期历史才产生预测（也是首个预测点的下标）。
    n_trials: 该策略是在多少种策略中被挑出来的（用于校正）。
    alpha: 显著性水平。
    ``**kw``: 透传给 ``strategy_fn``。

    返回
    ----
    dict，含：
    - ``n_predictions`` / ``n``: 实际评估期数
    - ``window`` / ``avg_k``: 窗口长度与平均选号个数
    - ``mean`` / ``baseline`` / ``delta`` / ``relative_lift``: 命中数统计
    - ``t`` / ``p_value`` / ``p_uncorrected`` / ``bonferroni_p`` / ``bh_p``
      / ``adjusted_p`` / ``survives_correction``（以上为**单侧**口径）
    - ``p_two_sided`` / ``adjusted_p_two_sided`` / ``survives_correction_two_sided``
      / ``verdict_two_sided``（**双侧**；判「策略显著更差」必须用它们，
      否则该结论在单侧口径下不可达 —— 见 strategy_significance 的警告）
    - ``sign_test``: 符号检验结果
    - ``verdict``: "significant-after-correction" | "not-significant" |
                   "insufficient-data"
    - ``hit_series``: ``[{"issue", "hits", "expected", "k"}]``（最多 200 条）
    - ``method`` / ``message_zh``: 方法学口径与中文结论

    边界：``window >= len(draws)``、空数据、``window < 1`` 时返回
    ``verdict="insufficient-data"`` 且 ``n_predictions=0``，不抛异常。
    ``strategy_fn`` 本身抛出的异常会被逐期捕获并计入 ``errors``，不中断整体评估。
    """
    valid = _valid_draws(draws)
    n_total = len(valid)
    w = int(window)
    base_out: Dict = {
        "window": w,
        "n_predictions": 0, "n": 0,
        "mean": None, "baseline": BASELINE_MEAN_HITS, "delta": None,
        "relative_lift": None, "avg_k": None,
        "t": None, "p_value": None, "p_uncorrected": None,
        "bonferroni_p": None, "bh_p": None, "adjusted_p": None,
        "survives_correction": False, "sign_test": None,
        # V10-14 / D-1：双侧口径（单侧字段保留；判「更差」必须用这两个）
        "p_two_sided": None, "adjusted_p_two_sided": None,
        "survives_correction_two_sided": False,
        "verdict_two_sided": "insufficient-data",
        "verdict": "insufficient-data", "hit_series": [], "errors": 0,
        "n_trials": max(1, int(n_trials)), "alpha": float(alpha),
        "baseline_mean_hits": BASELINE_MEAN_HITS,
        "baseline_sd_hits": BASELINE_SD_HITS,
        "method": ("walk-forward：仅用第 t 期之前的历史产生第 t 期预测，无未来信息；"
                   "基线为逐期 6*k_t/33（超几何均值），k 为策略当期选号个数；"
                   "标准单注 k=6 时基线 = 6*6/33 = 1.0909。"),
        "message_zh": "",
    }
    if not callable(strategy_fn):
        base_out["message_zh"] = "strategy_fn 不是可调用对象。"
        return base_out
    if w < 1 or n_total <= w:
        base_out["message_zh"] = (
            f"可用期数 {n_total} 不足以做 window={w} 的滚动评估"
            "（需要至少 window+1 期）。"
        )
        return base_out

    hits: List[float] = []
    exps: List[float] = []
    ks: List[int] = []
    series: List[Dict] = []
    errors = 0
    for t in range(w, n_total):
        history = valid[:t]
        try:
            picked = list(strategy_fn(history, **kw))
        except Exception:
            errors += 1
            continue
        try:
            pick = {int(x) for x in picked}
        except (TypeError, ValueError):
            errors += 1
            continue
        pick = {p for p in pick if 1 <= p <= R_MAX}
        k = len(pick)
        if k == 0:
            errors += 1
            continue
        h = float(len(pick & set(valid[t]["reds"])))
        hits.append(h)
        exps.append(PICK * k / R_MAX)  # 超几何均值 k*n/N
        ks.append(k)
        series.append({
            "issue": valid[t].get("issue"),
            "hits": h,
            "expected": round(exps[-1], 4),
            "k": k,
        })

    base_out["n_predictions"] = len(hits)
    base_out["n"] = len(hits)
    base_out["errors"] = errors
    base_out["hit_series"] = series[:200]
    if not hits:
        base_out["message_zh"] = "没有任何一期产生有效预测（可能 strategy_fn 全部失败）。"
        return base_out

    base_out["avg_k"] = float(np.mean(ks))
    base_out["mean"] = float(np.mean(hits))
    base_out["baseline"] = float(np.mean(exps))
    base_out["delta"] = base_out["mean"] - base_out["baseline"]
    base_out["relative_lift"] = (base_out["delta"] / base_out["baseline"]
                                if base_out["baseline"] else None)

    sig = strategy_significance(hits, exps, n_trials=n_trials, alpha=alpha)
    for key in ("mean", "baseline_mean", "sd", "se", "t", "p_value", "p_uncorrected",
                "bonferroni_p", "bh_p", "adjusted_p", "survives_correction",
                "sign_test", "direction", "verdict", "message_zh", "n_trials", "alpha",
                # V10-14 / D-1：双侧口径一并透传（旧键一个不少）
                "p_two_sided", "adjusted_p_two_sided",
                "survives_correction_two_sided", "verdict_two_sided"):
        if key in sig:
            base_out[key] = sig[key]
    base_out["baseline"] = float(np.mean(exps))
    base_out["mean"] = float(np.mean(hits))
    base_out["delta"] = base_out["mean"] - base_out["baseline"]
    base_out["relative_lift"] = (base_out["delta"] / base_out["baseline"]
                                if base_out["baseline"] else None)
    base_out["tests_note"] = (
        "本函数只报告统计显著性，不宣称策略有效；survives_correction=True "
        "仅表示值得在独立样本上复现。"
    )
    return base_out
