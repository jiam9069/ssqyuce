"""LLM 规律提炼器 + 假说库（V5-2）——自进化系统的第 2 环。

本模块在整条闭环里的位置
------------------------
``battle.py``（第 1 环）回答「哪个出号方法在逐期实战里打得更好」；
本模块（第 2 环）把**战绩**变成**可执行的规律假说**；
后续（第 3 环）由引擎按假说的 ``action`` 改变权重 / 候选生成，
再拿新的战斗回放结果回到 ``score_hypothesis`` / ``resolve_registry``
判决「保留还是淘汰」。

与 ``engine.llm_*`` 的分工
--------------------------
``engine`` 里的 LLM 是「选号者」（组合结构优化）；
本模块的 LLM 是「规律提出者」——它读战绩表与统计候选，
产出**可实战验证的假说**，每条必须能被下一期真实检验。

===================== 用户口径（本模块的核心约定） =====================
用户已明确：他充分理解彩票的随机独立性，**不需要再被提醒**。
他定义的「准确性」是「基于历史数据找规律 → 用于实战 → 通过实战结果对比」。
因此：

* 每条假说必须**可实战验证**，即必须有 ``action`` 能落到
  「下一注怎么选」或「给哪个方法更高权重」；
* **绝不产出**「彩票是随机的所以无法预测」这类不可验证的假说
  （``_reject_llm_hypothesis`` 里有对应的拦截词表）；
* 结论强度一律由文献纪律决定：期数低于
  ``config.METHOD_RECOMMENDATION_MIN_ISSUES`` 判「样本不足」，
  绝不用个别几期下支持/反对结论。这与用户口径不冲突——它保证的是
  「一次随机波动不会被误当成规律写进库里」。

============================ 动作白名单 ============================
``action`` 只允许下面 6 种（``ALLOWED_ACTION_TYPES``），
LLM 返回的其它动作类型一律丢弃并记录原因：

======================  ==========================================================
type                    字段与语义
======================  ==========================================================
``boost_method``        ``{"type","method","weight"}`` 给某出号方法加权/降权（0.2~3.0）
``prefer_numbers``      ``{"type","numbers":[1-33 红球]}`` 在候选生成时偏好这些红球
``blue_prefer``         ``{"type","blues":[1-16 蓝球]}`` 在蓝球覆盖时偏好这些蓝球
``prefer_structure``    ``{"type","rule":"<结构规则>"}`` 按结构规则影响选号
                        （重号/邻号/隔期复出/和值/奇偶/三区），规则白名单见
                        ``ALLOWED_STRUCTURE_RULES``
``blue_dedupe``         ``{"type","enabled":true}`` 多注之间蓝球去重覆盖
``blue_uniform``        ``{"type","enabled":true}`` 蓝球整期回退均匀（放弃蓝球模型）
======================  ==========================================================

``prefer_structure`` 的存在理由：重号 / 邻号 / 和值 / 三区这类规律**天然不是一组
固定号码**，而是「怎么选」的规则。若强行塞进 ``prefer_numbers`` 就只能留空列表，
等于绕过号码校验。给它们一个显式的、带规则白名单的动作类型，才能让
「每条假说都有可执行动作」这条硬约束对**所有**家族真正成立。

============================ 纯函数边界 ============================
除 ``register_hypotheses`` / ``resolve_registry`` 两个显式落盘入口外，
本模块不写盘；网络访问只发生在 ``mine_hypotheses(use_llm=True)`` 的 LLM 分支，
且必须走 ``llm_client`` 并由 ``_ask_llm`` 统一兜住所有失败。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import config, features as F
from . import llm_client

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

R_MAX, B_MAX = 33, 16

#: 假说动作白名单（详见模块 docstring）
ALLOWED_ACTION_TYPES: Tuple[str, ...] = (
    "boost_method",
    "prefer_numbers",
    "blue_prefer",
    "prefer_structure",
    "blue_dedupe",
    "blue_uniform",
    # V6：维度化动作——把「筛选条件」直接变成出号偏置。
    # 没有它们，凡是按「近 W 期热号 / 遗漏 / 邻号 / 共现」表述的规律
    # （也就是 LLM 与维度扫描最自然产出的那类）都无法进入出号，
    # 假说库只会堆满「无法执行」的空话。
    "prefer_red_filter",
    "prefer_blue_filter",
)

#: ``prefer_structure`` 允许的结构化选号规则
ALLOWED_STRUCTURE_RULES: Tuple[str, ...] = (
    "keep_last",      # 保留上期号码（重号）
    "avoid_last",     # 回避上期号码
    "neighbors_of_last",  # 纳入上期号码的邻号
    "gap2",           # 纳入上隔期（t-2）号码
    "sum_target",     # 和值向目标靠拢
    "odd_target",     # 奇偶配比向目标靠拢
    "zone_target",    # 三区分布向主导形态靠拢
)

#: 生命周期状态
HYPOTHESIS_STATUSES: Tuple[str, ...] = ("proposed", "validated", "rejected", "retired")

#: 统计挖掘使用的窗口（期）
MINING_WINDOWS: Tuple[int, ...] = (50, 100, 200)

#: 频率偏移 TOP-K
TOP_K = 3

#: 单个规律族的显著性门槛（alpha），超过该 p 值不产出假说
FAMILY_ALPHA = 0.05

#: ``boost_method`` 允许的权重区间
MIN_METHOD_WEIGHT, MAX_METHOD_WEIGHT = 0.2, 3.0

#: 各家族写进假说的搜索次数（多重检验校正用）——
#: 「扫描 33 个号码取最高频」已做了 33 次比较，取出 3 个还额外有组合代价，
#: 这里用最保守的 33 作为 ``n_trials``。
_N_TRIALS_NUMBER = R_MAX
_N_TRIALS_BLUE = B_MAX

#: 均值类检验用的每期红球命中标准差（与 ``rigor.BASELINE_SD_HITS`` 同源：
#: 超几何(33,6,6) 的标准差）。只在无法取得逐期序列时用于效应量/功效标注。
_HITS_SD = 1.0655

#: 不可实战验证的措辞（含这些短语的 LLM 假说一律丢弃并记录原因）
_UNVERIFIABLE_PHRASES: Tuple[str, ...] = (
    "随机所以无法预测", "随机，所以无法预测", "无法预测", "不可预测",
    "无法提高中奖", "不能提高中奖", "不存在规律", "没有规律可言",
    "纯随机", "完全随机", "放弃预测", "没有必要预测", "随机性决定",
    "不应预测", "无法优化", "没有可预测性",
)


# --------------------------------------------------------------------------
# 0. 小工具
# --------------------------------------------------------------------------

def _registry_path(registry_path: Optional[Any] = None) -> Path:
    """假说库 JSON 路径：显式传入优先，否则从 ``config.DATA_DIR`` 派生。

    默认 ``DATA_DIR/hypotheses.json``——**不硬编码绝对路径**，
    ``LOTT_HOME`` 改变时自动跟随。

    入参可以是 ``str`` 或 ``Path``：统一归一成 ``Path`` 再返回，
    否则调用方按字符串传路径时，``path.parent`` / ``path.exists()`` 会
    AttributeError（默认路径是 ``Path`` 所以那条分支不易暴露）。
    """
    if registry_path is not None:
        return Path(str(registry_path))
    return Path(config.DATA_DIR) / "hypotheses.json"


def _now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any, default: float = 0.0) -> float:
    """安全转 float（None / 非数值 / NaN / inf 一律回退默认值）。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _as_int(value: Any, default: int = 0) -> int:
    """安全转 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    """数值钳制。"""
    return float(min(max(value, lo), hi))


def _round(value: Any, nd: int = 4) -> float:
    """四舍五入到 nd 位（保持 JSON 友好）。"""
    return round(_as_float(value), nd)


def _wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 区间下界（小样本比例不做正态近似）。

    **V40 薄壳**：公式唯一真源是 `stats.rigor.wilson_lower_bound`
    （审计 P2-1：四份并行实现 → 一份真源）。保留同名私有薄壳是因为
    `tests/test_v10_cost.py::test_wilson_lower_matches_hypothesis_implementation`
    以「api_app._wilson_lower 对拍 hypothesis._wilson_lower」的形式
    钉住口径一致性 —— 双薄壳对拍仍逐位相等，对拍守卫的价值保留。
    """
    from .stats.rigor import wilson_lower_bound as _truth
    return _truth(hits, n, z)


def _safe_chi2(observed_1based: Sequence[float]) -> Dict:
    """卡方均匀性检验（懒加载 ``scipy``，失败返回未知）。

    ``observed_1based``：下标 0..m 的计数数组，只用 1..m（号码 1 起始）。
    返回 ``{statistic, df, p_value, m, n, expect}``；
    ``p_value`` 为 None 表示无法检验（此时该家族不产出假说，绝不硬算）。
    """
    obs = np.asarray([_as_float(x) for x in list(observed_1based)[1:]], dtype=float)
    m = int(obs.size)
    total = float(obs.sum())
    unknown = {"statistic": None, "df": m - 1 if m else 0, "p_value": None,
               "m": m, "n": int(total), "expect": None}
    if m < 2 or total <= 0:
        return unknown
    try:
        from scipy import stats as _st  # 懒加载：避免无 scipy 环境直接 import 失败
    except Exception:  # noqa: BLE001 —— 缺依赖时退回「无法检验」，绝不抛给调用方
        return unknown
    exp = total / m
    stat = float(np.sum((obs - exp) ** 2 / exp))
    return {
        "statistic": stat,
        "df": m - 1,
        "p_value": float(_st.chi2.sf(stat, m - 1)),
        "m": m,
        "n": int(total),
        "expect": float(exp),
    }


def _bonferroni(p: Optional[float], n_trials: int) -> Optional[float]:
    """Bonferroni 校正：``min(1, p * n_trials)``；p 为 None 时返回 None。"""
    if p is None:
        return None
    return float(min(1.0, max(0.0, float(p)) * max(1, int(n_trials))))


def _make_hyp_id(family: str, statement: str) -> str:
    """假说 ID：``<family>-<statement 的 sha1 前 10 位>``（同 statement 稳定可去重）。"""
    digest = hashlib.sha1(f"{family}|{statement}".encode("utf-8")).hexdigest()[:10]
    return f"{family}-{digest}"


def _build_hypothesis(family: str, statement: str, action: Dict,
                      params: Optional[Dict] = None,
                      evidence: Optional[Dict] = None,
                      source: str = "stat",
                      n_samples: int = 0,
                      created_at: Optional[str] = None) -> Dict:
    """按统一结构组装一条假说（唯一的结构出口，保证字段齐全）。"""
    return {
        "id": _make_hyp_id(family, statement),
        "family": str(family),
        "statement": str(statement),
        "action": action,
        "params": params if isinstance(params, dict) else {},
        "evidence": evidence if isinstance(evidence, dict) else {},
        "source": source,
        "n_samples": int(n_samples),
        "created_at": created_at or _now_iso(),
    }


# --------------------------------------------------------------------------
# 1. 确定性统计挖掘（不依赖 LLM，保证 LLM 不可用时也有产出）
# --------------------------------------------------------------------------

def _valid_draws(draws: Optional[Sequence[Dict]]) -> List[Dict]:
    """过滤出结构合法的开奖（reds 为 6 个 1-33 去重整数、blue 在 1-16）。"""
    out: List[Dict] = []
    for d in draws or []:
        if not isinstance(d, dict):
            continue
        reds = d.get("reds")
        blue = d.get("blue")
        try:
            rr = sorted(int(x) for x in reds)
            bb = int(blue)
        except (TypeError, ValueError):
            continue
        if len(rr) != 6 or len(set(rr)) != 6 or not all(1 <= x <= R_MAX for x in rr):
            continue
        if not 1 <= bb <= B_MAX:
            continue
        out.append({"issue": d.get("issue"), "reds": rr, "blue": bb, "date": d.get("date")})
    return out


def _freq_offset_reds(draws: List[Dict], window: int) -> Optional[Dict]:
    """红球近 ``window`` 期频率偏移（观察 vs 理论）。"""
    n = len(draws)
    if window <= 0 or n < window:
        return None
    sl = draws[-window:]
    freq = F.red_frequency(sl)
    exp = window * 6 / R_MAX
    chi = _safe_chi2(freq)
    ranked = sorted(range(1, R_MAX + 1), key=lambda x: -freq[x])
    top = ranked[:TOP_K]
    return {
        "window": window, "n": window, "top": top,
        "freq": {k: int(freq[k]) for k in top},
        "expect": _round(exp, 2),
        "lift": {k: _round(freq[k] / exp - 1.0) for k in top},
        "statistic": chi["statistic"], "p_value": chi["p_value"], "m": chi["m"],
        "overall_ratio": _round(float(freq[1:].sum()) / (exp * R_MAX), 4),
    }


def _freq_offset_blues(draws: List[Dict], window: int) -> Optional[Dict]:
    """蓝球近 ``window`` 期频率偏移（观察 vs 1/16）。"""
    n = len(draws)
    if window <= 0 or n < window:
        return None
    sl = draws[-window:]
    freq = F.blue_frequency(sl)
    exp = window / B_MAX
    chi = _safe_chi2(freq)
    ranked = sorted(range(1, B_MAX + 1), key=lambda x: -freq[x])
    top = ranked[:TOP_K]
    return {
        "window": window, "n": window, "top": top,
        "freq": {k: int(freq[k]) for k in top},
        "expect": _round(exp, 2),
        "lift": {k: _round(freq[k] / exp - 1.0) for k in top},
        "statistic": chi["statistic"], "p_value": chi["p_value"], "m": chi["m"],
    }


def _omission_ratio_top(draws: List[Dict]) -> Optional[Dict]:
    """遗漏回补比（当前遗漏 / 平均遗漏）TOP-K，红蓝各取。"""
    n = len(draws)
    if n < 30:
        return None
    om_r = F.current_omission_red(draws)
    avg_r = F.avg_omission(F.red_frequency(draws), n)
    ratio_r = np.divide(om_r[1:], avg_r[1:], out=np.ones(R_MAX), where=avg_r[1:] > 0)
    order_r = sorted(range(R_MAX), key=lambda i: -float(ratio_r[i]))
    top_r = [i + 1 for i in order_r[:TOP_K]]

    om_b = F.current_omission_blue(draws)
    avg_b = F.avg_omission(F.blue_frequency(draws), n)
    ratio_b = np.divide(om_b[1:], avg_b[1:], out=np.ones(B_MAX), where=avg_b[1:] > 0)
    order_b = sorted(range(B_MAX), key=lambda i: -float(ratio_b[i]))
    top_b = [i + 1 for i in order_b[:TOP_K]]

    return {
        "n": n,
        "top_reds": top_r,
        "ratio_reds": {k: _round(ratio_r[k - 1]) for k in top_r},
        "top_blues": top_b,
        "ratio_blues": {k: _round(ratio_b[k - 1]) for k in top_b},
    }


def _repeat_neighbor_stats(draws: List[Dict], window: int = 200) -> Optional[Dict]:
    """上期号码的「重号 / 邻号 / 隔期复出」率，与理论基线比较。"""
    n = len(draws)
    if n < 3:
        return None
    sl = draws[-(window + 1):] if n > window else draws
    rep = rep_exp = nb = nb_exp = gap = 0
    pairs = 0
    for a, b in zip(sl, sl[1:]):
        pairs += 1
        sa, sb = set(a["reds"]), set(b["reds"])
        rep += len(sa & sb)
        # 邻号：与上期某个红球相差 1 的号（不含上期本身）
        neigh = set()
        for x in sa:
            for y in (x - 1, x + 1):
                if 1 <= y <= R_MAX and y not in sa:
                    neigh.add(y)
        nb += len(neigh & sb)
        rep_exp += 6 * 6 / R_MAX
        nb_exp += 6 * len(neigh) / R_MAX
    if pairs == 0:
        return None
    gap_hits = 0
    gap_pairs = 0
    for i in range(2, len(sl)):
        gap_pairs += 1
        gap_hits += len(set(sl[i - 2]["reds"]) & set(sl[i]["reds"]))
    return {
        "n": pairs,
        "repeat_mean": _round(rep / pairs), "repeat_expect": _round(rep_exp / pairs),
        "repeat_lift": _round(rep / max(1e-9, rep_exp) - 1.0),
        "neighbor_mean": _round(nb / pairs), "neighbor_expect": _round(nb_exp / pairs),
        "neighbor_lift": _round(nb / max(1e-9, nb_exp) - 1.0),
        "gap_mean": _round(gap_hits / gap_pairs) if gap_pairs else 0.0,
        "gap_expect": _round(6 * 6 / R_MAX),
        "window": int(min(window, n - 1)),
    }


def _parse_zone(raw: Any) -> Optional[List[int]]:
    """把三区比解析成 ``[a, b, c]``（``features.zone_counts`` 给的是元组或字符串）。

    解析失败或总和不为 6 时返回 None —— 动作校验会因此拒绝该假说，
    而不是产出一条动作非法的记录。
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            z = [int(x) for x in raw]
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, str):
        try:
            z = [int(x) for x in raw.strip().strip("()[]").split(",")]
        except (TypeError, ValueError):
            return None
        if len(z) != 3:
            return None
    else:
        return None
    if any(x < 0 or x > 6 for x in z) or sum(z) != 6:
        return None
    return z


def _structure_recent_offset(draws: List[Dict], window: int = 50) -> Optional[Dict]:
    """和值 / 奇偶 / 三区分布的近期偏移（观察 vs 全历史）。"""
    n = len(draws)
    if n < window + 30:
        return None
    recent = draws[-window:]
    base = draws[:-window] if n > window else draws
    if not base:
        return None

    def _sum_mean(ds: List[Dict]) -> float:
        return float(np.mean(F.sums(ds))) if ds else 0.0

    def _odd_mean(ds: List[Dict]) -> float:
        return float(np.mean(F.odd_counts(ds))) if ds else 0.0

    s_r, s_b = _sum_mean(recent), _sum_mean(base)
    o_r, o_b = _odd_mean(recent), _odd_mean(base)

    zc_r: Dict[str, int] = {}
    for t in F.zone_counts(recent):
        zc_r[str(t)] = zc_r.get(str(t), 0) + 1
    zc_b: Dict[str, int] = {}
    for t in F.zone_counts(base):
        zc_b[str(t)] = zc_b.get(str(t), 0) + 1
    top_zone = max(zc_r, key=lambda k: zc_r[k]) if zc_r else None
    return {
        "n": int(window), "n_base": int(len(base)),
        "sum_recent": _round(s_r, 2), "sum_base": _round(s_b, 2),
        "sum_delta": _round(s_r - s_b, 2),
        "odd_recent": _round(o_r, 3), "odd_base": _round(o_b, 3),
        "odd_delta": _round(o_r - o_b, 3),
        "zone_top": top_zone,
        "zone_top_rate": _round(zc_r.get(top_zone, 0) / max(1, len(recent)), 3) if top_zone else None,
        "zone_top_rate_base": _round(zc_b.get(top_zone, 0) / max(1, len(base)), 3) if top_zone else None,
    }


def _method_rows(report: Optional[Dict]) -> List[Dict]:
    """从 battle 报告里取出可回放的方法战绩行。"""
    if not isinstance(report, dict):
        return []
    rows = report.get("results")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("replayable", True)]


def _method_trend(report: Optional[Dict], recent: int = 100) -> Optional[Dict]:
    """各方法「近 N 期 vs 早期」的逐期红球命中差（衰减 / 上升趋势）。

    只用报告里已有的逐期明细（``per_issue``），比较**同一方法**在两个时间
    片段上的表现——同一条序列的前后段，不是方法之间的横截面。
    """
    rows = _method_rows(report)
    trends: List[Dict] = []
    for r in rows:
        per = r.get("per_issue")
        if not isinstance(per, list) or len(per) < recent + 30:
            continue
        series = [_as_float(x.get("red_hits")) for x in per if isinstance(x, dict)]
        if len(series) < recent + 30:
            continue
        seg_new, seg_old = series[-recent:], series[:-recent]
        try:
            from scipy import stats as _st
            t_stat, p_val = _st.ttest_ind(seg_new, seg_old, equal_var=False)
        except Exception:  # noqa: BLE001
            t_stat, p_val = None, None
        trends.append({
            "method": str(r.get("method")),
            "n": len(series), "n_recent": len(seg_new), "n_old": len(seg_old),
            "recent": _round(float(np.mean(seg_new))),
            "old": _round(float(np.mean(seg_old))),
            "delta": _round(float(np.mean(seg_new)) - float(np.mean(seg_old))),
            "t": _round(t_stat, 3) if t_stat is not None else None,
            "p_value": _as_float(p_val, 1.0) if p_val is not None else None,
        })
    if not trends:
        return None
    trends.sort(key=lambda x: -abs(x["delta"]))
    return {"window_recent": int(recent), "n_methods": len(trends), "methods": trends}


def _significant(evidence: Dict, n_trials: int = 1) -> bool:
    """该家族的观测是否达到显著性门槛（Bonferroni 校正后 p < FAMILY_ALPHA）。"""
    p = evidence.get("p_value")
    if p is None:
        return True  # 无法检验的家族（如纯计数/结构性事实）不受 p 门槛限制
    return bool(_bonferroni(p, n_trials) < FAMILY_ALPHA)


def _mine_frequency(draws: List[Dict]) -> List[Dict]:
    """规律族 1：多窗口频率偏移（50/100/200 期，红球 + 蓝球）。"""
    out: List[Dict] = []
    for w in MINING_WINDOWS:
        red = _freq_offset_reds(draws, w)
        if red and red["p_value"] is not None and _significant(red, _N_TRIALS_NUMBER):
            nums = list(red["top"])
            out.append(_build_hypothesis(
                family="freq_offset_red",
                statement=(f"近 {w} 期红球 {nums} 出现频率高于理论期望"
                           f"（卡方 p={red['p_value']:.3g}，扫描 33 号已按"
                           f"Bonferroni 校正），下注时优先覆盖。"),
                action={"type": "prefer_numbers", "numbers": nums},
                params={"window": w, "top_k": TOP_K, "numbers": nums},
                evidence={"n": red["n"], "lift": _round(float(np.mean(list(red['lift'].values())))),
                          "top_freq": red["freq"], "expect": red["expect"],
                          "p_value": red["p_value"], "statistic": red["statistic"],
                          "p_adjusted": _bonferroni(red["p_value"], _N_TRIALS_NUMBER),
                          "n_trials": _N_TRIALS_NUMBER, "metric": "红球出现频次",
                          "baseline": red["expect"]},
                n_samples=red["n"],
            ))
        blue = _freq_offset_blues(draws, w)
        if blue and blue["p_value"] is not None and _significant(blue, _N_TRIALS_BLUE):
            blues = list(blue["top"])
            out.append(_build_hypothesis(
                family="freq_offset_blue",
                statement=(f"近 {w} 期蓝球 {blues} 出现频率偏离均匀"
                           f"（p={blue['p_value']:.3g}），蓝球覆盖优先纳入。"),
                action={"type": "blue_prefer", "blues": blues},
                params={"window": w, "top_k": TOP_K, "blues": blues},
                evidence={"n": blue["n"], "lift": _round(float(np.mean(list(blue['lift'].values())))),
                          "top_freq": blue["freq"], "expect": blue["expect"],
                          "p_value": blue["p_value"], "statistic": blue["statistic"],
                          "p_adjusted": _bonferroni(blue["p_value"], _N_TRIALS_BLUE),
                          "n_trials": _N_TRIALS_BLUE, "metric": "蓝球出现频次",
                          "baseline": blue["expect"]},
                n_samples=blue["n"],
            ))
    return out


def _mine_omission(draws: List[Dict]) -> List[Dict]:
    """规律族 2：遗漏回补比（当前遗漏 / 平均遗漏）TOP-K。"""
    om = _omission_ratio_top(draws)
    if not om:
        return []
    out: List[Dict] = []
    if om["top_reds"]:
        nums = list(om["top_reds"])
        ratios = om["ratio_reds"]
        out.append(_build_hypothesis(
            family="omission_ratio_red",
            statement=(f"红球 {nums} 的遗漏回补比（当前遗漏/平均遗漏）达 "
                       f"{max(ratios.values()):.2f}，属回补压力最大档，"
                       f"下注时优先纳入。"),
            action={"type": "prefer_numbers", "numbers": nums},
            params={"top_k": TOP_K, "numbers": nums,
                    "boost_method": {"type": "boost_method", "method": "stat:omission",
                                     "weight": 1.2}},
            evidence={"n": om["n"], "lift": _round(max(ratios.values()) - 1.0),
                      "ratios": ratios, "metric": "遗漏回补比",
                      "baseline": 1.0, "p_value": None,
                      "note_zh": "漏补比是确定性统计量，不做假设检验；"
                                 "其可证伪性由后续实战回放给出。"},
            n_samples=om["n"],
        ))
    if om["top_blues"]:
        blues = list(om["top_blues"])
        ratios_b = om["ratio_blues"]
        out.append(_build_hypothesis(
            family="omission_ratio_blue",
            statement=(f"蓝球 {blues} 的遗漏回补比最高（{max(ratios_b.values()):.2f}），"
                       f"蓝球覆盖优先纳入。"),
            action={"type": "blue_prefer", "blues": blues},
            params={"top_k": TOP_K, "blues": blues},
            evidence={"n": om["n"], "lift": _round(max(ratios_b.values()) - 1.0),
                      "ratios": ratios_b, "metric": "蓝球遗漏回补比",
                      "baseline": 1.0, "p_value": None},
            n_samples=om["n"],
        ))
    return out


def _mine_repeat_neighbor(draws: List[Dict]) -> List[Dict]:
    """规律族 3：上期号码的重号 / 邻号 / 隔期复出率。"""
    st = _repeat_neighbor_stats(draws)
    if not st:
        return []
    out: List[Dict] = []
    rep = _as_float(st["repeat_mean"])
    rep_lift = _as_float(st["repeat_lift"])
    rule = "keep_last" if rep_lift >= 0 else "avoid_last"
    out.append(_build_hypothesis(
        family="repeat_confirmed" if rep_lift >= 0 else "repeat_avoided",
        statement=(f"最近 {st['n']} 期实测每期平均重号 {rep:.3f} 个，"
                   f"理论期望 {st['repeat_expect']:.3f} 个"
                   f"（{'高' if rep_lift >= 0 else '低'} {abs(rep_lift) * 100:.1f}%）；"
                   + ("候选生成时保留 1 个上期号码。" if rep_lift >= 0
                      else "候选生成时回避上期号码。")),
        action={"type": "prefer_structure", "rule": rule},
        params={"mode": rule, "expected_repeat": rep},
        evidence={"n": st["n"], "lift": rep_lift, "observed": rep,
                  "baseline": st["repeat_expect"], "metric": "每期平均重号数",
                  "p_value": None},
        n_samples=st["n"],
    ))
    nb = _as_float(st["neighbor_mean"])
    nb_lift = _as_float(st["neighbor_lift"])
    if nb > 0:
        out.append(_build_hypothesis(
            family="neighbor_number",
            statement=(f"最近 {st['n']} 期实测每期平均命中邻号 {nb:.3f} 个，"
                       f"理论期望 {st['neighbor_expect']:.3f} 个"
                       f"（偏离 {nb_lift * 100:+.1f}%）；候选生成时纳入上期号码的邻号。"),
            action={"type": "prefer_structure", "rule": "neighbors_of_last"},
            params={"mode": "neighbors_of_last"},
            evidence={"n": st["n"], "lift": nb_lift, "observed": nb,
                      "baseline": st["neighbor_expect"], "metric": "每期平均邻号数",
                      "p_value": None},
            n_samples=st["n"],
        ))
    out.append(_build_hypothesis(
        family="gap2_reappear",
        statement=(f"隔期复出（上上期号码在本期再现）实测每期 {st['gap_mean']:.3f} 个，"
                   f"理论期望 {st['gap_expect']:.3f} 个；候选生成时按该幅度纳入隔期号。"),
        action={"type": "prefer_structure", "rule": "gap2"},
        params={"mode": "gap2"},
        evidence={"n": st["n"], "lift": _round(st["gap_mean"] - st["gap_expect"]),
                  "observed": st["gap_mean"], "baseline": st["gap_expect"],
                  "metric": "隔期复出数", "p_value": None},
        n_samples=st["n"],
    ))
    return out


def _mine_structure(draws: List[Dict]) -> List[Dict]:
    """规律族 4：和值 / 奇偶 / 三区分布的近期偏移。"""
    st = _structure_recent_offset(draws)
    if not st:
        return []
    out: List[Dict] = []
    if abs(_as_float(st["sum_delta"])) >= 5.0:
        direction = "上移" if _as_float(st["sum_delta"]) > 0 else "下移"
        out.append(_build_hypothesis(
            family="sum_shift",
            statement=(f"近 {st['n']} 期红球和值均值 {st['sum_recent']}，"
                       f"全历史 {st['sum_base']}，和值重心{direction} "
                       f"{abs(_as_float(st['sum_delta'])):.1f}；候选组合的和值应贴近 "
                       f"{st['sum_recent']}。"),
            action={"type": "prefer_structure", "rule": "sum_target",
                    "target_sum": st["sum_recent"]},
            params={"target_sum": st["sum_recent"], "direction": direction},
            evidence={"n": st["n"], "lift": _round(_as_float(st["sum_delta"]) / max(1e-9, _as_float(st["sum_base"]))),
                      "observed": st["sum_recent"], "baseline": st["sum_base"],
                      "metric": "和值均值", "p_value": None},
            n_samples=st["n"],
        ))
    if abs(_as_float(st["odd_delta"])) >= 0.2:
        out.append(_build_hypothesis(
            family="odd_even_shift",
            statement=(f"近 {st['n']} 期每期平均奇数 {st['odd_recent']} 个，"
                       f"全历史 {st['odd_base']} 个（差 {_as_float(st['odd_delta']):+.3f}）；"
                       f"候选组合的奇偶配比按该偏离微调。"),
            action={"type": "prefer_structure", "rule": "odd_target",
                    "target_odd": st["odd_recent"]},
            params={"target_odd": st["odd_recent"]},
            evidence={"n": st["n"], "lift": _round(_as_float(st["odd_delta"]) / 3.0),
                      "observed": st["odd_recent"], "baseline": st["odd_base"],
                      "metric": "每期奇数个数", "p_value": None},
            n_samples=st["n"],
        ))
    if st["zone_top"]:
        out.append(_build_hypothesis(
            family="zone_concentration",
            statement=(f"近 {st['n']} 期三区比 {st['zone_top']} 占 "
                       f"{_as_float(st['zone_top_rate']) * 100:.1f}%"
                       f"（全历史 {_as_float(st['zone_top_rate_base']) * 100:.1f}%）；"
                       f"候选组合的三区分布向该主导形态靠拢。"),
            action={"type": "prefer_structure", "rule": "zone_target",
                    "zone": _parse_zone(st["zone_top"])},
            params={"zone": st["zone_top"], "target_rate": st["zone_top_rate"]},
            evidence={"n": st["n"], "lift": _round(_as_float(st["zone_top_rate"]) - _as_float(st["zone_top_rate_base"])),
                      "observed": st["zone_top_rate"], "baseline": st["zone_top_rate_base"],
                      "metric": "主导三区比出现率", "p_value": None},
            n_samples=st["n"],
        ))
    return out


def _mine_method_trend(report: Optional[Dict]) -> List[Dict]:
    """规律族 5：各方法「近 N 期 vs 全局」的衰减 / 上升趋势。"""
    tr = _method_trend(report)
    if not tr:
        return []
    min_issues = int(getattr(config, "METHOD_RECOMMENDATION_MIN_ISSUES", 60))
    out: List[Dict] = []
    for row in tr["methods"]:
        delta = _as_float(row["delta"])
        p_val = row.get("p_value")
        # 只用战绩数据、不做 33 路号码扫描 → n_trials = 方法数
        if not _significant({"p_value": p_val}, max(1, tr["n_methods"])):
            continue
        if abs(delta) < 0.02:
            continue
        method = row["method"]
        up = delta > 0
        weight = _clip(1.0 + abs(delta), MIN_METHOD_WEIGHT, MAX_METHOD_WEIGHT)
        out.append(_build_hypothesis(
            family="method_trend_up" if up else "method_trend_down",
            statement=(f"方法 {method} 在近 {row['n_recent']} 期逐期红球命中 "
                       f"{row['recent']}，此前 {row['n_old']} 期为 {row['old']}"
                       f"（{delta:+.3f}/期，p={p_val if p_val is None else round(p_val, 4)}）；"
                       + (f"下一期起把该方法权重提到 {weight:.2f}。"
                          if up else f"下一期起把该方法权重降到 {weight:.2f}。")),
            action={"type": "boost_method", "method": method, "weight": _round(weight, 3)},
            params={"method": method, "recent_window": row["n_recent"],
                    "metric": "red_hits", "direction": "up" if up else "down"},
            evidence={"n": row["n"], "lift": _round(delta / max(1e-9, _as_float(row["old"]))),
                      "observed": row["recent"], "baseline": row["old"],
                      "p_value": p_val, "metric": "逐期红球命中数",
                      "n_trials": tr["n_methods"],
                      "p_adjusted": _bonferroni(p_val, tr["n_methods"])},
            n_samples=row["n"],
        ))
    return out


def _mine_blue_structure(report: Optional[Dict], draws: List[Dict]) -> List[Dict]:
    """规律族 6：蓝球结构性策略（逐期明细存在时用实测收益率支撑）。"""
    out: List[Dict] = []
    rows = {r.get("method"): r for r in _method_rows(report)}
    cov, uni = rows.get("blue_coverage"), rows.get("uniform")

    if cov and uni:
        per_c = [x for x in (cov.get("per_issue") or []) if isinstance(x, dict)]
        per_u = [x for x in (uni.get("per_issue") or []) if isinstance(x, dict)]
        n = min(len(per_c), len(per_u))
        if n > 0:
            hit_c = sum(1 for x in per_c[:n] if _as_float(x.get("blue_hit")) >= 0.5)
            hit_u = sum(1 for x in per_u[:n] if _as_float(x.get("blue_hit")) >= 0.5)
            lift = (hit_c - hit_u) / n
            out.append(_build_hypothesis(
                family="blue_dedupe_coverage",
                statement=(f"逐期实战回放 {n} 期中，蓝球去重覆盖（blue_coverage）"
                           f"每期至少中 1 个蓝球的期数占比 {hit_c / n:.3f}，"
                           f"独立重复（uniform）为 {hit_u / n:.3f}"
                           f"（提升 {lift * 100:+.1f} 个百分点）；多注时启用蓝球去重覆盖。"),
                action={"type": "blue_dedupe", "enabled": True},
                params={"mode": "dedupe", "n_tickets_per_issue":
                        _as_int(cov.get("n_tickets_per_issue"), 0)},
                evidence={"n": n, "lift": _round(lift),
                          "observed": _round(hit_c / n), "baseline": _round(hit_u / n),
                          "metric": "每期至少中 1 蓝球的期数占比",
                          "structural": True,
                          "note_zh": "去重覆盖的「至少中 1 蓝」概率 = min(k,16)/16 是"
                                     "数学恒等式的结构性收益；其可证伪性体现在"
                                     "实测期数占比与恒等式预期是否一致。"},
                n_samples=n,
            ))
    cov_rate = _as_float(cov.get("blue_hit_rate")) if cov else 0.0
    uni_rate = _as_float(uni.get("blue_hit_rate")) if uni else 0.0
    if cov and uni and cov_rate < uni_rate:
        out.append(_build_hypothesis(
            family="blue_uniform_fallback",
            statement=(f"回放中蓝球覆盖方案逐注命中率 {cov_rate:.4f} 低于均匀方案 "
                       f"{uni_rate:.4f}；下一期起蓝球回退均匀分布。"),
            action={"type": "blue_uniform", "enabled": True},
            params={"mode": "uniform"},
            evidence={"n": _as_int(cov.get("issues"), 0), "lift": _round(cov_rate - uni_rate),
                      "observed": _round(cov_rate), "baseline": _round(uni_rate),
                      "metric": "逐注蓝球命中率", "p_value": None},
            n_samples=_as_int(cov.get("issues"), 0),
        ))
    return out


#: 统计挖掘的家族清单（供调用方与前端展示）
STAT_FAMILIES: Tuple[str, ...] = (
    "freq_offset_red", "freq_offset_blue", "omission_ratio_red", "omission_ratio_blue",
    "repeat_confirmed", "repeat_avoided", "neighbor_number", "gap2_reappear",
    "sum_shift", "odd_even_shift", "zone_concentration",
    "method_trend_up", "method_trend_down",
    "blue_dedupe_coverage", "blue_uniform_fallback",
)


def mine_stat_hypotheses(report: Optional[Dict], draws: Optional[Sequence[Dict]]) -> List[Dict]:
    """确定性统计挖掘：从历史开奖与战斗战绩里产出候选规律假说。

    完全不依赖 LLM，也不访问网络——这是 LLM 不可用时的产出保证。
    报告为空 / 历史为空时返回空列表（不抛异常）。
    """
    ds = _valid_draws(draws)
    try:
        out: List[Dict] = []
        out += _mine_frequency(ds)
        out += _mine_omission(ds)
        out += _mine_repeat_neighbor(ds)
        out += _mine_structure(ds)
        out += _mine_method_trend(report)
        out += _mine_blue_structure(report, ds)
    except Exception as e:  # noqa: BLE001 —— 挖掘失败不能拖垮调用方
        print(f"[hypothesis] 统计挖掘部分失败，返回已产出结果: {e}")
        return []
    # 去重（同 family+statement 只留一条）
    seen = set()
    uniq: List[Dict] = []
    for h in out:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        uniq.append(h)
    return uniq


# --------------------------------------------------------------------------
# 2. LLM 结构校验与白名单过滤（LLM 输出不可信，绝不直接采信）
# --------------------------------------------------------------------------

def _clean_numbers(raw: Any, lo: int, hi: int, limit: int = 8) -> List[int]:
    """清洗号码列表：整数、去重、升序、范围校验。"""
    out: List[int] = []
    for x in raw if isinstance(raw, (list, tuple)) else []:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if lo <= v <= hi and v not in out:
            out.append(v)
    return sorted(out)[:limit]


def _normalize_action(action: Any) -> Tuple[Optional[Dict], Optional[str]]:
    """动作白名单校验与规范化。

    返回 ``(规范化动作, 拒绝原因)``：通过时原因为 None，不通过时动作为 None。
    只接受白名单内的 5 种动作，其余一律拒绝并给出中文原因。
    """
    if not isinstance(action, dict):
        return None, "action 不是对象"
    atype = action.get("type")
    if not isinstance(atype, str) or atype not in ALLOWED_ACTION_TYPES:
        return None, (f"动作类型 {atype!r} 不在白名单 "
                      f"{list(ALLOWED_ACTION_TYPES)} 内")
    if atype == "boost_method":
        method = action.get("method")
        if not isinstance(method, str) or not method.strip():
            return None, "boost_method 缺少 method"
        weight = _as_float(action.get("weight"), 1.0)
        if weight <= 0:
            return None, f"boost_method 权重非法（{action.get('weight')!r}）"
        return ({"type": "boost_method", "method": method.strip(),
                 "weight": _round(_clip(weight, MIN_METHOD_WEIGHT, MAX_METHOD_WEIGHT), 3)},
                None)
    if atype == "prefer_numbers":
        nums = _clean_numbers(action.get("numbers"), 1, R_MAX)
        if not nums:
            return None, "prefer_numbers 缺少 1-33 范围内的有效红球"
        return {"type": "prefer_numbers", "numbers": nums}, None
    if atype in ("prefer_red_filter", "prefer_blue_filter"):
        # V6：维度化动作——形状必须能被 discovery.make_filter 实例化，
        # 否则「可验证」只是一句空话。这里只校验形状与白名单，参数合法性由
        # discovery 的维度表把关（校验函数集中在一处，避免两套漂移）。
        spec = action.get("filter")
        if not isinstance(spec, dict) or not isinstance(spec.get("key"), str) or not spec["key"]:
            return None, f"{atype} 缺少 filter.key"
        try:
            from . import discovery
        except ImportError as e:
            return None, f"维度模块不可用：{e}"
        if spec["key"] not in discovery.FAMILY_BY_KEY:
            return None, f"维度 {spec['key']!r} 不在维度表内"
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            return None, f"{atype} 的 filter.params 必须是对象"
        want = "red" if atype == "prefer_red_filter" else "blue"
        if discovery.FAMILY_BY_KEY[spec["key"]]["target"] != want:
            return None, f"维度 {spec['key']!r} 的目标不是{want}球"
        gain = _as_float(action.get("gain"), 1.5)
        return {"type": atype, "filter": {"key": spec["key"], "params": params},
                "gain": _round(_clip(gain, 1.0, 3.0), 3)}, None
    if atype == "prefer_structure":
        rule = action.get("rule")
        if not isinstance(rule, str) or rule not in ALLOWED_STRUCTURE_RULES:
            return None, (f"prefer_structure 的 rule {rule!r} 不在规则白名单 "
                          f"{list(ALLOWED_STRUCTURE_RULES)} 内")
        out = {"type": "prefer_structure", "rule": rule}
        # 可选目标值：和值 / 奇数个数 / 三区形态（数值越界即拒绝，不静默改写）
        for key, lo, hi in (("target_sum", 21.0, 183.0),
                            ("target_odd", 0.0, 6.0)):
            if action.get(key) is None:
                continue
            val = _as_float(action.get(key), float("nan"))
            if not (lo <= val <= hi):
                return None, f"prefer_structure 的 {key}={action.get(key)!r} 超出合理范围 [{lo}, {hi}]"
            out[key] = _round(val, 3)
        if action.get("zone") is not None:
            zone = action.get("zone")
            if not isinstance(zone, (list, tuple)) or len(zone) != 3:
                return None, "prefer_structure 的 zone 必须是长度为 3 的数组"
            try:
                z = [int(x) for x in zone]
            except (TypeError, ValueError):
                return None, "prefer_structure 的 zone 含非整数"
            if any(x < 0 or x > 6 for x in z) or sum(z) != 6:
                return None, f"prefer_structure 的 zone {z} 非法（每区 0-6 且总和为 6）"
            out["zone"] = z
        return out, None
    if atype == "blue_prefer":
        blues = _clean_numbers(action.get("blues"), 1, B_MAX)
        if not blues:
            return None, "blue_prefer 缺少 1-16 范围内的有效蓝球"
        return {"type": "blue_prefer", "blues": blues}, None
    # blue_dedupe / blue_uniform：布尔开关，默认开启
    enabled = action.get("enabled", True)
    return {"type": atype, "enabled": bool(enabled)}, None


def _looks_unverifiable(text: str) -> Optional[str]:
    """命中「不可实战验证」措辞时返回原因，否则 None（用户明确排斥这类假说）。"""
    low = str(text)
    for phrase in _UNVERIFIABLE_PHRASES:
        if phrase in low:
            return f"假说属于不可实战验证的表述（命中禁用措辞「{phrase}」）"
    return None


def _reject_llm_hypothesis(raw: Any) -> Tuple[Optional[Dict], Optional[str]]:
    """校验单条 LLM 假说：结构 + 白名单 + 可验证性。

    返回 ``(规范化的原始假说, 拒绝原因)``。任何缺字段、动作不在白名单、
    或措辞不可实战验证的条目都会被拒绝并给出中文原因。
    """
    if not isinstance(raw, dict):
        return None, "条目不是 JSON 对象"
    statement = raw.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return None, "缺少 statement"
    statement = statement.strip()
    bad = _looks_unverifiable(statement)
    if bad:
        return None, bad
    action, why = _normalize_action(raw.get("action"))
    if action is None:
        return None, why or "action 非法"
    n_samples = _as_int(raw.get("n_samples"), 0)
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    # evidence 至少要有可验证的落点：n / lift / p_value 之一
    if not any(k in evidence for k in ("n", "lift", "p_value", "observed", "baseline")):
        return None, "evidence 缺少样本量或观测差值（无依据的假说不可验证）"
    if "n" not in evidence:
        evidence = dict(evidence)
        evidence["n"] = n_samples
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    family = raw.get("family")
    if not isinstance(family, str) or not family.strip():
        family = f"llm_{action['type']}"
    return (_build_hypothesis(
        family=family.strip()[:60],
        statement=statement[:400],
        action=action,
        params=params,
        evidence=evidence,
        source="llm",
        n_samples=n_samples,
    ), None)


def _parse_llm_hypotheses(res: Optional[dict]) -> Tuple[List[Dict], List[str]]:
    """把 LLM 返回解析成 (通过校验的假说列表, 拒绝原因列表)。

    容忍 ``{"hypotheses": [...]}`` 与裸数组两种形态；其余结构全部拒绝。
    """
    if res is None:
        return [], ["LLM 返回内容无法解析为 JSON"]
    if isinstance(res, dict):
        items = res.get("hypotheses")
        if items is None:
            items = res.get("items")
    elif isinstance(res, list):
        items = res
    else:
        items = None
    if not isinstance(items, list):
        return [], ["LLM 返回结构中缺少 hypotheses 数组"]

    passed: List[Dict] = []
    rejected: List[str] = []
    for i, raw in enumerate(items):
        hyp, why = _reject_llm_hypothesis(raw)
        if hyp is None:
            rejected.append(f"第 {i + 1} 条被丢弃：{why}")
            continue
        passed.append(hyp)
    # LLM 侧自我去重（同 id 只留第一条）
    seen = set()
    uniq: List[Dict] = []
    for h in passed:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        uniq.append(h)
    return uniq, rejected


# --------------------------------------------------------------------------
# 3. LLM 提示词与调用（失败一律优雅降级）
# --------------------------------------------------------------------------

HYPOTHESIS_SYSTEM_PROMPT = (
    "你是双色球历史数据的「规律分析助手」。用户的口径是：基于历史数据找规律 → 用于实战 → "
    "通过实战结果对比来验证。他不需要「随机性」提醒，也不要笼统的观察。\n"
    "你的任务：阅读给定的方法战绩表与统计候选，提出**可实战验证的新假说与组合策略**。\n"
    "硬要求：\n"
    "1. 每条假说必须给出依据（哪个方法/特征、在多少期样本上、好多少——用具体数字），"
    "并给出**具体动作**。\n"
    "2. 不要输出无法验证的笼统描述，也不要输出「因为随机所以无法预测」这类结论。\n"
    "3. action.type 只能取以下五种之一（其它一律无效）：\n"
    "   - {\"type\":\"boost_method\",\"method\":\"<方法名>\",\"weight\":0.2~3.0}\n"
    "   - {\"type\":\"prefer_numbers\",\"numbers\":[1-33 的红球，最多 8 个]}\n"
    "   - {\"type\":\"blue_prefer\",\"blues\":[1-16 的蓝球，最多 8 个]}\n"
    "   - {\"type\":\"blue_dedupe\",\"enabled\":true}\n"
    "   - {\"type\":\"blue_uniform\",\"enabled\":true}\n"
    "4. 只输出 JSON，不要输出 JSON 之外的任何文字。"
)


def build_hypothesis_prompt(report: Optional[Dict], stats: Optional[dict],
                            stat_hypotheses: Sequence[Dict],
                            draws: Optional[Sequence[Dict]] = None) -> str:
    """组装「规律提炼」提示词（战绩表 + 统计候选 + 输出 schema）。"""
    rows = _method_rows(report)
    battle_brief = [
        {
            "method": r.get("method"), "issues": _as_int(r.get("issues")),
            "red_hits_mean": _round(r.get("red_hits_mean")),
            "blue_hit_rate": _round(r.get("blue_hit_rate")),
            "prize_rate_ge5": _round(r.get("prize_rate_ge5")),
            "roi": _round(r.get("roi")),
        }
        for r in rows
    ]
    cmp_out = (report or {}).get("comparison") if isinstance(report, dict) else None
    pairs: List[Dict] = []
    if isinstance(cmp_out, dict):
        for p in (cmp_out.get("pairs") or [])[:12]:
            if not isinstance(p, dict):
                continue
            pairs.append({
                "pair": p.get("pair"), "n_issues": _as_int(p.get("n_issues")),
                "delta_red_hits": _round(p.get("delta_red_hits")),
                "delta_blue_rate": _round(p.get("delta_blue_rate")),
                "p_adjusted": p.get("p_adjusted"), "verdict": p.get("verdict"),
            })
    stat_brief = [
        {"family": h.get("family"), "statement": h.get("statement"),
         "action": h.get("action"), "evidence": h.get("evidence")}
        for h in list(stat_hypotheses)[:12]
    ]
    ds = _valid_draws(draws)
    recent_txt = json.dumps(
        [{"issue": d["issue"], "reds": d["reds"], "blue": d["blue"]} for d in ds[-12:]],
        ensure_ascii=False)
    n_issues = _as_int((report or {}).get("n_issues")) if isinstance(report, dict) else 0
    min_issues = _as_int((report or {}).get("min_issues")) if isinstance(report, dict) else 0
    return (
        f"## 方法实战战绩（walk-forward 回放 {n_issues} 期，样本门槛 {min_issues} 期）\n"
        f"```json\n{json.dumps(battle_brief, ensure_ascii=False)}\n```\n"
        f"## 方法两两对比（校正后 p 值与判决）\n"
        f"```json\n{json.dumps(pairs, ensure_ascii=False)}\n```\n"
        f"## 统计挖掘已产出的候选规律（请**不要重复**这些，只补充新的）\n"
        f"```json\n{json.dumps(stat_brief, ensure_ascii=False)}\n```\n"
        f"## 最近 12 期开奖\n{recent_txt}\n"
        f"## 统计摘要（可为空）\n```json\n"
        f"{json.dumps(stats, ensure_ascii=False)[:1500] if stats else '{}'}\n```\n"
        "请输出 2-5 条**新的**可实战验证假说：\n"
        '{"hypotheses":[{"family":"简短英文族名","statement":"一句话规律陈述（含样本量与差多少）",'
        '"action":{"type":"boost_method|prefer_numbers|blue_prefer|blue_dedupe|blue_uniform",...},'
        '"params":{},"evidence":{"n":样本期数,"lift":观测差值,"p_value":可选},'
        '"n_samples":样本期数}]}\n'
        "每条都要能落到「下一注怎么选」或「给哪个方法更高权重」。"
    )


def _ask_llm(report: Optional[Dict], stats: Optional[dict],
             stat_hypotheses: Sequence[Dict],
             draws: Optional[Sequence[Dict]] = None,
             max_tokens: int = 1800,
             timeout: Optional[float] = None) -> Tuple[List[Dict], Optional[str], List[str]]:
    """调用 LLM 提炼假说；**任何失败都不抛出**。

    返回 ``(通过校验的 LLM 假说, 错误信息或 None, 被丢弃条目原因列表)``。
    覆盖的失败面：``config.LLM_DISABLED``、通道未配置、超时、HTTP 错误、
    ``LLMChannelError``、返回内容不是 JSON、JSON 结构不对、条目缺字段、
    动作不在白名单——全部降级为「无 LLM 假说 + 一条错误说明」。
    """
    if config.LLM_DISABLED:
        return [], "LLM 已停用（config.LLM_DISABLED=True），仅使用统计假说", []
    try:
        res = llm_client.chat_json(
            HYPOTHESIS_SYSTEM_PROMPT,
            build_hypothesis_prompt(report, stats, stat_hypotheses, draws),
            max_tokens=max_tokens,
            temperature=0.6,
            timeout=timeout,
            deadline=time.time() + float(timeout or config.LLM_TIMEOUT),
            strict=False,
        )
    except llm_client.LLMChannelError as e:
        return [], f"LLM 通道不可用: {e}", []
    except Exception as e:  # noqa: BLE001 —— 网络/解析等任何异常都不能让调用方崩
        return [], f"LLM 提炼失败: {type(e).__name__}: {e}", []
    if res is None:
        return [], "LLM 未返回可解析的 JSON（通道未配置、超时或模型返回空内容）", []
    hyps, rejected = _parse_llm_hypotheses(res)
    if not hyps and rejected:
        return [], "LLM 返回的假说全部未通过结构校验或动作白名单", rejected
    return hyps, None, rejected


# --------------------------------------------------------------------------
# 4. 对外主入口：mine_hypotheses
# --------------------------------------------------------------------------

def mine_hypotheses(report: Optional[Dict], draws: Optional[Sequence[Dict]],
                    use_llm: Optional[bool] = None,
                    stats: Optional[dict] = None,
                    max_tokens: int = 1800,
                    timeout: Optional[float] = None) -> Dict:
    """挖掘规律假说：**先确定性统计挖掘，再用 LLM 提炼**。

    参数
    ----
    report: ``battle.battle_report()`` 的输出（各方法战绩 + 两两对比结论）。
            传空 dict / None 时不崩，只是失去「方法趋势」与「蓝球结构」两族。
    draws:  按时序升序的历史开奖 ``[{issue, reds, blue}, ...]``。
    use_llm: ``None``（默认）跟随 ``config.LLM_DISABLED``；``False`` 强制只用统计；
             ``True`` 强制尝试 LLM（仍然会在失败时降级）。
    stats:  可选统计摘要（``features.compute_features`` 的结果），供 LLM 参考。
    max_tokens / timeout: LLM 调用的输出预算与单次超时（秒）。

    返回
    ----
    ``{hypotheses, llm_used, llm_error, stat_count, llm_count,
       llm_rejected, llm_attempted, families}``

    * ``llm_used``：**真的有 LLM 假说入库**才为 True（通道失败时为 False）；
    * ``llm_error``：失败原因（中文，可直接展示）；成功时为 None；
    * ``llm_rejected``：被结构校验 / 白名单过滤掉的 LLM 条目原因（保留痕迹）；
    * 统计假说永远先产出，因此 LLM 全挂也不影响调用方拿到可用假说。
    """
    stat_hyps = mine_stat_hypotheses(report, draws)
    want_llm = (not config.LLM_DISABLED) if use_llm is None else bool(use_llm)
    llm_hyps: List[Dict] = []
    llm_error: Optional[str] = None
    rejected: List[str] = []
    if want_llm:
        llm_hyps, llm_error, rejected = _ask_llm(
            report, stats, stat_hyps, draws=draws,
            max_tokens=max_tokens, timeout=timeout)
    elif use_llm is False:
        llm_error = "调用方指定 use_llm=False，仅使用统计假说"

    # LLM 补充的假说与统计候选合并去重（同 id 时保留统计版：它有更强的可复现依据）
    merged = list(stat_hyps)
    seen = {h["id"] for h in merged}
    for h in llm_hyps:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        merged.append(h)

    families = sorted({h.get("family", "") for h in merged if h.get("family")})
    return {
        "hypotheses": merged,
        "llm_used": bool(llm_hyps),
        "llm_error": llm_error,
        "stat_count": len(stat_hyps),
        "llm_count": len(llm_hyps),
        "llm_rejected": rejected,
        "llm_attempted": bool(want_llm),
        "families": families,
    }


# --------------------------------------------------------------------------
# 5. 假说库：注册 / 打分 / 收敛 / 读取
# --------------------------------------------------------------------------

def _empty_registry() -> Dict:
    """空的假说库骨架。

    ``counters`` 是 V10-02 前置注册的**库根累计台账**（R1 的唯一校正分母）。
    它必须被 _load_registry 原样读回，否则每次读到的 n_tested_cumulative 都是 0,
    累计口径校正就形同虚设（这是实现时真实踩到的 bug）。
    """
    return {"version": "v10-1", "updated_at": None, "hypotheses": [],
            "counters": {"n_tested_cumulative": 0, "last_issue": None, "batches": []}}


def _load_registry(registry_path: Optional[Any] = None) -> Dict:
    """读取假说库；文件缺失 / JSON 损坏 / 结构不对时**返回空库而不抛异常**。"""
    path = _registry_path(registry_path)
    try:
        if not path.exists():
            return _empty_registry()
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        print(f"[hypothesis] 假说库读取失败（{path}）：{e}；按空库继续")
        return _empty_registry()
    if isinstance(raw, list):  # 容忍裸数组形态
        raw = {"hypotheses": raw}
    if not isinstance(raw, dict):
        return _empty_registry()
    hyps = raw.get("hypotheses")
    reg = _empty_registry()
    reg["version"] = str(raw.get("version") or "v10-1")
    reg["updated_at"] = raw.get("updated_at")
    reg["hypotheses"] = [h for h in hyps if isinstance(h, dict)] if isinstance(hyps, list) else []
    # V10-02：库根累计台账必须原样读回（R1 的唯一校正分母）
    cnt = raw.get("counters")
    if isinstance(cnt, dict):
        reg["counters"] = {
            "n_tested_cumulative": _as_int(cnt.get("n_tested_cumulative"), 0),
            "last_issue": cnt.get("last_issue"),
            "batches": cnt.get("batches") if isinstance(cnt.get("batches"), list) else [],
        }
    return reg


def _save_registry(reg: Dict, registry_path: Optional[Any] = None) -> bool:
    """原子写入假说库（先写 .tmp 再 os.replace），失败返回 False 而不抛异常。"""
    path = _registry_path(registry_path)
    reg = dict(reg)
    reg["updated_at"] = _now_iso()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except OSError as e:
        print(f"[hypothesis] 假说库写入失败（{path}）：{e}")
        return False
    return True


def _dedup_key(hyp: Dict) -> str:
    """去重键：优先用 ``statement``（按需求），缺失时退回家族+动作。"""
    st = hyp.get("statement")
    if isinstance(st, str) and st.strip():
        return f"statement:{st.strip()}"
    return f"shape:{hyp.get('family')}|{json.dumps(hyp.get('action'), ensure_ascii=False, sort_keys=True)}"


def register_hypotheses(hyps: Optional[Sequence[Dict]],
                        registry_path: Optional[Any] = None) -> Dict:
    """把假说注册进假说库（**本模块仅有的两个落盘入口之一**）。

    规则
    ----
    * 同 ``statement`` 去重：重复注册**不新建记录**，而是累加 ``attempts``
      并刷新 ``last_seen_at`` / ``evidence`` / ``n_samples``；
    * 新假说初始 ``status="proposed"``、``attempts=1``、``supports=0`` / ``refutes=0``；
    * 结构非法（无 ``statement`` 或动作不在白名单）的条目直接跳过并计数。

    返回 ``{registered, deduped, total, skipped, registry_path}``：
    ``registered`` 为新增条数，``deduped`` 为命中已有 statement 的条数。
    """
    reg = _load_registry(registry_path)
    exist = {_dedup_key(h): h for h in reg["hypotheses"]}
    registered = deduped = skipped = 0
    for hyp in hyps or []:
        if not isinstance(hyp, dict):
            skipped += 1
            continue
        action, _why = _normalize_action(hyp.get("action"))
        statement = hyp.get("statement")
        if action is None or not isinstance(statement, str) or not statement.strip():
            skipped += 1
            continue
        norm = dict(hyp)
        norm["action"] = action
        norm["statement"] = statement.strip()
        key = _dedup_key(norm)
        cur = exist.get(key)
        if cur is not None:
            cur["attempts"] = _as_int(cur.get("attempts"), 0) + 1
            cur["last_seen_at"] = _now_iso()
            for field in ("evidence", "n_samples", "params", "source"):
                if norm.get(field) is not None:
                    cur[field] = norm[field]
            deduped += 1
            continue
        # 新记录：statement 才是去重主键，ID 仅作展示/回放关联用，
        # 因此同 family 下 statement 不同 → ID 自动加序号后缀，保证全局唯一。
        if not norm.get("id"):
            base_id = _make_hyp_id(str(norm.get("family") or "misc"), norm["statement"])
            used = {str(h.get("id")) for h in reg["hypotheses"]}
            hid, seq = base_id, 1
            while hid in used:
                seq += 1
                hid = f"{base_id}-{seq}"
            norm["id"] = hid
        rec = dict(norm)
        rec["status"] = rec.get("status") if rec.get("status") in HYPOTHESIS_STATUSES else "proposed"
        rec["attempts"] = max(1, _as_int(rec.get("attempts"), 1))
        rec["supports"] = _as_int(rec.get("supports"), 0)
        rec["refutes"] = _as_int(rec.get("refutes"), 0)
        rec["created_at"] = rec.get("created_at") or _now_iso()
        rec["last_seen_at"] = _now_iso()
        reg["hypotheses"].append(rec)
        exist[key] = rec
        registered += 1
    _save_registry(reg, registry_path)
    return {
        "registered": registered,
        "deduped": deduped,
        "total": len(reg["hypotheses"]),
        "skipped": skipped,
        "registry_path": str(_registry_path(registry_path)),
    }


def _extract_replay_metric(replay_result: Optional[Dict]) -> Tuple[Optional[float], int, str]:
    """从回放结果里取出该假说关心的指标值、样本量与指标名。

    取值优先级：``metric_value`` > ``delta_red_hits`` > ``red_hits_mean`` > ``roi``。
    样本量取 ``n_issues`` / ``issues``，二者都缺失时退回 ``per_issue`` 长度。
    """
    if not isinstance(replay_result, dict):
        return None, 0, "unknown"
    metric = "unknown"
    value: Optional[float] = None
    for key in ("metric_value", "delta_red_hits", "red_hits_mean", "roi"):
        if replay_result.get(key) is not None:
            value = _as_float(replay_result.get(key))
            metric = {"metric_value": "metric_value", "delta_red_hits": "红球命中差",
                      "red_hits_mean": "平均红球命中", "roi": "返奖率"}[key]
            break
    n = _as_int(replay_result.get("n_issues"), _as_int(replay_result.get("issues"), 0))
    if n <= 0 and isinstance(replay_result.get("per_issue"), list):
        n = len(replay_result["per_issue"])
    return value, n, metric


def score_hypothesis(hyp: Optional[Dict], replay_result: Optional[Dict]) -> Dict:
    """用一次战斗回放结果给假说打分（**纯函数，不写盘**）。

    判决口径
    --------
    * ``样本量 < config.METHOD_RECOMMENDATION_MIN_ISSUES``（默认 60）→「样本不足」，
      绝不用个别几期下支持/反对结论；
    * 回放结果自带 ``delta_red_hits``（相对基线）时：|delta| ≥ 0.05 且样本达标
      才判支持/反对，否则「无显著差异」（样本够但差异太小）；
    * 否则用回放绝对值对比假说证据里的 ``baseline``（不得低于该基线）；
    * ``replay_result`` 里的 ``p_value`` / ``p_adjusted`` 若存在，按 alpha 收紧判决。

    返回 ``{hyp_id, family, metric, delta, n_issues, min_issues, verdict,
            basis, direction, p_value, p_adjusted}``。
    """
    hyp = hyp if isinstance(hyp, dict) else {}
    min_issues = int(getattr(config, "METHOD_RECOMMENDATION_MIN_ISSUES", 60))
    alpha = float(getattr(config, "METHOD_RECOMMENDATION_ALPHA", 0.05))
    value, n, metric = _extract_replay_metric(replay_result)
    evidence = hyp.get("evidence") if isinstance(hyp.get("evidence"), dict) else {}
    hyp_id = str(hyp.get("id") or _make_hyp_id(str(hyp.get("family") or "misc"),
                                                str(hyp.get("statement") or "")))

    out = {
        "hyp_id": hyp_id,
        "family": hyp.get("family"),
        "metric": metric,
        "delta": None,
        "lift": None,
        "n_issues": int(n),
        "min_issues": min_issues,
        "verdict": "样本不足",
        "basis": "无可用回放结果",
        "direction": "none",
        "p_value": None,
        "p_adjusted": None,
    }
    if value is None:
        return out

    delta: Optional[float] = None
    basis = ""
    if isinstance(replay_result, dict) and replay_result.get("delta_red_hits") is not None:
        delta = _as_float(replay_result.get("delta_red_hits"))
        basis = "回放给出的红球命中差（相对基线）"
    elif evidence.get("baseline") is not None:
        delta = value - _as_float(evidence.get("baseline"))
        basis = f"回放实测 {value:.4f} 对比假说证据基线 {_as_float(evidence.get('baseline')):.4f}"
    else:
        delta = value - _as_float(evidence.get("lift"), 0.0)
        basis = "回放实测对比假说证据中的观测差值"

    out["delta"] = _round(delta)
    base_ref = evidence.get("baseline")
    out["lift"] = _round(delta / abs(_as_float(base_ref)) if base_ref not in (None, 0)
                         else delta)
    out["basis"] = basis

    if n < min_issues:
        out["verdict"] = "样本不足"
        out["basis"] = f"{basis}；样本 {n} 期 < 门槛 {min_issues} 期，不下结论"
        return out

    p_adj = replay_result.get("p_adjusted") if isinstance(replay_result, dict) else None
    p_raw = replay_result.get("p_value") if isinstance(replay_result, dict) else None
    out["p_value"] = _as_float(p_raw) if p_raw is not None else None
    out["p_adjusted"] = _as_float(p_adj) if p_adj is not None else None

    # 提供了校正后 p 值：以它为准（样本足但检验不显著 → 无显著差异）
    if out["p_adjusted"] is not None:
        if out["p_adjusted"] >= alpha:
            out["verdict"] = "无显著差异"
            out["direction"] = "none"
            out["basis"] = (f"{basis}；{n} 期样本，校正后 p={out['p_adjusted']:.4g} "
                            f"≥ {alpha}，与波动不可区分")
            return out
        out["verdict"] = "支持" if delta > 0 else ("反对" if delta < 0 else "无显著差异")
        out.setdefault("direction", "up" if delta > 0 else "down")
        out["direction"] = "up" if delta > 0 else ("down" if delta < 0 else "none")
        out["basis"] = (f"{basis}；{n} 期样本，校正后 p={out['p_adjusted']:.4g} < {alpha}")
        return out

    # 无 p 值：用最小可检出幅度做门槛（默认 0.05 个红球/期）
    threshold = 0.05
    if delta >= threshold:
        out["verdict"], out["direction"] = "支持", "up"
    elif delta <= -threshold:
        out["verdict"], out["direction"] = "反对", "down"
    else:
        out["verdict"], out["direction"] = "无显著差异", "none"
    out["basis"] = (f"{basis}；{n} 期样本（≥ 门槛 {min_issues}），"
                    f"最小可检出幅度 {threshold}")
    return out


# ---------------------------------------------------------------------------
# V12-21 补集反转（complement inversion）辅助函数
# ---------------------------------------------------------------------------
#
# 依据：hits(S) + hits(complement(S)) ≡ 6（红球每期 6 个号必落其内）
#       ⇒ delta_S + delta_comp ≡ 0
# 所以「显著为负」≡「其补集显著为正」。历史实现把负 delta 判 rejected 后直接丢弃，
# 使假设类对补集不封闭（half-open under complement）。以下三个函数负责把
# 负效应**无损地**转成补集候选，同时保证 FWER 不变（不计入独立检验数）。

def _hyp_filter_spec(hyp: Dict) -> Optional[Dict]:
    """取出假说绑定的维度规格 ``{key, params, target}``；没有则 None。"""
    act = hyp.get("action") if isinstance(hyp.get("action"), dict) else {}
    spec = act.get("filter") if isinstance(act.get("filter"), dict) else None
    if not spec:
        # 结构类假说走 evolution 的映射（与 patterns_runtime 同口径）
        try:
            from . import evolution
            spec = evolution.hypo_structure_filter(hyp)
        except Exception:  # noqa: BLE001
            spec = None
    if not isinstance(spec, dict) or not spec.get("key"):
        return None
    return spec


def _build_complement_hypothesis(hyp: Dict, score: Dict) -> Optional[Dict]:
    """由一条「显著为负」的假说，构造其补集候选（纯函数，不落盘）。

    返回已带 ``status="proposed"``、``complement_of`` 标记的假说 dict；
    若该假说没有可识别的维度规格（无法构造补集），返回 None。
    """
    spec = _hyp_filter_spec(hyp)
    if not spec:
        return None
    from . import discovery as DIS
    try:
        flt = DIS.make_filter(str(spec.get("key")),
                              spec.get("params") if isinstance(spec.get("params"), dict) else None)
    except KeyError:
        return None
    comp = DIS.complement_filter(flt)
    delta = score.get("delta")
    base_id = str(hyp.get("id"))
    stmt = str(hyp.get("statement") or "")
    out = {
        "id": f"{base_id}__comp",
        "family": hyp.get("family") or "complement",
        "source": "stat",
        "statement": (f"〔补集〕{stmt}（原假说显著为负 Δ={delta}，其补集等价于"
                      f"显著为正 Δ={(-float(delta)) if delta is not None else None}；"
                      f"由 hits(S)+hits(Sᶜ)≡6 恒等式得出，不是新增检验）"),
        "action": {"type": "prefer_red_filter" if (flt.get("target") or "red") == "red"
                   else "prefer_blue_filter",
                   "filter": {"key": comp["key"], "params": comp.get("params") or {},
                              "target": comp.get("target") or "red",
                              "complement": True}},
        "target": comp.get("target") or "red",
        "hyp_status": None,
        # 与 patterns_runtime 的 usable 门同口径：补集候选不自动可用，
        # 仍需 A/B（status=validated）+ 前置注册（prereg_ok）三道门。
        "status": "proposed",
        "supports": 0,
        "refutes": 0,
        "prereg_status": None,
        "prereg_ok": False,
        "complement_of": base_id,
        "is_complement": True,
        "complement_delta": (-float(delta)) if delta is not None else None,
        "evidence": dict(hyp.get("evidence") or {}),
        "created_at": _now_iso(),
    }
    return out


def _upsert_complement(reg: Dict, comp: Dict) -> bool:
    """把补集候选幂等写入登记簿。已存在（同 id）则只刷新，不新增。"""
    hyps = reg.setdefault("hypotheses", [])
    cid = comp["id"]
    for i, h in enumerate(hyps):
        if str(h.get("id")) == cid:
            # 保留已有的状态推进（supports/refutes/status），只刷新叙述与 delta
            for k in ("statement", "complement_delta", "evidence"):
                if comp.get(k) is not None:
                    h[k] = comp[k]
            return False
    hyps.append(comp)
    return True


def resolve_registry(replay_results: Optional[Sequence[Dict]],
                     registry_path: Optional[Any] = None) -> Dict:
    """按实战回放结果更新假说库状态（**本模块仅有的两个落盘入口之一**）。

    状态迁移
    --------
    * 显著支持 → ``validated``（进入 ``active_hypotheses``，供预测引擎消费）
    * 显著反对 → ``rejected``（不再参与后续预测，但留档供复盘）
    * 样本不足 → 保持 ``proposed``（不因一次小样本翻脸）
    * 已 ``rejected`` 的假说即使再次受支持也**不自动复活**（需人工 ``retired`` 后重注册），
      避免在噪声上来回抖动。

    返回 ``{validated, rejected, insufficient, unchanged, scored, registry_path}``，
    其中前三个是假说 ID 列表，``unchanged`` 是本次未改状态的条数。
    """
    reg = _load_registry(registry_path)
    hyps = reg["hypotheses"]
    results = [r for r in (replay_results or []) if isinstance(r, dict)]

    validated: List[str] = []
    rejected: List[str] = []
    insufficient: List[str] = []
    comps: List[Dict] = []
    unchanged = 0
    scored = 0
    for hyp in hyps:
        before = hyp.get("status") or "proposed"
        matched: Optional[Dict] = None
        for res in results:
            # 逐假说打分：回放结果里可带 hyp_id 精确指定，否则按方法名关联
            target = res.get("hyp_id")
            if target is not None and str(target) != str(hyp.get("id")):
                continue
            if target is None and res.get("method") is not None:
                act = hyp.get("action") if isinstance(hyp.get("action"), dict) else {}
                if str(act.get("method") or "") != str(res.get("method")):
                    continue
            matched = res
            break
        if matched is None:
            unchanged += 1
            continue
        s = score_hypothesis(hyp, matched)
        scored += 1
        hyp["last_score"] = s
        hyp["last_scored_at"] = _now_iso()
        verdict = s["verdict"]
        if verdict == "支持":
            hyp["supports"] = _as_int(hyp.get("supports"), 0) + 1
        elif verdict == "反对":
            hyp["refutes"] = _as_int(hyp.get("refutes"), 0) + 1

        if verdict == "样本不足":
            insufficient.append(str(hyp.get("id")))
            if before != "proposed" and before != "validated":
                unchanged += 1
            else:
                hyp["status"] = "proposed"
            continue
        if verdict == "支持":
            if before == "rejected":
                unchanged += 1  # 已被否决的假说不因单次支持自动复活
                continue
            hyp["status"] = "validated"
            validated.append(str(hyp.get("id")))
            continue
        if verdict == "反对":
            # V12-21 补集反转（裁决 R-8）：**不要**把负效应直接埋掉。
            #
            # hits(S) + hits(complement(S)) ≡ 6  ⇒  delta_S + delta_comp ≡ 0
            # 一个「显著为负」的过滤器，其补集就是「显著为正」的过滤器。历史实现
            # 在这里直接判 rejected，等于把一半信号当失败销毁（P2 §4.2b 实测：
            # 注入 6 号 +10% 偏置时，唯一被检出的通道正是 cold 的**负** delta）。
            #
            # 处理：原假说仍记 rejected（它作为「正向断言」确实被数据否定了，
            # 这一点必须诚实保留），同时**登记一条补集候选**，再由同一套
            # p_adjusted 口径判定它是否成立。补集与原本是同一次检验的两种写法，
            # 故**不计入 n_tested**，FWER 不变（返回体见 complements / n_tested_delta）。
            hyp["status"] = "rejected"
            rejected.append(str(hyp.get("id")))
            comp = _build_complement_hypothesis(hyp, s)
            if comp:
                comps.append(comp)
            continue
        unchanged += 1  # 无显著差异：维持原状态

    # 补集候选并入登记簿：同 spec 幂等（重复出现只更新计数，不新增条目）
    added_comps = 0
    for comp in comps:
        if _upsert_complement(reg, comp):
            added_comps += 1

    _save_registry(reg, registry_path)
    return {
        "validated": validated,
        "rejected": rejected,
        "insufficient": insufficient,
        "unchanged": unchanged,
        "scored": scored,
        "complements": [c["id"] for c in comps],
        "complements_added": added_comps,
        # FWER 不变：补集是同一次检验的另一种写法，不得计入独立检验数
        "n_tested_delta": 0,
        "registry_path": str(_registry_path(registry_path)),
    }


def active_hypotheses(registry_path: Optional[Any] = None) -> List[Dict]:
    """返回可参与后续预测的假说：``status == "validated"``。

    被 ``rejected`` / ``retired`` 的假说不出现在这里（但仍留在库中供复盘）。
    """
    reg = _load_registry(registry_path)
    return [h for h in reg["hypotheses"] if h.get("status") == "validated"]


def load_registry(registry_path: Optional[Any] = None) -> Dict:
    """读取整个假说库（只读，供 API / 前端展示生命周期）。"""
    return _load_registry(registry_path)


def registry_path() -> str:
    """默认假说库路径（字符串，便于展示与日志）。"""
    return str(_registry_path(None))


# ==========================================================================
# V10-02 前置注册（pre-registration）—— 写者实现
#
# 为什么需要它（docs/reviews/R2_STATISTICS_REVIEW.md §3.1 + V10 方案 §3.3）：
#   V4/V6 已证明在 3515 期数据上"什么都找不到"，而每次新搜索都会抬高多重比较成本。
#   前置注册把"我找了 N 次"变成"我只检验了事先承诺的 M 个"，M 是**先于读结果**承诺的。
#
# 【B1 修复要点（S2 二次评审）】
#   v2.0 方案只在读侧（patterns_runtime.usable）用了 prereg_ok，**从未规定谁写它**。
#   而 U1–U5 测试全部 monkeypatch load_registry（直接伪造 prereg_ok=True 的记录），
#   所以即使实现者永远不置 True，测试也全绿 —— 即方案自己警告的"白做"。
#   本节的 evaluate_prereg() 就是**唯一的写者**，调用点写死在 api_app._scheduler_step。
# ==========================================================================

# ---- R2 六条约束的常量（缺任一条，前置注册就只是一句口号） ----
MIN_BATCH_SIZE = 10          # R2：每批最小注册数。**这是下限，不是"达标值"**
                             #     （N=100 时：M_min=1→FWER 0.994、10→0.401、50→0.098）
MAX_BATCH_SIZE = 200         # R5：一次性批量上限
MAX_REGISTER_PER_ISSUE = 10  # R5：注册速率上限（条/期）
FROZEN_ISSUES = 150          # R4：注册后最短冻结期（≈1 年，一个完整检验窗）

PREREG_STATUSES = ("registered", "running", "passed", "provisional", "failed", "invalidated")
# 终态：不可回退到 registered（R4 切断 optional stopping）
PREREG_TERMINAL = ("passed", "failed", "invalidated")


def _spec_canonical(hyp: Dict) -> Dict:
    """实验规范形式（R3：**不含** ``evaluate_after_issue``）。

    改截止期号只是"改预约"，不产生新实验 —— 这封死了
    "同一实验换 20 个截止期 = 20 批 = FWER 0.642" 的洗白路径。
    """
    act = hyp.get("action") if isinstance(hyp.get("action"), dict) else {}
    filt = act.get("filter") if isinstance(act.get("filter"), dict) else {}
    return {
        "family_key": str(filt.get("key") or hyp.get("family") or ""),
        "params": filt.get("params") if isinstance(filt.get("params"), dict) else {},
        "target": str(filt.get("target") or "red"),
        "test": str(hyp.get("prereg_test") or "t"),
        "direction": str(hyp.get("prereg_direction") or "two-sided"),
        "min_samples": _as_int(hyp.get("prereg_min_samples"), 100),
    }


def spec_exp_id(hyp: Dict) -> str:
    """``exp_id = sha1(canonical(spec))``；同 spec 重注册返回同一 id（幂等）。"""
    import hashlib
    canon = json.dumps(_spec_canonical(hyp), ensure_ascii=False,
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def _registry_counters(reg: Dict) -> Dict:
    """库根计数器（R1 的唯一校正分母所在）。缺失时惰性初始化。"""
    c = reg.get("counters")
    if not isinstance(c, dict):
        c = {}
        reg["counters"] = c
    c.setdefault("n_tested_cumulative", 0)
    c.setdefault("last_issue", None)
    c.setdefault("batches", [])
    return c


def register_prereg(hyps: Optional[Sequence[Dict]],
                    registry_path: Optional[Any] = None,
                    issue: Optional[str] = None,
                    evaluate_after_issue: Optional[int] = None,
                    batch_id: Optional[str] = None,
                    reason: str = "") -> Dict:
    """把候选规律**登记为先于读结果的实验**（R1–R6 约束在此强制）。

    与 ``register_hypotheses`` 的关系：本函数在注册**之后**补写 prereg_* 字段，
    不改变原去重/累加 attempts 语义（``attempts`` 仍由前者维护）。

    约束校验（任一不满足则**不登记**，返回 errors）：
      * R5：``len(hyps) > MAX_BATCH_SIZE`` → 拒绝；同批超过一个期的速率上限 → 拒绝
      * R2：``len(hyps) < MIN_BATCH_SIZE`` → 登记但状态保持 registered、**标记 not_judgeable**
            （凑不满 10 条必须并入下一批，不得单独成批判定）

    返回 ``{registered, deduped, batch_id, batch_size, judgeable, errors, registry_path}``。
    """
    import time
    items = [h for h in (hyps or []) if isinstance(h, dict)]
    errors: List[str] = []
    if len(items) > MAX_BATCH_SIZE:
        errors.append(f"R5：单批 {len(items)} 条超过上限 {MAX_BATCH_SIZE}"
                      f"（超限需分批，或补充 reason 后人工审批）")
        if not reason:
            return {"registered": 0, "deduped": 0, "batch_id": None, "batch_size": len(items),
                    "judgeable": False, "errors": errors, "registry_path": None}

    reg = _load_registry(registry_path)
    counters = _registry_counters(reg)
    judgeable = len(items) >= MIN_BATCH_SIZE           # R2
    bid = batch_id or f"b{int(time.time())}"

    existing = {h.get("exp_id"): h for h in reg["hypotheses"] if h.get("exp_id")}
    registered = deduped = 0
    now = time.time()
    for hyp in items:
        eid = spec_exp_id(hyp)
        target = existing.get(eid)
        if target is not None:
            # R3：同 spec 重注册 → 同一 exp_id，只更新预约，不新建记录
            target["evaluate_after_issue"] = _as_int(evaluate_after_issue, target.get("evaluate_after_issue") or 0)
            target["frozen_until_issue"] = _as_int(target.get("evaluate_after_issue"), 0) + FROZEN_ISSUES
            deduped += 1
            continue
        hyp = dict(hyp)
        hyp["exp_id"] = eid
        hyp["prereg_at"] = now
        hyp["registered_issue"] = issue
        hyp["batch_id"] = bid
        hyp["evaluate_after_issue"] = _as_int(evaluate_after_issue,
                                              _as_int(issue, 0) if issue else 0)
        hyp["frozen_until_issue"] = _as_int(hyp["evaluate_after_issue"], 0) + FROZEN_ISSUES
        hyp["prereg_status"] = "registered"
        hyp["prereg_ok"] = False
        hyp["scope"] = "cumulative"
        hyp["divergence"] = False
        hyp["alpha"] = None
        hyp["checked_at"] = None
        hyp["prereg_reason"] = "registered" if judgeable else (
            f"R2：本批仅 {len(items)} 条 < {MIN_BATCH_SIZE}，不得单独成批判定，并入下一批")
        reg["hypotheses"].append(hyp)
        existing[eid] = hyp
        registered += 1

    counters["batches"].append({
        "batch_id": bid, "size": len(items), "issue": issue,
        "judgeable": judgeable, "reason": reason, "at": _now_iso(),
    })
    _save_registry(reg, registry_path)
    return {"registered": registered, "deduped": deduped, "batch_id": bid,
            "batch_size": len(items), "judgeable": judgeable, "errors": errors,
            "registry_path": str(_registry_path(registry_path))}


def _bonferroni(p: float, m: int) -> float:
    """Bonferroni 校正（m 为检验次数；m<=0 视为不校正）。"""
    try:
        return min(1.0, float(p) * max(1, int(m)))
    except (TypeError, ValueError):
        return 1.0


def evaluate_prereg(registry_path: Optional[Any] = None,
                    issue: Optional[str] = None,
                    alpha: float = 0.05,
                    dry_run: bool = False,
                    draws: Optional[Sequence[Dict]] = None) -> Dict:
    """对**已到期**的注册实验做判定，回写 ``prereg_*`` 字段并落盘。

    这是 V10 前置注册的**唯一写者**。调用点写死为 ``api_app._scheduler_step()``，
    紧跟 ``_evolution_step()`` 之后、生成预测之前（见 V10 方案 §3.3.5b）。

    判定对象（四个条件同时满足才进入判定，否则原样跳过）：
      1. ``prereg_status`` 属于 ("registered", "running")
      2. ``current_issue >= evaluate_after_issue``（到期）
      3. 所在 batch 的注册数 >= MIN_BATCH_SIZE（R2）
      4. ``current_issue >= frozen_until_issue``（R4 冻结期已过）

    判定口径（R1）：``p_batch`` 与 ``p_cumulative`` 都算，
    **以 ``p_cumulative`` 为唯一判定依据**；两者结论不同置 ``divergence=True``。

    幂等语义：
      * 同一 exp_id 重复调用不新增记录，只更新 prereg_* 字段；
      * 已处终态（passed/failed/invalidated）且仍在冻结期内 → 直接跳过，不重判（R4）；
      * ``provisional`` 允许后续重判（可能升级为 passed），其余终态不可回退。

    ``dry_run=True``：照算并返回结果，但不写盘。
    ``draws``：可选，注入开奖序列供测试；缺省从 db 读取。
    """
    import time
    from . import discovery

    reg = _load_registry(registry_path)
    counters = _registry_counters(reg)
    now_issue = _as_int(issue, counters.get("last_issue") or 0)
    if counters.get("last_issue") is None and issue:
        counters["last_issue"] = issue

    if draws is None:
        try:
            from . import db
            draws = db.load_draws()
        except Exception as e:                       # noqa: BLE001
            print(f"[hypothesis] evaluate_prereg 无法读取开奖数据：{e}")
            return {"checked": 0, "skipped": 0, "error": str(e)}

    checked = skipped = 0
    passed_ids: List[str] = []
    provisional_ids: List[str] = []
    failed_ids: List[str] = []
    invalidated_ids: List[str] = []
    updated: List[str] = []

    # 先统计每个 batch 的实际规模（R2）
    batch_sizes: Dict[str, int] = {}
    for h in reg["hypotheses"]:
        b = h.get("batch_id")
        if b:
            batch_sizes[b] = batch_sizes.get(b, 0) + 1

    for hyp in reg["hypotheses"]:
        status = str(hyp.get("prereg_status") or "")
        if status not in ("registered", "running", "provisional"):
            skipped += 1
            continue
        # 条件 2：到期
        if now_issue < _as_int(hyp.get("evaluate_after_issue"), 0):
            skipped += 1
            continue
        # 条件 3（R2）：批规模够
        if batch_sizes.get(hyp.get("batch_id"), 0) < MIN_BATCH_SIZE:
            skipped += 1
            continue
        # 条件 4（R4）：冻结期已过
        if now_issue < _as_int(hyp.get("frozen_until_issue"), 0):
            skipped += 1
            continue
        # R4：终态在冻结期内不得重判
        if status in PREREG_TERMINAL and now_issue < _as_int(hyp.get("frozen_until_issue"), 0):
            skipped += 1
            continue

        spec = _spec_canonical(hyp)
        if not spec.get("family_key") or spec["family_key"] not in discovery.FAMILY_BY_KEY:
            hyp.update(prereg_status="invalidated", prereg_ok=False, checked_at=time.time(),
                       prereg_reason=f"维度族 {spec.get('family_key')!r} 不在维度表内",
                       divergence=False)
            invalidated_ids.append(str(hyp.get("id")))
            updated.append(str(hyp.get("id")))
            checked += 1
            continue

        try:
            filt = discovery.make_filter(spec["family_key"], spec.get("params") or {})
            rep = discovery.regime_of_filter(draws, filt)
        except Exception as e:                       # noqa: BLE001
            hyp.update(prereg_status="invalidated", prereg_ok=False, checked_at=time.time(),
                       prereg_reason=f"as-of 回放失败：{e}", divergence=False)
            invalidated_ids.append(str(hyp.get("id")))
            updated.append(str(hyp.get("id")))
            checked += 1
            continue

        # p 值取法（真实踩坑记录）：regime_of_filter 的顶层**没有** p_adjusted_recent，
        # 近期窗口的校正 p 在 rep["recent"]["p_adjusted"]（近 150 期），
        # 全局分段的在 rep["eras"][-1]["p_adjusted"]。
        # 取错字段会让 p 恒为 1.0 → 所有实验被判 failed（本体已被 E1 测试捕获）。
        p_raw = (rep.get("recent") or {}).get("p_adjusted")
        if p_raw is None:
            eras = rep.get("eras") or []
            if eras:
                p_raw = eras[-1].get("p_adjusted")
        try:
            p_val = float(p_raw) if p_raw is not None else 1.0
        except (TypeError, ValueError):
            p_val = 1.0

        checked += 1
        counters["n_tested_cumulative"] = _as_int(counters.get("n_tested_cumulative"), 0) + 1
        m_cum = _as_int(counters.get("n_tested_cumulative"), 1)
        m_batch = batch_sizes.get(hyp.get("batch_id"), 1)
        p_batch = _bonferroni(p_val, m_batch)
        p_cum = _bonferroni(p_val, m_cum)

        hyp["p_batch"] = round(p_batch, 10)
        hyp["p_cumulative"] = round(p_cum, 10)
        hyp["n_tested_batch"] = m_batch
        hyp["n_tested_cumulative"] = m_cum
        hyp["alpha"] = float(alpha)
        hyp["checked_at"] = time.time()
        hyp["divergence"] = bool((p_batch < alpha) != (p_cum < alpha))

        # 判定：以累计口径为唯一依据（R1）；R6：批内显著但累计不显著 → provisional
        if p_cum < alpha:
            hyp.update(prereg_status="passed", prereg_ok=True, scope="cumulative",
                       prereg_reason=f"累计口径校正后 p={p_cum:.4g} < {alpha}")
            passed_ids.append(str(hyp.get("id")))
        elif p_batch < alpha:
            hyp.update(prereg_status="provisional", prereg_ok=False, scope="cumulative",
                       prereg_reason=(f"R6：批内 p={p_batch:.4g} 显著但累计 p={p_cum:.4g} "
                                      f"不显著（n_tested_cumulative={m_cum}），不得用于出号"))
            provisional_ids.append(str(hyp.get("id")))
        else:
            hyp.update(prereg_status="failed", prereg_ok=False, scope="cumulative",
                       prereg_reason=f"累计口径校正后 p={p_cum:.4g} >= {alpha}")
            failed_ids.append(str(hyp.get("id")))
        updated.append(str(hyp.get("id")))

    if not dry_run and (checked or updated):
        _save_registry(reg, registry_path)

    return {
        "checked": checked, "skipped": skipped,
        "passed": len(passed_ids), "provisional": len(provisional_ids),
        "failed": len(failed_ids), "invalidated": len(invalidated_ids),
        "updated_ids": updated,
        "passed_ids": passed_ids, "provisional_ids": provisional_ids,
        "failed_ids": failed_ids, "invalidated_ids": invalidated_ids,
        "n_tested_cumulative": _as_int(counters.get("n_tested_cumulative"), 0),
        "dry_run": bool(dry_run),
    }
