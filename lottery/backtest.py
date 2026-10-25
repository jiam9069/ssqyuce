"""Walk-forward 回测框架：每条规律在历史上滚动验证，做显著性检验与分级。

原则：预测第 i+1 期时只允许使用前 i 期数据（严格防未来信息）。
分级：A = 显著性通过（BH 校正后 p<0.05）且方向符合假设；B = raw p<0.20 且方向符合；
      其余为 C（不参与预测，仅展示）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

import datetime

from . import db, features as F, patterns as P
from .stats.rigor import benjamini_hochberg as _bh_truth
from .stats.rigor import wilson_lower_bound as _wilson_lower_truth
from .stats.rigor import wilson_interval as _wilson_interval_truth


# 双色球开奖节奏：每周二、四、日（2003-02-23 首期即为周日）。
# 实测 2021-2025 每年 150~151 期、期号连续无跳号、间隔以 2/3 天为主（每周三期）。
# 历史上每年最多 154 期（2003-2005），**从未出现 4xx 期号**——所以
# 「seq+1 > 999 就跳到下一年 001」这种写法虽然合法，却掩盖了「本地数据落后了一整年」
# 这个真实故障（实测：本地只有 2026118（10-15，周四），而 10-17 应开 2026119，
# 旧逻辑把用户看到的「目标期号」显示成了明年第 401 期）。
DRAW_WEEKDAYS = (1, 3, 6)      # 周一=0 → 周二/周四/周日（与 config.DRAW_WEEKDAYS 同口径）
MAX_SEQ_PER_YEAR = 160         # 真实最多 154，留出余量只用于识别异常


def next_issue(issue: str) -> str:
    """期号 +1（同一自然年内递增，期末不跨年跳号）。

    注意：这是**纯序号推算**，它假设本地数据与官方同步。若本地数据陈旧（例如官方
    已开 2026119 而本地只到 2026118），本函数给出的 2026119 仍然正确；
    真正会出错的是「跨年跳号」的老写法，见 ``DRAW_WEEKDAYS`` 上方注释。
    """
    year = int(issue[:4])
    seq = int(issue[4:])
    return f"{year:04d}{seq + 1:03d}"


def expected_issue(draws: List[Dict], today: Optional[datetime.date] = None) -> Dict:
    """按**真实开奖日历**推算"下一个应开期号"，并给出数据新鲜度。

    为什么要这么做：期号不能靠猜。用户的诉求是「上期实际是 2026118、目标期号应准确」，
    而准确的唯一依据是「现在离上一期开奖过了几个开奖日」。

    口径（实测本机 3515 期）：
      * 开奖日为每周二/四/日（``DRAW_WEEKDAYS``），每年 150~151 期、期号连续；
      * 以最后一期开奖日为起点，数「今天之前还应当开过几次」，逐次 +1；
      * 跨年时**不**跳到 001，而是按「同一年内最多 ``MAX_SEQ_PER_YEAR`` 期」判断——
        超过就说明本地数据缺了整整一年，此时给出 ``stale_year=True`` 让界面明说。

    返回::

        {issue, basis_issue, basis_date, days_since, missed_draws, stale_days,
         next_draw_date, expected_draws_missing, stale_year, note_zh}
    """
    today = today or datetime.date.today()
    if not draws:
        return {"issue": None, "basis_issue": None, "basis_date": None,
                "days_since": None, "missed_draws": 0, "stale_days": None,
                "next_draw_date": None, "expected_draws_missing": True,
                "stale_year": False, "note_zh": "本地暂无开奖数据，无法推算目标期号"}
    last = draws[-1]
    try:
        base_date = datetime.date(*map(int, str(last["date"]).split("-")))
    except (TypeError, ValueError):
        base_date = today
    seq = int(str(last["issue"])[4:])
    year = int(str(last["issue"])[:4])
    # 逐个数开奖日：base_date 之后的第一个开奖日，到「今天或今天之前的最后一个开奖日」
    # 之间一共还有几个开奖日 = 本地缺了几期
    d = base_date
    missed = 0
    while True:
        d = d + datetime.timedelta(days=1)
        if d.weekday() in DRAW_WEEKDAYS:
            if d >= today:
                break
            missed += 1
    next_draw = d
    issue_seq = seq + missed + 1
    stale_year = issue_seq > MAX_SEQ_PER_YEAR
    stale_days = max(0, (today - base_date).days)
    return {
        "issue": f"{year:04d}{issue_seq:03d}",
        "basis_issue": last.get("issue"),
        "basis_date": last.get("date"),
        "days_since": stale_days,
        "missed_draws": missed,
        "stale_days": stale_days,
        "next_draw_date": next_draw.isoformat(),
        "expected_draws_missing": missed > 0,
        "stale_year": stale_year,
        "note_zh": (
            f"本地最新开奖 {last.get('issue')}（{last.get('date')}），"
            + (f"按每周二/四/日节奏，其后已又有 {missed} 个开奖日仍未入库，"
               f"数据落后 {stale_days} 天；目标期号应为 {year:04d}{issue_seq:03d}" if missed else
               f"目标期号应为 {year:04d}{issue_seq:03d}（下一开奖日 {next_draw.isoformat()}）")
            + ("；⚠️ 期号推算超过一年量级，本地数据可能缺了整整一年，请先检查数据源"
               if stale_year else "")
        ),
    }


def _draw_value(outcome: str, draw: Dict) -> float:
    """数值类规律的目标值：和值 / 跨度 / AC 值。"""
    if outcome == "numeric_span":
        return float(max(draw["reds"]) - min(draw["reds"]))
    if outcome == "numeric_ac":
        return float(F.ac_value(draw["reds"]))
    return float(sum(draw["reds"]))


def bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR 校正。

    **V40 薄壳**：公式唯一真源是 `stats.rigor.benjamini_hochberg`
    （审计 P2-1：三份并行实现 → 一份真源；recon 数值对拍三份 max diff = 0.0）。
    """
    return _bh_truth(pvals)


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
    """正态近似 Wilson 置信区间。

    **V40 薄壳**：公式唯一真源是 `stats.rigor.wilson_lower_bound` /
    `wilson_interval`（审计 P2-1：四份并行实现 → 一份真源）。
    历史边界语义 `total==0 → (0.0, 0.0)` 与真源区间版的 `(None, None)` 不同 ——
    本壳**保留旧语义**：调用方 `backtest_pattern` 把 CI 直接写进
    `result["backtest"]["ci_lower"]` 数值聚合链，None 会改变下游形状。
    """
    if total == 0:
        return (0.0, 0.0)
    lo = _wilson_lower_truth(successes, total, z)
    _, hi = _wilson_interval_truth(successes, total, z)
    return (lo, hi)


def backtest_pattern(pattern: Dict, draws: List[Dict], min_start: int = 120) -> Dict:
    """在 draws（时间升序）上滚动验证 pattern。min_start: 至少多少期历史才开始验证。"""
    key = pattern["key"]
    outcome = pattern["outcome"]
    kind = "red" if outcome == "red_fav" else ("blue" if outcome == "blue_fav" else "numeric")
    horizon = pattern.get("horizon", 1)

    hit_counts, fav_sizes, exp_vec, p0_vec = [], [], [], []
    margin_series = []  # per-trigger (trigger_index, hits, exp, margin)
    n_triggers = 0
    numeric_vals, numeric_all, numeric_base = [], [], None

    for i in range(min_start, len(draws)):
        history = draws[:i]
        res = P.run_pattern(pattern, history)
        if not res["triggered"]:
            continue
        n_triggers += 1
        action = res["action"]

        if outcome.startswith("numeric_"):
            hist_vals = np.array([_draw_value(outcome, d) for d in history])
            mean = float(np.mean(hist_vals)) if len(hist_vals) else 0.0
            target = draws[i]
            numeric_vals.append(abs(_draw_value(outcome, target) - mean))
            if numeric_base is None:
                numeric_all = [abs(_draw_value(outcome, d) - mean)
                               for d in draws[min_start:]]
                numeric_base = float(np.mean(numeric_all)) if numeric_all else 0.0
            continue

        fav = action.get(outcome, [])
        if not fav:
            continue
        k = len(fav)
        fav_sizes.append(k)

        if kind == "red":
            p0_i = 1 - math.comb(33 - k, 6) / math.comb(33, 6)
            if horizon > 1:
                # 多期：期望 = horizon * 单期期望（总命中次数）
                exp_i = horizon * 6 * k / 33
                targets = [draws[i + h] for h in range(horizon) if i + h < len(draws)]
                hits_i = sum(len(set(t["reds"]) & set(fav)) for t in targets)
            else:
                exp_i = 6 * k / 33
                targets = [draws[i]]
                hits_i = len(set(targets[0]["reds"]) & set(fav))
        else:  # blue
            exp_i = 1 - (1 - k / 16) ** horizon
            p0_i = exp_i
            targets = [draws[i + h] for h in range(horizon) if i + h < len(draws)]
            hits_i = int(any(t["blue"] in fav for t in targets))

        hit_counts.append(hits_i)
        exp_vec.append(exp_i)
        p0_vec.append(p0_i)
        margin_series.append({
            "trigger_issue": draws[i]["issue"],
            "hits": int(hits_i),
            "expected": round(float(exp_i), 4),
            "margin": round(float(hits_i - exp_i), 4),
        })

    result = {"key": key, "name_zh": pattern["name_zh"], "kind": pattern["kind"],
              "desc": pattern["desc"], "horizon": horizon, "n": n_triggers,
              "backtest": {}}

    if outcome.startswith("numeric_"):
        if len(numeric_vals) >= 10 and numeric_base is not None:
            arr = np.array(numeric_vals)
            margin = float(numeric_base - np.mean(arr))
            res_tt = stats.ttest_1samp(arr, np.mean(numeric_all))
            p_two = float(res_tt.pvalue)
            direction = "toward_mean" if margin > 0 else "away_from_mean"
            result["margin"] = margin
            result["p_value"] = min(1.0, p_two / 2 if direction == "toward_mean" else 1.0 - p_two / 2)
            result["direction"] = direction
            lo, hi = _wilson_ci(int(np.mean(arr) < np.mean(numeric_all)), len(arr))
            metric_name = {"numeric_sum": "均值偏差|sum|",
                            "numeric_span": "均值偏差|span|",
                            "numeric_ac": "均值偏差|AC|"}.get(outcome, "numeric")
            result["backtest"] = {
                "metric": metric_name, "avg_triggered": float(np.mean(arr)),
                "avg_baseline": numeric_base, "margin": margin,
                "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            }
        else:
            result["margin"], result["p_value"], result["direction"] = 0.0, 1.0, "none"
        return result

    if n_triggers < 10 or not hit_counts:
        result.update(margin=0.0, p_value=1.0, p_adj=1.0, direction="none", grade="C")
        result["backtest"] = {
            "n": n_triggers, "avg_hits": 0.0, "expected": 0.0,
            "hit_rate_at_least1": 0.0, "p0": 0.0, "avg_fav_size": 0.0,
            "ci_lower": None, "ci_upper": None, "series": [],
        }
        return result

    hits = np.array(hit_counts, dtype=float)
    exp = np.array(exp_vec, dtype=float)
    p0 = np.array(p0_vec, dtype=float)
    avg = float(hits.mean())
    avg_exp = float(exp.mean())
    at_least1 = int(np.sum(hits > 0))
    rate = at_least1 / n_triggers
    margin = avg - avg_exp
    direction = "above" if margin >= 0 else "below"
    diffs = hits - exp
    t_res = stats.ttest_1samp(diffs, 0.0)
    p_two = float(t_res.pvalue)
    p_one = p_two / 2 if direction == "above" else 1.0 - p_two / 2
    # Wilson CI on hit rate
    ci_lo, ci_hi = _wilson_ci(at_least1, n_triggers)
    result.update(margin=margin, p_value=float(np.clip(p_one, 0.0, 1.0)),
                  direction=direction, grade="C")
    result["backtest"] = {
        "n": n_triggers, "avg_hits": avg, "expected": avg_exp, "margin": margin,
        "hit_rate_at_least1": rate, "p0": float(p0.mean()),
        "avg_fav_size": float(np.mean(fav_sizes)) if fav_sizes else 0.0,
        "t_stat": float(t_res.statistic),
        "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
        "series": margin_series[:200],  # 最近 200 个触发点
    }
    return result


def run_all_backtests(draws: List[Dict], min_start: int = 300) -> List[Dict]:
    results = [backtest_pattern(p, draws, min_start=min_start) for p in P.PATTERNS]
    for r in results:
        r["sample_size"] = r.get("n", 0)
    set_res = [r for r in results if r["direction"] in ("above", "below")]
    pvals = [r["p_value"] for r in set_res]
    adj = bh_adjust(pvals)
    for r, a in zip(set_res, adj):
        r["p_adj"] = a
    for r in results:
        r.setdefault("p_adj", 1.0)
        r.setdefault("grade", "C")
        claimed = r.get("claimed_direction", "above")
        if r.get("n", 0) >= 30 and r["direction"] in ("above", "below"):
            if r.get("p_adj", 1.0) < 0.05:
                if r["direction"] == claimed:
                    r["grade"] = "A"
                else:
                    r["grade"] = "C"
                    r["refuted"] = True
            elif r["p_value"] < 0.20 and r["direction"] == claimed:
                r["grade"] = "B"
        if r.get("refuted"):
            r["backtest"]["note"] = (
                "显著但方向与假设相反：该类别号码的命中显著低于随机，"
                "假设被证伪（过拟合/后见之明典型）。"
            )
        db.save_pattern_results([r])
    return results


def summarize(results: List[Dict]) -> Dict:
    grades = {"A": 0, "B": 0, "C": 0}
    for r in results:
        grades[r.get("grade", "C")] += 1
    return {
        "total": len(results),
        "grades": grades,
        "significant": [r for r in results if r.get("grade") in ("A", "B")],
        "notes": "彩票为随机事件：样本外回测能有效识别过拟合规律。"
                 "未发现显著规律属于正常且诚实的结论。",
    }
