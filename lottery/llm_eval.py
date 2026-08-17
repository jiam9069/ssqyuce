"""LLM 离线评估（M3.1）：三通道 walk-forward —— stat / stat_llm / random。

诚实边界：双色球开奖是独立随机事件。本模块不承诺提升命中率，目标是把
「LLM 相对纯统计/随机基线的差异与成本」量化为可复现、可入库、可在 UI 查看的证据：
- stat：纯统计 + 结构约束（use_llm=False / use_ml=False，与现有离线评估口径一致）；
- stat_llm：stat + LLM 三轮流程（同生产预测）；
- random：同注数均匀随机（固定种子，可复现）。

指标：红球平均命中 / 蓝球命中率 / ≥五等奖率 / 奖级 / ROI；
成本：LLM 调用数、token 估算、每百万 token 价格（LOTT_LLM_EVAL_PRICE_PER_1M 可配）、耗时。
统计：paired wilcoxon / sign-test，BH 校正跨对比。
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from . import config, db, engine as E, evaluate as EV, llm_client


def _run_with_timeout(fn: Callable, timeout_s: float):
    """在 daemon 线程中执行 fn，硬超时兜底（上游连接级 stall 时避免拖死评估）。"""
    box: Dict[str, object] = {}

    def _worker() -> None:
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"LLM 通道执行超时（>{timeout_s:.0f}s），该期按失败计")
    if "e" in box:
        raise box["e"]  # type: ignore[arg-type]
    return box.get("r")

CHANNELS = ("stat", "stat_llm", "random")
CHANNEL_LABEL = {"stat": "纯统计", "stat_llm": "统计+LLM", "random": "随机基线"}


def _pair_p(delta: List[float]) -> Dict:
    """paired 显著性检验：wilcoxon 优先，样本不足用 sign-test。"""
    from scipy import stats as _st
    d = np.array(delta, dtype=float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"p": 1.0, "method": "no-difference", "mean_delta": 0.0}
    try:
        if len(nz) >= 5:
            res = _st.wilcoxon(nz)
            p = float(res.pvalue)
            method = "wilcoxon"
        else:
            pos = int(np.sum(nz > 0))
            p = 2 * min(_st.binom.cdf(pos, len(nz), 0.5),
                        1 - _st.binom.cdf(pos - 1, len(nz), 0.5))
            p = float(min(p, 1.0))
            method = "sign-test"
        return {"p": round(p, 4), "method": method,
                "mean_delta": round(float(d.mean()), 5)}
    except Exception:  # noqa: BLE001
        return {"p": 1.0, "method": "n/a", "mean_delta": round(float(d.mean()), 5)}


def _bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg 多重检验校正（保序）。"""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    for rank, i in enumerate(order, 1):
        adj[i] = min(1.0, pvals[i] * n / rank)
    running = 1.0
    for i in reversed(order):
        running = min(running, adj[i])
        adj[i] = running
    return adj


def run_llm_eval(draws: List[Dict], issues: Optional[int] = None,
                 n_tickets: Optional[int] = None, seed: Optional[int] = None,
                 progress_cb: Optional[Callable[[float, str], None]] = None) -> Dict:
    """三通道 walk-forward 评估并入库 llm_eval_results 表。"""
    issues = int(issues or config.LLM_EVAL_ISSUES)
    n_tickets = int(n_tickets or config.LLM_EVAL_TICKETS)
    seed = int(seed if seed is not None else config.LLM_EVAL_SEED)
    issues = max(3, min(issues, max(3, len(draws) - 301)))
    run_id = f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    start = len(draws) - issues - 1
    if start < 300:
        start = 300
    issues_actual = len(draws) - 1 - start

    runs: Dict[str, List[Dict]] = {ch: [] for ch in CHANNELS}
    usage: Dict[str, Dict] = {ch: {"tokens": 0, "cost_usd": 0.0,
                                   "duration_ms": 0, "calls": 0} for ch in CHANNELS}
    llm_empty_issues = 0   # LLM 通道失败（无候选）的期数，如实计入口径
    per_metric: Dict[str, Dict[str, List[float]]] = {
        ch: {k: [] for k in ("red_hits_mean", "blue_hit_rate", "prize_rate_ge5", "roi")}
        for ch in CHANNELS}
    ctx = None
    total = issues_actual * len(CHANNELS)
    done = 0

    llm_client.usage_reset()
    t_start = time.time()
    for i in range(start, len(draws) - 1):
        history = draws[:i]          # 只允许目标期之前的数据（walk-forward）
        target = draws[i]            # 实际开奖
        if ctx is None:
            ctx = E.constraint_ctx(history)

        def _safe(tickets: List[Dict]) -> List[Dict]:
            tickets = (tickets or [])[:n_tickets]
            if tickets:
                return tickets
            # 兜底：空结果时用均匀约束票
            out = []
            rng_fb = random.Random(seed * 7919 + i + 7)
            for _ in range(n_tickets):
                t = E.sample_stat_ticket(None, None, ctx, rng_fb, uniform=True)
                if t:
                    out.append(t)
            return out

        # ---- stat 通道 ----
        res_stat = E.predict_next(history, use_llm=False, n_tickets=n_tickets,
                                  persist=False, use_ml=False,
                                  rng=random.Random(seed * 7919 + i))
        runs["stat"].append(EV.tickets_result(_safe(res_stat["tickets"]), target))

        # ---- stat_llm 通道（计量 LLM 用量；看门狗防上游连接级挂起）----
        llm_client.usage_reset()
        t0 = time.time()
        try:
            res_llm = _run_with_timeout(
                lambda: E.predict_next(history, use_llm=True, n_tickets=n_tickets,
                                       persist=False, use_ml=False,
                                       rng=random.Random(seed * 7919 + i + 1),
                                       llm_samples=1,   # 评估降本：每期仅 1 次 LLM 选号采样
                                       llm_verify=False),  # 校验轮仅生产启用（口径见 note）
                timeout_s=900,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[llm_eval] stat_llm 通道异常/超时，该期回退 stat 输出: {e}")
            res_llm = {"tickets": []}
        dt_ms = int((time.time() - t0) * 1000)
        us = llm_client.usage_snapshot()
        tokens = (us["prompt_chars"] + us["completion_chars"]) // 4
        llm_cands = res_llm["tickets"] if res_llm else []
        if not llm_cands:
            # LLM 通道失败（上游超时/连接挂起/无候选）→ 生产口径为回退纯统计输出，如实记录
            llm_empty_issues += 1
            llm_cands = res_stat["tickets"]
        runs["stat_llm"].append(EV.tickets_result(_safe(llm_cands), target))
        usage["stat_llm"]["tokens"] += tokens
        usage["stat_llm"]["cost_usd"] += tokens / 1e6 * config.LLM_EVAL_PRICE_PER_1M
        usage["stat_llm"]["duration_ms"] += dt_ms
        usage["stat_llm"]["calls"] += us["calls"]

        # ---- random 通道（独立种子）----
        rand_tickets = []
        rng_r = random.Random(seed * 7919 + i + 2)
        for _ in range(n_tickets):
            t = E.sample_stat_ticket(None, None, ctx, rng_r, uniform=True)
            if t:
                rand_tickets.append(t)
        runs["random"].append(EV.tickets_result(rand_tickets, target))

        # 逐期指标
        for ch in CHANNELS:
            r = runs[ch][-1]
            n = max(1, len(r["red_hits"]))
            per_metric[ch]["red_hits_mean"].append(
                float(np.mean(r["red_hits"])) if r["red_hits"] else 0.0)
            per_metric[ch]["blue_hit_rate"].append(
                float(np.mean(r["blue_hits"])) if r["blue_hits"] else 0.0)
            per_metric[ch]["prize_rate_ge5"].append(
                float(np.mean(np.array(r["levels"]) >= 5)) if r["levels"] else 0.0)
            per_metric[ch]["roi"].append((r["reward"] - n * 2.0) / (n * 2.0))

        done += 3
        if progress_cb:
            progress_cb(done / total,
                        f"评估中 {i - start + 1}/{issues_actual} 期（stat / stat_llm / random 三通道）...")

    elapsed = round(time.time() - t_start, 1)

    # ---- 汇总与 paired 检验 ----
    summary_meta: Dict[str, Dict] = {}
    for ch in CHANNELS:
        agg = EV.aggregate(runs[ch])
        summary_meta[ch] = {
            "metrics": agg,
            "tokens": usage[ch]["tokens"],
            "cost_usd": round(usage[ch]["cost_usd"], 4),
            "duration_ms": usage[ch]["duration_ms"],
            "calls": usage[ch]["calls"],
        }

    metric_keys = ("red_hits_mean", "blue_hit_rate", "prize_rate_ge5", "roi")
    comparisons: List[Dict] = []
    pair_order = (("stat_llm", "stat"), ("stat_llm", "random"), ("stat", "random"))
    for a, b in pair_order:
        pvals = []
        for k in metric_keys:
            delta = [x - y for x, y in zip(per_metric[a][k], per_metric[b][k])]
            res = _pair_p(delta)
            pvals.append(res["p"])
            comparisons.append({"pair": f"{a}_vs_{b}", "metric": k,
                                "mean_delta": res["mean_delta"], "p": res["p"],
                                "method": res["method"]})
        adj = _bh_adjust(pvals)
        for k, p_adj in zip(metric_keys, adj):
            comparisons.append({"pair": f"{a}_vs_{b}", "metric": k + "_adj",
                                "p_adj": round(p_adj, 4)})

    # ---- 落库（逐期明细 + 每通道汇总）----
    rows: List[Dict] = []
    for idx in range(issues_actual):
        for ch in CHANNELS:
            r = runs[ch][idx]
            rows.append({
                "channel": ch,
                "issue": target_issue(draws, start, idx),
                "red_hits": round(per_metric[ch]["red_hits_mean"][idx], 3),
                "blue_hit": int(round(per_metric[ch]["blue_hit_rate"][idx])),
                "prize_level": r.get("best_level", 0),
                "roi": round(per_metric[ch]["roi"][idx], 4),
            })
    for ch in CHANNELS:
        rows.append({
            "channel": ch,
            "issue": None, "red_hits": None, "blue_hit": None, "prize_level": 0, "roi": None,
            "tokens": usage[ch]["tokens"],
            "cost_usd": usage[ch]["cost_usd"],
            "duration_ms": usage[ch]["duration_ms"],
            "metrics_json": (summary_meta[ch]["metrics"] if ch != "stat_llm" else {
                **summary_meta[ch]["metrics"],
                "_meta": {"llm_empty_issues": llm_empty_issues, "llm_samples": 1},
            }),
            "p_values_json": comparisons,
        })
    db.save_llm_eval_rows(run_id, created_at, issues_actual, n_tickets, seed, rows)

    per_issue: Dict[str, List[Dict]] = {}
    for idx in range(issues_actual):
        iss = target_issue(draws, start, idx)
        for ch in CHANNELS:
            per_issue.setdefault(ch, []).append({
                "issue": iss,
                "red_hits": round(per_metric[ch]["red_hits_mean"][idx], 3),
                "blue_hit": int(round(per_metric[ch]["blue_hit_rate"][idx])),
                "roi": round(per_metric[ch]["roi"][idx], 4),
            })

    result = {
        "run_id": run_id,
        "created_at": created_at,
        "window_issues": issues_actual,
        "tickets": n_tickets,
        "seed": seed,
        "llm_samples": 1,
        "llm_empty_issues": llm_empty_issues,
        "elapsed_seconds": elapsed,
        "channels": summary_meta,
        "per_issue": per_issue,
        "comparisons": comparisons,
        "llm_used_models": _used_llm_models(),
        "note": (
            "三通道 walk-forward（每期只用此前数据）。Δ 与 paired p 值回答「LLM 值不值开」；"
            "random 为同注数均匀随机基线。LLM 通道存在模型噪声，stat/random 通道固定种子可复现。"
            "cost 为按 token 估算（LOTT_LLM_EVAL_PRICE_PER_1M，默认 $1/1M token），仅作展示。"
            "评估口径：stat_llm = 观察 + 选号两轮（llm_samples=1），第三轮校验（M3.2）仅生产启用，"
            "其边际成本/收益待后续增量评估。若上游 LLM 中途失败，该期 stat_llm 如实回退为 stat 输出，"
            "并在指标旁标注 llm_empty_issues（LLM 失败期数）。"
        ),
    }
    print(f"[llm_eval] 完成 run_id={run_id} 期数={issues_actual} 耗时={elapsed}s，"
          f"LLM token={usage['stat_llm']['tokens']}")
    return result


def target_issue(draws: List[Dict], start: int, idx: int) -> str:
    return draws[start + idx]["issue"]


def _used_llm_models() -> List[str]:
    try:
        return [c.get("model") for c in config.llm_model_list()]
    except Exception:  # noqa: BLE001
        return []


def load_latest_report() -> Optional[Dict]:
    """读取最近一次已完成的 LLM 评估 run（供 UI 展示）。"""
    run_id = db.latest_llm_eval_run()
    if not run_id:
        return None
    return db.load_llm_eval_run(run_id)
