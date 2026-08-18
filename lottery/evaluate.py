"""评估体系：奖级判定、离线 walk-forward 引擎回测（对照随机基线）、在线预测对照。"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import db, engine as E

# 奖级现金（二等/一等为浮动奖，用历史均值近似，标注 *)
PRIZE_CASH = {1: 6_500_000.0, 2: 180_000.0, 3: 3000.0, 4: 200.0, 5: 10.0, 6: 5.0}
PRIZE_NAME = {1: "一等奖(≈*)" , 2: "二等奖(≈*)", 3: "三等奖", 4: "四等奖", 5: "五等奖", 6: "六等奖"}


def prize_level(reds_pred: List[int], blue_pred: int,
                reds_act: List[int], blue_act: int) -> int:
    r = len(set(reds_pred) & set(reds_act))
    b = blue_pred == blue_act
    if r == 6 and b:
        return 1
    if r == 6:
        return 2
    if r == 5 and b:
        return 3
    if r == 5 or (r == 4 and b):
        return 4
    if r == 4 or (r == 3 and b):
        return 5
    if b:
        return 6
    return 0


def tickets_result(tickets: List[Dict], draw: Dict) -> Dict:
    rewards = []
    levels = []
    red_hits = []
    blue_hits = []
    for t in tickets:
        lvl = prize_level(t["reds"], t["blue"], draw["reds"], draw["blue"])
        levels.append(lvl)
        r = len(set(t["reds"]) & set(draw["reds"]))
        red_hits.append(r)
        b = int(t["blue"] == draw["blue"])
        blue_hits.append(b)
        rewards.append(PRIZE_CASH.get(lvl, 0.0))
    return {
        "red_hits": red_hits,
        "blue_hits": blue_hits,
        "levels": levels,
        "reward": float(sum(rewards)),
        "best_level": max(levels) if levels else 0,
    }


def aggregate(runs: List[Dict]) -> Dict:
    if not runs:
        return {}
    all_red = np.concatenate([r["red_hits"] for r in runs])
    all_blue = np.concatenate([r["blue_hits"] for r in runs])
    all_lvl = np.concatenate([r["levels"] for r in runs])
    rewards = np.array([r["reward"] for r in runs])
    n_tickets_total = int(sum(len(r["red_hits"]) for r in runs))
    return {
        "issues": len(runs),
        "tickets": n_tickets_total,
        "red_hits_mean": float(np.mean(all_red)),
        "red_hits_dist": {int(k): int(v) for k, v in
                          zip(*np.unique(all_red, return_counts=True))},
        "blue_hit_rate": float(np.mean(all_blue)),
        "prize_rate_ge5": float(np.mean(all_lvl >= 5)),
        "prize_rate_ge4": float(np.mean(all_lvl >= 4)),
        "best_level_run": int(np.max(all_lvl)) if len(all_lvl) else 0,
        "reward_total": float(np.sum(rewards)),
        "reward_mean_per_issue": float(np.mean(rewards)),
        "cost_total": n_tickets_total * 2.0,
        "roi": (float(np.sum(rewards)) - n_tickets_total * 2.0) / max(1.0, n_tickets_total * 2.0),
    }


def offline_backtest(draws: List[Dict], issues: int = 120,
                     n_tickets: int = 10, use_llm: bool = False,
                     seed: int = 7) -> Dict:
    """walk-forward：只用第 i 期之前数据预测第 i+1 期，对照实际开奖；
    同时用相同注数的均匀随机票做基线。"""
    import random
    from . import engine as E

    runs_sys, runs_rand = [], []
    start = max(len(draws) - issues - 1, 300)
    rng = random.Random(seed)
    ctx = None
    for i in range(start, len(draws) - 1):
        history = draws[:i]
        target = draws[i]  # 实际开奖 (第 i 期)
        # 引擎离线回测不启用 ML：ML 的诚实证据由 /api/ml/eval 单独给出
        res = E.predict_next(history, use_llm=use_llm, n_tickets=n_tickets,
                             persist=False, use_ml=False)
        runs_sys.append(tickets_result(res["tickets"], target))
        # 随机基线：均匀约束票
        if ctx is None:
            ctx = E.constraint_ctx(history)
        rand_tickets = []
        for _ in range(n_tickets):
            t = E.sample_stat_ticket(None, None, ctx, rng, uniform=True)
            if t:
                rand_tickets.append(t)
        runs_rand.append(tickets_result(rand_tickets, target))

    agg_sys = aggregate(runs_sys)
    agg_rand = aggregate(runs_rand)
    return {
        "n_issues": len(runs_sys),
        "n_tickets_per_issue": n_tickets,
        "use_llm": use_llm,
        "system": agg_sys,
        "random_baseline": agg_rand,
        "delta_red_hits": agg_sys.get("red_hits_mean", 0) - agg_rand.get("red_hits_mean", 0),
        "delta_blue_rate": agg_sys.get("blue_hit_rate", 0) - agg_rand.get("blue_hit_rate", 0),
        "note": "对照随机基线：红色球平均命中差/蓝球命中率差越接近 0，"
                "越说明系统预测不优于随机——这是随机彩票的正常结论。",
    }


def online_check() -> Dict:
    """把已保存的预测与对应实际开奖对照，写入 eval_results。"""
    draws = db.load_draws()
    draw_by_issue = {d["issue"]: d for d in draws}
    issues = db.recent_prediction_issues(200)
    checked = 0
    for issue in issues:
        if issue not in draw_by_issue:
            continue
        preds = db.load_predictions(issue)
        if not preds:
            continue
        # 兼容 M4.1 前生成的预测：首次对照时补写方法快照
        if not db.load_eval_meta(issue):
            db.save_eval_meta(issue, preds, result={"llm_used": any(str(t.get("method", "")).startswith("llm:") for t in preds)})
        by_method = {}
        for t in preds:
            by_method.setdefault(str(t.get("method", "mixed")), []).append(t)
        all_res = tickets_result(preds, draw_by_issue[issue])
        db.save_eval(issue, int(np.mean(all_res["red_hits"])), int(np.mean(all_res["blue_hits"])),
                     all_res["reward"], len(preds))
        for method, method_tickets in by_method.items():
            db.save_eval_details(issue, method, method_tickets, draw_by_issue[issue])
        checked += 1
    rows = db.load_eval()
    if rows:
        reds = [r["red_hits"] for r in rows]
        blues = [r["blue_hit"] for r in rows]
        rewards = [r["reward"] for r in rows]
        cost = sum(r["ticket_count"] for r in rows) * 2.0
        summary = {
            "checked": len(rows),
            "avg_red_hits": float(np.mean(reds)),
            "blue_hit_rate": float(np.mean(blues)),
            "reward_total": float(np.sum(rewards)),
            "cost_total": cost,
            "roi": (float(np.sum(rewards)) - cost) / max(1.0, cost),
        }
    else:
        summary = {"checked": 0}
    return {"newly_checked": checked, "summary": summary, "rows": rows[-30:]}