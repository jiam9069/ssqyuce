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
        "levels_dist": {int(k): int(v) for k, v in
                        zip(*np.unique(all_lvl, return_counts=True))},
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
        #
        # ⚠️ rng 必须透传（V36，2026-10-24 审计 P0-4）：
        # 原先这里不传 rng，`predict_next` 回落到未播种的 `random.Random()`，
        # 而本函数的 `rng = random.Random(seed)` 只用于下面的随机基线 ——
        # 于是"同 seed 可复现"只对基线成立、对系统组不成立。实测同数据同
        # seed=7 两次得到 delta_red_hits = -0.0400 / -0.1133，跨过
        # hypothesis 的 |delta| ≥ 0.05 判决阈值 ⇒ 同一份数据判决可能翻转。
        # 修法：每期派生一个**互相独立但确定**的子种子，避免各期共用同一条
        # 随机流（那样后期结果会取决于前面跑过多少期，期数一变全变）。
        _period_rng = random.Random(seed * 1_000_003 + i)
        res = E.predict_next(history, use_llm=use_llm, n_tickets=n_tickets,
                             persist=False, use_ml=False, rng=_period_rng)
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


# ---------- V10-04b：报表口径分列（R1 P1-4） ----------
# 方法族口径与 methods.FAMILIES 对齐；未知族一律归 "other"，**绝不**并进 stat/ml。
METHOD_FAMILIES = ("llm", "stat", "blend", "ml", "uniform", "other")

# 各族的人话名称（报表/前端直接用，避免前后端各写一份）
FAMILY_NAME_ZH = {
    "llm": "LLM 推理通道",
    "stat": "统计基线",
    "blend": "融合（blend）",
    "ml": "ML 模型",
    "uniform": "均匀随机对照",
    "other": "其它/未标注",
}


def method_family(method: object) -> str:
    """取方法族：``llm:minimax-m3`` → ``llm``，``ml`` → ``ml``，``stat:freq`` → ``stat``。"""
    fam = str(method or "").strip().lower().split(":", 1)[0]
    return fam if fam in METHOD_FAMILIES else "other"


def _family_block(rows: List[Dict]) -> Dict:
    """单族汇总（组内各 method 的注级合并）。返回体不含任何跨族字段。"""
    n_tickets = len(rows)
    reds = [float(r["red_hits"]) for r in rows]
    blues = [float(r["blue_hit"]) for r in rows]
    levels = [int(r["prize_level"] or 0) for r in rows]
    rewards = [float(r["reward"] or 0.0) for r in rows]
    costs = [float(r["ticket_cost"] or 2.0) for r in rows]
    cost_total = float(sum(costs))
    reward_total = float(sum(rewards))
    return {
        "n_issues": len({r["issue"] for r in rows}),
        "n_tickets": n_tickets,
        "methods": sorted({str(r["method"]) for r in rows}),
        "red_hits_mean": float(np.mean(reds)) if reds else None,
        "blue_hit_rate": float(np.mean(blues)) if blues else None,
        "prize_rate_ge5": float(np.mean([1.0 if lv >= 5 else 0.0 for lv in levels])) if levels else None,
        "best_level": max(levels) if levels else 0,
        "reward_total": reward_total,
        "cost_total": cost_total,
        "roi": ((reward_total - cost_total) / cost_total) if cost_total else None,
    }


def method_breakdown(limit_issues: Optional[int] = None) -> Dict:
    """按方法族**分列**汇总方法战绩（V10-04b / R1 P1-4：不得相加）。

    为什么必须分列、且**不能**给"全族合计"：
      - ``eval_results`` 一行是"该期所有方法所有注"的混合均值（旧实现还对整数
        取均值再 ``int()`` 截断，把 1.5 抹成 1），用它做方法对比会把 LLM 的
        差表现与统计基线平均掉 —— 这是"LLM 到底行不行"永远测不出来的根因；
      - ``eval_details`` 本就按 ``(issue, method, seq)`` 逐注存，天然可分列。

    因此本函数返回 ``families``（llm / stat / blend / ml / uniform / other 各自一块），
    **不返回** ``all`` / ``total`` 之类跨族字段：调用方无法"顺手相加"，
    不同注数、不同方法族的命中率没有可加的公共分母。
    """
    rows = db.load_eval_details(limit_issues=limit_issues)
    by_family: Dict[str, List[Dict]] = {}
    for r in rows:
        by_family.setdefault(method_family(r.get("method")), []).append(r)
    families = {fam: _family_block(by_family[fam])
                for fam in METHOD_FAMILIES if fam in by_family}
    # 逐 method 明细（族内细分：如 llm:minimax-m3 vs llm:glm-5.3）
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)
    methods = []
    for method in sorted(by_method):
        block = _family_block(by_method[method])
        block["method"] = method
        block["family"] = method_family(method)
        methods.append(block)
    return {
        "n_issues": len({r["issue"] for r in rows}),
        "n_tickets": len(rows),
        "families": families,
        "family_names": {fam: FAMILY_NAME_ZH[fam] for fam in families},
        "methods": methods,
        "note": ("方法战绩按族分列：llm / stat / blend / ml / uniform 各算各的，"
                 "**不得相加或取加权平均**（各族注数与口径不同，混合后会掩盖真实差异）。"
                 "本报表只描述历史命中，不构成中奖预测或投资建议。"),
    }


def online_check() -> Dict:
    """把已保存的预测与对应实际开奖对照，写入 eval_results（组合口径）与
    eval_details（方法口径），并按方法族分列汇总。

    V10-04b（R1 P1-4）两处口径修正：
      1. ``eval_results`` 的均值**不再 int() 截断**，保留浮点（旧写法把 1.5 抹成 1）；
      2. ``summary`` 不再输出"跨方法混合的 avg_red_hits / blue_hit_rate"，
         改为 ``by_family`` 分列（+ 组合级的金额字段，金额是同一期内不同方法
         共同产生的，不属于方法比较）。

    ⚠️ ``eval_results`` 一行就是该期所有方法所有注的合计，因此它只作
    「组合级/在线对照表」使用，**不得**用于方法对比（见 ``method_breakdown()``）。
    """
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
        # 组合口径：保留浮点均值（不再 int() 截断），仅供在线对照表展示
        db.save_eval(issue, round(float(np.mean(all_res["red_hits"])), 4),
                     round(float(np.mean(all_res["blue_hits"])), 4),
                     all_res["reward"], len(preds))
        for method, method_tickets in by_method.items():
            db.save_eval_details(issue, method, method_tickets, draw_by_issue[issue])
        checked += 1
    rows = db.load_eval()
    summary: Dict = {"checked": len(rows)}
    if rows:
        rewards = [float(r["reward"] or 0.0) for r in rows]
        cost = sum(int(r["ticket_count"] or 0) for r in rows) * 2.0
        breakdown = method_breakdown()
        # 金额/ROI 是同一期内的组合合计（非方法比较），与分列口径并列给出
        summary.update({
            "scope": "portfolio",          # 明确这是组合口径，不是方法口径
            "tickets": sum(int(r["ticket_count"] or 0) for r in rows),
            "reward_total": float(np.sum(rewards)),
            "cost_total": cost,
            "roi": (float(np.sum(rewards)) - cost) / max(1.0, cost),
            "by_family": breakdown["families"],
            "mixed_note": ("avg_red_hits / blue_hit_rate 已移除：混算 llm 与 stat/ml "
                           "的命中率会把差异平均掉。方法战绩请看 by_family 分列。"),
        })
    return {"newly_checked": checked, "summary": summary, "rows": rows[-30:]}


# ---------- V10-05：两套样本并排（仅追加；不改上面任何既有签名） ----------

def samples_side_by_side(mode: str = "replay") -> Dict:
    """真实交战样本（``eval_details``）与离线回放样本（``replay_results``）**并排**。

    报表侧的稳定入口（实现与测试在 ``lottery/replay.py``，此处懒导入以避免
    ``evaluate`` ↔ ``replay`` 的循环 import）。

    **为什么只并排、不合并**：真实交战样本只有 30 注量级（MDE ≈ ±32%），离线回放
    是几千注量级的 walk-forward 样本；两者样本量、口径、独立性都不同，相加得到的
    "总命中率"没有公共分母，会把"回放很准"错误地记到"真实预测"头上。
    返回体两块各自带 ``source`` / ``n_issues`` / ``n_tickets`` / ``mde``，
    **不含**任何跨来源合计字段，且 ``mixable=False``。
    """
    from . import replay as R
    return R.samples_side_by_side(mode=mode)
