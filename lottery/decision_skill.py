"""LLM 决策技能评估（P2 从 api_app 的 V4 域切出的共享服务）。

## 这不是端点，是服务

本模块只含**纯计算函数**（Wilson 下界、决策技能、边际价值、可用规律计数），
被多个端点共用（`/api/llm/cost`、`/api/llm/arm`、调度器、战斗回放）。

## 它们是"LLM 到底有没有用"的唯一可测口径

V10 的裁决是：LLM 被错误地部署在"预测下一期开奖"这个数学上不可达的岗位上。
挪到决策层之后，它的价值必须用**确定性的判分器**来衡量 —— 也就是这里。
`decide_from_wilson` / `decision_skill` 给出的是"相对随机基线有没有优势"的
统计判断，而不是模型的自评分数。模型的自评 confidence 不是概率（实测集中在
50.1–51.3，几乎是个常数）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import config, db

#: 决策技能的随机基线：LLM 臂相对基线臂「胜」的概率在无技能时就是 0.5
DECISION_SKILL_BASELINE = 0.5

def _wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% 区间下界（小样本比例不做正态近似）。

    **V40 薄壳**：公式唯一真源是 `stats.rigor.wilson_lower_bound`
    （审计 P2-1：四份并行实现 → 一份真源）。原注释称「内联是为了
    不跨模块 import 私有函数」—— 但 stats 包是**纯数学层**（不 import
    业务模块的纪律有测试守卫），从这里 import 它没有反向依赖问题；
    内联反而造成「真源修 bug 副本不跟进」的漂移风险（V36 P0-1 的根因）。
    """
    from .stats.rigor import wilson_lower_bound as _truth
    return _truth(hits, n, z)


def decide_from_wilson(wilson_lower: Optional[float],
                       baseline: float = DECISION_SKILL_BASELINE) -> Dict:
    """判据本体（纯函数，便于钉边界）：Wilson 下界 ≤ 基线 → 应关停。

    ``wilson_lower is None``（无决定性配对样本）→ ``should_disable=None``：
    **无证据不等于证否**，此时既不关停也不宣称有技能。
    """
    if wilson_lower is None:
        return {
            "should_disable": None,
            "baseline": float(baseline),
            "wilson_lower": None,
            "reason_zh": "没有可判定的配对样本：下界无从计算，不做关停判断（也不宣称有技能）。",
        }
    w = float(wilson_lower)
    disable = w <= float(baseline)
    return {
        "should_disable": bool(disable),
        "baseline": float(baseline),
        "wilson_lower": w,
        "reason_zh": (
            f"决策技能 Wilson 95% 下界 {w:.4f} "
            + ("≤" if disable else ">")
            + f" 随机基线 {float(baseline):.2f} → "
            + ("应关停该决策通道（保守口径：连下界都追不平随机，不能证明有技能）。"
               if disable else
               "下界高于随机基线，暂不关停（仍受 llm_can_pick 等其他开关约束）。")),
    }


def decision_skill(rows: Optional[List[Dict]] = None, *,
                   baseline: float = DECISION_SKILL_BASELINE,
                   max_tickets: int = 20) -> Dict:
    """LLM **决策技能**：同期逐注配对比较 ``llm:*`` 臂 vs 基线族（stat/ml/blend/uniform）。

    为什么用"胜率 vs 0.5"而不是均值差：技能被定义为"决策比随机更好"的**二分类**
    能力，胜率可以直接套 Wilson 区间；均值差需要分布假设，而在 30 注量级上任何
    近似都不可信（``rigor._min_detectable_effect``：n=30 → 40.7%）。

    口径（与 ``battle.llm_arm_report`` 的配对纪律一致，见 V10-09）：
      * **同期**：两臂必须在同一 ``issue`` 都有记录；
      * **同注数**：该期两臂注数必须相等（不同规模的期剔除并计数）；
      * **逐注配对**：按 ``seq`` 一一对应；逐期取**均值**后比较（注间相关，
        不能把每注当独立样本）；
      * **平局剔除**：两臂逐期均值相等时该期不进 Wilson 分母（否则平局会被
        当成"没输"，系统性抬高胜率）——平局数单独给出。
    """
    from . import evaluate as EV
    if rows is None:
        rows = db.load_eval_details()
    rows = [dict(r) for r in (rows or [])]
    by_issue: Dict[str, Dict[str, List[Dict]]] = {}
    for r in rows:
        issue = str(r.get("issue") or "")
        method = str(r.get("method") or "")
        if not issue or not method:
            continue
        by_issue.setdefault(issue, {}).setdefault(method, []).append(r)

    wins = losses = ties = 0
    dropped_mismatch = 0
    used_issues: List[str] = []
    for issue, methods in by_issue.items():
        llm_rows = sorted((r for m, rs in methods.items() if m.startswith("llm:")
                           for r in rs), key=lambda x: int(x.get("seq") or 0))
        base_rows = sorted((r for m, rs in methods.items()
                            if not m.startswith("llm:")
                            and EV.method_family(m) in ("stat", "ml", "blend", "uniform")
                            for r in rs), key=lambda x: int(x.get("seq") or 0))
        if not llm_rows or not base_rows:
            continue
        if len(llm_rows) != len(base_rows):
            dropped_mismatch += 1
            continue
        if max_tickets and len(llm_rows) > int(max_tickets):
            continue
        llm_mean = sum(float(r.get("red_hits") or 0) for r in llm_rows) / len(llm_rows)
        base_mean = sum(float(r.get("red_hits") or 0) for r in base_rows) / len(base_rows)
        if llm_mean > base_mean:
            wins += 1
        elif llm_mean < base_mean:
            losses += 1
        else:
            ties += 1
            continue
        used_issues.append(issue)

    decisive = wins + losses
    win_rate = (wins / decisive) if decisive else None
    wilson = _wilson_lower(wins, decisive) if decisive else None
    return {
        "ok": True,
        "metric": "decision_skill_win_rate",
        "n_issues_compared": decisive + ties,
        "n_decisive": decisive,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "dropped_ticket_mismatch": dropped_mismatch,
        "win_rate": None if win_rate is None else round(float(win_rate), 6),
        "wilson_lower": None if wilson is None else round(float(wilson), 6),
        **decide_from_wilson(wilson, baseline=baseline),
        "issues": used_issues,
        "note_zh": ("技能 = LLM 决策在**同期配对**中胜过基线（stat/ml/blend/uniform）"
                    "的比例；无技能时该比例应为 0.5。**平局剔除**且平局数单列。"
                    "小样本下 Wilson 下界天然偏低 → 关停是**保守**决定而非「证明无用」；"
                    "反过来，样本不足时也不得宣称有技能。"),
    }


def _usable_decision_count() -> Dict:
    """分子①：LLM 产出的**可落地决策**数（契约可落地 = 维度 key 在维度表内）。"""
    from . import llm_lab
    claims = list((llm_lab.load_claims() or {}).get("claims") or [])
    try:
        from . import discovery
        known = set(discovery.FAMILY_BY_KEY.keys())
    except Exception:                                  # noqa: BLE001
        known = set()
    usable = 0
    for c in claims:
        c = c or {}
        key = str(c.get("key") or c.get("dimension") or "").strip()
        if not key:
            act = c.get("action") if isinstance(c.get("action"), dict) else {}
            filt = act.get("filter") if isinstance(act.get("filter"), dict) else {}
            key = str(filt.get("key") or "")
        if key and key in known:
            usable += 1
    return {"n_claims": len(claims), "n_usable": usable,
            "source": "data/llm_claims.json",
            "definition_zh": ("可用决策 = 维度 key 落在 discovery.FAMILY_BY_KEY 内、"
                              "能注册成可检验实验的 claim（契约可落地）。")}


def _passed_pattern_count() -> Dict:
    """分子②：**通过**规律的条数（前置注册 ``passed``，V10-02 的唯一写者判定）。"""
    from . import hypothesis
    try:
        reg = hypothesis.load_registry() or {}
    except Exception as e:                             # noqa: BLE001
        return {"n_passed": None, "n_hypotheses": None,
                "source": "data/hypotheses.json", "error": f"{type(e).__name__}: {e}"}
    hyps = list(reg.get("hypotheses") or [])
    passed = sum(1 for h in hyps if str((h or {}).get("prereg_status") or "") == "passed")
    validated = sum(1 for h in hyps if str((h or {}).get("status") or "") == "validated")
    return {"n_passed": passed, "n_validated": validated, "n_hypotheses": len(hyps),
            "source": "data/hypotheses.json",
            "definition_zh": ("通过规律 = prereg_status==passed（前置注册按预注册检验通过）。"
                              "validated（A/B 晋升）单列，不与 passed 相加。")}


def llm_marginal_value(cost: Optional[Dict] = None) -> Dict:
    """「AI 边际价值」：每 ¥1 换回的可用决策数 / 通过规律数。

    **零数据不返回 0**：没有已计费成本时两个比值都是 ``None`` 并说明原因；
    有成本但分子为 0 时返回 ``0.0``（"花了钱什么都没换到"是结论，不是缺数据）。
    """
    if cost is None:
        from .llm_client import cost_report
        cost = cost_report()
    yuan = cost.get("yuan")
    decisions = _usable_decision_count()
    patterns = _passed_pattern_count()

    def _ratio(numerator):
        if yuan is None or numerator is None or float(yuan) <= 0:
            return None
        return round(float(numerator) / float(yuan), 6)

    if yuan is None:
        reason = "成本日志里没有已计费调用（yuan=null），比值无法计算 —— 不用 0 假装有值。"
    elif float(yuan) <= 0:
        reason = "已计费成本为 0，比值无意义。"
    else:
        reason = ("比值 = 分子 ÷ 已计费人民币估算额。分子为 0 时给 0.0（真实结论："
                  "花了钱没换到可审计产物），分母缺失时给 null（缺数据）——两者必须区分展示。")
    return {
        "yuan": yuan,
        "per_yuan_usable_decisions": _ratio(decisions.get("n_usable")),
        "per_yuan_passed_patterns": _ratio(patterns.get("n_passed")),
        "numerator_usable_decisions": decisions,
        "numerator_passed_patterns": patterns,
        "reason_zh": reason,
        "honesty_zh": ("这两个数**不是** ROI，也不代表 AI 有价值：LLM 默认不出号"
                       "（llm_can_pick=False），可用决策与通过规律都可能在真随机世界里"
                       "长期为 0。本指标只回答「花掉的每一块钱换回了多少可审计产物」。"),
    }
