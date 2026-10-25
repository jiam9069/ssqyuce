"""V11 S5：规律生命周期 —— 判断规律是否在衰减，并适时「退居二线」（monitor）。

用户诉求（2026-10-18）
---------------------
「用 AI 来判断哪些规律可能掉下去，适时把这些规律预测退居二线，只观察不使用了。」

设计见 ``docs/reviews/AI_DEMOTION_DESIGN.md``。本模块是本需求的**全部实现**，
核心是三层结构，**AI 只在最外一层**，且每一层都能被关掉而系统照常运行：

    ① 统计层（金标准，纯规则、零 LLM）    build_payload / decide_rule_state
    ② 生命周期层（确定性状态机 + 滞后）    step_lifecycle（纯函数）/ run_cycle（IO）
    ③ AI 层（只能「读表说话」）            judge_with_llm（可选）
             ↓ 输出被 validate_judgements 逐条核对 → 计入 llm_demotion_skill

三条不可违反的纪律（与 config.py 的注释、设计文档一致）
-----------------------------------------------------
1. **动作权归确定性代码**：``decide_rule_state`` 是纯函数、零 LLM 依赖；
   LLM 的 ``verdict`` 只是「复核意见」，**永远不改变状态**。
   R3 评审已确立：AI 输出必须能被确定性代码验证；而「AI 同意才 demote」会死锁。
2. **不得制造死锁**：LLM 不可用 / 状态文件损坏 / 指标算不出 → 全部降级为
   「不动作」或「按原样放行」，**绝不会**让规律无法使用，也不会让衰减的规律赖在一线。
   不变式：LLM 可用与否，``states`` 结果**逐字段相同**（有测试守着）。
3. **退居二线 = monitor，不是 rejected**：``monitor`` 是「证据不足、继续观察」，
   ``rejected`` 是「有反向证据」。本模块**任何路径都不写 rejected**。

关键函数
--------
* :func:`build_metrics`    —— 从多尺度扫描 + 阶段画像构造结构化指标（可复现）
* :func:`decide_rule_state`—— **纯函数**：指标 + 当前状态 + streak → 新状态
* :func:`step_lifecycle`   —— **纯函数**：把一次判定推进状态机（含滞后）
* :func:`apply_states`     —— 把状态翻译到快照（monitor → usable=False）
* :func:`run_cycle`        —— 一次完整周期（IO：读库、算指标、推进、落盘）
* :func:`validate_judgements` / :func:`skill_stats` —— AI 输出的机器校验与技能度量
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config

STATE_VERSION = "v11-1"
STATE_FILE = getattr(config, "LIFECYCLE_STATE_FILE", config.DATA_DIR / "rule_lifecycle.json")

#: 生命周期状态机里的三个状态。
#: ``active``  —— 规律的偏置在出号链路里生效；
#: ``monitor`` —— **退居二线**：只观察、只展示，不进 usable_rules；
#: ``retired`` —— 人工/规则永久退役（本模块不会自动产生，留给人工接口）。
STATES = ("active", "monitor", "retired")
#: 自动迁移允许写入的目标态。**刻意不含 retired / rejected**（纪律 3）。
AUTO_TARGETS = ("active", "monitor")

#: 判定方向
DIRECTIONS = ("hold", "demote", "upgrade")
#: LLM 允许输出的 verdict
VERDICTS = ("keep", "demote", "upgrade")

#: 判据代号（写进证据，便于用户与测试逐条核对）
CRITERION_LABELS = {
    "D1": "长期窗口一致为负（|Δ| 达到该档检出下限）",
    "D2": "最近两个阶段连续为负（幅度达到该段检出下限的一半）",
    "D3": "近窗加权校正后显著为负",
    "U1": "长期窗口一致为正且通过升级前置条件",
}

#: 两次阶段之间允许的最短间隔（期）。低于此值的"连续两段"可能是同一批数据的重复切分。
_MIN_ERA_GAP = 1


# ---------------------------------------------------------------------------
# ① 统计层：结构化指标
# ---------------------------------------------------------------------------

def _num(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
            return None
        return f
    except (TypeError, ValueError):
        return None


def decide_rule_state(metrics: Dict, current_state: str = "unmanaged",
                      streak: int = 0, dwell: int = 0,
                      thresholds: Optional[Dict] = None) -> Dict:
    """**纯函数**：给定指标与当前状态，给出这一次判定的方向（不动状态）。

    这是整个机制的**唯一权威判据**，零 LLM 依赖、无 IO、可穷举单测。

    参数
    ----
    metrics: :func:`build_metrics` 的产物（或同构 dict）。缺字段一律视为"算不出"，
             对应判据**不成立**（宁可不动作，也不误判）。
    current_state: ``"unmanaged"`` / ``"active"`` / ``"monitor"`` / ``"retired"``。
    streak: 当前已累计的同向次数（由状态机维护，这里只用于展示）。
    dwell: 自上次迁移以来经过的周期数。
    thresholds: ``{"mde_frac": 0.5, "alpha": 0.05}``，缺省用模块常量。

    返回::

        {"direction": "hold"|"demote"|"upgrade",
         "criteria": ["D1", ...],          # 命中的判据代号
         "reasons": ["...", ...],          # 人读理由（中文，可直接进 UI）
         "evidence": {...},                # 判定所依据的关键数字（可复算）
         "metrics_ok": bool}               # 指标是否足以判定
    """
    th = dict(thresholds or {})
    alpha = float(th.get("alpha", 0.05))
    mde_frac = float(th.get("mde_frac", 0.5))
    m = metrics if isinstance(metrics, dict) else {}

    scales = m.get("scales") if isinstance(m.get("scales"), dict) else {}
    eras = m.get("eras") if isinstance(m.get("eras"), list) else []
    recent = m.get("recent") if isinstance(m.get("recent"), dict) else {}
    prereq = m.get("upgrade_prereq") if isinstance(m.get("upgrade_prereq"), dict) else {}

    d1 = _d1_long_window(scales, alpha)
    d2 = _d2_two_eras(eras, mde_frac)
    d3 = _d3_recent(recent, alpha)
    u1 = _u1_upgrade(scales, alpha, prereq)

    criteria = [c for c, hit in (("D1", d1), ("D2", d2), ("D3", d3)) if hit]
    reasons: List[str] = []
    evidence: Dict[str, Any] = {
        "scales_all": _slim(scales.get("all")),
        "eras_tail": [_slim(e) for e in eras[-2:]],
        "recent": _slim(recent),
    }
    metrics_ok = bool(scales.get("all") and _num((scales.get("all") or {}).get("delta")) is not None)

    if criteria:
        for code in criteria:
            reasons.append(_criterion_reason(code, evidence))
        return {"direction": "demote", "criteria": criteria,
                "reasons": reasons, "evidence": evidence, "metrics_ok": metrics_ok}

    if u1:
        return {"direction": "upgrade", "criteria": ["U1"],
                "reasons": [_criterion_reason("U1", evidence)],
                "evidence": evidence, "metrics_ok": metrics_ok}

    reasons.append("三条衰减判据（D1/D2/D3）与升级判据（U1）均未触发："
                   "没有足够证据说明这条规律在衰减或恢复，维持现状")
    return {"direction": "hold", "criteria": [], "reasons": reasons,
            "evidence": evidence, "metrics_ok": metrics_ok}


def _slim(block: Any) -> Dict:
    """只保留判定需要的字段（也让前端展示与证据核对更稳定）。"""
    if not isinstance(block, dict):
        return {}
    keys = ("n", "n_eff", "delta", "z", "p_raw", "p_value", "p_adjusted",
            "mde_abs", "mde", "significant", "credible", "era")
    return {k: block.get(k) for k in keys if k in block}


def _d1_long_window(scales: Dict, alpha: float) -> bool:
    """D1：长期档一致为负，且幅度达到该档检出下限（否则只是噪声级的负）。"""
    blk = scales.get("all") if isinstance(scales.get("all"), dict) else {}
    delta = _num(blk.get("delta"))
    p = _num(blk.get("p_raw"))
    mde = _num(blk.get("mde_abs"))
    if delta is None or p is None or mde is None:
        return False
    return bool(delta < 0 and p < alpha and abs(delta) >= mde)


def _d2_two_eras(eras: Sequence[Dict], mde_frac: float) -> bool:
    """D2：最近两个（n>=60 的）阶段连续为负，且各自幅度达到该段 MDE 的一半。

    为什么用 0.5×MDE 而不是要求显著：单阶段显著在 24 个阶段里本来就罕见
    （实测约 3%），要求"两段都显著"永远不成立；而"方向一致 + 量级不可忽略"
    是一个可以稳定复算、也不会被单段波动骗到的门槛。**本判据不声称显著。**
    """
    usable = [e for e in (eras or []) if isinstance(e, dict)
              and (e.get("n") is None or (_num(e.get("n")) or 0) >= 60)
              and _num(e.get("delta")) is not None]
    if len(usable) < 2:
        return False
    a, b = usable[-2], usable[-1]
    for blk in (a, b):
        delta = _num(blk.get("delta"))
        mde = _num(blk.get("mde"))
        if delta is None or delta >= 0:
            return False
        if mde is not None and mde > 0 and abs(delta) < mde_frac * mde:
            return False
    return True


def _d3_recent(recent: Dict, alpha: float) -> bool:
    """D3：近窗加权效应为负，且校正后 p < alpha（沿用 regime 的锁定校正口径）。"""
    delta = _num(recent.get("delta"))
    p_adj = _num(recent.get("p_adjusted"))
    if p_adj is None:
        p_adj = _num(recent.get("p_value"))
    if delta is None or p_adj is None:
        return False
    return bool(delta < 0 and p_adj < alpha)


def _u1_upgrade(scales: Dict, alpha: float, prereq: Dict) -> bool:
    """U1：长期档一致为正且够大，**并且**通过既有的"能不能进一线"三条件。

    **升级不放宽**：本机制只负责"能不能回一线"，不负责"能不能进一线"。
    这一条防止「退居二线/恢复」变成绕过前置注册（prereg_ok）的后门。
    """
    blk = scales.get("all") if isinstance(scales.get("all"), dict) else {}
    delta = _num(blk.get("delta"))
    p = _num(blk.get("p_raw"))
    mde = _num(blk.get("mde_abs"))
    if delta is None or p is None or mde is None:
        return False
    strong = bool(delta > 0 and p < alpha and abs(delta) >= mde)
    return bool(strong and prereq.get("ok") is True)


def _criterion_reason(code: str, evidence: Dict) -> str:
    blk = evidence.get("scales_all") or {}
    if code == "D1":
        return (f"D1 长期窗口：Δ={blk.get('delta')}（p={blk.get('p_raw')}，"
                f"该档检出下限 {blk.get('mde_abs')}）—— 一致为负且幅度超过下限")
    if code == "D2":
        tail = evidence.get("eras_tail") or []
        bits = "；".join(f"{e.get('era')} Δ={e.get('delta')}" for e in tail if e)
        return f"D2 连续两个阶段为负：{bits}（幅度均达该段检出下限的一半以上）"
    if code == "D3":
        rc = evidence.get("recent") or {}
        return (f"D3 近窗加权：Δ={rc.get('delta')}，校正后 p={rc.get('p_adjusted')} "
                f"< 0.05 且方向为负")
    if code == "U1":
        return (f"U1 长期窗口：Δ={blk.get('delta')}（p={blk.get('p_raw')}）一致为正且超过 "
                f"检出下限，且已通过前置注册 + 验证 + 来源检查 → 可以回到一线")
    return CRITERION_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# ② 生命周期层：状态机（纯函数）
# ---------------------------------------------------------------------------

def _norm_state(state: Any) -> str:
    s = str(state or "").strip().lower()
    return s if s in STATES else "unmanaged"


def step_lifecycle(decision: Dict, state_rec: Optional[Dict],
                   thresholds: Optional[Dict] = None,
                   issue: Optional[str] = None) -> Dict:
    """**纯函数**：把一次判定推进状态机，返回新的状态记录（不落盘）。

    滞后（hysteresis）规则
    --------------------
    * 同向连续 ``K`` 次才迁移（demote 与 upgrade 对称），任一次反向或 hold 即清零；
    * 迁移后 ``MIN_DWELL`` 个周期内不再评估（dwell 未满 → 直接 hold）；
    * 已处于目标态时不再重复迁移（不会每期刷 ``since``）。

    ``state_rec`` 缺省（``None``）= 首次管理：继承 ``current_state``（通常由快照的
    ``usable`` + ``tier`` 推出），并清空 pending。**首次不迁移**——第一次观察只建立
    基线，这样"上线当天就把 17 条规律判退"这种事故不可能发生。
    """
    th = dict(thresholds or {})
    k_demote = max(1, int(th.get("k_demote", getattr(config, "LIFECYCLE_K_DEMOTE", 2))))
    k_upgrade = max(1, int(th.get("k_upgrade", getattr(config, "LIFECYCLE_K_UPGRADE", 2))))
    min_dwell = max(0, int(th.get("min_dwell", getattr(config, "LIFECYCLE_MIN_DWELL", 3))))

    decision = decision if isinstance(decision, dict) else {}
    direction = str(decision.get("direction") or "hold")
    if direction not in DIRECTIONS:
        direction = "hold"
    rec = _blank_rec(state_rec, issue)
    current = _norm_state(rec.get("state"))
    reasons: List[str] = []
    migrated = False

    # 退役态永不由自动判据改动（只能人工恢复）
    if current == "retired":
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "frozen"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False,
                       ["状态为 retired（人工退役），自动判据不改动"])

    pending_dir = _norm_pending((rec.get("pending") or {}).get("direction"))
    pending_n = int((rec.get("pending") or {}).get("streak") or 0)
    dwell = int(rec.get("dwell") or 0)

    # 首次建立基线：只记录，不迁移
    if state_rec is None:
        rec["dwell"] = 0
        rec["pending"] = {"direction": None, "streak": 0}
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "baseline"}
        rec["eval_count"] = 1
        return _finish(rec, current, False,
                       [f"首次纳入生命周期管理，建立基线（当前 {current}）；本次不迁移"])

    if dwell < min_dwell:
        # 冷却期**必须递增** dwell：否则计数永远停在 0，冷却永不到期（冻结 bug）。
        rec["dwell"] = dwell + 1
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "dwell"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False,
                       [f"刚迁移 {dwell} 个周期（< 冷却 {min_dwell} 期），本次不评估"])

    if direction == "hold":
        rec["pending"] = {"direction": None, "streak": 0}
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "hold"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False, ["本次判定为 hold，滞后计数清零"])

    # 目标态与当前态相同 → 不需要迁移（也不累计 streak，避免"待迁移"永远挂着）
    if direction == "demote" and current == "monitor":
        rec["pending"] = {"direction": None, "streak": 0}
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "already"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False, ["已处于 monitor（二线），无需再次退居"])
    if direction == "upgrade" and current == "active":
        rec["pending"] = {"direction": None, "streak": 0}
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "already"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False, ["已处于 active（一线），无需再次升级"])

    need = k_demote if direction == "demote" else k_upgrade
    streak = (pending_n + 1) if pending_dir == direction else 1
    target = "monitor" if direction == "demote" else "active"
    if target not in AUTO_TARGETS:                       # 防御性：绝不写 retired/rejected
        target = "monitor"

    if streak < need:
        rec["pending"] = {"direction": direction, "streak": streak}
        rec["last_eval"] = {"issue": issue, "direction": direction, "action": "pending"}
        rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
        return _finish(rec, current, False,
                       [f"{direction} 证据第 {streak}/{need} 次成立，尚未达到切换门槛"])

    # 迁移
    rec["prev_state"] = current
    rec["state"] = target
    rec["since"] = issue
    rec["dwell"] = 0
    rec["pending"] = {"direction": None, "streak": 0}
    rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
    hist = list(rec.get("history") or [])
    hist.append({"issue": issue, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "from": current, "to": target, "direction": direction,
                 "criteria": list(decision.get("criteria") or []),
                 "streak": streak})
    rec["history"] = hist[-20:]
    rec["last_eval"] = {"issue": issue, "direction": direction, "action": "migrated"}
    migrated = True
    if target == "monitor":
        reasons.append(f"连续 {streak} 次判定衰减 → 退居二线（monitor）："
                       f"不再参与出号，仍留在列表里继续观察（不是被证伪）")
    else:
        reasons.append(f"连续 {streak} 次判定恢复 → 回到一线（active）")
    for r in decision.get("reasons") or []:
        reasons.append(str(r))
    return _finish(rec, target, migrated, reasons)


def _blank_rec(state_rec: Optional[Dict], issue: Optional[str]) -> Dict:
    src = dict(state_rec or {})
    return {
        "state": _norm_state(src.get("state") if state_rec is not None else
                             getattr(config, "LIFECYCLE_DEFAULT_STATE", "unmanaged")),
        "since": src.get("since"),
        "prev_state": src.get("prev_state"),
        "pending": dict(src.get("pending") or {"direction": None, "streak": 0}),
        "dwell": int(src.get("dwell") or 0),
        "eval_count": int(src.get("eval_count") or 0),
        "history": list(src.get("history") or []),
        "last_eval": src.get("last_eval"),
        "first_seen": src.get("first_seen") or issue,
    }


def _norm_pending(v: Any) -> Optional[str]:
    s = str(v or "").strip().lower()
    return s if s in ("demote", "upgrade") else None


def _finish(rec: Dict, state: str, migrated: bool, reasons: Sequence[str]) -> Dict:
    out = dict(rec)
    out["state"] = state
    out["migrated"] = bool(migrated)
    out["reasons"] = [str(r) for r in reasons if r]
    return out


# ---------------------------------------------------------------------------
# ③ 把状态翻译到快照（这是"退居二线"真正生效的地方）
# ---------------------------------------------------------------------------

def infer_current_state(rule: Dict) -> str:
    """从快照条目推断「这条规律现在算不算在一线」。

    ``active`` 当且仅当 ``usable is True``；其余（monitor/rejected/扫描线索）
    一律 ``monitor`` —— 注意这里**只是起点**，不是最终态：被判 rejected 的规律
    不会被本模块"升"回 active（升级还要过 U1 的前置条件）。
    """
    if not isinstance(rule, dict):
        return "monitor"
    return "active" if rule.get("usable") else "monitor"


def apply_states(snapshot: Optional[Dict], states: Optional[Dict],
                 path_meta: Optional[Dict] = None) -> Dict:
    """把生命周期状态注入快照：**只做降级，不做放行**。

    * ``state == "monitor"`` 且快照里 ``usable`` 为 True → 置 ``usable=False``、
      ``usable_kind=None``、``lifecycle_suppressed=True``，并保留原 ``tier``
      （tier 表达"证据强度"，state 表达"是否在用"，两者不能混）。
    * 其它情况一律原样返回 —— **本函数永远不会把 usable 从 False 改成 True**，
      因此它不可能成为绕过前置注册的后门。

    ``snapshot`` 为 None 或 ``states`` 为空时原样返回（失败即放行，纪律 2）。
    """
    if not isinstance(snapshot, dict):
        return snapshot or {}
    if not states:
        return snapshot
    # 「无匹配」直通：states 非空但没有命中任何 rule -> 完全原样返回。
    # （t8：apply_states(snap, {"unknown": ...}) == snap 必须成立。）
    rule_ids = {str(r.get("id")) for r in (snapshot.get("rules") or []) if isinstance(r, dict)}
    if not (rule_ids & set(states.keys())):
        return snapshot
    out = dict(snapshot)
    rules: List[Dict] = []
    n_active = n_demoted = 0
    for rule in (snapshot.get("rules") or []):
        r = dict(rule) if isinstance(rule, dict) else rule
        if not isinstance(r, dict):
            rules.append(r)
            continue
        st = states.get(str(r.get("id")) or "")
        state = _norm_state((st or {}).get("state")) if isinstance(st, dict) else "unmanaged"
        # 没有对应状态条目（未纳管）：**原样保留**，不注入任何 lifecycle 字段。
        # 否则快照多出 lifecycle_state:null，前端/测试的"直通"语义被破坏
        # （t8 回归：states 非空但无匹配 rule 时，out == snapshot 必须成立）。
        if state == "unmanaged":
            rules.append(r)
            continue
        r["lifecycle_state"] = state
        if state == "active":
            n_active += 1
        if state == "monitor" and r.get("usable"):
            r["lifecycle_suppressed"] = True
            r["lifecycle_note_zh"] = (
                "已退居二线（monitor）：衰减判据连续成立，本期起不再参与出号；"
                "仍留在列表里继续观察 —— 这不是被证伪")
            r["tier_before_lifecycle"] = r.get("tier")
            r["usable"] = False
            r["usable_kind"] = None
            n_demoted += 1
        rules.append(r)
    out["rules"] = rules
    counts = dict(out.get("counts") or {})
    if n_demoted:
        # 重新计数：必须与 rules 里真正的 usable 一致。
        # V12-04（裁决 R-2）：**不再跨层相加** —— ``usable`` 是兼容字段，
        # 语义收紧为「长期可用条数」；跨层合计只放在 ``all_tiers_sum``。
        counts["usable_long_term"] = sum(1 for r in rules if isinstance(r, dict)
                                         and r.get("usable_kind") == "long_term")
        counts["usable_phase"] = sum(1 for r in rules if isinstance(r, dict)
                                     and r.get("usable_kind") == "phase")
        counts["usable"] = counts["usable_long_term"]
        counts["all_tiers_sum"] = counts["usable_long_term"] + counts["usable_phase"]
        if isinstance(counts.get("by_layer"), dict):
            counts["by_layer"] = dict(counts["by_layer"])
            counts["by_layer"]["usable_long_term"] = counts["usable_long_term"]
            counts["by_layer"]["usable_phase"] = counts["usable_phase"]
    counts["lifecycle_active"] = n_active
    counts["lifecycle_demoted"] = n_demoted
    out["counts"] = counts
    out["lifecycle"] = {
        "states_file": str(STATE_FILE),
        "applied": True,
        "n_states": len(states),
        "n_demoted_now": n_demoted,
        **(path_meta or {}),
    }
    return out


# ---------------------------------------------------------------------------
# 指标构造（读真实数据；与纯函数分开，便于测试注入）
# ---------------------------------------------------------------------------

def build_metrics(draws: Sequence[Dict], dim: Dict,
                  upgrade_prereq: Optional[Dict] = None,
                  scales: Optional[Sequence[str]] = None) -> Dict:
    """对单条规律算结构化指标：多尺度 5 档 + 逐阶段 + 近窗加权。

    全部走既有确定性代码（``multiscale.scale_scan`` / ``discovery.regime_of_filter``），
    无随机、无未来信息（两者都是 as-of）。任何子步骤失败都只让对应字段缺失，
    ``decide_rule_state`` 会把缺字段的判据视为不成立。
    """
    from . import discovery, multiscale

    key = str((dim or {}).get("key") or "")
    params = dict((dim or {}).get("params") or {})
    out: Dict[str, Any] = {"dim": {"key": key, "params": params},
                           "scales": {}, "eras": [], "recent": {},
                           "upgrade_prereq": dict(upgrade_prereq or {}),
                           "errors": []}
    if not key:
        out["errors"].append("维度缺失")
        return out
    try:
        scan = multiscale.scale_scan(draws, key, params, scales=scales)
        out["scales"] = dict(scan.get("scales") or {})
        out["recommendation"] = scan.get("recommendation")
        if scan.get("error"):
            out["errors"].append(str(scan["error"]))
    except Exception as exc:                             # noqa: BLE001
        out["errors"].append(f"scale_scan 失败: {type(exc).__name__}: {exc}")
    try:
        flt = discovery.make_filter(key, params)
        rep = discovery.regime_of_filter(draws, flt, start=200)
        eras = []
        for e in (rep.get("eras") or []):
            eras.append({"era": e.get("era"), "n": e.get("n"),
                         "delta": e.get("delta"), "mde": e.get("mde"),
                         "p_value": e.get("p_value"), "p_adjusted": e.get("p_adjusted")})
        out["eras"] = eras
        recent = rep.get("recent") or {}
        out["recent"] = {k: recent.get(k) for k in
                         ("n", "n_eff", "delta", "p_value", "p_adjusted", "mde")}
        out["regime_status"] = rep.get("status")
    except Exception as exc:                             # noqa: BLE001
        out["errors"].append(f"regime_of_filter 失败: {type(exc).__name__}: {exc}")
    return out


def upgrade_prereq_from_rule(rule: Dict) -> Dict:
    """从快照条目取"能不能进一线"的三条件（与 patterns_runtime 同口径）。

    任何一项缺失都判 False（安全默认）。这是 U1 的门槛，**不是**本模块新造的规则：
    直接复用 ``patterns_runtime.build_snapshot`` 里已经算好的 ``usable`` 语义。
    """
    r = rule if isinstance(rule, dict) else {}
    src = str(r.get("source") or "stat")
    prereg = (r.get("prereg") or {}).get("status")
    hyp_status = r.get("hyp_status")
    # usable 为 True 说明三条件已全部满足；usable_phase 也算（阶段档打开时）
    already = bool(r.get("usable"))
    ok = bool(already and src != "llm")
    return {"ok": ok, "source": src, "prereg_status": prereg,
            "hyp_status": hyp_status,
            "note_zh": ("已满足进入一线的前置条件" if ok else
                        "未满足进入一线的前置条件（阶段证据/验证/前置注册/来源），"
                        "因此不能由本机制升级")}




# ---------------------------------------------------------------------------
# 状态文件（IO）—— 损坏即降级为空状态 + 告警，绝不抛异常
# ---------------------------------------------------------------------------

def load_states(path=None) -> Dict:
    """读状态文件。**任何异常都返回空状态**（纪律 2：失败即放行）。

    返回 ``{"version", "updated_at", "base_issue", "thresholds", "rules": {...},
    "corrupt": bool, "warn": str}``。
    """
    p = path or STATE_FILE
    empty = {"version": STATE_VERSION, "updated_at": None, "base_issue": None,
             "thresholds": _thresholds(), "rules": {}, "corrupt": False, "warn": ""}
    try:
        raw = open(p, "r", encoding="utf-8").read()
    except FileNotFoundError:
        return empty
    except OSError as exc:
        return {**empty, "corrupt": True, "warn": f"状态文件读取失败：{exc}（按空状态处理）"}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return {**empty, "corrupt": True, "warn": f"状态文件损坏：{exc}（按空状态处理）"}
    if not isinstance(data, dict) or not isinstance(data.get("rules"), dict):
        return {**empty, "corrupt": True, "warn": "状态文件结构非法（rules 不是对象）"}
    if str(data.get("version") or "") != STATE_VERSION:
        return {**empty, "corrupt": True,
                "warn": f"状态文件版本 {data.get('version')!r} != {STATE_VERSION}（按空状态处理）"}
    out = {**empty, **data}
    out["corrupt"] = False
    return out


def save_states(states: Dict, path=None) -> Dict:
    """原子写（tmp + os.replace）。失败只返回 ``ok=False``，不抛异常。"""
    p = path or STATE_FILE
    data = dict(states or {})
    data["version"] = STATE_VERSION
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        import os as _os
        _os.replace(str(tmp), str(p))
    except OSError as exc:
        print(f"[lifecycle] 状态文件写入失败（{p}）：{exc}")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "path": str(p), "n_rules": len(data.get("rules") or {})}


def _thresholds() -> Dict:
    return {
        "k_demote": int(getattr(config, "LIFECYCLE_K_DEMOTE", 2)),
        "k_upgrade": int(getattr(config, "LIFECYCLE_K_UPGRADE", 2)),
        "min_dwell": int(getattr(config, "LIFECYCLE_MIN_DWELL", 3)),
        "alpha": 0.05,
        "mde_frac": 0.5,
    }


# ---------------------------------------------------------------------------
# AI 层：输出 schema 校验 + 技能度量（全部纯函数）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是彩票规律库的生命周期复核员。你的任务**不是**发现新规律，也**不是**预测开奖，"
    "而是：对我给出的每一条规律的**已算好的统计指标**做一次复核，判断它是否在衰减。\n"
    "硬性规则：\n"
    "1. 只能引用我给出的字段，不得引入任何未给出的数字；引用时 field 必须是我给出的路径"
    "（例如 scales.all.delta / recent.p_adjusted / eras.P23.delta）。\n"
    "2. 短窗口的『显著』不可信：每个档位都给了检出下限 mde_abs，若 |delta| < mde_abs，"
    "则该档的显著性属于样本波动，**不得**作为衰减证据。\n"
    "3. 若三条判据（长期窗口一致为负、连续两个阶段为负、近窗校正后显著为负）都不成立，"
    "就应判 keep —— 不要为了给出结论而编造。\n"
    "4. 输出严格 JSON，不要任何 JSON 之外的文字。"
)


def build_llm_payload(entries: Sequence[Dict]) -> Dict:
    """构造喂给 LLM 的**最小**结构化输入（只含能核对的数字）。"""
    rows = []
    for e in entries or []:
        m = (e or {}).get("metrics") or {}
        rows.append({
            "rule_id": (e or {}).get("rule_id"),
            "statement": str((e or {}).get("statement") or "")[:120],
            "dim": m.get("dim"),
            "lifecycle_state": (e or {}).get("state"),
            "tier_now": (e or {}).get("tier"),
            "scales": m.get("scales"),
            "eras_tail": (m.get("eras") or [])[-3:],
            "recent": m.get("recent"),
            "gold_direction": (e or {}).get("gold_direction"),
        })
    return {"rules": rows}


def build_llm_prompt(payload: Dict) -> str:
    body = json.dumps(payload, ensure_ascii=False)[:16000]
    return (
        f"## 规律指标（全部为已算好的 as-of 统计事实）\n```json\n{body}\n```\n\n"
        "请对**每一条**规律给出复核意见，输出：\n"
        '{"judgements":[{"rule_id":"...","verdict":"keep|demote|upgrade",'
        '"confidence":0.0-1.0,"reason":"一句话（<=500字），必须引用具体数字",'
        '"evidence":[{"field":"scales.all.delta","value":-0.0073}],'
        '"watch_next":["eras.P24"]}]}\n'
        "说明：verdict 只是你的复核意见，不会直接改变系统状态；系统会用确定性判据核对它。"
    )


def validate_judgements(raw: Any, payload: Dict) -> Dict:
    """把 LLM 输出逐条核对（V1–V6）。**任何一条失败只作废该条**，不抛异常。

    返回 ``{"judgements": [...已附带 valid/errors/gold 的列表...], "n_valid", "n_invalid"}``。
    """
    payload = payload if isinstance(payload, dict) else {}
    by_id = {str(r.get("rule_id")): r for r in (payload.get("rules") or [])
             if isinstance(r, dict) and r.get("rule_id") is not None}
    items = raw.get("judgements") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        items = []
    out: List[Dict] = []
    seen = set()
    for item in items[:200]:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("rule_id") or "")
        errors: List[str] = []
        gold_dir = "hold"
        src = by_id.get(rid)
        if rid in seen:
            errors.append("duplicate_rule")
        seen.add(rid)
        if src is None:
            errors.append("unknown_rule")
        else:
            gold_dir = str(src.get("gold_direction") or "hold")
        verdict = str(item.get("verdict") or "").lower()
        if verdict not in VERDICTS:
            errors.append("bad_verdict")
        conf = _num(item.get("confidence"))
        if conf is None or not (0.0 <= conf <= 1.0):
            errors.append("confidence_out_of_range")
        ev = item.get("evidence")
        if not isinstance(ev, list) or not ev:
            errors.append("evidence_missing")
        else:
            for e in ev[:6]:
                if not isinstance(e, dict):
                    errors.append("evidence_bad_item")
                    break
                f = str(e.get("field") or "")
                got = _num(e.get("value"))
                want = _lookup(src, f) if src is not None else None
                if want is None:
                    errors.append("evidence_unknown_field")
                    break
                if got is None or abs(got - float(want)) > 1e-9 + 1e-6 * abs(float(want)):
                    errors.append("evidence_value_mismatch")
                    break
        reason = str(item.get("reason") or "").strip()
        if not reason or len(reason) > 500:
            errors.append("reason_missing")
        if not errors and verdict in VERDICTS:
            # V6 方向一致（keep 永远允许；动作必须与金标准同向）
            if gold_dir == "demote" and verdict == "upgrade":
                errors.append("verdict_contradicts_gold")
            elif gold_dir == "upgrade" and verdict == "demote":
                errors.append("verdict_contradicts_gold")
            elif gold_dir == "hold" and verdict != "keep":
                errors.append("verdict_contradicts_gold")
        out.append({"rule_id": rid, "verdict": verdict,
                    "confidence": conf, "reason": reason,
                    "evidence": ev if isinstance(ev, list) else [],
                    "watch_next": [str(x) for x in (item.get("watch_next") or [])][:4],
                    "gold_direction": gold_dir,
                    "valid": not errors, "errors": errors})
    return {"judgements": out,
            "n_valid": sum(1 for j in out if j["valid"]),
            "n_invalid": sum(1 for j in out if not j["valid"]),
            "n_rules": len(by_id)}


_LOOKUP_ERA_PREFIX = "eras."


def _lookup(src: Optional[Dict], field: str) -> Optional[float]:
    """按 ``a.b.c`` 路径在 payload 行里取值；支持 ``eras.<SEG>.delta``。

    返回 None 表示"字段不存在"（与"值为 None"等价处理——都不允许被引用）。
    """
    if not src or not field:
        return None
    # eras.P23.delta 这类：在 eras_tail 里按 era 名匹配
    if field.startswith(_LOOKUP_ERA_PREFIX):
        rest = field[len(_LOOKUP_ERA_PREFIX):]
        seg, _, key = rest.rpartition(".")
        for e in (src.get("eras_tail") or []):
            if isinstance(e, dict) and str(e.get("era")) == seg:
                return _num(e.get(key))
        return None
    cur: Any = src
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return _num(cur)


def skill_stats(judgements: Sequence[Dict],
                baseline: Optional[float] = None) -> Dict:
    """AI 复核技能的度量：与金标准一致率 + Wilson 下界 + 分组混淆。

    **样本不足不下结论**：``n_action < 10`` 时只报描述统计。

    基线（V12-13）：默认 ``baseline=None`` 时用**哑判官基线**——「全答金标准中最常见
    那一类」所能取得的准确率（= 最常见金标准的占比）。为什么必须换掉原来的 ``0.5``：
    一个闭眼全答 ``keep`` 的假判官，在 ``rule_lifecycle`` 上实测准确率 **99.19%**
    （金标准几乎全是 ``hold``）；用 ``0.5`` 当基线，等于把「不比抛硬币差」当成达标，
    而这个门槛哑判官闭眼就能过。显式传 ``baseline`` 仍走旧语义（向后兼容，返回体
    ``baseline_kind="fixed"``）。
    """
    rows = [j for j in (judgements or []) if isinstance(j, dict)]
    valid = [j for j in rows if j.get("valid")]
    n = len(valid)
    agree = sum(1 for j in valid if _agrees(j))
    n_action = sum(1 for j in valid if j.get("gold_direction") in ("demote", "upgrade"))
    p = (agree / n) if n else None
    lo, hi = wilson_interval(agree, n) if n else (None, None)
    by_gold: Dict[str, Dict[str, int]] = {}
    for j in valid:
        g = str(j.get("gold_direction") or "hold")
        v = str(j.get("verdict") or "?")
        slot = by_gold.setdefault(g, {})
        slot[v] = int(slot.get(v, 0)) + 1
    if baseline is None:
        baseline_kind = "dumb_judge"
        baseline = _dumb_judge_share(valid)
    else:
        baseline_kind = "fixed"
        baseline = float(baseline)
    concl = "insufficient"
    if n >= 10 and n_action >= 10:
        concl = "below-baseline" if (lo is not None and lo <= baseline) else "ok"
    return {
        "n": n, "n_invalid": sum(1 for j in rows if not j.get("valid")),
        "n_action": n_action, "agree": agree, "agreement": p,
        "wilson_lo": lo, "wilson_hi": hi, "baseline": float(baseline),
        "baseline_kind": baseline_kind,
        "by_gold": by_gold, "conclusion": concl,
        "note_zh": _skill_note(n, n_action, p, lo, concl, baseline, baseline_kind),
    }


def _dumb_judge_share(valid: Sequence[Dict]) -> float:
    """哑判官基线 = 最常见金标准类别的占比（全答那一类能拿到的准确率上限）。"""
    if not valid:
        return 0.5
    counts: Dict[str, int] = {}
    for j in valid:
        g = str(j.get("gold_direction") or "hold")
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) / float(len(valid))


def _agrees(j: Dict) -> bool:
    gold = str(j.get("gold_direction") or "hold")
    v = str(j.get("verdict") or "")
    if gold == "hold":
        return v == "keep"
    return v == gold


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson 95% 区间。

    **V40 薄壳**：公式唯一真源是 `stats.rigor.wilson_interval`
    （审计 P2-1：四份并行实现 → 一份真源）。签名、返回值、
    边界语义（`n<=0 → (None, None)`）全部不变；
    `tests/test_v11_rule_lifecycle.py::test_t10_wilson_matches_reference`
    的参考值 `(0,10)→hi≈0.2775`、`(5,10)→0.2366/0.7634` 继续钉住。
    """
    from .stats.rigor import wilson_interval as _truth
    return _truth(k, n, z)


def _skill_note(n: int, n_action: int, p: Optional[float],
                lo: Optional[float], concl: str,
                baseline: float = 0.5, baseline_kind: str = "fixed") -> str:
    if n == 0:
        return "尚无 AI 复核记录"
    base_txt = ("哑判官基线" if baseline_kind == "dumb_judge" else "基线")
    if concl == "insufficient":
        return (f"已复核 {n} 条（其中金标准为动作的 {n_action} 条）。"
                f"动作样本 < 10，本报告只作描述、不下『AI 是否跟得上』的结论 —— "
                f"在当前数据下判据多为 hold，这是预期结果。")
    if concl == "below-baseline":
        return (f"一致率 {p:.1%}（Wilson 下界 {lo:.1%}）不优于{base_txt} {baseline:.1%}，"
                f"AI 复核未显示出增量价值 → 建议自动降级为『只展示』。")
    return (f"一致率 {p:.1%}（Wilson 下界 {lo:.1%}）优于{base_txt} {baseline:.1%}")


# ---------------------------------------------------------------------------
# 一次完整周期（IO 编排）
# ---------------------------------------------------------------------------

def run_cycle(draws: Sequence[Dict], snapshot: Optional[Dict] = None,
              rules: Optional[Sequence[Dict]] = None,
              path=None, issue: Optional[str] = None,
              use_llm: bool = True, persist: bool = True,
              max_rules: Optional[int] = None) -> Dict:
    """跑一轮生命周期判定并落盘。

    **顺序**：读状态 → 逐条算指标 → 纯函数判定 → 推进状态机 → 写状态 → （可选）AI 复核。
    LLM 在**最后**调用，且它的输出**不影响** states（纪律 1），只写入
    ``data/rule_lifecycle.json`` 的 ``judgements`` 段与技能统计。

    ``rules`` 缺省取快照的 ``rules[]``；``snapshot`` 缺省从 ``patterns_runtime`` 读，
    读不到就用 ``rules`` 参数（测试用）。
    """
    from . import patterns_runtime

    draws = list(draws or [])
    issue = str(issue or ((draws[-1] or {}).get("issue") if draws else "") or "")
    if snapshot is None:
        snapshot = patterns_runtime.load_snapshot()
    src_rules = list(rules if rules is not None else
                     ((snapshot or {}).get("rules") or []))
    max_n = int(max_rules or getattr(config, "LIFECYCLE_MAX_RULES", 60))

    states = load_states(path)
    th = states.get("thresholds") or _thresholds()
    recs: Dict[str, Dict] = dict(states.get("rules") or {})
    results: List[Dict] = []
    migrated: List[Dict] = []
    errors: List[str] = []
    if states.get("corrupt"):
        errors.append(states.get("warn") or "状态文件损坏，按空状态处理")

    # 优先级：usable 的（真正在一线、值得盯）排前面，其次按 tier
    def _rank(r: Dict) -> tuple:
        return (0 if r.get("usable") else 1,
                0 if r.get("tier") != "monitor" else 1,
                str(r.get("id") or ""))
    ordered = sorted([r for r in src_rules if isinstance(r, dict) and r.get("id")],
                     key=_rank)[:max_n]

    for rule in ordered:
        rid = str(rule.get("id"))
        rec = recs.get(rid)
        if rec is None:
            # 首次纳入：继承"现在算不算在一线"，不迁移
            base_state = infer_current_state(rule)
            rec = {"state": base_state, "since": issue, "first_seen": issue,
                   "pending": {"direction": None, "streak": 0}, "dwell": 0,
                   "eval_count": 0, "history": []}
        try:
            metrics = build_metrics(draws, rule.get("dim") or {},
                                    upgrade_prereq=upgrade_prereq_from_rule(rule))
        except Exception as exc:                          # noqa: BLE001
            metrics = {"errors": [f"{type(exc).__name__}: {exc}"]}
            errors.append(f"{rid}: 指标计算失败 {exc}")
        decision = decide_rule_state(metrics, current_state=rec.get("state"),
                                     streak=int((rec.get("pending") or {}).get("streak") or 0),
                                     dwell=int(rec.get("dwell") or 0), thresholds=th)
        new_rec = step_lifecycle(decision, rec if recs.get(rid) is not None else None,
                                 thresholds=th, issue=issue)
        recs[rid] = new_rec
        row = {"rule_id": rid, "statement": rule.get("statement"),
               "dim": rule.get("dim"), "tier": rule.get("tier"),
               "usable": bool(rule.get("usable")),
               "state_before": rec.get("state"), "state_after": new_rec["state"],
               "migrated": bool(new_rec.get("migrated")),
               "direction": decision.get("direction"),
               "criteria": decision.get("criteria") or [],
               "reasons": new_rec.get("reasons") or [],
               "pending": new_rec.get("pending"),
               "metrics": metrics}
        results.append(row)
        if new_rec.get("migrated"):
            migrated.append({"rule_id": rid, "from": new_rec.get("prev_state"),
                             "to": new_rec["state"], "criteria": decision.get("criteria") or [],
                             "reasons": new_rec.get("reasons") or []})

    # 清理已不在快照里的陈旧记录（保留 30 天，避免每次重算都清零）
    live = {str(r.get("id")) for r in ordered}
    for rid in list(recs):
        if rid not in live:
            hist = recs[rid].get("history") or []
            if not hist:
                recs.pop(rid, None)

    states_out = {"rules": recs, "thresholds": th, "base_issue": issue,
                  "last_run": time.strftime("%Y-%m-%d %H:%M:%S")}

    # AI 层（最后，且绝不影响 states）
    llm_block = _llm_review(results, states_out, draws, use_llm=use_llm)

    states_out["judgements"] = llm_block.get("judgements") or []
    states_out["skill"] = llm_block.get("skill") or {}
    states_out["llm_note"] = llm_block.get("note_zh")
    saved = save_states(states_out, path=path) if persist else {"ok": True, "skipped": True}

    n_action = sum(1 for r in results if (r.get("direction") or "hold") != "hold")
    summary = {
        "ok": bool(saved.get("ok", True)),
        "issue": issue,
        "n_rules": len(results),
        "n_action": n_action,
        "n_migrated": len(migrated),
        "migrated": migrated,
        "n_baseline": sum(1 for r in results
                          if (recs.get(str(r["rule_id"])) or {}).get("eval_count") == 1),
        "counts": _state_counts(recs),
        "errors": errors[:10],
        "llm": {k: llm_block.get(k) for k in ("ok", "used", "error", "note_zh")},
        "saved": saved,
        "rules": results,
        "note_zh": _cycle_note(results, migrated, llm_block),
    }
    return summary


def _state_counts(recs: Dict) -> Dict:
    out = {"active": 0, "monitor": 0, "retired": 0, "unmanaged": 0}
    for rec in (recs or {}).values():
        st = _norm_state((rec or {}).get("state"))
        out[st] = int(out.get(st, 0)) + 1
    return out


def _cycle_note(results: Sequence[Dict], migrated: Sequence[Dict], llm: Dict) -> str:
    if not results:
        return "规律快照为空：没有可评估的规律。"
    if migrated:
        return (f"本期有 {len(migrated)} 条规律变更生命周期状态；"
                f"其中退居二线为 monitor（仍继续观察，不属于被证伪）。")
    if not any(r.get("usable") for r in results):
        return ("本期**没有**规律在一线（usable=0），因此也**没有对象**可以退居二线 —— "
                "这与『本期无需退居二线』是两回事，请勿认为功能未生效。")
    return ("本期**无需**退居二线：各规律的三条衰减判据均未连续成立，"
            "维持现状（正常状态）。")


# ---------------------------------------------------------------------------
# AI 复核（可选层；任何失败都只影响解释，不影响状态）
# ---------------------------------------------------------------------------

DEF_REVIEW = {"ok": True, "used": False, "error": None,
              "judgements": [], "skill": {}, "note_zh": "AI 复核未启用"}


def _llm_review(results: Sequence[Dict], states_out: Dict,
                draws: Sequence[Dict], use_llm: bool) -> Dict:
    """调用 LLM 做复核；**绝不影响 states**（本函数只读 results）。

    关停遵循既有 ``decision_disabled`` 机制（``methods.decision_state``），
    purpose 用 ``"lifecycle"``，归入 ``llm_client.DECISION_PURPOSES``。
    """
    if not use_llm:
        return {**DEF_REVIEW, "note_zh": "AI 复核未启用（调用方关闭）"}
    try:
        from . import llm_client, methods
        if getattr(config, "LLM_DISABLED", False):
            return {**DEF_REVIEW, "note_zh": "AI 未参与：LLM 已被停用。"}
        if bool(methods.decision_state().get("disabled")):
            return {**DEF_REVIEW, "note_zh": "AI 未参与：决策通道已关停（decision_disabled）。"}
        if not config.llm_configured():
            return {**DEF_REVIEW, "note_zh": "AI 未参与：LLM 通道未配置。"}
        entries = []
        for r in results:
            rid = str(r.get("rule_id"))
            entries.append({"rule_id": rid, "statement": r.get("statement"),
                            "metrics": r.get("metrics"), "state": r.get("state_after"),
                            "tier": r.get("tier"), "gold_direction": r.get("direction")})
        payload = build_llm_payload(entries)
        raw = llm_client.chat_json(
            SYSTEM_PROMPT, build_llm_prompt(payload), max_tokens=2000,
            temperature=0.2, purpose="lifecycle", strict=False)
        if not isinstance(raw, dict):
            return {**DEF_REVIEW, "ok": False, "error": "LLM 未返回可解析 JSON",
                    "note_zh": "AI 未参与：模型未返回可解析 JSON（状态判定不受影响）。"}
        checked = validate_judgements(raw, payload)
        skill = skill_stats(checked.get("judgements") or [])
        return {"ok": True, "used": True, "error": None,
                "judgements": checked.get("judgements") or [],
                "skill": skill,
                "note_zh": (f"AI 复核 {checked.get('n_valid', 0)}/{checked.get('n_rules', 0)} 条"
                            f"通过核对（{skill.get('note_zh')}）。"
                            f"AI 意见不改变状态，仅用于解释与技能度量。")}
    except Exception as exc:                              # noqa: BLE001
        return {**DEF_REVIEW, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "note_zh": f"AI 未参与：{type(exc).__name__}（状态判定不受影响）。"}


def report(path=None) -> Dict:
    """给 API/前端用的只读视图：当前状态 + 最近判定 + 技能。"

    **同时给出两种"没动作"的含义**（Lead 要求区分的两条文案）：
    ``no_target``  —— 没有规律在一线（usable=0），无对象可退；
    ``nothing_to_do`` —— 有在一线的规律，但本期判据均未连续成立。
    """
    st = load_states(path)
    recs = st.get("rules") or {}
    counts = _state_counts(recs)
    usable_now = 0
    try:
        from . import patterns_runtime
        snap = patterns_runtime.load_snapshot()
        usable_now = int(((snap or {}).get("counts") or {}).get("usable") or 0)
    except Exception:                                     # noqa: BLE001
        usable_now = 0
    if usable_now == 0:
        mode, note = "no_target", (
            "本期**没有**规律在一线（usable=0），因此没有对象可以退居二线。"
            "列表里现有的规律本来就处于『仅监控』——它们不是被本功能降级的。")
    else:
        mode, note = "nothing_to_do", (
            "本期**无需**退居二线：各规律的三条衰减判据均未连续成立，维持现状。")
    if st.get("corrupt"):
        note = (st.get("warn") or "状态文件异常") + "；本期按『没有任何规律被退居二线』处理。"
        mode = "degraded"
    return {
        "version": STATE_VERSION,
        "updated_at": st.get("updated_at"),
        "base_issue": st.get("base_issue"),
        "thresholds": st.get("thresholds") or _thresholds(),
        "counts": counts,
        "usable_now": usable_now,
        "mode": mode,
        "note_zh": note,
        "rules": [{"rule_id": rid, **{k: v for k, v in (rec or {}).items()
                                      if k != "history"},
                   "history": (rec or {}).get("history") or []}
                  for rid, rec in sorted(recs.items())],
        "judgements": st.get("judgements") or [],
        "skill": st.get("skill") or {},
        "llm_note": st.get("llm_note"),
        "corrupt": bool(st.get("corrupt")),
    }
