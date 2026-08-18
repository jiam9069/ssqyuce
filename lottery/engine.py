"""集成预测引擎：统计模型 + LLM 通道 → 硬约束过滤 → 置信度评分 → Top-N 输出。"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

import numpy as np

from . import backtest as BT
from . import config, db, features as F, llm_client, methods as METH, ml_model, models as M

R_MAX, B_MAX = 33, 16


# ---------- 硬约束 ----------

def _last_sums(draws: List[Dict], n: int = 500) -> np.ndarray:
    return F.sums(draws[-n:]) if len(draws) >= n else F.sums(draws)


def constraint_ctx(draws: List[Dict]) -> Dict:
    s = _last_sums(draws)
    zc = F.zone_counts(draws[-500:] if len(draws) >= 500 else draws)
    zc_hist = {}
    for z in zc:
        zc_hist[z] = zc_hist.get(z, 0) + 1
    top_zones = [k for k, _ in sorted(zc_hist.items(), key=lambda x: -x[1])[:10]]
    oc = F.odd_counts(draws[-500:] if len(draws) >= 500 else draws)
    odd_hist = {}
    for o in oc:
        odd_hist[int(o)] = odd_hist.get(int(o), 0) + 1
    odd_range = sorted(
        (k for k, v in odd_hist.items() if v / max(1, len(oc)) >= 0.02)
    )
    return {
        "sum_min": float(np.percentile(s, 5)),
        "sum_max": float(np.percentile(s, 95)),
        "top_zones": top_zones,
        "odd_range": odd_range if odd_range else [2, 3, 4],
    }


def pass_constraints(reds: List[int], blue: int, ctx: Dict) -> bool:
    s = sum(reds)
    if not (ctx["sum_min"] <= s <= ctx["sum_max"]):
        return False
    z1 = sum(1 for r in reds if 1 <= r <= 11)
    z2 = sum(1 for r in reds if 12 <= r <= 22)
    z3 = 6 - z1 - z2
    if (z1, z2, z3) not in ctx["top_zones"]:
        return False
    odd = sum(1 for r in reds if r % 2 == 1)
    if odd not in ctx["odd_range"]:
        return False
    return True


# ---------- 统计采样 ----------

def _weighted_sample(weights: np.ndarray, k: int, rng: random.Random) -> List[int]:
    pool = list(range(1, len(weights) + 1))
    w = weights.astype(float).copy()
    chosen = []
    for _ in range(k):
        if w.sum() <= 0:
            w = np.ones_like(w)
        p = (w / w.sum()).tolist()
        pick = rng.choices(pool, weights=p, k=1)[0]
        chosen.append(pick)
        w[pick - 1] = 0.0
    return sorted(chosen)


def sample_stat_ticket(red_blend: np.ndarray, blue_blend: np.ndarray,
                       ctx: Dict, rng: random.Random, uniform: bool = False) -> Optional[Dict]:
    """按混合概率加权不放回抽样，硬约束不过则重试。"""
    for _ in range(60):
        if uniform:
            reds = sorted(rng.sample(range(1, R_MAX + 1), 6))
            blue = rng.randint(1, B_MAX)
        else:
            reds = _weighted_sample(red_blend, 6, rng)
            blue = _weighted_sample(blue_blend, 1, rng)[0]
        if pass_constraints(reds, blue, ctx):
            return {"reds": reds, "blue": blue}
    return None


def ensemble_mass(red_blend: np.ndarray, blue_blend: np.ndarray,
                  reds: List[int], blue: int) -> float:
    """候选号在混合分布上的概率质量（0-100 比例尺）。"""
    r_mass = float(np.mean(red_blend[np.array(reds) - 1]))
    b_mass = float(blue_blend[blue - 1])
    norm_r = float(np.max(red_blend))
    norm_b = float(np.max(blue_blend))
    if norm_r <= 0:
        norm_r = 1.0
    if norm_b <= 0:
        norm_b = 1.0
    return 100.0 * (0.75 * r_mass / norm_r + 0.25 * b_mass / norm_b)


def _brier_blend(models_dict: Dict, draws: List[Dict]) -> tuple:
    """基于 Brier score 的模型概率融合（替代等权平均）。"""
    n_eval = min(50, len(draws) - 300)
    if n_eval < 10:
        # 数据不足，等权
        n = len(models_dict)
        red_blend = sum(m["red"] for m in models_dict.values()) / n
        blue_blend = sum(m["blue"] for m in models_dict.values()) / n
        return red_blend, blue_blend
    
    weights = {}
    for name, model in models_dict.items():
        red_p = model["red"]
        blue_p = model["blue"]
        brier_red, brier_blue, count = 0.0, 0.0, 0
        for i in range(-n_eval, 0):
            t = draws[i]
            tr = np.zeros(33)
            for r in t["reds"]: tr[r-1] = 1.0
            brier_red += float(np.mean((red_p - tr) ** 2))
            tb = np.zeros(16)
            tb[t["blue"]-1] = 1.0
            brier_blue += float(np.mean((blue_p - tb) ** 2))
            count += 1
        if count > 0:
            avg = (brier_red + brier_blue) / (2 * count)
            weights[name] = max(0.01, 1.0 / max(avg, 0.001))
        else:
            weights[name] = 0.01
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    print(f"[engine] Brier 权重: {weights}")
    
    red_blend = np.zeros(33)
    blue_blend = np.zeros(16)
    for name, model in models_dict.items():
        w = weights.get(name, 0)
        red_blend += w * model["red"]
        blue_blend += w * model["blue"]
    red_blend = red_blend / red_blend.sum() * 6
    blue_blend = blue_blend / blue_blend.sum()
    return red_blend, blue_blend


# ---------- 主流程 ----------

def build_context(draws: List[Dict], stats: Dict, patterns: List[Dict]) -> Dict:
    return {
        "stats": stats,
        "recent": stats.get("recent", []),
        "patterns": [
            {
                "key": p.get("key"), "name_zh": p.get("name_zh"), "kind": p.get("kind"),
                "grade": p.get("grade"), "margin": p.get("margin"),
                "p_value": p.get("p_value"), "p_adj": p.get("p_adj"),
                "n": p.get("sample_size"),
                "desc": (p.get("desc") or "")[:80],
                "ci": [
                    p.get("backtest", {}).get("ci_lower"),
                    p.get("backtest", {}).get("ci_upper"),
                ],
            }
            for p in patterns if p.get("grade") in ("A", "B")
        ],
        "feedback": _last_feedback(draws),
        "constraints": constraint_ctx(draws),
    }


def _last_feedback(draws: List[Dict]) -> Optional[Dict]:
    """上期预测 vs 实际开奖的命中摘要（供 LLM 回馈下一期，M3.2）。"""
    if len(draws) < 2:
        return None
    last = draws[-1]
    try:
        preds = db.load_predictions(last["issue"])
    except Exception:  # noqa: BLE001
        preds = []
    if not preds:
        return None
    try:
        from . import evaluate as EV
        res = EV.tickets_result(preds, last)
        return {
            "issue": last["issue"],
            "n_tickets": len(preds),
            "avg_red_hits": round(float(np.mean(res["red_hits"])), 2),
            "blue_hits": int(sum(res["blue_hits"])),
            "best_level": res["best_level"],
            "reward": round(res["reward"], 1),
        }
    except Exception:  # noqa: BLE001
        return None


def _parse_tickets_response(res: Optional[dict], method_name: str) -> List[Dict]:
    """校验并规范化 LLM 返回的 tickets（含 evidence / counter_evidence / structure_scores）。"""
    out: List[Dict] = []
    if not res or not isinstance(res.get("tickets"), list):
        return out
    for t in res["tickets"]:
        try:
            reds = sorted(int(x) for x in t["reds"])
            blue = int(t["blue"])
            conf = float(t.get("confidence", 50))
            if len(set(reds)) != 6 or not all(1 <= x <= R_MAX for x in reds):
                continue
            if not 1 <= blue <= B_MAX:
                continue
            out.append({
                "reds": reds, "blue": blue, "method": f"llm:{method_name}",
                "confidence": conf,
                "reasoning": str(t.get("reasoning", ""))[:300],
                "patterns_used": [str(x) for x in t.get("patterns_used", [])],
                "evidence": t.get("evidence") if isinstance(t.get("evidence"), dict) else {},
                "counter_evidence": (t.get("counter_evidence")
                                     if isinstance(t.get("counter_evidence"), list) else []),
                "structure_scores": (t.get("structure_scores")
                                     if isinstance(t.get("structure_scores"), dict) else {}),
            })
        except (TypeError, ValueError):
            continue
    return out


def _pick_verify_cfg(model_cfgs: List[Dict]) -> Optional[Dict]:
    """第三轮校验模型：优先 LOTT_LLM_VERIFY_MODEL，否则回退第一个可用模型。"""
    if config.LLM_VERIFY_MODEL:
        for c in model_cfgs:
            if c.get("model") == config.LLM_VERIFY_MODEL or c.get("name") == config.LLM_VERIFY_MODEL:
                return c
    return model_cfgs[0] if model_cfgs else None


def llm_tickets(draws: List[Dict], stats: Dict, patterns: List[Dict],
                rng: random.Random, llm_samples: Optional[int] = None,
                llm_verify: Optional[bool] = None) -> List[Dict]:
    """多模型 LLM 采样生成候选（失败自动降级为空）。

    llm_samples: 覆盖全局 LLM_SAMPLES（离线评估可传 1 以控制成本与时长）。
    llm_verify: 是否执行第三轮校验（默认跟随 config.LLM_VERIFY_ENABLED）。
    """
    from concurrent.futures import ThreadPoolExecutor
    if config.LLM_DISABLED:
        return []
    model_cfgs = config.llm_model_list()
    if config.LLM_EVAL_MODEL:
        # 评估专用模型（LOTT_LLM_EVAL_MODEL）：评估期间限定单模型，控制成本
        filtered = [c for c in model_cfgs
                    if c.get("model") == config.LLM_EVAL_MODEL or c.get("name") == config.LLM_EVAL_MODEL]
        if filtered:
            model_cfgs = filtered
    if not model_cfgs:
        print("[llm] 无可用模型配置，跳过 LLM 通道")
        return []
    ctx = build_context(draws, stats, patterns)

    # 观察轮次
    obs = llm_client.chat_json(
        llm_client.SYSTEM_BASE,
        llm_client.observations_prompt(
            llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"],
            feedback=ctx.get("feedback")),
        max_tokens=1600, temperature=0.7, model_cfg=model_cfgs[0],
    )
    if obs is None:
        print("[llm] 观察生成失败，跳过 LLM 通道")
        return []

    n_models = len(model_cfgs)
    tickets: List[Dict] = []

    def _ticket_call(cfg: Dict) -> List[Dict]:
        res = llm_client.chat_json(
            llm_client.SYSTEM_BASE,
            llm_client.tickets_prompt(
                llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"], obs,
                feedback=ctx.get("feedback")),
            max_tokens=2000, temperature=0.9, model_cfg=cfg,
        )
        return _parse_tickets_response(res, cfg["name"])

    n_samples = int(llm_samples) if llm_samples else config.LLM_SAMPLES
    calls = [model_cfgs[i % n_models] for i in range(n_samples)]
    with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as ex:
        futures = [ex.submit(_ticket_call, cfg) for cfg in calls]
        for f in futures:
            tickets.extend(f.result())

    # 第三轮校验（M3.2）：critique → 发现问题才 refine（低温 + 校验模型）
    if llm_verify is None:
        llm_verify = config.LLM_VERIFY_ENABLED
    if llm_verify and tickets:
        vcfg = _pick_verify_cfg(model_cfgs)
        if vcfg:
            try:
                critique = llm_client.chat_json(
                    llm_client.SYSTEM_BASE,
                    llm_client.critique_prompt(
                        llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"],
                        ctx.get("feedback") or {}, tickets),
                    max_tokens=600, temperature=0.2, model_cfg=vcfg)
                if critique and critique.get("verdict") == "problematic":
                    refined = llm_client.chat_json(
                        llm_client.SYSTEM_BASE,
                        llm_client.refine_prompt(
                            llm_client.compact_stats(ctx["stats"]), critique, tickets,
                            feedback=ctx.get("feedback")),
                        max_tokens=2000, temperature=0.2, model_cfg=vcfg)
                    parsed = _parse_tickets_response(refined, vcfg.get("model", "verify"))
                    if parsed:
                        tickets = parsed
                        print(f"[llm] 第三轮校验已修正选号（{len(parsed)} 注，校验模型 {vcfg.get('model')}）")
            except Exception as ex:  # noqa: BLE001
                print(f"[llm] 第三轮校验异常，保留原候选: {ex}")
    return tickets


def _ml_result_block(ml_entry: Optional[Dict], use_ml: bool,
                         extra: Optional[Dict] = None) -> Dict:
    """构造预测结果里的 ML 元数据块。"""
    extra = extra or {}
    if not use_ml or ml_entry is None:
        return {"enabled": bool(use_ml), "ready": ml_entry is not None, **extra}
    metrics = ml_model.get_ml_metrics(ml_entry["red"], ml_entry["blue"])
    return {
        "enabled": True,
        "ready": True,
        "red_avg_brier": metrics.get("red_avg_brier_cal"),
        "blue_avg_brier": metrics.get("blue_avg_brier_cal"),
        "red_ece": metrics.get("red_ece"),
        "blue_ece": metrics.get("blue_ece"),
        "trained_at": ml_entry.get("trained_at"),
        **extra,
    }


def predict_next(draws: List[Dict], use_llm: Optional[bool] = None,
                 n_tickets: Optional[int] = None, persist: bool = True,
                 use_ml: Optional[bool] = None,
                 rng: Optional[random.Random] = None,
                 llm_samples: Optional[int] = None,
                 llm_verify: Optional[bool] = None) -> Dict:
    """对下一期生成预测。

    use_ml: 是否把 M2 ML 概率模型（GBDT+RF 集成）并入 Brier 加权融合；
            默认跟随 config.ML_ENABLED。
    """
    if use_llm is None:
        use_llm = not config.LLM_DISABLED
    if use_ml is None:
        use_ml = config.ML_ENABLED
    # M4.2 方法 A/B 开关：按运行模式取生效规格（production=严格过滤 / research=全部启用）
    methods_spec = METH.effective_spec(config.METHOD_MODE, config.METHODS_SPEC)
    # 关闭的通道不生成候选（LLM 同时省 API 成本）
    use_llm = use_llm and METH.is_enabled("llm", methods_spec)
    use_ml = use_ml and METH.is_enabled("ml", methods_spec)
    n_tickets = n_tickets or config.N_TICKETS
    issue = BT.next_issue(draws[-1]["issue"])

    stats = F.compute_features(draws)
    patterns = db.load_patterns()
    ctx = constraint_ctx(draws)

    # 统计模型 + M2 ML 概率模型 → 混合概率（Brier 加权融合）
    # M4.2：仅保留开关内启用的统计基线（stat:xxx）；全关时回退均匀分布兜底
    bl = {name: m for name, m in M.build_models(draws).items()
          if METH.is_enabled(f"stat:{name}", methods_spec)}
    if not bl:
        bl = {"uniform": M.uniform_model()}
    ml_entry, ml_extra = None, {}
    if use_ml and ml_model.HAS_SKLEARN and len(draws) >= config.ML_MIN_START + 10:
        if ml_model.ml_ready(draws):
            ml_entry = ml_model.get_ml_models(draws)
            if ml_entry is not None:
                bl["ml"] = {
                    "red": ml_model.predict_red_probs(ml_entry["red"], draws),
                    "blue": ml_model.predict_blue_probs(ml_entry["blue"], draws),
                    "name": "ml",
                }
        else:
            # 尚未训练完成（如后台预热中）：本轮先不阻塞请求，仅标记状态
            ml_extra["warming_up"] = True
    red_blend, blue_blend = _brier_blend(bl, draws)

    rng = rng if rng is not None else random.Random()
    candidates: List[Dict] = []

    # 各统计模型 + M2 ML 模型分别采样
    for name, model in bl.items():
        for _ in range(2):
            t = sample_stat_ticket(model["red"], model["blue"], ctx, rng)
            if t:
                t["method"] = "ml" if name == "ml" else f"stat:{name}"
                candidates.append(t)
    # 均匀对照（M4.2 开关同样适用）
    if METH.is_enabled("uniform", methods_spec):
        for _ in range(2):
            t = sample_stat_ticket(None, None, ctx, rng, uniform=True)
            if t:
                t["method"] = "uniform"
                candidates.append(t)

    # LLM 候选（任何异常都降级为纯统计，绝不让 LLM 拖死整次预测）
    llm_cands = []  # type: ignore
    if use_llm:
        try:
            llm_cands = llm_tickets(draws, stats, patterns, rng,
                                    llm_samples=llm_samples,
                                    llm_verify=llm_verify)
        except Exception as e:  # noqa: BLE001
            print(f"[engine] LLM 通道异常，降级为纯统计: {e}")
            llm_cands = []
    candidates.extend(llm_cands)
    llm_models_used = sorted({
        t["method"].split(":", 1)[1] for t in llm_cands if t["method"].startswith("llm:")
    })
    # M4.2 兜底：最终按生效开关过滤候选（allow/deny 模式下丢弃关闭通道的票）
    candidates = METH.filter_candidates(candidates, methods_spec)

    # 去重 + 评分
    seen = set()
    scored: List[Dict] = []
    for t in candidates:
        key = (tuple(t["reds"]), t["blue"])
        if key in seen:
            continue
        seen.add(key)
        if t["method"].startswith("llm:"):
            em = ensemble_mass(red_blend, blue_blend, t["reds"], t["blue"])
            conf = round(20 + 0.45 * (0.5 * t.get("confidence", 50) + 0.5 * em), 1)
        else:
            em = ensemble_mass(red_blend, blue_blend, t["reds"], t["blue"])
            conf = round(10 + 0.25 * em, 1)
        t["confidence"] = min(100.0, max(1.0, conf))
        scored.append(t)

    # 蓝球分散
    scored.sort(key=lambda x: -x["confidence"])
    picked: List[Dict] = []
    blue_count = {}
    for t in scored:
        if len(picked) >= n_tickets:
            break
        if blue_count.get(t["blue"], 0) >= max(2, n_tickets // 5):
            continue
        blue_count[t["blue"]] = blue_count.get(t["blue"], 0) + 1
        picked.append(t)

    result = {
        "issue": issue,
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "target_draw": {"issue": draws[-1]["issue"], "date": draws[-1]["date"],
                        "reds": draws[-1]["reds"], "blue": draws[-1]["blue"]},
        "tickets": picked,
        "llm_used": use_llm and bool(llm_cands),
        "llm_models": llm_models_used,
        "red_probs": red_blend.tolist(),
        "blue_probs": blue_blend.tolist(),
        "patterns_summary": {
            "A": sum(1 for p in patterns if p["grade"] == "A"),
            "B": sum(1 for p in patterns if p["grade"] == "B"),
            "C": sum(1 for p in patterns if p["grade"] == "C"),
        },
        "ml": _ml_result_block(ml_entry, use_ml, ml_extra),
        "note": ("样本外回测未发现稳定显著的规律，预测仅基于统计结构的均衡建议；"
                 "置信度为模型结构分，不构成中奖概率。理性购彩。"),
    }
    if persist:
        db.save_features(issue, stats)
        db.save_predictions(issue, picked)
        # M4.1：保存方法、版本、模型与 LLM 配置快照，供开奖后长期对照
        db.save_eval_meta(issue, picked, result=result)
    return result
