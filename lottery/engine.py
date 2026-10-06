"""集成预测引擎：统计模型 + LLM 通道 → 硬约束过滤 → 置信度评分 → Top-N 输出。"""
from __future__ import annotations

import json
import random
import time
from typing import Dict, List, Optional

import numpy as np

from . import backtest as BT
from . import config, db, features as F, llm_client, methods as METH, ml_model, models as M
from .llm_client import LLMChannelError

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


def _brier_blend(models_dict: Dict, draws: List[Dict], adaptive: Optional[Dict] = None) -> tuple:
    """基于 Brier score 的模型概率融合（替代等权平均），叠加 U3 自适应降权。

    adaptive: {method: weight}，来自 METH.adaptive_weights()；None = 不施加。
    蓝球始终并入 M.blue_specialist()（U2），避免蓝球“裸奔”。
    """
    models_dict = dict(models_dict)
    # U2：蓝球专用模型恒在混合里（红球仅按调用方给定的模型池）
    blue_spec = {"blue_specialist": {
        "red": None, "blue": M.blue_specialist(draws), "name": "blue_specialist"}}
    n_eval = min(50, len(draws) - 300)
    # 融合用模型池：red 用 models_dict，blue 用 models_dict + blue_specialist
    red_pool = {k: m["red"] for k, m in models_dict.items() if m.get("red") is not None}
    blue_pool = {k: m["blue"] for k, m in models_dict.items() if m.get("blue") is not None}
    blue_pool["blue_specialist"] = blue_spec["blue_specialist"]["blue"]

    if n_eval < 10:
        # 数据不足，等权（blue_specialist 权重与统计模型对等）
        n = len(red_pool) or 1
        red_blend = sum(red_pool.values()) / n if red_pool else np.full(R_MAX, 1.0 / R_MAX)
        blue_blend = sum(blue_pool.values()) / len(blue_pool) if blue_pool else np.full(B_MAX, 1.0 / B_MAX)
        return _apply_adaptive_red(red_blend, adaptive, red_pool), blue_blend

    weights_red: Dict[str, float] = {}
    weights_blue: Dict[str, float] = {}
    for name, red_p in red_pool.items():
        brier_red, count = 0.0, 0
        for i in range(-n_eval, 0):
            t = draws[i]
            tr = np.zeros(33)
            for r in t["reds"]: tr[r-1] = 1.0
            brier_red += float(np.mean((red_p - tr) ** 2))
            count += 1
        if count > 0:
            weights_red[name] = max(0.01, 1.0 / max(brier_red / count, 0.001))
        else:
            weights_red[name] = 0.01
    for name, blue_p in blue_pool.items():
        brier_blue, count = 0.0, 0
        for i in range(-n_eval, 0):
            t = draws[i]
            tb = np.zeros(16)
            tb[t["blue"]-1] = 1.0
            brier_blue += float(np.mean((blue_p - tb) ** 2))
            count += 1
        if count > 0:
            weights_blue[name] = max(0.01, 1.0 / max(brier_blue / count, 0.001))
        else:
            weights_blue[name] = 0.01

    # U3 自适应降权：对 red 池的方法按权重打折
    if adaptive:
        for name in list(weights_red.keys()):
            aw = adaptive.get(name, 1.0)
            weights_red[name] = weights_red.get(name, 0.01) * aw

    wtr = sum(weights_red.values()) or 1.0
    wtb = sum(weights_blue.values()) or 1.0
    weights_red = {k: v / wtr for k, v in weights_red.items()}
    weights_blue = {k: v / wtb for k, v in weights_blue.items()}
    print(f"[engine] Brier 红球权重: {weights_red} | 蓝球权重: {weights_blue}")

    red_blend = np.zeros(33)
    for name, m in red_pool.items():
        red_blend += weights_red.get(name, 0) * m
    blue_blend = np.zeros(16)
    for name, m in blue_pool.items():
        blue_blend += weights_blue.get(name, 0) * m
    red_blend = _apply_adaptive_red(red_blend, adaptive, red_pool)
    if red_blend.sum() <= 0:
        red_blend = np.full(R_MAX, 1.0 / R_MAX)
    else:
        red_blend = red_blend / red_blend.sum() * 6
    blue_blend = blue_blend / blue_blend.sum()
    return red_blend, blue_blend


def _apply_adaptive_red(red_blend: np.ndarray, adaptive: Optional[Dict],
                        red_pool: Dict) -> np.ndarray:
    """把自适应权重施加到红球融合上（blue 侧已在权重表打折）。"""
    if not adaptive:
        return red_blend
    out = np.zeros_like(red_blend, dtype=float)
    for name, m in red_pool.items():
        out += adaptive.get(name, 1.0) * m
    return out if out.sum() > 0 else red_blend


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
                llm_verify: Optional[bool] = None,
                deadline: Optional[float] = None,
                strict: bool = False,
                red_probs: Optional[np.ndarray] = None,
                blue_probs: Optional[np.ndarray] = None) -> List[Dict]:
    """多模型 LLM 采样生成候选（U3 角色重定位：组合结构优化，非自由猜号）。

    red_probs/blue_probs: 统计/ML 的 33/16 维概率分布，供 LLM 在硬约束下挑选
            红球（LLM 不再越权自由选号）；蓝球最终由 blue_probs（含 blue_specialist）
            加权覆盖，**LLM 的 blue 仅作提示并被引擎覆盖**。

    llm_samples: 覆盖全局 LLM_SAMPLES（离线评估可传 1 以控制成本与时长）。
    llm_verify: 是否执行第三轮校验（默认跟随 config.LLM_VERIFY_ENABLED）。
    deadline: LLM 阶段墙钟截止时间戳（time.time() 口径），钳制每次调用超时。
    strict: True 时通道不可用/超时/输出不可解析直接抛 LLMChannelError（Web 预测
            快速失败，不再降级）；False 保持原有“失败返回空列表降级统计”语义。
    """
    from concurrent.futures import ThreadPoolExecutor
    if config.LLM_DISABLED:
        if strict:
            raise LLMChannelError("LLM 已被停用（设置页「停用 LLM」或 LOTT_LLM_DISABLED=1）")
        return []
    model_cfgs = config.llm_model_list()
    if config.LLM_EVAL_MODEL:
        # 评估专用模型（LOTT_LLM_EVAL_MODEL）：评估期间限定单模型，控制成本
        filtered = [c for c in model_cfgs
                    if c.get("model") == config.LLM_EVAL_MODEL or c.get("name") == config.LLM_EVAL_MODEL]
        if filtered:
            model_cfgs = filtered
    if not model_cfgs:
        if strict:
            raise LLMChannelError("LLM 通道未配置（API 地址 / Key / 模型缺失，请在设置页配置并保存）")
        print("[llm] 无可用模型配置，跳过 LLM 通道")
        return []
    ctx = build_context(draws, stats, patterns)

    # 观察轮次：多模型轮转尝试（任一成功即用）。
    # 单模型挂掉（限流/参数不兼容/余额）不应废掉整个 LLM 通道——
    # 配置多个模型（LOTT_LLM_MODEL_LIST / LOTT_LLM_EXTRA_MODELS）即自动获得观察轮容错。
    obs_rotation: List[Dict] = []
    _seen = set()
    for c in model_cfgs:
        k = (str(c.get("base_url")), str(c.get("model")))
        if k in _seen:
            continue
        _seen.add(k)
        obs_rotation.append(c)
    obs = None
    last_obs_err: Optional[str] = None
    for c in obs_rotation:
        try:
            obs = llm_client.chat_json(
                llm_client.SYSTEM_BASE,
                llm_client.observations_prompt(
                    llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"],
                    feedback=ctx.get("feedback")),
                max_tokens=1600, temperature=0.7, model_cfg=c,
                deadline=deadline, strict=strict,
            )
        except LLMChannelError as e:
            last_obs_err = str(e)
            obs = None
        if obs is not None:
            break
    if obs is None:
        if strict:
            raise LLMChannelError(last_obs_err or
                                  "LLM 观察轮返回内容无法解析为 JSON（全部模型尝试失败）")
        print(f"[llm] 观察生成失败（已尝试 {len(obs_rotation)} 个模型），跳过 LLM 通道")
        return []

    n_models = len(model_cfgs)
    tickets: List[Dict] = []

    def _ticket_call(cfg: Dict) -> List[Dict]:
        res = llm_client.chat_json(
            llm_client.SYSTEM_BASE,
            llm_client.tickets_prompt(
                llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"], obs,
                feedback=ctx.get("feedback"),
                red_probs=red_probs.tolist() if red_probs is not None else None,
                blue_probs=blue_probs.tolist() if blue_probs is not None else None),
            max_tokens=2000, temperature=0.9, model_cfg=cfg,
            deadline=deadline, strict=strict,
        )
        return _parse_tickets_response(res, cfg["name"])

    n_samples = int(llm_samples) if llm_samples else config.LLM_SAMPLES
    calls = [model_cfgs[i % n_models] for i in range(n_samples)]
    call_errors: List[str] = []
    with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as ex:
        futures = [ex.submit(_ticket_call, cfg) for cfg in calls]
        for f in futures:
            try:
                tickets.extend(f.result())
            except LLMChannelError as e:  # strict 模式下单个采样失败不影响其余采样
                call_errors.append(str(e))
    if strict and not tickets:
        raise LLMChannelError(call_errors[0] if call_errors
                              else "LLM 选号轮未返回任何可解析的候选")

    # 第三轮校验（M3.2）：critique → 发现问题才 refine（低温 + 校验模型）
    # 校验轮失败不致命（已有候选在手），严格模式下同样保留原候选
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
                    max_tokens=600, temperature=0.2, model_cfg=vcfg,
                    deadline=deadline, strict=False)
                if critique and critique.get("verdict") == "problematic":
                    refined = llm_client.chat_json(
                        llm_client.SYSTEM_BASE,
                        llm_client.refine_prompt(
                            llm_client.compact_stats(ctx["stats"]), critique, tickets,
                            feedback=ctx.get("feedback")),
                        max_tokens=2000, temperature=0.2, model_cfg=vcfg,
                        deadline=deadline, strict=False)
                    parsed = _parse_tickets_response(refined, vcfg.get("model", "verify"))
                    if parsed:
                        tickets = parsed
                        print(f"[llm] 第三轮校验已修正选号（{len(parsed)} 注，校验模型 {vcfg.get('model')}）")
            except Exception as ex:  # noqa: BLE001
                print(f"[llm] 第三轮校验异常，保留原候选: {ex}")
    # U3：LLM 不再决定蓝球 —— 蓝球由 blue_probs（含 blue_specialist）加权覆盖
    if blue_probs is not None and tickets:
        _bp = np.asarray(blue_probs, dtype=float)
        if _bp.sum() > 0:
            _bp = _bp / _bp.sum()
        for t in tickets:
            t["blue"] = _weighted_sample(_bp, 1, rng)[0]
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


# ---------- U2 蓝球模式与覆盖 ----------

def _apply_blue_mode(blue_blend: np.ndarray) -> np.ndarray:
    """按配置把蓝球概率切到目标口径：uniform 模式 / 运行命中率回退 → 均匀 16。

    蓝球命中理论基线 1/16=6.25%。若模型长期（近 BLUE_FALLBACK_WINDOW 期）
    命中率显著低于该值，则该期回退均匀，避免被模型“带偏”。
    """
    import numpy as _np
    if config.BLUE_MODE == "uniform":
        return _np.full(B_MAX, 1.0 / B_MAX)
    # 运行命中率回退（读近 N 期蓝球命中）
    rate = METH._blue_running_rate(config.BLUE_FALLBACK_WINDOW)
    if rate is not None and rate < config.BLUE_FALLBACK_MIN_RATE:
        print(f"[engine] 蓝球模型近{config.BLUE_FALLBACK_WINDOW}期命中率 "
              f"{rate:.3f} < {config.BLUE_FALLBACK_MIN_RATE}，本周期回退均匀 16")
        return _np.full(B_MAX, 1.0 / B_MAX)
    return blue_blend


def _pick_blue_coverage(scored: List[Dict], n_tickets: int) -> List[Dict]:
    """picked 阶段：优先让 n 注尽量覆盖 n 个不同蓝球（U2 蓝球覆盖优先）。

    先按置信度降序尽量每个蓝球只取 1 注铺满 n；铺不满（候选池不足）再放宽
    允许同蓝重复补齐（与旧逻辑同蓝上限 max(2, n//5) 对齐兜底）。
    """
    if n_tickets <= 0:
        return []
    if not scored:
        return []
    ordered = sorted(scored, key=lambda x: -x["confidence"])
    picked: List[Dict] = []
    used_blue = set()
    # 第一遍：每个蓝球取置信度最高的一注
    for t in ordered:
        if len(picked) >= n_tickets:
            break
        if t["blue"] in used_blue:
            continue
        used_blue.add(t["blue"])
        picked.append(t)
    # 第二遍：不足则放宽同蓝上限补齐
    if len(picked) < n_tickets:
        cap = max(2, n_tickets // 5)
        blue_count = {}
        for t in picked:
            blue_count[t["blue"]] = blue_count.get(t["blue"], 0) + 1
        for t in ordered:
            if len(picked) >= n_tickets:
                break
            if t["blue"] in used_blue and blue_count.get(t["blue"], 0) >= cap:
                continue
            if t["blue"] not in used_blue:
                used_blue.add(t["blue"])
            blue_count[t["blue"]] = blue_count.get(t["blue"], 0) + 1
            picked.append(t)
    return picked[:n_tickets]


def _pick_legacy_blue(scored: List[Dict], n_tickets: int) -> List[Dict]:
    """旧版蓝球分散：同蓝上限 max(2, n//5)。仅在 LOTT_BLUE_COVER=0 时使用（向后兼容）。"""
    if n_tickets <= 0 or not scored:
        return []
    ordered = sorted(scored, key=lambda x: -x["confidence"])
    picked: List[Dict] = []
    blue_count = {}
    for t in ordered:
        if len(picked) >= n_tickets:
            break
        if blue_count.get(t["blue"], 0) >= max(2, n_tickets // 5):
            continue
        blue_count[t["blue"]] = blue_count.get(t["blue"], 0) + 1
        picked.append(t)
    return picked


def _expand_blue_compound(base_reds: List[int], blue_probs: np.ndarray,
                          k: int, rng: random.Random,
                          red_blend: Optional[np.ndarray] = None,
                          blue_blend: Optional[np.ndarray] = None) -> List[Dict]:
    """U2 蓝球复式：红 6 固定 + 蓝 k 个不同 → k 注。

    蓝球按 blue_probs 加权不放回抽样（蓝球覆盖 k 个不同号）。
    """
    k = max(1, min(k, B_MAX))
    blues = _weighted_sample(blue_probs, k, rng)
    out: List[Dict] = []
    for b in blues:
        t = {"reds": list(base_reds), "blue": b, "method": "blend:blue_compound"}
        if red_blend is not None and blue_blend is not None:
            t["confidence"] = min(100.0, max(1.0, round(10 + 0.25 * ensemble_mass(
                red_blend, blue_blend, t["reds"], t["blue"]), 1)))
        else:
            t["confidence"] = 50.0
        out.append(t)
    return out


def _expand_dan_tuo(dan: List[int], blue: int, n_tickets: int,
                    rng: random.Random, red_blend: np.ndarray,
                    ctx: Dict, blue_blend: Optional[np.ndarray] = None) -> List[Dict]:
    """U2 红胆拖：胆 n 个固定，围绕胆从剩余红池抽“拖”补齐 6 红，产出 n_tickets 注。

    胆来自模型红球概率 Top；拖号从去掉胆后的红池按概率加权抽样。
    """
    dan = sorted(set(dan))
    n_dan = len(dan)
    if n_dan >= 6:
        t = {"reds": dan[:6], "blue": blue, "method": "blend:dan_tuo"}
        t["confidence"] = 50.0
        return [t]
    need = 6 - n_dan
    # 剩余红池：除胆外按 red_blend 加权（仅对 remaining 建立权重，长度一致）
    remaining = [x for x in range(1, R_MAX + 1) if x not in dan]
    base_w = red_blend.astype(float).copy()
    for d in dan:
        base_w[d - 1] = 0.0
    rem_w = np.array([base_w[x - 1] for x in remaining], dtype=float)
    out: List[Dict] = []
    seen = set()
    for _ in range(n_tickets * 8):
        if len(out) >= n_tickets:
            break
        if rem_w.sum() <= 0:
            rem_w = np.ones(len(remaining))
        p = (rem_w / rem_w.sum()).tolist()
        picks = set()
        tries = 0
        while len(picks) < need and tries < 200:
            c = rng.choices(remaining, weights=p, k=1)[0]
            picks.add(c)
            tries += 1
        if len(picks) < need:
            continue
        reds = sorted(list(dan) + list(picks))
        if pass_constraints(reds, blue, ctx):
            key = tuple(reds)
            if key in seen:
                continue
            seen.add(key)
            t = {"reds": reds, "blue": blue, "method": "blend:dan_tuo"}
            if blue_blend is not None:
                t["confidence"] = min(100.0, max(1.0, round(10 + 0.25 * ensemble_mass(
                    red_blend, blue_blend, reds, blue), 1)))
            else:
                t["confidence"] = 50.0
            out.append(t)
    return out


# ---------- U4 组合覆盖优化 ----------

def coverage_optimize(pool: List[Dict], n_tickets: int, r: int = 3,
                      n_blues: int = 4,
                      trials: int = 3000, rng: Optional[random.Random] = None) -> List[Dict]:
    """U4 覆盖优化器：从候选池选 n_tickets 注，最大化红球“至少命中 r 个”覆盖 +
    蓝球覆盖。贪心初始化 + 模拟退火精修。

    双色球独立随机，无法让期望命中率超数学极限；此处目标是“覆盖尽可能多
    不同红球组合”，让用户“中一次小奖”的体感覆盖最大化（结构层面）。
    """
    rng = rng if rng is not None else random.Random()
    pool = [dict(t) for t in pool]
    if n_tickets <= 0 or not pool:
        return []
    if len(pool) <= n_tickets:
        return pool[:n_tickets]

    # 贪心：逐步加入使“覆盖红球并集”增益最大的一注（红球并集是 r-覆盖的上近似）
    def _union_gain(current: set, t: Dict) -> int:
        return len(set(t["reds"]) - current)

    selected: List[Dict] = []
    covered: set = set()
    remaining = list(pool)
    while len(selected) < n_tickets and remaining:
        best = max(remaining, key=lambda t: _union_gain(covered, t))
        remaining.remove(best)
        covered.update(best["reds"])
        selected.append(best)

    # 模拟退火：交换单注，评分 = 红球并集大小 + 蓝球覆盖数权重
    blue_w = 1.5
    current = list(selected)
    cur_score = len(covered) + blue_w * len({t["blue"] for t in current})
    best_sel = list(current)
    best_score = cur_score
    for it in range(trials):
        if not remaining:
            remaining = [t for t in pool if t not in current]
        if not remaining:
            break
        i = rng.randrange(len(current))
        j = rng.randrange(len(remaining))
        old_t = current[i]
        new_t = remaining[j]
        current[i] = new_t
        cur_union = set()
        for t in current:
            cur_union.update(t["reds"])
        nb_score = len(cur_union) + blue_w * len({t["blue"] for t in current})
        delta = nb_score - cur_score
        # 接受更优；等分时以随温度衰减的概率接受（探索），随迭代降温
        if delta > 0 or rng.random() < (0.5 / (1 + it ** 0.6)):
            cur_score = nb_score
            if nb_score > best_score:
                best_score = nb_score
                best_sel = list(current)
        else:
            current[i] = old_t
    return best_sel[:n_tickets]


def predict_next(draws: List[Dict], use_llm: Optional[bool] = None,
                 n_tickets: Optional[int] = None, persist: bool = True,
                 use_ml: Optional[bool] = None,
                 rng: Optional[random.Random] = None,
                 llm_samples: Optional[int] = None,
                 llm_verify: Optional[bool] = None,
                 llm_required: bool = False,
                 llm_deadline: Optional[float] = None,
                 bet_mode: Optional[str] = None,
                 coverage: bool = False,
                 coverage_r: Optional[int] = None) -> Dict:
    """对下一期生成预测。

    use_ml: 是否把 M2 ML 概率模型（GBDT+RF 集成）并入 Brier 加权融合；
            默认跟随 config.ML_ENABLED。
    llm_required: M4.5 快速失败。True（Web 端明确勾选「使用大模型」）时，
            LLM 通道不可用/超时/输出异常直接抛 LLMChannelError，由 API 层
            转为「预测失败」明确提示，不再静默降级为纯统计模型。
            False（调度器/离线评估/CLI）保持原有优雅降级语义。
    llm_deadline: LLM 阶段墙钟截止时间戳；llm_required 时默认取
            now + config.LLM_TOTAL_TIMEOUT，保证响应落在代理超时窗口内。
    bet_mode: U2 投注模式。single（默认，单式）/ blue_compound（蓝球复式：
            红 6 固定 + 蓝 k 个）/ dan_tuo（红胆拖）。None = 跟随 config.BET_MODE。
    coverage: U4 覆盖优化。True 时从候选池做“至少命中 coverage_r 个红球”+蓝球覆盖
            的组合优化（贪心+退火），而非逐注独立采样。
    coverage_r: coverage 模式的目标红球命中数（默认 config.COVERAGE_R）。
    """
    if use_llm is None:
        use_llm = not config.LLM_DISABLED
    if use_ml is None:
        use_ml = config.ML_ENABLED
    if bet_mode is None:
        bet_mode = config.BET_MODE
    if bet_mode not in ("single", "blue_compound", "dan_tuo"):
        bet_mode = "single"
    # M4.2 方法 A/B 开关：按运行模式取生效规格（production=严格过滤 / research=全部启用）
    methods_spec = METH.effective_spec(config.METHOD_MODE, config.METHODS_SPEC)
    # U3 自适应降权权重（production 且开启时；否则空 = 不施加）
    adaptive = METH.adaptive_weights()
    # 关闭的通道不生成候选（LLM 同时省 API 成本）
    use_llm = use_llm and METH.is_enabled("llm", methods_spec)
    use_ml = use_ml and METH.is_enabled("ml", methods_spec)
    # U3：被自适应降权（downweighted）且权重大幅下调的方法在 production 移出候选池
    if adaptive:
        _drop = {m for m, w in adaptive.items() if w <= config.ADAPTIVE_DOWNWEIGHT_FACTOR}
        # 仅对已知方法族做过滤（统一按“族:叶子”全名匹配候选 method）
    # M4.5 快速失败：明确要求 LLM 却用不上 → 直接失败并给出可读原因
    if llm_required and not use_llm:
        if config.LLM_DISABLED:
            reason = "LLM 已被停用（设置页「停用 LLM」或 LOTT_LLM_DISABLED=1）"
        elif not config.llm_configured():
            reason = "LLM 通道未配置（API 地址 / Key / 模型缺失，请在设置页配置并保存）"
        else:
            reason = "LLM 方法通道已被方法开关关闭（LOTT_METHODS / 设置页方法开关）"
        raise LLMChannelError(reason)
    if llm_required and llm_deadline is None:
        llm_deadline = time.time() + config.LLM_TOTAL_TIMEOUT
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
    red_blend, blue_blend = _brier_blend(bl, draws, adaptive=adaptive or None)
    # U2：蓝球模式与运行命中率回退
    blue_blend = _apply_blue_mode(blue_blend)

    rng = rng if rng is not None else random.Random()
    candidates: List[Dict] = []

    # U3 自适应降权后的候选池：被大幅降权的方法不再产候选（production 且开启）
    def _method_allowed(method_name: str) -> bool:
        if not adaptive:
            return True
        w = adaptive.get(method_name, 1.0)
        return w > config.ADAPTIVE_DOWNWEIGHT_FACTOR

    # 各统计模型 + M2 ML 模型分别采样
    for name, model in bl.items():
        mname = "ml" if name == "ml" else f"stat:{name}"
        if not _method_allowed(mname):
            continue
        for _ in range(2):
            t = sample_stat_ticket(model["red"], model["blue"], ctx, rng)
            if t:
                t["method"] = mname
                candidates.append(t)
    # 均匀对照（M4.2 开关同样适用）
    if METH.is_enabled("uniform", methods_spec):
        for _ in range(2):
            t = sample_stat_ticket(None, None, ctx, rng, uniform=True)
            if t:
                t["method"] = "uniform"
                candidates.append(t)

    # LLM 候选
    # M4.5：Web 预测（llm_required=True）严格模式——大模型不可用或超时直接抛错，
    # 明确提示「预测失败」，不支持降级为纯统计；调度器/评估保持优雅降级。
    llm_cands = []  # type: ignore
    if use_llm:
        if llm_required:
            llm_cands = llm_tickets(draws, stats, patterns, rng,
                                    llm_samples=llm_samples,
                                    llm_verify=llm_verify,
                                    deadline=llm_deadline, strict=True,
                                    red_probs=red_blend, blue_probs=blue_blend)
            if not llm_cands:
                raise LLMChannelError("LLM 选号轮未返回任何有效候选（模型输出不可解析）")
        else:
            try:
                llm_cands = llm_tickets(draws, stats, patterns, rng,
                                        llm_samples=llm_samples,
                                        llm_verify=llm_verify,
                                        red_probs=red_blend, blue_probs=blue_blend)
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

    # LLM 实际生成注数模式（默认开，LOTT_LLM_ONLY_OUTPUT=0 关闭）：
    # 当 LLM 通道产出了有效候选时，最终注数以 LLM **实际返回**的注数为准，
    # 不再用统计/ML 候选补齐到 N_TICKETS——保证“显示注数 = LLM 真实产出注数”。
    # 仅在 single + 非 coverage 的标准路径生效；coverage / blue_compound / dan_tuo
    # 是显式的组合策略模式，其“展开成 k 注”语义与本模式冲突，故保持原逻辑。
    llm_only = bool(
        config.LLM_ONLY_OUTPUT
        and llm_cands
        and not coverage
        and bet_mode == "single"
    )
    if llm_only:
        llm_scored = [t for t in scored if t["method"].startswith("llm:")]
        # LLM 候选已过（红球,蓝球）去重；按置信度降序，最多保留 n_tickets 注，
        # 但**不补齐**到 n_tickets——LLM 产出多少就展示多少。
        llm_scored.sort(key=lambda x: -x["confidence"])
        picked = llm_scored[:n_tickets]
    # U4：coverage 模式 — 从候选池做组合覆盖优化
    elif coverage:
        cr = int(coverage_r if coverage_r is not None else config.COVERAGE_R)
        picked = coverage_optimize(scored, n_tickets, r=cr,
                                   n_blues=config.COVERAGE_BLUE_K,
                                   trials=config.COVERAGE_TRIALS, rng=rng)
    elif bet_mode == "blue_compound":
        # U2 蓝球复式：取红球置信度最高的一注红（固定），展开 k 个蓝球
        k = max(1, min(config.BLUE_COMPOUND_K, B_MAX, n_tickets))
        if scored:
            base = sorted(scored, key=lambda x: -x["confidence"])[0]
            picked = _expand_blue_compound(base["reds"], blue_blend, k, rng,
                                           red_blend=red_blend, blue_blend=blue_blend)
        else:
            picked = []
    elif bet_mode == "dan_tuo":
        # U2 红胆拖：胆 = 红球概率 top DAN_TUO_DANS，围绕胆拖多注
        dan = [int(x) for x in np.argsort(-red_blend)[:config.DAN_TUO_DANS] + 1]
        blue = int(np.argmax(blue_blend)) + 1
        picked = _expand_dan_tuo(dan, blue, n_tickets, rng, red_blend, ctx,
                                 blue_blend=blue_blend)
    else:
        # 默认 single：蓝球覆盖优先
        scored.sort(key=lambda x: -x["confidence"])
        picked = _pick_blue_coverage(scored, n_tickets) if config.BLUE_COVER else _pick_legacy_blue(scored, n_tickets)

    shortfall_reason = None
    if len(picked) < n_tickets:
        if llm_only:
            # 非缺陷：按设计以 LLM 实际产出注数为准，不再补齐统计票
            shortfall_reason = "llm_only_actual_count"
        else:
            shortfall_reason = "all_methods_disabled" if not scored else "candidate_pool_exhausted"
    n_distinct_blue = len({t["blue"] for t in picked})
    result = {
        "issue": issue,
        "requested_tickets": n_tickets,
        "actual_tickets": len(picked),
        "shortfall_reason": shortfall_reason,
        "llm_only_output": llm_only,
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "target_draw": {"issue": draws[-1]["issue"], "date": draws[-1]["date"],
                        "reds": draws[-1]["reds"], "blue": draws[-1]["blue"]},
        "tickets": picked,
        "llm_used": use_llm and bool(llm_cands),
        "llm_models": llm_models_used,
        "red_probs": red_blend.tolist(),
        "blue_probs": blue_blend.tolist(),
        "bet_mode": bet_mode,
        "coverage_mode": coverage,
        "blue_covered": n_distinct_blue,
        "blue_coverage_rate": round(n_distinct_blue / B_MAX, 4),
        "adaptive": METH.adaptive_status() if config.ADAPTIVE_ENABLED else None,
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
