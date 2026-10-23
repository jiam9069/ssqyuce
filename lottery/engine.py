"""集成预测引擎：统计模型 + LLM 通道 → 硬约束过滤 → 置信度评分 → Top-N 输出。"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import backtest as BT
from . import config, db, features as F, llm_client, methods as METH, ml_model, models as M
from .llm_client import LLMChannelError

# 基础常量下沉到 `engine_base.py`（断循环导入，见该文件头注释）。
# 仍在此处导出：`engine.R_MAX` 等 40+ 处外部引用不变。
from .engine_base import R_MAX, B_MAX, TICKET_COST  # noqa: E402


# `InsufficientDataError` 的真源已随 S1 迁到 `engine_constraints.py`
# （它由 `constraint_ctx` 抛出，放在一起才对）。这里重新导出，
# 保持 `except ENG.InsufficientDataError` 与 `from lottery.engine import ...` 可用。
from .engine_constraints import (   # noqa: E402
    _InsufficientDataError as InsufficientDataError,
)


# ---------- 硬约束 ----------

# ===========================================================================
# S1/S2/S4 硬约束、加权采样与混合分布质量 —— 已于 D-02 拆出
# 到 `lottery/engine_constraints.py`（原 34–143 行，逐字迁移）。
#
# 诚实边界：`pass_constraints` 是硬过滤，**改变不了任何一注的中奖概率**。
# ===========================================================================
from .engine_constraints import (   # noqa: E402
    _last_sums,
    constraint_ctx,
    pass_constraints,
    _weighted_sample,
    sample_stat_ticket,
    ensemble_mass,
)

# ---------- V4-B：融合权重（walk-forward 样本外 + softmax + 上限 + 收缩） ----------
#
# 历史缺陷（v4 前）：_brier_blend 用「最近 50 期」算样本内 Brier，再以
#   weights = max(0.01, 1/brier) 定权。三重错误叠加：
#   (a) 样本内：模型本身就是在这些期上拟合/平滑出来的，Brier 系统性偏乐观；
#   (b) 1/brier 无上界：样本内 brier→0 时权重→+∞（实测样本内 brier 0.001
#       会拿到约 149x 于均匀模型的权重），是追逐噪声而非贝叶斯融合；
#   (c) 无权重上限、无收缩：小样本抖动直接变成单一方法独裁。
# 现改为：严格 walk-forward 样本外 Brier → softmax(-brier/T) → 权重上限 → 收缩。

# 融合权重评估窗口（期）。旧值 50 保持不变，仅用于样本外评估。
# walk-forward 评估时必须保留的最小训练历史（期）。用于把评估窗口截断到
# len(draws)-300 之内，保证每个评估点都有足够历史重建模型。
# 少于该期数不做样本外评估 → 退化为等权（数据不足时诚实等权，不假装有信息）。
# softmax 温度 T：w ∝ exp(-brier/T)。T 越大越接近等权。红球 Brier 的合理量级
# 约 0.10-0.20（均匀模型 33 维、单期 6 命中约 0.149），取 0.05 使「brier 相差
# 0.05」≈ 权重比 e≈2.7x —— 有区分度但远不到独裁。旧 1/brier 在同样差距下可
# 达数十倍。蓝球 16 维 Brier 量级更小，用单独的温度。
# 任一方法在融合中的权重上限。防止单一方法独裁（旧逻辑无上限）。
# softmax 前的 Brier 保护下限，避免极端 0 值导致 exp 溢出/独裁。
# 注意：这不是旧的「1/brier 放大」——它只用于数值稳定，权重动态范围仍由温度控制。
# 权重收缩系数 λ：final = (1-λ)·uniform + λ·data。λ=0.5 防小样本抖动。


# ===========================================================================
# S3 融合权重与 Brier 加权 —— 已于 D-02 拆出到 `lottery/engine_blend.py`
#
# 迁出 `_BlendArray` / `_with_meta` / `_softmax_weights` / `_cap_weights` /
# `_rebuild_probs` / `_model_builder_table` / `_walk_forward_brier` /
# `_brier_blend` / `_renormalize` / `_apply_adaptive_red`（原 178–525 行，逐字）。
#
# ⚠ 与 `_last_sums` / `_weighted_sample` / `ensemble_mass` 的关系：
# 那三个与 S1/S2 的硬约束函数混杂，留在本文件（第 4 批再处理）。
# `ensemble_mass` 用到 `_BlendArray`，通过本薄壳访问，**不直接 import 子模块**。
# ===========================================================================
from .engine_blend import (
    # 融合权重超参（真源已迁到 engine_blend.py，此处重新导出保持兼容）
    BLEND_EVAL_WINDOW,
    BLEND_MIN_TRAIN,
    BLEND_MIN_EVAL,
    BLEND_SOFTMAX_TEMP_RED,
    BLEND_SOFTMAX_TEMP_BLUE,
    MAX_MODEL_WEIGHT,
    BLEND_BRIER_FLOOR,
    WEIGHT_SHRINKAGE,
          # noqa: E402
    _BlendArray,
    _with_meta,
    _softmax_weights,
    _cap_weights,
    _rebuild_probs,
    _model_builder_table,
    _MODEL_BUILDERS,
    _walk_forward_brier,
    _brier_blend,
    _renormalize,
    _apply_adaptive_red,
)


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
        # V7：把「阶段时效 + 验证结论」真正交给 LLM，而不是只给 A/B 级战绩。
        # usable=可用于出号；monitor=只观察（新兴/衰减/陈旧）；rejected=不可用。
        "regime_rules": _regime_rules_for_prompt(draws),
    }

# ===========================================================================
# S5 规律出号与规则应用 —— 已于 D-02 拆出到 `lottery/engine_rules.py`
# 原 128–335 行，逐字迁移。
# 诚实边界：规则不改变开奖随机性；最强候选经多重检验校正后 p=0.2995，不显著。
# ===========================================================================
from .engine_rules import (   # noqa: E402
    _regime_rules_for_prompt,
    _last_feedback,
    _rules_hit,
    _norm_sorted_reds,
    _rule_matches_ticket,
    _rule_noise_explainer,
    apply_rules_to_candidates,
)


def _sanitize_patterns_used(raw, known: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """把 LLM 返回的 ``patterns_used`` 过一遍白名单（V14-A4）。

    ## 为什么必须校验（产品审查员实测的���一幻觉通道）

    原实现是 ``"patterns_used": [str(x) for x in t.get("patterns_used", [])]`` ——
    **纯透传、无任何校验**，而 ``web/static/app.js:771`` 会把它原样渲染成
    「规律引用：…」。于是 LLM 随口写出一个根本不存在的规律名，
    前端就当成"系统真的用了这条规律"展示出来。

    当前 ``patterns_runtime.usable = 0``，这意味着**任何非空引用都是幻觉** ——
    系统此刻没有任何已证实的规律可用，LLM 却可以声称用了。

    ## 白名单口径

    `known` 为当前**真实存在且可用**的规律名集合（由调用方从快照传入）。
    未在集合内的一律丢弃，并记录被丢弃的原文，供审计而不是静默忽略。
    若 `known` 为空（当前生产正是这种情况），则**任何引用都会被丢弃** ——
    这是正确行为：没有可用规律时，"我用了规律"必然是假的。

    返回值包含 ``dropped``，让上层可以把"LLM 说了什么但被拒"留痕。
    """
    items: List[str] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            s = str(x).strip()
            if s:
                items.append(s)
    elif isinstance(raw, str) and raw.strip():
        # 少数模型会把数组写成逗号分隔字符串
        items = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]

    allowed = {str(x).strip() for x in (known or []) if str(x).strip()}
    kept = [s for s in items if s in allowed]
    dropped = [s for s in items if s not in allowed]
    return {"kept": kept, "dropped": dropped, "had_any": bool(items)}


def _parse_tickets_response(res: Optional[dict], method_name: str,
                            known_patterns: Optional[Iterable[str]] = None,
                            shape_ctx: Optional[Dict] = None) -> List[Dict]:
    """校验并规范化 LLM 返回的 tickets（含 evidence / counter_evidence / structure_scores）。

    `known_patterns`（V14-A4）是当前真实可用的规律名白名单；
    ``patterns_used`` 里不在白名单内的引用会被丢弃并记入
    ``patterns_used_dropped``，不再原样透传到前端当作"证据"。

    ## V28：逐注 L2 结构校验（`shape_ctx`）

    V18 把严格模式的门控依据改成 L2 结构约束，并承诺「任何人拿发过的票重跑
    `check_shape()` 就能验证」。但 2026-10-23 生产核查发现：

        issue=2026122 seq=4  [1,6,14,22,30,32]  → check_shape() = False
        原因：奇数个数=1，而 odd_range=[2,3,4]

    **根因**：门控只校验了 `shape_constraint_ctx()` 的**字段齐全**（7 个必需键在），
    从没校验过**发出去的票本身**。统计通道天然合规（`sample_stat_ticket` 内部
    按 ctx 采样），只有 LLM 通道漏了 —— 它吐的是自由文本号码串，不经过采样器。

    现在：`shape_ctx` 给定时逐注校验，**不丢弃**违规票（丢弃会让 LLM 票悄悄变成
    0 注，与 V16 修的「静默降级」同类），而是在每注上标记：

        l2_ok: bool                 该注是否满足 L2 结构约束
        l2_violations: List[str]    违反的具体维度（人读，如 "odd_count=1 不在 [2,3,4]"）

    门控（`routes_predict`）看到任何一注 `l2_ok=False` 即可据此拒绝整批 ——
    「没有可逐注复核的凭据就不出号」才是 L2 兜底的真实含义。
    """
    from . import shape_rules as _SR
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
            pat = _sanitize_patterns_used(t.get("patterns_used"), known_patterns)
            # V28：逐注 L2 校验（不丢弃，只标记 —— 丢弃=静默降级）
            _l2_ok: Optional[bool] = None
            _l2_viol: List[str] = []
            if shape_ctx is not None:
                _l2_ok = bool(_SR.check_shape(reds, shape_ctx))
                if not _l2_ok:
                    _l2_viol = _SR.explain_shape(reds, shape_ctx)
            out.append({
                "reds": reds, "blue": blue, "method": f"llm:{method_name}",
                "l2_ok": _l2_ok, "l2_violations": _l2_viol,
                "confidence": conf,
                "reasoning": str(t.get("reasoning", ""))[:300],
                "patterns_used": pat["kept"],
                "patterns_used_dropped": pat["dropped"],
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
                blue_probs: Optional[np.ndarray] = None,
                model_cfgs: Optional[List[Dict]] = None,
                shape_ctx: Optional[Dict] = None) -> List[Dict]:
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
    # V25：`model_cfgs` 由调用方注入（用户生效配置）时优先使用。
    #
    # 为什么必须有这条注入路径：旧实现**无条件**读 `config.llm_model_list()`
    # （全局/管理员默认通道），于是用户在设置页配了自己的 base_url / Key / model
    # 并保存后，**出号仍然走全局默认**。2026-10-23 实测：用户自备
    # `group/subscription @ magpie.ff88.de`，出号却去调
    # `agent-router/deepseek-v4-flash` 并拿到 HTTP 503「无可用渠道」——
    # 即"设置页显示已配置，出号却用另一条已死的通道"。
    # 这与 V24 修的 `/api/llm/test` 是同一类缺陷，只是发生在出号链路。
    if model_cfgs:
        model_cfgs = [dict(c) for c in model_cfgs]
    else:
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
    obs_errors: List[str] = []
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
            obs_errors.append(f"{c.get('model') or c.get('name') or '?'}: {e}")
            obs = None
        if obs is not None:
            break
    if obs is None:
        if strict:
            if obs_errors:
                # 报错要把**全部**模型的失败原因都带上，并以主通道（列表首个）的原因开头：
                # 否则尾部模型（例如设置页换模型后残留的旧模型 404）会把真正的原因
                # （如「模型只返回推理内容、未产出正文」）掩盖掉，让人查错方向跑偏。
                raise LLMChannelError(
                    f"LLM 观察轮全部模型失败（{len(obs_errors)} 个）——"
                    + "；".join(obs_errors))
            raise LLMChannelError("LLM 观察轮返回内容无法解析为 JSON（全部模型尝试失败）")
        print(f"[llm] 观察生成失败（已尝试 {len(obs_rotation)} 个模型），跳过 LLM 通道")
        return []

    n_models = len(model_cfgs)
    tickets: List[Dict] = []

    # V14-A4：**LLM 规律引用的白名单**。
    #
    # 只允许引用「真正传给 LLM 的那批规律」（ctx["patterns"]，A/B 级且已过回测）。
    # 不在此集合内的一律丢弃 —— 因为 LLM 不受约束时会把不存在的规律名写进
    # patterns_used，而前端会把它当"系统真的用了这条规律"渲染出来。
    # 当前生产 usable=0 时，ctx["patterns"] 可能为空，那时**任何引用都会被丢弃**，
    # 这正是正确行为：没有可用规律时，"我用了规律"必然是幻觉。
    _known_patterns = {
        str(p.get("key")) for p in (ctx.get("patterns") or [])
        if p and p.get("key")
    }
    _known_patterns |= {
        str(p.get("name_zh")) for p in (ctx.get("patterns") or [])
        if p and p.get("name_zh")
    }

    def _ticket_call(cfg: Dict) -> List[Dict]:
        res = llm_client.chat_json(
            llm_client.SYSTEM_BASE,
            llm_client.tickets_prompt(  # noqa
                llm_client.compact_stats(ctx["stats"]), ctx["recent"], ctx["patterns"], obs,
                feedback=ctx.get("feedback"),
                red_probs=red_probs.tolist() if red_probs is not None else None,
                blue_probs=blue_probs.tolist() if blue_probs is not None else None,
                regime_rules=ctx.get("regime_rules"),
                shape_ctx=shape_ctx),
            max_tokens=2000, temperature=0.9, model_cfg=cfg,
            deadline=deadline, strict=strict,
        )
        # `name` 与 `model` 是两种配置形态：全局配置（config.llm_model_list）
        # 两个都有，而**用户自备配置**（user_llm_store.effective_for_user）
        # 只有 model。V25 第一版在这里用 `cfg["name"]` 直接 KeyError，
        # 让"出号走用户通道"这个修复一上线就 500（实测 `预测引擎异常: 'name'`）。
        # 配置来源不同、形态就可能不同 —— 读取必须容忍，不能假定。
        return _parse_tickets_response(
            res, cfg.get("name") or cfg.get("model") or "?",
            _known_patterns, shape_ctx)

    n_samples = int(llm_samples) if llm_samples else config.LLM_SAMPLES
    calls = [model_cfgs[i % n_models] for i in range(n_samples)]
    call_errors: List[str] = []

    # MU-2：**把 contextvars 显式带进工作线程**。
    #
    # 实测问题：`ThreadPoolExecutor` 提交的可调用对象**不继承** contextvars
    # （不同于 AnyIO 在线程池里复制上下文的行为）。于是 `llm_client` 记账时
    # 拿到的 `current_user_id()` 恒为 None → 用量记不到任何人头上 →
    # 配额永远显示 0，**而且不会报任何错**（实测：真实 LLM 出号成功、
    # llm_used=true，但 today_tokens_used 仍是 0）。
    #
    # 修法是标准做法：在**提交处**捕获当前上下文，再在线程里跑。
    # 必须用 `copy_context()`（复制），不能把同一个 Context 对象传给多个线程 ——
    # Context 不允许被并发进入，会抛 "cannot enter context: already entered"。
    import contextvars as _ctxvars
    _parent_ctx = _ctxvars.copy_context()

    def _run_with_ctx(cfg: Dict) -> List[Dict]:
        return _parent_ctx.copy().run(_ticket_call, cfg)

    with ThreadPoolExecutor(max_workers=min(len(calls), 4)) as ex:
        futures = [ex.submit(_run_with_ctx, cfg) for cfg in calls]
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
                    parsed = _parse_tickets_response(refined, vcfg.get("model", "verify"),
                                                    _known_patterns, shape_ctx)
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


def _latest_audit_digest() -> Optional[Dict]:
    """V12-13：``predict_next`` 里消费的最近一次 AI 核对摘要（**只读、零阻塞**）。

    V10 §2.1.6 冻结的 ``result["ai_audit"]`` 形状::

        None | {"post", "verdict", "correct": None|bool, "ts"}

    * ``degraded``（LLM 没做）→ ``correct=None``，前端必须显示「本次未取得」；
    * **任何**异常都返回 ``None``（读审计不得把出号打挂）。
    """
    try:
        from . import ai_audit
        latest = ai_audit.load_audits(limit=1)
        if not latest:
            return None
        row = latest[0]
        verdict = str(row.get("verdict") or "")
        return {
            "post": row.get("post"),
            "verdict": verdict,
            # degraded = 没做，不是做错 —— 必须为 None，不许落成 False
            "correct": None if verdict == "degraded" else bool(verdict == "pass"),
            "ts": row.get("ts"),
        }
    except Exception:                                          # noqa: BLE001
        return None


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

# ===========================================================================
# S6 蓝球模式与覆盖规划 —— 已于 D-02 拆出到 `lottery/engine_blue.py`
# 原 759–1026 行，逐字迁移。诚实边界（返奖比恒 15.625%）见子模块头注释。
# ===========================================================================
from .engine_blue import (          # noqa: E402
    _min_significant_blue_sample,
    _blue_fallback_min_sample,
    _apply_blue_mode,
    blue_coverage_plan,
    _plan_blue_coverage,
    _pick_blue_coverage,
    _pick_legacy_blue,
)


# ---------- V4-C：奖金期望模型（覆盖优化的目标函数） ----------
#
# 奖级规则（中国福利彩票双色球官方口径，与 lottery/evaluate.py:prize_level 一致）：
#   一等奖 6 红 + 蓝      二等奖 6 红           三等奖 5 红 + 蓝
#   四等奖 5 红 / 4 红+蓝  五等奖 4 红 / 3 红+蓝  六等奖 蓝（3 红以下也可能六等奖）
# 单注成本固定 ¥2。
#
# **浮动奖金诚实声明**：一/二等奖为浮动奖（奖池分配），既无固定值也不可预知。
# 本模块强制要求调用方传入一个**明确的假设常量**，并在返回结果里以
# `jackpot_assumption` 字段原样标注，绝不假装知道真实金额。默认假设取自
# evaluate.PRIZE_CASH 的历史均值近似（一等 ¥6,500,000 / 二等 ¥180,000）。
TICKET_COST = 2.0
# ===========================================================================
# S7 奖金期望模型 —— 已于 D-02 拆出到 `lottery/engine_prize.py`
#
# 薄壳保留在这里，使 `engine.expected_prize` / `engine.PRIZE_CASH_FIXED` 等
# 仍可从本模块取到（外部 27 处引用不动）。子模块内容是**逐字迁移**，逻辑未改。
#
# 为什么在文件末尾而不是文件头做转发：本块被后面几个函数引用（如
# `_single_ticket_ev` 读 `PRIZE_CASH_FIXED`），末尾导入不算循环——
# `engine_prize` 只 `from . import engine as E` 惰性取用，不在 import 期解引用。
# ===========================================================================
from .engine_prize import (          # noqa: E402  逐字迁移，位置受限
    PRIZE_CASH_FIXED,
    JACKPOT_ASSUMPTION,
    _portfolio_blue_hit_prob,
    expected_prize,
    portfolio_expected_value,
)


# ===========================================================================
# V10-03 票单分布（解析计算，**不是模拟**）
# ===========================================================================
# S8 票单分布（V10-03）—— 已于 D-02 拆出到 `lottery/engine_distribution.py`
# 原 704–1220 行，逐字迁移（517 行，28 个符号）。
#
# **整块迁移、不拆散**：数学与三个边界共用一组常量，且 `_single_ticket_ev`
# 直接读 `PRIZE_CASH_FIXED`/`TICKET_COST`/`B_MAX`。拆散会让同一套数学
# 分散多处，比留在一个文件更难核对。
# ===========================================================================
from .engine_distribution import (   # noqa: E402
    _popcount_lut,
    _popcount33,
    _red_mask,
    _all_red_masks,
    _validate_tickets,
    _pairwise_stats,
    _variance_from_abar,
    _var_ratio,
    _variance_stats,
    _prob_at_least_r_single,
    _poisson_binomial_at_least_one,
    _iid_tail_probs,
    _exact_red_tail,
    p_at_least_r_mc,
    _single_ticket_ev,
    ticket_distribution,
    _expand_blue_compound,
    _expand_dan_tuo,
    RED_PER_TICKET,
    SIGMA1_SQ,
    E_HITS_PER_TICKET,
    VAR_RATIO_SLOPE,
    REDUNDANCY_NEUTRAL,
    REDUNDANCY_MAX,
    EV_RATIO_SIXTH,
    PRIZE_ASSUMPTION_TABLE_A,
    PRIZE_ASSUMPTION_TABLE_B,
    _EXACT_TAIL_CACHE_MAX,
)

# ---------- U4 组合覆盖优化（V10-03：可插拔目标 + 蓝球职责边界） ----------
#
# 冻结契约（V10-03 §3.4.3）：
#
#   **蓝球由 blue_coverage_plan 唯一负责（闭式最优 P(≥1蓝)=min(n,16)/16），
#   目标函数只作用于红球子问题。**
#
# 这条契约来自一个真实缺陷：V4 的 ``coverage_optimize`` 在选完红球后用
# ``plan`` 强制改写蓝球，而目标函数里却仍含 ``distinct_blue`` 项 —— 优化器
# 在为一个**马上会被覆盖掉的属性**爬坡（零梯度 + 把蓝球噪声注入红球选择）。
# 现在：``blue_probs is not None``（plan 生效）时，**任何目标都不含蓝球项**，
# 并在 ``coverage_optimize_compare()`` 的 ``objective_meta.gradient_blue_degenerate``
# 里如实回报 True；只有 plan 不生效（蓝球保持候选池原值）时，蓝球才是真实
# 自由度，才允许进入目标。
#
# 四种目标（config.COVERAGE_OBJECTIVE / 别名表 config.COVERAGE_OBJECTIVE_ALIASES）：
#
#   ev_conserving（默认，别名 expected_value / ev）
#       EV 守恒 —— 同注数下 EV 恒为 n×单注EV，与票单结构**无关**，所以目标里
#       **没有也不允许有 EV 项**；只比较红球结构：并集覆盖 − 冗余度 Ā。
#       这是唯一诚实的目标：它优化的东西（覆盖/方差结构）真实存在，
#       但**不宣称**提高期望收益。
#   pmax_high
#       直接最大化 P(≥1 注命中 ≥r 红)。**不得**用「最小化冗余度」或「最大化
#       并集覆盖」当代理：n=10 时铺开反而比 iid 低约 2.6%（票间负相关）。
#       搜索期用固定种子「公共随机数」MC 估计（噪声 <1%），结果须用
#       ``ticket_distribution()`` 给出精确值与 ``var_ratio``（方差代价）。
#   coverage
#       旧口径（V4）：红球并集 + 1.5×蓝球覆盖数，保留向后兼容。
#   custom_weight
#       用户偏好加权（P2-3 承接）：权重来自 config.COVERAGE_OBJECTIVE_WEIGHTS
#       或调用参数；项：pmax_high / union / neg_redundancy / blue_coverage。
#
# **任何模式都不许把目标写成「提高期望」。**

# ===========================================================================
# S9 组合覆盖优化 —— 已于 D-02 拆出到 `lottery/engine_orchestrate.py`
# 原 588–910 行，逐字迁移。
# 诚实边界：优化的是给定目标函数（保 EV / 最大 P(≥k红) / 最大覆盖），
# **不是提高中奖概率** —— 所有目标由组合数学决定，长期回报恒为负。
# ===========================================================================
from .engine_orchestrate import (   # noqa: E402
    COVERAGE_OBJECTIVES,
    DEFAULT_CUSTOM_WEIGHTS,
    _PMAX_SEARCH_DRAWS,
    _normalize_coverage_objective,
    _custom_objective_weights,
    _PmaxEstimator,
    _ticket_objective,
    _objective_meta,
    coverage_optimize,
    coverage_optimize_compare,
)


def channels_snapshot(*, force_production: bool = False) -> Dict:
    """本期出号通道快照（V10-04b：出号决策的**唯一**读口）。

    为什么不再直读 `config.LLM_ONLY_OUTPUT`（R3 P0-4 / R1 P0-1）：
    出号口径散落在 6 层开关里（LOTT_METHODS / LOTT_METHOD_MODE /
    LOTT_LLM_ONLY_OUTPUT / LOTT_LLM_DISABLED / methods_config.json / .env），
    引擎自己再判一次必然与 `methods.resolve_channels()` 漂移——
    "族被关掉但 only 仍生效"、"research 模式偷偷出 LLM 注"都来自这种二次判定。

    取值顺序：优先用 `config.CHANNELS` 的冻结快照（设置页保存后由
    `config.refresh_channels()` 刷新）；快照缺失或不含 V10 字段时才现算。
    """
    snap = getattr(config, "CHANNELS", None)
    if force_production or not isinstance(snap, dict) or "llm_can_pick" not in snap:
        return METH.resolve_channels(force_production=force_production)
    return snap


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
                 coverage_r: Optional[int] = None,
                 llm_model_cfgs: Optional[List[Dict]] = None) -> Dict:
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
    issue = BT.expected_issue(draws)["issue"]

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
            # ⚠️ 这一支（llm_required=True）正是**用户点「强制重新预测」**走的路径。
            # 第一版我只给下面的 else 支传了 model_cfgs，漏了这里 ——
            # 于是"用户配置注入"在最关键的路径上完全没生效，用户拿到的仍是
            # 全局默认通道的 503。必须两支都传。
            llm_cands = llm_tickets(draws, stats, patterns, rng,
                                    llm_samples=llm_samples,
                                    llm_verify=llm_verify,
                                    deadline=llm_deadline, strict=True,
                                    red_probs=red_blend, blue_probs=blue_blend,
                                    model_cfgs=llm_model_cfgs, shape_ctx=ctx)
            if not llm_cands:
                raise LLMChannelError("LLM 选号轮未返回任何有效候选（模型输出不可解析）")
        else:
            try:
                llm_cands = llm_tickets(draws, stats, patterns, rng,
                                        llm_samples=llm_samples,
                                        llm_verify=llm_verify,
                                        red_probs=red_blend, blue_probs=blue_blend,
                                        model_cfgs=llm_model_cfgs, shape_ctx=ctx)
            except Exception as e:  # noqa: BLE001
                print(f"[engine] LLM 通道异常，降级为纯统计: {e}")
                llm_cands = []
    candidates.extend(llm_cands)
    llm_models_used = sorted({
        t["method"].split(":", 1)[1] for t in llm_cands if t["method"].startswith("llm:")
    })
    # M4.2 兜底：最终按生效开关过滤候选（allow/deny 模式下丢弃关闭通道的票）
    candidates = METH.filter_candidates(candidates, methods_spec)

    # V7：规律 → 出号。规律快照（阶段时效 + 维度扫描 + 假说库）由
    # patterns_runtime 定期刷新，这里只读不重算（<1ms），并按 shadow/active 模式记录影响。
    from . import patterns_runtime
    _rule_snap = patterns_runtime.load_snapshot(current_issue=issue)

    # 去重 + 评分
    seen = set()
    scored: List[Dict] = []
    for t in candidates:
        key = (tuple(t["reds"]), t["blue"])
        if key in seen:
            continue
        seen.add(key)
        # ------------------------------------------------------------------
        # 候选置信度（P5′：M3 §8.3 H1 修复）
        # ------------------------------------------------------------------
        #
        # 历史实现给 LLM 族恒定 **+20** 的偏置（外加 0.45 vs 0.25 的斜率），
        # 于是 LLM 注**必然**排在候选池最前 —— 而排序纯按 confidence 降序。
        # 实测某期 25 注里 24 注来自 LLM，尽管 `llm_output_mode="off"`。
        #
        # 危害不是"排序偏好"这么轻，而是**让「LLM 行不行」永久不可测**：
        # 唯一能评估它的对照臂 `battle.llm_arm_report` 需要同期的非 LLM 票作分母，
        # 偏置把非 LLM 票全挤掉 → `n_pairs` 恒为 0 → 永远得不到结论。
        #
        # 现在：两族**同一基准**（`LLM_CONF_BIAS` / `OTHER_CONF_BIAS` 默认都是 0），
        # 形状上的差异（LLM 有自评项）保留 —— 那是族内信息，不是族间偏袒。
        #
        # ⚠️ 置信度是**结构分，不是中奖概率**。这里改的只是"谁排在前面"，
        # 不改变任何号码出现的概率（LEAD_VERDICT §8.1）。
        em = ensemble_mass(red_blend, blue_blend, t["reds"], t["blue"])
        if t["method"].startswith("llm:"):
            conf = round(config.LLM_CONF_BIAS + 0.45 * (0.5 * t.get("confidence", 50) + 0.5 * em), 1)
        else:
            conf = round(config.OTHER_CONF_BIAS + 0.25 * em, 1)
        t["confidence"] = min(100.0, max(1.0, conf))
        scored.append(t)

    # V7：把可用规律作用到**已评分候选**上（shadow 只记录、active 才真正改变号码）
    _rule_info = apply_rules_to_candidates(scored, draws, _rule_snap)
    _rule_active = _rule_info.get("mode") == "active"
    if _rule_active:
        # 用重排后的顺序重建 scored（置信度已在函数内更新）
        order = {(tuple(t["reds"]), t["blue"]): i for i, t in enumerate(scored)}
        scored.sort(key=lambda t: order.get((tuple(t["reds"]), t["blue"]), 10 ** 6))
    _rule_refs = _rule_info.get("evidence_by_ticket") or {}

    # LLM 出号通道（V10-04b：判据一律来自 resolve_channels() 快照，不再直读
    # config.LLM_ONLY_OUTPUT）。三种模式：
    #   off  → LLM 只提供候选，最终出号由确定性排序 + 蓝球覆盖计划决定；
    #   only → LLM 通道产出有效候选时，最终注数以 LLM **实际返回**的注数为准，
    #          不再用统计/ML 候选补齐到 N_TICKETS（"显示注数 = LLM 真实产出"）；
    #   mix  → LLM 与 stat/ml 候选同池混合排序后补齐到 N_TICKETS（= 默认路径）。
    # 且必须 `llm_can_pick`（族启用 + 非 off + production）：族被关掉时即使
    # 兼容开关 LOTT_LLM_ONLY_OUTPUT=1 也**绝不**走 LLM 出号（拒绝列表优先）。
    # 仅在 single + 非 coverage 的标准路径生效；coverage / blue_compound / dan_tuo
    # 是显式的组合策略模式，其"展开成 k 注"语义与本模式冲突，故保持原逻辑。
    _channels = channels_snapshot()
    _llm_mode = str(_channels.get("llm_output_mode") or "off")
    llm_only = bool(
        _channels.get("llm_can_pick")
        and _llm_mode == "only"
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
    # U4/V4：coverage 模式 — 从候选池做组合覆盖优化（默认奖金期望目标）
    elif coverage:
        cr = int(coverage_r if coverage_r is not None else config.COVERAGE_R)
        picked = coverage_optimize(scored, n_tickets, r=cr,
                                   n_blues=config.COVERAGE_BLUE_K,
                                   trials=config.COVERAGE_TRIALS, rng=rng,
                                   objective=config.COVERAGE_OBJECTIVE,
                                   blue_probs=blue_blend)
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
        # 默认 single：蓝球覆盖优先（V4：先定蓝球计划，再配对红球）
        scored.sort(key=lambda x: -x["confidence"])
        if config.BLUE_COVER:
            picked = _plan_blue_coverage(scored, n_tickets, blue_blend)
        else:
            picked = _pick_legacy_blue(scored, n_tickets)

    shortfall_reason = None
    if len(picked) < n_tickets:
        if llm_only:
            # 非缺陷：按设计以 LLM 实际产出注数为准，不再补齐统计票
            shortfall_reason = "llm_only_actual_count"
        else:
            shortfall_reason = "all_methods_disabled" if not scored else "candidate_pool_exhausted"
    # V7：票级「规律引用」——只写**真正参与该注计算**的规律，
    # 不再由 LLM 自行填写 pattern-key（那样既不可核验，也容易变成装饰性文字）。
    for t in picked:
        refs = _rule_refs.get((tuple(t["reds"]), t["blue"])) or []
        if refs:
            t["patterns_used"] = list(refs)
            t.setdefault("evidence", {})
            if isinstance(t.get("evidence"), dict):
                t["evidence"]["规律引用"] = "；".join(refs)
        elif _rule_info.get("mode") == "off":
            t.setdefault("evidence", {})
            if isinstance(t.get("evidence"), dict) and "规律引用" not in t["evidence"]:
                t["evidence"]["规律引用"] = "无可用规律（0 条通过阶段验证）"
        elif _rule_info.get("usable_rules"):
            t.setdefault("evidence", {})
            if isinstance(t.get("evidence"), dict) and "规律引用" not in t["evidence"]:
                t["evidence"]["规律引用"] = "可用规律未命中本注（该注完全由概率模型决定）"
    n_distinct_blue = len({t["blue"] for t in picked})
    # V4-C：覆盖指标与期望奖金（诚实口径：覆盖提高中奖频率，不提高期望收益）
    _blue_hit_prob = _portfolio_blue_hit_prob([t["blue"] for t in picked])
    _blue_hit_naive = (1.0 - (1.0 - 1.0 / B_MAX) ** len(picked)) if picked else 0.0
    _portfolio_ev = portfolio_expected_value(picked) if picked else None
    # V10-03：票单分布元数据（coverage 模式）。用 iid 近似保证热路径开销恒定
    # （精确枚举 C(33,6) 留给评估/报表/测试；需要时传 with_distribution_exact=True）。
    _coverage_dist = None
    if coverage and picked:
        try:
            _coverage_dist = ticket_distribution(picked, with_distribution_exact=False)
        except (ValueError, TypeError):
            _coverage_dist = None
    result = {
        "issue": issue,
        "requested_tickets": n_tickets,
        "actual_tickets": len(picked),
        "shortfall_reason": shortfall_reason,
        "llm_only_output": llm_only,
        # V10-04b（§2.1.6）：把本期生效通道一并回传，前端与审计据此渲染/核对，
        # 不再各自猜"LLM 到底能不能出号"。
        "channels": {
            "spec_raw": _channels.get("spec_raw"),
            "source": _channels.get("source"),
            "mode": _channels.get("mode"),
            "families": _channels.get("families"),
            "llm_output_mode": _llm_mode,
            "llm_can_pick": bool(_channels.get("llm_can_pick")),
            "fingerprint": _channels.get("fingerprint"),
        },
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
        "coverage_objective": config.COVERAGE_OBJECTIVE if coverage else None,
        "coverage_distribution": _coverage_dist,
        "blue_covered": n_distinct_blue,
        "blue_coverage_rate": round(n_distinct_blue / B_MAX, 4),
        "blue_hit_prob": round(_blue_hit_prob, 6),
        "blue_hit_prob_naive": round(_blue_hit_naive, 6),
        "portfolio_ev": _portfolio_ev,
        "adaptive": METH.adaptive_status() if config.ADAPTIVE_ENABLED else None,
        "rules": {
            "mode": _rule_info.get("mode"),
            "usable": len(_rule_info.get("usable_rules") or []),
            "n_boosted": _rule_info.get("n_boosted", 0),
            "n_resorted": _rule_info.get("n_resorted", 0),
            "swapped": _rule_info.get("swapped") or [],
            "note_zh": _rule_info.get("note_zh"),
            "usable_rules": _rule_info.get("usable_rules") or [],
            "snapshot": _rule_info.get("snapshot"),
            "apply_enabled": patterns_runtime.apply_enabled(),
        },
        "patterns_summary": {
            "A": sum(1 for p in patterns if p["grade"] == "A"),
            "B": sum(1 for p in patterns if p["grade"] == "B"),
            "C": sum(1 for p in patterns if p["grade"] == "C"),
        },
        "ml": _ml_result_block(ml_entry, use_ml, ml_extra),
        # ===== V28：`ai_audit` 从响应里移除（它永远是 None）=====
        #
        # V12-13 设了这个字段，本意是「读最近一次 AI 核对判定」。但 8 个版本下来：
        #   · `llm_audit` 表 **0 行**；
        #   · 唯一的写入路径 `ai_audit.save_audit()` **没有任何调用点**；
        #   · 唯一的触发器 `POST /api/llm/audit/run` 前端从未调用过。
        # 所以这个字段自诞生起就是恒定 `None` —— 它让响应看起来"有 AI 核对机制"，
        # 实际上什么都没有。这正是本项目最反对的**假承诺**。
        #
        # V28 直接不返回该字段（前端本来也没有渲染它，计数为 0）。
        # 审计模块本身保留（`/api/llm/audit/run` 仍可用，需要时手动触发并落库），
        # 只是不再在每期预测响应里挂一个恒空的占位。
        "note": ((f"已叠加 {len(_rule_info.get('usable_rules') or [])} 条通过阶段验证的规律"
                  if _rule_info.get("usable_rules") else
                  "样本外回测未发现稳定显著的规律，预测仅基于统计结构的均衡建议")
                 + "；置信度为模型结构分，不构成中奖概率。理性购彩。"),
    }
    if persist:
        db.save_features(issue, stats)
        db.save_predictions(issue, picked)
        # M4.1：保存方法、版本、模型与 LLM 配置快照，供开奖后长期对照
        db.save_eval_meta(issue, picked, result=result)
    return result


# ---------- V12-11 预算内最优组合求解器（L2-e） ----------
#
# 目标函数（``optimal_portfolio(n, objective)``）：
#
#   min_var            最小化 var_ratio = 1+(n−1)(11/54)(Ā−12/11)
#                      ⇔ 最小化 Ā ⇔ 最小化 ΣA。和值 ΣA 是**按注可分离的凸函数**
#                      （ΣA = ½(Σ_x cnt_x² − 6n)，g(cnt)=cnt² 凸），所以
#                      「号位均摊（每个红号出现 floor/ceil(6n/33) 次）」是全局最优，
#                      且任意一个均摊构造都达到同一 ΣA*（= 最优值，非近似）。
#   max_p_any_blue     仅优化蓝球覆盖：P(≥1蓝) = min(n,16)/16（闭式，与红球无关）。
#   max_p_at_least_r   在候选池（硬约束筛选后）里贪心 + 局部搜索最大化
#                      全枚举精确 P(≥1注命中 ≥r 红, r=4)。
#
# **所有目标都不宣称提高 EV**：同注数下 EV = n×单注EV，与注单结构无关
# （见返回值 ``ev_ratio_note`` 与 tests/test_v12_portfolio.py 的反向用例）。

# ===========================================================================
# S10 组合求解器 —— 已于 D-02 拆出到 `lottery/engine_solver.py`
# 原 1297 行至文件末尾，逐字迁移。
# 诚实边界：优化的是给定目标函数（最小方差/最大 P(≥1中蓝)），**不是中奖概率**。
# ===========================================================================
from .engine_solver import (   # noqa: E402
    PORTFOLIO_OBJECTIVES,
    PORTFOLIO_P_MAX_AT_LEAST_R,
    _balanced_min_sum_a,
    _balanced_counts,
    _balanced_portfolio,
    _portfolio_plan,
    _candidate_pool,
    _exact_p_at_least_r,
    optimal_portfolio,
    _search_max_p_at_least_r,
)
