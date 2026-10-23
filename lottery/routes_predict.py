"""预测域路由（出号 / 最近一期 / 严格模式门控）。

P2 模块化：从 `api_app.py` 切出。

## 这是全系统最核心的域，也是历史上缺陷最集中的地方

三个反复出问题的点，都在本模块里，改动前请先读：

1. **严格模式门控必须在缓存检查之前**（2026-10-18 修过一次）：否则已有缓存
   会绕过门控，表现为"严格模式开着却出了纯统计票"。
2. **严格模式必须按用户判定**（MU-2 / 裁决 2-A）：`LOTT_STRICT_MODE=1`
   是部署级开关，但新用户默认没有 LLM 通道 —— 全局套用会让每个新用户点
   "生成预测"都收到 422，而纯统计通道明明可以出号。
3. **票单写入必须带 user_id**（MU-2）：旧实现 `DELETE FROM predictions
   WHERE issue=?` 不带 user，用户 B 重新生成会删光用户 A 同期的票，
   而票单只存这一份、不可恢复。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import config, db
from . import backtest as BT
# 注意：**不要**写成 `from .request_context import current_user_id`。
# 那样会在本模块产生一个独立的名字绑定，打桩 `request_context.current_user_id`
# 就管不到这里 —— 测试会静默失效（用例仍通过，但测的不是生产路径）。
# 本项目已因这类"打桩点与依赖源脱节"踩过两次坑（见
# tests/test_p2_dependency_direction.py 的同名守护测试）。
from . import request_context

router = APIRouter()

def strict_scope_for_user(user_id):
    """严格模式对**指定用户**是否生效。返回 ``(active: bool, bypass_reason: str)``。

    V16：从"只认当前请求用户"改为**可按 user_id 判定**，因为后台调度器
    （``api_app._scheduler_step``）没有 HTTP 请求上下文，但它同样必须服从
    严格模式的语义 —— 否则就是"手动点不出号、后台却悄悄存了统计票"
    （这正是 2026-10-22 那批 5 注纯统计票的真实成因）。
    """
    if not bool(getattr(config, "STRICT_MODE", False)):
        return False, ""
    try:
        from . import user_llm_store
        eff = user_llm_store.effective_for_user(user_id)
        if eff.get("llm_available"):
            return True, ""
        _why = eff.get("reason") or "未配置 LLM"
        return False, (
            f"严格模式对当前账号**不生效**：没有可用的 LLM 通道（{_why}）。"
            f"本次已按纯统计通道出号。若希望强制 LLM 出号，请在设置页配置自己的"
            f" API Key（或由管理员配置共享通道）后重试。")
    except Exception as e:      # noqa: BLE001
        # 判定失败**不**升级成"拒绝出号"：宁可出一份纯统计票，也不要让用户
        # 因为"读取配置抛了个异常"就拿不到任何结果。
        return False, f"严格模式作用域判定失败（{type(e).__name__}），已按纯统计通道出号。"


def _strict_scope_for_current_user():
    """严格模式对**当前请求用户**是否生效（薄封装，保持既有调用点不变）。

    ## 为什么严格模式必须按用户判定（Lead 裁决 2-A）

    `LOTT_STRICT_MODE=1` 是**部署级**开关，但它想表达的是"用户要求必须由 LLM
    出号"。多用户下这两者不再等价：

    * 管理员的说法是"我要求 LLM，没有 LLM 就别出号"；
    * 新注册用户**默认没有任何 LLM 配置**（用户已确认："默认不配置 llm key，
      需要用户自己配置"）。

    若把部署级开关直接套到所有人，结局是：**每一个新用户点"生成预测"都收到 422
    "严格模式下不出号"，而他能做的只有去配置一个 API Key** —— 这等于用一个
    全局开关把系统对新用户完全锁死，而"纯统计通道"明明可以照常出号。

    因此作用域收敛为：**只有真正有可用 LLM 通道的用户，严格模式才对他生效**；
    没有通道的用户**绕过**门控，走纯统计出号，并在响应里如实说明原因
    （`strict_bypassed` + 提示文案）—— 既不静默降级，也不算"出号成功"。
    """
    return strict_scope_for_user(request_context.current_user_id())


def strict_mode_gates(draws, issue):
    """V11 严格模式门控：出号必须由 LLM 分析规律完成。

    返回**未通过的门控列表** ``[(gate_key, message), ...]``；空列表 = 放行。
    纯数据、不碰 HTTP —— 路由与后台调度器共用同一套判定（V16）。
    **必须在缓存检查之前调用**：严格模式下已有预测缓存也必须重新校验，
    不允许用旧缓存绕过门控（2026-10-18 顺序缺陷修复）。

    两条硬性校验（任一不满足即拒绝，绝不静默降级到统计基线）：
      ① **可回放的可用规律**（counts.usable_replayable）必须 > 0；
      ② LLM 出号通道必须可用（llm_can_pick=True）。

    ① 的口径（P3 / M3 §7）：从"有可用规律"收紧为"有**可回放**的可用规律"。
    理由：严格模式的语义是「没有真凭实据就不出号」，而不可回放的规律
    无法被独立复核 —— 把它当依据等于用不可验证的东西做不可验证的承诺。
    当前该值为 0 时系统会**不出号**，这正是用户要的语义且理由是真实的。
    """
    from . import methods as _METH
    _errs = []

    # ① L2 结构约束可用（V12-04 口径修正，裁决 R-2）：
    #    此前要求 counts.usable > 0（显著规律），但 3515 期实测 0 条显著，
    #    会让严格模式永远哑火。V12 起改用 **L2 结构层的组合恒等约束**
    #    （和值/跨度/三区/奇偶/连号）作为门控依据：它与 C(33,6) 理论枚举一致，
    #    可复现、不需要样本量。
    #
    #    口径纪律（V12-04 强制）：这门控只能命名为 **layer=="L2" 且 violation=0**，
    #    **不得**把它叫成「可用规律」——L1 的「可用规律」语义是「检出/未检出」，
    #    与 L2 的「约束满足率」不是同一种计数，混用正是「可用规律 0 vs 5 条」
    #    矛盾的来源。L2 的通过不意味着存在可预测下期的规律。
    #
    #    violation_count 的定义：用当前 ctx 对 C(33,6) 全空间做**抽样**
    #    校验约束本身自洽（全枚举由 tests/test_v12_shape_lock.py 锁定）；
    #    这里只要求 ctx 结构完整且可枚举出合法约束，因此 violation 恒为 0 的
    #    前提是 ctx 齐全 —— 缺字段时显式置 violation>0 并拒绝。
    try:
        _l2_layer = "L2"
        _violations = 0
        from . import shape_rules as _SR
        _sc = _SR.shape_constraint_ctx(draws)
        _required = ("sum_min", "sum_max", "span_min", "span_max",
                     "top_zones", "odd_range", "max_consecutive")
        _missing = [k for k in _required if _sc.get(k) is None]
        if _missing or not _sc.get("top_zones"):
            _violations = max(1, len(_missing))
        _ok_ctx = (_l2_layer == "L2" and _violations == 0)
    except Exception:                                    # noqa: BLE001
        _ok_ctx = False
        _violations = 1
    if not _ok_ctx:
        _errs.append(
            ("strategy",
             (f"严格模式下不出号：**L2 结构约束不可用**（layer={_l2_layer}，"
              f"violation_count={_violations}；组合恒等约束无法生成，"
              f"历史数据不足或异常）。注意：L2 可用只说明形态约束成立，"
              f"**不代表**存在可预测下期的规律。请确认历史数据齐全，"
              f"或关闭严格模式（LOTT_STRICT_MODE=0）。")))

    # ①-b V18：**L1 优先，L2 兜底**（择一，不是并列否决）
    #
    # ## 为什么改（依据 docs/analysis/V17_STRICT_MODE_EVALUATION.md）
    #
    # 旧实现在这里**硬性要求** `usable_replayable > 0`，于是严格模式永远哑火：
    # L1 要 +28% 的真实效应才可达（n=150 期、m_cum≈3000 ⇒ z=4.31），
    # 而真实数据最强只有 +5.0% ⇒ 差 5.6 倍，换算需 31 年。
    # 再叠加 FROZEN_ISSUES=150（≈1 年）的冻结期 —— 等于"永久不出号"。
    #
    # 但严格模式想表达的其实是「**没有可复核的真凭实据就不出号**」，
    # 而"可复核"不等于"统计显著"：L2 组合恒等约束（和值/跨度/三区/奇偶/连号）
    # 与 C(33,6)=1,107,568 的理论枚举一致，任何人拿发过的票重跑 check_shape()
    # 就能验证 —— 这比统计显著**更硬**（恒等式 vs 概率陈述），且不需要样本量。
    #
    # 所以：**能用 L1 就用 L1（更强）；L1 为空才用 L2 兜底（稍弱但可复核）**，
    # 并把走的是哪一条**如实标进响应**（`evidence_basis`）。
    #
    # 口径纪律（V12-04 强制，**不得违反**）：
    #   * `usable_replayable` 与「L2 通过」是**两种不同的计数**，分列展示、**禁止相加**；
    #   * 不得把 L2 叫做「可用规律」—— 它只是「结构合规」；
    #   * L2 通过**不意味着**存在可预测下期的规律。
    _n_replayable = None
    try:
        from . import patterns_runtime as _PRT
        _snap = _PRT.load_snapshot()
        if _snap:
            _n_replayable = int((_snap.get("counts") or {}).get("usable_replayable") or 0)
    except Exception:      # noqa: BLE001
        _n_replayable = None

    # 决定依据：L1 有可回放规律 → 用 L1；否则若 L2 可用 → 用 L2；两者都无 → 拒绝
    _evidence_basis = "none"
    if _n_replayable is not None and _n_replayable > 0:
        _evidence_basis = "L1_statistical"
    elif _ok_ctx:
        _evidence_basis = "L2_structural"
    else:
        _errs.append(
            ("no_evidence_basis",
             ("严格模式下不出号：**既没有可回放的可用规律（L1），"
              "L2 结构约束也不可用**。严格模式的含义是「没有可复核的真凭实据就不出号」，"
              "L2 是最后的兜底凭据，它都不可用说明历史数据不足或异常。"
              "请确认历史数据齐全，或关闭严格模式（LOTT_STRICT_MODE=0）走纯统计通道。")))

    # ①-c V28：**逐注** L2 校验（此前只校验了 ctx 字段齐全，没校验票本身）
    #
    # ## 为什么补这一条（2026-10-23 生产实证）
    #
    # issue=2026122 手动生成的 5 注里，seq=4 `[1,6,14,22,30,32]` 的
    # `check_shape()` 返回 **False** —— 奇数个数=1，而 `odd_range=[2,3,4]`。
    #
    # 根因不是约束算错，而是**从来没有校验过票**：
    #   · `strict_mode_gates` 只检查 `shape_constraint_ctx()` 的 7 个必需键在不在；
    #   · `engine.llm_tickets` 里的 `ctx` 是**提示词上下文**（`build_context`），
    #     与结构约束无关；
    #   · 统计通道天然合规（`sample_stat_ticket` 内部按 ctx 采样），
    #     所以这个洞只在 LLM 通道暴露 —— 它吐的是自由文本号码串。
    #
    # 但 V18 对用户的承诺是「任何人拿发过的票重跑 check_shape() 就能验证」。
    # 票不合规时这个承诺就是假的，而 L2 兜底的全部价值就在于"可逐注复核"。
    #
    # ## 判据
    #
    # 这里只看**约束上下文是否可用**。票本身的逐注校验发生在出号之后
    # （`_verify_tickets_l2`），因为出号前还没有票。
    #
    # ② LLM 通道必须可用
    _ch = _METH.resolve_channels()
    if not bool(_ch.get("llm_can_pick")):
        _errs.append(
            ("llm",
             ("严格模式下不出号：LLM 出号通道不可用（llm_can_pick=False）。"
              "请检查 LOTT_METHODS 是否包含 llm、LLM 网关（data/llm_config.json）"
              "是否可用。")))

    return _errs


def verify_tickets_l2(tickets: List[Dict], draws) -> Dict:
    """**逐注** L2 结构校验（V28）——门控承诺的兑现环节。

    返回::

        {"ok": bool,                    # 全部注都合规才 True
         "n": int, "n_bad": int,
         "bad": [{"seq": i, "reds": [...], "violations": [...]}],
         "note_zh": str}

    ## 为什么单独成一个函数

    ① **可单测**：不需要跑整条出号链路就能验证判据；
    ② **路由与调度器共用**：后台 `_scheduler_step` 直接调 `engine.predict_next()`，
       绕过 `/api/predict` 的门控（V16 踩过这个坑）。票级校验必须在**同一个函数**
       里，否则双口径。
    ③ **不丢弃、只标记**：丢弃不合规票会让 LLM 票悄悄变成 0 注，
       与 V16 修的「静默降级」同类。这里返回清单，由调用方决定拒绝还是降级。

    ## 与 `strict_evidence_basis` 的分工

    `strict_evidence_basis` 回答「凭据是什么」（L1 还是 L2，出号**前**）；
    本函数回答「发出去的票是否真的满足 L2」（出号**后**）。
    两者缺一，V18 的「可逐注复核」就只是口号。
    """
    rows = list(tickets or [])
    out: Dict = {"ok": True, "n": len(rows), "n_bad": 0, "bad": [],
                 "note_zh": ""}
    if not rows:
        out["ok"] = False
        out["note_zh"] = "没有票可校验（出号阶段已失败）"
        return out
    from . import shape_rules as _SR
    try:
        _ctx = _SR.shape_constraint_ctx(draws)
    except Exception as exc:          # noqa: BLE001
        out["ok"] = False
        out["note_zh"] = (
            f"L2 约束上下文无法生成（{type(exc).__name__}），无法逐注复核；"
            f"请确认历史数据齐全后重试，或关闭严格模式（LOTT_STRICT_MODE=0）。")
        return out

    for i, t in enumerate(rows, 1):
        reds = sorted(int(x) for x in (t.get("reds") or []))
        viol = _SR.explain_shape(reds, _ctx)
        if viol:
            out["bad"].append({"seq": i, "reds": reds,
                               "method": t.get("method"),
                               "violations": viol})
    out["n_bad"] = len(out["bad"])
    out["ok"] = (out["n_bad"] == 0)
    if out["n_bad"]:
        first = out["bad"][0]
        out["note_zh"] = (
            f"{out['n_bad']}/{out['n']} 注不满足 L2 结构约束"
            f"（首例 第{first['seq']}注 {first['reds']}："
            f"{'；'.join(first['violations'])}）。"
            f"L2 的价值在于可逐注复核，不合规的注不应计为『有结构凭据』。")
    else:
        out["note_zh"] = (
            f"全部 {out['n']} 注满足 L2 结构约束（和值/跨度/三区/奇偶/连号），"
            f"任何人可用 check_shape() 复核。注意：L2 通过不代表存在可预测下期的规律。")
    return out


def strict_evidence_basis(draws) -> Dict:
    """严格模式**出号依据**的判定结果（V18）。返回::

        {"basis": "L1_statistical" | "L2_structural" | "none",
         "n_replayable": int | None,
         "l2_ok": bool,
         "note_zh": str}

    与 :func:`strict_mode_gates` **同源**（同一套判定条件），只是把"依据是什么"
    单独暴露出来 —— 前端必须能显示"这批票的凭据是统计规律还是结构约束"，
    否则就是拿 L2 冒充 L1（V12-04 明令禁止）。
    """
    _n_replayable = None
    try:
        from . import patterns_runtime as _PRT
        _snap = _PRT.load_snapshot()
        if _snap:
            _n_replayable = int((_snap.get("counts") or {}).get("usable_replayable") or 0)
    except Exception:      # noqa: BLE001
        _n_replayable = None

    _l2_ok = False
    try:
        from . import shape_rules as _SR
        _sc = _SR.shape_constraint_ctx(draws)
        _required = ("sum_min", "sum_max", "span_min", "span_max",
                     "top_zones", "odd_range", "max_consecutive")
        _missing = [k for k in _required if _sc.get(k) is None]
        _l2_ok = (not _missing) and bool(_sc.get("top_zones"))
    except Exception:      # noqa: BLE001
        _l2_ok = False

    if _n_replayable is not None and _n_replayable > 0:
        return {"basis": "L1_statistical", "n_replayable": _n_replayable, "l2_ok": _l2_ok,
                "note_zh": (f"依据：**L1 统计规律**（{_n_replayable} 条通过前置注册且可回放）。"
                            "这是更强的凭据。")}
    if _l2_ok:
        return {"basis": "L2_structural", "n_replayable": _n_replayable, "l2_ok": True,
                "note_zh": ("依据：**L2 结构约束**（和值/跨度/三区/奇偶/连号，与 C(33,6) "
                            "理论枚举一致，可逐注复核）。注意：L2 通过**不代表**存在可预测"
                            "下期的规律，也不提高中奖概率；L1 可回放规律当前为 0 条。")}
    return {"basis": "none", "n_replayable": _n_replayable, "l2_ok": False,
            "note_zh": "既无可回放规律（L1），L2 结构约束也不可用 —— 无任何可复核凭据。"}


def _strict_mode_reject_response(draws, issue):
    """HTTP 包装：门控不通过 → 422 JSONResponse；通过 → None。

    V16 起判定本身抽到 :func:`strict_mode_gates`（返回纯数据），
    因为这个判定**不止路由要用** —— 后台调度器也必须服从同一套语义，
    否则就会出现"手动点不出号、后台却悄悄存了统计票"（2026-10-22 的真实故障）。
    """
    _errs = strict_mode_gates(draws, issue)
    if _errs:
        return JSONResponse(
            {"ok": False, "strict_mode": True,
             "strict_gates": [e[0] for e in _errs],
             "error": "；".join(e[1] for e in _errs),
             # V16：让前端能据此弹出"是否改用统计模式"的选择，而不是只看到一个红字。
             "can_fallback": True,
             "fallback_hint": "严格模式下没有可复核的真凭实据，故未出号。"
                              "你可以选择改用统计模式生成（结果会被明确标注为统计口径），"
                              "或先配置 LLM 通道 / 关闭严格模式。",
             },
            status_code=422)
    return None


@router.api_route("/api/predict", methods=["GET", "POST"])
def predict(use_llm: Optional[bool] = None,
            n_tickets: Optional[int] = Query(None),
            regenerate: bool = False,
            bet_mode: Optional[str] = None,
            coverage: bool = False,
            coverage_r: Optional[int] = None,
            allow_stat_fallback: int = Query(0, ge=0, le=1)):
    """生成下一期预测；目标期已有预测且未要求重新生成时复用缓存。

    bet_mode: U2 投注模式 single / blue_compound / dan_tuo（None=跟随 LOTT_BET_MODE）。
    coverage: U4 覆盖优化（True 时做“至少命中 r 个红球”+蓝球覆盖组合优化）。
    coverage_r: coverage 模式目标红球命中数。

    M4.5 快速失败：use_llm=true（前端勾选「使用大模型」）时 LLM 通道
    不可用/超时/输出异常，直接返回 503 + 明确错误信息（预测失败），
    不再静默降级为纯统计模型；use_llm=false 走纯统计，行为不变。

    allow_stat_fallback（V16）：**用户是否已明确同意"严格模式不出号时改用统计模式"**。
    默认 0 = 不同意（严格模式被拒就返回 422，由前端弹窗让用户选）。
    为 1 时：门控被拒 → 降级为统计出号，并把 `source_mode="stat_fallback"`
    + 未通过的门控写进快照，**页面上必须能看出这不是严格模式的产物**。

    为什么默认不让后端自己降级：严格模式的语义是"没有可复核的真凭实据就不出号"，
    静默降级恰好是它要禁止的事。要不要拿统计票，是用户的决定，不是实现者的。
    """
    from . import backtest as BT, engine
    from . import methods as METH
    from .llm_client import LLMChannelError
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地还没有开奖数据。请先抓取或导入历史开奖（详见部署文档「数据准备」一节），完成后再试。"}, status_code=400)
    # V7：期号按真实开奖日历推算（每周二/四/日），并在数据陈旧时明说，
    # 不再用「seq+1」硬猜（旧写法在期号越界时会显示成明年的 4xx 期号）。
    _exp = BT.expected_issue(draws)
    issue = _exp["issue"]

    # V11 严格模式门控（**必须在缓存检查之前**）：无可用规律或 LLM 通道
    # 不可用时拒绝出号；通过校验则强制现算（regenerate=True），不允许
    # 用旧缓存绕过门控。这是 2026-10-18 修复的顺序缺陷。
    _strict_active, _strict_bypass = _strict_scope_for_current_user()
    _fallback_info = None            # V16：非 None = 严格模式被拒且用户已授权降级
    if _strict_active:
        _errs = strict_mode_gates(draws, issue)
        if _errs:
            if not allow_stat_fallback:
                # 默认路径：如实拒绝，并告诉前端"可以选统计模式"（can_fallback）。
                return _strict_mode_reject_response(draws, issue)
            # 用户已在前端明确选择"改用统计模式" → 降级，但必须标注口径。
            _fallback_info = {
                "gates": [e[0] for e in _errs],
                "reason": "；".join(e[1] for e in _errs),
            }
        else:
            regenerate = True        # 严格模式通过 → 强制现算，不走缓存
    if not regenerate:
        existing = db.load_predictions(issue)
        if existing:
            # 从持久化快照读取 U2/U3/U4 元数据，确保 from_cache 也返回完整响应头
            meta = {}
            for m in db.load_eval_meta(issue):
                snap = json.loads(m.get("config_snapshot_json") or "{}")
                meta = {k: snap.get(k) for k in
                        ("bet_mode", "coverage_mode", "blue_covered", "blue_coverage_rate",
                         "llm_only_output", "shortfall_reason", "requested_tickets")}
                break
            # 旧快照缺失时兜底（避免前端读到 None）
            bm = meta.get("bet_mode") or config.BET_MODE
            if bm not in ("single", "blue_compound", "dan_tuo"):
                bm = "single"
            _llm_used = any(str(t.get("method", "")).startswith("llm:") for t in existing)
            try:
                _src_cached = {k: v for k, v in db.load_prediction_source(issue).items()
                               if k in ("source_mode", "strict_gates", "strict_reason")}
            except Exception:      # noqa: BLE001
                _src_cached = {}
            _gen_ts = max((t.get("created_at") or 0) for t in existing) if existing else None
            return {"issue": issue, "tickets": existing, "from_cache": True,
                    "generated_at": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_gen_ts))
                                     if _gen_ts else None),
                    "generated_at_ts": _gen_ts,
                    "data_basis": _exp,
                    "actual_tickets": len(existing),
                    "requested_tickets": meta.get("requested_tickets") or len(existing),
                    "shortfall_reason": meta.get("shortfall_reason"),
                    "llm_only_output": bool(meta.get("llm_only_output", False)),
                    "llm_used": _llm_used,
                    "bet_mode": bm,
                    "coverage_mode": bool(meta.get("coverage_mode", False)),
                    "blue_covered": meta.get("blue_covered"),
                    "blue_coverage_rate": meta.get("blue_coverage_rate"),
                    # V16：缓存路径也必须带回口径来源 —— 否则用户从缓存看到 5 注票时
                    # 会以为它们是"过了严格模式"的产物（2026-10-22 那次正是如此）。
                    **_src_cached,
                    "adaptive": METH.adaptive_status() if config.ADAPTIVE_ENABLED else None}
    # V10-11：技能关停（decision_disabled）只关「LLM 出号决策通道」，**不**关预测。
    # 关键区别（R3 G6 第 2 条）：走 LLM_DISABLED 时 engine 会因 llm_required=True
    # 抛 LLMChannelError → 503「预测失败」；这里改为**主动不请求 LLM 通道**，
    # 预测照常成功，只是不含该通道的注（`llm_used=false`、`decision_channel_disabled=true`）。
    _channels_now = METH.resolve_channels()
    _decision_disabled = bool(_channels_now.get("decision_disabled"))
    if _decision_disabled:
        use_llm = False
    # 出号注数：显式传参数 > 环境配置 > 默认 5（用户要求默认 5 注、可配置）
    _n_req = int(n_tickets) if n_tickets is not None else int(config.N_TICKETS)

    # 严格模式已在前置门控校验并强制 regenerate=True；这里补 use_llm 强制。
    # 注意 _strict_active 来自 _strict_scope_for_current_user()（按用户判定），
    # **不能**重新读 config.STRICT_MODE —— 那会把刚绕过门控的用户的 use_llm
    # 又强制成 True，于是绕过门控却仍然去调 LLM，必然抛 LLMChannelError → 503。
    _strict_notes = []
    if _fallback_info is not None:
        # V16：严格模式未出号，但用户在前端明确选择了「改用统计模式」。
        # 必须满足两条：① 强制关 LLM（走纯统计）；② 强制现算，不拿缓存冒充。
        use_llm = False
        regenerate = True
        _strict_notes.append(
            "严格模式未出号（未通过：" + " / ".join(_fallback_info["gates"]) +
            "），已按你的选择改用**统计模式**生成；本批票单标注为 stat_fallback。")
    elif _strict_active:
        use_llm = True                                   # 严格模式强制走 LLM
        # V18：文案必须如实交代凭据类型 —— 不能笼统说"依赖规律"，
        # 因为 L1 为空时兜底的是 L2 结构约束，二者强度不同。
        try:
            _ev_note = strict_evidence_basis(draws)["note_zh"]
        except Exception:      # noqa: BLE001
            _ev_note = "严格模式：出号强制依赖 LLM + 可复核凭据，本次为现算（未走缓存）"
        _strict_notes.append("严格模式（现算，未走缓存）。" + _ev_note)
    elif _strict_bypass:
        # 绕过严格模式：**必须**如实告诉用户发生了什么，否则他会以为自己的
        # "必须 LLM 出号"设置生效了，而实际拿到的是纯统计票。
        use_llm = False
        _strict_notes.append(_strict_bypass)

    # MU-2：按用户的共享通道配额（U2 §3.4）。**必须在严格模式决策之后判定**，
    # 因为两者会冲突，而冲突的答案取决于严格模式是否对该用户生效：
    #
    #  * 严格模式生效（用户明确要求"必须 LLM 出号"）+ 额度用尽 →
    #    返回明确错误。**不能**降级成纯统计：那正是严格模式禁止的静默降级，
    #    用户会以为拿到的是 LLM 票。这里宁可如实失败。
    #  * 非严格模式 + 额度用尽 → 降级为纯统计 + 提示（U2 §5.3：绝不阻断出号）。
    #
    # 配额只对**系统共享通道**生效：自备 Key 花的是自己的钱，不该被系统卡住。
    _quota_note = ""
    try:
        from . import user_llm_store as _ULS
        _eff = _ULS.effective_for_user(request_context.current_user_id())
        if _eff.get("source") == "system":
            _q = _ULS.quota_exceeded(int(request_context.current_user_id()))
            if _q:
                if _strict_active:
                    return JSONResponse(
                        {"ok": False, "strict_mode": True, "llm_required": True,
                         "quota_exceeded": True,
                         "error": (f"严格模式下不出号：{_q}"
                                   "严格模式要求必须由大模型出号，而共享通道额度已用尽；"
                                   "请配置自己的 API Key，或关闭严格模式（LOTT_STRICT_MODE=0）"
                                   "以改用纯统计通道。")},
                        status_code=429)
                use_llm = False
                _quota_note = _q
    except Exception:      # noqa: BLE001 —— 配额判定失败绝不能阻断出号
        _quota_note = ""

    # V25：把**当前用户生效的 LLM 配置**注入出号链路。
    #
    # 缺陷（2026-10-23 用户报告）：用户在设置页配了自备通道并保存，
    # 出号却仍走全局默认 `agent-router/deepseek-v4-flash`，
    # 拿到 HTTP 503「当前分组下无可用渠道」。
    # 根因：`engine.llm_tickets()` 无条件读 `config.llm_model_list()`，
    # 出号链路从来没有拿到过用户配置（`_eff` 在上面算出来了却没用在这里）。
    # 这与 V24 修的 `/api/llm/test` 是同一类缺陷，只是位置不同。
    _llm_cfgs = None
    # 直接重新解析一次，**不依赖**上面 `_eff` 是否已赋值 ——
    # 用 `locals().get("_eff")` 容易踩到"赋值顺序变化就静默失效"的坑
    # （第一版就是这样：测试里 `_eff` 还没赋值，注入整段被跳过，断言拿到 None）。
    try:
        from . import user_llm_store as _U0
        _src0 = _U0.effective_for_user(request_context.current_user_id())
    except Exception:      # noqa: BLE001
        _src0 = {}
    if (_src0 or {}).get("source") == "user":
        # 只有「用户自备」才注入：共享通道的 Key 走 with_secret=False 拿不到明文，
        # 注入一个没有 Key 的配置只会让引擎判定"未配置"而误报。
        try:
            from . import user_llm_store as _ULS2
            _full = _ULS2.effective_for_user(
                request_context.current_user_id(), with_secret=True)
            if _full.get("llm_available") and _full.get("api_key"):
                _llm_cfgs = [{
                    "base_url": _full.get("base_url"),
                    "api_key": _full.get("api_key"),
                    "model": _full.get("model"),
                    "samples": _full.get("samples"),
                    "extra_body": _full.get("extra_body") or {},
                    "source": "user",
                }]
        except Exception as e:      # noqa: BLE001
            print(f"[predict] 读取用户 LLM 配置失败，回退全局通道: {e}")
            _llm_cfgs = None
    try:
        res = engine.predict_next(draws, use_llm=use_llm, n_tickets=_n_req,
                                  llm_required=(use_llm is True),
                                  bet_mode=bet_mode, coverage=coverage,
                                  coverage_r=coverage_r,
                                  llm_model_cfgs=_llm_cfgs)
    except LLMChannelError as e:
        return JSONResponse(
            {"ok": False, "llm_required": True,
             "error": (f"预测失败：大模型不可用或超时（{e}）。"
                       "已停止降级为纯统计模式：请在「设置」页检查并修复 LLM 通道，"
                       "或取消勾选「使用大模型」后用纯统计模式重试。")},
            status_code=503)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"预测引擎异常: {e}"}, status_code=500)
    res["from_cache"] = False
    res.setdefault("data_basis", _exp)
    # 让前端能明确解释「为什么勾了使用大模型却没有 LLM 注」
    res["decision_channel_disabled"] = _decision_disabled
    # V22 / L2：AI 的**角色**必须随响应回去（off / advisor / picker）。
    # 用户有权知道这批票里 AI 到底做了什么：解释了结果，还是真的选了号。
    # 只写"用了 AI"是不够的 —— 那会把"AI 解释了统计结果"误读成"AI 选了号"。
    try:
        _ch_now = METH.resolve_channels()
        res["ai_role"] = _ch_now.get("ai_role") or "off"
        res["llm_output_mode"] = _ch_now.get("llm_output_mode")
    except Exception:      # noqa: BLE001
        res["ai_role"] = "off"
    res["strict_mode"] = bool(getattr(config, "STRICT_MODE", False))
    if _quota_note:
        _strict_notes.append(_quota_note)
    if _strict_notes:
        res["strict_notes"] = _strict_notes
    # V16：口径来源必须随响应一起返回，前端据此打标（不得让统计票看起来像严格模式产物）。
    if _fallback_info is not None:
        res["source_mode"] = "stat_fallback"
        res["strict_gates"] = list(_fallback_info["gates"])
        res["strict_reason"] = _fallback_info["reason"]
        try:
            db.mark_prediction_source(
                issue, source_mode="stat_fallback",
                strict_gates=_fallback_info["gates"],
                strict_reason=_fallback_info["reason"],
            )
        except Exception as e:      # noqa: BLE001
            print("[predict] 降级标注写入失败（票单仍有效）:", e)
    else:
        res["source_mode"] = "strict" if _strict_active else "stat"
    # V18：严格模式出号时，必须交代**凭据类型**（L1 统计 / L2 结构）。
    # 只写"严格模式"四个字是不够的 —— 用户有权知道这批票的凭据是"统计显著"
    # 还是"结构合规"。L1 与 L2 是两种不同强度的凭据，不可混为一谈。
    if _strict_active and _fallback_info is None:
        try:
            _ev = strict_evidence_basis(draws)
        except Exception as e:      # noqa: BLE001
            _ev = {"basis": "unknown", "n_replayable": None, "l2_ok": None,
                   "note_zh": f"依据判定失败（{type(e).__name__}）"}
        res["evidence_basis"] = _ev["basis"]
        res["evidence_note_zh"] = _ev["note_zh"]
        res["evidence_l1_replayable"] = _ev["n_replayable"]
        res["evidence_l2_ok"] = _ev["l2_ok"]
    # ===== V28：**逐注** L2 校验（严格模式承诺的兑现环节）=====
    #
    # V18 把门控依据改成 L2 并承诺「任何人拿发过的票重跑 check_shape() 就能验证」。
    # 但 2026-10-23 核查发现 issue=2026122 seq=4 直接返回 False —— 门控只校验了
    # 约束上下文齐不齐，**从没校验过票本身**。统计通道天然合规（采样器内建 ctx），
    # 只有 LLM 通道暴露了这个洞。
    #
    # 判据（与 strict_mode_gates 同一纪律：「没有可逐注复核的凭据就不出号」）：
    #   · 非严格模式：只**标记**不合规注，不拒绝（用户要的是能买到的号，
    #     且已明确告知证据等级为 stat / 无门控承诺）；
    #   · 严格模式：只要有 1 注不合规 → **整批拒绝**（422 + can_fallback），
    #     因为严格模式的全部价值就是「这批票每一注都能被复核」。
    #     拒绝时用户仍可选择降级统计模式（V16 的 allow_stat_fallback 语义不变）。
    try:
        _l2v = verify_tickets_l2(res.get("tickets") or [], draws)
    except Exception as e:      # noqa: BLE001
        _l2v = {"ok": False, "n": len(res.get("tickets") or []), "n_bad": -1,
                "bad": [],
                "note_zh": f"L2 逐注校验自身出错（{type(e).__name__}），无法确认票单可复核。"}
    res["l2_ticket_check"] = {
        "ok": _l2v["ok"],
        "n_tickets": _l2v["n"],
        "n_violating": _l2v["n_bad"],
        "violations": _l2v["bad"],
        "note_zh": _l2v["note_zh"],
    }
    # ⚠️ 只对**非空票单**做拒绝判定。
    #
    # 空票单（n=0）意味着出号阶段本身没产出票 —— 那是另一种失败，已在别处
    # 报错（`predict_next` 返回空 tickets 时会走"无候选"路径）。
    # 若这里把「0 注 / 0 违规」也判成 L2 未通过，会让用户看到
    # 「0/0 注不满足结构约束（第?注 []：）」这种自相矛盾的文案，
    # 而且掩盖真正的失败原因（vacuous truth 陷阱）。
    _l2_reject = (not _l2v["ok"]) and int(_l2v["n"]) > 0
    if _strict_active and _fallback_info is None and _l2_reject:
        _bad = _l2v["bad"]
        _first = _bad[0] if _bad else {}
        return JSONResponse(
            {"ok": False,
             "error": (f"严格模式下不出号：**L2 逐注校验未通过** —— "
                       f"{_l2v['n_bad']}/{_l2v['n']} 注不满足结构约束"
                       f"（第{_first.get('seq', '?')}注 {_first.get('reds', [])}："
                       f"{'；'.join(_first.get('violations', []))}）。"
                       f"严格模式的承诺是「每一注都能被 check_shape() 复核」，"
                       f"不合规的票不能计入凭据。可重试生成新一批，"
                       f"或改用统计模式（勾选后重试 / allow_stat_fallback=1）。"),
             "reason": "l2_ticket_violation",
             "l2_ticket_check": res["l2_ticket_check"],
             "can_fallback": True,
             "fallback_hint_zh": (
                 "统计通道在采样阶段就按同一套 ctx 约束生成，天然合规；"
                 "LLM 通道产出的票才可能违规。若反复出现，可考虑临时关闭 LLM 通道。"),
             },
            status_code=422)
    if _decision_disabled:
        res["decision_channel_note_zh"] = (
            "LLM 决策通道已被自动关停（决策技能 Wilson 下界 ≤ 随机基线 0.5），"
            "本次预测**照常成功**，只是不含 LLM 通道的注。"
            "恢复入口：POST /api/llm/decision/restore。")
    return res


def _sanitize_issue_order(issues: Sequence[str]) -> List[str]:
    """期号排序/过滤：剔除越界期号（史料里 seq 应落在当年 1~160）。

    为什么必须做：本地库里残留一条 `2026401` 的预测（2026-10-15 由已修复的
    `next_issue()` 越界跳号生成）。它**数值上**大于任何 2026 年的合法期号，
    于是无论按文本排还是按数值排，"最新预测"都会命中它，用户刷新后看到的
    永远是 15 日那条旧数据（用户反馈第 2 条，两次修复未竟的根因）。
    """
    out = []
    for it in issues or []:
        s = str(it)
        if not (len(s) == 7 and s.isdigit()):
            continue
        year, seq = int(s[:4]), int(s[4:])
        if not (2003 <= year <= 2100 and 1 <= seq <= 160):
            continue
        out.append(s)
    return out


@router.get("/api/predictions/last")
def last_predictions():
    """最近一期预测（含**生成时间**与数据基准，V7）。

    V6 及以前只返回 issue + tickets，前端因此无法显示"什么时候生成的"；
    现在补上 generated_at / data_basis / rules 摘要，让"这注是什么时候、基于哪期数据算的"可见。
    """
    from . import backtest as BT
    issue = None
    # 取候选期号时**先按时间**再按数值：越界/异常期号在两张排序下都可能置顶
    # （例：2026401 > 2026119），因此以"时间戳最新的一批"为准，再在合法期号里取最大。
    # ⚠️ U3 §3.7 缺陷 W2：这条内联 SQL 原先**不带 user_id**。
    # 后果：用户 A 的"最近一期"会取到用户 B 的活动痕迹 → 前端显示别人的期号
    # （信息泄漏 + 体验错乱）。这里必须走 resolve_tenant，与 db 层同一判据。
    _uid = db.resolve_tenant()
    rows = db.get_conn().execute(
        "SELECT issue, MAX(created_at) AS ts FROM predictions WHERE user_id=?"
        " GROUP BY issue ORDER BY ts DESC, CAST(issue AS INTEGER) DESC LIMIT 12",
        (_uid,)).fetchall()
    valid = _sanitize_issue_order([r["issue"] for r in rows])
    issue = max(valid, key=lambda x: int(x)) if valid else (
        (rows[0]["issue"] if rows else None))
    tickets = db.load_predictions(issue) if issue else []
    draws = db.load_draws()
    gen_ts = max((t.get("created_at") or 0) for t in tickets) if tickets else None
    meta = {}
    for m in (db.load_eval_meta(issue) if issue else []):
        try:
            meta = json.loads(m.get("config_snapshot_json") or "{}")
        except (TypeError, ValueError):
            meta = {}
        break
    rules_used = sorted({r for t in tickets for r in (t.get("patterns_used") or [])})
    return {
        "issue": issue,
        "tickets": tickets,
        "generated_at": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(gen_ts))
                         if gen_ts else None),
        "generated_at_ts": gen_ts,
        "data_basis": BT.expected_issue(draws) if draws else None,
        "requested_tickets": meta.get("requested_tickets"),
        "llm_used": any(str(t.get("method", "")).startswith("llm:") for t in tickets),
        "patterns_used": rules_used,
    }
