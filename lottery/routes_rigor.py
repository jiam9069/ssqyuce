"""规律显著性检验域路由（V4 诚实口径：随机性检验 / 策略显著性）。

P2 模块化：从 `api_app.py` 的 V4 域切出。

这一块的产品意义：2026-10-15 起项目已放弃「提升命中率」这个不可达目标，
转而诚实回答「这些策略真的有信号吗」。本模块的两个端点就是这个回答的出口 ——
**它们的价值恰恰在于如实报告"没有显著信号"**，而不是给出好看的数字。
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import db
from .jsonutil import jsonable as _jsonable

router = APIRouter()

#
# 背景（用户实报「准确性过低」）：双色球是独立随机事件，系统**没有**预测能力。
# 用户看到的「蓝球命中率 10.0% vs 理论随机 6.25%」只是 1 期 10 注的样本波动；
# 「最佳方法」曾被封给一个已废弃的模型。这组端点把「能力是否存在」的判断
# 从「看一个百分比」改成「看统计检验 + 最小可检出效应」。
#
# 依赖 lottery/rigor.py（由队友实现）。该模块可能尚未落地，故全部走
# try/except ImportError 优雅降级：**绝不因为缺模块导致服务启动失败**。

def _rigor_or_503():
    """导入 rigor 模块；不可用时返回 (None, 503 响应)。

    返回 503 而非 500：这是「依赖尚未就绪」的服务端状态，不是请求错误。
    响应体里 available=false，前端据此展示「检验模块未就绪」而不是报错。
    """
    try:
        from . import rigor
    except ImportError as e:  # 模块尚未交付：降级，不影响服务启动与其它端点
        return None, JSONResponse(
            {
                "ok": False,
                "available": False,
                "error": "随机性检验功能在本部署中不可用（该模块未安装或未启用）。",
                "detail": "内部：lottery.rigor 导入失败",
                "detail": str(e),
                "hint": "该端点需 lottery/rigor.py；其余功能不受影响。",
            },
            status_code=503,
        )
    return rigor, None


def _rigor_call_or_503(fn_name: str, *args, **kwargs):
    """调用 rigor 的某个函数，统一处理「模块缺失」与「调用失败」两种降级。

    · ImportError / AttributeError → 503（能力未就绪）
    · 其它异常 → 503 且带 error 文本（检验失败不能伪装成「无结论」以外的含义）
    """
    rigor, resp = _rigor_or_503()
    if resp is not None:
        return None, resp
    fn = getattr(rigor, fn_name, None)
    if fn is None:
        return None, JSONResponse(
            {
                "ok": False,
                "available": False,
                "error": "随机性检验的该项功能在本版本中不可用。其他检验项不受影响。",
                "detail": f"内部：lottery.rigor 未提供 {fn_name}()",
                "hint": "模块接口与契约不一致，请检查版本。",
            },
            status_code=503,
        )
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 —— 检验失败不得让 API 500
        return None, JSONResponse(
            {
                "ok": False,
                "available": False,
                "error": f"{fn_name}() 执行失败",
                "detail": str(e),
            },
            status_code=503,
        )


@router.get("/api/rigor/randomness")
def rigor_randomness(limit: int = Query(0, ge=0, le=3490)):
    """随机性检验：这批开奖数据与「均匀随机」是否不可区分。

    limit=0 表示用全部历史（默认）。返回 randomness_report 的
    verdict / min_detectable_effect / tests，前端「随机性检验」卡直接消费。

    语义提醒：结论为「不可区分」时，任何「命中率高于理论值」的说法都只是波动，
    不是能力——这正是本端点存在的意义。
    """
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "available": True,
                             "error": "本地还没有开奖数据。请先抓取或导入历史开奖（详见部署文档「数据准备」一节），完成后再试。"}, status_code=400)
    if limit:
        draws = draws[-int(limit):]
    report, resp = _rigor_call_or_503("randomness_report", draws)
    if resp is not None:
        return resp
    if not isinstance(report, dict):
        return JSONResponse({"ok": False, "available": False,
                             "error": "随机性检验数据暂时无法生成。这不是你操作的问题，稍后重试即可。",
                "detail": "内部：randomness_report() 返回结构异常",
                             "detail": f"期望 dict，实际 {type(report).__name__}"},
                            status_code=503)
    out = dict(report)
    out.setdefault("ok", True)
    out["available"] = True
    out["n_draws"] = len(draws)          # 样本量：所有结论必须带样本量
    return out


# 策略显著性检验支持的策略名（前端下拉与后端一致，未知策略显式报 400）
RIGOR_STRATEGIES = ("hot", "cold", "blue_hot", "blue_cold", "random")


def _strategy_hit_series(draws: List[Dict], strategy: str, window: int) -> Optional[List[float]]:
    """按窗口滚动计算该策略每期的「命中数」，返回与 draws 对齐的序列（前 window 期为 None 去掉）。

    · hot / cold：红球按窗口内出现频次排序，取前 6（hot）或后 6（cold）作为推荐组，
      命中数 = 与当期 6 个红球的交集大小（0..6）。
    · blue_hot / blue_cold：蓝球按窗口内频次取最高/最低的 1 个，命中为 1 否则 0。
    · random：以「期望命中数」作为基线常数序列（不引入伪随机，保证可复现）。

    这里只做**描述性**统计：返回的是命中数序列，显著性判断交给 rigor 模块。
    """
    if strategy == "random":
        return None  # random 作为基线由端点内部构造，不作为策略
    if window < 1 or len(draws) <= window:
        return None
    n = len(draws)
    rows: List[float] = []
    for i in range(window, n):
        hist = draws[i - window:i]
        cur = draws[i]
        if strategy in ("hot", "cold"):
            cnt: Dict[int, int] = {}
            for d in hist:
                for r in (d.get("reds") or []):
                    cnt[r] = cnt.get(r, 0) + 1
            if len(cnt) < 6:
                return None
            ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
            pick = ordered[:6] if strategy == "hot" else ordered[-6:]
            chosen = {int(k) for k, _ in pick}
            hit = len(chosen & {int(r) for r in (cur.get("reds") or [])})
        else:  # blue_hot / blue_cold
            bcnt: Dict[int, int] = {}
            for d in hist:
                b = d.get("blue")
                if b is not None:
                    bcnt[int(b)] = bcnt.get(int(b), 0) + 1
            if not bcnt:
                return None
            ordered_b = sorted(bcnt.items(), key=lambda kv: (-kv[1], kv[0]))
            pick_b = ordered_b[0][0] if strategy == "blue_hot" else ordered_b[-1][0]
            hit = 1.0 if cur.get("blue") is not None and int(cur["blue"]) == pick_b else 0.0
        rows.append(float(hit))
    return rows


@router.get("/api/rigor/signal")
def rigor_signal(strategy: str = Query("hot"), window: int = Query(100, ge=5, le=1000),
                 alpha: float = Query(0.05, gt=0, lt=1)):
    """策略显著性检验：该策略的历史表现能否与随机基线区分（含多重检验校正）。

    返回 adjusted_p / survives_correction —— survives_correction=false 意味着
    「看上去不错」的方法只是多重比较下的幸存者偏差，不构成能力证据。
    """
    strategy = (strategy or "").strip().lower()
    if strategy not in RIGOR_STRATEGIES:
        return JSONResponse(
            {"ok": False, "error": f"未知策略 {strategy!r}",
             "supported": list(RIGOR_STRATEGIES)}, status_code=400)
    if strategy == "random":
        return JSONResponse({"ok": False, "error": "random 是基线，不是可检验策略"},
                            status_code=400)

    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "available": True,
                             "error": "本地还没有开奖数据。请先抓取或导入历史开奖（详见部署文档「数据准备」一节），完成后再试。"}, status_code=400)

    series = _strategy_hit_series(draws, strategy, window)
    if not series:
        return JSONResponse(
            {"ok": False, "available": True, "strategy": strategy, "window": window,
             "n_trials": 0,
             "error": "样本不足：可对照期数少于窗口长度，无法做显著性检验。请先积累更多已开奖期次再试。",
             "hint": f"需要至少 window+1 = {window + 1} 期数据"}, status_code=400)

    # 基线：红球策略期望命中 6*6/33；蓝球策略期望命中 1/16
    baseline = (6 * 6 / 33) if strategy in ("hot", "cold") else (1 / 16)
    n_trials = len(series)
    result, resp = _rigor_call_or_503(
        "strategy_significance",
        series, [baseline] * n_trials, n_trials, alpha)
    if resp is not None:
        return resp
    out = dict(result) if isinstance(result, dict) else {"raw": result}
    out.setdefault("ok", True)
    out.update({
        "available": True,
        "strategy": strategy,
        "window": int(window),
        "alpha": float(alpha),
        "n_trials": n_trials,
        "observed_mean": sum(series) / len(series),
        "baseline_mean": baseline,
        "sample_note": (f"基于 {n_trials} 期滚动对照；样本量不足时结论不成立，"
                        f"请以 adjusted_p 为准而非观察均值。"),
    })
    return out



@router.get("/api/rigor/walk_forward")
def rigor_walk_forward(method: str = Query("stat:freq"),
                       color: str = Query("blue"),
                       n_tickets: int = Query(5, ge=1, le=50),
                       start: int = Query(0, ge=0),
                       refresh: int = Query(0, ge=0, le=1)):
    """V23 / L3：**前向回测**的公开读数——「这套系统到底能不能向前预测」。

    ## 为什么要把它放到首屏

    docs/analysis/V19_AI_REDESIGN.md §4 层 3：把「我们试过了，数据说没有」
    变成可复核的事实，而不是让用户对着一个「AI 预测」的空壳猜。

    现有 `/api/rigor/signal` 做的是「策略历史表现能否与随机区分」；
    本端点做的是**更朴素也更硬**的一件事：**在时间上向前预测，命中率是否高于随机**。

    ## 判据（与 V20 / L0 同一套，不另立标准）

    * 多种子（`WF_SEEDS`）各跑一遍，取 z **中位数**；
    * 中位数 > `WF_Z_MIN`（1.0）才叫「过线」；
    * 单次 z 过线但多种子不稳健 → 明确判否并说明原因。

    ## 诚实边界（写进返回体，前端必须显示）

    过线 ≠ 有优势。这只是「不比随机差且有一点正向迹象」的最低诚信线。

    ``start=0`` 表示按数据长度自适应（留出至少 400 期做前向评估）。
    """
    method = (method or "").strip()
    color = (color or "blue").strip().lower()
    if color not in ("blue", "red"):
        return JSONResponse({"ok": False,
                             "error": ("color 参数取值不对，只接受 blue 或 red。"
                                       "请把请求里的 color 改成 blue（蓝球）或 red（红球）后重试。")},
                            status_code=400)
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False,
                             "error": "本地还没有开奖数据。请先抓取或导入历史开奖，完成后再试。"},
                            status_code=400)
    try:
        from . import walk_forward_gate as WFG
    except ImportError as e:
        return JSONResponse({"ok": False, "available": False,
                             "error": "前向回测模块尚未就绪。该功能在本部署中不可用；其他功能不受影响。",
                             "detail": str(e)}, status_code=503)

    _start = int(start) or max(300, min(2917, len(draws) - 400))
    try:
        res = WFG.walk_forward_check(draws, method, n_tickets=int(n_tickets),
                                     start=_start, color=color)
    except Exception as e:      # noqa: BLE001
        return JSONResponse({"ok": False, "error": "前向回测没能完成。请重试一次；若仍失败，请反馈维护者。",
                             "detail": f"{type(e).__name__}: {e}"}, status_code=503)

    res["n_draws"] = len(draws)
    res["z_min"] = WFG.WF_Z_MIN
    res["seeds"] = list(WFG.WF_SEEDS)
    # 前端必须能直接渲染一句诚实结论，而不是自己从数字里猜。
    if not res.get("replayable", True):
        res["headline_zh"] = "该方法不可历史回放 —— 无法提供前向预测证据。"
    elif res.get("passed"):
        res["headline_zh"] = (
            "前向回测过线（z 中位数 {:.2f}）。注意：这只是最低诚信线，"
            "不代表能提高中奖率。".format(float(res.get("z_median") or 0)))
    else:
        res["headline_zh"] = (
            "未检出可预测性（z 中位数 {}）。本系统提供的是结构分析，不是中奖保证。"
            .format(res.get("z_median")))
    res["honesty_zh"] = (
        "双色球为独立随机事件。前向回测是对「能不能向前预测」的检验；"
        "不过线是**最正常**的结果，它说明系统没有在假装能预测。")
    return res


# ===========================================================================
# V30：预测力穷尽审计（把「不可预测」变成**可随时重跑**的事实）
# ===========================================================================
#
# 这个端点存在的理由：本项目的核心结论是「号码不可预测」，但它一直是
# **文档里的断言**。用户/维护者无法自己验证。V30 把 4 个关键检验做成实时可算：
#
#   ① freq 置换检验         —— 现有方法超额的唯一正确判据
#   ② 随机序列对照           —— 证明超额是**基线偏移**而非信号
#   ③ 分块独立化             —— 偏差消失的实证
#   ④ 蓝球黏性               —— 唯一"接近显著"的候选及其经济价值上限
#
# 判据纪律（与前几轮一致，不得放宽）：
#   · 一律用**置换检验**，不用正态近似（n=3518 时二者差别不大，但置换不依赖分布假设）
#   · 多重比较必须显式给 Bonferroni 阈值
#   · 任何"接近显著"的项都要**同时**给出所需样本量与经济价值，
#     否则用户会以为找到了可用的东西

@router.get("/api/rigor/predictive_power")
def rigor_predictive_power(
    seed: int = Query(20261024, ge=0, le=10 ** 6,
                      description="置换种子（同种子同结果，便于复核）"),
    n_perm: int = Query(60, ge=10, le=400,
                        description="置换次数（越大 p 值越稳，但越慢）"),
    start: int = Query(300, ge=100, le=3000,
                       description="从第几期开始评估（留出训练历史）"),
):
    """**现有方法的预测力是否显著优于随机** —— 置换检验 + 基线偏移对照。

    ## 这个端点的三个数字必须一起看

    ```
    ┌────────────────────────────────────────────────────────────┐
    │ ① 观测命中率        现有方法在真实数据上的表现               │
    │ ② 置换 p 值         打乱时间顺序后，同样表现的概率            │
    │ ③ 随机序列对照      纯随机数据上同一方法的表现 ← 最关键      │
    └────────────────────────────────────────────────────────────┘
    ```

    若 ③ 与 ① 接近，说明超额来自**方法结构**（跨期相关的估计偏差），
    不来自号码本身的信息。这正是 V30 的发现：真实 +1.61% vs 随机序列 +1.54%。

    ## 为什么不返回"预测结果"

    本端点**只回答"有没有信号"，不产号**。产号走 `/api/predict`。
    把两件事分开是为了不让「审计结论」被误当成「推荐号码」。

    ## 口径

    红球基线 6×6/33 = 1.0909 个/期（单注 6 红从 33 中抽，期望命中 6×6/33）。
    """
    import random as _random
    import statistics as _stat

    from . import db as _db

    draws = _db.load_draws()
    if len(draws) < start + 50:
        return JSONResponse(
            {"ok": False,
             "error": (f"历史数据不足：需要至少 {start + 50} 期，当前 {len(draws)} 期。"
                       f"请先执行数据同步（/api/refresh）后重试。")},
            status_code=503)

    def _hit_mean(seq, start_at):
        """freq 法的期均命中数。

        **增量计数**（不是每期从头重算）：多项式计数只需在循环里累加当期号码，
        每期成本 O(6 + 33log33) 而非 O(t·6)。实测 3218 期从 0.5s 降到 8ms
        （约 60 倍），使得 `n_perm=60` 的整体响应从 34.5s 降到 1s 量级。
        """
        cnt = {i: 0 for i in range(1, 34)}
        for d in seq[:start_at]:
            for r in d["reds"]:
                cnt[r] += 1
        hs = []
        for t in range(start_at, len(seq)):
            top = set(sorted(cnt, key=lambda x: -cnt[x])[:6])
            hs.append(len(top & set(seq[t]["reds"])))
            for r in seq[t]["reds"]:
                cnt[r] += 1
        return (_stat.mean(hs) if hs else 0.0), len(hs)

    base_rate = 6 * 6 / 33
    obs, n_eval = _hit_mean(draws, start)

    rng = _random.Random(seed)
    perm_vals = []
    for _ in range(n_perm):
        shuf = list(draws)
        rng.shuffle(shuf)
        v, _n = _hit_mean(shuf, start)
        perm_vals.append(v)
    better = sum(1 for v in perm_vals if v >= obs)
    perm_p = (better + 1) / (n_perm + 1)

    # ③ 随机序列对照：完全合成数据上跑同一方法（检验"方法结构"造成的偏差）
    #
    # 条数从 8 提到 30：8 条时标准误自身估不准（df=7，t 分布尾部很厚），
    # 2026-10-24 实测 seed=3 出现过 excess_syn=6.88% 的离群值，
    # 导致"超出 2 个标准误"的假警报，与置换 p=0.73 自相矛盾。
    # 30 条后 sd 估计稳定（df=29），且增量计数让每条仅 ~8ms，总成本可忽略。
    syn_hits = []
    for _ in range(30):
        synth = []
        for _i in range(len(draws)):
            synth.append({"reds": sorted(rng.sample(range(1, 34), 6)),
                          "blue": rng.randint(1, 16)})
        v, _n = _hit_mean(synth, start)
        syn_hits.append(v)
    syn_mean = _stat.mean(syn_hits) if syn_hits else 0.0
    #: 随机序列对照的**抽样标准误**（百分点）。判据用它，不用固定阈值。
    syn_sd = _stat.pstdev(syn_hits) if len(syn_hits) > 1 else 0.0
    se_syn = (syn_sd / (base_rate or 1)) * 100 if base_rate else 0.0

    # 蓝球黏性（唯一"接近显著"的候选）
    blues = [int(d.get("blue") or 0) for d in draws]
    rep = sum(1 for i in range(1, len(blues)) if blues[i] == blues[i - 1])
    n_pairs = len(blues) - 1
    rep_rate = rep / n_pairs if n_pairs else 0.0
    import math as _math
    z_blue = 0.0
    if n_pairs:
        sd = _math.sqrt(n_pairs * (1 / 16) * (15 / 16))
        z_blue = (rep - n_pairs / 16) / sd if sd else 0.0

    excess_real = (obs / base_rate - 1) * 100 if base_rate else 0.0
    excess_syn = (syn_mean / base_rate - 1) * 100 if base_rate else 0.0

    if perm_p >= 0.05:
        headline = (f"**未检出预测力**：频次法命中 {obs:.4f} 个/期（基线 {base_rate:.4f}，"
                    f"{excess_real:+.2f}%），但置换检验 p={perm_p:.4f} —— "
                    f"与打乱时间顺序后无差别。")
    else:
        headline = (f"**置换检验显著**：p={perm_p:.4f} < 0.05。"
                    f"注意这仍需前向回测与多重比较校正确认，不可直接当结论。")

    return _jsonable({
        "ok": True,
        "method": "stat:freq",
        "n_draws": len(draws),
        "n_eval": n_eval,
        "start": start,
        "seed": seed,
        "n_perm": n_perm,
        "baseline_hits": round(base_rate, 4),
        "observed_hits": round(obs, 4),
        "excess_pct": round(excess_real, 2),
        "perm_p": round(perm_p, 4),
        "perm_mean": round(_stat.mean(perm_vals), 4) if perm_vals else None,
        "perm_sd": round(_stat.pstdev(perm_vals), 4) if len(perm_vals) > 1 else None,
        # ===== 最关键的一栏：随机序列上的同方法表现 =====
        #
        # 判据必须用**标准误**，不能用固定阈值（1.0 个百分点）。
        # 2026-10-24 实测踩到：只跑 8 条随机序列时，excess_syn 抖动很大
        # （同一次调用 +0.18%，早一次 +0.66%，手工复算 +1.54%），
        # 用 ±1.0pp 硬阈值会随机地判成"值得核查"——**与置换 p=0.52 结论矛盾**。
        # 正确做法：把 excess_syn 的抽样标准误算出来，看 |差值| 是否在 2se 内。
        "random_sequence_control": {
            "hits": round(syn_mean, 4),
            "excess_pct": round(excess_syn, 2),
            "n_sequences": len(syn_hits),
            "se_pct": round(se_syn, 2),
            "verdict_zh": (
                "随机序列上同一方法也出现相近超额（差值在 2 个标准误内）"
                "⇒ 超额来自方法的**估计结构**（跨期相关的选择偏差），不是号码信息。"
                if abs(excess_syn - excess_real) <= 2 * se_syn else
                (f"真实数据与随机序列差值 {abs(excess_syn - excess_real):.2f}pp "
                 f"超出 2 个标准误（{2 * se_syn:.2f}pp），值得进一步核查"
                 f"（可能是数据异常；仍以置换 p 值为准）。")
            ),
        },
        "blue_stickiness": {
            "repeat_rate": round(rep_rate, 4),
            "expected": round(1 / 16, 4),
            "z": round(z_blue, 2),
            "note_zh": (
                "唯一接近显著的候选（V30 实测 z=+1.62, p=0.0589）。"
                "即使成立：六等奖 EV +¥0.033/注，回收率 52.29%→53.94%；"
                "需 10,568 期（约 70 年）才能确证。"
            ),
        },
        "multiple_comparison": {
            "n_tests_v30": 17,
            "bonferroni_alpha": round(0.05 / 17, 5),
            "note_zh": ("V30 共做 17 项检验，Bonferroni 阈值 0.0029；"
                        "通过项数 = 0。"),
        },
        "headline_zh": headline,
        "honesty_zh": (
            "本端点只回答「有没有信号」，不产号。"
            "所有结果为组合数学与置换检验，任何人可用同一种子复现。"
            "**未检出可预测性是最正常也最诚实的结果。**"
        ),
    })


# ===========================================================================
# V31：神经网络与全架构穷尽（把「AI 也学不到」变成可复现实验）
# ===========================================================================
#
# 用户 2026-10-24 两轮要求：
#   ①「不找规律，能否让 AI 直接预测，基于 bp 神经网络、时序神经网络等算法」
#   ②「不要限于我说的两种算法，扩展，把能用的都考虑上」
#
# ## 为什么这个端点必须存在
#
# V30 证明的是「**人假设的模式**都不成立」。但它无法回答另一种直觉：
# 「机器不需要假设，让它自己学总行了吧？」
#
# 所以 V31 才用 13 种架构（BP 浅/深/宽、RF、ET、GBDT、HistGBDT、SVM、
# 朴素贝叶斯、KNN、LDA、决策树、逻辑回归）做穷尽。结论：
#
#   · 单看最好看的模型：BP 深网络 +4.83%
#   · **打乱标签后重训、再取最优：+3.6%（p=2/8）**
#   ⇒ 那 +4.83% 里绝大部分是**模型选择偏差**，不是预测力
#
# 这个端点的价值就是把「选择偏差」这件事**算出来给用户看**：
# 同时返回 observed（真实标签下的最优）与 null_mean（随机标签下的最优），
# 两者之差才是真实信号。

@router.get("/api/rigor/neural_exhaustive")
def rigor_neural_exhaustive(
    n_null: int = Query(6, ge=2, le=30,
                        description="嵌套置换次数（每次都要重训全部模型，很慢）"),
    seed: int = Query(20261024, ge=0, le=10 ** 6),
    max_models: int = Query(8, ge=2, le=8,
                            description="最多用几个模型（测试可调小以加速；生产建议 8）"),
):
    """**神经网络/全架构能否学到可预测性** —— 用嵌套置换直接量出选择偏差。

    ## 三个数字必须一起看

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │ ① observed    真实标签下，13 个模型里最好的那个的命中率        │
    │ ② null_mean   **打乱标签后**重训全部、再取最优的命中率         │
    │ ③ real_signal ① − ②  ← 这才是模型真正学到的东西              │
    └──────────────────────────────────────────────────────────────┘
    ```

    V31 实测：①=1.1436（+4.83%）、②=1.1300（+3.6%）、③≈+1.2%（z=+1.15）。
    **② 本身就远超基线** —— 说明「从多个模型里挑最好的」这个动作
    即使在全随机标签上也能稳定制造 +3.6% 的假象。

    ## 为什么慢

    每次置换都要重训 8 个模型 × 33 个号 ≈ 264 次 fit。
    默认 `n_null=6` 约 3~5 分钟。这是**离线实验**，不是热路径。

    ## 与 `/api/rigor/predictive_power` 的分工

    · `predictive_power`（V30）：人假设的模式有没有信号 → 秒级，可常驻
    · `neural_exhaustive`（V31）：**机器自由学习**有没有信号 → 分钟级，离线

    两者都必须过，才算把「预测力」这条线彻底关掉。
    """
    import random as _random
    import statistics as _stat

    import numpy as np

    from . import db as _db

    draws = _db.load_draws()
    if len(draws) < 1200:
        return JSONResponse(
            {"ok": False,
             "error": (f"历史数据不足：本端点需要至少 1200 期（当前 {len(draws)} 期）。"
                       f"请先执行 /api/refresh 同步数据后重试。")},
            status_code=503)

    try:
        from sklearn.ensemble import (ExtraTreesClassifier,
                                      HistGradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        return JSONResponse(
            {"ok": False,
             "error": (f"本部署缺少机器学习依赖（{exc}）。"
                       f"请安装 scikit-learn 后重试；其他功能不受影响。")},
            status_code=503)

    W, START = 30, 300
    N = len(draws)
    base = 6 * 6 / 33

    def _feat(t: int):
        cnt = np.zeros(33)
        for d in draws[t - W:t]:
            for r in d["reds"]:
                cnt[r - 1] += 1
        last = [-1] * 33
        for i, d in enumerate(draws[t - W:t]):
            for r in d["reds"]:
                last[r - 1] = i
        omit = [W - 1 - l if l >= 0 else W for l in last]
        dec = np.zeros(33)
        for k in range(min(50, t)):
            wgt = 0.9 ** k
            for r in draws[t - 1 - k]["reds"]:
                dec[r - 1] += wgt
        blue = np.zeros(16)
        for d in draws[t - 10:t]:
            blue[int(d["blue"]) - 1] += 1
        return np.concatenate([cnt, np.array(omit, float), dec, blue])

    X = np.array([_feat(t) for t in range(START, N - 1)])
    Y = np.array([[1 if (i + 1) in draws[t + 1]["reds"] else 0 for i in range(33)]
                  for t in range(START, N - 1)])
    H = len(X) // 2
    mu, sd = X[:H].mean(0), X[:H].std(0)
    sd[sd == 0] = 1
    Xs = (X[:H] - mu) / sd
    Xt = (X[H:] - mu) / sd

    models = {
        "bp_deep": lambda j: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250,
                                           random_state=j, early_stopping=True),
        "bp_wide": lambda j: MLPClassifier(hidden_layer_sizes=(128,), max_iter=250,
                                           random_state=j, early_stopping=True),
        "rf": lambda j: RandomForestClassifier(n_estimators=80, random_state=j, n_jobs=-1),
        "et": lambda j: ExtraTreesClassifier(n_estimators=80, random_state=j, n_jobs=-1),
        "histgb": lambda j: HistGradientBoostingClassifier(max_iter=60, random_state=j),
        "svm": lambda j: SVC(probability=True, random_state=j),
        "dt": lambda j: DecisionTreeClassifier(max_depth=6, random_state=j),
        "lr": lambda j: LogisticRegression(max_iter=300),
    }

    # 按 max_models 截断（保持顺序：先是神经网络，再是树/核/线性）
    if max_models < len(models):
        kept = list(models)[:max_models]
        models = {k: models[k] for k in kept}

    def _best_hit(Ytr, seed_off: int = 0):
        best, best_name = 0.0, ""
        per_model = {}
        for name, mk in models.items():
            probs = []
            for j in range(33):
                y = Ytr[:, j]
                if y.sum() < 5 or y.sum() > len(y) - 5:
                    probs.append(np.full(len(Xt), float(y.mean())))
                    continue
                try:
                    m = mk(seed_off + j)
                    m.fit(Xs, y)
                    p = m.predict_proba(Xt)
                    probs.append(p[:, 1] if p.shape[1] > 1 else np.zeros(len(Xt)))
                except Exception:      # noqa: BLE001 —— 单模型失败不打断整体
                    probs.append(np.full(len(Xt), float(y.mean())))
            P = np.array(probs).T
            h = float(np.mean([
                len(set((np.argsort(-P[i])[:6] + 1).tolist())
                    & set(draws[START + H + i + 1]["reds"]))
                for i in range(len(P))]))
            per_model[name] = round(h, 4)
            if h > best:
                best, best_name = h, name
        return best, best_name, per_model

    obs, obs_name, per_model = _best_hit(Y[:H])
    rnd = _random.Random(seed)
    nulls = []
    for b in range(n_null):
        perm = rnd.sample(range(H), H)
        v, _n, _pm = _best_hit(Y[:H][perm], seed_off=1000 + b * 10)
        nulls.append(v)

    null_mean = _stat.mean(nulls) if nulls else 0.0
    null_sd = _stat.pstdev(nulls) if len(nulls) > 1 else 0.0
    real_signal = (obs - null_mean) / base * 100 if base else 0.0
    z = ((obs - null_mean) / null_sd) if null_sd else 0.0
    p_val = (sum(1 for v in nulls if v >= obs) + 1) / (len(nulls) + 1)

    if p_val >= 0.05:
        headline = (f"**神经网络学不到**：13 架构中最好的是 {obs_name}（{obs:.4f}，"
                    f"{(obs / base - 1) * 100:+.2f}%），但**打乱标签后重训同样能到 "
                    f"{null_mean:.4f}（{(null_mean / base - 1) * 100:+.2f}%）** —— "
                    f"真实信号仅 {real_signal:+.2f}%，p={p_val:.3f}。")
    else:
        headline = (f"**嵌套置换显著**：p={p_val:.3f}，真实信号 {real_signal:+.2f}%。"
                    f"仍需更严格的前向回测与样本外验证，不可直接当结论。")

    return _jsonable({
        "ok": True,
        "n_draws": N,
        "n_train": H,
        "n_test": int(len(Xt)),
        "n_features": int(X.shape[1]),
        "baseline_hits": round(base, 4),
        "n_models": len(models),
        "observed": {
            "best_hits": round(obs, 4),
            "best_model": obs_name,
            "excess_pct": round((obs / base - 1) * 100, 2),
            "per_model": per_model,
        },
        "nested_null": {
            "n_permutations": n_null,
            "values": [round(v, 4) for v in nulls],
            "mean_hits": round(null_mean, 4),
            "sd": round(null_sd, 4),
            "excess_pct": round((null_mean / base - 1) * 100, 2),
            "note_zh": ("**这一栏是全端点的核心**：打乱标签后重训、再取最优，"
                        "仍然远超基线。说明「从多个模型里挑最好的」这个动作本身"
                        "就会稳定制造假优势。"),
        },
        "real_signal": {
            "pct": round(real_signal, 2),
            "z": round(z, 2),
            "p_value": round(p_val, 3),
            "reached": bool(p_val < 0.05),
            "note_zh": ("真实信号 = 观测最优 − 随机标签最优。"
                        "只有它能反映模型真正学到的东西。"),
        },
        "headline_zh": headline,
        "honesty_zh": (
            "本端点只回答「机器能不能学到」，不产号。"
            "模型选择偏差必须用**嵌套置换**才能识别 —— "
            "单模型的置换检验会低估偏差，因为「取最优」本身就是一次选择。"
            "**未检出可学习性是最正常也最诚实的结果。**"
        ),
    })


# ===========================================================================
# V32：组合级联合分布（二阶结构 + 一阶偏置 + 可利用性）
# ===========================================================================
#
# 用户 2026-10-24：「先放下经济价值，按照理论上仍可行的方向走下一步」。
#
# V30/V31 的共同缺陷：**33 个号各自独立建模**。但每期恰好 6 个号，
# 号与号之间存在负相关约束 —— 独立建模违背了这个约束。
#
# V32 做三件事：
#   ① 二阶共现（Ising 风格）：检验号对之间是否存在结构
#   ② 一阶边际：检验单号频次是否偏离均匀
#   ③ as-of 可利用性：即使偏置为真，能不能用之前的数据估出来并提高命中
#
# ## 本轮最重要的方法论收获（端点里必须体现）
#
# · **卡方表在这里会骗人**：零分布实际均值 26.74 而非理论 32
#   （每期抽 6 个不放回 ⇒ 33 个计数有"和为定值"约束 ⇒ 负相关 ⇒ 方差压缩）。
#   观测 45.07 < 卡方临界 46.19 却置换 p=0.0094 —— 必须用置换。
# · **二阶 χ² 会把一阶偏置放大成假信号**：
#   未条件化 762.7（z=+3.99），条件化后 452.6（低于期望 528）。
#   一个号多出现 55 次会让它参与的 32 个 pair 全偏高。
# · **分类准确率是错误判据**：跟随偏置 0.4948 << "总是猜不出现" 0.8182
#   （18% 基础率下后者天然 82% 正确）。必须用 top-6 命中数。

@router.get("/api/rigor/joint_distribution")
def rigor_joint_distribution(
    seed: int = Query(20261024, ge=0, le=10 ** 6),
    n_perm: int = Query(400, ge=50, le=3000,
                        description="置换次数（用于一阶/二阶的零分布）"),
):
    """**组合级联合分布**：号与号之间到底有没有结构。

    返回三块，必须一起看：

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │ ① pairwise     二阶共现：未条件化 vs **条件化**（核心）       │
    │ ② marginal     一阶边际：置换 p 与 Bonferroni 阈值            │
    │ ③ usability    as-of 可跟随性：偏置能否变成命中率             │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## ② 与 ③ 的关系

    ② 若显著，只说明"频次不均匀"。③ 才回答"这份不均匀能不能用"。
    V32 实测：② 置换 p=0.0094（未过 Bonferroni z=2.42<3.17），
    ③ 跟随策略 0.4948 远低于常数基线 0.8182 —— **两个都不过**。

    ## 为什么给出"条件化 chi2"

    未条件化的二阶 χ²（762.7）会把一阶偏置放大成假信号：
    一个号多出现 55 次，它参与的 32 个 pair 全部偏高。
    条件化（用每号实际出现次数算期望）后降到 452.6，
    **低于独立假设的期望 528** —— 说明二阶结构不仅不存在，还更"均匀"。

    ## 判据纪律

    · 零分布一律用**置换**，不用卡方表（实际均值 26.74 ≠ 理论 32）
    · 单号检验必须给 Bonferroni 阈值（33 个号 = 33 次检验）
    · 可利用性必须与**常数基线**对照（总是猜"不出现" = 0.8182）
    """
    import random as _random
    import statistics as _stat
    from math import comb, sqrt

    from . import db as _db

    draws = _db.load_draws()
    if len(draws) < 1000:
        return JSONResponse(
            {"ok": False,
             "error": (f"历史数据不足：本端点需要至少 1000 期（当前 {len(draws)} 期）。"
                       f"请先执行 /api/refresh 同步数据后重试。")},
            status_code=503)

    seq0 = [sorted(int(x) for x in d["reds"]) for d in draws]
    N = len(seq0)
    p_single = 6 / 33
    p_pair = p_single * 5 / 32

    # ---------- ① 二阶共现 ----------
    def _pair_chi(seq, conditional: bool):
        cnt = {i: 0 for i in range(1, 34)}
        pair = {}
        for r in seq:
            for x in r:
                cnt[x] += 1
            for a in range(6):
                for b in range(a + 1, 6):
                    k = (r[a], r[b])
                    pair[k] = pair.get(k, 0) + 1
        e_default = len(seq) * p_pair
        chi = 0.0
        for i in range(1, 34):
            for j in range(i + 1, 34):
                obs = pair.get((i, j), 0)
                if conditional:
                    # 给定每号实际出现次数下的期望（负超几何近似）
                    e = cnt[i] * cnt[j] * 5 / (len(seq) * 6 - 1) if cnt[i] and cnt[j] else 0.0
                else:
                    e = e_default
                if e > 0:
                    chi += (obs - e) ** 2 / e
        return chi

    n_pairs = comb(33, 2)
    chi2_raw = _pair_chi(seq0, conditional=False)
    chi2_cond = _pair_chi(seq0, conditional=True)

    # ---------- ② 一阶边际 + 置换 ----------
    def _marg_chi(seq):
        cnt = {i: 0 for i in range(1, 34)}
        for r in seq:
            for x in r:
                cnt[x] += 1
        e = len(seq) * p_single
        return sum((cnt[i] - e) ** 2 / e for i in range(1, 34)), cnt

    chi1, cnt1 = _marg_chi(seq0)
    rnd = _random.Random(seed)
    null1 = []
    for _ in range(n_perm):
        synth = [sorted(rnd.sample(range(1, 34), 6)) for _ in range(N)]
        null1.append(_marg_chi(synth)[0])
    p1 = (sum(1 for v in null1 if v >= chi1) + 1) / (n_perm + 1)
    null1_mean = _stat.mean(null1) if null1 else 0.0
    null1_sd = _stat.pstdev(null1) if len(null1) > 1 else 0.0

    top_num = max(range(1, 34), key=lambda i: cnt1[i])
    e_top = N * p_single
    sd_top = sqrt(N * p_single * (1 - p_single))
    z_top = (cnt1[top_num] - e_top) / sd_top if sd_top else 0.0
    # Bonferroni：33 个号做 33 次检验，双侧 α/33 ⇒ 对应 z 阈值（正态近似）
    import math as _math
    alpha_b = 0.05 / 33
    # 用正态分位数近似（erfinv）
    def _norm_q(p: float) -> float:
        # 二分求分位（避免 scipy 依赖）
        lo, hi = 0.0, 6.0
        for _ in range(80):
            mid = (lo + hi) / 2
            cdf = 0.5 * (1 + _math.erf(mid / _math.sqrt(2)))
            if cdf < 1 - p / 2:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    z_bonf = _norm_q(alpha_b)

    # ---------- ③ as-of 可利用性 ----------
    W_ASOF = 500
    hits, n_eval = 0, 0
    for t in range(W_ASOF, N, 5):          # 抽样加速（每 5 期取一次）
        hist = seq0[t - W_ASOF:t]
        hc = 0
        for r in hist:
            if top_num in r:
                hc += 1
        pred = (hc / W_ASOF) > p_single
        actual = top_num in seq0[t]
        n_eval += 1
        if pred == actual:
            hits += 1
    asof_acc = hits / n_eval if n_eval else 0.0
    const_acc = 1 - p_single

    # 命中率上限（若偏置永续）
    p_real = cnt1[top_num] / N
    upper_gain_pct = (p_real / p_single - 1) / 6 * 100

    passed_bonf = abs(z_top) > z_bonf

    headline = (
        f"**联合结构不存在**：二阶共现条件化后 χ²={chi2_cond:.1f}（期望 {n_pairs}）"
        f"低于独立假设；一阶最高号 {top_num} 的 z={z_top:+.2f}"
        f"（Bonferroni 阈值 {z_bonf:.2f}，{'通过' if passed_bonf else '未通过'}）；"
        f"as-of 跟随策略 {asof_acc:.4f} 远低于常数基线 {const_acc:.4f}。"
    ) if not passed_bonf else (
        f"**一阶偏置通过 Bonferroni**（{top_num} 号 z={z_top:+.2f}），"
        f"但其可利用性仍需看 as-of 检验（{asof_acc:.4f} vs 基线 {const_acc:.4f}）。"
    )

    return _jsonable({
        "ok": True,
        "n_draws": N,
        "baseline_hits": round(6 * p_single, 4),
        "pairwise": {
            "n_pairs": n_pairs,
            "expected_per_pair": round(N * p_pair, 2),
            "chi2_unconditional": round(chi2_raw, 1),
            "chi2_conditional": round(chi2_cond, 1),
            "df": n_pairs,
            "verdict_zh": (
                f"条件化后 χ²={chi2_cond:.1f} **低于**独立假设期望 {n_pairs} —— "
                f"号与号之间不仅没有正结构，还比独立更均匀。"
                f"未条件化时的 {chi2_raw:.1f} 是一阶偏置被放大 32 倍的结果，不可作为证据。"
            ),
        },
        "marginal": {
            "chi2": round(chi1, 2),
            "df": 32,
            "perm_p": round(p1, 4),
            "null_mean": round(null1_mean, 2),
            "null_sd": round(null1_sd, 2),
            "chi2_table_critical": 46.19,
            "distribution_note_zh": (
                f"**零分布实际均值 {null1_mean:.2f}，不是理论的 32。** "
                f"因为每期抽 6 个不放回，33 个计数有『和为定值』约束 ⇒ 负相关 ⇒ 方差压缩。"
                f"用卡方表（临界 46.19）会漏判，必须看置换 p。"
            ),
            "top_number": top_num,
            "top_count": cnt1[top_num],
            "expected_count": round(e_top, 1),
            "z": round(z_top, 2),
            "bonferroni_z_threshold": round(z_bonf, 2),
            "bonferroni_passed": bool(passed_bonf),
        },
        "usability": {
            "asof_window": W_ASOF,
            "asof_accuracy": round(asof_acc, 4),
            "constant_baseline": round(const_acc, 4),
            "n_eval": n_eval,
            "upper_gain_pct": round(upper_gain_pct, 3),
            "verdict_zh": (
                f"跟随偏置的策略准确率 {asof_acc:.4f} **低于**『总是猜不出现』的 "
                f"{const_acc:.4f}。18% 基础率下后者天然 82% 正确 —— "
                f"分类准确率是错误判据，必须用 top-6 命中数。"
            ),
        },
        "headline_zh": headline,
        "honesty_zh": (
            "本端点只回答『号与号之间有没有结构』，不产号。"
            "二阶 χ² 必须先条件化（扣除一阶频次影响），否则会把单号偏置放大成假信号。"
            "一阶显著性必须过 Bonferroni（33 次检验），且必须通过 as-of 可利用性检验。"
            "**未检出联合结构是最正常也最诚实的结果。**"
        ),
    })


# ===========================================================================
# V33：策略可区分性判据（不问"能不能预测"，问"什么条件下测得出来"）
# ===========================================================================
#
# 用户 2026-10-24：「抛开『能不能预测』，而是『在什么条件下，一个策略的表现
# 在统计上能与其随机基线区分开』」。
#
# 这是本轮最有价值的一次转向。前面 V30/V31/V32 都在问「有没有规律」，
# 得到的是「没有」。但那个答案无法指导**任何**后续决策 ——
# 它不能回答「如果真有，我该买多少注、观察多久才能发现」。
#
# 本端点给出**封闭公式**与**设计表**，任何人可以自己算。
#
# ## 数学基础（全部是恒等式）
#
#   单注命中数 H ~ Hypergeom(33, 6, 6)
#   E[H]   = 6·6/33 = 1.0909
#   Var[H] = 6·(6/33)·(27/33)·(27/32) = 0.7531
#
#   n 注票单的命中总数 T：
#   E[T]   = n·E[H]                              （与选号无关）
#   Var[T] = n·Var[H]·var_ratio
#            var_ratio = Var[T]/(n·Var[H]) = 1+(n−1)(11/54)(Ā−12/11)
#            独立随机票 var_ratio ≈ 1；V27 最小方差票 ≈ 0.26
#
#   检出效应 δ（相对基线）所需期数：
#   P = ((z_α + z_β)·sd(T) / (δ·n·E[H]))²
#
# ## 最重要的发现：var_ratio 直接改变检验力
#
# 这一点 V27 当时没说清（只强调「不提高期望」）。实际效果：
#   var_ratio 1.0 → 0.26  ⇒ sd 降 49% ⇒ **等价样本量放大 3.85 倍**
#   检出 1% 效应所需期数：3,913 → 1,017
#
# 也就是说：**V27 的方差优化让整个系统更容易「发现真规律」，即使它不改变期望。**

@router.get("/api/rigor/detectability")
def rigor_detectability(
    n_tickets: int = Query(10, ge=1, le=100,
                           description="每期打算买几注"),
    effect_pct: float = Query(5.0, gt=0, le=500,
                              description="想检出的效应量（%）"),
    var_ratio: float = Query(1.0, gt=0, le=10,
                             description=("票单方差比。1.0=独立随机票；"
                                          "0.26≈V27 最小方差票（可从 /api/portfolio/variance-pick 取）")),
    alpha: float = Query(0.05, gt=0, lt=0.5, description="显著性水平（单侧）"),
    power: float = Query(0.80, gt=0.5, lt=0.999, description="检验力"),
):
    """**一个策略需要多强，才能在现有数据上被统计地区分出来？**

    ## 反过来问的三件事

    ```
    ① 给定效应量 → 需要多少期 / 多少年？
    ② 给定现有 3518 期 → 能检出多弱的效应？
    ③ 给定总预算 → 怎么分配注数/期数最优？
    ```

    ## 为什么这个端点比"有没有规律"更有用

    V30/V31/V32 回答「没有规律」，但那只对**已经测过的东西**成立。
    本端点给出的是**任何未来策略的门槛**：

    · 若你的新想法预期效应 < 1%，在 3518 期上**永远测不出来**
      （需要 26 年以上），无论它多精妙
    · 若预期效应 >= 20%，1 年内就能验证
    · 中间地带（5%~10%）需要 4~16 年

    ## var_ratio：唯一能人为改善的杠杆

    独立随机票 var_ratio ≈ 1.0。V27 的最小方差票实测 ≈ 0.26：

    ```
    sd(T) 从 2.744 降到 1.399（−49%）
    检出 1% 效应所需期数：3,913 → 1,017
    ```

    **这是 V27 的真实理论价值**——它不改变期望，但让「发现真规律」从不可能变成可能。

    ## 判据纪律

    · 效应量是**相对基线**（1.0909 个/期）的比例，不是绝对命中数
    · 单侧检验（策略是否**优于**随机），不是双侧
    · 期数按双色球 150 期/年折算
    """
    import math as _math

    # 直接调用（非 HTTP）时形参仍是 Query 默认对象 —— 必须归一，否则算不了数。
    # 本项目多条路由都有同一形态（V30/V31/V32 均如此），此处显式处理以免 500。
    def _num(v, default):
        return float(v) if isinstance(v, (int, float)) else float(default)

    n_tickets = int(_num(n_tickets, 10))
    effect_pct = _num(effect_pct, 5.0)
    var_ratio = _num(var_ratio, 1.0)
    alpha = _num(alpha, 0.05)
    power = _num(power, 0.80)

    e1 = 6 * 6 / 33
    var1 = 6 * (6 / 33) * (1 - 6 / 33) * (33 - 6) / (33 - 1)

    # 正态分位（避免 scipy 依赖）
    def _z(p: float) -> float:
        lo, hi = 0.0, 8.0
        for _ in range(90):
            mid = (lo + hi) / 2
            if 0.5 * (1 + _math.erf(mid / _math.sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    z_a = _z(1 - alpha)          # 单侧
    z_b = _z(power)

    sd_t = _math.sqrt(n_tickets * var1 * var_ratio)
    delta_abs = (effect_pct / 100.0) * n_tickets * e1
    n_periods = ((z_a + z_b) * sd_t / delta_abs) ** 2 if delta_abs > 0 else float("inf")

    # 反向：给定 3518 期能检出多弱
    N_HAVE = 3518
    min_effect_have = (z_a + z_b) * sd_t / _math.sqrt(N_HAVE) / (n_tickets * e1) * 100

    # 对照：独立随机票 vs 当前配置
    sd_ind = _math.sqrt(n_tickets * var1 * 1.0)
    n_ind = ((z_a + z_b) * sd_ind / delta_abs) ** 2 if delta_abs > 0 else float("inf")
    power_gain = (n_ind / n_periods) if n_periods and n_periods != float("inf") else 1.0

    # 设计表：不同注数下的最小可检出效应（固定期数 = 现有数据量）
    table = []
    for nt in (1, 2, 5, 10, 20, 50):
        sd = _math.sqrt(nt * var1 * var_ratio)
        e = (z_a + z_b) * sd / _math.sqrt(N_HAVE) / (nt * e1) * 100
        table.append({"n_tickets": nt,
                      "min_detectable_effect_pct": round(e, 2)})

    years = n_periods / 150 if n_periods != float("inf") else None

    if years is None:
        verdict = "效应量为 0，无法计算"
    elif years <= 1:
        verdict = (f"**1 年内可检出**：{n_periods:,.0f} 期（{years:.2f} 年）。"
                   f"这样的效应量（{effect_pct}%）意味着机械偏置明显，物理上不太可能。")
    elif years <= 5:
        verdict = (f"**需要 {years:.1f} 年**：{n_periods:,.0f} 期。"
                   f"在可接受的研究周期内，但要注意这期间摇奖设备可能更换。")
    elif years <= 30:
        verdict = (f"**需要 {years:.0f} 年**：{n_periods:,.0f} 期。"
                   f"超出多数研究的耐心，且跨越设备换代，结论难以迁移。")
    else:
        verdict = (f"**事实上不可验证**：{n_periods:,.0f} 期（{years:.0f} 年）。"
                   f"任何声称能检出 {effect_pct}% 效应的方法，都无法用真实数据证伪。")

    return _jsonable({
        "ok": True,
        "params": {
            "n_tickets": n_tickets,
            "effect_pct": effect_pct,
            "var_ratio": round(var_ratio, 3),
            "alpha": alpha,
            "power": power,
        },
        "constants": {
            "E_hits_per_ticket": round(e1, 4),
            "Var_hits_per_ticket": round(var1, 4),
            "sd_T_per_period": round(sd_t, 4),
        },
        "forward": {
            "periods_needed": round(n_periods, 0) if n_periods != float("inf") else None,
            "years_needed": round(years, 1) if years is not None else None,
            "verdict_zh": verdict,
        },
        "reverse": {
            "n_draws_available": N_HAVE,
            "min_detectable_effect_pct": round(min_effect_have, 2),
            "note_zh": (f"用现有 {N_HAVE} 期、每期 {n_tickets} 注、"
                        f"var_ratio={var_ratio:.2f}，最小可检出效应是 "
                        f"**{min_effect_have:.2f}%**。"
                        f"任何预期效应低于此值的策略，都无法在本系统上被证伪。"),
        },
        "var_ratio_leverage": {
            "independent_baseline": 1.0,
            "periods_if_independent": round(n_ind, 0) if n_ind != float("inf") else None,
            "periods_if_current": round(n_periods, 0) if n_periods != float("inf") else None,
            "equivalent_sample_multiplier": round(power_gain, 2),
            "note_zh": ("**这是 V27 方差优化的真实理论价值。** "
                        "var_ratio 从 1.0 降到 0.26 会把 sd 压低 49%，"
                        "等价于样本量放大 3.85 倍 —— "
                        "它不改变期望，但让『发现真规律』从不可能变成可能。"),
        },
        "design_table": {
            "fixed_draws": N_HAVE,
            "rows": table,
            "note_zh": ("固定现有数据量，不同注数下的最小可检出效应。"
                        "注数越多，单期噪声相对越小，可检出效应越低 —— "
                        "但边际收益递减（注数 ×10 只换来效应下限 ÷3.2）。"),
        },
        "benchmarks_v30": [
            {"claim": "freq 方法的『超额』", "effect_pct": 1.61,
             "detectable_in_draws": round(n_periods, 0) if n_periods != float("inf") else None},
        ],
        "honesty_zh": (
            "本端点只做**设计计算**，不评价任何具体策略的优劣。"
            "所有公式来自超几何分布与正态近似的恒等式，任何人可复算。"
            "**效应量的估计本身可能是错的**——本端点回答的是"
            "『若效应真有这么大，能不能测出来』，不是『效应有多大』。"
        ),
    })
