"""票单组合数学域路由（V4：期望值 / 分布）。

P2 模块化：从 `api_app.py` 的 V4 域切出。

## 这两个端点回答的是「票长什么样」，不是「能中多少」

组合期望值恒等于 `注数 × 2 元 × (1-返奖率)`，与号码选择无关 ——
这是本项目反复强调的诚实边界（LEAD_VERDICT §8.1）。
`/api/portfolio/expected-value` 存在的意义，是**让用户自己算一遍并看到它不随选号改变**。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import config, db
from .jsonutil import jsonable as _jsonable

router = APIRouter()

def _hypergeom(k: int, K: int, n: int, N: int) -> float:
    """超几何分布概率：N 个球中 K 个命中类，抽 n 个恰有 k 个命中类。"""
    from math import comb
    if k < 0 or k > K or (n - k) < 0 or (n - k) > (N - K):
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


def ticket_expected_prize() -> Dict:
    """单注期望奖金（元）与各奖级贡献拆解。

    按官方奖级规则与均匀随机假设精确计算：
      · 红中 6 + 蓝中 1 → 一等；红 6 蓝 0 → 二等；红 5 蓝 1 → 三等；
        红 5 蓝 0 或 红 4 蓝 1 → 四等；红 4 蓝 0 或 红 3 蓝 1 → 五等；蓝 1 → 六等。
      · 一等奖为浮动奖金，这里取 evaluate.PRIZE_CASH 的近似值并明确标注为估算。
    该值与「系统选什么号」**无关**——这正是要传达给用户的结论。
    """
    from . import evaluate
    N, K_red, n_red = 33, 6, 6
    p_red = {k: _hypergeom(k, K_red, n_red, N) for k in range(7)}
    p_blue_hit, p_blue_miss = 1 / 16, 15 / 16

    # 奖级概率（P(红=k) × P(蓝)）
    levels = {
        1: p_red[6] * p_blue_hit,
        2: p_red[6] * p_blue_miss,
        3: p_red[5] * p_blue_hit,
        4: p_red[5] * p_blue_miss + p_red[4] * p_blue_hit,
        5: p_red[4] * p_blue_miss + p_red[3] * p_blue_hit,
        6: (p_red[2] * p_blue_hit + p_red[1] * p_blue_hit + p_red[0] * p_blue_hit),
    }
    ev = 0.0
    breakdown = []
    for lvl in sorted(levels):
        p = levels[lvl]
        cash = float(evaluate.PRIZE_CASH.get(lvl, 0.0))
        contrib = p * cash
        ev += contrib
        breakdown.append({
            "level": lvl,
            "name": evaluate.PRIZE_NAME.get(lvl, f"{lvl} 等奖"),
            "probability": p,
            "prize_cash": cash,
            "expected_contribution": contrib,
            "prize_is_estimate": lvl in (1, 2),   # 一/二等奖浮动，取近似值
        })
    return {
        "expected_prize_per_ticket": ev,
        "hit_any_prize_probability": sum(levels.values()),
        "breakdown": breakdown,
    }


@router.get("/api/portfolio/expected-value")
def portfolio_expected_value(limit: int = Query(1, ge=1, le=20)):
    """当前推荐注组合的期望奖金分析（期望值为负是彩票的数学事实）。

    返回每注期望奖金、总投入、期望回报与 ROI；用于把一个「看起来能中奖」的
    组合摆回数学现实：长期期望回报率显著小于 1，且与选号方法无关。
    """
    tickets: List[Dict] = db.load_last_predictions() or []
    issue = None
    if tickets:
        issue = tickets[0].get("issue")
    elif hasattr(db, "recent_prediction_issues"):
        for iss in db.recent_prediction_issues(limit):
            got = db.load_predictions(iss)
            if got:
                tickets, issue = got, iss
                break

    price = config.TICKET_PRICE
    n_tickets = len(tickets)
    cost = price * n_tickets

    ev = ticket_expected_prize()
    ev_per_ticket = float(ev["expected_prize_per_ticket"])

    return {
        "ok": True,
        "available": True,
        "issue": issue,
        "n_tickets": n_tickets,
        "has_tickets": bool(n_tickets),
        "ticket_price": price,
        "total_cost": round(cost, 2),
        "expected_return_per_ticket": round(ev_per_ticket, 4),
        "expected_return_total": round(ev_per_ticket * n_tickets, 4),
        "expected_roi": round((ev_per_ticket - price) / price, 4),
        "hit_any_prize_probability": ev["hit_any_prize_probability"],
        "breakdown": ev["breakdown"],
        "sample_note": ("期望值按固定奖级与官方概率精确计算，与本系统历史命中**无关**；"
                        "彩票是负期望游戏，长期必亏是数学结论而非系统缺陷。"),
        "honesty": "双色球为独立随机事件，本系统不承诺中奖率，也不改变期望值。",
    }


# ==========================================================================
# V10-10：前端三卡的数据接口（票单分布 / 前置注册 / ML 诊断 / LLM 对照臂）
#
# 【序列化纪律】以下所有端点返回体**不得含 numpy 标量/数组**（``np.float64``、
# ``np.int64``、``np.ndarray`` 都不是 JSON 可序列化类型，FastAPI 默认编码器会抛
# ``TypeError: Object of type float64 is not JSON serializable``）。底层统计函数
# （``ticket_distribution`` / ``method_breakdown`` / ``battle`` / ``ml_model``）
# 虽然已尽量返回原生类型，但新增字段随时可能漏 —— 故统一过一遍
# ``_jsonable()`` 做深拷贝+类型归一，**不信任**上游"应该是原生类型"。
# ==========================================================================

@router.get("/api/portfolio/distribution")
def portfolio_distribution(issues: int = Query(1, ge=1, le=20),
                           exact: bool = Query(True)):
    """V10-10「🧾 票单分布」卡：当前推荐注组合的解析分布统计。

    数据源唯一：``engine.ticket_distribution()``（V10-03）。无预测时用固定示例票
    （2 注不重复号）以便前端永远能渲染一张完整的卡，但会明确标 ``source="sample"``。

    ``exact=True``（默认）：P(≥r红) 走 C(33,6) 全枚举（诚实口径，n=10 约 0.1~0.2s）；
    ``exact=False``：iid 近似（热路径/慢机器），返回体里 ``p_at_least_r_method``
    会变成 ``"iid_approx"``，前端据此提示"近似值不可当结论"。

    **诚实边界**：改变票单分布（冗余度/方差比）只改变命中数的**波动结构**，
    **不改变期望值** —— ``ev_ratio_sixth`` / ``ev_ratio_full`` 与选号无关，恒为常数。
    """
    from . import engine as E
    tickets: List[Dict] = db.load_last_predictions() or []
    issue = None
    source = "prediction"
    if tickets:
        issue = tickets[0].get("issue")
    else:
        # 无预测时的示例票：只为了让卡片有内容，不冒充真实推荐
        source = "sample"
        tickets = [
            {"reds": [1, 2, 3, 4, 5, 6], "blue": 1, "method": "sample"},
            {"reds": [7, 8, 9, 10, 11, 12], "blue": 2, "method": "sample"},
        ]
    try:
        dist = E.ticket_distribution(tickets, with_distribution_exact=bool(exact))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": f"票单分布计算失败: {exc}"},
                            status_code=400)
    return _jsonable({
        "ok": True,
        "issue": issue,
        "source": source,
        "source_zh": ("当前推荐注" if source == "prediction" else "示例票（暂无预测）"),
        "exact": bool(exact),
        "distribution": dist,
        # 这三条是"诚实边界"，前端必须原样展示，不得省略：
        "honesty": {
            "distribution_changes_ev": False,
            "note_zh": ("改变票单分布（冗余度 / 方差 / 尾部概率）只改变命中数与奖金的"
                        "**波动结构**，不改变期望值：任何组合的返奖比都恒等于"
                        "ev_ratio_sixth / ev_ratio_full，与选号方法无关。"),
            "prize_assumption": dist.get("prize_assumption"),
        },
        "note_zh": "分布统计是组合结构事实，不是中奖预测；双色球为独立随机事件。",
    })


# ---------------------------------------------------------------------------
# V14-B2 / B3：大众偏好度 + 蓝球覆盖（两个新端点）
#
# 这两个端点是「A+B」方向的产品落点：
#   · popularity：预测「多少人会选这种号」（人群行为可预测，号码不可预测）
#   · blue-coverage：蓝球去重覆盖（组合恒等式的真实概率增益）
#
# 两者都**强制**携带披露，且都不出现在出号门控里 —— 它们是"看懂自己的票"，
# 不是"预测下一期"。
# ---------------------------------------------------------------------------

#: 请求路径等待模型的最长时间（秒）。
#:
#: 取值理由：预热通常 6~7s，2 秒足以覆盖"预热刚好在收尾"的常见情形，
#: 省掉一次不必要的 503；而一旦超时就立刻放弃，**宁可显示"未就绪"
#: 也不让用户为一个后台任务排队**。前端有对应降级文案，不会白屏。
_POPULARITY_WAIT_SECONDS = 2.0

#: 「预热中」时告诉客户端的重试间隔（秒），同时写进 `Retry-After` 头。
#: 供前端显示「约 5 秒后就绪」，而不是让用户干等或以为功能坏了。
#: 预热中时给客户端的**保守兜底**重试间隔（秒），同时写进 `Retry-After` 头。
#:
#: 实际值走 `popularity.suggest_retry_after()` 的**动态估值**（基于上次实测耗时
#: + 当前已耗时下界，夹到 [5,30]）。这个常量是「拿不到动态值」时的兜底，
#: 也是文案区间的上界来源。
#:
#: 为什么取 30 而不是 5（Lead 裁决，2026-10-21 实测）：
#:   fit 耗时实测区间 6.06s（容器单核）/ 10.3s / 12.64s（争 CPU 时 40s+）/ 38.55s（最差）。
#:   旧值 5 相对实测低估 2.6~7.7 倍 —— 用户照提示 5 秒后刷新仍是 503，
#:   再刷还是 503。被消耗的是用户对披露的信任，而这正是本项目整轮在挣的东西。
#:   30 = 常态上界之上、极端之下（20 不够，实测到过 38.55s）。
_POPULARITY_RETRY_AFTER_SECONDS = 30


@router.get("/api/portfolio/popularity")
def portfolio_popularity(
    reds: str = Query(..., description="逗号分隔的 6 个红球，如 1,2,3,4,5,6"),
    sales: float = Query(
        0.0,
        description=("预计本期销量（元）。⚠️ 当前**不参与档位判定**：档位按「每亿元"
                     "销量注数」的标准化分位标定，与本期销量无关。该参数仅为未来"
                     "扩展保留，传任意值（含负数/极大值/inf/nan）都不改变返回结果。"),
    ),
) -> Dict:
    """查询该号码形态的「大众程度」档位。

    ## 它说的和它不说的话

    **说**：这种形态在历史上被多少彩民选中（低/中/高三档）。
    **不说**：下期会开什么号；会不会中奖。

    实测效应量：形态对头奖注数（`p1`）的增量 R² ≈ 0.008（0.8%），
    置换 p ≈ 0.005 —— 真实但极小，**只能排序，不能定值**。

    ## 参数边界（非法输入一律 400，不落到 500）

    `reds` 必须是 6 个 1~33 内互不相同的整数；非整数、个数不符、重复、越界 → 400。
    中文逗号与前后空格会被宽容规范化。

    ## `sales` 是空转参数（重要，别被签名骗了）

    `PopularityModel.describe()` **不使用 `sales` 的值**：档位在标准化空间上按分位
    标定，与本期销量无关。所以 `sales` 传 -5、1e308、inf、nan 都不会报错、也都不会
    改变结果 —— 这既是"不会 500"的原因，也是一个**陷阱**：调用方不要以为传了销量
    档位会更准。详见 `popularity.PopularityModel.describe` 内的注释。

    ## 模型未就绪 → 503

    `popularity.get_model()` 内部吞掉拟合异常并返回未拟合对象（不会抛），
    因此这里靠 `describe()` 的 `ok=False` 判定，返回 **503** 而非 500。
    """
    try:
        parts = [int(x) for x in str(reds).replace("，", ",").split(",") if x.strip()]
    except ValueError:
        return JSONResponse({"ok": False, "error": "reds 必须是逗号分隔的整数"}, status_code=400)
    if len(parts) != 6:
        return JSONResponse({"ok": False, "error": f"需要 6 个红球，收到 {len(parts)} 个"},
                            status_code=400)
    if len(set(parts)) != 6 or not all(1 <= x <= 33 for x in parts):
        return JSONResponse({"ok": False, "error": "红球必须是 1~33 内互不相同的 6 个数"},
                            status_code=400)

    from . import popularity as POP
    # 请求路径用**带超时**的取法：预热线程正在拟合时，最多等 2 秒，
    # 超时立刻返回未拟合对象走 503（前端有"模型未就绪"降级文案）。
    # 绝不用 `get_model()` 无超时持锁 —— 那会让用户为后台任务排队，
    # 是 Product 指出的「看起来加了预热、实际一点没省」的最坏形态。
    model = POP.get_model_for_request(timeout=_POPULARITY_WAIT_SECONDS)
    out = model.describe(parts, sales if sales else None)
    if not out.get("ok"):
        # ===== 区分「预热中」与「样本不足」=====
        #
        # 这两件事**用户该做的事完全不同**，合并成一句会误导：
        #   · 预热中（约 5~20s）：刷新一下就有 —— 说成"数据不足"会让用户以为功能坏了；
        #   · 样本不足（n_train < 300）：短期无解，只能等更多期开奖。
        # 这与 D-07「一个 usable 两个意思」同类：不是数字错，是用户无法区分该做什么。
        #
        # 用 `is_warming()` 而不是 `_CACHE is None`：超时返回的未拟合对象**不写 _CACHE**，
        # 所以 `_CACHE is None` 在两种情况下都成立，区分不出来。
        # `describe()` 现在会回传 `unfitted_reason`（warming / insufficient_data /
        # fit_failed），端点据此给对文案，而不是压成一句「数据不足或尚未拟合」。
        # 优先级：**实时**的 is_warming() 高于 describe() 里存的原因。
        # 理由：如果此刻正有拟合在跑，「预热中」就是当下最准确的原因；
        # 而 `unfitted_reason` 可能是上一次失败留下的历史值（例如先失败过
        # insufficient_data，之后预热重试又启动了）。
        # ===== 原因判定优先级（从高到低）=====
        #   ① 实时 is_warming()  —— 万一有人手动调了 `get_model()`，那是真在拟合
        #   ② warming_disabled   —— 预热被配置关闭，本进程内**永远不会**就绪
        #   ③ describe() 给的历史原因 —— warming / insufficient_data / fit_failed
        #
        # 为什么需要 ②：`LOTT_POPULARITY_WARMUP=0` 时，重试循环所在的
        # `_popularity_warmup()` 整段被同一开关门控、请求路径又不自 fit
        # ⇒ 模型永不就绪。此时若还说「稍候会自动重试」就是**假承诺**：
        # 用户会一直刷新一件永远不回来的事。
        # 注意 ① 必须高于 ②：预热关了但有人手动在拟合时，说「正在拟合」是真的。
        _live_warming = POP.is_warming()
        if _live_warming:
            why = "warming"
        else:
            why = out.get("unfitted_reason") or "unknown"
            if why in ("warming", "unknown") and not POP.warmup_enabled():
                why = "warming_disabled"

        if why == "warming":
            # **动态**估值（Lead 裁决）：基于上次实测 fit 耗时，夹到 [5,30]。
            # 写死 5s 在慢负载下会让用户反复撞 503；写死 20s 在快机器上让用户白等。
            retry_after = POP.suggest_retry_after()
            # ===== 文案不插值（Product 指出的坑）=====
            #
            # 旧写法 `"...约 %d 秒后自动可用" % retry_after` 会把**点值**印给用户，
            # 而真实区间是 6~40s —— 等于把「点值」又写死了一遍，白改。
            # 现在：**文案给人（给区间），字段给机器（给保守上界）**，两件事分开。
            # 将来调常量不会牵动文案，也就不会再出现「文案说 30、实际 38」这种漂移。
            return JSONResponse(
                {"ok": False,
                 "reason": "model_warming",
                 "unfitted_reason": "warming",
                 "retry_after": retry_after,
                 "error": "模型未就绪：正在拟合（通常 10~30 秒），稍后刷新即可"},
                status_code=503,
                headers={"Retry-After": str(retry_after)})

        if why == "insufficient_data":
            return JSONResponse(
                {"ok": False,
                 "reason": "insufficient_data",
                 "unfitted_reason": "insufficient_data",
                 "retry_after": None,
                 "error": ("模型未就绪：历史样本不足（需 ≥300 期含销量与头奖注数的"
                           "开奖），当前无法给出档位")},
                status_code=503)

        if why == "warming_disabled":
            # `retry_after: null` 是这条的重点：告诉前端「别等了」，
            # 而不是给一个假的 5 秒让用户反复刷新。
            return JSONResponse(
                {"ok": False,
                 # `reason` 直接给**细分值**：它是契约里最显眼的字段，
                 # 第三方调用方会只读它。给笼统的 model_not_ready 会让读者分叉错
                 # （Product 核 P3 时第一眼看的就是 reason）。
                 "reason": "warming_disabled",
                 # 兼容别名（下一版可删）：保留给已按旧契约写的调用方。
                 "unfitted_reason": "warming_disabled",
                 "retry_after": None,
                 "error": ("模型未就绪：后台预热已被配置关闭"
                           "（LOTT_POPULARITY_WARMUP=0），本进程内不会自动就绪")},
                status_code=503)

        if why == "retry_exhausted":
            # ===== 第三种「假承诺」（2026-10-21 实测确认）=====
            #
            # 预热循环 `for attempt in range(1, 11)` 跑完就退出；
            # 但 `_LAST_UNFIT_REASON` 仍是 `fit_failed`，旧代码据此给出
            # `retry_after=30` + 「稍候会自动重试」—— **而不会再有任何重试**。
            # 用户会一直刷新一件永远不回来的事。
            #
            # 所以：`retry_after: null` + 明确说「已重试 N 次仍失败」。
            return JSONResponse(
                {"ok": False,
                 "reason": "retry_exhausted",
                 "unfitted_reason": "retry_exhausted",
                 "retry_after": None,
                 "error": ("模型未就绪：已自动重试多次仍失败，"
                           "本进程内不会再自动重试（需检查日志或重启服务）")},
                status_code=503)

        # fit_failed / unknown：模型未就绪但不是数据问题也不是预热 —— 需看日志。
        # 文案保留「未就绪 / 未拟合」字样，便于通用降级提示匹配。
        return JSONResponse(
            {"ok": False,
             # 兜底分支：`why` 可能是 fit_failed（细分、有信息量）或 unknown（无信息量）。
             # 原先一律写成笼统的 model_not_ready，导致「机器读的主字段比人读的字段更粗」。
             # 现在：有细分值就给细分值；`unknown` 本身对调用方无意义，回落到 model_not_ready。
             "reason": ("model_not_ready" if why == "unknown" else why),
             "unfitted_reason": why,
             "retry_after": _POPULARITY_RETRY_AFTER_SECONDS,
             "error": "模型未就绪（拟合未完成或失败），稍候会自动重试"},
            status_code=503,
            headers={"Retry-After": str(_POPULARITY_RETRY_AFTER_SECONDS)})
    out["reds"] = sorted(parts)
    return _jsonable(out)


@router.get("/api/portfolio/blue-coverage")
def portfolio_blue_coverage(
    n_tickets: int = Query(10, ge=1, le=100, description="打算买几注"),
    full_ev: bool = Query(True, description="是否附带整注返奖比（含浮动奖估算）"),
) -> Dict:
    """蓝球去重覆盖计划 + **三个必须同屏的数字**。

    ## 为什么这个端点要求「三个数字必须一起看」

    单看「中奖概率从 47.6% 提到 62.5%」是激励性的，但配上
    「成本 20 元、六等奖期望回收 3.13 元、返奖比恒为 15.625%」才是完整的真相。

    所以本端点**永远**同时返回概率、成本、期望回收与 `disclosure` 文本，
    不存在只取概率的路径。

    ## n > 16 时返奖比**不再恒定**（端点必须自己说清楚）

    恒等式「返奖比 = 5/32 = 15.625%，与注数无关」的前提是 `p = n/16`，
    **只在 n ≤ 16 成立**。蓝球池只有 16 个，n > 16 时 `p` 被封顶 1.0，
    而成本仍线性增长 ⇒ 返奖比被稀释：

        n=16 → 15.625%    n=17 → 14.71%    n=50 → 5.00%    n=100 → 2.50%

    因此返回体带 `coverage_saturated`（n > 16 时为 true），披露文本也随之
    改口为"覆盖已饱和"。**这一条是防"数字与文案自相矛盾"的守卫** ——
    若哪天有人把披露改回无条件的"恒为…与注数无关"，`tests/test_v14_portfolio_api.py`
    与 `tests/test_v14_blue_coverage.py` 都会失败。

    ## 参数边界

    `n_tickets` 经 `Query(ge=1, le=100)` 校验，HTTP 层越界返回 **422**
    （FastAPI 契约层拦截，不经函数体）。直接调用函数时 `n_tickets ≤ 0`
    会按 0 注处理（`optimize` 内 `max(0, n)`），返回完整的"0 注"披露而非崩溃。
    """
    from . import blue_coverage as BC
    full_ratio = None
    assumption = None
    #: 本端点自身的降级项（D-17）。**正常路径下必须为空** ——
    #: 否则会「永远报降级」，用户习惯性忽略，比静默更糟（判据 4）。
    ev_degraded: List[Dict] = []
    if full_ev:
        try:
            from . import engine
            # V14-A4 修正：`_single_ticket_ev` 的第一个参数必须是 **dict**。
            # 传整数会被静默忽略并回落到 PRIZE_ASSUMPTION_TABLE_A
            # （first=500万 / second=15万），这正是「同一返奖比两个值」的来源。
            # 这里显式传入实测中位口径（一等 684 万 / 二等 20.5 万，来自 3266 期
            # p1_amt / p2_amt），并把它作为 `prize_assumption` 一并返回 ——
            # 项目规则要求「一/二等奖为浮动奖，必须连假设表一起展示」。
            assumption = {"first": 6_840_000, "second": 205_000,
                          "label": "实测中位（3266 期 p1_amt/p2_amt）"}
            # ===== V14 起改用 `strict`（2026-10-21 裁决，别改回去）=====
            #
            # `_single_ticket_ev` 刻意提供两个六等奖口径（`engine.py:1770` 原话：
            # 「两者都给出，让口径可核对而不是二选一」）：
            #   · `published` —— V4 起对外发表值，六等奖取蓝球**边际**概率 pb=0.0625，
            #     与高奖级**重复计数**，系统性偏高约 0.9pp；
            #   · `strict`    —— 按 `evaluate.prize_level` 的**互斥**奖级计算，
            #     六等奖用 (p_r(0)+p_r(1)+p_r(2))·pb，语义正确。
            #
            # 为什么消费方必须选 strict（四条同时成立才定的）：
            #   ① 语义正确：一注只能归入**一个**奖级，published 的重复计数不成立；
            #   ② 与 Lead 红线表一致：52.29%（= 1.0458133/2，`popularity.py:42` 的出处）；
            #   ③ 与同页 popularity 卡的可见文案一致（「约 52.3% 回收」）——
            #      否则同一页面两张卡对同一个量给出 52.3% vs 53.2%，用户一除就发现；
            #   ④ 数值更低，符合本项目"宁可少说"的披露原则。
            #
            # ⚠️ **不要删 engine 里的 published 分支**：它承担"可复现 V4 历史发表值"
            # （docs/V10 §3.4.5 冻结的 ev_ratio_full≈0.45673）的能力，是刻意保留的
            # 可核对设计。错的是消费者各挑一个且无一致性检查，不是口径本身。
            ev = engine._single_ticket_ev(assumption, sixth="strict")
            full_ratio = float(ev["ev_ratio"])
            # `_single_ticket_ev` 返回的 `prize_assumption` 是**人读字符串**，
            # 不是 dict；这里两者都保留：dict 供结构化渲染，text 供披露文本。
            assumption = dict(assumption, text=str(ev.get("prize_assumption") or ""))
        except Exception as exc:     # noqa: BLE001 —— 取不到就只给六等奖口径
            full_ratio = None
            assumption = None
            # ===== D-17：不许静默，也不许只进日志 =====
            #
            # 降级为「只给六等奖口径」本身是**合理的能力受限**（不是崩溃），
            # 但用户必须知道**为什么缺少整注返奖比**，否则会以为这个数不存在。
            #
            # Lead 的要求分三层，缺一不可：
            #   ① 不能静默（旧代码 `except: pass`）；
            #   ② 不能只进日志（日志没人看）；
            #   ③ **必须进 `degraded_items`**，带 id + reason + how_to_restore。
            #
            # 这里的措辞刻意区分「能力受限」与「故障」：写明是「整注口径不可用」，
            # 而不是笼统的「发生错误」——因为六等奖口径仍然正常返回。
            ev_degraded = [{
                "id": "ev_ratio_full",
                "reason": (f"整注返奖比（含一/二等奖浮动奖）本次不可用："
                           f"{type(exc).__name__}: {exc}。"
                           f"六等奖口径（返奖比恒为 15.625%）不受影响，仍已正常返回。"),
                "how_to_restore": (
                    "检查 lottery/engine.py 的 _single_ticket_ev 是否可用，"
                    "以及奖金假设表（_PRIZE_ASSUMPTION）配置是否正确；"
                    "修复后重新请求本端点即可恢复。"),
            }]
    out = BC.optimize(n_tickets, full_ev_ratio=full_ratio,
                      prize_assumption=assumption)
    # 降级项在**端点层**汇编（`BC.optimize` 属 blue_coverage 的写域，不改它）。
    # 正常路径下 `ev_degraded` 为空 ⇒ 响应里不出现该项 ⇒ 「没异常时不误报」（判据 4）。
    out["degraded_items"] = ev_degraded
    # 显式标注六等奖口径 —— 今天为这一个数倒了三个人核算，
    # 加一个字段，后面任何人看一眼响应就知道它用的是哪个口径，不必读代码。
    # 取值只有 "strict"（本端点）与 "published"（引擎默认/历史发表值）。
    # full_ev=False（未请求整注口径）时没有 ev_ratio_full，此字段为 None ——
    # 不能填 "published"，那会让调用方以为拿到了一个实际的整注口径值。
    out["sixth_mode"] = "strict" if full_ratio is not None else None
    out["ok"] = True
    return _jsonable(out)


# ==========================================================================
# V14 能力清单端点：把"系统做不到什么"变成可核对的 API
#
# 【为什么需要这个端点】
# 用户曾因"同一系统两个 usable"怀疑系统造假 —— `idle_mining.usable`（证据项）与
# `patterns_runtime.usable_replayable`（可回放出号规律）是两个不同口径，数字不同
# 是**正确**的，但没有一处统一说明，用户只能看到两个互相矛盾的数字。
#
# 本端点的价值不是展示能力，而是**让用户自己看到系统不吹牛**：
#   · 每条结论必须带 `evidence`（实测数字 + 出处）与 `honest_note`；
#   · 明确区分 `value` 是「文档记录值」还是「本次实测值」，不许混用；
#   · 四条结论里有三条是"做不到/极弱"，只有一条是真实收益且立刻附成本与返奖比。
#
# 【绝对禁止】
# 不许出现营销措辞，不许把"能排序"说成"能预测"，不许省略负面证据。
# ==========================================================================

def _capability_items(degraded_sink: Optional[List[Dict]] = None) -> List[Dict]:
    """能力清单的四条结论（纯数据，便于单测直接比对）。

    `evidence` 里的每个数字都必须能在代码或 `docs/reviews/` 找到出处，
    `source` 字段写明出处；`measured_now=True` 表示该值是**本次进程实测**，
    否则是**文档记录值**（可能随数据更新而漂移，故标注口径）。
    """
    # 当前实测（本次进程）：popularity 模型的增量 R² / 置换 p。
    # 注意这与文档写的 +0.0077 / p=0.005 **不一致** —— 口径相同（扩展窗口时序
    # + 置换 200 次），但数据期数不同（文档基于 3266 期，当前库 3517 期）。
    # 如实并列，不挑一个好看的说。
    #: 本段自身的降级项（D-17 同类，2026-10-21 自查发现）。
    #:
    #: 缺陷：旧代码 `except: _r2_now, _p_now = None, None` 是**静默**的。
    #: 用户在 `popularity_rank` 的 evidence 里看到「模型未拟合，暂无实测」，
    #: 却**分不清**是「模型本来就没拟合」还是「读取模型时崩了」——
    #: 前者是数据不足（等更多期），后者是故障（要查日志）。
    #: 这与 D-17 修的 `ev_ratio_full`、以及 D-07 的 `usable` 两义同族：
    #: **不是数字错，是用户无法区分该做什么。**
    # `degraded_sink` 由端点传入，用于把本段的降级项汇编进最终响应的 `degraded_items`。
    cap_degraded: List[Dict] = degraded_sink if degraded_sink is not None else []
    try:
        from . import popularity as POP
        _m = POP.get_model()
        _r2_now = (round(float(_m.r2_increment), 5) if _m.fitted else None)
        _p_now = (round(float(_m.p_value), 4) if _m.fitted else None)
    except Exception as exc:     # noqa: BLE001 —— 取不到就不给"实测"字段
        _r2_now, _p_now = None, None
        cap_degraded.append({
            "id": "popularity_measured",
            "reason": (f"「大众偏好排序」的**本次实测值**读取失败："
                       f"{type(exc).__name__}: {exc}。"
                       f"该项的文档记录值（R²=+0.0077 / 置换 p=0.005）不受影响，"
                       f"仍已正常给出；结论本身不变（依然只是「能做但极弱」）。"),
            "how_to_restore": (
                "检查 lottery/popularity.py 的 get_model() 与数据库可用性；"
                f"若是预热未完成，稍后重试即可；修复后重新请求本端点。"),
        })

    return [
        {
            "id": "predict_numbers",
            "name_zh": "预测下一期号码",
            "verdict": "cannot",
            "verdict_zh": "做不到",
            "summary_zh": ("号码本身不可预测。系统全部分析的正确结论是"
                           "「本期开奖与历史无关」，不是「下一期会开什么」。"),
            "evidence": [
                {"claim_zh": "红球频次卡方检验",
                 "value": "χ²=44.84, df=32, p=0.0654",
                 "reading_zh": "不能在 α=0.05 下拒绝「均匀」。注意 p=0.065 仍 > 0.05，"
                               "**没有**达到显著；即使显著也只说明历史频次有起伏，"
                               "不构成下期预测力。",
                 "source": "docs/reviews/V14_STAT_REVIEW.md（q2d_chisq.py，全史实测）"},
                {"claim_zh": "条件熵：知道上期是否出现，能降多少不确定性",
                 "value": "H=5.0355 vs 5.0431，仅降 0.0076 bit",
                 "reading_zh": "降幅 0.15%，在噪声内 —— 上期开奖对下期几乎零信息。",
                 "source": "docs/reviews/V14_TEAM_VERDICT.md:19"},
                {"claim_zh": "最强候选在最大统计量零分布下的 p 值",
                 "value": "max e=28.34 → p=0.2995（4000 次零分布）",
                 "reading_zh": "纯随机数据中出现「至少一条这么强」的概率约 30%。"
                               "**候选看起来的强度是多重比较造成的假象**。",
                 "source": "docs/reviews/V14_TEAM_VERDICT.md:23"},
            ],
            "honest_note": (
                "本系统不会、也不能提高任何一注的中奖概率。单注头奖概率恒为 "
                "1/17,721,088。所有「按规律出号」的模块都受结构门控约束，"
                "它筛选的是**形态**，不是提高命中。"
            ),
        },
        {
            "id": "detect_bias",
            "name_zh": "检出现实中是否存在号码偏置",
            "verdict": "partially",
            "verdict_zh": "能检出 ≥+8%/号，检不出 +5%~+6%/号",
            "summary_zh": ("这是统计检验的灵敏度问题，不是算法缺陷。以现有 3516 期的"
                           "信息量，偏置约 +6% 以下注定检不出，会如实报 0 条证据项 —— "
                           "**那不是「系统证明没有偏置」，而是「系统分辨不了」。**"),
            "evidence": [
                {"claim_zh": "注入实验：人为注入偏置后能否检出",
                 "value": "+5% → e=13.44 漏报；+6% → e=53.94 漏报；"
                          "+8% → e=1.23×10⁴ 检出",
                 "reading_zh": "检出边界在 +6%~+8% 之间；把门槛从 2700 降到 960 "
                               "（降 2.8 倍）后边界**完全没有移动**。",
                 "source": "docs/reviews/V14_STAT_REVIEW.md:240（q4f_value_of_fix.py）"},
                {"claim_zh": "通过门槛所需的相对效应量",
                 "value": "V0 门槛 2700 需 delta=0.0683（+6.3%）；"
                          "V2 门槛 960 需 delta=0.0645（+5.9%）",
                 "reading_zh": "门槛降 2.8 倍，所需效应量只降 0.4 个百分点 —— "
                               "e 值对效应量是四次方级敏感。",
                 "source": "docs/reviews/V14_STAT_REVIEW.md:175"},
                {"claim_zh": "最强真实候选的实测效应",
                 "value": "delta=0.0496（相对 +4.5%），低于检出门槛 +5.9%",
                 "reading_zh": "所以生产环境报 0 条是**如实**结果，不是漏算。",
                 "source": "docs/reviews/V14_STAT_REVIEW.md:177"},
            ],
            "honest_note": (
                "「证据项 0 条」\u2260「数据无偏置」。当系统报 0 时，正确读法是"
                "「在现有样本量与 α=0.05 下无证据」，而不是「已证明随机」。"
                "若未来真有 +5% 级偏置，系统会继续报 0。"
            ),
        },
        {
            "id": "popularity_rank",
            "name_zh": "预判「这种号有多少人买」（大众偏好排序）",
            "verdict": "weak",
            "verdict_zh": "能做，但极弱（只能排序，不能定值）",
            "summary_zh": ("号码是随机的，但**选号的人不是**。形态对头奖注数 `p1` "
                           "的解释力真实存在，但只有约 1%，因此系统只输出"
                           "低/中/高**档位**，从不输出预测注数。"),
            "evidence": [
                {"claim_zh": "形态对 p1 的增量 R²（文档记录值，3266 期）",
                 "value": "+0.0077（0.77%）；3/5/8/10 折恒为 0.0077；"
                          "剔除上 1% 异常值后 +0.0186",
                 "reading_zh": "效应真实但极小。销量本身的 R²=+0.0205，"
                               "即 p1 主要由销量驱动而非形态。",
                 "measured_now": False,
                 "source": "lottery/popularity.py 模块 docstring（实测留档）"},
                {"claim_zh": "同一指标的**当前库实测值**（3517 期，本次进程）",
                 "value": (f"R²={_r2_now}，置换 p={_p_now}" if _r2_now is not None
                           else "模型未拟合，暂无实测"),
                 "reading_zh": ("与文档记录值不同属于**正常**：两者口径相同"
                                "（扩展窗口时序 + 置换），但数据期数不同。"
                                "这里如实并列，不挑一个更好看的。"),
                 "measured_now": True,
                 "source": "popularity.get_model() 运行时实测"},
                {"claim_zh": "置换检验",
                 "value": "p ≈ 0.005（文档）/" + (f"{_p_now}（当前实测）"
                                                if _p_now is not None else "—"),
                 "reading_zh": "显著 \u2260 有用：显著只说明「不是零」，"
                               "R²≈1% 说明「几乎预测不了」。",
                 "source": "同上"},
            ],
            "honest_note": (
                "0.77%~1.1% 的解释力只能给出粗略排序，**不能给出可用数值**。"
                "所以本系统只输出档位。即便用它避开热门号，"
                "实测期望增益仅 0.005~0.18 元/注（占 2 元成本的 0.3%~9%），"
                "且**中奖概率完全不变**，长期期望仍为负。"
            ),
        },
        {
            "id": "blue_coverage",
            "name_zh": "蓝球去重覆盖",
            "verdict": "real_but_no_ev_gain",
            "verdict_zh": "真实提高中奖**频率**，不提高期望收益",
            "summary_zh": ("这是系统里**唯一**数学上确凿可兑现的改进：把 n 注的蓝球取"
                           "互不相同，P(≥1 注中蓝) 从 1-(15/16)^n 变成 min(n,16)/16。"
                           "但它改变的是**中奖分布**，不是期望值。"),
            "evidence": [
                {"claim_zh": "10 注：至少中一次蓝球的概率",
                 "value": "随机重复 47.55% → 去重覆盖 62.50%（成本 20 元）",
                 "reading_zh": "提升是组合恒等式，全枚举可证，零造假风险。",
                 "source": "lottery/blue_coverage.py 模块 docstring 对照表"},
                {"claim_zh": "六等奖返奖比（n ≤ 16）",
                 "value": "恒为 15.625%，与注数无关",
                 "reading_zh": "这是「覆盖优化不改变期望」的直接证据："
                               "(5×n/16)/(2n) = 5/32，n 被完全约掉。",
                 "source": "blue_coverage.optimize 实测 n=1..16"},
                {"claim_zh": "n > 16 时返奖比会**下降**（覆盖饱和）",
                 "value": "n=17 → 14.71%；n=50 → 5.00%；n=100 → 2.50%",
                 "reading_zh": "蓝球池只有 16 个，超出部分必然重复："
                               "概率封顶 1.0 而成本继续线性涨。",
                 "source": "blue_coverage.optimize 实测；见 coverage_saturated 字段"},
                {"claim_zh": "整注（一~六等全计）返奖比",
                 "value": "约 53.2%（依赖奖金假设：一等 684 万 / 二等 20.5 万）",
                 "reading_zh": "一/二等奖是浮动奖，必须连假设一起看；"
                               "长期仍为负期望。",
                 "source": "/api/portfolio/blue-coverage 的 prize_assumption 字段"},
            ],
            "honest_note": (
                "覆盖优化提高的是**中奖频率（体感）**，不是期望收益。"
                f"同样 10 注，成本 20 元，六等奖期望回收仅 3.13 元。"
                "中头奖概率恒为 1/17,721,088，**不受选号或覆盖策略影响**。"
            ),
        },
    ]


@router.get("/api/portfolio/capabilities")
def portfolio_capabilities() -> Dict:
    """能力清单：系统「做不到什么」的逐条自证，带实测数字与出处。

    ## 这个端点为什么存在

    用户曾因**「同一系统两个 usable」**怀疑造假：

    * `idle_mining` 的 `rule_evidence`（也叫过 `usable`）= 统计上过阈的证据项条数；
    * `patterns_runtime` 的 `usable_replayable` = 能**可复现回放出号**的规律条数。

    两者数字不同是**正确**的（一个是"有证据"，一个是"能出号"），但如果没一处
    统一说明，用户只会看到两个打架的数字。本端点把口径摊开，并把
    "预测号码：做不到"这类结论明确写出来。

    ## 返回结构契约

    * `capabilities[*].verdict` ∈ `cannot` / `partially` / `weak` /
      `real_but_no_ev_gain` —— **只有最后一类是真收益，且必附"不提高期望"**；
    * `capabilities[*].evidence[*]` 每条都带 `source`（出处），
      带 `measured_now=True` 的是本次进程实测，否则是文档记录值；
    * 顶层 `disclaimer` 固定陈述「本系统不改变中奖概率与期望值」。

    ## 不做什么

    不做营销文案，不把"能排序"说成"能预测"，不省略任何负面证据，
    不使用「提高中奖率/提高命中率/更划算/稳赚」这类词。
    """
    # 传入 sink：让 `_capability_items` 里那处 `except` 的降级项
    # 能被汇编进响应的 `degraded_items`（否则它仍是静默的 —— D-17 同类）。
    _cap_degraded: List[Dict] = []
    items = _capability_items(degraded_sink=_cap_degraded)
    return _jsonable({
        "ok": True,
        "n_capabilities": len(items),
        "capabilities": items,
        "verdict_summary": {
            "cannot": 1,          # 预测号码
            "partially": 1,       # 检出偏置（有灵敏度下限）
            "weak": 1,            # 大众偏好排序（只能排序）
            "real_but_no_ev_gain": 1,   # 蓝球覆盖（提高频率不提高期望）
        },
        "disclaimer": (
            "本系统不改变任何一注的中奖概率，也不改变期望值。"
            "双色球为独立随机事件，长期期望为负（整注返奖比约 53.2%）。"
        ),
        # 降级项（D-17）：正常路径为空 —— 不误报。
        # 目前只有一项可能：`popularity_measured`（本次实测值读取失败）。
        "degraded_items": _cap_degraded,
        "degraded": bool(_cap_degraded),
        "usage_note_zh": (
            "读这个端点的正确方式：先看「做不到」和「极弱」的四条，"
            "再看唯一一条真实收益（蓝球覆盖）后面跟着的成本与返奖比。"
            "凡是本端点没写进 evidence 的数字，都不要当成结论。"
        ),
    })


# ===========================================================================
# V27：方差优化选号（`variance_picker` 的唯一读口）
# ===========================================================================
#
# 这个端点存在的理由：用户问「能不能用『降低方差』这个思路出号」。
# 答案是**能**，但要三方同屏——优化了什么、没优化什么、凭什么复核。
#
# 三个数字必须一起看（缺一就是误导）：
#   ① 方差比降到多少（真实增益，任何人都能用 ticket_distribution 复核）
#   ② 并集覆盖 / P(≥4红)（副产物，同样可复核）
#   ③ 期望收益**没变**（EV = n × ¥1.0458 恒等式）
#
# 若只给 ① 不给 ③，用户会以为「方差小 = 更划算」，那是假的。

@router.get("/api/portfolio/variance-pick")
def portfolio_variance_pick(
    n_tickets: int = Query(10, ge=1, le=50,
                           description="要买几注（1~50）"),
    seed: int = Query(42, ge=0, le=10 ** 6,
                      description="随机种子（同种子同结果，便于复核）"),
    iterations: Optional[int] = Query(
        None, ge=100, le=200000,
        description="退火迭代次数；默认 max(2000, 400×n_tickets)"),
    with_baseline: bool = Query(
        True, description="是否附带与「同样注数的均匀随机票单」的对照（约 0.3~2s）"),
    baseline_samples: int = Query(
        60, ge=10, le=300,
        description="随机对照的抽样次数（越多越稳，但越慢）"),
):
    """按**方差最小化**出号 —— 这是本系统唯一"能改进"的方向，但不是提高收益。

    ## 它优化的是什么

    命中总数 T 的方差。对 n 注票单有解析式（组合恒等式，不是拟合）：

        E[T]   = n · 12/11                       与选号**无关**
        Var[T] = n·σ₁² · [1 + (n−1)(11/54)(Ā − 12/11)]

    所以方差只取决于重号平均 Ā。最小化 Ā ⇒ 最小化方差。

    ## 实测对照（n=10，多组抽样平均）

        ┌──────────┬────────────┬───────────┬──────┐
        │ 策略     │ P(≥4红)    │ var_ratio │ 并集 │
        ├──────────┼────────────┼───────────┼──────┤
        │ 均匀随机 │ 0.047768   │ 1.0031    │ 28.8 │
        │ 最小方差 │ 0.048767   │ 0.2019    │ 33.0 │
        │ 变化     │ +2.09%     │ −79.9%    │ +4.2 │
        └──────────┴────────────┴───────────┴──────┘

    ## 必须同屏的诚实边界

    ✅ **方差降到约 20%**（命中总数更稳定）
    ✅ **P(≥1 注命中 ≥4 红) 提高约 1~2%**（并集扩大的副产物）
    ❌ **期望收益完全不变**：EV 恒为 n × ¥1.0458（52.29% 回收）
    ❌ **单注头奖概率恒为 1/17,721,088**，不因选号改变

    第 3 条是这个端点的核心：**方差小 ≠ 更划算**。
    方差的降低改善的是「结果的稳定性」，不是「期望」。

    ## 区别于其它端点

    · `/api/portfolio/distribution`：给**已买的**票做诊断（事后）
    · `/api/portfolio/variance-pick`：给**要买的**票做设计（事前，本端点）
    · `/api/predict`：走严格模式门控出号（那套判据针对"预测力"，与本端点无关）

    本端点**不进严格模式门控**——因为它不声称有预测力，它只做组合结构优化。
    把它塞进门控会造成语义混淆（门控问的是"有规律吗"，这里问的是"结构好不好"）。
    """
    from . import variance_picker as VP

    try:
        reds_list = VP.pick_min_variance(
            n_tickets, seed=seed, iterations=iterations)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not reds_list:
        return JSONResponse(
            {"ok": False,
             "error": ("n_tickets 必须 ≥1：请把注数改成 1~50 之间的整数后重试"
                       "（0 注无法生成任何票单）")},
            status_code=400)

    summary = VP.summarize(reds_list, exact=True)
    out = {
        "ok": True,
        "method": "min_var",
        "method_zh": "最小方差选号（组合结构优化，不预测号码）",
        "n_tickets": len(reds_list),
        "seed": seed,
        "iterations": iterations if iterations is not None else max(2000, 400 * n_tickets),
        "tickets": [{"reds": r, "blue": None} for r in reds_list],
        "summary": summary,
        "objective_zh": (
            "目标是最小化命中总数的方差（等价于最小化票间平均重号 Ā），"
            "并集覆盖作为平手判据。**不优化**期望收益——它无法被优化。"
        ),
        "honesty": {
            "ev_unchanged": True,
            "ev_unchanged_zh": (
                "期望收益与选号无关：EV 恒为 n × ¥1.0458（52.29% 回收）。"
                "方差降低改善的是结果稳定性，不是期望。"
            ),
            "no_prediction": True,
            "no_prediction_zh": (
                "本端点不预测下一期号码。单注头奖概率恒为 1/17,721,088，"
                "任何选号策略都不改变它。"
            ),
            "p_at_least_r_note_zh": (
                "P(≥1 注命中 ≥r 红) 的提升来自**并集覆盖扩大**，"
                "不是来自'选号更准'——它是副产物，不是优化目标。"
            ),
        },
        "how_to_verify_zh": (
            "拿返回的 tickets 调 POST /api/portfolio/distribution 或 "
            "engine.ticket_distribution()，var_ratio 与 p_at_least_r 应完全一致"
            "（同一套精确枚举口径）。"
        ),
    }
    if with_baseline:
        try:
            out["baseline"] = VP.compare_to_random(
                reds_list, n_samples=baseline_samples, seed=seed ^ 0x5A5A)
        except Exception as exc:      # noqa: BLE001 —— 对照失败不影响主结果
            # D-17：降级必须可见，不能静默省略 baseline 字段
            out["baseline"] = {
                "ok": False,
                "reason": "baseline_failed",
                "error_zh": (
                    f"随机对照计算失败（{type(exc).__name__}），"
                    "主结果（tickets/summary）不受影响；"
                    "可加 with_baseline=false 跳过对照后重试。"
                ),
            }
    return _jsonable(out)
