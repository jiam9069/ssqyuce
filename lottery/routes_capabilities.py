"""能力清单端点 `GET /api/capabilities`（D-04）。

## 这个端点与 `/api/portfolio/capabilities` 的分工

两者是**不同层级**，不是重复实现：

* `/api/portfolio/capabilities` —— 面向「票怎么选」的四条结论，
  带完整的长文披露，给人读；
* `/api/capabilities`（本文件） —— 面向**机器与验收**的结构化清单：
  `can` / `cannot` / `degraded` / `constants` 四段，其中 `constants` 的每一个
  数字都**在请求时从代码实时读取**，供 CI、前端与独立审计逐项比对。

`constants` 段是本端点存在的核心理由。项目历史上多次出现「文档写一个值、
代码算出另一个值」的漂移（例如奖金假设从表 A 的 500 万改成实测中位 684 万后，
返奖比从 45.67% 变成 53.19%）。写死字面量的清单无法发现这种漂移，
实时读取的清单会在漂移发生的瞬间暴露它。

## 诚实纪律

* `cannot` 段必须包含四条**硬否定**：不可预测号码、期望为负、六等奖返奖比恒定、
  蓝球覆盖只改频率不改期望；
* 不得出现「提高中奖率 / 提高命中率 / 更划算 / 稳赚」；
* `degraded` 段只报告**当前真实**处于降级状态的项，不预填假数据。

## 为什么 `degraded` 用 503 而不是 200

`degraded` 是**本清单的内容**（说明"哪一项当前不可用"），清单本身能正常产出时
仍返回 200 —— 这与 `/api/portfolio/popularity` 在模型未拟合时返回 503 不冲突：
那个端点的全部产出就是模型结论，模型没了它就无可交付；而能力清单的产出是
"陈述"，降级项恰恰是陈述的一部分。

若能力清单本身都算不出来（例如常量源模块导入失败），才整体 503 + degraded。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .jsonutil import jsonable as _jsonable

router = APIRouter()

#: 一等奖奖金假设（元）—— 与 `routes_portfolio.portfolio_blue_coverage` 保持一致。
#:
#: 取 3266 期 `p1_amt` / `p2_amt` 的实测中位，而非 V4 表 A 的 500 万/15 万。
#: 表的切换有据可查（git `db634e3`），此处只是复用同一口径，**不重新发明数字**。
_PRIZE_ASSUMPTION = {
    "first": 6_840_000,
    "second": 205_000,
    "label": "实测中位（3266 期 p1_amt/p2_amt）",
}


# ==========================================================================
# constants：全部实时读取，一个都不写死
# ==========================================================================

def _live_constants() -> Dict[str, Any]:
    """从代码实时读取五个锁定的硬数字。

    返回体带 `sources`（每个值来自哪个符号），这样一旦某项取不到，
    调用方看到的是 `None` + 原因，而不是一个看起来正常的错值。

    **绝不在这里写 `306` / `72350` / `0.05` 之类的字面量** —— 那会让本端点
    失去唯一的意义。数字必须来自函数调用或常量引用。
    """
    out: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    # α（名义显著性）
    try:
        from . import eprocess as EP
        out["alpha"] = float(EP.DEFAULT_ALPHA)
        sources["alpha"] = "lottery.eprocess.DEFAULT_ALPHA"
    except Exception as exc:                     # noqa: BLE001 —— 取不到就如实报告
        out["alpha"] = None
        errors["alpha"] = f"{type(exc).__name__}: {exc}"

    # 搜索空间（单因子 / 跨族交叉）
    try:
        from . import candidate_space as CS
        out["single"] = int(CS.total_single())
        sources["single"] = "lottery.candidate_space.total_single()"
    except Exception as exc:                     # noqa: BLE001
        out["single"] = None
        errors["single"] = f"{type(exc).__name__}: {exc}"

    try:
        from . import candidate_space as CS
        out["cross"] = int(CS.total_cross())
        sources["cross"] = "lottery.candidate_space.total_cross()"
    except Exception as exc:                     # noqa: BLE001
        out["cross"] = None
        errors["cross"] = f"{type(exc).__name__}: {exc}"

    # 单注全奖级返奖比（依赖奖金假设，假设值一并回传）
    #
    # ===== 口径必须与 `/api/portfolio/blue-coverage` 一致：`strict` =====
    #
    # 2026-10-21 齐交付实测：本端点原用 `published`（0.531925），而 D-01 要求前端
    # 展示 52.3%（strict 口径，0.522907）—— **同一系统两个返奖比**，正是本轮在清的病。
    #
    # 两个口径的区别（`engine._single_ticket_ev` 刻意双提供，不得删任一分支）：
    #   · `published` —— V4 起对外发表值，六等奖取蓝球**边际** pb，与高奖级
    #     **重复计数**，系统性偏高约 0.9pp；
    #   · `strict`    —— 按 `evaluate.prize_level` 的**互斥**奖级计算，语义正确。
    # 语义正确性要求一注只归入一个奖级 ⇒ 消费者统一走 `strict`。
    try:
        from . import engine
        ev = engine._single_ticket_ev(_PRIZE_ASSUMPTION, sixth="strict")
        out["ev_single"] = round(float(ev["ev_ratio"]), 6)
        # 这个字符串是给人读的**溯源说明**，必须与实际调用逐字一致 ——
        # 它写 'published' 而代码走 'strict' 就是在骗读者（齐交付特别指出）。
        sources["ev_single"] = ("lottery.engine._single_ticket_ev("
                                "first=6,840,000, second=205,000, sixth='strict')"
                                "['ev_ratio']")
        # 显式标注口径：任何人看一眼响应就知道是哪个模式，不必读代码。
        out["sixth_mode"] = "strict"
    except Exception as exc:                     # noqa: BLE001
        out["ev_single"] = None
        out["sixth_mode"] = None
        errors["ev_single"] = f"{type(exc).__name__}: {exc}"

    # 六等奖返奖比（恒 5×P(红≤2)/32 ≈ 14.723%，与注数无关；V36 修正旧值 5/32）
    try:
        from . import blue_coverage as BC
        out["ev_sixth"] = round(float(BC.optimize(1)["ev_ratio_sixth"]), 6)
        sources["ev_sixth"] = "lottery.blue_coverage.optimize(1)['ev_ratio_sixth']"
    except Exception as exc:                     # noqa: BLE001
        out["ev_sixth"] = None
        errors["ev_sixth"] = f"{type(exc).__name__}: {exc}"

    out["sources"] = sources
    out["prize_assumption"] = dict(_PRIZE_ASSUMPTION)
    out["derivation_zh"] = {
        "ev_sixth": ("六等奖要求蓝中 **且** 红≤2 ⇒ 单注概率 P(红≤2)/16=0.0588925；"
                     "n 注期望奖金 = n×该概率×5，成本 2n ⇒ 返奖比 = 5×P(红≤2)/32 ≈ 0.1472314，n 被约掉。"),
        "ev_single": "六奖级概率 × 奖金求和 ÷ 2 元成本；一/二等奖为浮动额，取实测中位。",
    }
    if errors:
        out["errors"] = errors
    return out


# ==========================================================================
# can / cannot / degraded
# ==========================================================================

def _can_items() -> List[Dict[str, Any]]:
    """**真的能做到**的事。每项必须附证据脚本路径（可复跑）。"""
    return [
        {
            "id": "blinded_audit",
            "title": "盲法自证：用置换与零分布检验「规律」是不是噪声",
            "evidence_script": "tests/test_p4_eprocess.py",
            "honest_bound": (
                "只能否定「有强规律」，不能证明「绝对无规律」；"
                "检出下限受样本量约束，效应量低于门槛时与噪声不可分。"
            ),
        },
        {
            "id": "replayable_rules",
            "title": "规律的生命周期管理：注册 / 回放 / 淘汰（usable_replayable）",
            "evidence_script": "tests/test_v14_naming_discipline.py",
            "honest_bound": (
                "「能回放」= 历史快照里可复现，**不等于**对下一期有效；"
                "当前生产快照 usable_replayable = 0，即没有可出号规律。"
            ),
        },
        {
            "id": "popularity_rank",
            "title": "大众偏好排序：估计「多少人会选这注号」的档位",
            "evidence_script": "tests/test_v14_popularity.py",
            "honest_bound": (
                "增量 R² 仅约 0.8%~1.1%，**只能排序不能定值**；"
                "而且它预测的是人群行为，不是开奖号码。"
            ),
        },
        {
            "id": "blue_coverage",
            "title": "蓝球去重覆盖：n 注把蓝球取互不相同",
            "evidence_script": "tests/test_v14_blue_coverage.py",
            "honest_bound": (
                "只提高「至少中一次蓝球」的频率（10 注 47.55% → 62.50%），"
                "**不改变期望收益**；n>16 后概率封顶 100%，再加注只涨成本。"
            ),
        },
        {
            "id": "ticket_distribution",
            "title": "票单分布解析：给定票组的命中数分布与尾部概率（C(33,6) 全枚举）",
            "evidence_script": "tests/test_v10_ticket_distribution.py",
            "honest_bound": (
                "描述的是「已买的这组票」的波动结构，与选号好坏无关，"
                "也不改变期望值。"
            ),
        },
    ]


def _cannot_items() -> List[Dict[str, Any]]:
    """**做不到**的事。这是本端点最该被读到的一段。"""
    return [
        {
            "id": "predict_numbers",
            "title": "预测下一期开奖号码",
            "why_not": (
                "双色球是独立随机事件，历史不影响未来。系统对该命题做了三重独立检验，"
                "全部给出否定结论：红球频次卡方检验 p=0.0654（不显著）；"
                "条件熵相对独立基线只降约 0.0076 bit；"
                "跨族搜索的最大统计量经置换校正后 p=0.2995。"
                "所谓「规律」都落在噪声范围内。"
            ),
            "evidence": (
                "p=0.0654（红球 χ²）；条件熵降 0.0076 bit；"
                "max-stat 校正后 p=0.2995（docs/reviews/V14_STAT_REVIEW.md）"
            ),
        },
        {
            "id": "positive_expectation",
            "title": "获得正期望收益（长期不亏）",
            "why_not": (
                "彩票按固定返奖设计，任何选号、覆盖或资金策略都无法改变单注返奖比。"
                "单注全奖级返奖比由 `constants.ev_single` 实时给出（约 53%），"
                "即每投入 2 元期望回收约 1.06 元。这是数学结论，不是系统缺陷，"
                "也不是可以「优化」掉的东西。"
            ),
            "evidence": "constants.ev_single（实时读取 engine._single_ticket_ev）；长期期望为负",
        },
        {
            "id": "raise_sixth_ratio",
            "title": "让六等奖返奖比高于 5/32（15.625%）",
            "why_not": (
                "六等奖只要求蓝球命中，每注独立 1/16，期望命中注数恒为 n/16，"
                "成本恒为 2n，两者相除 n 被约掉 —— 返奖比恒为 5/32，**与注数、"
                "与是否去重覆盖都无关**。任何宣称能抬高这个比值的设计都必然在骗人。"
            ),
            "evidence": "constants.ev_sixth（实时读取 blue_coverage.optimize）；恒等式 (n/16×5)/(2n)=5/32",
        },
        {
            "id": "coverage_changes_expectation",
            "title": "用蓝球覆盖改变期望回报率",
            "why_not": (
                "蓝球覆盖改变的是**中奖频率的分布**：10 注去重覆盖把「至少中一次蓝球」"
                "从 47.55% 抬到 62.50%，但成本 20 元、六等奖期望回收 3.125 元、"
                "返奖比仍恒为 15.625%。它让「多久中一次」变得好看，"
                "不让「赚不赚钱」发生变化。"
            ),
            "evidence": "10 注：47.55% → 62.50%，成本 20 元 / 期望回收 3.125 元 / 返奖比 15.625%",
        },
    ]


def _degraded_items() -> List[Dict[str, Any]]:
    """**当前真实**处于降级状态的项。不预填、不猜。"""
    items: List[Dict[str, Any]] = []

    # 1) 闲时挖掘停机（LOTT_IDLE_MINING=0 为默认）
    try:
        from . import idle_mining
        st = idle_mining.status()
        if not st.get("enabled"):
            items.append({
                "id": "idle_mining",
                "reason": (
                    "闲时挖掘当前关闭（LOTT_IDLE_MINING 未开启）。"
                    f"已判定候选 {st.get('checked', 0)} 个，"
                    f"证据项 {st.get('rule_evidence_count', 0)} 条。"
                    "关闭是有意的默认值：挖掘只产出「证据项」，不进生产快照、"
                    "不影响出号。"
                ),
                "how_to_restore": (
                    "设置环境变量 LOTT_IDLE_MINING=1 并重启服务；"
                    "或调用 POST /api/mining/run 手动跑一轮。"
                ),
            })
    except Exception as exc:                     # noqa: BLE001
        items.append({
            "id": "idle_mining",
            "reason": f"挖掘状态读取失败：{type(exc).__name__}: {exc}",
            "how_to_restore": "检查 lottery/idle_mining.py 状态文件与权限。",
        })

    # 2) 大众偏好模型未拟合
    try:
        from . import popularity as POP
        model = POP.get_model()
        if not getattr(model, "fitted", False):
            items.append({
                "id": "popularity_model",
                "reason": (
                    "大众偏好模型尚未拟合（历史数据不足或 p1/sales 缺失）。"
                    "此时 /api/portfolio/popularity 返回 503，而不是给一个假档位。"
                ),
                "how_to_restore": "补齐含 p1、sales 的历史期次后重新拟合（需 ≥300 期）。",
            })
    except Exception as exc:                     # noqa: BLE001
        items.append({
            "id": "popularity_model",
            "reason": f"模型状态读取失败：{type(exc).__name__}: {exc}",
            "how_to_restore": "检查 lottery/popularity.py 与数据库可用性。",
        })

    # 3) 无可用回放规律（生产快照 usable_replayable = 0）
    try:
        from . import patterns_runtime
        snap = patterns_runtime.load_snapshot() or {}
        counts = snap.get("counts") or {}
        if int(counts.get("usable_replayable") or 0) == 0:
            items.append({
                "id": "usable_replayable",
                "reason": (
                    "生产快照里可用且可回放的规律数为 0 —— "
                    "系统当前没有任何被认定可用于出号的规律。"
                    "这不是加载失败，是审计结论：历史规律无法通过回放检验。"
                ),
                "how_to_restore": (
                    "POST /api/rules/refresh 重建快照并回放；"
                    "若结论仍为 0，说明确实没有通过检验的规律（不应为此降门槛）。"
                ),
            })
    except Exception as exc:                     # noqa: BLE001
        items.append({
            "id": "usable_replayable",
            "reason": f"规律快照读取失败：{type(exc).__name__}: {exc}",
            "how_to_restore": "POST /api/rules/refresh 重建快照。",
        })

    return items


# ==========================================================================
# 端点
# ==========================================================================

@router.get("/api/capabilities")
def capabilities() -> Any:
    """系统能力清单：能做什么 / 做不到什么 / 当前哪项降级 / 五个锁定常量。

    `constants` 全部实时读取，供 CI 与独立审计逐项比对红线表。
    """
    try:
        consts = _live_constants()
    except Exception as exc:                     # noqa: BLE001 —— 连常量都算不出才整体降级
        return JSONResponse(
            {"ok": False, "degraded": True,
             "error": f"能力清单常量读取失败：{type(exc).__name__}: {exc}"},
            status_code=503)

    can = _can_items()
    cannot = _cannot_items()
    degraded = _degraded_items()

    missing = [k for k in ("alpha", "single", "cross", "ev_single", "ev_sixth")
               if consts.get(k) is None]

    return _jsonable({
        "ok": True,
        # ===== 降级语义：三个字段各管一件事，互不混用（2026-10-21 修复）=====
        #
        # 缺陷背景（齐交付实测）：旧版
        #     "degraded": bool(missing)        # 只反映"5 个常量取没取全"
        #     "degraded_items": degraded       # 实际可能有 2 项能力不可用
        # 于是出现 `degraded=False` 而 `degraded_items` 非空的矛盾状态。
        # 而 D-03 验收标准第 3 条明确要求「前端拿到 degraded=true 时必须显示不可用」——
        # 前端照 `degraded` 判断，会把 2 个降级项显示成正常，**直接违反「降级必须可辨识」**。
        #
        # 这正是本项目记录过的坑换字段名重演：
        # `idle_mining.usable`（20 条）vs `patterns_runtime.usable`（0 条）曾让
        # 两个"可用"并存。**只修判定不改名，等于把歧义留在字段名里。**
        #
        # 现在的契约（三者是可推导关系，有测试守卫）：
        #   · `degraded`            —— **整体**是否降级 = 常量不全 或 有能力项不可用
        #   · `constants_degraded`  —— 仅常量是否取不全（旧 `degraded` 的真实含义，改名）
        #   · `degraded_items`      —— 具体哪些能力项不可用
        #   · `degraded_constants`  —— 具体哪些常量缺失
        "degraded": bool(missing) or bool(degraded),
        "constants_degraded": bool(missing),
        "degraded_constants": missing,
        # ===== 口径声明：放**顶层**，不塞进 constants（2026-10-21 修复）=====
        #
        # 缺陷背景（齐交付实测）：`_live_constants()` 里已写 `out["sixth_mode"]="strict"`，
        # 但本函数的响应体是**白名单重新组装**的：
        #     "constants": {k: consts.get(k) for k in ("alpha","single","cross",
        #                                              "ev_single","ev_sixth")}
        # `sixth_mode` 不在那五元组里 ⇒ **被静默丢弃**。字段写进了代码却不出现在响应，
        # 比没写更糟：下一个人读 `_live_constants()` 会以为它在那儿。
        #
        # 为什么放顶层而不是塞进 constants：它不是"某个常量的值"，而是
        # **整个 constants 段的口径声明**（`ev_single` 依赖它才有意义）。
        # 塞进 constants 会让人误以为它只描述 `ev_single` 一个数。
        "sixth_mode": consts.get("sixth_mode"),
        "can": can,
        "cannot": cannot,
        "degraded_items": degraded,
        "constants": {k: consts.get(k) for k in
                      ("alpha", "single", "cross", "ev_single", "ev_sixth")},
        "constants_meta": {
            "sources": consts.get("sources", {}),
            "prize_assumption": consts.get("prize_assumption"),
            "derivation_zh": consts.get("derivation_zh"),
            "errors": consts.get("errors", {}),
            "note_zh": (
                "本段每个数字都在请求时从代码实时读取，不是字面量。"
                "若与红线表不一致，说明代码或文档之一发生了漂移，应当先查清再改文档。"
            ),
        },
        "summary": {
            "n_can": len(can),
            "n_cannot": len(cannot),
            "n_degraded": len(degraded),
            "verdict_zh": (
                "系统能审计、能排序、能改分布；"
                "不能预测号码、不能给出正期望、不能抬高返奖比。"
            ),
        },
        "disclaimer": (
            "本系统不改变任何一注的中奖概率，也不改变期望值。"
            "双色球为独立随机事件，长期期望为负。"
            "所有数字均可在本机复跑，`can[*].evidence_script` 给出脚本路径。"
        ),
    })
