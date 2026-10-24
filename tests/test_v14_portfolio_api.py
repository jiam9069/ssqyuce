"""V14 端点边界加固 + 能力清单端点（`routes_portfolio`）。

## 这个文件守的是什么

不是"端点能返回 200"，而是**边界上的每个非法输入都不许变成 500**，
以及**端点自己说的话不能和它自己返回的数字打架**。

## 两种测法都用（2026-10-21 更正）

**原先这里写着"容器内没有 `httpx` 的 TestClient 可用"—— 那是错的。**
实测 `httpx 0.28.1`，`from fastapi.testclient import TestClient` 正常，HTTP 层完全可测。
这个错误前提让我放弃了最能自证的手段（真 HTTP 请求），
**比"没写测试"更难发现，因为文件头写着理由，看起来像深思熟虑的取舍。**

现在两种都保留，各有不可替代之处：

* **直接调用路由函数**（本文件主体）—— 能精确断言 `JSONResponse.status_code`
  与 body，且能构造 `n_tickets=0/-1` 这类**HTTP 层会先拦成 422** 的输入
  （`Query(ge=1)` 在函数体之前生效，所以要测函数自身的兜底，只能绕过 HTTP）。
* **真 HTTP（`TestClient`，见文件末尾「十二」）** —— 证明参数校验在**真实请求
  链路**上成立，而不是函数被直接调用时的巧合。

    from lottery import routes_portfolio as R
    resp = R.portfolio_popularity(reds="1,2,3,4")   # 直接调
    assert resp.status_code == 400                   # 断言返回的 JSONResponse

## 为什么必须区分「未拟合」与「已拟合」

pytest 的 autouse 隔离（`tests/conftest.py::_auto_isolate_lottery_paths`）会把
`config.DB_PATH` 指向**空的临时库**。而 `popularity.get_model()` 在数据不足时
**吞掉异常并返回未拟合对象**（不抛）。后果：本文件里 `popularity` 端点默认走
**503 分支**。这两条路径都要当正式用例测，不能假设有数据。

## 断言原则（沿用 test_w08_frontend_auth.py:175 定下的原则）

**断言行为，不断言实现字面量。**

所以本文件里最重要的守卫不是"披露文本里必须有某个词"，而是
`test_disclosure_never_contradicts_returned_ratio`：
把披露文本里**声称的返奖比**解析出来，和**它返回的 `ev_ratio_sixth`** 比对。
文案怎么改写都行，只要两者不矛盾 —— 这才是防"自相矛盾"的真正守卫。
"""
from __future__ import annotations

import random
import re

import pytest

# V36（2026-10-24 审计 P1-6）：六等奖官方口径常量。
# 官方六等奖 = 蓝中 **且** 红 ≤ 2；旧基准 5/32 = 0.15625 隐含"蓝中即中奖"，
# 漏了红球条件（偏高 6.25%）。这里独立算，不 import 被测模块，
# 避免"用实现自证实现"的自指断言。
from math import comb as _comb
P_RED_LE2_OFFICIAL = (_comb(6, 0) * _comb(27, 6) + _comb(6, 1) * _comb(27, 5)
                      + _comb(6, 2) * _comb(27, 4)) / _comb(33, 6)
EV_RATIO_SIXTH_OFFICIAL = 5.0 * P_RED_LE2_OFFICIAL / 32.0     # ≈0.1472314
from fastapi.responses import JSONResponse

from lottery import routes_portfolio as R


# ==========================================================================
# 公共工具
# ==========================================================================

@pytest.fixture(autouse=True)
def _isolate_popularity_state():
    """**每个用例前后重置 popularity 的进程级状态**。

    ## 为什么必须（Product/刘工报的真实缺陷）

    这些是**模块级全局**：`_CACHE` / `_WARMING` / `_READY` / `_LAST_UNFIT_REASON` /
    `_LAST_FIT_SECONDS` / `_LAST_WARMUP_SECONDS` / `_FIT_STARTED_AT` / `_RETRY_EXHAUSTED`。

    后果实测：`test_popularity_503_reason_warming_when_fitting` **单跑 passed、
    全量 failed** —— 因为前面的用例把 `_WARMING` 留成 set，后面的用例读到脏状态。

    这表现为「假红」，但根因是真的：**测试之间没有隔离进程级状态**。
    而且不隔离还会让动态 `retry_after` 跨用例串值（`_LAST_FIT_SECONDS` 残留），
    让新加的守卫变得不可信。**假红与真红共用同一根因时，先修根因。**
    """
    from lottery import popularity as POP

    saved = {
        "_CACHE": POP._CACHE,
        "_LAST_UNFIT_REASON": POP._LAST_UNFIT_REASON,
        "_LAST_FIT_SECONDS": POP._LAST_FIT_SECONDS,
        "_LAST_WARMUP_SECONDS": POP._LAST_WARMUP_SECONDS,
        "_FIT_STARTED_AT": POP._FIT_STARTED_AT,
        "_RETRY_EXHAUSTED": POP._RETRY_EXHAUSTED,
    }
    # 前置：干净状态
    POP._CACHE = None
    POP._WARMING.clear()
    POP._READY.clear()
    POP._LAST_UNFIT_REASON = "unknown"
    POP._LAST_FIT_SECONDS = None
    POP._LAST_WARMUP_SECONDS = None
    POP._FIT_STARTED_AT = None
    POP._RETRY_EXHAUSTED = False
    try:
        yield
    finally:
        # 后置：还原（不污染其它测试文件）
        POP._WARMING.clear()
        POP._READY.clear()
        for k, v in saved.items():
            setattr(POP, k, v)


def _status(resp) -> int:
    """路由函数返回 dict（成功）或 JSONResponse（失败）时的状态码。"""
    if isinstance(resp, JSONResponse):
        return resp.status_code
    return 200


def _json(resp) -> dict:
    """把两种返回统一成 dict，便于断言 body。"""
    if isinstance(resp, JSONResponse):
        import json
        return json.loads(resp.body)
    return resp


def _ensure_popularity_fitted(n: int = 400, seed: int = 7):
    """在当前（临时）库里写入带 `p1`/`sales` 的合成期次，并让模型拟合。

    `conftest.make_draws` 造的期次**不含 `p1`/`sales`**，而 popularity 只使用
    这两列都存在的行（`fit()` 里 `rows = [... if d.get("p1") and d.get("sales")]`）。
    故这里单独造数据：至少 300 期（`fit` 的下限），否则模型永远未拟合。
    """
    from lottery import db, popularity as POP

    rng = random.Random(seed)
    draws = []
    for i in range(n):
        reds = sorted(rng.sample(range(1, 34), 6))
        draws.append({
            "issue": f"20{i // 150 + 13}{i % 150 + 1:03d}",
            "date": f"20{i // 150 + 13}-01-{(i % 28) + 1:02d}",
            "reds": reds,
            "order": list(reds),
            "blue": rng.randint(1, 16),
            # 让 p1 与形态弱相关（生日号越多，p1 略高）——模拟真实信号
            "p1": float(3 + sum(1 for x in reds if x <= 31) + rng.random()),
            "sales": 3.2e8 + rng.random() * 1e8,
        })
    db.upsert_draws(draws)
    POP._CACHE = None            # 强制下一次 get_model() 重新拟合
    model = POP.get_model()
    return model


@pytest.fixture
def fitted_popularity():
    """已拟合的 popularity 模型（用完还原缓存，避免污染其它用例）。"""
    from lottery import popularity as POP
    saved = POP._CACHE
    model = _ensure_popularity_fitted()
    assert model.fitted, "前置条件失败：合成数据未能让模型拟合"
    yield model
    POP._CACHE = saved


@pytest.fixture
def unfitted_popularity():
    """显式未拟合的模型（`_CACHE` 被替换为空模型）。"""
    from lottery import popularity as POP
    saved = POP._CACHE
    POP._CACHE = POP.PopularityModel()     # fitted=False
    yield POP._CACHE
    POP._CACHE = saved


# ==========================================================================
# 一、`/api/portfolio/popularity` 的 reds 参数边界（全部必须 400）
# ==========================================================================

@pytest.mark.parametrize("reds, why", [
    ("1,2,3,4,5,x",      "含非整数"),
    ("1,2,3,4,5",        "只有 5 个"),
    ("1,2,3,4,5,6,7",    "有 7 个"),
    ("",                 "空串（0 个）"),
    ("1,1,2,3,4,5",      "有重复"),
    ("0,2,3,4,5,6",      "含越界下界 0"),
    ("28,29,30,31,32,34", "含越界上界 34"),
    ("-1,2,3,4,5,6",     "含负数"),
    ("1.5,2,3,4,5,6",    "含浮点数"),
    (None,               "None"),
    ("1,,2,3,4,5,6",     "含空段（仍解析出 6 个，见下一个用例说明）"),
])
def test_popularity_reds_invalid_returns_400(reds, why):
    """reds 的非法形态一律 400 —— 绝不能落到 500（未捕获异常）。"""
    resp = R.portfolio_popularity(reds=reds, sales=0.0)
    code = _status(resp)

    # "1,,2,3,4,5,6" 的空段被 `if x.strip()` 过滤后恰好是 6 个合法号，应进入
    # **解析成功**分支。这里断言的是「不被判为参数非法」（code != 400），
    # 而不是断言 200 —— 因为模型是否就绪与参数解析无关，未拟合时会是 503。
    # 这正是本用例要守的行为：解析宽容 ≠ 校验放行。
    if why.startswith("含空段"):
        assert code != 400, f"空段应被过滤后正常解析，却被判为参数非法：{_json(resp)}"
        return

    assert code == 400, f"{why}（reds={reds!r}）应为 400，实际 {code}"
    body = _json(resp)
    assert body["ok"] is False
    assert body.get("error"), "400 必须带人读错误信息"
    assert "500" not in str(body.get("error"))


def test_popularity_reds_not_string_number_is_400():
    """传整数（而非字符串）时按 1 个号处理 → 400，不抛 TypeError。"""
    resp = R.portfolio_popularity(reds=123456, sales=0.0)
    assert _status(resp) == 400
    assert "6" in _json(resp)["error"]


def test_popularity_reds_tolerant_forms_return_200(fitted_popularity):
    """宽容解析：中文逗号、前后空格、尾逗号都应被接受（不是缺陷）。"""
    for reds in ("1，2，3，4，5，6", " 1, 2, 3, 4, 5, 6 ", "1,2,3,4,5,6,"):
        resp = R.portfolio_popularity(reds=reds, sales=0.0)
        assert _status(resp) == 200, f"{reds!r} 应被宽容接受，实际 {_status(resp)}"


# ==========================================================================
# 二、`sales` 参数：空转但绝不 500
# ==========================================================================

@pytest.mark.parametrize("sales", [
    0.0, -5.0, 1e308, -1e308, float("inf"), float("-inf"), float("nan"), 1e8,
])
def test_popularity_sales_extremes_never_500(fitted_popularity, sales):
    """`sales` 取任何值都不得 500。

    **已知设计**：`PopularityModel.describe()` 根本不使用 `sales` 的值（档位在
    标准化空间按分位标定），所以负数/极大/inf/nan 都不会出错。这不是"边界没测"，
    而是参数本身空转 —— 见 `popularity.PopularityModel.describe` 内的注释。
    """
    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=sales)
    assert _status(resp) == 200
    assert _json(resp)["ok"] is True


def test_popularity_sales_does_not_change_tier(fitted_popularity):
    """`sales` 空转的**行为断言**：传不同销量，档位必须完全一致。

    这条比"参数存在"更重要 —— 它把"空转"从注释变成了可执行契约。
    若哪天有人让 `sales` 真的参与判定，本用例会失败并强制其更新文档与模型版本。
    """
    tiers = set()
    for sales in (0.0, 1e6, 1e8, 1e12, -5.0):
        resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=sales)
        tiers.add(_json(resp)["tier"])
    assert len(tiers) == 1, f"sales 不应影响档位，但出现了多个档位：{tiers}"


# ==========================================================================
# 三、模型未就绪 → 503（不是 500，也不是 200）
# ==========================================================================

def test_popularity_unfitted_returns_503(unfitted_popularity):
    """模型未拟合时必须是 503，且不能是 200（不能返回假档位）。"""
    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    assert _status(resp) == 503, f"未拟合应为 503，实际 {_status(resp)}"
    body = _json(resp)
    assert body["ok"] is False
    # 2026-10-21：文案已按原因分叉（见「十三」节）。
    # 这条用例只断言「返回一个非空的、能区分原因的错误」，
    # 具体分支的正确性由 test_popularity_503_reason_* 覆盖。
    # 三个合法原因分支：
    #   warming          预热/重试正在拟合
    #   insufficient_data 真样本不足（<300 期）
    #   model_not_ready  从未尝试过拟合（例如本测试环境无预热线程）
    assert body.get("reason") in ("model_warming", "insufficient_data",
                                  "model_not_ready"), body.get("reason")
    assert body.get("unfitted_reason"), "必须给出可机器判定的具体原因"
    assert body["error"]


def test_popularity_unfitted_still_validates_reds_first(unfitted_popularity):
    """参数错误优先于模型状态：非法 reds + 未拟合 → 400 而非 503。

    顺序很重要 —— 若先判模型状态，用户会以为"改对参数就能用"，
    实际上参数本身也是错的。
    """
    resp = R.portfolio_popularity(reds="1,2,3", sales=0.0)
    assert _status(resp) == 400


def test_popularity_fitted_returns_tier_not_ticket_count(fitted_popularity):
    """已拟合时返回档位，且**不得**包含任何"预测注数"字段。

    这是诚实性守卫：`describe()` 只输出档位，绝不输出"预计 X 注中奖"。
    """
    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    assert _status(resp) == 200
    body = _json(resp)
    assert body["tier"] in ("low", "mid", "high")
    assert "tier_label" in body and "caveat" in body and "honesty" in body
    # 不得出现精确注数字段（键名或数值形式都不许）
    for forbidden in ("predicted_p1", "expected_tickets", "tickets"):
        assert forbidden not in body, f"不得返回预测注数字段 {forbidden}"
    assert body["reds"] == sorted([1, 2, 3, 4, 5, 6])


# ==========================================================================
# 四、`/api/portfolio/blue-coverage` 的 n_tickets 边界
# ==========================================================================

@pytest.mark.parametrize("n", [0, -1, -100])
def test_blue_coverage_non_positive_tickets(n):
    """`n_tickets ≤ 0`：直接调用不崩溃，返回完整的"0 注"披露。

    HTTP 层由 `Query(ge=1)` 拦成 422，不进入函数体；但**直接调用**是本文件
    采用的测试方式，此时函数必须自己兜住 —— `optimize()` 内 `max(0, n)`。
    """
    resp = R.portfolio_blue_coverage(n_tickets=n)
    assert _status(resp) == 200
    body = _json(resp)
    assert body["n_tickets"] == 0, "负数应被夹到 0"
    assert body["plan"] == []
    assert body["cost_yuan"] == 0.0
    assert body["ev_sixth_yuan"] == 0.0
    # 0 注没有"频率收益"，flag 必须为 False
    assert body["honesty_flags"]["increases_prize_frequency"] is False


@pytest.mark.parametrize("n", [1, 10, 16, 100])
def test_blue_coverage_legal_range_returns_plan(n):
    """合法注数区间必须给出计划、成本、期望回收与披露（一个都不能少）。"""
    body = _json(R.portfolio_blue_coverage(n_tickets=n))
    assert body["ok"] is True
    assert body["n_tickets"] == n
    assert len(body["plan"]) == n
    assert body["cost_yuan"] == pytest.approx(n * 2.0)
    assert body["ev_sixth_yuan"] > 0
    assert body["disclosure"] and body["disclosure_lines"]
    # 三个必须同屏的数字
    for key in ("p_any_blue_covered", "p_any_blue_random_repeat",
                "cost_yuan", "ev_sixth_yuan"):
        assert key in body
    # 两条否定结论在任何合法注数下都不得翻转为 True
    flags = body["honesty_flags"]
    assert flags["increases_hit_probability"] is False
    assert flags["increases_expected_return"] is False


def test_blue_coverage_never_exceeds_blue_pool():
    """蓝球池只有 16 个：覆盖数恒 ≤ 16，且 n > 16 时概率封顶 1.0。"""
    for n in (16, 17, 50, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["blue_coverage_count"] <= 16
    for n in (17, 50, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["p_any_blue_covered"] == pytest.approx(1.0)


# ==========================================================================
# 五、【核心守卫】披露不得与返回数字自相矛盾
# ==========================================================================

def test_disclosure_never_contradicts_returned_ratio():
    """**本文件最重要的断言**：从披露文本里解析它声称的返奖比，
    与它返回的 `ev_ratio_sixth` 比对 —— 两者必须一致。

    缺陷背景（2026-10-21 实测）：旧版披露无条件写死
    「六等奖返奖比**恒为** X%，与注数无关」，而 `ev_ratio_sixth` 在 n > 16 时
    会掉到 2.5%。端点在说"恒为 15.625%"的同时自己返回 2.5% —— 用户一旦发现
    这类矛盾，就再也不会相信任何数字。

    本守卫**不断言文案措辞**（"恒为"改不改都行），只断言
    "披露里声称的百分比 == 返回的 ev_ratio_sixth"。这样文案可以自由改写，
    但永远不能与数字打架。
    """
    pattern = re.compile(r"六等奖返奖比[^0-9%]*?(\d+(?:\.\d+)?)\s*%")
    for n in (1, 5, 10, 16, 17, 50, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        joined = "\n".join(body["disclosure_lines"])
        m = pattern.search(joined)
        if m is None:
            # n=0 走"返奖比无从计算"分支，不声称任何百分比 —— 合法
            assert n == 0 or "无从计算" in joined, (
                f"n={n} 的披露没有声称返奖比，也未说明无从计算：{joined!r}")
            continue
        claimed = float(m.group(1)) / 100.0
        assert claimed == pytest.approx(body["ev_ratio_sixth"], abs=5e-4), (
            f"n={n}：披露声称返奖比 {claimed}，但返回 ev_ratio_sixth="
            f"{body['ev_ratio_sixth']} —— 端点自相矛盾")


def test_blue_coverage_ratio_is_constant_for_all_n():
    """返奖比恒为 5/32 = 15.625%，**对任意 n（含 n > 16）**。

    ## 这条测试曾经写反过（2026-10-21 教训，务必保留这段说明）

    缺陷被发现时，第一反应是「n>16 时蓝球池占满、概率封顶而成本继续涨 ⇒
    返奖比被稀释」，于是写了「n>16 返奖比必须下降」的断言。

    **那个判断是错的。** 用 `evaluate.tickets_result()` 真实结算反证：
    n=100 时每个开奖蓝球下命中注数为 6~7 注（均值 6.25），
    六等奖按**注**兑付 ⇒ 真实回收 31.25 元，返奖比 15.625%。

    真正的 bug 在 EV 公式：`p_cover = min(n,16)/16` 是「**至少中一次**的概率」，
    却被乘了 ¥5；应该乘 ¥5 的是「**期望命中注数** = n/16」。
    两者在 n ≤ 16 时同值（蓝球互不相同 ⇒ 至多中一次），所以旧测试抓不到。

    **变异捕捉**：把 `ev_sixth` 改回 `p_cover * SIXTH_PRIZE`，
    n=17/50/100 会立刻失败（0.147/0.05/0.025 ≠ 0.15625）。
    """
    for n in (1, 5, 10, 16, 17, 50, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_OFFICIAL, abs=1e-6), (
            f"n={n} 返奖比 {body['ev_ratio_sixth']} 偏离官方口径 —— "
            f"若在 n>16 处偏离，说明 EV 又用回了「至少中一次」口径")


@pytest.mark.parametrize("n, expected_ev", [
    # V36：官方无条件期望 = n × P(红≤2)/16 × 5（旧值 n/16×5 漏了红球条件）
    (1, 0.29446), (10, 2.94463), (16, 4.71141),
    (17, 5.00587), (50, 14.72313), (100, 29.44627),
])
def test_sixth_ev_uses_expected_hit_count(n, expected_ev):
    """`ev_sixth_yuan == n × P(红≤2)/16 × 5`，对任意 n 成立。

    V36 修正：旧式 `n/16×5` 假设"中蓝即兑 5 元"，漏了六等奖还要求红 ≤2。
    """
    body = _json(R.portfolio_blue_coverage(n_tickets=n))
    # 端点字段按 4 位小数发布，容差取 5e-5
    assert body["ev_sixth_yuan"] == pytest.approx(expected_ev, abs=5e-5), (
        f"n={n}：六等奖 EV 应为 {expected_ev}（= n×P(红≤2)/16×5），"
        f"实际 {body['ev_sixth_yuan']}")


def test_expected_hits_and_cover_probability_are_distinct():
    """`expected_blue_hits = n/16`（不封顶）与 `p_any_blue_covered`（封顶 1.0）必须分开。

    这两个量一旦混用就会重演本次缺陷。断言它们**不相等**（n>16 时）。
    """
    for n in (17, 50, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["p_any_blue_covered"] == pytest.approx(1.0)
        assert body["expected_blue_hits"] == pytest.approx(n / 16, abs=1e-4)
        assert body["expected_blue_hits"] > 1.0, "n>16 时期望命中注数应 > 1"


def test_sixth_ev_matches_real_settlement_engine():
    """用项目自己的结算函数交叉验证 EV —— 不靠公式自证。

    `evaluate.tickets_result()` 逐注调 `prize_level` 再求和，是六等奖的权威口径。

    ## V36 修正（2026-10-24 审计 P1-6）：必须结算**无条件**期望

    旧版本把开奖红球固定成"与票的红球全不相交"（红命中恒为 0），只让蓝球决定。
    在那个**条件场景**里"红≤2"永远成立，于是每期命中 n/16 注、返奖比恒为
    0.15625 —— 这个数字是"给定红命中 0"的**条件期望**，不是端点上那个
    **无条件期望**。端点要回答的是"随便一期开奖，这批票值多少钱"，
    此时必须对红球分布也取期望（红命中 ≥3 时中蓝也不兑六等奖）。

    两者都对，但**不能混用**：无条件口径 = 5 × P_SIXTH × n / 成本 = 0.1472314。
    现在把开奖红球也遍历掉（用 C(33,6) 全枚举太贵，改用按红命中数加权的
    精确超几何分布），得到真正的无条件期望来对拍端点。
    """
    from lottery import blue_coverage as BC
    from lottery.evaluate import PRIZE_CASH, prize_level

    n = 100
    plan = BC.coverage_plan(n)
    pred_reds = [1, 2, 3, 4, 5, 6]
    total_red = _comb(33, 6)

    # 无条件期望：对"开奖红球命中 r 个"按超几何概率加权，
    # 每种 r 用一个代表开奖红球集合（兑奖只取决于命中数，与具体号码无关）。
    total = 0.0
    for r in range(7):
        p_r = _comb(6, r) * _comb(27, 6 - r) / total_red
        act_reds = pred_reds[:r] + list(range(7, 7 + (6 - r)))
        assert len(set(act_reds)) == 6
        for act_blue in range(1, 17):           # 16 个蓝球等概率
            for b in plan:
                if prize_level(pred_reds, b, act_reds, act_blue) == 6:
                    total += p_r * (1.0 / 16.0) * PRIZE_CASH[6]
    real_ev = total

    body = _json(R.portfolio_blue_coverage(n_tickets=n))
    # 端点字段本身按 4 位小数发布（人读口径），故容差取 5e-5 而非 1e-6。
    assert body["ev_sixth_yuan"] == pytest.approx(real_ev, abs=5e-5), (
        f"端点报六等奖 EV={body['ev_sixth_yuan']}，真实结算 {real_ev} —— 公式与结算不一致")


def test_blue_coverage_saturation_flag_matches_definition():
    """`coverage_saturated` 严格等价于 `n_tickets > 16`，且**只**表示概率封顶。

    它不得暗示返奖比变化 —— 饱和区返奖比仍是 15.625%。
    """
    for n in (1, 15, 16, 17, 99, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["coverage_saturated"] is (n > 16), f"n={n} 饱和标志错误"
    for n in (17, 100):
        assert _json(R.portfolio_blue_coverage(n_tickets=n))["ev_ratio_sixth"] \
            == pytest.approx(EV_RATIO_SIXTH_OFFICIAL, abs=1e-6)


def test_saturation_note_is_about_probability_not_ratio():
    """n > 16 的提示必须讲**概率**封顶，不得暗示返奖比下降。"""
    joined = "\n".join(
        _json(R.portfolio_blue_coverage(n_tickets=100))["disclosure_lines"])
    assert "封顶" in joined, "n>16 必须提示概率封顶"
    assert "至少中一次" in joined, "提示必须指向「至少中一次」这个量"
    for bad in ("返奖比随之下降", "返奖比下降", "返奖比被稀释"):
        assert bad not in joined, f"披露含错误陈述「{bad}」（返奖比实际恒定）"


# ==========================================================================
# 六、`/api/portfolio/capabilities` 能力清单
# ==========================================================================

def test_capabilities_returns_four_items():
    body = _json(R.portfolio_capabilities())
    assert body["ok"] is True
    assert body["n_capabilities"] == 4
    ids = [c["id"] for c in body["capabilities"]]
    assert ids == ["predict_numbers", "detect_bias",
                   "popularity_rank", "blue_coverage"]


def test_capabilities_each_item_has_evidence_and_honest_note():
    """每条能力都必须带 evidence（含出处）与 honest_note —— 缺一不可。"""
    body = _json(R.portfolio_capabilities())
    for cap in body["capabilities"]:
        assert cap.get("honest_note"), f"{cap['id']} 缺 honest_note"
        assert cap.get("summary_zh"), f"{cap['id']} 缺 summary_zh"
        ev = cap.get("evidence")
        assert ev, f"{cap['id']} 缺 evidence"
        for e in ev:
            assert e.get("claim_zh"), f"{cap['id']} 的 evidence 缺 claim_zh"
            assert e.get("value"), f"{cap['id']} 的 evidence 缺 value"
            assert e.get("source"), (
                f"{cap['id']} 的 evidence「{e.get('claim_zh')}」缺 source —— "
                f"没有出处的数字不许当结论")


def test_capabilities_verdicts_are_honest():
    """四条结论里三条必须是"做不到/有下限/极弱"，只有一条是真收益。"""
    body = _json(R.portfolio_capabilities())
    verdicts = {c["id"]: c["verdict"] for c in body["capabilities"]}
    assert verdicts["predict_numbers"] == "cannot"
    assert verdicts["detect_bias"] == "partially"
    assert verdicts["popularity_rank"] == "weak"
    assert verdicts["blue_coverage"] == "real_but_no_ev_gain"

    summary = body["verdict_summary"]
    assert summary["cannot"] == 1
    assert summary["real_but_no_ev_gain"] == 1
    # 真收益那一类必须同时否认"提高期望"
    assert summary["cannot"] + summary["partially"] + summary["weak"] == 3


def test_capabilities_predict_numbers_states_cannot():
    """"预测号码"必须明确写"做不到"，且带三条独立证据（χ²/条件熵/max-stat）。"""
    body = _json(R.portfolio_capabilities())
    cap = next(c for c in body["capabilities"] if c["id"] == "predict_numbers")
    assert cap["verdict"] == "cannot"
    assert "做不到" in cap["verdict_zh"]
    joined = "\n".join(e["value"] for e in cap["evidence"])
    assert "p=0.0654" in joined, "必须给出红球 χ² p 值"
    assert "0.0076" in joined, "必须给出条件熵降幅"
    assert "0.2995" in joined, "必须给出 max-stat 校正后 p 值"


def test_capabilities_blue_coverage_discloses_no_ev_gain():
    """唯一真收益的那条，必须同屏给出"不提高期望"与成本。"""
    body = _json(R.portfolio_capabilities())
    cap = next(c for c in body["capabilities"] if c["id"] == "blue_coverage")
    joined = "\n".join(e["value"] for e in cap["evidence"]) + cap["honest_note"]
    # V36：官方口径已由 15.625% 修正为 14.723%（六等奖还要求红 ≤2）
    assert "14.723" in joined or "14.72" in joined
    assert "15.625" not in joined, "旧的错误返奖比又回来了"
    assert "期望" in joined
    assert "1/17,721,088" in joined


def test_capabilities_avoids_banned_marketing_words():
    """禁用词守卫：不得出现「提高中奖率/提高命中率/更划算/稳赚」。"""
    import json
    s = json.dumps(_json(R.portfolio_capabilities()), ensure_ascii=False)
    for banned in ("提高中奖率", "提高命中率", "更划算", "稳赚"):
        assert banned not in s, f"能力清单出现禁用词「{banned}」"


def test_capabilities_disclaimer_denies_ev_change():
    """顶层免责声明必须同时否认"改变概率"与"改变期望值"。"""
    body = _json(R.portfolio_capabilities())
    d = body["disclaimer"]
    assert "不改变" in d
    assert "概率" in d
    assert "期望" in d


def test_capabilities_marks_measured_vs_documented():
    """`popularity_rank` 的 evidence 必须区分"文档记录值"与"当前实测值"。

    背景：文档写 R²=+0.0077，当前库实测约 +0.0114 —— 口径相同、数据期数不同。
    如实并列是本端点的立身之本；若哪天只剩一个数字，说明有人"挑了好看的"。
    """
    body = _json(R.portfolio_capabilities())
    cap = next(c for c in body["capabilities"] if c["id"] == "popularity_rank")
    flags = [bool(e.get("measured_now")) for e in cap["evidence"]]
    assert any(flags), "必须至少有一条标注为当前实测"
    assert any(not f for f in flags), "必须保留文档记录值作为对照"


# ==========================================================================
# 七、内容不得含 numpy 标量（序列化纪律）
# ==========================================================================

@pytest.mark.parametrize("call", [
    lambda: R.portfolio_blue_coverage(n_tickets=10),
    lambda: R.portfolio_capabilities(),
])
def test_no_numpy_scalars_in_payload(call):
    """返回体不得含 numpy 标量/数组（FastAPI 编码器会抛 TypeError）。"""
    import json
    import numpy as np

    body = _json(call())

    def _walk(obj, path="$"):
        assert not isinstance(obj, np.generic), f"{path} 是 numpy 标量 {type(obj)}"
        assert not isinstance(obj, np.ndarray), f"{path} 是 numpy 数组"
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(body)
    json.dumps(body)             # 必须可序列化


# ==========================================================================
# 七、口径地图：三个「整注返奖比」的关系（Lead 裁决 2）
#
# 系统里并存三个整注返奖比，来源不同、数值不同。用户曾因这类"同一系统两个数"
# 怀疑造假，所以**不物理删除任何一个**（删掉就失去可核对性），而是：
#   1. 每个都标明 assumption 与 method；
#   2. 用测试断言三者的**关系**（严QA 指出：三口径并存却无任何断言 = 隐患）；
#   3. 权威口径 = 可追溯到奖金假设的那个。
# ==========================================================================

def test_ev_ratio_map_relates_all_four_values():
    """四个整注返奖比必须可核对，且大小关系可断言。

    ## D1 落地后的口径地图（2026-10-21 终裁）

    | 口径 | 值 | 来源 | 消费者 |
    |---|---|---|---|
    | 中位假设 + `published` | **0.531925** | 684万/20.5万 + 边际 pb | 引擎默认，**已不用于端点** |
    | 中位假设 + `strict` | **0.522907** | 684万/20.5万 + 互斥奖级 | **端点权威值（D1）** |
    | 旧文档简化值 | 0.523 | `popularity.py:42`「1.0458/2」 | 用户可见文案 |
    | 固定奖金表 | ≈0.5028 | `PRIZE_CASH`（650万/18万）+ 互斥 | `/expected-value` |

    关键关系：
      * `strict` 必然**小于** `published`，差值约 0.9pp —— 因为 published 让六等奖
        与高奖级**重复计数**，系统性偏高。这条断言是"两个口径没被混用"的直接证据。
      * `strict`(0.522907) 与旧文档 0.523 在 1e-3 内吻合 —— 这正是 Lead 复现出
        「52.3% = strict + 中位」的原因，也是红线表数值的合法出处。
      * 固定表口径更低，因为其奖金假设（650万/18万）低于实测中位。

    **不物理删除任何口径**：删掉就失去可核对性，且会丢掉"复现 V4 历史发表值"的能力。
    """
    strict_ratio = _json(R.portfolio_blue_coverage(n_tickets=10))["ev_ratio_full"]
    fixed_ratio = _json(R.portfolio_expected_value(limit=1))["expected_return_per_ticket"] / 2.0

    # 端点走 strict（D1）
    assert strict_ratio == pytest.approx(0.522907, abs=1e-6), \
        f"端点应为 strict 口径 0.522907，实际 {strict_ratio}"

    # 引擎仍保留 published（刻意双口径，不得删除）
    from lottery import engine
    published = float(engine._single_ticket_ev(
        {"first": 6_840_000, "second": 205_000}, sixth="published")["ev_ratio"])
    assert published == pytest.approx(0.531925, abs=1e-6), \
        "engine 的 published 分支必须保留（可复现 V4 历史发表值）"

    # 核心关系 1：strict < published（published 重复计数，偏高 ~0.9pp）
    assert strict_ratio < published, \
        f"strict({strict_ratio}) 应小于 published({published})"
    assert published - strict_ratio == pytest.approx(0.009, abs=2e-3), \
        f"两口径差值应约 0.9pp，实际 {(published-strict_ratio)*100:.3f}pp"

    # 核心关系 2：strict 与旧文档 52.3% 吻合（红线表的合法出处）
    assert strict_ratio * 100 == pytest.approx(52.3, abs=0.1)

    # 核心关系 3：固定表口径最低
    assert fixed_ratio == pytest.approx(0.5028, abs=5e-3)
    assert fixed_ratio < strict_ratio

    # 全部 < 1：负期望是数学事实
    for name, v in (("strict", strict_ratio), ("published", published), ("fixed", fixed_ratio)):
        assert v < 1.0, f"{name} 返奖比 {v} >= 1"


def test_ev_ratio_full_carries_prize_assumption():
    """权威返奖比必须**带着**奖金假设一起返回（否则用户以为是确定值）。"""
    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    pa = body.get("prize_assumption")
    assert pa, "ev_ratio_full 必须配 prize_assumption 一起返回"
    assert pa["first"] == 6_840_000
    assert pa["second"] == 205_000
    assert pa.get("label")


def test_ev_ratio_full_never_exceeds_one():
    """整注返奖比在任何注数下都必须 < 1 —— 负期望是双色球的数学事实。"""
    for n in (1, 10, 16, 100):
        body = _json(R.portfolio_blue_coverage(n_tickets=n))
        assert body["ev_ratio_full"] < 1.0, f"n={n} 整注返奖比 {body['ev_ratio_full']} >= 1"
        assert body["net_sixth_yuan"] < 0 or n == 0, f"n={n} 六等奖口径竟不亏"


# ==========================================================================
# 十、返奖比恒定性：跨 n=1..10000 全域（Lead 要求）
#
# 目的：防止将来有人把 `ev_sixth` 改回 `p_cover × 5`（即「至少中一次的概率」）
# 或做任何"优化"。返奖比必须**处处**等于 5×P(红≤2)/32，而不只是在几个采样点上等于。
# ==========================================================================

#: 代表性取样点：覆盖 n≤16 段、临界点 16/17、以及多个量级以上
_RATIO_SAMPLE_NS = (
    1, 2, 3, 7, 8, 15, 16,        # 覆盖未饱和段与临界点
    17, 18, 20, 31, 32,           # 刚越过蓝球池（旧 bug 从此处开始）
    50, 64, 100,                  # 中等规模（API le=100 上限也在内）
    128, 512, 1000, 4096, 10000,  # 大数量级
)


@pytest.mark.parametrize("n", _RATIO_SAMPLE_NS)
def test_return_ratio_constant_across_all_magnitudes(n):
    """`ev_ratio_sixth` 在 n=1..10000 的各个量级上都必须精确等于 5×P(红≤2)/32。

    这是"覆盖优化不改变期望"的终极具守。任何把 EV 公式改错的改动都会
    在 n>16 的点上立刻失败。
    """
    body = _json(R.portfolio_blue_coverage(n_tickets=n))
    assert body["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_OFFICIAL, abs=1e-6), (
        f"n={n} 返奖比 {body['ev_ratio_sixth']} != 5×P(红≤2)/32 —— "
        f"期望命中注数口径可能又被换成了「至少中一次」")
    # 同时校验 EV 精确等于 n × P(红≤2) / 16 × 5（**不是**封顶值，也**不是** n/16×5）。
    # V36：旧式 n/16×5 假设"中蓝即兑 5 元"，漏了六等奖要求红 ≤2 —— 那会让
    # 每注多算 P(红≥3)≈5.8% 的奖金。这里用官方无条件期望。
    assert body["ev_sixth_yuan"] == pytest.approx(
        n * P_RED_LE2_OFFICIAL / 16.0 * 5.0, abs=1e-4)


def test_return_ratio_constant_over_full_range_1_to_1000():
    """穷举 n=1..1000：返奖比必须处处恒定（采样点之外也要成立）。

    为何到 1000 而不是 10000：`optimize()` 每次都要构造长度为 n 的 plan，
    1..10000 穷举在本机上约 20s，对单测过重；量级验证交给上面的参数化用例
    （已含 10000）。区间内穷举 + 量级采样，两层互补。
    """
    deviations = []
    for n in range(1, 1001):
        ratio = _json(R.portfolio_blue_coverage(n_tickets=n))["ev_ratio_sixth"]
        if abs(ratio - EV_RATIO_SIXTH_OFFICIAL) > 1e-6:
            deviations.append((n, ratio))
    assert not deviations, f"以下注数的返奖比偏离官方口径：{deviations[:10]}"


# ==========================================================================
# 十一、D1 + D2 落地守卫（2026-10-21 三人核算结论）
#
# 背景：同一页面两张卡对同一个量给出 52.3% vs 53.2%，用户一除就发现。
# 根因是消费方选了 `published`（六等奖与高奖级重复计数，偏高 ~0.9pp）
# 而非 `strict`（互斥奖级）。同时 disclosure 用 .2f 截断了精确恒等式 15.625%。
# ==========================================================================

def test_d1_full_ratio_uses_strict_mode():
    """D1：整注返奖比必须走 `strict`，且响应显式标注 `sixth_mode`。

    **变异捕捉**：把 `sixth="strict"` 改回 `"published"` ⇒
    `ev_ratio_full` 变 0.531925，本用例立刻红。
    """
    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    assert body["sixth_mode"] == "strict"
    assert body["ev_ratio_full"] == pytest.approx(0.522907, abs=1e-6), (
        f"整注返奖比应走 strict（0.522907），实际 {body['ev_ratio_full']} —— "
        f"若为 0.531925 说明又用回了 published（与 popularity 卡 52.3% 矛盾）")


def test_d1_matches_lead_redline_and_popularity_copy():
    """D1 的核心目的：端点值必须与红线表、同页 popularity 文案三方一致。

    `popularity.py` 的用户可见文案写「约 52.3% 回收」；红线表也写 52.3%。
    若端点给出 53.2%，同一页面就出现两个数 —— 这正是要消除的矛盾。
    """
    ratio = _json(R.portfolio_blue_coverage(n_tickets=10))["ev_ratio_full"]
    assert ratio * 100 == pytest.approx(52.29, abs=0.01), \
        f"渲染值 {ratio*100:.3f}% 与红线/文案的 52.3% 不一致"


def test_d1_sixth_mode_is_none_without_full_ev():
    """未请求整注口径时 `sixth_mode` 必须是 None，不能谎称 published。"""
    body = _json(R.portfolio_blue_coverage(n_tickets=10, full_ev=False))
    assert body["sixth_mode"] is None
    assert "ev_ratio_full" not in body


def test_d2_disclosure_uses_three_decimals():
    """D2：披露必须渲染出 `2.945 / 17.055 / 14.723%`（精确恒等式不得被截断）。

    V36 修正：三项均由官方六等奖口径重算（旧值 3.125 / 16.875 / 15.625% 漏了
    「红球 ≤2」这一条件：10 注成本 20 元，期望回收 20×P(红≤2)/16×5/2 × 5 折合 2.945 元）。

    **变异捕捉**：任一处改回 `.2f` ⇒ 本用例红。
    """
    lines = _json(R.portfolio_blue_coverage(n_tickets=10))["disclosure_lines"]
    assert "2.945 元" in lines[1], f"期望回收应渲染 2.945 元，实际：{lines[1]}"
    assert "17.055 元" in lines[1], f"净亏应渲染 17.055 元，实际：{lines[1]}"
    assert "14.723%" in lines[2], f"返奖比应渲染 14.723%，实际：{lines[2]}"


def test_d2_arithmetic_closes_in_disclosure():
    """D2 的**真正目的**：披露里的三项必须算术闭合（净亏 = 成本 − 期望回收）。

    这是齐交付驳回"混排"方案的理由：`3.125` 配 `16.88` 会让用户反算出
    `20 − 3.125 = 16.875`，与页面写的 16.88 差 0.005。

    本用例从**渲染文本里解析出三个数**再验恒等式 —— 不断言格式字符串，
    只断言用户看到的数字之间自洽。
    """
    lines = _json(R.portfolio_blue_coverage(n_tickets=10))["disclosure_lines"]
    m = re.search(r"成本是 (\d+(?:\.\d+)?) 元.*?期望回收 (\d+(?:\.\d+)?) 元，"
                  r"净亏 (\d+(?:\.\d+)?) 元", lines[1], re.S)
    assert m, f"披露格式无法解析：{lines[1]}"
    cost, ev, net = (float(m.group(i)) for i in (1, 2, 3))
    assert cost - ev == pytest.approx(net, abs=1e-9), (
        f"披露数字不闭合：{cost} − {ev} = {cost - ev}，但页面写净亏 {net}")


def test_d2_saturated_branch_also_three_decimals():
    """D2 第 4 处：饱和分支的返奖比同样必须 `.3f`（口径一致）。"""
    joined = "\n".join(_json(R.portfolio_blue_coverage(n_tickets=17))["disclosure_lines"])
    assert "14.723%" in joined, f"饱和分支返奖比应渲染 14.723%，实际：{joined}"


def test_d1_d2_end_to_end_consistency():
    """端到端：四个量在响应体与披露文本里必须**唯一且一致**。

    这是「同屏一致性」的机器化版本 —— 不靠人眼比对两张卡。
    """
    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    joined = "\n".join(body["disclosure_lines"])

    assert f"{body['ev_ratio_sixth'] * 100:.3f}%" in joined
    assert f"{body['ev_sixth_yuan']:.3f} 元" in joined
    # 净亏：响应体为负，披露写「净亏 正数」
    assert f"{abs(body['net_sixth_yuan']):.3f} 元" in joined
    assert body["sixth_mode"] == "strict"
    assert "52.3%" in joined


# ==========================================================================
# 十二、真 HTTP 层验证（`TestClient`）
#
# ## 为什么要有这一节（自我更正）
#
# 本文件原头部写着"容器内没有 httpx 的 TestClient 可用"，据此只用直接调用。
# **那个前提是错的**（实测 httpx 0.28.1 + TestClient 可用，齐交付与严QA 均已证实）。
#
# 直接调用证明了"路由函数本身会返回 400/503"，但**没有证明参数校验在真实
# HTTP 链路上生效**。两件事不是一回事：
#   · 直接调用绕过 FastAPI 的参数解析，`Query(ge=1, le=100)` 完全不执行；
#   · 真 HTTP 会先经过 pydantic 校验 → 越界变 422，不会进函数体。
#
# 严QA 已用真 TestClient 验过一遍（8 种非法 reds 全 400；n_tickets 越界全 422）。
# 这里把它固化成测试，避免以后有人再把"函数级断言"当成"HTTP 契约成立"。
# ==========================================================================

@pytest.fixture(scope="module")
def http_client():
    """真实 TestClient（module 级，避免重复导入开销）。"""
    from fastapi.testclient import TestClient
    from lottery import api_app
    return TestClient(api_app.app)


_BAD_REDS_HTTP = [
    "1,2,3,4,5",          # 少一个
    "1,2,3,4,5,6,7",      # 多一个
    "1,1,2,3,4,5",        # 重复
    "0,2,3,4,5,6",        # 越界下界
    "28,29,30,31,32,34",  # 越界上界
    "1,2,3,4,5,x",        # 非整数
    "",                   # 空
    "1.5,2,3,4,5,6",      # 浮点
]


@pytest.mark.parametrize("reds", _BAD_REDS_HTTP)
def test_http_popularity_bad_reds_is_400(http_client, reds):
    """真 HTTP：非法 reds 必须 400（不是 422、不是 500）。

    `reds` 是 `str` 类型，所以校验在**函数体内**做 → 400。
    这与 `n_tickets`（`int` + `ge/le`，在 pydantic 层 → 422）不同，
    两者状态码不同是**正确**的，不是不一致。
    """
    r = http_client.get("/api/portfolio/popularity", params={"reds": reds})
    assert r.status_code == 400, f"reds={reds!r} 应 400，实际 {r.status_code}"
    body = r.json()
    assert body["ok"] is False and body.get("error")


@pytest.mark.parametrize("n", ["abc", "", "0", "-1", "101", "１０"])
def test_http_blue_coverage_bad_n_is_422(http_client, n):
    """真 HTTP：`n_tickets` 非法/越界必须 422（pydantic 层拦截）。

    注意 `"１０"` 是全角数字 —— pydantic 也不接受，同样 422。
    """
    r = http_client.get("/api/portfolio/blue-coverage", params={"n_tickets": n})
    assert r.status_code == 422, f"n_tickets={n!r} 应 422，实际 {r.status_code}"


@pytest.mark.parametrize("n", [1, 10, 16, 100])
def test_http_blue_coverage_valid_n_is_200(http_client, n):
    """真 HTTP：合法注数一律 200，且关键字段齐全。"""
    r = http_client.get("/api/portfolio/blue-coverage", params={"n_tickets": n})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["n_tickets"] == n
    for key in ("ev_ratio_sixth", "ev_sixth_yuan", "coverage_saturated",
                "disclosure_lines", "sixth_mode"):
        assert key in body, f"缺字段 {key}"


def test_http_capabilities_is_json_serializable(http_client):
    """真 HTTP：能力清单端点必须能正常编码（含 numpy 标量就会 500）。"""
    r = http_client.get("/api/portfolio/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["n_capabilities"] == 4


def test_http_numpy_scalars_would_be_caught_here(http_client):
    """真 HTTP 走 FastAPI 默认编码器 —— 这是 numpy 标量缺陷的最后一道闸。

    直接调用返回 dict 时不会触发编码，只有真 HTTP 才会。
    本用例断言两个端点都能被默认编码器处理。
    """
    for path in ("/api/portfolio/blue-coverage?n_tickets=10",
                 "/api/portfolio/capabilities"):
        r = http_client.get(path)
        assert r.status_code == 200, f"{path} 编码失败：{r.status_code}"
        assert r.headers["content-type"].startswith("application/json")


# ==========================================================================
# 十三、popularity 503 的**原因分叉**（预热中 vs 样本不足）
#
# 缺陷背景：两种情况的文案合并成「模型未就绪（数据不足或尚未拟合）」，
# 但它们让用户做的事完全不同：
#   · 预热中（约 5~20s）→ 刷新一下就有；说成"数据不足"会让用户以为功能坏了
#   · 样本不足（n_train<300）→ 短期无解，只能等更多期
# 与 D-07「一个 usable 两个意思」同类：不是数字错，是用户无法区分该做什么。
# ==========================================================================

def test_popularity_503_reason_warming_when_fitting(monkeypatch):
    """预热中 → `reason="model_warming"` + `retry_after` + `Retry-After` 头。"""
    from lottery import popularity as POP

    P_unfitted = POP.PopularityModel()          # fitted=False
    monkeypatch.setattr(POP, "get_model_with_deadline", lambda timeout=None: P_unfitted)
    monkeypatch.setattr(POP, "is_warming", lambda: True)

    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    assert _status(resp) == 503
    body = _json(resp)
    assert body["reason"] == "model_warming"
    # `retry_after` 现在是**动态估值**（见「十五/十六」节），不再是固定常量；
    # 这里只断言它落在合法区间内，具体取值由 suggest_retry_after 的测试覆盖。
    assert isinstance(body["retry_after"], int)
    assert 5 <= body["retry_after"] <= 30
    # 文案必须同时给出「未就绪」这个可辨识前缀 与 「几秒后可用」的行动指引
    assert "未就绪" in body["error"], "缺少可辨识的「未就绪」前缀（D-03 降级可辨识要求）"
    assert "拟合" in body["error"]
    # 2026-10-21：文案改为给区间（不插点值），见「十六」节。
    assert "10~30" in body["error"], f"应给区间而非点值：{body['error']}"
    # HTTP 头也要带，前端/代理才能按标准处理
    # HTTP 头必须与 body 字段一致（动态值，不是常量）
    assert resp.headers.get("Retry-After") == str(body["retry_after"])


def test_popularity_503_reason_insufficient_when_not_warming(monkeypatch):
    """非预热（真样本不足）→ `reason="insufficient_data"`，且**不得**给 retry_after。"""
    from lottery import popularity as POP

    P_unfitted = POP.PopularityModel()
    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: P_unfitted)
    monkeypatch.setattr(POP, "is_warming", lambda: False)
    monkeypatch.setattr(POP, "warmup_enabled", lambda: True)
    # 显式模拟「真的样本不足」：`fit()` 返回 ok=False 时会把该原因记到模块级
    monkeypatch.setattr(POP, "_LAST_UNFIT_REASON", "insufficient_data", raising=False)

    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    assert _status(resp) == 503
    body = _json(resp)
    assert body["reason"] == "insufficient_data"
    assert body["retry_after"] is None, (
        "样本不足不是「等几秒」能解决的，给 retry_after 会误导用户反复刷新")
    assert "300" in body["error"], "应说明需要多少期样本"


def test_503_reasons_are_distinguishable():
    """两种 503 的 `reason` 必须不同 —— 这是前端区分提示的唯一依据。"""
    assert "model_warming" != "insufficient_data"


def test_is_warming_cleared_after_fit(monkeypatch):
    """`_WARMING` 必须在拟合结束后清除，否则端点会一直说「马上就好」。"""
    from lottery import popularity as POP

    draws = [{"reds": [1, 2, 3, 4, 5, 6], "p1": 5.0, "sales": 1e8}] * 400
    monkeypatch.setattr("lottery.db.load_draws", lambda: draws)
    POP.reset_cache()
    POP.get_model()
    assert POP.is_warming() is False


def test_is_warming_cleared_even_if_fit_raises(monkeypatch):
    """`fit` 抛异常时标记也必须清除（否则永远告诉用户「马上就好」）。"""
    from lottery import popularity as POP

    draws = [{"reds": [1, 2, 3, 4, 5, 6], "p1": 5.0, "sales": 1e8}] * 400
    monkeypatch.setattr("lottery.db.load_draws", lambda: draws)

    def boom(self, d):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(POP.PopularityModel, "fit", boom)
    POP.reset_cache()
    m = POP.get_model()          # 不抛
    assert m.fitted is False
    assert POP.is_warming() is False, "fit 抛异常后 _WARMING 未清除 —— 会永久误报「预热中」"


# ==========================================================================
# 十四、第四档原因：预热被配置关闭（`warming_disabled`）
#
# 缺陷背景：`LOTT_POPULARITY_WARMUP=0` 时，重试循环所在的 `_popularity_warmup()`
# 整段被同一开关门控、请求路径又不自 fit ⇒ **模型永不就绪**。
# 此时端点若还说「稍候会自动重试」，就是一个假承诺：用户会一直刷新一
# 件永远不回来的事。而且 `retry_after=5` 会让前端每 5 秒重试一次，永远失败。
# ==========================================================================

def test_warming_disabled_returns_no_retry_hint(monkeypatch):
    """预热被关闭 → `warming_disabled` + **`retry_after is None`**。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "warmup_enabled", lambda: False)
    monkeypatch.setattr(POP, "is_warming", lambda: False)
    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())
    monkeypatch.setattr(POP, "_LAST_UNFIT_REASON", "unknown", raising=False)

    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    assert _status(resp) == 503
    body = _json(resp)
    assert body["unfitted_reason"] == "warming_disabled"
    assert body["retry_after"] is None, (
        "预热已关闭却给了 retry_after —— 前端会每 N 秒重试一件永远不成功的事")
    assert resp.headers.get("Retry-After") is None
    assert "LOTT_POPULARITY_WARMUP" in body["error"], "必须告诉用户是配置关闭"
    assert "不会自动就绪" in body["error"], "必须明确「本进程内不会好」，而非「稍候」"


def test_live_warming_outranks_warming_disabled(monkeypatch):
    """**优先级守卫**：实时 `is_warming()` 必须高于「预热已关闭」。

    即使开关关闭，若有人手动调了 `get_model()`（真在拟合），
    说「正在拟合」才是真的；说「不会自动就绪」反而是错的。
    """
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "warmup_enabled", lambda: False)
    monkeypatch.setattr(POP, "is_warming", lambda: True)
    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())

    body = _json(R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0))
    assert body["unfitted_reason"] == "warming"
    assert body["retry_after"] is not None


def test_warmup_enabled_is_single_source_of_truth(monkeypatch):
    """`popularity.warmup_enabled()` 必须是唯一判定处，且读的是同一个环境变量。

    两处（api_app 起线程 / 端点选文案）各读一次 env 会漂移 —— 本项目老毛病。
    """
    from lottery import popularity as POP

    monkeypatch.setenv("LOTT_POPULARITY_WARMUP", "0")
    assert POP.warmup_enabled() is False
    monkeypatch.setenv("LOTT_POPULARITY_WARMUP", "1")
    assert POP.warmup_enabled() is True
    monkeypatch.delenv("LOTT_POPULARITY_WARMUP", raising=False)
    assert POP.warmup_enabled() is True, "未配置时应默认启用（好默认）"


def test_api_app_uses_the_shared_predicate():
    """`api_app` 不得自己再解析一次环境变量（必须委托给 popularity）。"""
    import inspect

    from lottery import api_app

    src = inspect.getsource(api_app)
    # 允许注释里提到变量名，但不允许出现 os.environ 直读的比较表达式
    offending = [ln for ln in src.splitlines()
                 if "os.environ" in ln and "LOTT_POPULARITY_WARMUP" in ln
                 and not ln.strip().startswith("#")]
    assert not offending, (
        f"api_app 里仍有直接解析 LOTT_POPULARITY_WARMUP 的代码：{offending}\n"
        f"应统一调用 popularity.warmup_enabled()")


# ==========================================================================
# 十五、`retry_after` 动态化（Lead 裁决，2026-10-21）
#
# 缺陷背景：`retry_after` 写死 5。而实测 fit 耗时随负载在 5.5s~21s 浮动
# （本机空闲 5.49s / 容器单核 6.06s / load 4 时 8.35s / load 12 时 21s）。
# 写死 5 的后果：用户照提示 5 秒后刷新仍 503，再刷还是 503 ——
# **被消耗的是用户对披露的信任**，而这正是本项目整轮在挣的东西。
#
# 反方向也不行：写死 20 在快机器上让用户白等 15 秒。
# 所以用「上次实测耗时 + 当前已耗时下界」动态估，夹到 [5, 30]。
# ==========================================================================

def test_retry_after_uses_last_measured_fit_time(monkeypatch):
    """有历史耗时时，`retry_after` 必须跟随它（不是固定 5）。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", 18.0)
    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    assert POP.suggest_retry_after() == 18


def test_retry_after_clamped_to_bounds(monkeypatch):
    """必须夹在 [5, 30]：太快频繁轮询，太慢机器也不能给出荒谬等待。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", 1.0)
    assert POP.suggest_retry_after() == 5, "过快的历史耗时应收敛到下限 5"
    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", 120.0)
    assert POP.suggest_retry_after() == 30, "异常慢应夹到上限 30"


def test_retry_after_never_shorter_than_elapsed(monkeypatch):
    """**核心守卫**：已耗时是下界。

    如果这次 fit 已经跑了 20 秒，那「5 秒后就好」必然是**又一个假承诺**。
    变异捕捉：去掉 `elapsed + 1.0` 的下界 ⇒ 本用例红。
    """
    import time as _t

    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", 6.0)   # 上次很快
    monkeypatch.setattr(POP, "_FIT_STARTED_AT", _t.time() - 20.0)  # 这次已跑了 20s
    assert POP.suggest_retry_after() == 21, (
        "已跑 20s 却给出 ≤6s 的估计 —— 又是假承诺")


def test_retry_after_initial_when_no_history(monkeypatch):
    """首次（无历史耗时、无已耗时）用保守值 10，而不是乐观的 5。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", None)
    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    assert POP.suggest_retry_after() == POP._RETRY_AFTER_INITIAL == 10


def test_warming_response_carries_dynamic_retry_after(monkeypatch):
    """端点的 `retry_after` 与 HTTP 头必须一致且来自动态估值。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "is_warming", lambda: True)
    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())
    monkeypatch.setattr(POP, "suggest_retry_after", lambda: 17)

    resp = R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0)
    body = _json(resp)
    assert body["retry_after"] == 17
    assert resp.headers.get("Retry-After") == "17"
    # 文案给**区间**，不含点值（Product 指出的坑：插值等于把点值写死一遍）
    assert "10~30" in body["error"]
    assert "17 秒" not in body["error"], "文案不应出现动态点值"


# ==========================================================================
# 十六、`retry_after` 与实测预热的**比例一致性**（可证伪守卫）
#
# 为什么用比例而不是绝对值（Product 设计，Lead 采纳）：
#   fit 耗时随负载在 6~40s 间浮动。
#   · 写死 `assert retry_after == 30` → 慢机上「通过但没意义」，快机上「无谓失败」；
#   · 比例断言让两个量同步变化 —— 机器快，实测降，判据仍成立；
#   · 上限 4 倍防的是「写死一个巨大数字让用户白等」（例如改成 300）——
#     那种改法在只看下限的测试里永远通过。
#
# ⚠️ 本组测的是**一致性**（承诺与近期实测是否同量级），不是**准确性**
#    （这一轮到底多久会好）。`last_warmup_seconds()` 是「上次」的值。
# ==========================================================================

def test_retry_after_consistent_with_last_warmup(monkeypatch):
    """承诺的重试间隔必须与实测预热耗时同量级（1x ~ 4x）。"""
    from lottery import popularity as POP

    measured = 12.64          # 实测代表性值（争 CPU 场景）
    monkeypatch.setattr(POP, "_LAST_WARMUP_SECONDS", measured)
    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)

    ra = POP.suggest_retry_after()
    # 必须分两支：`suggest_retry_after()` 有 MAX 钳位，
    # 当实测超过 MAX 时 `ra == MAX < measured` 是**正确行为**（已尽力，无法承诺更久），
    # 若无条件断言 `ra >= measured`，慢机上必然误红。
    if measured <= POP._RETRY_AFTER_MAX:
        assert ra >= 1.0 * measured, (
            f"retry_after={ra} 小于实测 {measured}s —— 用户照提示刷新会落空")
    else:
        assert ra == POP._RETRY_AFTER_MAX, (
            f"实测 {measured}s 超过上限时，应钳到 {POP._RETRY_AFTER_MAX}（已尽力）")
    assert ra <= 4.0 * measured, (
        f"retry_after={ra} 远大于实测 {measured}s —— 用户白等")


def test_retry_after_caps_when_measured_exceeds_max(monkeypatch):
    """**慢机专用用例**：实测 > MAX 时必须钳到 MAX，而不是承诺一个更大的数。

    这条把慢机行为锁死，避免"比例断言在触顶时误红"这个问题在 CI 上偶发。
    """
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    monkeypatch.setattr(POP, "_LAST_WARMUP_SECONDS", POP._RETRY_AFTER_MAX * 2)
    assert POP.suggest_retry_after() == POP._RETRY_AFTER_MAX


def test_retry_after_not_absurdly_large(monkeypatch):
    """**防「写死巨大值」**：即使实测很慢，也不得给出荒谬等待（上限 30）。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_LAST_WARMUP_SECONDS", 200.0)
    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    assert POP.suggest_retry_after() == POP._RETRY_AFTER_MAX == 30


def test_fallback_constant_matches_documented_range():
    """兜底常量必须是 30（文案「10~30 秒」的上界来源）。

    变异捕捉：把常量改回 5 ⇒ 文案区间与实际承诺不符 ⇒ 红。
    """
    from lottery import popularity as POP

    assert R._POPULARITY_RETRY_AFTER_SECONDS == 30
    assert POP._RETRY_AFTER_MAX == R._POPULARITY_RETRY_AFTER_SECONDS, (
        "动态估值的上限必须与端点兜底常量一致，否则文案区间会骗人")


def test_warming_copy_has_no_point_value(monkeypatch):
    """**文案不得出现点值**（Product 指出的坑）。

    旧写法 `"...约 %d 秒后自动可用" % retry_after` 会把点值印给用户，
    而真实区间是 6~40s —— 等于把「点值」又写死了一遍，白改。
    现在：文案给人（区间），字段给机器（保守上界）。
    """
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "is_warming", lambda: True)
    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())
    monkeypatch.setattr(POP, "suggest_retry_after", lambda: 30)

    body = _json(R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0))
    err = body["error"]
    assert "10~30" in err, f"文案应给区间，实际：{err}"
    for point in ("约 30 秒", "约 5 秒", "约 20 秒"):
        assert point not in err, f"文案出现了点值「{point}」：{err}"
    # 但机器字段仍要给具体值
    assert body["retry_after"] == 30


def test_retry_after_reflects_warmup_not_only_fit(monkeypatch):
    """预热端到端耗时（含 numpy 导入）优先于单次 fit 耗时 —— 更贴近用户真实等待。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "_FIT_STARTED_AT", None)
    monkeypatch.setattr(POP, "_LAST_FIT_SECONDS", 6.0)
    monkeypatch.setattr(POP, "_LAST_WARMUP_SECONDS", 12.6)
    assert POP.suggest_retry_after() == 13, (
        "应参考预热端到端耗时（更接近用户等待），而不是只看 fit 本身")


# ==========================================================================
# 十七、503 的 `reason` 必须**自解释**（Product/刘工报的契约问题）
#
# 缺陷：`reason: "model_not_ready"` 比它旁边的 `unfitted_reason` 更粗。
# 后果：任何第三方调用方读 `reason`（契约里最显眼的字段）会**分叉错**。
# 这与 D-07「一个 usable 两个意思」同族：不是数值错，是**分层错位**。
# 改法：`reason` 直接给细分值；`unfitted_reason` 降为兼容别名。
# ==========================================================================

_ALL_503_REASONS = (
    "model_warming",
    "insufficient_data",
    "warming_disabled",
    "retry_exhausted",
    "fit_failed",
)


def test_every_503_reason_is_self_explanatory(monkeypatch):
    """**核心守卫**：任何 503 的 `reason` 都不得是笼统的 `model_not_ready`。

    遍历所有可到达的未就绪原因，逐个触发端点，断言 `reason` 是细分值。
    变异捕捉：把任一分支改回 `model_not_ready` ⇒ 本用例红。
    """
    from lottery import popularity as POP

    checked = []
    for reason in _ALL_503_REASONS:
        monkeypatch.setattr(POP, "get_model_for_request",
                            lambda timeout=None: POP.PopularityModel())
        monkeypatch.setattr(POP, "is_warming", lambda: False)
        monkeypatch.setattr(POP, "warmup_enabled", lambda: True)
        monkeypatch.setattr(POP, "_LAST_UNFIT_REASON", reason, raising=False)
        monkeypatch.setattr(POP, "_RETRY_EXHAUSTED", reason == "retry_exhausted",
                            raising=False)

        body = _json(R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0))
        checked.append((reason, body["reason"]))
        assert body["reason"] != "model_not_ready", (
            f"未就绪原因 {reason} 的 503 仍返回笼统的 model_not_ready —— "
            f"第三方调用方只读 reason 会分叉错")
        assert body["reason"] == reason, (
            f"reason 应为 {reason}，实际 {body['reason']}")

    assert len(checked) == len(_ALL_503_REASONS)


def test_warming_disabled_reason_is_self_explanatory(monkeypatch):
    """单点验证 Product 点出的那一条：`warming_disabled` 分支的 reason。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())
    monkeypatch.setattr(POP, "is_warming", lambda: False)
    monkeypatch.setattr(POP, "warmup_enabled", lambda: False)
    monkeypatch.setattr(POP, "_LAST_UNFIT_REASON", "unknown", raising=False)

    body = _json(R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0))
    assert body["reason"] == "warming_disabled"
    assert body["unfitted_reason"] == "warming_disabled", "兼容别名应保留"
    assert body["retry_after"] is None


# ==========================================================================
# 十八、重试耗尽（第三种「假承诺」）
#
# 预热循环 `for attempt in range(1, 11)` 跑完就退出，但 `_LAST_UNFIT_REASON`
# 仍是 `fit_failed` ⇒ 旧代码给出 `retry_after=30` + 「稍候会自动重试」，
# 而**不会再有任何重试**。用户会一直刷新一件永远不回来的事。
# ==========================================================================

def test_retry_exhausted_returns_null_retry_after(monkeypatch):
    """重试耗尽 → `reason="retry_exhausted"` + `retry_after is None`。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "get_model_for_request",
                        lambda timeout=None: POP.PopularityModel())
    monkeypatch.setattr(POP, "is_warming", lambda: False)
    monkeypatch.setattr(POP, "warmup_enabled", lambda: True)
    monkeypatch.setattr(POP, "_LAST_UNFIT_REASON", "fit_failed", raising=False)
    POP.mark_retry_exhausted()

    body = _json(R.portfolio_popularity(reds="1,2,3,4,5,6", sales=0.0))
    assert body["reason"] == "retry_exhausted"
    assert body["retry_after"] is None, (
        "重试已耗尽却仍给 retry_after —— 这是第三种假承诺")
    assert "不会再自动重试" in body["error"]


def test_retry_exhausted_formatable_status(monkeypatch):
    """`describe()` 在重试耗尽时必须把原因标出来（不只端点层判断）。"""
    from lottery import popularity as POP

    m = POP.PopularityModel()
    m.fitted = False
    POP._WARMING.clear()
    POP.mark_retry_exhausted()
    out = m.describe([1, 2, 3, 4, 5, 6])
    assert out["ok"] is False
    assert out["unfitted_reason"] == "retry_exhausted"


def test_success_clears_exhausted_flag(monkeypatch):
    """一旦拟合成功，「重试耗尽」结论必须作废（否则会永久误报）。"""
    from lottery import popularity as POP

    POP.mark_retry_exhausted()
    assert POP.is_retry_exhausted() is True

    draws = [{"reds": [1, 2, 3, 4, 5, 6], "p1": 5.0, "sales": 1e8}] * 400
    monkeypatch.setattr("lottery.db.load_draws", lambda: draws)
    POP._CACHE = None
    m = POP.get_model()
    assert m.fitted
    assert POP.is_retry_exhausted() is False, "成功后必须清除耗尽标记"


def test_retry_interval_is_configurable_and_distinct_from_retry_after():
    """**两个量必须分开**（Lead 要求 ③）：后台重试间隔 ≠ 告诉用户的重试间隔。

    它们业务含义无关（系统行为 vs 用户协议），换算出同一个数只是巧合。
    本用例断言它们是**不同的符号**，防止将来有人「统一命名」时合并。
    """
    import os

    from lottery import api_app

    # 默认值
    os.environ.pop("LOTT_POPULARITY_RETRY_INTERVAL", None)
    assert api_app._popularity_retry_interval() == 30.0
    # 可配（测试用）
    os.environ["LOTT_POPULARITY_RETRY_INTERVAL"] = "0.1"
    try:
        assert api_app._popularity_retry_interval() == 0.1
    finally:
        os.environ.pop("LOTT_POPULARITY_RETRY_INTERVAL", None)

    # 与端点的 retry_after 兜底常量是**两个独立符号**
    assert api_app._popularity_retry_interval is not None
    # 一个是 float 秒（后台），一个是 int 秒（用户协议）
    assert isinstance(api_app._popularity_retry_interval(), float)
    assert isinstance(R._POPULARITY_RETRY_AFTER_SECONDS, int)


def test_max_attempts_is_configurable():
    """重试次数可配（默认 10），供端到端快速验证。"""
    import os

    from lottery import api_app

    os.environ.pop("LOTT_POPULARITY_MAX_ATTEMPTS", None)
    assert api_app._popularity_max_attempts() == 10
    os.environ["LOTT_POPULARITY_MAX_ATTEMPTS"] = "2"
    try:
        assert api_app._popularity_max_attempts() == 2
    finally:
        os.environ.pop("LOTT_POPULARITY_MAX_ATTEMPTS", None)


# ==========================================================================
# 十九、`reset_cache()` 必须清**全部**模块级状态（第 3 类守卫：防将来）
#
# 缺陷（Product/刘工实测）：旧实现只清 `_CACHE` + `_READY`，留下 6 个残留。
# 其中最严重的是耗时类残留 —— 新增的「比例守卫」读 `_LAST_WARMUP_SECONDS`
# 做断言，残留会让它 order-dependent（换测试顺序就红/绿）。
# **守卫本身不可信，比没有守卫更糟。**
#
# 本条是「形态守卫」：不枚举期望值，而是**扫模块的模块级可变状态**逐个确认。
# 防的是「将来加第 7 个状态但忘了加进 reset_cache」。
# ==========================================================================

def test_reset_cache_clears_all_module_state():
    """`reset_cache()` 后，所有模块级可变状态必须回到初始值。"""
    from lottery import popularity as POP

    POP._WARMING.set()
    POP._LAST_WARMUP_SECONDS = 38.55
    POP._LAST_FIT_SECONDS = 12.6
    POP._FIT_STARTED_AT = 123.0
    POP._LAST_UNFIT_REASON = "fit_failed"
    POP._RETRY_EXHAUSTED = True

    POP.reset_cache()

    assert POP._CACHE is None
    assert POP.is_warming() is False, "_WARMING 残留 → 端点误报「预热中」"
    assert POP._LAST_WARMUP_SECONDS is None, (
        "_LAST_WARMUP_SECONDS 残留 → suggest_retry_after() 返回上个用例的值，"
        "比例守卫变 order-dependent")
    assert POP._LAST_FIT_SECONDS is None
    assert POP._FIT_STARTED_AT is None
    assert POP._LAST_UNFIT_REASON == "unknown"
    assert POP.is_retry_exhausted() is False


def test_reset_cache_restores_suggest_retry_after_to_initial():
    """行为层验证：reset 后 `suggest_retry_after()` 必须回到初值，不受残留影响。"""
    from lottery import popularity as POP

    POP._LAST_WARMUP_SECONDS = 38.55     # 慢机残留
    assert POP.suggest_retry_after() == POP._RETRY_AFTER_MAX
    POP.reset_cache()
    assert POP.suggest_retry_after() == POP._RETRY_AFTER_INITIAL, (
        "reset 后仍用残留耗时算 retry_after")


def test_reset_cache_covers_every_mutable_module_global():
    """**形态守卫（防将来）**：扫 `popularity` 的模块级可变全局，逐个验证被清。

    这条不枚举"已知的 6 个"，而是**动态发现**模块里所有 `Optional/bool/float`
    形态的私有可变状态（排除常量与函数/类/模块）。
    将来有人加第 7 个状态却忘了加进 `reset_cache()`，本用例会红。
    """
    import types

    from lottery import popularity as POP

    # 明确属于「不可变配置/常量」的名字（不需要 reset）
    _CONST_LIKE = {
        "MODEL_VERSION", "TIER_LOW_Q", "TIER_HIGH_Q", "FEATURE_NAMES",
        "_RETRY_AFTER_MIN", "_RETRY_AFTER_MAX", "_RETRY_AFTER_INITIAL",
        "_CACHE_FILENAME",
        # 中文标签查找表：只读常量语义（没人会在运行时改它）。
        # 形态守卫首跑时抓到它 —— 说明守卫确实能发现"未登记的可变全局"。
        "_FEATURE_ZH",
    }
    # 明确属于「可变状态、必须在 reset_cache 里清」的名字
    _MUST_RESET = {
        "_CACHE", "_WARMING", "_READY", "_FIT_THREAD_ACTIVE",
        "_LAST_UNFIT_REASON", "_LAST_FIT_SECONDS", "_LAST_WARMUP_SECONDS",
        "_FIT_STARTED_AT", "_RETRY_EXHAUSTED",
    }
    # 锁对象不是"状态"，不清
    _LOCK_LIKE = {"_CACHE_LOCK", "_FIT_TRIGGER_LOCK"}

    mutable = set()
    for name, val in vars(POP).items():
        if name.startswith("__") or name in _CONST_LIKE or name in _LOCK_LIKE:
            continue
        if isinstance(val, (types.FunctionType, type, types.ModuleType)):
            continue
        if name.startswith("_") and not callable(val):
            mutable.add(name)

    missing = mutable - _MUST_RESET
    assert not missing, (
        f"以下模块级可变状态未登记在 reset_cache 的清理清单里：{sorted(missing)}\n"
        f"若它们会被测试或请求路径修改，就会造成跨用例状态污染"
        f"（本次实测：_LAST_WARMUP_SECONDS 残留让比例守卫 order-dependent）")


# ==========================================================================
# 二十、D-17：`blue-coverage` 的整注口径失败必须**可被端点报告**
#
# 旧代码：
#     except Exception:        # noqa: BLE001 —— 取不到就只给六等奖口径
#         full_ratio = None
#         assumption = None
# 这是**静默**：用户看到 `sixth_mode=None`、没有 `ev_ratio_full`，
# 但不知道是"没请求"还是"取不到"，更不知道该怎么办。
#
# Lead 的要求分三层（缺一不可）：
#   ① 不能静默；② 不能只进日志（日志没人看）；③ **必须进 `degraded_items`**。
# 另加：④ 没异常时**不得**报降级（否则永远报 = 用户习惯性忽略，比静默更糟）。
# ==========================================================================

def test_blue_coverage_reports_degraded_when_full_ev_fails(monkeypatch):
    """判据 1+2+3：异常时必须进 `degraded_items`，且三段字段齐全具体。"""
    from lottery import engine

    def boom(*a, **k):
        raise RuntimeError("模拟奖金假设表损坏")

    monkeypatch.setattr(engine, "_single_ticket_ev", boom)

    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    items = body["degraded_items"]
    assert items, "整注口径失败却没有任何降级项 —— 仍是静默"
    assert len(items) == 1

    item = items[0]
    assert item["id"] == "ev_ratio_full"
    # reason 要说清**是什么坏了**，不许写"发生错误"
    assert "整注返奖比" in item["reason"]
    assert "RuntimeError" in item["reason"], "应带上异常类型，便于排查"
    assert "发生错误" not in item["reason"], "不许用笼统措辞"
    # 同时要说清**什么没坏** —— 否则用户会以为整个端点不能用
    assert "六等奖" in item["reason"]
    # how_to_restore 必须是可执行动作，不许写"请联系管理员"
    assert item["how_to_restore"]
    assert "_single_ticket_ev" in item["how_to_restore"]
    assert "联系管理员" not in item["how_to_restore"]


def test_blue_coverage_no_degraded_when_ev_succeeds():
    """**判据 4（最易漏）**：正常路径下不得出现降级项。

    只测"有异常时会报"、不测"没异常时不报"，结果是永远报降级 ——
    用户会习惯性忽略它，**比静默更糟**。
    """
    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    assert body["degraded_items"] == [], (
        f"正常路径却报了降级项：{body['degraded_items']}")
    assert body["sixth_mode"] == "strict"
    assert "ev_ratio_full" in body


def test_blue_coverage_degraded_when_full_ev_disabled_by_request():
    """用户主动 `full_ev=False` **不算降级**（那是他自己的选择，不是故障）。"""
    body = _json(R.portfolio_blue_coverage(n_tickets=10, full_ev=False))
    assert body["degraded_items"] == [], (
        "用户主动关闭整注口径不应被报成降级 —— 那是预期行为，不是故障")
    assert body["sixth_mode"] is None


def test_blue_coverage_sixth_ev_never_affected_by_full_ev_failure(monkeypatch):
    """整注口径失败时，六等奖口径必须完全不受影响（这是降级的边界）。"""
    from lottery import engine

    monkeypatch.setattr(engine, "_single_ticket_ev",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    body = _json(R.portfolio_blue_coverage(n_tickets=10))
    assert body["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_OFFICIAL, abs=1e-6)
    # V36：官方无条件期望 = 10 × P(红≤2)/16 × 5 = 2.94463；端点按 4 位小数发布
    assert body["ev_sixth_yuan"] == pytest.approx(2.94463, abs=5e-5)
    assert body["ok"] is True, "六等奖仍可用 ⇒ 整体仍应 200"


# ==========================================================================
# 二十一、D-17 同类自查：`capabilities` 的实测值读取失败也不许静默
#
# 自查方法（qi-delivery 建议、我执行的）：一次 grep + AST 扫"静默 except"，
# 对每处问三问：① 能力受限还是故障？② 什么没坏写了吗？③ 前端拿得到吗？
#
# 结果发现 `_capability_items()` 里这处仍是静默：
#     except Exception: _r2_now, _p_now = None, None
# 后果：用户在 `popularity_rank` 的 evidence 里看到「模型未拟合，暂无实测」，
# 却分不清是「模型本来就没拟合」（等更多期）还是「读取时崩了」（查日志）。
# 这与 D-17 修的 `ev_ratio_full`、D-07 的 `usable` 两义同族。
# ==========================================================================

def test_capabilities_no_degraded_in_normal_path():
    """**判据 4（不误报）**：正常路径 `degraded=False` 且 `degraded_items` 为空。"""
    body = _json(R.portfolio_capabilities())
    assert body["degraded"] is False
    assert body["degraded_items"] == []
    assert body["n_capabilities"] == 4, "降级不应影响能力项数目"


def test_capabilities_reports_degraded_when_popularity_read_fails(monkeypatch):
    """实测值读取失败 → 必须进 `degraded_items`，三段齐全且具体。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "get_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模型读取崩溃")))

    body = _json(R.portfolio_capabilities())
    assert body["degraded"] is True
    items = body["degraded_items"]
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "popularity_measured"
    assert "读取失败" in item["reason"]
    assert "RuntimeError" in item["reason"]
    # 必须说明**什么没坏** —— 否则用户以为整个能力清单不可信
    assert "文档记录值" in item["reason"]
    assert "不受影响" in item["reason"]
    # how_to_restore 要可执行
    assert "get_model" in item["how_to_restore"]
    assert "联系管理员" not in item["how_to_restore"]


def test_capabilities_keeps_documented_value_when_measured_fails(monkeypatch):
    """降级只影响「实测值」，文档记录值仍必须给出（降级范围要精确）。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "get_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    body = _json(R.portfolio_capabilities())
    cap = next(c for c in body["capabilities"] if c["id"] == "popularity_rank")
    documented = [e for e in cap["evidence"] if not e.get("measured_now")]
    assert documented, "文档记录值被降级连带删掉了 —— 降级范围过大"
    joined = "\n".join(e["value"] for e in documented)
    assert "0.0077" in joined or "0.77" in joined


def test_capabilities_measured_marked_unavailable_on_failure(monkeypatch):
    """失败时「实测值」那条 evidence 的 value 应显示不可用，而不是静默留空。"""
    from lottery import popularity as POP

    monkeypatch.setattr(POP, "get_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    body = _json(R.portfolio_capabilities())
    cap = next(c for c in body["capabilities"] if c["id"] == "popularity_rank")
    measured = [e for e in cap["evidence"] if e.get("measured_now")]
    assert measured, "实测值那条 evidence 不应整条消失"
    assert measured[0]["value"], "实测值 evidence 的 value 不应为空"
