"""V10-10 测试：前端三卡 + 四个新 API（票单分布 / 前置注册 / ML 诊断 / 样本并排）。

覆盖：
1. 端点可用性：四个（+ 样本并排、LLM 对照臂）端点均 HTTP 200，且返回体可
   ``json.dumps``（**不含 numpy 标量/数组** —— 这是最容易漏的坑，因为底层
   ``ticket_distribution`` / ``battle`` / ``ml_model`` 内部大量用 np）。
2. 字段完整性：三卡需要的字段一个不少（缺一个前端就是空白/undefined）。
3. 诚实边界：``mixable=False`` 且**无跨来源合计字段**；票单分布必须带
   ``prize_assumption`` 与"改变分布不改变 EV"提示；ML 诊断必须带
   "trusted=False 是诚实结论"说明。
4. 前端契约：``?v=`` 升到 20261018.5 且 app.js/style.css 一致；``cfgLlm``
   不得硬编码 checked；三卡 DOM 锚点存在。

规格来源：``docs/V10_IMPROVEMENT_PLAN.md`` §3.10（V10-10）。
"""
from __future__ import annotations

# V36（2026-10-24 审计 P1-6）：六等奖官方口径常量（独立算，不 import 被测模块）。
# 官方六等奖 = 蓝中 **且** 红 ≤ 2；旧基准 0.15625 = 5/32 隐含"蓝中即中奖"，
# 漏了红球条件（偏高 6.25%）。正确值 = 5×P(红≤2)/32 ≈ 0.1472314。
from math import comb as _comb_v36
P_RED_LE2_V36 = (_comb_v36(6, 0) * _comb_v36(27, 6) + _comb_v36(6, 1) * _comb_v36(27, 5)
                 + _comb_v36(6, 2) * _comb_v36(27, 4)) / _comb_v36(33, 6)
EV_RATIO_SIXTH_V36 = 5.0 * P_RED_LE2_V36 / 32.0

import json
import re
from pathlib import Path

import os
import pytest

# D-3：`/api/ml/diagnosis` 会跑 `diagnose_ml_channels`（红/蓝两通道 × 多号码 ×
# 零分布重训），实测单次 >100s。默认档跳过这几个重计算用例，用
# LOTT_SLOW_TESTS=1 跑完整验证。
_SLOW = os.environ.get("LOTT_SLOW_TESTS") == "1"
_slow_only = pytest.mark.skipif(
    not _SLOW, reason="ML 诊断端点重计算（D-3），默认档跳过；LOTT_SLOW_TESTS=1 运行")

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_HTML = WEB_DIR / "index.html"
APP_JS = WEB_DIR / "static" / "app.js"
STYLE_CSS = WEB_DIR / "static" / "style.css"

def _js_code() -> str:
    """剥离注释后的 app.js（D-18b 合并：实现统一在 `_source_assert.js_code`）。
    ⚠️ 四份测试文件原各写一份，正则相同。合并消除「第二真源」——
       任一份被改另外三份不会跟着变且无人发现。
    """
    from _source_assert import js_code as _js_code_impl
    return _js_code_impl(APP_JS.read_text(encoding="utf-8"))


# 明确的跨来源/跨族合计字段名：这些**不得**出现在禁止相加的返回体里
_BANNED_SUM_KEYS = ("combined", "pooled", "total", "merged", "sum", "weighted")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from lottery import api_app
    return TestClient(api_app.app)


@pytest.fixture()
def with_draws(seeded_db):
    """造 420 期合成开奖（ML 诊断需要 ≥ min_start+20 期）。"""
    return seeded_db(n=420)


def _assert_jsonable(payload, where: str) -> None:
    """返回体必须能原样 json.dumps（numpy 标量会在这里炸）。"""
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:      # pragma: no cover - 失败即缺陷
        raise AssertionError(f"{where} 返回体不可 JSON 序列化（疑似 numpy 对象）: {exc}")


def _collect_numpy(payload, path: str = "$") -> list:
    """递归找出返回体里的 numpy 标量/数组（显式检查，不依赖 json.dumps 的运气）。"""
    np = pytest.importorskip("numpy")
    found = []
    if isinstance(payload, np.generic):
        found.append(path)
    elif isinstance(payload, np.ndarray):
        found.append(path)
    elif isinstance(payload, dict):
        for k, v in payload.items():
            found.extend(_collect_numpy(v, f"{path}.{k}"))
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            found.extend(_collect_numpy(v, f"{path}[{i}]"))
    return found


# ============================================================ 1) 票单分布端点

def test_distribution_endpoint_ok_and_fields(client):
    """``GET /api/portfolio/distribution`` → 200 且 26 个核心字段齐全。"""
    r = client.get("/api/portfolio/distribution?exact=false")
    assert r.status_code == 200
    body = r.json()
    _assert_jsonable(body, "distribution")
    assert body["ok"] is True
    assert body["source"] in ("prediction", "sample")
    d = body["distribution"]
    for key in ("n", "union_coverage", "redundancy", "redundancy_neutral",
                "redundancy_max", "expected_hits", "p_at_least_r", "p_any_blue",
                "ev_ratio_sixth", "ev_ratio_full", "prize_assumption",
                "var_T", "n_sigma1_sq", "var_ratio", "var_ratio_note",
                "p_at_least_r_iid", "p_at_least_r_method"):
        assert key in d, f"ticket_distribution 缺少字段 {key}"
    # 中性值必须是 12/11；仅六等奖返奖比恒为 0.1472314
    assert d["redundancy_neutral"] == pytest.approx(12 / 11)
    assert d["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)
    assert set(d["p_at_least_r"]) == {"3", "4", "5"}


def test_distribution_ev_is_distribution_invariant(client):
    """**诚实边界**：不同票组的返奖比恒等 —— "改变分布不改变 EV"必须真的成立。

    用两组差异极大的票（一组高度冗余、一组完全不相交）算两次，
    ``ev_ratio_sixth`` / ``ev_ratio_full`` 必须完全相同，而 ``redudancy`` 必须不同。
    """
    a = client.get("/api/portfolio/distribution?exact=false").json()["distribution"]
    assert a["ev_ratio_sixth"] == pytest.approx(EV_RATIO_SIXTH_V36, abs=1e-6)
    first_sixth, first_full = a["ev_ratio_sixth"], a["ev_ratio_full"]

    from lottery import engine as E
    # 高度冗余：10 注只差 1 个号的组合（冗余度远高于中性值）
    tickets = []
    for k in range(10):
        reds = sorted([1, 2, 3, 4, 5, 6][:5] + [7 + k])
        tickets.append({"reds": reds, "blue": 1 + (k % 16), "method": "t"})
    b = E.ticket_distribution(tickets, with_distribution_exact=False)
    assert b["redundancy"] > b["redundancy_neutral"], "构造的票应高度冗余"
    assert b["ev_ratio_sixth"] == pytest.approx(first_sixth), "EV 不随分布变化"
    assert b["ev_ratio_full"] == pytest.approx(first_full), "EV 不随分布变化"


def test_distribution_carries_honesty_note_and_prize_assumption(client):
    """票单分布必须自带"改变分布不改变 EV"提示条与奖金假设（前端不能用默认值补）。"""
    body = client.get("/api/portfolio/distribution?exact=false").json()
    honesty = body["honesty"]
    assert honesty["distribution_changes_ev"] is False
    assert "不改变期望值" in honesty["note_zh"]
    assert honesty["prize_assumption"], "必须连 prize_assumption 一起给"
    assert honesty["prize_assumption"] == body["distribution"]["prize_assumption"]


def test_distribution_exact_vs_iid_method_flag(client):
    """``exact=True/False`` 必须如实反映在 ``p_at_least_r_method`` 上（口径不可混）。"""
    exact = client.get("/api/portfolio/distribution?exact=true").json()["distribution"]
    approx = client.get("/api/portfolio/distribution?exact=false").json()["distribution"]
    assert exact["p_at_least_r_method"] in ("exact_enum", "iid_approx")
    assert approx["p_at_least_r_method"] == "iid_approx"


# ============================================================ 2) 前置注册端点

def test_registry_endpoint_ok_and_fields(client):
    """``GET /api/registry/status`` → 200，计数键与累计检验数齐全。"""
    r = client.get("/api/registry/status")
    assert r.status_code == 200
    body = r.json()
    _assert_jsonable(body, "registry")
    assert body["ok"] is True
    for key in ("registered", "passed", "provisional", "failed", "n_tested_cumulative",
                "items", "counts", "snapshot", "note_zh"):
        assert key in body, f"registry/status 缺少字段 {key}"
    assert isinstance(body["items"], list)
    assert isinstance(body["n_tested_cumulative"], int)
    assert "前置注册" in body["note_zh"]


def test_registry_endpoint_lists_prereg_reasons(client, isolated_env):
    """登记真实前置实验后，端点必须带回**每条规则的理由**与状态计数。

    这里**不**打桩任何路径：``hypothesis._registry_path()`` 从 ``config.DATA_DIR``
    派生（conftest 已把它隔离到临时目录），端点读的是同一个文件 —— 只有当端点
    真的走 ``load_registry()`` 时，本断言才可能通过。
    """
    from lottery import hypothesis

    reg_path = hypothesis._registry_path()
    assert str(reg_path).startswith(str(isolated_env.tmp)), \
        f"隔离失效：假说库落在 {reg_path}，本用例会写进真实 data/"
    hyps = [{
        "id": f"h{i}", "source": "stat", "status": "candidate",
        "statement": f"测试规律 {i}",
        "action": {"type": "prefer_red_filter", "filter": {
            "key": "odd_count", "params": {"k": i % 3 + 1}}},
    } for i in range(3)]
    hypothesis.register_prereg(hyps, registry_path=reg_path, issue="2026100",
                               evaluate_after_issue=2026100, reason="测试批次")

    body = client.get("/api/registry/status").json()
    _assert_jsonable(body, "registry")
    assert body["n_total"] >= 1, f"应至少列出一条已登记实验，实际 {body['items']}"
    item = body["items"][0]
    for key in ("exp_id", "prereg_status", "reason", "n_tested_cumulative",
                "p_cumulative", "divergence", "source"):
        assert key in item, f"prereg 条目缺少 {key}"
    assert item["prereg_status"] in hypothesis.PREREG_STATUSES
    assert item["reason"], "每条规则必须带 prereg 理由（前端要展示为什么不可用）"


# ============================================================ 3) ML 诊断端点

@_slow_only
def test_ml_diagnosis_endpoint_ok_and_honesty(client, with_draws):
    """``GET /api/ml/diagnosis`` → 200，含 trusted / tier / p_value 与诚实说明。"""
    r = client.get("/api/ml/diagnosis?max_numbers=1&n_null=5&max_issues=200")
    assert r.status_code == 200
    body = r.json()
    _assert_jsonable(body, "ml/diagnosis")
    assert body["ok"] is True
    assert isinstance(body["trusted"], bool)
    assert body["tier"] in ("signal_found", "monitor")
    chans = body["channels"]
    assert chans, "必须返回至少一个通道"
    for name, ch in chans.items():
        assert "trusted" in ch and "tier" in ch, f"{name} 缺少 trusted/tier"
        assert float(ch["p_value"]) == ch["p_value"], f"{name}.p_value 必须是原生 float"
    # 诚实边界：trusted=False 不是故障
    assert "trusted=False" in body["honesty_zh"]
    assert "不是故障" in body["honesty_zh"] or "不是 bug" in body["honesty_zh"]


def test_ml_diagnosis_without_data_is_graceful(client):
    """空库时不得 500：要么 400 带 error，要么 200 带 ok=False（前端能渲染原因）。"""
    r = client.get("/api/ml/diagnosis?max_numbers=1&n_null=5")
    assert r.status_code in (200, 400)
    body = r.json()
    _assert_jsonable(body, "ml/diagnosis-empty")
    if r.status_code == 400:
        assert body.get("ok") is False and body.get("error")
    else:
        assert "ok" in body


@_slow_only
def test_ml_diagnosis_cache_reuse(client, with_draws, monkeypatch):
    """同参数第二次调用命中进程内缓存（诊断一次 10~30s，页面刷新不该反复重算）。

    ⚠️ 缓存是**进程级**的，必须在本用例里清空：否则前面用例预热过的缓存会让
    "第一次调用"直接命中，断言失去意义（模块级状态跨用例泄漏，不是产品缺陷）。
    """
    from lottery import api_app
    monkeypatch.setattr(api_app, "_ML_DIAG_CACHE", {})
    params = "?max_numbers=1&n_null=5&max_issues=200"
    first = client.get("/api/ml/diagnosis" + params).json()
    second = client.get("/api/ml/diagnosis" + params).json()
    assert first.get("cached") is False
    assert second.get("cached") is True
    assert second["trusted"] == first["trusted"]


# ============================================================ 4) 样本并排 / LLM 对照臂

def test_samples_side_by_side_is_not_mixable(client):
    """两套样本并排：``mixable=False``，且**不得**出现任何跨来源合计字段。"""
    r = client.get("/api/samples/side-by-side")
    assert r.status_code == 200
    body = r.json()
    _assert_jsonable(body, "samples/side-by-side")
    assert body["mixable"] is False
    assert "禁止相加" in body["note_zh"]
    for side in ("live", "replay"):
        blk = body[side]
        for key in ("source", "n_issues", "n_tickets", "mde"):
            assert key in blk, f"{side} 缺少 {key}"
    # 顶层与两块内都不得有合计字段
    for banned in _BANNED_SUM_KEYS:
        assert banned not in body, f"返回体含跨来源合计字段 {banned!r}，违反 mixable=False"
    assert "combined" not in body["live"]
    assert "combined" not in body["replay"]


def test_llm_arm_endpoint_ok(client, with_draws):
    """``GET /api/llm/arm`` → 200，含 verdict / eligible 等决策字段（无数据也不 500）。"""
    r = client.get("/api/llm/arm")
    assert r.status_code == 200
    body = r.json()
    _assert_jsonable(body, "llm/arm")
    assert body["ok"] is True
    rep = body["report"]
    for key in ("verdict", "eligible", "n_issues", "reason_zh"):
        assert key in rep, f"llm_arm_report 缺少 {key}"
    # brief 摘要必须与 report 同源（前端徽标用 brief）
    assert body["brief"]["verdict"] == rep["verdict"]

    r2 = client.get("/api/llm/arm?brief=true")
    assert r2.status_code == 200
    assert "report" not in r2.json()


@_slow_only
def test_all_new_endpoints_have_no_numpy_objects(client, with_draws):
    """**统一闸门**：所有新端点返回体递归检查，不得含 np.float64 / np.ndarray。"""
    urls = ["/api/portfolio/distribution?exact=false",
            "/api/registry/status",
            "/api/ml/diagnosis?max_numbers=1&n_null=5&max_issues=200",
            "/api/llm/arm",
            "/api/samples/side-by-side"]
    for url in urls:
        r = client.get(url)
        assert r.status_code == 200, f"{url} 应 200，实际 {r.status_code}"
        body = r.json()
        bad = _collect_numpy(body)
        assert not bad, f"{url} 返回体含 numpy 对象：{bad[:5]}"
        _assert_jsonable(body, url)


# ============================================================ 5) 前端契约

def test_frontend_assets_version_bumped_together():
    """``?v=`` 必须升到 >= 20261018.5，且 app.js / style.css 版本一致。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    m_app = re.search(r"app\.js\?v=(\d{8})\.(\d+)", html)
    m_css = re.search(r"style\.css\?v=(\d{8})\.(\d+)", html)
    assert m_app and m_css, "index.html 缺少 ?v= 缓存击穿参数"
    app_ver = (int(m_app.group(1)), int(m_app.group(2)))
    css_ver = (int(m_css.group(1)), int(m_css.group(2)))
    assert app_ver == css_ver, "app.js 与 style.css 的 ?v= 必须同步升版本"
    assert app_ver >= (20261018, 5), f"V10-10 必须升 ?v=，实际 {app_ver}"
    # 旧版本号不得残留
    assert "?v=20261018.4" not in html


def test_cfgllm_checkbox_still_not_hardcoded_checked():
    """回归：``cfgLlm`` 仍不得硬编码 checked（V10-04b 的修复不能被 V10-10 冲掉）。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    m = re.search(r"<input[^>]*id=\"cfgLlm\"[^>]*>", html)
    assert m, "index.html 找不到 cfgLlm 复选框"
    assert "checked" not in m.group(0)


def test_three_cards_dom_and_renderers_exist():
    """三卡（+ 样本并排）的 DOM 锚点与渲染函数必须都在，否则前端只显示空白卡。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    for anchor in ("distBody", "preregBody", "mlDiagBody", "samplesBody",
                   "channelBadges"):
        assert f'id="{anchor}"' in html, f"index.html 缺少 {anchor} 锚点"
    for fn in ("renderDistribution", "renderRegistry", "renderMlDiagnosis",
               "renderSamplesSideBySide", "renderChannelBadges"):
        assert f"function {fn}(" in js, f"app.js 缺少 {fn}"
    # 三个数据源必须都真的被请求（不能只有 renderer 没有 fetch）
    for path in ("/api/portfolio/distribution", "/api/registry/status",
                 "/api/ml/diagnosis", "/api/samples/side-by-side"):
        assert path in js, f"app.js 未请求 {path}"


def test_frontend_renders_honesty_and_no_sum():
    """前端必须渲染诚实边界提示与「禁止相加」，且不得渲染合计行。"""
    js = _js_code()   # 剥离注释：「禁止相加」在注释里也出现
    assert "v10-honesty" in js, "缺少诚实边界提示条样式类"
    assert "改变分布不改变 EV" in js
    assert "trusted=False 是随机彩票的诚实结论" in js
    assert "禁止相加" in js
    assert "mixable" in js
    # CSS 必须定义了提示条与徽标样式（否则提示条不可见 = 等于省略）
    css = STYLE_CSS.read_text(encoding="utf-8")
    for cls in (".v10-honesty", ".ch-badge", ".v10-metrics", ".v10-side"):
        assert cls in css, f"style.css 缺少 {cls} 样式"


def test_vector_badge_shows_families_and_llm_can_pick():
    """通道徽标的数据源是 /api/info 的 channels（families + llm_can_pick）。"""
    js = APP_JS.read_text(encoding="utf-8")
    assert "renderChannelBadges(info && info.channels)" in js, \
        "徽标必须用 /api/info 的 channels 渲染（唯一判据）"
    assert "llm_can_pick" in js
    assert "FAMILY_LABEL" in js


@_slow_only
def test_ml_diagnosis_cache_key_includes_dataset_fingerprint(client, with_draws, monkeypatch):
    """缓存必须按**数据集**失效：新开奖入库后同参数不得返回旧的 trusted 结论。

    只按参数缓存的话，"新数据到了"仍会拿到 10 分钟前算出的结论——
    而结论正是这个端点唯一的产品价值，缓存把它钉死在旧样本上等于错报。
    """
    from lottery import api_app, db

    monkeypatch.setattr(api_app, "_ML_DIAG_CACHE", {})
    params = "?max_numbers=1&n_null=5&max_issues=200"
    first = client.get("/api/ml/diagnosis" + params).json()
    assert first["cached"] is False
    assert first["dataset_fingerprint"] == f"{db.count_draws()}:{db.max_issue()}"

    # 追加一期新开奖 → 指纹变化 → 必须重算而不是命中旧缓存
    db.upsert_draws([{
        "issue": "2013999",
        "date": "2013-12-31",
        "reds": [1, 2, 3, 4, 5, 6],
        "order": [1, 2, 3, 4, 5, 6],
        "blue": 7,
    }])
    second = client.get("/api/ml/diagnosis" + params).json()
    assert second["cached"] is False, "数据集变化后必须重算，不得命中旧缓存"
    assert second["dataset_fingerprint"] != first["dataset_fingerprint"]
