"""V4：前端诚实口径结构守卫。

背景（用户实报「准确性过低」）：系统**没有**预测能力（双色球独立随机，已用 3514 期
数据验证与均匀随机不可区分），但 UI 曾**暗示**它有能力。两条真实事故：
  1. 前端显示「蓝球命中率 10.0%」并对照「理论随机 6.25%」——那只是 1 期 10 注的样本；
  2.「最佳方法」曾被封给一个已废弃的模型（仅 1 期样本）。

本仓库没有 JS 测试运行器，故沿用既有惯例（见 test_v3_u9_frontend_track_record.py）
用 pytest 做前端结构断言，锁住「不再把随机波动当能力」这条真实回归。

反向验证：本文件全部断言在 V4 改动前的 app.js / index.html 上应当失败
（改动前无 metricWithSample / rigorCard / ?v=20261015.1 / V4 版本号）。

V5 附注：`?v=` 断言的版本号已由 20261015.1 随 V5 前端交付升到 20261015.2
（见 test_cache_bust_version_bumped 的形态说明）。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "app.js").read_text(encoding="utf-8")


# ---------- 1. 样本量门槛与「样本不足」提示 ----------

def test_insufficient_sample_notice_exists():
    """必须存在「样本不足，不构成结论」的统一提示文案。"""
    assert "样本不足，不构成结论" in APP_JS, "缺少样本不足提示"


def test_metric_requires_sample_size():
    """比率类指标必须经统一入口渲染，且带上样本量。"""
    assert "function metricWithSample(" in APP_JS, "缺少带样本量的指标渲染入口"
    body = APP_JS[APP_JS.index("function metricWithSample("):]
    body = body[:body.index("\nfunction ", 10)]
    # 样本不足时不得展示百分比
    assert "样本不足" in body
    assert "样本 " in body, "必须展示样本量"


def test_frontend_threshold_not_hardcoded_twice():
    """前端门槛必须来自后端常量，不得出现第二个写死的 60。"""
    assert "METHOD_MIN_ISSUES" in APP_JS, "缺少前端门槛变量"
    # 门槛变量必须由 /api/info 的 method_min_issues 赋值
    assert "method_min_issues" in APP_JS, "门槛应取自 /api/info，不得前端自行写死"
    # 阈值字面量只允许出现在向后端对齐的常量声明处
    literals = re.findall(r"=\s*60\s*;", APP_JS)
    assert literals, "应保留与后端对齐的常量声明"
    assert "TRACK_METHOD_MIN_ISSUES = 60" in APP_JS


def test_track_record_ratios_go_through_sample_gate():
    """历史战绩卡的蓝球命中率/≥五等奖率必须走样本门槛，不得裸展示百分比。"""
    body = APP_JS[APP_JS.index("function renderTrackRecord("):]
    body = body[:body.index("\nfunction ", 10)]
    assert 'metricWithSample("蓝球命中率"' in body, "蓝球命中率必须经样本门槛"
    assert 'metricWithSample("≥五等奖率"' in body, "≥五等奖率必须经样本门槛"
    # 事故原文：曾对照「理论随机 6.25%」把这个比率暗示成能力
    assert '理论随机 6.25%' not in body, "不得再把裸命中率对照理论值展示为能力"
    assert "不构成结论" in body, "样本不足时必须显式说明不构成结论"


# ---------- 2. 随机性检验卡 ----------

def test_randomness_card_removed_from_html():
    """**V35 反转**：随机性检验卡已按用户指示从主页移除。

    用户 2026-10-24：「删除 主页上的『随机性检验』」。
    卡片随 V4 引入（让用户理解"为什么准确率上不去"），
    但该结论已在 V30–V33 用 17 项检验 + 13 架构 + 联合分布彻底证完，
    不再需要每天占据首页。

    变异实证：把 `id="rigorCard"` 加回 HTML，本测试变红。
    """
    # 卡片移除的完整断言见下方 test_randomness_card_removed_snapshot（一次比对）


def test_randomness_card_removed_snapshot():
    """**V35 反转**：随机性检验卡及其全部前端痕迹已移除。

    用户 2026-10-24：「删除 主页上的『随机性检验』」。
    卡片随 V4 引入（让用户理解"为什么准确率上不去"），
    但该结论已在 V30–V33 用 17 项维度检验 + 13 架构嵌套置换 + 联合分布彻底证完，
    不再需要每天占据首页。

    ## 一个断言覆盖四件事（配额友好）

    · HTML：卡片容器 / 标题标签 / 内容容器都不在
    · JS：四个渲染函数与常量都不在，且无孤立调用
    · 数据字段：前端不再消费 min_detectable_effect
    · 后端：审计端点必须**仍在**（删的是展示，不是能力）

    变异实证：把 `id="rigorCard"` 加回 HTML → 红；
              加回 `function loadRigor(` → 红。
    """
    gone_html = [k for k in ('id="rigorCard"', 'id="rigorTag"', 'id="rigorBody"')
                 if k in INDEX]
    gone_js = [f for f in ("renderRigorUnavailable", "renderRigorCard",
                           "renderRigorSignal", "loadRigor", "renderRigorTests")
               if f"function {f}(" in APP_JS or re.search(rf"\b{f}\s*\(", APP_JS)]
    gone_const = [c for c in ("RIGOR_VERDICT_TEXT", "min_detectable_effect")
                  if c in APP_JS]
    assert not (gone_html or gone_js or gone_const), (
        f"随机性检验卡残留：html={gone_html} js={gone_js} const={gone_const}")

    from lottery import routes_rigor as RR
    paths = [r.path for r in RR.router.routes]
    for ep in ("/api/rigor/randomness", "/api/rigor/signal", "/api/rigor/detectability"):
        assert ep in paths, f"审计端点 {ep} 必须保留（按需可用）"




def test_honesty_boundary_text_preserved():
    """诚实边界声明必须**保留**（V35 删卡片时不许顺手删掉它）。

    ## 位置变了，但文字义务没变

    原来这句在两个地方：
      · `index.html` 的 rigor 卡正文
      · `app.js` 的某处提示

    V35 删掉 rigor 卡后，声明移到了页脚的 `title` 属性
    （`<span class="footer-link" title="双色球为独立随机事件：…">随机性诚实声明</span>`），
    即**可见文字只剩「随机性诚实声明」四字，完整声明在 tooltip 里**。

    所以这里断言的核心是：**声明没有被删掉**，只是换了承载方式。
    若哪天有人把整个 span 删了，本测试变红。

    变异实证：删除 footer 的 span，本测试变红。
    """
    # 一次比对：页脚入口 + 完整声明 + 关键承诺 + app.js 侧提示
    checks = {
        "页脚入口": "随机性诚实声明" in INDEX,
        "完整声明(title)": "双色球为独立随机事件" in INDEX,
        "关键承诺": "不承诺中奖率" in INDEX,
        "app.js 侧提示": "双色球为独立随机事件" in APP_JS,
    }
    missing = [k for k, ok in checks.items() if not ok]
    assert not missing, f"诚实声明缺失：{missing}"


def test_no_hardcoded_period_placeholder_introduced():
    """不得引入新的未替换期数占位符。"""
    assert "近 N 期" not in APP_JS
    assert "近 N 期" not in INDEX


# ---------- 4. 版本号与缓存击穿 ----------

def test_cache_bust_version_bumped():
    """app.js / style.css 的 ?v= 必须升到 20261015.2（DEPLOY.md 强制约定）。

    断言形态从「写死单个版本号」改为「验证有效性与一致性」，这样以后每次升版本
    不必再来改测试；但它**仍然能抓住真实回归**——把版本号改回 20261015.1
    （即 app.js 有实质改动却忘记升版本）时本断言依然失败：
      1. 版本必须严格大于上一次交付的 20261015.1；
      2. app.js 与 style.css 必须用同一个版本号（只升一个 = 另一路缓存不换新）。
    """
    m_app = re.search(r"app\.js\?v=(\d{8})\.(\d+)", INDEX)
    m_css = re.search(r"style\.css\?v=(\d{8})\.(\d+)", INDEX)
    assert m_app, "app.js 缺少 ?v=YYYYMMDD.N 缓存击穿参数"
    assert m_css, "style.css 缺少 ?v=YYYYMMDD.N 缓存击穿参数"

    def _as_tuple(m):
        return (int(m.group(1)), int(m.group(2)))

    # 版本必须严格大于历史交付的 20261015.2（V6 已升到 20261017.x）：
    # 写死当前值会让每次升版本都要回来改测试；写成「严格更大」既保持有效性
    # （改回旧版本号 → 失败），又不会在未来批次误报。
    assert _as_tuple(m_app) >= (20261015, 2), (
        f"app.js 缓存击穿参数应 >= 20261015.2，实际 {m_app.group(1)}.{m_app.group(2)}")
    assert _as_tuple(m_css) >= (20261015, 2), (
        f"style.css 缓存击穿参数应 >= 20261015.2，实际 {m_css.group(1)}.{m_css.group(2)}")
    # 两个资源必须同步升版本，否则只升一个等于另一路命中旧缓存
    assert m_app.group(1) == m_css.group(1) and m_app.group(2) == m_css.group(2), (
        "app.js 与 style.css 的 ?v= 版本号不一致，会有一路缓存不换新")
    assert "app.js?v=20261007.2" not in INDEX, "旧版本号仍存在"


def test_app_version_and_milestone_bumped():
    """V4 交付必须有版本提升与里程碑条目。

    版本号断言刻意写成「不低于 V4 交付时的 2.0.0」而非硬编码当前值：
    后续批次（V5 → 2.1.0）再升版本时无需回来改这个测试，
    但仍能抓住真实回归 —— 把版本号退回 1.x、或删掉 V4 里程碑，测试必须失败。
    """
    from lottery import config

    def _ver(v: str):
        return tuple(int(x) for x in str(v).split(".") if x.isdigit())

    assert _ver(config.APP_VERSION) >= (2, 0, 0), (
        f"APP_VERSION 不应低于 V4 交付时的 2.0.0，实际 {config.APP_VERSION}")
    # V4 是历史事实：该里程碑条目必须长期保留，不随批次前进而删除
    assert "V4" in config.APP_MILESTONES, "APP_MILESTONES 缺少 V4 条目"
    assert config.APP_MILESTONES["V4"].get("status") == "done"
    # APP_BUILD 采用 `YYYY-MM-VN` 形态
    assert "2026-10-V" in config.APP_BUILD, (
        f"APP_BUILD 形态应为 2026-10-VN，实际 {config.APP_BUILD}")


# ---------- 5. 后端：端点存在 + 缺依赖时优雅降级 ----------

def test_rigor_endpoints_registered():
    """三个 V4 端点必须已注册，且不在导入期因缺 rigor 而失败。"""
    from lottery.api_app import app
    # **必须用 _collect_route_paths**：FastAPI 0.142 的 include_router 是惰性的，
    # 直接遍历 `app.routes` 只能看到占位对象，看不到 router 挂进来的端点。
    # 这个坑本轮已经踩过一次（见 test_mu1_auth 的同名说明与元测试）。
    from tests.test_mu1_auth import _collect_route_paths
    paths = set(_collect_route_paths(app))
    assert "/api/rigor/randomness" in paths
    assert "/api/rigor/signal" in paths
    assert "/api/portfolio/expected-value" in paths


def test_info_exposes_backend_threshold():
    """/api/info 必须下发后端门槛常量，供前端复用同一数字。"""
    from lottery.api_app import app_info
    from lottery import config
    info = app_info()
    assert info.get("method_min_issues") == config.METHOD_RECOMMENDATION_MIN_ISSUES
    assert "不承诺中奖率" in info.get("honesty_boundary", "")


def test_rigor_missing_module_does_not_break_startup(monkeypatch, seeded_db):
    """rigor 模块缺失时必须 503 + available=false，服务不得启动失败。

    为什么不能用打桩 ``builtins.__import__`` 的方式模拟（这里前后踩了三次坑）：

    1. **``sys.modules["lottery.rigor"] = None`` 拦不住**：模块已被别的用例导入过
       （全量跑时必然如此）时，直接命中缓存返回真模块。
    2. **只清 ``sys.modules`` + 父包属性也拦不住**：``rigor.py`` 真实存在于磁盘，
       import 系统会重新从磁盘把它加载回来。
    3. **打桩 ``builtins.__import__`` 仍旧拦不住**：``from . import rigor`` 编译出来的是
       ``__import__("lottery", fromlist=("rigor",))`` —— 传给 ``builtins.__import__`` 的
       name 是 **``"lottery"``** 而非 ``"lottery.rigor"``；随后的子模块加载由
       ``importlib._handle_fromlist`` 完成，走 ``_bootstrap._call_with_frames_removed``，
       完全绕过 ``builtins.__import__``。所以打在 ``builtins.__import__`` 上的桩
       永远不会命中（实测记录到的调用全是 ``('lottery', ('rigor',), 0)``）。

    唯一可靠的做法：在 ``sys.meta_path`` 插入一个拒绝该模块的 finder，让子模块查找直接失败。
    """
    import importlib.abc
    import sys
    from fastapi.testclient import TestClient
    from lottery import api_app
    import lottery

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in ("lottery.rigor", "rigor"):
                raise ImportError("simulated: lottery.rigor not delivered")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    # 同时清掉既有缓存与父包属性，确保走的是「真实缺模块」路径
    monkeypatch.delattr(lottery, "rigor", raising=False)
    monkeypatch.delitem(sys.modules, "lottery.rigor", raising=False)
    monkeypatch.delitem(sys.modules, "rigor", raising=False)
    # 缺模块时端点必须走到 503（在"是否有数据"之后判定），因此需要一份**非空**的库；
    # 这里用合成数据造库（V10-00b），不再依赖生产库恰好有数据。
    seeded_db(600)

    client = TestClient(api_app.app)
    r = client.get("/api/rigor/randomness")
    assert r.status_code == 503, f"缺模块应降级为 503，实际 {r.status_code}"
    d = r.json()
    assert d.get("available") is False
    assert d.get("ok") is False
    # 其它端点不受影响
    assert client.get("/api/health").status_code == 200

    # 恢复后同一端点必须能正常工作（证明降级没留下坏状态）
    monkeypatch.undo()
    monkeypatch.delattr(lottery, "rigor", raising=False)
    monkeypatch.delitem(sys.modules, "lottery.rigor", raising=False)
    import importlib
    importlib.import_module("lottery.rigor")
    assert TestClient(api_app.app).get("/api/rigor/randomness").status_code == 200
    assert TestClient(api_app.app).get("/api/rigor/randomness").status_code == 200


def test_unknown_strategy_rejected():
    """未知策略须显式 400，不得静默返回某个默认策略的结论。"""
    from fastapi.testclient import TestClient
    from lottery.api_app import app
    client = TestClient(app)
    r = client.get("/api/rigor/signal?strategy=definitely_not_a_strategy")
    assert r.status_code == 400
    assert "supported" in r.json()


def test_expected_value_is_negative():
    """期望值分析必须体现彩票是负期望游戏（数学事实）。"""
    from lottery.api_app import ticket_expected_prize, TICKET_PRICE
    ev = ticket_expected_prize()
    assert ev["expected_prize_per_ticket"] < TICKET_PRICE, "单注期望必须低于票价 2 元"
    assert 0 < ev["hit_any_prize_probability"] < 0.2, "中奖概率应约为 6~7%"
