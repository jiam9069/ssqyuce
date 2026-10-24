"""V34：主页移除三张卡（前向回测 / 方差优化选号 / 本期规律使用情况）。

用户 2026-10-24 指示：
·「去掉主页的『本期规律使用情况』，既然无法找到规律，挖掘就没有意义了」
·「去掉主页的『前向回测』和『方差优化选号』」

## 守什么

1. **前端真的消失**：三张卡的容器、渲染函数、启动调用都不在
2. **后端不许趁机删掉**：`/api/rigor/walk_forward`、`/api/portfolio/variance-pick`
   仍须注册 —— 它们从「每日展示」降级为「按需审计」，不是废弃
3. **不留孤立调用**：删掉渲染函数后，不能有代码还在调它（会 ReferenceError）

## 为什么单独一个文件

三张卡分属 V7 / V23 / V27 三个不同交付，各自的测试文件守的是"当时的功能"。
本文件守的是"后来的移除"，两者语义相反，必须分开，否则改一处会撞另一处。

## 配额纪律

前端是 HTML/JS 无法 AST 解析，故把结构压成**一个快照**整体比对
（见 `_snapshot`），只消耗一次外部文件读取配额，失败时 diff 直接定位。
"""
from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]


#: 模块级只读一次（B 类守卫统计的是「read_text 子串断言」调用点数，
#: 集中读一次是它推荐的做法；避免 3 个用例各读一遍消耗配额）。
_HTML = (_ROOT / "web" / "index.html").read_text(encoding="utf-8")
_JS = (_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
_CODE = "\n".join(ln for ln in _JS.split("\n")
                   if not ln.lstrip().startswith("//"))


def _snapshot() -> dict:
    """把三张卡的移除情况压成一个可整体比对的事实字典。"""
    html, code = _HTML, _CODE

    removed_containers = [k for k in ("ruleCard", "walkForwardCard",
                                      "variancePickCard")
                          if f'id="{k}"' in html]
    removed_fns = [f for f in ("renderRuleUsage", "loadWalkForward",
                               "loadVariancePick")
                   if f"function {f}" in code or re.search(rf"\b{f}\s*\(", code)]
    orphan_ids = [i for i in ("wfBody", "wfTag", "vpBody", "vpTag",
                              "vpHonesty", "vpSeedNote", "vpRun", "vpN",
                              "ruleBody", "ruleTag")
                  if f'"#{i}"' in code]

    # 启动序列不得引用已删函数
    boot_ok = True
    i = code.find("loadRigor()")
    if i > 0:
        seg = code[max(0, i - 500):i + 800]
        boot_ok = not any(f in seg for f in removed_fns) if removed_fns else True

    return {"containers": removed_containers,
            "functions": removed_fns,
            "orphan_ids": orphan_ids,
            "boot_clean": boot_ok}


def test_homepage_removal_snapshot():
    """一个断言覆盖四件事：容器 / 函数 / 孤立 id / 启动序列。"""
    assert _snapshot() == {"containers": [], "functions": [],
                           "orphan_ids": [], "boot_clean": True}


def test_backend_endpoints_still_registered():
    """**关键**：移除的是展示，不是审计能力。后端路由必须仍在。"""
    from lottery import routes_portfolio as RP
    from lottery import routes_rigor as RR

    rigor = [r.path for r in RR.router.routes]
    port = [r.path for r in RP.router.routes]
    assert "/api/rigor/walk_forward" in rigor, (
        "前向回测端点必须保留（按需审计用）")
    assert "/api/portfolio/variance-pick" in port, (
        "方差优化端点必须保留（按需审计用）")


def test_backend_contract_unchanged(seeded_db):
    """后端契约不得因前端移除而缩水（参数级 + 返回值级，均可执行）。"""
    import inspect

    from lottery import routes_rigor as RR

    sig = inspect.signature(RR.rigor_walk_forward)
    for p in ("method", "color", "n_tickets", "start"):
        assert p in sig.parameters, f"walk_forward 缺参数 {p}"

    seeded_db(900, seed=7)
    r = RR.rigor_walk_forward(method="stat:freq", color="blue",
                              n_tickets=5, start=0, refresh=0)
    assert isinstance(r, dict), "应返回 dict（数据足够时）"
    for k in ("headline_zh", "honesty_zh", "z_min", "seeds"):
        assert k in r, f"walk_forward 契约缺 {k}"


def test_variance_pick_backend_still_works():
    """方差优化端点仍可调用（只是首页不再展示）。"""
    from lottery.routes_portfolio import portfolio_variance_pick
    r = portfolio_variance_pick(n_tickets=3, seed=1, iterations=300,
                                with_baseline=False, baseline_samples=5)
    assert r["ok"] is True
    assert len(r["tickets"]) == 3
    assert r["honesty"]["ev_unchanged"] is True
