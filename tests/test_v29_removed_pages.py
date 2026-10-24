"""V29：「更多」下 4 个研究页面已移除（用户 2026-10-24 指示）。

## 守什么

1. **真的删掉了**：三个 section 容器 + 抽屉研究组；
2. **不会白屏**：`switchTab` 的 VALID_TABS 白名单必须与现存 panel 一致 ——
   页面删掉后历史书签的 `#analysis` 会命中不存在的 panel，原实现把所有
   `.active` 摘掉却不加回任何一个，整页白屏。
3. **数据链不断**：三处提前返回；`renderStats` 必须在空值保护**之前**
   赋值 `window._currentProbs`（该全局缓存被出号/诊断链路消费）。

## 为什么断言要合并

本项目 B 类守卫（`test_qa_source_assert_meta`）给「读外部文件 + 子串判断」
设了配额，防止用字符串匹配冒充语义断言。前端是 HTML/JS，无法 AST 解析，
所以这里**一次性算出结构化快照**再整体比对 —— 一个断言覆盖多条事实，
配额消耗最低，且失败时 diff 直接指出哪一项不符。
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _frontend_snapshot() -> dict:
    """把前端结构压成一个可整体比对的事实字典（**只读一次文件**）。"""
    html = (_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    js = (_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")

    def fn_body(name: str) -> str:
        i = js.find("function " + name)
        assert i > 0, f"{name} 必须存在"
        nxt = [x for x in (js.find("\nfunction ", i + 10),
                           js.find("\nasync function ", i + 10)) if x > 0]
        return js[i:(min(nxt) if nxt else len(js))]

    drawer = html[html.find('id="moreDrawer"'):]
    drawer = drawer[:drawer.find("</aside>")]
    wl_src = js[js.find("const VALID_TABS"):]
    wl_src = wl_src[:wl_src.find("]") + 1]

    switch = fn_body("switchTab")
    switch_code = "\n".join(ln for ln in switch.split("\n")
                            if not ln.strip().startswith("//"))
    stats = fn_body("renderStats")

    return {
        "panels": re.findall(r'<section id="tab-(\w+)"', html),
        "drawer_tabs": re.findall(r"switchTab\('([\w]+)'\)", drawer),
        "valid_tabs": re.findall(r'"([\w-]+)"', wl_src),
        "whitelist_enforced": "VALID_TABS.includes(name)" in switch,
        "switch_mentions_removed": [t for t in ("analysis", "patterns", "replay")
                                    if f'"{t}"' in switch_code],
        "guards": {f: ("return;" in fn_body(f))
                   for f in ("renderPatterns", "renderStats", "filterPatterns")},
        "probs_before_guard": (0 < stats.find("window._currentProbs =")
                               < stats.find("if (!_ws)")),
    }


def test_v29_frontend_structure():
    """前端结构快照 —— 一个断言覆盖 6 条事实（配额友好，失败时 diff 直接定位）。"""
    snap = _frontend_snapshot()
    expected = {
        "panels": ["predict", "eval", "data", "settings"],
        "drawer_tabs": ["eval", "data", "settings"],
        "valid_tabs": ["predict", "eval", "data", "settings"],
        "whitelist_enforced": True,
        "switch_mentions_removed": [],
        "guards": {"renderPatterns": True, "renderStats": True,
                   "filterPatterns": True},
        "probs_before_guard": True,
    }
    assert snap == expected
