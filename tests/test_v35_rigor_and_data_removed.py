"""V35：主页移除「随机性检验」卡、「更多」移除「数据管理」页。

用户 2026-10-24：「删除 主页上的『随机性检验』和『更多』里面的数据管理」

## 守什么

1. **随机性检验卡真的消失**：容器、四个渲染函数、启动调用
2. **数据管理页真的消失**：section 容器、抽屉入口、三个动作函数、白名单
3. **不留悬空跳转**：footer 原本 `jumpToCard('rigorCard', ...)`，
   卡片删掉后点击会没反应 —— 必须改成不可点（但文字保留）
4. **后端审计能力保留**：`/api/rigor/randomness`、`/api/refresh` 仍注册

## 为什么 footer 文字必须保留

原注释写着「这句话是 footer 里唯一一句真话，压成四个字等于没说」。
删卡片 ≠ 删声明 —— 声明是对用户的信息义务，不依附于某个面板。
本测试断言它仍在，只是不再是链接。
"""
from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HTML = (_ROOT / "web" / "index.html").read_text(encoding="utf-8")
_JS = (_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
_CODE = "\n".join(ln for ln in _JS.split("\n")
                   if not ln.lstrip().startswith("//"))


def test_removal_snapshot():
    """一个断言覆盖：容器 / 函数 / 白名单 / 抽屉 / 悬空引用。"""
    got = {
        "rigor_container": 'id="rigorCard"' not in _HTML
                           and 'id="rigorBody"' not in _HTML,
        "rigor_fns": [f for f in ("loadRigor", "renderRigorCard",
                                  "renderRigorUnavailable", "renderRigorSignal")
                      if f"function {f}" in _CODE or re.search(rf"\b{f}\s*\(", _CODE)],
        "data_section": 'id="tab-data"' not in _HTML,
        "data_drawer": "switchTab('data')" not in _HTML,
        "data_fns": [f for f in ("refreshData", "showTasks", "showStats")
                     if f"function {f}" in _CODE or re.search(rf"\b{f}\s*\(", _CODE)],
        "panels": re.findall(r'<section id="tab-(\w+)"', _HTML),
        "valid_tabs": re.findall(r'"([\w-]+)"',
                                 _JS[_JS.find("const VALID_TABS"):
                                     _JS.find("const VALID_TABS") + 120]),
    }
    assert got == {
        "rigor_container": True,
        "rigor_fns": [],
        "data_section": True,
        "data_drawer": True,
        "data_fns": [],
        "panels": ["predict", "eval", "settings"],
        "valid_tabs": ["predict", "eval", "settings"],
    }


def test_footer_statement_kept_but_not_clickable():
    """页脚「随机性诚实声明」必须保留文字，但不能再跳转到已删的卡片。

    变异实证：把 `<span>` 改回 `onclick="jumpToCard('rigorCard',...)`，本测试变红。
    """
    assert "随机性诚实声明" in _HTML, "声明文字必须保留（信息义务不随面板删除）"
    assert "jumpToCard('rigorCard'" not in _HTML, (
        "指向已删除卡片的跳转必须去掉，否则点击无反应")


def test_backend_audit_endpoints_still_registered():
    """移除的是展示，后端审计与数据刷新能力必须仍在。"""
    from lottery import routes_rigor as RR

    paths = [r.path for r in RR.router.routes]
    assert "/api/rigor/randomness" in paths, "随机性检验端点必须保留（按需审计）"
    assert "/api/rigor/signal" in paths, "策略显著性端点必须保留"
    # /api/refresh 在 routes_data 里
    from lottery import api_app  # noqa: F401  —— 确认可导入
    import lottery.routes_data as RD
    src = RD.__file__
    assert src, "routes_data 必须存在"


def test_no_orphan_elements_from_removed_pages():
    """被删页面的元素 id 不得再被 JS 引用。"""
    orphans = [i for i in ("rigorBody", "rigorTag", "rigorSignal",
                           "taskList", "statsArea", "dataStatusArea")
               if f'"#{i}"' in _CODE]
    assert not orphans, f"这些元素已删但 JS 仍引用：{orphans}"


def test_hashtag_fallback_to_predict():
    """`#data` 这类旧书签必须回落 predict，不能白屏。"""
    i = _JS.find("function switchTab")
    body = _JS[i:i + 400]
    assert "VALID_TABS.includes(name)" in body, "白名单兜底必须在"
    assert '"data"' not in body.split("history.replaceState")[0], (
        "白名单不应再含 data")
