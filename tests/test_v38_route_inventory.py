"""路由清单一致性（V38，2026-10-24 审计 P2）。

## 为什么需要

审计时探测 API 时发现：我按"看起来应该有"的路径去请求
（`/api/predictions`、`/api/data/draws`、`/api/draws/{id}` …），
全部返回 404 —— 但 404 来自"路由不存在"，而不是"参数被拒"。

这暴露了两件事：

1. **测试里没有一份可执行的 API 清单**，所以"前端调了个不存在的端点"
   这类缺陷只会在用户点下去时表现为静默 404（或前端 catch 后不提示）；
2. 反过来，后端删了某个端点、前端没跟上，同样无人发现。

本文件把"前端实际 fetch 的路径"与"app 真实注册的路由"对撞，
任何前端调用而路由不存在的情况都会变成一条可读的失败。

## 口径

* 只检查**静态字面量**路径（`fetch("/api/xxx")` / `api("/api/xxx")`）。
  动态拼接（模板字符串带变量）跳过 —— 无法静态判定，硬判会假红。
* 允许出现在 `_KNOWN_DYNAMIC` 里的前缀：那些确实是运行时拼的。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "web" / "static" / "app.js"
INDEX_HTML = ROOT / "web" / "index.html"

#: 前端静态字面量的提取模式。
#: app.js 里两种调用形态：`api("/api/x")`（带封装的 helper）与裸 `fetch("/api/x")`。
_PATH_RE = re.compile(r"""(?:\bapi|\bfetch)\(\s*["'`](/api/[^"'`\s$?]*)[?`"']""")

#: 确实由运行时拼接的前缀（无法静态判定，跳过）。
_KNOWN_DYNAMIC = (
    "/api/tasks/",          # `api('/api/tasks/' + id)`
    "/api/llm_lab/",        # `api('/api/llm_lab/' + name)`
    "/api/admin/users/",    # `api('/api/admin/users/' + userId[, '/reset-password'])`
)


def _registered_routes() -> set:
    """app 上真实注册的路径模板集合（含参数化形态）。

    ## 为什么用 OpenAPI 而不是遍历 app.routes

    当前 FastAPI 版本把 include_router 的结果包成 `_IncludedRouter` 结点，
    **外层拿不到 path**（实测 `app.routes` 里 20 个对象全是 path=None）。
    这正是"看起来在检查路由、实际一个都没检查"的典型陷阱 ——
    第一版就是这么写的，两个用例都红了却不是产品问题。

    `app.openapi()["paths"]` 是 FastAPI 自己算出来的真值（95 条），
    既包含参数化模板（`/api/tasks/{task_id}`），也不受内部结构变化影响。
    代价是触发一次 schema 构建（~10ms），对单测可接受。
    """
    from lottery import api_app
    schema = api_app.app.openapi()
    return set(schema.get("paths", {}).keys())


def _frontend_paths() -> set:
    paths = set()
    for f in (APP_JS, INDEX_HTML):
        if not f.exists():
            continue
        for m in _PATH_RE.finditer(f.read_text(encoding="utf-8")):
            paths.add(m.group(1))
    return paths


def _matches(path: str, templates: set) -> bool:
    """静态路径能否匹配某个注册模板（简单参数化处理 :id / {id}）。"""
    if path in templates:
        return True
    for t in templates:
        if "{" not in t and ":" not in t:
            continue
        # 把模板的参数段换成通配
        pat = re.sub(r"\{[^}]+\}", "[^/]+", t)
        pat = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "[^/]+", pat)
        if re.fullmatch(pat, path):
            return True
    return False


def test_frontend_api_calls_have_registered_routes():
    """前端 fetch 的每个静态 /api 路径，后端必须有对应路由。

    失败通常意味着：后端删/改了端点而前端没跟上（用户点下去静默 404），
    或者前端拼错了路径。
    """
    templates = _registered_routes()
    assert templates, "没取到任何路由 —— app 导入异常？"

    missing = []
    for p in sorted(_frontend_paths()):
        if any(p.startswith(pre) for pre in _KNOWN_DYNAMIC):
            continue
        if not _matches(p, templates):
            missing.append(p)
    assert not missing, (
        "前端调用了后端不存在的路径（会静默 404）：\n  "
        + "\n  ".join(missing)
        + "\n\n若这是新端点，请先在 routes_*.py 注册；"
          "若是路径拼错，请改前端。动态拼接的路径请加入 _KNOWN_DYNAMIC 并注释原因。")


def test_route_table_is_nonempty_and_has_core_endpoints():
    """核心端点必须存在 —— 防止"重构时顺手删掉"。

    这比"每个端点都测一遍"便宜得多，却能挡住最伤的那类删除。
    """
    templates = _registered_routes()
    core = [
        "/api/health",
        "/api/info",
        "/api/predict",
        "/api/capabilities",
        "/api/auth/login",
        "/api/auth/me",
        "/api/user/llm/config",
    ]
    missing = [c for c in core if c not in templates]
    assert not missing, f"核心端点缺失：{missing}"


def test_no_duplicate_operation_ids():
    """OperationId 不得在不同路径间重复（重复会让生成式客户端互相覆盖）。

    用 OpenAPI schema 里的 operationId 判定：同一 operationId 出现在
    **两个不同路径**上才是真问题（同一路径的 GET/POST 同名属 FastAPI
    对 `api_route(methods=[...])` 的已知行为，会告警但不影响契约）。
    """
    from lottery import api_app

    schema = api_app.app.openapi()
    seen: dict = {}
    for path, ops in schema.get("paths", {}).items():
        for method, op in ops.items():
            oid = op.get("operationId")
            if not oid:
                continue
            seen.setdefault(oid, set()).add(path)

    dupes = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    assert not dupes, (
        "这些 operationId 被不同路径复用，会让 OpenAPI 客户端生成互相覆盖：\n"
        + "\n".join(f"  {k}: {v}" for k, v in dupes.items()))
