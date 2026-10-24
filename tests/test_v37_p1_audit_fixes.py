"""V37（2026-10-24 审计 P1 批次）回归锁。

本批修的是**安全与工程**类 P1，与 V36 的 P0 批次分开成文件，
因为它们的失效方式不同：P0 是"算错了"，P1 是"边界没守住"。

每条断言都对着**外部可观察行为**（HTTP 状态码 / 返回值 / 并发计数），
不依赖实现内部细节 —— 上一批的教训是"注释写着一致，但没有用例钉住"。
"""
from __future__ import annotations

import threading

import pytest


# ===========================================================================
# P1-1：登录等 scrypt 路径不得阻塞事件循环
# ===========================================================================

def test_login_handlers_are_not_running_scrypt_inline():
    """`auth_api` 里所有 scrypt 调用必须经 `_offload`。

    为什么不用"测耗时"来断言：耗时时序在 CI 上不稳定（机器负载一变就假红）。
    这里改为**静态+行为**双保险：
      ① 静态：确认 `_offload` 存在且被用于每个哈希调用点；
      ② 行为：直接在事件循环里调一次，确认它没把 loop 卡到无法插入其它任务。

    第 ② 条用一个"并发心跳协程"量最大间隔：若 scrypt 在协程体内同步执行，
    心跳会被冻结 ~150-550ms；offload 之后心跳应能在此期间继续推进。
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "lottery" / "auth_api.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    # `_offload` 必须作为**函数定义**存在（AST 断言，不做文本子串 ——
    # 文本匹配会被注释/docstring 骗，本项目已因此栽过）
    assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "_offload" for n in tree.body), \
        "P1-1 的 _offload 辅助函数不见了"
    text = src.read_text(encoding="utf-8")

    # 收集所有 `auth.hash_password / verify_password / set_password / create_user`
    # 的 Call 节点，确认它们的父链上有 `_offload`（作为实参出现）或本身被 await 包着。
    risky = {"hash_password", "verify_password", "set_password", "create_user"}
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == "auth" and fn.attr in risky):
            line = text.splitlines()[node.lineno - 1]
            # 允许两种写法：`await _offload(auth.xxx, ...)` 或 ash 出现在
            # `_offload(` 的实参里（跨行时该行可能不含 await）
            window = "\n".join(text.splitlines()[max(0, node.lineno - 4):node.lineno + 3])
            if "_offload" not in window:
                bad.append(f"{src.name}:{node.lineno} {fn.attr} 未 offload")
    assert not bad, "这些 scrypt 调用仍在事件循环里同步执行：\n" + "\n".join(bad)


def test_offload_does_not_block_event_loop():
    """行为验证：`_offload` 里的阻塞函数不会冻结事件循环。

    用 `asyncio.run` 而非 `pytest.mark.asyncio`：本项目**没有**装
    pytest-asyncio/pytest-anyio，标了 asyncio 的用例会直接报
    "async def functions are not natively supported"。为了让守卫在任何
    环境都生效，这里自己起一个 loop。
    """
    import asyncio
    import time

    from lottery import auth_api

    async def main():
        ticks = []

        async def ticker(stop):
            prev = time.perf_counter()
            while not stop.is_set():
                ticks.append(time.perf_counter() - prev)
                prev = time.perf_counter()
                await asyncio.sleep(0.005)

        def slow():
            time.sleep(0.30)          # 模拟一次 scrypt 的阻塞时长
            return "done"

        stop = threading.Event()
        t = asyncio.create_task(ticker(stop))
        t0 = time.perf_counter()
        got = await auth_api._offload(slow)
        elapsed = time.perf_counter() - t0
        stop.set()
        await t
        return got, elapsed, ticks

    got, elapsed, ticks = asyncio.run(main())

    assert got == "done"
    assert elapsed >= 0.28, "offload 竟然没真的执行函数？"
    assert ticks, "心跳没有推进 —— 说明整段被阻塞了"
    assert max(ticks) < 0.20, (
        f"事件循环被冻结 {max(ticks)*1000:.0f}ms —— offload 没生效"
        f"（阻塞函数仍在 loop 里跑）")


# ===========================================================================
# P1-2：SSRF —— base_url 校验
# ===========================================================================

SSRF_VECTORS = [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:6379",
    "http://localhost:8080/v1",
    "http://10.0.0.5/v1",
    "http://192.168.1.1/v1",
    "http://172.16.0.1/v1",
    "http://100.64.1.1/v1",
    "http://[::1]:8080/v1",
    "http://[::ffff:127.0.0.1]/v1",
    "http://metadata.google.internal/",
    "file:///etc/passwd",
    "gopher://127.0.0.1:6379/_x",
    "",
    "not-a-url",
]


@pytest.mark.parametrize("url", SSRF_VECTORS)
def test_ssrf_vectors_are_rejected(url):
    from lottery import url_guard
    ok, why = url_guard.check_llm_base_url(url)
    assert not ok, f"未拦截 SSRF 目标：{url}"
    assert why, "拒绝时必须给出可执行原因（项目 D20 纪律）"


def test_public_llm_endpoints_are_allowed():
    from lottery import url_guard
    for url in ("https://api.openai.com/v1", "https://api.deepseek.com/v1"):
        ok, why = url_guard.check_llm_base_url(url)
        assert ok, f"误拦了正常公网端点 {url}: {why}"


def test_private_url_allowed_only_with_explicit_optin(monkeypatch):
    """自建内网 LLM 必须由环境变量显式放行，默认一律拒。"""
    from lottery import url_guard
    monkeypatch.delenv("LOTT_LLM_ALLOW_PRIVATE_URL", raising=False)
    assert not url_guard.check_llm_base_url("http://10.0.0.5/v1")[0]
    assert not url_guard.allow_private_from_env()

    monkeypatch.setenv("LOTT_LLM_ALLOW_PRIVATE_URL", "1")
    assert url_guard.allow_private_from_env()
    assert url_guard.check_llm_base_url(
        "http://10.0.0.5/v1", allow_private=True)[0]
    # 即便放行内网，协议仍必须合法
    assert not url_guard.check_llm_base_url(
        "file:///etc/passwd", allow_private=True)[0]


def test_upstream_echo_is_redacted():
    """上游响应体不得原样回传（否则 SSRF 从"盲打"变"可读响应"）。"""
    from lottery import llm_client

    # 内网探测的典型响应：既非 JSON 也不是给人看的
    assert "ami-id" not in llm_client._redact_echo("ami-id\nami-launch-index\n")
    assert "internal" not in llm_client._redact_echo("<html><body>internal</body></html>")
    assert "Redis" not in llm_client._redact_echo("-DENIED Redis is running")
    # 但结构化错误必须保留（用户排查自己配置需要它）
    assert "invalid api key" in llm_client._redact_echo(
        '{"error":{"message":"invalid api key"}}')


# ===========================================================================
# P1-3：角色变更失效会话
# ===========================================================================

def test_role_change_revokes_sessions(seeded_db):
    """降权必须撤销该用户全部会话（与改密/停用同口径）。"""
    from lottery import auth

    u = auth.create_user("revoketest", "RevokePassw0rd", role="admin")
    sess = auth.create_session(u["user_id"], ip="127.0.0.1", user_agent="t")
    assert auth.resolve_session(sess["session_id"]) is not None

    auth.revoke_user_sessions(u["user_id"], reason="role_user")
    auth.invalidate_session_cache()
    assert auth.resolve_session(sess["session_id"]) is None, (
        "会话没有被撤销 —— 降权后旧会话仍带着 role=admin 的缓存快照")


def test_admin_patch_role_calls_revocation():
    """静态守卫：`admin_patch_user` 改 role 的分支必须调用撤销。"""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "lottery" / "auth_api.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))

    fn = next(n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "admin_patch_user")

    # 找出 `if "role" in payload:` 那个 If 结点，在其 body 里找两个调用。
    # 用 AST 而不是切片文本：切片依赖"下一个 if 的字符串长什么样"，
    # 顺序一改就静默失效（本项目 __source_assert 就是为了根治这类假绿）。
    role_if = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # 匹配 `"role" in payload` 与 `payload.__contains__("role")` 两种等价写法。
        # 注意 `"role" in payload` 的常量在 **left** 侧（Compare.left），
        # 第一版只查 comparators 侧，于是永远找不到 —— 断言报的是
        # "找不到分支"，看起来像产品改了，实际是我的匹配写窄了。
        if isinstance(test, ast.Compare) and len(test.comparators) == 1:
            cands = [test.left, test.comparators[0]]
            if any(isinstance(c, ast.Constant) and c.value == "role"
                   for c in cands):
                role_if = node
                break
    assert role_if is not None, "找不到 `if \"role\" in payload:` 分支"

    called = set()
    for node in ast.walk(role_if):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                called.add(f.attr)
            elif isinstance(f, ast.Name):
                called.add(f.id)
    assert "revoke_user_sessions" in called, (
        "改 role 的分支里没有 revoke_user_sessions —— P1-3 回归了")
    assert "invalidate_session_cache" in called, (
        "改 role 后没清会话缓存 —— 30s 内旧快照仍写着 admin")


# ===========================================================================
# P1-4：文档端点关闭时不得回 401
# ===========================================================================

def test_docs_endpoints_404_when_disabled(monkeypatch):
    """关闭文档时 `/docs` 必须 404（不存在），而不是 401（存在但没登录）。

    401 会告诉扫描器"这个路径存在"，而真实情况是它没被注册。
    """
    from fastapi.testclient import TestClient
    from lottery import config, api_app

    monkeypatch.setattr(config, "API_DOCS_ENABLED", False, raising=False)
    c = TestClient(api_app.app)
    for p in ("/docs", "/openapi.json", "/redoc"):
        code = c.get(p).status_code
        assert code == 404, f"{p} 返回 {code}，应 404（不该泄露存在性）"


# ===========================================================================
# P1-9：配置默认值一致
# ===========================================================================

def test_scheduler_default_matches_compose():
    """代码默认、compose 默认、README 必须说同一件事。"""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    # config 侧：用 AST 找 `os.environ.get("LOTT_SCHEDULER", "1")` 这个真实比较，
    # 而不是匹配源码文本（文本会被注释里的同名字符串骗）。
    tree = ast.parse((root / "lottery" / "config.py").read_text(encoding="utf-8"))
    found_default = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "get"):
            continue
        args = [a for a in node.args if isinstance(a, ast.Constant)]
        if len(args) >= 2 and args[0].value == "LOTT_SCHEDULER":
            found_default = args[1].value
            break
    assert found_default == "1", (
        f"config 里 LOTT_SCHEDULER 的默认值是 {found_default!r}，应为 '1' "
        "—— 与 compose 漂移（P1-9）")

    # compose 侧：YAML 里没有 AST，只能文本匹配；但先剥掉注释再比，
    # 避免"注释里写着 LOTT_SCHEDULER:-1"就算通过。
    compose_raw = (root / "docker-compose.yml").read_text(encoding="utf-8")
    compose_code = "\n".join(
        ln for ln in compose_raw.splitlines() if not ln.lstrip().startswith("#"))
    assert "LOTT_SCHEDULER:-1" in compose_code, "compose 默认不是 1"


# ===========================================================================
# P1-10：后台任务线程的连接必须关闭
# ===========================================================================

def test_bg_task_closes_thread_connection():
    """静态守卫：`_bg_run` 的 finally 里必须 db.close()。"""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "lottery" / "bg_tasks.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_bg_run")

    # 在 `_bg_run` 的 AST 里找 `db.close()` 调用（AST 不看注释与 docstring，
    # 所以"注释里写了 db.close()"不会被误判为已修）
    has_close = False
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "close"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "db"):
            has_close = True
            break
    assert has_close, (
        "后台一次性线程没有关闭自己的连接（P1-10）—— "
        "thread-local 连接会滞留 fd 并拖慢 WAL checkpoint")


def test_bg_run_completes_and_marks_task(seeded_db):
    """行为验证：任务能正常完成（加 finally 后不能把正常路径打断）。"""
    import time

    from lottery import bg_tasks, db

    tid = bg_tasks._bg_run("probe", lambda progress: {"note_zh": "ok"})
    for _ in range(100):
        t = db.load_task(tid)
        if t and t.get("status") in ("completed", "failed"):
            break
        time.sleep(0.05)
    t = db.load_task(tid)
    assert t is not None
    assert t["status"] == "completed", f"任务状态异常：{t.get('status')} {t.get('message')}"


# ===========================================================================
# P1-13：并发闸门
# ===========================================================================

def test_concurrency_gate_blocks_excess_and_releases():
    from lottery import concurrency as CC

    bucket = "predict"
    limit = CC.DEFAULT_LIMITS[bucket]
    held = []
    try:
        for _ in range(limit):
            ctx = CC.acquire(bucket)
            ctx.__enter__()
            held.append(ctx)
        assert CC.inflight(bucket) == limit
        with pytest.raises(CC.Busy):
            CC.acquire(bucket).__enter__()
    finally:
        for c in held:
            c.__exit__(None, None, None)
    assert CC.inflight(bucket) == 0, "释放后计数没有归零（会永久占位）"
    # 释放后必须能再次获取
    ctx = CC.acquire(bucket)
    ctx.__enter__()
    ctx.__exit__(None, None, None)


def test_concurrency_buckets_are_independent():
    from lottery import concurrency as CC
    b = CC.acquire("backtest")
    b.__enter__()
    try:
        assert CC.inflight("backtest") == 1
        assert CC.inflight("predict") == 0
    finally:
        b.__exit__(None, None, None)


def test_lightweight_endpoints_are_not_gated():
    """探活/鉴权/静态资源不得被并发闸门挡住（否则容器会被判 unhealthy）。"""
    from lottery import concurrency as CC
    for path in ("/api/health", "/api/info", "/api/auth/login",
                 "/api/auth/me", "/static/app.js", "/"):
        assert CC.bucket_for(path) is None, f"{path} 不该受限流"


def test_domain_resolving_to_private_ip_is_rejected(monkeypatch):
    """域名能解析、但解析到内网 → 必须拒（否则改个域名就绕过 IP 黑名单）。"""
    import socket

    from lottery import url_guard

    def fake_getaddrinfo(host, *a, **k):
        return [(socket.AF_INET, 1, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    ok, why = url_guard.check_llm_base_url("https://evil.example.com/v1")
    assert not ok, "域名解析到内网却没有被拦"
    assert "10.0.0.5" in why


def test_unresolvable_domain_is_not_rejected(monkeypatch):
    """**解析失败不该否决** —— 无 DNS 的环境（CI/离线）会把正常地址误拦。

    V37 第一版犯过这个错：把"解析不到"判成不安全，导致
    `https://a.example.com` 在 CI 里保存配置直接 400，
    看起来像产品坏了，实际是校验策略过度 fail-closed。
    真正要拦的是"能确认是内网"；"无法确认"交给发请求前那道闸。
    """
    import socket

    from lottery import url_guard

    def boom(host, *a, **k):
        raise socket.gaierror("no dns here")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    ok, why = url_guard.check_llm_base_url("https://api.example.com/v1")
    assert ok, f"解析失败被误判为不安全：{why}"


# ===========================================================================
# P1（健壮性）：解析器不得因一行坏数据毁掉整批
# ===========================================================================

_GOOD_LINE = ("2003001 2003-02-23 10 11 12 13 26 28 11 26 28 11 13 10 12 "
              + " ".join(["0"] * 16))
_GOOD_LINE2 = ("2003002 2003-02-27 04 09 19 20 21 26 12 09 04 26 21 19 20 "
               + " ".join(["0"] * 16))


def test_parse_text_skips_bad_lines_by_default():
    """一行坏数据不该让整批（几千期）作废。

    旧实现任一行出错就 raise，而源站偶尔会掺 HTML 报错页/空行/截断行，
    结果是"刷新后一期都没有"。修后：能解析的照常返回，坏行如实报出。
    """
    from lottery import parser
    bad = []
    out = parser.parse_text(
        _GOOD_LINE + "\n<html>error</html>\n" + _GOOD_LINE2, on_bad=bad)
    assert len(out) == 2, f"容错失败，只解析出 {len(out)} 期"
    assert len(bad) == 1 and bad[0][0] == 2, f"坏行报告不对：{bad}"


def test_parse_text_strict_still_raises():
    """需要"一个坏行都不许"的场景可显式 strict（保留旧语义，不破坏既有调用）。"""
    from lottery import parser
    with pytest.raises(parser.ParseError):
        parser.parse_text(_GOOD_LINE + "\nGARBAGE\n" + _GOOD_LINE2, strict=True)


def test_parse_text_all_bad_still_raises():
    """**全部**解析不出来时必须报错 —— 那更像"源改版/取到错误页"，
    安静返回空列表会让上层以为"这批没有新数据"。"""
    from lottery import parser
    with pytest.raises(parser.ParseError):
        parser.parse_text("<html>not data at all</html>")
