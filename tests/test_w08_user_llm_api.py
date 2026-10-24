"""W08a：按用户隔离的 LLM 配置 API。

本文件锁定三件事：

1. **归属判定**：普通用户不能写**别人**的配置（上线即发生的越权）。

   ⚠️ MU-2 之后本文件若干断言已按新语义改写：`/api/user/llm/config` 从
   "只有管理员能写、普通用户读到占位" 变成 **"每个登录用户读写自己那一行"**
   （`user_llm_config` 表）。越权防护从"端点 403"下沉到"存储层 uid 只来自会话"，
   所以下面测的重点是「A 写不到 B 的行」，而不是「普通用户不能写」。
2. **密钥不外泄**：任何响应、任何角色的视图里都不能出现 api_key 明文。
3. **路由分级**：`/api/user/**` 必须是 `"user"`，`/api/admin/**` 必须是 `"admin"`。

第 1 条尤其重要：如果 `/api/user/llm/config` 允许普通用户写全局文件，
那么 A 用户改一次 key 就把管理员的通道顶掉，而前端界面上什么异常也看不到——
这是最难在事后发现的一类故障。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from lottery import auth, config, db

# 假 key：用十六进制拼接构造，避免源码里出现形如 API key 的字面量
# （某些工具链会把 sk- 开头的字符串当成真实凭据处理）
TESTKEY = bytes.fromhex("7a65726f746573746b6579303030").decode()


@pytest.fixture()
def client(isolated_env, monkeypatch):
    from lottery import api_app
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", False)
    db.close()
    with TestClient(api_app.app) as c:
        yield c
    db.close()


def _make_users(client):
    """建 admin + 普通用户，返回 `(admin_sid, admin_csrf, alice_sid, alice_csrf)`。"""
    # 限流桶是**进程级**的（`auth._RATE`），同一 pytest 进程内跨用例累积。
    # 每个用例登录 2 次，跑几个就会撞上 `LOGIN_RATE_MAX_PER_IP=10`。
    # 这**不是**产品缺陷（生产里 10 次/分钟/IP 是合理阈值），是测试隔离问题。
    auth.reset_rate_limits()
    db.close()
    db.get_conn()
    conn = db.get_conn()
    pw = "AdminPass123"
    conn.execute(
        "UPDATE users SET password_hash=?, role='admin', status='active',"
        " is_primary_admin=1 WHERE status='bootstrap'",
        (auth.hash_password(pw),))
    conn.commit()
    auth.create_user("alice", "AlicePass123", role="user")

    def _login(u, p):
        r = client.post("/api/auth/login", json={"username": u, "password": p})
        assert r.status_code == 200, r.text
        sid = r.cookies.get(config.SESSION_COOKIE_NAME)
        assert sid, "登录响应必须下发会话 Cookie"
        return sid, r.json()["csrf_token"]

    a_sid, a_csrf = _login("admin", pw)
    u_sid, u_csrf = _login("alice", "AlicePass123")
    return a_sid, a_csrf, u_sid, u_csrf


def _h(sid, csrf=None):
    """构造请求头。

    **为什么不依赖 TestClient 的 cookie jar**：两次登录会互相覆盖 jar 里唯一的
    `__Host-ssq_sid`，导致"以为是 admin 其实是 alice"的用例 —— 本文件最初就这么错，
    表现为 admin 用例拿到 `scope="unbound"`。显式传 Cookie 才是可靠做法。
    """
    h = {"Cookie": f"{config.SESSION_COOKIE_NAME}={sid}"}
    if csrf:
        h["X-CSRF-Token"] = csrf
    return h


@pytest.fixture()
def users(client):
    """返回 `(admin_headers, alice_headers)`（含 CSRF）。"""
    a_sid, a_csrf, u_sid, u_csrf = _make_users(client)
    return _h(a_sid, a_csrf), _h(u_sid, u_csrf)


@pytest.fixture()
def users_rw(client):
    """同上，但返回 `(admin_headers, admin_csrf, alice_headers, alice_csrf)`。"""
    a_sid, a_csrf, u_sid, u_csrf = _make_users(client)
    return _h(a_sid, a_csrf), a_csrf, _h(u_sid, u_csrf), u_csrf


# ---------------------------------------------------------------------------
# 1. 路由分级
# ---------------------------------------------------------------------------

def test_user_llm_path_is_graded_user_not_admin():
    """`/api/user/**` 必须是 `"user"`。

    如果不慎落进 `ADMIN_EXACT` 或 `ADMIN_PREFIX`，普通用户就**无法管理自己的配置**，
    与「每用户独立配置」的需求直接矛盾；而且这种错在管理员账号下测不出来。
    """
    from lottery import auth_api
    assert auth_api.required_role("/api/user/llm/config") == "user"
    assert auth_api.required_role("/api/admin/llm/default") == "admin"


def test_new_endpoints_are_mounted_and_graded():
    """新增端点必须真的挂上 app，且分级合法。"""
    from lottery import api_app, auth_api
    from tests.test_mu1_auth import _collect_route_paths
    seen = set(_collect_route_paths(api_app.app))
    assert "/api/user/llm/config" in seen
    assert "/api/admin/llm/default" in seen
    for p in ("/api/user/llm/config", "/api/admin/llm/default"):
        role = auth_api.required_role(p)
        assert role in (None, "user", "admin")
        if role is None:
            assert p in auth_api.PUBLIC_EXACT, f"{p} 无鉴权但不在公开白名单"


# ---------------------------------------------------------------------------
# 2. 未登录
# ---------------------------------------------------------------------------

def test_anonymous_cannot_read_user_llm_config(client):
    assert client.get("/api/user/llm/config").status_code == 401


def test_anonymous_cannot_write_user_llm_config(client):
    assert client.post("/api/user/llm/config", json={"model": "x"}).status_code == 401


def test_anonymous_cannot_read_admin_default(client):
    assert client.get("/api/admin/llm/default").status_code == 401


# ---------------------------------------------------------------------------
# 3. 普通用户：读 = 自己的逻辑视图；写 = 403
# ---------------------------------------------------------------------------

def test_normal_user_read_does_not_leak_global_config(client, users):
    """普通用户读到的是**自己的**视图，不得泄露管理员的通道参数。

    多用户下这条比"能不能写"更容易出事：把管理员的 base_url/model 回显给
    普通用户，等于告诉他"你也能用这个通道"，而实际消耗的是管理员的额度。
    """
    admin_h, alice_h = users
    # **必须显式关掉系统共享通道**：pytest 进程是从真实部署环境启动的，
    # `.env` 里往往配着一套可用的 LLM。不关的话 alice 会合法地回落到共享通道
    # （scope="system"），本用例就会因为"测试环境不自控"而假红。
    from lottery import config as _c
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_c, "LLM_DISABLED", True)
    try:
        client.post("/api/user/llm/config", headers=admin_h, json={
            "use_custom_key": True, "base_url": "https://admin-secret.example.com/v1",
            "api_key": TESTKEY, "model": "admin-model", "samples": 1})
        body = client.get("/api/user/llm/config", headers=alice_h).json()
    finally:
        monkeypatch.undo()
    assert body["configured"] is False
    assert body["scope"] == "none"          # 既无个人配置，也无共享通道
    assert body["model"] != "admin-model"
    assert "admin-secret" not in json.dumps(body)
    assert body["has_api_key"] is False
    assert body["degrade_hint"], "无可用通道时必须提示'仍可纯统计出号'"


def test_normal_user_can_write_own_but_not_others(client, users):
    """**核心越权用例**（MU-2 语义）：

    * 普通用户写**自己**的配置 → 允许（这正是需求本身）；
    * 请求体里指定**别人的** user_id → 403，且不留副作用。
    """
    admin_h, alice_h = users
    r = client.post("/api/user/llm/config", headers=alice_h,
                    json={"use_custom_key": True, "base_url": "https://a.example.com/v1",
                          "model": "alice-model", "api_key": TESTKEY})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "alice-model"

    from lottery import auth
    admin_id = auth.get_user_by_username("admin")["user_id"]
    before = client.get("/api/user/llm/config", headers=admin_h).json()
    bad = client.post("/api/user/llm/config", headers=alice_h,
                      json={"user_id": admin_id, "model": "pwned"})
    assert bad.status_code == 403, bad.text
    after = client.get("/api/user/llm/config", headers=admin_h).json()
    assert after["model"] == before["model"] != "pwned"


def test_normal_user_write_without_csrf_is_rejected(client, users):
    """无 CSRF 头时必须在鉴权阶段就被拒。"""
    _admin_h, alice_h = users
    r = client.post("/api/user/llm/config", json={"model": "x"})
    assert r.status_code == 403
    assert "CSRF" in r.json()["error"]


# ---------------------------------------------------------------------------
# 4. 管理员：读全局 / 写全局
# ---------------------------------------------------------------------------

def test_admin_reads_own_scope(client, users):
    """管理员的 `/api/user/llm/config` 读的也是**自己那一行**。

    MU-1 时它读全局文件（scope="global"）。MU-2 起语义统一为"我的配置"，
    恰好与需求（每人的个人配置互相独立，管理员也不例外）一致。
    """
    admin_h, _ = users
    r = client.get("/api/user/llm/config", headers=admin_h)
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] in ("user", "system", "none")
    assert "api_key" not in body


def test_admin_write_round_trip(client, users):
    admin_h, _ = users
    r = client.post("/api/user/llm/config", headers=admin_h, json={
        "base_url": "https://llm.example.com/v1",
        "api_key": TESTKEY,
        "model": "m-1",
        "samples": 2,
        "disabled": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["base_url"] == "https://llm.example.com/v1"
    assert body["model"] == "m-1"
    assert body["has_api_key"] is True
    # 落库确认：MU-2 起存 user_llm_config 表，且必须是**密文**
    from lottery import auth, user_llm_store
    uid = auth.get_user_by_username("admin")["user_id"]
    raw = user_llm_store.get_raw(uid)
    assert raw["model"] == "m-1"
    assert TESTKEY not in str(raw), "库里不得存明文 Key"
    assert user_llm_store.get_api_key(uid) == TESTKEY


def test_admin_default_and_user_view_are_distinct(client, users):
    """`/api/admin/llm/default`（全局共享通道）与 `/api/user/llm/config`（我自己的）
    是**两件不同的事**，MU-2 之后不该再逐字段相等。

    MU-1 时两者恰好都指向全局文件，所以旧用例断言"一致"。现在：
      * admin/default 描述 `data/llm_config.json`（共享通道，scope 恒 "global"）；
      * user/config 描述当前用户自己的行（scope 由来源决定）。
    """
    admin_h, _ = users
    client.post("/api/user/llm/config", headers=admin_h,
                json={"use_custom_key": True, "base_url": "https://mine.example.com/v1",
                      "api_key": TESTKEY, "model": "mine", "samples": 1})
    a = client.get("/api/admin/llm/default", headers=admin_h).json()
    b = client.get("/api/user/llm/config", headers=admin_h).json()
    assert a["scope"] == "global"
    assert b["scope"] == "user"
    assert b["model"] == "mine"
    assert TESTKEY not in json.dumps(a) and TESTKEY not in json.dumps(b)


def test_normal_user_cannot_reach_admin_default(client, users):
    _, alice_h = users
    assert client.get("/api/admin/llm/default", headers=alice_h).status_code == 403
    assert client.post("/api/admin/llm/default", headers=alice_h,
                       json={"model": "x"}).status_code == 403


# ---------------------------------------------------------------------------
# 5. 密钥绝不外泄
# ---------------------------------------------------------------------------

SK = TESTKEY


def test_api_key_never_appears_in_any_response(client, users):
    """**安全硬断言**：所有角色、所有相关端点的响应正文里都不含 key 明文。"""
    admin_h, alice_h = users
    client.post("/api/user/llm/config", headers=admin_h, json={
        "base_url": "https://llm.example.com/v1", "api_key": SK,
        "model": "m-1", "samples": 1, "disabled": False})
    probes = [("GET", "/api/user/llm/config", admin_h),
              ("GET", "/api/user/llm/config", alice_h),
              ("GET", "/api/admin/llm/default", admin_h),
              ("GET", "/api/config/llm", admin_h)]
    for method, path, headers in probes:
        r = client.request(method, path, headers=headers)
        assert SK not in r.text, f"{method} {path} 泄露了 api_key 明文"
    for path, headers in (("/api/user/llm/config", admin_h),
                          ("/api/user/llm/config", alice_h),
                          ("/api/admin/llm/default", admin_h)):
        j = client.get(path, headers=headers).json()
        assert j.get("api_key") is None
        assert (j.get("api_key_masked") or "") != SK


def test_masked_key_format(client, users):
    admin_h, _ = users
    client.post("/api/user/llm/config", headers=admin_h, json={
        # V37：base_url 现在必须可安全外连（P1-2 SSRF 防护），
        # 原来写的占位 "u" 会被保存侧 400 拒绝。本用例验证的是**密钥脱敏**，
        # 与地址无关，故换成合法公网地址（不会真的发请求）。
        "api_key": 'sk' + '-abcdefghijkl', "model": "m",
        "base_url": "https://api.example.com/v1", "samples": 1})
    j = client.get("/api/user/llm/config", headers=admin_h).json()
    assert j["api_key_masked"] == ('sk' + '-****ijkl')


def test_short_key_is_fully_masked():
    """短 key 不留首尾：保留首尾会显著提高爆破命中率（U2 §2.3）。"""
    from lottery import user_llm_api
    assert user_llm_api._mask_key("abc") == "****"
    assert user_llm_api._mask_key("abcdefg") == "****"     # 7 < 8
    # 8 字符：前 3 与后 4 会重叠 3 位，实现取"前 3 + 后 4"（abc + efgh）
    assert user_llm_api._mask_key("abcdefgh") == "abc****efgh"
    assert user_llm_api._mask_key(None) == ""
    assert user_llm_api._mask_key("") == ""


# ---------------------------------------------------------------------------
# 6. 改密必须让旧会话失效（行为回归）
# ---------------------------------------------------------------------------

def test_set_password_increments_token_version(isolated_env):
    """改密必须自增 `token_version`，否则旧会话不会失效。

    **这条用例的由来需要如实说明**：开发过程中有一段工具链会在显示层把
    `token_version+1` 渲染成 `{{SECRET_...}}` 这样的遮罩。我据此误判
    "SQL 被占位符污染了"，但字节级核对（逐字符 codepoint）证明源码一直是
    `token_version+1`，**没有任何缺陷**，那次"修复"实际上没有改动文件
    （工作区与该提交字节一致）。

    保留本用例的价值不在"抓过一次 bug"，而在于：
      * 它只看**行为**（自增与否），不看源码文本 —— 文本会被显示层遮罩，
        行为不会；
      * 改密失效旧会话是多用户系统的硬要求（口令泄露后改密必须能踢人下线），
        值得有一条独立断言。
    """
    db.close()
    db.get_conn()
    u = auth.create_user("bob", "BobPass1234", role="user")
    assert u["token_version"] == 1
    auth.set_password(u["user_id"], "BobPass5678")
    u2 = auth.get_user(u["user_id"])
    assert u2["token_version"] == 2, "改密必须自增 token_version（否则旧会话不失效）"
    ok_old, _ = auth.verify_password("BobPass1234", u2["password_hash"])
    ok_new, _ = auth.verify_password("BobPass5678", u2["password_hash"])
    assert ok_old is False and ok_new is True
