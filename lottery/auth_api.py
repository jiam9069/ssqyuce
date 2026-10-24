"""多用户认证与管理 API（MU-1 / W05）。

本模块只负责**HTTP 层**：分级判定（`required_role`）+ 认证端点 + 管理员用户管理端点。
业务逻辑与存储全部在 `lottery/auth.py`，两者不混。

## 分级原则（deny-by-default）

* 未在表里列出的路径 → **默认需要登录**（`"user"`）。这样"新增端点忘记定级"
  会变成"需要登录"而不是"匿名可用"——失败方向是安全的一侧。
* 危险操作（写库、烧 LLM 钱、改全局配置、跨用户读）→ `"admin"`。
* 少数必须匿名可达的（首页、健康检查、登录/引导）→ `None`。
* `/api/user/**` 一律 `"user"`：它是「**当前登录人**自己的东西」，天然不需要管理员，
  也**绝不能**落进 `/api/admin/` —— 那会把"管理我自己的配置"误判为管理权限。

配套有一条**参数化测试**遍历 `app.routes`，断言"每条路由都有明确分级"，
这样将来新增端点若忘了登记，测试会直接红（见 tests/test_mu1_auth.py）。

## 鉴权执行模式

`config.AUTH_MODE`：
* `off`（默认）——只解析身份、不拦截。用于"前端登录页尚未落地"的灰度期，
  避免强制拦截导致线上整站 401。
* `enforce`——deny-by-default，未登录一律 401。
切换前必须先完成前端登录页与引导建号。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import functools

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from . import auth, config


async def _offload(fn, *args, **kwargs):
    """把**阻塞 CPU** 的工作挪出事件循环（V36，2026-10-24 审计 P1-1）。

    ## 为什么必须挪

    `auth_login` 是 `async def`，而 scrypt 是纯 CPU 的同步调用（32MB / 145-550ms 每次）。
    在协程体里直接调用它，会**冻结整个事件循环**：所有并发请求（包括容器 healthcheck
    的 `/api/health`）都要排队等这一次哈希算完。实测单次登录造成 heartbeat 最大间隔
    550ms；而登录接口**匿名可达**，并发打几次就能把整站拖到不可用。

    `_hash_sem`（最并发 4）只限制了内存峰值，**没有**限制事件循环被阻塞 ——
    那是个正确但不相干的设计：它防的是 32MB × N 把内存打爆，不是防冻结。

    ## 为什么用默认 executor

    不另建线程池：默认 executor 与 FastAPI 给同步路由用的**是同一个**，
    因此「并发上限」由同一个池统一约束，不会出现两套池互相打架。
    阻塞调用挪过去后，事件循环可以继续处理别的请求。

    `functools.partial` 用于传关键字参数（`run_in_executor` 只接受位置参数）。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

router = APIRouter()

# ---------------------------------------------------------------------------
# 分级表
# ---------------------------------------------------------------------------

# 完全匿名可达（必须精确匹配）
#
# ⚠️ 加入此集合 = 该路径**不受任何鉴权保护**，两重含义：
#   ① 多用户模式下任何人可访问；
#   ② 部署了 `LOTT_TOKEN` 时也不再要求 Bearer Token。
# 因此这里只放"必须匿名才能工作"的三类：前端单页、容器探活、登录/引导入口。
# 曾经误把 `/api/info` 放进来，导致配了 LOTT_TOKEN 后它反而变公开
# （被 tests/test_m4_5_engineering.py 的既有断言抓住）——**不要**再加回。
PUBLIC_EXACT = frozenset({
    "/",                     # 前端单页（只是 HTML 外壳，不含数据）
    "/api/health",           # 探活（容器 healthcheck 依赖它，绝不能要登录）
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/me",          # 未登录时应返回 {authenticated: false} 而不是 401
    "/api/auth/bootstrap",   # 首次引导（受一次性 env token 保护）
})

PUBLIC_PREFIX = ("/static/",)

#: 文档端点（V36，2026-10-24 审计 P1-4）。
#:
#: 这些路径在 `LOTT_API_DOCS=0`（默认）时 FastAPI **根本不注册**，
#: 但鉴权中间件跑在路由之前 —— 于是它们会先被判定成 "user"，未登录返回 401。
#: 401 等于告诉扫描器「这个路径存在，只是你没登录」；而真实情况是它不存在。
#: 现在显式把它们归为「不存在」，中间件放行后由路由层自然给 404。
#: 注意：仅当文档确实关闭时才这样处理 —— 若运维显式开启文档，
#: 它们就该按普通 user 资源受保护（`_docs_disabled()` 里判断）。
DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})


def _docs_disabled() -> bool:
    try:
        return not bool(config.API_DOCS_ENABLED)
    except Exception:          # noqa: BLE001
        return True

# 管理员专用（危险写操作 / 全局配置 / 花真金白银的端点）
ADMIN_EXACT = frozenset({
    # 全局配置与数据抓取
    "/api/config/llm",            # GET 也含 base_url/model，不该给普通用户
    "/api/methods/config",
    "/api/refresh",
    "/api/data/reconcile",
    "/api/data/reconcile/history",
    # 烧 LLM token 的动作
    "/api/llm/test",
    "/api/llm/audit/run",
    "/api/llm/decision/restore",
    "/api/llm_lab/run/async",
    "/api/llm_lab/register",
    "/api/eval/llm/run",
    "/api/rules/closed-loop/async",
    "/api/audit/anomaly/run",
    # 重算 / 写库 / 回放
    "/api/evolution/run",
    "/api/mining/run",
    "/api/ml/eval",
    "/api/patterns/backtest",
    "/api/eval/backtest",
    "/api/eval/online",
    "/api/regime/refresh",
    "/api/replay/diagnose",
    "/api/rules/lifecycle/run",
    "/api/rules/refresh",
    "/api/experiments/randomness-selfcheck",
    # 系统级只读（含他人/全局信息）
    "/api/llm/cost",
    "/api/tasks",
    # W08a：全局默认 LLM 配置（含 base_url/model；普通用户只看自己的视图）
    "/api/admin/llm/default",
})

ADMIN_PREFIX = ("/api/admin/",)


def required_role(path: str) -> Optional[str]:
    """返回 ``None``(公开) / ``"user"``(需登录) / ``"admin"``(需管理员)。

    **纯函数**：不读全局状态、不碰 DB —— 因此可以被参数化测试完整覆盖。
    """
    if path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIX):
        return None
    # V36（P1-4）：文档关闭时，这些路径**不存在**，不该由鉴权层回 401。
    # 放行 → 路由层自然 404，与"随机不存在的路径"表现一致，不泄露存在性。
    if path in DOCS_PATHS and _docs_disabled():
        return None
    if path.startswith(ADMIN_PREFIX) or path in ADMIN_EXACT:
        return "admin"
    return "user"     # ← deny-by-default


# ---------------------------------------------------------------------------
# 请求辅助
# ---------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    """取真实客户端 IP。

    容器内看到的源地址恒为 docker 网关（`172.18.0.1`），uvicorn 默认只信
    127.0.0.1 的 XFF，因此这里**显式优先** nginx 注入的 `X-Real-IP`。
    它只用于限流与审计，不参与鉴权判定（防伪造 IP 绕过登录）。
    """
    for h in ("x-real-ip", "x-forwarded-for"):
        v = request.headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return request.client.host if request.client else ""


def session_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """读 Cookie → 会话。无则 None。"""
    sid = request.cookies.get(config.SESSION_COOKIE_NAME)
    if not sid:
        return None
    return auth.resolve_session(sid)


def _set_session_cookie(resp, session_id: str) -> None:
    resp.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=session_id,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,                      # JS 读不到，降低 XSS 窃取面
        secure=config.SESSION_COOKIE_SECURE,  # 生产 HTTPS 下必须 True
        samesite="lax",                     # 兼顾跨站导航与 CSRF
        path="/",
    )


def _clear_session_cookie(resp) -> None:
    resp.delete_cookie(config.SESSION_COOKIE_NAME, path="/")


def _csrf_ok(request: Request, session: Dict[str, Any]) -> bool:
    """写操作必须带 `X-CSRF-Token`（双提交校验，U1 §2.4）。"""
    return auth.require_csrf(session.get("session") or {},
                             request.headers.get("x-csrf-token"))


def _err(msg: str, code: int = 400, **extra) -> JSONResponse:
    body = {"ok": False, "error": msg}
    body.update(extra)
    return JSONResponse(body, status_code=code)


def _me_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    """对外暴露的用户字段白名单（**绝不返回 password_hash**）。"""
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "role": user["role"],
        "status": user["status"],
        "display_name": user.get("display_name") or "",
        "is_primary_admin": bool(user.get("is_primary_admin")),
        "last_login_at": user.get("last_login_at"),
    }


# ---------------------------------------------------------------------------
# 认证端点
# ---------------------------------------------------------------------------

@router.get("/api/auth/me")
def auth_me(request: Request):
    """当前登录态。**未登录返回 200 + authenticated=false**（不是 401）。

    为什么不用 401：前端启动探测要区分"没登录"和"接口坏了"，
    401 会让前端的统一错误处理把它当异常弹窗。

    ## `csrf_token` 下发（API-1 v1.0 §3.1，2026-10-21）

    已登录时一并返回**本会话自己的** `csrf_token`，与 `/api/auth/login` 下发的
    是同一个值。

    为什么必须有这条：`csrf_token` 原先只在 login 响应体里给。前端刷新页面后
    内存态清空、Cookie 仍在 → 会话有效但拿不到 csrf → **所有写操作恒 403**
    （实测：`{"ok":false,"error":"CSRF 校验失败"}`），且该 403 不带
    `auth_required`，前端无法识别为「需要重新登录」。

    ⚠️ **不得每次调用新生成**：契约 §3.1 明确要求返回会话记录里的那个值。
    若在此处 `new_csrf_token()`，多标签页 / 并发请求会各自拿到不同值而互相
    失效，缺陷从「刷新丢 csrf」变成「每请求都换 csrf」，更隐蔽也更难查。
    """
    sess = session_from_request(request)
    if not sess:
        return {"ok": True, "authenticated": False, "user": None,
                "auth_enforced": config.auth_enforced(), "csrf_token": None}

    # API-1 v1.1 §3.1 硬约束：authenticated===true 时 csrf_token **必须非空**。
    # 正常路径 create_session 一定写了；但老数据/手工插库可能为空。若这里用
    # `or None` 兜底，就把「已登录 + null」这个**按 v1.1 属契约违背**的形态合法化了。
    # 处理：为空则补一个并持久化回该会话（同时满足「非空」与「稳定」）。
    csrf = (sess.get("session") or {}).get("csrf_token") or ""
    if not csrf:
        csrf = auth.ensure_session_csrf(sess["session"]["session_id"])
    return {"ok": True, "authenticated": True, "user": _me_payload(sess["user"]),
            "auth_enforced": config.auth_enforced(),
            # 取自会话记录（或补齐后已持久化的值），**不新造临时值**。
            "csrf_token": csrf}


@router.post("/api/auth/login")
async def auth_login(request: Request):
    """用户名 + 口令登录。失败原因**统一**为"用户名或口令错误"（不区分）。"""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    if not isinstance(payload, dict):
        return _err("请求体必须是 JSON 对象")
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    ip = client_ip(request)

    if not username or not password:
        return _err("用户名和口令均为必填")

    limited = auth.login_rate_limited(ip, username)
    if limited:
        auth.audit("auth.login_failed", actor_name=username, ip=ip,
                   result="denied", detail={"reason": "rate_limited"})
        return _err(limited, code=429)

    user = auth.get_user_by_username(username)
    if user is None:
        # 仍走一次哈希，避免"用户不存在"因耗时更短而被时序侧信道枚举。
        # P1-1：探针哈希同样必须挪出事件循环（它也是 145~550ms 的 CPU 活）。
        # 两次 scrypt 都要 offload：hash_password 生成探针串同样是 145~550ms 的 CPU 活。
        _dummy = await _offload(auth.hash_password, "dummy-placeholder")
        await _offload(auth.verify_password, password, _dummy)
        auth.audit("auth.login_failed", actor_name=username, ip=ip,
                   result="failed", detail={"reason": "no_such_user"})
        return _err("用户名或口令错误", code=401)

    if auth.is_locked(user):
        auth.audit("auth.login_failed", actor_name=username, ip=ip,
                   actor_id=user["user_id"], result="failed",
                   detail={"reason": "locked"})
        return _err("账号已被临时锁定，请稍后再试或联系管理员", code=423)

    # P1-1：阻塞的 scrypt 校验挪到线程池，否则整站被这一次登录冻结
    ok, needs_rehash = await _offload(
        auth.verify_password, password, user.get("password_hash"))
    if not ok:
        auth.record_login_failure(user["user_id"])
        auth.audit("auth.login_failed", actor_name=username, ip=ip,
                   actor_id=user["user_id"], result="failed",
                   detail={"reason": "bad_password"})
        return _err("用户名或口令错误", code=401)

    # 【裁决 7】status 闸门先于角色判断：bootstrap 零权限
    if user.get("status") != "active":
        auth.audit("auth.login_failed", actor_name=username, ip=ip,
                   actor_id=user["user_id"], result="denied",
                   detail={"reason": f"status={user.get('status')}"})
        hint = ("管理员账号尚未完成初始化，请先完成引导"
                if user.get("status") == "bootstrap" else "账号不可用")
        return _err(hint, code=403)

    if needs_rehash:
        # 参数升级：登录成功时顺手重算（PHC 串的向前兼容价值就在这）
        try:
            # P1-1：set_password 内部同样跑 scrypt，必须 offload
            await _offload(auth.set_password, user["user_id"], password,
                           action="user.password_change")
        except Exception:  # noqa: BLE001
            pass

    auth.record_login_success(user["user_id"])
    sess = auth.create_session(user["user_id"], ip=ip,
                               user_agent=request.headers.get("user-agent", ""))
    auth.audit("auth.login", actor_id=user["user_id"], actor_name=username, ip=ip)

    fresh = auth.get_user(user["user_id"]) or user
    resp = JSONResponse({
        "ok": True, "user": _me_payload(fresh),
        "csrf_token": sess["csrf_token"],
        "expires_at": sess["expires_at"],
    })
    _set_session_cookie(resp, sess["session_id"])
    return resp


@router.post("/api/auth/logout")
def auth_logout(request: Request):
    """登出。无会话时也返回 200（幂等）。"""
    sid = request.cookies.get(config.SESSION_COOKIE_NAME)
    if sid:
        sess = auth.resolve_session(sid)
        auth.revoke_session(sid, actor_id=(sess or {}).get("session", {}).get("user_id"))
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@router.post("/api/auth/password")
async def auth_change_password(request: Request):
    """登录用户自改口令。改密后**所有旧会话失效**（需重新登录）。"""
    sess = session_from_request(request)
    if not sess:
        return _err("未登录", code=401)
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    old_pw = str(payload.get("old_password") or "")
    new_pw = str(payload.get("new_password") or "")
    user = sess["user"]

    # P1-1：scrypt 校验必须 offload（与登录同源，否则改密也能冻结全站）
    ok, _ = await _offload(auth.verify_password, old_pw, user.get("password_hash"))
    if not ok:
        auth.audit("user.password_change", actor_id=user["user_id"],
                   ip=client_ip(request), result="denied",
                   detail={"reason": "bad_old_password"})
        return _err("原口令不正确", code=400)
    perr = auth.validate_new_password(new_pw)
    if perr:
        return _err(perr)
    await _offload(auth.set_password, user["user_id"], new_pw,
                   actor_id=user["user_id"], ip=client_ip(request))
    resp = JSONResponse({"ok": True, "relogin_required": True})
    _clear_session_cookie(resp)
    return resp


@router.post("/api/auth/bootstrap")
async def auth_bootstrap(request: Request):
    """首次引导：用一次性 env token 把 bootstrap 行升级为可用的主管理员。

    安全约束（U1 §5.2 / §4.1 不变量 2）：
      * 口令来自 `LOTT_BOOTSTRAP_TOKEN`，**没有默认值**；未配置则整条通道关闭。
      * 成功后立刻从进程 env 中删除该 token（一次性）。
      * `role='admin'` + `is_primary_admin=1` + `status='active'` 三件事
        **必须在同一事务里**完成——绝不允许出现"零密码但有主 admin 标记"的中间态。
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    token = str(payload.get("token") or "")
    password = str(payload.get("password") or "")
    ip = client_ip(request)

    if not auth.claim_bootstrap_token(token):
        auth.audit("auth.bootstrap", ip=ip, result="denied",
                   detail={"reason": "bad_or_absent_token"})
        return _err("引导口令无效或未启用引导（LOTT_BOOTSTRAP_TOKEN）", code=403)

    perr = auth.validate_new_password(password)
    if perr:
        return _err(perr)

    # 找一个待初始化的 bootstrap 行；若没有（例如已初始化过）则拒绝重复引导
    from . import db
    conn = db.get_conn()
    row = conn.execute(
        "SELECT user_id, username FROM users WHERE status='bootstrap' AND deleted_at IS NULL"
        " ORDER BY user_id LIMIT 1").fetchone()
    if row is None:
        return _err("没有待初始化的管理员账号（可能已完成引导）", code=409)

    uid = int(row["user_id"])
    # P1-1：引导建号同样要算 scrypt，offload 出事件循环
    pw_hash = await _offload(auth.hash_password, password)
    now = time.time()
    try:
        conn.execute("BEGIN IMMEDIATE")
        # 三步在同一事务：设密 + 升 admin + 置主 admin 标记
        conn.execute(
            "UPDATE users SET password_hash=?, role='admin', status='active',"
            " is_primary_admin=1, failed_count=0, locked_until=0, updated_at=?"
            " WHERE user_id=? AND status='bootstrap'",
            (pw_hash, now, uid))
        conn.commit()
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        return _err(f"引导失败：{type(e).__name__}", code=500)

    auth.invalidate_session_cache()
    auth.audit("auth.bootstrap", actor_id=uid, target_id=uid, ip=ip,
               target_name=row["username"])
    auth.consume_bootstrap_token()      # 一次性：从进程 env 移除
    return {"ok": True, "user_id": uid, "username": row["username"],
            "note": "引导完成。请立刻从部署 .env 中删除 LOTT_BOOTSTRAP_TOKEN 并重启一次。"}


# ---------------------------------------------------------------------------
# 管理员：用户管理
# ---------------------------------------------------------------------------

def _require_admin(request: Request) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    """管理员守卫（端点内部自检，与中间件形成双保险）。"""
    sess = session_from_request(request)
    if not sess:
        return None, _err("未登录", code=401)
    user = sess["user"]
    # 裁决 7：status 已在 auth.load_active_user 里保证了 'active'
    if user.get("role") != "admin":
        auth.audit("authz.denied", actor_id=user["user_id"],
                   ip=client_ip(request), result="denied",
                   detail={"path": request.url.path, "required": "admin"})
        return None, _err("需要管理员权限", code=403)
    return sess, None


@router.get("/api/admin/users")
def admin_list_users(request: Request):
    """用户列表（不含 password_hash）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    from . import db
    rows = db.get_conn().execute(
        "SELECT user_id, username, role, status, display_name, is_primary_admin,"
        " last_login_at, created_at, deleted_at FROM users ORDER BY user_id").fetchall()
    return {"ok": True, "users": [dict(r) for r in rows]}


@router.post("/api/admin/users")
async def admin_create_user(request: Request):
    """建用户。`password` 必填（不允许建无口令账号，除内部 bootstrap）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    role = str(payload.get("role") or "user")
    display_name = str(payload.get("display_name") or "")
    if not password:
        return _err("必须为新用户设置口令")
    try:
        # P1-1：create_user 内部跑 scrypt，必须 offload
        u = await _offload(auth.create_user, username, password, role=role,
                           display_name=display_name,
                           actor_id=sess["user"]["user_id"],
                           ip=client_ip(request))
    except ValueError as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        if "UNIQUE" in str(e):
            return _err("用户名已存在", code=409)
        return _err(f"创建失败：{type(e).__name__}", code=500)
    return {"ok": True, "user": _me_payload(u)}


@router.patch("/api/admin/users/{user_id}")
async def admin_patch_user(user_id: int, request: Request):
    """改状态 / 角色 / 显示名。含**自保护**：不允许把自己降级或停用。
    """
    sess, err = _require_admin(request)
    if err:
        return err
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    actor = sess["user"]
    if int(user_id) == int(actor["user_id"]):
        return _err("不能修改自己的账号状态或角色", code=400)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")

    target = auth.get_user(int(user_id))
    if target is None or target.get("deleted_at"):
        return _err("用户不存在", code=404)

    if "status" in payload:
        new_status = str(payload["status"])
        # 自保护：不能把最后一个可用管理员停用（否则系统再无人可管）
        if (target.get("role") == "admin" and new_status != "active"
                and auth.count_active_admins(exclude_user_id=int(user_id)) == 0):
            return _err("这是最后一个可用管理员，不能停用", code=409)
        try:
            auth.set_status(int(user_id), new_status, actor_id=actor["user_id"],
                            ip=client_ip(request))
        except ValueError as e:
            return _err(str(e))

    if "role" in payload:
        new_role = str(payload["role"])
        if new_role not in ("admin", "user"):
            return _err("role 必须是 admin 或 user")
        if (target.get("role") == "admin" and new_role != "admin"
                and auth.count_active_admins(exclude_user_id=int(user_id)) == 0):
            return _err("这是最后一个可用管理员，不能降级", code=409)
        from . import db
        conn = db.get_conn()
        conn.execute("UPDATE users SET role=?, updated_at=? WHERE user_id=?",
                     (new_role, time.time(), int(user_id)))
        conn.commit()
        # P1-3（V36，2026-10-24 审计）：**降权必须撤销该用户的全部会话**。
        #
        # 为什么必须：`auth.require_csrf`／`/api/auth/me`／`_require_admin` 读的是
        # **缓存里的会话快照**（`sess["user"]["role"]`），缓存 TTL 30s。不撤销的话，
        # 一个被从 admin 降成 user 的人在被降级后的 ≤30s 内，其旧会话里仍写着
        # `role=admin`，任何依赖该快照的角色判断都会放过他。
        # 主闸门（中间件）每请求重读 DB，所以真实危害被压在 30s ——
        # 但"防御纵深里有一层持续 30s 失效"本身就是缺陷，且改密/停用都撤销了会话，
        # 唯独降权没撤销，属口径不一致。
        #
        # 与 set_status 保持同一套动作（revoke_user_sessions + invalidate_session_cache），
        # 避免出现第二套语义。
        if str(target.get("role")) != new_role:
            auth.revoke_user_sessions(int(user_id), reason=f"role_{new_role}")
            auth.invalidate_session_cache()
        auth.audit("user.role_change", actor_id=actor["user_id"],
                   target_id=int(user_id), ip=client_ip(request),
                   detail={"from": target.get("role"), "to": new_role})

    if "display_name" in payload:
        from . import db
        conn = db.get_conn()
        conn.execute("UPDATE users SET display_name=?, updated_at=? WHERE user_id=?",
                     (str(payload["display_name"])[:64], time.time(), int(user_id)))
        conn.commit()

    return {"ok": True, "user": _me_payload(auth.get_user(int(user_id)) or {})}


@router.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: int, request: Request):
    """管理员重置口令（响应与审计里**绝不能出现新口令**）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    new_pw = str(payload.get("new_password") or "")
    target = auth.get_user(int(user_id))
    if target is None or target.get("deleted_at"):
        return _err("用户不存在", code=404)
    try:
        await _offload(auth.set_password, int(user_id), new_pw,
                       actor_id=sess["user"]["user_id"],
                       action="user.password_reset",
                       ip=client_ip(request))
    except ValueError as e:
        return _err(str(e))
    return {"ok": True, "note": "口令已重置，该用户所有会话已失效"}


@router.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, request: Request):
    """软删除用户。含自保护（不能删自己 / 删最后一个管理员）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    actor = sess["user"]
    if int(user_id) == int(actor["user_id"]):
        return _err("不能删除自己的账号", code=400)
    target = auth.get_user(int(user_id))
    if target is None or target.get("deleted_at"):
        return _err("用户不存在", code=404)
    if (target.get("role") == "admin"
            and auth.count_active_admins(exclude_user_id=int(user_id)) == 0):
        return _err("这是最后一个可用管理员，不能删除", code=409)
    auth.soft_delete_user(int(user_id), actor_id=actor["user_id"],
                          ip=client_ip(request))
    return {"ok": True}


@router.get("/api/admin/audit")
def admin_audit(request: Request, limit: int = Query(100, ge=1, le=1000),
                action: Optional[str] = Query(None)):
    """审计查询（按时间倒序）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    from . import db
    conn = db.get_conn()
    if action:
        rows = conn.execute(
            "SELECT * FROM user_audit_log WHERE action=? ORDER BY audit_id DESC LIMIT ?",
            (action, int(limit))).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM user_audit_log ORDER BY audit_id DESC LIMIT ?",
            (int(limit),)).fetchall()
    return {"ok": True, "events": [dict(r) for r in rows]}
