"""FastAPI 应用：REST API + 静态单页前端。"""
from __future__ import annotations

import json
import contextvars
import hmac
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

class _NoCacheStaticFiles(StaticFiles):
    """静态资源禁用缓存，确保前端更新后浏览器立即拿到新版。"""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

from . import config, data_fetcher, db
from . import auth_api
from . import user_llm_api
from . import routes_llm
from . import routes_tasks
from . import routes_eval
from . import routes_llm_eval
from . import routes_methods
from . import routes_ml
from . import routes_rigor
from . import routes_portfolio
from . import routes_capabilities
from . import routes_data
from . import routes_patterns
from . import routes_predict
from . import routes_diagnosis
from . import routes_llm_ops
from . import routes_audit
from . import routes_rules
# P2：`_jsonable` 已被 10 个端点共用，提到 jsonutil 以免各域反向依赖 api_app。
# 保留同名别名，既有调用点与测试零改动。
from .jsonutil import jsonable as _jsonable
from .decision_skill import DECISION_SKILL_BASELINE  # noqa: E402  常量已随服务模块搬走
# P2：这些符号已随域切分搬走，但**既有测试与内部调用按名字 import 它们**。
# 名字是契约、实现位置不是 —— 这里显式再导出，避免破坏外部引用。
# 用 `# noqa: F401` 标明是有意保留的未使用导入。
from .decision_skill import (  # noqa: F401
    _passed_pattern_count,
    _usable_decision_count,
    _wilson_lower,
    decide_from_wilson,
    decision_skill,
    llm_marginal_value,
)
from .routes_portfolio import (  # noqa: F401
    _hypergeom,
    ticket_expected_prize,
)
TICKET_PRICE = config.TICKET_PRICE   # noqa: F401  已提到 config（领域常量）
# 数据域端点同样按名字被测试引用（`from lottery.api_app import app_info`），
# 再导出以保持契约。
from .routes_data import app_info, health  # noqa: F401
# 其余被测试按名字引用的内部符号（严格模式门控 / 决策自动关停）。
# 这两个是历史缺陷最集中的地方，测试直接打它们 —— 保留名字就是保留可测性。
from .routes_predict import (  # noqa: F401
    _strict_mode_reject_response,
    _strict_scope_for_current_user,
)
from .routes_llm_ops import _maybe_auto_disable_decision  # noqa: F401
from .bg_tasks import _bg_run  # noqa: F401  已提为共享基础设施（多域共用）
from .evolution_service import (  # noqa: F401  调度器与 API 共用
    _evolution_state_load,
    _evolution_state_path,
    _evolution_state_store,
    _evolution_step,
)
from . import migrations
from . import features as F, diagnose as D, mining as M

# 当前请求的用户上下文（仅 contextvar；**不用于 api_app→db 的租户传参**，
# 那条路径必须用显式参数 —— 裸 threading.Thread 不继承 contextvar，
# 丢失后会静默落到 SYSTEM 写错数据，而且测试不会红。见 Lead 裁决 9。）
#
# P2：contextvar 本身已搬到 `lottery/request_context.py`。原因是依赖方向倒置——
# `db.py` 和 `llm_client.py` 是**底层库**，却为了读一个 contextvar 反向依赖
# 3500 行的 Web 应用模块（靠把 import 写进函数体来躲循环）。搬到独立模块后
# 变成 `api_app → request_context ← db`，方向单向。
#
# 这里保留同名别名，是为了不破坏既有调用点与测试里的
# `monkeypatch.setattr(api_app, "_current_user_id", ...)`。
from .request_context import (                    # noqa: E402
    _current_user_id,
    current_user_id,
    reset_current_user_id,
    set_current_user_id,
)

app = FastAPI(
    title="双色球号码组合分析系统",
    version=config.APP_VERSION,
    # W01：文档端点默认关闭。`/docs` `/redoc` `/openapi.json` **不在 `/api/` 前缀下**，
    # 天然绕过 `optional_api_auth`，是当前生产最容易被忽略的暴露面（实测公网 200）。
    # 本地开发需要时用 `LOTT_API_DOCS=1` 打开。
    docs_url="/docs" if config.API_DOCS_ENABLED else None,
    redoc_url="/redoc" if config.API_DOCS_ENABLED else None,
    openapi_url="/openapi.json" if config.API_DOCS_ENABLED else None,
)
# W01：CORS 收敛到显式白名单（原先 `["*"]`）。前端 `app.js` 用相对路径 fetch，
# 属同源请求、本不依赖 CORS；放开 `*` 只会扩大第三方页面的跨域读取面。
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── MU-1 / W05：身份解析的**唯一入口**（Lead 裁决 8）──────────────────────
#
# 为什么必须是模块级函数，而不是中间件里的内联逻辑：
#   `app.dependency_overrides` 对 HTTP 中间件**无效**。若解析内联在中间件里，
#   测试就无法用一个 fixture 统一注入身份，只能把 36 处 `TestClient` 逐个改成
#   "注入真实 session cookie" —— 爆炸半径从 1 处变成 36 处。
#   外提之后 conftest 一行 monkeypatch 即全局生效：
#       monkeypatch.setattr(api_app, "_resolve_session_user_id", lambda r: ADMIN_ID)
#
# 三条纪律：① 模块级；② 同步签名（不起协程）；③ 不读 contextvar、不依赖中间件状态。
# **无有效会话时返回 None，绝不回退 admin / SYSTEM_USER_ID**（那是提权后门）。
def _resolve_session_user_id(request: Request):
    """Cookie → user_id；无效/未登录返回 ``None``。"""
    from . import auth_api
    sess = auth_api.session_from_request(request)
    if not sess:
        return None
    return int(sess["user"]["user_id"])


def _load_active_user_for_middleware(user_id: int):
    from . import auth
    return auth.load_active_user(int(user_id))


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """多用户鉴权（deny-by-default）+ 旧 LOTT_TOKEN 兼容。

    执行顺序（U1 §7.3，**顺序不可调换**）：
      1. 求所需角色（公开 → 直接放行）
      2. 解析身份（唯一入口 `_resolve_session_user_id`）
      3. **status 闸门**：只接受 `status='active'`（bootstrap/locked/disabled 一律 403）
      4. **角色闸门**：admin-only 路径校验角色
      5. 注入 contextvar（供 llm_client 等深层库读取），finally 必须清理

    第 3 步必须先于第 4 步：`status='bootstrap'` 的行可能写着 `role='admin'`，
    若先判角色再判 status，这一行就是一个**零密码管理员**。

    `config.AUTH_MODE != 'enforce'` 时只做"解析 + 记录"，不拦截 —— 这是留给
    "前端登录页尚未落地"的灰度期。切换前必须完成前端登录页与引导建号。
    """
    from . import auth_api

    path = request.url.path
    need = auth_api.required_role(path)

    # ① 旧 LOTT_TOKEN 兼容层（**先于**多用户分级，恢复既有语义）：
    #    配了 LOTT_TOKEN 时保护所有 /api/*，唯二例外是 /api/health（容器探活）
    #    与登录/引导入口（否则本身就要 token 才能登录，死锁）。
    #    这条曾经被写成"只对公开路径跳过校验"，导致配了 token 后 /api/info
    #    反而公开 —— 既有测试 test_m4_5_engineering 抓住了它。教训：
    #    **白名单只能收窄，不能因为"路径小"就顺手放行**。
    if config.API_TOKEN and path.startswith("/api/") \
            and path not in _TOKEN_EXEMPT:
        if not _token_ok(request):
            return _unauthorized_bearer()

    if need is None:
        return await call_next(request)

    # ② 解析身份
    uid = None
    try:
        uid = _resolve_session_user_id(request)
    except Exception:  # noqa: BLE001 —— 解析失败一律视为未登录，绝不放行
        uid = None

    # 灰度期：不拦截，但把身份写进 contextvar，便于后续按用户化改造渐进生效
    if not config.auth_enforced():
        # V36（P0-5）：必须同时打上"这是一次 HTTP 请求"的标记，否则
        # `db.resolve_tenant` 会把"无身份"误判成"后台线程"而兜底到主管理员，
        # 让匿名请求写别人的票单。见 request_context.in_request 的注释。
        from . import request_context as _rc
        _req_tok = _rc.set_in_request(True)
        if uid is not None:
            _current_user_id.set(uid)
        try:
            return await call_next(request)
        finally:
            _rc.reset_in_request(_req_tok)
            _current_user_id.set(None)

    if uid is None:
        # 旧 LOTT_TOKEN 仍可作为"机器对机器"的兼容通道（若部署时配了）
        if config.API_TOKEN and _token_ok(request):
            return await call_next(request)
        return JSONResponse({"ok": False, "error": "未登录",
                             "auth_required": True}, status_code=401,
                            headers={"WWW-Authenticate": "Cookie"})

    # ③ status 闸门（先于角色判断）
    user = _load_active_user_for_middleware(uid)
    if user is None:
        auth_api.auth.audit("authz.denied", actor_id=uid, ip=auth_api.client_ip(request),
                            result="denied",
                            detail={"path": path, "reason": "inactive_status"})
        return JSONResponse({"ok": False, "error": "账户不可用"}, status_code=403)

    # ④ 角色闸门
    if need == "admin" and user.get("role") != "admin":
        auth_api.auth.audit("authz.denied", actor_id=uid, ip=auth_api.client_ip(request),
                            result="denied",
                            detail={"path": path, "required": "admin"})
        return JSONResponse({"ok": False, "error": "需要管理员权限"}, status_code=403)

    # ⑤ contextvar：仅供 llm_client 这类**无法改签名**的深层库读取
    #    V36（P0-5）：同样要打 in_request 标记（详见 request_context 注释），
    #    否则已登录请求在 `resolve_tenant` 眼里和后台线程无法区分。
    from . import request_context as _rc2
    _req_tok2 = _rc2.set_in_request(True)
    _current_user_id.set(uid)
    try:
        return await call_next(request)
    finally:
        _rc2.reset_in_request(_req_tok2)
        _current_user_id.set(None)      # AnyIO 工作线程会被复用，不清理会串身份


# 旧 LOTT_TOKEN 的豁免路径（**必须保持极小**）：只有容器探活与登录入口。
# 其余 /api/* 在配置了 LOTT_TOKEN 时一律要求 Bearer —— 这是 V4 起就有的语义，
# 多用户改造不得放宽它（放宽等于给存量部署留了一个静默后门）。
_TOKEN_EXEMPT = frozenset({"/api/health", "/api/auth/login", "/api/auth/bootstrap",
                           "/api/auth/logout", "/api/auth/me"})


def _token_ok(request: Request) -> bool:
    """旧 LOTT_TOKEN 校验（Bearer）。

    ⚠️ **必须转 bytes**（BUG-1 同源，2026-10-21 实测）：`got` 来自 `Authorization`
    请求头，是外部可控 str；含非 ASCII 时 `hmac.compare_digest(str, str)` 抛
    TypeError → 该请求返回 500（而非 401）。任何人发 `Authorization: Bearer 中文`
    即可触发。三处同源缺陷见 `auth.require_csrf` / `auth.claim_bootstrap_token`。
    """
    if not config.API_TOKEN:
        return False
    got = request.headers.get("authorization", "")
    return hmac.compare_digest(str(got).encode("utf-8"),
                               f"Bearer {config.API_TOKEN}".encode("utf-8"))


def _unauthorized_bearer() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "需要 Bearer Token"}, status_code=401,
                        headers={"WWW-Authenticate": "Bearer"})


# 认证与管理路由（登录/登出/me/引导 + 用户管理 + 审计）
app.include_router(auth_api.router)
# W08a：按用户隔离的 LLM 配置（当前归属规则见 user_llm_api 模块 docstring）
app.include_router(user_llm_api.router)
# P2：按域切分的路由模块。api_app 只负责组装（建 app / 中间件 / include / 静态）。
app.include_router(routes_llm.router)
app.include_router(routes_tasks.router)
app.include_router(routes_eval.router)
app.include_router(routes_llm_eval.router)
app.include_router(routes_methods.router)
app.include_router(routes_ml.router)
app.include_router(routes_rigor.router)
app.include_router(routes_portfolio.router)
# D-04：能力清单（can / cannot / degraded / constants 四段，constants 实时读取）
app.include_router(routes_capabilities.router)
app.include_router(routes_data.router)
app.include_router(routes_patterns.router)
app.include_router(routes_predict.router)
app.include_router(routes_diagnosis.router)
app.include_router(routes_llm_ops.router)
app.include_router(routes_audit.router)
app.include_router(routes_rules.router)


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- 页面 ----------



@app.get("/")
def index():
    resp = FileResponse(WEB_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ---------- 开奖日自动调度（可选，LOTT_SCHEDULER=1） ----------

def _scheduler_step():
    """开奖日（二/四/日）21:35 后：抓取 → 在线对照 → **跑一轮进化** → 生成下期预测。

    V5 起在生成预测之前先跑进化（``_evolution_step``），这样「找规律 → 实战验证 →
    用已验证规律预测」是同一个调度周期内完成的，而不是靠人工手动触发。
    """
    from . import backtest as BT, engine, evaluate
    now = __import__("datetime").datetime.now()
    if now.weekday() not in config.DRAW_WEEKDAYS:
        return
    if now.strftime("%H:%M") < config.DRAW_TIME:
        return
    try:
        info = data_fetcher.fetch_and_update()
        print("[scheduler] 数据已更新:", info.get("inserted_new"), "期")
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 抓取失败:", e)
        return
    check_result = evaluate.online_check()
    from . import notify
    print("[scheduler] 通知结果:", notify.notify_after_check(check_result))

    # V5：每期开奖后自动跑一轮进化（失败不影响预测生成）
    try:
        _evolution_step()
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 进化失败（不影响预测）:", e)

    # ── MU-0 / W06：前置注册的**写入侧**（这是全系统最上游的一处断链）──
    #
    # 背景（V13 + MU 评审实测）：`hypothesis.register_prereg()` 此前在非测试代码里
    # **调用点为 0**，于是：
    #   register_prereg 没人调用 → 假说没有 prereg_status
    #   → evaluate_prereg 只处理 prereg_status ∈ {registered,running,provisional}
    #   → prereg_ok 永远不可能为 True → usable_rules() 恒空
    #   → 界面永远「可用规律 0 条」，且**注入多强的真实效应都是 0**。
    #
    # **顺序必须在 evaluate_prereg 之前**：本期扫描出的候选应当在同一轮调度里
    # 就进入注册簿（虽然它要等 150 期冻结期后才判定），否则每期都"晚一期"。
    # 失败只降级、绝不中断调度（与既有约定一致）。
    try:
        from . import prereg_scheduler
        # V36 修复（2026-10-24 审计 P1-12）：此处原先用 `draws`，但 `draws`
        # 要到本函数后半段（约 :400）才被赋值 ⇒ 必然 NameError，被下面的
        # `except` 静默吞掉，于是**前置注册的写入侧从未真正执行过**——
        # 而上面那段注释恰恰在宣告"这处断链已经接通"。注释不能当校验用。
        # 现在就地取数据，不再依赖后续的局部变量。
        _draws_for_prereg = db.load_draws()
        _scan = prereg_scheduler.register_scan_candidates(
            _draws_for_prereg,
            issue=str(_draws_for_prereg[-1]["issue"]) if _draws_for_prereg else "")
        print("[scheduler] 前置注册·写入:", {
            k: _scan.get(k) for k in
            ("ok", "scanned", "registered", "deduped", "judgeable", "error")})
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 前置注册·写入失败（不影响预测）:", e)

    # V10-02：前置注册判定（**唯一写者**的调用点，见 docs/V10_IMPROVEMENT_PLAN.md §3.3.5b）
    # 必须在快照刷新**之前**跑，否则本期已到期的实验要等下一期才生效（时序错误）。
    # 失败只降级、不中断（沿用 _evolution_step 的既有约定：判定失败只影响 usable 条数）。
    try:
        from . import hypothesis
        # issue=None → 由 evaluate_prereg 从库里读最近一期作为"当前期号"
        _prereg = hypothesis.evaluate_prereg()
        print("[scheduler] 前置注册判定:", {k: _prereg.get(k) for k in
                                            ("checked", "passed", "provisional", "failed",
                                             "invalidated", "n_tested_cumulative")})
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 前置注册判定失败（不影响预测）:", e)

    # V7：生成预测前先重建规律快照，让「刚验证出来的规律」本期就能进入出号链路。
    # 失败只记录、不影响预测（快照是可选加速层，缺失时引擎照常出号）。
    try:
        from . import patterns_runtime
        print("[scheduler] 规律快照:", patterns_runtime.refresh(db.load_draws()))
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 规律快照刷新失败（不影响预测）:", e)

    # V11：每期自动推进规律生命周期（退居二线判定）。
    # 顺序：在快照刷新**之后**、生成预测**之前** —— 这样本期被判定为
    # 退居二线（monitor）的规律，会通过下一轮 build_snapshot 注入的
    # lifecycle 状态立即从 usable_rules() 里消失，本期出号就不再使用它。
    # 失败只降级（沿用既定约定），绝不影响当期预测生成。
    try:
        from . import rule_lifecycle as _RL
        _cycle = _RL.run_cycle(db.load_draws(), use_llm=False, persist=True)
        _mig = _cycle.get("migrated") or []
        print("[scheduler] 生命周期:", "迁移", len(_mig),
              "条" if _mig else "无需迁移", "|",
              {k: v for k, v in (_cycle.get("counts") or {}).items()
               if k in ("active", "monitor", "unmanaged")})
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 生命周期推进失败（不影响预测）:", e)

    draws = db.load_draws()
    exp = BT.expected_issue(draws)
    if exp.get("expected_draws_missing"):
        print("[scheduler] 警告：数据源落后于开奖日历，缺", exp.get("missed_draws"), "期")
    issue = exp["issue"]
    if not db.load_predictions(issue):
        # V10-11：决策通道被关停时不再请求 LLM（省钱且不产出被关停通道的注），
        # 但预测**照常生成**（这就是与 LLM_DISABLED 的区别：后者会 503）。
        from . import methods as METH
        _llm_on = not bool(METH.resolve_channels().get("decision_disabled"))

        # ── V16：严格模式优先 + 降级必须可见 ──────────────────────────
        #
        # ## 修的是什么（2026-10-22 的真实故障）
        #
        # 严格模式门控此前只挂在 `/api/predict` 路由上，而这里**直接调
        # `engine.predict_next()`，绕过门控**。结果是：用户在页面上点"生成预测"
        # 收到 422「严格模式下不出号」，后台却悄悄往库里存了 5 注纯统计票
        # （实测 issue=2026122，5 条 method 全是 stat:*/uniform，无一条 llm:）。
        #
        # 用户要的语义（2026-10-23 明确）：
        #   ① 自动生成时**严格模式优先**；
        #   ② 严格模式不出号 → **自动降级统计模式**，但**必须标注清楚**是统计口径；
        #   ③ 手动生成时**弹窗让用户自己选**（已在 routes_predict + 前端实现）。
        #
        # 所以这里：先按同一套门控判定；被拒就降级为统计，并把
        # `source_mode="stat_fallback"` + 拒绝原因写进配置快照，
        # 让前端/校验层能一眼看出「这批票不是严格模式的产物」。
        from . import routes_predict as _RP
        _uid = 1                      # 调度器产出归属主租户（与 db.resolve_tenant 默认一致）
        _strict_active, _strict_bypass = _RP.strict_scope_for_user(_uid)
        _fallback = None              # None = 未降级
        if _strict_active:
            _errs = _RP.strict_mode_gates(draws, issue)
            if _errs:
                _fallback = {
                    "gates": [e[0] for e in _errs],
                    "reason": "；".join(e[1] for e in _errs),
                }
                _llm_on = False       # 降级为纯统计
                print("[scheduler] 严格模式不出号（门控未过：",
                      ", ".join(_fallback["gates"]),
                      "）→ 按用户既定策略降级为**统计模式**并标注 source_mode=stat_fallback")
        elif _strict_bypass:
            # 该账号没有 LLM 通道 → 严格模式本就不对他生效，走统计但同样标注
            _fallback = {"gates": ["no_llm_channel"], "reason": _strict_bypass}
            _llm_on = False

        _res = engine.predict_next(draws, use_llm=_llm_on)
        # ===== V28：调度器也必须做**逐注** L2 校验 =====
        #
        # 为什么不能只在 `/api/predict` 里做：V16 的教训就是「门控只在路由上，
        # 调度器直连 `engine.predict_next()` 绕过它」。票级校验同理 ——
        # 若只挂在路由，后台自动生成仍会落库不合规的 LLM 票，而前端看不到任何标记。
        #
        # 与路由的差别：调度器**不拒绝**（自动生成没有"用户重新点一次"的交互），
        # 而是把违规事实写进快照，让 `/api/predictions/last` 能显示出来。
        # 严格模式下若违规，则整批不落库更合适 —— 但那会与「严格模式优先→自动
        # 降级统计」的既定策略冲突，故这里统一为「照常落库 + 标注」，
        # 并在日志里明确报出。
        if isinstance(_res, dict):
            try:
                _l2v = _RP.verify_tickets_l2(_res.get("tickets") or [], draws)
                _res["l2_ticket_check"] = {
                    "ok": _l2v["ok"], "n_tickets": _l2v["n"],
                    "n_violating": _l2v["n_bad"], "violations": _l2v["bad"],
                    "note_zh": _l2v["note_zh"],
                }
                if _l2v["n_bad"]:
                    print("[scheduler] ⚠ L2 逐注校验未通过：",
                          _l2v["note_zh"])
            except Exception as e:      # noqa: BLE001
                print("[scheduler] L2 逐注校验出错（不影响票单）:", e)
        # 把降级事实写进快照（engine 已落库，这里补标口径）
        if _fallback is not None and isinstance(_res, dict):
            try:
                db.mark_prediction_source(
                    issue, source_mode="stat_fallback",
                    strict_gates=_fallback["gates"],
                    strict_reason=_fallback["reason"],
                    user_id=_uid,
                )
            except Exception as e:      # noqa: BLE001
                print("[scheduler] 降级标注写入失败（不影响票单）:", e)
        print("[scheduler] 已生成", issue, "预测",
              ("（严格模式未出号 → 统计模式 · 已标注 stat_fallback）" if _fallback
               else ("（LLM 决策通道已关停，本次为纯统计/ML）" if not _llm_on else "")))


def _scheduler_loop():
    import time as _time
    while True:
        try:
            _scheduler_step()
        except Exception as e:  # noqa: BLE001
            print("[scheduler] 异常:", e)
        _time.sleep(1800)


# ---------- 闲时挖掘常驻循环（可选，LOTT_IDLE_MINING=1） ----------

#: 两轮挖掘之间的间隔（秒）。可用 `LOTT_IDLE_INTERVAL` 覆盖。
#:
#: 调参依据（2026-10-20 实测）：单候选墙钟约 100 ms（全史 3216 期 + 限流）。
#: 空间总量 72656 个，纯计算约 2.0 小时。用户给定「3 天观察窗口」，
#: 要让 3 天覆盖全部空间，每轮的吞吐只要满足 72656 / 259200s ≈ 0.28 候选/秒即可。
#:
#: 默认 40 候选/轮 × 300 秒间隔 ≈ 0.133 候选/秒 → 3 天约 34 600 个（覆盖 48%）。
#: 要 3 天全覆盖则设 `LOTT_IDLE_INTERVAL=120`（40 候选/120s ≈ 0.33 候选/秒）。
#: CPU 侧不受影响：限流按真实 CPU 时间闭环控制，间隔只决定「多久启动下一批」。
IDLE_MINING_INTERVAL = int(os.environ.get("LOTT_IDLE_INTERVAL", "300") or "300")

#: 每轮判定的候选上限。可用 `LOTT_IDLE_BUDGET` 覆盖。
#: 轮要小、轮要多 —— 这样断点更细（中断损失小），也让 CPU 占空比更平稳，
#: 优于「一次跑几小时」把机器吃得忽高忽低。
IDLE_MINING_BUDGET = int(os.environ.get("LOTT_IDLE_BUDGET", "40") or "40")


def _idle_mining_step() -> dict:
    """跑一轮闲时挖掘。**失败只降级，绝不抛出**。

    与 `_scheduler_step` 的关键区别：这里**没有开奖日/时间的门控**。
    闲时挖掘的语义是「有空就跑一点」的常驻任务，不是「开奖日 21:35 后的
    一次性动作」。塞进 `_scheduler_step` 会让它一周只跑 3 次、每次只跑一轮，
    直接违背 P4 的细水长流设计（124 维度要分批走完）。
    """
    from . import idle_mining
    if not idle_mining.enabled():
        return {"ok": False, "reason": "idle_mining_disabled"}
    return idle_mining.run_once(budget_candidates=IDLE_MINING_BUDGET)


def _idle_mining_loop():
    import time as _time
    while True:
        try:
            r = _idle_mining_step()
            # 日志要能分辨「候选」（单候选过 α 阈、未校正）与「可用」
            # （通过 e-BH 族级校正）—— 两者混在一起会让运维以为挖到了假规律。
            print("[idle] 挖掘轮次:", {k: r.get(k) for k in
                                       ("ok", "processed", "completed_total",
                                        "candidates", "usable", "interrupted",
                                        "remaining", "reason")})
        except Exception as e:  # noqa: BLE001 —— 后台任务绝不能拖垮前台
            print("[idle] 异常:", e)
        _time.sleep(IDLE_MINING_INTERVAL)


def _popularity_warmup():
    """启动后台预热「大众偏好度」模型。

    ## 为什么需要这条线程（2026-10-21 实测）

    `PopularityModel.fit()` 同步跑在请求路径上，冷启动首次要 **7.3 秒**
    （进程外 HTTP 端到端 28.5s，含 numpy 惰性导入）。而
    `loadV10Cards()` 是**用户进「评估报告」就自动触发**的 ——
    等于每次服务重启后的第一个用户都要盯着两张卡转圈半分钟。

    拆解（3517 期实测）：
        load_draws()            0.06s
        fit() 总计               7.99s
          └─ _out_of_sample_check  ~7s   ← 200 次置换检验（纯 Python）
          └─ _ols(全量)            0.01s

    ## 为什么不改统计本身

    `_out_of_sample_check` 产出的是 `r2_increment` 与置换 p 值 —— 它们是
    **披露给用户的效应量**（"只够排序、不足以定值"的依据），删掉或减少 n_perm
    会让结论变弱。所以这里**一个字都不改统计**，只把耗时挪到用户到来之前。

    ## 为什么不是「懒加载 + 首请求等待」

    那正是现状（`get_model()` 在没缓存时同步拟合）。预热的意义是把
    「第一个用户承担 7 秒」改成「服务启动时承担 7 秒」。

    ## 失败必须静默

    模型拟合失败绝不能影响主链路（这是 `get_model()` 里既有的约定：
    `except Exception: pass`）。预热线程同样 —— 失败就打一行日志，
    端点侧仍会退回 503「模型未就绪」，前端已有对应降级。
    """
    import time as _time
    t0 = _time.time()
    try:
        # ---- 显式 import numpy：把惰性导入的代价也挪到用户到来之前 ----
        # 2026-10-21 实测：容器内端到端 28~29s，而纯 fit() 只有 6~7s，
        # 差值主要在首次 numpy 导入（sklearn 也会连带拖进来）。
        # 不预热掉它，第一个请求仍会付这笔钱。
        try:
            import numpy  # noqa: F401
        except Exception:        # noqa: BLE001
            pass

        from . import popularity

        # ---- 有缓存就直接加载，不必预热 ----
        # 预热只在「跨进程缓存缺失/失效」时才有意义：
        #   · 有缓存：`_load_cached` 约 0.2s，预热是纯浪费；
        #   · 无缓存：预热把 6~7s 挪到启动后、用户到来前，仍有价值。
        # 判断方式与 `get_model()` 内部完全一致（同一指纹口径），不另造逻辑。
        try:
            from . import db
            rows = popularity._training_rows(db.load_draws())
            if popularity._load_cached(rows) is not None:
                print("[warmup] 大众偏好度：命中落盘缓存，跳过预热（%.2fs）"
                      % (_time.time() - t0))
                return
        except Exception:          # noqa: BLE001 —— 判断失败就走预热，宁可多做不可漏做
            pass

        # ---- 带重试：请求路径已不再触发拟合（结构判据要求）， ----
        # 所以「预热失败 = 功能永久不可用」必须由这里兜住。
        # 每 30 秒重试一次，直到拟合成功或达到上限次数（避免无限刷日志）。
        # 重试间隔**可配**（`LOTT_POPULARITY_RETRY_INTERVAL`，默认 30s）。
        # 可配的意义：让「重试耗尽」的端到端验证能在几秒内跑完，而不是等 300s。
        #
        # ⚠️ 注意这个量与 `routes_portfolio._POPULARITY_RETRY_AFTER_SECONDS` 是
        # **两个不同的量，不许合并命名**：
        #   · 本值 = **后台**多久重试一次；
        #   · retry_after = 告诉**用户**多久后刷新。
        #   两者业务含义无关（一个是系统行为，一个是用户协议），
        #   换算出同一个数只是巧合。有测试守卫这一点。
        _RETRY_INTERVAL = _popularity_retry_interval()
        _MAX_ATTEMPTS = _popularity_max_attempts()
        m = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            m = popularity.get_model()
            if m.fitted:
                break
            print("[warmup] 第 %d 次拟合未成功，%.2fs 后重试（%d/%d）"
                  % (attempt, _RETRY_INTERVAL, attempt, _MAX_ATTEMPTS))
            _time.sleep(_RETRY_INTERVAL)

        # 重试**耗尽**（循环跑满仍未 fitted）：必须显式标记，
        # 否则端点会继续说「稍候会自动重试」—— 那是第三种假承诺。
        if m is None or not m.fitted:
            try:
                popularity.mark_retry_exhausted()
            except Exception:        # noqa: BLE001
                pass

        if m is not None and m.fitted:
            _elapsed = _time.time() - t0
            # 记录实测预热耗时 —— 供 `retry_after` 的一致性守卫比对
            # （承诺的重试间隔应与近期实测同量级，见 suggest_retry_after 的注释）
            try:
                popularity.record_warmup_seconds(_elapsed)
            except Exception:        # noqa: BLE001 —— 记录失败不影响预热
                pass
            print("[warmup] 大众偏好度模型已就绪: %.2fs（R²增量=%.5f）"
                  % (_elapsed, float(getattr(m, "r2_increment", 0.0) or 0.0)))
        else:
            print("[warmup] 大众偏好度模型未拟合（数据不足？）: %.2fs" % (_time.time() - t0))
    except Exception as e:      # noqa: BLE001 —— 预热失败绝不影响服务启动
        print("[warmup] 大众偏好度预热失败（不影响主链路）: %s" % e)


def _popularity_retry_interval() -> float:
    """后台预热的重试间隔（秒）。默认 30，可用 `LOTT_POPULARITY_RETRY_INTERVAL` 覆盖。

    **可配的唯一目的是测试**：验证「重试耗尽」的终态时，10 次 × 30s = 300s
    不可接受；设成 0.1 就能在几秒内跑完完整流程。

    ⚠️ 与 `routes_portfolio._POPULARITY_RETRY_AFTER_SECONDS`（告诉**用户**
    多久后刷新）是**两个不同的量**，不可合并命名。
    """
    raw = (os.environ.get("LOTT_POPULARITY_RETRY_INTERVAL") or "").strip()
    if not raw:
        return 30.0
    try:
        val = float(raw)
        return val if val > 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


def _popularity_max_attempts() -> int:
    """预热最多重试几次。默认 10，可用 `LOTT_POPULARITY_MAX_ATTEMPTS` 覆盖（测试用）。"""
    raw = (os.environ.get("LOTT_POPULARITY_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 10
    try:
        val = int(raw)
        return val if val > 0 else 10
    except (TypeError, ValueError):
        return 10


def _popularity_warmup_enabled() -> bool:
    """预热开关的**唯一**读取处（委托给 `popularity.warmup_enabled()`）。

    保留这一层薄封装的原因：`api_app` 在任何 import 阶段都不该直接依赖
    `popularity` 的内部实现细节；而判断逻辑本身必须只有一份。
    """
    from . import popularity
    return popularity.warmup_enabled()


# ---------- 启动：开奖日自动调度 + 闲时挖掘（均默认关闭，各自独立开关） ----------
if config.SCHEDULER_ENABLED:
    import threading
    threading.Thread(target=_scheduler_loop, daemon=True).start()

# P4：闲时挖掘用**独立线程**，不受 SCHEDULER_ENABLED 影响 ——
# 两个开关语义不同：LOTT_SCHEDULER 管「开奖日自动出号」，
# LOTT_IDLE_MINING 管「有空时慢慢挖规律」。绑在一起会让
# 「只想挖规律、不想自动出号」的场景无法配置。
if (os.environ.get("LOTT_IDLE_MINING") or "0").strip() == "1":
    import threading as _threading
    _threading.Thread(target=_idle_mining_loop, daemon=True).start()

# V14（2026-10-21）：大众偏好度模型启动预热。
# **默认开启**，与上面两个开关不同 —— 它不产生任何新行为，只是把
# 「第一个请求要等的 7 秒」挪到「服务启动时」。不需要它时设
# LOTT_POPULARITY_WARMUP=0 即可（例如跑单测时想省 7 秒）。
# 用 getenv 判断「显式关闭」而不是「显式开启」：漏配环境变量时应当
# 得到**好的**默认（用户不等待），而不是坏默认。
# 谓词统一走 `popularity.warmup_enabled()` —— 单一事实源。
# 端点也要用同一个判断来决定 503 该说「稍候自动就绪」还是「本进程内不会就绪」，
# 两处各读一次 env 迟早会漂移（本项目老毛病）。
if _popularity_warmup_enabled():
    import threading as _warm_threading
    _warm_threading.Thread(target=_popularity_warmup, daemon=True).start()
