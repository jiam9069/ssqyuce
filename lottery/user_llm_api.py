"""按用户隔离的 LLM 配置 API（W08a / MU-2 的前置切片）。

## 为什么单独一个模块，而不是塞进 auth_api.py

`auth_api.py` 的职责是「谁是谁」（身份与授权），本模块的职责是「你的模型用哪个 key」。
两者混在一起会让 `PUBLIC_EXACT` / `ADMIN_EXACT` 这类安全敏感表与业务配置逻辑纠缠，
后续 MU-2 真正落地 `user_llm_config` 表时改动面也无法收敛。分开后：
* `auth_api.required_role()` 只需登记 2 条路径；
* `user_llm_api` 只关心「当前用户」与「配置读写」。

## 当前实现范围（**诚实标注**）

本次只做 **HTTP 契约 + 归属判定 + 分级**，存储仍落在全局 `data/llm_config.json`，
`GET` 会带 `scope="global"` 明确告知前端「尚未真正隔离」。真正隔离（`user_llm_config`
表 + Fernet 加密 + 预算分键 + `config_fp` 缓存键）属于 MU-2，是需要改
`llm_client` / `engine` 的大块工作，不能靠改个端点的归属来假装完成。

**但归属判定必须现在做对**：如果 `/api/user/llm/config` 允许任何用户写全局文件，
那么普通用户改一次 key 就会把管理员的通道顶掉 —— 这是多用户上线即会发生的越权。
所以当前语义是：
* 普通用户读 → 返回自己的**逻辑视图**（未配置时如实返回「未配置」，不泄露管理员 key）；
* 普通用户写 → **403**，并明说原因（尚未隔离，请联系管理员）；
* 管理员写 → 落全局文件，`scope="global"`。
MU-2 完成后只需替换 `_load_view` / `_save_view` 两个函数，端点契约与前端不用再动。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import auth, config

router = APIRouter()


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------

def _err(msg: str, code: int = 400, **extra) -> JSONResponse:
    body: Dict[str, Any] = {"ok": False, "error": msg}
    body.update(extra)
    return JSONResponse(body, status_code=code)


def _session(request: Request) -> Optional[Dict[str, Any]]:
    from . import auth_api
    return auth_api.session_from_request(request)


def _csrf_ok(request: Request, sess: Dict[str, Any]) -> bool:
    return auth.require_csrf(sess.get("session") or {},
                            request.headers.get("x-csrf-token"))


def _read_global_file() -> Dict[str, Any]:
    """读 `data/llm_config.json`（**只读，不触发任何模型调用**）。"""
    import json
    try:
        if config.LLM_CONFIG_FILE.exists():
            data = json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (ValueError, OSError):
        pass
    return {}


def _mask_key(key: Optional[str]) -> str:
    """脱敏显示：`sk-****abcd`。

    规则（U2 §2.3）：长度 >= 8 时保留前 3 与后 4；更短的**全部隐去**，
    因为短 key 保留首尾会显著提高爆破命中率。
    """
    if not key:
        return ""
    if len(key) < 8:
        return "****"
    return f"{key[:3]}****{key[-4:]}"


def _public_view(data: Dict[str, Any], *, configured: bool,
                 scope: str, source: str) -> Dict[str, Any]:
    """对外字段白名单。**绝不含 api_key 明文**。

    这条不是"顺手脱敏"：`GET /api/config/llm` 原本连 `api_key` 都不返回，
    但前端 `loadLlmConfig()` 里写了 `config.api_key` 的读取分支——
    新增 `api_key_masked` 字段而非改 `api_key` 语义，避免既有代码误会。
    """
    return {
        "ok": True,
        "scope": scope,                 # "user" | "global" | "unbound"
        "source": source,               # "user_table" | "global_file" | "none"
        "configured": bool(configured),
        "disabled": bool(data.get("disabled", False)),
        "base_url": str(data.get("base_url") or ""),
        "model": str(data.get("model") or ""),
        "samples": int(data.get("samples") or 3),
        "api_key_masked": _mask_key(data.get("api_key")),
        "has_api_key": bool(data.get("api_key")),
    }


# ---------------------------------------------------------------------------
# 用户自己的配置
# ---------------------------------------------------------------------------

@router.get("/api/user/llm/config")
def get_user_llm_config(request: Request):
    """读**当前用户**的 LLM 配置视图。

    未配置 = 200 + `configured=false`，**不是** 404 —— 前端要区分"没配"和
    "接口坏了"，404 会被统一错误处理当成异常弹窗。

    响应**绝不含明文 Key**，只给 `api_key_masked` 与 `has_api_key`。
    """
    sess = _session(request)
    if not sess:
        return _err("未登录", code=401)
    user = sess["user"]
    uid = int(user["user_id"])

    from . import user_llm_store
    v = user_llm_store.get_view(uid)
    eff = user_llm_store.effective_for_user(uid)

    out = {
        "ok": True,
        "user_id": uid,
        # scope 语义（前端据此决定界面语气与可编辑性）：
        #   "user"   = 个人配置（本账号自己的，可编辑）
        #   "system" = 正在用系统共享通道（个人未配，回落到管理员那份）
        #   "none"   = 两条路都没有 → 纯统计出号
        "scope": eff["source"],
        "configured": bool(eff["llm_available"]),
        "source": eff["source"],
        "reason": eff.get("reason") or "",
        "use_custom_key": bool(v["use_custom_key"]),
        "disabled": bool(v["disabled"]),
        "base_url": v["base_url"],
        "model": v["model"],
        "model_list": v["model_list"],
        "samples": int(v["samples"]),
        "api_key_masked": v["api_key_masked"],
        "has_api_key": bool(v["has_api_key"]),
        "daily_token_quota": int(v["daily_token_quota"]),
        "today_tokens_used": int(v["today_tokens_used"]),
        "today_calls_used": int(v["today_calls_used"]),
        "qps_limit": int(v["qps_limit"]),
        "updated_at": v["updated_at"],
        # 兜底提示：没有可用通道时明确告诉用户"你仍然可以出号"
        "degrade_hint": ("" if eff["llm_available"] else
                         "当前账号没有可用的 LLM 通道，预测会走纯统计模式照常出号"
                         "（不需要任何 Key）。如需启用大模型，请在下方填写自己的配置。"),
        # 系统共享通道是否已被管理员配置（普通用户看得到"有没有"，看不到具体值）
        "system_channel_configured": bool(config.llm_configured()),
    }
    # learned_max_tokens 只对"当前实际生效的通道"有意义
    if eff["llm_available"] and eff["base_url"] and eff["model"]:
        try:
            from . import llm_client
            out["learned_max_tokens"] = llm_client.learned_max_tokens(
                eff["base_url"], eff["model"])
        except Exception:      # noqa: BLE001 —— 纯装饰信息
            out["learned_max_tokens"] = None
    return out


@router.post("/api/user/llm/config")
async def save_user_llm_config(request: Request):
    """保存**当前用户自己的** LLM 配置（MU-2 起对所有登录用户开放）。

    ## 为什么这里不再有 `role != admin → 403`

    MU-1 阶段这条端点的语义是"管理全局配置"，所以普通用户被拒。MU-2 之后它
    写的是 `user_llm_config` 里**这个用户自己那一行**，把"管理我自己的配置"
    判成需要管理员权限，与需求（每用户隔离配置）直接矛盾。

    越权防护改由**存储层**保证：`user_llm_store.save(uid, ...)` 的 uid 只来自
    会话，绝不接受请求体里的 user_id —— 用户无法给别人的行写配置。
    """
    sess = _session(request)
    if not sess:
        return _err("未登录", code=401)
    user = sess["user"]
    uid = int(user["user_id"])

    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    if not isinstance(payload, dict):
        return _err("请求体必须是 JSON 对象")

    # 显式拒绝越权字段：即便存储层有白名单，也要让"试图改别人"变成可见错误，
    # 而不是被白名单静默忽略（后者会让攻击者误以为可能有效，排查时也无痕迹）。
    if "user_id" in payload and int(payload.get("user_id") or 0) not in (0, uid):
        from . import auth_api
        auth_api.auth.audit(
            "authz.denied", actor_id=uid, ip=auth_api.client_ip(request),
            result="denied",
            detail={"path": "/api/user/llm/config", "reason": "user_id_mismatch"})
        return _err("不能修改其他用户的配置", code=403)

    # P1-2（V36，2026-10-24 审计）：保存时先校验 base_url 可安全外连。
    # 这是**即时反馈**用的第一道闸；真正的安全边界在 llm_client.chat()
    # （历史配置/全局通道/其他写入路径都从那里过）。
    if payload.get("base_url"):
        from . import url_guard
        _ok, _why = url_guard.check_llm_base_url(
            str(payload["base_url"]),
            allow_private=url_guard.allow_private_from_env())
        if not _ok:
            from . import auth_api
            auth_api.auth.audit(
                "llm_config.unsafe_url", actor_id=uid,
                ip=auth_api.client_ip(request), result="denied",
                detail={"reason": "unsafe_base_url"})
            return _err(_why, code=400)

    from . import user_llm_store
    try:
        user_llm_store.save(uid, payload)
    except Exception as e:      # noqa: BLE001
        # SecretBoxUnavailable（主密钥不可用）必须明确告知，**不能**静默存明文
        return _err(f"保存失败：{e}", code=503)

    from . import auth_api
    auth_api.auth.audit(
        "llm_config.user_save", actor_id=uid, ip=auth_api.client_ip(request),
        detail={"keys": sorted(k for k in payload if k != "api_key"),
                "key_changed": bool(payload.get("api_key"))})
    return get_user_llm_config(request)


# ---------------------------------------------------------------------------
# 管理员：全局默认
# ---------------------------------------------------------------------------

def _require_admin(request: Request):
    from . import auth_api
    return auth_api._require_admin(request)


@router.get("/api/admin/llm/default")
def get_global_llm_default(request: Request):
    """全局默认 LLM 配置（**管理员专用**，含 base_url/model，不含 key 明文）。"""
    sess, err = _require_admin(request)
    if err:
        return err
    return _public_view(_read_global_file(), configured=config.llm_configured(),
                        scope="global", source="global_file")


@router.post("/api/admin/llm/default")
async def save_global_llm_default(request: Request):
    """写全局默认配置。管理员专用（分级表里 `/api/admin/` 前缀已兜底，
    这里再自检一次，形成双保险）。"""
    from . import auth_api
    sess, err = _require_admin(request)
    if err:
        return err
    if not _csrf_ok(request, sess):
        return _err("CSRF 校验失败", code=403)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _err("请求体必须是 JSON")
    if not isinstance(payload, dict):
        return _err("请求体必须是 JSON 对象")
    from . import llm_config_store
    res = llm_config_store.save(payload)
    if not isinstance(res, dict) or not res.get("ok"):
        return _err((res or {}).get("error") or "保存失败", code=500)
    auth_api.auth.audit("llm_config.global_save",
                        actor_id=sess["user"]["user_id"],
                        ip=auth_api.client_ip(request),
                        detail={"keys": sorted(k for k in payload if k != "api_key")})
    return _public_view(_read_global_file(), configured=config.llm_configured(),
                        scope="global", source="global_file")
