"""LLM 推理通道：OpenAI 兼容端点，结构化 JSON 输出，带重试与降级。"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Dict, List, Optional

import requests

# 调用计量（M3.1 LLM 离线评估用；线程安全累加，仅统计成功返回的调用）
_usage = {"calls": 0, "prompt_chars": 0, "completion_chars": 0}
_usage_lock = threading.Lock()

from . import config
from . import url_guard as _url_guard


# ==========================================================================
# V10-11：成本口径（读响应 usage + append-only 成本日志）
#
# 为什么必须做（docs/reviews/R3_AI_REVIEW.md **G6**）：
#   改动前 ``chat()`` **完全没读** 响应里的 ``data["usage"]``（旧代码 R3 定位在
#   ``llm_client.py:241-251``），只有"提示字符数 / 完成字符数"这种与计费无关的
#   代理指标。于是所有"成本"报表的分母都建立在**没有数据**上：既不能说明花了
#   多少钱，也就无法回答"每 ¥1 换回多少可用决策"。
#
# 本节的四条纪律：
#   1. **全部出口记账**：正常返回 / 重试后成功 / 超时 / 异常（含"未配置"这类
#      未真正发请求的出口）都在 ``chat()`` 退出前落一行，不允许有静默出口。
#   2. **缺失 usage 时降级为 ``None``，绝不写 0**：0 会被当成"确实免费"混进均值，
#      把成本系统性拉低；``None`` 表示"未知"，报表按"已计费调用数"单独统计。
#   3. **日志绝不写密钥**：一行只有 model / tokens / purpose / yuan / ts / ok
#      （+ attempts / 价格假设）。**不写** base_url、api_key、Authorization、
#      请求/响应正文（提示词里可能含用户数据）。
#   4. **append-only 且不得撑爆磁盘**：按大小轮转到 ``.1``（默认保留 1 份），
#      单文件上限默认 2 MB（约 1.5~2 万行）。
# ==========================================================================

#: 成本日志文件名（放在 ``config.DATA_DIR``，即挂载卷，容器重启不丢）
COST_LOG_NAME = "llm_cost_log.jsonl"

#: 单文件上限（字节）与轮转保留份数（``llm_cost_log.jsonl.1``）
COST_LOG_MAX_BYTES = int(os.environ.get("LOTT_LLM_COST_LOG_MAX_BYTES", str(2 * 1024 * 1024)))
COST_LOG_ROTATE_KEEP = int(os.environ.get("LOTT_LLM_COST_LOG_ROTATE_KEEP", "1"))

#: 美元→人民币汇率（**仅用于把既有 ``LLM_EVAL_PRICE_PER_1M``(USD) 折算成人民币**，
#: 不是实时汇率；可在 .env 覆盖 ``LOTT_USD_CNY``）
USD_TO_CNY = float(os.environ.get("LOTT_USD_CNY", "7.2"))

_cost_lock = threading.Lock()


def cost_log_path():
    """成本日志路径。**函数而非模块常量**：``config.DATA_DIR`` 会被测试/部署替换。"""
    return config.DATA_DIR / COST_LOG_NAME


def price_per_1m_yuan() -> float:
    """估算价（人民币元 / 百万 token）——门槛口径，集中一处便于披露假设。

    优先级：``LOTT_LLM_PRICE_PER_1M_YUAN``（显式人民币价）＞
    ``config.LLM_EVAL_PRICE_PER_1M``(USD) × ``LOTT_USD_CNY``。
    """
    raw = os.environ.get("LOTT_LLM_PRICE_PER_1M_YUAN")
    if raw not in (None, ""):
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    try:
        usd = float(getattr(config, "LLM_EVAL_PRICE_PER_1M", 1.0))
    except (TypeError, ValueError):
        usd = 1.0
    return float(usd) * float(USD_TO_CNY)


def _yuan_for_tokens(tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    """按 token 数估算人民币花费；**任一份缺失即返回 None**（不拿 0 冒充）。"""
    if tokens_in is None and tokens_out is None:
        return None
    total = int(tokens_in or 0) + int(tokens_out or 0)
    return round(total / 1_000_000.0 * price_per_1m_yuan(), 8)


def parse_usage(data: object) -> Dict[str, Optional[int]]:
    """从响应体解析 ``usage.prompt_tokens`` / ``usage.completion_tokens``。

    返回 ``{"tokens_in": int|None, "tokens_out": int|None, "has_usage": bool}``。
    **缺失即 None**：缺字段、非 dict、非数字、负数一律不猜（尤其不写 0）。
    """
    out: Dict[str, Optional[int]] = {"tokens_in": None, "tokens_out": None,
                                     "has_usage": False}
    if not isinstance(data, dict):
        return out
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return out

    def _as_tokens(value):
        if isinstance(value, bool):          # bool 是 int 的子类，必须先行排除
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            return int(value) if value >= 0 else None
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    tokens_in = _as_tokens(usage.get("prompt_tokens"))
    tokens_out = _as_tokens(usage.get("completion_tokens"))
    if tokens_in is not None or tokens_out is not None:
        out["has_usage"] = True
    out["tokens_in"] = tokens_in
    out["tokens_out"] = tokens_out
    return out


#: 属于"决策/研究"性质的调用用途。``decision_disabled`` 只关这些，不关 Web 预测。
DECISION_PURPOSES = frozenset({"decision", "lab", "evolution", "hypothesis", "prereg"})


def infer_purpose(depth: int = 2) -> str:
    """调用方模块名（``engine`` / ``llm_lab`` / ``api_app`` / ``hypothesis`` …）。

    为什么用调用栈而不是让每个调用点传参：``chat()`` 的调用点分布在
    ``engine`` / ``llm_lab`` / ``hypothesis`` / ``api_app``，本任务的文件域不含
    那几个模块（不得改动）。因此把"调用方模块名"作为**兜底**用途标签，稳定、
    零侵入，且足够回答"钱花在出号轮还是研究轮"。调用点可以随时用显式
    ``purpose=`` 覆盖它（显式优先）。任何异常都退化为 ``"unspecified"``，
    **绝不**因为打标签而让调用失败。
    """
    try:
        import sys
        frame = sys._getframe(1)
        for _ in range(max(1, int(depth)) + 3):
            if frame is None:
                break
            name = str(frame.f_globals.get("__name__") or "")
            if name and name != __name__:
                short = name.rsplit(".", 1)[-1]
                return short or "unspecified"
            frame = frame.f_back
    except Exception:                             # noqa: BLE001 - 打标签绝不影响调用
        pass
    return "unspecified"


def _bill_new(model: str, purpose: Optional[str]) -> Dict:
    """开始一次 ``chat()`` 的账单累加器（跨重试累加，整个调用只落一行）。"""
    return {
        "model": str(model or ""),
        "purpose": str(purpose or "unspecified"),
        "attempts": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "usage_seen": 0,
        "t0": time.time(),
    }


def _bill_add_usage(bill: Dict, data: object) -> None:
    """把某次 HTTP 200 响应里的 usage 累加进账单（缺失即不累加、不补 0）。"""
    parsed = parse_usage(data)
    if not parsed["has_usage"]:
        return
    bill["usage_seen"] += 1
    if parsed["tokens_in"] is not None:
        bill["tokens_in"] += parsed["tokens_in"]
    if parsed["tokens_out"] is not None:
        bill["tokens_out"] += parsed["tokens_out"]


def _rotate_if_needed(path) -> None:
    """按大小轮转（append-only 且不撑爆磁盘）。保留 ``.1``，更老的直接丢弃。

    轮转是"丢弃最旧"策略：成本账本用于**近期**边际价值判断，不是审计凭证；
    真要长期留存请从 ``.1`` 备份出去（DEPLOY.md 的备份脚本已覆盖 data/）。
    """
    try:
        if not path.exists() or path.stat().st_size < COST_LOG_MAX_BYTES:
            return
        if COST_LOG_ROTATE_KEEP <= 0:
            path.unlink()
            return
        import shutil
        shutil.move(str(path), str(path) + ".1")
    except OSError as e:                          # pragma: no cover - 磁盘异常不阻塞调用
        print(f"[llm] 成本日志轮转失败: {e}")


def _append_cost_row(row: Dict) -> None:
    """append-only 写一行（失败只打印，**绝不影响** LLM 调用本身）。"""
    path = cost_log_path()
    try:
        with _cost_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:                          # pragma: no cover - 磁盘异常不阻塞调用
        print(f"[llm] 成本日志写入失败: {e}")


def _bill_finish(bill: Dict, *, ok: bool, error: Optional[str] = None) -> Dict:
    """结束账单：算出 yuan 并落一行，返回该行（供测试与调试）。"""
    usage_seen = int(bill.get("usage_seen") or 0)
    tokens_in = int(bill.get("tokens_in") or 0) if usage_seen else None
    tokens_out = int(bill.get("tokens_out") or 0) if usage_seen else None
    row = {
        "ts": round(float(bill.get("t0") or time.time()), 3),
        "model": bill.get("model") or "",
        "purpose": bill.get("purpose") or "unspecified",
        "tokens_in": tokens_in,              # None = 上游未返回 usage（未知，不是 0）
        "tokens_out": tokens_out,
        "yuan": _yuan_for_tokens(tokens_in, tokens_out),
        "ok": bool(ok),
        "attempts": int(bill.get("attempts") or 0),
        "price_per_1m_yuan": price_per_1m_yuan(),   # 披露估算假设，便于事后重算
        "usage_seen": usage_seen,
    }
    if error:
        row["error"] = str(error)[:200]
    _append_cost_row(row)

    # MU-2：把这次消费记到**发起人**名下（配额统计用）。
    #
    # 放在 `_bill_finish` 是因为它是全部 4 个出口（成功 / strict 抛错 / 降级 /
    # 超时）的**唯一交汇点** —— 在 `chat()` 的每个 return 前各插一次，
    # 迟早漏掉某个分支，表现为"某些调用不计入配额"。
    #
    # 归属用 `current_user_id()`（contextvar）：`llm_client` 是**无法改签名**的
    # 深层库，正是 Lead 裁决 9 里"contextvar 只留给这类库"的场景。
    # 无上下文（后台线程/调度器）时返回 None → 记为系统消费，不记到某个人头上。
    try:
        # P2：改依赖 `request_context`（纯状态模块），不再反向 import `api_app`
        from . import request_context
        _uid = request_context.current_user_id()
        if _uid is not None:
            from . import user_llm_store
            user_llm_store.record_usage(
                int(_uid),
                tokens=(int(tokens_in or 0) + int(tokens_out or 0)),
                calls=1 if ok else 0)
    except Exception:      # noqa: BLE001 —— 用量统计绝不影响主流程
        pass

    return row


def read_cost_log(limit: Optional[int] = None) -> List[Dict]:
    """读取成本日志（含已轮转的 ``.1``），按时间升序。坏行直接跳过。"""
    rows: List[Dict] = []
    path = cost_log_path()
    sources = [str(path) + ".1", str(path)]
    for src in sources:
        try:
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        rows.append(rec)
        except OSError:
            continue
    rows.sort(key=lambda r: float(r.get("ts") or 0.0))
    if limit:
        rows = rows[-int(limit):]
    return rows


def cost_report(limit: Optional[int] = None) -> Dict:
    """成本报表：总调用 / 总 token / 总花费 / 按 model 与 purpose 分组。

    **零数据时的口径**：``n_calls == 0`` → 金额与 token 全部返回 ``None``
    （而不是 0），并给出 ``reason_zh`` 说明"没有数据"；绝不用 0 假装有值。
    """
    rows = read_cost_log(limit=limit)
    billed = [r for r in rows if r.get("tokens_in") is not None or r.get("tokens_out") is not None]
    ok_rows = [r for r in rows if r.get("ok")]
    tokens_in = sum(int(r.get("tokens_in") or 0) for r in billed)
    tokens_out = sum(int(r.get("tokens_out") or 0) for r in billed)
    yuan = round(sum(float(r.get("yuan") or 0.0) for r in billed), 6)

    def _group(key: str) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for r in rows:
            name = str(r.get(key) or "unknown")
            g = out.setdefault(name, {"n_calls": 0, "n_billed": 0, "n_ok": 0,
                                      "tokens_in": 0, "tokens_out": 0, "yuan": 0.0})
            g["n_calls"] += 1
            if r.get("ok"):
                g["n_ok"] += 1
            if r in billed:
                g["n_billed"] += 1
                g["tokens_in"] += int(r.get("tokens_in") or 0)
                g["tokens_out"] += int(r.get("tokens_out") or 0)
                g["yuan"] = round(g["yuan"] + float(r.get("yuan") or 0.0), 6)
        # 分组内没有已计费行时，金额/token 一律 None（不拿 0 冒充）
        for g in out.values():
            if g["n_billed"] == 0:
                g["tokens_in"] = None
                g["tokens_out"] = None
                g["yuan"] = None
        return out

    if not rows:
        return {
            "ok": True,
            "available": False,
            "n_calls": 0,
            "n_billed": 0,
            "n_ok": 0,
            "tokens_in": None,
            "tokens_out": None,
            "tokens_total": None,
            "yuan": None,
            "price_per_1m_yuan": price_per_1m_yuan(),
            "by_model": {},
            "by_purpose": {},
            "first_ts": None,
            "last_ts": None,
            "log_path": str(cost_log_path()),
            "reason_zh": ("成本日志为空（还没有可计费的 LLM 调用记录）："
                          "总花费/总 token 一律返回 null，**不**用 0 假装有值。"),
        }
    return {
        "ok": True,
        "available": True,
        "n_calls": len(rows),
        "n_billed": len(billed),
        "n_ok": len(ok_rows),
        "tokens_in": tokens_in if billed else None,
        "tokens_out": tokens_out if billed else None,
        "tokens_total": (tokens_in + tokens_out) if billed else None,
        "yuan": yuan if billed else None,
        "price_per_1m_yuan": price_per_1m_yuan(),
        "by_model": _group("model"),
        "by_purpose": _group("purpose"),
        "first_ts": rows[0].get("ts"),
        "last_ts": rows[-1].get("ts"),
        "log_path": str(cost_log_path()),
        "price_note_zh": (
            "金额是**估算**：yuan = (tokens_in + tokens_out) / 1e6 × price_per_1m_yuan。"
            "价格来自 LOTT_LLM_PRICE_PER_1M_YUAN，缺省 = LOTT_LLM_EVAL_PRICE_PER_1M(USD) "
            f"× LOTT_USD_CNY(={USD_TO_CNY})，不是账单原文；token 数才是账本口径。"),
        "unknown_usage_note_zh": (
            f"{len(rows) - len(billed)} 次调用的上游响应**没有 usage 字段**（或调用未真正发出），"
            "这些行的 token/金额为 null 而不是 0 —— 它们不参与均值与边际价值的分母。"),
    }


# ---------- 推理型模型的「可用输出预算」学习（线程安全，落盘容错） ----------
# 背景：部分推理型模型（deepseek-v4-flash / glm-5.3 等）会把整个 max_tokens 预算
# 花在 reasoning_content 上，content 为空。不同模型需要的预算差一个数量级
# （实测同一个模型 8000 不够、32000 才吐出正文），因此：
#   1) 一旦探测到「只推理、无正文」，就把预算抬到 LOTT_LLM_MAX_TOKENS 重试；
#   2) 成功时记住该模型实际可行的预算，之后的调用直接从该预算起步，避免每次
#      预测都从小预算被截断重试（省一次 30~50s 的往返）。
_budget_lock = threading.RLock()  # 可重入：remember_max_tokens 内层会再取 _budget_load
_budget_cache: Optional[Dict[str, int]] = None


def _budget_file():
    return config.DATA_DIR / "llm_budget.json"


def _budget_key(url: str, model: str) -> str:
    return f"{url}|{model}"


def _budget_load() -> Dict[str, int]:
    global _budget_cache
    with _budget_lock:
        if _budget_cache is not None:
            return _budget_cache
        data: Dict[str, int] = {}
        try:
            path = _budget_file()
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = {str(k): int(v) for k, v in raw.items()
                            if isinstance(v, (int, float)) and v > 0}
        except (ValueError, OSError, TypeError):
            data = {}
        _budget_cache = data
        return data


def learned_max_tokens(url: str, model: str) -> Optional[int]:
    """该 (端点, 模型) 已知可行的输出预算；未学到返回 None。"""
    return _budget_load().get(_budget_key(url, model))


def remember_max_tokens(url: str, model: str, max_tokens: int) -> None:
    """记住某模型实际可行的输出预算（仅变大时落盘）。"""
    try:
        mt = int(max_tokens)
    except (TypeError, ValueError):
        return
    if mt <= 0:
        return
    with _budget_lock:
        data = _budget_load()
        key = _budget_key(url, model)
        if data.get(key, 0) >= mt:
            return
        data[key] = mt
        try:
            path = _budget_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(path))
        except OSError as e:
            print(f"[llm] 输出预算记忆写入失败: {e}")


def reset_learned_budgets() -> None:
    """清空内存中的预算记忆（测试与切换通道后调用；不删除落盘文件）。"""
    global _budget_cache
    with _budget_lock:
        _budget_cache = {}


class LLMChannelError(RuntimeError):
    """LLM 通道不可用 / 超时（异常消息面向用户可读）。

    M4.5 快速失败：Web 预测明确要求使用大模型（use_llm=true）时，
    通道未配置、被停用、HTTP 错误、读取超时或总耗时超限都会抛出本异常，
    由 API 层转换为「预测失败」的明确提示，不再静默降级为纯统计模型。
    """


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON（容忍 markdown 代码块与前后杂讯）。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    # 尝试整体解析
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 提取第一个平衡的 {...}
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def chat(system: str, user: str, max_tokens: int = 2000,
         temperature: float = 0.8, timeout: Optional[float] = None,
         model_cfg: Optional[Dict] = None,
         deadline: Optional[float] = None, strict: bool = False,
         purpose: Optional[str] = None) -> Optional[str]:
    """调用 OpenAI 兼容 chat/completions，返回 content。

    model_cfg: {"name","base_url","api_key","model"}；缺省用主通道配置。
    仓库不内置任何 URL / Key：未配置通道时返回 None（上层降级为统计模型）。
    deadline: 单次预测 LLM 阶段的墙钟截止时间戳（time.time() 口径）；
              每次尝试的单次超时会被钳制在剩余预算内，超预算立即放弃。
    strict:  True 时任何失败（未配置/HTTP 错误/超时/空返回）抛 LLMChannelError，
            供 Web 预测快速失败；False 保持原有降级语义（返回 None）。
    purpose: V10-11 成本日志的用途标签（``decision`` / ``lab`` / ``evolution`` …
             或任意自由标签）。None → 用调用方模块名兜底（见 :func:`infer_purpose`）。

    **成本记账（V10-11）**：本函数的**每一个**出口（停用/未配置/正常返回/
    重试后成功/超时/异常）都会写一行 ``data/llm_cost_log.jsonl``；
    上游未返回 ``usage`` 时 token 记为 ``None``（未知），绝不写 0。
    """
    _purpose = str(purpose) if purpose else infer_purpose(depth=2)
    if config.LLM_DISABLED:
        bill = _bill_new("", _purpose)
        _bill_finish(bill, ok=False, error="llm_disabled")
        if strict:
            raise LLMChannelError("LLM 已被停用（设置页「停用 LLM」或 LOTT_LLM_DISABLED=1）")
        return None
    requested_max_tokens = int(max_tokens)
    if model_cfg is None:
        if not (config.LLM_BASE_URL and config.LLM_API_KEY and config.LLM_MODEL_LIST):
            msg = ("LLM 通道未配置（请设置 LOTT_LLM_BASE_URL / LOTT_LLM_API_KEY / "
                   "LOTT_LLM_MODEL，或在设置页配置并保存）")
            bill = _bill_new(str(getattr(config, "LLM_MODEL", "") or ""), _purpose)
            _bill_finish(bill, ok=False, error="not_configured")
            if strict:
                raise LLMChannelError(msg)
            print("[llm] " + msg + "，降级为纯统计模型")
            return None
        base_url = config.LLM_BASE_URL
        url = base_url + "/chat/completions"
        api_key = config.LLM_API_KEY
        model = config.LLM_MODEL_LIST[0]
    else:
        # V38（2026-10-24 审计 P2）：**必需字段缺失要给可执行错误，不能 KeyError → 500**。
        #
        # 用户配置是任意 JSON（`/api/user/llm/config` 接受 dict），只要漏了
        # `model` / `base_url` / `api_key` 里任何一个，这里就会抛 KeyError，
        # 上层 except 成 500「预测引擎异常: KeyError」—— 用户完全不知道要改什么。
        # 而"配了一半"是极常见的中间状态（先填地址、还没来得及填模型名）。
        _missing = [k for k in ("base_url", "model", "api_key")
                    if not str(model_cfg.get(k) or "").strip()]
        if _missing:
            msg = (f"LLM 配置不完整：缺少 {'、'.join(_missing)}。"
                   "请到设置页补齐这三项（地址/模型名/API Key）后重试；"
                   "或先取消勾选「使用大模型」，改用统计模式出号。")
            bill = _bill_new(str(model_cfg.get("model") or ""), _purpose)
            _bill_finish(bill, ok=False, error="incomplete_config")
            if strict:
                raise LLMChannelError(msg)
            print("[llm] " + msg + "，降级为纯统计模型")
            return None
        base_url = str(model_cfg["base_url"]).rstrip("/")
        # P1-2（V36，2026-10-24 审计）：**发请求前的最后一道闸**。
        # 只在校验保存侧不够 —— 历史已存配置、管理员全局通道、将来任何
        # 绕过 save 的写入路径都直接到这里。用户自填 base_url 让服务器去
        # POST，构成认证后 SSRF，且错误路径会把响应前 200 字符回传。
        _ok, _why = _url_guard.check_llm_base_url(
            base_url, allow_private=_url_guard.allow_private_from_env())
        if not _ok:
            msg = ("LLM 服务地址不可用：" + _why +
                   " 如这是自建内网服务，请让管理员设置 "
                   "LOTT_LLM_ALLOW_PRIVATE_URL=1 后再试。")
            bill = _bill_new(str(model_cfg.get("model") or ""), _purpose)
            _bill_finish(bill, ok=False, error="unsafe_base_url")
            if strict:
                raise LLMChannelError(msg)
            print("[llm] " + msg + "，降级为纯统计模型")
            return None
        url = base_url + "/chat/completions"
        api_key = str(model_cfg["api_key"])
        model = str(model_cfg["model"])
    # 账单累加器：跨重试累加 usage，整个 chat() 只落一行（含 attempts）。
    bill = _bill_new(model, _purpose)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # M4.5：合并请求级附加参数（reasoning_effort / enable_thinking 等）。
    # 默认 LOTT_LLM_EXTRA_BODY 应用于所有模型；LOTT_LLM_EXTRA_BODY_MAP 按模型覆盖。
    # 核心请求字段受保护，附加参数只能新增网关特有开关，不能改写请求主体。
    extra = dict(config.LLM_EXTRA_BODY_DEFAULT)
    per_model = config.LLM_EXTRA_BODY_BY_MODEL.get(model)
    if per_model:
        extra.update(per_model)
    for k in ("model", "messages", "max_tokens", "temperature"):
        extra.pop(k, None)
    if extra:
        payload.update(extra)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # 该模型此前学到的可用输出预算：直接从它起步，省掉一次被截断的往返。
    # 只对「正式轮次」（观察/选号/连接探测，≥1000）生效：critique(600) 这类
    # 辅助小请求不值得为它多花几十秒推理预算。
    _learned = learned_max_tokens(base_url, model)
    if _learned and requested_max_tokens >= 1000:
        _cap = int(config.LLM_MAX_TOKENS) if int(config.LLM_MAX_TOKENS) > 0 else int(_learned)
        _target = min(int(_learned), _cap)
        if _target > int(payload.get("max_tokens", 0)):
            payload["max_tokens"] = _target
            # 大预算单次就是几十秒，直接用宽松超时，别先被 60s 判超时白重试一轮
            timeout = max(timeout or 0, config.LLM_LONG_TIMEOUT)
    last_err = None
    reasoning_escalated = False  # 推理型模型：已注入 thinking:disabled 并抬升输出预算
    timeout_escalated = False    # 读取超时：已注入 thinking:disabled 并抬升输出预算
    rate_retries = 0  # 429/5xx 瞬态错误退避重试计数（不消耗 attempt）
    t_start = time.time()  # M3.4：单次 chat 总耗时硬上限，防止上游挂起拖死整条链
    attempt = 0
    while attempt < 3:
        attempt += 1
        if time.time() - t_start > 600:
            last_err = f"chat 总耗时超过 600s 上限（当前第 {attempt} 次尝试），放弃"
            break
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 1.0:
                last_err = "LLM 总耗时超限（超出本次预测的大模型时间预算，放弃后续尝试）"
                break
            # 单次尝试超时钳制在剩余预算内，保证整条链在预算内返回
            eff_timeout = min(timeout or config.LLM_TIMEOUT, remaining)
        else:
            eff_timeout = timeout or config.LLM_TIMEOUT
        try:
            # V10-11：真实发出的请求都算一次尝试（含最终抛异常的那些）
            bill["attempts"] = int(bill.get("attempts") or 0) + 1
            r = requests.post(url, json=payload, headers=headers, timeout=eff_timeout)
            if r.status_code != 200:
                # P1-2：回显要脱敏 —— 上游可能是任意用户指定地址，
                # 原样回吐响应体等于把内网探测变成「可读响应」。
                last_err = f"HTTP {r.status_code}: {_redact_echo(r.text)}"
                # 429 限流 / 5xx 网关错误属瞬态：短退避后重试（不消耗 attempt，限 3 次）
                if (r.status_code == 429 or r.status_code >= 500) and rate_retries < 3:
                    rate_retries += 1
                    sleep_s = min(2.0 * rate_retries, 6.0)
                    if deadline is not None:
                        sleep_s = min(sleep_s, max(0.0, deadline - time.time() - 1.0))
                    if sleep_s >= 0.5:
                        print(f"[llm] HTTP {r.status_code}（瞬态），退避 {sleep_s:.0f}s 重试"
                              f"（第 {rate_retries}/3 次）")
                        time.sleep(sleep_s)
                        attempt -= 1
                        continue
                continue
            data = r.json()
            # V10-11：**每一次** HTTP 200 都读 usage（含后续重试/空正文的 200）——
            # 上游对这些请求同样计费，漏记就等于把成本算低。
            _bill_add_usage(bill, data)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if content:
                with _usage_lock:
                    _usage["calls"] += 1
                    _usage["prompt_chars"] += sum(
                        len(m.get("content") or "") for m in payload["messages"])
                    _usage["completion_chars"] += len(content)
                remember_max_tokens(base_url, model, payload.get("max_tokens", 0))
                _bill_finish(bill, ok=True)          # 出口 1：正常返回 / 重试后成功
                return content
            # content 为空：可能 reasoning_content 吃光了 max_tokens（推理型模型）
            finish = data["choices"][0].get("finish_reason")
            reasoning = msg.get("reasoning_content") or ""
            # U6/推理型兜底：注入 thinking:disabled（部分网关有效），并把输出预算一次性
            # 抬到 LOTT_LLM_MAX_TOKENS —— 实测小步放大（×3）不足以让这类模型吐出正文。
            if not reasoning_escalated and reasoning and finish in ("length", "stop"):
                reasoning_escalated = True
                payload.pop("reasoning_effort", None)
                payload["thinking"] = {"type": "disabled"}
                cap = max(int(config.LLM_MAX_TOKENS), int(payload.get("max_tokens") or 0))
                if cap > int(payload.get("max_tokens") or 0):
                    payload["max_tokens"] = cap
                    print(f"[llm] {model} 只返回推理内容（reasoning_content {len(reasoning)} 字），"
                          f"注入 thinking:disabled 并把输出预算抬到 {cap} 重试一次")
                else:
                    print(f"[llm] {model} 只返回推理内容（reasoning_content {len(reasoning)} 字），"
                          f"注入 thinking:disabled 重试一次（预算已在上限 {cap}）")
                timeout = max(timeout or 0, config.LLM_LONG_TIMEOUT)
                attempt -= 1
                continue
            if reasoning:
                last_err = (f"模型只返回推理内容（reasoning_content {len(reasoning)} 字），"
                            f"{payload.get('max_tokens')} 输出预算内未产出正文；"
                            f"请更换模型或调大 LOTT_LLM_MAX_TOKENS")
            else:
                last_err = f"模型返回空 content（finish_reason={finish}）"
            break
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            # U6：读取超时也可能是推理型模型在海量 reasoning 上耗时；先注入 thinking:disabled
            # 并把输出预算抬到上限，再重试一次。
            if isinstance(e, requests.exceptions.ReadTimeout) and not timeout_escalated:
                timeout_escalated = True
                payload.pop("reasoning_effort", None)
                payload["thinking"] = {"type": "disabled"}
                cap = max(int(config.LLM_MAX_TOKENS), int(payload.get("max_tokens") or 0))
                payload["max_tokens"] = cap
                timeout = max(timeout or 0, config.LLM_LONG_TIMEOUT)
                print(f"[llm] 读取超时，注入 thinking:disabled 并把输出预算抬到 {cap} 重试一次")
                attempt -= 1
                continue
    if strict:
        _bill_finish(bill, ok=False, error=last_err)      # 出口 3：strict 抛错
        raise LLMChannelError(f"LLM 调用失败: {last_err}")
    # 降级：记录但不抛出，让上层走统计兜底
    _bill_finish(bill, ok=False, error=last_err)          # 出口 4：超时/异常后降级
    print(f"[llm] 调用失败（3 次重试后）: {last_err}")
    return None


def chat_json(system: str, user: str, max_tokens: int = 2000,
              temperature: float = 0.8,
              model_cfg: Optional[Dict] = None,
              deadline: Optional[float] = None,
              strict: bool = False,
              timeout: Optional[float] = None,
              purpose: Optional[str] = None) -> Optional[dict]:
    text = chat(system, user, max_tokens=max_tokens, temperature=temperature,
                model_cfg=model_cfg, deadline=deadline, strict=strict, timeout=timeout,
                purpose=purpose)
    if not text:
        return None
    return _extract_json(text)


def compact_stats(stats: dict) -> dict:
    """把完整统计报告压缩为 LLM 友好摘要（去掉 33/16 维全量数组）。"""
    d = {"issue": stats["issue"], "last_reds": stats["last_reds"], "last_blue": stats["last_blue"]}
    for wname, w in stats["windows"].items():
        r, b = w["red"], w["blue"]
        freq_sorted = sorted(enumerate(r["freq"]), key=lambda x: -x[1])
        om_sorted = sorted(enumerate(r["omission_current"]), key=lambda x: -x[1])
        bf_sorted = sorted(enumerate(b["freq"]), key=lambda x: -x[1])
        bo_sorted = sorted(enumerate(b["omission_current"]), key=lambda x: -x[1])
        d[wname] = {
            "n": r["n_draws"],
            "红球频率TOP8": [(i + 1, int(v)) for i, v in freq_sorted[:8]],
            "红球频率BOTTOM8": [(i + 1, int(v)) for i, v in freq_sorted[-8:]],
            "红球当前遗漏TOP8": [(i + 1, int(v)) for i, v in om_sorted[:8]],
            "和值均值/分位": {"mean": round(r["sum_mean"], 1), "pct": {p: round(v, 1) for p, v in r["sum_pct"].items()}},
            "三区比历史Top5": list(r["zone_hist"].items())[:5],
            "奇偶分布": r["odd_hist"],
            "重号均值": round(r["repeat_mean"], 3),
            "连号率/同尾率": [round(r["consecutive_rate"], 3), round(r["same_tail_rate"], 3)],
            "蓝球频率TOP5": [(i + 1, int(v)) for i, v in bf_sorted[:5]],
            "蓝球遗漏TOP5": [(i + 1, int(v)) for i, v in bo_sorted[:5]],
            "蓝球重号率": round(b["repeat_rate"], 3),
        }
    d["recent"] = stats["recent"]
    return d



def critique_prompt(stats_json, recent, patterns, observations, tickets, feedback=None):
    """第3轮：质疑选号（输出 JSON verdict/issues/suggestions）。"""
    return (
        "你是审稿人。请审查以下候选号码，找出结构性问题（和值/三区/奇偶/跨度极端、蓝球过度集中、"
        "无依据的号码、与回测规律矛盾等）。保持双色球随机性的诚实立场，不要夸大任何信号。\n"
        f"## 统计摘要\n{str(stats_json)[:500]}\n"
        f"## 回测规律\n{str(patterns)[:400]}\n"
        f"## 上期回馈\n{str(feedback or {})[:200]}\n"
        f"## 候选号码\n{json.dumps(tickets, ensure_ascii=False)[:1200]}\n"
        '仅输出 JSON：{"verdict":"ok"|"problematic","issues":["问题1","问题2"],'
        '"suggestions":{"0":{"reds":[6个红球],"blue":1,"confidence":0-100,"reasoning":"修正理由"}}}；'
        "若无重大问题，输出 {\"verdict\":\"ok\"}，不要修改任何号码。"
    )


def refine_prompt(stats_json, critique, tickets, feedback=None):
    """第3轮：基于批判意见生成修正后的选号（与 tickets_prompt 相同 schema）。"""
    return (
        "基于审稿意见修正候选号码，只修正被指出的问题，其余号码尽量保留。\n"
        f"## 统计报告\n```json\n{json.dumps(stats_json, ensure_ascii=False) if isinstance(stats_json, dict) else stats_json}\n```\n"
        f"## 审稿意见\n{json.dumps(critique, ensure_ascii=False)[:800]}\n"
        f"## 原候选\n{json.dumps(tickets, ensure_ascii=False)[:1400]}\n"
        f"## 上期回馈\n{str(feedback or {})[:200]}\n"
        f"请生成 {config.TICKETS_PER_LLM_CALL} 注修正候选，输出与 tickets_prompt 相同 schema"
        "（含 evidence / counter_evidence / structure_scores）。"
    )


SYSTEM_BASE = (
    "你是双色球数据分析助手。双色球每期从1-33中摇出6个红球、从1-16中摇出1个蓝球，"
    "开奖在理论上是独立随机事件。你的任务是：基于给定的统计报告与历史走势，"
    "给出「可检验的结构化观察」与「选号建议」，并如实承认不存在稳定可预测的规律。"
    "所有输出必须是合法JSON，不要输出任何JSON之外的文字。"
)


def observations_prompt(stats_json: dict, recent: list, patterns: list,
                            feedback: Optional[dict] = None) -> str:
    """第 1 轮：让 LLM 归纳长/中/短期的可检验观察（含规律明细与上期回馈）。"""
    fb = f"## 上期预测回馈（命中情况）\n{json.dumps(feedback, ensure_ascii=False)}\n\n" if feedback else ""
    return (
        "以下是本期开奖前的统计报告（长/中/短三个窗口）、最近20期走势、样本外回测规律明细"
        "（含样本量/边际/p_adj/威尔逊区间）以及上期预测回馈。\n\n"
        f"## 统计报告 JSON\n```json\n{json.dumps(stats_json, ensure_ascii=False)}\n```\n\n"
        f"## 最近走势\n{json.dumps(recent, ensure_ascii=False)}\n\n"
        f"## 已回测规律明细（仅 B/C 级弱信号，n/边际/p_adj/威尔逊区间）\n{json.dumps(patterns, ensure_ascii=False)}\n\n"
        + fb
        + '请输出：{"long_term": ["观察1(附统计依据)", ...3-5条], "mid_term": [...], '
        '"short_term": [...], "caveats": ["承认随机性与不可预测的说明"]}\n'
        "每条观察必须引用具体数字（频率/遗漏/区间比等），不得空谈。"
    )


def tickets_prompt(stats_json: dict, recent: list, patterns: list, observations: dict,
                     feedback: Optional[dict] = None,
                     red_probs: Optional[list] = None,
                     blue_probs: Optional[list] = None,
                     regime_rules: Optional[list] = None,
                     shape_ctx: Optional[dict] = None) -> str:
    """第 2 轮：组合结构优化（U3 LLM 角色重定位）。

    双色球独立随机，LLM 不再“自由猜测号码”。给定统计模型/ML 的 33/16 维概率
    分布，LLM 只在硬约束下做**组合结构优化**（和值/三区/奇偶/跨度均衡、红球
    分散），并从高概率红球区间内挑红球；蓝球由专用 blue_specialist 决定，
    这里给出的 blue 仅为提示（引擎会以 blue_specialist 覆盖）。
    """
    fb = f"## 上期预测回馈（命中情况）\n{json.dumps(feedback, ensure_ascii=False)}\n" if feedback else ""
    red_hint = ""
    if red_probs:
        ranked = sorted(enumerate(red_probs, start=1), key=lambda x: -x[1])
        top = ranked[:15]
        red_hint = ("## 统计/ML 红球概率分布 Top15\n"
                    + json.dumps([(n, round(p, 4)) for n, p in top], ensure_ascii=False) + "\n")
    # V28：把 L2 约束**明确写进提示词**。
    #
    # 为什么必须写：2026-10-23 生产核查发现 LLM 产出的 seq=4 `[1,6,14,22,30,32]`
    # 奇数个数=1，而 odd_range=[2,3,4] —— 不合规。此前提示词只说"结构均衡"，
    # 没有把可校验的具体阈值给它，模型只能猜。
    # 现在把阈值**原样**列出来（与 shape_rules.check_shape 同一份 ctx），
    # 让模型有明确目标；即使它仍不合规，解析层也会逐注标记（不是丢弃）。
    shape_hint = ""
    if shape_ctx:
        _zones = shape_ctx.get("top_zones") or []
        _zones_txt = "、".join("-".join(str(x) for x in z) for z in _zones[:6])
        _odd = shape_ctx.get("odd_range") or []
        shape_hint = (
            "## 必须满足的结构硬约束（任何一注不含规都会被标记）\n"
            f"- 红球和值 ∈ [{shape_ctx.get('sum_min', 0):.0f}, {shape_ctx.get('sum_max', 0):.0f}]\n"
            f"- 跨度（最大−最小） ∈ [{shape_ctx.get('span_min', 0):.0f}, {shape_ctx.get('span_max', 0):.0f}]\n"
            f"- 三区比（1-11/12-22/23-33）必须是以下之一：{_zones_txt}\n"
            f"- 奇数个数 ∈ {list(_odd)}\n"
            f"- 连号组数 ≤ {shape_ctx.get('max_consecutive', 2)}\n"
        )

    return (
        "你是双色球组合结构优化器。双色球为独立随机事件，**不要试图‘猜中’号码**，"
        "你的职责是在给定概率分布与硬约束下，产出**结构上均衡**、**红球覆盖尽量分散**的候选组合。\n"
        f"## 统计报告\n```json\n{json.dumps(stats_json, ensure_ascii=False)}\n```\n"
        f"## 最近走势\n{json.dumps(recent, ensure_ascii=False)}\n"
        f"## 已回测规律明细（n/边际/p_adj/威尔逊区间）\n{json.dumps(patterns, ensure_ascii=False)}\n"
        + (f"## 规律的**阶段时效**清单（tier=usable 才可用于出号；monitor 仅供观察，"
           f"rejected 已判不可用——不得引用作选号理由）\n"
           f"{json.dumps(regime_rules, ensure_ascii=False)}\n" if regime_rules else "")
        + f"## 你的观察\n{json.dumps(observations, ensure_ascii=False)}\n"
        + red_hint
        + shape_hint
        + fb
        + f"请生成 {config.TICKETS_PER_LLM_CALL} 注候选，输出形如：\n"
        '{"tickets": [{"reds": [6个1-33不重复升序整数], "blue": 1个1-16整数(仅供结构示意，实际由专用蓝球模型决定), '
        '"confidence": 0-100整数(你的结构置信度，不代表中奖概率), '
        '"reasoning": "一句话理由(必须引用具体统计数字)", '
        '"patterns_used": ["引用的规律key列表(可以为空)"], '
        '"evidence": {"统计依据": "具体数字，如：遗漏区间6-10的号码近50期出现率34%", "规律引用": "pattern-key"}, '
        '"counter_evidence": ["为什么不选其它号的1-2条具体理由"], '
        '"structure_scores": {"和值": 1-10, "奇偶": 1-10, "三区": 1-10, "跨度": 1-10}}]}\n'
        "硬约束：红球从概率分布 Top 区间内选择且尽量分散（不同注之间少重复号）；"
        "和值落在历史常见区间，三区比/奇偶比不要极端。若某尺度没有可靠信号，请如实降低置信度并说明。"
        "evidence 必须引用上文具体数字，禁止编造。"
        "patterns_used 只能填 tier=usable 的规律（按「维度key 参数」写）；没有可用规律时请留空数组并如实说明。"
    )


def usage_reset() -> None:
    """清空调用计量（评估每期前调用）。"""
    with _usage_lock:
        _usage.update({"calls": 0, "prompt_chars": 0, "completion_chars": 0})


def usage_snapshot() -> Dict:
    """返回调用计量快照：{calls, prompt_chars, completion_chars}。"""
    with _usage_lock:
        return dict(_usage)

def _redact_echo(text: str, limit: int = 200) -> str:
    """把上游响应体压成**不含内网信息**的短摘要（P1-2，V36，2026-10-24 审计）。

    为什么不能原样回吐：``base_url`` 由用户填写，服务器会替他去 POST。
    若把响应体前 200 字符原样带回给用户，这个 SSRF 就从「盲打」升级成
    「可读响应」—— 能读云元数据、内网服务 banner、甚至内网 API 数据。

    做法：
      * 只保留**结构化错误信息**（JSON 里的 message/error 字段），那是用户
        排查自己模型配置真正需要的东西；
      * 非 JSON / HTML / 二进制一律折叠成「响应体已省略」；
      * 纯文本（如 Redis banner、nginx 错误页）同样省略 —— 它们的价值对
        "排查 LLM 配置" 几乎为零，而泄漏面很大。
    """
    raw = (text or "").strip()
    if not raw:
        return "(空响应)"
    head = raw[:800]
    if head.startswith("<"):
        return "(响应体已省略：非 JSON 结构)"
    try:
        data = json.loads(head if len(raw) <= 800 else raw)
    except Exception:                            # noqa: BLE001
        return "(响应体已省略：非 JSON)"
    msg = ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or err.get("type") or "")
        elif err:
            msg = str(err)
        if not msg:
            msg = str(data.get("message") or data.get("msg") or "")
    if not msg:
        return "(响应体已省略)"
    return msg[:limit]
