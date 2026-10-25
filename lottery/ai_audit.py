"""V12-13：AI 的两个「可机器 100% 判定」岗位 + 打分卡。

为什么这个模块存在（问题不是「AI 不聪明」，而是「以前没人给它标准答案」）
------------------------------------------------------------------------
V12 之前的闭环一直把 LLM 放在**没有标准答案**的岗位上（预测下一期开奖、排序我们的
候选、做因果归因），于是「AI 有没有用」这个问题永远测不出来：连金标准都不存在，
打出来的分数只能反映「它像不像我们的后端规则」。

本模块把 LLM 挪到两个**答案由确定性代码唯一确定**的岗位上：

* ``post="result_narrator"`` —— **自然语言核对员**：把一张已经算好的逐注对照表
  翻译成每个 method 一行的摘要。这是**算术复述**，允许误差 **0**。
* ``post="contract_checker"`` —— **契约合规检查器**：判定一批 filters 是否落在
  维度表声明的档位内。答案由 ``discovery.FAMILY_BY_KEY`` 唯一确定，是**纯判定**。

四态判定（本模块能成立的关键：区分「没做」和「做错」）
------------------------------------------------------
================  ==========================================  ================
verdict           触发条件                                    计入正确率分母
================  ==========================================  ================
``degraded``      LLM 未配置 / 超时 / 非 JSON / 通道异常        **否**（这是没做）
``invalid``       JSON 合法但 schema 不合规（缺字段/类型错/多余）**是**（它声称做了）
``fail``          schema 合规但字段与复算值不符                  **是**
``pass``          全部字段精确一致（0 误差）                    **是**
================  ==========================================  ================

正确率 = ``pass / (pass + fail + invalid)``，``degraded`` **单列**。
**禁止**只报 pass 率——那会把「经常不做」伪装成「很少做错」。

明确排除（V10 §3.2.3-5 / R3 §5 末段，写在这里防止下一个人重新加回来）
---------------------------------------------------------------------
1. 让 LLM 排序我们的候选（测的是「能否猜出后端 confidence 规则」，不是能否预测）；
2. 让 LLM 做因果归因（没有金标准，无法打分）；
3. 让 LLM 出号并指望它「被显著性验证」（该约束集合恒为空，V10 已判定为设计死锁）；
4. 让 LLM 在 30 期样本上「证明」自己有用或无用（N=30 的 Wilson CI 已含随机基线）。

铁律：两岗位的输出**都不产生任何号码**、**都不写 predictions**、**都不进候选池/
规律库**；只写 ``llm_audit`` 表（仅追加，``INSERT OR IGNORE``）+ 一个只读接口。
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import db

POST_NARRATOR = "result_narrator"
POST_CONTRACT = "contract_checker"
POSTS = (POST_NARRATOR, POST_CONTRACT)

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_INVALID = "invalid"
VERDICT_DEGRADED = "degraded"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_INVALID, VERDICT_DEGRADED)

#: 计入正确率分母的三态（degraded 是「没做」，单列）
SCORED_VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_INVALID)

#: 契约检查器的一致率闸门（V10 §3.2.2）
CONTRACT_AGREE_MIN = 0.99

#: 「无中奖」的奖级名（与 evaluate.PRIZE_NAME 的取值空间互斥）
NO_PRIZE = "无"


# ---------------------------------------------------------------------------
# 0. 通用小工具
# ---------------------------------------------------------------------------

def _canon(obj: Any) -> str:
    """确定性 JSON 序列化（审计指纹用；键序与分隔符全部固定）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_id_for(post: str, run_id: str, ts_ns: int) -> str:
    """``audit_id = sha1(post|run_id|ts_ns)``——重试同一轮时幂等（V10 §2.1.5 冻结）。"""
    raw = f"{post}|{run_id}|{int(ts_ns)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _is_int(x: Any) -> bool:
    """严格整数（bool 在 Python 里是 int 子类，必须排除）。"""
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _round4(x: float) -> float:
    """与 V10 §3.2.1 冻结口径一致：先 ``round(...,4)`` 再比，误差允许 0。"""
    return round(float(x), 4)


def _prize_name(level: Any) -> str:
    """奖级 → 中文名（直接复用 ``evaluate.PRIZE_NAME``，含一二等奖的 ``(≈*)`` 后缀）。

    为什么必须复用而不是另写一张表：判定器和被判定者若各自持有一份奖级名表，
    「金标准」就出现了第二个来源，比对本身失去意义（V10 §3.2.1）。
    """
    from . import evaluate
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return NO_PRIZE
    if lvl <= 0:
        return NO_PRIZE
    return evaluate.PRIZE_NAME.get(lvl, NO_PRIZE)




# ---------------------------------------------------------------------------
# 1. 岗位 A：确定性结果的自然语言核对员（result_narrator）
# ---------------------------------------------------------------------------

def golden_narrator(input_obj: Dict) -> Dict:
    """岗位 A 的**金标准**：用确定性代码从 ``input_obj["rows"]`` 复算摘要。

    复算口径**统一走** :func:`evaluate.aggregate`（V10 §3.2.1 明文要求），
    不在本模块另写一套均值逻辑——否则「金标准」出现第二个来源。

    返回::

        {"per_method": [{method, n, red_mean, blue_rate, best_level}],   # 按 method 首次出现序
         "window": {...输入原样...},
         "methods": [method, ...]}
    """
    rows = list((input_obj or {}).get("rows") or [])
    window = (input_obj or {}).get("window") or {}

    # 按 method 分组，但保持**首次出现顺序**（稳定输出，便于 diff）
    order: List[str] = []
    grouped: Dict[str, List[Dict]] = {}
    for r in rows:
        m = str(r.get("method"))
        if m not in grouped:
            grouped[m] = []
            order.append(m)
        grouped[m].append(r)

    per_method: List[Dict] = []
    for m in order:
        grp = grouped[m]
        n = len(grp)
        # 走 evaluate.aggregate 的口径：它吃 runs=[{red_hits, blue_hits, levels, reward}]
        runs = [{
            "red_hits": [int(x.get("red_hits") or 0)],
            "blue_hits": [int(x.get("blue_hit") or 0)],
            "levels": [int(x.get("level") or 0)],
            "reward": float(x.get("reward") or 0.0),
        } for x in grp]
        agg = _aggregate_flat(grp, runs)
        best_level = _best_level(grp)
        per_method.append({
            "method": m,
            "n": int(n),
            "red_mean": _round4(agg["red_hits_mean"]),
            "blue_rate": _round4(agg["blue_hit_rate"]),
            "best_level": best_level,
        })

    return {"per_method": per_method, "window": window, "methods": order}


def _aggregate_flat(grp: Sequence[Dict], runs: Sequence[Dict]) -> Dict:
    """``evaluate.aggregate`` 的逐注平铺调用。

    为什么逐注构造 ``runs``：``evaluate.aggregate`` 的口径是「每期一个 run，run 内含
    若干注」（``np.concatenate([r["red_hits"] ...])``）。本岗位的输入是**行**级
    （一行 = 一期 × 一个 method × 一个 seq），所以每行当一个 run、run 内 1 注——
    平铺后 ``np.mean`` 等价于「对全部行取均值」，与 §3.2.1 的复算规则逐字一致。
    """
    from . import evaluate
    return evaluate.aggregate(list(runs))


def _best_level(grp: Sequence[Dict]) -> str:
    """出现次数最多的奖级；**并列取最高奖级**；无中奖返回 ``"无"``。

    「最高奖级」= 奖级数字最小（一等奖 > 二等奖 > …）。
    """
    counts: Dict[int, int] = {}
    for r in grp:
        lvl = r.get("level")
        try:
            lv = int(lvl)
        except (TypeError, ValueError):
            lv = 0
        if lv <= 0:
            continue
        counts[lv] = counts.get(lv, 0) + 1
    if not counts:
        return NO_PRIZE
    top = max(counts.values())
    # 并列时取奖级序最优（数字最小）
    tied = sorted(lv for lv, c in counts.items() if c == top)
    return _prize_name(tied[0])


_NARRATOR_METHOD_KEYS = {"method", "n", "red_mean", "blue_rate", "best_level"}
_NARRATOR_TOP_KEYS = {"per_method", "window", "notes_zh"}


def judge_narrator(input_obj: Dict, output_obj: Any) -> Dict:
    """岗位 A 机器判定（允许误差 **0**）。返回 ``{"verdict", "fail_fields", ...}``。

    ``output_obj`` 为 ``None`` 表示 LLM 不可用/非 JSON → ``degraded``（不进分母）。
    schema 不合规则 ``invalid``；字段不符则 ``fail``，``fail_fields`` 精确到
    ``per_method[1].red_mean`` 这种粒度（V10 §3.2.1 冻结规则）。
    """
    if output_obj is None:
        return _verdict(VERDICT_DEGRADED, [], "LLM 未配置或未返回可解析的 JSON")

    gold = golden_narrator(input_obj)
    fail_fields: List[str] = []

    # ---- schema 校验（任何一条不合规 → invalid）----
    if not isinstance(output_obj, dict):
        return _verdict(VERDICT_INVALID, ["output"], "顶层不是 JSON 对象")
    extra = set(output_obj) - _NARRATOR_TOP_KEYS
    missing = _NARRATOR_TOP_KEYS - set(output_obj)
    if extra:
        return _verdict(VERDICT_INVALID, sorted(f"output.{k}" for k in extra),
                        f"多余字段：{sorted(extra)}")
    if missing:
        return _verdict(VERDICT_INVALID, sorted(f"output.{k}" for k in missing),
                        f"缺字段：{sorted(missing)}")

    got_pm = output_obj.get("per_method")
    if not isinstance(got_pm, list):
        return _verdict(VERDICT_INVALID, ["per_method"], "per_method 不是数组")

    got_methods = []
    for i, item in enumerate(got_pm):
        if not isinstance(item, dict):
            return _verdict(VERDICT_INVALID, [f"per_method[{i}]"], "条目不是对象")
        iextra = set(item) - _NARRATOR_METHOD_KEYS
        imiss = _NARRATOR_METHOD_KEYS - set(item)
        if iextra:
            return _verdict(VERDICT_INVALID,
                            sorted(f"per_method[{i}].{k}" for k in iextra),
                            f"per_method[{i}] 多余字段：{sorted(iextra)}")
        if imiss:
            return _verdict(VERDICT_INVALID,
                            sorted(f"per_method[{i}].{k}" for k in imiss),
                            f"per_method[{i}] 缺字段：{sorted(imiss)}")
        # 类型严格：n 必须是 int（不接受 "12"），数值必须 number，字符串必须 str
        if not _is_int(item["n"]):
            return _verdict(VERDICT_INVALID, [f"per_method[{i}].n"],
                            f"n 类型错误：{type(item['n']).__name__}")
        for k in ("red_mean", "blue_rate"):
            if not _is_number(item[k]):
                return _verdict(VERDICT_INVALID, [f"per_method[{i}].{k}"],
                                f"{k} 类型错误：{type(item[k]).__name__}")
        for k in ("method", "best_level"):
            if not isinstance(item[k], str):
                return _verdict(VERDICT_INVALID, [f"per_method[{i}].{k}"],
                                f"{k} 类型错误：{type(item[k]).__name__}")
        got_methods.append(item["method"])
    if not isinstance(output_obj.get("notes_zh"), str):
        return _verdict(VERDICT_INVALID, ["notes_zh"], "notes_zh 不是字符串")

    # ---- method 集合必须完全相等（多一个/少一个 → invalid）----
    if sorted(got_methods) != sorted(gold["methods"]):
        return _verdict(VERDICT_INVALID, ["per_method"],
                        f"method 集合不一致：期望 {sorted(gold['methods'])}，"
                        f"实际 {sorted(got_methods)}")

    # ---- window 三个整数严格相等 ----
    gold_win = gold.get("window") or {}
    got_win = output_obj.get("window")
    if not isinstance(got_win, dict):
        return _verdict(VERDICT_INVALID, ["window"], "window 不是对象")
    for k in ("from_issue", "to_issue", "n_rows"):
        if k not in got_win:
            return _verdict(VERDICT_INVALID, [f"window.{k}"], f"window 缺 {k}")
        if not _is_int(got_win.get(k)):
            return _verdict(VERDICT_INVALID, [f"window.{k}"],
                            f"window.{k} 类型错误（必须是整数）")
        if gold_win.get(k) is not None and int(got_win[k]) != int(gold_win[k]):
            fail_fields.append(f"window.{k}")
    for k in set(got_win) - {"from_issue", "to_issue", "n_rows"}:
        return _verdict(VERDICT_INVALID, [f"window.{k}"], f"window 多余字段 {k}")

    # ---- 逐字段精确比对（误差 0）----
    gold_by = {g["method"]: g for g in gold["per_method"]}
    seen: Dict[str, int] = {}
    for i, item in enumerate(got_pm):
        m = item["method"]
        if m not in gold_by:
            continue                        # 集合相等已在上方保证，这里只防重复
        idx = seen.get(m, 0)
        seen[m] = idx + 1
        if idx > 0:                          # 同一 method 出现两次
            fail_fields.append(f"per_method[{i}].method")
            continue
        g = gold_by[m]
        if int(item["n"]) != int(g["n"]):
            fail_fields.append(f"per_method[{i}].n")
        # 先按同一 round(...,4) 规整再比，仍要求差值 == 0
        if _round4(item["red_mean"]) != _round4(g["red_mean"]):
            fail_fields.append(f"per_method[{i}].red_mean")
        if _round4(item["blue_rate"]) != _round4(g["blue_rate"]):
            fail_fields.append(f"per_method[{i}].blue_rate")
        if str(item["best_level"]) != str(g["best_level"]):
            fail_fields.append(f"per_method[{i}].best_level")

    if fail_fields:
        return _verdict(VERDICT_FAIL, fail_fields,
                        f"核对错误：{', '.join(fail_fields)}", gold=gold)
    return _verdict(VERDICT_PASS, [], "核对一致（0 误差）", gold=gold)


# ---------------------------------------------------------------------------
# 2. 岗位 B：契约合规检查器（contract_checker）
# ---------------------------------------------------------------------------

def golden_ok(key: str, params: Dict) -> Tuple[bool, List[str], str]:
    """岗位 B 的**金标准**：三项既有校验的合取（与生产同源，禁止另写档位表）。

    a) ``llm_lab._params_ok(key, params)``
    b) ``evolution.discovery_params_ok({"key": key, "params": params})``
       ← 注意签名：接收**单个 dict**
    c) ``key in discovery.FAMILY_BY_KEY``

    a/b 结论不一致 → **保守取 False** 且 ``reason="gold_disagreement"``（计入分歧计数器）。

    返回 ``(ok, offending_keys, reason_zh)``。``ok=False`` 时 ``offending`` **不得为空**
    ——否则无法审计它错在哪（V10 §3.2.2 可审计性闸门）。
    """
    from . import discovery, evolution, llm_lab

    key = str(key)
    params = dict(params or {})

    unknown_key = key not in discovery.FAMILY_BY_KEY
    try:
        ok_a = bool(llm_lab._params_ok(key, params))
    except Exception:                                   # noqa: BLE001 - 校验失败即保守 False
        ok_a = False
    try:
        ok_b = bool(evolution.discovery_params_ok({"key": key, "params": params}))
    except Exception:                                   # noqa: BLE001
        ok_b = False

    if ok_a != ok_b:
        offending = sorted(k for k in params) or ["key"]
        return False, offending, "gold_disagreement"

    if unknown_key:
        return False, ["key"], "unknown_key"

    if ok_a and ok_b:
        return True, [], ""

    # 定位不合规的参数（与档位表逐项比对，给出可读理由）
    fam = discovery.FAMILY_BY_KEY.get(key) or {}
    tiers = list(fam.get("params") or [])
    offending = sorted(k for k in params if all(dict(t).get(k) != params.get(k) for t in tiers))
    if not offending:
        # 参数整体不在组合表里（每个字段单独看都合法，但组合不存在）
        offending = sorted(params) or ["params"]
        return False, offending, "组合不在档位表中"
    return False, offending, f"参数越界：{', '.join(offending)}"


def _contract_golden_items(items: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, it in enumerate(items):
        key = str((it or {}).get("key"))
        params = (it or {}).get("params") or {}
        ok, offending, reason = golden_ok(key, params)
        out.append({"idx": i, "key": key, "params": params,
                    "ok": ok, "offending": offending, "reason": reason})
    return out


def judge_contract(input_obj: Dict, output_obj: Any) -> Dict:
    """岗位 B 机器判定：逐条布尔一致率 ``agree >= 0.99`` → pass，否则 fail。

    ``output_obj is None`` → ``degraded``。``fail_fields`` 列出不一致的 idx 字符串。
    """
    if output_obj is None:
        return _verdict(VERDICT_DEGRADED, [], "LLM 未配置或未返回可解析的 JSON")

    items = (input_obj or {}).get("items")
    if not isinstance(items, list) or not items:
        return _verdict(VERDICT_INVALID, ["items"], "输入 items 为空或不是数组")
    golden = _contract_golden_items(items)

    if not isinstance(output_obj, dict):
        return _verdict(VERDICT_INVALID, ["output"], "顶层不是 JSON 对象")
    got_items = output_obj.get("items")
    if not isinstance(got_items, list):
        return _verdict(VERDICT_INVALID, ["items"], "输出 items 不是数组")
    if len(got_items) != len(items):
        return _verdict(VERDICT_INVALID, ["items"],
                        f"items 数量不一致：期望 {len(items)}，实际 {len(got_items)}")

    # ---- schema：每条必须有 idx(整数) / ok(布尔) / offending(数组) ----
    for i, gi in enumerate(got_items):
        if not isinstance(gi, dict):
            return _verdict(VERDICT_INVALID, [f"items[{i}]"], "条目不是对象")
        if not _is_int(gi.get("idx")):
            return _verdict(VERDICT_INVALID, [f"items[{i}].idx"], "idx 必须是整数")
        if not isinstance(gi.get("ok"), bool):
            return _verdict(VERDICT_INVALID, [f"items[{i}].ok"], "ok 必须是布尔值")
        if not isinstance(gi.get("offending"), list):
            return _verdict(VERDICT_INVALID, [f"items[{i}].offending"],
                            "offending 必须是数组")

    by_idx: Dict[int, Dict] = {}
    for gi in got_items:
        idx = int(gi["idx"])
        if idx in by_idx:
            return _verdict(VERDICT_INVALID, [f"items[{idx}]"], f"idx 重复：{idx}")
        by_idx[idx] = gi
    if sorted(by_idx) != list(range(len(items))):
        return _verdict(VERDICT_INVALID, ["items"],
                        f"idx 集合不完整：期望 0..{len(items) - 1}，实际 {sorted(by_idx)}")

    # ---- 可审计性：金标准 ok=False 的条目，LLM 的 offending 不得为空 ----
    unauditable: List[str] = []
    for g in golden:
        gi = by_idx[g["idx"]]
        if g["ok"] is False and not gi["offending"]:
            unauditable.append(f"items[{g['idx']}].offending")
    if unauditable:
        return _verdict(VERDICT_INVALID, unauditable,
                        "ok=false 但 offending 为空，无法审计错在哪")

    # ---- 逐条布尔比对 ----
    mismatches: List[str] = []
    for g in golden:
        gi = by_idx[g["idx"]]
        if bool(gi["ok"]) != bool(g["ok"]):
            mismatches.append(str(g["idx"]))
    agree = (len(items) - len(mismatches)) / float(len(items))
    summary = {"n": len(items), "n_ok": sum(1 for g in golden if g["ok"]),
               "n_bad": sum(1 for g in golden if not g["ok"]), "agree": round(agree, 6)}

    if agree < CONTRACT_AGREE_MIN:
        return _verdict(VERDICT_FAIL, mismatches,
                        f"一致率 {agree:.2%} < {CONTRACT_AGREE_MIN:.0%}",
                        gold={"items": golden, "summary": summary})

    # ---- summary 若给出则必须与复算一致（可审计，不影响 agree 闸门）----
    got_summary = output_obj.get("summary")
    if isinstance(got_summary, dict):
        for k in ("n", "n_ok", "n_bad"):
            if k in got_summary and _is_int(got_summary.get(k)):
                if int(got_summary[k]) != int(summary[k]):
                    mismatches.append(f"summary.{k}")
        if mismatches:
            return _verdict(VERDICT_FAIL, mismatches,
                            f"摘要数字与复算不符：{', '.join(mismatches)}",
                            gold={"items": golden, "summary": summary})
    return _verdict(VERDICT_PASS, [], f"一致率 {agree:.2%}",
                    gold={"items": golden, "summary": summary})


# ---------------------------------------------------------------------------
# 3. 打分卡（degraded 单列，不进分母）
# ---------------------------------------------------------------------------

def _verdict(kind: str, fail_fields: List[str], note_zh: str,
             gold: Optional[Dict] = None) -> Dict:
    assert kind in VERDICTS, kind
    out = {
        "verdict": kind,
        "fail_fields": list(fail_fields),
        "note_zh": note_zh,
        # 只有三态进分母；degraded 是「没做」
        "counts_toward_score": kind in SCORED_VERDICTS,
    }
    if gold is not None:
        out["golden"] = gold
    return out


def scorecard(records: Sequence[Dict]) -> Dict:
    """两岗位的**共用打分卡**：``pass / (pass + fail + invalid)``，``degraded`` 单列。

    为什么不能只报 pass 率：那会把「经常不做」伪装成「很少做错」。同时给出
    ``n_correct`` / ``n_wrong`` / ``n_invalid`` / ``n_skipped`` 四个分开的数（V12 §1.3 硬性准入）。

    ``n_correct`` 只数 ``pass``；``skipped``（degraded）**永不**计入。
    """
    n_correct = n_wrong = n_invalid = n_skipped = 0
    by_post: Dict[str, Dict[str, int]] = {}
    for r in records or []:
        post = str(r.get("post") or "?")
        v = str(r.get("verdict") or VERDICT_INVALID)
        slot = by_post.setdefault(post, {k: 0 for k in VERDICTS})
        if v not in VERDICTS:
            v = VERDICT_INVALID
        slot[v] += 1
        if v == VERDICT_PASS:
            n_correct += 1
        elif v == VERDICT_FAIL:
            n_wrong += 1
        elif v == VERDICT_INVALID:
            n_invalid += 1
        elif v == VERDICT_DEGRADED:
            n_skipped += 1
    denom = n_correct + n_wrong + n_invalid
    score = (n_correct / denom) if denom else None
    lo, hi = _wilson(n_correct, denom) if denom else (None, None)
    return {
        "n_total": len(records or []),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_invalid": n_invalid,
        "n_skipped": n_skipped,
        "denominator": denom,
        "score": score,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "by_post": by_post,
        "note_zh": _score_note(n_correct, n_wrong, n_invalid, n_skipped, score),
    }


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    import math
    if n <= 0:
        return None, None
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def _score_note(n_correct: int, n_wrong: int, n_invalid: int, n_skipped: int,
                score: Optional[float]) -> str:
    if n_correct + n_wrong + n_invalid == 0:
        if n_skipped:
            return f"本批 {n_skipped} 次尝试全部未取得 AI 结果（degraded），不计入正确率"
        return "尚无 AI 核对记录"
    head = (f"对 {n_correct} / 错 {n_wrong} / 不合规 {n_invalid} / 未取得 {n_skipped}"
            f"（正确率 {score:.1%}，分母 {n_correct + n_wrong + n_invalid}）")
    if n_skipped:
        head += "；未取得的次数单列，不并入分母"
    return head




# ---------------------------------------------------------------------------
# 4. 落库（仅追加）+ 只读查询
# ---------------------------------------------------------------------------

def save_audit(post: str, run_id: str, input_obj: Dict, output_obj: Any,
               judgement: Dict, model: str = "", ts_ns: Optional[int] = None) -> Dict:
    """写 ``llm_audit``（``INSERT OR IGNORE``：审计历史不可被覆盖，重试幂等）。

    返回 ``{"audit_id", "inserted": bool}``。
    """
    if post not in POSTS:
        raise ValueError(f"未知岗位：{post!r}（必须是 {POSTS}）")
    ts_ns = int(time.time() * 1e9) if ts_ns is None else int(ts_ns)
    aid = audit_id_for(post, run_id, ts_ns)
    conn = db.get_conn()
    cur = conn.execute(
        "INSERT OR IGNORE INTO llm_audit "
        "(audit_id, run_id, post, model, input_json, output_json, verdict, fail_fields, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (aid, str(run_id), str(post), str(model or ""),
         _canon(input_obj), _canon(output_obj) if output_obj is not None else "",
         str(judgement.get("verdict") or VERDICT_INVALID),
         _canon(list(judgement.get("fail_fields") or [])),
         time.time()),
    )
    conn.commit()
    return {"audit_id": aid, "inserted": bool(cur.rowcount)}


def load_audits(post: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """读最近 N 条判定（只读）。``fail_fields`` / JSON 字段会被反序列化。"""
    conn = db.get_conn()
    limit = max(1, min(int(limit or 50), 500))
    if post:
        rows = conn.execute(
            "SELECT * FROM llm_audit WHERE post=? ORDER BY ts DESC LIMIT ?",
            (str(post), limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM llm_audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    out: List[Dict] = []
    for r in rows:
        d = dict(r)
        for k, default in (("fail_fields", []),):
            try:
                d[k] = json.loads(d.get(k) or "[]")
            except (ValueError, TypeError):
                d[k] = default
        for k in ("input_json", "output_json"):
            raw = d.pop(k, "")
            try:
                d[k] = json.loads(raw) if raw else None
            except (ValueError, TypeError):
                d[k] = None
        out.append(d)
    return out


def latest_audit(post: Optional[str] = None) -> Dict:
    """``/api/llm/audit/latest`` 的数据源：最近一次判定 + 分数卡。

    无记录时 ``available=False``（**不许**渲染成「AI 已核对」）。
    """
    records = load_audits(post=post, limit=200)
    if not records:
        return {"available": False,
                "note_zh": "本次未取得 AI 核对结果（尚无审计记录）",
                "scorecard": scorecard([])}
    newest = records[0]
    return {
        "available": True,
        "post": newest.get("post"),
        "verdict": newest.get("verdict"),
        "fail_fields": newest.get("fail_fields") or [],
        "ts": newest.get("ts"),
        "model": newest.get("model"),
        "correct": None if newest.get("verdict") == VERDICT_DEGRADED
        else bool(newest.get("verdict") == VERDICT_PASS),
        "note_zh": _verdict_text(newest.get("verdict"), newest.get("fail_fields") or []),
        "scorecard": scorecard(records),
    }


def _verdict_text(verdict: Any, fail_fields: Sequence[str]) -> str:
    v = str(verdict or "")
    if v == VERDICT_PASS:
        return "AI 核对一致（0 误差）"
    if v == VERDICT_FAIL:
        return f"AI 核对错误：{', '.join(fail_fields) if fail_fields else '（未指明字段）'}"
    if v == VERDICT_INVALID:
        return "AI 输出不合规"
    if v == VERDICT_DEGRADED:
        return "本次未取得 AI 核对结果"
    return "未知判定"


# ---------------------------------------------------------------------------
# 5. 输入构造（从 eval_details / 候选快照，只读）
# ---------------------------------------------------------------------------

def build_narrator_input(rows: Iterable[Dict], limit_issues: Optional[int] = None) -> Dict:
    """把 ``eval_details`` 行整理成岗位 A 的冻结输入 schema。

    行来源：``data/ssq.db`` 的 ``eval_details``（主键 ``(issue, method, seq)``），**只读**，
    不重新跑预测。奖级由 ``level``（整数）映射成中文名——调用方给 ``best_level`` 时以
    ``level`` 为准，避免两套口径。
    """
    rows = list(rows or [])
    if not rows:
        return {"window": {"from_issue": 0, "to_issue": 0, "n_rows": 0}, "rows": []}
    issues = sorted({int(r.get("issue") or 0) for r in rows})
    if limit_issues:
        issues = issues[-int(limit_issues):]
        keep = set(issues)
        rows = [r for r in rows if int(r.get("issue") or 0) in keep]
    out_rows: List[Dict] = []
    for r in rows:
        out_rows.append({
            "method": str(r.get("method") or "?"),
            "issue": int(r.get("issue") or 0),
            "seq": int(r.get("seq") or 0),
            "red_hits": int(r.get("red_hits") or 0),
            "blue_hit": int(r.get("blue_hit") or 0),
            "level": int(r.get("level") or 0),
            "reward": float(r.get("reward") or 0.0),
        })
    used = sorted({x["issue"] for x in out_rows})
    return {"window": {"from_issue": used[0], "to_issue": used[-1],
                       "n_rows": len(out_rows)},
            "rows": out_rows}


def build_contract_input(items: Sequence[Dict]) -> Dict:
    """把候选 filters 整理成岗位 B 的冻结输入 schema（只读快照）。"""
    out: List[Dict] = []
    for i, it in enumerate(items or []):
        it = it or {}
        out.append({"idx": i, "key": str(it.get("key") or ""),
                    "params": dict(it.get("params") or {}),
                    "target": str(it.get("target") or "red")})
    return {"items": out}


NARRATOR_SYSTEM = (
    "你是一名数据核对员。用户会给你一张已经算好的逐注对照表（JSON），"
    "你的唯一任务是把它汇总成每个 method 一行的结构化摘要。\n"
    "铁律：\n"
    "1) 你必须原样复述数字，禁止任何预测、建议、因果解释或与开奖相关的推测；\n"
    "2) red_mean = 该 method 全部行 red_hits 的算术平均，保留 4 位小数（round(x,4)）；\n"
    "3) blue_rate = 该 method 全部行 blue_hit 的算术平均，保留 4 位小数；\n"
    "4) n = 该 method 的行数（整数）；\n"
    "5) best_level = 该 method 出现次数最多的奖级名（并列取最高奖级；无中奖填 \"无\"），"
    "奖级名必须原样取自输入的 level_name 字段；\n"
    "6) window 原样回填输入的 window；\n"
    "7) 只输出 JSON，不要输出解释。"
)


def narrator_prompt(input_obj: Dict) -> str:
    """把输入渲染成给 LLM 的 user prompt（附每行奖级名，避免 LLM 自己猜奖级表）。"""
    rows = []
    for r in (input_obj or {}).get("rows") or []:
        rows.append({
            "method": r.get("method"), "issue": r.get("issue"), "seq": r.get("seq"),
            "red_hits": r.get("red_hits"), "blue_hit": r.get("blue_hit"),
            "level_name": _prize_name(r.get("level")), "reward": r.get("reward"),
        })
    payload = {"window": (input_obj or {}).get("window") or {}, "rows": rows}
    return (
        "请按以下 JSON schema 输出摘要：\n"
        '{"per_method":[{"method":str,"n":int,"red_mean":number,'
        '"blue_rate":number,"best_level":str}],'
        '"window":{"from_issue":int,"to_issue":int,"n_rows":int},"notes_zh":str}\n\n'
        "输入对照表：\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


CONTRACT_SYSTEM = (
    "你是一名契约合规检查器。用户给你一批 filters（key + params），"
    "你判断每条的参数是否落在系统维度表声明的档位内。\n"
    "铁律：\n"
    "1) 你不知道档位表，只被要求按你自己对参数合法性的判断给出 ok；"
    "系统会用确定性代码复算并逐条比对；\n"
    "2) 凡 ok=false 的条目，offending 必须列出至少一个出问题的字段名，不得为空；\n"
    "3) 不产生任何号码、不给出候选；\n"
    "4) 只输出 JSON，不要输出解释。"
)


def contract_prompt(input_obj: Dict) -> str:
    items = (input_obj or {}).get("items") or []
    return (
        "请按以下 JSON schema 输出：\n"
        '{"items":[{"idx":int,"ok":bool,"offending":[str],"reason_zh":str}],'
        '"summary":{"n":int,"n_ok":int,"n_bad":int}}\n\n'
        "输入：\n" + json.dumps({"items": items}, ensure_ascii=False, sort_keys=True)
    )


# ---------------------------------------------------------------------------
# 6. 一次完整运行（1 次 LLM 调用；不写 predictions、不产生号码）
# ---------------------------------------------------------------------------

def run_post(post: str, input_obj: Dict, *, use_llm: bool = True,
             model_cfg: Optional[Dict] = None, max_tokens: int = 1200,
             timeout: Optional[float] = None, run_id: Optional[str] = None,
             persist: bool = True) -> Dict:
    """跑一个岗位：**只发 1 次** LLM 调用（``max_tokens ≤ 1200``），然后机器判定并落库。

    为什么只 1 次：``llm_lab`` 的三轮流程单次可跑 3×180s，会把成本报表的分母算错
    （V10 §3.2.3-1）。本岗位是**复述/判定**，一次调用足够。

    返回判定 + 分数卡；LLM 不可用时 ``verdict="degraded"`` 且 ``degraded=True``。
    """
    if post not in POSTS:
        raise ValueError(f"未知岗位：{post!r}（必须是 {POSTS}）")
    started = time.time()
    run_id = run_id or f"{post}-{int(started * 1000)}"
    output: Any = None
    model_name = ""
    error: Optional[str] = None

    if use_llm:
        try:
            from . import llm_client
            if post == POST_NARRATOR:
                output = llm_client.chat_json(NARRATOR_SYSTEM, narrator_prompt(input_obj),
                                              max_tokens=int(max_tokens),
                                              temperature=0.0, model_cfg=model_cfg,
                                              timeout=timeout, purpose="decision")
            else:
                output = llm_client.chat_json(CONTRACT_SYSTEM, contract_prompt(input_obj),
                                              max_tokens=int(max_tokens),
                                              temperature=0.0, model_cfg=model_cfg,
                                              timeout=timeout, purpose="decision")
            if isinstance(model_cfg, dict):
                model_name = str(model_cfg.get("model") or "")
        except Exception as e:                          # noqa: BLE001 - 任何异常都是「没做」
            output = None
            error = f"{type(e).__name__}: {e}"
    else:
        error = "use_llm=False"

    judge = (judge_narrator(input_obj, output) if post == POST_NARRATOR
             else judge_contract(input_obj, output))
    out = dict(judge)
    out["post"] = post
    out["run_id"] = run_id
    out["degraded"] = judge["verdict"] == VERDICT_DEGRADED
    out["elapsed_ms"] = int((time.time() - started) * 1000)
    if error:
        out["error"] = error

    if persist:
        saved = save_audit(post, run_id, input_obj, output, judge, model=model_name)
        out["audit_id"] = saved["audit_id"]
        out["inserted"] = saved["inserted"]
        out["scorecard"] = scorecard(load_audits(post=post, limit=200))
    return out
