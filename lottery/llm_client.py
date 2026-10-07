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
         deadline: Optional[float] = None, strict: bool = False) -> Optional[str]:
    """调用 OpenAI 兼容 chat/completions，返回 content。

    model_cfg: {"name","base_url","api_key","model"}；缺省用主通道配置。
    仓库不内置任何 URL / Key：未配置通道时返回 None（上层降级为统计模型）。
    deadline: 单次预测 LLM 阶段的墙钟截止时间戳（time.time() 口径）；
              每次尝试的单次超时会被钳制在剩余预算内，超预算立即放弃。
    strict:  True 时任何失败（未配置/HTTP 错误/超时/空返回）抛 LLMChannelError，
            供 Web 预测快速失败；False 保持原有降级语义（返回 None）。
    """
    if config.LLM_DISABLED:
        if strict:
            raise LLMChannelError("LLM 已被停用（设置页「停用 LLM」或 LOTT_LLM_DISABLED=1）")
        return None
    requested_max_tokens = int(max_tokens)
    if model_cfg is None:
        if not (config.LLM_BASE_URL and config.LLM_API_KEY and config.LLM_MODEL_LIST):
            msg = ("LLM 通道未配置（请设置 LOTT_LLM_BASE_URL / LOTT_LLM_API_KEY / "
                   "LOTT_LLM_MODEL，或在设置页配置并保存）")
            if strict:
                raise LLMChannelError(msg)
            print("[llm] " + msg + "，降级为纯统计模型")
            return None
        base_url = config.LLM_BASE_URL
        url = base_url + "/chat/completions"
        api_key = config.LLM_API_KEY
        model = config.LLM_MODEL_LIST[0]
    else:
        base_url = str(model_cfg["base_url"]).rstrip("/")
        url = base_url + "/chat/completions"
        api_key = str(model_cfg["api_key"])
        model = str(model_cfg["model"])
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
            r = requests.post(url, json=payload, headers=headers, timeout=eff_timeout)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
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
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if content:
                with _usage_lock:
                    _usage["calls"] += 1
                    _usage["prompt_chars"] += sum(
                        len(m.get("content") or "") for m in payload["messages"])
                    _usage["completion_chars"] += len(content)
                remember_max_tokens(base_url, model, payload.get("max_tokens", 0))
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
        raise LLMChannelError(f"LLM 调用失败: {last_err}")
    # 降级：记录但不抛出，让上层走统计兜底
    print(f"[llm] 调用失败（3 次重试后）: {last_err}")
    return None


def chat_json(system: str, user: str, max_tokens: int = 2000,
              temperature: float = 0.8,
              model_cfg: Optional[Dict] = None,
              deadline: Optional[float] = None,
              strict: bool = False,
              timeout: Optional[float] = None) -> Optional[dict]:
    text = chat(system, user, max_tokens=max_tokens, temperature=temperature,
                model_cfg=model_cfg, deadline=deadline, strict=strict, timeout=timeout)
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
                     blue_probs: Optional[list] = None) -> str:
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
    return (
        "你是双色球组合结构优化器。双色球为独立随机事件，**不要试图‘猜中’号码**，"
        "你的职责是在给定概率分布与硬约束下，产出**结构上均衡**、**红球覆盖尽量分散**的候选组合。\n"
        f"## 统计报告\n```json\n{json.dumps(stats_json, ensure_ascii=False)}\n```\n"
        f"## 最近走势\n{json.dumps(recent, ensure_ascii=False)}\n"
        f"## 已回测规律明细（n/边际/p_adj/威尔逊区间）\n{json.dumps(patterns, ensure_ascii=False)}\n"
        f"## 你的观察\n{json.dumps(observations, ensure_ascii=False)}\n"
        + red_hint
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
    )


def usage_reset() -> None:
    """清空调用计量（评估每期前调用）。"""
    with _usage_lock:
        _usage.update({"calls": 0, "prompt_chars": 0, "completion_chars": 0})


def usage_snapshot() -> Dict:
    """返回调用计量快照：{calls, prompt_chars, completion_chars}。"""
    with _usage_lock:
        return dict(_usage)