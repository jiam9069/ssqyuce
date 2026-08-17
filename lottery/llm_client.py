"""LLM 推理通道：OpenAI 兼容端点，结构化 JSON 输出，带重试与降级。"""
from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional

import requests

# 调用计量（M3.1 LLM 离线评估用；线程安全累加，仅统计成功返回的调用）
_usage = {"calls": 0, "prompt_chars": 0, "completion_chars": 0}
_usage_lock = threading.Lock()

from . import config


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
         model_cfg: Optional[Dict] = None) -> Optional[str]:
    """调用 OpenAI 兼容 chat/completions，返回 content（失败返回 None）。

    model_cfg: {"name","base_url","api_key","model"}；缺省用主通道配置。
    仓库不内置任何 URL / Key：未配置通道时直接返回 None（上层降级为统计模型）。
    """
    if config.LLM_DISABLED:
        return None
    if model_cfg is None:
        if not (config.LLM_BASE_URL and config.LLM_API_KEY and config.LLM_MODEL_LIST):
            print("[llm] 未配置 LLM 通道（请设置 LOTT_LLM_BASE_URL / LOTT_LLM_API_KEY / "
                  "LOTT_LLM_MODEL 或 .env），降级为纯统计模型")
            return None
        url = config.LLM_BASE_URL + "/chat/completions"
        api_key = config.LLM_API_KEY
        model = config.LLM_MODEL_LIST[0]
    else:
        url = str(model_cfg["base_url"]).rstrip("/") + "/chat/completions"
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    boosted = False  # 推理型模型（如 deepseek-v4-flash）空 content 时放大预算重试一次
    t_start = time.time()  # M3.4：单次 chat 总耗时硬上限，防止上游挂起拖死整条链
    for attempt in range(3):
        if time.time() - t_start > 600:
            last_err = f"chat 总耗时超过 600s 上限（当前第 {attempt} 次尝试），放弃"
            break
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=timeout or config.LLM_TIMEOUT)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
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
                return content
            # content 为空：可能 reasoning_content 吃光了 max_tokens
            finish = data["choices"][0].get("finish_reason")
            reasoning = msg.get("reasoning_content") or ""
            if not boosted and finish == "length" and reasoning:
                boosted = True
                payload["max_tokens"] = max(payload.get("max_tokens", 0) * 3, 6000)
                timeout = 120
                print(f"[llm] {model} 推理占满预算，放大 max_tokens 重试一次")
                attempt -= 1
                continue
            return None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if isinstance(e, requests.exceptions.ReadTimeout) and not boosted:
                boosted = True
                payload["max_tokens"] = max(payload.get("max_tokens", 0) * 3, 6000)
                timeout = 240
                print(f"[llm] 读取超时，放大 max_tokens 重试一次")
                attempt -= 1
                continue
    # 降级：记录但不抛出，让上层走统计兜底
    print(f"[llm] 调用失败（3 次重试后）: {last_err}")
    return None


def chat_json(system: str, user: str, max_tokens: int = 2000,
              temperature: float = 0.8,
              model_cfg: Optional[Dict] = None) -> Optional[dict]:
    text = chat(system, user, max_tokens=max_tokens, temperature=temperature,
                model_cfg=model_cfg)
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
                     feedback: Optional[dict] = None) -> str:
    """第 2 轮：基于观察生成多注候选号码（含依据/反证/结构分 schema）。"""
    fb = f"## 上期预测回馈（命中情况）\n{json.dumps(feedback, ensure_ascii=False)}\n" if feedback else ""
    return (
        "基于以下统计报告与你的观察，生成多注候选号码。\n"
        f"## 统计报告\n```json\n{json.dumps(stats_json, ensure_ascii=False)}\n```\n"
        f"## 最近走势\n{json.dumps(recent, ensure_ascii=False)}\n"
        f"## 已回测规律明细（n/边际/p_adj/威尔逊区间）\n{json.dumps(patterns, ensure_ascii=False)}\n"
        f"## 你的观察\n{json.dumps(observations, ensure_ascii=False)}\n"
        + fb
        + f"请生成 {config.TICKETS_PER_LLM_CALL} 注候选，输出形如：\n"
        '{"tickets": [{"reds": [6个1-33不重复升序整数], "blue": 1个1-16整数, '
        '"confidence": 0-100整数(你的结构置信度，不代表中奖概率), '
        '"reasoning": "一句话理由(必须引用具体统计数字)", '
        '"patterns_used": ["引用的规律key列表(可以为空)"], '
        '"evidence": {"统计依据": "具体数字，如：遗漏区间6-10的号码近50期出现率34%", "规律引用": "pattern-key"}, '
        '"counter_evidence": ["为什么不选其它号的1-2条具体理由"], '
        '"structure_scores": {"和值": 1-10, "奇偶": 1-10, "三区": 1-10, "跨度": 1-10}}]}\n'
        "约束：和值尽量落在历史常见区间，三区比/奇偶比不要极端，蓝球尽量分散。"
        "若某尺度没有可靠信号，请如实降低置信度并说明。evidence 必须引用上文具体数字，禁止编造。"
    )


def usage_reset() -> None:
    """清空调用计量（评估每期前调用）。"""
    with _usage_lock:
        _usage.update({"calls": 0, "prompt_chars": 0, "completion_chars": 0})


def usage_snapshot() -> Dict:
    """返回调用计量快照：{calls, prompt_chars, completion_chars}。"""
    with _usage_lock:
        return dict(_usage)