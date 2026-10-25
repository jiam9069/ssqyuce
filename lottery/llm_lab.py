"""LLM 规律工坊（llm_lab）：让大模型真正参与「找规律」，而不是只看一张战绩表。

现状问题（V5 实测）
------------------
1. **LLM 看到的上下文极少**：``hypothesis.build_hypothesis_prompt`` 只喂了方法战绩、
   统计候选的前 12 条、最近 12 期开奖。**没有任何全历史序列、没有维度表、没有阶段切分**。
   于是 LLM 只能泛泛地提「重号延续」这类和统计挖掘重复的假说（实测 13 条全是套话）。
2. **找不到可验证的信号**：``evolution.verify_hypothesis_ab`` 对没有结构规则的假说
   （如 ``boost_method`` / ``prefer_numbers``）恒返回 Δ=0，所以无论 LLM 说什么，
   都会得到「无显著差异」→ 永久停在 ``proposed``。「AI 参与了但永远没结论」。
3. **没有阶段维度**：LLM 完全不知道「这条规律在过去哪个阶段有效、现在还灵不灵」。

本模块的做法（四段式，全部可复现）
----------------------------------
``build_lab_context``   把**可计算**的东西一次性交给 LLM：
    维度表（20 族 91 组参数）、as-of 扫描候选（含校正后 p）、阶段画像、
    号码/结构特征的分年时间序列、最近 12 期开奖、以及「搜索成本」提示。
``ask_llm``             多轮：第 1 轮提出规律，第 2 轮自我质疑（skeptic pass），
    第 3 轮据质疑修订。三轮都用结构化 JSON，任何失败都优雅降级。
``evaluate_claims``     LLM 的每条规律都**必须在维度空间里落地**才能被验证：
    - 说「用近 30 期热号」→ 映射到 ``hot/W=30/k=6`` → 直接跑 as-of 回放；
    - 说「上期邻号」→ ``neighbor/offset=1``；
    - 说「自己组合两个维度」→ ``combine(mode=intersection|union)``；
    - 实在映射不到 → 记 ``unmappable`` 并说明原因，**绝不算作验证通过**。
``persist_claims``      落 ``data/llm_claims.json``：每条规律带可映射性、as-of 效应、
    阶段画像、结论（support/reject/noise/unmappable），供前端与后续进化消费。

这样「让 AI 发挥更多作用」落在三件可检验的事上：**给了它更多维度的输入**、
**给它的每条产出都配了一条能跑出数字的验证路径**、**给每条规律配了阶段时效判定**。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, discovery, llm_client, regime

CLAIMS_FILE = config.DATA_DIR / "llm_claims.json"
MAX_CLAIMS_STORED = 200

# 单条规律的结论口径（与 regime 的状态机对齐，但更靠前：这里是对「LLM 提出的
# 具体维度组合」的判定，不是回归库里的假说生命周期）
VERDICTS = {
    "support": "支持（as-of 回放效应显著且阶段一致）",
    "reject": "反对（效应显著为负）",
    "regime_only": "仅阶段有效（整体不显著但要盯住）",
    "noise": "与随机无异",
    "unmappable": "不可验证（无法落在任何维度上）",
    "insufficient": "样本不足",
}

SYSTEM_PROMPT = (
    "你是一位严谨的彩票数据统计学家。当前任务是：在双色球历史开奖数据中寻找**可证伪**的规律，"
    "并且必须判定每条规律的**阶段有效性**（哪段时间有效、现在还灵不灵）。\n"
    "硬性要求：\n"
    "1. 只能基于我给出的、已经算好的统计事实发言，不许编造数字；引用数字时必须与我给的一致。\n"
    "2. 每条规律必须落到给定的「维度表」里（族名 + 参数），否则无法被回放验证，会被判为不可验证。\n"
    "3. 必须显式回答阶段问题：这条规律在哪些年份段显著、最近 3 年是否仍有效、为什么。\n"
    "4. 我给出的数据里**绝大多数维度都不显著**（这是彩票应有的结果）。如果你找不到真信号，"
    "就直接说找不到，并提出「应该继续检查哪些维度/交互」，不要为了凑数编造规律。\n"
    "5. 输出严格 JSON，不要任何解释性文字。"
)


# --------------------------------------------------------------------------
# 上下文构建：把可计算的事实一次性交给 LLM
# --------------------------------------------------------------------------



def _era_series(draws: Sequence[Dict]) -> Dict[str, Any]:
    """号码/结构特征的分年序列——这是 LLM 判断「阶段」最直接的原料。"""
    eras = regime.eras_for(draws, mode="calendar")  # LLM 上下文用自然年，便于人读
    years: Dict[str, Dict[str, Any]] = {}
    for e in eras:
        chunk = draws[e["start"]:e["end"]]
        reds = np.zeros(33)
        for d in chunk:
            for r in d.get("reds") or []:
                if 1 <= int(r) <= 33:
                    reds[int(r) - 1] += 1
        blues = np.zeros(16)
        for d in chunk:
            b = int(d.get("blue") or 0)
            if 1 <= b <= 16:
                blues[b - 1] += 1
        sums = [sum(d.get("reds") or []) for d in chunk]
        odd = [sum(1 for r in (d.get("reds") or []) if int(r) % 2) for d in chunk]
        top = np.argsort(-reds)[:5] + 1
        hot_b = np.argsort(-blues)[:3] + 1
        years[e["name"]] = {
            "期数": e["n"],
            "红球最热5": [int(x) for x in top],
            "红球最冷5": [int(x) for x in (np.argsort(reds)[:5] + 1)],
            "蓝球最热3": [int(x) for x in hot_b],
            "和值均值": round(float(np.mean(sums)) if sums else 0, 1),
            "和值标准差": round(float(np.std(sums)) if sums else 0, 1),
            "奇数均值": round(float(np.mean(odd)) if odd else 0, 2),
        }
    return {"阶段划分": [{"名称": e["name"], "期数": e["n"],
                        "期号": [e.get("first_issue"), e.get("last_issue")]} for e in eras],
            "分阶段统计": years}


def _recent(draws: Sequence[Dict], n: int = 12) -> List[Dict]:
    return [{"issue": d.get("issue"), "reds": d.get("reds"), "blue": d.get("blue")}
            for d in draws[-n:]]


def build_lab_context(draws: Sequence[Dict], scan: Optional[Dict] = None,
                      max_claims: int = 20) -> Dict[str, Any]:
    """组装 LLM 工坊的完整输入（全部为可计算事实，无未来信息泄漏）。

    ``scan`` 可传入 ``discovery.discover()`` 的结果以复用（否则现场跑一遍快速版）。
    """
    draws = list(draws or [])
    if not draws:
        return {"error": "无开奖数据"}
    if scan is None:
        scan = discovery.discover(draws, top=12)
    eras = _era_series(draws)
    latest = regime.stage_snapshot(draws)
    # as-of 快速扫描的候选（含校正后 p，让 LLM 看到「搜索成本」）
    candidates = []
    for it in (scan.get("items") or [])[:12]:
        q = it.get("quick") or {}
        candidates.append({
            "维度": it["filter"]["key"], "中文": it["filter"]["name_zh"],
            "参数": it["filter"]["params"], "目标": it["filter"]["target"],
            "近{}期均值".format(scan.get("quick_window") or ""): q.get("mean"),
            "期望": q.get("expected"), "差值": q.get("delta"), "z": q.get("z"),
            "未校正p": round(q["p"], 5) if q.get("p") is not None else None,
            "BH校正p": round(q["p_adjusted"], 5) if q.get("p_adjusted") is not None else None,
            "阶段状态": it.get("status_zh"), "阶段说明": it.get("reason_zh"),
        })
    return {
        "总期数": len(draws),
        "最近开奖": _recent(draws, 12),
        "维度表": discovery.family_catalog(),
        "维度空间大小": discovery.combo_space_size(),
        "扫描统计": {
            "已测试组合数": scan.get("n_tested"),
            "纯随机预期假阳性数": scan.get("expected_false_positives"),
            "最优BH校正p": scan.get("best_quick_p_adjusted"),
            "结论": scan.get("verdict"),
            "说明": scan.get("note_zh"),
        },
        "候选（按|z|排序，均为as-of回放）": candidates,
        **eras,
        "当前状态快照": {
            "期号": latest.get("issue"),
            "近50期红球频次": (latest.get("red_freq") or {}).get(50, {}).get("count"),
            "当前遗漏": latest.get("omission"),
            "平均遗漏": [None if (isinstance(x, float) and np.isnan(x)) else x
                       for x in (latest.get("omission_avg") or [])],
            "近50期结构": latest.get("structure"),
        },
        "统计检测力提示": {
            "单阶段最小可检出效应（红球命中/期）": 0.22,
            "说明": "单一年份(约150期)要达到校正显著，需要每期至少多命中约 0.22 个红球的效应；"
                    "更小的效应在统计上无法区分于噪声。",
        },
        "可调用的维度族": [f["key"] for f in discovery.FAMILIES],
    }


def build_user_prompt(ctx: Dict[str, Any], round_no: int = 1,
                      critique: Optional[List[Dict]] = None) -> str:
    """第 1 轮：提规律；第 2 轮：自我质疑；第 3 轮：据质疑修订。"""
    body = json.dumps(ctx, ensure_ascii=False)[:14000]
    if round_no == 1:
        return (
            f"## 统计上下文（全部为 as-of 计算结果）\n```json\n{body}\n```\n"
            "请输出 3~6 条**可验证**的规律探索结论，每条必须：\n"
            "- ``filters``：1~2 个维度组合（每个含 key 与 params），mode 为 intersection/union；\n"
            "- ``stage_view``：这条规律在哪些阶段显著、最近是否失效；\n"
            "- ``mechanism``：为什么可能成立（若无机理，请直说是数据巧合）；\n"
            "- ``expectation``：你预期 as_of 回放的效应方向与量级。\n"
            "另外给出 ``next_dimensions``：你认为值得继续挖但当前维度表没有覆盖的角度。\n"
            '输出：{"claims":[{"title":"...","filters":[{"key":"hot","params":{"window":30,"k":6}}],'
            '"mode":"intersection","stage_view":"...","mechanism":"...",'
            '"expectation":{"direction":"up","delta":0.05},"confidence":0.3}],'
            '"next_dimensions":["..."],"overall":"一句话结论"}'
        )
    if round_no == 2:
        return (
            f"## 上下文\n```json\n{body}\n```\n"
            f"## 你上一轮给出的规律\n```json\n{json.dumps(critique or [], ensure_ascii=False)[:4000]}\n```\n"
            "请对**每一条**做怀疑式审查，回答：\n"
            "1) 它和我给出的搜索成本（已测组合数/预期假阳性数）相比，是否只是搜索出来的假阳性？\n"
            "2) 阶段证据是否只来自单一年份（单年样本约 150 期，很容易偶然显著）？\n"
            "3) 机理是否真实存在，还是对随机噪声的事后叙事？\n"
            "给出 kept（值得保留）/ dropped（应丢弃）与理由，并补充任何被遗漏的更值得验证的方向。\n"
            '输出：{"reviews":[{"title":"...","verdict":"kept|dropped","reason":"..."}],'
            '"overall":"..."}'
        )
    return (
        f"## 上下文\n```json\n{body}\n```\n"
        f"## 怀疑式审查结论\n```json\n{json.dumps(critique or [], ensure_ascii=False)[:4000]}\n```\n"
        "请据此给出**最终版**规律清单（只保留经得起审查的；如果全部被否，就返回空列表并说明原因），"
        "每条仍须可在维度表里落地。\n"
        '输出：{"claims":[{"title":"...","filters":[{"key":"...","params":{}}],"mode":"intersection",'
        '"stage_view":"...","mechanism":"...","expectation":{"direction":"up","delta":0.05},'
        '"confidence":0.3}],"dropped_summary":"...","overall":"..."}'
    )


# --------------------------------------------------------------------------
# LLM 调用（任何失败都降级，绝不抛出）
# --------------------------------------------------------------------------

def ask_llm(ctx: Dict[str, Any], rounds: int = 3,
            max_tokens: int = 2400, progress=None) -> Dict[str, Any]:
    """三轮 LLM 规律探讨。返回 ``{claims, reviews, overall, next_dimensions,
    llm_used, error, per_round}``；通道不可用时整体降级为 ``llm_used=False``。"""
    out: Dict[str, Any] = {"claims": [], "reviews": [], "overall": None,
                            "next_dimensions": [], "llm_used": False,
                            "error": None, "per_round": []}
    if config.LLM_DISABLED:
        out["error"] = "LLM 已停用（config.LLM_DISABLED=True）"
        return out
    timeout = float(getattr(config, "LLM_LONG_TIMEOUT", 180))
    res1 = None
    try:
        res1 = llm_client.chat_json(
            SYSTEM_PROMPT, build_user_prompt(ctx, 1), max_tokens=max_tokens,
            temperature=0.7, timeout=timeout,
            deadline=time.time() + timeout * 2, strict=False)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"第 1 轮失败: {type(e).__name__}: {e}"
        return out
    if not isinstance(res1, dict):
        out["error"] = "第 1 轮未返回可解析 JSON（通道未配置/超时/模型返回空内容）"
        return out
    claims = _normalize_claims(res1.get("claims"))
    out["per_round"].append({"round": 1, "claims": len(claims), "ok": True})
    out["next_dimensions"] = [str(x) for x in (res1.get("next_dimensions") or [])][:8]
    out["overall"] = res1.get("overall")
    reviews: List[Dict] = []
    if rounds >= 2 and claims:
        if progress:
            progress(0.4, f"第 2 轮：自我质疑（对 {len(claims)} 条候选逐条审查）…")
        try:
            res2 = llm_client.chat_json(
                SYSTEM_PROMPT, build_user_prompt(ctx, 2, claims), max_tokens=max_tokens,
                temperature=0.4, timeout=timeout,
                deadline=time.time() + timeout * 2, strict=False)
        except Exception as e:  # noqa: BLE001
            res2 = None
            out["error"] = f"第 2 轮失败: {type(e).__name__}: {e}"
        if isinstance(res2, dict):
            reviews = [r for r in (res2.get("reviews") or []) if isinstance(r, dict)]
            out["per_round"].append({"round": 2, "reviews": len(reviews), "ok": True})
            # 只保留 kept（或没被判 dropped 的）
            dropped = {str(r.get("title")) for r in reviews
                       if str(r.get("verdict", "")).lower().startswith("drop")}
            kept_claims = [c for c in claims if str(c.get("title")) not in dropped]
            if rounds >= 3:
                if progress:
                    progress(0.65, "第 3 轮：据审查结论修订…")
                try:
                    res3 = llm_client.chat_json(
                        SYSTEM_PROMPT, build_user_prompt(ctx, 3, reviews),
                        max_tokens=max_tokens, temperature=0.3, timeout=timeout,
                        deadline=time.time() + timeout * 2, strict=False)
                except Exception as e:  # noqa: BLE001
                    res3 = None
                    out["error"] = f"第 3 轮失败: {type(e).__name__}: {e}"
                if isinstance(res3, dict) and res3.get("claims") is not None:
                    final = _normalize_claims(res3.get("claims"))
                    out["per_round"].append({"round": 3, "claims": len(final), "ok": True})
                    out["overall"] = res3.get("overall") or out["overall"]
                    claims = final
                else:
                    claims = kept_claims
            else:
                claims = kept_claims
        else:
            out["per_round"].append({"round": 2, "reviews": 0, "ok": False})
    out["claims"] = claims
    out["reviews"] = reviews
    out["llm_used"] = bool(claims) or bool(out["per_round"])
    return out


def _normalize_claims(raw: Any) -> List[Dict]:
    """LLM 输出 → 统一结构（只做结构清洗，不做判断）。"""
    out = []
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        flt = item.get("filters")
        if isinstance(flt, dict):
            flt = [flt]
        if not isinstance(flt, list):
            flt = []
        filters = []
        for f in flt:
            if isinstance(f, dict) and f.get("key"):
                filters.append({"key": str(f.get("key")), "params": f.get("params") or {}})
        exp = item.get("expectation") if isinstance(item.get("expectation"), dict) else {}
        out.append({
            "title": str(item.get("title") or "")[:120],
            "filters": filters[:3],
            "mode": "intersection" if str(item.get("mode")) == "intersection" else "union",
            "stage_view": str(item.get("stage_view") or "")[:400],
            "mechanism": str(item.get("mechanism") or "")[:400],
            "expectation": {"direction": str(exp.get("direction") or ""),
                             "delta": exp.get("delta")},
            "confidence": item.get("confidence"),
        })
    return out[:8]


# --------------------------------------------------------------------------
# 落地验证：把 LLM 的规律映射到维度空间并跑 as-of 回放
# --------------------------------------------------------------------------

def _params_ok(key: str, params: Dict) -> bool:
    """参数必须在维度表声明的档位里——否则等价于私自新开搜索维度，不能算已验证。"""
    fam = discovery.FAMILY_BY_KEY.get(key)
    if not fam:
        return False
    if not fam["params"]:
        return True
    return bool(params) and params in fam["params"]


def evaluate_claim(claim: Dict, draws: Sequence[Dict], start: int = 200) -> Dict:
    """单条 LLM 规律的判定：能否落地 → as-of 回放 → 阶段画像 → 结论。

    结论口径（``verdict``）：
      * ``unmappable``：维度名/参数不在维度表里 → **不可验证**（不计入发现，也不计入否定）
      * ``insufficient``：可回放样本 < 200 期
      * ``reject``：校正后 p < 0.05 且效应方向为负
      * ``support``：校正后 p < 0.05 且方向为正，且最近窗口未反向
      * ``regime_only``：整体不显著，但至少一个阶段显著（可能是阶段性规律）
      * ``noise``：整体与所有阶段都不显著
    """
    base = {"title": claim.get("title"), "verdict": "unmappable", "verdict_zh": VERDICTS["unmappable"],
            "reason": "", "filters": claim.get("filters"), "mode": claim.get("mode"),
            "regime": None, "effect": None, "p_adjusted": None, "n": 0}
    filters = claim.get("filters") or []
    if not filters:
        base["reason"] = "LLM 没有给出任何维度（filters 为空）"
        return base
    known = []
    unknown = []
    for f in filters:
        if _params_ok(str(f.get("key")), dict(f.get("params") or {})):
            known.append(discovery.make_filter(str(f["key"]), dict(f.get("params") or {})))
        else:
            unknown.append(f"{f.get('key')}{f.get('params') or ''}")
    if not known:
        base["reason"] = f"维度/参数不在维度表内：{', '.join(unknown)}（参数必须取自维度表档位）"
        return base
    if unknown:
        base["reason"] = f"部分维度无法落地，仅验证：{', '.join(f['label'] for f in known)}"
    try:
        flt = known[0] if len(known) == 1 else discovery.combine(known, claim.get("mode") or "union")
    except ValueError as e:
        base["reason"] = f"维度组合失败：{e}"
        return base
    try:
        rep = discovery.regime_of_filter(draws, flt, start=start)
    except Exception as e:  # noqa: BLE001
        base["reason"] = f"回放失败：{type(e).__name__}: {e}"
        return base
    scan = rep.get("scan") or {}
    n = int(scan.get("n") or 0)
    base["n"] = n
    base["label"] = flt["label"]
    base["regime"] = {
        "status": rep.get("status"), "status_zh": rep.get("label_zh"),
        "reason": rep.get("reason_zh"),
        "eras": [{k: e.get(k) for k in ("era", "n", "mean", "delta", "p_value", "p_adjusted")}
                 for e in rep.get("eras") or []],
        "recent": {k: (rep.get("recent") or {}).get(k)
                   for k in ("window", "n", "mean", "delta", "p_value", "p_adjusted")},
        "n_sig_eras": rep.get("n_sig_eras"),
        "last_significant_era": rep.get("last_significant_era"),
    }
    recent = rep.get("recent") or {}
    base["effect"] = recent.get("delta")
    base["p_adjusted"] = recent.get("p_adjusted")
    if n < 200:
        base["verdict"], base["verdict_zh"] = "insufficient", VERDICTS["insufficient"]
        base["reason"] = f"可回放样本仅 {n} 期 < 200，无法判定"
        return base
    status = rep.get("status")
    p_adj = recent.get("p_adjusted")
    delta = recent.get("delta")
    if p_adj is not None and p_adj < regime.ALPHA:
        if delta is not None and delta < 0:
            base["verdict"], base["verdict_zh"] = "reject", VERDICTS["reject"]
            base["reason"] = f"近窗效应 {delta:+.4f}，校正后 p={p_adj:.4g} 显著为负"
        else:
            base["verdict"], base["verdict_zh"] = "support", VERDICTS["support"]
            base["reason"] = f"近窗效应 {delta:+.4f}，校正后 p={p_adj:.4g} 显著为正"
        return base
    if status in ("persistent", "fading", "stale", "emerging", "flipped", "reversed"):
        base["verdict"], base["verdict_zh"] = "regime_only", VERDICTS["regime_only"]
        base["reason"] = rep.get("reason_zh")
        return base
    base["verdict"], base["verdict_zh"] = "noise", VERDICTS["noise"]
    base["reason"] = rep.get("reason_zh") or "整体与各阶段均不显著"
    return base


def evaluate_claims(claims: Sequence[Dict], draws: Sequence[Dict],
                    start: int = 200) -> List[Dict]:
    out = []
    for c in claims:
        try:
            out.append(evaluate_claim(c, draws, start=start))
        except Exception as e:  # noqa: BLE001
            out.append({"title": c.get("title"), "verdict": "unmappable",
                        "verdict_zh": VERDICTS["unmappable"],
                        "reason": f"评估异常：{type(e).__name__}: {e}"})
    return out


# --------------------------------------------------------------------------
# 落盘与读取
# --------------------------------------------------------------------------

def load_claims(path=None) -> Dict:
    p = path or CLAIMS_FILE
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": "v6-1", "updated_at": None, "claims": [], "runs": []}


def discovery_target(key: str) -> str:
    """维度族的目标（red/blue）——供 evolution 决定用哪种偏置动作。"""
    fam = discovery.FAMILY_BY_KEY.get(str(key))
    return (fam or {}).get("target") or "red"


def persist_run(result: Dict, path=None) -> Dict:
    """把一次工坊运行写入 ``data/llm_claims.json``（保留最近 N 条 + 运行摘要）。"""
    p = path or CLAIMS_FILE
    data = load_claims(p)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    claims = data.get("claims") or []
    # 去重：同名规律只保留最新一次判定
    titles = {c.get("title") for c in (result.get("claims") or [])}
    claims = [c for c in claims if c.get("title") not in titles]
    claims = list(result.get("claims") or []) + claims
    runs = data.get("runs") or []
    runs.insert(0, {
        "at": now, "llm_used": bool(result.get("llm_used")),
        "n_claims": len(result.get("claims") or []),
        "summary": result.get("summary") or {},
        "overall": result.get("overall"),
        "error": result.get("error"),
    })
    data.update({"version": "v6-1", "updated_at": now,
                 "claims": claims[:MAX_CLAIMS_STORED], "runs": runs[:20]})
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp.replace(p)
    except OSError as e:
        return {"ok": False, "error": f"写入失败：{e}"}
    return {"ok": True, "path": str(p), "stored": len(data["claims"])}


def run_closed_loop(draws: Sequence[Dict], rounds: int = 3, use_llm: bool = True,
                    persist: bool = True, progress=None) -> Dict:
    """**AI 规律闭环**：挖掘 → 验证 → 注册 → 更新规律快照 → 可用于出号。

    这是用户第 4 条诉求（"要用 AI 来挖掘规律、验证规律、使用规律"）的单一入口，
    也是把 V6 的三块能力串成一条可重复执行的流水线：

    1. **挖掘**：维度空间 as-of 扫描（91 组）+ LLM 三轮「提规律 → 自我质疑 → 修订」；
    2. **验证**：LLM 每条规律强制映射维度表并做 as-of 回放 + 9 阶段画像（`evaluate_claims`）；
    3. **注册**：只把 support/reject/regime_only 落进假说库（`evolution.register_llm_lab_claims`）；
    4. **使用**：重建规律运行时快照（`patterns_runtime.refresh`），
       让「阶段持续有效 + 通过验证」的规律真正进入 `engine.predict_next` 的出号链路；
    5. **复算阶段时效**：给假说库每条规律写回阶段状态（`evolution.annotate_hypotheses`）。

    每一步的产出都带样本量与阶段状态；任何一步失败只影响该步（结果里 ``errors`` 列出），
    绝不把失败包装成结论。
    """
    from . import evolution, patterns_runtime

    t0 = time.time()
    steps: List[Dict] = []
    errors: List[str] = []

    def _p(frac: float, msg: str) -> None:
        """进度回调（Web 端转成后台任务的 progress/message，前端可见）。

        这一步是用户反馈「点击后 15 分钟没有任何反应」的直接修复：整个闭环
        1~8 分钟、其中 LLM 三轮占绝大部分时间，没有进度就只能靠猜。
        """
        if progress is None:
            return
        try:
            progress(min(max(float(frac), 0.0), 0.99), msg)
        except Exception:  # noqa: BLE001
            pass

    _p(0.05, "① 维度扫描（91 组 as-of 回放，约 5~10 秒）…")
    lab = run_lab(draws, rounds=rounds, use_llm=use_llm, persist=persist,
                  progress=lambda f, m: _p(0.05 + 0.55 * f, m))
    steps.append({"step": "mine_and_verify", "llm_used": lab.get("llm_used"),
                  "n_claims": len(lab.get("claims") or []),
                  "summary": lab.get("summary"), "error": lab.get("error")})
    if lab.get("error") and not lab.get("llm_used"):
        errors.append(f"LLM 环节降级：{lab.get('error')}（维度扫描结果仍然有效）")

    _p(0.62, "③ 注册可验证规律进假说库…")
    reg = {}
    try:
        reg = evolution.register_llm_lab_claims(draws)
        steps.append({"step": "register", "picked": reg.get("picked"),
                      "registered": (reg.get("registered") or {}).get("registered"),
                      "skipped": reg.get("skipped_n")})
    except Exception as e:  # noqa: BLE001
        errors.append(f"注册失败：{type(e).__name__}: {e}")

    _p(0.70, "④ 重算每条规律的阶段时效…")
    ann = {}
    try:
        ann = evolution.annotate_hypotheses(draws)
        steps.append({"step": "annotate_regime", **{k: ann.get(k) for k in
                                                   ("annotated", "no_dimension", "total")}})
    except Exception as e:  # noqa: BLE001
        errors.append(f"阶段标注失败：{type(e).__name__}: {e}")

    _p(0.85, "⑤ 重建规律快照（供本期出号使用）…")
    snap = {}
    try:
        snap = patterns_runtime.refresh(draws)
        steps.append({"step": "refresh_runtime", "counts": (snap.get("snapshot") or {}).get("counts")})
    except Exception as e:  # noqa: BLE001
        errors.append(f"规律快照刷新失败：{type(e).__name__}: {e}")

    _p(1.0, "闭环完成")
    return {
        "ok": True,
        "closed_loop": True,
        "llm_used": bool(lab.get("llm_used")),
        "claims": lab.get("claims") or [],
        "summary": lab.get("summary") or {},
        "overall": lab.get("overall"),
        "next_dimensions": lab.get("next_dimensions") or [],
        "scan_summary": lab.get("scan_summary"),
        "steps": steps,
        "errors": errors,
        "runtime": {"counts": (snap.get("snapshot") or {}).get("counts"),
                    "base_issue": (snap.get("snapshot") or {}).get("base_issue"),
                    "generated_at": (snap.get("snapshot") or {}).get("generated_at_text"),
                    "usable_rules": (snap.get("snapshot") or {}).get("usable_rules")},
        "elapsed_sec": round(time.time() - t0, 2),
        "note_zh": ("闭环已跑完：" + (f"LLM 参与并给出 {len(lab.get('claims') or [])} 条可验证规律；"
                                     if lab.get("llm_used") else "LLM 未参与（仅维度扫描）；")
                    + f"规律快照现有 {((snap.get('snapshot') or {}).get('counts') or {}).get('usable', 0)} 条可用规律。"),
    }


def summarize_verdicts(claims: Sequence[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    for c in claims:
        v = c.get("verdict") or "unknown"
        counts[v] = counts.get(v, 0) + 1
    return counts


def run_lab(draws: Sequence[Dict], rounds: int = 3, use_llm: bool = True,
            persist: bool = True, scan: Optional[Dict] = None,
            progress=None) -> Dict:
    """完整跑一次规律工坊：上下文 →（多轮）LLM → 逐条落地验证 → 落盘。

    ``use_llm=False`` 时只做「维度扫描 + 阶段画像」，用于无 LLM 环境下的自检。
    """
    draws = list(draws or [])
    t0 = time.time()
    if progress:
        progress(0.0, "维度扫描中…")
    if scan is None:
        scan = discovery.discover(draws, top=12)
    result: Dict[str, Any] = {
        "ok": True, "llm_used": False, "error": None,
        "claims": [], "reviews": [], "overall": None, "next_dimensions": [],
        "scan_summary": {
            "n_tested": scan.get("n_tested"),
            "expected_false_positives": scan.get("expected_false_positives"),
            "best_quick_p_adjusted": scan.get("best_quick_p_adjusted"),
            "verdict": scan.get("verdict"),
            "status_count": scan.get("status_count"),
        },
    }
    if use_llm:
        ctx = build_lab_context(draws, scan)
        if progress:
            progress(0.1, "第 1 轮：让大模型提规律…")
        llm = ask_llm(ctx, rounds=rounds, progress=progress)
        result.update({"error": llm.get("error"), "reviews": llm.get("reviews") or [],
                       "overall": llm.get("overall"),
                       "next_dimensions": llm.get("next_dimensions") or [],
                       "per_round": llm.get("per_round") or []})
        if progress:
            progress(0.9, "逐条落地验证（as-of 回放 + 阶段画像）…")
        evaluated = evaluate_claims(llm.get("claims") or [], draws)
        result["claims"] = _merge_claims(llm.get("claims") or [], evaluated)
        result["llm_used"] = bool(llm.get("llm_used"))
    result["summary"] = summarize_verdicts(result["claims"])
    result["elapsed_sec"] = round(time.time() - t0, 2)
    if persist:
        result["persisted"] = persist_run(result)
    return result


def _merge_claims(raw: Sequence[Dict], evaluated: Sequence[Dict]) -> List[Dict]:
    """把 LLM 原始表述与回放判定合并成一条可展示记录。"""
    out = []
    for r, e in zip(raw, evaluated):
        out.append({**e,
                    "stage_view": r.get("stage_view"),
                    "mechanism": r.get("mechanism"),
                    "expectation": r.get("expectation"),
                    "llm_confidence": r.get("confidence")})
    return out
