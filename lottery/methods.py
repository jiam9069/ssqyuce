"""M4.2 方法 A/B 开关：LOTT_METHODS 配置驱动的通道过滤（仅标准库，无第三方依赖）。

方法全名形如 ``stat:freq`` / ``blend:brier`` / ``llm:minimax-m3`` / ``ml`` / ``uniform``，
由 ``族:叶子`` 组成，族自引擎/models/eval 输出的 method 字段而来。

开关语义（环境变量 ``LOTT_METHODS``，逗号或空格分隔，未设置 = 全部启用）：
- 空 / 未设置 / ``all``          → 全部启用；
- 含 ``-`` 前缀令牌（拒绝列表）  → 除令牌外全部启用，如 ``-llm`` 仅关闭 LLM 通道；
- 无 ``-`` 前缀令牌（允许列表）  → 仅令牌及其族启用，如 ``stat,ml`` 仅保留统计基线 + ML。
令牌可写全名（``stat:freq``）或族名（``stat`` / ``llm``），族名匹配该族下全部方法。

对外 API：``implement_spec`` → ``normalize_method`` → ``is_enabled`` → ``filter_candidates``。
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Set

ENV_NAME = "LOTT_METHODS"
MODE_ENV_NAME = "LOTT_METHOD_MODE"   # 运行模式：production / research

# 本系统已知的方法族（用于族名匹配；未知令牌按字面精确匹配兜底）
FAMILIES = ("stat", "blend", "llm", "ml", "uniform")

# 运行模式合法取值
MODES = ("production", "research")

# 开关规格：{"mode": "all" | "allow" | "deny", "tokens": Set[str]}
Spec = Dict

# 方法注册表：全名 / 族 / 中文说明，供前端设置页与文档展示
METHOD_REGISTRY: List[Dict] = [
    {"method": "stat:freq",     "family": "stat",     "desc": "频率加权统计基线"},
    {"method": "stat:omission", "family": "stat",     "desc": "遗漏加权统计基线"},
    {"method": "stat:markov",   "family": "stat",     "desc": "马尔可夫转移统计基线"},
    {"method": "stat:bayes",    "family": "stat",     "desc": "贝叶斯平滑统计基线"},
    {"method": "blend:brier",   "family": "blend",    "desc": "Brier 加权概率融合"},
    {"method": "ml",            "family": "ml",       "desc": "GBDT+RF 概率模型（M2）"},
    {"method": "llm",           "family": "llm",      "desc": "LLM 推理通道（多模型采样 + 第三轮校验）"},
    {"method": "uniform",       "family": "uniform",  "desc": "均匀随机对照基线"},
]


def normalize_method(method: object) -> str:
    """规整方法名为小写全名（str 安全），如 stat:freq / llm:minimax-m3 / ml / uniform。"""
    return str(method or "").strip().lower()


def _family(method: str) -> str:
    """取方法族：'stat:freq' -> 'stat'，'ml' -> 'ml'。"""
    return method.split(":", 1)[0] if method else ""


def implement_spec(raw: Optional[str] = None) -> Spec:
    """把 LOTT_METHODS 原始字符串（缺省读环境变量）解析为开关规格。

    返回 {"mode": "all"|"allow"|"deny", "tokens": 规范化令牌集合}。
    """
    tokens: Set[str] = set()
    deny: Set[str] = set()
    if raw is None:
        raw = os.environ.get(ENV_NAME, "")
    for tok in str(raw or "").replace(",", " ").split():
        tok = normalize_method(tok)
        if not tok:
            continue
        if tok == "all":
            return {"mode": "all", "tokens": set()}
        if tok.startswith("-"):
            deny.add(tok[1:])
        else:
            tokens.add(tok)
    if deny:
        return {"mode": "deny", "tokens": deny}
    if tokens:
        return {"mode": "allow", "tokens": tokens}
    return {"mode": "all", "tokens": set()}


def normalize_mode(mode: object) -> str:
    """规整运行模式为合法取值（production 默认）。"""
    m = normalize_method(mode)
    return m if m in MODES else "production"


def registry() -> List[Dict]:
    """已知方法注册表（全名 / 族 / 中文说明）的深拷贝，供 API 与前端展示。"""
    return [dict(e) for e in METHOD_REGISTRY]


def validate_raw(raw: object) -> Optional[str]:
    """校验 LOTT_METHODS 原始字符串；返回错误信息（None = 合法）。

    令牌形如 ``stat`` / ``stat:freq`` / ``-llm``：仅含字母数字、下划线、
    冒号、点，可选 ``-`` 前缀表示关闭；未知令牌按字面精确匹配兜底（向后兼容）。
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) > 500:
        return "methods 过长（≤500 字符）"
    for tok in text.replace(",", " ").split():
        tok = normalize_method(tok)
        if not tok or tok == "all":
            continue
        core = tok[1:] if tok.startswith("-") else tok
        if not core:
            return f"非法令牌 {tok!r}：- 前缀后缺少方法名"
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_:." for c in core):
            return (f"非法令牌 {tok!r}：仅允许 字母/数字/下划线/冒号/点"
                    "（- 前缀表示关闭该通道）")
    return None


def effective_spec(mode: object = None, spec: Optional[Spec] = None) -> Spec:
    """按运行模式返回**生效**开关规格。

    - research（研究模式）：忽略 A/B 开关，全部方法启用（用于方法对比实验，
      与决策规则「未经 120 期 paired 验证的方法仅以研究模式存在」对应）；
    - production（生产模式，默认）：正常应用 spec（缺省按环境变量解析）。
    """
    if normalize_mode(mode) == "research":
        return {"mode": "all", "tokens": set()}
    return spec if spec is not None else implement_spec()


def is_enabled(method: object, spec: Optional[Spec] = None) -> bool:
    """判断单个方法（全名或族名）是否在开关内。spec 缺省按环境变量即时解析。

    全部启用模式下未知方法视为启用（保持向后兼容）；allow/deny 模式下无 method
    的对象视为禁用。
    """
    if spec is None:
        spec = implement_spec()
    m = normalize_method(method)
    if not m:
        return spec["mode"] == "all"
    if spec["mode"] == "all":
        return True
    fam = _family(m)
    if spec["mode"] == "deny":
        return m not in spec["tokens"] and fam not in spec["tokens"]
    # allow：全名或族名任一命中即启用
    return m in spec["tokens"] or fam in spec["tokens"]


def filter_candidates(tickets: Iterable[Dict], spec: Optional[Spec] = None) -> List[Dict]:
    """按开关过滤候选票，仅保留启用的方法产生的票（保持原始顺序）。"""
    if spec is None:
        spec = implement_spec()
    return [t for t in tickets if is_enabled(t.get("method"), spec)]


# ==========================================================================
# U3 方法自适应降权（仅标准库；持久化到 data/adaptive_state.json，不改 SQLite schema）
#
# 诚实口径：双色球为独立随机事件，任何方法都不该被“神化”。本模块做的是——
#   连续 K 期命中低于 uniform 随机基线的方法，在 production 模式自动降权 / 移出候选池；
#   研究模式（LOTT_METHOD_MODE=research）始终展示全部方法，不删除任何方法。
# 权重以方法“族:叶子”全名为键（如 stat:markov / llm:deepseek-v4-flash / uniform）。
# ==========================================================================


def _adaptive_path() -> str:
    from . import config
    return str(config.ADAPTIVE_STATE_FILE)


def _family_leaf(method: str) -> str:
    return _family(method)


def _load_adaptive() -> Dict:
    """读取自适应状态：{method: {"strike": int, "weight": float, "periods": int}}。"""
    import json, os
    try:
        with open(_adaptive_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_adaptive(state: Dict) -> None:
    import json, os
    from . import config
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.ADAPTIVE_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(config.ADAPTIVE_STATE_FILE))
    except OSError as e:  # noqa: BLE001
        print(f"[methods] 自适应状态写入失败: {e}")


def _per_method_issue_rows(limit: int = 500) -> Dict[str, List[Dict]]:
    """直接从 eval_details 读逐注明细（不改 SQLite schema），按方法分组、按期排序。

    返回 {method: [{issue, red_hits, blue_hit, prize_level}]}。
    """
    from . import db
    conn = db.get_conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT issue, method, red_hits, blue_hit, prize_level "
        "FROM eval_details ORDER BY issue ASC, method, seq").fetchall()[-limit * 200:]]
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    return out


def _issue_means(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """按期聚合每方法的 red_hits 均值 / blue_hit 率 / ≥五等奖率。"""
    by_issue: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        it = by_issue.setdefault(r["issue"], {})
        agg = it.setdefault(r["method"], {"r_sum": 0.0, "b_sum": 0.0, "p5": 0, "n": 0})
        agg["r_sum"] += float(r["red_hits"])
        agg["b_sum"] += float(r["blue_hit"])
        agg["p5"] += 1 if int(r["prize_level"]) >= 5 else 0
        agg["n"] += 1
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for issue, methods in by_issue.items():
        out[issue] = {
            m: {"red": v["r_sum"] / v["n"],
                "blue": v["b_sum"] / v["n"],
                "prize": v["p5"] / v["n"]}
            for m, v in methods.items()
        }
    return out


def refresh_adaptive_weights(window: Optional[int] = None) -> Dict:
    """每期开奖后调用：比较各方法与 uniform 基线，更新累计降权决策。

    返回 {method: {"weight": w, "downweighted": bool, "strike": int, "periods": int,
                   "red_hits_mean": float|None, "blue_hit_rate": float|None}}。
    """
    from . import config
    window = int(window if window is not None else config.ADAPTIVE_WINDOW)
    metric = config.ADAPTIVE_METRIC if config.ADAPTIVE_METRIC in ("red", "blue", "prize") else "red"

    per_method = _per_method_issue_rows(window * 50)
    # 按期聚合成 {issue: {method: metrics}}
    issue_map = _issue_means(sum(per_method.values(), []))
    baseline = "uniform"

    strikes = {}          # method -> 当前连续“低于基线”计数（本次窗口）
    totals = {}           # method -> 累计命中统计
    n_periods = {}        # method -> 有效配对期数
    for issue, methods in issue_map.items():
        if baseline not in methods:
            continue
        base_val = methods[baseline][metric]
        for m, v in methods.items():
            if m == baseline:
                continue
            val = v[metric]
            totals.setdefault(m, {"reds": 0.0, "blues": 0.0, "prizes": 0.0, "n": 0})
            # _issue_means 输出键为 red/blue/prize（按期聚合的均/率）
            totals[m]["reds"] += v["red"]
            totals[m]["blues"] += v["blue"]
            totals[m]["prizes"] += v["prize"]
            totals[m]["n"] += 1
            n_periods[m] = totals[m]["n"]
            strikes[m] = strikes.get(m, 0) + (1 if val < base_val else 0)

    # 结合持久化 strike（跨重启保留连续劣绩计数）
    prev = _load_adaptive()
    result: Dict[str, Dict] = {}
    for m, n_p in n_periods.items():
        prev_s = prev.get(m, {}).get("strike", 0)
        # 若本次窗口内该期确比基线差则累加，否则重置为 0
        strike = strikes.get(m, 0)
        cum_strike = prev_s if n_p < window else (strike if strike > 0 else 0)
        t = totals[m]
        red_mean = round(t["reds"] / max(1, t["n"]), 4)
        blue_rate = round(t["blues"] / max(1, t["n"]), 4)
        prize_rate = round(t["prizes"] / max(1, t["n"]), 4)
        downweighted = bool(config.ADAPTIVE_ENABLED and cum_strike >= int(config.ADAPTIVE_K))
        weight = config.ADAPTIVE_DOWNWEIGHT_FACTOR if downweighted else 1.0
        result[m] = {
            "weight": weight, "downweighted": downweighted,
            "strike": cum_strike, "periods": n_p,
            "red_hits_mean": red_mean, "blue_hit_rate": blue_rate,
            "prize_rate_ge5": prize_rate,
        }
    # 研究模式不降权（仅记录）
    if config.METHOD_MODE == "research":
        for m in result:
            result[m]["weight"] = 1.0
            result[m]["downweighted"] = False
    _save_adaptive(result)
    return result


def adaptive_weights() -> Dict:
    """返回 {method_fullname: weight}，供引擎融合与候选过滤使用。

    研究模式或自适应关闭时全部返回 1.0；production 模式优先用已计算权重。
    """
    from . import config
    if not config.ADAPTIVE_ENABLED or config.METHOD_MODE == "research":
        return {}
    return {m: v.get("weight", 1.0) for m, v in _load_adaptive().items()}


def adaptive_status() -> Dict:
    """供 /api/info 与 /api/eval/cumulative 展示方法自适应降权状态。"""
    from . import config
    state = refresh_adaptive_weights()
    enabled = config.ADAPTIVE_ENABLED and config.METHOD_MODE != "research"
    return {
        "enabled": enabled,
        "mode": config.METHOD_MODE,
        "k": int(config.ADAPTIVE_K),
        "metric": config.ADAPTIVE_METRIC,
        "baseline": "uniform",
        "window": int(config.ADAPTIVE_WINDOW),
        "methods": state,
        "note": ("连续 K 期命中低于随机基线的方法在 production 模式被自动降权；"
                 "双色球为独立随机事件，此降权仅为诚实聚焦，不构成中奖保证。"),
    }


def _blue_running_rate(limit: int = 30) -> Optional[float]:
    """蓝球模型近 limit 期命中率（按 method 蓝球命中统计；无样本返回 None）。"""
    from . import config
    rows = _per_method_issue_rows(limit * 50)
    blues: List[int] = []
    for r in sum(rows.values(), []):
        blues.append(int(r["blue_hit"]))
    if not blues:
        return None
    return sum(blues) / len(blues)
