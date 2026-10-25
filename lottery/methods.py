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
import time
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
    # MU-2：方法战绩必须按用户隔离，否则会拿别人的票判定"我的方法好不好"
    rows = [dict(r) for r in conn.execute(
        "SELECT issue, method, red_hits, blue_hit, prize_level "
        "FROM eval_details WHERE user_id=? ORDER BY issue ASC, method, seq",
        (db.resolve_tenant(),)).fetchall()[-limit * 200:]]
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


# ==========================================================================
# V10-00a 契约冻结：resolve_channels() —— 本期出号通道的**唯一**判据
#
# 为什么需要它（docs/reviews/R1_ARCHITECTURE_REVIEW.md P0-1）：
#   项目已有 6 层开关（LOTT_METHODS / LOTT_METHOD_MODE / LOTT_LLM_ONLY_OUTPUT /
#   LOTT_LLM_DISABLED / methods_config.json / .env），v1.0 方案还想加第 7 套
#   LOTT_PREDICT_CHANNELS —— 必然造成设置漂移。本函数把它们收敛为一个返回体。
#
# 优先级链（硬性）：
#   1) data/methods_config.json （设置页写入，import 期由 config 加载）
#   2) .env: LOTT_METHODS / LOTT_METHOD_MODE / LOTT_LLM_ONLY_OUTPUT
#   3) 代码默认: "stat,ml,uniform"，llm_output_mode="off"
#
# 【关键实现纪律 —— S2 二次评审 B3】
#   config.py 会把 methods_config.json 的生效值**写回 os.environ**
#   （见 config._apply_runtime_methods / _load_runtime_methods_config），
#   因此 `os.environ.get("LOTT_METHODS")` 永久**无法区分**"来自 json"和"来自 .env"。
#   判定来源必须用"查文件是否存在" + "`in os.environ`"（键是否被设置过），
#   **不得**用 .get(...) is not None。
# ==========================================================================

DEFAULT_SPEC_RAW = "stat,ml,uniform"     # 代码默认：不带 llm / blend
DEFAULT_MODE = "production"
DEFAULT_LLM_OUTPUT_MODE = "off"

# LLM 出号模式合法取值
#
# ## V22 / L2 新增 `advisory`：AI 只做解释，不参与选号
#
# 依据 docs/analysis/V19_AI_REDESIGN.md §4 层 2 与 §1.2 实测：LLM 票与统计票的
# 历史战绩无统计差异（45 注：llm 红球 1.265 / 统计 1.182，基线 1.091），
# 而 LLM 参与出号要付 503 风险、延迟与成本。既然选号无增量，
# 就让 AI 去做它**真正擅长且可感**的事：解释统计结果、复盘开奖归因。
#
#   off      → 完全不使用 LLM（默认）
#   advisory → **不参与选号**（llm_can_pick=False），但允许被调用来做
#              「解释 / 复盘」；响应里会标注 ai_role="advisor"
#   only / mix → 保留原有出号语义（向后兼容，未删除）
#
# **不删除 only/mix** 的理由：它们是已交付的行为，可能有人依赖；
# L2 只是新增一个更诚实的模式并把它设为推荐值，而不是强行改变既有语义。
LLM_OUTPUT_MODES = ("off", "advisory", "only", "mix")


def _methods_config_file():
    """延迟导入，避免 methods <-> config 的循环 import。"""
    from . import config
    return config.METHODS_CONFIG_FILE


def _read_methods_config() -> Dict:
    """读取 methods_config.json；不存在/损坏返回 {}。"""
    import json
    path = _methods_config_file()
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


# ---------- V10-11：技能关停状态（写 data/llm_config.json，由 resolve_channels 读取） ----------
#
# 为什么状态放 ``llm_config.json`` 而**不是** os.environ（R3 G6 三条约束）：
#   1. **不得与 ``LLM_DISABLED`` 共用**：那个开关会让 Web 预测直接 503
#      （``config.LLM_DISABLED`` → ``engine.predict_next(llm_required=True)`` 抛
#      ``LLMChannelError``）。技能关停只关"LLM 出号决策通道"，预测必须照常成功。
#   2. **必须可持久**：写 ``os.environ`` 只在当前进程有效，容器重启即失效。
#   3. **必须与既有优先级链一致**：``llm_config.json`` 是设置页与运行时配置的
#      住地（``config.load_runtime_llm_config`` 读它），所以放在这里并由
#      ``resolve_channels()`` 统一读取，不新增第 7 层开关。

#: 关停标志在 llm_config.json 里的键名（冻结）
DECISION_DISABLED_KEY = "decision_disabled"
DECISION_REASON_KEY = "decision_disabled_reason"
DECISION_AT_KEY = "decision_disabled_at"


def _llm_config_file():
    """``data/llm_config.json`` 路径（延迟导入 config，避免循环 import）。

    **必须容错**：``config`` 在 import 期（``config.py:175``）就调
    ``resolve_channels()``，那时 ``config.LLM_CONFIG_FILE``（定义在 :421）**还不存在**。
    因此这里用 ``getattr`` 兜底到 ``DATA_DIR / "llm_config.json"``——两者是同一路径，
    只是绕开"半初始化的 config 模块"。
    """
    from . import config
    path = getattr(config, "LLM_CONFIG_FILE", None)
    if path is not None:
        return path
    return config.DATA_DIR / "llm_config.json"


def read_llm_runtime_config(path=None) -> Dict:
    """读 ``llm_config.json``（容错：不存在/损坏 → ``{}``）。

    注意：**只读键，不打印内容** —— 文件里有 ``api_key``。
    """
    import json
    p = path if path is not None else _llm_config_file()
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def decision_state(path=None) -> Dict:
    """当前"LLM 决策通道"关停状态（供 ``resolve_channels()`` 与 API 读取）。

    返回 ``{"disabled": bool, "reason": str, "at": float|None, "source": str}``。
    ``source`` 为 ``llm_config.json`` 或 ``default``，便于前端说明"这个关停是谁写的"。
    """
    conf = read_llm_runtime_config(path)
    if DECISION_DISABLED_KEY not in conf:
        return {"disabled": False, "reason": "", "at": None, "source": "default"}
    at = conf.get(DECISION_AT_KEY)
    try:
        at = float(at) if at is not None else None
    except (TypeError, ValueError):
        at = None
    return {
        "disabled": bool(conf.get(DECISION_DISABLED_KEY)),
        "reason": str(conf.get(DECISION_REASON_KEY) or ""),
        "at": at,
        "source": "llm_config.json",
    }


def set_decision_disabled(disabled: bool, reason: str = "", *, path=None) -> Dict:
    """写入/解除技能关停状态，**保留 ``llm_config.json`` 里的其它键**。

    先写 ``.tmp`` 再 ``os.replace``（原子替换），避免半截文件让
    ``config.load_runtime_llm_config`` 解析失败而**丢掉整份配置**。
    返回写入后的状态（同 :func:`decision_state`）。
    """
    import json
    import os as _os
    p = path if path is not None else _llm_config_file()
    conf = read_llm_runtime_config(p)
    conf[DECISION_DISABLED_KEY] = bool(disabled)
    if disabled:
        conf[DECISION_REASON_KEY] = str(reason or "")
        conf[DECISION_AT_KEY] = time.time()
    else:
        # 恢复：清掉 reason/at（留一个明确的 False 仍然比删键更可审计）
        conf[DECISION_REASON_KEY] = str(reason or "")
        conf[DECISION_AT_KEY] = None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(str(tmp), str(p))
    return decision_state(p)


def resolve_layer() -> str:
    """判定生效开关来自哪一层：``methods_config.json`` / ``.env`` / ``default``。

    顺序（不得调换）：
      1. 文件存在且含 raw/mode → 文件层（它覆盖一切）
      2. 否则 ``"LOTT_METHODS" in os.environ`` 或 ``"LOTT_METHOD_MODE" in os.environ``
         → .env 层（用 ``in`` 而不是 ``.get``，见上方纪律说明）
      3. 否则 default
    """
    conf = _read_methods_config()
    if conf.get("raw") or conf.get("mode") or conf.get("llm_output_mode"):
        return "methods_config.json"
    if ENV_NAME in os.environ or MODE_ENV_NAME in os.environ or "LOTT_LLM_ONLY_OUTPUT" in os.environ:
        return ".env"
    return "default"


def _resolve_llm_output_mode(conf: Dict, fam_llm: bool) -> str:
    """解析 LLM 出号模式，并处理与族开关的冲突。

    冲突规则：拒绝列表优先 —— 若 LLM 族被关闭，则出号模式强制 off（幂等、只 warning）。
    """
    raw_mode = conf.get("llm_output_mode")
    if not raw_mode:
        # 兼容旧开关 LOTT_LLM_ONLY_OUTPUT（1→only / 0→off）；代码默认已改为 0
        legacy = os.environ.get("LOTT_LLM_ONLY_OUTPUT")
        if legacy is None:
            raw_mode = DEFAULT_LLM_OUTPUT_MODE
        else:
            raw_mode = "only" if str(legacy).strip() == "1" else "off"
    mode = str(raw_mode).strip().lower()
    if mode not in LLM_OUTPUT_MODES:
        mode = DEFAULT_LLM_OUTPUT_MODE
    if not fam_llm and mode != "off":
        _warn_once("llm_output_mode 被族开关覆盖为 off（拒绝列表优先）")
        return "off"
    return mode


_WARNED: Set[str] = set()


def _warn_once(msg: str) -> None:
    """幂等告警（同一消息只打一次，避免热路径刷屏）。"""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[methods] {msg}")


def resolve_channels(*, force_production: bool = False) -> Dict:
    """本期出号通道的**唯一**判据（V10-00a 契约冻结，返回 schema 只增不改）。

    返回字段：
      spec_raw        生效的原始开关串
      source          命中哪一层：methods_config.json / .env / default
      mode            production / research
      families        各族是否启用 {stat, blend, llm, ml, uniform}
      llm_output_mode off / only / mix
      llm_can_pick    families["llm"] 且 llm_output_mode != "off" 且 mode == production
      research_arm    研究对照臂 {stat: bool, llm_probe: bool}
      fingerprint     sha1(canonical_json(上面除 fingerprint 外全部字段))[:16]

    force_production=True 供回放/测试绕过 research 模式。
    """
    import hashlib
    import json

    conf = _read_methods_config()
    source = resolve_layer()

    # 生效 raw / mode：文件层优先，其次 env，最后代码默认
    if conf.get("raw") is not None:
        spec_raw = str(conf.get("raw") or "").strip()
    else:
        spec_raw = (os.environ.get(ENV_NAME) or "").strip()
        # V10：代码默认从"全部启用(空=all)"改为 allow-list ``stat,ml,uniform``
        # —— "把最没用的通道(llm)设成主力"的根因修复。空串在旧语义里=全部启用，
        #    这里必须显式落到 DEFAULT_SPEC_RAW，否则 llm/blend 仍会被启用。
        if not spec_raw:
            spec_raw = DEFAULT_SPEC_RAW

    if conf.get("mode") is not None:
        mode = normalize_mode(conf.get("mode"))
    else:
        mode = normalize_mode(os.environ.get(MODE_ENV_NAME))
    if force_production:
        mode = "production"

    spec = implement_spec(spec_raw)

    # research 模式：全部方法启用（保持现状语义）
    effective = effective_spec(mode, spec)
    families = {fam: bool(is_enabled(fam, effective)) for fam in FAMILIES}

    # V12 严格模式：强制启用 llm 族（用户指令「必须 LLM 出号」）。
    # methods 配置（.env LOTT_METHODS）默认不含 llm（V10 起 stat,ml,uniform），
    # 非严格模式保持原样；严格模式下必须放行，否则 llm_family=False 直接否决。
    if os.environ.get("LOTT_STRICT_MODE", "0") == "1":
        families["llm"] = True

    llm_output_mode = _resolve_llm_output_mode(conf, families["llm"])
    # V10-11：技能关停（决策技能 Wilson 下界 ≤ 随机基线 → 关掉"LLM 出号决策通道"）。
    # **只影响 llm_can_pick**：预测链路照常出号（统计/ML/uniform），不会像
    # LLM_DISABLED 那样让 Web 预测 503。状态来自 data/llm_config.json。
    _decision = decision_state()
    decision_disabled = bool(_decision["disabled"])
    # V12 严格模式豁免（用户 2026-10-18 指令：「必须 LLM 出号，否则不出号」）：
    # 严格模式开启时强制 llm_can_pick=True —— 覆盖 V10 的默认 off（"LLM 无信息"）
    # 与决策技能关停（decision_disabled）。这是用户明确选择的严格语义；
    # 非严格模式仍走 V10 诚实默认。
    # 不在此处判断 LLM 是否"已配置"：配置判定交给出号时的 llm_tickets
    # （缺配置则拒绝/降级，Web 路径走 LLMChannelError → 503 明确报错，
    # 仍符合"LLM 不可用则不出号"的严格语义）。
    _strict = os.environ.get("LOTT_STRICT_MODE", "0") == "1"
    # V22 / L2：`advisory` 模式**永不出号** —— 无论是否严格模式、是否启用 llm 族。
    # 这是该模式的全部意义：AI 做解释，不做选号。若它在这里仍能 pick，
    # "AI 不参与选号"就只是文案而不是事实。
    _advisory = (llm_output_mode == "advisory")
    llm_can_pick = bool(
        families["llm"] and llm_output_mode not in ("off", "advisory")
        and mode == "production" and not decision_disabled
    )
    if _strict and families["llm"] and not _advisory:
        llm_can_pick = True
    # AI 在本次出号中的角色（供响应/前端标注；不参与任何判据）。
    #
    # ⚠️ 不能只看 llm_can_pick：生产实测 STRICT_MODE=1 时，
    # V12 的严格模式豁免会把 `llm_can_pick` 强制成 True，而 `llm_output_mode`
    # 仍是 off（配置没改）。若据此报 ai_role=picker，就会出现
    # **「模式 off 却说 AI 参与选号」**的自相矛盾读数（V22 上线后被这条实测抓到）。
    # 正确口径是：只有真正能产出 LLM 注的模式（only/mix）才算 picker。
    if _advisory:
        ai_role = "advisor"
    elif llm_can_pick and llm_output_mode in ("only", "mix"):
        ai_role = "picker"
    else:
        ai_role = "off"

    research_arm = {
        "stat": bool(families["stat"]),
        "llm_probe": bool(families["llm"] and mode == "research"),
    }

    body = {
        "spec_raw": spec_raw,
        "source": source,
        "mode": mode,
        "families": families,
        "llm_output_mode": llm_output_mode,
        "llm_can_pick": llm_can_pick,
        # V22：AI 的**角色**（off / advisor / picker）。前端必须能显示它，
        # 否则用户会以为"AI 参与了选号"而其实只是解释了结果。
        "ai_role": ai_role,
        "research_arm": research_arm,
        # ---- V10-11 新增（schema 只增不改）----
        "decision_disabled": decision_disabled,
        "decision_disabled_at": _decision["at"],
        "decision_disabled_reason": _decision["reason"],
        "decision_disabled_source": _decision["source"],
    }
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    body["fingerprint"] = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
    return body
