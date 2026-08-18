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
