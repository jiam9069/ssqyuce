"""多维度规律发现（discovery）：把「找规律」从 47 条固定规则扩成可搜索的维度空间。

为什么需要它
------------
V5 的规律库是 47 条**手写规则**（号码/遗漏/结构三类），每条规则自带一个固定的
触发条件与动作。这带来两个后果：

1. **维度面窄**：与日期相关的维度、跨期的转移/联动维度、形态序列维度、
   位置与间距维度、冷热的时间尺度维度……都没有覆盖，用户自然觉得「规律不多」。
2. **不可搜索**：规则是「人工挑出来的」，谁也没法回答「这两个维度交叉之后有没有东西」。

本模块提供两件事：

* ``FAMILIES`` —— 一张**维度表**。每个维度给出：中文名、定义函数（as-of，只用
  ``history``）、期望值函数（基线）、以及该维度里可枚举的**参数档位**。
  参数档位是搜索空间——LLM 与 ``discover()`` 要"找规律"，就是在这个空间里挑组合。
* ``asof_scan`` / ``discover`` —— **严格 as-of** 的逐期回放。给定一组「筛选条件」
  （如「近 30 期出现 ≥4 次的号码」「遗漏 ≥ 平均遗漏 1.5 倍的号码」），
  在每一期 t 只用 ``draws[:t]`` 计算候选集合，再看 t 期真实开奖是否命中，
  得到一条逐期 0/1 或配对差值序列 —— 交给 ``regime.regime_report`` 做阶段时效判定。

## 与既有模块的边界

* 不产生最终号码、不参与融合（那是 ``engine``）；
* 不做假说生命周期管理（那是 ``hypothesis``）；
* 本模块只负责「把一种规律表述变成可回放的逐期序列」，并给出阶段画像。
  A/B 实战验证（改出号行为）由 ``evolution`` 负责，LLM 提出新维度由 ``llm_lab`` 负责。

## as-of 的硬约束（避免回测虚高）

``Filter`` 的定义函数签名一律是 ``fn(history) -> set[int]``，**只能看到当期之前的数据**。
所有阈值都必须是历史内部的分位或计数，不许使用全样本常量。实测口径（V6 诊断）：
用全样本 top8 与 as-of top8 相比，平均每期多"命中" 0.031 个红球 —— 这就是事后阈值
凭空制造出来的收益，足以把任何 p 值推到显著边缘。
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import Counter, deque as _deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import regime

R_MAX = 33
B_MAX = 16
PICK = 6
P_RED = PICK / R_MAX          # 6/33
P_BLUE = 1.0 / B_MAX
MEAN_HITS = PICK * P_RED      # 1.0909
SD_HITS = math.sqrt(PICK * P_RED * (1 - P_RED) * (R_MAX - PICK) / (R_MAX - 1))  # 0.8678


# --------------------------------------------------------------------------
# 基础工具：as-of 统计量
# --------------------------------------------------------------------------

def _window(history: Sequence[Dict], w: int) -> List[Dict]:
    h = list(history or [])
    return h[-int(w):] if w and w < len(h) else h


def red_count(history, window: Optional[int] = None) -> np.ndarray:
    """as-of 红球出现次数（长度 33，下标 = 号码-1）。"""
    c = _ctx(history)
    if c is not None:
        return c.red_count(int(window or max(1, c.n)))
    history = list(history or [])
    cnt = np.zeros(R_MAX, dtype=float)
    for d in _window(history, window or len(history) or 1):
        for r in d.get("reds") or []:
            if 1 <= int(r) <= R_MAX:
                cnt[int(r) - 1] += 1.0
    return cnt


def blue_count(history, window: Optional[int] = None) -> np.ndarray:
    c = _ctx(history)
    if c is not None:
        return c.blue_count(int(window or max(1, c.n)))
    history = list(history or [])
    cnt = np.zeros(B_MAX, dtype=float)
    for d in _window(history, window or len(history) or 1):
        b = int(d.get("blue") or 0)
        if 1 <= b <= B_MAX:
            cnt[b - 1] += 1.0
    return cnt


class ScanContext(dict):
    """as-of 增量状态缓存：回放时逐期「前进一期」，而不是每期重算全历史。

    这是把整批扫描从「不可用」压到「交互可用」的关键：遗漏、平均遗漏、连出期数
    这类量在 t → t+1 时只需 O(6) 更新，而 naïve 的每期重算全历史是 O(n) ——
    3515 期 × 91 个组合 = 3×10^5 次全历史扫描，实测 40 秒以上。

    用法（回放循环内）::

        ctx = ScanContext.new(draws[:start])
        for t in range(start, n):
            ctx.history = ...
            chosen = flt["selector"](ctx)     # 选择器优先消费 ctx
            ctx.advance(draws[t])
    """

    USE_CTX = True        # 选择器可判断当前 history 是否是 ScanContext

    @classmethod
    def new(cls, history: Sequence[Dict]) -> "ScanContext":
        ctx = cls()
        ctx["n"] = 0
        ctx["last"] = np.full(R_MAX + 1, -1, dtype=int)     # 下标 1..33
        ctx["gaps_sum"] = np.zeros(R_MAX)
        ctx["gaps_cnt"] = np.zeros(R_MAX, dtype=int)
        ctx["red_rows"] = []
        ctx["blue_rows"] = []
        ctx["issues"] = []
        ctx["dates"] = []
        for d in history or []:
            ctx.advance(d)
        return ctx

    def advance(self, draw: Dict) -> None:
        n = int(self["n"])
        last = self["last"]
        for r in draw.get("reds") or []:
            r = int(r)
            if not (1 <= r <= R_MAX):
                continue
            prev = int(last[r])
            if prev >= 0:
                self["gaps_sum"][r - 1] += float(n - prev - 1)
                self["gaps_cnt"][r - 1] += 1
            last[r] = n
        self["red_rows"].append(tuple(int(r) for r in (draw.get("reds") or [])))
        self["blue_rows"].append(int(draw.get("blue") or 0))
        self["issues"].append(draw.get("issue"))
        self["dates"].append(draw.get("date"))
        self["n"] = n + 1

    # 派生缓存：clone 时不复制（数值上无影响，按需重建比深拷贝 4096 槽的共现表更便宜）
    _DERIVED = ("_slide", "_wd_slide", "_pair_slide", "_pat")

    # ---- V12-01：支持切片，修「形态族 30 档全部 KeyError 被静默吞掉」----
    #
    # 背景（docs/analysis/LEAD_VERDICT.md §1.2）：V11/V12 新增的 8 个形态族选择器
    # 统一写成 ``history[-w:]`` 再遍历 dict 取 ``d.get("reds")``。但 as-of 回放传进来
    # 的是 ``ScanContext``（dict 子类），``ctx[-30:]`` 会走 dict.__getitem__(slice)
    # → ``KeyError: slice(-30, None, None)``；该异常被 asof_scan 的
    # ``except Exception: chosen = set()`` 吞成空集，于是 k≡0、exp≡0，最后被
    # ``discover()`` 的 ``if sd <= 0: continue`` 二次跳过 —— 这 30 档因此
    # **从未被检验**，却让系统对外声称「124 维全搜过」。
    #
    # 修法：切片返回「按时间升序的 draw 字典列表」，语义与 ``list(draws)[a:b]``
    # 完全一致，使选择器无需感知自己拿到的是 list 还是 ScanContext。
    # 只拦 slice，键访问（ctx["n"] 等）行为不变。
    #
    # ⚠️ 性能要点（V12-01b）：**只构造被请求的那一段**，绝不整条历史重建。
    # 初版曾一次性把所有期还原成 dict 并缓存，但 ``advance()`` 每期都会让缓存失效，
    # 于是「30 个形态档 × 3315 期」每次都各重建 3315 条 dict（实测 0.23ms/次
    # × 30 档 × 3315 期 ≈ 23 秒），把 discover() 从 ~1s 拖到 106s，
    # 线上 /api/rules/runtime 一度从毫秒级退化到 3 分 45 秒。
    # 现在按 slice 的 ``indices()`` 只取需要的窗口（典型 w=30/100，开销可忽略）。
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice_rows(key)
        return super().__getitem__(key)

    # 注意：**不要**重写 __len__。
    # tests/test_v6_regime.py::test_scan_context_n_does_not_use_dict_len 明确守护
    # 「len(ctx) 是键数量，不能当历史期数用」——那是 dict 子类的固有语义，改动它
    # 会让 ``len(ctx)==100`` 与 ``ctx.n==100`` 巧合相等，从而掩盖真正的误用。
    # 选择器里若需要期数，请用 ``ctx.n`` 或对切片取 ``len(ctx[-w:])``。
    def _slice_rows(self, sl: slice) -> List[Dict]:
        """按 slice 语义还原**仅该窗口**的 draw 列表（升序，与 list 完全一致）。"""
        rows = self["red_rows"]
        n = len(rows)
        start, stop, step = sl.indices(n)
        idx = range(start, stop, step)
        blues, issues, dates = self["blue_rows"], self["issues"], self["dates"]
        return [{"reds": list(rows[i]), "blue": blues[i],
                 "issue": issues[i], "date": dates[i]}
                for i in idx]

    def _rows_as_draws(self) -> List[Dict]:
        """把**全部**历史还原成与 ``db.load_draws()`` 同形状的 draw 列表（升序）。

        仅供确实需要全量的调用方使用；选择器的 ``history[-w:]`` 走
        ``_slice_rows``，不会走到这里（见上面的性能说明）。
        ``red_rows`` 在 ``new()`` 时空列表、每 ``advance`` 追加一期，与 draws
        一一对应，因此下标可直接对齐，无需另存原始 dict。
        """
        return self._slice_rows(slice(None))

    def clone(self, keep_derived: bool = False) -> "ScanContext":
        """复制当前状态（分支回放用）。

        ``keep_derived=False``（默认）时丢弃派生缓存——调用方会从 ``red_rows``
        重建，结果完全一致（有 150 组 as-of 一致性测试守着）。
        """
        ctx = ScanContext()
        for k, v in self.items():
            if not keep_derived and k in self._DERIVED:
                continue
            ctx[k] = v.copy() if hasattr(v, "copy") else v
        return ctx

    # ---- 派生量 ----
    @property
    def n(self) -> int:
        return int(self["n"])

    def last_seen(self) -> np.ndarray:
        """长度 34（下标 1..33）的最后开出下标；-1 = 从未开出。"""
        return self["last"]

    def omission_arr(self) -> np.ndarray:
        n = self.n
        last = self["last"]
        return np.array([float(n - 1 - last[k]) if last[k] >= 0 else float(n)
                         for k in range(1, R_MAX + 1)])

    def avg_arr(self) -> np.ndarray:
        gs, gc = self["gaps_sum"], self["gaps_cnt"]
        return np.array([gs[k] / gc[k] if gc[k] else float("nan") for k in range(R_MAX)])

    def streak_arr(self) -> np.ndarray:
        out = np.zeros(R_MAX)
        rows = self["red_rows"]
        for k in range(1, R_MAX + 1):
            s = 0
            for row in reversed(rows):
                if k in row:
                    s += 1
                else:
                    break
            out[k - 1] = s
        return out

    # ---- 带窗口/按星期/按形态的增量缓存 ----
    def _ctr(self, key: str, size: int) -> np.ndarray:
        arr = self.get(key)
        if arr is None:
            arr = np.zeros(size)
            self[key] = arr
        return arr

    def weekday_top(self, k: int, window: int = 400) -> set:
        """最近开奖日（星期几）上，**最近 window 期内**同星期几最常开出的 k 个红球。

        为什么用「最近 window 期内同星期几」而不是「全部同星期几」：开奖日固定为
        周二/四/日，若按全历史统计，每条星期几的样本量就是整段历史——等于没有区分
        时间尺度，且与「近期是否生效」的诉求背离。这里与 naive 实现（严格 as-of，
        只取 ``draws[-window:]`` 里同星期几的行）口径一致。
        """
        n = self.n
        if n < 20:
            return set()
        cur = _weekday(self["dates"][-1])
        if cur < 0:
            return set()
        st = self.get("_wd_slide")
        if st is None:
            st = {}
            self["_wd_slide"] = st
        key = ("wd", int(window), cur)
        s2 = st.get(key)
        if s2 is None:
            s2 = {"n": 0, "cnt": np.zeros(R_MAX), "hist": []}
            st[key] = s2
        cnt = s2["cnt"]
        rows, dates = self["red_rows"], self["dates"]
        while s2["n"] < n:
            i = s2["n"]
            if _weekday(dates[i]) == cur:
                for r in rows[i]:
                    cnt[r - 1] += 1.0
                s2["hist"].append(i)
                # 窗口=「最近 window 期开奖」里同星期几的行（不是「最近 window 个同星期几」），
                # 否则 W=400 实际会覆盖 600 期：W=400 时 400 个同星期几的行跨越约 1200 期。
                # 这正是排查中发现的口径漂移，naive 路径是唯一权威口径。
                while s2["hist"] and s2["hist"][0] < i - int(window):
                    old = s2["hist"].pop(0)
                    for r in rows[old]:
                        cnt[r - 1] -= 1.0
            s2["n"] = i + 1
        out = set()
        for idx in np.argsort(-cnt)[:int(k)]:
            out.add(int(idx) + 1)
        return out

    def pair_top(self, window: int, top: int) -> Optional[set]:
        """号码对共现（滑窗增量）：返回共现最多的 top 对中的号码集合。

        维护「最近 window 期」的 2048 槽共现计数：CTX 每前进一期只做
        进 15 对 / 出 15 对，O(1) 摊销。没有该滑窗缓存时返回 None（由调用方全量算）。
        """
        window = int(window)
        slide = self.get("_pair_slide")
        if slide is None:
            slide = {}
            self["_pair_slide"] = slide
        SIZE = 1 << 12
        key = ("pair", window)
        st = slide.get(key)
        if st is None:
            st = {"n": 0, "cnt": [0] * SIZE, "rows": []}
            slide[key] = st
        n = self.n
        cnt = st["cnt"]
        rows = st["rows"]
        while st["n"] < n:
            i = st["n"]
            rr = self["red_rows"][i]
            for x in range(len(rr) - 1):
                a = rr[x]
                for y in range(x + 1, len(rr)):
                    b = rr[y]
                    if a > b:
                        a, b = b, a
                    cnt[(a << 6) | b] += 1
            rows.append(i)
            if len(rows) > window:
                old = rows.pop(0)
                ro = self["red_rows"][old]
                for x in range(len(ro) - 1):
                    a = ro[x]
                    for y in range(x + 1, len(ro)):
                        b = ro[y]
                        if a > b:
                            a, b = b, a
                        cnt[(a << 6) | b] -= 1
            st["n"] = i + 1
        best = heapq.nlargest(int(top), ((c, code) for code, c in enumerate(cnt) if c > 0),
                             key=lambda t: t[0])
        out = set()
        for c, code in best:
            out.add(code >> 6)
            out.add(code & 63)
        return out

    def pattern_top(self, feature: str, k: int, min_samples: int = 20) -> set:
        """形态条件延续：上期结构特征相同的历史期之后，哪些号码最常开出。

        增量缓存：维护「每个特征值 → 号码计数」与「每个特征值 → 样本数」，
        CTX 前进一期时，把**当期**的号码计入「上一期特征值」的桶里（因为那是
        「上一期特征为 v 时下一期开什么」的样本）。整个回放 O(n)。
        """
        n = self.n
        if n < 40:
            return set()
        pats = self.get("_pat")
        if pats is None:
            pats = {}
            self["_pat"] = pats
        cache = pats.get("h|%s" % feature)
        if cache is None:
            cache = {"n": 0, "counts": {}, "vals": []}
            pats["h|%s" % feature] = cache
        if cache["n"] != n:
            # 增量前进：每多一期 i，就把 rows[i] 计入「上一期特征值」的桶，
            # 即「上一期特征为 v 时下一期开出哪些号码」的样本。
            table = _FEATURE_TABLES[feature]
            counts, vals = cache["counts"], cache["vals"]
            rows = self["red_rows"]
            i = cache["n"]
            while i < n:
                vals.append(table(rows[i]))
                if i >= 1:
                    v = vals[i - 1]
                    arr = counts.get(v)
                    if arr is None:
                        arr = np.zeros(R_MAX)
                        counts[v] = arr
                    for r in rows[i]:
                        arr[r - 1] += 1.0
                i += 1
            cache["n"] = n
        counts = cache["counts"]
        vals = cache["vals"]
        cur = vals[-1] if vals else None
        arr = counts.get(cur)
        if arr is None or arr.sum() < min_samples * PICK:
            return set()
        out = set()
        for idx in np.argsort(-arr)[:int(k)]:
            out.add(int(idx) + 1)
        return out

    def _slider(self, window: int):
        """滑窗计数（红/蓝共用）：增量维护「最近 window 期」的每号码计数。

        实现要点：CTX 每前进一期只做 O(len(window)) 的「进一出」，取用时直接读数组。
        朴素写法是每个选择器每期重扫窗口（W=400 时每期 400×6 次加法 → 3515 期
        约 8 亿次操作，实测单个维度就要 1.2 秒），这里把它压到 O(1) 摊销。
        """
        window = int(window)
        slide = self.get("_slide")
        if slide is None:
            slide = {}
            self["_slide"] = slide
        key = ("cnt", window)
        st = slide.get(key)
        if st is None:
            st = {"n": 0, "red": np.zeros(R_MAX), "blue": np.zeros(B_MAX)}
            slide[key] = st
        n = self.n
        while st["n"] < n:
            i = st["n"]
            for r in self["red_rows"][i]:
                st["red"][r - 1] += 1.0
            b = self["blue_rows"][i]
            if 1 <= b <= B_MAX:
                st["blue"][b - 1] += 1.0
            out_i = i - window
            if out_i >= 0:
                for r in self["red_rows"][out_i]:
                    st["red"][r - 1] -= 1.0
                ob = self["blue_rows"][out_i]
                if 1 <= ob <= B_MAX:
                    st["blue"][ob - 1] -= 1.0
            st["n"] = i + 1
        return st["red"], st["blue"]

    def red_count(self, window: int) -> np.ndarray:
        return self._slider(int(window))[0]

    def blue_count(self, window: int) -> np.ndarray:
        return self._slider(int(window))[1]

    def blue_last(self, back: int = 1) -> int:
        rows = self["blue_rows"]
        return int(rows[-back]) if len(rows) >= back else 0


# ---- 兼容层：选择器既可以收到 list[dict]（严格 as-of 语义），也可以收到 ScanContext ----

def _ctx(history) -> Optional[ScanContext]:
    # 不能用 isinstance：``class ScanContext(dict)`` 与内建 ``dict`` 是两种类型。
    return history if isinstance(history, ScanContext) else None


def _last_rows(history, back: int = 1) -> List[int]:
    c = _ctx(history)
    if c is not None:
        rows = c["red_rows"]
        return [int(r) for r in rows[-back]] if len(rows) >= back else []
    return _last_reds(history, back)


def _n_of(history) -> int:
    """当前可用历史期数。

    **注意**：ScanContext 是 dict 的子类，``len(ctx)`` 是键的数量（8 个），
    不是历史期数——朴素写法 ``len(history or [])`` 在 CTX 路径下会得到 8，
    从而触发「样本不足」直接返回空集（实测导致 2 个维度在回放中恒为空，
    悄悄拉低整批扫描的效应）。所有长度判断必须走这里。
    """
    c = _ctx(history)
    return c.n if c is not None else len(history or [])


def omission(history) -> np.ndarray:
    """as-of 当前遗漏（距上次开出已过多少期；从未开出 = 历史期数）。"""
    c = _ctx(history)
    if c is not None:
        return c.omission_arr()
    return ScanContext.new(history).omission_arr()


def omission_avg(history) -> np.ndarray:
    """as-of 平均遗漏（该号码历史上两次开出之间的间隔均值；无历史 → nan）。"""
    c = _ctx(history)
    if c is not None:
        return c.avg_arr()
    return ScanContext.new(history).avg_arr()


def _streak(history) -> np.ndarray:
    """as-of 连出期数（当前连续开出几期）。"""
    c = _ctx(history)
    if c is not None:
        return c.streak_arr()
    return ScanContext.new(history).streak_arr()


def _last_reds(history: Sequence[Dict], back: int = 1) -> List[int]:
    h = list(history or [])
    if len(h) < back:
        return []
    return [int(r) for r in (h[-back].get("reds") or [])]


def _features(history: Sequence[Dict]) -> Dict[str, Any]:
    """一次算齐所有 as-of 派生量（缓存到 history 对象上，避免逐期重复计算）。"""
    reds = [sorted(int(r) for r in (d.get("reds") or [])) for d in history or []]
    sums = [sum(r) for r in reds if len(r) == PICK]
    spans = [(max(r) - min(r)) for r in reds if len(r) == PICK]
    acs = []
    for r in reds:
        if len(r) == PICK:
            diffs = {r[j] - r[i] for i in range(PICK) for j in range(i + 1, PICK)}
            acs.append(len(diffs) - (PICK - 1))
    odd = [sum(1 for x in r if x % 2) for r in reds]
    big = [sum(1 for x in r if x >= 17) for r in reds]
    routes = [tuple(sum(1 for x in r if x % 3 == k) for k in (0, 1, 2)) for r in reds]
    zones = [tuple(sum(1 for x in r if lo <= x <= hi) for lo, hi in ((1, 11), (12, 22), (23, 33)))
             for r in reds]
    head = [min(r) if r else 0 for r in reds]
    tail = [max(r) if r else 0 for r in reds]
    gaps = []
    for r in reds:
        if len(r) == PICK:
            gaps.extend([r[i + 1] - r[i] for i in range(PICK - 1)])
    f = {
        "reds": reds, "sums": sums, "spans": spans, "acs": acs,
        "odd": odd, "big": big, "routes": routes, "zones": zones,
        "head": head, "tail": tail, "gaps": gaps,
        "blue": [int(d.get("blue") or 0) for d in history or []],
        "issues": [d.get("issue") for d in history or []],
        "dates": [d.get("date") for d in history or []],
    }
    return f


def _quantile(seq: Sequence[float], q: float, default: float = 0.0) -> float:
    if not seq:
        return default
    return float(np.quantile(np.asarray(seq, dtype=float), q))


# --------------------------------------------------------------------------
# 维度表：每个维度 = 一个"从 as-of 历史导出号码/属性"的函数 + 参数档位
# --------------------------------------------------------------------------
#
# 约定：每个族给出
#   key/name_zh/dim_zh   维度名
#   ratio                 说明（LLM 读这个来决定要不要用它）
#   params                可枚举的参数档位（搜索空间）
#   red_selector(params) -> fn(history)->set[int]     产出红球候选集
#   blue_selector(params) -> fn(history)->set[int]    可选，产出蓝球候选集
#
# 选择器只做"选出 k 个号码"，命中数 = |候选 ∩ 开奖|，期望 = 6k/33（k 为候选大小，
# 逐期按实际 k 计算，避免"少选号码"被误判为变差）。

def _sel_hot(params: Dict) -> Callable[[Sequence[Dict]], set]:
    w, k = int(params["window"]), int(params["k"])
    def fn(history):
        cnt = red_count(history, w)
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_cold(params: Dict) -> Callable[[Sequence[Dict]], set]:
    w, k = int(params["window"]), int(params["k"])
    def fn(history):
        cnt = red_count(history, w)
        return set((np.argsort(cnt)[:k] + 1).tolist())
    return fn


def _sel_omission_high(params: Dict) -> Callable[[Sequence[Dict]], set]:
    k = int(params["k"])
    def fn(history):
        return set((np.argsort(-omission(history))[:k] + 1).tolist())
    return fn


def _sel_omission_ratio(params: Dict) -> Callable[[Sequence[Dict]], set]:
    k = int(params["k"])
    def fn(history):
        om = omission(history)
        avg = omission_avg(history)
        ratio = np.where(np.isnan(avg) | (avg <= 0), om, om / np.maximum(avg, 0.5))
        return set((np.argsort(-ratio)[:k] + 1).tolist())
    return fn


def _sel_streak(params: Dict) -> Callable[[Sequence[Dict]], set]:
    k = int(params["k"])
    def fn(history):
        return set((np.argsort(-_streak(history))[:k] + 1).tolist())
    return fn


def _sel_last(params: Dict) -> Callable[[Sequence[Dict]], set]:
    back = int(params.get("back", 1))
    def fn(history):
        return set(_last_rows(history, back))
    return fn


def _sel_neighbors(params: Dict) -> Callable[[Sequence[Dict]], set]:
    back = int(params.get("back", 1))
    offset = int(params.get("offset", 1))
    def fn(history):
        base = _last_rows(history, back)
        return {x + offset for x in base if 1 <= x + offset <= R_MAX}
    return fn


def _sel_same_tail(params: Dict) -> Callable[[Sequence[Dict]], set]:
    back = int(params.get("back", 1))
    def fn(history):
        tails = {x % 10 for x in _last_rows(history, back)}
        return {x for x in range(1, R_MAX + 1) if x % 10 in tails}
    return fn


def _sel_same_zone(params: Dict) -> Callable[[Sequence[Dict]], set]:
    back = int(params.get("back", 1))
    zone_id = int(params.get("zone", 0))  # 0=少开的区 1=多开的区 2=固定第一区…
    fixed = params.get("fixed_zone")
    def fn(history):
        if fixed is not None:
            z = int(fixed)
            lo, hi = ((1, 11), (12, 22), (23, 33))[z]
            return set(range(lo, hi + 1))
        c = _ctx(history)
        raw_rows = [list(r) for r in c["red_rows"][-6:]] if c is not None else [
            [int(v) for v in (d.get("reds") or [])] for d in _window(history, 6)]
        zs = [tuple(sum(1 for x in r if lo <= x <= hi) for lo, hi in ((1, 11), (12, 22), (23, 33)))
              for r in raw_rows]
        if not zs:
            return set()
        tot = [sum(z[i] for z in zs) for i in range(3)]
        idx = int(np.argmin(tot)) if zone_id == 0 else int(np.argmax(tot))
        lo, hi = ((1, 11), (12, 22), (23, 33))[idx]
        return set(range(lo, hi + 1))
    return fn


def _sel_link(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """上期开奖的「黄金分割/加减」派生集合（民间玩法，作为维度显式检验）。"""
    back = int(params.get("back", 1))
    mode = str(params.get("mode", "div3"))
    def fn(history):
        base = _last_rows(history, back)
        out = set()
        for x in base:
            if mode == "plus_minus":
                for d in (1, 2, 3, 5):
                    for y in (x - d, x + d):
                        if 1 <= y <= R_MAX:
                            out.add(y)
            elif mode == "div3":
                for y in (x // 3, x // 3 + 11, x // 3 + 22):
                    if 1 <= y <= R_MAX:
                        out.add(y)
            elif mode == "double_mod":
                for y in (x * 2 % 33 + 1, x * 3 % 33 + 1):
                    if 1 <= y <= R_MAX:
                        out.add(y)
        return out
    return fn


def _sel_pair(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """号码对共现：近 W 期共现最多的 top 对中的号码。"""
    w = int(params.get("window", 100))
    top = int(params.get("top", 3))
    SIZE = 1 << 12                    # 号码对编码：a*64+b，a,b ∈ 1..33 → 最大 33*64+33=2145

    def fn(history):
        c = _ctx(history)
        if c is not None:
            st = c.pair_top(w, top)
            if st is not None:
                return st
        rows = c["red_rows"][-w:] if c is not None else [
            tuple(int(r) for r in (d.get("reds") or [])) for d in _window(history, w)]
        cnt = [0] * SIZE
        for rr in rows:
            for i in range(len(rr) - 1):
                a = rr[i]
                for j in range(i + 1, len(rr)):
                    b = rr[j]
                    if a > b:
                        a, b = b, a
                    cnt[(a << 6) | b] += 1
        best = sorted(range(SIZE), key=lambda k: -cnt[k])[:int(top)]
        out = set()
        for code in best:
            if cnt[code] <= 0:
                break
            out.add(code >> 6)
            out.add(code & 63)
        return out
    return fn


_WD_CACHE: Dict[str, int] = {}


def _weekday(iso) -> int:
    key = str(iso)
    v = _WD_CACHE.get(key)
    if v is not None:
        return v
    try:
        import datetime
        y, m, dd = key.split("-")
        v = datetime.date(int(y), int(m), int(dd)).weekday()
    except Exception:  # noqa: BLE001
        v = -1
    _WD_CACHE[key] = v
    return v


def _sel_weekday(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """日期维度：过去在「同一星期几」开得最多的号码。

    开奖日固定是周二/四/日，这条维度检验的是「开奖日 ↔ 号码」这种民间说法。
    as-of 口径：只用历史开奖日期本身的星期，不看未来。
    """
    w = int(params.get("window", 400))
    k = int(params.get("k", 6))
    def fn(history):
        if _n_of(history) < 20:
            return set()
        c = _ctx(history)
        if c is not None:
            return c.weekday_top(k, w)
        # 非增量路径：用「最近 w 期里同星期几的行」统计
        rows = [(d.get("date"), tuple(int(v) for v in (d.get("reds") or [])))
                for d in _window(history, w)]
        if not rows:
            return set()
        cur = _weekday(rows[-1][0])
        if cur < 0:
            return set()
        src = [r for dt, r in rows if _weekday(dt) == cur]
        cnt = np.zeros(R_MAX)
        for row in src:
            for r in row:
                cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_pattern_continuation(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """形态延续：上期某个结构特征（奇偶/大小/路比/和值尾档）在本期按历史条件概率转移。"""
    feature = str(params.get("feature", "odd"))
    k = int(params.get("k", 6))
    table = {
        "odd": lambda r: sum(1 for x in r if x % 2),
        "big": lambda r: sum(1 for x in r if x >= 17),
        "route0": lambda r: sum(1 for x in r if x % 3 == 0),
        "sumtail": lambda r: sum(r) % 10,
        "span": lambda r: (max(r) - min(r)) // 5,
    }
    def fn(history):
        if _n_of(history) < 40:
            return set()
        c = _ctx(history)
        if c is not None:
            return c.pattern_top(feature, k, min_samples=20)
        rows = [sorted(int(v) for v in (d.get("reds") or [])) for d in history or []]
        rows = [r for r in rows if len(r) == PICK]
        if len(rows) < 40:
            return set()
        vals = [table[feature](r) for r in rows]
        cur = vals[-1]
        follow = [rows[i + 1] for i in range(len(rows) - 1) if vals[i] == cur]
        if len(follow) < 20:
            return set()
        cnt = np.zeros(R_MAX)
        for r in follow:
            for x in r:
                cnt[x - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _blue_hot(params: Dict) -> Callable[[Sequence[Dict]], set]:
    w = int(params.get("window", 20))
    def fn(history):
        cnt = blue_count(history, w)
        return {int(np.argmax(cnt)) + 1}
    return fn


def _blue_cold(params: Dict) -> Callable[[Sequence[Dict]], set]:
    w = int(params.get("window", 20))
    def fn(history):
        cnt = blue_count(history, w)
        return {int(np.argmin(cnt)) + 1}
    return fn


def _blue_repeat(params: Dict) -> Callable[[Sequence[Dict]], set]:
    back = int(params.get("back", 1))
    def fn(history):
        c = _ctx(history)
        if c is not None:
            b = c.blue_last(back)
            return {b} if b else set()
        h = list(history or [])
        if len(h) < back:
            return set()
        return {int(h[-back].get("blue") or 0)}
    return fn


def _blue_omission(params: Dict) -> Callable[[Sequence[Dict]], set]:
    def fn(history):
        c = _ctx(history)
        if c is not None:
            rows = c["blue_rows"]
            last = {}
            for i, b in enumerate(rows):
                if 1 <= b <= B_MAX:
                    last[b] = i
            n = len(rows)
            om = {k: n - 1 - last.get(k, -1) for k in range(1, B_MAX + 1)}
        else:
            h = list(history or [])
            last = {}
            for i, d in enumerate(h):
                last[int(d.get("blue") or 0)] = i
            om = {k: len(h) - 1 - last.get(k, -1) for k in range(1, B_MAX + 1)}
        return {max(om, key=lambda k: om[k])}
    return fn


def _blue_neighbor(params: Dict) -> Callable[[Sequence[Dict]], set]:
    offset = int(params.get("offset", 1))
    back = int(params.get("back", 1))
    def fn(history):
        c = _ctx(history)
        if c is not None:
            b = c.blue_last(back)
        else:
            h = list(history or [])
            if len(h) < back:
                return set()
            b = int(h[-back].get("blue") or 0)
        if not b:
            return set()
        return {x for x in (b - offset, b + offset) if 1 <= x <= B_MAX}
    return fn


def _blue_parity(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """蓝球奇偶/大小的轮动维度：上期为奇则本期取偶（或反之）。"""
    mode = str(params.get("mode", "parity"))
    def fn(history):
        c = _ctx(history)
        b = c.blue_last(1) if c is not None else int((list(history or []) or [{}])[-1].get("blue") or 0)
        if not b:
            return set()
        if mode == "parity":
            want = 0 if b % 2 else 1  # 反向
            return {k for k in range(1, B_MAX + 1) if k % 2 == want}
        # 大小：1-8 / 9-16
        small = b <= 8
        return {k for k in range(1, B_MAX + 1) if (k <= 8) != small}
    return fn


_FEATURE_TABLES: Dict[str, Callable[[Sequence[int]], int]] = {
    "odd": lambda r: sum(1 for x in r if x % 2),
    "big": lambda r: sum(1 for x in r if x >= 17),
    "route0": lambda r: sum(1 for x in r if x % 3 == 0),
    "sumtail": lambda r: sum(r) % 10,
    "span": lambda r: (max(r) - min(r)) // 5,
}

# ---- V12 新增：结构 / 形态 / 组合维度（用户 2026-10-18 要求"从各维度发掘"）----

def _sel_sum_range(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """和值区间：近 W 期和值落入 [lo,hi) 的历史期之后，哪些号码最常开出。"""
    w, lo, hi, k = int(params["window"]), int(params.get("lo") or 0),         int(params.get("hi") or 999), int(params.get("k") or 6)
    def fn(history):
        rows = history[-w:]
        n = len(rows)
        if n == 0:
            return set()
        cnt = np.zeros(33)
        for d in rows:
            sm = sum(d.get("reds") or [])
            if lo <= sm < hi:
                for r in d["reds"]:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_span_range(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """跨度档：近 W 期跨度落入 [lo,hi) 的历史期之后，哪些号码最常开出。"""
    w, lo, hi, k = int(params["window"]), int(params.get("lo") or 0),         int(params.get("hi") or 999), int(params.get("k") or 6)
    def fn(history):
        rows = history[-w:]
        cnt = np.zeros(33)
        for d in rows:
            rs = d.get("reds") or []
            if not rs:
                continue
            sp = max(rs) - min(rs)
            if lo <= sp < hi:
                for r in rs:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_parity_ratio(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """奇偶配置：近 W 期奇数个数=o 的历史期之后，选高频号码。"""
    w, o, k = int(params["window"]), int(params.get("odd") or 3), int(params.get("k") or 6)
    def fn(history):
        cnt = np.zeros(33)
        for d in history[-w:]:
            rs = d.get("reds") or []
            n_odd = sum(1 for r in rs if r % 2 == 1)
            if n_odd == o:
                for r in rs:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_big_small(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """大小配置：近 W 期大号(>16)个数=b 的历史期之后，选高频号码。"""
    w, b, k = int(params["window"]), int(params.get("big") or 3), int(params.get("k") or 6)
    def fn(history):
        cnt = np.zeros(33)
        for d in history[-w:]:
            rs = d.get("reds") or []
            n_big = sum(1 for r in rs if r > 16)
            if n_big == b:
                for r in rs:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_route012(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """012路配置：近 W 期 0路个数=z0 的历史期之后，选高频号码。"""
    w, z0, k = int(params["window"]), int(params.get("route0") or 2), int(params.get("k") or 6)
    def fn(history):
        cnt = np.zeros(33)
        for d in history[-w:]:
            rs = d.get("reds") or []
            n0 = sum(1 for r in rs if r % 3 == 0)
            if n0 == z0:
                for r in rs:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_consecutive(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """连号：近 W 期出现连号对(x,x+1)中的号码，选高频的 k 个。"""
    w, k = int(params["window"]), int(params.get("k") or 6)
    def fn(history):
        cnt = np.zeros(33)
        for d in history[-w:]:
            rs = sorted(d.get("reds") or [])
            for i in range(len(rs) - 1):
                if rs[i + 1] == rs[i] + 1:
                    cnt[rs[i] - 1] += 1
                    cnt[rs[i + 1] - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_ac_value(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """AC值（复杂度）：近 W 期 AC 值=av 的历史期之后，选高频号码。"""
    w, av, k = int(params["window"]), int(params.get("ac") or 8), int(params.get("k") or 6)
    def fn(history):
        cnt = np.zeros(33)
        for d in history[-w:]:
            rs = sorted(d.get("reds") or [])
            diffs = {rs[j] - rs[i] for i in range(len(rs)) for j in range(i + 1, len(rs))}
            ac = len(diffs) - (len(rs) - 1)
            if ac == av:
                for r in rs:
                    cnt[r - 1] += 1
        return set((np.argsort(-cnt)[:k] + 1).tolist())
    return fn


def _sel_hot_cold_mix(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """冷热混合：热号 top(a) ∪ 冷号 top(b)，检验"一热配一冷"民间说法。"""
    w, a, b = int(params["window"]), int(params.get("hot") or 3), int(params.get("cold") or 3)
    def fn(history):
        cnt = red_count(history, w)
        hot = set((np.argsort(-cnt)[:a] + 1).tolist())
        cold = set((np.argsort(cnt)[:b] + 1).tolist())
        return hot | cold
    return fn


def _sel_same_tail_group(params: Dict) -> Callable[[Sequence[Dict]], set]:
    """尾数连出：近 W 期各尾数出现次数，选最热的 k 个尾数对应的号码。"""
    w, k = int(params["window"]), int(params.get("k") or 5)
    def fn(history):
        tail_cnt = np.zeros(10)
        for d in history[-w:]:
            for r in d.get("reds") or []:
                tail_cnt[r % 10] += 1
        top_tails = set((np.argsort(-tail_cnt)[:k]).tolist())
        return {r for r in range(1, 34) if r % 10 in top_tails}
    return fn


# ==========================================================================
# S7 分布类维度的逐期统计量（stat_fn(history, actual) -> (obs, exp)）
#
# 约定：``obs - exp`` 的长期均值为 0 即「该维度无效应」。逐期只用 history（严格 as-of），
# 未来期不会进入。任何缺失/异常都返回 (0.0, 0.0) 而不是抛错 —— 单期失败不该中断回放。
# ==========================================================================

def _stat_order_pos(params: Dict):
    """位置-号码独立性（as-of 选号 + 下一期命中，零假设下严格零均值）。

    对每个开奖位置 p（第 1..6 位），用近 ``window`` 期历史统计「该位置各号码出现次数」，
    取该位置出现最多的 ``k`` 个号码作为「该位置偏爱集」。
    * ``obs`` = 本期 6 个位置上，实际号码落在**各自位置**偏爱集内的个数（0..6）。
    * ``exp`` = 零假设（位置与号码独立、号码均匀）下的期望 = ``6·k/33``。

    独立时每个位置的偏爱集与本期号码无关，故 ``P(命中)=k/33``、``E[obs]=6k/33`` ——
    ``obs-exp`` 的**长期均值严格为 0**（不会像互信息那样因有限历史 / 稀疏列联表
    产生系统偏置，实测插值 PMI 在纯随机数据上 z 高达数十，是假阳性）。
    存在**位置偏置**（某位置的号码分布明显偏离均匀）时，该位置的偏爱集集中到少数号，
    本期命中率随之升高 → ``obs>exp``，可检出。

    ``order`` 缺失/长度异常返回 (0,0) 优雅降级；历史 < 30 期也返回 (0,0)。

    **性能**（V15-b）：位置计数用**滑动窗口差分**维护（进窗 +1、出窗 -1），不再每期重扫
    最近 ``w`` 期。原实现每期重建 ``(6, 34)`` 并遍历 300 期 × 6 位置，单族占
    ``discover()`` 全时约 9.4s（总 26s）。差分后与逐期重扫**数值完全等价**（同一窗口、
    同一计数），只是把 O(w·6) 降到 O(6)。
    """
    w = int(params.get("window", 300))
    k = int(params.get("k", 6))
    exp0 = PICK * k / R_MAX
    # 滑动窗口状态（闭包内，单次扫描独占一个 selector 实例）
    _pos_cnt = np.zeros((PICK, R_MAX + 1))
    _buf: "deque[Optional[tuple]]" = _deque()   # 已进窗的 order（None = 该期无有效 order）
    _n_hist = 0
    _last_len: Optional[int] = None       # 上次调用时 history 的长度（滑窗连续性判据）

    def _add(order: Optional[tuple], sign: int) -> int:
        """把一期的 order 计入/移出窗口计数，返回对 ``_n_hist`` 的增量。"""
        if not order:
            return 0
        for p in range(PICK):
            num = int(order[p])
            if 1 <= num <= R_MAX:
                _pos_cnt[p, num] += sign
        return sign

    def fn(history, actual):
        nonlocal _n_hist, _buf, _last_len
        order = actual.get("order")
        if not order or len(order) != PICK:
            return 0.0, 0.0
        # **O(1) 滑窗**：as-of 回放每期把上一期 actual 追加进 history，故
        # ``history[-1]`` 就是「上一次调用时的本期」。用「上次见到的期末位置」判断
        # 这一期该滑进哪一条、该滑出哪一条，全程不重建窗口列表。
        # 首次/非顺序调用（长度不连续）时退化为一次性重建，保证与「每期重扫
        # history[-w:]」严格等价。
        hl = len(history)
        if _last_len is None or hl != _last_len + 1:
            _pos_cnt[:] = 0.0
            _buf = _deque()
            _n_hist = 0
            for d in history[-w:]:
                o = d.get("order")
                tu = (tuple(int(x) for x in o) if (o and len(o) == PICK) else None)
                _buf.append(tu)
                _n_hist += _add(tu, +1)
        else:
            # 滑进一条：上一次的本期 = history[-1]
            d = history[-1]
            o = d.get("order")
            tu = (tuple(int(x) for x in o) if (o and len(o) == PICK) else None)
            _buf.append(tu)
            _n_hist += _add(tu, +1)
            # 滑出一条：窗口超过 w 时丢掉最老的一条
            while len(_buf) > w:
                _n_hist += _add(_buf.popleft(), -1)
        _last_len = hl
        if _n_hist < 30:
            return 0.0, 0.0
        hits = 0
        for p in range(PICK):
            num = int(order[p])
            if not (1 <= num <= R_MAX):
                continue
            fav = set((np.argsort(-_pos_cnt[p, 1:])[:k] + 1).tolist())
            if num in fav:
                hits += 1
        return float(hits), float(exp0)
    return fn


def _stat_number_bias(params: Dict):
    """逐号偏置：近 W 期最超配的单个号码（二项 z 最大）下一期是否开出。

    以 ``PICK/33≈0.1818`` 为单号每期基线概率。按近 W 期的超配 z 选出一个号码，
    若它在本期开出观测为 1、否则 0；期望为它在近 W 期的基线概率（与基线一致时
    效应为 0）。多重性：本族本身即「对 33 个号取最强者」，Bonferroni 按 33 计
    （见 note_zh 与现实期数门槛）。历史不足 W/2 或号码数异常时返回 (0, 0)。

    **性能**（V15-b）：窗口内的逐号计数用**滑动窗口**维护（进窗 +1、出窗 -1），
    不再每期调 ``red_count(hist, len(hist))`` 全量重扫。原实现单族占 ``discover()``
    约 5.7s。与逐期重扫数值严格等价（同一窗口、同一计数）。
    """
    w = int(params.get("window", 300))
    p_base = PICK / R_MAX
    _min_n = max(30, w // 2)
    _cnt = np.zeros(R_MAX + 1)
    _buf: "deque[tuple]" = _deque()
    _last_len: Optional[int] = None

    def fn(history, actual):
        nonlocal _last_len, _buf
        # **O(1) 滑窗**（与 ``_stat_order_pos`` 同一套机制）：as-of 回放每期追加一条，
        # ``history[-1]`` 即「上次调用时的本期」。用长度连续性判断滑窗，不重建窗口。
        hl = len(history)
        if _last_len is None or hl != _last_len + 1:
            _cnt[:] = 0.0
            _buf = _deque()
            for d in history[-w:]:
                t = tuple(sorted(int(x) for x in (d.get("reds") or [])))
                _buf.append(t)
                for x in t:
                    if 1 <= x <= R_MAX:
                        _cnt[x] += 1
        else:
            t = tuple(sorted(int(x) for x in ((history[-1].get("reds")) or [])))
            _buf.append(t)
            for x in t:
                if 1 <= x <= R_MAX:
                    _cnt[x] += 1
            while len(_buf) > w:
                for x in _buf.popleft():
                    if 1 <= x <= R_MAX:
                        _cnt[x] -= 1
        _last_len = hl
        n = len(_buf)
        if n < _min_n:
            return 0.0, 0.0
        # 二项 z（相对 p_base）：(obs - n*p) / sqrt(n*p*(1-p))
        sd = math.sqrt(max(n * p_base * (1 - p_base), 1e-9))
        z = (_cnt[1:] - n * p_base) / sd
        pick = int(np.argmax(z)) + 1                    # 最超配的号（下标 0..32 → 1..33）
        reds = set(int(x) for x in (actual.get("reds") or []))
        obs = 1.0 if pick in reds else 0.0
        # 期望：**无条件基线** PICK/R_MAX（而非该号的近窗频率）。
        # 若用近窗频率作期望，等于把"超配"本身当成了零假设目标 —— 最超配号的近窗
        # 频率必然 > 基线，下一期回归到基线时会产生系统性负偏差（实测 z≈-5 的假阳）。
        # 正确口径：检验"最超配的号下一期是否仍高于无条件基线" —— 零假设下 E[obs]=基线。
        exp = p_base
        return obs, exp
    return fn


def _stat_sum_dist(params: Dict):
    """和值分布：逐期取和值在 C(33,6) 精确零分布下的 -log p，与期望熵之差。

    ``obs = -log P(sum_t)``，``exp = E[-log P(S)] = 熵``（由精确分布算出，缓存）。
    和值分布与理论一致时，``E[obs] = exp``（都等于熵），故长期均值 0。
    系统性偏离（某些和值段过多/过少）会抬高/降低 ``obs``。

    **性能**（V15-b）：零分布与熵在整个回放过程中**完全不变**，必须在闭包构造时算一次，
    不能每期重算 —— 曾经在 ``fn`` 里每期做 ``counts.astype(float)`` + 除法 + 163 元素 ``log``
    求和，让 ``discover()`` 从 15.2s 涨到 24.7s（+63%）。零分布本身由 ``rigor`` 侧缓存，
    这里只需把**派生量**（每和值的 ``-log p`` 与熵）预先算好。
    """
    from . import rigor
    sums, counts, total = rigor.sum_distribution()
    probs = counts.astype(float) / float(total)
    with np.errstate(divide="ignore"):
        neg_log_p = -np.log(probs)                     # 每和值的 -log P(S)
    entropy = float(-(probs * np.log(probs)).sum())    # E[-log P(S)] 精确值

    def fn(history, actual):
        reds = actual.get("reds")
        if not reds or len(reds) != PICK:
            return 0.0, 0.0
        s = int(sum(int(x) for x in reds))
        idx = int(np.searchsorted(sums, s))
        if idx >= len(sums) or int(sums[idx]) != s:
            return 0.0, 0.0
        v = float(neg_log_p[idx])
        if not math.isfinite(v):
            return 0.0, 0.0
        return v, entropy
    return fn


FAMILIES: List[Dict] = [
    # ---- S7 新增：discover 此前未覆盖的维度（P2 §7.1 S7）----
    # 这三个维度**不是**「选号集 ∩ 开奖」型，而是分布检验型：用 ``stat_fn`` 逐期产出
    # 一对 ``(obs, exp)``，交给同一套 ``delta_series → regime_report`` 阶段机。
    # 它们的价值是「换一个正确的提问方式」，不是「立刻出规律」——见各条 note_zh。
    {"key": "order_pos", "name_zh": "位置-号码独立性", "dim_zh": "开奖位置 × 号码（6×33 列联）",
     "ratio": ("每个开奖位置（第1..6位）的号码分布是否独立于位置；"
               "逐期用位置-号码互信息（PMI），独立时长期均值为 0"),
     "target": "red", "stat": _stat_order_pos,
     "note_zh": ("P2 §2.2 核心缺口：`order` 字段此前 0 处被 discover 读取。"
                 "这是检验「号码出现的位置是否携带信息」，df=(6-1)(33-1)=160。"
                 "缺失 order 的期次按 0 效应处理（优雅降级）。"),
     "params": [{"window": w, "k": k} for w in (100, 300, 600) for k in (6, 10)]},
    {"key": "number_bias", "name_zh": "逐号偏置", "dim_zh": "单号二项检验（Bonferroni 33）",
     "ratio": ("近 W 期最超配的单个号码（按二项 z 最大）下一期是否开出；"
               "等价于对 33 个单号各做一次二项检验再取最强者，多重性按 33 计"),
     "target": "red", "stat": _stat_number_bias,
     "note_zh": ("诚实结论（P2）：单号 σ=0.4536，+3% 需约 40.7 万期、+10% 需约 3.7 万期"
                 "才可能检出。这条的价值是「换一个正确的提问方式」，"
                 "不是「立刻就出规律」——历史再长也不够。"),
     "params": [{"window": w} for w in (100, 300, 600)]},
    {"key": "sum_dist", "name_zh": "和值分布", "dim_zh": "红球和值分箱 χ²",
     "ratio": ("开奖和值是否偏离 C(33,6) 精确分布；逐期取和值在精确零分布下的"
               "-log p 与期望熵的差，长期均值为 0 表示和值分布与理论一致"),
     "target": "red", "stat": _stat_sum_dist,
     "note_zh": ("零分布用 rigor.sum_distribution() 的 C(33,6)=1_107_568 全枚举精确分布，"
                 "不自行拍参数。这是「和值整体分布是否偏移」的检验，"
                 "不是「某期和值落在哪」的预测。"),
     "params": [{}]},
    # ---- 红球：频率维度（时间尺度） ----
    {"key": "hot", "name_zh": "热度延续", "dim_zh": "红球频率(热)",
     "ratio": "近 W 期出现最多的 k 个红球，下一期是否延续",
     "target": "red", "selector": _sel_hot,
     "params": [{"window": w, "k": k} for w in (10, 20, 30, 50, 100, 200) for k in (5, 6, 8)]},
    {"key": "cold", "name_zh": "冷号回补", "dim_zh": "红球频率(冷)",
     "ratio": "近 W 期出现最少的 k 个红球，下一期是否补出",
     "target": "red", "selector": _sel_cold,
     "params": [{"window": w, "k": k} for w in (10, 20, 30, 50, 100, 200) for k in (5, 6, 8)]},
    # ---- 红球：遗漏维度 ----
    {"key": "omission_high", "name_zh": "遗漏最久优先", "dim_zh": "红球遗漏",
     "ratio": "当前遗漏最长的 k 个红球是否即将开出",
     "target": "red", "selector": _sel_omission_high,
     "params": [{"k": k} for k in (4, 6, 8, 10)]},
    {"key": "omission_ratio", "name_zh": "遗漏回补比", "dim_zh": "红球遗漏/平均遗漏",
     "ratio": "当前遗漏 ÷ 自身平均遗漏 最大的 k 个红球",
     "target": "red", "selector": _sel_omission_ratio,
     "params": [{"k": k} for k in (4, 6, 8, 10)]},
    {"key": "streak", "name_zh": "连出号", "dim_zh": "红球连出期数",
     "ratio": "连续开出期数最多的 k 个红球（超热）",
     "target": "red", "selector": _sel_streak,
     "params": [{"k": k} for k in (2, 3, 4)]},
    # ---- 红球：跨期联动维度 ----
    {"key": "repeat", "name_zh": "重号", "dim_zh": "上期号码复出",
     "ratio": "上期 6 个红球在本期复出的个数",
     "target": "red", "selector": _sel_last, "params": [{"back": 1}]},
    {"key": "repeat2", "name_zh": "隔期复出", "dim_zh": "上上期号码复出",
     "ratio": "上上期红球在本期复出（隔期）",
     "target": "red", "selector": _sel_last, "params": [{"back": 2}]},
    {"key": "neighbor", "name_zh": "邻号", "dim_zh": "上期号码±d",
     "ratio": "上期红球 ±d 的邻号在本期出现",
     "target": "red", "selector": _sel_neighbors,
     "params": [{"back": b, "offset": o} for b in (1, 2) for o in (1, 2, 3)]},
    {"key": "same_tail", "name_zh": "同尾号", "dim_zh": "尾数联动",
     "ratio": "与上期号码同尾数（个位相同）的号码是否在本期出现",
     "target": "red", "selector": _sel_same_tail, "params": [{"back": 1}, {"back": 2}]},
    {"key": "zone_lag", "name_zh": "区间回补", "dim_zh": "三区分布",
     "ratio": "近 6 期开得最少的区（或最多）的 11 个号码是否回补",
     "target": "red", "selector": _sel_same_zone, "params": [{"zone": 0}, {"zone": 1}]},
    {"key": "link", "name_zh": "派生号（除3/倍数）", "dim_zh": "上期派生集合",
     "ratio": "由上期号码派生的 除3区间 / ±差 / 倍数 集合",
     "target": "red", "selector": _sel_link,
     "params": [{"mode": m, "back": b} for m in ("div3", "plus_minus", "double_mod") for b in (1, 2)]},
    {"key": "pair", "name_zh": "共现号码对", "dim_zh": "号码对共现",
     "ratio": "近 W 期共现次数最多的 top 对中的号码",
     "target": "red", "selector": _sel_pair,
     "params": [{"window": w, "top": t} for w in (100, 300) for t in (3, 6)]},
    # ---- 日期维度 ----
    {"key": "weekday", "name_zh": "开奖日维度", "dim_zh": "星期几 → 号码",
     "ratio": "历史上同一星期几开得最多的 k 个号码（检验开奖日是否影响分布）",
     "target": "red", "selector": _sel_weekday,
     "params": [{"window": w, "k": 6} for w in (400, 1000)]},
    # ---- 形态序列维度 ----
    {"key": "pattern_cont", "name_zh": "形态条件延续", "dim_zh": "结构特征 → 条件分布",
     "ratio": "上期结构特征（奇偶/大小/0路/和值尾/跨度档）相同的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_pattern_continuation,
     "params": [{"feature": f, "k": 6} for f in ("odd", "big", "route0", "sumtail", "span")]},
    # ---- 蓝球维度 ----
    {"key": "blue_hot", "name_zh": "蓝球热号", "dim_zh": "蓝球频率(热)",
     "ratio": "近 W 期最热的蓝球延续", "target": "blue", "selector": _blue_hot,
     "params": [{"window": w} for w in (10, 20, 50, 100)]},
    {"key": "blue_cold", "name_zh": "蓝球冷号", "dim_zh": "蓝球频率(冷)",
     "ratio": "近 W 期最冷的蓝球回补", "target": "blue", "selector": _blue_cold,
     "params": [{"window": w} for w in (10, 20, 50, 100)]},
    {"key": "blue_repeat", "name_zh": "蓝球重号", "dim_zh": "蓝球上期复出",
     "ratio": "上期蓝球本期再次出现", "target": "blue", "selector": _blue_repeat,
     "params": [{"back": 1}, {"back": 2}]},
    {"key": "blue_omission", "name_zh": "蓝球遗漏最久", "dim_zh": "蓝球遗漏",
     "ratio": "遗漏最久的蓝球开出", "target": "blue", "selector": _blue_omission,
     "params": [{}]},
    {"key": "blue_neighbor", "name_zh": "蓝球邻号", "dim_zh": "蓝球±d",
     "ratio": "上期蓝球 ±d", "target": "blue", "selector": _blue_neighbor,
     "params": [{"offset": 1}, {"offset": 2}]},
    {"key": "blue_parity", "name_zh": "蓝球奇偶/大小轮动", "dim_zh": "蓝球形态轮动",
     "ratio": "上期蓝球奇则本期选偶（或大小反向）", "target": "blue", "selector": _blue_parity,
     "params": [{"mode": "parity"}, {"mode": "size"}]},

    # ---- V12 新增：结构 / 形态 / 组合维度（用户 2026-10-18 要求"从各维度发掘"）----
    {"key": "sum_range", "name_zh": "和值区间条件", "dim_zh": "和值区间",
     "ratio": "近 W 期和值处于某区间的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_sum_range,
     "params": [{"window": 30, "lo": 45, "hi": 75},
                {"window": 30, "lo": 75, "hi": 105},
                {"window": 30, "lo": 105, "hi": 135},
                {"window": 30, "lo": 135, "hi": 165},
                {"window": 100, "lo": 75, "hi": 105},
                {"window": 100, "lo": 105, "hi": 135}]},
    {"key": "span_range", "name_zh": "跨度档条件", "dim_zh": "跨度档",
     "ratio": "近 W 期跨度为某范围的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_span_range,
     "params": [{"window": 30, "lo": 0, "hi": 15},
                {"window": 30, "lo": 15, "hi": 25},
                {"window": 30, "lo": 25, "hi": 33},
                {"window": 100, "lo": 10, "hi": 25}]},
    {"key": "parity_cfg", "name_zh": "奇偶个数条件", "dim_zh": "奇偶配置",
     "ratio": "近 W 期奇数个数为 o 的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_parity_ratio,
     "params": [{"window": 30, "odd": 2}, {"window": 30, "odd": 3},
                {"window": 30, "odd": 4}, {"window": 100, "odd": 3}]},
    {"key": "big_small_cfg", "name_zh": "大小个数条件", "dim_zh": "大小配置",
     "ratio": "近 W 期大号个数为 b 的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_big_small,
     "params": [{"window": 30, "big": 2}, {"window": 30, "big": 3},
                {"window": 30, "big": 4}, {"window": 100, "big": 3}]},
    {"key": "route012_cfg", "name_zh": "012路个数条件", "dim_zh": "012路配置",
     "ratio": "近 W 期 0路个数为 z 的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_route012,
     "params": [{"window": 30, "route0": 1}, {"window": 30, "route0": 2},
                {"window": 30, "route0": 3}, {"window": 100, "route0": 2}]},
    {"key": "consecutive_hot", "name_zh": "连号热码", "dim_zh": "连号",
     "ratio": "近 W 期出现过连号对的号码中，最常开的 k 个",
     "target": "red", "selector": _sel_consecutive,
     "params": [{"window": 30, "k": 6}, {"window": 100, "k": 6}]},
    {"key": "ac_value", "name_zh": "AC值条件", "dim_zh": "AC值",
     "ratio": "近 W 期 AC 值为某档的历史期之后，哪些号码最常开出",
     "target": "red", "selector": _sel_ac_value,
     "params": [{"window": 100, "ac": 7}, {"window": 100, "ac": 8},
                {"window": 100, "ac": 9}, {"window": 100, "ac": 10}]},
    {"key": "hot_cold_mix", "name_zh": "冷热混合", "dim_zh": "冷热混合",
     "ratio": "热号 top(a) ∪ 冷号 top(b)（民间「一热配一冷」说法）",
     "target": "red", "selector": _sel_hot_cold_mix,
     "params": [{"window": 30, "hot": 3, "cold": 3},
                {"window": 100, "hot": 3, "cold": 3},
                {"window": 30, "hot": 4, "cold": 2}]},
    {"key": "tail_group", "name_zh": "尾数连出", "dim_zh": "尾数",
     "ratio": "近 W 期最热的 k 个尾数对应的所有号码",
     "target": "red", "selector": _sel_same_tail_group,
     "params": [{"window": 30, "k": 5}, {"window": 100, "k": 5}]},
]

FAMILY_BY_KEY: Dict[str, Dict] = {f["key"]: f for f in FAMILIES}


def family_catalog() -> List[Dict]:
    """维度表（不含函数对象，可直接 JSON 化给 LLM / 前端）。"""
    return [
        {"key": f["key"], "name_zh": f["name_zh"], "dim_zh": f["dim_zh"],
         "ratio": f["ratio"], "target": f["target"],
         "n_param_sets": len(f["params"]),
         # note_zh：诚实结论（如 number_bias 的「+3% 需 40.7 万期」）。前端维度表
         # 必须把它显示出来，避免「能搜到维度」被误读成「立刻有规律」。
         "note_zh": f.get("note_zh"),
         "params": f["params"]}
        for f in FAMILIES
    ]


def combo_space_size() -> int:
    """可枚举的「维度 × 参数」单因子空间大小（规律搜索规模的直观指标）。"""
    return sum(len(f["params"]) for f in FAMILIES)


# --------------------------------------------------------------------------
# 单因子 / 组合筛选条件
# --------------------------------------------------------------------------

def make_filter(key: str, params: Optional[Dict] = None) -> Dict:
    """把（族, 参数）实例化成一个可回放的筛选条件。"""
    fam = FAMILY_BY_KEY.get(key)
    if not fam:
        raise KeyError(f"未知维度族: {key}")
    params = dict(params or (fam["params"][0] if fam["params"] else {}))
    out = {
        "key": key, "name_zh": fam["name_zh"], "dim_zh": fam["dim_zh"],
        "target": fam["target"], "params": params,
        "label": f"{fam['name_zh']}{params or ''}",
    }
    if fam.get("stat") is not None:
        # S7 分布类维度：没有「选号集」，用一个逐期统计量代替 selector。
        # asof_scan 检测到 stat_fn 会切到 _stat_scan（见该函数 docstring）。
        out["stat_fn"] = fam["stat"](params)
        out["ratio"] = fam.get("ratio")
        out["note_zh"] = fam.get("note_zh")
    else:
        out["selector"] = fam["selector"](params)
    return out


def complement_filter(flt: Dict) -> Dict:
    """返回一个筛选条件的**补集**（V12-21 补集反转，裁决 R-8）。

    数学前提（`docs/analysis/P2_STATS_WHY_ZERO.md` §4.2b，Lead 与 P6 各自实测）：

        hits(S) + hits(complement(S)) ≡ 6     （红球，每期 6 个号必落其内）
        delta_S + delta_{comp(S)} ≡ 0          （``X_S + X_comp = 6`` 恒等式）

    也就是说，**一个「显著为负」的过滤器，其补集就是「显著为正」的过滤器**。
    而历史实现把 ``delta < 0`` 判为「反对」→ ``status="rejected"``（见
    ``hypothesis.resolve_registry``），等于把一半信号当失败销毁。

    关键：补集与原本是**同一次检验的两种写法**，不是新的一次检验。因此
    调用方**不得**把补集计入 ``n_tested``（FWER 不变）；返回体里的
    ``complement_of`` 就是给调用方的显式标记。

    蓝色同理（``blue`` 域用 1..16 的补集）。
    """
    if not flt:
        return {}
    target = flt.get("target") or "red"
    universe = set(range(1, (B_MAX if target == "blue" else R_MAX) + 1))
    inner = flt.get("selector")

    def _comp_selector(history):
        seen = set(int(x) for x in (inner(history) if inner else ()) or ())
        return universe - seen

    out = dict(flt)
    out["selector"] = _comp_selector
    out["complement_of"] = flt.get("label") or flt.get("key")
    out["is_complement"] = True
    base_zh = flt.get("name_zh") or flt.get("label") or "筛选"
    out["name_zh"] = f"〔补集〕{base_zh}"
    out["label"] = f"〔补集〕{flt.get('label') or base_zh}"
    return out


def complement_of_filter(flt: Dict) -> Optional[Dict]:
    """若该筛选条件是一次显著的**负**效应，返回其补集；否则 None。

    判定用 ``delta``（本期命中均值 − 期望）。``delta >= 0`` 时补集只会更差，
    没有登记价值，返回 None。
    """
    if not flt:
        return None
    delta = flt.get("delta")
    if delta is None:
        return None
    try:
        if float(delta) >= 0:
            return None
    except (TypeError, ValueError):
        return None
    return complement_filter(flt)


def combine(filters: Sequence[Dict], mode: str = "union") -> Dict:
    """把多个筛选条件合成一个（交集 / 并集）——这是「跨维度交叉检验」的入口。

    只支持同 target（红/蓝）之间的组合；跨 target 组合无意义（红蓝独立开奖）。
    """
    fs = [f for f in filters if f]
    if not fs:
        raise ValueError("至少需要一个筛选条件")
    targets = {f["target"] for f in fs}
    if len(targets) > 1:
        raise ValueError("不能跨红/蓝组合筛选条件")
    mode = "intersection" if mode not in ("union", "intersection") else mode
    def fn(history):
        sets = [f["selector"](history) for f in fs]
        if mode == "intersection":
            out = set(sets[0])
            for s in sets[1:]:
                out &= set(s)
            return out
        out: set = set()
        for s in sets:
            out |= set(s)
        return out
    label = (" ∩ ".join(f["name_zh"] for f in fs) if mode == "intersection"
             else " ∪ ".join(f["name_zh"] for f in fs))
    return {"key": "+".join(f["key"] for f in fs), "name_zh": label,
            "target": fs[0]["target"], "params": {"mode": mode}, "selector": fn,
            "label": label, "components": [f["key"] for f in fs], "mode": mode}


# --------------------------------------------------------------------------
# 严格 as-of 回放
# --------------------------------------------------------------------------

def _stat_scan(draws: Sequence[Dict], flt: Dict, start: int, end: int,
               ctx: Optional["ScanContext"] = None) -> Dict:
    """**分布类维度的 as-of 回放**（S7）：不走「选号集 ∩ 开奖」，而是逐期算一个
    ``(obs, exp)`` 配对统计量，仍然交给同一套 ``delta_series → regime_report`` 阶段机。

    ``flt["stat_fn"](ctx, actual) -> (obs, exp)``：``ctx`` 是**只看��� start 期之前**
    历史（严格 as-of，不许泄漏未来），``actual`` 是第 t 期开奖。返回 ``(obs, exp)``
    两个实数，``obs-exp`` 的长期均值就是该维度的效应量。

    返回结构与 ``asof_scan`` 对齐（``obs``/``exp``/``k_series``/``n``/``start``/``end``），
    下游 ``pad_series`` / ``regime_of_filter`` 无需区分两种来源。
    """
    draws = list(draws or [])
    end = len(draws) if end is None else min(int(end), len(draws))
    start = max(0, int(start))
    stat_fn = flt["stat_fn"]
    obs: List[float] = []
    exp: List[float] = []
    ks: List[int] = []
    series: List[Dict] = []
    # 严格 as-of：stat_fn 只拿到 draws[:t] 这段历史（`hist`）与当期开奖 `actual`，
    # 未来期绝不会进入 hist —— 与 selector 的 CTX 语义一致（N6：不泄漏未来）。
    hist: List[Dict] = list(draws[:start])
    for t in range(start, end):
        actual = draws[t]
        try:
            o, e = stat_fn(hist, actual)
            o = float(o) if o is not None else 0.0
            e = float(e) if e is not None else 0.0
        except Exception:  # noqa: BLE001
            o, e = 1.0, 0.0
        obs.append(o)
        exp.append(e)
        ks.append(1)
        series.append({"issue": actual.get("issue"), "obs": round(o, 4),
                       "expected": round(e, 4)})
        hist.append(actual)
    return {"target": flt.get("target", "red"), "n": len(obs), "obs": obs, "exp": exp,
            "k_series": ks, "k_mean": 1.0, "k_active_mean": 1.0,
            "k_active_ratio": 1.0 if ks else 0.0,
            "hit_series": series[-200:], "start": start, "end": end}


def asof_scan(draws: Sequence[Dict], flt: Dict, start: int = 200,
              end: Optional[int] = None, ctx: Optional[ScanContext] = None) -> Dict:
    """逐期 as-of 回放一个筛选条件，返回「每期观测序列」。

    对 red target：``obs[i] = |筛选出的红球 ∩ 第 i 期开奖|``，``exp[i] = 6*|k_i|/33``；
    对 blue target：``obs[i] = 1{筛选出的蓝球包含第 i 期蓝球}``，``exp[i] = |k_i|/16``。

    返回 ``{target, n, obs, exp, k_series, k_mean, k_active_mean, hit_series,
    start, end}``；``obs``/``exp`` 长度与 ``draws[start:end]`` 对齐，可直接送
    ``regime.regime_report``（需先用 ``pad_series`` 补前 start 期占位以对齐 draws）。

    ## `k_series` 为什么必须有（V14-A2，2026-10-21）

    `k_mean` 是**全段平均**，包含筛选集为空（k=0）的期次。用它给整段 z 序列定标是错的：
    空期的命中数必然是 0、方差也必然是 0，把它们算进"平均 k"会同时低估 σ 和
    错配每期的期望。实测（`hot{k=7,w=10} ∩ omission_high{k=10}`）：

        2576 期 k=0 + 641 期 k≈6  →  k_mean = 1.1955
        真实 σ（按活跃期 k=6 计） = 0.8678，而按 k_mean 算 = 0.4204
        ⇒ z 被放大 2.1 倍、e 被放大 4 倍（k_mean=0.219 时高达 23 倍）

    commit `38d44ff` 声称"逐候选按 k 定标"，但当时 `asof_scan` **只返回 k_mean**，
    下游拿不到逐期 k，只能退回归一化到全段均值 —— 这就是它没有真正生效的原因。
    本次把 `ks` 原样返回成 `k_series`，下游 `eprocess.z_scores_from_scan` 改为逐期定标。

    `k_mean` 保留不变（多个下游在用它做退化判断），另加 `k_active_mean`
    （只统计 k>0 的期）供需要"活跃期平均筛选集"的地方使用。
    """
    draws = list(draws or [])
    end = len(draws) if end is None else min(int(end), len(draws))
    start = max(0, int(start))
    if flt.get("stat_fn") is not None:
        return _stat_scan(draws, flt, start=start, end=end, ctx=ctx)
    red_sel = flt["target"] == "red"
    obs: List[float] = []
    exp: List[float] = []
    ks: List[int] = []
    series: List[Dict] = []
    if ctx is None:
        ctx = ScanContext.new(draws[:start])
    elif ctx.n != start:
        raise ValueError(f"传入的 CTX 期数 {ctx.n} 与 start={start} 不一致")
    for t in range(start, end):
        actual = draws[t]
        try:
            chosen = set(int(x) for x in (flt["selector"](ctx) or set()))
        except Exception:  # noqa: BLE001 —— 单个维度失败不应中断整条回放
            chosen = set()
        if red_sel:
            hits = len(chosen & set(int(x) for x in (actual.get("reds") or [])))
            k = len(chosen)
            e = PICK * k / R_MAX
        else:
            k = len(chosen)
            hits = 1 if int(actual.get("blue") or 0) in chosen else 0
            e = k / B_MAX
        obs.append(float(hits))
        exp.append(float(e))
        ks.append(k)
        series.append({"issue": actual.get("issue"), "hits": hits, "expected": round(e, 4),
                       "k": k})
        ctx.advance(actual)
    active = [k for k in ks if k > 0]
    return {"target": flt["target"], "n": len(obs), "obs": obs, "exp": exp,
            "k_series": ks,
            "k_mean": float(np.mean(ks)) if ks else 0.0,
            "k_active_mean": float(np.mean(active)) if active else 0.0,
            "k_active_ratio": (len(active) / len(ks)) if ks else 0.0,
            "hit_series": series[-200:], "start": start, "end": end}


def pad_series(obs: Sequence[float], draws: Sequence[Dict], start: int,
               fill: Optional[float] = 0.0) -> List[float]:
    """把 as-of 观测序列前补 ``start`` 个占位值，使其与 ``draws`` 等长。

    占位期不参与任何阶段检验（``regime`` 的 ``pad`` 参数会整段跳过），
    这样阶段划分仍然按真实年份，且不会因为填充值污染早期阶段。
    """
    n = len(draws)
    out = [float(fill if fill is not None else 0.0)] * min(int(start), n)
    out.extend(float(x) for x in obs)
    if len(out) < n:
        out.extend([float(fill or 0.0)] * (n - len(out)))
    return out[:n]


def regime_of_filter(draws: Sequence[Dict], flt: Dict, start: int = 200,
                     end: Optional[int] = None, min_era_n: int = regime.MIN_ERA_N,
                     n_tested: Optional[int] = None, long_term_exclude: int = 0) -> Dict:
    """一个筛选条件的完整阶段画像（as-of 回放 → 阶段检验 → 状态机）。

    ``n_tested``：本次扫描的假设总数，透传给 ``regime_report`` 供「长期效应」档
    （``persistent_effect``）做整体校正；缺省 None（不校正，安全默认）。
    ``long_term_exclude``：长期档检验要排除的末尾期数（= discover 快速筛选窗口），
    避免「用哪段选它、又拿那段验它」的选择性推断（见 ``regime.long_term_effect``）。
    """
    scan = asof_scan(draws, flt, start=start, end=end)
    pad = float(np.mean(scan["exp"])) if scan["exp"] else 0.0
    series = pad_series(scan["obs"], draws, scan["start"], fill=pad)
    # **口径陷阱（V8 修）**：命中数指标（obs=本期命中个数，exp=期望命中数）不能当二项
    # 比例喂给 era_significance —— 二项口径把「每期命中几个」当成「每期 0/1」，
    # 会算出方差 = p(1-p) 而非超几何方差，并且把 2 个命中当成 1 个（均值被截断）。
    # 正确做法：用**配对差值**（obs-exp，均值 0 = 无效应）走 t 检验，与 battle/rigor 同口径。
    obs_node = pad_series(scan["obs"], draws, scan["start"], fill=pad)
    p0 = pad_series(scan["exp"], draws, scan["start"], fill=pad)
    delta_series = [float(o) - float(e) for o, e in zip(obs_node, p0)]
    rep = regime.regime_report(delta_series, draws, p0=None, label=flt["label"],
                               min_era_n=min_era_n, pad=int(scan["start"]),
                               n_tested=n_tested, long_term_exclude=long_term_exclude)
    rep["scan"] = {k: scan[k] for k in ("target", "n", "k_mean", "start", "end")}
    rep["filter"] = {"key": flt["key"], "name_zh": flt["name_zh"],
                     "target": flt["target"], "params": flt.get("params", {})}
    # 效应换算成「每期红球命中差」或「蓝球命中率差」，便于跨族比较
    rep["effect_native"] = rep["recent"].get("delta")
    return rep




_DISCOVER_CACHE: Dict[str, Any] = {}


def _draws_key(draws: Sequence[Dict]) -> str:
    last = draws[-1].get("issue") if draws else ""
    return f"{len(draws)}|{last}"


def discover(draws: Sequence[Dict], start: int = 200, families: Optional[Sequence[str]] = None,
             top: int = 20, fast_window: int = 600, min_era_n: int = regime.MIN_ERA_N,
             include_blue: bool = True, min_window_start: int = 200,
             use_cache: bool = True) -> Dict:
    """二维统计扫描：枚举维度族 × 参数，对每个组合做 as-of 回放 + 阶段画像。

    两阶段设计（控制成本）：

    1. **快速筛选**：只在最近 ``fast_window`` 期上算效应与二项 p 值；
       达不到 ``quick_alpha`` 的候选连阶段检验都不做。
    2. **完整阶段画像**：通过筛选的候选再跑全历史 as-of 回放 + 阶段检验。

    由于是"扫描"，最终结论必须按**扫描总数**做多重检验校正——返回里给出
    ``n_tested`` 与 ``expected_false_positives``（= n_tested × alpha），
    让界面能直接回答「这批发现是发现还是噪声」。
    """
    draws = list(draws or [])
    n = len(draws)
    keys = list(families) if families else list(FAMILY_BY_KEY.keys())
    if not include_blue:
        keys = [k for k in keys if not k.startswith("blue_")]
    combos: List[Dict] = []
    for key in keys:
        fam = FAMILY_BY_KEY.get(key)
        if not fam:
            continue
        for params in fam["params"]:
            combos.append(make_filter(key, params))
    # 快速筛选窗口最多回看 fast_window 期；同时保证窗口起点不早于该维度所需的最小历史
    # （例如 W=200 的热号需要 200 期历史才能算），否则窗口前段全是空筛选会稀释效应。
    quick_start = max(int(start), min(int(min_window_start), n - int(fast_window)))
    records: List[Dict] = []
    # 一次性扫描：所有维度共享同一个「已前进到 quick_start」的 CTX，
    # 每个选择器各自维护自己的增量缓存（_slide/_wd_slide/_pair_slide/_pat），
    # 互不干扰；这样 91 个组合不必各自重扫一遍前 600 期。
    base_ctx = ScanContext.new(draws[:quick_start])
    for flt in combos:
        scan = asof_scan(draws, flt, start=quick_start, end=n, ctx=base_ctx.clone())
        if not scan["obs"]:
            continue
        obs = np.asarray(scan["obs"], float)
        exp = float(np.mean(scan["exp"])) if scan["exp"] else 0.0
        mean = float(obs.mean())
        if scan["n"] < 50:
            continue
        if flt.get("stat_fn") is not None:
            # 分布类维度：用 (obs-exp) 的经验标准差做配对 t 的标准误（与 regime 同口径）。
            dser = obs - np.asarray(scan["exp"], float)
            dsd = float(dser.std(ddof=1)) if dser.size >= 2 else 0.0
            if dsd <= 0:
                continue
            se = dsd / math.sqrt(scan["n"])
            z = float(dser.mean()) / se
            p = float(2 * math.erfc(abs(z) / math.sqrt(2)))
            records.append({"filter": {"key": flt["key"], "name_zh": flt["name_zh"],
                                       "target": flt["target"], "params": flt.get("params", {})},
                            "quick": {"n": scan["n"], "mean": round(float(dser.mean()), 4),
                                      "expected": round(exp, 4), "delta": round(float(dser.mean()), 4),
                                      "z": round(z, 3), "p": p},
                            "_flt": flt})
            continue
        sd = math.sqrt(max(exp * (1 - exp / max(PICK, 1)) if flt["target"] == "red" else exp * (1 - exp), 0.0))
        if sd <= 0:
            continue
        k_eff = max(1.0, scan["k_mean"])
        # red：命中数的每期方差（超几何近似）→ 标准误
        if flt["target"] == "red":
            var = k_eff * P_RED * (1 - P_RED) * (R_MAX - PICK) / (R_MAX - 1)
        else:
            var = exp * (1 - exp)
        se = math.sqrt(max(var, 1e-9) / scan["n"])
        z = (mean - exp) / se
        p = float(2 * math.erfc(abs(z) / math.sqrt(2)))
        records.append({"filter": {"key": flt["key"], "name_zh": flt["name_zh"],
                                   "target": flt["target"], "params": flt.get("params", {})},
                        "quick": {"n": scan["n"], "mean": round(mean, 4),
                                  "expected": round(exp, 4), "delta": round(mean - exp, 4),
                                  "z": round(z, 3), "p": p},
                        "_flt": flt})
    n_tested = len(records) if combos else 0
    quick_ps = [r["quick"]["p"] for r in records]
    adj = regime.family_adjust(quick_ps)
    for r, a in zip(records, adj):
        r["quick"]["p_adjusted"] = a
    # 取 |z| 最大的若干条做完整阶段画像（含全历史阶段检验）
    records.sort(key=lambda r: -abs(r["quick"]["z"]))
    # ---- S1：候选稀释修复（P2 §7.1 S1）----
    # 旧实现直接 ``records[:top]``，同一族的参数化（如 cold{w=10,k=5/6/8}）会各自
    # 占据一个名额，把真正**独立**的假设挤出阶段画像。实测（dilution.json）：
    # 注入 +20% 时 124 条里 BH 显著 50.5 条，但经 top-12 裁剪后阶段 g3 仍为 0
    # —— 瓶颈在裁剪，不在信号。
    #
    # 改法：**按族去重**后取 top-N —— 同一 ``key`` 只保留 ``|z|`` 最强的一条进入
    # 阶段画像；其余参数档**仍完整保留在 records / all_quick 里**供前端展示，
    # 只是不再占用阶段画像名额。于是 top-N 覆盖的是 N 个**独立维度**，而不是
    # N 个同族变体。族内最强那条（原本可能因名额被同族兄弟挤掉）现在一定能进画像。
    #
    # **n_tested 绝不因此改变**：去重只是"阶段画像展示哪几条"的取舍，**不是**
    # 少做检验。FWER 的分母是「扫描了多少个假设」，仍是 ``len(records)``（去重前）。
    # 这一点写进返回体 note 与 docs/analysis/V15_ASSUMPTION_SPACE.md。
    _seen_keys: set = set()
    _dedup: List[Dict] = []
    for _r in records:
        _k = _r["filter"]["key"]
        if _k in _seen_keys:
            continue
        _seen_keys.add(_k)
        _dedup.append(_r)
    short = _dedup[:max(1, int(top))]
    for r in short:
        flt = r.pop("_flt")
        try:
            rep = regime_of_filter(draws, flt, start=start, min_era_n=min_era_n,
                                   n_tested=n_tested, long_term_exclude=int(fast_window))
        except Exception as e:  # noqa: BLE001
            rep = {"status": "error", "label_zh": "阶段检验失败",
                   "reason_zh": f"{type(e).__name__}: {e}"}
        r["regime"] = rep
        r["status"] = rep.get("status")
        r["status_zh"] = rep.get("label_zh")
        r["reason_zh"] = rep.get("reason_zh")
        r["recent_delta"] = (rep.get("recent") or {}).get("delta")
        r["recent_p_adjusted"] = (rep.get("recent") or {}).get("p_adjusted")
        r["n_sig_eras"] = rep.get("n_sig_eras")
    # 去重后 short 不再是 records 的前缀，不能用切片；按 id 排除已处理的条目。
    _short_ids = {id(r) for r in short}
    for r in records:
        if id(r) not in _short_ids:
            r.pop("_flt", None)
    status_count: Dict[str, int] = {}
    for r in short:
        status_count[r.get("status") or "unknown"] = status_count.get(r.get("status") or "unknown", 0) + 1
    best_quick_adj = min((r["quick"]["p_adjusted"] for r in records
                          if r["quick"].get("p_adjusted") is not None), default=None)
    result = {
        "periods_cached": quick_start,
        "n_tested": n_tested,
        "alpha": regime.ALPHA,
        "expected_false_positives": round(n_tested * regime.ALPHA, 2),
        "quick_window": int(quick_start),
        "best_quick_p_adjusted": best_quick_adj,
        "verdict": ("扫描后无任何维度达到校正显著" if (best_quick_adj is None or best_quick_adj >= regime.ALPHA)
                    else "存在候选，需逐条看阶段画像"),
        "status_count": status_count,
        "items": [{k: v for k, v in r.items() if k != "_flt"} for r in short],
        "all_quick": [{k: v for k, v in r.items() if k != "_flt"} for r in records],
        "families": [f["key"] for f in FAMILIES if f["key"] in keys],
        "note_zh": (f"共枚举 {n_tested} 个维度×参数组合，纯随机下预期约 "
                    f"{n_tested * regime.ALPHA:.1f} 个会出现未校正 p<0.05；"
                    "只有校正后仍 <0.05 的才算线索。"),
    }
    if use_cache:
        _DISCOVER_CACHE[_draws_key(draws)] = result
    return result


