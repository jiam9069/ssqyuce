"""规律库：把"规律"定义为 触发条件 + 候选动作（应选/应避号码集合）。

所有规律只用目标期之前的历史（walk-forward，禁止未来信息），
由 backtest.py 在历史上滚动验证并做显著性分级。
"""
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Callable, Dict, List

import numpy as np

from . import features as F

REDS, BLUE = 33, 16
_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


# ---------- 工具 ----------

def _comb_p(n, k):
    from math import comb
    return comb(n, k)


def red_base(k: int):
    """红球：k 个号码在单注 6 红中至少出现 1 个的理论概率与期望命中数。"""
    p0 = 1 - _comb_p(REDS - k, 6) / _comb_p(REDS, 6) if k <= REDS else 0.0
    return {"p0": p0, "exp": 6 * k / REDS}


def blue_base(k: int):
    """蓝球：k 个号码在单期 1 蓝中命中的理论概率。"""
    return {"p0": k / BLUE, "exp": k / BLUE}


# ---------- 规律定义 ----------

def _t_always(history: List[Dict]) -> bool:
    return True


def _act_blue_cold_back(history: List[Dict]) -> Dict:
    om = F.current_omission_blue(history)
    mask = np.where(om[1:] >= 20)[0] + 1
    if len(mask) == 0:
        return {}
    order = mask[np.argsort(-om[mask])]
    return {"blue_fav": [int(x) for x in order[:1]]}


def _act_red_freq_reversion(history: List[Dict]) -> Dict:
    freq = F.red_frequency(history)
    n = len(history)
    exp = n * 6 / REDS
    dev = (freq[1:] - exp) / np.sqrt(exp) if exp > 0 else np.zeros(REDS)
    mask = np.where(dev <= -2.0)[0] + 1
    if len(mask) == 0:
        return {}
    return {"red_fav": [int(x) for x in mask]}


def _act_red_omission_pressure(history: List[Dict]) -> Dict:
    om = F.current_omission_red(history)
    avg = F.avg_omission(F.red_frequency(history), len(history))
    ratio = np.divide(om[1:], avg[1:], out=np.ones(REDS), where=avg[1:] > 0)
    mask = np.where((om[1:] >= 25) & (ratio >= 1.5))[0] + 1
    if len(mask) == 0:
        return {}
    return {"red_fav": [int(x) for x in mask]}


def _act_zone_low_rebound(history: List[Dict]) -> Dict:
    sl = history[-6:] if len(history) >= 6 else history
    zc = F.zone_counts(sl)
    totals = [0, 0, 0]
    for z in zc:
        for i in range(3):
            totals[i] += z[i]
    low = int(np.argmin(totals))
    lo, hi = {0: (1, 11), 1: (12, 22), 2: (23, 33)}[low]
    return {"red_fav": list(range(lo, hi + 1))}


def _act_parity_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    odd = sum(1 for r in history[-1]["reds"] if r % 2 == 1)
    if odd >= 4:
        return {"red_fav": [x for x in range(1, REDS + 1) if x % 2 == 0]}
    if odd <= 2:
        return {"red_fav": [x for x in range(1, REDS + 1) if x % 2 == 1]}
    return {}


def _act_repeat(history: List[Dict]) -> Dict:
    if not history:
        return {}
    return {"red_fav": history[-1]["reds"], "blue_fav": [history[-1]["blue"]]}


def _act_adjacent(history: List[Dict]) -> Dict:
    if not history:
        return {}
    last = history[-1]["reds"]
    fav = set()
    for r in last:
        if r - 1 >= 1:
            fav.add(r - 1)
        if r + 1 <= REDS:
            fav.add(r + 1)
    bfav = set()
    b = history[-1]["blue"]
    if b - 1 >= 1:
        bfav.add(b - 1)
    if b + 1 <= BLUE:
        bfav.add(b + 1)
    return {"red_fav": sorted(fav), "blue_fav": sorted(bfav)}


def _act_recent_hot(history: List[Dict]) -> Dict:
    sl = history[-30:] if len(history) >= 30 else history
    freq = F.red_frequency(sl)
    top = np.argsort(freq[1:])[-6:][::-1] + 1
    bfreq = F.blue_frequency(sl)
    btop = np.argsort(bfreq[1:])[-2:][::-1] + 1
    return {"red_fav": [int(x) for x in top], "blue_fav": [int(x) for x in btop]}


def _act_recent_cold(history: List[Dict]) -> Dict:
    sl = history[-30:] if len(history) >= 30 else history
    freq = F.red_frequency(sl)
    bot = np.argsort(freq[1:])[:6] + 1
    return {"red_fav": [int(x) for x in bot]}


def _t_sum_extreme(history: List[Dict]) -> bool:
    if len(history) < 200:
        return len(history) >= 30
    s = F.sums(history)
    p85, p15 = np.percentile(s, 85), np.percentile(s, 15)
    return s[-1] >= p85 or s[-1] <= p15


# ---------- 新增规律（M1 扩展） ----------


def _act_prime_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    pr = sum(1 for r in history[-1]["reds"] if r in _PRIMES)
    if pr >= 4:
        return {"red_fav": [r for r in range(1, REDS + 1) if r not in _PRIMES]}
    if pr <= 1:
        return {"red_fav": [r for r in range(1, REDS + 1) if r in _PRIMES]}
    return {}


def _act_size_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    small = sum(1 for r in history[-1]["reds"] if 1 <= r <= 16)
    if small >= 4:
        return {"red_fav": list(range(17, REDS + 1))}
    if small <= 1:
        return {"red_fav": list(range(1, 17))}
    return {}


def _act_route_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    r0 = sum(1 for r in history[-1]["reds"] if r % 3 == 0)
    if r0 >= 3:
        return {"red_fav": [r for r in range(1, REDS + 1) if r % 3 != 0]}
    if r0 <= 1:
        return {"red_fav": [r for r in range(1, REDS + 1) if r % 3 == 0]}
    return {}


def _act_consecutive_groups(history: List[Dict]) -> Dict:
    if not history:
        return {}
    rs = sorted(history[-1]["reds"])
    fav = set()
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and rs[j+1] - rs[j] == 1:
            j += 1
        if j > i:
            fav.update(rs[i:j+1])
        i = j + 1
    return {"red_fav": sorted(fav)} if fav else {}


def _act_tail_groups(history: List[Dict]) -> Dict:
    if not history:
        return {}
    groups = defaultdict(list)
    for r in history[-1]["reds"]:
        groups[r % 10].append(r)
    fav = set()
    for nums in groups.values():
        if len(nums) > 1:
            fav.update(nums)
    return {"red_fav": sorted(fav)} if fav else {}


def _act_head_regression(history: List[Dict]) -> Dict:
    if not history:
        return {}
    h = sorted(history[-1]["reds"])[0]
    if h <= 3:
        return {"red_fav": list(range(5, 13))}
    if h >= 14:
        return {"red_fav": list(range(1, 9))}
    return {}


def _act_tail_regression(history: List[Dict]) -> Dict:
    if not history:
        return {}
    t = sorted(history[-1]["reds"])[-1]
    if t >= 32:
        return {"red_fav": list(range(23, 31))}
    if t <= 22:
        return {"red_fav": list(range(26, REDS + 1))}
    return {}


def _act_omit_bin_hot(history: List[Dict]) -> Dict:
    om = F.current_omission_red(history)
    fav = [int(i + 1) for i in range(REDS) if 6 <= om[i + 1] <= 15]
    return {"red_fav": fav} if fav else {}


def _act_cold_release(history: List[Dict]) -> Dict:
    om = F.current_omission_red(history)
    cold = [int(i + 1) for i in range(REDS) if om[i + 1] >= 20]
    if len(cold) < 4:
        return {}
    sl = history[-30:] if len(history) >= 30 else history
    hot = F.red_frequency(sl)
    top = np.argsort(hot[1:])[-3:][::-1] + 1
    combined = list(set(cold) | set(top.tolist()))
    return {"red_fav": sorted(combined)[:10]}


def _act_hot_retreat(history: List[Dict]) -> Dict:
    if len(history) < 3:
        return {}
    appear = set(history[-1]["reds"]) | set(history[-2]["reds"]) | set(history[-3]["reds"])
    return {"red_fav": [r for r in range(1, REDS + 1) if r not in appear]}


def _act_blue_size_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    b = history[-1]["blue"]
    if b <= 4:
        return {"blue_fav": list(range(9, BLUE + 1))}
    if b >= 13:
        return {"blue_fav": list(range(1, 8))}
    return {}


def _act_blue_odd_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    b = history[-1]["blue"]
    if b % 2 == 1:
        return {"blue_fav": [x for x in range(1, BLUE + 1) if x % 2 == 0]}
    return {"blue_fav": [x for x in range(1, BLUE + 1) if x % 2 == 1]}


def _act_blue_route_rebound(history: List[Dict]) -> Dict:
    if not history:
        return {}
    b = history[-1]["blue"]
    r = b % 3
    fav = [x for x in range(1, BLUE + 1) if x % 3 != r]
    return {"blue_fav": fav}


def _act_blue_omit_bin(history: List[Dict]) -> Dict:
    om = F.current_omission_blue(history)
    fav = [int(i + 1) for i in range(BLUE) if 5 <= om[i + 1] <= 10]
    return {"blue_fav": fav} if fav else {}


def _act_blue_adjacent(history: List[Dict]) -> Dict:
    """蓝球邻号延续：上期蓝球 +/-1 的号码。"""
    if not history:
        return {}
    b = history[-1]["blue"]
    fav = sorted({x for x in (b - 1, b + 1) if 1 <= x <= 16})
    return {"blue_fav": fav} if fav else {}


def _act_blue_alt_repeat(history: List[Dict]) -> Dict:
    """蓝球隔期复出：上上期蓝球在下一期出现。"""
    if len(history) < 2:
        return {}
    return {"blue_fav": [history[-2]["blue"]]}


def _act_blue_hot_cold_switch(history: List[Dict]) -> Dict:
    """蓝球热冷切换：近 20 期最热的蓝球（排除上期）延续热度。"""
    sl = history[-20:] if len(history) >= 20 else history
    if len(sl) < 3:
        return {}
    f = F.blue_frequency(sl)
    last = sl[-1]["blue"]
    order = np.argsort(-f[1:]) + 1
    fav = [int(x) for x in order if int(x) != last and f[x] > 0][:2]
    return {"blue_fav": fav} if fav else {}


def _act_diagonal(history: List[Dict]) -> Dict:
    if len(history) < 2:
        return {}
    prev2 = history[-2]["reds"]
    fav = set()
    for r in prev2:
        if r - 1 >= 1:
            fav.add(r - 1)
        if r + 1 <= REDS:
            fav.add(r + 1)
    return {"red_fav": sorted(fav)} if fav else {}


def _act_gap(history: List[Dict]) -> Dict:
    if not history:
        return {}
    last = history[-1]["reds"]
    fav = set()
    for r in last:
        if r - 2 >= 1:
            fav.add(r - 2)
        if r + 2 <= REDS:
            fav.add(r + 2)
    return {"red_fav": sorted(fav)} if fav else {}


def _act_repeat_by_count(history: List[Dict]) -> Dict:
    if len(history) < 2:
        return {}
    rp = len(set(history[-2]["reds"]) & set(history[-1]["reds"]))
    if rp == 0:
        return {"red_fav": list(history[-1]["reds"])}
    if rp >= 2:
        return {"red_fav": [r for r in range(1, REDS + 1) if r not in history[-1]["reds"]]}
    return {}


def _act_pair_cooccur(history: List[Dict]) -> Dict:
    if len(history) < 50:
        return {}
    sl = history[-50:]
    pairs = Counter()
    for d in sl:
        rs = sorted(d["reds"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pairs[(rs[i], rs[j])] += 1
    top_pairs = [p for p, _ in pairs.most_common(3)]
    fav = set()
    for a, b in top_pairs:
        fav.add(a)
        fav.add(b)
    return {"red_fav": sorted(fav)} if fav else {}


# ---------- 新增触发器 ----------


def _t_prime_extreme(history: List[Dict]) -> bool:
    if len(history) < 30:
        return False
    pr = F.prime_counts(history)
    return float(pr[-1]) >= 4 or float(pr[-1]) <= 1


def _t_size_extreme(history: List[Dict]) -> bool:
    if len(history) < 30:
        return False
    sz = F.size_counts(history)
    return float(sz[-1]) >= 4 or float(sz[-1]) <= 1


def _t_route_extreme(history: List[Dict]) -> bool:
    if len(history) < 30:
        return False
    rt = F.route_counts(history)
    return float(rt[-1]) >= 3 or float(rt[-1]) <= 1


def _t_head_extreme(history: List[Dict]) -> bool:
    if len(history) < 30:
        return False
    fl = F.first_last_reds(history)
    return int(fl[-1, 0]) <= 3 or int(fl[-1, 0]) >= 14


def _t_tail_extreme(history: List[Dict]) -> bool:
    if len(history) < 30:
        return False
    fl = F.first_last_reds(history)
    return int(fl[-1, 1]) >= 32 or int(fl[-1, 1]) <= 22


def _t_cold_pool(history: List[Dict]) -> bool:
    om = F.current_omission_red(history)
    return int(np.sum(om[1:] >= 20)) >= 4


def _t_hot_streak(history: List[Dict]) -> bool:
    if len(history) < 3:
        return False
    s1 = set(history[-1]["reds"])
    s2 = set(history[-2]["reds"])
    s3 = set(history[-3]["reds"])
    return bool(s1 & s2 & s3)


def _t_blue_size_extreme(history: List[Dict]) -> bool:
    if not history:
        return False
    b = history[-1]["blue"]
    return b <= 4 or b >= 13


def _t_blue_odd_extreme(history: List[Dict]) -> bool:
    if not history:
        return False
    return history[-1]["blue"] % 2 != 0


def _t_repeat_count(history: List[Dict]) -> bool:
    if len(history) < 2:
        return False
    rp = len(set(history[-2]["reds"]) & set(history[-1]["reds"]))
    return rp == 0 or rp >= 2


def _t_span_extreme(history: List[Dict]) -> bool:
    """上期跨度处于历史极端分位（<=P15 或 >=P85）时触发。"""
    if len(history) < 60:
        return False
    sp = F.span(history)
    lo, hi = np.percentile(sp, 15), np.percentile(sp, 85)
    cur = sp[-1]
    return bool(cur <= lo or cur >= hi)


def _t_ac_extreme(history: List[Dict]) -> bool:
    """上期 AC 值处于历史极端分位时触发。"""
    if len(history) < 60:
        return False
    acs = np.array([F.ac_value(d["reds"]) for d in history], dtype=float)
    lo, hi = np.percentile(acs, 15), np.percentile(acs, 85)
    cur = acs[-1]
    return bool(cur <= lo or cur >= hi)


def _t_sum_var_contract(history: List[Dict]) -> bool:
    """近 10 期和值波动收窄到历史低位（<=P30）时触发。"""
    if len(history) < 40:
        return False
    sums = F.sums(history)
    cur_std = float(np.std(sums[-10:]))
    stds = np.array([float(np.std(sums[j:j + 10]))
                     for j in range(len(sums) - 10)], dtype=float)
    if len(stds) < 20:
        return False
    return bool(cur_std <= np.percentile(stds, 30))


# ---------- 规律清单 ----------

PATTERNS: List[Dict] = [
    {
        "key": "blue_cold_back",
        "name_zh": "蓝球冷号回补",
        "kind": "long",
        "desc": "蓝球遗漏 >=20 期后，下一期（或数期内）出现概率是否高于均匀基线 1/16",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_cold_back,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "red_freq_reversion",
        "name_zh": "红球频率均值回归",
        "kind": "long",
        "desc": "长期频率显著低于期望(z<=-2)的红球，下一期是否补出",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_red_freq_reversion,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "red_omission_pressure",
        "name_zh": "红球遗漏压力",
        "kind": "long",
        "desc": "遗漏>=25 且超过自身平均遗漏 1.5 倍的红球，下一期是否回补",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_red_omission_pressure,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "zone_low_rebound",
        "name_zh": "低活跃区回补（中期轮动）",
        "kind": "mid",
        "desc": "近6期三区合计最少的区间，其 11 个号码下一期命中是否高于随机",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_zone_low_rebound,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "parity_rebound",
        "name_zh": "奇偶比例回归",
        "kind": "mid",
        "desc": "上期奇数>=4 选偶数、上期奇数<=2 选奇数，验证奇偶是否轮动",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_parity_rebound,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "sum_regression",
        "name_zh": "和值回归",
        "kind": "mid",
        "desc": "上期和值处于历史极端(>P85 或 <P15)时，下一期和值是否更接近均值",
        "horizon": 1,
        "trigger_fn": _t_sum_extreme,
        "action_fn": lambda h: {},
        "outcome": "numeric_sum",
        "base_fn": None,
    },
    {
        "key": "repeat_trend",
        "name_zh": "重号延续",
        "kind": "short",
        "desc": "上期 6 个红球在下一期复出数量是否高于随机期望 6*6/33~1.09",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_repeat,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "adjacent_trend",
        "name_zh": "邻号延续",
        "kind": "short",
        "desc": "上期红球+/-1 邻号、蓝球+/-1 邻号在下一期命中是否高于随机",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_adjacent,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "recent_hot",
        "name_zh": "短期热号",
        "kind": "short",
        "desc": "近30期最热的 6 个红球（+2 个蓝球）下一期是否延续热度",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_recent_hot,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "recent_cold",
        "name_zh": "短期冷号",
        "kind": "short",
        "desc": "近30期最冷的 6 个红球下一期是否反弹",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_recent_cold,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "blue_repeat",
        "name_zh": "蓝球延续",
        "kind": "short",
        "desc": "上期蓝球下一期重复命中率 vs 1/16",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": lambda h: {"blue_fav": [h[-1]["blue"]]} if h else {},
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    # ----- M1 新增规律 -----
    {
        "key": "prime_rebound",
        "name_zh": "质合比轮动",
        "kind": "mid",
        "desc": "上期质数>=4 选合数、<=1 选质数，验证质合是否轮动",
        "horizon": 1,
        "trigger_fn": _t_prime_extreme,
        "action_fn": _act_prime_rebound,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "size_rebound",
        "name_zh": "大小比轮动",
        "kind": "mid",
        "desc": "上期小号(1-16)>=4 选大号、<=1 选小号",
        "horizon": 1,
        "trigger_fn": _t_size_extreme,
        "action_fn": _act_size_rebound,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "route_rebound",
        "name_zh": "012路比轮动",
        "kind": "mid",
        "desc": "上期0路(被3整除)>=3 选非0路、<=1 选0路",
        "horizon": 1,
        "trigger_fn": _t_route_extreme,
        "action_fn": _act_route_rebound,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "consecutive_groups",
        "name_zh": "连号组延续",
        "kind": "short",
        "desc": "上期连号组内号码在下一期复出的概率",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_consecutive_groups,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "tail_groups",
        "name_zh": "同尾组延续",
        "kind": "short",
        "desc": "上期同尾组内号码在下一期复出的概率",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_tail_groups,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "head_regression",
        "name_zh": "龙头回归",
        "kind": "mid",
        "desc": "龙头(最小红)<=3 或 >=14 时向均值回归选号",
        "horizon": 1,
        "trigger_fn": _t_head_extreme,
        "action_fn": _act_head_regression,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "tail_regression",
        "name_zh": "凤尾回归",
        "kind": "mid",
        "desc": "凤尾(最大红)>=32 或 <=22 时向均值回归选号",
        "horizon": 1,
        "trigger_fn": _t_tail_extreme,
        "action_fn": _act_tail_regression,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "omit_bin_hot",
        "name_zh": "遗漏适中区间热点",
        "kind": "short",
        "desc": "当前遗漏在 6-15 区间的红球是否即将回补",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_omit_bin_hot,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "cold_release",
        "name_zh": "冷号集中释放",
        "kind": "short",
        "desc": "冷号池(遗漏>=20)>=4 个时，未来 2 期是否集中爆发",
        "horizon": 2,
        "trigger_fn": _t_cold_pool,
        "action_fn": _act_cold_release,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "hot_retreat",
        "name_zh": "超热退潮",
        "kind": "short",
        "desc": "连热3期的号码停止热出，应排除",
        "horizon": 1,
        "trigger_fn": _t_hot_streak,
        "action_fn": _act_hot_retreat,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "blue_size_rebound",
        "name_zh": "蓝球大小轮动",
        "kind": "short",
        "desc": "上期蓝<=4 选大号 9-16、>=13 选小号 1-8",
        "horizon": 1,
        "trigger_fn": _t_blue_size_extreme,
        "action_fn": _act_blue_size_rebound,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "blue_odd_rebound",
        "name_zh": "蓝球奇偶轮动",
        "kind": "short",
        "desc": "上期蓝奇则选偶、偶则选奇",
        "horizon": 1,
        "trigger_fn": _t_blue_odd_extreme,
        "action_fn": _act_blue_odd_rebound,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "blue_route_rebound",
        "name_zh": "蓝球012路轮动",
        "kind": "short",
        "desc": "上期蓝 mod 3 的余数对应的路别轮动",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_route_rebound,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "blue_omit_bin",
        "name_zh": "蓝球遗漏适中区间",
        "kind": "short",
        "desc": "当前遗漏在 5-10 区间的蓝球是否即将回补",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_omit_bin,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "diagonal",
        "name_zh": "斜连号延续",
        "kind": "short",
        "desc": "上上期号码 +/-1 邻号在下一期命中",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_diagonal,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "gap_number",
        "name_zh": "间隔号延续",
        "kind": "short",
        "desc": "上期号码 +/-2 间隔号在下一期命中",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_gap,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "repeat_by_count",
        "name_zh": "重号个数条件分布",
        "kind": "mid",
        "desc": "上期重0个 fav 上期号码(期待回补)；重>=2个则 exclude",
        "horizon": 1,
        "trigger_fn": _t_repeat_count,
        "action_fn": _act_repeat_by_count,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    {
        "key": "pair_cooccur",
        "name_zh": "号码对共现",
        "kind": "short",
        "desc": "近50期最常共现的 top-3 对中的号码",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_pair_cooccur,
        "outcome": "red_fav",
        "base_fn": red_base,
    },
    # ----- M2 新增规律（29 -> 35） -----
    {
        "key": "blue_adjacent",
        "name_zh": "蓝球邻号延续",
        "kind": "short",
        "desc": "上期蓝球 +/-1 邻号在下一期命中是否高于 2/16",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_adjacent,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "blue_alt_repeat",
        "name_zh": "蓝球隔期复出",
        "kind": "short",
        "desc": "上上期蓝球在下一期再次出现是否高于 1/16",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_alt_repeat,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "blue_hot_cold_switch",
        "name_zh": "蓝球热冷切换",
        "kind": "short",
        "desc": "近20期最热蓝球（排除上期）延续热度是否高于基线",
        "horizon": 1,
        "trigger_fn": _t_always,
        "action_fn": _act_blue_hot_cold_switch,
        "outcome": "blue_fav",
        "base_fn": blue_base,
    },
    {
        "key": "span_regression",
        "name_zh": "跨度回归",
        "kind": "mid",
        "desc": "上期跨度处于极端分位时，下期跨度是否向均值回归",
        "horizon": 1,
        "trigger_fn": _t_span_extreme,
        "action_fn": lambda h: {},
        "outcome": "numeric_span",
        "base_fn": None,
    },
    {
        "key": "ac_regression",
        "name_zh": "AC值回归",
        "kind": "mid",
        "desc": "上期 AC 值处于极端分位时，下期 AC 值是否向均值回归",
        "horizon": 1,
        "trigger_fn": _t_ac_extreme,
        "action_fn": lambda h: {},
        "outcome": "numeric_ac",
        "base_fn": None,
    },
    {
        "key": "sum_var_contract",
        "name_zh": "和值方差收缩",
        "kind": "mid",
        "desc": "近10期和值波动收窄到低位后，下期和值是否回归均值",
        "horizon": 1,
        "trigger_fn": _t_sum_var_contract,
        "action_fn": lambda h: {},
        "outcome": "numeric_sum",
        "base_fn": None,
    },
]


def run_pattern(pattern: Dict, history: List[Dict]) -> Dict:
    """在给定历史上执行规律，返回触发状态与动作。"""
    out = {"triggered": bool(pattern["trigger_fn"](history)), "action": {}}
    if out["triggered"]:
        out["action"] = pattern["action_fn"](history) or {}
    return out


