"""闲时挖掘调度器（P4）。

## 它要解决的问题

系统已经有挖掘引擎（`discovery`/`multiscale`/`regime`），但**没有让它在闲时跑起来**。
M2 的说法很准：「闲时自动挖掘」要建的不是新引擎，而是**给已有引擎装一个能真正
跑起来的调度**。

一天全量交叉能轻松产出上万次检验（124 维度 × 5 窗口 × 23 分段 = 14 260）。
没有治理的后果是精确的：在 m=14260 处，Bonferroni 阈值 z=4.63，
把系统的最小可检效应从 +5.9% 抬到 +7.4%，**而且是永久的**（累计口径不回落）。
这就是为什么本模块的判定必须走 e-process（见 `eprocess.py`）。

## 三条硬约束

### 1. 在线请求优先（让路）

挖掘是 CPU 密集的，跑满会把 40 线程池吃光 → healthcheck 超时 → 容器被判不健康。
本模块用**轮次 + 批间让路**实现：每一小批之后 `time.sleep(0)` 让出 GIL，
并在检测到"最近有在线请求"时加大间隔。

### 2. 可中断、可续跑

中段杀进程（部署、OOM）不能丢进度，也不能重头再来。
状态存 `data/idle_mining_state.json`：已完成到哪个维度、哪些候选已判定。

### 3. 总开关

`LOTT_IDLE_MINING=0` 时整个模块不启动。默认为 **0（关闭）**：
这是新机制，先观察再开，而不是默认改变生产行为。

## 它输出什么

**不是**"找到 N 条规律"这类结论，而是**候选的判定记录**：
每条带 `e_value / alpha_spent / n_obs / decision`（L-C 报告层要求）。
新增的"可用规律"条数在真实数据上大概率是 **0** —— 那是诚实的结果。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config, db, eprocess

#: 每批处理的候选数（批间让路，避免长时间占满 CPU）
BATCH_SIZE = 8
#: 批间基础休眠（秒）
BASE_SLEEP = 0.05
#: 检测到在线请求压力时加大的休眠
PRESSURE_SLEEP = 0.5
#: 无压力时批间的最小让出（给 GIL / SQLite 连接喘口气，不是为省 CPU）
BATCH_GAP = 0.01
#: 判定所需最小观测数（低于此不下结论）
MIN_OBS = 30

#: **CPU 占空比上限 —— 口径是「挖掘进程自身可用配额」，不是整机。**
#:
#: ## 这个数字到底在量什么（V40 补正；此前文档含混过一版）
#:
#: `_system_cpu_pct()` 返回归一化占用率 0..100，**分母是「挖掘进程可用的核数」**：
#:
#: * 容器内：分母 = cgroup `cpu.max` 配额核数（本机实测 **2.4 核**），
#:   即 `usage_usec 增量 / (2.4 × 采样窗)`。
#: * 非容器：分母 = 整机核数，即 `/proc/stat` 总占用。
#:
#: 两种情况下 `sys_pct` 都是「占**我这边的分母**的百分之几」，所以
#: `CPU_LIMIT_PCT` 只可能是「**占同一分母**的百分之几」，跟着分母走：
#:
#: | 环境 | 分母 | `60` 等于 |
#: |---|---|---|
#: | 容器（本机 cgroup 限 2.4 核） | 2.4 核 | **1.44 核**（上限） |
#: | 非容器（整机 4 核） | 4 核 | 2.4 核（上限） |
#:
#: **不是**「整机 60%」这个固定物理量。容器里同一个 `60` 算出更小的绝对核数，
#: 偏保守 —— 可接受，但必须写清楚，否则后人会以为容器限流算错了。
#:
#: ## 一个已知的口径不一致（已核查，故意保留）
#:
#: `_throttle()` 里换算「一个核值多少百分点」用 `one_core = 100 / _cpu_count()`，
#: 而 `_cpu_count()` 读 `sched_getaffinity`（**宿主 4 核**），不是 cgroup 配额 2.4。
#: 容器内因此得到 `one_core = 25`，而分母实际是 2.4 核 ⇒ 真正的「一个核」≈ 41.7 点。
#:
#: 后果：容器里把「一个核」**想小了**（25 < 41.7），`headroom < one_core` 更易成立
#: ⇒ **更早让路 ⇒ 更保守**。方向安全（只会少用 CPU，不会超用），
#: 且 cgroup 2.4 核硬限是最终兜底。不改的理由：让 `_cpu_count()` 返回配额核数会
#: 破坏 `_system_cpu_pct()` 的单一分母假设，牵动面大于收益。此处如实记录，不假装精确。
#:
#: ## 历史坑（第一版错在这里，别重犯）
#:
#: 第一版把 `r` 当成「本进程可用的核心比例」，算出 `t_sleep ≥ 0.667·t_work`，
#: 等于把进程压到单核 60%。实测代价：单候选墙钟 0.05s 拖到 0.29s，吞吐 2.9 候选/秒。
CPU_LIMIT_PCT = float(os.environ.get("LOTT_IDLE_CPU_PCT", "60") or "60")
#: 控制环的采样窗口（秒）：每这么长时间校一次占空比，避免抖动
CPU_SAMPLE_WINDOW = 2.0


def _cpu_count() -> int:
    """本机可用核心数（用于把占空比换算成核数）。"""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _read_cgroup_cpu_usage() -> Optional[int]:
    """读**本容器**的 CPU 使用（微秒）。读不到返回 None。

    ## 为什么必须优先用 cgroup（V39，2026-10-24 审计 P2）

    在容器里 `/proc/stat` 显示的是**宿主整机**的 CPU（实测 docker exec 看到
    的 `cpu 43429336...` 是整机累计，不是本容器的）。于是 `_system_cpu_pct`
    在共享宿主上会**把其它容器的占用当成自己的** —— 别的容器跑满时，
    本容器会误以为"整机已满"而过度限流；反过来本容器吃满但宿主空闲时，
    又会不限流。读宿主而不是读自己，是这套限流的真实偏差。

    cgroup v2 的 `/sys/fs/cgroup/cpu.stat` 里 `usage_usec` 是**本容器**的
    CPU 时间，配合 `cpu.max` 的配额/周期能算出本容器的占用率。这才能回答
    "我的容器现在用了多少"，而不是"宿主整机用了多少"。
    """
    try:
        with open("/sys/fs/cgroup/cpu.stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_cgroup_cpu_quota() -> Optional[float]:
    """读本容器的 CPU 配额（核心数）。读不到返回 None。

    `cpu.max` 形如 `240000 100000`（配额微秒 / 周期微秒）= 2.4 核。
    """
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as f:
            parts = f.read().split()
        if len(parts) == 2:
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return quota / period
    except (OSError, ValueError, IndexError):
        return None
    return None


def _system_cpu_pct() -> Optional[float]:
    """整机（或本容器）当前 CPU 使用率（%）。

    `os.getloadavg()` 给的是 1 分钟平均负载，对「实时限流」太迟钝。

    **V39 口径修正**：优先读**本容器**的 cgroup `usage_usec` 差分 —— 这才是
    "我用了多少"，而不是"宿主整机用了多少"。容器里 `/proc/stat` 是宿主视图，
    在共享宿主上会把别人的占用算成自己的。

    算容器占用率：`usage 增量 / (配额核数 × 周期 × 采样窗口)`，钳到 0..100。
    若 cgroup 读不到（非容器环境），回退到宿主 `/proc/stat`（保留原行为）。
    读不到返回 None（调用方跳过控制）。
    """
    # --- 优先：容器 cgroup ---
    quota = _read_cgroup_cpu_quota()          # 2.4 等；None = 读不到
    # **只有配额与 usage 都能读到时才用容器口径**。非容器环境里
    # `/sys/fs/cgroup/cpu.stat` 可能读到宿主的累计值、而 cpu.max 不存在 ——
    # 那会把宿主占用当成容器的，与原 `/proc/stat` 语义混淆。宁可回退。
    if quota:
        u0 = _read_cgroup_cpu_usage()
        if u0 is not None:
            time.sleep(0.05)
            u1 = _read_cgroup_cpu_usage()
            if u1 is not None and u1 > u0:
                cores_used = (u1 - u0) / 1e6 / 0.05   # 采样窗内的平均核心数
                return min(100.0, 100.0 * cores_used / quota)

    # --- 回退：宿主 /proc/stat（非容器环境）---
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        vals = [int(x) for x in parts[1:11]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        total = sum(vals)
        time.sleep(0.05)
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts2 = f.readline().split()
        vals2 = [int(x) for x in parts2[1:11]]
        idle2 = vals2[3] + (vals2[4] if len(vals2) > 4 else 0)
        total2 = sum(vals2)
        dt = total2 - total
        if dt <= 0:
            return None
        return 100.0 * (1.0 - (idle2 - idle) / dt)
    except (OSError, ValueError, IndexError):
        return None


def _throttle(t_work: float) -> None:
    """按 CPU 占空比目标休眠，让闲时挖掘不超过 `CPU_LIMIT_PCT`。

    ## 口径 = 「占挖掘进程可用配额的百分比」，见 `CPU_LIMIT_PCT` 的模块级说明

    简版：`sys_pct` 与 `CPU_LIMIT_PCT` **共用同一个分母**（容器内 = cgroup 配额核数，
    非容器 = 整机核数），所以比较是自洽的。不要把它读成「整机固定 60%」。

    限流的真实职责是「别让前台被挤」，不是「把单线程压到 60%」。

    第一版把 `r` 当成「本进程可用的核心比例」，算出
    `t_sleep ≥ 0.667·t_work`，等于把进程压到**单核 60%**，比目标严了一个数量级。
    实测代价：单候选墙钟 0.05s 拖到 0.29s，吞吐仅 2.9 候选/秒，1 天窗口完不成。

    ## 判据

    只在**余量不足一个核**时才让路：

        one_core_pct = 100 / n_cores            # 宿主 4 核 → 25%
        headroom = CPU_LIMIT_PCT - sys_pct      # 还剩多少百分点
        headroom >= one_core_pct  → 不让路     # 余量够本线程吃满，直接跑
        headroom <  one_core_pct  → 按缺口比例休眠

    例（非容器）：已用 50%、上限 60% ⇒ 余量 10 < 25 ⇒ 让路；
                   已用 10% ⇒ 余量 50 ≥ 25 ⇒ 全速跑。

    **注意一个已知的保守偏差**：`one_core` 用宿主核数算（4 核 → 25 点），
    而容器内分母是 cgroup 配额 2.4 核（真正的「一个核」≈ 41.7 点）。
    所以容器里会**更早**判成「余量不足」→ 更保守。方向安全，详见模块级说明。
    """
    if t_work <= 0:
        time.sleep(BASE_SLEEP)
        return

    one_core = 100.0 / _cpu_count()
    sys_pct = _system_cpu_pct()

    if sys_pct is not None:
        if sys_pct >= CPU_LIMIT_PCT:
            # 整机已到/超限额：本批不干活，等它降下来
            time.sleep(CPU_SAMPLE_WINDOW)
            return
        headroom = CPU_LIMIT_PCT - sys_pct
        if headroom < one_core:
            # 余量已不足一个核：按缺口比例让路（最多压到不干活的一半）
            frac = headroom / one_core              # 0..1
            need = t_work * (1.0 - frac) / max(frac, 0.05)
            time.sleep(min(max(need, 0.0), 30.0))
            return
    # 读不到整机占用时按「余量充足」处理 —— 仍有 cgroup 2.4 核硬限兜底

    # 余量充足，但若有在线请求仍让一小档，保证前台响应延迟不劣化
    if _online_pressure():
        time.sleep(PRESSURE_SLEEP)
    else:
        time.sleep(BATCH_GAP)

#: 候选空间版本。2 = 生成式开放空间（单因子网格 + 跨族交叉）。
#: 旧断点（1 = 封闭的 124 维遍历）语义不兼容，读到就重置游标。
SPACE_VERSION = 2
#: 只保留 e_value 最高的若干个 monitor 候选（避免断点膨胀到几十 MB）
MAX_MONITOR_TOP = 50

#: 交叉层「族对 + 模式」真实档位数的缓存（V14-A1）。
#: `candidate_space._cross_plan()` 自身已有缓存，这里只是把换算结果留住，
#: 避免每轮每个候选都重建一次 dict。
_FAMILY_PAIRS_CACHE: Optional[Dict[str, int]] = None


def _fresh_state() -> Dict[str, Any]:
    """断点的初始结构（新空间语义）。"""
    return {
        "space_version": SPACE_VERSION,
        "cursors": {"single": 0, "cross": 0},
        "by_decision": {},
        "candidates": {},
        "usable": {},
        "monitor_top": {},
        "n_checked": 0,
        "runs": 0,
        "total_candidates": 0,
        "updated_at": None,
    }


def _migrate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """把旧版断点迁移到当前候选空间语义。

    旧断点用 `completed_dimensions`（124 个「族:参数」字符串）记录进度。
    新空间是**开放**的且层内顺序不同 —— 旧的已完成集合无法映射成游标，
    强行保留会导致「以为挖过、其实没挖」。因此版本不符时重置游标，
    只保留 runs 等无害计数，并留一条 `migrated_from` 说明。
    """
    if int(state.get("space_version") or 0) >= SPACE_VERSION:
        return state
    old_checked = len(state.get("completed_dimensions") or {})
    fresh = _fresh_state()
    fresh["runs"] = int(state.get("runs") or 0)
    fresh["migrated_from"] = {
        "space_version": int(state.get("space_version") or 1),
        "old_completed_dimensions": old_checked,
        "note_zh": "候选空间已改为生成式开放空间，旧进度不兼容，游标重置",
    }
    return fresh


def enabled() -> bool:
    """总开关。**默认关闭** —— 新机制先观察再开。"""
    return (os.environ.get("LOTT_IDLE_MINING") or "0").strip() == "1"


def state_path() -> Path:
    return Path(config.DATA_DIR) / "idle_mining_state.json"


def load_state() -> Dict[str, Any]:
    """读断点。坏文件一律当作"从头开始"，不让它卡死整个调度。"""
    try:
        p = state_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (ValueError, OSError):
        pass
    return _fresh_state()


def save_state(state: Dict[str, Any]) -> None:
    """原子写断点（先写 tmp 再 replace），避免半写坏文件。"""
    try:
        p = state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        state["updated_at"] = time.time()
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(p))
    except OSError as e:
        print(f"[idle] 断点保存失败: {e}")


def _online_pressure() -> bool:
    """是否有在线请求在跑（用于决定让路力度）。

    读共享连接不现实（会与请求抢锁），这里用**进程线程数**做轻量代理：
    uvicorn 的 worker 线程在活跃时会增加。够用且零成本。
    """
    try:
        import threading
        return threading.active_count() > 4
    except Exception:      # noqa: BLE001
        return False


def judge_candidate(z_scores: List[float], *,
                    alpha: float = eprocess.DEFAULT_ALPHA,
                    tau: float = eprocess.DEFAULT_TAU) -> Dict[str, Any]:
    """对单个候选给出 e-process 判定（委托给 `eprocess.verdict`）。

    本函数存在的意义是**固定入口**：所有闲时挖掘的判定都走这里，
    将来换统计方法只需改这一处，不必在调度逻辑里找散落的判断。
    """
    v = eprocess.verdict(z_scores, tau=tau, alpha=alpha, min_obs=MIN_OBS)
    v["source"] = "idle_mining"
    return v


def run_once(*, budget_candidates: int = 200,
             min_start: int = 300) -> Dict[str, Any]:
    """跑一轮闲时挖掘（可中断、可续跑、**只挖没挖过的**）。

    `budget_candidates` 是**本轮上限**（不是总量）：闲时挖掘应当"细水长流"，
    每次只推进一点，避免一次跑几个小时把机器占满。

    ## 与旧实现的关键区别：从「遍历封闭集合」改为「推进开放空间」

    旧实现遍历 `discovery.FAMILY_BY_KEY` 里写死的 124 个组合，跑完即
    `processed=0` 永久空转 —— 那是**重复验证**，不是挖掘。
    现在改为在 `candidate_space` 的**开放空间**上按游标单向推进：
    单因子 692 项 → 跨族交叉 406 368 项，且交叉层是现有 29 族
    **从未检验过**的搜索空间。

    ## 为什么不做「回头复检 monitor」

    单向推进意味着一个候选只判定一次，monitor 不再回头观察。
    这是刻意的取舍：用户的诉求是「去找新的、之前没检验过的规律」，
    而不是反复验证同一批。回头复检属于另一件事（数据增长带来的持续观察），
    若要做应作为独立的复检队列，不该占用挖掘预算。

    返回运行摘要（也写进断点）。
    """
    if not enabled():
        return {"ok": False, "reason": "idle_mining_disabled",
                "note_zh": "闲时挖掘未启用（LOTT_IDLE_MINING=0）"}

    from . import candidate_space as CS, discovery

    draws = db.load_draws()
    if len(draws) < min_start + 50:
        return {"ok": False, "reason": "insufficient_data",
                "note_zh": f"数据不足（需 ≥{min_start + 50} 期，当前 {len(draws)}）"}

    state = _migrate_state(load_state())
    cursors: Dict[str, int] = dict(state.get("cursors") or {})
    by_decision: Dict[str, int] = dict(state.get("by_decision") or {})
    # ``candidates`` = 单候选过 α 阈的（尚未做族级校正）；
    # ``usable`` 由轮末 e-BH 从 candidates 里筛出，不在这里累加。
    candidates: Dict[str, Any] = dict(state.get("candidates") or {})
    monitor_top: Dict[str, Any] = dict(state.get("monitor_top") or {})
    n_checked = int(state.get("n_checked") or 0)

    n_draws = len(draws)
    # D1（2026-10-20 四专家论证裁决）：**全史增长窗口，不再截断 500 期**。
    #
    # 旧实现 `scan_window = min(500, ...)` 有两个独立缺陷：
    #   1) **统计效力**：500 期只能检出 Δ≥0.119（+10.9%）的效应，全史 Δ≥0.048
    #      （+4.4%）；对 Δ=0.05 的长周期效应，500 期检出率 6.5%、全史 56.6%（8.7×）。
    #   2) **正确性**：滑窗每次重启 = 每次都算「新检验」，Ville 不等式的
    #      anytime-valid 保证**不覆盖跨窗口复查**。实测滑窗 W=500、每 30 期看一次
    #      的纯噪声误报率 = **7.80% > α=0.05**（保证被破坏）；累积增长窗口 = **2.46%**（受控）。
    # 真实数据佐证：`sum_range{45-75,w65}` 在滑窗下 e=12.4，累积下仅 1.83 —— 滑窗高估 6.8×。
    #
    # 增长窗口才是 e-process 超鞅的标准归宿：t 持续增长，效应项按 O(t) 累积、
    # 惩罚项仅 O(log t)，这正是「长周期规律一旦存在就该被找到」的数学基础。
    quick_start = min_start

    processed = 0
    skipped = 0
    interrupted = False
    layers_done: List[str] = []
    max_skip = CS.layer_size(CS.LAYER_SINGLE) + CS.layer_size(CS.LAYER_CROSS) + 8
    batch_cpu_start = time.process_time()

    while processed < budget_candidates and skipped < max_skip:
        layer = _next_layer(cursors, CS)
        if layer is None:
            layers_done = [CS.LAYER_SINGLE, CS.LAYER_CROSS]
            break

        idx = int(cursors.get(layer) or 0)
        if idx >= CS.layer_size(layer):
            cursors[layer] = CS.layer_size(layer)
            continue

        got = CS.candidate_at(layer, idx)
        cursors[layer] = idx + 1        # 游标无论如何前进（不重复挖同一个索引）
        if not got:
            skipped += 1                # 非法候选：不消耗预算
            continue

        flt, fp = got
        try:
            scan = discovery.asof_scan(draws, flt, start=quick_start, end=n_draws)
            obs = scan.get("obs") or []
            exp = scan.get("exp") or []
            if len(obs) >= MIN_OBS and len(obs) == len(exp):
                # **逐候选按筛选集大小 k 定标**（D3，2026-10-21 生产事故修复）。
                #
                # 旧代码写死 sigma=0.8678（红球 k=6 时的标准差）/0.24（蓝球 k=1）。
                # 单因子候选 k 恰在 6 附近，问题没暴露；但交叉层的并集
                # 会让 k 涨到 10~15，此时真实 sigma=1.1862，而代码仍按
                # 0.8678 去除 ⇒ z 被系统性放大 1.37 倍 ⇒ e 值高估 290 倍
                # ⇒ 31 条 `hot ∪ sum_range` 被误判为「可用」。
                z_scores = eprocess.z_scores_from_scan(scan, flt.get("target") or "red")
                v = judge_candidate(z_scores)
                v["layer"] = layer
                v["index"] = idx
                v["label"] = flt.get("name_zh") or flt.get("key")
                v["target"] = flt.get("target")
                d = v.get("decision") or "unknown"
                by_decision[d] = by_decision.get(d, 0) + 1
                if d == "usable":
                    # **单候选过阈只是「候选」，不是「可用」** ——
                    # e-process 的 anytime-valid 只对单个假设负责，
                    # 族级假发现率必须由 e-BH 在轮末统一治理（见 _recompute_usable）。
                    candidates[fp] = v
                else:
                    _keep_monitor_top(monitor_top, fp, v)
            else:
                d = "underpowered"
                by_decision[d] = by_decision.get(d, 0) + 1
        except Exception as e:  # noqa: BLE001 —— 单个候选失败不中断整轮
            d = "error"
            by_decision[d] = by_decision.get(d, 0) + 1
            monitor_top.pop(fp, None)

        n_checked += 1
        processed += 1
        if processed % BATCH_SIZE == 0:
            # 闭环限流：按本批真实 CPU 时间算应休眠多久（用户约束：整机 ≤60%）
            t_now = time.process_time()
            _throttle(t_now - batch_cpu_start)
            batch_cpu_start = time.process_time()

    if processed >= budget_candidates:
        interrupted = True

    # 轮末统一做族级校正：把「候选」里经 e-BH 仍成立的升级为「证据项」。
    #
    # **V14-A4 命名改造**：这个内部变量原来叫 `usable`，与 `patterns_runtime`
    # 的生产快照字段 `counts.usable`（= 长期可用规律数）**同名不同义**：
    #
    #   idle_mining  的 usable = 通过本层 e-BH 的「挖掘证据项」，**不进快照、不影响出号**
    #   patterns_runtime 的 usable = 生产快照里"可回放且可用"的规律，是出号门控依据
    #
    # 实测两边同时为 0 纯属巧合：本轮修复前 idle_mining 曾有 19~20 条过阈候选
    # （`by_decision.usable=20`）而 `patterns_runtime.usable=0`。
    # 用户若看到「20 条可用」与「0 条可用」并存，会直接认定系统造假 ——
    # 哪怕每个数字单独看都对。这是"同一系统两个 usable"的真实风险。
    #
    # 现改名为 `rule_evidence`（证据项），并在契约里保留 `usable` 作为**弃用别名**
    # 只是为了让旧调用方不报错；新代码一律读 `rule_evidence`。
    rule_evidence = _recompute_usable(candidates, n_checked)

    state.update({
        "cursors": cursors,
        "by_decision": by_decision,
        "candidates": candidates,
        "rule_evidence": rule_evidence,
        # 兼容旧断点读取方（前端/脚本）—— 值相同但语义已由新字段承载
        "usable": rule_evidence,
        "monitor_top": monitor_top,
        "n_checked": n_checked,
    })
    state["runs"] = int(state.get("runs") or 0) + 1
    state["total_candidates"] = n_checked
    save_state(state)

    remain_single = max(0, CS.layer_size(CS.LAYER_SINGLE)
                        - int(cursors.get(CS.LAYER_SINGLE) or 0))
    remain_cross = max(0, CS.layer_size(CS.LAYER_CROSS)
                       - int(cursors.get(CS.LAYER_CROSS) or 0))
    return {
        "ok": True,
        "processed": processed,
        "completed_total": n_checked,
        # 权威字段（新）
        "rule_evidence": len(rule_evidence),
        # 弃用别名：仅兼容旧日志/脚本，**不得**与 patterns_runtime.usable 并列展示
        "usable": len(rule_evidence),
        "usable_deprecated": True,
        "candidates": len(candidates),
        "interrupted": interrupted,
        "layers_done": layers_done,
        "remaining": {"single": remain_single, "cross": remain_cross},
        "cursors": dict(cursors),
        "note_zh": (f"本轮判定 {processed} 个**此前未检验过**的候选"
                    f"（累计 {n_checked}）；单候选过阈 {len(candidates)} 条，"
                    f"经 e-BH 族级校正后得到证据项 {len(rule_evidence)} 条。"
                    f"剩余：单因子 {remain_single} + 跨族交叉 {remain_cross}。"
                    f"（证据项是「挖掘层的统计结论」，不是出号依据 —— "
                    f"出号门控看 patterns_runtime.usable_replayable。）"),
    }


def _recompute_usable(candidates: Dict[str, Any],
                      n_checked: int) -> Dict[str, Any]:
    """用 **分层 e-BH** 对「单候选过阈」的候选做族级 FDR 校正。

    ## 为什么要分层

    交叉层有 72350 个候选。若整层共用一个 m，k=1 的阈值 = ``m/α`` 会大到
    任何真实数据都够不到（``72656/0.05 = 145 万``）。分层把 m 降到「一个族对 ×
    一种模式」的量级，才让 e-BH 在真实数据上有分辨力。

    ## V14-A1 修正（2026-10-21，团队四人共同指出的缺陷）

    旧实现有两处让 m 失真，**方向都是把 m 算小**：

    1. **分组键塌陷**：``key = "cross|" + parts[1]`` 只取 mode，实测只产生 2 个组
       （`cross|0` = 11079、`cross|1` = 26793），而设计意图是 268 族对 × 2 模式。
       现在用 `_group_key_of()` 取「mode + 族A + 族B」四段。
    2. **m 用代数常数**：交叉层硬编码 135（其实是该分布的**均值**），
       单因子层硬编码 306。实测族对档位数中位只有 35，78% 的族对低于 135。
       现在 `_group_checked()` 逐组查 `candidate_space` 真源。

    修正后的量级变化（以 `cross|1` 为例）：m 从 135 → 26793，阈值从 2700 → 535860，
    **低估 198 倍的部分被消除**。这不保证「能选出东西」（数据本身无信号），
    但保证 m 不再是凭空捏造的数字。

    ## m 的记账原则（SPACE §6.2，三条硬规则，未变）

      1. m **单调递增、永不复位**（复位 = 重置式作弊，α 会被放大）
      2. 指纹不含期号（否则同一假设会重复计入 m）
      3. `usable` 每轮**全量重算**而非并集（否则旧的假阳性永远洗不掉）

    ## 输入口径（统计审查员更正的一条，必须写清）

    `candidates` 按设计只包含**单候选过 α 阈**（e ≥ 1/α）的候选，不是全部检验。
    这不违反 e-BH：`ebh_selected` 的判据是 ``e_(k) >= m/(k·α)``，而
    ``e < 1`` 的候选永远不会被选中（阈值最小也有 ``1/α``），所以只传 e ≥ 1 的候选
    与传全部候选**结果等价**，且省内存。真正决定成败的是 **m 是否取对**。

    实测（`data/idle_mining_state.json`，20 条过阈）：`cross|1` 组内有 18 条候选，
    `ebh_selected` 会逐个 rank 比较（rank 能走到 18），
    但**每一档都没能越过门槛**，因此 ``k_star = 0``、返回空集 ——
    这才是 `usable` 为 0 的真实机制，不是「rank 恒为 1」。
    """
    if not candidates:
        return {}

    groups: Dict[str, List[str]] = {}
    for fp in candidates:
        groups.setdefault(_group_key_of(fp), []).append(fp)

    selected: Dict[str, Any] = {}
    for gkey, fps in groups.items():
        e_vals = [float((candidates[fp] or {}).get("e_value") or 0.0) for fp in fps]
        m_group = _group_checked(gkey, n_checked, len(fps))
        sel = eprocess.ebh_selected(e_vals, m_total=m_group)
        for i in sel:
            v = dict(candidates[fps[i]])
            v["ebh_group"] = gkey
            v["ebh_m"] = m_group
            v["ebh_threshold_k1"] = m_group / eprocess.DEFAULT_ALPHA
            selected[fps[i]] = v
    return selected


def _family_pairs() -> Dict[str, int]:
    """交叉层每个「族对 + 模式」组的真实档位数（= 该组应记的 m）。

    ## 为什么必须问 `candidate_space` 而不是用常数（V14-A1，2026-10-21）

    旧实现用两个字面量：单因子 `min(n_checked, 306)`、交叉 `max(n_in_group, 135)`。

    * `306` 恰好等于单因子层真实规模（29 族档位总和实测 306），**它对**；
    * `135` 是**该分布的均值，不是每个组的值**。实测族对档位数
      min / 中位 / mean / max = **1 / 35 / 135 / 2304**，且 **78%（210/268）的族对低于 135**。

    把均值当常数用，代价全落在低于均值的组上：`repeat×repeat2` 真实 m=1 被放大 **135 倍**。
    该组的阈值于是从 `1/α = 20` 被抬到 `135/α = 2700`，任何真实效应都过不去。

    ## 返回什么

    ``{"cross|<mode_idx>|<族A>|<族B>": 档位数, ...}``，键与 `_group_key_of(fp)` 一致。
    档位数取 ``len(A 的档位) × len(B 的档位)`` —— 与 `probe_cross`/`_cross_plan` 同源。
    """
    global _FAMILY_PAIRS_CACHE
    if _FAMILY_PAIRS_CACHE is not None:
        return _FAMILY_PAIRS_CACHE

    out: Dict[str, int] = {}
    try:
        from . import candidate_space as CS
        pairs, _order = CS._cross_plan()
        singles = CS._single_index()
        for ia, ib, cnt in pairs:
            # 用两侧首档取族名：同族对内部族名唯一，取任一样本即可
            ka = singles[ia[0]][0] if ia else "?"
            kb = singles[ib[0]][0] if ib else "?"
            for mi in range(len(CS.CROSS_MODES)):
                out[f"cross|{mi}|{ka}|{kb}"] = int(cnt)
    except Exception:            # noqa: BLE001 —— 结构探测失败不应拖垮判定
        out = {}
    _FAMILY_PAIRS_CACHE = out
    return out


def _group_key_of(fp: str) -> str:
    """把候选指纹映射到「分层 e-BH 的组键」。

    ## 旧实现的塌陷（V14-A1，团队四人共同指出）

    旧代码：``key = "cross|" + parts[1]`` —— ``parts[1]`` 只是模式下标（0/1），
    而 docstring 写的是「按族对+模式各一组」。实测结果：**整个交叉层只有 2 个组**
    （`cross|0` 有 11079 个候选、`cross|1` 有 26793 个），
    与「268 族对 × 2 模式 = 536 组」的意图相差 268 倍。

    后果：26793 个候选被塞进 1 个组，m 只记 135 ⇒ 阈值低估 **198 倍**
    （应为 ``26793/α = 535860``，实际只有 ``135/α = 2700``）。

    ## 现在的规则

    指纹形如 ``cross|<mode>|<族A>:<params>|<族B>:<params>``，
    组键取前四段（mode + 两个族名，**不含具体参数**）——
    这才对应「一个族对 × 一种模式」这一自然层。单因子层仍是单独一组。
    """
    if fp.startswith("single|"):
        return "single"
    parts = fp.split("|")
    if len(parts) < 4:
        return "cross|" + (parts[1] if len(parts) > 1 else "?")
    fa = parts[2].split(":", 1)[0]
    fb = parts[3].split(":", 1)[0]
    return f"cross|{parts[1]}|{fa}|{fb}"


def _group_checked(gkey: str, n_checked: int, n_in_group: int) -> int:
    """某一组的检验总数（= 分层 e-BH 的 m）。

    ## 原则：m 取「这一组一共检验了多少档位」，不取任何代数常数

    * **单因子层**：真实规模 = ``candidate_space.layer_size("single")``
      （实测 306 = 29 族档位之和）。这里直接用真源，不用字面量。
    * **交叉层**：按「族对 + 模式」查 ``_family_pairs()`` 的真实档位数
      （中位 35、均值 135、max 2304 —— **没有单一常数能代表它**）。

    兜底：查不到组时退回 ``max(n_in_group, 1)``。**绝不退回 135 这类常数** ——
    那正是本缺陷的成因。宁可 m 偏小被下游 FDR 检查发现，也不要偷偷放大 135 倍。

    `n_checked` 参数保留只为兼容既有调用/测试签名，**不再参与计算**：
    它曾是 `min(n_checked, 306)` 的输入，而那正是「用全局累计截断成常数」的老毛病。
    """
    if gkey == "single":
        try:
            from . import candidate_space as CS
            return max(int(CS.layer_size(CS.LAYER_SINGLE)), n_in_group, 1)
        except Exception:        # noqa: BLE001
            return max(n_in_group, 1)

    real = _family_pairs().get(gkey)
    if real:
        # 组内已过阈数不可能超过该组总档位；取 max 防御指纹口径漂移
        return max(int(real), n_in_group, 1)
    return max(n_in_group, 1)


def _next_layer(cursors: Dict[str, int], CS) -> Optional[str]:
    """按 single → cross 的顺序找出下一个**尚有余量**的层。"""
    for layer in (CS.LAYER_SINGLE, CS.LAYER_CROSS):
        if int(cursors.get(layer) or 0) < CS.layer_size(layer):
            return layer
    return None


def _keep_monitor_top(top: Dict[str, Any], fp: str, v: Dict[str, Any]) -> None:
    """保留 e_value 最高的若干 monitor 候选，供「最接近发现」展示。

    只留 top-N：monitor 是绝大多数（几十万），全存会让断点膨胀到几十 MB。
    """
    if v.get("decision") != "monitor":
        return
    top[fp] = {k: v.get(k) for k in ("e_value", "n_obs", "label", "layer", "index")}
    if len(top) > MAX_MONITOR_TOP:
        worst = min(top, key=lambda k: float(top[k].get("e_value") or 0.0))
        top.pop(worst, None)



def status() -> Dict[str, Any]:
    """供 /api/health 与运维查看的进度快照。

    关键字段是 **`progress`（开放空间的挖掘进度）** —— 这是「闲时挖掘到哪了、
    还剩多少没挖」的唯一可信来源。旧实现只有一个 `completed_dimensions` 计数，
    看不出空间有多大、是否已经空转。
    """
    from . import candidate_space as CS

    state = _migrate_state(load_state())
    cursors = state.get("cursors") or {}
    space = CS.stats()
    single_done = min(int(cursors.get(CS.LAYER_SINGLE) or 0), space["single_total"])
    cross_done = min(int(cursors.get(CS.LAYER_CROSS) or 0), space["cross_total"])
    checked = int(state.get("n_checked") or 0)
    total_space = space["single_total"] + space["cross_total"]
    return {
        "enabled": enabled(),
        "space_version": SPACE_VERSION,
        # 兼容旧字段名：语义已变为「累计判定过的候选数」
        "completed_dimensions": checked,
        "checked": checked,
        "runs": int(state.get("runs") or 0),
        "total_candidates": checked,
        "by_decision": dict(state.get("by_decision") or {}),
        # **区分「候选」与「证据项」**：前者只是单候选过了 α 阈（数万级搜索里
        # 纯随机也会跑出一批），后者才通过 e-BH 族级校正。两者混为一谈就是骗自己。
        "candidate_count": len(state.get("candidates") or {}),
        # V14-A4：权威字段。旧名 `usable_count` 与 patterns_runtime.usable 撞词，
        # 已弃用（保留只为不打断旧脚本），并列展示会让人以为系统自相矛盾。
        "rule_evidence_count": len(state.get("rule_evidence")
                                  or state.get("usable") or {}),
        "usable_count_deprecated": len(state.get("usable") or {}),
        # ===== D-07：`usable` 一词两义的**机器可读标注** =====
        #
        # 背景：本模块曾输出 `by_decision.usable`（单候选过阈、未做族级校正），
        # 而 `patterns_runtime` 的生产快照有 `counts.usable`（可回放且可用于出号）。
        # 同一系统两个 "usable" 不同值，用户看到即认定造假 —— 哪怕每个数字单独看都对。
        #
        # 本轮**只加标注、先不删**旧字段（删除留到下一版本，届时改名 e_reject_candidates）。
        # 标注的作用：任何消费方只要能读到本段，就必须知道这两个词不是一回事，
        # 且**禁止并列展示**。前端与报表用它做断言，而不是靠文档约定。
        "usable_semantics": {
            "this_module_authoritative_field": "rule_evidence_count",
            "deprecated_aliases": ["usable_count_deprecated", "by_decision.usable"],
            "meaning_zh": (
                "本模块的「证据项」= 单候选过 α 阈后再经 e-BH 族级校正的统计结论；"
                "它**不进生产快照、不影响出号**。"
            ),
            "collides_with": {
                "field": "patterns_runtime.counts.usable",
                "meaning_zh": (
                    "生产快照里的「可用规律」= 可回放且被认定可用于出号的规律，"
                    "是出号门控依据。与本节字段**同名不同义**。"
                ),
            },
            "never_show_together": True,
            "reason_zh": (
                "两者数值不同是**正确**的（一个是「有证据」，一个是「能出号」）。"
                "但并列展示会让用户以为是两个互相矛盾的数字，"
                "从而怀疑整个系统的诚实性。若要同时说明，必须明确区分口径，"
                "不得只说「可用 20 条 / 可用 0 条」。"
            ),
            "rename_plan": "下一版本将 `usable` 弃用别名改名为 `e_reject_candidates`。",
        },
        "progress": {
            "single_done": single_done,
            "single_total": space["single_total"],
            "cross_done": cross_done,
            "cross_total": space["cross_total"],
            "space_total": total_space,
            "remaining": max(0, total_space - single_done - cross_done),
            "pct": round(100.0 * (single_done + cross_done) / total_space, 4)
            if total_space else 0.0,
        },
        "monitor_top": sorted(
            (state.get("monitor_top") or {}).items(),
            key=lambda kv: -float((kv[1] or {}).get("e_value") or 0.0))[:5],
        "updated_at": state.get("updated_at"),
        "migrated_from": state.get("migrated_from"),
    }
