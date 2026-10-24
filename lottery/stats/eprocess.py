"""e-process 判定器（P4 闲时挖掘的统计核心）。

## 为什么闲时挖掘**必须**用 e-process，而不是固定样本量的校正

"闲时全天候挖掘"意味着**反复看数据、随时可能停止**。这在统计上叫
optional continuation —— 而固定样本量的任何校正（包括 Bonferroni）都会被
"看多少次"摧毁：

* 每天看一次、连续看 30 天，α=0.05 的检验实际错误率约 **1-(1-0.05)^30 ≈ 78%**；
* 全量 Bonferroni 也救不了，因为它假设"检验次数是**事先固定**的"，
  而闲时挖掘的次数取决于数据长什么样（看到有苗头就多看几次）。

e-process 是专为这种场景设计的：它对**任意停止规则**都有效
（Ville 不等式：`P(sup_t M_t >= 1/α) <= α`），看多少次都不膨胀。

## 数学构造（标准混合超鞅，Howard et al. 2021）

观测 `X_i` 标准化为 `z_i`（H0 下 ≈ N(0,1)），`S_t = Σ_{i<=t} z_i`，
先验 `θ ~ N(0, τ²)`：

    M_t = 1/sqrt(1+τ²t) · exp( τ²S_t² / (2(1+τ²t)) )

    E_H0[M_t] <= 1,    P_H0(∃t: M_t >= 1/α) <= α

`α=0.05` ⇒ 阈值 `1/α = 20`，`log 20 = 2.996`。

## τ 的选择

τ 决定"对哪个效应量最敏感"。M2 §4.3 用实测效应（σ=0.8678）标定后建议
**τ = 0.10**：在"小效应慢一点、大效应快一点"之间最平衡。

    cold +10%/号（Δ=0.054）→ 约 2490 期首次穿越
    cold +50%/号（Δ=0.135）→ 约 400 期

## 这个模块的边界

**只做数学，不碰 IO、不碰 DB、不碰配置**。这样它能被独立测试到
（"注入偏置必检出 + 纯随机不误报"是项目的一贯验收口径），
调度层再怎么改都不会动到判定逻辑。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

#: 默认先验尺度（M2 §4.3 标定结果）
DEFAULT_TAU = 0.10
#: 默认显著性水平
DEFAULT_ALPHA = 0.05


def log_e_value(z_scores: Sequence[float], *, tau: float = DEFAULT_TAU) -> float:
    """返回 log M_t（对数域，避免长序列下溢）。

    `z_scores` 是已标准化的观测值序列。空序列返回 0（= M_0 = 1，未积累证据）。
    """
    t = len(z_scores)
    if t <= 0:
        return 0.0
    s = float(sum(z_scores))
    tau2 = tau * tau
    denom = 1.0 + tau2 * t
    # log M_t = -0.5*log(1+τ²t) + τ²S_t²/(2(1+τ²t))
    return -0.5 * math.log(denom) + (tau2 * s * s) / (2.0 * denom)


def e_value(z_scores: Sequence[float], *, tau: float = DEFAULT_TAU) -> float:
    """返回 M_t（非对数域）。长序列可能溢出为 inf，判断用 `verdict()`。"""
    return math.exp(log_e_value(z_scores, tau=tau))


def threshold(alpha: float = DEFAULT_ALPHA) -> float:
    """拒绝 H0 的阈值 `1/alpha`。"""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0,1) 之间")
    return 1.0 / alpha


def is_significant(z_scores: Sequence[float], *, tau: float = DEFAULT_TAU,
                   alpha: float = DEFAULT_ALPHA) -> bool:
    """e-value 是否已越过 `1/alpha`（anytime-valid 的拒绝判据）。"""
    return log_e_value(z_scores, tau=tau) >= math.log(threshold(alpha))


def first_crossing(z_scores: Sequence[float], *, tau: float = DEFAULT_TAU,
                   alpha: float = DEFAULT_ALPHA) -> Optional[int]:
    """首次越过阈值的期数（1-based）；从未越过返回 None。

    用于回答"这个效应多快能被发现"，也是标定 τ 的工具。
    """
    log_thr = math.log(threshold(alpha))
    tau2 = tau * tau
    s = 0.0
    for i, z in enumerate(z_scores, start=1):
        s += float(z)
        denom = 1.0 + tau2 * i
        log_m = -0.5 * math.log(denom) + (tau2 * s * s) / (2.0 * denom)
        if log_m >= log_thr:
            return i
    return None


def verdict(z_scores: Sequence[float], *, tau: float = DEFAULT_TAU,
            alpha: float = DEFAULT_ALPHA,
            min_obs: int = 30) -> Dict:
    """给出可审计的判定（供 L-C 报告层直接渲染）。

    返回字段刻意与 M2 §4.2 的报告层要求对齐 —— **缺任一字段前端应拒绝渲染**：
    `e_value` / `log_e_value` / `alpha_spent` / `n_obs` / `threshold` / `decision`。
    """
    n = len(z_scores)
    log_m = log_e_value(z_scores, tau=tau)
    thr = threshold(alpha)
    sig = log_m >= math.log(thr)
    if n < min_obs:
        decision = "underpowered"        # 样本不足，不下结论（也**不**当作"有效"）
    elif sig:
        decision = "usable"
    else:
        decision = "monitor"             # 尚无证据，但保持监控（不是"否决"）
    return {
        "e_value": math.exp(log_m) if log_m < 700 else float("inf"),
        "log_e_value": log_m,
        "alpha_spent": alpha if sig else 0.0,
        "n_obs": n,
        "threshold": thr,
        "tau": tau,
        "decision": decision,
        "note_zh": (
            "样本不足，不下结论" if decision == "underpowered"
            else "e-value 越过 1/α，可在任意停止规则下拒绝 H0" if sig
            else "尚无足够证据；e-process 允许继续观察而不膨胀错误率"),
    }


def expected_crossing_issues(delta: float, sigma: float = 0.8678,
                             *, tau: float = DEFAULT_TAU,
                             alpha: float = DEFAULT_ALPHA) -> Optional[int]:
    """给定真实效应，估算首次穿越所需的期数（用于 `planned_mde` 与资源规划）。

    `delta` 是相对基线的绝对效应（如 +0.054 表示每号多 0.054 次命中）。
    这是近似解析估算，用于规划，不是判定依据。
    """
    if sigma <= 0:
        return None
    zbar = abs(delta) / sigma
    if zbar <= 1e-9:
        return None
    # 用首达时间的粗略近似：E[M_t] 的下界 exp(τ²(zbar·t)²/(2(1+τ²t))) 达 1/α
    # ⇒ τ²(zbar·t)²/(2(1+τ²t)) ≈ log(1/α)
    # 解一元二次可得，这里直接用数值搜索，稳健且够快
    log_thr = math.log(threshold(alpha))
    tau2 = tau * tau
    for t in range(1, 200000):
        s_over_denom = tau2 * (zbar * t) ** 2 / (2.0 * (1.0 + tau2 * t))
        if s_over_denom - 0.5 * math.log(1.0 + tau2 * t) >= log_thr:
            return t
    return None


# --------------------------------------------------------------------------
# 多重检验治理：e-BH
# --------------------------------------------------------------------------

def ebh_selected(e_values: Sequence[float], m_total: Optional[int] = None,
                 alpha: float = DEFAULT_ALPHA) -> Set[int]:
    """e-BH（Wang & Ramdas 2022）：在**大量并行假设**下控制 FDR。

    ## 为什么必须有这一层

    e-process 的 anytime-valid 保证是**针对单个假设**的：一个假设被反复查看、
    随时停止，错误率都不膨胀。但它**不对「同时检验 40 万个不同假设」负责** ——
    那是族错误率（FWER）/ 假发现率（FDR）的问题，单个 e-process 无能为力。

    闲时挖掘把候选空间从 124 扩到数万（单因子网格 + 跨族交叉）之后，这一层
    就不再是可选项：纯随机数据上必然有一定数量的候选「e-value 越过 1/α」，
    直接把它们报成「找到规律」就是在骗自己。实测抓到过 4 条这样的假阳性
    （`hot{window:5,k:16}` 等），这正是本函数存在的理由。

    ## 判据

    把 m 个 e-value 降序排列 ``e_(1) ≥ … ≥ e_(m)``，取最大的 k 使
    ``e_(k) ≥ m / (k·α)``，则前 k 个被选中。此时 FDR ≤ α。

    注意阈值随 k 变化：k 越大阈值 ``m/(kα)`` 越小。因此 ``e < 1`` 的候选
    永远不会被选中（阈值最小也有 ``1/α``），调用方可以只传 ``e ≥ 1`` 的候选，
    用 ``m_total``（**总检验数**，含未入选者）保证分母正确。

    ## 已知局限（必须说清楚，实测数据支撑）

    **m 越大，单个发现所需的 e-value 越高（≈ m/α）。** 这不是实现缺陷，
    而是多重检验的固有代价 —— 但与本项目的数据现实结合后，结论很尖锐：

    * 验收口径里的强注入偏置「cold +50%/号」，600 期单候选 e-value 实测 **28.1**；
    * 而 m=44（一个中等族的档位数）时阈值已达 880，m=306 时 6120。

    也就是说：**只要做了族级校正，连强注入偏置都检不出。** 因此
    `idle_mining` 把「单候选过阈」与「通过 e-BH」分成两级分别记录
    （`candidates` / `usable`），前端必须分开呈现，不能让未经校正的
    候选冒充结论。`usable` 长期为 0 是这张数据上的**数学必然**，
    3516 期真实开奖与均匀随机不可区分（红球卡方 p=0.061）。

    想放宽只能在**减小搜索空间**（少挖几个维度）上做，不能靠调小 m ——
    调小 m 等于偷偷降低显著性门槛，会直接制造假发现。
    """
    m = int(m_total) if m_total is not None else len(e_values)
    if m <= 0 or not e_values:
        return set()
    alpha = float(alpha) if alpha else DEFAULT_ALPHA
    order = sorted(range(len(e_values)), key=lambda i: -float(e_values[i]))
    k_star = 0
    for rank, idx in enumerate(order, start=1):
        e = float(e_values[idx])
        if e >= m / (rank * alpha):
            k_star = rank
    return set(order[:k_star])


#: 红球/蓝球号码池大小（双色球规则：红球 33 选 6，蓝球 16 选 1）
R_MAX = 33
R_PICK = 6
B_MAX = 16

#: 历史遗留的固定标准差（k=6 时的红球值）。**不要在新代码里用它做 z 标准化。**
#: 保留命名只为兼容既有引用；新代码请用 `sigma_hypergeom()`。
SIGMA_RED_K6 = 0.8678


def sigma_hypergeom(k: float, target: str = "red") -> float:
    """给定筛选集大小 ``k``，返回「命中数」的超几何标准差。

    ## 为什么必须按 k 变化（2026-10-21 生产事故）

    旧代码对所有候选都用固定 ``sigma = 0.8678``（这是我写在
    ``idle_mining`` 里的常量，取的是 k=6 时的红球标准差）。
    单因子候选的 k 恰好都在 6 附近，所以问题一直没暴露。

    但**交叉层用了并集（union）之后 k 会变成 10~15**：
    实测 ``hot{k=7} ∪ sum_range`` 的 k_mean = **11.21**，
    此时真实标准差是 **1.1862**，而代码仍按 0.8678 去除 ⇒
    z 被系统性地放大 **1.37 倍** ⇒ e 值被高估 **290 倍**。

    后果：31 条候选被误判为「可用」，全是同一个
    ``hot ∪ sum_range`` 模板 —— 那不是规律，是方差口径错误的产物。

    ## 公式

    命中数服从超几何分布（从 R_MAX 个号里选 R_PICK 个开奖号，
    我们的筛选集有 k 个号）：

        Var = k · p · (1-p) · (N-k)/(N-1),  p = R_PICK/R_MAX

    有限总体校正项 ``(N-k)/(N-1)`` 不能省 —— k 越大它越接近 1，
    但 k 小时会显著缩小方差（k=1 时 sigma=0.354，比 k=6 的 0.868 小得多）。

    蓝球同理：``Var = k · (1/B_MAX) · (1 - 1/B_MAX)``。
    """
    if k is None or (isinstance(k, float) and k != k):     # NaN
        k = float(R_PICK)
    # V39（2026-10-24 审计 P2）：**边界输入必须显式报错，不能静默钳回 0**。
    #
    # 旧实现把负值钳成 0、把 >R_MAX 的 k 用 `max(fpc,0)` 变成 0 —— 于是
    # k=-5 和 k=50 都"安全地"返回 0，看起来没问题，实际是**调用方传错参数
    # 被悄悄吞掉**。特别危险的是 k>R_MAX：k 一旦超过全集大小，"命中数方差为 0"
    # 在数学上是对的（全选必全中），但那不是调用方想要的"这个筛选集的标准差"，
    # 而是说明筛选集构造出了问题。宁可用 ValueError 让错误可见。
    kf = float(k)
    if target == "blue":
        hi = float(B_MAX)
    else:
        hi = float(R_MAX)
    if not (0.0 <= kf <= hi):
        raise ValueError(
            f"筛选集大小 k={kf!r} 越界：必须在 0..{int(hi)} 之间"
            f"（target={target}）。请检查筛选集的构造 —— "
            "k 超过号码总数通常说明候选生成逻辑有 bug，不是正常边界。")
    k = kf
    if target == "blue":
        p = 1.0 / B_MAX
        var = k * p * (1.0 - p)
    else:
        p = float(R_PICK) / float(R_MAX)
        N = float(R_MAX)
        fpc = (N - k) / (N - 1.0) if N > 1 else 1.0
        var = k * p * (1.0 - p) * max(fpc, 0.0)
    return math.sqrt(var) if var > 0 else 0.0


def z_scores_from_scan(scan: Dict, target: str = "red") -> List[float]:
    """把 ``discovery.asof_scan`` 的结果转成 z 序列，**逐期按当期的 k 定标**。

    ## 四条必须守住的规则（每条都对应一次生产事故）

    ### 1. 按 k 定标，不用固定 sigma

    见 `sigma_hypergeom`：固定 0.8678 遇上并集的大筛选集会低估方差
    1.23 倍，e 值高估 61 倍。

    ### 2. **空期（exp=0）绝不能跳过**

    这是本函数第一版犯的错，代价更大。当时想的是「筛选集为空 ⇒ 该期无证据」，
    于是 `continue` 掉。但空期恰恰是**最强的证据**：它说明筛选条件没触发、
    命中数必然是 0。跳过它等于只统计「触发了的那些期」，是典型的选择偏差。

    实测 `hot{k=7,w=10} ∩ omission_high{k=10}`（k_mean=0.219）：

        跳过空期（n=641）  e = 11 351   ← 假阳性，过阈值 2700
        保留空期（n=3217） e =     1.75  ← 正确

    **高估 6490 倍。** 空期必须作为 z=0 计入序列。

    ### 3. 整段恒空时返回**空序列**，而不是全零序列

    若某候选的回放期里筛选集**恒为空**，那么它不携带任何信息：
    既不能说"有效"也不能说"无效"，应交给调用方判 `underpowered`。
    若这里返回 100 个 z=0，`e_value` 会算出 ≈1（看着"无证据"），
    但它同时让 `n_obs=100` 通过了 `min_obs` 检查 —— 语义上把
    「从未触发」误当成「检验过、无效应」，正是 V12 裁决里点名的
    「从未检验与检验了无效应不可区分」。因此必须返回空。

    ### 4. 每期用**该期自己的 k** 定标，不用全段 k_mean（V14-A2，2026-10-21）

    前三条修好之后仍然错：`sigma_hypergeom(scan["k_mean"])` 算的是**全段平均**
    筛选集大小，而 `k_mean` 把 k=0 的空期也平均进去了。实测：

        2576 空期 + 641 活跃期(k≈6) → k_mean = 1.1955
        σ(k_mean=1.1955) = 0.4204   vs   σ(k=6) = 0.8678
        ⇒ z 放大 2.1 倍；e 的虚高随信号强度急剧放大
        （合成验证：delta=0.10 时 14.6 倍，delta=0.20 时 4.5 万倍）

    commit `38d44ff` 的说明写"逐候选按 k 定标"，但当时 `asof_scan` 不返回逐期 k，
    下游只能归一化到全段均值 —— 所以那次的修复**没有真正生效**。
    现在 `asof_scan` 返回 `k_series`，这里逐期取 `sigma_hypergeom(k_i)`。

    当 `k_series` 缺失（旧断点、外部构造的 scan dict）时，退回 `k_mean` 口径并
    保持与历史一致的行为 —— 兼容性优先，但**新代码路径一律走逐期**。

    ## 关于 n 的含义

    返回序列的长度恒等于回放期数（除非整段 k_mean=0）。
    调用方的 `min_obs` 因此比较的是「回放了多少期」而不是「触发了多少期」，语义正确。
    """
    obs = scan.get("obs") or []
    exp = scan.get("exp") or []
    ks = scan.get("k_series")
    k_mean = float(scan.get("k_mean") or 0.0)

    # 规则 3：整段恒空 ⇒ 无信息，返回空序列交由调用方判 underpowered
    if ks and len(ks) == len(obs):
        if not any(int(k) > 0 for k in ks):
            return []
        sigmas = [sigma_hypergeom(k, target) for k in ks]
    else:
        s = sigma_hypergeom(k_mean, target)
        if s <= 0:
            return []           # 整段筛选集恒为空：该候选无意义
        sigmas = [s] * len(obs)

    out: List[float] = []
    for i, (o, e) in enumerate(zip(obs, exp)):
        e = float(e)
        o = float(o)
        if e <= 0.0:
            # 空期：命中必然为 0，计入 z=0（**不是跳过**）。见规则 2。
            out.append(0.0)
            continue
        sg = sigmas[i] if i < len(sigmas) else 0.0
        if sg <= 0:
            # 该期有期望却没有方差（理论上不该发生）：按无信息处理
            out.append(0.0)
            continue
        out.append((o - e) / sg)
    return out
