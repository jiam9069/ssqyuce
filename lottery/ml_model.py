"""ML 概率模型（M2）：GBDT + 随机森林投票集成 + 概率校准 + 蓝球独立投票。

架构：
- 红球：33 个号码各训练一个 (RandomForest + HistGradientBoosting) 软投票分类器，
  再用 CalibratedClassifierCV(sigmoid) 把原始概率校准为诚实概率；
- 蓝球：16 选 1 拆成 16 个二分类，由 RF + HistGB + 经典 GBDT 三模型投票 + 校准；
- 输出：33 维红球 / 16 维蓝球概率（归一化），供 engine 的 Brier 加权融合使用；
- 评估：walk-forward 滚动评估（每 ML_REFIT_EVERY 期重训一次），对照均匀基线，
  给出 Brier / log-loss / 校准曲线（可靠性图）/ paired 显著性检验。

诚实边界：双色球为独立随机事件，ML 概率只用于结构配平与校准研究，
不承诺提高中奖率。
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from . import config, features as F

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
    HAS_HISTGB = True
except ImportError:
    HAS_SKLEARN = False
    HAS_HISTGB = False


_FEATURE_NAMES = [
    "freq_5", "freq_10", "freq_20", "freq_30", "freq_50", "freq_150",
    "omit_cur", "omit_avg", "omit_ratio",
    "appeared_1", "appeared_2", "appeared_3", "appeared_5", "appeared_10", "appeared_20",
    "zone_1", "zone_2", "zone_3",
    "sum_value", "span_value",
]

_BLUE_FEATURE_NAMES = [
    "freq_5", "freq_10", "freq_20", "freq_30", "freq_50", "freq_150",
    "omit_cur", "omit_avg", "omit_ratio",
    "appeared_1", "appeared_2", "appeared_3", "appeared_5", "appeared_10", "appeared_20",
    "sum_value",
]


# ---------- 特征工程 ----------

def _compute_features(history: List[Dict], number: int) -> List[float]:
    """为单个红球号码计算特征向量（与 M1 一致，保持可复现）。"""
    n = len(history)
    feats = []

    for w in (5, 10, 20, 30, 50, 150):
        sl = F.window_slice(history, w)
        f = F.red_frequency(sl)
        feats.append(float(f[number]))

    om_cur = F.current_omission_red(history)
    om_avg = F.avg_omission(F.red_frequency(history), n)
    feats.append(float(om_cur[number]))
    feats.append(float(om_avg[number]))
    feats.append(float(om_cur[number]) / max(float(om_avg[number]), 0.01))

    for k in (1, 2, 3, 5, 10, 20):
        k = min(k, n)
        appeared = sum(1 for d in history[-k:] if number in d["reds"])
        feats.append(float(appeared) / k)

    zc = F.zone_counts(history[-10:]) if n >= 10 else history
    z_counts = [0, 0, 0]
    for z in zc:
        for i in range(3):
            z_counts[i] += z[i]
    for zc_val in z_counts:
        feats.append(float(zc_val) / max(len(zc), 1))

    recent = history[-10:] if n >= 10 else history
    if recent:
        feats.append(float(np.mean(F.sums(recent))))
        feats.append(float(np.mean(F.span(recent))))
    else:
        feats.append(0.0)
        feats.append(0.0)

    return feats


def _compute_blue_features(history: List[Dict], number: int) -> List[float]:
    """蓝球特征向量。"""
    n = len(history)
    feats = []

    for w in (5, 10, 20, 30, 50, 150):
        sl = history[-w:] if w <= n else history
        f = F.blue_frequency(sl)
        feats.append(float(f[number]))

    om_cur = F.current_omission_blue(history)
    om_avg = F.avg_omission(F.blue_frequency(history), n)
    feats.append(float(om_cur[number]))
    feats.append(float(om_avg[number]))
    feats.append(float(om_cur[number]) / max(float(om_avg[number]), 0.01))

    for k in (1, 2, 3, 5, 10, 20):
        k = min(k, n)
        appeared = sum(1 for d in history[-k:] if d["blue"] == number)
        feats.append(float(appeared) / k)

    if n >= 10:
        feats.append(float(np.mean(F.sums(history[-10:]))))
    else:
        feats.append(0.0)

    return feats


# ---------- 模型构造 ----------

def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """正样本权重 = 倒频率，抵消类别不平衡。"""
    pos = float(np.mean(y))
    if pos <= 0 or pos >= 1:
        return np.ones(len(y))
    w = np.where(y == 1, (1.0 - pos) / pos, 1.0)
    return w


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=max(20, config.ML_N_ESTIMATORS),
        max_depth=config.ML_MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def _make_histgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max(25, config.ML_N_ESTIMATORS),
        max_depth=min(4, max(2, config.ML_MAX_DEPTH)),
        random_state=42,
    )


def _make_classic_gbdt() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=max(20, min(60, config.ML_N_ESTIMATORS)),
        max_depth=min(3, max(2, config.ML_MAX_DEPTH)),
        learning_rate=0.05,
        random_state=42,
    )


def _red_voting() -> VotingClassifier:
    """红球二分类投票：RF + HistGB（速度优先）。"""
    ests = [("rf", _make_rf())]
    if HAS_HISTGB:
        ests.append(("gb", _make_histgb()))
    return VotingClassifier(estimators=ests, voting="soft", n_jobs=-1)


def _blue_voting() -> VotingClassifier:
    """蓝球二分类投票：RF + HistGB + 经典 GBDT（蓝球样本更少，多为胜者）。"""
    ests = [("rf", _make_rf())]
    if HAS_HISTGB:
        ests.append(("gb", _make_histgb()))
    ests.append(("gbdt", _make_classic_gbdt()))
    return VotingClassifier(estimators=ests, voting="soft", n_jobs=-1)


def _calibrated(clf, X, y, cv: int = 3):
    """sigmoid（Platt）校准；样本不足时退化为原分类器。"""
    if len(np.unique(y)) < 2 or len(y) < 30:
        return clf
    try:
        cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=cv)
        cal.fit(X, y)
        return cal
    except Exception:  # noqa: BLE001
        return clf


# ---------- 数据集构建（per-number） ----------

def _build_red_datasets(draws: List[Dict], min_start: int,
                        numbers: Optional[List[int]] = None) -> Dict[int, tuple]:
    """返回 {num: (X, y)}，X 每行 = 用 draws[:i+1] 预测 draws[i+1]。

    ``numbers``：只构建这些号码（默认全部 33 个）。诊断只需少数号码时用它省时间
    （特征计算是训练链路里最贵的部分，按号码数线性缩放）。
    """
    nums = sorted({int(n) for n in numbers} & set(range(1, 34))) if numbers else list(range(1, 34))
    out: Dict[int, tuple] = {num: ([], []) for num in nums}
    for i in range(min_start, len(draws) - 1):
        history = draws[:i + 1]
        target_reds = set(draws[i + 1]["reds"])
        for num in nums:
            feats = _compute_features(history, num)
            out[num][0].append(feats)
            out[num][1].append(1 if num in target_reds else 0)
    return {num: (np.array(x, dtype=float), np.array(y, dtype=float))
            for num, (x, y) in out.items()}


def _build_blue_datasets(draws: List[Dict], min_start: int,
                         numbers: Optional[List[int]] = None) -> Dict[int, tuple]:
    """返回 {num: (X, y)}，``numbers`` 同上（默认全部 16 个蓝球）。"""
    nums = sorted({int(n) for n in numbers} & set(range(1, 17))) if numbers else list(range(1, 17))
    out: Dict[int, tuple] = {num: ([], []) for num in nums}
    for i in range(min_start, len(draws) - 1):
        history = draws[:i + 1]
        target_blue = draws[i + 1]["blue"]
        for num in nums:
            feats = _compute_blue_features(history, num)
            out[num][0].append(feats)
            out[num][1].append(1 if num == target_blue else 0)
    return {num: (np.array(x, dtype=float), np.array(y, dtype=float))
            for num, (x, y) in out.items()}


def _split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    n_test = max(1, int(len(X) * test_ratio))
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def _bin_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict:
    """校准曲线（可靠性图）分箱统计 + ECE + Brier 分解（V10-08）。

    返回新增键：
      * ``brier`` / ``reliability`` / ``resolution`` / ``uncertainty``：Murphy 分解，
        恒等式 ``brier = reliability - resolution + uncertainty`` 在 bin 均值口径下
        **精确成立**（见 `brier_decomposition`），误差仅浮点级；
      * ``brier_sample``：逐样本（未分箱）Brier，与 ``brier`` 的差 = 箱内预测方差项；
      * ``resolution_skill``：``resolution - reliability``，模型相对基础率带来的
        Brier 净改善；<=0 表示「不比永远预测基础率更好」——随机彩票上的正常结果。
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    ece, total = 0.0, max(1, len(p))
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}", "n": 0,
                         "mean_pred": None, "freq": None})
            continue
        mp = float(np.mean(p[m]))
        fr = float(np.mean(y[m]))
        rows.append({"bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}", "n": cnt,
                     "mean_pred": round(mp, 4), "freq": round(fr, 4)})
        ece += cnt * abs(mp - fr)
    out = {"bins": rows, "ece": round(ece / total, 4)}
    dec = brier_decomposition(p, y, n_bins=n_bins)
    out.update(dec)
    out["brier_decomposition"] = dec      # 显式子块，避免与分箱字段混淆
    return out


def _brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict:
    """Murphy 分解的**精确**实现（内部用，不做四舍五入，供自洽断言）。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    # V38（2026-10-24 审计 P2）：**校验 y 是二值**。
    #
    # Murphy 分解 `brier = reliability - resolution + uncertainty` 的推导前提是
    # y ∈ {0,1}（uncertainty = ō(1−ō) 只在二值时是"基准预测的 Brier"）。
    # 若传进 0/1 之外的值（例如 0.5，或误把概率当真值），恒等式会**破裂**：
    # 实测 y=[0.5] 时 brier=0 而 rel−res+unc=0.25，gap=−0.25 ——
    # 一个看起来"分数很好"的结果，实际是接口被误用的产物。
    # 二值输入下恒等式精确到 1e-16（已实测），所以这里只挡非二值。
    n = len(y)
    if n and not np.all((y == 0) | (y == 1)):
        bad = np.unique(y[(y != 0) & (y != 1)])[:5]
        raise ValueError(
            f"Brier 分解的 y 必须是二值 0/1，收到 {bad.tolist()} 之类取值。"
            "请确认传入的是真实标签而不是概率；若确实要评概率预测，"
            "请改用连续型评分（如 log score），该分解不适用。")
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.0, "brier_sample": 0.0, "n": 0}
    base = float(np.mean(y))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rel = res = 0.0
    p_bin = p.copy()
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        mp = float(np.mean(p[m]))
        fr = float(np.mean(y[m]))
        p_bin[m] = mp                      # 用箱均值替代，恒等式才有精确形式
        rel += cnt * (mp - fr) ** 2
        res += cnt * (fr - base) ** 2
    rel /= n
    res /= n
    unc = base * (1.0 - base)
    brier = float(np.mean((p_bin - y) ** 2))   # 分箱后 Brier：= rel - res + unc
    return {
        "brier": brier,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "brier_sample": float(np.mean((p - y) ** 2)),
        "resolution_skill": res - rel,
        "n": int(n),
        "n_bins": int(n_bins),
    }


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict:
    """Brier 分数分解（V10-08 校准诊断）：``brier = reliability - resolution + uncertainty``。

    三项含义（概率预测的经典 Murphy 分解）：
      * **reliability（可靠性）**：预测概率与实测频率的偏离。**越低越好**——
        接近 0 说明「说 p=0.2 的样本里确实约 20% 发生」，即被校准；
      * **resolution（分辨率）**：各箱实测频率相对基础率的离散度。**越高越好**——
        说明模型确实把「更可能发生」和「更不可能发生」区分开了；
      * **uncertainty（不确定性）**：只由基础率决定（``ō(1-ō)``），与模型无关。

    实现口径（**重要，否则恒等式不成立**）：三个分解项都建立在**箱均值预测**
    （``p`` 在每个概率桶内被替换成该桶的均值）之上，此时

        brier_binned = Σ_k n_k (p_k - y_i)² / N = reliability - resolution + uncertainty

    是精确恒等式（误差仅浮点级 ±1e-15），可用作自证。逐样本（未分箱）Brier
    另放在 ``brier_sample``；两者之差即「箱内预测方差项」，桶数越多它越小。

    额外返回 ``resolution_skill = resolution - reliability``：
    相对「永远预测基础率」的 Brier 净改善。

    诚实边界：双色球为独立随机事件，正确实现的模型应输出**接近均匀**的概率，
    此时 resolution ≈ reliability（``resolution_skill ≈ 0``）。若某次训练在真实数据上
    出现很大的 ``resolution_skill``，那不是模型厉害，而是过拟合/数据泄漏的信号。
    """
    d = _brier_decomposition(p, y, n_bins=n_bins)
    return {
        "brier": _pround(d["brier"]),
        "reliability": _pround(d["reliability"]),
        "resolution": _pround(d["resolution"]),
        "uncertainty": _pround(d["uncertainty"]),
        "brier_sample": _pround(d["brier_sample"]),
        "resolution_skill": _pround(d["resolution_skill"]),
        # 恒等式在展示值上千分位一致；这里同时给出精确残差供自证
        "identity_gap": _pround(d["brier"] - (d["reliability"] - d["resolution"]
                                              + d["uncertainty"]), 15),
        "n": d["n"], "n_bins": d["n_bins"],
    }


# ---------- V10-08：置换重要性 + 打乱标签对照（ML 通道可信度判据） ----------
#
# 为什么这是「能否信任 ML 通道」的判据，而不是「提升 ML 效果」的工具：
# 双色球是独立随机事件，一个**正确实现**的概率模型应当学不到任何东西——
# 它的特征重要性应与「把 y 随机打乱后再训练」的模型无法区分。若真实模型的
# 重要性**显著高于**打乱标签对照，说明它抓到了真实数据里的某种关联；若没有，
# 那么任何看似漂亮的重要性/校准曲线都只是过拟合噪声，该通道必须降级为 monitor。
#
# 判据（置换检验，两种聚合层级）：
#   单数据集（`shuffled_label_test`）：
#     1. 在真实 (X, y) 上训练「真实模型」，在测试集上算逐特征置换重要性；
#     2. 把 y 随机打乱后**重训同样结构**的模型（打乱标签对照），同样算重要性；
#     3. 重复 n_null 次得到零分布；用 max-over-features 统计量吸收多重比较；
#     4. p = (1 + #{零分布 max ≥ 真实 max}) / (1 + n_null)；p < alpha 才算显著。
#   多号码（`diagnose_ml_channel`）：每次零复制里给每个号码各训一个打乱标签模型，
#   统计量取「各号码最强特征重要性的平均」（见 `_combine_signal_test`）。
#
# 注意：这是一个 **α 水平的检验**，不是「打乱标签必须永远不显著」——单次运行本身
# 有 α 概率（正确地）报显著。自证方式看的是多个独立打乱下的**拒绝率**（测试 T7）。

# ---- 零复制次数 B（V10-08 / D-2 修正）----
#
# 置换检验的 p = (1 + #{零复制统计量 ≥ 真实统计量}) / (1 + B)，因此**最小可达 p
# 就是 1/(B+1)**。B 太小会出两个致命问题（D-2 实测踩过）：
#   1. **分辨率不足**：B=20 时最小 p = 1/21 ≈ 0.0476，刚好卡在 α=0.05 判定线上。
#      于是「显著」等价于「真实统计量超过全部 20 个零复制」——门槛近乎「比零分布
#      最大值略高」，而不是 5% 水平。任何一点向上偏差都会被放大成高假阳性率
#      （Lead 实测 B=20 时 trusted=True 假阳性率 35%）；
#   2. **Monte-Carlo 误差大**：p̂ 的标准误 ≈ sqrt(p(1-p)/B)，B=20 时在 p≈0.05 处
#      约 0.048，p 几乎只剩「0.048 或更大」两档。
#
# 因此默认用 **standard（B=200）**，最小 p = 1/201 ≈ 0.005（α/10 量级）；
# **fast（B=20）** 仅供演示/冒烟，输出一律带 ``advisory_only=True``，
# 上层 ``diagnose_ml_channels`` 也不会把 fast 结果当成结论。
DIAG_NULL_FAST = 20         # 演示档：约 1/(B+1)=0.048 分辨率，**不可用于下结论**
DIAG_NULL_STANDARD = 200    # 结论档：1/201 ≈ 0.005 分辨率
# 库函数默认值：**fast**。理由：底层函数是"可被频繁调用的诊断原语"（单测、
# 页面卡片、逐通道探测），默认 200 会让任何一次调用从 ~10s 变成 ~100s。
# 结论档必须显式要求：``mode="standard"`` 或 ``n_null>=79``；而 ``diagnose_ml_channels``
# （对外唯一入口）默认就是 standard。低分辨率调用的输出一律带
# ``advisory_only=True``，顶层会被强制降级为 trusted=False / tier=monitor。
DIAG_DEFAULT_NULL = DIAG_NULL_FAST
DIAG_DEFAULT_REPEATS = 6    # 每特征置换重复次数


def _null_resolution(B: int, alpha: float = 0.05) -> Dict:
    """零复制次数 B 是否足以在 α 水平上「有分辨率」地下结论。

    判据：最小可达 p = 1/(B+1) 必须 ≤ α/4。含义是「判定阈值不能被单个零复制
    左右的抖动决定」——B=20 时 1/21 = 0.048 > 0.0125，故判为分辨率不足（仅演示）。
    B ≥ ceil(4/α) − 1 = 79（α=0.05）才算够。
    """
    B = max(0, int(B))
    min_p = 1.0 / (1.0 + B)
    need_b = max(1, int(np.ceil(4.0 / float(alpha))) - 1)
    return {
        "n_null": B,
        "min_attainable_p": round(min_p, 6),
        "required_min_p": round(float(alpha) / 4.0, 6),
        "required_n_null": need_b,
        "ok": bool(min_p <= float(alpha) / 4.0),
        "mc_se_at_alpha": round(float(np.sqrt(float(alpha) * (1 - float(alpha)) / max(1, B))), 5),
    }


def _resolve_n_null(mode: Optional[str], n_null: Optional[int],
                    alpha: float = 0.05) -> Dict:
    """把 (mode, n_null) 解析成实际使用的 B，并给出是否仅为「演示档」。"""
    if n_null is not None:
        B = int(n_null)
    elif str(mode or "standard").lower().startswith("f"):
        B = DIAG_NULL_FAST
    else:
        B = DIAG_NULL_STANDARD
    B = max(2, B)
    info = _null_resolution(B, alpha)
    info["mode"] = ("fast" if (B <= DIAG_NULL_FAST) else
                    ("standard" if info["ok"] else "custom"))
    info["advisory_only"] = not info["ok"]
    return info


def _brier(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or len(p) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def _pround(x, ndigits: int = 10) -> float:
    """概率/分解项的对外舍入：位数必须比自洽判据（±1e-9）更细。

    历史坑：对外只保留 6 位时，``brier = reliability - resolution + uncertainty``
    的舍入误差可达 ~1e-6，看起来像「恒等式不成立」。对外保留 10 位（内部仍是双精度），
    恒等式在**展示值**上也成立到 1e-9，UI/JSON 也不会出现长尾浮点噪音。
    """
    try:
        return round(float(x), int(ndigits))
    except (TypeError, ValueError):
        return float("nan")


def _pos_proba(clf, X: np.ndarray) -> np.ndarray:
    """取正类概率；退化模型（训练集只有一个类别）时返回该类别常数。"""
    proba = np.asarray(clf.predict_proba(np.asarray(X, dtype=float)), dtype=float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    cls = float(np.asarray(getattr(clf, "classes_", [0])).ravel()[0])
    return np.full(len(X), cls)


def _make_diag_learner(n_estimators: int = 40, max_depth: int = 3, seed: int = 42):
    """诊断用轻量学习器：**与生产同结构**（RF + HistGB 软投票），只缩小规模。

    两处**刻意**与生产不同，都是为了「诊断读数可解释」：
      * 不施加 ``class_weight=balanced`` / 平衡样本权重。平衡权重会把概率整体抬到
        0.5 附近（实测纯噪声数据上输出均值 0.49，而基础率只有 0.20），可靠性图
        的偏差项于是反映「加权造成的固有偏移」而不是「有没有学到信息」。生产管线
        靠其后的 Platt 校准修掉这件事；诊断不做校准，就必须先去掉这层偏移。
      * ``n_jobs=1``：uvicorn 线程池内用多进程/多线程后端派发任务容易死锁
        （``mining.py`` 有同样的注释）。诊断路径本来就不是热路径。
    """
    rf = RandomForestClassifier(
        n_estimators=max(10, int(n_estimators)),
        max_depth=max(1, int(max_depth)),
        random_state=int(seed), n_jobs=1,
    )
    ests = [("rf", rf)]
    if HAS_HISTGB:
        ests.append(("gb", HistGradientBoostingClassifier(
            max_iter=max(10, int(n_estimators)),
            max_depth=max(1, int(max_depth)), random_state=int(seed))))
    return VotingClassifier(estimators=ests, voting="soft", n_jobs=1)


def permutation_importance(estimator, X, y, feature_names: Optional[List[str]] = None,
                           n_repeats: int = DIAG_DEFAULT_REPEATS,
                           seed: int = 0) -> Dict:
    """置换重要性：逐列打乱特征取值，度量 Brier 的**上升量**（越大越重要）。

    ``estimator`` 须已 fit；重要性在 (X, y) 上评估（应为测试集，防泄漏）。
    返回 ``{"ok", "base_brier", "importances": [{feature, importance, std, ...}]}``，
    其中 ``importance = mean(Brier(打乱后) − Brier(原))``，单位与 Brier 相同。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_feat = X.shape[1] if X.ndim == 2 else 0
    if X.ndim != 2 or len(X) == 0 or n_feat == 0:
        return {"ok": False, "error": "空数据集", "importances": [], "base_brier": None}
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_feat)]
    names = names[:n_feat]
    base = _brier(_pos_proba(estimator, X), y)
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    rows = []
    for j, name in enumerate(names):
        deltas = []
        for _ in range(max(1, int(n_repeats))):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])           # 只打乱第 j 列，其余列保持
            deltas.append(_brier(_pos_proba(estimator, Xp), y) - base)
        arr = np.asarray(deltas, dtype=float)
        rows.append({
            "feature": name, "index": j,
            "importance": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
        })
    return {"ok": True, "base_brier": base, "importances": rows,
            "n_repeats": int(n_repeats), "n_samples": int(len(y))}


def reliability_noise_band(bins: List[Dict], z: Optional[float] = None,
                           alpha: float = 0.05, min_n: int = 20) -> Dict:
    """给可靠性图每个桶配「二项抽样噪声带」，判断偏差是否只是抽样波动。

    每桶允许偏差 ``z * sqrt(f(1-f)/n)``（f=实测频率，n=桶样本量）：
      * ``z`` 为 None 时按 **Bonferroni 校正**取 ``norm.ppf(1 - alpha/(2k))``（k=参与
        检验的桶数）：10 桶 → z≈2.81。不校正的话，「10 个桶里至少 1 个出 z=2 带」的
        概率高达 37%，纯噪声数据也会频繁报「校准异常」——这正是本函数要避免的假警报；
      * 桶样本量 n < ``min_n`` 的桶只并列展示、不参与判定（二项带在小 n 上不可靠，
        一个 n=1 的桶能贡献 0.5 以上的假偏差）。

    纯噪声数据上正确实现的模型应给出 ``within_noise_band=True``；
    ``n_outside`` / ``worst_excess`` 用于定位到底是哪个桶跑偏。
    """
    usable = [r for r in (bins or [])
              if int(r.get("n") or 0) >= max(1, int(min_n))
              and r.get("mean_pred") is not None and r.get("freq") is not None]
    k = len(usable)
    if z is None:
        try:
            from scipy import stats as _st
            z_eff = float(_st.norm.ppf(1.0 - float(alpha) / (2.0 * k))) if k else float("inf")
        except Exception:  # noqa: BLE001
            z_eff = 2.81 if k <= 10 else 3.0
    else:
        z_eff = float(z)
    rows, worst, outside, checked = [], 0.0, 0, 0
    for r in bins or []:
        n = int(r.get("n") or 0)
        if n <= 0 or r.get("mean_pred") is None or r.get("freq") is None:
            rows.append({"bin": r.get("bin"), "n": n, "band": None, "within": None})
            continue
        f, mp = float(r["freq"]), float(r["mean_pred"])
        band = z_eff * float(max(f * (1.0 - f), 1e-12) / n) ** 0.5
        gap = abs(mp - f)
        if n < max(1, int(min_n)):
            rows.append({"bin": r["bin"], "n": n, "mean_pred": mp, "freq": f,
                         "gap": round(gap, 4), "band": round(band, 4),
                         "within": None, "small_sample": True})
            continue
        within = bool(gap <= band + 1e-12)
        checked += 1
        if not within:
            outside += 1
        worst = max(worst, gap - band)
        rows.append({"bin": r["bin"], "n": n, "mean_pred": mp, "freq": f,
                     "gap": round(gap, 4), "band": round(band, 4), "within": within})
    return {"z": round(z_eff, 3) if np.isfinite(z_eff) else None,
            "alpha": float(alpha), "min_n": int(min_n), "bins": rows,
            "n_checked": checked, "n_outside": outside,
            "within_noise_band": bool(checked > 0 and outside == 0),
            "worst_excess": round(float(worst), 4)}


def _fit_and_importance(X_tr, y_tr, X_te, y_te, names,
                        n_estimators: int, max_depth: int,
                        n_repeats: int, seed: int) -> Dict:
    """训练一个诊断模型并返回「测试集置换重要性 + 测试集预测概率」。

    真实臂与打乱标签对照臂都走这一个函数，保证两边**结构、超参、置换种子完全一致**
    ——否则 p 值没有意义。
    """
    clf = _make_diag_learner(n_estimators, max_depth, seed=42)
    clf.fit(np.asarray(X_tr, dtype=float), np.asarray(y_tr, dtype=float))
    pi = permutation_importance(clf, np.asarray(X_te, dtype=float),
                               np.asarray(y_te, dtype=float), names,
                               n_repeats=n_repeats, seed=seed)
    imps = [float(r["importance"]) for r in pi["importances"]]
    return {"importances": imps, "p_te": _pos_proba(clf, np.asarray(X_te, dtype=float))}


def shuffled_label_test(
    X, y, feature_names: Optional[List[str]] = None, *,
    n_repeats: int = DIAG_DEFAULT_REPEATS,
    n_null: Optional[int] = None,
    mode: Optional[str] = None,
    alpha: float = 0.05,
    seed: int = 0,
    test_ratio: float = 0.2,
    n_estimators: int = 40,
    max_depth: int = 3,
    n_bins: int = 10,
) -> Dict:
    """置换重要性 **vs 打乱标签对照**：ML 通道可信度的核心判据。

    流程：
      1. 真实 (X, y) 前 80% 训练、后 20% 测试（时序切分，防未来信息），在测试集上
         算逐特征置换重要性（打乱该列 → Brier 上升多少）；
      2. 把 y **整列打乱**后**重训同样结构**的模型（打乱标签对照），同样在测试集
         （用打乱后的 y）上算重要性 → 得到「什么也没学到时的重要性」零分布；
      3. 重复 ``n_null`` 次；用 max-over-features 统计量（多重比较校正已内含）：
         ``p = (1 + #{零分布 max ≥ 真实 max}) / (1 + n_null)``；
      4. p < ``alpha`` 才算「真实重要性显著高于打乱标签对照」。

    同时给出可靠性图 + Brier 分解 + 二项噪声带。

    返回的关键结论：
      * ``importance_significant``：真实重要性是否显著高于打乱标签对照；
      * ``trusted``：与之等价——**不显著就是不可信**（不允许假装模型有用）；
      * ``tier``：``"monitor"``（不可信/只观察）或 ``"signal_found"``（检出信号）；
      * ``reason`` / ``reasons`` / ``warnings``：可直接展示给用户的原因。

    诚实边界：``trusted=True`` 只表示「检测到高于噪声的关联」，**不承诺**命中率提升；
    真实彩票数据上的正确期望结论是 ``trusted=False``。
    """
    t0 = time.time()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return {"ok": False, "error": "空数据集", "trusted": False,
                "importance_significant": False, "tier": "monitor",
                "reason": "空数据集，无法诊断"}
    n, n_feat = X.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_feat)]
    names = names[:n_feat]
    if n < 40:
        return {"ok": False, "error": f"样本不足（{n} < 40），诊断无意义",
                "trusted": False, "importance_significant": False, "tier": "monitor",
                "reason": f"样本不足（{n} < 40）"}
    n_test = max(1, int(n * test_ratio))
    n_tr = n - n_test
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    if len(np.unique(y_tr)) < 2:
        return {"ok": False, "error": "训练集只有单一类别，无法训练",
                "trusted": False, "importance_significant": False, "tier": "monitor",
                "reason": "训练集标签单一，无法训练/诊断"}

    rng = np.random.RandomState(int(seed) % (2 ** 32))
    nul = _resolve_n_null(mode, n_null, alpha)
    n_null_eff = int(nul["n_null"])

    # 1) 真实模型
    real_clf = _make_diag_learner(n_estimators, max_depth, seed=42)
    real_clf.fit(X_tr, y_tr)
    real_pi = permutation_importance(real_clf, X_te, y_te, names,
                                     n_repeats=n_repeats, seed=seed + 1)
    real_imp = [float(r["importance"]) for r in real_pi["importances"]]
    p_te = _pos_proba(real_clf, X_te)
    p_tr = _pos_proba(real_clf, X_tr)

    # 2) 打乱标签对照：同样的结构、同样的切分、同样的置换种子
    null_per_feat: List[List[float]] = [[] for _ in range(n_feat)]
    null_maxes: List[float] = []
    for _ in range(max(1, n_null_eff)):
        yb = rng.permutation(y)                 # 整列打乱 → 切断 y 与 X 的一切关联
        yb_tr, yb_te = yb[:n_tr], yb[n_tr:]
        if len(np.unique(yb_tr)) < 2:
            continue
        try:
            clf = _make_diag_learner(n_estimators, max_depth, seed=42)
            clf.fit(X_tr, yb_tr)
        except Exception as e:  # noqa: BLE001
            print(f"[ml] 打乱标签对照训练失败（忽略该次）: {e}")
            continue
        pi = permutation_importance(clf, X_te, yb_te, names,
                                    n_repeats=n_repeats, seed=seed + 1)
        imps = [float(r["importance"]) for r in pi["importances"]]
        for j, v in enumerate(imps[:n_feat]):
            null_per_feat[j].append(v)
        null_maxes.append(max(imps) if imps else 0.0)

    n_null_done = len(null_maxes)
    rows = []
    for j, name in enumerate(names):
        nulls = np.asarray(null_per_feat[j] or [0.0], dtype=float)
        p_val = (1.0 + float(np.sum(nulls >= real_imp[j]))) / (1.0 + float(len(nulls)))
        rows.append({
            "feature": name, "index": j,
            "importance": real_imp[j],
            "null_mean": float(np.mean(nulls)),
            "null_p95": float(np.percentile(nulls, 95)),
            "p_value": round(float(p_val), 4),
            # 注意：这是**逐特征、未做多重比较校正**的 p 值，仅供定位；
            # 通道能不能信只看下面的 ``importance_significant``（max-T 统计量）。
            "significant": bool(p_val < alpha),
        })
    rows.sort(key=lambda r: -r["importance"])

    max_real = float(max(real_imp)) if real_imp else 0.0
    null_maxes_arr = np.asarray(null_maxes or [0.0], dtype=float)
    n_ge = int(np.sum(null_maxes_arr >= max_real))
    p_max = (1.0 + n_ge) / (1.0 + float(len(null_maxes_arr)))
    null_max_p95 = float(np.percentile(null_maxes_arr, 95))
    significant = bool(p_max < alpha and max_real > 0.0)

    # 3) 校准诊断（真实模型在测试集上的预测质量）
    cal = _bin_metrics(p_te, y_te, n_bins=n_bins)
    band = reliability_noise_band(cal["bins"])
    base_rate = float(np.mean(y_te))
    uniform_brier = _brier(np.full(len(y_te), base_rate), y_te)

    reasons: List[str] = []
    warnings: List[str] = []
    best = rows[0] if rows else {"feature": None, "importance": 0.0}
    if significant:
        reasons.append(
            f"真实特征重要性显著高于打乱标签对照（max-T p={p_max:.3f} < α={alpha}，"
            f"最佳特征 {best['feature']} = {best['importance']:.5f}）")
        tier = "signal_found"
    else:
        reasons.append(
            f"真实特征重要性**不**显著高于打乱标签对照（max-T p={p_max:.3f} ≥ α={alpha}，"
            f"零分布 95 分位 {null_max_p95:.5f} ≥ 真实最大 {max_real:.5f}）："
            f"模型未学到超出噪声的结构")
        tier = "monitor"
    advisory_only = bool(nul["advisory_only"])
    if advisory_only:
        # 注意：**不**改写 tier —— tier 的既有语义（"signal_found"/"monitor"）是
        # 「检验本身检出了什么」，被 17 条既有用例钉住。低分辨率问题用
        # advisory_only + warnings 表达，并在 `diagnose_ml_channels` 顶层
        # 强制降级（那里才是"能不能当结论展示"的唯一出口）。
        reasons.append(
            f"⚠️ 仅演示：零复制次数 B={n_null_eff} 的最小可达 p="
            f"{nul['min_attainable_p']:.4f} > α/4={nul['required_min_p']:.4f}，"
            f"分辨率不足（需 B ≥ {nul['required_n_null']}），本结论不可当真")
        warnings.append(
            f"零复制次数不足（B={n_null_eff}，需 ≥{nul['required_n_null']}）："
            f"仅演示、不做结论")
    if not band["within_noise_band"]:
        warnings.append(
            f"可靠性图有 {band['n_outside']}/{band['n_checked']} 个桶超出二项噪声带"
            f"（Bonferroni z={band['z']}），存在系统性校准偏差或分箱过细")
    if float(cal.get("resolution", 0.0)) - float(cal.get("reliability", 0.0)) > 0.02:
        warnings.append(
            "resolution 明显大于 reliability（净技能 > 0.02）：真实随机数据上不应出现，"
            "优先怀疑过拟合/数据泄漏")

    trusted = bool(significant)
    if not trusted:
        reason = reasons[0] + " → ML 通道应降级为 monitor（只监控，不参与出号加权）"
    else:
        reason = reasons[0] + " → 可继续研究，但仍不承诺命中率提升"
    return {
        "ok": True,
        "alpha": float(alpha),
        "n_samples": int(n), "n_train": int(n_tr), "n_test": int(n_test),
        "n_features": int(n_feat), "feature_names": names,
        "test_brier": _brier(p_te, y_te),
        "train_brier": _brier(p_tr, y_tr),
        "test_brier_uniform": uniform_brier,
        "base_rate": base_rate,
        "importance": rows,
        "best_feature": best["feature"],
        "best_importance": float(best["importance"]),
        "null_max_p95": null_max_p95,
        "null_maxes": [float(v) for v in null_maxes_arr.tolist()],
        "n_null": int(n_null_done),
        "null_resolution": nul,
        "advisory_only": advisory_only,
        "mode": nul["mode"],
        "p_value": round(float(p_max), 4),
        "min_attainable_p": round(float(1.0 / (1.0 + len(null_maxes_arr))), 4),
        "importance_significant": significant,
        "trusted": trusted,
        "tier": tier,
        "reason": reason,
        "reasons": reasons,
        "warnings": warnings,
        "calibration": {**cal, "noise_band": band,
                        "brier_decomposition": brier_decomposition(p_te, y_te, n_bins=n_bins)},
        "seconds": round(time.time() - t0, 2),
    }


def _combine_signal_test(real_imps: List[List[float]], null_imps: List[List[List[float]]],
                         alpha: float = 0.05) -> Dict:
    """跨号码合并「置换重要性 vs 打乱标签」的显著性检验（联合零分布）。

    ``real_imps``：``[号码][特征]`` 的真实重要性；``null_imps``：``[零复制][号码][特征]``。
    两个统计量（都用同一个零分布比较，因此**多重比较已被统计量本身吸收**）：

      * ``T_mean``（主判据）= 每个号码最强特征重要性的**平均**。信号散布在多个号码
        时比「全局最大」更有功效（单个号码的偶然尖峰会被平均掉）；
      * ``T_max`` = 所有（号码 × 特征）里的最大重要性，用于回答「是否存在任意一个
        号码×特征组合显著」。

    为什么不用「逐号码 p 值 + Bonferroni」：零复制次数 B 有限，最小可达 p 是
    ``1/(B+1)``，再乘号码数 m 必然 > α，**永远无法拒绝**。联合零分布与 m 无关。

    ⚠️ **B 必须足够大（D-2）**：``min_attainable_p = 1/(B+1)``。B=20 时它是 0.0476，
    与 α=0.05 几乎重合 → 检验退化成「真实统计量是否超过全部零复制」，任何一点
    向上偏差都会被放大（实测 trusted=True 假阳性率 35%）。返回体里给出
    ``resolution``（含 ``ok``：min_p ≤ α/4）供上层判断该结论能否当结论用。
    """
    m = len(real_imps)
    B = len(null_imps)
    real_per_num_max = [max(v) if v else 0.0 for v in real_imps]
    real_T_mean = float(np.mean(real_per_num_max)) if m else 0.0
    real_T_max = float(max((max(v) for v in real_imps if v), default=0.0))
    null_T_mean, null_T_max = [], []
    for b in range(B):
        per_num_max = [max(v) if v else 0.0 for v in null_imps[b]]
        null_T_mean.append(float(np.mean(per_num_max)) if per_num_max else 0.0)
        null_T_max.append(float(max((max(v) for v in null_imps[b] if v), default=0.0)))
    ntm = np.asarray(null_T_mean or [0.0], dtype=float)
    ntx = np.asarray(null_T_max or [0.0], dtype=float)
    p_mean = (1.0 + float(np.sum(ntm >= real_T_mean))) / (1.0 + float(len(ntm)))
    p_max = (1.0 + float(np.sum(ntx >= real_T_max))) / (1.0 + float(len(ntx)))
    return {
        "T_mean": real_T_mean, "T_max": real_T_max,
        "null_T_mean_p95": float(np.percentile(ntm, 95)),
        "null_T_max_p95": float(np.percentile(ntx, 95)),
        "null_T_mean": [float(v) for v in ntm.tolist()],
        "null_T_max": [float(v) for v in ntx.tolist()],
        "p_value": round(float(p_mean), 4),
        "p_value_mean_stat": round(float(p_mean), 4),
        "p_value_max_stat": round(float(p_max), 4),
        "min_attainable_p": round(float(1.0 / (1.0 + max(1, len(ntm)))), 4),
        "resolution": _null_resolution(len(ntm), alpha),
        "significant": bool(p_mean < alpha and real_T_max > 0.0),
        "n_null": int(len(ntm)), "n_numbers": int(m),
    }


def diagnose_ml_channel(draws: List[Dict], *, color: str = "red",
                        min_start: Optional[int] = None,
                        max_issues: Optional[int] = 800,
                        numbers: Optional[List[int]] = None,
                        max_numbers: int = 6,
                        n_repeats: int = 4,
                        n_null: Optional[int] = None,
                        mode: Optional[str] = None,
                        alpha: float = 0.05, seed: int = 0,
                        n_estimators: int = 30, max_depth: int = 3,
                        test_ratio: float = 0.2,
                        n_bins: int = 10) -> Dict:
    """诊断 ML 通道是否可信：逐号码置换重要性 + **联合**打乱标签零分布。

    为什么逐号码而不是把所有号码池化成一个矩阵（这个选择有统计上的理由）：
    池化后每个号码的标签边际率不同（号码流行度），而特征里含 ``freq_*/omit_*``，
    这些特征本身就足以「认出是哪个号码」——模型于是能靠「号码身份 → 该号码历史
    命中率」拿到看似真实的重要性，而这只是边际率噪声，不是可预测性。实测池化版本
    在纯随机数据上就会被这个混杂顶到边缘（p≈0.09）。生产模型本来就是**每个号码一个
    分类器**，所以逐号码诊断既更干净，也更贴近线上结构。

    零分布怎么来（联合零分布，见 `_combine_signal_test`）：每个零复制里，把**每个
    号码**的 y 各自整列打乱、各自重训一个同结构模型，于是得到一个「所有号码都没
    学到东西」的世界的一整套重要性；主统计量 = 各号码最强特征重要性的平均。

    耗时与功效：
      * 默认只诊断 ``max_numbers`` 个号码（在 1..N 里等距取样）以控制耗时；
      * ``max_issues`` 限制每个号码参与诊断的期数（默认 800；传 None 用全部历史，
        功效更高但更慢）。真实数据上默认约 30~60 秒。
        样本量直接决定功效：每号码只有 ~200 期时测试集仅 ~40 行，种在特征里的弱
        信号都测不出来。「不显著」不能当作「信号不存在」的证据——
        真实彩票数据上 ``trusted=False`` 是预期结论，但请按
        ``n_samples_per_number`` 说明样本量。
    """
    if not HAS_SKLEARN:
        return {"ok": False, "error": "scikit-learn 未安装，无法诊断",
                "trusted": False, "importance_significant": False, "tier": "monitor"}
    # 零复制次数：默认走 standard（B=200）；显式 n_null 优先；fast 仅演示
    nul = _resolve_n_null(mode, n_null, alpha)
    n_null_eff = int(nul["n_null"])
    color = "blue" if str(color).lower().startswith("b") else "red"
    names = _BLUE_FEATURE_NAMES if color == "blue" else _FEATURE_NAMES
    builder = _build_blue_datasets if color == "blue" else _build_red_datasets
    span = 16 if color == "blue" else 33
    min_start = min_start or config.ML_MIN_START
    if len(draws) < min_start + 20:
        return {"ok": False, "error": f"历史数据不足（{len(draws)} < {min_start + 20}）",
                "trusted": False, "importance_significant": False, "tier": "monitor",
                "reason": "数据不足，无法诊断"}
    ms = int(min_start)
    if max_issues:
        ms = max(ms, len(draws) - 1 - int(max_issues))
    if numbers:
        test_numbers = sorted({int(n) for n in numbers} & set(range(1, span + 1)))
    else:
        k = max(1, min(int(max_numbers), span))
        if k > 1:
            test_numbers = sorted({max(1, min(span, int(round(1 + i * (span - 1) / (k - 1)))))
                                   for i in range(k)})
        else:
            test_numbers = [1 + (span - 1) // 2]
    if not test_numbers:
        return {"ok": False, "error": "没有可诊断的号码", "trusted": False,
                "importance_significant": False, "tier": "monitor"}
    try:
        datasets = builder(draws, ms, numbers=test_numbers)
    except TypeError:      # 兼容被替换过的 builder（测试替身）
        datasets = builder(draws, ms)

    # 切分（时序切分：前 80% 训练、后 20% 测试）
    splits: List[Dict] = []
    for num in test_numbers:
        Xi, yi = datasets.get(num, (None, None))
        if Xi is None or len(yi) < 40:
            continue
        Xi = np.asarray(Xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        n_test = max(1, int(len(yi) * test_ratio))
        n_tr = len(yi) - n_test
        if len(np.unique(yi[:n_tr])) < 2:
            continue
        splits.append({"number": int(num), "X": Xi, "y": yi,
                       "n_tr": n_tr, "n_test": n_test,
                       "X_tr": Xi[:n_tr], "y_tr": yi[:n_tr],
                       "X_te": Xi[n_tr:], "y_te": yi[n_tr:]})
    if not splits:
        return {"ok": False, "error": "所有号码样本不足/单一类别，无法诊断",
                "trusted": False, "importance_significant": False, "tier": "monitor"}

    # 真实臂
    for d in splits:
        r = _fit_and_importance(d["X_tr"], d["y_tr"], d["X_te"], d["y_te"], names,
                                n_estimators, max_depth, n_repeats, seed + d["number"])
        d["imp"] = r["importances"]
        d["p_te"] = r["p_te"]
    real_imps = [d["imp"] for d in splits]

    # 联合零分布：每个复制里每个号码各自打乱 y、各自重训
    rng = np.random.RandomState(int(seed) % (2 ** 32) + 17)
    null_imps: List[List[List[float]]] = []
    for _ in range(max(1, n_null_eff)):
        per_num: List[List[float]] = []
        for d in splits:
            yb = rng.permutation(d["y"])            # 整列打乱该号码的 y
            yb_tr, yb_te = yb[:d["n_tr"]], yb[d["n_tr"]:]
            if len(np.unique(yb_tr)) < 2:
                per_num.append([0.0] * len(names))
                continue
            try:
                r = _fit_and_importance(d["X_tr"], yb_tr, d["X_te"], yb_te, names,
                                        n_estimators, max_depth, n_repeats,
                                        seed + d["number"])
                per_num.append(r["importances"])
            except Exception as e:  # noqa: BLE001
                print(f"[ml] 零分布训练失败（号码 {d['number']}，忽略该次）: {e}")
                per_num.append([0.0] * len(names))
        null_imps.append(per_num)

    agg = _combine_signal_test(real_imps, null_imps, alpha=alpha)

    # 每号码明细（逐特征 p 值由同一套零分布给出）
    per_number: List[Dict] = []
    for i_num, d in enumerate(splits):
        # 该号码的逐特征零分布（来自同一套联合零分布的第 i_num 个模型）
        null_feat = [[rep[i_num][j] for rep in null_imps] for j in range(len(names))]
        feats = []
        for j, name in enumerate(names):
            nulls = np.asarray(null_feat[j] if j < len(null_feat) else [0.0], dtype=float)
            pv = (1.0 + float(np.sum(nulls >= d["imp"][j]))) / (1.0 + float(len(nulls)))
            feats.append({"feature": name, "importance": d["imp"][j],
                          "null_p95": float(np.percentile(nulls, 95)),
                          "p_value": round(float(pv), 4)})
        feats.sort(key=lambda r: -r["importance"])
        cal = _bin_metrics(d["p_te"], d["y_te"], n_bins=n_bins)
        band = reliability_noise_band(cal["bins"])
        per_number.append({
            "number": d["number"],
            "n_samples": int(len(d["y"])), "n_test": int(d["n_test"]),
            "best_feature": feats[0]["feature"] if feats else None,
            "best_importance": feats[0]["importance"] if feats else 0.0,
            "top_features": feats[:3],
            "test_brier": round(_brier(d["p_te"], d["y_te"]), 4),
            "test_brier_uniform": round(_brier(
                np.full(len(d["y_te"]), float(np.mean(d["y_te"]))), d["y_te"]), 4),
            "brier": round(float(cal.get("brier", 0.0)), 4),
            "reliability": float(cal.get("reliability", 0.0)),
            "resolution": float(cal.get("resolution", 0.0)),
            "uncertainty": float(cal.get("uncertainty", 0.0)),
            "ece": float(cal.get("ece", 0.0)),
            "within_noise_band": bool(band["within_noise_band"]),
            "calibration": cal.get("bins", []),
        })

    cal_summary = {
        "n_numbers": len(per_number),
        "mean_brier": round(float(np.mean([d["brier"] for d in per_number])), 4),
        "mean_ece": round(float(np.mean([d["ece"] for d in per_number])), 4),
        "mean_reliability": round(float(np.mean([d["reliability"] for d in per_number])), 6),
        "mean_resolution": round(float(np.mean([d["resolution"] for d in per_number])), 6),
        "mean_uncertainty": round(float(np.mean([d["uncertainty"] for d in per_number])), 6),
        "n_outside_noise_band": int(sum(1 for d in per_number if not d["within_noise_band"])),
        "within_noise_band": bool(all(d["within_noise_band"] for d in per_number)),
    }
    trusted = bool(agg["significant"])
    advisory_only = bool(nul["advisory_only"])
    if trusted:
        reason = (f"真实特征重要性显著高于打乱标签对照"
                  f"（平均号码内最强重要性 {agg['T_mean']:.5f} vs 零分布 95 分位 "
                  f"{agg['null_T_mean_p95']:.5f}，p={agg['p_value']:.3f} < α={alpha}）"
                  f" → 可继续研究，但不承诺命中率提升")
        tier = "signal_found"
    else:
        reason = (f"真实特征重要性**不**显著高于打乱标签对照"
                  f"（平均号码内最强重要性 {agg['T_mean']:.5f} ≤ 零分布 95 分位 "
                  f"{agg['null_T_mean_p95']:.5f}，p={agg['p_value']:.3f} ≥ α={alpha}）"
                  f" → ML 通道应降级为 monitor（只监控，不参与出号加权）")
        tier = "monitor"
    # D-2：零复制次数不足时，**任何**方向都不能当结论（tier 保持原语义，
    # 由 advisory_only + 上层强制降级表达，见 diagnose_ml_channels）
    if advisory_only:
        reason = (f"⚠️ 仅演示（不做结论）：零复制次数 B={n_null_eff} 的最小可达 p="
                  f"{nul['min_attainable_p']:.4f} > α/4={nul['required_min_p']:.4f}，"
                  f"分辨率不足（需要 B ≥ {nul['required_n_null']}）。"
                  f"原始读数：{reason}")
    warnings: List[str] = []
    if advisory_only:
        warnings.append(
            f"零复制次数 B={n_null_eff} 不足以在 α={alpha} 上下结论"
            f"（min p={nul['min_attainable_p']:.4f}，需 B ≥ {nul['required_n_null']}）："
            f"本结果是**演示口径**，不得作为 trusted 结论展示或用于任何决策")
    if not cal_summary["within_noise_band"]:
        warnings.append(
            f"{cal_summary['n_outside_noise_band']}/{cal_summary['n_numbers']} 个号码的"
            f"可靠性图有桶超出二项噪声带")
    if cal_summary["mean_resolution"] - cal_summary["mean_reliability"] > 0.02:
        warnings.append("平均 resolution 明显大于 reliability：优先怀疑过拟合/数据泄漏")
    return {
        "ok": True,
        "channel": f"ml_{color}", "color": color,
        "numbers": [d["number"] for d in per_number],
        "per_number": per_number,
        "p_value": agg["p_value"],
        "p_value_mean_stat": agg["p_value_mean_stat"],
        "p_value_max_stat": agg["p_value_max_stat"],
        "T_mean": agg["T_mean"], "T_max": agg["T_max"],
        "null_T_mean_p95": agg["null_T_mean_p95"],
        "null_T_max_p95": agg["null_T_max_p95"],
        "min_attainable_p": agg["min_attainable_p"],
        "n_null": agg["n_null"],
        "null_resolution": nul,
        "advisory_only": advisory_only,
        "mode": nul["mode"],
        "alpha": float(alpha),
        "n_numbers": len(per_number),
        "importance_significant": bool(agg["significant"]),
        "trusted": trusted,
        # 演示档的 tier 一律 monitor（D-2：不让低分辨率结果冒充结论）
        "tier": tier,
        "reason": reason,
        "warnings": warnings,
        "calibration_summary": cal_summary,
        "min_start": ms,
        "max_issues": int(max_issues) if max_issues else None,
        "n_samples_per_number": int(per_number[0]["n_samples"]),
        "power_note": ("非显著不等于「信号不存在」：每号码样本量越小，最小可检出效应越大；"
                       "请结合 n_samples_per_number 与 null_T_mean_p95 解读"),
    }


def diagnose_ml_channels(draws: List[Dict], colors=("red", "blue"), **kwargs) -> Dict:
    """红/蓝两个通道一起诊断（供 API 一次性返回）。

    **顶层结论取最保守者（D-2）**：两个通道结论不一致时，顶层
    ``trusted`` 必须是 ``all(...)``（任一通道不可信 → 整体不可信），
    ``tier`` 必须取**最保守**的档（只要有一个通道是 ``monitor``，顶层就是
    ``monitor``）。历史缺陷：旧实现用
    ``"signal_found" if any(...) else "monitor"``，于是
    ``ml_red=monitor`` + ``ml_blue=signal_found`` 时顶层给出
    ``trusted=False`` 却 ``tier=signal_found`` —— 两个字段自相矛盾，
    界面上看起来像「有信号但不可信」。

    另外：只要有任一通道处于**演示档**（``advisory_only``），顶层一律
    ``trusted=False`` / ``tier="monitor"``，并给出 ``advisory_warning_zh``。
    """
    # 对外入口默认走 **standard**（B=200）：它是"能不能当结论"的唯一出口，
    # 因此不能沿用底层原语的 fast 默认。调用方显式传 n_null/mode 时以显式值为准。
    if "mode" not in kwargs and "n_null" not in kwargs:
        kwargs = {**kwargs, "mode": "standard"}
    out: Dict = {"ok": True, "channels": {}}
    for c in colors:
        out["channels"][f"ml_{c}"] = diagnose_ml_channel(draws, color=c, **kwargs)
    vals = list(out["channels"].values())
    tiers = {str(v.get("tier")) for v in vals}
    advisory = [str(v.get("channel")) for v in vals if v.get("advisory_only")]
    out["channel_tiers"] = {str(v.get("channel")): v.get("tier") for v in vals}
    out["channel_trusted"] = {str(v.get("channel")): bool(v.get("trusted")) for v in vals}
    # ---- V20 / L0：加一道**前向回测**关（依据 docs/analysis/V19_AI_REDESIGN.md §2.3）----
    #
    # 为什么单靠置换检验不够：2026-10-23 实测 ml_blue 在 B=200 下 p=0.010、
    # 换两个种子都显著（0.010 / 0.005），看起来"稳"；但同一数据做严格前向回测，
    # 命中率 5.00% 低于基线 6.25%（z=-1.26）。**置换检验显著 ≠ 可预测。**
    #
    # 因此：通道要称为 signal_found，必须同时过
    #   ① 置换检验 p < α（diagnose_ml_channel 已算）
    #   ② 前向回测多种子 z 中位数 > WF_Z_MIN
    # 任一条不过 → monitor。这就是本模块被 `walk_forward_gate.combine_gates` 覆盖的位置。
    try:
        from . import walk_forward_gate as _WFG
        _wf = _WFG.walk_forward_check(draws, "stat:freq",
                                      n_tickets=5, start=kwargs.get("wf_start", 2917),
                                      color="blue")
    except Exception as e:                     # noqa: BLE001
        _wf = {"ok": False, "reason_zh": f"前向回测不可用（{type(e).__name__}: {e}）",
               "replayable": None, "z": None, "passed": False}
    out["walk_forward"] = _wf
    _alpha_gate = float(kwargs.get("alpha") or 0.05)
    # ⚠️ 口径纪律（V10-08 / D-2 既有契约，不得破坏）：
    #   **通道级**的 `tier`/`trusted` 是「置换检验本身检出了什么」，必须原样保留
    #   （tests/test_v10_ml_calibration.py::test_t21 守着：ml_red=monitor +
    #   ml_blue=signal_found 时 channel_tiers 必须照实反映）。
    #   前向回测这一关只作用在**顶层**判定上，并且单独暴露为 `dual_gate`。
    #
    #   第一版我直接改了通道的 tier/trusted，把 t21 弄红了 —— 那是越权修改
    #   「置换检验读数的语义」，而它是有独立含义的。
    _dual: Dict[str, Dict] = {}
    for c_name, c_val in out["channels"].items():
        # 兼容：通道结果里没有 `p_value` 时（内置替身/旧调用方），
        # 沿用「通道自己的 tier」作为置换检验那一关的结论 ——
        # 否则会把"没带 p 值"误判成"未做置换检验"，把原本 signal 的通道全降级，
        # 那是口径错位而不是收紧（V20 实测踩到，t21 变红）。
        _perm = c_val.get("p_value")
        if _perm is None and c_val.get("tier") == "signal_found":
            _perm = 0.0          # 通道已判 signal，等价于置换检验显著
        _g = _WFG.combine_gates(_perm, _wf, alpha=_alpha_gate)
        c_val["dual_gate"] = _g
        c_val["tier_permutation_only"] = c_val.get("tier")   # 便于前端对照
        _dual[c_name] = _g
    vals = list(out["channels"].values())
    tiers = {str(v.get("tier")) for v in vals}
    # 顶层是否「全部通道都过了双重门槛」——只有这时才可能是 signal_found
    out["all_dual_pass"] = bool(_dual) and all(
        g.get("tier") == "signal_found" for g in _dual.values())

    # 最保守：任一通道不可信 → 整体不可信
    out["trusted"] = bool(vals) and all(bool(v.get("trusted")) for v in vals)
    # 最保守：任一通道 monitor（或缺失）→ 顶层 monitor；只有全部 signal 才 signal。
    # V20：**再加一道**——全部通道的置换检验都显著还不够，前向回测也必须全过。
    # 这就是「双重门槛」在顶层的落点（ml_blue 实测：置换 p=0.010 但前向回测 z=-0.06）。
    out["tier"] = ("signal_found"
                   if (vals and tiers == {"signal_found"} and out["all_dual_pass"])
                   else "monitor")
    if (vals and tiers == {"signal_found"} and not out["all_dual_pass"]):
        out["dual_gate_warning_zh"] = (
            "置换检验全部通道显著，但**前向回测未过**，故顶层降为 monitor。"
            "依据 docs/analysis/V19_AI_REDESIGN.md §2.3：置换检验显著 ≠ 可预测。")
    out["n_channels"] = len(vals)
    out["n_channels_trusted"] = int(sum(1 for v in vals if v.get("trusted")))
    out["advisory_only"] = bool(advisory)
    if advisory:
        out["tier"] = "monitor"
        out["trusted"] = False
        out["advisory_warning_zh"] = (
            f"⚠️ 仅演示（不做结论）：{', '.join(advisory)} 的零复制次数不足，"
            f"无法在 α 水平上下结论。请用 mode='standard'（B=200）或 "
            f"n_null>=79 重跑；在此之前**不得**把 trusted/tier 当作结论展示。")
    else:
        out["advisory_warning_zh"] = None
    if vals and tiers == {"signal_found"} and out["n_channels_trusted"] == len(vals):
        out["consistency_zh"] = "所有通道一致：均检出信号（仍需独立样本复现）。"
    elif not out["trusted"]:
        out["consistency_zh"] = (
            f"通道结论不一致或均为 monitor（{out['channel_tiers']}）："
            f"顶层按**最保守**口径给出 trusted=False / tier=monitor。")
    else:
        out["consistency_zh"] = "所有通道一致。"
    return out


# ---------- 训练 ----------

def train_red_models(
    draws: List[Dict], min_start: Optional[int] = None
) -> Optional[Dict]:
    """训练 33 个红球号码：集成投票 + sigmoid 校准。返回模型与指标。"""
    if not HAS_SKLEARN:
        print("[ml] sklearn 未安装，跳过 ML 模型")
        return None
    min_start = min_start or config.ML_MIN_START
    if len(draws) < min_start + 10:
        print(f"[ml] 历史数据不足（{len(draws)} < {min_start + 10}），跳过训练")
        return None
    t0 = time.time()
    datasets = _build_red_datasets(draws, min_start)

    models: Dict[int, object] = {}
    metrics: Dict[int, Dict] = {}
    all_bin_p, all_bin_y = [], []

    for num in range(1, 34):
        X, y = datasets[num]
        X_tr, y_tr, X_te, y_te = _split(X, y)
        if len(np.unique(y_tr)) < 2 or len(y_tr) < 20:
            continue
        clf = _red_voting()
        try:
            clf.fit(X_tr, y_tr, sample_weight=_balanced_weights(y_tr))
            cal = _calibrated(clf, X_tr, y_tr, cv=config.ML_CAL_CV)
        except Exception as e:  # noqa: BLE001
            print(f"[ml] 红球号码 {num} 训练失败: {e}")
            continue
        models[num] = cal

        if len(X_te) > 0:
            p_raw = clf.predict_proba(X_te)[:, 1]
            p_cal = cal.predict_proba(X_te)[:, 1]
            b_raw = float(np.mean((p_raw - y_te) ** 2))
            b_cal = float(np.mean((p_cal - y_te) ** 2))
            eps = 1e-9
            ll_raw = float(-np.mean(y_te * np.log(np.clip(p_raw, eps, 1)) +
                                   (1 - y_te) * np.log(np.clip(1 - p_raw, eps, 1))))
            ll_cal = float(-np.mean(y_te * np.log(np.clip(p_cal, eps, 1)) +
                                    (1 - y_te) * np.log(np.clip(1 - p_cal, eps, 1))))
            metrics[num] = {
                "brier_raw": round(b_raw, 4), "brier_cal": round(b_cal, 4),
                "logloss_raw": round(ll_raw, 4), "logloss_cal": round(ll_cal, 4),
                "n_test": int(len(y_te)),
            }
            all_bin_p.extend(p_cal.tolist())
            all_bin_y.extend(y_te.tolist())

    if not models:
        return None
    calibration = _bin_metrics(np.array(all_bin_p), np.array(all_bin_y))
    result = {
        "models": models,
        "metrics": metrics,
        "calibration": calibration,
        "feature_names": _FEATURE_NAMES,
        "train_seconds": round(time.time() - t0, 1),
    }
    if metrics:
        avg_b = float(np.mean([m["brier_cal"] for m in metrics.values()]))
        print(f"[ml] 红球模型训练完成，{len(models)} 个号码，"
              f"平均测试 Brier(校准后)={avg_b:.4f}，ECE={calibration['ece']:.4f}，"
              f"耗时 {result['train_seconds']}s")
    return result


def train_blue_models(
    draws: List[Dict], min_start: Optional[int] = None
) -> Optional[Dict]:
    """训练 16 个蓝球号码：三模型投票 + 校准（蓝球独立建模）。"""
    if not HAS_SKLEARN:
        return None
    min_start = min_start or config.ML_MIN_START
    if len(draws) < min_start + 10:
        return None
    t0 = time.time()
    datasets = _build_blue_datasets(draws, min_start)

    models: Dict[int, object] = {}
    metrics: Dict[int, Dict] = {}
    all_bin_p, all_bin_y = [], []

    for num in range(1, 17):
        X, y = datasets[num]
        X_tr, y_tr, X_te, y_te = _split(X, y)
        if len(np.unique(y_tr)) < 2 or len(y_tr) < 20:
            continue
        clf = _blue_voting()
        try:
            clf.fit(X_tr, y_tr, sample_weight=_balanced_weights(y_tr))
            cal = _calibrated(clf, X_tr, y_tr, cv=config.ML_CAL_CV)
        except Exception as e:  # noqa: BLE001
            print(f"[ml] 蓝球号码 {num} 训练失败: {e}")
            continue
        models[num] = cal
        if len(X_te) > 0:
            p_raw = clf.predict_proba(X_te)[:, 1]
            p_cal = cal.predict_proba(X_te)[:, 1]
            b_raw = float(np.mean((p_raw - y_te) ** 2))
            b_cal = float(np.mean((p_cal - y_te) ** 2))
            eps = 1e-9
            ll_cal = float(-np.mean(y_te * np.log(np.clip(p_cal, eps, 1)) +
                                    (1 - y_te) * np.log(np.clip(1 - p_cal, eps, 1))))
            metrics[num] = {
                "brier_raw": round(b_raw, 4), "brier_cal": round(b_cal, 4),
                "logloss_cal": round(ll_cal, 4), "n_test": int(len(y_te)),
            }
            all_bin_p.extend(p_cal.tolist())
            all_bin_y.extend(y_te.tolist())

    if not models:
        return None
    calibration = _bin_metrics(np.array(all_bin_p), np.array(all_bin_y))
    return {
        "models": models,
        "metrics": metrics,
        "calibration": calibration,
        "feature_names": _BLUE_FEATURE_NAMES,
        "train_seconds": round(time.time() - t0, 1),
    }


# ---------- 模块级缓存（数据版本 + 特征版本变化才重训） ----------

_ML_CACHE: Dict[str, Dict] = {}
_CACHE_LOCK = threading.Lock()
_TRAIN_EVENT: Optional[threading.Event] = None
_TRAIN_KEY: Optional[str] = None
_TRAIN_LOCK = threading.Lock()


def model_schema_version() -> str:
    """当前特征/结构版本（V10-08）。读 config 而非固化，便于测试与热更新。"""
    return str(getattr(config, "MODEL_SCHEMA_VERSION", "0"))


def _cache_key(draws: List[Dict]) -> Optional[str]:
    """缓存键 = 特征版本 + 数据版本。

    历史缺陷（R1 P2-5）：旧键只有 ``"{期数}:{末期号}"``，改特征后 key 不变，
    于是 `data/ml_models/<key>.joblib`（55MB/个）被静默复用，模型与代码不一致。
    现在把 ``MODEL_SCHEMA_VERSION`` 放进 key：改特征必须递增该常量，旧缓存自动失效。
    """
    if not draws:
        return None
    return f"v{model_schema_version()}:{len(draws)}:{draws[-1]['issue']}"


def model_cache_dir() -> "object":
    return config.DATA_DIR / "ml_models"


def prune_model_cache(keep: int = 2, dry_run: bool = False) -> Dict:
    """只保留最近 ``keep`` 个模型 joblib（按 mtime），删除其余（V10-08 磁盘治理）。

    `data/ml_models/` 里的 joblib 是**可重训缓存**：每个文件是
    ``{"red": {models, metrics, calibration}, "blue": ..., "key", "trained_at",
    "red_brier", "blue_brier"}``，由 `data/ssq.db` 的开奖历史训练得到
    （单个约 55MB = 33 个红球 + 16 个蓝球模型）。删除只影响「下次要不要重训」，
    不丢任何源数据（源数据只有 ``data/ssq.db``）。只有在确认 DB 可读
    （``db.count_draws() > 0``）时调用本函数才是安全的。

    可执行验收（V10-08）：``ls data/ml_models/*.joblib | wc -l`` ≤ 2。

    返回 ``{"kept": [...], "removed": [...], "freed_bytes": int, "dir": str}``；
    ``dry_run=True`` 只报告不删除。
    """
    d = model_cache_dir()
    out: Dict = {"kept": [], "removed": [], "freed_bytes": 0, "dir": str(d)}
    if not d.exists():
        return out
    files = sorted(
        (p for p in d.glob("*.joblib") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    for p in files[:max(0, int(keep))]:
        out["kept"].append(p.name)
    for p in files[max(0, int(keep)):]:
        try:
            size = p.stat().st_size
            if not dry_run:
                p.unlink()
            out["removed"].append(p.name)
            out["freed_bytes"] += size
        except OSError as e:  # noqa: BLE001
            print(f"[ml] 删除旧模型失败 {p.name}: {e}")
    return out


def ml_peek(draws: List[Dict]) -> Optional[Dict]:
    """非阻塞查看缓存（从不触发训练）。"""
    key = _cache_key(draws)
    if key is None:
        return None
    with _CACHE_LOCK:
        return _ML_CACHE.get(key)


def ml_training_in_progress() -> bool:
    with _TRAIN_LOCK:
        return _TRAIN_EVENT is not None


def ml_ready(draws: List[Dict]) -> bool:
    """缓存是否已就绪（不触发训练）。供 engine 查询。"""
    return ml_peek(draws) is not None


def _model_path(key: str) -> "object":
    return model_cache_dir() / f"{key}.joblib"


def get_ml_models(draws: List[Dict], force: bool = False) -> Optional[Dict]:
    """返回 {"red":..., "blue":..., "key":..., "ready": bool}。

    数据版本未变则命中缓存；多个调用方并发训练时只训一次，
    其余线程等待同一训练完成后直接读缓存。
    """
    global _TRAIN_EVENT, _TRAIN_KEY
    if not HAS_SKLEARN or not config.ML_ENABLED:
        return None
    key = _cache_key(draws)
    if key is None:
        return None
    with _CACHE_LOCK:
        if not force and key in _ML_CACHE:
            return _ML_CACHE[key]
        if not force and _ML_CACHE.get("error", {}).get("key") == key:
            return None

    # 磁盘持久化：重建容器/重启后直接从 data 卷加载，无需重训
    if not force:
        path = _model_path(key)
        if path.exists():
            try:
                import joblib
                entry = joblib.load(path)
                if entry and entry.get("key") == key and entry.get("red") and entry.get("blue"):
                    with _CACHE_LOCK:
                        _ML_CACHE.clear()
                        _ML_CACHE[key] = entry
                    print(f"[ml] 已从磁盘加载模型（{key}）")
                    return entry
            except Exception as e:  # noqa: BLE001
                print(f"[ml] 磁盘模型加载失败，重新训练: {e}")

    # 已有同 key 训练在进行：等待其完成
    with _TRAIN_LOCK:
        wait_ev = _TRAIN_EVENT if (_TRAIN_EVENT is not None and _TRAIN_KEY == key) else None
    if wait_ev is not None:
        wait_ev.wait()
        with _CACHE_LOCK:
            if key in _ML_CACHE:
                return _ML_CACHE[key]
        return None

    # 抢到训练权
    ev = threading.Event()
    with _TRAIN_LOCK:
        if _TRAIN_EVENT is not None and _TRAIN_KEY == key:
            wait_ev = _TRAIN_EVENT
            _TRAIN_EVENT = None  # 本线程不重复训练
        else:
            wait_ev = None
            _TRAIN_EVENT = ev
            _TRAIN_KEY = key
    if wait_ev is not None:
        wait_ev.wait()
        with _CACHE_LOCK:
            if key in _ML_CACHE:
                return _ML_CACHE[key]
        return None

    try:
        red = train_red_models(draws)
        blue = train_blue_models(draws)
        if red is None:
            with _CACHE_LOCK:
                _ML_CACHE["error"] = {"key": key, "reason": "数据不足"}
            return None
        entry = {
            "red": red, "blue": blue, "key": key,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "red_brier": round(float(np.mean([m["brier_cal"]
                                              for m in red["metrics"].values()])), 4)
            if red["metrics"] else None,
            "blue_brier": round(float(np.mean([m["brier_cal"]
                                               for m in blue["metrics"].values()])), 4)
            if blue and blue["metrics"] else None,
        }
        with _CACHE_LOCK:
            _ML_CACHE.clear()
            _ML_CACHE[key] = entry
        try:
            import joblib
            path = _model_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(entry, path)
            print(f"[ml] 模型已持久化到 {path.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[ml] 模型持久化失败: {e}")
        return entry
    finally:
        with _TRAIN_LOCK:
            _TRAIN_EVENT = None
            _TRAIN_KEY = None
        ev.set()


def ml_status(draws: List[Dict]) -> Dict:
    """供 /api/ml/status 使用。"""
    if not HAS_SKLEARN:
        return {"enabled": False, "sklearn": False, "reason": "scikit-learn 未安装"}
    if not config.ML_ENABLED:
        return {"enabled": False, "sklearn": True, "reason": "LOTT_ML_ENABLED=0"}
    entry = ml_peek(draws)
    if entry is None:
        reason = "后台训练中…" if ml_training_in_progress() else "尚未训练（首个预测请求自动触发）"
        return {"enabled": True, "sklearn": True, "ready": False, "reason": reason}
    return {
        "enabled": True, "sklearn": True, "ready": True,
        "key": entry["key"], "trained_at": entry["trained_at"],
        "red_brier": entry["red_brier"], "blue_brier": entry["blue_brier"],
        "red_calibration_ece": entry["red"]["calibration"]["ece"],
        "blue_calibration_ece": entry["blue"]["calibration"]["ece"],
        "red_train_seconds": entry["red"]["train_seconds"],
        "blue_train_seconds": entry["blue"]["train_seconds"],
    }


def start_ml_warmup(draws: List[Dict]) -> None:
    """后台线程预热 ML 模型，避免首个预测请求阻塞。"""
    if not HAS_SKLEARN or not config.ML_ENABLED or not draws:
        return
    def _run():
        try:
            get_ml_models(draws)
        except Exception as e:  # noqa: BLE001
            print(f"[ml] 后台预热失败: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------- 预测接口 ----------

def predict_red_probs(ml_red: Optional[Dict], draws: List[Dict],
                      normalize: bool = True) -> np.ndarray:
    """使用 ML 模型预测下期红球概率（33 维）。"""
    if ml_red is None or not HAS_SKLEARN or "models" not in ml_red:
        return np.full(33, 1.0 / 33)
    history = draws
    probs = np.zeros(33)
    for num in range(1, 34):
        if num in ml_red["models"]:
            feats = np.array([_compute_features(history, num)])
            try:
                probs[num - 1] = ml_red["models"][num].predict_proba(feats)[0, 1]
            except Exception:  # noqa: BLE001
                probs[num - 1] = 0.0
    s = probs.sum()
    if s <= 0:
        return np.full(33, 1.0 / 33)
    if normalize:
        probs = probs / s * 6  # 期望命中 6 个（与融合层一致）
    else:
        probs = probs / s
    return probs


def predict_blue_probs(ml_blue: Optional[Dict], draws: List[Dict]) -> np.ndarray:
    """使用 ML 模型预测下期蓝球概率（16 维，归一化为概率分布）。"""
    if ml_blue is None or not HAS_SKLEARN or "models" not in ml_blue:
        return np.full(16, 1.0 / 16)
    history = draws
    probs = np.zeros(16)
    for num in range(1, 17):
        if num in ml_blue["models"]:
            feats = np.array([_compute_blue_features(history, num)])
            try:
                probs[num - 1] = ml_blue["models"][num].predict_proba(feats)[0, 1]
            except Exception:  # noqa: BLE001
                probs[num - 1] = 0.0
    s = probs.sum()
    if s <= 0:
        return np.full(16, 1.0 / 16)
    return probs / s


def get_ml_metrics(ml_red: Optional[Dict], ml_blue: Optional[Dict]) -> Dict:
    """聚合 ML 模型评估指标（供 API/前端展示）。"""
    metrics = {}
    if ml_red and ml_red.get("metrics"):
        ms = list(ml_red["metrics"].values())
        metrics["red_avg_brier_raw"] = round(float(np.mean([m["brier_raw"] for m in ms])), 4)
        metrics["red_avg_brier_cal"] = round(float(np.mean([m["brier_cal"] for m in ms])), 4)
        metrics["red_avg_logloss_raw"] = round(float(np.mean([m["logloss_raw"] for m in ms])), 4)
        metrics["red_avg_logloss_cal"] = round(float(np.mean([m["logloss_cal"] for m in ms])), 4)
        metrics["red_ece"] = ml_red["calibration"]["ece"]
        metrics["red_n_test"] = int(sum(m["n_test"] for m in ms))
        metrics["red_calibration"] = ml_red["calibration"]["bins"]
    if ml_blue and ml_blue.get("metrics"):
        ms = list(ml_blue["metrics"].values())
        metrics["blue_avg_brier_raw"] = round(float(np.mean([m["brier_raw"] for m in ms])), 4)
        metrics["blue_avg_brier_cal"] = round(float(np.mean([m["brier_cal"] for m in ms])), 4)
        metrics["blue_avg_logloss_cal"] = round(float(np.mean([m["logloss_cal"] for m in ms])), 4)
        metrics["blue_ece"] = ml_blue["calibration"]["ece"]
        metrics["blue_n_test"] = int(sum(m["n_test"] for m in ms))
        metrics["blue_calibration"] = ml_blue["calibration"]["bins"]
    return metrics


# ---------- walk-forward 滚动评估（ML vs 均匀基线 + paired 检验） ----------

def evaluate_ml_walkforward(
    draws: List[Dict],
    min_start: Optional[int] = None,
    window: Optional[int] = None,
    refit_every: Optional[int] = None,
    train_window: Optional[int] = 800,
    seed: int = 7,
) -> Dict:
    """滚动评估 ML 概率质量：每 refit_every 期用此刻历史重训一次（防未来信息），
    逐期比较 ML 概率与均匀基线的 Brier / log-loss，并做 paired 显著性检验。

    train_window: 重训时最多使用最近 N 期历史（评估专用，控制耗时；
                  生产预测仍用全量历史训练）。
    """
    if not HAS_SKLEARN:
        return {"ok": False, "error": "scikit-learn 未安装，无法评估"}
    if not config.ML_ENABLED:
        return {"ok": False, "error": "ML 模型已停用（LOTT_ML_ENABLED=0）"}
    min_start = min_start or config.ML_MIN_START
    window = window or config.ML_EVAL_WINDOW
    refit_every = refit_every or config.ML_REFIT_EVERY

    if len(draws) < min_start + 15:
        return {"ok": False, "error": f"历史数据不足（{len(draws)} 期，需 ≥{min_start + 15}）"}

    start = max(min_start, len(draws) - window)
    t0 = time.time()

    red_ml_b, red_uni_b, blue_ml_b, blue_uni_b = [], [], [], []
    red_ml_ll, red_uni_ll, blue_ml_ll, blue_uni_ll = [], [], [], []
    cal_p, cal_y = [], []  # 红球 pooled
    cal_pb, cal_yb = [], []  # 蓝球 pooled
    refits = 0
    red_model = blue_model = None

    for i in range(start, len(draws)):
        history = draws[:i]
        target = draws[i]
        if (i - start) % refit_every == 0 and i > min_start + 5:
            hist = history[-train_window:] if train_window and len(history) > train_window else history
            tmin = min(min_start, max(50, int(len(hist) * 0.6)))
            red_model = train_red_models(hist, min_start=tmin)
            blue_model = train_blue_models(hist, min_start=tmin)
            refits += 1

        # 红球
        oh_r = np.zeros(33)
        for r in target["reds"]:
            oh_r[r - 1] = 1.0
        p_r = predict_red_probs(red_model, history, normalize=False)
        p_r = np.clip(p_r, 1e-6, 1 - 1e-6)
        u_r = np.full(33, 1.0 / 33)
        red_ml_b.append(float(np.mean((p_r - oh_r) ** 2)))
        red_uni_b.append(float(np.mean((u_r - oh_r) ** 2)))
        red_ml_ll.append(float(-np.mean(oh_r * np.log(p_r) + (1 - oh_r) * np.log(1 - p_r))))
        red_uni_ll.append(float(-np.mean(oh_r * np.log(u_r) + (1 - oh_r) * np.log(1 - u_r))))
        cal_p.extend(p_r.tolist())
        cal_y.extend(oh_r.tolist())

        # 蓝球
        oh_b = np.zeros(16)
        oh_b[target["blue"] - 1] = 1.0
        p_b = predict_blue_probs(blue_model, history)
        p_b = np.clip(p_b, 1e-6, 1 - 1e-6)
        u_b = np.full(16, 1.0 / 16)
        blue_ml_b.append(float(np.mean((p_b - oh_b) ** 2)))
        blue_uni_b.append(float(np.mean((u_b - oh_b) ** 2)))
        blue_ml_ll.append(float(-np.mean(oh_b * np.log(p_b) + (1 - oh_b) * np.log(1 - p_b))))
        blue_uni_ll.append(float(-np.mean(oh_b * np.log(u_b) + (1 - oh_b) * np.log(1 - u_b))))
        cal_pb.extend(p_b.tolist())
        cal_yb.extend(oh_b.tolist())

    if not red_ml_b:
        return {"ok": False, "error": "评估窗口为空"}

    def _pair_p(delta: List[float]) -> Dict:
        """delta>0 表示 ML 更优（基线 Brier 更大）。"""
        from scipy import stats as _st
        d = np.array(delta)
        nz = d[d != 0]
        if len(nz) == 0:
            return {"p": 1.0, "method": "no-difference", "mean_delta": 0.0}
        mean_d = float(np.mean(d))
        try:
            if len(nz) >= 5:
                res = _st.wilcoxon(nz)
                return {"p": round(float(res.pvalue), 4), "method": "wilcoxon",
                        "mean_delta": round(mean_d, 5)}
            pos = int(np.sum(nz > 0))
            pv = 2 * min(_st.binom.cdf(pos, len(nz), 0.5),
                         1 - _st.binom.cdf(pos - 1, len(nz), 0.5))
            return {"p": round(float(min(pv, 1.0)), 4), "method": "sign-test",
                    "mean_delta": round(mean_d, 5)}
        except Exception:  # noqa: BLE001
            return {"p": 1.0, "method": "n/a", "mean_delta": round(mean_d, 5)}

    res = {
        "ok": True,
        "n_issues": len(red_ml_b),
        "window": window,
        "refit_every": refit_every,
        "train_window": train_window,
        "refits": refits,
        "seconds": round(time.time() - t0, 1),
        "red": {
            "brier_ml": round(float(np.mean(red_ml_b)), 4),
            "brier_uniform": round(float(np.mean(red_uni_b)), 4),
            "logloss_ml": round(float(np.mean(red_ml_ll)), 4),
            "logloss_uniform": round(float(np.mean(red_uni_ll)), 4),
            "paired": _pair_p([round(a - b, 6) for a, b in zip(red_uni_b, red_ml_b)]),
            "calibration": _bin_metrics(np.array(cal_p), np.array(cal_y)).get("bins", []),
        },
        "blue": {
            "brier_ml": round(float(np.mean(blue_ml_b)), 4),
            "brier_uniform": round(float(np.mean(blue_uni_b)), 4),
            "logloss_ml": round(float(np.mean(blue_ml_ll)), 4),
            "logloss_uniform": round(float(np.mean(blue_uni_ll)), 4),
            "paired": _pair_p([round(a - b, 6) for a, b in zip(blue_uni_b, blue_ml_b)]),
            "calibration": _bin_metrics(np.array(cal_pb), np.array(cal_yb)).get("bins", []),
        },
        "conclusion": (
            "ML 概率与均匀基线通常无显著差异（随机彩票的正常结论）；"
            "Brier/log-loss/校准曲线用于量化模型是否被校准，而非承诺命中率提升。"
        ),
    }
    print(f"[ml] walk-forward 评估完成：{res['n_issues']} 期 / {refits} 次重训，"
          f"耗时 {res['seconds']}s")
    return res


# 兼容别名（供旧调用点使用）
train_red_model = train_red_models
train_blue_model = train_blue_models
