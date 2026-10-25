"""大众偏好度模型（B-2）：预测「这一注号码会有多少人跟着中奖」。

## 它要回答的问题，以及它**不**回答的问题

本系统已反复证明：**号码本身不可预测**。红球频次 χ²=44.84/df=32（p=0.065）、
蓝球 χ²=8.81（p=0.887）、单球边际熵 5.0429 vs 理论上限 5.0444（差 0.0015 bit，噪声内）、
条件熵「给定上期是否出现」只降 0.0076 bit、最强候选在最大统计量零分布下 p=0.2995。
**任何声称"预测下期开什么号"的模型都是假的。**

但这个模型预测的不是号码，而是**人群行为**：`p1`（一等奖中奖注数）。
号码是随机的，而选号的人不是 —— 大量彩民按生日、同尾、连号选号，
所以「哪些号码组合会被很多人选」是**可学习的**。

    ❌ 不回答：下期会开什么号（做不到）
    ✅ 回答：这一注号码"大众程度"有多高，从而在真中奖时预计会被摊薄多少

## 实测效应量（务必如实告知用户）

| 检验 | 值 |
|---|---|
| 形态对 `p1` 的增量 R²（控制销量后，扩展窗口时序 CV） | **+0.0077**（0.77%） |
| 不同折数（3/5/8/10）下 | **恒为 +0.0077**（不随切分漂移） |
| 剔除上 1% 异常值后 | +0.0186 |
| 置换检验（200 次） | **p = 0.0050** |
| 销量本身的 R² | +0.0205（`p1` 主要由销量驱动） |

**0.77% 的解释力意味着：它能给出粗略排序，不能给出可用数值。**
所以对外只输出「档位」（低/中/高），**不输出精确预测注数** —— 后者会让人误以为精确。

## 钱的角度（这是唯一能兑现的价值，也是唯一必须同屏披露的限制）

头奖单注金额 = 当期一等奖总奖金 ÷ 中奖注数。历史实测（3266 期）：

    极端热门  p1≥21 : 平均 5,212,219 元
    常态      4≤p1≤10: 平均 7,636,006 元

但**不能**把这个差额当成"避开热门的收益"：奖池规模才是金额主因
（`rho(pool,amt)=+0.50`），而金额对 p1 的弹性实测只有 **−0.088**
（纯恒等式 `amt=pool/p1` 应为 −1；两者几乎抵消）。
换算成期望增益只有 **0.005 ~ 0.18 元/注**（占 2 元成本的 0.3%~9%）。

**而中奖概率完全不变，长期期望仍为负（单注 EV≈1.0458 元 / 2 元 = 52.3%）。**
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 模型版本（口径变更时必须升版，断点/缓存据此失效）
MODEL_VERSION = "pop-1"

#: 档位划分的**分位点**（在「预测值自己的历史分布」上标定，不是固定注数）。
#:
#: 为什么不能写死注数（2026-10-21 实测踩到）：预测值的中位是 8.8、真实 `p1` 中位是 6
#: —— 模型系统性高估（因为目标是"每亿销量的注数"，而销量分布右偏）。
#: 若阈值写死 `≤4 / ≥11`，落在 low 档的只有 17%、high 档只有 19%，
#: 且**与真实的冷热程度不对应**（`[1,2,3,4,5,6]` 被判成"常见"）。
#: 改用分位后，三档各占约 1/3，且排序与真实冷热一致（这是排序模型，不是回归模型）。
TIER_LOW_Q = 0.33         # 预测值下三分位以下 → 偏冷门
TIER_HIGH_Q = 0.67        # 上三分位以上 → 偏热门

#: 形态特征名（顺序即特征顺序，不可随意调整）
FEATURE_NAMES: Tuple[str, ...] = (
    "birthday_count",     # ≤31 的号码个数（生日号效应，最强的单项）
    "same_tail",          # 同尾号重复数（6 - 不同个位数）
    "consecutive",        # 连号个数
    "sum_reds",           # 红球和值
    "spread",             # 极差（max - min）
    "even_count",         # 偶数个数
    "zone_count",         # 三区覆盖数（1-11/12-22/23-33）
)


def features_of(reds: Sequence[int]) -> Dict[str, float]:
    """从 6 个红球算形态特征（纯函数，可单测）。

    这些特征全部来自**号码的外形**，与开奖概率无关 —— 它们影响的是
    「多少人会选这种号」，不是「这种号会不会开」。
    """
    r = sorted(int(x) for x in reds)
    if len(r) != 6:
        raise ValueError(f"需要 6 个红球，收到 {len(r)} 个")
    return {
        "birthday_count": float(sum(1 for v in r if v <= 31)),
        "same_tail": float(6 - len({v % 10 for v in r})),
        "consecutive": float(sum(1 for i in range(1, 6) if r[i] - r[i - 1] == 1)),
        "sum_reds": float(sum(r)),
        "spread": float(r[-1] - r[0]),
        "even_count": float(sum(1 for v in r if v % 2 == 0)),
        "zone_count": float(len({(v - 1) // 11 for v in r})),
    }


def feature_vector(reds: Sequence[int]) -> List[float]:
    f = features_of(reds)
    return [f[k] for k in FEATURE_NAMES]


def tier_of(predicted_p1: float, lo: float, hi: float) -> str:
    """按**分位阈值**把预测值映射成档位（对外只给档位，不给数字）。

    `lo`/`hi` 由 `PopularityModel.fit()` 在训练集预测值上算出并保存，
    这样档位始终是「相对同期彩民习惯」的三分位，而不是绝对注数。
    """
    if predicted_p1 <= lo:
        return "low"          # 偏冷门：该形态历史上较少人覆盖
    if predicted_p1 >= hi:
        return "high"         # 偏热门：可能被较多彩民覆盖
    return "mid"


TIER_LABELS: Dict[str, str] = {
    "low": "偏冷门",
    "mid": "常见",
    "high": "偏热门",
}

#: 档位对应的诚实说明（前端必须原样展示，不得删改）
TIER_CAVEATS: Dict[str, str] = {
    "low": "该号码形态在历史上被较少彩民选中。**这不提高中奖概率**；"
           "只在中奖时略微减少分奖人数。",
    "mid": "该号码形态的大众程度接近历史中位。",
    "high": "该号码形态历史上被较多彩民选中（如偏小号、同尾集中）。"
            "同样中奖时可能分给更多人。**这不降低中奖概率**。",
}


class PopularityModel:
    """线性最小二乘（岭回归）—— 刻意用最简单、最可解释的模型。

    ## 为什么不用 GBDT / 神经网络

    实测对照（时序切分，样本外 R²）：

        Ridge               : +0.0232
        HistGB              : +0.0099
        HistGB(仅形态)       : −0.0544   ← 过拟合，比均值基线还差

    形态的真实增量只有 **0.77%**，在这个量级上复杂模型只会把噪声学进去。
    更重要的理由：**可解释**。线性系数能直接告诉用户"小号越多越热门"，
    这正是本功能要传达的信息；一个 R²=0.01 的黑箱毫无产品价值。

    ## 为什么先做残差化

    `p1` 主要由销量驱动（R² +0.0205），而销量与号码形态无关。
    若不先扣除销量，模型会把"这期销量高"误学成"这种形态热门"。
    因此训练目标是 ``p1 / (sales / 1e8)`` —— 每亿元销量的中奖注数。
    """

    def __init__(self) -> None:
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.n_train: int = 0
        self.r2_increment: float = 0.0
        self.p_value: float = 1.0
        self.fitted: bool = False
        #: 档位分位阈值（在训练集预测值上标定）
        self.tier_lo: float = 0.0
        self.tier_hi: float = 0.0

    # ---------- 训练 ----------

    def fit(self, draws: Sequence[Dict]) -> Dict[str, Any]:
        """用历史开奖拟合。**样本外验证与训练在同一函数里完成并留档**。

        只使用 ``p1`` 与 ``sales`` 都存在的期次（实测 3428 期）。
        """
        rows = [d for d in draws
                if d.get("p1") is not None and d.get("sales")
                and d.get("reds") and len(d["reds"]) == 6]
        if len(rows) < 300:
            return {"ok": False, "reason": "insufficient_data",
                    "note_zh": f"样本不足（{len(rows)} 期，需 ≥300）"}

        # 标准化目标：每亿元销量的中奖注数
        y = [float(d["p1"]) / (float(d["sales"]) / 1e8) for d in rows]
        X = [feature_vector(d["reds"]) for d in rows]

        n = len(rows)
        split = int(n * 0.7)
        self.r2_increment, self.p_value = _out_of_sample_check(X, y, split)

        # 全量重拟合（部署用）
        self.coef_, self.intercept_ = _ols(X, y)
        self.n_train = n
        self.fitted = True
        # 档位分位阈值：在训练集自己的预测值上标定
        preds = sorted(self._raw_predict(x) for x in X)
        self.tier_lo = _quantile(preds, TIER_LOW_Q)
        self.tier_hi = _quantile(preds, TIER_HIGH_Q)
        return {
            "ok": True,
            "n_train": n,
            "model_version": MODEL_VERSION,
            "r2_increment": round(self.r2_increment, 5),
            "p_value": round(self.p_value, 4),
            "coefficients": dict(zip(FEATURE_NAMES, [round(c, 4) for c in self.coef_])),
            "note_zh": (
                f"形态对头奖注数的增量解释力 R²={self.r2_increment:.4f}"
                f"（置换 p={self.p_value:.4f}）。**效应真实但极小**："
                f"可给出粗略排序，不能给出可用数值；不提高中奖概率。"
            ),
        }

    # ---------- 预测 ----------

    def _raw_predict(self, x: Sequence[float]) -> float:
        """标准化空间的预测（每亿元销量的中奖注数）。"""
        if self.coef_ is None:
            return 0.0
        return max(self.intercept_ + sum(c * v for c, v in zip(self.coef_, x)), 0.0)

    def predict_p1(self, reds: Sequence[int], sales: Optional[float] = None) -> Optional[float]:
        """预测该号码形态的"每亿元销量中奖注数"；传 sales 时换算成注数。"""
        if not self.fitted or self.coef_ is None:
            return None
        z = self._raw_predict(feature_vector(reds))
        if sales:
            return z * (float(sales) / 1e8)
        return z

    def describe(self, reds: Sequence[int], sales: Optional[float] = None) -> Dict[str, Any]:
        """对外输出：档位 + 依据 + 强制披露。**不返回精确注数。**

        ## 口径必须一致（2026-10-21 实测踩到）

        `tier_lo` / `tier_hi` 是在**标准化空间**（每亿元销量的注数）上标定的，
        所以档位也必须用同一口径的预测值来比较。

        第一版写成 ``pred = self.predict_p1(reds, sales)``，传了 `sales` 时预测值
        被乘上 `sales/1e8`（实测约 3.3 倍），再拿去和按未乘销量标定的阈值比 ——
        结果 3056/3428 条历史期次全被判成 "high"，档位完全失效。
        现在档位一律用未乘销量的标准化预测值，`sales` 只影响（已停用的）注数换算。
        """
        if not self.fitted or self.coef_ is None:
            # ===== 区分「未就绪」的三种原因（2026-10-21）=====
            #
            # 旧版只回 `reason="model_not_fitted"`，端点再压成一句
            # 「模型未就绪（数据不足或尚未拟合）」—— 把三种「用户该做完全不同事」
            # 的情况合并了：
            #   · warming          预热中，约几秒就好 → 刷新
            #   · insufficient_data 真的样本不足（<300 期）→ 短期无解
            #   · fit_failed       拟合异常 → 需要看日志/修代码
            # 合并会让预热期用户以为「数据永远不够，这功能废了」。
            # 这与 D-07「一个 usable 两个意思」同类：拆字段，不要拆文案。
            if _WARMING.is_set():
                why = "warming"
            elif _RETRY_EXHAUSTED:
                # 重试已耗尽：必须与「正在重试」区分开，否则又是一个假承诺。
                why = "retry_exhausted"
            else:
                why = _LAST_UNFIT_REASON
            return {
                "ok": False,
                "reason": "model_not_fitted",
                "unfitted_reason": why,
            }

        z = self._raw_predict(feature_vector(reds))       # 标准化空间，与阈值同口径
        tier = tier_of(z, self.tier_lo, self.tier_hi)
        f = features_of(reds)

        reasons: List[str] = []
        if self.coef_:
            order = sorted(zip(FEATURE_NAMES, self.coef_),
                           key=lambda kv: -abs(kv[1] * f[kv[0]]))
            for name, coef in order[:2]:
                contribution = coef * f[name]
                if abs(contribution) < 1e-9:
                    continue
                word = "偏热门" if contribution > 0 else "偏冷门"
                reasons.append(f"{_FEATURE_ZH[name]}（{f[name]:g}）使其{word}")

        return {
            "ok": True,
            "model_version": MODEL_VERSION,
            "tier": tier,
            "tier_label": TIER_LABELS[tier],
            "reasons": reasons,
            "caveat": TIER_CAVEATS[tier],
            "honesty": (
                "本项只反映「多少人可能选同样的号」；"
                "中头奖概率恒为 1/17,721,088，不受选号影响。"
                "期望收益仍为负（约 52.3% 回收）。"
            ),
            "r2_increment": round(self.r2_increment, 5),
        }


_FEATURE_ZH = {
    "birthday_count": "生日号个数",
    "same_tail": "同尾号",
    "consecutive": "连号",
    "sum_reds": "和值",
    "spread": "极差",
    "even_count": "偶数个数",
    "zone_count": "三区覆盖",
}


# ---------- 数值内核（不依赖 numpy，便于单测与低依赖运行） ----------

def _ols(X: Sequence[Sequence[float]], y: Sequence[float],
         ridge: float = 1e-6) -> Tuple[List[float], float]:
    """最小二乘（带极小岭项保证可解）。返回 ``(coef, intercept)``。

    用正规方程 + 高斯消元，特征数只有 7，规模极小，无需 numpy。
    """
    n = len(X)
    p = len(X[0]) if n else 0
    if not n or not p:
        return [], 0.0

    # 中心化，避免截距与特征共线
    mx = [sum(row[j] for row in X) / n for j in range(p)]
    my = sum(y) / n
    Xc = [[row[j] - mx[j] for j in range(p)] for row in X]
    yc = [v - my for v in y]

    # A = Xc^T Xc + ridge*I, b = Xc^T yc
    A = [[sum(Xc[i][a] * Xc[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    for j in range(p):
        A[j][j] += ridge * n
    b = [sum(Xc[i][j] * yc[i] for i in range(n)) for j in range(p)]

    coef = _solve(A, b)
    intercept = my - sum(c * m for c, m in zip(coef, mx))
    return coef, intercept


def _solve(A: List[List[float]], b: List[float]) -> List[float]:
    """高斯消元（列主元）。p 很小，纯 Python 足够。"""
    p = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(p):
            if r == col:
                continue
            factor = M[r][col]
            if factor:
                M[r] = [a - factor * bc for a, bc in zip(M[r], M[col])]
    return [M[i][p] for i in range(p)]


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if not n:
        return 0.0
    my = sum(y_true) / n
    ss_tot = sum((v - my) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _ts_predict(X: Sequence[Sequence[float]], y: Sequence[float],
                folds: int = 5) -> List[float]:
    """扩展窗口（expanding window）预测：第 k 折用前 k 段训练、预测第 k+1 段。

    **不用随机 KFold**：时间序列上随机切分会让"用未来预测过去"，
    这是本项目历史上反复踩过的坑（如 500 期滑窗造成的 e 值高估）。
    """
    n = len(y)
    size = n // (folds + 1)
    pred = [float("nan")] * n
    for k in range(1, folds + 1):
        tr_end = size * k
        te_end = size * (k + 1)
        coef, icpt = _ols(X[:tr_end], y[:tr_end])
        for i in range(tr_end, te_end):
            pred[i] = icpt + sum(c * v for c, v in zip(coef, X[i]))
    return pred


def _out_of_sample_check(X: Sequence[Sequence[float]], y: Sequence[float],
                         split: int, n_perm: int = 200) -> Tuple[float, float]:
    """返回 ``(形态增量 R², 置换检验 p 值)``。

    增量定义为「含形态的模型 R² − 仅截距（均值）基线 R²」，
    在**扩展窗口时序预测**上计算 —— 与训练集不同分布的部分才算数。

    p 值用置换：打乱 y 后重做同一流程 200 次，看真实增量在零分布中的位置。
    **置换会破坏形态与 y 的联系**，因此零分布正确刻画了"没有信号"时的期望。
    """
    pr_full = _ts_predict(X, y)
    ok = [i for i, v in enumerate(pr_full) if not math.isnan(v)]
    if not ok:
        return 0.0, 1.0
    yt = [y[i] for i in ok]
    pf = [pr_full[i] for i in ok]
    # 基线：训练段均值（用同一批预测点的前序均值近似）
    base = [sum(y[:i]) / i if i else yt[0] for i in ok]
    inc = _r2(yt, pf) - _r2(yt, base)

    # 置换（次数少：纯 Python，200 次 × 1000 点 × 7 特征已约 3~5 秒）
    rng_state = 12345
    null: List[float] = []
    for _ in range(n_perm):
        yp = _shuffle(y, rng_state)
        rng_state += 7919
        pr2 = _ts_predict(X, yp)
        pf2 = [pr2[i] for i in ok]
        null.append(_r2(yt, pf2) - _r2(yt, base))
    ge = sum(1 for v in null if v >= inc)
    return inc, (ge + 1) / (len(null) + 1)


def _shuffle(seq: Sequence[float], seed: int) -> List[float]:
    """确定性洗牌（用 LCG，避免依赖 random 全局状态影响其它模块）。"""
    out = list(seq)
    state = seed
    for i in range(len(out) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2 ** 31)
        j = state % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


# ---------- 进程级单例（避免每次请求都重训） ----------

_CACHE: Optional[PopularityModel] = None

#: 保护 `_CACHE` 的构造过程 —— 见 `get_model()` 的「为什么必须加锁」。
_CACHE_LOCK = threading.Lock()

#: 落盘缓存的默认路径（相对项目根；`config.DATA_DIR` 下）。
_CACHE_FILENAME = "popularity_model.json"

#: 「模型正在拟合」标记 —— 供端点区分「预热中」与「样本不足」。
#:
#: ## 为什么不能用 `_CACHE is None` 判断
#:
#: `get_model_with_deadline()` 超时时返回的是一个**未拟合对象**，且按设计
#: **不写入 `_CACHE`**（避免污染：否则后续请求会一直看到「有对象但未 fitted」，
#: 每次都走完整路径）。所以 `_CACHE is None` 在两种情况下都成立，区分不出来：
#:   · 预热正在跑（等几秒就好）
#:   · 数据不足（`n_train < 300`，短期内无解）
#:
#: 这两件事**用户该做的事完全不同** —— 前者只需刷新，后者是数据问题。
#: 合并成一句「模型未就绪」会让预热期用户以为功能坏了。
#: 这与 D-07「一个 usable 两个意思」是同一类问题：不是数字错，是用户无法区分该做什么。
_WARMING = threading.Event()

#: 「已有后台拟合线程在跑」—— 防止并发请求各自 spawn 一个 fit 线程（20 个请求 = 20 次 fit）。
_FIT_THREAD_ACTIVE = threading.Event()

#: 保护「检查+启动线程」这段临界区（`_FIT_THREAD_ACTIVE` 的检查与置位必须原子）。
_FIT_TRIGGER_LOCK = threading.Lock()

#: 模型就绪信号。请求路径用 `_READY.wait(timeout)` 等别人的成果，
#: **不自己加权** —— 这是「9.29s 全额阻塞」的结构性修复。
_READY = threading.Event()

#: 最近一次拟合**为什么没成功**（`insufficient_data` / `fit_failed` / `unknown`）。
#:
#: 用途：`describe()` 在未拟合时回传 `unfitted_reason`，让端点能给对文案。
#: 现状（qi-delivery/严QA 实测）：`fit()` 返回 `ok=False` 时信息被丢弃，
#: 端点只能笼统说「数据不足」，把「预热中」和「真样本不足」混为一谈。
_LAST_UNFIT_REASON: str = "unknown"

#: 上一次成功拟合实测耗时（秒）。用于给「预热中」的 `Retry-After` **动态估值**。
#:
#: ## 为什么不能写死
#:
#: 实测：同一台机器上 fit 耗时随负载在 5.5s ~ 21s 之间浮动
#: （本机空闲 5.49s / 容器单核 6.06s / 本机 load 4 时 8.35s / load 12 时 21s）。
#: 写死 5s：慢时用户 5 秒后重试仍是 503，以为功能坏了；
#: 写死 20s：快时用户白等 15 秒。
#: 用「上次实测耗时」当估计，两种负载下都接近真实。
_LAST_FIT_SECONDS: Optional[float] = None

#: 当前这次拟合的开始时刻（`time.time()` 口径）。用于给 `retry_after` 一个
#: **已耗时下界**：如果已经跑了 20 秒还说「5 秒后就好」，那依然是个假承诺。
_FIT_STARTED_AT: Optional[float] = None

#: 上一次**预热**（含读盘/拟合全过程）的实测耗时（秒）。
#:
#: 与 `_LAST_FIT_SECONDS` 的区别：后者只计 `fit()` 本身，本值计预热线程的
#: 端到端耗时（含 numpy 导入等），更贴近用户真实等待。
#: 用途：作为**一致性守卫**的对照量 —— 承诺的 `retry_after` 与近期实测是否同量级。
#: ⚠️ 它是「上次」的值，测的是**一致性**（承诺与近期实际是否同量级），
#: 不是**准确性**（这次多久会好）。读它的人不要当精确断言用。
_LAST_WARMUP_SECONDS: Optional[float] = None

#: `suggest_retry_after()` 的上下界（秒）。上限防止异常慢机器给出荒谬的等待。
_RETRY_AFTER_MIN = 5
_RETRY_AFTER_MAX = 30
#: 首次（还没有历史耗时可参考）用的保守估计。
_RETRY_AFTER_INITIAL = 10


def is_warming() -> bool:
    """模型是否正在后台拟合（预热线程或重试循环正在跑 `fit`）。"""
    return _WARMING.is_set()


def warmup_enabled() -> bool:
    """后台预热/重试是否启用（`LOTT_POPULARITY_WARMUP=0` 关闭）。

    ## 为什么这个谓词必须只有一份

    它决定**两件相互依赖**的事：

      * `api_app` —— 要不要起预热线程（进而决定「会不会自动重试」）；
      * `routes_portfolio` —— 503 该说「正在拟合，稍候自动就绪」
        还是「本进程内**不会**自动就绪」。

    如果两处各读一次环境变量，一旦有人改了判断方式（例如支持 `false`/`no`），
    就会出现「线程没起，但端点还承诺会自动好」—— 用户会一直刷新一件永远不回来的事。
    **本项目已经因为两套逻辑各说各话出过问题**（见 `_get_model_locked` 注释）。
    """
    import os
    return (os.environ.get("LOTT_POPULARITY_WARMUP") or "1").strip() != "0"


def _set_unfit_reason(reason: str, exc: Optional[BaseException] = None) -> None:
    """记录「为什么没拟合成功」—— 供端点给出可区分的文案。

    `exc` 非空时额外打一行 error 级日志。**不能只 `pass`**：
    一个被 `except` 吞掉的失败 = 一个永远不会报警的故障
    （本项目已因此让悬空引用静默存在过几个版本）。
    """
    global _LAST_UNFIT_REASON
    _LAST_UNFIT_REASON = reason or "unknown"
    if exc is not None:
        import sys
        print("[popularity] 拟合失败 reason=%s: %s: %s"
              % (_LAST_UNFIT_REASON, type(exc).__name__, exc), file=sys.stderr)




def record_warmup_seconds(seconds: float) -> None:
    """记录一次预热的端到端耗时（由 `api_app._popularity_warmup` 调用）。"""
    global _LAST_WARMUP_SECONDS
    _LAST_WARMUP_SECONDS = float(seconds)


#: 预热重试是否**已耗尽**（`_MAX_ATTEMPTS` 次全失败）。
#:
#: ## 为什么需要这个状态（2026-10-21 第三种「假承诺」）
#
# 前两种已修：
#   ① `warming_disabled` —— 预热被配置关闭，「永不发生」却承诺会重试；
#   ② 固定 `retry_after=5` —— 比实际短 2.6~7.7 倍；
# 这一种是**重试耗尽后仍说「稍候会自动重试」**：
#   预热循环 `for attempt in range(1, 11)` 跑完就退出了，
#   但 `_LAST_UNFIT_REASON` 仍是 `fit_failed`，端点据此给出
#   `retry_after=30` + 「稍候会自动重试」—— 而**不会再有任何重试**。
#   用户会一直刷新一件永远不回来的事。
_RETRY_EXHAUSTED = False


def mark_retry_exhausted() -> None:
    """标记预热重试已耗尽（由 `api_app._popularity_warmup` 在循环结束时调用）。"""
    global _RETRY_EXHAUSTED
    _RETRY_EXHAUSTED = True


def clear_retry_exhausted() -> None:
    """清除「重试已耗尽」标记（新的预热轮次开始、或拟合成功时调用）。"""
    global _RETRY_EXHAUSTED
    _RETRY_EXHAUSTED = False


def is_retry_exhausted() -> bool:
    """重试是否已耗尽 —— 端点据此决定 `retry_after` 该给值还是 `null`。"""
    return _RETRY_EXHAUSTED


def last_warmup_seconds() -> Optional[float]:
    """上次预热实测耗时（秒）；从未预热过时为 `None`。

    **语义**：反映「近期这台机器上预热大概要多久」，供一致性守卫比对
    `retry_after` 是否与之同量级。**不是**对本轮等待时间的精确预测。
    """
    return _LAST_WARMUP_SECONDS


def suggest_retry_after() -> int:
    """给「模型预热中」的建议重试间隔（秒）—— 基于上次实测耗时，动态。

    ## 为什么不写死（Lead 裁决）

    fit 耗时随负载在 **5.5s ~ 21s** 浮动（实测：本机空闲 5.49s、容器单核 6.06s、
    本机 load 4 时 8.35s、load 12 时 21s）。

      * 写死 5s → 慢负载时用户 5 秒后重试仍是 503，**以为功能坏了**；
      * 写死 20s → 快的时候让用户**白等 15 秒**。

    用「上次实测耗时」作为估计，两种负载下都接近真实；首次无历史时用保守值 10s。
    再夹到 `[5, 30]`：上限防止异常慢机器给出荒谬等待，下限避免过于频繁轮询。
    """
    import time as _time
    # 已耗时是**下界**：如果这次已经跑了 20 秒，那「5 秒后就好」必然又是假承诺。
    # 没有历史耗时可参考时，它同时充当估计值（`_RETRY_AFTER_INITIAL` 兜底）。
    elapsed = (_time.time() - _FIT_STARTED_AT) if _FIT_STARTED_AT else 0.0

    # 参考量优先级：预热端到端耗时（含 numpy 导入，最贴近用户真实等待）
    # > 单次 fit 耗时 > 保守初值。取**较大者**与已耗时下界。
    ref = _LAST_WARMUP_SECONDS or _LAST_FIT_SECONDS
    if ref:
        est = max(float(ref), elapsed + 1.0)
    else:
        est = max(float(_RETRY_AFTER_INITIAL), elapsed + 1.0)

    return int(min(_RETRY_AFTER_MAX, max(_RETRY_AFTER_MIN, round(est))))


def reset_cache() -> None:
    """清空**全部**进程内可变状态（测试与 `force_refit` 用）。

    ## 为什么必须清全部，而不只是模型缓存（2026-10-21 实测）

    旧实现只清 `_CACHE` 与 `_READY`，留下 5 个残留：

    | 残留 | 后果 |
    |---|---|
    | `_WARMING` | `is_warming()` 跨用例为真 → 端点误报「预热中」 |
    | `_LAST_WARMUP_SECONDS` | `suggest_retry_after()` 返回**上个用例**的值 |
    | `_LAST_FIT_SECONDS` | 同上 |
    | `_FIT_STARTED_AT` | 参与「已耗时下界」计算 → 估计值异常 |
    | `_LAST_UNFIT_REASON` | `describe()` 报告错误的未就绪原因 |
    | `_RETRY_EXHAUSTED` | 端点误报「不会再重试」 |

    **其中耗时类残留最严重**：新增的「比例守卫」正是读 `_LAST_WARMUP_SECONDS`
    做断言。它一旦被上个用例的残留值影响，守卫就变成 **order-dependent** ——
    同一份代码换个测试顺序就可能红或绿。
    **守卫本身不可信，比没有守卫更糟。**

    这条不是"测试卫生"问题，是"守卫可信度"问题。
    """
    global _CACHE, _LAST_WARMUP_SECONDS, _LAST_FIT_SECONDS, _FIT_STARTED_AT
    global _LAST_UNFIT_REASON, _RETRY_EXHAUSTED
    with _CACHE_LOCK:
        _CACHE = None
        _READY.clear()
        _WARMING.clear()
        _LAST_WARMUP_SECONDS = None
        _LAST_FIT_SECONDS = None
        _FIT_STARTED_AT = None
        _LAST_UNFIT_REASON = "unknown"
        _RETRY_EXHAUSTED = False


# ---------- 落盘缓存（跨进程重启复用，跳过 7 秒重拟合） ----------
#
# ## 为什么需要（2026-10-21 实测）
#
# `fit()` 里 `_out_of_sample_check` 要跑 200 次置换 + 滚动重拟合 ≈ 6~7 秒，
# 而它产出的是 **训练期验证指标**（`r2_increment` / `p_value`），推理时不需要重复计算。
# 每次进程重启都重跑一遍是纯浪费，且直接落在用户请求路径上。
#
# ## 指纹必须覆盖「会影响结果」的输入（否则 K 族第 7 个变体：看起来对）
#
# 只看 `MODEL_VERSION` **不够** —— 新增开奖期次会让 `tier_lo/tier_hi` 过期
# （分位点是在训练集预测分布上标定的）。所以指纹取：
#   · `model_version`  —— 口径变更时手工升版，强制失效
#   · `n_draws`        —— 有效训练期数
#   · `last_issue`     —— 最后一期期号（`n_draws` 相同但换了期次也能察觉）
#   · `n_features`     —— 特征个数（`FEATURE_NAMES` 改动同样会破坏系数向量）
#
# 指纹不匹配时**必须重算**，不允许「先用旧的、后台再更新」——
# 那正是会让用户看到旧值的写法。

def _cache_path() -> Path:
    """落盘缓存路径（`config.DATA_DIR/popularity_model.json`）。"""
    from . import config
    return Path(config.DATA_DIR) / _CACHE_FILENAME


def _fingerprint_of(rows: Sequence[Dict]) -> Dict[str, Any]:
    """由**实际参与训练的行**计算指纹（不是由全库，避免无关期次扰动）。

    `fit()` 只用 `p1` 与 `sales` 都存在的行，所以指纹也必须基于同一集合 ——
    否则「库里新增了一期但没有 p1」会导致无谓的重拟合。
    """
    issues = [str(r.get("issue") or "") for r in rows]
    return {
        "model_version": MODEL_VERSION,
        "n_draws": len(rows),
        "last_issue": max(issues) if issues else "",
        "n_features": len(FEATURE_NAMES),
    }


def _model_to_dict(m: PopularityModel) -> Dict[str, Any]:
    """把已拟合模型序列化成可直接落盘的 dict。"""
    return {
        "coef": list(m.coef_ or []),
        "intercept": m.intercept_,
        "tier_lo": m.tier_lo,
        "tier_hi": m.tier_hi,
        # 训练期指标也要存：`describe()` 会返回 `r2_increment`，
        # 若缓存丢掉它，热启动后该字段会变成 None（Product 特别提醒过）。
        "r2_increment": m.r2_increment,
        "p_value": m.p_value,
        "n_train": m.n_train,
    }


def _model_from_dict(d: Dict[str, Any]) -> PopularityModel:
    """从缓存 dict 重建模型对象（**不重跑 fit**）。"""
    m = PopularityModel()
    m.coef_ = list(d["coef"])
    m.intercept_ = float(d["intercept"])
    m.tier_lo = float(d["tier_lo"])
    m.tier_hi = float(d["tier_hi"])
    m.r2_increment = float(d.get("r2_increment") or 0.0)
    m.p_value = float(d.get("p_value") or 1.0)
    m.n_train = int(d.get("n_train") or 0)
    m.fitted = True
    return m


def _load_cached(rows: Sequence[Dict]) -> Optional[PopularityModel]:
    """读盘；指纹匹配才返回模型，否则 `None`。任何异常都当作未命中。"""
    try:
        path = _cache_path()
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != _fingerprint_of(rows):
            return None                       # 数据/口径变了 ⇒ 必须重算
        return _model_from_dict(payload["model"])
    except Exception:        # noqa: BLE001 —— 缓存坏了只当没有，绝不影响主链路
        return None


def _save_cached(m: PopularityModel, rows: Sequence[Dict]) -> None:
    """写盘；失败静默（写不了就用内存结果，不影响功能）。"""
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": _fingerprint_of(rows),
            "model": _model_to_dict(m),
            "fitted_at": _now_iso(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)                     # 原子替换，避免读到半个文件
    except Exception:        # noqa: BLE001
        pass


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def _training_rows(draws: Sequence[Dict]) -> List[Dict]:
    """`fit()` 实际会采用的行（与 `fit` 内的过滤条件逐字一致）。

    指纹必须基于这个集合而不是全库 —— 否则「新增一期但缺 p1」会造成无谓的
    重拟合，而「新增一期且含 p1」又必须触发重拟合。两者靠它区分。
    """
    return [d for d in draws
            if d.get("p1") is not None and d.get("sales")
            and d.get("reds") and len(d["reds"]) == 6]


def _load_cached_only() -> Optional[PopularityModel]:
    """只读落盘缓存（不拟合）。读盘约 0.15s，请求路径可接受。"""
    try:
        from . import db
        rows = _training_rows(db.load_draws())
        return _load_cached(rows)
    except Exception:        # noqa: BLE001
        return None


def trigger_background_fit() -> None:
    """触发一次**后台**拟合；已有拟合在跑则什么都不做（不阻塞调用方）。

    ## 为什么请求路径只能「触发」不能「完成」（2026-10-21 实测）

    旧版请求走 `get_model_with_deadline(timeout=2.0)`，而那个 `timeout` **只约束等锁**：

        acquired = _CACHE_LOCK.acquire(timeout=2.0)   # 只限"等锁"
        if acquired:
            return _get_model_locked(...)             # ← 拿到锁后全额 fit，无超时

    于是「预热线程还没取到锁」的那个瞬间（正是每次重启后第一个用户的时刻），
    请求会自己 fit 6~9 秒，然后返回**已拟合的真模型** —— 端点因此走不到 503 分支，
    用户等了 9 秒拿到 200，**以为一切正常**。这是最坏的一种：慢且不自知。

    正确结构是**让用户路径与拟合路径彻底解耦**：
    请求只负责「触发」，不负责「完成」。用户立刻拿到「预热中」，后台继续算。

    ## 为什么必须防重复触发

    并发 20 个请求时，若每个都 spawn 一个拟合线程，就是 20 次 fit（CPU 打爆）。
    用 `_FIT_THREAD_ACTIVE` 保证同时只有一个后台拟合线程。
    """
    with _FIT_TRIGGER_LOCK:
        if _FIT_THREAD_ACTIVE.is_set():
            return
        _FIT_THREAD_ACTIVE.set()
        _WARMING.set()          # 立即置位，让紧随其后的 is_warming() 判定正确
        _READY.clear()

    def _runner():
        try:
            get_model()
        except Exception:        # noqa: BLE001 —— 后台拟合失败绝不影响主链路
            pass
        finally:
            _WARMING.clear()
            _FIT_THREAD_ACTIVE.clear()

    threading.Thread(target=_runner, daemon=True, name="popularity-fit").start()


def get_model_for_request(timeout: float = 2.0) -> PopularityModel:
    """**请求路径专用**：永不自 fit，最多等 `timeout` 秒看别人的成果。

    顺序：
      1. 内存已就绪 → 立刻返回（热态 0.01s 级）；
      2. 落盘缓存命中 → 直接加载（约 0.15s），**跳过 fit**；
      3. 都没有 → **触发后台拟合**，最多等 `timeout` 秒；
      4. 仍没好 → 返回**未拟合对象**（且不写 `_CACHE`），端点据此走 503。

    ## 关键性质（有测试守卫）

    * **绝不自己 fit** —— 这是「9.29s 全额阻塞」缺陷的根因修复；
    * **超时返回的未拟合对象不写 `_CACHE`** —— 否则后续请求会一直看到
      「有对象但未 fitted」，每次都走完整路径；
    * 用户最多等 `timeout`，**宁可显示「预热中」也不为后台任务排队**。
    """
    global _CACHE          # 下面要赋值（缓存命中时回填），必须声明

    # 1. 内存命中
    if _CACHE is not None and _CACHE.fitted:
        return _CACHE

    # 2. 落盘缓存（不拟合）
    cached = _load_cached_only()
    if cached is not None:
        _CACHE = cached
        return cached

    # 3. **不触发 fit**，只等后台（预热/重试循环）的成果
    #
    # 为什么不在这里 `trigger_background_fit()`：结构判据
    # `no_fit_on_request_path.py` 用 AST 传递闭包检查「请求入口是否可达 fit」，
    # 而 `trigger_background_fit → get_model → _get_model_locked → fit` 是一条
    # **静态可达**链。即使运行时它在另一个线程里、不会阻塞请求，结构上仍算违规。
    #
    # 这个判据比行为判据（`elapsed < 3s`）可靠得多 —— 后者会被负载、超时阈值、
    # 缓存状态影响，三次迭代里骗过我们三次。所以这里选择**从结构上断开**：
    # 拟合只由启动时注册的后台循环（`api_app._popularity_warmup`，带重试）负责。
    #
    # 代价：若后台循环从未启动，功能就不可用（返回 503）。因此 `api_app` 必须在
    # import 时启动该循环，且失败要重试 —— 见 `_popularity_warmup` 的重试逻辑。
    _READY.wait(timeout=max(0.0, float(timeout)))

    if _CACHE is not None and _CACHE.fitted:
        return _CACHE
    # 4. 未就绪：返回空模型，**不污染 `_CACHE`**
    return PopularityModel()


def get_model_with_deadline(timeout: float = 2.0,
                            force_refit: bool = False) -> PopularityModel:
    """**已废弃别名** —— 保留名字只为兼容，行为等同 `get_model_for_request()`。

    ## 为什么保留而不是删除（2026-10-21）

    这个函数是 D-16 的**中间产物**，它的旧实现只给 `acquire()` 加了超时，
    拿到锁后仍然调用 `_get_model_locked()` **全额 fit**：

        acquired = _CACHE_LOCK.acquire(timeout=2.0)   # 只限"等锁"
        if acquired:
            return _get_model_locked(...)             # ← 拿到锁就自 fit，无超时

    后果比原缺陷更糟：自 fit 完成后 `_CACHE` 是**已拟合真模型**，端点因此
    走不到 503 分支 —— 用户等 9 秒拿到 `200`，**以为一切正常**。

    `docs/delivery/repro/no_fit_on_request_path.py`（AST 结构判据）把它精确
    标为唯一违规链：`get_model_with_deadline → _get_model_locked → fit`。

    ## 现在它做什么

    **什么都不做 extra** —— 直接委托 `get_model_for_request()`，即：
    内存命中 → 返回；落盘缓存命中 → 加载；都没有 → **触发后台拟合后立即返回**
    未拟合对象（**绝不在这个调用栈里 fit**）。

    保留函数名而不是删掉，是为了不打断任何既有调用方；同时也让结构判据
    （它的 `REQUEST_ENTRIES` 里写着这个名字）继续有效而非"因符号消失而跳过"。
    **改判据让它通过是作弊，改实现让它真的合规才对。**
    """
    return get_model_for_request(timeout=timeout)


def _get_model_locked(force_refit: bool = False) -> PopularityModel:
    """**调用方必须已持有 `_CACHE_LOCK`**：真拟合 + 落盘。

    ## 这个函数是「唯一允许 fit 的地方」（结构判据 `no_fit_on_request_path.py`）

    请求路径（`get_model_for_request` / `get_model_with_deadline`）**永不**调用它 ——
    那样会让用户在请求里等 6~9 秒（D-16 根因）。请求路径只负责
    `trigger_background_fit()` 触发，由后台线程走到这里完成拟合。

    只有两个合法调用点：
      * `get_model()` —— 预热线程用，本来就该 fit；
      * 后台拟合线程（经 `get_model()` 间接进来）。

    抽成独立函数是为了让「谁可以 fit」在结构上只有一个入口，
    AST 判据才能精确指出违规链（`请求入口 → _get_model_locked → fit`）。
    """
    global _CACHE
    from . import db
    draws = db.load_draws()
    rows = _training_rows(draws)

    # ---- 跨进程缓存：指纹匹配则跳过 fit（省 6~7s） ----
    if not force_refit:
        cached = _load_cached(rows)
        if cached is not None:
            _CACHE = cached
            _READY.set()
            return _CACHE

    m = PopularityModel()
    # 标记「正在拟合」—— 让并发请求与端点能区分「等几秒」与「数据不足」。
    # 必须用 try/finally 清除，否则一旦 fit 抛异常，标记会永远挂着，
    # 端点会一直告诉用户「约 5 秒后就绪」，而实际永远不会就绪。
    _WARMING.set()
    info: Dict[str, Any] = {}
    import time as _time
    global _FIT_STARTED_AT
    _t_fit = _time.time()
    _FIT_STARTED_AT = _t_fit
    try:
        info = m.fit(draws) or {}
    except Exception as exc:        # noqa: BLE001 —— 模型失败绝不影响主链路
        # 不再只 `pass`：异常被静默吞掉等于「永远不会报警的故障」。
        # 至少留下原因，供 describe() 回传 `unfitted_reason="fit_failed"`。
        _set_unfit_reason("fit_failed", exc)
    finally:
        # 记录本次实测耗时 —— `suggest_retry_after()` 靠它给动态值。
        # 只在真的跑过 fit 时更新（缓存命中路径不会走到这里）。
        global _LAST_FIT_SECONDS
        _LAST_FIT_SECONDS = _time.time() - _t_fit
        _FIT_STARTED_AT = None
        _WARMING.clear()
    if not m.fitted and _LAST_UNFIT_REASON == "unknown":
        # `fit` 正常返回但 ok=False（典型：样本不足）——记录它自己的 reason
        _set_unfit_reason(str(info.get("reason") or "fit_failed"), None)
    _CACHE = m
    if m.fitted:
        clear_retry_exhausted()  # 成功了，之前的「耗尽」结论作废
        _READY.set()             # 通知所有在等的请求：成果好了
        # 写盘失败绝不能影响功能（磁盘满 / 只读挂载 / 权限）。
        # `_save_cached` 内部已有 try/except，这里再兜一层 —— 因为它可能被替换、
        # 或将来有人改了它却漏了异常保护。**模型能用比缓存写成功重要得多。**
        try:
            _save_cached(m, rows)
        except Exception:        # noqa: BLE001
            pass
    return m


def get_model(force_refit: bool = False) -> PopularityModel:
    """取全局模型；未训练/强制刷新时重新拟合并落盘。

    ## 为什么必须加锁（2026-10-21 实测：预热曾因此完全失效）

    旧实现无锁，只有末尾一句 `_CACHE = m`。时序灾难：

        t=0.0  预热线程进来，`_CACHE is None` → 开始 fit（6~7s）
        t=0.3  用户请求进来，`_CACHE` **仍是 None**（预热还没写完）→ 也开始 fit
        结果：两线程各跑一次 fit，用户等的是**自己那次** —— 预热等于白做

    实测：`fit` 被执行 **2 次**，用户线程仍等 **16.78s**。
    多用户并发更糟 —— 每个请求各 fit 一次，CPU 打爆。

    ## 修法：快路径不取锁 + 双重检查 + `fitted` 判定

    * **快路径不取锁** —— 热态必须保持 0.01s 级，不能被锁拖慢；
    * **双重检查** —— 拿到锁后先看是否已被别的线程填好，是则直接返回；
    * **`_CACHE.fitted` 判定** —— 防止「拟合失败留下的半成品」被当成已就绪。

    ## 落盘缓存参与顺序

    内存未命中 → 读盘（指纹匹配则**跳过 fit**）→ 仍未命中才真 fit 并写盘。
    锁管**同一进程内**只 fit 一次；缓存管**跨进程重启**复用结果。两者不重复。
    """
    global _CACHE
    # ---- 快路径：已就绪直接返回，不取锁（热态性能靠这里） ----
    if _CACHE is not None and _CACHE.fitted and not force_refit:
        return _CACHE

    with _CACHE_LOCK:
        # ---- 双重检查：等锁期间可能已被预热线程填好 ----
        if _CACHE is not None and _CACHE.fitted and not force_refit:
            return _CACHE
        return _get_model_locked(force_refit=force_refit)



def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """线性插值分位数（输入必须已升序）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)
