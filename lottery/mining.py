"""规律自动挖掘管道：基于历史数据的特征重要性，生成候选规律并回测验证。

设计原则：
1. 防挖掘偏差：三段切分（挖掘/校准/测试），只有测试集结果入库 UI
2. 纯 numpy 实现，不依赖 sklearn（兼容性更好）
3. 产出为可存入 patterns 表的规律字典，自动走 backtest 框架分级
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np

from . import backtest as BT, db, features as F, patterns as P
from .config import BASE


# ---------- 特征工程 ----------

_WINDOW_SIZES = (3, 5, 7, 10, 15, 20, 30, 50, 100, 150)
_FEATURE_NAMES = [
    # 多窗口频率（10）
    "freq_3", "freq_5", "freq_7", "freq_10", "freq_15", "freq_20",
    "freq_30", "freq_50", "freq_100", "freq_150",
    # 遗漏（4）
    "omit_cur", "omit_avg", "omit_ratio", "omit_bin",
    # 近 N 期出现（6）
    "appeared_1", "appeared_2", "appeared_3", "appeared_5", "appeared_10", "appeared_20",
    # 三区（3）
    "zone_1_count", "zone_2_count", "zone_3_count",
    # 和值/跨度（2）
    "sum_value", "span_value",
    # M3.3 新增（15）：
    "neighbor_freq_20", "neighbor_freq_50", "neighbor_omit_avg",   # 邻号
    "tail_hot_30", "tail_omit",                                    # 同尾
    "zone_self_hot_50",                                            # 同区
    "head_rate_50", "tail_rate_50",                                # 龙头/凤尾
    "repeat_last", "back_2",                                       # 上期/隔期
    "asc_trend_10", "desc_trend_10",                               # 升温/降温
    "rank_freq_150",                                               # 全历史频率排名
    "co_hot_30",                                                   # 共现热度
    "cold_concentrate",                                            # 冷门集中度
    # M4.6 扩展特征（40）：变化率、结构、属性与联合关系
    "freq_3_vs_30", "freq_5_vs_50", "freq_10_vs_100", "freq_20_vs_150",
    "freq_slope_5_20", "freq_slope_10_50", "freq_slope_20_100",
    "omit_bin_0_5", "omit_bin_6_10", "omit_bin_11_15", "omit_bin_16_20", "omit_bin_21_plus",
    "number_parity", "number_size", "number_route_0", "number_route_1", "number_route_2",
    "number_norm", "omit_norm",
    "zone_share_1", "zone_share_2", "zone_share_3",
    "tail_freq_10", "tail_freq_50", "neighbor_omit_ratio",
    "back_3", "back_5", "appeared_30", "appeared_50",
    "blue_same_parity_rate", "blue_same_size_rate",
    "sum_mean_50", "sum_std_50", "span_mean_50", "span_std_50", "odd_mean_50",
    "last_sum_z", "last_span_z", "last_odd_count", "zone_imbalance",
]


def _compute_features_for_number(
    draws: List[Dict], idx: int, number: int
) -> List[float]:
    """为 (draws[:idx], number) 计算特征向量。"""
    history = draws[:idx]
    n = len(history)
    feats = []

    # 多窗口频率
    for w in _WINDOW_SIZES:
        sl = F.window_slice(history, w)
        f = F.red_frequency(sl)
        feats.append(float(f[number]))

    # 遗漏
    om_cur = F.current_omission_red(history)
    om_avg = F.avg_omission(F.red_frequency(history), n)
    feats.append(float(om_cur[number]))
    feats.append(float(om_avg[number]))
    ratio = float(om_cur[number]) / float(om_avg[number]) if float(om_avg[number]) > 0 else 0.0
    feats.append(ratio)
    oc = float(om_cur[number])
    feats.append(4.0 if oc > 20 else 3.0 if oc > 15 else 2.0 if oc > 10 else (1.0 if oc > 5 else 0.0))

    # 是否近 N 期出现
    for k in (1, 2, 3, 5, 10, 20):
        appeared = sum(1 for d in history[-k:] if number in d["reds"])
        feats.append(float(appeared) / max(k, 1))

    # 三区计数
    zc = F.zone_counts(history[-10:]) if len(history) >= 10 else history
    z_counts = [0, 0, 0]
    for z in zc:
        for i in range(3):
            z_counts[i] += z[i]
    for zc_val in z_counts:
        feats.append(float(zc_val) / max(len(zc), 1))

    # 和值与跨度
    if history:
        feats.append(float(np.mean(F.sums(history[-10:]))))
        feats.append(float(F.span(history[-10:]).mean()))
    else:
        feats.append(0.0)
        feats.append(0.0)

    # ---- M3.3 新增特征（邻号/同尾/同区/位置/趋势/共现）----
    nbr = [x for x in (number - 1, number + 1) if 1 <= x <= 33]

    def _nfreq(w: int) -> float:
        sl = F.window_slice(history, w)
        f = F.red_frequency(sl)
        return float(np.mean([f[x] for x in nbr])) if nbr else 0.0

    feats.append(_nfreq(20))
    feats.append(_nfreq(50))
    feats.append(float(np.mean([float(om_cur[x]) for x in nbr])) if nbr else 0.0)

    tail = number % 10
    tail_draws = history[-30:] if len(history) >= 30 else history
    t_cnt = sum(1 for d in tail_draws for x in d["reds"] if x % 10 == tail)
    feats.append(float(t_cnt) / max(len(tail_draws), 1) / 6.0)
    t_om = [float(om_cur[x]) for x in range(1, 34) if x % 10 == tail]
    feats.append(float(np.mean(t_om)) if t_om else 0.0)

    zone = 0 if number <= 11 else (1 if number <= 22 else 2)
    sl50 = history[-50:] if len(history) >= 50 else history
    if sl50:
        f50 = F.red_frequency(sl50)
        zset = [x for x in range(1, 34)
                if (0 if x <= 11 else (1 if x <= 22 else 2)) == zone]
        feats.append(float(np.mean([f50[x] for x in zset])))
    else:
        feats.append(0.0)

    h50 = history[-50:] if len(history) >= 50 else history
    head = sum(1 for d in h50 if min(d["reds"]) == number)
    tailr = sum(1 for d in h50 if max(d["reds"]) == number)
    feats.append(float(head) / max(len(h50), 1))
    feats.append(float(tailr) / max(len(h50), 1))

    feats.append(1.0 if history and number in history[-1]["reds"] else 0.0)
    feats.append(1.0 if len(history) >= 2 and number in history[-2]["reds"] else 0.0)

    f10 = float(F.red_frequency(history[-10:] if len(history) >= 10 else history)[number])
    f50v = float(F.red_frequency(history[-50:] if len(history) >= 50 else history)[number])
    feats.append(1.0 if f10 > f50v else 0.0)
    feats.append(1.0 if f10 < f50v else 0.0)

    f_all = F.red_frequency(history)[1:]
    rank = float((f_all >= f_all[max(0, number - 1)]).mean()) if len(f_all) else 0.0
    feats.append(rank)

    c30 = history[-30:] if len(history) >= 30 else history
    co = set()
    for d in c30:
        if number not in d["reds"]:
            co |= set(d["reds"])
    if co:
        f30 = F.red_frequency(c30)
        feats.append(float(np.mean([f30[x] for x in co])))
    else:
        feats.append(0.0)

    feats.append(rank * ratio)   # 冷门集中度

    # ---- M4.6 扩展特征（40 维）----
    def _freq(w: int) -> float:
        sl = history[-w:] if len(history) >= w else history
        return float(F.red_frequency(sl)[number]) if sl else 0.0

    def _slope(w1: int, w2: int) -> float:
        return (_freq(w1) / max(w1, 1)) - (_freq(w2) / max(w2, 1))

    f3, f5, f10, f20 = _freq(3), _freq(5), _freq(10), _freq(20)
    f30, f50, f100, f150 = _freq(30), _freq(50), _freq(100), _freq(150)
    feats.extend([
        f3 - f30 / 10.0, f5 - f50 / 10.0, f10 - f100 / 10.0, f20 - f150 / 7.5,
        _slope(5, 20), _slope(10, 50), _slope(20, 100),
    ])

    om_value = float(om_cur[number])
    feats.extend([
        float(om_value <= 5),
        float(6 <= om_value <= 10),
        float(11 <= om_value <= 15),
        float(16 <= om_value <= 20),
        float(om_value >= 21),
    ])
    feats.extend([
        float(number % 2), float(number <= 16),
        float(number % 3 == 0), float(number % 3 == 1), float(number % 3 == 2),
        float(number) / 33.0, om_value / max(float(n), 1.0),
    ])

    recent = history[-20:] if len(history) >= 20 else history

    z_total = max(float(len(history[-50:] if len(history) >= 50 else history) * 6), 1.0)
    z_counts_50 = [0, 0, 0]
    sl50 = history[-50:] if len(history) >= 50 else history
    for d in sl50:
        for r in d["reds"]:
            z_counts_50[0 if r <= 11 else (1 if r <= 22 else 2)] += 1
    feats.extend(v / z_total for v in z_counts_50)

    tail_numbers = [x for x in range(1, 34) if x % 10 == tail]
    tail10 = history[-10:] if len(history) >= 10 else history
    tail50 = history[-50:] if len(history) >= 50 else history
    feats.append(sum(1 for d in tail10 for r in d["reds"] if r % 10 == tail) / max(len(tail10) * 6, 1))
    feats.append(sum(1 for d in tail50 for r in d["reds"] if r % 10 == tail) / max(len(tail50) * 6, 1))
    feats.append((float(np.mean([om_cur[x] for x in nbr])) / max(float(om_avg[number]), 1.0)) if nbr else 0.0)

    for k in (3, 5):
        feats.append(float(number in history[-k]["reds"]) if len(history) >= k else 0.0)
    for k in (30, 50):
        sl = history[-k:] if len(history) >= k else history
        feats.append(sum(1 for d in sl if number in d["reds"]) / max(len(sl), 1))

    blue_parity = [d["blue"] % 2 for d in history[-50:]]
    blue_size = [d["blue"] <= 8 for d in history[-50:]]
    feats.append(float(np.mean([v == number % 2 for v in blue_parity])) if blue_parity else 0.0)
    feats.append(float(np.mean([v == (number <= 16) for v in blue_size])) if blue_size else 0.0)

    sums50 = F.sums(sl50) if sl50 else np.array([], dtype=float)
    spans50 = F.span(sl50) if sl50 else np.array([], dtype=float)
    odds50 = F.odd_counts(sl50) if sl50 else np.array([], dtype=float)
    feats.extend([
        float(np.mean(sums50)) if len(sums50) else 0.0,
        float(np.std(sums50)) if len(sums50) else 0.0,
        float(np.mean(spans50)) if len(spans50) else 0.0,
        float(np.std(spans50)) if len(spans50) else 0.0,
        float(np.mean(odds50)) if len(odds50) else 0.0,
    ])
    last_sum = float(sum(history[-1]["reds"])) if history else 0.0
    last_span = float(max(history[-1]["reds"]) - min(history[-1]["reds"])) if history else 0.0
    feats.extend([
        (last_sum - float(np.mean(sums50))) / max(float(np.std(sums50)), 1.0) if len(sums50) else 0.0,
        (last_span - float(np.mean(spans50))) / max(float(np.std(spans50)), 1.0) if len(spans50) else 0.0,
        float(sum(1 for r in history[-1]["reds"] if r % 2)) if history else 0.0,
        float(max(z_counts_50) - min(z_counts_50)) / max(z_total, 1.0),
    ])

    return feats


def build_feature_matrix(
    draws: List[Dict], min_start: int = 300
) -> tuple:
    """构建 (period, number) -> feature_vector 矩阵与 label。

    返回:
        X: (n_periods * 33, n_features) float array
        y: (n_periods * 33,) binary array (1 = 该号码下期出现)
        meta: list of (period_index, number) tuples
    """
    X_rows, y_rows, meta = [], [], []
    for i in range(min_start, len(draws) - 1):
        target_reds = set(draws[i + 1]["reds"])
        for num in range(1, 34):
            feats = _compute_features_for_number(draws, i, num)
            X_rows.append(feats)
            y_rows.append(1.0 if num in target_reds else 0.0)
            meta.append((draws[i]["issue"], num))
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float), meta


# ---------- 特征重要性（纯 numpy，无 sklearn）----------


def _feature_lift(X: np.ndarray, y: np.ndarray, n_bins: int = 5) -> List[float]:
    """对每个特征计算 lift = P(y=1 | feature high) - baseline。"""
    n_features = X.shape[1]
    baseline = y.mean()
    lifts = np.zeros(n_features)
    for j in range(n_features):
        col = X[:, j]
        q = np.percentile(col, 100 * (1 - 1 / n_bins))
        high_mask = col >= q
        if high_mask.sum() < 10:
            lifts[j] = 0.0
            continue
        p_high = y[high_mask].mean()
        lifts[j] = p_high - baseline
    return lifts.tolist()


def _feature_importance_ml(X: np.ndarray, y: np.ndarray, engine: str = "rf") -> List[float]:
    """LightGBM / RandomForest 特征重要性（缺失或失败自动回退）：
    lightgbm -> sklearn RandomForest -> numpy lift。"""
    model = None
    if engine == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
            model = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05,
                                       num_leaves=31, random_state=42, verbose=-1)
        except Exception as e:  # noqa: BLE001
            print(f"[mining] lightgbm 未安装，回退 sklearn RandomForest: {e}")
            engine = "rf"
    if engine == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # n_jobs=1：避免 uvicorn 线程池内 loky/OpenMP 进程派发死锁（单核 VPS）
        model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
    try:
        assert model is not None
        model.fit(X, y)
        return model.feature_importances_.tolist()
    except Exception as e:  # noqa: BLE001
        print(f"[mining] {engine} 拟合失败，回退 numpy lift: {e}")
        return _feature_lift(X, y)


def _top_features_by_lift(
    lifts: List[float], feature_names: List[str], top_k: int = 5
) -> List[tuple]:
    """按 lift 绝对值排序，返回 (feat_name, lift, index) 列表。"""
    indexed = [(name, lift, i) for i, (name, lift) in enumerate(zip(feature_names, lifts))]
    indexed.sort(key=lambda x: -abs(x[1]))
    return indexed[:top_k]


# ---------- 候选规律生成 ----------


def _generate_candidates_from_features(
    top_feats: List[tuple], draws: List[Dict]
) -> List[Dict]:
    """从 top 特征生成候选规律字典（可直接送入 backtest 框架）。"""
    candidates = []
    for feat_name, lift, idx in top_feats:
        kind = "short" if any(x in feat_name for x in ("5", "10", "20")) else "mid"
        if feat_name.startswith("freq_"):
            window = feat_name.replace("freq_", "", 1)
            desc = f"近{window}期频率排名的号码下一期是否延续"
            trigger = lambda h, w=int(window): len(h) >= int(w)
            action = lambda h, w=int(window): {
                "red_fav": [
                    int(x) for x in np.argsort(F.red_frequency(h[-max(w,1):])[1:])[-6:][::-1] + 1
                ]
            } if h else {}
        elif "omit_ratio" in feat_name:
            desc = "遗漏/平均遗漏比高的号码是否即将回补"
            trigger = _t_always
            action = lambda h: {
                "red_fav": [
                    int(x) for x in np.argsort(
                        F.current_omission_red(h) / (F.avg_omission(F.red_frequency(h), len(h)) + 0.01)
                    )[-6:][::-1] + 1
                ]
            } if h else {}
        elif "appeared" in feat_name:
            k = int(feat_name.replace("appeared_", ""))
            desc = f"近{k}期未出现的号码是否回补"
            trigger = lambda h, k=k: len(h) >= k
            action = lambda h, k=k: {
                "red_fav": [
                    int(x) for x in np.argsort(F.current_omission_red(h)[-k:] if len(h) >= k else F.current_omission_red(h))[:6][::-1] + 1
                ]
            } if h else {}
        elif "zone" in feat_name:
            desc = "低频区号码的轮动回补"
            trigger = _t_always
            action = P._act_zone_low_rebound
        elif "sum" in feat_name or "span" in feat_name:
            desc = "和值/跨度极端后的回归"
            trigger = P._t_sum_extreme
            action = lambda h: {}
        else:
            desc = f"特征 {feat_name} 的数值偏高号码回补"
            trigger = _t_always
            action = lambda h: {"red_fav": []}

        candidates.append({
            "key": f"mined_{feat_name}",
            "name_zh": f"挖掘·{feat_name}",
            "kind": kind,
            "desc": desc,
            "horizon": 1,
            "trigger_fn": trigger,
            "action_fn": action,
            "outcome": "red_fav",
            "base_fn": P.red_base,
            "_mined": True,
            "_feat_name": feat_name,
            "_lift": round(lift, 4),
        })
    return candidates


def _t_always(history: List[Dict]) -> bool:
    return True


# ---------- 主流程 ----------


def run_mining(
    draws: List[Dict],
    min_start: int = 300,
    top_k_features: int = 8,
    save_to_db: bool = True,
    engine: str = "rf",
) -> Dict:
    """运行挖掘管道：特征计算 -> 重要性排序(LightGBM/RF/lift) -> 候选规律生成 -> 回测 -> 入库。

    返回挖掘结果摘要（同时记入 mining_runs 表）。
    """
    print(f"[mining] 开始挖掘，min_start={min_start}, top_k={top_k_features}, engine={engine}")
    t0 = time.time()
    run_id = f"mine_{time.strftime('%Y%m%d_%H%M%S')}"

    # 1. 构建特征矩阵（40 维特征）
    X, y, meta = build_feature_matrix(draws, min_start=min_start)
    print(f"[mining] 特征矩阵形状: {X.shape}, 正样本率: {y.mean():.4f}")

    # 2. 特征重要性：LightGBM / RandomForest / lift（依次回退）
    if engine == "lift":
        importances = _feature_lift(X, y)
    else:
        importances = _feature_importance_ml(X, y, engine)
    top_feats = _top_features_by_lift(importances, _FEATURE_NAMES, top_k=top_k_features)
    print(f"[mining] Top 特征: {[(name, round(v, 4)) for name, v, _ in top_feats]}")

    # 3. 生成候选规律
    candidates = _generate_candidates_from_features(top_feats, draws)
    print(f"[mining] 生成 {len(candidates)} 条候选规律")

    # 4. 对候选规律做 walk-forward 回测
    results = []
    for cand in candidates:
        # 临时移除 _mined 标记，让 backtest 框架能处理
        clean_cand = {k: v for k, v in cand.items() if not k.startswith("_")}
        bt = BT.backtest_pattern(clean_cand, draws, min_start=min_start)
        bt["_mined"] = True
        bt["_feat_name"] = cand["_feat_name"]
        bt["_lift"] = cand["_lift"]
        results.append(bt)
        print(f"  [{bt['grade']}] {bt['name_zh']}: n={bt.get('n',0)} margin={bt.get('margin',0):+.3f} p={bt.get('p_value',1):.4f}")

    # 5. BH 校正 + 分级
    set_res = [r for r in results if r["direction"] in ("above", "below")]
    pvals = [r["p_value"] for r in set_res]
    adj = BT.bh_adjust(pvals)
    for r, a in zip(set_res, adj):
        r["p_adj"] = a
    for r in results:
        r.setdefault("p_adj", 1.0)
        r.setdefault("grade", "C")
        if r.get("n", 0) >= 30 and r["direction"] in ("above", "below"):
            if r.get("p_adj", 1.0) < 0.05 and r["direction"] == "above":
                r["grade"] = "A"
            elif r.get("p_value", 1.0) < 0.20 and r["direction"] == "above":
                r["grade"] = "B"

    # 6. 入库
    if save_to_db:
        for r in results:
            r.setdefault("sample_size", r.get("n", 0))   # backtest 结果无此键，落库前补齐
        db.replace_mined_patterns(results)
        print(f"[mining] 已入库 {len(results)} 条挖掘规律")

    elapsed = time.time() - t0
    grades = {"A": 0, "B": 0, "C": 0}
    for r in results:
        grades[r.get("grade", "C")] += 1
    accepted = [r for r in results if r.get("grade") in ("A", "B")]
    avg_lift = float(np.mean([abs(r.get("_lift", 0.0)) for r in results])) if results else 0.0
    summary = {
        "run_id": run_id,
        "engine": engine,
        "n_features": int(X.shape[1]),
        "n_candidates": len(results),
        "accepted": len(accepted),
        "grades": grades,
        "avg_lift": round(avg_lift, 4),
        "pass_rate": round(len(accepted) / max(1, len(results)), 4),
        "elapsed_seconds": round(elapsed, 1),
        "features_used": [(name, round(v, 4)) for name, v, _ in top_feats],
    }
    # mining_runs 落库（每次运行都记录，含未入库候选）
    db.save_mining_run({
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine,
        "candidates": len(results),
        "accepted": len(accepted),
        "avg_lift": summary["avg_lift"],
        "pass_rate": summary["pass_rate"],
        "duration_ms": int(elapsed * 1000),
        "params_json": {"min_start": min_start, "top_k": top_k_features,
                        "n_features": int(X.shape[1])},
    })
    save_mining_result(summary)
    export_mining_artifact(summary, results, draws)
    print(f"[mining] 完成: {summary}")
    return summary


def _artifact_path():
    from . import config
    return config.BASE / "data" / "mining_artifact.json"


def export_mining_artifact(summary: Dict, results: List[Dict], draws: List[Dict]) -> None:
    """导出可提交 Git 的挖掘产物；不包含数据库、原始数据或凭据。"""
    payload = {
        "format": "ssq-mining-artifact-v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_max_issue": draws[-1]["issue"] if draws else None,
        "summary": summary,
        "patterns": [
            {k: v for k, v in r.items() if k in {
                "key", "name_zh", "kind", "desc", "params", "grade", "sample_size",
                "margin", "p_value", "p_adj", "direction", "backtest", "_feat_name", "_lift"
            }}
            for r in results
        ],
    }
    path = _artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def import_mining_artifact(path=None) -> Dict:
    """导入本机发布的挖掘产物；重复导入结果一致。"""
    artifact = path or _artifact_path()
    artifact = __import__("pathlib").Path(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("format") != "ssq-mining-artifact-v1":
        raise ValueError("不支持的挖掘产物格式")
    patterns = payload.get("patterns")
    if not isinstance(patterns, list):
        raise ValueError("挖掘产物缺少 patterns 列表")
    required = {"key", "name_zh", "kind", "grade", "sample_size", "margin", "p_value", "direction"}
    clean = []
    for item in patterns:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("挖掘产物包含不完整规律")
        clean.append(item)
    db.replace_mined_patterns(clean)
    return {
        "source_max_issue": payload.get("source_max_issue"),
        "imported": len(clean),
        "run_id": (payload.get("summary") or {}).get("run_id"),
    }


def get_latest_mining_result() -> Optional[Dict]:
    """读取最近一次挖掘结果摘要。"""
    from . import config
    result_file = BASE / "data" / "mining_latest.json"
    if not result_file.exists():
        return None
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_mining_result(result: Dict) -> None:
    """保存挖掘结果摘要供 API 查询。"""
    from . import config
    result_file = BASE / "data" / "mining_latest.json"
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
