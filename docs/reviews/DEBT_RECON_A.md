# DEBT_RECON_A — walk-forward 重复实现 × 错误返回形态 · 只读侦察报告

- 侦察对象：`/root/DSH/应用/ssqyuce`（APP_VERSION 4.19.0，git `3785736` fix(v39)）
- 日期：2026-10-25；方法：AST 全量扫描（`ast` 模块解析 `lottery/` 全部 return 语句）+ 逐文件精读 + 数值对拍（在 `.venv` Python 3.13.5 实跑）+ 调用点 grep
- 纪律：只读。除本报告外未修改任何文件，未跑 pytest。

---

## 0. TL;DR（先给结论）

1. **任务书把审计原文转述错了**。审计 P2-1 的「三/五份重复统计实现」指的是 **三份 BH 校正 + 五份 Wilson 区间**（`AUDIT_2026-10-24_ARCH.md:152`），**不是 walk-forward**。walk-forward 实现确实有 8 个循环体，但它们**绝大多数不是同一算法的副本**，且项目已有明确的"一份实现"纪律并落实了（见 §2.3）。
2. **真正可合并的重复是 BH×3 与 Wilson×4**：数值逐位一致（实拍 max diff = 0.0 与 1e-16 级），薄壳合并零测试变红。**建议合并**。
3. **walk-forward 八个实现不建议合并**：评估对象（票 / 概率向量 / 规律触发 / 维度选择器 / ML 重训 / LLM 调用 / 通用策略函数）与统计量（Brier / 命中 z / 配对 t / Wilcoxon）各不相同，"都写 `hist = draws[:t]`"不构成重复实现。
4. **错误返回形态：审计"四形态并存"部分成立、数字有误报成分**。真实 HTTP 错误形态高度统一（117 处路由级错误响应 100% 带 `ok`+`error`）；真问题只有两个小项：**12 个端点 HTTP 200 + `ok:false`**、**≥6 处 dict 内 `detail` 键重复**。审计把**内部函数的 `{"ok": False}` 返回值（41 处）**与 HTTP 形态混在一起计数，属口径污染（本侦察用 AST 区分了两者）。
5. `HTTPException` 在 `lottery/` 下 **0 处**使用（全 grep 确认）。

---

## 第一部分：walk-forward / 滚动回测实现清单

判定标准：函数体内存在 `for t in ...: hist = draws[:t]` 形态的"逐期前推"循环。共 **8 个独立循环体** + 2 个滚动窗口变体。以下按"评估对象"分组。

### 1.1 方法级（出号方法 → 逐期出票 → 对奖）

#### W1 `battle.replay_method` —— 方法级唯一权威实现

- **签名**（`lottery/battle.py:231`）：
  `replay_method(method: str, draws: List[Dict], n_tickets: int = 10, start: int = 300, seed: int = 42, objective: str = "expected_value") -> Dict`
- **算法**：对 t ∈ [start, len(draws))：`hist = draws[:t]`（battle.py:311），`ctx = _ctx_for_t(t, ...)`（约束上下文逐期重建，**不做窗口缓存**——缓存会改变出号结果，battle.py 模块头有实测论证），`rng = random.Random(seed + t)` 按期确定种子，`_tickets_for_method(m, hist, n_t, rng, ctx)` 出票，`EV.tickets_result` 对 `draws[t]` 打分。
- **返回**（battle.py:267-276）：`method, replayable, reason, issues, n_tickets_total, red_hits_mean, blue_hit_rate, prize_rate_ge5, reward_total, cost_total, roi, reward_total_ex_top, roi_ex_top, n_top_prize_issues, per_issue[{issue, red_hits, blue_hit, prize_level, reward, n_tickets}], n_tickets_per_issue, start, seed, objective, note_zh`。
- **调用点**：
  - `lottery/battle.py:391`（`replay_all`）
  - `lottery/walk_forward_gate.py:126`、`:195`（多种子循环）
  - `lottery/evolution.py:726`、`:730`（假说 A/B 基线）
  - `tests/test_v4_battle.py` 30+ 处直接调用（42 个测试）
- **测试覆盖**：`test_v4_battle.py`（42 个：防未来泄漏双向测试、同种子可复现、逐方法字段完整性、边界空数据等）。

#### W2 `walk_forward_gate.walk_forward_check` —— **不是重复实现，是 W1 的判据包装**

- **签名**（`lottery/walk_forward_gate.py:103`）：
  `walk_forward_check(draws: Sequence[Dict], method: str = "stat:freq", *, n_tickets: int = 5, start: int = 300, seed: int = 42, color: str = "blue") -> Dict`
- **算法**：调 `battle.replay_method`（:126）拿 `per_issue`，把命中数换算到**随机基线**上算 z：蓝球 = 二项 z（基线 1/16，命中数按注累计，绝不用被取整的 `per_issue.blue_hit`——V20 踩过的口径坑，:160-175 注释）；红球 = 逐期平均命中的 t 统计量（基线 6×6/33）。再对 `WF_SEEDS=[42,7,99]` 各跑一遍（:195），取 **z 中位数** 过 `WF_Z_MIN=1.0` 门槛。
- **返回**（实跑确认）：`ok, method, color, replayable, n, hits, hit_rate, baseline, z, z_min, passed, reason_zh, z_by_seed[3], z_median, n_negative, robust`。
- **调用点**：`lottery/routes_rigor.py:274`（端点 `/api/rigor/walk_forward`）、`lottery/ml_model.py:1120`（`diagnose_ml_channels` 顶层双门槛）、`lottery/ai_hypothesis.py:283`（假设证伪第三关）；`tests/test_v20_walk_forward_gate.py`（11 个）、`tests/test_v23_walk_forward_ui.py`（6 个）、`tests/test_v10_ml_calibration.py:819`（monkeypatch 打桩）。
- **模块 docstring 自述**（walk_forward_gate.py:47-50）："只**读** `battle.replay` 的结果，**不重复实现** walk-forward（那套 as-of 纪律已经过 3 年迭代与大量测试，重写只会引入新 bug）"。

#### W3 `replay.offline_replay / replay_rows` —— W1 的落库版，**刻意复用而非重写**

- **签名**（`lottery/replay.py:403`）：
  `offline_replay(draws=None, methods=None, channels=None, mode="replay", n_tickets=5, start=300, seed=7, window=500, persist=True) -> Dict`（内层 `replay_rows` 在 :296）
- **算法**：对 t ∈ [start, n)：`history = draws[:t]`（replay.py:268），热度阈值/约束上下文用 **window=500 滚动窗口**（`as_of_hot_reds` / `_ctx_as_of`，这是与 W1 的真实差异点），出票直接调 **`battle._tickets_for_method`**（replay.py:238，注释自述"方法级出号只有**一份**实现，避免两份漂移"），逐期产出落 `replay_results` 表的行（含 `channels_fp` 指纹），另带 as-of 诊断字段。
- **返回**：`mode, channels, channels_fp, methods, skipped, start, window, n_tickets_per_issue, seed, n_issues, n_rows, n_written, per_method{…含 MDE 与显著性}, rows[], as_of_zh, sample_note_zh`。
- **调用点**：生产代码**无直接调用者**（grep 全仓仅 `lottery/replay.py` 自身与 `tests/test_v10_replay.py` 21 个测试）。对外暴露的是 `replay.samples_side_by_side`（被 `routes_llm_ops.py:163`、`evaluate.py:294` 调）——它读的是**已落库**的 `replay_results` 行。**生产调用链未确认**（可能只靠测试/手动触发写入）。

### 1.2 概率向量级（模型分布 → Brier）

#### W4 `engine_blend._walk_forward_brier` —— 融合定权专用

- **签名**（`lottery/engine_blend.py:178`）：
  `_walk_forward_brier(pool: Dict[str, Dict], draws: List[Dict], n_eval: int) -> Dict[str, Dict[str, float]]`
- **算法**：对 t ∈ [n−n_eval, n)：`hist = draws[:t]`（:218），对池中每个方法用 `_rebuild_probs(name, hist)` **只用历史重建概率分布**（拒绝用调用方传入的、在全量历史上拟合好的向量——那会乐观，:185-190 注释），与第 t 期 one-hot 真值算 Brier，逐方法累计。
- **返回**：`{method: {"red": float, "blue": float, "n": int}}`（仅成功评估的方法）。
- **调用点**：`lottery/engine_blend.py:299`（`_brier_blend` 内，softmax 定权 + 权重上限 + 收缩）；`lottery/engine.py:98` 再导出。
- **测试**：`tests/test_v4_blend.py` 4 个直接测试（`test_all_reconstructible_methods_report_walk_forward:240`、`test_walk_forward_never_uses_future_data:258`——改未来→Brier 不变、`test_walk_forward_brier_is_above_in_sample_brier:287` 等）。

### 1.3 规律级（单条 pattern → 触发 → 显著性分级）

#### W5 `backtest.backtest_pattern`

- **签名**（`lottery/backtest.py:145`）：
  `backtest_pattern(pattern: Dict, draws: List[Dict], min_start: int = 120) -> Dict`
- **算法**：对 i ∈ [min_start, n)：`history = draws[:i]`（:150），`P.run_pattern(pattern, history)` 判断触发；红球命中 = 与当期红球交集，期望 = 6k/33（含 horizon 多期外推）；数值型 outcome 走均值偏差分支。聚合后单样本 t 检验 + Wilson CI + 单侧 p（:255-263）。
- **返回**：`{key, name_zh, kind, desc, horizon, n, backtest:{n, avg_hits, expected, margin, hit_rate_at_least1, p0, avg_fav_size, t_stat, ci_lower, ci_upper, series[:200]}, margin, p_value, p_adj, direction, grade}`。
- **调用点**：`lottery/backtest.py:273`（`run_all_backtests`，BH 校正 + A/B/C 分级）、`lottery/mining.py:431`（挖掘候选规律复评，:441 再调 `BT.bh_adjust`）。
- **测试**：**无直接测试**。`tests/test_v7_rules_closure.py`、`tests/test_v8_eras_tasks.py` 只 import BT 测 `next_issue`/`expected_issue`/`DRAW_WEEKDAYS`；`backtest_pattern`/`run_all_backtests` 全 tests/ 零命中。**规律级回测目前处于测试盲区**（与审计 QA「22 个模块无测试触达」一致）。

### 1.4 引擎级（整套生产引擎 vs 均匀随机基线）

#### W6 `evaluate.offline_backtest`

- **签名**（`lottery/evaluate.py:83`）：
  `offline_backtest(draws: List[Dict], issues: int = 120, n_tickets: int = 10, use_llm: bool = False, seed: int = 7) -> Dict`
- **算法**：对 i ∈ [start, n−1)（start = max(n−issues−1, 300)）：`history = draws[:i]`，每期派生独立子种子 `random.Random(seed*1_000_003 + i)`（V36 P0-4 修的非确定性，:108）调 `E.predict_next(history, use_ml=False, rng=…)` 出系统票；同 ctx 下采 n 注均匀票作随机基线；`tickets_result` 打分后 `aggregate` 汇总。
- **返回**：`{n_issues, n_tickets_per_issue, use_llm, system:{…}, random_baseline:{…}, delta_red_hits, delta_blue_rate, note}`。
- **调用点**：`lottery/routes_eval.py:26`（端点 `POST /api/eval/backtest`）、`lottery/cli.py:65`。
- **测试**：`tests/test_v36_p0_audit_fixes.py:194`（`test_offline_backtest_passes_rng_into_engine`，同数据同 seed 两次结果一致）。

### 1.5 ML 级（重训循环，本质不同）

#### W7 `ml_model.evaluate_ml_walkforward`

- **签名**（`lottery/ml_model.py:1608`）：
  `evaluate_ml_walkforward(draws, min_start=None, window=None, refit_every=None, train_window=800, seed=7) -> Dict`
- **算法**：对 i ∈ [max(min_start, n−window), n)：`history = draws[:i]`；每 `refit_every` 期用 `history[-train_window:]` **重训**红/蓝模型（防未来信息），逐期算 ML vs 均匀基线的 Brier / log-loss + pooled 校准数据；配对 Wilcoxon / 符号检验。
- **返回**：`{ok, n_issues, window, refit_every, train_window, refits, seconds, red:{brier_ml, brier_uniform, logloss_ml, logloss_uniform, paired{p,method,mean_delta}, calibration[bins]}, blue:{…}}`。
- **调用点**：`lottery/routes_ml.py:36`（端点 `POST /api/ml/eval`）。
- **测试**：`tests/test_v10_ml_calibration.py`（23 个）。
- **本质差异**：唯一带**训练循环**的 walk-forward（每期重训成本远超回放预算，这也是 battle 把 `ml` 列为不可回放的原因，battle.py:55-58）。

### 1.6 LLM 级（三通道含外部调用，本质不同）

#### W8 `llm_eval.run_llm_eval`

- **签名**（`lottery/llm_eval.py:91`）：
  `run_llm_eval(draws, issues=None, n_tickets=None, seed=None, progress_cb=None) -> Dict`
- **算法**：三通道（stat / stat_llm / random）对 i ∈ [start, n−1)：`history = draws[:i]`（:125），stat 通道走 `E.predict_next(use_llm=False)`，stat_llm 通道带看门狗超时调 LLM（每期仅 1 次采样降本），random 通道均匀票；V36 P1-14 修复：LLM 失败期整对剔除，不污染配对检验。
- **返回**：run_id + 逐通道逐指标序列 + paired p + 用量（tokens/cost）；落 `llm_eval_results` 表。
- **调用点**：`lottery/routes_llm_eval.py:45`。
- **测试**：`test_llm_*`、`tests/test_v10_llm_arm.py`（静态守卫 T1：对照臂不得走 replay）。

### 1.7 通用策略函数级（生产无调用者）

#### W9 `stats.rigor.evaluate_signal`

- **签名**（`lottery/stats/rigor.py:1206`）：
  `evaluate_signal(draws, strategy_fn, window, n_trials=1, alpha=0.05, **kw) -> Dict`
- **算法**：对 t ∈ [window, n)：`history = valid[:t]`，`strategy_fn(history, **kw)` 返回红球集合，命中数 vs 逐期超几何期望 6k_t/33（k 随策略当期选号数浮动），序列送 `strategy_significance` 做 t 检验 + 符号检验 + Bonferroni（n_trials 校正）。
- **返回**（实跑确认 33 个 key）：`n_predictions, n, window, avg_k, mean, baseline, delta, relative_lift, t, p_value, p_uncorrected, bonferroni_p, bh_p, adjusted_p, survives_correction, p_two_sided, adjusted_p_two_sided, survives_correction_two_sided, verdict_two_sided, sign_test, verdict, hit_series[:200], errors, method, message_zh, …`。
- **调用点**：**生产代码零调用**（全仓 grep 仅 `stats/rigor.py` 自身与 `tests/test_v4_rigor.py` 11 个、`tests/test_v10_calibration.py` 3 个）。`/api/rigor/signal`（routes_rigor.py:186）**没有**走它——而是用 `_strategy_hit_series`（:126，见 1.8）自己产序列、直接调 `strategy_significance`（:203）。
- **这是"审计说重复但其实不重复/不接线"的典型**：它与 1.8 是同构度最高的对（都是滚动命中序列 + strategy_significance），但一个在生产端点外、一个在端点内，且基线口径不同（见 §2.2）。

### 1.8 端点内联的滚动窗口变体

#### W10 `routes_rigor._strategy_hit_series`

- **签名**（`lottery/routes_rigor.py:126`）：
  `_strategy_hit_series(draws: List[Dict], strategy: str, window: int) -> Optional[List[float]]`
- **算法**：对 i ∈ [window, n)：`hist = draws[i-window:i]`——**滚动窗口**而非全量 as-of（hot/cold 只看窗口内频次）；hit = 推荐组与当期交集；blue_hot/blue_cold 是 0/1。自述"只做描述性统计，显著性交给 rigor 模块"。
- **返回**：`List[float]`（纯序列，不是 dict）。
- **调用点**：`routes_rigor.py:186` 端点 `/api/rigor/signal` 内唯一调用。
- **测试**：`test_v4_rigor.py` 未见直接覆盖该私有函数（经端点间接触达，**未确认**）。

### 1.9 维度选择器级（discovery，服务多个上游）

#### W11 `discovery.asof_scan`

- **签名**（`lottery/discovery.py:1540`）：
  `asof_scan(draws, flt, start=200, end=None, ctx=None) -> Dict`
- **算法**：对 t ∈ [start, end)：`flt["selector"](ctx)` 在**只含前 t 期**的 `ScanContext`（增量缓存，逐期前进一期而非重算全史，:95-103）上取候选集，obs = 命中 / exp = 6k/33（或蓝球 k/16）。
- **返回**：`{target, n, obs, exp, k_series, k_mean, k_active_mean, hit_series, start, end}`。
- **调用点**：`discovery.regime_of_filter:1636`、`multiscale.py:130`（`_window_series`）、`idle_mining.py:452`（闲时挖掘快速筛选）、`routes_rules.py` 的 discover/核验端点链。
- **测试**：150 组 as-of 一致性测试（discovery.py:203 自述）+ `test_v6_regime.py:273 test_asof_scan_is_strictly_walk_forward`。

---

## 第二部分：判断 —— 谁真重复、谁同构、谁本质不同

### 2.1 审计原文还原（关键证据）

`docs/reviews/AUDIT_2026-10-24_ARCH.md` **P2-1 标题**（:152）：

> **P2-1 三份 BH 校正 + 五份 Wilson 区间并行实现，真源与副本无一致性守卫**

即审计说的「三/五份」= **BH×3、Wilson×5（含 re-export 计为一份）**，与 walk-forward 无关。SUMMARY.md:119 的压缩表述「三/五份重复统计实现」丢掉了主语，任务书进一步误读成「walk-forward 统计实现」。**本侦察判定：任务书层面的"债务 A"命题本身不成立**——walk-forward 从来不是审计指的重复项。

### 2.2 walk-forward 十一实现的关系判定

| 实现 | 评估对象 | 统计量 | 训练/测试切分 | 判定 |
|---|---|---|---|---|
| W1 `battle.replay_method` | 出号方法→票 | 命中/ROI/奖级 | 全量 as-of | **唯一权威**（方法级） |
| W2 `walk_forward_check` | W1 的 per_issue | 二项 z / t + 多种子中位数 | 复用 W1 | **薄包装**，非重复 |
| W3 `replay.offline_replay` | 出号方法→票→落库 | 同 W1 + MDE | as-of + 500 滚动窗口阈值 | **薄包装**（复用 `battle._tickets_for_method`），差异仅在阈值窗口 |
| W4 `_walk_forward_brier` | 概率向量 | Brier | 全量 as-of，n_eval 截断 | **本质不同**（对象=分布，非票） |
| W5 `backtest_pattern` | 规律触发事件 | t + Wilson + BH 分级 | as-of（min_start 起） | **本质不同**（对象=触发事件，含 horizon） |
| W6 `offline_backtest` | 整套引擎 vs 随机 | 命中差 | as-of，尾窗 issues | **本质不同**（对象=生产引擎 E2E） |
| W7 `evaluate_ml_walkforward` | ML 概率 | Brier/log-loss + 配对 Wilcoxon | **每 refit_every 期重训** | **本质不同**（含训练循环） |
| W8 `run_llm_eval` | 三通道出票 | 配对差 + p | as-of + LLM 外调 | **本质不同**（外部调用、成本约束） |
| W9 `evaluate_signal` | 通用策略函数 | t + 符号 + Bonferroni | 全量 as-of | **同构于 W10 但未接线**（生产零调用） |
| W10 `_strategy_hit_series` | 内置 hot/cold 策略 | （序列，检验在外） | **滚动窗口**（非全量） | 与 W9 同构度最高 |
| W11 `asof_scan` | 维度选择器 | obs/exp 序列 | 全量 as-of（增量 ctx） | **本质不同**（对象=选择器，下游 4 处复用） |

**结论（不含糊）：**

- **真正重复的同一算法（可合并成一份、其余改薄壳）：不在 walk-forward 里，而是 BH×3 与 Wilson×4**（见 §2.4）。walk-forward 里唯一的"包装已有"关系（W2、W3 → W1）**已经是薄包装，无需动作**。
- **参数不同但同构、可合并到一个带参公共函数的：仅 W9 ↔ W10 一对**，但合并收益为负（见下）。
- **本质不同、不能硬合并的：W1/W4/W5/W6/W7/W8/W11 共 7 个**。它们的共同点只有一行 `hist = draws[:t]`——那是 walk-forward 的定义本身，不是可提取的实现。统计量（Brier vs 二项 z vs 配对 t vs Wilcoxon）、评估对象（票 vs 分布 vs 触发 vs 选择器 vs 重训模型 vs LLM 调用）两两不同。
- W9/W10 不值得合并的三条理由：① W9 生产零调用（合并它不减少任何运行路径）；② 基线口径不同——W9 是逐期 6k_t/33（k 浮动、红球 only），W10 蓝球策略基线是 1/16 且 hit 为 0/1（routes_rigor.py:187-190），合并必须先给 `evaluate_signal` 扩蓝球语义，接口破坏面大；③ W10 是滚动窗口、W9 是全量 as-of，窗口语义不同，合并反而引入"哪个窗口对"的口径争论。`/api/rigor/signal` 已有测试钉住行为（`test_v4_rigor.py` 53 个中含 signal 链路）。

### 2.3 佐证：项目已在防止 walk-forward 重复

- `replay.py:236-240`：出票直接 `return battle._tickets_for_method(...)`，注释"方法级出号只有**一份**实现，避免『回放模块自己再写一套』造成两份漂移（V6 的教训：阈值口径写两遍必错）"。
- `walk_forward_gate.py:47-50`：docstring 明确"只读 battle.replay 的结果，不重复实现 walk-forward"。
- `test_qa_self_guards.py:66-82`：甚至有测试守卫"engine 内 `from . import backtest` 不得解析成错误模块"的依赖方向。
- git 历史：`walk_forward_gate.py` 自 V20（c60a298）创建后**从未被改动**（1 个 commit）；`backtest.py` 4 个 commit。重复实现若存在漂移，通常表现为高频并行修改——没有这个信号。

### 2.4 真重复项：BH×3 与 Wilson×4 的合并可行性

#### 实测数值对拍（本机 .venv 实跑，2026-10-25）

- BH：`backtest.bh_adjust`（backtest.py:116）、`llm_eval._bh_adjust`（llm_eval.py:72）、`stats.rigor.benjamini_hochberg`（stats/rigor.py:150）——12 个随机 p 值 + `[0.5,0.5,0.1]` 重复 p 边界，**三份 max diff = 0.0**。
- Wilson：`backtest._wilson_ci`（backtest.py:131，返回上下界）、`hypothesis._wilson_lower`（hypothesis.py:191）、`decision_skill._wilson_lower`（decision_skill.py:25）、`rule_lifecycle.wilson_interval`（rule_lifecycle.py:833）——`k=3,n=10` 四份下界 **0.10778928748621182** 逐位一致（rule_lifecycle 末位差 1e-16，float 舍入）。

（注：审计 P2-1 数的"五份 Wilson"含 `api_app.py:54` 的 re-export，那是同一对象，实际独立实现是 4 份。）

#### 合并后目标函数

```python
# lottery/stats/rigor.py 新增（或收编 hypothesis 版公式）：
def wilson_lower_bound(hits: int, n: int, z: float = 1.96) -> float: ...
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]: ...
# BH 已有真源：stats.rigor.benjamini_hochberg
```

原函数改薄壳（保留函数名，调用点零改动）：

| 薄壳 | 现定义 | 指向 |
|---|---|---|
| `backtest.bh_adjust` | backtest.py:116 | `stats.rigor.benjamini_hochberg` |
| `llm_eval._bh_adjust` | llm_eval.py:72 | 同上 |
| `backtest._wilson_ci` | backtest.py:131 | `stats.rigor.wilson_interval` |
| `hypothesis._wilson_lower` | hypothesis.py:191 | `stats.rigor.wilson_lower_bound` |
| `decision_skill._wilson_lower` | decision_skill.py:25 | 同上 |
| `rule_lifecycle.wilson_interval` | rule_lifecycle.py:833 | `stats.rigor.wilson_interval` |

#### 需要改动的调用点清单（逐个）

改的是**定义处**，调用点全部不动：`backtest.py:278`、`mining.py:441`、`llm_eval.py:235`、`backtest.py:222,258`、`decision_skill.py:130`、`rule_lifecycle.py:788`、`api_app.py:54`（re-export 不变）、`routes_llm_ops.py:27`（从 decision_skill import，不变）。

#### 风险点

1. **浮点精度**：新 stats 实现若公式项序不同，末位可差 1e-16。守卫测试 `test_v10_cost.py:366` 用 `abs=1e-12` 容差，安全。
2. **stats 包纪律**：`stats/__init__.py:23` 要求"只允许标准库，numpy 需纯 Python 回退"。Wilson 用 `math.sqrt` 即可（rule_lifecycle 版已示范）；BH 的 stats 版已用 numpy（stats/rigor.py:154 既有事实），保持即可。
3. **hypothesis._wilson_lower 疑似死代码**：全仓无内部调用者（仅 decision_skill 注释提到它、test_v10_cost 拿它对拍）。合并时可顺手删或留薄壳，二选一需 Lead 定（留薄壳则测试零改动）。

#### 合并后哪些现有测试会变红

**逐个核查结论：零个。**
- `tests/test_v10_cost.py::test_wilson_lower_matches_hypothesis_implementation`（:366）：对拍 `api_app._wilson_lower` vs `hypothesis._wilson_lower`——两者都成同一真源的薄壳后仍逐位相等，绿。
- `tests/test_v11_rule_lifecycle.py::test_t10_wilson_matches_reference`（:548）：钉具体数值 `(0,10)→≈0.2065`——薄壳不改数值，绿。
- BH：`backtest.bh_adjust`/`llm_eval._bh_adjust` **无任何测试直接引用**（grep 零命中）；`mining.py:441`、`llm_eval.py:235` 走薄壳名不变。
- 前置条件：stats 包不得 import lottery 业务模块（包纪律 2）——BH/Wilson 是纯函数，满足。

#### 建议

**值得合并，且是低风险高确定性的一类**：6 个定义改薄壳 + stats 包收编 2 个纯函数，预计 <60 行 diff，零测试变红，消除审计 P2-1 指出的"真源修 bug 副本不跟进"风险。**walk-forward 十一实现：不建议合并**——它们不是同一算法的副本，硬合并会把"票/分布/触发/重训/外调"五种对象塞进一个函数签名，破坏 8 组契约测试（test_v4_battle 42、test_v4_blend 18、test_v10_replay 21、test_v20 11、test_v23 6、test_v10_calibration 31、test_v4_rigor 53、test_v10_ml_calibration 23），收益仅为删掉每人一行的 `draws[:t]`。建议在审计账本上把"债务 A"改写为"已核实：BH/Wilson 真重复（可合并）；walk-forward 为多口径并存，非重复"。

---

## 第三部分：错误返回形态实测

### 3.1 统计方法

AST 解析 `lottery/` 全部 70 个 .py 的每条 `return`（不跨嵌套函数），按「是否 FastAPI 路由函数」（decorator 为 `@router.get/post/...`）与「返回表达式形态」分类。这比 grep 可靠：grep 的 `JSONResponse(` 134 = 127 条 return + 7 条 `return None, JSONResponse(...)` 元组形态（helper 返回值由路由转交，仍是 HTTP 响应）。

### 3.2 总量与路由/内部区分

| 形态 | 路由函数内 | 内部函数内 | 合计 |
|---|---|---|---|
| `return JSONResponse({...})` | 119 | 8（中间件/辅助器，见 3.4） | 127 |
| `return {"ok": False, ...}`（裸 dict） | **4**（全部是 `POST /api/llm/test` 的 200 业务结果，routes_llm.py:87,124,152,161） | **41** | 45 |
| `return {…}` 无 `ok` 键（无信封业务对象） | 11 | ~ | ~ |
| `return {...}` 总数 | 47 | 394 | 441 |
| `return HTTPException(...)` | **0** | **0** | **0** |
| 其他 return（业务对象/表达式） | 127 | 1576 | 1703 |

**关键区分（任务书要求的）**：41 处内部裸 dict（`ml_model.py:1623-1681`、`idle_mining.py:392-399`、`popularity.py:175,256`、`patterns_runtime.py:391-427`、`rule_lifecycle.py`、`data_check.py:32`、`variance_picker.py:210,277`、`ai_hypothesis.py:167-187`、`data_audit.py:1030`、`llm_config_store.py:109`、`prereg_scheduler.py:239`、`evolution.py:470` 等）**全部是内部 API 约定**（被路由再包装或被其他模块消费），**不是 HTTP 形态**。审计把这一层混进"错误返回形态"计数，属于口径污染——用户对此的怀疑（"内部函数返回值不是形态不统一"）**成立**。

（任务书实测数"83 处裸 `{"ok": False`"的口径未确认——按本文口径：裸 return 41 + `out = {"ok": False,...}` 模板初始化 8 + 其他散布形态，textual `{"ok": False` 总数 184，其中 108 处在 `JSONResponse({` 内。）

### 3.3 A 类：真实 HTTP 错误响应的精确画像

路由函数内 `JSONResponse + ok=False` 共 **117 处**；再加中间件/辅助器构造的 HTTP 错误 9 处定义（`api_app.py` 5 处：并发闸门 503、鉴权 401×2/403×2、Bearer 401；`routes_predict._strict_mode_reject_response` 422×1；`auth_api._err` 与 `user_llm_api._err` 两个工厂函数共 56 个调用点）。

**状态码分布（路由级 117 处）**：

| 状态码 | 数量 | 备注 |
|---|---|---|
| 400 | 55 | 参数/数据不足类 |
| 503 | 45 | 能力未就绪/内部失败降级 |
| **200** | **8** | routes_diagnosis.py:216、routes_llm_ops.py:48,147,165,168,239,299、routes_rules.py:412 |
| 500 | 4 | routes_audit×2、routes_predict:600、routes_tasks:123 |
| 502/403/404/422/429 | 各 1 | 502=routes_data:104，422=predict 严格门控，429=LLM 配额，403=CSRF，404=task 不存在 |
| （401/423/409） | （鉴权层） | 在 `api_app` 中间件与 `auth_api._err`（:212，`{"ok":False,"error":msg}` 工厂，code 缺省 400） |

**响应体 key 组合分布（117 处，集合无序）**：

| 组合 | 处数 |
|---|---|
| `{ok, error}` | 62 |
| `{ok, error, detail}` | 20 |
| `{ok, available, error, detail}` | 14 |
| `{ok, available, error}` | 7 |
| `{ok, error, reason, retry_after, unfitted_reason}` | 5 |
| 其他语义组合（method/llm_required/csrf_required/strict_mode/hint/supported/can_fallback 等） | 9 |

**key 出现频次**：`ok` 117/117、`error` 117/117（**100% 双必带**）、`detail` 34、`available` 22、`reason` 6、`unfitted_reason`/`retry_after` 各 5，其余为端点语义字段各 1-2。

**结论**：真实 HTTP 错误形态**本质上只有 1 种信封**（`ok:false + error`，可选 detail/available）+ 2 个合法特例（422/429 的语义扩展字段）。审计所称"四形态并存"中：
- 「真 4xx/5xx + ok:false」= 主流形态，**高度统一**；
- 「HTTP 200 + ok:false」= **真问题但仅 12 个端点**（8 处 JSONResponse + 4 处 llm/test 裸 dict），集中在 `routes_llm_ops.py`（监控会把失败当成功，审计影响描述成立）；
- 「裸业务对象无信封」11 处（health/app_info/predict/patterns items 等）是**有意的探活与前端直读契约**，不是错误形态（它们没有 `ok:false`）；
- 「detail 键重复」= **≥6 处确认**：`routes_rigor.py:109`、`routes_rules.py:60`（`_battle_or_503`）、`:541`（battle_report）、`:568`、`:575`、`:614`——后者覆盖前者，`detail` 永远只剩异常文本、丢失"内部：xxx() 抛出异常"诊断行。审计指出的这个 bug 属实。

### 3.4 前端消费面（web/static/app.js）

- **HTTP 状态码错误**：统一走 `api()` 包装器（app.js:367-395）——非 2xx 时解析 body、**优先取 `j.error`** 作提示（:385），`err.status`/`err.payload` 挂在异常上，401/403/423 进 `handleAuthFailure`（403+CSRF 自动重试一次）。**fallback 存在**：`j.error || ("HTTP " + r.status)`。
- **HTTP 200 + ok:false**：23+ 处显式判 `r.ok === false`（app.js:2486,2673,2825,2885,2958,3056,3109,3176,3694,3887,4761,4883,4993…），读 `.error` 20+ 处。**如果后端把 200+ok:false 改成真 4xx/5xx**，这些分支会落到 `api()` 的 throw 路径——提示文本仍是 `j.error`（fallback 兼容），但**会丢失各加载器里针对"降级卡片"的专门渲染**（例如 app.js:2958-2960 ML 诊断不可用时的专用 note 样式）。改造时需逐端点确认展示降级是否可接受。
- **读到的字段清单**：`error`（主导）、`reason`（:3457,3461,3900）、`detail`（**仅 1 处** :4995，且写作 `res.detail || res.error`）、`hint`（:4188）、`can_fallback`/`fallback_hint`（:780）、`unfitted_reason` + `retry_after`（:2608-2652，注释明确两者是层级关系非备选）、`llm_required`（:1254 注释）、`action`（表格列展示）。
- 前端**从未依赖** `detail` 作主提示（唯一一处还带 error fallback）→ 修"detail 重复键"对前端零风险。

### 3.5 第三部分结论

审计"错误返回四形态并存"**部分成立、定性偏重**：
1. 成立的部分：HTTP 200 + `ok:false` 确有 12 个端点（8 JSONResponse + 4 llm/test dict），监控语义污染是真的；`detail` 重复键 ≥6 处是真的。
2. 夸大的部分：把 41 处**内部函数** `{"ok": False}` 与 11 处**无 ok 键的业务对象**计入"错误形态"。前者是内部 API 约定（正确设计），后者是探活/展示契约（无错误语义）。真实 HTTP 错误信封只有 1 种（`ok+error` 100% 覆盖），统一度远好于"四形态并存"的表述。
3. 修复建议排序：① 6 处 detail 重复键（一行/处，零风险）；② 12 处 200+ok:false 改真实状态码（需逐端点过一遍前端降级渲染，test_v16_strict_fallback / test_v28_llm_l2_gate 可能钉住个别 200 语义——**未确认**，改前需跑这两文件）；③ 无信封业务对象**保持不动**。

---

## 附：本报告的证据可复现性

- AST 扫描脚本与中间结果：`/tmp/scan2.py`、`/tmp/recs.json`（2317 条 return 记录）。
- 数值对拍：BH 三份 max diff=0.0；Wilson 四份 k=3,n=10 下界逐位 0.10778928748621182。
- `walk_forward_check`/`evaluate_signal` 返回 key 来自合成数据实跑。
- 未确认项已在文中标注：W3 的生产调用链、W10 的直接测试覆盖、83 计数口径对账、12 处 200+ok:false 对 test_v16/v28 的影响。
