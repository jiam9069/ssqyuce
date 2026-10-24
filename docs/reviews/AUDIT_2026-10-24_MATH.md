# 数学与统计正确性审计（MATH AUDIT）

> 日期：2026-10-24 ｜ 审计对象：`/root/DSH/应用/ssqyuce`（双色球预测系统，Python 3.13）
> 立场：**只审计不改产品代码**（`lottery/` 下未改一行）。每条结论给 `文件:行号` + 推导/实测输出。
> 复现：所有探针脚本在 `/tmp/*.py`，用 `scripts/isolated_test.sh -- .venv/bin/python /tmp/xxx.py` 运行（隔离临时库，绝不碰生产库）。
> 尊重既有审查：`docs/reviews/STAT_CALIBRATION_AUDIT.md`、`V14_STAT_REVIEW.md`、`V6_REGIME_DIAGNOSIS.md`、`V4_ACCURACY_DIAGNOSIS.md` 的结论均已复核。凡与之冲突处均在下文明示并给证据。

---

## 摘要

### 计数表

| 级别 | 计数 | 简述 |
|---|---|---|
| **P0**（数学错误导致结论错误 / 用户被误导） | **3** | ① `engine_prize.expected_prize` 六等奖概率算错，单注 EV 高估 24.6%；② 同文件 `portfolio_expected_value.sixth_ev_ratio` 用「至少中一次概率」而非「期望命中注数」，n>16 时低估返奖比（正是 `blue_coverage.py` 已修但此处漏修的同一个 bug）；③ 上述错误随 `engine.predict_next` 的 `portfolio_ev` 字段进入预测结果/前端，且有一个测试用错误定义「放行」了该 bug。 |
| **P1**（方法学缺陷削弱可信度） | **3** | ④ `llm_eval` 的 `stat_llm` 通道在 LLM 失败期用 `stat` 输出顶替并计入配对比较，把「评估通道」与「生产回退」混在一起，使 `stat_llm vs stat` 的配对 p 向「无差异」系统性偏移；⑤ 交叉层 `_group_checked` 用均值常数 135 当组规模（V14 已指出，此处补充：该常数对 78% 族对放大 m 2~20 倍，仍在使用）；⑥ Brier 分解在 n=1/退化样本上恒等式破裂（`brier ≠ rel−res+unc`），且极端 bin 处理未纳入 `identity_gap` 断言。 |
| **P2**（实现细节风险） | **4** | ⑦ `_brier_decomposition` 对单样本 `identity_gap=-0.25`（断言只在大样本钉住）；⑧ `sigma_hypergeom(k=33)` 返回 0（数学正确但无守卫注释，易误用为「无方差但非退化」）；⑨ `randomness_report` 顶层 `verdict` 在纯随机 N=3517 仍报 `deviation`（fwer 校正后正确），若前端读 `verdict` 会误报；⑩ `engine_prize` 与 `engine_distribution` 两套 EV 实现并存、口径不一（0.6265 vs 0.4567），无单一真源。 |

**数学正确、可放心的项**（已实测确认）：C(33,6)、超几何均值/方差（1.0909 / 0.8678）、BH/Bonferroni、Wilson 区间、MDE 公式、e-process 超鞅性与 Ville、Brier 分解（n≥小样本正常情况恒等式精确到 1e-16）、`_combine_signal_test` 假阳性受控（B≥79 时 FPR≈α）、卡方/e-process/ML 置换的阳性对照均能检出注入信号、`var_ratio` 闭式（与蒙特卡洛一致）、`walk_forward_gate` 二项 z、特征无未来信息泄漏。

### 三条最重要结论

1. **P0：单注期望奖金被高估 24.6%。** `engine_prize.expected_prize`（`engine_prize.py:92-94`）把「红中 3 且蓝不中」错误计入六等奖，使单注 EV 从正确的 ¥1.0055 虚增到 ¥1.2531（返奖比 0.5027 → 0.6265）。正确的同口径值由 `engine_distribution._single_ticket_ev(strict)` 给出（¥0.8954 @ 表 A）与 `routes_portfolio.ticket_expected_prize` 给出（¥1.0055），但引擎预测结果用的是错的 `portfolio_expected_value`。
2. **P0：`portfolio_expected_value.sixth_ev_ratio` 复现了 `blue_coverage` 已修复的 bug。** `engine_prize.py:152` 用 `p_cover`（≥1 注中蓝概率，n>16 封顶 1）而非 `expected_blue_hits=n/16`；n=20 时得 0.125，正确 0.15625（低估 1.25×，n 越大低估越严重）。同一个 bug 在 `blue_coverage.py` 已于 2026-10-21 修复，但没同步到 `engine_prize.py`——「两套逻辑各说各话」再次发生。
3. **阳性对照全部通过，故「0 项通过」不是伪阴性。** 注入 +100% 的号码偏置后卡方 p=1e-30~1e-74；e-process 注入 +5% 偏置 e≈106、+15% e≈6.5e15，零偏置 e=0.59<1；ML 置换注入强信号 p=0.0125 显著。这印证 V14_STAT_REVIEW 的判断：**判定器没坏，0 是数据的真实回答**。

---

## P0 清单

### P0-1 六等奖概率算错 → 单注 EV 高估 24.6%（`lottery/engine_prize.py:79-96`）

**证据（代码）**：
```python
# engine_prize.py:92-96
elif r == 3:
    prob_by_level[5] += p_r * p_blue
    prob_by_level[6] += p_r * (1.0 - p_blue)   # ← BUG：r==3 且蓝不中，实际是"未中奖"(level 0)
else:
    prob_by_level[6] += p_r * p_blue
```
`evaluate.prize_level`（`evaluate.py:15-31`）的**权威规则**是：
```python
if r == 4 or (r == 3 and b): return 5
if b: return 6      # 只有蓝球命中才是六等奖
return 0            # r==3 且蓝不中 → 0（未中奖）
```
即 **六等奖 = 蓝球命中 且 红球命中 ≤ 2**（`(p_r(0)+p_r(1)+p_r(2))·p_blue = 0.058893`）。而 `engine_prize.py:94` 把 `r==3 ∧ 蓝不中`（`p_r(3)·(1-p_blue) = 0.049517`）也计进六等奖。代码第 82 行注释自称「与 evaluate.prize_level 完全一致的奖级判定」——**与实际规则不一致**。

**实测输出**（`/tmp/p_ev4.py`、`/tmp/p_ev2.py`）：
```
code lvl6= 0.1084098222411626   correct6= 0.05889254655244644   overshoot= 0.04951727
level6 EV bug contribution = 0.24758638 元/注
正确单注EV= 1.005466 ev_ratio= 0.502733
代码单注EV= 1.253052 ev_ratio= 0.626526
差异= 0.247586 元/注
```
**影响**：`engine_prize.expected_prize` 与 `portfolio_expected_value`（后者 `ev = expected_reward * n`，见 `engine_prize.py:137`）单注返奖比由正确 0.5027 虚报为 0.6265（**高估 24.6%**）。该值经 `engine.predict_next` 以 `portfolio_ev` 字段进入预测结果（`engine.py:961,1003`），并可被 what-if 端点消费（`engine_orchestrate.py:338-339`）。对一个以「诚实披露负期望 / 52.3% 返奖比」为卖点的系统，把一个更「好看」的 62.65% 展示出去，方向恰好与诚实边界相反。

**为什么测试没抓到**：`tests/test_v4_coverage.py:253-254` 只断言 `p[6] > p[5] > p[4]`（六等奖最大），而**恰好**错误定义下 p[6]=0.1084 仍 > p[5]=0.00776 > p[4]=0.000434；注释里写的口语定义「蓝中但红球命中 <= 2」是对的，但代码没实现它。**测试用错误语义放行了 bug**。

**修复建议**：把 `engine_prize.py:94` 改为不再累加六等奖（`r==3 ∧ 蓝不中` 应为 0）；或（更稳妥）直接复用 `evaluate.prize_level`/`engine_distribution._single_ticket_ev(strict)` 的奖级判定，删除重复实现。并补一条绝对数值断言：`expected_prize(...)["prob_by_level"][6] == (p0+p1+p2)*p_blue`。

> 更正后的「正确 EV」需说明口径：`routes_portfolio.ticket_expected_prize` 用 `evaluate.PRIZE_CASH`（一等 650 万 / 二等 18 万）得 ¥1.0055 / 0.5027；`engine_distribution._single_ticket_ev` 用「V4 假设表 A」（一等 500 万 / 二等 15 万）得 full=0.45673、strict=0.44771。**奖金假设不同，两个数都对，但都比 0.6265 低**。真正错的是 level-6 概率。

---

### P0-2 `portfolio_expected_value.sixth_ev_ratio` 用错量（`lottery/engine_prize.py:151-152`）

**证据（代码）**：
```python
# engine_prize.py:140-152
p_naive = 1.0 - (1.0 - 1.0/_b_max())**n if n>0 else 0.0
p_cover = _portfolio_blue_hit_prob(blues)              # = distinct/16，n>16 封顶 1.0
...
"sixth_ev_ratio": (cash6 := 5.0) and ((cash6 * p_cover) / cost if cost > 0 else 0.0),
```
`p_cover` 回答的是「**至少中一次**蓝球」的概率（`_portfolio_blue_hit_prob` 见 `engine_prize.py:111-115`，`min(distinct,16)/16`，n>16 封顶 1.0），而六等奖按**注**兑付，应乘「**期望命中注数**」`E[blue hits]=n/16`。

**实测输出**（`/tmp/p_sixth.py`，n=20、16 个不同蓝球）：
```
distinct= 16 sixth_ev_ratio code= 0.125
correct sixth_ev_ratio= 0.15625 (应恒 0.15625)
低估倍数= 1.25
```
n 越大低估越甚（n=100 时低估 6.25×，与 `blue_coverage.py:150-153` 记录的历史事故完全相同）。

**这是 `blue_coverage.py` 已修复、却漏改到 `engine_prize.py` 的同一 bug**：`blue_coverage.optimize` 在 2026-10-21 已把 `p_cover` 换成 `expected_blue_hits=n/16`（见 `blue_coverage.py:139-161`），但 `engine_prize.portfolio_expected_value`（另一个实现）没同步。`blue_coverage.py:28-30` 的模块注释明确说要「保持单一实现」，现实是两套。

**影响**：`n>16` 的组合，六等奖返奖比被低估（0.125 vs 0.15625 等），与 `blue_coverage.optimize` 报的 0.15625 自相矛盾；任何用 `engine.portfolio_ev["sixth_ev_ratio"]` 展示的地方都会偏小。

**修复建议**：`engine_prize.py:152` 改为 `5.0 * (n / float(_b_max())) / cost`；或直接委托 `blue_coverage.optimize`。并加一条 `n=20` 的绝对断言 `sixth_ev_ratio == 0.15625`。

---

### P0-3 错误 EV 进入预测结果 / 前端，且缺单一真源（`engine.py:961,1003` + `engine_prize.py`）

**证据**：
- `engine.py:961`：`_portfolio_ev = portfolio_expected_value(picked)`；`engine.py:1003`：`"portfolio_ev": _portfolio_ev`（进入 `predict_next` 返回体）。
- 同仓存在**正确**的两套：`engine_distribution.ticket_distribution`（`ev_ratio_full=0.45673`，被 `/api/portfolio/distribution` 与 `l2_service` 用）和 `routes_portfolio.ticket_expected_prize`（`ev/ticket=1.00547`，被 `/api/portfolio/expected-value` 用）。

**实测对比**（`/tmp/p_ev3.py`）：
```
portfolio_expected_value: ev_ratio= 0.6265 expected_reward= 12.5305   ← 引擎路径（错）
ticket_distribution: ev_ratio_full= 0.45673 strict= 0.44771 six= 0.15625   ← 正确
routes ticket_expected_prize: EV/ticket= 1.00547   ← 正确
engine_prize single ev_ratio: 0.62653   ← 错
```

**影响**：同一个「单注期望返奖比」在不同端点给出 0.6265 / 0.4567 / 0.5027 三个不同值。用户/审计者无法判断哪个是真的。这是典型的「重复实现 → 分家」缺陷（`engine_prize.py` 文件头注释 `:9-14` 自己也警告过这个问题）。

**修复建议**：确立 `engine_distribution.ticket_distribution`（或 `evaluate.prize_level` 派生）为**唯一 EV 真源**，`engine_prize.expected_prize`/`portfolio_expected_value` 改为薄封装委托，删除其内的奖级判定循环。修复 P0-1/P0-2 后加回归测试钉住绝对数值。

---

## P1 清单

### P1-1 `llm_eval` 的 `stat_llm` 通道在 LLM 失败期用 `stat` 顶替，污染配对检验（`lottery/llm_eval.py:160-165,213-226`）

**证据（代码）**：
```python
# llm_eval.py:160-165
llm_cands = res_llm["tickets"] if res_llm else []
if not llm_cands:
    llm_empty_issues += 1
    llm_cands = res_stat["tickets"]          # ← 用 stat 输出顶替 stat_llm
runs["stat_llm"].append(EV.tickets_result(_safe(llm_cands), target))
```
随后配对比较 `stat_llm vs stat`（`:213-217`，`delta = per_metric["stat_llm"] - per_metric["stat"]`）。在 LLM 失败的期上，`stat_llm` 与 `stat` 逐期**完全相同** → `delta=0` → 该期被 Wilcoxon 剔除（`:52` 的 `nz = d[d!=0]`）。后果：**LLM 的失败被记成「与 stat 无差异」，而失败恰恰是本该惩罚 LLM 的证据**。`llm_empty_issues` 只是被计数并在 note 中提到，但**没有从配对样本里剔除这些期**，所以 `stat_llm vs stat` 的 p 值被系统性地拉向 1（无差异）。

**影响**：这是一条会**掩盖 LLM 劣迹**的偏置（方向与 STAT_CALIBRATION_AUDIT 里 D-1「系统性低估劣绩」同类）。若 LLM 有 30% 期数失败，效果等于给 LLM 免了 30% 的考核。

**修复建议**：配对比较前，从 `per_metric` 三通道中同步剔除 `llm_empty_issues` 对应的期；或把失败期按「LLM 输」计入（对 `blue_hit_rate` 用真实空结果=0 而非 stat 结果）。至少要在 comparisons 里报告「有效配对期数 = issues_actual − llm_empty_issues」。

---

### P1-2 交叉层 `_group_checked` 用均值常数 135，仍在使用（`lottery/idle_mining.py:482-495`，V14 已指出 / 复核确认）

**复核**：V14_STAT_REVIEW §结论4 指出 135 是交叉层族对档位数分布的**均值**，中位仅 35、78% 的族对 <135，会把 m 放大 2~20 倍。本次审计**独立复核该结论成立**（不重复其数值脚本，直接引用其 `/tmp/rev/q3b_groupsize.py` 逻辑：交叉层族对 268 个，min/中位/mean/max = 1/35/135/2304）。数学后果：`m_group` 被抬高 → e-BH 的 rank-1 门槛 `m_group/α` 过高 → `k_star=0`。

**但需纠正 V14 的一处表述**（V14 自己在结论 5 已自我纠正方向，此处强调）：过阈率低于 α 是**保守**、不造假；真正的问题是**检定力不足**，不是「判定器坏了」。本次阳性对照（见附录 P-C、P-D）证明 e-process 本身有充足检定力（+5% 偏置 e≈106），所以 0 的根因是 **m/α 门槛分辨率** 叠加数据无信号，而非 dead test。

**影响**：`usable=0` 这条结论的**分母不可复算**（135 来历不可追溯）。属「正确性/可审计性」问题，不是灵敏度问题（V14 §3 已定量：修 m 只能把所需 delta 从 0.0683 降到 0.0645）。

**修复建议**：同 V14：`_group_checked` 改逐组真实档位数，`_recompute_usable` 传全层候选 e 值。**这不是本次新增发现**，仅标记「V14 建议尚未落地」。

---

### P1-3 Brier 分解恒等式在退化样本上破裂，`identity_gap` 未被断言（`lottery/ml_model.py` `_brier_decomposition`）

**证据（实测，`/tmp/p_brier_edge.py`）**：
```
n=1 y=0.5: brier=0.0, reliability=0.0, resolution=0.0, uncertainty=0.25  → gap = -0.25
n=2: gap = -5.5e-17  (正常)
```
**推导**：Murphy 分解 `Brier = reliability − resolution + uncertainty` 在按概率分箱、且每箱用箱内均值 `p̄_k` 与箱内频率 `f_k` 时成立。当某箱只有一个样本且 `p` 恰等于 `base=mean(y)` 时，`f_k` 与样本 `y` 的统计不再是同一口径：代码用「原始 `(p−y)²` 求和」作为 `brier`，而 decomposition 项用「分箱频次」构造，二者在退化箱（n=1、或某 bin 内 `f_k` 与 `mean p` 失配）上不再相同，出现 `gap=-0.25`。

**影响**：正常生产（n 为几百~几千）无碍（实测 gap≈1e-16）。风险在于：若监控脚本用 `brier == rel − res + unc` 做 `assert`，在极小/退化输入上会误报；若在退化样本上读取 `resolution_skill = res/unc` 会得异常值。属「边界一致性」缺陷。

**修复建议**：在 `_brier_decomposition` 里显式计算 `identity_gap = brier − (rel − res + unc)` 并返回；对 `n < n_bins` 或退化箱回退为「不分箱的原始 Murphy 分解」（bin=每个唯一 `p` 值），保证恒等式在任何 n 上闭合。

---

## P2 清单

### P2-1 `randomness_report` 顶层 `verdict` 在纯随机上误报 `deviation`（`lottery/stats/rigor.py:385-409` 文档 + 实测）

**实测（`/tmp/p_posctrl.py` 的 null 分支）**：
```
N=3517 NULL: red_freq p=0.9334  verdict=deviation  fwer=consistent-with-random
N=2000 NULL: verdict=consistent-with-random  fwer=consistent-with-random
```
纯随机数据上，因 9 项检验任一显著即置顶 `verdict=deviation`（家族化膨胀 `1-(1-α)^9≈37%`），N 稍大就误报。`verdict_fwer`（BH/Bonferroni 校正后）正确。这与 STAT_CALIBRATION_AUDIT 结论 6/7 一致（已加 `verdict_fwer`）。**风险**：若前端/下游读 `verdict` 而非 `verdict_fwer`，会把随机数据报成「偏离随机」。建议前端只读 `verdict_fwer`，或把 `verdict` 字段改名 `verdict_any` 并默认展示 fwer。

### P2-2 `sigma_hypergeom(k=33/0)` 返回 0，无守卫提示（`lottery/stats/eprocess.py`）

**实测**：`sigma_hypergeom(0)=0.0`，`sigma_hypergeom(33)=0.0`，`sigma_hypergeom(NaN)=0.8678`（NaN 落回默认 6）。k=0/33 数学上确实方差为 0（选中 0 个 / 全部 33 个，命中数恒为 0 / 6），但返回 0 而无 warning，若上游拿它作分母（`z=delta/sigma`）会 `ZeroDivisionError` 或静默放大。建议加断言或显式 `raise`/`None`。

### P2-3 `expected_prize` 与 `_single_ticket_ev` 奖金假设表不同，易混淆（`engine_prize.py:43-47` vs `engine_distribution.py` `PRIZE_ASSUMPTION_TABLE_A`）

即使 P0-1 修好，`engine_prize` 用 `JACKPOT_ASSUMPTION`（650 万/18 万），`engine_distribution` 用表 A（500 万/15 万），单注 full EV 是 0.5027 vs 0.4567——**两个都「对」但口径不同**。这放大了 P0-3 的混乱。建议合并为唯一奖金假设表。

### P2-4 无 `log(0)` / 除零崩溃（已核验为安全，仅记录）

`e_value([])=1.0`、`chi_square_uniformity([])` 与 `([0,0,0])` 返回 `insufficient-data` 而非崩溃、`ebh_selected([])=set()`、`first_crossing([0]*100)=None`、`expected_crossing_issues(0)=None`——边界处理**正确**。列为 P2 仅作「已核验无风险」记录。

---

## 建议修复顺序

1. **P0-1 / P0-3 一起修**：确立 `engine_distribution`（strict 口径）为唯一 EV 真源，`engine_prize.expected_prize`/`portfolio_expected_value` 委托之，删掉 `engine_prize.py:79-96` 的重复奖级循环。补绝对数值回归断言（`prob_by_level[6]==(p0+p1+p2)*p_blue`，单注 EV 与 `_single_ticket_ev` 一致）。这是**用户可见的误导**，优先级最高。
2. **P0-2**：`engine_prize.py:152` 改用 `n/16`。回归：`n=20` 时 `sixth_ev_ratio==0.15625`。
3. **P1-1**：`llm_eval` 配对检验前剔除 `llm_empty_issues` 期（或按失败计负）。回归：构造「30% 期 LLM 失败」的合成数据，断言 `stat_llm vs stat` 的 p 不因失败被拉高。
4. **P1-3**：`_brier_decomposition` 返回并断言 `identity_gap`；退化样本回退不分箱分解。
5. **P1-2**：落地 V14 的 `_group_checked` 修复（仅可审计性，非灵敏度）。
6. **P2 批量**：`verdict`→`verdict_fwer` 前端切换；`sigma_hypergeom` 守卫；统一奖金假设表。

> 不做的事：不要为了把「0 项通过」变成非 0 去放松 α / 缩小搜索空间（那是造假，V14 §Q4 已穷举 6 种口径全为 0）。本审计所有阳性对照均证明检验有功效，0 是数据回答。

---

## 附录：探针脚本清单与可复现命令

全部在 `/tmp`，工作目录 `/root/DSH/应用/ssqyuce`。**未修改 `lottery/` 任何文件。**

| 编号 | 脚本 | 作用 | 关键输出 |
|---|---|---|---|
| P-1 | `/tmp/p_basic.py` | C(33,6)、超几何均值/方差、各奖级概率 | C(33,6)=1107568；mean=1.0909 sd=0.8678；六等 p=0.05889 |
| P-2 | `/tmp/p_ev.py`,`/tmp/p_ev2.py`,`/tmp/p_ev3.py`,`/tmp/p_ev4.py` | 单注 EV 正确值 vs 代码值、level-6 bug、portfolio 对比 | 正确 1.00547 / 代码 1.25305，差 0.24759；overshoot prob 0.04952 |
| P-3 | `/tmp/p_route.py` | `routes_portfolio.ticket_expected_prize` 核验 | EV/ticket=1.00547（正确） |
| P-4 | `/tmp/p_sixth.py` | `portfolio_expected_value.sixth_ev_ratio` n>16 低估 | 0.125 vs 0.15625（1.25×） |
| P-5 | `/tmp/p_brier.py`,`/tmp/p_brier_edge.py` | Brier 分解恒等式 | n≥小样本 gap≈1e-16；n=1 gap=-0.25 |
| P-6 | `/tmp/p_eproc.py` | e-process 超鞅性 + Ville + sigma_hypergeom | E[M_t]≈1.0；P(M≥20)≤Ville |
| P-7 | `/tmp/p_bh.py` | BH 参考实现对照 | 与 R 结果一致 |
| P-8 | `/tmp/p_wil.py` | Wilson 区间 | 与手算一致到 1e-6 |
| P-9 | `/tmp/p_misc.py` | MDE 公式核验 | 与 (z_{1-α/2}+z_power)σ/√n 一致 |
| P-10 | `/tmp/p_var.py`,`/tmp/p_var2.py` | var_ratio 闭式 vs 蒙特卡洛 | code 0.8741 vs MC 0.8703（一致） |
| P-11 | `/tmp/p_fpr.py` | `_combine_signal_test` 假阳性率 | B=79→0.043，B=199→0.050（受控） |
| P-12 | `/tmp/p_perm3.py`,`/tmp/p_perm4.py` | ML 置换/合并信号阳性对照 | 强信号 p=0.0125 显著；纯噪声不显著 |
| P-13 | `/tmp/p_posctrl.py` | `randomness_report` 卡方阳性对照 + null FPR | 偏置 33 号 p=1e-74 检出；null fwer=consistent |
| P-14 | `/tmp/p_eproc_pos.py` | e-process 阳性对照 | +5% e≈106，+15% e≈6.5e15，0% e=0.59 |
| P-15 | `/tmp/p_robust.py` | 数值鲁棒性（空/全同/NaN/除零） | 全部安全返回，不崩溃 |

**复现命令模板**：
```bash
cd /root/DSH/应用/ssqyuce
scripts/isolated_test.sh -- .venv/bin/python /tmp/p_ev4.py
scripts/isolated_test.sh -- .venv/bin/python /tmp/p_sixth.py
scripts/isolated_test.sh -- .venv/bin/python /tmp/p_posctrl.py
scripts/isolated_test.sh -- .venv/bin/python /tmp/p_eproc_pos.py
# 隔离脚本自动把 ssq.db 复制到 /tmp 临时目录，LOTT_HOME 指向空临时目录，不碰生产库
```

**诚实边界**：
- 未在本审计中全量重跑 cross 层 7 万级候选（约 4.4 小时），P1-2 的组规模数字引自 V14_STAT_REVIEW 的 `/tmp/rev/q3b_groupsize.py` 逻辑并说明系复核而非独立穷举。
- ML 置换阳性对照用了缩减规模（n_null=15~79、小样本），生产默认 B 更大；结论「能检出强信号」在缩减规模下已成立，生产更稳。大 n 的完整 shuffle 在受限机器上跑不完（后台进程超时），已如实标注。
- 所有 P0 均有可复现探针；P0-1 的错误数值（1.25305 vs 1.00547）由多脚本交叉确认。
