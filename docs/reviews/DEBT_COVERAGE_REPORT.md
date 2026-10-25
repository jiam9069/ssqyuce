# V40 覆盖率欠债报告：无直测模块的定向补测

**泳道**：coverage-agent　**日期**：2026-10-25　**基线**：`1919 passed, 24 skipped, 1 xfailed`

---

## 一、实测的"无直测模块"清单（不是照抄清单，是 AST 全扫描测出来的）

方法：对 `tests/**` 做全量 AST 解析，收集所有 `import lottery.*` / `from lottery.X import y`
（第一版脚本漏了 `from lottery.X import y` 形式，已修正重跑），对 `lottery/*.py` 与
`lottery/stats/*.py` 逐一求"被测试文件直接 import 的次数"。

**结果：23 个模块 0 直测**（对照 lead 给的候选，多发现了 `cli.py` 和 `patterns.py`）：

| 模块 | 直测文件数 | 决定 | 理由 |
|---|---|---|---|
| `engine_blue` | 0 | **补**（a） | `blue_coverage_plan` 是全项目唯一"真实可得的改进"，概率入参/边界形状无测试 |
| `engine_solver` | 0 | **补**（a） | 号位均摊构造与精确枚举是真数学；`test_v12_portfolio` 只测 `engine` 门面，未触达本模块的直接函数 |
| `engine_orchestrate` | 0 | **补**（a） | `_ticket_objective` 的零梯度契约、CRN 估计器确定性，错一个都不响 |
| `engine_rules` | 0 | **补**（a） | off/shadow/active 三模式从未被分支级测试 |
| `engine_constraints` | 0 | **补**（a） | `pass_constraints` 此前只有奇偶单维直测（`test_qa_self_guards`），和值/三区/跨度/连号维靠"恰好" |
| `jsonutil` | 0 | **补**（a） | 10 个端点共用的 numpy→JSON 枢纽，深度保护从未验证 |
| `diagnose` | 0 | **补**（a） | 结构画像字段与相似注排序全无真值锁；`/api/diagnose` 端点也没有 HTTP 测试 |
| `llm_config_store` | 0 | **补**（a） | 部分更新语义、`read()` 不泄露 api_key——安全敏感 |
| `data_fetcher` | 0 | **补**（b） | 网络边界 + V38 诚实计数；测试里只有 monkeypatch 替换它的用法 |
| `data_check` | 1 | **补**（b） | `reconcile` 已有直测（`test_m4_4`），但 `run`/`fetch_source`/审计落库分支为零 |
| `cli` | 0 | **补**（b） | 分发边界 + doctor 契约；**实测发现 `online_check` 子命令分发损坏**（见三.1） |
| `routes_ml` | 0 | **补**（b） | 空库 400 分支从未跑过 |
| `routes_llm_eval` | 0 | **补**（b） | 前置校验分支从未跑过 |
| `routes_patterns` | 0 | **补**（b） | `_pattern_summary` 计数有真实分支 |
| `engine_base` | 0 | **不补** | 纯常量（R_MAX/B_MAX/TICKET_COST），无逻辑可测；写了就是空转 |
| `decision_skill` | 0 | **不补** | 已有**充分间接覆盖**：`from lottery.api_app import decision_skill` 的 13 处断言（`test_v10_cost.py`，含配对口径/注数不匹配剔除/Wilson 边界/None 语义）经 facade 等价触达同一函数对象 |
| `patterns` | 0 | **不补** | 91 个维度的注册表 + 行为函数已被 `discovery.make_filter`（`test_v7_rules_closure`、`test_p4_*`）间接触达；单补注册表形状是空转 |
| `routes_audit` | 0 | **不补** | HTTP 级已命中 4/7 端点（`test_v12_*`），余为纯透传 |
| `routes_diagnosis` | 0 | **不补** | 2/2 端点已被 `test_v10_frontend_cards` 命中 |
| `routes_eval` | 0 | **不补** | 3/7 已命中，余下是 `db.load_eval()` 一行透传 |
| `routes_llm_ops` | 0 | **不补** | 9/9 端点全被命中（`test_v10_cost` / `test_v10_llm_arm`） |
| `routes_methods` | 0 | **不补** | 2/2 已命中（`test_m4_2_api`） |
| `routes_rules` | 0 | **不补** | 6/18 命中；未命中的多为"跑一轮挖掘"的长任务包装，分支在 `bg_tasks`/`patterns_runtime` 已测 |

> 精确化说明：`routes_*` 的"命中"用 OpenAPI 实测（`app.openapi()` 的 95 条注册路径
> vs 测试实际请求过的 67 条），不是 grep 字符串。

---

## 二、新增测试与变异验证记录

新增 **72** 个测试（a=44，b=28），**每个真值锁都做过 worktree 变异验证**。
变异只在 `git worktree /tmp/wt_cov`（HEAD=3785736）注入，验后即还原，共享工作区零污染。

### 变异 → 抓获对照表（全部实测"变异后变红"）

| # | 变异文件:行 | 变异内容 | 被抓获的断言 | 结果 |
|---|---|---|---|---|
| 1 | `engine_blue.py:180` | 排序方向反转（`argsort(-probs)` → `argsort(probs)`） | `orders_by_probability_descending` | 红 ✅ |
| 2 | `engine_blue.py:172` | 删 isfinite/sum≤0 守卫 | `falls_back_to_uniform_on_bad_probs` | 红 ✅ |
| 3 | `engine_blue.py:181` | `min(k, B_MAX)` → `min(k, B_MAX+1)` | distinct-blues 断言 | 红 ✅ |
| 4 | `engine_blue.py:186` | >16 补号 `min(order,key=…)` → `order[0]` | `fills_least_used_first` | 红 ✅ |
| 5 | `engine_solver.py:65` | 均摊 `i < extra` → `i <= extra` | `balanced` 三条 | 红 ✅ |
| 6 | `engine_solver.py`（ΣA 公式） | `c(c-1)/2` → `c²/2` | `min_sum_a` 手算表 | 红 ✅ |
| 7 | `engine_solver.py:241` | 蓝球编号 `i+1` → `i` | `portfolio_plan_blues_are_distinct` | 红 ✅ |
| 8 | `engine_solver.py`（`_portfolio_plan` 下界） | `1 <= n` → `0 <= n` | `rejects_out_of_range` | 红 ✅ |
| 9 | `engine_distribution.py:286` | 全枚举 `>= r` → `> r` | `hypergeometric_truth`（0.004901 真值） | 红 ✅ |
| 10 | `engine_orchestrate.py:162` | `blue_owned_by_plan` 分支残留蓝球项 | `no_blue_gradient_when_owned` | 红 ✅ |
| 11 | `engine_orchestrate.py` `PmaxEstimator.p` | 返回值加 `1e-9·random()` | `common_random_numbers_are_deterministic` | 红 ✅ |
| 12 | `engine_rules.py:194` | `if apply_enabled()` → `if True`（影子也排序） | `shadow_boosts…never_reorders` | 红 ✅ |
| 13 | `engine_rules.py:183` | `RULE_CONF_BOOST * n` → `RULE_CONF_BOOST` | shadow 的 confidence 增量断言 | 红 ✅ |
| 14 | `engine_constraints.py:95` | 奇偶 `not in` → `in` | `independently_blocks` + `consecutive_group` | 红 ✅ |
| 15 | `engine_constraints.py:87` | 和值边界反转 | `independently_blocks` | 红 ✅ |
| 16 | `engine_constraints.py:99` | 删跨度检查 | `independently_blocks` | 红 ✅ |
| 17 | `engine_constraints.py:102` | 连号"段头计数"→"相邻对计数" | `consecutive_group_counting`（第二个样本） | 红 ✅ |
| 18 | `engine_constraints.py:48` | 删空输入守卫 | `rejects_empty_history` | 红 ✅ |
| 19 | `jsonutil.py:31` | 深度上限 12 → 1000 | `depth_guard` | 红 ✅ |
| 20 | `jsonutil.py:39` | 删 numpy 分支 | `converts_numpy` | 红 ✅ |
| 21 | `features.py:114`（AC 公式） | `len(diffs)-(k-1)` → `len(diffs)-k` | `structure_profile_hand_computed`（ac=10） | 红 ✅ |
| 22 | `diagnose.py:68` | 对称差 → 交集反推 | `hamming_and_ordering` | 红 ✅ |
| 23 | `diagnose.py:71` | 排序 tiebreak 反转（`-x[1]` → `x[1]`） | `hamming_and_ordering`（加 E 样本后） | 红 ✅ |

### 变异过程中抓到的"我自己的假绿"（如实报告，共 3 处，全部重写）

1. **`falls_back_to_uniform`（变异 #2 先不红）**：第一版用 NaN×16/全负/全零当"坏
   输入"——但 `np.argsort` 对这些退化输入恰好仍产出 [1,2,3,4]，删守卫不红，是
   **不可观察变异**。第二版换成 [3,0,1,0…]，又发现它**根本不是坏输入**（和 4>0，
   是合法参数，回退反而错）。最终版用"合法形状 + 含单个 NaN"，删守卫必红。
2. **`consecutive_group_counting`（变异 #17 不红）**：样本 [1,2,9,10,20,21] 下
   "段头计数"与"相邻对计数"恰好都得 3。加了第二个样本 [1,2,3,9,10,11]
   （4 个相邻对 / 2 段）后变异变红。
3. **`find_similar_draws` 的 tiebreak（变异 #23 两次不红）**：第一版样本没有
   "同 diff、蓝中 vs 蓝不中"的对；第二版补的 E 样本 [3,7,8,15,22,31] 对称差其实是
   **4**（凭直觉造样本造错了）。最终 E=[3,7,8,15,23,30]（对称差 2、蓝不中）才抓住。

> 这 3 处都印证了铁律：**没有变异验证的"绿"不可信**。

### 不可观察变异的说明（非假绿，是等价代码）

- `n_tickets <= 0` → `< 0`：`k=0` 时 `order[:min(0,16)]=[]`，行为逐位相同，
  该变异在当前实现下**不可观察**，不属于测试缺口。
- 均匀回退权重 `1/B_MAX` → `1/(B_MAX+1)`：只影响内部权重不影响排序输出，同理。

---

## 三、额外发现（超出"补测试"本身的产出）

### 3.1 【真 bug】`lottery online_check` 子命令分发损坏（未修，待裁决）

`cli.main()` 的分发是 `globals().get("cmd_" + args.cmd)`（`cli.py:179`），而
`online_check` 的处理函数叫 `cmd_online`（`cli.py:81`）⇒ `lottery online_check`
**必然报"未知命令 online_check"退出码 2**（隔离环境实测复现）。
已用**存量缺陷锁**钉住：`test_cli_online_check_dispatch_is_broken_today`。
修复时把函数改名为 `cmd_online_check` 并同步更新 `handler_of` 映射表即可。

### 3.2 【conftest 缺口】`llm_config_store.save()` 污染同 worker 后续测试（修复归属 test-speed 泳道）

**发现路径**：我写 `test_llm_config_read_never_exposes_api_key` 时调了
`llm_config_store.save()`；跨文件组合跑时 `test_routes_llm_eval_run_requires_llm_configured`
（b 文件，依赖"LLM 未配置"前提）在 a 文件之后跑就会翻车。

**根因**：`llm_config_store.py:66-78` 直接赋值 `config.LLM_BASE_URL / LLM_API_KEY /
LLM_MODEL / LLM_MODEL_LIST / LLM_SAMPLES / LLM_DISABLED` 并写对应 `LOTT_LLM_*`
环境变量，这 6 个属性**不在** `tests/conftest.py` 的 `_ISOLATED_CONFIG_ATTRS`
（`tests/conftest.py:50`）名单里 ⇒ 隔离 fixture 恢复时不还原它们 ⇒
`llm_configured()`（`config.py:581`）在后续测试里恒 True。

**复现**（本机有真实 `data/llm_config.json` 时必现）：

```bash
scripts/isolated_test.sh -- .venv/bin/python -m pytest \
  tests/test_v40_module_coverage_a.py tests/test_v40_module_coverage_b.py -q
# 修复前的失败点：test_routes_llm_eval_run_requires_llm_configured
```

**处置**：
- 我的 4 个 `save()` 用例已加**文件内自兜底** fixture `llm_config_guard`
  （快照/恢复 6 属性 + 5 个环境变量），不再依赖 conftest 扩名单；
- b 文件的"LLM 未配置"用例改为**显式** `monkeypatch.setattr(config, "LLM_DISABLED", True)`
  制造前提（不再依赖环境恰好没配 LLM）；
- conftest `_ISOLATED_CONFIG_ATTRS` 扩名单由 **test-speed 泳道**落地，两边互补不冲突。

### 3.3 真值锁的外部推导来源（供复核）

| 断言 | 真值 | 推导 |
|---|---|---|
| P(≥1 蓝) 去重覆盖 | n/16 | 组合论证（覆盖数/16）；对照 `1-(15/16)^n` 独立算 |
| 最小显著样本 | 735 | `ceil((1.96·√(p(1-p))/Δ)²)`，p=1/16、Δ=0.0175 |
| ΣA\* 表 | 0…93 | 交换论证 + `divmod(6n,33)` 手算 |
| p_ge4 单注 | 0.004901 | 超几何 `C(6,4)C(27,2)/C(33,6)` |
| 两注不相交 p_ge4 | 0.009802 | 容斥等式（4+4=8>6 ⇒ 互斥） |
| AC 值 | 10 | 15 个差值去重 − 5，手数 |
| 结构画像 | 84/(4,2)/(3,2,1)/27 | 手算 |
| ev_conserving | 12 / 15 / 0 | 并集−Ā 手算 |
| `_pattern_summary` | {A:2,C:1} | 手数 |

---

## 四、最终数字

- 新增测试：**72**（`test_v40_module_coverage_a.py` 44 + `test_v40_module_coverage_b.py` 28）
- a+b 组合（两种顺序）：`72 passed`
- 变异验证：**23 处变异全部变红**；过程中抓出并重写了我自己的 3 处假绿
- 明确不补：**10 个模块**（清单与理由见第一节；候选里其余模块属于纯透传/已有充分
  间接覆盖/纯常量，补了只会虚增测试数）
- 全量回归结果见本轮验收跑（由 lead/test-speed 汇总）
