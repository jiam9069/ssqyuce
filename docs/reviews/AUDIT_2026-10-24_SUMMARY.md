# 双色球系统 · 综合审计总报告（2026-10-24）

- 审计对象：`/root/DSH/应用/ssqyuce`（APP_VERSION `4.15.0` / APP_BUILD `2026-10-V35`，git HEAD `6c30f88`）
- 容器：`ssq-predictor`（healthy，`python:3.11-slim` → 实际 Python **3.11.17**）
- 审计日期：2026-10-24
- 团队构成：资深架构师、数学家/统计学家、资深程序员、测试工程师（4 名子智能体并发）+ Lead 独立复核

## 0. 审计纪律与产物

**只读审计**。`lottery/`、`web/`、`tests/` 一行未改（`git status` 已验证）。所有实测跑在 `scripts/isolated_test.sh` 隔离的临时库副本上；变异测试在 `git worktree` 中进行，用后已删除。

| 分报告 | 内容 |
|---|---|
| [AUDIT_2026-10-24_ARCH.md](AUDIT_2026-10-24_ARCH.md) | 架构层：依赖方向、并发、配置漂移、部署形态 |
| [AUDIT_2026-10-24_MATH.md](AUDIT_2026-10-24_MATH.md) | 数学统计：奖级/EV、检验效力、阳性对照 |
| [AUDIT_2026-10-24_CODE.md](AUDIT_2026-10-24_CODE.md) | 代码层：鉴权、并发、SQL、SSRF、前端 XSS |
| [AUDIT_2026-10-24_QA.md](AUDIT_2026-10-24_QA.md) | 测试质量与黑盒实测：CSRF、假绿、边界 |
| 本文件 | 去重合并后的**总清单**与优先级 |

**全量回归基线**（Lead 亲测，21 分钟）：
```
1 failed, 1823 passed, 24 skipped, 1 xfailed in 1259.23s
FAILED tests/test_v14_engine_split.py::test_predict_next_output_is_byte_identical_to_pre_split_baseline
```

---

## 1. 总缺陷计数（去重后）

| 级别 | 条数 | 一句话 |
|---|---|---|
| **P0** | **5** | 每一条都足以单独阻断发布 |
| **P1** | **17** | 生产事故 / 数据完整性 / 可信度 |
| **P2** | **18** | 可维护性与防御纵深 |

---

## 2. P0 清单（5 条，全部经 Lead 独立复核）

### P0-1 六等奖判定错误 → 单注 EV 高估 24.6%
- **位置**：`lottery/engine_prize.py:93-94`
- **错误**：`r==3 且蓝不中`（3红+0蓝，官方**不中奖**）被计入六等奖。
- **实测**：逐 (r,blue) 14 组合对拍，唯一不一致项就是 `r=3,b=0`；六等奖概率 1/9 vs 官方 1/17；中奖概率合计 0.1166 vs 0.0671；单注 EV ¥1.2531 vs ¥1.0055。
- **自相矛盾**：同仓 `evaluate.prize_level:29` 判 3+0 为 `0`（正确），`engine_prize` 注释却称"与 evaluate.prize_level 完全一致"。
- **影响**：`portfolio_ev` 字段进入预测结果（`engine.py:1003`），前端展示与选号目标函数都基于错误数值，方向是**高估收益**。
- **修复**：删除 `prob_by_level[6] += p_r * (1.0 - p_blue)` 这一行。

### P0-2 跨租户越权读取任务（IDOR）
- **位置**：`lottery/routes_tasks.py:29-57`（`_task_visible_to_current_user` 恒 `True`）+ `db.py:1033`（`load_task` 无 user 过滤）
- **实测**：用户 B（uid=17）带合法会话读用户 A（uid=16）的私有任务 → **200**，返回 `result_json` 全文。
- **前提**：`task_id` 为 `{kind}_{8hex}`（`bg_tasks.py:43`），32 bit 可枚举。
- **修复**：`return task.get("user_id") in (None, current_user_id())`，并对 `/api/tasks/{id}` 与 `/api/tasks` 同口径。

### P0-3 `/api/predict` 写端点不校验 CSRF
- **位置**：`lottery/routes_predict.py:356` `predict()` 从未调用 `_csrf_ok`
- **实测**（Lead 复核）：`AUTH_MODE=enforce` 下
  - 对照写端点 `/api/user/llm/config`：无 token → **403 CSRF 校验失败**；错 token → 403
  - 被测 `/api/predict`：无 token → **200 并写库**；错 token → **200 并写库**
- **影响**：跨站可伪造"生成预测并写入票单"，违背项目自述的"写操作必须带 CSRF 双提交校验"。
- **修复**：在 `predict()` 入口加 `if not auth_api._csrf_ok(request, sess): return 403`（与其他写端点同构）。

### P0-4 `predict_next` 非确定性 → 回测结论随重复运行漂移
- **位置**：`lottery/engine.py:767` `rng = rng if rng is not None else random.Random()`（无种子）
- **实测**（Lead）：
  - 同一进程、同一输入连跑两次 → 指纹 `1be697ea…` vs `ad1f33f1…`；差异 33 处，**票号本身不同**（`tickets[0]/reds[0]` 7 vs 5）
  - 显式传 `random.Random(42)` → 三次完全相同（`f39a6c0e…`）⇒ 根因确认是未播种 RNG
- **影响链**：`evaluate.py:93` 造了 `random.Random(seed)` 但只传给随机基线（`:107`），**没传给** `predict_next`（`:99`）→ 离线回测系统组不可复现：
  ```
  同数据同 seed=7 跑两次：delta_red_hits = -0.0400  vs  -0.1133
  ```
  而 `hypothesis.py:1445` 的判决阈值是 `|delta| ≥ 0.05` ⇒ **同一份数据可能一次判"无显著差异"、一次判"反对"**。
- **修复**：`predict_next` 增加 `seed` 参数并默认固定（如 `config.PREDICT_SEED = 20261024`）；`evaluate.offline_backtest` 把 `rng` 透传进去；`routes_predict` 生产路径同样固定种子以保证同一期可复现。

### P0-5 无会话请求被静默归属主管理员（`AUTH_MODE=off` 默认路径）
- **位置**：`api_app.py:190-197` + `db.py:539-541`
- **实测**：`AUTH_MODE=off` 时 `resolve_tenant(None)` → **1**（主管理员 uid）。`docker-compose.yml:49` 默认 `LOTT_AUTH_MODE=off`。
- **现状**：本机 `.env:38` 已设 `enforce`，暂时安全；但**默认值与部署文档没有把 enforce 变成上线必设**。
- **修复**：compose 默认改 `enforce`；`off` 时 `resolve_tenant(None)` 应返回 `SYSTEM_USER_ID` 而非主管理员。

---

## 3. P1 清单（17 条，按主题）

### 安全与鉴权
1. **登录阻塞事件循环** — `auth_api.py:223-294` 在 `async def` 里同步跑 scrypt，单次 145–550ms 冻结全站；匿名可达。实测 heartbeat 最大间隔 550ms。（CODE）
2. **认证后 SSRF + 响应体回显** — 用户可写任意 `base_url`（`user_llm_api.py:171`），`llm_client.py:582` 把 `HTTP {code}: {r.text[:200]}` 经 503 回传。（CODE）
3. **角色降级不失效会话** — `auth_api.py:498-512` 改 role 未 `token_version+=1`；中间件重读 DB 兜住主闸门，故列 P1 下沿。（CODE）
4. **`/docs` `/openapi.json` `/redoc` 返回 401 而非 404** — 泄露端点存在性（Lead 实测三者为 401）。（QA）

### 数据与计算正确性
5. **`sixth_ev_ratio` 随 n 漂移** — `engine_prize.py:152` 用 `p_cover` 而非 `n/16`；实测 n=1→0.15625、n=20→0.125、n=100→**0.025**。这正是 `blue_coverage.py` 已修却漏同步的同一个 bug。（MATH）
6. **`ev_ratio_sixth = 0.15625` 推导不成立** — 官方应 0.294463；"仅蓝位"口径应 0.3125。（MATH）
7. **真值锁指纹失效** — `test_v14_engine_split.py:270` 期望 `09cef0f6…` 实际 `68730403…`；干净 HEAD worktree 同样红 ⇒ 数据/环境漂移。（QA + Lead 复核）
8. **`n_tickets` 无范围校验** — `-3` → 200 + **0 注** + `shortfall_reason=null`；`0/1/100000` 静默变 5。（QA + Lead 实测）

### 架构与并发
9. **配置三方漂移集群** — `LOTT_SCHEDULER`（compose `1` vs 代码 `0`）、`LOTT_LLM_ONLY_OUTPUT`/`N_TICKETS`（README 与代码相反）、`.env.example` 缺整段 W08 鉴权键。（ARCH）
10. **后台一次性线程连接泄漏** — `bg_tasks.py:66` 每任务起线程各自 `db.get_conn()` 且从不 `db.close()`（`db.py:247-286`）。（ARCH）
11. **依赖方向守卫只守一条** — `test_p2_dependency_direction.py` 只禁"底层 import api_app"；`evaluate.py → engine` 等倒置裸奔。（ARCH）
12. **调度器断链** — `api_app.py:344` 读 `draws`，`:388` 才赋值 ⇒ 必抛 `NameError` 被 `except` 吞掉，prereg 注册永不执行。（CODE + Lead AST 复核）
13. **重计算端点无并发保护** — 无速率限制中间件，`/api/predict` 可并发打满。（ARCH）

### 统计方法学
14. **`llm_eval` 失败期用 `stat` 顶替并计入配对** — 使 `stat_llm vs stat` 的 p 向"无差异"偏移。（MATH）
15. **`_group_checked` 用均值常数 135 当组规模** — 对多数族对放大 m。（MATH）

### 测试工程
16. **全量回归 21 分钟** — 无法高频运行，回归保护退化为发布前仪式。（QA）
17. **22 个模块无测试文件直接触达** — 含 `patterns.py`(883)、`routes_rules.py`(626)、`engine_prize`（P0-1/P0-2 所在地）。（QA）

---

## 4. P2 清单（18 条，摘要）

代码层（7）：`structure_scores` 值未转义 → 存储型 XSS（`app.js:847` + `engine.py:247`）；`admin_patch_user` 陈旧缓存；`save_pattern_results` 异常留开事务持写锁；`updates_draws` 返回值语义（"处理"报成"新增"）；`_throttle` 读宿主 `/proc/stat` 而非 cgroup；登录 cookie `LOTT_COOKIE_SECURE=0` 与 `__Host-` 前缀冲突；`model` 配置缺字段导致 500。

数学层（4）：`sixth="published"` 非互斥口径（概率合计 0.0707 > 官方 0.0671，高估 2.01%）；`_brier_decomposition` 对非二值 `y` 恒等式破裂（接口未校验，二值输入精确到 1e-16）；`sigma_hypergeom(k=33)` 边界无守卫；`randomness_report` 顶层 `verdict` 未过 FWER 会被误读。

架构层（4）：三/五份重复统计实现；错误返回四形态并存；死代码集群；docs 端点靠删路由保护。

测试层（3）：51 个文件用源码文本断言（把 `_brier_blend` 清空仍有大量用例绿）；路由清单缺失（前端调不存在的端点只静默 404）；`shortfall_reason` 语义空转。

---

## 5. 已核查**未发现问题**的项（避免误伤）

- **统计检验效力（阳性对照全通过）**：注入 +100% 号码偏置 → 卡方 p=1e-74 检出；e-process 注入 +5% → e≈106，零偏置 e=0.59<1；ML 置换注入强信号 p=0.0125。**「0 项通过」是数据真实回答，不是伪阴性**。
- **基础数学**：C(33,6)=1107568、总组合 17721088、红球命中期望 1.0909、蓝球 1/16、超几何方差、`var_ratio` 闭式斜率 11/54、Wilson 区间、MDE、BH/Bonferroni、`walk_forward_gate` 二项 z —— 全部正确。
- **安全（QA 实测未发现缺陷）**：租户隔离（跨用户 LLM 写 403）、admin 端点非管理员 403、伪造 sid 401、错误响应无堆栈/SQL/路径泄漏、静态目录穿越、并发幂等。
- **数据库**：迁移事务（`BEGIN IMMEDIATE` + rollback + SAVEPOINT）、`save_predictions` 租户作用域、thread-local 连接、`secretbox` Fernet 0600 —— 设计扎实。
- **数据完整性**：3518 期，无重复期号、无越界/重复红球、日期单调；仅 2003001–2003089 共 89 期销量/奖金字段为 0（历史缺失）。

---

## 6. 建议修复顺序

| 顺序 | 条目 | 理由 |
|---|---|---|
| 1 | **P0-3 CSRF**（1 行守卫） | 唯一可被远端直接触发的安全缺口，改动最小 |
| 2 | **P0-2 IDOR**（1 个函数） | 跨租户数据泄露，改动极小 |
| 3 | **P0-4 随机种子** | 回测判决可翻转，动摇整个"可信证据"体系 |
| 4 | **P0-1 六等奖**（1 行） | 修正 EV 口径，与 P1-5/6 一并处理 |
| 5 | **P0-5 compose 默认 enforce** | 防环境回退到匿名可写 |
| 6 | P1-5/6/7/8 | 口径统一 + 真值锁重构 + 输入校验 |
| 7 | P1-12 调度器断链 | 一行前置 `draws = db.load_draws()` |
| 8 | P1-1/2/3/4 | 安全纵深 |
| 9 | 测试工程（变异元测试、分层并行、定向用例） | 防止同类问题复发 |
| 10 | P2 全部 | 常态化清理 |

---

## 7. 一个值得记录的横向结论

5 条 P0 里有 **3 条**（六等奖、`sixth_ev_ratio`、依赖守卫缺口）的共性是：**同一个正确实现已经存在，但副本没有同步**。`blue_coverage.py` 修了 `n/16` 的 bug 却没同步 `engine_prize.py`；`evaluate.prize_level` 判对了却没人对拍 `engine_prize`。仓库的注释密度极高、自我说明很充分——**但"声称一致"从不被机器校验**。

因此最高杠杆的长期改法不是逐条修 bug，而是：**给每一对"应当一致"的实现加对拍测试**（奖级表 ×2、EV ×3、覆盖率口径 ×2）。这一条能同时防住本报告 P0-1、P1-5、P1-6 的复发。
