# 双色球预测系统 · 架构层缺陷审计报告

- 审计对象：`/root/DSH/应用/ssqyuce`（APP_VERSION `4.15.0` / APP_BUILD `2026-10-V35`，Python 3.13 + FastAPI + SQLite + 静态前端，容器 `ssq-predictor`）
- 审计日期：2026-10-24
- 审计人：架构评审（只读，未修改任何产品文件）
- 姊妹报告：`docs/reviews/AUDIT_2026-10-24_CODE.md`（代码级：IDOR/scrypt 阻塞/SSRF/scheduler NameError/XSS）。**本报告不重复其条目**，凡重叠处以「见 CODE 审计」标注。

---

## 0. 方法论与已知结论的尊重

**只读方式**：用 `ast` 静态解析 `lottery/*.py` 88 个模块的 import 关系、`grep` 定位配置/开关/线程/连接管理、`docker stats` 取运行证据、对照 `docker-compose.yml` / `.env.example` / `.env`（仅看键名与非密值）/ README / DEPLOY。所有结论尽量给 `file:line`。

**已裁决、本报告不再重提的结论**（尊重既有评审）：

- **无循环依赖**（V14 §1.3 已用 AST 全扫描确认 0 个环）。本报告不再把"循环依赖"作为缺陷。
- **`engine.py` 是双上帝模块、`ml_model.py` 45% 是寄生统计实验、`battle.py` 函数粒度过粗**（V14 §1.2 已判定"`engine.py` 已无法维护"）。本报告只在其上补充"依赖方向守卫测试未覆盖的缺口"，不重复"该拆分"的泛论。
- **`routes_*` 横向隔离 100% 达成、纵向是"薄路由+深调用"**（V14 §1.4）。不重复。
- **`usable` 长期为 0 是数据现实不是缺陷**（V14 §结论 4）。不重复。
- **V4 起 `rigor.py` 的双向自证、`bh_p=None` 不误选**等统计口径纪律，视为已定。

**本次新增视角**：把审计重心放在 V14 未展开的四个面——(A) 依赖方向**守卫测试的真实覆盖边界**、(B) 全局状态与后台线程/连接的**并发安全**、(C) 配置/开关的**三方漂移**、(D) 路由层的**契约一致性**与**部署运行形态**。

---

## 1. 摘要

### 缺陷计数

| 级别 | 条数 | 主题分布 |
|---|---|---|
| **P0** | 2 | 鉴权默认三重关闭 + 无会话静默回退主管理员（off 模式）；Python 版本/部署形态漂移 |
| **P1** | 7 | 依赖方向守卫覆盖缺口；后台线程连接泄漏；配置三方漂移集群；`resolve_channels` 契约未守住 LLM 调用；README 默认值反向；重计算端点无并发保护；调度器与请求线程共享全局 |
| **P2** | 8 | 三/五份重复统计实现；错误返回四形态并存；死代码集群；docs 端点靠删路由保护；`.env.example` 缺安全段；等 |

> 说明：P0 的第 1 条在**当前这台机器**已被 `.env` 的 `LOTT_AUTH_MODE=enforce` 临时化解（见 P0-1 的"实际影响"），但**默认与文档路径**仍是危险值，且它是架构级"默认不安全"设计，故仍列 P0。

### 三条最重要的结论

1. **鉴权是"全局中间件 + 路径分级表"，不是路由级依赖；在默认 `AUTH_MODE=off` 下，无会话请求会被 `resolve_tenant` 静默归属到主管理员**（`api_app.py:191-197` + `db.py:539-541`）。当前机器靠 `.env` 显式 `enforce` 挡住，但 compose/`.env.example`/DEPLOY 都没有把 `enforce` 变成"上线必设"，一旦回到默认就是"匿名可写主管理员票单"。这是**架构层的默认不安全**，不同于 CODE 审计里已列的 IDOR（那条是登录后越权）。

2. **配置三方漂移已成集群**：`LOTT_SCHEDULER` compose 默认被反转成 `1`（代码默认 `0`，README 又写 `0`）；`LOTT_LLM_ONLY_OUTPUT`/`LOTT_N_TICKETS` README 写反（`1`/`10` vs 代码 `0`/`5`）；`.env.example` **完全没有** W08 鉴权段（`LOTT_AUTH_MODE`/`LOTT_MASTER_KEY`/`LOTT_BOOTSTRAP_TOKEN` 一个都没有）；`LOTT_BACKUP_DATA_URL`/`LOTT_RULE_APPLY`/`LOTT_PHASE_USABLE` example 有、compose 不透传。**这正是 V10/V12 反复踩过的"改 .env 不生效"坑的系统性复发**，且没有任何一致性测试守它。

3. **`tests/test_p2_dependency_direction.py` 只守住了"底层不反向 import `api_app`"这一条，没有守住跨层的其他方向**——例如 `evaluate.py → engine`（`evaluate.py` 被路由当"评估层"调用，却直接捅进集成引擎），以及 `routes_task` 可见性这类"架构约定 vs 实现"的缺口。守卫测试让人误以为"依赖方向已被守住"，实际只覆盖了一个特定坏味道。

---

## 2. P0 清单

### P0-1 鉴权默认三重关闭；无会话请求被静默解析为主管理员（off/灰度模式）

- **证据**：
  - 中间件在 `AUTH_MODE != enforce` 时"只解析身份、不拦截"，且**无会话时根本不写 contextvar**就放行：`lottery/api_app.py:190-197`
    ```python
    if not config.auth_enforced():
        if uid is not None:
            _current_user_id.set(uid)
        try:
            return await call_next(request)      # uid=None 也照样放行
        finally:
            _current_user_id.set(None)
    ```
  - `LOTT_TOKEN` 兼容层命中 Bearer 也不写身份：`api_app.py:201-202`。
  - 归属解析的兜底顺序是"显式 → contextvar → **主管理员** → SYSTEM"：`lottery/db.py:523-542`，其中 `db.py:539-541`
    ```python
    admin = _single_tenant_admin_id()
    if admin:
        return int(admin)          # 无会话 HTTP 请求 → 落到主管理员
    ```
  - `save_predictions` 用 `resolve_tenant(user_id)` 决定写谁：`db.py:557`，DELETE/INSERT 均带该 uid。
  - `/api/predict`、`/api/predictions/last`、`/api/predictions/history` **均未显式要求登录会话**，只靠 contextvar：`routes_predict.py:356-363`（签名无 `current_user` 依赖）、`routes_predict.py:502-504`（`request_context.current_user_id()` 可为 None）。
  - 默认值链路：`config.py:103` `AUTH_MODE="off"`；`docker-compose.yml:49` `LOTT_AUTH_MODE=${LOTT_AUTH_MODE:-off}`；`.env.example` **无此键**（见 P2-2）；DEPLOY 无"上线必设 enforce"清单。
  - 反证（同一坑曾被自家注释点名，但未被机制化）：`request_context.py:47-50` 明确要求"绝不回退到 admin"，`api_app.py:84-86` 也声称"丢失后会静默落到 SYSTEM 写错数据"——可 `resolve_tenant` 仍回退 admin。
- **实际影响**（如实标注）：**当前本机 `.env` 已设 `LOTT_AUTH_MODE=enforce`**（实测键值），因此运行中的 `ssq-predictor` 不受此条影响。但：①默认部署（按 `.env.example`/README 只填 LLM 三项）会回到 `off`；②`off` 是"前端登录页未落地"时的灰度值，一旦任何人为了排障临时改回 `off`，无会话的 `/api/predict?regenerate=1` 会 **DELETE 主管理员该期票单并重写**（票单只存一份，不可恢复——`db.py:552-555` 注释自述）；③`/api/llm/test` 在 uid=None 时回退管理员配置并 `with_secret=True` 用管理员 Key 发外部请求（`routes_llm.py:92-106`）。
- **修复建议（最小改动，符合仓库风格）**：
  1. 新增路由级依赖 `require_current_user()`（读 `request_context.current_user_id()`，None 时 401），挂到 `/api/predict`（写分支）、`/api/predictions/last|history`。
  2. 让 `resolve_tenant()` 的"主管理员回退"**只在无 HTTP 请求上下文**（调度器/CLI 显式场景）生效；HTTP 路径 uid=None 一律返回 `SYSTEM_USER_ID` 或直接 401。这与 `request_context.py:47-50` 的既有约定一致，不算新发明。
  3. `LOTT_AUTH_MODE != enforce` 时启动打印一次 WARN（"鉴权灰度中，全部 /api 匿名可写"）。
- **置信度**：高（代码路径完整可见）。属架构级"默认不安全"，与 CODE 审计 P0-1 IDOR 不同根（那条是登录后越权，本条是未登录回退）。

### P0-2 部署 Python 版本与项目声明漂移：镜像 `python:3.11-slim`，本地/项目为 3.13

- **证据**：
  - `Dockerfile:1` `FROM python:3.11-slim`。
  - 本机 venv 实测 `Python 3.13.5`；任务背景与 PROJECT_CONTEXT 口径为 Python 3.13。
  - `requirements.txt` 只写 `fastapi>=0.104` / `uvicorn>=0.24`，未 pin Python 兼容上限，也无 `requires-python`。
- **影响**：本地测试（3.13）通过 ≠ 容器内（3.11）通过。3.11→3.13 之间在 `sqlite3`、`statistics`、`typing` 的弃用/行为变化、`numpy/scipy` ABI、`contextvars` 复制语义等处都存在差异。当前测试全绿基于 3.13，容器跑 3.11 属"未在目标解释器上验证"。这是一个**静默的漂移**：Dockerfile 一旦被当作"生产真相"，本地开发永远看不到 3.11 的差异。
- **修复建议（最小）**：二选一——①`Dockerfile` 改 `python:3.13-slim` 与项目声明对齐；②若刻意锁 3.11，则在 README/DEPLOY 明示"生产=3.11、开发=3.13"，并在 CI 加 3.11 矩阵。推荐①。同时在 `requirements.txt`/`pyproject` 写 `requires-python`。
- **置信度**：高（版本号是硬事实）。是否已产生实际故障：**未验证**（需在 3.11 容器内跑全量 pytest 对照）。

---

## 3. P1 清单

### P1-1 依赖方向守卫测试只覆盖"底层不 import api_app"，其他跨层方向裸奔

- **证据**：`tests/test_p2_dependency_direction.py:57-73` 只禁止非 api_app 模块 `import api_app`；`test_route_modules_do_not_import_api_app:203-209` 只禁 `routes_* → api_app`。**没有任何断言禁止"低层/领域层 import 集成引擎或更高层"**。
  - 实测 AST：`evaluate.py → {engine, replay}`（`lottery/evaluate.py` 顶层 import；`evaluate.py:272-285` 还 lazy import replay）。`evaluate` 被 `routes_eval.py` 当作"评估层"调用，生产者-消费者方向本应是 `routes → evaluate → db`，但 `evaluate` 又直接捅进 `engine`（上帝模块）。
  - `engine` 的 fanin=14（`api_app, battle, blue_coverage, cli, evaluate, evolution, l2_service, llm_eval, replay, routes_capabilities, routes_llm, routes_portfolio, routes_predict, routes_tasks`）——V14 已记录 fanin，但**没有一个测试把这个"集成引擎被谁依赖"的方向钉住**。
- **影响**：V14 说"没有测试守着的重构会悄悄回滚"。当前守卫只守住一个坏味道（`api_app` 反向依赖），一旦有人新增 `db.py → engine` 或 `evaluate → battle`，测试照样全绿，分层继续烂。**给人"依赖方向已受控"的错觉**。
- **修复建议（最小）**：在 `test_p2_dependency_direction.py` 增加一张"层级表"（如 `L0=config/jsonutil/request_context/secretbox`，`L1=db/methods/stats`，`L2=engine/discovery/battle/replay/evaluate`，`L3=routes_*/api_app`），断言 `L(n)` 不得 import `L(>n)`，并对既有例外（`evaluate→engine`）显式白名单 + 注释"历史遗留，待拆分 engine 后消除"。纯静态测试，零产品改动。
- **置信度**：高（测试内容与 import 图均已实测）。

### P1-2 后台线程各自 `get_conn()` 建连接且从不关闭 → 连接/句柄随任务数累积

- **证据**：
  - `db.get_conn()` 是**每线程一条连接**、首次调用建连、只在该线程 `close()` 时释放：`db.py:247-266`；`db.close()` 只关当前线程：`db.py:269-286`。
  - `bg_tasks._bg_run` 每次任务 `threading.Thread(...).start()` 一个**新建 daemon 线程**（`bg_tasks.py:66-67`），线程里 `db.get_conn()`（`bg_tasks.py:44` 用 `conn = db.get_conn()`）→ 该线程**结束时既不 `db.close()` 也不归还连接**。线程对象被回收后，`threading.local()` 上的连接随之失去引用，但 **SQLite 文件句柄的显式 `conn.close()` 未被调用**，依赖 GC/`__del__` 兜底。
  - 同类：`api_app.py:689`（scheduler）、`api_app.py:697`（idle_mining）、`api_app.py:710`（popularity warmup）三个常驻线程各持一条永久连接（可接受，数量固定 3）；但 **每次 `_bg_run` 产生一次性线程 + 一次性连接**，是累积源。
  - 现有注释 `db.py:205-206` 只考虑到"线程数 ≤40"，**没考虑 bg 任务线程是一次性、可无限创建**的。
- **影响**：每触发一次 AI 闭环/LLM 工坊/离线评估（`_bg_run` 调用点：ML 诊断、AI 闭环、LLM 工坊、离线评估），就泄漏一条文件句柄直到 GC。长跑容器的 `SSH`/`fd` 计数缓增；WAL 模式下泄漏的读连接还会让 WAL 文件无法 checkpoint 回收（reader 存在时 checkpoint 受阻）。当前容器 `PIDS=3`（空闲），未观测到压力，但**这是随使用量线性增长的资源债**。
- **修复建议（最小）**：在 `_bg_run._run` 的 `finally` 里调用 `db.close()`（它只关当前线程连接，语义正好）；或在 `get_conn` 的每线程连接上注册 `weakref.finalize`。一行到三行级改动。常驻线程（scheduler/idle/warmup）无需改。
- **置信度**：中高（连接泄漏机制确定；实际 fd 增长**未验证**——建议 `grep -c "" /proc/<pid>/fd` 前后对比或压测确认）。

### P1-3 `resolve_channels()` 声称是出号通道唯一判据，但"是否调用 LLM"由另一套 `is_enabled` 决定

- **证据**：
  - `engine.predict_next` 里 LLM 是否被调用由 `use_llm = use_llm and METH.is_enabled("llm", methods_spec)` 决定：`engine.py:720`；随后 `if use_llm:`（`engine.py:799`）才真正调 `llm_tickets`。
  - `is_enabled` 的依据是 `implement_spec()`（纯 env 派生）：`methods.py:130-147`。
  - 而交付文档宣称"出号通道唯一判据是 `methods.resolve_channels()`"（PROJECT_CONTEXT §3 v3.0.0、compose:87-89 自述），`resolve_channels` 额外吸收 `llm_only_output`/`decision_disabled`/`methods_config.json`（`methods.py:551-664`）。**这两套对"LLM 是否参与"的解释不完全等价**：`is_enabled` 只看 allow/deny 列表，不看 `decision_disabled`（设置页"停用 LLM 决策"）与 `llm_only_output`。
  - 佐证：`routes_predict.py:455-458` 又单独读 `resolve_channels()["decision_disabled"]` 来把 `use_llm=False`，说明"停用决策"这一态**绕开了 engine 内的 `is_enabled`**，靠路由层补丁传递。
- **影响**：契约冻结的表述（"唯一判据"）与实现（两处判据 + 路由补丁）不一致。若未来新增一个"只调 `resolve_channels` 不调 `is_enabled`"的入口，`decision_disabled` 的 LLM 调用就会漏网——与 V10 修过的"报表把 LLM0 与 stat3 平均"同类的"看起来收敛、实际分叉"。
  - **注**：本轮**未复现**"LLM 族被关仍调 LLM"的 P0 级问题（配置调研 agent 曾提出，但我核到 `engine.py:720` 已有 `is_enabled` 门控，该 P0 **不成立**，已降级为本 P1 的"双判据分叉"）。故**不**把"LLM 被关仍调用"列为 P0。
- **修复建议（最小）**：`engine.predict_next` 顶部用一次 `_ch = METH.resolve_channels()`，令 `use_llm = use_llm and (_ch.get("families",{}).get("llm") is not False) and not _ch.get("decision_disabled")`，删掉 `routes_predict.py:455-458` 的补丁；补一条"`LOTT_LLM_DISABLED`/decision_disabled 下 `llm_tickets` 零调用"的测试。符合"resolve_channels 唯一判据"的既有裁决。
- **置信度**：中高（双判据事实确定；是否已在某个配置下真的多调/少调 LLM **未逐一穷举验证**）。

### P1-4 README 的 `LOTT_LLM_ONLY_OUTPUT` / `LOTT_N_TICKETS` / `LOTT_METHODS` 默认值与代码相反

- **证据**：
  - `README.md:445` 写 `LOTT_LLM_ONLY_OUTPUT` 默认 `1`、`:446` 写 `LOTT_N_TICKETS` 默认 `10`、`:706` 写"默认开"；对照代码 `config.py:231`（`"0"`）、`config.py:218`（`"5"`）、compose `:80`/`:92`、`.env.example:43-44` 均为 `5`/`0`。
  - `README.md:455` 写 `LOTT_METHODS` 空值="全部启用"；代码 `methods.py:363,580-581` 空值="stat,ml,uniform"（不含 llm）。
- **影响**：运维/读者按 README 理解会以为"LLM 产多少出多少、默认 10 注、空值全开"，实际是"统计主导、5 注、不含 LLM"。这正是 V10 契约冻结想消除的口径不一致，文档层又反弹。调试出号口径时极易误判。
- **修复建议（最小）**：README 三处改为 `0`/`5`/"空=stat,ml,uniform"，删除 `:706`"默认开"。纯文档改动。
- **置信度**：高（数值对照确定）。

### P1-5 `LOTT_SCHEDULER` compose 默认被反转（1），与代码默认（0）、README（0）三方互斥

- **证据**：`config.py:629` 默认 `"0"`；`docker-compose.yml:35` `LOTT_SCHEDULER=${LOTT_SCHEDULER:-1}`；`README.md:448` 写 `0`；本机 `.env` 实测 `LOTT_SCHEDULER=1`。
- **影响**：Docker 部署在无 `.env` 覆盖时**自动开启开奖日自动出号 + 严格模式调度链路**（`api_app.py:687`），与 compose 自己对 idle_mining 的原则（compose:36-38"新机制先观察再开，默认 0"）精神相悖。若同时 `STRICT_MODE=1`（本机即是），调度器会在开奖日强制走 LLM 出号，额度/失败面放大。文档读者以为默认关。
- **修复建议（最小）**：compose 改 `:-0`，或 README 改为"compose 默认 1"并说明。推荐前者（与"先观察再开"纪律一致）。
- **置信度**：高（三方值确定；实际是否开启取决于 .env，本机已开）。

### P1-6 无鉴权的重计算端点 + 无并发保护 + 单 worker：一个登录用户可拖垮全站

- **证据**：
  - 单 worker：`cli.py:93` `uvicorn.run("lottery.api_app:app", ...)` **无 `workers=`**（=1 进程）。同步 `def` 端点走默认 AnyIO 线程池；重计算端点全是同步 `def`。
  - 重型同步端点且对普通用户开放：`routes_rules.py:510-518` `/api/battle/report`（`limit=0` 默认全量 walk-forward）、`routes_rigor.py:89` `/api/rigor/randomness`（`limit=0` 全量）、`routes_rules.py:428` `/api/rules/runtime`、`routes_data.py:118` `/api/features`（每次 `db.load_draws()` 全表 + 全量计算，无 lru_cache）。
  - **无任何限流/并发信号量中间件**：`grep RateLimit|slowapi|semaphore lottery/api_app.py` 为空；`docker-compose.yml` 也无 nginx 层限流证据（DEPLOY 未见）。
- **影响**：单进程下 N 个并发重计算会占满线程池与 CPU（compose 硬限 2.4 核），前台探活在预热期已可能超时（PROJECT_CONTEXT §2 记录"预热期 healthcheck 短暂 unhealthy"）。这是一个**可用性放大面**：匿名（off 模式）或任意登录用户即可触发。
- **修复建议（最小）**：给上述端点加 **(draws 指纹 → 结果) 的进程内 LRU 缓存**（它们输入是全量开奖，指纹稳定，缓存命中率高），并对 `/api/battle/report` `limit=0`、`/api/rigor/randomness` `limit=0` 改为"admin-only 或走 `/api/tasks` 异步"。不改鉴权模型即可显著降面。
- **置信度**：中高（单 worker、无中间件、同步重计算均确定；"拖垮"程度依赖并发量与数据规模，当前 3490 期）。

### P1-7 Scheduler / idle_mining 后台线程与请求线程共享全局可变状态

- **证据**：常驻线程在 **import 期**启动（`api_app.py:687-710`），与请求线程共处同一进程；它们通过 `db`（每线程连接，安全）+ 若干**模块级全局**交互。`_bg_run` 显式用 `copy_context()` 修了 contextvar（`bg_tasks.py:66`，注释记载过"配额永不计数"事故），但常驻线程（`_scheduler_loop`/`_idle_mining_loop`）**没有显式身份 → 调 `resolve_tenant(None)` 会落到主管理员/SYSTEM**，其写入（如在线对照 `eval_results`、任务）归属到兜底身份。
  - 另：`_initialized` 是进程级全局（`db.py:212`），`db.close()` 会把它与当前线程连接一起重置（`db.py:278-286`）——单测里大量 `db.close()` 做隔离，若与后台线程并发（`_bg_run` 线程在跑），一个线程 `close()` 重置 `_initialized=False` 会让**别的**线程下次 `get_conn` 重新跑 `SCHEMA`+迁移（幂等但会抢 `_INIT_LOCK`）。生产不调 `close()`，故仅测试面风险。
- **影响**：调度器写的在线评估/任务归属到主管理员而非"系统"（数据语义轻微错位，与 CODE 审计的 IDOR 不同）；测试面存在"后台线程 + db.close() 竞态"导致偶发失败的理论风险。两者均非数据损坏级别。
- **修复建议（最小）**：常驻线程内部所有 `resolve_tenant` 调用显式传 `SYSTEM_USER_ID`（`db.resolve_tenant(SYSTEM_USER_ID)`），不依赖兜底；测试里避免在 `_bg_run` 运行期 `db.close()`。纯收窄。
- **置信度**：中（常驻线程身份兜底确定；测试竞态**未复现**，理论风险）。

---

## 4. P2 清单

### P2-1 三份 BH 校正 + 五份 Wilson 区间并行实现，真源与副本无一致性守卫

- **证据**：
  - BH：真源 `stats/rigor.py:150 benjamini_hochberg` / `:175 adjust_pvalues`；副本 `backtest.py:116 bh_adjust`（被 `mining.py:441` 调）、`llm_eval.py:72 _bh_adjust`（被 `llm_eval.py:223` 调）；族级另有 `regime.py:490 family_adjust`/`:511 family_adjust_locked`。
  - Wilson：`backtest.py:131 _wilson_ci`、`hypothesis.py:191 _wilson_lower`、`decision_skill.py:25 _wilson_lower`（注释自述"为不跨模块 import 私有函数"而复制）、`rule_lifecycle.py:864 wilson_interval`、`api_app.py:54` re-export。
- **影响**：真源若修 bug（排序/NaN/稳定性），副本不跟进即静默分叉；`stats/__init__` 自称"已迁入 stats 包"但实现四处散落。属可维护性债，非即时正确性 bug（当前数值应一致，`test_v10_cost.py:366` 只钉了其中两处）。
- **修复建议**：`backtest.bh_adjust`/`llm_eval._bh_adjust` 改调 `stats.rigor.benjamini_hochberg`（保留函数名做薄壳）；Wilson 上下界下沉到 `stats/`（纯数学）供各处调用。最小改动、符合仓库"stats 是引力中心"的 V14 §4.4 既有裁决。
- **置信度**：高。

### P2-2 `.env.example` 缺整段 W08/W01/P4 安全与关键变量（新人唯一入口是"开放鉴权"实例）

- **证据**：`.env.example` 全文**无** `LOTT_AUTH_MODE`、`LOTT_BOOTSTRAP_TOKEN`、`LOTT_MASTER_KEY`、`LOTT_COOKIE_NAME/SECURE`、`LOTT_SESSION_TTL`、`LOTT_CORS_ORIGINS`、`LOTT_API_DOCS`、`LOTT_IDLE_*`、`LOTT_ML_ENABLED`、`LOTT_BLUE_*`、`LOTT_COVERAGE_*`、`LOTT_LIFECYCLE_*`、`LOTT_RULE_APPLY`、`LOTT_PHASE_USABLE`（逐键 grep 确认）。
- **影响**：example 是唯一新人入口（`example:4` 自述）。按它建 `.env` → 得到开放鉴权 + 默认口径实例，且与 P0-1 叠加。属"compose 有、example 无"的单向漂移。
- **修复建议**：example 补 W08/W01/P4 三段（可全用注释行给默认+风险说明）；`test_w01_deploy_hardening.py` 增加"example 必须列出全部 compose 键"的守护。文档改动。
- **置信度**：高。

### P2-3 `LOTT_BACKUP_DATA_URL` / `LOTT_RULE_APPLY` / `LOTT_PHASE_USABLE` / `LOTT_SCRYPT_N` / `LOTT_POPULARITY_WARMUP` 在代码/example/README 出现，但 compose 不透传

- **证据**：`LOTT_BACKUP_DATA_URL`（`config.py:138`，`example:59`，README:454）— compose 无；`LOTT_RULE_APPLY`（`patterns_runtime.py:487`，README:49 "设 1 才重排"）— compose 无；`LOTT_PHASE_USABLE`（`config.py:409`）— compose 无；`LOTT_SCRYPT_N`（`auth.py:56` 注释"可一键降级"）— compose 无；`LOTT_POPULARITY_WARMUP`（`popularity.py:521`）— compose 无。
- **影响**：Docker 部署下这些旋钮**改 .env 不生效**——正是 V10 已修过的方法开关坑（compose:87-89 自认过）的同构复发。`LOTT_BACKUP_DATA_URL` 不生效 → `data_check.py` 恒 `not_configured`；`LOTT_RULE_APPLY` 不生效 → 文档承诺的"规律改号"在容器内永远打不开。
- **修复建议**：compose `environment` 补这 5 个键（默认取安全值）；新增"compose ↔ config 默认值一致性"测试（见 P2-4）。
- **置信度**：高。

### P2-4 无"compose / example / config 默认"一致性守卫测试

- **证据**：`tests/test_w01_deploy_hardening.py`（41-182 行）只测 docs/cors/sqlite，**不解析 docker-compose.yml 或 .env.example**。P1-4/P1-5/P2-3 的漂移全部来自"无守卫"。
- **影响**：每次改默认值靠人工同步，漂移必然复发（本次已抓出 Scheduler 反转、README 反向）。
- **修复建议**：新增测试解析 `docker-compose.yml` 的 `environment` 与 `config.py` 的常量默认，对"允许不同"的白名单外强制相等；同时断言"config 读取的每个 `LOTT_*` 要么在 compose，要么在显式白名单"。纯测试。
- **置信度**：高。

### P2-5 路由层错误返回四形态并存（含 HTTP 200 + `ok:false`，监控当成功）

- **证据**：
  - `HTTP 200 + {"ok": false}`：`routes_llm_ops.py:49,147,165,170,240,299`（如 `:240` `status_code=200`）、`routes_rules.py:411-413`（200）。
  - 真实 4xx/5xx + `{"ok": false}`：`routes_data.py:104`、`routes_predict.py:570`。
  - 裸业务对象无信封：`routes_data.py:46-62`、`routes_tasks.py:100`。
  - 细节 bug：`routes_rules.py:58-65` `_battle_or_503` 的字典里 `"detail"` 键**写了两次**（后者覆盖前者）。
- **影响**：前端只能靠 body 判断失败，监控/告警看不到 200 里的错误；上游缓存会把失败当成功。契约一致性差。
- **修复建议**：统一 `{"ok": bool, "error": str, "detail": ...}` + 真实状态码；删 `status_code=200` 的错误返回；修重复 `detail` 键。可分批做，先修重复 key（一行）。
- **置信度**：高。

### P2-6 `/docs` `/openapi.json` 靠"不注册路由"保护，一旦 `LOTT_API_DOCS=1` 即匿名暴露；nginx 无兜底

- **证据**：`api_app.py:108-110`（`docs_url/redoc_url/openapi_url` 依 `config.API_DOCS_ENABLED`）；`config.py:125` 默认 `0`；`docker-compose.yml:65` 默认 `0`。默认 → 404，安全。但保护模型是"删路由"而非"鉴权"：`LOTT_API_DOCS=1` 后三者在**非 enforce 模式下匿名 200**（中间件 `need="user"` 但非 enforce 直接放行，`api_app.py:191-196`）。`DEPLOY.md:311` nginx `location /` 未拦截 `/docs`。
- **影响**：当前默认部署安全，但开调试/误设即全量 schema（含所有路径与模型）暴露，与 AUTH_MODE 无关。
- **修复建议**：nginx 显式 `location ~ ^/(docs|redoc|openapi.json) { return 404; }`；或把文档挂到受保护子路径。DEPLOY 加一行。
- **置信度**：高（代码事实）；本机当前 `LOTT_API_DOCS` 未设 → 安全。

### P2-7 被移除页面（V29/V34/V35）的前端渲染链整块残留为死代码

- **证据**（app.js 5140 行）：
  - `renderPatterns`/`filterPatterns` 在 V29 后永远提前 `return`，其后 echarts 绘制块为死代码：`app.js:1894-1959`（`loadAll:4607` 仍每首屏空调用一次）。
  - 数据分析整链 `setWin/setHeatMode/setBHeatMode/renderWindow/renderOmitScatter/renderOmitBins/renderBlueTrend`：`app.js:1470-1680`，对应 HTML 容器已不存在。
  - 回放整链 `populateReplaySelect/loadReplay/renderReplayItem`：`app.js:626-664`，无调用者，`replayData` 状态变量遗留。
- **影响**：维护噪声；`test_v29_removed_pages.py:56-59` 反而**要求**保留 `return;` 保护，形成"为测试留死代码"的负向激励。
- **修复建议**：删渲染块只留 `allPatterns = items;` 等数据赋值；同步更新 v29 测试快照。前端改动需同步升 `?v=`（DEPLOY 纪律）。
- **置信度**：高。

### P2-8 后端 39 个端点前端 0 调用（含历史别名未标注），其中若干已删页面端点仍在同步重算

- **证据**：静态比对 app.js 的 `/api/...` 与后端注册，后端 39 个路由无前端调用（如 `/api/config/llm` `routes_llm.py:36/43` 被 `/api/user/llm/config` 取代；`/api/portfolio/capabilities` `routes_portfolio.py:696` 与 `/api/capabilities` 并存；`/api/data/reconcile*` `routes_data.py:25/31` 无 UI 无测试）。V34/V35 **有意保留**了 `/api/rigor/*`、`/api/refresh` 作审计 API（测试 `test_v34:74`、`test_v35:69` 钉住必须仍在）。
- **影响**：缺"谁在用"的判断依据，删除/合并时易误伤；审计 API 与残留 API 混在一起。
- **修复建议**：给路由加 `# audit-only` / `# legacy-alias: replaced by ...` / `# removed-page` 注释标签，由 `routes_capabilities.py` 汇总；不删（尊重 V34/V35 裁决）。
- **置信度**：高（静态比对）。

---

## 5. 建议的修复顺序（按 ROI 与风险）

| 顺序 | 条目 | 理由 | 改动面 |
|---|---|---|---|
| 1 | **P0-1 鉴权默认 + resolve_tenant 回退** | 架构级默认不安全，一旦回灰度即"匿名写主管理员票"；且与 CODE/SSRF 无关，是独立入口 | 2-3 个路由加依赖 + `resolve_tenant` 条件收窄 + 启动 WARN |
| 2 | **P1-5 + P1-4 + P2-3 + P2-4 配置漂移集群** | 全是"改 .env 不生效/文档反了"，一次 compose+README+example 补齐 + 加一致性测试全部锁死 | 文档/compose + 1 个新测试，零逻辑风险 |
| 3 | **P0-2 Dockerfile Python 版本** | 一行改 `3.13-slim`（或显式声明），消除"本地绿≠容器绿"的静默漂移 | 1 行 |
| 4 | **P1-1 依赖方向层级表测试** | 纯静态测试，钉住 `evaluate→engine` 这类缺口，防回滚 | 1 个测试文件 |
| 5 | **P1-2 bg 线程连接泄漏** | `finally: db.close()`，防长跑 fd/WAL 累积 | 1-3 行 |
| 6 | **P1-3 resolve_channels 收敛 LLM 调用判据** | 兑现"唯一判据"契约，删路由补丁 | engine 顶部 + 1 测试 |
| 7 | **P1-6 重计算端点缓存/admin 化** | 单 worker 可用性加固 | 各端点加 LRU / limit=0 转异步 |
| 8 | P2-5/6/7/8 契约与死代码清理 | 可维护性，分批做 | 分散、低风险 |
| — | **P1-7** 后台线程身份显式传 SYSTEM | 语义收窄，与 6 同批 | 小 |
| — | 与 **CODE 审计** 的 IDOR(P0)/SSRF(P1)/scrypt 阻塞(P1)/scheduler NameError(P1)/XSS(P2) 已另行跟踪，不在本报告排序内 | 避免重复 | — |

---

## 6. 覆盖声明与未验证项

**已核查未发现架构级缺陷的面**（避免误伤/交代覆盖）：

- **SQLite 连接安全**：`_CONN` 单连接旧实现确已修为每线程连接 + WAL + `busy_timeout` + `synchronous=NORMAL`（`db.py:175-266`），且单 worker（`cli.py:93` 无 `workers=`）使"进程内多线程共享一连接"的自锁问题不再成立。这部分是**做对了的**，不列缺陷。
- **contextvar 跨后台线程**：`_bg_run` 已用 `copy_context().run` 修好（`bg_tasks.py:16-19,66`），历史事故已闭环。
- **循环依赖**：AST 全扫 0 个（复现 V14）。
- **resolve_channels 对"最终是否允许 LLM 注出现"的收敛**：`engine.py:889-895` 确实在最终选取阶段守住了（V14 未覆盖这点，本报告确认它成立）。

**明确标注"未验证"的项**：

- P0-2（3.11 容器内是否真有运行时差异）——需在 `python:3.11` 镜像跑全量 pytest 对照；本报告只认定"版本漂移"这一事实。
- P1-2（fd/WAL 泄漏的实际速率）——需 `/proc/<pid>/fd` 前后对比或压测。
- P1-3（是否在某个具体配置下真的多调/少调 LLM）——需按 `LOTT_LLM_DISABLED`/`decision_disabled`/`LOTT_METHODS=-llm` 组合矩阵实测调用次数。
- P1-7（测试面 `db.close()` 竞态）——理论风险，未复现。

---

*本报告仅审计，未修改任何产品文件。所有 file:line 基于 APP_BUILD `2026-10-V35` / HEAD 当前工作树。*
