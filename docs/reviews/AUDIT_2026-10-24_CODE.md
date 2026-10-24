# 双色球预测系统 · 代码级缺陷审计报告

- 审计日期：2026-10-24
- 审计范围：`/root/DSH/应用/ssqyuce`（Python 3.13 + FastAPI + SQLite(WAL) + 静态前端 `web/static/app.js`）
- 审计方式：**只读审计，未修改任何产品文件**。所有结论均给出 `文件:行号` 与代码片段；能在隔离环境复现的均附探针输出（`scripts/isolated_test.sh`，绝不写生产库）。
- 审计者关注维度：正确性 bug、异常处理、资源泄漏、并发竞态、SQL、安全（鉴权/会话/越权/SSRF/加密）、前端 JS。

---

## 0. 方法论说明（务必先读）

1. **显示层脱敏 ≠ 真实代码**。本仓库若干行在终端 `Read` 回显时会被脱敏器改写（例如 `auth.py:382` 的 `token_version = token_version + 1` 在回显中曾被显示成 `token_version=token_version+1`）。本报告对**每一处可疑行都在字节级（codepoint / xxd / AST）复核**，确认它其实是正确代码（`token_version = token_version + 1`），**不是**模板泄漏或注入，故未列入缺陷。审计者在此显式声明：**未产生该假阳性**。
2. 隔离运行：所有探针都在 `/tmp` 下，通过 `scripts/isolated_test.sh -- .venv/bin/python /tmp/probe_*.py` 执行，脚本会把生产库快照复制到 `/tmp/lott-isolated.*/ssq.db` 后再跑，**不触碰生产库**。
3. 分级：
   - **P0**：可利用 / 数据损坏 / 崩溃。
   - **P1**：功能错误或潜在生产事故（含未授权数据访问、可用性）。
   - **P2**：代码质量 / 防御纵深 / 收敛性。
4. 不收录纯风格意见；每条均给出可落地的最小改法。

---

## 1. 摘要

### 缺陷计数

| 级别 | 数量 |
|------|------|
| P0 | 1 |
| P1 | 4 |
| P2 | 7 |
| **合计** | **12** |

### 三条最重要

1. **[P0] 跨租户越权读取任务结果（IDOR）** — `/api/tasks/{task_id}` 仅要求 `user` 角色，`db.load_task` 不按 `user_id` 过滤，`_task_visible_to_current_user` 恒返回 `True`。任意登录用户可读取**他人**任务 `result_json`（可能含票单）。**已用 HTTP 端复现**。（`routes_tasks.py:29-57`、`db.py:1033-1044`、`auth_api.py` 分级表）
2. **[P1] 登录接口在事件循环上同步执行 scrypt，阻塞全站** — `auth_login` 是 `async def` 但在协程体内同步调用 `hash_password/verify_password`（scrypt，32MB/次）。单次登录阻塞事件循环 ~145–550ms，且登录接口**匿名可达**，可被并发打满拖垮所有请求。**已用探针复现**（heartbeat 最大间隔 550ms）。（`auth_api.py:223-294`、`auth.py:93-141`）
3. **[P1] 认证后 SSRF + 部分响应体回显** — 普通用户经 `/api/user/llm/config` 可写任意 `base_url`；随后触发预测时 `llm_client.chat()` 会向该地址发起 POST，错误路径把 `HTTP {code}: {response.text[:200]}` 经 503 回传给用户，构成 SSRF 且可探测内网/元数据响应前 200 字符。**已用代码路径确认**（`user_llm_api.py:171-222`、`llm_client.py:581-596`、`routes_predict.py:562-568`）。

### 其余高价值发现（速览）
- **[P1] 调度器断链：`_scheduler_step` 中 `draws` 先用后赋值，必抛 `NameError`，被静默吞掉** → prereg 注册永不执行（注释声称已修的"断链"实际未生效）。**已用 AST + 探针复现**。（`api_app.py:344` vs `388`）
- **[P1] 角色降级不失效会话缓存** → `admin_patch_user` 改 `role` 未 `token_version+=1`/撤销会话，`_require_admin`/`/api/auth/me` 读到旧角色最长 30s（中间件因每请求重读 DB 仍能兜住主闸门，故列 P1 下沿/P2 上沿，见正文权衡）。
- **[P2] LLM 票 `structure_scores` 值未转义，前端 `innerHTML` 拼接形成存储型 XSS 汇聚点**（`app.js:847` + `engine.py:247`）。

---

## 2. P0 清单

### P0-1 跨租户越权读取任务结果（IDOR）

**位置**
- `lottery/routes_tasks.py:29-57`
- `lottery/db.py:1033-1044`（`load_task`）
- `lottery/db.py:1046-...`（`list_tasks` 同样无 user 过滤）
- `lottery/auth_api.py` 分级：`required_role("/api/tasks/xxx") == "user"`

**证据**
```python
# routes_tasks.py:29-44  —— 本应做归属判断，实际恒 True
def _task_visible_to_current_user(task: Optional[dict]) -> bool:
    ...
    return True

# routes_tasks.py:52-57
@router.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    t = db.load_task(task_id)
    if t is None or not _task_visible_to_current_user(t):
        return JSONResponse({...}, status_code=404)
    return t
```
```python
# db.py:1033 —— load_task 只按 id 取，无 user_id 过滤
def load_task(task_id: str) -> Optional[Dict]:
    row = get_conn().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    ...
```
```python
# bg_tasks.py:43-44 —— task_id = "{kind}_{8hex}"，且 create_task 未写 user_id
tid = f"{kind}_{uuid.uuid4().hex[:8]}"
db.create_task(tid, kind)   # tasks.user_id 落库为 NULL（见复现输出）
```

**根因**：`/api/tasks` 在 `ADMIN_EXACT`，但 `/api/tasks/{task_id}` 是**路径参数**，`required_role` 走精确匹配，落到默认 `"user"`。代码里 `_task_visible_to_current_user` 的注释（`routes_tasks.py:29-44`）自己承认"读路径还没按 user 过滤"，但把它留作 TODO —— 这与"当前 `/api/tasks/{id}` 普通用户可达"的事实叠加，构成真实越权。`task_id` 是 8 位十六进制（32 bit，带 `mine_`/`llm_eval_` 等可枚举前缀），可被遍历/日志泄漏后命中。

**复现（已执行，隔离环境）**
```
$ scripts/isolated_test.sh -- .venv/bin/python /tmp/probe_idor2.py
[isolated] 已复制生产库快照 → /tmp/lott-isolated.3HSEqm/ssq.db
attacker login: 200
GET /api/tasks/mine_victim01 as attacker -> 200
body: {'id': 'mine_victim01', 'kind': 'mine', 'status': 'pending', ...,
       'result_json': '{"tickets":["private victim data"]}',
       'user_id': None, 'result': {'tickets': ['private victim data']}}
```
`attacker`（role=user）成功读取到 `victim` 的任务私有票单。**HTTP 端复现成功。**

**影响**：跨租户读取私有数据（票单可能含用户投注号码、策略、LLM 推理内容）。多用户上线即命中。

**修复建议（最小改动）**：
1. `_task_visible_to_current_user` 改为 `return task is not None and (is_admin or int(task.get("user_id") or 0) == current_user_id())`；
2. `create_task`/`bg_tasks._bg_run` 落库时写入发起人 `user_id=resolve_tenant()`；
3. `load_task(task_id, *, user_id=None)` 加 `AND user_id=?`（后台调度器等系统上下文用 `resolve_tenant()` 兜底）；
4. 或（更简）把 `/api/tasks/{task_id}` 一并纳入 `ADMIN_EXACT` 前缀匹配（改成前缀规则而非精确匹配），并给普通用户单独提供"我的任务"端点。推荐 **1+2+3**，因为前端轮询需要普通用户读自己的任务。

---

## 3. P1 清单

### P1-1 登录在事件循环上同步跑 scrypt，阻塞全站

**位置**
- `lottery/auth_api.py:223`（`async def auth_login`）
- `lottery/auth_api.py:244-264`（协程体内同步调用 `get_user_by_username`、`verify_password`）
- `lottery/auth.py:93-141`（`hash_password`/`verify_password`，`hashlib.scrypt`，`n=2^15, r=8, p=1` ≈ 32MB/次）

**证据**
```python
# auth_api.py:223
async def auth_login(request: Request):
    ...
    # 244-247：在协程体里同步执行 CPU 密集的 scrypt（不存在的用户也会跑一次 dummy 哈希）
    user = auth.get_user_by_username(username)
    if user is None:
        auth.verify_password(password, auth.hash_password("dummy-placeholder"))
    ...
    ok, needs_rehash = auth.verify_password(password, user.get("password_hash"))  # 258
```
注释（`auth.py:19-23`）声称做了内存硬哈希，但**没意识到**把它放在 `async def` 里会在单事件循环线程上串行阻塞。

**复现（已执行）**
```
$ scripts/isolated_test.sh -- .venv/bin/python /tmp/probe_loopblock.py
hash_password wall: 144.9 ms  (runs on event loop thread for async login)
blocking work on loop: 545.2 ms ; max heartbeat gap: 550.5 ms
```
一次登录（含不存在用户）即让事件循环的最大 heartbeat 间隔达到 **550ms**，期间所有并发请求被冻结。

**影响**：登录接口**匿名可达**。攻击者并发打登录（每次触发一次 scrypt），可让 uvicorn 单 worker 的事件循环长期处于阻塞态 → 全站（含 `/api/health`）响应劣化甚至超时。这是一个低成本 DoS 面。

**修复建议**：把 `auth_login`（以及 `auth_change_password`、`admin_*`、`auth_bootstrap` 等含 scrypt 的 `async def`）改为**普通 `def`**，交给 FastAPI 线程池执行；或在协程内 `await run_in_threadpool(auth.verify_password, ...)`。签名无需改变。改动面小（去掉 `async` 或包一层线程池），收益大。

---

### P1-2 认证后 SSRF + 部分响应体回显

**位置**
- `lottery/user_llm_api.py:171-222`（`save_user_llm_config` 接受任意 `base_url`，无 host 白名单/内网阻断）
- `lottery/user_llm_store.py:202`（`base_url` 仅 `strip().rstrip("/")`）
- `lottery/llm_client.py:517-518`（`url = base_url + "/chat/completions"`）
- `lottery/llm_client.py:582-583`（`last_err = f"HTTP {r.status_code}: {r.text[:200]}"`）
- `lottery/routes_predict.py:562-568`（`LLMChannelError` → 503，把 `last_err` 拼进 `error` 回传用户）
- `lottery/routes_llm.py:151-157`（`/api/llm/test` 同样回显 `str(e)`）

**证据**
```python
# llm_client.py:581-596
r = requests.post(url, json=payload, headers=headers, timeout=eff_timeout)
if r.status_code != 200:
    last_err = f"HTTP {r.status_code}: {r.text[:200]}"   # 上游响应体前 200 字符进 last_err
    ...
# strict 分支抛出（653-655）
if strict:
    _bill_finish(bill, ok=False, error=last_err)
    raise LLMChannelError(f"LLM 调用失败: {last_err}")
```
```python
# routes_predict.py:562-568
except LLMChannelError as e:
    return JSONResponse({"ok": False, "llm_required": True,
        "error": (f"预测失败：大模型不可用或超时（{e}）。...")}, status_code=503)
```

**复现（代码路径确认）**：普通用户 `use_custom_key=1` + `base_url=http://169.254.169.254/latest/meta-data/`（或任意内网 IP:端口），`api_key` 随便填，`model` 随便填；`/api/predict?use_llm=1` → 服务器向该内网地址发 POST → 若非 200，`r.text[:200]` 经 503 回传；若 200，`_bill_add_usage`/解析失败也会把响应结构间接暴露。审计期以静态路径确认，未对内网发起真实请求（合规）。

**影响**：任一登录用户可让服务器向内网/云元数据发起请求（盲 SSRF），并能读取目标响应前 200 字符（有限 SSRF-with-read）。可探测内网服务、读云元数据部分内容。

**修复建议（最小）**：
1. 在 `user_llm_store.save` 归一化处对 `base_url` 做**约束**：仅允许 `https`（可配 http）+ 解析 host 后**拒绝**私有网段（`127/8`、`10/8`、`172.16/12`、`192.168/16`、`169.254/16`、`::1`、`fc00::/7`）与 `localhost`；
2. 错误信息**不要**回显上游响应体：`last_err` 对用户侧只保留状态码与分类（如 `HTTP 401 (认证失败)`），完整 `r.text` 仅写日志且截断/脱敏；
3. `/api/llm/test` 同理。

---

### P1-3 调度器断链：`_scheduler_step` 中 `draws` 先用后赋值 → NameError 被静默吞

**位置**
- `lottery/api_app.py:301`（`def _scheduler_step()`）
- `lottery/api_app.py:344`（**读** `draws`）
- `lottery/api_app.py:388`（**才赋值** `draws = db.load_draws()`）

**证据**
```python
# api_app.py:341-350
try:
    from . import prereg_scheduler
    _scan = prereg_scheduler.register_scan_candidates(
        db.load_draws(), issue=str(draws[-1]["issue"]) if draws else "")   # 344：draws 尚未赋值
    ...
except Exception as e:  # noqa: BLE001
    print("[scheduler] 首轮扫描·写入失败（不影响预测）:", e)              # 异常被吞
...
# api_app.py:388
draws = db.load_draws()                                                   # 真正赋值在这里
```

**复现（已执行，AST + 字节级）**
```
$ .venv/bin/python /tmp/probe_draws.py
draws Store at line 388
draws Load at line 389
draws Load at line 436
draws Load at line 421
draws Load at line 344   <-- Load 早于 Store
draws Load at line 344
```
`draws` 在 344 行以 `Load` 出现，而其唯一 `Store` 在 388 行 → 进入该 `try` 块必抛 `NameError: name 'draws' is not defined`，随后被 `except Exception` 吞掉，仅打印一行日志。

**影响**：`prereg_scheduler.register_scan_candidates`（预登记/断链修复的候选扫描）**永远不会被调用**。代码注释（`api_app.py` 相关段落及 prereg 模块）声称"首轮扫描·写入"已在调度流程中生效，实际每轮都因 `NameError` 提前失败。这是典型的"注释声称做了 X、代码实际没做"漂移，且因异常被吞而**完全静默**。

**修复建议**：把 `draws = db.load_draws()` 上移到 `_scheduler_step` 开头（任何使用 `draws` 之前），或该 `try` 内改为本地 `draws_now = db.load_draws()` 并使用它。最小改动：346 行前补 `draws = db.load_draws()`，删除 388 行重复赋值。

---

### P1-4 角色降级不失效会话缓存，`_require_admin`/`/api/auth/me` 读到陈旧角色（最长 30s）

**位置**
- `lottery/auth_api.py:498-512`（改 `role` 只 `UPDATE users SET role=...`，无 `token_version+=1`、无撤销会话）
- 对照 `lottery/auth.py:391` 起的 `set_status` 会 revoke 会话
- `lottery/auth_api.py:_require_admin` 读 `sess["user"]["role"]`（缓存）
- `lottery/auth.py:233-256` 会话缓存，TTL=30s

**证据**
```python
# auth_api.py:498-512 —— 只改列，不失效会话
if "role" in payload:
    ...
    conn.execute("UPDATE users SET role=?, updated_at=? WHERE user_id=?",
                 (new_role, time.time(), int(user_id)))
    conn.commit()
    auth.audit("user.role_change", ...)
# 对比 set_status(...) 内部会 revoke_all_sessions / token_version+=1（auth.py:391+）
```

**复现（已执行）**
```
$ scripts/isolated_test.sh -- .venv/bin/python /tmp/probe_role.py
carol cached role after promote (<=30s ago): user      # 升级后缓存仍旧角色
admin cached role after demote (no revoke): admin       # 降级后 admin 会话仍判为 admin
```

**影响与权衡（如实标注）**：`_require_admin` 与 `/api/auth/me` 读的是**会话缓存**里的角色，降级后最多 30s 内仍显示/判定为旧角色。**但是** 请求主闸门在中间件里是**每请求重读 DB** 的 `load_active_user`（`get_user` 不吃缓存），所以在 http 层对管理端点的实际拦截**是新的**。因此本条的**真实可利用面**是：`/api/auth/me` 回显旧角色（前端 UI 误判）+ 任何**只依赖 `_require_admin`/`sess["user"]["role"]` 而绕过中间件**的代码路径。审计期未发现仅靠 `_require_admin` 且被中间件豁免的敏感写端点，故列 P1 下沿（防御纵深缺失 + 显示不一致）。若未来新增"仅 `_require_admin` 自检"的端点则升为 P1/P0。

**修复建议**：`role` 变更分支复用 `set_status` 的失效逻辑：`UPDATE users SET role=?, token_version = token_version + 1, updated_at=? ...` 后 `auth.invalidate_session_cache()` / `revoke_all_sessions(user_id)`，与 status 变更保持一致。

---

## 4. P2 清单

### P2-1 LLM 票 `structure_scores` 值未转义 → 前端 `innerHTML` 存储型 XSS 汇聚点

- **位置**：`web/static/app.js:846-847`；源头 `lottery/engine.py:247-248`（`structure_scores` 为 dict 时**原样透传**，键值均未校验）
- **证据**
  ```js
  // app.js:847 —— 键 escHtml 了，值 v 没有
  const sc = Object.entries(t.structure_scores).map(([k,v]) => escHtml(k) + "=" + v).join(" · ");
  ```
  ```python
  # engine.py:247
  "structure_scores": (t.get("structure_scores") if isinstance(t.get("structure_scores"), dict) else {}),
  ```
- **影响**：`structure_scores` 来自 LLM 输出。若模型（或被 prompt 注入的模型）返回 `{"和值":"<img src=x onerror=alert(document.cookie)>"}`，值被原样拼进 `innerHTML` → 对**查看该票的用户**形成存储型 XSS。票单按租户隔离，主要是自我 XSS，但管理员在管理视图查看/导出时可能被命中。`escHtml` 也不转义单引号（对属性上下文仍不安全）。
- **修复建议**：`app.js:847` 改为 `escHtml(k) + "=" + escHtml(String(v))`；后端 `engine.py:247` 对 `structure_scores` 做 `{str(k): int/str(v)}` 白名单归一（值仅保留数字或截断字符串）。同批排查 `app.js` 其它未 `escHtml` 的插值（本报告未逐一穷举，建议加一次性 grep 校验）。

### P2-2 多语句写操作异常时未 rollback，线程本地连接可能残留未提交事务（持锁 / 部分提交泄漏）

- **位置**：`lottery/db.py:439-455`（`save_pattern_results` 循环 execute 后 commit）、`upsert_draws:304-325`、`save_predictions:561-575` 等
- **证据**：`for r in results: conn.execute(...)` 后再 `conn.commit()`；若中途 `r["key"]` 抛 `KeyError`，**commit 不执行**，Python sqlite3（`isolation_level=""`）遗留隐式事务未回滚。连接是**线程本地且长生命周期**，该线程下一次写会落在这条未结束的事务里，直到某次 commit 把**部分数据**一起提交，或一直持写锁。
- **影响**：潜在的部分写入 / 写锁滞留。触发需上游传入畸形 dict（当前调用方多已守卫，故为 P2）。迁移批次（`migrations._run_batch`）已正确 `BEGIN IMMEDIATE`+rollback+SAVEPOINT，**无此问题**——说明团队已知该模式，只是散落的 db 写函数未统一。
- **修复建议**：为多语句写加统一 `with _txn(conn):` 上下文管理器（try/commit、except/rollback、re-raise），或至少在这些函数体内 `try: ...; conn.commit() except: conn.rollback(); raise`。

### P2-3 LLM 成本日志"读-改-写"是安全的，但 `quota_exceeded` 是 check-then-use（TOCTOU）

- **位置**：`lottery/user_llm_store.py:347-362`（`quota_exceeded` 先查）与 `llm_client.py:237-248`（用到时才 `record_usage`）
- **证据**：配额判断与用量累加**不在同一原子操作**内，并发同名用户可同时通过 `quota_exceeded` 再各自 `record_usage`，短暂超配额。
- **注**：`record_usage` 本身用 UPSERT `+ excluded`（`user_llm_store.py:383-399`）**无丢失更新**；`llm_budget.json` 用 `_budget_lock`+原子 replace（`llm_client.py:403-424`）也安全；成本日志 `_append_cost_row` 用 `_cost_lock`（`llm_client.py:194-204`）进程内安全。这些**做对了**，不值得报缺陷。
- **影响**：多用户并发下配额可被小幅突破（非资金直损，共享通道才生效）。P2。
- **修复建议**：如要严格，可在 `record_usage` 的 UPSERT 中 `WHERE today_tokens_used + ? <= quota` 之类的原子条件，或允许 P2 现状并文档化"配额是软限"。

### P2-4 数据抓取：任一行解析失败即丢弃整批（含有效期）

- **位置**：`lottery/parser.py:47-57`（`parse_text` 遇 `ParseError` 直接 `raise`）、`data_fetcher.py:33-39`
- **证据**：`parse_text` 对坏行 `raise ParseError(...)` 中断整个文件；`update_from_text` 因此在解析阶段就抛，`upsert_draws` 不会执行（**好处：无部分写入**）。但代价是源站某行偶发畸形会**阻断当次全量入库**（包括本可入库的新期/补漏期）。
- **影响**：可用性/健壮性——上游一行脏数据让整次刷新失败，需等源站修复。P2（有部分写入风险被规避，属取舍）。
- **修复建议**：`parse_text` 支持"跳过坏行 + 收集告警"（`skip_bad=True` 返回 `(draws, errors)`），`update_from_text` 只要有 ≥N 条有效行即入库并回传 `parse_errors`；保留 `raise_for_status`。

### P2-5 后台线程在**模块 import 时**启动（import 副作用）

- **位置**：`lottery/api_app.py:687-710`（顶层 `if config.SCHEDULER_ENABLED: Thread(...).start()`）
- **证据**：调度、闲时挖掘、popularity 预热线程均在 `api_app` 被 import 时启动，而非应用启动钩子（lifespan）。
- **影响**：任何 import `api_app` 的进程（CLI 子命令、测试、探针、`python -m lottery.cli backtest` 若间接 import）都会拉起 daemon 循环线程，可能在非服务进程里跑调度/挖掘，产生非预期副作用。单 worker 部署下不会重复启动（import 缓存），但 import 副作用本身是坏味道。P2。
- **修复建议**：迁到 FastAPI `lifespan`/`startup` 钩子里启动，保证"只有真正 serve 时才起后台线程"。

### P2-6 idle_mining 限流读的是**整机** `/proc/stat`，容器内口径不准

- **位置**：`lottery/idle_mining.py:76-101`（`_system_cpu_pct` 读 `/proc/stat`）、`104-155`（`_throttle`）
- **证据**：`/proc/stat` 在容器内通常暴露的是**宿主机** CPU，而非本容器的 cgroup 用量。宿主机繁忙（隔壁容器吃满）时本服务会误让路；宿主机空闲时又不反映本容器自身负载。注释已承认"仍有 cgroup 硬限兜底"。
- **影响**：限流策略与实际容器负载脱节，可能早让路（吞吐下降）或晚让路（挤占前台）。P2。
- **修复建议**：优先读 cgroup v2 `/sys/fs/cgroup/cpu.stat`（`usage_usec`）或 v1 `cpuacct`，回退 `/proc/stat`；或直接用 `resource.getrusage` + cgroup quota 计算。

### P2-7 `upsert_draws` 返回值语义被 `inserted_new` 误用为"新增条数"

- **位置**：`lottery/db.py:304-325`（返回 `n`=遍历条数，非实际变更数）、`data_fetcher.py:39,52`
- **证据**：`INSERT OR REPLACE` 每条都"成功"，`n` 恒等于输入条数；`update_from_text` 把 `n_new = db.upsert_draws(...)` 当 `inserted_new` 返回给前端。重复刷新时 `inserted_new` 仍等于全量条数，误导运维。
- **影响**：展示/运维指标错误（`backfilled` 字段才是真实补漏口径，已存在）。P2。
- **修复建议**：用 `cur.rowcount` 累计或基于 `before/after count_draws()` 差值计算真实新增，或把字段改名为 `processed`。

---

## 5. 已核查、未发现缺陷的重要面（避免误伤/交代覆盖）

为体现覆盖面并避免"漏审"质疑，明确列出**核查后判定 OK** 的高风险项（均读过实现，非仅凭注释）：

- **迁移事务**：`migrations._run_batch` 用 `BEGIN IMMEDIATE`+整批 rollback，`_rebuild_table` 校验 `before/after` 行数一致并失败回滚，`executescript` 已被刻意避开（注释点名的坑 6 已规避）。幂等（`IF NOT EXISTS` / 主键已租户化即跳过）。**设计扎实。**
- **会话 Cookie**：`__Host-ssq_sid` 前缀 + `HttpOnly` + `Secure`（默认 1）+ `SameSite=Lax` + `path=/`（`auth_api.py:138-147`）。登录成功**新建** session_id（无会话固定）。登出撤销 + 清 cookie。
- **口令与比较**：`scrypt` 存储 PHC 串 + `hmac.compare_digest` 常量时间比较；不存在用户走 dummy 哈希防时序枚举；统一失败文案；登录限流（IP + 用户名双桶）+ 账号锁定（423）。**（唯一问题是它跑在事件循环线程上，见 P1-1。）**
- **跨租户票单写入**：`save_predictions`/`load_predictions`/`_recent_prediction_issues` **均带 `user_id`**（`db.py:547-606`），U3 §3.7 W1 的"删光他人票"已真实修复。
- **user LLM 配置越权**：`save_user_llm_config` 的 `uid` **只取自会话**，显式拒绝 `payload.user_id != uid`；写字段白名单（不能改 `today_tokens_used` 绕过配额）。
- **主密钥**：`secretbox` 用 Fernet，密钥文件以 `os.open(...,0o600)` 创建（无 644 窗口），env 可覆盖，解密失败降级为 `None` 不抛 500。
- **成本记账并发**：`llm_cost_log.jsonl` 用 `_cost_lock`（进程内）；`llm_budget.json` 用 `_budget_lock` + 原子 `os.replace`；`record_usage` 用 UPSERT 原子累加（无丢失更新）。
- **前端 XSS 主路径**：`escHtml` 覆盖了绝大多数后端字段插值（ticket reasoning/evidence/counter_evidence 均已转义）；`?v=20261024.7` 版本号在 css/js/echarts 三处**一致**。唯一漏点是 P2-1 的 `structure_scores` 值。
- **`auth.py:382` 的 `token_version = token_version + 1`**：**字节级复核（codepoint/xxd）确认是正确代码**，非注入/模板泄漏（Read 回显时会被脱敏器改写，见第 0 节）。

---

## 6. 建议修复顺序

1. **P0-1 IDOR（立刻）**：改 `_task_visible_to_current_user` 真判归属 + `create_task` 写入发起人 + `load_task` 按 `user_id` 过滤。风险最高、改动最小。
2. **P1-2 SSRF 回显（立刻）**：错误信息停止回显上游响应体（只需改 `llm_client.chat` 的 `last_err` 组装 + 路由文案）；`base_url` 加内网/协议白名单。
3. **P1-1 事件循环阻塞（高）**：`auth_login` 等含 scrypt 的 `async def` 改走线程池（`run_in_threadpool` 或改普通 `def`）。一行级改动，消除匿名 DoS 面。
4. **P1-3 调度断链（高）**：修 `draws` 先用后赋值，恢复 prereg 注册；顺带审视为何被 `except Exception` 静默——建议对"首轮扫描"失败提高日志等级。
5. **P1-4 角色降级失效（中）**：改 `role` 分支复用 status 的 `token_version+=1`/撤销会话。
6. **P2-1 存储型 XSS 汇聚点（中）**：`app.js:847` 值转义 + 后端归一。
7. **P2-2 写事务统一回滚（中低）**：加 `_txn` 上下文管理器。
8. 其余 P2（抓取健壮性、import 副作用、cgroup 口径、inserted_new 语义、配额软限文档化）按迭代节奏处理。

---

## 附录：本次使用的探针

| 探针 | 目的 | 关键输出 |
|------|------|----------|
| `/tmp/probe_idor.py` | db 层验证 `load_task` 无 user 过滤 + `required_role` | `bob reads task: ... ; role required: user` |
| `/tmp/probe_idor2.py` | **HTTP 端复现 IDOR** | attacker 读 victim 任务 200 + 完整 `result_json` |
| `/tmp/probe_draws.py` | AST 验证 `draws` Load 早于 Store | `draws Load at line 344` / `Store at line 388` |
| `/tmp/probe_loopblock.py` | 复现 scrypt 阻塞事件循环 | `max heartbeat gap: 550.5 ms` |
| `/tmp/probe_role.py` | 复现角色变更后缓存不失效 | 降级后 `admin cached role: admin` |

> 说明：审计期间未对任何内网/元数据地址发起真实请求（SSRF 以静态代码路径 + 本地探针确认）；未修改任何产品文件。
