"""SQLite schema migrations for M4.5（V10-00d 加固：批次事务 + 迁移前自动备份）.

Migrations are ordered, idempotent, and recorded in schema_version. Existing
installations are safe because each migration checks its target objects.

V10-00d 加固点（对应 ``docs/reviews/R4_RELEASE_REVIEW.md`` P0-C 第 1 条
「迁移是懒加载自动执行的，无回滚路径」）：

1. **批次事务**：所有待执行迁移在**同一个事务**里跑完；任一迁移抛异常 →
   整体 ``ROLLBACK``，``schema_version`` 一行都不会前进（不会留下"跑到一半"的
   半迁移状态），异常原样抛出。
   * 注意 SQLite/Python 的坑：``sqlite3`` 的 LEGACY 隔离级别（默认
     ``isolation_level=""``）对 **DDL 不开启隐式事务**，``CREATE TABLE`` 会立即
     自带提交，事后 ``ROLLBACK`` 撤不回来。因此这里显式在 ``isolation_level=None``
     （autocommit）下 ``BEGIN IMMEDIATE``，保证 DDL 也在同一事务内。
   * 若调用方**已有未提交事务**，改用 ``SAVEPOINT`` 嵌套：失败只回滚到迁移开始处，
     **绝不提交**调用方未完成的工作。
2. **迁移前自动备份**：真正有待执行迁移时，先用 ``sqlite3.Connection.backup()``
   （一致性热备，不是 ``cp``）备份到**与被迁移库同目录**的
   ``<db 文件名>.pre-migrate-v<当前版本>``（例如
   ``data/ssq.db.pre-migrate-v2``）。备份失败 → **中止迁移**（fail-closed）。
   全新库（``schema_version`` 还是 0、没有既有数据）不做无意义备份。
3. ``status()`` 语义不变，供 ``/api/health`` 暴露 ``schema_version``。
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


def _ensure_reconcile_runs(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS reconcile_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        checked_at REAL NOT NULL,
        secondary_url TEXT DEFAULT '',
        status TEXT NOT NULL,
        summary_json TEXT NOT NULL DEFAULT '{}'
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reconcile_runs_checked_at ON reconcile_runs(checked_at)")


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    for table, column, definition in (
        ("predictions", "evidence_json", "TEXT DEFAULT ''"),
        ("draws", "source", "TEXT NOT NULL DEFAULT '17500'"),
    ):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not columns:
            continue
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_replay_results(conn: sqlite3.Connection) -> None:
    """V10-05：离线回放分表（``docs/V10_IMPROVEMENT_PLAN.md`` §2.1.5 冻结 DDL）。

    要点（S2 B4）：
      * 主键是 ``(method, mode, issue)``——**必须含 mode**。同一 method 会同时以
        ``mode='replay'``（离线回放）与 ``mode='research_arm'``（LLM 研究对照臂）
        写入；若主键只有 ``(method, issue)``，``INSERT OR REPLACE`` 会让两者互相
        覆盖，直接毁掉对方那行。
      * ``issue`` 用 **TEXT**，与 ``draws.issue`` / ``eval_details.issue`` 一致
        （跨表按 issue 关联时不会因类型亲和性产生 ``0001`` vs ``1`` 的错配）。
      * forward-only：只有 ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX``，
        无 DROP / RENAME / 改类型（``db.get_conn()`` 首次连库即自动落地，
        容器启动后没有回滚路径）。
      * 本表与真实交战样本 ``eval_details`` **物理分表**：两套样本口径/样本量不同，
        **禁止相加**（见 ``lottery/replay.py: samples_side_by_side``）。
    """
    conn.execute("""CREATE TABLE IF NOT EXISTS replay_results (
        method        TEXT    NOT NULL,
        mode          TEXT    NOT NULL DEFAULT 'replay',
        issue         TEXT    NOT NULL,
        n_tickets     INTEGER NOT NULL DEFAULT 0,
        red_hits      INTEGER NOT NULL DEFAULT 0,
        best_red_hit  INTEGER NOT NULL DEFAULT 0,
        blue_hit      INTEGER NOT NULL DEFAULT 0,
        prize         INTEGER NOT NULL DEFAULT 0,
        channels_fp   TEXT    NOT NULL DEFAULT '',
        created_at    REAL    NOT NULL,
        PRIMARY KEY (method, mode, issue)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_issue ON replay_results(issue)")


def _ensure_llm_audit(conn: sqlite3.Connection) -> None:
    """V12-13：LLM 两岗位（核对员 / 契约检查器）的判定记录表。

    背景：V10 §2.1.5 的冻结 DDL 里本来就有 ``llm_audit``，但 V10-01 实际**从未落地**
    （``lottery/ai_audit.py`` 不存在、全仓 ``grep llm_audit`` 零命中），所以第 3 号迁移
    只建了 ``replay_results``。V12-13 补上这张表，按 V10 §2.1.5 原文的冻结 DDL 建，
    **字段一字不改**——已有的文档契约不该因为「晚建了一个版本」而漂移。

    写语义（V10 §2.1.5 冻结）：``INSERT OR IGNORE``，**仅追加**；判定历史不可被覆盖。
    ``audit_id = sha1(post|run_id|ts_ns)`` 保证重试幂等。

    forward-only：只有 ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX``，
    无 DROP / RENAME / 改类型（``db.get_conn()`` 首次连库即自动落地，无回滚路径）。
    """
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_audit (
        audit_id     TEXT PRIMARY KEY,
        run_id       TEXT NOT NULL,
        post         TEXT NOT NULL,
        model        TEXT NOT NULL DEFAULT '',
        input_json   TEXT NOT NULL DEFAULT '',
        output_json  TEXT NOT NULL DEFAULT '',
        verdict      TEXT NOT NULL,
        fail_fields  TEXT NOT NULL DEFAULT '[]',
        ts           REAL NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_audit_post ON llm_audit(post, ts)")


def _ensure_identity_tables(conn: sqlite3.Connection) -> None:
    """MU-1 / W04：身份三表 + bootstrap 主管理员行。

    依据 `docs/reviews/U1_IDENTITY.md` §4.1/§4.2/§4.3 的冻结 DDL。

    ## 为什么 `users.status` 的 CHECK 必须有**四个**值

    `bootstrap` 少写一个就会让本迁移的 INSERT 被 CHECK 拒绝 → 整批事务回滚
    → **上线失败**。这是 Lead 在评审阶段裁决过的硬冲突（U1 原本只写了三个值）。

    ## 为什么 bootstrap 行是 `role='user'`

    U1 §4.1 的建议：插入时用 `role='user'` + `is_primary_admin=0`，
    只有"设密成功"的同一事务里才升为 `role='admin'` + `is_primary_admin=1`
    + `status='active'`。这样即使将来中间件的 status 闸门写漏，role 也不会额外授权
    （纵深防御）。反过来若插入即 `role='admin'`，一旦 status 闸门失效就是一个
    **零密码管理员**。

    ## 为什么 `username` 保留表内 UNIQUE + 软删除时改名

    表内 UNIQUE 是 Lead 冻结的约定（`username TEXT UNIQUE NOT NULL`）。
    但 UNIQUE 会让"删除后重建同名用户"永久失败，所以软删除时把用户名改写为
    ``name#deleted#<user_id>`` 释放原名（见 `auth.soft_delete_user`）。

    forward-only：仅 CREATE TABLE/INDEX 与一条幂等 INSERT，无 DROP/RENAME。
    """
    # ---- users（U1 §4.1）----
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username        TEXT    NOT NULL UNIQUE,
        password_hash   TEXT,
        role            TEXT    NOT NULL DEFAULT 'user'
                                CHECK(role IN ('admin','user')),
        status          TEXT    NOT NULL DEFAULT 'active'
                                CHECK(status IN ('bootstrap','active','disabled','locked')),
        display_name    TEXT    NOT NULL DEFAULT '',
        is_primary_admin INTEGER NOT NULL DEFAULT 0,
        failed_count    INTEGER NOT NULL DEFAULT 0,
        locked_until    REAL    NOT NULL DEFAULT 0,
        last_login_at   REAL,
        token_version   INTEGER NOT NULL DEFAULT 1,
        llm_profile     TEXT    NOT NULL DEFAULT 'default',
        external_id     TEXT,
        deleted_at      REAL,
        created_at      REAL    NOT NULL,
        updated_at      REAL    NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users(status, deleted_at)")
    # 主管理员唯一性锚点：全表最多一行 is_primary_admin=1（Lead 裁决 2）
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_primary_admin"
        " ON users(is_primary_admin) WHERE is_primary_admin = 1")

    # ---- user_sessions（U1 §4.2）----
    conn.execute("""CREATE TABLE IF NOT EXISTS user_sessions (
        session_id   TEXT    PRIMARY KEY,
        user_id      INTEGER NOT NULL,
        created_at   REAL    NOT NULL,
        expires_at   REAL    NOT NULL,
        last_seen_at REAL    NOT NULL,
        ip           TEXT    NOT NULL DEFAULT '',
        user_agent   TEXT    NOT NULL DEFAULT '',
        csrf_token   TEXT    NOT NULL DEFAULT '',
        revoked_at   REAL
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user"
        " ON user_sessions(user_id, revoked_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_exp ON user_sessions(expires_at)")

    # ---- user_audit_log（U1 §4.3）----
    conn.execute("""CREATE TABLE IF NOT EXISTS user_audit_log (
        audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        actor_id    INTEGER,
        actor_name  TEXT    NOT NULL DEFAULT '',
        action      TEXT    NOT NULL,
        target_id   INTEGER,
        target_name TEXT    NOT NULL DEFAULT '',
        result      TEXT    NOT NULL DEFAULT 'ok',
        ip          TEXT    NOT NULL DEFAULT '',
        detail_json TEXT    NOT NULL DEFAULT '{}'
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_uaudit_ts ON user_audit_log(ts DESC)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_uaudit_act ON user_audit_log(action, ts DESC)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_uaudit_tgt ON user_audit_log(target_id, ts DESC)")

    # ---- bootstrap 主管理员行（幂等：仅当表内一个存活用户都没有时插入）----
    #
    # `role='user'` + `is_primary_admin=0` + `status='bootstrap'`：零权限占位，
    # 等引导流程设密成功后在同一事务里升级（U1 §4.1 不变量 2）。
    # 不加 `role='admin'`，否则"零密码 + admin"这一行就是提权漏洞。
    now = time.time()
    conn.execute(
        "INSERT INTO users(username, password_hash, role, status, display_name,"
        " is_primary_admin, token_version, created_at, updated_at)"
        " SELECT 'admin', NULL, 'user', 'bootstrap', '系统管理员', 0, 1, ?, ?"
        " WHERE NOT EXISTS (SELECT 1 FROM users WHERE deleted_at IS NULL)",
        (now, now),
    )



# ---------------------------------------------------------------------------
# MU-2：数据租户化（迁移 6 / 7 / 8）
# ---------------------------------------------------------------------------
#
# 编号说明（以**实际上号的迁移**为准，不照抄文档编号）
# --------------------------------------------------
# `docs/reviews/U3_DATA_MODEL.md` §7.1 给的是「5=users、6=sessions、7=llm_config、
# 8=租户化」，但 MU-1 落地时把身份**三表合进了 5 号**（users + user_sessions +
# user_audit_log），于是实际占号变成：5=身份三表、6/7 空号、8=租户化。
# 迁移号是**全局串行**的，占号必须以代码里真实的 MIGRATIONS 表为准 ——
# 照抄文档会造成"以为还有 6/7 可用、结果撞号"。
#
# 因此本文只用 **6 / 7 / 8** 三个号：
#   6 = user_llm_config（U2 冻结 DDL）
#   7 = issue_baseline（U3 §4.6 的 L0 基线轨道；空表，接口先冻结）
#   8 = 既有表租户化（predictions + eval_* 重建主键 / tasks、llm_* 加列）

# 系统主体哨兵：与 config.SYSTEM_USER_ID 一致，**不插入 users 表**。
# 语义 = "系统/历史"，不是任何具体的人。
_SYSTEM_USER_ID = 0


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.OperationalError:
        return []


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _pk_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """按主键顺序返回列名（`PRAGMA table_info` 的 pk 字段是 1-based 序号）。"""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    pk = [(r[5], r[1]) for r in rows if r[5]]
    return [name for _, name in sorted(pk)]


def _require_admin_id(conn: sqlite3.Connection) -> int:
    """**动态**解析主管理员 id（Lead 裁决 1：绝不假设 admin == user_id 1）。

    为什么不能写死 1：SQLite `AUTOINCREMENT` **不复用 ID**。实测「清空 users
    再重新插入」拿到的是 2。若历史 40 行被回填到字面量 1，而 1 已不存在（或更糟：
    被一个普通用户占用），表现为"数据凭空消失"或"越权可见"。

    解析顺序：`is_primary_admin=1` → `role='admin' AND status='active'`。
    两者都解析不到时**不臆造**，返回 SYSTEM_USER_ID 并告警——这样历史数据仍在
    （归属"系统/历史"），而不是挂到一个错误的人头上。
    """
    row = conn.execute(
        "SELECT user_id FROM users WHERE is_primary_admin=1 AND deleted_at IS NULL"
        " ORDER BY user_id LIMIT 1").fetchone()
    if row:
        return int(row[0])
    row = conn.execute(
        "SELECT user_id FROM users WHERE role='admin' AND status='active'"
        " AND deleted_at IS NULL ORDER BY user_id LIMIT 1").fetchone()
    if row:
        return int(row[0])
    _log("未解析到可用管理员，历史行归 SYSTEM_USER_ID=0（不臆造归属）")
    return _SYSTEM_USER_ID


def _rebuild_table(conn: sqlite3.Connection, table: str, ddl: str,
                   columns: Sequence[str], *, default_user_id: int,
                   has_user_col: bool) -> None:
    """SQLite 通用重建：建新表 → 拷数据 → 删旧表 → 改名 → 重建索引。

    SQLite **不支持** `ALTER TABLE ... MODIFY PRIMARY KEY`，改主键只能重建
    （U3 §3.3）。整个过程在迁移批次的**同一个事务**里，失败即整批回滚。

    为什么不用 `executescript`：Python ``sqlite3.executescript`` 会**先提交挂起事务**，
    在迁移事务里用它 = 把"全有或全无"打散成"跑到一半"（U3 §3.3 坑 6）。
    """
    if not _table_exists(conn, table):
        # 全新库：SCHEMA 已建目标结构，这里只需兜底建表（IF NOT EXISTS）
        conn.execute(ddl)
        return
    cols = _table_columns(conn, table)
    if has_user_col and "user_id" in _pk_columns(conn, table):
        return                              # 已租户化（幂等）
    before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    conn.execute(f"DROP TABLE IF EXISTS {table}__new")
    conn.execute(ddl.replace(f"CREATE TABLE IF NOT EXISTS {table}",
                             f"CREATE TABLE {table}__new", 1))
    select_cols = ", ".join(columns)
    if has_user_col:
        # 用**参数**写回填值，不拼字面量（避免任何"user_id = 1"式的硬编码）。
        # 注意：SELECT 后面必须是列名，不能是占位符 —— 占位符只能在 VALUES 里用。
        conn.execute(
            f"INSERT INTO {table}__new (user_id, {select_cols})"
            f" SELECT ?, {select_cols} FROM {table}",
            (int(default_user_id),))
    else:
        conn.execute(
            f"INSERT INTO {table}__new ({select_cols}) SELECT {select_cols} FROM {table}")
    conn.execute(f"DROP TABLE {table}")
    conn.execute(f"ALTER TABLE {table}__new RENAME TO {table}")

    after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if after != before:
        raise RuntimeError(f"迁移重建 {table} 行数不一致：{before} → {after}（整批回滚）")
    _log(f"{table} 已租户化（{after} 行，user_id 默认 {default_user_id}）")


def _ensure_user_llm_config(conn: sqlite3.Connection) -> None:
    """迁移 6：`user_llm_config`（U2 §3.4 冻结 DDL，按 Lead 裁决去掉物理外键）。

    为什么去掉 `REFERENCES users(user_id) ON DELETE CASCADE`：
    本库**未启用** `PRAGMA foreign_keys=ON`（`db.py` 从未设置），写了也不生效。
    "写了不生效"比"不写"更危险——后续看代码的人会以为级联删除受保护。

    为什么 `api_key_encrypted` 而不是明文：密钥必须可逆取回（要发请求），
    所以不能口令哈希；用 Fernet（AES-128-CBC + HMAC）加密，主密钥来自 env 或
    `data/.secret_key`（0600）。
    """
    conn.execute("""CREATE TABLE IF NOT EXISTS user_llm_config (
        user_id              INTEGER PRIMARY KEY,
        use_custom_key       INTEGER NOT NULL DEFAULT 0 CHECK(use_custom_key IN (0,1)),
        disabled             INTEGER NOT NULL DEFAULT 0 CHECK(disabled IN (0,1)),
        base_url             TEXT,
        api_key_encrypted    TEXT,
        api_key_masked       TEXT,
        model                TEXT,
        model_list_json      TEXT NOT NULL DEFAULT '[]',
        samples              INTEGER NOT NULL DEFAULT 3 CHECK(samples BETWEEN 1 AND 10),
        extra_body_json      TEXT NOT NULL DEFAULT '{}',
        daily_token_quota    INTEGER NOT NULL DEFAULT 50000,
        today_tokens_used    INTEGER NOT NULL DEFAULT 0,
        today_calls_used     INTEGER NOT NULL DEFAULT 0,
        last_reset_date      TEXT NOT NULL DEFAULT '',
        qps_limit            INTEGER NOT NULL DEFAULT 2,
        updated_at           REAL NOT NULL
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ulc_updated ON user_llm_config(updated_at)")


def _ensure_issue_baseline(conn: sqlite3.Connection) -> None:
    """迁移 7：`issue_baseline` —— **V28 起不再建表**（保留函数体为 no-op）。

    ## 为什么删掉这张表

    它由 U3 §4.6 规划而来，目的是缓存「L0 事实层」基线（同一期所有人看到同一条
    baseline）。但**写入方从未实现**：全仓库没有任何代码往它写数据，
    跑了 8 个版本的迁移后仍是 0 行，也没有任何端点读它。

    一张永远为空的表比"没这张表"更糟：它会让后续维护者以为有这套机制在跑，
    从而在排查「为什么基线不一致」时找错方向。

    ## 为什么保留这个函数（而不是整段删掉）

    `MIGRATIONS` 列表按序号升级，历史上已经到过 7。若把 (7, ...) 整项删掉，
    老库升级时会跳号，`schema_version` 的连续性被破坏。
    所以保留占位、不再建表 —— 已存在的空表由 V28 的清理脚本 DROP。
    """
    # no-op：V28 起不再创建（原表从未被写入，见 docstring）
    return None


# 既有表的列清单（显式列出：重建时**不用** SELECT *，避免列序变化导致错位）
_PRED_COLS = ["issue", "seq", "reds", "blue", "confidence", "method",
              "reasoning", "patterns_used", "evidence_json", "created_at"]
_EVAL_META_COLS = ["issue", "method", "app_version", "app_build", "model_weights_json",
                   "config_snapshot_json", "llm_enabled", "n_tickets", "task_id",
                   "status", "created_at"]
_EVAL_DETAIL_COLS = ["issue", "method", "seq", "red_hits", "blue_hit", "prize_level",
                     "reward", "ticket_cost", "net_return", "evaluated_at"]
_EVAL_RESULT_COLS = ["issue", "red_hits", "blue_hit", "reward", "ticket_count", "created_at"]


def _tenantize_legacy_tables(conn: sqlite3.Connection) -> None:
    """迁移 8：既有表租户化（U3 §7.2）。

    ## 幂等必须是"逐子步骤"，不能是"整函数"

    全新库也会执行迁移（`SCHEMA` 已把新结构建好）。如果写成函数开头一句
    "predictions 已有 user_id 就 return"，那么在全新库上**整段会被跳过**，
    导致 tasks.user_id / issue_baseline / llm_audit.user_id 永远不建
    （U3 §7.2 踩坑预告）。所以每个子步骤各自判存在。

    ## 核心规则

    `db.SCHEMA` = 目标态（只对全新库生效）；本迁移 = 从旧态升级到目标态。
    两者描述同一个终态，必须同步修改。

    ## 回填锚点

    **动态解析** admin（`_require_admin_id`），绝不写死 `user_id=1`（Lead 裁决 1）。
    """
    admin_id = _require_admin_id(conn)

    # ---- predictions：主键 (issue,seq) → (user_id,issue,seq) ----
    _rebuild_table(conn, "predictions", """CREATE TABLE IF NOT EXISTS predictions (
        user_id       INTEGER NOT NULL DEFAULT 0,
        issue         TEXT    NOT NULL,
        seq           INTEGER NOT NULL,
        reds          TEXT    NOT NULL,
        blue          INTEGER NOT NULL,
        confidence    REAL    NOT NULL,
        method        TEXT    NOT NULL,
        reasoning     TEXT    DEFAULT '',
        patterns_used TEXT    DEFAULT '',
        evidence_json TEXT    DEFAULT '',
        config_fp     TEXT    DEFAULT '',
        created_at    REAL,
        PRIMARY KEY (user_id, issue, seq)
    )""", _PRED_COLS, default_user_id=admin_id, has_user_col=True)

    # ---- eval_meta：主键 (issue,method) → (user_id,issue,method) ----
    _rebuild_table(conn, "eval_meta", """CREATE TABLE IF NOT EXISTS eval_meta (
        user_id             INTEGER NOT NULL DEFAULT 0,
        issue               TEXT    NOT NULL,
        method              TEXT    NOT NULL,
        app_version         TEXT    NOT NULL DEFAULT '',
        app_build           TEXT    NOT NULL DEFAULT '',
        model_weights_json  TEXT    DEFAULT '',
        config_snapshot_json TEXT   DEFAULT '',
        llm_enabled         INTEGER DEFAULT 0,
        n_tickets           INTEGER DEFAULT 0,
        task_id             TEXT    DEFAULT '',
        status              TEXT    DEFAULT 'predicted',
        created_at          REAL,
        PRIMARY KEY (user_id, issue, method)
    )""", _EVAL_META_COLS, default_user_id=admin_id, has_user_col=True)

    # ---- eval_details：主键 (issue,method,seq) → (user_id,issue,method,seq) ----
    _rebuild_table(conn, "eval_details", """CREATE TABLE IF NOT EXISTS eval_details (
        user_id      INTEGER NOT NULL DEFAULT 0,
        issue        TEXT    NOT NULL,
        method       TEXT    NOT NULL,
        seq          INTEGER NOT NULL,
        red_hits     INTEGER NOT NULL,
        blue_hit     INTEGER NOT NULL,
        prize_level  INTEGER NOT NULL DEFAULT 0,
        reward       REAL    NOT NULL DEFAULT 0,
        ticket_cost  REAL    NOT NULL DEFAULT 2,
        net_return   REAL    NOT NULL DEFAULT -2,
        evaluated_at REAL,
        PRIMARY KEY (user_id, issue, method, seq)
    )""", _EVAL_DETAIL_COLS, default_user_id=admin_id, has_user_col=True)

    # ---- eval_results：主键 (issue) → (user_id,issue) ----
    _rebuild_table(conn, "eval_results", """CREATE TABLE IF NOT EXISTS eval_results (
        user_id      INTEGER NOT NULL DEFAULT 0,
        issue        TEXT    NOT NULL,
        red_hits     REAL,
        blue_hit     REAL,
        reward       REAL    DEFAULT 0,
        ticket_count INTEGER DEFAULT 0,
        created_at   REAL,
        PRIMARY KEY (user_id, issue)
    )""", _EVAL_RESULT_COLS, default_user_id=admin_id, has_user_col=True)

    # ---- tasks：纯加列（SQLite 允许带常量默认值的 ADD COLUMN）----
    # 保持**可空**：NULL = 系统/后台任务（U3 §6.2）。改成 NOT NULL 只能重建表，
    # 而"NULL=系统"这个语义本身正好需要可空。
    #
    # 注意 `_table_exists` 守卫：迁移可能在**只建了 schema_version 的最小库**上跑
    # （见 test_m4_5_engineering 的幂等用例）。此时 tasks 压根不存在，直接 ALTER
    # 会抛 "no such table"，把整批事务打回滚。**不要**在这里顺手 CREATE TABLE ——
    # 那会让迁移承担 SCHEMA 的职责，两处 DDL 迟早漂移。
    if _table_exists(conn, "tasks") and "user_id" not in _table_columns(conn, "tasks"):
        conn.execute("ALTER TABLE tasks ADD COLUMN user_id INTEGER")
        conn.execute("UPDATE tasks SET user_id=? WHERE user_id IS NULL", (admin_id,))
        _who = f"admin={admin_id}" if admin_id else "SYSTEM_USER_ID=0（未解析到管理员）"
        _log(f"tasks 已加 user_id（历史任务归属 {_who}）")

    # ---- llm_audit / llm_eval_results：纯加列（当前 0 行）----
    for t in ("llm_audit", "llm_eval_results"):
        if _table_exists(conn, t) and "user_id" not in _table_columns(conn, t):
            conn.execute(f"ALTER TABLE {t} ADD COLUMN user_id INTEGER")

    # ---- 索引（按读路径反推，U3 §3.5）----
    # 同样必须逐条判存在：迁移可能在只建了 schema_version 的最小库上跑。
    _IDX = [
        ("idx_pred_issue", "predictions", "(issue)"),
        ("idx_pred_user_issue_ts", "predictions", "(user_id, issue, created_at)"),
        ("idx_eval_meta_method", "eval_meta", "(method)"),
        ("idx_eval_meta_user", "eval_meta", "(user_id, issue)"),
        ("idx_eval_details_method", "eval_details", "(method)"),
        ("idx_eval_details_user", "eval_details", "(user_id, issue, method)"),
        ("idx_tasks_user", "tasks", "(user_id, created_at)"),
    ]
    for name, table, cols in _IDX:
        if _table_exists(conn, table):
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}{cols}")


MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "reconcile audit table", _ensure_reconcile_runs),
    (2, "legacy provenance columns", _ensure_legacy_columns),
    # V10-05（§2.1.5 编号冻结：这是第 3 号，V10 期间不再新增第 4 号）
    (3, "v10 replay results", _ensure_replay_results),
    # V12-13（V12 期间新增第 4 号：补上 V10-01 从未落地的 llm_audit 表）
    (4, "v12 llm audit table", _ensure_llm_audit),
    # MU-1 / W04（多用户批次第 5 号：身份三表 + bootstrap 主管理员行）
    # 编号占位已冻结：5=users/sessions/audit、6=（保留给 MU-2 的数据租户化）。
    (5, "mu1 identity tables", _ensure_identity_tables),
    # MU-2：6/7 是 MU-1 合并身份三表后留下的空号，见上方「编号说明」
    (6, "mu2 user llm config", _ensure_user_llm_config),
    (7, "mu2 issue baseline", _ensure_issue_baseline),
    (8, "mu2 tenantize legacy tables", _tenantize_legacy_tables),
]

_SCHEMA_VERSION_DDL = """CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at REAL NOT NULL
)"""

# 迁移前热备的兜底参数（V10 二次评审 A2）：只读连接 + 有限超时 + 分页拷贝，
# 避免在库被占用时无限重试导致进程挂死。
_BACKUP_TIMEOUT = 10.0          # 秒；sqlite3.connect(timeout=...)
_BACKUP_PAGE_CHUNK = 256        # 每批拷贝页数，减小长时间持锁窗口


def _log(msg: str) -> None:
    """迁移日志（与其他模块 ``[xxx]`` 前缀口径一致）。"""
    print(f"[migrations] {msg}")


# ---------- 版本与备份 ----------

def current_version(conn: sqlite3.Connection) -> int:
    """读取当前 schema_version（表不存在视为 0）。"""
    try:
        row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    except sqlite3.OperationalError:      # 表还不存在
        return 0
    if row is None:
        return 0
    # 注意：列名是 COALESCE(MAX(version),0)，用下标取更稳（row_factory 可能是 Row 或 tuple）
    return int(row[0])


def latest_version() -> int:
    return MIGRATIONS[-1][0] if MIGRATIONS else 0


def db_file_path(conn: sqlite3.Connection) -> Optional[Path]:
    """连接对应的主库文件路径；``:memory:`` / 临时库返回 None。"""
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        return None
    for row in rows:
        try:
            name, path = row[1], row[2]
        except (IndexError, KeyError):
            continue
        if name == "main" and path:
            return Path(path)
    return None


def backup_path_for(conn: sqlite3.Connection, version: int) -> Optional[Path]:
    """迁移前备份的目标路径：``<库文件>.pre-migrate-v<version>``（与库同目录）。"""
    path = db_file_path(conn)
    if path is None:
        return None
    return path.with_name(f"{path.name}.pre-migrate-v{int(version)}")


def has_existing_data(conn: sqlite3.Connection) -> bool:
    """库内是否已有用户数据（而非刚创建的空库）。

    V10 二次评审 A1 修复背景
    ------------------------
    旧实现用 ``current > 0`` 作为"是否需要备份"的判据，但**没有 ``schema_version``
    表的老库** ``current_version()`` 返回 **0**，于是"最需要备份的老库升级"反而
    **一行备份都不做**——正好击中"生产库零备份"这个立项理由。

    改为按"有没有既有数据"判断：
      * 空库（刚 created、无任何用户表数据）→ 不备份（没有可丢的东西）；
      * **有数据的库（无论有没有 schema_version 表）→ 必须备份**。
    """
    # 除迁移自身维护的 schema_version 外，是否还有其它含数据的表
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for name in names:
        if name == "schema_version":
            continue
        try:
            row = conn.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone()
        except sqlite3.Error:
            continue
        if row is not None:
            return True
    return False


def backup_before_migrate(conn: sqlite3.Connection, version: int) -> Optional[Path]:
    """迁移前一致性热备（``Connection.backup``），返回备份文件路径。

    先写 ``.tmp`` 再 ``os.replace``，避免备份过程中留下的半截文件被误当作备份。

    V10 二次评审 A2 修复背景
    ------------------------
    旧实现直接用**调用方的** ``conn`` 做 ``Connection.backup()``。当调用方已有未提交
    事务时，``backup()`` 遇 BUSY 会无限重试 → 进程挂死，且让 SAVEPOINT 嵌套分支
    永远不可达（变异删掉该分支测试仍全绿 = 零覆盖）。

    改为：**另开一个只读连接**做热备，与调用方事务完全隔离：
      * 不参与调用方事务，因此不会触发 BUSY 自杀锁；
      * 也不会把调用方未提交的中间态写进备份。
    并对 ``backup()`` 设置页数上限与超时兜底，仍失败则抛错（fail-closed）。
    """
    dest = backup_path_for(conn, version)
    if dest is None:
        return None
    path = db_file_path(conn)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # 独立只读连接：与调用方事务解耦，避免 BUSY 无限重试
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_BACKUP_TIMEOUT)
    try:
        target = sqlite3.connect(str(tmp), timeout=_BACKUP_TIMEOUT)
        try:
            src.backup(target, pages=_BACKUP_PAGE_CHUNK)
        finally:
            target.close()
    finally:
        src.close()
    os.replace(str(tmp), str(dest))
    return dest


# ---------- 批次事务执行 ----------

def _execute_pending(conn: sqlite3.Connection,
                     pending: Sequence[Tuple[int, str, Callable[[sqlite3.Connection], None]]]) -> None:
    conn.execute(_SCHEMA_VERSION_DDL)
    for version, name, migration in pending:
        migration(conn)
        conn.execute("INSERT INTO schema_version(version, name, applied_at) VALUES (?,?,?)",
                     (version, name, time.time()))


def _run_batch(conn: sqlite3.Connection,
               pending: Sequence[Tuple[int, str, Callable[[sqlite3.Connection], None]]]) -> None:
    """在**一个事务**里执行全部待执行迁移；任一失败 → 整体回滚。"""
    if conn.in_transaction:
        # 调用方已有未提交事务：用 SAVEPOINT 嵌套，失败只回滚到迁移开始处
        conn.execute("SAVEPOINT ssq_migrations")
        try:
            _execute_pending(conn, pending)
        except BaseException:
            conn.execute("ROLLBACK TO ssq_migrations")
            conn.execute("RELEASE ssq_migrations")
            raise
        conn.execute("RELEASE ssq_migrations")
        return

    original_isolation = conn.isolation_level
    try:
        # autocommit + 显式 BEGIN：LEGACY 模式下 DDL 不会被包进事务，必须显式开
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            _execute_pending(conn, pending)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        try:
            conn.isolation_level = original_isolation
        except sqlite3.Error:      # pragma: no cover - 仅在连接已关闭等极端情况
            pass


def apply(conn: sqlite3.Connection, *, backup: bool = True) -> int:
    """执行全部待执行迁移，返回执行后的 schema_version。

    * ``backup=True``（默认）：有待执行迁移且**库内已有数据**时，先做一致性热备
      （见 :func:`backup_before_migrate`）；备份失败则**不迁移**（fail-closed）。
      **注意**：判据是"有没有既有数据"，不是"schema_version > 0"。老库（无
      ``schema_version`` 表、但含真实开奖数据）升级时 ``current == 0``，旧实现
      因此跳过备份——这是 V10 二次评审 A1 修复的真实缺陷。
    * 幂等：已是最新版本时直接返回，不备份、不写库。
    * 失败：整体回滚（含 DDL），``schema_version`` 不前进，异常原样抛出。
    """
    current = current_version(conn)
    pending = [(v, n, f) for v, n, f in MIGRATIONS if v > current]
    if not pending:
        return int(current)

    if backup:
        # A1：老库（current==0）同样必须备份——判据改为"库内是否有既有数据"
        if has_existing_data(conn):
            # fail-closed：备份不成功就不迁移，避免"迁移后才发现无路可退"
            backup_before_migrate(conn, current)
        else:
            _log("库内无既有数据，跳迁移前备份（空库无可丢内容）")

    _run_batch(conn, pending)
    return current_version(conn)


def status(conn: sqlite3.Connection) -> dict:
    return {"version": current_version(conn), "latest": latest_version()}
