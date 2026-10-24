"""SQLite 存取层：开奖数据、特征、规律回测、预测、在线评估。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional

from . import config
from . import methods as _methods

SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
  issue TEXT PRIMARY KEY,
  date TEXT NOT NULL,
  r1 INTEGER, r2 INTEGER, r3 INTEGER, r4 INTEGER, r5 INTEGER, r6 INTEGER,
  blue INTEGER NOT NULL,
  o1 INTEGER, o2 INTEGER, o3 INTEGER, o4 INTEGER, o5 INTEGER, o6 INTEGER,
  sales INTEGER DEFAULT 0, pool INTEGER DEFAULT 0,
  p1 INTEGER DEFAULT 0, p1_amt INTEGER DEFAULT 0,
  p2 INTEGER DEFAULT 0, p2_amt INTEGER DEFAULT 0,
  p3 INTEGER DEFAULT 0, p3_amt INTEGER DEFAULT 0,
  p4 INTEGER DEFAULT 0, p4_amt INTEGER DEFAULT 0,
  p5 INTEGER DEFAULT 0, p5_amt INTEGER DEFAULT 0,
  p6 INTEGER DEFAULT 0, p6_amt INTEGER DEFAULT 0,
  source TEXT NOT NULL DEFAULT '17500',
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(date);

-- M4.4：多源对账审计，仅记录结果，不修改主开奖数据
CREATE TABLE IF NOT EXISTS reconcile_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  checked_at REAL NOT NULL,
  secondary_url TEXT DEFAULT '',
  status TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reconcile_runs_checked_at ON reconcile_runs(checked_at);

CREATE TABLE IF NOT EXISTS features (
  issue TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  created_at REAL
);

CREATE TABLE IF NOT EXISTS patterns (
  key TEXT PRIMARY KEY,
  name_zh TEXT NOT NULL,
  kind TEXT NOT NULL,            -- long / mid / short
  desc TEXT,
  params TEXT,
  grade TEXT DEFAULT 'C',        -- A / B / C
  sample_size INTEGER DEFAULT 0,
  margin REAL DEFAULT 0,
  p_value REAL DEFAULT 1.0,
  p_adj REAL DEFAULT 1.0,
  direction TEXT DEFAULT '',
  backtest_json TEXT,
  updated_at REAL
);

-- MU-2：主键 (issue,seq) → (user_id,issue,seq)。
-- ⚠️ 本 SCHEMA 只对**全新库**生效；老库由 migrations 迁移 8 重建。
-- 两者必须描述同一个终态（改这里就要改迁移，反之亦然）。
CREATE TABLE IF NOT EXISTS predictions (
  user_id INTEGER NOT NULL DEFAULT 0,   -- 0 = SYSTEM_USER_ID（兜底语义）
  issue TEXT NOT NULL,
  seq INTEGER NOT NULL,
  reds TEXT NOT NULL,            -- "1,4,7,12,28,31"
  blue INTEGER NOT NULL,
  confidence REAL NOT NULL,
  method TEXT NOT NULL,          -- stat / llm / mixed
  reasoning TEXT DEFAULT '',
  patterns_used TEXT DEFAULT '',
  evidence_json TEXT DEFAULT '',
  config_fp TEXT DEFAULT '',     -- 生成时的配置指纹（缓存键，见 U3 §4.7）
  created_at REAL,
  PRIMARY KEY (user_id, issue, seq)
);
CREATE INDEX IF NOT EXISTS idx_pred_issue ON predictions(issue);
-- ⚠️ 其余新索引（引用 user_id 的）**不能放在 SCHEMA 里**：
-- SCHEMA 在 `migrations.apply()` **之前**执行，而老库的 predictions 此时还没有
-- user_id 列 → `CREATE INDEX ... (user_id, ...)` 直接抛
-- "no such column: user_id"，整个 get_conn() 失败（生产升级当场起不来）。
-- 这类索引只能由迁移 8 创建（那里逐条判表存在）。

CREATE TABLE IF NOT EXISTS eval_results (
  user_id INTEGER NOT NULL DEFAULT 0,
  issue TEXT NOT NULL,
  red_hits REAL,
  blue_hit REAL,
  reward REAL DEFAULT 0,
  ticket_count INTEGER DEFAULT 0,
  created_at REAL,
  PRIMARY KEY (user_id, issue)
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,           -- predict / backtest / eval / mine / llm_eval
  status TEXT DEFAULT 'pending',-- pending / running / completed / failed
  progress REAL DEFAULT 0.0,
  message TEXT DEFAULT '',
  result_json TEXT DEFAULT '',
  created_at REAL,
  updated_at REAL
);

CREATE TABLE IF NOT EXISTS llm_eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,              -- 如 llm_eval_20260817_100001
  created_at TEXT NOT NULL,
  window_issues INTEGER NOT NULL,    -- 窗口期数
  tickets INTEGER NOT NULL,          -- 每期每通道注数
  channel TEXT NOT NULL,             -- stat | stat_llm | random
  issue TEXT,                        -- 逐期明细行填期号；汇总行(NULL)
  red_hits REAL, blue_hit INTEGER, prize_level INTEGER, roi REAL,
  tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0.0, duration_ms INTEGER DEFAULT 0,
  metrics_json TEXT DEFAULT '',      -- 汇总行：EV.aggregate 结果
  p_values_json TEXT DEFAULT '',     -- 汇总行：paired 检验结果
  seed INTEGER
);
CREATE INDEX IF NOT EXISTS idx_llm_eval_run ON llm_eval_results(run_id);

-- M4.1：预测方法与配置不可变快照
CREATE TABLE IF NOT EXISTS eval_meta (
  user_id INTEGER NOT NULL DEFAULT 0,
  issue TEXT NOT NULL,
  method TEXT NOT NULL,
  app_version TEXT NOT NULL DEFAULT '',
  app_build TEXT NOT NULL DEFAULT '',
  model_weights_json TEXT DEFAULT '',
  config_snapshot_json TEXT DEFAULT '',
  llm_enabled INTEGER DEFAULT 0,
  n_tickets INTEGER DEFAULT 0,
  task_id TEXT DEFAULT '',
  status TEXT DEFAULT 'predicted',
  created_at REAL,
  PRIMARY KEY (user_id, issue, method)
);
CREATE INDEX IF NOT EXISTS idx_eval_meta_method ON eval_meta(method);

CREATE TABLE IF NOT EXISTS eval_details (
  user_id INTEGER NOT NULL DEFAULT 0,
  issue TEXT NOT NULL,
  method TEXT NOT NULL,
  seq INTEGER NOT NULL,
  red_hits INTEGER NOT NULL,
  blue_hit INTEGER NOT NULL,
  prize_level INTEGER NOT NULL DEFAULT 0,
  reward REAL NOT NULL DEFAULT 0,
  ticket_cost REAL NOT NULL DEFAULT 2,
  net_return REAL NOT NULL DEFAULT -2,
  evaluated_at REAL,
  PRIMARY KEY (user_id, issue, method, seq)
);
CREATE INDEX IF NOT EXISTS idx_eval_details_method ON eval_details(method);

CREATE TABLE IF NOT EXISTS mining_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  engine TEXT NOT NULL,              -- lightgbm / rf / lift
  candidates INTEGER DEFAULT 0,      -- 候选规律数
  accepted INTEGER DEFAULT 0,        -- B 级及以上入库数
  avg_lift REAL DEFAULT 0.0,
  pass_rate REAL DEFAULT 0.0,        -- accepted / candidates
  duration_ms INTEGER DEFAULT 0,
  params_json TEXT DEFAULT ''
);
"""

# ---------------------------------------------------------------------------
# 连接管理（P2 稳定性修复：单连接 → 每线程连接）
# ---------------------------------------------------------------------------
#
# ## 修复的真实故障（2026-10-20 生产）
#
# `/api/auth/login` 持续 500：`sqlite3.OperationalError: database is locked`，
# 容器重启即恢复、但会复发。根因是历史实现：
#
#     _CONN = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
#
# **一条连接 + check_same_thread=False + 40 线程池，且没有任何锁**。
# `check_same_thread=False` 只是关掉 Python 的跨线程检查，并没有让连接变线程安全。
# 多线程同时向同一条连接发语句时：
#   * 事务边界互相串（A 的 commit 提交了 B 未完成的写）；
#   * 一个线程开着写事务没提交，另一个线程来写 → **自己等自己的锁** →
#     `database is locked`。这是 `busy_timeout` 救不了的情形：它等的是同进程内
#     另一条语句持有的锁，不会因为时间流逝而释放。
#
# ## 修法：每线程一条连接
#
# 这是 SQLite 官方推荐的多线程用法，也是**改动面最小**的正确解法：
# `get_conn()` 签名不变，200+ 处调用点一行都不用改。
#
# WAL 模式下多连接并发是安全且高效的：读不阻塞写、写不阻塞读，
# 写与写之间由 SQLite 自己的文件锁 + `busy_timeout` 串行化（这是跨连接的情形，
# 与"同进程自锁"本质不同，busy_timeout 在这里**能**正常工作）。
#
# ## 注意事项
#
# * 每个线程的连接在首次调用时建立；`close()` 只关当前线程的。
# * 进程内可能积累若干空闲连接。单机部署下数量 = 线程数（≤40），可忽略。
# * `SCHEMA` / 迁移只在**建库时**跑一次，用进程级锁保证不会被并发重复执行。

_TLS = threading.local()
# 只在"首次建库/迁移"时用到，保证 SCHEMA + 迁移不被多个线程重复执行
_INIT_LOCK = threading.Lock()
_initialized = False


def _configure(conn: sqlite3.Connection) -> None:
    """连接级 pragma（每个连接都要设一遍）。

    1) `busy_timeout=5000` —— 跨连接写冲突时**等待**而不是立刻抛错；
    2) `journal_mode=WAL` —— 读不阻塞写、写不阻塞读。WAL 是**持久属性**
       （写在库文件头），设一次长期生效，重复设置无害；
    3) `synchronous=NORMAL` —— WAL 下的推荐值：崩溃安全性与 FULL 等价
       （不丢已提交事务），但少一次 fsync。
    """
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")


def _ensure_initialized(conn: sqlite3.Connection) -> None:
    """建表 + 迁移（幂等，进程内只真正执行一次）。"""
    global _initialized
    if _initialized:
        return
    with _INIT_LOCK:
        if _initialized:
            return
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn.executescript(SCHEMA)
        from . import migrations
        migrations.apply(conn)
        _ensure_columns(conn)
        conn.commit()
        _initialized = True


def get_conn() -> sqlite3.Connection:
    """返回**当前线程**的连接（首次调用时建立）。

    签名与历史实现一致 —— 这是刻意为之：改造的是内部实现，
    不是 200+ 处调用点。

    测试上下文的**生产库护栏**挂在这里：这是全项目唯一的建连入口，
    放在此处的覆盖率是 100%（改 config 里定义一个没人调的函数等于没做）。
    """
    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        return conn
    # 测试上下文里若 DB_PATH 指向生产库 → 直接 raise（见 config.guard_prod_db）。
    # 只在建连那一刻检查，正常生产运行无 pytest 痕迹，不触发。
    config.guard_prod_db()
    conn = sqlite3.connect(str(config.DB_PATH), timeout=5.0)
    _configure(conn)
    _ensure_initialized(conn)
    _TLS.conn = conn
    return conn


def close():
    """关闭**当前线程**的连接，并允许下次调用时重新初始化。

    测试大量依赖它做隔离（`db.close()` 后再连临时库）。多线程下只关当前
    线程是正确语义 —— 关别人的连接会让那个线程下一次操作炸掉。

    同时重置 `_initialized`：测试会把 `config.DB_PATH` 换到新库，
    若保留"已初始化"标记，新库就永远不会建表。
    """
    global _initialized
    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:      # noqa: BLE001 —— 已坏的连接关闭也可能抛
            pass
        _TLS.conn = None
    _initialized = False


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """存量库幂等补列。"""
    pred_cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "evidence_json" not in pred_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN evidence_json TEXT DEFAULT ''")
    draw_cols = {r["name"] for r in conn.execute("PRAGMA table_info(draws)").fetchall()}
    if "source" not in draw_cols:
        conn.execute("ALTER TABLE draws ADD COLUMN source TEXT NOT NULL DEFAULT '17500'")
    conn.commit()




# ---------- draws ----------

def upsert_draws(draws: Iterable[Dict]) -> int:
    conn = get_conn()
    n = 0
    now = time.time()
    for d in draws:
        conn.execute(
            """INSERT OR REPLACE INTO draws
               (issue,date,r1,r2,r3,r4,r5,r6,blue,o1,o2,o3,o4,o5,o6,
                sales,pool,p1,p1_amt,p2,p2_amt,p3,p3_amt,p4,p4_amt,p5,p5_amt,p6,p6_amt,source,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d["issue"], d["date"], *d["reds"], d["blue"], *d["order"],
                d.get("sales", 0), d.get("pool", 0),
                *d.get("prizes", [0] * 12)[0:2], *d.get("prizes", [0] * 12)[2:4],
                *d.get("prizes", [0] * 12)[4:6], *d.get("prizes", [0] * 12)[6:8],
                *d.get("prizes", [0] * 12)[8:10], *d.get("prizes", [0] * 12)[10:12],
                d.get("source", "17500"), now,
            ),
        )
        n += 1
    conn.commit()
    return n


def save_reconcile_run(result: Dict, secondary_url: str = "") -> int:
    """保存一次对账审计结果；只记录结果，不改变 draws。"""
    status = "ok" if result.get("ok") else str(result.get("status") or "mismatch")
    cur = get_conn().execute(
        "INSERT INTO reconcile_runs (checked_at, secondary_url, status, summary_json) VALUES (?,?,?,?)",
        (time.time(), secondary_url, status, json.dumps(result, ensure_ascii=False)),
    )
    get_conn().commit()
    return int(cur.lastrowid)


def load_reconcile_runs(limit: int = 30) -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM reconcile_runs ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["summary"] = json.loads(item.pop("summary_json") or "{}")
        except (TypeError, ValueError):
            item["summary"] = {}
        out.append(item)
    return out


def max_issue() -> Optional[str]:
    row = get_conn().execute("SELECT MAX(issue) m FROM draws").fetchone()
    return row["m"] if row and row["m"] else None


def count_draws() -> int:
    return get_conn().execute("SELECT COUNT(*) c FROM draws").fetchone()["c"]


def load_draws(limit: Optional[int] = None) -> List[Dict]:
    sql = "SELECT * FROM draws ORDER BY issue ASC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = get_conn().execute(sql).fetchall()
    return [_row_to_draw(r) for r in rows]


def load_last_draws(n: int) -> List[Dict]:
    rows = get_conn().execute(
        # 数值排序：文本排序会把越界期号（2026401）排在 2026119 之前
        "SELECT * FROM draws ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?", (int(n),)
    ).fetchall()
    return [_row_to_draw(r) for r in reversed(rows)]


#: 奖级键（一~六等奖的中奖注数与单注金额）。列名在 `draws` 表中确实存在，
#: 之前 `_row_to_draw` **从不读取**它们 —— 见 `_row_to_draw` 的说明。
PRIZE_LEVELS = (1, 2, 3, 4, 5, 6)


def _row_to_draw(r: sqlite3.Row) -> Dict:
    """把一行 `draws` 转成 dict。

    ## 为什么要把 p1..p6 / *_amt 也带出来（V14-B，2026-10-21）

    在此之前这个函数只返回 `issue/date/reds/blue/order/sales/pool/source`，
    **`p1`、`p1_amt`、`p2..p6` 被直接丢弃** —— 而它们在库里有真实数据：

        p1      （一等奖注数）  3266 期非零，1~153 注，中位 6，另有 251 期真零（无人中）
        p1_amt  （单注金额）    3269 期，139 万~1500 万（封顶），中位 684 万
        p2..p6                 约 3428 期

    这些字段是**全项目唯一能反映"人群怎么选号"的观测量**。号码本身不可预测
    （红球 χ² p=0.065、边际熵 5.0429/5.0444），但 `p1` 可预测 —— 因为它是人的行为。
    实测（控制销量后）``p1`` 与号码形态的 Spearman 相关：生日号 +0.1389（p<0.001）、
    同尾号 +0.0543（p=0.0015）。丢弃这些字段等于主动放弃唯一有信号的那条轴。

    ## 兼容性

    纯**增量**：原有键一个不变、顺序不变、语义不变，只是多出 `p1..p6` 与
    `p1_amt..p6_amt`（键名与列名同构）。老调用方按 `draw["reds"]` 取值不受影响。
    `source` 的 `in r.keys()` 判据保留 —— 少数历史建库路径没有该列。
    """
    out = {
        "issue": r["issue"], "date": r["date"],
        "reds": [r[f"r{i}"] for i in range(1, 7)],
        "blue": r["blue"],
        "order": [r[f"o{i}"] for i in range(1, 7)],
        "sales": r["sales"], "pool": r["pool"],
        "source": r["source"] if "source" in r.keys() else "17500",
    }
    # 奖级字段：列可能缺失（早期 schema），逐个探测而不是假定存在。
    keys = r.keys()
    for lv in PRIZE_LEVELS:
        for suffix in ("", "_amt"):
            col = f"p{lv}{suffix}"
            if col in keys:
                out[col] = r[col]
    return out


# ---------- features ----------

def save_features(issue: str, payload: dict):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO features (issue, payload, created_at) VALUES (?,?,?)",
        (issue, json.dumps(payload, ensure_ascii=False), time.time()),
    )
    conn.commit()



# ---------- patterns ----------

def save_pattern_results(results: List[Dict]):
    conn = get_conn()
    now = time.time()
    for r in results:
        conn.execute(
            """INSERT OR REPLACE INTO patterns
               (key,name_zh,kind,desc,params,grade,sample_size,margin,p_value,p_adj,direction,backtest_json,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r["key"], r["name_zh"], r["kind"], r.get("desc", ""),
                json.dumps(r.get("params", {}), ensure_ascii=False),
                r["grade"], r["sample_size"], r["margin"], r["p_value"],
                r.get("p_adj", r["p_value"]), r["direction"],
                json.dumps(r.get("backtest", {}), ensure_ascii=False), now,
            ),
        )
    conn.commit()


def load_patterns(grade_filter: Optional[str] = None) -> List[Dict]:
    sql = "SELECT * FROM patterns"
    params: List[str] = []
    if grade_filter:
        sql += " WHERE grade=?"
        params.append(grade_filter)
    sql += " ORDER BY kind, key"
    rows = get_conn().execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d.get("params") or "{}")
        d["backtest"] = json.loads(d.get("backtest_json") or "{}")
        d.pop("backtest_json", None)
        out.append(d)
    return out


def replace_mined_patterns(results: List[Dict]) -> None:
    """幂等替换本机发布的挖掘规律，不影响内置规律。"""
    conn = get_conn()
    conn.execute("DELETE FROM patterns WHERE key LIKE 'mined_%'")
    conn.commit()
    if results:
        save_pattern_results(results)


# ---------- 租户解析（MU-2）----------
#
# 为什么要一个统一的解析函数，而不是在每个读写下写 `user_id or 0`：
#
# 1. **串号是静默的**。少写一个 `AND user_id=?`，用户 B 重新生成就会删光用户 A
#    同期的票（U3 §3.7 缺陷 W1），而且票单只存这一份、**不可恢复**。
#    把它收进一个函数，才有唯一的地方能测"到底用的是谁"。
# 2. **后台线程不继承 contextvars**（裸 threading.Thread / ThreadPoolExecutor）。
#    调度器、boot 预测、异步任务都会丢上下文；必须有确定性兜底，否则要么炸、
#    要么静默落到错误的人头上。
# 3. **兜底绝不能是写死的 1**。`AUTOINCREMENT` 不复用 ID（实测清空重建拿到 2），
#    写死 1 在"删过 admin 再重建"的库上会指向不存在的人或普通用户。

SYSTEM_USER_ID = 0            # 与 config.SYSTEM_USER_ID 一致；不插入 users 表


def _single_tenant_admin_id() -> Optional[int]:
    """单租户/无会话场景下的归属：动态解析主管理员，解析不到返回 None。

    结果**不缓存**：管理员可能被停用/删除/新建，缓存会让"归属"在运行中途漂移。
    单次查询走 `users` 的小表索引，代价可忽略（相比一次预测计算）。
    """
    try:
        row = get_conn().execute(
            "SELECT user_id FROM users WHERE is_primary_admin=1 AND deleted_at IS NULL"
            " ORDER BY user_id LIMIT 1").fetchone()
        if row:
            return int(row[0])
        row = get_conn().execute(
            "SELECT user_id FROM users WHERE role='admin' AND status='active'"
            " AND deleted_at IS NULL ORDER BY user_id LIMIT 1").fetchone()
        if row:
            return int(row[0])
    except Exception:      # noqa: BLE001 —— 老库/迁移未跑时不要因为解析归属而炸
        return None
    return None


def resolve_tenant(user_id: Optional[int] = None) -> int:
    """解析"这次读写算谁的"。顺序：显式 → 请求上下文 → 主管理员 → SYSTEM。

    显式参数永远优先：后台任务/调度器必须**显式**传入发起人，不能依赖上下文。

    ## V36 修正（2026-10-24 审计 P0-5）：HTTP 请求上下文里 **绝不** 兜底到主管理员

    "主管理员兜底"存在的正当理由只有一个：**后台线程不继承 contextvars**
    （见上面第 2 条）—— 调度器/boot 预测丢了上下文后，需要有一个确定的落点。
    但它被错误地用到了 HTTP 请求路径上：`AUTH_MODE=off` 时中间件对无会话请求
    直接放行且不写 contextvar，于是匿名请求走到这里 → 返回主管理员 uid →
    匿名 `POST /api/predict?regenerate=1` 会 DELETE 并重写**主管理员**的票单。

    现在改为：**只有在非 HTTP 上下文**（没有 `request_context` 的请求标记）时
    才启用管理员兜底；HTTP 请求且无身份 ⇒ 归 `SYSTEM_USER_ID`（谁也不属于），
    绝不自作主张记到某个人名下。
    """
    if user_id is not None:
        return int(user_id)
    in_request = False
    try:
        # P2：改依赖 `request_context`（纯状态模块），不再反向 import `api_app`。
        # 依赖方向：db → request_context ← api_app，单向。
        from . import request_context
        in_request = bool(request_context.in_request())
        uid = request_context.current_user_id()
        if uid:
            return int(uid)
    except Exception:      # noqa: BLE001 —— 无请求上下文
        in_request = False
    # HTTP 请求但没解析出身份：不冒认任何人（V36，P0-5）。
    if in_request:
        return SYSTEM_USER_ID
    admin = _single_tenant_admin_id()
    if admin:
        return int(admin)
    return SYSTEM_USER_ID


# ---------- predictions ----------

def save_predictions(issue: str, tickets: List[Dict], *,
                     user_id: Optional[int] = None,
                     config_fp: str = "") -> int:
    """幂等替换**某个用户**某一期的票单，返回实际写入归属的 user_id。

    ⚠️ U3 §3.7 缺陷 W1（最严重的串号）：旧实现的 ``DELETE FROM predictions
    WHERE issue=?`` **不带 user_id**。多用户下一旦用户 B 重新生成，就会删光
    用户 A 同期的票 —— 而票单只存这一份，**不可恢复**。
    现在 DELETE 与 INSERT 都带 user_id。
    """
    uid = resolve_tenant(user_id)
    conn = get_conn()
    conn.execute("DELETE FROM predictions WHERE user_id=? AND issue=?", (uid, issue))
    now = time.time()
    for i, t in enumerate(tickets, 1):
        conn.execute(
            """INSERT INTO predictions
               (user_id,issue,seq,reds,blue,confidence,method,reasoning,
                patterns_used,evidence_json,config_fp,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid, issue, i, ",".join(map(str, t["reds"])), t["blue"], t["confidence"],
                t.get("method", "mixed"), t.get("reasoning", ""),
                json.dumps(t.get("patterns_used", []), ensure_ascii=False),
                json.dumps(t.get("evidence", {}), ensure_ascii=False),
                str(config_fp or ""), now,
            ),
        )
    conn.commit()
    return uid


def load_predictions(issue: str, *, user_id: Optional[int] = None) -> List[Dict]:
    uid = resolve_tenant(user_id)
    rows = get_conn().execute(
        "SELECT * FROM predictions WHERE user_id=? AND issue=? ORDER BY seq", (uid, issue)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["reds"] = [int(x) for x in d["reds"].split(",")]
        d["patterns_used"] = json.loads(d.get("patterns_used") or "[]")
        d["evidence"] = json.loads(d.get("evidence_json") or "{}")
        d.pop("evidence_json", None)
        out.append(d)
    return out


def _recent_prediction_issues(limit: int = 12,
                              *, user_id: Optional[int] = None) -> List[str]:
    """按**落库时间**取最近用过的期号（比按期号排序更贴近"哪一期是最新的"）。

    历史 bug 遗留的越界期号（如 2026401）在数值上也大于同年合法期号，
    纯按期号排会把它当"最新"，这正是"刷新后又是旧数据"的根因。
    """
    rows = get_conn().execute(
        "SELECT issue, MAX(created_at) AS ts FROM predictions WHERE user_id=?"
        " GROUP BY issue ORDER BY ts DESC, CAST(issue AS INTEGER) DESC LIMIT ?",
        (resolve_tenant(user_id), int(limit))).fetchall()
    return [r["issue"] for r in rows]


def _valid_issue(s: str) -> bool:
    """期号自洽性：7 位数字、年份 2003~2100、当年序号 1~160（历史最多 154）。"""
    s = str(s or "")
    if not (len(s) == 7 and s.isdigit()):
        return False
    return 2003 <= int(s[:4]) <= 2100 and 1 <= int(s[4:]) <= 160


def load_last_predictions(*, user_id: Optional[int] = None) -> List[Dict]:
    """最近一期预测：先按落库时间取候选，再在**合法期号**里取最大者。

    全部候选都不合法时（库被历史 bug 污染），退回时间戳最新的那条并如实保留，
    绝不静默返回空列表让前端以为"没有历史预测"。
    """
    uid = resolve_tenant(user_id)
    cand = _recent_prediction_issues(user_id=uid)
    valid = [i for i in cand if _valid_issue(i)]
    issue = max(valid, key=lambda x: int(x)) if valid else (cand[0] if cand else None)
    return load_predictions(issue, user_id=uid) if issue else []


def recent_prediction_issues(n: int = 20, *,
                             user_id: Optional[int] = None) -> List[str]:
    """最近 n 个预测期号：合法期号按期号降序，越界期号置尾（不丢弃、不隐藏）。

    ⚠️ U3 §3.7 缺陷 W3：旧实现不带 user，`online_check()` 会把**所有人**的票
    混在一起评估 → 战绩串号。
    """
    rows = get_conn().execute(
        "SELECT DISTINCT issue FROM predictions WHERE user_id=?"
        " ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?",
        (resolve_tenant(user_id), int(n))
    ).fetchall()
    got = [r["issue"] for r in rows]
    valid = sorted([i for i in got if _valid_issue(i)], key=lambda x: -int(x))
    invalid = [i for i in got if not _valid_issue(i)]
    return (valid + invalid)[: int(n)]


# ---------- eval ----------

def save_eval_meta(issue: str, tickets: List[Dict], result: Optional[Dict] = None,
                   task_id: str = "", *, user_id: Optional[int] = None) -> None:
    """保存一次预测的方法/版本/配置快照，历史评估不依赖当前配置。

    ⚠️ U3 §3.7 缺陷 W4：``INSERT OR REPLACE`` + 主键无 user →
    不同用户的配置快照会**互相覆盖**。主键已含 user_id，现在各存各的。
    """
    uid = resolve_tenant(user_id)
    conn = get_conn()
    now = time.time()
    result = result or {}
    methods: Dict[str, List[Dict]] = {}
    for t in tickets:
        methods.setdefault(str(t.get("method", "mixed")), []).append(t)
    for method, rows in methods.items():
        conn.execute(
            """INSERT OR REPLACE INTO eval_meta
               (user_id,issue,method,app_version,app_build,model_weights_json,
                config_snapshot_json,llm_enabled,n_tickets,task_id,status,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, issue, method, config.APP_VERSION, config.APP_BUILD,
             json.dumps({"ml": result.get("ml", {}), "llm_models": result.get("llm_models", [])}, ensure_ascii=False),
             json.dumps({
                 "llm_used": bool(result.get("llm_used")),
                  "requested_tickets": result.get("requested_tickets"),
                  "actual_tickets": result.get("actual_tickets", len(tickets)),
                  "shortfall_reason": result.get("shortfall_reason"),
                 "n_tickets": len(rows),
                 # M4.2：记录生成预测时的方法开关模式与规格快照，供开奖后长期对照
                 "method_mode": getattr(config, "METHOD_MODE", "production"),
                 "methods_raw": getattr(config, "METHODS_RAW", ""),
                 "configured_spec": {
                      "mode": config.METHODS_SPEC.get("mode", "all"),
                      "tokens": sorted(config.METHODS_SPEC.get("tokens", set())),
                  },
                  "effective_spec": {
                      "mode": _methods.effective_spec(config.METHOD_MODE, config.METHODS_SPEC).get("mode", "all"),
                      "tokens": sorted(_methods.effective_spec(config.METHOD_MODE, config.METHODS_SPEC).get("tokens", set())),
                  },
                  "methods_spec": {
                     "mode": config.METHODS_SPEC.get("mode", "all"),
                     "tokens": sorted(config.METHODS_SPEC.get("tokens", set())),
                 },
                 # U2/U3/U4：记录投注/覆盖模式与蓝球覆盖快照，供「从缓存返回预测」重建
                 # 完整响应头（避免 from_cache 丢元数据，前端新特性依赖这些字段）
                 "bet_mode": result.get("bet_mode"),
                 "coverage_mode": result.get("coverage_mode"),
                 "blue_covered": result.get("blue_covered"),
                 "blue_coverage_rate": result.get("blue_coverage_rate"),
                 # LLM 实际注数模式：True = 注数以 LLM 真实产出为准（未补齐统计票）
                 "llm_only_output": bool(result.get("llm_only_output", False)),
             }, ensure_ascii=False),
             int(bool(result.get("llm_used") or method.startswith("llm:"))), len(rows), task_id,
             "predicted", now),
        )
    conn.commit()


def mark_prediction_source(issue: str, *, source_mode: str,
                           strict_gates: Optional[List[str]] = None,
                           strict_reason: str = "",
                           user_id: Optional[int] = None) -> int:
    """给已落库的预测**补标口径来源**（V16），返回受影响行数。

    ## 为什么需要它

    严格模式门控被拒后，后台调度器会**降级为统计模式**自动出号（用户既定策略），
    但这种降级必须**可见** —— 否则用户看到 5 注票，会默认它们是"过了严格模式"的产物。
    2026-10-22 的故障就是这样：`issue=2026122` 的 5 条全是 `stat:*`，而界面上
    没有任何"这是降级来的"标记。

    ## 写在哪

    写进 `eval_meta.config_snapshot_json` 的 `source_mode` 等字段 —— **不改表结构**
    （加列要跑迁移，而这是一次纯展示口径修复，不值得引入 schema 风险）。
    读侧由 :func:`load_prediction_source` 取回。

    `source_mode` 取值：
      * ``"strict"``        —— 过了严格模式门控（有可复核规律 + LLM 通道可用）
      * ``"stat_fallback"`` —— 严格模式未出号，按既定策略降级为统计模式（**必须标注**）
      * ``"stat"``          —— 本来就没开严格模式，正常统计出号
    """
    uid = resolve_tenant(user_id)
    conn = get_conn()
    rows = conn.execute(
        "SELECT method, config_snapshot_json FROM eval_meta WHERE user_id=? AND issue=?",
        (uid, issue),
    ).fetchall()
    n = 0
    for r in rows:
        d = dict(r)
        try:
            snap = json.loads(d.get("config_snapshot_json") or "{}")
        except Exception:          # noqa: BLE001
            snap = {}
        snap["source_mode"] = str(source_mode)
        snap["strict_gates"] = list(strict_gates or [])
        snap["strict_reason"] = str(strict_reason or "")
        conn.execute(
            "UPDATE eval_meta SET config_snapshot_json=? WHERE user_id=? AND issue=? AND method=?",
            (json.dumps(snap, ensure_ascii=False), uid, issue, d["method"]),
        )
        n += 1
    conn.commit()
    return n


def load_prediction_source(issue: str, *, user_id: Optional[int] = None) -> Dict:
    """读回某期预测的口径来源（V16）。无记录时返回 ``source_mode="unknown"``。

    网关取**任一条**方法的快照即可 —— `mark_prediction_source` 是整期统一标注的。
    """
    uid = resolve_tenant(user_id)
    row = get_conn().execute(
        "SELECT config_snapshot_json FROM eval_meta WHERE user_id=? AND issue=? LIMIT 1",
        (uid, issue),
    ).fetchone()
    if not row:
        return {"source_mode": "unknown", "strict_gates": [], "strict_reason": ""}
    try:
        snap = json.loads(row[0] or "{}")
    except Exception:              # noqa: BLE001
        snap = {}
    return {
        "source_mode": snap.get("source_mode") or "unknown",
        "strict_gates": snap.get("strict_gates") or [],
        "strict_reason": snap.get("strict_reason") or "",
    }


def save_eval_details(issue: str, method: str, tickets: List[Dict], draw: Dict,
                      *, user_id: Optional[int] = None) -> Dict:
    """幂等保存逐注开奖对照明细，并返回方法汇总。"""
    from . import evaluate as EV
    uid = resolve_tenant(user_id)
    conn = get_conn()
    now = time.time()
    conn.execute("DELETE FROM eval_details WHERE user_id=? AND issue=? AND method=?",
                 (uid, issue, method))
    total_reward = 0.0
    red_hits, blue_hits, levels = [], [], []
    for seq, ticket in enumerate(tickets, 1):
        level = EV.prize_level(ticket["reds"], ticket["blue"], draw["reds"], draw["blue"])
        red = len(set(ticket["reds"]) & set(draw["reds"]))
        blue = int(ticket["blue"] == draw["blue"])
        reward = float(EV.PRIZE_CASH.get(level, 0.0))
        total_reward += reward
        red_hits.append(red); blue_hits.append(blue); levels.append(level)
        conn.execute(
            """INSERT INTO eval_details
               (user_id,issue,method,seq,red_hits,blue_hit,prize_level,reward,
                ticket_cost,net_return,evaluated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, issue, method, seq, red, blue, level, reward, 2.0, reward - 2.0, now),
        )
    conn.commit()
    cost = len(tickets) * 2.0
    return {"issue": issue, "method": method, "tickets": len(tickets),
            "red_hits_mean": sum(red_hits) / max(1, len(red_hits)),
            "blue_hit_rate": sum(blue_hits) / max(1, len(blue_hits)),
            "best_level": max(levels) if levels else 0,
            "reward": total_reward, "cost": cost,
            "roi": (total_reward - cost) / max(1.0, cost)}


def load_eval_meta(issue: Optional[str] = None, *,
                   user_id: Optional[int] = None) -> List[Dict]:
    sql = "SELECT * FROM eval_meta WHERE user_id=?"
    args: List = [resolve_tenant(user_id)]
    if issue:
        sql += " AND issue=?"; args.append(issue)
    sql += " ORDER BY issue, method"
    return [dict(r) for r in get_conn().execute(sql, args).fetchall()]


def _mean_ci95(values: List[float]) -> Dict:
    """返回均值及正态近似 95% CI；样本不足时仍明确返回 n。"""
    import math
    n = len(values)
    if not n:
        return {"mean": None, "low": None, "high": None, "n": 0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "low": mean, "high": mean, "n": n}
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    margin = 1.96 * sd / math.sqrt(n)
    return {"mean": mean, "low": mean - margin, "high": mean + margin, "n": n}


def _rolling_issue_metrics(rows: List[Dict], windows=(10, 30, 60)) -> List[Dict]:
    by_issue: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        item = by_issue.setdefault(r["issue"], {"red": [], "blue": [], "roi": []})
        item["red"].append(float(r["red_hits"]))
        item["blue"].append(float(r["blue_hit"]))
        item["roi"].append((float(r["reward"]) - float(r["ticket_cost"])) /
                            max(1.0, float(r["ticket_cost"])))
    issues = sorted(by_issue)
    out = []
    for idx, issue in enumerate(issues):
        point = {"issue": issue}
        for window in windows:
            selected = issues[max(0, idx - window + 1):idx + 1]
            vals = {k: [v for i in selected for v in by_issue[i][k]] for k in ("red", "blue", "roi")}
            point[f"w{window}"] = {"red_hits": _mean_ci95(vals["red"]),
                                    "blue_hit_rate": _mean_ci95(vals["blue"]),
                                    "roi": _mean_ci95(vals["roi"])}
        out.append(point)
    return out


def cumulative_eval(method: Optional[str] = None, limit: int = 120) -> Dict:
    sql = "SELECT * FROM eval_details"
    args: List = []
    if method:
        sql += " WHERE method=?"; args.append(method)
    sql += " ORDER BY CAST(issue AS INTEGER) DESC, method, seq LIMIT ?"
    args.append(int(limit) * 50)
    rows = [dict(r) for r in get_conn().execute(sql, args).fetchall()]
    by_method: Dict[str, Dict] = {}
    for r in rows:
        m = r["method"]
        item = by_method.setdefault(m, {"method": m, "issues": set(), "tickets": 0, "red": [], "blue": [], "levels": [], "reward": 0.0, "cost": 0.0, "rows": []})
        item["issues"].add(r["issue"]); item["tickets"] += 1
        item["red"].append(r["red_hits"]); item["blue"].append(r["blue_hit"]); item["levels"].append(r["prize_level"])
        item["reward"] += r["reward"]; item["cost"] += r["ticket_cost"]; item["rows"].append(r)
    out = []
    for item in by_method.values():
        cost = item["cost"]
        out.append({"method": item["method"], "issues": len(item["issues"]), "tickets": item["tickets"],
                    "red_hits_mean": sum(item["red"]) / max(1, len(item["red"])),
                    "blue_hit_rate": sum(item["blue"]) / max(1, len(item["blue"])),
                    "prize_rate_ge5": sum(x >= 5 for x in item["levels"]) / max(1, len(item["levels"])),
                    "reward_total": item["reward"], "cost_total": cost,
                    "roi": (item["reward"] - cost) / max(1.0, cost),
                    "rows": list(reversed(item["rows"][-limit:]))})
    for item, group in zip(out, by_method.values()):
        item["rolling"] = _rolling_issue_metrics(group["rows"][-limit:])
    return {"methods": out, "sample_limit": limit,
            "ci": "normal_approx_95", "rolling_windows": [10, 30, 60]}


def _two_sided_sign_p(values: List[float]) -> Optional[float]:
    """无 scipy 依赖的双侧 sign-test p 值，忽略恰好相等的配对。"""
    signs = [1 if v > 0 else -1 for v in values if v != 0]
    n = len(signs)
    if not n:
        return None
    k = min(sum(s > 0 for s in signs), sum(s < 0 for s in signs))
    return min(1.0, 2.0 * sum(__import__("math").comb(n, i) for i in range(k + 1)) / (2 ** n))


def method_recommendations(limit: Optional[int] = None, min_sample: Optional[int] = None) -> Dict:
    """按期比较方法与 uniform 基线，输出 M4.2 保留/关闭建议。

    只有同一期同时存在两种方法且达到 min_sample 才给出建议；120 期规则只产生
    ``disable_candidate`` 提示，不会静默修改生产配置。奖励差异仅用于方法比较，
    不代表中奖概率或投资建议。
    """
    from . import config
    limit = int(limit if limit is not None else config.METHOD_RECOMMENDATION_DISABLE_ISSUES)
    min_sample = int(min_sample if min_sample is not None else config.METHOD_RECOMMENDATION_MIN_ISSUES)
    rows = [dict(r) for r in get_conn().execute(
        "SELECT issue, method, SUM(reward) reward, SUM(ticket_cost) cost, "
        "AVG(red_hits) red, AVG(blue_hit) blue FROM eval_details "
        "GROUP BY issue, method ORDER BY CAST(issue AS INTEGER) DESC").fetchall()]
    by_issue: Dict[str, Dict[str, Dict]] = {}
    for r in rows:
        by_issue.setdefault(r["issue"], {})[r["method"]] = r
    baseline = "uniform"
    methods = sorted({r["method"] for r in rows if r["method"] != baseline})
    recommendations = []
    for method in methods:
        pairs = [(d[method], d[baseline]) for d in by_issue.values()
                 if method in d and baseline in d]
        pairs = pairs[:int(limit)]
        diffs = [float(a["reward"] or 0) - float(b["reward"] or 0) for a, b in pairs]
        p = _two_sided_sign_p(diffs)
        n = len(pairs)
        status = "insufficient_sample"
        if n >= int(min_sample):
            status = "keep_or_research" if (p is not None and p < 0.05) else "monitor"
            if n >= 120 and (p is None or p >= 0.05) and sum(float(a["cost"] or 0) for a, _ in pairs) > 0:
                status = "disable_candidate"
        recommendations.append({"method": method, "baseline": baseline, "paired_issues": n,
                                "window": min(int(limit), n), "paired_sign_p": p,
                                "mean_reward_delta": (sum(diffs) / n if n else None),
                                "cost_total": sum(float(a["cost"] or 0) for a, _ in pairs),
                                "status": status,
                                "action": "手动关闭并保留研究模式" if status == "disable_candidate" else "继续累积样本"})
    return {"baseline": baseline, "limit": int(limit), "min_sample": int(min_sample),
            "recommendations": recommendations,
            "note": "paired sign-test 仅作运营筛查；样本不足不自动决策，系统不会自动修改方法开关。"}


def save_eval(issue: str, red_hits: float, blue_hit: float, reward: float,
              ticket_count: int, *, user_id: Optional[int] = None):
    """写入**组合级（portfolio）**逐期对照行。

    V10-04b（R1 P1-4）：``red_hits`` / ``blue_hit`` 现在保留浮点均值，
    不再 ``int()`` 截断（旧写法把 1.5 抹成 1，系统性低估命中）。
    ⚠️ 本表**不含 method 列**，一行 = 该期全部方法所有注的合计，
    因此它**不能**用于方法对比；方法战绩一律按族分列汇总
    （``evaluate.method_breakdown()`` / ``eval_details``）。
    """
    uid = resolve_tenant(user_id)
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO eval_results
           (user_id,issue,red_hits,blue_hit,reward,ticket_count,created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (uid, issue, float(red_hits), float(blue_hit), reward, ticket_count, time.time()),
    )
    conn.commit()


def load_eval_details(limit_issues: Optional[int] = None, *,
                      user_id: Optional[int] = None) -> List[Dict]:
    """读取逐注方法明细（``eval_details``，主键 ``(issue, method, seq)``）。

    ``limit_issues`` 只取最近 N 期（按数值期号倒序再正序返回），
    避免全表扫描拖慢报表。返回按 ``(issue, method, seq)`` 升序。
    """
    uid = resolve_tenant(user_id)
    conn = get_conn()
    if limit_issues:
        picked = [r["issue"] for r in conn.execute(
            "SELECT DISTINCT issue FROM eval_details WHERE user_id=? "
            "ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?", (uid, int(limit_issues))
        ).fetchall()]
        if not picked:
            return []
        marks = ",".join("?" for _ in picked)
        sql = (f"SELECT * FROM eval_details WHERE user_id=? AND issue IN ({marks}) "
               "ORDER BY issue, method, seq")
        return [dict(r) for r in conn.execute(sql, [uid] + picked).fetchall()]
    return [dict(r) for r in conn.execute(
        "SELECT * FROM eval_details WHERE user_id=? ORDER BY issue, method, seq",
        (uid,)).fetchall()]


def load_eval(*, user_id: Optional[int] = None) -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM eval_results WHERE user_id=? ORDER BY issue",
        (resolve_tenant(user_id),)).fetchall()
    return [dict(r) for r in rows]


# ---------- tasks ----------

def create_task(task_id: str, kind: str) -> None:
    now = time.time()
    get_conn().execute(
        "INSERT INTO tasks (id, kind, status, progress, message, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (task_id, kind, "pending", 0.0, "", now, now),
    )
    get_conn().commit()


def update_task(task_id: str, status: str, progress: float, message: str) -> None:
    get_conn().execute(
        "UPDATE tasks SET status=?, progress=?, message=?, updated_at=? WHERE id=?",
        (status, float(progress), message, time.time(), task_id),
    )
    get_conn().commit()


def complete_task(task_id: str, result_json: str) -> None:
    get_conn().execute(
        "UPDATE tasks SET status='completed', progress=1.0, result_json=?, updated_at=? WHERE id=?",
        (result_json, time.time(), task_id),
    )
    get_conn().commit()


def fail_task(task_id: str, message: str) -> None:
    get_conn().execute(
        "UPDATE tasks SET status='failed', message=?, updated_at=? WHERE id=?",
        (message, time.time(), task_id),
    )
    get_conn().commit()


def load_task(task_id: str) -> Optional[Dict]:
    row = get_conn().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except Exception:
            pass
    return d


def list_tasks(limit: int = 20) -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (int(limit),)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("result_json"):
            try:
                d["result"] = json.loads(d["result_json"])
            except Exception:
                pass
        out.append(d)
    return out


# ---------- LLM 离线评估（M3.1） ----------

def save_llm_eval_rows(run_id: str, created_at: str, window_issues: int,
                       tickets: int, seed: int, rows: List[Dict]) -> None:
    """写入一次 LLM 评估 run 的全部行（逐期明细 + 每通道汇总）。"""
    conn = get_conn()
    conn.execute("DELETE FROM llm_eval_results WHERE run_id=?", (run_id,))
    sql = (
        "INSERT INTO llm_eval_results"
        "(run_id,created_at,window_issues,tickets,channel,issue,red_hits,blue_hit,prize_level,roi,"
        "tokens,cost_usd,duration_ms,metrics_json,p_values_json,seed)"
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    for r in rows:
        conn.execute(sql, (
            run_id, created_at, window_issues, tickets, r["channel"], r.get("issue"),
            r.get("red_hits"), r.get("blue_hit"), r.get("prize_level", 0), r.get("roi", 0.0),
            r.get("tokens", 0), r.get("cost_usd", 0.0), r.get("duration_ms", 0),
            json.dumps(r.get("metrics_json") or {}, ensure_ascii=False),
            json.dumps(r.get("p_values_json") or {}, ensure_ascii=False),
            seed,
        ))
    conn.commit()


def latest_llm_eval_run() -> Optional[str]:
    row = get_conn().execute(
        "SELECT run_id FROM llm_eval_results WHERE issue IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def load_llm_eval_run(run_id: str) -> Optional[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM llm_eval_results WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    if not rows:
        return None
    first = rows[0]
    summary: Dict[str, Dict] = {}
    per_issue: Dict[str, List[Dict]] = {}
    for r in rows:
        ch = r["channel"]
        if r["issue"] is None:
            summary[ch] = {
                "metrics": json.loads(r["metrics_json"] or "{}"),
                "p_values": json.loads(r["p_values_json"] or "{}"),
                "tokens": r["tokens"] or 0,
                "cost_usd": r["cost_usd"] or 0.0,
                "duration_ms": r["duration_ms"] or 0,
            }
        else:
            per_issue.setdefault(ch, []).append({
                "issue": r["issue"],
                "red_hits": r["red_hits"],
                "blue_hit": r["blue_hit"],
                "roi": r["roi"],
            })
    emp = None
    for ch, v in summary.items():
        meta = (v.get("metrics") or {}).get("_meta")
        if meta and meta.get("llm_empty_issues") is not None:
            emp = meta["llm_empty_issues"]
    out = {
        "run_id": run_id,
        "created_at": first["created_at"],
        "window_issues": first["window_issues"],
        "tickets": first["tickets"],
        "seed": first["seed"],
        "summary": summary,
        "per_issue": per_issue,
    }
    if emp is not None:
        out["llm_empty_issues"] = emp
    return out


# ---------- 规律挖掘运行记录（M3.3） ----------

def save_mining_run(r: Dict) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO mining_runs "
        "(run_id,created_at,engine,candidates,accepted,avg_lift,pass_rate,duration_ms,params_json) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (r["run_id"], r["created_at"], r["engine"], int(r.get("candidates", 0)),
         int(r.get("accepted", 0)), float(r.get("avg_lift", 0.0)),
         float(r.get("pass_rate", 0.0)), int(r.get("duration_ms", 0)),
         json.dumps(r.get("params_json") or {}, ensure_ascii=False)))
    conn.commit()


def list_mining_runs(limit: int = 20) -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM mining_runs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["params"] = json.loads(d.pop("params_json") or "{}")
        except Exception:  # noqa: BLE001
            d["params"] = {}
        out.append(d)
    return out