"""SQLite 存取层：开奖数据、特征、规律回测、预测、在线评估。"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Dict, Iterable, List, Optional

from . import config

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
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(date);

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

CREATE TABLE IF NOT EXISTS predictions (
  issue TEXT NOT NULL,
  seq INTEGER NOT NULL,
  reds TEXT NOT NULL,            -- "1,4,7,12,28,31"
  blue INTEGER NOT NULL,
  confidence REAL NOT NULL,
  method TEXT NOT NULL,          -- stat / llm / mixed
  reasoning TEXT DEFAULT '',
  patterns_used TEXT DEFAULT '',
  evidence_json TEXT DEFAULT '',
  created_at REAL,
  PRIMARY KEY (issue, seq)
);
CREATE INDEX IF NOT EXISTS idx_pred_issue ON predictions(issue);

CREATE TABLE IF NOT EXISTS eval_results (
  issue TEXT PRIMARY KEY,
  red_hits INTEGER,
  blue_hit INTEGER,
  reward REAL DEFAULT 0,
  ticket_count INTEGER DEFAULT 0,
  created_at REAL
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

_CONN: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.executescript(SCHEMA)
        _ensure_columns(_CONN)
        _CONN.commit()
    return _CONN


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """存量库幂等补列（M3.2：predictions.evidence_json）。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "evidence_json" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN evidence_json TEXT DEFAULT ''")
        conn.commit()


def close():
    global _CONN
    if _CONN is not None:
        _CONN.close()
        _CONN = None


# ---------- draws ----------

def upsert_draws(draws: Iterable[Dict]) -> int:
    conn = get_conn()
    n = 0
    now = time.time()
    for d in draws:
        conn.execute(
            """INSERT OR REPLACE INTO draws
               (issue,date,r1,r2,r3,r4,r5,r6,blue,o1,o2,o3,o4,o5,o6,
                sales,pool,p1,p1_amt,p2,p2_amt,p3,p3_amt,p4,p4_amt,p5,p5_amt,p6,p6_amt,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d["issue"], d["date"], *d["reds"], d["blue"], *d["order"],
                d.get("sales", 0), d.get("pool", 0),
                *d.get("prizes", [0] * 12)[0:2], *d.get("prizes", [0] * 12)[2:4],
                *d.get("prizes", [0] * 12)[4:6], *d.get("prizes", [0] * 12)[6:8],
                *d.get("prizes", [0] * 12)[8:10], *d.get("prizes", [0] * 12)[10:12],
                now,
            ),
        )
        n += 1
    conn.commit()
    return n


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
        "SELECT * FROM draws ORDER BY issue DESC LIMIT ?", (int(n),)
    ).fetchall()
    return [_row_to_draw(r) for r in reversed(rows)]


def _row_to_draw(r: sqlite3.Row) -> Dict:
    return {
        "issue": r["issue"], "date": r["date"],
        "reds": [r[f"r{i}"] for i in range(1, 7)],
        "blue": r["blue"],
        "order": [r[f"o{i}"] for i in range(1, 7)],
        "sales": r["sales"], "pool": r["pool"],
    }


# ---------- features ----------

def save_features(issue: str, payload: dict):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO features (issue, payload, created_at) VALUES (?,?,?)",
        (issue, json.dumps(payload, ensure_ascii=False), time.time()),
    )
    conn.commit()


def load_features(issue: Optional[str] = None) -> Optional[Dict]:
    if issue is None:
        row = get_conn().execute("SELECT payload FROM features ORDER BY issue DESC LIMIT 1").fetchone()
    else:
        row = get_conn().execute("SELECT payload FROM features WHERE issue=?", (issue,)).fetchone()
    return json.loads(row["payload"]) if row else None


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


# ---------- predictions ----------

def save_predictions(issue: str, tickets: List[Dict]):
    conn = get_conn()
    conn.execute("DELETE FROM predictions WHERE issue=?", (issue,))
    now = time.time()
    for i, t in enumerate(tickets, 1):
        conn.execute(
            """INSERT INTO predictions
               (issue,seq,reds,blue,confidence,method,reasoning,patterns_used,evidence_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                issue, i, ",".join(map(str, t["reds"])), t["blue"], t["confidence"],
                t.get("method", "mixed"), t.get("reasoning", ""),
                json.dumps(t.get("patterns_used", []), ensure_ascii=False),
                json.dumps(t.get("evidence", {}), ensure_ascii=False), now,
            ),
        )
    conn.commit()


def load_predictions(issue: str) -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM predictions WHERE issue=? ORDER BY seq", (issue,)
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


def load_last_predictions() -> List[Dict]:
    row = get_conn().execute(
        "SELECT issue FROM predictions ORDER BY issue DESC LIMIT 1"
    ).fetchone()
    return load_predictions(row["issue"]) if row else []


def recent_prediction_issues(n: int = 20) -> List[str]:
    rows = get_conn().execute(
        "SELECT DISTINCT issue FROM predictions ORDER BY issue DESC LIMIT ?", (int(n),)
    ).fetchall()
    return [r["issue"] for r in rows]


# ---------- eval ----------

def save_eval(issue: str, red_hits: int, blue_hit: int, reward: float, ticket_count: int):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO eval_results (issue,red_hits,blue_hit,reward,ticket_count,created_at)
           VALUES (?,?,?,?,?,?)""",
        (issue, red_hits, int(blue_hit), reward, ticket_count, time.time()),
    )
    conn.commit()


def load_eval() -> List[Dict]:
    rows = get_conn().execute("SELECT * FROM eval_results ORDER BY issue").fetchall()
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
    return {
        "run_id": run_id,
        "created_at": first["created_at"],
        "window_issues": first["window_issues"],
        "tickets": first["tickets"],
        "seed": first["seed"],
        "summary": summary,
        "per_issue": per_issue,
    }


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