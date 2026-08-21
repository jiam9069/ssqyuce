"""数据采集：下载 ssq.TXT，增量入库，保留原始快照。"""
from __future__ import annotations

import datetime
from typing import List, Optional

import requests

from . import config, db, parser


def fetch_text(timeout: int = 60) -> str:
    r = requests.get(config.DATA_URL, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def save_snapshot(text: str) -> str:
    fname = config.RAW_DIR / f"ssq_{datetime.date.today().isoformat()}.txt"
    fname.write_text(text, encoding="utf-8")
    return str(fname)


def update_from_text(text: str) -> dict:
    """解析并增量入库，返回统计信息。"""
    draws = parser.parse_text(text)
    for draw in draws:
        draw["source"] = "17500"
    if not draws:
        raise ValueError("数据源为空")
    local_max = db.max_issue()
    new_draws = [d for d in draws if local_max is None or d["issue"] > local_max]
    n_new = db.upsert_draws(new_draws) if new_draws else 0
    return {
        "remote_total": len(draws),
        "remote_last": draws[-1]["issue"],
        "remote_last_date": draws[-1]["date"],
        "local_total": db.count_draws(),
        "local_max": db.max_issue(),
        "inserted_new": n_new,
    }


def fetch_and_update() -> dict:
    text = fetch_text()
    snapshot = save_snapshot(text)
    info = update_from_text(text)
    info["snapshot"] = snapshot
    return info


def load_local(limit: Optional[int] = None):
    return db.load_draws(limit=limit)