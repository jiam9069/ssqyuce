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
    """解析并增量入库，返回统计信息。

    **不只按 max_issue 追加**：源站偶尔会晚于开奖（或本地曾漏抓一期），
    此时源站的上一期会在本地**缺失**，而 `d["issue"] > local_max` 会把它永远挡在门外。
    V7 改为用 `upsert_draws` 全量对齐（同 issue 覆盖、缺的补齐），并用
    `backfilled` 报出「补回了几期历史缺号」——这是「目标期号不准」的根治措施。
    """
    draws = parser.parse_text(text)
    for draw in draws:
        draw["source"] = "17500"
    if not draws:
        raise ValueError("数据源为空")
    local_max_before = db.max_issue()
    # V38（P2）：必须区分「新插入」与「覆盖已有」。
    # 旧写法拿 upsert_draws 的遍历计数当"新增期数"，重复抓同一份数据时
    # 会报"新增 3518 期"而实际零变化，直接误导"数据是否落后"的判断。
    _det = db.upsert_draws_detailed(draws)
    n_new = _det["inserted"]
    remote_issues = {d["issue"] for d in draws}
    # 「补回了几期」= 源站有、而本次入库前本地本来没有的期号
    local_before = {d["issue"] for d in db.load_draws()}
    backfilled = sorted(remote_issues - local_before)
    local_max = db.max_issue()
    return {
        "remote_total": len(draws),
        "remote_last": draws[-1]["issue"],
        "remote_last_date": draws[-1]["date"],
        "local_total": db.count_draws(),
        "local_max": local_max,
        "local_max_before": local_max_before,
        "inserted_new": n_new,
        "updated_existing": _det["updated"],
        # 这批里本地原先**没有**的期号（含补回的漏抓期）
        "backfilled": backfilled,
        "backfilled_n": len(backfilled),
    }


def fetch_and_update() -> dict:
    text = fetch_text()
    snapshot = save_snapshot(text)
    info = update_from_text(text)
    info["snapshot"] = snapshot
    return info


def load_local(limit: Optional[int] = None):
    return db.load_draws(limit=limit)