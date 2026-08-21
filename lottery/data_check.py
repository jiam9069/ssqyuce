"""M4.4 多源开奖对账：比较期号与开奖号码，不修改主库。"""
from __future__ import annotations
from typing import Dict, Iterable, List
from . import config, parser
import requests


def fetch_source(url: str, timeout: int = 60) -> List[Dict]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = "utf-8"
    return parser.parse_text(response.text)


def reconcile(primary: Iterable[Dict], secondary: Iterable[Dict]) -> Dict:
    a = {d["issue"]: d for d in primary}
    b = {d["issue"]: d for d in secondary}
    common = sorted(set(a) & set(b))
    mismatches = []
    for issue in common:
        if a[issue]["reds"] != b[issue]["reds"] or a[issue]["blue"] != b[issue]["blue"]:
            mismatches.append({"issue": issue, "primary": {"reds": a[issue]["reds"], "blue": a[issue]["blue"]},
                               "secondary": {"reds": b[issue]["reds"], "blue": b[issue]["blue"]}})
    return {"primary_total": len(a), "secondary_total": len(b), "common": len(common),
            "only_primary": sorted(set(a) - set(b)), "only_secondary": sorted(set(b) - set(a)),
            "mismatches": mismatches, "ok": not mismatches and not (set(a) ^ set(b))}


def run(url: str | None = None, timeout: int = 60) -> Dict:
    secondary_url = url or config.BACKUP_DATA_URL
    if not secondary_url:
        return {"ok": False, "status": "not_configured", "error": "未配置 LOTT_BACKUP_DATA_URL"}
    primary = fetch_source(config.DATA_URL, timeout)
    secondary = fetch_source(secondary_url, timeout)
    result = reconcile(primary, secondary)
    result["secondary_url"] = secondary_url
    # 审计写入失败不应掩盖对账结果，也不应影响主数据流程。
    try:
        from . import db
        result["audit_id"] = db.save_reconcile_run(result, secondary_url)
    except Exception as exc:  # noqa: BLE001
        result["audit_error"] = str(exc)
    return result
