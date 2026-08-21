"""M4.3 开奖通知：Webhook / 企业微信 / SMTP，失败不阻塞主流程。"""
from __future__ import annotations

import json
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Dict, List

from . import config, db


def _draw_text(draw: Dict) -> str:
    return f"{draw.get('issue', '')}：{' '.join(f'{int(x):02d}' for x in draw.get('reds', []))} + {int(draw.get('blue', 0)):02d}"


def _summary_text(result: Dict) -> str:
    rows = result.get("rows") or []
    if not rows:
        return "本次未发现可对照的历史预测。"
    latest = rows[-1]
    return (f"最近对照 {result.get('newly_checked', 0)} 期；"
            f"最新期 {latest.get('issue', '')}，红球命中 {latest.get('red_hits', '-') }，"
            f"蓝球 {'命中' if latest.get('blue_hit') else '未中'}，奖金 ¥{float(latest.get('reward', 0) or 0):.0f}。")


def _post_json(url: str, payload: Dict, timeout: float = 10.0) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")


def _send_webhook(text: str) -> None:
    url = config.NOTIFY_WEBHOOK
    if not url:
        return
    # 企业微信机器人需要 msgtype/text；Server酱兼容 webhook 可直接接收 text。
    payload = {"msgtype": "text", "text": {"content": text}}
    _post_json(url, payload)


def _send_email(subject: str, text: str) -> None:
    if not (config.NOTIFY_SMTP_HOST and config.NOTIFY_EMAIL_TO and config.NOTIFY_EMAIL_FROM):
        return
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.NOTIFY_EMAIL_FROM, config.NOTIFY_EMAIL_TO
    msg.set_content(text)
    with smtplib.SMTP(config.NOTIFY_SMTP_HOST, config.NOTIFY_SMTP_PORT, timeout=15) as smtp:
        if config.NOTIFY_SMTP_TLS:
            smtp.starttls()
        if config.NOTIFY_SMTP_USER:
            smtp.login(config.NOTIFY_SMTP_USER, config.NOTIFY_SMTP_PASSWORD)
        smtp.send_message(msg)


def notify_after_check(result: Dict) -> Dict:
    """开奖对照后发送通知；每个通道独立失败并返回状态，不抛出异常。"""
    if not result.get("newly_checked"):
        return {"sent": False, "reason": "no_new_evaluation", "channels": {}}
    rows = result.get("rows") or []
    issue = rows[-1].get("issue") if rows else db.max_issue()
    draws = {d["issue"]: d for d in db.load_draws()}
    draw = draws.get(issue, {"issue": issue, "reds": [], "blue": 0})
    text = f"双色球开奖对照通知\n开奖：{_draw_text(draw)}\n{_summary_text(result)}"
    statuses: Dict[str, Dict] = {}
    for name, fn in (("webhook", lambda: _send_webhook(text)),
                     ("email", lambda: _send_email(f"双色球开奖对照 {issue}", text))):
        configured = (name == "webhook" and bool(config.NOTIFY_WEBHOOK)) or (name == "email" and bool(config.NOTIFY_EMAIL_TO))
        if not configured:
            statuses[name] = {"configured": False, "sent": False}
            continue
        try:
            fn(); statuses[name] = {"configured": True, "sent": True}
        except Exception as exc:  # noqa: BLE001
            print(f"[notify] {name} 发送失败（已静默降级）: {exc}")
            statuses[name] = {"configured": True, "sent": False, "error": str(exc)}
    return {"sent": any(x.get("sent") for x in statuses.values()), "issue": issue, "channels": statuses}


def status() -> Dict:
    return {"webhook_configured": bool(config.NOTIFY_WEBHOOK),
            "email_configured": bool(config.NOTIFY_EMAIL_TO and config.NOTIFY_SMTP_HOST),
            "smtp_host": config.NOTIFY_SMTP_HOST or None,
            "email_to": config.NOTIFY_EMAIL_TO or None}
