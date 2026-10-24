"""任务系统 + 诊断 + 挖掘 + 历史预测 路由（P2 域切分）。

这四块放同一个模块，是因为它们的共同点是**都只依赖 db / mining / diagnose**，
没有相互调用、也不碰租户状态或中间件。拆成 4 个文件会得到 4 个几行的模块，
反而增加导航成本 —— 域模块的粒度应该跟**依赖簇**走，不是跟 URL 前缀走。

任务系统（`/api/tasks*`）值得注意的是 MU-2 的**可见性**问题：
`db.list_tasks` / `db.load_task` 目前按 id 取、不做 user 过滤。
后台任务的 `result_json` 可能含票单，属于用户私有数据。
`tasks.user_id` 列已经加好了（迁移 8），真正的按用户过滤放在断言里说明为什么
现在还没启用 —— 见 `_task_visible_to_current_user`。
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import db
from . import diagnose as D
from . import mining as M

router = APIRouter()


# ---------- 任务系统 ----------

def _task_visible_to_current_user(task: Optional[dict]) -> bool:
    """任务是否对当前请求者可见（V36 修复，2026-10-24 审计 P0-2）。

    ## 曾经的写法（缺陷）

    V35 及以前本函数恒 `return True`，注释解释为"任务列表是管理员运维视图，
    普通用户拿不到，所以过滤等于没有过滤"。**这个推理是错的**：
    `/api/tasks` 在 ADMIN_EXACT 里，但 `/api/tasks/{task_id}` **不在**
    （`required_role` 按精确路径匹配，带 id 的路径落到 `"user"` 兜底）。
    结果任何登录用户只要知道 task_id（`{kind}_{8hex}`，32 bit 可枚举）
    就能读到别人 `result_json` 里的票单。审计已用 HTTP 实测复现（uid B 读 uid A → 200）。

    ## 判定规则

    * 任务**无归属**（`user_id` 为 NULL/0）：视为系统/调度器产出，登录用户可见；
      这与 `db.resolve_tenant` 的"显式 → 上下文 → 主管理员 → SYSTEM"不冲突，
      因为 NULL 恰好表示"没有发起人"。
    * 任务**有归属**：只有归属者本人可见。
    * 无请求上下文（后台/测试直调）：只允许无归属任务，**绝不放行**有归属的。

    管理员不在此处特判：`/api/tasks` 列表页已是 admin-only，单条读取按归属走，
    管理员若需运维全景，应走列表接口而不是穿透单条。
    """
    if task is None:
        return False
    owner = task.get("user_id")
    if owner in (None, 0, ""):
        return True
    from . import request_context
    uid = request_context.current_user_id()
    if uid is None:
        return False
    try:
        return int(owner) == int(uid)
    except (TypeError, ValueError):
        return False


@router.get("/api/tasks")
def list_tasks(limit: int = Query(20, ge=1, le=100)):
    return db.list_tasks(limit)


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    t = db.load_task(task_id)
    if t is None or not _task_visible_to_current_user(t):
        return JSONResponse({"ok": False, "error": "找不到该任务。请确认任务编号是否正确，或刷新任务列表后重试。"}, status_code=404)
    return t


# ---------- 诊断 ----------

@router.get("/api/diagnose")
def diagnose_endpoint(
    reds: str = Query(..., description="逗号分隔的6个红球，如 1,4,7,12,28,31"),
    blue: int = Query(..., ge=1, le=16),
):
    try:
        r_list = [int(x) for x in reds.split(",")]
    except ValueError:
        return JSONResponse({"ok": False, "error": "reds 必须是逗号分隔的整数"}, status_code=400)
    if len(r_list) != 6 or len(set(r_list)) != 6:
        return JSONResponse({"ok": False, "error": "必须恰好 6 个不重复红球"}, status_code=400)
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地还没有开奖数据。请先抓取或导入历史开奖（详见部署文档「数据准备」一节），完成后再试。"}, status_code=400)
    return D.diagnose(r_list, blue, draws)


# ---------- 挖掘 ----------

@router.post("/api/mining/run")
def run_mining(min_start: int = Query(300, ge=120, le=1000),
               engine: str = Query("rf")):
    import json
    import uuid
    from . import backtest as BT
    task_id = f"mine_{uuid.uuid4().hex[:8]}"
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地还没有开奖数据。请先抓取或导入历史开奖（详见部署文档「数据准备」一节），完成后再试。"}, status_code=400)
    db.create_task(task_id, "mine")
    # 同步执行（挖掘较快，通常 <10s）
    try:
        db.update_task(task_id, "running", 0.1, "正在计算特征...")
        result = M.run_mining(draws, min_start=min_start, save_to_db=True, engine=engine)
        db.update_task(task_id, "completed", 1.0, "完成")
        db.complete_task(task_id, json.dumps(result, ensure_ascii=False))
        # 同时刷新规律列表
        BT.run_all_backtests(draws, min_start=min_start)
        return {"ok": True, "task_id": task_id, "result": result}
    except Exception as e:  # noqa: BLE001
        db.fail_task(task_id, str(e))
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/api/replay/diagnose")
def replay_diagnose(n: int = Query(20, ge=3, le=60), use_ml: bool = Query(True)):
    """最近 N 期「系统预测 vs 实际开奖」反事实回放（纯统计+ML，固定种子可复现）。"""
    import random
    import numpy as np
    from . import engine as E, evaluate as EV
    draws = db.load_draws()
    if len(draws) < n + 305:
        return JSONResponse({"ok": False, "error": f"数据不足（需 ≥{n + 305} 期）"},
                            status_code=400)
    rng = random.Random(20260817)
    results, rows = [], []
    for i in range(len(draws) - n, len(draws)):
        history = draws[:i]
        target = draws[i]
        res = E.predict_next(history, use_llm=False, n_tickets=10, persist=False,
                             use_ml=use_ml, rng=rng)
        tr = EV.tickets_result(res["tickets"], target)
        results.append(tr)
        rows.append({
            "issue": target["issue"], "date": target["date"],
            "red_hits_mean": round(float(np.mean(tr["red_hits"])), 3),
            "blue_hit": int(sum(tr["blue_hits"])),
            "prize_level": tr["best_level"],
            "reward": round(tr["reward"], 2),
        })
    agg = EV.aggregate(results)
    return {"ok": True, "use_ml": use_ml, "n": len(rows), "rows": rows,
            "aggregate": agg,
            "note": ("最近 N 期反事实回放：每期只用该期之前数据预测（纯统计+ML，固定种子可复现），"
                     "对照实际开奖。LLM 通道不参与（成本与噪声考虑）。")}


@router.get("/api/mining/reports")
def mining_reports(limit: int = Query(20, ge=1, le=100)):
    """挖掘运行历史（M3.3）：engine/候选数/通过率/lift/耗时 等运行记录。"""
    return {"runs": db.list_mining_runs(limit)}


# ---------- 历史预测 ----------

@router.get("/api/predictions/history")
def predictions_history(limit: int = Query(50, ge=1, le=200)):
    """历史预测与对应开奖对照。

    MU-2：`recent_prediction_issues` / `load_predictions` 都已按租户过滤
    （`db.resolve_tenant()` 取当前用户），所以这里天然只返回**我自己的**票。
    不需要在本函数里再拼一遍 user 条件 —— 那样反而容易与 db 层的口径漂移。
    """
    from . import evaluate
    issues = db.recent_prediction_issues(limit)
    draws_map = {d["issue"]: d for d in db.load_draws()}
    out = []
    for issue in issues:
        preds = db.load_predictions(issue)
        actual = draws_map.get(issue)
        if actual and preds:
            res = evaluate.tickets_result(preds, actual)
            out.append({
                "issue": issue,
                "date": actual["date"],
                "actual": {"reds": actual["reds"], "blue": actual["blue"]},
                "predictions": preds,
                "result": res,
            })
        elif preds:
            out.append({"issue": issue, "predictions": preds})
    return out
