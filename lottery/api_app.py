"""FastAPI 应用：REST API + 静态单页前端。"""
from __future__ import annotations

import json
import hmac
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

class _NoCacheStaticFiles(StaticFiles):
    """静态资源禁用缓存，确保前端更新后浏览器立即拿到新版。"""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

from . import config, data_fetcher, db
from . import features as F, diagnose as D, mining as M

app = FastAPI(title="双色球智能预测分析系统", version=config.APP_VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def optional_api_auth(request: Request, call_next):
    """LOTT_TOKEN 配置后保护 /api/*；静态前端和健康探活保持可加载。"""
    if config.API_TOKEN and request.url.path.startswith("/api/"):
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {config.API_TOKEN}"
        if not hmac.compare_digest(auth, expected):
            return JSONResponse({"ok": False, "error": "需要 Bearer Token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- 数据 ----------

@app.get("/api/data/reconcile")
def data_reconcile():
    from . import data_check
    return data_check.run()


@app.get("/api/data/reconcile/history")
def reconcile_history(limit: int = Query(30, ge=1, le=500)):
    """返回多源对账审计记录，便于追踪告警与恢复。"""
    return {"items": db.load_reconcile_runs(limit)}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "issues": db.count_draws(),
        "max_issue": db.max_issue(),
        "eval_methods": [r["method"] for r in db.cumulative_eval(limit=1).get("methods", [])],
        "app_build": config.APP_BUILD,
        "uptime_seconds": max(0, int(__import__('time').time() - config.STARTED_AT)),
        "api_auth_enabled": bool(config.API_TOKEN),
        "recent_tasks": db.list_tasks(limit=5),
    }


@app.get("/api/info")
def app_info():
    """系统版本与里程碑状态（主页页脚徽标与「关于」展示用）。"""
    return {
        "name": "双色球智能预测分析系统",
        "version": config.APP_VERSION,
        "build": config.APP_BUILD,
        "milestones": config.APP_MILESTONES,
        "plan": "docs/UPGRADE_PLAN.md + docs/M3_M4_PLAN.md",
    }


@app.post("/api/refresh")
def refresh():
    """抓取远程最新开奖并增量入库（返回统计信息）。"""
    try:
        info = data_fetcher.fetch_and_update()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return {"ok": True, **info}


@app.get("/api/draws/latest")
def latest_draws(n: int = Query(20, ge=1, le=200)):
    draws = db.load_last_draws(n)
    return [{
        "issue": d["issue"], "date": d["date"], "reds": d["reds"], "blue": d["blue"],
        "sum": sum(d["reds"]),
    } for d in draws]


@app.get("/api/draws/history")
def draw_history(n: int = Query(500, ge=1, le=3490)):
    """返回按序号排列的开奖历史（用于走势图）。"""
    draws = db.load_last_draws(n)
    return {
        "issues": [d["issue"] for d in draws],
        "sums": [sum(d["reds"]) for d in draws],
        "blues": [d["blue"] for d in draws],
        "reds": [d["reds"] for d in draws],
    }


@app.get("/api/features")
def get_features():
    """最新一期的多尺度统计报告（含长/中/短窗口）。"""
    draws = db.load_draws()
    return F.compute_features(draws)


# ---------- 规律 ----------

@app.get("/api/patterns")
def get_patterns():
    return {
        "items": db.load_patterns(),
        "summary": _pattern_summary(db.load_patterns(grade_filter=None)),
    }


def _pattern_summary(patterns: List[Dict]) -> Dict:
    g = {"A": 0, "B": 0, "C": 0}
    for p in patterns:
        g[p.get("grade", "C")] = g.get(p.get("grade", "C"), 0) + 1
    return {"A": g["A"], "B": g["B"], "C": g["C"]}


@app.post("/api/patterns/backtest")
def run_backtests():
    from . import backtest as BT
    draws = db.load_draws()
    results = BT.run_all_backtests(draws, min_start=300)
    return {
        "items": db.load_patterns(),
        "summary": _pattern_summary(results),
        "note": ("walk-forward 样本外回测：只用目标期之前的历史，禁用未来信息。"
                 "A=显著，B=弱信号，C=不通过。"),
    }


# ---------- 预测 ----------

@app.api_route("/api/predict", methods=["GET", "POST"])
def predict(use_llm: Optional[bool] = None, n_tickets: int = 10,
            regenerate: bool = False):
    """生成下一期预测；目标期已有预测且未要求重新生成时复用缓存。"""
    from . import backtest as BT, engine
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地暂无开奖数据，请先刷新"}, status_code=400)
    issue = BT.next_issue(draws[-1]["issue"])
    if not regenerate:
        existing = db.load_predictions(issue)
        if existing:
            return {"issue": issue, "tickets": existing, "from_cache": True,
                    "llm_used": any(t["method"] == "llm" for t in existing)}
    res = engine.predict_next(draws, use_llm=use_llm, n_tickets=n_tickets)
    res["from_cache"] = False
    return res


@app.get("/api/predictions/last")
def last_predictions():
    issue = None
    rows = db.get_conn().execute(
        "SELECT issue FROM predictions ORDER BY issue DESC LIMIT 1").fetchone()
    if rows:
        issue = rows["issue"]
    return {"issue": issue, "tickets": db.load_predictions(issue) if issue else []}


# ---------- 评估 ----------

@app.get("/api/eval")
def eval_view():
    rows = db.load_eval()
    return rows


@app.post("/api/eval/backtest")
def eval_backtest(issues: int = Query(120), n: int = Query(10)):
    from . import evaluate
    draws = db.load_draws()
    res = evaluate.offline_backtest(draws, issues=min(issues, 200), n_tickets=n,
                                    use_llm=False)
    return res


@app.post("/api/eval/online")
def eval_online():
    from . import evaluate, notify
    result = evaluate.online_check()
    result["notification"] = notify.notify_after_check(result)
    return result


@app.get("/api/notify/status")
def notify_status():
    from . import notify
    return notify.status()


@app.get("/api/eval/cumulative")
def eval_cumulative(method: Optional[str] = Query(None), limit: int = Query(120, ge=1, le=1000)):
    """M4.1：按预测方法返回逐注事实与累计统计。"""
    return db.cumulative_eval(method=method, limit=limit)


@app.get("/api/eval/meta")
def eval_meta(issue: Optional[str] = Query(None)):
    return {"items": db.load_eval_meta(issue)}


@app.get("/api/eval/recommendations")
def eval_recommendations(limit: Optional[int] = Query(None, ge=1, le=1000),
                        min_sample: Optional[int] = Query(None, ge=1, le=1000)):
    """M4.2：按期开出的方法与 uniform 基线做 paired 筛查，输出运营建议。"""
    return db.method_recommendations(limit=limit, min_sample=min_sample)


@app.get("/api/eval/export.csv")
def eval_export_csv(method: Optional[str] = Query(None), limit: int = Query(1000, ge=1, le=5000)):
    import csv
    import io
    from fastapi.responses import StreamingResponse
    report = db.cumulative_eval(method=method, limit=limit)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["issue", "method", "seq", "red_hits", "blue_hit", "prize_level", "reward", "ticket_cost", "net_return", "evaluated_at"])
    for group in report["methods"]:
        for row in group.get("rows", []):
            writer.writerow([row.get(k, "") for k in ("issue", "method", "seq", "red_hits", "blue_hit", "prize_level", "reward", "ticket_cost", "net_return", "evaluated_at")])
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=eval_cumulative.csv"})


# ---------- M3.1 LLM 离线评估 ----------

@app.post("/api/eval/llm/run")
def llm_eval_run(issues: int = Query(60, ge=3, le=200),
                 n: int = Query(5, ge=1, le=20),
                 seed: Optional[int] = Query(None, ge=0)):
    """M3.1 LLM 离线评估：三通道 walk-forward（stat / stat+llm / random）后台任务。"""
    import threading
    import uuid
    from . import llm_eval
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地暂无开奖数据"}, status_code=400)
    if len(draws) < 305:
        return JSONResponse({"ok": False, "error": f"数据不足（需 ≥305 期，当前 {len(draws)}）"},
                            status_code=400)
    if not config.llm_configured():
        return JSONResponse({"ok": False,
                             "error": "LLM 通道未配置，无法运行 stat_llm 对比（请先在设置页配置并保存）"},
                            status_code=400)
    task_id = f"llm_eval_{uuid.uuid4().hex[:8]}"
    db.create_task(task_id, "llm_eval")

    def _work():
        try:
            db.update_task(task_id, "running", 0.01, "三通道 walk-forward 评估准备中...")
            res = llm_eval.run_llm_eval(
                draws, issues=issues, n_tickets=n, seed=seed,
                progress_cb=lambda p, m: db.update_task(task_id, "running", round(p, 4), m))
            db.complete_task(task_id, json.dumps(res, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            db.fail_task(task_id, str(e))

    threading.Thread(target=_work, daemon=True).start()
    return {"ok": True, "task_id": task_id,
            "note": ("后台任务运行中（60 期 × 三通道，LLM 通道较慢，约 5~30 分钟）。"
                     "轮询 /api/tasks/<task_id> 完成后查看 /api/eval/llm/latest。")}


@app.get("/api/eval/llm/latest")
def llm_eval_latest():
    from . import llm_eval
    report = llm_eval.load_latest_report()
    if report is None:
        return {"status": "none"}
    return {"status": "ready", "report": report}


# ---------- 任务系统 ----------

@app.get("/api/tasks")
def list_tasks(limit: int = Query(20, ge=1, le=100)):
    return db.list_tasks(limit)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    t = db.load_task(task_id)
    if t is None:
        return JSONResponse({"ok": False, "error": "task not found"}, status_code=404)
    return t


# ---------- 诊断 ----------

@app.get("/api/diagnose")
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
        return JSONResponse({"ok": False, "error": "暂无数据"}, status_code=400)
    return D.diagnose(r_list, blue, draws)


# ---------- 挖掘 ----------

@app.post("/api/mining/run")
def run_mining(min_start: int = Query(300, ge=120, le=1000),
               engine: str = Query("rf")):
    from . import backtest as BT
    import uuid
    task_id = f"mine_{uuid.uuid4().hex[:8]}"
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "暂无数据"}, status_code=400)
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
    except Exception as e:
        db.fail_task(task_id, str(e))
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/mining/latest")
def get_latest_mining():
    return M.get_latest_mining_result() or {"status": "none"}


# ---------- 回放诊断（M3.5） ----------

@app.post("/api/replay/diagnose")
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


@app.get("/api/mining/reports")
def mining_reports(limit: int = Query(20, ge=1, le=100)):
    """挖掘运行历史（M3.3）：engine/候选数/通过率/lift/耗时 等运行记录。"""
    return {"runs": db.list_mining_runs(limit)}


# ---------- 历史预测 ----------

@app.get("/api/predictions/history")
def predictions_history(limit: int = Query(50, ge=1, le=200)):
    issues = db.recent_prediction_issues(limit)
    draws_map = {d["issue"]: d for d in db.load_draws()}
    out = []
    for issue in issues:
        preds = db.load_predictions(issue)
        actual = draws_map.get(issue)
        if actual and preds:
            from . import evaluate
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



# ---------- LLM 配置管理（前台「设置」页，写入 data/llm_config.json） ----------

@app.get("/api/config/llm")
def get_llm_config():
    from . import config
    return {
        "disabled": config.LLM_DISABLED,
        "base_url": config.LLM_BASE_URL,
        "model": config.LLM_MODEL,
        "samples": config.LLM_SAMPLES,
        "configured": config.llm_configured(),
    }


@app.post("/api/config/llm")
def update_llm_config(payload: dict):
    from . import config
    import os
    if "base_url" in payload:
        config.LLM_BASE_URL = (payload.get("base_url") or "").strip().rstrip("/") or None
        os.environ["LOTT_LLM_BASE_URL"] = config.LLM_BASE_URL or ""
    if "api_key" in payload:
        config.LLM_API_KEY = (payload.get("api_key") or "").strip() or None
        os.environ["LOTT_LLM_API_KEY"] = config.LLM_API_KEY or ""
    if "model" in payload:
        config.LLM_MODEL = (payload.get("model") or "").strip() or "minimax-m3"
        os.environ["LOTT_LLM_MODEL"] = config.LLM_MODEL
        if config.LLM_MODEL not in config.LLM_MODEL_LIST:
            config.LLM_MODEL_LIST = [config.LLM_MODEL] + config.LLM_MODEL_LIST
    if "samples" in payload:
        try:
            config.LLM_SAMPLES = max(1, min(20, int(payload["samples"])))
        except (TypeError, ValueError):
            config.LLM_SAMPLES = 3
        os.environ["LOTT_LLM_SAMPLES"] = str(config.LLM_SAMPLES)
    if "disabled" in payload:
        config.LLM_DISABLED = bool(payload["disabled"])
        os.environ["LOTT_LLM_DISABLED"] = "1" if config.LLM_DISABLED else "0"

    # 持久化到数据目录（挂载卷，容器重启不丢失）
    config.LLM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    conf = {
        "disabled": config.LLM_DISABLED,
        "base_url": config.LLM_BASE_URL or "",
        "api_key": config.LLM_API_KEY or "",
        "model": config.LLM_MODEL or "",
        "samples": config.LLM_SAMPLES,
    }
    try:
        config.LLM_CONFIG_FILE.write_text(
            json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return {"ok": False, "error": f"配置写入失败: {e}"}
    return {"ok": True, "configured": config.llm_configured()}



@app.post("/api/llm/test")
def test_llm_connection():
    """用当前配置发起一次最小对话，验证 LLM 通道连通性。"""
    from . import config, llm_client
    import time
    if config.LLM_DISABLED:
        return {"ok": False, "error": "LLM 已停用（在设置中启用后重试）"}
    cfgs = config.llm_model_list()
    if not cfgs:
        return {"ok": False, "error": "LLM 未配置（请先填写 API 地址 / Key / 模型并保存）"}
    cfg = cfgs[0]
    t0 = time.time()
    try:
        text = llm_client.chat(
            "你是连接测试助手。", "请只回复：连接成功",
            max_tokens=50, temperature=0.0, timeout=25, model_cfg=cfg)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"调用异常: {e}"}
    dt_ms = int((time.time() - t0) * 1000)
    if text:
        return {"ok": True, "time_ms": dt_ms, "reply": text.strip()[:100]}
    return {"ok": False, "error": "模型无返回（请检查 API 地址 / Key / 模型名）"}


# ---------- M4.2 方法 A/B 开关（Web 设置页配置，写入 data/methods_config.json） ----------

@app.get("/api/methods/status")
def methods_status():
    """当前方法开关：模式 / 原始字符串 / 解析规格 / 方法注册表 / 各方法族开关状态。"""
    from . import methods as METH
    st = config.methods_status()
    eff = METH.effective_spec(config.METHOD_MODE, config.METHODS_SPEC)
    st["effective"] = {"mode": eff.get("mode", "all"),
                       "tokens": sorted(eff.get("tokens", set()))}
    st["registry"] = METH.registry()
    st["families"] = {fam: METH.is_enabled(fam, eff) for fam in METH.FAMILIES}
    return st


@app.post("/api/methods/config")
def update_methods_config(payload: dict):
    """更新方法开关：{methods, mode}，立即生效并持久化，影响之后的预测。

    methods: LOTT_METHODS 字符串（逗号/空格分隔；- 前缀 = 关闭；留空 = 全部启用）；
    mode: production（严格按开关过滤）/ research（忽略开关、全部启用）。
    """
    from . import methods as METH
    raw = payload.get("methods")
    mode = payload.get("mode")
    if raw is not None and not isinstance(raw, str):
        return JSONResponse({"ok": False, "error": "methods 必须是字符串"},
                            status_code=400)
    if raw is not None:
        err = METH.validate_raw(raw)
        if err:
            return JSONResponse({"ok": False, "error": err}, status_code=400)
    if mode is not None:
        m = str(mode).strip().lower() if isinstance(mode, str) else None
        if m not in METH.MODES:
            return JSONResponse(
                {"ok": False, "error": f"mode 必须是 {'/'.join(METH.MODES)}，收到 {mode!r}"},
                status_code=400)
        mode = m
    try:
        st = config.set_methods(raw, mode)
    except (ValueError, OSError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, **st}


# ---------- M2 ML 概率模型 ----------

@app.get("/api/ml/status")
def ml_status():
    from . import ml_model
    draws = db.load_draws()
    return ml_model.ml_status(draws)


@app.post("/api/ml/eval")
def ml_eval(window: int = Query(30, ge=10, le=200),
            refit_every: int = Query(15, ge=1, le=50),
            train_window: int = Query(800, ge=200, le=3490)):
    """ML 概率模型 walk-forward 滚动评估：Brier/log-loss/校准曲线 + paired 检验。

    注意：重训使用最近 train_window 期历史（默认 800），通常耗时 2~5 分钟。
    """
    from . import ml_model
    draws = db.load_draws()
    if not draws:
        return JSONResponse({"ok": False, "error": "本地暂无开奖数据"}, status_code=400)
    return ml_model.evaluate_ml_walkforward(
        draws, window=window, refit_every=refit_every, train_window=train_window)


# ---------- 页面 ----------



@app.get("/")
def index():
    resp = FileResponse(WEB_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ---------- 开奖日自动调度（可选，LOTT_SCHEDULER=1） ----------

def _scheduler_step():
    """开奖日（二/四/日）21:35 后：抓取 → 在线对照 → 生成下期预测。"""
    from . import backtest as BT, engine, evaluate
    now = __import__("datetime").datetime.now()
    if now.weekday() not in config.DRAW_WEEKDAYS:
        return
    if now.strftime("%H:%M") < config.DRAW_TIME:
        return
    try:
        info = data_fetcher.fetch_and_update()
        print("[scheduler] 数据已更新:", info.get("inserted_new"), "期")
    except Exception as e:  # noqa: BLE001
        print("[scheduler] 抓取失败:", e)
        return
    check_result = evaluate.online_check()
    from . import notify
    print("[scheduler] 通知结果:", notify.notify_after_check(check_result))
    draws = db.load_draws()
    issue = BT.next_issue(draws[-1]["issue"])
    if not db.load_predictions(issue):
        engine.predict_next(draws, use_llm=True)
        print("[scheduler] 已生成", issue, "预测")


def _scheduler_loop():
    import time as _time
    while True:
        try:
            _scheduler_step()
        except Exception as e:  # noqa: BLE001
            print("[scheduler] 异常:", e)
        _time.sleep(1800)


# M2：后台预热 ML 模型（首个预测请求不阻塞；训练结果按数据版本缓存）
try:
    from . import ml_model
    _boot_draws = db.load_draws()
    ml_model.start_ml_warmup(_boot_draws)
except Exception:  # noqa: BLE001
    pass

if config.SCHEDULER_ENABLED:
    import threading
    threading.Thread(target=_scheduler_loop, daemon=True).start()