"""V40 覆盖率欠债 B 类：**I/O 与错误路径**的无直测模块。

## 这个文件为什么存在

`data_fetcher` / `data_check` / `cli` / `routes_*` 这些模块的**主路径**在测试里
根本不存在（全站 23 个模块零直测 import，见 `docs/reviews/DEBT_COVERAGE_REPORT.md`）。
它们真正会出事的不是"算错"，而是**边界行为**：

* 源站返回错误页 ⇒ 整批数据作废还是跳过坏行？
* 网络超时/500 ⇒ 抛什么？调用方能接住吗？
* 空响应 ⇒ 静默写 0 期，还是明确报错？
* 「新增 N 期」在重复抓取时会不会骗人？

这些路径**在正常网络下永远不会跑到**，所以只能靠打桩强制触发。

## 铁律遵循

* **禁止源码文本断言**：本文件全部是"调用函数 + 断言返回值/异常"。
* **错误路径断言的是"可读的失败"**，不是"抛了某个异常"就算数：
  每条都钉住异常类型 + 错误信息里的关键语义（调用方/用户要靠它排查）。
* 网络一律打桩（`monkeypatch.setattr(requests, "get")`），不打真实网络。

## 变异验证

见报告「变异验证记录」。要点：把 `raise_for_status()` 删掉、把
`inserted_new` 改成遍历计数、把空数据检查删掉 —— 对应断言都会变红。
"""
from __future__ import annotations

import pytest

#: 一行**合法**的 ssq.TXT 记录（31 字段，已实测解析通过）。
#: 期号刻意用 2099xx：绝不与生产库/合成 fixture 的期号段冲突。
_ROW = ("2099001 2025-11-12 03 07 08 15 21 30 9 3 7 8 15 21 30 "
        "412345678 567890123 2 10000000 8 200000 100 3000 1500 200 200 10 0 0 0 0")
_ROW_2 = ("2099002 2025-11-15 04 07 08 15 21 30 9 4 7 8 15 21 30 "
          "412345678 567890123 2 10000000 8 200000 100 3000 1500 200 200 10 0 0 0 0")


# ===========================================================================
# ① data_fetcher —— 网络边界 + 增量入库的诚实计数
# ===========================================================================

class _Resp:
    """最小 requests.Response 替身：只实现被测代码真正调用的两件事。"""

    def __init__(self, text: str = "", status: int = 200):
        self.text = text
        self.status_code = status
        self.encoding = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_fetch_text_sets_utf8_and_propagates_http_error(monkeypatch):
    """`fetch_text` 必须：① 显式设 utf-8（源站页面编码漂移时不乱码）；
    ② **传播** HTTP 错误（把 500 当成功会让下游把错误页当数据）。

    变体验证：删掉 `raise_for_status()` 后 `pytest.raises` 不再触发 ⇒ 本条红。
    """
    from lottery import data_fetcher

    monkeypatch.setattr(data_fetcher.requests, "get",
                        lambda url, timeout=60: _Resp("hello", 200))
    assert data_fetcher.fetch_text() == "hello"

    monkeypatch.setattr(data_fetcher.requests, "get",
                        lambda url, timeout=60: _Resp("oops", 500))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        data_fetcher.fetch_text()


def test_fetch_text_passes_timeout_through(monkeypatch):
    """超时必须**传给 requests**（默认 60s）。不传就是一个永不返回的挂起风险。"""
    from lottery import data_fetcher

    seen = {}

    def fake_get(url, timeout=60):
        seen["url"], seen["timeout"] = url, timeout
        return _Resp("x")

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    data_fetcher.fetch_text(timeout=7)
    assert seen["timeout"] == 7, "超时参数没有传到底层请求"
    data_fetcher.fetch_text()
    assert seen["timeout"] == 60, "默认超时应为 60s"


def test_save_snapshot_writes_into_isolated_raw_dir():
    """快照必须落在 `config.RAW_DIR` 下、内容逐字节一致。

    这是"保留原始快照"承诺的唯一实现点：写错目录就等于没有快照
    （而写进真实 `data/raw` 又会污染生产数据 —— 由 conftest 的隔离 fixture 兜住）。
    """
    from pathlib import Path

    from lottery import config, data_fetcher

    path = data_fetcher.save_snapshot("原始内容\n第二行")
    p = Path(path)
    assert p.parent == config.RAW_DIR, f"快照没写进 RAW_DIR：{p.parent}"
    assert p.read_text(encoding="utf-8") == "原始内容\n第二行"
    assert p.name.startswith("ssq_") and p.suffix == ".txt"


def test_update_from_text_is_idempotent_and_counts_honestly():
    """**重复抓取必须报 `inserted_new=0`**（这是 V38 修过的"新增 3518 期"假数字）。

    手算语义：
      第一次喂 1 期（新期号）⇒ inserted_new=1、updated_existing=0
      第二次喂同样 1 期        ⇒ inserted_new=0、updated_existing=1
      再喂 2 期（其中 1 期已存在）⇒ inserted_new=1、updated_existing=1
    变体验证：把 `inserted_new` 换回 `_det["processed"]`（遍历计数）⇒ 第二次会报 1 ⇒ 红。
    """
    from lottery import data_fetcher

    first = data_fetcher.update_from_text(_ROW)
    assert first["remote_total"] == 1
    assert first["remote_last"] == "2099001"
    assert first["remote_last_date"] == "2025-11-12"
    assert first["inserted_new"] == 1 and first["updated_existing"] == 0
    assert first["local_max"] == "2099001"

    second = data_fetcher.update_from_text(_ROW)
    assert second["inserted_new"] == 0, "重复抓取谎报新增期数（V38 回归）"
    assert second["updated_existing"] == 1
    assert second["local_max_before"] == second["local_max"] == "2099001"

    third = data_fetcher.update_from_text(_ROW + "\n" + _ROW_2)
    assert third["remote_total"] == 2
    assert third["inserted_new"] == 1, "只有 2099002 是新的"
    assert third["updated_existing"] == 1
    assert third["local_max"] == "2099002"


def test_update_from_text_raises_on_empty_payload():
    """空响应 ⇒ 明确 `ValueError("数据源为空")`。

    静默返回 0 期会让"刷新后一期都没有"变成一个无声的故障。
    """
    from lottery import data_fetcher

    with pytest.raises(ValueError, match="数据源为空"):
        data_fetcher.update_from_text("")
    with pytest.raises(ValueError, match="数据源为空"):
        data_fetcher.update_from_text("   \n\t\n  ")


def test_update_from_text_skips_bad_lines_but_reports_truthfully():
    """**坏行不整体作废**：能解析的入库，坏行跳过（V37 的健壮性修复）。

    旧实现遇到任意坏行就 raise ⇒ 源站多打一个空行、整批几千期全废，
    前端表现为"刷新后一期都没有"。这里验证：
      ① 混合文本（1 坏 + 1 好）⇒ remote_total=1（只数解析成功的）；
      ② 全是坏行 ⇒ 抛 ParseError（这更像取到了错误页，必须让调用方知道）。
    """
    from lottery import data_fetcher
    from lottery.parser import ParseError

    mixed = "这一行不是开奖数据\n" + _ROW
    info = data_fetcher.update_from_text(mixed)
    assert info["remote_total"] == 1, "坏行导致其他有效行被丢弃"
    assert info["remote_last"] == "2099001"

    with pytest.raises(ParseError, match="无法解析"):
        data_fetcher.update_from_text("<html>502 Bad Gateway</html>")


def test_update_from_text_marks_source_and_updates_max():
    """入库时必须打上 `source` 标记；`backfilled` 不得把**新增**误报成**补回**。

    手算：本地已有 2099001，源站给 2099001 + 2099002 ⇒ 都是新增/更新，
    `backfilled`（"源站有、而本次入库前本地本来没有"按 upsert 之后的本地集算）
    不该含 2099002。字符串序下 2099129 > 2099002，`local_max` 应推进。
    """
    from lottery import db, data_fetcher

    data_fetcher.update_from_text(_ROW)
    info = data_fetcher.update_from_text(_ROW + "\n" + _ROW_2)
    assert info["backfilled"] == [], f"新增期被误报成补回：{info['backfilled']}"
    assert info["backfilled_n"] == 0

    row_129 = _ROW.replace("2099001", "2099129")
    info2 = data_fetcher.update_from_text(row_129)
    assert info2["remote_total"] == 1
    assert info2["local_max"] == "2099129", "local_max 未按字符串序推进"

    rows = db.load_draws(limit=3)
    assert all(r.get("source") == "17500" for r in rows
               if str(r["issue"]).startswith("2099")), "入库记录未打 source 标记"


# ===========================================================================
# ② data_check —— 多源对账（错误的"成功"比失败更危险）
# ===========================================================================

def test_reconcile_ok_requires_identical_sets_and_numbers():
    """`ok` 只在**期号集合相同且号码全同**时为 True。

    四档手算：
      两侧相同                     → ok True
      只有一侧有某期               → ok False（only_primary/only_secondary 如实列出）
      同期红球顺序不同（[6..1]）   → list 逐位比较 ⇒ 视为不一致（对账口径）
      两侧皆空                     → ok True（对账无事可做不是失败）
    """
    from lottery.data_check import reconcile

    a = [{"issue": "1", "reds": [1, 2, 3, 4, 5, 6], "blue": 7}]
    assert reconcile(a, list(a))["ok"] is True

    only_p = reconcile(a, [])
    assert only_p["ok"] is False
    assert only_p["only_primary"] == ["1"] and only_p["only_secondary"] == []

    only_s = reconcile([], a)
    assert only_s["ok"] is False
    assert only_s["only_secondary"] == ["1"]

    reversed_reds = [{"issue": "1", "reds": [6, 5, 4, 3, 2, 1], "blue": 7}]
    res = reconcile(reversed_reds, a)
    assert res["ok"] is False, "红球顺序不同必须判为不一致（对账要求逐位相同）"
    assert res["mismatches"][0]["primary"]["reds"] == [6, 5, 4, 3, 2, 1]

    empty = reconcile([], [])
    assert empty["ok"] is True and empty["common"] == 0


def test_reconcile_reports_blue_only_mismatch():
    """只有蓝球不同也必须进 mismatches（不能只比红球）。"""
    from lottery.data_check import reconcile

    a = [{"issue": "1", "reds": [1, 2, 3, 4, 5, 6], "blue": 7}]
    b = [{"issue": "1", "reds": [1, 2, 3, 4, 5, 6], "blue": 8}]
    res = reconcile(a, b)
    assert res["ok"] is False
    assert len(res["mismatches"]) == 1
    assert res["mismatches"][0]["primary"]["blue"] == 7
    assert res["mismatches"][0]["secondary"]["blue"] == 8


def test_run_returns_not_configured_without_backup_url(monkeypatch):
    """未配置备用源 ⇒ `status="not_configured"` 且**不发任何请求**。"""
    from lottery import data_check

    calls = []

    def _no_request(url, timeout=60):  # pragma: no cover - 不应被调用
        calls.append(url)
        raise AssertionError("未配置备用源时不应发起请求")

    monkeypatch.setattr(data_check.requests, "get", _no_request)
    res = data_check.run(url=None)
    assert res["ok"] is False
    assert res["status"] == "not_configured"
    assert "LOTT_BACKUP_DATA_URL" in res["error"]
    assert calls == []


def test_run_detects_mismatch_and_writes_audit_row(monkeypatch):
    """`run` 必须：抓两个源 → 对账 → **把结果落审计表**并回传 audit_id。

    打桩让主源与备源各返回一份文本（备源红球差一个号）⇒ ok=False 且
    `audit_id` 有值。审计落库是"你说的是真的吗"这条链路的起点，不能漏。
    """
    from lottery import data_check, db

    responses = iter([_Resp(_ROW), _Resp(_ROW.replace(" 03 07 08", " 04 07 08"))])
    monkeypatch.setattr(data_check.requests, "get", lambda url, timeout=60: next(responses))

    res = data_check.run(url="http://backup.example/ssq.TXT")
    assert res["ok"] is False
    assert len(res["mismatches"]) == 1
    assert res["mismatches"][0]["primary"]["reds"][0] == 3
    assert res["secondary_url"] == "http://backup.example/ssq.TXT"
    assert isinstance(res["audit_id"], int) and res["audit_id"] > 0
    assert "audit_error" not in res

    rows = db.load_reconcile_runs()
    assert any(r["id"] == res["audit_id"] for r in rows), "对账结果没有落审计表"


def test_run_succeeds_when_both_sources_agree(monkeypatch):
    """两侧一致 ⇒ ok=True、mismatches 为空（正例，防"永远报不一致"）。"""
    from lottery import data_check

    monkeypatch.setattr(data_check.requests, "get",
                        lambda url, timeout=60: _Resp(_ROW))
    res = data_check.run(url="http://backup.example/ssq.TXT")
    assert res["ok"] is True
    assert res["common"] == 1
    assert res["mismatches"] == []


def test_fetch_source_propagates_http_error(monkeypatch):
    """`fetch_source` 必须传播 HTTP 错误（503 不得被当成"0 期数据"）。"""
    from lottery import data_check

    monkeypatch.setattr(data_check.requests, "get",
                        lambda url, timeout=60: _Resp("", 503))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        data_check.fetch_source("http://bad.example/ssq.TXT")


# ===========================================================================
# ③ cli —— 命令分发与自检的边界
# ===========================================================================

def test_cli_unknown_command_exits_nonzero():
    """未知子命令 ⇒ argparse 以退出码 2 结束（不是静默无操作）。"""
    from lottery import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["nonsense"])
    assert exc.value.code == 2


def test_cli_requires_a_subcommand():
    """不给子命令 ⇒ 同样 SystemExit(2)（`required=True`）。"""
    from lottery import cli

    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_cli_every_subcommand_has_a_handler():
    """**注册过的子命令必须都有对应处理函数**（防止加了 parser 忘了函数）。

    ⚠️ 本用例实测发现一个**存量缺陷**（V40 报告已记录，修复留给 lead 裁决）：
    `main()` 的分发逻辑是 `globals().get("cmd_" + args.cmd)`，而
    `online_check` 子命令对应的函数名是 `cmd_online`（不带 `_check`）
    ⇒ **`lottery online_check` 实际会报"未知命令 online_check"并退出码 2**
    （隔离环境实测复现）。函数名历史上就与命令名不一致，本用例**按现状**
    钉住映射 —— 修复时应把函数改名为 `cmd_online_check`（或分发处做映射），
    然后同步更新这里的表。
    """
    from lottery import cli

    # parser 名 → 处理函数名的**现有**映射（读源码逐个确认过）
    handler_of = {
        "fetch": "cmd_fetch", "stats": "cmd_stats", "backtest": "cmd_backtest",
        "predict": "cmd_predict", "offline_eval": "cmd_offline_eval",
        "online_check": "cmd_online", "mine": "cmd_mine",
        "import_mining": "cmd_import_mining", "doctor": "cmd_doctor",
        "serve": "cmd_serve",
    }
    missing = [cmd for cmd, fn in handler_of.items()
               if not callable(getattr(cli, fn, None))]
    assert not missing, f"这些子命令没有处理函数：{missing}"


def test_cli_online_check_dispatch_is_broken_today():
    """`lottery online_check` 必须可分发（V40 修复后的行为锁）。

    历史：V40 审计实测复现了分发 bug —— `main()` 按 `cmd_<子命令名>` 找
    处理函数，而函数叫 `cmd_online`，用户敲 `lottery online_check`
    （--help 里显示的命令名）直接报「未知命令 online_check」（退出码 2）。
    修复（lead 泳道）：`lottery/cli.py` 在 `cmd_online` 后加别名
    `cmd_online_check = cmd_online`，一行补上 --help 承诺的命令名。

    本用例从「钉住坏行为」翻转为「锁住修复」：
    - 分发必须成功到达处理函数（不再 SystemExit(2)）；
    - 且输出为 JSON 摘要（可解析）—— 防止别名指到一个空函数。
    """
    import io
    import json as _json
    from contextlib import redirect_stdout

    from lottery import cli

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            cli.main(["online_check"])
    except SystemExit as exc:
        if exc.code == 2:
            raise AssertionError(
                "lottery online_check 无法分发（exit 2）—— "
                "cmd_online_check 别名丢失？见 lottery/cli.py 的修复注释。")
        # 其他 SystemExit 视为业务层退出，仍校验已产出的 stdout
    out = buf.getvalue()
    # stdout 前几行可能是 config 加载日志（`[config] ...`），JSON 后面还跟着
    # 人类可读明细行（"2026119 红命中1.4 …"）—— **有已结算数据时明细非空**。
    # 第一版用 json.loads(out[find("{"):])，单跑（空库、明细为空）绿、
    # 全量跑（库里前序测试写过数据、明细非空）必红 —— 又一个"只在真空里绿"
    # 的断言（Extra data: line 58 column 3）。改用 raw_decode：只解析
    # 首个完整 JSON 值，尾部明细不影响。
    json_start = out.find("{")
    assert json_start != -1, f"online_check 的 stdout 里没有任何 JSON 摘要：{out[:200]!r}"
    summary, _end = _json.JSONDecoder().raw_decode(out[json_start:])
    assert "checked" in summary, f"JSON 摘要缺 checked 键：{list(summary)[:8]}"


def test_cli_doctor_reports_ok_and_warn_lists():
    """`doctor` 返回 `{"ok": int, "warn": [...]}`，且 warn 项都是非空字符串。

    不断言具体条数（依赖环境里 web/vendor 是否存在），只钉**结构契约**：
    运维/部署脚本要解析这个返回值。
    """
    import argparse

    from lottery import cli

    res = cli.cmd_doctor(argparse.Namespace())
    assert set(res) == {"ok", "warn"}
    assert isinstance(res["ok"], int) and res["ok"] >= 0
    assert isinstance(res["warn"], list)
    for w in res["warn"]:
        assert isinstance(w, str) and w.strip(), f"空警告项：{w!r}"


# ===========================================================================
# ④ routes_* —— 只补「有分支」的，不补纯透传
# ===========================================================================
#
# 说明（报告里有完整理由）：`routes_audit / routes_diagnosis / routes_eval /
# routes_llm_ops / routes_rules / routes_methods` 等模块的端点几乎都已被
# HTTP 级测试打到（实测命中数见报告表），因此这里**不重复**。
# 下面只补三个「有真实分支但从未被触达」的：
#   * `routes_ml`        —— 空库必须 400 而不是 500（前端要能提示"先抓数据"）
#   * `routes_llm_eval`  —— 前置校验（期数不足 / LLM 未配置）
#   * `routes_patterns`  —— `_pattern_summary` 的 A/B/C 计数

def test_routes_ml_status_on_empty_db_is_not_ready():
    """空库时 `/api/ml/status` 必须给出 `ready=False` + 可读 `reason`（不是抛异常）。"""
    from lottery import routes_ml

    res = routes_ml.ml_status()
    assert res["ready"] is False
    assert isinstance(res["reason"], str) and res["reason"].strip()


def test_routes_ml_eval_rejects_empty_db_with_actionable_400():
    """空库跑 ML 评估 ⇒ 400 + **可操作**的 error（告诉用户"先抓数据"）。

    500 会让前端只能显示"服务器错误"，用户不知道下一步做什么。
    """
    from lottery import routes_ml

    res = routes_ml.ml_eval(window=30, refit_every=15, train_window=800)
    assert res.status_code == 400
    body = res.body.decode("utf-8")
    assert "还没有开奖数据" in body
    assert "抓取" in body or "导入" in body, f"错误里没给出下一步：{body}"


def test_routes_llm_eval_latest_returns_none_status_when_absent():
    """没有离线评估报告 ⇒ `{"status": "none"}`（前端据此显示"尚未运行"）。"""
    from lottery import routes_llm_eval

    assert routes_llm_eval.llm_eval_latest() == {"status": "none"}


def test_routes_llm_eval_run_guards_short_history(seeded_db):
    """期数不足 305 ⇒ 400 且 error 里**带上实际期数**（便于判断还差多少）。

    用 `seeded_db(100)` 造 100 期：既覆盖"有数据但不够"这条分支，
    又不必等真实数据。
    """
    from lottery import routes_llm_eval

    seeded_db(100)
    res = routes_llm_eval.llm_eval_run(issues=60, n=5, seed=0)
    assert res.status_code == 400
    body = res.body.decode("utf-8")
    assert "305" in body, f"门槛值未出现在错误里：{body}"


def test_routes_llm_eval_run_requires_llm_configured(seeded_db, monkeypatch):
    """期数够但 LLM 未配置 ⇒ 400，且提示"先在设置页配置"（不是静默跳过 LLM 通道）。

    ⚠️ 环境依赖说明：本机 `data/llm_config.json` 里可能有已配置的真实通道
    （`config.LLM_BASE_URL/LLM_API_KEY` 模块全局**不在** conftest 隔离名单），
    所以"LLM 未配置"这个前提必须**显式制造**——直接关掉 LLM_DISABLED，
    并清掉三个已配置字段。这也是 V40 审计发现的 conftest 缺口的实证：
    若不显式打桩，这条用例在本机会静默走"已配置"分支而测不到目标分支。
    """
    from lottery import config, routes_llm_eval

    seeded_db(320)
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    res = routes_llm_eval.llm_eval_run(issues=60, n=5, seed=0)
    assert res.status_code == 400
    body = res.body.decode("utf-8")
    assert "LLM 通道未配置" in body
    assert "设置" in body, f"没告诉用户去哪里配置：{body}"


@pytest.mark.parametrize("rows,expected", [
    ([], {"A": 0, "B": 0, "C": 0}),
    ([{"grade": "A"}, {"grade": "A"}, {"grade": "C"}], {"A": 2, "B": 0, "C": 1}),
    ([{"grade": "B"}], {"A": 0, "B": 1, "C": 0}),
    ([{"no_grade": 1}], {"A": 0, "B": 0, "C": 1}),          # 缺 grade ⇒ 归 C
])
def test_routes_patterns_summary_counts(rows, expected):
    """`_pattern_summary` 的 A/B/C 计数（缺 grade 归 C，不得静默丢弃）。

    手算：空集 → 全 0；两 A 一 C → A=2/C=1；缺键行按 C 计。
    """
    from lottery import routes_patterns

    assert routes_patterns._pattern_summary(rows) == expected


def test_routes_patterns_get_returns_items_and_summary():
    """`GET /api/patterns` 的返回体必须同时有 items 与 summary。"""
    from lottery import routes_patterns

    body = routes_patterns.get_patterns()
    assert set(body) == {"items", "summary"}
    assert isinstance(body["items"], list)
    assert set(body["summary"]) == {"A", "B", "C"}
