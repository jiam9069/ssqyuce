"""V10-11：成本口径（读 usage 的账本）+ 「AI 边际价值」+ LLM 决策技能关停。

规格来源
--------
* ``docs/reviews/R3_AI_REVIEW.md`` **G6**（三个真实问题）：
  1. ``llm_client.chat()`` **完全没读** ``data["usage"]`` → 成本分母不可信；
  2. 关停不能复用 ``LLM_DISABLED``（那会让 Web 预测直接 503）；
  3. 关停必须可持久（不能写 ``os.environ``），且必须走 ``resolve_channels()``。
* ``docs/V10_IMPROVEMENT_PLAN.md`` V10-11 小节。

本文件守的事（对应任务书 5 条核心测试）
---------------------------------------
① ``decision_disabled=true`` 时**预测仍成功**、只是不含该通道（防误伤的核心里）；
② usage 解析正确 + **缺失即 None 而不是 0**（0 会污染均值）；
③ 成本日志**不含密钥**（``sk-`` / ``api_key`` / base_url 一个字都不许出现）；
④ **重启后关停状态仍生效**（真起一个子进程重新 import config 验证）；
⑤ Wilson 下界判据的边界（恰好 0.5 / 略低 / 略高 / 无样本）；
⑥ 成本报表零调用时返回 **null 而非 0**。

另有：4 个出口全覆盖记账、重试累加、日志轮转、恢复入口、schema 只增不改、
自动关停只凭"有决定性样本"（无数据不关停）。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ==================== 公共夹具与工具 ====================

@pytest.fixture()
def env(isolated_env, monkeypatch):
    """隔离环境 + 干净 LLM 配置（库/配置文件都在 tmp，绝不碰生产 data/）。"""
    from lottery import config, db
    monkeypatch.setenv("LOTT_HOME", str(isolated_env.base))
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test-SECRET-KEY")
    monkeypatch.setattr(config, "LLM_MODEL", "fake-model")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["fake-model"])
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])
    monkeypatch.delenv("LOTT_LLM_PRICE_PER_1M_YUAN", raising=False)
    db.close()
    return isolated_env


class _Resp:
    """伪造的 HTTP 响应。"""

    def __init__(self, status: int = 200, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)[:200]

    def json(self):
        return self._payload


def _ok_payload(content: str = '{"ok": true}', tokens=(1000, 500)):
    payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if tokens is not None:
        payload["usage"] = {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}
    return payload


def _patch_post(monkeypatch, responses):
    """把 ``requests.post`` 换成按顺序返回 responses 的假函数（列表末尾重复）。"""
    from lottery import llm_client
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        idx = min(len(calls) - 1, len(responses) - 1)
        r = responses[idx]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    return calls


def _cost_log_rows():
    from lottery import llm_client
    return llm_client.read_cost_log()


def _write_llm_config(**kwargs):
    from lottery import config
    config.LLM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.LLM_CONFIG_FILE.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2),
                                      encoding="utf-8")


def _mk_eval_rows(llm_hits, base_hits, *, n_tickets=5, start_issue=2026001):
    """造 eval_details 行：第 i 期 llm 臂平均命中 llm_hits[i]、基线臂 base_hits[i]。"""
    rows = []
    for i, (lh, bh) in enumerate(zip(llm_hits, base_hits)):
        issue = str(start_issue + i)
        for seq in range(1, n_tickets + 1):
            rows.append({"issue": issue, "method": "llm:fake", "seq": seq,
                         "red_hits": int(lh), "blue_hit": 0, "prize_level": 0,
                         "reward": 0.0, "ticket_cost": 2.0})
            rows.append({"issue": issue, "method": "stat:freq", "seq": seq,
                         "red_hits": int(bh), "blue_hit": 0, "prize_level": 0,
                         "reward": 0.0, "ticket_cost": 2.0})
    return rows


# ==========================================================================
# ① 4 个出口统一记账 + usage 解析（含缺失 → None）
# ==========================================================================

def test_success_exit_records_usage_and_yuan(env, monkeypatch):
    """出口 1（正常返回）：读到 usage，token 与估算金额都落库。"""
    from lottery import llm_client as L

    _patch_post(monkeypatch, [_Resp(200, _ok_payload(tokens=(1200, 300)))])
    assert L.chat("s", "u", purpose="lab") == '{"ok": true}'

    rows = _cost_log_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["tokens_in"] == 1200 and row["tokens_out"] == 300
    assert row["purpose"] == "lab" and row["ok"] is True and row["model"] == "fake-model"
    assert row["attempts"] == 1
    # 金额 = 1500 tokens / 1e6 × 单价（单价假设随行落库，可事后重算）
    assert row["yuan"] == pytest.approx(1500 / 1e6 * row["price_per_1m_yuan"], rel=1e-9)
    assert row["yuan"] > 0


def test_missing_usage_degrades_to_none_not_zero(env, monkeypatch):
    """出口 1 但上游**没返回 usage**：token/yuan 必须是 None，绝不能被写成 0。"""
    from lottery import llm_client as L

    _patch_post(monkeypatch, [_Resp(200, _ok_payload(tokens=None))])
    assert L.chat("s", "u", purpose="decision") == '{"ok": true}'

    row = _cost_log_rows()[0]
    assert row["tokens_in"] is None and row["tokens_out"] is None
    assert row["yuan"] is None, "缺失 usage 时写 0 会把均值系统性拉低"
    report = L.cost_report()
    assert report["n_calls"] == 1 and report["n_billed"] == 0
    assert report["yuan"] is None and report["tokens_total"] is None


def test_parse_usage_edge_cases(env):
    """usage 解析的边界：缺字段/非 dict/字符串数字/负数/bool 一律不猜。"""
    from lottery import llm_client as L

    assert L.parse_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}) == {
        "tokens_in": 5, "tokens_out": 7, "has_usage": True}
    assert L.parse_usage({"usage": {"prompt_tokens": "42"}})["tokens_in"] == 42
    for bad in ({}, {"usage": None}, {"usage": "x"}, {"usage": {"prompt_tokens": -1}},
                {"usage": {"completion_tokens": True}}, None, []):
        parsed = L.parse_usage(bad)
        assert parsed["tokens_in"] is None and parsed["tokens_out"] is None
        assert parsed["has_usage"] is False


def test_all_failure_exits_are_logged(env, monkeypatch):
    """出口 2/3/4 + 未配置 + 已停用：**每一个**出口都要落一行（不允许静默出口）。"""
    import requests as _requests

    from lottery import config, llm_client as L

    # 出口 2：已停用（config.LLM_DISABLED）
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    assert L.chat("s", "u", purpose="p1") is None
    monkeypatch.setattr(config, "LLM_DISABLED", False)

    # 出口：未配置（不发请求）
    monkeypatch.setattr(config, "LLM_BASE_URL", None)
    assert L.chat("s", "u", purpose="p2") is None
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://api.example.com/v1")

    # 出口 3：超时 → 非 strict 降级返回 None（重试后仍失败）
    _patch_post(monkeypatch, [_requests.exceptions.ReadTimeout("read timed out")])
    assert L.chat("s", "u", purpose="p3") is None

    # 出口 4：strict 抛错（5xx 会退避重试，测试里把 sleep 置空，避免白等 12s）
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    _patch_post(monkeypatch, [_Resp(500, {"error": "boom"})])
    with pytest.raises(L.LLMChannelError):
        L.chat("s", "u", purpose="p4", strict=True)

    rows = _cost_log_rows()
    assert [r["purpose"] for r in rows] == ["p1", "p2", "p3", "p4"]
    assert all(r["ok"] is False for r in rows)
    assert all(r["tokens_in"] is None and r["yuan"] is None for r in rows)
    # 真的发出过请求的出口要记录尝试次数：
    # 读取超时 ∈ {3 次重试，外加一次 thinking:disabled 抬预算重试} → 4 次。
    assert rows[2]["attempts"] >= 3
    assert rows[3]["attempts"] >= 1


def test_retry_accumulates_usage_into_one_row(env, monkeypatch):
    """同一 chat() 内的重试**累加** usage 且只落一行（含 attempts）。"""
    from lottery import llm_client as L

    _patch_post(monkeypatch, [_Resp(429, {"error": "rate"}), _Resp(200, _ok_payload())])
    monkeypatch.setattr(L.time, "sleep", lambda s: None)      # 免等待

    assert L.chat("s", "u", purpose="lab") == '{"ok": true}'
    rows = _cost_log_rows()
    assert len(rows) == 1, "一次 chat() 只应有一行（否则成本会重复计）"
    row = rows[0]
    assert row["attempts"] == 2 and row["ok"] is True
    assert row["tokens_in"] == 1000 and row["tokens_out"] == 500


def test_purpose_explicit_wins_and_inferred_from_caller(env, monkeypatch):
    """``purpose=`` 显式优先；缺省用调用方模块名兜底（不逼其它模块改签名）。"""
    from lottery import llm_client as L

    _patch_post(monkeypatch, [_Resp(200, _ok_payload())])
    L.chat("s", "u", purpose="explicit-tag")
    # 无 purpose：调用方是本测试模块 → 标签应为 test_v10_cost（或至少非 unspecified）
    L.chat("s", "u")
    rows = _cost_log_rows()
    assert rows[0]["purpose"] == "explicit-tag"
    assert rows[1]["purpose"] not in ("", "unspecified")


# ==========================================================================
# ③ 成本日志不含密钥 + 轮转
# ==========================================================================

def test_cost_log_never_contains_secrets(env, monkeypatch):
    """日志只记账本字段：``sk-`` / ``api_key`` / base_url / Authorization 一个字都不许有。"""
    from lottery import llm_client as L

    monkeypatch.setattr(L.time, "sleep", lambda s: None)   # 5xx 退避重试不必真等
    _patch_post(monkeypatch, [_Resp(200, _ok_payload())])
    L.chat("s", "u", purpose="lab")
    _patch_post(monkeypatch, [_Resp(500, {"error": "boom"})])
    L.chat("s", "u", purpose="lab")
    L.chat("sk-should-not-leak", "u", purpose="lab")

    raw = L.cost_log_path().read_text(encoding="utf-8")
    assert raw.strip(), "日志应为空的话本测试没有意义"
    for secret in ("sk-test-SECRET-KEY", "api_key", "base_url", "Authorization",
                   "Bearer", "api.example.com", "magpie"):
        assert secret not in raw, f"成本日志泄露了 {secret!r}"
    # 每个必需字段都在
    for row in _cost_log_rows():
        assert set(("model", "tokens_in", "tokens_out", "purpose", "yuan", "ts", "ok")) <= set(row)


def test_cost_log_rotates_and_caps_disk(env, monkeypatch):
    """append-only 但**按大小轮转**：单文件不超上限，只保留 1 份历史。"""
    from lottery import llm_client as L

    monkeypatch.setattr(L, "COST_LOG_MAX_BYTES", 800)
    monkeypatch.setattr(L, "COST_LOG_ROTATE_KEEP", 1)
    _patch_post(monkeypatch, [_Resp(200, _ok_payload())])
    for i in range(60):
        L.chat("s", "u", purpose=f"p{i}")

    path = L.cost_log_path()
    rotated = Path(str(path) + ".1")
    assert path.exists() and rotated.exists()
    assert path.stat().st_size <= 800 + 400, "轮转后当前文件必须被截断到上限附近"
    # 轮转不丢"总量"太多：两份加起来的行数应远小于 60 但大于等于 1 份
    assert len(_cost_log_rows()) < 60
    assert len(_cost_log_rows()) >= 1


# ==========================================================================
# ④ 成本报表：零调用返回 null 而非 0 + 分组
# ==========================================================================

def test_cost_report_zero_calls_returns_null_not_zero(env):
    """零调用：金额/token 一律 null（不是 0），并给出原因。"""
    from lottery import llm_client as L

    report = L.cost_report()
    assert report["available"] is False
    assert report["n_calls"] == 0
    assert report["yuan"] is None and report["tokens_in"] is None
    assert report["tokens_total"] is None
    assert report["by_model"] == {} and report["by_purpose"] == {}
    assert report["reason_zh"]

    mv = __import__("lottery.api_app", fromlist=["x"]).llm_marginal_value(report)
    assert mv["yuan"] is None
    assert mv["per_yuan_usable_decisions"] is None
    assert mv["per_yuan_passed_patterns"] is None
    assert "null" in mv["reason_zh"] or "没有" in mv["reason_zh"]


def test_cost_report_groups_by_model_and_purpose(env, monkeypatch):
    """按 model 与 purpose 分组，且每组各自计数。"""
    from lottery import config, llm_client as L

    _patch_post(monkeypatch, [_Resp(200, _ok_payload(tokens=(100, 100)))])
    L.chat("s", "u", purpose="lab")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["other-model"])
    monkeypatch.setattr(config, "LLM_MODEL", "other-model")
    L.chat("s", "u", purpose="decision")
    L.chat("s", "u", purpose="decision")

    report = L.cost_report()
    assert report["n_calls"] == 3 and report["n_billed"] == 3
    assert set(report["by_model"]) == {"fake-model", "other-model"}
    assert report["by_model"]["other-model"]["n_calls"] == 2
    assert report["by_purpose"]["decision"]["n_calls"] == 2
    assert report["tokens_in"] == 300 and report["tokens_out"] == 300


def test_api_cost_endpoint_returns_report_and_marginal_value(env, client):
    """``GET /api/llm/cost``：零数据时 200 且 nullable 字段为 null。"""
    r = client.get("/api/llm/cost")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["cost"]["available"] is False
    assert body["cost"]["yuan"] is None
    assert body["marginal_value"]["per_yuan_usable_decisions"] is None
    assert "decision" in body and body["decision"]["decision_disabled"] in (False, True)
    json.dumps(body)          # 必须可序列化（无 numpy 对象）


def test_marginal_value_zero_when_cost_but_no_numerator(env, client, monkeypatch):
    """有成本但分子为 0 → **0.0**（真实结论）；分母缺失 → None。两者必须能区分。"""
    from lottery import llm_client as L
    from lottery.api_app import llm_marginal_value

    _patch_post(monkeypatch, [_Resp(200, _ok_payload())])
    L.chat("s", "u", purpose="lab")
    mv = llm_marginal_value()
    assert mv["yuan"] is not None and mv["yuan"] > 0
    # 干净的 tmp 环境里：没有 claim、没有通过的规律 → 分子 0 → 比值 0.0
    assert mv["numerator_usable_decisions"]["n_usable"] == 0
    assert mv["per_yuan_usable_decisions"] == 0.0
    assert mv["per_yuan_passed_patterns"] in (0.0, None)


# ==========================================================================
# ⑤ Wilson 判据边界
# ==========================================================================

def test_wilson_boundary_exactly_at_baseline():
    """判据边界：恰好 0.5 → 关停（≤）；略低 → 关停；略高 → 不关停；无样本 → None。"""
    from lottery.api_app import decide_from_wilson

    assert decide_from_wilson(0.5)["should_disable"] is True
    lo = math.nextafter(0.5, 0.0)
    hi = math.nextafter(0.5, 1.0)
    assert decide_from_wilson(lo)["should_disable"] is True
    assert decide_from_wilson(hi)["should_disable"] is False
    none_case = decide_from_wilson(None)
    assert none_case["should_disable"] is None and none_case["wilson_lower"] is None
    # 基线可调（0.5 是"无技能"的默认值，不硬编码在比较里）
    assert decide_from_wilson(0.6, baseline=0.55)["should_disable"] is False


def test_wilson_lower_matches_hypothesis_implementation():
    """同一口径只能有一个公式：与 ``hypothesis._wilson_lower`` 数值一致。"""
    from lottery import hypothesis
    from lottery.api_app import _wilson_lower

    for wins, n in ((0, 1), (1, 2), (3, 5), (17, 20), (0, 30), (30, 30), (1, 59)):
        assert _wilson_lower(wins, n) == pytest.approx(
            hypothesis._wilson_lower(wins, n), rel=0, abs=1e-12)
    assert _wilson_lower(0, 0) == 0.0


def test_wilson_lower_pinned_to_official_reference_values():
    """**V40 真值锁**：Wilson 下界必须命中**外部推导的参考值**。

    ## 为什么需要这条（变异实测教训）

    合并前的既有测试只有两类：
    * 对拍测试（api_app vs hypothesis）—— 两边都指向同一真源后，
      「真源错了」两边一起错，对拍**恒绿**，失去检出力；
    * 参考值测试（rule_lifecycle 的 `(0,10)→hi≈0.2775, abs=2e-3`）——
      容差 2e-3 抓不住结构性错误。

    变异实测：把真源的 `(centre - margin) / denom` 改成经典的
    `centre - margin`（漏除 denom），**上面两类测试全部仍然绿** ——
    而该变异在 (1,2) 处偏差 0.1816、(30,30) 处偏差 0.1135，
    是教科书级的实现错误。

    本测试的参考值全部由外部推导（statsmodels
    `proportion_confint(method="wilson")` 与 R `prop.test` 同公式），
    手算复核：k=1,n=2，z=1.96 → 下界 0.094529…；容差 1e-6，
    任何结构性错法（少除 denom、漏 max(0,·)、z 取 1.64 等）都会红。
    """
    from lottery.api_app import _wilson_lower

    # statsmodels proportion_confint(1, 2, method="wilson") → (0.0945290, 0.7898399)
    assert _wilson_lower(1, 2) == pytest.approx(0.094529, abs=1e-6)
    # (3,5) → 下界 0.230720…
    assert _wilson_lower(3, 5) == pytest.approx(0.230720, abs=1e-6)
    # (17,20) → 下界 0.639577…
    assert _wilson_lower(17, 20) == pytest.approx(0.639577, abs=1e-6)
    # (30,30) → 下界 0.886483…
    assert _wilson_lower(30, 30) == pytest.approx(0.886483, abs=1e-6)


def test_decision_skill_pairs_same_issue_and_drops_ties():
    """技能口径：同期配对、注数不等剔除、平局剔除（平局数单列）。"""
    from lottery.api_app import decision_skill

    rows = _mk_eval_rows([3, 1, 2, 5], [1, 3, 2, 1])     # 胜/负/平/胜
    skill = decision_skill(rows=rows)
    assert (skill["wins"], skill["losses"], skill["ties"]) == (2, 1, 1)
    assert skill["n_decisive"] == 3
    assert skill["win_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert skill["wilson_lower"] is not None and skill["wilson_lower"] < 2 / 3

    # 注数不等的期被剔除并计数
    mismatched = list(rows) + [
        {"issue": "2026010", "method": "llm:fake", "seq": i, "red_hits": 4,
         "blue_hit": 0, "prize_level": 0, "reward": 0.0, "ticket_cost": 2.0}
        for i in (1, 2, 3, 4, 5)]
    mismatched += [
        {"issue": "2026010", "method": "stat:freq", "seq": i, "red_hits": 0,
         "blue_hit": 0, "prize_level": 0, "reward": 0.0, "ticket_cost": 2.0}
        for i in (1, 2)]
    skill2 = decision_skill(rows=mismatched)
    assert skill2["dropped_ticket_mismatch"] == 1
    assert skill2["n_decisive"] == 3


# ==========================================================================
# ① 关停后预测仍成功（防误伤的核心）
# ==========================================================================

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from lottery import api_app
    return TestClient(api_app.app)


@pytest.fixture()
def with_draws(seeded_db):
    return seeded_db(n=420)


def test_decision_disabled_prediction_still_succeeds(env, with_draws, monkeypatch):
    """① ``decision_disabled=true`` → ``/api/predict?use_llm=true`` **仍 200 成功**，
    只是不含 LLM 通道的注（对照：``LLM_DISABLED`` 那条路会 503）。"""
    from lottery import config, engine, llm_client, methods as METH
    from fastapi.testclient import TestClient
    from lottery import api_app

    # LLM 家族与"可出号"都打开（否则 llm_can_pick 本来就是 False，测不出开关的作用）
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    monkeypatch.setattr(config, "LLM_ONLY_OUTPUT", "mix", raising=False)
    monkeypatch.setattr(config, "ML_ENABLED", False, raising=False)

    METH.set_decision_disabled(True, "测试：Wilson 下界 0.2 ≤ 0.5")
    ch = METH.resolve_channels()
    assert ch["decision_disabled"] is True
    assert ch["llm_can_pick"] is False, "关停必须体现在唯一判据上"

    # 任何 LLM 调用都应被"根本不发起"——若发起则测试失败（避免真联网/计费）
    def _boom(*a, **k):
        raise AssertionError("decision_disabled 时不得调用 LLM 通道")
    monkeypatch.setattr(engine, "llm_tickets", _boom, raising=False)
    monkeypatch.setattr(llm_client, "chat", _boom, raising=False)

    r = TestClient(api_app.app).post("/api/predict?n_tickets=5&regenerate=true&use_llm=true")
    assert r.status_code == 200, f"关停决策通道不得让预测失败：{r.status_code} {r.text[:200]}"
    body = r.json()
    assert body.get("tickets"), "预测必须照常出号"
    assert body["decision_channel_disabled"] is True
    assert all(not str(t.get("method", "")).startswith("llm:") for t in body["tickets"]), \
        "关停后不该含 LLM 通道的注"
    assert "decision_channel_note_zh" in body

    # 恢复后开关回到 False（不影响其它配置）
    METH.set_decision_disabled(False)


def test_decision_disabled_does_not_touch_llm_disabled(env):
    """关停**不复用** ``LLM_DISABLED``：后者仍是 False（否则 Web 预测会 503）。"""
    from lottery import config, methods as METH

    assert config.LLM_DISABLED is False
    METH.set_decision_disabled(True, "测试")
    try:
        assert config.LLM_DISABLED is False, "技能关停绝不能顺手打开 LLM_DISABLED"
        ch = METH.resolve_channels()
        assert ch["decision_disabled"] is True
        assert "decision_disabled" in ch and "llm_can_pick" in ch
        # 新字段为"只增不改"：既有冻结字段一个不少
        for key in ("spec_raw", "source", "mode", "families", "llm_output_mode",
                    "llm_can_pick", "research_arm", "fingerprint"):
            assert key in ch, f"resolve_channels 丢失冻结字段 {key}"
    finally:
        METH.set_decision_disabled(False)


def test_llm_enabled_but_decision_disabled_blocks_can_pick(env, monkeypatch):
    """LLM 家族开着、模式 mix 时：关停前 llm_can_pick=True，关停后必须变 False。"""
    from lottery import config, methods as METH

    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    _write_llm_config(disabled=False, api_key="sk-x", base_url="https://x/v1",
                      model="m", llm_output_mode="mix")
    monkeypatch.setattr(METH, "_read_methods_config",
                        lambda: {"raw": "all", "mode": "production", "llm_output_mode": "mix"})
    METH.set_decision_disabled(False)
    assert METH.resolve_channels()["llm_can_pick"] is True

    METH.set_decision_disabled(True, "Wilson 下界 0.31")
    ch = METH.resolve_channels()
    assert ch["llm_can_pick"] is False and ch["decision_disabled"] is True
    assert "0.31" in ch["decision_disabled_reason"]
    METH.set_decision_disabled(False)
    assert METH.resolve_channels()["llm_can_pick"] is True


# ==========================================================================
# ② 持久化：重启后仍生效（真子进程重新 import config）
# ==========================================================================

def test_decision_state_persists_across_restart(env):
    """④ 关停写进 ``llm_config.json``；**新进程**重新 import 后 resolve_channels 仍读到。"""
    from lottery import config, methods as METH

    METH.set_decision_disabled(True, "自动关停：Wilson 下界 0.40 ≤ 0.50")
    conf = json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    assert conf["decision_disabled"] is True
    assert conf["decision_disabled_reason"].startswith("自动关停")

    # 真重启：子进程只读环境变量 LOTT_HOME，重新 import config/methods
    code = (
        "import json;"
        "from lottery import methods as M;"
        "ch=M.resolve_channels();"
        "print(json.dumps({'decision_disabled': ch['decision_disabled'],"
        "'reason': ch['decision_disabled_reason'],"
        "'llm_can_pick': ch['llm_can_pick']}, ensure_ascii=False))"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          cwd=str(Path(__file__).resolve().parent.parent),
                          env={**os.environ, "LOTT_HOME": str(env.base)},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert payload["decision_disabled"] is True, "重启后关停状态丢失（说明写进了 os.environ）"
    assert payload["reason"].startswith("自动关停")
    assert payload["llm_can_pick"] is False


def test_restore_endpoint_clears_flag_and_preserves_other_keys(env, client):
    """恢复入口：只清关停标志，``disabled`` / ``api_key`` / ``model`` 等键原样保留。"""
    from lottery import config, methods as METH

    _write_llm_config(disabled=False, api_key="sk-keep-me", base_url="https://x/v1",
                      model="keep-model", samples=2)
    METH.set_decision_disabled(True, "自动关停")
    assert METH.decision_state()["disabled"] is True

    r = client.post("/api/llm/decision/restore?reason=manual")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["restored"] is True
    assert body["decision_state"]["disabled"] is False
    assert body["channels"]["decision_disabled"] is False

    conf = json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    assert conf["api_key"] == "sk-keep-me"
    assert conf["model"] == "keep-model" and conf["disabled"] is False
    assert conf["decision_disabled"] is False


def test_decision_state_defaults_without_config(env):
    """没有 llm_config.json 时：状态为未关停、来源 default（不抛异常）。"""
    from lottery import methods as METH

    assert not __import__("lottery.config", fromlist=["x"]).LLM_CONFIG_FILE.exists()
    st = METH.decision_state()
    assert st == {"disabled": False, "reason": "", "at": None, "source": "default"}


# ==========================================================================
# 自动关停：只凭"有决定性样本"（无数据不关停）+ 状态端点
# ==========================================================================

def test_auto_disable_requires_decisive_sample(env, with_draws):
    """无配对样本 → **不写**关停（无证据不等于证否）；有样本且下界≤0.5 → 写。"""
    from lottery import db, methods as METH
    from lottery.api_app import _maybe_auto_disable_decision

    # (a) 库里没有 eval_details → 不关停
    out = _maybe_auto_disable_decision()
    assert out["applied"] is False
    assert out["skill"]["n_decisive"] == 0
    assert METH.decision_state()["disabled"] is False

    # (b) 注入"LLM 连输 5 期" → Wilson 下界 0 < 0.5 → 自动关停并落盘
    for issue, lh, bh in (("2026001", 0, 3), ("2026002", 1, 2), ("2026003", 0, 1),
                          ("2026004", 2, 3), ("2026005", 0, 2)):
        for seq in range(1, 3):
            db.save_eval_details(issue, "llm:fake",
                                 [{"reds": [1, 2, 3, 4, 5, 6], "blue": 1}] * 2,
                                 {"reds": [1, 2, 3, 4, 5, 6], "blue": 1})
    # 直接写 eval_details 太绕：改用 db 接口逐注落库
    conn = db.get_conn()
    conn.execute("DELETE FROM eval_details")
    for issue, lh, bh in (("2026001", 0, 3), ("2026002", 1, 2), ("2026003", 0, 1),
                          ("2026004", 2, 3), ("2026005", 0, 2)):
        for seq in range(1, 3):
            conn.execute("INSERT INTO eval_details (issue,method,seq,red_hits,blue_hit,"
                         "prize_level,reward,ticket_cost,net_return,evaluated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (issue, "llm:fake", seq, lh, 0, 0, 0.0, 2.0, -2.0, 1.0))
            conn.execute("INSERT INTO eval_details (issue,method,seq,red_hits,blue_hit,"
                         "prize_level,reward,ticket_cost,net_return,evaluated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (issue, "stat:freq", seq, bh, 0, 0, 0.0, 2.0, -2.0, 1.0))
    conn.commit()

    out2 = _maybe_auto_disable_decision()
    assert out2["skill"]["n_decisive"] == 5
    assert out2["skill"]["wilson_lower"] == 0.0
    assert out2["skill"]["should_disable"] is True
    assert out2["applied"] is True
    assert METH.decision_state()["disabled"] is True
    # 同进程内的引擎快照也必须被刷新（engine.channels_snapshot 读 config.CHANNELS）
    from lottery import config as _cfg
    assert _cfg.CHANNELS.get("decision_disabled") is True
    assert _cfg.CHANNELS.get("llm_can_pick") is False

    # (c) 注入"LLM 全胜" → 下界 > 0.5 时不关停（并保持未关停）
    METH.set_decision_disabled(False)
    conn.execute("DELETE FROM eval_details")
    for i in range(30):
        issue = f"20270{i:02d}"
        for seq in range(1, 3):
            conn.execute("INSERT INTO eval_details (issue,method,seq,red_hits,blue_hit,"
                         "prize_level,reward,ticket_cost,net_return,evaluated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (issue, "llm:fake", seq, 4, 0, 0, 0.0, 2.0, -2.0, 1.0))
            conn.execute("INSERT INTO eval_details (issue,method,seq,red_hits,blue_hit,"
                         "prize_level,reward,ticket_cost,net_return,evaluated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (issue, "stat:freq", seq, 0, 0, 0, 0.0, 2.0, -2.0, 1.0))
    conn.commit()
    out3 = _maybe_auto_disable_decision()
    assert out3["skill"]["n_decisive"] == 30
    assert out3["skill"]["wilson_lower"] > 0.5
    assert out3["skill"]["should_disable"] is False
    assert out3["applied"] is False
    assert METH.decision_state()["disabled"] is False


def test_decision_status_endpoint(env, client):
    """``GET /api/llm/decision/status``：判据、状态、通道快照齐全且可序列化。"""
    from lottery import methods as METH

    METH.set_decision_disabled(True, "端点测试")
    r = client.get("/api/llm/decision/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["decision_state"]["disabled"] is True
    assert body["channels"]["decision_disabled"] is True
    assert body["skill"]["baseline"] == 0.5
    assert "Wilson" in body["criterion_zh"]
    json.dumps(body)
    METH.set_decision_disabled(False)


def test_scheduler_step_disables_llm_when_decision_disabled(env, with_draws, monkeypatch):
    """调度器路径同样遵守关停：``_scheduler_step`` 以 ``use_llm=False`` 调引擎，
    并**仍然生成本期预测**（关停不影响出号，这就是与 LLM_DISABLED 的区别）。"""
    from lottery import api_app, backtest as BT, engine as _engine
    from lottery import evaluate, methods as METH, notify

    METH.set_decision_disabled(True, "调度测试")
    monkeypatch.setattr(api_app.config, "DRAW_WEEKDAYS", (0, 1, 2, 3, 4, 5, 6))
    monkeypatch.setattr(api_app.config, "DRAW_TIME", "00:00")

    class _F:
        @staticmethod
        def fetch_and_update():
            return {"inserted_new": 0}

    monkeypatch.setattr(api_app, "data_fetcher", _F, raising=False)
    monkeypatch.setattr(evaluate, "online_check", lambda: {"ok": True})
    monkeypatch.setattr(notify, "notify_after_check", lambda r: "skipped")
    monkeypatch.setattr(api_app, "_evolution_step", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(BT, "expected_issue", lambda draws: {"issue": "2026999"})
    monkeypatch.setattr(api_app.db, "load_predictions", lambda i: [])

    calls = []
    monkeypatch.setattr(_engine, "predict_next",
                        lambda *a, **k: calls.append(k.get("use_llm")), raising=False)

    api_app._scheduler_step()
    assert calls == [False], f"关停后调度器不得请求 LLM；实际 use_llm={calls}"
    METH.set_decision_disabled(False)
