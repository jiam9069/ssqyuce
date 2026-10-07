"""U8：推理型模型输出预算自适应 + 设置页模型列表唯一事实来源。

背景（线上真实故障）：设置页把模型从旧模型换成 agent-router/deepseek-v4-flash 后，
「设置页测试」通过但「强制重新生成」报 HTTP 404 —— 404 说的是**旧模型**，与真实原因
（新模型把整个 max_tokens 预算花在 reasoning_content 上、content 为空）完全无关。
本文件锁定三条修复：

1. 设置页保存的模型是唯一事实来源：LLM_MODEL_LIST 被**替换**而非前插，
   旧模型不会再混进观察轮轮转并抢占最终报错；
2. 观察轮全部模型失败时，报错聚合**所有**模型的原因，并以主通道（首个）开头；
3. 检测到「只返回推理内容、无正文」时，把输出预算一次抬到 LOTT_LLM_MAX_TOKENS
   重试，并记住该模型可行的预算，后续调用直接从它起步。
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient


def _mk_draws(n=400):
    out = []
    reds = [3, 7, 11, 15, 22, 30]
    for i in range(n):
        out.append({
            "issue": f"2026{i:04d}",
            "date": "2026-01-01",
            "reds": list(reds),
            "order": list(reds),
            "blue": (i % 16) + 1,
        })
    return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", "")
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY_MAP", "")
    from lottery import config, db, llm_client
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    config.LLM_CONFIG_FILE = config.DATA_DIR / "llm_config.json"
    db.close()
    db.upsert_draws(_mk_draws())
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_DEFAULT", {})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_BY_MODEL", {})
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 32000)
    monkeypatch.setattr(config, "LLM_LONG_TIMEOUT", 180.0)
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://fake.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["reasoning-model"])
    monkeypatch.setattr(config, "LLM_MODEL", "reasoning-model")
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])
    llm_client.reset_learned_budgets()
    yield config, db, llm_client
    llm_client.reset_learned_budgets()


def _resp(payload):
    return type("_R", (), {
        "status_code": 200, "text": "",
        "json": lambda self: payload,
    })()


def _reasoning_only():
    return _resp({"choices": [{
        "message": {"content": "", "reasoning_content": "想了很久……"},
        "finish_reason": "length"}]})


def _ok(content='{"ok": 1}'):
    return _resp({"choices": [{
        "message": {"content": content, "reasoning_content": ""},
        "finish_reason": "stop"}]})


# ---------- 1. 输出预算自适应 ----------

def test_reasoning_only_escalates_to_cap_and_learns(env, monkeypatch):
    """只返回推理内容 → 注入 thinking:disabled 并把预算抬到上限；成功后退化为 1 次调用。"""
    config, _, llm_client = env
    captured, calls = [], {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append(dict(json))
        calls["n"] += 1
        return _reasoning_only() if calls["n"] == 1 else _ok()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    out = llm_client.chat("s", "u", max_tokens=1600)

    assert out == '{"ok": 1}'
    assert calls["n"] == 2
    assert captured[0]["max_tokens"] == 1600               # 先按请求预算试一次
    assert captured[1]["max_tokens"] == config.LLM_MAX_TOKENS
    assert captured[1]["thinking"] == {"type": "disabled"}
    # 学到该模型的可用预算
    assert llm_client.learned_max_tokens(config.LLM_BASE_URL, "reasoning-model") == \
        config.LLM_MAX_TOKENS


def test_learned_budget_used_from_first_call(env, monkeypatch):
    """已学过预算的模型：第一次请求就直接用大预算，不再被截断重试。"""
    config, _, llm_client = env
    llm_client.remember_max_tokens(config.LLM_BASE_URL, "reasoning-model",
                                   config.LLM_MAX_TOKENS)
    captured = []
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda url, json=None, headers=None, timeout=None:
                        (captured.append(dict(json)), _ok())[1])

    out = llm_client.chat("s", "u", max_tokens=2000)
    assert out == '{"ok": 1}'
    assert len(captured) == 1
    assert captured[0]["max_tokens"] == config.LLM_MAX_TOKENS


def test_reasoning_only_at_cap_error_is_actionable(env, monkeypatch):
    """预算已在上限仍只有推理内容：不得无限重试，报错要指向真正的原因与开关。"""
    config, _, llm_client = env
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 4000)
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _reasoning_only()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    with pytest.raises(llm_client.LLMChannelError) as ei:
        llm_client.chat("s", "u", max_tokens=1600, strict=True)
    msg = str(ei.value)
    assert "只返回推理内容" in msg
    assert "LOTT_LLM_MAX_TOKENS" in msg
    assert calls["n"] == 2          # 1600 → 4000，不再无限放大


def test_small_auxiliary_call_not_inflated(env, monkeypatch):
    """critique(600) 这类辅助小请求不套用大预算，避免为它多花几十秒。"""
    config, _, llm_client = env
    llm_client.remember_max_tokens(config.LLM_BASE_URL, "reasoning-model",
                                   config.LLM_MAX_TOKENS)
    captured = []
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda url, json=None, headers=None, timeout=None:
                        (captured.append(dict(json)), _ok())[1])
    llm_client.chat("s", "u", max_tokens=600)
    assert captured[0]["max_tokens"] == 600


# ---------- 2. 设置页模型列表 = 唯一事实来源 ----------

def test_settings_save_replaces_model_list(env, monkeypatch):
    config, _, _ = env
    monkeypatch.setattr(config, "LLM_MODEL_LIST",
                        ["group/auto-deepseek-v4-flash-vision"])   # .env 里的旧模型
    from lottery import api_app
    client = TestClient(api_app.app)
    r = client.post("/api/config/llm", json={
        "base_url": "https://magpie.example.com/v1",
        "api_key": "sk-new",
        "model": "agent-router/deepseek-v4-flash",
        "samples": 1,
    })
    assert r.status_code == 200 and r.json()["ok"] is True
    assert config.LLM_MODEL_LIST == ["agent-router/deepseek-v4-flash"]   # 旧模型被替换掉
    # 落盘配置同样只保留新模型
    import json as _json
    saved = _json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    assert saved["model"] == "agent-router/deepseek-v4-flash"
    assert client.get("/api/config/llm").json()["model_list"] == \
        ["agent-router/deepseek-v4-flash"]


def test_settings_save_keeps_unknown_keys(env):
    """设置页保存不应抹掉手工写入的 extra_body / total_timeout。"""
    config, _, _ = env
    config.LLM_CONFIG_FILE.write_text(
        '{"extra_body": {"thinking": {"type": "disabled"}}, "total_timeout": 120}',
        encoding="utf-8")
    from lottery import api_app
    client = TestClient(api_app.app)
    client.post("/api/config/llm", json={"model": "m1"})
    import json as _json
    saved = _json.loads(config.LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    assert saved["extra_body"] == {"thinking": {"type": "disabled"}}
    assert saved["total_timeout"] == 120


# ---------- 4. /api/llm/test：真实结构化出文探测 ----------

def test_llm_test_probe_reports_budget(env, monkeypatch):
    """测试接口必须真的走一遍结构化出文（并回报预算），而不是只 ping 连接。"""
    config, _, llm_client = env
    seen = {}

    def fake_chat_json(system, user, **kw):
        seen.update(kw)
        seen["prompt_len"] = len(user)
        return {"long_term": ["x"], "mid_term": [], "short_term": [], "caveats": []}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    from lottery import api_app
    client = TestClient(api_app.app)
    r = client.post("/api/llm/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "reasoning-model"
    assert "long_term" in body["json_keys"]
    # 走的是真实观察轮提示（而非一句“请回复连接成功”）
    assert seen["prompt_len"] > 500
    assert seen["strict"] is True


def test_llm_test_probe_surfaces_reasoning_error(env, monkeypatch):
    config, _, llm_client = env

    def fake_chat_json(system, user, **kw):
        raise llm_client.LLMChannelError(
            "模型只返回推理内容（reasoning_content 3000 字），32000 输出预算内未产出正文")

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    from lottery import api_app
    client = TestClient(api_app.app)
    body = client.post("/api/llm/test").json()
    assert body["ok"] is False
    assert "只返回推理内容" in body["error"]
    assert "LOTT_LLM_MAX_TOKENS" in body["hint"]


# ---------- 5. 观察轮报错聚合 ----------
def test_observation_error_reports_all_models(env, monkeypatch):
    """旧模型 404 不得掩盖主通道的真实失败原因。"""
    config, db, llm_client = env
    monkeypatch.setattr(config, "LLM_MODEL_LIST",
                        ["agent-router/deepseek-v4-flash", "group/old-model"])
    from lottery import engine, features as F

    def fake_chat_json(system, user, model_cfg=None, **kw):
        model = model_cfg["model"]
        if model == "agent-router/deepseek-v4-flash":
            raise llm_client.LLMChannelError(
                "模型只返回推理内容（reasoning_content 2544 字），32000 输出预算内未产出正文")
        raise llm_client.LLMChannelError('HTTP 404: magpie knows no model "group/old-model"')

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    draws = db.load_draws()
    with pytest.raises(llm_client.LLMChannelError) as ei:
        engine.llm_tickets(draws, F.compute_features(draws), [], rng=None,
                           llm_samples=1, llm_verify=False, strict=True)
    msg = str(ei.value)
    assert "只返回推理内容" in msg            # 真正的原因在前面
    assert "group/old-model" in msg            # 旧模型的 404 也在，不丢信息
    assert msg.index("只返回推理内容") < msg.index("group/old-model")
