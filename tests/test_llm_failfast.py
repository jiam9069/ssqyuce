"""M4.5 LLM 快速失败（fail-fast）测试。

背景：Web「强制重新预测」勾选大模型时，若 LLM 通道不可用/超时，
必须直接返回明确的“预测失败”提示（503 + error 详情），禁止静默
降级为纯统计模型；LLM 阶段总耗时受 LLM_TOTAL_TIMEOUT 预算钳制，
保证响应落在反向代理 / CDN 超时窗口内（避免 Cloudflare HTTP 524）。
"""
import time

import pytest


def _mk_draws(n=400):
    """构造足够长的伪开奖序列（避开 ML 最少期数分支的干扰）。"""
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
def fresh_env(tmp_path, monkeypatch):
    """隔离的 DATA_DIR / DB 与干净的 LLM 配置。"""
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    db.close()
    db.upsert_draws(_mk_draws())
    # 复位方法开关，避免其它测试污染
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    return config, db


def _configure_fake_channel(monkeypatch, config):
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://fake.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["fake-model"])
    monkeypatch.setattr(config, "LLM_MODEL", "fake-model")
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])


# ---------- llm_client.chat：strict / deadline ----------

def test_chat_strict_raises_on_http_error(fresh_env, monkeypatch):
    config, _ = fresh_env
    _configure_fake_channel(monkeypatch, config)
    from lottery import llm_client

    class _Resp:
        status_code = 400
        text = '{"error":{"code":"insufficient_user_quota"}}'

    seen = []
    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(timeout)
        return _Resp()
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    # 非 strict：保持降级语义（返回 None，不抛错）
    assert llm_client.chat("s", "u") is None
    # strict：抛 LLMChannelError，消息含 HTTP 400
    with pytest.raises(llm_client.LLMChannelError, match="HTTP 400"):
        llm_client.chat("s", "u", strict=True)
    assert len(seen) == 6  # 两次调用各重试 3 次


def test_chat_deadline_clamps_per_attempt_timeout(fresh_env, monkeypatch):
    """deadline 生效：每次尝试的超时被钳制在剩余预算内，boost 也不例外。"""
    config, _ = fresh_env
    _configure_fake_channel(monkeypatch, config)
    monkeypatch.setattr(config, "LLM_TIMEOUT", 60.0)
    from lottery import llm_client

    seen = []
    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(timeout)
        raise llm_client.requests.exceptions.ReadTimeout("read timed out")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    t0 = time.time()
    with pytest.raises(llm_client.LLMChannelError):
        llm_client.chat("s", "u", deadline=t0 + 5.0, strict=True)
    # 假 post 立即返回，总耗时应远小于 5s 预算
    assert time.time() - t0 < 4.0
    # 所有尝试（含读取超时 boost 的 240s）都被钳制在 ≤5s
    assert seen and all(t <= 5.0 for t in seen)


def test_chat_unconfigured_strict_raises(fresh_env, monkeypatch):
    config, _ = fresh_env
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", None)
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(config, "LLM_MODEL_LIST", [])
    from lottery import llm_client
    with pytest.raises(llm_client.LLMChannelError, match="未配置"):
        llm_client.chat("s", "u", strict=True)


# ---------- engine.predict_next：llm_required 快速失败 ----------

def test_predict_llm_required_channel_down_raises(fresh_env, monkeypatch):
    """通道配置存在但调用失败：llm_required=True 直接抛错，False 优雅降级。"""
    config, _ = fresh_env
    _configure_fake_channel(monkeypatch, config)
    from lottery import engine, llm_client

    def fake_chat(system, user, **kw):
        if kw.get("strict"):
            raise llm_client.LLMChannelError("HTTP 400: credit insufficient balance")
        return None  # 非 strict：模拟旧降级路径
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    draws = _mk_draws(310)
    with pytest.raises(llm_client.LLMChannelError, match="insufficient"):
        engine.predict_next(draws, use_llm=True, llm_required=True, persist=False)

    # 非 llm_required（调度器/评估语义）：降级为统计，正常返回
    res = engine.predict_next(draws, use_llm=True, llm_required=False, persist=False)
    assert res["llm_used"] is False
    assert res["tickets"]


def test_predict_llm_required_unconfigured_raises(fresh_env, monkeypatch):
    config, _ = fresh_env
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", None)
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(config, "LLM_MODEL_LIST", [])
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])
    from lottery import engine, llm_client

    with pytest.raises(llm_client.LLMChannelError, match="未配置"):
        engine.predict_next(_mk_draws(310), use_llm=True, llm_required=True,
                            persist=False)


def test_predict_llm_required_method_switch_off_raises(fresh_env, monkeypatch):
    config, _ = fresh_env
    _configure_fake_channel(monkeypatch, config)
    from lottery import engine, methods, llm_client
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", methods.implement_spec("-llm"))

    with pytest.raises(llm_client.LLMChannelError, match="方法开关"):
        engine.predict_next(_mk_draws(310), use_llm=True, llm_required=True,
                            persist=False)


def test_predict_llm_required_no_persist_on_failure(fresh_env, monkeypatch):
    """快速失败时不得写入任何预测（避免半成品落库）。"""
    config, db = fresh_env
    _configure_fake_channel(monkeypatch, config)
    from lottery import engine, llm_client, backtest as BT

    def fake_chat(system, user, **kw):
        raise llm_client.LLMChannelError("HTTP 401: bad key")
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    draws = db.load_draws()
    issue = BT.next_issue(draws[-1]["issue"])
    with pytest.raises(llm_client.LLMChannelError):
        engine.predict_next(draws, use_llm=True, llm_required=True, persist=True)
    assert db.load_predictions(issue) == []


# ---------- API 层：/api/predict 503 明确失败 ----------

def test_api_predict_llm_failfast_503(fresh_env, monkeypatch):
    config, db = fresh_env
    _configure_fake_channel(monkeypatch, config)
    from lottery import llm_client
    from fastapi.testclient import TestClient
    from lottery import api_app

    def fake_chat(system, user, **kw):
        if kw.get("strict"):
            raise llm_client.LLMChannelError("HTTP 400: credit insufficient balance")
        return None
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    client = TestClient(api_app.app)
    r = client.post("/api/predict?n_tickets=10&regenerate=true&use_llm=true")
    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert "大模型不可用或超时" in body["error"]
    assert "insufficient" in body["error"]

    # use_llm=false：纯统计模式不受影响，正常 200
    r2 = client.post("/api/predict?n_tickets=10&regenerate=true&use_llm=false")
    assert r2.status_code == 200
    assert r2.json()["tickets"]
