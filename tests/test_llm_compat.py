"""M4.5 LLM 网关兼容性测试：请求级附加参数 / 429 退避重试 / 观察轮多模型轮转。

背景：api.b.ai 等聚合网关的推理型模型（glm-5.3-flash / qwen3.8-flash 等）会把
max_tokens 全部耗在 reasoning_content 上（content 为空），且各模型需要的控制参数
不同（glm: reasoning_effort=low；qwen: enable_thinking=false + reasoning_effort=none），
同时免费池存在瞬态 429（TPM / 并发限制）。本文件验证：
1. LOTT_LLM_EXTRA_BODY / LOTT_LLM_EXTRA_BODY_MAP 正确合并进请求体（保留字段受保护）；
2. chat() 对 429/5xx 做短退避重试（不消耗 attempt），最终成功；
3. llm_tickets() 观察轮在模型间轮转容错（单模型挂掉不影响整个通道）。
"""
import pytest


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
def fresh_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", "")
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY_MAP", "")
    from lottery import config, db
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    db.close()
    db.upsert_draws(_mk_draws())
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    # 复位附加参数（避免跨测试污染）
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_DEFAULT", {})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_BY_MODEL", {})
    return config, db


def _fake_channel(monkeypatch, config, models=("glm-5.3-flash",)):
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://fake.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", list(models))
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])


# ---------- 1. extra body 合并 ----------

def test_extra_body_default_and_per_model(fresh_env, monkeypatch):
    config, _ = fresh_env
    _fake_channel(monkeypatch, config)
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_DEFAULT", {"reasoning_effort": "low"})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_BY_MODEL", {
        "qwen3.8-flash": {"enable_thinking": False, "reasoning_effort": "none"},
    })
    from lottery import llm_client

    captured = []

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append(json)
        class _R:
            status_code = 200
            text = ""
            def json(self):
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        return _R()
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    # 主通道模型：默认参数生效
    llm_client.chat("s", "u", model_cfg={"name": "glm", "base_url": "https://x/v1",
                                         "api_key": "k", "model": "glm-5.3-flash"})
    assert captured[-1]["reasoning_effort"] == "low"
    assert "enable_thinking" not in captured[-1]

    # 按模型覆盖：qwen 参数叠加并覆盖默认
    llm_client.chat("s", "u", model_cfg={"name": "qwen", "base_url": "https://x/v1",
                                         "api_key": "k", "model": "qwen3.8-flash"})
    assert captured[-1]["reasoning_effort"] == "none"
    assert captured[-1]["enable_thinking"] is False

    # 保留字段不可被附加参数改写
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_DEFAULT",
                        {"model": "evil", "messages": [], "temperature": 0.99})
    llm_client.chat("s", "u", temperature=0.5,
                    model_cfg={"name": "glm", "base_url": "https://x/v1",
                               "api_key": "k", "model": "glm-5.3-flash"})
    assert captured[-1]["model"] == "glm-5.3-flash"
    assert len(captured[-1]["messages"]) == 2
    assert captured[-1]["temperature"] == 0.5


def test_extra_body_env_parsing(tmp_path, monkeypatch):
    """LOTT_LLM_EXTRA_BODY / _MAP 环境变量解析（含非法值容错）。"""
    import importlib
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", '{"reasoning_effort":"low"}')
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY_MAP",
                       '{"qwen3.8-flash":{"enable_thinking":false},"bad-entry":42}')
    from lottery import config as cfg
    importlib.reload(cfg)
    try:
        assert cfg.LLM_EXTRA_BODY_DEFAULT == {"reasoning_effort": "low"}
        assert cfg.LLM_EXTRA_BODY_BY_MODEL == {"qwen3.8-flash": {"enable_thinking": False}}
    finally:
        monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", "not-json")
        monkeypatch.setenv("LOTT_LLM_EXTRA_BODY_MAP", "")
        importlib.reload(cfg)
        assert cfg.LLM_EXTRA_BODY_DEFAULT == {}   # 非法 JSON 容错为空
        # 恢复现场，避免污染后续导入
        monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", "")
        importlib.reload(cfg)


# ---------- 2. 429 退避重试 ----------

def test_chat_backoff_retry_on_429(fresh_env, monkeypatch):
    config, _ = fresh_env
    _fake_channel(monkeypatch, config)
    from lottery import llm_client

    calls = {"n": 0}

    class _R429:
        status_code = 429
        text = '{"error":{"message":"Concurrency limit 1200"}}'

    class _R200:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "正常"}, "finish_reason": "stop"}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _R429() if calls["n"] <= 2 else _R200()
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))

    text = llm_client.chat("s", "u", strict=True)
    assert text == "正常"
    assert calls["n"] == 3          # 两次 429 + 一次成功
    assert len(sleeps) == 2         # 两次退避
    assert all(s > 0 for s in sleeps)


def test_chat_429_gives_up_within_budget(fresh_env, monkeypatch):
    """429 持续出现：退避限 3 次后按普通失败处理（strict 抛错）。"""
    config, _ = fresh_env
    _fake_channel(monkeypatch, config)
    from lottery import llm_client

    class _R429:
        status_code = 429
        text = "rate"

    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: _R429())
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(llm_client.LLMChannelError, match="HTTP 429"):
        llm_client.chat("s", "u", strict=True)
    assert len(sleeps) == 3         # 退避次数封顶


# ---------- 3. 观察轮多模型轮转 ----------

def _real_stats(db):
    from lottery import features as F
    return F.compute_features(db.load_draws())


def test_observation_round_rotates_models(fresh_env, monkeypatch):
    """观察轮：模型 A 持续失败 → 自动切到模型 B；选号轮 B 仍出票 → 整体成功。"""
    config, db = fresh_env
    _fake_channel(monkeypatch, config, models=("glm-5.3-flash", "qwen3.8-flash"))
    monkeypatch.setattr(config, "LLM_VERIFY_ENABLED", False)
    from lottery import engine, llm_client

    order = []

    def fake_chat_json(system, user, model_cfg=None, **kw):
        model = model_cfg["model"]
        is_obs = "long_term" in user and '"tickets"' not in user
        if is_obs:
            order.append(("obs", model))
        if model == "glm-5.3-flash":
            raise llm_client.LLMChannelError("HTTP 429: Concurrency limit")
        if is_obs:
            return {"long_term": ["观察"], "mid_term": [], "short_term": [], "caveats": []}
        return {"tickets": [{"reds": [1, 2, 3, 4, 5, 6], "blue": 7, "confidence": 60}]}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)

    tickets = engine.llm_tickets(db.load_draws(), _real_stats(db), [], rng=None,
                                 llm_samples=2, llm_verify=False, strict=True)
    # 观察轮先试 glm（失败）再试 qwen（成功）
    assert ("obs", "glm-5.3-flash") in order
    assert order[-1] == ("obs", "qwen3.8-flash")
    # 选号采样 [glm, qwen]：glm 失败被收集，qwen 出 1 注
    assert len(tickets) == 1
    assert tickets[0]["method"] == "llm:qwen3.8-flash"


def test_observation_round_all_models_fail_strict(fresh_env, monkeypatch):
    """观察轮全部模型失败：strict 抛出最后一个错误（消息可读）。"""
    config, db = fresh_env
    _fake_channel(monkeypatch, config, models=("glm-5.3-flash", "qwen3.8-flash"))
    from lottery import engine, llm_client

    def fake_chat_json(system, user, model_cfg=None, **kw):
        raise llm_client.LLMChannelError(f"HTTP 400: {model_cfg['model']} 余额不足")
    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)

    with pytest.raises(llm_client.LLMChannelError, match="qwen3.8-flash"):
        engine.llm_tickets(db.load_draws(), _real_stats(db), [], rng=None,
                           llm_samples=1, llm_verify=False, strict=True)


def test_observation_rotation_non_strict_degrades(fresh_env, monkeypatch):
    """非 strict（调度器/评估）：模型 A 失败自动换 B，不废掉整个 LLM 通道。"""
    config, db = fresh_env
    _fake_channel(monkeypatch, config, models=("glm-5.3-flash", "qwen3.8-flash"))
    monkeypatch.setattr(config, "LLM_VERIFY_ENABLED", False)
    from lottery import engine, llm_client

    def fake_chat_json(system, user, model_cfg=None, **kw):
        model = model_cfg["model"]
        is_obs = "long_term" in user and '"tickets"' not in user
        if model == "glm-5.3-flash":
            return None  # 非 strict 语义：失败返回 None
        if is_obs:
            return {"long_term": ["观察"], "mid_term": [], "short_term": [], "caveats": []}
        return {"tickets": [{"reds": [5, 9, 13, 21, 27, 33], "blue": 3, "confidence": 55}]}
    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)

    tickets = engine.llm_tickets(db.load_draws(), _real_stats(db), [], rng=None,
                                 llm_samples=2, llm_verify=False, strict=False)
    assert tickets and tickets[0]["method"] == "llm:qwen3.8-flash"
