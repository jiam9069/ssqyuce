"""U6 自动关闭思考兜底测试：换新推理型模型（未显式配 extra_body）时，
系统应在检测到 reasoning_content 非空时自动注入 thinking:disabled 重试，
而不是只放大 max_tokens（避免超时）。"""
from __future__ import annotations

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY", "")
    monkeypatch.setenv("LOTT_LLM_EXTRA_BODY_MAP", "")
    from lottery import config
    config.DATA_DIR = tmp_path / "data"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = tmp_path / "test.db"
    monkeypatch.setattr(config, "METHODS_RAW", "")
    monkeypatch.setattr(config, "METHOD_MODE", "production")
    monkeypatch.setattr(config, "METHODS_SPEC", {"mode": "all", "tokens": set()})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_DEFAULT", {})
    monkeypatch.setattr(config, "LLM_EXTRA_BODY_BY_MODEL", {})
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://fake.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["new-reasoning-model"])
    monkeypatch.setattr(config, "LLM_EXTRA_MODELS", [])
    return config


def test_auto_inject_thinking_disabled_on_reasoning_content(env, monkeypatch):
    """模型返回空 content + reasoning_content 非空时，自动注入 thinking:disabled 并重试。"""
    from lottery import llm_client

    captured = []
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append(dict(json))
        calls["n"] += 1
        if calls["n"] == 1:
            # 第一次：返回空 content + reasoning（推理型模型典型症状）
            return type("_R", (), {
                "status_code": 200, "text": "",
                "json": lambda self: {"choices": [{
                    "message": {"content": "", "reasoning_content": "thinking..."},
                    "finish_reason": "length"}]},
            })()
        # 第二次（注入 thinking:disabled 后）：返回正常 content
        return type("_R", (), {
            "status_code": 200, "text": "",
            "json": lambda self: {"choices": [{
                "message": {"content": "{\"ok\": 1}", "reasoning_content": ""},
                "finish_reason": "stop"}]},
        })()
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    out = llm_client.chat("s", "u", model_cfg={
        "name": "new", "base_url": "https://x/v1", "api_key": "k",
        "model": "new-reasoning-model"})

    assert out == "{\"ok\": 1}"
    # 第二次请求应注入 thinking:disabled 并移除 reasoning_effort
    assert calls["n"] == 2
    assert captured[-1].get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in captured[-1]


def test_no_reasoning_content_no_thinking_inject(env, monkeypatch):
    """模型正常返回 content 且无 reasoning 时，不注入 thinking:disabled。"""
    from lottery import llm_client

    captured = []

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append(dict(json))
        return type("_R", (), {
            "status_code": 200, "text": "",
            "json": lambda self: {"choices": [{
                "message": {"content": "{\"ok\": 1}", "reasoning_content": ""},
                "finish_reason": "stop"}]},
        })()
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    out = llm_client.chat("s", "u", model_cfg={
        "name": "new", "base_url": "https://x/v1", "api_key": "k",
        "model": "new-reasoning-model"})

    assert out == "{\"ok\": 1}"
    assert "thinking" not in captured[-1]
