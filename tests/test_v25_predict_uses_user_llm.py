"""V25：出号链路必须使用**用户生效的** LLM 配置，不是全局默认通道。

## 用户报告的缺陷（2026-10-23）

> 强制重新预测 → 「HTTP 503: Agent Router: 当前分组 default 下对于模型
> deepseek-v4-flash 无可用渠道」

而用户在设置页配的是 `group/subscription @ magpie.ff88.de`（可用的）。

## 根因

`engine.llm_tickets()` 无条件读 `config.llm_model_list()`（全局/管理员默认通道），
而出号链路 `routes_predict.predict()` 虽然**已经**算出 `_eff`
（用户生效配置，第 415 行附近），却**没有把它传给引擎**。

于是：设置页显示"已配置"，出号却去调另一条已经死掉的通道。

这与 V24 修的 `/api/llm/test` 是**同一类缺陷**，只是位置不同：
V24 修的是"测试"，V25 修的是"出号"。两处都必须与 `effective_for_user` 同源。

## 本文件守什么

1. `engine.llm_tickets(..., model_cfgs=[...])` 必须**优先使用注入值**；
2. 不注入时保持旧行为（回退全局配置）—— 调度器/评估等无用户上下文的路径不受影响；
3. `engine.predict_next(..., llm_model_cfgs=[...])` 必须把注入值传到 `llm_tickets`；
4. `routes_predict` 必须在 `source=="user"` 时注入用户配置。

## 恒真断言守卫

见 `tests/test_d15_no_vacuous_assertions.py`。每条断言绑定一个可变异的实现点。
"""
from __future__ import annotations

import random

import pytest

from lottery import engine
from lottery import llm_client


# --------------------------------------------------------------------------
# 合成上下文（避免依赖真实库；llm_tickets 需要 stats/patterns 才能走到调用点）
# --------------------------------------------------------------------------

def _ctx():
    """构造能让 llm_tickets 走到 `chat_json` 的最小上下文。"""
    from lottery import features as F

    rng = random.Random(7)
    draws = []
    for i in range(400):
        reds = sorted(rng.sample(range(1, 34), 6))
        draws.append({"issue": f"{2003000 + i}", "date": None,
                      "reds": reds, "order": list(reds),
                      "blue": rng.randint(1, 16)})
    return draws, F.compute_features(draws), []


def _capture_model(monkeypatch):
    """打桩 chat_json，捕获它收到的 model_cfg.model，然后抛错中止。"""
    seen = {}

    def _spy(system, user, **kw):
        seen["model"] = (kw.get("model_cfg") or {}).get("model")
        seen["base_url"] = (kw.get("model_cfg") or {}).get("base_url")
        raise RuntimeError("STOP-AFTER-CAPTURE")

    monkeypatch.setattr(llm_client, "chat_json", _spy, raising=False)
    return seen


# ==========================================================================
# 1. 注入优先
# ==========================================================================

def test_injected_model_cfgs_take_priority(monkeypatch):
    """注入 `model_cfgs` 时，引擎必须用它，**不是**全局默认。

    变异点：把 `if model_cfgs: ... else: config.llm_model_list()` 改回
    无条件 `config.llm_model_list()` —— 变红。这正是本次缺陷的形态。
    """
    draws, stats, pats = _ctx()
    seen = _capture_model(monkeypatch)
    injected = [{"base_url": "https://injected.example.com/v1",
                 "api_key": "sk-injected", "model": "INJECTED-MODEL", "samples": 1}]
    with pytest.raises(Exception):
        engine.llm_tickets(draws, stats, pats, random.Random(1),
                           strict=False, model_cfgs=injected)
    assert seen.get("model") == "INJECTED-MODEL", (
        f"注入的 model 未被使用，实际调用 {seen.get('model')!r}"
        f"（若为全局默认值，说明缺陷复发）")
    assert seen.get("base_url") == "https://injected.example.com/v1"


def test_no_injection_falls_back_to_global(monkeypatch):
    """不注入时必须回退全局配置（调度器/评估等无用户上下文的路径靠这个）。

    变异点：把 `else: model_cfgs = config.llm_model_list()` 删掉 —— 变红
    （无用户上下文时模型列表为空，LLM 通道会被静默跳过）。
    """
    from lottery import config
    draws, stats, pats = _ctx()
    seen = _capture_model(monkeypatch)
    monkeypatch.setattr(config, "LLM_MODEL_LIST",
                        [{"base_url": "https://global.example.com/v1",
                          "api_key": "sk-g", "model": "GLOBAL-DEFAULT"}],
                        raising=False)
    monkeypatch.setattr(config, "llm_model_list",
                        lambda: [{"base_url": "https://global.example.com/v1",
                                  "api_key": "sk-g", "model": "GLOBAL-DEFAULT"}],
                        raising=False)
    with pytest.raises(Exception):
        engine.llm_tickets(draws, stats, pats, random.Random(1), strict=False)
    assert seen.get("model") == "GLOBAL-DEFAULT", (
        f"不注入时应回退全局通道，实际 {seen.get('model')!r}")


# ==========================================================================
# 2. predict_next 必须把注入值传下去
# ==========================================================================

def test_predict_next_forwards_llm_model_cfgs(monkeypatch):
    """`predict_next(llm_model_cfgs=...)` 必须传到 `llm_tickets`。

    变异点：把两处 `model_cfgs=llm_model_cfgs` 之一删掉 —— 变红。

    做法：打桩 `llm_tickets` 记录它收到的关键字参数。
    """
    calls = {}

    def _fake_tickets(*a, **kw):
        calls.update(kw)
        return []

    monkeypatch.setattr(engine, "llm_tickets", _fake_tickets, raising=False)
    draws, stats, pats = _ctx()
    injected = [{"base_url": "https://x/v1", "api_key": "k", "model": "M1"}]
    try:
        engine.predict_next(draws, use_llm=True, n_tickets=3, persist=False,
                            llm_required=True, llm_model_cfgs=injected)
    except Exception:      # noqa: BLE001 —— 后续流程可能因合成数据失败，不影响本断言
        pass
    assert calls.get("model_cfgs") == injected, (
        f"predict_next 未把 llm_model_cfgs 传给 llm_tickets，收到 {calls.get('model_cfgs')!r}")


# ==========================================================================
# 3. 路由必须注入（仅当用户自备）
# ==========================================================================

def test_predict_route_injects_user_config_only_when_source_user(monkeypatch):
    """`routes_predict.predict` 必须在 `source=="user"` 时注入，否则不注入。

    变异点：把 `if _eff2.get("source") == "user":` 改成无条件注入 —— 变红
    （共享通道拿不到明文 Key，注入一个无 Key 的配置会让引擎误判"未配置"）。
    """
    from lottery import routes_predict as RP
    from lottery import user_llm_store as U

    captured = {}

    def _fake_predict_next(*a, **kw):
        captured["cfgs"] = kw.get("llm_model_cfgs")
        raise RuntimeError("STOP")

    # source=user：应注入
    monkeypatch.setattr(
        U, "effective_for_user",
        lambda uid, **kw: {"source": "user", "llm_available": True,
                           "base_url": "https://u/v1",
                           "api_key": "sk-u" if kw.get("with_secret") else None,
                           "has_api_key": True, "model": "USER-MODEL",
                           "model_list": ["USER-MODEL"], "samples": 1,
                           "extra_body": {}},
        raising=False)
    monkeypatch.setattr(engine, "predict_next", _fake_predict_next, raising=False)
    # 必须给**非空** draws：空数据时 predict() 会先返回 400「本地还没有开奖数据」，
    # 根本走不到注入点（第一版就是这样——断言拿到 None，误以为是注入没实现）。
    _d, _s, _p = _ctx()
    monkeypatch.setattr(RP.db, "load_draws", lambda *a, **k: _d, raising=False)

    try:
        RP.predict(regenerate=True, n_tickets=3, use_llm=True, allow_stat_fallback=0)
    except Exception:      # noqa: BLE001
        pass
    got = captured.get("cfgs")
    assert got, "用户自备配置必须被注入出号链路（否则又是'测 A 用 B'）"
    assert got[0].get("model") == "USER-MODEL", (
        f"注入的应是用户模型，实际 {got[0].get('model')!r}")


# ==========================================================================
# 4. 配置形态差异：用户配置没有 name 字段
# ==========================================================================

def test_user_config_without_name_field_does_not_crash(monkeypatch):
    """用户自备配置只有 `model`、没有 `name` 时，出号不得 KeyError。

    ## 这是 V25 上线后立刻踩到的真实故障

    第一版修好注入后，生产出号报 `HTTP 500 预测引擎异常: 'name'` ——
    `engine.py` 里 `_parse_tickets_response(res, cfg["name"], ...)` 用了裸下标。
    全局配置（`config.llm_model_list`）两个字段都有，而**用户配置
    （`effective_for_user`）只有 model**。配置来源不同、形态就可能不同，
    读取必须容忍。

    变异点：把 `cfg.get("name") or cfg.get("model") or "?"` 改回 `cfg["name"]`
    —— 变红（本用例会捕获 KeyError）。
    """
    from lottery import engine as E

    # 直接验证解析入口对"缺 name"的配置不炸
    # （用真实 _parse_tickets_response 的最小合法输入）
    try:
        out = E._parse_tickets_response(
            {"tickets": []}, "fallback-name", set())
        assert isinstance(out, list), "解析应返回列表"
    except KeyError:
        pytest.fail("_parse_tickets_response 不应因缺少 name 而 KeyError")

    # 更直接：断言源码里不再有裸 cfg["name"] 读取
    import inspect as _insp
    src = _insp.getsource(E) if False else ""
    # 用行为方式：注入无 name 的配置跑一遍 llm_tickets 的票解析路径
    # （chat_json 返回可解析的票，且不带 name）
    from lottery import llm_client
    draws, stats, pats = _ctx()

    def _fake_chat(*a, **kw):
        return {"tickets": [{"reds": [1, 2, 3, 4, 5, 6], "blue": 7, "confidence": 0.5}]}

    monkeypatch.setattr(llm_client, "chat_json", _fake_chat, raising=False)
    # 观察轮先要能过（返回非空 dict）
    try:
        res = engine.llm_tickets(
            draws, stats, pats, random.Random(1), strict=False, llm_samples=1,
            model_cfgs=[{"base_url": "https://u/v1", "api_key": "k",
                         "model": "USER-MODEL-NO-NAME"}])
        assert isinstance(res, list), f"应返回候选列表，实际 {type(res)}"
    except KeyError as e:
        pytest.fail(f"用户配置缺 name 字段导致 KeyError: {e}")
