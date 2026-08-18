"""M4.2 方法 A/B 开关测试（纯标准库，无外部依赖）。"""
import pytest


def test_normalize_method():
    from lottery.methods import normalize_method
    assert normalize_method(" stat:FREQ ") == "stat:freq"
    assert normalize_method("llm:minimax-m3") == "llm:minimax-m3"
    assert normalize_method(None) == ""
    assert normalize_method(5) == "5"


def test_spec_default_all_enabled():
    from lottery.methods import implement_spec, is_enabled
    spec = implement_spec("")
    assert spec == {"mode": "all", "tokens": set()}
    assert is_enabled("stat:freq", spec)
    assert is_enabled("llm:minimax-m3", spec)


def test_spec_deny_family():
    from lottery.methods import implement_spec, is_enabled
    spec = implement_spec("-llm")
    assert spec["mode"] == "deny"
    assert not is_enabled("llm:minimax-m3", spec)
    assert not is_enabled("llm", spec)
    assert is_enabled("stat:freq", spec)
    assert is_enabled("ml", spec)


def test_spec_allow_family_and_exact():
    from lottery.methods import implement_spec, is_enabled
    spec = implement_spec("stat,ml")
    assert spec["mode"] == "allow"
    assert is_enabled("stat:freq", spec)
    assert is_enabled("stat:markov", spec)
    assert is_enabled("ml", spec)
    assert not is_enabled("llm:x", spec)
    assert not is_enabled("uniform", spec)

    exact = implement_spec("stat:freq,llm")
    assert is_enabled("stat:freq", exact)
    assert not is_enabled("stat:markov", exact)
    assert is_enabled("llm:deepseek-chat", exact)


def test_spec_comma_or_space_separated():
    from lottery.methods import implement_spec
    assert implement_spec("stat,ml") == implement_spec("stat ml")


def test_filter_candidates():
    from lottery.methods import filter_candidates, implement_spec
    tickets = [
        {"reds": [1, 2, 3, 4, 5, 6], "blue": 1, "method": "stat:freq"},
        {"reds": [1, 2, 3, 4, 5, 7], "blue": 2, "method": "llm:minimax-m3"},
        {"reds": [1, 2, 3, 4, 5, 8], "blue": 3, "method": "ml"},
        {"reds": [1, 2, 3, 4, 5, 9], "blue": 4, "method": "uniform"},
    ]
    assert filter_candidates(tickets) == tickets  # 默认全部启用
    only_stat = filter_candidates(tickets, implement_spec("stat"))
    assert [t["method"] for t in only_stat] == ["stat:freq"]
    no_llm = filter_candidates(tickets, implement_spec("-llm"))
    assert "llm:minimax-m3" not in [t["method"] for t in no_llm]


# ---------- M4.2 运行模式 / 校验 / 注册表 ----------

def test_normalize_mode():
    from lottery.methods import normalize_mode
    assert normalize_mode("RESEARCH") == "research"
    assert normalize_mode(" production ") == "production"
    assert normalize_mode("bogus") == "production"  # 非法回退生产
    assert normalize_mode(None) == "production"


def test_effective_spec_research_disables_filtering():
    from lottery.methods import effective_spec, implement_spec, is_enabled
    strict = implement_spec("-llm")
    assert is_enabled("llm:x", strict) is False
    eff = effective_spec("research", strict)
    assert eff["mode"] == "all"
    assert is_enabled("llm:x", eff) is True     # 研究模式忽略开关
    assert effective_spec("production", strict) is strict
    # 默认（无 spec）按环境变量解析
    assert effective_spec("research")["mode"] == "all"


def test_validate_raw():
    from lottery.methods import validate_raw
    assert validate_raw("stat,ml") is None
    assert validate_raw("-llm") is None
    assert validate_raw("stat:freq") is None
    assert validate_raw("stat:") is None       # 尾随冒号等价于族名 stat，无害
    assert validate_raw("") is None
    assert validate_raw(" ") is None
    assert validate_raw("$bad") is not None    # 非法字符
    assert validate_raw("a" * 600) is not None  # 过长


def test_registry():
    from lottery.methods import registry, FAMILIES
    reg = registry()
    assert reg and {r["method"] for r in reg} >= {"stat:freq", "llm", "ml", "uniform"}
    for e in reg:
        assert e["family"] in FAMILIES
        assert e["desc"]
        assert e["method"] == e["method"].lower()
    assert registry() == registry()  # 深拷贝稳定


# ---------- config.set_methods（运行时更新 + 持久化） ----------

def test_config_set_methods(tmp_path, monkeypatch):
    from lottery import config as cfg
    # 复位 env 与全局，避免本测试通过 set_methods 污染后续“默认行为”测试
    monkeypatch.setenv("LOTT_METHODS", "")
    monkeypatch.setenv("LOTT_METHOD_MODE", "production")
    monkeypatch.setattr(cfg, "METHODS_CONFIG_FILE", tmp_path / "methods_config.json")
    st = cfg.set_methods(raw="-llm", mode="research")
    assert st["mode"] == "research"
    assert st["raw"] == "-llm"
    assert cfg.METHOD_MODE == "research"
    assert cfg.METHODS_RAW == "-llm"
    assert cfg.METHODS_SPEC["mode"] == "deny"
    assert cfg.METHODS_CONFIG_FILE.exists()
    # 再切回生产默认，并验证 methods_status
    st2 = cfg.set_methods(raw="", mode="production")
    assert st2["mode"] == "production"
    assert st2["spec"]["mode"] == "all"
    assert cfg.METHODS_SPEC["mode"] == "all"
    # 还原 env，隔离对本文件其它测试的影响
    monkeypatch.setenv("LOTT_METHODS", "")
    monkeypatch.setenv("LOTT_METHOD_MODE", "production")


def test_methods_status_snapshot_shape():
    from lottery import config as cfg
    st = cfg.methods_status()
    assert set(st) == {"mode", "raw", "spec"}
    assert st["spec"]["mode"] in ("all", "allow", "deny")
    assert isinstance(st["spec"]["tokens"], list)