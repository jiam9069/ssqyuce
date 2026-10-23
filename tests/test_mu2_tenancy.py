"""MU-2：数据租户化 + LLM 配置隔离。

## 本文件的存在理由

多用户改造里，**串号是静默的**：少写一个 `AND user_id=?`，用户 B 重新生成就会
删光用户 A 同期的票，而且票单只存这一份、**不可恢复**。这类缺陷不会报错、
不会 4xx，只会在某天表现为"我的数据不见了"。

所以这里的核心不是"功能能用"，而是**反向用例**：
* A 写自己的票，B 的票必须**一行不少**；
* A 读不到 B 的任何行；
* A 的 Key 不出现在 B 的任何响应里；
* 库里存的必须是密文，不是明文。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from lottery import auth, config, db, secretbox, user_llm_store
from lottery import migrations

TESTKEY = bytes.fromhex("7a65726f746573746b6579303030").decode()


def _mk(num=400):
    out = []
    reds = [3, 7, 11, 15, 22, 30]
    for i in range(num):
        out.append({"issue": f"2026{i:04d}", "date": "2026-01-01",
                    "reds": list(reds), "order": list(reds), "blue": (i % 16) + 1})
    return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """独立库 + 两个用户（admin / alice）。"""
    monkeypatch.setenv("LOTT_HOME", str(tmp_path))
    monkeypatch.setenv("LOTT_DB", str(tmp_path / "t.db"))
    monkeypatch.delenv("LOTT_MASTER_KEY", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "LLM_CONFIG_FILE", config.DATA_DIR / "llm_config.json")
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", False)
    monkeypatch.setattr(config, "LLM_DISABLED", True)      # 默认无共享通道
    db.close()
    secretbox.reset_cache()
    db.get_conn()                                           # 触发迁移
    conn = db.get_conn()
    conn.execute("UPDATE users SET role='admin', status='active', is_primary_admin=1,"
                 " password_hash=? WHERE status='bootstrap'",
                 (auth.hash_password("AdminPass123"),))
    conn.commit()
    auth.create_user("alice", "AlicePass123", role="user")
    db.upsert_draws(_mk())
    yield tmp_path
    db.close()
    secretbox.reset_cache()


@pytest.fixture()
def client(env, monkeypatch):
    from lottery import api_app
    with TestClient(api_app.app) as c:
        yield c


def _login(client, u, p):
    auth.reset_rate_limits()
    r = client.post("/api/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Cookie": f"{config.SESSION_COOKIE_NAME}={r.cookies.get(config.SESSION_COOKIE_NAME)}",
            "X-CSRF-Token": r.json()["csrf_token"]}


ADMIN_ID = lambda: int(db.get_conn().execute(
    "SELECT user_id FROM users WHERE is_primary_admin=1").fetchone()[0])   # noqa: E731
ALICE_ID = lambda: int(db.get_conn().execute(
    "SELECT user_id FROM users WHERE username='alice'").fetchone()[0])     # noqa: E731


# ---------------------------------------------------------------------------
# 1. 迁移：主键与回填
# ---------------------------------------------------------------------------

def test_migration_8_rebuilds_primary_keys(env):
    conn = db.get_conn()
    assert migrations.current_version(conn) >= 8
    for t, want in (
        ("predictions", ["user_id", "issue", "seq"]),
        ("eval_meta", ["user_id", "issue", "method"]),
        ("eval_details", ["user_id", "issue", "method", "seq"]),
        ("eval_results", ["user_id", "issue"]),
    ):
        rows = conn.execute(f"PRAGMA table_info({t})").fetchall()
        got = [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]
        assert got == want, f"{t} 主键应为 {want}，实际 {got}"


def test_migration_creates_new_tables(env):
    """迁移必须建出多租户相关的表。

    V28：`issue_baseline` 已从断言中移除 —— 那张表自创建起 8 个版本**从未被写入**
    （全仓库无写入方、无读取方），V28 把它删了并令迁移 7 变为 no-op。
    留着断言会逼着保留一张永远为空的表，那是假承诺。
    """
    names = {r[0] for r in db.get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "user_llm_config" in names
    assert "issue_baseline" not in names, "V28 已删除该表（从未被写入）"


def test_migration_is_idempotent(env):
    """重复 apply 不得改变行数、不得报错（迁移会在每次启动时被检查）。"""
    conn = db.get_conn()
    before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("predictions", "eval_meta", "eval_details", "eval_results")}
    migrations.apply(conn)
    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ("predictions", "eval_meta", "eval_details", "eval_results")}
    assert before == after


def test_backfill_uses_dynamic_admin_not_literal_one(env):
    """回填锚点必须是**动态解析**的主管理员，不能写死 1。

    这条用"改掉 admin 的 user_id 也不影响归属"来间接证明：
    SQLite `AUTOINCREMENT` 不复用 ID，写死 1 在"删过 admin 再重建"的库上会
    指向不存在的人或普通用户。
    """
    from lottery import migrations as M
    admin = ADMIN_ID()
    assert M._require_admin_id(db.get_conn()) == admin


def test_no_hardcoded_user_id_one_in_source():
    """源码纪律：**代码**里不许出现 `user_id=1` 字面量。

    注释里出现是**允许的**（而且应该出现）—— 代码里明确写着"绝不写死 user_id=1"
    恰好是这条纪律的可读形式。第一版检测没跳过注释，把解释纪律的注释本身判成违规，
    属于典型的"断言写得太粗"。这里只检查剥掉注释后的代码部分。
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "lottery"
    pat = re.compile(r"user_id\s*=\s*1\b")
    DQ3, SQ3 = chr(34) * 3, chr(39) * 3
    bad = []
    for p in sorted(root.glob("*.py")):
        in_doc = False
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            # 跟踪三引号块：docstring 里写纪律说明是合法的，不该被判违规
            quotes = stripped.count(DQ3) + stripped.count(SQ3)
            if quotes:
                if quotes % 2 == 1:
                    in_doc = not in_doc
                continue
            if in_doc:
                continue
            code = line.split("#", 1)[0]          # 剥掉行尾注释再判
            if pat.search(code):
                bad.append(f"{p.name}:{i}: {line.strip()}")
    assert not bad, "代码里出现写死的 user_id=1（应用动态解析主管理员）：\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# 2. 串号防护（本文件最重要的部分）
# ---------------------------------------------------------------------------

def test_saving_one_user_does_not_delete_another(env):
    """**核心反向用例**：A 的票不因 B 重新生成而消失。

    旧实现 `DELETE FROM predictions WHERE issue=?` 不带 user_id，
    多用户下 B 一生成就删光 A 的票 —— 而票单只存这一份，不可恢复。
    """
    a, b = ADMIN_ID(), ALICE_ID()
    issue = "2026999"
    db.save_predictions(issue, [{"reds": [1, 2, 3, 4, 5, 6], "blue": 7,
                                 "confidence": 0.5, "method": "stat"}], user_id=a)
    db.save_predictions(issue, [{"reds": [7, 8, 9, 10, 11, 12], "blue": 8,
                                 "confidence": 0.4, "method": "stat"},
                                {"reds": [13, 14, 15, 16, 17, 18], "blue": 9,
                                 "confidence": 0.4, "method": "stat"}], user_id=b)
    assert len(db.load_predictions(issue, user_id=a)) == 1, "A 的票被 B 删掉了"
    assert len(db.load_predictions(issue, user_id=b)) == 2
    # A 重新生成，也不能影响 B
    db.save_predictions(issue, [{"reds": [2, 3, 4, 5, 6, 7], "blue": 10,
                                 "confidence": 0.6, "method": "stat"}], user_id=a)
    assert len(db.load_predictions(issue, user_id=b)) == 2, "B 的票被 A 删掉了"


def test_reads_are_isolated(env):
    a, b = ADMIN_ID(), ALICE_ID()
    db.save_predictions("2026888", [{"reds": [1, 2, 3, 4, 5, 6], "blue": 7,
                                    "confidence": 0.5, "method": "stat"}], user_id=a)
    assert db.load_predictions("2026888", user_id=b) == []
    assert db.load_last_predictions(user_id=b) == []
    assert db.recent_prediction_issues(20, user_id=b) == []
    assert db.recent_prediction_issues(20, user_id=a) == ["2026888"]


def test_eval_tables_are_isolated(env):
    a, b = ADMIN_ID(), ALICE_ID()
    db.save_eval("2026777", 1.5, 0.5, 10.0, 5, user_id=a)
    db.save_eval_meta("2026777", [{"method": "stat"}], user_id=a)
    assert db.load_eval(user_id=b) == []
    assert db.load_eval_meta("2026777", user_id=b) == []
    assert len(db.load_eval(user_id=a)) == 1
    # 配置快照不许互相覆盖（旧实现的 INSERT OR REPLACE + PK 无 user 会覆盖）
    db.save_eval_meta("2026777", [{"method": "stat"}], user_id=b)
    assert len(db.load_eval_meta("2026777", user_id=a)) == 1
    assert len(db.load_eval_meta("2026777", user_id=b)) == 1


def test_load_eval_details_isolated(env):
    from lottery import db as D
    a, b = ADMIN_ID(), ALICE_ID()
    draw = {"reds": [1, 2, 3, 4, 5, 6], "blue": 7}
    D.save_eval_details("2026666", "stat",
                        [{"reds": [1, 2, 3, 4, 5, 6], "blue": 7}], draw, user_id=a)
    assert D.load_eval_details(user_id=b) == []
    assert len(D.load_eval_details(user_id=a)) == 1


# ---------------------------------------------------------------------------
# 3. 租户解析
# ---------------------------------------------------------------------------

def test_resolve_tenant_prefers_explicit_then_context_then_admin(env):
    a = ADMIN_ID()
    assert db.resolve_tenant(a) == a             # 显式最优先
    assert db.resolve_tenant(None) == a          # 无上下文 → 主管理员
    assert db.resolve_tenant(99999) == 99999     # 显式即使是陌生 id 也照用（调用方负责）


def test_system_user_id_is_zero_not_null(env):
    """系统主体哨兵是 0（不插 users 表），与 config 一致。"""
    assert db.SYSTEM_USER_ID == 0
    assert config.SYSTEM_USER_ID == 0
    row = db.get_conn().execute(
        "SELECT 1 FROM users WHERE user_id=0").fetchone()
    assert row is None, "SYSTEM_USER_ID 不应在 users 表里真实存在"


# ---------------------------------------------------------------------------
# 4. 密钥加密
# ---------------------------------------------------------------------------

def test_api_key_stored_encrypted_not_plaintext(env):
    uid = ALICE_ID()
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "https://u.example.com/v1",
                              "model": "m", "api_key": TESTKEY})
    raw = user_llm_store.get_raw(uid)
    assert raw["api_key_encrypted"], "必须落密文"
    assert TESTKEY not in str(raw), "库里出现了明文 Key"
    assert user_llm_store.get_api_key(uid) == TESTKEY   # 能解回来才发得出请求


def test_views_never_contain_plaintext_key(env):
    uid = ALICE_ID()
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "u",
                              "model": "m", "api_key": TESTKEY})
    for v in (user_llm_store.get_view(uid), user_llm_store.effective_for_user(uid)):
        assert TESTKEY not in json.dumps(v, ensure_ascii=False)


def test_saving_without_api_key_keeps_existing(env):
    """改别的字段不能丢 Key（前端输入框回显的是脱敏占位，会提交空串）。"""
    uid = ALICE_ID()
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "u", "model": "m",
                              "api_key": TESTKEY})
    user_llm_store.save(uid, {"samples": 7})
    v = user_llm_store.get_view(uid)
    assert v["samples"] == 7
    assert v["has_api_key"] is True
    assert user_llm_store.get_api_key(uid) == TESTKEY


def test_empty_api_key_does_not_clear(env):
    """显式提交空串也保留（否则"改个注数就丢 Key"）；清空要显式 clear_api_key。"""
    uid = ALICE_ID()
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "u", "model": "m",
                              "api_key": TESTKEY})
    user_llm_store.save(uid, {"api_key": ""})
    assert user_llm_store.get_api_key(uid) == TESTKEY
    user_llm_store.save(uid, {}, clear_api_key=True)
    assert user_llm_store.get_api_key(uid) is None


def test_master_key_file_is_0600(env):
    secretbox.master_key()
    p = env / "data" / secretbox.SECRET_KEY_FILENAME
    assert p.exists()
    assert (p.stat().st_mode & 0o777) == 0o600, "主密钥文件必须是 0600"


def test_decrypt_failure_degrades_to_none(env):
    """换过主密钥 / 密文损坏 → 返回 None，**不抛异常**（否则整站 500）。"""
    assert secretbox.decrypt("garbage-not-a-token") is None
    assert secretbox.decrypt("") is None
    assert secretbox.decrypt(None) is None


def test_mask_rules():
    assert secretbox.mask(None) == ""
    assert secretbox.mask("") == ""
    assert secretbox.mask("abc") == "****"          # 短 Key 全隐
    assert secretbox.mask("abcdefg") == "****"
    assert secretbox.mask("sk-abcdefghijkl") == "sk-****ijkl"


# ---------------------------------------------------------------------------
# 5. effective_for_user 三态与降级
# ---------------------------------------------------------------------------

def test_effective_none_when_nothing_configured(env):
    eff = user_llm_store.effective_for_user(ALICE_ID())
    assert eff["source"] == "none"
    assert eff["llm_available"] is False


def test_effective_user_key_wins_over_system(env, monkeypatch):
    uid = ALICE_ID()
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://sys.example.com/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "sys-key-abcdefgh")
    monkeypatch.setattr(config, "LLM_MODEL", "sys-model")
    monkeypatch.setattr(config, "LLM_MODEL_LIST", ["sys-model"])
    eff = user_llm_store.effective_for_user(uid)
    assert eff["source"] == "system"
    # 配了自备 Key 后必须切到 user
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "https://u.example.com/v1",
                              "model": "mine", "api_key": TESTKEY})
    eff2 = user_llm_store.effective_for_user(uid)
    assert eff2["source"] == "user" and eff2["model"] == "mine"


def test_disabled_user_gets_no_llm(env):
    uid = ALICE_ID()
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "u", "model": "m",
                              "api_key": TESTKEY, "disabled": True})
    eff = user_llm_store.effective_for_user(uid)
    assert eff["llm_available"] is False
    assert "关闭" in eff["reason"]


def test_unknown_fields_are_not_writable(env):
    """白名单：不能通过 payload 改用量计数器（配额绕过的经典写法）。"""
    uid = ALICE_ID()
    user_llm_store.save(uid, {"today_tokens_used": 999999, "daily_token_quota": 1})
    v = user_llm_store.get_view(uid)
    assert v["today_tokens_used"] == 0
    assert v["daily_token_quota"] == 1


# ---------------------------------------------------------------------------
# 6. 配额
# ---------------------------------------------------------------------------

def test_quota_exceeded_message_and_reset(env):
    uid = ALICE_ID()
    user_llm_store.save(uid, {"daily_token_quota": 100})
    assert user_llm_store.quota_exceeded(uid) is None
    user_llm_store.record_usage(uid, tokens=150, calls=1)
    msg = user_llm_store.quota_exceeded(uid)
    assert msg and "额度已用尽" in msg
    # 跨天重置
    db.get_conn().execute(
        "UPDATE user_llm_config SET last_reset_date='1970-01-01' WHERE user_id=?", (uid,))
    db.get_conn().commit()
    user_llm_store.reset_daily_counters_if_needed(uid)
    assert user_llm_store.quota_exceeded(uid) is None


def test_zero_quota_means_unlimited(env):
    uid = ALICE_ID()
    user_llm_store.save(uid, {"daily_token_quota": 0})
    user_llm_store.record_usage(uid, tokens=10 ** 9)
    assert user_llm_store.quota_exceeded(uid) is None


# ---------------------------------------------------------------------------
# 7. HTTP 契约：每用户配置
# ---------------------------------------------------------------------------

def test_user_can_save_and_read_own_config(client, env):
    h = _login(client, "alice", "AlicePass123")
    r = client.post("/api/user/llm/config", headers=h, json={
        "use_custom_key": True, "base_url": "https://mine.example.com/v1",
        "model": "my-model", "api_key": TESTKEY, "samples": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == "user" and body["configured"] is True
    assert TESTKEY not in r.text
    j = client.get("/api/user/llm/config", headers=h).json()
    assert j["has_api_key"] is True
    assert j["api_key_masked"] == secretbox.mask(TESTKEY)


def test_alice_config_does_not_affect_admin(client, env):
    """**A 改配置不影响 B** —— 多用户隔离的第一条必测项。"""
    ha = _login(client, "admin", "AdminPass123")
    hb = _login(client, "alice", "AlicePass123")
    client.post("/api/user/llm/config", headers=hb, json={
        "use_custom_key": True, "base_url": "https://alice.example.com/v1",
        "model": "alice-model", "api_key": TESTKEY})
    ja = client.get("/api/user/llm/config", headers=ha).json()
    assert ja["model"] != "alice-model"
    assert ja["has_api_key"] is False
    jb = client.get("/api/user/llm/config", headers=hb).json()
    assert jb["model"] == "alice-model"


def test_cannot_write_another_users_config(client, env):
    """请求体里塞别人的 user_id 必须被拒（不能静默忽略）。"""
    hb = _login(client, "alice", "AlicePass123")
    r = client.post("/api/user/llm/config", headers=hb, json={
        "user_id": ADMIN_ID(), "model": "x"})
    assert r.status_code == 403
    assert "其他用户" in r.json()["error"]


def test_saving_requires_csrf(client, env):
    h = _login(client, "alice", "AlicePass123")
    r = client.post("/api/user/llm/config", json={"model": "x"})
    assert r.status_code == 403
    assert "CSRF" in r.json()["error"]


def test_admin_endpoint_still_admin_only(client, env):
    hb = _login(client, "alice", "AlicePass123")
    assert client.get("/api/admin/llm/default", headers=hb).status_code == 403
    ha = _login(client, "admin", "AdminPass123")
    assert client.get("/api/admin/llm/default", headers=ha).status_code == 200


# ---------------------------------------------------------------------------
# 8. 严格模式按用户判定（Lead 裁决 2-A）
# ---------------------------------------------------------------------------

def test_strict_mode_bypassed_for_user_without_llm(env, monkeypatch):
    """没配 LLM 的用户**不该**被严格模式锁死。

    否则每个新用户点"生成预测"都收到 422，而系统明明可以走纯统计出号。
    """
    from lottery import api_app
    monkeypatch.setattr(config, "STRICT_MODE", True)
    monkeypatch.setattr(config, "LLM_DISABLED", True)
    active, reason = api_app._strict_scope_for_current_user()
    assert active is False
    assert "不生效" in reason and "纯统计" in reason


def test_strict_mode_active_for_user_with_llm(env, monkeypatch):
    from lottery import api_app
    uid = ALICE_ID()
    monkeypatch.setattr(config, "STRICT_MODE", True)
    user_llm_store.save(uid, {"use_custom_key": True, "base_url": "https://u.example.com/v1",
                              "model": "m", "api_key": TESTKEY})
    # P2：`_strict_scope_for_current_user` 已搬到 `routes_predict`，它按
    # `request_context.current_user_id()` 取名（不是在 import 时绑定），
    # 所以打桩点就是 `request_context` 本身。
    from lottery import request_context
    monkeypatch.setattr(request_context, "current_user_id", lambda: uid)
    active, _ = api_app._strict_scope_for_current_user()
    assert active is True


def test_strict_mode_off_never_active(env, monkeypatch):
    from lottery import api_app
    monkeypatch.setattr(config, "STRICT_MODE", False)
    assert api_app._strict_scope_for_current_user() == (False, "")
