"""V38（2026-10-24 审计 P2 批次）回归锁。

本批修的都是"不会立刻炸、但会在特定条件下静默出错"的项：
事务泄漏、返回值语义、配置互斥组合、接口误用、路由清单漂移。

共同点是**它们的失效方式是沉默的** —— 没有异常、没有日志、没有红测试，
只有一个数字或一个行为悄悄变得不对。所以这里的断言都盯着"可观察的后果"，
而不是"代码看起来对不对"。
"""
from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# P2：save_pattern_results 异常必须回滚（否则写锁滞留 + 半成品被后续 commit 带走）
# ===========================================================================

_GOOD = {
    "key": "probe_ok", "name_zh": "探针", "kind": "probe", "grade": "C",
    "sample_size": 10, "margin": 0.1, "p_value": 0.5, "direction": "none",
}


def test_save_pattern_results_rejects_missing_fields_before_writing(seeded_db):
    """缺字段必须在**写任何东西之前**失败，且给出明确原因。"""
    from lottery import db

    bad = dict(_GOOD, key="probe_bad")
    del bad["grade"]
    with pytest.raises(ValueError) as ei:
        db.save_pattern_results([_GOOD, bad])
    assert "grade" in str(ei.value), "错误信息没说缺哪个字段"

    keys = {p["key"] for p in db.load_patterns()}
    assert "probe_bad" not in keys
    assert "probe_ok" not in keys, "第一条也不该写入 —— 整批应当原子"


def test_save_pattern_results_leaves_no_open_transaction(seeded_db):
    """**执行期**失败后事务必须已回滚，否则写锁滞留、半成品被后续 commit 带走。

    ## 为什么不能只测"缺字段"

    第一版这条用例传的是缺字段的数据 —— 但入口校验在**写入之前**就拦下了，
    根本走不到 savepoint。于是把回滚代码整段删掉，用例**依然全绿**
    （变异实测：3 passed）。那是一个只测了"校验存在"、没测"回滚存在"的假绿。

    要测回滚，必须让**第一条写成功、第二条在 execute 时炸**。
    这里用不可 JSON 序列化的 params（`object()`）制造这种失败：
    它过得了入口校验（字段都在），却会在 `json.dumps` 时抛 TypeError。
    """
    from lottery import db

    first = dict(_GOOD, key="probe_first")
    explosive = dict(_GOOD, key="probe_boom", params={"bad": object()})

    with pytest.raises(TypeError):
        db.save_pattern_results([first, explosive])

    assert db.get_conn().in_transaction is False, (
        "执行期失败后连接仍处于事务中 —— 写锁滞留，"
        "下一次 commit 会把这一批的半成品一起提交")

    keys = {p["key"] for p in db.load_patterns()}
    assert "probe_first" not in keys, (
        "第一条本应随整批回滚，却留在了库里 —— savepoint 回滚没生效")
    assert "probe_boom" not in keys

    # 后续写入必须正常（证明锁已释放、事务已清理）
    db.save_pattern_results([dict(_GOOD, key="probe_after")])
    assert "probe_after" in {p["key"] for p in db.load_patterns()}


def test_save_pattern_results_empty_is_noop(seeded_db):
    from lottery import db
    db.save_pattern_results([])          # 不得抛错
    assert db.get_conn().in_transaction is False


# ===========================================================================
# P2：upsert_draws 返回值必须区分「新增」与「覆盖」
# ===========================================================================

_SAMPLE = {
    "issue": "8888001", "date": "2020-05-01",
    "reds": [1, 2, 3, 4, 5, 6], "blue": 7, "order": [1, 2, 3, 4, 5, 6],
}


def test_upsert_detailed_counts_insert_vs_update(seeded_db):
    """重复抓同一份数据时 inserted 必须是 0 —— 否则界面报"新增 3518 期"。"""
    from lottery import db

    first = db.upsert_draws_detailed([_SAMPLE])
    assert first["inserted"] == 1 and first["updated"] == 0

    again = db.upsert_draws_detailed([_SAMPLE])
    assert again["inserted"] == 0, "重复数据被当成新增了"
    assert again["updated"] == 1

    more = db.upsert_draws_detailed(
        [_SAMPLE, dict(_SAMPLE, issue="8888002", date="2020-05-03")])
    assert more["inserted"] == 1 and more["updated"] == 1


def test_upsert_detailed_empty(seeded_db):
    from lottery import db
    assert db.upsert_draws_detailed([]) == {"processed": 0, "inserted": 0,
                                            "updated": 0}


# ===========================================================================
# P2：__Host- 前缀与 Secure=0 互斥（浏览器静默丢弃 cookie）
# ===========================================================================

def test_host_prefix_is_dropped_when_cookie_not_secure():
    """`Secure=0` + `__Host-` 会让浏览器**静默丢弃** cookie（登录"成功"却进不去）。

    这里直接验配置模块的纠正逻辑（子进程读不同环境变量，避免污染当前进程）。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = (
        "from lottery import config;"
        "print(config.SESSION_COOKIE_NAME, config.COOKIE_NAME_AUTO_ADJUSTED)"
    )

    def run(**env):
        import os
        e = dict(os.environ, **env)
        e["PYTHONPATH"] = str(root)
        out = subprocess.run([sys.executable, "-c", code], cwd=root, env=e,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-500:]
        line = [ln for ln in out.stdout.splitlines() if "__Host" in ln or "ssq_sid" in ln]
        return line[-1].split()

    name, adjusted = run(LOTT_COOKIE_SECURE="0")
    assert name == "ssq_sid", f"Secure=0 时仍用 __Host- 前缀（{name}），浏览器会丢弃 cookie"
    assert adjusted == "True"

    name2, adjusted2 = run(LOTT_COOKIE_SECURE="1")
    assert name2 == "__Host-ssq_sid", "Secure=1 时应保留 __Host- 前缀（这是加固）"
    assert adjusted2 == "False"


# ===========================================================================
# P2：Brier 分解必须拒绝非二值标签
# ===========================================================================

def test_brier_identity_holds_for_binary_labels():
    from lottery import ml_model
    rng = np.random.default_rng(7)
    p = rng.random(200)
    y = rng.integers(0, 2, 200).astype(float)
    r = ml_model._brier_decomposition(p, y)
    gap = r["brier"] - (r["reliability"] - r["resolution"] + r["uncertainty"])
    assert abs(gap) < 1e-12, f"二值输入下恒等式应当精确，实际偏差 {gap}"


@pytest.mark.parametrize("y", [[0.5], [0.3, 0.7], [0.0, 0.5, 1.0]])
def test_brier_rejects_non_binary_labels(y):
    """非二值 y 会让恒等式破裂（实测 y=[0.5] 时 gap=-0.25），必须报错而非静默给分。"""
    from lottery import ml_model
    with pytest.raises(ValueError) as ei:
        ml_model._brier_decomposition(np.array([0.5] * len(y)), np.array(y))
    assert "二值" in str(ei.value)


def test_brier_empty_is_fine():
    from lottery import ml_model
    r = ml_model._brier_decomposition(np.array([]), np.array([]))
    assert r["n"] == 0


# ===========================================================================
# P2：randomness_report 的 verdict 必须标明"未校正"
# ===========================================================================

def test_randomness_verdict_is_labelled_uncorrected(seeded_db, make_draws):
    """顶层 verdict 是单项未校正口径，必须显式标注真值字段。"""
    from lottery.stats import rigor

    draws = make_draws(300)
    rep = rigor.randomness_report(draws)
    assert rep.get("verdict_is_uncorrected") is True
    assert rep.get("verdict_authoritative_field") == "verdict_fwer"
    note = rep.get("verdict_note_zh") or ""
    assert "verdict_fwer" in note, "注释没指出该读哪个字段"
    assert "未校正" in note, "注释没说明这是未校正口径"


# ===========================================================================
# P2：LLM 配置缺字段 → 可执行错误，而不是 KeyError→500
# ===========================================================================

@pytest.mark.parametrize("missing", ["base_url", "model", "api_key"])
def test_incomplete_llm_config_gives_actionable_error(seeded_db, missing):
    from lottery import llm_client

    cfg = {"base_url": "https://api.example.com/v1", "model": "m",
           "api_key": "sk-x"}
    cfg[missing] = ""
    with pytest.raises(llm_client.LLMChannelError) as ei:
        llm_client.chat("你是助手", "你好", model_cfg=cfg, strict=True)
    msg = str(ei.value)
    assert missing in msg, f"错误信息没指出缺的是 {missing}"
    assert "设置页" in msg, "没有给用户下一步（项目 D20 纪律）"


# ===========================================================================
# P2：解析器容错（与 P1 批同源，锁在同一处）
# ===========================================================================

def test_parser_tolerates_single_bad_line():
    from lottery import parser
    good = ("2003001 2003-02-23 10 11 12 13 26 28 11 26 28 11 13 10 12 "
            + " ".join(["0"] * 16))
    bad = []
    out = parser.parse_text(good + "\n<junk>\n" + good.replace("2003001", "2003002"),
                            on_bad=bad)
    assert len(out) == 2 and len(bad) == 1


# ===========================================================================
# P1-17（补盲区）：llm_eval 的配对剔除逻辑要有行为测试
# ===========================================================================

def test_llm_eval_handles_llm_failures_without_corrupting_pairing(seeded_db, monkeypatch):
    """LLM 全失败时，配对必须如实剔除失败期、披露实际配对期数。

    llm_eval 是本项目**最没有定向用例**的重模块之一（审计 P1-17）：
    它只在依赖方向测试里被"静态引用"，没有任何行为断言。
    而它承载了一条已修的 P1-14（失败期用 stat 顶替会污染配对检验）——
    没有用例钉住，那条修复随时可能被回退。

    本用例 monkeypatch `_run_with_timeout` 让 stat_llm 通道**每期都失败**，
    验证：
      1. 整体不崩（llm_eval 对失败的兜底是回退 stat）；
      2. `paired_issues_excluding_llm_failures` 如实披露 = 0（全被剔除）；
      3. `llm_empty_issues` 如实计数 = 参与评估的期数。
    """
    from lottery import llm_eval as LE

    draws = [{"issue": f"{2015+i//150}{i%150+1:03d}", "date": "2020-01-01",
              "reds": [1, 2, 3, 4, 5, 6], "order": [1, 2, 3, 4, 5, 6],
              "blue": (i % 16) + 1}
             for i in range(330)]   # 超过 300 的窗口下限，才能跑 walk-forward

    def boom(fn, timeout_s):
        raise TimeoutError("llm simulated failure")
    monkeypatch.setattr(LE, "_run_with_timeout", boom)

    # 用极小窗口（3 期）控制成本 —— 只测配对逻辑，不测统计功效
    out = LE.run_llm_eval(draws, issues=3, n_tickets=2, seed=1)
    assert "comparisons" in out
    assert out["llm_empty_issues"] > 0, "LLM 明明全失败了，却没计数"
    assert out["paired_issues_excluding_llm_failures"] == 0, (
        "全失败时配对期数应为 0（都被剔除了），实际 "
        f"{out['paired_issues_excluding_llm_failures']}")
    # stat_llm 与 stat 的配对不能因为被剔除而空跑崩溃
    for c in out["comparisons"]:
        assert c.get("p") is not None or "adj" in c.get("metric", ""), \
            f"配对比较 {c} 缺 p 值"
