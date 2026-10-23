"""V28：LLM 票的逐注 L2 结构校验（含提示词硬约束）。

## 为什么需要这个文件

V18 把严格模式的门控依据改成 L2 结构约束，并承诺：

> 「任何人拿发过的票重跑 `check_shape()` 就能验证——比统计显著更硬。」

2026-10-23 生产核查发现这个承诺是**假的**：

    issue=2026122 seq=4  [1,6,14,22,30,32]
    check_shape() → False
    原因：奇数个数=1，而 odd_range=[2,3,4]

根因不是约束算错，而是**从来没有校验过票**：
  · `strict_mode_gates` 只检查 `shape_constraint_ctx()` 的 7 个必需键在不在；
  · `engine.llm_tickets` 里的 `ctx` 是**提示词上下文**（`build_context`），与结构约束无关；
  · 统计通道天然合规（`sample_stat_ticket` 内部按 ctx 采样），
    所以这个洞只在 LLM 通道暴露 —— 它吐的是自由文本号码串。

本文件的测试守四件事：
1. **判定正确**：违规票必须被识别，合规票不能被误判（双向）；
2. **`check_shape` 与 `explain_shape` 同源**：`ok == (violations == [])`；
3. **不丢弃、只标记**：解析层保留违规票（丢弃 = V16 那类静默降级）；
4. **门控真的会拒绝**：严格模式下违规 → 422，且带可操作信息。
"""
from __future__ import annotations

import pytest

from lottery import routes_predict as RP
from lottery import shape_rules as SR
from lottery import db


#: 2026-10-23 生产实际发出去的 5 注（issue=2026122）。
#: seq=4 是真实违规样本，其余 4 注合规 —— 用它做回归，防止「修好了但改错判据」。
PROD_TICKETS_2026122 = [
    {"reds": [7, 14, 16, 25, 26, 32], "blue": 4, "method": "stat:markov"},
    {"reds": [4, 8, 9, 18, 19, 29], "blue": 13, "method": "stat:omission"},
    {"reds": [2, 11, 13, 20, 26, 31], "blue": 16, "method": "stat:freq"},
    {"reds": [1, 6, 14, 22, 30, 32], "blue": 7, "method": "llm:group/subscription"},
    {"reds": [3, 8, 15, 24, 29, 31], "blue": 2, "method": "llm:group/subscription"},
]

#: 生产实际的违规点
PROD_BAD_REDS = [1, 6, 14, 22, 30, 32]


@pytest.fixture()
def draws(make_draws):
    """合成开奖数据（600 期），**不依赖生产库**。

    为什么不用真库：本项目 conftest 强制隔离（`LOTT_DB` 指向临时库），
    直连 `db.load_draws()` 会拿到空表并触发 config.py 的隔离守卫报错。
    而 L2 约束 ctx 只依赖「足够多的历史期」这一件事，合成数据完全够用。
    """
    from lottery import db as _db
    rows = make_draws(600, seed=42)
    _db.upsert_draws(rows)
    return _db.load_draws()


@pytest.fixture()
def shape_ctx(draws):
    return SR.shape_constraint_ctx(draws)


# ---------- 1. 判定正确（双向） ----------

def test_production_violating_ticket_is_caught(draws):
    """2026-10-23 生产实证的 seq=4 必须被识别为违规。

    变异实证：把 `verify_tickets_l2` 里的 `out["ok"] = (n_bad == 0)` 改成
    恒 True，本测试立刻变红。
    """
    r = RP.verify_tickets_l2(PROD_TICKETS_2026122, draws)
    assert r["ok"] is False
    assert r["n"] == 5
    assert r["n_bad"] == 1, f"应恰好 1 注违规，实际 {r['n_bad']}"
    bad = r["bad"][0]
    assert bad["seq"] == 4
    assert sorted(bad["reds"]) == PROD_BAD_REDS
    assert bad["method"].startswith("llm:"), "违规应发生在 LLM 通道"
    assert bad["violations"], "必须给出具体违反的维度"
    assert any("奇数" in v or "odd" in v for v in bad["violations"]), (
        f"违规原因应是奇偶比，实际 {bad['violations']}")


def test_compliant_tickets_pass(draws):
    """同一批里那 4 注合规的，不能被误判（防止「一刀切全否」）。"""
    good = [t for t in PROD_TICKETS_2026122 if sorted(t["reds"]) != PROD_BAD_REDS]
    r = RP.verify_tickets_l2(good, draws)
    assert r["ok"] is True
    assert r["n_bad"] == 0
    assert r["n"] == 4


def test_all_good_tickets_pass(draws):
    """全合规时 ok=True 且 note 提到「补不代表可预测」的诚实边界。"""
    good = [t for t in PROD_TICKETS_2026122 if sorted(t["reds"]) != PROD_BAD_REDS]
    r = RP.verify_tickets_l2(good, draws)
    assert r["ok"] is True
    assert "不代表" in r["note_zh"] or "L2 通过" in r["note_zh"]


# ---------- 2. check_shape 与 explain_shape 同源 ----------

def test_explain_shape_matches_check_shape(draws, shape_ctx):
    """`check_shape(r,ctx) == (explain_shape(r,ctx) == [])` 必须恒成立。

    两个函数若漂移，会出现「门控说合规、报告说违规」或反之 —— 用户无从相信。
    用真实历史开奖号 + 生产那 5 注 + 若干构造样本做双向核对。

    变异实证：把 `explain_shape` 里的 `odd_range` 判断删掉，本测试变红
    （seq=4 会出现 check=False 但 explain=[] 的矛盾）。
    """
    samples = [sorted(t["reds"]) for t in PROD_TICKETS_2026122]
    # 加真实历史开奖（应当大概率合规，但不必强求）
    for d in draws[-30:]:
        samples.append(sorted(d["reds"]))
    # 加明显违规的构造样本
    samples += [
        [1, 2, 3, 4, 5, 6],        # 和值/跨度/三区都极端
        [1, 3, 5, 7, 9, 11],       # 全奇数
        [2, 4, 6, 8, 10, 12],      # 全偶数
        PROD_BAD_REDS,
        [1, 2, 3, 31, 32, 33],
    ]
    for reds in samples:
        ok = SR.check_shape(reds, shape_ctx)
        viol = SR.explain_shape(reds, shape_ctx)
        assert ok == (viol == []), (
            f"{reds}: check_shape={ok} 但 explain_shape={viol} —— 两者必须同源")


def test_explain_shape_names_the_actual_metric(draws, shape_ctx):
    """违规说明必须点名**具体维度与数字**，不能只说「不合规」。"""
    viol = SR.explain_shape([1, 3, 5, 7, 9, 11], shape_ctx)   # 全奇数
    joined = "；".join(viol)
    assert viol
    assert "奇数" in joined
    # 必须带实际值，而不只是约束范围
    assert "6" in joined or "奇数个数=6" in joined


def test_explain_shape_handles_wrong_length():
    """红球个数不对时必须给明确错误，不能静默返回 []。"""
    viol = SR.explain_shape([1, 2, 3], {})
    assert viol
    assert "6" in viol[0]


# ---------- 3. 不丢弃、只标记 ----------

def test_parse_retains_violating_ticket_with_marker(shape_ctx):
    """解析层必须**保留**违规票并打标，而不是丢弃。

    为什么：丢弃会让 LLM 票悄悄变成 0 注 —— 与 V16 修的「静默降级」同类。
    用户会看到「5 注变 4 注」但不知道为什么。

    变异实证：把 `_parse_tickets_response` 里 `continue` 加在 `not _l2_ok`
    之后（即丢弃），本测试变红（长度 2 → 1）。
    """
    from lottery import engine as E
    res = {"tickets": [
        {"reds": [3, 8, 15, 24, 29, 31], "blue": 2, "confidence": 60},
        {"reds": PROD_BAD_REDS, "blue": 7, "confidence": 55},
    ]}
    out = E._parse_tickets_response(res, "testmodel", set(), shape_ctx)
    assert len(out) == 2, "违规票必须保留（不丢弃）"
    good, bad = out
    assert good.get("l2_ok") is True
    assert bad.get("l2_ok") is False
    assert bad.get("l2_violations"), "违规注必须带具体违规说明"


def test_parse_without_shape_ctx_leaves_l2_unset():
    """不传 shape_ctx 时（旧调用路径）不能瞎标 l2_ok，必须是 None。"""
    from lottery import engine as E
    res = {"tickets": [{"reds": [3, 8, 15, 24, 29, 31], "blue": 2, "confidence": 60}]}
    out = E._parse_tickets_response(res, "m", set())
    assert out[0].get("l2_ok") is None
    assert out[0].get("l2_violations") == []


# ---------- 4. 提示词硬约束 ----------

def test_tickets_prompt_contains_numeric_constraints(shape_ctx):
    """提示词必须带**可校验的数字阈值**，不能只说「结构均衡」。

    变异实证：删掉 `tickets_prompt` 里的 `shape_hint` 段，本测试变红。
    """
    from lottery import llm_client as LC
    p = LC.tickets_prompt({}, [], [], {}, shape_ctx=shape_ctx)
    assert "结构硬约束" in p
    # 五项约束的关键词都要在
    for kw in ("和值", "跨度", "三区比", "奇数个数", "连号"):
        assert kw in p, f"提示词缺少约束维度：{kw}"
    # 必须是具体数字（不是「合理范围」这种模糊说法）
    assert str(int(shape_ctx["odd_range"][0])) in p
    assert str(int(shape_ctx["sum_min"])) in p or "66" in p


def test_tickets_prompt_without_ctx_has_no_empty_section():
    """不给 shape_ctx 时不能出现空的约束段落（会造成模型困惑）。"""
    from lottery import llm_client as LC
    p = LC.tickets_prompt({}, [], [], {})
    assert "结构硬约束" not in p
    assert "请生成" in p, "主体提示词仍须完整"


# ---------- 5. 门控真的会拒绝 ----------

def test_strict_mode_rejects_batch_with_violation(draws):
    """严格模式下只要有 1 注违规，整批必须被拒（422 + can_fallback）。

    这里直接验证判据函数，不跑整条出号链路（LLM 调用不可控）。
    """
    r = RP.verify_tickets_l2(PROD_TICKETS_2026122, draws)
    assert r["ok"] is False
    # 门控的判据就是 r["ok"] —— 严格模式下据此 422
    assert r["n_bad"] >= 1


def test_verify_tickets_l2_empty_is_not_ok(draws):
    """空票单不能判为「合规」（vacuous truth 会掩盖出号失败）。"""
    r = RP.verify_tickets_l2([], draws)
    assert r["ok"] is False
    assert "没有票" in r["note_zh"]


def test_strict_rejection_message_is_actionable(draws):
    """严格模式拒绝时的文案必须告诉用户**该做什么**（D-20 纪律）。

    真实拒绝发生在 `predict()` 里构造 JSONResponse；这里复算它引用的字段，
    确保文案的每个成分都真的存在（否则会 KeyError 或印出 "?"）。
    """
    r = RP.verify_tickets_l2(PROD_TICKETS_2026122, draws)
    assert r["ok"] is False
    bad = r["bad"]
    assert bad, "拒绝路径依赖 bad 列表"
    first = bad[0]
    # 这三项是 422 文案里插值用的，缺任一就会印出问号
    assert first.get("seq") == 4
    assert first.get("reds") == PROD_BAD_REDS
    assert first.get("violations"), "文案需要具体违规项"
    # 判据必须有据可查（用户能自己重跑同一函数）
    assert "L2" in r["note_zh"]
    assert "check_shape" in r["note_zh"] or "复核" in r["note_zh"]
    # 拒绝路径的契约：reason 标识稳定 + 允许降级。
    # 用 AST 行为断言（项目 A 类守卫禁止 getsource 子串断言 —— 会被 docstring 骗）。
    from tests._source_assert import code_text
    src = code_text(RP.predict)
    assert '"l2_ticket_violation"' in src or "'l2_ticket_violation'" in src, (
        "拒绝 reason 标识必须稳定")
    assert "can_fallback" in src, "严格模式被拒必须允许用户选择降级"
    assert "422" in src, "逐注校验失败必须是 422（而非 500/400）"

def test_empty_tickets_do_not_trigger_l2_rejection(draws):
    """**空票单不得触发 L2 拒绝** —— 那是另一种失败，不能混为一谈。

    2026-10-23 实测踩到：`verify_tickets_l2([])` 返回 ok=False（防 vacuous truth
    是对的），但若 predict 据此 422，文案会变成
    「0/0 注不满足结构约束（第?注 []：）」—— 自相矛盾且掩盖真正原因。

    判据：空票单 → ok=False 但 n=0，predict 侧必须用 `n > 0` 门控才拒绝。
    变异实证：删掉 `_l2_reject` 里的 `int(_l2v["n"]) > 0`，
    `test_strict_on_allows_when_rules_available` 变红。
    """
    r = RP.verify_tickets_l2([], draws)
    assert r["ok"] is False, "空票单不能判合规（vacuous truth）"
    assert r["n"] == 0
    assert "没有票" in r["note_zh"]
    # predict 的拒绝判据必须排除 n=0 的情形
    from tests._source_assert import code_text
    src = code_text(RP.predict)
    assert '_l2_reject' in src, "拒绝判据必须有名字，便于审计"
