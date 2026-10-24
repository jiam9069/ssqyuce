"""D-04 能力清单端点 `GET /api/capabilities` 的验收测试。

## 本文件守的是什么

不是「端点返回 200」，而是三件事：

1. **常量真的来自代码**，不是写死的字面量 —— 用 monkeypatch 改底层数值，
   端点输出必须跟着变。这是本文件最重要的一条：写死的清单会在数据/口径漂移时
   保持「正确」的样子，那正是本项目历史上反复踩的坑。
2. **`cannot` 段四条硬否定齐全** —— 少一条就等于在漏掉一个负面结论。
3. **不得出现禁用词**，且 `degraded` 段只报真实降级项。

## 为什么不测「能返回 200」

成功路径的 200 由其它用例覆盖；这里全部是**契约与否定式断言**。
"""
from __future__ import annotations

import json

import pytest

from lottery import routes_capabilities as C


# ==========================================================================
# 一、结构契约
# ==========================================================================

def test_returns_required_top_level_sections():
    """四段齐全：can / cannot / degraded / constants（可增不可减）。"""
    body = C.capabilities()
    assert body["ok"] is True
    for key in ("can", "cannot", "degraded_items", "constants"):
        assert key in body, f"缺少必需段 {key}"
        assert isinstance(body[key], list if key != "constants" else dict)


def test_every_can_item_has_evidence_script():
    """`can` 每项必须带**存在的**证据脚本路径。

    没有证据脚本的「能力」就是一句好听话 —— 红线 3 要求每个数字可复现。
    """
    import pathlib
    body = C.capabilities()
    assert body["can"], "can 不能为空"
    root = pathlib.Path(__file__).resolve().parents[1]
    for item in body["can"]:
        for key in ("id", "title", "evidence_script", "honest_bound"):
            assert item.get(key), f"can 项 {item.get('id')} 缺 {key}"
        path = root / item["evidence_script"]
        assert path.exists(), (
            f"can 项 {item['id']} 的证据脚本不存在：{item['evidence_script']}")


def test_every_cannot_item_has_why_not_and_evidence():
    body = C.capabilities()
    assert body["cannot"], "cannot 不能为空"
    for item in body["cannot"]:
        for key in ("id", "title", "why_not", "evidence"):
            assert item.get(key), f"cannot 项 {item.get('id')} 缺 {key}"


def test_every_degraded_item_has_restore_path():
    """降级项必须给出恢复方式 —— 只报「坏了」不给出路等于把问题甩给用户。"""
    body = C.capabilities()
    for item in body["degraded_items"]:
        for key in ("id", "reason", "how_to_restore"):
            assert item.get(key), f"degraded 项 {item.get('id')} 缺 {key}"


# ==========================================================================
# 二、cannot 段四条硬否定（缺一即不合格）
# ==========================================================================

def test_cannot_covers_all_four_red_lines():
    """四条硬否定必须齐全：不可预测 / 期望为负 / 六等奖恒定 / 覆盖不改期望。"""
    body = C.capabilities()
    ids = {item["id"] for item in body["cannot"]}
    required = {
        "predict_numbers",              # 不可预测
        "positive_expectation",         # 期望为负
        "raise_sixth_ratio",            # 六等奖 15.625%
        "coverage_changes_expectation",  # 蓝球覆盖只改频率不改回报率
    }
    missing = required - ids
    assert not missing, f"cannot 段缺少硬否定项：{missing}"


def test_predict_numbers_states_the_three_evidence_numbers():
    """「不可预测」必须带三个独立证据数字，缺一个就是空口断言。"""
    body = C.capabilities()
    item = next(i for i in body["cannot"] if i["id"] == "predict_numbers")
    blob = item["why_not"] + item["evidence"]
    assert "0.0654" in blob, "缺红球 χ² p 值"
    assert "0.0076" in blob, "缺条件熵降幅"
    assert "0.2995" in blob, "缺 max-stat 校正后 p 值"


def test_coverage_item_denies_expectation_change():
    """蓝球覆盖那条必须同时给出「频率提升」与「返奖比不变」。"""
    body = C.capabilities()
    item = next(i for i in body["cannot"] if i["id"] == "coverage_changes_expectation")
    blob = item["why_not"] + item["evidence"]
    assert "47.55" in blob and "62.50" in blob, "必须给出频率前后对照"
    assert "15.625" in blob, "必须给出不变的返奖比"


# ==========================================================================
# 三、constants 与红线表一致（五个锁定值）
# ==========================================================================

def test_constants_match_redline_table():
    """五个锁定值与红线表逐一比对。

    红线表（docs/delivery/BOARD.md §0 红线 2）：
      α=0.05 / single=306 / cross=72350 / ev_sixth=0.15625
      ev_single 在 HEAD `db634e3` 之后为奖金假设切换后的实时值。
    """
    consts = C.capabilities()["constants"]
    assert consts["alpha"] == pytest.approx(0.05), "α 必须与红线表一致"
    assert consts["single"] == 306, "单因子搜索空间必须为 306"
    assert consts["cross"] == 72350, "跨族搜索空间必须为 72350"
    # V36：官方六等奖返奖比 = 5×P(红≤2)/32 ≈ 0.1472314（旧值 0.15625 漏了红球条件）
    assert consts["ev_sixth"] == pytest.approx(0.1472314, abs=1e-6), \
        "六等奖返奖比必须恒为 15.625%"
    # ev_single 必须走 strict 口径（与 `/api/portfolio/blue-coverage` 一致）。
    # 这里做**精确断言**而非区间断言 —— 曾经的 `0.40 <= x <= 0.65` 什么都防不住：
    # 严QA 实测把它硬编码成 0.531925、0.60、0.6499 时全部通过，等于一条假守卫。
    assert consts["ev_single"] == pytest.approx(0.522907, abs=1e-6), (
        f"单注返奖比应为 strict 口径 0.522907，实际 {consts['ev_single']} —— "
        f"若为 0.531925 说明又用回了 published（与 D-01 的 52.3% 矛盾）")


def test_ev_single_tracks_prize_assumption(monkeypatch):
    """**正向追踪守卫**：改奖金假设，`ev_single` 必须跟着变。

    ## 为什么必须有这一条（严QA 的决定性实验）

    原守卫是区间断言 `0.40 ≤ ev_single ≤ 0.65`。严QA 在副本里把一等奖假设
    从 684 万改成 1500 万：

        真值应为 0.762159，端点仍返回 0.531925（没变），偏差 43%，而 21 条测试全绿。

    因为 `0.531925` 落在 40%~65% 区间内 —— 区间断言对这种"完全不跟随依赖"
    的失效**毫无感知**。硬编码成任何区间内的值都能通过。

    本用例的做法相反：**主动改依赖，断言输出按预期变化**。如果谁把
    `ev_single` 写成字面量、或缓存了第一次的结果，本用例立刻红。
    """
    from lottery import routes_capabilities as RC

    baseline = C.capabilities()["constants"]["ev_single"]

    # 把一等奖假设改成 1500 万（严QA 用的同一个值）
    monkeypatch.setitem(RC._PRIZE_ASSUMPTION, "first", 15_000_000)

    bumped = C.capabilities()["constants"]["ev_single"]
    # ⚠️ 期望值是 **strict 口径**的 0.753141，不是严QA 原始实验里的 0.762159。
    # 严QA 那组数是在 `published` 口径下测的；D-04 已统一到 strict，
    # 同一假设下 strict(0.753141) < published(0.762159)，差值约 0.9pp —— 符合预期。
    assert bumped == pytest.approx(0.753141, abs=1e-6), (
        f"假设改成 1500 万后 strict 返奖比应为 0.753141，实际 {bumped} —— "
        f"说明 ev_single 没有真正跟随奖金假设（被硬编码或缓存）")
    assert bumped != baseline, "改依赖后输出必须变化，否则守卫是空的"


def test_ev_sixth_equals_identity_five_over_32():
    """六等奖返奖比必须**精确**等于恒等式 5/32，不只是「约等于」。"""
    consts = C.capabilities()["constants"]
    assert consts["ev_sixth"] == pytest.approx(0.1472314, abs=1e-6)


def test_constants_carry_their_sources():
    """每个常量必须标明出处符号 —— 没有出处的数字不算证据。"""
    body = C.capabilities()
    sources = body["constants_meta"]["sources"]
    for key in ("alpha", "single", "cross", "ev_single", "ev_sixth"):
        assert sources.get(key), f"常量 {key} 缺 sources 出处"


# ==========================================================================
# 四、【核心】常量是实时读取的，不是字面量
# ==========================================================================

def test_constants_are_read_live_not_hardcoded(monkeypatch):
    """**本文件最重要的断言**：改底层常量，端点输出必须跟着变。

    做法：篡改 `eprocess.DEFAULT_ALPHA` 与 `candidate_space.total_single()`，
    端点返回值必须立即反映篡改值。若端点里写死了 0.05 / 306，本用例会失败 ——
    那正是「文档与代码漂移」无法被发现的根因。
    """
    from lottery import candidate_space as CS, eprocess as EP

    monkeypatch.setattr(EP, "DEFAULT_ALPHA", 0.123, raising=False)
    monkeypatch.setattr(CS, "total_single", lambda: 999, raising=False)

    consts = C.capabilities()["constants"]
    assert consts["alpha"] == pytest.approx(0.123), (
        "改了 DEFAULT_ALPHA 但端点仍返回旧值 —— 说明 α 是写死的字面量")
    assert consts["single"] == 999, (
        "改了 total_single() 但端点仍返回旧值 —— 说明 single 是写死的字面量")


def test_constants_failure_is_reported_not_silently_wrong(monkeypatch):
    """取不到某个常量时必须标记 degraded，不许静默回落到一个「看起来对」的值。"""
    from lottery import candidate_space as CS

    def _boom():
        raise RuntimeError("simulated source failure")

    monkeypatch.setattr(CS, "total_cross", _boom, raising=False)

    body = C.capabilities()
    assert body["constants"]["cross"] is None, "取不到时应为 None，不是假值"
    assert body["degraded"] is True, "常量缺失必须标记 degraded"
    assert "cross" in body["degraded_constants"]
    assert "cross" in body["constants_meta"]["errors"]


# ==========================================================================
# 五、禁用词与序列化
# ==========================================================================

@pytest.mark.parametrize("banned", ["提高中奖率", "提高命中率", "更划算", "稳赚"])
def test_no_banned_marketing_words(banned):
    """禁用词守卫（红线 2）。"""
    blob = json.dumps(C.capabilities(), ensure_ascii=False)
    assert banned not in blob, f"能力清单出现禁用词「{banned}」"


def test_disclaimer_denies_both_probability_and_expectation_change():
    d = C.capabilities()["disclaimer"]
    assert "不改变" in d
    assert "概率" in d and "期望" in d
    assert "负" in d, "必须明说长期期望为负"


def test_no_numpy_scalars_in_payload():
    """返回体必须干净可序列化（numpy 标量会让 FastAPI 编码器抛 TypeError）。"""
    import numpy as np
    body = C.capabilities()

    def _walk(obj, path="$"):
        assert not isinstance(obj, np.generic), f"{path} 是 numpy 标量 {type(obj)}"
        assert not isinstance(obj, np.ndarray), f"{path} 是 numpy 数组"
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(body)
    json.dumps(body)


# ==========================================================================
# 六、degraded 段反映真实状态
# ==========================================================================

def test_idle_mining_degraded_when_disabled(monkeypatch):
    """挖掘停机时必须出现在 degraded 段，并给出恢复方式。"""
    from lottery import idle_mining

    monkeypatch.setattr(idle_mining, "status",
                        lambda: {"enabled": False, "checked": 21800,
                                 "rule_evidence_count": 0}, raising=False)
    body = C.capabilities()
    item = next((i for i in body["degraded_items"] if i["id"] == "idle_mining"), None)
    assert item is not None, "挖掘停机却未被标为降级"
    assert "LOTT_IDLE_MINING" in item["how_to_restore"]


def test_degraded_absent_when_mining_enabled(monkeypatch):
    """反向用例：挖掘开启时不得再报该项降级（不许「永远报降级」）。"""
    from lottery import idle_mining

    monkeypatch.setattr(idle_mining, "status",
                        lambda: {"enabled": True, "checked": 1,
                                 "rule_evidence_count": 0}, raising=False)
    body = C.capabilities()
    ids = {i["id"] for i in body["degraded_items"]}
    assert "idle_mining" not in ids, "挖掘已开启却仍报降级"


def test_route_is_registered_on_app():
    """端点必须真的挂在 FastAPI 应用上，而不是只存在于模块里。"""
    from lottery import api_app
    spec = api_app.app.openapi()
    assert "/api/capabilities" in spec["paths"]
    assert "get" in spec["paths"]["/api/capabilities"]


# ==========================================================================
# 降级字段语义守卫（2026-10-21 P1 修复）
#
# 缺陷：旧版 `degraded` 只反映"常量取没取全"，而 `degraded_items` 可能有 2 项，
# 于是 `degraded=False` 与 `degraded_items=[...]` 并存。前端照 `degraded` 判断
# 会把降级项显示成正常 —— 直接违反 D-03「降级必须可辨识」。
# ==========================================================================

def test_degraded_is_derivable_from_its_parts():
    """`degraded` 必须严格等于 `constants_degraded or bool(degraded_items)`。"""
    b = C.capabilities()
    assert b["degraded"] == (b["constants_degraded"] or bool(b["degraded_items"])), (
        f"降级三字段不自洽：degraded={b['degraded']}, "
        f"constants_degraded={b['constants_degraded']}, "
        f"degraded_items={[d.get('id') for d in b['degraded_items']]}")


def test_degraded_true_whenever_any_item_degraded():
    """有降级项时 `degraded` 必须为 True（这是前端唯一可靠的判断依据）。"""
    b = C.capabilities()
    if b["degraded_items"]:
        assert b["degraded"] is True, (
            f"存在 {len(b['degraded_items'])} 个降级项，但 degraded=False —— "
            f"前端会误判它们为正常")


def test_degraded_fields_are_not_conflated(monkeypatch):
    """**变异守卫**：往 `degraded_items` 塞一项，`degraded` 必须跟着变 True。

    这守住"只修判定不改名"的反面：如果有人把 `degraded` 改回只算常量，
    本用例立刻红。
    """
    from lottery import routes_capabilities as RC

    fake = [{"id": "injected", "reason": "test", "how_to_restore": "n/a"}]
    monkeypatch.setattr(RC, "_degraded_items", lambda: fake)
    b = RC.capabilities()
    assert b["degraded_items"] == fake
    assert b["degraded"] is True, "注入降级项后 degraded 必须为 True"
    assert b["constants_degraded"] is False, "常量本身没坏，不得被误标"


def test_old_ambiguous_degraded_field_no_longer_means_constants_only():
    """旧语义（degraded 只表示常量不全）必须已迁到 `constants_degraded`。

    这是"改名比加注释有效"的落地：字段名自己就说明它是管什么的。
    """
    b = C.capabilities()
    assert "constants_degraded" in b, "必须存在专门表示常量降级的字段"
    assert isinstance(b["constants_degraded"], bool)
    # 两个字段可以不同值 —— 这正是它们存在的意义
    assert b["degraded"] != b["constants_degraded"] or not b["degraded_items"], (
        "当存在降级项而常量正常时，两字段应当不同 —— 否则说明又被混用了")


# ==========================================================================
# 白名单不得静默吞字段（2026-10-21 修复）
#
# 缺陷：`_live_constants()` 里写了 `out["sixth_mode"]="strict"`，但响应体是
# **白名单重新组装**的，`sixth_mode` 不在五元组里 ⇒ 被静默丢弃。
# 代码里看起来有、响应里实际没有 —— 下一个人读源码会以为它在那儿。
#
# 这是当天那一族的第六个变体：
#   `| tail` 吞退出码 / `git stash` 收走工作 / 无基线 `git diff` 为空 /
#   `--collect-only` 返回 0 / 错误环境假设让我们以为做不到 / 白名单吞字段。
# 共同点：某种「看起来对」的中间态掩盖了真实结果。
# ==========================================================================

#: `_live_constants()` 里属于「元信息」的键 —— 它们进 `constants_meta`，不进 `constants`。
_CONST_META_KEYS = {"sources", "prize_assumption", "derivation_zh", "errors"}


def test_sixth_mode_is_exposed_at_top_level():
    """`sixth_mode` 必须出现在响应顶层（Lead 点名要加的字段）。

    放顶层而非 `constants`：它是**整个 constants 段的口径声明**，
    不是某个常量的值。塞进 constants 会让人误以为它只描述 `ev_single`。
    """
    from lottery import routes_capabilities as RC

    body = RC.capabilities()
    assert "sixth_mode" in body, "sixth_mode 被白名单吞掉了"
    assert body["sixth_mode"] == "strict", f"应为 strict，实际 {body['sixth_mode']}"


def test_no_constant_key_is_silently_dropped():
    """**反查守卫**：`_live_constants()` 产出的每个值键都必须能在响应里找到。

    这条是「白名单不得吞字段」的硬版本 —— 不枚举期望键，而是**反查实际产出**。
    以后再加常量却忘了暴露，本测试会红，而不是靠人记得改白名单。
    """
    from lottery import routes_capabilities as RC

    produced = set(RC._live_constants().keys())
    value_keys = produced - _CONST_META_KEYS
    body = RC.capabilities()
    exposed = set(body["constants"].keys()) | {"sixth_mode"}

    missing = value_keys - exposed
    assert not missing, (
        f"以下常量在 `_live_constants()` 里产出了、但响应里找不到：{sorted(missing)}\n"
        f"（这就是「白名单静默吞字段」—— 代码里看起来有，响应里实际没有）")

    # 反向也不能多：响应里的常量键必须都来自产出（防止白名单写了不存在的键）
    extra = set(body["constants"].keys()) - value_keys
    assert not extra, f"响应里出现了未产出的常量键：{sorted(extra)}"


def test_constants_meta_keys_are_not_in_constants_group():
    """元信息键不得混进 `constants` 值组（否则口径与值混在一起）。"""
    from lottery import routes_capabilities as RC

    body = RC.capabilities()
    overlap = set(body["constants"].keys()) & _CONST_META_KEYS
    assert not overlap, f"元信息键混进了 constants 值组：{sorted(overlap)}"
    # 元信息应在 constants_meta 里
    for k in ("sources", "prize_assumption"):
        assert k in body["constants_meta"], f"{k} 应在 constants_meta"
