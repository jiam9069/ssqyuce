"""V39（2026-10-24 审计 P2 收尾批次）回归锁。

这一批处理的是"边界与口径"类问题：它们不改变主要行为，但会在特定输入下
给出**看起来合理、实际错误**的结果。共同特征是沉默——没有异常、没有日志。

因此这里的断言都盯着"边界输入的表现"与"默认值的方向"：
默认值错了，所有没显式传参的调用方都会中招，而没人会发现。
"""
from __future__ import annotations

import pytest


# ===========================================================================
# 六等奖口径：默认必须是 strict（互斥奖级），不能是 published（重复计数）
# ===========================================================================

def test_single_ticket_ev_defaults_to_strict():
    """`_single_ticket_ev` 不传 sixth 时必须走 strict。

    为什么重要：`sixth="published"` 把六等奖按"蓝球命中"的**边际**概率算，
    与三/四/五等奖重复计数，概率合计 0.0707 > 官方 0.0671，EV 高估 2.01%。
    它只应作为"对外发表口径"用于并排对照，**不该是默认值** ——
    默认 published 意味着任何忘记传参的新调用方都会静默拿到错误的数。
    """
    from lottery import engine_distribution as ED

    default = ED._single_ticket_ev()
    strict = ED._single_ticket_ev(sixth="strict")
    published = ED._single_ticket_ev(sixth="published")

    assert default["ev_yuan"] == pytest.approx(strict["ev_yuan"], rel=1e-12), (
        "默认口径不是 strict —— 未显式传 sixth 的调用方会拿到重复计数口径")
    # 反向锁：published 确实更高（重复计数的方向），且两者必须可区分
    assert published["ev_yuan"] > strict["ev_yuan"], (
        "published 应当因重复计数而略高于 strict；若相等说明两口径被合并了")
    # 重复计数的**正确**判据：published 的奖级概率合计 > strict 的。
    #
    # 注意别写成"合计 > 1"：`by_level_prob` 是**各奖级的中奖概率**（量级 0.07），
    # 不是归一化分布，其合计本来就远小于 1。重复计数体现在
    # published 把六等奖按"蓝中"的边际概率 0.0625 计入，而 strict 只计
    # "蓝中且红≤2" 的 0.0589 —— 前者更大，差值正是被重复计入的那部分。
    sum_strict = sum(strict["by_level_prob"].values())
    sum_published = sum(published["by_level_prob"].values())
    assert sum_strict <= 1.0 + 1e-12, "strict 口径的奖级概率合计不应超过 1"
    assert sum_published > sum_strict + 1e-9, (
        "published 口径应当因重复计数而比 strict 多算一部分概率")


# ===========================================================================
# sigma_hypergeom 边界：越界必须报错，不能静默返回 0
# ===========================================================================

@pytest.mark.parametrize("k", [-1, -0.5, 34, 50, 100])
def test_sigma_hypergeom_rejects_out_of_range(k):
    """k 越界（<0 或 >R_MAX）必须 ValueError。

    旧实现把负值钳成 0、把 >R_MAX 用 `max(fpc,0)` 变成方差 0 —— 于是
    k=-5 与 k=50 都"安全地"返回 0.0，看起来没问题，实际是**调用方传错
    被悄悄吞掉**。k 超过号码总数通常说明筛选集构造有 bug，不是正常边界。
    """
    from lottery.stats.eprocess import sigma_hypergeom
    with pytest.raises(ValueError) as ei:
        sigma_hypergeom(k)
    assert "越界" in str(ei.value)


@pytest.mark.parametrize("k,expect_positive", [(0, False), (1, True), (6, True)])
def test_sigma_hypergeom_valid_range(k, expect_positive):
    """合法区间内仍正常返回；k=0 时方差为 0（空筛集，无命中）。"""
    from lottery.stats.eprocess import sigma_hypergeom
    s = sigma_hypergeom(k)
    assert (s > 0) is expect_positive, f"k={k} 得到 sigma={s}"


def test_sigma_hypergeom_k_equals_full_set_is_zero():
    """k=R_MAX 时 sigma=0 是**数学正确**（全集必中 6 个，方差为 0）。

    审计曾把它当"无守卫"报出来；实测 k=33 返回 0 是对的。
    这里把"正确"钉住，避免后人为了消除"0"而改成错误的值。
    """
    from lottery.stats.eprocess import sigma_hypergeom
    assert sigma_hypergeom(33) == 0.0


def test_sigma_hypergeom_blue_bound():
    """蓝球 target 的合法区间是 0..B_MAX(16)，17 越界。"""
    from lottery.stats.eprocess import sigma_hypergeom
    assert sigma_hypergeom(16, target="blue") >= 0.0
    with pytest.raises(ValueError):
        sigma_hypergeom(17, target="blue")


# ===========================================================================
# 闲时挖掘限流：容器内应读 cgroup 而非宿主 /proc/stat
# ===========================================================================

def test_cgroup_readers_degrade_gracefully():
    """cgroup 读不到时必须返回 None（调用方回退），不得抛异常。

    非容器环境没有 `/sys/fs/cgroup/cpu.max`；某些 cgroup v1 环境也没有
    `cpu.stat` 的 usage_usec。两个读取器都必须安静地返回 None，
    否则限流路径会把整个闲时挖掘线程搞崩。
    """
    from lottery import idle_mining as IM

    q = IM._read_cgroup_cpu_quota()
    u = IM._read_cgroup_cpu_usage()
    assert q is None or (isinstance(q, float) and q > 0)
    assert u is None or (isinstance(u, int) and u >= 0)


def test_system_cpu_pct_prefers_container_quota(monkeypatch):
    """当配额与 usage 都可读时，占用率必须按**配额**归一（容器口径）。

    构造：配额 2.0 核，50ms 内 usage 增长 50_000 usec
    （= 0.05 核秒 / 0.05 秒 = 1.0 核）⇒ 相对 2.0 核的占用率 = 50%。
    若错误地按宿主 /proc/stat 算，结果会完全不同。
    """
    from lottery import idle_mining as IM

    seq = iter([1_000_000, 1_050_000])
    monkeypatch.setattr(IM, "_read_cgroup_cpu_quota", lambda: 2.0)
    monkeypatch.setattr(IM, "_read_cgroup_cpu_usage", lambda: next(seq))

    got = IM._system_cpu_pct()
    assert got is not None
    assert got == pytest.approx(50.0, abs=1.0), (
        f"容器口径应为 50%（1 核 / 2 核配额），实际 {got}")


def test_system_cpu_pct_falls_back_without_quota(monkeypatch):
    """配额读不到时不得使用 cgroup usage（那是宿主或未定义语义），应回退。"""
    from lottery import idle_mining as IM

    monkeypatch.setattr(IM, "_read_cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(IM, "_read_cgroup_cpu_usage", lambda: 12345)
    got = IM._system_cpu_pct()
    assert got is None or 0.0 <= got <= 100.0


# ===========================================================================
# 默认值方向的元守卫
# ===========================================================================

def test_no_dangerous_default_in_ev_caliber():
    """元守卫：`sixth` 参数不得再出现 `="published"` 默认。

    这条不测行为，只防"有人把默认值改回去" —— 改回去不会有任何行为测试变红，
    只会在未来某次无关改动里静默产生错误的 EV。
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "lottery" / "engine_distribution.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        names = [a.arg for a in args.args + args.kwonlyargs]
        if "sixth" not in names:
            continue
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for d in defaults:
            if isinstance(d, ast.Constant) and d.value == "published":
                raise AssertionError(
                    f"{node.name} 的 sixth 默认值又变回 'published' —— "
                    "那会让未传参的调用方拿到重复计数口径（P2 回归）")
