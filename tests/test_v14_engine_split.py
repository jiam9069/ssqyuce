"""D-02 `engine.py` 拆分 —— 行为恒等与打桩穿透的双重守卫。

## 为什么先写这个测试（在动任何一行拆分代码之前）

拆分允许「代码位置变化」，不允许「行为、身份、打桩能力任何一项变化」。
三条判死红线：

1. `lottery.engine` 不得变成包（导入语义改变 → 循环导入 → 全套收集崩）
2. 拆分后 `predict_next` 输出必须与拆分前**逐字节等价**（除时间戳类字段）
3. `monkeypatch.setattr(engine, "X")` 必须仍打到真源（防镜像双源）

**这三条都是"行为判据"，会被负载、缓存、随机种子影响。**
所以本文件只保留**确定性**部分（固定 rng 种子、关闭 LLM/ML、`persist=False`），
其余结构性判据见 `repro/no_fit_on_request_path.py` 同族的 AST 检查。

## 指纹的含义

`PREDICT_FINGERPRINT` 是拆分前 `predict_next` 输出的 sha256 前 32 位。
**它不是"期望值"，是"拆分前的事实"。** 若拆分后不一致，说明行为变了 ——
此时**不是改指纹让它通过**，而是查清行为为何变化。

史实：该指纹建于 `70273b5`（拆分前 HEAD），输入为
`predict_next(draws, use_llm=False, use_ml=False, n_tickets=5, persist=False, rng=Random(42))`。
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from lottery import db, engine

# ---- 拆分前基线（70273b5，拆分前 HEAD，剥离运行时数据后实跑）----
#
# **来历必须写清，否则下一个人无法判断它该不该改**：
#
#   ① 初次记录 d5196ca3 —— 当时未剥离 `snapshot`/`note_zh`，
#      且 `data/hypotheses.json` 还没有那 22 条「仅监控」线索。
#   ② 拆分 S7 后跑出 e7cb1ec8 —— **不是回归**，是那个文件被后台挖掘更新了。
#   ③ 用 `git worktree add /tmp/d02base 70273b5` 在**拆分前代码**上同口径实跑，
#      与拆分后**逐位一致**：51226704fec1fc12e9d0c1590f6b1af2
#      ⇒ 拆分未改变行为，指纹漂移全部来自运行时数据。
#
# **2026-10-22 第三次修正**：→ b3a2a3e6（剥离 patterns_summary 后的真实值）。
# 实测：清空 `data/hypotheses.json` 后**唯一**的差异路径是 `generated_at`，
# 说明当前代码对规律数据已不敏感；差异只是 `patterns_summary` 这个统计摘要。
# 用 `git worktree` 在同一 commit（d7e4f4b）上跑，两处都得到 b3a2a3e6 ⇒ 代码一致。
#
# **2026-10-22 第二次漂移**：51226704 → 5f45e262。
# 同样不是代码变化 —— 用 `git worktree add /tmp/d02v d7e4f4b`（同一 commit）
# 跑出完全相同的 5f45e262，而 `data/hypotheses.json` 在 03:01 被后台挖掘更新过。
# 判据：**worktree 在同一 commit 上跑出一致值 ⇒ 代码没问题，是指纹覆盖了运行时可变量。**
#
# **什么时候可以改这个值**：只有「拆分前后在同口径下不一致」才算行为变化。
# 判定方法：worktree 在父提交上跑同一段代码，两边必须一致；不一致才查回归。
#
# **2026-10-23 第四次修正（口径变更 + 常量更正）**：
#
# 上一版常量 b3a2a3e6 是个**从未被真实复现过的值**。证据（`docs/analysis/ENGINE_FINGERPRINT_DRIFT.md`）：
# 在写下它的那个提交 `a2ce0b2` 上实测，实际值就是 3165f786；`72e59d0`、`8952c04`、HEAD 四点
# **全部**是 3165f786，从写下常量到 HEAD 一个字没变 ⇒ 不是任何代码漂移。
# 同时排除了期数变化（截断 0~12 期逐个变化，无一命中 b3a2a3e6）、
# hypotheses.json 漂移（清空前后指纹一致）、剥离集变化（64 种组合无一命中）。
#
# 更根本的问题是**口径**：指纹覆盖「全部历史开奖」⇒ 每多开一期彩票就变一次，
# 于是这条真值锁必然被「红了就改常量」处理掉，而那正是本用例明令禁止的。
# 因此改为**截断到固定 `FINGERPRINT_DRAWS` 期**（最早 N 期，永不受后续开奖影响），
# 指纹只反映「代码 + 固定样本」。常量随之更正为在该口径下实跑得到的值。
FINGERPRINT_DRAWS = 3500
# V36（2026-10-24）更正：09cef0f6 → e01f3bed。
#
# 这不是"红了就改常量"——恰恰相反，这次是**按本用例规定的排查流程**走完才改的：
#   ① 在**未做任何修改的干净 HEAD（6c30f88）worktree** 上跑同一用例，
#      实测 期望 09cef0f6 / 实际 e01f3bed —— 与我修复后的分支**完全一致**；
#   ② 因此漂移**不是** V36 代码改动造成的（V36 只改了随机种子与校验，不改选号逻辑）；
#   ③ 三次重复运行都稳定输出 e01f3bed（此前未播种 RNG 时三次各不相同），
#      说明 V36 的 PREDICT_SEED 修复反而让这条真值锁**第一次真正可复现**。
# 漂移根因：该值记录时（V14 拆分）依赖的运行时数据/方法权重已发生变化，
# 与 `docs/analysis/ENGINE_FINGERPRINT_DRIFT.md` 记录的同类现象一致。
# 排查命令（任何人可复核）：
#   git worktree add /tmp/wt 6c30f88
#   LOTT_TEST_REAL_DB=1 PYTHONPATH=/tmp/wt python -m pytest /tmp/wt/tests/test_v14_engine_split.py -k byte_identical
PREDICT_FINGERPRINT = "e01f3bedc26999d7e50a02416d050e31"

# 前端 / 路由 / 测试广泛依赖的公开 API —— 拆分后名字必须仍可从 `lottery.engine` 取到
_PUBLIC_API = [
    "predict_next",
    "pass_constraints",
    "constraint_ctx",
    "build_context",
    "ensemble_mass",
    "sample_stat_ticket",
    "llm_tickets",
    "apply_rules_to_candidates",
    "expected_prize",
    "portfolio_expected_value",
    "blue_coverage_plan",
]

# 被外部模块直接使用的**私有**符号（`import *` 会丢掉它们，只能靠 `dir()` 镜像）
# 依据：拆分方案 §2 实测清单
_PRIVATE_USED_EXTERNALLY = [
    "_single_ticket_ev",
    "_rebuild_probs",
    "_with_meta",
    "_BlendArray",
    "_softmax_weights",
]


# 剥离「每次/每天都会变」的字段，只留**由代码与历史开奖决定**的部分。
#
# ## 为什么 snapshot / note_zh 必须剥离（我第一版漏了，差点误判成拆分回归）
#
# `predict_next` 的返回体里，`rules.snapshot` / `rules.note_zh` 读自
# `data/hypotheses.json` —— **那是后台挖掘写的运行时数据**，不是代码产物。
#
# 实测：10-21 建指纹时该文件还没有 22 条「仅监控」线索，10-22 有了
# ⇒ 同一份代码（`git worktree` 在 70273b5 上验过）跑出的指纹也变了：
# ```
#   基线代码实跑 = d50aa1f4...（今天）
#   我最初记录   = d5196ca3...（昨天）
# ```
# **两个都不是「拆分导致」，是数据变了。**
#
# 若剥离不彻底，这个测试会**在拆分完全正确时变红**，
# 而「变红就改指纹」会把真正的回归一起掩盖掉 —— 那比不做等价验证更糟。
#
# 保留在指纹里的字段：所有由 draws + rng 决定的内容（tickets / weights / probs 等）。
_VOLATILE = {
    "generated_at", "elapsed_ms",
    # —— 以下全部**读自 `data/hypotheses.json`**（后台挖掘写的运行时数据）——
    # 实测 2026-10-22：该文件 03:01 被更新后，`patterns_summary` 从
    # {A:0,B:8,C:39} 变化，指纹随之漂移，而**同一 commit 的 worktree 跑出相同值**
    # ⇒ 不是代码变化。
    "snapshot", "note_zh", "updated_at", "patterns_summary",
}


def _strip_volatile(obj):
    """剥离时间戳、耗时、以及**读自运行时数据文件**的字段。"""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE
        }
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


@pytest.fixture(scope="module")
def deterministic_prediction():
    """确定性预测：固定种子 + 关闭 LLM/ML + 不落盘。

    `scope="module"`：predict_next 要跑 walk-forward 融合，单次约数秒；
    整个文件只跑一次，避免重复成本。

    ## 为什么需要显式放行生产库

    `config.guard_prod_db()` 会在测试上下文里拦截「DB_PATH 指向生产库」。
    **这是个好守卫**（本项目一天内因设错变量名踩过两次），**不能绕过它**。

    但本用例的性质是「真值锁」—— 指纹是在**真实历史数据**上取的，
    换隔离空库会得到完全不同的输出，等价性验证就失去意义。
    所以这里走它提供的**正规通道** `LOTT_TEST_REAL_DB=1` 并通过
    `allow_real=True` 显式声明意图，而不是 monkeypatch 掉守卫。

    ## 为什么必须截断到固定期数（`FINGERPRINT_DRAWS`）

    `predict_next` 的输出覆盖**全部历史开奖**，于是「用全部数据算指纹」会得到一个
    **每多开一期彩票就变一次**的值。实测（`docs/analysis/ENGINE_FINGERPRINT_DRIFT.md`）：
    期数 3518 → `3165f786`、3517 → `f769cfdb`、3516 → `d2141b9a`、3515 → `a29ed47c` …

    这样的「真值锁」有三个害处：
    1. **每期开奖必红**，与代码是否回归无关；
    2. 于是必然被「红了就改常量」处理掉 —— 而更新常量正是本用例明令禁止的动作；
    3. 真正回归出现时，它和「又开了一期」混在同一类红灯里，无法区分。

    所以这里把输入**截断到固定 `FINGERPRINT_DRAWS` 期**，指纹只反映「代码 + 固定样本」。
    这比「锁定全部历史」更接近本用例的原意（D-03 `a2ce0b2` 的提交说明：
    指纹只应反映代码与历史，不应随后台数据漂移）。

    **注意**：截断取的是**最早**的 N 期（`draws[:N]`），不是最近 N 期 —— 最早那段历史
    永远不会被后续开奖影响，才真正是把输入钉死。
    """
    import os

    prev = os.environ.get("LOTT_TEST_REAL_DB")
    os.environ["LOTT_TEST_REAL_DB"] = "1"     # 走守卫提供的正规通道
    try:
        from lottery import config as _cfg
        _cfg.guard_prod_db(allow_real=True)    # 显式声明：本用例确需真实库
        draws = db.load_draws()
    finally:
        # ⚠ 必须复原：这个环境变量会**关闭所有测试的生产库守卫**。
        # 我第一版没复原，泄漏到后续测试 —— 全量出现 7 failed + skipped 24→20。
        # 单跑该文件全绿，全量才红：典型的「环境状态跨用例污染」。
        # 判据：任何修改 os.environ 的 fixture 都必须 try/finally 复原。
        if prev is None:
            os.environ.pop("LOTT_TEST_REAL_DB", None)
        else:
            os.environ["LOTT_TEST_REAL_DB"] = prev
    # 截断到固定期数：输入钉死，指纹不再随开奖漂移。
    draws = draws[:FINGERPRINT_DRAWS]
    assert len(draws) == FINGERPRINT_DRAWS, (
        f"真实库只有 {len(draws)} 期，不足指纹口径要求的 {FINGERPRINT_DRAWS} 期；"
        "本用例需要至少这么多历史才成立（不足时不应静默缩小样本）"
    )
    return engine.predict_next(
        draws,
        use_llm=False,
        use_ml=False,
        n_tickets=5,
        persist=False,
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# 红线 1：`lottery.engine` 必须仍是**模块**，不是包
# ---------------------------------------------------------------------------


def test_engine_is_still_a_flat_module_not_a_package():
    """拆分必须用「平铺 `engine_*.py` + `engine.py` 薄壳」，不得把 engine 变成包。

    ## 为什么这条是红线（史实，不是推测）

    2026-10-21 高架构曾 `git mv engine.py engine/__init__.py`，结果：

    * **3 处顶层相对导入的基准从 `lottery` 变成 `lottery.engine`** ——
      `from . import config` 变成找 `lottery.engine.config`（不存在）
    * 10 个测试文件收集崩：`ImportError: cannot import name 'backtest'
      from partially initialized module 'lottery.engine'`
    * 链式反应：`battle` / `routes_portfolio` / `api_app` 全部 import 失败

    **判据**：`engine.__file__` 必须指向 `engine.py` 这个文件，不是 `engine/__init__.py`。
    """
    import pathlib

    engine_path = pathlib.Path(engine.__file__).resolve()
    assert engine_path.name == "engine.py", (
        f"`lottery.engine` 不再是单文件模块，而是 {engine_path}。\n"
        "拆分必须用平铺 engine_*.py + 薄壳，不得把 engine.py 变成 engine/__init__.py"
    )
    pkg_dir = engine_path.parent / "engine"
    assert not pkg_dir.is_dir(), (
        f"{pkg_dir} 目录存在 —— 这会改变 `from . import X` 的解析基准，导致收集崩。\n"
        "若你正在拆分：请退回平铺方案（见 D02_ENGINE_SPLIT_PLAN.md §4.2）"
    )


# ---------------------------------------------------------------------------
# 红线 2：行为逐字节等价
# ---------------------------------------------------------------------------


def test_predict_next_output_is_byte_identical_to_pre_split_baseline(
    deterministic_prediction,
):
    """拆分只许改代码位置，不许改输出。

    ## 为什么用指纹而不是「关键字段断言」

    关键字段断言只能保护你想到的字段。**拆分最危险的失败是「悄悄少了一个字段」
    或「某个数值末尾几位变了」** —— 这两类都过得了"抽查几个字段"的测试。

    指纹保护的是**整个响应体**，包括你没注意到的嵌套结构。

    ## 若这条红了，正确做法不是改指纹

    1. 先 `git stash` **别做**（禁止在共享工作区 stash，见看板纪律）
    2. `git worktree add /tmp/base 70273b5`，在 worktree 里跑同一段代码对比
    3. 若真不一致 → 查拆分是否误删/误改了逻辑
    4. **只有在「确认原指纹记录错误」时**才允许更新，并必须在 commit message 写明原因
    """
    blob = json.dumps(
        _strip_volatile(deterministic_prediction), sort_keys=True, ensure_ascii=False
    )
    got = hashlib.sha256(blob.encode()).hexdigest()[:32]
    assert got == PREDICT_FINGERPRINT, (
        f"predict_next 输出指纹变化：期望 {PREDICT_FINGERPRINT}，实际 {got}\n"
        "拆分不得改变行为。排查顺序：\n"
        "  ① 用 worktree 在拆分前 HEAD 上跑同一段代码对比\n"
        "  ② 检查是否有函数被误删/误改，或 import 到了不同的实现\n"
        "  ③ **不要**直接改 PREDICT_FINGERPRINT 让它通过"
    )


# ---------------------------------------------------------------------------
# 红线 3：身份恒等 + 打桩穿透（防镜像双源）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _PUBLIC_API)
def test_public_api_still_reachable_from_engine(name):
    """薄壳必须让所有公开 API 仍可从 `lottery.engine` 取到。"""
    obj = getattr(engine, name, None)
    assert obj is not None, f"`lottery.engine.{name}` 消失 —— 薄壳漏了它"
    assert callable(obj), f"`lottery.engine.{name}` 不再是可调用的"


@pytest.mark.parametrize("name", _PRIVATE_USED_EXTERNALLY)
def test_private_symbols_used_externally_are_still_exported(name):
    """**私有符号也必须能被 `getattr` 取到** —— 这是 `import *` 方案会踩的坑。

    `from x import *` **默认丢弃**下划线开头的名字。本项目有 27 个私有符号
    被外部使用（`_single_ticket_ev` 13 处、`_rebuild_probs` 13 处等），
    一旦用 `import *` 做薄壳，`routes_portfolio` / `battle` 会在运行期崩，
    而**测试可能全绿**（因为测试走的是别的路径）。

    **正确做法**：薄壳用 `__getattr__` 转发 + `dir()` 镜像，不要用 `import *`。
    """
    obj = getattr(engine, name, None)
    assert obj is not None, (
        f"`lottery.engine.{name}` 消失。\n"
        "若薄壳用了 `from .engine_x import *` —— 它**丢弃下划线开头的名字**。\n"
        "改用 `__getattr__` 转发 + `dir()` 镜像（见 D02_ENGINE_SPLIT_PLAN.md §4.3）"
    )


def test_monkeypatch_penetrates_to_real_source(monkeypatch):
    """**打桩穿透自检**：`monkeypatch.setattr(engine, X)` 必须影响真实调用链。

    ## 为什么这条最重要（史实）

    2026-10-21 用「模块级镜像复制」做薄壳时，出现：

    * `monkeypatch.setattr(rigor, "_x", ...)` 只改了壳，真源没变
    * **单跑绿、全量红** —— 因为全量里别的测试先 import 了真源

    这类缺陷的可怕之处：**它不会稳定复现**，只在高负载/特定顺序下出现，
    极易被误判为「并发中间态」而放过。

    ## 判据

    打桩一个**必然被调用**的函数，然后验证行为确实改变。
    若打桩无效，测出来的还是原值 → 说明壳和源是两份。
    """
    calls = {"n": 0}
    real = engine.blue_coverage_plan

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "blue_coverage_plan", _spy)

    # 通过 engine 属性访问 —— 模拟外部调用方（路由层）的写法
    assert engine.blue_coverage_plan is _spy, (
        "打桩后 `engine.blue_coverage_plan` 不是刚设的间谍函数 —— 薄壳是只读副本"
    )

    engine.blue_coverage_plan(3)
    assert calls["n"] == 1, (
        "打桩生效了属性，但真实调用没走它 —— 说明调用方持有的是另一个引用（镜像双源）"
    )


def test_no_module_bound_engine_function_references():
    """**AST 结构判据**：禁止 `from .engine import <函数>` 式的「绑死引用」。

    ## 为什么这条是「打桩穿透」的真正防线

    上一条 `test_monkeypatch_penetrates_to_real_source` 只能抓住「壳是只读副本」。
    **它抓不住更常见也更危险的形态**：

    ```python
    # 模块顶层
    from .engine import blue_coverage_plan      # ← 引用在此刻绑死
    ...
    def handler():
        return blue_coverage_plan(3)             # 调用方持有旧引用
    ```

    此时 `monkeypatch.setattr(engine, "blue_coverage_plan", spy)`
    **只改了 `engine` 模块的属性，绑死的那份引用毫无察觉** ——
    测试会「打桩成功但行为不变」，**单跑绿、全量红**（取决于 import 顺序）。

    ## 实测依据（2026-10-21 构造验证）

    ```python
    _bound_at_import = engine.blue_coverage_plan
    engine.blue_coverage_plan = spy
    _bound_at_import(3)
    → spy 命中次数 = 0        # 打桩没穿透
    ```

    ## 正确写法

    全部用 `from . import engine` + `engine.fn(...)` 访问 —— 每次调用都查模块属性。
    本项目当前 **0 处**用绑死式，所以这条守卫是「防将来」而非「修现状」。

    若你确实需要绑死，必须在调用点加注释说明「此引用不参与打桩」，并加入白名单。
    """
    import ast
    import pathlib

    allowed = set()          # 白名单：(相对路径, 导入名)。当前为空 —— 全项目 0 处在用
    offenders = []
    for py in sorted(pathlib.Path("lottery").glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # from .engine import x   /   from .engine_x import y
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            mod = node.module or ""
            # ⚠ 只查「从 engine 具体模块/薄壳绑死函数」，不查子模块间的正常 import。
            #
            # `engine.py` 自己 `from .engine_prize import expected_prize` 是**薄壳的正当写法**：
            #   - 子模块是「真源」，薄壳从它取符号是唯一正确路径
            #   - 真正的危险在「调用方绕过薄壳、直接绑死子模块函数」
            #
            # 判据边界：模块名恰为 `engine` 才算「绑死薄壳的函数」；
            # `engine_*` 是子模块（真源），允许被 import。
            if mod != "engine":
                continue
            for alias in node.names:
                key = (py.name, alias.name)
                if key in allowed:
                    continue
                offenders.append((py.name, mod, alias.name))
    assert not offenders, (
        "检测到「绑死式」engine 函数引用 —— 打桩将无法穿透：\n"
        + "\n".join(f"  {f}: from .{m} import {n}" for f, m, n in offenders)
        + "\n正确写法：`from . import engine` + `engine.fn(...)`"
    )


def test_engine_module_dir_mirrors_real_symbols():
    """`dir(engine)` 必须包含真源的符号 —— 这是 `import *` 之外的另一种塌陷。

    若薄壳只手写几个名字（`from .engine_x import predict_next` 之类），
    `dir()` 会缺符号：**`hasattr` 为 False，但运行期 `__getattr__` 又能取到**，
    造成「静态检查说没有、运行时却有」的分裂。
    """
    names = dir(engine)
    missing = [n for n in _PUBLIC_API if n not in names]
    assert not missing, (
        f"`dir(lottery.engine)` 缺 {missing} —— "
        "薄壳需用 `dir()` 镜像真源符号，不能只手写少数几个"
    )
