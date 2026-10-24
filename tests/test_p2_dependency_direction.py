"""V13 P2：模块化第一步的守护测试（依赖方向 + 打桩点一致性）。

## 这一步做了什么

把 `_current_user_id` contextvar 从 `api_app.py` 搬到 `lottery/request_context.py`，
并把 `db.py` / `llm_client.py` / `user_llm_api.py` 的反向依赖改掉：

改造前（倒置）：
```
db.py          ──> api_app.py
llm_client.py  ──> api_app.py
user_llm_api   ──> api_app.py
```
改造后（单向）：
```
api_app.py ──> request_context.py <── db.py
                                  <── llm_client.py
api_app.py ──> llm_config_store.py <── user_llm_api.py
```

## 为什么这个方法值得测

`api_app.py` 是 3500 行的 Web 应用模块。底层库为了读一个 contextvar 而依赖它，
后果是：无法独立测试、循环依赖只能靠"把 import 写进函数体"来躲（M1 §1.2
统计有 70+ 处这种懒加载）、切分 api_app 时处处受阻。

**没有测试守着的"重构"会悄悄回滚**：下次有人图省事再写一句
`from . import api_app`，一切照常运行，没人会发现分层又倒了。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

LOTTERY = Path(__file__).resolve().parent.parent / "lottery"


def _imports_of(path: Path):
    """返回该文件里所有 `from . import X` / `from lottery import X` 的模块名集合。

    用 AST 而不是 grep：grep 会把注释与字符串里的同名字样算进去
    （本项目已多次因此误判）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module is None:
                for a in node.names:
                    names.add(a.name)
            elif node.level == 1 and node.module:
                names.add(node.module)
    return names


def test_no_low_level_module_imports_api_app():
    """**核心**：底层库不得反向依赖 Web 应用模块 `api_app`。

    这不只是洁癖 —— 它决定了 `db`/`llm_client` 能否被单独 import 与单独测试。
    """
    # 允许出现的例外（都是有理由的、且不在底层）
    allowed = {"auth_api", "user_llm_api"}
    offenders = []
    for p in sorted(LOTTERY.glob("*.py")):
        if p.name in ("api_app.py",) or p.name in allowed:
            continue
        if "api_app" in _imports_of(p):
            offenders.append(p.name)
    assert not offenders, (
        f"这些模块反向 import 了 api_app：{offenders}\n"
        "底层库请改用 lottery.request_context（纯状态）或 "
        "lottery.llm_config_store（配置读写），不要依赖 Web 应用模块。")


def test_request_context_is_pure_state():
    """`request_context` 必须是纯状态模块：不 import 任何业务模块。

    一旦它开始 import db/config 之类，就会重新变成一个新的循环依赖中心。
    """
    names = _imports_of(LOTTERY / "request_context.py")
    banned = {"db", "config", "api_app", "llm_client", "engine"}
    assert not (names & banned), (
        f"request_context 不应依赖业务模块，实际依赖：{names & banned}")


def test_llm_config_store_does_not_import_api_app():
    names = _imports_of(LOTTERY / "llm_config_store.py")
    assert "api_app" not in names


def test_api_app_reexports_contextvar_for_compat():
    """`api_app._current_user_id` 等名字必须仍可用。

    既有测试与中间件都按这些名字访问；搬迁不该改契约。
    """
    from lottery import api_app, request_context
    assert api_app._current_user_id is request_context._current_user_id
    assert api_app.current_user_id is request_context.current_user_id


def _goes_through_low_level(window: str) -> bool:
    """该测试片段是否经由底层库（它们直接 import request_context）。"""
    markers = ("llm_client", "db.", "engine", "_run_chat", "chat_json",
               "save_predictions", "record_usage", "resolve_tenant")
    return any(mk in window for mk in markers)


def test_stub_points_match_real_dependency_sources():
    """**打桩点一致性**：测试里 monkeypatch 的名字必须是真正的依赖源。

    P2 搬迁时踩到的真问题：测试打桩 `api_app.current_user_id`，而搬迁后
    `llm_client` 直接调 `request_context.current_user_id` —— 打桩**静默失效**，
    用例照样通过，但它测的已经不是生产路径了。

    这类"打桩点与依赖源脱节"是本项目假绿的老来源（W06 的路由枚举、
    路由分级测试都是同一类）。这里用静态扫描把它钉死。
    """
    tests_dir = Path(__file__).resolve().parent
    offenders = []
    for p in sorted(tests_dir.glob("test_*.py")):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(
                r'monkeypatch\.setattr\(\s*api_app\s*,\s*"current_user_id"', text):
            # 同一次 setattr 附近 400 字符内必须同时打到 request_context
            window = text[m.start(): m.start() + 400]
            # 判断"这条用例测的是哪条路径"：
            #  * 直接调 `api_app.<某函数>`（不经过底层库）→ 只打 api_app 是对的；
            #  * 走 llm_client / db / engine（它们 import request_context）→ 必须两边都打。
            # 用"同一函数体内是否出现底层库调用"来区分。
            if "request_context" not in window and _goes_through_low_level(window):
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{p.name}:{line}")
    assert not offenders, (
        f"这些地方只打了 api_app.current_user_id，没有同时打 request_context：{offenders}\n"
        "两条路径都要打桩，否则会被静默短路。")


def test_llm_config_endpoints_delegate_to_store():
    """`/api/config/llm` 的实现应在 `llm_config_store`，路由层只留薄壳。

    P2 第二步后，端点本体已搬进 `routes_llm.py`，所以这里检查的是路由模块。
    """
    candidates = [LOTTERY / "routes_llm.py", LOTTERY / "api_app.py"]
    holder = None
    for p in candidates:
        if p.exists() and "def update_llm_config(" in p.read_text(encoding="utf-8"):
            holder = p
            break
    assert holder is not None, "找不到 update_llm_config 的定义处"
    text = holder.read_text(encoding="utf-8")
    seg = text[text.index("def update_llm_config("):]
    end = seg.find("\ndef ")
    seg = seg[:end] if end > 0 else seg
    assert "llm_config_store.save" in seg, (
        f"update_llm_config 在 {holder.name} 里没有委托给 llm_config_store.save")


def test_route_modules_have_no_undefined_globals():
    """**静态检查**：域模块里用到的全局名必须都有定义（import 或本模块内定义）。

    这是 P2 切分时踩到的真问题：用脚本把端点块从 api_app 搬到域模块，
    **块里用到的 `JSONResponse` 没有跟着搬**（它原本在 api_app 的顶层 import 里）。
    3 个模块同时缺，只有 `test_m4_2_api::test_methods_config_validation` 抓到，
    因为另外两条路径的用例没触发那个分支。

    运行期才炸的 NameError 属于"分文件时最容易漏、最难预判"的一类，值得用
    静态分析兜住：宁可让这条测试告诉我要补哪个 import，也不要等线上 500。
    """
    import builtins
    missing = {}
    for p in sorted(LOTTERY.glob("routes_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        defined = set(dir(builtins))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    defined.add(a.asname or a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    defined.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)      # `except Exception as e:`
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        bad = sorted(used - defined)
        if bad:
            missing[p.name] = bad
    assert not missing, (
        "这些域模块用到了未定义的名字（切分时漏搬 import）：\n" +
        "\n".join(f"  {k}: {v}" for k, v in missing.items()))


def test_route_modules_do_not_import_api_app():
    """域路由模块不得 import api_app（否则切分等于白做）。"""
    offenders = []
    for p in sorted(LOTTERY.glob("routes_*.py")):
        if "api_app" in _imports_of(p):
            offenders.append(p.name)
    assert not offenders, f"路由模块反向 import 了 api_app：{offenders}"


def test_api_app_keeps_only_assembly_responsibilities():
    """`api_app.py` 的角色应收缩为「组装」：顶层 `@app.` 端点数量持续下降。

    这不是硬性上限，而是一条**防回滚**哨兵：P2 的目标是把端点下沉到域模块，
    如果有人习惯性地往 api_app 里加新端点，数量会涨上去并触发提醒。
    需要调高阈值时，请先确认新端点确实无法归入任何现有域。
    """
    text = (LOTTERY / "api_app.py").read_text(encoding="utf-8")
    n = text.count("\n@app.")
    assert n <= 71, (
        f"api_app.py 里仍有 {n} 个端点（阈值 71）。P2 的目标是把它们下沉到 "
        f"routes_*.py；新增端点请放进对应域模块。")


# ---------------------------------------------------------------------------
# P1-11（V36，2026-10-24 审计）：守卫只守了一条，补上通用分层
# ---------------------------------------------------------------------------
#
# 审计原文：「`test_p2_dependency_direction.py` 只禁『底层 import api_app』，
# 没有守住跨层的其他方向 —— 例如 `evaluate.py → engine`，会让『依赖方向已被
# 守住』变成错觉。」
#
# 这个判断成立：上面唯一的核心用例确实是"单点禁令"。下面按**层号**给出通用规则，
# 让"某个模块 import 了它不该 import 的东西"变成一条可枚举的失败，
# 而不是靠人看见才想起。

#: 分层编号（数字越小越底层，只能被上层依赖，不能反向依赖上层）。
#: 分组依据是**职责**而不是文件名字母序：
LAYERS = {
    # L0 纯状态 / 纯常量 / 纯数据结构（不依赖任何其它 lottery 模块）
    0: {"request_context", "jsonutil", "engine_base"},
    # L1 基础设施：配置、DB、加密、统计原语、**外部 I/O 客户端**。
    #
    # `llm_client` / `methods` 放在 L1 而不是"服务层"，理由是实测出来的：
    # 它们被 engine/engine_* / hypothesis / decision_skill / ai_audit / llm_lab /
    # data_audit / replay 等**十余个领域模块**引用。
    # 把它们标成上层会让分层检查对着十几个"违规"报警 —— 那些报警描述的不是
    # 设计问题，而是我的分组没反映真实依赖结构。分层的价值在于指出**倒置**，
    # 不在于裁决"HTTP 客户端算不算基础设施"。它们是叶子（只依赖 config），
    # 所以归 L1 既准确又让检查聚焦在真正的环上。
    #
    # `methods` 同理：它是**通道开关的纯函数表**（resolve_channels 等），
    # 不发起 I/O、不碰 DB，属于"配置的延伸"。
    1: {"config", "db", "secretbox", "migrations", "parser",
        "stats", "url_guard", "concurrency", "llm_client", "methods",
        "llm_config_store",
        # features 只依赖 config 的窗口常量，是"配置的派生"，不是领域结论。
        # models 也依赖它 → 二者与 config 同层最诚实。
        "features", "models"},
    # L2 领域算法：特征、规律、引擎、回测、评估
    2: {"features", "patterns", "regime", "mining", "engine", "engine_prize",
        "engine_distribution", "engine_blue", "engine_blend", "engine_rules",
        "engine_solver", "engine_constraints", "engine_orchestrate",
        "evaluate", "backtest", "battle", "hypothesis", "evolution",
        "ml_model", "multiscale", "popularity", "blue_coverage",
        "candidate_space", "variance_picker", "discovery", "rule_lifecycle",
        "shape_rules", "decision_skill", "selfcheck", "walk_forward_gate",
        "spec_replay", "replay", "ai_audit", "ai_hypothesis", "llm_lab",
        "llm_eval", "data_audit", "data_check", "data_fetcher",
        "patterns_runtime", "prereg_scheduler", "eprocess", "diagnose",
        "rigor", "rule_protocol"},
    # L3 服务层：任务调度、通知、租户存储（**会发起 I/O 或持有线程**的才在这层）
    3: {"user_llm_store", "bg_tasks", "notify", "idle_mining",
        "evolution_service", "l2_service"},
    # L4 Web 装配：路由与 app（它们可以依赖下面所有层）
    4: {"api_app", "auth", "auth_api", "routes_audit", "routes_capabilities",
        "routes_data", "routes_diagnosis", "routes_eval", "routes_llm",
        "routes_llm_eval", "routes_llm_ops", "routes_methods", "routes_ml",
        "routes_patterns", "routes_portfolio", "routes_predict", "routes_rigor",
        "routes_rules", "routes_tasks", "cli", "user_llm_api"},
}

#: 已知且**有明确理由**的上行依赖（白名单）。
#: 加条目必须写明理由 —— 否则它就变成"永久豁免"的垃圾抽屉。
KNOWN_UPWARD = {
    # evaluate 调 engine.predict_next 做 walk-forward 评估。这是**循环**：
    # engine 也需要 evaluate 的 prize_level 结算。历史上已存在，
    # 拆分重构（V14 engine_split）没有解决它。
    # 记为已知债务：改动时至少要意识到"评估与引擎互相依赖"。
    ("evaluate", "engine"),
    # battle 建立在 evaluate 之上（回放判据）同时又要驱动 engine
    ("battle", "engine"),
    # db → evaluate：`save_eval_details` 需要按奖级结算票（`evaluate.prize_level`
    # 与 PRIZE_CASH）。严格说这是倒置：持久化层不该懂"怎么算奖"。
    # 正确修法是把 prize_level 下沉到 L0/L1（它是纯组合数学），但那会牵动
    # evaluate / battle / engine 三处导入，属独立重构，不在本次"修安全 P1"范围。
    # 登记为债务：新增 db 功能时不要再往 evaluate 里加东西。
    ("db", "evaluate"),
    # llm_client → user_llm_store：记录用量（quota 记账）需要知道"当前用户"。
    # 同样是倒置（客户端不该认识租户存储），修法是把记账下沉/回调注入。
    # 登记为债务：这是"记账必须发生在调用点"造成的，改动需同时动配额语义。
    ("llm_client", "user_llm_store"),
}


def _layer_of(mod: str):
    for lv, members in LAYERS.items():
        if mod in members:
            return lv
    return None


def test_no_unknown_upward_dependencies():
    """逐模块检查：不得依赖**更高层**的模块（白名单除外）。

    这条替代了原来"只禁 api_app"的单点禁令，能抓出任何新增的跨层倒置。
    副作用（有意为之）：往 L0/L1 里加 import 会立刻变红，逼人想清楚
    "这个依赖该不该存在"，而不是顺手 import。
    """
    offenders = []
    for p in sorted(LOTTERY.glob("*.py")):
        mod = p.stem
        if mod == "__init__":
            continue
        my = _layer_of(mod)
        if my is None:
            continue
        for dep in _imports_of(p):
            if dep == mod:
                continue
            dl = _layer_of(dep)
            if dl is None or dl <= my:
                continue
            if (mod, dep) in KNOWN_UPWARD:
                continue
            offenders.append(f"{mod}(L{my}) → {dep}(L{dl})")
    assert not offenders, (
        "发现未登记的上行依赖（低层 import 了更高层）：\n  "
        + "\n  ".join(offenders)
        + "\n\n两条路：① 改成本层内可解决的问题；② 若确实必须，"
          "加入 KNOWN_UPWARD 并写明理由（那是债务登记，不是豁免）。")


def test_layers_table_covers_all_modules():
    """分层表必须覆盖所有模块 —— 否则新模块会静默逃过上面的检查。"""
    actual = {p.stem for p in LOTTERY.glob("*.py")} - {"__init__"}
    known = {m for members in LAYERS.values() for m in members}
    missing = sorted(actual - known)
    assert not missing, (
        f"这些模块没有归层，因此逃过了依赖方向检查：{missing}\n"
        "请把它们放进 LAYERS 的合适层级。")


def test_known_upward_entries_are_real():
    """白名单里不许留"已经不存在"的条目 —— 过期豁免会让检查形同虚设。"""
    stale = []
    for (a, b) in KNOWN_UPWARD:
        pa = LOTTERY / f"{a}.py"
        if not pa.exists() or b not in _imports_of(pa):
            stale.append(f"{a} → {b}")
    assert not stale, (
        f"KNOWN_UPWARD 里这些条目已经不再成立，请删掉：{stale}")
