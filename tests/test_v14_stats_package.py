"""V14-A：`lottery/stats/` 包的结构守卫。

## 为什么需要这组测试

判断层的数学内核（`eprocess` / `rigor`）已迁到 `lottery/stats/`，
`lottery/eprocess.py` 与 `lottery/rigor.py` 变成只做再导出的兼容薄壳。

薄壳这种结构有一个**特有的失效模式**：它看起来还在，但内容悄悄漂移了 ——
比如有人往薄壳里加业务逻辑、或者薄壳漏导出某个符号导致调用方 `AttributeError`。
后者在本次迁移中**真实发生过**：第一版薄壳用 `import *`，
漏掉下划线私有符号（`rigor._valid_draws`），一次性造成 16 个用例失败。

本文件把「薄壳必须与真源完全等价」和「数学包必须保持纯净」写成断言。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
LOTTERY = REPO / "lottery"


# ---------- 1. 薄壳与真源完全等价 ----------

@pytest.mark.parametrize("mod,src_mod", [
    ("eprocess", "lottery.stats.eprocess"),
    ("rigor", "lottery.stats.rigor"),
])
def test_thin_shell_re_exports_same_objects(mod, src_mod):
    """薄壳导出的每个符号必须与真源是**同一个对象**（不是复制/重新实现）。"""
    import importlib

    shell = importlib.import_module(f"lottery.{mod}")
    real = importlib.import_module(src_mod)

    # 真源里所有非 dunder 符号都必须能在薄壳里取到，且 is 相同
    checked = 0
    for name in dir(real):
        if name.startswith("__"):
            continue
        assert hasattr(shell, name), (
            f"薄壳 lottery/{mod}.py 缺少符号 {name!r} —— "
            f"调用方会 AttributeError（本次迁移曾因 import* 漏私有符号失败 16 例）"
        )
        assert getattr(shell, name) is getattr(real, name), (
            f"lottery/{mod}.py.{name} 与 {src_mod}.{name} 不是同一对象 —— "
            f"薄壳里可能被重新实现了一份"
        )
        checked += 1
    assert checked > 5, f"{src_mod} 符号过少（{checked}），检查可能失效"


def test_thin_shell_carries_private_symbols():
    """**回归防护**：下划线私有符号也必须被镜像。

    第一版薄壳用 `from .stats.rigor import *`，而 `import *` 不导出私有名，
    导致 `selfcheck.py` 调用 `rigor._valid_draws` 时直接 AttributeError。
    """
    from lottery import rigor

    assert hasattr(rigor, "_valid_draws"), (
        "薄壳漏导出私有符号 _valid_draws —— "
        "这正是 import* 方案的缺陷（selfcheck 依赖它）"
    )
    assert callable(rigor._valid_draws)


def test_thin_shell_has_no_logic_of_its_own():
    """薄壳不得包含**业务逻辑** —— 只允许转发机制。

    判据：不允许定义任何**函数**；只允许一个代理类 `_ProxyModule`
    （它承载转发机制本身，不是业务逻辑）。

    ## 为什么允许这一个类

    早期版本用「镜像复制」实现薄壳，结果真源与薄壳产生**两份状态**：
    `monkeypatch.setattr(rigor, "X", fake)` 只改薄壳、不改真源，
    打桩期间同名字符串在两个模块里指向不同对象，全量跑测试才复现（依赖顺序）。
    改用模块级代理（`__getattr__` 读真源、`__setattr__` 写双边）后该问题消失，
    代价是薄壳里必须有一个 `_ProxyModule` 类。

    因此断言收紧为：**不得定义函数**、**不得定义除 `_ProxyModule` 外的类**。
    若将来有人往薄壳里塞业务函数，仍会被抓到。
    """
    for mod in ("eprocess", "rigor"):
        src = (LOTTERY / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        funcs = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert not funcs, (
            f"lottery/{mod}.py 是兼容薄壳，不应定义函数 {funcs} —— "
            f"逻辑属于 lottery/stats/{mod}.py"
        )
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        assert set(classes) <= {"_ProxyModule"}, (
            f"lottery/{mod}.py 只允许代理类 _ProxyModule，实际定义了 {classes}"
        )


# ---------- 2. 数学包必须纯净 ----------

@pytest.mark.parametrize("name", ["eprocess", "rigor"])
def test_stats_module_imports_only_stdlib(name):
    """`lottery/stats/` 下的实现**只允许标准库**（外加 numpy/scipy）。

    这是抽包的意义所在：能让「这套数学靠不靠谱」在不拉起 db/config/engine
    的情况下被单独验证。
    """
    src = (LOTTERY / "stats" / f"{name}.py").read_text(encoding="utf-8")
    allowed = {"__future__", "itertools", "math", "typing", "functools",
               "collections", "numpy", "scipy"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                assert top in allowed, f"stats/{name}.py 引入了 {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            assert top in allowed, f"stats/{name}.py 引入了 {node.module}"


@pytest.mark.parametrize("name", ["eprocess", "rigor"])
def test_stats_module_has_no_side_effects(name):
    """不得有项目内相对导入、环境变量、文件 IO、网络。

    **V40 改造**：原来是 `assert bad not in src` 的子串扫荡 ——
    在模块 docstring 里写「本模块不使用 `os.environ`」会让守卫**假红**，
    而把真实代码 `import os` 删掉、只留注释里的字样，又会让它**假绿**。
    现改为 AST 级：分别检查 import 语句、属性访问、函数调用。
    """
    path = LOTTERY / "stats" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        # 1) 项目内相对导入：`from . import x` / `from .. import x`
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            raise AssertionError(f"stats/{name}.py 含被禁止的相对导入（level={node.level}）")
        # 2) 被禁的顶层 import
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in {"os", "requests", "urllib",
                                                    "socket", "sqlite3"}, \
                    f"stats/{name}.py 含被禁止的 import: {a.name!r}"
        # 3) 裸 `open(...)` 调用
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            raise AssertionError(f"stats/{name}.py 含被禁止的 open() 调用")
        # 4) `os.environ` / `os.getenv` 之类的属性访问
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            raise AssertionError(f"stats/{name}.py 访问了环境变量 .{node.attr}")


# ---------- 3. 边界：业务模块不得反向被数学模块依赖 ----------

def test_stats_package_does_not_import_business_modules():
    """`lottery/stats/` 不得 import 任何业务模块（db/config/engine/...）。

    否则「数学层」就被业务反向绑定了，抽包等于白做。
    """
    business = {"db", "config", "engine", "discovery", "hypothesis",
                "evolution", "idle_mining", "api_app", "llm_client"}
    for f in (LOTTERY / "stats").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
                top = node.module.split(".")[0]
                assert top not in business, f"{f.name} 依赖了业务模块 {top}"


def test_idle_mining_uses_stats_package_directly():
    """判定调度层应直接引用数学包，让依赖方向在 import 上可见。

    允许 `from . import eprocess`（走薄壳）与 `from .stats import eprocess`
    两种写法，但**至少有一处**直接指向 stats —— 否则抽包只是搬了个文件。

    **V40 改造**：原为 `"stats" in src or "eprocess" in src` 的全文子串匹配 ——
    注释里提一句 `stats` 就能通过（假绿），而实际上**完全没有 import**。
    现改为 AST：只认真实的 `Import` / `ImportFrom` 节点。
    """
    tree = ast.parse((LOTTERY / "idle_mining.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 and node.module:
                imported.add(node.module.split(".")[0])
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
    assert {"stats", "eprocess"} & imported, (
        f"idle_mining 未 import 数学包（实际导入集里没有 stats/eprocess）：{sorted(imported)}"
    )
