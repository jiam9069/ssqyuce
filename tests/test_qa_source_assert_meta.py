"""元守卫：源码结构断言不得回潮（D-18）。

## 这个文件守的不是功能，是「测试写法的趋势」

## 背景

一批测试用「源码文本里是否出现某个字符串」来验证行为：

    src = inspect.getsource(fn)
    assert "cover_bonus" not in src        # ← 会被 docstring 骗

    blob = (WEB/"app.js").read_text()
    assert "某按钮" in blob                 # ← 会被 // 注释骗

两个方向都会错：
* **假绿**：删掉真实实现、只在注释/docstring 里留着该词 ⇒ 断言仍通过
* **假红**：在注释里写「本函数不使用 X」⇒ 断言无辜变红，**逼人改注释而不是改代码**

2026-10-21 一天内，这两种错在 4 处被独立发现（`in src` 被 docstring 骗、
`_js_code()` 的注释、`:393` 的 docstring 误报、`rule_evidence` 的反向误报）。

## 本文件的两条元守卫

### A 类归零（硬约束）

`inspect.getsource` 得到的 **Python 源码文本**做子串匹配 ⇒ **必须为 0 处**。

为什么可以是 0：A 类共 10 处（按比较数），已全部改成
`_source_assert.code_text()` / `has_call()` / `has_compare()`（D-18a，`fa16a02`）。

**A 类一旦回潮，说明有人又写了那种会被自己的注释骗的断言。**

### B 类不增长（软约束）

`read_text()` 得到的**外部文件文本**（JS/HTML/CSS）做子串匹配 ⇒ 当前 **115 处**，不得增加。

为什么不能归零：这类断言检查的是**前端文案/结构**，
没有等价的 AST（JS 没有 Python 那种标准 AST 入口），
只能「剥注释后子串匹配」。**成本高于收益，故只在新增时告警。**

**注意基线数字的口径**：按**比较数**计（一行里两个 `in` 算两处）。
若按「断言行」计会小一些 —— **报数必须写明单位**（本项目已因口径不清吵过三次）。

## 怎么改（遇到本守卫报红时）

| 情况 | 做法 |
|---|---|
| 你要加 A 类断言 | 改用 `_source_assert.code_text(fn)` 或 `has_call` / `has_compare` |
| 你要加 B 类断言 | 用剥离注释后的文本（见 `_source_assert` 的思路），并**同步下调本文件基线** |
| 你在删代码导致 B 类减少 | 同步下调 `B_BASELINE` —— 下调基线本身就是交付证据 |

**下调基线是有意义的动作**，不要嫌麻烦跳过去：它让「进度」可度量。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent

#: A 类（Python 源码 getsource 子串断言）的硬上限。
#: D-18a 已把 10 处全部 AST 化，因此必须是 0。
A_BASELINE = 0

#: B 类（外部文件 read_text 子串断言）的基线，按**比较数**计。
#: 实测 2026-10-21：115 处 / 15 个文件。
#: 改一处就下调一处 —— 基线只许降，不许升。
B_BASELINE = 145    # 2026-10-24 V34：-1（移除三张主页卡后，相关前端断言合并/删除）
#
# 为什么这一处必要、且无法合并掉：
#   V23 把「前向回测」搬到首屏（L3）。要守的是「卡片真的存在于 index.html」——
#   这是**结构性事实**，没有等价的 AST 可查（index.html 不是 Python 代码，
#   也不是 JS，`_source_assert` 只覆盖 py/js 两种剥注释路径）。
#   已把它与 JS 的若干子串合并成**同一条断言**（`test_frontend_walk_forward_card_and_rendering`），
#   所以这里只 +1，而不是 +5。
#   若不守这一条：后端算得再准，前端不显示也等于没做 —— 那是"交付了但用户看不到"。
                #   147 → 145：`app.js` 改用 `_source_assert.js_code`（剥注释）后，
                #     原先经 `read_text` 的 2 处不再计入 B 类 —— 这是**改造**，
                #     不是删断言。下调基线的 diff 本身就是改造进度的证据。
                #
                #   D-19 新增的净增长：145 - 138 = 7 处
                #     —— `app.js` 已改用 `_source_assert.js_code`（剥注释，不计入 B 类），
                #        剩下 9 处是 `index.html` 的 DOM 锚点（`id="capabilitiesCard"`）
                #        与 `style.css` 的结构块（`.cap-grid` 两栏布局）。
                #
                #   **为什么这 9 处不剥注释也算合法**：
                #     HTML 的 `id=` 与 CSS 的 `.cap-grid {` 是**结构标记**，
                #     不是「文案里出现某词」—— 注释里写 `id="capabilitiesCard"`
                #     不会让 DOM 真的存在该元素，也不会让 CSS 真的有两栏。
                #     B 类要防的是「注释里提一句就让断言通过/变红」，这两项不受影响。
                #
                #   **判定方法（可复跑）**：对该文件做「只在注释里加目标词 → 断言应仍绿」
                #     与「删真实标记 → 断言应红」双向变异，见 D-19 变异实证。
                #
                # 2026-10-21 16:4x 对齐：_classify 扩到别名追踪后实测 138
                #   （原 96 是误设 —— _classify 未追踪 _js_code 薄别名时的可见数，
                #     李工指出那是元守卫盲区，不是断言减少。修复后回升至 138）
                #   教训：元守卫可见数上升=修复分类器；下降=断言被删或盲区扩大。
                #   基线必须反映"分类器修复后"的真实，否则"不增长"防的是空气。


def _classify(tree: ast.AST) -> dict:
    """找出「被赋值为源码/文件文本」的变量名 → 类别。

    ## D-18b 漏检修复：追踪一层别名（2026-10-21）

    旧实现只认**直接赋值** `inspect.getsource(fn)` / `path.read_text()`。
    但实测 `_js_code()` 合并到 `_source_assert` 后，各测试文件的局部
    `_js_code()` 变成了**薄别名**：内部调用 `_source_assert.js_code(APP_JS.read_text(...))`。

    于是 `read_text` 不再是被断言直接使用的变量名 ⇒ 旧 `_classify` 看不到它们 ⇒
    **元守卫漏检了那两个文件**（`test_mu2_frontend_llm` / `test_v10_channel_wiring`）。

    **后果**：任何人写一个本地包装函数
        def _my_js(): return APP_JS.read_text()
        assert "禁止相加" in _my_js()
    就能绕过元守卫 —— 因为 `_my_js()` 不是"直接 read_text 的变量"。

    ## 修法：分两步收集

    1. **文本源**：被赋值为 `getsource`/`read_text` 的变量；
    2. **别名**：被赋值为「返回文本的函数调用」的变量。

    返回文本的函数 = 函数体里 `return` 了 文本源 或 调用（递归一层）。

    ## ⚠️ 已知边界（D-22，2026-10-21 实测，**当前代码库 0 实例**）

    以下三种写法**不会被追踪**，断言会绕过元守卫（写在这里而不是只写在报告里，
    因为 docstring 是每个读代码的人都会看到的地方）：

    1. **逆序定义的两层包装**
       ```python
       def _outer(): return _inner()      # ← 定义在 _inner 之前
       def _inner(): return APP.read_text()
       blob = _outer(); assert "x" in blob     # 漏检
       ```
       根因：第 2 遍是**单次** `ast.walk`，依赖函数出现顺序。

    2. **嵌套调用** `blob = _wrap(_js_code())`
       根因：第 3 遍只看 `node.value.func`，看不到**作为参数**的调用。

    3. **派生容器** `blob = _js_code().splitlines()`
       根因：同上，`.splitlines()` 是属性调用，不是被标记的函数名。

    **实测影响**：真实代码里「比较对象非变量、但内部含文本源」的 `in` 断言
    **= 0 处**（脚本扫描 `tests/test_*.py` 全部文件）。所以这是**未来风险**，
    不是当前失真。

    **若将来引入这些写法**：把第 2/3 遍改成**循环到不动点**（`while changed:`），
    并让第 3 遍扫描整个 `node.value` 子树而非只看 `.func`。
    在改之前，遇到这三种写法请**手动确认元守卫是否数到了它**。
    """
    sources = {}
    # 第一遍：找直接文本源
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        d = ast.dump(node.value)
        if "getsource" in d:
            kind = "py_source"
        elif "read_text" in d or ("open" in d and ".read" in d):
            kind = "file_text"
        else:
            continue
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            if isinstance(t, ast.Name):
                sources[t.id] = kind

    # 第二遍：找「返回文本的函数」→ 把这些函数的名字也标成对应类别
    # 这样 `def _js_code(): return _source_assert.js_code(APP_JS.read_text())`
    # 里 `_js_code` 会被标成 file_text（因为它内部最终读到文本）。
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ret_kinds = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                d = ast.dump(sub.value)
                if "read_text" in d or "getsource" in d:
                    ret_kinds.add("file_text" if "read_text" in d else "py_source")
                # return 里调用了已标为文本源的变量/函数
                for name, kind in sources.items():
                    if name in d:
                        ret_kinds.add(kind)
        if ret_kinds:
            # 只有一个类别时标注；多个类别取 file_text（读文件的更常见）
            kind = "file_text" if "file_text" in ret_kinds else "py_source"
            sources[node.name] = kind

    # 第三遍：本地变量被赋值为「已标记函数」的调用 → 也标同类
    # 例：`js = _js_code()` 而 `_js_code` 已标为 file_text ⇒ `js` 也是 file_text。
    # 只追一层：`js = wrap(_js_code())` 中 `wrap` 不是已标记函数，不追踪（已知边界）。
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if callee in sources:
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    sources[t.id] = sources[callee]

    return sources


def _count(kind_wanted: str) -> tuple:
    """统计指定类别的「子串断言」数量，返回 (总数, [(文件, 行号), ...])。

    ## 「一处」的定义（两个已裁决的口径，改动前必须先改这里）

    1. **以比较操作数为单位**，不以 assert 语句为单位。
       理由：`assert "a" in x and "b" in x` 记为 2 处。
             若按语句算，每条已有 assert 可免费再加任意多条件 ——
             而那正是本守卫要防的攻击形态（往已有断言里塞新条件）。

    2. **只算「用于判定的比较」，不算「产生被比较值的代码」**。
       函数定义 `def _js_code(): return read_text()` 不计入；
       断言 `assert "x" in _js_code()` 计入 1 处。
       理由：否则抽取函数会让计数虚增，而实际断言数未变。

    实测两口径全库差异：按语句 532 / 按比较 619（差 16%）——
    选错口径会让保护范围整体偏移，所以这条不是细节。

    ## 这个计数**不等于运行时实际执行的检查次数**（李工指出的边界）

    `assert ("a" in x) or ("c" in y)` 里 `"c" in y` 会被计 1 处，
    **但短路时它可能永远不执行**。

    这不是计数错误，是「静态计数」的固有性质：
    本守卫数的是「代码里有多少处文本比较」，不是「跑起来会执行多少次」。
    用途是「防止新增文本匹配断言」（静态性质），所以静态计数是正确的选择。

    **若有人想用它推断运行时的检查密度，会得到错误结论 —— 别这么用。**
    """
    total = 0
    where = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        kinds = _classify(tree)
        if not kinds:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            for sub in ast.walk(node.test):
                if not isinstance(sub, ast.Compare):
                    continue
                for op, comp in zip(sub.ops, sub.comparators):
                    if not isinstance(op, (ast.In, ast.NotIn)):
                        continue
                    # 支持两种右操作数：
                    #  * 变量：`assert "x" in src`
                    #  * 函数调用：`assert "x" in _js_code()`（其函数名已由 _classify 标记）
                    #    例：`assert "候选注单" in _level2()`，`_level2` 已标为 file_text。
                    #    `comp.func` 是 Name("_level2")（本地函数）或 Attribute("js_code")。
                    name = None
                    if isinstance(comp, ast.Name):
                        name = comp.id
                    elif isinstance(comp, ast.Call):
                        f = comp.func
                        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if name is not None and kinds.get(name) == kind_wanted:
                        total += 1
                        where.append((p.name, node.lineno))
    return total, where


def test_no_new_python_source_string_assertions():
    """**A 类必须为 0**：不得用 `inspect.getsource` 做子串匹配。

    改用 `tests/_source_assert.py` 的：
    * `code_text(fn)` —— 剥 docstring/注释后的纯代码文本（标识符/表达式）
    * `has_call(fn, attr=..., string_arg=...)` —— 真实调用（含引号形态）
    * `has_compare(fn, op=..., string_const=...)` —— 真实比较（含引号形态）

    ⚠️ **不要用 `ast.unparse` 输出做含引号的子串判断** ——
    它把 `"x"` 规范化成 `'x'`，会让断言假红（已实测）。
    """
    total, where = _count("py_source")
    assert total <= A_BASELINE, (
        f"发现 {total} 处 `getsource` 子串断言（基线 {A_BASELINE}）：\n"
        + "\n".join(f"  {f}:{ln}" for f, ln in where[:10])
        + "\n改用 _source_assert.code_text / has_call / has_compare —— "
          "字符串匹配会被 docstring 骗（假绿）也会误伤注释（假红）。")


def test_external_file_string_assertions_do_not_grow():
    """**B 类不得增长**：`read_text` 子串断言当前 115 处（按比较数）。

    这类断言检查前端文案/结构，没有等价的 AST。
    正确形态是「**先剥注释**再做子串匹配」——
    见 `test_v14_naming_discipline.py` 的 `_strip_js_comments` /
    `_strip_html_comments`，以及 `test_mu2_frontend_llm.py` 的 `_js_code()`。

    **若你删了代码导致数量下降，请同步下调 `B_BASELINE`** ——
    下调基线的 diff 本身就是「改造进度」的交付证据。
    """
    total, where = _count("file_text")
    # 语义：**只防增长**。`total < B_BASELINE` 是允许的（说明你改造了、
    # 但还没把基线调下来 —— 由下面的黄线提醒，不阻断）。
    assert total <= B_BASELINE, (
        f"外部文件子串断言从 {B_BASELINE} 增到 {total}（+{total - B_BASELINE}）：\n"
        + "\n".join(f"  {f}:{ln}" for f, ln in where[-8:])
        + "\n新增此类断言请先剥注释；若确实必要，请说明理由并上调 B_BASELINE。")
    # 基线只许降不许升：实测数少于基线时**提示**下调（不强制红 ——
    # 否则「删了代码但忘了改基线」会让守卫一直绿着而基线失真）。
    #
    # 为什么不直接 assert total == B_BASELINE：
    # 那样「别人删了一处」会立刻让本守卫红，而他并不知道要改这个文件 ——
    # 会变成「守卫因为正确的改动而红」，属于误报。
    # 这里选择「不红，但把基线写进断言消息」的方式不可行（绿了看不到消息），
    # 故改为：只在落后时留一条 xfail 提示 —— 见下方函数。
    pass


def test_b_baseline_should_be_lowered_to_match_reality():
    """**基线同步提示**：实测数低于基线时，应把 `B_BASELINE` 下调到相等。

    ## 为什么单独一条，而不是合进上面那条

    合进去会两难：
    * `assert total <= B_BASELINE` → 删代码后仍绿，基线**永久失真**
    * `assert total == B_BASELINE` → 别人删代码会**无辜变红**（他不知要改这里）

    分开后：
    * 上面那条：**只防增长**（红线，必须红）
    * 这条：**提醒同步**（黄线，可 xfail）

    ## 怎么用

    改造完一处 B 类断言后，把 `B_BASELINE` 减 1。
    若忘了，这条会提示 —— **但不会阻断别人**（xfail 不失败）。
    """
    total, _ = _count("file_text")
    if total == B_BASELINE:
        return
    if total < B_BASELINE:
        pytest.xfail(
            f"实测 {total} 处 < 基线 {B_BASELINE} —— 请把 B_BASELINE 下调到 {total}"
            f"（下调基线的 diff 就是改造进度的证据）")
    # total > B_BASELINE 由上面那条负责报红，这里不重复
