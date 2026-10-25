"""Python 源码的结构化断言工具（V40，B 类改造第三批）。

## 为什么需要它（而不是继续用 `read_text() + in`）

`tests/_source_assert.py` 解决的是「**手里已有函数对象**」的 AST 化。
但仓库里还有一批断言拿不到函数对象：

    src = (REPO / "lottery" / "routes_predict.py").read_text(encoding="utf-8")
    assert "rule_evidence" not in src

它们守的是**整个模块**级事实（"这个文件里不许出现某标识符"），
`inspect.getsource` 用不上，于是退化成子串匹配 —— 又回到两个老毛病：

* **假绿**：`# rule_evidence 不该接进来` 这行注释就足以让 `not in` 变**绿**，
  但把注释删掉、代码原样接进来，断言**又是绿的** —— 它守的是"文本里没这个词"，
  不是"代码没这么写"。
* **假红**：在注释里解释「不使用 rule_evidence」是**自然写法**，却会让守卫无辜变红，
  逼人去删注释而不是审视代码。

本模块把这类断言提升到 AST：

* `module_identifiers(path)` —— 模块里**真实代码**中的标识符集合（剥注释与 docstring）
* `module_strings(path)` —— 模块里**真实代码**中的字符串字面量集合（剥注释与 docstring）
* `module_code_text(path)` —— 剥注释与 docstring 后的纯代码文本
* `identifiers_of(source)` / `strings_of(source)` —— 同上，但输入是源码字符串

## 与 `_source_assert` 的分工

| 场景 | 用哪个 |
|---|---|
| 手里有函数对象，断言函数体 | `_source_assert.code_text / has_call / has_compare` |
| 只能拿到文件路径，断言整个模块 | **本模块** |

## 前端文件（.js/.html/.css）不走这里

它们不是 Python，`ast` 解析不了。前端继续用 `_source_assert.js_code`
（剥 JS 注释）以及各测试文件自己的剥 HTML/CSS 注释函数 —— 那是正确做法，
不需要改，也不计入本模块的适用范围。
"""
from __future__ import annotations

import ast
import pathlib
from typing import Set, Union

PathLike = Union[str, pathlib.Path]


def _strip_docstrings(tree: ast.AST) -> None:
    """原地剥掉模块与所有函数/类的 docstring（第一个语句若是字符串常量）。"""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:]


def _parse(source: str) -> ast.Module:
    tree = ast.parse(source)
    _strip_docstrings(tree)
    return tree


def identifiers_of(source: str) -> Set[str]:
    """源码中**真实代码**里出现的全部标识符（变量、函数名、属性名、调用名…）。

    注释天然不在 AST 里，所以不必额外处理；docstring 已剥。
    """
    names: Set[str] = set()
    for node in ast.walk(_parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword):
            # 关键字实参 `f(rule_evidence=1)` —— `ast.keyword` **不是** `ast.arg`，
            # 单独漏掉这一类会让"传参接线"的守卫假绿（实测踩到）。
            if node.arg:
                names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def strings_of(source: str) -> Set[str]:
    """源码中**真实代码**里出现的全部字符串字面量。"""
    return {
        node.value
        for node in ast.walk(_parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def code_text_of(source: str) -> str:
    """剥掉 docstring 后的纯代码文本（`ast.unparse`）。

    ⚠️ 与 `_source_assert.code_text` 同样的限制：`unparse` 会把双引号
    规范化成单引号，**含引号的断言不能用这条路**，改用 `strings_of`。
    """
    tree = _parse(source)
    return ast.unparse(ast.Module(body=tree.body, type_ignores=[]))


def _read(path: PathLike) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def module_identifiers(path: PathLike) -> Set[str]:
    """模块文件里**真实代码**中的标识符集合。"""
    return identifiers_of(_read(path))


def module_strings(path: PathLike) -> Set[str]:
    """模块文件里**真实代码**中的字符串字面量集合。"""
    return strings_of(_read(path))


def module_code_text(path: PathLike) -> str:
    """模块文件剥掉 docstring 后的纯代码文本。"""
    return code_text_of(_read(path))


def module_has_dict_key(path: PathLike, key: str) -> bool:
    """模块里是否存在以该字符串为**字典键 / 字面量实参**的写法。

    专门服务「`"usable_deprecated": True` 这种键必须存在」的断言 ——
    子串匹配连注释里的同名字样都算，这里只认真实字典键。
    """
    return key in module_strings(path)


def module_mentions_attr(path: PathLike, attr: str) -> bool:
    """模块代码里是否出现该属性/标识符（不含注释与 docstring）。"""
    return attr in module_identifiers(path)
