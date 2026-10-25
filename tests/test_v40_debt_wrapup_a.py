"""V40：审计剩余债务的回归锁（第一批）。

本文件覆盖审计里剩下的几条「语义空转 / 口径含混」类债务。它们的共同点是
**计算结果没错，但结论没被传到用户面前、或被写错在文档里** —— 属于可信度债，
不是功能债。

## 本文件守的四组

1. `shortfall_reason` 语义不空转：后端算出的三种原因，前端必须真的渲染出来。
   此前后端在 `engine.py:943-949` 认真区分了 `llm_only_actual_count` /
   `candidate_pool_exhausted` / `all_methods_disabled`，而前端只显示
   "5 / 请求 10" —— 用户分不清"系统坏了"与"设计如此"。
2. `CPU_LIMIT_PCT` 的口径文档必须说清「分母是什么」。
   历史上一版把它写成「整机 60%」，而 V39 之后 `_system_cpu_pct()` 在容器内
   的分母是 cgroup 配额（2.4 核），同一个 60 代表的绝对核数变了。
3. `_src_ast` 辅助模块自身的辨伪能力：它必须能区分「注释里有这个词」与
   「代码里真的用了这个词」—— 否则 B 类 AST 化就是把假绿换个地方藏。
4. B 类基线只许降：本轮 AST 化之后，基线必须同步下调（否则基线永久失真，
   防的是空气）。
"""
from __future__ import annotations

import pathlib

import pytest

from _src_ast import identifiers_of, module_has_dict_key, module_identifiers, strings_of

REPO = pathlib.Path(__file__).resolve().parent.parent
APP_JS = REPO / "web" / "static" / "app.js"
IDLE = REPO / "lottery" / "idle_mining.py"


# ===========================================================================
# 1. shortfall_reason 语义不空转
# ===========================================================================

#: 后端 `engine.py:943-949` 会产出的三种原因。这是**外部给定**的清单：
#: 每一种都必须在前端有对应文案，否则该原因等同没算。
SHORTFALL_REASONS = (
    "llm_only_actual_count",
    "candidate_pool_exhausted",
    "all_methods_disabled",
)


@pytest.mark.parametrize("reason", SHORTFALL_REASONS)
def test_frontend_explains_every_shortfall_reason(reason):
    """每一个后端原因都必须在前端有可读文案。

    变异验证：把 `app.js` 里 `_shortTxt` 映射表中的 `candidate_pool_exhausted`
    删掉 → 本测试变红（而所有旧测试仍绿，因为没人断言过前端契约）。
    """
    from _source_assert import js_code
    code = js_code(APP_JS.read_text(encoding="utf-8"))
    assert f"{reason}:" in code or f"{reason} :" in code, (
        f"前端没有为 shortfall_reason={reason!r} 提供文案 —— "
        "后端算出的原因到不了用户眼前，等于没算")


def test_frontend_shortfall_text_is_not_vacuous():
    """反空转：文案必须真的被拼进输出，而不是定义了就没人用。

    只检查「映射表里有这个 key」会假绿 —— 定义完不插进 `items.push` 也行。

    实现说明：`_shortTxt` 与 `items.push(...)` 在源码里跨行铺开
    （映射表在 957 行，push 在 962-964 行），所以**按行找必然误判**。
    这里改为把「从 `_shortTxt` 的定义到下一个 items.push 结束」这段
    整体取出，再确认 push 的参数里引用了 `_shortTxt`。
    """
    from _source_assert import js_code
    code = js_code(APP_JS.read_text(encoding="utf-8"))
    assert "_shortTxt" in code, "前端没有 shortfall 文案变量"

    idx = code.index("const _shortTxt")
    window = code[idx:idx + 900]          # 定义 + 紧随其后的 push 语句
    push_at = window.find("items.push")
    assert push_at != -1, "找不到紧随 shortfall 文案之后的 items.push"
    push_stmt = window[push_at:push_at + 400]
    assert "_shortTxt" in push_stmt, (
        "`_shortTxt` 被计算但没进入紧随其后的 items.push —— 用户仍然看不到原因\n"
        f"push 语句片段：{push_stmt[:200]!r}")


# ===========================================================================
# 2. CPU_LIMIT_PCT 口径文档必须写清分母
# ===========================================================================

def test_cpu_limit_pct_documents_its_denominator():
    """`CPU_LIMIT_PCT` **自身紧邻的注释块**必须说明分母是什么。

    ## 为什么窗口必须收窄（变异实测踩到的假绿）

    第一版取的是「定义前 4000 字符」，结果变异测试里把该注释块改回含糊的
    `#: 相对整机`、删掉 cgroup/配额/分母，**断言仍然通过** —— 因为文件下方
    `_system_cpu_pct()` 的 docstring 里也写着 cgroup，凑满了关键词。
    那是典型的「关键词扫荡」假绿：只要文件里还有别处提到，就永远绿。

    现改为：从常量定义往上找到**最近的一个注释块边界**（连续 `#:` 行），
    只在这个块内要求关键词齐全。

    ## 为什么值得一条测试守文档

    V39 把 `_system_cpu_pct()` 改成容器内优先读 cgroup 配额（实测 2.4 核），
    但常量注释仍写着「整机 60% = 2.4 核」。同一个数字 `60` 在新旧口径下
    代表的绝对 CPU 差 40%，而**代码一行没变** —— 只有文档能告诉后人哪个对。
    """
    lines = IDLE.read_text(encoding="utf-8").splitlines()
    def_idx = next(i for i, ln in enumerate(lines)
                   if ln.startswith("CPU_LIMIT_PCT = "))
    # 向上收集紧邻的连续注释块（`#:` 文档注释）
    block: list[str] = []
    i = def_idx - 1
    while i >= 0 and lines[i].lstrip().startswith("#:"):
        block.append(lines[i])
        i -= 1
    text = "\n".join(reversed(block))
    assert text.strip(), "CPU_LIMIT_PCT 上方没有文档注释块（`#:` 行）"
    for word in ("cgroup", "配额", "分母"):
        assert word in text, (
            f"CPU_LIMIT_PCT 紧邻的注释块里缺少 {word!r}（只检查该块，不看文件其他位置）\n"
            f"实际块内容：\n{text}")


def test_throttle_docstring_mentions_conservative_bias():
    """`_throttle` 必须记下「one_core 用宿主核数、分母用配额」造成的保守偏差。

    这是**已知且故意保留**的口径不一致：`one_core = 100 / sched_getaffinity`
    在容器里算 25 点，而真正的「一个核」约为 41.7 点 ⇒ 更早让路 ⇒ 更保守。
    方向安全，但必须留痕 —— 否则未来有人会以为限流算错了并去「修」它，
    反而修破一致性。
    """
    from lottery import idle_mining
    doc = idle_mining._throttle.__doc__ or ""
    assert "保守" in doc or "偏差" in doc, (
        "_throttle 的 docstring 没有记录 one_core 与分母不一致造成的保守偏差")


# ===========================================================================
# 3. `_src_ast` 自身的辨伪能力（防止 AST 化把假绿换个地方藏）
# ===========================================================================

def test_identifiers_ignores_comments_and_docstrings():
    """注释与 docstring 里的词不得被算作「标识符」。"""
    src = (
        '"""模块 docstring 里提到 rule_evidence 不算。"""\n'
        "# 注释里提到 rule_evidence 也不算\n"
        "def f():\n"
        '    """函数 docstring 里提到 rule_evidence 同样不算。"""\n'
        "    return 1\n"
    )
    assert "rule_evidence" not in identifiers_of(src)
    assert "rule_evidence" not in strings_of(src)


def test_identifiers_catches_real_usage():
    """真实代码引用必须被抓到（名字 / 属性 / 关键字实参 / 字典键四种形态）。"""
    assert "rule_evidence" in identifiers_of("x = rule_evidence\n")
    assert "rule_evidence" in identifiers_of("x = obj.rule_evidence\n")
    assert "rule_evidence" in identifiers_of("f(rule_evidence=1)\n")
    assert "rule_evidence" in strings_of('d = {"rule_evidence": 1}\n')


def test_idle_mining_marks_usable_alias_deprecated_via_ast():
    """`usable_deprecated` 必须是**真实字典键**，不能只是注释里的字样。

    这是把 `test_v14_naming_discipline` 那条断言 AST 化后的等价锁，
    单独留一条是为了让"AST 化没改变契约"这件事可被独立验证。
    """
    assert module_has_dict_key(IDLE, "usable_deprecated")


def test_routes_predict_does_not_wire_rule_evidence():
    """**架构守卫**（AST 版）：`rule_evidence` 不得进入出号门控的代码。"""
    assert "rule_evidence" not in module_identifiers(REPO / "lottery" / "routes_predict.py")


# ===========================================================================
# 4. B 类基线只许降
# ===========================================================================

def test_b_baseline_reflects_ast_conversion():
    """B 类基线必须等于实测，不允许仍停在旧值。

    为什么单独锁：元守卫 `test_external_file_string_assertions_do_not_grow`
    只防增长，删掉断言后会一直绿着，**基线永久失真**。
    这条把它钉住：基线必须等于实测，差一个都不行。
    """
    from test_qa_source_assert_meta import B_BASELINE, _count

    total, _where = _count("file_text")
    assert total == B_BASELINE, (
        f"B 类实测 {total} 处，基线写的是 {B_BASELINE} —— "
        "改造完必须同步下调基线，否则基线失真，守卫防的是空气")
