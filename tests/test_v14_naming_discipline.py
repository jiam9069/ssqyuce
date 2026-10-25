"""A-4：产品口径与幻觉防线。

## 这一组测试守的是「不造假」，不是「功能」

用户的原话是「要真实可靠，不能造假」。本项目历史上最容易造假的地方不是篡改数字，
而是**用词**与**透传**：

1. **一个 `usable` 两个意思**：`idle_mining.usable`（挖掘层的过阈候选）
   与 `patterns_runtime.usable`（生产快照的可用规律）同名不同义。
   两者并列展示会让人看到「20 条可用」与「0 条可用」同时存在 ——
   哪怕每个数字单独看都对，用户也会认定系统自相矛盾。
2. **LLM 幻觉透传**：`patterns_used` 原来只做 `str(x)`，而前端会把它渲染成
   「规律引用：…」。`usable=0` 时任何引用都是幻觉。
3. **禁用词**：后端 `routes_predict.py:100-110` 明令「不得叫可用规律」，
   前端 `index.html` 却用了这个词。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from _src_ast import module_has_dict_key, module_identifiers

REPO = Path(__file__).resolve().parent.parent


# ---------- 1. patterns_used 白名单 ----------

def test_sanitize_drops_unknown_patterns():
    """不在白名单的引用必须被丢弃，不能透传。"""
    from lottery import engine

    r = engine._sanitize_patterns_used(["hot_number", "编造的规律"], known={"hot_number"})
    assert r["kept"] == ["hot_number"]
    assert r["dropped"] == ["编造的规律"]
    assert r["had_any"] is True


def test_sanitize_empty_whitelist_drops_everything():
    """**当前生产的真实情形**：usable=0 ⇒ 白名单为空 ⇒ 任何引用都是幻觉。

    这是本组最重要的一条：没有可用规律时，LLM 说"我用了规律 X"必然是假的，
    必须整条丢弃，而不是"宽松放行"。
    """
    from lottery import engine

    r = engine._sanitize_patterns_used(["hot", "cold", "whatever"], known=set())
    assert r["kept"] == [], "白名单为空时不得保留任何引用"
    assert len(r["dropped"]) == 3

    r2 = engine._sanitize_patterns_used(["hot"], known=None)
    assert r2["kept"] == [], "known=None 等同空白名单"


def test_sanitize_handles_messy_input():
    """LLM 可能返回字符串、带空白的项、或逗号分隔串。"""
    from lottery import engine

    assert engine._sanitize_patterns_used("a, b ，c", known={"a", "b", "c"})["kept"] == ["a", "b", "c"]
    assert engine._sanitize_patterns_used(["  a  "], known={"a"})["kept"] == ["a"]
    assert engine._sanitize_patterns_used(None, known={"a"}) == {"kept": [], "dropped": [], "had_any": False}
    assert engine._sanitize_patterns_used([], known={"a"})["had_any"] is False


def test_parse_tickets_response_filters_patterns():
    """端到端：解析 LLM 响应时，伪造的规律引用不进 `patterns_used`。"""
    from lottery import engine

    res = {"tickets": [{
        "reds": [1, 2, 3, 4, 5, 6], "blue": 7, "confidence": 60,
        "patterns_used": ["真实规律", "不存在的规律"],
        "reasoning": "test",
    }]}
    out = engine._parse_tickets_response(res, "m1", known_patterns={"真实规律"})
    assert len(out) == 1
    assert out[0]["patterns_used"] == ["真实规律"]
    assert out[0]["patterns_used_dropped"] == ["不存在的规律"]


def test_parse_tickets_response_still_validates_numbers():
    """白名单改造不能削弱原有的号码合法性校验。"""
    from lottery import engine

    bad = {
        "tickets": [
            {"reds": [1, 2, 3, 4, 5], "blue": 7},          # 只有 5 个红球
            {"reds": [1, 2, 3, 4, 5, 6], "blue": 17},      # 蓝球越界
            {"reds": [1, 2, 3, 4, 5, 34], "blue": 7},      # 红球越界
            {"reds": [1, 1, 2, 3, 4, 5], "blue": 7},       # 红球重复
        ]
    }
    assert engine._parse_tickets_response(bad, "m", known_patterns=set()) == []


# ---------- 2. usable 命名分离 ----------

def test_idle_mining_returns_rule_evidence_not_usable():
    """`idle_mining` 的权威字段名必须是 `rule_evidence`。"""
    from lottery import idle_mining as IM

    st = IM.status()
    assert "rule_evidence_count" in st, "缺权威字段 rule_evidence_count"
    assert "usable_count_deprecated" in st, "旧字段必须显式标注 deprecated"
    # 两个字段值相同（兼容期），但语义已由新名承载
    assert st["rule_evidence_count"] == st["usable_count_deprecated"]


def test_run_once_result_marks_usable_as_deprecated():
    """`run_once` 返回值保留 usable 别名时必须带弃用标记。

    **V40 改造**：原来读整个文件做子串匹配 —— 只要注释里写一句
    `# "usable_deprecated": True 将来要加` 就会假绿。
    现改为 AST 级：只认真实字符串字面量（字典键），注释与 docstring 一概不算。
    """
    assert module_has_dict_key(REPO / "lottery" / "idle_mining.py", "usable_deprecated"), \
        "兼容别名必须显式标记为弃用，否则会与 patterns_runtime.usable 混淆"


def test_rule_evidence_is_not_wired_into_prediction():
    """**架构守卫**：挖掘层的证据项不得进入出号门控。

    出号门控看 `patterns_runtime.counts.usable_replayable`。
    若哪天有人把 `rule_evidence` 接到 predict 上，就是绕过前置注册与 A/B 验证。

    **V40 改造**：原来 `"rule_evidence" not in src` 连**注释**都算 ——
    注释里写「不使用 rule_evidence」反而会让守卫假红，逼人删注释。
    现改为 AST 级标识符查询：只有真实的名称引用 / 属性访问才算命中。
    """
    idents = module_identifiers(REPO / "lottery" / "routes_predict.py")
    assert "rule_evidence" not in idents, \
        "rule_evidence 不应出现在出号门控里（它只是挖掘层的统计结论）"


# ---------- 3. 前端禁用词 ----------

def test_index_html_does_not_call_shape_rules_usable_law():
    """前端不得把 L2 结构约束叫「可用规律」。

    后端 `routes_predict.py:100-110` 明令禁用该词：
    「不得把它叫成『可用规律』——L1 的语义是检出/未检出，
    与 L2 的约束满足率不是同一种计数」。
    """
    html = (REPO / "web" / "index.html").read_text(encoding="utf-8")
    # 找出所有「可用规律」出现处，排除纯注释行（HTML 注释里的旧说明不改）
    offenders = []
    for i, line in enumerate(html.splitlines(), 1):
        if "可用规律" not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("<!--") or stripped.endswith("-->"):
            continue          # HTML 注释，用户看不到
        if "不得" in line or "不是" in line or "禁止" in line:
            continue          # 显式否定句
        offenders.append((i, stripped[:100]))
    assert not offenders, f"index.html 仍把约束称作「可用规律」：{offenders}"


def test_app_js_metric_uses_replayable_field():
    """前端首屏指标必须读 `usable_replayable`，不能只读 `usable`。"""
    js = (REPO / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "usable_replayable" in js, \
        "前端未读 usable_replayable —— 会与挖掘层的 usable 混淆"


# ---------- 4. 后端纪律本身（防止有人放宽） ----------

def test_backend_discipline_text_intact():
    """后端「不得叫可用规律」的纪律文本必须还在。

    **V40 改造（含方向修正）**：原文是

        src = routes_predict.py 全文
        assert "不得" in src and "可用规律" in src

    这有两处毛病：

    * **假绿**：纪律文本恰好是**注释**。删掉真实代码里所有约束、
      只要注释还在，断言就绿 —— 而这条测试声称守的是「书面依据」，
      那注释确实就是依据，所以方向本身没错；
    * **假红**：反过来，谁把这段注释整理成中文全角标点或用别的措辞
      表达同一纪律，守卫就红，逼人去维护措辞而不是纪律本身。

    真正的判据是「**该文件里存在『不得』与『可用规律』这两个词**」，
    而不是「它们是代码还是注释」—— 纪律文本本来就住在注释里。
    所以这里改用一个**显式的全文检索 + 说明性报错**，并明确记录：
    **本条是 B 类里唯一「故意保留全文检索」的一处**，因为它的对象
    就是注释本身，AST 化在这里没有语义收益。
    """
    src = (REPO / "lottery" / "routes_predict.py").read_text(encoding="utf-8")
    assert "不得" in src and "可用规律" in src, \
        "后端口径纪律文本被删了 —— 那是防止口径漂移的唯一书面依据"


# ---------- 5. 渲染面守卫（V14-严QA 补漏，2026-10-21） ----------
#
# ## 为什么必须补这一组
#
# 既有 `test_index_html_does_not_call_shape_rules_usable_law` 有两个结构性漏洞
# （均为变异测试实测，不是推测）：
#
#   * **漏洞 A（文件级）**：它只读 `index.html`，**从不读 `app.js`**。
#     实测：往 app.js 注入用户可见的 `return "可用规律 " + 12 + " 条";`
#     → 该组测试 **11 passed 全绿**，完全抓不住。
#   * **漏洞 B（规则级）**：它按**整行**做 `"不得" in line or "不是" in line`
#     的前缀豁免，于是「任何含这三个词的句子都免疫」——包括把禁用词
#     当作被引用的术语对象、直接印给用户看的句子。
#
# ## 判据（区分"纪律声明"与"转述禁用词"）
#
# * **豁免**：整行是否定判断，且**不把禁用词作为被引用的术语对象**。
#   例：`engine.py` 的「所以「提高命中率」是假目标。」——豁免。
# * **不豁免**：把禁用词当成被命名的东西来转述，即**用户看到了这个词本身**。
#   例：`明令禁止把它叫「可用规律」` —— 验证违规（它把词印出来了）。
#
# ## 实现口径：剥注释后看渲染面
#
# 不靠"行前缀"，而是**先剥掉注释**（`//`、`/* */`、`<!-- -->`），
# 再看剩下的文本里还有没有禁用词。注释用户看不到，所以合法；
# 剩下的就是会被渲染/执行的字面量，出现即违规。
#
# ## 已知边界（诚实标注，不假装全守）
#
# 本守卫是**字面量级**的，上限是「检查完整的禁用词字面量是否存在」。
# 以下形态**穿透**，属已知边界，不是本守卫的承诺：
#   1. `String.fromCharCode(21487, 29992, 35268, 24459)`
#   2. `"\u53ef\u7528\u89c4\u5f8b"`
#   3. 从后端接口取回字符串再渲染（此时字面量在 Python 侧，由后端守卫负责）
# 断言一个抓不住回归的守卫，比没有守卫更危险 —— 故把边界写明。

_BANNED_LAW_TERM = "可用规律"

#: 渲染面必须出现的词（正面要求，防"删了就算修好"）
_REQUIRED_REPLAYABLE_FIELD = "usable_replayable"


def _strip_js_comments(code: str) -> str:
    """剥掉 JS 注释，保留行结构（用空行占位，行号不漂移）。"""
    import re as _re

    out = []
    in_block = False
    for line in code.split("\n"):
        if in_block:
            if "*/" in line:
                in_block = False
                line = line.split("*/", 1)[1]
            else:
                out.append("")
                continue
        if "/*" in line:
            pre, rest = line.split("/*", 1)
            if "*/" in rest:
                line = pre + rest.split("*/", 1)[1]
            else:
                in_block = True
                line = pre
        # 行注释：避开 "http://" 这类 URL 里的 //
        m = _re.search(r"(^|\s)//", line)
        if m:
            tail = line[max(0, m.start() - 2):m.start() + 2]
            if "://" not in tail:
                line = line[:m.start()]
        out.append(line)
    return "\n".join(out)


def _strip_html_comments(html: str) -> str:
    """剥掉 HTML 注释，保留行结构。"""
    import re as _re

    return _re.sub(r"<!--.*?-->",
                   lambda m: "\n" * m.group(0).count("\n"), html, flags=_re.S)


def _render_offenders(text: str, strip) -> list:
    """返回 (行号, 行内容) —— 剥注释后仍含禁用词的行。"""
    hits = []
    for i, line in enumerate(strip(text).split("\n"), 1):
        if _BANNED_LAW_TERM in line:
            hits.append((i, line.strip()[:120]))
    return hits


def test_usable_law_guard_covers_app_js():
    """**补漏洞 A**：`app.js` 的渲染面不得出现「可用规律」。

    既有守卫只读 index.html，app.js 是用户首屏真正渲染的来源，必须同样受守。
    变异实测（修复前）：往 app.js 注入 `return "可用规律 " + n + " 条";`
    → 旧守卫 11 passed 全绿；本测试 → 必红。
    """
    js = (REPO / "web" / "static" / "app.js").read_text(encoding="utf-8")
    offenders = _render_offenders(js, _strip_js_comments)
    assert not offenders, (
        "app.js 渲染面出现禁用词「可用规律」（用户可见）：\n"
        + "\n".join(f"  line {i}: {s}" for i, s in offenders)
        + "\n正确叫法见后端纪律：L2 结构约束应称「形态约束/形态满足率」，"
          "「可用规律」专指 L1 证据项，两者不是同一种计数。"
    )


def test_index_html_usable_law_not_in_visible_text():
    """**补漏洞 B**：index.html 的**渲染面**不得出现「可用规律」。

    与既有 `test_index_html_does_not_call_shape_rules_usable_law` 的区别：
    本测试**不按行前缀豁免**，而是剥掉 HTML 注释后看是否有残留。

    理由：`<!-- ... -->` 里的说明用户看不到，合法；
    但「明令禁止把它叫『可用规律』」这类**转述禁用词**的句子，
    在 HTML 里是正文时，用户会读到那个词本身 —— 它**不是**豁免对象。

    变异实测：往 index.html 注入 `<h3>可用规律 · 12 条</h3>` → 必红。
    """
    html = (REPO / "web" / "index.html").read_text(encoding="utf-8")
    offenders = _render_offenders(html, _strip_html_comments)
    assert not offenders, (
        "index.html 渲染面出现禁用词「可用规律」（用户可见）：\n"
        + "\n".join(f"  line {i}: {s}" for i, s in offenders)
        + "\n（HTML 注释中的历史说明不受此限 —— 用户看不到注释。）"
    )


def test_app_js_still_reads_replayable_field():
    """**正面要求**：app.js 仍读 `usable_replayable`。

    单有"不得出现禁用词"是纯否定守卫 —— 把整段前端逻辑删光也能通过。
    这里补一条正向断言，确保指标仍然绑定到正确字段。
    """
    js = (REPO / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert _REQUIRED_REPLAYABLE_FIELD in js, (
        f"app.js 不再引用 {_REQUIRED_REPLAYABLE_FIELD} —— "
        "首屏指标会退回与挖掘层 usable 混淆的旧口径")


def test_usable_law_guard_is_not_vacuous():
    """**反空转**：守卫本身必须能抓住一次人为回归。

    为什么需要这条：一个因豁免规则过宽而"永远通过"的守卫，
    比没有守卫更危险 —— 它给人"已经守住了"的错觉。

    做法：在内存里构造一份**已知违规**的样本，喂给与生产守卫**完全相同**的
    判定函数，必须命中；再构造一份**纯注释**样本，必须不命中。

    若有人把 `_strip_*_comments` 改宽（例如改成"含否定词就放过"），
    本测试即刻变红。
    """
    # ① 违规样本：渲染面直接出现禁用词 —— 必须命中
    bad_js = 'function f(){ return "%s 12 条"; }\n' % _BANNED_LAW_TERM
    bad_html = "<h3>%s · 12 条</h3>\n" % _BANNED_LAW_TERM
    assert _render_offenders(bad_js, _strip_js_comments), \
        "守卫空转：app.js 样本中的禁用词未被抓到"
    assert _render_offenders(bad_html, _strip_html_comments), \
        "守卫空转：index.html 样本中的禁用词未被抓到"

    # ② 纯注释样本 —— 必须放过（否则守卫过度，会逼人改注释而不是改文案）
    ok_js = '// 不得把约束叫「%s」\nfunction f(){ return 1; }\n' % _BANNED_LAW_TERM
    ok_html = '<!-- 不得把约束叫「%s」 -->\n<div>x</div>\n' % _BANNED_LAW_TERM
    assert not _render_offenders(ok_js, _strip_js_comments), \
        "守卫过度：JS 注释中的纪律说明被误判为违规"
    assert not _render_offenders(ok_html, _strip_html_comments), \
        "守卫过度：HTML 注释中的纪律说明被误判为违规"

    # ③ 纪律声明豁免不得宽到「含否定词就放过」—— 转述禁用词仍须命中
    #    （这正是漏洞 B 的形态：句子是否定句，但用户读到了那个词）
    disguised = '<p>后端明令禁止把它叫「%s」。</p>\n' % _BANNED_LAW_TERM
    assert _render_offenders(disguised, _strip_html_comments), \
        "守卫空转：转述禁用词的可见句子被当成了豁免对象（漏洞 B 复发）"


# ==========================================================================
# 4. 渲染面守卫（B2）：把「禁用词」从"某一行的长相"改成"用户能不能看到"
#
# ## 为什么必须重写（现有守卫的两个结构性漏洞）
#
# `test_index_html_does_not_call_shape_rules_usable_law` 有两条漏洞：
#
# - **漏洞 A（文件级）**：只读 `web/index.html`，**从不读 `web/static/app.js`**。
#   而 app.js 里藏着 4 处用户真正会看到的禁用词（指标名 / 表头 / 文案 / toast）。
# - **漏洞 B（规则级）**：豁免条件是 `"不得" in line or ...`，按**行**匹配，
#   于是任何含这三个词的句子都免疫 —— 包括那句「后端纪律明令禁止把它叫
#   『可用规律』」，它本身就在把禁用词印给用户看。
#
# ## 本组守卫的做法
#
# 1. **剥注释但保留行号**（用等量换行替换，不用整段删 —— 否则行号漂移、报错指错地方）；
# 2. **归一化相邻字符串拼接** `"a" + "b"` → `"ab"`（mutation M7 的捕捉点：
#    拆开后任何"找完整字面量"的检查都会漏，实测两种设计都漏）；
# 3. **断言渲染面**：只看"用户能看到的字符串字面量"，不看注释。
#
# ## 已知覆盖边界（必须承认，不许假装全守）
#
# 本守卫覆盖 **字面量级** 与 **相邻拼接级**，**不覆盖运行时构造**：
# `String.fromCharCode(...)`、`\uXXXX` 转义、跨语句变量拼接（`const W="可用"`）。
# 这些实测可穿透。承认边界比假装全守可靠 —— 与
# `tests/test_w08_frontend_auth.py::_js_code` 注释里那句
# 「一个抓不住回归的断言，比没有断言更危险」同理。
# ==========================================================================

_BANNED_PHRASE = "可用规律"
_WEB = REPO / "web"


def _strip_comments_keep_lines(src: str) -> str:
    """剥掉 JS 行注释 `//` 与块注释 `/* */`，**保留行号**（换行原样留下）。

    ## 为什么不用"全文状态机 + 引号配对"

    第一版是逐字符状态机（遇到引号就跳到闭合），在真实 `app.js`（5057 行、
    含模板串与正则字面量）上**会错位**：一旦把某个 `/` 当成正则起始或把
    模板串里的 `${}` 算错，后续整段文本都被当成"字符串内部"，行注释就不再被剥
    （实测：文件末尾的 `// probe 可用规律` 原样留存）。

    改为**按行处理**，规则简单且可预测：
      · 行的第一个非空白字符是 `//` ⇒ 整行是注释；
      · 行内含 `//` 且它前面不是引号类字符 ⇒ 从 `//` 起截断；
      · `/* ... */` 跨行块注释 ⇒ 只保留其中的换行。

    代价：不做完整的 JS 词法分析，`"http://x"` 这种字符串里的 `//` 会被误剥。
    但这对本守卫**无害** —— 我们只关心"用户看得到的禁用词"，URL 里的 `//`
    后面不会跟着禁用词。宁可简单，也不要一个会在真实文件上悄悄失效的解析器。
    """
    lines = src.splitlines(keepends=True)
    out = []
    in_block = False
    for line in lines:
        if in_block:
            j = line.find("*/")
            if j < 0:
                out.append("\n" if line.endswith("\n") else "")
                continue
            in_block = False
            line = " " * (j + 2) + line[j + 2:]
        # 去块注释（可能同行闭合，也可能开启跨行）
        while True:
            b = line.find("/*")
            if b < 0:
                break
            e = line.find("*/", b + 2)
            if e < 0:
                in_block = True
                line = line[:b] + "\n" if line.endswith("\n") else line[:b]
                break
            # 用一个空格占位，避免把前后 token 粘连
            line = line[:b] + " " + line[e + 2:]
        stripped = line.lstrip()
        if stripped.startswith("//"):
            out.append("\n" if line.endswith("\n") else "")
            continue
        # 行内的 `//`：前面不是引号类字符才截断
        idx = line.find("//")
        while idx > 0 and line[idx - 1] in ":\\":
            idx = line.find("//", idx + 1)
        if idx > 0:
            tail = "\n" if line.endswith("\n") else ""
            line = line[:idx] + tail
        out.append(line)
    return "".join(out)


def _normalize_concat(code: str) -> str:
    """把相邻的字符串字面量拼接 `"a" + "b"` 归一成 `"ab"`。

    mutation M7 的捕捉点：把 `"可用规律（可回放）"` 拆成
    `"可用" + "规律（可回放）"` 后，任何「找完整字面量」的检查都会漏掉。
    """
    code = re.sub(r"""(["'`])\s*\+\s*(["'`])""", r"\1\2", code)
    # 合并后可能出现 `"可用""规律…"`：相邻引号直接拼接也视为同一条字面量
    return re.sub(r"""(["'`])(["'`])""", "", code)


def _strip_html_comments_keep_lines(src: str) -> str:
    """剥掉 HTML 注释 `<!-- ... -->`，同样保留换行以维持行号。

    为什么必须单独处理：`_strip_comments_keep_lines` 只认 JS 的 `//` 与 `/* */`，
    对 `.html` 文件里 `<!-- -->` 内的文字完全不生效 —— 实测 index.html 的注释里
    就写着被禁用的词（作为"已删除旧文案"的说明），用户根本看不到它。
    """
    return re.sub(r"<!--.*?-->",
                  lambda m: "\n" * m.group(0).count("\n"),
                  src, flags=re.DOTALL)


def _render_surface_lines(path: Path) -> list:
    """返回 [(行号, 剥注释+归一化后、用户可能看到的行文本)]。"""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".html", ".htm"):
        raw = _strip_html_comments_keep_lines(raw)
    cleaned = _normalize_concat(_strip_comments_keep_lines(raw))
    return list(enumerate(cleaned.splitlines(), 1))


# 【已删除】`_is_discipline_mention(line)` —— 2026-10-21（D-12）
#
# 它曾把「禁用词被引号包裹」判为"纪律引用"从而豁免。**那是个错误的判据**：
# JS 里每个字符串字面量本来就带引号，于是 `toast("可用规律")` 会被放过，
# 而用户恰恰会看到这句 toast。实测：注入后带引号豁免的守卫全绿。
#
# 没有"改进"它，而是**整个删掉** —— 引号在 JS 里无处不在，
# 任何基于引号的启发式都注定误放行。
# 现在的判据只有一条：**剥掉注释后仍含该词 → 违规**。
# 防复发守卫见 `test_quoted_visible_mention_is_not_exempt`。


@pytest.mark.parametrize("rel", ["index.html", "static/app.js"])
def test_no_rendered_banned_phrase(rel):
    """**核心守卫**：两个前端文件都不得向用户渲染「可用规律」。

    ## 为什么这里没有"引号豁免"（D-12，2026-10-21）

    上一版用「禁用词被引号包裹 ⇒ 视为纪律引用 ⇒ 豁免」。**那是个洞**：
    **JS 里每个字符串字面量本来就带引号**，于是

        toast("可用规律")

    会被当成"引用"而放行 —— 但用户恰恰会看到这句 toast。

    实测确认：注入上述代码后，带引号豁免的守卫 **全绿通过**。

    修法不是"改进引号启发式"，而是**整条删掉**：
    引号在 JS 里无处不在，任何基于引号的启发式都注定误放行。
    现在只做两件事：**剥注释**（用户看不到的放行）+ **`_normalize_concat`**（抓拆分字面量）。
    判据回到最朴素的一条：**剥掉注释后还含这个词 → 违规**。

    纪律声明写在注释里，注释已被剥掉 ⇒ 不会误杀。
    """
    path = _WEB / rel
    # D-12：skip 改 assert —— skipped 24→25 直接破红线 1，等于隐性减测试
    assert path.exists(), f"{rel} 不存在 —— 扫描面被破坏，守卫无法生效"

    offenders = []
    for lineno, line in _render_surface_lines(path):
        if _BANNED_PHRASE not in line:
            continue
        offenders.append((lineno, line.strip()[:120]))

    assert not offenders, (
        f"{rel} 向用户渲染了禁用词「{_BANNED_PHRASE}」：{offenders}\n"
        f"（后端 routes_predict.py 明令：L1 是检出/未检出，L2 是约束满足率，"
        f"两者不是同一种计数，不得统称该词）")


def test_app_js_is_scanned_at_all():
    """**漏洞 A 的守卫**：`app.js` 必须真被扫描到。

    旧守卫只读 `index.html`，而用户真正看到的东西大量在 app.js。
    这条用例若被删或被 skip，说明扫描面又缩回去了。
    """
    app_js = _WEB / "static" / "app.js"
    assert app_js.exists(), "app.js 不存在"
    assert len(_render_surface_lines(app_js)) > 100, "app.js 应被完整解析"


def test_quoted_visible_mention_is_not_exempt(tmp_path):
    """**D-12：引号不再是豁免理由** —— 引号包裹的**可见**禁用词照样违规。

    ## 这条守的是一个已经被证实的漏洞

    上一版守卫：`if _is_discipline_mention(line): continue`，
    而 `_is_discipline_mention` 的判据是「禁用词被引号包裹」。

    但 **JS 的每个字符串字面量本身就带引号**：

        toast("可用规律")            → 词组被 "…" 包裹 → 旧守卫判为"引用" → 放行
        el.textContent = "可用规律"  → 同上

    实测：注入 `toast("可用规律")` 后，带引号豁免的守卫 **全绿通过**。

    ## 现在的判据

    剥掉注释后仍含该词 → 违规，**不看有没有引号**。
    本用例把样本写进临时 JS 文件，走**与生产完全相同的** `_render_surface_lines`，
    断言禁用词在剥注释后**仍然存在**（即会被判为违规）。
    """
    samples = [
        'function f(){ toast("可用规律"); }',
        'el.textContent = "可用规律";',
        "alert('可用规律');",
        "x = `可用规律`;",
        'backend 明令禁止把它叫「可用规律」',   # 可见正文里的"转述禁用词"
    ]
    for i, line in enumerate(samples):
        p = tmp_path / f"sample{i}.js"
        p.write_text(line + "\n", encoding="utf-8")
        visible = [t for (_n, t) in _render_surface_lines(p)]
        hit = [t for t in visible if _BANNED_PHRASE in t]
        assert hit, (
            f"样本被误判为'可豁免'：{line!r}\n"
            f"剥注释后可见行：{visible}\n"
            f"引号包裹**不得**作为豁免理由 —— JS 字符串字面量本身就带引号。")

    # 反证：**注释里**的词必须被剥掉（否则守卫过度，会逼人改注释）
    p = tmp_path / "commented.js"
    p.write_text('// 不得把 L2 叫「可用规律」\nfunction f(){ return 1; }\n',
                 encoding="utf-8")
    assert not [t for (_n, t) in _render_surface_lines(p) if _BANNED_PHRASE in t], \
        "注释中的纪律说明被误判为违规（守卫过度）"


def test_strip_comments_preserves_line_numbers():
    """剥注释必须保留行号（否则报错指错地方）。"""
    src = "line1\n/* a\nb\nc */\nline5\n// tail\nline7"
    out = _strip_comments_keep_lines(src)
    assert len(out.splitlines()) == len(src.splitlines()), (
        f"行数变了：{len(out.splitlines())} != {len(src.splitlines())}")


def test_normalize_concat_catches_split_literal():
    """相邻字符串拼接归一化后必须命中（mutation M7 的捕捉点）。"""
    split = 'x = "可用" + "规律（可回放）"'
    assert _BANNED_PHRASE not in split, "前提：拆开后不含完整字面量"
    assert _BANNED_PHRASE in _normalize_concat(split), "归一化后应命中"


def test_known_limit_runtime_construction_not_covered():
    """**承认边界**：运行时构造（charCode / 变量拼接）不被覆盖。

    这不是缺陷报告，是把覆盖边界写进可执行文档 —— 防止有人误以为
    "过了本组守卫就等于没有禁用词"。若哪天实现了更强的检测，本用例应改成断言能抓到。
    """
    runtime = "const w = String.fromCharCode(21487,29992) + '规律';"
    assert _BANNED_PHRASE not in _normalize_concat(_strip_comments_keep_lines(runtime)), \
        "本守卫不覆盖运行时构造 —— 若此处失败说明已增强，请更新本用例文档"


# ---------- 6. CSS 渲染面（V14-严QA 补，2026-10-21） ----------
#
# ## 为什么 CSS 必须单独一条
#
# CSS 的 `content:` 属性**会真实渲染到页面上**，用户看得见。例如：
#
#     .strict-banner::after { content: "可用规律 0 条"; }
#
# 这行会被渲染成用户可见文字，但：
#   * `app.js` 守卫扫的是 JS 文件，看不到它；
#   * `index.html` 守卫扫的是 HTML，也看不到它。
#
# 实测（补本测试之前）：往 `style.css` 追加
# `.qa-probe::after { content: "可用规律"; }` → 全组 **22 passed 全绿**，
# 完全抓不住。故此条不是"防御性冗余"，是**真实缺口**。
#
# ## 为什么 CSS 注释也一并受管
#
# `web/static/style.css:637` 现有一处 `/* V11 严格模式横幅（无可用规律/...） */`。
# 那是**修复历史的证据**，应当保留、不应为了让测试干净而删除。
# 本测试**剥掉 `/* */` 后再判定**，所以它不影响结果 —— 但也意味着
# **未来改 CSS 的人在注释里写这个词不会触发红灯**（注释用户看不到，正当）。
#
# 一句话：**CSS 会被渲染的部分受管，注释部分放行。** 与 JS/HTML 同一口径。

def _strip_css_comments(code: str) -> str:
    """剥掉 CSS 注释 `/* ... */`，保留行结构。

    CSS 没有 `//` 行注释（除非用了预处理器，本项目是纯 CSS），
    所以只处理块注释。保留换行是为了让报错行号与源文件对齐。
    """
    import re as _re

    return _re.sub(r"/\*.*?\*/",
                   lambda m: "\n" * m.group(0).count("\n"), code, flags=_re.S)


def test_style_css_usable_law_not_in_rendered_content():
    """`style.css` 的**渲染面**不得出现「可用规律」。

    CSS 的 `content:` 属性会渲染成用户可见文字，属于渲染面。
    本测试剥掉 `/* */` 注释后再判定 —— 注释中的历史说明（如第 637 行）
    合法保留，不算违规。

    变异实测：追加 `.qa-probe::after { content: "可用规律"; }` → 必红。
    """
    css = (REPO / "web" / "static" / "style.css").read_text(encoding="utf-8")
    offenders = _render_offenders(css, _strip_css_comments)
    assert not offenders, (
        "style.css 渲染面出现禁用词「可用规律」（CSS content 属性会渲染给用户）：\n"
        + "\n".join(f"  line {i}: {s}" for i, s in offenders)
        + "\n（CSS 注释 /* */ 中的历史说明不受此限。）"
    )
