"""U9：历史战绩卡的样本口径与页面顺序（前端结构回归守卫）。

背景（用户实报）：卡片标题写的是「系统近 N 期 · 诚实口径」——**N 是从未替换的字面量**，
既不是实际期数也不是查询上限；而在只有 1 期样本时，前端把各方法的 `issues` 直接相加
（6 个方法 × 1 期 = 错报 6 期）。本仓库没有 JS 测试运行器，故用 pytest 做结构断言，
锁住这两条真实回归：口径必须由接口实际返回的样本决定、且不得再走「相加」的错误聚合。
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "app.js").read_text(encoding="utf-8")


def test_no_literal_period_placeholder_in_titles():
    """标题里不得再出现未替换的占位符（近 N 期 / 近 X 期 之类）。"""
    assert "近 N 期" not in INDEX
    assert "近 N 期" not in APP_JS
    # 「历史战绩」标题的期数必须是 JS 运行时写入的（有 id），不能写死在 HTML 里
    m = re.search(r"🏆 历史战绩.*?</h2>", INDEX, re.S)
    assert m, "未找到历史战绩卡片标题"
    assert 'id="trackRecordTag"' in m.group(0)


def test_track_record_card_sits_below_win_receipt():
    """历史战绩卡必须排在「上期开奖回执」之后。"""
    receipt = INDEX.index('id="winReceiptCard"')
    track = INDEX.index('id="trackRecord"')
    assert track > receipt, "历史战绩卡应位于上期开奖回执卡下方"


def test_track_record_counts_distinct_issues_not_sum():
    """期数须按 issue 去重求并集：各方法 issues 相加会把同一期重复计 N 次。"""
    assert "tIssues += (g.issues" not in APP_JS, "不得再把各方法的 issues 直接相加"
    assert "issueSet.add(" in APP_JS, "期数应按 rows 里的 issue 去重"
    assert "issueSet.size" in APP_JS


def test_track_record_reports_query_cap_separately():
    """查询上限（sample_limit）只能作为上限展示，不能当作实际样本量。"""
    assert "sample_limit" in APP_JS
    body = APP_JS[APP_JS.index("function renderTrackRecord("):]
    body = body[:body.index("\nfunction ", 10)]
    assert "查询上限" in body, "应把 sample_limit 明确标注为查询上限"
    assert "尚未达到查询上限" in body, "样本未满时应如实说明"


def test_track_record_gates_best_method_on_sample_size():
    """样本不足时不得给出「最佳方法」结论（与后端 60 期筛查阈值对齐）。"""
    body = APP_JS[APP_JS.index("function renderTrackRecord("):]
    body = body[:body.index("\nfunction ", 10)]
    assert "TRACK_METHOD_MIN_ISSUES" in body
    assert "远不足以比较方法优劣" in body


def test_track_record_error_is_not_rendered_as_zero_sample():
    """取数失败必须显式报错——不能伪装成「样本为 0」。"""
    assert "renderTrackRecordError" in APP_JS
    assert "不代表样本为 0" in APP_JS
