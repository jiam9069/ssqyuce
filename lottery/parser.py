"""ssq.TXT 行解析。数据源格式（已实测验证，每行固定 31 字段）：

期号 日期 红1..红6(升序) 蓝 红1..红6(摇奖顺序) 销量 奖池
一等奖注数 一等奖奖金 二等奖注数 二等奖奖金 三等奖注数 三等奖奖金
四等奖注数 四等奖奖金 五等奖注数 五等奖奖金 六等奖注数 六等奖奖金 预留 预留
"""
from __future__ import annotations

from typing import Dict, List, Optional

FIELD_COUNT = 31


class ParseError(ValueError):
    pass


def parse_line(line: str) -> Dict:
    parts = line.split()
    if len(parts) != FIELD_COUNT:
        raise ParseError(f"字段数异常: {len(parts)} != {FIELD_COUNT} | 行: {line[:80]}")
    issue, date = parts[0], parts[1]
    reds = [int(x) for x in parts[2:8]]
    blue = int(parts[8])
    order = [int(x) for x in parts[9:15]]
    try:
        sales, pool = int(parts[15]), int(parts[16])
        prizes = [int(x) for x in parts[17:29]]
    except ValueError:
        sales, pool, prizes = 0, 0, [0] * 12
    if len(set(reds)) != 6 or not all(1 <= x <= 33 for x in reds):
        raise ParseError(f"红球非法: {reds}")
    if not 1 <= blue <= 16:
        raise ParseError(f"蓝球非法: {blue}")
    return {
        "issue": issue,
        "date": date,
        "reds": reds,
        "blue": blue,
        "order": order,
        "sales": sales,
        "pool": pool,
        "prizes": prizes,
    }


def parse_text(text: str, *, strict: bool = False,
               on_bad: Optional[list] = None) -> List[Dict]:
    """解析整段文本，返回有效开奖记录。

    ## 坏行为什么不再整体抛错（V37，2026-10-24 审计 P2 的健壮性项）

    旧实现遇到**任意一行**坏数据就 ``raise``，于是上游源站多打一个空行、
    多一段 HTML 报错页、或某行字段被截断，整批数据（几千期）**全部作废**，
    前端表现为"刷新后一期都没有"。而源站出这类小毛病是常态。

    正确语义：**能解析的都入库，解析不了的如实报出**。
    所以默认 ``strict=False`` —— 跳过坏行并继续；
    需要"一个坏行都不许有"的场景（如自检）显式传 ``strict=True`` 拿回旧行为。

    ``on_bad`` 若传入一个 list，会被追加 ``(行号, 原因)``，便于调用方上报。
    """
    draws: List[Dict] = []
    bad: List[tuple] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            draws.append(parse_line(line))
        except ParseError as e:
            bad.append((i, str(e)))
            if strict:
                raise ParseError(f"第 {i} 行: {e}")
    if on_bad is not None:
        on_bad.extend(bad)
    if not draws and bad:
        # 一条都没解析出来、又全是坏行：这更像"源变了/取到了错误页"，
        # 必须让调用方知道，不能安静返回空列表
        raise ParseError(
            f"全部 {len(bad)} 行都无法解析（首个错误在第 {bad[0][0]} 行：{bad[0][1]}）。"
            "可能是数据源改版或取到了错误页面，请检查数据源地址后重试。")
    return draws