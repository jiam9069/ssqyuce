"""V40-infra：测试分层基础设施的元测试（守测试基础设施本身）。

背景：分层标记（fast/slow/integration）落地后，最大的风险不是"标记写错"，
而是**分层悄悄变成跳过**——某天有人在 pytest.ini 的 addopts 里加了 -m，
或者改坏了收集，全量数字静默缩水，检出能力跟着蒸发。本文件守三件事：

1. markers 全部注册（--markers 可解析）——防打字错导致 -m 表达式失效
   （未注册的 marker 在 -m 里会让整条表达式匹配不到任何测试）；
2. 分层标记不会让任何测试被默认跳过：pytest.ini 的 addopts 不带 -m；
3. 全量收集数下界锁：>= 1990。构成（2026-10-25 实测）：
   1919（任务基线）+ 12（test_v40_debt_wrapup_a）+ 1（wilson 真值锁）
   + 64（coverage 两文件当期）− 收尾批次波动缓冲。coverage 批次继续
   增加只会让真实数更大（三连跑当天实测 2028）；锁下界防的是
   "有人偷偷 skip 整批测试"，精确等值由验收报告人工核对。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
REQUIRED_MARKERS = ("fast", "slow", "integration")
COLLECTION_FLOOR = 1990


def _pytest(*args: str) -> str:
    """跑一次只读的 pytest 子命令，返回合并输出（不落任何测试状态）。"""
    out = subprocess.run(
        [str(PY), "-m", "pytest", *args, "-q", "--rootdir=" + str(REPO)],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    # rc=5 是"没有收集到测试"（--markers 不适用），对收集类子命令同样合法
    assert out.returncode in (0, 5), out.stdout[-2000:] + out.stderr[-500:]
    return out.stdout + out.stderr


def test_all_markers_registered_in_pytest_ini():
    """tests/pytest.ini 必须注册全部分层 marker，--markers 可解析。

    注意 args 必须带 `tests`：pytest 的 inifile 发现基于 args 的共同祖先，
    无 args 时不会命中 tests/pytest.ini（rootdir≠inifile）。
    """
    out = _pytest("--markers", "tests")
    for m in REQUIRED_MARKERS:
        assert f"pytest.mark.{m}" in out, f"marker {m} 未注册：\n{out[:800]}"


def test_pytest_ini_addopts_has_no_default_filter():
    """默认 addopts 不得带 -m —— 分层只分组不跳过的静态保证。

    若有人往 addopts 加了 -m，裸 `pytest tests` 会静默排除一批测试，
    全量数字缩水而 CI 依旧全绿 —— 这正是本文件要钉死的退化路径。
    """
    ini = (REPO / "tests" / "pytest.ini").read_text(encoding="utf-8")
    lines = [l for l in ini.splitlines() if l.strip().startswith("addopts")]
    assert lines, "tests/pytest.ini 缺 addopts 行（分层配置被删了？）"
    assert "-m" not in lines[0], (
        f"默认 addopts 带 -m 过滤（分层变成默认跳过）：{lines[0].strip()}")


def test_full_collection_count_meets_floor():
    """全量收集数 >= 1990；构成与缓冲见模块 docstring。"""
    out = _pytest("--collect-only", "tests")
    m = re.search(r"(\d+) tests? collected", out)
    assert m, f"无法解析收集数：\n{out[-600:]}"
    n = int(m.group(1))
    assert n >= COLLECTION_FLOOR, (
        f"收集数 {n} < {COLLECTION_FLOOR} —— 有测试被 skip / 删除 / 收集失败？"
        "分层改动后全量必须一个不少。")
