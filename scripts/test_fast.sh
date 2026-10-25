#!/usr/bin/env bash
# test_fast.sh —— 日常开发快速反馈入口：跑 fast 子集（= 排除 slow 标记）。
#
# 用法：
#   scripts/test_fast.sh                     # fast 子集（-m "not slow"）
#   scripts/test_fast.sh -k registry         # 额外参数原样透传给 pytest
#
# ⚠ fast ≠ 完整测试：分层只"分组"不"跳过"，slow 集合的测试一条没少、
#    只是被这个入口排除。发布前 / 全量验收必须跑全量：
#      scripts/isolated_test.sh -- .venv/bin/python -m pytest tests -q
#
# 为什么仍走 isolated_test.sh：隔离与生产库护栏是保命的（2026-10-21 有
# 测试账号写进生产库），快速跑不豁免隔离。
#
# 为什么不加 -e（与 isolated_test.sh 同款理由，2026-10-21 踩过）：
# 加 -e 会让 trap 读到错误那步的退出码而非 pytest 的码，退出码语义被改变，
# 「测试失败」和「环境没搭好」会混成一件事。这里 exec 原样透传 rc，
# CI 可直接用。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$REPO_ROOT/scripts/isolated_test.sh" -- \
  "$REPO_ROOT/.venv/bin/python" -m pytest tests -m "not slow" -q "$@"
