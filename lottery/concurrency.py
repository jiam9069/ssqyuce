"""重计算端点的并发闸门（V36，2026-10-24 审计 P1-13）。

## 问题

`/api/predict`、`/api/backtest/*`、`/api/portfolio/*`、`/api/rigor/*` 这些端点
每次要跑数秒到数分钟的 CPU 计算（walk-forward 融合、C(33,6) 枚举、置换检验）。
项目此前**只有登录限流**（`auth.login_rate_limited`），没有任何面向重计算的
并发保护 —— 任意登录用户并发刷几次，就能把容器 CPU 打满
（compose 上限 2.4 核），连带把只读端点也拖慢。

## 为什么是"并发闸门"而不是"速率限制"

速率限制（每秒 N 次）挡不住这个问题：攻击者可以**每 10 秒发 20 个**
并发请求，速率完全合规，但同时在算的仍是 20 个重活。
真正要限制的是**同时在跑的数量**（in-flight），因为 CPU 是它吃满的。

所以这里用信号量：超过上限立即 **503 + 可执行指引**（不是排队 ——
排队会让请求越堆越多，最终全部超时，体验更差）。

## 为什么按"领域"分桶

预测、回测、组合优化彼此独立，一个卡住不该饿死另一个。
每桶独立计数，桶容量可按端点开销配置。

## 不覆盖的端点

纯读端点（`/api/health`、`/api/info`、静态资源、`/api/auth/*`）不在
保护范围内 —— 它们本身不重，且探活必须永远可达（容器 healthcheck 依赖它）。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Optional

#: 每个桶的并发上限。默认值按"容器 2.4 核"设定：
#: 预测单请求约 300ms~2s CPU，回测可达分钟级，故预测桶宽一点、回测桶窄一点。
DEFAULT_LIMITS: Dict[str, int] = {
    "predict": 3,
    "backtest": 1,
    "portfolio": 2,
    "rigor": 1,
    "mining": 1,
    "default": 2,
}

#: 路径前缀 → 桶名。**顺序敏感**：从上到下第一个匹配生效，具体在前。
_BUCKETS = (
    ("/api/predict", "predict"),
    ("/api/backtest", "backtest"),
    ("/api/portfolio", "portfolio"),
    ("/api/rigor", "rigor"),
    ("/api/mining", "mining"),
    ("/api/evolution", "mining"),
    ("/api/llm/audit", "mining"),
    ("/api/llm_lab", "mining"),
)

_sems: Dict[str, threading.BoundedSemaphore] = {}
_sems_lock = threading.Lock()
_inflight: Dict[str, int] = {}


def _sem(bucket: str) -> threading.BoundedSemaphore:
    with _sems_lock:
        s = _sems.get(bucket)
        if s is None:
            s = threading.BoundedSemaphore(DEFAULT_LIMITS.get(bucket, 2))
            _sems[bucket] = s
        return s


def bucket_for(path: str) -> Optional[str]:
    """该路径属于哪个并发桶；None = 不受限（只读/鉴权/探活）。"""
    for prefix, name in _BUCKETS:
        if path.startswith(prefix):
            return name
    return None


def inflight(bucket: str) -> int:
    with _sems_lock:
        return _inflight.get(bucket, 0)


@contextmanager
def _slot(bucket: str):
    """占用一个并发位；拿不到就抛 Busy。"""
    sem = _sem(bucket)
    if not sem.acquire(blocking=False):
        raise Busy(bucket)
    with _sems_lock:
        _inflight[bucket] = _inflight.get(bucket, 0) + 1
    try:
        yield
    finally:
        with _sems_lock:
            _inflight[bucket] = max(0, _inflight.get(bucket, 1) - 1)
        sem.release()


class Busy(RuntimeError):
    """并发位已满。"""

    def __init__(self, bucket: str):
        self.bucket = bucket
        super().__init__(bucket)


def acquire(bucket: str):
    """同步端点用：返回一个**可手动释放**的上下文管理器。"""
    return _slot(bucket)
