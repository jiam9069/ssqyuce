"""全局测试隔离与公共 fixture（V10-00b）。

为什么需要这个文件（缺陷背景）
------------------------------
`docs/reviews/R4_RELEASE_REVIEW.md` P0-B 实测：本目录此前**没有 conftest.py**，
19 处用例用裸赋值 ``config.DB_PATH = tmp_path / ...`` 把全局配置改到临时库且
不还原；而 ``lottery/db.py`` 还有模块级连接缓存 ``_CONN``。结果是：

* 某用例把库换到临时目录后，**排在它后面的用例**都在这个（可能是空的）临时库上跑；
* 全量 ``pytest -q`` 当时能过 **纯属文件字母序巧合**（污染源都在 ``test_v4_*`` 之前，
  唯一受害用例是 ``test_v4_honest_ui.py``）。新增的 ``test_v10_*`` 排在 ``test_v4_*``
  **之后**，正好落在污染源下游，会以 ``count_draws() == 0`` 直接假红；
* 反过来，按字符序以外的顺序（例如 ``pytest tests/test_m4_eval.py tests/x.py``）
  也能立刻复现。

本文件提供的机制
----------------
1. ``isolated_env``（经 autouse 的 ``_auto_isolate_lottery_paths`` **自动应用**）：
   每个测试**天然**拿到独立的临时 ``BASE / DATA_DIR / RAW_DIR / DB_PATH`` 以及
   ``config`` 上由 DATA_DIR 派生的配置文件路径（``llm_budget.json`` /
   ``methods_config.json`` / ``adaptive_state.json`` / ``llm_config.json``），
   另外同步重定向两个把 DATA_DIR 在 import 时固化下来的模块级常量
   （``llm_lab.CLAIMS_FILE``、``patterns_runtime.SNAPSHOT_FILE``）。
   测试前后都会 ``db.close()`` 重置连接缓存，所以既不会串库，也**不会写入生产
   ``data/``**；结束时按原值恢复，**泄漏为零**（即使用例自己又直接改了 ``config``，
   ``monkeypatch.undo()`` + 显式还原也会把它拉回原值）。
2. ``seeded_db``：在当前临时库里写入合成开奖数据，替代「依赖真实库恰好有数据」。
   新用例**不得**直接赋值 ``config.DB_PATH``，也不得假设生产库内容。
3. ``make_draws``：合成开奖数据工厂（结构同 ``db.load_draws()`` 的元素），供用例自己造库。

约定：``config.DB_PATH = ...`` 这类裸赋值在本目录已全部清理；需要换库时用
``monkeypatch.setattr(config, "DB_PATH", ...)`` + ``db.close()``，或直接用本文件夹具。
"""
from __future__ import annotations

import datetime as _dt
import os as _os
import random
from types import SimpleNamespace

# 必须在任何 lottery.* 模块被 import **之前**设好：
# api_app 在导入期会启动 `boot-rules-refresh` 后台线程（sleep 20 后连 SQLite）。
# pytest 会话会反复 import 该模块，累积的线程与 joblib 多进程后端叠加后
# 实测导致 **Segmentation fault**（2026-10-18，全量跑挂）。
# 因此测试环境统一关掉这个「启动期补刷」，改用显式调用来测试相关逻辑。
_os.environ.setdefault("LOTT_BOOT_REFRESH_RULES", "0")

import pytest

# 需要隔离的 config 属性（全部与"数据落在哪里"有关）
_ISOLATED_CONFIG_ATTRS = (
    "BASE",
    "DATA_DIR",
    "RAW_DIR",
    "DB_PATH",
    "LLM_BUDGET_FILE",
    "METHODS_CONFIG_FILE",
    "ADAPTIVE_STATE_FILE",
    "LLM_CONFIG_FILE",
    # V10-04b：CHANNELS 是 resolve_channels() 的派生快照，config.set_methods() 会刷新它。
    # 不隔离的话，某个用例调过 set_methods 之后，后续用例看到的 CHANNELS 就与
    # 「当前 env + methods_config.json」不一致（test_v10_channels 的
    # "快照 == 现算" 断言会随机红）。
    "CHANNELS",
    # V40-infra（2026-10-25）：LLM 通道六属性也必须隔离。根因证据链：
    # (a) 机制 —— lottery/llm_config_store.py:66-87 的 save() 会**直接赋值**
    #     config.LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_MODEL_LIST /
    #     LLM_SAMPLES / LLM_DISABLED 模块全局，并同时写 os.environ 的
    #     LOTT_LLM_* 五个变量；
    # (b) 复现 —— pytest tests/test_v40_module_coverage_a.py::test_llm_config_read_never_exposes_api_key
    #     tests/test_v40_module_coverage_b.py::test_routes_llm_eval_run_requires_llm_configured
    #     组合跑稳定 1 failed（前测试 save() 后泄漏，单跑两文件各自全绿）；
    # (c) 因果 —— 名单外泄漏 → 同 worker 后续测试 config.llm_configured() 恒 True
    #     → coverage_b 的"未配置⇒400"断言永远拿不到（全量跑稳定假红）。
    # 为什么放 conftest 而不是各文件自带 guard（coverage_a 已自兜底一层）：
    # 文件级 guard 是"每个受害者自己疗伤"，save() 是公共 API，未来任何文件
    # 调它都会踩同一个坑 —— 漏 copy guard 的文件就是下一个假红；conftest 是
    # "堵住伤害源头"，与 isolated_test.sh「靠工具默认值，不靠记得」同一设计
    # 原则。两层恢复的是同一个 import 时真实值，幂等不冲突。
    "LLM_DISABLED",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_MODEL_LIST",
    "LLM_SAMPLES",
)

_MISSING = object()


def _real_config_snapshot():
    """在收集阶段（任何 fixture 生效前）快照 config 的真实取值。"""
    from lottery import config

    return {name: getattr(config, name, _MISSING) for name in _ISOLATED_CONFIG_ATTRS}


_REAL_CONFIG = _real_config_snapshot()


@pytest.fixture
def isolated_env(tmp_path):
    """当前测试专属的临时数据根目录；自动应用到所有测试（见下方 autouse fixture）。"""
    from lottery import config, db
    from lottery import llm_lab, patterns_runtime

    base = tmp_path / "isolated"
    data = base / "data"
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(tmp=tmp_path, base=base, data=data, raw=raw,
                          db_path=data / "ssq.db")

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(config, "BASE", base)
        mp.setattr(config, "DATA_DIR", data)
        mp.setattr(config, "RAW_DIR", raw)
        mp.setattr(config, "DB_PATH", env.db_path)
        mp.setattr(config, "LLM_BUDGET_FILE", data / "llm_budget.json")
        mp.setattr(config, "METHODS_CONFIG_FILE", data / "methods_config.json")
        mp.setattr(config, "ADAPTIVE_STATE_FILE", data / "adaptive_state.json")
        mp.setattr(config, "LLM_CONFIG_FILE", data / "llm_config.json")
        # 这两个模块在 import 时就把 DATA_DIR 固化成了模块常量，必须一起重定向，
        # 否则测试会读写真实的 data/llm_claims.json 与 data/patterns_runtime.json。
        mp.setattr(llm_lab, "CLAIMS_FILE", data / "llm_claims.json")
        mp.setattr(patterns_runtime, "SNAPSHOT_FILE", data / "patterns_runtime.json")
        # 丢弃指向旧（可能是生产）库的连接缓存，保证后续 get_conn() 连到临时库
        db.close()
        yield env
    finally:
        # 先断开临时库连接，再撤销补丁、并显式把真实值写回（双保险：泄漏为零）
        db.close()
        mp.undo()
        # llm_config_store.save() 除了写 config 模块全局，还写了 os.environ 的
        # LOTT_LLM_*（见上方名单注释 (a)）。模块常量已由下面的显式恢复兜住，
        # env 一并 pop 掉：.env 只在 import 时读一次进模块常量，pop 不影响
        # 后续测试的取值路径，防的是「有测试直接读 os.environ」的旁路。
        for _k in ("LOTT_LLM_BASE_URL", "LOTT_LLM_API_KEY", "LOTT_LLM_MODEL",
                   "LOTT_LLM_SAMPLES", "LOTT_LLM_DISABLED"):
            _os.environ.pop(_k, None)
        for name, value in _REAL_CONFIG.items():
            if value is not _MISSING:
                setattr(config, name, value)


@pytest.fixture(autouse=True)
def _auto_isolate_lottery_paths(request):
    """所有用例自动获得独立临时库/数据目录，不依赖测试文件字母序。

    **真实库专用通道（W02）**：设 ``LOTT_TEST_REAL_DB=1`` 时**跳过隔离**，
    让用例直连生产库 ``data/ssq.db``。只给"真值锁"这类必须在真实数据上跑的
    用例使用（例：``tests/test_v12_shape_lock.py`` 的 68.2992% 通过率）。

    为什么需要这条通道：默认隔离会让 `count_draws()==0`，真值锁用例只能 skip，
    于是那个常量**永远不会被真正验证**——"有测试"变成"有测试文件"。
    默认关闭，避免污染常规回归（隔离本身是为修"字母序巧合全绿"才加的）。
    """
    # **登录限流桶是进程级全局状态**，必须在每个用例之间清空。
    #
    # 背景（本轮实测）：`auth._RATE` 是模块级 dict，跨用例累积。
    # `test_mu2_tenancy` 每个用例登录 2 次，跑几个就把
    # `LOGIN_RATE_MAX_PER_IP=10` 的桶灌满 → **排在后面的 test_mu1_auth 全部 429**。
    # 表现为"某个文件单独跑全绿、和另一个文件一起跑就红"，极难排查。
    #
    # 曾经只在 test_w08_user_llm_api 的 fixture 里 reset 过一次 —— 那是**局部修复**，
    # 只保护了当次踩坑的那一个文件。放在 autouse 里才是全局解决：
    # 任何文件、任何顺序都不会互相影响。
    try:
        from lottery import auth
        auth.reset_rate_limits()
    except Exception:      # noqa: BLE001 —— 尚未 import 到也无妨
        pass

    if _os.environ.get("LOTT_TEST_REAL_DB") == "1":
        yield None
        try:
            from lottery import auth as _a
            _a.reset_rate_limits()
        except Exception:  # noqa: BLE001
            pass
        return
    request.getfixturevalue("isolated_env")
    yield None
    try:
        from lottery import auth as _a
        _a.reset_rate_limits()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def make_draws():
    """合成开奖数据工厂：``make_draws(600)`` → 600 期升序、结构同 ``db.load_draws()``。"""

    def _factory(n: int = 600, seed: int = 42, start_year: int = 2013):
        rng = random.Random(seed)
        # 单独一条 RNG 专门打乱 order，**不消耗主 rng 流** —— 否则会改变后续期次的
        # reds/blue 抽样序列（连带 test_v14_engine_split 等指纹锁全变）。
        orng = random.Random(seed ^ 0x5A5A5A)
        start = _dt.date(start_year, 1, 1)
        draws = []
        for i in range(n):
            reds = sorted(rng.sample(range(1, 34), 6))
            # ``order`` 是**开奖顺序**（第 1..6 位），真实数据里它并非升序 —— 升序只是
            # ``reds`` 的排序结果。V15 的 order_pos 维度会真实检验「位置是否携带号码信息」，
            # 若这里仍填 sorted(reds)，等于人造了「位置 1 恒为最小值」的强位置偏置，
            # order_pos 会正确地把这个 fixture 假象当成显著效应检出（实测 z≈23）。
            # 故这里给 order 一个随机排列（与真实开奖一致：位置与号码大小无关）。
            order = list(reds)
            orng.shuffle(order)
            draws.append({
                "issue": f"{start_year + i // 150}{i % 150 + 1:03d}",
                "date": (start + _dt.timedelta(days=3 * i)).isoformat(),
                "reds": reds,
                "order": order,
                "blue": rng.randint(1, 16),
            })
        return draws

    return _factory


@pytest.fixture
def seeded_db(isolated_env, make_draws):
    """把合成开奖数据写进当前临时库；返回写入的 draw 列表。"""
    from lottery import db

    def _seed(n: int = 600, seed: int = 42):
        draws = make_draws(n=n, seed=seed)
        db.upsert_draws(draws)
        return draws

    return _seed
