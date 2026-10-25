"""全局配置：路径、多尺度窗口、LLM 通道（OpenAI 兼容，多模型可自定义）。

安全约定：仓库中不保存任何 LLM API URL / Key，全部通过环境变量注入。
配置方式见 .env.example（Docker Compose 自动读取 .env；本地运行请先 source）。
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

BASE = Path(os.environ.get("LOTT_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("LOTT_DB", str(DATA_DIR / "ssq.db")))

# ---------------------------------------------------------------------------
# 生产库路径护栏（2026-10-21）
#
# ## 为什么要有这个
#
# `os.environ.get(K, default)` 的行为是：**变量名拼错不报错，静默回落到默认值**，
# 而本项目的默认值恰好是**生产库**。于是：
#
#   LOTT_DATA_DIR=/tmp/xx .venv/bin/python -c "..."   # 该变量不存在，被忽略
#   → 所有 TestClient 写入 静默落到 ssqyuce/data/ssq.db
#
# 这不是假设：同一天内两个不同的人独立踩了同一个坑。
#   * 李工：设 `LOTT_DATA_DIR` → 造出 tester1；
#   * 高架构：设 `LOTT_DATA_DIR` → 造出 probe_admin 等 6 个（含 2 个 admin）。
# 两次都"以为隔离了"，实际写了生产库，且**没有任何报错**。
#
# ## 护栏做什么
#
# 在**测试上下文**（pytest 运行中，或显式 `LOTT_REQUIRE_ISOLATION=1`）里，
# 若实际生效的 `DB_PATH` 就是生产库，**直接抛异常中止**，把"静默写生产库"
# 变成"一次响亮的失败"。
#
# 生产运行不受影响（无 pytest 痕迹、未设该 flag，护栏不触发）；
# 真实库专用通道 `LOTT_TEST_REAL_DB=1`（真值锁用例）显式豁免。
# ---------------------------------------------------------------------------

#: 生产库规范路径（本文件所在包的上一级 = ssqyuce 项目根）
PROD_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ssq.db"


def is_test_context() -> bool:
    """当前是否处于测试上下文。判据取"存在即算"，宁可多拦一次也不漏。"""
    if os.environ.get("LOTT_REQUIRE_ISOLATION") == "1":
        return True
    return any(os.environ.get(k) for k in
               ("PYTEST_CURRENT_TEST", "PYTEST_VERSION", "PYTEST_ADDOPTS"))


def guard_prod_db(*, allow_real: bool = False) -> None:
    """测试上下文里若 DB_PATH 指向生产库，直接 raise。

    `allow_real=True` 与 `LOTT_TEST_REAL_DB=1` 供真实库专用通道使用。
    """
    if allow_real or os.environ.get("LOTT_TEST_REAL_DB") == "1":
        return
    if not is_test_context():
        return
    try:
        same = DB_PATH.resolve() == PROD_DB_PATH.resolve()
    except OSError:          # 路径不可解析时按"非生产库"处理，避免误杀
        return
    if same:
        raise RuntimeError(
            "测试上下文里检测到 DB_PATH 指向生产库，已中止。\n"
            f"  实际生效路径: {DB_PATH}\n"
            f"  生产库路径  : {PROD_DB_PATH}\n"
            "常见原因（本项目一天内踩过两次）：\n"
            "  * 设了不存在的变量名 `LOTT_DATA_DIR` —— 它不报错，会被静默忽略；\n"
            "  * 真正生效的是 `LOTT_DB`（数据库文件）与 `LOTT_HOME`（数据根）；\n"
            "  * 或代码里直接给 config.DB_PATH 赋了值。\n"
            "正确做法：用 scripts/isolated_test.sh，或显式设 LOTT_DB=/tmp/<uniq>.db 后\n"
            "**打印实际生效路径**（不要只凭『我设了变量』就认定隔离成功）。\n"
            "确需在真实库上跑（真值锁用例）时，显式设 LOTT_TEST_REAL_DB=1。"
        )

DATA_URL = "http://e.17500.cn/getData/ssq.TXT"
STARTED_AT = time.time()
# M4.5：配置后启用所有 /api/* 的 Bearer Token 校验；空值保持兼容开放。
API_TOKEN = (os.environ.get("LOTT_TOKEN") or "").strip() or None

# ---------- 多用户身份（MU-1）----------
# 系统主体哨兵（Lead 裁决 3）：调度器/后台任务等"非人类主体"写入数据时用它归属。
#
# 为什么是 0 而不是 NULL：
#   `NULL` 在 SQL 三值逻辑里与任何值比较都不相等，`WHERE user_id != ?` 会**漏掉**
#   系统任务行；也无法回答"这是谁的任务"。0 是可查询、可审计、可显式过滤的哨兵。
#
# 为什么必须定义在这里而不是各文件写字面量：
#   调度器、tasks 默认值、predictions 回填兜底三处共用它；分散写 0 将来改值会漏改。
#   **0 不插入 users 表**（它不是人）。
SYSTEM_USER_ID = 0

# 鉴权执行模式（W05）。`off` = 仅解析身份、不拦截（灰度过期用）；
# `enforce` = deny-by-default，未登录一律 401。
# 默认 off 的原因：前端登录界面尚未落地（W08），此时强制拦截会让线上不可用。
# **切换前必须先完成前端登录页 + 引导建号**，否则整站会 401。
AUTH_MODE = (os.environ.get("LOTT_AUTH_MODE") or "off").strip().lower()


def auth_enforced() -> bool:
    """鉴权是否处于强制拦截状态。"""
    return AUTH_MODE == "enforce"


# 会话 Cookie 名。`__Host-` 前缀是浏览器强制的加固：必须 Secure + Path=/ + 无 Domain，
# 因此子域无法覆盖它（防"子域写入同名 Cookie"的会话固定攻击）。
# 本地纯 http 调试时浏览器不存 Secure Cookie，可用 LOTT_COOKIE_NAME=ssq_sid 临时降级。
_cookie_name_raw = (os.environ.get("LOTT_COOKIE_NAME") or "__Host-ssq_sid").strip()
# Cookie 的 Secure 属性。生产（nginx 443）必须 True；本地 http 联调可设 0。
SESSION_COOKIE_SECURE = (os.environ.get("LOTT_COOKIE_SECURE") or "1").strip() == "1"

# V38（2026-10-24 审计 P2）：`__Host-` 前缀与 Secure=0 是**互斥**组合，
# 浏览器会**静默丢弃**这种 cookie —— 表现为"登录接口返回 200，但下一个请求
# 仍是未登录"，而且控制台没有任何报错，是最难排查的一类配置错误。
#
# 曾有人按注释"本地 http 联调可设 0"操作，忘了同时改 Cookie 名，于是：
#   Set-Cookie: __Host-ssq_sid=...; Secure=缺  → 浏览器拒收
#   → 登录"成功"但永远进不去。
#
# 现在自动纠正：Secure=0 时把 `__Host-` 前缀去掉（保留 ssq_sid 语义），
# 并在启动日志里明说一次，避免"我明明配了却没生效"。
if not SESSION_COOKIE_SECURE and _cookie_name_raw.startswith("__Host-"):
    SESSION_COOKIE_NAME = _cookie_name_raw[len("__Host-"):]
    COOKIE_NAME_AUTO_ADJUSTED = True
else:
    SESSION_COOKIE_NAME = _cookie_name_raw
    COOKIE_NAME_AUTO_ADJUSTED = False


# ---------- W01（多用户改造 P0 部署前置）----------
# 文档端点（/docs、/redoc、/openapi.json）默认**关闭**。
# 为什么必须默认关：这三条路径**不在 `/api/` 前缀下**，天然绕过 `optional_api_auth`
# 中间件 —— 也就是说"给 /api/* 加鉴权"完全保护不到它们。生产实测
# `http://<ip>:18000/docs` 返回 200，等于把全部路由结构、参数 schema 对外公开。
# 本地开发需要时显式 `LOTT_API_DOCS=1` 打开。
API_DOCS_ENABLED = (os.environ.get("LOTT_API_DOCS") or "0").strip() == "1"

# CORS 允许来源（逗号分隔）。默认只允许本站与本地开发端口。
# 为什么收紧要保留：前端 `web/static/app.js` 用的是**相对路径** fetch，属同源请求，
# 本不需要 CORS 头；`allow_origins=["*"]` 只会让任意第三方页面拿到跨域读取能力。
# 需要跨域接入时用 `LOTT_CORS_ORIGINS` 显式追加，不再回落到 `*`。
_DEFAULT_CORS_ORIGINS = "https://ssq.ff88.de,http://127.0.0.1:18000,http://localhost:18000"
CORS_ORIGINS = [
    o.strip() for o in
    (os.environ.get("LOTT_CORS_ORIGINS") or _DEFAULT_CORS_ORIGINS).split(",")
    if o.strip()
]
# M4.4：可选备用源，需返回与 ssq.TXT 相同的 31 字段格式；空值则只运行主源。
BACKUP_DATA_URL = (os.environ.get("LOTT_BACKUP_DATA_URL") or "").strip() or None

# ---------- 版本信息（前端主页 / API / GitHub 说明统一引用） ----------

APP_VERSION = "4.20.0"         # v40：审计剩余债务收尾（AST化/合并/死代码/测试基建）

# 单注 2 元（双色球官方单价，固定事实）。
# 放在 config 而不是某个路由模块里：它被 evaluate / portfolio / rigor 共用，
# 属于领域常量，不属于任何单个端点域。
TICKET_PRICE = 2.0
APP_BUILD = "2026-10-V40"      # 构建标识（V40：审计剩余债务收尾批次）
APP_MILESTONES = {
    "V40": {"status": "done", "desc": "按 2026-10-24 审计剩余债务收尾（三泳道并发：lead + 测试基建 + 定向补测，含只读侦察）。审计 40 条中最后 15 条债务的处理结果：①**『三/五份重复统计实现』实为 BH×3 + Wilson×4**（侦察还原审计原文：SUMMARY 压缩表述丢了主语，walk-forward 的 11 个实现是 2 薄包装 + 9 个本质不同算法，不能硬合并）；真重复部分已合并——`stats/rigor` 新增 wilson_lower_bound/wilson_interval 纯 stdlib 真源，backtest.bh_adjust/_wilson_ci、llm_eval._bh_adjust、hypothesis._wilson_lower、decision_skill._wilson_lower、rule_lifecycle.wilson_interval 六副本全部薄壳化，数值逐位对拍（BH max diff=0.0；Wilson ≤1e-16 浮点尾数）。**变异测试抓出真窟窿**：合并后对拍测试恒绿（两边同真源），且『漏除 denom』的经典错法既有测试全部放过（容差 2e-3 太宽）——补外部参考值真值锁（容差 1e-6，4 点，statsmodels 同公式），变异实证：真源绿/变异红。②**死代码清扫**：全仓零引用扫描（排除路由装饰器/re-export）确认 16 个死函数，删 13 个（_fmt_list/load_local/_resolve_all/active_pattern_keys/_latest_audit_digest/window_features/rolling_series/last_unfit_reason/_family_leaf/expected_counts/llm_discovery_candidates/_prize_rank/dumb_judge_baseline）；**保留 3 个有据可查的**：rule_protocol.register_family/get_family（M2 设计文档明确『纯类型+注册表，无行为，不接线』是故意设计）、discovery._features（保守）。删后 pyflakes 计数与干净 HEAD 逐位一致（205=205），零孤儿导入。③**Python 源码子串断言 AST 化**：新建 tests/_src_ast.py（identifiers_of/module_has_dict_key 等，含 keyword 实参形态——实测踩到 ast.keyword 不是 ast.arg 的假绿），改造 4 组断言（naming_discipline×2、stats_package×2，含『8 个 bad not in src 扫荡→AST 查 import/属性/调用』），B_BASELINE 141→139 同步下调（下调本身即改造证据）；双向变异实证：注释里留字样不再假绿/假红。④**shortfall_reason 语义空转修复**：后端 engine.py:943-949 认真区分三种原因（llm_only_actual_count/candidate_pool_exhausted/all_methods_disabled），前端却只渲染『5 / 请求 10』——补 _shortTxt 文案映射（含『按 LLM 实际产出为准，非缺陷』），测试断言三种 reason 都有可读文案且真的进入 items.push（反空转锁）。⑤**CPU_LIMIT_PCT 口径文档补正**：V39 把 _system_cpu_pct 分母改成容器 cgroup 配额（2.4 核）后，常量注释仍写『整机 60%=2.4 核』——同一个 60 在两口径下绝对核数差 40%。重写注释为『分母=挖掘进程可用配额』并如实记录一个故意保留的保守偏差（one_core 用宿主核数 25 点 vs 配额口径 41.7 点，方向偏保守安全）；测试锁注释块必须含 cgroup/配额/分母三词（变异实证：第一版窗口太宽被『文件别处提到 cgroup』骗过，收紧到紧邻注释块后变异变红）。⑥**测试基建（test-speed 泳道）**：pytest-xdist 落地——全量 -n4 三连跑 265/277/276s 全绿零 flaky（干净树串行约 400-460s）；v31 神经穷尽 9 测试占全量 77% 单文件，标 @pytest.mark.slow 后 fast 子集 253s 全绿；pytest.ini 注册 markers 且 addopts 无 -m（分层只分组不跳过，收集守恒 2031=2022+9）；scripts/test_fast.sh 快速入口仍走 isolated_test.sh 生产库护栏；3 个元测试锁基础设施。**系统性根因发现（最有价值的副产品）**：conftest 隔离名单缺 LLM 通道六属性（llm_config_store.save() 直写 config.LLM_BASE_URL/API_KEY/MODEL 模块全局+os.environ，同 worker 后续测试 llm_configured() 恒 True）——扩名单+env pop 修复，注释含完整证据链。⑦**定向补测（coverage 泳道）**：AST 全扫确认 23 个模块零直测，按『有计算逻辑补真值锁、纯透传不补』原则补 13 个（engine_blue/solver/orchestrate/rules/constraints/jsonutil/diagnose/llm_config_store/data_fetcher/data_check/cli/routes_ml/routes_llm_eval/routes_patterns），新增 72 测试；**23 处变异全部抓红**，过程中抓出并重写自己 3 处假绿；明确不补 10 个并逐条说明理由（纯常量/facade 已触达/HTTP 级已命中）。⑧**cli 分发真 bug**：`lottery online_check`（--help 显示的命令名）因函数叫 cmd_online 而 main 按 cmd_<name> 找 ⇒ 永远『未知命令』——加别名 cmd_online_check=cmd_online 修复，coverage 泳道的存量缺陷锁按其自我指示翻转为修复锁（并修掉该锁自身两处顺序敏感缺陷：日志前缀、非空库明细行的 Extra data）。⑨**错误返回形态审计项按侦察结论降级**：AST 全扫 2317 条 return 证实 HTTP 错误信封只有 1 种（117 处 ok+error 100% 覆盖，HTTPException 零处），41 处裸 {'ok':False} 全是内部函数约定——审计口径污染；真问题仅 12 处 200+ok:false 与 6 处 detail 键重复，登记为后续小项不属本轮。测试 1919 → 2031（+112：debt_wrapup 11 + wilson 真值锁 1 + coverage 72 + infra 3 + cli 修复锁翻转与既有归位等；全量 -n4 265-277s 全绿零 flaky / fast 253s 全绿）。"},
    "V39": {"status": "done", "desc": "按 2026-10-24 审计 P2 收尾批次（口径默认 / 边界校验 / 容器限流）。本批处理的是『边界与口径』类问题——不改变主要行为，但在特定输入下给出『看起来合理、实际错误』的结果，特征是沉默：无异常、无日志。①**`sixth` 默认口径 published → strict**（engine_distribution._single_ticket_ev）：published 把六等奖按『蓝球命中』的边际概率算，与三/四/五等奖重复计数（奖级概率合计 0.0707 vs 官方 0.0671，EV 高估 2.01%）。生产路由早已显式传 strict，但**默认值**仍是 published —— 任何忘传参的新调用方都会静默拿到错误数。改默认 strict，published 保留仅作并排对照。**自纠**：第一版断言写成『published 概率合计 > 1』，实际 by_level_prob 是奖级概率（量级 0.07）不是归一化分布，合计本来就 <1 —— 正确判据是 published 合计 > strict 合计。②**`sigma_hypergeom` 边界校验**：旧实现把负值钳 0、把 k>R_MAX 用 max(fpc,0) 变 0，于是 k=-5 与 k=50 都『安全地』返回 0.0——看似正常，实为调用方传错被静默吞掉。k 超总数通常说明筛选集构造有 bug。现对 k<0 或 >上限（红 33/蓝 16）抛 ValueError；k=33 返回 0 是**数学正确**（全集必中 6 个，方差 0），专门测试钉住防后人改错。③**`_system_cpu_pct` 容器感知**：容器内 /proc/stat 显示的是**宿主整机** CPU（docker exec 实测），共享宿主上会把别人容器的占用当成自己的——别容器跑满时本容器误判『整机满』过度限流，本容器吃满而宿主空闲时又不限流。现优先读本容器 cgroup：`/sys/fs/cgroup/cpu.stat` 的 usage_usec 差分 ÷ 配额核数（cpu.max）。**关键守卫**：只有 quota 与 usage 都能读到时才用容器口径——非容器环境 cpu.stat 可能读到宿主的累计值，若 quota 不存在就回退 /proc/stat，不改变非容器行为。测试 1903 → 1919（新增 tests/test_v39_p2_wrapup.py 15 例；全量 1919 passed / 24 skipped / 1 xfailed，406s；变异实证 3/3 抓红：sixth 改回 published 被行为锁+元守卫双抓、撤 sigma 校验 6 抓红、撤容器口径 1 抓红）。"},
    "V38": {"status": "done", "desc": "按 2026-10-24 审计修复 P2 批次（7 条 + 路由清单守卫）。①**P2 事务泄漏**（db.save_pattern_results）：Python sqlite3 在首条 DML 前隐式开事务，旧实现循环后 commit、中途异常既不 commit 也不 rollback ⇒ 该线程 thread-local 连接上留着**开着的写事务持锁**，下一次 commit 会把这一批半成品连同后续写入一起提交（部分数据永久落库且无人知）。修法：整批 SAVEPOINT + 失败 ROLLBACK/RELEASE + 入口字段校验（缺字段在写任何数据前抛明确 ValueError）。**过程中抓到自己写的假绿**：第一版用例传「缺字段」数据，但入口校验在执行前就拦下、根本走不到 savepoint —— 把回滚整段删掉用例依然 3 passed。改为制造**执行期**失败（params 里放 object()，过得了校验但 json.dumps 时 TypeError），变异随即变红。这条教训值得记：测「异常处理」必须让异常真的发生在被处理的那一层。②**P2 返回值语义**（upsert_draws）：返回的是遍历计数，而 INSERT OR REPLACE 每次都成功 ⇒ 重复抓同一份数据会报『新增 3518 期』而实际零变化，直接误导『数据是否落后』的判断。新增 upsert_draws_detailed 用一次 SELECT 区分 inserted/updated，data_fetcher 改用该口径并新增 updated_existing 字段。③**P2 cookie 配置陷阱**：`__Host-` 前缀要求 Secure，而 `LOTT_COOKIE_SECURE=0` 与默认 cookie 名是**互斥组合**——浏览器**静默丢弃**该 cookie，表现为『登录 200 但下个请求仍未登录』且控制台无任何报错，是最难排查的一类。现自动去前缀（Secure=0 时用 ssq_sid）并置 COOKIE_NAME_AUTO_ADJUSTED 标记。④**P2 Brier 分解接口误用**：Murphy 恒等式前提是 y∈{0,1}；传非二值（如 0.5）时恒等式破裂（实测 y=[0.5] 时 brier=0 而 rel−res+unc=0.25，gap=−0.25）——一个看起来『分数很好』的结果其实是误用产物。加二值校验，二值输入下恒等式精确到 1e-16。⑤**P2 randomness_report 顶层 verdict 会被误读**：它是**单项未校正**口径，实测同一份数据 verdict=deviation 而 verdict_fwer=consistent-with-random（N=3517 时纯随机也有 37% 概率出现至少一次假阳性）。保留原语义（向后兼容）但新增 verdict_is_uncorrected / verdict_authoritative_field / verdict_note_zh 三个标注字段，指明做整体结论必须读 verdict_fwer。⑥**P2 LLM 配置缺字段→500**：用户配置是任意 JSON，漏 model/base_url/api_key 任一项会 KeyError 被兜成 500『预测引擎异常: KeyError』，用户不知道要改什么。改为给可执行错误（指出缺哪项 + 指向设置页 + 提供改用统计模式的出路），并记 unsafe/不完整到账单。⑦**解析器容错**：parse_text 任一行坏数据即整批作废（上游掺一行 HTML 报错页就让几千期全废，表现为『刷新后一期都没有』），改为默认跳过并回报坏行、strict=True 保留旧语义、**全坏行仍抛错**（那更像源改版）。⑧**路由清单守卫**（新 tests/test_v38_route_inventory.py）：用 app.openapi()['paths']（曾误用 app.routes，当前 FastAPI 把 include_router 包成 _IncludedRouter 且 path=None ⇒ 20 个对象一个都取不到路径，『看起来在检查、实际零覆盖』）对撞前端 fetch 的静态路径；另加核心端点存在性检查。测试 1884 → 1903（新增 19 例；变异实证 M2/M3/M4 全抓红）。"},
    "V37": {"status": "done", "desc": "按 2026-10-24 审计修复安全与工程类 P1（9 条）。①**P1-1 登录阻塞事件循环**：`auth_login` 是 async def 却在协程体内同步跑 scrypt（32MB / 145~550ms），实测冻结整站，且登录**匿名可达**。`_hash_sem`（最并发 4）只限内存峰值、不限事件循环阻塞 —— 正确但不相干。修法：新增 `_offload()`（默认 executor，与同步路由共用同一个池，不另建第二套），把 6 处 scrypt 调用（登录校验、dummy 探针哈希、改密、引导建号、管理员重置、管理员建号）全部挪出 loop。A/B 实测（真 uvicorn + /api/auth/me 心跳）：修前最大断档 282ms、修后 130ms（基线 p95 96ms），差值恰为一次 scrypt。②**P1-2 认证后 SSRF + 响应体回显**：用户可写任意 base_url，服务器主动 POST，错误路径把 `HTTP {code}: {r.text[:200]}` 经 503 回传 ⇒ 盲打变可读响应。新增 `lottery/url_guard.py`：拒 file/gopher 等协议、拒字面量内网与保留地址（含 100.64/10 CGNAT、169.254.169.254、IPv4-mapped IPv6）、拒云元数据主机名、域名解析任一 IP 内网即拒；**两处校验**（保存侧给即时反馈、`chat()` 发请求前是真实边界）；新增 `_redact_echo()` 只保留 JSON 错误 message，其余折叠。过程中修正了自己的一次过度 fail-closed：**解析失败不等于不安全** —— 第一版把 DNS 查询失败也判拒绝，导致无 DNS 环境下 `https://a.example.com` 保存直接 400（CI 里看起来像产品坏了），改为『能确认内网才拒，无法确认放行给下一道闸』。③**P1-3 降权不失效会话**：改 role 未撤销会话，而 require_csrf/`/api/auth/me`/_require_admin 读的是缓存快照（TTL 30s）⇒ 被降权者 ≤30s 内旧会话仍写着 role=admin。补 `revoke_user_sessions + invalidate_session_cache`，与 set_status 同口径（此前只有它在撤，口径不一致）。④**P1-4 文档端点泄露存在性**：`/docs` `/openapi.json` `/redoc` 关闭时被鉴权层判成 user → 401（=告诉扫描器』存在但没登录』），改由 `DOCS_PATHS + _docs_disabled()` 放行到路由层自然 404。⑤**P1-9 配置三方漂移**：`LOTT_SCHEDULER` 代码默认 0 / compose 默认 1 ⇒ 『python 直接跑』与』compose 跑』是两个系统；统一为 1；`.env.example` **整段缺** W08 鉴权键（AUTH_MODE/MASTER_KEY/BOOTSTRAP_TOKEN/SESSION_TTL/COOKIE_*），按 example 部署会得到『看起来配好、实际没鉴权』的实例，已补齐并加 LOTT_LLM_ALLOW_PRIVATE_URL / LOTT_PREDICT_SEED 说明。⑥**P1-10 后台线程连接泄漏**：`_bg_run` 每个任务起一次性线程，thread-local 连接从不关闭，滞留 fd 并拖慢 WAL checkpoint；finally 里 `db.close()`（语义正是』关当前线程连接』，安全）。⑦**P1-13 重计算端点无并发保护**：此前只有登录限流。新增 `lottery/concurrency.py` 信号量闸门（**限在飞数量而非速率** —— 速率限制挡不住『每 10 秒发 20 个并发』，而 CPU 是被在飞数量吃满的），按领域分桶、拿不到立即 503+Retry-After（不排队，排队会让全部超时）；探活/静态/鉴权不在名单（healthcheck 必须永远可达）。⑧**P1-11 依赖方向守卫只守一条**：原测试只禁』底层 import api_app『，`evaluate→engine` 等裸奔，制造』依赖已受控』错觉。改为按 L0~L4 **分层表**逐模块检查上行依赖，并加两条元守卫（分层表必须覆盖全部模块，否则新模块静默逃过；KNOWN_UPWARD 白名单不许留过期条目）。分层按**实测依赖结构**校准而非按名字猜：llm_client/methods/features/models 被十余个领域模块引用，标成上层会报十几个假违规 —— 那描述的不是设计问题而是分组错。登记 4 条真实债务：evaluate→engine、battle→engine、db→evaluate（结算）、llm_client→user_llm_store（记账）。⑨**P1-14 LLM 失败期污染配对检验**：llm_eval 在 LLM 失败期用 stat 输出顶替并计入 stat_llm，差值恒 0 ⇒ 配对均值被往 0 拉、p 值往『无差异』推，系统性掩盖 LLM 劣绩；改为整对剔除失败期，并新增 `paired_issues_excluding_llm_failures` 披露实际参与配对的期数。测试 1848 → 1884（新增 tests/test_v37_p1_audit_fixes.py 33 例；全量 1884 passed / 24 skipped / 1 xfailed，444s；变异实证：分层守卫注入 L1→L4 违规能抓红）。"},
    "V36": {"status": "done", "desc": "按 2026-10-24 四岗位并发审计（架构/数学/代码/测试）修复 5 条 P0。审计报告 docs/reviews/AUDIT_2026-10-24_SUMMARY.md，共 40 条缺陷（P0 5 / P1 17 / P2 18），本批次只修 P0 与不可分割的口径一致性。①**P0-1 六等奖判定错**（engine_prize.py:93）：`r==3 且蓝不中`（3红+0蓝，官方不中奖）被计入六等奖，六等奖概率虚增为 1/9（官方 1/17），单注 EV 高估 24.6%（¥1.2531 vs ¥1.0055）；同仓 evaluate.prize_level 判对了却无人对拍，而该处注释恰恰写着「与 evaluate.prize_level 完全一致」——教训：**注释不能当校验**。②**P0-2 跨租户 IDOR**（routes_tasks.py:44）：`_task_visible_to_current_user` 恒 True，`/api/tasks/{id}` 只要求 user 角色（ADMIN_EXACT 按精确路径匹配，带 id 的路径落到兜底），任意登录用户可读他人 result_json 票单；已 HTTP 实测复现（uid B 读 uid A → 200），修后 404。③**P0-3 /api/predict 无 CSRF**：GET+POST 同路由且 POST 会写库，enforce 下错误/缺失 token 均 200 并落库，而对照写端点 /api/user/llm/config 返回 403；现 POST 在 enforce 下强制 X-CSRF-Token，GET 保持只读不强制（否则打断所有只读前端调用）。④**P0-4 predict_next 非确定性**（engine.py:767）：`random.Random()` 未播种，同进程同输入两次票号不同（指纹 1be697ea vs ad1f33f1，差异 33 处）；更严重的是 evaluate.offline_backtest 的 seed 只喂给随机基线、没喂给引擎，导致同数据同 seed 两次回测 delta_red_hits = -0.0400 / -0.1133，跨过 hypothesis 判决阈值 |delta|≥0.05 ⇒ **判决可翻转**；现 config.PREDICT_SEED 固定种子 + 回测按每期派生独立子种子透传。⑤**P0-5 无会话请求冒认主管理员**（db.py:539）：AUTH_MODE=off 时中间件放行且不写 contextvar，resolve_tenant 兜底返回主管理员 uid ⇒ 匿名 POST /api/predict?regenerate=1 删改管理员票单；现新增 request_context.in_request 标记区分「无请求的后台线程」与「无身份的 HTTP 请求」，后者归 SYSTEM_USER_ID；compose 默认 LOTT_AUTH_MODE 由 off 改 enforce。配套修：⑥P1-5 sixth_ev_ratio 用 p_cover（≥1中蓝概率）而非期望命中注数 n/16，n=20/100 时低至 0.125/0.025，与 blue_coverage.py 2026-10-21 已修的是同一 bug 未同步；⑦P1-6 六等奖概率真源统一为 engine_prize.P_SIXTH（= P(红≤2)/16 = 0.0588925466），EV_RATIO_SIXTH 由 (5/16)/2=0.15625 修正为 0.1472314，blue_coverage/engine_distribution/engine_prize 三处引用同一常量——**审计报告里我最初写的 0.294463 漏除了 2 元成本，实现时已纠正并在注释留痕**；⑧P1-12 调度器断链：api_app.py:344 读 draws 而 :388 才赋值 ⇒ 必抛 NameError 被 except 吞掉，前置注册写入侧从未执行（而上方注释正宣告「这处断链已接通」），现就地取数；⑨P1-8 n_tickets 加 ge=1/le=100，此前 -3 → 200 + 0 注 + shortfall_reason=null。测试 1825 → 1848（全量 1848 passed / 24 skipped / 1 xfailed，耗时 399s；审计前基线是 1823 passed + 1 failed，耗时 1259s —— 修掉未播种 RNG 后指纹真值锁首次稳定，耗时也降了 68%）。"},
    "V35": {"status": "done", "desc": "移除「随机性检验」卡与「数据管理」页（用户 2026-10-24：「删除 主页上的『随机性检验』和『更多』里面的数据管理」）。删除内容：①#rigorCard「🎲 随机性检验」（V4）+ 四个渲染函数 loadRigor/renderRigorCard/renderRigorUnavailable/renderRigorSignal（共约 3,500 字符）；②#tab-data「⚙️ 数据管理」+ 抽屉入口 + 三个动作函数 refreshData/showTasks/showStats；③switchTab 的 VALID_TABS 由 [predict,eval,data,settings] 收为 [predict,eval,settings]。关键处理——**页脚声明保留但去掉跳转**：原 footer 有 `jumpToCard('rigorCard','predict')`，卡片删掉后点击无反应；改为 `<span class=\"footer-link\">随机性诚实声明</span>`（文字保留）。理由写进注释：删卡片 ≠ 删声明，声明是对用户的信息义务，不依附于某个面板；原注释「压成四个字等于没说」同理适用于整句删除。移除的是展示不是能力：后端 /api/rigor/randomness、/api/rigor/signal、/api/refresh **全部保留注册**，test_v35 明确断言这些端点必须仍在。踩到并修复的坑：①按大括号配对切函数时 `async ` 前缀留在原地 → `async async function`；连续删三次后累积成 `async async async`。先用正则批量清理，结果误删了合法函数的 async（`await` 报错），**立即从备份恢复**并改用「按行 + 括号配对」的保守切法，一次成功。这条教训已写进 commit：字符串/正则批量改 JS 前必须先 node --check，且不能对 async 这类修饰符做全局替换；②V29 的结构快照测试因 data 页移除而失效，已同步更新期望值。测试 1820 → 1825（新增 5 例，变异实证 2/2 抓红：卡片加回 HTML、footer 跳转加回）。"},
    "V34": {"status": "done", "desc": "主页移除三张卡（用户 2026-10-24 两轮指示：「去掉主页的『本期规律使用情况』，既然无法找到规律，挖掘就没有意义了」+「去掉主页的『前向回测』和『方差优化选号』」）。删除内容：①#ruleCard「🧩 本期规律使用情况」（V7）+ renderRuleUsage() 函数 3219 字符及其调用；②#walkForwardCard「🎯 前向回测」（V23）+ loadWalkForward()；③#variancePickCard「📐 方差优化选号」（V27）+ loadVariancePick() 及 vpRun/vpN 绑定。关键设计——**移除的是展示，不是审计能力**：后端 /api/rigor/walk_forward 与 /api/portfolio/variance-pick **保留注册且契约不变**（需要时仍可手动调用），测试 test_v34 明确断言这两条路由必须仍在，防止后续借'删前端'顺手删掉审计工具。踩到并修复的坑：①用大括号配对法切函数时，把 `async ` 前缀留在原地，导致 `async async function loadRigor` 语法错误 —— 已清理并以 node --check 验证；②renderRuleUsage 还有第二处调用（renderPredictions 内），首轮漏删 → ReferenceError 风险，已一并移除；③runClosedLoop 的 renderArea 回退到已删除的 #ruleBody，改为回退到 #rgLabBody（规律研究页实验区），无该区时传 null（runBackgroundTask 已支持）；④测试配额：首版 test_v34 用了 3 处 unsourced 文件读取断言 + test_v23 改写后从 1 增至 4，把 B_BASELINE 顶到 149；按项目纪律把 v34 结构压成**单个快照断言**、v23 四处合并为一处，最终降到 145，**同步下调 B_BASELINE 146 → 145**（下调本身即改造进度证据）。测试 1816 → 1820（v34 新增 4 例；v23 的 test_frontend_walk_forward_card_and_rendering 改写为 test_frontend_walk_forward_card_removed，语义反转）；变异实证 2/2 抓红（卡片加回 HTML、删除后端路由）。另修复：renderPredMeta 里仍有 2 处裸读 `r.usable`（已弃用别名），与 V14-A4 命名纪律冲突 —— 改为读 `usable_replayable`，并把标签从「可用」改成「可回放」，让 test_v14_naming_discipline 两条守卫继续有效。"},
    "V33": {"status": "done", "desc": "策略可区分性判据 —— 从「能不能预测」转向「什么条件下测得出来」（用户 2026-10-24：「抛开『能不能预测』，而是『在什么条件下，一个策略的表现在统计上能与其随机基线区分开』」）。V30/V31/V32 的局限是只对已测过的东西成立，无法指导任何后续决策；V33 给出**封闭公式 + 设计表**。数学基础（全恒等式）：H~Hypergeom(33,6,6)，E[H]=1.0909，Var[H]=0.7531；E[T]=n·1.0909；Var[T]=n·Var[H]·var_ratio；检出所需期数 P=((z_α+z_β)·sd(T)/(δ·n·E[H]))²。核心发现：①**效应量门槛**：≥20% 需 10 期(0.1年)可检出；10% 需 39 期(0.3年)；5% 需 156 期(1.0年)；2% 需 978 期(6.5年)；1.6%（V30 实测的超额）需 1,528 期(10.2年)；1% 需 3,913 期(26.1年)。**V30 那个 +1.61% 从一开始就不可能被证实——它低于现有 3518 期的检出下限 1.05%**，这独立佐证了 V30『它是估计偏差』的结论。②**反向能力边界**：3518 期×每期 10 注×独立票 → 最小可检出效应 1.05%（1 注 3.33%、5 注 1.49%、20 注 0.75%、50 注 0.47%）。任何预期效应低于此值的策略，在本系统上**永远无法被证伪**——不是『测出不显著』，而是『没有能力测』。③**var_ratio 是唯一人为杠杆（V27 的真正价值）**：var_ratio 1.0→0.26 使 sd 从 2.744 降到 1.399（−49%），**等价样本量放大 3.85 倍**：检出 1% 需期数 3,913→1,017；同样 3518 期可检出效应 1.05%→0.54%。V27 当时只说『不提高期望』，其实还有另一半：**方差优化提高了检验力**——一个 2% 的弱效应用独立票永远测不出，用最小方差票有可能测出。④**预算分配**：总注数固定时，注数/期 × 期数的各种组合给出**相同**可检出效应（互相抵消）；但若注数固定，观察 3518 期比 150 期可检出效应小 4.9 倍 → 优先延长观察期，其次降 var_ratio，最后才加注数。新增 GET /api/rigor/detectability：四块同屏（forward 正向期数/年数+三档结论句、reverse 反向能力边界、var_ratio_leverage 等价样本倍数、design_table 不同注数的可检出效应）；形参做 Query 归一（直接调用不 500）。报告 docs/analysis/V33_DETECTABILITY.md。测试 1806 → 1816（新增 10 例，3 组变异实证全部抓红；过程中发现并修复了一个真实测试缺陷：verdict 三档只测头尾会漏掉中间档被写死，已补中间档『不含1年内』断言）。"},
    "V32": {"status": "done", "desc": "组合级联合分布 —— 首次检验「号与号之间有没有结构」（用户 2026-10-24「先放下经济价值，按照理论上仍可行的方向走下一步」）。V30/V31 的共同缺陷是 33 个号各自独立建模，违背「每期恰好 6 个」的负相关约束；V32 建**联合分布**。先排除两条：①直接对 C(33,6)=1,107,568 组合建模需 7,384 年数据；②序列级生成模型无转移结构（V30 自相关全在 2se 内）。三项实测：①**二阶共现（Ising）**：未条件化 χ²=624.0（df=528，最热对 6-14 共现 136 次 vs 期望 99.9，z=+3.99，看起来极显著）；但**条件化后 χ²=452.5 低于独立假设期望 528** —— 号与号不仅无正结构，还比独立更均匀。未条件化时的超额全部来自一阶偏置被放大 32 倍。②**一阶边际**：χ²=45.07（df=32），卡方表临界 46.19 判不显著，但**置换 p=0.0094 判显著** —— 因为零分布实际均值 26.97 而非理论 32（每期抽 6 个不放回 ⇒ 33 个计数和为定值 ⇒ 负相关 ⇒ 方差压缩），**用卡方表会漏判**。这是本轮最重要的方法论收获。但 14 号 695 次 vs 期望 639.6，z=+2.42 **未过 Bonferroni 阈值 z=3.17**（33 个号 = 33 次检验）。③**as-of 可利用性**（决定性关卡）：用最近 500 期估计偏置并跟随，准确率仅 0.2781，**远低于「总是猜不出现」的常数基线 0.8182** —— 18% 基础率下后者天然 82% 正确，分类准确率是错误判据。即使偏置永续，整注增益上限仅 +1.443%（1.0909 → 1.1066）。新增 GET /api/rigor/joint_distribution：pairwise/marginal/usability 三块同屏，2.8s 返回。端点体现三条判据纪律：零分布用置换不用卡方表、二阶 χ² 必须条件化、可利用性必须与常数基线对照。报告 docs/analysis/V32_JOINT_DISTRIBUTION.md。测试 1796 → 1806（新增 10 例，3 组变异实证全部抓红：条件化失效、零分布用有放回抽样、常数基线改成 0.5）。"},
    "V31": {"status": "done", "desc": "神经网络与全架构穷尽 —— AI 能不能直接学（用户 2026-10-24 两轮指示：「不找规律，能否让 AI 直接预测，基于 bp 神经网络、时序神经网络等算法来预测」+「不要限于我说的两种算法，扩展，把能用的都考虑上」）。结论：**13 种架构全部学不到**。特征 115 维（30期频次33 + 30期遗漏33 + 时间衰减33 + 蓝球16），多标签二分类（33 号各自是否出现），严格时序切分 1608/1609。实测排序：BP深(128-64-32) 1.1436(+4.83%) / 决策树 1.1262(+3.23%) / BP宽(256) 1.1206(+2.72%) / RF 1.1168(+2.38%) / SVM 1.1162(+2.32%) / HistGBDT 1.1119(+1.92%) / ET 1.1094(+1.69%) / KNN 1.0957(+0.44%) / LDA 1.0895(-0.13%) / BP(64-32) 1.0814(-0.87%) / GBDT 1.0820(-0.81%) / 逻辑回归 1.0839(-0.64%) / 朴素贝叶斯 1.0864(-0.41%)；基线 1.0909。**关键证据（嵌套置换）**：单模型置换时 BP(64-32) 观测 1.0758 比随机重训均值 1.1008 还低（z=-1.29, 19/20 次置换超过观测）；但正确的检验必须「每次置换都重跑全部模型再取最优」—— 实测打乱标签后最优仍达 1.1300(+3.6%)，而真实标签最优 1.1436(+4.83%)，p=2/8。⇒ 那 +4.83% 里约 +3.6% 来自「从多个模型里挑最好的」这个动作本身，真实信号仅 +1.2%（z=+1.15，不显著）。结构性原因：①标签是纯噪声，万能逼近定理的前提（存在可逼近函数）不成立；②BP深网络约 25700 参数 / 1608 样本 = 16 参数/样本，必然过拟合（这也解释了单模型在不同种子下从 1.0758 漂到 1.1436）；③无时序结构可学（相邻期重号 p=0.70、和值自相关全在 2se 内），LSTM 门控退化为恒等映射（实测时序衰减网络 1.0957/+0.44%）。新增 GET /api/rigor/neural_exhaustive：把 observed/nested_null/real_signal 三栏同屏，直接量出选择偏差（带 max_models 参数，测试限 3 个模型把耗时从 6min 降到 2.5min）。端点只回答「机器能不能学到」，不产号。报告 docs/analysis/V31_NEURAL_NETWORKS_EXHAUSTIVE.md。测试 1787 → 1796（新增 9 例，3 组变异实证全部抓红：零分布不重训、headline 固定显著、p 值不用加一校正）。"},
    "V30": {"status": "done", "desc": "预测力穷尽审计 —— 多维度多方法（用户 2026-10-24「不限于规律一个方法，下面研究增强预测力，多维度多方法去思考」）。结论：**17 项检验无一通过 Bonferroni（0/17）**。关键发现——现有方法的「超额」是估计偏差不是信号：freq 在真实数据上 +1.61%，而**纯随机序列上同一方法也 +1.54%**；分块独立化（每50期固定一组号）后偏差归零到 +0.06%。检验清单：①freq 时间置换 B=200 → p=0.5000；②红球边际χ²=45.07 p=1.0000；③蓝球边际χ²=8.71 p=1.0000；④相邻期重号 1.0830 p=0.7010；⑤尾数边际χ²=372.59 p=1.0000；⑥三区比多样性 p=1.0000；⑦含连号占比 0.6521 p=1.0000；⑧和值均值 100.9969 p=1.0000；⑨蓝球黏性 0.0691 p=0.0532（唯一接近显著，B=20000 复核 p=0.0589, z=+1.62）；⑩开奖顺序 6 个位置首次检验，位置4 χ²=46.73 超临界但内部置换 B=2000 得 p=0.4560（6 次检验 H0 下至少一次超临界概率 0.265，教科书级假阳性）；⑪红蓝联合独立性 χ²=432.1 df=480 → 独立；⑫和值/跨度/奇偶 lag1-5 自相关全在 2se 内；⑬ML 逐号回归系数均值 -0.0065 < 噪声量级 0.0213；⑭冷热对比（freq 正序 1.1252 vs 逆序 1.0356）差距小于噪声。经济论证：蓝球黏性即使成立，六等奖 EV 仅 +¥0.033/注（回收率 52.29%→53.94%），确证需 10,568 期≈70 年；而现有 3518 期的检出下限是 +4.1%（要检出 +1.6% 需 22,969 期≈153 年）。新增 GET /api/rigor/predictive_power：把上述审计做成**可随时重跑**的端点，强制四栏同屏——观测命中/置换p/随机序列对照/蓝球黏性，并带多重比较阈值；用增量计数把 n_perm=60 的耗时从 34.5s 降到 0.7s。端点只回答「有没有信号」，**不产号**（产号仍走 /api/predict），避免审计结论被误当推荐号码。报告 docs/analysis/V30_PREDICTIVE_POWER_EXHAUSTIVE.md。测试 1777 → 1786（新增 9 例，3 组变异实证全部抓红：perm_p 常量化、headline 固定结论、删蓝球经济价值说明）。"},
    "V29": {"status": "done", "desc": "按用户指示移除「更多」抽屉下 4 个研究页面（用户 2026-10-24「主页，更多，下面的四个页面功能都可以不要」）：删掉 tab-analysis（数据分析）/ tab-patterns（规律研究）/ tab-replay（历史回放）三个 section 容器 + 抽屉里对应按钮；「规律时效」经查并非独立页面（openRegimeTab() 实为 switchTab(eval)+setEvalSub(regime)），故只删抽屉入口、保留 eval 页内的 regime 子标签（用户确认）。关键防护：① switchTab 新增 VALID_TABS 白名单（predict/eval/data/settings）——页面删掉后历史书签的 #analysis 会命中不存在的 panel，原实现把所有 .active 摘掉却不加回任何一个，导致整页白屏；现在未知 tab 一律回落 predict 并纠正 hash。② switchTab 里删掉 analysis/patterns/replay 三个分支（否则 hashchange 仍能切到不存在的 tab）。③ 补三处空值保护：renderPatterns 提前返回（原本 patSummary 的 textContent 会在 null 抛 TypeError 打断整条 loadAll 链）、renderStats 先无条件赋值 window._currentProbs 再判断（该全局缓存被出号/诊断链路消费，若因页面消失提前 return 会把真实功能悄悄弄坏）、filterPatterns 提前返回。④ 保留全部数据加载函数本体（renderWindow/renderPatterns/populateReplaySelect 等）——它们与 allFeatures/allPatterns 共享数据管线，评估报告页仍在消费，贸然删除会牵动整条链路。实测：无头 Chrome 经 CDP 加载公网首页，.tab-panel 由 7 个降为 4 个（predict/eval/data/settings）、抽屉按钮由 8 个降为 4 个（评估报告/数据管理/设置/用户管理）、活动面板 tab-predict、控制台 0 个 JS 异常（仅 2 个既有 401 资源请求）。"},
    "V28": {"status": "done", "desc": "LLM 票逐注 L2 校验 + 清理从未产出的功能（用户 2026-10-23 指示「按 B+C」「将一些没实际用途的功能去掉」）：①**实测缺口**——issue=2026122 手动生成的 5 注中 seq=4 `[1,6,14,22,30,32]` 的 check_shape() 返回 False（奇数个数=1，odd_range=[2,3,4]），而 V18 对用户的承诺是「任何人拿发过的票重跑 check_shape() 就能验证」。根因不是约束算错，而是**从来没校验过票**：strict_mode_gates 只检查 shape_constraint_ctx() 的 7 个必需键在不在；engine.llm_tickets 里的 ctx 是**提示词上下文**（build_context）与结构约束无关；统计通道天然合规（sample_stat_ticket 内部按 ctx 采样），所以这个洞只在 LLM 通道暴露（它吐自由文本）。②修法 C（源头）：新增 shape_rules.explain_shape() 逐条说明违反哪一维（与 check_shape 同源，测试双向核对 ok==(violations==[])）；tickets_prompt 新增「必须满足的结构硬约束」段，把和值/跨度/三区比/奇偶/连号的**具体数字阈值**原样写进提示词（此前只说『结构均衡』，模型只能猜）。③修法 B（拦截+可见）：新增 routes_predict.verify_tickets_l2() 逐注校验并返回违规清单（**不丢弃**——丢弃会让 LLM 票悄悄变 0 注，是 V16 那类静默降级）；严格模式下任一注违规 → 422 + can_fallback=true（用户仍可选降级统计）；非严格模式只标记进 l2_ticket_check 字段。调度器 api_app 同样接校验（V16 教训：门控只在路由、调度器直连 engine.predict_next 绕过它）。④空票单陷阱：verify_tickets_l2([]) 返回 ok=False（防 vacuous truth），但 0 注不得触发 422——否则文案变成「0/0 注不满足结构约束（第?注 []：）」自相矛盾，已用 _l2_reject 的 n>0 门控隔离。⑤清理：删除 issue_baseline 表（迁移 7 创建、8 个版本 0 行、全仓库无写入方无读取方），迁移函数改为 no-op 保留序号连续性；从 predict 响应移除恒为 None 的 ai_audit 字段（llm_audit 表 0 行、save_audit 无任何调用点、前端渲染计数为 0）。测试 1762 → 1781（新增 19 例，4 组变异实证全部抓红：ok 恒 True、删除奇偶校验、违规票改为丢弃、删除提示词约束）。"},
    "V27": {"status": "done", "desc": "方差优化选号（用户 2026-10-23 指示「按照『四、可以做的是方差，而不是期望值』这个思路能做预测码」）：新增 lottery/variance_picker.py —— 用组合恒等式优化**票单结构**，不预测号码。数学依据：E[T]=n·12/11（与选号无关，恒等式）；Var[T]=n·σ₁²·[1+(n−1)(11/54)(Ā−12/11)]，方差只取决于重号平均 Ā。算法：模拟退火最小化 var_ratio（并集覆盖作平手判据）。实测对照（n=10，多组平均）：随机 var_ratio=1.0031/P(≥4红)=0.047768/并集28.8；最小方差 var_ratio=0.2019/P(≥4红)=0.048767/并集33.0 ⇒ 方差 −79.9%、P(≥4红) +2.09%、并集 +4.2。关键设计：①目标**不是**最大化 P(≥4红)——实测直接优化它会把 var_ratio 推到 1.40（比随机还差），因为它在搜更集中的票单；最小化方差是 Ā 的连续函数，退火稳定收敛，并集扩大是**副产物**；②var_T_of 与 engine_distribution._variance_stats 用同一套协方差式（测试逐位核对差 <1e-9），拒绝第二套口径；③新增 GET /api/portfolio/variance-pick，强制三样同屏：tickets/summary/honesty，honesty 必带 ev_unchanged=True（EV 恒为 n×¥1.0458、52.29% 回收）与 no_prediction=True（单注头奖恒为 1/17,721,088）；④baseline 用 iid 分支（var_ratio/union 仍是精确组合量，仅尾部概率差 0.3%~0.6%），把 n=60 抽样的对照耗时从 11s 降到 0.01s；⑤本端点**不进严格模式门控**——门控问的是「有预测力吗」，这里问的是「结构好不好」，语义不同不得混用。测试 1744 → 1762（新增 18 例，5 组变异实证全部抓红：协方差常量改错、目标反号、删 ev_unchanged、种子失效、baseline 静默省略）。"},
    "V25": {"status": "done", "desc": "修出号链路误用全局默认通道 —— 用户自备 LLM 配置不生效（用户 2026-10-23 报告「强制重新预测 → HTTP 503 Agent Router 当前分组 default 下对于模型 deepseek-v4-flash 无可用渠道」，而用户配的是可用的 group/subscription @ magpie.ff88.de）：engine.llm_tickets() 无条件读 config.llm_model_list()（全局默认），而 routes_predict.predict() 虽已算出 _eff（用户生效配置）却从未传给引擎 —— 与 V24 修的 /api/llm/test 是同一类缺陷、位置不同。修法：①llm_tickets 新增 model_cfgs 形参，有注入则优先、无则回退全局（调度器/评估等无用户上下文路径不受影响）；②predict_next 新增 llm_model_cfgs 并**两支都传**（llm_required=True 与 else 分支 —— 前者正是用户点「强制重新预测」走的路径，第一版只给 else 传，导致最关键的路径没生效）；③routes_predict 在 source==user 时用 with_secret=True 取明文并发注入（共享通道拿不到明文 Key，注入无 Key 配置会让引擎误判未配置）。实测对照：注入 INJECTED-MODEL 时引擎调用 INJECTED-MODEL；不注入时调用 agent-router/deepseek-v4-flash。测试 1739 → 1743，变异实证 2/2 抓红（含用户实际踩的 strict 分支漏传）。"},
    "V24": {"status": "done", "desc": "修 LLM 连接测试误用全局默认通道（用户 2026-10-23 报告「用户自己填写自定义模型，保存配置，测试是否正常。貌似测的还是默认的，不是手动填写的」）：/api/llm/test 旧实现只用 config.llm_model_list()（全局/管理员默认通道），**从头到尾没读用户自备配置** —— 用户填了自己的 base_url/Key/model 并保存后点测试，实际测的是另一条通道；若自备通道是坏的（地址错、Key 无效、模型名拼错），测试仍显示成功，这是最坏的一种误导。生产实证：用户1 生效来源 user、自备 model=group/subscription @magpie.ff88.de，而测试端点用的是全局 agent-router/deepseek-v4-flash。修法：改走 user_llm_store.effective_for_user(uid, with_secret=True) —— 全项目唯一的「该用户实际生效参数」判据（出号链路也用它），测试与出号从此同源。返回值新增 source(user/system/none) 与 base_url_host，让人看得出测的是哪条通道；无用户通道时才退回全局并标 source=system。实测用户自备通道短提示 3.0s 成功、长提示（真实观察轮）会超 180s 预算 —— 旧代码把后者也算作全局通道的成功，掩盖了真实差异。测试 1735 → 1739。"},
    "V23": {"status": "done", "desc": "L3 前向回测首屏公开 —— 把「未检出可预测性」变成可复核的事实（AI 重设计第四层，依据 docs/analysis/V19_AI_REDESIGN.md §4 层3）：新增 GET /api/rigor/walk_forward?method=&color=&n_tickets=&start=，复用 V20 的 walk_forward_gate（同一种子组 WF_SEEDS、同一门槛 WF_Z_MIN），返回 z_median/z_by_seed/n/hit_rate/baseline 并强制附带 headline_zh（结论句）与 honesty_zh（诚实声明）—— 不只丢数字让人自己猜。判据纪律：z_min 必须等于 WF_Z_MIN、seeds 必须等于 WF_SEEDS（不许另立标准）。非法 color 返回 400 **且不返回任何命中率读数**（防止被误读成结论）。红球通道基线 6*6/33、蓝球 1/16，口径分开。首屏新增「🎯 前向回测」卡片（index.html #walkForwardCard + app.js loadWalkForward），启动即加载、失败自降级不阻塞首屏。实测读数：stat:freq 蓝球 z 中位数 -0.06、命中率 6.22% vs 基线 6.25%，未检出可预测性 —— 这是最正常也最诚实的结果。测试 1728 → 1735；B_BASELINE 145 → 146（+1，已注明为何必要且无法合并）。"},
    "V22": {"status": "done", "desc": "L2 AI 角色分离 —— 新增 advisory 模式，AI 只解释、不参与选号（AI 重设计第三层，依据 docs/analysis/V19_AI_REDESIGN.md §1.2/§4 层2）：生产实测 45 注匹配样本，llm 票红球 1.265 / stat 票 1.182（基线 1.091）、蓝球 LLM 0.035 vs 统计 0.073（基线 0.0625），差异全在 1 个标准差内 —— LLM 选号无增量，却要付 503 风险、延迟与 token 成本。改动：LLM_OUTPUT_MODES 由 (off, only, mix) 扩为 (off, **advisory**, only, mix)；advisory 模式下 llm_can_pick 恒为 False（不产号），且**严格模式也不得覆盖**（历史严格模式会强制 llm_can_pick=True，若不排除 advisory，该承诺会在严格部署下被静默推翻）；新增 ai_role ∈ {off, advisor, picker} 并随 resolve_channels 快照与预测响应返回，前端元信息显示「🤖 AI 角色：仅解释（未参与选号）」或「参与选号」。only/mix 语义**一字未改**（向后兼容，不删既有行为）；llm 族关闭时 advisory 也回落 off（沿用拒绝列表优先纪律）。测试 1721 → 1728。"},
    "V21": {"status": "done", "desc": "L1 AI 假设生成器 —— 把 LLM 从「挑参数」改成「提可证伪假设」（AI 重设计第二层，依据 docs/analysis/V19_AI_REDESIGN.md §1.1/§4 层1）：旧 llm_lab 要求 LLM 输出 filters:[{key,params}]，输出空间=134 档排列组合，是搜索不是推理（枚举一遍只要 12.8~17s，比问 LLM 快两个数量级），且 data/llm_claims.json 的 claims/runs 长期为 0（从未产出）。新增 lottery/ai_hypothesis.py：LLM 输出改为 question+test_spec+why+falsifier，test_spec.key 必须取 FAMILIES 已实现族、params 必须完全匹配已声明档位（自造参数一律拒——无实现则无从验证），无 falsifier 一律拒（不可证伪不是科学假设）。自动证伪三关：①存在性 discover 能跑出该族；②阶段 regime 非 flipped/error；③前向回测（复用 V20 的 walk_forward_gate，起点按数据长度自适应，避免短样本 n=0 退化）。前向回测对象是方法通道 stat:freq 而非维度本身，返回值强制带 wf_caveat_zh 说明此口径。新增端点 GET /api/ai_hypothesis/schema（公开提示词供审计）与 POST /api/ai_hypothesis/falsify（只证伪、不调 LLM、不产号码）。失败项原样保留并带原因——记录「问过什么、答案是没有」本身就是产出。生产实测 4 条真实假设全部三关跑通、结论 walk_forward_rejected。测试 1708 → 1721。"},
    "V20": {"status": "done", "desc": "L0 前向回测判据门（AI 重设计第一层，依据 docs/analysis/V19_AI_REDESIGN.md §2.3）：修「置换检验显著 ≠ 可预测」这一方法论缺口。实测证据：ml_blue 在 B=200 下 p=0.010、换种子 0/7 分别 0.010/0.005（看起来稳定显著），但同一数据做严格前向回测命中率 5.00% vs 基线 6.25%、z=-1.26 —— 判据与前向回测直接矛盾。新增 lottery/walk_forward_gate.py：①walk_forward_check 复用 battle.replay_method 的 as-of 纪律，给出相对随机基线的 z；②**多种子稳健性**（WF_SEEDS=[42,7,99]，要求 z 中位数过线）—— 实测 stat:freq 红球单次 z=1.74 但三种子 1.74/0.83/-0.33，换种子就翻转，必须判否；③combine_gates 双重门槛：置换 p<α 且前向回测 z>1.0 才算 signal_found，任一不过降 monitor，不可回放（llm）标 unverifiable 且不伪造 z。接入 ml_model.diagnose_ml_channels 顶层：通道级 tier/trusted 保持原语义（t21 契约），只在顶层加 all_dual_pass 约束。口径坑已修三次：per_issue.blue_hit 是 round(mean) 不能累计（否则 z=-14 荒谬结论）、红球是每期平均命中数不能套百分号、判据是「中位数」不是「单次 z」。测试 1697 → 1708。"},
    "V18": {"status": "done", "desc": "严格模式换门控依据（依据 docs/analysis/V17_STRICT_MODE_EVALUATION.md，用户 2026-10-23 批准）：严格模式此前硬性要求 usable_replayable>0（L1 统计规律），而 L1 在 3515 期上需 +28% 真实效应才可达（n=150、m_cum≈3000 ⇒ z=4.31），真实数据最强仅 +5.0%（差 5.6 倍，换算需 31 年），再叠加 FROZEN_ISSUES=150 冻结期 ⇒ 严格模式永久不出号。改法：门控依据改为「L1 有则用 L1（更强）；无则用 L2 结构约束兜底（可逐注复核）」，并在响应/前端如实标注 evidence_basis ∈ {L1_statistical, L2_structural}。L2 是组合恒等约束（和值/跨度/三区/奇偶/连号），与 C(33,6)=1,107,568 理论枚举一致，任何人拿发过的票重跑 check_shape() 即可验证——比统计显著更硬，且不需要样本量。可行性实测：L2 约束在 3518 期真实数据上 7 个必需字段无一缺失（violation=0）；issue=2026122 那 5 注降级票 5/5 全部满足 L2 约束，即凭据早已存在、只是门控不认。红线未动：ALPHA=0.05、ERA_FAMILY_M=32+locked、FROZEN_ISSUES=150、USABLE_STATUSES、累计口径。口径纪律：L1 与 L2 是两种不同计数，分列展示禁止相加，不得把 L2 叫做「可用规律」，L2 通过不代表存在可预测下期的规律、不提高中奖概率。测试 1689 → 1697。"},
    "V16": {"status": "done", "desc": "严格模式降级可见化（依据用户 2026-10-23 指示）：修「手动点不出号、后台却悄悄存了统计票」的语义漏洞。故障实证：issue=2026122 的 5 注全是 stat:*/uniform、无一条 llm:，因为严格模式门控只挂在 /api/predict 路由上，而后台调度器 _scheduler_step 直接调 engine.predict_next() 绕过它。改法：①门控判定抽为纯数据函数 strict_mode_gates() + strict_scope_for_user(user_id)，路由与调度器共用同一套语义（不再有两套口径）；②后台自动生成改为「严格模式优先 → 不出号则降级统计模式」，并把 source_mode=stat_fallback + 未通过门控写进快照（新增 db.mark_prediction_source / load_prediction_source，不改表结构）；③手动生成在严格模式被拒时返回 422 + can_fallback，前端弹窗让用户自己选是否改用统计模式，用户确认（allow_stat_fallback=1）才降级，默认绝不自动降级；④缓存路径同样带回 source_mode，前端元信息显著标注「口径：统计模式（严格模式未出号）」或「口径：严格模式」。测试 1680 → 1689（新增 9 例，含真实调用路由断言与 3 组变异实证；源码守卫改用 _source_assert.js_code 剥注释，未上调 B_BASELINE 配额）。"},
    "V15": {"status": "done", "desc": "扩假设空间，只扩问题、不放宽判据（依据 docs/analysis/P2_STATS_WHY_ZERO.md §7.1 与 docs/analysis/V15_ASSUMPTION_SPACE.md）：①S1 候选稀释修复——discover 阶段画像按族去重（同族只留 |z| 最强一条），top-20 画像内不同族数 13→20，n_tested 分母不变（去重不是减检验次数）；②S2 新增 persistent_effect「长期效应」档——长期档整体 Bonferroni 显著（m=n_tested）∧ 近 K=3 段方向一致，不要求每段显著，把 H0 下 P≈3e-5 的断崖改成斜坡；USABLE_STATUSES 仍为 ('persistent',) 一字未动，且必须同样过 validated+prereg_ok+source!=llm；关键修正是长期检验须剔除用于挑选候选的近 fast_window 段（否则纯随机误报率 0.20，修正后 0.0~0.03）；③S7 新增 order_pos/number_bias/sum_dist 三族——order（开奖顺序）字段此前被 discovery 读取 0 次，现可检位置偏置（+35% 注入检出 0.783，此前恒 0）；三族走 stat_fn 分布检验通道，n_tested 124→134。红线 N1–N7 全部未动：ALPHA=0.05、ERA_FAMILY_M=32+locked、累计口径 Bonferroni、usable 三条件、as-of 语义、只降不升、≥4红/≥5红不可检验。双侧验收：纯随机 persistent_effect 族级误报 0.03/200（≤FWER 0.05）、number_bias 单号 +10% 检出 0.0（正确——需 3.7 万期，诚实结论）。测试 891 → 1680 passed（首次全量零失败；含引擎指纹口径改为固定 3500 期，真值锁不再每期开奖都红）。"},
    "V12": {"status": "done", "desc": "规律「窄/假/半」三重修复（依据 docs/analysis/LEAD_VERDICT.md）：①V12-01 修复 30/124 档从未被检验——V11 新增 8 个形态族用 history[-w:] 切片，但 asof_scan 传入 ScanContext（dict 子类）触发 KeyError 被 except Exception 静默吞成空集，再被 sd<=0 二次跳过；给 ScanContext 加切片支持后 n_tested 由 94 → 124、E[FP] 4.70 → 6.20，8 族全部进入搜索（真实数据 best_quick_p_adjusted=0.0804，与预测逐位吻合，仍 >0.05 故「修好≠找到」）。②V12-21 补集反转：hits(S)+hits(Sᶜ)≡6 ⇒ delta_S+delta_Sᶜ≡0，负效应本就是补集的正效应；原 resolve_registry 把 delta<0 判 rejected 直接丢弃，等于销毁一半信号；现在原假说仍记 rejected（诚实保留）并同时登记补集候选（仍需 A/B+前置注册三关门），complement_of 标记来源、n_tested_delta 恒 0 保证 FWER 不变。③V12-04 拆除跨层求和：patterns_runtime 与 rule_lifecycle 两处 usable=long+phase 改为 by_layer 分列 + all_tiers_sum 显式命名，避免首屏「N 条可用其中 0 条有预测力」。④V12-12 形态通过率全枚举精确化：span 由手写 0.95 修正为 0.999853、zones 0.90→0.799152、sum 0.90→0.906411、consec 0.99→0.997042、parity 由 2 万次抽样 0.8257→精确 0.825159；生产通过率锁定 68.2992%（结构锁 + 真值锁 + 反向用例）。测试 858 → 891 passed。"},
    "M1": {"status": "done",    "desc": "前端 Tab 工作台 + 规律库扩容 29 条 + 自动挖掘管道 + 任务系统"},
    "M2": {"status": "done",    "desc": "GBDT/RF 概率模型 + 滚动 Brier 加权融合 + 概率校准 + 蓝球独立投票 + ML walk-forward 评估"},
    "M3": {"status": "done", "desc": "研究闭环：LLM 三轮辩论完善与离线评估、挖掘管道增强、规律研究台"},
    "M4": {"status": "in_progress", "desc": "长期运营：在线累积报表、方法 A/B 开关、通知、多源对账、运维工程"},
    "M5": {"status": "done", "desc": "交互升级：回放命中可视化、数据分析面板增强（走势/分布/遗漏/蓝球）、评估报告信息架构重组"},
    "U1": {"status": "done", "desc": "极简预测首页：7 Tab 改 2 层导航（本期推荐 + 更多抽屉），票卡默认折叠 + 匹配度星级"},
    "U2": {"status": "done", "desc": "蓝球专项：blue_specialist 专用模型 + 蓝球覆盖优先 + 运行命中率回退 + 复式/胆拖投注模式"},
    "U3": {"status": "done", "desc": "方法自适应降权 + LLM 角色重定位（组合结构优化，蓝球交由专用模型）"},
    "U4": {"status": "done", "desc": "组合命中覆盖优化：贪心 + 模拟退火的 coverage_r 红球 / 蓝球覆盖优化器"},
    "U5": {"status": "done", "desc": "单注复盘战绩卡 + 跟我买一键选注 + 中奖回执闭环"},
    "U6": {"status": "done", "desc": "推理型模型自动关闭思考兜底（reasoning_content 非空时注入 thinking:disabled 重试）"},
    "U7": {"status": "done", "desc": "LLM 实际生成注数模式：以大模型真实产出注数为准，不再用统计/ML 候选补齐"},
    "U8": {"status": "done", "desc": "推理型模型输出预算自适应（含预算记忆）+ 设置页模型唯一事实来源 + 观察轮报错聚合 + 上期开奖回执列出上期推荐逐注命中"},
    "U9": {"status": "done", "desc": "历史战绩口径修正：期数取接口实际去重样本（不再是用字面量 N / 各方法相加），样本不足不给最佳方法结论；卡片下移至「上期开奖回执」之后"},
    "V4": {"status": "done", "desc": "诚实口径改造：/api/rigor/randomness + /api/rigor/signal + /api/portfolio/expected-value 三个端点（缺 rigor 模块时优雅降级 503）；前端不再把随机波动当能力展示——所有比率必须带样本量与「样本不足，不构成结论」提示，新增「随机性检验」卡展示最小可检出提升量"},
    "V9": {"status": "done", "desc": "①修复「刷新后又显示旧预测」的根因：本地残留一条越界期号 2026401（2026-10-15 由已修复的 next_issue() 生成），它在文本与数值排序下都大于同年合法期号，导致 /api/predictions/last 永远返回它。改为按落库时间取候选 + 期号合法性过滤（_valid_issue），并清理脏数据；同时把库内 7 处 ORDER BY issue DESC 全部改为 CAST 数值排序。②「我的号码诊断」改为整注粘贴输入（01 06 12 21 24 31 + 12），不再分蓝/红两个框；解析支持带 + / 不带 + / 逗号 / 前导零。③规则分层语义修正：noise（各阶段未检出）不再是 rejected，改归 monitor——界面上「5 条不可用」实际是「证据不足」，与「已证伪」区分开；并把「为什么没有可用规律」写成可读原因（三道门槛 + 实测的重号率 bootstrap 置信区间）。"},
    "V8": {"status": "done", "desc": "阶段切分与体验修复：①阶段改为数据驱动的自适应切分（默认 block=150 期一段，历史约 23 段；可切回自然年 9 段对照），并对命中数指标改用配对差值 t 检验口径（此前误用二项比例，方差算错导致 p 失真）；实测 family-wise 假阳性 0.045、+0.15 红球/期效应检出力 0.65（自然年口径仅 0.31）②长耗时 AI 闭环/LLM 工坊改为后台任务（/api/rules/closed-loop/async、/api/llm_lab/run/async），前端每 3 秒轮询显示阶段+进度+已用时间，可离开页面 ③全站号码格式统一为 01 06 12 21 24 31 + 12（formatTicket/formatReds/parseRedsInput 单一来源），诊断卡回显统一格式并支持空格/逗号/顿号输入 ④预测/健康接口防前端缓存，服务启动后异步刷新规律快照 ⑤修复「更多→规律时效」入口抽屉未收起"},
    "V7": {"status": "done", "desc": "AI 规律闭环 + 数据对齐：①目标期号改为按真实开奖日历推算（每周二/四/日，BT.expected_issue），数据落后时明确报「缺几期/落后几天」，不再出现 2026401 这类越界期号；抓取改为全量对齐（补回漏抓期）。②本期推荐显示生成时间、数据基准、注数口径；/api/predictions/last 返回 generated_at/data_basis。③新增 lottery/patterns_runtime.py 规律运行时快照（data/patterns_runtime.json），把「阶段时效 + 维度扫描 + 假说库」压缩成低延迟可用清单，engine.predict_next 每期消费：可用规律=阶段持续有效且 A/B 验证通过，其余只监控/否决；默认 shadow 模式（记录影响不改号），LOTT_RULE_APPLY=1 生效。④票级「规律引用」由引擎按真实命中写入，不再由 LLM 自填。⑤/api/rules/{runtime,refresh,closed-loop} 与前端「本期规律使用情况」卡：挖掘→验证→注册→刷新快照→出号，一键闭环"},
    "V6": {"status": "done", "desc": "阶段有效性 + 多维度规律：新增 lottery/regime.py（9 阶段切分 + 逐阶段检验 + BH 校正 + 近 3 期加权 + 持续/新兴/衰减/陈旧/反转/伪规律状态机 + 近 150 期加权检验）、lottery/discovery.py（20 维度族 × 91 组参数的 as-of 快速筛选与阶段画像，单次扫描约数秒）、lottery/llm_lab.py（LLM 规律工坊：全历史阶段序列 + 维度表 + 搜索成本喂给模型，三轮提规律/自我质疑/修订，每条规律强制映射维度表并做 as-of 回放，映射不到判不可验证）；A/B 验证修复（两臂独立采样，结构类假说不再恒为 0 差异；对照组由 uniform 改为实际部署基线）；前端新增「规律时效」工作台（阶段热力图 + 维度扫描 + LLM 工坊 + 检索式核验）"},
    "V5": {"status": "done", "desc": "自进化系统：lottery/battle.py 方法级 walk-forward 战斗回放（红球 Wilcoxon + 蓝球 McNemar + Bonferroni/BH 校正）+ lottery/hypothesis.py LLM 规律提炼与假说库（LLM 失败必降级）+ lottery/evolution.py 进化闭环（假说→出号策略→A/B 实战验证→validated/rejected 裁决）；调度器每期开奖后自动跑一轮进化；前端「实战战绩」子页呈现战绩榜（ROI 双口径）、两两对比（未校正 N→校正 M）、规律库与进化操作"},
}

# ---------- LLM 通道（全部来自环境变量，无仓库内置密钥/地址） ----------

LLM_DISABLED = os.environ.get("LOTT_LLM_DISABLED", "0") == "1"

# 主通道：OpenAI 兼容端点
LLM_BASE_URL = (os.environ.get("LOTT_LLM_BASE_URL") or "").rstrip("/") or None
LLM_API_KEY = os.environ.get("LOTT_LLM_API_KEY") or None
LLM_MODEL = os.environ.get("LOTT_LLM_MODEL") or "minimax-m3"

# 多模型：同一主通道下使用的模型列表（逗号分隔），默认仅 LLM_MODEL
_models_env = [m.strip() for m in os.environ.get("LOTT_LLM_MODEL_LIST", "").split(",") if m.strip()]
LLM_MODEL_LIST: List[str] = _models_env or ([LLM_MODEL] if LLM_MODEL else [])

# 附加独立通道：JSON 数组，每个元素 {"name","base_url","api_key","model"}
LLM_EXTRA_MODELS: List[Dict] = []
try:
    raw = os.environ.get("LOTT_LLM_EXTRA_MODELS", "")
    if raw.strip():
        LLM_EXTRA_MODELS = json.loads(raw)
        if not isinstance(LLM_EXTRA_MODELS, list):
            LLM_EXTRA_MODELS = []
except (json.JSONDecodeError, TypeError):
    LLM_EXTRA_MODELS = []

LLM_SAMPLES = int(os.environ.get("LOTT_LLM_SAMPLES", "3"))          # LLM 多轮采样次数（并发）
TICKETS_PER_LLM_CALL = int(os.environ.get("LOTT_TICKETS_PER_CALL", "5"))
# 最终输出注数（V11 用户要求：默认 5 注，可配置手动改个数）
N_TICKETS = int(os.environ.get("LOTT_N_TICKETS", "5"))              # 最终输出注数（默认 5）
# 出号随机种子（V36，2026-10-24 审计 P0-4）。
#
# 为什么必须固定：`engine.predict_next` 未传 rng 时原先用未播种的
# `random.Random()`，导致同一期同一数据连跑两次票号不同；而离线回测
# (`evaluate.offline_backtest`) 的 `seed` 只传给了随机基线、没传给引擎，
# 于是同 seed 两次回测 `delta_red_hits` 会取 -0.04 / -0.11 两种值，
# 恰好跨过 hypothesis 的 |delta| ≥ 0.05 判决阈值 ⇒ **同一份数据的结论翻转**。
#
# 固定种子的代价：同一天内多次请求会得到同样的票。这**不是缺陷**，
# 因为「同一期、同一数据、同一模型」本就该给出同一个答案；重抽的需求
# 由 `regenerate=True` + 显式传入不同 rng 表达，而不是靠熵。
PREDICT_SEED = int(os.environ.get("LOTT_PREDICT_SEED", "20261024"))
# V11 严格模式（用户 2026-10-18 指令）：
#   1=开：出号**必须**由 LLM 分析规律完成；LLM 不可用或可用规律数为 0 时
#       **拒绝出号**并返回明确错误（绝不静默降级到统计基线）。
#   0=关（默认）：保持 V10 诚实边界，stat/ml 兜底出号。
# 注意：开启前请确认 LLM 网关可用（data/llm_config.json），否则系统将出不了号。
STRICT_MODE = os.environ.get("LOTT_STRICT_MODE", "0") == "1"
# LLM_ONLY_OUTPUT（V10 起默认 **0**；历史上默认 1）
#   =1（only）：LLM 通道产出候选时，最终注数以 LLM 实际返回的注数为准，不用统计/ML 补齐。
#   =0（off，**当前默认**）：LLM 只提供候选，最终出号仍由确定性排序 + 蓝球覆盖计划决定。
# V10 改默认值的理由（docs/reviews/V10_REVIEW_SUMMARY.md G1 + R3）：双色球为独立随机
# 事件，LLM 无可利用信息；把它设为默认出号主力会让"最没用的通道"决定结果
# （最近 4 期真实开奖 20 注、蓝球 0/20）。保留本开关作为紧急回退。
LLM_ONLY_OUTPUT = os.environ.get("LOTT_LLM_ONLY_OUTPUT", "0") == "1"
LLM_TIMEOUT = float(os.environ.get("LOTT_LLM_TIMEOUT", "60"))
# 大输出预算（推理型模型）单次调用的宽松超时：预算抬到上限后单次要跑 40~120s，
# 仍用 60s 会先被判读取超时而白白重试一次。
LLM_LONG_TIMEOUT = float(os.environ.get("LOTT_LLM_LONG_TIMEOUT", "180"))
# 推理型模型（如 deepseek-v4-flash / glm-5.3 等）会把 max_tokens 全部耗在
# reasoning_content 上，content 为空且 finish_reason=length。此时按需把输出预算
# 抬到本上限再重试一次（两次都能被 deadline 钳制）。
# 默认 32000：实测 agent-router/deepseek-v4-flash 需要 ≥3 万输出预算才吐出正文。
LLM_MAX_TOKENS = int(os.environ.get("LOTT_LLM_MAX_TOKENS", "32000"))
# 学到的「每个模型可用输出预算」持久化文件（首次探测成功后记住，避免每次预测
# 都从被截断的小预算重试）。键为 "<base_url>|<model>"。
LLM_BUDGET_FILE = DATA_DIR / "llm_budget.json"
# M4.5 快速失败：单次预测 LLM 阶段（观察+选号+校验全部轮次）的墙钟预算（秒）。
# 默认 75s：加统计部分约 5s 后仍在 Cloudflare 100s 代理超时之内，避免 HTTP 524。
# 超预算即判定“大模型超时”，Web 预测直接失败提示，不再降级为纯统计。
LLM_TOTAL_TIMEOUT = float(os.environ.get("LOTT_LLM_TOTAL_TIMEOUT", "75"))

# M4.5：请求级附加参数（合并进每个 chat/completions 请求体）。
# 背景：部分网关的推理型模型会把 max_tokens 全部耗在 reasoning 上（content 为空），
# 需要按模型下发 reasoning_effort / enable_thinking 等开关控制思考行为。
# LOTT_LLM_EXTRA_BODY：JSON 对象，对所有模型生效（如 {"reasoning_effort":"low"}）。
# LOTT_LLM_EXTRA_BODY_MAP：JSON 对象（模型名 → 参数对象），叠加在默认之上，按模型覆盖。
LLM_EXTRA_BODY_DEFAULT: Dict = {}
LLM_EXTRA_BODY_BY_MODEL: Dict[str, Dict] = {}


def _load_extra_body_env() -> None:
    global LLM_EXTRA_BODY_DEFAULT, LLM_EXTRA_BODY_BY_MODEL
    raw = (os.environ.get("LOTT_LLM_EXTRA_BODY") or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                LLM_EXTRA_BODY_DEFAULT = obj
            else:
                print("[config] 警告：LOTT_LLM_EXTRA_BODY 必须是 JSON 对象，已忽略")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[config] 警告：LOTT_LLM_EXTRA_BODY 解析失败（{e}），已忽略")
    raw = (os.environ.get("LOTT_LLM_EXTRA_BODY_MAP") or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                cleaned = {str(k): v for k, v in obj.items() if isinstance(v, dict)}
                bad = len(obj) - len(cleaned)
                if bad:
                    print(f"[config] 警告：LOTT_LLM_EXTRA_BODY_MAP 有 {bad} 个非对象值已忽略")
                LLM_EXTRA_BODY_BY_MODEL = cleaned
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[config] 警告：LOTT_LLM_EXTRA_BODY_MAP 解析失败（{e}），已忽略")


_load_extra_body_env()

# ---------- M4.2 方法 A/B 开关 ----------
# 运营筛查阈值：60 期开始提示，连续 120 期无显著差异才允许人工考虑关闭。
METHOD_RECOMMENDATION_MIN_ISSUES = int(os.environ.get("LOTT_METHOD_RECOMMENDATION_MIN_ISSUES", "60"))
METHOD_RECOMMENDATION_DISABLE_ISSUES = int(os.environ.get("LOTT_METHOD_RECOMMENDATION_DISABLE_ISSUES", "120"))
METHOD_RECOMMENDATION_ALPHA = float(os.environ.get("LOTT_METHOD_RECOMMENDATION_ALPHA", "0.05"))

# ---------- V7 规律 → 出号 ----------
# 规律命中一个号码给该注加多少「置信度分」。刻意温和（默认 1.0/个）：
# 规律只能在候选池内部重排，不能凌驾于概率模型之上。
# 尺度依据（实测）：候选池内置信度区间只有约 20~35 分、极差 3~6 分，
# 所以 1 个命中（+1.0）足以在池内挪动一两位，但打不过「整体高 4 分以上」的注——
# 这正是"温和偏置"想要的力度。调大（如 3.0）会让规律压过概率模型，不建议。
RULE_CONF_BOOST = float(os.environ.get("LOTT_RULE_CONF_BOOST", "1.0"))

# ---------------------------------------------------------------------------
# 候选置信度的**族偏置**（P5′，M3 §8.3 H1 修复）
# ---------------------------------------------------------------------------
#
# ## 修的是什么
#
# 历史实现里 LLM 族有**恒定 +20** 的置信度加成（外加 0.45 vs 0.25 的斜率）：
#
#     llm:  conf = 20 + 0.45 * (0.5*自评 + 0.5*ensemble)    # ≈ 50–51
#     其它: conf = 10 + 0.25 * ensemble                     # ≈ 10–35
#
# 后果是 LLM 注**必然**排在候选池最前（排序纯按 confidence 降序），
# 实测某期 25 注里 24 注来自 LLM，而 `llm_output_mode="off"` 只控制"注数"、
# 不影响力排序。于是 V10 "LLM 不做主力"的诚实默认被实质推翻，
# 而界面上还按旧口径叙事。
#
# 更严重的后果（M3 H1）：**它让"LLM 到底行不行"永久不可测**。
# 唯一能评估 LLM 的对照臂 `battle.llm_arm_report` 需要同期的非 LLM 票作分母，
# 而偏置把非 LLM 票全挤掉了 → `n_pairs` 恒为 0。
#
# ## 为什么做成可配置且默认为 0
#
# 默认 0 = 各族用**同一公式**排行，对照实验重新变得可能。
# 保留配置项是为了：若确实想跑"LLM 优先"的实验，可以显式设置并留下痕迹，
# 而不是把它写死在代码里、让所有人默默承受。
#
# 注意：置信度本身是**结构分，不是中奖概率**（UI 文案已声明）。
# 这里调整的只是"谁排在前面"，不改变任何号码的概率。
LLM_CONF_BIAS = float(os.environ.get("LOTT_LLM_CONF_BIAS", "0"))
#: 其它族（stat/ml/blend）的偏置，默认 0（与 LLM 族同一起跑线）
OTHER_CONF_BIAS = float(os.environ.get("LOTT_OTHER_CONF_BIAS", "0"))
# LOTT_METHODS（逗号/空格分隔，未设置 = 全部启用）：可关闭/保留任意方法通道
# （stat 各基线 / ML / LLM / uniform），令牌可写全名（stat:freq）或族名（stat）。
# LOTT_METHOD_MODE：production（默认，严格按开关过滤）/ research（忽略开关、全部启用，
# 供方法对比实验；与决策规则「未经 120 期 paired 验证的方法仅以研究模式存在」对应）。
from . import methods as _methods

METHODS_RAW = (os.environ.get("LOTT_METHODS") or "").strip()          # 原始开关字符串
METHOD_MODE = _methods.normalize_mode(os.environ.get("LOTT_METHOD_MODE"))  # production / research
METHODS_SPEC = _methods.implement_spec(METHODS_RAW)                   # 解析规格，供引擎过滤候选

# 运行时方法配置持久化（Web 设置页写入，优先于 .env，重启不丢失）
METHODS_CONFIG_FILE = DATA_DIR / "methods_config.json"

# ---------- V10-00a：开关来源标记 + 通道冻结快照 ----------
# METHODS_SOURCE：生效的方法开关来自哪一层（"methods_config.json" / ".env" / "default"）。
# **必须在 import 期一次性定值**，不得在 resolve_channels() 里现算后回写
# —— config 自己会把 json 的生效值写回 os.environ，任何"现算+回写"都会重演来源污染。
# 判定逻辑见 methods.resolve_layer()（查文件优先 + `in os.environ`，不用 .get）。
METHODS_SOURCE = _methods.resolve_layer()

# CHANNELS：本期通道的冻结快照（resolve_channels 的只读缓存）。
# 用途：让 engine/battle/evaluate/api 只读一处，而不是各自去读环境变量（第 7 套开关的根源）。
# 注意：这是 import 期快照；设置页改开关后必须调 refresh_channels() 刷新。
CHANNELS = _methods.resolve_channels()


def refresh_channels() -> Dict:
    """重新解析并刷新 CHANNELS 快照（设置页保存开关后调用）。"""
    global CHANNELS, METHODS_SOURCE
    METHODS_SOURCE = _methods.resolve_layer()
    CHANNELS = _methods.resolve_channels()
    return CHANNELS

# ---------- U2 蓝球专项 ----------
# LOTT_BLUE_MODE：model（默认，用 blue_specialist 等模型）/ uniform（整期回退 16 个均匀随机）
BLUE_MODE = (os.environ.get("LOTT_BLUE_MODE") or "model").strip().lower()
if BLUE_MODE not in ("model", "uniform"):
    BLUE_MODE = "model"
# LOTT_BLUE_COVER：picked 阶段是否按“蓝球去重覆盖优先”（1=是，默认；0=沿用旧的同蓝上限）
BLUE_COVER = os.environ.get("LOTT_BLUE_COVER", "1") == "1"
# 蓝球模型运行命中率回退阈值：近 BLUE_FALLBACK_WINDOW 期命中率低于该值 → 该期回退 uniform
BLUE_FALLBACK_WINDOW = int(os.environ.get("LOTT_BLUE_FALLBACK_WINDOW", "30"))
BLUE_FALLBACK_MIN_RATE = float(os.environ.get("LOTT_BLUE_FALLBACK_MIN_RATE", "0.045"))
# V4-C 修复：回退判定必须达到的最小样本量（期）。30 期样本的命中率标准误为
# sqrt(0.0625*0.9375/30) ≈ 4.42%，远大于 6.25%-4.5%=1.75% 的差额——在这个样本量
# 上做回退决策等于在噪声上决策。0（默认）= 由 engine._min_significant_blue_sample()
# 用 1.96*sd/delta 反推（≈735 期）；显式设正值则用它，但仍不会低于 BLUE_FALLBACK_WINDOW。
BLUE_FALLBACK_MIN_SAMPLE = int(os.environ.get("LOTT_BLUE_FALLBACK_MIN_SAMPLE", "0"))
# 随机蓝球命中理论基线（1/16）
BLUE_RANDOM_RATE = 1.0 / 16

# ---------- U2 投注模式（single / blue_compound / dan_tuo） ----------
BET_MODE = (os.environ.get("LOTT_BET_MODE") or "single").strip().lower()
if BET_MODE not in ("single", "blue_compound", "dan_tuo"):
    BET_MODE = "single"
# blue_compound：红 6 固定 + 蓝 k 个 → k 注（蓝球复式展开）
BLUE_COMPOUND_K = int(os.environ.get("LOTT_BLUE_COMPOUND_K", "5"))
# dan_tuo：红胆 n 个固定 + 红拖补齐 6 红 + 蓝 1 个 → 展开多注
DAN_TUO_DANS = int(os.environ.get("LOTT_DAN_TUO_DANS", "4"))

# ---------- U3 方法自适应降权 ----------
# ---------- V11「阶段可用」档（phase_usable）----------
# 用户诉求（2026-10-18）：「不要只用一套极严标准把一切判死，也要能看见阶段性规律，
# 只要当前可用、能指导预测就行。」
#
# 诚实前提（docs/reviews/PHASE_USABLE_STUDY.md 四组独立检验）：
#   ① 3515 期上当前连 emerging 级别规律都没有（0/91）；
#   ② 最近窗口 6/8 维度同向，但随机窗口 15% 就能出现；
#   ③ 滚动用「最近 150 期最佳」出号 3165 期：Δ=+0.010，t=0.64 **不显著**；
#   ④ **相邻窗口无持续性**（r=0.124，容差 ±0.151）—— 这是最致命的一条：
#      「当前阶段有效」不能预测「下一期有效」，所以该档**不可用于出号的理论依据**。
#
# 但仍然提供该档，因为用户有权选择，且系统会**用实测数据说话**：
#   * 默认关闭（0）；
#   * 打开后只有「长期 persistent」与「阶段 emerging/fading」两类进入候选；
#   * 阶段类必须同时满足：前置注册通过 + 最近阶段 A/B 不劣于基线；
#   * 界面对两类**分别**统计真实的配对 Δ 与 t 值，让效果自我暴露；
#   * 累计配对检验显示明显劣于基线（t < -PHASE_USABLE_AUTO_OFF_T）→ 自动关闭。
LOTT_PHASE_USABLE = os.environ.get("LOTT_PHASE_USABLE", "0") == "1"
# 自动关闭阈值：阶段类规律的累计配对 t 值低于此值即自动关闭（默认 -2.0）
PHASE_USABLE_AUTO_OFF_T = float(os.environ.get("LOTT_PHASE_USABLE_AUTO_OFF_T", "-2.0"))
# 阶段类规律进入候选所需的最小配对期数（样本不足不统计、也不自动关闭）

# 启动期后台补刷规律快照：默认开。
# 关闭它用于「pytest / CLI / 任何会反复 import api_app 的场景」——
# 每次 import 都起一个后台线程会累积并发连 SQLite，叠加 joblib 多进程后
# 实测触发 Segmentation fault（2026-10-18）。测试由 conftest.py 置 0。
BOOT_REFRESH_RULES = os.environ.get("LOTT_BOOT_REFRESH_RULES", "1") == "1"

# LOTT_ADAPTIVE_ENABLED：1=开（默认），0=关（保留人工 A/B 开关语义）
ADAPTIVE_ENABLED = os.environ.get("LOTT_ADAPTIVE_ENABLED", "1") == "1"
# 连续 LOTT_ADAPTIVE_K 期命中低于随机基线 → 自动降权 50%（0.5）或移出候选池
ADAPTIVE_K = int(os.environ.get("LOTT_ADAPTIVE_K", "20"))
ADAPTIVE_WINDOW = int(os.environ.get("LOTT_ADAPTIVE_WINDOW", "60"))
ADAPTIVE_DOWNWEIGHT_FACTOR = float(os.environ.get("LOTT_ADAPTIVE_DOWNWEIGHT_FACTOR", "0.5"))
# 命中比较口径：red（红球平均命中）/ blue（蓝球命中率）/ prize（≥五等奖率）
ADAPTIVE_METRIC = (os.environ.get("LOTT_ADAPTIVE_METRIC") or "red").strip().lower()
# 自适应状态持久化文件（不修改 SQLite schema）
ADAPTIVE_STATE_FILE = DATA_DIR / "adaptive_state.json"

# ---------- U4 组合覆盖优化 ----------
# coverage 模式：目标“至少命中 r 个红球”，蓝球覆盖数，退火迭代上限
COVERAGE_R = int(os.environ.get("LOTT_COVERAGE_R", "3"))
COVERAGE_TRIALS = int(os.environ.get("LOTT_COVERAGE_TRIALS", "3000"))
COVERAGE_BLUE_K = int(os.environ.get("LOTT_COVERAGE_BLUE_K", "4"))
# V4-C 覆盖优化目标函数：
#   expected_value（默认）= 奖金期望 + 蓝球覆盖奖金（真实可优化的口径）
#   coverage              = 旧口径（红球并集大小 + 蓝球覆盖数），向后兼容
#
# V10-00a 别名表：值域变更时旧值必须**继续可用**（不得报错）。
#   expected_value → ev_conserving   （V10 新名：诚实口径——EV 守恒，只比冗余度，
#                                     对应 docs V10 §3.4 的可插拔目标）
#   ev_conserving  → ev_conserving
#   coverage       → coverage
#   pmax_high / custom_weight 直通（V10 新增目标，见 docs/V10 §3.4）
COVERAGE_OBJECTIVE_ALIASES = {
    "expected_value": "ev_conserving",
    "ev": "ev_conserving",
    "ev_conserving": "ev_conserving",
    "coverage": "coverage",
    "pmax_high": "pmax_high",
    "custom_weight": "custom_weight",
}
_raw_objective = (os.environ.get("LOTT_COVERAGE_OBJECTIVE") or "expected_value").strip().lower()
COVERAGE_OBJECTIVE = COVERAGE_OBJECTIVE_ALIASES.get(_raw_objective, "ev_conserving")


def methods_status() -> Dict:
    """当前方法开关状态：模式 / 原始字符串 / 解析规格（供 API 与前端展示）。"""
    return {
        "mode": METHOD_MODE,
        "raw": METHODS_RAW,
        "spec": {
            "mode": METHODS_SPEC.get("mode", "all"),
            "tokens": sorted(METHODS_SPEC.get("tokens", set())),
        },
    }


def set_methods(raw: Optional[str] = None, mode: Optional[str] = None) -> Dict:
    """运行时更新方法开关（Web 设置页写入，立即生效并持久化）。

    raw: LOTT_METHODS 字符串（None = 不变）；mode: production / research（None = 不变）。
    同时更新模块全局与 os.environ，供引擎“后续预测”即时读取。
    """
    global METHODS_RAW, METHOD_MODE, METHODS_SPEC
    if raw is not None:
        METHODS_RAW = str(raw or "").strip()
        err = _methods.validate_raw(METHODS_RAW)
        if err:
            raise ValueError(err)
        os.environ["LOTT_METHODS"] = METHODS_RAW
    if mode is not None:
        METHOD_MODE = _methods.normalize_mode(mode)
        os.environ["LOTT_METHOD_MODE"] = METHOD_MODE
    METHODS_SPEC = _methods.implement_spec(METHODS_RAW)
    if not _save_methods_config():
        raise OSError("方法配置持久化失败")
    # V10-04b：出号通道快照必须同步刷新，否则 engine 读到的还是启动时的旧口径
    # （settings 页刚关掉 llm，引擎却仍按 llm_can_pick=True 出号）。
    refresh_channels()
    print(f"[config] 方法开关已更新: mode={METHOD_MODE}, raw='{METHODS_RAW}'")
    return methods_status()


def _load_runtime_methods_config() -> None:
    """启动时读取 data/methods_config.json（若存在），覆盖方法开关配置。"""
    global METHODS_RAW, METHOD_MODE, METHODS_SPEC
    try:
        if not METHODS_CONFIG_FILE.exists():
            return
        conf = json.loads(METHODS_CONFIG_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        print("[config] 警告：methods_config.json 解析失败，忽略运行时方法配置")
        return
    changed = False
    if "raw" in conf and isinstance(conf["raw"], str):
        candidate = conf["raw"].strip()
        if _methods.validate_raw(candidate):
            print("[config] 警告：methods_config.json 含非法 methods，忽略运行时配置")
            return
        METHODS_RAW = candidate
        os.environ["LOTT_METHODS"] = METHODS_RAW
        changed = True
    if "mode" in conf and isinstance(conf["mode"], str):
        METHOD_MODE = _methods.normalize_mode(conf["mode"])
        os.environ["LOTT_METHOD_MODE"] = METHOD_MODE
        changed = True
    if changed:
        METHODS_SPEC = _methods.implement_spec(METHODS_RAW)
        print(f"[config] 已加载运行时方法配置（mode={METHOD_MODE}, raw='{METHODS_RAW}'）")


def _save_methods_config() -> None:
    """把当前方法开关写入 data/methods_config.json（与 llm_config.json 同目录）。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"mode": METHOD_MODE, "raw": METHODS_RAW}, ensure_ascii=False, indent=2)
        tmp = METHODS_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(METHODS_CONFIG_FILE))
    except OSError as e:
        print(f"[config] 方法配置写入失败: {e}")
        return False
    return True

# ---------- M3 LLM 离线评估与研究专用配置 ----------

LLM_EVAL_ISSUES = int(os.environ.get("LOTT_LLM_EVAL_ISSUES", "60"))     # 三通道 walk-forward 窗口（期）
LLM_EVAL_TICKETS = int(os.environ.get("LOTT_LLM_EVAL_TICKETS", "5"))    # 每期每通道注数
LLM_EVAL_SEED = int(os.environ.get("LOTT_LLM_EVAL_SEED", "42"))         # 随机基线种子（stat/random 可复现）
LLM_EVAL_MODEL = os.environ.get("LOTT_LLM_EVAL_MODEL") or None          # 评估专用模型（可选，轻量优先，控成本）
LLM_EVAL_PRICE_PER_1M = float(os.environ.get("LOTT_LLM_EVAL_PRICE_PER_1M", "1.0"))  # 每百万 token 估算价（USD，仅展示）
LLM_VERIFY_MODEL = os.environ.get("LOTT_LLM_VERIFY_MODEL") or None      # 第三轮校验模型（M3.2 接入）
LLM_VERIFY_ENABLED = os.environ.get("LOTT_LLM_VERIFY", "1") == "1"      # 第三轮校验开关（生产默认开；离线评估按通道可控）


def llm_configured() -> bool:
    """是否存在任何可用的 LLM 通道（主通道或附加通道）。"""
    if LLM_DISABLED:
        return False
    if LLM_BASE_URL and LLM_API_KEY and LLM_MODEL_LIST:
        return True
    return bool(LLM_EXTRA_MODELS)


def llm_model_list() -> List[Dict]:
    """返回全部可用模型配置（主通道模型列表 + 附加通道），供引擎多模型采样。"""
    cfgs: List[Dict] = []
    if not LLM_DISABLED and LLM_BASE_URL and LLM_API_KEY:
        for m in LLM_MODEL_LIST:
            cfgs.append({
                "name": m, "base_url": LLM_BASE_URL, "api_key": LLM_API_KEY, "model": m,
            })
    for extra in LLM_EXTRA_MODELS:
        name = extra.get("name") or extra.get("model", "extra")
        if extra.get("base_url") and extra.get("api_key") and extra.get("model"):
            cfgs.append({
                "name": name, "base_url": str(extra["base_url"]).rstrip("/"),
                "api_key": str(extra["api_key"]), "model": str(extra["model"]),
            })
    return cfgs

# 多尺度窗口（0 表示全量；其余为最近 N 期）
WINDOWS = {"long": 0, "mid": 150, "short": 30}

# 回测参数

# 离线评估

# ---------- M2 ML 概率模型（GBDT + 随机森林 + 校准） ----------

ML_ENABLED = os.environ.get("LOTT_ML_ENABLED", "1") == "1"          # 是否把 ML 概率接入集成融合
ML_MIN_START = int(os.environ.get("LOTT_ML_MIN_START", "300"))      # 最少历史期数才开始训练
ML_REFIT_EVERY = int(os.environ.get("LOTT_ML_REFIT_EVERY", "10"))   # 滚动评估时每 N 期重训一次
ML_EVAL_WINDOW = int(os.environ.get("LOTT_ML_EVAL_WINDOW", "60"))   # ML 滚动评估窗口（期）
ML_N_ESTIMATORS = int(os.environ.get("LOTT_ML_N_ESTIMATORS", "60")) # 树数量（RF 与 GBDT 通用上限）
ML_MAX_DEPTH = int(os.environ.get("LOTT_ML_MAX_DEPTH", "5"))        # 树深度
ML_CAL_CV = int(os.environ.get("LOTT_ML_CAL_CV", "2"))              # 概率校准折叠数（小=快）

# ML 模型「特征/结构」版本号（V10-08 / R1 P2-5）。
#
# 为什么需要它：`ml_model._cache_key` 过去只是 `"{期数}:{末期号}"`，不含任何
# 特征定义版本。于是**改特征工程后 key 不变** → `data/ml_models/<key>.joblib`
# （单个 55MB）被静默复用，跑出来的模型与代码不一致，表现为「改了没生效」。
#
# 约定（硬规则）：任何会改变模型输入/输出语义的改动，**必须递增这个常量**：
#   * 改 `_FEATURE_NAMES` / `_BLUE_FEATURE_NAMES` 的维数或含义；
#   * 改 `_compute_features` / `_compute_blue_features` 的计算口径；
#   * 改集成结构、校准方式、标签定义（y 的构造）。
# 递增后旧 joblib 自动失效（key 变了），旧文件可安全删除（见
# `ml_model.prune_model_cache`）。仅调整超参（树数/深度/CV 折数）无需递增，
# 但若要让改动立刻生效也应递增，否则会继续命中旧缓存。
MODEL_SCHEMA_VERSION = "2"        # v2：V10-08 首版带版本号的缓存键（此前无版本，v1 仅指历史）

# ---------- V11 S5：规律生命周期（衰减 → 退居二线 monitor）----------
# 用户诉求（2026-10-18）：「用 AI 来判断哪些规律可能掉下去，适时把这些规律预测
# 退居二线，只观察不使用了。」
#
# 设计见 docs/reviews/AI_DEMOTION_DESIGN.md。三条纪律固化在这里：
#   ① **动作权归确定性状态机**（rule_lifecycle.decide_rule_state，纯函数、零 LLM），
#      LLM 只做「复核意见 + 解释」，其 Skill 可度量、可自动降级；因此本模块**任何**
#      故障都不会让规律无法使用，也不会让衰减的规律赖在一线。
#   ② **目标态只能是 monitor**（证据不足、继续观察），绝不自动写 rejected
#      （rejected 只来自 hypothesis.resolve_registry 的「反对」判定）。
#   ③ **必须有滞后**：连续 K 次同向判定才迁移，迁移后 MIN_DWELL 期内不再评估，
#      避免噪声让状态每期来回跳。
#
# 为什么这三个数是 2/2/3：连续 2 次同向的误触发率 ≈ 单次误触发率²，再叠加 3 期
# 冷却，每周误切换率 <1%（蒙特卡洛验收见 tests/test_v11_rule_lifecycle.py）。
LIFECYCLE_STATE_FILE = DATA_DIR / "rule_lifecycle.json"
LIFECYCLE_K_DEMOTE = max(1, int(os.environ.get("LOTT_LIFECYCLE_K_DEMOTE", "2")))
LIFECYCLE_K_UPGRADE = max(1, int(os.environ.get("LOTT_LIFECYCLE_K_UPGRADE", "2")))
LIFECYCLE_MIN_DWELL = max(0, int(os.environ.get("LOTT_LIFECYCLE_MIN_DWELL", "3")))
# 单次周期最多评估多少条规律（防止规律库膨胀后一个周期跑太久）
LIFECYCLE_MAX_RULES = max(1, int(os.environ.get("LOTT_LIFECYCLE_MAX_RULES", "60")))

# 调度（开奖日：周二/四/日 21:35 后自动 抓取+评估+预测）
#
# V36（2026-10-24 审计 P1-9）：默认值 0 → **1**，与 docker-compose.yml
# （`${LOTT_SCHEDULER:-1}`）和 README 对齐。
#
# 历史上这里是 0、compose 是 1 —— 于是"直接用 python 跑"和"用 compose 跑"
# 是两个不同系统：本地跑不出定时预测，而容器里会跑。这类漂移最难排查，
# 因为代码看起来"没毛病"，只是没人注意到三处默认值不一致。
# 统一取 1：单机部署本来就该自动跟开奖跑；需要关闭时显式 LOTT_SCHEDULER=0。
SCHEDULER_ENABLED = os.environ.get("LOTT_SCHEDULER", "1") == "1"
DRAW_WEEKDAYS = (1, 3, 6)  # 周一=0 … 周日=6 -> 周二/四/日
DRAW_TIME = "21:35"

# ---------- M4.3 开奖通知 ----------
NOTIFY_WEBHOOK = (os.environ.get("LOTT_NOTIFY_WEBHOOK") or "").strip()
NOTIFY_SMTP_HOST = (os.environ.get("LOTT_NOTIFY_EMAIL_SMTP_HOST") or "").strip()
NOTIFY_SMTP_PORT = int(os.environ.get("LOTT_NOTIFY_EMAIL_SMTP_PORT", "587"))
NOTIFY_SMTP_TLS = os.environ.get("LOTT_NOTIFY_EMAIL_SMTP_TLS", "1") == "1"
NOTIFY_SMTP_USER = (os.environ.get("LOTT_NOTIFY_EMAIL_SMTP_USER") or "").strip()
NOTIFY_SMTP_PASSWORD = os.environ.get("LOTT_NOTIFY_EMAIL_SMTP_PASSWORD") or ""
NOTIFY_EMAIL_FROM = (os.environ.get("LOTT_NOTIFY_EMAIL_FROM") or "").strip()
NOTIFY_EMAIL_TO = (os.environ.get("LOTT_NOTIFY_EMAIL_TO") or "").strip()


# ---------- 运行时 LLM 配置持久化（Web 界面写入，优先于 .env） ----------

LLM_CONFIG_FILE = DATA_DIR / "llm_config.json"


def load_runtime_llm_config() -> None:
    """启动时读取 data/llm_config.json（若存在），覆盖 LLM 通道配置。"""
    global LLM_DISABLED, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MODEL_LIST, LLM_SAMPLES
    global LLM_VERIFY_ENABLED, LLM_TOTAL_TIMEOUT
    global LLM_EXTRA_BODY_DEFAULT, LLM_EXTRA_BODY_BY_MODEL
    if not LLM_CONFIG_FILE.exists():
        return
    try:
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            conf = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("[config] 警告：llm_config.json 解析失败，忽略运行时配置")
        return
    if "disabled" in conf:
        LLM_DISABLED = bool(conf["disabled"])
    if conf.get("base_url"):
        LLM_BASE_URL = str(conf["base_url"]).rstrip("/") or None
    if conf.get("api_key"):
        LLM_API_KEY = str(conf["api_key"])
    if conf.get("model"):
        LLM_MODEL = str(conf["model"])
        # U6 修复：设置页保存的模型是唯一事实来源，**替换**（而非追加）模型列表，
        # 只保留这一个有效模型，避免旧通道的 glm/qwen（magpie 上 404）混入多模型轮转。
        LLM_MODEL_LIST = [LLM_MODEL]
    if isinstance(conf.get("samples"), int) and conf["samples"] > 0:
        LLM_SAMPLES = int(conf["samples"])
    if isinstance(conf.get("verify"), bool):
        LLM_VERIFY_ENABLED = bool(conf["verify"])
    if isinstance(conf.get("total_timeout"), (int, float)) and conf["total_timeout"] > 0:
        LLM_TOTAL_TIMEOUT = float(conf["total_timeout"])
    # U6 修复：设置页通道可自带 extra_body（如关闭思考），避免推理型模型
    # 因 reasoning_content 占满预算而触发“放大重试”导致总耗时超限。
    if isinstance(conf.get("extra_body"), dict) and conf["extra_body"]:
        LLM_EXTRA_BODY_DEFAULT = dict(conf["extra_body"])
        if LLM_MODEL:
            LLM_EXTRA_BODY_BY_MODEL = dict(LLM_EXTRA_BODY_BY_MODEL)
            LLM_EXTRA_BODY_BY_MODEL[LLM_MODEL] = dict(conf["extra_body"])
    print(f"[config] 已加载运行时 LLM 配置（{LLM_CONFIG_FILE.name}）")

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 加载运行时配置（须在 DATA_DIR 创建后）
load_runtime_llm_config()
_load_runtime_methods_config()
