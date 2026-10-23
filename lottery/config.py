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
SESSION_COOKIE_NAME = (os.environ.get("LOTT_COOKIE_NAME") or "__Host-ssq_sid").strip()
# Cookie 的 Secure 属性。生产（nginx 443）必须 True；本地 http 联调可设 0。
SESSION_COOKIE_SECURE = (os.environ.get("LOTT_COOKIE_SECURE") or "1").strip() == "1"


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

APP_VERSION = "4.9.0"          # v27：新增方差优化选号（组合结构优化，不预测号码）

# 单注 2 元（双色球官方单价，固定事实）。
# 放在 config 而不是某个路由模块里：它被 evaluate / portfolio / rigor 共用，
# 属于领域常量，不属于任何单个端点域。
TICKET_PRICE = 2.0
APP_BUILD = "2026-10-V27"      # 构建标识（V27 交付批次）
APP_MILESTONES = {
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
SCHEDULER_ENABLED = os.environ.get("LOTT_SCHEDULER", "0") == "1"
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
