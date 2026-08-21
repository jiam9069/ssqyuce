# 双色球智能预测分析系统 · M3 / M4 升级规划

> 姊妹文档：[docs/UPGRADE_PLAN.md](UPGRADE_PLAN.md)（v2.0 总方案，M1/M2 已完成）。
> 本文档是对 **M3 研究闭环**（1–2 周，短期冲刺）与 **M4 长期运营**（持续迭代）的任务级规划：
> **M3 的工程级开发方案（v0.7.0，文件级任务/排期/验收/部署）另见 [docs/M3_DEV_PLAN.md](M3_DEV_PLAN.md)。**
> 现状盘点 → 任务拆分 → 验收标准 → 工作量 → 风险与决策规则。
>
> 延续项目诚实边界原则：双色球开奖是独立随机事件，**任何规律/模型提升都必须能通过
> 样本外回测 + 多重检验校正**，否则不进入 UI。M3/M4 的最大价值是**把无规律量化为可用证据**，
> 并让系统在长期运营中**可度量、可比较、可维护**。

---

## 1. 现状盘点（已核对的真实代码状态，2026-08）

| 模块 | 已上线（当前状态） | M3/M4 缺口 |
|---|---|---|
| 前端 Tab 工作台 | M1 完成：预测中心 / 数据分析 / 规律研究 / 历史回放 / 评估报告 / 数据管理 / 设置 | 无大缺口；细节见 M3.4 |
| 规律库 | 35 条（跨度/龙头凤尾/质合/大小/012路/连号/同尾/AC值/遗漏区间/共现/蓝球独立结构等） | 触发时间线数据已存（backtest_json），前端仅展示汇总与边际曲线，**未做逐触发时间线/多选对比** |
| 规律自动挖掘 | M1 基础版：特征 lift → 候选规律 → 回测分级入库（三段切分防偏差） | ① 特征仅 20 维、基于 lift，**未用 LightGBM/RF 特征重要性**；② 无挖掘报告/合格率/入库明细页面 |
| LLM 通道 | 三轮已实现（观察→选号→校验/质疑），多模型并发采样，结构化输出 | ① 提示词未注入**规律回测明细表 + 上期预测回馈**；② 输出 schema 缺 evidence / counter_evidence / structure_scores 落库展示；③ **无 LLM 离线评估** |
| ML 概率模型 | M2 完成：GBDT+RF 软投票、蓝球独立、校准（sigmoid）、Brier 加权融合、walk-forward 评估（Brier/log-loss/校准/ECE/paired） | 评估结果已可展示；可加 ML 逐期命中累积曲线与特征解释摘要 |
| 评估体系 | 离线引擎回测 + 在线对照入库（eval_results） | ① 无 eval_meta（方法组合/权重/LLM 开关切片的**长期对比**）；② 无累积报表；③ 无 A/B 开关 |
| 数据管道 | 单源 e.17500.cn，增量入库 + 原始快照 | 无第二源交叉校验、无对账告警、特征每次全量重算、无 schema_version 迁移机制 |
| 运维 | Docker Compose、日志轮转、healthcheck、可选开奖日调度、数据卷持久化 | ① 无开奖通知；② 公网无鉴权选项；③ ECharts fallback 声明 /static/vendor/ 但本地 vendor 文件缺失 |

---

## 2. M3 研究闭环（核心冲刺，建议 1–2 周）

> 目标：让系统能**持续产出并证伪规律**、能**用数据回答 LLM/ML 值不值得开**、把研究过程变成
> 可交互可视化的研究台。所有产出必须有回测证据（随机基线 + paired 检验）。

### M3.0 快速修补（0.5 天，P0）
- [ ] 补齐 web/static/vendor/echarts.min.js（服务器 curl 下载放入镜像/仓库），真正做到断网可用；
- [ ] 主页版本徽标与 /api/info（已随 v0.6.0 完成：health.version + 页脚版本号）；
- [ ] 资产版本统一用构建时间戳管理（当前手改 ?v=，改为读 /api/info 的 build 字段拼接）。

### M3.1 LLM 离线评估（核心，2–4 天，P0）
**背景**：README 声称 LLM 值不值得开待回答，目前离线评估全部 use_llm=False。
**任务**：
- 新增 lottery/llm_eval.py：在 60–120 期 walk-forward 上跑三条通道 —— stat（纯统计+ML）、stat+llm、random（同注数随机基线）；
- 指标：红球平均命中、蓝球命中率、≥五等奖率、ROI、每期成本（token/耗时/费用近似）；
- 统计：paired 显著性检验（wilcoxon/sign-test，复用 ml_model 的 paired 检验），BH 校正跨通道；
- 落库：新增 llm_eval_results 表（run_id / 窗口 / 通道 / 指标 / p 值 / 成本 / 日期）；
- 前端：评估报告 Tab 增加 LLM 离线评估卡片（三通道对比柱状图 + 累积曲线 + 成本行）；
- 固定随机种子，评估结果可复现。
**验收**：一页报表回答 LLM 相对纯统计的 Δ 及其 p 值、每期成本，结论无论正负都入库展示。

### M3.2 LLM 通道完善（1–2 天，P1）
- 提示词注入：规律回测明细表（n / 边际 / p_adj / 等级 / Wilson CI）+ 上期预测回馈（命中情况）；
- 输出 schema 扩展：每注 evidence（引用的具体统计数字）、counter_evidence、structure_scores（和值/奇偶/三区/跨度评分）；predictions 表加列或 JSON 扩展字段，前端票卡展开依据展示；
- 第三轮校验默认开启低温度 + 不同模型（已有通道支持，仅需接入默认配置 LOTT_LLM_VERIFY_MODEL）。

### M3.3 规律挖掘管道增强（2–3 天，P0）
- 特征升级到 40+ 维：合并 ml_model 已用的特征集 + 遗漏区间/邻号/区间邻居/位置模式；
- 挖掘器：LightGBM → 特征重要性 Top-K → 自动生成候选规律文本（如号码 x 在 y 遗漏区间后出现率 lift）→ 送入 backtest 框架（walk-forward + BH）→ 合格（B 级及以上）才入库；挖掘/校准/测试三段切分保持；
- 产出挖掘报告：候选数 / 入库数 / 平均 lift / 合格率 / 历史运行对比（存 mining_runs 表）；
- CLI：python -m lottery.cli mine --engine lightgbm（保持 sklearn 可选、numpy 兜底）。

### M3.4 规律研究台增强（1–2 天，P1）
- 规律详情：**触发时间线**（近 N 个触发期高亮命中/未命中）+ 逐触发期命中 vs 期望**滚动曲线**（backtest_json 已含逐期序列）；
- 多选规律 → 边际对比图（升级现有单条形图）；证伪规律红牌徽标；
- 新增规律自动挖掘结果页签（见 M3.3 挖掘报告）。

### M3.5 回放与诊断收尾（1 天，P2）
- 历史回放：每期附当时所用方法/权重/LLM 开关快照（读 predictions.method 与 eval_meta 时间线）；
- 号码诊断：相似注检索展示 Top-K 明细（现仅聚合统计），支持导出 CSV。

### M3 验收标准（Exit Criteria）
1. LLM 离线评估报告可在 UI 查看，含三通道对比与 paired p 值、每期成本 —— 结论明确值/不值/无显著差异；
2. 挖掘管道在 3490 期数据上跑出 ≥1 条 B 级及以上新候选，且全部候选均有回测参数可追溯；
3. 任何进入 UI 的新规律/新结论都带随机基线 + p 值徽标；
4. 规律详情可查看触发时间线与滚动边际曲线；
5. 全流程回归：fetch → backtest → predict → 在线对照全链路无回归。

---

## 3. M4 长期运营（W3 起持续迭代）

> 目标：让系统**长跑可信**——评估可累积可比较、数据有冗余、运维可控、文档与合规完整。

### M4.1 在线累积评估报表（P0）
- 新增 eval_meta 表：每次预测记录 method 全名（如 stat:freq、blend:brier、llm:minimax-m3）、模型权重快照、LLM 开关、n_tickets、version；开奖后对照累加；
- 新增 /api/eval/cumulative：按方法切片输出**每期 ROI / 蓝球命中率 / 红球命中滚动均值**（10/30/60 期窗口）；
- 前端评估报告：累积曲线（带 95% CI 阴影）+ 方法对比表 + 导出 CSV。

### M4.2 方法 A/B 开关（P1）
- 配置 LOTT_METHODS（默认全部启用）：可关闭/保留任意方法通道（stat 各基线 / ML / LLM）；
- 在线评估按方法切片长期对比，每 60 期给出是否仍值得保留建议（paired 检验 + 成本）；
- 决策规则：某方法连续 120 期与基线无显著差异且成本为正 → 默认关闭并提示，可手动重新开启。

### M4.3 开奖通知（P1）
- 新增通知模块 lottery/notify.py：开奖对照完成后推送 —— Server酱 / 邮件 SMTP / 企业微信 webhook（三选一或全开）；
- 配置环境变量：LOTT_NOTIFY_WEBHOOK、LOTT_NOTIFY_EMAIL_*，Docker Compose 透传；
- 通知内容：期号、开奖号、系统预测命中摘要（红/蓝/奖级/ROI），失败静默降级。

### M4.4 数据多源对账（P1）
- 增加备用源（福彩官网 / 500.com），data_check.py 每日对账三源期号 + 开奖号，不一致 → 告警日志 +（可选）通知；
- 特征增量缓存：features 表按期缓存，新期只增量计算窗口特征，避免全量重算；
- draws 表加 source 字段（默认 17500），支持来源追溯。

### M4.5 工程与运维（P2）
- **schema_version 迁移机制**：新增 schema_version 表 + lottery/migrations/ 目录，启动时顺序执行迁移；
- **可选鉴权**：LOTT_TOKEN 配置后，公网所有 /api/* 要求 Bearer Token（未配置保持开放，兼容本地）；
- 备份：DEPLOY.md 已有 cron 备份脚本，补充 data/llm_config.json 与 web/static/vendor 备份说明；
- 日志与监控：容器日志轮转已有；/api/health 增加 version（已做）与 uptime/最近任务状态；
- 性能：ML 预热已有缓存；评估类长任务全部走 tasks 表异步（已有框架，补齐 ML eval 任务化）。

### M4.6 文档与合规（P2，持续）
- README 升级记录随版本更新（已建「版本记录」章节）；
- 免责声明与理性购彩提示保持前置；公网部署默认建议加反代 + 基础认证。

### M4 验收标准（Exit Criteria）
1. 每期开奖后自动对照并按方法累计，累积报表 10 期后可生成；
2. 双源数据对账日检通过 ≥30 天无告警或告警可解释；
3. schema_version 迁移在存量库上可平滑升级（新增表/列不丢数据）；
4. 通知任选其一实测送达，失败不阻塞主流程；
5. 版本信息（/api/health + 页脚 + README）三者一致。

---

## 3.7 M4.1 首个切片实施状态（v0.8.0）

- ✅ `eval_meta`：保存预测方法、版本、构建标识、LLM 状态与配置快照；
- ✅ `eval_details`：按期号 / 方法 / 注序保存逐注红蓝命中、奖级、奖金、成本与净收益；
- ✅ 在线对照兼容旧预测，并幂等写入 M4.1 明细；
- ✅ `/api/eval/cumulative`：按方法输出累计指标与逐期明细；
- ✅ `/api/eval/meta`：查询预测元数据快照；
- ✅ `/api/eval/export.csv`：导出累计评估逐注明细；
- ✅ 评估报告增加累计报表与 CSV 导出入口；
- ✅ 版本基线更新为 v0.8.0 / 2026-08-M4.1，M3 标记完成、M4 标记进行中；
- ✅ Python 编译检查与 M4 评估存储幂等冒烟通过。

> 真实数据验收（2026-08）：数据 3490 期，最新期 2026093；已有预测为未开奖期 2026094，
> 因此 `eval_results` / `eval_meta` / `eval_details` 尚无真实对照记录。在线对照、累计 API、CSV
> 路由和原始数据解析已通过只读检查；待 2026094 开奖后执行首次真实对照。

## 3.8 M4.2 方法开关实施状态

- ✅ `lottery/methods.py`：`implement_spec` / `normalize_method` / `is_enabled` / `filter_candidates`
  四个纯标准库函数，解析 `LOTT_METHODS`（默认全部启用；`-` 前缀 = 拒绝列表，无前缀 = 允许列表，族名/全名皆可）；
- ✅ `engine.predict_next` 接入开关：关闭的 stat 基线不进 Brier 融合（全关回退均匀兜底），
  LLM 通道关闭不发起调用（省成本），uniform/最终候选均按开关过滤；
- ✅ `.env.example` 补充 `LOTT_METHODS` 示例；`tests/test_m4_2_methods.py` 覆盖开关语义。
- ⏳ 待办：60 期后 paired 检验建议、escalation 决策规则落地（见 M4.2 后续）。

## 3.9 M4.2 方法开关（配置/API/前端）实施状态

> 在 3.8 的 `lottery/methods.py` 纯函数与引擎接入基础上，补齐运行模式、Web 配置
> 与持久化，使方法 A/B 开关可在生产/研究两种模式下由管理员在设置页在线调整。

- ✅ **运行模式 `LOTT_METHOD_MODE`（production / research，默认 production）**：
  `research` 忽略 A/B 开关、全部方法启用（对应决策规则「未经 120 期 paired
  验证的方法仅以研究模式存在」）；`production` 严格按 `LOTT_METHODS` 过滤。
- ✅ **`config.set_methods(raw, mode)`**：运行时更新模块全局与 `os.environ`
  （`LOTT_METHODS` / `LOTT_METHOD_MODE`），重解析 `METHODS_SPEC`，并写
  `data/methods_config.json` 持久化（启动时 `_load_runtime_methods_config`
  优先加载，重启不丢失，仅影响之后生成的预测）。
- ✅ **`lottery/methods.py` 扩展**：`normalize_mode` / `effective_spec(mode, spec)`
  / `validate_raw` / `registry()`（方法全名·族·中文说明注册表）。
- ✅ **引擎按模式过滤**：`engine.predict_next` 改用 `METH.effective_spec(
  config.METHOD_MODE, config.METHODS_SPEC)` 作为唯一生效规格，默认行为不变。
- ✅ **API**：`GET /api/methods/status`（模式 / raw / 解析 spec / 生效 spec /
  方法族开关 / 注册表描述）；`POST /api/methods/config`（接受 `{methods, mode}`，
  校验 methods 字符串与 mode 合法性，`set_methods` 即时生效并持久化）。
- ✅ **评估快照**：`db.save_eval_meta` 的 `config_snapshot_json` 增加
  `method_mode` / `methods_raw` / `methods_spec`，历史评估不依赖当前配置。
- ✅ **前端**：设置页新增「🎛️ 预测方法开关」卡片（模式下拉 + `LOTT_METHODS`
  输入 + 保存 + 族开关/注册表表格）；评估页顶部显示当前方法开关（族启用状态）。
- ✅ **测试**：`tests/test_m4_2_methods.py` 扩展 `normalize_mode` /
  `effective_spec` / `validate_raw` / `registry` / `set_methods`；Python 编译检查与
  API/配置冒烟通过。

> 后续（不在本次范围）：60 期后 paired 检验自动给出「是否仍值得保留」建议、以及
> 连续 120 期无显著差异时默认关闭并提示的 escalation 决策规则落地。

## 3.10 M4.2 方法建议筛查实施状态

- ✅ 新增 `db.method_recommendations()`：按期开奖方法聚合，与 `uniform` 基线建立同期配对；
- ✅ 无第三方依赖的双侧 sign-test，输出同期样本量、p 值、奖金差、成本与状态；
- ✅ 新增 `GET /api/eval/recommendations`，默认 120 期 / 60 期最低样本；
- ✅ 评估页新增「方法建议」入口，明确“只提示、不自动关闭”，避免未经人工确认改变生产配置；
- ✅ 120 期且校正前 p≥0.05 的方法仅标记 `disable_candidate`，建议手动关闭并保留研究模式；
- ✅ 测试覆盖 paired 数据不足与 API 冒烟。

> 注意：当前筛查是运营提示，不等价于科研结论；正式决策仍应结合完整 paired 检验、多重校正、LLM 成本与至少 120 期样本外证据。

## 3.11 M4.3 开奖通知实施状态

- ✅ 新增 `lottery/notify.py`：Webhook 与 SMTP 两类通知通道；
- ✅ `/api/eval/online` 对照完成后触发通知，调度器同步接入；
- ✅ 每个通知通道独立异常隔离，发送失败只记录日志，不阻塞评估与下期预测；
- ✅ 新增 `/api/notify/status` 查看通道配置状态；
- ✅ Docker Compose 与 `.env.example` 已透传通知配置；
- ✅ 测试覆盖未产生新对照时不发送、Webhook 失败静默降级。

> 通知默认关闭。配置 SMTP 密码或 Webhook 时请只写入服务器 `.env`，不得提交仓库。

## 3.12 M4.4 多源对账首个切片

- ✅ 新增 `lottery/data_check.py`：备用源抓取、期号集合对齐、开奖号码差异检测；
- ✅ 新增 `GET /api/data/reconcile`，未配置备用源时返回 `not_configured`，不会影响主流程；
- ✅ 新增 `LOTT_BACKUP_DATA_URL` 配置；对账为只读操作，不覆盖主库；
- ⏳ 对账历史落库、告警通知、特征增量缓存及完整备用源适配将在后续切片实现。

## 4. 里程碑时间建议

| 里程碑 | 建议排期 | 关键交付 |
|---|---|---|
| M3.0 快速修补 | 第 1 天 | ECharts 本地化、版本徽标（本次已做）、挖掘报告页 |
| M3.1 LLM 离线评估 | 第 2–4 天 | llm_eval 报表 + 三通道对比 |
| M3.2 LLM 通道完善 | 第 3–4 天（与 M3.1 并行） | evidence/回测明细注入 |
| M3.3 挖掘增强 | 第 5–7 天 | LightGBM 挖掘 + 挖掘报告 |
| M3.4 规律研究台 | 第 8–9 天 | 触发时间线 + 多选对比 |
| M3.5 收尾回归 | 第 10–12 天 | 全链路回归 + 验收 |
| M4 各项 | W3 起每项 1–3 天，按 P0→P1→P2 滚动 | 见第 3 节 |

## 5. 风险与决策规则

| 风险 | 缓解 |
|---|---|
| LLM 评估成本高（120 期 × 多模型 × 多注） | 抽样窗口 60 期起步、注数 5、评估期间用轻量模型；成本逐期记录并可视化 |
| 挖掘管道生产看起来显著的假规律 | 三段切分 + BH 校正 + 测试集结果才入库；任何入库规律带完整回测参数 |
| 在线累积报表初期样本不足 | 前端标注累计 N 期（样本不足置信区间宽），满 60 期才给出建议开关 |
| 多源对账告警噪音 | 对账规则白名单化（期号缺失/号码差异分级别），静默降级 + 汇总日报 |
| 迁移机制引入新 bug | 迁移脚本先于业务代码执行 + 启动自检（schema_version 校验），失败即停 |

**决策规则（统一口径）**：新增方法/通道/规律，只有满足
「样本外 120 期 + paired p<0.05（校正后）或作为结构配平工具（非命中率承诺）」二者之一才允许进入默认配置；
其余只作为研究模式开关存在，UI 明示。

## 6. 建议本周执行顺序

1. M3.0（快速修补，半天）——立即可见；
2. M3.1 LLM 离线评估（3 天）——本阶段最有说服力的交付；
3. M3.3 挖掘增强（3 天）——延续规律持续产出主线；
4. M3.4 研究台（2 天）——把已有 backtest_json 数据可视化；
5. M3.2/3.5 穿插完成，最后全链路回归。
