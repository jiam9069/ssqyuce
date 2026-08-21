# 项目长期开发上下文

> 本文档是后续开发的工作记忆，项目根目录为 `D:\AI\Deepseek-harness\CP`。

## 1. 项目定位

双色球智能预测分析系统：Python/FastAPI 后端、SQLite 持久化、Docker Compose 部署、静态 HTML/CSS/JavaScript 前端。系统遵守诚实边界：双色球开奖是独立随机事件；任何方法、规律或 LLM/ML 通道都必须通过样本外回测、随机基线和配对检验，多重检验校正后才能讨论是否保留。系统不承诺中奖率，不提供投注建议。

## 2. 当前发布状态

- 发布版本：`v0.8.1`
- 构建标识：`2026-08-M4.5`
- 当前主分支提交：`a42f0ea docs: publish v0.8.1 M4.5 release notes`
- GitHub：`https://github.com/jiam9069/ssqyuce`
- BF.US 部署目录：`/opt/ssqyuce`
- BF.US 服务：Docker Compose，端口 `18000`
- 最近验证：容器 `healthy`，`/api/info` 返回 v0.8.1 / 2026-08-M4.5。

## 3. 已完成能力

### M1-M3

- 单页 Tab 工作台：预测中心、数据分析、规律研究、历史回放、评估报告、数据管理、设置。
- 统计规律库、walk-forward 样本外回测、Wilson CI、t/sign/Wilcoxon 检验、BH 校正、A/B/C 分级。
- GBDT + RandomForest 概率模型、Brier 加权融合、概率校准、ML walk-forward 评估。
- LLM 三阶段链路、结构化 evidence/counter_evidence/structure_scores、LLM 离线三通道评估及成本估算。
- 规律挖掘、特征重要性、mining_runs、研究台红牌预警/多选对比、历史回放诊断。

### M4.1

- `eval_meta` 不可变预测/配置快照。
- `eval_details` 逐期逐方法逐注明细，幂等在线对照。
- 累计评估 API、10/30/60 滚动指标、近似 95% CI、CSV 导出。

### M4.2

- `LOTT_METHODS` 支持空值/all、allow-list、`-` deny-list、族名/全方法名。
- `LOTT_METHOD_MODE=production/research`；research 忽略开关进行全方法对比。
- 方法配置校验、原子持久化到 `data/methods_config.json`、启动时校验加载。
- 预测结果含 `requested_tickets`、`actual_tickets`、`shortfall_reason`。
- 方法建议 API：paired sign-test 对比 uniform；只提示、不自动关闭。

### M4.3

- `lottery/notify.py` 支持 Webhook/SMTP。
- 在线对照和调度器触发通知；通道隔离、失败静默降级。
- `/api/notify/status`。

### M4.4

- `lottery/data_check.py`：备用源抓取、期号和开奖号码对账，只读不覆盖主库。
- `draws.source` 来源追溯，主源默认 `17500`。
- `reconcile_runs` 对账审计表。
- `/api/data/reconcile` 与 `/api/data/reconcile/history`。
- 当前仍未完成：完整来源适配器、对账告警通知、特征增量缓存。

### M4.5 首个切片

- `lottery/migrations.py`：有序、幂等 SQLite 迁移，`schema_version` 当前版本 2。
- 存量列/表迁移：`draws.source`、`predictions.evidence_json`、`reconcile_runs`。
- `LOTT_TOKEN` 配置后保护全部 `/api/*`，未配置保持开放兼容。
- `/api/health` 提供 `uptime_seconds`、`api_auth_enabled`、最近 5 条任务。
- Docker Compose 和 `.env.example` 已透传配置。

## 4. 关键文件

- `lottery/config.py`：路径、版本、环境变量、运行时配置。
- `lottery/db.py`：SQLite schema、迁移调用、开奖/预测/评估/任务/审计存取。
- `lottery/migrations.py`：schema migration registry。
- `lottery/engine.py`：预测候选生成、方法开关和 shortfall。
- `lottery/evaluate.py`：离线/在线评估。
- `lottery/api_app.py`：FastAPI 路由、鉴权中间件、健康检查、调度器。
- `lottery/notify.py`：通知通道。
- `lottery/data_check.py`：多源对账。
- `README.md`：用户文档、配置、API、版本记录。
- `docs/M3_M4_PLAN.md`：M3/M4 规划、实施状态、验收与风险。
- `tests/`：回归测试，当前本地全套 `30 passed, 1 warning`。

## 5. 运行与交付约定

1. 工作目录固定为 `D:\AI\Deepseek-harness\CP`；不要把 DSH checkout 当作项目目录。
2. 使用 `read`/`edit`/`write`/`glob`/`grep` 检查和修改项目文件；不要提交运行态密钥和数据：`.env`、`data/llm_config.json`、`data/methods_config.json`、SQLite/raw 数据。
3. 每次较大修改必须本地运行 `python -m pytest -q`、compileall、前端 `node --check`、`git diff --check`。
4. 提交后同步 GitHub；若本机 HTTPS 不通，可通过 BF.US SSH Git remote 推送。之后在 BF.US `/opt/ssqyuce` 拉取/重建/验证。
5. BF.US 首次重启后可能因 fetch/backtest/ML 预热需要数分钟；不要把启动阶段的 curl reset 直接当成失败，应查看 `docker compose logs`，等待容器 `healthy` 后再验证 `/api/health`、`/api/info`。
6. Token/SMTP/Webhook 只写服务器 `.env`，最终回复不得展示秘密。

## 6. 下一步优先级

P0：
- 修正长任务状态/清理机制，避免 `tasks` 中永久 running 或历史结果过大；补任务列表/详情 API 的可观测性。
- 补生产级备份与恢复校验，覆盖 SQLite、LLM/runtime config、raw 快照和前端静态资源。
- 增加最近抓取、在线评估、通知、对账状态到 health/运维摘要。

P1：
- 多源对账分级告警、可选通知和对账日报；增加真实备用源适配器。
- 特征按 issue 增量缓存，避免每次全量重算。
- 通知幂等、审计和重试策略。
- M4.2 方法建议完善 BH 校正、决策历史和人工确认记录；禁止自动关闭生产方法。

P2：
- 长任务统一异步化，完善任务取消/超时/重试。
- 更完整的 schema migration 测试和启动自检。
- 反向代理/HTTPS/限流文档与公网安全加固。

## 7. 验收原则

- 不能把随机波动表述为稳定命中率提升。
- 不能宣称 M4.1 真实数据验收完成，除非有开奖后的真实 `online_check` 和累计报告证据。
- 每次代码改动都要同时更新 README/规划文档/版本字段（如影响发布状态）。
- 完成后必须记录本地测试、GitHub 提交、BF.US 部署提交和远程健康验证结果。
