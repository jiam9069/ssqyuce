# M3 版本开发方案（v0.7.0 · 研究闭环）

> 对应 [docs/M3_M4_PLAN.md](M3_M4_PLAN.md) 第 2 节「M3 研究闭环」的**工程级落地文档**：
> 给出文件级任务分解、DB/API/前端变更、排期与验收标准。
>
> 前置：M1 / M2 已上线（v0.6.0）；目标版本 **v0.7.0（build 2026-08-M3）**。
>
> 项目诚实边界：双色球开奖是独立随机事件。M3 **不承诺提升命中率**，目标是把
> 「无规律 / 无显著差异 / LLM 值不值开」量化为**可复现、可在 UI 查看的证据**。

---

## 1. 目标与范围

### 1.1 本次目标
1. **回答「LLM 值不值得开」**：60–120 期 walk-forward 三通道对比（stat / stat+llm / random），
   paired 显著性检验 + 每期成本；
2. **规律可持续产出与证伪**：挖掘管道升级 LightGBM 特征重要性，自动生成候选规律并走回测入库闭环，产出挖掘报告；
3. **研究过程可视化**：规律触发时间线、滚动边际曲线、多选对比、挖掘报告页；
4. **全链路无回归**：fetch → backtest → predict → 在线对照。

### 1.2 非目标（Out of Scope）
- 多数据源对账、开奖通知、方法 A/B 开关、schema 迁移框架 → 归入 M4；
- 任何「提升命中率」的承诺；
- 前端构建链引入（保持零构建 SPA）。

### 1.3 版本与发版
- `APP_VERSION`：0.6.0 → **0.7.0**；`APP_BUILD`：2026-08-M2 → **2026-08-M3**；
- `APP_MILESTONES["M3"]`：开发中 →（完成时）「已上线」；
- 资产缓存号：`?v=20260816.5` → **`?v=20260816.6`**（与 build 对齐，见 M3.0）。

---

## 2. 里程碑与排期（10–12 天）

| 里程碑 | 排期 | 内容 | 优先级 |
|---|---|---|---|
| M3.0 工程修补 | D1 | 资产版本统一、CLI doctor、评估/挖掘任务化入口 | P0 |
| M3.1 LLM 离线评估 | D2–D5 | llm_eval.py + 报表 + 成本 | P0 |
| M3.2 LLM 通道完善 | D4–D5（与 M3.1 并行） | evidence / 回测明细注入 / 校验轮 | P1 |
| M3.3 挖掘管道增强 | D6–D8 | LightGBM + mining_runs + 挖掘报告 | P0 |
| M3.4 规律研究台 | D9–D10 | 触发时间线 + 多选对比 | P1 |
| M3.5 收尾回归 | D11–D12 | 诊断导出 + 全链路回归 + 验收 | P2 |

---

## 3. 任务分解（文件级）

### M3.0 工程修补（0.5 天，P0）

| # | 任务 | 落点 |
|---|---|---|
| 1 | 资产版本统一：index.html 资源 `?v=` 与 `/api/info.build` 对齐，纳入发版 checklist | web/index.html、lottery/config.py |
| 2 | 新增 CLI doctor：校验缓存号与 build 一致、vendor 文件存在、APP_VERSION 三处一致 | lottery/cli.py |
| 3 | 新增 M3 配置项：`LLM_EVAL_ISSUES`(60) / `LLM_EVAL_TICKETS`(5) / `LLM_EVAL_SEED`(42) / `LLM_EVAL_MODEL` | lottery/config.py、.env.example |
| 4 | 长任务统一走 tasks 框架（llm-eval、mine 改为异步任务） | lottery/api_app.py |

> 注：ECharts 本地化、主页版本徽标、/api/info 已在 v0.6.0 完成，M3 不再重复。

### M3.1 LLM 离线评估（2–4 天，P0）

**新文件**：`lottery/llm_eval.py`

**流程**：
1. 取最近 N 期（默认 60，可配 120）做 walk-forward 重放；
2. 每期三通道并行产出：
   - `stat`：现有纯统计 + ML 流程（use_llm=False）；
   - `stat_llm`：现有完整流程（M3.2 后含 evidence 新 schema）；
   - `random`：同注数随机选号（固定种子，结果可复现）；
3. 每期记录：红球命中数 / 蓝球命中 / 五等及以上 / 奖金（固定奖级表）/ ROI / token 用量 / 耗时 / 估算费用；
4. 汇总：三通道指标表 + Δ(stat_llm − stat) + paired 检验（wilcoxon / sign-test，复用 `ml_model._pair_p`）+ BH 校正；
5. 落库 `llm_eval_results`；固定 seed，重复运行一致。

**DB 变更**（db.py 建表，幂等 CREATE IF NOT EXISTS）：

```sql
CREATE TABLE IF NOT EXISTS llm_eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,              -- 如 eval_20260817_001
  created_at TEXT NOT NULL,
  window_issues INTEGER NOT NULL,    -- 窗口期数
  tickets INTEGER NOT NULL,
  channel TEXT NOT NULL,             -- stat | stat_llm | random
  issue TEXT,                        -- 逐期明细行填期号；汇总行为空
  red_hits INTEGER, blue_hit INTEGER, prize_level INTEGER, roi REAL,
  tokens INTEGER, cost_usd REAL, duration_ms INTEGER,
  metrics_json TEXT,                 -- 汇总指标（汇总行）
  p_values_json TEXT,                -- paired 检验结果（汇总行）
  seed INTEGER
);
```

**API**：
- `POST /api/eval/llm/run`（任务化，返回 task_id；参数 issues / tickets / seed）
- `GET /api/eval/llm/latest`（最近一次 run 的汇总 + 明细）
- `GET /api/tasks/<id>`（已有，用于进度轮询）

**前端**（评估报告 Tab）：
- 「LLM 离线评估」卡片：三通道柱状图（红球命中 / 蓝球命中 / ≥五等率 / ROI）+ 逐期累积曲线 + 成本行 + p 值徽标（显著 / 不显著）；
- 结论行无论正负都展示，如「stat+llm 相对 stat：Δ=+0.02，p=0.43 → 无显著差异」。

**验收**：一页报表能回答 LLM 的 Δ 与 p 值及每期成本；固定 seed 复跑两次结果一致。

### M3.2 LLM 通道完善（1–2 天，P1）

| # | 任务 | 落点 |
|---|---|---|
| 1 | 提示词注入规律回测明细表：n / 边际 / p_adj / 等级 / Wilson CI（Top-K 条） | lottery/llm_client.py（prompt 构建） |
| 2 | 提示词注入上期预测回馈（命中情况摘要） | lottery/llm_client.py |
| 3 | 输出 schema 扩展：`evidence` / `counter_evidence` / `structure_scores` | schema + predictions 表 |
| 4 | predictions 表追加 `evidence_json` 列（缺失则幂等 ALTER） | db.py |
| 5 | 第三轮校验默认开启：低温度 + `LOTT_LLM_VERIFY_MODEL` 指定不同模型 | lottery/config.py + engine 默认值 |
| 6 | 前端票卡「展开依据」：展示 evidence / counter_evidence / structure_scores | web/static/app.js、web/index.html |

### M3.3 规律挖掘管道增强（2–3 天，P0）

| # | 任务 | 落点 |
|---|---|---|
| 1 | 特征集 20 → 40+：合并 ml_model 特征 + 遗漏区间 / 邻号 / 区间邻居 / 位置模式 | `lottery/features.py`（新） |
| 2 | 挖掘器支持 LightGBM（sklearn/RandomForest 保底，numpy 兜底降级） | lottery/mining.py |
| 3 | 特征重要性 Top-K → 自动生成候选规律文本（模板：号码 x 在 y 遗漏区间后出现率 lift） | lottery/mining.py |
| 4 | 候选规律走现有 backtest 框架（walk-forward + BH），B 级及以上才入库 | 复用 lottery/backtest.py |
| 5 | `mining_runs` 表：run_id / engine / 候选数 / 入库数 / 平均 lift / 合格率 / 耗时 | db.py |
| 6 | CLI：`python -m lottery.cli mine --engine lightgbm --top-k 20` | lottery/cli.py |
| 7 | API：`GET /api/mining/reports`（历史对比）、`GET /api/mining/candidates` | lottery/api_app.py |
| 8 | 前端：规律研究页新增「挖掘报告」子页签 | web/ |

**DB 变更**：

```sql
CREATE TABLE IF NOT EXISTS mining_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  engine TEXT NOT NULL,
  candidates INTEGER, accepted INTEGER,
  avg_lift REAL, pass_rate REAL,
  duration_ms INTEGER, params_json TEXT
);
```

### M3.4 规律研究台增强（1–2 天，P1）

| # | 任务 | 落点 |
|---|---|---|
| 1 | API：`GET /api/patterns/<id>`（详情 + 触发时间线，读 backtest_json 逐期序列） | lottery/api_app.py |
| 2 | API：`GET /api/patterns/compare?ids=…`（多规律边际对比） | lottery/api_app.py |
| 3 | 前端规律详情：触发时间线（高亮命中/未命中）+ 逐期命中 vs 期望滚动曲线 | web/static/app.js |
| 4 | 前端列表多选（checkbox）→ 边际对比叠柱图 | web/static/app.js |
| 5 | 证伪规律红牌徽标（BH 后不显著 / p_adj ≥ 0.05） | web/static/app.js、style.css |

### M3.5 回放与诊断收尾（1 天，P2）

| # | 任务 | 落点 |
|---|---|---|
| 1 | 历史回放：每期展示当时所用方法（读 predictions.method） | lottery/api_app.py、web |
| 2 | 号码诊断：相似注检索 Top-K 明细（结构距离 + 历史命中分布） | lottery/api_app.py 新增 `/api/diagnose/similar` |
| 3 | 诊断结果导出 CSV | web |

---

## 4. 版本与配置变更汇总

| 项 | 变更 |
|---|---|
| lottery/config.py | `APP_VERSION=0.7.0`、`APP_BUILD=2026-08-M3`、`APP_MILESTONES["M3"]="已上线"`（发版时）；新增 `LLM_EVAL_*`、`LOTT_LLM_VERIFY_MODEL` |
| requirements.txt | 尝试新增 `lightgbm`（需评估镜像体积；安装失败/过大可用 sklearn 特征重要性兜底，lightgbm 标为可选） |
| db.py | 建 `llm_eval_results`、`mining_runs`；`predictions` 幂等加 `evidence_json` 列 |
| web/index.html | 资源 `?v=20260816.6`；新增挖掘报告子页签入口；票卡依据展开区 |
| web/static/app.js | 评估报告 LLM 卡片、规律时间线/对比、挖掘报告、诊断导出 |
| web/static/style.css | 时间线/对比图、红牌徽标、成本表样式 |
| .env.example、docker-compose.yml | 透传 `LLM_EVAL_*`、`LOTT_LLM_VERIFY_MODEL` |

---

## 5. 验收标准（Exit Criteria）

1. LLM 离线评估报表可在 UI 查看：三通道 + paired p 值 + 每期成本，结论明确「值 / 不值 / 无显著差异」；
2. 挖掘管道在现有数据上产出 ≥1 条 B 级及以上新候选，且回测参数可追溯（三段切分 + BH）；
3. 进入 UI 的任何新规律 / 新结论都带随机基线与 p 值徽标；
4. 规律详情可查看触发时间线与滚动边际曲线；多选对比可用；
5. 全链路回归：fetch → backtest → predict → 在线对照无回归；
6. 版本一致性：`/api/health.version` == 页脚 == README == **0.7.0**。

---

## 6. 部署与回归流程

1. 本地改代码 → `python -m compileall lottery` 语法检查；
2. 上传变更文件到 BF.US `/opt/ssqyuce`（保留目录结构）；
3. `docker compose up -d --build`；如遇容器名冲突：`docker rm -f ssq-predictor && docker compose up -d`；
4. 轮询 `/api/health`（启动需 5–15 分钟：fetch + backtest 预热）；
5. 触发 `llm-eval` 任务 → 轮询 `/api/tasks/<id>` → 评估报告 Tab 核对图表与 p 值；
6. 挖矿回归：`python -m lottery.cli mine --engine lightgbm` → 检查 mining_runs 报告；
7. 完整回归：跑一次 predict（use_llm=False / True）与在线对照；
8. 提交并推送（server：`git add . && git commit && git push`）。

---

## 7. 风险与决策规则

| 风险 | 缓解 |
|---|---|
| LLM 评估成本高（120 期 × 多模型 × 多注） | 60 期起步、注数 5、评估用轻量模型；成本逐期记录并可视化 |
| LightGBM 引入镜像膨胀 / 安装失败 | requirements 可选；sklearn 特征重要性兜底；镜像体积超限则拆层 |
| 挖掘产出外表显著的假规律 | 三段切分 + BH 校正 + 测试集结果才入库 |
| 报表初期样本不足 | 前端标注累计 N 期；满 60 期才下结论 |
| 长任务阻塞 | 全部走 tasks 异步框架，前端轮询进度 |

**统一决策规则**：新增方法 / 通道 / 规律，只有满足「样本外 120 期 + paired p<0.05（校正后）」或
「作为结构配平工具（非命中率承诺）」二者之一才允许进入默认配置；其余只作为研究模式开关存在，UI 明示。

---

## 8. 发版 Checklist（M3 完成时）

- [ ] `APP_VERSION` / `APP_BUILD` / `APP_MILESTONES` 已更新且三处一致（health / 页脚 / README）
- [ ] LLM 离线评估报表生成一版并截图存档
- [ ] mining_runs 最新一版报告存档
- [ ] 规律时间线 / 多选对比手工验证通过
- [ ] README「最近升级概要」追加 v0.7.0（M3）小节
- [ ] docs 三件套（UPGRADE_PLAN / M3_M4_PLAN / M3_DEV_PLAN）同步提交
- [ ] 全链路回归通过，线上 health OK

---

## 9. 实施状态（2026-08-17 逐条核销）

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M3.0 工程奠基 | ✅ | 缓存号统一 v=20260816.6→20260817.7；CLI doctor；config M3 参数块；任务系统复用 |
| M3.1 LLM 离线评估 | ✅ | llm_eval.py + llm_eval_results 表 + POST/GET /api/eval/llm/* + 前端报表卡；3 期冒烟完成（stat 1.333 / stat_llm 0.667 / random 1.0） |
| M3.2 LLM 通道完善 | ✅ | evidence/counter_evidence/structure_scores schema 落库+前端展示；回测明细(n/p_adj/CI)与上期回馈注入 prompt；第三轮校验默认开（LOTT_LLM_VERIFY=1） |
| M3.3 挖掘增强 | ✅ | 特征 20→40 维；LightGBM→RF→lift 自动回退；mining_runs 表+报告 API+前端；冒烟 mine_20260817_164538（8 候选/1 A 级/40 特征） |
| M3.4 规律研究台 | ✅ | 红牌预警（近 20 期边际下滑/连续负边际）；多选对比（边际序列叠加图）；触发时间线（chPatSeries 已有+索引修复） |
| M3.5 回放诊断 | ✅ | POST /api/replay/diagnose 最近 N 期反事实回放；冒烟 20 期红球 1.197/注 vs 随机 1.09；v0.7.0 版本升级三处一致 |
| 工程加固 | ✅ | LLM 单次 600s 硬上限+重试超时 120s；predict_next LLM 异常兜底降级；sklearn n_jobs=1 防死锁；sqlite 侧边文件 gitignore |
| 60 期全量评估 | 🔄 | task llm_eval_6fd26b28 后台跑批中（预计 2~4h）；完成后按决策规则定默认配置 |

提交记录：2f7064f → ad94a24 → de573d6 → e306815 → 0b6b1bd → 7ac185f（推送 github.com:jiam9069/ssqyuce main）