# 双色球智能预测分析系统

> 一款福彩双色球预测系统，基于历史开奖数据，用 AI 分析长期、中期、短期规律，预测下一期开奖数据。

用 **大模型（LLM）+ 统计方法** 分析双色球历史开奖数据，寻找长/中/短期规律，
并对每条规律做 **样本外回测与显著性检验**，最终由「统计基线模型 + LLM 推理」集成输出
**带置信度与推理依据**的下期候选号码。

> ⚠️ **诚实边界**：双色球每期开奖是独立随机事件。本系统的样本外回测结果（见下文）实证了
> **不存在稳定、显著优于随机的规律**（全部规律经多重检验校正后 p 值均未通过 0.05 阈值）。
> 因此本系统的价值不在于"提高命中率"，而在于：
> 1. 用可证伪的方式量化各种流行"规律"的真实证据强度；
> 2. 在约束下生成结构均衡的候选组合与可解释的推理理由；
> 3. 用与随机基线的对照评估，诚实地展示"投注期望回报为负"。
>
> 系统输出仅供研究参考，不构成中奖概率与投注建议。**理性购彩，量力而行。**

## 🚀 最近升级概要（M1-M3 已上线 · M4.1 开发中 · v0.8.0）

- **M1 前端工作台**：单页重构为 Tab 工作台（预测中心 / 数据分析 / 规律研究 / 历史回放 / 评估报告 / 数据管理 / 设置）；
  预测可配置（注数 / LLM 开关）、每注复制 / 收藏 / 展开依据、号码点选统计卡、「我的号码诊断」、历史预测回放；
  规律库 11 → 29 条（跨度 / 龙头凤尾 / 质合比 / 大小比 / 012 路 / 连号 / 同尾 / AC 值 / 遗漏区间 / 号码对共现 / 蓝球独立结构等）；
  自动规律挖掘管道（三段切分防偏差，一次性产出 35 条全维度规律）与任务系统（长操作异步化）上线。
- **M2 模型升级**：新增 GBDT + 随机森林概率模型（33 维红球 / 16 维蓝球，独立建模 + 独立投票）；
  集成融合从等权平均升级为**滚动 Brier 加权**，置信度接入**概率校准**（温度缩放 / sigmoid）；
  新增 ML walk-forward 滚动评估：Brier / log-loss / 校准曲线（可靠性图）/ ECE / **paired 显著性检验**（wilcoxon / sign-test），全程对照均匀随机基线。
- **诚实结论**：M2 评估的目标不是"提高命中率"，而是量化"ML 与随机基线无显著差异"这一事实并做结构配平 ——
  所有新增模型/规律仍须通过样本外回测 + 多重检验校正才能进入 UI（详见下方回测表与 docs/UPGRADE_PLAN.md）。
- **M3 研究闭环**：LLM 离线三通道评估（统计 vs 统计+LLM vs 随机基线，paired + BH 校正，token 成本估算）；
  LLM 生成链路升级（回测规律明细/上期预测回馈/evidence 结构化/第三轮校验）；规律挖掘 40 维特征 +
  LightGBM/RandomForest 重要性 + mining_runs 运行报告；规律研究台新增红牌预警与多选对比；系统回放诊断；
  全程坚持"双色球为独立随机，任何声称的规律都需样本外 + 多重检验校正"的诚实口径。

> 下一步 **M3 研究闭环**、**M4 长期运营** 的完整规划（任务拆分 / 验收标准 / 工作量 / 风险）见
> [docs/M3_M4_PLAN.md](docs/M3_M4_PLAN.md)；版本与里程碑状态也可通过 GET /api/info 与主页页脚徽标查看。

## 特性

- **数据管道**：抓取 `http://e.17500.cn/getData/ssq.TXT`（每期固定 31 字段），增量入库
  （SQLite），保留原始快照；已实测 3490 期（2003001 → 2026093），期号无缺、号码无越界。
- **多尺度特征工程**：长期（全量）/ 中期（近150期）/ 短期（近30期）三窗口，输出频率、
  遗漏、和值、三区比、奇偶、连号/同尾/重号等指标。
- **规律库 + Walk-forward 回测**：35 条规律（长/中/短期，覆盖跨度、龙头凤尾、质合比、大小比、
  012 路、连号、同尾、AC 值、遗漏区间、号码对共现、蓝球独立结构等），仅用目标期之前的历史滚动验证，
  逐触发期计算期望与边际、Wilson 置信区间，t 检验 + Benjamini-Hochberg 多重检验校正，规律分级 A/B/C
  （显著/弱信号/不通过），显著但方向相反者标记"证伪"；另有自动挖掘管道持续产出候选规律。
- **预测引擎**：4 个统计基线模型（频率加权、遗漏回补、马尔可夫转移、贝叶斯更新）+ **M2 ML 概率模型**
  （GBDT + 随机森林，33 维红球 / 16 维蓝球独立建模）→ 滚动 **Brier 加权融合** → 硬约束
  （和值分位、三区比、奇偶分布、号码去重、蓝球分散）+ 概率校准 + LLM 多轮推理（观察 → 选号 → 校验，
  多温度并发采样投票），最终输出 Top-10 注 + 置信度 + 理由。
- **评估闭环**：离线 walk-forward 引擎回测与 **ML 滚动评估**（对照均匀随机基线，输出 Brier /
  log-loss / 校准曲线 / ECE / paired 显著性检验）；开奖后在线对照入库。
- **部署友好**：Python 标准栈，FastAPI 直接托管零构建单页前端（ECharts CDN），
  单进程运行，低资源占用；可选 Docker / 开奖日自动调度。

## 安装部署（新手向导）

> 二选一：**方式一 Docker（推荐，最省事）** 或 **方式二 本地 Python**。
> 两种方式都**可以不配置 LLM 直接运行**（自动降级为纯统计模型），配了 LLM 才有 AI 推理选号。

### 第 0 步：准备

- 一台机器：本地电脑或 VPS 都行，**要求很低**（1 核 CPU / 512MB 内存即可）；
- 可选：一个 OpenAI 兼容的 LLM 服务（DeepSeek / 智谱 / 通义等任意一家），拿到它的 **API 地址 + Key + 模型名**。

### 方式一：Docker Compose（推荐，新手最省事）

```bash
# 1) 安装 Docker（已装可跳过；Ubuntu/Debian 一条命令）
curl -fsSL https://get.docker.com | sh

# 2) 下载项目
git clone https://github.com/jiam9069/ssqyuce.git
cd ssqyuce

# 3) 配置 LLM（可选：不配置也能跑，只是没有 AI 推理）
cp .env.example .env          # 复制模板
vim .env                      # 填写你自己的 LOTT_LLM_BASE_URL / LOTT_LLM_API_KEY / LOTT_LLM_MODEL

# 4) 一键启动（首次约 2-5 分钟：构建镜像 → 自动抓取全部历史数据 → 回测 → 生成首期预测）
docker compose up -d --build

# 5) 打开浏览器
#    本机:  http://localhost:18000      服务器: http://服务器IP:18000
curl http://localhost:18000/api/health   # 返回 {"status":"ok",...} 即成功
```

**日常命令**（都在项目目录执行）：

| 想做什么 | 命令 |
|---|---|
| 看运行日志 | `docker compose logs -f` |
| 重启 | `docker compose restart` |
| 停止 | `docker compose down` |
| 手动刷新数据 + 生成预测 | `docker compose exec lottery python -m lottery.cli fetch` 然后 `... predict` |
| 升级到最新版 | `git pull && docker compose up -d --build`（数据在 `data/` 目录，不会丢） |

### 方式二：本地 Python（不用 Docker）

```bash
# 1) 需要 Python 3.10+；以下命令在项目目录执行
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) 配置 LLM（可选）
cp .env.example .env
set -a && source .env && set +a      # 让 .env 里的环境变量生效（Windows 建议用 WSL / Git Bash）

# 3) 抓数据 → 回测 → 生成预测（按顺序执行一次即可）
.venv/bin/python -m lottery.cli fetch
.venv/bin/python -m lottery.cli backtest
.venv/bin/python -m lottery.cli predict

# 4) 打开 Web 界面 http://localhost:18000
.venv/bin/python -m lottery.cli serve
```

更多命令：`python -m lottery.cli stats`（多尺度统计）、`offline_eval`（离线回测）、
`online_check`（在线对照）。

### 装好之后

- **全自动**：开奖日（周二/四/日）21:35 后系统自动「抓取 → 在线对照 → 生成下期预测」，无需干预（Docker 方式默认开启）；
- **换端口**：改 `docker-compose.yml` 里 `"18000:18000"` 左边那个数字后 `docker compose up -d`；
- **VPS 长期运行**（防火墙、HTTPS、数据迁移、每日备份、故障排查）：见 [DEPLOY.md](DEPLOY.md)。

## Web 界面

单页中文仪表盘（dark 主题，ECharts）：

- **预测区**：Top-10 候选注（红/蓝球圆点、置信度条、方法徽标、LLM 推理理由）；
- **多尺度统计**：33 格红球频率热力、遗漏 Top10、和值走势、三区比堆积（三窗口切换）；
- **规律库**：全部 11 条规律的回测表格（样本量/期望/边际/p 值/adj_p/等级）与边际条形图；
- **历史走势**：最近 20 期开奖号矩阵；
- **评估区**：系统 vs 随机基线的命中率/奖金/ROI 对比与红球命中分布。

## 配置（环境变量）

> **密钥与地址不入库**：所有 LLM URL / Key 只从环境变量或 `.env` 读取（模板见
> `.env.example`），仓库与镜像中不含任何模型凭据。未配置时系统自动降级为纯统计模型。

| 变量 | 默认 | 说明 |
|---|---|---|
| `LOTT_LLM_BASE_URL` | （必填） | OpenAI 兼容 LLM 端点（DeepSeek/智谱/通义等均可） |
| `LOTT_LLM_API_KEY` | （必填） | LLM Key |
| `LOTT_LLM_MODEL` | `minimax-m3` | 主模型 |
| `LOTT_LLM_MODEL_LIST` | 空 | 同一通道多模型，逗号分隔，采样时按模型轮转 |
| `LOTT_LLM_EXTRA_MODELS` | 空 | 附加独立通道，JSON 数组 `[{"name","base_url","api_key","model"}]` |
| `LOTT_LLM_SAMPLES` | `3` | LLM 采样轮数（并发） |
| `LOTT_LLM_DISABLED` | `0` | `1` = 关闭 LLM，仅统计模型 |
| `LOTT_N_TICKETS` | `10` | 输出注数 |
| `LOTT_SCHEDULER` | `0` | `1` = 开奖日（周二/四/日）21:35 后自动 抓取+对照+预测 |
| `LOTT_BOOT_PREDICT` | `1` | 容器启动时若下期无预测则自动生成 |
| `PORT` | `18000` | Web 服务端口（Docker/环境变量） |
| `LOTT_DB` | `data/ssq.db` | 数据库路径 |

**多模型示例**（写入 `.env`）：

```bash
LOTT_LLM_BASE_URL=https://api.example.com/v1
LOTT_LLM_API_KEY=sk-xxx
LOTT_LLM_MODEL_LIST=model-a,model-b          # 同一通道多个模型
LOTT_LLM_EXTRA_MODELS=[{"name":"my-ds","base_url":"https://api.deepseek.com/v1","api_key":"sk-xxx","model":"deepseek-chat"},{"name":"my-glm","base_url":"https://open.bigmodel.cn/api/paas/v4","api_key":"xxx","model":"glm-4.5"}]
```

## 离线回测结果（2026-08-15，样本外，3190 触发点/规律）

| 规律 | 尺度 | avg命中 | 期望 | 边际 | p值(adj) | 等级 |
|---|---|---|---|---|---|---|
| 蓝球冷号回补 | long | 0.058 | 0.062 | -0.004 | 0.95 | C |
| 红球频率均值回归 | long | 0.206 | 0.212 | -0.006 | 0.95 | C |
| 红球遗漏压力 | long | 0.199 | 0.197 | +0.002 | 0.76 | C |
| 低活跃区回补 | mid | 2.047 | 2.000 | +0.047 | 0.058 | B |
| 奇偶比例回归 | mid | 3.011 | 2.996 | +0.015 | 0.68 | C |
| 和值回归 | mid | – | – | -0.263 | 1.00 | C |
| 重号延续 | short | 1.085 | 1.091 | -0.006 | 0.94 | C |
| 邻号延续 | short | 1.921 | 1.953 | -0.032 | 0.96 | C |
| 短期热号 | short | 1.095 | 1.091 | +0.004 | 0.76 | C |
| 短期冷号 | short | 1.107 | 1.091 | +0.016 | 0.50 | B |
| 蓝球延续 | short | 0.068 | 0.062 | +0.006 | 0.48 | B |

**解读**：经 BH 多重检验校正后**没有一条规律达到 0.05 显著**（A 级为空），仅 3 条弱信号（B）。
这与"彩票独立随机"的理论一致——系统的防过拟合框架如愿工作：
它把流行的"冷号回补""均值回归"等说法变成了可证伪的量化结论。

引擎离线回测（60 期 × 10 注/期，对照同注数随机基线）：

| 指标 | 系统 | 随机基线 |
|---|---|---|
| 红球平均命中 | 1.08–1.14 | 1.10 |
| 蓝球命中率 | 6.0–6.7% | 6.3%（=1/16） |
| ≥五等奖命中率 | ≈6.2% | ≈6.7% |
| ROI（2元/注） | **-78% ~ -83%** | **-65%** |

## 项目结构

```
lottery/
  config.py       # 配置（路径/窗口/LLM/调度）
  data_fetcher.py # 抓取 + 增量入库
  parser.py       # 31 字段解析与校验
  db.py           # SQLite 存取（开奖/特征/规律/预测/评估）
  features.py     # 多尺度统计指标
  patterns.py     # 规律库（触发条件 + 候选动作）
  backtest.py     # walk-forward 回测 + BH 校正 + 分级
  models.py       # 统计基线模型（频率/遗漏/马尔可夫/贝叶斯）
  llm_client.py   # LLM 通道（JSON 结构化输出、降级）
  engine.py       # 集成预测引擎（硬约束 + 置信度 + Top-N）
  evaluate.py     # 奖级判定 + 离线/在线评估
  api_app.py      # FastAPI + 开奖日调度
  cli.py          # 命令行
web/               # 单页前端（index.html + static/，ECharts CDN，零构建）
data/              # SQLite 数据库与原始快照（自动生成）
Dockerfile / docker-compose.yml
```

## 长期运行（VPS）

新手快速上手见上文「安装部署 —— 方式一」。上生产 / 长期跑在服务器上需要额外处理：
环境准备、数据热迁移（把本机 `data/` 同步过去，免去首启抓取）、防火墙放行 18000、
反向代理 + HTTPS、每日备份与恢复 —— 完整步骤见 [DEPLOY.md](DEPLOY.md)。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/refresh` | 抓取最新开奖并增量入库 |
| GET | `/api/features` | 最新多尺度统计报告 |
| GET | `/api/patterns` | 规律回测结果 |
| POST | `/api/patterns/backtest` | 重新运行回测 |
| GET/POST | `/api/predict?regenerate=1` | 生成/获取下一期预测 |
| GET | `/api/predictions/last` | 最近一次预测 |
| GET | `/api/draws/history` | 开奖历史（走势图用） |
| POST | `/api/eval/backtest` | 离线引擎回测（约 60-120s） |
| POST | `/api/eval/online` | 在线预测对照 |
| GET | `/api/ml/status` | ML 概率模型状态（是否就绪 / 已训练） |
| POST | `/api/ml/eval` | ML walk-forward 评估（Brier/log-loss/校准/paired，约 2-5 分钟） |
| POST | `/api/mining/run` | 运行规律自动挖掘 |
| GET | `/api/info` | 系统版本与 M1-M4 里程碑状态 |
| GET | `/api/predictions/history` | 历史预测回放（对照实际开奖） |


## 版本记录

### v0.7.0（M3 研究闭环，2026-08 上线）

- **LLM 离线评估**：三通道 walk-forward（统计 / 统计+LLM / 随机基线）60 期 × 5 注，paired 检验（wilcoxon/sign-test）+ BH 多重校正；LLM 失败期如实回退统计输出并统计 llm_empty_issues；token 成本估算（默认 $1/1M）
- **LLM 构建增强**：prompt 注入样本外回测规律明细（n / p_adj / 威尔逊区间）与上期预测回馈；tickets 输出 evidence / counter_evidence / structure_scores 结构化字段并落库展示；第三轮校验（critique → 问题才 refine，默认开启，评估口径关闭）
- **挖掘管道增强**：特征 20 → 40 维（邻号/同尾/同区/龙头凤尾/升降温/共现/冷门集中度）；重要性引擎 LightGBM → RandomForest → numpy lift 自动回退；mining_runs 运行记录 + /api/mining/reports + 前端挖掘报告
- **规律研究台**：红牌预警（近 20 期边际下滑 / 连续负边际）、多选对比（边际时间序列叠加）
- **回放诊断**：最近 N 期「系统预测 vs 实际」反事实回放（纯统计+ML，固定种子）
- **工程加固**：LLM 单次调用总耗时 600s 硬上限防挂起、sklearn 单线程防死锁（n_jobs=1 + OMP/MKL/OPENBLAS 限线程）、sqlite 侧边文件 gitignore

### v0.6.0（M2 模型升级，2026-08 上线）

- **ML 概率模型**：GBDT（HistGradientBoosting）+ 随机森林软投票，33 维红球 / 16 维蓝球独立建模、独立投票
- **融合升级**：集成从等权平均改为滚动 **Brier 加权融合**（含 ML 通道），置信度接入概率校准（sigmoid）
- **ML 评估**：`/api/ml/eval` walk-forward 滚动评估，输出 Brier / log-loss / 校准曲线 / ECE / paired 显著性检验（wilcoxon / sign-test），对照均匀随机基线
- **模型持久化**：训练好的模型缓存到 data 卷，容器重建免重训（数据版本触发重训）
- **ML 预热**：启动后台预热 + 非阻塞 /api/ml/status，首个预测请求不被训练阻塞
- **页面新增「ML 模型评估」按钮与评估摘要卡片**

### v0.5.0（M1 前端工作台 + 规律扩容，2026-08 上线）

- **规律库扩容**：从 11 条扩展到 29 条，覆盖质合比/大小比/012路/连号组/同尾组/龙头凤尾回归/遗漏区间/斜连号/间隔号/号码对共现等维度；后经 M2 补齐至 35 条
- **自动规律挖掘管道**：`python -m lottery.cli mine` 基于特征 lift 自动生成候选规律并回测分级
- **回测增强**：支持 horizon>1（多期检验）、Wilson 置信区间、边际时间序列存储
- **号码诊断 API**：`/api/diagnose` 输入自选号码返回结构画像 + 历史相似注命中分布
- **预测概率输出**：预测结果增加 red_probs/blue_probs（33/16 维模型概率向量）
- **前端 Tab 工作台**：预测中心/数据分析/规律研究/历史回放/评估报告/数据管理 六模块
- **交互增强**：号码球点选查看详情/预测票卡复制收藏/规律筛选排序/边际趋势图
- **任务系统**：tasks 表 + /api/tasks 端点，支持异步长操作

## 已知限制与后续方向

- LLM 通道依赖第三方兼容端点，支持任意 OpenAI 兼容服务与多模型自定义（主通道 + 附加通道）；
- **LLM 离线评估尚未落地**（M3 计划）：目前离线评估默认关闭 LLM，M3 将做 60-120 期 walk-forward
  的「纯统计 / 统计+LLM / 随机」三方对比，量化 LLM 真实贡献与成本；
- 可扩展：更多规律假设（含官方"期号尾数""生肖"等民俗说法，统一送入回测证伪）、
  交易所行情无关；在线累积评估报表、方法 A/B 开关、开奖通知、多源数据对账列入 M4。

## 升级方案 v2.0

针对三大痛点 —— **规律不多 / 准确率不高 / UI 人机交互不佳** —— 的完整升级路线
（规律库扩容到 35+、ML 概率模型与自适应融合、置信度校准、UI 工作台重构等），
见 [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md)。

**当前进度**：M1 ✅（前端工作台）与 M2 ✅（模型升级）已上线；**M3 研究闭环 / M4 长期运营**
的详细规划见 [docs/M3_M4_PLAN.md](docs/M3_M4_PLAN.md)。