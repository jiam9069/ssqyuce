# VPS 部署方案（Docker Compose · 长期运行 · 端口 18000）

本方案把一个"轻量 + 自动调度"的实例长期跑在 VPS 上：

- **资源需求极低**：1 vCPU / 512MB–1GB 内存 / 1GB 磁盘即可（LLM 推理走远端 API，本地无 GPU 负担）；
- **长期运行**：`restart: unless-stopped` 开机自启 + 崩溃自动拉起；日志自动轮转；
- **自动闭环**：开奖日（周二/四/日）21:35（Asia/Shanghai）自动「抓取 → 在线对照 → 生成下期预测」；无下期预测时启动即自动生成；
- **数据持久化**：`./data` 目录挂载卷，数据库 + 原始快照都在里面，备份 = 拷贝该目录。

## 一、VPS 准备

```bash
# Ubuntu / Debian：安装 Docker + Compose 插件
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # 重新登录后生效（或直接用 sudo）
docker --version && docker compose version
```

若 VPS 无法访问 `get.docker.com`，用发行版源：
`sudo apt update && sudo apt install -y docker.io docker-compose-v2`

## 二、代码上服务器

```bash
# 方式 A：git（推荐，便于日后 docker compose 一键升级）
cd /opt && git clone <你的仓库地址> ssq && cd ssq

# 方式 B：从本机 rsync 同步
rsync -av --exclude=.venv --exclude=.git /root/harness/CP/ root@<VPS>:/opt/ssq/
```

## 三、迁移本地数据（推荐，可"热启动"）

把本机已积累的数据（3490 期开奖 + 预测记录）带过去，避免首启重新抓取：

```bash
rsync -av /root/harness/CP/data/ root@<VPS>:/opt/ssq/data/
```

> 跳过此步也可以：容器首次启动会自动抓取全量数据并生成首期预测（约 2–3 分钟）。
> 注意：`data/raw` 快照、`data/ssq.db` 都随目录一起同步。

## 四、构建并启动

先配置 LLM 凭据（**仓库不含任何 URL/Key**，全部走 `.env`，该文件已 gitignore）：

```bash
cd /opt/ssq
cp .env.example .env
# 编辑 .env，填入你自己的 LLM 通道（DeepSeek/智谱/通义等任意 OpenAI 兼容服务均可）
#   LOTT_LLM_BASE_URL=...  LOTT_LLM_API_KEY=...  LOTT_LLM_MODEL=...
# 多模型可选：LOTT_LLM_MODEL_LIST=模型A,模型B  或  LOTT_LLM_EXTRA_MODELS=[{...}]
```

```bash
docker compose up -d --build          # 首次构建约 2–5 分钟（拉取 python:3.11-slim + pip 依赖）
docker compose logs -f --tail 50      # 观察启动日志（首次含 数据同步→回测→生成预测）
```

验证：

```bash
curl http://<VPS_IP>:18000/api/health
# {"status":"ok","issues":3490,"max_issue":"2026093"}
```

浏览器打开 `http://<VPS_IP>:18000` 即可使用。

## 五、防火墙 / 安全组

```bash
# 主机防火墙（若有 ufw）
sudo ufw allow 18000/tcp && sudo ufw enable
```

阿里云/腾讯云/AWS 等需在**安全组**里放行 TCP 18000。

## 六、本机挖掘并发布到 BF.US

VPS 不执行自动挖掘。挖掘在本机完成后，只提交 `data/mining_artifact.json`；数据库、原始快照和密钥仍不会进入 Git。

### Windows 本机（PowerShell）

```powershell
# 在工作区 D:\AI\Deepseek-harness\CP 执行
.venv\Scripts\python.exe -m lottery.cli fetch
.venv\Scripts\python.exe -m lottery.cli mine --engine rf --min-start 300 --top-k 8
.venv\Scripts\python.exe -m lottery.cli import_mining data\mining_artifact.json  # 可选：验证可导入

git add lottery data/mining_artifact.json entrypoint.sh DEPLOY.md
git commit -m "chore: publish local mining artifact"
git push origin main
```

`mine` 会更新本机 SQLite，同时生成 `data/mining_artifact.json`。该文件仅包含挖掘规律、统计显著性和来源期号，不包含开奖数据库或凭据。

### BF.US 远程重建

```bash
cd /opt/ssqyuce
git pull --ff-only origin main
docker compose up -d --build
docker compose logs --tail 80 lottery
curl -fsS http://127.0.0.1:18000/api/health
```

容器启动时会自动导入 `/app/data/mining_artifact.json`；重复重建是幂等的。若本次不发布产物，启动会明确跳过挖掘，不会在 VPS 上自动运行 `mine`。

## 七、长期运行与运维

| 事项 | 操作 |
|---|---|
| 查看日志 | `docker compose logs -f --tail 100`（日志已轮转 10MB×3，不会撑爆磁盘） |
| 手动刷数据 | `docker compose exec lottery python -m lottery.cli fetch` |
| 手动生成预测 | `docker compose exec lottery python -m lottery.cli predict` |
| 重启服务 | `docker compose restart` |
| 升级版本 | `git pull && docker compose up -d --build`（数据库在 `./data`，不受影响） |
| 服务器重启 | 无需操作，`restart: unless-stopped` 自动拉起 |

### 备份（建议 cron 每日）

```bash
# /etc/cron.d/ssq-backup
0 3 * * * root tar -czf /backup/ssq_$(date +\%F).tar.gz -C /opt/ssq data && find /backup -name 'ssq_*.tar.gz' -mtime +14 -delete
```

恢复：`docker compose down && tar -xzf /backup/ssq_xxx.tar.gz -C /opt/ssq && docker compose up -d`。

## 八、公网访问：反代 + HTTPS（可选，推荐）

主机 18000 端口**不直接暴露公网**，改用反向代理 + 自动 HTTPS：

Caddy（一条命令搞定证书）：

```caddyfile
# /etc/caddy/Caddyfile
ssq.example.com {
    reverse_proxy 127.0.0.1:18000
}
```

```bash
sudo apt install -y caddy && sudo systemctl restart caddy
```

Nginx：

```nginx
server {
    listen 443 ssl;
    server_name ssq.example.com;
    # ssl_certificate ...（如用 certbot）
    location / { proxy_pass http://127.0.0.1:18000; proxy_set_header Host $host; }
}
```

> 此时主机只需放行 80/443，18000 仅监听 127.0.0.1。

## 九、常见问题

- **端口占用/想换端口**：改 `docker-compose.yml` 中 `"18000:18000"`（左侧），或仅改左侧为 `"新端口:18000"` 保持容器内不变；
- **LLM 未生效/调用失败**：仓库不含任何模型凭据，需先在 `/opt/ssq/` 创建 `.env`（`cp .env.example .env` 后填写你自己的 `LOTT_LLM_BASE_URL` / `LOTT_LLM_API_KEY` / `LOTT_LLM_MODEL`），然后 `docker compose up -d`。确认 VPS 能访问该 API 域名；可临时设 `LOTT_LLM_DISABLED=1` 降级为纯统计模型排查；
- **多模型**：同一通道多模型用 `LOTT_LLM_MODEL_LIST=模型A,模型B`；完全独立的通道（不同 URL/Key）用 `LOTT_LLM_EXTRA_MODELS=[{...}]`；
- **预测很慢/超时**：推理型模型（如 deepseek 系列的 reasoner）单期约 2–4 分钟属正常；换轻量模型（如 minimax-m3）较快；
- **时区**：已设 `TZ=Asia/Shanghai`，调度以北京时间为准；
- **合规**：页面与输出已含免责声明，勿对外声称"可预测中奖"；建议反代 + 简单鉴权（Caddy `basic_auth`）后再公网开放。

## 十、架构速览（容器内）

```
entrypoint.sh（启动：同步数据→回测→按需生成预测）
  └─ python -m lottery.cli serve  (uvicorn, :18000)
      ├─ 单页前端（web/，ECharts CDN，零构建）
      ├─ REST API（lottery/api_app.py）
      └─ 调度线程（LOTT_SCHEDULER=1：开奖日 21:35 后自动闭环）
数据卷：/app/data（ssq.db + raw 快照）
```