#!/bin/sh
# 容器启动入口：初始化数据 → 跑一次回测 → 视需生成首期预测 → 启动 Web 服务
set -e

echo "[boot] 同步开奖数据..."
python -m lottery.cli fetch || echo "[boot] 抓取失败，稍后由开奖日调度自动重试"

if [ -f /app/data/mining_artifact.json ]; then
  echo "[boot] 导入本机挖掘产物..."
  python -m lottery.cli import_mining /app/data/mining_artifact.json
else
  echo "[boot] 未发现本机挖掘产物，跳过自动挖掘"
fi

echo "[boot] 运行规律回测..."
python -m lottery.cli backtest || true

if [ "${LOTT_BOOT_PREDICT:-1}" = "1" ]; then
  echo "[boot] 检查是否需要生成预测..."
  python - <<'PYEOF'
import os
from lottery import db, backtest as BT
draws = db.load_draws()
if not draws:
    print("[boot] 本地暂无数据，跳过预测")
else:
    issue = BT.next_issue(draws[-1]["issue"])
    if db.load_predictions(issue):
        print(f"[boot] {issue} 已有预测，跳过")
    else:
        print(f"[boot] 未找到 {issue} 的预测，开始生成（LLM 推理约 1-2 分钟）...")
        from lottery import engine
        engine.predict_next(draws, use_llm=None)
        print(f"[boot] {issue} 预测已生成")
PYEOF
fi

echo "[boot] 启动 Web 服务 :${PORT:-18000}"
exec python -m lottery.cli serve --host 0.0.0.0 --port "${PORT:-18000}"