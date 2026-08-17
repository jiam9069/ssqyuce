"""全局配置：路径、多尺度窗口、LLM 通道（OpenAI 兼容，多模型可自定义）。

安全约定：仓库中不保存任何 LLM API URL / Key，全部通过环境变量注入。
配置方式见 .env.example（Docker Compose 自动读取 .env；本地运行请先 source）。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

BASE = Path(os.environ.get("LOTT_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("LOTT_DB", str(DATA_DIR / "ssq.db")))

DATA_URL = "http://e.17500.cn/getData/ssq.TXT"

# ---------- 版本信息（前端主页 / API / GitHub 说明统一引用） ----------

APP_VERSION = "0.7.0"          # 语义化版本：M1 工作台 + M2 模型升级 + M3 研究闭环已上线
APP_BUILD = "2026-08-M3"       # 构建标识（M4 规划中，见 docs/M3_M4_PLAN.md）
APP_MILESTONES = {
    "M1": {"status": "done",    "desc": "前端 Tab 工作台 + 规律库扩容 29 条 + 自动挖掘管道 + 任务系统"},
    "M2": {"status": "done",    "desc": "GBDT/RF 概率模型 + 滚动 Brier 加权融合 + 概率校准 + 蓝球独立投票 + ML walk-forward 评估"},
    "M3": {"status": "planned", "desc": "研究闭环：LLM 三轮辩论完善与离线评估、挖掘管道增强、规律研究台"},
    "M4": {"status": "planned", "desc": "长期运营：在线累积报表、方法 A/B 开关、通知、多源对账、运维工程"},
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
N_TICKETS = int(os.environ.get("LOTT_N_TICKETS", "10"))             # 最终输出注数
LLM_TIMEOUT = float(os.environ.get("LOTT_LLM_TIMEOUT", "60"))

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
BACKTEST_MIN_N = int(os.environ.get("LOTT_BT_MIN_N", "30"))   # 最少触发样本量
BACKTEST_HORIZON = int(os.environ.get("LOTT_BT_HORIZON", "1"))  # 结果检验窗口（期）

# 离线评估
OFFLINE_EVAL_ISSUES = int(os.environ.get("LOTT_OFFLINE_EVAL_ISSUES", "120"))

# ---------- M2 ML 概率模型（GBDT + 随机森林 + 校准） ----------

ML_ENABLED = os.environ.get("LOTT_ML_ENABLED", "1") == "1"          # 是否把 ML 概率接入集成融合
ML_MIN_START = int(os.environ.get("LOTT_ML_MIN_START", "300"))      # 最少历史期数才开始训练
ML_REFIT_EVERY = int(os.environ.get("LOTT_ML_REFIT_EVERY", "10"))   # 滚动评估时每 N 期重训一次
ML_EVAL_WINDOW = int(os.environ.get("LOTT_ML_EVAL_WINDOW", "60"))   # ML 滚动评估窗口（期）
ML_N_ESTIMATORS = int(os.environ.get("LOTT_ML_N_ESTIMATORS", "60")) # 树数量（RF 与 GBDT 通用上限）
ML_MAX_DEPTH = int(os.environ.get("LOTT_ML_MAX_DEPTH", "5"))        # 树深度
ML_CAL_CV = int(os.environ.get("LOTT_ML_CAL_CV", "2"))              # 概率校准折叠数（小=快）

# 调度（开奖日：周二/四/日 21:35 后自动 抓取+评估+预测）
SCHEDULER_ENABLED = os.environ.get("LOTT_SCHEDULER", "0") == "1"
DRAW_WEEKDAYS = (1, 3, 6)  # 周一=0 … 周日=6 -> 周二/四/日
DRAW_TIME = "21:35"


# ---------- 运行时 LLM 配置持久化（Web 界面写入，优先于 .env） ----------

LLM_CONFIG_FILE = DATA_DIR / "llm_config.json"


def load_runtime_llm_config() -> None:
    """启动时读取 data/llm_config.json（若存在），覆盖 LLM 通道配置。"""
    global LLM_DISABLED, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MODEL_LIST, LLM_SAMPLES
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
        if LLM_MODEL not in LLM_MODEL_LIST:
            LLM_MODEL_LIST = [LLM_MODEL] + list(LLM_MODEL_LIST)
    if isinstance(conf.get("samples"), int) and conf["samples"] > 0:
        LLM_SAMPLES = int(conf["samples"])
    print(f"[config] 已加载运行时 LLM 配置（{LLM_CONFIG_FILE.name}）")

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 加载运行时配置（须在 DATA_DIR 创建后）
load_runtime_llm_config()
