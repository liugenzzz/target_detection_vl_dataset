"""全局配置，全部通过环境变量注入。

本地跑：在终端 set/export 环境变量
Docker：改 docker-compose.yml 的 environment
"""

import os


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ---------------- 后端 Qwen 模型服务 ----------------
QWEN_API_URL = os.getenv("QWEN_API_URL", "http://192.168.78.36:3012/v1/chat/completions")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen3.6-27B")
QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "300"))

# 温度别压到 0.1：极低温配上高度重复的 JSON 结构，容易诱发重复生成死循环。
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.35"))

# 输出上限。这是唯一能真正拉住模型的手段 —— 实测模型无视 prompt 里的数量限制，
# 密集场景会一直吐到上限为止。生成速度约 35 token/秒，这个值直接决定难图的最坏耗时。
# 60 个框的 JSON 约 2700 token，设 3000 比较合适（难图最坏约 85 秒）。
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "3000"))

QWEN_ENABLE_THINKING = _as_bool(os.getenv("QWEN_ENABLE_THINKING", "false"))

# 重复惩罚，vLLM 扩展参数。实测对本问题效果有限，调太高会让 JSON 格式先崩，别超过 1.1。
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

# ---------------- logprobs（置信度参考） ----------------
ENABLE_LOGPROBS = _as_bool(os.getenv("ENABLE_LOGPROBS", "true"))
TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5"))

# ---------------- 坐标系 ----------------
# 实测 Qwen3.6 返回 0~1000 千分比坐标，即使 prompt 明确要求像素坐标也一样。
# 所以这里写死，不靠自动判断（图片尺寸接近 1000 时自动判断会误判）。
#   per_mille = 0~1000 千分比   pixel = 绝对像素   relative = 0~1 归一化   auto = 自动判断
COORD_MODE = os.getenv("COORD_MODE", "per_mille")

# ---------------- 类别表 ----------------
CLASSES_YAML_PATH = os.getenv("CLASSES_YAML_PATH", "./classes.yaml")

# ---------------- 结果过滤 ----------------
# 单个类别最多保留多少个框，超出的标记为无效。设 0 关闭。
MAX_BOXES_PER_CLASS = int(os.getenv("MAX_BOXES_PER_CLASS", "50"))

# 单个框占整图面积超过这个比例时标记为可疑（只标记，不丢弃）。设 1.0 关闭。
BIG_BOX_RATIO = float(os.getenv("BIG_BOX_RATIO", "0.5"))

# ---------------- 验证图 ----------------
# true  = 问题框也画出来（粗红框 + [!]），用于判断拦截是否正确
# false = 只画有效框，用于查看最终真正进训练集的结果
DRAW_INVALID = _as_bool(os.getenv("DRAW_INVALID", "false"))

# ---------------- 日志 ----------------
LOG_DIR = os.getenv("LOG_DIR", "./logs")
SAVE_RAW_LOG = _as_bool(os.getenv("SAVE_RAW_LOG", "true"))
