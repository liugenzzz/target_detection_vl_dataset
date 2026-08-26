"""配置加载。三层覆盖：default.yaml < local.yaml < 环境变量。

服务器上只改 local.yaml，不动代码，也不动 default.yaml。
密钥类走环境变量，不落盘：VLM_API_KEY、VLM_API_URL。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = CONFIG_DIR / "default.yaml"
LOCAL_PATH = CONFIG_DIR / "local.yaml"

# 环境变量 -> 配置路径。用于注入密钥和部署相关的地址。
ENV_OVERRIDES = {
    "VLM_API_KEY": ("vlm", "api_key"),
    "VLM_API_URL": ("vlm", "api_url"),
    "VLM_MODEL": ("vlm", "model"),
    "LABELS_DIR": ("paths", "labels_dir"),
    "IMAGES_DIR": ("paths", "images_dir"),
    "CLASSES_YAML": ("paths", "classes_yaml"),
    "OUTPUT_DIR": ("paths", "output_dir"),
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """读一个 yaml。解析失败时给出能直接照做的提示，而不是甩一段 yaml 内部堆栈。

    最常见的失败是 Windows 路径：YAML 双引号字符串把反斜杠当转义符，
    "F:\AI-Haishi\..." 里的 \A 是非法转义。
    """
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        hint = ""
        if "unknown escape character" in str(exc):
            hint = (
                "\n\n这是 Windows 路径的经典问题：YAML 的【双引号】字符串会把反斜杠"
                "当成转义符。\n把路径改成下面任意一种写法（推荐第一种）：\n"
                "    labels_dir: 'F:\\AI-Haishi\\project\\labels'      单引号，不处理转义\n"
                "    labels_dir: \"F:/AI-Haishi/project/labels\"        正斜杠\n"
                "    labels_dir: \"F:\\\\AI-Haishi\\\\project\\\\labels\"  双反斜杠"
            )
        raise ValueError(f"{path} 不是合法的 YAML：\n{exc}{hint}") from None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            cfg.setdefault(section, {})[key] = value
    return cfg


class Config(dict):
    """点号取值的配置对象：cfg.get_path("vlm.api_url")。"""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get_path(dotted)
        if value in (None, ""):
            env = next((e for e, (sec, k) in ENV_OVERRIDES.items()
                        if f"{sec}.{k}" == dotted), None)
            msg = [f"配置项 {dotted} 未设置。"]
            if not LOCAL_PATH.exists():
                msg.append(f"先复制一份配置：\n"
                           f"    cp {LOCAL_PATH.with_name('local.yaml.example')} {LOCAL_PATH}")
            else:
                msg.append(f"请在 {LOCAL_PATH} 里填写它。")
                msg.append("Windows 路径记得用单引号：labels_dir: 'F:\\xxx\\labels'")
            if env:
                msg.append(f"也可以用环境变量注入：{env}=...")
            raise ValueError("\n".join(msg))
        return value


def load_config(extra_path: str | Path | None = None) -> Config:
    """加载配置。extra_path 若给出，优先级最高（低于环境变量）。"""
    cfg = _load_yaml(DEFAULT_PATH)
    if LOCAL_PATH.exists():
        cfg = _deep_merge(cfg, _load_yaml(LOCAL_PATH))
    if extra_path:
        p = Path(extra_path)
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件：{p}")
        cfg = _deep_merge(cfg, _load_yaml(p))
    return Config(_apply_env(cfg))
