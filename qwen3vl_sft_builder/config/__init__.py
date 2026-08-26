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
            raise ValueError(
                f"配置项 {dotted} 未设置。请在 {LOCAL_PATH} 里填写"
                f"（可从 local.yaml.example 复制），或用对应的环境变量注入。"
            )
        return value


def load_config(extra_path: str | Path | None = None) -> Config:
    """加载配置。extra_path 若给出，优先级最高（低于环境变量）。"""
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8")) or {}
    if LOCAL_PATH.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(LOCAL_PATH.read_text(encoding="utf-8")) or {})
    if extra_path:
        p = Path(extra_path)
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件：{p}")
        cfg = _deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return Config(_apply_env(cfg))
