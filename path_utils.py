"""项目内路径：数据库存相对路径，运行时按项目根目录解析。"""

from __future__ import annotations

import os
from pathlib import Path

from config import BASE_DIR

_MARKERS = (
    "engines/GPT-SoVITS/",
    "engines\\GPT-SoVITS\\",
    "data/voice_packs/",
    "data\\voice_packs\\",
    "data/voices/",
    "data\\voices\\",
    "data/",
    "data\\",
)


def to_project_relative(path: str | Path) -> str:
    """尽量转为相对项目根的路径，便于换机器部署。"""
    p = Path(path).resolve()
    base = Path(BASE_DIR).resolve()
    try:
        return str(p.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(p)


def resolve_project_path(path: str | None) -> str | None:
    """读取 DB/配置中的路径，自动适配本机项目目录。"""
    if not path or not str(path).strip():
        return None
    raw = str(path).strip()
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    if not p.is_absolute():
        candidate = Path(BASE_DIR) / raw
        if candidate.is_file():
            return str(candidate.resolve())
    normalized = raw.replace("\\", "/")
    for marker in _MARKERS:
        idx = normalized.find(marker.replace("\\", "/"))
        if idx >= 0:
            suffix = normalized[idx:]
            candidate = Path(BASE_DIR) / suffix.replace("/", os.sep)
            if candidate.is_file():
                return str(candidate.resolve())
    return raw


def resolve_voice_profile(profile: dict) -> dict:
    out = dict(profile)
    for key in ("ref_audio_path", "gpt_weights_path", "sovits_weights_path"):
        if out.get(key):
            out[key] = resolve_project_path(out[key])
    return out
