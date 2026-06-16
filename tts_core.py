"""ChatTTS 封装：懒加载模型、文本规范化、合成与保存。"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from config import ASSET_DIR, TTS_SAMPLE_RATE

_lock = threading.Lock()
_engine = None


def _ensure_transformers_compat() -> None:
    """ChatTTS 0.2.5 访问 past_key_values.layers；部分 transformers 只有 key_cache。"""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if getattr(DynamicCache, "_chattts_compat_patched", False):
        return

    if not hasattr(DynamicCache, "layers"):

        @property
        def _layers_compat(self):
            if hasattr(self, "key_cache"):
                return self.key_cache
            return []

        DynamicCache.layers = _layers_compat  # type: ignore[attr-defined]

    DynamicCache._chattts_compat_patched = True  # type: ignore[attr-defined]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_engine():
    _ensure_transformers_compat()
    import ChatTTS

    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        if not ASSET_DIR.is_dir():
            raise RuntimeError(f"模型目录不存在: {ASSET_DIR}")
        chat = ChatTTS.Chat()
        ok = chat.load(
            compile=False,
            source="custom",
            custom_path=str(ASSET_DIR.resolve()),
        )
        if not ok:
            raise RuntimeError("ChatTTS 模型加载失败，请检查 asset 目录是否完整")
        _engine = chat
        return _engine


def synthesize(text: str) -> tuple[np.ndarray, float]:
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("请输入要合成的文本")
    wav = get_engine().infer([cleaned])[0]
    duration = len(wav) / TTS_SAMPLE_RATE
    return wav, duration


def save_wav(wav: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, TTS_SAMPLE_RATE)
    return out
