"""GPT-SoVITS 本地推理封装（直接 import 引擎代码，非 HTTP API）。"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from config import (
    GPT_SOVITS_DIR,
    SOVITS_PROMPT_LANG,
    SOVITS_TEXT_LANG,
    SOVITS_VERSION,
)

_lock = threading.Lock()
_pipeline = None
_weights_key: tuple[str | None, str | None] | None = None


def is_engine_present() -> bool:
    return (GPT_SOVITS_DIR / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").is_file()


def is_pretrained_ready() -> bool:
    base = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
    v2_gpt = base / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    v2_sovits = base / "gsv-v2final-pretrained" / "s2G2333k.pth"
    hubert = base / "chinese-hubert-base"
    bert = base / "chinese-roberta-wwm-ext-large"
    return v2_gpt.is_file() and v2_sovits.is_file() and hubert.is_dir() and bert.is_dir()


def engine_status_message() -> str:
    if not is_engine_present():
        return "未找到 GPT-SoVITS 源码，请运行 setup_sovits.ps1"
    if not is_pretrained_ready():
        return "GPT-SoVITS 预训练模型未就绪，请在 engines/GPT-SoVITS 下执行 install.ps1"
    return "就绪"


@contextmanager
def _gsovits_runtime():
    """切换工作目录并注入 sys.path，满足 GPT-SoVITS 内部相对路径。"""
    if not is_engine_present():
        raise RuntimeError(engine_status_message())
    root = str(GPT_SOVITS_DIR.resolve())
    old_cwd = os.getcwd()
    added: list[str] = []
    for p in (root, os.path.join(root, "GPT_SoVITS"), os.path.join(root, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(old_cwd)
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _get_pipeline(gpt_weights: str | None = None, sovits_weights: str | None = None):
    global _pipeline, _weights_key
    key = (gpt_weights, sovits_weights)
    with _lock:
        if _pipeline is not None and _weights_key == key:
            return _pipeline
        if not is_pretrained_ready():
            raise RuntimeError(engine_status_message())

        with _gsovits_runtime():
            import torch
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

            device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = TTS_Config(
                {
                    "custom": {
                        "version": SOVITS_VERSION,
                        "device": device,
                        "is_half": device == "cuda",
                    }
                }
            )
            pipe = TTS(cfg)
            if sovits_weights and Path(sovits_weights).is_file():
                pipe.init_vits_weights(str(Path(sovits_weights).resolve()))
            if gpt_weights and Path(gpt_weights).is_file():
                pipe.init_t2s_weights(str(Path(gpt_weights).resolve()))

            _pipeline = pipe
            _weights_key = key
            return _pipeline


def synthesize(
    text: str,
    ref_audio_path: str,
    prompt_text: str,
    gpt_weights: str | None = None,
    sovits_weights: str | None = None,
) -> tuple[np.ndarray, int]:
    text = text.strip()
    prompt_text = prompt_text.strip()
    ref = Path(ref_audio_path)
    if not text:
        raise ValueError("合成文本不能为空")
    if not prompt_text:
        raise ValueError("参考音频对应的文本不能为空")
    if not ref.is_file():
        raise FileNotFoundError(f"参考音频不存在: {ref}")

    with _gsovits_runtime():
        pipe = _get_pipeline(gpt_weights, sovits_weights)
        inputs = {
            "text": text,
            "text_lang": SOVITS_TEXT_LANG,
            "ref_audio_path": str(ref.resolve()),
            "prompt_text": prompt_text,
            "prompt_lang": SOVITS_PROMPT_LANG,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": 1.0,
            "parallel_infer": True,
            "streaming_mode": False,
        }
        sr, audio = pipe.run(inputs)
        if audio is None or len(audio) == 0:
            raise RuntimeError("GPT-SoVITS 合成失败，未返回音频")
        return np.asarray(audio, dtype=np.float32), int(sr)


def save_wav(wav: np.ndarray, path: str | Path, sample_rate: int) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, sample_rate)
    return out
