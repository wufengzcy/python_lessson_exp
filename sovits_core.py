"""GPT-SoVITS 本地推理封装（直接 import 引擎代码，非 HTTP API）。"""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

import audio_utils
from config import (
    GPT_SOVITS_DIR,
    SOVITS_PROMPT_LANG,
    SOVITS_TEXT_LANG,
    SOVITS_VERSION,
)

_lock = threading.Lock()
_pipeline = None
_weights_key: tuple[str | None, str | None] | None = None

_SENT_END = re.compile(r"(?<=[。！？!?])")


def is_engine_present() -> bool:
    return (GPT_SOVITS_DIR / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").is_file()


def is_pretrained_ready() -> bool:
    base = GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
    v2_gpt = base / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    v2_sovits = base / "gsv-v2final-pretrained" / "s2G2333k.pth"
    hubert = base / "chinese-hubert-base"
    bert = base / "chinese-roberta-wwm-ext-large"
    return v2_gpt.is_file() and v2_sovits.is_file() and hubert.is_dir() and bert.is_dir()


def _missing_inference_deps() -> list[str]:
    missing: list[str] = []
    for mod, pip_name in (
        ("fast_langdetect", "fast_langdetect>=0.3.1"),
        ("split_lang", "split-lang"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    return missing


def engine_status_message() -> str:
    if not is_engine_present():
        return "未找到 GPT-SoVITS 源码，请运行 setup_sovits.ps1"
    if not is_pretrained_ready():
        return "GPT-SoVITS 预训练模型未就绪，请在 engines/GPT-SoVITS 下执行 install.ps1"
    missing = _missing_inference_deps()
    if missing:
        return "缺少推理依赖: " + ", ".join(missing) + "。请执行: pip install " + " ".join(missing)
    return "就绪"


_torchaudio_patched = False


def _patch_torchaudio_load() -> None:
    """新版 torchaudio 默认走 torchcodec，Windows 上常不可用；改用 soundfile。"""
    global _torchaudio_patched
    if _torchaudio_patched:
        return
    import torch
    import torchaudio

    def _load_with_soundfile(uri, *args, **kwargs):
        data, sr = sf.read(str(uri), always_2d=True)
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            tensor = torch.from_numpy(audio.reshape(1, -1).copy())
        else:
            tensor = torch.from_numpy(audio.T.copy())
        return tensor, sr

    torchaudio.load = _load_with_soundfile
    _torchaudio_patched = True


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
    _patch_torchaudio_load()
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


def _normalize_text_for_infer(text: str) -> str:
    """顿号在 cut5 会被切成极短片段易漏读，统一为逗号停顿。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", "", text)
    text = text.replace("、", "，")
    return text.strip()


def _split_by_sentence(text: str) -> list[str]:
    """仅按句号/问号/叹号切分，逗号与顿号留在句内。"""
    text = _normalize_text_for_infer(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    if not parts:
        return [text]
    kept: list[str] = []
    for part in parts:
        if re.sub(r"[\W_]+", "", part, flags=re.UNICODE):
            kept.append(part)
    return kept or [text]


def _estimate_max_sec(text: str) -> float:
    """单段 max_sec：按段内字数估算，避免长文全局上限导致后半段被截断。"""
    chars = len(text.replace("\n", "").replace(" ", ""))
    est = chars / 3.0 + 2.0
    return max(5.0, min(est, 50.0))


def _trim_segment_tail(wav: np.ndarray, sr: int, text: str) -> np.ndarray:
    """仅裁切明显拖尾的过长音频，不用全文上限截断长合成结果。"""
    chars = len(text.replace("\n", "").replace(" ", ""))
    expected = chars / 3.0 + 1.0
    cap = max(expected * 1.6 + 1.5, 6.0)
    max_samples = int(sr * cap)
    if len(wav) > max_samples:
        return wav[:max_samples]
    return wav


def _pick_intra_pause(gpt_weights: str | None) -> float:
    """句内逗号处由引擎 cut0 + 整段合成；多句之间额外静音。"""
    return 0.22 if gpt_weights else 0.28


def _run_pipe_chunk(
    pipe,
    *,
    chunk: str,
    ref_for_infer: Path,
    prompt_for_infer: str,
    gpt_weights: str | None,
) -> tuple[np.ndarray, int]:
    pipe.configs.max_sec = _estimate_max_sec(chunk)
    rep_penalty = 1.62 if gpt_weights else 1.38
    inputs = {
        "text": chunk,
        "text_lang": SOVITS_TEXT_LANG,
        "ref_audio_path": str(ref_for_infer.resolve()),
        "prompt_text": prompt_for_infer,
        "prompt_lang": SOVITS_PROMPT_LANG,
        "text_split_method": "cut0",
        "batch_size": 1,
        "speed_factor": 1.0,
        "parallel_infer": False,
        "split_bucket": False,
        "fragment_interval": 0.0,
        "top_k": 8,
        "repetition_penalty": rep_penalty,
        "streaming_mode": False,
    }
    result = pipe.run(inputs)
    sr, audio = None, None
    for sr, audio in result:
        pass
    if audio is None or len(audio) == 0:
        raise RuntimeError(f"GPT-SoVITS 合成失败，片段未返回音频: {chunk[:40]}…")
    audio_arr = np.asarray(audio)
    if audio_arr.dtype == np.int16 or np.max(np.abs(audio_arr)) > 2:
        wav = audio_arr.astype(np.float32) / 32768.0
    else:
        wav = audio_arr.astype(np.float32)
    wav = _trim_segment_tail(wav, int(sr), chunk)
    return wav, int(sr)


def synthesize(
    text: str,
    ref_audio_path: str,
    prompt_text: str,
    gpt_weights: str | None = None,
    sovits_weights: str | None = None,
) -> tuple[np.ndarray, int]:
    text = _normalize_text_for_infer(text)
    prompt_text = prompt_text.strip()
    ref = Path(ref_audio_path)
    if not text:
        raise ValueError("合成文本不能为空")
    if not prompt_text:
        raise ValueError("参考音频对应的文本不能为空")
    if not ref.is_file():
        raise FileNotFoundError(f"参考音频不存在: {ref}")

    ref_for_infer, prompt_for_infer = audio_utils.prepare_ref_audio_for_inference(
        ref, prompt_text
    )

    chunks = _split_by_sentence(text)
    if len(chunks) == 1 and len(chunks[0]) > 120:
        chunks = _split_long_clause(chunks[0])

    with _gsovits_runtime():
        pipe = _get_pipeline(gpt_weights, sovits_weights)
        pause = _pick_intra_pause(gpt_weights)
        segments: list[np.ndarray] = []
        sr_out = 32000

        for i, chunk in enumerate(chunks):
            wav, sr_out = _run_pipe_chunk(
                pipe,
                chunk=chunk,
                ref_for_infer=ref_for_infer,
                prompt_for_infer=prompt_for_infer,
                gpt_weights=gpt_weights,
            )
            segments.append(wav)
            if i < len(chunks) - 1:
                segments.append(np.zeros(int(sr_out * pause), dtype=np.float32))

        if not segments:
            raise RuntimeError("GPT-SoVITS 合成失败，未生成任何音频片段")
        return np.concatenate(segments), sr_out


def _split_long_clause(text: str, max_chars: int = 80) -> list[str]:
    """超长单句按逗号再切，仍不把顿号级碎片交给引擎。"""
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in re.split(r"(?<=[，；;])", text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    merged: list[str] = []
    buf = ""
    for part in parts:
        if not buf:
            buf = part
        elif len(buf) + len(part) <= max_chars:
            buf += part
        else:
            merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)
    return merged or [text]


def save_wav(wav: np.ndarray, path: str | Path, sample_rate: int) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, sample_rate)
    return out
