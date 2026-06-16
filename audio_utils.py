"""参考音频导入：支持 WAV/MP3/M4A 等，统一转为 WAV 供 GPT-SoVITS 使用。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_IMPORT_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm"}
)

IMPORT_FILETYPES = [
    ("常见音频", "*.wav *.mp3 *.m4a *.flac *.ogg"),
    ("WAV", "*.wav"),
    ("M4A", "*.m4a"),
    ("MP3", "*.mp3"),
    ("全部", "*.*"),
]

_FFMPEG_FORMATS = frozenset({".mp3", ".m4a", ".aac", ".wma", ".webm", ".ogg"})


def _ffmpeg_exe() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    sf.write(str(path), audio.astype(np.float32), sample_rate)


INFER_REF_MIN_SECONDS = 3.0
INFER_REF_MAX_SECONDS = 10.0
INFER_REF_TARGET_SECONDS = 8.0


def align_prompt_text_for_infer(
    prompt_text: str,
    total_audio_seconds: float,
    infer_seconds: float,
) -> str:
    """参考音频被截短时，同步截短参考文本，避免音文不对齐。"""
    prompt_text = prompt_text.strip()
    if not prompt_text or infer_seconds >= total_audio_seconds - 0.1:
        return prompt_text

    ratio = infer_seconds / total_audio_seconds
    cut = max(1, int(len(prompt_text) * ratio))
    trimmed = prompt_text[:cut].strip()
    for sep in "。！？.!?，,":
        idx = trimmed.rfind(sep)
        if idx >= max(0, len(trimmed) - 24):
            return trimmed[: idx + 1].strip()
    return trimmed


def prepare_ref_audio_for_inference(
    src: str | Path,
    prompt_text: str = "",
    dst: str | Path | None = None,
) -> tuple[Path, str]:
    """GPT-SoVITS 推理要求参考音频 3~10 秒；过长则截取开头并同步截短参考文本。"""
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"参考音频不存在: {src_path}")

    prompt_text = prompt_text.strip()
    duration = get_duration(src_path)
    if duration < INFER_REF_MIN_SECONDS:
        raise ValueError(
            f"参考音频过短（{duration:.1f}s）。GPT-SoVITS 合成需要 "
            f"{INFER_REF_MIN_SECONDS:g}~{INFER_REF_MAX_SECONDS:g} 秒的参考音频，"
            "请重新录制或导入更长的样本。"
        )
    if duration <= INFER_REF_MAX_SECONDS:
        return src_path, prompt_text

    keep_seconds = min(INFER_REF_TARGET_SECONDS, INFER_REF_MAX_SECONDS)
    audio, sr = sf.read(str(src_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio[: int(keep_seconds * sr)]

    out_path = Path(dst) if dst else src_path.with_name(f"{src_path.stem}_infer.wav")
    _write_wav(out_path, audio, sr)
    aligned_prompt = align_prompt_text_for_infer(prompt_text, duration, keep_seconds)
    return out_path, aligned_prompt


def trim_to_max_duration(path: str | Path, max_seconds: float) -> float:
    """将 WAV 裁剪到指定时长（保留开头），返回最终秒数。"""
    wav_path = Path(path)
    audio, sr = sf.read(str(wav_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    max_samples = int(max_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        _write_wav(wav_path, audio, sr)
    return len(audio) / sr


def get_duration(path: str | Path) -> float:
    return float(sf.info(str(path)).duration)


def _convert_with_ffmpeg(src: Path, dst: Path, sample_rate: int | None) -> None:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError(
            "读取 M4A/MP3 等格式需要 FFmpeg。"
            "请运行 setup_sovits.ps1，或执行：pip install imageio-ffmpeg"
        )
    cmd = [ffmpeg, "-y", "-i", str(src), "-ac", "1"]
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or f"FFmpeg 转换失败: {src.name}")
    if not dst.is_file():
        raise RuntimeError(f"FFmpeg 未生成输出文件: {dst}")


def convert_to_wav(
    src: str | Path,
    dst: str | Path,
    *,
    sample_rate: int | None = None,
) -> Path:
    """将任意支持的音频文件转为 WAV。"""
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.is_file():
        raise FileNotFoundError(f"音频不存在: {src_path}")

    ext = src_path.suffix.lower()
    if ext not in SUPPORTED_IMPORT_EXTENSIONS:
        raise ValueError(f"不支持的音频格式: {ext or '(无扩展名)'}")

    if ext == ".wav" and dst_path.resolve() != src_path.resolve():
        try:
            audio, sr = sf.read(str(src_path), always_2d=False)
            _write_wav(dst_path, np.asarray(audio), sample_rate or sr)
            return dst_path
        except Exception:
            pass

    if ext in _FFMPEG_FORMATS or ext == ".wav":
        _convert_with_ffmpeg(src_path, dst_path, sample_rate)
        return dst_path

    try:
        import librosa

        audio, sr = librosa.load(str(src_path), sr=sample_rate, mono=True)
        _write_wav(dst_path, audio, sample_rate or sr)
        return dst_path
    except Exception as exc:
        raise RuntimeError(f"无法读取音频 {src_path.name}: {exc}") from exc
