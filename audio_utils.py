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
