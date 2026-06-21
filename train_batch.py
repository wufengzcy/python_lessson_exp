"""从文件夹批量加载 GPT-SoVITS 微调样本。"""

from __future__ import annotations

from pathlib import Path


def load_samples_from_folder(folder: str | Path) -> list[tuple[str, str]]:
    """
    读取训练样本目录。
    优先 manifest.tsv（每行：文件名<TAB或|>文本），否则匹配同名 .wav + .txt。
    返回 [(音频绝对路径, 文本), ...]
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    manifest = root / "manifest.tsv"
    if manifest.is_file():
        samples: list[tuple[str, str]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                fname, text = line.split("\t", 1)
            elif "|" in line:
                fname, text = line.split("|", 1)
            else:
                raise ValueError(f"manifest.tsv 格式错误（需 Tab 或 | 分隔）: {line}")
            audio = root / fname.strip()
            if not audio.is_file():
                raise FileNotFoundError(f"manifest 指向的音频不存在: {audio}")
            text = text.strip()
            if not text:
                raise ValueError(f"样本 {fname} 的文本为空")
            samples.append((str(audio.resolve()), text))
        if not samples:
            raise ValueError("manifest.tsv 中没有有效样本")
        return samples

    samples = []
    for wav in sorted(root.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{txt.name} 为空")
        samples.append((str(wav.resolve()), text))

    if not samples:
        raise ValueError(
            f"{root} 中未找到样本。请放置 manifest.tsv，或成对的 sample.wav + sample.txt"
        )
    return samples
