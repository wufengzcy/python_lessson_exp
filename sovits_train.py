"""GPT-SoVITS 微调训练：调用引擎自带脚本（subprocess，非 HTTP）。"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import audio_utils
from config import GPT_SOVITS_DIR, GPT_SOVITS_EXP_ROOT, SOVITS_VERSION, VOICES_DIR
from train_watchdog import (
    STALL_SECONDS_PREP,
    STALL_SECONDS_TRAIN,
    run_with_watchdog,
    watch_paths_for_step,
)

ProgressCallback = Callable[[str], None]

# GPT-SoVITS s2_train 只接受 0.6～54 秒样本，这里略留余量。
FINETUNE_MIN_SECONDS = 0.6
FINETUNE_MAX_SECONDS = 50.0


def _log(cb: ProgressCallback | None, msg: str) -> None:
    if cb:
        cb(msg)


def _build_sovits_env(root: Path, extra: dict | None = None) -> dict:
    """GPT-SoVITS 子进程需要与 webui 一致的 PYTHONPATH 与环境变量。"""
    env = os.environ.copy()
    path_entries = [
        str(root),
        str(root / "GPT_SoVITS"),
        str(root / "tools"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        path_entries + ([existing] if existing else [])
    )
    if not shutil.which("ffmpeg", path=env.get("PATH", "")):
        try:
            import imageio_ffmpeg

            ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
            env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
        except ImportError:
            pass
    if extra:
        env.update(extra)
    if os.name == "nt":
        env["USE_LIBUV"] = "0"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> str:
    """短任务同步执行（无看门狗）。"""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or f"命令失败: {' '.join(cmd)}")
    return proc.stdout or ""


def _run_step(
    step: int,
    step_label: str,
    cmd: list[str],
    cwd: Path,
    env: dict,
    opt_dir: Path,
    root: Path,
    exp_name: str,
    progress: ProgressCallback | None,
) -> str:
    stall = STALL_SECONDS_TRAIN if step >= 4 else STALL_SECONDS_PREP
    paths = watch_paths_for_step(step, opt_dir, root, exp_name, SOVITS_VERSION)
    log_path = opt_dir / f"step{step}.log"
    return run_with_watchdog(
        cmd,
        cwd,
        env,
        step_label=step_label,
        watch_paths=paths,
        progress=progress,
        stall_seconds=stall,
        log_path=log_path,
    )


def _clean_preprocess_artifacts(opt_dir: Path) -> None:
    """清理上次失败留下的预处理缓存，避免官方脚本因文件已存在而跳过。"""
    for pattern in ("2-name2text*.txt", "6-name2semantic*.tsv", "events.out.tfevents.*"):
        for path in opt_dir.glob(pattern):
            path.unlink(missing_ok=True)
    for name in ("train.log", "config.json"):
        (opt_dir / name).unlink(missing_ok=True)
    for name in ("3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn", "eval"):
        path = opt_dir / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _merge_text_parts(opt_dir: Path, all_parts: int = 1) -> None:
    lines: list[str] = []
    for i in range(all_parts):
        part = opt_dir / f"2-name2text-{i}.txt"
        if not part.is_file():
            raise RuntimeError(f"步骤 1 未生成 {part.name}")
        content = part.read_text(encoding="utf-8").strip()
        if content:
            lines.extend(content.split("\n"))
        part.unlink(missing_ok=True)
    if not lines:
        raise RuntimeError(
            "步骤 1 文本特征为空。常见原因：缺少 jieba_fast / G2PWModel，"
            "或参考文本与音频不匹配。请重新运行 setup_sovits.ps1 后重试。"
        )
    (opt_dir / "2-name2text.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_semantic_parts(opt_dir: Path, all_parts: int = 1) -> None:
    lines = ["item_name\tsemantic_audio"]
    for i in range(all_parts):
        part = opt_dir / f"6-name2semantic-{i}.tsv"
        if not part.is_file():
            raise RuntimeError(f"步骤 3 未生成 {part.name}")
        content = part.read_text(encoding="utf-8").strip()
        if content:
            lines.extend(content.split("\n"))
        part.unlink(missing_ok=True)
    if len(lines) <= 1:
        raise RuntimeError("步骤 3 语义 token 为空")
    (opt_dir / "6-name2semantic.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_preprocess_ready(opt_dir: Path) -> None:
    missing: list[str] = []
    for rel in ("2-name2text.txt", "6-name2semantic.tsv"):
        if not (opt_dir / rel).is_file():
            missing.append(rel)
    for rel in ("4-cnhubert", "5-wav32k"):
        path = opt_dir / rel
        if not path.is_dir() or not any(path.iterdir()):
            missing.append(rel)
    if missing:
        raise RuntimeError("预处理产物不完整: " + ", ".join(missing))


def prepare_dataset(
    user_id: int,
    profile_name: str,
    samples: list[tuple[str, str]],
) -> tuple[str, str, str, str]:
    """
    复制训练样本并生成 GPT-SoVITS 训练用 meta.list。
    samples: [(音频路径, 对应文本), ...]，建议 3～10 条、每条 3～30 秒。
    返回 (wav_dir, list_path, exp_name, ref_prompt_text)。
    """
    if not samples:
        raise ValueError("至少需要一个训练样本")

    safe_name = "".join(
        c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in profile_name
    ).strip("_") or "voice"
    exp_name = f"user{user_id}_{safe_name}"
    root = Path(VOICES_DIR) / str(user_id) / safe_name
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, (wav_src, transcript) in enumerate(samples, start=1):
        transcript = transcript.strip()
        if not transcript:
            raise ValueError(f"第 {i} 条样本的参考文本不能为空")
        dst_wav = wav_dir / f"sample_{i:03d}.wav"
        audio_utils.convert_to_wav(wav_src, dst_wav)
        duration = audio_utils.trim_to_max_duration(dst_wav, FINETUNE_MAX_SECONDS)
        if duration < FINETUNE_MIN_SECONDS:
            raise ValueError(
                f"第 {i} 条样本过短（{duration:.1f}s），每条至少需要 {FINETUNE_MIN_SECONDS}s"
            )
        lines.append(f"{dst_wav.resolve()}|{safe_name}|zh|{transcript}")

    list_path = root / "meta.list"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ref_wav = root / "reference.wav"
    shutil.copy2(wav_dir / "sample_001.wav", ref_wav)
    ref_prompt = samples[0][1].strip()
    return str(wav_dir), str(list_path), exp_name, ref_prompt


def _ensure_train_dirs(opt_dir: Path, engine_root: Path) -> None:
    """GPT-SoVITS 保存 checkpoint 前不会自动建 logs_s2_* 目录。"""
    for name in (
        f"logs_s2_{SOVITS_VERSION}",
        f"logs_s1_{SOVITS_VERSION}",
        "eval",
    ):
        (opt_dir / name).mkdir(parents=True, exist_ok=True)
    for name in ("SoVITS_weights_v2", "GPT_weights_v2", "TEMP"):
        (engine_root / name).mkdir(parents=True, exist_ok=True)


def _latest_weight(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def run_finetune(
    list_path: str,
    wav_dir: str,
    exp_name: str,
    progress: ProgressCallback | None = None,
    sovits_epochs: int = 3,
    gpt_epochs: int = 5,
) -> tuple[str | None, str | None]:
    """
    执行 GPT-SoVITS 数据预处理 + SoVITS/GPT 微调。
    返回 (sovits_weights_path, gpt_weights_path)。
    """
    if not GPT_SOVITS_DIR.is_dir():
        raise RuntimeError("未找到 engines/GPT-SoVITS，请先运行 setup_sovits.ps1")

    python = sys.executable
    root = GPT_SOVITS_DIR.resolve()
    GPT_SOVITS_EXP_ROOT.mkdir(parents=True, exist_ok=True)
    opt_dir = root / "logs" / exp_name
    opt_dir.mkdir(parents=True, exist_ok=True)
    _clean_preprocess_artifacts(opt_dir)

    bert_dir = root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    ssl_dir = root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
    pretrained_s2g = root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"
    pretrained_s2d = Path(str(pretrained_s2g).replace("s2G", "s2D"))
    pretrained_s1 = root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

    env = _build_sovits_env(
        root,
        {
            "inp_text": str(Path(list_path).resolve()),
            "inp_wav_dir": str(Path(wav_dir).resolve()),
            "exp_name": exp_name,
            "opt_dir": str(opt_dir.resolve()),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "True",
            "version": SOVITS_VERSION,
            "bert_pretrained_dir": str(bert_dir.resolve()),
        },
    )

    _log(progress, "步骤 1/5：提取文本特征…")
    step1_out = _run_step(
        1,
        "步骤 1/5：提取文本特征",
        [python, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        root,
        env,
        opt_dir,
        root,
        exp_name,
        progress,
    )
    try:
        _merge_text_parts(opt_dir)
    except RuntimeError as exc:
        if "Traceback" in step1_out:
            tail = step1_out[step1_out.rfind("Traceback") :]
            raise RuntimeError(f"{exc}\n\n{tail}") from exc
        raise

    env["cnhubert_base_dir"] = str(ssl_dir)
    _log(progress, "步骤 2/5：提取 Hubert 特征…")
    _run_step(
        2,
        "步骤 2/5：提取 Hubert 特征",
        [python, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        root,
        env,
        opt_dir,
        root,
        exp_name,
        progress,
    )

    env["pretrained_s2G"] = str(pretrained_s2g.resolve())
    env["s2config_path"] = "GPT_SoVITS/configs/s2.json"
    _log(progress, "步骤 3/5：提取语义 token…")
    _run_step(
        3,
        "步骤 3/5：提取语义 token",
        [python, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
        root,
        env,
        opt_dir,
        root,
        exp_name,
        progress,
    )
    _merge_semantic_parts(opt_dir)
    _verify_preprocess_ready(opt_dir)

    import json

    s2_cfg_path = root / "GPT_SoVITS" / "configs" / "s2.json"
    with open(s2_cfg_path, encoding="utf-8") as f:
        s2_data = json.load(f)
    s2_data["train"]["batch_size"] = 1
    s2_data["train"]["epochs"] = sovits_epochs
    s2_data["train"]["pretrained_s2G"] = str(pretrained_s2g)
    s2_data["train"]["pretrained_s2D"] = str(pretrained_s2d)
    s2_data["train"]["gpu_numbers"] = "0"
    s2_data["train"]["if_save_latest"] = True
    s2_data["train"]["if_save_every_weights"] = True
    s2_data["train"]["save_every_epoch"] = max(1, sovits_epochs // 2)
    s2_data["model"]["version"] = SOVITS_VERSION
    s2_data["data"]["exp_dir"] = str(opt_dir)
    s2_data["s2_ckpt_dir"] = str(opt_dir)
    s2_data["save_weight_dir"] = "SoVITS_weights_v2"
    s2_data["name"] = exp_name
    s2_data["version"] = SOVITS_VERSION
    tmp_s2 = root / "TEMP" / f"tmp_s2_{exp_name}.json"
    tmp_s2.parent.mkdir(parents=True, exist_ok=True)
    tmp_s2.write_text(json.dumps(s2_data), encoding="utf-8")

    _ensure_train_dirs(opt_dir, root)
    _log(progress, "步骤 4/5：微调 SoVITS…")
    _run_step(
        4,
        "步骤 4/5：微调 SoVITS",
        [python, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_s2)],
        root,
        env,
        opt_dir,
        root,
        exp_name,
        progress,
    )

    import yaml

    s1_cfg_path = root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
    with open(s1_cfg_path, encoding="utf-8") as f:
        s1_data = yaml.safe_load(f)
    s1_data["train"]["batch_size"] = 1
    s1_data["train"]["epochs"] = gpt_epochs
    s1_data["pretrained_s1"] = str(pretrained_s1)
    s1_data["train"]["save_every_n_epoch"] = max(1, gpt_epochs // 2)
    s1_data["train"]["if_save_every_weights"] = True
    s1_data["train"]["if_save_latest"] = True
    s1_data["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    s1_data["train"]["exp_name"] = exp_name
    if os.name == "nt":
        s1_data["data"]["num_workers"] = 0
    s1_data["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
    s1_data["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
    s1_data["output_dir"] = str(opt_dir / f"logs_s1_{SOVITS_VERSION}")
    tmp_s1 = root / "TEMP" / f"tmp_s1_{exp_name}.yaml"
    with open(tmp_s1, "w", encoding="utf-8") as f:
        yaml.dump(s1_data, f, allow_unicode=True)

    _ensure_train_dirs(opt_dir, root)
    _log(progress, "步骤 5/5：微调 GPT…")
    _run_step(
        5,
        "步骤 5/5：微调 GPT",
        [python, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_s1)],
        root,
        env,
        opt_dir,
        root,
        exp_name,
        progress,
    )

    sovits_w = _latest_weight(str(root / "SoVITS_weights_v2" / f"{exp_name}_*.pth"))
    gpt_w = _latest_weight(str(root / "GPT_weights_v2" / f"{exp_name}-e*.ckpt"))
    if not sovits_w or not gpt_w:
        raise RuntimeError("训练完成但未找到输出权重，请检查 engines/GPT-SoVITS/SoVITS_weights_v2")
    _log(progress, "训练完成")
    return sovits_w, gpt_w
