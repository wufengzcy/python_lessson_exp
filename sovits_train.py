"""GPT-SoVITS 微调训练：调用引擎自带脚本（subprocess，非 HTTP）。"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from config import GPT_SOVITS_DIR, GPT_SOVITS_EXP_ROOT, SOVITS_VERSION, VOICES_DIR

ProgressCallback = Callable[[str], None]


def _log(cb: ProgressCallback | None, msg: str) -> None:
    if cb:
        cb(msg)


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
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


def prepare_dataset(
    user_id: int,
    profile_name: str,
    wav_src: str,
    transcript: str,
) -> tuple[str, str, str]:
    """复制样本并生成 GPT-SoVITS 训练用 list 文件。返回 (wav_dir, list_path, exp_name)。"""
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("参考文本不能为空")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in profile_name)
    exp_name = f"user{user_id}_{safe_name}"
    root = Path(VOICES_DIR) / str(user_id) / safe_name
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dst_wav = wav_dir / "sample.wav"
    shutil.copy2(wav_src, dst_wav)

    list_path = root / "meta.list"
    line = f"{dst_wav.resolve()}|{safe_name}|zh|{transcript}"
    list_path.write_text(line + "\n", encoding="utf-8")
    return str(wav_dir), str(list_path), exp_name


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
    sovits_epochs: int = 6,
    gpt_epochs: int = 10,
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

    bert_dir = root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    ssl_dir = root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
    pretrained_s2g = root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"
    pretrained_s2d = Path(str(pretrained_s2g).replace("s2G", "s2D"))
    pretrained_s1 = root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

    env = os.environ.copy()
    env["inp_text"] = str(Path(list_path).resolve())
    env["inp_wav_dir"] = str(Path(wav_dir).resolve())
    env["exp_name"] = exp_name
    env["opt_dir"] = str(opt_dir.resolve())
    env["i_part"] = "0"
    env["all_parts"] = "1"
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["is_half"] = "True"

    _log(progress, "步骤 1/5：提取文本特征…")
    _run([python, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], root, env)

    env["cnhubert_base_dir"] = str(ssl_dir)
    _log(progress, "步骤 2/5：提取 Hubert 特征…")
    _run([python, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], root, env)

    _log(progress, "步骤 3/5：提取语义 token…")
    _run([python, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], root, env)

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

    _log(progress, "步骤 4/5：微调 SoVITS…")
    _run([python, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_s2)], root, env)

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
    s1_data["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
    s1_data["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
    s1_data["output_dir"] = str(opt_dir / f"logs_s1_{SOVITS_VERSION}")
    tmp_s1 = root / "TEMP" / f"tmp_s1_{exp_name}.yaml"
    with open(tmp_s1, "w", encoding="utf-8") as f:
        yaml.dump(s1_data, f, allow_unicode=True)

    _log(progress, "步骤 5/5：微调 GPT…")
    _run([python, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_s1)], root, env)

    sovits_w = _latest_weight(str(root / "SoVITS_weights_v2" / f"{exp_name}_*.pth"))
    gpt_w = _latest_weight(str(root / "GPT_weights_v2" / f"{exp_name}-e*.ckpt"))
    if not sovits_w or not gpt_w:
        raise RuntimeError("训练完成但未找到输出权重，请检查 engines/GPT-SoVITS/SoVITS_weights_v2")
    _log(progress, "训练完成")
    return sovits_w, gpt_w
