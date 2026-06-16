"""声线包：打包微调权重 + 参考音，换机器后一键导入。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import db
from config import BASE_DIR, GPT_SOVITS_DIR


def _latest(glob_pattern: str, directory: Path) -> Path | None:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def pack_finetune_15(
    *,
    pack_id: str = "finetune_15",
    voice_name: str = "15样本训练",
    gpt_glob: str = "user3_15-e*.ckpt",
    sovits_glob: str = "user3_15_e*.pth",
    ref_audio: Path | None = None,
    prompt_text: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    gpt_dir = GPT_SOVITS_DIR / "GPT_weights_v2"
    sovits_dir = GPT_SOVITS_DIR / "SoVITS_weights_v2"
    gpt_src = _latest(gpt_glob, gpt_dir)
    sovits_src = _latest(sovits_glob, sovits_dir)
    if not gpt_src or not sovits_src:
        raise FileNotFoundError("未找到 user3_15 权重，请先完成 15 样本微调训练。")
    ref_src = ref_audio or Path(BASE_DIR) / "data" / "voices" / "3" / "15" / "reference.wav"
    if not ref_src.is_file():
        raise FileNotFoundError(f"参考音频不存在: {ref_src}")
    if not prompt_text:
        prompt_text = (
            "今天天气真的不错，阳光洒在脸上暖洋洋的，我们下午一起去公园走走，"
            "顺便呼吸一下新鲜空气吧。"
        )

    install_gpt_rel = f"engines/GPT-SoVITS/GPT_weights_v2/{gpt_src.name}"
    install_sovits_rel = f"engines/GPT-SoVITS/SoVITS_weights_v2/{sovits_src.name}"
    install_ref_rel = "data/voice_packs/finetune_15/reference.wav"

    target = out_dir or Path(BASE_DIR) / "deploy" / "voice_packs" / pack_id
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(gpt_src, target / "gpt.ckpt")
    shutil.copy2(sovits_src, target / "sovits.pth")
    shutil.copy2(ref_src, target / "reference.wav")

    meta = {
        "pack_id": pack_id,
        "voice_name": voice_name,
        "mode": "finetuned",
        "prompt_text": prompt_text,
        "files": {
            "gpt": "gpt.ckpt",
            "sovits": "sovits.pth",
            "reference": "reference.wav",
        },
        "install": {
            "gpt": install_gpt_rel,
            "sovits": install_sovits_rel,
            "reference": install_ref_rel,
        },
    }
    (target / "pack.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def import_voice_pack(pack_dir: str | Path, *, username: str = "admin") -> int:
    """把声线包安装到本项目，并在数据库注册（使用相对路径）。"""
    root = Path(pack_dir).resolve()
    meta_path = root / "pack.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 pack.json: {root}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta["files"]
    install = meta["install"]

    for key in ("gpt", "sovits", "reference"):
        src = root / files[key]
        dst = Path(BASE_DIR) / install[key].replace("/", os.sep)
        if not src.is_file():
            raise FileNotFoundError(f"声线包文件缺失: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    user = db.get_user_by_username(username)
    if not user:
        raise ValueError(f"用户不存在: {username}，请先运行 main.py 初始化数据库并登录注册。")

    rel_ref = install["reference"]
    rel_gpt = install["gpt"]
    rel_sovits = install["sovits"]

    existing = db.list_voice_profiles_by_user(user["id"])
    for p in existing:
        if p["name"] == meta["voice_name"] and p["status"] == "ready":
            db.delete_voice_profile(p["id"], user["id"])

    profile_id = db.create_voice_profile(
        user["id"],
        meta["voice_name"],
        rel_ref,
        meta["prompt_text"],
        mode=meta.get("mode", "finetuned"),
        gpt_weights_path=rel_gpt,
        sovits_weights_path=rel_sovits,
        status="ready",
    )
    return profile_id
