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


def _install_pack_files(root: Path, meta: dict) -> dict[str, str]:
    """复制声线包文件到项目目录，返回安装后的相对路径。"""
    files = meta["files"]
    install = meta["install"]
    installed: dict[str, str] = {}

    for key in ("gpt", "sovits", "reference"):
        src = root / files[key]
        rel_dst = install[key]
        dst = Path(BASE_DIR) / rel_dst.replace("/", os.sep)
        if not src.is_file():
            raise FileNotFoundError(f"声线包文件缺失: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        installed[key] = rel_dst
    return installed


def _register_voice_for_user(user: dict, meta: dict, install: dict[str, str]) -> int:
    voice_name = meta["voice_name"]
    for p in db.list_voice_profiles_by_user(user["id"]):
        if p["name"] == voice_name and p["status"] == "ready":
            db.delete_voice_profile(p["id"], user["id"])

    return db.create_voice_profile(
        user["id"],
        voice_name,
        install["reference"],
        meta["prompt_text"],
        mode=meta.get("mode", "finetuned"),
        gpt_weights_path=install["gpt"],
        sovits_weights_path=install["sovits"],
        status="ready",
    )


def import_voice_pack(
    pack_dir: str | Path,
    *,
    username: str | None = None,
    all_users: bool = True,
) -> list[dict]:
    """把声线包安装到本项目，并在数据库注册（使用相对路径）。

    默认给数据库中所有用户各注册一条声线，避免同学用自己注册的账号登录后看不到声线。
    若只想导入到指定账号，传 username= 且 all_users=False。
    """
    root = Path(pack_dir).resolve()
    meta_path = root / "pack.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 pack.json: {root}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    install = _install_pack_files(root, meta)

    if all_users:
        users = db.list_users()
        if not users:
            raise ValueError("数据库中尚无用户，请先运行 main.py 并完成一次登录/注册。")
    else:
        if not username:
            raise ValueError("请指定 username，或使用 all_users=True。")
        user = db.get_user_by_username(username)
        if not user:
            raise ValueError(
                f"用户不存在: {username}。请先运行 main.py 初始化数据库；"
                f"或改用 --all-users 给所有账号导入。"
            )
        users = [user]

    results: list[dict] = []
    for user in users:
        profile_id = _register_voice_for_user(user, meta, install)
        results.append(
            {
                "user_id": user["id"],
                "username": user["username"],
                "profile_id": profile_id,
                "voice_name": meta["voice_name"],
            }
        )
    return results


def verify_voice_pack(pack_id: str = "finetune_15", *, voice_name: str | None = None) -> dict:
    """检查声线包文件与数据库注册是否完整。"""
    from path_utils import resolve_project_path

    pack_dir = Path(BASE_DIR) / "deploy" / "voice_packs" / pack_id
    meta_path = pack_dir / "pack.json"
    report: dict = {
        "pack_dir": str(pack_dir),
        "pack_exists": pack_dir.is_dir(),
        "pack_json": meta_path.is_file(),
        "files_ok": {},
        "profiles_by_user": [],
        "issues": [],
    }

    if not report["pack_json"]:
        report["issues"].append(f"缺少 pack.json: {meta_path}")
        return report

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name = voice_name or meta.get("voice_name", "")
    install = meta.get("install", {})

    for key in ("gpt", "sovits", "reference"):
        rel = install.get(key, "")
        resolved = resolve_project_path(rel)
        ok = bool(resolved and Path(resolved).is_file())
        report["files_ok"][key] = {"rel": rel, "resolved": resolved, "ok": ok}
        if not ok:
            report["issues"].append(f"权重/参考音缺失 ({key}): {rel}")

    for user in db.list_users():
        profiles = [
            p
            for p in db.list_voice_profiles_by_user(user["id"], ready_only=True)
            if not name or p["name"] == name
        ]
        if profiles:
            report["profiles_by_user"].append(
                {"username": user["username"], "profiles": [p["name"] for p in profiles]}
            )

    if not report["profiles_by_user"]:
        report["issues"].append(
            f"数据库中没有任何用户拥有 ready 声线「{name}」。"
            "请运行: python scripts/import_voice_pack.py deploy/voice_packs/finetune_15"
        )

    return report
