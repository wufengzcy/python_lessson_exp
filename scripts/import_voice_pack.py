"""导入声线包（自动复制权重并用相对路径写入数据库）。

用法:
  python scripts/import_voice_pack.py deploy/voice_packs/finetune_15
  python scripts/import_voice_pack.py deploy/voice_packs/finetune_15 --user admin
  python scripts/import_voice_pack.py deploy/voice_packs/finetune_15 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db
from path_utils import resolve_project_path
from voice_pack import import_voice_pack, verify_voice_pack


def _print_verify(report: dict) -> int:
    print(f"声线包目录: {report['pack_dir']}")
    print(f"pack.json: {'OK' if report['pack_json'] else '缺失'}")
    for key, info in report.get("files_ok", {}).items():
        status = "OK" if info["ok"] else "缺失"
        print(f"  {key}: [{status}] {info['rel']}")
        if info.get("resolved"):
            print(f"         -> {info['resolved']}")

    if report.get("profiles_by_user"):
        print("数据库声线:")
        for row in report["profiles_by_user"]:
            names = ", ".join(row["profiles"])
            print(f"  用户 {row['username']}: {names}")
    else:
        print("数据库声线: （无）")

    if report.get("issues"):
        print("\n问题:")
        for issue in report["issues"]:
            print(f"  - {issue}")
        return 1

    print("\n检查通过。请重启 main.py，引擎选「GPT-SoVITS 克隆」，声线选对应名称。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="导入离线声线包")
    parser.add_argument(
        "pack_dir",
        nargs="?",
        default="deploy/voice_packs/finetune_15",
        help="声线包目录（含 pack.json）",
    )
    parser.add_argument("--user", help="仅导入到指定用户名（默认导入到所有用户）")
    parser.add_argument(
        "--all-users",
        action="store_true",
        default=None,
        help="导入到数据库中所有用户（默认行为）",
    )
    parser.add_argument("--verify", action="store_true", help="仅检查文件与数据库，不导入")
    args = parser.parse_args()

    db.init_db()

    if args.verify:
        return _print_verify(verify_voice_pack())

    pack_dir = Path(args.pack_dir)
    if not pack_dir.is_absolute():
        pack_dir = ROOT / pack_dir

    all_users = True
    if args.user:
        all_users = False
    elif args.all_users is False:
        all_users = False

    try:
        results = import_voice_pack(pack_dir, username=args.user, all_users=all_users)
    except (FileNotFoundError, ValueError) as exc:
        print(f"导入失败: {exc}")
        return 1

    voice_name = results[0]["voice_name"] if results else "（未知）"
    print(f"导入成功，声线「{voice_name}」已注册到以下账号:")
    for row in results:
        print(f"  - {row['username']} (profile_id={row['profile_id']})")

    sample = db.get_voice_profile(results[0]["profile_id"]) if results else None
    if sample:
        for key in ("gpt_weights_path", "sovits_weights_path", "ref_audio_path"):
            rel = sample.get(key)
            resolved = resolve_project_path(rel)
            ok = resolved and Path(resolved).is_file()
            print(f"  {key}: {'OK' if ok else '缺失'} -> {rel}")

    print(
        "\n下一步:\n"
        "  1. 若 main.py 已打开，请关闭后重新启动\n"
        "  2. 用上面列出的任意账号登录\n"
        "  3. 主界面 → 合成引擎 →「GPT-SoVITS 克隆」\n"
        f"  4. 声线 →「{voice_name} (finetuned)」\n"
        "  5. 仍看不到时运行: python scripts/import_voice_pack.py --verify"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
