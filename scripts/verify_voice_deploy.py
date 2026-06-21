"""检查 15 样本声线包是否已正确部署。用法: python scripts/verify_voice_deploy.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db
from voice_pack import verify_voice_pack


def main() -> int:
    db.init_db()
    report = verify_voice_pack()

    print("=" * 60)
    print("智声助手 · 声线部署检查")
    print("=" * 60)

    ok = True
    if not report["pack_exists"]:
        print(f"[失败] 声线包目录不存在: {report['pack_dir']}")
        print("       请将 finetune_15 放到 deploy/voice_packs/finetune_15/")
        ok = False
    else:
        print(f"[OK] 声线包目录: {report['pack_dir']}")

    for key, info in report.get("files_ok", {}).items():
        if info["ok"]:
            print(f"[OK] {key}: {info['resolved']}")
        else:
            print(f"[失败] {key} 缺失: {info['rel']}")
            ok = False

    users = db.list_users()
    print(f"\n数据库用户 ({len(users)}): {', '.join(u['username'] for u in users) or '（无）'}")

    if report.get("profiles_by_user"):
        print("\n已注册声线:")
        for row in report["profiles_by_user"]:
            print(f"  {row['username']}: {', '.join(row['profiles'])}")
    else:
        print("\n[失败] 没有任何用户拥有 ready 声线")
        print("       运行: python scripts/import_voice_pack.py deploy/voice_packs/finetune_15")
        ok = False

    print("\n" + "=" * 60)
    if ok:
        print("检查通过。启动 main.py 后:")
        print("  引擎 → GPT-SoVITS 克隆")
        print("  声线 → 15样本训练 (finetuned)")
        return 0

    print("存在问题，请按上面提示修复。")
    for issue in report.get("issues", []):
        print(f"  · {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
