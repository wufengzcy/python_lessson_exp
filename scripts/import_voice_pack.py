"""导入声线包（自动复制权重并用相对路径写入数据库）。用法: python scripts/import_voice_pack.py deploy/voice_packs/finetune_15"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db
from voice_pack import import_voice_pack


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/import_voice_pack.py deploy/voice_packs/finetune_15")
        return 1
    db.init_db()
    pack_dir = Path(sys.argv[1])
    if not pack_dir.is_absolute():
        pack_dir = ROOT / pack_dir
    profile_id = import_voice_pack(pack_dir)
    print(f"导入成功，声线 ID={profile_id}。请打开 main.py，引擎选 GPT-SoVITS 克隆即可使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
