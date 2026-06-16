"""打包 4 样本微调声线，供同学离线拷贝。用法: python scripts/pack_finetune_4.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_pack import pack_finetune_4


def main() -> int:
    out = pack_finetune_4()
    print(f"已打包到: {out}")
    print("请把整个 finetune_4 文件夹发给同学（约 230MB），不要提交到 Git。")
    print("同学导入: python scripts/import_voice_pack.py deploy/voice_packs/finetune_4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
