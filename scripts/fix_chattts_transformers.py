"""修复 ChatTTS 与 transformers 版本冲突。

报错示例: 'DynamicCache' object has no attribute 'layers'

用法（在项目根目录）:
  .\\.venv\\Scripts\\Activate.ps1
  python scripts/fix_chattts_transformers.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_PIN = "transformers>=4.46,<=4.50"


def main() -> int:
    pip = ROOT / ".venv" / "Scripts" / "pip.exe"
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not pip.is_file():
        print("未找到 .venv，请先运行 setup_tts_env.ps1")
        return 1

    print(f"安装兼容版本: {TRANSFORMERS_PIN}")
    subprocess.check_call([str(pip), "install", TRANSFORMERS_PIN])

    sys.path.insert(0, str(ROOT))
    from tts_core import get_engine, normalize_text

    print("验证 transformers ...")
    subprocess.check_call(
        [str(python), "-c", "import transformers; print('transformers', transformers.__version__)"]
    )

    print("验证 ChatTTS 合成（短句）...")
    wav = get_engine().infer([normalize_text("你好，ChatTTS 环境测试。")])[0]
    import numpy as np

    arr = np.asarray(wav)
    if arr.size == 0:
        print("合成返回空音频，请检查 asset/ 模型是否完整")
        return 1

    print(f"ChatTTS OK，样本长度 {len(arr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
