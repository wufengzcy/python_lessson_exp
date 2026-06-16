"""Apply GPT-SoVITS patches required on Windows / PyTorch 2.6+ (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENGINE = PROJECT_DIR / "engines" / "GPT-SoVITS" / "GPT_SoVITS"
S2_TRAIN = ENGINE / "s2_train.py"
S1_TRAIN = ENGINE / "s1_train.py"
DDP_MARKER = "voice_assistant: Windows single-GPU skips DDP"
TORCH_LOAD_MARKER = "voice_assistant: torch.load weights_only=False"
S1_FIT_MARKER = "weights_only=False"


def _insert_after(text: str, anchor: str, insert: str) -> str | None:
    if insert.strip() in text:
        return text
    if anchor not in text:
        return None
    return text.replace(anchor, anchor + insert, 1)


def patch_torch_load(script: Path) -> bool:
    if not script.is_file():
        print(f"skip: {script} not found")
        return False
    text = script.read_text(encoding="utf-8")
    if TORCH_LOAD_MARKER in text:
        print(f"{script.name}: torch.load patch already applied")
        return True
    block = (
        f"\n\n# {TORCH_LOAD_MARKER}\n"
        "def _patch_torch_load() -> None:\n"
        "    _orig_load = torch.load\n\n"
        "    def _load(*args, **kwargs):\n"
        '        if kwargs.get("weights_only") is None:\n'
        "            kwargs[\"weights_only\"] = False\n"
        "        return _orig_load(*args, **kwargs)\n\n"
        "    torch.load = _load  # type: ignore[method-assign]\n\n\n"
        "_patch_torch_load()\n"
    )
    updated = _insert_after(text, "import torch\n", block)
    if updated is None:
        print(f"{script.name}: could not insert torch.load patch")
        return False
    script.write_text(updated, encoding="utf-8")
    print(f"{script.name}: applied torch.load patch")
    return True


def patch_s2_train() -> bool:
    if not S2_TRAIN.is_file():
        print(f"skip: {S2_TRAIN} not found")
        return False
    text = S2_TRAIN.read_text(encoding="utf-8")
    if DDP_MARKER in text:
        print("s2_train.py: Windows DDP patch already applied")
        return True

    old_helper_anchor = 'device = "cpu"  # cuda以外的设备，等mps优化后加入\n\n\ndef main():'
    new_helper = (
        'device = "cpu"  # cuda以外的设备，等mps优化后加入\n\n\n'
        "def _model_ref(model):\n"
        '    return model.module if hasattr(model, "module") else model\n\n\n'
        "def main():"
    )
    if old_helper_anchor not in text:
        print("s2_train.py: unexpected layout, manual patch may be needed")
        return False
    text = text.replace(old_helper_anchor, new_helper, 1)

    old_ddp = (
        "    if torch.cuda.is_available():\n"
        "        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)\n"
        "        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)\n"
        "    else:\n"
        "        net_g = net_g.to(device)\n"
        "        net_d = net_d.to(device)"
    )
    new_ddp = (
        f"    # {DDP_MARKER}\n"
        "    use_ddp = torch.cuda.is_available() and not (os.name == \"nt\" and n_gpus <= 1)\n"
        "    if use_ddp:\n"
        "        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)\n"
        "        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=False)\n"
        "    elif torch.cuda.is_available():\n"
        "        net_g = net_g.cuda(rank)\n"
        "        net_d = net_d.cuda(rank)\n"
        "    else:\n"
        "        net_g = net_g.to(device)\n"
        "        net_d = net_d.to(device)"
    )
    if old_ddp not in text:
        print("s2_train.py: DDP block not found, manual patch may be needed")
        return False
    text = text.replace(old_ddp, new_ddp, 1)

    old_g_load = (
        "                net_g.module.load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2G, map_location=\"cpu\", weights_only=False)[\"weight\"],\n"
        "                    strict=False,\n"
        "                )\n"
        "                if torch.cuda.is_available()\n"
        "                else net_g.load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2G, map_location=\"cpu\", weights_only=False)[\"weight\"],\n"
        "                    strict=False,\n"
        "                ),"
    )
    new_g_load = (
        "                _model_ref(net_g).load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2G, map_location=\"cpu\", weights_only=False)[\"weight\"],\n"
        "                    strict=False,\n"
        "                ),"
    )
    if old_g_load in text:
        text = text.replace(old_g_load, new_g_load, 1)

    old_d_load = (
        "                net_d.module.load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2D, map_location=\"cpu\", weights_only=False)[\"weight\"], strict=False\n"
        "                )\n"
        "                if torch.cuda.is_available()\n"
        "                else net_d.load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2D, map_location=\"cpu\", weights_only=False)[\"weight\"],\n"
        "                ),"
    )
    new_d_load = (
        "                _model_ref(net_d).load_state_dict(\n"
        "                    torch.load(hps.train.pretrained_s2D, map_location=\"cpu\", weights_only=False)[\"weight\"], strict=False\n"
        "                ),"
    )
    if old_d_load in text:
        text = text.replace(old_d_load, new_d_load, 1)

    S2_TRAIN.write_text(text, encoding="utf-8")
    print("s2_train.py: applied Windows DDP patch")
    return True


def patch_s1_train() -> bool:
    if not S1_TRAIN.is_file():
        print(f"skip: {S1_TRAIN} not found")
        return False
    text = S1_TRAIN.read_text(encoding="utf-8")
    changed = False
    old_fit = "        trainer.fit(model, data_module, ckpt_path=ckpt_path)"
    new_fit = (
        "        trainer.fit(\n"
        "            model,\n"
        "            data_module,\n"
        "            ckpt_path=ckpt_path,\n"
        "            weights_only=False,\n"
        "        )"
    )
    if old_fit in text:
        text = text.replace(old_fit, new_fit, 1)
        changed = True
        print("s1_train.py: trainer.fit uses weights_only=False")
    elif S1_FIT_MARKER in text and "trainer.fit(" in text:
        print("s1_train.py: trainer.fit patch already applied")
    S1_TRAIN.write_text(text, encoding="utf-8")
    return changed or S1_FIT_MARKER in text


def main() -> int:
    results = [
        patch_s2_train(),
        patch_s1_train(),
        patch_torch_load(S2_TRAIN),
        patch_torch_load(S1_TRAIN),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
