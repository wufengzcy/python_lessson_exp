# 声线包离线分享

把 `finetune_15` 整个文件夹（约 230MB）用 U 盘/网盘发给同学，**不要提交 Git**。

## 你（打包）

```powershell
cd voice_assistant
.\.venv\Scripts\Activate.ps1
python scripts/pack_finetune_15.py
```

生成目录：`deploy/voice_packs/finetune_15/`

## 同学（导入，一条命令）

先按 INSTALL.md 装好环境和 GPT-SoVITS，再把收到的 `finetune_15` 文件夹放到：

`voice_assistant/deploy/voice_packs/finetune_15/`

然后执行：

```powershell
cd voice_assistant
.\.venv\Scripts\Activate.ps1
python scripts/import_voice_pack.py deploy/voice_packs/finetune_15
```

打开 main.py → 引擎选 **GPT-SoVITS 克隆** → 声线 **15样本训练**。

无需改 app.db、无需替换路径。
