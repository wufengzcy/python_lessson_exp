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

`deploy/voice_packs/finetune_15/`（项目根目录下，不是 voice_assistant 子目录）

然后执行：

```powershell
cd python_lessson_exp
.\.venv\Scripts\Activate.ps1
python scripts/import_voice_pack.py deploy/voice_packs/finetune_15
python scripts/verify_voice_deploy.py
```

打开 main.py → 引擎选 **GPT-SoVITS 克隆** → 声线 **15样本训练 (finetuned)**。

默认会给数据库里所有用户都注册该声线；仅导入 admin 时用 `--user admin`。

无需改 app.db、无需手动选择 .ckpt 文件。
