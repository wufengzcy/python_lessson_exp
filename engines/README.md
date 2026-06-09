# GPT-SoVITS 引擎（本地集成）

本目录用于放置 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 源码，由智声助手直接 `import` 其 Python 模块进行推理与训练（**不调用 HTTP API**）。

## 首次安装

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_sovits.ps1
```

或手动：

```powershell
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git engines/GPT-SoVITS
cd engines/GPT-SoVITS
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Device CU126 -Source HF-Mirror
```

安装完成后，`GPT_SoVITS/pretrained_models/` 下应有 v2 底模。

## 说明

- `engines/GPT-SoVITS/` 已加入 `.gitignore`（体积过大）
- 训练日志与权重默认在 `engines/GPT-SoVITS/logs/` 与 `SoVITS_weights_v2/`、`GPT_weights_v2/`
