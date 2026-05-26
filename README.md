# 智声助手 · ChatTTS 课设项目

基于 **Python + Tkinter + SQLite + ChatTTS** 的 AI 文本转语音桌面应用。

## 功能

- 用户登录 / 注册
- 文本转语音合成、播放、导出
- 合成历史记录与搜索
- 管理员后台（用户、记录、操作日志）

## 环境要求

- Windows 10+
- Python **3.12**（不要用 3.14）
- NVIDIA 显卡（可选，推荐 cu130 版 PyTorch）

## 快速开始

```powershell
cd voice_assistant
powershell -ExecutionPolicy Bypass -File .\setup_tts_env.ps1
```

模型需单独下载到 `asset/` 目录，详见 `setup_tts_env.ps1` 或课设文档。

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

默认管理员：`admin` / `admin123`

## 项目结构

```
voice_assistant/
├── main.py           # 入口
├── config.py         # 配置
├── db.py             # 数据库
├── tts_core.py       # ChatTTS 封装
├── schema.sql        # 建表脚本
├── setup_tts_env.ps1 # 环境安装脚本
└── ui/               # 界面
```
