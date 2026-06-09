# 智声助手 · ChatTTS 课设项目

基于 **Python + Tkinter + SQLite + ChatTTS** 的 AI 文本转语音桌面应用。

## 功能

- 用户登录 / 注册
- 文本转语音合成、播放、导出（**ChatTTS** + **GPT-SoVITS 声线克隆**）
- 用户录制/导入参考音频，零样本克隆或本地微调训练
- 合成历史记录与搜索
- 管理员后台（用户、记录、操作日志）

## 环境要求

- Windows 10+
- Python **3.12**（不要用 3.14）
- NVIDIA 显卡（可选，推荐 cu130 版 PyTorch）

## 快速开始

详细步骤见 **[INSTALL.md](./INSTALL.md)**（含 Python 安装、PyTorch、模型下载、排错）。

```powershell
cd python_lessson_exp
powershell -ExecutionPolicy Bypass -File .\setup_tts_env.ps1
# 按 INSTALL.md 第 6 节下载模型到 asset/
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
├── sovits_core.py    # GPT-SoVITS 推理封装
├── sovits_train.py   # GPT-SoVITS 微调脚本调用
├── setup_sovits.ps1  # GPT-SoVITS 安装
├── schema.sql        # 建表脚本
├── setup_tts_env.ps1 # 环境一键安装脚本
├── INSTALL.md        # 详细安装文档
└── ui/               # 界面
```
