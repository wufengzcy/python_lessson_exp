# 智声助手 · 环境安装指南

本文档说明如何在 **Windows** 上从零配置 **ChatTTS + PyTorch + 本项目** 的运行环境。

> 适用项目：[python_lessson_exp](https://github.com/wufengzcy/python_lessson_exp)（分支 `feature-Xiao`）

---

## 目录

1. [环境要求](#1-环境要求)
2. [获取代码](#2-获取代码)
3. [安装 Python 3.12](#3-安装-python-312)
4. [方式 A：一键脚本安装（推荐）](#4-方式-a一键脚本安装推荐)
5. [方式 B：手动逐步安装](#5-方式-b手动逐步安装)
6. [下载 ChatTTS 模型（必做）](#6-下载-chattts-模型必做)
7. [验证安装](#7-验证安装)
8. [启动程序](#8-启动程序)
9. [GPT-SoVITS 声线克隆（可选）](#9-gpt-sovits-声线克隆可选)
10. [常见问题](#10-常见问题)
11. [附录：路径对照表](#11-附录路径对照表)

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 |
| Python | **3.12.x**（必须，不要用 3.13 / 3.14） |
| 磁盘空间 | 约 5GB（PyTorch + 模型 + 缓存） |
| 显卡 | 可选；有 NVIDIA 显卡时合成更快 |
| 网络 | 首次需下载 PyTorch 与 ChatTTS 模型 |

### 显卡与 PyTorch 版本对照

| 显卡 | 推荐 PyTorch 源 |
|------|----------------|
| RTX 50 系（如 RTX 5060） | `cu130` |
| RTX 30 / 40 系 | `cu126` 或 `cu130` |
| 无 NVIDIA 显卡 | 官方 CPU 版（见下文） |

---

## 2. 获取代码

```powershell
git clone -b feature-Xiao https://github.com/wufengzcy/python_lessson_exp.git
cd python_lessson_exp
```

若已克隆，切换到正确分支：

```powershell
git checkout feature-Xiao
git pull
```

---

## 3. 安装 Python 3.12

### 方法 1：winget（推荐）

```powershell
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
```

安装完成后 **关闭并重新打开 PowerShell**，确认版本：

```powershell
py -3.12 --version
```

应显示 `Python 3.12.x`。

### 方法 2：官网安装

1. 打开 [Python 3.12 下载页](https://www.python.org/downloads/release/python-3120/)
2. 下载 Windows installer（64-bit）
3. 安装时勾选 **「Add python.exe to PATH」**

### 若 PowerShell 无法运行脚本

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 4. 方式 A：一键脚本安装（推荐）

项目自带 `setup_tts_env.ps1`，会自动：

- 创建虚拟环境 `.venv`
- 安装 PyTorch（cu130）与 ChatTTS
- 配置 pip / HuggingFace 缓存环境变量（默认 E 盘）

### 4.1 修改脚本路径（如需要）

用记事本打开 `setup_tts_env.ps1`，确认以下变量与你的电脑一致：

```powershell
$PythonRoot = "E:\Python312"          # Python 3.12 安装目录
$ProjectDir = "E:\...\voice_assistant" # 本项目目录（clone 后的路径）
$CacheRoot  = "E:\python_cache"        # 缓存目录，可改为 C:\python_cache
```

若 Python 是通过 winget / 默认路径安装的，常见路径为：

```
C:\Users\<用户名>\AppData\Local\Programs\Python\Python312\python.exe
```

此时可将 `$PythonRoot` 改为上述目录的上一级，或改用手动安装（方式 B）。

### 4.2 执行脚本

```powershell
cd <你的项目目录>
powershell -ExecutionPolicy Bypass -File .\setup_tts_env.ps1
```

脚本结束后应看到类似输出：

```
torch 2.x.x+cu130 cuda 13.0 ok True
```

> **注意**：脚本末尾的 `ChatTTS OK` 可能仍提示模型下载失败，这是正常的——模型需按 [第 6 节](#6-下载-chattts-模型必做) 单独下载。

---

## 5. 方式 B：手动逐步安装

适合路径不在 E 盘、或希望完全掌控每一步的同学。

### 5.1 进入项目并创建虚拟环境

```powershell
cd <你的项目目录>

# 创建 venv（任选一种）
py -3.12 -m venv .venv
# 或: E:\Python312\python.exe -m venv .venv

.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python --version
```

确认输出为 **Python 3.12.x**，且命令行前出现 `(.venv)`。

### 5.2 配置镜像与缓存（推荐，国内网络）

```powershell
$CacheRoot = "E:\python_cache"   # 可改为 C:\python_cache

New-Item -ItemType Directory -Force -Path "$CacheRoot\pip", "$CacheRoot\huggingface", "$CacheRoot\tmp" | Out-Null

$env:PIP_CACHE_DIR = "$CacheRoot\pip"
$env:HF_HOME = "$CacheRoot\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

如需永久生效，可在「系统环境变量」中添加：

| 变量名 | 值 |
|--------|-----|
| `PIP_CACHE_DIR` | `E:\python_cache\pip` |
| `HF_HOME` | `E:\python_cache\huggingface` |
| `HF_ENDPOINT` | `https://hf-mirror.com` |

### 5.3 安装 PyTorch

**NVIDIA 显卡 — RTX 50 系（cu130）：**

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**NVIDIA 显卡 — RTX 30 / 40 系（cu126）：**

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**无 NVIDIA 显卡（CPU 版）：**

```powershell
pip install torch torchaudio
```

### 5.4 安装 ChatTTS 及依赖

```powershell
pip install ChatTTS numpy soundfile huggingface_hub requests
```

### 5.5 验证 GPU（有显卡时）

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu_ok', torch.cuda.is_available())"
```

期望：`gpu_ok True`。

---

## 6. 下载 ChatTTS 模型（必做）

模型约 **1GB**，**不在 Git 仓库中**，每位同学需本地下载一次。

ChatTTS 自带下载器直连 `huggingface.co`，国内容易失败，请用 **镜像 + huggingface_hub** 下载。

### 6.1 镜像下载（推荐）

```powershell
cd <你的项目目录>
.\.venv\Scripts\Activate.ps1
$env:HF_ENDPOINT = "https://hf-mirror.com"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='2Noise/ChatTTS', local_dir='asset')"
```

下载完成后，权重位于：

```
asset/asset/Decoder.safetensors
asset/asset/DVAE.safetensors
asset/asset/Embed.safetensors
asset/asset/Vocos.safetensors
asset/asset/gpt/model.safetensors
asset/asset/tokenizer/...
```

（多一层 `asset/asset/` 是正常现象。）

### 6.2 从同学处拷贝（省流量）

若组内已有同学下载完成，直接复制整个 **`asset`** 文件夹到项目根目录即可，跳过 6.1。

### 6.3 浏览器手动下载

打开镜像站模型页：[hf-mirror.com/2Noise/ChatTTS](https://hf-mirror.com/2Noise/ChatTTS/tree/main)

下载 `asset` 目录下全部文件，按相同目录结构放入本项目 `asset/` 文件夹。

---

## 7. 验证安装

### 7.1 验证模型加载

```powershell
cd <你的项目目录>
.\.venv\Scripts\Activate.ps1

python -c "import ChatTTS; from pathlib import Path; p=str(Path('asset').resolve()); c=ChatTTS.Chat(); print('load', c.load(compile=False, source='custom', custom_path=p))"
```

期望输出：`load True`

### 7.2 验证语音合成

```powershell
python -c @"
import ChatTTS, soundfile as sf
from pathlib import Path
p = str(Path('asset').resolve())
chat = ChatTTS.Chat()
chat.load(compile=False, source='custom', custom_path=p)
text = '你好，这是 ChatTTS 环境测试，安装成功。'
wav = chat.infer([text])[0]
sf.write('test_tts.wav', wav, 24000)
print('已生成 test_tts.wav')
"@
```

用播放器打开项目目录下的 **`test_tts.wav`**，能听到中文即表示环境 OK。

---

## 8. 启动程序

每次使用前：

```powershell
cd <你的项目目录>
.\.venv\Scripts\Activate.ps1
$env:HF_ENDPOINT = "https://hf-mirror.com"

python main.py
```

### 默认账号

| 用户名 | 密码 | 角色 |
|--------|------|------|
| `admin` | `admin123` | 管理员 |

也可在登录界面自行注册普通用户。

### 程序功能简述

1. 登录后输入要合成的文字
2. 点击 **「开始合成」**（首次会加载模型，约数秒）
3. 合成完成后可 **播放**、**导出 WAV**
4. 历史记录保存在本地 SQLite，可搜索与双击回填

合成音频默认保存在 `data/outputs/`（首次运行自动创建）。

---

## 9. GPT-SoVITS 声线克隆（可选）

项目已集成 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 源码（`engines/GPT-SoVITS`），通过 **本地 Python import / subprocess 调训练脚本** 实现，**不依赖 HTTP API**。

### 9.1 安装 GPT-SoVITS 引擎

在已完成 ChatTTS 环境的基础上：

```powershell
cd <项目目录>
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\setup_sovits.ps1
```

脚本会 clone 仓库并下载 v2 预训练底模（数 GB，需较长时间）。

额外依赖（若缺失）：

```powershell
pip install pyyaml peft librosa
```

### 9.2 使用流程

1. 启动 `python main.py` 并登录  
2. 点击 **「我的声线」**  
3. 导入参考 WAV（3～60 秒干净人声）+ 填写参考文本  
4. **零样本**：点「保存为零样本声线」（用底模 + 参考音频克隆，最快）  
5. **微调**：点「开始微调训练」（调用 GPT-SoVITS 自带训练脚本）  
6. 回到主界面，引擎选 **「GPT-SoVITS 克隆」**，选择声线后合成  

---

## 10. 常见问题

### Q1：`ModuleNotFoundError: pybase16384.backends.cython._core`

**原因**：Python 版本过新（3.13 / 3.14），ChatTTS 依赖不兼容。  
**解决**：卸载旧 venv，用 **Python 3.12** 重建 `.venv`。

```powershell
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
# 然后重新安装依赖
```

### Q2：`pip install torch ... cu124` 找不到包

**原因**：Python 3.12+ 或新显卡不支持 cu124。  
**解决**：改用 **cu130** 或 **cu126**（见第 5.3 节）。

### Q3：`load False` 或模型下载 connection reset

**原因**：ChatTTS 内置下载器不走镜像。  
**解决**：按 [第 6 节](#6-下载-chattts-模型必做) 用 `snapshot_download` 下载，或拷贝同学的 `asset` 文件夹。

### Q4：合成时报 `torchcodec` / `torchaudio.save` 错误

**原因**：新版 torchaudio 保存方式变更。  
**解决**：本项目已用 `soundfile` 保存，请 `pip install soundfile`；不要单独改回 `torchaudio.save`。

### Q5：文本里有换行，合成异常或漏读

**原因**：ChatTTS 不支持换行符。  
**解决**：输入框中避免多行；程序内部会自动将换行替换为空格。

### Q6：`pip` 报 `Cache entry deserialization failed`

**原因**：pip 缓存损坏。  
**解决**：可忽略（不影响安装）；或删除 `$env:PIP_CACHE_DIR` 对应目录后重试。

### Q7：RTX 5060 提示 sm_120 不兼容

**原因**：PyTorch 版本与显卡架构不匹配。  
**解决**：安装 **cu130** 版 PyTorch（见 5.3 节），并更新 NVIDIA 驱动到最新。

---

## 11. 附录：路径对照表

以下为组内参考配置（可按自己电脑修改）：

| 用途 | 默认路径 |
|------|----------|
| 项目目录 | `E:\cursor_files\python_homeworks\voice_assistant` |
| Python 3.12 | `E:\Python312\python.exe` |
| 虚拟环境 | `<项目目录>\.venv` |
| pip 缓存 | `E:\python_cache\pip` |
| HuggingFace 缓存 | `E:\python_cache\huggingface` |
| ChatTTS 模型 | `<项目目录>\asset\` |
| GPT-SoVITS 源码 | `<项目目录>\engines\GPT-SoVITS\` |
| 用户声线样本 | `<项目目录>\data\voices\` |
| 合成输出 | `<项目目录>\data\outputs\` |
| 本地数据库 | `<项目目录>\data\app.db` |

---

## 依赖清单（参考）

手动安装时可对照：

```
torch + torchaudio  (cu130 / cu126 / CPU)
ChatTTS
numpy
soundfile
huggingface_hub
requests
```

Tkinter 随 Python 标准库提供，无需额外安装。

---

如有问题，请在组内 Issue 或联系分支维护者（feature-Xiao）。
