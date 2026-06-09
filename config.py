import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
VOICES_DIR = os.path.join(DATA_DIR, "voices")
ASSET_DIR = Path(BASE_DIR) / "asset"

GPT_SOVITS_DIR = Path(BASE_DIR) / "engines" / "GPT-SoVITS"
GPT_SOVITS_EXP_ROOT = GPT_SOVITS_DIR / "logs"

DB_PATH = os.path.join(DATA_DIR, "app.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

TTS_MODEL_NAME = "ChatTTS"
TTS_ENGINE_CHAT = "chattts"
TTS_ENGINE_SOVITS = "gpt-sovits"
TTS_SAMPLE_RATE = 24000
DEFAULT_LANGUAGE = "zh"

SOVITS_VERSION = "v2"
SOVITS_TEXT_LANG = "zh"
SOVITS_PROMPT_LANG = "zh"
