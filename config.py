import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
ASSET_DIR = Path(BASE_DIR) / "asset"

DB_PATH = os.path.join(DATA_DIR, "app.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

TTS_MODEL_NAME = "ChatTTS"
TTS_SAMPLE_RATE = 24000
DEFAULT_LANGUAGE = "zh"
