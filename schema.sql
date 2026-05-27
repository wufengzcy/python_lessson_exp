-- 本地语音笔记与指令助手 - SQLite 建表脚本
-- 执行方式: sqlite3 data/app.db < schema.sql
-- 或在 db.py 的 init_db() 中自动执行

PRAGMA foreign_keys = ON;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    salt            TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 转写记录表
CREATE TABLE IF NOT EXISTS transcriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    audio_path      TEXT    NOT NULL,
    text            TEXT,
    duration_sec    REAL    DEFAULT 0,
    model_name      TEXT    DEFAULT 'tiny',
    language        TEXT    DEFAULT 'zh',
    status          TEXT    NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    error_message   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 指令执行日志
CREATE TABLE IF NOT EXISTS command_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    source_text     TEXT    NOT NULL,
    command_name    TEXT    NOT NULL,
    command_args    TEXT,
    success         INTEGER NOT NULL DEFAULT 0 CHECK (success IN (0, 1)),
    message         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 系统操作日志（管理员查看）
CREATE TABLE IF NOT EXISTS operation_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,
    action          TEXT    NOT NULL,
    detail          TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 索引（检索、分页）
CREATE INDEX IF NOT EXISTS idx_transcriptions_user_id ON transcriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON transcriptions(created_at);
CREATE INDEX IF NOT EXISTS idx_transcriptions_text ON transcriptions(text);
CREATE INDEX IF NOT EXISTS idx_command_logs_user_id ON command_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_operation_logs_created_at ON operation_logs(created_at);

-- 默认管理员由 db.py 在 init_db() 中创建（密码需在代码里哈希后插入）
