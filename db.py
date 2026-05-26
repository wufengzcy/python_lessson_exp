"""
数据库模块 db.py

函数列表（按模块分组）:

【初始化】
  init_db()                         建表、建目录、创建默认管理员
  get_connection()                  获取 sqlite3 连接（row_factory=Row）

【用户 users - CRUD】
  create_user(username, password, role='user') -> int | None
  get_user_by_username(username) -> dict | None
  get_user_by_id(user_id) -> dict | None
  list_users() -> list[dict]                    # 管理员
  update_user_password(user_id, new_password) -> bool
  update_user_role(user_id, role) -> bool       # 管理员
  delete_user(user_id) -> bool                  # 管理员，不可删最后一个 admin

【转写 transcriptions - CRUD】
  create_transcription(user_id, audio_path, ...) -> int
  update_transcription_result(transcription_id, text, duration_sec, status, error_message=None) -> bool
  get_transcription(transcription_id) -> dict | None
  list_transcriptions_by_user(user_id, limit=50, offset=0) -> list[dict]
  list_all_transcriptions(limit=50, offset=0) -> list[dict]   # 管理员
  search_transcriptions(user_id, keyword, limit=50) -> list[dict]
  delete_transcription(transcription_id, user_id=None) -> bool  # user_id 用于权限校验

【指令日志 command_logs】
  create_command_log(user_id, source_text, command_name, ...) -> int
  list_command_logs_by_user(user_id, limit=50) -> list[dict]
  list_all_command_logs(limit=100) -> list[dict]                # 管理员

【操作日志 operation_logs】
  create_operation_log(user_id, action, detail=None) -> int
  list_operation_logs(limit=100) -> list[dict]                  # 管理员

【统计（进阶可视化用）】
  count_transcriptions_by_day(user_id=None, days=7) -> list[dict]
  count_commands_by_name(user_id=None, days=7) -> list[dict]
"""

import hashlib
import os
import secrets
import sqlite3
from typing import Any

from config import (
    DATA_DIR,
    DB_PATH,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    SCHEMA_PATH,
)


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def init_db() -> None:
    """初始化数据库：执行 schema.sql，创建默认管理员。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_connection()
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()

        admin = get_user_by_username(DEFAULT_ADMIN_USERNAME)
        if admin is None:
            create_user(
                DEFAULT_ADMIN_USERNAME,
                DEFAULT_ADMIN_PASSWORD,
                role="admin",
            )
            create_operation_log(
                None,
                "init_db",
                f"创建默认管理员 {DEFAULT_ADMIN_USERNAME}",
            )
    finally:
        conn.close()


# ---------- users ----------


def create_user(username: str, password: str, role: str = "user") -> int | None:
    try:
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        conn = get_connection()
        try:
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, role)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, salt, role),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except sqlite3.IntegrityError:
        return None


def verify_user(username: str, password: str) -> dict | None:
    user = get_user_by_username(username)
    if user is None:
        return None
    if _hash_password(password, user["salt"]) != user["password_hash"]:
        return None
    return user


def get_user_by_username(username: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_users() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_user_password(user_id: int, new_password: str) -> bool:
    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE users
            SET password_hash = ?, salt = ?, updated_at = datetime('now', 'localtime')
            WHERE id = ?
            """,
            (password_hash, salt, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_user_role(user_id: int, role: str) -> bool:
    if role not in ("user", "admin"):
        return False
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE users SET role = ?, updated_at = datetime('now', 'localtime')
            WHERE id = ?
            """,
            (role, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_user(user_id: int) -> bool:
    user = get_user_by_id(user_id)
    if user is None:
        return False
    if user["role"] == "admin":
        conn = get_connection()
        try:
            admin_count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            ).fetchone()[0]
            if admin_count <= 1:
                return False
        finally:
            conn.close()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------- transcriptions ----------


def create_transcription(
    user_id: int,
    audio_path: str,
    model_name: str = "tiny",
    language: str = "zh",
    status: str = "pending",
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO transcriptions (user_id, audio_path, model_name, language, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, audio_path, model_name, language, status),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_transcription_result(
    transcription_id: int,
    text: str,
    duration_sec: float,
    status: str,
    error_message: str | None = None,
) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE transcriptions
            SET text = ?, duration_sec = ?, status = ?, error_message = ?
            WHERE id = ?
            """,
            (text, duration_sec, status, error_message, transcription_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_transcription(transcription_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?",
            (transcription_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_transcriptions_by_user(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, audio_path, text, duration_sec, model_name, status, created_at
            FROM transcriptions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_all_transcriptions(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT t.*, u.username
            FROM transcriptions t
            JOIN users u ON t.user_id = u.id
            ORDER BY t.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_transcriptions(
    user_id: int,
    keyword: str,
    limit: int = 50,
) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, audio_path, text, duration_sec, status, created_at
            FROM transcriptions
            WHERE user_id = ? AND text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, f"%{keyword}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_transcription(transcription_id: int, user_id: int | None = None) -> bool:
    conn = get_connection()
    try:
        if user_id is not None:
            cur = conn.execute(
                "DELETE FROM transcriptions WHERE id = ? AND user_id = ?",
                (transcription_id, user_id),
            )
        else:
            cur = conn.execute(
                "DELETE FROM transcriptions WHERE id = ?",
                (transcription_id,),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------- command_logs ----------


def create_command_log(
    user_id: int,
    source_text: str,
    command_name: str,
    success: bool,
    command_args: str | None = None,
    message: str | None = None,
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO command_logs
            (user_id, source_text, command_name, command_args, success, message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                source_text,
                command_name,
                command_args,
                1 if success else 0,
                message,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_command_logs_by_user(user_id: int, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM command_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_all_command_logs(limit: int = 100) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT c.*, u.username
            FROM command_logs c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------- operation_logs ----------


def create_operation_log(
    user_id: int | None,
    action: str,
    detail: str | None = None,
) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO operation_logs (user_id, action, detail)
            VALUES (?, ?, ?)
            """,
            (user_id, action, detail),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_operation_logs(limit: int = 100) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT o.*, u.username
            FROM operation_logs o
            LEFT JOIN users u ON o.user_id = u.id
            ORDER BY o.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------- stats ----------


def count_transcriptions_by_day(
    user_id: int | None = None,
    days: int = 7,
) -> list[dict]:
    conn = get_connection()
    try:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT date(created_at) AS day, COUNT(*) AS cnt
                FROM transcriptions
                WHERE created_at >= datetime('now', 'localtime', ?)
                GROUP BY date(created_at)
                ORDER BY day
                """,
                (f"-{days} days",),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT date(created_at) AS day, COUNT(*) AS cnt
                FROM transcriptions
                WHERE user_id = ? AND created_at >= datetime('now', 'localtime', ?)
                GROUP BY date(created_at)
                ORDER BY day
                """,
                (user_id, f"-{days} days"),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_commands_by_name(
    user_id: int | None = None,
    days: int = 7,
) -> list[dict]:
    conn = get_connection()
    try:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT command_name, COUNT(*) AS cnt
                FROM command_logs
                WHERE created_at >= datetime('now', 'localtime', ?)
                GROUP BY command_name
                ORDER BY cnt DESC
                """,
                (f"-{days} days",),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT command_name, COUNT(*) AS cnt
                FROM command_logs
                WHERE user_id = ? AND created_at >= datetime('now', 'localtime', ?)
                GROUP BY command_name
                ORDER BY cnt DESC
                """,
                (user_id, f"-{days} days"),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
