"""训练看门狗：实时刷新进度，检测长时间无响应并立刻报错。"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str], None]

POLL_SECONDS = 15
STALL_SECONDS_PREP = 10 * 60
STALL_SECONDS_TRAIN = 8 * 60


class TrainingStalledError(RuntimeError):
    """训练 subprocess 长时间无任何文件/日志更新。"""


def _path_stat(path: Path) -> tuple[float, int]:
    try:
        if path.is_file():
            st = path.stat()
            return st.st_mtime, st.st_size
        if path.is_dir():
            latest = 0.0
            count = 0
            for item in path.rglob("*"):
                if item.is_file():
                    count += 1
                    latest = max(latest, item.stat().st_mtime)
            return latest, count
    except OSError:
        pass
    return 0.0, 0


def _format_duration(seconds: int) -> str:
    minutes, sec = divmod(max(0, seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}小时{minutes}分"
    return f"{minutes}分{sec:02d}秒"


def format_subprocess_error(output: str) -> str:
    text = (output or "").strip()
    if "Traceback" in text:
        tail = text[text.rfind("Traceback") :]
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        return "\n".join(lines[-10:])
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and not line.startswith("INFO:"):
            return line[:800]
    return text[:800] if text else "训练子进程失败"


def watch_paths_for_step(
    step: int,
    opt_dir: Path,
    engine_root: Path,
    exp_name: str,
    sovits_version: str = "v2",
) -> list[Path]:
    del exp_name
    paths: list[Path] = [opt_dir]
    train_log = opt_dir / "train.log"
    if train_log.exists() or step >= 4:
        paths.append(train_log)
    if step >= 4:
        paths.extend(
            [
                opt_dir / f"logs_s2_{sovits_version}",
                engine_root / "SoVITS_weights_v2",
            ]
        )
    if step >= 5:
        paths.extend(
            [
                opt_dir / f"logs_s1_{sovits_version}",
                engine_root / "GPT_weights_v2",
            ]
        )
    return paths


def run_with_watchdog(
    cmd: list[str],
    cwd: Path,
    env: dict,
    *,
    step_label: str,
    watch_paths: list[Path],
    progress: ProgressCallback | None = None,
    stall_seconds: int = STALL_SECONDS_TRAIN,
    log_path: Path | None = None,
) -> str:
    """运行 subprocess，轮询日志/权重目录；超时无进展则终止并报错。"""
    if progress:
        progress(step_label)

    snapshots = {path: _path_stat(path) for path in watch_paths}
    start = time.time()
    last_activity = start

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    else:
        log_handle = subprocess.DEVNULL  # type: ignore[assignment]

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    try:
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            now = time.time()
            elapsed = int(now - start)

            changed = False
            for path in watch_paths:
                current = _path_stat(path)
                if current != snapshots.get(path):
                    snapshots[path] = current
                    changed = True
            if changed:
                last_activity = now

            idle = int(now - last_activity)
            if progress:
                if changed:
                    progress(
                        f"{step_label}（已运行 {_format_duration(elapsed)} · 训练进行中）"
                    )
                elif idle >= 60:
                    progress(
                        f"{step_label}（已运行 {_format_duration(elapsed)} · "
                        f"⚠ {idle // 60} 分 {idle % 60} 秒无新进展）"
                    )

            if idle >= stall_seconds:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                raise TrainingStalledError(
                    f"训练疑似卡死：{step_label}\n"
                    f"已连续 {_format_duration(idle)} 没有任何日志/权重文件更新。\n"
                    "建议：1) 任务管理器结束残留 python 进程；2) 关闭占内存的程序或增大虚拟内存；"
                    "3) 改用「保存为零样本声线」。"
                )
    finally:
        if log_path and log_handle is not subprocess.DEVNULL:
            log_handle.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    returncode = proc.returncode if proc.returncode is not None else proc.wait()
    output = ""
    if log_path and log_path.is_file():
        output = log_path.read_text(encoding="utf-8", errors="replace")

    if returncode != 0:
        detail = format_subprocess_error(output)
        raise RuntimeError(detail or f"命令失败: {' '.join(cmd)}")
    return output
