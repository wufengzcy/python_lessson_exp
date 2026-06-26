"""主窗口：TTS 合成、播放、历史记录、管理后台。"""

from __future__ import annotations

import os
import threading
import tkinter as tk
import winsound
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import db
from config import OUTPUT_DIR, TTS_ENGINE_CHAT, TTS_ENGINE_SOVITS, TTS_MODEL_NAME
from ui.theme import COLORS, FONTS, apply_theme, center_window, make_card


class MainWindow(tk.Tk):
    """登录后的主界面（固定居中窗口，非全屏）。"""

    def __init__(self, current_user: dict):
        super().__init__()
        self.current_user = current_user
        self.is_synthesizing = False
        self.current_wav_path: str | None = None
        self.current_record_id: int | None = None
        self.voice_profile_map: dict[str, int] = {}

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.title("智声助手")
        self.minsize(880, 620)
        apply_theme(self)
        center_window(self, 960, 680)

        self._build_header()
        self._build_main_area()
        self._build_history_area()
        self._build_status_bar()
        self._refresh_history()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------

    def _build_header(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 14))
        header.pack(fill=tk.X)

        left = ttk.Frame(header, style="Header.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, text="智声助手", font=FONTS["title"], background=COLORS["header"]).pack(
            anchor=tk.W
        )
        ttk.Label(
            left,
            text="ChatTTS · GPT-SoVITS 双引擎",
            font=FONTS["subtitle"],
            foreground=COLORS["muted"],
            background=COLORS["header"],
        ).pack(anchor=tk.W, pady=(2, 0))

        right = ttk.Frame(header, style="Header.TFrame")
        right.pack(side=tk.RIGHT)
        role_label = "管理员" if self.current_user["role"] == "admin" else "用户"
        ttk.Label(
            right,
            text=f"{self.current_user['username']}  ·  {role_label}",
            background=COLORS["header"],
            foreground=COLORS["muted"],
        ).pack(side=tk.RIGHT, padx=(0, 10))
        if self.current_user["role"] == "admin":
            ttk.Button(right, text="管理", style="Secondary.TButton", command=self._open_admin).pack(
                side=tk.RIGHT, padx=4
            )
        ttk.Button(right, text="我的声线", style="Secondary.TButton", command=self._open_voice_clone).pack(
            side=tk.RIGHT, padx=4
        )
        ttk.Button(right, text="退出", style="Secondary.TButton", command=self._on_close).pack(
            side=tk.RIGHT, padx=4
        )

        sep = tk.Frame(self, height=1, bg=COLORS["border"])
        sep.pack(fill=tk.X)

    def _build_main_area(self):
        body = ttk.Frame(self, padding=(18, 14, 18, 8))
        body.pack(fill=tk.BOTH, expand=True)

        card = make_card(body, padding=16)
        card.pack(fill=tk.BOTH, expand=True)

        engine_row = ttk.Frame(card, style="Card.TFrame")
        engine_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(engine_row, text="合成引擎", style="Card.TLabel").pack(side=tk.LEFT)
        self.engine_var = tk.StringVar(value="ChatTTS 默认")
        self.combo_engine = ttk.Combobox(
            engine_row,
            textvariable=self.engine_var,
            values=["ChatTTS 默认", "GPT-SoVITS 克隆"],
            state="readonly",
            width=18,
        )
        self.combo_engine.pack(side=tk.LEFT, padx=(8, 16))
        self.combo_engine.bind("<<ComboboxSelected>>", self._on_engine_changed)

        ttk.Label(engine_row, text="声线", style="Card.TLabel").pack(side=tk.LEFT)
        self.voice_var = tk.StringVar(value="（请先创建声线）")
        self.combo_voice = ttk.Combobox(
            engine_row,
            textvariable=self.voice_var,
            state="disabled",
            width=24,
        )
        self.combo_voice.pack(side=tk.LEFT, padx=(8, 0))
        self._refresh_voice_profiles()

        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top, text="输入文本", style="Card.TLabel", font=FONTS["body_bold"]).pack(
            side=tk.LEFT
        )
        self.lbl_char_count = ttk.Label(top, text="0 字", style="CardMuted.TLabel")
        self.lbl_char_count.pack(side=tk.RIGHT)

        text_wrap = ttk.Frame(card, style="Card.TFrame")
        text_wrap.pack(fill=tk.BOTH, expand=True)

        self.text_input = tk.Text(
            text_wrap,
            height=8,
            wrap=tk.WORD,
            font=FONTS["body"],
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["primary"],
            padx=10,
            pady=10,
            insertbackground=COLORS["primary"],
        )
        scroll = ttk.Scrollbar(text_wrap, command=self.text_input.yview)
        self.text_input.config(yscrollcommand=scroll.set)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_input.insert(
            tk.END,
            "你好，欢迎使用智声助手。请输入要合成的文字，然后点击开始合成。",
        )
        self.text_input.bind("<KeyRelease>", self._update_char_count)
        self._update_char_count()

        actions = ttk.Frame(card, style="Card.TFrame")
        actions.pack(fill=tk.X, pady=(14, 0))

        ttk.Button(
            actions,
            text="开始合成",
            style="Primary.TButton",
            command=self._start_synthesize,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="播放", style="Secondary.TButton", command=self._play_audio).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="停止", style="Secondary.TButton", command=self._stop_audio).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="导出 WAV", style="Secondary.TButton", command=self._export_wav).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="清空", style="Secondary.TButton", command=self._clear_text).pack(
            side=tk.LEFT, padx=4
        )

        self.progress = ttk.Progressbar(card, mode="indeterminate", length=200)
        self.progress.pack(fill=tk.X, pady=(12, 0))

    def _on_engine_changed(self, _event=None):
        if self.engine_var.get().startswith("GPT"):
            self.combo_voice.config(state="readonly")
            self._refresh_voice_profiles()
        else:
            self.combo_voice.config(state="disabled")
            self.voice_var.set("")

    def _refresh_voice_profiles(self):
        self.voice_profile_map.clear()
        names: list[str] = []
        for p in db.list_voice_profiles_by_user(self.current_user["id"], ready_only=True):
            label = f"{p['name']} ({p['mode']})"
            names.append(label)
            self.voice_profile_map[label] = p["id"]
        self.combo_voice["values"] = names or ["（暂无可用声线）"]
        if self._get_selected_engine() == TTS_ENGINE_SOVITS:
            if names:
                current = self.voice_var.get()
                if current in self.voice_profile_map:
                    self.voice_var.set(current)
                else:
                    self.voice_var.set(names[0])
            else:
                self.voice_var.set("（暂无可用声线）")
        else:
            self.voice_var.set("")

    def _get_selected_engine(self) -> str:
        if self.engine_var.get().startswith("GPT"):
            return TTS_ENGINE_SOVITS
        return TTS_ENGINE_CHAT

    def _get_selected_voice_profile(self) -> dict | None:
        pid = self.voice_profile_map.get(self.voice_var.get())
        if pid is None:
            return None
        from path_utils import resolve_voice_profile

        profile = db.get_voice_profile(pid)
        return resolve_voice_profile(profile) if profile else None

    def _open_voice_clone(self):
        from ui.voice_clone_window import VoiceCloneWindow

        VoiceCloneWindow(self, self.current_user, on_changed=self._refresh_voice_profiles)

    def _build_history_area(self):
        frame = ttk.LabelFrame(self, text="  合成历史  ", style="Card.TLabelframe", padding=12)
        frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 8))

        search_bar = ttk.Frame(frame, style="Card.TFrame")
        search_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(search_bar, text="搜索", style="Card.TLabel").pack(side=tk.LEFT)
        self.entry_search = ttk.Entry(search_bar, width=28)
        self.entry_search.pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(search_bar, text="查找", style="Secondary.TButton", command=self._search_history).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(search_bar, text="刷新", style="Secondary.TButton", command=self._refresh_history).pack(
            side=tk.LEFT
        )

        cols = ("id", "created_at", "duration", "preview")
        col_names = {"id": "ID", "created_at": "时间", "duration": "时长", "preview": "文本摘要"}
        widths = {"id": 50, "created_at": 150, "duration": 70, "preview": 420}

        tree_wrap = ttk.Frame(frame, style="Card.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        self.tree_history = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=5)
        for c in cols:
            self.tree_history.heading(c, text=col_names[c])
            self.tree_history.column(c, width=widths[c], anchor=tk.W if c == "preview" else tk.CENTER)

        scroll_y = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree_history.yview)
        self.tree_history.configure(yscrollcommand=scroll_y.set)
        self.tree_history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_history.bind("<Double-1>", self._load_history_item)

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=COLORS["header"], height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="就绪 · 首次合成将加载模型，请稍候")
        tk.Label(
            bar,
            textvariable=self.status_var,
            bg=COLORS["header"],
            fg=COLORS["muted"],
            font=FONTS["small"],
            anchor=tk.W,
            padx=14,
        ).pack(fill=tk.X, ipady=6)

    # ---------- 事件 ----------

    def _update_char_count(self, _event=None):
        text = self.text_input.get("1.0", tk.END).strip()
        self.lbl_char_count.config(text=f"{len(text)} 字")

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self._update_char_count()

    def _start_synthesize(self):
        import tts_core

        if self.is_synthesizing:
            return
        raw_text = self.text_input.get("1.0", tk.END).strip()
        text = tts_core.normalize_text(raw_text)
        if not text:
            messagebox.showwarning("提示", "请输入要合成的文本", parent=self)
            return

        self.is_synthesizing = True
        self.progress.start(12)
        engine = self._get_selected_engine()
        profile = self._get_selected_voice_profile() if engine == TTS_ENGINE_SOVITS else None
        if engine == TTS_ENGINE_SOVITS:
            if profile is None:
                self.is_synthesizing = False
                messagebox.showwarning("提示", "请先在「我的声线」中创建可用声线", parent=self)
                return
            self._set_status("GPT-SoVITS 合成中，首次加载较慢…")
            model_name = f"GPT-SoVITS:{profile['name']}"
        else:
            self._set_status("正在合成，首次运行需加载模型…")
            model_name = TTS_MODEL_NAME

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            OUTPUT_DIR, f"tts_{self.current_user['id']}_{ts}.wav"
        )
        record_id = db.create_transcription(
            self.current_user["id"],
            audio_path=output_path,
            model_name=model_name,
            language="zh",
            status="processing",
        )

        def worker():
            import sovits_core
            import tts_core

            try:
                if engine == TTS_ENGINE_SOVITS:
                    assert profile is not None
                    wav, sr = sovits_core.synthesize(
                        text,
                        profile["ref_audio_path"],
                        profile["prompt_text"],
                        gpt_weights=profile.get("gpt_weights_path"),
                        sovits_weights=profile.get("sovits_weights_path"),
                    )
                    import soundfile as sf

                    sf.write(output_path, wav, sr)
                    duration = len(wav) / sr
                else:
                    wav, duration = tts_core.synthesize(text)
                    tts_core.save_wav(wav, output_path)
                db.update_transcription_result(record_id, text, duration, "done")
                self.after(
                    0,
                    lambda: self._on_synthesize_done(text, output_path, record_id, duration),
                )
            except Exception as e:
                err_msg = str(e)
                db.update_transcription_result(record_id, text, 0, "failed", err_msg)
                self.after(0, lambda msg=err_msg: self._on_synthesize_error(msg))
            finally:
                self.after(0, self._finish_synthesize_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_synthesize_ui(self):
        self.is_synthesizing = False
        self.progress.stop()

    def _on_synthesize_done(self, text: str, wav_path: str, record_id: int, duration: float):
        self.current_wav_path = wav_path
        self.current_record_id = record_id
        self._set_status(f"合成完成 · 时长 {duration:.1f} 秒 · 已保存")
        self._refresh_history()
        db.create_operation_log(self.current_user["id"], "tts_synthesize", f"合成 {len(text)} 字")

    def _on_synthesize_error(self, err: str):
        messagebox.showerror("合成失败", err, parent=self)
        self._set_status(f"失败: {err}")
        self._refresh_history()

    def _play_audio(self):
        if not self.current_wav_path or not os.path.isfile(self.current_wav_path):
            messagebox.showwarning("提示", "请先完成一次合成", parent=self)
            return
        try:
            winsound.PlaySound(self.current_wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._set_status(f"正在播放: {os.path.basename(self.current_wav_path)}")
        except Exception as e:
            messagebox.showerror("播放失败", str(e), parent=self)

    def _stop_audio(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self._set_status("已停止播放")

    def _export_wav(self):
        if not self.current_wav_path or not os.path.isfile(self.current_wav_path):
            messagebox.showwarning("提示", "没有可导出的音频", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="导出 WAV",
            defaultextension=".wav",
            initialfile=os.path.basename(self.current_wav_path),
            filetypes=[("WAV 音频", "*.wav"), ("全部", "*.*")],
        )
        if not path:
            return
        import shutil

        shutil.copy2(self.current_wav_path, path)
        messagebox.showinfo("成功", f"已导出到:\n{path}", parent=self)

    def _refresh_history(self):
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)
        rows = db.list_transcriptions_by_user(self.current_user["id"])
        for r in rows:
            preview = (r.get("text") or "")[:48]
            dur = r.get("duration_sec") or 0
            dur_text = f"{dur:.1f}s" if dur else "-"
            self.tree_history.insert(
                "",
                tk.END,
                values=(r["id"], r["created_at"], dur_text, preview),
            )

    def _search_history(self):
        kw = self.entry_search.get().strip()
        if not kw:
            self._refresh_history()
            return
        rows = db.search_transcriptions(self.current_user["id"], kw)
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)
        for r in rows:
            preview = (r.get("text") or "")[:48]
            dur = r.get("duration_sec") or 0
            dur_text = f"{dur:.1f}s" if dur else "-"
            self.tree_history.insert(
                "",
                tk.END,
                values=(r["id"], r["created_at"], dur_text, preview),
            )

    def _load_history_item(self, _event):
        sel = self.tree_history.selection()
        if not sel:
            return
        tid = self.tree_history.item(sel[0])["values"][0]
        row = db.get_transcription(tid)
        if not row:
            return
        if row.get("text"):
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, row["text"])
            self._update_char_count()
        audio_path = row.get("audio_path") or ""
        if audio_path and os.path.isfile(audio_path):
            self.current_wav_path = audio_path
            self.current_record_id = row["id"]
            self._set_status(f"已加载历史记录 #{row['id']}")

    def _open_admin(self):
        AdminWindow(self, self.current_user)

    def _on_close(self):
        winsound.PlaySound(None, winsound.SND_PURGE)
        self.destroy()


class AdminWindow(tk.Toplevel):
    """管理员后台。"""

    def __init__(self, master, current_user: dict):
        super().__init__(master)
        if current_user["role"] != "admin":
            messagebox.showerror("权限不足", "需要管理员账号", parent=master)
            self.destroy()
            return

        self.title("智声助手 · 管理后台")
        apply_theme(self)
        center_window(self, 820, 520)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.tab_users = ttk.Frame(notebook, padding=8)
        self.tab_trans = ttk.Frame(notebook, padding=8)
        self.tab_logs = ttk.Frame(notebook, padding=8)
        notebook.add(self.tab_users, text="用户")
        notebook.add(self.tab_trans, text="合成记录")
        notebook.add(self.tab_logs, text="操作日志")

        self._build_users_tab()
        self._build_trans_tab()
        self._build_logs_tab()

    def _build_users_tab(self):
        cols = ("id", "username", "role", "created_at")
        tree = ttk.Treeview(self.tab_users, columns=cols, show="headings")
        for c, name in zip(cols, ("ID", "用户名", "角色", "注册时间")):
            tree.heading(c, text=name)
            tree.column(c, width=120 if c != "username" else 180)
        tree.pack(fill=tk.BOTH, expand=True)
        for u in db.list_users():
            tree.insert("", tk.END, values=(u["id"], u["username"], u["role"], u["created_at"]))

    def _build_trans_tab(self):
        cols = ("id", "username", "status", "created_at", "preview")
        tree = ttk.Treeview(self.tab_trans, columns=cols, show="headings")
        names = {"id": "ID", "username": "用户", "status": "状态", "created_at": "时间", "preview": "摘要"}
        for c in cols:
            tree.heading(c, text=names[c])
        tree.pack(fill=tk.BOTH, expand=True)
        for r in db.list_all_transcriptions():
            preview = (r.get("text") or "")[:36]
            tree.insert(
                "",
                tk.END,
                values=(r["id"], r["username"], r["status"], r["created_at"], preview),
            )

    def _build_logs_tab(self):
        cols = ("id", "username", "action", "detail", "created_at")
        tree = ttk.Treeview(self.tab_logs, columns=cols, show="headings")
        names = {"id": "ID", "username": "用户", "action": "动作", "detail": "详情", "created_at": "时间"}
        for c in cols:
            tree.heading(c, text=names[c])
        tree.pack(fill=tk.BOTH, expand=True)
        for r in db.list_operation_logs():
            tree.insert(
                "",
                tk.END,
                values=(
                    r["id"],
                    r.get("username") or "-",
                    r["action"],
                    r.get("detail") or "",
                    r["created_at"],
                ),
            )
