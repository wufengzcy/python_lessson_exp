"""GPT-SoVITS 声线管理：导入参考音频、零样本克隆、微调训练。"""

from __future__ import annotations

import os
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import db
import sovits_core
import sovits_train
from config import VOICES_DIR
from ui.theme import COLORS, FONTS, apply_theme, center_window, make_card


class VoiceCloneWindow(tk.Toplevel):
    """用户声线创建与管理。"""

    def __init__(self, master, current_user: dict, on_changed=None):
        super().__init__(master)
        self.current_user = current_user
        self.on_changed = on_changed
        self.ref_audio_path: str | None = None
        self.is_training = False

        os.makedirs(VOICES_DIR, exist_ok=True)

        self.title("智声助手 · 我的声线")
        apply_theme(self)
        center_window(self, 720, 560)

        self._build_form()
        self._build_list()
        self._refresh_list()

    def _build_form(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        card = make_card(outer, padding=16)
        card.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(card, text="创建声线", style="Card.TLabel", font=FONTS["body_bold"]).pack(
            anchor=tk.W
        )
        ttk.Label(
            card,
            text="导入 3～60 秒干净人声，填写录音里说的文字。可先零样本克隆，也可微调训练。",
            style="CardMuted.TLabel",
        ).pack(anchor=tk.W, pady=(4, 12))

        row1 = ttk.Frame(card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="声线名称", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        self.entry_name = ttk.Entry(row1)
        self.entry_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        row2 = ttk.Frame(card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="参考音频", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        self.lbl_audio = ttk.Label(row2, text="未选择", style="CardMuted.TLabel")
        self.lbl_audio.pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(row2, text="导入 WAV", style="Secondary.TButton", command=self._import_wav).pack(
            side=tk.LEFT
        )

        row3 = ttk.Frame(card, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="参考文本", style="Card.TLabel", width=10).pack(side=tk.LEFT, anchor=tk.N)
        self.text_prompt = tk.Text(
            row3,
            height=3,
            wrap=tk.WORD,
            font=FONTS["body"],
            bg=COLORS["input_bg"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.text_prompt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.text_prompt.insert(tk.END, "请填写参考音频里实际说出的内容，尽量与录音一致。")

        status = sovits_core.engine_status_message()
        self.lbl_engine = ttk.Label(card, text=f"引擎状态：{status}", style="CardMuted.TLabel")
        self.lbl_engine.pack(anchor=tk.W, pady=(10, 0))

        btns = ttk.Frame(card, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(
            btns,
            text="保存为零样本声线",
            style="Primary.TButton",
            command=self._save_zero_shot,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btns,
            text="开始微调训练",
            style="Secondary.TButton",
            command=self._start_train,
        ).pack(side=tk.LEFT)

        self.lbl_progress = ttk.Label(card, text="", style="CardMuted.TLabel")
        self.lbl_progress.pack(anchor=tk.W, pady=(8, 0))

    def _build_list(self):
        frame = ttk.LabelFrame(self, text="  我的声线  ", style="Card.TLabelframe", padding=12)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        cols = ("id", "name", "mode", "status", "created_at")
        names = {"id": "ID", "name": "名称", "mode": "模式", "status": "状态", "created_at": "创建时间"}
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=names[c])
            self.tree.column(c, width=90 if c != "name" else 140)
        self.tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="删除选中", style="Secondary.TButton", command=self._delete_selected).pack(
            anchor=tk.E, pady=(8, 0)
        )

    def _import_wav(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="选择参考音频",
            filetypes=[("WAV", "*.wav"), ("全部", "*.*")],
        )
        if path:
            self.ref_audio_path = path
            self.lbl_audio.config(text=os.path.basename(path))

    def _validate(self) -> tuple[str, str, str] | None:
        name = self.entry_name.get().strip()
        prompt = self.text_prompt.get("1.0", tk.END).strip()
        if not name:
            messagebox.showwarning("提示", "请填写声线名称", parent=self)
            return None
        if not self.ref_audio_path:
            messagebox.showwarning("提示", "请先导入参考音频", parent=self)
            return None
        if not prompt:
            messagebox.showwarning("提示", "请填写参考文本", parent=self)
            return None
        return name, prompt, self.ref_audio_path

    def _copy_ref_audio(self, user_id: int, name: str, src: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        dst_dir = Path(VOICES_DIR) / str(user_id) / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "reference.wav"
        shutil.copy2(src, dst)
        return str(dst.resolve())

    def _save_zero_shot(self):
        data = self._validate()
        if not data:
            return
        if not sovits_core.is_pretrained_ready():
            messagebox.showerror("不可用", sovits_core.engine_status_message(), parent=self)
            return
        name, prompt, src = data
        ref_path = self._copy_ref_audio(self.current_user["id"], name, src)
        db.create_voice_profile(
            self.current_user["id"],
            name,
            ref_path,
            prompt,
            mode="zero_shot",
            status="ready",
        )
        db.create_operation_log(self.current_user["id"], "voice_zero_shot", f"创建声线 {name}")
        messagebox.showinfo("成功", f"声线「{name}」已保存，可在主界面选择 GPT-SoVITS 合成。", parent=self)
        self._refresh_list()
        if self.on_changed:
            self.on_changed()

    def _start_train(self):
        if self.is_training:
            return
        data = self._validate()
        if not data:
            return
        if not sovits_core.is_pretrained_ready():
            messagebox.showerror("不可用", sovits_core.engine_status_message(), parent=self)
            return
        if not messagebox.askyesno(
            "确认训练",
            "微调将调用 GPT-SoVITS 本地训练脚本，需 NVIDIA 显卡，可能耗时较久。是否继续？",
            parent=self,
        ):
            return

        name, prompt, src = data
        ref_path = self._copy_ref_audio(self.current_user["id"], name, src)
        profile_id = db.create_voice_profile(
            self.current_user["id"],
            name,
            ref_path,
            prompt,
            mode="finetuned",
            status="training",
        )
        self.is_training = True
        self.lbl_progress.config(text="训练已开始…")

        def worker():
            try:
                wav_dir, list_path, exp_name = sovits_train.prepare_dataset(
                    self.current_user["id"], name, ref_path, prompt
                )

                def progress(msg: str):
                    self.after(0, lambda: self.lbl_progress.config(text=msg))

                sovits_w, gpt_w = sovits_train.run_finetune(
                    list_path, wav_dir, exp_name, progress=progress
                )
                db.update_voice_profile_status(
                    profile_id,
                    "ready",
                    gpt_weights_path=gpt_w,
                    sovits_weights_path=sovits_w,
                )
                self.after(0, lambda: self._on_train_done(name, exp_name))
            except Exception as e:
                db.update_voice_profile_status(profile_id, "failed", str(e))
                self.after(0, lambda: self._on_train_error(str(e)))
            finally:
                self.after(0, self._finish_train_ui)

        threading.Thread(target=worker, daemon=True).start()
        self._refresh_list()

    def _on_train_done(self, name: str, exp_name: str):
        self.lbl_progress.config(text=f"训练完成：{name}")
        messagebox.showinfo("完成", f"声线「{name}」微调完成，可使用 GPT-SoVITS 合成。", parent=self)
        db.create_operation_log(self.current_user["id"], "voice_train", f"训练完成 {exp_name}")
        self._refresh_list()
        if self.on_changed:
            self.on_changed()

    def _on_train_error(self, err: str):
        self.lbl_progress.config(text=f"训练失败：{err}")
        messagebox.showerror("训练失败", err, parent=self)
        self._refresh_list()

    def _finish_train_ui(self):
        self.is_training = False

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        mode_map = {"zero_shot": "零样本", "finetuned": "微调"}
        for r in db.list_voice_profiles_by_user(self.current_user["id"]):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r["id"],
                    r["name"],
                    mode_map.get(r["mode"], r["mode"]),
                    r["status"],
                    r["created_at"],
                ),
            )

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        pid = self.tree.item(sel[0])["values"][0]
        if messagebox.askyesno("确认", "确定删除该声线？", parent=self):
            db.delete_voice_profile(pid, self.current_user["id"])
            self._refresh_list()
            if self.on_changed:
                self.on_changed()
