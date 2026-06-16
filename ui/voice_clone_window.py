"""GPT-SoVITS 声线管理：导入参考音频、零样本克隆、微调训练。"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import audio_utils
import db
from config import VOICES_DIR
from train_batch import load_samples_from_folder
from ui.theme import COLORS, FONTS, apply_theme, center_window, make_card


class VoiceCloneWindow(tk.Toplevel):
    """用户声线创建与管理。"""

    def __init__(self, master, current_user: dict, on_changed=None):
        super().__init__(master)
        self.current_user = current_user
        self.on_changed = on_changed
        self.ref_audio_path: str | None = None
        self.train_samples: list[tuple[str, str]] = []
        self.is_training = False

        os.makedirs(VOICES_DIR, exist_ok=True)

        self.title("智声助手 · 我的声线")
        apply_theme(self)
        center_window(self, 720, 640)

        self._build_form()
        self._build_list()
        self._refresh_list()

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
            text="导入 3～30 秒干净人声（支持 WAV / M4A / MP3）。微调建议添加 3 条以上样本，每条文本与录音一致。",
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
        ttk.Button(row2, text="导入音频", style="Secondary.TButton", command=self._import_audio).pack(
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

        row4 = ttk.Frame(card, style="Card.TFrame")
        row4.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row4, text="训练样本", style="Card.TLabel", width=10).pack(side=tk.LEFT, anchor=tk.N)
        sample_wrap = ttk.Frame(row4, style="Card.TFrame")
        sample_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.list_train_samples = tk.Listbox(
            sample_wrap,
            height=3,
            font=FONTS["small"],
            bg=COLORS["input_bg"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.list_train_samples.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sample_btns = ttk.Frame(sample_wrap, style="Card.TFrame")
        sample_btns.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            sample_btns,
            text="导入文件夹",
            style="Secondary.TButton",
            command=self._import_train_folder,
        ).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            sample_btns,
            text="加入列表",
            style="Secondary.TButton",
            command=self._add_train_sample,
        ).pack(fill=tk.X)
        ttk.Button(
            sample_btns,
            text="清空",
            style="Secondary.TButton",
            command=self._clear_train_samples,
        ).pack(fill=tk.X, pady=(6, 0))
        self.lbl_sample_count = ttk.Label(
            sample_wrap, text="0 条（微调建议 ≥3）", style="CardMuted.TLabel"
        )
        self.lbl_sample_count.pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_engine = ttk.Label(
            card, text="引擎状态：检测中…", style="CardMuted.TLabel"
        )
        self.lbl_engine.pack(anchor=tk.W, pady=(10, 0))
        self.after(100, self._refresh_engine_status)

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
        ttk.Label(
            card,
            text="训练期间每 15 秒刷新进度；若 8～10 分钟无新进展将自动判定卡死并停止。",
            style="CardMuted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

    def _refresh_engine_status(self):
        import sovits_core

        self.lbl_engine.config(text=f"引擎状态：{sovits_core.engine_status_message()}")

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

    def _import_audio(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="选择参考音频",
            filetypes=audio_utils.IMPORT_FILETYPES,
        )
        if path:
            ext = Path(path).suffix.lower()
            if ext and ext not in audio_utils.SUPPORTED_IMPORT_EXTENSIONS:
                messagebox.showwarning(
                    "格式不支持",
                    f"暂不支持 {ext}，请选择 WAV、M4A、MP3 等常见格式。",
                    parent=self,
                )
                return
            self.ref_audio_path = path
            self.lbl_audio.config(text=os.path.basename(path))

    def _validate(self) -> tuple[str, str, str] | None:
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("提示", "请填写声线名称", parent=self)
            return None
        if self.train_samples:
            return name, self.train_samples[0][1], self.train_samples[0][0]
        prompt = self.text_prompt.get("1.0", tk.END).strip()
        if not self.ref_audio_path:
            messagebox.showwarning("提示", "请先导入参考音频或训练文件夹", parent=self)
            return None
        if not prompt:
            messagebox.showwarning("提示", "请填写参考文本", parent=self)
            return None
        return name, prompt, self.ref_audio_path

    def _add_train_sample(self):
        data = self._validate()
        if not data:
            return
        _name, prompt, src = data
        self.train_samples.append((src, prompt))
        self._refresh_train_sample_list()
        messagebox.showinfo(
            "已添加",
            f"已加入第 {len(self.train_samples)} 条训练样本。\n"
            "可继续导入其他音频并填写对应文本后再点「加入列表」。",
            parent=self,
        )

    def _import_train_folder(self):
        folder = filedialog.askdirectory(
            parent=self,
            title="选择训练样本文件夹（含 manifest.tsv 或成对 wav+txt）",
            initialdir=str(Path(VOICES_DIR).parent / "train_batches"),
        )
        if not folder:
            return
        try:
            samples = load_samples_from_folder(folder)
        except (OSError, ValueError) as e:
            messagebox.showerror("导入失败", str(e), parent=self)
            return
        self.train_samples = samples
        self.ref_audio_path = samples[0][0]
        self.lbl_audio.config(text=os.path.basename(samples[0][0]))
        self.text_prompt.delete("1.0", tk.END)
        self.text_prompt.insert(tk.END, samples[0][1])
        self._refresh_train_sample_list()
        messagebox.showinfo(
            "导入成功",
            f"已从文件夹加载 {len(samples)} 条样本。\n"
            "请填写声线名称，然后点击「开始微调训练」。",
            parent=self,
        )

    def _clear_train_samples(self):
        self.train_samples.clear()
        self._refresh_train_sample_list()

    def _refresh_train_sample_list(self):
        self.list_train_samples.delete(0, tk.END)
        for i, (path, text) in enumerate(self.train_samples, start=1):
            preview = text[:24] + ("…" if len(text) > 24 else "")
            self.list_train_samples.insert(tk.END, f"{i}. {os.path.basename(path)} · {preview}")
        count = len(self.train_samples)
        hint = f"{count} 条"
        if count < 3:
            hint += "（微调建议 ≥3）"
        self.lbl_sample_count.config(text=hint)

    def _collect_train_samples(self, prompt: str, src: str) -> list[tuple[str, str]]:
        if self.train_samples:
            return list(self.train_samples)
        return [(src, prompt)]

    def _copy_ref_audio(self, user_id: int, name: str, src: str) -> str:
        safe = "".join(
            c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in name
        ).strip("_") or "voice"
        dst_dir = Path(VOICES_DIR) / str(user_id) / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "reference.wav"
        try:
            audio_utils.convert_to_wav(src, dst)
        except Exception as e:
            raise RuntimeError(f"参考音频转换失败: {e}") from e
        return str(dst.resolve())

    def _save_zero_shot(self):
        import sovits_core

        data = self._validate()
        if not data:
            return
        if not sovits_core.is_pretrained_ready():
            messagebox.showerror("不可用", sovits_core.engine_status_message(), parent=self)
            return
        name, prompt, src = data
        try:
            ref_path = self._copy_ref_audio(self.current_user["id"], name, src)
        except RuntimeError as e:
            messagebox.showerror("音频导入失败", str(e), parent=self)
            return
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
        import sovits_core

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
            "微调将调用 GPT-SoVITS 本地训练，需 NVIDIA 显卡。\n"
            "建议准备 3～10 条样本（每条 3～30 秒、文本与录音一致），"
            "样本越多效果越稳。\n"
            "训练前请关闭其他占内存的程序。是否继续？",
            parent=self,
        ):
            return

        name, prompt, src = data
        samples = self._collect_train_samples(prompt, src)
        profile_id = db.create_voice_profile(
            self.current_user["id"],
            name,
            samples[0][0],
            samples[0][1],
            mode="finetuned",
            status="training",
        )
        self.is_training = True
        self.lbl_progress.config(text="训练已开始…", foreground=COLORS["muted"])

        def progress(msg: str):
            def update(m=msg):
                self.lbl_progress.config(text=m)
                if "⚠" in m or "卡死" in m:
                    self.lbl_progress.config(foreground=COLORS["danger"])
                else:
                    self.lbl_progress.config(foreground=COLORS["muted"])

            self.after(0, update)

        def worker():
            import sovits_train

            try:
                wav_dir, list_path, exp_name, ref_prompt = sovits_train.prepare_dataset(
                    self.current_user["id"], name, samples
                )
                ref_path = str((Path(wav_dir).parent / "reference.wav").resolve())
                db.update_voice_profile_status(
                    profile_id,
                    "training",
                    ref_audio_path=ref_path,
                    prompt_text=ref_prompt,
                )

                sovits_w, gpt_w = sovits_train.run_finetune(
                    list_path, wav_dir, exp_name, progress=progress
                )
                from path_utils import to_project_relative

                db.update_voice_profile_status(
                    profile_id,
                    "ready",
                    gpt_weights_path=to_project_relative(gpt_w) if gpt_w else None,
                    sovits_weights_path=to_project_relative(sovits_w) if sovits_w else None,
                    ref_audio_path=to_project_relative(ref_path),
                    prompt_text=ref_prompt,
                )
                self.after(0, lambda n=name, ex=exp_name: self._on_train_done(n, ex))
            except Exception as e:
                err_msg = str(e)
                db.update_voice_profile_status(profile_id, "failed", err_msg)
                self.after(0, lambda msg=err_msg: self._on_train_error(msg))
            finally:
                self.after(0, self._finish_train_ui)

        threading.Thread(target=worker, daemon=True).start()
        self._refresh_list()

    def _on_train_done(self, name: str, exp_name: str):
        self.lbl_progress.config(text=f"训练完成：{name}")
        self.train_samples.clear()
        self._refresh_train_sample_list()
        messagebox.showinfo("完成", f"声线「{name}」微调完成，可使用 GPT-SoVITS 合成。", parent=self)
        db.create_operation_log(self.current_user["id"], "voice_train", f"训练完成 {exp_name}")
        self._refresh_list()
        if self.on_changed:
            self.on_changed()

    def _on_train_error(self, err: str):
        from train_watchdog import format_subprocess_error

        short = format_subprocess_error(err) if len(err) > 200 else err
        self.lbl_progress.config(
            text=f"训练失败：{short[:120]}", foreground=COLORS["danger"]
        )
        title = "训练卡死" if "卡死" in err else "训练失败"
        messagebox.showerror(title, short, parent=self)
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
