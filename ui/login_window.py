"""登录 / 注册窗口。"""

import tkinter as tk
from tkinter import messagebox, ttk

import db
from ui.theme import COLORS, FONTS, apply_theme, center_window, make_card


class LoginWindow(tk.Tk):
    """应用入口；登录成功后销毁并打开主窗口。"""

    def __init__(self, on_login_success):
        super().__init__()
        self.on_login_success = on_login_success

        self.title("智声助手 · 登录")
        self.resizable(False, False)
        apply_theme(self)
        center_window(self, 440, 520)

        self._build_ui()
        self.entry_username.focus_set()
        self.bind("<Return>", lambda _e: self._on_login())

    def _build_ui(self):
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="智声助手", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="AI 文本转语音 · 课设演示版",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 20))

        card = make_card(outer, padding=22)
        card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(card, text="欢迎回来", style="Card.TLabel", font=FONTS["body_bold"]).pack(
            anchor=tk.W
        )
        ttk.Label(
            card,
            text="登录后即可使用 ChatTTS 合成与管理历史记录",
            style="CardMuted.TLabel",
        ).pack(anchor=tk.W, pady=(4, 18))

        form = ttk.Frame(card, style="Card.TFrame")
        form.pack(fill=tk.X)

        ttk.Label(form, text="用户名", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.entry_username = ttk.Entry(form, width=28)
        self.entry_username.grid(row=1, column=0, sticky=tk.EW, pady=(0, 14))

        ttk.Label(form, text="密码", style="Card.TLabel").grid(row=2, column=0, sticky=tk.W, pady=(0, 4))
        self.entry_password = ttk.Entry(form, width=28, show="•")
        self.entry_password.grid(row=3, column=0, sticky=tk.EW, pady=(0, 18))
        form.columnconfigure(0, weight=1)

        btn_row = ttk.Frame(card, style="Card.TFrame")
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="登录", style="Primary.TButton", command=self._on_login).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6)
        )
        ttk.Button(btn_row, text="注册", style="Secondary.TButton", command=self._on_register).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        hint = ttk.Frame(outer, style="Card.TFrame", padding=(4, 16, 4, 0))
        hint.pack(fill=tk.X)
        ttk.Label(
            hint,
            text="默认管理员  admin / admin123",
            style="CardMuted.TLabel",
            foreground=COLORS["muted"],
        ).pack(anchor=tk.CENTER)

    def _validate_input(self) -> tuple[str, str] | None:
        username = self.entry_username.get().strip()
        password = self.entry_password.get().strip()
        if not username or not password:
            messagebox.showwarning("提示", "用户名和密码不能为空", parent=self)
            return None
        return username, password

    def _on_login(self):
        data = self._validate_input()
        if data is None:
            return
        username, password = data
        try:
            user = db.verify_user(username, password)
            if user is None:
                messagebox.showerror("登录失败", "用户名或密码错误", parent=self)
                return
            db.create_operation_log(user["id"], "login", f"用户 {username} 登录")
            self.destroy()
            self.on_login_success(user)
        except Exception as e:
            messagebox.showerror("错误", f"登录异常: {e}", parent=self)

    def _on_register(self):
        data = self._validate_input()
        if data is None:
            return
        username, password = data
        try:
            uid = db.create_user(username, password, role="user")
            if uid is None:
                messagebox.showerror("注册失败", "用户名已存在", parent=self)
                return
            db.create_operation_log(uid, "register", f"新用户 {username} 注册")
            messagebox.showinfo("成功", "注册成功，请登录", parent=self)
        except Exception as e:
            messagebox.showerror("错误", f"注册异常: {e}", parent=self)
