"""Tkinter 主题：配色与 ttk 样式。"""

import tkinter as tk
from tkinter import ttk

COLORS = {
    "bg": "#eef1f8",
    "card": "#ffffff",
    "border": "#d8deea",
    "primary": "#4f6ef7",
    "primary_hover": "#3d58d6",
    "primary_soft": "#e8edff",
    "text": "#1e293b",
    "muted": "#64748b",
    "success": "#10b981",
    "danger": "#ef4444",
    "header": "#f8faff",
    "input_bg": "#fbfcff",
}

FONTS = {
    "title": ("Microsoft YaHei UI", 18, "bold"),
    "subtitle": ("Microsoft YaHei UI", 10),
    "body": ("Microsoft YaHei UI", 10),
    "body_bold": ("Microsoft YaHei UI", 10, "bold"),
    "small": ("Microsoft YaHei UI", 9),
    "mono": ("Consolas", 9),
}


def center_window(window: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
    window.update_idletasks()
    sw = window.winfo_screenwidth()
    sh = window.winfo_screenheight()
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


def apply_theme(root: tk.Tk | tk.Toplevel) -> ttk.Style:
    root.configure(bg=COLORS["bg"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["body"])
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Card.TFrame", background=COLORS["card"])
    style.configure("Header.TFrame", background=COLORS["header"])
    style.configure(
        "Card.TLabelframe",
        background=COLORS["card"],
        bordercolor=COLORS["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=COLORS["card"],
        foreground=COLORS["text"],
        font=FONTS["body_bold"],
    )
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"])
    style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=FONTS["small"])
    style.configure("CardMuted.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=FONTS["small"])
    style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["title"])
    style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=FONTS["subtitle"])

    style.configure(
        "Primary.TButton",
        background=COLORS["primary"],
        foreground="#ffffff",
        borderwidth=0,
        focusthickness=0,
        padding=(14, 8),
        font=FONTS["body_bold"],
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLORS["primary_hover"]), ("disabled", "#a5b4fc")],
        foreground=[("disabled", "#eef2ff")],
    )
    style.configure(
        "Secondary.TButton",
        background=COLORS["card"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        borderwidth=1,
        padding=(12, 7),
    )
    style.map("Secondary.TButton", background=[("active", COLORS["primary_soft"])])
    style.configure("TEntry", fieldbackground=COLORS["input_bg"], bordercolor=COLORS["border"], padding=4)
    style.configure(
        "Treeview",
        background=COLORS["card"],
        fieldbackground=COLORS["card"],
        foreground=COLORS["text"],
        rowheight=28,
        bordercolor=COLORS["border"],
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["header"],
        foreground=COLORS["muted"],
        font=FONTS["body_bold"],
        relief="flat",
    )
    style.map("Treeview", background=[("selected", COLORS["primary_soft"])], foreground=[("selected", COLORS["text"])])
    return style


def make_card(parent: tk.Widget, padding: int = 16) -> ttk.Frame:
    frame = ttk.Frame(parent, style="Card.TFrame", padding=padding)
    frame.configure(relief="flat")
    return frame
