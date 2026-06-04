from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd

from transcribe_mic import DEFAULT_CONFIG, configure_local_storage, download_configured_model, load_config, main as transcribe_cli_main, save_config as write_config, today_output_path


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
CONFIG_PATH = APP_DIR / "config.json"
CONFIG_EXAMPLE_PATH = APP_DIR / "config.example.json"
BUNDLED_CONFIG_EXAMPLE_PATH = RESOURCE_DIR / "config.example.json"
ICON_PATH = RESOURCE_DIR / "assets" / "fennenote.ico"
ICON_PNG_PATH = RESOURCE_DIR / "assets" / "fennec-ear-icon.png"
BUDDY_IMAGE_PATH = RESOURCE_DIR / "assets" / "fennenote-listening-buddy.png"
BUDDY_STATE_IMAGE_PATHS = {
    "idle": RESOURCE_DIR / "assets" / "fennenote-state-idle.png",
    "listening": RESOURCE_DIR / "assets" / "fennenote-state-listening.png",
    "writing": RESOURCE_DIR / "assets" / "fennenote-state-writing.png",
}
IGNORED_PROCESS_OUTPUT_MARKERS = (
    "Xet Storage is enabled",
    "hf_xet",
    "huggingface_hub\\file_download.py",
    "huggingface_hub/file_download.py",
    "cache-system uses symlinks",
    "To support symlinks on Windows",
    "activate Developer Mode",
)
CUDA_DLL_DIRS = [
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"),
    Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA Canvas"),
]

LANGUAGE_LABELS = {
    "auto": "自动识别",
    "zh": "简体中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
}
LANGUAGE_CODES = {label: code for code, label in LANGUAGE_LABELS.items()}

THEME = {
    "app_bg": "#f7efe1",
    "panel": "#fffaf1",
    "panel_alt": "#fff4df",
    "ink": "#332821",
    "muted": "#786a5d",
    "line": "#e7d7bd",
    "line_strong": "#cfb88f",
    "sand": "#d9a441",
    "sand_dark": "#9a6820",
    "teal": "#159b91",
    "teal_dark": "#0f6f69",
    "teal_soft": "#dff4ef",
    "green": "#22b86f",
    "danger": "#d45b4c",
    "canvas": "#fffdf8",
    "quiet_bar": "#2f2a25",
}


def ensure_cuda_dll_path() -> None:
    paths = [str(path) for path in CUDA_DLL_DIRS if path.exists()]
    if not paths:
        return
    os.environ["PATH"] = ";".join(paths + [os.environ.get("PATH", "")])
    for path in paths:
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):
            pass


def ensure_bundled_config_template() -> None:
    if CONFIG_EXAMPLE_PATH.exists() or not BUNDLED_CONFIG_EXAMPLE_PATH.exists():
        return
    try:
        CONFIG_EXAMPLE_PATH.write_text(BUNDLED_CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass


def prepare_process_environment(config: dict) -> dict[str, str]:
    configure_local_storage(config)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def decode_process_output_line(raw_line: bytes | str) -> str:
    if isinstance(raw_line, str):
        return raw_line
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return raw_line.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_line.decode("utf-8", errors="replace")


class TranscriberGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FenneNote - 芬妮笔记")
        self.set_window_icon()
        self.geometry("980x680")
        self.minsize(860, 560)

        self.process: subprocess.Popen[bytes] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.level_queue: queue.Queue[tuple[float, float]] = queue.Queue()
        self.level_stream: sd.InputStream | None = None
        self.preview_noise_floor = 0.003
        self.level_history: deque[float] = deque(maxlen=160)
        self.raw_level_history: deque[float] = deque(maxlen=160)
        self.bar_history: deque[float] = deque(maxlen=96)
        self.preview_triggered = False
        self.last_trigger_time = 0.0
        self.display_level = 0.0
        self.display_raw_level = 0.0
        self.tail_position = 0
        self.is_paused = False
        self.buddy_state = "idle"
        self.writing_until = 0.0
        self.reply_server: ThreadingHTTPServer | None = None
        self.reply_server_thread: threading.Thread | None = None
        self.reply_bubble: tk.Toplevel | None = None

        ensure_bundled_config_template()
        self.config_data = load_config(CONFIG_PATH)
        self.devices = self.list_input_devices()
        self.vars = self.create_vars()

        self.setup_style()
        self.build_ui()
        self.start_level_monitor()
        self.configure_reply_bubble_server()
        self.refresh_tail(reset=True)
        self.after(150, self.drain_process_output)
        self.after(80, self.drain_level_preview)
        if bool(self.config_data.get("auto_start", False)):
            self.after(700, self.start_transcriber)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_window_icon(self) -> None:
        try:
            if ICON_PATH.exists():
                self.iconbitmap(default=str(ICON_PATH))
                return
            if ICON_PNG_PATH.exists():
                self._window_icon = tk.PhotoImage(file=str(ICON_PNG_PATH))
                self.iconphoto(True, self._window_icon)
        except tk.TclError:
            pass

    def create_vars(self) -> dict[str, tk.Variable]:
        device_index = self.config_data.get("mic_device")
        language_code = self.config_data.get("language_mode", self.config_data.get("language", "zh"))
        return {
            "model": tk.StringVar(value=str(self.config_data.get("model", "small"))),
            "device": tk.StringVar(value="cuda"),
            "compute_type": tk.StringVar(value=str(self.config_data.get("compute_type", "int8_float16"))),
            "language": tk.StringVar(value=LANGUAGE_LABELS.get(str(language_code), "简体中文")),
            "simplify_chinese": tk.BooleanVar(value=bool(self.config_data.get("simplify_chinese", True))),
            "input_gain": tk.DoubleVar(value=float(self.config_data.get("input_gain", DEFAULT_CONFIG["input_gain"]))),
            "record_threshold": tk.DoubleVar(value=float(self.config_data.get("record_threshold", self.config_data.get("rms_threshold", 0.01)))),
            "transcribe_threshold": tk.DoubleVar(value=float(self.config_data.get("transcribe_threshold", 0.015))),
            "adaptive_threshold": tk.BooleanVar(value=bool(self.config_data.get("adaptive_threshold", True))),
            "pre_roll_seconds": tk.DoubleVar(value=float(self.config_data.get("pre_roll_seconds", 1.5))),
            "transcribe_pause_seconds": tk.DoubleVar(value=float(self.config_data.get("transcribe_pause_seconds", 0.5))),
            "silence_seconds": tk.DoubleVar(value=float(self.config_data.get("silence_seconds", 1.2))),
            "max_phrase_seconds": tk.DoubleVar(value=float(self.config_data.get("max_phrase_seconds", 12.0))),
            "cache_retention_minutes": tk.DoubleVar(value=float(self.config_data.get("cache_retention_minutes", DEFAULT_CONFIG["cache_retention_minutes"]))),
            "mic_device": tk.StringVar(value=self.device_label_for(device_index)),
            "mic_level": tk.DoubleVar(value=0.0),
            "mic_level_text": tk.StringVar(value="原始 0.000 / 增益后 0.000 / 录音 0.010 / 转写 0.015"),
            "trigger_state": tk.StringVar(value="状态：监听中"),
            "status": tk.StringVar(value="就绪"),
            "model_cache_status": tk.StringVar(value="模型缓存：可提前安装"),
            "auto_start": tk.BooleanVar(value=bool(self.config_data.get("auto_start", False))),
            "reply_bubble_enabled": tk.BooleanVar(value=bool(self.config_data.get("reply_bubble_enabled", DEFAULT_CONFIG["reply_bubble_enabled"]))),
            "reply_bubble_port": tk.IntVar(value=int(self.config_data.get("reply_bubble_port", DEFAULT_CONFIG["reply_bubble_port"]))),
            "reply_bubble_seconds": tk.DoubleVar(value=float(self.config_data.get("reply_bubble_seconds", DEFAULT_CONFIG["reply_bubble_seconds"]))),
            "reply_bubble_token": tk.StringVar(value=str(self.config_data.get("reply_bubble_token", DEFAULT_CONFIG["reply_bubble_token"]))),
            "rabiroute_enabled": tk.BooleanVar(value=bool(self.config_data.get("rabiroute_enabled", DEFAULT_CONFIG["rabiroute_enabled"]))),
            "rabiroute_url": tk.StringVar(value=str(self.config_data.get("rabiroute_url", DEFAULT_CONFIG["rabiroute_url"]))),
            "rabiroute_token": tk.StringVar(value=str(self.config_data.get("rabiroute_token", DEFAULT_CONFIG["rabiroute_token"]))),
            "rabiroute_source": tk.StringVar(value=str(self.config_data.get("rabiroute_source", DEFAULT_CONFIG["rabiroute_source"]))),
        }

    def setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg=THEME["app_bg"])
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("TFrame", background=THEME["app_bg"])
        style.configure("Panel.TFrame", background=THEME["panel"])
        style.configure("Brand.TFrame", background=THEME["panel_alt"])
        style.configure("TLabel", background=THEME["panel"], foreground=THEME["ink"])
        style.configure("Root.TLabel", background=THEME["app_bg"], foreground=THEME["ink"])
        style.configure("BrandTitle.TLabel", background=THEME["panel_alt"], foreground=THEME["ink"], font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("BrandSub.TLabel", background=THEME["panel_alt"], foreground=THEME["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", background=THEME["panel"], foreground=THEME["ink"], font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Muted.TLabel", background=THEME["panel"], foreground=THEME["muted"])
        style.configure("Status.TLabel", background=THEME["panel"], foreground=THEME["teal_dark"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Chip.TLabel", background=THEME["teal_soft"], foreground=THEME["teal_dark"], padding=(10, 4), font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TLabelframe", background=THEME["panel"], bordercolor=THEME["line"], lightcolor=THEME["line"], darkcolor=THEME["line"])
        style.configure("TLabelframe.Label", background=THEME["panel"], foreground=THEME["sand_dark"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=(12, 6), background="#fff7e8", foreground=THEME["ink"], bordercolor=THEME["line_strong"], focusthickness=0)
        style.map("TButton", background=[("active", "#ffefd0"), ("pressed", "#ead1a2")])
        style.configure("Primary.TButton", padding=(18, 7), background=THEME["teal"], foreground="#ffffff", bordercolor=THEME["teal_dark"], font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Primary.TButton", background=[("active", "#19ada2"), ("pressed", THEME["teal_dark"]), ("disabled", "#b7c8c5")], foreground=[("disabled", "#f2f2f2")])
        style.configure("Danger.TButton", padding=(12, 6), background="#fff0ec", foreground=THEME["danger"], bordercolor="#efb4aa")
        style.map("Danger.TButton", background=[("active", "#ffe1da"), ("pressed", "#f5c2b8")])
        style.configure("TCheckbutton", background=THEME["panel"], foreground=THEME["ink"], indicatorcolor="#f5e5c8", indicatordiameter=14)
        style.map("TCheckbutton", indicatorcolor=[("selected", THEME["teal"]), ("active", THEME["sand"])])
        style.configure("TCombobox", fieldbackground="#fffdf8", background="#fff7e8", foreground=THEME["ink"], arrowcolor=THEME["teal_dark"], bordercolor=THEME["line"])
        style.configure("Horizontal.TScale", background=THEME["panel"], troughcolor="#ecdabc")
        style.configure("TNotebook", background=THEME["panel"], borderwidth=0, tabmargins=(10, 10, 10, 0))
        style.configure("TNotebook.Tab", background=THEME["panel_alt"], foreground=THEME["muted"], padding=(16, 7), bordercolor=THEME["line"], font=("Microsoft YaHei UI", 9, "bold"))
        style.map(
            "TNotebook.Tab",
            background=[("selected", THEME["teal_soft"]), ("active", "#f8ead0")],
            foreground=[("selected", THEME["teal_dark"]), ("active", THEME["ink"])],
        )
        style.configure("Fenne.Horizontal.TProgressbar", troughcolor="#eadcc4", background=THEME["teal"], bordercolor=THEME["line"], lightcolor=THEME["teal"], darkcolor=THEME["teal_dark"])

    def build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(16, 12), style="Brand.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        if ICON_PNG_PATH.exists():
            try:
                self.brand_icon_image = tk.PhotoImage(file=str(ICON_PNG_PATH)).subsample(8, 8)
                ttk.Label(toolbar, image=self.brand_icon_image, style="BrandSub.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 12))
            except tk.TclError:
                pass
        brand = ttk.Frame(toolbar, style="Brand.TFrame")
        brand.grid(row=0, column=1, rowspan=2, sticky="w", padx=(0, 24))
        ttk.Label(brand, text="FenneNote", style="BrandTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(brand, text="芬妮笔记 · 本地 GPU 实时听写", style="BrandSub.TLabel").grid(row=1, column=0, sticky="w")

        self.start_button = ttk.Button(toolbar, text="开始", command=self.start_transcriber, style="Primary.TButton")
        self.start_button.grid(row=0, column=2, rowspan=2, padx=(0, 8))
        self.pause_button = ttk.Button(toolbar, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=3, rowspan=2, padx=(0, 8))
        self.stop_button = ttk.Button(toolbar, text="停止", command=self.stop_transcriber, state="disabled", style="Danger.TButton")
        self.stop_button.grid(row=0, column=4, rowspan=2, padx=(0, 14))
        ttk.Button(toolbar, text="保存配置", command=self.save_config).grid(row=0, column=5, rowspan=2, padx=(0, 8))
        ttk.Button(toolbar, text="打开目录", command=self.open_output_folder).grid(row=0, column=6, rowspan=2, padx=(0, 16))
        ttk.Label(toolbar, text="CUDA", style="Chip.TLabel").grid(row=0, column=7, rowspan=2, sticky="e")

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=14, pady=14)

        settings_shell = ttk.Frame(main, padding=0, style="Panel.TFrame")
        settings_shell.rowconfigure(0, weight=1)
        settings_shell.columnconfigure(0, weight=1)
        settings_notebook = ttk.Notebook(settings_shell)
        settings_notebook.grid(row=0, column=0, sticky="nsew")
        main.add(settings_shell, weight=0)

        def add_settings_page(title: str) -> ttk.Frame:
            page_shell = ttk.Frame(settings_notebook, style="Panel.TFrame")
            page_shell.rowconfigure(0, weight=1)
            page_shell.columnconfigure(0, weight=1)
            canvas = tk.Canvas(page_shell, width=370, bg=THEME["panel"], highlightthickness=0)
            scrollbar = ttk.Scrollbar(page_shell, orient="vertical", command=canvas.yview)
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            page = ttk.Frame(canvas, padding=14, style="Panel.TFrame")
            page.columnconfigure(0, weight=1)
            page_window = canvas.create_window((0, 0), window=page, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            page.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda event, c=canvas, w=page_window: c.itemconfigure(w, width=event.width))
            settings_notebook.add(page_shell, text=title)
            return page

        input_settings = add_settings_page("输入")
        trigger_settings = add_settings_page("触发")
        app_settings = add_settings_page("应用")
        route_settings = add_settings_page("路由")

        input_group = ttk.LabelFrame(input_settings, text="输入与模型", padding=12)
        input_group.grid(row=0, column=0, sticky="ew")
        input_group.columnconfigure(1, weight=1)
        row = 0
        mic_combo, row = self.add_combo(input_group, row, "麦克风", "mic_device", [self.device_label_for(None)] + [label for _, label in self.devices])
        mic_combo.bind("<<ComboboxSelected>>", lambda _event: self.restart_level_monitor())
        _combo, row = self.add_combo(input_group, row, "模型", "model", ["small", "medium", "large-v3", "distil-large-v3"])
        model_actions = ttk.Frame(input_group, style="Panel.TFrame")
        model_actions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        model_actions.columnconfigure(1, weight=1)
        self.download_model_button = ttk.Button(model_actions, text="安装模型", command=self.install_selected_model)
        self.download_model_button.grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(model_actions, textvariable=self.vars["model_cache_status"], style="Muted.TLabel", wraplength=235).grid(row=0, column=1, sticky="ew")
        row += 1
        ttk.Label(input_group, text="运行设备").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Label(input_group, text="GPU / CUDA").grid(row=row, column=1, sticky="w", pady=5)
        row += 1
        _combo, row = self.add_combo(input_group, row, "计算精度", "compute_type", ["int8_float16", "float16", "int8", "int8_float32", "float32"])
        _combo, row = self.add_combo(input_group, row, "语言", "language", list(LANGUAGE_CODES.keys()))
        ttk.Label(input_group, text="文字转换").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Checkbutton(input_group, text="输出简体中文", variable=self.vars["simplify_chinese"]).grid(row=row, column=1, sticky="w", pady=5)

        trigger_group = ttk.LabelFrame(trigger_settings, text="触发与分段", padding=12)
        trigger_group.grid(row=0, column=0, sticky="ew")
        trigger_group.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(trigger_group, text="触发模式").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Checkbutton(trigger_group, text="动态识别环境底噪", variable=self.vars["adaptive_threshold"]).grid(row=row, column=1, sticky="w", pady=5)
        row += 1
        row = self.add_slider(trigger_group, row, "麦克风增益", "input_gain", 1.0, 5.0, 0.1)
        row = self.add_slider(trigger_group, row, "录音触发阈值", "record_threshold", 0.01, 0.04, 0.001)
        row = self.add_slider(trigger_group, row, "转写判定阈值", "transcribe_threshold", 0.01, 0.06, 0.001)
        row = self.add_slider(trigger_group, row, "触发前保留秒数", "pre_roll_seconds", 0.0, 3.0, 0.1)
        row = self.add_slider(trigger_group, row, "低于转写线等待秒数", "transcribe_pause_seconds", 0.2, 2.0, 0.1)
        row = self.add_slider(trigger_group, row, "噪声丢弃等待秒数", "silence_seconds", 1.0, 8.0, 0.1)
        row = self.add_slider(trigger_group, row, "最长分段秒数", "max_phrase_seconds", 10.0, 120.0, 1.0)

        presets = ttk.LabelFrame(trigger_settings, text="办公室预设", padding=12)
        presets.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        presets.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(presets, text="灵敏", command=lambda: self.set_thresholds(0.010, 0.015)).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(presets, text="办公室", command=lambda: self.set_thresholds(0.020, 0.030)).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(presets, text="严格", command=lambda: self.set_thresholds(0.040, 0.050)).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        app_group = ttk.LabelFrame(app_settings, text="启动行为", padding=12)
        app_group.grid(row=0, column=0, sticky="ew")
        app_group.columnconfigure(1, weight=1)
        ttk.Checkbutton(app_group, text="启动后自动开始", variable=self.vars["auto_start"]).grid(row=0, column=0, columnspan=2, sticky="w")
        self.add_slider(app_group, 1, "缓存保留分钟", "cache_retention_minutes", 0.0, 60.0, 1.0)

        bubble_group = ttk.LabelFrame(app_settings, text="反向路由气泡", padding=12)
        bubble_group.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        bubble_group.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            bubble_group,
            text="启用左下角气泡",
            variable=self.vars["reply_bubble_enabled"],
            command=self.configure_reply_bubble_server,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row = 1
        ttk.Label(bubble_group, text="监听端口").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Spinbox(
            bubble_group,
            from_=1024,
            to=65535,
            increment=1,
            textvariable=self.vars["reply_bubble_port"],
            width=10,
            command=self.configure_reply_bubble_server,
        ).grid(row=row, column=1, sticky="w", pady=5)
        row += 1
        row = self.add_slider(bubble_group, row, "持续秒数", "reply_bubble_seconds", 1.0, 10.0, 0.5)
        ttk.Label(bubble_group, text="访问令牌").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Entry(bubble_group, textvariable=self.vars["reply_bubble_token"], show="*", width=24).grid(row=row, column=1, sticky="ew", pady=5)

        storage_group = ttk.LabelFrame(app_settings, text="本地数据", padding=12)
        storage_group.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        storage_group.columnconfigure(1, weight=1)
        ttk.Label(storage_group, text="配置").grid(row=0, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Label(storage_group, text=str(CONFIG_PATH), style="Muted.TLabel", wraplength=250).grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(storage_group, text="模型缓存").grid(row=1, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Label(storage_group, text=str(APP_DIR / "cache" / "models"), style="Muted.TLabel", wraplength=250).grid(row=1, column=1, sticky="ew", pady=5)
        folder_buttons = ttk.Frame(storage_group, style="Panel.TFrame")
        folder_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        folder_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(folder_buttons, text="打开转写目录", command=self.open_output_folder).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(folder_buttons, text="打开缓存目录", command=self.open_cache_folder).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        route_group = ttk.LabelFrame(route_settings, text="RabiRoute 输出", padding=12)
        route_group.grid(row=0, column=0, sticky="ew")
        route_group.columnconfigure(1, weight=1)
        ttk.Checkbutton(route_group, text="转写完成后推送到 RabiRoute", variable=self.vars["rabiroute_enabled"]).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row = 1
        row = self.add_entry(route_group, row, "推送 URL", "rabiroute_url")
        row = self.add_entry(route_group, row, "来源 ID", "rabiroute_source")
        ttk.Label(route_group, text="访问令牌").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Entry(route_group, textvariable=self.vars["rabiroute_token"], show="*").grid(row=row, column=1, sticky="ew", pady=5)

        right = ttk.Frame(main, padding=0, style="Panel.TFrame")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        main.add(right, weight=1)

        monitor = ttk.Frame(right, padding=14, style="Panel.TFrame")
        monitor.grid(row=0, column=0, sticky="ew")
        monitor.columnconfigure(0, weight=1)
        monitor.columnconfigure(1, weight=0)
        ttk.Label(monitor, text="实时监测", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(monitor, textvariable=self.vars["trigger_state"], style="Status.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(monitor, textvariable=self.vars["status"], style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        meter = ttk.Frame(monitor, style="Panel.TFrame")
        meter.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        meter.columnconfigure(0, weight=1)
        self.level_bar = ttk.Progressbar(meter, variable=self.vars["mic_level"], maximum=0.04, style="Fenne.Horizontal.TProgressbar")
        self.level_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(meter, textvariable=self.vars["mic_level_text"], style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        self.wave_canvas = tk.Canvas(monitor, height=132, bg=THEME["canvas"], highlightthickness=1, highlightbackground=THEME["line"])
        self.wave_canvas.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.buddy_images: dict[str, tk.PhotoImage] = {}
        for state, path in BUDDY_STATE_IMAGE_PATHS.items():
            if not path.exists():
                continue
            try:
                self.buddy_images[state] = tk.PhotoImage(file=str(path))
            except tk.TclError:
                pass
        if not self.buddy_images and BUDDY_IMAGE_PATH.exists():
            try:
                self.buddy_images["listening"] = tk.PhotoImage(file=str(BUDDY_IMAGE_PATH))
            except tk.TclError:
                pass

        transcript_frame = ttk.Frame(right, padding=(14, 0, 14, 14), style="Panel.TFrame")
        transcript_frame.grid(row=1, column=0, sticky="nsew")
        transcript_frame.columnconfigure(0, weight=1)
        transcript_frame.rowconfigure(1, weight=1)

        preview_header = ttk.Frame(transcript_frame, padding=(0, 14, 0, 8), style="Panel.TFrame")
        preview_header.grid(row=0, column=0, columnspan=2, sticky="ew")
        preview_header.columnconfigure(0, weight=1)
        ttk.Label(preview_header, text="转写预览", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(preview_header, text="清空预览", command=self.clear_preview).grid(row=0, column=1, sticky="e")

        self.transcript = tk.Text(
            transcript_frame,
            wrap="word",
            undo=False,
            bg=THEME["canvas"],
            fg=THEME["ink"],
            insertbackground=THEME["teal_dark"],
            relief="flat",
            padx=14,
            pady=12,
            font=("Microsoft YaHei UI", 10),
            selectbackground=THEME["teal_soft"],
            selectforeground=THEME["ink"],
            highlightthickness=1,
            highlightbackground=THEME["line"],
            highlightcolor=THEME["teal"],
        )
        self.transcript.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.transcript.configure(yscrollcommand=scrollbar.set)
        self.transcript.tag_configure("placeholder", foreground=THEME["muted"])
        self.transcript.tag_configure("log", foreground=THEME["sand_dark"])
        self.transcript.tag_configure("text", foreground=THEME["ink"])
        self.show_placeholder()

    def add_combo(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> tuple[ttk.Combobox, int]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=32)
        combo.grid(row=row, column=1, sticky="ew", pady=5)
        return combo, row + 1

    def add_entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", pady=5)
        return row + 1

    def add_slider(self, parent: ttk.Frame, row: int, label: str, key: str, start: float, end: float, step: float) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", pady=5)
        holder.columnconfigure(0, weight=1)
        ttk.Scale(holder, variable=self.vars[key], from_=start, to=end).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(holder, textvariable=self.vars[key], width=8).grid(row=0, column=1)
        return row + 1

    def set_threshold(self, value: float) -> None:
        self.vars["record_threshold"].set(value)
        self.vars["transcribe_threshold"].set(max(value, float(self.vars["transcribe_threshold"].get())))
        self.update_level_text(float(self.vars["mic_level"].get()))

    def set_thresholds(self, record_threshold: float, transcribe_threshold: float) -> None:
        self.vars["record_threshold"].set(record_threshold)
        self.vars["transcribe_threshold"].set(max(record_threshold, transcribe_threshold))
        self.update_level_text(float(self.vars["mic_level"].get()))

    def update_level_text(self, level: float, raw_level: float | None = None) -> None:
        record_threshold = self.current_preview_threshold()
        transcribe_threshold = float(self.vars["transcribe_threshold"].get())
        peak = max(self.level_history) if self.level_history else level
        raw = level if raw_level is None else raw_level
        self.vars["mic_level_text"].set(f"原始 {raw:.3f} / 增益后 {level:.3f} / 录音 {record_threshold:.3f} / 转写 {transcribe_threshold:.3f} / 峰值 {peak:.3f}")

    def current_preview_threshold(self) -> float:
        min_threshold = float(self.vars["record_threshold"].get())
        if not bool(self.vars["adaptive_threshold"].get()):
            return min_threshold
        multiplier = float(self.config_data.get("adaptive_threshold_multiplier", DEFAULT_CONFIG.get("adaptive_threshold_multiplier", 2.5)))
        margin = float(self.config_data.get("adaptive_threshold_margin", DEFAULT_CONFIG.get("adaptive_threshold_margin", 0.004)))
        return max(min_threshold, self.preview_noise_floor * multiplier + margin)

    def start_level_monitor(self) -> None:
        self.stop_level_monitor()
        device = self.selected_device_index()
        sample_rate = int(self.config_data.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
        blocksize = max(512, int(sample_rate * 0.08))

        def callback(indata, frames, callback_time, status):
            if status:
                return
            raw_samples = indata[:, 0].astype(np.float32)
            raw_level = float(np.sqrt(np.mean(np.square(raw_samples)))) if raw_samples.size else 0.0
            samples = raw_samples
            gain = float(self.vars["input_gain"].get())
            if gain != 1.0:
                samples = np.clip(samples * gain, -1.0, 1.0)
            level = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            self.level_queue.put((level, raw_level))

        try:
            self.level_stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=sample_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            )
            self.level_stream.start()
        except Exception as exc:
            self.level_stream = None
            self.vars["status"].set(f"音量预览启动失败：{exc}")

    def stop_level_monitor(self) -> None:
        if self.level_stream is None:
            return
        try:
            self.level_stream.stop()
            self.level_stream.close()
        except Exception:
            pass
        self.level_stream = None

    def restart_level_monitor(self) -> None:
        self.start_level_monitor()
        self.vars["status"].set("已切换麦克风")

    def drain_level_preview(self) -> None:
        latest: float | None = None
        latest_raw: float | None = None
        while not self.level_queue.empty():
            latest, latest_raw = self.level_queue.get_nowait()
        if latest is not None:
            latest_raw = latest_raw or 0.0
            self.display_level = (self.display_level * 0.75) + (latest * 0.25)
            self.display_raw_level = (self.display_raw_level * 0.75) + (latest_raw * 0.25)
            threshold = self.current_preview_threshold()
            transcribe_threshold = float(self.vars["transcribe_threshold"].get())
            if latest < threshold:
                self.preview_noise_floor = (self.preview_noise_floor * 0.95) + (latest * 0.05)
            now = time.monotonic()
            if latest >= threshold:
                self.preview_triggered = True
                self.last_trigger_time = now
            elif self.preview_triggered and (now - self.last_trigger_time) >= float(self.vars["silence_seconds"].get()):
                self.preview_triggered = False

            self.level_history.append(self.display_level)
            self.raw_level_history.append(self.display_raw_level)
            self.bar_history.append(self.display_level)
            max_scale = max(0.04, threshold, transcribe_threshold, max(self.level_history, default=0.0)) * 1.15
            self.level_bar.configure(maximum=max_scale)
            shown = min(self.display_level, max_scale)
            self.vars["mic_level"].set(shown)
            if self.preview_triggered:
                state = "状态：录音中，已达到转写条件" if max(self.level_history, default=0.0) >= transcribe_threshold else "状态：录音中，等待达到转写条件"
            else:
                state = "状态：监听中，等待超过录音阈值"
            self.vars["trigger_state"].set(state)
            self.update_buddy_state(latest, threshold)
            self.update_level_text(self.display_level, self.display_raw_level)
            self.draw_scrolling_bars(max_scale, threshold, transcribe_threshold)
        else:
            self.update_level_text(float(self.vars["mic_level"].get()), self.raw_level_history[-1] if self.raw_level_history else None)
        self.after(80, self.drain_level_preview)

    def update_buddy_state(self, latest_level: float, record_threshold: float) -> None:
        now = time.monotonic()
        if now < self.writing_until:
            self.buddy_state = "writing" if int(now * 2.5) % 2 == 0 else "listening"
        elif latest_level >= record_threshold or self.preview_triggered:
            self.buddy_state = "listening"
        else:
            self.buddy_state = "idle"

    def draw_scrolling_bars(self, max_scale: float, record_threshold: float, transcribe_threshold: float) -> None:
        canvas = self.wave_canvas
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.delete("all")
        record_y = height - min(record_threshold / max_scale, 1.0) * height
        transcribe_y = height - min(transcribe_threshold / max_scale, 1.0) * height
        canvas.create_rectangle(0, 0, width, height, fill=THEME["canvas"], outline="")
        for guide_index in range(1, 4):
            y = height * guide_index / 4
            canvas.create_line(0, y, width, y, fill="#f2e5cf", width=1)
        canvas.create_line(0, record_y, width, record_y, fill=THEME["danger"], width=2)
        record_label_y = max(10, min(height - 10, record_y - 8))
        transcribe_label_y = max(10, min(height - 10, transcribe_y - 8))
        if abs(record_label_y - transcribe_label_y) < 14:
            record_label_y = max(10, record_label_y - 8)
            transcribe_label_y = min(height - 10, transcribe_label_y + 8)
        canvas.create_text(8, record_label_y, text="录音线", anchor="w", fill=THEME["danger"], font=("Microsoft YaHei UI", 9, "bold"))
        canvas.create_line(0, transcribe_y, width, transcribe_y, fill=THEME["sand"], width=2, dash=(4, 3))
        canvas.create_text(width - 8, transcribe_label_y, text="转写线", anchor="e", fill=THEME["sand_dark"], font=("Microsoft YaHei UI", 9, "bold"))
        values = list(self.bar_history)
        if not values:
            return
        gap = 3
        bar_width = 4
        max_bars = max(1, int(width // (bar_width + gap)))
        values = values[-max_bars:]
        if len(values) < max_bars:
            values = [0.0] * (max_bars - len(values)) + values
        for index, value in enumerate(values):
            x0 = index * (bar_width + gap)
            x1 = x0 + bar_width
            normalized = min(value / max_scale, 1.0)
            bar_height = max(2.0, normalized * (height - 8)) if value > 0 else 0
            y0 = height - bar_height
            color = THEME["quiet_bar"] if value < record_threshold else THEME["teal"]
            if value >= transcribe_threshold:
                color = THEME["green"]
            canvas.create_rectangle(x0, y0, x1, height, fill=color, outline="")
            if value >= record_threshold:
                canvas.create_rectangle(x0, max(y0, height - 4), x1, height, fill="#b8f0d8", outline="")
        buddy_image = self.buddy_images.get(self.buddy_state) or self.buddy_images.get("listening")
        if buddy_image:
            canvas.create_image(width - 96, 70, image=buddy_image, anchor="center")

    def list_input_devices(self) -> list[tuple[int, str]]:
        devices: list[tuple[int, str]] = []
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) > 0:
                host_api = sd.query_hostapis(device["hostapi"])["name"]
                devices.append((index, f"{index}: {device['name']} ({host_api})"))
        return devices

    def device_label_for(self, index: int | None) -> str:
        if index is None:
            return "系统默认麦克风"
        for device_index, label in getattr(self, "devices", []):
            if device_index == index:
                return label
        return f"{index}: 未知设备"

    def selected_device_index(self) -> int | None:
        label = self.vars["mic_device"].get()
        if label == "系统默认麦克风":
            return None
        try:
            return int(label.split(":", 1)[0])
        except ValueError:
            return None

    def collect_config(self) -> dict:
        config = DEFAULT_CONFIG.copy()
        config.update(load_config(CONFIG_PATH))
        config.update(
            {
                "model": self.vars["model"].get(),
                "auto_start": bool(self.vars["auto_start"].get()),
                "device": "cuda",
                "compute_type": self.vars["compute_type"].get(),
                "language": LANGUAGE_CODES.get(self.vars["language"].get(), "zh"),
                "language_mode": LANGUAGE_CODES.get(self.vars["language"].get(), "zh"),
                "simplify_chinese": bool(self.vars["simplify_chinese"].get()),
                "adaptive_threshold": bool(self.vars["adaptive_threshold"].get()),
                "input_gain": round(float(self.vars["input_gain"].get()), 1),
                "record_threshold": round(float(self.vars["record_threshold"].get()), 4),
                "transcribe_threshold": round(max(float(self.vars["record_threshold"].get()), float(self.vars["transcribe_threshold"].get())), 4),
                "rms_threshold": round(float(self.vars["record_threshold"].get()), 4),
                "pre_roll_seconds": round(float(self.vars["pre_roll_seconds"].get()), 1),
                "transcribe_pause_seconds": round(float(self.vars["transcribe_pause_seconds"].get()), 1),
                "silence_seconds": round(float(self.vars["silence_seconds"].get()), 2),
                "max_phrase_seconds": round(float(self.vars["max_phrase_seconds"].get()), 1),
                "cache_dir": str(config.get("cache_dir", DEFAULT_CONFIG["cache_dir"])),
                "cache_retention_minutes": round(max(0.0, min(60.0, float(self.vars["cache_retention_minutes"].get()))), 1),
                "mic_device": self.selected_device_index(),
                "reply_bubble_enabled": bool(self.vars["reply_bubble_enabled"].get()),
                "reply_bubble_port": max(1024, min(65535, int(self.vars["reply_bubble_port"].get()))),
                "reply_bubble_seconds": round(max(1.0, min(10.0, float(self.vars["reply_bubble_seconds"].get()))), 1),
                "reply_bubble_token": self.vars["reply_bubble_token"].get().strip(),
                "rabiroute_enabled": bool(self.vars["rabiroute_enabled"].get()),
                "rabiroute_url": self.vars["rabiroute_url"].get().strip(),
                "rabiroute_token": self.vars["rabiroute_token"].get().strip(),
                "rabiroute_source": self.vars["rabiroute_source"].get().strip() or "fennenote",
            }
        )
        return config

    def save_config(self) -> None:
        config = self.collect_config()
        write_config(CONFIG_PATH, config)
        self.config_data = config
        self.configure_reply_bubble_server()
        self.vars["status"].set("配置已保存")

    def install_selected_model(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("安装模型", "转写正在运行，请先停止后再安装模型。")
            return
        config = self.collect_config()
        write_config(CONFIG_PATH, config)
        self.config_data = config
        model_name = str(config.get("model", DEFAULT_CONFIG["model"]))
        self.download_model_button.configure(state="disabled")
        self.vars["model_cache_status"].set(f"模型缓存：正在安装 {model_name}")
        self.append_log(f"正在安装模型：{model_name}")

        def status_callback(code: str, message: str) -> None:
            self.output_queue.put(f"FN_STATUS|{code}|{message}")

        def worker() -> None:
            try:
                download_configured_model(config, status_callback=status_callback)
            except Exception as exc:
                self.output_queue.put(f"FN_STATUS|model_download_error|模型安装失败：{exc}")
            finally:
                self.output_queue.put("__MODEL_DOWNLOAD_DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def start_transcriber(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.save_config()
        self.is_paused = False
        self.tail_position = 0
        self.show_placeholder()

        if getattr(sys, "frozen", False):
            command = [sys.executable, "--transcribe-child", "--config", str(CONFIG_PATH)]
        else:
            command = [sys.executable, str(APP_DIR / "transcribe_mic.py"), "--config", str(CONFIG_PATH)]
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        child_env = prepare_process_environment(self.collect_config())
        self.process = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            startupinfo=startupinfo,
            env=child_env,
        )
        threading.Thread(target=self.read_process_output, daemon=True).start()
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="暂停")
        self.stop_button.configure(state="normal")
        self.vars["status"].set("正在启动")

    def read_process_output(self) -> None:
        assert self.process and self.process.stdout
        for raw_line in self.process.stdout:
            self.output_queue.put(decode_process_output_line(raw_line))
        self.output_queue.put("__PROCESS_EXITED__")

    def drain_process_output(self) -> None:
        while not self.output_queue.empty():
            line = self.output_queue.get_nowait()
            if line == "__PROCESS_EXITED__":
                self.start_button.configure(state="normal")
                self.pause_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
                self.vars["status"].set("已停止")
            elif line == "__MODEL_DOWNLOAD_DONE__":
                self.download_model_button.configure(state="normal")
            elif line.startswith("FN_STATUS|"):
                self.handle_worker_status(line)
            elif line.startswith("["):
                self.append_text(line)
            else:
                if any(marker in line for marker in IGNORED_PROCESS_OUTPUT_MARKERS):
                    continue
                status = line.strip()[:90] or "运行中"
                if "cublas64_12.dll" in line or "Library cublas64" in line:
                    status = "CUDA 运行库缺失，已停止。请使用 run_gui.ps1 创建 GPU 环境，或安装匹配的 CUDA/cuDNN。"
                self.vars["status"].set(status)
                if status:
                    self.append_log(status)
        self.after(150, self.drain_process_output)

    def handle_worker_status(self, line: str) -> None:
        parts = line.strip().split("|", 2)
        if len(parts) < 3:
            return
        _prefix, code, message = parts
        self.vars["status"].set(message)
        if code in {"model_loading", "model_download_start", "model_download_ready", "model_download_error", "queued", "transcribing", "discarded", "fatal"}:
            self.append_log(message)
        if code == "model_download_ready":
            self.vars["model_cache_status"].set("模型缓存：已安装")
        elif code == "model_download_error":
            self.vars["model_cache_status"].set("模型缓存：安装失败")
        if code == "transcribing":
            self.writing_until = time.monotonic() + 2.8

    def append_text(self, line: str) -> None:
        self.writing_until = time.monotonic() + 2.8
        self.clear_placeholder()
        self.transcript.insert(tk.END, line, "text")
        self.transcript.see(tk.END)

    def append_log(self, line: str) -> None:
        self.clear_placeholder()
        self.transcript.insert(tk.END, f"系统：{line.strip()}\n", "log")
        self.transcript.see(tk.END)

    def configure_reply_bubble_server(self) -> None:
        self.stop_reply_bubble_server()
        if not bool(self.vars["reply_bubble_enabled"].get()):
            return
        try:
            port = max(1024, min(65535, int(self.vars["reply_bubble_port"].get())))
        except (tk.TclError, ValueError):
            port = int(DEFAULT_CONFIG["reply_bubble_port"])
            self.vars["reply_bubble_port"].set(port)
        token = self.vars["reply_bubble_token"].get().strip()
        owner = self

        class ReplyHandler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def do_POST(self) -> None:
                if self.path not in {"/reply", "/bubble", "/rabiroute/reply"}:
                    self.send_response(404)
                    self.end_headers()
                    return
                expected = token
                provided = self.headers.get("X-FenneNote-Token", "")
                if expected and provided != expected:
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(min(length, 65536)).decode("utf-8-sig") or "{}")
                except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                    self.send_response(400)
                    self.end_headers()
                    return
                title = str(payload.get("title") or payload.get("sender") or "RabiRoute")
                text = str(payload.get("text") or payload.get("message") or payload.get("content") or "").strip()
                if not text:
                    self.send_response(400)
                    self.end_headers()
                    return
                owner.after(0, lambda: owner.show_reply_bubble(title, text))
                self.send_response(204)
                self.end_headers()

        try:
            self.reply_server = ThreadingHTTPServer(("127.0.0.1", port), ReplyHandler)
            self.reply_server_thread = threading.Thread(target=self.reply_server.serve_forever, daemon=True)
            self.reply_server_thread.start()
            self.vars["status"].set(f"反向气泡监听：127.0.0.1:{port}/reply")
        except OSError as exc:
            self.reply_server = None
            self.reply_server_thread = None
            self.vars["status"].set(f"反向气泡监听失败：{exc}")

    def stop_reply_bubble_server(self) -> None:
        if self.reply_server is None:
            return
        server = self.reply_server
        self.reply_server = None
        self.reply_server_thread = None
        try:
            server.shutdown()
            server.server_close()
        except OSError:
            pass

    def show_reply_bubble(self, title: str, text: str) -> None:
        if not bool(self.vars["reply_bubble_enabled"].get()):
            return
        if self.reply_bubble is not None and self.reply_bubble.winfo_exists():
            self.reply_bubble.destroy()
        bubble = tk.Toplevel(self)
        self.reply_bubble = bubble
        bubble.overrideredirect(True)
        bubble.attributes("-topmost", True)
        bubble.configure(bg=THEME["teal_dark"])
        body = tk.Frame(bubble, bg=THEME["teal_soft"], padx=14, pady=12, highlightthickness=1, highlightbackground=THEME["teal"])
        body.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(body, text=title[:48], bg=THEME["teal_soft"], fg=THEME["teal_dark"], font=("Microsoft YaHei UI", 10, "bold"), anchor="w").pack(fill="x")
        tk.Label(body, text=text[:240], bg=THEME["teal_soft"], fg=THEME["ink"], font=("Microsoft YaHei UI", 10), justify="left", anchor="w", wraplength=300).pack(fill="x", pady=(6, 0))
        bubble.update_idletasks()
        width = max(280, min(360, bubble.winfo_reqwidth()))
        height = max(90, bubble.winfo_reqheight())
        screen_width = bubble.winfo_screenwidth()
        screen_height = bubble.winfo_screenheight()
        x = 18
        y = screen_height - height - 58
        bubble.geometry(f"{width}x{height}+{x}+{y}")
        seconds = max(1.0, min(10.0, float(self.vars["reply_bubble_seconds"].get())))
        bubble.after(int(seconds * 1000), lambda: bubble.destroy() if bubble.winfo_exists() else None)

    def show_placeholder(self) -> None:
        self.transcript.delete("1.0", tk.END)
        self.transcript.insert(
            tk.END,
            "点击左上角“开始”后，说话内容会显示在这里。\n"
            "如果音量条有波动但没有文字，通常是触发阈值过高、说话时间太短，或模型仍在加载。\n",
            "placeholder",
        )

    def clear_placeholder(self) -> None:
        if self.transcript.tag_ranges("placeholder"):
            self.transcript.delete("1.0", tk.END)

    def clear_preview(self) -> None:
        self.show_placeholder()

    def toggle_pause(self) -> None:
        if not self.process or not self.process.stdin or self.process.poll() is not None:
            return
        command = b"r\n" if self.is_paused else b"p\n"
        self.process.stdin.write(command)
        self.process.stdin.flush()
        self.is_paused = not self.is_paused
        self.pause_button.configure(text="继续" if self.is_paused else "暂停")
        self.vars["status"].set("已暂停" if self.is_paused else "运行中")

    def stop_transcriber(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.write(b"q\n")
                self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.terminate()
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.vars["status"].set("已停止")

    def refresh_tail(self, reset: bool = False) -> None:
        if reset:
            self.tail_position = 0
        try:
            path = today_output_path((APP_DIR / str(self.collect_config()["output_dir"])).resolve())
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(self.tail_position)
                    data = handle.read()
                    self.tail_position = handle.tell()
                if data and (not self.process or self.process.poll() is not None):
                    self.append_text(data)
        except Exception:
            pass
        self.after(1000, self.refresh_tail)

    def open_output_folder(self) -> None:
        output_dir = (APP_DIR / str(self.collect_config()["output_dir"])).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(output_dir)])

    def open_cache_folder(self) -> None:
        cache_dir = (APP_DIR / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(cache_dir)])

    def on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("实时语音转文字", "停止转写并关闭窗口吗？"):
                return
            self.stop_transcriber()
        self.stop_reply_bubble_server()
        self.stop_level_monitor()
        self.destroy()


def main() -> int:
    if "--transcribe-child" in sys.argv:
        sys.argv.remove("--transcribe-child")
        try:
            return transcribe_cli_main()
        except Exception as exc:
            print(f"FN_STATUS|fatal|转写子进程异常：{exc}", flush=True)
            traceback.print_exc()
            return 1
    ensure_cuda_dll_path()
    configure_local_storage()
    app = TranscriberGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
