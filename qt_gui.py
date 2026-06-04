from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from transcribe_mic import (
    DEFAULT_CONFIG,
    DOWNLOADABLE_MODELS,
    MODEL_PROFILES,
    configure_local_storage,
    load_config,
    model_is_installed,
    save_config as write_config,
)


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
ICON_PATH = APP_DIR / "assets" / "fennenote.ico"
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
MODEL_SOURCE_LABELS = {"local": "本地模型", "api": "API 模型"}
MODEL_SOURCE_CODES = {label: code for code, label in MODEL_SOURCE_LABELS.items()}
API_PROVIDER_LABELS = {"dashscope": "千问 / DashScope", "openai_compatible": "OpenAI Compatible"}
API_PROVIDER_CODES = {label: code for code, label in API_PROVIDER_LABELS.items()}
API_MODELS_BY_PROVIDER = {
    "dashscope": ["qwen-audio-turbo", "qwen-audio-asr", "qwen2.5-omni-7b"],
    "openai_compatible": ["gpt-4o-transcribe", "whisper-1"],
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


class LevelWorker(QObject):
    level = Signal(float, float)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.config: dict = {}
        self.device: int | None = None

    def configure(self, config: dict, device: int | None) -> None:
        self.config = config.copy()
        self.device = device

    def run(self) -> None:
        self.running = True
        sample_rate = int(self.config.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
        blocksize = max(512, int(sample_rate * 0.08))
        gain = float(self.config.get("input_gain", DEFAULT_CONFIG["input_gain"]))

        def callback(indata, _frames, _time_info, status):
            if status:
                return
            raw_samples = indata[:, 0].astype(np.float32)
            raw_level = float(np.sqrt(np.mean(np.square(raw_samples)))) if raw_samples.size else 0.0
            samples = np.clip(raw_samples * gain, -1.0, 1.0) if gain != 1.0 else raw_samples
            level = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            self.level.emit(level, raw_level)

        try:
            with sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=sample_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            ):
                while self.running:
                    time.sleep(0.05)
        except Exception as exc:
            self.error.emit(f"音量预览启动失败：{exc}")

    def stop(self) -> None:
        self.running = False


class ProcessWorker(QObject):
    line = Signal(str)
    exited = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.process: subprocess.Popen[bytes] | None = None
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        self.config = config.copy()

    def run(self) -> None:
        python_exe = Path(sys.executable)
        if python_exe.name.lower() == "pythonw.exe":
            python_exe = python_exe.with_name("python.exe")
        command = [str(python_exe), str(APP_DIR / "transcribe_mic.py"), "--config", str(CONFIG_PATH)]
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                startupinfo=startupinfo,
                env=prepare_process_environment(self.config),
            )
            assert self.process.stdout is not None
            for raw_line in self.process.stdout:
                self.line.emit(decode_process_output_line(raw_line).rstrip())
            self.exited.emit(self.process.wait())
        except Exception as exc:
            self.line.emit(f"启动失败：{exc}")
            self.exited.emit(-1)

    def send(self, text: str) -> None:
        if not self.process or self.process.poll() is not None or not self.process.stdin:
            return
        try:
            self.process.stdin.write(text.encode("utf-8"))
            self.process.stdin.flush()
        except OSError:
            pass

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


class WaveWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(140)
        self.values: deque[float] = deque(maxlen=120)
        self.record_threshold = 0.01
        self.transcribe_threshold = 0.015
        self.scale = 0.04

    def push(self, level: float, record_threshold: float, transcribe_threshold: float) -> None:
        self.values.append(level)
        self.record_threshold = record_threshold
        self.transcribe_threshold = transcribe_threshold
        self.scale = max(0.04, max(self.values, default=0.0), record_threshold, transcribe_threshold) * 1.15
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        painter.fillRect(rect, QColor("#fffafc"))
        painter.setPen(QPen(QColor("#f2dbe3"), 1))
        for index in range(1, 4):
            y = rect.height() * index / 4
            painter.drawLine(0, y, rect.width(), y)
        record_y = rect.height() - min(self.record_threshold / self.scale, 1.0) * rect.height()
        transcribe_y = rect.height() - min(self.transcribe_threshold / self.scale, 1.0) * rect.height()
        painter.setPen(QPen(QColor("#d45b4c"), 2))
        painter.drawLine(0, record_y, rect.width(), record_y)
        painter.setPen(QPen(QColor("#f3a43b"), 2, Qt.DashLine))
        painter.drawLine(0, transcribe_y, rect.width(), transcribe_y)
        values = list(self.values)
        if not values:
            return
        gap = 3
        bar_width = 5
        max_bars = max(1, rect.width() // (bar_width + gap))
        values = values[-max_bars:]
        x = rect.width() - len(values) * (bar_width + gap)
        for value in values:
            normalized = min(value / self.scale, 1.0)
            height = max(2, normalized * (rect.height() - 8)) if value > 0 else 0
            color = QColor("#4b3d46")
            if value >= self.record_threshold:
                color = QColor("#28aaa4")
            if value >= self.transcribe_threshold:
                color = QColor("#22b86f")
            painter.fillRect(QRectF(x, rect.height() - height, bar_width, height), color)
            x += bar_width + gap


class FenneNoteQt(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_cuda_dll_path()
        self.config_data = load_config(CONFIG_PATH)
        self.devices = self.list_input_devices()
        self.level_thread: QThread | None = None
        self.level_worker: LevelWorker | None = None
        self.process_thread: QThread | None = None
        self.process_worker: ProcessWorker | None = None
        self.preview_noise_floor = 0.003
        self.display_level = 0.0
        self.display_raw_level = 0.0
        self.level_history: deque[float] = deque(maxlen=160)
        self.build_ui()
        self.apply_config_to_ui()
        self.start_level_worker()

    def build_ui(self) -> None:
        self.setWindowTitle("FenneNote - Qt Control Console")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 760)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #fff5f7; color: #352b31; font-family: "Microsoft YaHei UI"; }
            #nav { background: #28252b; }
            #nav QLabel { color: #fff8fb; background: transparent; }
            QListWidget { background: #28252b; color: #fff8fb; border: none; outline: 0; }
            QListWidget::item { padding: 12px 14px; border-radius: 6px; margin: 3px 8px; }
            QListWidget::item:selected { background: #f26f95; color: white; }
            QFrame#card, QGroupBox { background: #fffdf9; border: 1px solid #efd8dd; border-radius: 8px; }
            QGroupBox { margin-top: 12px; padding: 16px 12px 12px 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #a56513; }
            QPushButton { background: #fff8fb; border: 1px solid #e7b9c4; border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #ffe8f0; }
            QPushButton#primary { background: #28aaa4; color: white; border-color: #13746f; font-weight: 700; }
            QPushButton#danger { color: #d45b4c; background: #fff0ec; border-color: #efb4aa; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #fffdf8; border: 1px solid #efd8dd; border-radius: 6px; padding: 6px; }
            QPlainTextEdit { background: #fffafc; border: 1px solid #efd8dd; border-radius: 8px; padding: 10px; }
            """
        )

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("nav")
        nav.setFixedWidth(190)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(12, 18, 12, 12)
        title = QLabel("FenneNote")
        title.setStyleSheet("font-size: 18px; font-weight: 800;")
        nav_layout.addWidget(title)
        subtitle = QLabel("Kanban Console")
        subtitle.setStyleSheet("color: #d7c7cf;")
        nav_layout.addWidget(subtitle)
        self.nav_list = QListWidget()
        for text in ("总览", "输入", "模型", "触发", "应用", "路由"):
            QListWidgetItem(text, self.nav_list)
        nav_layout.addWidget(self.nav_list, 1)
        layout.addWidget(nav)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)
        layout.addWidget(main, 1)

        topbar = QFrame()
        topbar.setObjectName("card")
        topbar_layout = QHBoxLayout(topbar)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        topbar_layout.addWidget(self.status_label, 1)
        self.start_button = QPushButton("开始")
        self.start_button.setObjectName("primary")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.save_button = QPushButton("保存配置")
        self.folder_button = QPushButton("打开目录")
        for button in (self.start_button, self.pause_button, self.stop_button, self.save_button, self.folder_button):
            topbar_layout.addWidget(button)
        main_layout.addWidget(topbar)

        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages, 1)
        self.build_dashboard_page()
        self.build_input_page()
        self.build_model_page()
        self.build_trigger_page()
        self.build_app_page()
        self.build_route_page()
        self.nav_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        self.start_button.clicked.connect(self.start_transcriber)
        self.pause_button.clicked.connect(lambda: self.send_process_command("p\n"))
        self.stop_button.clicked.connect(self.stop_transcriber)
        self.save_button.clicked.connect(self.save_config)
        self.folder_button.clicked.connect(self.open_output_folder)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def page(self) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 10, 14)
        layout.setSpacing(14)
        scroll.setWidget(container)
        self.pages.addWidget(scroll)
        return scroll, container, layout

    def build_dashboard_page(self) -> None:
        scroll, container, layout = self.page()
        hero = QFrame()
        hero.setObjectName("card")
        hero_layout = QVBoxLayout(hero)
        title = QLabel("Fenne Control Room")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #f26f95;")
        hero_layout.addWidget(title)
        self.trigger_state_label = QLabel("状态：监听中")
        hero_layout.addWidget(self.trigger_state_label)
        layout.addWidget(hero)

        monitor = QGroupBox("实时监听")
        monitor_layout = QVBoxLayout(monitor)
        self.level_text = QLabel("原始 0.000 / 增益后 0.000 / 录音 0.010 / 转写 0.015")
        monitor_layout.addWidget(self.level_text)
        self.wave = WaveWidget()
        monitor_layout.addWidget(self.wave)
        layout.addWidget(monitor)

        transcript_group = QGroupBox("转写预览")
        transcript_layout = QVBoxLayout(transcript_group)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("转写结果会显示在这里。")
        transcript_layout.addWidget(self.transcript, 1)
        layout.addWidget(transcript_group, 1)

    def build_input_page(self) -> None:
        _, _, layout = self.page()
        group = QGroupBox("输入与模型来源")
        form = QFormLayout(group)
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("系统默认麦克风", None)
        for index, label in self.devices:
            self.mic_combo.addItem(label, index)
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(MODEL_SOURCE_CODES.keys()))
        self.local_model_combo = QComboBox()
        self.input_api_provider_combo = QComboBox()
        self.input_api_provider_combo.addItems(list(API_PROVIDER_CODES.keys()))
        self.input_api_model_combo = QComboBox()
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8_float16", "float16", "int8", "int8_float32", "float32"])
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(LANGUAGE_CODES.keys()))
        self.simplify_check = QCheckBox("输出简体中文")
        form.addRow("麦克风", self.mic_combo)
        form.addRow("模型来源", self.source_combo)
        form.addRow("本地模型", self.local_model_combo)
        form.addRow("API 提供方", self.input_api_provider_combo)
        form.addRow("API 模型", self.input_api_model_combo)
        form.addRow("计算精度", self.compute_combo)
        form.addRow("语言", self.language_combo)
        form.addRow("", self.simplify_check)
        layout.addWidget(group)
        self.source_combo.currentTextChanged.connect(self.refresh_source_visibility)
        self.input_api_provider_combo.currentTextChanged.connect(self.refresh_input_api_models)

    def build_model_page(self) -> None:
        _, _, layout = self.page()
        provider = QGroupBox("API Provider 配置")
        form = QFormLayout(provider)
        self.api_enabled_check = QCheckBox("启用 API Provider")
        self.api_provider_id = QLineEdit()
        self.api_provider_combo = QComboBox()
        self.api_provider_combo.addItems(list(API_PROVIDER_CODES.keys()))
        self.api_model_combo = QComboBox()
        self.api_base_url = QLineEdit()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.provider_status = QLabel("API Provider：未启用")
        actions = QHBoxLayout()
        self.activate_api_button = QPushButton("设为当前 API 来源")
        self.validate_api_button = QPushButton("校验配置")
        actions.addWidget(self.activate_api_button)
        actions.addWidget(self.validate_api_button)
        form.addRow("", self.api_enabled_check)
        form.addRow("Provider ID", self.api_provider_id)
        form.addRow("Provider 类型", self.api_provider_combo)
        form.addRow("模型", self.api_model_combo)
        form.addRow("Base URL", self.api_base_url)
        form.addRow("API Key", self.api_key)
        form.addRow("状态", self.provider_status)
        form.addRow("", actions)
        layout.addWidget(provider)

        local = QGroupBox("本地模型缓存")
        local_layout = QVBoxLayout(local)
        self.local_model_list = QListWidget()
        local_layout.addWidget(self.local_model_list)
        layout.addWidget(local, 1)
        self.activate_api_button.clicked.connect(self.activate_api_provider)
        self.validate_api_button.clicked.connect(self.validate_api_provider)
        self.api_provider_combo.currentTextChanged.connect(self.refresh_api_models)
        self.api_provider_combo.currentTextChanged.connect(self.sync_provider_to_input)
        self.api_model_combo.currentTextChanged.connect(self.sync_provider_to_input)
        self.input_api_provider_combo.currentTextChanged.connect(self.sync_input_to_provider)
        self.input_api_model_combo.currentTextChanged.connect(self.sync_input_to_provider)

    def build_trigger_page(self) -> None:
        _, _, layout = self.page()
        summary = QGroupBox("当前生效参数")
        summary_layout = QVBoxLayout(summary)
        self.trigger_summary = QLabel()
        self.trigger_summary.setWordWrap(True)
        summary_layout.addWidget(self.trigger_summary)
        layout.addWidget(summary)
        group = QGroupBox("阈值与分段")
        form = QFormLayout(group)
        self.adaptive_check = QCheckBox("动态识别环境底噪")
        self.input_gain = self.spin(1.0, 5.0, 0.1)
        self.record_threshold = self.spin(0.001, 0.2, 0.001, 3)
        self.transcribe_threshold = self.spin(0.001, 0.2, 0.001, 3)
        self.pre_roll = self.spin(0.0, 5.0, 0.1)
        self.pause_seconds = self.spin(0.1, 5.0, 0.1)
        self.silence_seconds = self.spin(0.2, 12.0, 0.1)
        self.max_phrase = self.spin(5.0, 180.0, 1.0)
        for label, widget in (
            ("触发模式", self.adaptive_check),
            ("麦克风增益", self.input_gain),
            ("录音触发阈值", self.record_threshold),
            ("转写判定阈值", self.transcribe_threshold),
            ("触发前保留秒数", self.pre_roll),
            ("低于转写线等待秒数", self.pause_seconds),
            ("噪声丢弃等待秒数", self.silence_seconds),
            ("最长分段秒数", self.max_phrase),
        ):
            form.addRow(label, widget)
        layout.addWidget(group)
        for widget in (self.input_gain, self.record_threshold, self.transcribe_threshold, self.pre_roll, self.pause_seconds, self.silence_seconds, self.max_phrase):
            widget.valueChanged.connect(self.update_trigger_summary)
        self.adaptive_check.stateChanged.connect(self.update_trigger_summary)

    def build_app_page(self) -> None:
        _, _, layout = self.page()
        group = QGroupBox("应用设置")
        form = QFormLayout(group)
        self.auto_start_check = QCheckBox("启动后自动开始")
        self.cache_retention = self.spin(0.0, 60.0, 1.0)
        self.bubble_enabled = QCheckBox("启用左下角气泡")
        self.bubble_port = QSpinBox()
        self.bubble_port.setRange(1024, 65535)
        self.bubble_seconds = self.spin(1.0, 10.0, 0.5)
        self.bubble_token = QLineEdit()
        self.bubble_token.setEchoMode(QLineEdit.Password)
        form.addRow("", self.auto_start_check)
        form.addRow("缓存保留分钟", self.cache_retention)
        form.addRow("", self.bubble_enabled)
        form.addRow("气泡端口", self.bubble_port)
        form.addRow("气泡秒数", self.bubble_seconds)
        form.addRow("气泡令牌", self.bubble_token)
        layout.addWidget(group)

    def build_route_page(self) -> None:
        _, _, layout = self.page()
        group = QGroupBox("RabiRoute 输出")
        form = QFormLayout(group)
        self.route_enabled = QCheckBox("转写完成后推送到 RabiRoute")
        self.route_url = QLineEdit()
        self.route_source = QLineEdit()
        self.route_token = QLineEdit()
        self.route_token.setEchoMode(QLineEdit.Password)
        form.addRow("", self.route_enabled)
        form.addRow("推送 URL", self.route_url)
        form.addRow("来源 ID", self.route_source)
        form.addRow("访问令牌", self.route_token)
        layout.addWidget(group)

    def spin(self, minimum: float, maximum: float, step: float, decimals: int = 1) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        return widget

    def apply_config_to_ui(self) -> None:
        config = self.config_data
        self.compute_combo.setCurrentText(str(config.get("compute_type", "int8_float16")))
        language_code = str(config.get("language_mode", config.get("language", "zh")))
        self.language_combo.setCurrentText(LANGUAGE_LABELS.get(language_code, "简体中文"))
        self.simplify_check.setChecked(bool(config.get("simplify_chinese", True)))
        self.source_combo.setCurrentText(MODEL_SOURCE_LABELS.get(str(config.get("model_source", "local")), "本地模型"))
        provider_label = API_PROVIDER_LABELS.get(str(config.get("api_provider", "dashscope")), "千问 / DashScope")
        self.api_provider_combo.setCurrentText(provider_label)
        self.input_api_provider_combo.setCurrentText(provider_label)
        self.api_provider_id.setText(str(config.get("api_provider_id", config.get("api_provider", "dashscope"))))
        self.api_enabled_check.setChecked(bool(config.get("api_provider_enabled", False)))
        self.api_base_url.setText(str(config.get("api_base_url", "")))
        self.api_key.setText(str(config.get("api_key", "")))
        self.input_gain.setValue(float(config.get("input_gain", DEFAULT_CONFIG["input_gain"])))
        self.record_threshold.setValue(float(config.get("record_threshold", 0.01)))
        self.transcribe_threshold.setValue(float(config.get("transcribe_threshold", 0.015)))
        self.adaptive_check.setChecked(bool(config.get("adaptive_threshold", True)))
        self.pre_roll.setValue(float(config.get("pre_roll_seconds", 1.5)))
        self.pause_seconds.setValue(float(config.get("transcribe_pause_seconds", 0.5)))
        self.silence_seconds.setValue(float(config.get("silence_seconds", 1.2)))
        self.max_phrase.setValue(float(config.get("max_phrase_seconds", 12.0)))
        self.auto_start_check.setChecked(bool(config.get("auto_start", False)))
        self.cache_retention.setValue(float(config.get("cache_retention_minutes", DEFAULT_CONFIG["cache_retention_minutes"])))
        self.bubble_enabled.setChecked(bool(config.get("reply_bubble_enabled", DEFAULT_CONFIG["reply_bubble_enabled"])))
        self.bubble_port.setValue(int(config.get("reply_bubble_port", DEFAULT_CONFIG["reply_bubble_port"])))
        self.bubble_seconds.setValue(float(config.get("reply_bubble_seconds", DEFAULT_CONFIG["reply_bubble_seconds"])))
        self.bubble_token.setText(str(config.get("reply_bubble_token", "")))
        self.route_enabled.setChecked(bool(config.get("rabiroute_enabled", False)))
        self.route_url.setText(str(config.get("rabiroute_url", DEFAULT_CONFIG["rabiroute_url"])))
        self.route_source.setText(str(config.get("rabiroute_source", DEFAULT_CONFIG["rabiroute_source"])))
        self.route_token.setText(str(config.get("rabiroute_token", "")))
        self.refresh_api_models()
        self.refresh_input_api_models()
        self.api_model_combo.setCurrentText(str(config.get("api_model", self.api_model_combo.currentText())))
        self.input_api_model_combo.setCurrentText(self.api_model_combo.currentText())
        self.refresh_local_models()
        self.refresh_source_visibility()
        self.update_trigger_summary()
        self.update_provider_status()

    def refresh_api_models(self) -> None:
        provider = API_PROVIDER_CODES.get(self.api_provider_combo.currentText(), "dashscope")
        models = API_MODELS_BY_PROVIDER.get(provider, API_MODELS_BY_PROVIDER["dashscope"])
        current = self.api_model_combo.currentText()
        self.api_model_combo.blockSignals(True)
        self.api_model_combo.clear()
        self.api_model_combo.addItems(models)
        self.api_model_combo.setCurrentText(current if current in models else models[0])
        self.api_model_combo.blockSignals(False)
        if provider == "dashscope" and not self.api_base_url.text().strip():
            self.api_base_url.setText("https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.update_provider_status()

    def refresh_input_api_models(self) -> None:
        provider = API_PROVIDER_CODES.get(self.input_api_provider_combo.currentText(), "dashscope")
        models = API_MODELS_BY_PROVIDER.get(provider, API_MODELS_BY_PROVIDER["dashscope"])
        current = self.input_api_model_combo.currentText()
        self.input_api_model_combo.blockSignals(True)
        self.input_api_model_combo.clear()
        self.input_api_model_combo.addItems(models)
        self.input_api_model_combo.setCurrentText(current if current in models else models[0])
        self.input_api_model_combo.blockSignals(False)

    def sync_input_to_provider(self) -> None:
        self.refresh_input_api_models()
        self.api_provider_combo.blockSignals(True)
        self.api_model_combo.blockSignals(True)
        self.api_provider_combo.setCurrentText(self.input_api_provider_combo.currentText())
        self.refresh_api_models()
        self.api_model_combo.setCurrentText(self.input_api_model_combo.currentText())
        self.api_provider_combo.blockSignals(False)
        self.api_model_combo.blockSignals(False)
        self.update_provider_status()

    def sync_provider_to_input(self) -> None:
        self.input_api_provider_combo.blockSignals(True)
        self.input_api_model_combo.blockSignals(True)
        self.input_api_provider_combo.setCurrentText(self.api_provider_combo.currentText())
        self.refresh_input_api_models()
        self.input_api_model_combo.setCurrentText(self.api_model_combo.currentText())
        self.input_api_provider_combo.blockSignals(False)
        self.input_api_model_combo.blockSignals(False)

    def refresh_local_models(self) -> None:
        self.local_model_combo.clear()
        self.local_model_list.clear()
        config = self.collect_config()
        rows = []
        for model_name in DOWNLOADABLE_MODELS:
            try:
                installed = model_is_installed(config, model_name)
            except Exception:
                installed = False
            label = f"{'已下载' if installed else '未下载'} · {model_name} · {'本地可用' if installed else '需要先下载'}"
            rows.append((0 if installed else 1, model_name, label))
        for _order, model_name, label in sorted(rows, key=lambda item: (item[0], item[1])):
            self.local_model_combo.addItem(label, model_name)
            self.local_model_list.addItem(label)
        current_model = str(self.config_data.get("model", DEFAULT_CONFIG["model"]))
        index = self.local_model_combo.findData(current_model)
        if index >= 0:
            self.local_model_combo.setCurrentIndex(index)

    def refresh_source_visibility(self) -> None:
        api_mode = MODEL_SOURCE_CODES.get(self.source_combo.currentText(), "local") == "api"
        self.local_model_combo.setVisible(not api_mode)
        self.input_api_provider_combo.setVisible(api_mode)
        self.input_api_model_combo.setVisible(api_mode)

    def update_provider_status(self) -> None:
        key_state = "Key 已填写" if self.api_key.text().strip() else "Key 未填写"
        enabled = "启用" if self.api_enabled_check.isChecked() else "未启用"
        self.provider_status.setText(f"{self.api_provider_id.text().strip() or '未命名'} / {self.api_provider_combo.currentText()} / {self.api_model_combo.currentText()} / {enabled} / {key_state}")

    def update_trigger_summary(self) -> None:
        record = self.record_threshold.value()
        transcribe = max(record, self.transcribe_threshold.value())
        self.trigger_summary.setText(
            f"增益 {self.input_gain.value():.1f}x / 录音线 {record:.3f} / 转写线 {transcribe:.3f} / "
            f"前置 {self.pre_roll.value():.1f}s / 等待 {self.pause_seconds.value():.1f}s / "
            f"丢弃 {self.silence_seconds.value():.1f}s / 最长 {self.max_phrase.value():.0f}s / "
            f"{'动态阈值' if self.adaptive_check.isChecked() else '固定阈值'}"
        )

    def list_input_devices(self) -> list[tuple[int, str]]:
        devices = []
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) > 0:
                host_api = sd.query_hostapis(device["hostapi"])["name"]
                devices.append((index, f"{index}: {device['name']} ({host_api})"))
        return devices

    def selected_device_index(self) -> int | None:
        return self.mic_combo.currentData()

    def collect_config(self) -> dict:
        config = DEFAULT_CONFIG.copy()
        config.update(load_config(CONFIG_PATH))
        model_name = self.local_model_combo.currentData() or config.get("model", DEFAULT_CONFIG["model"])
        record = self.record_threshold.value()
        config.update(
            {
                "model": model_name,
                "model_source": MODEL_SOURCE_CODES.get(self.source_combo.currentText(), "local"),
                "api_provider": API_PROVIDER_CODES.get(self.api_provider_combo.currentText(), "dashscope"),
                "api_provider_id": self.api_provider_id.text().strip() or API_PROVIDER_CODES.get(self.api_provider_combo.currentText(), "dashscope"),
                "api_provider_enabled": self.api_enabled_check.isChecked(),
                "api_model": self.api_model_combo.currentText(),
                "api_base_url": self.api_base_url.text().strip(),
                "api_key": self.api_key.text().strip(),
                "auto_start": self.auto_start_check.isChecked(),
                "device": "cuda",
                "compute_type": self.compute_combo.currentText(),
                "language": LANGUAGE_CODES.get(self.language_combo.currentText(), "zh"),
                "language_mode": LANGUAGE_CODES.get(self.language_combo.currentText(), "zh"),
                "simplify_chinese": self.simplify_check.isChecked(),
                "adaptive_threshold": self.adaptive_check.isChecked(),
                "input_gain": round(self.input_gain.value(), 1),
                "record_threshold": round(record, 4),
                "transcribe_threshold": round(max(record, self.transcribe_threshold.value()), 4),
                "rms_threshold": round(record, 4),
                "pre_roll_seconds": round(self.pre_roll.value(), 1),
                "transcribe_pause_seconds": round(self.pause_seconds.value(), 1),
                "silence_seconds": round(self.silence_seconds.value(), 2),
                "max_phrase_seconds": round(self.max_phrase.value(), 1),
                "cache_retention_minutes": round(self.cache_retention.value(), 1),
                "mic_device": self.selected_device_index(),
                "reply_bubble_enabled": self.bubble_enabled.isChecked(),
                "reply_bubble_port": self.bubble_port.value(),
                "reply_bubble_seconds": round(self.bubble_seconds.value(), 1),
                "reply_bubble_token": self.bubble_token.text().strip(),
                "rabiroute_enabled": self.route_enabled.isChecked(),
                "rabiroute_url": self.route_url.text().strip(),
                "rabiroute_token": self.route_token.text().strip(),
                "rabiroute_source": self.route_source.text().strip() or "fennenote",
            }
        )
        return config

    def save_config(self) -> None:
        config = self.collect_config()
        write_config(CONFIG_PATH, config)
        self.config_data = config
        self.status_label.setText("配置已保存")
        self.update_provider_status()
        self.refresh_local_models()

    def activate_api_provider(self) -> None:
        self.api_enabled_check.setChecked(True)
        self.source_combo.setCurrentText("API 模型")
        self.save_config()
        self.status_label.setText("已切换到 API Provider；服务层尚未接入")

    def validate_api_provider(self) -> None:
        missing = []
        if not self.api_provider_id.text().strip():
            missing.append("Provider ID")
        if not self.api_model_combo.currentText().strip():
            missing.append("模型")
        if not self.api_base_url.text().strip():
            missing.append("Base URL")
        if not self.api_key.text().strip():
            missing.append("API Key")
        if missing:
            QMessageBox.warning(self, "Provider 配置", "还缺少：" + ", ".join(missing))
        else:
            QMessageBox.information(self, "Provider 配置", "字段完整。连通性测试需要接入 API 服务层。")

    def current_preview_threshold(self) -> float:
        record = self.record_threshold.value()
        if not self.adaptive_check.isChecked():
            return record
        multiplier = float(self.config_data.get("adaptive_threshold_multiplier", DEFAULT_CONFIG.get("adaptive_threshold_multiplier", 2.5)))
        margin = float(self.config_data.get("adaptive_threshold_margin", DEFAULT_CONFIG.get("adaptive_threshold_margin", 0.004)))
        return max(record, self.preview_noise_floor * multiplier + margin)

    def start_level_worker(self) -> None:
        self.stop_level_worker()
        self.level_thread = QThread(self)
        self.level_worker = LevelWorker()
        self.level_worker.configure(self.collect_config(), self.selected_device_index())
        self.level_worker.moveToThread(self.level_thread)
        self.level_thread.started.connect(self.level_worker.run)
        self.level_worker.level.connect(self.on_level)
        self.level_worker.error.connect(self.status_label.setText)
        self.level_thread.start()

    def stop_level_worker(self) -> None:
        if self.level_worker:
            self.level_worker.stop()
        if self.level_thread:
            self.level_thread.quit()
            self.level_thread.wait(1000)
        self.level_worker = None
        self.level_thread = None

    def on_level(self, level: float, raw_level: float) -> None:
        self.display_level = (self.display_level * 0.75) + (level * 0.25)
        self.display_raw_level = (self.display_raw_level * 0.75) + (raw_level * 0.25)
        threshold = self.current_preview_threshold()
        transcribe = max(self.record_threshold.value(), self.transcribe_threshold.value())
        if level < threshold:
            self.preview_noise_floor = (self.preview_noise_floor * 0.95) + (level * 0.05)
        self.level_history.append(self.display_level)
        peak = max(self.level_history, default=self.display_level)
        self.level_text.setText(f"原始 {self.display_raw_level:.3f} / 增益后 {self.display_level:.3f} / 录音 {threshold:.3f} / 转写 {transcribe:.3f} / 峰值 {peak:.3f}")
        self.trigger_state_label.setText("状态：录音中" if level >= threshold else "状态：监听中")
        self.wave.push(self.display_level, threshold, transcribe)

    def start_transcriber(self) -> None:
        config = self.collect_config()
        write_config(CONFIG_PATH, config)
        if config.get("model_source") == "api":
            QMessageBox.information(self, "API 模型", "API/千问配置入口已保存，但转写服务层尚未接入。")
            return
        if self.process_thread and self.process_thread.isRunning():
            return
        self.process_thread = QThread(self)
        self.process_worker = ProcessWorker()
        self.process_worker.configure(config)
        self.process_worker.moveToThread(self.process_thread)
        self.process_thread.started.connect(self.process_worker.run)
        self.process_worker.line.connect(self.on_process_line)
        self.process_worker.exited.connect(self.on_process_exited)
        self.process_thread.start()
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.status_label.setText("正在启动")

    def send_process_command(self, command: str) -> None:
        if self.process_worker:
            self.process_worker.send(command)

    def stop_transcriber(self) -> None:
        if self.process_worker:
            self.process_worker.stop()

    def on_process_line(self, line: str) -> None:
        if line.startswith("FN_STATUS|"):
            parts = line.split("|", 2)
            if len(parts) == 3:
                self.status_label.setText(parts[2])
                self.transcript.appendPlainText(f"系统：{parts[2]}")
            return
        if line.startswith("["):
            self.transcript.appendPlainText(line)
        elif line:
            self.status_label.setText(line[:100])
            self.transcript.appendPlainText(f"系统：{line}")

    def on_process_exited(self, _code: int) -> None:
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止")
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait(1000)
        self.process_thread = None
        self.process_worker = None

    def open_output_folder(self) -> None:
        output_dir = APP_DIR / str(self.collect_config().get("output_dir", "transcripts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def closeEvent(self, event) -> None:
        self.stop_transcriber()
        self.stop_level_worker()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = FenneNoteQt()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
