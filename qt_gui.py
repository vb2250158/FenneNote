from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
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
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from transcribe_mic import (
    DEFAULT_CONFIG,
    DOWNLOADABLE_MODELS,
    MODEL_PROFILES,
    configure_local_storage,
    delete_model_cache,
    download_configured_model,
    load_config,
    model_cache_root,
    model_is_installed,
    save_config as write_config,
)


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
ICON_PATH = APP_DIR / "assets" / "fennenote.ico"
BUDDY_STATE_IMAGE_PATHS = {
    "idle": APP_DIR / "assets" / "fennenote-state-idle.png",
    "listening": APP_DIR / "assets" / "fennenote-state-listening.png",
    "writing": APP_DIR / "assets" / "fennenote-state-writing.png",
}
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
API_MODEL_OPTIONS_BY_PROVIDER = {
    "dashscope": [
        {
            "id": "qwen3-asr-flash",
            "label": "推荐 ASR · qwen3-asr-flash · 多语种 · ¥0.00022/秒起",
            "tier": "专用语音识别 / 生产优先",
            "best_for": "持续录音、会议记录、普通话/方言/多语种转写。",
            "price": "北京约 ¥0.00022/秒；新加坡约 ¥0.00026/秒；按音频时长计费。",
            "note": "官方推荐的新 ASR。只做转写时通常比 Omni 更直接、更好估算成本。",
        },
        {
            "id": "qwen-audio-asr",
            "label": "旧 ASR · qwen-audio-asr · 中英识别 · 免费体验",
            "tier": "旧版专用语音识别 / Beta",
            "best_for": "中文、英文短音频转写；想沿用旧接口时使用。",
            "price": "目前仅供免费体验；额度用完后不可调用，官方推荐迁移 Qwen3 ASR。",
            "note": "支持语言较少，不建议作为新的生产默认项。",
        },
        {
            "id": "qwen-audio-turbo",
            "label": "音频理解 · qwen-audio-turbo · 可问答 · 免费体验",
            "tier": "音频理解 / 对话模型",
            "best_for": "让模型理解音频内容、总结、回答“这段音频在说什么”。",
            "price": "目前仅供免费体验；音频约 25 token/秒；额度用完后推荐 Qwen-Omni。",
            "note": "不是纯 ASR，转写准确率和长音频能力通常不如专用 ASR。",
        },
        {
            "id": "qwen2.5-omni-7b",
            "label": "多模态 · qwen2.5-omni-7b · 音频/图像/视频 · ¥38/百万音频 token",
            "tier": "多模态理解 + 文本/语音输出",
            "best_for": "需要同一个模型处理文字、图片、视频、音频，或后续做语音对话。",
            "price": "国内约：文本输入 ¥0.6/M、音频输入 ¥38/M、视觉输入 ¥2/M、文本输出 ¥2.4-6/M、音频输出 ¥76/M。",
            "note": "能力更宽，但只做语音转文字时成本和复杂度通常更高。",
        },
    ],
    "openai_compatible": [
        {
            "id": "gpt-4o-transcribe",
            "label": "高准确率 · gpt-4o-transcribe · 新转写 · $6/百万音频 token",
            "tier": "OpenAI 新一代转写",
            "best_for": "对准确率、语言识别和复杂音频更敏感的转写任务。",
            "price": "官方列价：音频输入 $6/M token；文本输入 $2.5/M，输出 $10/M。",
            "note": "OpenAI 官方兼容；第三方 OpenAI Compatible 服务价格可能不同。",
        },
        {
            "id": "whisper-1",
            "label": "兼容便宜 · whisper-1 · 老牌通用 · $0.006/分钟",
            "tier": "Whisper 兼容 / 通用转写",
            "best_for": "成本稳定、兼容性优先、对最新准确率要求不高的任务。",
            "price": "OpenAI 官方列价：$0.006/分钟。",
            "note": "老模型，生态成熟；准确率通常不如 gpt-4o-transcribe。",
        },
    ],
}
API_MODELS_BY_PROVIDER = {
    provider: [option["id"] for option in options]
    for provider, options in API_MODEL_OPTIONS_BY_PROVIDER.items()
}
THEME = {
    "app_bg": "#fff7ef",
    "panel": "#fffdf8",
    "panel_tint": "#fff4df",
    "panel_soft": "#fffaf2",
    "nav": "#243437",
    "nav_hover": "#315052",
    "nav_text": "#fff8ec",
    "nav_muted": "#d7c9b7",
    "ink": "#342b24",
    "muted": "#7b6b5d",
    "line": "#efd9bd",
    "line_strong": "#e2bd8c",
    "sand": "#f3a43b",
    "sand_dark": "#9c6412",
    "teal": "#22a99f",
    "teal_dark": "#0f6e6b",
    "teal_soft": "#e4f7f3",
    "rose": "#ef7f8f",
    "rose_soft": "#ffe6e7",
    "peach": "#ffd7a6",
    "green": "#20b874",
    "danger": "#d45b4c",
    "canvas": "#fffaf2",
    "quiet_bar": "#6a5a52",
}


def api_model_options(provider: str) -> list[dict[str, str]]:
    return API_MODEL_OPTIONS_BY_PROVIDER.get(provider, API_MODEL_OPTIONS_BY_PROVIDER["dashscope"])


def api_model_option(provider: str, model_id: str) -> dict[str, str]:
    options = api_model_options(provider)
    for option in options:
        if option["id"] == model_id:
            return option
    return options[0]


def current_api_model_id(combo: QComboBox) -> str:
    value = combo.currentData()
    return str(value or combo.currentText()).strip()


def set_api_model_id(combo: QComboBox, model_id: str) -> None:
    index = combo.findData(model_id)
    if index >= 0:
        combo.setCurrentIndex(index)
        return
    fallback = combo.findText(model_id)
    if fallback >= 0:
        combo.setCurrentIndex(fallback)


def api_model_detail_text(provider: str, model_id: str) -> str:
    option = api_model_option(provider, model_id)
    return (
        f"层级：{option['tier']}\n"
        f"优势：{option['best_for']}\n"
        f"价格：{option['price']}\n"
        f"备注：{option['note']}"
    )


def populate_api_model_combo(combo: QComboBox, provider: str, current_model_id: str) -> None:
    combo.clear()
    for option in api_model_options(provider):
        combo.addItem(option["label"], option["id"])
        combo.setItemData(combo.count() - 1, api_model_detail_text(provider, option["id"]), Qt.ToolTipRole)
    set_api_model_id(combo, current_model_id)
    if combo.currentIndex() < 0 and combo.count():
        combo.setCurrentIndex(0)


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
        if getattr(sys, "frozen", False):
            command = [str(python_exe), "--transcriber", "--config", str(CONFIG_PATH)]
        else:
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


class ModelOperationWorker(QObject):
    status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, operation: str, model_name: str, config: dict) -> None:
        super().__init__()
        self.operation = operation
        self.model_name = model_name
        self.config = config.copy()

    def run(self) -> None:
        def status_callback(_code: str, message: str) -> None:
            self.status.emit(message)

        try:
            if self.operation == "download":
                download_configured_model(self.config, status_callback=status_callback)
                self.finished.emit(True, f"模型已安装：{self.model_name}")
            elif self.operation == "delete":
                delete_model_cache(self.config, self.model_name, status_callback=status_callback)
                self.finished.emit(True, f"模型已删除：{self.model_name}")
            else:
                self.finished.emit(False, f"未知模型操作：{self.operation}")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class NumericSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, decimals: int = 1, suffix: str = "") -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.decimals = decimals
        self.suffix = suffix
        steps = max(1, int(round((maximum - minimum) / step)))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, steps // 10))
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(72)
        self.value_label.setStyleSheet("font-family: Consolas, 'Microsoft YaHei UI'; color: #342b24; font-weight: 700;")
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.setValue(minimum)

    def value(self) -> float:
        raw = self.minimum + self.slider.value() * self.step
        return min(self.maximum, max(self.minimum, raw))

    def setValue(self, value: float) -> None:
        bounded = min(self.maximum, max(self.minimum, float(value)))
        index = int(round((bounded - self.minimum) / self.step))
        self.slider.setValue(index)
        self.update_label()

    def on_slider_changed(self, _value: int) -> None:
        self.update_label()
        self.valueChanged.emit(self.value())

    def update_label(self) -> None:
        self.value_label.setText(f"{self.value():.{self.decimals}f}{self.suffix}")


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
        painter.fillRect(rect, QColor(THEME["canvas"]))
        painter.setPen(QPen(QColor("#f1dfc7"), 1))
        for index in range(1, 4):
            y = rect.height() * index / 4
            painter.drawLine(0, y, rect.width(), y)
        record_y = rect.height() - min(self.record_threshold / self.scale, 1.0) * rect.height()
        transcribe_y = rect.height() - min(self.transcribe_threshold / self.scale, 1.0) * rect.height()
        painter.setPen(QPen(QColor(THEME["danger"]), 2))
        painter.drawLine(0, record_y, rect.width(), record_y)
        painter.setPen(QPen(QColor(THEME["sand"]), 2, Qt.DashLine))
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
            color = QColor(THEME["quiet_bar"])
            if value >= self.record_threshold:
                color = QColor(THEME["teal"])
            if value >= self.transcribe_threshold:
                color = QColor(THEME["green"])
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
        self.running_config: dict | None = None
        self.model_thread: QThread | None = None
        self.model_worker: ModelOperationWorker | None = None
        self.model_operation_running = False
        self.refreshing_local_models = False
        self.model_status_labels: dict[str, QLabel] = {}
        self.model_select_buttons: dict[str, QPushButton] = {}
        self.model_download_buttons: dict[str, QPushButton] = {}
        self.model_delete_buttons: dict[str, QPushButton] = {}
        self.preview_noise_floor = 0.003
        self.display_level = 0.0
        self.display_raw_level = 0.0
        self.level_history: deque[float] = deque(maxlen=160)
        self.buddy_state = ""
        self.writing_until = 0.0
        self.buddy_pixmaps = self.load_buddy_pixmaps()
        self.buddy_timer = QTimer(self)
        self.buddy_timer.timeout.connect(self.refresh_buddy_state)
        self.buddy_timer.start(240)
        self.build_ui()
        self.set_buddy_state("idle")
        self.apply_config_to_ui()
        self.start_level_worker()

    def load_buddy_pixmaps(self) -> dict[str, QPixmap]:
        pixmaps: dict[str, QPixmap] = {}
        for state, path in BUDDY_STATE_IMAGE_PATHS.items():
            if not path.exists():
                continue
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                pixmaps[state] = pixmap
        return pixmaps

    def set_buddy_state(self, state: str) -> None:
        if state not in self.buddy_pixmaps:
            state = "listening" if "listening" in self.buddy_pixmaps else next(iter(self.buddy_pixmaps), "")
        if not state or state == self.buddy_state:
            return
        self.buddy_state = state
        self.update_buddy_view()

    def update_buddy_view(self) -> None:
        if not hasattr(self, "buddy_image"):
            return
        pixmap = self.buddy_pixmaps.get(self.buddy_state)
        if not pixmap:
            self.buddy_image.clear()
            return
        self.buddy_image.setPixmap(pixmap.scaled(168, 168, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        labels = {
            "idle": "芬妮在打瞌睡，等你说话",
            "listening": "芬妮正在听",
            "writing": "芬妮正在记录",
        }
        self.buddy_caption.setText(labels.get(self.buddy_state, "芬妮待命中"))

    def refresh_buddy_state(self) -> None:
        if time.monotonic() < self.writing_until:
            self.set_buddy_state("writing" if int(time.monotonic() * 3) % 2 == 0 else "listening")
        elif self.buddy_state == "writing":
            self.set_buddy_state("listening" if self.display_level >= self.current_preview_threshold() else "idle")

    def build_ui(self) -> None:
        self.setWindowTitle("FenneNote - Qt Control Console")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 760)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {THEME["app_bg"]};
                color: {THEME["ink"]};
                font-family: "Microsoft YaHei UI";
                font-size: 9.5pt;
            }}
            #nav {{
                background: {THEME["nav"]};
                border-right: 1px solid #1c292b;
            }}
            #nav QLabel {{ color: {THEME["nav_text"]}; background: transparent; }}
            #navSub {{ color: {THEME["nav_muted"]}; }}
            QListWidget {{
                background: {THEME["nav"]};
                color: {THEME["nav_text"]};
                border: none;
                outline: 0;
            }}
            QListWidget::item {{
                padding: 12px 14px;
                border-radius: 8px;
                margin: 4px 8px;
            }}
            QListWidget::item:hover {{ background: {THEME["nav_hover"]}; }}
            QListWidget::item:selected {{
                background: {THEME["sand"]};
                color: #2b2018;
            }}
            QFrame#card, QFrame#heroCard, QFrame#buddyBubble, QFrame#compactOverview, QGroupBox {{
                background: {THEME["panel"]};
                border: 1px solid {THEME["line"]};
                border-radius: 8px;
            }}
            QFrame#heroCard {{
                background: {THEME["panel_tint"]};
                border-color: {THEME["line_strong"]};
            }}
            QFrame#compactOverview {{
                background: rgba(255, 253, 248, 180);
                border-color: {THEME["peach"]};
            }}
            QFrame#buddyBubble {{
                background: {THEME["panel_soft"]};
                border-color: {THEME["peach"]};
            }}
            QFrame#logDrawer {{
                background: {THEME["panel"]};
                border: 1px solid {THEME["line_strong"]};
                border-radius: 8px;
            }}
            QPushButton#logToggle {{
                background: {THEME["sand"]};
                color: #2b2018;
                border-color: {THEME["sand_dark"]};
                font-weight: 800;
                padding: 6px 8px;
            }}
            QPushButton#logToggle:hover {{
                background: {THEME["peach"]};
            }}
            QPlainTextEdit#logText {{
                font-family: Consolas, "Microsoft YaHei UI";
                font-size: 9pt;
            }}
            QGroupBox {{
                margin-top: 12px;
                padding: 16px 12px 12px 12px;
                font-weight: 700;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {THEME["sand_dark"]};
            }}
            QLabel#heroTitle {{
                color: {THEME["sand_dark"]};
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#buddyCaption {{
                color: {THEME["teal_dark"]};
                font-weight: 700;
            }}
            QPushButton {{
                background: #fffaf4;
                border: 1px solid {THEME["line_strong"]};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QPushButton:hover {{ background: #ffefd6; }}
            QPushButton#primary {{
                background: {THEME["teal"]};
                color: white;
                border-color: {THEME["teal_dark"]};
                font-weight: 700;
            }}
            QPushButton#danger {{
                color: {THEME["danger"]};
                background: #fff0ec;
                border-color: #efb4aa;
            }}
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: #fffdf8;
                border: 1px solid {THEME["line"]};
                border-radius: 7px;
                padding: 6px;
            }}
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {THEME["sand"]};
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: #f3dfc4;
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: {THEME["teal"]};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {THEME["sand"]};
                border: 1px solid {THEME["sand_dark"]};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {THEME["peach"]};
            }}
            QPlainTextEdit {{
                background: {THEME["canvas"]};
                border: 1px solid {THEME["line"]};
                border-radius: 8px;
                padding: 10px;
            }}
            """
        )

        root = QWidget()
        self.root_widget = root
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
        subtitle = QLabel("Fennec Listen Desk")
        subtitle.setObjectName("navSub")
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
        self.build_log_drawer(root)

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
        self.log_button = QPushButton("日志")
        self.log_button.setAccessibleName("打开或收起运行日志")
        for button in (self.start_button, self.save_button, self.folder_button, self.log_button):
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

        self.start_button.clicked.connect(self.toggle_transcriber)
        self.pause_button.clicked.connect(lambda: self.send_process_command("p\n"))
        self.stop_button.clicked.connect(self.stop_transcriber)
        self.save_button.clicked.connect(self.save_config)
        self.folder_button.clicked.connect(self.open_output_folder)
        self.log_button.clicked.connect(self.toggle_log_drawer)
        self.pause_button.setEnabled(False)
        self.pause_button.setVisible(False)
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)

    def build_log_drawer(self, parent: QWidget) -> None:
        self.log_drawer_expanded = False
        self.log_drawer = QFrame(parent)
        self.log_drawer.setObjectName("logDrawer")
        shadow = QGraphicsDropShadowEffect(self.log_drawer)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(96, 73, 45, 58))
        self.log_drawer.setGraphicsEffect(shadow)

        drawer_layout = QVBoxLayout(self.log_drawer)
        drawer_layout.setContentsMargins(12, 12, 12, 12)
        drawer_layout.setSpacing(10)

        log_header = QHBoxLayout()
        self.log_title = QLabel("运行日志")
        self.log_title.setStyleSheet("font-weight: 800;")
        self.clear_log_button = QPushButton("清空")
        self.clear_log_button.setAccessibleName("清空运行日志")
        self.clear_log_button.clicked.connect(lambda: self.log_text.clear())
        self.collapse_log_button = QPushButton("收起")
        self.collapse_log_button.setAccessibleName("收起运行日志")
        self.collapse_log_button.clicked.connect(lambda: self.set_log_drawer_expanded(False))
        log_header.addWidget(self.log_title, 1)
        log_header.addWidget(self.clear_log_button)
        log_header.addWidget(self.collapse_log_button)
        drawer_layout.addLayout(log_header)

        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("暂无日志")
        drawer_layout.addWidget(self.log_text, 1)

        self.set_log_drawer_expanded(False)

    def toggle_log_drawer(self) -> None:
        self.set_log_drawer_expanded(not self.log_drawer_expanded)

    def set_log_drawer_expanded(self, expanded: bool) -> None:
        self.log_drawer_expanded = expanded
        self.log_drawer.setVisible(expanded)
        if hasattr(self, "log_button"):
            self.log_button.setText("收起日志" if expanded else "日志")
        self.update_log_drawer_geometry()

    def update_log_drawer_geometry(self) -> None:
        if not hasattr(self, "log_drawer") or not self.log_drawer_expanded:
            return
        rect = self.root_widget.rect()
        margin = 14
        width = min(420, max(340, rect.width() // 3))
        height = max(320, rect.height() - 104)
        x = rect.width() - width - margin
        y = 76
        self.log_drawer.setGeometry(x, y, width, height)
        self.log_drawer.raise_()

    def append_log(self, message: str) -> None:
        if not message or not hasattr(self, "log_text"):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
        hero.setObjectName("heroCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_layout.setSpacing(18)
        hero_text = QVBoxLayout()
        title = QLabel("Fenne Control Room")
        title.setObjectName("heroTitle")
        hero_text.addWidget(title)
        hero_subtitle = QLabel("芬妮会陪你守住听写现场：打瞌睡、认真听、开心记录。")
        hero_subtitle.setWordWrap(True)
        hero_subtitle.setStyleSheet(f"color: {THEME['muted']};")
        hero_text.addWidget(hero_subtitle)

        model_overview = QFrame()
        model_overview.setObjectName("compactOverview")
        model_overview_layout = QGridLayout(model_overview)
        model_overview_layout.setContentsMargins(10, 8, 10, 8)
        model_overview_layout.setHorizontalSpacing(10)
        model_overview_layout.setVerticalSpacing(4)
        model_overview_layout.setColumnStretch(1, 1)
        self.overview_source_label = QLabel("来源")
        self.overview_model_label = QLabel("模型")
        self.overview_state_label = QLabel("状态")
        self.overview_detail_label = QLabel()
        for label in (self.overview_source_label, self.overview_model_label, self.overview_state_label):
            label.setStyleSheet("font-size: 14px; font-weight: 800;")
        self.overview_detail_label.setWordWrap(True)
        self.overview_detail_label.setStyleSheet(f"color: {THEME['muted']};")
        model_overview_layout.addWidget(QLabel("模型来源"), 0, 0)
        model_overview_layout.addWidget(self.overview_source_label, 0, 1)
        model_overview_layout.addWidget(QLabel("启动模型"), 1, 0)
        model_overview_layout.addWidget(self.overview_model_label, 1, 1)
        model_overview_layout.addWidget(QLabel("当前状态"), 2, 0)
        model_overview_layout.addWidget(self.overview_state_label, 2, 1)
        model_overview_layout.addWidget(self.overview_detail_label, 3, 0, 1, 2)
        hero_text.addWidget(model_overview)

        self.trigger_state_label = QLabel("状态：监听中")
        self.trigger_state_label.setStyleSheet(f"color: {THEME['teal_dark']}; font-weight: 700;")
        hero_text.addWidget(self.trigger_state_label)
        hero_text.addStretch(1)
        hero_layout.addLayout(hero_text, 1)

        buddy_card = QFrame()
        buddy_card.setObjectName("buddyBubble")
        buddy_layout = QVBoxLayout(buddy_card)
        buddy_layout.setContentsMargins(14, 10, 14, 10)
        self.buddy_image = QLabel()
        self.buddy_image.setAlignment(Qt.AlignCenter)
        self.buddy_image.setMinimumSize(172, 138)
        self.buddy_caption = QLabel("芬妮待命中")
        self.buddy_caption.setObjectName("buddyCaption")
        self.buddy_caption.setAlignment(Qt.AlignCenter)
        buddy_layout.addWidget(self.buddy_image)
        buddy_layout.addWidget(self.buddy_caption)
        hero_layout.addWidget(buddy_card, 0)
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
        self.input_api_model_combo.setMinimumContentsLength(44)
        self.input_api_model_detail = QLabel()
        self.input_api_model_detail.setWordWrap(True)
        self.input_api_model_detail.setStyleSheet("color: #6f5962;")
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
        form.addRow("模型说明", self.input_api_model_detail)
        form.addRow("计算精度", self.compute_combo)
        form.addRow("语言", self.language_combo)
        form.addRow("", self.simplify_check)
        layout.addWidget(group)
        self.source_combo.currentTextChanged.connect(self.refresh_source_visibility)
        self.local_model_combo.currentIndexChanged.connect(self.on_local_model_selected)
        self.input_api_provider_combo.currentTextChanged.connect(self.refresh_input_api_models)
        self.input_api_model_combo.currentIndexChanged.connect(self.update_input_api_model_detail)

    def build_model_page(self) -> None:
        _, _, layout = self.page()
        provider = QGroupBox("API Provider 配置")
        form = QFormLayout(provider)
        self.api_enabled_check = QCheckBox("启用 API Provider")
        self.api_provider_id = QLineEdit()
        self.api_provider_combo = QComboBox()
        self.api_provider_combo.addItems(list(API_PROVIDER_CODES.keys()))
        self.api_model_combo = QComboBox()
        self.api_model_combo.setMinimumContentsLength(44)
        self.api_model_detail = QLabel()
        self.api_model_detail.setWordWrap(True)
        self.api_model_detail.setStyleSheet("color: #6f5962;")
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
        form.addRow("模型说明", self.api_model_detail)
        form.addRow("Base URL", self.api_base_url)
        form.addRow("API Key", self.api_key)
        form.addRow("状态", self.provider_status)
        form.addRow("", actions)
        layout.addWidget(provider)

        local = QGroupBox("本地模型缓存")
        local_layout = QVBoxLayout(local)
        self.model_cache_status = QLabel("模型缓存：检查中")
        self.model_cache_status.setWordWrap(True)
        self.model_cache_status.setStyleSheet(f"color: {THEME['teal_dark']}; font-weight: 700;")
        local_layout.addWidget(self.model_cache_status)
        self.local_model_list = QVBoxLayout()
        self.local_model_list.setSpacing(8)
        local_layout.addLayout(self.local_model_list)
        for model_name in DOWNLOADABLE_MODELS:
            row = QFrame()
            row.setObjectName("card")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            info = QVBoxLayout()
            profile = MODEL_PROFILES[model_name]
            name_label = QLabel(model_name)
            name_label.setStyleSheet("font-weight: 800;")
            detail_label = QLabel(f"{profile['parameters']} / {profile['required_vram']} / {profile['relative_speed']} · {profile['description']}")
            detail_label.setWordWrap(True)
            detail_label.setStyleSheet(f"color: {THEME['muted']};")
            status_label = QLabel("检查中")
            status_label.setStyleSheet(f"color: {THEME['teal_dark']};")
            info.addWidget(name_label)
            info.addWidget(detail_label)
            info.addWidget(status_label)
            row_layout.addLayout(info, 1)
            select_button = QPushButton("选择")
            download_button = QPushButton("下载")
            delete_button = QPushButton("删除")
            delete_button.setObjectName("danger")
            select_button.clicked.connect(lambda _checked=False, name=model_name: self.select_local_model(name))
            download_button.clicked.connect(lambda _checked=False, name=model_name: self.install_model(name))
            delete_button.clicked.connect(lambda _checked=False, name=model_name: self.delete_model(name))
            row_layout.addWidget(select_button)
            row_layout.addWidget(download_button)
            row_layout.addWidget(delete_button)
            self.model_status_labels[model_name] = status_label
            self.model_select_buttons[model_name] = select_button
            self.model_download_buttons[model_name] = download_button
            self.model_delete_buttons[model_name] = delete_button
            self.local_model_list.addWidget(row)
        layout.addWidget(local, 1)
        self.activate_api_button.clicked.connect(self.activate_api_provider)
        self.validate_api_button.clicked.connect(self.validate_api_provider)
        self.api_provider_combo.currentTextChanged.connect(self.refresh_api_models)
        self.api_provider_combo.currentTextChanged.connect(self.sync_provider_to_input)
        self.api_model_combo.currentTextChanged.connect(self.sync_provider_to_input)
        self.api_model_combo.currentIndexChanged.connect(self.update_api_model_detail)
        self.input_api_provider_combo.currentTextChanged.connect(self.sync_input_to_provider)
        self.input_api_model_combo.currentTextChanged.connect(self.sync_input_to_provider)
        self.api_key.textChanged.connect(self.update_model_overview)

    def build_trigger_page(self) -> None:
        _, _, layout = self.page()
        summary = QGroupBox("当前生效参数")
        summary_layout = QVBoxLayout(summary)
        self.trigger_wave = WaveWidget()
        summary_layout.addWidget(self.trigger_wave)
        self.trigger_summary = QLabel()
        self.trigger_summary.setWordWrap(True)
        summary_layout.addWidget(self.trigger_summary)
        layout.addWidget(summary)
        group = QGroupBox("阈值与分段")
        form = QFormLayout(group)
        self.adaptive_check = QCheckBox("动态识别环境底噪")
        self.input_gain = self.slider(1.0, 5.0, 0.1, 1, "x")
        self.record_threshold = self.slider(0.01, 0.04, 0.001, 3)
        self.transcribe_threshold = self.slider(0.01, 0.06, 0.001, 3)
        self.pre_roll = self.slider(0.0, 3.0, 0.1, 1, "s")
        self.min_phrase = self.slider(0.2, 10.0, 0.1, 1, "s")
        self.pause_seconds = self.slider(0.2, 2.0, 0.1, 1, "s")
        self.silence_seconds = self.slider(1.0, 8.0, 0.1, 1, "s")
        self.max_phrase = self.slider(10.0, 120.0, 1.0, 0, "s")
        for label, widget in (
            ("触发模式", self.adaptive_check),
            ("麦克风增益", self.input_gain),
            ("录音触发阈值", self.record_threshold),
            ("转写判定阈值", self.transcribe_threshold),
            ("触发前保留秒数", self.pre_roll),
            ("最短有效录音秒数", self.min_phrase),
            ("低于转写线等待秒数", self.pause_seconds),
            ("噪声丢弃等待秒数", self.silence_seconds),
            ("最长分段秒数", self.max_phrase),
        ):
            form.addRow(label, widget)
        layout.addWidget(group)
        for widget in (self.input_gain, self.record_threshold, self.transcribe_threshold, self.pre_roll, self.min_phrase, self.pause_seconds, self.silence_seconds, self.max_phrase):
            widget.valueChanged.connect(self.update_trigger_summary)
        self.adaptive_check.stateChanged.connect(self.update_trigger_summary)

    def build_app_page(self) -> None:
        _, _, layout = self.page()
        group = QGroupBox("应用设置")
        form = QFormLayout(group)
        self.auto_start_check = QCheckBox("启动后自动开始")
        self.cache_retention = self.slider(0.0, 60.0, 1.0, 0, " 分钟")
        self.bubble_enabled = QCheckBox("启用左下角气泡")
        self.bubble_port = self.slider(1024.0, 65535.0, 1.0, 0)
        self.bubble_seconds = self.slider(1.0, 10.0, 0.5, 1, "s")
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
        self.route_test_button = QPushButton("测试连接")
        self.route_status = QLabel("未测试")
        self.route_status.setWordWrap(True)
        self.route_status.setStyleSheet(f"color: {THEME['muted']};")
        form.addRow("", self.route_enabled)
        form.addRow("推送 URL", self.route_url)
        form.addRow("来源 ID", self.route_source)
        form.addRow("访问令牌", self.route_token)
        form.addRow("连接测试", self.route_test_button)
        form.addRow("状态", self.route_status)
        layout.addWidget(group)
        self.route_test_button.clicked.connect(self.test_rabiroute_connection)

    def slider(self, minimum: float, maximum: float, step: float, decimals: int = 1, suffix: str = "") -> NumericSlider:
        return NumericSlider(minimum, maximum, step, decimals, suffix)

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
        self.min_phrase.setValue(float(config.get("min_phrase_seconds", DEFAULT_CONFIG["min_phrase_seconds"])))
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
        self.set_selected_api_model(str(config.get("api_model", current_api_model_id(self.api_model_combo))))
        self.refresh_local_models()
        self.refresh_source_visibility()
        self.update_trigger_summary()
        self.update_provider_status()

    def selected_api_provider_code(self) -> str:
        return API_PROVIDER_CODES.get(self.api_provider_combo.currentText(), "dashscope")

    def selected_input_api_provider_code(self) -> str:
        return API_PROVIDER_CODES.get(self.input_api_provider_combo.currentText(), "dashscope")

    def selected_api_model_id(self) -> str:
        return current_api_model_id(self.api_model_combo)

    def selected_input_api_model_id(self) -> str:
        return current_api_model_id(self.input_api_model_combo)

    def set_selected_api_model(self, model_id: str) -> None:
        set_api_model_id(self.api_model_combo, model_id)
        set_api_model_id(self.input_api_model_combo, self.selected_api_model_id())
        self.update_api_model_detail()
        self.update_input_api_model_detail()

    def refresh_api_models(self) -> None:
        provider = self.selected_api_provider_code()
        models = API_MODELS_BY_PROVIDER.get(provider, API_MODELS_BY_PROVIDER["dashscope"])
        current = self.selected_api_model_id()
        self.api_model_combo.blockSignals(True)
        populate_api_model_combo(self.api_model_combo, provider, current if current in models else models[0])
        self.api_model_combo.blockSignals(False)
        if provider == "dashscope" and not self.api_base_url.text().strip():
            self.api_base_url.setText("https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.update_api_model_detail()
        self.update_provider_status()

    def refresh_input_api_models(self) -> None:
        provider = self.selected_input_api_provider_code()
        models = API_MODELS_BY_PROVIDER.get(provider, API_MODELS_BY_PROVIDER["dashscope"])
        current = self.selected_input_api_model_id()
        self.input_api_model_combo.blockSignals(True)
        populate_api_model_combo(self.input_api_model_combo, provider, current if current in models else models[0])
        self.input_api_model_combo.blockSignals(False)
        self.update_input_api_model_detail()

    def sync_input_to_provider(self) -> None:
        self.refresh_input_api_models()
        self.api_provider_combo.blockSignals(True)
        self.api_model_combo.blockSignals(True)
        self.api_provider_combo.setCurrentText(self.input_api_provider_combo.currentText())
        self.refresh_api_models()
        set_api_model_id(self.api_model_combo, self.selected_input_api_model_id())
        self.api_provider_combo.blockSignals(False)
        self.api_model_combo.blockSignals(False)
        self.update_api_model_detail()
        self.update_provider_status()

    def sync_provider_to_input(self) -> None:
        self.input_api_provider_combo.blockSignals(True)
        self.input_api_model_combo.blockSignals(True)
        self.input_api_provider_combo.setCurrentText(self.api_provider_combo.currentText())
        self.refresh_input_api_models()
        set_api_model_id(self.input_api_model_combo, self.selected_api_model_id())
        self.input_api_provider_combo.blockSignals(False)
        self.input_api_model_combo.blockSignals(False)
        self.update_input_api_model_detail()

    def update_api_model_detail(self) -> None:
        provider = self.selected_api_provider_code()
        model_id = self.selected_api_model_id()
        self.api_model_detail.setText(api_model_detail_text(provider, model_id))
        self.api_model_combo.setToolTip(api_model_detail_text(provider, model_id))
        self.update_model_overview()

    def update_input_api_model_detail(self) -> None:
        provider = self.selected_input_api_provider_code()
        model_id = self.selected_input_api_model_id()
        self.input_api_model_detail.setText(api_model_detail_text(provider, model_id))
        self.input_api_model_combo.setToolTip(api_model_detail_text(provider, model_id))
        self.update_model_overview()

    def refresh_local_models(self) -> None:
        if self.refreshing_local_models:
            return
        self.refreshing_local_models = True
        current_model = self.selected_local_model_id()
        fallback_model = str(self.config_data.get("model", DEFAULT_CONFIG["model"]))
        if not current_model:
            current_model = fallback_model
        self.local_model_combo.blockSignals(True)
        self.local_model_combo.clear()
        config = self.collect_config()
        rows = []
        for model_name in DOWNLOADABLE_MODELS:
            try:
                installed = model_is_installed(config, model_name)
                cache_root = model_cache_root(config, model_name)
            except Exception:
                installed = False
                cache_root = None
            is_current = model_name == current_model
            label = f"{'已下载' if installed else '未下载'} · {model_name} · {'本地可用' if installed else '需要先下载'}"
            rows.append((0 if installed else 1, model_name, label))
            state_text = "当前 / 已下载" if is_current and installed else "当前 / 未下载" if is_current else "已下载" if installed else "未下载"
            if cache_root is not None and installed:
                state_text = f"{state_text} · {cache_root.name}"
            status_label = self.model_status_labels.get(model_name)
            if status_label is not None:
                status_label.setText(state_text)
            select_button = self.model_select_buttons.get(model_name)
            download_button = self.model_download_buttons.get(model_name)
            delete_button = self.model_delete_buttons.get(model_name)
            if select_button is not None:
                select_button.setEnabled(not is_current and not self.model_operation_running)
            if download_button is not None:
                download_button.setText("检查" if installed else "下载")
                download_button.setEnabled(not self.model_operation_running)
            if delete_button is not None:
                delete_button.setEnabled(installed and not self.model_operation_running)
        for _order, model_name, label in sorted(rows, key=lambda item: (item[0], item[1])):
            self.local_model_combo.addItem(label, model_name)
        index = self.local_model_combo.findData(current_model)
        if index >= 0:
            self.local_model_combo.setCurrentIndex(index)
        self.local_model_combo.blockSignals(False)
        current_installed = self.local_model_installed(current_model, config)
        self.model_cache_status.setText(f"{current_model}：{'已下载，可开始转写' if current_installed else '未下载，请先下载模型'}")
        self.refreshing_local_models = False
        self.update_start_button_state()

    def selected_local_model_id(self) -> str:
        value = self.local_model_combo.currentData()
        return str(value or self.config_data.get("model", DEFAULT_CONFIG["model"])).strip()

    def local_model_installed(self, model_name: str, config: dict | None = None) -> bool:
        check_config = config or self.collect_config()
        if check_config.get("model_source") == "api":
            return True
        try:
            return model_is_installed(check_config, model_name)
        except Exception:
            return False

    def local_model_mode(self) -> bool:
        return MODEL_SOURCE_CODES.get(self.source_combo.currentText(), "local") == "local"

    def update_start_button_state(self) -> None:
        if not hasattr(self, "start_button"):
            return
        if self.transcriber_running():
            self.set_start_button_mode(running=True)
            self.start_button.setEnabled(True)
            self.update_model_overview()
            return
        self.set_start_button_mode(running=False)
        if self.local_model_mode():
            self.start_button.setEnabled(self.local_model_installed(self.selected_local_model_id()) and not self.model_operation_running)
        else:
            self.start_button.setEnabled(not self.model_operation_running)
        self.update_model_overview()

    def transcriber_running(self) -> bool:
        return bool(self.process_thread and self.process_thread.isRunning())

    def set_start_button_mode(self, running: bool) -> None:
        text = "停止" if running else "开始"
        object_name = "danger" if running else "primary"
        if self.start_button.text() == text and self.start_button.objectName() == object_name:
            return
        self.start_button.setText(text)
        self.start_button.setObjectName(object_name)
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.start_button.update()

    def on_local_model_selected(self) -> None:
        if self.refreshing_local_models:
            return
        model_name = self.selected_local_model_id()
        self.config_data["model"] = model_name
        self.update_start_button_state()
        self.refresh_local_models()
        if self.local_model_installed(model_name):
            return
        answer = QMessageBox.question(
            self,
            "下载模型",
            f"模型 {model_name} 还没有下载。\n\n是否现在下载这个模型？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.install_model(model_name)

    def select_local_model(self, model_name: str) -> None:
        index = self.local_model_combo.findData(model_name)
        if index >= 0:
            self.local_model_combo.setCurrentIndex(index)
        self.config_data["model"] = model_name
        self.save_config()
        if not self.local_model_installed(model_name):
            answer = QMessageBox.question(
                self,
                "下载模型",
                f"已选择 {model_name}，但它还没有下载。\n\n是否现在下载？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.install_model(model_name)
        self.refresh_local_models()

    def set_model_operation_busy(self, busy: bool) -> None:
        self.model_operation_running = busy
        self.refresh_local_models()

    def install_model(self, model_name: str | None = None) -> None:
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.information(self, "下载模型", "转写正在运行，请先停止后再管理模型。")
            return
        if self.model_operation_running:
            return
        model_name = model_name or self.selected_local_model_id()
        index = self.local_model_combo.findData(model_name)
        if index >= 0:
            self.local_model_combo.blockSignals(True)
            self.local_model_combo.setCurrentIndex(index)
            self.local_model_combo.blockSignals(False)
        config = self.collect_config()
        config["model"] = model_name
        write_config(CONFIG_PATH, config)
        self.config_data = config
        self.model_cache_status.setText(f"{model_name}：正在下载或检查")
        self.status_label.setText(f"正在下载或检查模型：{model_name}")
        self.start_model_operation("download", model_name, config)

    def delete_model(self, model_name: str) -> None:
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.information(self, "删除模型", "转写正在运行，请先停止后再管理模型。")
            return
        if self.model_operation_running:
            return
        answer = QMessageBox.question(
            self,
            "删除模型",
            f"删除本地模型缓存：{model_name}？\n\n下次使用需要重新下载。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.model_cache_status.setText(f"{model_name}：正在删除")
        self.status_label.setText(f"正在删除模型缓存：{model_name}")
        self.start_model_operation("delete", model_name, self.collect_config())

    def start_model_operation(self, operation: str, model_name: str, config: dict) -> None:
        self.set_model_operation_busy(True)
        self.model_thread = QThread(self)
        self.model_worker = ModelOperationWorker(operation, model_name, config)
        self.model_worker.moveToThread(self.model_thread)
        self.model_thread.started.connect(self.model_worker.run)
        self.model_worker.status.connect(self.on_model_operation_status)
        self.model_worker.finished.connect(self.on_model_operation_finished)
        self.model_worker.finished.connect(self.model_thread.quit)
        self.model_worker.finished.connect(self.model_worker.deleteLater)
        self.model_thread.finished.connect(self.model_thread.deleteLater)
        self.model_thread.start()

    def on_model_operation_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.model_cache_status.setText(message)
        self.append_log(message)

    def on_model_operation_finished(self, ok: bool, message: str) -> None:
        self.status_label.setText(message if ok else f"模型操作失败：{message}")
        self.model_cache_status.setText(message if ok else f"模型操作失败：{message}")
        self.append_log(message if ok else f"模型操作失败：{message}")
        self.set_model_operation_busy(False)
        self.model_thread = None
        self.model_worker = None

    def refresh_source_visibility(self) -> None:
        api_mode = MODEL_SOURCE_CODES.get(self.source_combo.currentText(), "local") == "api"
        self.local_model_combo.setVisible(not api_mode)
        self.input_api_provider_combo.setVisible(api_mode)
        self.input_api_model_combo.setVisible(api_mode)
        self.input_api_model_detail.setVisible(api_mode)
        self.update_start_button_state()

    def update_provider_status(self) -> None:
        key_state = "Key 已填写" if self.api_key.text().strip() else "Key 未填写"
        enabled = "启用" if self.api_enabled_check.isChecked() else "未启用"
        option = api_model_option(self.selected_api_provider_code(), self.selected_api_model_id())
        self.provider_status.setText(f"{self.api_provider_id.text().strip() or '未命名'} / {self.api_provider_combo.currentText()} / {option['id']} / {enabled} / {key_state}")
        self.update_model_overview()

    def update_model_overview(self, *_args) -> None:
        if not hasattr(self, "overview_model_label"):
            return
        running = self.transcriber_running()
        try:
            config = self.running_config.copy() if running and self.running_config else self.collect_config()
        except Exception:
            config = self.config_data.copy()
        source = str(config.get("model_source", "local"))
        if source == "api":
            provider = str(config.get("api_provider", config.get("api_provider_id", "dashscope")))
            provider_label = API_PROVIDER_LABELS.get(provider, provider or "DashScope")
            model_id = str(config.get("api_model", "qwen3-asr-flash") or "qwen3-asr-flash")
            key_ready = bool(str(config.get("api_key", "")).strip())
            self.overview_source_label.setText(f"API · {provider_label}")
            self.overview_model_label.setText(model_id)
            if running:
                state = "运行中"
                detail = "后台正在使用这个 API 模型接收麦克风分段并转写。"
            elif key_ready:
                state = "Key 已填写，可开始"
                detail = "点击开始会使用该千问模型进行 API 转写。"
            else:
                state = "Key 未填写，不能开始"
                detail = "请先在模型页或输入页填入阿里云 DashScope API Key。"
        else:
            model_name = str(config.get("model", DEFAULT_CONFIG["model"]))
            installed = self.local_model_installed(model_name, config)
            profile = MODEL_PROFILES.get(model_name, {})
            self.overview_source_label.setText("本地模型")
            self.overview_model_label.setText(model_name)
            if running:
                state = "运行中"
                detail = f"后台正在使用本地 {model_name} 模型转写。"
            elif installed:
                state = "已下载，可开始"
                detail = profile.get("description", "点击开始会使用这个本地模型。")
            else:
                state = "未下载，不能开始"
                detail = "请先在模型页下载该模型，或切换到 API 模型。"
        self.overview_state_label.setText(state)
        self.overview_detail_label.setText(detail)

    def update_trigger_summary(self) -> None:
        record = self.record_threshold.value()
        transcribe = max(record, self.transcribe_threshold.value())
        self.trigger_summary.setText(
            f"增益 {self.input_gain.value():.1f}x / 录音线 {record:.3f} / 转写线 {transcribe:.3f} / "
            f"前置 {self.pre_roll.value():.1f}s / 等待 {self.pause_seconds.value():.1f}s / "
            f"最短 {self.min_phrase.value():.1f}s / 丢弃 {self.silence_seconds.value():.1f}s / 最长 {self.max_phrase.value():.0f}s / "
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
                "api_model": self.selected_api_model_id(),
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
                "min_phrase_seconds": round(self.min_phrase.value(), 1),
                "transcribe_pause_seconds": round(self.pause_seconds.value(), 1),
                "silence_seconds": round(self.silence_seconds.value(), 2),
                "max_phrase_seconds": round(self.max_phrase.value(), 1),
                "cache_retention_minutes": round(self.cache_retention.value(), 1),
                "mic_device": self.selected_device_index(),
                "reply_bubble_enabled": self.bubble_enabled.isChecked(),
                "reply_bubble_port": int(self.bubble_port.value()),
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
        self.append_log("配置已保存")
        self.update_provider_status()
        self.refresh_local_models()

    def activate_api_provider(self) -> None:
        self.api_enabled_check.setChecked(True)
        self.source_combo.setCurrentText("API 模型")
        self.save_config()
        self.status_label.setText("已切换到 API Provider；开始后将使用千问 ASR")
        self.append_log("已切换到 API Provider；开始后将使用千问 ASR")

    def validate_api_provider(self) -> None:
        missing = []
        if not self.api_provider_id.text().strip():
            missing.append("Provider ID")
        if not self.selected_api_model_id():
            missing.append("模型")
        if not self.api_base_url.text().strip():
            missing.append("Base URL")
        if not self.api_key.text().strip():
            missing.append("API Key")
        if missing:
            QMessageBox.warning(self, "Provider 配置", "还缺少：" + ", ".join(missing))
        else:
            QMessageBox.information(self, "Provider 配置", "字段完整。点击开始后会用选中的千问模型进行 API 转写。")

    def test_rabiroute_connection(self) -> None:
        config = self.collect_config()
        url = str(config.get("rabiroute_url", "")).strip()
        if not url:
            QMessageBox.warning(self, "RabiRoute", "请先填写推送 URL。")
            return
        now = datetime.now()
        payload = {
            "type": "voice_transcript",
            "source": str(config.get("rabiroute_source", "fennenote") or "fennenote"),
            "text": "FenneNote RabiRoute 连接测试。",
            "startedAt": now.isoformat(),
            "endedAt": now.isoformat(),
            "durationSeconds": 0.0,
            "peak": 0.0,
            "time": int(now.timestamp()),
            "messageId": f"fennenote-test-{int(now.timestamp() * 1000)}",
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "FenneNote",
            },
        )
        token = str(config.get("rabiroute_token", "")).strip()
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"连接失败：HTTP {exc.code} {body[:160]}"
            self.route_status.setText(message)
            QMessageBox.warning(self, "RabiRoute", message)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            message = f"连接失败：{exc}"
            self.route_status.setText(message)
            QMessageBox.warning(self, "RabiRoute", message)
            return
        if 200 <= status < 300:
            self.route_enabled.setChecked(True)
            config = self.collect_config()
            config["rabiroute_enabled"] = True
            write_config(CONFIG_PATH, config)
            self.config_data = config
            message = f"连接成功：RabiRoute 已接收测试事件（HTTP {status}）"
            self.route_status.setText(message)
            self.status_label.setText(message)
            self.append_log(message)
            QMessageBox.information(self, "RabiRoute", message)
        else:
            message = f"连接失败：HTTP {status}"
            self.route_status.setText(message)
            QMessageBox.warning(self, "RabiRoute", message)

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
        self.level_worker.error.connect(self.append_log)
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
        if time.monotonic() >= self.writing_until:
            self.set_buddy_state("listening" if level >= threshold else "idle")
        self.trigger_state_label.setText("状态：录音中，芬妮正在听" if level >= threshold else "状态：监听中，芬妮在待命")
        self.wave.push(self.display_level, threshold, transcribe)
        self.trigger_wave.push(self.display_level, threshold, transcribe)

    def toggle_transcriber(self) -> None:
        if self.transcriber_running():
            self.stop_transcriber()
        else:
            self.start_transcriber()

    def start_transcriber(self) -> None:
        config = self.collect_config()
        write_config(CONFIG_PATH, config)
        if config.get("model_source") == "api":
            provider = str(config.get("api_provider", config.get("api_provider_id", ""))).strip().lower()
            missing = []
            if provider != "dashscope":
                missing.append("Provider 请选择 千问 / DashScope")
            if not str(config.get("api_model", "")).strip():
                missing.append("模型")
            if not str(config.get("api_key", "")).strip():
                missing.append("API Key")
            if missing:
                QMessageBox.warning(self, "API 模型", "还不能启动：" + "、".join(missing))
                return
        model_name = str(config.get("model", DEFAULT_CONFIG["model"]))
        if not self.local_model_installed(model_name, config):
            answer = QMessageBox.question(
                self,
                "模型未下载",
                f"当前选择的本地模型 {model_name} 还没有下载。\n\n是否现在下载？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.install_model(model_name)
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
        self.running_config = config.copy()
        self.update_start_button_state()
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.status_label.setText("正在启动")
        self.append_log("正在启动转写进程")
        self.set_buddy_state("listening")

    def send_process_command(self, command: str) -> None:
        if self.process_worker:
            self.process_worker.send(command)

    def stop_transcriber(self) -> None:
        if self.process_worker:
            self.status_label.setText("正在停止")
            self.append_log("正在停止转写进程")
            self.start_button.setEnabled(False)
            self.process_worker.stop()

    def on_process_line(self, line: str) -> None:
        if line.startswith("FN_STATUS|"):
            parts = line.split("|", 2)
            if len(parts) == 3:
                self.status_label.setText(parts[2])
                self.append_log(parts[2])
            return
        if line.startswith("["):
            self.writing_until = time.monotonic() + 2.8
            self.set_buddy_state("writing")
            self.transcript.appendPlainText(line)
        elif line:
            self.status_label.setText(line[:100])
            self.append_log(line)

    def on_process_exited(self, _code: int) -> None:
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止")
        self.append_log("转写进程已停止")
        self.running_config = None
        self.writing_until = 0.0
        self.set_buddy_state("idle")
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait(1000)
        self.process_thread = None
        self.process_worker = None
        self.update_start_button_state()

    def open_output_folder(self) -> None:
        output_dir = APP_DIR / str(self.collect_config().get("output_dir", "transcripts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def closeEvent(self, event) -> None:
        self.stop_transcriber()
        self.stop_level_worker()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_log_drawer_geometry()


def main() -> int:
    if "--transcriber" in sys.argv:
        transcriber_argv = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "--transcriber")]
        old_argv = sys.argv
        try:
            sys.argv = transcriber_argv
            from transcribe_mic import main as transcriber_main

            return transcriber_main()
        finally:
            sys.argv = old_argv
    app = QApplication(sys.argv)
    window = FenneNoteQt()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
