from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from transcribe_mic import (
    DEFAULT_CONFIG,
    DOWNLOADABLE_MODELS,
    MODEL_PROFILES,
    app_path,
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
EAR_ICON_PATH = APP_DIR / "assets" / "fennec-ear-icon.png"
LEGACY_OUMUQ_URL = "http://127.0.0.1:8790/api/speak"
BUDDY_STATE_IMAGE_PATHS = {
    "idle": APP_DIR / "assets" / "fennenote-state-idle.png",
    "listening": APP_DIR / "assets" / "fennenote-state-listening.png",
    "writing": APP_DIR / "assets" / "fennenote-state-writing.png",
}
VOICE_CLONE_READ_ALOUD_TEXT = (
    "你好，我正在录制一段声音样本。今天的天气还不错，窗外有一点风，"
    "我会用平常说话的速度和音量，把这段文字自然地读完。"
    "请记住我的声音、语气和停顿方式。"
)
VOICE_CLONE_SAMPLE_PRESETS = {
    "neutral": {
        "label": "中性音色",
        "vector": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65],
        "instruction": "用自然、清楚、平稳的中文说话，保持日常聊天的音色和节奏，不要播音腔。",
        "text": VOICE_CLONE_READ_ALOUD_TEXT,
    },
    "happy": {
        "label": "开心明亮",
        "vector": [0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.08],
        "instruction": "用开心、明亮、带一点笑意的自然中文说话，语气轻快但不要夸张。",
        "text": "太好了，今天真的很顺利。我一想到这件事就忍不住想笑，语气可以轻快一点，像是在和熟人分享一个好消息。",
    },
    "sad": {
        "label": "低落伤心",
        "vector": [0.0, 0.0, 0.82, 0.0, 0.0, 0.35, 0.0, 0.12],
        "instruction": "用低落、轻声、稍慢的自然中文说话，情绪伤心但不要哭腔过重。",
        "text": "我有一点难过，说话会慢一些，声音也会轻一点。不是在哭，只是心情低落，像把话慢慢说给信任的人听。",
    },
    "excited": {
        "label": "兴奋惊喜",
        "vector": [0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.72, 0.03],
        "instruction": "用兴奋、惊喜、有精神的自然中文说话，语速可以稍快，但不要喊叫。",
        "text": "等一下，这也太厉害了吧。我有点兴奋，语速可以稍微快一点，声音更有精神，但不要喊得太夸张。",
    },
    "angry": {
        "label": "不满生气",
        "vector": [0.0, 0.78, 0.0, 0.0, 0.18, 0.0, 0.05, 0.08],
        "instruction": "用坚定、压着火气、略带不满的自然中文说话，保持清楚克制，不要吼叫。",
        "text": "这件事我真的有点不高兴。语气要更坚定，带一点压住火气的不满，但仍然保持清楚，不要大喊。",
    },
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
AUDIO_ROUTE_PRESET_LABELS = {
    "solo_voice_input": "Solo voice input - FenneNote listens to physical mic",
    "qq_mixed_output_mode": "QQ mixed output mode - QQ uses virtual mixed mic",
    "mixed_transcription_input": "Mixed transcription - mic plus computer audio input",
}
AUDIO_ROUTE_PRESET_CODES = {label: code for code, label in AUDIO_ROUTE_PRESET_LABELS.items()}
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
            "speaker_support": "普通转写不区分具体是谁；多发言人字幕请在“声纹识别”页开启，会另走阿里非实时 paraformer-v2 + diarization_enabled，再配合本地声纹把说话人 1/2/3 匹配成具体人名。",
            "note": "官方推荐的新 ASR。只做转写时通常比 Omni 更直接、更好估算成本。",
        },
        {
            "id": "qwen-audio-asr",
            "label": "旧 ASR · qwen-audio-asr · 中英识别 · 免费体验",
            "tier": "旧版专用语音识别 / Beta",
            "best_for": "中文、英文短音频转写；想沿用旧接口时使用。",
            "price": "目前仅供免费体验；额度用完后不可调用，官方推荐迁移 Qwen3 ASR。",
            "speaker_support": "不作为多发言人分离入口；需要识别具体是谁时，使用声纹页的非实时 paraformer-v2 diarization + 本地声纹匹配。",
            "note": "支持语言较少，不建议作为新的生产默认项。",
        },
        {
            "id": "qwen-audio-turbo",
            "label": "音频理解 · qwen-audio-turbo · 可问答 · 免费体验",
            "tier": "音频理解 / 对话模型",
            "best_for": "让模型理解音频内容、总结、回答“这段音频在说什么”。",
            "price": "目前仅供免费体验；音频约 25 token/秒；额度用完后推荐 Qwen-Omni。",
            "speaker_support": "不返回稳定的多发言人时间段；不要用它做说话人字幕，也不能可靠配合声纹识别具体是谁。",
            "note": "不是纯 ASR，转写准确率和长音频能力通常不如专用 ASR。",
        },
        {
            "id": "qwen2.5-omni-7b",
            "label": "多模态 · qwen2.5-omni-7b · 音频/图像/视频 · ¥38/百万音频 token",
            "tier": "多模态理解 + 文本/语音输出",
            "best_for": "需要同一个模型处理文字、图片、视频、音频，或后续做语音对话。",
            "price": "国内约：文本输入 ¥0.6/M、音频输入 ¥38/M、视觉输入 ¥2/M、文本输出 ¥2.4-6/M、音频输出 ¥76/M。",
            "speaker_support": "不是当前说话人分离入口；要识别具体是谁，优先用专用 ASR + 非实时 diarization，再配合本地声纹。",
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
            "speaker_support": "当前 FenneNote 没用它做多发言人分离；要识别具体是谁，说话人字幕仍走阿里非实时 diarization + 本地声纹。",
            "note": "OpenAI 官方兼容；第三方 OpenAI Compatible 服务价格可能不同。",
        },
        {
            "id": "whisper-1",
            "label": "兼容便宜 · whisper-1 · 老牌通用 · $0.006/分钟",
            "tier": "Whisper 兼容 / 通用转写",
            "best_for": "成本稳定、兼容性优先、对最新准确率要求不高的任务。",
            "price": "OpenAI 官方列价：$0.006/分钟。",
            "speaker_support": "不返回多发言人时间段；不能直接生成说话人字幕，也不能单独识别具体是谁。",
            "note": "老模型，生态成熟；准确率通常不如 gpt-4o-transcribe。",
        },
    ],
}
API_MODELS_BY_PROVIDER = {
    provider: [option["id"] for option in options]
    for provider, options in API_MODEL_OPTIONS_BY_PROVIDER.items()
}
THEME = {
    "app_bg": "#fff8f0",
    "panel": "#fffdf8",
    "panel_tint": "#fff4df",
    "panel_soft": "#fffaf2",
    "nav": "#20363a",
    "nav_hover": "#2d5558",
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
    "aqua": "#dff8f4",
    "rose": "#ef7f8f",
    "rose_soft": "#ffe6e7",
    "peach": "#ffd7a6",
    "green": "#20b874",
    "danger": "#d45b4c",
    "canvas": "#fffaf2",
    "quiet_bar": "#6a5a52",
    "shadow": "#d9b98a",
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
        f"说话人：{option.get('speaker_support', '未标注')}\n"
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


class ToggleSwitch(QCheckBox):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(34)

    def sizeHint(self):
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        return super().sizeHint().expandedTo(QRectF(0, 0, 66 + text_width, 34).size().toSize())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_width = 54
        track_height = 30
        knob_size = 26
        y = (self.height() - track_height) / 2
        track = QRectF(0, y, track_width, track_height)
        checked = self.isChecked()
        enabled = self.isEnabled()

        if checked:
            track_color = QColor(THEME["teal"] if enabled else "#a7bbb6")
            knob_color = QColor("#fffaf1")
        else:
            track_color = QColor("#8f8778" if enabled else "#c7beb0")
            knob_color = QColor("#fff6e8")

        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, track_height / 2, track_height / 2)

        knob_x = track_width - knob_size - 2 if checked else 2
        knob = QRectF(knob_x, y + 2, knob_size, knob_size)
        painter.setBrush(QColor(0, 0, 0, 28))
        painter.drawEllipse(knob.translated(0, 1))
        painter.setBrush(knob_color)
        painter.drawEllipse(knob)

        if self.text():
            painter.setPen(QColor(THEME["ink"] if enabled else "#9f9385"))
            text_rect = self.rect().adjusted(66, 0, 0, 0)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())


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

    def setRange(self, minimum: float, maximum: float, step: float | None = None) -> None:
        value = self.value()
        self.minimum = minimum
        self.maximum = maximum
        if step is not None:
            self.step = step
        steps = max(1, int(round((maximum - minimum) / self.step)))
        self.slider.setRange(0, steps)
        self.slider.setPageStep(max(1, steps // 10))
        self.setValue(value)

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
    playback_finished = Signal(bool, str, dict)
    playback_request_received = Signal(dict)
    reply_request_received = Signal(dict)
    voice_clone_sample_saved = Signal(bool, str, str)
    speaker_operation_finished = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        ensure_cuda_dll_path()
        self.config_data = load_config(CONFIG_PATH)
        self.devices = self.list_input_devices()
        self.system_audio_devices = self.list_system_audio_input_devices()
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
        self.playback_api_server: ThreadingHTTPServer | None = None
        self.playback_api_thread: threading.Thread | None = None
        self.active_playback_route: tuple[str, str, str, str, str] | None = None
        self.force_quit = False
        self.tray_icon: QSystemTrayIcon | None = None
        self.tray_hide_notice_shown = False
        self.preview_noise_floor = 0.003
        self.display_level = 0.0
        self.display_raw_level = 0.0
        self.level_history: deque[float] = deque(maxlen=160)
        self.buddy_state = ""
        self.buddy_state_changed_at = 0.0
        self.transcriber_activity_state = "idle"
        self.buddy_pixmaps = self.load_buddy_pixmaps()
        self.build_ui()
        self.setup_tray()
        self.playback_request_received.connect(self.handle_playback_api_request)
        self.reply_request_received.connect(self.handle_reply_api_request)
        self.voice_clone_sample_saved.connect(self.on_voice_clone_sample_saved)
        self.speaker_operation_finished.connect(self.on_speaker_operation_finished)
        self.set_buddy_state("idle")
        self.apply_config_to_ui()
        self.start_playback_api_server()
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
        self.buddy_state_changed_at = time.monotonic()
        self.update_buddy_view()

    def update_buddy_view(self) -> None:
        if not hasattr(self, "buddy_image"):
            return
        pixmap = self.buddy_pixmaps.get(self.buddy_state)
        if not pixmap:
            self.buddy_image.clear()
            return
        self.buddy_image.setPixmap(pixmap.scaled(124, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        labels = {
            "idle": "芬妮在打瞌睡，等你说话",
            "listening": "芬妮正在听",
            "writing": "芬妮正在记录",
        }
        self.buddy_caption.setText(labels.get(self.buddy_state, "芬妮待命中"))

    def setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.append_log("系统托盘不可用，关闭窗口将直接退出。")
            return
        icon = self.windowIcon()
        if icon.isNull() and ICON_PATH.exists():
            icon = QIcon(str(ICON_PATH))
            self.setWindowIcon(icon)
        if icon.isNull():
            icon = QApplication.style().standardIcon(QApplication.style().SP_ComputerIcon)

        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("FenneNote")
        menu = QMenu(self)

        show_action = QAction("打开 FenneNote", self)
        show_action.triggered.connect(self.show_main_window)
        menu.addAction(show_action)

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.quit_from_tray)
        menu.addAction(quit_action)

        tray.setContextMenu(menu)
        tray.activated.connect(self.on_tray_activated)
        tray.show()
        self.tray_icon = tray

    def on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.DoubleClick, QSystemTrayIcon.Trigger):
            self.show_main_window()

    def show_main_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def quit_from_tray(self) -> None:
        self.force_quit = True
        self.close()

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
            QLabel#navBrand {{
                color: {THEME["nav_text"]};
                font-size: 19px;
                font-weight: 900;
            }}
            QLabel#navChip {{
                color: {THEME["nav_muted"]};
                background: rgba(255, 248, 236, 24);
                border: 1px solid rgba(255, 248, 236, 32);
                border-radius: 8px;
                padding: 6px 8px;
            }}
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
            QFrame#listenDeck {{
                background: {THEME["panel_tint"]};
                border: 1px solid {THEME["line_strong"]};
                border-radius: 8px;
            }}
            QFrame#statusCard {{
                background: {THEME["panel"]};
                border: 1px solid {THEME["line"]};
                border-radius: 8px;
            }}
            QFrame#endpointCard {{
                background: {THEME["aqua"]};
                border: 1px solid #acdcd3;
                border-radius: 8px;
            }}
            QLabel#statusKicker {{
                color: {THEME["sand_dark"]};
                font-size: 9pt;
                font-weight: 800;
            }}
            QLabel#statusValue {{
                color: {THEME["ink"]};
                font-size: 12px;
                font-weight: 900;
            }}
            QLabel#statusNote {{
                color: {THEME["muted"]};
                font-size: 9pt;
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
                font-size: 18px;
                font-weight: 800;
            }}
            QLabel#heroKicker {{
                color: {THEME["teal_dark"]};
                font-size: 9pt;
                font-weight: 900;
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
            QComboBox, QLineEdit {{
                background: #fffdf8;
                border: 1px solid {THEME["line"]};
                border-radius: 7px;
                padding: 6px;
            }}
            QComboBox:hover, QLineEdit:hover {{
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
        nav.setFixedWidth(204)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(12, 18, 12, 12)
        brand_row = QHBoxLayout()
        brand_row.setSpacing(8)
        brand_icon = QLabel()
        brand_icon.setFixedSize(34, 34)
        if EAR_ICON_PATH.exists():
            pixmap = QPixmap(str(EAR_ICON_PATH))
            if not pixmap.isNull():
                brand_icon.setPixmap(pixmap.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        brand_title = QLabel("FenneNote")
        brand_title.setObjectName("navBrand")
        brand_row.addWidget(brand_icon)
        brand_row.addWidget(brand_title, 1)
        nav_layout.addLayout(brand_row)
        subtitle = QLabel("芬妮语音监听台")
        subtitle.setObjectName("navSub")
        nav_layout.addWidget(subtitle)
        nav_chip = QLabel("托盘常驻 · 本地优先")
        nav_chip.setObjectName("navChip")
        nav_layout.addWidget(nav_chip)
        self.nav_list = QListWidget()
        for text in ("总览", "输入", "模型", "触发", "应用", "声纹识别", "OumuQ 扩展", "RabiRoute 扩展"):
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
        self.folder_button = QPushButton("打开转写记录")
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
        self.build_speaker_recognition_page()
        self.build_voice_output_page()
        self.build_route_page()
        self.nav_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        self.start_button.clicked.connect(self.toggle_transcriber)
        self.pause_button.clicked.connect(lambda: self.send_process_command("p\n"))
        self.stop_button.clicked.connect(self.stop_transcriber)
        self.save_button.clicked.connect(self.save_config)
        self.folder_button.clicked.connect(self.open_output_folder)
        self.log_button.clicked.connect(self.toggle_log_drawer)
        self.playback_finished.connect(self.on_playback_finished)
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 10, 14)
        layout.setSpacing(14)
        scroll.setWidget(container)
        self.pages.addWidget(scroll)
        return scroll, container, layout

    def help_label(self, text: str, tooltip: str) -> QWidget:
        label = QWidget()
        layout = QHBoxLayout(label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        name = QLabel(text)
        mark = QToolButton()
        mark.setText("?")
        mark.setCursor(Qt.PointingHandCursor)
        mark.setAutoRaise(True)
        mark.setFixedSize(18, 18)
        mark.setStyleSheet(
            "QToolButton { border: 1px solid #d8c0ca; border-radius: 9px; color: #7b6670; background: #fff8fb; font-weight: 700; padding: 0; }"
            "QToolButton:hover { border-color: #c89aad; color: #5d4852; background: #fff0f6; }"
            "QToolButton:pressed { background: #f8dfe8; }"
        )
        def show_help() -> None:
            QToolTip.showText(mark.mapToGlobal(mark.rect().bottomLeft()), tooltip, mark)

        mark.clicked.connect(show_help)
        label.setToolTip(tooltip)
        name.setToolTip(tooltip)
        mark.setToolTip(tooltip)
        layout.addWidget(name)
        layout.addWidget(mark)
        layout.addStretch(1)
        return label

    def add_help_row(self, form: QFormLayout, label: str, widget: QWidget, tooltip: str) -> None:
        widget.setToolTip(tooltip)
        form.addRow(self.help_label(label, tooltip), widget)

    def path_row(self, *widgets: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, 1 if index == 0 else 0)
        return row

    def resolve_local_path(self, value: str, default: str) -> Path:
        return app_path((value or default).strip() or default).resolve()

    def output_dir_from_ui(self) -> Path:
        value = self.output_dir_edit.text() if hasattr(self, "output_dir_edit") else str(self.config_data.get("output_dir", DEFAULT_CONFIG["output_dir"]))
        return self.resolve_local_path(value, DEFAULT_CONFIG["output_dir"])

    def cache_dir_from_ui(self) -> Path:
        value = self.cache_dir_edit.text() if hasattr(self, "cache_dir_edit") else str(self.config_data.get("cache_dir", DEFAULT_CONFIG["cache_dir"]))
        return self.resolve_local_path(value, DEFAULT_CONFIG["cache_dir"])

    def audio_dir_from_ui(self) -> Path:
        return (self.cache_dir_from_ui() / "audio").resolve()

    def tts_guard_path_from_ui(self) -> Path:
        value = self.tts_guard_file.text() if hasattr(self, "tts_guard_file") else str(self.config_data.get("tts_guard_file", DEFAULT_CONFIG["tts_guard_file"]))
        return self.resolve_local_path(value, DEFAULT_CONFIG["tts_guard_file"])

    def oumuq_registry_path_from_ui(self) -> Path:
        value = self.oumuq_registry_path.text() if hasattr(self, "oumuq_registry_path") else str(self.config_data.get("oumuq_registry_path", DEFAULT_CONFIG["oumuq_registry_path"]))
        path = self.resolve_local_path(value, DEFAULT_CONFIG["oumuq_registry_path"])
        if path.exists():
            return path
        raw_path = Path((value or DEFAULT_CONFIG["oumuq_registry_path"]).strip() or DEFAULT_CONFIG["oumuq_registry_path"])
        if raw_path.is_absolute():
            return path
        for root in (APP_DIR, *APP_DIR.parents):
            candidate = root / raw_path
            if candidate.exists():
                return candidate.resolve()
            candidate = root / "OumuQ" / "voice-references" / "reference-index.json"
            if candidate.exists():
                return candidate.resolve()
        return path

    def speaker_registry_path_from_ui(self) -> Path:
        value = self.speaker_registry_file.text() if hasattr(self, "speaker_registry_file") else str(self.config_data.get("speaker_registry_file", DEFAULT_CONFIG["speaker_registry_file"]))
        return self.resolve_local_path(value, DEFAULT_CONFIG["speaker_registry_file"])

    def open_local_path(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def current_playback_api_token(self) -> str:
        if hasattr(self, "playback_api_token_edit"):
            return self.playback_api_token_edit.text().strip()
        return str(self.config_data.get("playback_api_token", DEFAULT_CONFIG["playback_api_token"])).strip()

    def start_playback_api_server(self) -> None:
        if not bool(self.config_data.get("playback_api_enabled", DEFAULT_CONFIG["playback_api_enabled"])):
            return
        port = int(self.config_data.get("playback_api_port", DEFAULT_CONFIG["playback_api_port"]))
        owner = self

        class PlaybackApiHandler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def send_json(self, status: int, body: dict) -> None:
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def authorized(self) -> bool:
                token = owner.current_playback_api_token()
                if not token:
                    return True
                auth = self.headers.get("Authorization", "")
                header_token = self.headers.get("X-FenneNote-Token", "")
                return auth == f"Bearer {token}" or header_token == token

            def do_GET(self) -> None:
                if self.path != "/healthz":
                    self.send_json(404, {"ok": False, "error": "not_found"})
                    return
                self.send_json(200, {
                    "ok": True,
                    "service": "fennenote-endpoint",
                    "playback": "/api/fennenote/playback",
                    "reply": "/api/fennenote/reply",
                })

            def do_POST(self) -> None:
                try:
                    if self.path not in {"/api/fennenote/playback", "/api/fennenote/reply"}:
                        self.send_json(404, {"ok": False, "error": "not_found"})
                        return
                    if not self.authorized():
                        self.send_json(401, {"ok": False, "error": "unauthorized"})
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 1024 * 1024:
                            raise ValueError("invalid content length")
                        payload = json.loads(self.rfile.read(length).decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("body must be a JSON object")
                        text = str(payload.get("text") or payload.get("message") or payload.get("content") or "").strip()
                        if not text:
                            raise ValueError("missing text")
                    except Exception as exc:
                        self.send_json(400, {"ok": False, "error": str(exc)})
                        return
                    request_id = f"fennenote-{int(time.time() * 1000)}"
                    payload["_request_id"] = request_id
                    if self.path == "/api/fennenote/reply":
                        owner.reply_request_received.emit(payload)
                        provider = "fennenote_reply"
                    else:
                        owner.playback_request_received.emit(payload)
                        provider = "fennenote_playback"
                    self.send_json(202, {
                        "ok": True,
                        "status": "queued",
                        "id": request_id,
                        "provider": provider,
                        "request": {"payload": payload},
                    })
                except Exception as exc:
                    self.send_json(500, {"ok": False, "error": str(exc)})

        try:
            self.playback_api_server = ThreadingHTTPServer(("127.0.0.1", port), PlaybackApiHandler)
        except OSError as exc:
            self.append_log(f"FenneNote 播放 API 启动失败：{exc}")
            return
        self.playback_api_thread = threading.Thread(target=self.playback_api_server.serve_forever, daemon=True)
        self.playback_api_thread.start()
        self.append_log(f"FenneNote 端已启动：http://127.0.0.1:{port}/api/fennenote/playback 和 /api/fennenote/reply")

    def stop_playback_api_server(self) -> None:
        if self.playback_api_server:
            self.playback_api_server.shutdown()
            self.playback_api_server.server_close()
        self.playback_api_server = None
        self.playback_api_thread = None

    def browse_directory_into(self, edit: QLineEdit, default: str) -> None:
        current = self.resolve_local_path(edit.text(), default)
        current.mkdir(parents=True, exist_ok=True)
        selected = QFileDialog.getExistingDirectory(self, "选择目录", str(current))
        if selected:
            edit.setText(selected)

    def browse_file_into(self, edit: QLineEdit, default: str) -> None:
        current = self.resolve_local_path(edit.text(), default)
        selected, _ = QFileDialog.getOpenFileName(self, "选择文件", str(current.parent), "JSON (*.json);;All files (*.*)")
        if selected:
            edit.setText(selected)

    def refresh_path_labels(self) -> None:
        if not hasattr(self, "audio_dir_value"):
            return
        self.audio_dir_value.setText(str(self.audio_dir_from_ui()))
        self.model_cache_value.setText(str((self.cache_dir_from_ui() / "models").resolve()))
        self.temp_cache_value.setText(str((self.cache_dir_from_ui() / "temp").resolve()))

    def overview_card(self, title: str, value: str, note: str = "") -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("statusCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("statusKicker")
        value_label = QLabel(value)
        value_label.setObjectName("statusValue")
        value_label.setWordWrap(True)
        note_label = QLabel(note)
        note_label.setObjectName("statusNote")
        note_label.setWordWrap(True)
        if note:
            card.setToolTip(note)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(note_label)
        return card, value_label, note_label

    def page_banner(self, title: str, text: str) -> QFrame:
        banner = QFrame()
        banner.setObjectName("endpointCard")
        layout = QVBoxLayout(banner)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("statusKicker")
        text_label = QLabel(text)
        text_label.setObjectName("statusNote")
        text_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(text_label)
        return banner

    def build_dashboard_page(self) -> None:
        scroll, container, layout = self.page()
        hero = QFrame()
        hero.setObjectName("listenDeck")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(14, 10, 14, 10)
        hero_layout.setSpacing(12)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(6)
        kicker = QLabel("FENNE LISTEN DESK")
        kicker.setObjectName("heroKicker")
        hero_text.addWidget(kicker)
        title = QLabel("芬妮语音监听台")
        title.setObjectName("heroTitle")
        hero_text.addWidget(title)
        hero_subtitle = QLabel("麦克风、转写、声纹和播放入口已收束在这里。")
        hero_subtitle.setWordWrap(True)
        hero_subtitle.setStyleSheet(f"color: {THEME['muted']};")
        hero_text.addWidget(hero_subtitle)

        cards = QGridLayout()
        cards.setHorizontalSpacing(8)
        cards.setVerticalSpacing(6)
        source_card, self.overview_source_label, source_note = self.overview_card("模型来源", "来源", "本地 GPU 或云端 ASR")
        model_card, self.overview_model_label, self.overview_detail_label = self.overview_card("启动模型", "模型", "")
        state_card, self.overview_state_label, state_note = self.overview_card("当前状态", "状态", "关闭窗口后会留在托盘")
        endpoint_card = QFrame()
        endpoint_card.setObjectName("endpointCard")
        endpoint_layout = QVBoxLayout(endpoint_card)
        endpoint_layout.setContentsMargins(10, 7, 10, 7)
        endpoint_layout.setSpacing(2)
        endpoint_title = QLabel("FenneNote 端")
        endpoint_title.setObjectName("statusKicker")
        endpoint_value = QLabel("8793 · reply / playback")
        endpoint_value.setObjectName("statusValue")
        endpoint_note = QLabel("RabiRoute 可反写文字，也可送播放请求。")
        endpoint_note.setObjectName("statusNote")
        endpoint_note.setWordWrap(True)
        endpoint_layout.addWidget(endpoint_title)
        endpoint_layout.addWidget(endpoint_value)
        endpoint_layout.addWidget(endpoint_note)
        cards.addWidget(source_card, 0, 0)
        cards.addWidget(model_card, 0, 1)
        cards.addWidget(state_card, 0, 2)
        cards.addWidget(endpoint_card, 0, 3)
        for column in range(4):
            cards.setColumnStretch(column, 1)
        hero_text.addLayout(cards)
        source_note.setText("")
        state_note.setText("")
        endpoint_note.setText("")

        self.trigger_state_label = QLabel("状态：监听中")
        self.trigger_state_label.setStyleSheet(f"color: {THEME['teal_dark']}; font-weight: 700;")
        hero_text.addWidget(self.trigger_state_label)
        hero_layout.addLayout(hero_text, 1)

        buddy_card = QFrame()
        buddy_card.setObjectName("buddyBubble")
        buddy_layout = QVBoxLayout(buddy_card)
        buddy_layout.setContentsMargins(10, 8, 10, 8)
        buddy_layout.setSpacing(4)
        self.buddy_image = QLabel()
        self.buddy_image.setAlignment(Qt.AlignCenter)
        self.buddy_image.setFixedSize(132, 104)
        self.buddy_caption = QLabel("芬妮待命中")
        self.buddy_caption.setObjectName("buddyCaption")
        self.buddy_caption.setAlignment(Qt.AlignCenter)
        buddy_layout.addWidget(self.buddy_image)
        buddy_layout.addWidget(self.buddy_caption)
        hero_layout.addWidget(buddy_card, 0)
        layout.addWidget(hero)

        monitor = QGroupBox("实时监听")
        monitor_layout = QVBoxLayout(monitor)
        self.level_text = QLabel("麦克风电平：原始 0.000 / 增益后 0.000 / 录音阈值 0.010 / 转写阈值 0.015")
        self.level_text.setStyleSheet(f"color: {THEME['teal_dark']}; font-weight: 700;")
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
        self.audio_route_preset_combo = QComboBox()
        self.audio_route_preset_combo.addItems(list(AUDIO_ROUTE_PRESET_CODES.keys()))
        self.audio_route_preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.audio_route_preset_combo.setMinimumContentsLength(12)
        self.mixed_input_enabled = ToggleSwitch("混合麦克风和电脑声音用于转写")
        self.system_audio_combo = QComboBox()
        self.system_audio_combo.addItem("不选择电脑声音输入", None)
        for index, label in self.system_audio_devices:
            self.system_audio_combo.addItem(label, index)
        self.system_audio_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.system_audio_combo.setMinimumContentsLength(12)
        self.system_audio_gain = self.slider(0.0, 3.0, 0.1, 1, "x")
        self.audio_route_note = QLabel(
            "先选择一个能代表电脑播放声的输入设备，例如立体声混音、虚拟声卡 Input、Steam/网易/AudioRelay 的回放输入；"
            "勾选上方开关后，开始转写时才会把它和麦克风混合。"
            "这不会创建 Windows 系统级虚拟麦克风，只会把两路声音混合后送 FenneNote 转写。"
        )
        self.audio_route_note.setWordWrap(True)
        self.audio_route_note.setStyleSheet("color: #6f5962;")
        self.source_combo = QComboBox()
        self.source_combo.addItems(list(MODEL_SOURCE_CODES.keys()))
        self.local_model_combo = QComboBox()
        self.input_api_provider_combo = QComboBox()
        self.input_api_provider_combo.addItems(list(API_PROVIDER_CODES.keys()))
        self.input_api_model_combo = QComboBox()
        self.input_api_model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.input_api_model_combo.setMinimumContentsLength(18)
        self.input_api_model_detail = QLabel()
        self.input_api_model_detail.setWordWrap(True)
        self.input_api_model_detail.setStyleSheet("color: #6f5962;")
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8_float16", "float16", "int8", "int8_float32", "float32"])
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(LANGUAGE_CODES.keys()))
        self.simplify_check = ToggleSwitch("输出简体中文")
        form.addRow("麦克风", self.mic_combo)
        form.addRow("音频路由预设", self.audio_route_preset_combo)
        form.addRow("", self.mixed_input_enabled)
        form.addRow("电脑声音输入", self.system_audio_combo)
        form.addRow("电脑声音增益", self.system_audio_gain)
        form.addRow("QQ 混音模式说明", self.audio_route_note)
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
        self.mixed_input_enabled.toggled.connect(self.refresh_mixed_input_visibility)

    def build_model_page(self) -> None:
        _, _, layout = self.page()
        provider = QGroupBox("API Provider 配置")
        form = QFormLayout(provider)
        self.api_enabled_check = ToggleSwitch("启用 API Provider")
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
        self.adaptive_check = ToggleSwitch("开启")
        self.input_gain = self.slider(1.0, 5.0, 0.1, 1, "x")
        self.record_threshold = self.slider(0.01, 0.04, 0.001, 3)
        self.transcribe_threshold = self.slider(0.01, 0.06, 0.001, 3)
        self.pre_roll = self.slider(0.0, 3.0, 0.1, 1, "s")
        self.min_phrase = self.slider(0.2, 10.0, 0.1, 1, "s")
        self.pause_seconds = self.slider(0.2, 2.0, 0.1, 1, "s")
        self.silence_seconds = self.slider(1.0, 8.0, 0.1, 1, "s")
        self.max_phrase = self.slider(10.0, 120.0, 1.0, 0, "s")
        for label, widget, tooltip in (
            ("自动适应环境噪声", self.adaptive_check, "根据房间底噪自动抬高开始录音的音量线。环境变吵时能减少误触发；讲话很轻时可以先关掉试试。"),
            ("麦克风音量放大", self.input_gain, "在进入转写前把麦克风采样整体放大。保存录音片段时，保存的也是放大后的音频。"),
            ("开始录音的音量线", self.record_threshold, "声音超过这条线后开始截取一段录音。太高会漏掉轻声，太低会把键盘声、风扇声也录进去。"),
            ("值得转写的音量线", self.transcribe_threshold, "一段录音的峰值至少达到这条线，才会送给模型转写。低于这条线的短噪声会被丢掉。"),
            ("句首多留几秒", self.pre_roll, "触发前额外带上一小段音频，避免第一个字已经说出口但录音刚开始。"),
            ("最短有效录音", self.min_phrase, "录音至少要持续这么久才算有效。调大可以过滤咳嗽、敲击等很短的声音。"),
            ("安静多久后切句", self.pause_seconds, "已经达到值得转写音量线后，如果安静超过这个时间，就把当前片段切开并开始转写。"),
            ("安静多久后丢弃杂音", self.silence_seconds, "如果这段声音一直没达到值得转写的音量线，安静超过这个时间后，当作杂音丢掉。"),
            ("单段最长录音", self.max_phrase, "一段录音最多保留这么久，到点就强制切段，避免一直说话或环境噪声让片段无限变长。"),
        ):
            self.add_help_row(form, label, widget, tooltip)
        layout.addWidget(group)
        for widget in (self.input_gain, self.record_threshold, self.transcribe_threshold, self.pre_roll, self.min_phrase, self.pause_seconds, self.silence_seconds, self.max_phrase):
            widget.valueChanged.connect(self.update_trigger_summary)
        self.adaptive_check.stateChanged.connect(self.update_trigger_summary)

    def build_app_page(self) -> None:
        _, _, layout = self.page()

        paths = QGroupBox("本地路径")
        path_form = QFormLayout(paths)
        self.output_dir_edit = QLineEdit()
        self.output_dir_browse_button = QPushButton("选择")
        self.output_dir_open_button = QPushButton("打开")
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_browse_button = QPushButton("选择")
        self.cache_dir_open_button = QPushButton("打开")
        self.open_audio_button = QPushButton("打开")
        self.open_config_button = QPushButton("打开配置")
        self.show_log_button = QPushButton("显示日志")
        self.tts_guard_file = QLineEdit()
        self.audio_dir_value = QLabel()
        self.model_cache_value = QLabel()
        self.temp_cache_value = QLabel()
        for label in (self.audio_dir_value, self.model_cache_value, self.temp_cache_value):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setStyleSheet(f"color: {THEME['muted']};")
        self.config_path_value = QLabel(str(CONFIG_PATH))
        self.config_path_value.setWordWrap(True)
        self.config_path_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.config_path_value.setStyleSheet(f"color: {THEME['muted']};")
        self.add_help_row(
            path_form,
            "转写记录目录",
            self.path_row(self.output_dir_edit, self.output_dir_browse_button, self.output_dir_open_button),
            "按日期长期保存转写 TXT 的目录，例如 transcripts/。这不是缓存，不会被缓存保留时间清理。",
        )
        self.add_help_row(
            path_form,
            "模型/运行缓存目录",
            self.path_row(self.cache_dir_edit, self.cache_dir_browse_button, self.cache_dir_open_button),
            "FenneNote 的本地缓存根目录。模型、Hugging Face 下载缓存、运行临时文件和录音片段默认都放在这里的子目录中。",
        )
        self.add_help_row(
            path_form,
            "录音片段目录",
            self.path_row(self.audio_dir_value, self.open_audio_button),
            "保存录音片段开启后，送去转写的 WAV 会保存到这个目录。它只受录音保留分钟影响。",
        )
        self.add_help_row(
            path_form,
            "本地模型缓存",
            self.model_cache_value,
            "Whisper / faster-whisper 模型下载目录。模型页的下载、检查、删除都作用在这里。",
        )
        self.add_help_row(
            path_form,
            "运行临时缓存",
            self.temp_cache_value,
            "运行临时文件目录，只受运行临时缓存保留分钟清理。",
        )
        self.add_help_row(
            path_form,
            "TTS guard 文件",
            self.tts_guard_file,
            "TTS 侧写入的本地 JSON 文件路径，例如 cache/tts_guard.json。设备隔离仍然是主防线。",
        )
        self.add_help_row(
            path_form,
            "配置文件",
            self.path_row(self.config_path_value, self.open_config_button),
            "当前本机 config.json。不要提交包含个人路径、API key、token 或公司环境地址的配置。",
        )
        self.add_help_row(
            path_form,
            "运行日志",
            self.show_log_button,
            "打开右侧运行日志抽屉。日志只在当前 GUI 会话内显示。",
        )
        layout.addWidget(paths)

        group = QGroupBox("录音保存与启动")
        form = QFormLayout(group)
        self.auto_start_check = ToggleSwitch("启动后自动开始")
        self.save_audio_segments_check = ToggleSwitch("开启")
        self.audio_retention = self.slider(1.0, 1440.0, 10.0, 0, " 分钟")
        self.cache_retention = self.slider(0.0, 60.0, 1.0, 0, " 分钟")
        form.addRow("", self.auto_start_check)
        self.add_help_row(form, "保存录音片段", self.save_audio_segments_check, "默认关闭，避免误存隐私。开启后，每段送去转写的麦克风音频会保存到录音片段目录，可作为以后训练或克隆自己声线的私有素材。")
        self.add_help_row(form, "录音保留分钟", self.audio_retention, "只影响录音片段目录里的 WAV。最低 1 分钟，默认 10 分钟，给声纹建档、异步识别和人工命名留下安全窗口。")
        self.add_help_row(form, "运行临时缓存保留分钟", self.cache_retention, "只清理运行临时缓存目录里的临时文件，不会删除模型缓存、转写记录，也不会删除录音片段。0 表示不自动清理临时缓存。")
        layout.addWidget(group)

        misc = QGroupBox("界面与防回流")
        misc_form = QFormLayout(misc)
        self.bubble_enabled = ToggleSwitch("启用左下角气泡")
        self.bubble_port = self.slider(1024.0, 65535.0, 1.0, 0)
        self.bubble_seconds = self.slider(1.0, 10.0, 0.5, 1, "s")
        self.bubble_token = QLineEdit()
        self.bubble_token.setEchoMode(QLineEdit.Password)
        self.tts_guard_enabled = ToggleSwitch("启用 TTS 防回流 guard（需要 TTS 侧写入本地 guard 文件）")
        self.tts_guard_note = QLabel(
            "Guard file supports fields such as ignore_until, text, tts_text, or recent_texts. "
            "Use it only as a backup; the primary rule is still FenneNote = physical mic, QQ = mixed virtual mic."
        )
        self.tts_guard_note.setWordWrap(True)
        self.tts_guard_note.setStyleSheet("color: #6f5962;")
        misc_form.addRow("", self.bubble_enabled)
        misc_form.addRow("气泡端口", self.bubble_port)
        misc_form.addRow("气泡秒数", self.bubble_seconds)
        misc_form.addRow("气泡令牌", self.bubble_token)
        self.add_help_row(misc_form, "TTS guard", self.tts_guard_enabled, "可选防回流保护。TTS 侧写入 guard 文件后，FenneNote 可以临时跳过播放中的语音，或丢弃和最近 TTS 文本很像的转写。")
        misc_form.addRow("防回流说明", self.tts_guard_note)
        layout.addWidget(misc)

        self.output_dir_browse_button.clicked.connect(lambda: self.browse_directory_into(self.output_dir_edit, DEFAULT_CONFIG["output_dir"]))
        self.output_dir_open_button.clicked.connect(self.open_output_folder)
        self.cache_dir_browse_button.clicked.connect(lambda: self.browse_directory_into(self.cache_dir_edit, DEFAULT_CONFIG["cache_dir"]))
        self.cache_dir_open_button.clicked.connect(self.open_cache_folder)
        self.open_audio_button.clicked.connect(self.open_audio_folder)
        self.open_config_button.clicked.connect(self.open_config_file)
        self.show_log_button.clicked.connect(lambda: self.set_log_drawer_expanded(True))
        self.cache_dir_edit.textChanged.connect(self.refresh_path_labels)
        self.output_dir_edit.textChanged.connect(self.refresh_path_labels)

    def build_voice_output_page(self) -> None:
        _, _, layout = self.page()
        layout.addWidget(self.page_banner(
            "OumuQ 输出端",
            "这里只管角色音色、播放测试和 TTS guard。Codex 或 RabiRoute 发来的播放包会先到 FenneNote，再交给 OumuQ。",
        ))

        group = QGroupBox("OumuQ 角色播放")
        form = QFormLayout(group)
        self.oumuq_url = QLineEdit()
        self.oumuq_registry_path = QLineEdit()
        self.oumuq_registry_browse_button = QPushButton("选择")
        self.oumuq_refresh_roles_button = QPushButton("刷新角色")
        self.oumuq_character_id = QComboBox()
        self.oumuq_character_id.setEditable(True)
        self.oumuq_language = QComboBox()
        self.oumuq_language.addItem("自动识别", "auto")
        self.oumuq_language.addItem("中文", "Chinese")
        self.oumuq_language.addItem("日语", "Japanese")
        self.oumuq_language.addItem("英语", "English")
        self.oumuq_character_name = QLabel("未选择")
        self.oumuq_character_name.setWordWrap(True)
        self.oumuq_character_name.setStyleSheet(f"color: {THEME['muted']};")
        self.oumuq_play_check = ToggleSwitch("请求 OumuQ/worker 播放")
        self.oumuq_guard_seconds = self.slider(1.0, 60.0, 0.5, 1, "s")
        self.oumuq_test_text = QPlainTextEdit()
        self.oumuq_test_text.setPlaceholderText("输入一小句测试语音文本")
        self.oumuq_test_text.setFixedHeight(88)
        self.oumuq_play_button = QPushButton("提交播放")
        self.oumuq_status = QLabel("未播放")
        self.oumuq_status.setWordWrap(True)
        self.oumuq_status.setStyleSheet(f"color: {THEME['muted']};")
        self.oumuq_roles: list[dict] = []
        self.add_help_row(
            form,
            "OumuQ URL",
            self.oumuq_url,
            "OumuQ 路由层 /api/speak 或兼容 worker /speak 地址。默认示例为 http://127.0.0.1:8780/api/speak。",
        )
        self.add_help_row(
            form,
            "角色注册表",
            self.path_row(self.oumuq_registry_path, self.oumuq_registry_browse_button, self.oumuq_refresh_roles_button),
            "读取 OumuQ 的 voice-references/reference-index.json，用来显示角色名、音色和引擎信息。",
        )
        self.add_help_row(form, "角色/音色 ID", self.oumuq_character_id, "传给 OumuQ 的 character_id。可以从注册表选择，也可以手动输入。")
        self.add_help_row(form, "语音语言", self.oumuq_language, "默认自动识别测试文本语言；也可以手动指定中文、日语或英语。")
        form.addRow("角色名", self.oumuq_character_name)
        form.addRow("", self.oumuq_play_check)
        self.add_help_row(
            form,
            "guard 秒数",
            self.oumuq_guard_seconds,
            "提交播放前写入 TTS guard 的保护窗口。实际防回流仍要求应用页启用 TTS guard。",
        )
        form.addRow("测试文本", self.oumuq_test_text)
        form.addRow("播放", self.oumuq_play_button)
        form.addRow("状态", self.oumuq_status)
        layout.addWidget(group)

        clone_group = QGroupBox("本人声音克隆")
        clone_form = QFormLayout(clone_group)
        self.clone_character_id = QLineEdit()
        self.clone_character_id.setPlaceholderText("例如 user_voice")
        self.clone_display_name = QLineEdit()
        self.clone_display_name.setPlaceholderText("例如 我的声音")
        self.clone_provider = QComboBox()
        self.clone_provider.addItem("千问 / DashScope CosyVoice", "qwen_api")
        self.clone_provider.addItem("本地 IndexTTS2", "indextts2")
        self.clone_reference_language = QComboBox()
        self.clone_reference_language.addItem("中文", "zh")
        self.clone_reference_language.addItem("日语", "ja")
        self.clone_reference_language.addItem("英语", "en")
        self.clone_speech_language = QComboBox()
        self.clone_speech_language.addItem("中文", "Chinese")
        self.clone_speech_language.addItem("日语", "Japanese")
        self.clone_speech_language.addItem("英语", "English")
        self.clone_target_model = QLineEdit("cosyvoice-v3.5-plus")
        self.clone_voice_id = QLineEdit()
        self.clone_voice_id.setPlaceholderText("千问注册后得到的 voice_id，可先留空")
        self.clone_audio_url = QLineEdit()
        self.clone_audio_url.setPlaceholderText("云端注册需要可访问的参考音频 URL；本地测试可先留空")
        self.clone_sample_path = QLineEdit()
        self.clone_sample_path.setReadOnly(True)
        self.clone_record_seconds = self.slider(15.0, 20.0, 1.0, 0, "s")
        self.clone_sample_preset = QComboBox()
        for preset_key, preset in VOICE_CLONE_SAMPLE_PRESETS.items():
            self.clone_sample_preset.addItem(str(preset["label"]), preset_key)
        self.clone_read_aloud_text = QPlainTextEdit()
        self.clone_read_aloud_text.setPlainText(VOICE_CLONE_READ_ALOUD_TEXT)
        self.clone_read_aloud_text.setFixedHeight(86)
        self.clone_record_button = QPushButton("录制样本")
        self.clone_import_button = QPushButton("导入样本")
        self.clone_write_button = QPushButton("写入 OumuQ 角色")
        self.clone_enroll_button = QPushButton("执行千问克隆")
        self.clone_open_samples_button = QPushButton("打开样本目录")
        self.clone_status = QLabel("未准备")
        self.clone_status.setWordWrap(True)
        self.clone_status.setStyleSheet(f"color: {THEME['muted']};")
        self.add_help_row(clone_form, "角色 ID", self.clone_character_id, "写入 OumuQ 的唯一角色/音色标识。建议只用英文、数字、下划线，例如 ak_gamer；之后播放请求会用这个 ID 选择音色。")
        self.add_help_row(clone_form, "角色名", self.clone_display_name, "给人看的名字，会显示在 FenneNote 和 OumuQ 的角色列表里，例如 秋雨。")
        self.add_help_row(clone_form, "克隆方式", self.clone_provider, "千问 / DashScope 会注册云端 voice_id，适合后续稳定复用；本地 IndexTTS2 使用本地样本作为参考音色，并支持真正的 8 维情绪向量。")
        self.add_help_row(clone_form, "样本语言", self.clone_reference_language, "录音样本主要使用的语言。和实际录音一致能提高克隆或参考音色效果。")
        self.add_help_row(clone_form, "输出语言", self.clone_speech_language, "之后用这个角色合成语音时默认传给 TTS worker 的语言。")
        self.add_help_row(clone_form, "目标模型", self.clone_target_model, "云端克隆要注册到的 TTS 模型。声音复刻和后续合成必须使用同一个模型；需要指令控制情绪时建议用 cosyvoice-v3.5-plus。")
        self.add_help_row(clone_form, "voice_id", self.clone_voice_id, "千问克隆成功后得到的云端音色 ID。已有 voice_id 可以直接填；留空则需要执行克隆流程生成。")
        self.add_help_row(clone_form, "参考音频 URL", self.clone_audio_url, "云端克隆通常需要服务端可访问的音频 URL。本地录音会先保存到样本目录；如果没有 URL，需要由 OumuQ 工具上传或处理后再注册。")
        self.add_help_row(clone_form, "本地样本", self.clone_sample_path, "当前录制或导入的本地参考音频路径。写入 OumuQ 角色和本地 IndexTTS2 测试会优先使用它。")
        self.add_help_row(clone_form, "录制时长", self.clone_record_seconds, "按上方克隆方式和目标模型动态限制。千问 CosyVoice 建议 15-20 秒；环境安静、语速自然、没有其他人说话时效果最好。")
        self.add_help_row(clone_form, "样本情绪", self.clone_sample_preset, "选择这次录音的用途。中性样本用于稳定音色；开心、低落、兴奋、生气样本用于后续本地情绪参考或 IndexTTS2 情绪调试。")
        self.add_help_row(clone_form, "朗读文本", self.clone_read_aloud_text, "录制时照着这段读。可以自行改成更顺口的短稿，但要保持单人、自然、无背景声。")
        self.add_help_row(clone_form, "样本", self.path_row(self.clone_record_button, self.clone_import_button, self.clone_open_samples_button), "录制新样本、导入已有 WAV/MP3，或打开保存样本的本地目录。")
        self.add_help_row(clone_form, "档案", self.path_row(self.clone_write_button, self.clone_enroll_button), "写入 OumuQ 角色会更新角色/音色注册表；执行千问克隆会请求 OumuQ 调用云端注册并回写 voice_id。")
        clone_form.addRow("状态", self.clone_status)
        layout.addWidget(clone_group)

        emotion_group = QGroupBox("指令 / 情绪测试")
        emotion_layout = QVBoxLayout(emotion_group)
        emotion_form = QFormLayout()
        self.clone_emotion_mode = QComboBox()
        self.clone_emotion_mode.addItem("自动温和", "auto-vector")
        self.clone_emotion_mode.addItem("手动向量", "vector")
        self.clone_emotion_alpha = self.slider(0.0, 1.0, 0.05, 2)
        self.clone_emotion_alpha.setValue(0.55)
        self.add_help_row(emotion_form, "情绪模式", self.clone_emotion_mode, "自动温和会用一组保守的情绪向量；手动向量会使用下面每个维度的滑块值。")
        self.add_help_row(emotion_form, "情绪强度", self.clone_emotion_alpha, "情绪向量整体混合强度。数值越高，TTS worker 收到的情绪倾向越明显。")
        emotion_layout.addLayout(emotion_form)
        self.clone_emotion_note = QLabel()
        self.clone_emotion_note.setWordWrap(True)
        self.clone_emotion_note.setStyleSheet(f"color: {THEME['muted']};")
        emotion_layout.addWidget(self.clone_emotion_note)
        self.clone_instruction_text = QPlainTextEdit()
        self.clone_instruction_text.setPlaceholderText("例如：用开心、明亮、带一点笑意的自然中文说话，语气轻快但不要夸张。")
        self.clone_instruction_text.setFixedHeight(64)
        self.add_help_row(
            emotion_form,
            "自然语言指令",
            self.clone_instruction_text,
            "千问 / DashScope CosyVoice 使用这个 instruction 控制音调、语速、情感和音色特点；本地 IndexTTS2 仍使用下方情绪向量。",
        )
        self.clone_emotion_sliders: dict[str, NumericSlider] = {}
        emotion_labels = [
            ("happy", "开心"),
            ("angry", "生气"),
            ("sad", "伤心"),
            ("afraid", "害怕"),
            ("disgusted", "厌恶"),
            ("melancholic", "忧郁"),
            ("surprised", "惊讶"),
            ("calm", "平静"),
        ]
        grid = QGridLayout()
        for index, (key, label) in enumerate(emotion_labels):
            slider = self.slider(0.0, 1.0, 0.05, 2)
            if key == "calm":
                slider.setValue(0.25)
            self.clone_emotion_sliders[key] = slider
            grid.addWidget(QLabel(label), index // 2, (index % 2) * 2)
            grid.addWidget(slider, index // 2, (index % 2) * 2 + 1)
        emotion_layout.addLayout(grid)
        self.clone_test_text = QPlainTextEdit()
        self.clone_test_text.setPlaceholderText("输入一小句，用刚写入的 OumuQ 角色做情绪测试")
        self.clone_test_text.setFixedHeight(72)
        self.clone_test_button = QPushButton("用克隆角色测试播放")
        emotion_layout.addWidget(self.clone_test_text)
        emotion_layout.addWidget(self.clone_test_button)
        layout.addWidget(emotion_group)

        note = QLabel(
            "这一页只放 OumuQ 相关扩展：角色注册表、角色音色、播放请求、TTS guard 测试。"
            "声纹识别、说话人字幕和 speaker registry 在左侧“声纹识别”页单独维护。"
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {THEME['muted']};")
        layout.addWidget(note)
        layout.addStretch(1)
        self.oumuq_play_button.clicked.connect(self.submit_oumuq_playback)
        self.oumuq_registry_browse_button.clicked.connect(lambda: self.browse_file_into(self.oumuq_registry_path, DEFAULT_CONFIG["oumuq_registry_path"]))
        self.oumuq_refresh_roles_button.clicked.connect(self.refresh_oumuq_roles)
        self.oumuq_character_id.currentTextChanged.connect(self.update_oumuq_character_name)
        self.clone_provider.currentIndexChanged.connect(self.update_clone_recording_constraints)
        self.clone_provider.currentIndexChanged.connect(self.update_clone_emotion_note)
        self.clone_target_model.textChanged.connect(self.update_clone_recording_constraints)
        self.clone_sample_preset.currentIndexChanged.connect(self.update_clone_sample_preset)
        self.clone_record_button.clicked.connect(self.record_voice_clone_sample)
        self.clone_import_button.clicked.connect(self.import_voice_clone_sample)
        self.clone_open_samples_button.clicked.connect(lambda: self.open_local_path(self.voice_clone_samples_dir()))
        self.clone_write_button.clicked.connect(self.write_oumuq_clone_profile)
        self.clone_enroll_button.clicked.connect(self.run_oumuq_voice_enrollment)
        self.clone_test_button.clicked.connect(self.submit_clone_test_playback)
        self.update_clone_recording_constraints()
        self.update_clone_emotion_note()
        self.update_clone_sample_preset()

    def build_speaker_recognition_page(self) -> None:
        _, _, layout = self.page()
        layout.addWidget(self.page_banner(
            "声音身份",
            "这里维护“谁在说话”的档案。多说话人分离统一走阿里非实时 ASR + diarization，再逐段匹配本地声纹。",
        ))

        speakers = QGroupBox("声纹库")
        speakers_layout = QVBoxLayout(speakers)
        self.speaker_registry_file = QLineEdit()
        self.speaker_refresh_button = QPushButton("刷新声纹列表")
        registry_row = self.path_row(self.speaker_registry_file, self.speaker_refresh_button)
        speakers_layout.addWidget(registry_row)
        self.speaker_table = QTableWidget(0, 7)
        self.speaker_table.setHorizontalHeaderLabels(["Speaker ID", "显示名", "类型", "OumuQ 角色", "样本数", "样本文本", "状态"])
        self.speaker_table.verticalHeader().setVisible(False)
        self.speaker_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.speaker_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.speaker_table.setMinimumHeight(210)
        self.speaker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.speaker_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        speakers_layout.addWidget(self.speaker_table)
        name_row = QHBoxLayout()
        self.speaker_name_edit = QLineEdit()
        self.speaker_name_edit.setPlaceholderText("选中一个声纹档案后，在这里输入显示名")
        self.save_speaker_name_button = QPushButton("保存名称")
        name_row.addWidget(QLabel("名称"), 0)
        name_row.addWidget(self.speaker_name_edit, 1)
        name_row.addWidget(self.save_speaker_name_button, 0)
        speakers_layout.addLayout(name_row)
        self.speaker_sample_note = QPlainTextEdit()
        self.speaker_sample_note.setPlainText(VOICE_CLONE_READ_ALOUD_TEXT)
        self.speaker_sample_note.setFixedHeight(58)
        self.speaker_sample_note.setPlaceholderText("录制或导入声纹样本时，可以在这里写台词/备注，会保存到声纹库。")
        speakers_layout.addWidget(self.speaker_sample_note)
        speaker_buttons = QHBoxLayout()
        self.add_user_speaker_button = QPushButton("添加本人声纹")
        self.import_speaker_sample_button = QPushButton("导入样本")
        self.rebuild_embedding_button = QPushButton("重算声纹")
        self.test_speaker_button = QPushButton("测试识别")
        self.delete_speaker_button = QPushButton("删除声纹")
        self.delete_speaker_button.setObjectName("danger")
        for button in (self.add_user_speaker_button, self.import_speaker_sample_button, self.rebuild_embedding_button, self.test_speaker_button, self.delete_speaker_button):
            speaker_buttons.addWidget(button)
        speaker_buttons.addStretch(1)
        speakers_layout.addLayout(speaker_buttons)
        layout.addWidget(speakers)

        mechanism = QGroupBox("识别机制")
        mechanism_layout = QVBoxLayout(mechanism)
        self.speaker_recognition_enabled = ToggleSwitch("启用声纹识别")
        self.speaker_recognition_enabled.setToolTip("总开关：勾选后才允许对录音片段做本地声纹匹配。单人录音会贴整段 speaker；多说话人需要同时开启说话人字幕。")
        mechanism_layout.addWidget(self.speaker_recognition_enabled)
        self.speaker_auto_enroll_enabled = ToggleSwitch("未匹配时自动创建声纹")
        self.speaker_auto_enroll_enabled.setToolTip("开启后，平时录音里遇到未匹配的声音会自动创建 unknown_* 档案，之后可以在声纹库里改名。关闭后只标记为未知，不写入新档案。")
        mechanism_layout.addWidget(self.speaker_auto_enroll_enabled)
        self.speaker_subtitle_enabled = ToggleSwitch("启用说话人字幕")
        self.speaker_subtitle_enabled.setToolTip("勾选后，API 转写会优先走阿里非实时 ASR + diarization，说话人时间段会在转写预览里分行显示；失败时回退普通转写。")
        mechanism_layout.addWidget(self.speaker_subtitle_enabled)
        threshold_form = QFormLayout()
        self.speaker_match_threshold = self.slider(0.50, 0.99, 0.01, 2)
        self.speaker_match_threshold.setToolTip("声纹相似度达到这个值才算匹配。越高越不容易认错人，但更容易把同一个人判成新声纹。默认 0.92。")
        self.add_help_row(threshold_form, "匹配阈值", self.speaker_match_threshold, "声纹相似度达到这个值才算匹配。越高越不容易认错人，但更容易把同一个人判成新声纹。默认 0.92。")
        mechanism_layout.addLayout(threshold_form)
        mechanism_text = QLabel(
            "当前阶段：已支持声纹库、样本导入、轻量本地 embedding、整段录音自动匹配、未匹配自动建档、TTS 角色映射和阿里非实时说话人分离。\n"
            "多发言人识别链路：开启说话人字幕后，FenneNote 会先按阿里返回的时间段分行，得到“说话人 1/2/3 在哪段说了什么”，再配合本地声纹库把它们匹配成具体人名。"
        )
        mechanism_text.setWordWrap(True)
        mechanism_text.setStyleSheet(f"color: {THEME['muted']};")
        mechanism_layout.addWidget(mechanism_text)
        layout.addWidget(mechanism)

        note = QLabel(
            "这一页只放声纹识别和说话人相关能力。OumuQ 只作为可能的 TTS 样本来源，不属于声纹识别本身。"
            "声纹库建议用单个角色/单个人的干净样本建档；混合录音只用于转写分离，不建议直接当作声纹样本。"
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {THEME['muted']};")
        layout.addWidget(note)
        layout.addStretch(1)
        self.speaker_refresh_button.clicked.connect(self.refresh_speaker_table)
        self.speaker_table.itemSelectionChanged.connect(self.on_speaker_selection_changed)
        self.save_speaker_name_button.clicked.connect(self.save_selected_speaker_name)
        self.add_user_speaker_button.clicked.connect(self.record_user_speaker_sample)
        self.import_speaker_sample_button.clicked.connect(self.import_speaker_sample)
        self.rebuild_embedding_button.clicked.connect(self.rebuild_selected_speaker_embedding)
        self.test_speaker_button.clicked.connect(self.test_speaker_identification)
        self.delete_speaker_button.clicked.connect(self.delete_selected_speaker)

    def build_route_page(self) -> None:
        _, _, layout = self.page()
        layout.addWidget(self.page_banner(
            "RabiRoute 输入端",
            "这里把 FenneNote 的语音转写推给 RabiRoute；文字反写和播放请求则走 RabiRoute 的 FenneNote 输出端回来。",
        ))
        group = QGroupBox("RabiRoute 扩展")
        form = QFormLayout(group)
        self.route_enabled = ToggleSwitch("转写完成后推送到 RabiRoute")
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

    def safe_identifier(self, value: str, fallback: str) -> str:
        cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or fallback

    def voice_clone_samples_dir(self) -> Path:
        character_id = self.safe_identifier(self.clone_character_id.text(), "user_voice") if hasattr(self, "clone_character_id") else "user_voice"
        path = (self.cache_dir_from_ui() / "voice-clone-samples" / character_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clone_recording_seconds_range(self) -> tuple[float, float, str]:
        provider = str(self.clone_provider.currentData() or "qwen_api")
        target_model = self.clone_target_model.text().strip().lower()
        if provider == "qwen_api" and "cosyvoice" in target_model:
            return 15.0, 20.0, "千问 CosyVoice 样本建议 15-20 秒。"
        if provider == "qwen_api":
            return 10.0, 20.0, "云端克隆样本建议控制在 10-20 秒。"
        return 5.0, 60.0, "本地参考音频可更长；建议先用 10-30 秒干净单人语音。"

    def update_clone_recording_constraints(self, *_args: object) -> None:
        if not hasattr(self, "clone_record_seconds"):
            return
        minimum, maximum, note = self.clone_recording_seconds_range()
        self.clone_record_seconds.setRange(minimum, maximum, 1.0)
        if hasattr(self, "clone_status") and self.clone_status.text() in {"未准备", ""}:
            self.clone_status.setText(note)

    def update_clone_emotion_note(self, *_args: object) -> None:
        if not hasattr(self, "clone_emotion_note"):
            return
        provider = str(self.clone_provider.currentData() or "qwen_api")
        if provider == "indextts2":
            self.clone_emotion_note.setText(
                "当前为本地 IndexTTS2：测试播放会把本地样本和 8 维情绪向量直接发给 IndexTTS2 worker。"
            )
        else:
            self.clone_emotion_note.setText(
                "当前为千问 / DashScope CosyVoice：测试播放会发送自然语言 instruction 控制音调、语速、情感和音色特点；8 维向量只作为生成指令的辅助意图。"
            )

    def selected_clone_sample_preset(self) -> tuple[str, dict]:
        preset_key = "neutral"
        if hasattr(self, "clone_sample_preset"):
            value = self.clone_sample_preset.currentData()
            if isinstance(value, str) and value in VOICE_CLONE_SAMPLE_PRESETS:
                preset_key = value
        return preset_key, VOICE_CLONE_SAMPLE_PRESETS[preset_key]

    def update_clone_sample_preset(self, *_args: object) -> None:
        if not hasattr(self, "clone_read_aloud_text"):
            return
        preset_key, preset = self.selected_clone_sample_preset()
        self.clone_read_aloud_text.setPlainText(str(preset["text"]))
        if hasattr(self, "clone_instruction_text"):
            self.clone_instruction_text.setPlainText(str(preset.get("instruction") or ""))
        if hasattr(self, "clone_emotion_mode"):
            self.clone_emotion_mode.setCurrentIndex(1 if preset_key != "neutral" else 0)
        if hasattr(self, "clone_emotion_alpha"):
            self.clone_emotion_alpha.setValue(0.75 if preset_key != "neutral" else 0.55)
        if hasattr(self, "clone_emotion_sliders"):
            order = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
            vector = preset.get("vector", [])
            for index, key in enumerate(order):
                if key in self.clone_emotion_sliders:
                    value = float(vector[index]) if isinstance(vector, list) and index < len(vector) else 0.0
                    self.clone_emotion_sliders[key].setValue(value)
        if hasattr(self, "clone_status"):
            self.clone_status.setText(f"已选择{preset['label']}样本。录音时请按朗读文本的情绪自然表达，不要表演过猛。")

    def write_voice_clone_sample_metadata(
        self,
        path: Path,
        character_id: str,
        display_name: str,
        preset_key: str,
        preset: dict,
        read_aloud_text: str,
        seconds: float,
        sample_rate: int,
    ) -> None:
        metadata = {
            "version": 1,
            "character_id": character_id,
            "display_name": display_name,
            "sample_kind": preset_key,
            "sample_label": preset.get("label"),
            "emotion_vector": preset.get("vector"),
            "emotion_alpha": 0.75 if preset_key != "neutral" else 0.55,
            "read_aloud_text": read_aloud_text,
            "audio_file": str(path),
            "duration_seconds": seconds,
            "sample_rate": sample_rate,
            "created_at": datetime.now().isoformat(),
            "created_by": "FenneNote",
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_wav_samples(self, path: Path, samples: np.ndarray, sample_rate: int) -> None:
        pcm = (np.clip(samples.reshape(-1), -1.0, 1.0) * 32767.0).astype(np.int16)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def record_voice_clone_sample(self) -> None:
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.information(self, "声音克隆", "转写正在运行，请先停止后再录制克隆样本，避免麦克风被同时占用。")
            return
        character_id = self.safe_identifier(self.clone_character_id.text(), "user_voice")
        self.clone_character_id.setText(character_id)
        minimum, maximum, note = self.clone_recording_seconds_range()
        seconds = min(maximum, max(minimum, float(self.clone_record_seconds.value())))
        self.clone_record_seconds.setValue(seconds)
        config = self.collect_config()
        sample_rate = int(config.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
        input_gain = float(config.get("input_gain", 1.0))
        device = self.selected_device_index()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        preset_key, preset = self.selected_clone_sample_preset()
        preset_label = str(preset.get("label") or preset_key)
        display_name = self.clone_display_name.text().strip()
        read_aloud_text = self.clone_read_aloud_text.toPlainText().strip()
        target = self.voice_clone_samples_dir() / f"{character_id}-{preset_key}-{stamp}-{seconds:.0f}s.wav"
        self.clone_record_button.setEnabled(False)
        self.clone_status.setText(f"正在录制 {preset_label}样本 {seconds:.0f} 秒，{note} 请照着朗读文本自然表达。")
        self.stop_level_worker()

        def worker() -> None:
            try:
                samples = sd.rec(int(sample_rate * seconds), samplerate=sample_rate, channels=1, dtype="float32", device=device)
                sd.wait()
                if input_gain != 1.0:
                    samples = np.clip(samples * input_gain, -1.0, 1.0)
                self.write_wav_samples(target, samples, sample_rate)
                self.write_voice_clone_sample_metadata(target, character_id, display_name, preset_key, preset, read_aloud_text, seconds, sample_rate)
                self.voice_clone_sample_saved.emit(True, str(target), f"已录制{preset_label}样本：{target}")
            except Exception as exc:
                self.voice_clone_sample_saved.emit(False, "", f"录制失败：{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def on_voice_clone_sample_saved(self, ok: bool, path: str, message: str) -> None:
        self.clone_record_button.setEnabled(True)
        if hasattr(self, "clone_enroll_button"):
            self.clone_enroll_button.setEnabled(True)
        self.start_level_worker()
        if ok:
            if path.startswith("cosyvoice") or path.startswith("qwen"):
                self.clone_voice_id.setText(path)
                self.refresh_oumuq_roles()
            else:
                self.clone_sample_path.setText(path)
        self.clone_status.setText(message)
        self.append_log(message)

    def import_voice_clone_sample(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "导入声音克隆样本",
            str(self.voice_clone_samples_dir()),
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if not selected:
            return
        source = Path(selected)
        target = self.voice_clone_samples_dir() / source.name
        if source.resolve() != target.resolve():
            target = target.with_name(f"{target.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{target.suffix}")
            shutil.copy2(source, target)
        self.clone_sample_path.setText(str(target))
        self.clone_status.setText(f"已导入克隆样本：{target}")
        self.append_log(f"已导入克隆样本：{target}")

    def load_oumuq_registry(self) -> dict:
        path = self.oumuq_registry_path_from_ui()
        if path.exists():
            try:
                registry = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"读取 OumuQ 注册表失败：{exc}") from exc
            if isinstance(registry, dict):
                registry.setdefault("version", 1)
                registry.setdefault("root", "voice-references")
                registry["characters"] = registry.get("characters") if isinstance(registry.get("characters"), list) else []
                return registry
        return {"version": 1, "root": "voice-references", "characters": []}

    def write_oumuq_registry(self, registry: dict) -> None:
        path = self.oumuq_registry_path_from_ui()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def clone_provider_fields(self) -> tuple[str, str]:
        provider = str(self.clone_provider.currentData() or "qwen_api")
        if provider == "indextts2":
            return "IndexTTS2", "http://127.0.0.1:8766"
        return "Qwen-TTS-API", "http://127.0.0.1:8767"

    def collect_voice_clone_emotion_samples(self) -> list[dict]:
        samples: list[dict] = []
        sample_dir = self.voice_clone_samples_dir()
        for metadata_path in sorted(sample_dir.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            audio_file = str(metadata.get("audio_file") or "").strip()
            if not audio_file or not Path(audio_file).exists():
                continue
            samples.append({
                "sample_kind": metadata.get("sample_kind"),
                "sample_label": metadata.get("sample_label"),
                "audio_file": audio_file,
                "emotion_vector": metadata.get("emotion_vector"),
                "emotion_alpha": metadata.get("emotion_alpha"),
                "duration_seconds": metadata.get("duration_seconds"),
                "created_at": metadata.get("created_at"),
            })
        return samples

    def write_oumuq_clone_profile(self) -> None:
        character_id = self.safe_identifier(self.clone_character_id.text(), "user_voice")
        display_name = self.clone_display_name.text().strip() or "我的声音"
        sample_path = self.clone_sample_path.text().strip()
        if not sample_path:
            QMessageBox.warning(self, "声音克隆", "请先录制或导入一段参考音频。")
            return
        engine, worker_url = self.clone_provider_fields()
        registry = self.load_oumuq_registry()
        characters = registry.setdefault("characters", [])
        character = {
            "id": character_id,
            "name": display_name,
            "display_name_zh": display_name,
            "tts_engine": engine,
            "worker_url": worker_url,
            "speech_language": str(self.clone_speech_language.currentData() or "Chinese"),
            "visible_language": "Chinese",
            "style_summary": "User-owned cloned voice prepared from local FenneNote reference audio.",
            "style_summary_zh": "由 FenneNote 本地参考音频准备的用户本人克隆音色。",
            "fallback_prompt_audio": sample_path,
            "api_voice_id": self.clone_voice_id.text().strip(),
            "api_clone_audio_url": self.clone_audio_url.text().strip(),
            "api_clone_target_model": self.clone_target_model.text().strip() or "cosyvoice-v3.5-plus",
            "api_clone_language_hint": str(self.clone_reference_language.currentData() or "zh"),
            "api_voice_instructions": self.clone_instruction_text.toPlainText().strip() if hasattr(self, "clone_instruction_text") else "",
            "send_instructions_by_default": False,
            "created_by": "FenneNote",
            "updated_at": datetime.now().isoformat(),
        }
        emotion_samples = self.collect_voice_clone_emotion_samples()
        if emotion_samples:
            character["emotion_samples"] = emotion_samples
            character["emotion_sample_folder"] = str(self.voice_clone_samples_dir())
        character = {key: value for key, value in character.items() if value not in ("", None)}
        for index, item in enumerate(characters):
            if isinstance(item, dict) and str(item.get("id", "")).strip() == character_id:
                characters[index] = {**item, **character}
                break
        else:
            characters.append(character)
        self.write_oumuq_registry(registry)
        self.write_voice_clone_request(character, sample_path)
        self.clone_status.setText(f"已写入 OumuQ 角色：{character_id}。如果 voice_id 为空，下一步由 OumuQ 克隆工具注册。")
        self.append_log(f"已写入 OumuQ 克隆角色：{character_id}")
        self.oumuq_character_id.setCurrentText(character_id)
        self.refresh_oumuq_roles()

    def write_voice_clone_request(self, character: dict, sample_path: str) -> None:
        request_dir = (self.cache_dir_from_ui() / "voice-clone-requests").resolve()
        request_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "version": 1,
            "status": "pending_voice_enrollment",
            "provider": str(self.clone_provider.currentData() or "qwen_api"),
            "character": character,
            "reference_audio_path": sample_path,
            "reference_audio_url": self.clone_audio_url.text().strip(),
            "created_at": datetime.now().isoformat(),
            "next_step": "OumuQ worker/tool should upload or use reference_audio_url, call provider voice enrollment, then write api_voice_id back to reference-index.json.",
        }
        path = request_dir / f"{character['id']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def oumuq_api_base_url(self) -> str:
        url = (self.oumuq_url.text().strip() or DEFAULT_CONFIG["oumuq_url"]).rstrip("/")
        for suffix in ("/api/speak", "/api/batch"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url

    def run_oumuq_voice_enrollment(self) -> None:
        character_id = self.safe_identifier(self.clone_character_id.text(), "user_voice")
        audio_url = self.clone_audio_url.text().strip()
        if not audio_url and self.clone_provider.currentData() == "qwen_api":
            QMessageBox.warning(
                self,
                "声音克隆",
                "千问 CosyVoice 克隆需要一个可访问的参考音频 URL。请先上传样本或填写签名 URL。",
            )
            return
        self.clone_enroll_button.setEnabled(False)
        self.clone_status.setText("正在请求 OumuQ 执行千问克隆...")
        payload = {
            "character_id": character_id,
            "reference_audio_url": audio_url or None,
            "reference_audio_path": self.clone_sample_path.text().strip() or None,
            "target_model": self.clone_target_model.text().strip() or "cosyvoice-v3-plus",
            "language_hints": [self.clone_reference_language.currentData()],
        }
        payload = {key: value for key, value in payload.items() if value not in ("", None, [])}
        endpoint = self.oumuq_api_base_url() + "/api/voice-clone/enroll"

        def worker() -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                endpoint,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json; charset=utf-8", "User-Agent": "FenneNote"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read().decode("utf-8", errors="replace")
                result = json.loads(body)
                voice_id = str(result.get("api_voice_id", "")).strip()
                message = f"千问克隆完成：{voice_id}" if voice_id else f"千问克隆返回：{body[:300]}"
                self.voice_clone_sample_saved.emit(True, voice_id, message)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self.voice_clone_sample_saved.emit(False, "", f"千问克隆失败：HTTP {exc.code} {body[:300]}")
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                self.voice_clone_sample_saved.emit(False, "", f"千问克隆失败：{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def selected_clone_emotion_vector(self) -> list[float]:
        order = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
        return [round(float(self.clone_emotion_sliders[key].value()), 3) for key in order]

    def submit_clone_test_playback(self) -> None:
        text = self.clone_test_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "声音克隆", "请先输入测试文本。")
            return
        character_id = self.safe_identifier(self.clone_character_id.text() or self.selected_oumuq_character_id(), "user_voice")
        provider = str(self.clone_provider.currentData() or "qwen_api")
        sample_path = self.clone_sample_path.text().strip()
        if provider == "indextts2" and not sample_path:
            QMessageBox.warning(self, "声音克隆", "本地 IndexTTS2 情绪向量测试需要先录制或导入一段本地样本。")
            return
        payload = {
            "text": text,
            "play": True,
            "character_id": character_id,
            "language": self.clone_speech_language.currentData() or "Chinese",
            "emotion_mode": self.clone_emotion_mode.currentData() or "auto-vector",
            "emotion_alpha": round(float(self.clone_emotion_alpha.value()), 2),
        }
        if provider == "indextts2":
            payload["worker_url"] = "http://127.0.0.1:8766"
            payload["prompt_audio"] = sample_path
        else:
            instruction = self.clone_instruction_text.toPlainText().strip() if hasattr(self, "clone_instruction_text") else ""
            if instruction:
                payload["instructions"] = instruction
                payload["send_instructions"] = True
                payload["emotion_text"] = instruction
        if payload["emotion_mode"] == "vector":
            payload["emotion_vector"] = self.selected_clone_emotion_vector()
        self.dispatch_playback_request(payload, "克隆声音测试")

    def load_oumuq_characters(self) -> list[dict]:
        path = self.oumuq_registry_path_from_ui()
        if not path.exists():
            return []
        try:
            registry = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.oumuq_status.setText(f"读取角色注册表失败：{exc}")
            return []
        characters = registry.get("characters", []) if isinstance(registry, dict) else []
        return [item for item in characters if isinstance(item, dict)]

    def refresh_oumuq_roles(self) -> None:
        current = self.oumuq_character_id.currentText().strip()
        self.oumuq_roles = self.load_oumuq_characters()
        self.oumuq_character_id.blockSignals(True)
        self.oumuq_character_id.clear()
        for character in self.oumuq_roles:
            character_id = str(character.get("id", "")).strip()
            if not character_id:
                continue
            display = str(character.get("display_name_zh") or character.get("display_name") or character.get("name") or character_id)
            self.oumuq_character_id.addItem(f"{display} ({character_id})", character_id)
        if current:
            matched = False
            for index in range(self.oumuq_character_id.count()):
                if self.oumuq_character_id.itemData(index) == current or self.oumuq_character_id.itemText(index) == current:
                    self.oumuq_character_id.setCurrentIndex(index)
                    matched = True
                    break
            if not matched:
                self.oumuq_character_id.setCurrentText(current)
        self.oumuq_character_id.blockSignals(False)
        self.update_oumuq_character_name()
        self.refresh_speaker_table()
        self.oumuq_status.setText(f"已读取 {len(self.oumuq_roles)} 个 OumuQ 角色。")

    def selected_oumuq_character_id(self) -> str:
        data = self.oumuq_character_id.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = self.oumuq_character_id.currentText().strip()
        if text.endswith(")") and "(" in text:
            return text.rsplit("(", 1)[-1].rstrip(")").strip()
        return text

    def set_combo_data(self, combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data == value or str(data) == str(value):
                combo.setCurrentIndex(index)
                return

    def update_oumuq_character_name(self, *_unused) -> None:
        character_id = self.selected_oumuq_character_id()
        for character in self.oumuq_roles:
            if str(character.get("id", "")).strip() != character_id:
                continue
            display = str(character.get("display_name_zh") or character.get("display_name") or character.get("name") or character_id)
            engine = str(character.get("tts_engine", "")).strip()
            language = str(character.get("speech_language", "")).strip()
            self.oumuq_character_name.setText(f"{display} / {character_id} / {engine or '未标注引擎'} / {language or '未标注语言'}")
            return
        self.oumuq_character_name.setText(character_id or "未选择")

    def speaker_rows(self) -> list[dict]:
        rows = self.load_speaker_registry().get("speakers", [])
        if not any(row.get("id") == "user_main" for row in rows):
            rows.insert(0, {
                "id": "user_main",
                "display_name": "用户本人",
                "kind": "human",
                "character_id": "",
                "samples": [],
                "status": "待采样",
            })
        known_ids = {str(row.get("id", "")) for row in rows}
        for character in self.oumuq_roles:
            character_id = str(character.get("id", "")).strip()
            if not character_id:
                continue
            speaker_id = f"tts_{character_id}"
            if speaker_id in known_ids:
                continue
            display = str(character.get("display_name_zh") or character.get("display_name") or character.get("name") or character_id)
            rows.append({
                "id": speaker_id,
                "display_name": f"{display} TTS",
                "kind": "tts",
                "character_id": character_id,
                "samples": [],
                "status": "自动映射，待导入样本/建模",
            })
        return sorted(rows, key=self.speaker_sort_key)

    def speaker_sort_key(self, row: dict) -> tuple:
        speaker_id = str(row.get("id", "")).strip()
        kind = str(row.get("kind", "")).strip().lower()
        display_name = str(row.get("display_name", "")).strip()
        character_id = str(row.get("character_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        samples = row.get("samples", [])
        sample_count = len(samples) if isinstance(samples, list) else 0
        if speaker_id == "user_main":
            group = 0
        elif kind == "human":
            group = 1
        elif speaker_id.startswith("unknown") or kind in {"unknown", ""}:
            group = 2
        elif kind == "tts" or speaker_id.startswith("tts_") or speaker_id.startswith("tts:"):
            group = 3
        else:
            group = 4
        is_named = bool(display_name) and display_name.lower() not in {speaker_id.lower(), "unknown"}
        return (group, 0 if is_named else 1, -sample_count, display_name.lower(), character_id.lower(), speaker_id.lower(), status)

    def load_speaker_registry(self) -> dict:
        path = self.speaker_registry_path_from_ui()
        if not path.exists():
            return {"version": 1, "speakers": []}
        try:
            registry = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.append_log(f"读取声纹库失败：{exc}")
            return {"version": 1, "speakers": []}
        if not isinstance(registry, dict):
            return {"version": 1, "speakers": []}
        speakers = registry.get("speakers", [])
        registry["speakers"] = [item for item in speakers if isinstance(item, dict)] if isinstance(speakers, list) else []
        registry.setdefault("version", 1)
        return registry

    def write_speaker_registry(self, registry: dict) -> None:
        path = self.speaker_registry_path_from_ui()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def speaker_profile_dir(self, speaker_id: str) -> Path:
        safe_id = self.safe_identifier(speaker_id, "unknown")
        path = (self.speaker_registry_path_from_ui().parent / safe_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def speaker_embedding_path(self, speaker_id: str) -> Path:
        return self.speaker_profile_dir(speaker_id) / "embedding.npy"

    def read_audio_file_mono(self, path: Path) -> tuple[np.ndarray, int]:
        suffix = path.suffix.lower()
        if suffix == ".wav":
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            if sample_width == 1:
                audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError(f"不支持的 WAV 位宽：{sample_width}")
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio.astype(np.float32), int(sample_rate)

        try:
            import av
        except ImportError as exc:
            raise RuntimeError("读取非 WAV 音频需要 av 依赖。") from exc
        chunks: list[np.ndarray] = []
        sample_rate = 0
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise RuntimeError("文件里没有音频流。")
            for frame in container.decode(stream):
                array = frame.to_ndarray().astype(np.float32)
                if array.ndim == 2:
                    array = array.mean(axis=0 if array.shape[0] <= array.shape[1] else 1)
                chunks.append(array.reshape(-1))
                sample_rate = int(frame.sample_rate or stream.rate or sample_rate)
        if not chunks or not sample_rate:
            raise RuntimeError("没有解码到可用音频。")
        audio = np.concatenate(chunks).astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:
            audio = audio / max(peak, 1.0)
        return audio, sample_rate

    def speaker_embedding_from_audio(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < sample_rate * 0.5:
            raise RuntimeError("样本太短，至少需要 0.5 秒语音。")
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak
        frame_size = max(256, int(sample_rate * 0.025))
        hop = max(128, int(sample_rate * 0.010))
        if audio.size < frame_size:
            audio = np.pad(audio, (0, frame_size - audio.size))
        frame_count = 1 + max(0, (audio.size - frame_size) // hop)
        if frame_count <= 0:
            raise RuntimeError("样本太短，无法计算声纹。")
        window = np.hamming(frame_size).astype(np.float32)
        bands = 32
        vectors: list[np.ndarray] = []
        energies: list[float] = []
        for index in range(frame_count):
            start = index * hop
            frame = audio[start:start + frame_size]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            energy = float(np.sqrt(np.mean(frame * frame)))
            if energy < 0.012:
                continue
            spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
            freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
            mask = (freqs >= 80.0) & (freqs <= min(7600.0, sample_rate / 2.0))
            selected = spectrum[mask]
            if selected.size < bands:
                continue
            splits = np.array_split(selected, bands)
            vectors.append(np.log1p(np.array([float(np.mean(part)) for part in splits], dtype=np.float32)))
            energies.append(energy)
        if not vectors:
            raise RuntimeError("没有检测到足够清晰的语音帧。")
        matrix = np.vstack(vectors)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        energy_values = np.array(energies, dtype=np.float32)
        extra = np.array([
            float(np.mean(energy_values)),
            float(np.std(energy_values)),
            float(np.percentile(energy_values, 75)),
            float(np.percentile(energy_values, 95)),
        ], dtype=np.float32)
        embedding = np.concatenate([mean, std, extra]).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise RuntimeError("声纹向量为空。")
        return embedding / norm

    def speaker_embedding_from_file(self, path: Path) -> np.ndarray:
        samples, sample_rate = self.read_audio_file_mono(path)
        return self.speaker_embedding_from_audio(samples, sample_rate)

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0:
            return 0.0
        return float(np.dot(left, right) / denom)

    def selected_speaker_id(self) -> str:
        items = self.speaker_table.selectedItems()
        if not items:
            return ""
        row = items[0].row()
        item = self.speaker_table.item(row, 0)
        return item.text().strip() if item else ""

    def selected_speaker_kind_and_character(self) -> tuple[str, str]:
        items = self.speaker_table.selectedItems()
        if not items:
            return "unknown", ""
        row = items[0].row()
        kind_item = self.speaker_table.item(row, 2)
        character_item = self.speaker_table.item(row, 3)
        return (
            kind_item.text().strip() if kind_item else "unknown",
            character_item.text().strip() if character_item else "",
        )

    def sample_path_text(self, sample) -> str:
        if isinstance(sample, dict):
            return str(sample.get("path") or sample.get("file") or "").strip()
        return str(sample).strip()

    def sample_metadata_summary(self, profile: dict) -> str:
        metadata = profile.get("sample_metadata", [])
        texts: list[str] = []
        if isinstance(metadata, list):
            for item in metadata:
                if not isinstance(item, dict):
                    continue
                text = str(
                    item.get("transcript_text")
                    or item.get("read_aloud_text")
                    or item.get("text")
                    or item.get("note")
                    or ""
                ).strip()
                if text and text not in texts:
                    texts.append(text)
        if texts:
            return " / ".join(texts)
        if isinstance(metadata, list) and metadata:
            latest = metadata[-1]
            if isinstance(latest, dict):
                text = str(latest.get("source") or "").strip()
                if text:
                    return text
        samples = profile.get("samples", [])
        if isinstance(samples, list) and samples:
            names: list[str] = []
            for sample in samples:
                path_text = self.sample_path_text(sample)
                if path_text:
                    names.append(Path(path_text).name)
            if names:
                return " / ".join(names)
        return ""

    def on_speaker_selection_changed(self) -> None:
        items = self.speaker_table.selectedItems()
        if not items:
            self.speaker_name_edit.clear()
            return
        row = items[0].row()
        name = self.speaker_table.item(row, 1)
        self.speaker_name_edit.setText(name.text() if name else "")

    def upsert_speaker_profile(self, profile: dict) -> None:
        registry = self.load_speaker_registry()
        speakers = registry.setdefault("speakers", [])
        speaker_id = str(profile.get("id", "")).strip()
        if not speaker_id:
            return
        for index, speaker in enumerate(speakers):
            if str(speaker.get("id", "")).strip() == speaker_id:
                merged = {**speaker, **profile}
                existing_samples = speaker.get("samples", [])
                new_samples = profile.get("samples", [])
                if isinstance(existing_samples, list) or isinstance(new_samples, list):
                    merged["samples"] = list(dict.fromkeys([
                        *([self.sample_path_text(item) for item in existing_samples] if isinstance(existing_samples, list) else []),
                        *([self.sample_path_text(item) for item in new_samples] if isinstance(new_samples, list) else []),
                    ]))
                existing_metadata = speaker.get("sample_metadata", [])
                new_metadata = profile.get("sample_metadata", [])
                if isinstance(existing_metadata, list) or isinstance(new_metadata, list):
                    merged["sample_metadata"] = [
                        *(existing_metadata if isinstance(existing_metadata, list) else []),
                        *(new_metadata if isinstance(new_metadata, list) else []),
                    ]
                speakers[index] = merged
                self.write_speaker_registry(registry)
                return
        speakers.append(profile)
        self.write_speaker_registry(registry)

    def add_speaker_sample(
        self,
        speaker_id: str,
        source_path: Path,
        kind: str,
        character_id: str = "",
        display_name: str = "",
        read_aloud_text: str = "",
        source: str = "import",
    ) -> Path:
        target_dir = self.speaker_profile_dir(speaker_id)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = target_dir / f"{speaker_id}-{stamp}{source_path.suffix.lower() or '.wav'}"
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        metadata = {
            "path": str(target),
            "source": source,
            "read_aloud_text": read_aloud_text.strip(),
            "recorded_at": datetime.now().isoformat(),
            "original_file": str(source_path),
        }
        self.upsert_speaker_profile({
            "id": speaker_id,
            "display_name": display_name or speaker_id,
            "kind": kind or "unknown",
            "character_id": character_id,
            "samples": [str(target)],
            "sample_metadata": [metadata],
            "status": "sampled_unmodeled",
            "updated_at": datetime.now().isoformat(),
        })
        return target

    def profile_for_speaker_id(self, speaker_id: str) -> dict:
        for profile in self.speaker_rows():
            if str(profile.get("id", "")).strip() == speaker_id:
                return profile
        return {}

    def rebuild_speaker_embedding(self, speaker_id: str) -> dict:
        profile = self.profile_for_speaker_id(speaker_id)
        samples = [Path(self.sample_path_text(item)) for item in profile.get("samples", []) if self.sample_path_text(item)]
        existing_samples = [path for path in samples if path.exists()]
        if not existing_samples:
            raise RuntimeError("这个声纹档案还没有可用样本。")
        embeddings = [self.speaker_embedding_from_file(path) for path in existing_samples]
        embedding = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise RuntimeError("声纹向量为空。")
        embedding = embedding / norm
        embedding_path = self.speaker_embedding_path(speaker_id)
        np.save(str(embedding_path), embedding)
        updated = {
            **profile,
            "id": speaker_id,
            "samples": [str(path) for path in existing_samples],
            "embedding_file": str(embedding_path),
            "embedding_model": "local_spectral_v1",
            "embedding_sample_count": len(existing_samples),
            "status": "modeled",
            "updated_at": datetime.now().isoformat(),
        }
        self.upsert_speaker_profile(updated)
        return updated

    def record_user_speaker_sample(self) -> None:
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.information(self, "声纹库", "转写正在运行，请先停止后再录制本人声纹样本，避免麦克风被同时占用。")
            return
        speaker_id = "user_main"
        seconds = 12.0
        config = self.collect_config()
        sample_rate = int(config.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
        input_gain = float(config.get("input_gain", 1.0))
        device = self.selected_device_index()
        target = self.speaker_profile_dir(speaker_id) / f"{speaker_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        read_aloud_text = self.speaker_sample_note.toPlainText().strip() if hasattr(self, "speaker_sample_note") else ""
        self.add_user_speaker_button.setEnabled(False)
        self.stop_level_worker()
        self.status_label.setText("正在录制本人声纹样本")

        def worker() -> None:
            try:
                samples = sd.rec(int(sample_rate * seconds), samplerate=sample_rate, channels=1, dtype="float32", device=device)
                sd.wait()
                if input_gain != 1.0:
                    samples = np.clip(samples * input_gain, -1.0, 1.0)
                self.write_wav_samples(target, samples, sample_rate)
                self.upsert_speaker_profile({
                    "id": speaker_id,
                    "display_name": "用户本人",
                    "kind": "human",
                    "character_id": "",
                    "samples": [str(target)],
                    "sample_metadata": [{
                        "path": str(target),
                        "source": "speaker_recording",
                        "read_aloud_text": read_aloud_text,
                        "recorded_at": datetime.now().isoformat(),
                        "duration_seconds": seconds,
                        "sample_rate": sample_rate,
                    }],
                    "status": "sampled_unmodeled",
                    "updated_at": datetime.now().isoformat(),
                })
                self.rebuild_speaker_embedding(speaker_id)
                self.speaker_operation_finished.emit(True, f"已录制并建模本人声纹样本：{target}")
            except Exception as exc:
                self.speaker_operation_finished.emit(False, f"录制本人声纹失败：{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def on_speaker_operation_finished(self, ok: bool, message: str) -> None:
        self.add_user_speaker_button.setEnabled(True)
        self.start_level_worker()
        self.refresh_speaker_table()
        self.status_label.setText("声纹操作完成" if ok else "声纹操作失败")
        self.append_log(message)
        if not ok:
            QMessageBox.warning(self, "声纹库", message)

    def import_speaker_sample(self) -> None:
        speaker_id = self.selected_speaker_id()
        if not speaker_id:
            QMessageBox.warning(self, "声纹库", "请先选中一个声纹档案。")
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "导入声纹样本",
            str(self.audio_dir_from_ui()),
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if not selected:
            return
        kind, character_id = self.selected_speaker_kind_and_character()
        name = self.speaker_name_edit.text().strip() or speaker_id
        read_aloud_text = self.speaker_sample_note.toPlainText().strip() if hasattr(self, "speaker_sample_note") else ""
        try:
            target = self.add_speaker_sample(speaker_id, Path(selected), kind, character_id, name, read_aloud_text, "manual_import")
            self.rebuild_speaker_embedding(speaker_id)
        except Exception as exc:
            QMessageBox.warning(self, "声纹库", f"导入或建模失败：{exc}")
            return
        self.refresh_speaker_table()
        self.append_log(f"已导入并建模声纹样本：{speaker_id} -> {target}")

    def rebuild_selected_speaker_embedding(self) -> None:
        speaker_id = self.selected_speaker_id()
        if not speaker_id:
            QMessageBox.warning(self, "声纹库", "请先选中一个声纹档案。")
            return
        try:
            profile = self.rebuild_speaker_embedding(speaker_id)
        except Exception as exc:
            QMessageBox.warning(self, "声纹库", f"重算失败：{exc}")
            return
        self.refresh_speaker_table()
        self.append_log(f"已重算声纹：{speaker_id}，样本数 {profile.get('embedding_sample_count', 0)}")

    def delete_selected_speaker(self) -> None:
        speaker_id = self.selected_speaker_id()
        if not speaker_id:
            QMessageBox.warning(self, "声纹库", "请先选中一个声纹档案。")
            return
        profile = self.profile_for_speaker_id(speaker_id)
        display_name = str(profile.get("display_name") or speaker_id)
        answer = QMessageBox.question(
            self,
            "删除声纹",
            f"确定删除声纹档案“{display_name}”（{speaker_id}）吗？\n\n会移除声纹库记录，并删除这个档案目录里的样本和 embedding。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        registry = self.load_speaker_registry()
        speakers = registry.get("speakers", [])
        if isinstance(speakers, list):
            registry["speakers"] = [
                speaker for speaker in speakers
                if str(speaker.get("id", "")).strip() != speaker_id
            ]
            self.write_speaker_registry(registry)
        profile_dir = self.speaker_profile_dir(speaker_id)
        if profile_dir.exists():
            try:
                shutil.rmtree(profile_dir)
            except OSError as exc:
                self.append_log(f"删除声纹目录失败：{profile_dir} / {exc}")
        self.speaker_name_edit.clear()
        self.refresh_speaker_table()
        self.append_log(f"已删除声纹档案：{speaker_id}")

    def identify_speaker_file(self, path: Path) -> tuple[dict | None, float, list[tuple[dict, float]]]:
        query = self.speaker_embedding_from_file(path)
        candidates: list[tuple[dict, float]] = []
        for profile in self.speaker_rows():
            embedding_file = str(profile.get("embedding_file", "")).strip()
            if not embedding_file or not Path(embedding_file).exists():
                continue
            try:
                embedding = np.load(embedding_file)
            except Exception:
                continue
            candidates.append((profile, self.cosine_similarity(query, embedding.astype(np.float32))))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if not candidates:
            return None, 0.0, []
        return candidates[0][0], candidates[0][1], candidates

    def test_speaker_identification(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择要测试识别的音频",
            str(self.audio_dir_from_ui()),
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if not selected:
            return
        try:
            best, score, candidates = self.identify_speaker_file(Path(selected))
        except Exception as exc:
            QMessageBox.warning(self, "声纹库", f"测试识别失败：{exc}")
            return
        if not best:
            QMessageBox.information(self, "声纹库", "还没有已建模的声纹档案。请先导入样本并重算声纹。")
            return
        top_lines = []
        for profile, candidate_score in candidates[:5]:
            top_lines.append(f"{profile.get('display_name') or profile.get('id')} ({profile.get('id')}): {candidate_score:.3f}")
        message = "最相似：{} / {} / {:.3f}\n\nTop 匹配：\n{}".format(
            best.get("display_name") or best.get("id"),
            best.get("id"),
            score,
            "\n".join(top_lines),
        )
        QMessageBox.information(self, "声纹测试结果", message)
        self.append_log("声纹测试：" + message.replace("\n", " | "))

    def save_selected_speaker_name(self) -> None:
        speaker_id = self.selected_speaker_id()
        name = self.speaker_name_edit.text().strip()
        if not speaker_id:
            QMessageBox.warning(self, "声纹库", "请先在声纹库里选中一行。")
            return
        if not name:
            QMessageBox.warning(self, "声纹库", "请输入声纹档案名称。")
            return
        selected_row = self.speaker_table.selectedItems()[0].row()
        kind_item = self.speaker_table.item(selected_row, 2)
        character_item = self.speaker_table.item(selected_row, 3)
        self.upsert_speaker_profile({
            "id": speaker_id,
            "display_name": name,
            "kind": kind_item.text().strip() if kind_item else "unknown",
            "character_id": character_item.text().strip() if character_item else "",
            "status": "named",
            "updated_at": datetime.now().isoformat(),
        })
        self.refresh_speaker_table()
        self.append_log(f"已保存声纹档案名称：{speaker_id} -> {name}")

    def refresh_speaker_table(self) -> None:
        rows = self.speaker_rows()
        self.speaker_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            samples = row.get("samples", [])
            sample_count = len(samples) if isinstance(samples, list) else 0
            values = [
                str(row.get("id", "")),
                str(row.get("display_name", "")),
                str(row.get("kind", "")),
                str(row.get("character_id", "")),
                str(sample_count),
                self.sample_metadata_summary(row),
                str(row.get("status", "未建模")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 5 and value:
                    item.setToolTip(value)
                if column == 6 and ("未建模" in value or "待" in value):
                    item.setForeground(QColor(THEME["muted"]))
                self.speaker_table.setItem(row_index, column, item)
        self.speaker_table.resizeColumnsToContents()

    def slider(self, minimum: float, maximum: float, step: float, decimals: int = 1, suffix: str = "") -> NumericSlider:
        return NumericSlider(minimum, maximum, step, decimals, suffix)

    def apply_config_to_ui(self) -> None:
        config = self.config_data
        self.output_dir_edit.setText(str(config.get("output_dir", DEFAULT_CONFIG["output_dir"])))
        self.cache_dir_edit.setText(str(config.get("cache_dir", DEFAULT_CONFIG["cache_dir"])))
        self.compute_combo.setCurrentText(str(config.get("compute_type", "int8_float16")))
        language_code = str(config.get("language_mode", config.get("language", "zh")))
        self.language_combo.setCurrentText(LANGUAGE_LABELS.get(language_code, "简体中文"))
        self.simplify_check.setChecked(bool(config.get("simplify_chinese", True)))
        self.audio_route_preset_combo.setCurrentText(
            AUDIO_ROUTE_PRESET_LABELS.get(str(config.get("audio_route_preset", "solo_voice_input")), AUDIO_ROUTE_PRESET_LABELS["solo_voice_input"])
        )
        self.mixed_input_enabled.setChecked(bool(config.get("mixed_input_enabled", DEFAULT_CONFIG["mixed_input_enabled"])))
        self.set_combo_data(self.system_audio_combo, config.get("system_audio_device", DEFAULT_CONFIG["system_audio_device"]))
        self.system_audio_gain.setValue(float(config.get("system_audio_gain", DEFAULT_CONFIG["system_audio_gain"])))
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
        self.save_audio_segments_check.setChecked(bool(config.get("save_audio_segments", DEFAULT_CONFIG["save_audio_segments"])))
        self.audio_retention.setValue(float(config.get("audio_retention_minutes", DEFAULT_CONFIG["audio_retention_minutes"])))
        self.bubble_enabled.setChecked(bool(config.get("reply_bubble_enabled", DEFAULT_CONFIG["reply_bubble_enabled"])))
        self.bubble_port.setValue(int(config.get("reply_bubble_port", DEFAULT_CONFIG["reply_bubble_port"])))
        self.bubble_seconds.setValue(float(config.get("reply_bubble_seconds", DEFAULT_CONFIG["reply_bubble_seconds"])))
        self.bubble_token.setText(str(config.get("reply_bubble_token", "")))
        self.tts_guard_enabled.setChecked(bool(config.get("tts_guard_enabled", DEFAULT_CONFIG["tts_guard_enabled"])))
        self.tts_guard_file.setText(str(config.get("tts_guard_file", DEFAULT_CONFIG["tts_guard_file"])))
        oumuq_url = str(config.get("oumuq_url", DEFAULT_CONFIG["oumuq_url"]))
        if oumuq_url.strip().rstrip("/") == LEGACY_OUMUQ_URL:
            oumuq_url = DEFAULT_CONFIG["oumuq_url"]
        self.oumuq_url.setText(oumuq_url)
        self.oumuq_registry_path.setText(str(config.get("oumuq_registry_path", DEFAULT_CONFIG["oumuq_registry_path"])))
        self.oumuq_character_id.setCurrentText(str(config.get("oumuq_character_id", DEFAULT_CONFIG["oumuq_character_id"])))
        self.set_combo_data(self.oumuq_language, str(config.get("oumuq_language", DEFAULT_CONFIG["oumuq_language"])))
        self.oumuq_play_check.setChecked(bool(config.get("oumuq_play", DEFAULT_CONFIG["oumuq_play"])))
        self.oumuq_guard_seconds.setValue(float(config.get("oumuq_guard_seconds", DEFAULT_CONFIG["oumuq_guard_seconds"])))
        self.speaker_registry_file.setText(str(config.get("speaker_registry_file", DEFAULT_CONFIG["speaker_registry_file"])))
        self.speaker_recognition_enabled.setChecked(bool(config.get("speaker_recognition_enabled", DEFAULT_CONFIG["speaker_recognition_enabled"])))
        self.speaker_auto_enroll_enabled.setChecked(bool(config.get("speaker_auto_enroll_enabled", DEFAULT_CONFIG["speaker_auto_enroll_enabled"])))
        self.speaker_subtitle_enabled.setChecked(bool(config.get("speaker_subtitle_enabled", DEFAULT_CONFIG["speaker_subtitle_enabled"])))
        self.speaker_match_threshold.setValue(float(config.get("speaker_match_threshold", DEFAULT_CONFIG["speaker_match_threshold"])))
        self.route_enabled.setChecked(bool(config.get("rabiroute_enabled", False)))
        self.route_url.setText(str(config.get("rabiroute_url", DEFAULT_CONFIG["rabiroute_url"])))
        self.route_source.setText(str(config.get("rabiroute_source", DEFAULT_CONFIG["rabiroute_source"])))
        self.route_token.setText(str(config.get("rabiroute_token", "")))
        self.refresh_api_models()
        self.refresh_input_api_models()
        self.set_selected_api_model(str(config.get("api_model", current_api_model_id(self.api_model_combo))))
        self.refresh_local_models()
        self.refresh_source_visibility()
        self.refresh_mixed_input_visibility()
        self.update_trigger_summary()
        self.update_provider_status()
        self.refresh_path_labels()
        self.refresh_oumuq_roles()

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

    def list_system_audio_input_devices(self) -> list[tuple[int, str]]:
        keywords = ("立体声混音", "stereo mix", "what u hear", "loopback", "monitor", "output", "speaker", "speakers", "virtual", "cable", "input", "wave")
        preferred: list[tuple[int, str]] = []
        fallback: list[tuple[int, str]] = []
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            host_api = sd.query_hostapis(device["hostapi"])["name"]
            label = f"{index}: {device['name']} ({host_api})"
            name = str(device.get("name", "")).lower()
            item = (index, label)
            if any(keyword in name for keyword in keywords):
                preferred.append(item)
            else:
                fallback.append(item)
        return preferred + fallback

    def selected_device_index(self) -> int | None:
        return self.mic_combo.currentData()

    def refresh_mixed_input_visibility(self) -> None:
        self.system_audio_combo.setEnabled(True)
        self.system_audio_gain.setEnabled(True)

    def collect_config(self) -> dict:
        config = DEFAULT_CONFIG.copy()
        config.update(load_config(CONFIG_PATH))
        model_name = self.local_model_combo.currentData() or config.get("model", DEFAULT_CONFIG["model"])
        record = self.record_threshold.value()
        config.update(
            {
                "model": model_name,
                "output_dir": self.output_dir_edit.text().strip() or DEFAULT_CONFIG["output_dir"],
                "cache_dir": self.cache_dir_edit.text().strip() or DEFAULT_CONFIG["cache_dir"],
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
                "save_audio_segments": self.save_audio_segments_check.isChecked(),
                "audio_retention_minutes": round(max(1.0, self.audio_retention.value()), 1),
                "mic_device": self.selected_device_index(),
                "audio_route_preset": AUDIO_ROUTE_PRESET_CODES.get(self.audio_route_preset_combo.currentText(), "solo_voice_input"),
                "mixed_input_enabled": self.mixed_input_enabled.isChecked(),
                "system_audio_device": self.system_audio_combo.currentData(),
                "system_audio_gain": round(float(self.system_audio_gain.value()), 1),
                "reply_bubble_enabled": self.bubble_enabled.isChecked(),
                "reply_bubble_port": int(self.bubble_port.value()),
                "reply_bubble_seconds": round(self.bubble_seconds.value(), 1),
                "reply_bubble_token": self.bubble_token.text().strip(),
                "tts_guard_enabled": self.tts_guard_enabled.isChecked(),
                "tts_guard_file": self.tts_guard_file.text().strip() or DEFAULT_CONFIG["tts_guard_file"],
                "oumuq_url": self.oumuq_url.text().strip() or DEFAULT_CONFIG["oumuq_url"],
                "oumuq_registry_path": self.oumuq_registry_path.text().strip() or DEFAULT_CONFIG["oumuq_registry_path"],
                "oumuq_character_id": self.selected_oumuq_character_id(),
                "oumuq_language": self.oumuq_language.currentData() or DEFAULT_CONFIG["oumuq_language"],
                "oumuq_play": self.oumuq_play_check.isChecked(),
                "oumuq_guard_seconds": round(max(0.0, self.oumuq_guard_seconds.value()), 1),
                "playback_api_enabled": bool(self.config_data.get("playback_api_enabled", DEFAULT_CONFIG["playback_api_enabled"])),
                "playback_api_port": int(self.config_data.get("playback_api_port", DEFAULT_CONFIG["playback_api_port"])),
                "playback_api_token": str(self.config_data.get("playback_api_token", DEFAULT_CONFIG["playback_api_token"])),
                "speaker_registry_file": self.speaker_registry_file.text().strip() or DEFAULT_CONFIG["speaker_registry_file"],
                "speaker_recognition_enabled": self.speaker_recognition_enabled.isChecked(),
                "speaker_auto_enroll_enabled": self.speaker_auto_enroll_enabled.isChecked(),
                "speaker_subtitle_enabled": self.speaker_subtitle_enabled.isChecked(),
                "speaker_match_threshold": round(float(self.speaker_match_threshold.value()), 2),
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
        self.restart_playback_api_server()
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

    def write_tts_guard_for_playback(self, text: str, character_id: str, guard_seconds: float) -> Path:
        path = self.tts_guard_path_from_ui()
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        recent_texts: list[dict] = []
        try:
            existing = json.loads(path.read_text(encoding="utf-8-sig"))
            existing_recent = existing.get("recent_texts", []) if isinstance(existing, dict) else []
            if isinstance(existing_recent, list):
                recent_texts = [item for item in existing_recent if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            recent_texts = []
        speaker_id = f"tts:{character_id}" if character_id else "tts:oumuq"
        current = {
            "time": now,
            "text": text,
            "speaker_id": speaker_id,
            "source": "fennenote_oumuq",
        }
        guard_state = {
            "active": True,
            "ignore_until": now + max(0.0, guard_seconds),
            "text": text,
            "tts_text": text,
            "recent_texts": [current, *recent_texts[:19]],
            "source": "fennenote_oumuq",
            "character_id": character_id,
            "speaker_id": speaker_id,
        }
        path.write_text(json.dumps(guard_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def restart_playback_api_server(self) -> None:
        self.stop_playback_api_server()
        self.start_playback_api_server()

    def dispatch_playback_request(self, payload: dict, status_prefix: str = "播放请求") -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            self.playback_finished.emit(False, "播放请求缺少 text", {})
            return
        url = self.oumuq_url.text().strip() or DEFAULT_CONFIG["oumuq_url"]
        outgoing = {key: value for key, value in payload.items() if not str(key).startswith("_")}
        character_id = str(outgoing.get("character_id") or self.selected_oumuq_character_id()).strip()
        guard_seconds = float(payload.get("_guard_seconds", self.oumuq_guard_seconds.value()) or 0.0)
        self.prepare_playback_route(outgoing, status_prefix)
        try:
            guard_path = self.write_tts_guard_for_playback(text, character_id, max(0.0, guard_seconds))
        except OSError as exc:
            self.playback_finished.emit(False, f"TTS guard 写入失败：{exc}", {})
            return
        self.append_log(f"{status_prefix}：已写入 TTS guard：{guard_path}")
        if not self.tts_guard_enabled.isChecked():
            self.append_log("提醒：应用页尚未启用 TTS guard，转写进程不会读取 guard 文件。")

        def worker() -> None:
            data = json.dumps(outgoing, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "FenneNote",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    try:
                        response_data = json.loads(body) if body.strip() else {}
                    except json.JSONDecodeError:
                        if body.lstrip().lower().startswith("<!doctype") or "<html" in body[:200].lower():
                            self.playback_finished.emit(False, f"OumuQ URL 返回了网页，不是播放接口 JSON。请检查端口/路径：{url}", {})
                            return
                        self.playback_finished.emit(False, f"OumuQ 返回内容不是 JSON：{body[:300]}", {})
                        return
                    if not isinstance(response_data, dict):
                        self.playback_finished.emit(False, f"OumuQ 返回 JSON 格式不符合预期：{body[:300]}", {})
                        return
                    message = f"OumuQ 已接收播放请求（HTTP {response.status}）：{body[:300]}"
                    self.playback_finished.emit(True, message, response_data)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self.playback_finished.emit(False, f"OumuQ 播放失败：HTTP {exc.code} {body[:300]}", {})
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.playback_finished.emit(False, f"OumuQ 播放失败：{exc}", {})

        threading.Thread(target=worker, daemon=True).start()

    def playback_route_key(self, payload: dict) -> tuple[str, str, str, str, str]:
        return (
            str(payload.get("worker_url", "")).strip(),
            str(payload.get("model") or payload.get("tts_model") or payload.get("api_model") or "").strip(),
            str(payload.get("character_id", "")).strip(),
            str(payload.get("language", "")).strip(),
            str(payload.get("tts_engine") or payload.get("engine") or "").strip(),
        )

    def prepare_playback_route(self, payload: dict, status_prefix: str) -> None:
        route_key = self.playback_route_key(payload)
        if route_key == self.active_playback_route:
            return
        self.active_playback_route = route_key
        worker_url, model, character_id, language, engine = route_key
        parts = [
            f"worker={worker_url or 'OumuQ 默认'}",
            f"model={model or '默认'}",
            f"character={character_id or '默认'}",
            f"language={language or 'auto'}",
            f"engine={engine or '默认'}",
        ]
        self.append_log(f"{status_prefix}：切换播放目标：" + " / ".join(parts))
        if not worker_url:
            return

        def worker_status_probe() -> None:
            try:
                status_url = worker_url.rstrip("/") + "/status"
                request = urllib.request.Request(status_url, headers={"User-Agent": "FenneNote"})
                with urllib.request.urlopen(request, timeout=3) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self.playback_finished.emit(True, f"播放目标已就绪：{body[:300]}", {})
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.playback_finished.emit(False, f"播放目标状态探测失败：{exc}", {})

        threading.Thread(target=worker_status_probe, daemon=True).start()

    def handle_playback_api_request(self, payload: dict) -> None:
        request_id = str(payload.get("_request_id", "")).strip()
        self.oumuq_status.setText(f"收到 RabiRoute 播放请求：{request_id or 'local'}")
        self.append_log(f"收到 RabiRoute 播放请求：{payload.get('text', '')}")
        self.dispatch_playback_request(payload, "RabiRoute 播放请求")

    def handle_reply_api_request(self, payload: dict) -> None:
        title = str(payload.get("title") or payload.get("sender") or "RabiRoute").strip()
        text = str(payload.get("text") or payload.get("message") or payload.get("content") or "").strip()
        if not text:
            return
        message = f"{title}：{text}" if title else text
        self.status_label.setText(message[:120])
        self.append_log(f"收到 RabiRoute 文字反写：{message}")
        if hasattr(self, "bubble_enabled") and self.bubble_enabled.isChecked():
            timeout_ms = int(max(1.0, self.bubble_seconds.value()) * 1000) if hasattr(self, "bubble_seconds") else 3000
            point = self.mapToGlobal(QPoint(24, max(24, self.height() - 96)))
            QToolTip.showText(point, message, self, self.rect(), timeout_ms)

    def submit_oumuq_playback(self) -> None:
        text = self.oumuq_test_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "OumuQ", "请先输入测试文本。")
            return
        payload = {
            "text": text,
            "play": self.oumuq_play_check.isChecked(),
        }
        language = self.oumuq_language.currentData()
        if language and language != "auto":
            payload["language"] = language
        character_id = self.selected_oumuq_character_id()
        if character_id:
            payload["character_id"] = character_id
        self.oumuq_play_button.setEnabled(False)
        self.oumuq_status.setText("正在提交播放请求...")
        self.dispatch_playback_request(payload, "手动播放请求")

    def find_audio_outputs(self, value) -> list[str]:
        outputs: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"output", "audio", "audio_file", "file", "path"} and isinstance(item, str):
                    suffix = Path(item).suffix.lower()
                    if suffix in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                        outputs.append(item)
                outputs.extend(self.find_audio_outputs(item))
        elif isinstance(value, list):
            for item in value:
                outputs.extend(self.find_audio_outputs(item))
        return list(dict.fromkeys(outputs))

    def register_oumuq_output_samples(self, response_data: dict) -> None:
        character_id = self.selected_oumuq_character_id()
        if not character_id:
            return
        outputs = self.find_audio_outputs(response_data)
        if not outputs:
            return
        speaker_id = f"tts_{character_id}"
        display_name = self.oumuq_character_name.text().split("/", 1)[0].strip() or character_id
        self.upsert_speaker_profile({
            "id": speaker_id,
            "display_name": f"{display_name} TTS",
            "kind": "tts",
            "character_id": character_id,
            "samples": outputs,
            "sample_metadata": [
                {
                    "path": output,
                    "source": "oumuq_playback",
                    "read_aloud_text": self.oumuq_text.toPlainText().strip() if hasattr(self, "oumuq_text") else "",
                    "recorded_at": datetime.now().isoformat(),
                }
                for output in outputs
            ],
            "status": "sampled_unmodeled",
            "updated_at": datetime.now().isoformat(),
            "created_by": "oumuq_playback",
        })
        self.refresh_speaker_table()
        self.append_log(f"已登记 OumuQ 输出样本到声纹档案：{speaker_id}（{len(outputs)} 个）")

    def on_playback_finished(self, ok: bool, message: str, response_data: dict) -> None:
        self.oumuq_play_button.setEnabled(True)
        self.oumuq_status.setText(message)
        self.status_label.setText("播放请求已提交" if ok else "播放请求失败")
        self.append_log(message)
        if ok:
            self.register_oumuq_output_samples(response_data)
        if not ok:
            QMessageBox.warning(self, "OumuQ", message)

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
        self.level_text.setText(f"麦克风电平：原始 {self.display_raw_level:.3f} / 增益后 {self.display_level:.3f} / 录音阈值 {threshold:.3f} / 转写阈值 {transcribe:.3f} / 峰值 {peak:.3f}")
        is_recording = self.display_level >= threshold or level >= threshold
        if self.transcriber_activity_state == "transcribing":
            self.trigger_state_label.setText("状态：正在转写，芬妮在记录")
        elif self.transcriber_activity_state == "recording":
            self.trigger_state_label.setText(
                f"状态：录音中，芬妮正在听；安静 {self.pause_seconds.value():.1f}s 后切句"
            )
        elif self.transcriber_activity_state == "queued":
            self.trigger_state_label.setText("状态：已切句，等待转写")
        elif self.transcriber_activity_state == "written":
            self.trigger_state_label.setText("状态：转写完成，继续监听")
        elif not self.transcriber_running() and is_recording:
            self.set_buddy_state("listening")
            self.trigger_state_label.setText(
                f"状态：音量预览中；启动转写后按安静 {self.pause_seconds.value():.1f}s 切句"
            )
        elif not self.transcriber_running():
            self.set_buddy_state("idle")
            self.trigger_state_label.setText("状态：监听中，芬妮在待命")
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
        self.transcriber_activity_state = "listening"
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
        if line.startswith("FN_TRANSCRIPT|"):
            try:
                payload = json.loads(line.split("|", 1)[1])
            except (IndexError, json.JSONDecodeError) as exc:
                self.append_log(f"转写预览解析失败：{exc}")
                return
            self.append_transcript_preview(payload)
            return
        if line.startswith("FN_STATUS|"):
            parts = line.split("|", 2)
            if len(parts) == 3:
                code = parts[1]
                self.status_label.setText(parts[2])
                self.append_log(parts[2])
                if code in {"recording", "queued", "listening"}:
                    self.transcriber_activity_state = code
                    self.set_buddy_state("listening")
                    if code == "recording":
                        self.trigger_state_label.setText(parts[2])
                    elif code == "queued":
                        self.trigger_state_label.setText("状态：已切句，等待转写")
                    else:
                        self.trigger_state_label.setText("状态：监听中，芬妮在待命")
                elif code == "transcribing":
                    self.transcriber_activity_state = code
                    self.set_buddy_state("writing")
                    self.trigger_state_label.setText("状态：正在转写，芬妮在记录")
                elif code == "written":
                    self.transcriber_activity_state = code
                    self.set_buddy_state("listening")
                    self.trigger_state_label.setText("状态：转写完成，继续监听")
                elif code in {"discarded", "tts_guard_skip", "tts_guard_echo", "api_error"}:
                    self.transcriber_activity_state = "listening"
                    self.set_buddy_state("listening")
                    self.trigger_state_label.setText("状态：监听中，芬妮在待命")
                elif code == "paused":
                    self.transcriber_activity_state = code
                    self.set_buddy_state("idle")
                    self.trigger_state_label.setText("状态：已暂停监听")
            return
        if line.startswith("["):
            self.transcript.appendPlainText(self.format_transcript_preview_line(line))
        elif line:
            self.status_label.setText(line[:100])
            self.append_log(line)

    def format_transcript_preview_line(self, line: str) -> str:
        if len(line) < 10 or not line.startswith("["):
            return line
        time_end = line.find("]")
        if time_end <= 0:
            return line
        timestamp = line[:time_end + 1]
        rest = line[time_end + 1:].strip()
        speaker_text = "未识别"
        confidence_text = ""
        if rest.startswith("["):
            speaker_end = rest.find("]")
            if speaker_end > 0:
                speaker_chunk = rest[1:speaker_end].strip()
                rest = rest[speaker_end + 1:].strip()
                if speaker_chunk:
                    parts = speaker_chunk.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            confidence = float(parts[1])
                            speaker_text = parts[0].strip() or "未识别"
                            confidence_text = f"（{confidence:.2f}）"
                        except ValueError:
                            speaker_text = speaker_chunk
                    else:
                        speaker_text = speaker_chunk
        return f"{timestamp} 发言人是：{speaker_text}{confidence_text}  {rest}".rstrip()

    def speaker_preview_label(self, speaker: dict | None) -> str:
        if not isinstance(speaker, dict):
            return "未识别"
        name = str(speaker.get("speaker_name") or speaker.get("speaker_id") or "未识别").strip() or "未识别"
        confidence = speaker.get("speaker_confidence")
        try:
            return f"{name}（{float(confidence):.2f}）"
        except (TypeError, ValueError):
            return name

    def append_transcript_preview(self, payload: dict) -> None:
        timestamp = str(payload.get("time") or "")
        if not timestamp:
            started_at = str(payload.get("started_at") or "")
            timestamp = started_at[11:19] if len(started_at) >= 19 else datetime.now().strftime("%H:%M:%S")
        turns = payload.get("speaker_turns")
        if isinstance(turns, list) and len(turns) > 1:
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = self.speaker_preview_label(turn)
                text = str(turn.get("text") or "").strip()
                start = float(turn.get("start", 0.0) or 0.0)
                end = float(turn.get("end", 0.0) or 0.0)
                time_range = f"+{start:.1f}-{end:.1f}s"
                self.transcript.appendPlainText(f"[{timestamp} {time_range}] 发言人是：{speaker}  {text}".rstrip())
            return
        speaker = self.speaker_preview_label(payload.get("speaker"))
        text = str(payload.get("text") or "").strip()
        self.transcript.appendPlainText(f"[{timestamp}] 发言人是：{speaker}  {text}".rstrip())

    def on_process_exited(self, _code: int) -> None:
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止")
        self.append_log("转写进程已停止")
        self.running_config = None
        self.transcriber_activity_state = "idle"
        self.set_buddy_state("idle")
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait(1000)
        self.process_thread = None
        self.process_worker = None
        self.update_start_button_state()

    def open_output_folder(self) -> None:
        output_dir = self.output_dir_from_ui()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.open_local_path(output_dir)

    def open_cache_folder(self) -> None:
        cache_dir = self.cache_dir_from_ui()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.open_local_path(cache_dir)

    def open_audio_folder(self) -> None:
        audio_dir = self.audio_dir_from_ui()
        audio_dir.mkdir(parents=True, exist_ok=True)
        self.open_local_path(audio_dir)

    def open_config_file(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.open_local_path(CONFIG_PATH)

    def closeEvent(self, event) -> None:
        if self.tray_icon and self.tray_icon.isVisible() and not self.force_quit:
            event.ignore()
            self.hide()
            if not self.tray_hide_notice_shown:
                self.tray_icon.showMessage(
                    "FenneNote 仍在运行",
                    "窗口已隐藏到系统托盘。双击托盘图标可以重新打开。",
                    QSystemTrayIcon.Information,
                    3000,
                )
                self.tray_hide_notice_shown = True
            return
        self.stop_transcriber()
        self.stop_level_worker()
        self.stop_playback_api_server()
        if self.tray_icon:
            self.tray_icon.hide()
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
    app.setQuitOnLastWindowClosed(False)
    window = FenneNoteQt()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
