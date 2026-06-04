from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
from opencc import OpenCC


CUDA_DLL_DIRS = [
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"),
    Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA Canvas"),
]

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

CONFIG_VERSION = 5

DEFAULT_CONFIG = {
    "config_version": CONFIG_VERSION,
    "auto_start": False,
    "model": "small",
    "device": "cuda",
    "compute_type": "int8_float16",
    "language": "zh",
    "language_mode": "zh",
    "simplify_chinese": True,
    "drop_non_chinese_in_zh": True,
    "initial_prompt": "以下是简体中文普通话办公场景转写，可能包含 Unity、Editor、GPU、CPU、AI、Bug、微信、项目、功能、消息、发送、测试等术语。请保持中文为简体，英文术语保留英文。",
    "output_dir": "transcripts",
    "cache_dir": "cache",
    "cache_retention_minutes": 0.0,
    "sample_rate": 16000,
    "chunk_seconds": 0.5,
    "pre_roll_seconds": 1.5,
    "min_phrase_seconds": 0.6,
    "max_phrase_seconds": 60.0,
    "transcribe_pause_seconds": 0.5,
    "silence_seconds": 5.0,
    "input_gain": 1.0,
    "record_threshold": 0.01,
    "transcribe_threshold": 0.015,
    "rms_threshold": 0.01,
    "adaptive_threshold": True,
    "adaptive_threshold_multiplier": 2.5,
    "adaptive_threshold_margin": 0.004,
    "vad_filter": False,
    "beam_size": 1,
    "condition_on_previous_text": False,
    "mic_device": None,
    "reply_bubble_enabled": True,
    "reply_bubble_port": 8792,
    "reply_bubble_seconds": 3.0,
    "reply_bubble_token": "",
    "rabiroute_enabled": False,
    "rabiroute_url": "http://127.0.0.1:8791/webhook",
    "rabiroute_token": "",
    "rabiroute_source": "fennenote",
}

MODEL_REPOSITORIES = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}

DOWNLOADABLE_MODELS = tuple(MODEL_REPOSITORIES.keys())

FASTER_WHISPER_HOMEPAGE = "https://github.com/SYSTRAN/faster-whisper"
OPENAI_WHISPER_HOMEPAGE = "https://github.com/openai/whisper"

MODEL_PERFORMANCE_ROWS = (
    {
        "family": "tiny",
        "parameters": "39M",
        "required_vram": "约 1 GB",
        "relative_speed": "约 10x",
        "fennenote_advice": "极轻量测试，中文质量较弱",
    },
    {
        "family": "base",
        "parameters": "74M",
        "required_vram": "约 1 GB",
        "relative_speed": "约 7x",
        "fennenote_advice": "快速草稿，短句可用",
    },
    {
        "family": "small",
        "parameters": "244M",
        "required_vram": "约 2 GB",
        "relative_speed": "约 4x",
        "fennenote_advice": "默认推荐，适合 8GB 显存常驻",
    },
    {
        "family": "medium",
        "parameters": "769M",
        "required_vram": "约 5 GB",
        "relative_speed": "约 2x",
        "fennenote_advice": "质量更好，占用和延迟更高",
    },
    {
        "family": "large",
        "parameters": "1550M",
        "required_vram": "约 10 GB",
        "relative_speed": "1x",
        "fennenote_advice": "准确率优先，8GB+Unity 不建议常驻",
    },
)

MODEL_FAMILY_SPECS = {row["family"]: row for row in MODEL_PERFORMANCE_ROWS}

MODEL_VARIANT_DATA = {
    "tiny.en": ("tiny", "仅英文", "英文实时草稿和功能测试，不适合中文。"),
    "tiny": ("tiny", "多语言，含中文", "最快的中文可用模型，适合确认流程，不适合正式记录。"),
    "base.en": ("base", "仅英文", "英文短句更稳，中文场景不要选这个。"),
    "base": ("base", "多语言，含中文", "低占用中文草稿，质量明显弱于 small。"),
    "small.en": ("small", "仅英文", "英文场景的轻量推荐，中文场景不要选这个。"),
    "small": ("small", "多语言，含中文", "FenneNote 默认推荐，速度、质量和 8GB 显存占用比较均衡。"),
    "medium.en": ("medium", "仅英文", "英文质量优先，显存和延迟高于 small。"),
    "medium": ("medium", "多语言，含中文", "中文质量优先时可选，但和 Unity 同开时要观察显存。"),
    "large-v1": ("large", "多语言，含中文", "第一版 large，历史兼容用途为主。"),
    "large-v2": ("large", "多语言，含中文", "第二版 large，准确率优先但不适合 8GB 常驻。"),
    "large-v3": ("large", "多语言，含中文", "当前 large-v3，质量优先；8GB 显存同时开 Unity 时慎用。"),
    "large": ("large", "多语言，含中文", "large-v3 的别名，效果和 large-v3 相同。"),
}

UPSTREAM_MODEL_REPOSITORIES = {
    "tiny.en": "openai/whisper-tiny.en",
    "tiny": "openai/whisper-tiny",
    "base.en": "openai/whisper-base.en",
    "base": "openai/whisper-base",
    "small.en": "openai/whisper-small.en",
    "small": "openai/whisper-small",
    "medium.en": "openai/whisper-medium.en",
    "medium": "openai/whisper-medium",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large": "openai/whisper-large-v3",
}

MODEL_PROFILES = {
    model_name: {
        "name": model_name,
        "publisher": "SYSTRAN",
        "repository": repository,
        "download_url": f"https://huggingface.co/{repository}",
        "homepage_url": FASTER_WHISPER_HOMEPAGE,
        "upstream_repository": UPSTREAM_MODEL_REPOSITORIES[model_name],
        "upstream_url": f"https://huggingface.co/{UPSTREAM_MODEL_REPOSITORIES[model_name]}",
        "family": family,
        "language_scope": language_scope,
        "parameters": MODEL_FAMILY_SPECS[family]["parameters"],
        "required_vram": MODEL_FAMILY_SPECS[family]["required_vram"],
        "relative_speed": MODEL_FAMILY_SPECS[family]["relative_speed"],
        "family_advice": MODEL_FAMILY_SPECS[family]["fennenote_advice"],
        "description": description,
    }
    for model_name, repository in MODEL_REPOSITORIES.items()
    for family, language_scope, description in (MODEL_VARIANT_DATA[model_name],)
}

PERSIST_ACROSS_CONFIG_VERSION_KEYS = {
    "auto_start",
    "mic_device",
}


def app_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return APP_DIR / path


def cleanup_old_files(directory: Path, max_age_seconds: float) -> None:
    if max_age_seconds < 0 or not directory.exists():
        return
    cutoff = time.time() - max_age_seconds
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink()
        except OSError:
            pass
    for path in sorted((item for item in directory.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def configure_local_storage(config: dict | None = None) -> dict[str, Path]:
    config = DEFAULT_CONFIG if config is None else config
    cache_root = app_path(str(config.get("cache_dir", DEFAULT_CONFIG["cache_dir"]))).resolve()
    dirs = {
        "cache": cache_root,
        "hf_home": cache_root / "huggingface",
        "hf_hub": cache_root / "huggingface" / "hub",
        "models": cache_root / "models",
        "temp": cache_root / "temp",
        "audio": cache_root / "audio",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(dirs["hf_home"])
    os.environ["HF_HUB_CACHE"] = str(dirs["hf_hub"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_hub"])
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_CACHE"] = str(dirs["hf_home"] / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(dirs["cache"])
    os.environ["TMP"] = str(dirs["temp"])
    os.environ["TEMP"] = str(dirs["temp"])
    tempfile.tempdir = str(dirs["temp"])

    retention_minutes = max(0.0, min(60.0, float(config.get("cache_retention_minutes", DEFAULT_CONFIG["cache_retention_minutes"]))))
    cleanup_old_files(dirs["audio"], retention_minutes * 60.0)
    cleanup_old_files(dirs["temp"], retention_minutes * 60.0)
    return dirs


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


@dataclass
class Phrase:
    audio: np.ndarray
    started_at: datetime
    ended_at: datetime
    peak: float


def emit_status(code: str, message: str) -> None:
    print(f"FN_STATUS|{code}|{message}", flush=True)


def load_config(path: Path) -> dict:
    if not path.exists():
        shutil.copyfile(path.with_name("config.example.json"), path)
        print(f"Created default config: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        user_config = json.load(handle)
    config = migrate_config(user_config)
    if config != user_config:
        save_config(path, config)
    return config


def migrate_config(user_config: dict) -> dict:
    user_version = int(user_config.get("config_version", 0) or 0)
    config = DEFAULT_CONFIG.copy()
    if user_version == CONFIG_VERSION:
        config.update(user_config)
    else:
        config.update({key: value for key, value in user_config.items() if key in DEFAULT_CONFIG})
        if user_version < 4:
            if float(user_config.get("input_gain", 8.0)) == 8.0:
                config["input_gain"] = DEFAULT_CONFIG["input_gain"]
            if float(user_config.get("cache_retention_minutes", 10.0)) == 10.0:
                config["cache_retention_minutes"] = DEFAULT_CONFIG["cache_retention_minutes"]
        if user_version < 5:
            config["vad_filter"] = DEFAULT_CONFIG["vad_filter"]
        for key in PERSIST_ACROSS_CONFIG_VERSION_KEYS:
            if key in user_config:
                config[key] = user_config[key]
        config["config_version"] = CONFIG_VERSION
    config["device"] = "cuda"
    if "record_threshold" not in config and "rms_threshold" in config:
        config["record_threshold"] = config["rms_threshold"]
    config["rms_threshold"] = config.get("record_threshold", DEFAULT_CONFIG["record_threshold"])
    config["transcribe_threshold"] = max(float(config["record_threshold"]), float(config["transcribe_threshold"]))
    config["cache_retention_minutes"] = max(0.0, min(60.0, float(config.get("cache_retention_minutes", DEFAULT_CONFIG["cache_retention_minutes"]))))
    return config


def save_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def model_repository_id(model_name: str) -> str:
    normalized = model_name.strip()
    if normalized in MODEL_REPOSITORIES:
        return MODEL_REPOSITORIES[normalized]
    if "/" in normalized:
        return normalized
    raise ValueError(f"当前版本不支持模型名称：{model_name}")


def model_cache_root(config: dict, model_name: str) -> Path:
    storage_dirs = configure_local_storage(config)
    repo_id = model_repository_id(model_name)
    return storage_dirs["models"] / f"models--{repo_id.replace('/', '--')}"


def model_snapshot_path(config: dict, model_name: str) -> Path | None:
    root = model_cache_root(config, model_name)
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in snapshots.iterdir():
        if snapshot.is_dir() and (snapshot / "model.bin").exists():
            return snapshot
    return None


def model_is_installed(config: dict, model_name: str) -> bool:
    return model_snapshot_path(config, model_name) is not None


def delete_model_cache(config: dict, model_name: str, status_callback=emit_status) -> None:
    storage_dirs = configure_local_storage(config)
    root = model_cache_root(config, model_name)
    lock_root = storage_dirs["models"] / ".locks" / root.name
    if not root.exists() and not lock_root.exists():
        status_callback("model_delete_ready", f"模型缓存不存在：{model_name}")
        return
    status_callback("model_delete_start", f"正在删除模型缓存：{model_name}")
    if root.exists():
        shutil.rmtree(root)
    if lock_root.exists():
        shutil.rmtree(lock_root)
    status_callback("model_delete_ready", f"模型缓存已删除：{model_name}")


def download_configured_model(config: dict, status_callback=emit_status) -> Path:
    storage_dirs = configure_local_storage(config)
    model_name = str(config.get("model", DEFAULT_CONFIG["model"]))
    model_repository_id(model_name)
    from faster_whisper.utils import download_model

    status_callback("model_download_start", f"正在下载或检查模型：{model_name}")
    model_path = Path(download_model(model_name, cache_dir=str(storage_dirs["models"]))).resolve()
    status_callback("model_download_ready", f"模型已安装：{model_name} -> {model_path}")
    return model_path


def today_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{datetime.now():%Y-%m-%d}.txt"


def append_line(output_dir: Path, started_at: datetime, text: str) -> None:
    text = " ".join(text.split())
    if not text:
        return
    line = f"[{started_at:%H:%M:%S}] {text}\n"
    with today_output_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    print(line, end="", flush=True)


def post_rabiroute_event(config: dict, phrase: Phrase, text: str) -> None:
    if not bool(config.get("rabiroute_enabled", False)):
        return
    url = str(config.get("rabiroute_url", "")).strip()
    if not url:
        return
    payload = {
        "type": "voice_transcript",
        "source": str(config.get("rabiroute_source", "fennenote") or "fennenote"),
        "text": text,
        "startedAt": phrase.started_at.isoformat(),
        "endedAt": phrase.ended_at.isoformat(),
        "durationSeconds": round((phrase.ended_at - phrase.started_at).total_seconds(), 2),
        "peak": round(float(phrase.peak), 4),
        "time": int(phrase.started_at.timestamp()),
        "messageId": f"fennenote-{int(phrase.started_at.timestamp() * 1000)}",
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
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status >= 400:
                emit_status("route_error", f"RabiRoute 推送失败：HTTP {response.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        emit_status("route_error", f"RabiRoute 推送失败：{exc}")


def simplify_text(text: str, converter: OpenCC | None) -> str:
    if converter is None:
        return text
    return converter.convert(text)


def contains_cjk(text: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", text) is not None


def should_keep_segment(segment) -> bool:
    no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
    avg_logprob = getattr(segment, "avg_logprob", 0.0)
    compression_ratio = getattr(segment, "compression_ratio", 0.0)
    if no_speech_prob > 0.75:
        return False
    if avg_logprob < -1.2 and no_speech_prob > 0.45:
        return False
    if compression_ratio > 2.8:
        return False
    return True


def should_keep_text(text: str, language: str | None, config: dict) -> bool:
    normalized = " ".join(text.lower().split())
    hallucinations = {
        "thank you",
        "thanks",
        "thanks for watching",
        "bye",
        "bye bye",
        "what's with this",
    }
    if normalized in hallucinations:
        return False
    if language == "zh" and bool(config.get("drop_non_chinese_in_zh", True)) and not contains_cjk(text):
        return False
    return True


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))


def keyboard_worker(commands: queue.Queue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        command = sys.stdin.readline()
        if not command:
            time.sleep(0.2)
            continue
        command = command.strip().lower()
        if command in {"p", "pause", "r", "resume", "q", "quit"}:
            commands.put(command)


def collect_phrases(config: dict, stop_event: threading.Event) -> queue.Queue[Phrase]:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    phrases: queue.Queue[Phrase] = queue.Queue()
    sample_rate = int(config["sample_rate"])
    chunk_frames = int(sample_rate * float(config["chunk_seconds"]))
    pre_roll_frames = int(sample_rate * float(config.get("pre_roll_seconds", 1.5)))
    min_frames = int(sample_rate * float(config["min_phrase_seconds"]))
    max_frames = int(sample_rate * float(config["max_phrase_seconds"]))
    transcribe_pause_frames_limit = int(sample_rate * float(config.get("transcribe_pause_seconds", 0.5)))
    silence_frames_limit = int(sample_rate * float(config["silence_seconds"]))
    record_threshold = float(config.get("record_threshold", config.get("rms_threshold", 0.01)))
    transcribe_threshold = float(config.get("transcribe_threshold", max(record_threshold, 0.015)))
    adaptive_threshold = bool(config.get("adaptive_threshold", True))
    adaptive_multiplier = float(config.get("adaptive_threshold_multiplier", 2.5))
    adaptive_margin = float(config.get("adaptive_threshold_margin", 0.004))
    input_gain = float(config.get("input_gain", 1.0))
    device = config.get("mic_device")

    def callback(indata, frames, callback_time, status):
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        if input_gain != 1.0:
            chunk = np.clip(chunk * input_gain, -1.0, 1.0)
        audio_queue.put(chunk)

    def worker() -> None:
        buffer: list[np.ndarray] = []
        pre_buffer: deque[np.ndarray] = deque()
        pre_buffer_frames = 0
        started_at: datetime | None = None
        silence_frames = 0
        transcribe_pause_frames = 0
        voiced_frames = 0
        noise_floor = max(0.0005, record_threshold / 3.0)
        phrase_peak = 0.0
        paused = False
        commands: queue.Queue[str] = queue.Queue()
        threading.Thread(target=keyboard_worker, args=(commands, stop_event), daemon=True).start()

        print("命令：p=暂停，r=继续，q=退出", flush=True)
        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_frames,
            dtype="float32",
            callback=callback,
        ):
            while not stop_event.is_set():
                while not commands.empty():
                    command = commands.get_nowait()
                    if command in {"q", "quit"}:
                        stop_event.set()
                    elif command in {"p", "pause"}:
                        paused = True
                        buffer.clear()
                        started_at = None
                        silence_frames = 0
                        transcribe_pause_frames = 0
                        voiced_frames = 0
                        print("Paused.")
                    elif command in {"r", "resume"}:
                        paused = False
                        print("Resumed.")

                try:
                    chunk = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if paused:
                    continue

                chunk_level = rms(chunk)
                current_chunk_buffered = False
                if adaptive_threshold and started_at is None:
                    dynamic_threshold = max(record_threshold, noise_floor * adaptive_multiplier + adaptive_margin)
                else:
                    dynamic_threshold = record_threshold

                chunk_is_voice = chunk_level >= dynamic_threshold
                if started_at is None:
                    pre_buffer.append(chunk)
                    pre_buffer_frames += len(chunk)
                    while pre_buffer_frames > pre_roll_frames and pre_buffer:
                        removed = pre_buffer.popleft()
                        pre_buffer_frames -= len(removed)

                    if adaptive_threshold and not chunk_is_voice:
                        noise_floor = (noise_floor * 0.95) + (chunk_level * 0.05)
                    if not chunk_is_voice:
                        continue

                    buffer = list(pre_buffer)
                    current_chunk_buffered = True
                    pre_seconds = pre_buffer_frames / sample_rate
                    started_at = datetime.now() - timedelta(seconds=pre_seconds)
                    pre_buffer.clear()
                    pre_buffer_frames = 0
                    phrase_peak = chunk_level

                if not current_chunk_buffered:
                    buffer.append(chunk)
                phrase_peak = max(phrase_peak, chunk_level)
                if chunk_is_voice:
                    voiced_frames += len(chunk)
                    silence_frames = 0
                else:
                    silence_frames += len(chunk)
                if chunk_level < transcribe_threshold:
                    transcribe_pause_frames += len(chunk)
                else:
                    transcribe_pause_frames = 0

                total_frames = sum(len(part) for part in buffer)
                phrase_ready_for_transcribe = phrase_peak >= transcribe_threshold
                phrase_done = phrase_ready_for_transcribe and transcribe_pause_frames >= transcribe_pause_frames_limit and voiced_frames >= min_frames
                phrase_stale_noise = silence_frames >= silence_frames_limit and voiced_frames >= min_frames
                phrase_too_long = total_frames >= max_frames

                if phrase_done or phrase_stale_noise or phrase_too_long:
                    audio = np.concatenate(buffer).astype(np.float32)
                    if phrase_peak >= transcribe_threshold:
                        ended_at = datetime.now()
                        phrases.put(Phrase(audio=audio, started_at=started_at, ended_at=ended_at, peak=phrase_peak))
                        emit_status("queued", f"已捕获待转写片段：{(ended_at - started_at).total_seconds():.1f} 秒，峰值 {phrase_peak:.3f}")
                    buffer.clear()
                    started_at = None
                    silence_frames = 0
                    transcribe_pause_frames = 0
                    voiced_frames = 0
                    phrase_peak = 0.0

    threading.Thread(target=worker, daemon=True).start()
    return phrases


def transcribe_loop(config: dict) -> None:
    storage_dirs = configure_local_storage(config)
    output_dir = app_path(str(config["output_dir"])).resolve()
    from faster_whisper import WhisperModel

    emit_status("model_loading", f"正在准备模型：{config['model']}（首次运行可能需要下载到本地 cache/models）")
    model = WhisperModel(
        config["model"],
        device=config["device"],
        compute_type=config["compute_type"],
        download_root=str(storage_dirs["models"]),
    )
    language = config.get("language_mode", config.get("language", "zh"))
    if language in {"auto", "", None}:
        language = None
    converter = OpenCC("t2s") if bool(config.get("simplify_chinese", True)) else None
    stop_event = threading.Event()
    phrases = collect_phrases(config, stop_event)
    emit_status("model_ready", f"模型已就绪：{config['model']} / {config['device']} / {config['compute_type']}")
    emit_status("output_ready", f"写入目录：{output_dir}")
    emit_status("cache_ready", f"本地缓存：{storage_dirs['cache']}")

    while not stop_event.is_set():
        try:
            phrase = phrases.get(timeout=0.2)
        except queue.Empty:
            continue

        initial_prompt = config.get("initial_prompt") or None
        emit_status("transcribing", f"正在转写：{(phrase.ended_at - phrase.started_at).total_seconds():.1f} 秒音频，峰值 {phrase.peak:.3f}")
        segments, _info = model.transcribe(
            phrase.audio,
            language=language,
            beam_size=int(config["beam_size"]),
            vad_filter=bool(config["vad_filter"]),
            condition_on_previous_text=bool(config["condition_on_previous_text"]),
            initial_prompt=initial_prompt,
        )
        text = "".join(segment.text for segment in segments if should_keep_segment(segment)).strip()
        text = simplify_text(text, converter)
        if not should_keep_text(text, language, config):
            emit_status("discarded", "本段已转写但没有保留有效文字，已丢弃")
            continue
        append_line(output_dir, phrase.started_at, text)
        post_rabiroute_event(config, phrase, text)
        emit_status("written", "转写完成，已写入今日文本")


def list_devices() -> None:
    print(sd.query_devices())


def main() -> int:
    ensure_cuda_dll_path()
    parser = argparse.ArgumentParser(description="Continuously transcribe microphone audio into daily text files.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit.")
    parser.add_argument("--download-model", action="store_true", help="Download/check the configured Whisper model and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.download_model:
        download_configured_model(config)
        return 0
    transcribe_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
