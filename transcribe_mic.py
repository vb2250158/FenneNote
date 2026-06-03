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

CONFIG_VERSION = 4

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
    "vad_filter": True,
    "beam_size": 1,
    "condition_on_previous_text": False,
    "mic_device": None,
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
                        phrases.put(Phrase(audio=audio, started_at=started_at, ended_at=datetime.now()))
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

    print("Loading Whisper model...")
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
    print(f"Writing transcript to: {output_dir}")
    print(f"Using local cache: {storage_dirs['cache']}")

    while not stop_event.is_set():
        try:
            phrase = phrases.get(timeout=0.2)
        except queue.Empty:
            continue

        initial_prompt = config.get("initial_prompt") or None
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
            continue
        append_line(output_dir, phrase.started_at, text)


def list_devices() -> None:
    print(sd.query_devices())


def main() -> int:
    ensure_cuda_dll_path()
    parser = argparse.ArgumentParser(description="Continuously transcribe microphone audio into daily text files.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    transcribe_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
