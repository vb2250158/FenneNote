from __future__ import annotations

import argparse
import base64
import contextlib
import io
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
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from difflib import SequenceMatcher
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
    "save_audio_segments": False,
    "audio_retention_minutes": 10.0,
    "sample_rate": 16000,
    "chunk_seconds": 0.5,
    "pre_roll_seconds": 1.5,
    "min_phrase_seconds": 0.6,
    "max_phrase_seconds": 60.0,
    "transcribe_pause_seconds": 0.5,
    "silence_seconds": 5.0,
    "input_gain": 1.0,
    "xiaoai_level_gain": 10.0,
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
    "audio_route_preset": "solo_voice_input",
    "mixed_input_enabled": False,
    "system_audio_device": None,
    "system_audio_gain": 1.0,
    "tts_guard_enabled": False,
    "tts_guard_file": "cache/tts_guard.json",
    "tts_guard_resume_margin_seconds": 0.8,
    "tts_guard_recent_text_window_seconds": 20.0,
    "tts_guard_similarity_threshold": 0.86,
    "oumuq_url": "http://127.0.0.1:8780/api/speak",
    "oumuq_registry_path": "../OumuQ/voice-references/reference-index.json",
    "oumuq_character_id": "",
    "oumuq_language": "auto",
    "oumuq_play": True,
    "oumuq_guard_seconds": 8.0,
    "playback_api_enabled": True,
    "playback_api_port": 8793,
    "playback_api_token": "",
    "speaker_registry_file": "cache/speakers/speaker_registry.json",
    "speaker_recognition_enabled": False,
    "speaker_subtitle_enabled": False,
    "speaker_auto_enroll_enabled": True,
    "speaker_unknown_prefix": "unknown",
    "speaker_match_threshold": 0.92,
    "speaker_diarization_upload_url": "http://127.0.0.1:8780/api/audio/upload-public",
    "speaker_diarization_model": "paraformer-v2",
    "speaker_diarization_speaker_count": 0,
    "model_source": "local",
    "api_provider": "dashscope",
    "api_provider_id": "dashscope",
    "api_provider_enabled": False,
    "api_model": "qwen3-asr-flash",
    "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "",
    "transcript_corrections_enabled": True,
    "transcript_corrections": {
        "森之灵": "森之宁",
        "森之名": "森之宁",
        "升之零": "森之宁",
        "生之零": "森之宁",
        "孙智宁": "森之宁",
        "孙之宁": "森之宁",
        "孙芝宁": "森之宁",
        "森之林": "森之宁",
        "森之岭": "森之宁",
    },
    "reply_bubble_enabled": True,
    "reply_bubble_port": 8792,
    "reply_bubble_seconds": 3.0,
    "reply_bubble_token": "",
    "rabiroute_enabled": False,
    "rabiroute_url": "http://127.0.0.1:8791/webhook",
    "rabiroute_token": "",
    "rabiroute_source": "fennenote",
    "rabiroute_manager_url": "http://127.0.0.1:8790",
    "rabiroute_route_id": "",
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
    "system_audio_device",
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
    audio_retention_minutes = max(1.0, float(config.get("audio_retention_minutes", DEFAULT_CONFIG["audio_retention_minutes"])))
    cleanup_old_files(dirs["audio"], audio_retention_minutes * 60.0)
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


def emit_transcript_preview(started_at: datetime, text: str, speaker_result: dict | None = None, speaker_turns: list[dict] | None = None) -> None:
    payload = {
        "started_at": started_at.isoformat(),
        "time": f"{started_at:%H:%M:%S}",
        "text": " ".join(text.split()),
        "speaker": speaker_result or {},
        "speaker_turns": speaker_turns or [],
    }
    print("FN_TRANSCRIPT|" + json.dumps(payload, ensure_ascii=False), flush=True)


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
    config["audio_retention_minutes"] = max(1.0, float(config.get("audio_retention_minutes", DEFAULT_CONFIG["audio_retention_minutes"])))
    config["save_audio_segments"] = bool(config.get("save_audio_segments", DEFAULT_CONFIG["save_audio_segments"]))
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


def append_line(output_dir: Path, started_at: datetime, text: str, speaker_result: dict | None = None) -> None:
    text = " ".join(text.split())
    if not text:
        return
    speaker_name = ""
    if speaker_result and speaker_result.get("speaker_name"):
        confidence = speaker_result.get("speaker_confidence")
        suffix = f" {float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""
        speaker_name = f"[{speaker_result.get('speaker_name')}{suffix}] "
    line = f"[{started_at:%H:%M:%S}] {speaker_name}{text}\n"
    with today_output_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def post_rabiroute_event(config: dict, phrase: Phrase, text: str, speaker_result: dict | None = None, speaker_turns: list[dict] | None = None) -> None:
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
    if speaker_result:
        payload.update({key: value for key, value in speaker_result.items() if value not in (None, "")})
    if speaker_turns:
        payload["speaker_turns"] = speaker_turns
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


def apply_transcript_corrections(text: str, config: dict) -> str:
    if not bool(config.get("transcript_corrections_enabled", True)):
        return text
    corrections = config.get("transcript_corrections", DEFAULT_CONFIG["transcript_corrections"])
    if not isinstance(corrections, dict):
        return text
    corrected = text
    for wrong, right in corrections.items():
        wrong_text = str(wrong).strip()
        right_text = str(right).strip()
        if wrong_text and right_text:
            corrected = corrected.replace(wrong_text, right_text)
    return corrected


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


def normalize_tts_guard_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def tts_guard_path(config: dict) -> Path:
    return app_path(str(config.get("tts_guard_file", DEFAULT_CONFIG["tts_guard_file"]))).resolve()


def load_tts_guard_state(config: dict) -> dict:
    if not bool(config.get("tts_guard_enabled", False)):
        return {}
    try:
        with tts_guard_path(config).open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def tts_guard_ignore_until(state: dict, config: dict) -> float:
    for key in ("ignore_until", "ignoreInputUntil", "ignore_until_unix"):
        try:
            return float(state.get(key))
        except (TypeError, ValueError):
            continue
    if bool(state.get("active", False)):
        margin = float(config.get("tts_guard_resume_margin_seconds", DEFAULT_CONFIG["tts_guard_resume_margin_seconds"]))
        return time.time() + max(0.0, margin)
    return 0.0


def tts_guard_should_skip_audio(config: dict) -> bool:
    state = load_tts_guard_state(config)
    return tts_guard_ignore_until(state, config) > time.time()


def tts_guard_recent_texts(config: dict, state: dict) -> list[str]:
    now = time.time()
    window = float(config.get("tts_guard_recent_text_window_seconds", DEFAULT_CONFIG["tts_guard_recent_text_window_seconds"]))
    texts: list[str] = []
    for key in ("text", "tts_text", "recent_tts_text"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    recent = state.get("recent_texts")
    if isinstance(recent, list):
        for item in recent:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            value = item.get("text") or item.get("tts_text")
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                timestamp = float(item.get("time", item.get("timestamp", now)))
            except (TypeError, ValueError):
                timestamp = now
            if now - timestamp <= window:
                texts.append(value.strip())
    return texts


def tts_guard_similarity(left: str, right: str) -> float:
    left_normalized = normalize_tts_guard_text(left)
    right_normalized = normalize_tts_guard_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        shorter = min(len(left_normalized), len(right_normalized))
        longer = max(len(left_normalized), len(right_normalized))
        return shorter / max(1, longer)
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def tts_guard_should_drop_text(config: dict, text: str) -> bool:
    state = load_tts_guard_state(config)
    threshold = float(config.get("tts_guard_similarity_threshold", DEFAULT_CONFIG["tts_guard_similarity_threshold"]))
    for recent_text in tts_guard_recent_texts(config, state):
        if tts_guard_similarity(text, recent_text) >= threshold:
            return True
    return False


def phrase_to_wav_bytes(phrase: Phrase, sample_rate: int) -> bytes:
    samples = np.asarray(phrase.audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def save_phrase_audio(audio_dir: Path, phrase: Phrase, sample_rate: int) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    stamp = phrase.started_at.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    duration = max(0.0, (phrase.ended_at - phrase.started_at).total_seconds())
    filename = f"{stamp}_dur{duration:.1f}s_peak{phrase.peak:.3f}.wav"
    path = audio_dir / filename
    path.write_bytes(phrase_to_wav_bytes(phrase, sample_rate))
    return path


def maybe_save_phrase_audio(config: dict, audio_dir: Path, phrase: Phrase) -> Path | None:
    if (
        not bool(config.get("save_audio_segments", DEFAULT_CONFIG["save_audio_segments"]))
        and not bool(config.get("speaker_recognition_enabled", False))
        and not bool(config.get("speaker_subtitle_enabled", False))
    ):
        return None
    try:
        saved_audio = save_phrase_audio(audio_dir, phrase, int(config["sample_rate"]))
    except Exception as exc:
        emit_status("audio_save_error", f"保存录音片段失败，已继续转写：{exc}")
        return None
    emit_status("audio_saved", f"已保存录音片段：{saved_audio}")
    retention_minutes = max(1.0, float(config.get("audio_retention_minutes", DEFAULT_CONFIG["audio_retention_minutes"])))
    try:
        cleanup_old_files(audio_dir, retention_minutes * 60.0)
    except Exception as exc:
        emit_status("audio_cleanup_error", f"清理旧录音失败，已继续转写：{exc}")
    return saved_audio


def load_speaker_registry(config: dict) -> dict:
    path = app_path(str(config.get("speaker_registry_file", DEFAULT_CONFIG["speaker_registry_file"]))).resolve()
    if not path.exists():
        return {"version": 1, "speakers": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "speakers": []}
    if not isinstance(data, dict):
        return {"version": 1, "speakers": []}
    speakers = data.get("speakers", [])
    data["speakers"] = [item for item in speakers if isinstance(item, dict)] if isinstance(speakers, list) else []
    data.setdefault("version", 1)
    return data


def write_speaker_registry(config: dict, registry: dict) -> None:
    path = app_path(str(config.get("speaker_registry_file", DEFAULT_CONFIG["speaker_registry_file"]))).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_identifier(value: str, fallback: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def speaker_profile_dir(config: dict, speaker_id: str) -> Path:
    registry_path = app_path(str(config.get("speaker_registry_file", DEFAULT_CONFIG["speaker_registry_file"]))).resolve()
    path = registry_path.parent / safe_identifier(speaker_id, "unknown")
    path.mkdir(parents=True, exist_ok=True)
    return path


def speaker_embedding_from_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < sample_rate * 0.5:
        raise RuntimeError("样本太短")
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    frame_size = max(256, int(sample_rate * 0.025))
    hop = max(128, int(sample_rate * 0.010))
    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))
    frame_count = 1 + max(0, (audio.size - frame_size) // hop)
    window = np.hamming(frame_size).astype(np.float32)
    vectors: list[np.ndarray] = []
    energies: list[float] = []
    for index in range(frame_count):
        frame = audio[index * hop:index * hop + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        energy = float(np.sqrt(np.mean(frame * frame)))
        if energy < 0.012:
            continue
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        selected = spectrum[(freqs >= 80.0) & (freqs <= min(7600.0, sample_rate / 2.0))]
        if selected.size < 32:
            continue
        vectors.append(np.log1p(np.array([float(np.mean(part)) for part in np.array_split(selected, 32)], dtype=np.float32)))
        energies.append(energy)
    if not vectors:
        raise RuntimeError("没有足够清晰的语音帧")
    matrix = np.vstack(vectors)
    energy_values = np.array(energies, dtype=np.float32)
    embedding = np.concatenate([
        matrix.mean(axis=0),
        matrix.std(axis=0),
        np.array([
            float(np.mean(energy_values)),
            float(np.std(energy_values)),
            float(np.percentile(energy_values, 75)),
            float(np.percentile(energy_values, 95)),
        ], dtype=np.float32),
    ]).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        raise RuntimeError("声纹向量为空")
    return embedding / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 0 else 0.0


def speaker_candidates(config: dict) -> list[tuple[dict, np.ndarray]]:
    registry = load_speaker_registry(config)
    candidates: list[tuple[dict, np.ndarray]] = []
    for profile in registry.get("speakers", []):
        embedding_file = str(profile.get("embedding_file", "")).strip()
        if not embedding_file:
            continue
        path = app_path(embedding_file).resolve() if not Path(embedding_file).is_absolute() else Path(embedding_file)
        if not path.exists():
            continue
        try:
            embedding = np.load(str(path)).astype(np.float32)
        except Exception:
            continue
        candidates.append((profile, embedding))
    return candidates


def match_speaker_embedding(query: np.ndarray, candidates: list[tuple[dict, np.ndarray]], threshold: float) -> dict:
    scored = [(profile, cosine_similarity(query, embedding)) for profile, embedding in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] >= threshold:
        profile, score = scored[0]
        return {
            "speaker_id": str(profile.get("id", "")),
            "speaker_name": str(profile.get("display_name") or profile.get("id") or ""),
            "speaker_kind": str(profile.get("kind") or "unknown"),
            "speaker_confidence": round(float(score), 3),
            "speaker_decision": "matched",
        }
    return {
        "speaker_id": "",
        "speaker_name": "未识别",
        "speaker_kind": "unknown",
        "speaker_confidence": round(float(scored[0][1]), 3) if scored else 0.0,
        "speaker_decision": "unknown",
    }


def identify_or_enroll_speaker(config: dict, phrase: Phrase, saved_audio: Path | None) -> dict | None:
    if not bool(config.get("speaker_recognition_enabled", False)):
        return None
    try:
        query = speaker_embedding_from_samples(np.asarray(phrase.audio, dtype=np.float32), int(config["sample_rate"]))
    except Exception as exc:
        emit_status("speaker_error", f"声纹计算失败：{exc}")
        return None
    threshold = float(config.get("speaker_match_threshold", 0.92))
    candidate_embeddings = speaker_candidates(config)
    result = match_speaker_embedding(query, candidate_embeddings, threshold)
    if result.get("speaker_decision") == "matched":
        emit_status("speaker_matched", f"声纹匹配：{result['speaker_name']} / {result['speaker_confidence']:.3f}")
        return result
    if not bool(config.get("speaker_auto_enroll_enabled", True)):
        emit_status("speaker_unknown", "声纹未匹配，未自动建档")
        return result
    prefix = safe_identifier(str(config.get("speaker_unknown_prefix", "unknown")), "unknown")
    speaker_id = f"{prefix}_{phrase.started_at:%Y%m%d_%H%M%S}"
    folder = speaker_profile_dir(config, speaker_id)
    embedding_file = folder / "embedding.npy"
    np.save(str(embedding_file), query)
    sample_paths = [str(saved_audio)] if saved_audio else []
    sample_metadata = [{
        "path": str(saved_audio),
        "source": "auto_enroll",
        "recorded_at": phrase.started_at.isoformat(),
        "duration_seconds": round((phrase.ended_at - phrase.started_at).total_seconds(), 3),
    }] if saved_audio else []
    profile = {
        "id": speaker_id,
        "display_name": speaker_id,
        "kind": "unknown",
        "character_id": "",
        "samples": sample_paths,
        "sample_metadata": sample_metadata,
        "embedding_file": str(embedding_file),
        "embedding_model": "local_spectral_v1",
        "embedding_sample_count": 1,
        "status": "auto_enrolled_pending_name",
        "created_by": "auto_enroll",
        "updated_at": datetime.now().isoformat(),
    }
    registry = load_speaker_registry(config)
    registry.setdefault("speakers", []).append(profile)
    write_speaker_registry(config, registry)
    emit_status("speaker_auto_enrolled", f"声纹未匹配，已自动创建：{speaker_id}")
    return {
        "speaker_id": speaker_id,
        "speaker_name": speaker_id,
        "speaker_kind": "unknown",
        "speaker_confidence": float(result.get("speaker_confidence", 0.0)),
        "speaker_decision": "auto_enrolled",
    }


def update_speaker_sample_transcript(config: dict, speaker_result: dict | None, audio_path: Path | None, text: str) -> None:
    if not speaker_result or not audio_path or not text.strip():
        return
    speaker_id = str(speaker_result.get("speaker_id") or "").strip()
    if not speaker_id:
        return
    try:
        registry = load_speaker_registry(config)
        audio_text = str(audio_path)
        updated = False
        for profile in registry.get("speakers", []):
            if str(profile.get("id") or "").strip() != speaker_id:
                continue
            metadata = profile.setdefault("sample_metadata", [])
            if not isinstance(metadata, list):
                metadata = []
                profile["sample_metadata"] = metadata
            target = None
            for item in metadata:
                if isinstance(item, dict) and str(item.get("path") or "").strip() == audio_text:
                    target = item
                    break
            if target is None:
                target = {"path": audio_text, "source": "transcription"}
                metadata.append(target)
            target["transcript_text"] = text.strip()
            target["transcribed_at"] = datetime.now().isoformat()
            profile["updated_at"] = datetime.now().isoformat()
            updated = True
            break
        if updated:
            write_speaker_registry(config, registry)
    except Exception as exc:
        emit_status("speaker_metadata_error", f"声纹样本文本回填失败：{exc}")


def upload_audio_for_diarization(config: dict, audio_path: Path) -> str:
    upload_url = str(config.get("speaker_diarization_upload_url", DEFAULT_CONFIG["speaker_diarization_upload_url"])).strip()
    if not upload_url:
        raise RuntimeError("speaker_diarization_upload_url is empty")
    data = json.dumps({
        "audio_path": str(audio_path),
        "namespace": "fennenote-asr",
        "remote_subdir": "voice-clone/fennenote",
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "FenneNote"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OumuQ upload HTTP {exc.code}: {error_text[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"OumuQ upload failed: {exc}") from exc
    payload = json.loads(response_text)
    audio_url = str(payload.get("audio_url") or payload.get("reference_audio_url") or "").strip()
    if not audio_url:
        raise RuntimeError(f"OumuQ upload response did not contain audio_url: {response_text[:500]}")
    return audio_url


def dashscope_transcription_endpoint(config: dict, suffix: str = "") -> str:
    return "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription" + suffix


def dashscope_task_endpoint(config: dict, task_id: str) -> str:
    return f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"


def submit_dashscope_diarization_task(config: dict, audio_url: str) -> str:
    api_key = str(config.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("DashScope API Key is empty.")
    model_name = str(config.get("speaker_diarization_model", DEFAULT_CONFIG["speaker_diarization_model"])).strip() or "paraformer-v2"
    parameters = {
        "diarization_enabled": True,
    }
    speaker_count = int(config.get("speaker_diarization_speaker_count", 0) or 0)
    if speaker_count > 0:
        parameters["speaker_count"] = speaker_count
    payload = {
        "model": model_name,
        "input": {"file_urls": [audio_url]},
        "parameters": parameters,
    }
    request = urllib.request.Request(
        dashscope_transcription_endpoint(config),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "User-Agent": "FenneNote",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope diarization submit HTTP {exc.code}: {error_text[:500]}") from exc
    payload = json.loads(response_text)
    task_id = str(payload.get("output", {}).get("task_id") or payload.get("task_id") or "").strip()
    if not task_id:
        raise RuntimeError(f"DashScope diarization submit did not return task_id: {response_text[:500]}")
    return task_id


def poll_dashscope_diarization_task(config: dict, task_id: str) -> dict:
    api_key = str(config.get("api_key", "")).strip()
    request = urllib.request.Request(
        dashscope_task_endpoint(config, task_id),
        data=b"",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "FenneNote"},
    )
    deadline = time.time() + 180.0
    last_payload: dict = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope diarization poll HTTP {exc.code}: {error_text[:500]}") from exc
        last_payload = json.loads(response_text)
        status = str(last_payload.get("output", {}).get("task_status") or last_payload.get("task_status") or "").upper()
        if status in {"SUCCEEDED", "SUCCESS"}:
            return last_payload
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"DashScope diarization task failed: {response_text[:500]}")
        time.sleep(2.0)
    raise RuntimeError(f"DashScope diarization task timed out: {json.dumps(last_payload, ensure_ascii=False)[:500]}")


def download_diarization_result(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "FenneNote"})
    with urllib.request.urlopen(request, timeout=60) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    return json.loads(response_text)


def collect_sentence_items(value) -> list[dict]:
    items: list[dict] = []
    if isinstance(value, dict):
        for key in ("sentences", "sentence", "transcripts", "results"):
            child = value.get(key)
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        items.extend(collect_sentence_items(item))
        if any(key in value for key in ("begin_time", "end_time", "speaker_id", "text")):
            items.append(value)
    elif isinstance(value, list):
        for item in value:
            items.extend(collect_sentence_items(item))
    return items


def ms_value(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number / 1000.0 if number > 100.0 else number


def speaker_result_for_time_range(config: dict, phrase: Phrase, start_seconds: float, end_seconds: float, fallback_name: str) -> dict:
    sample_rate = int(config["sample_rate"])
    start = max(0, int(start_seconds * sample_rate))
    end = min(len(phrase.audio), int(end_seconds * sample_rate))
    chunk = np.asarray(phrase.audio[start:end], dtype=np.float32)
    if chunk.size < sample_rate * 0.4:
        return {"speaker_id": "", "speaker_name": fallback_name, "speaker_kind": "unknown", "speaker_confidence": 0.0, "speaker_decision": "diarized"}
    candidates = speaker_candidates(config)
    if not candidates:
        return {"speaker_id": "", "speaker_name": fallback_name, "speaker_kind": "unknown", "speaker_confidence": 0.0, "speaker_decision": "diarized"}
    try:
        query = speaker_embedding_from_samples(chunk, sample_rate)
    except Exception:
        return {"speaker_id": "", "speaker_name": fallback_name, "speaker_kind": "unknown", "speaker_confidence": 0.0, "speaker_decision": "diarized"}
    result = match_speaker_embedding(query, candidates, float(config.get("speaker_match_threshold", 0.92)))
    if result.get("speaker_decision") == "matched":
        return result
    return {"speaker_id": "", "speaker_name": fallback_name, "speaker_kind": "unknown", "speaker_confidence": result.get("speaker_confidence", 0.0), "speaker_decision": "diarized"}


def transcribe_phrase_with_dashscope_diarization(config: dict, phrase: Phrase, audio_path: Path, converter: OpenCC | None) -> tuple[str, list[dict]]:
    audio_url = upload_audio_for_diarization(config, audio_path)
    emit_status("diarization_upload", "已上传录音片段，正在提交说话人分离")
    task_id = submit_dashscope_diarization_task(config, audio_url)
    emit_status("diarization_submitted", f"说话人分离任务已提交：{task_id}")
    task_payload = poll_dashscope_diarization_task(config, task_id)
    results = task_payload.get("output", {}).get("results", [])
    result_payload = task_payload
    if isinstance(results, list) and results:
        url = str(results[0].get("transcription_url") or results[0].get("url") or "").strip()
        if url:
            result_payload = download_diarization_result(url)
    sentence_items = collect_sentence_items(result_payload)
    turns: list[dict] = []
    for item in sentence_items:
        text = str(item.get("text") or item.get("sentence") or "").strip()
        if not text:
            continue
        text = simplify_text(text, converter)
        start_seconds = ms_value(item.get("begin_time", item.get("start_time", item.get("start", 0.0))))
        end_seconds = ms_value(item.get("end_time", item.get("stop_time", item.get("end", start_seconds))))
        if end_seconds <= start_seconds:
            continue
        provider_speaker = str(item.get("speaker_id") or item.get("speaker") or item.get("channel_id") or "").strip()
        fallback_name = f"说话人 {provider_speaker}" if provider_speaker else "未识别"
        speaker_result = speaker_result_for_time_range(config, phrase, start_seconds, end_seconds, fallback_name)
        turns.append({
            "start": start_seconds,
            "end": end_seconds,
            "provider_speaker_id": provider_speaker,
            "text": text,
            **speaker_result,
        })
    full_text = " ".join(str(turn.get("text", "")).strip() for turn in turns if str(turn.get("text", "")).strip()).strip()
    if not full_text:
        full_text = simplify_text(extract_dashscope_text(result_payload), converter)
    return full_text, turns


def phrase_to_audio_data_url(phrase: Phrase, sample_rate: int) -> str:
    encoded = base64.b64encode(phrase_to_wav_bytes(phrase, sample_rate)).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def dashscope_generation_url(config: dict) -> str:
    base_url = str(config.get("api_base_url", "")).strip()
    native_path = "/api/v1/services/aigc/multimodal-generation/generation"
    if native_path in base_url:
        return base_url
    return "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts: list[str] = []
        for key in ("text", "transcript", "content"):
            value = content.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, dict)):
                nested = content_to_text(value)
                if nested:
                    parts.append(nested)
        return " ".join(part.strip() for part in parts if part.strip())
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return " ".join(part.strip() for part in parts if part.strip())
    return ""


def extract_dashscope_text(payload: dict) -> str:
    output = payload.get("output", {})
    direct = content_to_text(output.get("text")) if isinstance(output, dict) else ""
    if direct:
        return direct.strip()

    choice_sets = []
    if isinstance(output, dict):
        choice_sets.append(output.get("choices"))
    choice_sets.append(payload.get("choices"))
    for choices in choice_sets:
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            text = content_to_text(message.get("content") if isinstance(message, dict) else choice.get("content"))
            if text:
                return text.strip()

    fallback_parts: list[str] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"text", "transcript"} and isinstance(item, str):
                    fallback_parts.append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return " ".join(part.strip() for part in fallback_parts if part.strip()).strip()


def transcribe_phrase_with_dashscope(config: dict, phrase: Phrase, language: str | None, converter: OpenCC | None) -> str:
    api_key = str(config.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("DashScope API Key is empty.")
    model_name = str(config.get("api_model", "")).strip() or "qwen3-asr-flash"
    prompt = str(config.get("initial_prompt", "")).strip() or "Please transcribe this audio accurately."
    content = [{"audio": phrase_to_audio_data_url(phrase, int(config["sample_rate"]))}]
    payload = {
        "model": model_name,
        "input": {
            "messages": [
                {"role": "system", "content": [{"text": prompt}]},
                {"role": "user", "content": content},
            ]
        },
    }
    asr_options = {"enable_itn": False}
    if language:
        asr_options["language"] = language
    payload["parameters"] = {"asr_options": asr_options}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        dashscope_generation_url(config),
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FenneNote",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope HTTP {exc.code}: {error_text[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"DashScope request failed: {exc}") from exc

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DashScope returned non-JSON response: {response_text[:500]}") from exc
    text = extract_dashscope_text(response_payload)
    if not text:
        raise RuntimeError(f"DashScope response did not contain transcript text: {response_text[:500]}")
    return simplify_text(text, converter)


def mimo_chat_completions_url(config: dict) -> str:
    configured_base_url = str(config.get("api_base_url") or "").strip()
    if not configured_base_url or "dashscope.aliyuncs.com" in configured_base_url:
        configured_base_url = os.environ.get("MIMO_BASE_URL", "") or "https://api.xiaomimimo.com/v1"
    base_url = configured_base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def transcribe_phrase_with_mimo(config: dict, phrase: Phrase, language: str | None, converter: OpenCC | None) -> str:
    api_key = os.environ.get("MIMO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MIMO_API_KEY is empty.")
    model_name = str(config.get("api_model", "")).strip() or "mimo-v2.5-asr"
    mimo_language = language if language in {"auto", "zh", "en"} else "auto"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": phrase_to_audio_data_url(phrase, int(config["sample_rate"]))
                        },
                    }
                ],
            }
        ],
        "asr_options": {"language": mimo_language},
    }
    request = urllib.request.Request(
        mimo_chat_completions_url(config),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FenneNote",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiMo ASR HTTP {exc.code}: {error_text[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"MiMo ASR request failed: {exc}") from exc

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MiMo ASR returned non-JSON response: {response_text[:500]}") from exc
    text = extract_dashscope_text(response_payload)
    if not text:
        raise RuntimeError(f"MiMo ASR response did not contain transcript text: {response_text[:500]}")
    return simplify_text(text, converter)


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
    mixed_input_enabled = bool(config.get("mixed_input_enabled", False))
    system_audio_device = config.get("system_audio_device")
    system_audio_gain = float(config.get("system_audio_gain", 1.0))
    device = config.get("mic_device")
    system_audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, callback_time, status):
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        if input_gain != 1.0:
            chunk = np.clip(chunk * input_gain, -1.0, 1.0)
        audio_queue.put(chunk)

    def system_audio_callback(indata, frames, callback_time, status):
        if status:
            print(f"System audio warning: {status}", file=sys.stderr)
        chunk = np.asarray(indata, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        if system_audio_gain != 1.0:
            chunk = np.clip(chunk * system_audio_gain, -1.0, 1.0)
        system_audio_queue.put(chunk.copy())

    def latest_system_chunk(target_frames: int) -> np.ndarray:
        latest: np.ndarray | None = None
        while True:
            try:
                latest = system_audio_queue.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return np.zeros(target_frames, dtype=np.float32)
        if latest.size < target_frames:
            latest = np.pad(latest, (0, target_frames - latest.size))
        elif latest.size > target_frames:
            latest = latest[:target_frames]
        return latest.astype(np.float32)

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
        with contextlib.ExitStack() as stack:
            stack.enter_context(sd.InputStream(
                device=device,
                channels=1,
                samplerate=sample_rate,
                blocksize=chunk_frames,
                dtype="float32",
                callback=callback,
            ))
            if mixed_input_enabled and system_audio_device is not None:
                try:
                    stack.enter_context(sd.InputStream(
                        device=system_audio_device,
                        channels=1,
                        samplerate=sample_rate,
                        blocksize=chunk_frames,
                        dtype="float32",
                        callback=system_audio_callback,
                    ))
                    emit_status("mixed_input_ready", f"已启用麦克风 + 电脑声音混合输入：设备 {system_audio_device}")
                except Exception as exc:
                    emit_status("mixed_input_error", f"电脑声音输入打开失败，仅使用麦克风：{exc}")
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
                        emit_status("paused", "已暂停监听")
                    elif command in {"r", "resume"}:
                        paused = False
                        emit_status("listening", "已继续监听")

                try:
                    chunk = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if paused:
                    continue

                if mixed_input_enabled and system_audio_device is not None:
                    system_chunk = latest_system_chunk(len(chunk))
                    chunk = np.clip(chunk + system_chunk, -1.0, 1.0)

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
                    emit_status(
                        "recording",
                        f"正在听：达到录音阈值，安静 {float(config.get('transcribe_pause_seconds', 0.5)):.1f} 秒后切句",
                    )

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
                    else:
                        emit_status("listening", "声音未达到转写阈值，继续监听")
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
    model_source = str(config.get("model_source", "local") or "local").lower()
    if model_source == "api":
        api_provider = str(config.get("api_provider", config.get("api_provider_id", "")) or "").lower()
        if api_provider not in {"dashscope", "千问 / dashscope", "mimo", "xiaomi_mimo", "xiaomi mimo"}:
            raise RuntimeError(f"Unsupported API provider: {api_provider or '(empty)'}. DashScope and MiMo are currently supported.")
        if api_provider in {"dashscope", "千问 / dashscope"} and not str(config.get("api_key", "")).strip():
            raise RuntimeError("DashScope API Key is empty.")
        if api_provider in {"mimo", "xiaomi_mimo", "xiaomi mimo"} and not os.environ.get("MIMO_API_KEY", "").strip():
            raise RuntimeError("MIMO_API_KEY is empty.")
        language = config.get("language_mode", config.get("language", "zh"))
        if language in {"auto", "", None}:
            language = None
        converter = OpenCC("t2s") if bool(config.get("simplify_chinese", True)) else None
        stop_event = threading.Event()
        phrases = collect_phrases(config, stop_event)
        provider_label = "MiMo" if api_provider in {"mimo", "xiaomi_mimo", "xiaomi mimo"} else "DashScope"
        default_model = "mimo-v2.5-asr" if provider_label == "MiMo" else "qwen3-asr-flash"
        emit_status("api_ready", f"API 模型已就绪：{provider_label} / {config.get('api_model', default_model)}")
        emit_status("output_ready", f"转写记录目录：{output_dir}")
        emit_status("cache_ready", f"本地缓存：{storage_dirs['cache']}")

        while not stop_event.is_set():
            try:
                phrase = phrases.get(timeout=0.2)
            except queue.Empty:
                continue
            if tts_guard_should_skip_audio(config):
                emit_status("tts_guard_skip", "TTS guard active; skipped one microphone phrase before transcription")
                continue

            emit_status("transcribing", f"正在转写：{(phrase.ended_at - phrase.started_at).total_seconds():.1f} 秒音频，峰值 {phrase.peak:.3f}")
            saved_audio = maybe_save_phrase_audio(config, storage_dirs["audio"], phrase)
            speaker_result = identify_or_enroll_speaker(config, phrase, saved_audio)
            speaker_turns: list[dict] = []
            if bool(config.get("speaker_subtitle_enabled", False)) and saved_audio:
                try:
                    text, speaker_turns = transcribe_phrase_with_dashscope_diarization(config, phrase, saved_audio, converter)
                except Exception as exc:
                    emit_status("diarization_error", f"说话人分离失败，回退普通转写：{exc}")
                    try:
                        if api_provider in {"mimo", "xiaomi_mimo", "xiaomi mimo"}:
                            text = transcribe_phrase_with_mimo(config, phrase, language, converter)
                        else:
                            text = transcribe_phrase_with_dashscope(config, phrase, language, converter)
                    except Exception as fallback_exc:
                        emit_status("api_error", f"API 转写失败：{fallback_exc}")
                        continue
            else:
                try:
                    if api_provider in {"mimo", "xiaomi_mimo", "xiaomi mimo"}:
                        text = transcribe_phrase_with_mimo(config, phrase, language, converter)
                    else:
                        text = transcribe_phrase_with_dashscope(config, phrase, language, converter)
                except Exception as exc:
                    emit_status("api_error", f"API 转写失败：{exc}")
                    continue
            text = apply_transcript_corrections(text, config)
            for turn in speaker_turns:
                if isinstance(turn, dict) and "text" in turn:
                    turn["text"] = apply_transcript_corrections(str(turn["text"]), config)
            if not should_keep_text(text, language, config):
                emit_status("discarded", "本段已转写但没有保留有效文字，已丢弃")
                continue
            if tts_guard_should_drop_text(config, text):
                emit_status("tts_guard_echo", "TTS guard dropped a transcript similar to recent TTS text")
                continue
            update_speaker_sample_transcript(config, speaker_result, saved_audio, text)
            append_line(output_dir, phrase.started_at, text, speaker_result)
            post_rabiroute_event(config, phrase, text, speaker_result, speaker_turns)
            emit_transcript_preview(phrase.started_at, text, speaker_result, speaker_turns)
            emit_status("written", "转写完成，已写入今日文本")
        return
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
    emit_status("output_ready", f"转写记录目录：{output_dir}")
    emit_status("cache_ready", f"本地缓存：{storage_dirs['cache']}")

    while not stop_event.is_set():
        try:
            phrase = phrases.get(timeout=0.2)
        except queue.Empty:
            continue
        if tts_guard_should_skip_audio(config):
            emit_status("tts_guard_skip", "TTS guard active; skipped one microphone phrase before transcription")
            continue

        initial_prompt = config.get("initial_prompt") or None
        emit_status("transcribing", f"正在转写：{(phrase.ended_at - phrase.started_at).total_seconds():.1f} 秒音频，峰值 {phrase.peak:.3f}")
        saved_audio = maybe_save_phrase_audio(config, storage_dirs["audio"], phrase)
        speaker_result = identify_or_enroll_speaker(config, phrase, saved_audio)
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
        text = apply_transcript_corrections(text, config)
        if not should_keep_text(text, language, config):
            emit_status("discarded", "本段已转写但没有保留有效文字，已丢弃")
            continue
        if tts_guard_should_drop_text(config, text):
            emit_status("tts_guard_echo", "TTS guard dropped a transcript similar to recent TTS text")
            continue
        speaker_turns: list[dict] = []
        update_speaker_sample_transcript(config, speaker_result, saved_audio, text)
        append_line(output_dir, phrase.started_at, text, speaker_result)
        post_rabiroute_event(config, phrase, text, speaker_result, speaker_turns)
        emit_transcript_preview(phrase.started_at, text, speaker_result, speaker_turns)
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
