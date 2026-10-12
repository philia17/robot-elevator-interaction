"""Portable flat configuration loading without third-party dependencies."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_USB_VID = "0c76"
DEFAULT_USB_PID = "161f"
DEFAULT_PULSE_SOURCE = "alsa_input.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"
DEFAULT_PULSE_SINK = "alsa_output.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"


@dataclass(frozen=True)
class VoiceControlConfig:
    usb_vid: str = DEFAULT_USB_VID
    usb_pid: str = DEFAULT_USB_PID
    pulse_source: str = DEFAULT_PULSE_SOURCE
    pulse_sink: str = DEFAULT_PULSE_SINK
    wake_prompt: str = ""
    confirm_prompt: str = ""
    floor_confirm_prompt_dir: str = ""
    confirmed_prompt: str = ""
    mismatch_prompt: str = ""
    retry_prompt: str = ""
    timeout_prompt: str = ""
    cancel_prompt: str = ""
    stop_prompt: str = ""
    english_prompt_dir: str = ""
    english_floor_confirm_prompt_dir: str = ""
    prompt_volume_percent: int = 50
    playback_timeout_seconds: float = 10.0
    post_playback_guard_seconds: float = 0.5
    vad_rms_threshold: int = 300
    vad_chunk_ms: int = 30
    vad_start_ms: int = 90
    vad_silence_ms: int = 600
    vad_preroll_ms: int = 240
    vad_min_speech_ms: int = 150
    vad_max_utterance_seconds: float = 8.0
    short_unknown_max_seconds: float = 1.0
    short_unknown_max_chars: int = 2
    # Legacy compatibility setting retained for old profiles. Punctuation-only
    # ASR is always silent from 0.3.3 onward; RMS does not determine semantics.
    no_speech_retry_peak_rms: int = 10_000
    session_awake_timeout_seconds: float = 30.0
    dry_run: bool = True
    floor_min: int = -2
    floor_max: int = 40
    confirmation_ttl_seconds: float = 30.0
    audit_dir: str = ""
    model_dir: str = ""
    model: str = ""
    tokens: str = ""
    vad: str = ""
    # Stage 1 compatibility alias. New profiles should use ``model``.
    model_path: str = ""
    model_sha256: str = ""
    asr_num_threads: int = 2
    asr_provider: str = "cpu"
    max_audio_bytes: int = 16 * 1024 * 1024

    def normalized_usb_vid(self) -> str:
        return self.usb_vid.lower().removeprefix("0x")

    def normalized_usb_pid(self) -> str:
        return self.usb_pid.lower().removeprefix("0x")

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.floor_min > self.floor_max:
            issues.append("floor_min must be <= floor_max")
        if self.confirmation_ttl_seconds <= 0:
            issues.append("confirmation_ttl_seconds must be positive")
        if self.asr_num_threads <= 0:
            issues.append("asr_num_threads must be positive")
        if self.max_audio_bytes <= 0:
            issues.append("max_audio_bytes must be positive")
        if not 0 <= self.prompt_volume_percent <= 100:
            issues.append("prompt_volume_percent must be between 0 and 100")
        if self.playback_timeout_seconds <= 0:
            issues.append("playback_timeout_seconds must be positive")
        if not 0 <= self.post_playback_guard_seconds <= 5:
            issues.append("post_playback_guard_seconds must be between 0 and 5")
        if not 1 <= self.vad_rms_threshold <= 32767:
            issues.append("vad_rms_threshold must be between 1 and 32767")
        if not 10 <= self.vad_chunk_ms <= 1000:
            issues.append("vad_chunk_ms must be between 10 and 1000")
        if self.vad_start_ms < self.vad_chunk_ms:
            issues.append("vad_start_ms must be at least vad_chunk_ms")
        if self.vad_silence_ms < self.vad_chunk_ms:
            issues.append("vad_silence_ms must be at least vad_chunk_ms")
        if self.vad_preroll_ms < 0:
            issues.append("vad_preroll_ms must not be negative")
        if self.vad_min_speech_ms <= 0:
            issues.append("vad_min_speech_ms must be positive")
        if self.vad_max_utterance_seconds <= 0:
            issues.append("vad_max_utterance_seconds must be positive")
        if self.short_unknown_max_seconds <= 0:
            issues.append("short_unknown_max_seconds must be positive")
        if self.short_unknown_max_chars <= 0:
            issues.append("short_unknown_max_chars must be positive")
        if not 0 <= self.no_speech_retry_peak_rms <= 32767:
            issues.append("no_speech_retry_peak_rms must be between 0 and 32767")
        if self.session_awake_timeout_seconds <= 0:
            issues.append("session_awake_timeout_seconds must be positive")
        if not re.fullmatch(r"[0-9a-f]{4}", self.normalized_usb_vid()):
            issues.append("usb_vid must be four hexadecimal digits")
        if not re.fullmatch(r"[0-9a-f]{4}", self.normalized_usb_pid()):
            issues.append("usb_pid must be four hexadecimal digits")
        for name, value in (("pulse_source", self.pulse_source),
                            ("pulse_sink", self.pulse_sink)):
            if not value or "REPLACE_ON_AGX" in value:
                issues.append(f"{name} must be replaced with a discovered stable name")
            if re.fullmatch(r"(?:hw:)?\d+(?:,\d+)?", value):
                issues.append(f"{name} must not use an ALSA card number")
        if self.model_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", self.model_sha256):
            issues.append("model_sha256 must be empty or 64 hexadecimal digits")
        for name, value in (
            ("model_dir", self.model_dir),
            ("model", self.model),
            ("tokens", self.tokens),
            ("vad", self.vad),
            ("model_path", self.model_path),
            ("wake_prompt", self.wake_prompt),
            ("confirm_prompt", self.confirm_prompt),
            ("floor_confirm_prompt_dir", self.floor_confirm_prompt_dir),
            ("confirmed_prompt", self.confirmed_prompt),
            ("mismatch_prompt", self.mismatch_prompt),
            ("retry_prompt", self.retry_prompt),
            ("timeout_prompt", self.timeout_prompt),
            ("cancel_prompt", self.cancel_prompt),
            ("stop_prompt", self.stop_prompt),
            ("english_prompt_dir", self.english_prompt_dir),
            ("english_floor_confirm_prompt_dir", self.english_floor_confirm_prompt_dir),
        ):
            if "REPLACE_ON_AGX" in value:
                issues.append(f"{name} must be replaced with an AGX-local path")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> VoiceControlConfig:
    env_path = os.environ.get("VOICE_CONTROL_CONFIG")
    config_path = Path(path) if path else (Path(env_path) if env_path else None)
    if config_path is None:
        return VoiceControlConfig()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = _load_flat_yaml(config_path)
    return config_from_mapping(data)


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load the flat scalar YAML subset used by the shipped config templates."""
    data: dict[str, Any] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"unsupported YAML at line {line_no}: expected key: value")
        key, value = (part.strip() for part in line.split(":", 1))
        data[key] = _yaml_scalar(value)
    return data


def _yaml_scalar(value: str) -> Any:
    if (value.startswith("\"") and value.endswith("\"")) or (
            value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def config_from_mapping(data: Mapping[str, Any]) -> VoiceControlConfig:
    return VoiceControlConfig(
        usb_vid=str(data.get("usb_vid", DEFAULT_USB_VID)),
        usb_pid=str(data.get("usb_pid", DEFAULT_USB_PID)),
        pulse_source=str(data.get("pulse_source", DEFAULT_PULSE_SOURCE)),
        pulse_sink=str(data.get("pulse_sink", DEFAULT_PULSE_SINK)),
        wake_prompt=str(data.get("wake_prompt", "")),
        confirm_prompt=str(data.get("confirm_prompt", "")),
        floor_confirm_prompt_dir=str(data.get("floor_confirm_prompt_dir", "")),
        confirmed_prompt=str(data.get("confirmed_prompt", "")),
        mismatch_prompt=str(data.get("mismatch_prompt", "")),
        retry_prompt=str(data.get("retry_prompt", "")),
        timeout_prompt=str(data.get("timeout_prompt", "")),
        cancel_prompt=str(data.get("cancel_prompt", "")),
        stop_prompt=str(data.get("stop_prompt", "")),
        english_prompt_dir=str(data.get("english_prompt_dir", "")),
        english_floor_confirm_prompt_dir=str(
            data.get("english_floor_confirm_prompt_dir", "")
        ),
        prompt_volume_percent=int(data.get("prompt_volume_percent", 50)),
        playback_timeout_seconds=float(data.get("playback_timeout_seconds", 10.0)),
        post_playback_guard_seconds=float(
            data.get("post_playback_guard_seconds", 0.5)
        ),
        vad_rms_threshold=int(data.get("vad_rms_threshold", 300)),
        vad_chunk_ms=int(data.get("vad_chunk_ms", 30)),
        vad_start_ms=int(data.get("vad_start_ms", 90)),
        vad_silence_ms=int(data.get("vad_silence_ms", 600)),
        vad_preroll_ms=int(data.get("vad_preroll_ms", 240)),
        vad_min_speech_ms=int(data.get("vad_min_speech_ms", 150)),
        vad_max_utterance_seconds=float(data.get("vad_max_utterance_seconds", 8.0)),
        short_unknown_max_seconds=float(data.get("short_unknown_max_seconds", 1.0)),
        short_unknown_max_chars=int(data.get("short_unknown_max_chars", 2)),
        no_speech_retry_peak_rms=int(data.get("no_speech_retry_peak_rms", 10_000)),
        session_awake_timeout_seconds=float(
            data.get("session_awake_timeout_seconds", 30.0)
        ),
        dry_run=_as_bool(data.get("dry_run", True)),
        floor_min=int(data.get("floor_min", -2)),
        floor_max=int(data.get("floor_max", 40)),
        confirmation_ttl_seconds=float(data.get("confirmation_ttl_seconds", 30.0)),
        audit_dir=str(data.get("audit_dir", "")),
        model_dir=str(data.get("model_dir", "")),
        model=str(data.get("model", "")),
        tokens=str(data.get("tokens", data.get("tokens_path", ""))),
        vad=str(data.get("vad", data.get("vad_model_path", ""))),
        model_path=str(data.get("model_path", "")),
        model_sha256=str(data.get("model_sha256", "")),
        asr_num_threads=int(data.get("asr_num_threads", 2)),
        asr_provider=str(data.get("asr_provider", "cpu")),
        max_audio_bytes=int(data.get("max_audio_bytes", 16 * 1024 * 1024)),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError("dry_run must be true or false")
