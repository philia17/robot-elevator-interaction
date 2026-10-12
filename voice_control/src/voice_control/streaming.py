"""Bounded energy-based utterance capture for the persistent dry-run session."""

from __future__ import annotations

import math
import os
import select
import subprocess
import sys
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Callable

from voice_control.audio import (
    BYTES_PER_SECOND,
    CHANNELS,
    MAX_STDERR_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    AudioCaptureError,
    AudioSizeLimitError,
    PcmCapture,
    _read_at_most,
    _stop_process,
)


@dataclass(frozen=True)
class EnergyVadSettings:
    rms_threshold: int = 300
    chunk_ms: int = 30
    start_ms: int = 90
    silence_ms: int = 600
    preroll_ms: int = 240
    min_speech_ms: int = 150
    max_utterance_seconds: float = 8.0

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not 1 <= self.rms_threshold <= 32_767:
            issues.append("vad_rms_threshold must be between 1 and 32767")
        if not 10 <= self.chunk_ms <= 1000:
            issues.append("vad_chunk_ms must be between 10 and 1000")
        if self.start_ms < self.chunk_ms:
            issues.append("vad_start_ms must be at least vad_chunk_ms")
        if self.silence_ms < self.chunk_ms:
            issues.append("vad_silence_ms must be at least vad_chunk_ms")
        if self.preroll_ms < 0:
            issues.append("vad_preroll_ms must not be negative")
        if self.min_speech_ms <= 0:
            issues.append("vad_min_speech_ms must be positive")
        if self.max_utterance_seconds <= 0:
            issues.append("vad_max_utterance_seconds must be positive")
        return issues


@dataclass(frozen=True)
class DetectedUtterance:
    capture: PcmCapture
    peak_rms: int
    end_reason: str


@dataclass(frozen=True)
class DetectorEvent:
    rms: int
    voice_started: bool = False
    utterance: DetectedUtterance | None = None


def pcm_rms(chunk: bytes) -> int:
    """Return RMS for signed 16-bit little-endian mono PCM without NumPy."""
    if not chunk or len(chunk) % SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM chunk must contain complete signed 16-bit samples")
    samples = array("h")
    samples.frombytes(chunk)
    if sys.byteorder == "big":
        samples.byteswap()
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return round(math.sqrt(mean_square))


class EnergyUtteranceDetector:
    """Pure chunk detector; callers own the microphone process lifecycle."""

    def __init__(self, settings: EnergyVadSettings) -> None:
        issues = settings.validate()
        if issues:
            raise ValueError("; ".join(issues))
        self.settings = settings
        self.chunk_frames = round(SAMPLE_RATE * settings.chunk_ms / 1000)
        self.chunk_bytes = self.chunk_frames * CHANNELS * SAMPLE_WIDTH_BYTES
        self._start_chunks = math.ceil(settings.start_ms / settings.chunk_ms)
        self._silence_chunks = math.ceil(settings.silence_ms / settings.chunk_ms)
        self._min_voice_chunks = math.ceil(settings.min_speech_ms / settings.chunk_ms)
        self._max_chunks = math.ceil(
            settings.max_utterance_seconds * 1000 / settings.chunk_ms
        )
        preroll_chunks = math.ceil(settings.preroll_ms / settings.chunk_ms)
        self._preroll: deque[bytes] = deque(maxlen=preroll_chunks)
        self._audio: list[bytes] = []
        self._started = False
        self._consecutive_voice = 0
        self._silence = 0
        self._voice_chunks = 0
        self._peak_rms = 0

    @property
    def started(self) -> bool:
        return self._started

    def feed(self, chunk: bytes) -> DetectorEvent:
        if len(chunk) != self.chunk_bytes:
            raise ValueError(
                f"expected {self.chunk_bytes} PCM bytes, received {len(chunk)}"
            )
        rms = pcm_rms(chunk)
        voiced = rms >= self.settings.rms_threshold

        if not self._started:
            self._preroll.append(chunk)
            self._consecutive_voice = self._consecutive_voice + 1 if voiced else 0
            if self._consecutive_voice < self._start_chunks:
                return DetectorEvent(rms=rms)
            self._started = True
            self._audio = list(self._preroll)
            self._preroll.clear()
            self._voice_chunks = self._consecutive_voice
            self._peak_rms = rms
            return DetectorEvent(rms=rms, voice_started=True)

        self._audio.append(chunk)
        self._peak_rms = max(self._peak_rms, rms)
        if voiced:
            self._voice_chunks += 1
            self._silence = 0
        else:
            self._silence += 1

        if len(self._audio) >= self._max_chunks:
            return DetectorEvent(rms=rms, utterance=self._finish("max_duration"))
        if self._silence >= self._silence_chunks:
            if self._voice_chunks >= self._min_voice_chunks:
                return DetectorEvent(rms=rms, utterance=self._finish("silence"))
            self._reset()
        return DetectorEvent(rms=rms)

    def _finish(self, reason: str) -> DetectedUtterance:
        utterance = DetectedUtterance(
            capture=PcmCapture(b"".join(self._audio)),
            peak_rms=self._peak_rms,
            end_reason=reason,
        )
        self._reset()
        return utterance

    def _reset(self) -> None:
        self._preroll.clear()
        self._audio = []
        self._started = False
        self._consecutive_voice = 0
        self._silence = 0
        self._voice_chunks = 0
        self._peak_rms = 0


def record_utterance(
    source: str,
    settings: EnergyVadSettings,
    *,
    max_audio_bytes: int,
    wait_timeout_seconds: float | None = None,
    on_voice_start: Callable[[int], None] | None = None,
) -> DetectedUtterance | None:
    """Wait for one utterance, then stop capture before ASR or playback.

    Returns ``None`` only when ``wait_timeout_seconds`` elapses before speech.
    Raw PCM remains in memory and the ``parec`` process is always terminated.
    """
    if not source or "REPLACE_ON_AGX" in source:
        raise AudioCaptureError("pulse_source is empty or still an AGX placeholder")
    if max_audio_bytes <= 0:
        raise ValueError("max_audio_bytes must be positive")
    if wait_timeout_seconds is not None and wait_timeout_seconds <= 0:
        raise ValueError("wait_timeout_seconds must be positive")
    max_utterance_bytes = math.ceil(settings.max_utterance_seconds * BYTES_PER_SECOND)
    if max_utterance_bytes > max_audio_bytes:
        raise AudioSizeLimitError(
            f"maximum utterance PCM size {max_utterance_bytes} exceeds limit "
            f"{max_audio_bytes} bytes"
        )

    detector = EnergyUtteranceDetector(settings)
    command = [
        "parec",
        f"--device={source}",
        "--format=s16le",
        f"--rate={SAMPLE_RATE}",
        f"--channels={CHANNELS}",
        "--raw",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise AudioCaptureError("parec was not found; install PulseAudio utilities") from exc
    except OSError as exc:
        raise AudioCaptureError(f"failed to start parec: {exc}") from exc

    if process.stdout is None or process.stderr is None:
        _stop_process(process, 2.0)
        raise AudioCaptureError("parec pipes were not created")

    stderr_parts: list[bytes] = []
    stderr_reader = threading.Thread(
        target=_read_at_most,
        args=(process.stderr, MAX_STDERR_BYTES + 1, stderr_parts),
        daemon=True,
    )
    stderr_reader.start()
    pending = bytearray()
    started_waiting = time.monotonic()
    try:
        while True:
            if (
                wait_timeout_seconds is not None
                and not detector.started
                and time.monotonic() - started_waiting >= wait_timeout_seconds
            ):
                return None
            ready, _, _ = select.select([process.stdout], [], [], 0.25)
            if not ready:
                if process.poll() is not None:
                    raise AudioCaptureError("parec stream ended unexpectedly")
                continue
            data = os.read(process.stdout.fileno(), detector.chunk_bytes * 8)
            if not data:
                raise AudioCaptureError("parec returned end-of-stream")
            pending.extend(data)
            while len(pending) >= detector.chunk_bytes:
                chunk = bytes(pending[:detector.chunk_bytes])
                del pending[:detector.chunk_bytes]
                event = detector.feed(chunk)
                if event.voice_started and on_voice_start:
                    on_voice_start(event.rms)
                if event.utterance is not None:
                    if len(event.utterance.capture.data) > max_audio_bytes:
                        raise AudioSizeLimitError("captured utterance exceeded max_audio_bytes")
                    return event.utterance
    finally:
        _stop_process(process, 2.0)
        stderr_reader.join(timeout=2.0)
