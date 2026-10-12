"""Bounded fixed-duration PCM capture from a stable PulseAudio source."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import BinaryIO

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES
DEFAULT_MAX_AUDIO_BYTES = 16 * 1024 * 1024
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 2.0
MAX_STDERR_BYTES = 64 * 1024


class AudioCaptureError(RuntimeError):
    """Base error for microphone capture failures."""


class AudioCaptureTimeout(AudioCaptureError):
    """The recorder failed to produce the requested audio before the deadline."""


class AudioSizeLimitError(AudioCaptureError):
    """The requested or returned audio exceeded the configured safety limit."""


@dataclass(frozen=True)
class PcmCapture:
    data: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    sample_width_bytes: int = SAMPLE_WIDTH_BYTES

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / BYTES_PER_SECOND


def _read_at_most(stream: BinaryIO, size: int, destination: list[bytes]) -> None:
    destination.append(stream.read(size))


def _stderr_text(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace").strip()
    return text[-2000:]


def _stop_process(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise AudioCaptureError("parec could not be terminated") from exc


def record_pcm(
    source: str,
    seconds: float,
    *,
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
    startup_grace_seconds: float = 3.0,
    cleanup_timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
) -> PcmCapture:
    """Capture exactly ``seconds`` of 16 kHz mono signed 16-bit little-endian PCM.

    Audio is kept in memory and never written to disk. The reader requests only
    the expected byte count, and both the requested size and returned size are
    checked against ``max_audio_bytes`` before returning.
    """
    if not source or "REPLACE_ON_AGX" in source:
        raise AudioCaptureError("pulse_source is empty or still an AGX placeholder")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if max_audio_bytes <= 0:
        raise ValueError("max_audio_bytes must be positive")

    expected_frames = round(seconds * SAMPLE_RATE)
    expected_bytes = expected_frames * CHANNELS * SAMPLE_WIDTH_BYTES
    if expected_bytes <= 0:
        raise ValueError("seconds is too small to capture one PCM sample")
    if expected_bytes > max_audio_bytes:
        raise AudioSizeLimitError(
            f"requested PCM size {expected_bytes} exceeds limit {max_audio_bytes} bytes"
        )

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
        )
    except FileNotFoundError as exc:
        raise AudioCaptureError("parec was not found; install PulseAudio utilities") from exc
    except OSError as exc:
        raise AudioCaptureError(f"failed to start parec: {exc}") from exc

    if process.stdout is None or process.stderr is None:
        _stop_process(process, cleanup_timeout_seconds)
        raise AudioCaptureError("parec pipes were not created")

    audio_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    audio_reader = threading.Thread(
        target=_read_at_most,
        args=(process.stdout, expected_bytes, audio_parts),
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=_read_at_most,
        args=(process.stderr, MAX_STDERR_BYTES + 1, stderr_parts),
        daemon=True,
    )
    audio_reader.start()
    stderr_reader.start()
    audio_reader.join(timeout=seconds + startup_grace_seconds)

    if audio_reader.is_alive():
        _stop_process(process, cleanup_timeout_seconds)
        audio_reader.join(timeout=cleanup_timeout_seconds)
        stderr_reader.join(timeout=cleanup_timeout_seconds)
        detail = _stderr_text(stderr_parts[0]) if stderr_parts else ""
        suffix = f": {detail}" if detail else ""
        raise AudioCaptureTimeout(f"parec capture timed out after {seconds:g} seconds{suffix}")

    _stop_process(process, cleanup_timeout_seconds)
    stderr_reader.join(timeout=cleanup_timeout_seconds)
    pcm = audio_parts[0] if audio_parts else b""
    stderr = stderr_parts[0] if stderr_parts else b""

    if len(pcm) > max_audio_bytes or len(pcm) > expected_bytes:
        raise AudioSizeLimitError(
            f"captured PCM size {len(pcm)} exceeds allowed {expected_bytes} bytes"
        )
    if len(pcm) != expected_bytes:
        detail = _stderr_text(stderr)
        suffix = f": {detail}" if detail else ""
        raise AudioCaptureError(
            f"parec returned {len(pcm)} of {expected_bytes} expected PCM bytes{suffix}"
        )
    if len(pcm) % SAMPLE_WIDTH_BYTES:
        raise AudioCaptureError("parec returned an incomplete 16-bit PCM sample")
    return PcmCapture(pcm)
