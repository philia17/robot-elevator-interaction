"""Lazy-loaded sherpa-onnx SenseVoice transcription adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voice_control.audio import PcmCapture, SAMPLE_RATE
from voice_control.config import VoiceControlConfig


class AsrError(RuntimeError):
    """ASR configuration, dependency, model, or inference failure."""


@dataclass(frozen=True)
class SenseVoicePaths:
    model: Path
    tokens: Path


def resolve_sense_voice_paths(config: VoiceControlConfig) -> SenseVoicePaths:
    model_dir = Path(config.model_dir).expanduser() if config.model_dir else None
    if model_dir:
        model = str(model_dir / "model.int8.onnx")
        tokens = str(model_dir / "tokens.txt")
    else:
        model = config.model or config.model_path
        tokens = config.tokens
    if not model:
        raise AsrError("SenseVoice model is not configured; set model or model_dir")
    if not tokens:
        raise AsrError("SenseVoice tokens are not configured; set tokens or model_dir")

    model_path = Path(model).expanduser()
    tokens_path = Path(tokens).expanduser()
    if not model_path.is_file():
        raise AsrError(f"SenseVoice model file not found: {model_path}")
    if not tokens_path.is_file():
        raise AsrError(f"SenseVoice tokens file not found: {tokens_path}")
    if config.model_sha256:
        digest = hashlib.sha256()
        with model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != config.model_sha256.lower():
            raise AsrError(
                "SenseVoice model SHA-256 mismatch: "
                f"expected {config.model_sha256.lower()}, got {actual}"
            )
    return SenseVoicePaths(model=model_path, tokens=tokens_path)


class SenseVoiceTranscriber:
    """Offline SenseVoice recognizer; heavy imports occur only when instantiated."""

    def __init__(self, config: VoiceControlConfig) -> None:
        paths = resolve_sense_voice_paths(config)
        try:
            import numpy as np
        except ImportError as exc:
            raise AsrError(
                "numpy is required for ASR; install voice-control[asr]"
            ) from exc
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise AsrError(
                "sherpa-onnx is required for ASR; install voice-control[asr]"
            ) from exc

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(paths.model),
                tokens=str(paths.tokens),
                num_threads=config.asr_num_threads,
                language="zh",
                use_itn=True,
                debug=False,
                provider=config.asr_provider,
            )
        except Exception as exc:
            raise AsrError(f"failed to load SenseVoice model: {exc}") from exc
        self._np = np

    def transcribe(self, capture: PcmCapture) -> str:
        if capture.sample_rate != SAMPLE_RATE or capture.channels != 1:
            raise AsrError("ASR requires 16 kHz mono PCM")
        if capture.sample_width_bytes != 2 or len(capture.data) % 2:
            raise AsrError("ASR requires complete signed 16-bit PCM samples")
        try:
            samples = self._np.frombuffer(capture.data, dtype=self._np.int16)
            samples = samples.astype(self._np.float32) / 32768.0
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self._recognizer.decode_stream(stream)
            text = (stream.result.text or "").strip()
        except Exception as exc:
            raise AsrError(f"SenseVoice transcription failed: {exc}") from exc
        if not text:
            raise AsrError("SenseVoice returned empty transcription")
        return text
