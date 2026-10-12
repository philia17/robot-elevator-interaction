import io
import json
import subprocess
import sys
import threading

import pytest

from voice_control import cli
from voice_control.asr_sherpa import AsrError, SenseVoiceTranscriber, resolve_sense_voice_paths
from voice_control.audio import (
    AudioCaptureTimeout,
    AudioSizeLimitError,
    BYTES_PER_SECOND,
    PcmCapture,
    record_pcm,
)
from voice_control.config import VoiceControlConfig
from voice_control.playback import play_prompt


class FakeProcess:
    def __init__(self, stdout=b"", stderr=b""):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("parec", timeout)
        return self.returncode


def test_record_pcm_uses_argument_list_and_captures_exact_pcm(monkeypatch):
    expected = BYTES_PER_SECOND
    observed = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeProcess(stdout=b"x" * expected)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    capture = record_pcm("stable-source", 1)
    assert len(capture.data) == expected

    assert observed["command"] == [
        "parec", "--device=stable-source", "--format=s16le",
        "--rate=16000", "--channels=1", "--raw",
    ]
    assert "shell" not in observed["kwargs"]


def test_play_prompt_uses_bounded_paplay_argument_list(tmp_path, monkeypatch):
    prompt = tmp_path / "ready.wav"
    prompt.write_bytes(b"RIFF-placeholder")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    play_prompt("stable-sink", prompt, volume_percent=45, timeout_seconds=4)
    assert observed["command"] == [
        "paplay", "--device=stable-sink", "--volume=29491", str(prompt),
    ]
    assert observed["kwargs"]["timeout"] == 4
    assert "shell" not in observed["kwargs"]


def test_record_pcm_rejects_request_above_size_limit(monkeypatch):
    def must_not_start(*args, **kwargs):
        raise AssertionError("parec must not start")

    monkeypatch.setattr(subprocess, "Popen", must_not_start)
    with pytest.raises(AudioSizeLimitError, match="requested PCM size"):
        record_pcm("stable-source", 2, max_audio_bytes=BYTES_PER_SECOND)


class BlockingReader:
    def __init__(self):
        self.released = threading.Event()

    def read(self, size=-1):
        self.released.wait(timeout=1)
        return b""


class BlockingProcess(FakeProcess):
    def __init__(self):
        super().__init__(stderr=b"device stalled")
        self.stdout = BlockingReader()

    def terminate(self):
        self.stdout.released.set()
        super().terminate()


def test_record_pcm_timeout_cleans_up_process(monkeypatch):
    process = BlockingProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(AudioCaptureTimeout, match="timed out"):
        record_pcm(
            "stable-source", 0.001, startup_grace_seconds=0.001,
            cleanup_timeout_seconds=0.1,
        )
    assert process.returncode == -15


def test_asr_reports_missing_model_before_loading_dependencies(tmp_path):
    config = VoiceControlConfig(
        model=str(tmp_path / "missing-model.int8.onnx"),
        tokens=str(tmp_path / "missing-tokens.txt"),
    )
    with pytest.raises(AsrError, match="model file not found"):
        SenseVoiceTranscriber(config)


def test_model_dir_has_priority_and_uses_fixed_sensevoice_filenames(tmp_path):
    model_dir = tmp_path / "sensevoice"
    model_dir.mkdir()
    expected_model = model_dir / "model.int8.onnx"
    expected_tokens = model_dir / "tokens.txt"
    expected_model.write_bytes(b"placeholder")
    expected_tokens.write_text("placeholder", encoding="utf-8")
    paths = resolve_sense_voice_paths(VoiceControlConfig(
        model_dir=str(model_dir),
        model=str(tmp_path / "ignored.onnx"),
        tokens=str(tmp_path / "ignored.txt"),
    ))
    assert paths.model == expected_model
    assert paths.tokens == expected_tokens


def test_model_sha256_mismatch_fails_closed(tmp_path):
    model = tmp_path / "model.int8.onnx"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"model")
    tokens.write_text("tokens", encoding="utf-8")
    with pytest.raises(AsrError, match="SHA-256 mismatch"):
        resolve_sense_voice_paths(VoiceControlConfig(
            model=str(model), tokens=str(tokens), model_sha256="0" * 64,
        ))


def test_asr_reports_missing_optional_dependency(tmp_path, monkeypatch):
    model = tmp_path / "model.int8.onnx"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"placeholder")
    tokens.write_text("placeholder", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(AsrError, match=r"voice-control\[asr\]"):
        SenseVoiceTranscriber(VoiceControlConfig(model=str(model), tokens=str(tokens)))


def test_listen_once_transcript_enters_existing_state_machine(tmp_path, monkeypatch, capsys):
    capture = PcmCapture(b"\x00\x00" * 16_000)

    class FakeTranscriber:
        def __init__(self, config):
            self.config = config

        def transcribe(self, received):
            assert received is capture
            return "帮我按三楼"

    monkeypatch.setattr(cli, "record_pcm", lambda *args, **kwargs: capture)
    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    audit = tmp_path / "audit.jsonl"
    assert cli.main(["--audit", str(audit), "listen-once", "--seconds", "1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript"] == "帮我按三楼"
    assert payload["intent"] == "PRESS_FLOOR"
    assert payload["state"] == "AWAITING_CONFIRMATION"
    assert payload["gateway"]["submission_count"] == 0
    assert payload["raw_audio_saved"] is False

    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert any(record.get("event") == "audio_capture" for record in records)
    assert any(record.get("event") == "asr_transcription" for record in records)
    assert all("audio" not in record or record.get("raw_audio_saved") is False
               for record in records)
