"""CLI entry points for text, fixed capture, and persistent audio dry-runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

from voice_control import __version__
from voice_control.asr_sherpa import AsrError, SenseVoiceTranscriber, resolve_sense_voice_paths
from voice_control.audio import AudioCaptureError, record_pcm
from voice_control.audit import AuditLogger, default_audit_path, iter_audit_records, summarize_audit
from voice_control.config import VoiceControlConfig, load_config
from voice_control.gateway import DryRunGateway
from voice_control.intents import Intent
from voice_control.playback import PlaybackError, play_prompt
from voice_control.parser import parse_utterance, response_language
from voice_control.state_machine import VoiceStateMachine
from voice_control.streaming import EnergyVadSettings, record_utterance


_ENGLISH_PROMPT_FILES = {
    "wake": "ready_en.wav",
    "confirm": "confirm_floor_en.wav",
    "confirmed": "confirmed_dry_run_en.wav",
    "mismatch": "floor_mismatch_en.wav",
    "retry": "retry_en.wav",
    "timeout": "timeout_en.wav",
    "cancel": "cancelled_en.wav",
    "stop": "stopped_dry_run_en.wav",
}


def _prompt_file_error(path: Path) -> str | None:
  """Return why a prompt is unusable, including header-only WAV files."""
  if not path.is_file():
    return f"prompt file not found: {path}"
  try:
    with wave.open(str(path), "rb") as prompt:
      if prompt.getnframes() <= 0:
        return f"prompt WAV has no audio frames: {path}"
      if prompt.getframerate() <= 0 or prompt.getnchannels() <= 0:
        return f"prompt WAV has invalid audio parameters: {path}"
  except (OSError, EOFError, wave.Error) as exc:
    return f"prompt WAV is unreadable: {path}: {exc}"
  return None


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="voice-control", description="Lift robot voice control (dry-run)")
  parser.add_argument("--config", type=Path, help="JSON/flat-YAML config (or VOICE_CONTROL_CONFIG)")
  parser.add_argument("--audit", type=Path, help="Audit JSONL path")

  sub = parser.add_subparsers(dest="command", required=True)

  sub.add_parser("doctor", help="Validate config and print hardware guidance")

  sub.add_parser("devices", help="Show configured USB/PulseAudio device identity")

  once = sub.add_parser("once", help="Process one utterance as pure-text dry-run")
  once.add_argument("--text", required=True, help="Utterance text")

  listen_once = sub.add_parser(
      "listen-once", help="Record fixed-duration audio and run local ASR dry-run")
  listen_once.add_argument("--seconds", type=float, required=True,
                           help="Fixed capture duration in seconds")

  session = sub.add_parser(
      "session", help="Continuously detect utterances and run a gated dry-run session")
  session.add_argument(
      "--max-utterances", type=int,
      help="Stop after this many detected utterances (default: run until Ctrl-C)")

  replay = sub.add_parser("replay", help="Replay audit JSONL")
  replay.add_argument("--audit", type=Path, required=True, help="Audit JSONL file")

  return parser


def _resolve_audit_path(config: VoiceControlConfig, override: Path | None) -> Path:
  if override:
    return override
  if config.audit_dir:
    return Path(config.audit_dir) / "audit.jsonl"
  return default_audit_path()


def cmd_doctor(config: VoiceControlConfig) -> int:
  issues = config.validate()
  model_status = "not_configured"
  if config.model_dir or config.model or config.model_path or config.tokens:
    try:
      paths = resolve_sense_voice_paths(config)
      model_status = f"ok: {paths.model}"
    except AsrError as exc:
      model_status = f"error: {exc}"
      issues.append(str(exc))
  prompt_statuses = {}
  for prompt_kind, prompt_value in (
      ("wake", config.wake_prompt),
      ("confirm", config.confirm_prompt),
      ("confirmed", config.confirmed_prompt),
      ("mismatch", config.mismatch_prompt),
      ("retry", config.retry_prompt),
      ("timeout", config.timeout_prompt),
      ("cancel", config.cancel_prompt),
      ("stop", config.stop_prompt),
  ):
    status = "not_configured"
    if prompt_value:
      prompt = Path(prompt_value)
      error = _prompt_file_error(prompt)
      if error is None:
        status = f"ok: {prompt}"
      else:
        status = f"error: {error}"
        issues.append(f"{prompt_kind} {error}")
    prompt_statuses[prompt_kind] = status
  floor_prompt_dir = Path(config.floor_confirm_prompt_dir)
  if not config.floor_confirm_prompt_dir:
    prompt_statuses["floor_confirm"] = "not_configured"
  elif not floor_prompt_dir.is_dir():
    prompt_statuses["floor_confirm"] = (
        f"error: floor prompt directory not found: {floor_prompt_dir}")
    issues.append(f"floor prompt directory not found: {floor_prompt_dir}")
  else:
    invalid_floor_prompts = [
        (path, error)
        for floor in range(config.floor_min, config.floor_max + 1)
        for path in [_floor_prompt_path(floor_prompt_dir, floor)]
        for error in [_prompt_file_error(path)]
        if error is not None
    ]
    if invalid_floor_prompts:
      preview = "; ".join(error for _, error in invalid_floor_prompts[:3])
      prompt_statuses["floor_confirm"] = (
          f"error: {len(invalid_floor_prompts)} floor prompts invalid; first: {preview}")
      issues.append(prompt_statuses["floor_confirm"])
    else:
      count = config.floor_max - config.floor_min + 1
      prompt_statuses["floor_confirm"] = f"ok: {floor_prompt_dir} ({count} files)"
  english_dir = Path(config.english_prompt_dir)
  for prompt_kind, filename in _ENGLISH_PROMPT_FILES.items():
    status_key = f"{prompt_kind}_en"
    prompt = english_dir / filename
    if not config.english_prompt_dir:
      prompt_statuses[status_key] = "not_configured"
    elif (error := _prompt_file_error(prompt)) is None:
      prompt_statuses[status_key] = f"ok: {prompt}"
    else:
      prompt_statuses[status_key] = f"error: {error}"
      issues.append(prompt_statuses[status_key])
  english_floor_dir = Path(config.english_floor_confirm_prompt_dir)
  if not config.english_floor_confirm_prompt_dir:
    prompt_statuses["floor_confirm_en"] = "not_configured"
  elif not english_floor_dir.is_dir():
    prompt_statuses["floor_confirm_en"] = (
        f"error: floor prompt directory not found: {english_floor_dir}")
    issues.append(prompt_statuses["floor_confirm_en"])
  else:
    invalid = [
        (path, error)
        for floor in range(config.floor_min, config.floor_max + 1)
        for path in [_floor_prompt_path(english_floor_dir, floor)]
        for error in [_prompt_file_error(path)]
        if error is not None
    ]
    if invalid:
      prompt_statuses["floor_confirm_en"] = (
          f"error: {len(invalid)} English floor prompts invalid; first: {invalid[0][1]}")
      issues.append(prompt_statuses["floor_confirm_en"])
    else:
      count = config.floor_max - config.floor_min + 1
      prompt_statuses["floor_confirm_en"] = (
          f"ok: {english_floor_dir} ({count} files)")
  payload = {
    "version": __version__,
    "dry_run": config.dry_run,
    "config_ok": not issues,
    "issues": issues,
    "model_status": model_status,
    "wake_prompt_status": prompt_statuses["wake"],
    "prompt_statuses": prompt_statuses,
    "hardware_guidance": {
      "usb": f"Look for USB device VID={config.normalized_usb_vid()} PID={config.normalized_usb_pid()}",
      "pulse_source": f"PipeWire/PulseAudio source name: {config.pulse_source}",
      "pulse_sink": f"PipeWire/PulseAudio sink name: {config.pulse_sink}",
      "note": "listen-once opens only the stable Pulse source; ALSA card numbers are not portable.",
    },
  }
  print(json.dumps(payload, ensure_ascii=False, indent=2))
  return 1 if issues else 0


def cmd_devices(config: VoiceControlConfig) -> int:
  payload = {
    "usb_vid": config.normalized_usb_vid(),
    "usb_pid": config.normalized_usb_pid(),
    "pulse_source": config.pulse_source,
    "pulse_sink": config.pulse_sink,
    "discovery": [
      "lsusb | grep -i 0c76",
      f"pactl list sources short | grep {config.pulse_source}",
      f"pactl list sinks short | grep {config.pulse_sink}",
    ],
  }
  print(json.dumps(payload, ensure_ascii=False, indent=2))
  return 0


def _outcome_payload(outcome, gateway: DryRunGateway, audit_path: Path, **extra) -> dict:
  payload = {
    "intent": outcome.parse.intent.value,
    "floor": outcome.parse.floor,
    "state": outcome.state.value,
    "safety_decision": outcome.safety_decision,
    "command_id": outcome.command_id,
    "generation": outcome.generation,
    "gateway": gateway.to_dict(),
    "audit_path": str(audit_path),
  }
  payload.update(extra)
  return payload


def _print_json_line(payload: dict) -> None:
  print(json.dumps(payload, ensure_ascii=False), flush=True)


def _has_speech_content(text: str) -> bool:
  """Treat punctuation-only SenseVoice output as no recognized speech."""
  return any(character.isalnum() for character in text)


_FILLER_CHARACTERS = frozenset("啊嗯哦噢呃诶哎唉欸哈额")
_PROMPT_ECHO_FRAGMENTS = {
    "wake": frozenset({"楼", "几楼"}),
}
_PROMPT_ECHO_WINDOW_SECONDS = 4.0


def _speech_content(text: str) -> str:
  return "".join(character.lower() for character in text if character.isalnum())


def _unknown_ignore_reason(
    text: str,
    *,
    reject_reason: str | None,
    duration_seconds: float,
    config: VoiceControlConfig,
    prompt_echo_kind: str | None = None,
) -> str | None:
  """Classify harmless UNKNOWN audio after command parsing has failed.

  Valid commands are handled before this helper.  In particular, short STOP,
  CANCEL, floor, and confirmation commands can never be discarded here.
  """
  content = _speech_content(text)
  if not content:
    return "no_speech"
  if all(character in _FILLER_CHARACTERS for character in content):
    return "filler"
  if content in _PROMPT_ECHO_FRAGMENTS.get(prompt_echo_kind, ()):
    return "prompt_echo"
  if (reject_reason == "unrecognized"
          and duration_seconds <= config.short_unknown_max_seconds
          and len(content) <= config.short_unknown_max_chars):
    return "short_unknown"
  return None


def _floor_prompt_path(directory: Path, floor: int) -> Path:
  suffix = f"m{abs(floor)}" if floor < 0 else str(floor)
  return directory / f"floor_{suffix}.wav"


def _localized_prompt(config: VoiceControlConfig, language: str, kind: str) -> str:
  if language == "en" and config.english_prompt_dir:
    return str(Path(config.english_prompt_dir) / _ENGLISH_PROMPT_FILES[kind])
  return {
      "wake": config.wake_prompt,
      "confirm": config.confirm_prompt,
      "confirmed": config.confirmed_prompt,
      "mismatch": config.mismatch_prompt,
      "retry": config.retry_prompt,
      "timeout": config.timeout_prompt,
      "cancel": config.cancel_prompt,
      "stop": config.stop_prompt,
  }[kind]


def _confirmation_prompt(
    config: VoiceControlConfig, floor: int | None, language: str = "zh",
) -> str:
  directory = (config.english_floor_confirm_prompt_dir
               if language == "en" else config.floor_confirm_prompt_dir)
  if directory and floor is not None:
    return str(_floor_prompt_path(Path(directory), floor))
  return _localized_prompt(config, language, "confirm")


def _play_configured_prompt(
    config: VoiceControlConfig,
    audit: AuditLogger,
    prompt_kind: str,
    prompt_path: str,
) -> tuple[bool, str | None]:
  if not prompt_path:
    return False, None
  try:
    play_prompt(
        config.pulse_sink,
        prompt_path,
        volume_percent=config.prompt_volume_percent,
        timeout_seconds=config.playback_timeout_seconds,
    )
    audit.log(
        "prompt_playback",
        prompt_kind=prompt_kind,
        success=True,
        prompt=prompt_path,
        volume_percent=config.prompt_volume_percent,
    )
    return True, None
  except PlaybackError as exc:
    error = str(exc)
    audit.log(
        "prompt_playback",
        prompt_kind=prompt_kind,
        success=False,
        prompt=prompt_path,
        error=error,
    )
    return False, error


def cmd_once(config: VoiceControlConfig, text: str, audit_path: Path) -> int:
  if not config.dry_run:
    print("Only dry_run=true is supported.", file=sys.stderr)
    return 2

  gateway = DryRunGateway()
  audit = AuditLogger(audit_path)
  machine = VoiceStateMachine(config=config, gateway=gateway, audit=audit)
  outcome = machine.process_text(text)

  result = _outcome_payload(outcome, gateway, audit_path)
  print(json.dumps(result, ensure_ascii=False, indent=2))
  return 0


def cmd_listen_once(config: VoiceControlConfig, seconds: float, audit_path: Path) -> int:
  """Capture -> SenseVoice -> existing state machine -> DryRunGateway only."""
  if not config.dry_run:
    print("listen-once is fail-closed unless dry_run=true.", file=sys.stderr)
    return 2
  issues = config.validate()
  if issues:
    print("Configuration is invalid: " + "; ".join(issues), file=sys.stderr)
    return 2

  gateway = DryRunGateway()
  audit = AuditLogger(audit_path)
  try:
    # Fail before recording if dependencies, model files, or checksums are bad.
    transcriber = SenseVoiceTranscriber(config)
    capture = record_pcm(
        config.pulse_source,
        seconds,
        max_audio_bytes=config.max_audio_bytes,
    )
    audit.log(
        "audio_capture",
        duration_seconds=round(capture.duration_seconds, 3),
        byte_count=len(capture.data),
        sample_rate=capture.sample_rate,
        channels=capture.channels,
        raw_audio_saved=False,
    )
    text = transcriber.transcribe(capture)
    audit.log("asr_transcription", text=text, engine="sherpa-onnx-sensevoice")
  except (AudioCaptureError, AsrError, ValueError) as exc:
    audit.log("listen_error", error_type=type(exc).__name__, message=str(exc))
    print(f"listen-once failed: {exc}", file=sys.stderr)
    return 1

  machine = VoiceStateMachine(config=config, gateway=gateway, audit=audit)
  outcome = machine.process_text(text)
  prompt_played, playback_error = (False, None)
  language = response_language(text) if outcome.parse.intent is Intent.WAKE else "zh"
  if outcome.parse.intent is Intent.WAKE:
    prompt_played, playback_error = _play_configured_prompt(
        config, audit, "wake", _localized_prompt(config, language, "wake"))
  result = _outcome_payload(
      outcome,
      gateway,
      audit_path,
      transcript=text,
      audio_seconds=round(capture.duration_seconds, 3),
      raw_audio_saved=False,
      response_language=language,
      prompt_played=prompt_played,
      playback_error=playback_error,
  )
  print(json.dumps(result, ensure_ascii=False, indent=2))
  return 1 if playback_error else 0


def _energy_vad_settings(config: VoiceControlConfig) -> EnergyVadSettings:
  return EnergyVadSettings(
      rms_threshold=config.vad_rms_threshold,
      chunk_ms=config.vad_chunk_ms,
      start_ms=config.vad_start_ms,
      silence_ms=config.vad_silence_ms,
      preroll_ms=config.vad_preroll_ms,
      min_speech_ms=config.vad_min_speech_ms,
      max_utterance_seconds=config.vad_max_utterance_seconds,
  )


def cmd_session(
    config: VoiceControlConfig,
    audit_path: Path,
    max_utterances: int | None,
) -> int:
  """Run a persistent half-duplex, wake-gated, dry-run voice session."""
  if not config.dry_run:
    print("session is fail-closed unless dry_run=true.", file=sys.stderr)
    return 2
  issues = config.validate()
  if max_utterances is not None and max_utterances <= 0:
    issues.append("max_utterances must be positive")
  if issues:
    print("Configuration is invalid: " + "; ".join(issues), file=sys.stderr)
    return 2

  audit = AuditLogger(audit_path)
  gateway = DryRunGateway()
  settings = _energy_vad_settings(config)
  _print_json_line({"event": "loading_model", "dry_run": True})
  try:
    transcriber = SenseVoiceTranscriber(config)
  except (AsrError, ValueError) as exc:
    audit.log("session_error", error_type=type(exc).__name__, message=str(exc))
    print(f"session failed: {exc}", file=sys.stderr)
    return 1

  machine = VoiceStateMachine(config=config, gateway=gateway, audit=audit)
  awake = False
  session_language = "zh"
  utterance_count = 0
  prompt_echo_kind = None
  prompt_echo_deadline = 0.0
  _print_json_line({
      "event": "session_started",
      "dry_run": True,
      "vad_rms_threshold": settings.rms_threshold,
      "raw_audio_saved": False,
  })
  audit.log(
      "session_started",
      dry_run=True,
      vad_rms_threshold=settings.rms_threshold,
      raw_audio_saved=False,
  )

  try:
    while max_utterances is None or utterance_count < max_utterances:
      phase = "command" if awake else "wake"
      _print_json_line({"event": "waiting_for_voice", "phase": phase})

      def on_voice_start(rms: int) -> None:
        audit.log("voice_detected", phase=phase, rms=rms)
        _print_json_line({"event": "voice_detected", "phase": phase, "rms": rms})

      detected = record_utterance(
          config.pulse_source,
          settings,
          max_audio_bytes=config.max_audio_bytes,
          wait_timeout_seconds=(
              config.session_awake_timeout_seconds if awake else None
          ),
          on_voice_start=on_voice_start,
      )
      if detected is None:
        audit.log("session_awake_timeout", timeout_seconds=config.session_awake_timeout_seconds)
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, "timeout",
            _localized_prompt(config, session_language, "timeout"))
        if prompt_played and config.post_playback_guard_seconds:
          time.sleep(config.post_playback_guard_seconds)
        _print_json_line({
            "event": "awake_timeout",
            "timeout_seconds": config.session_awake_timeout_seconds,
            "prompt_played": prompt_played,
            "playback_error": playback_error,
        })
        if playback_error:
          return 1
        awake = False
        session_language = "zh"
        machine = VoiceStateMachine(config=config, gateway=gateway, audit=audit)
        continue

      capture = detected.capture
      audit.log(
          "audio_capture",
          duration_seconds=round(capture.duration_seconds, 3),
          byte_count=len(capture.data),
          sample_rate=capture.sample_rate,
          channels=capture.channels,
          peak_rms=detected.peak_rms,
          end_reason=detected.end_reason,
          raw_audio_saved=False,
      )
      try:
        text = transcriber.transcribe(capture)
      except AsrError as exc:
        audit.log("asr_error", message=str(exc), raw_audio_saved=False)
        _print_json_line({
            "event": "recognition_error",
            "message": str(exc),
            "raw_audio_saved": False,
        })
        continue
      audit.log("asr_transcription", text=text, engine="sherpa-onnx-sensevoice")

      parsed = parse_utterance(
          text, floor_min=config.floor_min, floor_max=config.floor_max)

      if not _has_speech_content(text):
        audit.log(
            "no_speech_ignored",
            transcript=text,
            peak_rms=detected.peak_rms,
            retry_prompt_requested=False,
            prompt_played=False,
            raw_audio_saved=False,
        )
        _print_json_line({
            "event": "no_speech_ignored",
            "phase": phase,
            "transcript": text,
            "peak_rms": detected.peak_rms,
            "audio_seconds": round(capture.duration_seconds, 3),
            "retry_prompt_requested": False,
            "prompt_played": False,
            "playback_error": None,
            "raw_audio_saved": False,
        })
        continue

      ignore_reason = None
      if parsed.intent is Intent.UNKNOWN:
        ignore_reason = _unknown_ignore_reason(
            text,
            reject_reason=parsed.reject_reason,
            duration_seconds=capture.duration_seconds,
            config=config,
            prompt_echo_kind=(
                prompt_echo_kind
                if time.monotonic() <= prompt_echo_deadline
                else None
            ),
        )
      if ignore_reason is not None:
        audit.log(
            "non_command_ignored",
            reason=ignore_reason,
            transcript=text,
            reject_reason=parsed.reject_reason,
            duration_seconds=round(capture.duration_seconds, 3),
            peak_rms=detected.peak_rms,
            raw_audio_saved=False,
        )
        _print_json_line({
            "event": "non_command_ignored",
            "phase": phase,
            "ignore_reason": ignore_reason,
            "transcript": text,
            "reject_reason": parsed.reject_reason,
            "audio_seconds": round(capture.duration_seconds, 3),
            "peak_rms": detected.peak_rms,
            "prompt_played": False,
            "playback_error": None,
            "raw_audio_saved": False,
        })
        continue

      utterance_count += 1
      if not awake and parsed.intent is not Intent.WAKE:
        audit.log(
            "safety_decision",
            decision="reject_not_awake",
            source_text=text,
            intent=parsed.intent,
            floor=parsed.floor,
        )
        _print_json_line({
            "event": "result",
            "phase": phase,
            "intent": parsed.intent.value,
            "floor": parsed.floor,
            "state": "DORMANT",
            "safety_decision": "reject_not_awake",
            "command_id": None,
            "generation": machine.generation,
            "gateway": gateway.to_dict(),
            "audit_path": str(audit_path),
            "transcript": text,
            "audio_seconds": round(capture.duration_seconds, 3),
            "peak_rms": detected.peak_rms,
            "end_reason": detected.end_reason,
            "raw_audio_saved": False,
            "prompt_played": False,
            "playback_error": None,
        })
        continue

      outcome = machine.process_text(text)
      prompt_played, playback_error = (False, None)
      prompt_kind = None
      if outcome.parse.intent is Intent.WAKE:
        session_language = response_language(text)
        prompt_kind = "wake"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "wake"))
        awake = playback_error is None
      elif outcome.safety_decision in {
          "await_confirmation", "repeat_pending_floor", "pending_replaced",
      }:
        prompt_kind = "confirm_floor"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _confirmation_prompt(config, outcome.parse.floor, session_language),
        )
      elif outcome.safety_decision == "submitted":
        prompt_kind = "confirmed"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "confirmed"))
      elif outcome.safety_decision == "reject_confirm_floor_mismatch":
        prompt_kind = "mismatch"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "mismatch"))
      elif outcome.parse.intent is Intent.CANCEL:
        prompt_kind = "cancel"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "cancel"))
      elif outcome.parse.intent is Intent.STOP:
        prompt_kind = "stop"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "stop"))
      elif outcome.parse.intent is Intent.UNKNOWN:
        prompt_kind = "retry"
        prompt_played, playback_error = _play_configured_prompt(
            config, audit, prompt_kind,
            _localized_prompt(config, session_language, "retry"))
      if prompt_played and config.post_playback_guard_seconds:
        audit.log(
            "post_playback_guard",
            prompt_kind=prompt_kind,
            duration_seconds=config.post_playback_guard_seconds,
        )
        time.sleep(config.post_playback_guard_seconds)
      if prompt_played:
        prompt_echo_kind = prompt_kind
        prompt_echo_deadline = time.monotonic() + _PROMPT_ECHO_WINDOW_SECONDS
      result = _outcome_payload(
          outcome,
          gateway,
          audit_path,
          event="result",
          phase=phase,
          transcript=text,
          audio_seconds=round(capture.duration_seconds, 3),
          peak_rms=detected.peak_rms,
          end_reason=detected.end_reason,
          raw_audio_saved=False,
          response_language=session_language,
          prompt_played=prompt_played,
          prompt_kind=prompt_kind,
          playback_error=playback_error,
      )
      _print_json_line(result)
      if playback_error:
        return 1
      if outcome.parse.intent is Intent.STOP:
        _print_json_line({"event": "session_stopped", "reason": "spoken_stop"})
        return 0
      if outcome.safety_decision in {
          "submitted", "cancelled_pending", "cancel_no_pending", "status_reported",
          "reject_confirm_expired",
      }:
        awake = False
        session_language = "zh"
        machine = VoiceStateMachine(config=config, gateway=gateway, audit=audit)
  except KeyboardInterrupt:
    audit.log("session_stopped", reason="keyboard_interrupt")
    _print_json_line({"event": "session_stopped", "reason": "keyboard_interrupt"})
    return 0
  except (AudioCaptureError, ValueError) as exc:
    audit.log("session_error", error_type=type(exc).__name__, message=str(exc))
    print(f"session failed: {exc}", file=sys.stderr)
    return 1

  audit.log("session_stopped", reason="max_utterances", count=utterance_count)
  _print_json_line({
      "event": "session_stopped",
      "reason": "max_utterances",
      "count": utterance_count,
  })
  return 0


def cmd_replay(audit_path: Path) -> int:
  if not audit_path.is_file():
    print(f"Audit file not found: {audit_path}", file=sys.stderr)
    return 1
  for record in iter_audit_records(audit_path):
    print(json.dumps(record, ensure_ascii=False))
  summary = summarize_audit(audit_path)
  print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), file=sys.stderr)
  return 0


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  config = load_config(args.config)
  audit_path = _resolve_audit_path(config, args.audit)

  if args.command == "doctor":
    return cmd_doctor(config)
  if args.command == "devices":
    return cmd_devices(config)
  if args.command == "once":
    return cmd_once(config, args.text, audit_path)
  if args.command == "listen-once":
    return cmd_listen_once(config, args.seconds, audit_path)
  if args.command == "session":
    return cmd_session(config, audit_path, args.max_utterances)
  if args.command == "replay":
    return cmd_replay(args.audit)
  parser.error(f"unknown command: {args.command}")
  return 2


if __name__ == "__main__":
  raise SystemExit(main())
