import json
import wave

from voice_control.audit import summarize_audit
from voice_control.cli import cmd_doctor, main
from voice_control.config import VoiceControlConfig, load_config


def test_cli_once_text_and_replay(tmp_path, capsys):
    audit = tmp_path / "events.jsonl"
    assert main(["--audit", str(audit), "once", "--text", "帮我按三楼"]) == 0
    once_payload = json.loads(capsys.readouterr().out)
    assert once_payload["intent"] == "PRESS_FLOOR"
    assert once_payload["state"] == "AWAITING_CONFIRMATION"
    assert once_payload["gateway"]["submission_count"] == 0

    assert main(["replay", "--audit", str(audit)]) == 0
    assert summarize_audit(audit)["event_count"] > 0


def test_doctor_rejects_header_only_prompt_wav(tmp_path, capsys):
    empty_prompt = tmp_path / "empty.wav"
    with wave.open(str(empty_prompt), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)

    assert cmd_doctor(VoiceControlConfig(wake_prompt=str(empty_prompt))) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_ok"] is False
    assert "no audio frames" in payload["prompt_statuses"]["wake"]


def test_yaml_profiles_are_portable_and_agx_fails_closed():
    orin = load_config("config/orin_nano.yaml")
    assert orin.validate() == []
    assert orin.normalized_usb_vid() == "0c76"
    assert "usb-0c76" in orin.pulse_source
    assert "card" not in orin.pulse_source.lower()
    assert orin.prompt_volume_percent == 50
    assert orin.playback_timeout_seconds == 10
    assert orin.post_playback_guard_seconds == 0.5
    assert orin.confirm_prompt.endswith("confirm_floor_zh.wav")
    assert orin.floor_confirm_prompt_dir.endswith("prompts/floors")
    assert orin.confirmed_prompt.endswith("confirmed_dry_run_zh.wav")
    assert orin.mismatch_prompt.endswith("floor_mismatch_zh.wav")
    assert orin.english_prompt_dir.endswith("prompts/en")
    assert orin.english_floor_confirm_prompt_dir.endswith("prompts/en/floors")
    assert orin.vad_rms_threshold == 300
    assert orin.vad_silence_ms == 600
    assert orin.short_unknown_max_seconds == 1.0
    assert orin.short_unknown_max_chars == 2
    assert orin.no_speech_retry_peak_rms == 10000
    assert orin.session_awake_timeout_seconds == 30

    agx = load_config("config/agx.example.yaml")
    issues = agx.validate()
    assert any("pulse_source" in issue for issue in issues)
    assert any("pulse_sink" in issue for issue in issues)
    assert any("model_dir" in issue for issue in issues)
    assert any("wake_prompt" in issue for issue in issues)
    assert any("confirm_prompt" in issue for issue in issues)
    assert any("mismatch_prompt" in issue for issue in issues)
    assert any("floor_confirm_prompt_dir" in issue for issue in issues)
    assert any("english_prompt_dir" in issue for issue in issues)
    assert any("english_floor_confirm_prompt_dir" in issue for issue in issues)
    assert not any("vad" in issue for issue in issues)
