import json

from voice_control import cli
from voice_control.audio import PcmCapture
from voice_control.config import VoiceControlConfig
from voice_control.streaming import (
    DetectedUtterance,
    EnergyUtteranceDetector,
    EnergyVadSettings,
    pcm_rms,
)


def pcm_chunk(detector: EnergyUtteranceDetector, amplitude: int) -> bytes:
    return int(amplitude).to_bytes(2, "little", signed=True) * detector.chunk_frames


def test_energy_detector_emits_after_voice_then_silence():
    settings = EnergyVadSettings(
        rms_threshold=100,
        chunk_ms=20,
        start_ms=40,
        silence_ms=60,
        preroll_ms=40,
        min_speech_ms=40,
        max_utterance_seconds=2,
    )
    detector = EnergyUtteranceDetector(settings)
    quiet = pcm_chunk(detector, 10)
    voice = pcm_chunk(detector, 1000)

    assert pcm_rms(quiet) == 10
    assert pcm_rms(voice) == 1000
    assert not detector.feed(quiet).voice_started
    assert not detector.feed(voice).voice_started
    assert detector.feed(voice).voice_started
    detector.feed(voice)
    detector.feed(voice)
    detector.feed(quiet)
    detector.feed(quiet)
    event = detector.feed(quiet)

    assert event.utterance is not None
    assert event.utterance.end_reason == "silence"
    assert event.utterance.peak_rms == 1000
    assert event.utterance.capture.duration_seconds == 0.14


def test_energy_detector_discards_too_short_noise():
    settings = EnergyVadSettings(
        rms_threshold=100,
        chunk_ms=20,
        start_ms=20,
        silence_ms=40,
        preroll_ms=20,
        min_speech_ms=60,
        max_utterance_seconds=2,
    )
    detector = EnergyUtteranceDetector(settings)
    voice = pcm_chunk(detector, 500)
    quiet = pcm_chunk(detector, 0)

    assert detector.feed(voice).voice_started
    assert detector.feed(quiet).utterance is None
    assert detector.feed(quiet).utterance is None
    assert detector.started is False


def test_session_rejects_floor_before_wake_and_loads_model_once(
    tmp_path, monkeypatch, capsys,
):
    captures = [
        DetectedUtterance(PcmCapture(b"\x01\x00" * 3200), 900, "silence"),
        DetectedUtterance(PcmCapture(b"\x02\x00" * 3200), 1000, "silence"),
    ]
    transcripts = iter(["去三楼", "小易小艺"])
    init_count = 0
    playback_count = 0
    prompt = tmp_path / "ready.wav"
    prompt.write_bytes(b"RIFF-placeholder")

    class FakeTranscriber:
        def __init__(self, config):
            nonlocal init_count
            init_count += 1

        def transcribe(self, capture):
            return next(transcripts)

    def fake_record(*args, on_voice_start=None, **kwargs):
        detected = captures.pop(0)
        if on_voice_start:
            on_voice_start(detected.peak_rms)
        return detected

    def fake_play(*args, **kwargs):
        nonlocal playback_count
        playback_count += 1

    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "record_utterance", fake_record)
    monkeypatch.setattr(cli, "play_prompt", fake_play)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli, "load_config", lambda path=None: VoiceControlConfig(wake_prompt=str(prompt))
    )
    audit = tmp_path / "audit.jsonl"

    assert cli.main([
        "--audit", str(audit), "session", "--max-utterances", "2",
    ]) == 0

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = [event for event in events if event.get("event") == "result"]
    assert results[0]["transcript"] == "去三楼"
    assert results[0]["state"] == "DORMANT"
    assert results[0]["safety_decision"] == "reject_not_awake"
    assert results[0]["gateway"]["submission_count"] == 0
    assert results[1]["intent"] == "WAKE"
    assert results[1]["state"] == "LISTENING"
    assert results[1]["prompt_played"] is True
    assert init_count == 1
    assert playback_count == 1


def test_session_keeps_state_for_floor_confirmation(tmp_path, monkeypatch, capsys):
    transcripts = iter(["小易小易", "去三楼", "。", "确认三楼"])
    prompt = tmp_path / "ready.wav"
    prompt.write_bytes(b"RIFF-placeholder")
    floor_prompts = tmp_path / "floors"
    floor_prompts.mkdir()
    floor_prompt = floor_prompts / "floor_3.wav"
    floor_prompt.write_bytes(b"RIFF-placeholder")

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, capture):
            return next(transcripts)

    def fake_record(*args, on_voice_start=None, **kwargs):
        if on_voice_start:
            on_voice_start(800)
        return DetectedUtterance(
            PcmCapture(b"\x01\x00" * 3200), 800, "silence"
        )

    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "record_utterance", fake_record)
    played_prompts = []

    def fake_play(sink, prompt_path, **kwargs):
        played_prompts.append(prompt_path)

    monkeypatch.setattr(cli, "play_prompt", fake_play)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli, "load_config", lambda path=None: VoiceControlConfig(
            wake_prompt=str(prompt),
            confirm_prompt=str(prompt),
            confirmed_prompt=str(prompt),
            floor_confirm_prompt_dir=str(floor_prompts),
        )
    )

    assert cli.main([
        "--audit", str(tmp_path / "audit.jsonl"),
        "session", "--max-utterances", "3",
    ]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = [event for event in events if event.get("event") == "result"]
    ignored = [
        event for event in events if event.get("event") == "no_speech_ignored"
    ]
    assert [result["intent"] for result in results] == [
        "WAKE", "PRESS_FLOOR", "CONFIRM",
    ]
    assert results[1]["safety_decision"] == "await_confirmation"
    assert results[2]["safety_decision"] == "submitted"
    assert results[2]["gateway"]["submission_count"] == 1
    assert len(ignored) == 1
    assert ignored[0]["peak_rms"] == 800
    assert ignored[0]["prompt_played"] is False
    assert [result["prompt_kind"] for result in results] == [
        "wake", "confirm_floor", "confirmed",
    ]
    assert played_prompts == [str(prompt), str(floor_prompt), str(prompt)]


def test_unknown_filter_preserves_commands_and_silences_only_harmless_audio():
    config = VoiceControlConfig(
        short_unknown_max_seconds=1.0,
        short_unknown_max_chars=2,
    )
    assert cli._unknown_ignore_reason(
        "啊。", reject_reason="unrecognized", duration_seconds=0.87,
        config=config,
    ) == "filler"
    assert cli._unknown_ignore_reason(
        "嗯嗯", reject_reason="unrecognized", duration_seconds=2.0,
        config=config,
    ) == "filler"
    assert cli._unknown_ignore_reason(
        "答", reject_reason="unrecognized", duration_seconds=0.9,
        config=config,
    ) == "short_unknown"
    assert cli._unknown_ignore_reason(
        "确认", reject_reason="confirmation_missing_floor", duration_seconds=0.8,
        config=config,
    ) is None
    assert cli._unknown_ignore_reason(
        "帮我去那个地方", reject_reason="unrecognized", duration_seconds=1.5,
        config=config,
    ) is None
    assert cli._unknown_ignore_reason(
        "几楼。", reject_reason="unrecognized", duration_seconds=1.89,
        config=config, prompt_echo_kind="wake",
    ) == "prompt_echo"
    assert cli._unknown_ignore_reason(
        "几楼。", reject_reason="unrecognized", duration_seconds=1.89,
        config=config,
    ) is None
    assert cli._unknown_ignore_reason(
        "天气。", reject_reason="unrecognized", duration_seconds=1.2,
        config=config, prompt_echo_kind="wake",
    ) is None


def test_session_silences_loud_filler_without_consuming_utterance(
    tmp_path, monkeypatch, capsys,
):
    transcripts = iter(["小易小易", "啊。", "去三楼", "确认三楼"])
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"RIFF-placeholder")
    floor_prompts = tmp_path / "floors"
    floor_prompts.mkdir()
    (floor_prompts / "floor_3.wav").write_bytes(b"RIFF-placeholder")

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, capture):
            return next(transcripts)

    durations = iter([1.5, 0.87, 1.5, 1.5])

    def fake_record(*args, on_voice_start=None, **kwargs):
        duration = next(durations)
        if on_voice_start:
            on_voice_start(12_049)
        return DetectedUtterance(
            PcmCapture(b"\x01\x00" * round(duration * 16_000)),
            12_049,
            "silence",
        )

    played_prompts = []
    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "record_utterance", fake_record)
    monkeypatch.setattr(
        cli, "play_prompt",
        lambda sink, prompt_path, **kwargs: played_prompts.append(prompt_path),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli, "load_config", lambda path=None: VoiceControlConfig(
            wake_prompt=str(prompt),
            confirmed_prompt=str(prompt),
            retry_prompt=str(prompt),
            floor_confirm_prompt_dir=str(floor_prompts),
        )
    )

    assert cli.main([
        "--audit", str(tmp_path / "audit.jsonl"),
        "session", "--max-utterances", "3",
    ]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    ignored = [event for event in events
               if event.get("event") == "non_command_ignored"]
    results = [event for event in events if event.get("event") == "result"]
    assert len(ignored) == 1
    assert ignored[0]["ignore_reason"] == "filler"
    assert ignored[0]["peak_rms"] == 12_049
    assert ignored[0]["prompt_played"] is False
    assert [result["intent"] for result in results] == [
        "WAKE", "PRESS_FLOOR", "CONFIRM",
    ]
    assert results[-1]["gateway"]["submission_count"] == 1
    assert len(played_prompts) == 3


def test_english_wake_locks_english_prompts_for_the_session(
    tmp_path, monkeypatch, capsys,
):
    transcripts = iter(["嘿sri。", "Go to floor three.", "Yes."])
    english = tmp_path / "en"
    floors = english / "floors"
    floors.mkdir(parents=True)
    wake = english / "ready_en.wav"
    confirmed = english / "confirmed_dry_run_en.wav"
    floor_three = floors / "floor_3.wav"
    for prompt in (wake, confirmed, floor_three):
        prompt.write_bytes(b"RIFF-placeholder")

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, capture):
            return next(transcripts)

    def fake_record(*args, on_voice_start=None, **kwargs):
        if on_voice_start:
            on_voice_start(20_000)
        return DetectedUtterance(PcmCapture(b"\x01\x00" * 24_000), 20_000, "silence")

    played = []
    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "record_utterance", fake_record)
    monkeypatch.setattr(
        cli, "play_prompt",
        lambda sink, prompt_path, **kwargs: played.append(prompt_path),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli, "load_config", lambda path=None: VoiceControlConfig(
            english_prompt_dir=str(english),
            english_floor_confirm_prompt_dir=str(floors),
        )
    )

    assert cli.main([
        "--audit", str(tmp_path / "audit.jsonl"),
        "session", "--max-utterances", "3",
    ]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = [event for event in events if event.get("event") == "result"]
    assert [event["response_language"] for event in results] == ["en", "en", "en"]
    assert results[-1]["gateway"]["submission_count"] == 1
    assert played == [str(wake), str(floor_three), str(confirmed)]


def test_session_speaks_mismatch_then_allows_floor_replacement(
    tmp_path, monkeypatch, capsys,
):
    transcripts = iter([
        "小易小易", "去三楼", "确认四楼", "帮我按四楼", "确认四楼",
    ])
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"RIFF-placeholder")
    mismatch_prompt = tmp_path / "mismatch.wav"
    mismatch_prompt.write_bytes(b"RIFF-placeholder")
    floor_prompts = tmp_path / "floors"
    floor_prompts.mkdir()
    floor_three_prompt = floor_prompts / "floor_3.wav"
    floor_four_prompt = floor_prompts / "floor_4.wav"
    floor_three_prompt.write_bytes(b"RIFF-placeholder")
    floor_four_prompt.write_bytes(b"RIFF-placeholder")

    class FakeTranscriber:
        def __init__(self, config):
            pass

        def transcribe(self, capture):
            return next(transcripts)

    def fake_record(*args, on_voice_start=None, **kwargs):
        if on_voice_start:
            on_voice_start(20_000)
        return DetectedUtterance(
            PcmCapture(b"\x01\x00" * 3200), 20_000, "silence"
        )

    played_prompts = []
    monkeypatch.setattr(cli, "SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "record_utterance", fake_record)
    monkeypatch.setattr(
        cli, "play_prompt",
        lambda sink, prompt_path, **kwargs: played_prompts.append(prompt_path),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli, "load_config", lambda path=None: VoiceControlConfig(
            wake_prompt=str(prompt),
            confirmed_prompt=str(prompt),
            mismatch_prompt=str(mismatch_prompt),
            floor_confirm_prompt_dir=str(floor_prompts),
        )
    )

    assert cli.main([
        "--audit", str(tmp_path / "audit.jsonl"),
        "session", "--max-utterances", "5",
    ]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = [event for event in events if event.get("event") == "result"]
    assert [result["safety_decision"] for result in results] == [
        "wake_ack",
        "await_confirmation",
        "reject_confirm_floor_mismatch",
        "pending_replaced",
        "submitted",
    ]
    first_command_id = results[1]["command_id"]
    assert results[2]["command_id"] == first_command_id
    assert results[3]["command_id"] != first_command_id
    assert results[4]["command_id"] == results[3]["command_id"]
    assert results[4]["gateway"]["submission_count"] == 1
    assert played_prompts == [
        str(prompt),
        str(floor_three_prompt),
        str(mismatch_prompt),
        str(floor_four_prompt),
        str(prompt),
    ]
