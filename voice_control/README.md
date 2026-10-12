# Voice Control Core (`0.4.3`)

Portable local voice-control package for the lift robot. Stage 3 adds a persistent, energy-VAD-gated microphone session (`parec` → sherpa-onnx SenseVoice) to the deterministic Chinese parser, safety state machine, confirmation binding, half-duplex wake prompt, and JSONL audit log. The fixed-duration test path remains available. **It remains dry-run: there is no ROS, network, camera, robot, or elevator actuation.**

For a concise description of the implemented Chinese/English functions,
validated interaction flows, current limitations, and next work, see
the [current boundary](#current-boundary) below.

## Safety notices

- **Voice STOP is not a physical emergency stop.** Utterances such as “停止” only cancel the software voice-control session via `DryRunGateway.abort`. They do not cut motor power or replace a hardware e-stop.
- **Half-duplex playback creates a listening blind spot.** While TTS or confirmation audio plays through the configured sink, the microphone path may miss user speech. Design workflows accordingly (pause playback before listening, or accept missed interrupts).

## Default mode

Dry-run is mandatory (`dry_run: true` in config). `once`, `listen-once`, and `session` use `DryRunGateway`, which records `submit` / `abort` calls without touching elevators or robots.

## Package layout

```
voice_control/
├── src/voice_control/   # library + CLI
└── tests/               # pytest unit tests
```

## Configuration

Config is loaded from `VOICE_CONTROL_CONFIG` (the shipped flat YAML format or JSON) or built-in defaults. `config/orin_nano.yaml` is the development profile. `config/agx.example.yaml` deliberately fails `doctor` until its PulseAudio/PipeWire names are replaced on the AGX.

| Key | Default | Notes |
|-----|---------|-------|
| `usb_vid` | `0c76` | CI1302 USB vendor ID (device-discovery guidance) |
| `usb_pid` | `161f` | CI1302 USB product ID |
| `pulse_source` | stable USB source name | PipeWire/PulseAudio source name |
| `pulse_sink` | stable USB sink name | PipeWire/PulseAudio sink name |
| `*_prompt` | empty | WAV responses for wake, confirm, accepted, retry, timeout, cancel, and stop |
| `floor_confirm_prompt_dir` | empty | Pre-rendered `floor_N.wav` / `floor_mN.wav` responses |
| `english_prompt_dir` | empty | Pre-rendered English wake, result, retry, timeout, cancel, and stop WAVs |
| `english_floor_confirm_prompt_dir` | empty | Pre-rendered English floor confirmation WAVs |
| `prompt_volume_percent` | `50` | Per-prompt PulseAudio volume |
| `post_playback_guard_seconds` | `0.5` | Keep capture closed briefly for acoustic tail decay |
| `dry_run` | `true` | Must remain true in Stage 3 |
| `floor_min` / `floor_max` | `-2` / `40` | Supported floor range |
| `confirmation_ttl_seconds` | `30` | Pending command TTL |
| `model_dir` | empty | Preferred directory; always resolves fixed `model.int8.onnx` and `tokens.txt` |
| `model` / `tokens` | empty | Optional explicit fallback paths when `model_dir` is empty |
| `vad_rms_threshold` | `300` | Energy threshold; tune from live `peak_rms` observations |
| `vad_*_ms` | profile values | Voice start, silence endpoint, preroll, and minimum-speech timing |
| `vad_max_utterance_seconds` | `8` | Hard endpoint for one detected utterance |
| `short_unknown_max_seconds` / `short_unknown_max_chars` | `1.0` / `2` | Silence short UNKNOWN text after valid-command parsing |
| `no_speech_retry_peak_rms` | `10000` | Legacy compatibility field; no longer controls punctuation feedback |
| `session_awake_timeout_seconds` | `30` | Return to dormant if no command or confirmation follows wake |
| `vad` | empty | Reserved path for a future neural VAD model; not used by energy VAD |
| `model_path` | empty | Backward-compatible alias for `model` |
| `asr_num_threads` / `asr_provider` | `2` / `cpu` | sherpa-onnx runtime settings |
| `max_audio_bytes` | `16777216` | Hard in-memory PCM request/return limit |

ALSA card numbers are **never** hard-coded; use stable PulseAudio/PipeWire device names.

## CLI

```bash
# Install editable (optional, for `voice-control` entry point)
pip install -e ".[dev,asr]"

# Or run from this directory without installing
PYTHONPATH=src python -m voice_control --config config/orin_nano.yaml doctor
PYTHONPATH=src python -m voice_control devices
PYTHONPATH=src python -m voice_control once --text "去三楼"
PYTHONPATH=src python -m voice_control --config config/orin_nano.yaml \
  listen-once --seconds 4
PYTHONPATH=src python -m voice_control --config config/orin_nano.yaml \
  session --max-utterances 1
PYTHONPATH=src python -m voice_control replay --audit /path/to/audit.jsonl
```

### Commands

| Command | Purpose |
|---------|---------|
| `doctor` | Validate config, model files/checksum, dry-run status and hardware discovery guidance |
| `devices` | Show expected USB audio identity and PulseAudio/PipeWire names from config |
| `once --text TEXT` | Single-utterance dry-run through parser + state machine; appends audit JSONL |
| `listen-once --seconds N` | Record fixed-duration 16 kHz mono PCM, transcribe locally, then use the same dry-run state machine |
| `session [--max-utterances N]` | Wait continuously for bounded utterances; require wake before commands; reuse the loaded model |
| `replay --audit FILE` | Re-print audit events and summarize outcomes |

Both microphone commands invoke `parec` with an argument list (never a shell), use the configured stable Pulse source, bound capture time and bytes, clean up child processes, and keep PCM only in memory. `session` emits `voice_detected` as soon as the threshold is crossed, then transcribes after configured trailing silence. It closes capture before ASR and prompt playback, waits the short post-playback guard for acoustic tail decay, and then reopens capture. Punctuation-only ASR never reaches the state machine or consumes a valid-utterance test count; low-peak events are silently ignored, while high-peak events receive the retry prompt. The JSONL audit stores capture metadata, RMS, and recognized text, never raw audio.

When testing from a NoMachine desktop terminal, its `PULSE_SERVER` may point to
the NoMachine virtual audio server instead of the Nano's system PulseAudio
server. Select the system server for that command only:

```bash
PULSE_SERVER=unix:/run/user/1000/pulse/native \
  PYTHONPATH=src python -m voice_control --config config/orin_nano.yaml \
  listen-once --seconds 4

# Persistent mode (Ctrl-C to stop); use --max-utterances 1 for first validation.
PULSE_SERVER=unix:/run/user/1000/pulse/native \
  PYTHONPATH=src python -m voice_control --config config/orin_nano.yaml \
  session
```

## Controlled intents

`PRESS_FLOOR`, `CONFIRM`, `CANCEL`, `STOP`, `STATUS`, `WAKE`, `UNKNOWN`

`STOP` is recognized with **highest priority** in the parser. In the state machine, `STOP` from any state increments a generation token, calls idempotent `gateway.abort()`, and enters `STOP_LATCHED`.

In `session`, the dormant phase accepts only an exact repeated wake phrase. A floor command spoken before wake is audited as `reject_not_awake`. Floor commands after wake play a pre-rendered prompt that states the recognized floor and asks for confirmation, bound to `command_id + floor + TTL`. Chinese confirmation repeats the floor, such as `确认三楼`. In an English session, bare `Yes` binds to the current pending floor; without a pending command it is rejected. Other bare words such as `OK`, `Sure`, or `Confirm` remain rejected. The accepted prompt explicitly calls it a test command. Stale, mismatched, or missing confirmations never execute.

The deployed Chinese wake acknowledgement is `我在，请问要去几楼`.

The English wake phrase accepts the observed SenseVoice variants of `Hey Siri`
(`嘿 siri`, `嘿sri`, and `嘿y siri`) after conservative normalization. An
English wake locks the response language to English until submission,
cancellation, stop, or timeout. English floor requests, replacements,
confirmations, cancellation, and stop use the same state machine, command-ID
binding, TTL, and dry-run gateway as Chinese. Supported forms include
`Go to floor three`, `I want to go to the third floor`, `B1`,
`Change it to floor four`, compact transcripts such as `floor3`, and contextual
`Yes`. While a floor is pending, concise corrections such as `No, first floor`
replace the target and request confirmation again; the same phrase is rejected
outside confirmation context. Explicit
`Confirm floor four` remains supported for diagnostics.

While awaiting confirmation, another controlled floor request replaces the pending target and creates a new command ID plus TTL. This includes ordinary requests such as `去四楼`, `我想去四楼`, and `帮我按四楼`, change wording such as `改成四楼`, `换到四楼`, and `还是去四楼吧`, and conversational framing around an explicit action such as `算了，我想去四楼` or `等下，不对，现在我想去四楼`. A controlled correction such as `不是三楼，是四楼` selects the new target, while an unresolved choice such as `三楼还是四楼` is rejected. Repeating the same floor preserves the existing command ID and TTL. A mismatched confirmation such as `确认四楼` while floor 3 is pending never changes the target or submits; it plays `楼层不对，请重新说楼层或取消` and continues waiting.

Punctuation-only ASR, filler-only utterances, and configured short UNKNOWN text are silently ignored. Longer unrelated or unrecognized speech plays the retry prompt `抱歉，我没听懂，请再说一次` rather than claiming that the audio was not heard.

The Mandarin wake name may be transcribed as `小艺`, `小易`, or `小一`. Exactly two name units are required, and all nine two-unit combinations are accepted. A single name, three or more units, retired aliases, and text with extra words remain rejected.

## Tests

```bash
cd voice_control
pytest -q
```

Current expected result: `48 passed`.

## Current boundary

Stage 3 uses lightweight RMS energy detection, not a neural speech/noise classifier. Its threshold must be calibrated for the installed microphone and room. Runtime TTS, barge-in/echo cancellation, neural VAD, ROS actions, camera input, and real elevator interfaces remain later stages. The configured `vad` path is retained for a future neural model but is not loaded.

## Orin Nano → AGX migration

Orin Nano is for development validation; AGX is the final integration target. The core is host-independent and the ROS 2 gateway remains outside this wheel.

The **Orin Nano is the voice-system development and validation platform**. The **AGX is the final multimodal robot integration platform**. Both run the same wheel; only the device profile and separately copied model directory change.

1. Build a portable wheel on the development machine: `python -m pip wheel --no-deps . -w dist`. The wheel intentionally contains no model weights.
2. Copy the wheel and `config/agx.example.yaml` to the AGX. Create an independent environment there: `python3 -m venv .venv`, then install the wheel with `.venv/bin/pip install voice_control-*.whl`.
3. Discover the AGX's stable PulseAudio/PipeWire source and sink names. Replace both `REPLACE_ON_AGX...` values; do not copy an ALSA card number from Orin.
4. Copy the tested SenseVoice directory and wake-prompt WAV separately. Models and prompt audio are **not packaged in the wheel**. Set AGX-local `model_dir` and `wake_prompt`. The verified runtime pair is `sherpa-onnx==1.13.4` with `numpy==1.26.4`. Copy the tested model SHA-256 into `model_sha256`; `doctor` refuses a corrupted or different model. The neural `vad` path remains empty; copy the tested energy-VAD settings first, then calibrate on AGX.
5. Run `.venv/bin/voice-control --config agx.yaml doctor`. Do not integrate a real gateway until `config_ok` is true. Keep `dry_run: true` for the first AGX validation.
6. Run `listen-once`, then `session --max-utterances 1` on AGX and compare JSONL audits with the Orin validation. No code changes should be needed. Real multimodal/ROS integration remains outside this wheel and behind a future gateway.
