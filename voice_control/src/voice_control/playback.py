"""Bounded half-duplex prompt playback through a stable PulseAudio sink."""

from __future__ import annotations

import subprocess
from pathlib import Path

PULSE_VOLUME_NORM = 65_536


class PlaybackError(RuntimeError):
    """Prompt playback failed or could not be bounded safely."""


def play_prompt(
    sink: str,
    prompt: str | Path,
    *,
    volume_percent: int = 50,
    timeout_seconds: float = 10.0,
) -> None:
    """Play one prompt after capture has stopped; never invokes a shell."""
    if not sink or "REPLACE_ON_AGX" in sink:
        raise PlaybackError("pulse_sink is empty or still an AGX placeholder")
    path = Path(prompt)
    if not path.is_file():
        raise PlaybackError(f"prompt file not found: {path}")
    if not 0 <= volume_percent <= 100:
        raise PlaybackError("volume_percent must be between 0 and 100")
    if timeout_seconds <= 0:
        raise PlaybackError("timeout_seconds must be positive")

    linear_volume = round(PULSE_VOLUME_NORM * volume_percent / 100)
    command = [
        "paplay",
        f"--device={sink}",
        f"--volume={linear_volume}",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PlaybackError("paplay was not found; install PulseAudio utilities") from exc
    except subprocess.TimeoutExpired as exc:
        raise PlaybackError(
            f"prompt playback timed out after {timeout_seconds:g} seconds"
        ) from exc
    except OSError as exc:
        raise PlaybackError(f"failed to start paplay: {exc}") from exc

    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-2000:]
        suffix = f": {detail}" if detail else ""
        raise PlaybackError(f"paplay exited with {completed.returncode}{suffix}")
