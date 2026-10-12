"""Portable voice-control core (Stage 2 remains dry-run)."""

from voice_control.config import VoiceControlConfig, load_config
from voice_control.gateway import DryRunGateway
from voice_control.intents import Intent, ParseResult
from voice_control.parser import parse_utterance
from voice_control.state_machine import VoiceStateMachine

__all__ = [
    "DryRunGateway",
    "Intent",
    "ParseResult",
    "VoiceControlConfig",
    "VoiceStateMachine",
    "load_config",
    "parse_utterance",
]

__version__ = "0.4.3"
