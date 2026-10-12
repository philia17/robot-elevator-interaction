"""Intent and parse result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
  PRESS_FLOOR = "PRESS_FLOOR"
  CONFIRM = "CONFIRM"
  CANCEL = "CANCEL"
  STOP = "STOP"
  STATUS = "STATUS"
  WAKE = "WAKE"
  UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ParseResult:
  intent: Intent
  floor: Optional[int] = None
  raw_text: str = ""
  reject_reason: Optional[str] = None

  @property
  def accepted(self) -> bool:
    if self.intent == Intent.UNKNOWN:
      return False
    if self.intent == Intent.PRESS_FLOOR:
      return self.floor is not None and self.reject_reason is None
    return self.reject_reason is None
