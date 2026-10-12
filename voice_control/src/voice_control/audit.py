"""JSONL audit logging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


def utc_now_iso() -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AuditLogger:
  path: Path
  _opened: bool = field(default=False, repr=False)

  def __post_init__(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)

  def log(self, event_type: str, **fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
      "ts": utc_now_iso(),
      "event": event_type,
    }
    for key, value in fields.items():
      if value is None:
        continue
      if hasattr(value, "value"):
        record[key] = value.value
      else:
        record[key] = value
    with self.path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def iter_audit_records(path: Path) -> Iterator[dict[str, Any]]:
  with path.open(encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {line_no}") from exc


def summarize_audit(path: Path) -> dict[str, Any]:
  events = list(iter_audit_records(path))
  intents = [e.get("intent") for e in events if e.get("event") == "parse"]
  transitions = [e for e in events if e.get("event") == "state_transition"]
  outcomes = [e for e in events if e.get("event") == "execution_outcome"]
  return {
    "path": str(path),
    "event_count": len(events),
    "parse_intents": intents,
    "transition_count": len(transitions),
    "execution_outcomes": [o.get("outcome") for o in outcomes],
    "last_state": transitions[-1].get("to_state") if transitions else None,
  }


def default_audit_path(base_dir: Optional[Path] = None) -> Path:
  root = base_dir or Path.cwd() / ".voice_control"
  return root / "audit.jsonl"
