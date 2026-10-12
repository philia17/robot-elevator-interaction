"""Execution boundary. Stage one ships only a non-actuating dry-run gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

CompletionCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class SubmissionRecord:
    command_id: str
    floor: int
    generation: int


@dataclass
class DryRunGateway:
    auto_complete: bool = True
    submissions: list[SubmissionRecord] = field(default_factory=list)
    abort_count: int = 0
    _aborted: bool = field(default=False, repr=False)
    _callbacks: list[tuple[int, CompletionCallback]] = field(default_factory=list, repr=False)

    def submit(self, command_id: str, floor: int, generation: int,
               on_complete: CompletionCallback) -> None:
        self.submissions.append(SubmissionRecord(command_id, floor, generation))
        self._callbacks.append((generation, on_complete))
        if self.auto_complete and not self._aborted:
            self.complete_last()

    def complete_last(self, outcome: str = "success") -> None:
        if not self._callbacks:
            return
        generation, callback = self._callbacks.pop()
        callback(generation, outcome)

    def abort(self) -> bool:
        """Latch abort once. Repeated calls have no additional side effects."""
        if self._aborted:
            return False
        self._aborted = True
        self.abort_count += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"abort_count": self.abort_count, "submission_count": len(self.submissions),
                "aborted": self._aborted}
