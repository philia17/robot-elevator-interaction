"""Pure state machines for the fail-closed Phase 2 camera session."""

from dataclasses import dataclass


PROBE_STALE_EXIT = 75
SESSION_EXHAUSTED_EXIT = 76


class ProbeFreshnessDeadline:
    """Turn a persistent non-READY preview into a dedicated process exit."""

    def __init__(self, started_at, *, startup_timeout_s=30.0, stale_timeout_s=30.0):
        self.started_at = started_at
        self.startup_timeout_s = startup_timeout_s
        self.stale_timeout_s = stale_timeout_s
        self.ever_ready = False
        self.non_ready_since = None

    def update(self, now, status):
        if status == 'READY':
            self.ever_ready = True
            self.non_ready_since = None
            return None
        if not self.ever_ready:
            elapsed = now - self.started_at
            if elapsed >= self.startup_timeout_s:
                return f'preview never reached READY within {self.startup_timeout_s:.1f}s'
            return None
        if self.non_ready_since is None:
            self.non_ready_since = now
            return None
        elapsed = now - self.non_ready_since
        if elapsed >= self.stale_timeout_s:
            return f'preview remained non-READY for {elapsed:.1f}s'
        return None


@dataclass(frozen=True)
class SessionDecision:
    action: str
    attempt: int


class SessionRetryPolicy:
    """Allow one complete fresh-session restart and never form a restart loop."""

    def __init__(self, *, max_restarts=1):
        self.max_restarts = max_restarts
        self.attempt = 1

    def child_failed(self):
        if self.attempt <= self.max_restarts:
            self.attempt += 1
            return SessionDecision('restart_full_session', self.attempt)
        return SessionDecision('stop_fail_closed', self.attempt)
