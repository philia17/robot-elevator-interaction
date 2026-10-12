from lift_robot_camera_adapter.session_core import (
    ProbeFreshnessDeadline,
    PROBE_STALE_EXIT,
    SESSION_EXHAUSTED_EXIT,
    SessionRetryPolicy,
)
from lift_robot_camera_adapter import phase2_session


def test_probe_requires_ready_before_startup_deadline():
    gate = ProbeFreshnessDeadline(0, startup_timeout_s=30, stale_timeout_s=10)
    assert gate.update(29.9, 'WAITING') is None
    assert 'never reached READY' in gate.update(30.0, 'WAITING')


def test_probe_exits_after_continuous_non_ready_period():
    gate = ProbeFreshnessDeadline(0, startup_timeout_s=30, stale_timeout_s=10)
    assert gate.update(2, 'READY') is None
    assert gate.update(3, 'STALE') is None
    assert gate.update(8, 'WAITING') is None
    assert 'remained non-READY' in gate.update(13, 'STALE')


def test_new_ready_timestamp_sequence_resets_stale_deadline():
    gate = ProbeFreshnessDeadline(0, startup_timeout_s=30, stale_timeout_s=10)
    gate.update(2, 'READY')
    gate.update(3, 'STALE')
    assert gate.update(12, 'READY') is None
    assert gate.update(20, 'STALE') is None
    assert gate.update(29, 'WAITING') is None


def test_default_runtime_stale_boundary_is_thirty_seconds():
    gate = ProbeFreshnessDeadline(0)
    gate.update(1, 'READY')
    gate.update(2, 'STALE')
    assert gate.update(31.9, 'WAITING') is None
    assert 'remained non-READY' in gate.update(32.0, 'STALE')


def test_session_allows_exactly_one_complete_restart():
    policy = SessionRetryPolicy(max_restarts=1)
    first = policy.child_failed()
    assert (first.action, first.attempt) == ('restart_full_session', 2)
    second = policy.child_failed()
    assert (second.action, second.attempt) == ('stop_fail_closed', 2)


def test_probe_and_exhausted_session_have_distinct_exit_codes():
    assert PROBE_STALE_EXIT == 75
    assert SESSION_EXHAUSTED_EXIT == 76
    assert phase2_session.CAMERA_CONFLICT_EXIT == 77


def test_camera_process_conflicts_are_exact(tmp_path):
    proc_root = tmp_path / 'proc'
    commands = {
        101: [
            '/usr/bin/python3', '/opt/ros/humble/bin/ros2', 'launch',
            'lift_robot_camera_adapter_cpp', 'gemini335_canonical_components.launch.py',
        ],
        102: [
            '/opt/ros/humble/lib/rclcpp_components/component_container_mt',
            '--ros-args', '-r', '__node:=gemini335_canonical_container',
        ],
        103: [
            '/opt/ros/humble/lib/rclcpp_components/component_container_mt',
            '--ros-args', '-r', '__node:=unrelated_container',
        ],
    }
    for pid, tokens in commands.items():
        process_dir = proc_root / str(pid)
        process_dir.mkdir(parents=True)
        (process_dir / 'cmdline').write_bytes(b'\0'.join(token.encode() for token in tokens) + b'\0')
    conflicts = phase2_session.find_camera_process_conflicts(proc_root)
    assert [pid for pid, _command in conflicts] == [101, 102]


def test_attempt_refuses_existing_camera_without_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(
        phase2_session,
        'find_camera_process_conflicts',
        lambda: [(123, 'ros2 launch lift_robot_camera_adapter_cpp ...')],
    )
    result = phase2_session._run_attempt(
        1, tmp_path / 'formal.json', tmp_path / 'logs', {}, 0.01
    )
    assert result == 'conflict'


def test_outer_launcher_runs_at_most_two_fresh_attempts(tmp_path, monkeypatch):
    attempts = []

    def fail_attempt(attempt, *_args):
        attempts.append(attempt)
        return 'failed'

    monkeypatch.setattr(phase2_session, '_run_attempt', fail_attempt)
    monkeypatch.setattr(phase2_session.time, 'sleep', lambda _seconds: None)
    code = phase2_session.main([
        '--output', str(tmp_path / 'formal.json'),
        '--log-dir', str(tmp_path / 'logs'),
    ])
    assert code == SESSION_EXHAUSTED_EXIT
    assert attempts == [1, 2]


def test_outer_launcher_stops_after_first_success(tmp_path, monkeypatch):
    attempts = []

    def pass_attempt(attempt, *_args):
        attempts.append(attempt)
        return 'pass'

    monkeypatch.setattr(phase2_session, '_run_attempt', pass_attempt)
    code = phase2_session.main([
        '--output', str(tmp_path / 'formal.json'),
        '--log-dir', str(tmp_path / 'logs'),
    ])
    assert code == 0
    assert attempts == [1]


def test_outer_launcher_reports_camera_conflict_without_retry(tmp_path, monkeypatch):
    attempts = []

    def conflict_attempt(attempt, *_args):
        attempts.append(attempt)
        return 'conflict'

    monkeypatch.setattr(phase2_session, '_run_attempt', conflict_attempt)
    code = phase2_session.main([
        '--output', str(tmp_path / 'formal.json'),
        '--log-dir', str(tmp_path / 'logs'),
    ])
    assert code == phase2_session.CAMERA_CONFLICT_EXIT
    assert attempts == [1]
