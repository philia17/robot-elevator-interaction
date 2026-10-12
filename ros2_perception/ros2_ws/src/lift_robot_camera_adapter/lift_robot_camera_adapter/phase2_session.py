"""Non-ROS owner for one or two complete fresh Phase 2 camera sessions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .session_core import PROBE_STALE_EXIT, SESSION_EXHAUSTED_EXIT, SessionRetryPolicy


RESTART_INSTRUCTION = (
    '相机流程连续两次未恢复，已安全停止。请关闭残留窗口后，重新启动相机流程。'
)
CAMERA_CONFLICT_EXIT = 77
CAMERA_CONFLICT_INSTRUCTION = (
    '检测到其他 Gemini 相机流程，未启动 Phase 2。请先关闭实时拍照/3D 相机流程。'
)


def find_camera_process_conflicts(proc_root: Path = Path('/proc')):
    """Return exact host-local Gemini camera owners that would collide."""
    conflicts = []
    try:
        entries = proc_root.iterdir()
    except OSError:
        return conflicts
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            tokens = [
                token.decode(errors='replace')
                for token in (entry / 'cmdline').read_bytes().split(b'\0')
                if token
            ]
        except (OSError, ValueError):
            continue
        command = ' '.join(tokens)
        is_camera_launch = (
            any(token == 'ros2' or token.endswith('/ros2') for token in tokens)
            and 'launch' in tokens
            and 'lift_robot_camera_adapter_cpp' in tokens
            and 'gemini335_canonical_components.launch.py' in tokens
        )
        is_camera_container = (
            any(token.endswith('/component_container_mt') for token in tokens)
            and '__node:=gemini335_canonical_container' in tokens
        )
        if is_camera_launch or is_camera_container:
            conflicts.append((int(entry.name), command))
    return sorted(conflicts)


class OwnedProcess:
    def __init__(self, command, log_path, env):
        self.command = command
        self.log_path = log_path
        self.env = env
        self.process = None
        self.log_handle = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )

    def poll(self):
        return self.process.poll()

    def stop(self):
        if self.process is None:
            return
        try:
            pgid = os.getpgid(self.process.pid)
        except ProcessLookupError:
            pgid = self.process.pid
        for sig, timeout_s in (
            (signal.SIGCONT, 0.1),
            (signal.SIGINT, 4.0),
            (signal.SIGTERM, 2.0),
            (signal.SIGKILL, 2.0),
        ):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                continue
            break
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'process group {pgid} did not terminate') from exc
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'process group {pgid} left residual processes')
        if self.log_handle is not None:
            self.log_handle.close()
        self.process = None
        self.log_handle = None


def _promote_passed_evidence(attempt_output, formal_output):
    if not attempt_output.exists():
        return False
    document = json.loads(attempt_output.read_text(encoding='utf-8'))
    if document.get('result', {}).get('status') != 'pass':
        return False
    if formal_output.exists():
        raise RuntimeError(f'refusing to overwrite existing evidence: {formal_output}')
    formal_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(attempt_output, formal_output)
    return True


def _run_attempt(attempt, output_path, log_dir, env, poll_s):
    conflicts = find_camera_process_conflicts()
    if conflicts:
        for pid, command in conflicts:
            print(f'PHASE2_SESSION camera_conflict pid={pid} command={command}', flush=True)
        return 'conflict'
    attempt_dir = log_dir / f'attempt_{attempt}'
    attempt_output = attempt_dir / 'axis_observations.json'
    camera = OwnedProcess(
        ['ros2', 'launch', 'lift_robot_camera_adapter_cpp', 'gemini335_canonical_components.launch.py'],
        attempt_dir / 'camera.log',
        env,
    )
    probe = OwnedProcess(
        [
            'ros2', 'run', 'lift_robot_camera_adapter', 'phase2_optical_axis_probe',
            '--output', str(attempt_output),
        ],
        attempt_dir / 'probe.log',
        env,
    )
    camera.start()
    probe.start()
    print(f'PHASE2_SESSION attempt={attempt} started camera_pid={camera.process.pid} '
          f'probe_pid={probe.process.pid}', flush=True)
    try:
        while True:
            probe_code = probe.poll()
            camera_code = camera.poll()
            if probe_code is not None:
                if probe_code == 0 and _promote_passed_evidence(attempt_output, output_path):
                    print(f'PHASE2_SESSION PASS evidence={output_path}', flush=True)
                    return 'pass'
                if probe_code == 0:
                    print('PHASE2_SESSION operator closed probe before PASS', flush=True)
                    return 'cancelled'
                print(f'PHASE2_SESSION probe_exit={probe_code}', flush=True)
                return 'failed'
            if camera_code is not None:
                print(f'PHASE2_SESSION camera_exit={camera_code}', flush=True)
                return 'failed'
            time.sleep(poll_s)
    finally:
        probe.stop()
        camera.stop()
        if probe.process is not None or camera.process is not None:
            raise RuntimeError('owned processes were not fully cleared')


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--display', default=':1002')
    parser.add_argument('--cooldown-s', type=float, default=5.0)
    parser.add_argument('--poll-s', type=float, default=0.25)
    parsed = parser.parse_args(args)
    output_path = parsed.output.expanduser().resolve()
    log_dir = parsed.log_dir.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f'refusing to overwrite existing evidence: {output_path}')
    env = dict(os.environ)
    env['DISPLAY'] = parsed.display
    env.setdefault('XAUTHORITY', str(Path.home() / '.Xauthority'))
    policy = SessionRetryPolicy(max_restarts=1)
    try:
        while True:
            result = _run_attempt(policy.attempt, output_path, log_dir, env, parsed.poll_s)
            if result == 'pass' or result == 'cancelled':
                return 0
            if result == 'conflict':
                print(
                    f'PHASE2_SESSION CONFLICT: {CAMERA_CONFLICT_INSTRUCTION}',
                    file=sys.stderr,
                    flush=True,
                )
                return CAMERA_CONFLICT_EXIT
            decision = policy.child_failed()
            if decision.action == 'stop_fail_closed':
                print(f'PHASE2_SESSION FAILED: {RESTART_INSTRUCTION}', file=sys.stderr, flush=True)
                return SESSION_EXHAUSTED_EXIT
            print(
                f'PHASE2_SESSION full_restart attempt={decision.attempt} '
                f'cooldown={parsed.cooldown_s:.1f}s',
                flush=True,
            )
            time.sleep(parsed.cooldown_s)
    except KeyboardInterrupt:
        print('PHASE2_SESSION stopped by operator', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
