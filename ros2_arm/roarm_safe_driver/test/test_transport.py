import json
import math
import sys
import types

import pytest

from roarm_safe_driver.transport import (
    FakeSerialTransport,
    ReadOnlySerialTransport,
    SerialTransport,
    TrackingFakeSerialTransport,
)


def test_fake_transport_isolated_and_records_writes():
    transport = FakeSerialTransport()
    transport.open()
    transport.lines.put(b"hello\n")
    assert transport.readline() == b"hello\n"
    transport.write(b"command\n")
    assert transport.writes == [b"command\n"]
    transport.close()


def _parse_sample(transport):
    frame = json.loads(transport.sample_line().decode())
    assert frame["T"] == 1051
    # Mirror core.parse_feedback_line: ros gripper = pi - firmware g.
    return [frame["b"], frame["s"], frame["e"], frame["t"], frame["r"],
            math.pi - frame["g"]]


def test_tracking_fake_initial_pose_round_trips():
    ros_pose = [-0.0123, -0.7286, 2.0003, 1.0232, 0.0015, 0.0307]
    transport = TrackingFakeSerialTransport(ros_pose)
    assert _parse_sample(transport) == pytest.approx(ros_pose)
    frame = json.loads(transport.sample_line().decode())
    assert all(frame[key] == 1 for key in (
        "torswitchB", "torswitchS", "torswitchE",
        "torswitchT", "torswitchR", "torswitchG",
    ))
    assert frame["v"] == 1210


def test_tracking_fake_follows_t102_targets():
    transport = TrackingFakeSerialTransport([0.0] * 6)
    payload = {
        "T": 102,
        "base": 0.1, "shoulder": -0.2, "elbow": 1.9, "wrist": 0.5,
        "roll": 0.05, "hand": math.pi - 0.03, "spd": 17, "acc": 1,
    }
    transport.write((json.dumps(payload) + "\n").encode())
    assert _parse_sample(transport) == pytest.approx(
        [0.1, -0.2, 1.9, 0.5, 0.05, 0.03]
    )


def test_tracking_fake_follows_t101_gripper_firmware_frame():
    transport = TrackingFakeSerialTransport([0.0] * 6)
    # build_t101 sends firmware rad = pi - ros for the gripper (joint id 6).
    payload = {"T": 101, "joint": 6, "rad": math.pi - 0.25, "spd": 17, "acc": 1}
    transport.write((json.dumps(payload) + "\n").encode())
    assert _parse_sample(transport)[5] == pytest.approx(0.25)


def test_tracking_fake_ignores_non_command_writes():
    transport = TrackingFakeSerialTransport([0.0] * 6)
    transport.write(b'{"T":210,"cmd":1}\n')
    transport.write(b"not json\n")
    assert _parse_sample(transport) == pytest.approx([0.0] * 6)
    assert len(transport.writes) == 2


def test_tracking_fake_rejects_bad_initial_pose():
    with pytest.raises(ValueError):
        TrackingFakeSerialTransport([0.0] * 5)


def test_real_transport_rejects_unstable_tty_path_before_open():
    transport = SerialTransport("/dev/ttyUSB0", 115200)
    with pytest.raises(ValueError, match="stable"):
        transport.open()


def test_real_transport_sets_non_reset_control_lines_before_open(monkeypatch):
    events = []

    class RecordingSerial:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs))

        def __setattr__(self, name, value):
            events.append((name, value))
            object.__setattr__(self, name, value)

        def open(self):
            events.append(("open", None))

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=RecordingSerial))
    transport = SerialTransport("/dev/serial/by-id/example", 115200)
    transport.open()
    assert events == [
        ("construct", {"port": None, "baudrate": 115200, "timeout": 0.1}),
        ("dtr", True),
        ("rts", True),
        ("port", "/dev/serial/by-id/example"),
        ("open", None),
    ]


class _FragmentingSerial:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def readline(self):
        return self.chunks.pop(0) if self.chunks else b""


@pytest.mark.parametrize("transport_type", [SerialTransport, ReadOnlySerialTransport])
def test_real_transport_buffers_timeout_fragments_until_complete_newline(transport_type):
    transport = transport_type("/dev/serial/by-id/example", 115200)
    transport._serial = _FragmentingSerial(
        [b'{"T":1051,"b":0', b'.1}\r', b"\n"]
    )
    assert transport.readline() == b""
    assert transport.readline() == b""
    assert transport.readline() == b'{"T":1051,"b":0.1}\r\n'


@pytest.mark.parametrize("transport_type", [SerialTransport, ReadOnlySerialTransport])
def test_real_transport_fails_closed_on_unterminated_oversized_frame(transport_type):
    transport = transport_type("/dev/serial/by-id/example", 115200)
    transport._serial = _FragmentingSerial([b"x" * 4097])
    assert transport.readline() == b"__serial_frame_overflow__\n"


def test_read_only_transport_has_no_write_capability():
    transport = ReadOnlySerialTransport("/dev/serial/by-id/example", 115200)
    assert not hasattr(transport, "write")


def test_read_only_transport_rejects_unstable_tty_path_before_open():
    transport = ReadOnlySerialTransport("/dev/ttyUSB0", 115200)
    with pytest.raises(ValueError, match="stable"):
        transport.open()
