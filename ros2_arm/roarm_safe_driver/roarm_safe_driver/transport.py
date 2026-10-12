"""Serial transports used by the safe driver."""

from __future__ import annotations

import json
import math
import queue
import threading


class FakeSerialTransport:
    """In-memory transport for tests and hardware-isolated runtime checks."""

    def __init__(self) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()
        self.writes: list[bytes] = []
        self.closed = False

    def open(self) -> None:
        self.closed = False

    def readline(self) -> bytes:
        try:
            return self.lines.get(timeout=0.05)
        except queue.Empty:
            return b""

    def write(self, data: bytes) -> None:
        if self.closed:
            raise OSError("transport closed")
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True


class TrackingFakeSerialTransport(FakeSerialTransport):
    """Fake transport whose feedback echoes commanded T101/T102 targets.

    Used for hardware-isolated full-route rehearsals: written commands update
    an internal firmware-frame pose, and `sample_line()` renders it as a
    healthy T:1051 frame (all torque switches on, 12.10 V, zero loads).
    Motion is instantaneous — convergence timing is then governed purely by
    the driver's own terminal-settle policy.

    Firmware frame conventions (mirrors core.build_t101/build_t102 and
    core.parse_feedback_line): joints 0..4 are identical to ROS radians; the
    gripper is stored as firmware ``g`` where ``ros = pi - g``.
    """

    _T101_JOINT_ID_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def __init__(self, initial_positions_rad=None) -> None:
        super().__init__()
        ros = list(initial_positions_rad) if initial_positions_rad else [
            0.0, 0.0, 1.6, 0.0, 0.0, 0.0,
        ]
        if len(ros) != 6 or not all(math.isfinite(float(v)) for v in ros):
            raise ValueError("initial_positions_rad must contain six finite values")
        self._lock = threading.Lock()
        self._firmware = [float(v) for v in ros[:5]] + [math.pi - float(ros[5])]

    def write(self, data: bytes) -> None:
        super().write(data)
        try:
            obj = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(obj, dict):
            return
        with self._lock:
            if obj.get("T") == 101:
                index = self._T101_JOINT_ID_TO_INDEX.get(obj.get("joint"))
                if index is not None:
                    # T101 "rad" is already firmware-frame (gripper inverted
                    # by build_t101).
                    self._firmware[index] = float(obj["rad"])
            elif obj.get("T") == 102:
                # T102 named keys are firmware-frame ("hand" inverted by
                # build_t102).
                self._firmware = [
                    float(obj["base"]),
                    float(obj["shoulder"]),
                    float(obj["elbow"]),
                    float(obj["wrist"]),
                    float(obj["roll"]),
                    float(obj["hand"]),
                ]

    def sample_line(self) -> bytes:
        with self._lock:
            b, s, e, t, r, g = self._firmware
        frame = {
            "T": 1051,
            "b": b, "s": s, "e": e, "t": t, "r": r, "g": g,
            "tB": 0, "tS": 0, "tE": 0, "tT": 0, "tR": 0, "tG": 0,
            "torswitchB": 1, "torswitchS": 1, "torswitchE": 1,
            "torswitchT": 1, "torswitchR": 1, "torswitchG": 1,
            "v": 1210,
        }
        return (json.dumps(frame) + "\n").encode("utf-8")


class _NewlineFrameBuffer:
    """Accumulate serial timeout fragments until a complete newline frame exists."""

    # A T:1051 frame is small.  Treat a never-terminated stream larger than
    # this as malformed rather than retaining arbitrary serial garbage in RAM.
    MAX_PENDING_BYTES = 4096

    def __init__(self) -> None:
        self._pending = bytearray()

    def pop_complete(self) -> bytes:
        newline = self._pending.find(b"\n")
        if newline < 0:
            return b""
        frame = bytes(self._pending[: newline + 1])
        del self._pending[: newline + 1]
        return frame

    def append(self, chunk: bytes) -> bool:
        self._pending.extend(chunk)
        if len(self._pending) > self.MAX_PENDING_BYTES:
            self._pending.clear()
            return False
        return True

    def clear(self) -> None:
        self._pending.clear()


class SerialTransport:
    def __init__(self, port: str, baud_rate: int, timeout: float = 0.1) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial = None
        self._lock = threading.Lock()
        self._frames = _NewlineFrameBuffer()

    def open(self) -> None:
        if not self.port.startswith("/dev/serial/by-id/"):
            raise ValueError("real hardware requires a stable /dev/serial/by-id path")
        import serial

        device = serial.Serial(port=None, baudrate=self.baud_rate, timeout=self.timeout)
        device.dtr = True
        device.rts = True
        device.port = self.port
        device.open()
        self._serial = device
        self._frames.clear()

    def readline(self) -> bytes:
        if self._serial is None:
            return b""
        frame = self._frames.pop_complete()
        if frame:
            return frame
        chunk = self._serial.readline()
        if not chunk:
            return b""
        if not self._frames.append(chunk):
            # Preserve the driver's fail-closed parser contract for an
            # unterminated oversized serial frame.
            return b"__serial_frame_overflow__\n"
        return self._frames.pop_complete()

    def write(self, data: bytes) -> None:
        if self._serial is None:
            raise OSError("serial is not open")
        with self._lock:
            self._serial.write(data)
            self._serial.flush()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._frames.clear()


class ReadOnlySerialTransport:
    """Serial input boundary that deliberately has no write method."""

    def __init__(self, port: str, baud_rate: int, timeout: float = 0.1) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial = None
        self._frames = _NewlineFrameBuffer()

    def open(self) -> None:
        if not self.port.startswith("/dev/serial/by-id/"):
            raise ValueError("real hardware requires a stable /dev/serial/by-id path")
        import serial

        device = serial.Serial(
            port=None,
            baudrate=self.baud_rate,
            timeout=self.timeout,
            exclusive=True,
        )
        device.dtr = False
        device.rts = False
        device.port = self.port
        device.open()
        self._serial = device
        self._frames.clear()

    def readline(self) -> bytes:
        if self._serial is None:
            return b""
        frame = self._frames.pop_complete()
        if frame:
            return frame
        chunk = self._serial.readline()
        if not chunk:
            return b""
        if not self._frames.append(chunk):
            return b"__serial_frame_overflow__\n"
        return self._frames.pop_complete()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._frames.clear()
