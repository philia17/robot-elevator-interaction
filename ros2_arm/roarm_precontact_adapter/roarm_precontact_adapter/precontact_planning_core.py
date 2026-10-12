"""Trusted, no-motion planner for a freshly cached Button 1 observation.

The production default is :class:`NotImplementedPlanner`.  The trusted planner
is an opt-in, in-process composition root: it accepts only canonical bytes
already held by the adapter, binds those bytes to their caller-supplied digest,
creates the dynamic M0/normal releases, and invokes the installed outbound
route builder with injected offline IK, FK, and collision implementations.

This module imports no ROS, serial, driver, or robot APIs.  Its outputs are
release artifacts, never motion authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import secrets
import threading
from typing import Any, Mapping, Protocol, Sequence

from .precontact_execution_core import (
    AdmissionGates,
    DEFAULT_RELEASE_MANIFEST,
    NotImplementedExecutor,
    PrecontactExecutionCore,
    PrepareResult,
    RELEASE_SCHEMA,
    TARGET_SCHEMA,
    canonical_sha256 as execution_canonical_sha256,
    parse_frozen_target,
    parse_release_manifest,
)
from .precontact_outbound_route import (
    CollisionChecker,
    IKSolver,
    KinematicsChecker,
    MANIFEST_SCHEMA,
    PHASE,
    RouteBlocked,
    build_outbound_route,
    seal_manifest,
    seal_release,
    sha256_canonical,
)


FREEZE_REQUEST_SCHEMA = "button1_m0_freeze_request_v1"
STATIC_CATALOG_SCHEMA = "precontact_static_release_catalog_v1"
MAX_M0_TTL_MS = 1000.0
MAX_CACHE_TTL_S = 1.0
_STATIC_RELEASE_NAMES = (
    "r6_bundle",
    "tool_release",
    "standby_release",
    "urdf_release",
    "driver_profile_release",
    "collision_model_release",
)
_FREEZE_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "label",
        "target_id",
        "detection_id",
        "semantic_track_id",
        "source_stamp_ns",
        "source_frame_id",
        "expires_at_ns",
        "ttl_ms",
        "calibration_release_id",
        "point_camera_m",
        "normal_camera",
        "confidence",
        "rgb_depth_skew_ms",
        "depth_status",
        "normal_status",
        "ready_status",
        "normal_outward_evidence",
        "source_canonical_sha256",
    }
)


class PlanningBlocked(ValueError):
    """A cached observation, release, dependency, or route failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlanningBlocked(f"canonical JSON failed: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonicalize_button1_m0_freeze_request(
    observation: Mapping[str, Any],
) -> bytes:
    """Convert trusted M0 callback fields into the sole freeze-request bytes.

    The ROS shell supplies a plain mapping so this core remains ROS-free.  A
    normal is marked valid only when the upstream observation explicitly says
    ``VALID_OUTWARD`` and supplies positive outward evidence; the current
    M0Target3D message has no such execution-grade evidence and therefore
    canonicalizes to an unverified request that the planner will reject.
    """

    source = dict(_mapping(observation, "M0 callback observation"))
    stamp = source.get("source_stamp_ns")
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
        raise PlanningBlocked("M0 callback source_stamp_ns is invalid")
    target_label = str(source.get("target_label") or "").strip()
    effective_label = str(source.get("effective_label") or "").strip()
    detection_id = str(source.get("detection_id") or "").strip()
    semantic_track_id = str(source.get("semantic_track_id") or "").strip()
    if target_label != "1" or effective_label != "1":
        raise PlanningBlocked("M0 callback does not identify Button 1")
    if not detection_id or not semantic_track_id:
        raise PlanningBlocked("M0 callback detection/semantic identity is incomplete")
    ttl_ms = MAX_M0_TTL_MS
    normal_status = str(source.get("normal_direction_status") or "UNVERIFIED")
    outward = {
        "input_normal_was_flipped": source.get("input_normal_was_flipped") is True,
        "dot_n_out_p_minus_a": _finite(
            source.get("dot_n_out_p_minus_a", 0.0),
            "M0 callback dot_n_out_p_minus_a",
        ),
        "camera_side_dot": _finite(
            source.get("camera_side_dot", 0.0),
            "M0 callback camera_side_dot",
        ),
    }
    request = {
        "schema": FREEZE_REQUEST_SCHEMA,
        "label": "1",
        "target_id": f"button1:{semantic_track_id}:{detection_id}",
        "detection_id": detection_id,
        "semantic_track_id": semantic_track_id,
        "source_stamp_ns": stamp,
        "source_frame_id": str(source.get("source_frame_id") or "").strip(),
        "expires_at_ns": stamp + int(ttl_ms * 1_000_000),
        "ttl_ms": ttl_ms,
        "calibration_release_id": str(source.get("calibration_release_id") or "").strip(),
        "point_camera_m": list(_vec(source.get("point_camera_m"), 3, "M0 callback point")),
        "normal_camera": list(_unit(source.get("normal_camera"), "M0 callback normal")),
        "confidence": _finite(source.get("confidence"), "M0 callback confidence"),
        "rgb_depth_skew_ms": _finite(
            source.get("rgb_depth_skew_ms"),
            "M0 callback RGB/depth skew",
        ),
        "depth_status": (
            "VALID_DEPTH"
            if source.get("depth_valid") is True
            and source.get("surface_estimate_valid") is True
            and source.get("surface_provenance_complete") is True
            else "INVALID_DEPTH"
        ),
        "normal_status": normal_status,
        "ready_status": (
            "READY"
            if source.get("status") == "READY"
            and source.get("perception_ready") is True
            and source.get("motion_authorized") is False
            else "NOT_READY"
        ),
        "normal_outward_evidence": outward,
        "source_canonical_sha256": canonical_sha256(source),
    }
    return canonical_json_bytes(request)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanningBlocked(f"{field} must be an object")
    return value


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlanningBlocked(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PlanningBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanningBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise PlanningBlocked(f"{field} must be finite")
    return result


def _vec(value: Any, length: int, field: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise PlanningBlocked(f"{field} must contain exactly {length} finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vec(value, 3, field)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if not 0.999 <= magnitude <= 1.001:
        raise PlanningBlocked(f"{field} must be a unit vector")
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _canonical_mapping_from_bytes(value: bytes, field: str) -> Mapping[str, Any]:
    if not isinstance(value, bytes) or not value:
        raise PlanningBlocked(f"{field} must be non-empty canonical bytes")
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanningBlocked(f"{field} is not canonical JSON: {exc}") from exc
    parsed = _mapping(parsed, field)
    if canonical_json_bytes(parsed) != value:
        raise PlanningBlocked(f"{field} is not in the canonical JSON representation")
    return parsed


def _rotate_camera_to_base(
    quaternion_xyzw: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    x, y, z, w = _vec(quaternion_xyzw, 4, "r6.rotation_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise PlanningBlocked("r6.rotation_xyzw has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vx, vy, vz = _vec(vector, 3, "normal_camera")
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    rotated = tuple(sum(row[index] * (vx, vy, vz)[index] for index in range(3)) for row in matrix)
    return _unit(rotated, "normal_out_base")


def seal_static_release_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a static catalog for controlled offline fixtures/configuration.

    This helper only binds canonical bytes.  It does not approve releases and
    is never called on per-request client data by the trusted planner.
    """

    result = dict(value)
    result.pop("catalog_sha256", None)
    result["catalog_sha256"] = canonical_sha256(result)
    return result


def _validate_release_envelope(value: Any, name: str) -> Mapping[str, Any]:
    release = _mapping(value, f"static_releases.{name}")
    release_id = str(release.get("release_id") or "").strip()
    if not release_id:
        raise PlanningBlocked(f"static release {name} has no release_id")
    if release.get("status") != "released":
        raise PlanningBlocked(f"static release {name} is diagnostic or unreleased")
    payload = _mapping(release.get("payload"), f"static_releases.{name}.payload")
    artifact_hash = _sha256(release.get("artifact_sha256"), f"static_releases.{name}.artifact_sha256")
    if not secrets.compare_digest(sha256_canonical(payload), artifact_hash):
        raise PlanningBlocked(f"static release {name} payload hash mismatch")
    return release


def _validate_static_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    catalog = dict(_mapping(value, "static release catalog"))
    if catalog.get("schema") != STATIC_CATALOG_SCHEMA:
        raise PlanningBlocked("unexpected static release catalog schema")
    if catalog.get("status") != "released" or not str(catalog.get("catalog_id") or "").strip():
        raise PlanningBlocked("static release catalog is not released")
    supplied_hash = _sha256(catalog.get("catalog_sha256"), "catalog_sha256")
    unsigned = dict(catalog)
    unsigned.pop("catalog_sha256", None)
    if not secrets.compare_digest(canonical_sha256(unsigned), supplied_hash):
        raise PlanningBlocked("static release catalog hash mismatch")

    releases = _mapping(catalog.get("static_releases"), "static_releases")
    allowlist = _mapping(catalog.get("allowlisted_release_bindings"), "allowlisted_release_bindings")
    if set(releases) != set(_STATIC_RELEASE_NAMES) or set(allowlist) != set(_STATIC_RELEASE_NAMES):
        raise PlanningBlocked("static release catalog names an unexpected or missing release")
    for name in _STATIC_RELEASE_NAMES:
        release = _validate_release_envelope(releases.get(name), name)
        bindings = allowlist.get(name)
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)) or not bindings:
            raise PlanningBlocked(f"allowlist for {name} is empty")
        expected = {
            "release_id": release["release_id"],
            "artifact_sha256": release["artifact_sha256"],
        }
        normalized: list[dict[str, str]] = []
        for index, raw_binding in enumerate(bindings):
            binding = _mapping(raw_binding, f"allowlist.{name}[{index}]")
            if set(binding) != {"release_id", "artifact_sha256"}:
                raise PlanningBlocked(f"allowlist binding for {name} has unknown fields")
            normalized.append(
                {
                    "release_id": str(binding.get("release_id") or ""),
                    "artifact_sha256": _sha256(
                        binding.get("artifact_sha256"),
                        f"allowlist.{name}[{index}].artifact_sha256",
                    ),
                }
            )
        if expected not in normalized:
            raise PlanningBlocked(f"selected static release {name} is not allowlisted")

    r6 = _mapping(releases["r6_bundle"]["payload"], "r6 payload")
    if (
        r6.get("schema") != "camera_base_release_v1"
        or r6.get("relative") is not False
        or r6.get("topology_closed") is not True
        or r6.get("tf_status") != "released_execution"
    ):
        raise PlanningBlocked("relative, diagnostic, or topology-open r6 release is forbidden")
    tool = _mapping(releases["tool_release"]["payload"], "tool payload")
    if (
        tool.get("schema") != "accepted_tool_frame_release_v1"
        or tool.get("transform_complete") is not True
        or tool.get("contact_surface_accepted") is not True
        or tool.get("rotation_status") != "accepted"
    ):
        raise PlanningBlocked("diagnostic or incomplete tool release is forbidden")
    collision = _mapping(releases["collision_model_release"]["payload"], "collision payload")
    if collision.get("schema") != "collision_model_release_v1" or collision.get("model_status") != "released":
        raise PlanningBlocked("collision model is missing, diagnostic, or unreleased")

    policy = _mapping(catalog.get("route_policy"), "route_policy")
    if (
        policy.get("allowed_phase") != PHASE
        or policy.get("no_contact") is not True
        or policy.get("automatic_retry") is not False
        or abs(_finite(policy.get("precontact_distance_m"), "precontact_distance_m") - 0.050) > 1e-12
    ):
        raise PlanningBlocked("static catalog route policy is not outbound-only 50 mm no-contact")
    return json.loads(canonical_json_bytes(catalog))


def _validate_freeze_request(
    canonical_request_bytes: bytes,
    request_sha256: str,
    *,
    now_ns: int,
    expected_calibration_release_id: str,
) -> Mapping[str, Any]:
    supplied_hash = _sha256(request_sha256, "freeze request SHA-256")
    if not secrets.compare_digest(_sha256_bytes(canonical_request_bytes), supplied_hash):
        raise PlanningBlocked("freeze request bytes do not match request SHA-256")
    request = _canonical_mapping_from_bytes(canonical_request_bytes, "cached freeze request")
    if set(request) != _FREEZE_REQUEST_FIELDS:
        raise PlanningBlocked("freeze request has unknown or missing fields")
    if request.get("schema") != FREEZE_REQUEST_SCHEMA or request.get("label") != "1":
        raise PlanningBlocked("freeze request is not canonical Button 1 M0 v1")
    for field in ("target_id", "detection_id", "semantic_track_id"):
        if not str(request.get(field) or "").strip():
            raise PlanningBlocked(f"freeze request {field} is required")
    if request.get("source_frame_id") != "camera_optical_frame":
        raise PlanningBlocked("freeze request frame must be camera_optical_frame")
    stamp_ns = request.get("source_stamp_ns")
    expiry_ns = request.get("expires_at_ns")
    if (
        isinstance(stamp_ns, bool)
        or not isinstance(stamp_ns, int)
        or isinstance(expiry_ns, bool)
        or not isinstance(expiry_ns, int)
        or stamp_ns <= 0
        or stamp_ns > now_ns
    ):
        raise PlanningBlocked("freeze request timestamp is invalid")
    ttl_ms = _finite(request.get("ttl_ms"), "freeze request ttl_ms")
    if ttl_ms <= 0.0 or ttl_ms > MAX_M0_TTL_MS:
        raise PlanningBlocked("freeze request TTL is outside the allowed bound")
    if expiry_ns <= now_ns or expiry_ns > stamp_ns + int(ttl_ms * 1_000_000):
        raise PlanningBlocked("freeze request is stale or TTL-inconsistent")
    if request.get("calibration_release_id") != expected_calibration_release_id:
        raise PlanningBlocked("freeze request calibration release does not match the static catalog")
    if request.get("depth_status") != "VALID_DEPTH":
        raise PlanningBlocked("freeze request depth is not valid")
    if request.get("normal_status") != "VALID_OUTWARD":
        raise PlanningBlocked("freeze request normal is not valid outward")
    if request.get("ready_status") != "READY":
        raise PlanningBlocked("freeze request is not READY")
    if _finite(request.get("confidence"), "freeze request confidence") < 0.8:
        raise PlanningBlocked("freeze request confidence is below threshold")
    if _finite(request.get("rgb_depth_skew_ms"), "freeze request RGB/depth skew") > 50.0:
        raise PlanningBlocked("freeze request RGB/depth skew exceeds threshold")
    _vec(request.get("point_camera_m"), 3, "freeze request point_camera_m")
    _unit(request.get("normal_camera"), "freeze request normal_camera")
    outward = _mapping(request.get("normal_outward_evidence"), "normal_outward_evidence")
    if set(outward) != {
        "input_normal_was_flipped",
        "dot_n_out_p_minus_a",
        "camera_side_dot",
    }:
        raise PlanningBlocked("normal outward evidence has unknown or missing fields")
    if (
        outward.get("input_normal_was_flipped") is not False
        or _finite(outward.get("dot_n_out_p_minus_a"), "dot_n_out_p_minus_a") <= 0.0
        or _finite(outward.get("camera_side_dot"), "camera_side_dot") <= 0.0
    ):
        raise PlanningBlocked("normal outward evidence is ambiguous, flipped, or inward")

    _sha256(request.get("source_canonical_sha256"), "source_canonical_sha256")
    return request


@dataclass(frozen=True)
class PlannedPrecontactArtifacts:
    freeze_request_sha256: str
    source_canonical_sha256: str
    m0_frozen_release: Mapping[str, Any]
    outbound_release_manifest: Mapping[str, Any]
    outbound_route_artifact: Mapping[str, Any]
    execution_release: Mapping[str, Any]
    execution_release_manifest_sha256: str
    canonical_frozen_target_json: str
    frozen_target_sha256: str


@dataclass(frozen=True)
class M0FreezeIdentity:
    target_id: str
    detection_id: str
    semantic_track_id: str
    source_stamp_ns: int
    source_frame_id: str


@dataclass(frozen=True)
class FrozenM0CacheReceipt:
    """Opaque one-use handle returned to the trusted adapter observation path."""

    token: str
    request_sha256: str
    identity: M0FreezeIdentity
    expires_monotonic: float


@dataclass(frozen=True)
class _FrozenM0CacheEntry:
    canonical_request_bytes: bytes
    request_sha256: str
    identity: M0FreezeIdentity
    received_monotonic: float
    expires_monotonic: float


class FrozenM0CacheProvider(Protocol):
    def consume_exact(
        self,
        token: str,
        request_sha256: str,
        identity: M0FreezeIdentity,
        *,
        now_monotonic: float,
    ) -> _FrozenM0CacheEntry: ...


def _identity_from_request(request: Mapping[str, Any]) -> M0FreezeIdentity:
    stamp = request.get("source_stamp_ns")
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
        raise PlanningBlocked("freeze request source_stamp_ns is invalid")
    fields = {
        field: str(request.get(field) or "").strip()
        for field in ("target_id", "detection_id", "semantic_track_id", "source_frame_id")
    }
    if any(not value for value in fields.values()):
        raise PlanningBlocked("freeze request cache identity is incomplete")
    return M0FreezeIdentity(
        target_id=fields["target_id"],
        detection_id=fields["detection_id"],
        semantic_track_id=fields["semantic_track_id"],
        source_stamp_ns=stamp,
        source_frame_id=fields["source_frame_id"],
    )


class FrozenM0Cache:
    """Adapter-owned, thread-safe, opaque-token cache for canonical M0 bytes.

    ``observe`` belongs only on the trusted observation callback path.  No
    prepare/action/client API may accept or forward those bytes.  Every entry is
    short-lived and is removed before any digest, identity, or TTL verdict, so a
    failed attempt cannot be replayed with corrected client fields.
    """

    def __init__(self, *, ttl_s: float = MAX_CACHE_TTL_S) -> None:
        ttl = _finite(ttl_s, "cache ttl_s")
        if ttl <= 0.0 or ttl > MAX_CACHE_TTL_S:
            raise PlanningBlocked(f"cache ttl_s must be in (0, {MAX_CACHE_TTL_S:g}]")
        self._ttl_s = ttl
        self._entries: dict[str, _FrozenM0CacheEntry] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        canonical_request_bytes: bytes,
        *,
        receive_monotonic: float,
    ) -> FrozenM0CacheReceipt:
        received = _finite(receive_monotonic, "receive_monotonic")
        request = _canonical_mapping_from_bytes(
            canonical_request_bytes,
            "adapter observed canonical M0 freeze request",
        )
        if request.get("schema") != FREEZE_REQUEST_SCHEMA:
            raise PlanningBlocked("adapter observation has an unexpected freeze request schema")
        identity = _identity_from_request(request)
        request_hash = _sha256_bytes(canonical_request_bytes)
        token = secrets.token_urlsafe(32)
        expires = received + self._ttl_s
        entry = _FrozenM0CacheEntry(
            canonical_request_bytes=bytes(canonical_request_bytes),
            request_sha256=request_hash,
            identity=identity,
            received_monotonic=received,
            expires_monotonic=expires,
        )
        with self._lock:
            while token in self._entries:
                token = secrets.token_urlsafe(32)
            self._entries[token] = entry
        return FrozenM0CacheReceipt(token, request_hash, identity, expires)

    def consume_exact(
        self,
        token: str,
        request_sha256: str,
        identity: M0FreezeIdentity,
        *,
        now_monotonic: float,
    ) -> _FrozenM0CacheEntry:
        if not isinstance(token, str) or len(token) < 32:
            raise PlanningBlocked("freeze cache token is invalid")
        request_hash = _sha256(request_sha256, "freeze request SHA-256")
        now = _finite(now_monotonic, "now_monotonic")
        if not isinstance(identity, M0FreezeIdentity):
            raise PlanningBlocked("freeze request identity must be M0FreezeIdentity")
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None:
            raise PlanningBlocked("freeze cache token is unknown, consumed, or replayed")
        if now < entry.received_monotonic or now >= entry.expires_monotonic:
            raise PlanningBlocked("freeze cache token expired")
        actual_hash = _sha256_bytes(entry.canonical_request_bytes)
        if (
            not secrets.compare_digest(entry.request_sha256, request_hash)
            or not secrets.compare_digest(actual_hash, request_hash)
        ):
            raise PlanningBlocked("cached canonical bytes do not match request SHA-256")
        if identity != entry.identity:
            raise PlanningBlocked("freeze cache identity mismatch or token swap")
        return entry

    def discard(self, token: str) -> bool:
        """Consume an entry without revealing bytes, including failure cleanup."""

        if not isinstance(token, str) or not token:
            return False
        with self._lock:
            return self._entries.pop(token, None) is not None


class PrecontactPlanner(Protocol):
    implemented: bool

    def plan_cached_freeze(
        self,
        cache_token: str,
        request_sha256: str,
        identity: M0FreezeIdentity,
        *,
        now_ns: int,
        now_monotonic: float,
    ) -> PlannedPrecontactArtifacts: ...


class NotImplementedPlanner:
    """Production-safe default: no route/release can be generated."""

    implemented = False

    def plan_cached_freeze(
        self,
        cache_token: str,
        request_sha256: str,
        identity: M0FreezeIdentity,
        *,
        now_ns: int,
        now_monotonic: float,
    ) -> PlannedPrecontactArtifacts:
        raise PlanningBlocked("trusted in-process precontact planner is not implemented")


class TrustedInProcessPlanner:
    """Single-use planner over a static allowlisted release catalog.

    Request digests are consumed before planning, so duplicate and concurrent
    calls cannot produce a second release for the same frozen observation.
    """

    implemented = True

    def __init__(
        self,
        static_release_catalog: Mapping[str, Any],
        *,
        m0_cache: FrozenM0CacheProvider,
        ik_solver: IKSolver | None = None,
        kinematics_checker: KinematicsChecker | None = None,
        collision_checker: CollisionChecker | None = None,
    ) -> None:
        self._catalog = _validate_static_catalog(static_release_catalog)
        self._m0_cache = m0_cache
        self._ik_solver = ik_solver
        self._kinematics_checker = kinematics_checker
        self._collision_checker = collision_checker
        self._consumed_request_hashes: set[str] = set()
        self._lock = threading.Lock()

    def plan_cached_freeze(
        self,
        cache_token: str,
        request_sha256: str,
        identity: M0FreezeIdentity,
        *,
        now_ns: int,
        now_monotonic: float,
    ) -> PlannedPrecontactArtifacts:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0:
            raise PlanningBlocked("now_ns must be a positive integer")
        frozen_monotonic = _finite(now_monotonic, "now_monotonic")
        releases = _mapping(self._catalog["static_releases"], "static_releases")
        r6_release = _mapping(releases["r6_bundle"], "r6 release")
        request_hash = _sha256(request_sha256, "freeze request SHA-256")
        with self._lock:
            if request_hash in self._consumed_request_hashes:
                raise PlanningBlocked("freeze request replay rejected")
            self._consumed_request_hashes.add(request_hash)
        cached = self._m0_cache.consume_exact(
            cache_token,
            request_hash,
            identity,
            now_monotonic=frozen_monotonic,
        )
        request = _validate_freeze_request(
            cached.canonical_request_bytes,
            request_hash,
            now_ns=now_ns,
            expected_calibration_release_id=str(r6_release["release_id"]),
        )

        r6_payload = _mapping(r6_release["payload"], "r6 payload")
        normal_out_base = _rotate_camera_to_base(
            _vec(r6_payload.get("rotation_xyzw"), 4, "r6.rotation_xyzw"),
            _unit(request["normal_camera"], "freeze request normal_camera"),
        )
        outward = _mapping(request["normal_outward_evidence"], "normal_outward_evidence")
        m0_payload = {
            "schema": "m0_frozen_target_release_v1",
            "complete": True,
            "immutable": True,
            "frame_id": request["source_frame_id"],
            "source_stamp_ns": request["source_stamp_ns"],
            "expires_at_ns": request["expires_at_ns"],
            "ttl_ms": request["ttl_ms"],
            "confidence": request["confidence"],
            "rgb_depth_skew_ms": request["rgb_depth_skew_ms"],
            "target_id": request["target_id"],
            "detection_id": request["detection_id"],
            "semantic_track_id": request["semantic_track_id"],
            "calibration_id": request["calibration_release_id"],
            "point_camera_m": request["point_camera_m"],
            "normal_camera": request["normal_camera"],
            "freeze_request_sha256": request_hash,
            "source_canonical_sha256": request["source_canonical_sha256"],
            "depth_status": request["depth_status"],
            "normal_status": request["normal_status"],
            "ready_status": request["ready_status"],
        }
        m0_release = seal_release(
            f"m0-freeze-{request_hash[:16]}",
            "released",
            m0_payload,
        )
        normal_payload = {
            "schema": "normal_decision_release_v1",
            "target_id": request["target_id"],
            "source_stamp_ns": request["source_stamp_ns"],
            "status": "outward_verified",
            "ambiguous": False,
            "input_normal_was_flipped": False,
            "dot_n_out_p_minus_a": outward["dot_n_out_p_minus_a"],
            "camera_side_dot": outward["camera_side_dot"],
            "normal_out_base": list(normal_out_base),
            "freeze_request_sha256": request_hash,
        }
        normal_release = seal_release(
            f"normal-decision-{request_hash[:16]}",
            "released",
            normal_payload,
        )
        outbound_releases = {name: releases[name] for name in _STATIC_RELEASE_NAMES}
        outbound_releases["m0_frozen"] = m0_release
        outbound_releases["normal_decision"] = normal_release
        outbound_manifest = seal_manifest(
            {
                "schema": MANIFEST_SCHEMA,
                "manifest_id": f"dynamic-precontact-{request_hash[:16]}",
                "releases": outbound_releases,
                "route_policy": self._catalog["route_policy"],
                "static_catalog_id": self._catalog["catalog_id"],
                "static_catalog_sha256": self._catalog["catalog_sha256"],
                "freeze_request_sha256": request_hash,
                "source_canonical_sha256": request["source_canonical_sha256"],
            }
        )
        try:
            route = build_outbound_route(
                outbound_manifest,
                ik_solver=self._ik_solver,
                kinematics_checker=self._kinematics_checker,
                collision_checker=self._collision_checker,
                now_ns=now_ns,
            )
        except RouteBlocked as exc:
            raise PlanningBlocked(f"outbound route blocked: {exc}") from exc

        route_artifact_sha256 = execution_canonical_sha256(route)
        execution_release = {
            "schema": RELEASE_SCHEMA,
            "release_id": f"dynamic-precontact-execution-{route['route_hash'][:16]}",
            "execution_enabled": True,
            "execution_ready": True,
            "command_mode": "T101_SINGLE_JOINT_ONLY",
            "outbound_release_manifest_sha256": outbound_manifest["manifest_sha256"],
            "outbound_route_artifact_sha256": route_artifact_sha256,
            "outbound_route_artifact": route,
        }
        parsed_release = parse_release_manifest(execution_release)
        execution_hash = parsed_release.manifest_hash
        geometry = _mapping(route["geometry"], "route geometry")
        binding = _mapping(route["target_binding"], "route target binding")
        frozen_target = {
            "schema": TARGET_SCHEMA,
            "label": "1",
            "target_id": binding["target_id"],
            "detection_id": binding["detection_id"],
            "semantic_track_id": request["semantic_track_id"],
            "source_stamp_ns": binding["source_stamp_ns"],
            "source_frame_id": binding["source_frame_id"],
            "frozen_monotonic": frozen_monotonic,
            "normal_direction_status": "VALID_OUTWARD",
            "m0_frozen_artifact_sha256": binding["m0_frozen_artifact_sha256"],
            "A_base_m": geometry["A_base_m"],
            "N_out_base": geometry["N_out_base"],
            "P_pre_base_m": geometry["P_pre_base_m"],
            "outbound_release_manifest_sha256": outbound_manifest["manifest_sha256"],
            "execution_release_manifest_sha256": execution_hash,
            "freeze_request_sha256": request_hash,
            "source_canonical_sha256": request["source_canonical_sha256"],
        }
        canonical_target_bytes = canonical_json_bytes(frozen_target)
        canonical_target_json = canonical_target_bytes.decode("utf-8")
        target_hash = _sha256_bytes(canonical_target_bytes)
        parsed_target = parse_frozen_target(canonical_target_json, target_hash)
        if (
            parsed_target.execution_release_manifest_sha256 != execution_hash
            or parsed_target.outbound_release_manifest_sha256
            != parsed_release.outbound_release_manifest_sha256
            or parsed_target.m0_frozen_artifact_sha256
            != parsed_release.route.m0_frozen_artifact_sha256  # type: ignore[union-attr]
        ):
            raise PlanningBlocked("planner output hash chain is internally inconsistent")
        return PlannedPrecontactArtifacts(
            freeze_request_sha256=request_hash,
            source_canonical_sha256=str(request["source_canonical_sha256"]),
            m0_frozen_release=m0_release,
            outbound_release_manifest=outbound_manifest,
            outbound_route_artifact=route,
            execution_release=execution_release,
            execution_release_manifest_sha256=execution_hash,
            canonical_frozen_target_json=canonical_target_json,
            frozen_target_sha256=target_hash,
        )


def validate_planning_result(result: PlannedPrecontactArtifacts) -> None:
    """Revalidate the complete dynamic output before an adapter installs it."""

    if not isinstance(result, PlannedPrecontactArtifacts):
        raise PlanningBlocked("planner returned an unexpected result type")
    request_hash = _sha256(result.freeze_request_sha256, "planning result request hash")
    source_hash = _sha256(result.source_canonical_sha256, "planning result source hash")
    m0_release = _validate_release_envelope(result.m0_frozen_release, "dynamic_m0_frozen")
    m0_payload = _mapping(m0_release["payload"], "dynamic M0 payload")
    if (
        m0_payload.get("freeze_request_sha256") != request_hash
        or m0_payload.get("source_canonical_sha256") != source_hash
    ):
        raise PlanningBlocked("dynamic M0 release does not bind the cached freeze request")
    outbound = _mapping(result.outbound_release_manifest, "outbound release manifest")
    unsigned_outbound = dict(outbound)
    outbound_hash = _sha256(unsigned_outbound.pop("manifest_sha256", None), "outbound manifest hash")
    if canonical_sha256(unsigned_outbound) != outbound_hash:
        raise PlanningBlocked("outbound release manifest hash mismatch")
    outbound_releases = _mapping(outbound.get("releases"), "outbound releases")
    if outbound_releases.get("m0_frozen") != result.m0_frozen_release:
        raise PlanningBlocked("outbound manifest substituted another M0 release")
    if outbound.get("freeze_request_sha256") != request_hash:
        raise PlanningBlocked("outbound manifest does not bind the freeze request")
    if outbound.get("source_canonical_sha256") != source_hash:
        raise PlanningBlocked("outbound manifest does not bind the source hash")

    release = parse_release_manifest(result.execution_release)
    if release.manifest_hash != result.execution_release_manifest_sha256:
        raise PlanningBlocked("execution release manifest hash mismatch")
    if release.route is None:
        raise PlanningBlocked("planning result execution release has no route")
    if result.execution_release.get("outbound_route_artifact") != result.outbound_route_artifact:
        raise PlanningBlocked("execution release substituted another route artifact")
    if release.outbound_release_manifest_sha256 != outbound_hash:
        raise PlanningBlocked("execution release does not bind the outbound manifest")
    target = parse_frozen_target(
        result.canonical_frozen_target_json,
        result.frozen_target_sha256,
    )
    if (
        target.execution_release_manifest_sha256 != release.manifest_hash
        or target.outbound_release_manifest_sha256 != outbound_hash
        or target.m0_frozen_artifact_sha256 != m0_release["artifact_sha256"]
        or release.route.m0_frozen_artifact_sha256 != m0_release["artifact_sha256"]
    ):
        raise PlanningBlocked("planning result target/release/route hash chain mismatch")


@dataclass(frozen=True)
class DynamicPrepareOutcome:
    accepted: bool
    reason: str
    core: PrecontactExecutionCore
    prepare_result: PrepareResult | None = None
    planning_result: PlannedPrecontactArtifacts | None = None

    @property
    def target_snapshot_sha256(self) -> str:
        return "" if self.planning_result is None else self.planning_result.frozen_target_sha256


@dataclass(frozen=True)
class _PrivateFrozenM0Slot:
    canonical_request_bytes: bytes
    receipt: FrozenM0CacheReceipt


class DynamicPrepareTransaction:
    """Pure atomic bridge from private M0 cache to immutable execution core.

    It owns the only cache token.  Prepare callers may provide canonical bytes,
    their hash, UI request identity, and the observation identity, but never a
    token.  The private slot is removed before validation/planning and its cache
    entry is discarded in ``finally`` if the planner did not consume it.
    """

    def __init__(
        self,
        *,
        m0_cache: FrozenM0Cache,
        planner: PrecontactPlanner,
    ) -> None:
        self._cache = m0_cache
        self._planner = planner
        self._slot: _PrivateFrozenM0Slot | None = None
        self._seen_ui_request_ids: set[str] = set()
        self._lock = threading.Lock()

    def observe(
        self,
        canonical_request_bytes: bytes,
        *,
        receive_monotonic: float,
    ) -> FrozenM0CacheReceipt:
        receipt = self._cache.observe(
            canonical_request_bytes,
            receive_monotonic=receive_monotonic,
        )
        with self._lock:
            previous = self._slot
            self._slot = _PrivateFrozenM0Slot(bytes(canonical_request_bytes), receipt)
        if previous is not None:
            self._cache.discard(previous.receipt.token)
        return receipt

    def clear(self) -> None:
        """Invalidate the private slot when a newer callback is unusable."""

        with self._lock:
            previous = self._slot
            self._slot = None
        if previous is not None:
            self._cache.discard(previous.receipt.token)

    def _inert(self) -> PrecontactExecutionCore:
        return PrecontactExecutionCore(
            DEFAULT_RELEASE_MANIFEST,
            executor=NotImplementedExecutor(),
        )

    def prepare(
        self,
        canonical_client_freeze_json: str,
        client_freeze_sha256: str,
        ui_request_id: str,
        gates: AdmissionGates,
        *,
        now_ns: int,
        now_monotonic: float,
    ) -> DynamicPrepareOutcome:
        inert = self._inert()
        request_id = str(ui_request_id or "").strip()
        with self._lock:
            slot = self._slot
            self._slot = None
            replay = not request_id or request_id in self._seen_ui_request_ids
            if request_id:
                self._seen_ui_request_ids.add(request_id)
        if slot is None:
            return DynamicPrepareOutcome(False, "no private cached M0 observation", inert)
        try:
            if replay:
                raise PlanningBlocked("ui_request_id is missing or replayed")
            if not isinstance(canonical_client_freeze_json, str) or not canonical_client_freeze_json:
                raise PlanningBlocked("client freeze JSON is required")
            client_bytes = canonical_client_freeze_json.encode("utf-8")
            client_payload = _canonical_mapping_from_bytes(client_bytes, "client freeze request")
            client_hash = _sha256(client_freeze_sha256, "client freeze SHA-256")
            if (
                not secrets.compare_digest(_sha256_bytes(client_bytes), client_hash)
                or not secrets.compare_digest(client_hash, slot.receipt.request_sha256)
                or not secrets.compare_digest(client_bytes, slot.canonical_request_bytes)
            ):
                raise PlanningBlocked("client freeze bytes/hash do not match the private M0 cache")
            client_identity = _identity_from_request(client_payload)
            if client_identity != slot.receipt.identity:
                raise PlanningBlocked("client freeze identity does not match the private M0 cache")
            planned = self._planner.plan_cached_freeze(
                slot.receipt.token,
                slot.receipt.request_sha256,
                slot.receipt.identity,
                now_ns=now_ns,
                now_monotonic=now_monotonic,
            )
            validate_planning_result(planned)
            candidate = PrecontactExecutionCore(
                planned.execution_release,
                executor=NotImplementedExecutor(),
            )
            candidate_gates = replace(
                gates,
                execution_ready=candidate.release.execution_ready,
                route_outbound_only=candidate.release.route is not None,
                route_t101_only=(
                    candidate.release.command_mode == "T101_SINGLE_JOINT_ONLY"
                ),
            )
            prepared = candidate.prepare(
                planned.canonical_frozen_target_json,
                planned.frozen_target_sha256,
                candidate_gates,
                now_monotonic=now_monotonic,
            )
            if not prepared.accepted:
                return DynamicPrepareOutcome(
                    False,
                    prepared.reason,
                    inert,
                    prepare_result=prepared,
                    planning_result=planned,
                )
            return DynamicPrepareOutcome(
                True,
                prepared.reason,
                candidate,
                prepare_result=prepared,
                planning_result=planned,
            )
        except Exception as exc:  # noqa: BLE001 - injected planners fail closed
            # This broad boundary is intentional: an injected planner is not
            # allowed to leak partial state into the node's immutable core.
            return DynamicPrepareOutcome(False, f"dynamic prepare blocked: {exc}", inert)
        finally:
            self._cache.discard(slot.receipt.token)


__all__ = [
    "DynamicPrepareOutcome",
    "DynamicPrepareTransaction",
    "FREEZE_REQUEST_SCHEMA",
    "FrozenM0Cache",
    "FrozenM0CacheProvider",
    "FrozenM0CacheReceipt",
    "MAX_M0_TTL_MS",
    "MAX_CACHE_TTL_S",
    "M0FreezeIdentity",
    "NotImplementedPlanner",
    "PlannedPrecontactArtifacts",
    "PlanningBlocked",
    "PrecontactPlanner",
    "STATIC_CATALOG_SCHEMA",
    "TrustedInProcessPlanner",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonicalize_button1_m0_freeze_request",
    "seal_static_release_catalog",
    "validate_planning_result",
]
