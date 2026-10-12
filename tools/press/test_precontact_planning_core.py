"""End-to-end tests for the trusted, no-motion dynamic planning core."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from press.precontact_outbound_route import seal_release
from press.precontact_planning_core import (
    DynamicPrepareTransaction,
    FREEZE_REQUEST_SCHEMA,
    FrozenM0Cache,
    M0FreezeIdentity,
    NotImplementedPlanner,
    PlanningBlocked,
    STATIC_CATALOG_SCHEMA,
    TrustedInProcessPlanner,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize_button1_m0_freeze_request,
    seal_static_release_catalog,
)
from press.test_precontact_outbound_route import (
    ExactCollision,
    ExactIK,
    ExactKinematics,
    NOW_NS,
    _released_manifest,
)
from roarm_precontact_adapter.precontact_execution_core import (
    AdmissionGates,
    PrecontactExecutionCore,
    REQUIRED_CONFIRMATION_TEXT,
    canonical_sha256 as execution_canonical_sha256,
    parse_frozen_target,
    parse_release_manifest,
)


NOW_MONOTONIC = 10.0


def _static_catalog():
    manifest = _released_manifest()
    static_releases = {
        name: release
        for name, release in manifest["releases"].items()
        if name not in {"m0_frozen", "normal_decision"}
    }
    allowlist = {
        name: [
            {
                "release_id": release["release_id"],
                "artifact_sha256": release["artifact_sha256"],
            }
        ]
        for name, release in static_releases.items()
    }
    return seal_static_release_catalog(
        {
            "schema": STATIC_CATALOG_SCHEMA,
            "catalog_id": "synthetic-static-precontact-v1",
            "status": "released",
            "static_releases": static_releases,
            "allowlisted_release_bindings": allowlist,
            "route_policy": manifest["route_policy"],
        }
    )


def _cached_source_bytes():
    return canonical_json_bytes(
        {
            "schema": "synthetic_cached_m0_source_v1",
            "capture_id": "capture-1",
            "immutable": True,
        }
    )


def _freeze_request(**changes):
    selected_source = _cached_source_bytes()
    request = {
        "schema": FREEZE_REQUEST_SCHEMA,
        "label": "1",
        "target_id": "button-1",
        "detection_id": "det-1",
        "semantic_track_id": "semantic-button1-track-7",
        "source_stamp_ns": NOW_NS - 100_000_000,
        "source_frame_id": "camera_optical_frame",
        "expires_at_ns": NOW_NS + 500_000_000,
        "ttl_ms": 1000,
        "calibration_release_id": "r6-synthetic-release",
        "point_camera_m": [0.0, 0.0, 0.500],
        "normal_camera": [0.0, 0.0, 1.0],
        "confidence": 0.95,
        "rgb_depth_skew_ms": 2.0,
        "depth_status": "VALID_DEPTH",
        "normal_status": "VALID_OUTWARD",
        "ready_status": "READY",
        "normal_outward_evidence": {
            "input_normal_was_flipped": False,
            "dot_n_out_p_minus_a": 0.10,
            "camera_side_dot": 0.10,
        },
        "source_canonical_sha256": canonical_sha256_bytes(selected_source),
    }
    request.update(changes)
    request_bytes = canonical_json_bytes(request)
    return request_bytes, canonical_sha256(request)


def canonical_sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _planner(catalog=None, *, cache=None, collision=True):
    ik = ExactIK()
    return TrustedInProcessPlanner(
        _static_catalog() if catalog is None else catalog,
        m0_cache=FrozenM0Cache() if cache is None else cache,
        ik_solver=ik,
        kinematics_checker=ExactKinematics(ik),
        collision_checker=ExactCollision() if collision else None,
    )


def _all_gates():
    return AdmissionGates(
        **{field: True for field in AdmissionGates.__dataclass_fields__}
    )


def test_fresh_cached_request_builds_one_complete_mutually_bound_release_pair():
    request_bytes, request_hash = _freeze_request()
    cache = FrozenM0Cache()
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    planned = _planner(cache=cache).plan_cached_freeze(
        receipt.token,
        receipt.request_sha256,
        receipt.identity,
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )

    release = parse_release_manifest(planned.execution_release)
    target = parse_frozen_target(
        planned.canonical_frozen_target_json,
        planned.frozen_target_sha256,
    )
    route = planned.outbound_route_artifact
    m0 = planned.m0_frozen_release
    assert release.manifest_hash == planned.execution_release_manifest_sha256
    assert execution_canonical_sha256(planned.execution_release) == release.manifest_hash
    assert target.execution_release_manifest_sha256 == release.manifest_hash
    assert target.outbound_release_manifest_sha256 == route["release_bindings"]["manifest_sha256"]
    assert target.m0_frozen_artifact_sha256 == m0["artifact_sha256"]
    assert route["target_binding"]["m0_frozen_artifact_sha256"] == m0["artifact_sha256"]
    assert m0["payload"]["freeze_request_sha256"] == request_hash
    assert m0["payload"]["source_canonical_sha256"] == canonical_sha256_bytes(
        _cached_source_bytes()
    )
    assert planned.execution_release["outbound_route_artifact_sha256"] == execution_canonical_sha256(route)
    assert route["phases"] == ["OUTBOUND_TO_PRECONTACT"]
    assert route["motion_authorized"] is False
    assert target.p_pre_base_m[2] - target.a_base_m[2] == pytest.approx(0.050)

    core = PrecontactExecutionCore(planned.execution_release)
    prepared = core.prepare(
        planned.canonical_frozen_target_json,
        planned.frozen_target_sha256,
        _all_gates(),
        now_monotonic=NOW_MONOTONIC + 0.1,
    )
    assert prepared.accepted is True


def test_pure_dynamic_prepare_transaction_atomically_installs_only_the_planned_core():
    request_bytes, request_hash = _freeze_request()
    cache = FrozenM0Cache()
    transaction = DynamicPrepareTransaction(
        m0_cache=cache,
        planner=_planner(cache=cache),
    )
    receipt = transaction.observe(
        request_bytes,
        receive_monotonic=NOW_MONOTONIC - 0.1,
    )
    outcome = transaction.prepare(
        request_bytes.decode("utf-8"),
        request_hash,
        "ui-prepare-1",
        _all_gates(),
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    assert outcome.accepted is True
    assert outcome.core.state.value == "PREPARED_NO_MOTION"
    assert outcome.core.release.manifest_hash == (
        outcome.planning_result.execution_release_manifest_sha256
    )
    assert outcome.target_snapshot_sha256 == outcome.planning_result.frozen_target_sha256
    assert outcome.target_snapshot_sha256 != request_hash
    rejected, reason = outcome.core.reserve_goal(
        outcome.prepare_result.authorization_id,
        request_hash,
        REQUIRED_CONFIRMATION_TEXT,
        now_monotonic=NOW_MONOTONIC + 0.01,
    )
    assert rejected is False
    assert "target snapshot hash mismatch" in reason
    accepted, reason = outcome.core.reserve_goal(
        outcome.prepare_result.authorization_id,
        outcome.target_snapshot_sha256,
        REQUIRED_CONFIRMATION_TEXT,
        now_monotonic=NOW_MONOTONIC + 0.01,
    )
    assert accepted is True, reason
    with pytest.raises(PlanningBlocked, match="replay"):
        cache.consume_exact(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_monotonic=NOW_MONOTONIC,
        )


def test_default_dynamic_transaction_consumes_token_and_returns_inert_core():
    request_bytes, request_hash = _freeze_request()
    cache = FrozenM0Cache()
    transaction = DynamicPrepareTransaction(
        m0_cache=cache,
        planner=NotImplementedPlanner(),
    )
    receipt = transaction.observe(
        request_bytes,
        receive_monotonic=NOW_MONOTONIC - 0.1,
    )
    outcome = transaction.prepare(
        request_bytes.decode("utf-8"),
        request_hash,
        "ui-prepare-inert",
        _all_gates(),
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    assert outcome.accepted is False
    assert "not implemented" in outcome.reason
    assert outcome.core.state.value == "INERT"
    assert outcome.core.release.execution_ready is False
    with pytest.raises(PlanningBlocked, match="replay"):
        cache.consume_exact(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_monotonic=NOW_MONOTONIC,
        )


def test_client_mismatch_cannot_reach_planner_and_clears_to_inert():
    request_bytes, request_hash = _freeze_request()
    cache = FrozenM0Cache()
    transaction = DynamicPrepareTransaction(
        m0_cache=cache,
        planner=_planner(cache=cache),
    )
    transaction.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    forged = request_bytes.decode("utf-8").replace("button-1", "button-9")
    outcome = transaction.prepare(
        forged,
        request_hash,
        "ui-prepare-forged",
        _all_gates(),
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    assert outcome.accepted is False
    assert "do not match" in outcome.reason
    assert outcome.core.state.value == "INERT"
    second = transaction.prepare(
        request_bytes.decode("utf-8"),
        request_hash,
        "ui-prepare-after-failure",
        _all_gates(),
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    assert second.accepted is False
    assert "no private cached" in second.reason


def test_tampered_planning_result_is_revalidated_and_never_installed():
    request_bytes, request_hash = _freeze_request()
    cache = FrozenM0Cache()
    trusted = _planner(cache=cache)

    class TamperedPlanner:
        implemented = True

        def plan_cached_freeze(self, *args, **kwargs):
            planned = trusted.plan_cached_freeze(*args, **kwargs)
            planned.execution_release["release_id"] = "tampered-after-planning"
            return planned

    transaction = DynamicPrepareTransaction(
        m0_cache=cache,
        planner=TamperedPlanner(),
    )
    transaction.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    outcome = transaction.prepare(
        request_bytes.decode("utf-8"),
        request_hash,
        "ui-prepare-tampered-result",
        _all_gates(),
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    assert outcome.accepted is False
    assert "execution release manifest hash mismatch" in outcome.reason
    assert outcome.core.state.value == "INERT"
    assert outcome.core.release.route is None


def test_current_m0_canonicalizer_never_invents_outward_normal_authority():
    request = canonicalize_button1_m0_freeze_request(
        {
            "source_stamp_ns": NOW_NS - 100_000_000,
            "source_frame_id": "camera_optical_frame",
            "target_label": "1",
            "effective_label": "1",
            "detection_id": "det-1",
            "semantic_track_id": "track-1",
            "status": "READY",
            "perception_ready": True,
            "motion_authorized": False,
            "point_camera_m": [0.0, 0.0, 0.5],
            "normal_camera": [0.0, 0.0, 1.0],
            "confidence": 0.95,
            "rgb_depth_skew_ms": 2.0,
            "depth_valid": True,
            "surface_estimate_valid": True,
            "surface_provenance_complete": True,
            "calibration_release_id": "diagnostic-section-b-only",
            "normal_direction_status": "UNVERIFIED",
            "input_normal_was_flipped": False,
            "dot_n_out_p_minus_a": 0.0,
            "camera_side_dot": 0.0,
        }
    )
    payload = __import__("json").loads(request)
    assert payload["ready_status"] == "READY"
    assert payload["depth_status"] == "VALID_DEPTH"
    assert payload["normal_status"] == "UNVERIFIED"
    assert payload["normal_outward_evidence"]["dot_n_out_p_minus_a"] == 0.0


def test_default_not_implemented_planner_is_fail_closed():
    request_bytes, _ = _freeze_request()
    receipt = FrozenM0Cache().observe(
        request_bytes,
        receive_monotonic=NOW_MONOTONIC - 0.1,
    )
    with pytest.raises(PlanningBlocked, match="not implemented"):
        NotImplementedPlanner().plan_cached_freeze(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_stale_cached_request_is_rejected_before_route_building():
    request_bytes, _ = _freeze_request(expires_at_ns=NOW_NS)
    cache = FrozenM0Cache()
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    with pytest.raises(PlanningBlocked, match="stale"):
        _planner(cache=cache).plan_cached_freeze(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_client_hash_must_match_the_adapter_cached_bytes_exactly():
    request_bytes, _ = _freeze_request()
    cache = FrozenM0Cache()
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    with pytest.raises(PlanningBlocked, match="do(?:es)? not match"):
        _planner(cache=cache).plan_cached_freeze(
            receipt.token,
            "f" * 64,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_forged_cached_bytes_cannot_hide_behind_the_claimed_same_hash():
    request_bytes, request_hash = _freeze_request()
    forged = request_bytes.replace(b'"button-1"', b'"button-9"')

    class ForgedCache:
        def consume_exact(self, token, supplied_hash, identity, *, now_monotonic):
            return SimpleNamespace(
                canonical_request_bytes=forged,
                request_sha256=request_hash,
                identity=identity,
            )

    identity = M0FreezeIdentity(
        "button-1",
        "det-1",
        "semantic-button1-track-7",
        NOW_NS - 100_000_000,
        "camera_optical_frame",
    )
    with pytest.raises(PlanningBlocked, match="bytes do not match"):
        _planner(cache=ForgedCache()).plan_cached_freeze(
            "opaque-forged-token-that-is-at-least-32-bytes",
            request_hash,
            identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_cache_receive_ttl_expires_and_entry_is_consumed_fail_closed():
    request_bytes, _ = _freeze_request()
    cache = FrozenM0Cache(ttl_s=0.5)
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.5)
    planner = _planner(cache=cache)
    with pytest.raises(PlanningBlocked, match="expired"):
        planner.plan_cached_freeze(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )
    with pytest.raises(PlanningBlocked, match="replay"):
        cache.consume_exact(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_monotonic=NOW_MONOTONIC,
        )


def test_successful_request_hash_cannot_be_replayed():
    request_bytes, _ = _freeze_request()
    cache = FrozenM0Cache()
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    planner = _planner(cache=cache)
    planner.plan_cached_freeze(
        receipt.token,
        receipt.request_sha256,
        receipt.identity,
        now_ns=NOW_NS,
        now_monotonic=NOW_MONOTONIC,
    )
    with pytest.raises(PlanningBlocked, match="replay"):
        planner.plan_cached_freeze(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_token_swap_or_identity_swap_is_rejected_and_consumes_the_entry():
    first_bytes, _ = _freeze_request()
    second_bytes, _ = _freeze_request(
        target_id="button-1-second",
        detection_id="det-2",
        semantic_track_id="semantic-button1-track-8",
        source_stamp_ns=NOW_NS - 50_000_000,
    )
    cache = FrozenM0Cache()
    first = cache.observe(first_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    second = cache.observe(second_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    with pytest.raises(PlanningBlocked, match="do(?:es)? not match"):
        _planner(cache=cache).plan_cached_freeze(
            first.token,
            second.request_sha256,
            second.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )
    with pytest.raises(PlanningBlocked, match="replay"):
        cache.consume_exact(
            first.token,
            first.request_sha256,
            first.identity,
            now_monotonic=NOW_MONOTONIC,
        )

    third = cache.observe(first_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    with pytest.raises(PlanningBlocked, match="identity mismatch"):
        cache.consume_exact(
            third.token,
            third.request_sha256,
            second.identity,
            now_monotonic=NOW_MONOTONIC,
        )


def _replace_static_release(catalog, name, *, payload=None, status=None):
    candidate = copy.deepcopy(catalog)
    previous = candidate["static_releases"][name]
    replacement = seal_release(
        previous["release_id"],
        previous["status"] if status is None else status,
        previous["payload"] if payload is None else payload,
    )
    candidate["static_releases"][name] = replacement
    candidate["allowlisted_release_bindings"][name] = [
        {
            "release_id": replacement["release_id"],
            "artifact_sha256": replacement["artifact_sha256"],
        }
    ]
    return seal_static_release_catalog(candidate)


def test_relative_r6_is_rejected_even_when_hash_bound_and_allowlisted():
    catalog = _static_catalog()
    payload = copy.deepcopy(catalog["static_releases"]["r6_bundle"]["payload"])
    payload["relative"] = True
    catalog = _replace_static_release(catalog, "r6_bundle", payload=payload)
    with pytest.raises(PlanningBlocked, match="relative"):
        _planner(catalog)


def test_diagnostic_tool_is_rejected_even_when_hash_bound_and_allowlisted():
    catalog = _replace_static_release(_static_catalog(), "tool_release", status="diagnostic")
    with pytest.raises(PlanningBlocked, match="diagnostic or unreleased"):
        _planner(catalog)


def test_missing_collision_checker_never_emits_a_release():
    request_bytes, _ = _freeze_request()
    cache = FrozenM0Cache()
    receipt = cache.observe(request_bytes, receive_monotonic=NOW_MONOTONIC - 0.1)
    with pytest.raises(PlanningBlocked, match="no offline collision checker"):
        _planner(cache=cache, collision=False).plan_cached_freeze(
            receipt.token,
            receipt.request_sha256,
            receipt.identity,
            now_ns=NOW_NS,
            now_monotonic=NOW_MONOTONIC,
        )


def test_installed_package_is_the_ssot_and_tools_modules_are_logic_free_wrappers():
    repo_root = Path(__file__).resolve().parents[2]
    installed_route = (
        repo_root
        / "ros2_arm/roarm_precontact_adapter/roarm_precontact_adapter/precontact_outbound_route.py"
    )
    installed_planner = (
        repo_root
        / "ros2_arm/roarm_precontact_adapter/roarm_precontact_adapter/precontact_planning_core.py"
    )
    assert installed_route.is_file()
    assert installed_planner.is_file()
    for wrapper_name, installed_module in (
        ("precontact_outbound_route.py", "roarm_precontact_adapter.precontact_outbound_route"),
        ("precontact_planning_core.py", "roarm_precontact_adapter.precontact_planning_core"),
    ):
        wrapper = Path(__file__).with_name(wrapper_name)
        tree = ast.parse(wrapper.read_text())
        assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
        assert installed_module in wrapper.read_text()
