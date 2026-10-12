import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from press.t102_fast_runtime_envelope import (
    COMMAND_DELTA_CAPS_RAD,
    ENCODER_STEP_RAD,
    EXPECTED_PACKAGE_SHA256,
    HARD_JOINT_LIMITS_RAD,
    JOINT_NAMES,
    STANDARD_STANDBY_RAD,
    EnvelopeBlocked,
    canonical_sha256,
    load_and_validate_package,
    load_json,
    validate_command_transition,
    validate_package_document,
    validate_start_posture,
)
from press.t102_standard_standby_70mm_cache import (
    DEFAULT_V2_PACKAGE,
    benchmark_v2_preflight,
    verify_v2_preflight,
)


ROOT = Path(__file__).resolve().parents[2]
V1_PLAN = (
    ROOT
    / "config/press/prototype/t102_button1_standard_standby_70mm_demo_cache_20260815_v1/plan.json"
)


def package_copy():
    return copy.deepcopy(load_json(DEFAULT_V2_PACKAGE))


def reseal(package):
    package.pop("package_sha256", None)
    package["package_sha256"] = canonical_sha256(package)
    return package


def validate_mutation(mutate):
    package = package_copy()
    mutate(package)
    return validate_package_document(reseal(package), expected_package_sha256=None)


def test_v2_package_reuses_exact_persistent_four_target_route():
    package = load_and_validate_package(DEFAULT_V2_PACKAGE)
    v1_plan = json.loads(V1_PLAN.read_text(encoding="utf-8"))
    assert package["package_sha256"] == EXPECTED_PACKAGE_SHA256
    assert package["joint_names"] == list(JOINT_NAMES)
    assert package["outbound_targets_rad"] == v1_plan["anchors_rad"]
    assert len(package["outbound_targets_rad"]) == 4
    assert package["outbound_targets_rad"][-1] == package["endpoint"]["joint_positions_rad"]
    assert package["endpoint"]["intended_no_contact_distance_m"] == 0.070
    assert package["command_contract"] == {
        "command_interface": "T102_MULTI_JOINT_ABSOLUTE",
        "contact_included": False,
        "direction": "OUTBOUND_ONLY",
        "press_included": False,
        "return_included": False,
        "terminal_stage": "COARSE_PRECONTACT_STOP",
    }


@pytest.mark.parametrize(
    "bad_value",
    [
        [0.0] * 5,
        [0.0] * 7,
        "0 0 0 0 0 0",
        [True] + [0.0] * 5,
        ["0"] + [0.0] * 5,
        [float("nan")] + [0.0] * 5,
        [float("inf")] + [0.0] * 5,
        [float("-inf")] + [0.0] * 5,
    ],
)
def test_malformed_six_joint_vectors_fail_closed(bad_value):
    package = package_copy()
    package["outbound_targets_rad"][0] = bad_value
    with pytest.raises(EnvelopeBlocked, match="six|numeric|finite|canonical"):
        validate_package_document(reseal(package), expected_package_sha256=None)


def test_json_loader_rejects_duplicate_keys_and_nonfinite_constants(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
    with pytest.raises(EnvelopeBlocked, match="duplicate JSON key"):
        load_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(EnvelopeBlocked, match="non-finite JSON constant"):
        load_json(nonfinite)


def test_joint_name_order_and_start_tolerance_are_exact():
    with pytest.raises(EnvelopeBlocked, match="joint-name/order"):
        validate_mutation(
            lambda package: package["joint_names"].__setitem__(
                slice(0, 2), reversed(package["joint_names"][:2])
            )
        )
    package = load_and_validate_package(DEFAULT_V2_PACKAGE)
    at_limit = list(STANDARD_STANDBY_RAD)
    at_limit[2] += ENCODER_STEP_RAD
    assert validate_start_posture(package, at_limit)["accepted"] is True
    outside = list(at_limit)
    outside[2] += 1e-9
    with pytest.raises(EnvelopeBlocked, match="one-encoder-step tolerance"):
        validate_start_posture(package, outside)


@pytest.mark.parametrize("count", [3, 5])
def test_route_requires_exactly_four_targets(count):
    package = package_copy()
    package["outbound_targets_rad"] = (
        package["outbound_targets_rad"][:count]
        if count == 3
        else package["outbound_targets_rad"] + [package["outbound_targets_rad"][-1]]
    )
    with pytest.raises(EnvelopeBlocked, match="exactly four"):
        validate_package_document(reseal(package), expected_package_sha256=None)


@pytest.mark.parametrize("payload", ["press", "contact", "return", "retract"])
def test_any_added_press_contact_return_or_retract_payload_is_rejected(payload):
    package = package_copy()
    package[payload] = {"enabled": False}
    with pytest.raises(EnvelopeBlocked, match="keys mismatch"):
        validate_package_document(reseal(package), expected_package_sha256=None)


@pytest.mark.parametrize(
    "field", ["press_included", "contact_included", "return_included"]
)
def test_nested_press_contact_or_return_flag_cannot_be_enabled(field):
    package = package_copy()
    package["command_contract"][field] = True
    with pytest.raises(EnvelopeBlocked, match=f"{field} must remain false"):
        validate_package_document(reseal(package), expected_package_sha256=None)


def test_unresealed_mutation_fails_self_hash_and_wrong_stored_start_fails():
    package = package_copy()
    package["endpoint"]["target_fingertip_base_m"][0] += 0.001
    with pytest.raises(EnvelopeBlocked, match="self-hash mismatch"):
        validate_package_document(package, expected_package_sha256=None)
    package = package_copy()
    package["start"]["positions_rad"][0] += ENCODER_STEP_RAD
    with pytest.raises(EnvelopeBlocked, match="reviewed quantized Standard Standby"):
        validate_package_document(reseal(package), expected_package_sha256=None)


def test_wrong_scenario_endpoint_metadata_and_resealed_hash_fail_closed():
    with pytest.raises(EnvelopeBlocked, match="scenario mismatch"):
        validate_mutation(lambda package: package.__setitem__("scenario", "other"))
    with pytest.raises(EnvelopeBlocked, match="button metadata"):
        validate_mutation(lambda package: package["endpoint"].__setitem__("button", 2))
    with pytest.raises(EnvelopeBlocked, match="button metadata"):
        validate_mutation(lambda package: package["endpoint"].__setitem__("button", True))
    resealed = package_copy()
    resealed["endpoint"]["target_fingertip_base_m"][0] += 0.001
    reseal(resealed)
    with pytest.raises(EnvelopeBlocked, match="reviewed v2 package"):
        validate_package_document(resealed)


def test_hard_limits_delta_caps_tiny_steps_and_reversal_fail_closed():
    package = package_copy()
    package["outbound_targets_rad"][0][0] = HARD_JOINT_LIMITS_RAD[0][1] + ENCODER_STEP_RAD
    with pytest.raises(EnvelopeBlocked, match="hard joint limits"):
        validate_package_document(reseal(package), expected_package_sha256=None)

    package = package_copy()
    package["outbound_targets_rad"][0][0] = (
        package["start"]["positions_rad"][0] + COMMAND_DELTA_CAPS_RAD[0] + ENCODER_STEP_RAD
    )
    package["outbound_targets_rad"][0][0] = round(
        package["outbound_targets_rad"][0][0] / ENCODER_STEP_RAD
    ) * ENCODER_STEP_RAD
    with pytest.raises(EnvelopeBlocked, match="delta caps"):
        validate_package_document(reseal(package), expected_package_sha256=None)

    package = package_copy()
    tiny = list(package["start"]["positions_rad"])
    tiny[0] += ENCODER_STEP_RAD
    tiny[1] += ENCODER_STEP_RAD
    package["outbound_targets_rad"][0] = tiny
    with pytest.raises(EnvelopeBlocked, match="tiny-step"):
        validate_package_document(reseal(package), expected_package_sha256=None)

    package = package_copy()
    package["outbound_targets_rad"][1][0] = (
        package["outbound_targets_rad"][0][0] - 8 * ENCODER_STEP_RAD
    )
    with pytest.raises(EnvelopeBlocked, match="materially reverses"):
        validate_package_document(reseal(package), expected_package_sha256=None)


def test_endpoint_distance_must_decrease_and_correction_boundary_is_command_only():
    package = load_and_validate_package(DEFAULT_V2_PACKAGE)
    current = package["start"]["positions_rad"]
    target = package["outbound_targets_rad"][0]
    result = validate_command_transition(package, current, target)
    assert result["accepted"] is True
    assert set(result) == {
        "accepted",
        "deltas_rad",
        "maximum_delta_rad",
        "endpoint_distance_reduction_rad",
    }
    with pytest.raises(EnvelopeBlocked, match="does not reduce endpoint distance"):
        validate_command_transition(
            package,
            package["outbound_targets_rad"][1],
            package["outbound_targets_rad"][0],
            endpoint_rad=package["endpoint"]["joint_positions_rad"],
        )


def test_preflight_dynamic_instrumentation_opens_only_v2_json(monkeypatch):
    opened = []
    original_open = Path.open

    def recording_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("v2 preflight attempted a subprocess")

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(subprocess, "Popen", forbidden_subprocess)
    result = verify_v2_preflight(DEFAULT_V2_PACKAGE, STANDARD_STANDBY_RAD)
    assert result["state"] == "V2_FAST_PREFLIGHT_PASS"
    assert opened == [DEFAULT_V2_PACKAGE.resolve()]
    assert not any(path.suffix.lower() in {".urdf", ".xacro", ".stl"} for path in opened)
    assert not any("clearance" in path.name or "receipt" in path.name for path in opened)


def test_clean_child_import_preflight_has_no_heavy_import_open_or_subprocess():
    code = f"""
import json, pathlib, sys
events = []
order = []
def audit(event, args):
    if event == 'open':
        events.append(('open', str(args[0])))
    elif event == 'subprocess.Popen':
        events.append(('subprocess', str(args[0])))
sys.addaudithook(audit)
sys.path.insert(0, {str((ROOT / 'tools').resolve())!r})
import press.t102_smooth_precontact_demo as runner
original_load = runner.load_and_validate_package
def recording_load(*args, **kwargs):
    order.append('load')
    result = original_load(*args, **kwargs)
    order.append('validated')
    return result
runner.load_and_validate_package = recording_load
class RuntimeReached(Exception):
    pass
def fake_runtime_factory():
    order.append('runtime_factory')
    raise RuntimeReached()
class Args:
    confirmation_text = runner.CONFIRMATION
    package = pathlib.Path({str(DEFAULT_V2_PACKAGE.resolve())!r})
    output = pathlib.Path('/tmp/t102_fast_v2_should_not_write.json')
try:
    runner.execute(Args(), runtime_factory=fake_runtime_factory)
except RuntimeReached:
    pass
else:
    raise AssertionError('runner did not reach fake runtime after preflight')
forbidden_modules = sorted(name for name in sys.modules if name in ('rclpy','numpy','scipy','trimesh') or 'continuous_clearance' in name or 'precontact_static_kinematics' in name or 't102_outbound_route_builder' in name or 't102_three_large_step_coarse' in name)
forbidden_opens = [item for item in events if item[0] == 'open' and (str(item[1]).lower().endswith(('.urdf','.xacro','.stl')) or 'clearance' in str(item[1]).lower() or 'receipt' in str(item[1]).lower())]
print(json.dumps({{'state': 'V2_RUNNER_INITIALIZATION_INSTRUMENTED', 'order': order, 'forbidden_modules': forbidden_modules, 'forbidden_opens': forbidden_opens, 'subprocess_events': [item for item in events if item[0] == 'subprocess']}}))
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "state": "V2_RUNNER_INITIALIZATION_INSTRUMENTED",
        "order": ["load", "validated", "runtime_factory"],
        "forbidden_modules": [],
        "forbidden_opens": [],
        "subprocess_events": [],
    }


def test_deterministic_thousand_call_benchmark_meets_target():
    result = benchmark_v2_preflight(
        DEFAULT_V2_PACKAGE, STANDARD_STANDBY_RAD, iterations=1000
    )
    assert result["iterations"] == 1000
    assert result["p95_ms"] <= 10.0
    assert result["target_met"] is True
